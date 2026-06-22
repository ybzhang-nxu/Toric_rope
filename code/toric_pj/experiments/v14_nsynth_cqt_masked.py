from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from toric_pj.diagnostics.basis_projection import (
    Basis,
    axis_additive_fourier_basis,
    default_device,
    directional_jet_basis,
    normalize_columns,
    toric_fourier_basis,
)
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.experiments.real_digits_probe import relative_2d_table_basis


@dataclass(frozen=True)
class AudioRecord:
    key: str
    path: Path
    pitch: int | None = None
    velocity: int | None = None
    instrument_family: int | None = None
    instrument_source: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V14 NSynth CQT masked spectrogram reconstruction.")
    parser.add_argument("--data-root", type=str, default="data/nsynth")
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v14_nsynth_cqt_masked")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--seeds", type=str, default="426")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--min-samples", type=int, default=16)
    parser.add_argument("--train-count", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--cqt-bins", type=int, default=84)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--fmin-note", type=str, default="C1")
    parser.add_argument("--time-frames", type=int, default=128)
    parser.add_argument("--patch-time", type=int, default=4)
    parser.add_argument("--patch-freq", type=int, default=6)
    parser.add_argument(
        "--basis-list",
        type=str,
        default="no_pos_constant,axis_additive,toric_j0,toric_j1,toric_j2,dct_lowfreq33,relative_2d_table,toric_j2_coord_shuffle",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-rate", type=float, default=0.4)
    parser.add_argument("--mask-mode", type=str, default="random", choices=["random", "time_block", "freq_block", "rect_block"])
    parser.add_argument("--mask-block-time", type=int, default=0)
    parser.add_argument("--mask-block-freq", type=int, default=0)
    parser.add_argument("--export-bias-tables", action="store_true", help="Export learned scalar attention-bias offset tables after each run.")
    parser.add_argument("--fail-if-missing-data", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_nsynth_records(root: Path, *, max_samples: int) -> list[AudioRecord]:
    if not root.exists():
        return []
    metadata_by_dir: list[tuple[Path, dict[str, object]]] = []
    for meta_path in sorted(root.rglob("examples.json")):
        try:
            metadata_by_dir.append((meta_path.parent, json.loads(meta_path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue

    records: list[AudioRecord] = []
    for wav_path in sorted(root.rglob("*.wav")):
        key = wav_path.stem
        meta = find_metadata_for_wav(wav_path, key, metadata_by_dir)
        records.append(
            AudioRecord(
                key=key,
                path=wav_path,
                pitch=optional_int(meta.get("pitch")) if meta else None,
                velocity=optional_int(meta.get("velocity")) if meta else None,
                instrument_family=optional_int(meta.get("instrument_family")) if meta else None,
                instrument_source=optional_int(meta.get("instrument_source")) if meta else None,
            )
        )
        if len(records) >= max_samples:
            break
    return records


def find_metadata_for_wav(wav_path: Path, key: str, metadata_by_dir: list[tuple[Path, dict[str, object]]]) -> dict[str, object] | None:
    best: tuple[int, dict[str, object]] | None = None
    for meta_dir, metadata in metadata_by_dir:
        try:
            wav_path.relative_to(meta_dir)
        except ValueError:
            continue
        item = metadata.get(key)
        if isinstance(item, dict):
            score = len(meta_dir.parts)
            if best is None or score > best[0]:
                best = (score, item)
    if best is not None:
        return best[1]
    for _, metadata in metadata_by_dir:
        item = metadata.get(key)
        if isinstance(item, dict):
            return item
    return None


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cqt_cache_path(record: AudioRecord, args: argparse.Namespace, cache_dir: Path) -> Path:
    stat = record.path.stat()
    payload = json.dumps(
        {
            "path": str(record.path.resolve()),
            "mtime": stat.st_mtime_ns,
            "size": stat.st_size,
            "sample_rate": args.sample_rate,
            "clip_seconds": args.clip_seconds,
            "cqt_bins": args.cqt_bins,
            "bins_per_octave": args.bins_per_octave,
            "hop_length": args.hop_length,
            "fmin_note": args.fmin_note,
            "time_frames": args.time_frames,
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:20]
    return cache_dir / f"{record.key}_{digest}.npy"


def compute_cqt(record: AudioRecord, args: argparse.Namespace, cache_dir: Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cqt_cache_path(record, args, cache_dir)
    if path.exists():
        return np.load(path)
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for NSynth CQT; install librosa and soundfile.") from exc

    target_len = int(round(args.sample_rate * args.clip_seconds))
    y, _ = librosa.load(str(record.path), sr=args.sample_rate, mono=True, duration=args.clip_seconds)
    if y.shape[0] < target_len:
        y = np.pad(y, (0, target_len - y.shape[0]))
    elif y.shape[0] > target_len:
        y = y[:target_len]
    fmin = float(librosa.note_to_hz(args.fmin_note))
    cqt = librosa.cqt(
        y,
        sr=args.sample_rate,
        hop_length=args.hop_length,
        n_bins=args.cqt_bins,
        bins_per_octave=args.bins_per_octave,
        fmin=fmin,
    )
    spec = np.log1p(np.abs(cqt)).T.astype(np.float32)
    spec = fix_time_frames(spec, args.time_frames)
    np.save(path, spec)
    return spec


def fix_time_frames(spec: np.ndarray, time_frames: int) -> np.ndarray:
    if spec.shape[0] < time_frames:
        pad = np.zeros((time_frames - spec.shape[0], spec.shape[1]), dtype=spec.dtype)
        return np.concatenate([spec, pad], axis=0)
    return spec[:time_frames]


def load_cqt_dataset(
    records: list[AudioRecord],
    args: argparse.Namespace,
    *,
    cache_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[AudioRecord], list[AudioRecord], dict[str, object]]:
    specs = [compute_cqt(record, args, cache_dir) for record in records]
    data = torch.tensor(np.stack(specs), dtype=torch.float32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    perm = torch.randperm(data.shape[0], generator=gen, device=device)
    train_count = int(args.train_count) if args.train_count > 0 else max(1, int(round(0.8 * data.shape[0])))
    train_count = min(max(1, train_count), data.shape[0] - 1)
    train_idx = perm[:train_count]
    test_idx = perm[train_count:]
    train = data[train_idx]
    test = data[test_idx]
    mean = train.mean(dim=(0, 1), keepdim=True)
    std = train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    train = (train - mean) / std
    test = (test - mean) / std
    train_records = [records[int(idx.detach().cpu())] for idx in train_idx]
    test_records = [records[int(idx.detach().cpu())] for idx in test_idx]
    stats = {
        "mean_global": float(mean.mean().detach().cpu()),
        "std_global": float(std.mean().detach().cpu()),
        "train_count": int(train.shape[0]),
        "test_count": int(test.shape[0]),
    }
    return train, test, train_records, test_records, stats


def patchify(specs: torch.Tensor, patch_time: int, patch_freq: int) -> tuple[torch.Tensor, int, int]:
    n_samples, n_time, n_freq = specs.shape
    time_tokens = n_time // patch_time
    freq_tokens = n_freq // patch_freq
    cropped = specs[:, : time_tokens * patch_time, : freq_tokens * patch_freq]
    patches = cropped.reshape(n_samples, time_tokens, patch_time, freq_tokens, patch_freq)
    patches = patches.permute(0, 1, 3, 2, 4).reshape(n_samples, time_tokens * freq_tokens, patch_time * patch_freq)
    return patches, time_tokens, freq_tokens


def make_rect_positions(time_tokens: int, freq_tokens: int, device: torch.device) -> torch.Tensor:
    tt = torch.arange(time_tokens, device=device, dtype=torch.float32)
    ff = torch.arange(freq_tokens, device=device, dtype=torch.float32)
    t_grid, f_grid = torch.meshgrid(tt, ff, indexing="ij")
    return torch.stack([t_grid.reshape(-1), f_grid.reshape(-1)], dim=1)


def pairwise_d(positions: torch.Tensor) -> torch.Tensor:
    return positions[:, None, :] - positions[None, :, :]


def no_pos_basis(d: torch.Tensor) -> Basis:
    return Basis("no_pos_constant", torch.ones((d.shape[0], 1), device=d.device, dtype=d.dtype), ["const"], [0])


def nsynth_omegas(device: torch.device, dtype: torch.dtype, time_tokens: int, freq_tokens: int) -> list[torch.Tensor]:
    time_scale = max(float(time_tokens), 1.0)
    freq_scale = max(float(freq_tokens), 1.0)
    return [
        torch.tensor([2.0 * math.pi / time_scale, 2.0 * math.pi / freq_scale], device=device, dtype=dtype),
        torch.tensor([4.0 * math.pi / time_scale, 1.0 * math.pi / freq_scale], device=device, dtype=dtype),
        torch.tensor([1.0 * math.pi / time_scale, 3.0 * math.pi / freq_scale], device=device, dtype=dtype),
        torch.tensor([6.0 * math.pi / time_scale, 2.0 * math.pi / freq_scale], device=device, dtype=dtype),
    ]


def dct_lowfreq_basis(d: torch.Tensor, time_tokens: int, freq_tokens: int, *, num_atoms: int, name: str) -> Basis:
    time_size = 2 * time_tokens - 1
    freq_size = 2 * freq_tokens - 1
    t = d[:, 0] + float(time_tokens - 1)
    f = d[:, 1] + float(freq_tokens - 1)
    pairs: list[tuple[int, int]] = []
    for total in range(time_size + freq_size):
        for kt in range(total + 1):
            kf = total - kt
            if kt == 0 and kf == 0:
                continue
            if kt < time_size and kf < freq_size:
                pairs.append((kt, kf))
            if len(pairs) >= num_atoms:
                break
        if len(pairs) >= num_atoms:
            break
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    for kt, kf in pairs:
        col = torch.cos(math.pi / float(time_size) * (t + 0.5) * kt) * torch.cos(
            math.pi / float(freq_size) * (f + 0.5) * kf
        )
        cols.append(col)
        labels.append(f"dct_kt{kt}_kf{kf}")
    return Basis(name, torch.stack(cols, dim=1), labels, [0] * len(labels))


def build_bases(d: torch.Tensor, time_tokens: int, freq_tokens: int, *, seed: int) -> dict[str, Basis]:
    device = d.device
    dtype = d.dtype
    omegas = nsynth_omegas(device, dtype, time_tokens, freq_tokens)
    ex = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    ey = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)).reshape(-1)
    oblique = normalize_direction(torch.tensor([[1.0, -0.65]], device=device, dtype=dtype)).reshape(-1)
    dirs = [ex, ey, diag, oblique]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    shuffled = d[torch.randperm(d.shape[0], device=device, generator=gen)]
    scale = float(max(time_tokens, freq_tokens))
    bases = {
        "no_pos_constant": no_pos_basis(d),
        "axis_additive": axis_additive_fourier_basis(
            d,
            [2.0 * math.pi / max(float(time_tokens), 1.0), 2.0 * math.pi / max(float(freq_tokens), 1.0)],
            name="axis_additive",
        ),
        "toric_j0": toric_fourier_basis(d, omegas, name="toric_j0"),
        "toric_j1": directional_jet_basis(d, omegas, dirs, [0, 1], scale=scale, name="toric_j1"),
        "toric_j2": directional_jet_basis(d, omegas, dirs, [0, 1, 2], scale=scale, name="toric_j2"),
        "toric_j2_coord_shuffle": directional_jet_basis(
            shuffled,
            omegas,
            dirs,
            [0, 1, 2],
            scale=scale,
            name="toric_j2_coord_shuffle",
        ),
        "dct_lowfreq33": dct_lowfreq_basis(d, time_tokens, freq_tokens, num_atoms=32, name="dct_lowfreq33"),
        "relative_2d_table": relative_2d_table_basis(d),
    }
    return bases


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        bsz, n_positions, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(bsz, n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(float(self.head_dim))
        logits = logits + bias.unsqueeze(0)
        attn = torch.softmax(logits, dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", attn, v).transpose(1, 2).reshape(bsz, n_positions, self.dim)
        x = self.norm1(x + self.out(context))
        x = self.norm2(x + self.ff(x))
        return x


class CQTMaskedTransformer(nn.Module):
    def __init__(self, basis: Basis, *, n_positions: int, patch_dim: int, dim: int, depth: int, n_heads: int) -> None:
        super().__init__()
        matrix, _ = normalize_columns(basis.matrix.to(dtype=torch.float32))
        self.register_buffer("basis_matrix", matrix)
        self.n_positions = n_positions
        self.n_heads = n_heads
        self.coeff = nn.Parameter(torch.zeros(n_heads, matrix.shape[1], dtype=torch.float32, device=matrix.device))
        self.input = nn.Linear(patch_dim + 1, dim)
        self.blocks = nn.ModuleList([AttentionBlock(dim, n_heads) for _ in range(depth)])
        self.reconstruct = nn.Linear(dim, patch_dim)

    def attention_bias(self) -> torch.Tensor:
        logits = torch.einsum("nf,hf->hn", self.basis_matrix, self.coeff)
        return logits.reshape(self.n_heads, self.n_positions, self.n_positions)

    def forward_reconstruction(self, patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        visible = patches.masked_fill(mask.unsqueeze(-1), 0.0)
        indicator = (~mask).to(patches.dtype).unsqueeze(-1)
        x = self.input(torch.cat([visible, indicator], dim=-1))
        bias = self.attention_bias()
        for block in self.blocks:
            x = block(x, bias)
        return self.reconstruct(x)


def sample_batch(x: torch.Tensor, *, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device, generator=generator)
    return x[idx]


def train_and_eval(
    basis: Basis,
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    positions: torch.Tensor,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=train_x.device)
    gen.manual_seed(seed + 991)
    model = CQTMaskedTransformer(
        basis,
        n_positions=train_x.shape[1],
        patch_dim=train_x.shape[2],
        dim=args.dim,
        depth=args.depth,
        n_heads=args.n_heads,
    ).to(train_x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    time_tokens = int(positions[:, 0].max().item()) + 1
    freq_tokens = int(positions[:, 1].max().item()) + 1
    for _ in range(args.steps):
        batch = sample_batch(train_x, batch_size=args.batch_size, generator=gen)
        mask = make_mask(
            batch.shape[0],
            time_tokens,
            freq_tokens,
            device=batch.device,
            generator=gen,
            args=args,
        )
        opt.zero_grad(set_to_none=True)
        pred = model.forward_reconstruction(batch, mask)
        loss = (pred[mask] - batch[mask]).square().mean()
        loss.backward()
        opt.step()
    row = evaluate(model, basis, test_x, positions, args=args, seed=seed + 1991)
    if args.export_bias_tables:
        export_attention_bias_tables(model, basis, positions, output_dir=Path(args.output_dir), seed=seed)
    return row


def export_attention_bias_tables(
    model: CQTMaskedTransformer,
    basis: Basis,
    positions: torch.Tensor,
    *,
    output_dir: Path,
    seed: int,
) -> Path:
    bias = model.attention_bias().detach().to(dtype=torch.float64)
    d_pair = (positions[:, None, :] - positions[None, :, :]).reshape(-1, 2).to(torch.long)
    time_tokens = int(positions[:, 0].max().item()) + 1
    freq_tokens = int(positions[:, 1].max().item()) + 1
    height = 2 * time_tokens - 1
    width = 2 * freq_tokens - 1
    inverse = (d_pair[:, 0] + time_tokens - 1) * width + (d_pair[:, 1] + freq_tokens - 1)
    counts = torch.bincount(inverse, minlength=height * width).to(device=bias.device, dtype=bias.dtype).clamp_min(1.0)
    tables = []
    for head in range(bias.shape[0]):
        values = bias[head].reshape(-1)
        sums = torch.bincount(inverse, weights=values, minlength=height * width)
        tables.append((sums / counts).reshape(height, width))
    table_tensor = torch.stack(tables, dim=0).to(torch.float32).unsqueeze(0)
    axis_tensor = torch.zeros_like(table_tensor)
    residual_tensor = table_tensor.clone()
    dx_values = torch.arange(-(time_tokens - 1), time_tokens, device=positions.device, dtype=torch.float32)
    dy_values = torch.arange(-(freq_tokens - 1), freq_tokens, device=positions.device, dtype=torch.float32)
    export_dir = output_dir / "bias_exports" / f"{basis.name}_seed{seed}"
    export_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "nsynth",
        "representation": "cqt",
        "basis": basis.name,
        "seed": int(seed),
        "n_heads": int(model.n_heads),
        "n_positions": int(model.n_positions),
        "time_tokens": int(time_tokens),
        "freq_tokens": int(freq_tokens),
        "num_features": int(model.basis_matrix.shape[1]),
    }
    path = export_dir / "bias_tables.npz"
    np.savez_compressed(
        path,
        tables=table_tensor.detach().cpu().numpy(),
        axis_tables=axis_tensor.detach().cpu().numpy(),
        residual_tables=residual_tensor.detach().cpu().numpy(),
        dx_values=dx_values.detach().cpu().numpy(),
        dy_values=dy_values.detach().cpu().numpy(),
        metadata=np.array([metadata], dtype=object),
    )
    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def evaluate(
    model: CQTMaskedTransformer,
    basis: Basis,
    test_x: torch.Tensor,
    positions: torch.Tensor,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    gen = torch.Generator(device=test_x.device)
    gen.manual_seed(seed)
    model.eval()
    time_tokens = int(positions[:, 0].max().item()) + 1
    freq_tokens = int(positions[:, 1].max().item()) + 1
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    group_values: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {
        "onset": ([], []),
        "middle": ([], []),
        "decay": ([], []),
    }
    token_t = positions[:, 0].to(device=test_x.device)
    max_t = token_t.max().clamp_min(1.0)
    groups = {
        "onset": token_t <= 0.25 * max_t,
        "middle": (token_t > 0.25 * max_t) & (token_t < 0.75 * max_t),
        "decay": token_t >= 0.75 * max_t,
    }
    with torch.no_grad():
        for start in range(0, test_x.shape[0], args.eval_batch_size):
            batch = test_x[start : start + args.eval_batch_size]
            mask = make_mask(
                batch.shape[0],
                time_tokens,
                freq_tokens,
                device=batch.device,
                generator=gen,
                args=args,
            )
            pred = model.forward_reconstruction(batch, mask)
            preds.append(pred[mask].float())
            targets.append(batch[mask].float())
            for name, group in groups.items():
                group_mask = mask & group.reshape(1, -1)
                if bool(group_mask.any()):
                    group_values[name][0].append(pred[group_mask].float())
                    group_values[name][1].append(batch[group_mask].float())
    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)
    mse, r2 = mse_r2(pred_all, target_all)
    row: dict[str, object] = {
        "basis": basis.name,
        "metric": "masked_patch_r2",
        "score": r2,
        "mse": mse,
        "num_features": int(model.basis_matrix.shape[1]),
        "steps": int(args.steps),
        "mask_rate": float(args.mask_rate),
        "mask_mode": str(args.mask_mode),
    }
    for name, (group_preds, group_targets) in group_values.items():
        if group_preds:
            group_mse, group_r2 = mse_r2(torch.cat(group_preds), torch.cat(group_targets))
        else:
            group_mse, group_r2 = float("nan"), float("nan")
        row[f"{name}_mse"] = group_mse
        row[f"{name}_r2"] = group_r2
    return row


def make_mask(
    batch_size: int,
    time_tokens: int,
    freq_tokens: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.mask_mode == "random":
        return torch.rand((batch_size, time_tokens * freq_tokens), device=device, generator=generator) < args.mask_rate

    mask = torch.zeros((batch_size, time_tokens, freq_tokens), device=device, dtype=torch.bool)
    if args.mask_mode == "time_block":
        block_t = int(args.mask_block_time) if args.mask_block_time > 0 else max(1, round(time_tokens * args.mask_rate))
        block_t = min(block_t, time_tokens)
        for idx in range(batch_size):
            start = int(torch.randint(0, time_tokens - block_t + 1, (1,), device=device, generator=generator).item())
            mask[idx, start : start + block_t, :] = True
    elif args.mask_mode == "freq_block":
        block_f = int(args.mask_block_freq) if args.mask_block_freq > 0 else max(1, round(freq_tokens * args.mask_rate))
        block_f = min(block_f, freq_tokens)
        for idx in range(batch_size):
            start = int(torch.randint(0, freq_tokens - block_f + 1, (1,), device=device, generator=generator).item())
            mask[idx, :, start : start + block_f] = True
    elif args.mask_mode == "rect_block":
        default_side = max(args.mask_rate, 1e-6) ** 0.5
        block_t = int(args.mask_block_time) if args.mask_block_time > 0 else max(1, round(time_tokens * default_side))
        block_f = int(args.mask_block_freq) if args.mask_block_freq > 0 else max(1, round(freq_tokens * default_side))
        block_t = min(block_t, time_tokens)
        block_f = min(block_f, freq_tokens)
        for idx in range(batch_size):
            start_t = int(torch.randint(0, time_tokens - block_t + 1, (1,), device=device, generator=generator).item())
            start_f = int(torch.randint(0, freq_tokens - block_f + 1, (1,), device=device, generator=generator).item())
            mask[idx, start_t : start_t + block_t, start_f : start_f + block_f] = True
    else:
        raise ValueError(f"unknown mask mode: {args.mask_mode}")
    return mask.reshape(batch_size, time_tokens * freq_tokens)


def mse_r2(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    mse = torch.mean((pred - target).square())
    var = torch.mean((target - target.mean()).square()).clamp_min(1e-30)
    return float(mse.detach().cpu()), float((1.0 - mse / var).detach().cpu())


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["basis"]), []).append(row)
    aggregate: list[dict[str, object]] = []
    numeric = ["score", "mse", "onset_r2", "middle_r2", "decay_r2"]
    for basis, items in grouped.items():
        out: dict[str, object] = {"basis": basis, "n": len(items), "num_features": items[0]["num_features"]}
        for name in numeric:
            values = np.array([float(item[name]) for item in items], dtype=np.float64)
            out[f"{name}_mean"] = float(np.nanmean(values))
            out[f"{name}_std"] = float(np.nanstd(values))
        aggregate.append(out)
    aggregate.sort(key=lambda item: float(item["score_mean"]), reverse=True)
    return aggregate


def plot_results(aggregate: list[dict[str, object]], path: Path) -> None:
    if not aggregate:
        return
    labels = [str(row["basis"]) for row in aggregate]
    values = [float(row["score_mean"]) for row in aggregate]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#356f86")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Masked patch R2")
    ax.set_title("NSynth CQT masked reconstruction")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_missing_data(output_dir: Path, args: argparse.Namespace, records: list[AudioRecord], reason: str) -> dict[str, object]:
    status = {
        "status": "missing_data" if not records else "insufficient_data",
        "reason": reason,
        "data_root": str(Path(args.data_root)),
        "num_wav_files_found": len(records),
        "min_samples": int(args.min_samples),
        "expected_layout": [
            "data/nsynth/nsynth-test/audio/*.wav",
            "data/nsynth/nsynth-test/examples.json",
        ],
        "reproduction_command": (
            "PYTHONPATH=MetricToric/code python "
            "MetricToric/code/scripts/run_v14_nsynth_cqt_masked.py "
            "--data-root data/nsynth/nsynth-test --device cuda --max-samples 256 --steps 500"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nsynth_cqt_data_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    lines = [
        "# V14 NSynth CQT Masked Reconstruction",
        "",
        "Status: data unavailable on this machine.",
        "",
        f"Data root checked: `{status['data_root']}`",
        f"WAV files found: {status['num_wav_files_found']}",
        f"Minimum required for smoke: {status['min_samples']}",
        "",
        "Expected NSynth layout:",
        "",
        "- `data/nsynth/nsynth-test/audio/*.wav`",
        "- `data/nsynth/nsynth-test/examples.json`",
        "",
        "Once NSynth is present, run:",
        "",
        "```bash",
        str(status["reproduction_command"]),
        "```",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def write_report(output_dir: Path, summary: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V14 NSynth CQT Masked Reconstruction",
        "",
        f"Status: {summary['status']}",
        f"Dataset root: `{summary['data_root']}`",
        f"Records: {summary['num_records']} ({summary['train_count']} train / {summary['test_count']} test)",
        f"CQT: {summary['time_frames']}x{summary['cqt_bins']}, patch {summary['patch_time']}x{summary['patch_freq']}",
        f"Token grid: {summary['time_tokens']}x{summary['freq_tokens']}",
        f"Masking: {summary['mask_mode']} at rate {summary['mask_rate']}",
        "",
        "## Aggregate",
        "",
        "| basis | features | n | masked R2 | onset R2 | middle R2 | decay R2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + f"{row['basis']} | {int(row['num_features'])} | {int(row['n'])} | "
            + f"{float(row['score_mean']):.4f} +/- {float(row['score_std']):.4f} | "
            + f"{float(row['onset_r2_mean']):.4f} | "
            + f"{float(row['middle_r2_mean']):.4f} | "
            + f"{float(row['decay_r2_mean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This is the first real-data gate for the time-frequency higher-order jet story.",
            "It should be treated as a smoke/pilot unless run with enough NSynth examples,",
            "multiple seeds, and longer training.",
            "",
            "Artifacts:",
            "",
            "- `nsynth_cqt_results.csv`",
            "- `nsynth_cqt_aggregate.csv`",
        "- `nsynth_cqt_summary.json`",
        "- `nsynth_cqt_masked_r2.pdf`",
        "- `bias_exports/*/bias_tables.npz` when `--export-bias-tables` is enabled",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    records = discover_nsynth_records(data_root, max_samples=args.max_samples)
    if len(records) < args.min_samples:
        status = write_missing_data(
            output_dir,
            args,
            records,
            reason=f"found {len(records)} wav files, need at least {args.min_samples}",
        )
        if args.fail_if_missing_data:
            raise SystemExit(2)
        return status

    device = default_device(args.device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cqt_cache"
    train_specs, test_specs, train_records, test_records, stats = load_cqt_dataset(records, args, cache_dir=cache_dir, device=device)
    train_x, time_tokens, freq_tokens = patchify(train_specs, args.patch_time, args.patch_freq)
    test_x, _, _ = patchify(test_specs, args.patch_time, args.patch_freq)
    positions = make_rect_positions(time_tokens, freq_tokens, device)
    d = pairwise_d(positions).reshape(-1, 2)
    bases = build_bases(d, time_tokens, freq_tokens, seed=args.seed)
    selected = parse_str_list(args.basis_list)
    missing_bases = [name for name in selected if name not in bases]
    if missing_bases:
        raise ValueError(f"unknown bases: {missing_bases}; available={sorted(bases)}")

    rows: list[dict[str, object]] = []
    for seed in parse_int_list(args.seeds):
        for name in selected:
            row = train_and_eval(bases[name], train_x, test_x, positions, args=args, seed=seed)
            row.update({"seed": int(seed), "dataset": "nsynth", "representation": "cqt"})
            rows.append(row)
            write_csv(output_dir / "nsynth_cqt_results.csv", rows)
    aggregate = aggregate_rows(rows)
    write_csv(output_dir / "nsynth_cqt_aggregate.csv", aggregate)
    plot_results(aggregate, output_dir / "nsynth_cqt_masked_r2.pdf")
    summary: dict[str, object] = {
        "status": "ok",
        "data_root": str(data_root),
        "device": str(device),
        "num_records": len(records),
        "train_count": len(train_records),
        "test_count": len(test_records),
        "time_frames": int(args.time_frames),
        "cqt_bins": int(args.cqt_bins),
        "patch_time": int(args.patch_time),
        "patch_freq": int(args.patch_freq),
        "time_tokens": int(time_tokens),
        "freq_tokens": int(freq_tokens),
        "mask_rate": float(args.mask_rate),
        "mask_mode": str(args.mask_mode),
        "steps": int(args.steps),
        "seeds": parse_int_list(args.seeds),
        "bases": selected,
        "cache_dir": str(cache_dir),
        "normalization": stats,
        "results_csv": str(output_dir / "nsynth_cqt_results.csv"),
        "aggregate_csv": str(output_dir / "nsynth_cqt_aggregate.csv"),
    }
    (output_dir / "nsynth_cqt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stale_status = output_dir / "nsynth_cqt_data_status.json"
    if stale_status.exists():
        stale_status.unlink()
    write_report(output_dir, summary, aggregate)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
