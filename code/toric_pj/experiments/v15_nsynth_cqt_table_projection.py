from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import Basis, condition_number, normalize_columns
from toric_pj.experiments.v14_nsynth_cqt_masked import (
    compute_cqt,
    discover_nsynth_records,
    pairwise_d,
    patchify,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V15 NSynth CQT empirical offset-table projection.")
    parser.add_argument("--data-root", type=str, default="data/nsynth/nsynth-test")
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v15_nsynth_cqt_table_projection")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--min-samples", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--cqt-bins", type=int, default=84)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--fmin-note", type=str, default="C1")
    parser.add_argument("--time-frames", type=int, default=128)
    parser.add_argument("--patch-time", type=int, default=4)
    parser.add_argument("--patch-freq", type=int, default=6)
    parser.add_argument("--top-freqs", type=int, default=8)
    parser.add_argument("--dct-atoms", type=int, default=73)
    parser.add_argument("--ridge", type=float, default=1e-8)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_positions(time_tokens: int, freq_tokens: int, device: torch.device) -> torch.Tensor:
    tt = torch.arange(time_tokens, device=device, dtype=torch.float32)
    ff = torch.arange(freq_tokens, device=device, dtype=torch.float32)
    t_grid, f_grid = torch.meshgrid(tt, ff, indexing="ij")
    return torch.stack([t_grid.reshape(-1), f_grid.reshape(-1)], dim=1)


def offset_grid(time_tokens: int, freq_tokens: int, device: torch.device) -> torch.Tensor:
    dt = torch.arange(-(time_tokens - 1), time_tokens, device=device, dtype=torch.float32)
    df = torch.arange(-(freq_tokens - 1), freq_tokens, device=device, dtype=torch.float32)
    t_grid, f_grid = torch.meshgrid(dt, df, indexing="ij")
    return torch.stack([t_grid.reshape(-1), f_grid.reshape(-1)], dim=1)


def load_patch_data(args: argparse.Namespace, device: torch.device, cache_dir: Path) -> tuple[torch.Tensor, dict[str, object]]:
    records = discover_nsynth_records(Path(args.data_root), max_samples=args.max_samples)
    if len(records) < args.min_samples:
        raise RuntimeError(f"found {len(records)} NSynth wav files, need at least {args.min_samples}")
    specs = [compute_cqt(record, args, cache_dir) for record in records]
    data = torch.tensor(np.stack(specs), dtype=torch.float32, device=device)
    mean = data.mean(dim=(0, 1), keepdim=True)
    std = data.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    data = (data - mean) / std
    patches, time_tokens, freq_tokens = patchify(data, args.patch_time, args.patch_freq)
    stats = {
        "num_records": len(records),
        "time_tokens": int(time_tokens),
        "freq_tokens": int(freq_tokens),
        "patch_dim": int(patches.shape[-1]),
        "mean_global": float(mean.mean().detach().cpu()),
        "std_global": float(std.mean().detach().cpu()),
    }
    return patches, stats


def empirical_offset_table(patches: torch.Tensor, time_tokens: int, freq_tokens: int) -> torch.Tensor:
    n_samples, n_positions, patch_dim = patches.shape
    centered = patches - patches.mean(dim=(0, 1), keepdim=True)
    positions = make_positions(time_tokens, freq_tokens, patches.device)
    d_pair = pairwise_d(positions).reshape(-1, 2).to(torch.long)
    height = 2 * time_tokens - 1
    width = 2 * freq_tokens - 1
    inverse = (d_pair[:, 0] + time_tokens - 1) * width + (d_pair[:, 1] + freq_tokens - 1)
    counts = torch.bincount(inverse, minlength=height * width).to(dtype=torch.float64).clamp_min(1.0)
    sums = torch.zeros(height * width, device=patches.device, dtype=torch.float64)
    for idx in range(n_samples):
        x = centered[idx]
        sim = (x @ x.T).reshape(-1).to(torch.float64) / float(patch_dim)
        sums = sums + torch.bincount(inverse, weights=sim, minlength=height * width)
    return (sums / (counts * float(n_samples))).to(torch.float32).reshape(height, width)


def top_fft_omegas(table: torch.Tensor, *, k: int) -> list[torch.Tensor]:
    centered = table - table.mean()
    power = torch.fft.fft2(centered).abs().square()
    power[0, 0] = 0.0
    height, width = table.shape
    flat = power.reshape(-1)
    order = torch.argsort(flat, descending=True)
    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in order.detach().cpu().tolist():
        kt = item // width
        kf = item % width
        signed_t = kt if kt <= height // 2 else kt - height
        signed_f = kf if kf <= width // 2 else kf - width
        if signed_t == 0 and signed_f == 0:
            continue
        key = canonical_freq_key(signed_t, signed_f)
        if key in seen:
            continue
        seen.add(key)
        selected.append((signed_t, signed_f))
        if len(selected) >= k:
            break
    return [
        torch.tensor(
            [2.0 * math.pi * signed_t / float(height), 2.0 * math.pi * signed_f / float(width)],
            device=table.device,
            dtype=torch.float32,
        )
        for signed_t, signed_f in selected
    ]


def canonical_freq_key(kt: int, kf: int) -> tuple[int, int]:
    a = (kt, kf)
    b = (-kt, -kf)
    return min(a, b)


def full_jet_basis(
    d: torch.Tensor,
    omegas: list[torch.Tensor],
    *,
    order: int,
    time_tokens: int,
    freq_tokens: int,
    name: str,
) -> Basis:
    tau = d[:, 0] / float(max(time_tokens - 1, 1))
    phi = d[:, 1] / float(max(freq_tokens - 1, 1))
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    for omega_idx, omega in enumerate(omegas):
        phase = d @ omega.to(device=d.device, dtype=d.dtype)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        for total in range(order + 1):
            for rt in range(total + 1):
                rf = total - rt
                poly = torch.ones_like(cos_phase) if total == 0 else tau.pow(rt) * phi.pow(rf)
                cols.extend([poly * cos_phase, poly * sin_phase])
                labels.extend([f"w{omega_idx}_j{rt}{rf}_cos", f"w{omega_idx}_j{rt}{rf}_sin"])
                orders.extend([total, total])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def dct_top_indices(table: torch.Tensor, *, k: int) -> torch.Tensor:
    centered = table - table.mean()
    height, width = centered.shape
    t = torch.arange(height, device=table.device, dtype=table.dtype)
    f = torch.arange(width, device=table.device, dtype=table.dtype)
    kt = torch.arange(height, device=table.device, dtype=table.dtype)
    kf = torch.arange(width, device=table.device, dtype=table.dtype)
    ct = torch.cos(math.pi / float(height) * (t.reshape(-1, 1) + 0.5) * kt.reshape(1, -1))
    cf = torch.cos(math.pi / float(width) * (f.reshape(-1, 1) + 0.5) * kf.reshape(1, -1))
    coeff = (ct.T @ centered @ cf).square()
    coeff[0, 0] = 0.0
    _, idx = torch.topk(coeff.reshape(-1), k=min(k, coeff.numel() - 1))
    return torch.stack([idx // width, idx % width], dim=1)


def dct_basis(d: torch.Tensor, time_tokens: int, freq_tokens: int, indices: torch.Tensor, *, name: str) -> Basis:
    height = 2 * time_tokens - 1
    width = 2 * freq_tokens - 1
    t = d[:, 0] + float(time_tokens - 1)
    f = d[:, 1] + float(freq_tokens - 1)
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    for rank, (kt, kf) in enumerate(indices.detach().cpu().tolist(), start=1):
        col = torch.cos(math.pi / float(height) * (t + 0.5) * int(kt)) * torch.cos(
            math.pi / float(width) * (f + 0.5) * int(kf)
        )
        cols.append(col)
        labels.append(f"dct{rank}_kt{int(kt)}_kf{int(kf)}")
    return Basis(name, torch.stack(cols, dim=1), labels, [0] * len(labels))


def fit_projection(basis: Basis, target: torch.Tensor, *, ridge: float) -> dict[str, object]:
    y = target.reshape(-1, 1).to(device=basis.matrix.device, dtype=basis.matrix.dtype)
    x, norms = normalize_columns(basis.matrix)
    gram = x.T @ x
    rhs = x.T @ y
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coeff = torch.linalg.solve(gram + ridge * eye, rhs)
    pred = x @ coeff
    mse = torch.mean((pred - y).square())
    var = torch.mean((y - y.mean()).square()).clamp_min(1e-30)
    r2 = 1.0 - mse / var
    order_energy = order_contributions(x, coeff, basis.orders)
    return {
        "basis": basis.name,
        "num_features": int(basis.matrix.shape[1]),
        "r2": float(r2.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "condition": condition_number(x),
        "coeff_norm": float(torch.linalg.norm(coeff / norms.reshape(-1, 1)).detach().cpu()),
        "order0_energy": order_energy.get(0, 0.0),
        "order1_energy": order_energy.get(1, 0.0),
        "order2_energy": order_energy.get(2, 0.0),
    }


def order_contributions(x: torch.Tensor, coeff: torch.Tensor, orders: list[int]) -> dict[int, float]:
    pred = x @ coeff
    denom = torch.linalg.norm(pred).clamp_min(1e-30)
    out: dict[int, float] = {}
    for order in sorted(set(orders)):
        idx = torch.tensor([i for i, item in enumerate(orders) if item == order], device=x.device, dtype=torch.long)
        contrib = x[:, idx] @ coeff[idx]
        out[order] = float((torch.linalg.norm(contrib) / denom).detach().cpu())
    return out


def plot_table(table: torch.Tensor, path: Path) -> None:
    arr = table.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(arr, origin="lower", cmap="coolwarm")
    ax.set_title("NSynth CQT empirical offset covariance")
    ax.set_xlabel("frequency displacement")
    ax.set_ylabel("time displacement")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_projection(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["basis"]) for row in rows]
    values = [float(row["r2"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#3d6f6d")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Projection R2")
    ax.set_title("Projection of empirical NSynth CQT offset table")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    lines = [
        "# V15 NSynth CQT Empirical Table Projection",
        "",
        f"Records: {summary['num_records']}",
        f"Token grid: {summary['time_tokens']}x{summary['freq_tokens']}",
        f"Patch dim: {summary['patch_dim']}",
        "",
        "| basis | features | R2 | order0 energy | order1 energy | order2 energy | condition |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['basis']} | {int(row['num_features'])} | {float(row['r2']):.4f} | "
            + f"{float(row['order0_energy']):.3f} | {float(row['order1_energy']):.3f} | "
            + f"{float(row['order2_energy']):.3f} | {float(row['condition']):.2e} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This is a real-data projection diagnostic, not a downstream reconstruction score.",
            "The target is an empirical scalar offset table estimated from CQT patch covariance.",
            "It tests whether the measured NSynth time-frequency geometry has low-dimensional",
            "Toric/PJ structure before asking a small Transformer to exploit it.",
            "Because the PJ columns are non-orthogonal, order-energy entries are contribution",
            "diagnostics rather than normalized variance shares.",
            "",
            "Artifacts:",
            "",
            "- `projection_results.csv`",
            "- `offset_table.npy`",
            "- `offset_table_heatmap.pdf`",
            "- `projection_r2.pdf`",
            "- `summary.json`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cqt_cache"
    patches, stats = load_patch_data(args, device, cache_dir)
    time_tokens = int(stats["time_tokens"])
    freq_tokens = int(stats["freq_tokens"])
    table = empirical_offset_table(patches, time_tokens, freq_tokens)
    d = offset_grid(time_tokens, freq_tokens, device)
    omegas = top_fft_omegas(table, k=args.top_freqs)
    dct_indices_13 = dct_top_indices(table, k=max(12, min(args.dct_atoms, 12)))
    dct_indices_full = dct_top_indices(table, k=args.dct_atoms - 1)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    shuffled = d[torch.randperm(d.shape[0], device=device, generator=gen)]
    bases = [
        Basis("constant", torch.ones((d.shape[0], 1), device=device, dtype=torch.float32), ["const"], [0]),
        full_jet_basis(d, omegas, order=0, time_tokens=time_tokens, freq_tokens=freq_tokens, name=f"fft_top{args.top_freqs}_J0"),
        full_jet_basis(d, omegas, order=1, time_tokens=time_tokens, freq_tokens=freq_tokens, name=f"fft_top{args.top_freqs}_J1"),
        full_jet_basis(d, omegas, order=2, time_tokens=time_tokens, freq_tokens=freq_tokens, name=f"fft_top{args.top_freqs}_J2"),
        full_jet_basis(
            shuffled,
            omegas,
            order=2,
            time_tokens=time_tokens,
            freq_tokens=freq_tokens,
            name=f"fft_top{args.top_freqs}_J2_coord_shuffle",
        ),
        dct_basis(d, time_tokens, freq_tokens, dct_indices_13, name="dct_top13"),
        dct_basis(d, time_tokens, freq_tokens, dct_indices_full, name=f"dct_top{args.dct_atoms}"),
    ]
    rows = [fit_projection(basis, table.reshape(-1), ridge=args.ridge) for basis in bases]
    rows.sort(key=lambda row: float(row["r2"]), reverse=True)
    write_csv(output_dir / "projection_results.csv", rows)
    np.save(output_dir / "offset_table.npy", table.detach().cpu().numpy())
    plot_table(table, output_dir / "offset_table_heatmap.pdf")
    plot_projection(rows, output_dir / "projection_r2.pdf")
    summary = {
        "status": "ok",
        "data_root": args.data_root,
        "device": str(device),
        "cache_dir": str(cache_dir),
        "top_freqs": int(args.top_freqs),
        "dct_atoms": int(args.dct_atoms),
        **stats,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, summary, rows)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
