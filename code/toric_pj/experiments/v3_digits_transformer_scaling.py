from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn
from torch.nn import functional as F

from toric_pj.diagnostics.basis_projection import (
    Basis,
    axis_additive_fourier_basis,
    default_device,
    directional_jet_basis,
    normalize_columns,
    toric_fourier_basis,
)
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.experiments.real_digits_probe import make_positions, pairwise_d, raster_1d_basis, relative_2d_table_basis


PRUNED_REAL_DIGITS_GROUPS = {
    "const",
    "w0_r0",
    "w0_u0_r1",
    "w0_u1_r1",
    "w0_u1_r2",
    "w0_u2_r2",
    "w0_u3_r1",
    "w0_u3_r2",
    "w1_r0",
    "w1_u1_r2",
    "w1_u3_r2",
    "w2_r0",
    "w2_u0_r2",
    "w2_u1_r1",
    "w2_u1_r2",
    "w2_u3_r1",
    "w2_u3_r2",
}


def load_digits_split(device: torch.device, *, seed: int, train_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = load_digits()
    images = torch.tensor(data.data, device=device, dtype=torch.float32) / 16.0
    labels = torch.tensor(data.target, device=device, dtype=torch.long)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    perm = torch.randperm(images.shape[0], device=device, generator=gen)
    train_idx = perm[:train_count]
    test_idx = perm[train_count:]
    train_x = images[train_idx]
    test_x = images[test_idx]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std().clamp_min(1e-6)
    return (train_x - mean) / std, labels[train_idx], (test_x - mean) / std, labels[test_idx]


def no_pos_basis(d: torch.Tensor) -> Basis:
    return Basis("no_pos_constant", torch.ones((d.shape[0], 1), device=d.device, dtype=d.dtype), ["const"], [0])


def label_group(label: str) -> str:
    if label == "const":
        return label
    if label.endswith("_cos") or label.endswith("_sin"):
        return label.rsplit("_", 1)[0]
    return label


def prune_basis(basis: Basis, groups: set[str], *, name: str) -> Basis:
    keep = [idx for idx, label in enumerate(basis.labels) if label_group(label) in groups]
    idx = torch.tensor(keep, device=basis.matrix.device, dtype=torch.long)
    matrix = basis.matrix[:, idx]
    labels = [basis.labels[item] for item in keep]
    orders = [basis.orders[item] for item in keep]
    return Basis(name, matrix, labels, orders)


def build_bases(d: torch.Tensor, side: int, *, include_shuffle: bool = True, seed: int = 0) -> list[Basis]:
    omega_a = torch.tensor([0.78, 0.42], device=d.device, dtype=d.dtype)
    omega_b = torch.tensor([0.62, -0.58], device=d.device, dtype=d.dtype)
    omega_c = torch.tensor([0.35, 0.91], device=d.device, dtype=d.dtype)
    ex = torch.tensor([1.0, 0.0], device=d.device, dtype=d.dtype)
    ey = torch.tensor([0.0, 1.0], device=d.device, dtype=d.dtype)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1)
    oblique = normalize_direction(torch.tensor([[1.0, -0.65]], device=d.device, dtype=d.dtype)).reshape(-1)
    toric_pj = directional_jet_basis(
        d,
        [omega_a, omega_b, omega_c],
        [ex, ey, diag, oblique],
        [0, 1, 2],
        scale=float(side),
        name="toric_PJ_R2",
    )
    bases = [
        no_pos_basis(d),
        raster_1d_basis(d, side),
        axis_additive_fourier_basis(d, [0.78, 0.58], name="axis_additive"),
        toric_fourier_basis(d, [omega_a, omega_b, omega_c], name="toric_order0"),
        toric_pj,
        prune_basis(toric_pj, PRUNED_REAL_DIGITS_GROUPS, name="pruned_toric_PJ"),
        relative_2d_table_basis(d),
    ]
    if include_shuffle:
        gen = torch.Generator(device=d.device)
        gen.manual_seed(seed)
        shuffled = d[torch.randperm(d.shape[0], device=d.device, generator=gen)]
        bases.append(
            directional_jet_basis(
                shuffled,
                [omega_a, omega_b, omega_c],
                [ex, ey, diag, oblique],
                [0, 1, 2],
                scale=float(side),
                name="toric_PJ_R2_coord_shuffle",
            )
        )
    return bases


class RelPosBlock(nn.Module):
    def __init__(self, *, dim: int, n_heads: int, n_features: int, n_positions: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_positions = n_positions
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(ffn_mult) * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(ffn_mult) * dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.coeff = nn.Parameter(torch.zeros(n_heads, n_features))

    def forward(self, x: torch.Tensor, basis_matrix: torch.Tensor, bias_override: torch.Tensor | None = None) -> torch.Tensor:
        bsz = x.shape[0]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(bsz, self.n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, self.n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, self.n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(float(self.head_dim))
        if bias_override is None:
            bias = torch.einsum("nf,hf->hn", basis_matrix, self.coeff).reshape(
                self.n_heads, self.n_positions, self.n_positions
            )
        else:
            bias = bias_override.to(device=x.device, dtype=logits.dtype)
        logits = logits + bias.unsqueeze(0)
        attn = torch.softmax(logits, dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", attn, v).transpose(1, 2).reshape(bsz, self.n_positions, -1)
        x = self.norm1(x + self.dropout(self.out(context)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DeepPositionalTransformer(nn.Module):
    def __init__(
        self,
        basis: Basis,
        *,
        n_positions: int,
        input_dim: int,
        dim: int,
        n_heads: int,
        depth: int,
        ffn_mult: int,
        dropout: float,
        n_classes: int,
    ) -> None:
        super().__init__()
        matrix, _ = normalize_columns(basis.matrix.to(dtype=torch.float32))
        self.register_buffer("basis_matrix", matrix)
        self.basis_name = basis.name
        self.orders = list(basis.orders)
        self.n_positions = n_positions
        self.input = nn.Linear(input_dim, dim)
        self.blocks = nn.ModuleList(
            [
                RelPosBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_features=matrix.shape[1],
                    n_positions=n_positions,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.reconstruct = nn.Linear(dim, 1)
        self.classifier = nn.Linear(dim, n_classes)

    def encode(self, features: torch.Tensor, bias_overrides: list[torch.Tensor] | None = None) -> torch.Tensor:
        x = self.input(features)
        for idx, block in enumerate(self.blocks):
            bias_override = None if bias_overrides is None else bias_overrides[idx]
            x = block(x, self.basis_matrix, bias_override=bias_override)
        return x

    def forward_reconstruction(
        self,
        pixels: torch.Tensor,
        mask: torch.Tensor,
        *,
        noise_std: float = 0.0,
        bias_overrides: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        visible = pixels.masked_fill(mask, 0.0)
        if noise_std > 0:
            noise = noise_std * torch.randn_like(visible)
            visible = torch.where(mask, visible, visible + noise)
        features = torch.stack([visible, (~mask).to(pixels.dtype)], dim=-1)
        return self.reconstruct(self.encode(features, bias_overrides=bias_overrides)).squeeze(-1)

    def forward_classification(self, pixels: torch.Tensor, bias_overrides: list[torch.Tensor] | None = None) -> torch.Tensor:
        features = torch.stack([pixels, torch.ones_like(pixels)], dim=-1)
        return self.classifier(self.encode(features, bias_overrides=bias_overrides).mean(dim=1))

    def bias_stats(self) -> dict[str, float]:
        with torch.no_grad():
            coeff = torch.stack([block.coeff.detach() for block in self.blocks], dim=0)
            flat = coeff.reshape(-1, coeff.shape[-1])
            denom = torch.linalg.norm(flat).clamp_min(1e-12)
            out: dict[str, float] = {"coeff_norm": float(torch.linalg.norm(coeff).detach().cpu())}
            for order in sorted(set(self.orders)):
                idx = torch.tensor([i for i, item in enumerate(self.orders) if item == order], device=coeff.device, dtype=torch.long)
                if idx.numel() > 0:
                    out[f"order{order}_coeff_share"] = float((torch.linalg.norm(flat[:, idx]) / denom).detach().cpu())
            return out


def sample_batch(x: torch.Tensor, y: torch.Tensor, *, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device, generator=generator)
    return x[idx], y[idx]


def autocast_context(device: torch.device, amp: str):
    enabled = device.type == "cuda" and amp != "none"
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def make_model(basis: Basis, args: argparse.Namespace, *, n_positions: int) -> DeepPositionalTransformer:
    return DeepPositionalTransformer(
        basis,
        n_positions=n_positions,
        input_dim=2,
        dim=args.dim,
        n_heads=args.n_heads,
        depth=args.depth,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        n_classes=10,
    )


def evaluate_reconstruction(
    model: DeepPositionalTransformer,
    x: torch.Tensor,
    *,
    args: argparse.Namespace,
    seed: int,
    noise_std: float = 0.0,
) -> dict[str, float]:
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    targets = []
    preds = []
    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], args.eval_batch_size):
            batch = x[start : start + args.eval_batch_size]
            mask = torch.rand(batch.shape, device=batch.device, generator=gen) < args.mask_rate
            with autocast_context(batch.device, args.amp):
                pred = model.forward_reconstruction(batch, mask, noise_std=noise_std)
                loss = torch.mean((pred[mask] - batch[mask]).square())
            losses.append(loss.float())
            targets.append(batch[mask].float())
            preds.append(pred[mask].float())
    target = torch.cat(targets)
    pred_all = torch.cat(preds)
    mse = torch.mean((pred_all - target).square())
    var = torch.mean((target - target.mean()).square()).clamp_min(1e-30)
    model.train()
    return {"loss": float(torch.stack(losses).mean().detach().cpu()), "score": float((1.0 - mse / var).detach().cpu())}


def evaluate_classifier(model: DeepPositionalTransformer, x: torch.Tensor, y: torch.Tensor, *, args: argparse.Namespace) -> dict[str, float]:
    losses = []
    correct = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], args.eval_batch_size):
            batch = x[start : start + args.eval_batch_size]
            labels = y[start : start + args.eval_batch_size]
            with autocast_context(batch.device, args.amp):
                logits = model.forward_classification(batch)
                loss = F.cross_entropy(logits.float(), labels)
            losses.append(loss)
            correct.append((torch.argmax(logits, dim=-1) == labels).float())
    model.train()
    return {"loss": float(torch.stack(losses).mean().detach().cpu()), "score": float(torch.cat(correct).mean().detach().cpu())}


def run_task(
    basis: Basis,
    *,
    task: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=train_x.device)
    gen.manual_seed(seed + 101)
    model = make_model(basis, args, n_positions=train_x.shape[1]).to(train_x.device)
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps), eta_min=args.lr * 0.05)
    curves: list[dict[str, object]] = []
    best_score = -1e9
    best_step = -1
    wall_start = time.time()
    for step in range(args.steps):
        batch, labels = sample_batch(train_x, train_y, batch_size=args.batch_size, generator=gen)
        opt.zero_grad(set_to_none=True)
        with autocast_context(train_x.device, args.amp):
            if task == "classification":
                logits = model.forward_classification(batch)
                loss = F.cross_entropy(logits.float(), labels)
            elif task == "multitask":
                mask = torch.rand(batch.shape, device=batch.device, generator=gen) < args.mask_rate
                pred = model.forward_reconstruction(batch, mask)
                logits = model.forward_classification(batch)
                loss = F.cross_entropy(logits.float(), labels) + args.lambda_recon * torch.mean((pred[mask] - batch[mask]).square())
            else:
                mask = torch.rand(batch.shape, device=batch.device, generator=gen) < args.mask_rate
                noise_std = args.noise_std if task == "denoise" else 0.0
                pred = model.forward_reconstruction(batch, mask, noise_std=noise_std)
                loss = torch.mean((pred[mask] - batch[mask]).square())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            if task == "classification":
                train_metric = evaluate_classifier(model, train_x[: min(args.eval_subset, train_x.shape[0])], train_y[: min(args.eval_subset, train_y.shape[0])], args=args)
                test_metric = evaluate_classifier(model, test_x, test_y, args=args)
                metric_name = "accuracy"
            elif task == "multitask":
                train_metric = evaluate_classifier(model, train_x[: min(args.eval_subset, train_x.shape[0])], train_y[: min(args.eval_subset, train_y.shape[0])], args=args)
                test_metric = evaluate_classifier(model, test_x, test_y, args=args)
                metric_name = "accuracy"
            else:
                noise_std = args.noise_std if task == "denoise" else 0.0
                train_metric = evaluate_reconstruction(
                    model,
                    train_x[: min(args.eval_subset, train_x.shape[0])],
                    args=args,
                    seed=seed + 2000 + step,
                    noise_std=noise_std,
                )
                test_metric = evaluate_reconstruction(model, test_x, args=args, seed=seed + 3000 + step, noise_std=noise_std)
                metric_name = "masked_r2"
            if test_metric["score"] > best_score:
                best_score = test_metric["score"]
                best_step = step
            curves.append(
                {
                    "task": task,
                    "basis": basis.name,
                    "seed": seed,
                    "step": step,
                    "metric": metric_name,
                    "train_score": train_metric["score"],
                    "test_score": test_metric["score"],
                    "train_loss": train_metric["loss"],
                    "test_loss": test_metric["loss"],
                    "elapsed_sec": time.time() - wall_start,
                }
            )
    if task in {"classification", "multitask"}:
        final_train = evaluate_classifier(model, train_x[: min(args.eval_subset, train_x.shape[0])], train_y[: min(args.eval_subset, train_y.shape[0])], args=args)
        final_test = evaluate_classifier(model, test_x, test_y, args=args)
        metric_name = "accuracy"
    else:
        noise_std = args.noise_std if task == "denoise" else 0.0
        final_train = evaluate_reconstruction(
            model,
            train_x[: min(args.eval_subset, train_x.shape[0])],
            args=args,
            seed=seed + 4000,
            noise_std=noise_std,
        )
        final_test = evaluate_reconstruction(model, test_x, args=args, seed=seed + 5000, noise_std=noise_std)
        metric_name = "masked_r2"
    stats = model._orig_mod.bias_stats() if hasattr(model, "_orig_mod") else model.bias_stats()
    row: dict[str, object] = {
        "task": task,
        "basis": basis.name,
        "seed": seed,
        "metric": metric_name,
        "score": final_test["score"],
        "loss": final_test["loss"],
        "train_score": final_train["score"],
        "train_loss": final_train["loss"],
        "best_score": best_score,
        "best_step": best_step,
        "num_features": basis.matrix.shape[1],
        "param_count": sum(param.numel() for param in model.parameters()),
        "wall_sec": time.time() - wall_start,
        **stats,
    }
    return row, curves


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["basis"])), []).append(row)
    out = []
    for (task, basis), values in sorted(groups.items()):
        scores = np.array([float(row["score"]) for row in values])
        best = np.array([float(row["best_score"]) for row in values])
        out.append(
            {
                "task": task,
                "basis": basis,
                "n": len(values),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "score_best": float(scores.max()),
                "best_score_best": float(best.max()),
                "loss_mean": float(np.mean([float(row["loss"]) for row in values])),
                "train_score_mean": float(np.mean([float(row["train_score"]) for row in values])),
                "num_features": int(values[0]["num_features"]),
                "param_count": int(values[0]["param_count"]),
                "wall_sec_max": float(max(float(row["wall_sec"]) for row in values)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_results(output_dir: Path, aggregate_rows: list[dict[str, object]], curves: list[dict[str, object]]) -> None:
    tasks = sorted({str(row["task"]) for row in aggregate_rows})
    fig, axes = plt.subplots(max(1, len(tasks)), 1, figsize=(12, max(4, 4 * len(tasks))), squeeze=False)
    for ax, task in zip(axes.reshape(-1), tasks):
        vals = [row for row in aggregate_rows if row["task"] == task]
        x = np.arange(len(vals))
        y = [float(row["score_mean"]) for row in vals]
        err = [float(row["score_std"]) for row in vals]
        ax.bar(x, y, yerr=err, color="#5b718f")
        ax.set_xticks(x, [str(row["basis"]) for row in vals], rotation=25, ha="right")
        ax.set_title(task)
        ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(output_dir / "digits_scaling_scores.png", dpi=180)
    plt.close(fig)

    if curves:
        fig, ax = plt.subplots(figsize=(12, 5))
        for key in sorted({(row["task"], row["basis"]) for row in curves}):
            vals = [row for row in curves if (row["task"], row["basis"]) == key]
            grouped: dict[int, list[float]] = {}
            for row in vals:
                grouped.setdefault(int(row["step"]), []).append(float(row["test_score"]))
            steps = sorted(grouped)
            ax.plot(steps, [float(np.mean(grouped[step])) for step in steps], label=f"{key[0]} {key[1]}")
        ax.set_xlabel("step")
        ax.set_ylabel("test score")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "train_test_curves.png", dpi=180)
        plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], aggregate_rows: list[dict[str, object]]) -> None:
    lines = [
        "# V3-B Digits Transformer Scaling Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        "- Dataset: sklearn digits",
        f"- Mode: {summary['mode']}",
        f"- Depth / dim / heads: {summary['depth']} / {summary['dim']} / {summary['n_heads']}",
        f"- Steps: {summary['steps']}",
        f"- Seeds: {summary['seeds']}",
        f"- Wall seconds: {summary['wall_sec']:.2f}",
        "",
        "## Aggregate",
        "",
        "| task | basis | n | score mean | score std | best | train score | features | params |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + f"{row['task']} | {row['basis']} | {int(row['n'])} | {float(row['score_mean']):.4f} | "
            + f"{float(row['score_std']):.4f} | {float(row['score_best']):.4f} | {float(row['train_score_mean']):.4f} | "
            + f"{int(row['num_features'])} | {int(row['param_count'])} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Reconstruction/denoise scores are masked-pixel R2.",
            "- Classification/multitask scores are accuracy.",
            "- `toric_PJ_R2_coord_shuffle` is a geometry control: it keeps capacity while breaking the coordinate relation.",
            "",
            "Artifacts:",
            "",
            "- `digits_scaling_results.csv`",
            "- `digits_scaling_aggregate.csv`",
            "- `digits_scaling_curves.csv`",
            "- `digits_scaling_scores.png`",
            "- `train_test_curves.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_x, train_y, test_x, test_y = load_digits_split(device, seed=args.seed, train_count=args.train_count)
    positions = make_positions(args.side, device)
    d = pairwise_d(positions).reshape(-1, 2).to(torch.float32)
    bases = build_bases(d, args.side, include_shuffle=args.include_shuffle, seed=args.seed)
    selected_basis_names = set(parse_list(args.bases)) if args.bases != "all" else None
    if args.mode == "smoke" and selected_basis_names is None:
        selected_basis_names = {"no_pos_constant", "toric_order0", "toric_PJ_R2", "relative_2d_table"}
    selected_bases = [basis for basis in bases if selected_basis_names is None or basis.name in selected_basis_names]
    if args.seed_basis_order:
        seed_basis_order = parse_list(args.seed_basis_order)
        basis_seed_indices = {name: idx for idx, name in enumerate(seed_basis_order)}
    else:
        basis_seed_indices = {basis.name: idx for idx, basis in enumerate(selected_bases)}
    tasks = parse_list(args.tasks)
    if args.mode == "smoke" and args.tasks == "reconstruction,classification":
        tasks = ["reconstruction", "classification"]
    rows: list[dict[str, object]] = read_csv(output_dir / "digits_scaling_results.csv") if args.resume else []
    curves: list[dict[str, object]] = read_csv(output_dir / "digits_scaling_curves.csv") if args.resume else []
    completed = {
        (str(row.get("task")), str(row.get("basis")), int(float(row.get("seed", -1))))
        for row in rows
    }
    runs_done = 0
    wall_start = time.time()
    for basis_idx, basis in enumerate(selected_bases):
        for task_idx, task in enumerate(tasks):
            for seed_idx in range(args.seed_start_idx, args.seeds):
                basis_seed_idx = basis_seed_indices.get(basis.name, basis_idx)
                seed = args.seed + 100 * seed_idx + 17 * basis_seed_idx + 1009 * task_idx
                if (task, basis.name, seed) in completed:
                    continue
                if args.max_runs is not None and runs_done >= args.max_runs:
                    break
                row, task_curves = run_task(
                    basis,
                    task=task,
                    train_x=train_x,
                    train_y=train_y,
                    test_x=test_x,
                    test_y=test_y,
                    args=args,
                    seed=seed,
                )
                rows.append(row)
                curves.extend(task_curves)
                completed.add((task, basis.name, seed))
                runs_done += 1
                aggregate_rows = aggregate(rows)
                write_csv(output_dir / "digits_scaling_results.csv", rows)
                write_csv(output_dir / "digits_scaling_aggregate.csv", aggregate_rows)
                write_csv(output_dir / "digits_scaling_curves.csv", curves)
                partial = {
                    "device": str(device),
                    "mode": args.mode,
                    "depth": args.depth,
                    "dim": args.dim,
                    "n_heads": args.n_heads,
                    "steps": args.steps,
                    "seeds": args.seeds,
                    "tasks": tasks,
                    "bases": [basis.name for basis in selected_bases],
                    "seed_start_idx": args.seed_start_idx,
                    "seed_basis_order": args.seed_basis_order,
                    "max_runs": args.max_runs,
                    "resume": args.resume,
                    "runs_done_this_invocation": runs_done,
                    "wall_sec": time.time() - wall_start,
                    "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
                    if torch.cuda.is_available() and device.type == "cuda"
                    else 0,
                    "rows": rows,
                    "aggregate_rows": aggregate_rows,
                    "curves": curves,
                    "status": "partial",
                }
                (output_dir / "digits_scaling_summary.partial.json").write_text(
                    json.dumps(partial, indent=2), encoding="utf-8"
                )
            if args.max_runs is not None and runs_done >= args.max_runs:
                break
        if args.max_runs is not None and runs_done >= args.max_runs:
            break
    aggregate_rows = aggregate(rows)
    write_csv(output_dir / "digits_scaling_results.csv", rows)
    write_csv(output_dir / "digits_scaling_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "digits_scaling_curves.csv", curves)
    plot_results(output_dir, aggregate_rows, curves)
    peak_mem = 0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated(device))
    summary = {
        "device": str(device),
        "mode": args.mode,
        "depth": args.depth,
        "dim": args.dim,
        "n_heads": args.n_heads,
        "steps": args.steps,
        "seeds": args.seeds,
        "tasks": tasks,
        "bases": [basis.name for basis in selected_bases],
        "seed_start_idx": args.seed_start_idx,
        "seed_basis_order": args.seed_basis_order,
        "max_runs": args.max_runs,
        "resume": args.resume,
        "runs_done_this_invocation": runs_done,
        "wall_sec": time.time() - wall_start,
        "peak_cuda_memory_bytes": peak_mem,
        "rows": rows,
        "aggregate_rows": aggregate_rows,
        "curves": curves,
    }
    summary_path = output_dir / "digits_scaling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, summary, aggregate_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3-B sklearn digits Transformer scaling.")
    parser.add_argument("--mode", choices=["smoke", "main", "overnight"], default="smoke")
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=1400)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--ffn-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--tasks", type=str, default="reconstruction,classification")
    parser.add_argument("--bases", type=str, default="all")
    parser.add_argument("--mask-rate", type=float, default=0.35)
    parser.add_argument("--noise-std", type=float, default=0.12)
    parser.add_argument("--lambda-recon", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-subset", type=int, default=1024)
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--include-shuffle", action="store_true", default=True)
    parser.add_argument("--seed-start-idx", type=int, default=0)
    parser.add_argument("--seed-basis-order", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=915)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v3_digits_scaling")
    args = parser.parse_args()
    if args.depth is None:
        args.depth = {"smoke": 2, "main": 4, "overnight": 6}[args.mode]
    if args.dim is None:
        args.dim = {"smoke": 96, "main": 256, "overnight": 384}[args.mode]
    if args.n_heads is None:
        args.n_heads = {"smoke": 4, "main": 8, "overnight": 8}[args.mode]
    if args.batch_size is None:
        args.batch_size = {"smoke": 128, "main": 512, "overnight": 512}[args.mode]
    if args.steps is None:
        args.steps = {"smoke": 220, "main": 20000, "overnight": 50000}[args.mode]
    if args.seeds is None:
        args.seeds = {"smoke": 1, "main": 5, "overnight": 5}[args.mode]
    if args.eval_every is None:
        args.eval_every = max(50, args.steps // 40)
    return args


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key not in {"rows", "aggregate_rows", "curves"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
