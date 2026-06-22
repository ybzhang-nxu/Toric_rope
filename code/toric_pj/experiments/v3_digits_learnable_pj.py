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
from torch import nn
from torch.nn import functional as F

from toric_pj.diagnostics.basis_projection import Basis, default_device, normalize_columns
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.experiments.real_digits_probe import make_positions, pairwise_d, relative_2d_table_basis
from toric_pj.experiments.v3_digits_transformer_scaling import (
    DeepPositionalTransformer,
    autocast_context,
    build_bases,
    load_digits_split,
    sample_batch,
    write_csv,
)


def image_prior_geometry(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    omegas = torch.tensor([[0.78, 0.42], [0.62, -0.58], [0.35, 0.91]], device=device, dtype=dtype)
    dirs = torch.stack(
        [
            torch.tensor([1.0, 0.0], device=device, dtype=dtype),
            torch.tensor([0.0, 1.0], device=device, dtype=dtype),
            normalize_direction(torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)).reshape(-1),
            normalize_direction(torch.tensor([[1.0, -0.65]], device=device, dtype=dtype)).reshape(-1),
        ],
        dim=0,
    )
    return omegas, dirs


class LearnablePJBasis(nn.Module):
    def __init__(
        self,
        d: torch.Tensor,
        *,
        side: int,
        init: str,
        max_order: int = 2,
        coord_shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        d = d.to(dtype=torch.float32)
        if coord_shuffle:
            gen = torch.Generator(device=d.device)
            gen.manual_seed(seed)
            d = d[torch.randperm(d.shape[0], device=d.device, generator=gen)]
        self.register_buffer("d", d)
        self.side = float(side)
        self.max_order = int(max_order)
        prior_omegas, prior_dirs = image_prior_geometry(d.device, d.dtype)
        gen = torch.Generator(device=d.device)
        gen.manual_seed(seed + 91)
        if init == "random":
            omegas = 0.85 * torch.randn(prior_omegas.shape, device=d.device, dtype=d.dtype, generator=gen)
            dirs = normalize_direction(torch.randn(prior_dirs.shape, device=d.device, dtype=d.dtype, generator=gen))
        else:
            omegas = prior_omegas + 0.04 * torch.randn(prior_omegas.shape, device=d.device, dtype=d.dtype, generator=gen)
            dirs = normalize_direction(prior_dirs + 0.04 * torch.randn(prior_dirs.shape, device=d.device, dtype=d.dtype, generator=gen))
        self.omega = nn.Parameter(omegas)
        self.raw_dirs = nn.Parameter(dirs)
        self.register_buffer("order_mask", torch.ones(self.max_order, device=d.device, dtype=d.dtype))

    @property
    def normalized_dirs(self) -> torch.Tensor:
        return F.normalize(self.raw_dirs, p=2, dim=-1, eps=1e-12)

    def set_order_limit(self, limit: int) -> None:
        mask = torch.zeros_like(self.order_mask)
        limit = max(0, min(int(limit), self.max_order))
        if limit > 0:
            mask[:limit] = 1.0
        self.order_mask.copy_(mask)

    def feature_labels_orders(self) -> tuple[list[str], list[int]]:
        labels = ["const"]
        orders = [0]
        for wi in range(self.omega.shape[0]):
            labels.extend([f"w{wi}_r0_cos", f"w{wi}_r0_sin"])
            orders.extend([0, 0])
            for ui in range(self.raw_dirs.shape[0]):
                for order in range(1, self.max_order + 1):
                    labels.extend([f"w{wi}_u{ui}_r{order}_cos", f"w{wi}_u{ui}_r{order}_sin"])
                    orders.extend([order, order])
        return labels, orders

    def forward(self) -> torch.Tensor:
        d = self.d
        cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
        phase = torch.einsum("nd,fd->nf", d, self.omega)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        dirs = self.normalized_dirs
        coord = torch.einsum("nd,qd->nq", d, dirs) / max(self.side, 1e-12)
        for wi in range(self.omega.shape[0]):
            cols.extend([cos_phase[:, wi], sin_phase[:, wi]])
            for ui in range(dirs.shape[0]):
                for order in range(1, self.max_order + 1):
                    mask = self.order_mask[order - 1]
                    poly = coord[:, ui].pow(order)
                    cols.extend([mask * poly * cos_phase[:, wi], mask * poly * sin_phase[:, wi]])
        matrix = torch.stack(cols, dim=1)
        norms = torch.linalg.norm(matrix, dim=0).clamp_min(1e-12)
        return matrix / norms


class LearnableRelPosBlock(nn.Module):
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
        self.ff = nn.Sequential(nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.coeff = nn.Parameter(0.01 * torch.randn(n_heads, n_features))

    def forward(self, x: torch.Tensor, basis_matrix: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(bsz, self.n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, self.n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, self.n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(float(self.head_dim))
        bias = torch.einsum("nf,hf->hn", basis_matrix, self.coeff).reshape(self.n_heads, self.n_positions, self.n_positions)
        attn = torch.softmax(logits + bias.unsqueeze(0), dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", attn, v).transpose(1, 2).reshape(bsz, self.n_positions, -1)
        x = self.norm1(x + self.dropout(self.out(context)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class LearnablePJTransformer(nn.Module):
    def __init__(
        self,
        basis: LearnablePJBasis,
        *,
        n_positions: int,
        dim: int,
        n_heads: int,
        depth: int,
        ffn_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.learnable_basis = basis
        labels, orders = basis.feature_labels_orders()
        self.labels = labels
        self.orders = orders
        self.n_positions = n_positions
        self.input = nn.Linear(2, dim)
        self.blocks = nn.ModuleList(
            [
                LearnableRelPosBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_features=len(labels),
                    n_positions=n_positions,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.classifier = nn.Linear(dim, 10)

    def set_order_limit(self, limit: int) -> None:
        self.learnable_basis.set_order_limit(limit)

    def forward_classification(self, pixels: torch.Tensor) -> torch.Tensor:
        features = torch.stack([pixels, torch.ones_like(pixels)], dim=-1)
        x = self.input(features)
        basis_matrix = self.learnable_basis()
        for block in self.blocks:
            x = block(x, basis_matrix)
        return self.classifier(x.mean(dim=1))

    def geometry_stats(self) -> dict[str, object]:
        with torch.no_grad():
            prior_omegas, prior_dirs = image_prior_geometry(self.learnable_basis.omega.device, self.learnable_basis.omega.dtype)
            omega = self.learnable_basis.omega.detach()
            dirs = self.learnable_basis.normalized_dirs.detach()
            freq_err = torch.cdist(omega, prior_omegas).min(dim=1).values
            dir_align = torch.abs(dirs @ prior_dirs.T).max(dim=1).values
            coeff = torch.stack([block.coeff.detach() for block in self.blocks], dim=0)
            flat = coeff.reshape(-1, coeff.shape[-1])
            denom = torch.linalg.norm(flat).clamp_min(1e-12)
            stats: dict[str, object] = {
                "frequency_error_mean": float(freq_err.mean().detach().cpu()),
                "direction_alignment_mean": float(dir_align.mean().detach().cpu()),
                "omega": omega.detach().cpu().tolist(),
                "directions": dirs.detach().cpu().tolist(),
                "coeff_norm": float(torch.linalg.norm(coeff).detach().cpu()),
            }
            for order in sorted(set(self.orders)):
                idx = torch.tensor([i for i, item in enumerate(self.orders) if item == order], device=coeff.device, dtype=torch.long)
                if idx.numel() > 0:
                    stats[f"order{order}_coeff_share"] = float((torch.linalg.norm(flat[:, idx]) / denom).detach().cpu())
            return stats


def fixed_model_from_basis(basis: Basis, args: argparse.Namespace, *, n_positions: int) -> DeepPositionalTransformer:
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


def evaluate_classifier(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    losses = []
    correct = []
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


def train_classifier(
    model: nn.Module,
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    variant: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=train_x.device)
    gen.manual_seed(seed + 117)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps), eta_min=args.lr * 0.05)
    curves = []
    best_score = -1e9
    best_step = -1
    wall_start = time.time()
    for step in range(args.steps):
        if hasattr(model, "set_order_limit"):
            if "curriculum" in variant:
                frac = step / max(1, args.steps - 1)
                model.set_order_limit(0 if frac < 0.34 else 1 if frac < 0.67 else 2)
            else:
                model.set_order_limit(2)
        batch, labels = sample_batch(train_x, train_y, batch_size=args.batch_size, generator=gen)
        opt.zero_grad(set_to_none=True)
        with autocast_context(train_x.device, args.amp):
            logits = model.forward_classification(batch)
            loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            train_metric = evaluate_classifier(model, train_x[: min(args.eval_subset, train_x.shape[0])], train_y[: min(args.eval_subset, train_y.shape[0])], args=args)
            test_metric = evaluate_classifier(model, test_x, test_y, args=args)
            if test_metric["score"] > best_score:
                best_score = test_metric["score"]
                best_step = step
            curves.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "step": step,
                    "train_accuracy": train_metric["score"],
                    "test_accuracy": test_metric["score"],
                    "train_loss": train_metric["loss"],
                    "test_loss": test_metric["loss"],
                    "elapsed_sec": time.time() - wall_start,
                }
            )
    train_metric = evaluate_classifier(model, train_x[: min(args.eval_subset, train_x.shape[0])], train_y[: min(args.eval_subset, train_y.shape[0])], args=args)
    test_metric = evaluate_classifier(model, test_x, test_y, args=args)
    stats: dict[str, object] = {}
    if hasattr(model, "geometry_stats"):
        stats.update(model.geometry_stats())
    elif hasattr(model, "bias_stats"):
        stats.update(model.bias_stats())
    row = {
        "variant": variant,
        "seed": seed,
        "test_accuracy": test_metric["score"],
        "test_loss": test_metric["loss"],
        "train_accuracy": train_metric["score"],
        "train_loss": train_metric["loss"],
        "best_accuracy": best_score,
        "best_step": best_step,
        "param_count": sum(param.numel() for param in model.parameters()),
        "wall_sec": time.time() - wall_start,
        **{key: value for key, value in stats.items() if not isinstance(value, list)},
    }
    return row, curves


def make_variant_model(variant: str, d: torch.Tensor, fixed_bases: dict[str, Basis], args: argparse.Namespace) -> nn.Module:
    n_positions = args.side * args.side
    if variant == "fixed_toric_PJ_R2":
        return fixed_model_from_basis(fixed_bases["toric_PJ_R2"], args, n_positions=n_positions)
    if variant == "pruned_toric_PJ":
        return fixed_model_from_basis(fixed_bases["pruned_toric_PJ"], args, n_positions=n_positions)
    if variant == "relative_2d_table":
        return fixed_model_from_basis(fixed_bases["relative_2d_table"], args, n_positions=n_positions)
    if variant == "toric_PJ_R2_coord_shuffle":
        return fixed_model_from_basis(fixed_bases["toric_PJ_R2_coord_shuffle"], args, n_positions=n_positions)
    if variant.startswith("learnable"):
        init = "random" if "random" in variant else "near"
        coord_shuffle = "coord_shuffle" in variant
        basis = LearnablePJBasis(d, side=args.side, init=init, coord_shuffle=coord_shuffle, seed=args.seed)
        return LearnablePJTransformer(
            basis,
            n_positions=n_positions,
            dim=args.dim,
            n_heads=args.n_heads,
            depth=args.depth,
            ffn_mult=args.ffn_mult,
            dropout=args.dropout,
        )
    raise ValueError(f"unknown variant: {variant}")


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["variant"]), []).append(row)
    out = []
    for variant, values in sorted(groups.items()):
        acc = np.array([float(row["test_accuracy"]) for row in values])
        out.append(
            {
                "variant": variant,
                "n": len(values),
                "test_accuracy_mean": float(acc.mean()),
                "test_accuracy_std": float(acc.std()),
                "test_accuracy_best": float(acc.max()),
                "train_accuracy_mean": float(np.mean([float(row["train_accuracy"]) for row in values])),
                "best_accuracy_best": float(max(float(row["best_accuracy"]) for row in values)),
                "frequency_error_mean": float(np.mean([float(row.get("frequency_error_mean", np.nan)) for row in values]))
                if any("frequency_error_mean" in row for row in values)
                else float("nan"),
                "direction_alignment_mean": float(np.mean([float(row.get("direction_alignment_mean", np.nan)) for row in values]))
                if any("direction_alignment_mean" in row for row in values)
                else float("nan"),
                "param_count": int(values[0]["param_count"]),
                "wall_sec_max": float(max(float(row["wall_sec"]) for row in values)),
            }
        )
    return out


def plot_outputs(output_dir: Path, rows: list[dict[str, object]], curves: list[dict[str, object]]) -> None:
    aggregate_rows = aggregate(rows)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(aggregate_rows))
    ax.bar(x, [float(row["test_accuracy_mean"]) for row in aggregate_rows], color="#5f7295")
    ax.set_xticks(x, [str(row["variant"]) for row in aggregate_rows], rotation=25, ha="right")
    ax.set_ylabel("test accuracy")
    ax.set_title("V3-C fixed vs learnable PJ")
    fig.tight_layout()
    fig.savefig(output_dir / "learnable_digit_accuracy.png", dpi=180)
    plt.close(fig)

    if curves:
        fig, ax = plt.subplots(figsize=(12, 5))
        for variant in sorted({str(row["variant"]) for row in curves}):
            vals = [row for row in curves if row["variant"] == variant]
            grouped: dict[int, list[float]] = {}
            for row in vals:
                grouped.setdefault(int(row["step"]), []).append(float(row["test_accuracy"]))
            steps = sorted(grouped)
            ax.plot(steps, [float(np.mean(grouped[step])) for step in steps], label=variant)
        ax.set_xlabel("step")
        ax.set_ylabel("test accuracy")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "learnable_digit_curves.png", dpi=180)
        plt.close(fig)

    geom_rows = [row for row in rows if "omega" in row]
    if geom_rows:
        # Stored per-run geometry lives in JSON; rows only carry scalar stats. This plot is filled from summary in write_summary.
        pass


def plot_geometry(output_dir: Path, geometry_rows: list[dict[str, object]]) -> None:
    if not geometry_rows:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    for row in geometry_rows:
        omega = np.array(row["omega"], dtype=float)
        ax.scatter(omega[:, 0], omega[:, 1], label=str(row["variant"]), s=35)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("omega x")
    ax.set_ylabel("omega y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "learned_frequencies.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(theta), np.sin(theta), color="#cccccc")
    for row in geometry_rows:
        dirs = np.array(row["directions"], dtype=float)
        ax.scatter(dirs[:, 0], dirs[:, 1], label=str(row["variant"]), s=35)
    ax.set_aspect("equal")
    ax.set_xlabel("dir x")
    ax.set_ylabel("dir y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "learned_directions.png", dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], aggregate_rows: list[dict[str, object]]) -> None:
    lines = [
        "# V3-C Learnable PJ In Digits Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        f"- Steps: {summary['steps']}",
        f"- Seeds: {summary['seeds']}",
        f"- Depth / dim / heads: {summary['depth']} / {summary['dim']} / {summary['n_heads']}",
        f"- Wall seconds: {summary['wall_sec']:.2f}",
        "",
        "## Aggregate",
        "",
        "| variant | n | test acc mean | best | train acc | freq err | dir align | params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + f"{row['variant']} | {int(row['n'])} | {float(row['test_accuracy_mean']):.4f} | "
            + f"{float(row['test_accuracy_best']):.4f} | {float(row['train_accuracy_mean']):.4f} | "
            + f"{float(row['frequency_error_mean']):.4f} | {float(row['direction_alignment_mean']):.4f} | "
            + f"{int(row['param_count'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Fixed and learnable variants use the same classifier task and backbone size.",
            "- Learnable variants update spectral frequencies and jet directions through the attention bias.",
            "- Coordinate shuffle is included as a geometry control.",
            "",
            "Artifacts:",
            "",
            "- `learned_digit_results.csv`",
            "- `learned_digit_aggregate.csv`",
            "- `learned_digit_geometry.csv`",
            "- `learnable_digit_accuracy.png`",
            "- `learnable_digit_curves.png`",
            "- `learned_frequencies.png`",
            "- `learned_directions.png`",
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
    fixed_bases = {basis.name: basis for basis in build_bases(d, args.side, include_shuffle=True, seed=args.seed)}
    variants = parse_list(args.variants)
    rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    wall_start = time.time()
    for variant_idx, variant in enumerate(variants):
        for seed_idx in range(args.seeds):
            seed = args.seed + 100 * seed_idx + 1009 * variant_idx
            model = make_variant_model(variant, d, fixed_bases, args).to(device)
            row, variant_curves = train_classifier(
                model,
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                variant=variant,
                args=args,
                seed=seed,
            )
            rows.append(row)
            curves.extend(variant_curves)
            if hasattr(model, "geometry_stats"):
                stats = model.geometry_stats()
                geometry_rows.append({"variant": variant, "seed": seed, "omega": stats["omega"], "directions": stats["directions"]})
            aggregate_rows = aggregate(rows)
            write_csv(output_dir / "learned_digit_results.csv", rows)
            write_csv(output_dir / "learned_digit_aggregate.csv", aggregate_rows)
            write_csv(output_dir / "learned_digit_curves.csv", curves)
            write_csv(output_dir / "learned_digit_geometry.csv", geometry_rows)
            partial = {
                "device": str(device),
                "steps": args.steps,
                "seeds": args.seeds,
                "depth": args.depth,
                "dim": args.dim,
                "n_heads": args.n_heads,
                "variants": variants,
                "wall_sec": time.time() - wall_start,
                "rows": rows,
                "aggregate_rows": aggregate_rows,
                "geometry_rows": geometry_rows,
                "curves": curves,
                "status": "partial",
            }
            (output_dir / "learned_digit_summary.partial.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
    aggregate_rows = aggregate(rows)
    plot_outputs(output_dir, rows, curves)
    plot_geometry(output_dir, geometry_rows)
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() and device.type == "cuda" else 0
    summary = {
        "device": str(device),
        "steps": args.steps,
        "seeds": args.seeds,
        "depth": args.depth,
        "dim": args.dim,
        "n_heads": args.n_heads,
        "variants": variants,
        "wall_sec": time.time() - wall_start,
        "peak_cuda_memory_bytes": peak_mem,
        "rows": rows,
        "aggregate_rows": aggregate_rows,
        "geometry_rows": geometry_rows,
        "curves": curves,
    }
    summary_path = output_dir / "learned_digit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, summary, aggregate_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3-C learnable PJ inside sklearn digits Transformer.")
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=1400)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--variants",
        type=str,
        default="fixed_toric_PJ_R2,pruned_toric_PJ,relative_2d_table,learnable_near_PJ,learnable_random_PJ,learnable_curriculum_PJ,learnable_near_PJ_coord_shuffle",
    )
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-subset", type=int, default=1024)
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=381)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v3_digits_learnable_pj")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key not in {"rows", "aggregate_rows", "geometry_rows", "curves"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
