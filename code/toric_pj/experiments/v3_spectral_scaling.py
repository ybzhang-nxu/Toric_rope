from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from toric_pj.diagnostics.basis_projection import default_device, make_grid_2d, phase
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.experiments.learnable_spectral_geometry import target_value


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str
    omega: torch.Tensor
    direction: torch.Tensor


class BatchedSpectralPJ(nn.Module):
    """Batched scalar PJ kernel for multi-start geometry training."""

    def __init__(
        self,
        *,
        init_omega: torch.Tensor,
        init_dirs: torch.Tensor,
        max_order: int,
        scale: float,
        include_affine: bool = True,
        include_lc: bool = True,
    ) -> None:
        super().__init__()
        if init_omega.ndim != 2 or init_dirs.ndim != 2:
            raise ValueError("init_omega and init_dirs must be [starts, dims]")
        self.n_starts, self.n_dims = init_omega.shape
        self.max_order = int(max_order)
        self.scale = float(scale)
        self.include_affine = bool(include_affine)
        self.include_lc = bool(include_lc)

        dtype = init_omega.dtype
        device = init_omega.device
        self.omega = nn.Parameter(init_omega.clone())
        self.raw_dirs = nn.Parameter(init_dirs.clone())
        self.const = nn.Parameter(torch.zeros(self.n_starts, device=device, dtype=dtype))
        self.order0_cos = nn.Parameter(0.01 * torch.randn(self.n_starts, device=device, dtype=dtype))
        self.order0_sin = nn.Parameter(0.01 * torch.randn(self.n_starts, device=device, dtype=dtype))
        shape = (self.n_starts, self.max_order)
        self.jet_cos = nn.Parameter(0.01 * torch.randn(shape, device=device, dtype=dtype))
        self.jet_sin = nn.Parameter(0.01 * torch.randn(shape, device=device, dtype=dtype))
        self.lc_cos = nn.Parameter(0.01 * torch.randn(shape, device=device, dtype=dtype))
        self.lc_sin = nn.Parameter(0.01 * torch.randn(shape, device=device, dtype=dtype))
        self.affine_slope = nn.Parameter(torch.zeros(self.n_starts, self.n_dims, device=device, dtype=dtype))
        raw_lc = torch.log(torch.expm1(torch.tensor(self.scale, device=device, dtype=dtype).clamp_min(1e-6)))
        self.raw_lc_scale = nn.Parameter(raw_lc.expand(self.n_starts).clone())

        self.register_buffer("order_mask", torch.ones(self.max_order, device=device, dtype=dtype))

    @property
    def normalized_dirs(self) -> torch.Tensor:
        return F.normalize(self.raw_dirs, p=2, dim=-1, eps=1e-12)

    @property
    def lc_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_lc_scale) + 1e-3

    def set_order_limit(self, limit: int) -> None:
        limit = max(0, min(int(limit), self.max_order))
        mask = torch.zeros_like(self.order_mask)
        if limit > 0:
            mask[:limit] = 1.0
        self.order_mask.copy_(mask)

    def coefficient_parameters(self) -> list[nn.Parameter]:
        geometry = {id(self.omega), id(self.raw_dirs), id(self.raw_lc_scale)}
        return [param for param in self.parameters() if id(param) not in geometry]

    def geometry_parameters(self) -> list[nn.Parameter]:
        return [self.omega, self.raw_dirs, self.raw_lc_scale]

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        d = d.to(device=self.omega.device, dtype=self.omega.dtype)
        ph = torch.einsum("nd,sd->sn", d, self.omega)
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)
        out = self.const[:, None] + self.order0_cos[:, None] * cos_ph + self.order0_sin[:, None] * sin_ph

        dirs = self.normalized_dirs
        coord = torch.einsum("nd,sd->sn", d, dirs) / max(self.scale, 1e-12)
        raw = torch.einsum("nd,sd->sn", d, dirs)
        lc_scale = self.lc_scale[:, None].clamp_min(1e-12)
        beta = raw / torch.sqrt(raw.square() + lc_scale.square())
        phi = lc_scale * torch.asinh(raw / lc_scale)
        omega_mag = torch.linalg.norm(self.omega, dim=-1, keepdim=True)
        lc_phase = omega_mag * phi
        lc_cos_phase = torch.cos(lc_phase)
        lc_sin_phase = torch.sin(lc_phase)

        for order in range(1, self.max_order + 1):
            mask = self.order_mask[order - 1]
            poly = coord.pow(order)
            out = out + mask * self.jet_cos[:, order - 1 : order] * poly * cos_ph
            out = out + mask * self.jet_sin[:, order - 1 : order] * poly * sin_ph
            if self.include_lc:
                lc_poly = beta.pow(order)
                out = out + mask * self.lc_cos[:, order - 1 : order] * lc_poly * lc_cos_phase
                out = out + mask * self.lc_sin[:, order - 1 : order] * lc_poly * lc_sin_phase
        if self.include_affine:
            out = out - torch.einsum("nd,sd->sn", d / max(self.scale, 1e-12), self.affine_slope)
        return out


def target_specs(device: torch.device, dtype: torch.dtype) -> list[TargetSpec]:
    omega = torch.tensor([0.57, -0.38], device=device, dtype=dtype)
    direction = normalize_direction(torch.tensor([[1.0, -0.55]], device=device, dtype=dtype)).reshape(-1)
    return [
        TargetSpec("V3A_fourier", "fourier", omega, direction),
        TargetSpec("V3A_first_jet", "jet1", omega, direction),
        TargetSpec("V3A_second_jet", "jet2", omega, direction),
        TargetSpec("V3A_mixed_FJ_affine_LC", "mixed", omega, direction),
    ]


def v2_target_proxy(spec: TargetSpec) -> object:
    return argparse.Namespace(name=spec.name, kind=spec.kind, omega=spec.omega, direction=spec.direction)


def direction_alignment(direction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direction = F.normalize(direction, p=2, dim=-1, eps=1e-12)
    target = F.normalize(target.reshape(1, -1).to(device=direction.device, dtype=direction.dtype), p=2, dim=-1, eps=1e-12)
    return torch.abs(torch.sum(direction * target, dim=-1))


def frequency_error(omega: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.reshape(1, -1).to(device=omega.device, dtype=omega.dtype)
    return torch.minimum(torch.linalg.norm(omega - target, dim=-1), torch.linalg.norm(omega + target, dim=-1))


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.reshape(1, -1).to(device=pred.device, dtype=pred.dtype)
    mse = torch.mean((pred - target).square(), dim=-1)
    var = torch.mean((target - target.mean(dim=-1, keepdim=True)).square(), dim=-1).clamp_min(1e-30)
    return 1.0 - mse / var


def make_initializers(
    *,
    spec: TargetSpec,
    variant: str,
    n_restarts: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = spec.omega.device
    dtype = spec.omega.dtype
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    near_omega = spec.omega + torch.tensor([0.045, 0.035], device=device, dtype=dtype)
    near_dir = normalize_direction((spec.direction + torch.tensor([0.18, 0.10], device=device, dtype=dtype)).reshape(1, -1)).reshape(-1)
    if variant.startswith("near"):
        omega = near_omega.reshape(1, -1).repeat(n_restarts, 1)
        direction = near_dir.reshape(1, -1).repeat(n_restarts, 1)
        if n_restarts > 1:
            omega = omega + 0.015 * torch.randn(n_restarts, 2, device=device, dtype=dtype, generator=gen)
            direction = direction + 0.025 * torch.randn(n_restarts, 2, device=device, dtype=dtype, generator=gen)
        return omega, normalize_direction(direction)
    if variant.startswith("random_near_bank_mix"):
        half = max(1, n_restarts // 2)
        rand_omega = spec.omega.reshape(1, -1) + 0.32 * torch.randn(n_restarts - half, 2, device=device, dtype=dtype, generator=gen)
        rand_dir = normalize_direction(torch.randn(n_restarts - half, 2, device=device, dtype=dtype, generator=gen))
        near_omega_batch = near_omega.reshape(1, -1).repeat(half, 1)
        near_dir_batch = near_dir.reshape(1, -1).repeat(half, 1)
        omega = torch.cat([near_omega_batch, rand_omega], dim=0)
        direction = torch.cat([near_dir_batch, rand_dir], dim=0)
        return omega, normalize_direction(direction)
    omega = spec.omega.reshape(1, -1) + 0.32 * torch.randn(n_restarts, 2, device=device, dtype=dtype, generator=gen)
    direction = normalize_direction(torch.randn(n_restarts, 2, device=device, dtype=dtype, generator=gen))
    return omega, direction


def train_variant(
    *,
    spec: TargetSpec,
    train_grid: torch.Tensor,
    eval_grid: torch.Tensor,
    target_train: torch.Tensor,
    target_eval: torch.Tensor,
    variant: str,
    n_restarts: int,
    steps: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    init_omega, init_dirs = make_initializers(spec=spec, variant=variant, n_restarts=n_restarts, seed=seed)
    model = BatchedSpectralPJ(
        init_omega=init_omega,
        init_dirs=init_dirs,
        max_order=args.max_order,
        scale=float(args.train_radius),
        include_affine=True,
        include_lc=True,
    )
    coeff_params = model.coefficient_parameters()
    param_groups = [{"params": coeff_params, "lr": args.lr_coeff}]
    geom_params = model.geometry_parameters()
    param_groups.append({"params": geom_params, "lr": args.lr_geom})
    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
    history: list[dict[str, object]] = []
    best_eval = torch.full((n_restarts,), -1e9, device=train_grid.device, dtype=train_grid.dtype)
    best_step = torch.full((n_restarts,), -1, device=train_grid.device, dtype=torch.long)
    wall_start = time.time()

    for step in range(int(steps)):
        if variant.endswith("curriculum") or "curriculum" in variant:
            frac = step / max(1, steps - 1)
            if frac < 0.34:
                model.set_order_limit(0)
            elif frac < 0.67:
                model.set_order_limit(1)
            else:
                model.set_order_limit(args.max_order)
        else:
            model.set_order_limit(args.max_order)
        opt.zero_grad(set_to_none=True)
        pred = model(train_grid)
        target = target_train.reshape(1, -1)
        per_start_mse = torch.mean((pred - target).square(), dim=-1)
        coeff_penalty = torch.zeros((), device=train_grid.device, dtype=train_grid.dtype)
        for param in coeff_params:
            coeff_penalty = coeff_penalty + param.square().mean()
        lc_penalty = (torch.log(model.lc_scale) - math.log(float(args.train_radius))).square().mean()
        omega_norm = torch.linalg.norm(model.omega, dim=-1)
        clip_penalty = torch.relu(omega_norm - args.freq_clip).square().mean()
        repulsion = torch.zeros((), device=train_grid.device, dtype=train_grid.dtype)
        if args.repulsion > 0 and n_restarts > 1:
            dist = torch.cdist(model.omega, model.omega)
            eye = torch.eye(n_restarts, device=dist.device, dtype=torch.bool)
            repulsion = torch.exp(-dist.masked_fill(eye, 20.0).square() / 0.02).mean()
        loss = per_start_mse.mean() + args.coeff_l2 * coeff_penalty + args.lc_scale_reg * lc_penalty
        loss = loss + args.freq_clip_penalty * clip_penalty + args.repulsion * repulsion
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()
        with torch.no_grad():
            norm = torch.linalg.norm(model.omega, dim=-1, keepdim=True).clamp_min(1e-12)
            scale = torch.clamp(norm, max=args.freq_clip) / norm
            model.omega.mul_(scale)
            model.raw_dirs.copy_(normalize_direction(model.raw_dirs))

        if step % args.log_every == 0 or step == steps - 1:
            with torch.no_grad():
                eval_r2 = r2_score(model(eval_grid), target_eval)
                train_r2 = r2_score(model(train_grid), target_train)
                improved = eval_r2 > best_eval
                best_eval = torch.where(improved, eval_r2, best_eval)
                best_step = torch.where(improved, torch.full_like(best_step, step), best_step)
                history.append(
                    {
                        "target": spec.name,
                        "variant": variant,
                        "step": step,
                        "eval_r2_mean": float(eval_r2.mean().detach().cpu()),
                        "eval_r2_median": float(eval_r2.median().detach().cpu()),
                        "eval_r2_best": float(eval_r2.max().detach().cpu()),
                        "train_r2_mean": float(train_r2.mean().detach().cpu()),
                        "elapsed_sec": time.time() - wall_start,
                    }
                )

    with torch.no_grad():
        final_eval = r2_score(model(eval_grid), target_eval)
        final_train = r2_score(model(train_grid), target_train)
        freq_err = frequency_error(model.omega, spec.omega)
        align = direction_alignment(model.normalized_dirs, spec.direction)
        rows = []
        for idx in range(n_restarts):
            rows.append(
                {
                    "target": spec.name,
                    "kind": spec.kind,
                    "variant": variant,
                    "restart": idx,
                    "seed": seed,
                    "steps": steps,
                    "train_radius": args.train_radius,
                    "eval_radius": args.eval_radius,
                    "eval_r2": float(final_eval[idx].detach().cpu()),
                    "train_r2": float(final_train[idx].detach().cpu()),
                    "best_eval_r2": float(best_eval[idx].detach().cpu()),
                    "best_step": int(best_step[idx].detach().cpu()),
                    "frequency_error": float(freq_err[idx].detach().cpu()),
                    "direction_alignment": float(align[idx].detach().cpu()),
                    "lc_scale": float(model.lc_scale[idx].detach().cpu()),
                    "omega_x": float(model.omega[idx, 0].detach().cpu()),
                    "omega_y": float(model.omega[idx, 1].detach().cpu()),
                    "dir_x": float(model.normalized_dirs[idx, 0].detach().cpu()),
                    "dir_y": float(model.normalized_dirs[idx, 1].detach().cpu()),
                    "num_features": 1 + 2 + 2 * args.max_order + 2 + 2 * args.max_order,
                    "wall_sec": time.time() - wall_start,
                }
            )
    return rows, history


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["target"]), str(row["variant"])), []).append(row)
    out = []
    for (target, variant), values in sorted(groups.items()):
        evals = np.array([float(row["eval_r2"]) for row in values])
        best = np.array([float(row["best_eval_r2"]) for row in values])
        freq = np.array([float(row["frequency_error"]) for row in values])
        align = np.array([float(row["direction_alignment"]) for row in values])
        out.append(
            {
                "target": target,
                "variant": variant,
                "n": len(values),
                "eval_r2_mean": float(evals.mean()),
                "eval_r2_median": float(np.median(evals)),
                "eval_r2_best": float(evals.max()),
                "best_eval_r2_best": float(best.max()),
                "success_095": float(np.mean(evals >= 0.95)),
                "frequency_error_median": float(np.median(freq)),
                "direction_alignment_median": float(np.median(align)),
                "wall_sec_max": float(max(float(row["wall_sec"]) for row in values)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(output_dir: Path, aggregate_rows: list[dict[str, object]], history: list[dict[str, object]]) -> None:
    targets = sorted({str(row["target"]) for row in aggregate_rows})
    variants = sorted({str(row["variant"]) for row in aggregate_rows})
    fig, axes = plt.subplots(max(1, len(targets)), 1, figsize=(12, max(4, 3.4 * len(targets))), squeeze=False)
    for ax, target in zip(axes.reshape(-1), targets):
        vals = [row for row in aggregate_rows if row["target"] == target]
        vals = sorted(vals, key=lambda row: variants.index(str(row["variant"])))
        x = np.arange(len(vals))
        y = [float(row["eval_r2_median"]) for row in vals]
        ax.bar(x, y, color="#4d6f8f")
        ax.set_xticks(x, [str(row["variant"]) for row in vals], rotation=25, ha="right")
        ax.set_ylim(min(-0.2, min(y) - 0.05), 1.05)
        ax.set_title(target)
        ax.set_ylabel("median eval R2")
    fig.tight_layout()
    fig.savefig(output_dir / "spectral_scaling_eval_r2.png", dpi=180)
    plt.close(fig)

    if history:
        fig, ax = plt.subplots(figsize=(11, 5))
        for key in sorted({(row["target"], row["variant"]) for row in history}):
            vals = [row for row in history if (row["target"], row["variant"]) == key]
            vals = sorted(vals, key=lambda row: int(row["step"]))
            ax.plot([int(row["step"]) for row in vals], [float(row["eval_r2_best"]) for row in vals], label=f"{key[0]} {key[1]}")
        ax.set_xlabel("step")
        ax.set_ylabel("best eval R2")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "spectral_training_curves.png", dpi=180)
        plt.close(fig)

    learned = [row for row in aggregate_rows if "random" in str(row["variant"]) or "near" in str(row["variant"])]
    if learned:
        fig, ax1 = plt.subplots(figsize=(12, 4.8))
        labels = [f"{row['target']}\n{row['variant']}" for row in learned]
        x = np.arange(len(learned))
        ax1.bar(x - 0.18, [float(row["frequency_error_median"]) for row in learned], width=0.36, color="#8a6547")
        ax1.set_ylabel("median frequency error")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.18, [float(row["direction_alignment_median"]) for row in learned], width=0.36, color="#55795c")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("median direction alignment")
        ax1.set_xticks(x, labels, rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(output_dir / "geometry_recovery_scatter.png", dpi=180)
        plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], aggregate_rows: list[dict[str, object]]) -> None:
    lines = [
        "# V3-A Spectral Geometry Scaling Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        f"- Mode: {summary['mode']}",
        f"- Train radius: {summary['train_radius']}",
        f"- Eval radius: {summary['eval_radius']}",
        f"- Steps: {summary['steps']}",
        f"- Restarts: {summary['n_restarts']}",
        f"- Wall seconds: {summary['wall_sec']:.2f}",
        "",
        "## Aggregate",
        "",
        "| target | variant | n | median eval R2 | best eval R2 | success >=0.95 | median freq err | median dir align |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + f"{row['target']} | {row['variant']} | {int(row['n'])} | "
            + f"{float(row['eval_r2_median']):.4f} | {float(row['eval_r2_best']):.4f} | "
            + f"{float(row['success_095']):.3f} | {float(row['frequency_error_median']):.4f} | "
            + f"{float(row['direction_alignment_median']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation rules:",
            "",
            "- `success >=0.95` is the fraction of restarts with final eval R2 at least 0.95.",
            "- Best-of-many success with low median indicates a multi-modal optimization landscape.",
            "- High R2 with poor frequency/direction recovery indicates geometry non-identifiability in the train window.",
            "",
            "Artifacts:",
            "",
            "- `spectral_scaling_results.csv`",
            "- `spectral_scaling_aggregate.csv`",
            "- `spectral_scaling_history.csv`",
            "- `spectral_scaling_eval_r2.png`",
            "- `spectral_training_curves.png`",
            "- `geometry_recovery_scatter.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_specs(specs: list[TargetSpec], args: argparse.Namespace) -> list[TargetSpec]:
    if args.mode == "smoke":
        return [spec for spec in specs if spec.kind in {"jet2", "mixed"}]
    if args.targets == "all":
        return specs
    wanted = {item.strip() for item in args.targets.split(",") if item.strip()}
    return [spec for spec in specs if spec.kind in wanted or spec.name in wanted]


def selected_variants(args: argparse.Namespace) -> list[str]:
    if args.mode == "smoke":
        return ["near_long", "near_curriculum", "random_restart", "random_curriculum"]
    if args.variants == "all":
        return ["near_long", "near_curriculum", "random_restart", "random_curriculum", "random_near_bank_mix"]
    return [item.strip() for item in args.variants.split(",") if item.strip()]


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_grid = make_grid_2d(args.train_radius, signed=True, device=device, dtype=dtype)
    eval_grid = make_grid_2d(args.eval_radius, signed=True, device=device, dtype=dtype)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.time()
    rows: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    for spec in selected_specs(target_specs(device, dtype), args):
        target_train = target_value(v2_target_proxy(spec), train_grid, scale=float(args.train_radius))
        target_eval = target_value(v2_target_proxy(spec), eval_grid, scale=float(args.train_radius))
        for variant_idx, variant in enumerate(selected_variants(args)):
            variant_rows, variant_history = train_variant(
                spec=spec,
                train_grid=train_grid,
                eval_grid=eval_grid,
                target_train=target_train,
                target_eval=target_eval,
                variant=variant,
                n_restarts=args.n_restarts,
                steps=args.steps,
                seed=args.seed + 1009 * variant_idx,
                args=args,
            )
            rows.extend(variant_rows)
            history.extend(variant_history)
            aggregate_rows = aggregate(rows)
            write_csv(output_dir / "spectral_scaling_results.csv", rows)
            write_csv(output_dir / "spectral_scaling_aggregate.csv", aggregate_rows)
            write_csv(output_dir / "spectral_scaling_history.csv", history)
            partial_summary = {
                "device": str(device),
                "mode": args.mode,
                "train_radius": args.train_radius,
                "eval_radius": args.eval_radius,
                "steps": args.steps,
                "n_restarts": args.n_restarts,
                "seed": args.seed,
                "wall_sec": time.time() - wall_start,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
                if torch.cuda.is_available() and device.type == "cuda"
                else 0,
                "rows": rows,
                "aggregate_rows": aggregate_rows,
                "history": history,
                "status": "partial",
            }
            (output_dir / "spectral_scaling_summary.partial.json").write_text(
                json.dumps(partial_summary, indent=2), encoding="utf-8"
            )
    aggregate_rows = aggregate(rows)
    write_csv(output_dir / "spectral_scaling_results.csv", rows)
    write_csv(output_dir / "spectral_scaling_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "spectral_scaling_history.csv", history)
    plot_outputs(output_dir, aggregate_rows, history)
    peak_mem = 0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated(device))
    summary = {
        "device": str(device),
        "mode": args.mode,
        "dtype": args.dtype,
        "train_radius": args.train_radius,
        "eval_radius": args.eval_radius,
        "steps": args.steps,
        "n_restarts": args.n_restarts,
        "seed": args.seed,
        "wall_sec": time.time() - wall_start,
        "peak_cuda_memory_bytes": peak_mem,
        "rows": rows,
        "aggregate_rows": aggregate_rows,
        "history": history,
    }
    summary_path = output_dir / "spectral_scaling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, summary, aggregate_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3-A spectral geometry scaling.")
    parser.add_argument("--mode", choices=["smoke", "main", "overnight"], default="smoke")
    parser.add_argument("--targets", type=str, default="all")
    parser.add_argument("--variants", type=str, default="all")
    parser.add_argument("--train-radius", type=int, default=12)
    parser.add_argument("--eval-radius", type=int, default=24)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--n-restarts", type=int, default=None)
    parser.add_argument("--lr-coeff", type=float, default=0.035)
    parser.add_argument("--lr-geom", type=float, default=0.014)
    parser.add_argument("--coeff-l2", type=float, default=1e-7)
    parser.add_argument("--lc-scale-reg", type=float, default=1e-5)
    parser.add_argument("--freq-clip", type=float, default=1.6)
    parser.add_argument("--freq-clip-penalty", type=float, default=1e-3)
    parser.add_argument("--repulsion", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v3_spectral_scaling")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = {"smoke": 650, "main": 20000, "overnight": 50000}[args.mode]
    if args.n_restarts is None:
        args.n_restarts = {"smoke": 16, "main": 128, "overnight": 256}[args.mode]
    if args.log_every is None:
        args.log_every = max(25, args.steps // 80)
    if args.mode != "smoke" and args.repulsion == 0.0:
        args.repulsion = 1e-4
    return args


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key not in {"rows", "aggregate_rows", "history"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
