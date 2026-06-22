from __future__ import annotations

import argparse
import csv
import json
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
    condition_number,
    default_device,
    make_grid_2d,
    normalize_columns,
    phase,
)
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.models.learnable_toric_pj_bias import LearnableSpectralPJConfig, LearnableSpectralPJKernel


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str
    omega: torch.Tensor
    direction: torch.Tensor


def target_value(spec: TargetSpec, d: torch.Tensor, *, scale: float) -> torch.Tensor:
    omega = spec.omega.to(device=d.device, dtype=d.dtype)
    direction = normalize_direction(spec.direction.to(device=d.device, dtype=d.dtype).reshape(1, -1)).reshape(-1)
    coord = (d @ direction) / float(scale)
    ph = phase(d, omega)
    if spec.kind == "fourier":
        return torch.cos(ph)
    if spec.kind == "jet1":
        return coord * torch.cos(ph)
    if spec.kind == "jet2":
        return coord.square() * torch.cos(ph)
    if spec.kind == "mixed":
        raw = d @ direction
        lc_scale = float(scale)
        beta = raw / torch.sqrt(raw.square() + lc_scale**2)
        phi = lc_scale * torch.asinh(raw / lc_scale)
        affine_dir = normalize_direction(torch.tensor([[0.72, 0.38]], device=d.device, dtype=d.dtype)).reshape(-1)
        affine = -(d @ affine_dir) / float(scale)
        return (
            0.50 * torch.cos(ph)
            + 0.24 * coord * torch.sin(ph)
            + 0.16 * coord.square() * torch.cos(ph)
            + 0.20 * affine
            + 0.18 * beta.square() * torch.cos(torch.linalg.norm(omega) * phi)
        )
    raise ValueError(f"unknown target kind: {spec.kind}")


def dictionary_basis(
    d: torch.Tensor,
    *,
    omegas: torch.Tensor,
    directions: torch.Tensor,
    max_order: int,
    scale: float,
    include_affine: bool,
    include_lc: bool,
    name: str,
) -> Basis:
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    dirs = [normalize_direction(direction.reshape(1, -1)).reshape(-1) for direction in directions]
    for freq_idx, omega in enumerate(omegas):
        ph = phase(d, omega)
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)
        cols.extend([cos_ph, sin_ph])
        labels.extend([f"w{freq_idx}_r0_cos", f"w{freq_idx}_r0_sin"])
        orders.extend([0, 0])
        for dir_idx, direction in enumerate(dirs):
            coord = (d @ direction) / float(scale)
            for order in range(1, max_order + 1):
                poly = coord.pow(order)
                cols.extend([poly * cos_ph, poly * sin_ph])
                labels.extend([f"w{freq_idx}_u{dir_idx}_r{order}_cos", f"w{freq_idx}_u{dir_idx}_r{order}_sin"])
                orders.extend([order, order])
    if include_affine:
        for axis in range(d.shape[1]):
            cols.append(-d[:, axis] / float(scale))
            labels.append(f"affine_axis{axis}")
            orders.append(1)
    if include_lc:
        for freq_idx, omega in enumerate(omegas):
            omega_mag = torch.linalg.norm(omega)
            for dir_idx, direction in enumerate(dirs):
                raw = d @ direction
                beta = raw / torch.sqrt(raw.square() + float(scale) ** 2)
                phi = float(scale) * torch.asinh(raw / float(scale))
                ph = omega_mag * phi
                for order in range(1, max_order + 1):
                    poly = beta.pow(order)
                    cols.extend([poly * torch.cos(ph), poly * torch.sin(ph)])
                    labels.extend([f"lc_w{freq_idx}_u{dir_idx}_r{order}_cos", f"lc_w{freq_idx}_u{dir_idx}_r{order}_sin"])
                    orders.extend([order, order])
    return Basis(name=name, matrix=torch.stack(cols, dim=1), labels=labels, orders=orders)


def fit_fixed_train_eval(
    *,
    basis_train: Basis,
    basis_eval: Basis,
    target_train: torch.Tensor,
    target_eval: torch.Tensor,
    ridge: float,
) -> dict[str, float]:
    x_train, norms = normalize_columns(basis_train.matrix)
    x_eval = basis_eval.matrix / norms.clamp_min(1e-12)
    y_train = target_train.reshape(-1, 1)
    y_eval = target_eval.reshape(-1, 1)
    gram = x_train.T @ x_train
    rhs = x_train.T @ y_train
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coeff = torch.linalg.solve(gram + ridge * eye, rhs)
    pred_train = x_train @ coeff
    pred_eval = x_eval @ coeff
    train_mse = torch.mean((pred_train - y_train).square())
    eval_mse = torch.mean((pred_eval - y_eval).square())
    train_var = torch.mean((y_train - y_train.mean()).square()).clamp_min(1e-30)
    eval_var = torch.mean((y_eval - y_eval.mean()).square()).clamp_min(1e-30)
    return {
        "train_mse": float(train_mse.detach().cpu()),
        "eval_mse": float(eval_mse.detach().cpu()),
        "train_r2": float((1.0 - train_mse / train_var).detach().cpu()),
        "eval_r2": float((1.0 - eval_mse / eval_var).detach().cpu()),
        "condition": condition_number(x_train),
        "coeff_norm": float(torch.linalg.norm(coeff).detach().cpu()),
    }


def relative_table_row(
    spec: TargetSpec,
    train: torch.Tensor,
    eval_grid: torch.Tensor,
    target_train: torch.Tensor,
    target_eval: torch.Tensor,
    *,
    ridge: float,
) -> dict[str, object]:
    basis_train = relative_table_basis(train, name="relative_table")
    # Relative table is intentionally a train-window table; unseen eval displacements map to zero columns.
    train_keys = {tuple(int(v) for v in row): idx for idx, row in enumerate(train.to(torch.long).detach().cpu().tolist())}
    matrix_eval = torch.zeros((eval_grid.shape[0], basis_train.matrix.shape[1]), device=eval_grid.device, dtype=eval_grid.dtype)
    for row_idx, row in enumerate(eval_grid.to(torch.long).detach().cpu().tolist()):
        col = train_keys.get(tuple(int(v) for v in row))
        if col is not None:
            matrix_eval[row_idx, col] = 1.0
    basis_eval = Basis("relative_table", matrix_eval, basis_train.labels, basis_train.orders)
    metrics = fit_fixed_train_eval(
        basis_train=basis_train,
        basis_eval=basis_eval,
        target_train=target_train,
        target_eval=target_eval,
        ridge=ridge,
    )
    return {
        "target": spec.name,
        "variant": "relative_table_train_window",
        "seed": -1,
        "learn_omega": False,
        "learn_dirs": False,
        "num_features": basis_train.matrix.shape[1],
        "frequency_error": 0.0,
        "direction_alignment": 1.0,
        "steps_to_95_oracle": -1,
        **metrics,
    }


def relative_table_basis(d: torch.Tensor, *, name: str) -> Basis:
    unique, inverse = torch.unique(d.to(torch.long), dim=0, return_inverse=True)
    matrix = torch.zeros((d.shape[0], unique.shape[0]), device=d.device, dtype=d.dtype)
    matrix[torch.arange(d.shape[0], device=d.device), inverse] = 1.0
    labels = [f"rel_{int(x)}_{int(y)}" for x, y in unique.detach().cpu().tolist()]
    return Basis(name, matrix, labels, [0] * len(labels))


def fixed_variant_row(
    spec: TargetSpec,
    train: torch.Tensor,
    eval_grid: torch.Tensor,
    target_train: torch.Tensor,
    target_eval: torch.Tensor,
    *,
    variant: str,
    init_omega: torch.Tensor,
    init_dir: torch.Tensor,
    max_order: int,
    scale: float,
    ridge: float,
) -> dict[str, object]:
    omega = init_omega.reshape(1, -1)
    direction = init_dir.reshape(1, -1)
    basis_train = dictionary_basis(
        train,
        omegas=omega,
        directions=direction,
        max_order=max_order,
        scale=scale,
        include_affine=True,
        include_lc=True,
        name=variant,
    )
    basis_eval = dictionary_basis(
        eval_grid,
        omegas=omega,
        directions=direction,
        max_order=max_order,
        scale=scale,
        include_affine=True,
        include_lc=True,
        name=variant,
    )
    metrics = fit_fixed_train_eval(
        basis_train=basis_train,
        basis_eval=basis_eval,
        target_train=target_train,
        target_eval=target_eval,
        ridge=ridge,
    )
    return {
        "target": spec.name,
        "variant": variant,
        "seed": -1,
        "learn_omega": False,
        "learn_dirs": False,
        "num_features": basis_train.matrix.shape[1],
        "frequency_error": frequency_error(init_omega, spec.omega),
        "direction_alignment": direction_alignment(init_dir, spec.direction),
        "steps_to_95_oracle": -1,
        **metrics,
    }


def train_learned_variant(
    spec: TargetSpec,
    train: torch.Tensor,
    eval_grid: torch.Tensor,
    target_train: torch.Tensor,
    target_eval: torch.Tensor,
    *,
    variant: str,
    init_omega: torch.Tensor,
    init_dir: torch.Tensor,
    max_order: int,
    scale: float,
    steps: int,
    lr_coeff: float,
    lr_geom: float,
    coeff_l2: float,
    lc_scale_reg: float,
    seed: int,
    oracle_eval_r2: float,
) -> tuple[dict[str, object], list[float]]:
    torch.manual_seed(seed)
    config = LearnableSpectralPJConfig(
        n_dims=train.shape[1],
        n_freqs=1,
        n_dirs=1,
        max_order=max_order,
        scale=scale,
        include_affine=True,
        include_lc=True,
        learn_omega=True,
        learn_dirs=True,
        learn_lc_scale=True,
    )
    model = LearnableSpectralPJKernel(
        config,
        init_omega=init_omega.reshape(1, -1),
        init_dirs=init_dir.reshape(1, -1),
        dtype=train.dtype,
        device=train.device,
    )
    coeff_params = model.coefficient_parameters()
    param_groups = [{"params": coeff_params, "lr": lr_coeff}]
    geometry_params = model.geometry_parameters()
    if geometry_params:
        param_groups.append({"params": geometry_params, "lr": lr_geom})
    opt = torch.optim.AdamW(param_groups, weight_decay=0.0)
    history = []
    steps_to_95 = -1
    threshold = max(0.0, 0.95 * oracle_eval_r2)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(train)
        mse = torch.mean((pred - target_train).square())
        coeff_penalty = sum(param.square().mean() for param in coeff_params)
        lc_penalty = (torch.log(model.lc_scale) - np.log(scale)).square()
        loss = mse + coeff_l2 * coeff_penalty + lc_scale_reg * lc_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()
        if step % max(1, steps // 100) == 0 or step == steps - 1:
            with torch.no_grad():
                metrics = evaluate_prediction(model(train), target_train, model(eval_grid), target_eval)
                eval_r2 = float(metrics["eval_r2"])
                history.append(eval_r2)
                if steps_to_95 < 0 and eval_r2 >= threshold:
                    steps_to_95 = step
    with torch.no_grad():
        learned_omega = model.omega.detach().reshape(1, -1)
        learned_dir = model.normalized_dirs.detach().reshape(1, -1)
        basis_train = dictionary_basis(
            train,
            omegas=learned_omega,
            directions=learned_dir,
            max_order=max_order,
            scale=scale,
            include_affine=True,
            include_lc=True,
            name=variant,
        )
        basis_eval = dictionary_basis(
            eval_grid,
            omegas=learned_omega,
            directions=learned_dir,
            max_order=max_order,
            scale=scale,
            include_affine=True,
            include_lc=True,
            name=variant,
        )
        metrics = fit_fixed_train_eval(
            basis_train=basis_train,
            basis_eval=basis_eval,
            target_train=target_train,
            target_eval=target_eval,
            ridge=1e-9,
        )
        row = {
            "target": spec.name,
            "variant": variant,
            "seed": seed,
            "learn_omega": True,
            "learn_dirs": True,
            "num_features": basis_train.matrix.shape[1],
            "frequency_error": frequency_error(learned_omega.reshape(-1), spec.omega),
            "direction_alignment": direction_alignment(learned_dir.reshape(-1), spec.direction),
            "steps_to_95_oracle": steps_to_95,
            **metrics,
        }
    return row, history


def evaluate_prediction(
    pred_train: torch.Tensor,
    target_train: torch.Tensor,
    pred_eval: torch.Tensor,
    target_eval: torch.Tensor,
) -> dict[str, float]:
    train_mse = torch.mean((pred_train - target_train).square())
    eval_mse = torch.mean((pred_eval - target_eval).square())
    train_var = torch.mean((target_train - target_train.mean()).square()).clamp_min(1e-30)
    eval_var = torch.mean((target_eval - target_eval.mean()).square()).clamp_min(1e-30)
    return {
        "train_mse": float(train_mse.detach().cpu()),
        "eval_mse": float(eval_mse.detach().cpu()),
        "train_r2": float((1.0 - train_mse / train_var).detach().cpu()),
        "eval_r2": float((1.0 - eval_mse / eval_var).detach().cpu()),
    }


def model_num_features(*, max_order: int) -> int:
    # const + order0 cos/sin + jet cos/sin for orders + affine bias/slopes + LC cos/sin for orders.
    return 1 + 2 + 2 * max_order + 2 + 2 * max_order


def frequency_error(omega: torch.Tensor, target: torch.Tensor) -> float:
    omega = omega.to(device=target.device, dtype=target.dtype).reshape(-1)
    target = target.reshape(-1)
    err = torch.minimum(torch.linalg.norm(omega - target), torch.linalg.norm(omega + target))
    return float(err.detach().cpu())


def direction_alignment(direction: torch.Tensor, target: torch.Tensor) -> float:
    direction = normalize_direction(direction.to(device=target.device, dtype=target.dtype).reshape(1, -1)).reshape(-1)
    target = normalize_direction(target.reshape(1, -1)).reshape(-1)
    return float(torch.abs(torch.sum(direction * target)).detach().cpu())


def target_specs(device: torch.device) -> list[TargetSpec]:
    dtype = torch.float64
    omega = torch.tensor([0.57, -0.38], device=device, dtype=dtype)
    direction = normalize_direction(torch.tensor([[1.0, -0.55]], device=device, dtype=dtype)).reshape(-1)
    return [
        TargetSpec("V2A_fourier", "fourier", omega, direction),
        TargetSpec("V2A_first_jet", "jet1", omega, direction),
        TargetSpec("V2A_second_jet", "jet2", omega, direction),
        TargetSpec("V2A_mixed_FJ_affine_LC", "mixed", omega, direction),
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = make_grid_2d(args.train_radius, signed=True, device=device, dtype=torch.float64)
    eval_grid = make_grid_2d(args.eval_radius, signed=True, device=device, dtype=torch.float64)
    rows: list[dict[str, object]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    for spec in target_specs(device):
        target_train = target_value(spec, train, scale=float(args.train_radius))
        target_eval = target_value(spec, eval_grid, scale=float(args.train_radius))
        misaligned_omega = spec.omega + torch.tensor([0.17, -0.12], device=device, dtype=torch.float64)
        near_omega = spec.omega + torch.tensor([0.045, 0.035], device=device, dtype=torch.float64)
        misaligned_dir = normalize_direction(torch.tensor([[0.25, 1.0]], device=device, dtype=torch.float64)).reshape(-1)
        near_dir = normalize_direction((spec.direction + torch.tensor([0.18, 0.10], device=device, dtype=torch.float64)).reshape(1, -1)).reshape(-1)

        fixed = [
            ("frozen_misaligned", misaligned_omega, misaligned_dir),
            ("frozen_near_bank", near_omega, near_dir),
            ("oracle", spec.omega, spec.direction),
        ]
        oracle_eval_r2 = 1.0
        for variant, omega, direction in fixed:
            row = fixed_variant_row(
                spec,
                train,
                eval_grid,
                target_train,
                target_eval,
                variant=variant,
                init_omega=omega,
                init_dir=direction,
                max_order=args.max_order,
                scale=float(args.train_radius),
                ridge=args.ridge,
            )
            rows.append(row)
            if variant == "oracle":
                oracle_eval_r2 = max(0.0, float(row["eval_r2"]))
        rows.append(relative_table_row(spec, train, eval_grid, target_train, target_eval, ridge=args.ridge))

        histories[spec.name] = {}
        for variant in ["learned_random_init", "learned_near_init"]:
            for seed in args.seeds:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed + len(rows) * 17)
                if variant == "learned_near_init":
                    init_omega = near_omega
                    init_dir = near_dir
                else:
                    init_omega = spec.omega + 0.32 * torch.randn(2, device=device, dtype=torch.float64, generator=gen)
                    init_dir = normalize_direction(torch.randn(1, 2, device=device, dtype=torch.float64, generator=gen)).reshape(-1)
                row, history = train_learned_variant(
                    spec,
                    train,
                    eval_grid,
                    target_train,
                    target_eval,
                    variant=variant,
                    init_omega=init_omega,
                    init_dir=init_dir,
                    max_order=args.max_order,
                    scale=float(args.train_radius),
                    steps=args.steps,
                    lr_coeff=args.lr_coeff,
                    lr_geom=args.lr_geom,
                    coeff_l2=args.coeff_l2,
                    lc_scale_reg=args.lc_scale_reg,
                    seed=seed,
                    oracle_eval_r2=oracle_eval_r2,
                )
                rows.append(row)
                histories[spec.name][f"{variant}_{seed}"] = history

    csv_path = output_dir / "learnable_spectral_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = aggregate(rows)
    aggregate_csv_path = output_dir / "learnable_spectral_aggregate.csv"
    with aggregate_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    r2_plot = output_dir / "learnable_spectral_eval_r2.png"
    recovery_plot = output_dir / "learnable_spectral_recovery.png"
    plot_results(aggregate_rows, r2_plot, recovery_plot)

    summary = {
        "device": str(device),
        "train_radius": args.train_radius,
        "eval_radius": args.eval_radius,
        "seeds": args.seeds,
        "csv": str(csv_path),
        "aggregate_csv": str(aggregate_csv_path),
        "r2_plot": str(r2_plot),
        "recovery_plot": str(recovery_plot),
        "rows": rows,
        "aggregate_rows": aggregate_rows,
        "histories": histories,
    }
    summary_path = output_dir / "learnable_spectral_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, aggregate_rows, summary)
    return summary


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["target"]), str(row["variant"])), []).append(row)
    out = []
    for (target, variant), values in sorted(groups.items()):
        out.append(
            {
                "target": target,
                "variant": variant,
                "n": len(values),
                "num_features": int(values[0]["num_features"]),
                "eval_r2_mean": float(np.mean([float(row["eval_r2"]) for row in values])),
                "eval_r2_std": float(np.std([float(row["eval_r2"]) for row in values])),
                "train_r2_mean": float(np.mean([float(row["train_r2"]) for row in values])),
                "frequency_error_mean": float(np.mean([float(row["frequency_error"]) for row in values])),
                "direction_alignment_mean": float(np.mean([float(row["direction_alignment"]) for row in values])),
                "steps_to_95_median": float(np.median([float(row["steps_to_95_oracle"]) for row in values])),
            }
        )
    return out


def plot_results(rows: list[dict[str, object]], r2_path: Path, recovery_path: Path) -> None:
    targets = sorted({str(row["target"]) for row in rows})
    variants = [
        "frozen_misaligned",
        "frozen_near_bank",
        "learned_random_init",
        "learned_near_init",
        "oracle",
        "relative_table_train_window",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, target in zip(axes.reshape(-1), targets):
        values = [next(row for row in rows if row["target"] == target and row["variant"] == variant) for variant in variants]
        x = np.arange(len(variants))
        y = [float(row["eval_r2_mean"]) for row in values]
        ax.bar(x, y, color="#516f9c")
        ax.set_xticks(x, variants, rotation=25, ha="right")
        ax.set_ylim(min(-0.2, min(y) - 0.05), 1.05)
        ax.set_title(target)
        ax.set_ylabel("eval R^2")
        for idx, value in enumerate(y):
            ax.text(idx, min(value + 0.03, 1.02), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(r2_path, dpi=180)
    plt.close(fig)

    learned = [row for row in rows if str(row["variant"]).startswith("learned")]
    labels = [f"{row['target']}\n{row['variant']}" for row in learned]
    x = np.arange(len(learned))
    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    freq = [float(row["frequency_error_mean"]) for row in learned]
    align = [float(row["direction_alignment_mean"]) for row in learned]
    ax1.bar(x - 0.18, freq, width=0.36, color="#8b5f3d", label="frequency error")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, align, width=0.36, color="#4d7c59", label="direction alignment")
    ax1.set_xticks(x, labels, rotation=30, ha="right")
    ax1.set_ylabel("frequency error")
    ax2.set_ylabel("abs direction cosine")
    ax2.set_ylim(0, 1.05)
    ax1.set_title("Learned Spectral Geometry Recovery")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(recovery_path, dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, aggregate_rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    variants = [
        "frozen_misaligned",
        "frozen_near_bank",
        "learned_random_init",
        "learned_near_init",
        "oracle",
        "relative_table_train_window",
    ]
    lines = [
        "# V2-A Learnable Spectral Geometry Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        f"- Train radius: {summary['train_radius']}",
        f"- Eval radius: {summary['eval_radius']}",
        f"- Seeds: {summary['seeds']}",
        "",
        "Run command:",
        "",
        "```bash",
        "python scripts/run_v2_spectral.py --device cuda --output-dir results/v2_spectral",
        "```",
        "",
        "## Eval R2",
        "",
        "| target | " + " | ".join(variants) + " |",
        "|" + "---|" * (len(variants) + 1),
    ]
    targets = sorted({str(row["target"]) for row in aggregate_rows})
    for target in targets:
        cells = []
        for variant in variants:
            row = next(item for item in aggregate_rows if item["target"] == target and item["variant"] == variant)
            cells.append(f"{float(row['eval_r2_mean']):.4f}")
        lines.append("| " + target + " | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Learned Recovery",
            "",
            "| target | variant | freq error | dir alignment |",
            "|---|---|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        if str(row["variant"]).startswith("learned"):
            lines.append(
                "| "
                + f"{row['target']} | {row['variant']} | {float(row['frequency_error_mean']):.4f} | "
                + f"{float(row['direction_alignment_mean']):.4f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Oracle fixed geometry is the reference for this controlled stage.",
            "- Learned near initialization tests whether spectral geometry can refine toward the teacher rather than relying on an oracle.",
            "- Learned random initialization is deliberately harder and reports seed-averaged behavior.",
            "- The train-window relative table is an interpolation upper bound, not a compact extrapolating model.",
            "",
            "Artifacts:",
            "",
            "- `learnable_spectral_results.csv`",
            "- `learnable_spectral_aggregate.csv`",
            "- `learnable_spectral_summary.json`",
            "- `learnable_spectral_eval_r2.png`",
            "- `learnable_spectral_recovery.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-A learnable spectral geometry experiments.")
    parser.add_argument("--train-radius", type=int, default=12)
    parser.add_argument("--eval-radius", type=int, default=20)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--lr-coeff", type=float, default=0.045)
    parser.add_argument("--lr-geom", type=float, default=0.018)
    parser.add_argument("--coeff-l2", type=float, default=1e-7)
    parser.add_argument("--lc-scale-reg", type=float, default=1e-5)
    parser.add_argument("--ridge", type=float, default=1e-9)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_spectral")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key not in {"rows", "histories"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
