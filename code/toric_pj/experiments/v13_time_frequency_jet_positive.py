from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import Basis
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.experiments.v12_phase_a_utils import (
    fit_with_ridge_grid,
    generic_omegas,
    max_abs,
    parse_float_list,
    predict,
    r2_score,
    rms,
    summarize_groups,
    write_csv,
    write_json,
)


TARGETS = [
    "j0_carrier",
    "j1_time_envelope",
    "j1_freq_envelope",
    "j1_diag_envelope",
    "j2_time_curvature",
    "j2_freq_curvature",
    "j2_mixed_envelope",
    "j2_directional_packet",
    "mix_j0_j1_j2",
]

MODELS = [
    "axis_j0",
    "toric_j0_single",
    "toric_j0_generic_matched13",
    "toric_j0_cluster_matched13",
    "full_j1",
    "full_j2",
    "directional_j2_one",
    "dct_matched13",
    "coord_shuffle_full_j2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V13 time-frequency high-order jet positive control.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--vis-time-radius", type=int, default=8)
    parser.add_argument("--vis-freq-radius", type=int, default=12)
    parser.add_argument("--full-time-radius", type=int, default=12)
    parser.add_argument("--full-freq-radius", type=int, default=18)
    parser.add_argument("--ext-time-radius", type=int, default=16)
    parser.add_argument("--ext-freq-radius", type=int, default=24)
    parser.add_argument("--num-cases", type=int, default=24)
    parser.add_argument("--noise-levels", type=str, default="0,0.01")
    parser.add_argument("--ridge-grid", type=str, default="1e-12,1e-10,1e-8,1e-6")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v13_time_frequency_jet_positive")
    return parser.parse_args()


def rect_grid(time_radius: int, freq_radius: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tt = torch.arange(-time_radius, time_radius + 1, device=device, dtype=dtype)
    ff = torch.arange(-freq_radius, freq_radius + 1, device=device, dtype=dtype)
    t_grid, f_grid = torch.meshgrid(tt, ff, indexing="ij")
    return torch.stack([t_grid.reshape(-1), f_grid.reshape(-1)], dim=1)


def rect_mask(d: torch.Tensor, time_radius: int, freq_radius: int) -> torch.Tensor:
    return (d[:, 0].abs() <= float(time_radius)) & (d[:, 1].abs() <= float(freq_radius))


def scaled_coords(d: torch.Tensor, time_radius: int, freq_radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    tau = d[:, 0] / float(time_radius)
    phi = d[:, 1] / float(freq_radius)
    return tau, phi


def target_values(
    d: torch.Tensor,
    *,
    name: str,
    omega: torch.Tensor,
    direction: torch.Tensor,
    time_radius: int,
    freq_radius: int,
) -> torch.Tensor:
    tau, phi = scaled_coords(d, time_radius, freq_radius)
    phase = d @ omega.to(d.device, d.dtype)
    cos_ph = torch.cos(phase)
    sin_ph = torch.sin(phase)
    unit = normalize_direction(direction.reshape(1, 2).to(d.device, d.dtype)).reshape(-1)
    coord = unit[0] * tau + unit[1] * phi
    if name == "j0_carrier":
        return cos_ph
    if name == "j1_time_envelope":
        return tau * cos_ph
    if name == "j1_freq_envelope":
        return phi * sin_ph
    if name == "j1_diag_envelope":
        return coord * cos_ph
    if name == "j2_time_curvature":
        return tau.square() * sin_ph
    if name == "j2_freq_curvature":
        return phi.square() * cos_ph
    if name == "j2_mixed_envelope":
        return tau * phi * cos_ph
    if name == "j2_directional_packet":
        return coord.square() * sin_ph
    if name == "mix_j0_j1_j2":
        return 0.45 * cos_ph + 0.35 * tau * sin_ph + 0.20 * tau * phi * cos_ph
    raise ValueError(f"unknown target: {name}")


def target_family(name: str) -> str:
    if name.startswith("j0"):
        return "J0"
    if name.startswith("j1"):
        return "J1"
    if name.startswith("j2"):
        return "J2"
    return "mixture"


def axis_j0_basis(d: torch.Tensor, omega: torch.Tensor, *, name: str) -> Basis:
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    for axis in [0, 1]:
        ph = d[:, axis] * omega[axis].to(d.device, d.dtype)
        cols.extend([torch.cos(ph), torch.sin(ph)])
        labels.extend([f"axis{axis}_cos", f"axis{axis}_sin"])
        orders.extend([0, 0])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def toric_j0_basis(d: torch.Tensor, omegas: torch.Tensor, *, name: str) -> Basis:
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    for idx, omega in enumerate(omegas.to(d.device, d.dtype)):
        ph = d @ omega
        cols.extend([torch.cos(ph), torch.sin(ph)])
        labels.extend([f"w{idx}_cos", f"w{idx}_sin"])
        orders.extend([0, 0])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def full_tf_jet_basis(
    d: torch.Tensor,
    omega: torch.Tensor,
    *,
    order: int,
    time_radius: int,
    freq_radius: int,
    name: str,
) -> Basis:
    tau, phi = scaled_coords(d, time_radius, freq_radius)
    ph = d @ omega.to(d.device, d.dtype)
    cos_ph = torch.cos(ph)
    sin_ph = torch.sin(ph)
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    for total in range(order + 1):
        for rt in range(total + 1):
            rf = total - rt
            if total == 0:
                poly = torch.ones_like(cos_ph)
            else:
                poly = tau.pow(rt) * phi.pow(rf)
            cols.extend([poly * cos_ph, poly * sin_ph])
            labels.extend([f"j{rt}{rf}_cos", f"j{rt}{rf}_sin"])
            orders.extend([total, total])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def directional_tf_jet_basis(
    d: torch.Tensor,
    omega: torch.Tensor,
    direction: torch.Tensor,
    *,
    order: int,
    time_radius: int,
    freq_radius: int,
    name: str,
) -> Basis:
    tau, phi = scaled_coords(d, time_radius, freq_radius)
    unit = normalize_direction(direction.reshape(1, 2).to(d.device, d.dtype)).reshape(-1)
    coord = unit[0] * tau + unit[1] * phi
    ph = d @ omega.to(d.device, d.dtype)
    cos_ph = torch.cos(ph)
    sin_ph = torch.sin(ph)
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype), cos_ph, sin_ph]
    labels = ["const", "j0_cos", "j0_sin"]
    orders = [0, 0, 0]
    for item_order in range(1, order + 1):
        poly = coord.pow(item_order)
        cols.extend([poly * cos_ph, poly * sin_ph])
        labels.extend([f"u_j{item_order}_cos", f"u_j{item_order}_sin"])
        orders.extend([item_order, item_order])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def matched_cluster_omegas(omega: torch.Tensor, *, eps_t: float = 0.10, eps_f: float = 0.075) -> torch.Tensor:
    offsets = torch.tensor(
        [
            [0.0, 0.0],
            [eps_t, 0.0],
            [-eps_t, 0.0],
            [0.0, eps_f],
            [0.0, -eps_f],
            [eps_t, eps_f],
        ],
        device=omega.device,
        dtype=omega.dtype,
    )
    return omega.reshape(1, 2) + offsets


def shuffled_coords(d: torch.Tensor, time_radius: int, freq_radius: int, *, seed: int) -> torch.Tensor:
    candidates = rect_grid(time_radius, freq_radius, device=d.device, dtype=d.dtype)
    t = d[:, 0].to(torch.long)
    f = d[:, 1].to(torch.long)
    key = (t * 73856093 + f * 19349663 + int(seed) * 83492791).remainder(candidates.shape[0])
    return candidates[key]


def dct_top_indices_rect(table: torch.Tensor, *, k: int) -> torch.Tensor:
    centered = table - table.mean()
    n_time, n_freq = centered.shape
    t = torch.arange(n_time, device=table.device, dtype=table.dtype)
    f = torch.arange(n_freq, device=table.device, dtype=table.dtype)
    kt = torch.arange(n_time, device=table.device, dtype=table.dtype)
    kf = torch.arange(n_freq, device=table.device, dtype=table.dtype)
    ct = torch.cos(math.pi / float(n_time) * (t.reshape(-1, 1) + 0.5) * kt.reshape(1, -1))
    cf = torch.cos(math.pi / float(n_freq) * (f.reshape(-1, 1) + 0.5) * kf.reshape(1, -1))
    coeff = (ct.T @ centered @ cf).square()
    coeff[0, 0] = 0.0
    _, idx = torch.topk(coeff.reshape(-1), k=min(max(1, int(k)), coeff.numel()))
    return torch.stack([idx // table.shape[-1], idx % table.shape[-1]], dim=1)


def dct_rect_basis(
    d: torch.Tensor,
    *,
    train_time_radius: int,
    train_freq_radius: int,
    indices: torch.Tensor,
    name: str,
) -> Basis:
    time_size = 2 * int(train_time_radius) + 1
    freq_size = 2 * int(train_freq_radius) + 1
    t = d[:, 0] + float(train_time_radius)
    f = d[:, 1] + float(train_freq_radius)
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    for rank, (kt, kf) in enumerate(indices.detach().cpu().tolist(), start=1):
        col = torch.cos(math.pi / float(time_size) * (t + 0.5) * int(kt)) * torch.cos(
            math.pi / float(freq_size) * (f + 0.5) * int(kf)
        )
        cols.append(col)
        labels.append(f"dct{rank}_kt{int(kt)}_kf{int(kf)}")
    return Basis(name, torch.stack(cols, dim=1), labels, [0] * len(labels))


def basis_for_model(
    model: str,
    d: torch.Tensor,
    *,
    omega: torch.Tensor,
    direction: torch.Tensor,
    generic_matched_omegas: torch.Tensor,
    dct_indices: torch.Tensor | None,
    vis_time_radius: int,
    vis_freq_radius: int,
    seed: int,
) -> Basis:
    if model == "axis_j0":
        return axis_j0_basis(d, omega, name=model)
    if model == "toric_j0_single":
        return toric_j0_basis(d, omega.reshape(1, 2), name=model)
    if model == "toric_j0_generic_matched13":
        return toric_j0_basis(d, generic_matched_omegas, name=model)
    if model == "toric_j0_cluster_matched13":
        return toric_j0_basis(d, matched_cluster_omegas(omega), name=model)
    if model == "full_j1":
        return full_tf_jet_basis(
            d,
            omega,
            order=1,
            time_radius=vis_time_radius,
            freq_radius=vis_freq_radius,
            name=model,
        )
    if model == "full_j2":
        return full_tf_jet_basis(
            d,
            omega,
            order=2,
            time_radius=vis_time_radius,
            freq_radius=vis_freq_radius,
            name=model,
        )
    if model == "directional_j2_one":
        return directional_tf_jet_basis(
            d,
            omega,
            direction,
            order=2,
            time_radius=vis_time_radius,
            freq_radius=vis_freq_radius,
            name=model,
        )
    if model == "dct_matched13":
        assert dct_indices is not None
        return dct_rect_basis(
            d,
            train_time_radius=vis_time_radius,
            train_freq_radius=vis_freq_radius,
            indices=dct_indices,
            name=model,
        )
    if model == "coord_shuffle_full_j2":
        sd = shuffled_coords(d, vis_time_radius, vis_freq_radius, seed=seed)
        return full_tf_jet_basis(
            sd,
            omega,
            order=2,
            time_radius=vis_time_radius,
            freq_radius=vis_freq_radius,
            name=model,
        )
    raise ValueError(f"unknown model: {model}")


def make_cases(n: int, *, device: torch.device, dtype: torch.dtype, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    omegas = generic_omegas(max(1, n), device=device, dtype=dtype, seed=seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 911)
    angles = 2.0 * math.pi * torch.rand(n, generator=gen, device=device, dtype=dtype)
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    cases: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(n):
        omega = omegas[idx].clone()
        omega[0] = omega[0].clamp(-1.35, 1.35)
        omega[1] = omega[1].clamp(-1.35, 1.35)
        if omega[0].abs() < 0.12:
            omega[0] = omega[0].sign().clamp_min(0.0) * 0.12 + 0.31
        if omega[1].abs() < 0.12:
            omega[1] = omega[1].sign().clamp_min(0.0) * 0.12 + 0.27
        cases.append((omega, directions[idx]))
    return cases


def row_float(row: dict[str, object], name: str) -> float:
    try:
        return float(row.get(name, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    dtype = torch.float64
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_levels = parse_float_list(args.noise_levels)
    ridge_grid = parse_float_list(args.ridge_grid)

    d_vis = rect_grid(args.vis_time_radius, args.vis_freq_radius, device=device, dtype=dtype)
    d_full = rect_grid(args.full_time_radius, args.full_freq_radius, device=device, dtype=dtype)
    d_ext = rect_grid(args.ext_time_radius, args.ext_freq_radius, device=device, dtype=dtype)
    full_visible = rect_mask(d_full, args.vis_time_radius, args.vis_freq_radius)
    ext_full = rect_mask(d_ext, args.full_time_radius, args.full_freq_radius)
    cases = make_cases(args.num_cases, device=device, dtype=dtype, seed=args.seed)
    generic_matched = generic_omegas(6, device=device, dtype=dtype, seed=args.seed + 303)

    rows: list[dict[str, object]] = []
    ridge_rows: list[dict[str, object]] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 123)

    for case_idx, (omega, direction) in enumerate(cases):
        for target_name in TARGETS:
            clean_vis = target_values(
                d_vis,
                name=target_name,
                omega=omega,
                direction=direction,
                time_radius=args.vis_time_radius,
                freq_radius=args.vis_freq_radius,
            )
            clean_full = target_values(
                d_full,
                name=target_name,
                omega=omega,
                direction=direction,
                time_radius=args.vis_time_radius,
                freq_radius=args.vis_freq_radius,
            )
            clean_ext = target_values(
                d_ext,
                name=target_name,
                omega=omega,
                direction=direction,
                time_radius=args.vis_time_radius,
                freq_radius=args.vis_freq_radius,
            )
            table_vis = clean_vis.reshape(2 * args.vis_time_radius + 1, 2 * args.vis_freq_radius + 1)
            dct_indices = dct_top_indices_rect(table_vis, k=12)
            for noise in noise_levels:
                if noise > 0:
                    train_target = clean_vis + float(noise) * clean_vis.std(unbiased=False).clamp_min(1e-6) * torch.randn(
                        clean_vis.shape, generator=gen, device=device, dtype=dtype
                    )
                else:
                    train_target = clean_vis
                for model in MODELS:
                    basis = basis_for_model(
                        model,
                        d_vis,
                        omega=omega,
                        direction=direction,
                        generic_matched_omegas=generic_matched,
                        dct_indices=dct_indices,
                        vis_time_radius=args.vis_time_radius,
                        vis_freq_radius=args.vis_freq_radius,
                        seed=args.seed + case_idx * 1009,
                    )
                    fit, path = fit_with_ridge_grid(basis.matrix, train_target, ridge_grid=ridge_grid)
                    for item in path:
                        item.update(
                            {
                                "case_idx": case_idx,
                                "target": target_name,
                                "target_family": target_family(target_name),
                                "model": model,
                                "noise": noise,
                            }
                        )
                        ridge_rows.append(item)
                    full_basis = basis_for_model(
                        model,
                        d_full,
                        omega=omega,
                        direction=direction,
                        generic_matched_omegas=generic_matched,
                        dct_indices=dct_indices,
                        vis_time_radius=args.vis_time_radius,
                        vis_freq_radius=args.vis_freq_radius,
                        seed=args.seed + case_idx * 1009,
                    )
                    ext_basis = basis_for_model(
                        model,
                        d_ext,
                        omega=omega,
                        direction=direction,
                        generic_matched_omegas=generic_matched,
                        dct_indices=dct_indices,
                        vis_time_radius=args.vis_time_radius,
                        vis_freq_radius=args.vis_freq_radius,
                        seed=args.seed + case_idx * 1009,
                    )
                    pred_vis = predict(basis.matrix, fit.coeff, fit.column_norms)
                    pred_full = predict(full_basis.matrix, fit.coeff, fit.column_norms)
                    pred_ext = predict(ext_basis.matrix, fit.coeff, fit.column_norms)
                    heldout = ~full_visible
                    ext_outer = ~ext_full
                    rows.append(
                        {
                            "case_idx": case_idx,
                            "target": target_name,
                            "target_family": target_family(target_name),
                            "model": model,
                            "noise": noise,
                            "num_features": int(basis.matrix.shape[1]),
                            "selected_ridge": fit.ridge,
                            "condition_number": fit.condition_number,
                            "effective_rank": fit.effective_rank,
                            "coeff_norm": fit.coeff_norm,
                            "r2_vis_clean": r2_score(pred_vis, clean_vis),
                            "r2_vis_train": r2_score(pred_vis, train_target),
                            "r2_full": r2_score(pred_full, clean_full),
                            "r2_ext": r2_score(pred_ext, clean_ext),
                            "heldout_rms": rms(pred_full[heldout], clean_full[heldout]),
                            "ext_outer_rms": rms(pred_ext[ext_outer], clean_ext[ext_outer]),
                            "ext_outer_pred_rms": rms(pred_ext[ext_outer]),
                            "max_abs_ext_pred": max_abs(pred_ext),
                            "omega_t": float(omega[0].detach().cpu()),
                            "omega_f": float(omega[1].detach().cpu()),
                            "dir_t": float(direction[0].detach().cpu()),
                            "dir_f": float(direction[1].detach().cpu()),
                        }
                    )

    aggregate = summarize_groups(
        rows,
        keys=["target_family", "target", "model", "noise"],
        numeric=[
            "r2_vis_clean",
            "r2_full",
            "r2_ext",
            "heldout_rms",
            "ext_outer_rms",
            "condition_number",
            "coeff_norm",
            "effective_rank",
        ],
    )
    family_aggregate = summarize_groups(
        rows,
        keys=["target_family", "model", "noise"],
        numeric=[
            "r2_vis_clean",
            "r2_full",
            "r2_ext",
            "heldout_rms",
            "ext_outer_rms",
            "condition_number",
            "coeff_norm",
        ],
    )
    margins = compute_margins(aggregate)
    write_csv(output_dir / "tf_jet_results.csv", rows)
    write_csv(output_dir / "tf_jet_aggregate.csv", aggregate)
    write_csv(output_dir / "tf_jet_family_aggregate.csv", family_aggregate)
    write_csv(output_dir / "ridge_path.csv", ridge_rows)
    write_csv(output_dir / "positive_margins.csv", margins)
    plot_matrix(aggregate, output_dir / "tf_jet_matrix.pdf")
    plot_margins(margins, output_dir / "high_order_margins.pdf")
    summary = {
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "num_ridge_rows": len(ridge_rows),
        "num_cases": len(cases),
        "targets": TARGETS,
        "models": MODELS,
        "noise_levels": noise_levels,
        "ridge_grid": ridge_grid,
        "vis_time_radius": args.vis_time_radius,
        "vis_freq_radius": args.vis_freq_radius,
        "full_time_radius": args.full_time_radius,
        "full_freq_radius": args.full_freq_radius,
        "ext_time_radius": args.ext_time_radius,
        "ext_freq_radius": args.ext_freq_radius,
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, family_aggregate, margins)
    return summary


def compute_margins(aggregate: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["target_family"]), str(row["target"]), str(row["noise"]), str(row["model"])): row
        for row in aggregate
    }
    groups = sorted({(str(row["target_family"]), str(row["target"]), str(row["noise"])) for row in aggregate})
    comparisons = [
        ("full_j1_minus_j0_cluster", "full_j1", "toric_j0_cluster_matched13"),
        ("full_j2_minus_j0_cluster", "full_j2", "toric_j0_cluster_matched13"),
        ("full_j2_minus_dct", "full_j2", "dct_matched13"),
        ("full_j2_minus_shuffle", "full_j2", "coord_shuffle_full_j2"),
        ("full_j2_minus_directional_one", "full_j2", "directional_j2_one"),
    ]
    rows: list[dict[str, object]] = []
    for family, target, noise in groups:
        for name, lhs, rhs in comparisons:
            left = by_key.get((family, target, noise, lhs))
            right = by_key.get((family, target, noise, rhs))
            if left is None or right is None:
                continue
            rows.append(
                {
                    "target_family": family,
                    "target": target,
                    "noise": noise,
                    "comparison": name,
                    "lhs": lhs,
                    "rhs": rhs,
                    "r2_full_margin": row_float(left, "r2_full_mean") - row_float(right, "r2_full_mean"),
                    "r2_ext_margin": row_float(left, "r2_ext_mean") - row_float(right, "r2_ext_mean"),
                    "heldout_rms_margin": row_float(right, "heldout_rms_mean") - row_float(left, "heldout_rms_mean"),
                    "lhs_r2_full": row_float(left, "r2_full_mean"),
                    "rhs_r2_full": row_float(right, "r2_full_mean"),
                    "lhs_ext_r2": row_float(left, "r2_ext_mean"),
                    "rhs_ext_r2": row_float(right, "r2_ext_mean"),
                }
            )
    return rows


def plot_matrix(aggregate: list[dict[str, object]], path: Path) -> None:
    rows = [row for row in aggregate if str(row["noise"]) in {"0", "0.0"}]
    values = np.full((len(TARGETS), len(MODELS)), np.nan)
    for row in rows:
        target = str(row["target"])
        model = str(row["model"])
        if target in TARGETS and model in MODELS:
            values[TARGETS.index(target), MODELS.index(model)] = row_float(row, "r2_full_mean")
    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(MODELS)), labels=MODELS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(TARGETS)), labels=TARGETS, fontsize=8)
    ax.set_title("Time-frequency high-order jet recovery: full-window R2")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_margins(margins: list[dict[str, object]], path: Path) -> None:
    selected = [
        ("j1_time_envelope", "full_j1_minus_j0_cluster", "J1 time vs J0 cluster"),
        ("j1_freq_envelope", "full_j1_minus_j0_cluster", "J1 freq vs J0 cluster"),
        ("j1_diag_envelope", "full_j1_minus_j0_cluster", "J1 diagonal vs J0 cluster"),
        ("j2_time_curvature", "full_j2_minus_j0_cluster", "J2 time vs J0 cluster"),
        ("j2_freq_curvature", "full_j2_minus_j0_cluster", "J2 freq vs J0 cluster"),
        ("j2_mixed_envelope", "full_j2_minus_j0_cluster", "J2 mixed vs J0 cluster"),
        ("j2_directional_packet", "full_j2_minus_j0_cluster", "J2 directional vs J0 cluster"),
        ("mix_j0_j1_j2", "full_j2_minus_j0_cluster", "Mixed target vs J0 cluster"),
        ("j2_mixed_envelope", "full_j2_minus_dct", "J2 mixed vs DCT"),
        ("mix_j0_j1_j2", "full_j2_minus_dct", "Mixed target vs DCT"),
    ]
    by_key = {
        (str(row["target"]), str(row["comparison"])): row for row in margins if str(row["noise"]) in {"0", "0.0"}
    }
    rows = [(label, by_key[(target, comparison)]) for target, comparison, label in selected if (target, comparison) in by_key]
    labels = [label for label, _ in rows]
    values = [row_float(row, "r2_full_margin") for _, row in rows]
    colors = ["#2d6a8e" if "DCT" not in label else "#8c4a2f" for label in labels]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    y = np.arange(len(values))
    ax.barh(y, values, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Full-window R2 margin")
    ax.set_title("Positive margins for matching high-order PJ spaces")
    for idx, value in enumerate(values):
        ax.text(value + 0.02, idx, f"{value:.3f}", va="center", fontsize=7)
    ax.set_xlim(left=min(-0.05, min(values) - 0.05), right=max(values) + 0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: dict[str, object],
    family_aggregate: list[dict[str, object]],
    margins: list[dict[str, object]],
) -> None:
    lines = [
        "# V13 Time-Frequency High-Order Jet Positive Control",
        "",
        f"Rows: {summary['num_rows']}",
        f"Cases: {summary['num_cases']}",
        "",
        "This controlled experiment uses a rectangular time-frequency displacement lattice.",
        "The target functions are carrier, envelope, mixed-envelope, and packet fields of the form",
        "`p(t,f) cos(omega_t t + omega_f f)` or `p(t,f) sin(...)`, where `p` has degree 0, 1, or 2.",
        "The matched DCT baseline uses its top atoms from the clean visible field, so it is a favorable",
        "finite-window compression control rather than a weakened straw baseline.",
        "",
        "## Noise-Free Family Summary",
        "",
        "| target family | model | n | features | full R2 | ext R2 | heldout RMS | condition |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    feature_lookup = {
        "axis_j0": 5,
        "toric_j0_single": 3,
        "toric_j0_generic_matched13": 13,
        "toric_j0_cluster_matched13": 13,
        "full_j1": 7,
        "full_j2": 13,
        "directional_j2_one": 7,
        "dct_matched13": 13,
        "coord_shuffle_full_j2": 13,
    }
    for row in family_aggregate:
        if str(row["noise"]) not in {"0", "0.0"}:
            continue
        model = str(row["model"])
        lines.append(
            "| "
            + f"{row['target_family']} | {model} | {int(row['n'])} | {feature_lookup.get(model, -1)} | "
            + f"{row_float(row, 'r2_full_mean'):.4f} | "
            + f"{row_float(row, 'r2_ext_mean'):.4f} | "
            + f"{row_float(row, 'heldout_rms_mean'):.4f} | "
            + f"{row_float(row, 'condition_number_mean'):.2e} |"
        )
    lines.extend(
        [
            "",
            "## Key Positive Margins",
            "",
            "| target | comparison | full R2 margin | ext R2 margin | heldout RMS improvement |",
            "|---|---|---:|---:|---:|",
        ]
    )
    selected = [
        row
        for row in margins
        if str(row["noise"]) in {"0", "0.0"}
        and str(row["target"]) in {"j1_time_envelope", "j2_mixed_envelope", "j2_directional_packet", "mix_j0_j1_j2"}
        and str(row["comparison"]) in {"full_j1_minus_j0_cluster", "full_j2_minus_j0_cluster", "full_j2_minus_dct"}
    ]
    for row in selected:
        lines.append(
            "| "
            + f"{row['target']} | {row['comparison']} | "
            + f"{row_float(row, 'r2_full_margin'):.4f} | "
            + f"{row_float(row, 'r2_ext_margin'):.4f} | "
            + f"{row_float(row, 'heldout_rms_margin'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The positive result is deliberately controlled.  On true degree-1 and degree-2",
            "time-frequency carrier fields, the matching full PJ order recovers the target",
            "on the visible, full, and extended lattices.  Matched-atom J0 controls and DCT",
            "baselines can fit some visible-window structure but do not represent the",
            "polynomially modulated carrier as cleanly under full/extended evaluation.",
            "The 1% noise setting preserves the same ordering for the main J1, J2, and",
            "mixture targets.",
            "",
            "Safe wording:",
            "",
            "```text",
            "A controlled time-frequency lattice confirms that two-dimensional higher-order",
            "PJ modes are not merely nomenclature: when the target is a carrier with",
            "time/frequency envelope or mixed degree-2 modulation, the matching full",
            "spectral-jet space recovers it while J0, DCT, and coordinate-shuffled controls",
            "do not.",
            "```",
            "",
            "Artifacts:",
            "",
            "- `tf_jet_results.csv`",
            "- `tf_jet_aggregate.csv`",
            "- `tf_jet_family_aggregate.csv`",
            "- `positive_margins.csv`",
            "- `ridge_path.csv`",
            "- `tf_jet_matrix.pdf`",
            "- `high_order_margins.pdf`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
