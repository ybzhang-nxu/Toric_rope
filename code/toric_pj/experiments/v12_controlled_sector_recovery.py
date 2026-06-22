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

from toric_pj.experiments.v12_phase_a_utils import (
    dct_basis_from_indices,
    directional_jet_basis,
    fit_with_ridge_grid,
    full_multijet_basis,
    generic_omegas,
    max_abs,
    parse_float_list,
    predict,
    r2_score,
    rectangular_mask,
    rms,
    signed_grid,
    summarize_groups,
    top_dct_indices,
    toric_j0_basis,
    write_csv,
    write_json,
)
from toric_pj.experiments.v12_phase_a_utils import axis_j0_basis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V12 E0 controlled multivariate sector recovery.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--vis-radius", type=int, default=4)
    parser.add_argument("--full-radius", type=int, default=7)
    parser.add_argument("--ext-radius", type=int, default=16)
    parser.add_argument("--num-frequencies", type=int, default=20)
    parser.add_argument("--num-directions", type=int, default=20)
    parser.add_argument("--noise-levels", type=str, default="0,0.01,0.05")
    parser.add_argument("--ridge-grid", type=str, default="1e-12,1e-10,1e-8,1e-6")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v12_e0_controlled_sector_recovery")
    return parser.parse_args()


def random_directions(n: int, *, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    angles = 2.0 * math.pi * torch.rand(n, generator=gen, device=device, dtype=dtype)
    out = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    axis_like = (out[:, 0].abs() < 0.08) | (out[:, 1].abs() < 0.08)
    out[axis_like, 0] += 0.17
    out = out / torch.linalg.norm(out, dim=1, keepdim=True).clamp_min(1e-12)
    return out


def shuffled_coords(d: torch.Tensor, *, radius: int, seed: int) -> torch.Tensor:
    candidates = signed_grid(radius, device=d.device, dtype=d.dtype)
    dx = d[:, 0].to(torch.long)
    dy = d[:, 1].to(torch.long)
    key = (dx * 73856093 + dy * 19349663 + int(seed) * 83492791).remainder(candidates.shape[0])
    return candidates[key]


def target_values(d: torch.Tensor, *, name: str, omega: torch.Tensor, u: torch.Tensor, scale: float) -> torch.Tensor:
    phase = d @ omega.to(d.device, d.dtype)
    cos_ph = torch.cos(phase)
    sin_ph = torch.sin(phase)
    x = d[:, 0] / float(scale)
    y = d[:, 1] / float(scale)
    coord = (d @ u.to(d.device, d.dtype)) / float(scale)
    if name == "j0_cos":
        return cos_ph
    if name == "j0_sin":
        return sin_ph
    if name == "j1_x_cos":
        return x * cos_ph
    if name == "j1_y_sin":
        return y * sin_ph
    if name == "j1_u_cos":
        return coord * cos_ph
    if name == "j2_xx_cos":
        return x.square() * cos_ph
    if name == "j2_xy_sin":
        return x * y * sin_ph
    if name == "j2_u_cos":
        return coord.square() * cos_ph
    if name == "affine_x":
        return x
    if name == "affine_y":
        return y
    if name == "affine_xy":
        return x * y
    if name == "mix_j0_j1":
        return 0.65 * cos_ph + 0.35 * coord * sin_ph
    if name == "mix_j0_j2":
        return 0.55 * sin_ph + 0.30 * x * y * cos_ph + 0.15 * x
    if name == "mix_axis_oblique":
        return 0.45 * torch.cos(d[:, 0] * omega[0]) + 0.35 * cos_ph + 0.20 * x * y
    raise ValueError(f"unknown target: {name}")


TARGETS = [
    "j0_cos",
    "j0_sin",
    "j1_x_cos",
    "j1_y_sin",
    "j1_u_cos",
    "j2_xx_cos",
    "j2_xy_sin",
    "j2_u_cos",
    "affine_x",
    "affine_y",
    "affine_xy",
    "mix_j0_j1",
    "mix_j0_j2",
    "mix_axis_oblique",
]


def basis_for_model(
    model: str,
    d: torch.Tensor,
    *,
    omega: torch.Tensor,
    u: torch.Tensor,
    vis_radius: int,
    dct_indices: torch.Tensor | None,
    seed: int,
) -> torch.Tensor:
    if model == "axis_j0":
        return axis_j0_basis(d, omega=omega, name=model).matrix
    if model == "toric_j0":
        return toric_j0_basis(d, omega.reshape(1, 2), name=model).matrix
    if model == "full_j1":
        return full_multijet_basis(d, omega.reshape(1, 2), order=1, scale=float(vis_radius), name=model).matrix
    if model == "full_j2":
        return full_multijet_basis(d, omega.reshape(1, 2), order=2, scale=float(vis_radius), name=model).matrix
    if model == "directional_j1":
        return directional_jet_basis(
            d,
            omega.reshape(1, 2),
            directions=u.reshape(1, 2),
            order=1,
            scale=float(vis_radius),
            name=model,
        ).matrix
    if model == "directional_j2":
        return directional_jet_basis(
            d,
            omega.reshape(1, 2),
            directions=u.reshape(1, 2),
            order=2,
            scale=float(vis_radius),
            name=model,
        ).matrix
    if model == "dct_matched":
        assert dct_indices is not None
        return dct_basis_from_indices(d, train_radius=vis_radius, indices=dct_indices, name=model).matrix
    if model == "coord_shuffle_j2":
        sd = shuffled_coords(d, radius=vis_radius, seed=seed)
        return full_multijet_basis(sd, omega.reshape(1, 2), order=2, scale=float(vis_radius), name=model).matrix
    raise ValueError(f"unknown model: {model}")


MODELS = [
    "axis_j0",
    "toric_j0",
    "full_j1",
    "full_j2",
    "directional_j1",
    "directional_j2",
    "dct_matched",
    "coord_shuffle_j2",
]


def target_family(name: str) -> str:
    if name.startswith("j0"):
        return "J0"
    if name.startswith("j1"):
        return "J1"
    if name.startswith("j2"):
        return "J2"
    if name.startswith("affine"):
        return "affine"
    return "mixture"


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    dtype = torch.float64
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_levels = parse_float_list(args.noise_levels)
    ridge_grid = parse_float_list(args.ridge_grid)
    omegas = generic_omegas(args.num_frequencies, device=device, dtype=dtype, seed=args.seed)
    directions = random_directions(args.num_directions, device=device, dtype=dtype, seed=args.seed + 19)

    d_vis = signed_grid(args.vis_radius, device=device, dtype=dtype)
    d_full = signed_grid(args.full_radius, device=device, dtype=dtype)
    d_ext = signed_grid(args.ext_radius, device=device, dtype=dtype)
    full_visible = rectangular_mask(d_full, args.vis_radius)
    ext_full = rectangular_mask(d_ext, args.full_radius)

    rows: list[dict[str, object]] = []
    ridge_rows: list[dict[str, object]] = []
    case_count = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 123)

    for freq_idx, omega in enumerate(omegas):
        for dir_idx, u in enumerate(directions):
            if args.max_cases is not None and case_count >= args.max_cases:
                break
            case_count += 1
            for target_name in TARGETS:
                clean_vis = target_values(d_vis, name=target_name, omega=omega, u=u, scale=float(args.vis_radius))
                clean_full = target_values(d_full, name=target_name, omega=omega, u=u, scale=float(args.vis_radius))
                clean_ext = target_values(d_ext, name=target_name, omega=omega, u=u, scale=float(args.vis_radius))
                for noise in noise_levels:
                    if noise > 0:
                        train_target = clean_vis + float(noise) * clean_vis.std(unbiased=False).clamp_min(1e-6) * torch.randn(
                            clean_vis.shape, generator=gen, device=device, dtype=dtype
                        )
                    else:
                        train_target = clean_vis
                    dct_k = max(1, 13 - 1)
                    dct_indices = top_dct_indices(train_target.reshape(2 * args.vis_radius + 1, 2 * args.vis_radius + 1), k=dct_k)
                    for model in MODELS:
                        train_matrix = basis_for_model(
                            model,
                            d_vis,
                            omega=omega,
                            u=u,
                            vis_radius=args.vis_radius,
                            dct_indices=dct_indices,
                            seed=args.seed + 1000 * freq_idx + 17 * dir_idx,
                        )
                        fit, path = fit_with_ridge_grid(train_matrix, train_target, ridge_grid=ridge_grid)
                        for item in path:
                            item.update(
                                {
                                    "target": target_name,
                                    "target_family": target_family(target_name),
                                    "model": model,
                                    "freq_idx": freq_idx,
                                    "dir_idx": dir_idx,
                                    "noise": noise,
                                }
                            )
                            ridge_rows.append(item)
                        full_matrix = basis_for_model(
                            model,
                            d_full,
                            omega=omega,
                            u=u,
                            vis_radius=args.vis_radius,
                            dct_indices=dct_indices,
                            seed=args.seed + 1000 * freq_idx + 17 * dir_idx,
                        )
                        ext_matrix = basis_for_model(
                            model,
                            d_ext,
                            omega=omega,
                            u=u,
                            vis_radius=args.vis_radius,
                            dct_indices=dct_indices,
                            seed=args.seed + 1000 * freq_idx + 17 * dir_idx,
                        )
                        pred_vis = predict(train_matrix, fit.coeff, fit.column_norms)
                        pred_full = predict(full_matrix, fit.coeff, fit.column_norms)
                        pred_ext = predict(ext_matrix, fit.coeff, fit.column_norms)
                        heldout = ~full_visible
                        ext_outer = ~ext_full
                        rows.append(
                            {
                                "target": target_name,
                                "target_family": target_family(target_name),
                                "model": model,
                                "freq_idx": freq_idx,
                                "dir_idx": dir_idx,
                                "noise": noise,
                                "num_features": int(train_matrix.shape[1]),
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
                                "omega_x": float(omega[0].detach().cpu()),
                                "omega_y": float(omega[1].detach().cpu()),
                                "u_x": float(u[0].detach().cpu()),
                                "u_y": float(u[1].detach().cpu()),
                            }
                        )
        if args.max_cases is not None and case_count >= args.max_cases:
            break

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
    write_csv(output_dir / "sector_recovery_results.csv", rows)
    write_csv(output_dir / "sector_recovery_aggregate.csv", aggregate)
    write_csv(output_dir / "ridge_path.csv", ridge_rows)
    plot_heatmap(aggregate, output_dir / "sector_recovery_matrix.pdf")
    plot_tail(rows, output_dir / "tail_scatter.pdf")

    summary = {
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "num_ridge_rows": len(ridge_rows),
        "num_cases": case_count,
        "targets": TARGETS,
        "models": MODELS,
        "noise_levels": noise_levels,
        "ridge_grid": ridge_grid,
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, aggregate)
    return summary


def plot_heatmap(aggregate: list[dict[str, object]], path: Path) -> None:
    rows = [r for r in aggregate if str(r["noise"]) in {"0", "0.0"}]
    targets = TARGETS
    models = MODELS
    values = np.full((len(targets), len(models)), np.nan)
    for row in rows:
        target = str(row["target"])
        model = str(row["model"])
        if target in targets and model in models:
            values[targets.index(target), models.index(model)] = float(row.get("r2_full_mean", np.nan))
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    im = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(models)), labels=models, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(targets)), labels=targets, fontsize=8)
    ax.set_title("Controlled sector recovery: full-window R2")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tail(rows: list[dict[str, object]], path: Path) -> None:
    filtered = [row for row in rows if float(row["noise"]) == 0.0]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for model in MODELS:
        vals = [row for row in filtered if row["model"] == model]
        if not vals:
            continue
        x = [float(row["r2_vis_clean"]) for row in vals]
        y = [float(row["ext_outer_rms"]) for row in vals]
        ax.scatter(x, y, s=9, alpha=0.45, label=model)
    ax.set_xlabel("visible R2")
    ax.set_ylabel("extended outer RMS")
    ax.set_title("Visible fit versus extension tail")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V12 E0 Controlled Sector Recovery",
        "",
        f"Rows: {summary['num_rows']}",
        f"Cases: {summary['num_cases']}",
        "",
        "## Noise-free full-window R2 by target family",
        "",
        "| target family | model | n | full R2 mean | ext R2 mean | heldout RMS mean | condition mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    family_rows = summarize_groups(
        [row for row in aggregate if str(row.get("noise")) in {"0", "0.0"}],
        keys=["target_family", "model"],
        numeric=["r2_full_mean", "r2_ext_mean", "heldout_rms_mean", "condition_number_mean"],
    )
    for row in family_rows:
        lines.append(
            "| "
            + f"{row['target_family']} | {row['model']} | {int(row['n'])} | "
            + f"{float(row.get('r2_full_mean_mean', float('nan'))):.4f} | "
            + f"{float(row.get('r2_ext_mean_mean', float('nan'))):.4f} | "
            + f"{float(row.get('heldout_rms_mean_mean', float('nan'))):.4f} | "
            + f"{float(row.get('condition_number_mean_mean', float('nan'))):.2e} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `sector_recovery_results.csv`",
            "- `sector_recovery_aggregate.csv`",
            "- `ridge_path.csv`",
            "- `sector_recovery_matrix.pdf`",
            "- `tail_scatter.pdf`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
