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

from toric_pj.diagnostics.relative_table_geometry import load_bias_npz, relative_grid
from toric_pj.experiments.v12_phase_a_utils import (
    crop_visible_table,
    fit_with_ridge_grid,
    full_multijet_basis,
    max_abs,
    omegas_json,
    parse_csv_list,
    parse_float_list,
    parse_int_list,
    predict,
    r2_score,
    rectangular_mask,
    rms,
    select_omegas_from_source,
    signed_grid,
    summarize_groups,
    table_target,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V12 E4 compactified chart projection and tail scan.")
    parser.add_argument("--teacher-bias", required=True)
    parser.add_argument("--orders", type=str, default="1,2")
    parser.add_argument("--phase-charts", type=str, default="raw,asinh")
    parser.add_argument("--jet-charts", type=str, default="raw,bounded")
    parser.add_argument("--chart-scales", type=str, default="2,4,8,16")
    parser.add_argument("--feature-budget", type=int, default=108)
    parser.add_argument("--frequency-source", type=str, default="table_informed")
    parser.add_argument("--fit-targets", type=str, default="axis_plus_residual")
    parser.add_argument("--vis-radius", type=int, default=4)
    parser.add_argument("--full-radius", type=int, default=7)
    parser.add_argument("--ext-radius", type=int, default=16)
    parser.add_argument("--ridge-grid", type=str, default="1e-8,1e-6,1e-4,1e-2")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v12_e4_compactified_chart_projection_cifar10_10k")
    return parser.parse_args()


def phase_grad_mean(d: torch.Tensor, *, chart: str, scale: float) -> float:
    if chart == "raw":
        return 1.0
    if chart in {"asinh", "lc"}:
        grad = 1.0 / torch.sqrt(1.0 + (d / float(scale)).square())
        return float(grad.mean().detach().cpu())
    if chart == "log":
        grad = 1.0 / (1.0 + torch.abs(d) / float(scale))
        return float(grad.mean().detach().cpu())
    return float("nan")


def per_center_atoms(order: int) -> int:
    return 2 * ((order + 1) * (order + 2) // 2)


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, metadata = load_bias_npz(Path(args.teacher_bias), device=device)
    tables = bundle.tables.to(torch.float64)
    side = (int(tables.shape[-1]) + 1) // 2
    full_radius = int(args.full_radius or (side - 1))
    d_vis = signed_grid(args.vis_radius, device=device, dtype=torch.float64)
    d_full = relative_grid(side, device=device, dtype=torch.float64)
    d_ext = signed_grid(args.ext_radius, device=device, dtype=torch.float64)
    full_visible = rectangular_mask(d_full, int(args.vis_radius))
    ext_full = rectangular_mask(d_ext, full_radius)

    orders = parse_int_list(args.orders)
    phase_charts = parse_csv_list(args.phase_charts)
    jet_charts = parse_csv_list(args.jet_charts)
    chart_scales = parse_float_list(args.chart_scales)
    fit_targets = parse_csv_list(args.fit_targets)
    ridge_grid = parse_float_list(args.ridge_grid)

    rows: list[dict[str, object]] = []
    ridge_rows: list[dict[str, object]] = []
    for layer in range(tables.shape[0]):
        for head in range(tables.shape[1]):
            table = tables[layer, head]
            for fit_target in fit_targets:
                target, base = table_target(table, fit_target)
                target_vis = crop_visible_table(target, int(args.vis_radius))
                base_vis = crop_visible_table(base, int(args.vis_radius))
                table_vis = crop_visible_table(table, int(args.vis_radius))
                for order in orders:
                    k = max(1, (int(args.feature_budget) - 1) // per_center_atoms(order))
                    omegas = select_omegas_from_source(
                        args.frequency_source,
                        target_vis,
                        k=k,
                        seed=args.seed + 1000 * layer + 37 * head + 11 * order,
                    )
                    for phase_chart in phase_charts:
                        for jet_chart in jet_charts:
                            for chart_scale in chart_scales:
                                basis_name = f"{phase_chart}_phase_{jet_chart}_jet_J{order}_L{chart_scale:g}_k{k}"
                                train_matrix = full_multijet_basis(
                                    d_vis,
                                    omegas,
                                    order=order,
                                    scale=float(chart_scale),
                                    phase_chart=phase_chart,
                                    jet_chart=jet_chart,
                                    name=basis_name,
                                ).matrix
                                fit, path = fit_with_ridge_grid(train_matrix, target_vis.reshape(-1), ridge_grid=ridge_grid)
                                full_matrix = full_multijet_basis(
                                    d_full,
                                    omegas,
                                    order=order,
                                    scale=float(chart_scale),
                                    phase_chart=phase_chart,
                                    jet_chart=jet_chart,
                                    name=basis_name,
                                ).matrix
                                ext_matrix = full_multijet_basis(
                                    d_ext,
                                    omegas,
                                    order=order,
                                    scale=float(chart_scale),
                                    phase_chart=phase_chart,
                                    jet_chart=jet_chart,
                                    name=basis_name,
                                ).matrix
                                pred_vis_target = predict(train_matrix, fit.coeff, fit.column_norms).reshape_as(target_vis)
                                pred_full_target = predict(full_matrix, fit.coeff, fit.column_norms).reshape_as(table)
                                pred_ext_target = predict(ext_matrix, fit.coeff, fit.column_norms)
                                pred_vis_table = pred_vis_target + base_vis
                                pred_full_table = pred_full_target + base
                                heldout = ~full_visible
                                ext_outer = ~ext_full
                                row = {
                                    "dataset": metadata.get("dataset", "unknown"),
                                    "task": metadata.get("task", "unknown"),
                                    "teacher_basis": metadata.get("basis", "unknown"),
                                    "teacher_seed": metadata.get("seed", -1),
                                    "layer": layer,
                                    "head": head,
                                    "fit_target": fit_target,
                                    "frequency_source": args.frequency_source,
                                    "order": order,
                                    "phase_chart": phase_chart,
                                    "jet_chart": jet_chart,
                                    "chart_scale": float(chart_scale),
                                    "num_centers": int(k),
                                    "num_features": int(train_matrix.shape[1]),
                                    "selected_ridge": fit.ridge,
                                    "visible_target_r2": fit.r2,
                                    "visible_table_r2": r2_score(pred_vis_table, table_vis),
                                    "full_table_r2": r2_score(pred_full_table, table),
                                    "full_visible_gap": r2_score(pred_full_table, table) - r2_score(pred_vis_table, table_vis),
                                    "heldout_rms": rms(pred_full_table.reshape(-1)[heldout], table.reshape(-1)[heldout]),
                                    "extended_outer_pred_rms": rms(pred_ext_target[ext_outer]),
                                    "extended_outer_max_abs": max_abs(pred_ext_target[ext_outer]),
                                    "condition_number": fit.condition_number,
                                    "effective_rank": fit.effective_rank,
                                    "coeff_norm": fit.coeff_norm,
                                    "phase_grad_mean_full": phase_grad_mean(d_full, chart=phase_chart, scale=float(chart_scale)),
                                    "phase_grad_mean_ext": phase_grad_mean(d_ext, chart=phase_chart, scale=float(chart_scale)),
                                    "omegas": omegas_json(omegas),
                                }
                                rows.append(row)
                                for item in path:
                                    item.update(
                                        {
                                            "layer": layer,
                                            "head": head,
                                            "fit_target": fit_target,
                                            "order": order,
                                            "phase_chart": phase_chart,
                                            "jet_chart": jet_chart,
                                            "chart_scale": float(chart_scale),
                                            "num_centers": int(k),
                                            "num_features": int(train_matrix.shape[1]),
                                        }
                                    )
                                    ridge_rows.append(item)

    aggregate = summarize_groups(
        rows,
        keys=["fit_target", "order", "phase_chart", "jet_chart", "chart_scale"],
        numeric=[
            "visible_table_r2",
            "visible_target_r2",
            "full_table_r2",
            "full_visible_gap",
            "heldout_rms",
            "extended_outer_pred_rms",
            "extended_outer_max_abs",
            "condition_number",
            "effective_rank",
            "coeff_norm",
            "phase_grad_mean_full",
            "phase_grad_mean_ext",
            "num_features",
        ],
    )
    write_csv(output_dir / "compactified_chart_results.csv", rows)
    write_csv(output_dir / "compactified_chart_aggregate.csv", aggregate)
    write_csv(output_dir / "ridge_path.csv", ridge_rows)
    plot_pareto(aggregate, output_dir / "compactified_chart_pareto.pdf")

    summary = {
        "teacher_bias": args.teacher_bias,
        "metadata": metadata,
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "num_ridge_rows": len(ridge_rows),
        "orders": orders,
        "phase_charts": phase_charts,
        "jet_charts": jet_charts,
        "chart_scales": chart_scales,
        "feature_budget": int(args.feature_budget),
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, aggregate)
    return summary


def plot_pareto(aggregate: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    markers = {"raw": "o", "asinh": "s", "lc": "s", "log": "^"}
    for row in aggregate:
        x = float(row.get("visible_table_r2_mean", np.nan))
        y = float(row.get("heldout_rms_mean", np.nan))
        label = f"J{row['order']} {row['phase_chart']}/{row['jet_chart']} L{float(row['chart_scale']):g}"
        ax.scatter(x, y, marker=markers.get(str(row["phase_chart"]), "o"), s=40)
        ax.annotate(label, (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("visible table R2")
    ax.set_ylabel("heldout RMS")
    ax.set_title("Compactified chart Pareto")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V12 E4 Compactified Chart Projection",
        "",
        f"Teacher: `{summary['teacher_bias']}`",
        f"Rows: {summary['num_rows']}",
        "",
        "| target | J | phase | jet | L | n | vis R2 | full R2 | gap | heldout RMS | ext RMS | cond |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + f"{row['fit_target']} | {int(row['order'])} | {row['phase_chart']} | {row['jet_chart']} | "
            + f"{float(row['chart_scale']):g} | {int(row['n'])} | "
            + f"{float(row.get('visible_table_r2_mean', float('nan'))):.4f} | "
            + f"{float(row.get('full_table_r2_mean', float('nan'))):.4f} | "
            + f"{float(row.get('full_visible_gap_mean', float('nan'))):.4f} | "
            + f"{float(row.get('heldout_rms_mean', float('nan'))):.4f} | "
            + f"{float(row.get('extended_outer_pred_rms_mean', float('nan'))):.4f} | "
            + f"{float(row.get('condition_number_mean', float('nan'))):.2e} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `compactified_chart_results.csv`",
            "- `compactified_chart_aggregate.csv`",
            "- `ridge_path.csv`",
            "- `compactified_chart_pareto.pdf`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
