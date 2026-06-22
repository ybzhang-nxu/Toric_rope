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
    default_directions,
    directional_jet_basis,
    fit_with_ridge_grid,
    full_multijet_basis,
    omegas_json,
    parse_csv_list,
    parse_float_list,
    parse_int_list,
    predict,
    r2_score,
    select_omegas_from_source,
    summarize_groups,
    table_target,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V12 E1 fair teacher J0/J1/J2 projection.")
    parser.add_argument("--teacher-bias", required=True)
    parser.add_argument("--protocols", type=str, default="nested,matched")
    parser.add_argument("--frequency-sources", type=str, default="generic,table_informed")
    parser.add_argument("--basis-families", type=str, default="full,directional")
    parser.add_argument("--orders", type=str, default="0,1,2")
    parser.add_argument("--nested-centers", type=int, default=6)
    parser.add_argument("--matched-atom-budgets", type=str, default="54,108")
    parser.add_argument("--fit-targets", type=str, default="full_table,oblique_residual,axis_plus_residual")
    parser.add_argument("--ridge-grid", type=str, default="1e-8,1e-6,1e-4,1e-2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v12_e1_teacher_jetorder_projection_cifar10_10k")
    return parser.parse_args()


def per_center_atoms(order: int, family: str) -> int:
    if family == "full":
        return 2 * ((order + 1) * (order + 2) // 2)
    if family == "directional":
        return 2 + 8 * int(order)
    raise ValueError(f"unknown family: {family}")


def basis_matrix(
    d: torch.Tensor,
    omegas: torch.Tensor,
    *,
    family: str,
    order: int,
    scale: float,
    name: str,
) -> torch.Tensor:
    if family == "full":
        return full_multijet_basis(d, omegas, order=order, scale=scale, name=name).matrix
    if family == "directional":
        return directional_jet_basis(
            d,
            omegas,
            directions=default_directions(device=d.device, dtype=d.dtype),
            order=order,
            scale=scale,
            name=name,
        ).matrix
    raise ValueError(f"unknown basis family: {family}")


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, metadata = load_bias_npz(Path(args.teacher_bias), device=device)
    tables = bundle.tables.to(torch.float64)
    table_side = int(tables.shape[-1])
    side = (table_side + 1) // 2
    d = relative_grid(side, device=device, dtype=torch.float64)
    scale = float(max(1, side - 1))

    protocols = parse_csv_list(args.protocols)
    sources = parse_csv_list(args.frequency_sources)
    families = parse_csv_list(args.basis_families)
    orders = parse_int_list(args.orders)
    budgets = parse_int_list(args.matched_atom_budgets)
    fit_targets = parse_csv_list(args.fit_targets)
    ridge_grid = parse_float_list(args.ridge_grid)

    rows: list[dict[str, object]] = []
    ridge_rows: list[dict[str, object]] = []

    for layer in range(tables.shape[0]):
        for head in range(tables.shape[1]):
            table = tables[layer, head]
            for fit_target in fit_targets:
                target, base = table_target(table, fit_target)
                for protocol in protocols:
                    for source in sources:
                        for family in families:
                            protocol_budgets = [0] if protocol == "nested" else budgets
                            for budget in protocol_budgets:
                                for order in orders:
                                    if protocol == "nested":
                                        k = int(args.nested_centers)
                                        atom_budget = 1 + k * per_center_atoms(max(orders), family)
                                    elif protocol == "matched":
                                        k = max(1, (int(budget) - 1) // per_center_atoms(order, family))
                                        atom_budget = int(budget)
                                    else:
                                        raise ValueError(f"unknown protocol: {protocol}")
                                    omegas = select_omegas_from_source(
                                        source,
                                        target,
                                        k=k,
                                        seed=args.seed + 1000 * layer + 37 * head + 11 * order + int(budget),
                                    )
                                    basis_name = f"{protocol}_{source}_{family}_J{order}_k{k}_b{atom_budget}"
                                    matrix = basis_matrix(d, omegas, family=family, order=order, scale=scale, name=basis_name)
                                    fit, path = fit_with_ridge_grid(matrix, target.reshape(-1), ridge_grid=ridge_grid)
                                    pred_target = predict(matrix, fit.coeff, fit.column_norms).reshape_as(table)
                                    pred_table = pred_target + base
                                    row = {
                                        "dataset": metadata.get("dataset", "unknown"),
                                        "task": metadata.get("task", "unknown"),
                                        "teacher_basis": metadata.get("basis", "unknown"),
                                        "teacher_seed": metadata.get("seed", -1),
                                        "layer": layer,
                                        "head": head,
                                        "protocol": protocol,
                                        "frequency_source": source,
                                        "basis_family": family,
                                        "fit_target": fit_target,
                                        "order": order,
                                        "atom_budget": atom_budget,
                                        "num_centers": int(k),
                                        "num_features": int(matrix.shape[1]),
                                        "selected_ridge": fit.ridge,
                                        "target_fit_r2": fit.r2,
                                        "table_fit_r2": r2_score(pred_table, table),
                                        "residual_fit_r2": fit.r2 if fit_target != "full_table" else float("nan"),
                                        "delta_r2_prev_order": float("nan"),
                                        "condition_number": fit.condition_number,
                                        "effective_rank": fit.effective_rank,
                                        "coeff_norm": fit.coeff_norm,
                                        "singular_values": json.dumps(fit.singular_values),
                                        "omegas": omegas_json(omegas),
                                    }
                                    rows.append(row)
                                    for item in path:
                                        item.update(
                                            {
                                                "layer": layer,
                                                "head": head,
                                                "protocol": protocol,
                                                "frequency_source": source,
                                                "basis_family": family,
                                                "fit_target": fit_target,
                                                "order": order,
                                                "atom_budget": atom_budget,
                                                "num_centers": int(k),
                                                "num_features": int(matrix.shape[1]),
                                            }
                                        )
                                        ridge_rows.append(item)

    add_incremental_deltas(rows)
    aggregate = summarize_groups(
        rows,
        keys=["protocol", "frequency_source", "basis_family", "fit_target", "order", "atom_budget"],
        numeric=[
            "target_fit_r2",
            "table_fit_r2",
            "residual_fit_r2",
            "delta_r2_prev_order",
            "condition_number",
            "effective_rank",
            "coeff_norm",
            "num_centers",
            "num_features",
        ],
    )
    incremental = [row for row in rows if row["protocol"] == "nested" and int(row["order"]) > 0]
    write_csv(output_dir / "projection_results.csv", rows)
    write_csv(output_dir / "projection_aggregate.csv", aggregate)
    write_csv(output_dir / "incremental_by_head.csv", incremental)
    write_csv(output_dir / "ridge_path.csv", ridge_rows)
    plot_incremental_heatmaps(rows, output_dir)
    plot_pareto(aggregate, output_dir / "jetorder_pareto.pdf")

    summary = {
        "teacher_bias": args.teacher_bias,
        "metadata": metadata,
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "num_aggregate_rows": len(aggregate),
        "num_ridge_rows": len(ridge_rows),
        "protocols": protocols,
        "frequency_sources": sources,
        "basis_families": families,
        "orders": orders,
        "matched_atom_budgets": budgets,
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, aggregate)
    return summary


def add_incremental_deltas(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    keys = ["layer", "head", "protocol", "frequency_source", "basis_family", "fit_target", "atom_budget"]
    for row in rows:
        if row["protocol"] != "nested":
            continue
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    for group in grouped.values():
        by_order = {int(row["order"]): row for row in group}
        for order, row in by_order.items():
            if order - 1 in by_order:
                metric = "table_fit_r2" if row["fit_target"] == "full_table" else "target_fit_r2"
                row["delta_r2_prev_order"] = float(row[metric]) - float(by_order[order - 1][metric])


def plot_incremental_heatmaps(rows: list[dict[str, object]], output_dir: Path) -> None:
    candidates = [
        row
        for row in rows
        if row["protocol"] == "nested"
        and row["frequency_source"] == "table_informed"
        and row["basis_family"] == "full"
        and row["fit_target"] == "axis_plus_residual"
        and int(row["order"]) in {1, 2}
    ]
    if not candidates:
        return
    for order in [1, 2]:
        vals = [row for row in candidates if int(row["order"]) == order]
        if not vals:
            continue
        max_layer = max(int(row["layer"]) for row in vals)
        max_head = max(int(row["head"]) for row in vals)
        mat = np.full((max_layer + 1, max_head + 1), np.nan)
        for row in vals:
            mat[int(row["layer"]), int(row["head"])] = float(row["delta_r2_prev_order"])
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(f"Nested full table-informed Delta R2 J{order}")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        fig.tight_layout()
        fig.savefig(output_dir / f"nested_delta_J{order}_heatmap.pdf")
        plt.close(fig)


def plot_pareto(aggregate: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for protocol in sorted({str(row["protocol"]) for row in aggregate}):
        vals = [row for row in aggregate if row["protocol"] == protocol and row["fit_target"] == "axis_plus_residual"]
        if not vals:
            continue
        x = [float(row.get("num_features_mean", np.nan)) for row in vals]
        y = [float(row.get("table_fit_r2_mean", np.nan)) for row in vals]
        ax.scatter(x, y, label=protocol, alpha=0.75)
    ax.set_xlabel("features")
    ax.set_ylabel("table fit R2")
    ax.set_title("Teacher projection Pareto")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V12 E1 Teacher Jet-Order Projection",
        "",
        f"Teacher: `{summary['teacher_bias']}`",
        f"Rows: {summary['num_rows']}",
        "",
        "## Aggregate",
        "",
        "| protocol | source | family | target | order | budget | n | table R2 | target R2 | delta prev | cond | coeff norm |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + f"{row['protocol']} | {row['frequency_source']} | {row['basis_family']} | {row['fit_target']} | "
            + f"{int(row['order'])} | {int(row['atom_budget'])} | {int(row['n'])} | "
            + f"{float(row.get('table_fit_r2_mean', float('nan'))):.4f} | "
            + f"{float(row.get('target_fit_r2_mean', float('nan'))):.4f} | "
            + f"{float(row.get('delta_r2_prev_order_mean', float('nan'))):.4f} | "
            + f"{float(row.get('condition_number_mean', float('nan'))):.2e} | "
            + f"{float(row.get('coeff_norm_mean', float('nan'))):.2e} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `projection_results.csv`",
            "- `projection_aggregate.csv`",
            "- `incremental_by_head.csv`",
            "- `ridge_path.csv`",
            "- `nested_delta_J1_heatmap.pdf`",
            "- `nested_delta_J2_heatmap.pdf`",
            "- `jetorder_pareto.pdf`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
