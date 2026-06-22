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

from toric_pj.diagnostics.basis_projection import default_device, make_grid_2d
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.diagnostics.positional_ntk import (
    build_projected_pj_tangent_bank,
    effective_rank,
    grouped_kernel_metrics,
    kernel_from_features,
    matrix_condition,
)
from toric_pj.experiments.adaptive_teacher_2d import BRANCHES, build_targets


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    dtype = torch.float64
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d = make_grid_2d(args.radius, signed=True, device=device, dtype=dtype)
    omega_main = torch.tensor([0.37, 0.61], device=device, dtype=dtype)
    omega_beat = torch.tensor([0.0, 2.0 * math.pi / 8.0], device=device, dtype=dtype)
    omega_lc = 0.55
    ex = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    ey = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
    u_star = normalize_direction(torch.tensor([[1.0, -0.6]], device=device, dtype=dtype)).reshape(-1)
    s_star = normalize_direction(torch.tensor([[0.75, 0.35]], device=device, dtype=dtype)).reshape(-1)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)).reshape(-1)
    directions = [ex, ey, u_star, s_star, diag]

    targets = build_targets(
        d,
        radius=args.radius,
        omega_main=omega_main,
        omega_beat=omega_beat,
        u_star=u_star,
        s_star=s_star,
        omega_lc=omega_lc,
    )
    setups = {
        "amplitude_R2": False,
        "spectral_learnable_R2": True,
    }

    rows: list[dict[str, object]] = []
    setup_summaries: dict[str, dict[str, object]] = {}
    for setup_name, include_spectral_tangents in setups.items():
        bank = build_projected_pj_tangent_bank(
            d,
            radius=args.radius,
            omegas=[omega_main, omega_beat],
            directions=directions,
            max_order=2,
            include_spectral_tangents=include_spectral_tangents,
            include_lc=True,
            lc_omega=omega_lc,
        )
        kernel = kernel_from_features(bank.features)
        setup_summaries[setup_name] = {
            "num_features": int(bank.features.shape[1]),
            "effective_rank": effective_rank(kernel),
            "condition": matrix_condition(bank.features),
            "max_order_present": max(meta.order for meta in bank.metas),
        }
        for group_by in ["sector", "order", "source"]:
            for row in grouped_kernel_metrics(bank, targets, group_by=group_by):
                rows.append({"setup": setup_name, **row})

    csv_path = output_dir / "pj_ntk_group_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)

    alignment_rows = top_sector_alignment_rows(rows, targets)
    learned_comparison = compare_with_stage2(alignment_rows, Path(args.stage2_summary))
    alignment_path = output_dir / "pj_ntk_alignment_summary.csv"
    with alignment_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(alignment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(alignment_rows)

    plot_path = output_dir / "pj_ntk_sector_alignment.png"
    plot_sector_alignment(rows, plot_path)
    order_plot_path = output_dir / "pj_ntk_order_energy.png"
    plot_order_energy(rows, order_plot_path)

    summary = {
        "device": str(device),
        "num_points": int(d.shape[0]),
        "setups": setup_summaries,
        "metrics_csv": str(csv_path),
        "alignment_csv": str(alignment_path),
        "sector_alignment_plot": str(plot_path),
        "order_energy_plot": str(order_plot_path),
        "top_sector_alignment": alignment_rows,
        "stage2_comparison": learned_comparison,
    }
    summary_path = output_dir / "pj_ntk_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def top_sector_alignment_rows(
    rows: list[dict[str, object]], targets: dict[str, torch.Tensor]
) -> list[dict[str, object]]:
    sector_rows = [row for row in rows if row["group_by"] == "sector"]
    output: list[dict[str, object]] = []
    for setup in sorted({str(row["setup"]) for row in sector_rows}):
        subset = [row for row in sector_rows if row["setup"] == setup]
        for target_name in targets:
            align_key = f"align:{target_name}"
            best = max(subset, key=lambda row: float(row[align_key]))
            output.append(
                {
                    "setup": setup,
                    "target": target_name,
                    "top_ntk_sector": best["group"],
                    "top_ntk_alignment": float(best[align_key]),
                    "FJ_alignment": _alignment_for_group(subset, "FJ", align_key),
                    "affine_alignment": _alignment_for_group(subset, "affine", align_key),
                    "LC_alignment": _alignment_for_group(subset, "LC", align_key),
                }
            )
    return output


def _alignment_for_group(rows: list[dict[str, object]], group: str, align_key: str) -> float:
    for row in rows:
        if row["group"] == group:
            return float(row[align_key])
    return float("nan")


def compare_with_stage2(alignment_rows: list[dict[str, object]], stage2_summary_path: Path) -> dict[str, object]:
    if not stage2_summary_path.exists():
        return {"available": False, "reason": f"{stage2_summary_path} not found"}
    stage2 = json.loads(stage2_summary_path.read_text(encoding="utf-8"))
    learned = {
        row["target"]: {
            "top_branch_energy": row["top_branch_energy"],
            "top_branch_loo": row["top_branch_loo"],
        }
        for row in stage2["rows"]
    }
    comparisons = []
    for row in alignment_rows:
        target = row["target"]
        if target not in learned:
            continue
        comparisons.append(
            {
                **row,
                "learned_top_branch_energy": learned[target]["top_branch_energy"],
                "learned_top_branch_loo": learned[target]["top_branch_loo"],
                "matches_learned_loo": row["top_ntk_sector"] == learned[target]["top_branch_loo"],
            }
        )
    match_rate = (
        sum(1 for row in comparisons if row["matches_learned_loo"]) / len(comparisons)
        if comparisons
        else None
    )
    return {"available": True, "match_rate_vs_learned_loo": match_rate, "rows": comparisons}


def plot_sector_alignment(rows: list[dict[str, object]], path: Path) -> None:
    sector_rows = [row for row in rows if row["group_by"] == "sector" and row["setup"] == "spectral_learnable_R2"]
    targets = [key.removeprefix("align:") for key in sector_rows[0] if key.startswith("align:")]
    sectors = list(BRANCHES)
    values = np.zeros((len(targets), len(sectors)))
    for i, target in enumerate(targets):
        for j, sector in enumerate(sectors):
            values[i, j] = _alignment_for_group(sector_rows, sector, f"align:{target}")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(values, vmin=0.0, vmax=max(1.0, float(np.nanmax(values))), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(sectors)), sectors)
    ax.set_yticks(np.arange(len(targets)), targets)
    ax.set_title("PJ-NTK Kernel-Target Alignment by Sector")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_order_energy(rows: list[dict[str, object]], path: Path) -> None:
    order_rows = [row for row in rows if row["group_by"] == "order"]
    setups = sorted({str(row["setup"]) for row in order_rows})
    orders = sorted({int(row["group"]) for row in order_rows})
    values = np.zeros((len(setups), len(orders)))
    for i, setup in enumerate(setups):
        for j, order in enumerate(orders):
            match = [row for row in order_rows if row["setup"] == setup and int(row["group"]) == order]
            values[i, j] = float(match[0]["kernel_energy"]) if match else 0.0

    fig, ax = plt.subplots(figsize=(7, 3.8))
    im = ax.imshow(values, vmin=0.0, vmax=max(1.0, float(np.nanmax(values))), cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(orders)), [f"r{order}" for order in orders])
    ax.set_yticks(np.arange(len(setups)), setups)
    ax.set_title("Projected PJ-NTK Energy by Diagnostic Order")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run projected PJ-NTK diagnostics on a small 2D grid.")
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage3")
    parser.add_argument("--stage2-summary", type=str, default="results/stage2/adaptive_teacher_summary.json")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
