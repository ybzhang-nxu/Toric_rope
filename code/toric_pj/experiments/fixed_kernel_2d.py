from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import (
    Basis,
    axis_additive_fourier_basis,
    default_device,
    directional_jet_basis,
    fit_basis,
    full_multijet_basis,
    make_grid_2d,
    phase,
    separable_product_fourier_basis,
    toric_fourier_basis,
)
from toric_pj.diagnostics.direction_alignment import normalize_direction


def build_targets(d: torch.Tensor, omega: torch.Tensor, u_star: torch.Tensor, scale: float) -> dict[str, torch.Tensor]:
    ph = phase(d, omega)
    cos_ph = torch.cos(ph)
    return {
        "A1_oblique_fourier": cos_ph,
        "A2x_coordinate_first_jet": (d[:, 0] / scale) * cos_ph,
        "A2y_coordinate_first_jet": (d[:, 1] / scale) * cos_ph,
        "A3_directional_first_jet": ((d @ u_star) / scale) * cos_ph,
        "A4_mixed_second_jet": (d[:, 0] / scale) * (d[:, 1] / scale) * cos_ph,
    }


def build_bases(d: torch.Tensor, omega: torch.Tensor, u_star: torch.Tensor, scale: float) -> list[Basis]:
    dtype = d.dtype
    device = d.device
    ex = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    ey = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)).reshape(-1)
    anti = normalize_direction(torch.tensor([[1.0, -1.0]], device=device, dtype=dtype)).reshape(-1)
    return [
        axis_additive_fourier_basis(d, [float(omega[0]), float(omega[1])], name="axis_additive"),
        separable_product_fourier_basis(d, omega, name="separable_product"),
        toric_fourier_basis(d, [omega], name="toric_order0"),
        directional_jet_basis(
            d,
            [omega],
            [ex, ey],
            [0, 1],
            scale=scale,
            name="directional_R1_axes",
        ),
        directional_jet_basis(
            d,
            [omega],
            [u_star],
            [0, 1],
            scale=scale,
            name="directional_R1_star",
        ),
        directional_jet_basis(
            d,
            [omega],
            [diag],
            [0, 1, 2],
            scale=scale,
            name="directional_R2_single_diag",
        ),
        directional_jet_basis(
            d,
            [omega],
            [ex, ey, diag, anti],
            [0, 1, 2],
            scale=scale,
            name="directional_R2_multi",
        ),
        full_multijet_basis(d, omega, max_order=2, scale=scale, name="full_multijet_R2"),
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    dtype = torch.float64
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d = make_grid_2d(args.radius, signed=True, device=device, dtype=dtype)
    omega = torch.tensor([args.omega_x, args.omega_y], device=device, dtype=dtype)
    u_star = normalize_direction(torch.tensor([[args.u_x, args.u_y]], device=device, dtype=dtype)).reshape(-1)
    targets = build_targets(d, omega, u_star, scale=float(args.radius))
    bases = build_bases(d, omega, u_star, scale=float(args.radius))

    rows: list[dict[str, object]] = []
    for target_name, target in targets.items():
        for basis in bases:
            result = fit_basis(basis, target, ridge=args.ridge)
            rows.append(
                {
                    "target": target_name,
                    "basis": basis.name,
                    "mse": result.mse,
                    "r2": result.r2,
                    "condition": result.condition,
                    "coeff_norm": result.coeff_norm,
                    "top_energy_order": result.top_energy_order,
                    "top_loo_order": result.top_loo_order,
                    "order_energy": json.dumps(result.order_energy, sort_keys=True),
                    "order_loo": json.dumps(result.order_loo, sort_keys=True),
                }
            )

    csv_path = output_dir / "fixed_kernel_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = output_dir / "fixed_kernel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_r2_heatmap(rows, output_dir / "fixed_kernel_r2_heatmap.png")

    return {
        "device": str(device),
        "num_points": int(d.shape[0]),
        "csv": str(csv_path),
        "summary": str(summary_path),
        "plot": str(output_dir / "fixed_kernel_r2_heatmap.png"),
        "best_by_target": summary["best_by_target"],
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    best_by_target: dict[str, dict[str, object]] = {}
    for row in rows:
        target = str(row["target"])
        if target not in best_by_target or _is_better(row, best_by_target[target]):
            best_by_target[target] = row
    return {"best_by_target": best_by_target}


def _is_better(row: dict[str, object], incumbent: dict[str, object]) -> bool:
    r2 = float(row["r2"])
    old_r2 = float(incumbent["r2"])
    if r2 > old_r2 + 1e-9:
        return True
    if r2 < old_r2 - 1e-9:
        return False
    condition = float(row["condition"])
    old_condition = float(incumbent["condition"])
    if np.isfinite(condition) and not np.isfinite(old_condition):
        return True
    if not np.isfinite(condition) and np.isfinite(old_condition):
        return False
    return condition < old_condition


def plot_r2_heatmap(rows: list[dict[str, object]], path: Path) -> None:
    targets = sorted({str(row["target"]) for row in rows})
    bases = sorted({str(row["basis"]) for row in rows})
    values = np.full((len(targets), len(bases)), np.nan)
    for row in rows:
        values[targets.index(str(row["target"])), bases.index(str(row["basis"]))] = float(row["r2"])

    fig, ax = plt.subplots(figsize=(max(8, len(bases) * 1.3), max(4, len(targets) * 0.8)))
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(bases)), labels=bases, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(targets)), labels=targets)
    ax.set_title("Fixed-kernel containment: R^2")
    for i in range(len(targets)):
        for j in range(len(bases)):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D fixed-kernel Toric PJ containment experiments.")
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--omega-x", type=float, default=0.37)
    parser.add_argument("--omega-y", type=float, default=0.61)
    parser.add_argument("--u-x", type=float, default=1.0)
    parser.add_argument("--u-y", type=float, default=-0.6)
    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage1")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
