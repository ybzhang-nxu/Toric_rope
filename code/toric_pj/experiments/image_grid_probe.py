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
    normalize_columns,
    phase,
    toric_fourier_basis,
)
from toric_pj.diagnostics.direction_alignment import normalize_direction


def make_patch_displacements(side: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(-(side - 1), side, device=device, dtype=torch.float64)
    dx, dy = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((dx.reshape(-1), dy.reshape(-1)), dim=-1)


def image_teacher(d: torch.Tensor, side: int) -> torch.Tensor:
    scale = float(side)
    omega_texture = torch.tensor([0.78, 0.42], device=d.device, dtype=d.dtype)
    omega_diag = torch.tensor([0.62, -0.58], device=d.device, dtype=d.dtype)
    u = normalize_direction(torch.tensor([[1.0, -0.65]], device=d.device, dtype=d.dtype)).reshape(-1)
    radius2 = d[:, 0].square() + d[:, 1].square()
    local = torch.exp(-radius2 / (2.0 * (0.22 * scale) ** 2))
    jet = ((d @ u) / scale) * torch.cos(phase(d, omega_texture))
    return (
        0.42 * torch.cos(phase(d, omega_texture))
        + 0.28 * torch.cos(phase(d, omega_diag))
        + 0.18 * jet
        + 0.24 * local
        - 0.08 * torch.sqrt(radius2) / scale
    )


def raster_1d_basis(d: torch.Tensor, side: int) -> Basis:
    lag = d[:, 0] * side + d[:, 1]
    freqs = [0.09, 0.17, 0.31, 0.57]
    cols = [torch.ones_like(lag)]
    labels = ["const"]
    orders = [0]
    for idx, freq in enumerate(freqs):
        cols.extend([torch.cos(freq * lag), torch.sin(freq * lag)])
        labels.extend([f"raster_cos{idx}", f"raster_sin{idx}"])
        orders.extend([0, 0])
    matrix, _ = normalize_columns(torch.stack(cols, dim=1))
    return Basis("raster_1d", matrix, labels, orders)


def no_pos_basis(d: torch.Tensor) -> Basis:
    matrix = torch.ones((d.shape[0], 1), device=d.device, dtype=d.dtype)
    return Basis("no_2d_positional", matrix, ["const"], [0])


def relative_2d_table_basis(d: torch.Tensor) -> Basis:
    unique, inverse = torch.unique(d.to(torch.long), dim=0, return_inverse=True)
    matrix = torch.zeros((d.shape[0], unique.shape[0]), device=d.device, dtype=d.dtype)
    matrix[torch.arange(d.shape[0], device=d.device), inverse] = 1.0
    matrix, _ = normalize_columns(matrix)
    labels = [f"rel_{int(x)}_{int(y)}" for x, y in unique.detach().cpu().tolist()]
    return Basis("relative_2d_table", matrix, labels, [0] * len(labels))


def lc_basis(d: torch.Tensor, side: int, omegas: list[torch.Tensor], direction: torch.Tensor) -> Basis:
    scale = float(side)
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    phi = scale * torch.asinh(d / scale)
    raw = d @ direction
    beta = raw / torch.sqrt(raw.square() + scale**2)
    for idx, omega in enumerate(omegas):
        ph = phase(phi, omega)
        cols.extend([torch.cos(ph), torch.sin(ph)])
        labels.extend([f"lc_w{idx}_r0_cos", f"lc_w{idx}_r0_sin"])
        orders.extend([0, 0])
        for order in [1, 2]:
            poly = beta.pow(order)
            cols.extend([poly * torch.cos(ph), poly * torch.sin(ph)])
            labels.extend([f"lc_w{idx}_r{order}_cos", f"lc_w{idx}_r{order}_sin"])
            orders.extend([order, order])
    matrix, _ = normalize_columns(torch.stack(cols, dim=1))
    return Basis("toric_LC_PJ", matrix, labels, orders)


def build_bases(d: torch.Tensor, side: int, seed: int) -> list[Basis]:
    omega_texture = torch.tensor([0.78, 0.42], device=d.device, dtype=d.dtype)
    omega_diag = torch.tensor([0.62, -0.58], device=d.device, dtype=d.dtype)
    omega_extra = torch.tensor([0.21, 0.84], device=d.device, dtype=d.dtype)
    u = normalize_direction(torch.tensor([[1.0, -0.65]], device=d.device, dtype=d.dtype)).reshape(-1)
    ex = torch.tensor([1.0, 0.0], device=d.device, dtype=d.dtype)
    ey = torch.tensor([0.0, 1.0], device=d.device, dtype=d.dtype)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1)

    shuffled = d.clone()
    gen = torch.Generator(device=d.device)
    gen.manual_seed(seed)
    shuffled = shuffled[torch.randperm(shuffled.shape[0], device=d.device, generator=gen)]

    return [
        no_pos_basis(d),
        raster_1d_basis(d, side),
        axis_additive_fourier_basis(d, [float(omega_texture[0]), float(omega_texture[1])], name="axial_additive"),
        relative_2d_table_basis(d),
        toric_fourier_basis(d, [omega_texture, omega_diag, omega_extra], name="toric_RoPE_order0"),
        directional_jet_basis(
            d,
            [omega_texture, omega_diag, omega_extra],
            [ex, ey, u, diag],
            [0, 1, 2],
            scale=float(side),
            name="toric_PJ_R2",
        ),
        lc_basis(d, side, [omega_texture, omega_diag, omega_extra], u),
        directional_jet_basis(
            shuffled,
            [omega_texture, omega_diag, omega_extra],
            [ex, ey, u, diag],
            [0, 1, 2],
            scale=float(side),
            name="toric_PJ_R2_coord_shuffle",
        ),
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    d = make_patch_displacements(args.side, device)
    target = image_teacher(d, args.side)
    rows = []
    for basis in build_bases(d, args.side, args.seed):
        result = fit_basis(basis, target, ridge=args.ridge)
        rows.append(
            {
                "basis": basis.name,
                "mse": result.mse,
                "r2": result.r2,
                "condition": result.condition,
                "num_features": basis.matrix.shape[1],
                "top_energy_order": result.top_energy_order,
                "top_loo_order": result.top_loo_order,
                "order_energy": json.dumps(result.order_energy, sort_keys=True),
                "order_loo": json.dumps(result.order_loo, sort_keys=True),
            }
        )
    csv_path = output_dir / "image_grid_probe_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = output_dir / "image_grid_probe_r2.png"
    plot_results(rows, plot_path)
    summary = {
        "device": str(device),
        "side": args.side,
        "num_displacements": int(d.shape[0]),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "image_grid_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["basis"]) for row in rows]
    r2 = [float(row["r2"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(np.arange(len(labels)), r2, color="#5d7f3f")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R^2")
    ax.set_title("Image / Patch Grid Toric Probe")
    for i, value in enumerate(r2):
        ax.text(i, min(value + 0.02, 1.02), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image / patch-grid toric probe.")
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage5_image")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
