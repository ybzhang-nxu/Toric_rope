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


def make_positions(side: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(side, device=device, dtype=torch.float64)
    xx, yy = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def pairwise_displacements(positions: torch.Tensor) -> torch.Tensor:
    return positions[:, None, :] - positions[None, :, :]


def wave_kernel_teacher(d: torch.Tensor, side: int) -> torch.Tensor:
    scale = float(side)
    omega_wave = torch.tensor([0.73, 0.41], device=d.device, dtype=d.dtype)
    omega_shear = torch.tensor([0.47, -0.66], device=d.device, dtype=d.dtype)
    u = normalize_direction(torch.tensor([[1.0, -0.45]], device=d.device, dtype=d.dtype)).reshape(-1)
    v = normalize_direction(torch.tensor([[0.35, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1)
    x = (d @ u) / scale
    y = (d @ v) / scale
    wave_phase = phase(d, omega_wave)
    shear_phase = phase(d, omega_shear)
    return (
        0.48 * torch.cos(wave_phase)
        + 0.26 * x * torch.cos(wave_phase)
        - 0.18 * x.square() * torch.sin(wave_phase)
        + 0.20 * torch.cos(shear_phase)
        + 0.12 * y * torch.sin(shear_phase)
    )


def no_pos_basis(d: torch.Tensor) -> Basis:
    matrix = torch.ones((d.shape[0], 1), device=d.device, dtype=d.dtype)
    return Basis("no_pos_constant", matrix, ["const"], [0])


def raster_1d_basis(d: torch.Tensor, side: int) -> Basis:
    lag = d[:, 0] * side + d[:, 1]
    cols = [torch.ones_like(lag)]
    labels = ["const"]
    orders = [0]
    for idx, freq in enumerate([0.07, 0.15, 0.33, 0.61]):
        cols.extend([torch.cos(freq * lag), torch.sin(freq * lag)])
        labels.extend([f"raster_cos{idx}", f"raster_sin{idx}"])
        orders.extend([0, 0])
    matrix, _ = normalize_columns(torch.stack(cols, dim=1))
    return Basis("raster_1d", matrix, labels, orders)


def separable_order0_basis(d: torch.Tensor, omegas: list[torch.Tensor]) -> Basis:
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    for idx, omega in enumerate(omegas):
        omega = omega.to(device=d.device, dtype=d.dtype)
        ax = d[:, 0] * omega[0]
        ay = d[:, 1] * omega[1]
        cols.extend(
            [
                torch.cos(ax) * torch.cos(ay),
                torch.sin(ax) * torch.sin(ay),
                torch.sin(ax) * torch.cos(ay),
                torch.cos(ax) * torch.sin(ay),
            ]
        )
        labels.extend(
            [
                f"sep{idx}_cosx_cosy",
                f"sep{idx}_sinx_siny",
                f"sep{idx}_sinx_cosy",
                f"sep{idx}_cosx_siny",
            ]
        )
        orders.extend([0, 0, 0, 0])
    matrix, _ = normalize_columns(torch.stack(cols, dim=1))
    return Basis("separable_order0", matrix, labels, orders)


def relative_2d_table_basis(d: torch.Tensor) -> Basis:
    unique, inverse = torch.unique(d.to(torch.long), dim=0, return_inverse=True)
    matrix = torch.zeros((d.shape[0], unique.shape[0]), device=d.device, dtype=d.dtype)
    matrix[torch.arange(d.shape[0], device=d.device), inverse] = 1.0
    matrix, _ = normalize_columns(matrix)
    labels = [f"rel_{int(x)}_{int(y)}" for x, y in unique.detach().cpu().tolist()]
    return Basis("relative_2d_table", matrix, labels, [0] * len(labels))


def lc_pj_basis(d: torch.Tensor, side: int, omegas: list[torch.Tensor], directions: list[torch.Tensor]) -> Basis:
    scale = float(side)
    phi = scale * torch.asinh(d / scale)
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    for omega_idx, omega in enumerate(omegas):
        ph = phase(phi, omega)
        cols.extend([torch.cos(ph), torch.sin(ph)])
        labels.extend([f"lc_w{omega_idx}_r0_cos", f"lc_w{omega_idx}_r0_sin"])
        orders.extend([0, 0])
        for direction_idx, direction in enumerate(directions):
            direction = normalize_direction(direction.to(device=d.device, dtype=d.dtype).reshape(1, -1)).reshape(-1)
            raw = d @ direction
            beta = raw / torch.sqrt(raw.square() + scale**2)
            for order in [1, 2]:
                poly = beta.pow(order)
                cols.extend([poly * torch.cos(ph), poly * torch.sin(ph)])
                labels.extend(
                    [
                        f"lc_w{omega_idx}_u{direction_idx}_r{order}_cos",
                        f"lc_w{omega_idx}_u{direction_idx}_r{order}_sin",
                    ]
                )
                orders.extend([order, order])
    matrix, _ = normalize_columns(torch.stack(cols, dim=1))
    return Basis("toric_LC_PJ_R2", matrix, labels, orders)


def build_bases(d: torch.Tensor, side: int, seed: int) -> list[Basis]:
    omega_wave = torch.tensor([0.73, 0.41], device=d.device, dtype=d.dtype)
    omega_shear = torch.tensor([0.47, -0.66], device=d.device, dtype=d.dtype)
    omega_extra = torch.tensor([0.21, 0.88], device=d.device, dtype=d.dtype)
    ex = torch.tensor([1.0, 0.0], device=d.device, dtype=d.dtype)
    ey = torch.tensor([0.0, 1.0], device=d.device, dtype=d.dtype)
    u = normalize_direction(torch.tensor([[1.0, -0.45]], device=d.device, dtype=d.dtype)).reshape(-1)
    v = normalize_direction(torch.tensor([[0.35, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1)

    gen = torch.Generator(device=d.device)
    gen.manual_seed(seed)
    shuffled = d[torch.randperm(d.shape[0], device=d.device, generator=gen)]

    return [
        no_pos_basis(d),
        raster_1d_basis(d, side),
        axis_additive_fourier_basis(d, [float(omega_wave[0]), float(omega_wave[1])], name="axis_additive"),
        separable_order0_basis(d, [omega_wave, omega_shear]),
        toric_fourier_basis(d, [omega_wave, omega_shear, omega_extra], name="toric_order0"),
        directional_jet_basis(
            d,
            [omega_wave, omega_shear, omega_extra],
            [ex, ey, u, v],
            [0, 1, 2],
            scale=float(side),
            name="toric_PJ_R2",
        ),
        lc_pj_basis(d, side, [omega_wave, omega_shear, omega_extra], [ex, ey, u, v]),
        relative_2d_table_basis(d),
        directional_jet_basis(
            shuffled,
            [omega_wave, omega_shear, omega_extra],
            [ex, ey, u, v],
            [0, 1, 2],
            scale=float(side),
            name="toric_PJ_R2_coord_shuffle",
        ),
    ]


def fit_with_prediction(basis: Basis, target: torch.Tensor, ridge: float) -> tuple[dict[str, object], torch.Tensor]:
    result = fit_basis(basis, target, ridge=ridge)
    matrix, _ = normalize_columns(basis.matrix)
    y = target.reshape(-1, 1)
    gram = matrix.T @ matrix
    rhs = matrix.T @ y
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coeff = torch.linalg.solve(gram + ridge * eye, rhs)
    pred = (matrix @ coeff).reshape(-1)
    row = {
        "basis": basis.name,
        "kernel_mse": result.mse,
        "kernel_r2": result.r2,
        "condition": result.condition,
        "coeff_norm": result.coeff_norm,
        "num_features": basis.matrix.shape[1],
        "top_energy_order": result.top_energy_order,
        "top_loo_order": result.top_loo_order,
        "order_energy": json.dumps(result.order_energy, sort_keys=True),
        "order_loo": json.dumps(result.order_loo, sort_keys=True),
    }
    return row, pred


def field_next_state_r2(
    teacher_matrix: torch.Tensor,
    pred_matrix: torch.Tensor,
    *,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> float:
    device = teacher_matrix.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    n = teacher_matrix.shape[0]
    denom = torch.tensor(0.0, device=device, dtype=teacher_matrix.dtype)
    numer = torch.tensor(0.0, device=device, dtype=teacher_matrix.dtype)
    scale = float(n) ** 0.5
    for _ in range(eval_batches):
        fields = torch.randn(batch_size, n, device=device, dtype=teacher_matrix.dtype, generator=gen)
        target = fields @ teacher_matrix.T / scale
        pred = fields @ pred_matrix.T / scale
        numer = numer + torch.mean((pred - target).square())
        denom = denom + torch.mean(target.square())
    return float((1.0 - numer / denom.clamp_min(1e-30)).detach().cpu())


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = make_positions(args.side, device)
    d_pair = pairwise_displacements(positions)
    d_flat = d_pair.reshape(-1, 2)
    target = wave_kernel_teacher(d_flat, args.side)
    n_positions = positions.shape[0]
    teacher_matrix = target.reshape(n_positions, n_positions)

    rows: list[dict[str, object]] = []
    for basis in build_bases(d_flat, args.side, args.seed):
        row, pred = fit_with_prediction(basis, target, ridge=args.ridge)
        pred_matrix = pred.reshape(n_positions, n_positions)
        row["field_r2"] = field_next_state_r2(
            teacher_matrix,
            pred_matrix,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            seed=args.seed + len(rows),
        )
        rows.append(row)

    csv_path = output_dir / "lattice_physics_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "lattice_physics_r2.png"
    plot_results(rows, plot_path)
    summary = {
        "device": str(device),
        "side": args.side,
        "num_positions": int(n_positions),
        "num_pairs": int(d_flat.shape[0]),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "lattice_physics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["basis"]) for row in rows]
    kernel = [float(row["kernel_r2"]) for row in rows]
    field = [float(row["field_r2"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, kernel, width=width, label="Kernel R^2", color="#466a9f")
    ax.bar(x + width / 2, field, width=width, label="Next-state R^2", color="#9c6b35")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R^2")
    ax.set_title("Lattice Physics / Wave-Kernel Probe")
    ax.legend()
    for i, value in enumerate(kernel):
        ax.text(i - width / 2, min(value + 0.02, 1.02), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for i, value in enumerate(field):
        ax.text(i + width / 2, min(value + 0.02, 1.02), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lattice physics / wave-kernel toric probe.")
    parser.add_argument("--side", type=int, default=14)
    parser.add_argument("--ridge", type=float, default=1e-9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage5_lattice")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
