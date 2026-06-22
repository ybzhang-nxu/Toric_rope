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
from sklearn.datasets import load_digits

from toric_pj.diagnostics.basis_projection import (
    Basis,
    axis_additive_fourier_basis,
    default_device,
    directional_jet_basis,
    normalize_columns,
    phase,
    toric_fourier_basis,
)
from toric_pj.diagnostics.direction_alignment import normalize_direction


def make_positions(side: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(side, device=device, dtype=torch.float64)
    xx, yy = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def pairwise_d(positions: torch.Tensor) -> torch.Tensor:
    return positions[:, None, :] - positions[None, :, :]


def load_digit_tensors(device: torch.device, *, seed: int, train_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = load_digits()
    images = torch.tensor(data.data, device=device, dtype=torch.float64) / 16.0
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    perm = torch.randperm(images.shape[0], device=device, generator=gen)
    train_idx = perm[:train_count]
    test_idx = perm[train_count:]
    train = images[train_idx]
    test = images[test_idx]
    mean = train.mean(dim=0, keepdim=True)
    std = train.std().clamp_min(1e-6)
    return (train - mean) / std, (test - mean) / std


def no_pos_basis(d: torch.Tensor) -> Basis:
    matrix = torch.ones((d.shape[0], 1), device=d.device, dtype=d.dtype)
    return Basis("no_pos_constant", matrix, ["const"], [0])


def raster_1d_basis(d: torch.Tensor, side: int) -> Basis:
    lag = d[:, 0] * side + d[:, 1]
    cols = [torch.ones_like(lag)]
    labels = ["const"]
    orders = [0]
    for idx, freq in enumerate([0.21, 0.47, 0.83, 1.31]):
        cols.extend([torch.cos(freq * lag), torch.sin(freq * lag)])
        labels.extend([f"raster_cos{idx}", f"raster_sin{idx}"])
        orders.extend([0, 0])
    matrix, _ = normalize_columns(torch.stack(cols, dim=1))
    return Basis("raster_1d", matrix, labels, orders)


def relative_2d_table_basis(d: torch.Tensor) -> Basis:
    unique, inverse = torch.unique(d.to(torch.long), dim=0, return_inverse=True)
    matrix = torch.zeros((d.shape[0], unique.shape[0]), device=d.device, dtype=d.dtype)
    matrix[torch.arange(d.shape[0], device=d.device), inverse] = 1.0
    matrix, _ = normalize_columns(matrix)
    labels = [f"rel_{int(x)}_{int(y)}" for x, y in unique.detach().cpu().tolist()]
    return Basis("relative_2d_table", matrix, labels, [0] * len(labels))


def lc_basis(d: torch.Tensor, side: int, omegas: list[torch.Tensor], direction: torch.Tensor) -> Basis:
    scale = float(side)
    phi = scale * torch.asinh(d / scale)
    raw = d @ direction
    beta = raw / torch.sqrt(raw.square() + scale**2)
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
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
    return Basis("toric_LC_PJ_R2", matrix, labels, orders)


def build_bases(d: torch.Tensor, side: int, seed: int) -> list[Basis]:
    omega_a = torch.tensor([0.78, 0.42], device=d.device, dtype=d.dtype)
    omega_b = torch.tensor([0.62, -0.58], device=d.device, dtype=d.dtype)
    omega_c = torch.tensor([0.35, 0.91], device=d.device, dtype=d.dtype)
    ex = torch.tensor([1.0, 0.0], device=d.device, dtype=d.dtype)
    ey = torch.tensor([0.0, 1.0], device=d.device, dtype=d.dtype)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1)
    oblique = normalize_direction(torch.tensor([[1.0, -0.65]], device=d.device, dtype=d.dtype)).reshape(-1)

    gen = torch.Generator(device=d.device)
    gen.manual_seed(seed)
    shuffled = d[torch.randperm(d.shape[0], device=d.device, generator=gen)]

    return [
        no_pos_basis(d),
        raster_1d_basis(d, side),
        axis_additive_fourier_basis(d, [0.78, 0.58], name="axis_additive"),
        toric_fourier_basis(d, [omega_a, omega_b, omega_c], name="toric_order0"),
        directional_jet_basis(
            d,
            [omega_a, omega_b, omega_c],
            [ex, ey, diag, oblique],
            [0, 1, 2],
            scale=float(side),
            name="toric_PJ_R2",
        ),
        lc_basis(d, side, [omega_a, omega_b, omega_c], oblique),
        relative_2d_table_basis(d),
        directional_jet_basis(
            shuffled,
            [omega_a, omega_b, omega_c],
            [ex, ey, diag, oblique],
            [0, 1, 2],
            scale=float(side),
            name="toric_PJ_R2_coord_shuffle",
        ),
    ]


def design_matrix(images: torch.Tensor, basis: Basis, *, n_positions: int) -> torch.Tensor:
    b = basis.matrix.reshape(n_positions, n_positions, basis.matrix.shape[1])
    eye = torch.eye(n_positions, device=images.device, dtype=images.dtype).reshape(n_positions, n_positions, 1)
    b = b * (1.0 - eye)
    return torch.einsum("sk,qkm->sqm", images, b).reshape(images.shape[0] * n_positions, basis.matrix.shape[1])


def fit_reconstruction(
    basis: Basis,
    train: torch.Tensor,
    test: torch.Tensor,
    *,
    ridge: float,
) -> dict[str, object]:
    n_positions = train.shape[1]
    x_train, norms = normalize_columns(design_matrix(train, basis, n_positions=n_positions))
    x_test = design_matrix(test, basis, n_positions=n_positions) / norms.clamp_min(1e-12)
    y_train = train.reshape(-1, 1)
    y_test = test.reshape(-1, 1)
    gram = x_train.T @ x_train
    rhs = x_train.T @ y_train
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coeff = torch.linalg.solve(gram + ridge * eye, rhs)
    pred_train = x_train @ coeff
    pred_test = x_test @ coeff
    train_mse = torch.mean((pred_train - y_train).square())
    test_mse = torch.mean((pred_test - y_test).square())
    train_var = torch.mean((y_train - y_train.mean()).square()).clamp_min(1e-30)
    test_var = torch.mean((y_test - y_test.mean()).square()).clamp_min(1e-30)
    return {
        "basis": basis.name,
        "train_r2": float((1.0 - train_mse / train_var).detach().cpu()),
        "test_r2": float((1.0 - test_mse / test_var).detach().cpu()),
        "train_mse": float(train_mse.detach().cpu()),
        "test_mse": float(test_mse.detach().cpu()),
        "num_features": basis.matrix.shape[1],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_digit_tensors(device, seed=args.seed, train_count=args.train_count)
    positions = make_positions(args.side, device)
    d = pairwise_d(positions).reshape(-1, 2)
    rows = [fit_reconstruction(basis, train, test, ridge=args.ridge) for basis in build_bases(d, args.side, args.seed)]

    csv_path = output_dir / "real_digits_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "real_digits_r2.png"
    plot_results(rows, plot_path)
    summary = {
        "device": str(device),
        "dataset": "sklearn_digits",
        "side": args.side,
        "train_count": int(train.shape[0]),
        "test_count": int(test.shape[0]),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "real_digits_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["basis"]) for row in rows]
    values = [float(row["test_r2"]) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x, values, color="#5d7645")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, max(1.0, max(values) + 0.05))
    ax.set_ylabel("test R^2")
    ax.set_title("Real Digits Masked Pixel Reconstruction")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real sklearn-digits masked pixel reconstruction probe.")
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=1400)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage5_real_digits")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
