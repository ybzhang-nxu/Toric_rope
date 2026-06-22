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
    condition_number,
    default_device,
    directional_jet_basis,
    make_grid_2d,
    normalize_columns,
    phase,
)
from toric_pj.diagnostics.direction_alignment import normalize_direction


def damped_directional_jet_basis(
    d: torch.Tensor,
    *,
    omegas: list[torch.Tensor],
    directions: list[torch.Tensor],
    orders: list[int],
    scale: float,
    damping_direction: torch.Tensor,
    gamma: float,
    name: str,
) -> Basis:
    damping_direction = normalize_direction(damping_direction.reshape(1, -1)).reshape(-1)
    damp = torch.exp(-float(gamma) * (d @ damping_direction).clamp_min(0.0) / float(scale))
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    order_labels = [0]
    directions = [normalize_direction(direction.reshape(1, -1)).reshape(-1) for direction in directions]
    for omega_idx, omega in enumerate(omegas):
        ph = phase(d, omega)
        cos_ph = damp * torch.cos(ph)
        sin_ph = damp * torch.sin(ph)
        if 0 in orders:
            cols.extend([cos_ph, sin_ph])
            labels.extend([f"w{omega_idx}_r0_cos", f"w{omega_idx}_r0_sin"])
            order_labels.extend([0, 0])
        for direction_idx, direction in enumerate(directions):
            coord = (d @ direction) / float(scale)
            for order in orders:
                if order == 0:
                    continue
                poly = coord.pow(order)
                cols.extend([poly * cos_ph, poly * sin_ph])
                labels.extend(
                    [
                        f"w{omega_idx}_u{direction_idx}_r{order}_cos",
                        f"w{omega_idx}_u{direction_idx}_r{order}_sin",
                    ]
                )
                order_labels.extend([order, order])
    return Basis(name, torch.stack(cols, dim=1), labels, order_labels)


def lc_basis(
    d: torch.Tensor,
    *,
    directions: list[torch.Tensor],
    omega: float,
    scale: float,
    name: str,
) -> Basis:
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    for direction_idx, direction in enumerate(directions):
        direction = normalize_direction(direction.reshape(1, -1)).reshape(-1)
        raw = d @ direction
        phi = float(scale) * torch.asinh(raw / float(scale))
        beta = raw / torch.sqrt(raw.square() + float(scale) ** 2)
        ph = float(omega) * phi
        cols.extend([torch.cos(ph), torch.sin(ph)])
        labels.extend([f"u{direction_idx}_r0_cos", f"u{direction_idx}_r0_sin"])
        orders.extend([0, 0])
        for order in [1, 2]:
            poly = beta.pow(order)
            cols.extend([poly * torch.cos(ph), poly * torch.sin(ph)])
            labels.extend([f"u{direction_idx}_r{order}_cos", f"u{direction_idx}_r{order}_sin"])
            orders.extend([order, order])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def fit_train_eval(
    *,
    task: str,
    train_basis: Basis,
    eval_basis: Basis,
    train_target: torch.Tensor,
    eval_target: torch.Tensor,
    ridge: float,
) -> dict[str, object]:
    x_train, norms = normalize_columns(train_basis.matrix)
    x_eval = eval_basis.matrix / norms.clamp_min(1e-12)
    y_train = train_target.reshape(-1, 1)
    y_eval = eval_target.reshape(-1, 1)
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
        "task": task,
        "basis": train_basis.name,
        "train_r2": float((1.0 - train_mse / train_var).detach().cpu()),
        "eval_r2": float((1.0 - eval_mse / eval_var).detach().cpu()),
        "train_mse": float(train_mse.detach().cpu()),
        "eval_mse": float(eval_mse.detach().cpu()),
        "eval_pred_max_abs": float(torch.max(torch.abs(pred_eval)).detach().cpu()),
        "num_features": train_basis.matrix.shape[1],
        "condition": condition_number(x_train),
    }


def damping_rows(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    train = make_grid_2d(args.train_radius, signed=False, device=device, dtype=torch.float64)
    eval_grid = make_grid_2d(args.eval_radius, signed=False, device=device, dtype=torch.float64)
    omega = torch.tensor([0.31, 0.47], device=device, dtype=torch.float64)
    u = normalize_direction(torch.tensor([[0.9, 0.35]], device=device, dtype=torch.float64)).reshape(-1)
    s = normalize_direction(torch.tensor([[0.75, 0.55]], device=device, dtype=torch.float64)).reshape(-1)

    def target(d: torch.Tensor) -> torch.Tensor:
        x = (d @ u) / float(args.train_radius)
        damp = torch.exp(-args.gamma * (d @ s).clamp_min(0.0) / float(args.train_radius))
        return x.square() * damp * torch.cos(phase(d, omega))

    rows = []
    configs = [
        ("raw_no_damping", 0.0),
        ("dual_cone_damped", args.gamma),
        ("over_damped_control", args.gamma * 2.5),
        ("wrong_sign_control", -args.gamma),
    ]
    for name, gamma in configs:
        train_basis = damped_directional_jet_basis(
            train,
            omegas=[omega],
            directions=[u],
            orders=[0, 1, 2],
            scale=float(args.train_radius),
            damping_direction=s,
            gamma=gamma,
            name=name,
        )
        eval_basis = damped_directional_jet_basis(
            eval_grid,
            omegas=[omega],
            directions=[u],
            orders=[0, 1, 2],
            scale=float(args.train_radius),
            damping_direction=s,
            gamma=gamma,
            name=name,
        )
        rows.append(
            fit_train_eval(
                task="damping_constraint",
                train_basis=train_basis,
                eval_basis=eval_basis,
                train_target=target(train),
                eval_target=target(eval_grid),
                ridge=args.ridge,
            )
        )
    return rows


def spectral_rows(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    train = make_grid_2d(args.spectral_train_radius, signed=True, device=device, dtype=torch.float64)
    eval_grid = make_grid_2d(args.spectral_eval_radius, signed=True, device=device, dtype=torch.float64)
    omega_true = torch.tensor([0.57, -0.38], device=device, dtype=torch.float64)
    omega_aux = torch.tensor([0.22, 0.71], device=device, dtype=torch.float64)
    u = normalize_direction(torch.tensor([[1.0, -0.55]], device=device, dtype=torch.float64)).reshape(-1)
    ex = torch.tensor([1.0, 0.0], device=device, dtype=torch.float64)
    ey = torch.tensor([0.0, 1.0], device=device, dtype=torch.float64)

    def target(d: torch.Tensor) -> torch.Tensor:
        x = (d @ u) / float(args.spectral_train_radius)
        return torch.cos(phase(d, omega_true)) + 0.24 * x * torch.sin(phase(d, omega_true))

    configs = [
        ("frozen_misaligned_frequency", [omega_true + torch.tensor([0.09, -0.07], device=device), omega_aux]),
        ("frozen_near_bank", [omega_true + torch.tensor([0.04, 0.03], device=device), omega_true + torch.tensor([-0.05, 0.02], device=device), omega_aux]),
        ("learned_spectral_oracle", [omega_true, omega_aux]),
    ]
    rows = []
    for name, omegas in configs:
        train_basis = directional_jet_basis(
            train,
            omegas,
            [ex, ey, u],
            [0, 1],
            scale=float(args.spectral_train_radius),
            name=name,
        )
        eval_basis = directional_jet_basis(
            eval_grid,
            omegas,
            [ex, ey, u],
            [0, 1],
            scale=float(args.spectral_train_radius),
            name=name,
        )
        rows.append(
            fit_train_eval(
                task="spectral_geometry",
                train_basis=train_basis,
                eval_basis=eval_basis,
                train_target=target(train),
                eval_target=target(eval_grid),
                ridge=args.ridge,
            )
        )
    return rows


def lc_rows(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    train = make_grid_2d(args.lc_train_radius, signed=True, device=device, dtype=torch.float64)
    eval_grid = make_grid_2d(args.lc_eval_radius, signed=True, device=device, dtype=torch.float64)
    omega = 0.09
    ex = torch.tensor([1.0, 0.0], device=device, dtype=torch.float64)
    ey = torch.tensor([0.0, 1.0], device=device, dtype=torch.float64)
    u = normalize_direction(torch.tensor([[1.0, -0.65]], device=device, dtype=torch.float64)).reshape(-1)

    def target(d: torch.Tensor) -> torch.Tensor:
        raw = d @ u
        scale = float(args.lc_train_radius)
        phi = scale * torch.asinh(raw / scale)
        beta = raw / torch.sqrt(raw.square() + scale**2)
        return beta.square() * torch.cos(float(omega) * phi)

    configs = [
        ("coordinatewise_LC", [ex, ey], float(args.lc_train_radius)),
        ("directional_LC", [ex, ey, u], float(args.lc_train_radius)),
        ("wrong_scale_directional_LC", [ex, ey, u], float(args.lc_train_radius) / 8.0),
    ]
    rows = []
    for name, directions, scale in configs:
        train_basis = lc_basis(train, directions=directions, omega=omega, scale=scale, name=name)
        eval_basis = lc_basis(eval_grid, directions=directions, omega=omega, scale=scale, name=name)
        rows.append(
            fit_train_eval(
                task="lc_coordinate_ablation",
                train_basis=train_basis,
                eval_basis=eval_basis,
                train_target=target(train),
                eval_target=target(eval_grid),
                ridge=args.ridge,
            )
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = damping_rows(args, device) + spectral_rows(args, device) + lc_rows(args, device)

    csv_path = output_dir / "ablation_suite_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "ablation_suite_eval_r2.png"
    plot_results(rows, plot_path)
    summary = {
        "device": str(device),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "ablation_suite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    tasks = ["damping_constraint", "spectral_geometry", "lc_coordinate_ablation"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, task in zip(axes, tasks):
        values = [row for row in rows if row["task"] == task]
        labels = [str(row["basis"]) for row in values]
        eval_r2 = [float(row["eval_r2"]) for row in values]
        display_r2 = [min(1.0, max(-1.0, value)) for value in eval_r2]
        x = np.arange(len(labels))
        ax.bar(x, display_r2, color="#496f8f")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(task)
        ax.set_ylabel("eval R^2 (clipped)")
        for idx, value in enumerate(eval_r2):
            shown = display_r2[idx]
            label = "<-1" if value < -1.0 else f"{value:.2f}"
            ax.text(idx, min(shown + 0.03, 1.02), label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Toric PJ ablation suite.")
    parser.add_argument("--train-radius", type=int, default=32)
    parser.add_argument("--eval-radius", type=int, default=256)
    parser.add_argument("--spectral-train-radius", type=int, default=20)
    parser.add_argument("--spectral-eval-radius", type=int, default=40)
    parser.add_argument("--lc-train-radius", type=int, default=64)
    parser.add_argument("--lc-eval-radius", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--ridge", type=float, default=1e-9)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage_ablation")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
