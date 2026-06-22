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
from torch import nn
from torch.nn import functional as F

from toric_pj.diagnostics.basis_projection import Basis, default_device, make_grid_2d, normalize_columns, phase
from toric_pj.diagnostics.direction_alignment import normalize_direction


def damped_target(d: torch.Tensor, *, omega: torch.Tensor, direction: torch.Tensor, damping_dir: torch.Tensor, gamma: float, scale: float) -> torch.Tensor:
    coord = (d @ direction) / float(scale)
    damp = torch.exp(-float(gamma) * (d @ damping_dir).clamp_min(0.0) / float(scale))
    return coord.square() * damp * torch.cos(phase(d, omega))


def damped_basis(
    d: torch.Tensor,
    *,
    omega: torch.Tensor,
    direction: torch.Tensor,
    damping_dir: torch.Tensor,
    gamma: float,
    scale: float,
    name: str,
) -> Basis:
    coord = (d @ direction) / float(scale)
    damp = torch.exp(-float(gamma) * (d @ damping_dir).clamp_min(0.0) / float(scale))
    ph = phase(d, omega)
    cols = [torch.ones_like(coord)]
    labels = ["const"]
    orders = [0]
    for order in [0, 1, 2]:
        poly = torch.ones_like(coord) if order == 0 else coord.pow(order)
        cols.extend([damp * poly * torch.cos(ph), damp * poly * torch.sin(ph)])
        labels.extend([f"r{order}_cos", f"r{order}_sin"])
        orders.extend([order, order])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def lc_target(d: torch.Tensor, *, direction: torch.Tensor, omega_rate: float, scale: float) -> torch.Tensor:
    raw = d @ direction
    beta = raw / torch.sqrt(raw.square() + float(scale) ** 2)
    phi = float(scale) * torch.asinh(raw / float(scale))
    return beta.square() * torch.cos(float(omega_rate) * phi)


def lc_basis(d: torch.Tensor, *, directions: list[torch.Tensor], omega_rate: float, scale: float, name: str) -> Basis:
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    for dir_idx, direction in enumerate(directions):
        raw = d @ direction
        beta = raw / torch.sqrt(raw.square() + float(scale) ** 2)
        phi = float(scale) * torch.asinh(raw / float(scale))
        ph = float(omega_rate) * phi
        for order in [0, 1, 2]:
            poly = torch.ones_like(raw) if order == 0 else beta.pow(order)
            cols.extend([poly * torch.cos(ph), poly * torch.sin(ph)])
            labels.extend([f"u{dir_idx}_r{order}_cos", f"u{dir_idx}_r{order}_sin"])
            orders.extend([order, order])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def fit_train_eval(
    train_basis: Basis,
    eval_basis: Basis,
    train_target: torch.Tensor,
    eval_target: torch.Tensor,
    *,
    ridge: float,
) -> dict[str, float]:
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
        "train_r2": float((1.0 - train_mse / train_var).detach().cpu()),
        "eval_r2": float((1.0 - eval_mse / eval_var).detach().cpu()),
        "train_mse": float(train_mse.detach().cpu()),
        "eval_mse": float(eval_mse.detach().cpu()),
        "eval_pred_max_abs": float(torch.max(torch.abs(pred_eval)).detach().cpu()),
        "num_features": train_basis.matrix.shape[1],
    }


class LearnableDampedKernel(nn.Module):
    def __init__(self, *, omega: torch.Tensor, direction: torch.Tensor, damping_dir: torch.Tensor, scale: float, init_gamma: float) -> None:
        super().__init__()
        self.register_buffer("omega", omega)
        self.register_buffer("direction", direction)
        self.register_buffer("damping_dir", damping_dir)
        self.scale = float(scale)
        raw = torch.log(torch.expm1(torch.tensor(init_gamma, device=omega.device, dtype=omega.dtype).clamp_min(1e-6)))
        self.raw_gamma = nn.Parameter(raw)
        self.coeff = nn.Parameter(0.01 * torch.randn(7, device=omega.device, dtype=omega.dtype))

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        basis = damped_basis(
            d,
            omega=self.omega,
            direction=self.direction,
            damping_dir=self.damping_dir,
            gamma=float(self.gamma.detach()),
            scale=self.scale,
            name="learned",
        ).matrix
        # Recompute differentiable gamma-dependent columns.
        coord = (d @ self.direction) / self.scale
        damp = torch.exp(-self.gamma * (d @ self.damping_dir).clamp_min(0.0) / self.scale)
        ph = phase(d, self.omega)
        cols = [torch.ones_like(coord)]
        for order in [0, 1, 2]:
            poly = torch.ones_like(coord) if order == 0 else coord.pow(order)
            cols.extend([damp * poly * torch.cos(ph), damp * poly * torch.sin(ph)])
        basis = torch.stack(cols, dim=1)
        return basis @ self.coeff


class LearnableLCScaleKernel(nn.Module):
    def __init__(self, *, direction: torch.Tensor, omega_rate: float, init_scale: float) -> None:
        super().__init__()
        self.register_buffer("direction", direction)
        self.omega_rate = float(omega_rate)
        raw = torch.log(torch.expm1(torch.tensor(init_scale, device=direction.device, dtype=direction.dtype).clamp_min(1e-6)))
        self.raw_scale = nn.Parameter(raw)
        self.coeff = nn.Parameter(0.01 * torch.randn(7, device=direction.device, dtype=direction.dtype))

    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale) + 1e-6

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        raw = d @ self.direction
        scale = self.scale
        beta = raw / torch.sqrt(raw.square() + scale.square())
        phi = scale * torch.asinh(raw / scale)
        ph = self.omega_rate * phi
        cols = [torch.ones_like(raw)]
        for order in [0, 1, 2]:
            poly = torch.ones_like(raw) if order == 0 else beta.pow(order)
            cols.extend([poly * torch.cos(ph), poly * torch.sin(ph)])
        return torch.stack(cols, dim=1) @ self.coeff


def train_scalar_model(model: nn.Module, d: torch.Tensor, target: torch.Tensor, *, steps: int, lr: float, coeff_l2: float) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(d)
        loss = torch.mean((pred - target).square())
        if hasattr(model, "coeff"):
            loss = loss + coeff_l2 * model.coeff.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()


def run_damping(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    train = make_grid_2d(args.damping_train_radius, signed=False, device=device, dtype=torch.float64)
    calib = make_grid_2d(args.damping_calib_radius, signed=False, device=device, dtype=torch.float64)
    eval_grid = make_grid_2d(args.damping_eval_radius, signed=False, device=device, dtype=torch.float64)
    omega = torch.tensor([0.31, 0.47], device=device, dtype=torch.float64)
    direction = normalize_direction(torch.tensor([[0.9, 0.35]], device=device, dtype=torch.float64)).reshape(-1)
    damping_dir = normalize_direction(torch.tensor([[0.75, 0.55]], device=device, dtype=torch.float64)).reshape(-1)
    train_target = damped_target(train, omega=omega, direction=direction, damping_dir=damping_dir, gamma=args.gamma, scale=args.damping_train_radius)
    calib_target = damped_target(calib, omega=omega, direction=direction, damping_dir=damping_dir, gamma=args.gamma, scale=args.damping_train_radius)
    eval_target = damped_target(eval_grid, omega=omega, direction=direction, damping_dir=damping_dir, gamma=args.gamma, scale=args.damping_train_radius)
    rows = []
    for variant, gamma in [
        ("raw_no_damping", 0.0),
        ("wrong_sign_control", -args.gamma),
        ("dual_cone_oracle", args.gamma),
    ]:
        train_basis = damped_basis(train, omega=omega, direction=direction, damping_dir=damping_dir, gamma=gamma, scale=args.damping_train_radius, name=variant)
        eval_basis = damped_basis(eval_grid, omega=omega, direction=direction, damping_dir=damping_dir, gamma=gamma, scale=args.damping_train_radius, name=variant)
        rows.append(
            {
                "task": "dual_cone_damping",
                "variant": variant,
                "learned_value": gamma,
                **fit_train_eval(train_basis, eval_basis, train_target, eval_target, ridge=args.ridge),
            }
        )
    model = LearnableDampedKernel(
        omega=omega,
        direction=direction,
        damping_dir=damping_dir,
        scale=args.damping_train_radius,
        init_gamma=args.gamma / 4.0,
    )
    train_scalar_model(model, train, train_target, steps=args.steps, lr=args.lr, coeff_l2=args.coeff_l2)
    learned_gamma = float(model.gamma.detach().cpu())
    train_basis = damped_basis(train, omega=omega, direction=direction, damping_dir=damping_dir, gamma=learned_gamma, scale=args.damping_train_radius, name="learned_dual_cone_gamma")
    eval_basis = damped_basis(eval_grid, omega=omega, direction=direction, damping_dir=damping_dir, gamma=learned_gamma, scale=args.damping_train_radius, name="learned_dual_cone_gamma")
    rows.append(
        {
            "task": "dual_cone_damping",
            "variant": "learned_dual_cone_gamma",
            "learned_value": learned_gamma,
            **fit_train_eval(train_basis, eval_basis, train_target, eval_target, ridge=args.ridge),
        }
    )
    best_gamma = None
    best_calib = -float("inf")
    for gamma in np.linspace(0.15, 1.35, 31):
        train_basis = damped_basis(train, omega=omega, direction=direction, damping_dir=damping_dir, gamma=float(gamma), scale=args.damping_train_radius, name="calib")
        calib_basis = damped_basis(calib, omega=omega, direction=direction, damping_dir=damping_dir, gamma=float(gamma), scale=args.damping_train_radius, name="calib")
        calib_metrics = fit_train_eval(train_basis, calib_basis, train_target, calib_target, ridge=args.ridge)
        if calib_metrics["eval_r2"] > best_calib:
            best_calib = calib_metrics["eval_r2"]
            best_gamma = float(gamma)
    assert best_gamma is not None
    train_basis = damped_basis(train, omega=omega, direction=direction, damping_dir=damping_dir, gamma=best_gamma, scale=args.damping_train_radius, name="calibrated_dual_cone_gamma")
    eval_basis = damped_basis(eval_grid, omega=omega, direction=direction, damping_dir=damping_dir, gamma=best_gamma, scale=args.damping_train_radius, name="calibrated_dual_cone_gamma")
    rows.append(
        {
            "task": "dual_cone_damping",
            "variant": "calibrated_dual_cone_gamma",
            "learned_value": best_gamma,
            **fit_train_eval(train_basis, eval_basis, train_target, eval_target, ridge=args.ridge),
        }
    )
    return rows


def run_lc(args: argparse.Namespace, device: torch.device) -> list[dict[str, object]]:
    train = make_grid_2d(args.lc_train_radius, signed=True, device=device, dtype=torch.float64)
    calib = make_grid_2d(args.lc_calib_radius, signed=True, device=device, dtype=torch.float64)
    eval_grid = make_grid_2d(args.lc_eval_radius, signed=True, device=device, dtype=torch.float64)
    omega_rate = 0.09
    ex = torch.tensor([1.0, 0.0], device=device, dtype=torch.float64)
    ey = torch.tensor([0.0, 1.0], device=device, dtype=torch.float64)
    direction = normalize_direction(torch.tensor([[1.0, -0.65]], device=device, dtype=torch.float64)).reshape(-1)
    train_target = lc_target(train, direction=direction, omega_rate=omega_rate, scale=args.lc_train_radius)
    calib_target = lc_target(calib, direction=direction, omega_rate=omega_rate, scale=args.lc_train_radius)
    eval_target = lc_target(eval_grid, direction=direction, omega_rate=omega_rate, scale=args.lc_train_radius)
    rows = []
    for variant, directions, scale in [
        ("coordinatewise_LC", [ex, ey], args.lc_train_radius),
        ("wrong_scale_directional_LC", [direction], args.lc_train_radius / 8.0),
        ("directional_LC_oracle_scale", [direction], args.lc_train_radius),
    ]:
        train_basis = lc_basis(train, directions=directions, omega_rate=omega_rate, scale=scale, name=variant)
        eval_basis = lc_basis(eval_grid, directions=directions, omega_rate=omega_rate, scale=scale, name=variant)
        rows.append(
            {
                "task": "learned_lc_scale",
                "variant": variant,
                "learned_value": scale,
                **fit_train_eval(train_basis, eval_basis, train_target, eval_target, ridge=args.ridge),
            }
        )
    model = LearnableLCScaleKernel(direction=direction, omega_rate=omega_rate, init_scale=args.lc_train_radius / 5.0)
    train_scalar_model(model, train, train_target, steps=args.steps, lr=args.lr, coeff_l2=args.coeff_l2)
    learned_scale = float(model.scale.detach().cpu())
    train_basis = lc_basis(train, directions=[direction], omega_rate=omega_rate, scale=learned_scale, name="learned_directional_LC_scale")
    eval_basis = lc_basis(eval_grid, directions=[direction], omega_rate=omega_rate, scale=learned_scale, name="learned_directional_LC_scale")
    rows.append(
        {
            "task": "learned_lc_scale",
            "variant": "learned_directional_LC_scale",
            "learned_value": learned_scale,
            **fit_train_eval(train_basis, eval_basis, train_target, eval_target, ridge=args.ridge),
        }
    )
    best_scale = None
    best_calib = -float("inf")
    for scale in np.linspace(args.lc_train_radius / 4.0, args.lc_train_radius * 2.0, 36):
        train_basis = lc_basis(train, directions=[direction], omega_rate=omega_rate, scale=float(scale), name="calib")
        calib_basis = lc_basis(calib, directions=[direction], omega_rate=omega_rate, scale=float(scale), name="calib")
        calib_metrics = fit_train_eval(train_basis, calib_basis, train_target, calib_target, ridge=args.ridge)
        if calib_metrics["eval_r2"] > best_calib:
            best_calib = calib_metrics["eval_r2"]
            best_scale = float(scale)
    assert best_scale is not None
    train_basis = lc_basis(train, directions=[direction], omega_rate=omega_rate, scale=best_scale, name="calibrated_directional_LC_scale")
    eval_basis = lc_basis(eval_grid, directions=[direction], omega_rate=omega_rate, scale=best_scale, name="calibrated_directional_LC_scale")
    rows.append(
        {
            "task": "learned_lc_scale",
            "variant": "calibrated_directional_LC_scale",
            "learned_value": best_scale,
            **fit_train_eval(train_basis, eval_basis, train_target, eval_target, ridge=args.ridge),
        }
    )
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_damping(args, device) + run_lc(args, device)
    csv_path = output_dir / "stability_constraints_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = output_dir / "stability_constraints_eval_r2.png"
    plot_results(rows, plot_path)
    summary = {"device": str(device), "csv": str(csv_path), "plot": str(plot_path), "rows": rows}
    summary_path = output_dir / "stability_constraints_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, rows, summary)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    tasks = sorted({str(row["task"]) for row in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(12, 4.6), squeeze=False)
    for ax, task in zip(axes.reshape(-1), tasks):
        values = [row for row in rows if row["task"] == task]
        labels = [str(row["variant"]) for row in values]
        y = [float(row["eval_r2"]) for row in values]
        display = [max(-1.0, min(1.0, value)) for value in y]
        x = np.arange(len(labels))
        ax.bar(x, display, color="#6e7442")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("eval R2 (clipped)")
        ax.set_title(task)
        for idx, value in enumerate(y):
            label = "<-1" if value < -1.0 else f"{value:.2f}"
            ax.text(idx, min(display[idx] + 0.03, 1.02), label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# V2-C Stability Constraints Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        "",
        "Run command:",
        "",
        "```bash",
        "python scripts/run_v2_stability.py --device cuda --output-dir results/v2_stability",
        "```",
        "",
        "## Results",
        "",
        "| task | variant | learned value | eval R2 | max abs pred | features |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['task']} | {row['variant']} | {float(row['learned_value']):.4f} | "
            + f"{float(row['eval_r2']):.4f} | {float(row['eval_pred_max_abs']):.4g} | {int(row['num_features'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Dual-cone damping is compared against raw/no damping and wrong-sign damping controls.",
            "- The learned gamma variant starts from a smaller damping value and recovers a positive stable damping factor from the train window.",
            "- Directional LC is compared against coordinatewise LC and wrong-scale LC.",
            "- The learned LC-scale variant starts from a wrong scale and tests whether scale can be recovered from data.",
            "",
            "Artifacts:",
            "",
            "- `stability_constraints_results.csv`",
            "- `stability_constraints_summary.json`",
            "- `stability_constraints_eval_r2.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-C stability constraint experiments.")
    parser.add_argument("--damping-train-radius", type=int, default=28)
    parser.add_argument("--damping-calib-radius", type=int, default=56)
    parser.add_argument("--damping-eval-radius", type=int, default=160)
    parser.add_argument("--lc-train-radius", type=int, default=48)
    parser.add_argument("--lc-calib-radius", type=int, default=96)
    parser.add_argument("--lc-eval-radius", type=int, default=160)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--steps", type=int, default=650)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--coeff-l2", type=float, default=1e-7)
    parser.add_argument("--ridge", type=float, default=1e-9)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_stability")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
