from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from toric_pj.diagnostics.basis_projection import default_device, make_grid_2d, normalize_columns, phase
from toric_pj.diagnostics.direction_alignment import abs_cosine_alignment, normalize_direction


BRANCHES = ("FJ", "affine", "LC")


@dataclass(frozen=True)
class ComponentMeta:
    branch: str
    order: int
    label: str
    direction: tuple[float, float] | None = None
    omega: tuple[float, float] | None = None


class AdaptiveDictionaryModel(nn.Module):
    def __init__(self, matrix: torch.Tensor, metas: list[ComponentMeta]) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.metas = metas
        self.coeff = nn.Parameter(torch.zeros(matrix.shape[1], dtype=matrix.dtype, device=matrix.device))
        self.gate_logits = nn.Parameter(torch.zeros(len(BRANCHES), dtype=matrix.dtype, device=matrix.device))
        branch_ids = [BRANCHES.index(meta.branch) for meta in metas]
        self.register_buffer("branch_ids", torch.tensor(branch_ids, device=matrix.device, dtype=torch.long))

    @property
    def gates(self) -> torch.Tensor:
        return F.softplus(self.gate_logits)

    @property
    def gate_shares(self) -> torch.Tensor:
        gates = self.gates
        return gates / gates.sum().clamp_min(1e-12)

    def branch_outputs(self) -> torch.Tensor:
        raw = self.matrix * self.coeff.unsqueeze(0)
        outs = []
        for branch_id in range(len(BRANCHES)):
            mask = self.branch_ids == branch_id
            outs.append(raw[:, mask].sum(dim=1) if torch.any(mask) else torch.zeros(raw.shape[0], device=raw.device))
        return torch.stack(outs, dim=1)

    def forward(self) -> torch.Tensor:
        return (self.branch_outputs() * self.gates.reshape(1, -1)).sum(dim=1)


def build_dictionary(
    d: torch.Tensor,
    *,
    radius: int,
    omega_main: torch.Tensor,
    omega_beat: torch.Tensor,
    omega_lc: float,
    directions: list[torch.Tensor],
) -> tuple[torch.Tensor, list[ComponentMeta]]:
    columns: list[torch.Tensor] = []
    metas: list[ComponentMeta] = []
    scale = float(radius)
    dtype = d.dtype
    device = d.device

    def add(column: torch.Tensor, meta: ComponentMeta) -> None:
        columns.append(column)
        metas.append(meta)

    for omega_label, omega in [("main", omega_main), ("beat", omega_beat)]:
        ph = phase(d, omega)
        omega_tuple = tuple(float(x) for x in omega.detach().cpu())
        add(torch.cos(ph), ComponentMeta("FJ", 0, f"FJ_{omega_label}_cos", omega=omega_tuple))
        add(torch.sin(ph), ComponentMeta("FJ", 0, f"FJ_{omega_label}_sin", omega=omega_tuple))
        for dir_idx, direction in enumerate(directions):
            direction = normalize_direction(direction.reshape(1, -1)).reshape(-1)
            dir_tuple = tuple(float(x) for x in direction.detach().cpu())
            coord = (d @ direction) / scale
            for order in [1, 2]:
                poly = coord.pow(order)
                add(
                    poly * torch.cos(ph),
                    ComponentMeta("FJ", order, f"FJ_{omega_label}_u{dir_idx}_r{order}_cos", dir_tuple, omega_tuple),
                )
                add(
                    poly * torch.sin(ph),
                    ComponentMeta("FJ", order, f"FJ_{omega_label}_u{dir_idx}_r{order}_sin", dir_tuple, omega_tuple),
                )

    add(torch.ones(d.shape[0], device=device, dtype=dtype), ComponentMeta("affine", 0, "affine_const"))
    for dir_idx, direction in enumerate(directions):
        direction = normalize_direction(direction.reshape(1, -1)).reshape(-1)
        dir_tuple = tuple(float(x) for x in direction.detach().cpu())
        add(-(d @ direction) / scale, ComponentMeta("affine", 1, f"affine_u{dir_idx}", dir_tuple))

    lc_omega_tuple = (float(omega_lc),)
    for dir_idx, direction in enumerate(directions):
        direction = normalize_direction(direction.reshape(1, -1)).reshape(-1)
        dir_tuple = tuple(float(x) for x in direction.detach().cpu())
        raw = d @ direction
        phi = scale * torch.asinh(raw / scale)
        beta = raw / torch.sqrt(raw.square() + scale**2)
        ph = float(omega_lc) * phi
        add(torch.cos(ph), ComponentMeta("LC", 0, f"LC_u{dir_idx}_r0_cos", dir_tuple, lc_omega_tuple))
        add(torch.sin(ph), ComponentMeta("LC", 0, f"LC_u{dir_idx}_r0_sin", dir_tuple, lc_omega_tuple))
        for order in [1, 2]:
            poly = beta.pow(order)
            add(poly * torch.cos(ph), ComponentMeta("LC", order, f"LC_u{dir_idx}_r{order}_cos", dir_tuple, lc_omega_tuple))
            add(poly * torch.sin(ph), ComponentMeta("LC", order, f"LC_u{dir_idx}_r{order}_sin", dir_tuple, lc_omega_tuple))

    matrix = torch.stack(columns, dim=1)
    matrix, _ = normalize_columns(matrix)
    return matrix, metas


def build_targets(
    d: torch.Tensor,
    *,
    radius: int,
    omega_main: torch.Tensor,
    omega_beat: torch.Tensor,
    u_star: torch.Tensor,
    s_star: torch.Tensor,
    omega_lc: float,
) -> dict[str, torch.Tensor]:
    scale = float(radius)
    ph = phase(d, omega_main)
    fj = torch.cos(ph)
    affine = -(d @ s_star) / scale
    jet1 = ((d @ u_star) / scale) * fj
    jet2 = ((d @ u_star) / scale).pow(2) * fj
    beat = torch.cos(phase(d, omega_beat))
    raw = d @ u_star
    phi = scale * torch.asinh(raw / scale)
    beta = raw / torch.sqrt(raw.square() + scale**2)
    lc = beta.pow(2) * torch.cos(float(omega_lc) * phi)
    return {
        "B1_fourier_plus_affine": fj + 0.65 * affine,
        "B2_affine_weak_order2": 0.95 * affine + 0.18 * jet2,
        "B3_music_like": -0.45 * d[:, 0] / scale + 0.35 * beat + 0.16 * jet1,
        "B4_lc_teacher": lc,
    }


def train_one(
    matrix: torch.Tensor,
    metas: list[ComponentMeta],
    target: torch.Tensor,
    *,
    steps: int,
    lr: float,
    gate_l1: float,
    coeff_l2: float,
    seed: int,
) -> tuple[AdaptiveDictionaryModel, list[float]]:
    torch.manual_seed(seed)
    model = AdaptiveDictionaryModel(matrix, metas)
    with torch.no_grad():
        model.coeff.normal_(mean=0.0, std=1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model()
        mse = torch.mean((pred - target).square())
        reg = gate_l1 * model.gates.sum() + coeff_l2 * model.coeff.square().mean()
        loss = mse + reg
        loss.backward()
        opt.step()
        if step % max(1, steps // 100) == 0 or step == steps - 1:
            history.append(float(mse.detach().cpu()))
    return model, history


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

    matrix, metas = build_dictionary(
        d,
        radius=args.radius,
        omega_main=omega_main,
        omega_beat=omega_beat,
        omega_lc=omega_lc,
        directions=directions,
    )
    targets = build_targets(
        d,
        radius=args.radius,
        omega_main=omega_main,
        omega_beat=omega_beat,
        u_star=u_star,
        s_star=s_star,
        omega_lc=omega_lc,
    )

    rows: list[dict[str, object]] = []
    histories: dict[str, list[float]] = {}
    best_models: dict[str, AdaptiveDictionaryModel] = {}
    for target_name, target in targets.items():
        best_model: AdaptiveDictionaryModel | None = None
        best_history: list[float] = []
        best_mse = float("inf")
        for restart in range(args.restarts):
            model, history = train_one(
                matrix,
                metas,
                target,
                steps=args.steps,
                lr=args.lr,
                gate_l1=args.gate_l1,
                coeff_l2=args.coeff_l2,
                seed=args.seed + restart,
            )
            mse = float(torch.mean((model() - target).square()).detach().cpu())
            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_history = history
        assert best_model is not None
        rows.append(diagnostics_row(target_name, best_model, metas, target, u_star, omega_main))
        histories[target_name] = best_history
        best_models[target_name] = best_model

    csv_path = output_dir / "adaptive_teacher_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "adaptive_teacher_diagnostics.png"
    plot_diagnostics(rows, histories, plot_path)

    summary = {
        "device": str(device),
        "num_points": int(d.shape[0]),
        "num_components": int(matrix.shape[1]),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "adaptive_teacher_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def diagnostics_row(
    target_name: str,
    model: AdaptiveDictionaryModel,
    metas: list[ComponentMeta],
    target: torch.Tensor,
    u_star: torch.Tensor,
    omega_main: torch.Tensor,
) -> dict[str, object]:
    with torch.no_grad():
        pred = model()
        mse = torch.mean((pred - target).square())
        var = torch.mean((target - target.mean()).square()).clamp_min(1e-30)
        r2 = 1.0 - mse / var
        raw_branch = model.branch_outputs()
        gated_branch = raw_branch * model.gates.reshape(1, -1)
        total_norm = torch.linalg.norm(pred).clamp_min(1e-30)
        branch_energy = {
            branch: float((torch.linalg.norm(gated_branch[:, idx]) / total_norm).detach().cpu())
            for idx, branch in enumerate(BRANCHES)
        }
        base_mse = mse
        branch_loo = {
            branch: float((torch.mean((pred - gated_branch[:, idx] - target).square()) - base_mse).detach().cpu())
            for idx, branch in enumerate(BRANCHES)
        }
        order_energy, order_loo = order_diagnostics(model, metas, target)
        branch_order_energy, branch_order_loo = branch_order_diagnostics(model, metas, target)
        top_fj = top_component_alignment(model, metas, branch="FJ", target_direction=u_star, target_omega=omega_main)
        top_lc = top_component_alignment(model, metas, branch="LC", target_direction=u_star, target_omega=None)
        gates = {branch: float(model.gate_shares[idx].detach().cpu()) for idx, branch in enumerate(BRANCHES)}
    return {
        "target": target_name,
        "mse": float(mse.detach().cpu()),
        "r2": float(r2.detach().cpu()),
        "gate_share": json.dumps(gates, sort_keys=True),
        "branch_energy": json.dumps(branch_energy, sort_keys=True),
        "branch_loo": json.dumps(branch_loo, sort_keys=True),
        "top_branch_energy": max(branch_energy, key=branch_energy.get),
        "top_branch_loo": max(branch_loo, key=branch_loo.get),
        "order_energy": json.dumps(order_energy, sort_keys=True),
        "order_loo": json.dumps(order_loo, sort_keys=True),
        "branch_order_energy": json.dumps(branch_order_energy, sort_keys=True),
        "branch_order_loo": json.dumps(branch_order_loo, sort_keys=True),
        "top_order_energy": max(order_energy, key=order_energy.get),
        "top_order_loo": max(order_loo, key=order_loo.get),
        "top_fj_order_energy": _top_branch_order(branch_order_energy, "FJ"),
        "top_fj_order_loo": _top_branch_order(branch_order_loo, "FJ"),
        "top_lc_order_energy": _top_branch_order(branch_order_energy, "LC"),
        "top_lc_order_loo": _top_branch_order(branch_order_loo, "LC"),
        "top_fj_direction_alignment": top_fj["direction_alignment"],
        "top_fj_spectral_alignment": top_fj["spectral_alignment"],
        "top_fj_label": top_fj["label"],
        "top_lc_direction_alignment": top_lc["direction_alignment"],
        "top_lc_label": top_lc["label"],
    }


def order_diagnostics(
    model: AdaptiveDictionaryModel, metas: list[ComponentMeta], target: torch.Tensor
) -> tuple[dict[int, float], dict[int, float]]:
    pred = model()
    base_mse = torch.mean((pred - target).square())
    gated_columns = model.matrix * model.coeff.unsqueeze(0)
    gated_columns = gated_columns * model.gates[model.branch_ids].unsqueeze(0)
    total_norm = torch.linalg.norm(pred).clamp_min(1e-30)
    orders = sorted({meta.order for meta in metas})
    energy: dict[int, float] = {}
    loo: dict[int, float] = {}
    for order in orders:
        idx = torch.tensor([i for i, meta in enumerate(metas) if meta.order == order], device=target.device)
        contrib = gated_columns[:, idx].sum(dim=1)
        energy[order] = float((torch.linalg.norm(contrib) / total_norm).detach().cpu())
        loo[order] = float((torch.mean((pred - contrib - target).square()) - base_mse).detach().cpu())
    return energy, loo


def branch_order_diagnostics(
    model: AdaptiveDictionaryModel, metas: list[ComponentMeta], target: torch.Tensor
) -> tuple[dict[str, float], dict[str, float]]:
    pred = model()
    base_mse = torch.mean((pred - target).square())
    gated_columns = model.matrix * model.coeff.unsqueeze(0)
    gated_columns = gated_columns * model.gates[model.branch_ids].unsqueeze(0)
    total_norm = torch.linalg.norm(pred).clamp_min(1e-30)
    keys = sorted({(meta.branch, meta.order) for meta in metas})
    energy: dict[str, float] = {}
    loo: dict[str, float] = {}
    for branch, order in keys:
        idx = torch.tensor(
            [i for i, meta in enumerate(metas) if meta.branch == branch and meta.order == order],
            device=target.device,
        )
        contrib = gated_columns[:, idx].sum(dim=1)
        key = f"{branch}:r{order}"
        energy[key] = float((torch.linalg.norm(contrib) / total_norm).detach().cpu())
        loo[key] = float((torch.mean((pred - contrib - target).square()) - base_mse).detach().cpu())
    return energy, loo


def _top_branch_order(values: dict[str, float], branch: str) -> int | None:
    prefix = f"{branch}:r"
    branch_items = {key: value for key, value in values.items() if key.startswith(prefix)}
    if not branch_items:
        return None
    key = max(branch_items, key=branch_items.get)
    return int(key.split(":r", 1)[1])


def top_component_alignment(
    model: AdaptiveDictionaryModel,
    metas: list[ComponentMeta],
    *,
    branch: str,
    target_direction: torch.Tensor,
    target_omega: torch.Tensor | None,
) -> dict[str, object]:
    weights = torch.abs(model.coeff * model.gates[model.branch_ids])
    candidates = [
        (idx, float(weights[idx].detach().cpu()))
        for idx, meta in enumerate(metas)
        if meta.branch == branch and meta.direction is not None
    ]
    if not candidates:
        return {"label": "", "direction_alignment": None, "spectral_alignment": None}
    idx = max(candidates, key=lambda item: item[1])[0]
    meta = metas[idx]
    direction = torch.tensor(meta.direction, device=target_direction.device, dtype=target_direction.dtype)
    direction_alignment = float(abs_cosine_alignment(direction, target_direction).detach().cpu())
    spectral_alignment = None
    if target_omega is not None and meta.omega is not None and len(meta.omega) == 2:
        omega = torch.tensor(meta.omega, device=target_omega.device, dtype=target_omega.dtype)
        spectral_alignment = float(abs_cosine_alignment(omega, target_omega).detach().cpu())
    return {
        "label": meta.label,
        "direction_alignment": direction_alignment,
        "spectral_alignment": spectral_alignment,
    }


def plot_diagnostics(rows: list[dict[str, object]], histories: dict[str, list[float]], path: Path) -> None:
    targets = [str(row["target"]) for row in rows]
    branch_matrix = np.array(
        [[json.loads(str(row["branch_energy"]))[branch] for branch in BRANCHES] for row in rows]
    )
    gate_matrix = np.array([[json.loads(str(row["gate_share"]))[branch] for branch in BRANCHES] for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, values, title in [
        (axes[0], branch_matrix, "Functional Energy by Branch"),
        (axes[1], gate_matrix, "Gate Share"),
    ]:
        im = ax.imshow(values, vmin=0.0, vmax=max(1.0, float(np.nanmax(values))), aspect="auto", cmap="magma")
        ax.set_xticks(np.arange(len(BRANCHES)), BRANCHES)
        ax.set_yticks(np.arange(len(targets)), targets)
        ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for target, history in histories.items():
        axes[2].semilogy(history, label=target)
    axes[2].set_title("Training MSE")
    axes[2].set_xlabel("logged step")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive teacher sector-selection experiments.")
    parser.add_argument("--radius", type=int, default=24)
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--gate-l1", type=float, default=2e-5)
    parser.add_argument("--coeff-l2", type=float, default=1e-7)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage2")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
