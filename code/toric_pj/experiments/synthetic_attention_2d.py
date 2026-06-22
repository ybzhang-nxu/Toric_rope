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
from torch.nn import functional as F

from toric_pj.diagnostics.basis_projection import default_device, phase
from toric_pj.diagnostics.direction_alignment import abs_cosine_alignment, normalize_direction
from toric_pj.models.toric_pj_bias import ToricPJConfig, ToricPJBias


BRANCHES = ("FJ", "affine", "LC")


def make_positions(side: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(side, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(values, values, indexing="ij")
    coords = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    return coords - (side - 1) / 2.0


def pairwise_displacements(positions: torch.Tensor) -> torch.Tensor:
    return positions[:, None, :] - positions[None, :, :]


def teacher_logits(task: str, d: torch.Tensor, side: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    dtype = d.dtype
    device = d.device
    scale = float(side)
    if task == "E2_diagonal_copy":
        perp = normalize_direction(torch.tensor([[1.0, -1.0]], device=device, dtype=dtype)).reshape(-1)
        diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=device, dtype=dtype)).reshape(-1)
        omega = 0.78 * perp
        diag_coord = d @ diag
        logits = 3.0 * torch.cos(phase(d, omega)) - 0.10 * (diag_coord / scale).square()
        return logits, {"direction": perp, "omega": omega}
    if task == "E4_signed_jet_teacher":
        u = normalize_direction(torch.tensor([[1.0, -0.6]], device=device, dtype=dtype)).reshape(-1)
        omega = torch.tensor([0.37, 0.61], device=device, dtype=dtype)
        logits = 4.0 * ((d @ u) / scale) * torch.cos(phase(d, omega))
        return logits, {"direction": u, "omega": omega}
    raise ValueError(f"unknown task: {task}")


def init_model(task: str, side: int, device: torch.device) -> ToricPJBias:
    model = ToricPJBias(
        ToricPJConfig(
            n_heads=1,
            n_dims=2,
            n_freqs=3,
            n_dirs=5,
            max_order=2,
            lengths=(float(side), float(side)),
            use_lc=True,
            use_affine=True,
            use_damping=False,
        )
    ).to(device)
    with torch.no_grad():
        dtype = model.omega.dtype
        if task == "E2_diagonal_copy":
            model.omega[0, 0] = torch.tensor([0.78, -0.78], device=device, dtype=dtype)
            model.omega[0, 1] = torch.tensor([0.78, 0.78], device=device, dtype=dtype)
            model.omega[0, 2] = torch.tensor([0.37, 0.61], device=device, dtype=dtype)
        else:
            model.omega[0, 0] = torch.tensor([0.37, 0.61], device=device, dtype=dtype)
            model.omega[0, 1] = torch.tensor([0.0, 2.0 * math.pi / 8.0], device=device, dtype=dtype)
            model.omega[0, 2] = torch.tensor([0.78, -0.78], device=device, dtype=dtype)
        dirs = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, -0.6],
                [1.0, -1.0],
                [1.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        model.raw_dirs[0] = normalize_direction(dirs)
        model.gate_logits[0] = torch.tensor([0.0, -0.2, -0.6], device=device, dtype=dtype)
    return model


def train_task(args: argparse.Namespace, task: str) -> dict[str, object]:
    device = default_device(args.device)
    dtype = torch.float32
    positions = make_positions(args.side, device, dtype)
    d = pairwise_displacements(positions)
    logits_teacher, target_info = teacher_logits(task, d, args.side)
    teacher_prob = torch.softmax(logits_teacher / args.teacher_temperature, dim=-1)
    teacher_top = torch.argmax(teacher_prob, dim=-1)

    model = init_model(task, args.side, device)
    if args.freeze_geometry:
        model.omega.requires_grad_(False)
        model.raw_dirs.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    history: list[float] = []
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        logits = model(d)[..., 0]
        log_prob = torch.log_softmax(logits, dim=-1)
        kl = torch.mean(torch.sum(teacher_prob * (torch.log(teacher_prob.clamp_min(1e-12)) - log_prob), dim=-1))
        entropy_reg = args.gate_l1 * model.gate_values.sum()
        loss = kl + entropy_reg
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 120) == 0 or step == args.steps - 1:
            history.append(float(kl.detach().cpu()))

    with torch.no_grad():
        logits = model(d)[..., 0]
        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log_softmax(logits, dim=-1)
        kl = torch.mean(torch.sum(teacher_prob * (torch.log(teacher_prob.clamp_min(1e-12)) - log_prob), dim=-1))
        top1 = torch.mean((torch.argmax(prob, dim=-1) == teacher_top).float())
        branch = model.branch_outputs(d)
        gates = model.gate_values[:, :]
        gated = {
            "FJ": gates[0, 0] * branch["fj"][0],
            "affine": gates[0, 1] * branch["affine"][0],
            "LC": gates[0, 2] * branch["lc"][0],
        }
        pred = sum(gated.values())
        denom = torch.linalg.norm(pred).clamp_min(1e-30)
        branch_energy = {name: float((torch.linalg.norm(value) / denom).detach().cpu()) for name, value in gated.items()}
        branch_loo = {}
        base_kl = kl
        for name, value in gated.items():
            logits_loo = pred - value
            log_prob_loo = torch.log_softmax(logits_loo, dim=-1)
            kl_loo = torch.mean(
                torch.sum(teacher_prob * (torch.log(teacher_prob.clamp_min(1e-12)) - log_prob_loo), dim=-1)
            )
            branch_loo[name] = float((kl_loo - base_kl).detach().cpu())
        gate_share = {name: float(model.gate_shares[0, idx].detach().cpu()) for idx, name in enumerate(BRANCHES)}
        direction_align = max_direction_alignment(model, target_info["direction"])
        spectral_align = max_spectral_alignment(model, target_info["omega"])

    return {
        "task": task,
        "kl": float(kl.detach().cpu()),
        "top1_agreement": float(top1.detach().cpu()),
        "gate_share": gate_share,
        "branch_energy": branch_energy,
        "branch_loo": branch_loo,
        "top_branch_energy": max(branch_energy, key=branch_energy.get),
        "top_branch_loo": max(branch_loo, key=branch_loo.get),
        "max_direction_alignment": direction_align,
        "max_spectral_alignment": spectral_align,
        "history": history,
    }


def max_direction_alignment(model: ToricPJBias, target: torch.Tensor) -> float:
    dirs = model.normalized_dirs[0].detach().to(device=target.device, dtype=target.dtype)
    values = abs_cosine_alignment(dirs, target.reshape(1, -1).expand_as(dirs))
    return float(torch.max(values).detach().cpu())


def max_spectral_alignment(model: ToricPJBias, target: torch.Tensor) -> float:
    omegas = model.omega[0].detach().to(device=target.device, dtype=target.dtype)
    values = abs_cosine_alignment(omegas, target.reshape(1, -1).expand_as(omegas))
    return float(torch.max(values).detach().cpu())


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = ["E2_diagonal_copy", "E4_signed_jet_teacher"]
    rows = [train_task(args, task) for task in tasks]

    csv_rows = []
    for row in rows:
        csv_rows.append({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items() if key != "history"})
    csv_path = output_dir / "synthetic_attention_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    plot_path = output_dir / "synthetic_attention_training.png"
    plot_training(rows, plot_path)
    summary = {
        "device": str(default_device(args.device)),
        "side": args.side,
        "num_positions": args.side * args.side,
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "synthetic_attention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_training(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for row in rows:
        ax.semilogy(row["history"], label=str(row["task"]))
    ax.set_title("Synthetic Attention Teacher Distillation")
    ax.set_xlabel("logged step")
    ax.set_ylabel("KL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic attention bridge tasks.")
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--gate-l1", type=float, default=1e-5)
    parser.add_argument("--freeze-geometry", action="store_true", default=True)
    parser.add_argument("--learn-geometry", dest="freeze_geometry", action="store_false")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage4")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
