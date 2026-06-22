from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


@dataclass(frozen=True)
class RetrievalTask:
    name: str
    target_index: torch.Tensor
    target_direction: torch.Tensor
    target_omega: torch.Tensor


class BasisAttention(nn.Module):
    def __init__(self, basis: Basis, n_positions: int) -> None:
        super().__init__()
        self.basis = basis
        self.n_positions = n_positions
        self.coeff = nn.Parameter(torch.zeros(basis.matrix.shape[1], device=basis.matrix.device, dtype=basis.matrix.dtype))

    def logits(self) -> torch.Tensor:
        return (self.basis.matrix @ self.coeff).reshape(self.n_positions, self.n_positions)

    def attention(self) -> torch.Tensor:
        return torch.softmax(self.logits(), dim=-1)


def make_positions(side: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(side, device=device, dtype=torch.float64)
    xx, yy = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def pairwise_d(positions: torch.Tensor) -> torch.Tensor:
    return positions[:, None, :] - positions[None, :, :]


def diagonal_origin_targets(positions: torch.Tensor, side: int) -> torch.Tensor:
    xy = positions.to(torch.long)
    x = xy[:, 0]
    y = xy[:, 1]
    m = torch.minimum(x, y)
    key_x = x - m
    key_y = y - m
    return key_x * side + key_y


def signed_jet_targets(d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = normalize_direction(torch.tensor([[1.0, -0.6]], device=d.device, dtype=d.dtype)).reshape(-1)
    omega = torch.tensor([0.37, 0.61], device=d.device, dtype=d.dtype)
    scale = torch.max(torch.abs(d)).clamp_min(1.0)
    logits = 4.0 * ((d @ u) / scale) * torch.cos(phase(d.reshape(-1, 2), omega)).reshape_as(d[..., 0])
    logits = logits - 0.08 * torch.linalg.norm(d, dim=-1)
    return torch.argmax(logits, dim=-1), u, omega


def build_tasks(positions: torch.Tensor, side: int) -> list[RetrievalTask]:
    d = pairwise_d(positions)
    diag_dir = normalize_direction(torch.tensor([[1.0, 1.0]], device=positions.device, dtype=positions.dtype)).reshape(-1)
    diag_omega = 0.78 * diag_dir
    e4_index, e4_dir, e4_omega = signed_jet_targets(d)
    return [
        RetrievalTask("E2_diagonal_origin_copy", diagonal_origin_targets(positions, side), diag_dir, diag_omega),
        RetrievalTask("E4_signed_jet_argmax", e4_index, e4_dir, e4_omega),
    ]


def no_pos_basis(n_positions: int, device: torch.device) -> Basis:
    matrix = torch.ones((n_positions * n_positions, 1), device=device, dtype=torch.float64)
    return Basis("NoPE_uniform", matrix, ["const"], [0])


def raster_1d_basis(d: torch.Tensor, side: int) -> Basis:
    lag = d[..., 0] * side + d[..., 1]
    flat = lag.reshape(-1)
    cols = [torch.ones_like(flat)]
    labels = ["const"]
    orders = [0]
    for idx, freq in enumerate([0.07, 0.15, 0.33, 0.61]):
        cols.extend([torch.cos(freq * flat), torch.sin(freq * flat)])
        labels.extend([f"raster_cos{idx}", f"raster_sin{idx}"])
        orders.extend([0, 0])
    matrix, _ = normalize_columns(torch.stack(cols, dim=1))
    return Basis("raster_1d", matrix, labels, orders)


def relative_table_basis(d: torch.Tensor) -> Basis:
    flat = d.reshape(-1, 2).to(torch.long)
    unique, inverse = torch.unique(flat, dim=0, return_inverse=True)
    matrix = torch.zeros((flat.shape[0], unique.shape[0]), device=d.device, dtype=torch.float64)
    matrix[torch.arange(flat.shape[0], device=d.device), inverse] = 1.0
    matrix, _ = normalize_columns(matrix)
    labels = [f"rel_{int(x)}_{int(y)}" for x, y in unique.detach().cpu().tolist()]
    return Basis("relative_2d_table", matrix, labels, [0] * len(labels))


def build_bases(d: torch.Tensor, side: int) -> list[Basis]:
    flat = d.reshape(-1, 2)
    omega_diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1) * 0.78
    omega_signed = torch.tensor([0.37, 0.61], device=d.device, dtype=d.dtype)
    omega_oblique = torch.tensor([0.62, -0.58], device=d.device, dtype=d.dtype)
    ex = torch.tensor([1.0, 0.0], device=d.device, dtype=d.dtype)
    ey = torch.tensor([0.0, 1.0], device=d.device, dtype=d.dtype)
    u_signed = normalize_direction(torch.tensor([[1.0, -0.6]], device=d.device, dtype=d.dtype)).reshape(-1)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1)
    return [
        no_pos_basis(d.shape[0], d.device),
        raster_1d_basis(d, side),
        axis_additive_fourier_basis(flat, [0.78, 0.61], name="axis_additive"),
        toric_fourier_basis(flat, [omega_diag, omega_signed, omega_oblique], name="toric_order0"),
        directional_jet_basis(
            flat,
            [omega_diag, omega_signed, omega_oblique],
            [ex, ey, u_signed, diag],
            [0, 1, 2],
            scale=float(side),
            name="toric_PJ_R2",
        ),
        relative_table_basis(d),
    ]


def train_basis(
    basis: Basis,
    target_index: torch.Tensor,
    *,
    n_positions: int,
    steps: int,
    lr: float,
    coeff_l2: float,
) -> tuple[BasisAttention, list[float]]:
    model = BasisAttention(basis, n_positions)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rows = torch.arange(n_positions, device=target_index.device)
    history: list[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model.logits()
        loss = F.cross_entropy(logits, target_index)
        loss = loss + coeff_l2 * model.coeff.square().mean()
        loss.backward()
        opt.step()
        if step % max(1, steps // 100) == 0 or step == steps - 1:
            with torch.no_grad():
                nll = F.cross_entropy(model.logits(), target_index)
                history.append(float(nll.detach().cpu()))
    return model, history


def evaluate_model(
    model: BasisAttention,
    task: RetrievalTask,
    *,
    vocab_size: int,
    eval_batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, float]:
    n = model.n_positions
    attn = model.attention()
    top_key = torch.argmax(attn, dim=-1)
    key_agreement = torch.mean((top_key == task.target_index).float())
    gen = torch.Generator(device=task.target_index.device)
    gen.manual_seed(seed)
    acc_values = []
    nll_values = []
    for _ in range(eval_batches):
        tokens = torch.randint(0, vocab_size, (batch_size, n), device=task.target_index.device, generator=gen)
        one_hot = F.one_hot(tokens, num_classes=vocab_size).to(attn.dtype)
        probs = torch.einsum("qk,bkv->bqv", attn, one_hot).clamp_min(1e-12)
        target_tokens = tokens[:, task.target_index]
        pred = torch.argmax(probs, dim=-1)
        acc_values.append(torch.mean((pred == target_tokens).float()))
        nll = -torch.log(torch.gather(probs, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1))
        nll_values.append(torch.mean(nll))
    entropy = -torch.sum(attn * torch.log(attn.clamp_min(1e-12)), dim=-1).mean()
    return {
        "key_top1": float(key_agreement.detach().cpu()),
        "token_accuracy": float(torch.stack(acc_values).mean().detach().cpu()),
        "token_nll": float(torch.stack(nll_values).mean().detach().cpu()),
        "attention_entropy": float(entropy.detach().cpu()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = make_positions(args.side, device)
    d = pairwise_d(positions)
    tasks = build_tasks(positions, args.side)
    bases = build_bases(d, args.side)
    rows: list[dict[str, object]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    for task in tasks:
        histories[task.name] = {}
        for basis in bases:
            model, history = train_basis(
                basis,
                task.target_index,
                n_positions=positions.shape[0],
                steps=args.steps,
                lr=args.lr,
                coeff_l2=args.coeff_l2,
            )
            metrics = evaluate_model(
                model,
                task,
                vocab_size=args.vocab_size,
                eval_batches=args.eval_batches,
                batch_size=args.batch_size,
                seed=args.seed + len(rows),
            )
            with torch.no_grad():
                nll = F.cross_entropy(model.logits(), task.target_index)
            rows.append(
                {
                    "task": task.name,
                    "basis": basis.name,
                    "key_nll": float(nll.detach().cpu()),
                    "num_features": basis.matrix.shape[1],
                    **metrics,
                }
            )
            histories[task.name][basis.name] = history

    csv_path = output_dir / "trainable_attention_retrieval_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = output_dir / "trainable_attention_retrieval.png"
    plot_results(rows, plot_path)
    summary = {
        "device": str(device),
        "side": args.side,
        "num_positions": int(positions.shape[0]),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
        "histories": histories,
    }
    summary_path = output_dir / "trainable_attention_retrieval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    tasks = sorted({str(row["task"]) for row in rows})
    bases = [str(row["basis"]) for row in rows if row["task"] == tasks[0]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric, title in [
        (axes[0], "key_top1", "Teacher Key Top-1 Agreement"),
        (axes[1], "token_accuracy", "Random Token Retrieval Accuracy"),
    ]:
        width = 0.38
        x = np.arange(len(bases))
        for idx, task in enumerate(tasks):
            values = [float(row[metric]) for row in rows if row["task"] == task]
            ax.bar(x + (idx - 0.5) * width, values, width=width, label=task)
        ax.set_xticks(x, bases, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trainable synthetic attention retrieval bridge.")
    parser.add_argument("--side", type=int, default=12)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--coeff-l2", type=float, default=1e-7)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage4_retrieval")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
