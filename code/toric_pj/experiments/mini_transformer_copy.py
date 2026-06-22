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

from toric_pj.experiments.trainable_attention_retrieval import (
    RetrievalTask,
    build_bases,
    build_tasks,
    make_positions,
    pairwise_d,
)
from toric_pj.diagnostics.basis_projection import Basis, default_device


class PositionalCopyTransformer(nn.Module):
    """A minimal content model with trainable positional attention logits.

    It uses token embeddings as values and learns attention through a fixed
    positional dictionary. There is no direct supervision on teacher keys.
    """

    def __init__(self, basis: Basis, n_positions: int, vocab_size: int, dim: int, n_heads: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.basis = basis
        self.n_positions = n_positions
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.coeff = nn.Parameter(torch.zeros(n_heads, basis.matrix.shape[1], device=basis.matrix.device, dtype=torch.float32))
        self.embedding = nn.Embedding(vocab_size, dim)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, vocab_size)
        self.norm = nn.LayerNorm(dim)

    def attention_logits(self) -> torch.Tensor:
        matrix = self.basis.matrix.to(dtype=self.coeff.dtype)
        logits = torch.einsum("nm,hm->hn", matrix, self.coeff)
        return logits.reshape(self.n_heads, self.n_positions, self.n_positions)

    def attention(self) -> torch.Tensor:
        return torch.softmax(self.attention_logits(), dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(tokens)
        values = self.value(emb).reshape(tokens.shape[0], self.n_positions, self.n_heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)
        attn = self.attention()
        context = torch.einsum("hqk,bhkd->bhqd", attn, values)
        context = context.permute(0, 2, 1, 3).reshape(tokens.shape[0], self.n_positions, self.dim)
        return self.out(self.norm(context))


def train_model(
    basis: Basis,
    task: RetrievalTask,
    *,
    vocab_size: int,
    dim: int,
    n_heads: int,
    steps: int,
    batch_size: int,
    lr: float,
    coeff_l2: float,
    seed: int,
) -> tuple[PositionalCopyTransformer, list[float]]:
    torch.manual_seed(seed)
    device = task.target_index.device
    model = PositionalCopyTransformer(
        basis,
        n_positions=task.target_index.numel(),
        vocab_size=vocab_size,
        dim=dim,
        n_heads=n_heads,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1000)
    history: list[float] = []
    for step in range(steps):
        tokens = torch.randint(0, vocab_size, (batch_size, task.target_index.numel()), device=device, generator=gen)
        targets = tokens[:, task.target_index]
        opt.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss = loss + coeff_l2 * model.coeff.square().mean()
        loss.backward()
        opt.step()
        if step % max(1, steps // 100) == 0 or step == steps - 1:
            history.append(float(loss.detach().cpu()))
    return model, history


def evaluate_model(
    model: PositionalCopyTransformer,
    task: RetrievalTask,
    *,
    vocab_size: int,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> dict[str, float]:
    device = task.target_index.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 2000)
    accs = []
    losses = []
    with torch.no_grad():
        attn = model.attention().mean(dim=0)
        top_key = torch.argmax(attn, dim=-1)
        key_top1 = torch.mean((top_key == task.target_index).float())
        attn_entropy = -torch.sum(attn * torch.log(attn.clamp_min(1e-12)), dim=-1).mean()
        for _ in range(eval_batches):
            tokens = torch.randint(0, vocab_size, (batch_size, task.target_index.numel()), device=device, generator=gen)
            targets = tokens[:, task.target_index]
            logits = model(tokens)
            losses.append(F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1)))
            pred = torch.argmax(logits, dim=-1)
            accs.append(torch.mean((pred == targets).float()))
    return {
        "eval_loss": float(torch.stack(losses).mean().detach().cpu()),
        "token_accuracy": float(torch.stack(accs).mean().detach().cpu()),
        "key_top1": float(key_top1.detach().cpu()),
        "attention_entropy": float(attn_entropy.detach().cpu()),
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
            model, history = train_model(
                basis,
                task,
                vocab_size=args.vocab_size,
                dim=args.dim,
                n_heads=args.n_heads,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                coeff_l2=args.coeff_l2,
                seed=args.seed + len(rows),
            )
            metrics = evaluate_model(
                model,
                task,
                vocab_size=args.vocab_size,
                batch_size=args.batch_size,
                eval_batches=args.eval_batches,
                seed=args.seed + len(rows),
            )
            rows.append(
                {
                    "task": task.name,
                    "basis": basis.name,
                    "num_features": basis.matrix.shape[1],
                    **metrics,
                }
            )
            histories[task.name][basis.name] = history

    csv_path = output_dir / "mini_transformer_copy_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "mini_transformer_copy.png"
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
    summary_path = output_dir / "mini_transformer_copy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    tasks = sorted({str(row["task"]) for row in rows})
    bases = [str(row["basis"]) for row in rows if row["task"] == tasks[0]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, metric, title in [
        (axes[0], "token_accuracy", "Token Copy Accuracy"),
        (axes[1], "key_top1", "Mean-Head Key Top-1"),
    ]:
        x = np.arange(len(bases))
        width = 0.38
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
    parser = argparse.ArgumentParser(description="Run minimal Transformer random-token copy tasks.")
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.012)
    parser.add_argument("--coeff-l2", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage4_transformer")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
