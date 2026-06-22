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
from torch import nn
from torch.nn import functional as F

from toric_pj.diagnostics.basis_projection import Basis, axis_additive_fourier_basis, default_device, directional_jet_basis, normalize_columns, toric_fourier_basis
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.experiments.real_digits_probe import make_positions, pairwise_d, raster_1d_basis, relative_2d_table_basis


def load_digits_split(device: torch.device, *, seed: int, train_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = load_digits()
    images = torch.tensor(data.data, device=device, dtype=torch.float32) / 16.0
    labels = torch.tensor(data.target, device=device, dtype=torch.long)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    perm = torch.randperm(images.shape[0], device=device, generator=gen)
    train_idx = perm[:train_count]
    test_idx = perm[train_count:]
    train_x = images[train_idx]
    test_x = images[test_idx]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std().clamp_min(1e-6)
    return (train_x - mean) / std, labels[train_idx], (test_x - mean) / std, labels[test_idx]


def no_pos_basis(d: torch.Tensor) -> Basis:
    matrix = torch.ones((d.shape[0], 1), device=d.device, dtype=d.dtype)
    return Basis("no_pos_constant", matrix, ["const"], [0])


def build_bases(d: torch.Tensor, side: int) -> list[Basis]:
    omega_a = torch.tensor([0.78, 0.42], device=d.device, dtype=d.dtype)
    omega_b = torch.tensor([0.62, -0.58], device=d.device, dtype=d.dtype)
    omega_c = torch.tensor([0.35, 0.91], device=d.device, dtype=d.dtype)
    ex = torch.tensor([1.0, 0.0], device=d.device, dtype=d.dtype)
    ey = torch.tensor([0.0, 1.0], device=d.device, dtype=d.dtype)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1)
    oblique = normalize_direction(torch.tensor([[1.0, -0.65]], device=d.device, dtype=d.dtype)).reshape(-1)
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
        relative_2d_table_basis(d),
    ]


class PositionalTransformer(nn.Module):
    def __init__(self, basis: Basis, *, n_positions: int, input_dim: int, dim: int, n_heads: int, n_classes: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        matrix, _ = normalize_columns(basis.matrix.to(dtype=torch.float32))
        self.register_buffer("basis_matrix", matrix)
        self.n_positions = n_positions
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.coeff = nn.Parameter(torch.zeros(n_heads, matrix.shape[1], device=matrix.device, dtype=torch.float32))
        self.input = nn.Linear(input_dim, dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.reconstruct = nn.Linear(dim, 1)
        self.classifier = nn.Linear(dim, n_classes)

    def attention_bias(self) -> torch.Tensor:
        logits = torch.einsum("nf,hf->hn", self.basis_matrix, self.coeff)
        return logits.reshape(self.n_heads, self.n_positions, self.n_positions)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input(features)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        bsz = x.shape[0]
        q = q.reshape(bsz, self.n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, self.n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, self.n_positions, self.n_heads, self.head_dim).transpose(1, 2)
        logits = torch.einsum("bhqd,bhkd->bhqk", q, k) / float(self.head_dim) ** 0.5
        logits = logits + self.attention_bias().unsqueeze(0)
        attn = torch.softmax(logits, dim=-1)
        context = torch.einsum("bhqk,bhkd->bhqd", attn, v).transpose(1, 2).reshape(bsz, self.n_positions, -1)
        x = self.norm1(x + self.out(context))
        x = self.norm2(x + self.ff(x))
        return x

    def forward_reconstruction(self, pixels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        visible = pixels.masked_fill(mask, 0.0)
        features = torch.stack([visible, (~mask).to(pixels.dtype)], dim=-1)
        encoded = self.encode(features)
        return self.reconstruct(encoded).squeeze(-1)

    def forward_classification(self, pixels: torch.Tensor) -> torch.Tensor:
        features = torch.stack([pixels, torch.ones_like(pixels)], dim=-1)
        encoded = self.encode(features)
        return self.classifier(encoded.mean(dim=1))


def sample_batch(x: torch.Tensor, y: torch.Tensor, *, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device, generator=generator)
    return x[idx], y[idx]


def train_reconstruction(
    basis: Basis,
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=train_x.device)
    gen.manual_seed(seed + 1000)
    model = PositionalTransformer(
        basis,
        n_positions=train_x.shape[1],
        input_dim=2,
        dim=args.dim,
        n_heads=args.n_heads,
        n_classes=10,
    ).to(train_x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.recon_steps):
        batch, _ = sample_batch(train_x, torch.zeros(train_x.shape[0], device=train_x.device, dtype=torch.long), batch_size=args.batch_size, generator=gen)
        mask = torch.rand(batch.shape, device=batch.device, generator=gen) < args.mask_rate
        opt.zero_grad(set_to_none=True)
        pred = model.forward_reconstruction(batch, mask)
        loss = torch.mean((pred[mask] - batch[mask]).square())
        loss.backward()
        opt.step()
    return evaluate_reconstruction(model, basis.name, test_x, mask_rate=args.mask_rate, batch_size=args.eval_batch_size, seed=seed + 2000)


def evaluate_reconstruction(
    model: PositionalTransformer,
    basis_name: str,
    test_x: torch.Tensor,
    *,
    mask_rate: float,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    gen = torch.Generator(device=test_x.device)
    gen.manual_seed(seed)
    mses = []
    targets = []
    preds = []
    with torch.no_grad():
        for start in range(0, test_x.shape[0], batch_size):
            batch = test_x[start : start + batch_size]
            mask = torch.rand(batch.shape, device=batch.device, generator=gen) < mask_rate
            pred = model.forward_reconstruction(batch, mask)
            mses.append(torch.mean((pred[mask] - batch[mask]).square()))
            targets.append(batch[mask])
            preds.append(pred[mask])
    target = torch.cat(targets)
    pred_all = torch.cat(preds)
    mse = torch.mean((pred_all - target).square())
    var = torch.mean((target - target.mean()).square()).clamp_min(1e-30)
    return {
        "task": "masked_reconstruction",
        "basis": basis_name,
        "metric": "masked_r2",
        "score": float((1.0 - mse / var).detach().cpu()),
        "loss": float(torch.stack(mses).mean().detach().cpu()),
        "num_features": model.basis_matrix.shape[1],
    }


def train_classifier(
    basis: Basis,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=train_x.device)
    gen.manual_seed(seed + 3000)
    model = PositionalTransformer(
        basis,
        n_positions=train_x.shape[1],
        input_dim=2,
        dim=args.dim,
        n_heads=args.n_heads,
        n_classes=10,
    ).to(train_x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.cls_steps):
        batch, labels = sample_batch(train_x, train_y, batch_size=args.batch_size, generator=gen)
        opt.zero_grad(set_to_none=True)
        logits = model.forward_classification(batch)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
    return evaluate_classifier(model, basis.name, test_x, test_y, batch_size=args.eval_batch_size)


def evaluate_classifier(
    model: PositionalTransformer,
    basis_name: str,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, object]:
    losses = []
    accs = []
    with torch.no_grad():
        for start in range(0, test_x.shape[0], batch_size):
            batch = test_x[start : start + batch_size]
            labels = test_y[start : start + batch_size]
            logits = model.forward_classification(batch)
            losses.append(F.cross_entropy(logits, labels))
            accs.append(torch.mean((torch.argmax(logits, dim=-1) == labels).float()))
    return {
        "task": "classification",
        "basis": basis_name,
        "metric": "accuracy",
        "score": float(torch.stack(accs).mean().detach().cpu()),
        "loss": float(torch.stack(losses).mean().detach().cpu()),
        "num_features": model.basis_matrix.shape[1],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y = load_digits_split(device, seed=args.seed, train_count=args.train_count)
    positions = make_positions(args.side, device)
    d = pairwise_d(positions).reshape(-1, 2).to(torch.float32)
    bases = build_bases(d, args.side)
    rows: list[dict[str, object]] = []
    for idx, basis in enumerate(bases):
        rows.append(train_reconstruction(basis, train_x, test_x, args=args, seed=args.seed + idx))
        rows.append(train_classifier(basis, train_x, train_y, test_x, test_y, args=args, seed=args.seed + idx + 100))

    csv_path = output_dir / "real_digits_transformer_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = output_dir / "real_digits_transformer_scores.png"
    plot_results(rows, plot_path)
    summary = {
        "device": str(device),
        "dataset": "sklearn_digits",
        "train_count": args.train_count,
        "test_count": int(test_x.shape[0]),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "real_digits_transformer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, rows, summary)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    tasks = ["masked_reconstruction", "classification"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, task in zip(axes, tasks):
        values = [row for row in rows if row["task"] == task]
        labels = [str(row["basis"]) for row in values]
        scores = [float(row["score"]) for row in values]
        x = np.arange(len(labels))
        ax.bar(x, scores, color="#4e746f")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(task)
        ax.set_ylabel(values[0]["metric"])
        for idx, value in enumerate(scores):
            ax.text(idx, min(value + 0.02, 1.02), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# V2-E Real Digits Transformer Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        "- Dataset: sklearn digits",
        f"- Train / test: {summary['train_count']} / {summary['test_count']}",
        "",
        "Run command:",
        "",
        "```bash",
        "python scripts/run_v2_real_digits.py --device cuda --output-dir results/v2_real_digits",
        "```",
        "",
        "## Scores",
        "",
        "| task | basis | score | loss | features |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['task']} | {row['basis']} | {float(row['score']):.4f} | {float(row['loss']):.4f} | {int(row['num_features'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- The same small Transformer architecture is used for every basis; only the relative positional bias dictionary changes.",
            "- Masked reconstruction reports masked-pixel R2.",
            "- Classification reports test accuracy.",
            "- Relative table is the high-capacity upper-bound style baseline; Toric PJ is the compact structured baseline.",
            "",
            "Artifacts:",
            "",
            "- `real_digits_transformer_results.csv`",
            "- `real_digits_transformer_summary.json`",
            "- `real_digits_transformer_scores.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-E real digits small Transformer benchmark.")
    parser.add_argument("--side", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=1400)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--recon-steps", type=int, default=260)
    parser.add_argument("--cls-steps", type=int, default=320)
    parser.add_argument("--mask-rate", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=515)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_real_digits")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
