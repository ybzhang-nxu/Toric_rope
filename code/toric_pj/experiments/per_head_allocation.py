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

from toric_pj.diagnostics.basis_projection import default_device, make_grid_2d
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.experiments.adaptive_teacher_2d import BRANCHES, build_dictionary, build_targets


class MultiHeadAllocationModel(nn.Module):
    def __init__(self, matrix: torch.Tensor, branch_ids: torch.Tensor, *, n_heads: int, shared_gates: bool) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.register_buffer("branch_ids", branch_ids)
        self.n_heads = n_heads
        self.shared_gates = shared_gates
        self.coeff = nn.Parameter(1e-3 * torch.randn(n_heads, matrix.shape[1], device=matrix.device, dtype=matrix.dtype))
        gate_shape = (1, len(BRANCHES)) if shared_gates else (n_heads, len(BRANCHES))
        self.gate_logits = nn.Parameter(torch.zeros(gate_shape, device=matrix.device, dtype=matrix.dtype))

    @property
    def gates(self) -> torch.Tensor:
        gates = F.softplus(self.gate_logits)
        if self.shared_gates:
            return gates.expand(self.n_heads, -1)
        return gates

    @property
    def gate_shares(self) -> torch.Tensor:
        gates = self.gates
        return gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def branch_outputs(self) -> torch.Tensor:
        raw = torch.einsum("nm,hm->nhm", self.matrix, self.coeff)
        outs = []
        for branch_id in range(len(BRANCHES)):
            mask = self.branch_ids == branch_id
            outs.append(raw[:, :, mask].sum(dim=-1) if torch.any(mask) else torch.zeros(raw.shape[:2], device=raw.device))
        return torch.stack(outs, dim=-1)

    def head_outputs(self) -> torch.Tensor:
        return (self.branch_outputs() * self.gates.reshape(1, self.n_heads, len(BRANCHES))).sum(dim=-1)

    def forward(self) -> torch.Tensor:
        return self.head_outputs().mean(dim=1)


def gate_entropy(shares: torch.Tensor) -> torch.Tensor:
    return -torch.sum(shares * torch.log(shares.clamp_min(1e-12)), dim=-1).mean()


def gate_balance(shares: torch.Tensor) -> torch.Tensor:
    mean_share = shares.mean(dim=0)
    target = torch.full_like(mean_share, 1.0 / float(mean_share.numel()))
    return torch.mean((mean_share - target).square())


def gate_diversity(shares: torch.Tensor) -> float:
    if shares.shape[0] <= 1:
        return 0.0
    distances = []
    for i in range(shares.shape[0]):
        for j in range(i + 1, shares.shape[0]):
            distances.append(torch.mean(torch.abs(shares[i] - shares[j])))
    return float(torch.stack(distances).mean().detach().cpu())


def train_model(
    matrix: torch.Tensor,
    branch_ids: torch.Tensor,
    target: torch.Tensor,
    *,
    n_heads: int,
    shared_gates: bool,
    steps: int,
    lr: float,
    gate_l1: float,
    coeff_l2: float,
    entropy_weight: float,
    balance_weight: float,
    seed: int,
) -> tuple[MultiHeadAllocationModel, list[float]]:
    torch.manual_seed(seed)
    model = MultiHeadAllocationModel(matrix, branch_ids, n_heads=n_heads, shared_gates=shared_gates)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history: list[float] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model()
        mse = torch.mean((pred - target).square())
        shares = model.gate_shares
        reg = gate_l1 * model.gates.mean() + coeff_l2 * model.coeff.square().mean()
        if not shared_gates:
            reg = reg + entropy_weight * gate_entropy(shares) + balance_weight * gate_balance(shares)
        loss = mse + reg
        loss.backward()
        opt.step()
        if step % max(1, steps // 100) == 0 or step == steps - 1:
            history.append(float(mse.detach().cpu()))
    return model, history


def diagnostic_rows(
    *,
    target_name: str,
    model_name: str,
    model: MultiHeadAllocationModel,
    target: torch.Tensor,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    with torch.no_grad():
        pred = model()
        mse = torch.mean((pred - target).square())
        var = torch.mean((target - target.mean()).square()).clamp_min(1e-30)
        r2 = 1.0 - mse / var
        shares = model.gate_shares
        head_outputs = model.head_outputs()
        total_norm = torch.linalg.norm(pred).clamp_min(1e-30)
        head_energy = torch.linalg.norm(head_outputs, dim=0) / total_norm
        metric = {
            "target": target_name,
            "model": model_name,
            "mse": float(mse.detach().cpu()),
            "r2": float(r2.detach().cpu()),
            "gate_entropy": float(gate_entropy(shares).detach().cpu()),
            "gate_diversity": gate_diversity(shares),
            "top_branch_counts": json.dumps(
                {
                    BRANCHES[idx]: int(torch.sum(torch.argmax(shares, dim=-1) == idx).detach().cpu())
                    for idx in range(len(BRANCHES))
                },
                sort_keys=True,
            ),
        }
        gate_rows = []
        for head in range(model.n_heads):
            for branch_idx, branch in enumerate(BRANCHES):
                gate_rows.append(
                    {
                        "target": target_name,
                        "model": model_name,
                        "head": head,
                        "branch": branch,
                        "gate_share": float(shares[head, branch_idx].detach().cpu()),
                        "head_energy": float(head_energy[head].detach().cpu()),
                    }
                )
    return metric, gate_rows


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    dtype = torch.float64
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    d = make_grid_2d(args.radius, signed=True, device=device, dtype=dtype)

    omega_main = torch.tensor([0.37, 0.61], device=device, dtype=dtype)
    omega_beat = torch.tensor([0.0, 2.0 * torch.pi / 8.0], device=device, dtype=dtype)
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
    branch_ids = torch.tensor([BRANCHES.index(meta.branch) for meta in metas], device=device, dtype=torch.long)
    targets = build_targets(
        d,
        radius=args.radius,
        omega_main=omega_main,
        omega_beat=omega_beat,
        u_star=u_star,
        s_star=s_star,
        omega_lc=omega_lc,
    )

    metric_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    histories: dict[str, dict[str, list[float]]] = {}
    for target_name, target in targets.items():
        histories[target_name] = {}
        for model_name, shared in [("shared_gates", True), ("per_head_gates", False)]:
            model, history = train_model(
                matrix,
                branch_ids,
                target,
                n_heads=args.n_heads,
                shared_gates=shared,
                steps=args.steps,
                lr=args.lr,
                gate_l1=args.gate_l1,
                coeff_l2=args.coeff_l2,
                entropy_weight=args.entropy_weight,
                balance_weight=args.balance_weight,
                seed=args.seed + len(metric_rows),
            )
            metric, gates = diagnostic_rows(target_name=target_name, model_name=model_name, model=model, target=target)
            metric_rows.append(metric)
            gate_rows.extend(gates)
            histories[target_name][model_name] = history

    metrics_csv = output_dir / "per_head_allocation_metrics.csv"
    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    gates_csv = output_dir / "per_head_allocation_gates.csv"
    with gates_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gate_rows)

    plot_path = output_dir / "per_head_allocation_gates.png"
    plot_gates(gate_rows, plot_path)
    summary = {
        "device": str(device),
        "metrics_csv": str(metrics_csv),
        "gates_csv": str(gates_csv),
        "plot": str(plot_path),
        "metric_rows": metric_rows,
        "gate_rows": gate_rows,
        "histories": histories,
    }
    summary_path = output_dir / "per_head_allocation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, summary)
    return summary


def plot_gates(rows: list[dict[str, object]], path: Path) -> None:
    targets = sorted({str(row["target"]) for row in rows})
    branches = list(BRANCHES)
    fig, axes = plt.subplots(len(targets), 2, figsize=(9, 3.2 * len(targets)), squeeze=False)
    for i, target in enumerate(targets):
        for j, model in enumerate(["shared_gates", "per_head_gates"]):
            ax = axes[i, j]
            values = [row for row in rows if row["target"] == target and row["model"] == model]
            heads = sorted({int(row["head"]) for row in values})
            matrix = np.zeros((len(heads), len(branches)))
            for row in values:
                matrix[heads.index(int(row["head"])), branches.index(str(row["branch"]))] = float(row["gate_share"])
            im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
            ax.set_xticks(np.arange(len(branches)), branches)
            ax.set_yticks(np.arange(len(heads)), [f"h{head}" for head in heads])
            ax.set_title(f"{target} / {model}")
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object]) -> None:
    lines = [
        "# V2-D Per-Head Allocation Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        "",
        "Run command:",
        "",
        "```bash",
        "python scripts/run_v2_per_head_allocation.py --device cuda --output-dir results/v2_per_head_allocation",
        "```",
        "",
        "## Metrics",
        "",
        "| target | model | R2 | gate entropy | gate diversity | top branch counts |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary["metric_rows"]:
        lines.append(
            "| "
            + f"{row['target']} | {row['model']} | {float(row['r2']):.4f} | "
            + f"{float(row['gate_entropy']):.4f} | {float(row['gate_diversity']):.4f} | `{row['top_branch_counts']}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `shared_gates` and `per_head_gates` use the same number of heads and per-head coefficients.",
            "- `per_head_gates` only changes the allocation mechanism: each head gets independent FJ / affine / LC gates.",
            "- Gate diversity is the mean pairwise L1 distance between head gate-share vectors.",
            "- Higher diversity with similar R2 is evidence for interpretable head specialization.",
            "",
            "Artifacts:",
            "",
            "- `per_head_allocation_metrics.csv`",
            "- `per_head_allocation_gates.csv`",
            "- `per_head_allocation_summary.json`",
            "- `per_head_allocation_gates.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2 per-head allocation experiment.")
    parser.add_argument("--radius", type=int, default=18)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--gate-l1", type=float, default=1e-4)
    parser.add_argument("--coeff-l2", type=float, default=1e-7)
    parser.add_argument("--entropy-weight", type=float, default=2e-3)
    parser.add_argument("--balance-weight", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_per_head_allocation")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key not in {"gate_rows", "histories"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
