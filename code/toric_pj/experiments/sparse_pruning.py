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

from toric_pj.diagnostics.basis_projection import Basis, default_device, normalize_columns
from toric_pj.experiments import lattice_physics_probe, real_digits_probe


def group_from_label(label: str) -> str:
    if label == "const":
        return "const"
    parts = label.split("_")
    if len(parts) >= 4 and parts[0].startswith("w") and parts[1].startswith("u") and parts[2].startswith("r"):
        return "_".join(parts[:3])
    if len(parts) >= 3 and parts[0].startswith("w") and parts[1].startswith("r"):
        return "_".join(parts[:2])
    return "_".join(parts[:-1]) if "_" in label else label


def fit_selected(
    train_matrix: torch.Tensor,
    eval_matrix: torch.Tensor,
    train_target: torch.Tensor,
    eval_target: torch.Tensor,
    *,
    selected: list[int],
    ridge: float,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train_full, norms = normalize_columns(train_matrix)
    x_eval_full = eval_matrix / norms.clamp_min(1e-12)
    idx = torch.tensor(selected, device=train_matrix.device, dtype=torch.long)
    x_train = x_train_full[:, idx]
    x_eval = x_eval_full[:, idx]
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
    metrics = {
        "train_r2": float((1.0 - train_mse / train_var).detach().cpu()),
        "eval_r2": float((1.0 - eval_mse / eval_var).detach().cpu()),
        "train_mse": float(train_mse.detach().cpu()),
        "eval_mse": float(eval_mse.detach().cpu()),
    }
    return metrics, coeff, x_train_full, x_eval_full


def group_diagnostics(
    train_matrix: torch.Tensor,
    train_target: torch.Tensor,
    basis: Basis,
    *,
    ridge: float,
) -> tuple[dict[str, float], dict[str, float]]:
    selected = list(range(train_matrix.shape[1]))
    _, coeff, x_train, _ = fit_selected(train_matrix, train_matrix, train_target, train_target, selected=selected, ridge=ridge)
    y = train_target.reshape(-1, 1)
    pred = x_train @ coeff
    base_mse = torch.mean((pred - y).square())
    denom = torch.linalg.norm(pred).clamp_min(1e-30)
    groups = sorted({group_from_label(label) for label in basis.labels})
    loo: dict[str, float] = {}
    energy: dict[str, float] = {}
    for group in groups:
        idx = torch.tensor(
            [i for i, label in enumerate(basis.labels) if group_from_label(label) == group],
            device=train_matrix.device,
            dtype=torch.long,
        )
        contrib = x_train[:, idx] @ coeff[idx]
        energy[group] = float((torch.linalg.norm(contrib) / denom).detach().cpu())
        loo[group] = float((torch.mean((pred - contrib - y).square()) - base_mse).detach().cpu())
    return loo, energy


def select_groups(groups: dict[str, float], *, fraction: float) -> list[str]:
    ordered = sorted(groups, key=lambda group: groups[group], reverse=True)
    keep = max(1, int(np.ceil(len(ordered) * fraction)))
    selected = set(ordered[:keep])
    selected.add("const")
    return sorted(selected)


def selected_columns_for_groups(basis: Basis, groups: list[str]) -> list[int]:
    group_set = set(groups)
    return [idx for idx, label in enumerate(basis.labels) if group_from_label(label) in group_set]


def group_lasso_ranked_groups(
    train_matrix: torch.Tensor,
    train_target: torch.Tensor,
    basis: Basis,
    *,
    ridge: float,
    fraction: float,
    steps: int = 350,
    lr: float = 0.08,
    penalty: float = 5e-4,
) -> list[str]:
    x_train, _ = normalize_columns(train_matrix)
    y = train_target.reshape(-1, 1)
    coeff = torch.zeros((x_train.shape[1], 1), device=x_train.device, dtype=x_train.dtype, requires_grad=True)
    opt = torch.optim.Adam([coeff], lr=lr)
    groups = sorted({group_from_label(label) for label in basis.labels})
    group_indices = {
        group: torch.tensor(
            [idx for idx, label in enumerate(basis.labels) if group_from_label(label) == group],
            device=x_train.device,
            dtype=torch.long,
        )
        for group in groups
    }
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = x_train @ coeff
        mse = torch.mean((pred - y).square())
        group_penalty = sum(torch.linalg.norm(coeff[idx]) for idx in group_indices.values())
        loss = mse + penalty * group_penalty + ridge * coeff.square().mean()
        loss.backward()
        opt.step()
    scores = {
        group: float(torch.linalg.norm(coeff[idx]).detach().cpu())
        for group, idx in group_indices.items()
    }
    selected = select_groups(scores, fraction=fraction)
    return selected


def run_frontier(
    *,
    task: str,
    basis: Basis,
    train_matrix: torch.Tensor,
    eval_matrix: torch.Tensor,
    train_target: torch.Tensor,
    eval_target: torch.Tensor,
    ridge: float,
) -> list[dict[str, object]]:
    loo, energy = group_diagnostics(train_matrix, train_target, basis, ridge=ridge)
    all_groups = sorted({group_from_label(label) for label in basis.labels})
    rows: list[dict[str, object]] = []
    variants: list[tuple[str, list[str]]] = [("dense", all_groups)]
    for fraction in [0.75, 0.55, 0.35, 0.20]:
        variants.append((f"loo_top_{fraction:.2f}", select_groups(loo, fraction=fraction)))
    for fraction in [0.75, 0.55, 0.35, 0.20]:
        variants.append((f"energy_top_{fraction:.2f}", select_groups(energy, fraction=fraction)))
    variants.append(
        (
            "group_lasso_top_0.55",
            group_lasso_ranked_groups(
                train_matrix,
                train_target,
                basis,
                ridge=ridge,
                fraction=0.55,
            ),
        )
    )

    dense_eval = None
    for variant, groups in variants:
        selected = selected_columns_for_groups(basis, groups)
        metrics, _, _, _ = fit_selected(
            train_matrix,
            eval_matrix,
            train_target,
            eval_target,
            selected=selected,
            ridge=ridge,
        )
        if variant == "dense":
            dense_eval = metrics["eval_r2"]
        retained = metrics["eval_r2"] / max(float(dense_eval), 1e-12) if dense_eval is not None else 1.0
        rows.append(
            {
                "task": task,
                "variant": variant,
                "num_groups": len(groups),
                "num_features": len(selected),
                "compression_ratio": len(selected) / basis.matrix.shape[1],
                "retained_eval_r2_frac": retained,
                "selected_groups": json.dumps(groups),
                **metrics,
            }
        )
    return rows


def lattice_task(device: torch.device, *, seed: int, ridge: float) -> list[dict[str, object]]:
    train_side = 10
    eval_side = 14
    train_pos = lattice_physics_probe.make_positions(train_side, device)
    eval_pos = lattice_physics_probe.make_positions(eval_side, device)
    train_d = lattice_physics_probe.pairwise_displacements(train_pos).reshape(-1, 2)
    eval_d = lattice_physics_probe.pairwise_displacements(eval_pos).reshape(-1, 2)
    train_target = lattice_physics_probe.wave_kernel_teacher(train_d, train_side)
    eval_target = lattice_physics_probe.wave_kernel_teacher(eval_d, train_side)
    basis_train = next(basis for basis in lattice_physics_probe.build_bases(train_d, train_side, seed) if basis.name == "toric_PJ_R2")
    basis_eval = next(basis for basis in lattice_physics_probe.build_bases(eval_d, train_side, seed) if basis.name == "toric_PJ_R2")
    return run_frontier(
        task="lattice_wave_kernel",
        basis=basis_train,
        train_matrix=basis_train.matrix,
        eval_matrix=basis_eval.matrix,
        train_target=train_target,
        eval_target=eval_target,
        ridge=ridge,
    )


def real_digits_task(device: torch.device, *, seed: int, ridge: float) -> list[dict[str, object]]:
    side = 8
    train, test = real_digits_probe.load_digit_tensors(device, seed=seed, train_count=1400)
    positions = real_digits_probe.make_positions(side, device)
    d = real_digits_probe.pairwise_d(positions).reshape(-1, 2)
    basis = next(basis for basis in real_digits_probe.build_bases(d, side, seed) if basis.name == "toric_PJ_R2")
    train_matrix = real_digits_probe.design_matrix(train, basis, n_positions=side * side)
    eval_matrix = real_digits_probe.design_matrix(test, basis, n_positions=side * side)
    return run_frontier(
        task="real_digits_masked_reconstruction",
        basis=basis,
        train_matrix=train_matrix,
        eval_matrix=eval_matrix,
        train_target=train.reshape(-1),
        eval_target=test.reshape(-1),
        ridge=ridge,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = lattice_task(device, seed=args.seed, ridge=args.ridge) + real_digits_task(device, seed=args.seed, ridge=args.ridge)

    csv_path = output_dir / "sparse_pruning_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = output_dir / "sparse_pruning_frontier.png"
    plot_frontier(rows, plot_path)
    summary = {
        "device": str(device),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "sparse_pruning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, rows, summary)
    return summary


def plot_frontier(rows: list[dict[str, object]], path: Path) -> None:
    tasks = sorted({str(row["task"]) for row in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(12, 4.6), squeeze=False)
    for ax, task in zip(axes.reshape(-1), tasks):
        values = sorted([row for row in rows if row["task"] == task], key=lambda item: float(item["num_features"]))
        ax.plot(
            [int(row["num_features"]) for row in values],
            [float(row["eval_r2"]) for row in values],
            marker="o",
        )
        for row in values:
            label = str(row["variant"]).replace("_top_", "\n")
            ax.text(int(row["num_features"]), float(row["eval_r2"]), label, fontsize=7, ha="center", va="bottom")
        ax.set_title(task)
        ax.set_xlabel("features kept")
        ax.set_ylabel("eval R2")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# V2-B Sparse Pruning Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        "",
        "Run command:",
        "",
        "```bash",
        "python scripts/run_v2_pruning.py --device cuda --output-dir results/v2_pruning",
        "```",
        "",
        "## Frontier",
        "",
        "| task | variant | features | groups | eval R2 | retained |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['task']} | {row['variant']} | {int(row['num_features'])} | {int(row['num_groups'])} | "
            + f"{float(row['eval_r2']):.4f} | {float(row['retained_eval_r2_frac']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Dense Toric PJ_R2 is the reference dictionary.",
            "- LOO pruning ranks groups by direct functional loss when removed after dense fitting.",
            "- Energy pruning is a cheaper coefficient/contribution proxy.",
            "- The frontier reports whether compact sub-dictionaries retain most of dense performance.",
            "",
            "Artifacts:",
            "",
            "- `sparse_pruning_results.csv`",
            "- `sparse_pruning_summary.json`",
            "- `sparse_pruning_frontier.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-B sparse pruning experiments.")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_pruning")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
