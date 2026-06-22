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

from toric_pj.diagnostics.basis_projection import default_device
from toric_pj.experiments import lattice_physics_probe, real_digits_probe, sparse_pruning


def parse_json_dict(value: str) -> dict[str, float]:
    raw = json.loads(value)
    return {str(key): float(item) for key, item in raw.items()}


def add_group_rows(
    rows: list[dict[str, object]],
    *,
    task: str,
    energy: dict[str, float],
    loo: dict[str, float],
) -> None:
    for component in sorted(energy):
        rows.append(
            {
                "task": task,
                "allocation_type": "branch",
                "component": component,
                "energy": float(energy[component]),
                "loo": float(loo.get(component, 0.0)),
            }
        )


def add_order_rows(
    rows: list[dict[str, object]],
    *,
    task: str,
    energy: dict[str, float],
    loo: dict[str, float],
) -> None:
    for order in sorted(energy, key=lambda item: int(item)):
        rows.append(
            {
                "task": task,
                "allocation_type": "order",
                "component": f"order{order}",
                "energy": float(energy[order]),
                "loo": float(loo.get(order, 0.0)),
            }
        )


def load_music_language_image_lattice(rows: list[dict[str, object]], root: Path) -> None:
    with (root / "results/stage5_music/music_toric_results.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["basis"] == "toric_music_lc":
                add_group_rows(
                    rows,
                    task="music",
                    energy=parse_json_dict(row["group_energy"]),
                    loo=parse_json_dict(row["group_loo"]),
                )
                break

    with (root / "results/stage5_language/language_hierarchical_results.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["basis"] == "hierarchical_lc":
                add_group_rows(
                    rows,
                    task="language",
                    energy=parse_json_dict(row["group_energy"]),
                    loo=parse_json_dict(row["group_loo"]),
                )
                break

    with (root / "results/stage5_image/image_grid_probe_results.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["basis"] == "toric_PJ_R2":
                add_order_rows(
                    rows,
                    task="image_grid",
                    energy=parse_json_dict(row["order_energy"]),
                    loo=parse_json_dict(row["order_loo"]),
                )
                break

    with (root / "results/stage5_lattice/lattice_physics_results.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["basis"] == "toric_PJ_R2":
                add_order_rows(
                    rows,
                    task="lattice_wave",
                    energy=parse_json_dict(row["order_energy"]),
                    loo=parse_json_dict(row["order_loo"]),
                )
                break


def real_digits_order_rows(rows: list[dict[str, object]], device: torch.device, seed: int, ridge: float) -> None:
    side = 8
    train, _ = real_digits_probe.load_digit_tensors(device, seed=seed, train_count=1400)
    positions = real_digits_probe.make_positions(side, device)
    d = real_digits_probe.pairwise_d(positions).reshape(-1, 2)
    basis = next(basis for basis in real_digits_probe.build_bases(d, side, seed) if basis.name == "toric_PJ_R2")
    matrix = real_digits_probe.design_matrix(train, basis, n_positions=side * side)
    loo, energy = sparse_pruning.group_diagnostics(matrix, train.reshape(-1), basis, ridge=ridge)
    order_energy: dict[str, float] = {"0": 0.0, "1": 0.0, "2": 0.0}
    order_loo: dict[str, float] = {"0": 0.0, "1": 0.0, "2": 0.0}
    for group, value in energy.items():
        if "_r0" in group or group == "const":
            order = "0"
        elif "_r1" in group:
            order = "1"
        elif "_r2" in group:
            order = "2"
        else:
            order = "0"
        order_energy[order] += float(value)
        order_loo[order] += float(loo.get(group, 0.0))
    add_order_rows(rows, task="real_digits", energy=order_energy, loo=order_loo)


def normalize_energy_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    totals: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (str(row["task"]), str(row["allocation_type"]))
        totals[key] = totals.get(key, 0.0) + abs(float(row["energy"]))
    for row in rows:
        key = (str(row["task"]), str(row["allocation_type"]))
        total = max(totals[key], 1e-12)
        out.append({**row, "energy_share": abs(float(row["energy"])) / total})
    return out


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.root)
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    load_music_language_image_lattice(rows, root)
    real_digits_order_rows(rows, device, seed=args.seed, ridge=args.ridge)
    rows = normalize_energy_rows(rows)

    csv_path = output_dir / "task_adaptive_allocation_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    branch_plot = output_dir / "task_branch_allocation.png"
    order_plot = output_dir / "task_order_allocation.png"
    plot_heatmap(rows, allocation_type="branch", path=branch_plot)
    plot_heatmap(rows, allocation_type="order", path=order_plot)
    summary = {
        "device": str(device),
        "csv": str(csv_path),
        "branch_plot": str(branch_plot),
        "order_plot": str(order_plot),
        "rows": rows,
    }
    summary_path = output_dir / "task_adaptive_allocation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, rows, summary)
    return summary


def plot_heatmap(rows: list[dict[str, object]], *, allocation_type: str, path: Path) -> None:
    values = [row for row in rows if row["allocation_type"] == allocation_type]
    tasks = sorted({str(row["task"]) for row in values})
    components = sorted({str(row["component"]) for row in values})
    matrix = np.zeros((len(tasks), len(components)))
    for row in values:
        i = tasks.index(str(row["task"]))
        j = components.index(str(row["component"]))
        matrix[i, j] = float(row["energy_share"])
    fig, ax = plt.subplots(figsize=(max(7, len(components) * 0.8), max(4, len(tasks) * 0.8)))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=max(1e-6, matrix.max()))
    ax.set_xticks(np.arange(len(components)), components, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(tasks)), tasks)
    ax.set_title(f"Task {allocation_type} allocation")
    for i in range(len(tasks)):
        for j in range(len(components)):
            if matrix[i, j] > 0:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# V2-D Task-Adaptive Allocation Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        "",
        "Run command:",
        "",
        "```bash",
        "python scripts/run_v2_allocation.py --device cuda --output-dir results/v2_allocation",
        "```",
        "",
        "## Allocation",
        "",
        "| task | type | component | energy share | LOO |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + f"{row['task']} | {row['allocation_type']} | {row['component']} | "
            + f"{float(row['energy_share']):.4f} | {float(row['loo']):.4g} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Music and language use the group diagnostics from the v1 domain probes.",
            "- Image, lattice, and real digits are summarized by jet order allocation.",
            "- Energy share is normalized within each task/type so tasks with different raw scales can be compared.",
            "- LOO remains the stronger functional signal when components are comparable within the same task.",
            "",
            "Artifacts:",
            "",
            "- `task_adaptive_allocation_results.csv`",
            "- `task_adaptive_allocation_summary.json`",
            "- `task_branch_allocation.png`",
            "- `task_order_allocation.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2-D task-adaptive allocation diagnostics.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v2_allocation")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
