from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HEAD_PARAM_RE = re.compile(r"layer=(?P<layer>\d+),head=(?P<head>\d+),r<=?(?P<radius>[-+0-9.eE]+)")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    bias_npz: Path
    interventions_csv: Path


DEFAULT_TASKS = [
    TaskSpec(
        name="mnist",
        bias_npz=Path(
            "results/v4_offset_holdout_classification_mnist_radial_causal_10k_start5000/"
            "bias_exports/"
            "mnist_classification_table_informed_toric_PJ_R0_top110_seed426_holdoutR4_steps10000_"
            "h1_b10_cl23em05_tail10_g2_cosine_s5000_r5000/"
            "bias_tables.npz"
        ),
        interventions_csv=Path("results/v4_component_intervention_eval_mnist_l0_subsets_final/component_interventions.csv"),
    ),
    TaskSpec(
        name="rotated-mnist",
        bias_npz=Path(
            "results/v5_rotated_mnist_full_top110_10k_start5000/"
            "bias_exports/"
            "rotated-mnist_classification_table_informed_toric_PJ_R0_top110_seed426_holdoutR4_steps10000_"
            "h1_b10_cl23em05_tail10_g2_cosine_s5000_r5000/"
            "bias_tables.npz"
        ),
        interventions_csv=Path("results/v5_component_eval_rotated_mnist_final_layer0_individual/component_interventions.csv"),
    ),
    TaskSpec(
        name="affine-mnist",
        bias_npz=Path(
            "results/v5_affine_mnist_full_top110_10k_start5000/"
            "bias_exports/"
            "affine-mnist_classification_table_informed_toric_PJ_R0_top110_seed426_holdoutR4_steps10000_"
            "h1_b10_cl23em05_tail10_g2_cosine_s5000_r5000/"
            "bias_tables.npz"
        ),
        interventions_csv=Path("results/v5_component_eval_affine_mnist_final_layer0_individual/component_interventions.csv"),
    ),
    TaskSpec(
        name="fashion-mnist",
        bias_npz=Path(
            "results/v5_fashion_mnist_full_top110_10k_start5000/"
            "bias_exports/"
            "fashion-mnist_classification_table_informed_toric_PJ_R0_top110_seed426_holdoutR4_steps10000_"
            "h1_b10_cl23em05_tail10_g2_cosine_s5000_r5000/"
            "bias_tables.npz"
        ),
        interventions_csv=Path("results/v5_component_eval_fashion_mnist_final_layer0_individual/component_interventions.csv"),
    ),
    TaskSpec(
        name="cifar10",
        bias_npz=Path(
            "results/v5_cifar10_full_top110_10k_start3000/"
            "bias_exports/"
            "cifar10_classification_table_informed_toric_PJ_R0_top110_seed426_holdoutR4_steps10000_"
            "h1_b10_cl23em05_tail10_g2_cosine_s3000_r5000/"
            "bias_tables.npz"
        ),
        interventions_csv=Path("results/v5_component_eval_cifar10_final_layer0_individual/component_interventions.csv"),
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_head_param(value: str) -> tuple[int, int, float] | None:
    match = HEAD_PARAM_RE.search(value)
    if not match:
        return None
    return int(match.group("layer")), int(match.group("head")), float(match.group("radius"))


def centered_vector(table: np.ndarray) -> np.ndarray:
    x = np.asarray(table, dtype=np.float64)
    return (x - x.mean()).reshape(-1)


def table_on_common_offsets(
    table: np.ndarray,
    dx_values: np.ndarray,
    dy_values: np.ndarray,
    common_dx: np.ndarray,
    common_dy: np.ndarray,
) -> np.ndarray:
    dx_lookup = {int(round(float(value))): idx for idx, value in enumerate(dx_values)}
    dy_lookup = {int(round(float(value))): idx for idx, value in enumerate(dy_values)}
    dx_idx = [dx_lookup[int(round(float(value)))] for value in common_dx]
    dy_idx = [dy_lookup[int(round(float(value)))] for value in common_dy]
    return np.asarray(table, dtype=np.float64)[np.ix_(dx_idx, dy_idx)]


def centered_common_vectors(
    table_a: np.ndarray,
    dx_a: np.ndarray,
    dy_a: np.ndarray,
    table_b: np.ndarray,
    dx_b: np.ndarray,
    dy_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    common_dx = np.intersect1d(np.round(dx_a).astype(int), np.round(dx_b).astype(int))
    common_dy = np.intersect1d(np.round(dy_a).astype(int), np.round(dy_b).astype(int))
    if common_dx.size == 0 or common_dy.size == 0:
        raise ValueError("tables do not share any relative offsets")
    a = table_on_common_offsets(table_a, dx_a, dy_a, common_dx, common_dy)
    b = table_on_common_offsets(table_b, dx_b, dy_b, common_dx, common_dy)
    return centered_vector(a), centered_vector(b)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def format_float(value: float, digits: int = 4) -> float:
    if not math.isfinite(value):
        return value
    return round(float(value), digits)


def load_intervention_metrics(path: Path, *, layer: int, radius: float) -> dict[str, object]:
    rows = read_csv(path)
    full_score = float(next(row["score"] for row in rows if row["eval_mode"] == "full"))
    zero_score = float(next(row["score"] for row in rows if row["eval_mode"] == "zero_bias"))
    heads: dict[int, dict[str, float]] = {}
    for row in rows:
        mode = row["eval_mode"]
        if mode not in {"head_radial_ablate", "head_radial_keep"}:
            continue
        parsed = parse_head_param(row["eval_param"])
        if parsed is None:
            continue
        row_layer, head, row_radius = parsed
        if row_layer != layer or abs(row_radius - radius) > 1e-6:
            continue
        item = heads.setdefault(head, {})
        score = float(row["score"])
        if mode == "head_radial_ablate":
            item["ablate_score"] = score
            item["ablate_drop_pt"] = (full_score - score) * 100.0
        else:
            item["keep_score"] = score
            item["keep_delta_pt"] = (score - full_score) * 100.0
            item["keep_gain_pt"] = (score - zero_score) * 100.0
    return {"full_score": full_score, "zero_score": zero_score, "heads": heads}


def load_bias_bundle(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "tables": np.asarray(data["tables"], dtype=np.float64),
        "dx_values": np.asarray(data["dx_values"], dtype=np.float64),
        "dy_values": np.asarray(data["dy_values"], dtype=np.float64),
    }


def radial_arrays(dx_values: np.ndarray, dy_values: np.ndarray) -> np.ndarray:
    gx, gy = np.meshgrid(dx_values, dy_values, indexing="ij")
    return np.sqrt(gx * gx + gy * gy)


def table_energy_metrics(table: np.ndarray, radii: list[float], radial: np.ndarray) -> dict[str, float]:
    centered = np.asarray(table, dtype=np.float64) - float(np.mean(table))
    energy = centered * centered
    total = float(np.sum(energy))
    out: dict[str, float] = {
        "bias_std": float(np.std(centered)),
        "bias_l2": float(np.sqrt(total)),
        "bias_max_abs_centered": float(np.max(np.abs(centered))) if centered.size else float("nan"),
    }
    if total <= 1e-12:
        for radius in radii:
            out[f"energy_frac_r_le_{radius:g}"] = float("nan")
        return out
    for radius in radii:
        out[f"energy_frac_r_le_{radius:g}"] = float(np.sum(energy[radial <= radius]) / total)
    return out


def radial_profile_rows(
    *,
    task: str,
    head: int,
    table: np.ndarray,
    radial: np.ndarray,
    max_radius: int,
) -> list[dict[str, object]]:
    centered = np.asarray(table, dtype=np.float64) - float(np.mean(table))
    energy = centered * centered
    total = float(np.sum(energy))
    rows: list[dict[str, object]] = []
    for radius in range(max_radius + 1):
        cumulative = float(np.sum(energy[radial <= radius]) / total) if total > 1e-12 else float("nan")
        shell = np.abs(radial - radius) < 0.5
        shell_rms = float(np.sqrt(np.mean(energy[shell]))) if np.any(shell) else float("nan")
        rows.append(
            {
                "task": task,
                "layer": 0,
                "head": head,
                "radius": radius,
                "cumulative_energy_frac": cumulative,
                "shell_rms": shell_rms,
            }
        )
    return rows


def top_heads(head_rows: list[dict[str, object]], *, task: str, top_k: int) -> list[int]:
    candidates = [row for row in head_rows if row["task"] == task and row["layer"] == 0]
    candidates.sort(key=lambda row: float(row.get("ablate_drop_pt", float("-inf"))), reverse=True)
    return [int(row["head"]) for row in candidates[:top_k]]


def normalized_heatmap(table: np.ndarray) -> np.ndarray:
    centered = np.asarray(table, dtype=np.float64) - float(np.mean(table))
    scale = float(np.max(np.abs(centered)))
    if scale <= 1e-12:
        return centered
    return centered / scale


def plot_head_importance(
    path: Path,
    *,
    tasks: list[str],
    n_heads: int,
    metric_by_task_head: dict[tuple[str, int], dict[str, float]],
) -> None:
    drop = np.full((len(tasks), n_heads), np.nan, dtype=np.float64)
    keep_gain = np.full_like(drop, np.nan)
    for row_idx, task in enumerate(tasks):
        for head in range(n_heads):
            metrics = metric_by_task_head.get((task, head), {})
            drop[row_idx, head] = metrics.get("ablate_drop_pt", np.nan)
            keep_gain[row_idx, head] = metrics.get("keep_gain_pt", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
    vmax = max(1e-6, float(np.nanmax(np.abs(drop))))
    im = axes[0].imshow(drop, cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
    axes[0].set_title("Ablate drop at L0 r<=4 (pt)")
    axes[0].set_xticks(range(n_heads), [f"H{i}" for i in range(n_heads)])
    axes[0].set_yticks(range(len(tasks)), tasks)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    vmax_keep = max(1e-6, float(np.nanmax(np.abs(keep_gain))))
    im = axes[1].imshow(keep_gain, cmap="viridis", vmin=0.0, vmax=vmax_keep, aspect="auto")
    axes[1].set_title("Keep gain over zero at L0 r<=4 (pt)")
    axes[1].set_xticks(range(n_heads), [f"H{i}" for i in range(n_heads)])
    axes[1].set_yticks(range(len(tasks)), tasks)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_top_head_heatmaps(
    path: Path,
    *,
    tasks: list[str],
    top_by_task: dict[str, list[int]],
    tables_by_task: dict[str, np.ndarray],
    metric_by_task_head: dict[tuple[str, int], dict[str, float]],
    top_k: int,
) -> None:
    fig, axes = plt.subplots(len(tasks), top_k, figsize=(3.1 * top_k, 2.75 * len(tasks)), constrained_layout=True)
    if len(tasks) == 1:
        axes = np.expand_dims(axes, axis=0)
    if top_k == 1:
        axes = np.expand_dims(axes, axis=1)
    for row_idx, task in enumerate(tasks):
        for col_idx in range(top_k):
            ax = axes[row_idx, col_idx]
            heads = top_by_task[task]
            if col_idx >= len(heads):
                ax.axis("off")
                continue
            head = heads[col_idx]
            table = normalized_heatmap(tables_by_task[task][0, head])
            im = ax.imshow(table, cmap="coolwarm", vmin=-1.0, vmax=1.0, origin="lower")
            metrics = metric_by_task_head[(task, head)]
            ax.set_title(
                f"{task} H{head}\n"
                f"drop {metrics.get('ablate_drop_pt', float('nan')):.2f}pt, "
                f"keep {metrics.get('keep_score', float('nan')):.4f}",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == top_k - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_radial_profiles(
    path: Path,
    *,
    tasks: list[str],
    top_by_task: dict[str, list[int]],
    radial_rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.5, 2.25 * len(tasks)), constrained_layout=True)
    if len(tasks) == 1:
        axes = [axes]
    by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in radial_rows:
        by_key.setdefault((str(row["task"]), int(row["head"])), []).append(row)
    for ax, task in zip(axes, tasks):
        for head in top_by_task[task]:
            rows = sorted(by_key[(task, head)], key=lambda row: int(row["radius"]))
            xs = [int(row["radius"]) for row in rows]
            ys = [float(row["cumulative_energy_frac"]) for row in rows]
            ax.plot(xs, ys, marker="o", label=f"H{head}")
        ax.set_title(task)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("radius")
        ax.set_ylabel("cumulative energy")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", ncols=min(3, len(top_by_task[task])), fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mnist_alignment(
    path: Path,
    *,
    tasks: list[str],
    n_heads: int,
    metric_by_task_head: dict[tuple[str, int], dict[str, float]],
) -> None:
    values = np.full((len(tasks), n_heads), np.nan, dtype=np.float64)
    for row_idx, task in enumerate(tasks):
        for head in range(n_heads):
            values[row_idx, head] = metric_by_task_head.get((task, head), {}).get("best_abs_cos_to_mnist_top3", np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    im = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("Best abs cosine to MNIST top heads (H3/H6/H1)")
    ax.set_xticks(range(n_heads), [f"H{i}" for i in range(n_heads)])
    ax.set_yticks(range(len(tasks)), tasks)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    *,
    tasks: list[str],
    top_by_task: dict[str, list[int]],
    metric_by_task_head: dict[tuple[str, int], dict[str, float]],
    output_dir: Path,
) -> None:
    lines = [
        "# V5 Cross-Task Head Alignment",
        "",
        "Inputs are classification top110 final bias exports and final L0 r<=4 component interventions.",
        "",
        "## Top Heads",
        "",
        "| task | top heads by ablate drop | best keep score among top heads | max ablate drop pt |",
        "|---|---|---:|---:|",
    ]
    for task in tasks:
        heads = top_by_task[task]
        keep_scores = [metric_by_task_head[(task, head)].get("keep_score", float("nan")) for head in heads]
        drops = [metric_by_task_head[(task, head)].get("ablate_drop_pt", float("nan")) for head in heads]
        lines.append(
            "| "
            + task
            + " | "
            + ", ".join(f"H{head}" for head in heads)
            + f" | {max(keep_scores):.4f} | {max(drops):.2f} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- `{(output_dir / 'head_alignment_summary.csv').as_posix()}`",
        f"- `{(output_dir / 'radial_profiles.csv').as_posix()}`",
        f"- `{(output_dir / 'layer0_head_importance.png').as_posix()}`",
        f"- `{(output_dir / 'layer0_top_head_heatmaps.png').as_posix()}`",
        f"- `{(output_dir / 'layer0_top_head_radial_profiles.png').as_posix()}`",
        f"- `{(output_dir / 'mnist_top_head_shape_alignment.png').as_posix()}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = DEFAULT_TASKS
    task_names = [task.name for task in tasks]

    tables_by_task: dict[str, np.ndarray] = {}
    dx_by_task: dict[str, np.ndarray] = {}
    dy_by_task: dict[str, np.ndarray] = {}
    intervention_by_task: dict[str, dict[str, object]] = {}
    for task in tasks:
        if not task.bias_npz.exists():
            raise FileNotFoundError(task.bias_npz)
        if not task.interventions_csv.exists():
            raise FileNotFoundError(task.interventions_csv)
        bundle = load_bias_bundle(task.bias_npz)
        tables_by_task[task.name] = bundle["tables"]
        dx_by_task[task.name] = bundle["dx_values"]
        dy_by_task[task.name] = bundle["dy_values"]
        intervention_by_task[task.name] = load_intervention_metrics(task.interventions_csv, layer=args.layer, radius=args.radius)

    n_heads = int(next(iter(tables_by_task.values())).shape[1])
    mnist_refs = {
        head: tables_by_task["mnist"][args.layer, head]
        for head in [int(item) for item in args.mnist_reference_heads.split(",") if item.strip()]
    }

    summary_rows: list[dict[str, object]] = []
    radial_rows: list[dict[str, object]] = []
    metric_by_task_head: dict[tuple[str, int], dict[str, float]] = {}
    for task in task_names:
        tables = tables_by_task[task]
        radial = radial_arrays(dx_by_task[task], dy_by_task[task])
        interventions = intervention_by_task[task]
        head_metrics = interventions["heads"]
        full_score = float(interventions["full_score"])
        zero_score = float(interventions["zero_score"])
        for head in range(n_heads):
            table = tables[args.layer, head]
            metrics = dict(head_metrics.get(head, {}))
            energy = table_energy_metrics(table, [2.0, 4.0, 6.0], radial)
            cos_values = {}
            for ref_head, ref_table in mnist_refs.items():
                vec, ref_vec = centered_common_vectors(
                    table,
                    dx_by_task[task],
                    dy_by_task[task],
                    ref_table,
                    dx_by_task["mnist"],
                    dy_by_task["mnist"],
                )
                cos_values[f"cos_to_mnist_H{ref_head}"] = cosine(vec, ref_vec)
            best_ref_head = None
            best_abs_cos = float("nan")
            if cos_values:
                best_ref_key, best_ref_value = max(cos_values.items(), key=lambda item: abs(item[1]) if math.isfinite(item[1]) else -1.0)
                best_ref_head = best_ref_key.removeprefix("cos_to_mnist_H")
                best_abs_cos = abs(best_ref_value)
            row = {
                "task": task,
                "layer": args.layer,
                "head": head,
                "full_score": full_score,
                "zero_score": zero_score,
                "bias_gain_pt": (full_score - zero_score) * 100.0,
                "ablate_score": metrics.get("ablate_score", float("nan")),
                "ablate_drop_pt": metrics.get("ablate_drop_pt", float("nan")),
                "keep_score": metrics.get("keep_score", float("nan")),
                "keep_delta_pt": metrics.get("keep_delta_pt", float("nan")),
                "keep_gain_pt": metrics.get("keep_gain_pt", float("nan")),
                "best_abs_cos_to_mnist_top3": best_abs_cos,
                "best_abs_cos_mnist_ref_head": best_ref_head,
                **cos_values,
                **energy,
            }
            metric_by_task_head[(task, head)] = {key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
            metric_by_task_head[(task, head)]["keep_score"] = float(row["keep_score"])
            metric_by_task_head[(task, head)]["best_abs_cos_to_mnist_top3"] = float(best_abs_cos)
            summary_rows.append({key: format_float(value) if isinstance(value, float) else value for key, value in row.items()})

    top_by_task = {task: top_heads(summary_rows, task=task, top_k=args.top_k) for task in task_names}
    for task in task_names:
        radial = radial_arrays(dx_by_task[task], dy_by_task[task])
        for head in top_by_task[task]:
            radial_rows.extend(
                radial_profile_rows(
                    task=task,
                    head=head,
                    table=tables_by_task[task][args.layer, head],
                    radial=radial,
                    max_radius=args.max_radius,
                )
            )

    write_csv(output_dir / "head_alignment_summary.csv", summary_rows)
    write_csv(output_dir / "radial_profiles.csv", [{k: format_float(v) if isinstance(v, float) else v for k, v in row.items()} for row in radial_rows])
    plot_head_importance(output_dir / "layer0_head_importance.png", tasks=task_names, n_heads=n_heads, metric_by_task_head=metric_by_task_head)
    plot_top_head_heatmaps(
        output_dir / "layer0_top_head_heatmaps.png",
        tasks=task_names,
        top_by_task=top_by_task,
        tables_by_task=tables_by_task,
        metric_by_task_head=metric_by_task_head,
        top_k=args.top_k,
    )
    plot_radial_profiles(output_dir / "layer0_top_head_radial_profiles.png", tasks=task_names, top_by_task=top_by_task, radial_rows=radial_rows)
    plot_mnist_alignment(output_dir / "mnist_top_head_shape_alignment.png", tasks=task_names, n_heads=n_heads, metric_by_task_head=metric_by_task_head)
    write_report(output_dir / "REPORT.md", tasks=task_names, top_by_task=top_by_task, metric_by_task_head=metric_by_task_head, output_dir=output_dir)

    summary = {
        "output_dir": str(output_dir),
        "tasks": task_names,
        "top_by_task": top_by_task,
        "summary_csv": str(output_dir / "head_alignment_summary.csv"),
        "radial_profiles_csv": str(output_dir / "radial_profiles.csv"),
        "figures": [
            str(output_dir / "layer0_head_importance.png"),
            str(output_dir / "layer0_top_head_heatmaps.png"),
            str(output_dir / "layer0_top_head_radial_profiles.png"),
            str(output_dir / "mnist_top_head_shape_alignment.png"),
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/v5_cross_task_head_alignment")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-radius", type=int, default=6)
    parser.add_argument("--mnist-reference-heads", default="3,6,1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
