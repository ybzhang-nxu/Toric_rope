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


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def rounded(value: float, digits: int = 4) -> float:
    if not math.isfinite(value):
        return value
    return round(value, digits)


def first_row(path: str | Path) -> dict[str, str]:
    rows = read_csv(Path(path))
    if not rows:
        raise ValueError(f"empty csv: {path}")
    return rows[0]


def rows_by_mode(path: str | Path) -> dict[str, dict[str, str]]:
    rows = read_csv(Path(path))
    return {row["train_component_mode"]: row for row in rows}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def figure_a_alignment(output_dir: Path) -> list[dict[str, object]]:
    rows = read_csv(Path("results/v5_train_time_function_alignment/train_time_function_alignment.csv"))
    selected_roles = {"primary_top2", "primary_top3", "primary_top3_alt", "primary_top5"}
    rows = [row for row in rows if row["role"] in selected_roles]
    data: list[dict[str, object]] = []
    for row in rows:
        data.append(
            {
                "label": row["label"],
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "role": row["role"],
                "heads": row["heads"],
                "keep_gap_pt": rounded(float(row["keep_gap_pt"]), 2),
                "ablate_drop_pt": rounded(float(row["ablate_drop_pt"]), 2),
            }
        )
    write_csv(output_dir / "figure_a_head_function_alignment.csv", data)

    labels = [str(row["label"]).replace("MNIST ", "M").replace("Rotated ", "R").replace("Affine ", "A") for row in data]
    keep_gap = np.asarray([float(row["keep_gap_pt"]) for row in data], dtype=float)
    ablate_drop = np.asarray([float(row["ablate_drop_pt"]) for row in data], dtype=float)
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(10.6, 5.4), constrained_layout=True)
    ax.barh(y - 0.18, keep_gap, height=0.34, color="#4c78a8", label="full - keep (pt)")
    ax.barh(y + 0.18, ablate_drop, height=0.34, color="#f58518", label="full - ablate (pt)")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axvline(1.0, color="#777", linewidth=0.8, linestyle="--")
    ax.axvline(2.0, color="#555", linewidth=0.8, linestyle=":")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("score points")
    ax.set_title("A. Train-time function alignment")
    ax.legend(loc="lower right")
    fig.savefig(output_dir / "figure_a_head_function_alignment.png", dpi=200)
    plt.close(fig)
    return data


def chain_row(chain: str, stage: str, row: dict[str, str], *, source: str) -> dict[str, object]:
    return {
        "chain": chain,
        "stage": stage,
        "final": rounded(f(row, "final_score")),
        "visible": rounded(f(row, "final_visible_only_score")),
        "zero": rounded(f(row, "final_zero_bias_score")),
        "source": source,
    }


def figure_b_digit_chains(output_dir: Path) -> list[dict[str, object]]:
    specs = [
        (
            "Rotated cascade",
            [
                ("baseline", first_row("results/v5_rotated_mnist_full_top110_10k_start5000/offset_holdout_results.csv"), "full"),
                (
                    "L0 top keep",
                    rows_by_mode("results/v5_rotated_mnist_train_component_l0h347_10k_start5000/offset_holdout_results.csv")["keep"],
                    "L0H3/H4/H7 keep",
                ),
                (
                    "L0 all ablate",
                    first_row("results/v5_rotated_mnist_train_component_l0all_10k_start5000/offset_holdout_results.csv"),
                    "L0 all ablate",
                ),
                (
                    "L0+comp ablate",
                    first_row("results/v5_rotated_mnist_train_component_cond_l0all_l1h6_l2h3_10k_start5000/offset_holdout_results.csv"),
                    "L0 all + L1H6/L2H3 ablate",
                ),
            ],
        ),
        (
            "Affine pool",
            [
                ("baseline", first_row("results/v5_affine_mnist_full_top110_10k_start5000/offset_holdout_results.csv"), "full"),
                (
                    "L0 top keep",
                    rows_by_mode("results/v5_affine_mnist_train_component_l0h31476_10k_start5000/offset_holdout_results.csv")["keep"],
                    "L0 top5 keep",
                ),
                (
                    "L0 all ablate",
                    first_row("results/v5_affine_mnist_train_component_l0all_10k_start5000/offset_holdout_results.csv"),
                    "L0 all ablate",
                ),
                (
                    "L0+L1H2/H4",
                    first_row("results/v5_affine_mnist_train_component_cond_l0all_l1h2_l1h4_10k_start5000/offset_holdout_results.csv"),
                    "L0 all + L1H2/H4 ablate",
                ),
                (
                    "L0+L1H2/4/5/6",
                    first_row("results/v5_affine_mnist_train_component_cond_l0all_l1h2456_10k_start5000/offset_holdout_results.csv"),
                    "L0 all + L1H2/H4/H5/H6 ablate",
                ),
                (
                    "L0+L1 all",
                    first_row("results/v5_affine_mnist_train_component_l0all_l1all_10k_start5000/offset_holdout_results.csv"),
                    "L0 all + L1 all ablate",
                ),
            ],
        ),
    ]
    data: list[dict[str, object]] = []
    for chain, items in specs:
        for stage, row, source in items:
            data.append(chain_row(chain, stage, row, source=source))
    write_csv(output_dir / "figure_b_digit_compensation_chains.csv", data)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True, constrained_layout=True)
    for ax, (chain, _items) in zip(axes, specs):
        subset = [row for row in data if row["chain"] == chain]
        xs = np.arange(len(subset))
        ax.plot(xs, [float(row["final"]) for row in subset], marker="o", linewidth=2.0, label="final")
        ax.plot(xs, [float(row["zero"]) for row in subset], marker="s", linewidth=1.6, label="zero-bias")
        ax.set_xticks(xs, [str(row["stage"]) for row in subset], rotation=22, ha="right")
        ax.set_title(chain)
        ax.set_ylim(0.55, 1.0)
        ax.set_ylabel("accuracy" if ax is axes[0] else "")
        ax.legend(loc="lower left")
    fig.suptitle("B. Digit-family compensation chains")
    fig.savefig(output_dir / "figure_b_digit_compensation_chains.png", dpi=200)
    plt.close(fig)
    return data


def figure_c_cifar_reconstruction(output_dir: Path) -> list[dict[str, object]]:
    specs = [
        ("baseline", "results/v5_cifar10_reconstruction_top110_10k_start3000/offset_holdout_results.csv"),
        ("L1/L2 all", "results/v5_cifar10_reconstruction_train_component_l1all_l2all_10k_start3000/offset_holdout_results.csv"),
        ("L0-L3 all", "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_10k_start3000/offset_holdout_results.csv"),
        ("L0-L4 all", "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_10k_start3000/offset_holdout_results.csv"),
        ("L0-L5 all s426", "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_l5all_10k_start3000/offset_holdout_results.csv"),
        ("L0-L5 all s526", "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_l5all_seed526_10k_start3000/offset_holdout_results.csv"),
        ("L0-L5 all s626", "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_l5all_seed626_10k_start3000/offset_holdout_results.csv"),
    ]
    data: list[dict[str, object]] = []
    for stage, path in specs:
        data.append(chain_row("CIFAR10 reconstruction", stage, first_row(path), source=path))
    write_csv(output_dir / "figure_c_cifar_reconstruction_pool_exhaustion.csv", data)

    xs = np.arange(len(data))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
    ax.bar(xs - width, [float(row["final"]) for row in data], width=width, label="final", color="#4c78a8")
    ax.bar(xs, [float(row["visible"]) for row in data], width=width, label="visible/local", color="#54a24b")
    ax.bar(xs + width, [float(row["zero"]) for row in data], width=width, label="zero-bias", color="#bab0ac")
    ax.set_xticks(xs, [str(row["stage"]) for row in data], rotation=20, ha="right")
    ax.set_ylabel("reconstruction score")
    ax.set_ylim(0.15, 0.9)
    ax.set_title("C. CIFAR10 reconstruction local-pool exhaustion")
    ax.legend(loc="lower left")
    fig.savefig(output_dir / "figure_c_cifar_reconstruction_pool_exhaustion.png", dpi=200)
    plt.close(fig)
    return data


def figure_d_task_split(output_dir: Path) -> list[dict[str, object]]:
    specs = [
        ("MNIST cls", "digit cls", "results/v4_offset_holdout_classification_mnist_radial_causal_10k_start5000/offset_holdout_results.csv"),
        ("Rotated cls", "digit cls", "results/v5_rotated_mnist_full_top110_10k_start5000/offset_holdout_results.csv"),
        ("Affine cls", "digit cls", "results/v5_affine_mnist_full_top110_10k_start5000/offset_holdout_results.csv"),
        ("Fashion cls", "non-digit cls", "results/v5_fashion_mnist_full_top110_10k_start5000/offset_holdout_results.csv"),
        ("CIFAR cls", "natural cls", "results/v5_cifar10_full_top110_10k_start3000/offset_holdout_results.csv"),
        ("CIFAR recon", "natural recon", "results/v5_cifar10_reconstruction_top110_10k_start3000/offset_holdout_results.csv"),
    ]
    data: list[dict[str, object]] = []
    for label, group, path in specs:
        row = first_row(path)
        final = f(row, "final_score")
        zero = f(row, "final_zero_bias_score")
        data.append(
            {
                "label": label,
                "group": group,
                "final": rounded(final),
                "zero": rounded(zero),
                "bias_gain": rounded(final - zero),
                "source": path,
            }
        )
    write_csv(output_dir / "figure_d_task_split.csv", data)

    colors = {
        "digit cls": "#4c78a8",
        "non-digit cls": "#b279a2",
        "natural cls": "#e45756",
        "natural recon": "#54a24b",
    }
    xs = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9.5, 4.5), constrained_layout=True)
    ax.bar(xs, [float(row["bias_gain"]) for row in data], color=[colors[str(row["group"])] for row in data])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xs, [str(row["label"]) for row in data], rotation=25, ha="right")
    ax.set_ylabel("final - zero-bias")
    ax.set_title("D. Task split: where positional bias matters")
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markersize=8, label=group)
        for group, color in colors.items()
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.savefig(output_dir / "figure_d_task_split.png", dpi=200)
    plt.close(fig)
    return data


def figure_e_no_local_best_final_radial(output_dir: Path) -> list[dict[str, object]]:
    specs = [
        (
            "seed426",
            "final",
            "results/v5_component_eval_cifar10_reconstruction_train_ablate_l0all_l1all_l2all_l3all_l4all_l5all_all_layers/component_interventions.csv",
        ),
        (
            "seed426",
            "best",
            "results/v5_component_eval_cifar10_reconstruction_train_ablate_l0all_l1all_l2all_l3all_l4all_l5all_seed426_best_radial/component_interventions.csv",
        ),
        (
            "seed526",
            "final",
            "results/v5_component_eval_cifar10_reconstruction_train_ablate_l0all_l1all_l2all_l3all_l4all_l5all_seed526_all_layers/component_interventions.csv",
        ),
        (
            "seed526",
            "best",
            "results/v5_component_eval_cifar10_reconstruction_train_ablate_l0all_l1all_l2all_l3all_l4all_l5all_seed526_best_radial/component_interventions.csv",
        ),
        (
            "seed626",
            "final",
            "results/v5_component_eval_cifar10_reconstruction_train_ablate_l0all_l1all_l2all_l3all_l4all_l5all_seed626_all_layers/component_interventions.csv",
        ),
        (
            "seed626",
            "best",
            "results/v5_component_eval_cifar10_reconstruction_train_ablate_l0all_l1all_l2all_l3all_l4all_l5all_seed626_best_radial/component_interventions.csv",
        ),
    ]
    metric_order = ["zero", "full", "r<=2", "r<=4", "r<=6"]
    data: list[dict[str, object]] = []
    for seed, state, path in specs:
        for row in read_csv(Path(path)):
            mode = row["eval_mode"]
            if mode == "zero_bias":
                metric = "zero"
            elif mode == "full":
                metric = "full"
            elif mode == "radial_truncate" and row["eval_param"] in {"r<=2", "r<=4", "r<=6"}:
                metric = row["eval_param"]
            else:
                continue
            data.append(
                {
                    "seed": seed,
                    "state": state,
                    "metric": metric,
                    "score": rounded(f(row, "score")),
                    "delta_vs_full": rounded(f(row, "delta_vs_full")),
                    "gain_vs_zero": rounded(f(row, "gain_vs_zero")),
                    "source": path,
                }
            )
    write_csv(output_dir / "figure_e_no_local_best_final_radial.csv", data)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), sharey=True, constrained_layout=True)
    state_styles = {
        "final": {"color": "#4c78a8", "marker": "o"},
        "best": {"color": "#f58518", "marker": "s"},
    }
    for ax, seed in zip(axes, ["seed426", "seed526", "seed626"]):
        for state, style in state_styles.items():
            values = []
            for metric in metric_order:
                match = next(
                    row
                    for row in data
                    if row["seed"] == seed and row["state"] == state and row["metric"] == metric
                )
                values.append(float(match["score"]))
            ax.plot(
                np.arange(len(metric_order)),
                values,
                linewidth=2.0,
                label=state,
                **style,
            )
        ax.set_xticks(np.arange(len(metric_order)), metric_order)
        ax.set_title(seed)
        ax.set_ylim(0.2, 0.72)
        ax.set_ylabel("reconstruction score" if ax is axes[0] else "")
        ax.legend(loc="lower right")
    fig.suptitle("E. No-local-pool best/final radial comparison")
    fig.savefig(output_dir / "figure_e_no_local_best_final_radial.png", dpi=200)
    plt.close(fig)
    return data


def figure_f_no_local_three_seed_distribution(output_dir: Path) -> list[dict[str, object]]:
    specs = [
        (
            "seed426",
            "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_l5all_10k_start3000/offset_holdout_results.csv",
            "results/v5_component_eval_cifar10_reconstruction_train_ablate_l0all_l1all_l2all_l3all_l4all_l5all_all_layers/component_interventions.csv",
        ),
        (
            "seed526",
            "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_l5all_seed526_10k_start3000/offset_holdout_results.csv",
            "results/v5_component_eval_cifar10_reconstruction_train_ablate_l0all_l1all_l2all_l3all_l4all_l5all_seed526_all_layers/component_interventions.csv",
        ),
        (
            "seed626",
            "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_l5all_seed626_10k_start3000/offset_holdout_results.csv",
            "results/v5_component_eval_cifar10_reconstruction_train_ablate_l0all_l1all_l2all_l3all_l4all_l5all_seed626_all_layers/component_interventions.csv",
        ),
    ]
    metric_order = ["zero", "full", "r<=2", "r<=4", "r<=6", "visible"]
    values_by_metric: dict[str, list[float]] = {metric: [] for metric in metric_order}
    individual_rows: list[dict[str, object]] = []
    for seed, holdout_path, component_path in specs:
        holdout_row = first_row(holdout_path)
        component_rows = read_csv(Path(component_path))
        component_lookup = {
            ("zero", ""): next(row for row in component_rows if row["eval_mode"] == "zero_bias"),
            ("full", ""): next(row for row in component_rows if row["eval_mode"] == "full"),
            ("r<=2", ""): next(
                row
                for row in component_rows
                if row["eval_mode"] == "radial_truncate" and row["eval_param"] == "r<=2"
            ),
            ("r<=4", ""): next(
                row
                for row in component_rows
                if row["eval_mode"] == "radial_truncate" and row["eval_param"] == "r<=4"
            ),
            ("r<=6", ""): next(
                row
                for row in component_rows
                if row["eval_mode"] == "radial_truncate" and row["eval_param"] == "r<=6"
            ),
        }
        metric_values = {
            "zero": f(component_lookup[("zero", "")], "score"),
            "full": f(component_lookup[("full", "")], "score"),
            "r<=2": f(component_lookup[("r<=2", "")], "score"),
            "r<=4": f(component_lookup[("r<=4", "")], "score"),
            "r<=6": f(component_lookup[("r<=6", "")], "score"),
            "visible": f(holdout_row, "final_visible_only_score"),
        }
        for metric in metric_order:
            score = float(metric_values[metric])
            values_by_metric[metric].append(score)
            individual_rows.append(
                {
                    "row_type": "individual",
                    "metric": metric,
                    "seed": seed,
                    "score": rounded(score),
                    "mean": "",
                    "sd": "",
                    "source": holdout_path if metric == "visible" else component_path,
                }
            )

    aggregate_rows: list[dict[str, object]] = []
    for metric in metric_order:
        values = np.asarray(values_by_metric[metric], dtype=float)
        aggregate_rows.append(
            {
                "row_type": "aggregate",
                "metric": metric,
                "seed": "",
                "score": "",
                "mean": rounded(float(np.mean(values))),
                "sd": rounded(float(np.std(values, ddof=1))),
                "source": "",
            }
        )
    data = individual_rows + aggregate_rows
    write_csv(output_dir / "figure_f_no_local_three_seed_distribution.csv", data)

    means = np.asarray([float(row["mean"]) for row in aggregate_rows], dtype=float)
    sds = np.asarray([float(row["sd"]) for row in aggregate_rows], dtype=float)
    xs = np.arange(len(metric_order))
    colors = ["#bab0ac", "#4c78a8", "#72b7b2", "#54a24b", "#59a14f", "#f58518"]
    fig, ax = plt.subplots(figsize=(9.8, 4.8), constrained_layout=True)
    ax.bar(xs, means, yerr=sds, capsize=4, color=colors, alpha=0.78, label="mean +/- sd")
    seed_offsets = {"seed426": -0.12, "seed526": 0.0, "seed626": 0.12}
    seed_markers = {"seed426": "o", "seed526": "s", "seed626": "^"}
    for seed, offset in seed_offsets.items():
        scores = [
            float(
                next(row for row in individual_rows if row["seed"] == seed and row["metric"] == metric)["score"]
            )
            for metric in metric_order
        ]
        ax.scatter(
            xs + offset,
            scores,
            s=34,
            marker=seed_markers[seed],
            edgecolor="black",
            linewidth=0.45,
            label=seed,
            zorder=3,
        )
    baseline = first_row("results/v5_cifar10_reconstruction_top110_10k_start3000/offset_holdout_results.csv")
    baseline_full = f(baseline, "final_score")
    baseline_zero = f(baseline, "final_zero_bias_score")
    ax.axhline(baseline_full, color="#333333", linewidth=1.0, linestyle="--", label="baseline full")
    ax.axhline(baseline_zero, color="#777777", linewidth=1.0, linestyle=":", label="baseline zero")
    ax.set_xticks(xs, metric_order)
    ax.set_ylabel("reconstruction score")
    ax.set_ylim(0.18, 0.9)
    ax.set_title("F. No-local-pool three-seed distribution")
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    fig.savefig(output_dir / "figure_f_no_local_three_seed_distribution.png", dpi=200)
    plt.close(fig)
    return data


def single_bias_npz(root: str | Path) -> Path:
    matches = sorted(Path(root).glob("bias_exports/*/bias_tables.npz"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one bias_tables.npz under {root}, found {len(matches)}")
    return matches[0]


def load_bias_tables(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    return (
        np.asarray(data["tables"], dtype=np.float64),
        np.asarray(data["dx_values"], dtype=np.float64),
        np.asarray(data["dy_values"], dtype=np.float64),
    )


def centered_rms_heatmap(tables: np.ndarray) -> np.ndarray:
    centered = tables - np.mean(tables, axis=(-2, -1), keepdims=True)
    return np.sqrt(np.mean(np.square(centered), axis=(0, 1)))


def figure_g_cifar_bias_shape(output_dir: Path) -> list[dict[str, object]]:
    specs = [
        ("baseline", "results/v5_cifar10_reconstruction_top110_10k_start3000"),
        ("no-local s426", "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_l5all_10k_start3000"),
        ("no-local s526", "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_l5all_seed526_10k_start3000"),
        ("no-local s626", "results/v5_cifar10_reconstruction_train_component_l0all_l1all_l2all_l3all_l4all_l5all_seed626_10k_start3000"),
    ]
    heatmaps: dict[str, np.ndarray] = {}
    radial_by_run: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for label, root in specs:
        npz_path = single_bias_npz(root)
        tables, dx_values, dy_values = load_bias_tables(npz_path)
        heatmap = centered_rms_heatmap(tables)
        gx, gy = np.meshgrid(dx_values, dy_values, indexing="ij")
        radial = np.sqrt(np.square(gx) + np.square(gy))
        heatmaps[label] = heatmap
        radial_by_run[label] = radial
        rows.append(
            {
                "row_type": "summary",
                "run": label,
                "radius": "",
                "shell_rms": "",
                "cumulative_energy_frac": "",
                "center_rms": rounded(float(heatmap[radial == 0][0])),
                "global_rms": rounded(float(np.sqrt(np.mean(np.square(heatmap))))),
                "max_rms": rounded(float(np.max(heatmap))),
                "source": str(npz_path),
            }
        )
        total_energy = float(np.sum(np.square(heatmap)))
        for radius in range(11):
            shell = np.abs(radial - radius) < 0.5
            shell_rms = float(np.sqrt(np.mean(np.square(heatmap[shell])))) if np.any(shell) else float("nan")
            cumulative = float(np.sum(np.square(heatmap[radial <= radius])) / total_energy) if total_energy > 1e-12 else float("nan")
            rows.append(
                {
                    "row_type": "profile",
                    "run": label,
                    "radius": radius,
                    "shell_rms": rounded(shell_rms),
                    "cumulative_energy_frac": rounded(cumulative),
                    "center_rms": "",
                    "global_rms": "",
                    "max_rms": "",
                    "source": str(npz_path),
                }
            )

    no_local_labels = [label for label, _root in specs if label.startswith("no-local")]
    no_local_center = float(np.mean([float(next(row for row in rows if row["row_type"] == "summary" and row["run"] == label)["center_rms"]) for label in no_local_labels]))
    no_local_r6 = float(np.mean([float(next(row for row in rows if row["row_type"] == "profile" and row["run"] == label and row["radius"] == 6)["shell_rms"]) for label in no_local_labels]))
    no_local_r7 = float(np.mean([float(next(row for row in rows if row["row_type"] == "profile" and row["run"] == label and row["radius"] == 7)["shell_rms"]) for label in no_local_labels]))
    rows.append(
        {
            "row_type": "aggregate",
            "run": "no-local mean",
            "radius": "",
            "shell_rms": "",
            "cumulative_energy_frac": "",
            "center_rms": rounded(no_local_center),
            "global_rms": "",
            "max_rms": "",
            "r6_shell_rms": rounded(no_local_r6),
            "r7_shell_rms": rounded(no_local_r7),
            "source": "",
        }
    )
    write_csv(output_dir / "figure_g_cifar_bias_shape.csv", rows)

    vmax = max(float(np.max(heatmap)) for heatmap in heatmaps.values())
    fig = plt.figure(figsize=(13.5, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05])
    image_axes = [fig.add_subplot(grid[0, idx]) for idx in range(4)]
    profile_ax = fig.add_subplot(grid[1, :])
    last_image = None
    for ax, (label, _root) in zip(image_axes, specs):
        heatmap = heatmaps[label]
        last_image = ax.imshow(
            heatmap,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            extent=(-7, 7, -7, 7),
        )
        summary = next(row for row in rows if row["row_type"] == "summary" and row["run"] == label)
        ax.set_title(f"{label}\ncenter {float(summary['center_rms']):.4f}")
        ax.set_xlabel("dy")
        ax.set_ylabel("dx")
        ax.set_xticks([-6, 0, 6])
        ax.set_yticks([-6, 0, 6])
    if last_image is not None:
        fig.colorbar(last_image, ax=image_axes, fraction=0.028, pad=0.01, label="centered RMS")

    colors = {
        "baseline": "#333333",
        "no-local s426": "#4c78a8",
        "no-local s526": "#e45756",
        "no-local s626": "#54a24b",
    }
    for label, _root in specs:
        profile_rows = [row for row in rows if row["row_type"] == "profile" and row["run"] == label]
        profile_rows.sort(key=lambda row: int(row["radius"]))
        profile_ax.plot(
            [int(row["radius"]) for row in profile_rows],
            [float(row["shell_rms"]) for row in profile_rows],
            marker="o",
            linewidth=2.0,
            color=colors[label],
            label=label,
        )
    profile_ax.axvline(4, color="#777777", linestyle="--", linewidth=0.9)
    profile_ax.axvline(6, color="#999999", linestyle=":", linewidth=0.9)
    profile_ax.set_xlabel("offset radius")
    profile_ax.set_ylabel("shell centered RMS")
    profile_ax.set_title("CIFAR reconstruction bias shape: center spike vs no-local residual")
    profile_ax.set_ylim(0.0, max(0.55, vmax * 1.05))
    profile_ax.legend(loc="upper right", ncols=2)
    fig.savefig(output_dir / "figure_g_cifar_bias_shape.png", dpi=200)
    plt.close(fig)
    return rows


def write_report(output_dir: Path, summaries: dict[str, list[dict[str, object]]]) -> None:
    def display_pt(value: str) -> str:
        try:
            parsed = float(value)
        except ValueError:
            return value
        if not math.isfinite(parsed):
            return "NA"
        return f"{value}pt"

    lines = [
        "# V5 Mechanism Figures",
        "",
        "Generated from existing V5 CSV artifacts. No training is run.",
        "",
        "## Figures",
        "",
        "- `figure_a_head_function_alignment.png`: train-time function alignment.",
        "- `figure_b_digit_compensation_chains.png`: Rotated cascade vs Affine pool.",
        "- `figure_c_cifar_reconstruction_pool_exhaustion.png`: CIFAR reconstruction local pool exhaustion.",
        "- `figure_d_task_split.png`: task split by positional-bias gain.",
        "- `figure_e_no_local_best_final_radial.png`: no-local-pool best/final radial comparison.",
        "- `figure_f_no_local_three_seed_distribution.png`: no-local-pool three-seed distribution.",
        "- `figure_g_cifar_bias_shape.png`: CIFAR reconstruction bias heatmaps and radial profiles.",
        "",
        "## Key Numbers",
        "",
    ]
    role_rows = read_csv(Path("results/v5_train_time_function_alignment/role_aggregate.csv"))
    for role in ("primary_top2", "primary_top3", "pool_exhaustion"):
        row = next((item for item in role_rows if item["role"] == role), None)
        if row:
            lines.append(
                f"- {role}: mean keep gap {display_pt(row['mean_keep_gap_pt'])}, "
                f"mean ablate drop {display_pt(row['mean_ablate_drop_pt'])}."
            )
    cifar_seed426 = next(row for row in summaries["figure_c"] if row["stage"] == "L0-L5 all s426")
    cifar_seed526 = next(row for row in summaries["figure_c"] if row["stage"] == "L0-L5 all s526")
    cifar_seed626 = next(row for row in summaries["figure_c"] if row["stage"] == "L0-L5 all s626")
    affine_mid = next(
        row
        for row in summaries["figure_b"]
        if row["chain"] == "Affine pool" and row["stage"] == "L0+L1H2/4/5/6"
    )
    lines.append(
        f"- CIFAR reconstruction L0-L5 all seed426: final {cifar_seed426['final']}, "
        f"visible {cifar_seed426['visible']}, zero {cifar_seed426['zero']}."
    )
    lines.append(
        f"- CIFAR reconstruction L0-L5 all seed526: final {cifar_seed526['final']}, "
        f"visible {cifar_seed526['visible']}, zero {cifar_seed526['zero']}."
    )
    lines.append(
        f"- CIFAR reconstruction L0-L5 all seed626: final {cifar_seed626['final']}, "
        f"visible {cifar_seed626['visible']}, zero {cifar_seed626['zero']}."
    )
    lines.append(
        f"- Affine L0all+L1H2/H4/H5/H6: final {affine_mid['final']}, "
        f"zero {affine_mid['zero']}."
    )
    seed526_final_r6 = next(
        row
        for row in summaries["figure_e"]
        if row["seed"] == "seed526" and row["state"] == "final" and row["metric"] == "r<=6"
    )
    seed526_best_r6 = next(
        row
        for row in summaries["figure_e"]
        if row["seed"] == "seed526" and row["state"] == "best" and row["metric"] == "r<=6"
    )
    lines.append(
        f"- CIFAR no-local seed526 r<=6: final {seed526_final_r6['score']}, "
        f"best {seed526_best_r6['score']}."
    )
    no_local_full = next(
        row for row in summaries["figure_f"] if row["row_type"] == "aggregate" and row["metric"] == "full"
    )
    no_local_r6 = next(
        row for row in summaries["figure_f"] if row["row_type"] == "aggregate" and row["metric"] == "r<=6"
    )
    no_local_visible = next(
        row for row in summaries["figure_f"] if row["row_type"] == "aggregate" and row["metric"] == "visible"
    )
    lines.append(
        f"- CIFAR no-local final full mean {no_local_full['mean']} +/- {no_local_full['sd']} "
        f"over three seeds."
    )
    lines.append(
        f"- CIFAR no-local final r<=6 mean {no_local_r6['mean']} +/- {no_local_r6['sd']}; "
        f"visible mean {no_local_visible['mean']} +/- {no_local_visible['sd']}."
    )
    baseline_shape = next(
        row for row in summaries["figure_g"] if row["row_type"] == "summary" and row["run"] == "baseline"
    )
    no_local_shape = next(
        row for row in summaries["figure_g"] if row["row_type"] == "aggregate" and row["run"] == "no-local mean"
    )
    lines.append(
        f"- CIFAR bias-shape center RMS: baseline {baseline_shape['center_rms']}, "
        f"no-local mean {no_local_shape['center_rms']}."
    )
    lines.append(
        f"- CIFAR no-local shell RMS: r=6 mean {no_local_shape['r6_shell_rms']}, "
        f"r=7 mean {no_local_shape['r7_shell_rms']}."
    )
    lines += [
        "",
        "## Data Files",
        "",
        "- `figure_a_head_function_alignment.csv`",
        "- `figure_b_digit_compensation_chains.csv`",
        "- `figure_c_cifar_reconstruction_pool_exhaustion.csv`",
        "- `figure_d_task_split.csv`",
        "- `figure_e_no_local_best_final_radial.csv`",
        "- `figure_f_no_local_three_seed_distribution.csv`",
        "- `figure_g_cifar_bias_shape.csv`",
        "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    setup_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        "figure_a": figure_a_alignment(output_dir),
        "figure_b": figure_b_digit_chains(output_dir),
        "figure_c": figure_c_cifar_reconstruction(output_dir),
        "figure_d": figure_d_task_split(output_dir),
        "figure_e": figure_e_no_local_best_final_radial(output_dir),
        "figure_f": figure_f_no_local_three_seed_distribution(output_dir),
        "figure_g": figure_g_cifar_bias_shape(output_dir),
    }
    write_report(output_dir, summaries)
    summary = {
        "output_dir": str(output_dir),
        "figures": [
            str(output_dir / "figure_a_head_function_alignment.png"),
            str(output_dir / "figure_b_digit_compensation_chains.png"),
            str(output_dir / "figure_c_cifar_reconstruction_pool_exhaustion.png"),
            str(output_dir / "figure_d_task_split.png"),
            str(output_dir / "figure_e_no_local_best_final_radial.png"),
            str(output_dir / "figure_f_no_local_three_seed_distribution.png"),
            str(output_dir / "figure_g_cifar_bias_shape.png"),
        ],
        "data_csvs": [
            str(output_dir / "figure_a_head_function_alignment.csv"),
            str(output_dir / "figure_b_digit_compensation_chains.csv"),
            str(output_dir / "figure_c_cifar_reconstruction_pool_exhaustion.csv"),
            str(output_dir / "figure_d_task_split.csv"),
            str(output_dir / "figure_e_no_local_best_final_radial.csv"),
            str(output_dir / "figure_f_no_local_three_seed_distribution.csv"),
            str(output_dir / "figure_g_cifar_bias_shape.csv"),
        ],
        "report": str(output_dir / "REPORT.md"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/v5_mechanism_figures")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
