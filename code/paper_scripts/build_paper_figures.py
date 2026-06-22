#!/usr/bin/env python3
"""Build MetricToric paper figures from copied experiment artifacts.

The script intentionally regenerates compact publication figures from CSV/NPZ
records instead of embedding raw experiment PNGs.  It expects to live in
MetricToric/paper/scripts and writes PDFs into MetricToric/paper/figures.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PAPER_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIG_DIR = PAPER_DIR / "figures"

BLUE = "#2F5D8C"
TEAL = "#2F7F73"
ORANGE = "#C56B32"
RED = "#B24A48"
PURPLE = "#7761A7"
GRAY = "#6C6C6C"
LIGHT_GRAY = "#D8D8D8"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.titlesize": 10.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def result_path(relative: str) -> Path:
    path = RESULTS_DIR / relative
    if not path.exists():
        raise FileNotFoundError(f"missing result artifact: {path}")
    return path


def read_csv(relative: str) -> list[dict[str, str]]:
    path = result_path(relative)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = row.get(key, "")
    if raw in ("", "nan", "NaN", None):
        return default
    return float(raw)


def pick(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(expected) for key, expected in criteria.items())
    ]
    if not matches:
        detail = ", ".join(f"{k}={v}" for k, v in criteria.items())
        raise KeyError(f"no CSV row matched: {detail}")
    return matches[0]


def save_pdf(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def symmetric_limits(arr: np.ndarray) -> tuple[float, float]:
    limit = float(np.nanmax(np.abs(arr)))
    if limit == 0:
        limit = 1.0
    return -limit, limit


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(colors="#333333")
    ax.yaxis.label.set_color("#333333")
    ax.xaxis.label.set_color("#333333")
    ax.title.set_color("#222222")


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", dy: float = 0.006) -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        va = "bottom" if height >= 0 else "top"
        offset = dy if height >= 0 else -dy
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=6.5,
            color="#333333",
        )


def axial_projection(table: np.ndarray) -> np.ndarray:
    return table.mean(axis=1, keepdims=True) + table.mean(axis=0, keepdims=True) - table.mean()


def mixed_hessian(table: np.ndarray) -> np.ndarray:
    return (table[2:, 2:] - table[2:, :-2] - table[:-2, 2:] + table[:-2, :-2]) / 4.0


def make_manifest(rows: list[dict[str, str]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "manifest.csv"
    fieldnames = ["figure", "claim", "artifact_id", "source_file"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    title: str,
    extent: tuple[float, float, float, float],
    cbar_label: str = "",
) -> None:
    vmin, vmax = symmetric_limits(data)
    im = ax.imshow(
        data,
        origin="lower",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel(r"$\Delta y$")
    ax.set_xticks([-7, 0, 7] if extent[0] <= -7 else [-6, 0, 6])
    ax.set_yticks([-7, 0, 7] if extent[2] <= -7 else [-6, 0, 6])
    clean_axis(ax)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if cbar_label:
        cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=6.5)


def figure_1(manifest: list[dict[str, str]]) -> None:
    npz_rel = (
        "v4_cifar10_bias_export_10k/bias_exports/"
        "cifar10_reconstruction_relative_2d_table_seed477_steps10000/bias_tables.npz"
    )
    geom_rel = "v4_relative_table_geometry_10k/geometry_aggregate.csv"

    bundle = np.load(result_path(npz_rel), allow_pickle=True)
    table = bundle["tables"].astype(float).mean(axis=(0, 1))
    axis_table = bundle["axis_tables"].astype(float).mean(axis=(0, 1))
    residual = bundle["residual_tables"].astype(float).mean(axis=(0, 1))
    mixed = mixed_hessian(table)

    geom_rows = read_csv(geom_rel)
    basis_rows = [
        ("Axis", "axis_additive"),
        ("Table", "relative_2d_table"),
        ("PJ-J2", "toric_PJ_R2"),
        ("Shuffled", "toric_PJ_R2_coord_shuffle"),
    ]
    metric_specs = [
        ("Non-axial", "obl_ratio_mean", BLUE),
        ("Mixed", "mixed_ratio_mean", ORANGE),
        ("DCT mass", "topk_dct_mass_mean", TEAL),
    ]

    labels = []
    metrics = {name: [] for name, _, _ in metric_specs}
    for public_label, basis in basis_rows:
        row = pick(
            geom_rows,
            basis=basis,
            gauge="centered",
            boundary="interior_only",
        )
        labels.append(public_label)
        for name, key, _ in metric_specs:
            metrics[name].append(value(row, key))

    fig = plt.figure(figsize=(7.15, 4.65))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.25],
        height_ratios=[1.0, 0.95],
        wspace=0.5,
        hspace=0.72,
    )
    heatmap(fig.add_subplot(grid[0, 0]), table, "Mean table", (-7.5, 7.5, -7.5, 7.5))
    heatmap(fig.add_subplot(grid[0, 1]), axis_table, "Axial part", (-7.5, 7.5, -7.5, 7.5))
    heatmap(fig.add_subplot(grid[0, 2]), residual, "Non-axial residual", (-7.5, 7.5, -7.5, 7.5))
    heatmap(fig.add_subplot(grid[1, 0]), mixed, "Mixed second diff", (-6.5, 6.5, -6.5, 6.5))

    ax = fig.add_subplot(grid[1, 1:])
    x = np.arange(len(labels))
    width = 0.23
    for i, (name, _, color) in enumerate(metric_specs):
        offset = (i - 1) * width
        ax.bar(x + offset, metrics[name], width, label=name, color=color)
    ax.set_title("Normalized geometry diagnostics")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    clean_axis(ax)

    save_pdf(fig, "fig1_teacher_geometry.pdf")
    manifest.extend(
        [
            {
                "figure": "fig1_teacher_geometry.pdf",
                "claim": "relative table exposes axial, oblique, mixed, and spectral geometry",
                "artifact_id": "cifar10_relative_table_seed477_npz",
                "source_file": str((RESULTS_DIR / npz_rel).relative_to(PROJECT_DIR)),
            },
            {
                "figure": "fig1_teacher_geometry.pdf",
                "claim": "relative table exposes axial, oblique, mixed, and spectral geometry",
                "artifact_id": "geometry_aggregate",
                "source_file": str((RESULTS_DIR / geom_rel).relative_to(PROJECT_DIR)),
            },
        ]
    )


def figure_2(manifest: list[dict[str, str]]) -> None:
    teacher_rel = "v4_cifar10_bias_export_10k/real_vision_aggregate.csv"
    student_rel = "v4_metric_toric_pj_cifar10_10k/student_aggregate.csv"
    projection_rel = "v4_table_projection_cifar10_10k/projection_aggregate.csv"

    teacher_row = pick(read_csv(teacher_rel), basis="relative_2d_table")
    student_rows = read_csv(student_rel)
    projection_rows = read_csv(projection_rel)

    student_specs = [
        ("Teacher\n225", teacher_row, "final_score_mean", "score_std", GRAY),
        (
            "Table-informed\n109",
            pick(student_rows, basis="table_informed_toric_PJ_R0_top110"),
            "final_score_mean",
            "score_std",
            BLUE,
        ),
        (
            "Axis+PJ\n55",
            pick(student_rows, basis="axis_plus_toric_residual_R0_top55"),
            "final_score_mean",
            "score_std",
            TEAL,
        ),
        ("DCT\n34", pick(student_rows, basis="dct_top33"), "final_score_mean", "score_std", ORANGE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), gridspec_kw={"width_ratios": [1.05, 1.1]})

    ax = axes[0]
    labels = [spec[0] for spec in student_specs]
    means = [value(spec[1], spec[2]) for spec in student_specs]
    stds = [0.0 if label.startswith("Teacher") else value(spec[1], spec[3]) for label, spec in zip(labels, student_specs)]
    colors = [spec[4] for spec in student_specs]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="#333333", linewidth=0.4)
    bars[0].set_facecolor("white")
    bars[0].set_edgecolor(GRAY)
    bars[0].set_hatch("///")
    bars[0].set_linewidth(0.9)
    label_bars(ax, bars, "{:.3f}", dy=0.004)
    ax.set_title(r"Zoomed final task $R^2$")
    ax.set_ylabel(r"Final task $R^2$ (zoomed)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.80, 0.885)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    clean_axis(ax)

    ax = axes[1]
    series = [
        ("DCT", "topk_dct", ORANGE),
        ("Table-informed", "table_informed_toric_PJ_R0", BLUE),
        ("Axis+PJ", "axis_plus_toric_residual_R0", TEAL),
    ]
    budgets = ["33", "55", "110"]
    for label, variant, color in series:
        xs = []
        ys = []
        for budget in budgets:
            row = pick(
                projection_rows,
                variant=variant,
                fit_target="full_table",
                feature_budget=budget,
            )
            xs.append(value(row, "num_features"))
            ys.append(value(row, "table_fit_r2_mean"))
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.5, color=color, label=label)
    ax.set_title("Teacher-table fit")
    ax.set_xlabel("Atoms")
    ax.set_ylabel(r"Full-table fit $R^2$")
    ax.set_ylim(0.70, 1.01)
    ax.set_xlim(25, 116)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    clean_axis(ax)

    fig.tight_layout()
    save_pdf(fig, "fig2_student_compression.pdf")
    for artifact_id, rel in [
        ("teacher_task_row", teacher_rel),
        ("student_aggregate", student_rel),
        ("projection_aggregate", projection_rel),
    ]:
        manifest.append(
            {
                "figure": "fig2_student_compression.pdf",
                "claim": "compact table-initialized students and DCT compression baseline",
                "artifact_id": artifact_id,
                "source_file": str((RESULTS_DIR / rel).relative_to(PROJECT_DIR)),
            }
        )


def figure_3(manifest: list[dict[str, str]]) -> None:
    raw_rel = "v4_offset_holdout_cifar10_10k/offset_holdout_aggregate.csv"
    axis_reg_rel = "v4_offset_holdout_reg_cifar10_10k_h10b1_axis/offset_holdout_aggregate.csv"
    dct_eval_rel = (
        "v4_offset_holdout_eval_controls_dct_h1b10_cifar10_10k/"
        "offset_holdout_eval_controls_aggregate.csv"
    )
    dct_reg_rel = (
        "v4_offset_holdout_eval_controls_dct_h1b10_cifar10_10k/"
        "offset_holdout_aggregate.csv"
    )

    raw_rows = read_csv(raw_rel)
    axis_reg_rows = read_csv(axis_reg_rel)
    dct_eval_rows = read_csv(dct_eval_rel)
    dct_reg_rows = read_csv(dct_reg_rel)

    raw_specs = [
        ("Table", "relative_2d_table"),
        ("DCT\n34", "dct_top33"),
        ("Toric\n109", "table_informed_toric_PJ_R0_top110"),
        ("Axis+PJ\n55", "axis_plus_toric_residual_R0_top55"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1), gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    x = np.arange(len(raw_specs))
    width = 0.35
    full = []
    visible = []
    for _, basis in raw_specs:
        row = pick(raw_rows, basis=basis)
        full.append(value(row, "final_score_mean"))
        visible.append(value(row, "final_visible_only_mean"))
    ax.bar(x - width / 2, full, width, label="Full deployment", color=BLUE)
    ax.bar(x + width / 2, visible, width, label="Visible only", color=TEAL)
    ax.text(
        x[1] - width / 2,
        full[1] * 0.50,
        "-0.093",
        ha="center",
        va="center",
        fontsize=6.4,
        color="white",
    )
    ax.axhline(0, color="#444444", linewidth=0.7)
    ax.set_title("Visible fit vs deployment")
    ax.set_ylabel(r"Task $R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in raw_specs])
    ax.set_ylim(-0.18, 0.92)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    clean_axis(ax)

    ax = axes[1]
    dct_raw = pick(raw_rows, basis="dct_top33")
    dct_full = pick(dct_eval_rows, eval_mode="full")
    dct_visible = pick(dct_eval_rows, eval_mode="visible_only")
    dct_clamp = pick(dct_eval_rows, eval_mode="heldout_clamp")
    axis_raw = pick(raw_rows, basis="axis_plus_toric_residual_R0_top55")
    axis_reg = pick(axis_reg_rows, basis="axis_plus_toric_residual_R0_top55")
    gap_labels = ["DCT\nraw", "DCT\n+reg", "DCT\nclamp", "Axis+PJ\nraw", "Axis+PJ\n+reg"]
    gaps = [
        value(dct_raw, "final_extrapolation_gain_mean"),
        value(dct_full, "score_mean") - value(dct_visible, "score_mean"),
        value(dct_clamp, "score_mean") - value(dct_visible, "score_mean"),
        value(axis_raw, "final_extrapolation_gain_mean"),
        value(axis_reg, "final_extrapolation_gain_mean"),
    ]
    colors = [ORANGE, ORANGE, GRAY, TEAL, TEAL]
    bars = ax.bar(np.arange(len(gap_labels)), gaps, color=colors, edgecolor="#333333", linewidth=0.4)
    label_bars(ax, bars, "{:.2f}", dy=0.035)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_title("Deployment gap")
    ax.set_ylabel(r"Full $-$ visible")
    ax.set_xticks(np.arange(len(gap_labels)))
    ax.set_xticklabels(gap_labels)
    ax.set_ylim(-0.98, 0.10)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    clean_axis(ax)

    fig.tight_layout()
    save_pdf(fig, "fig3_boundary_holdout.pdf")
    for artifact_id, rel in [
        ("offset_holdout_raw", raw_rel),
        ("axis_boundary_regularized", axis_reg_rel),
        ("dct_eval_controls", dct_eval_rel),
        ("dct_boundary_regularized", dct_reg_rel),
    ]:
        manifest.append(
            {
                "figure": "fig3_boundary_holdout.pdf",
                "claim": "visible-window fit is not deployment safety",
                "artifact_id": artifact_id,
                "source_file": str((RESULTS_DIR / rel).relative_to(PROJECT_DIR)),
            }
        )


def figure_4(manifest: list[dict[str, str]]) -> None:
    rel = "v8_tinyimagenet_mechanism_summary/tinyimagenet_mechanism_summary.csv"
    rows = read_csv(rel)

    model_specs = [
        ("Toric PJ", "Toric/PJ", BLUE),
        ("Mixed r4", "Mixed r4", PURPLE),
    ]
    conditions = [
        ("Zero", "zero_bias_column"),
        ("Ablate\nL0-L3", "L0-L3 ablate r<=4"),
        ("Full", "full_column"),
        ("Keep\nL0-L3", "L0-L3 keep r<=4"),
    ]

    fig, ax = plt.subplots(figsize=(7.15, 3.0))
    group_x = np.arange(len(conditions))
    width = 0.32
    for idx, (public, model, color) in enumerate(model_specs):
        vals = []
        errs = []
        model_rows = [row for row in rows if row.get("model") == model]
        if not model_rows:
            raise KeyError(f"no rows for model={model}")
        for _, condition in conditions:
            if condition == "zero_bias_column":
                vals.append(value(model_rows[0], "zero"))
            elif condition == "full_column":
                vals.append(value(model_rows[0], "full"))
            else:
                row = pick(rows, model=model, condition=condition)
                vals.append(value(row, "score"))
            errs.append(0.0)
        offset = (idx - 0.5) * width
        bars = ax.bar(
            group_x + offset,
            vals,
            width,
            yerr=errs,
            capsize=3,
            label=public,
            color=color,
            edgecolor="#333333",
            linewidth=0.4,
        )
        label_bars(ax, bars, "{:.2f}", dy=0.012)
    ax.set_title("TinyImageNet local-pool intervention")
    ax.set_ylabel(r"Task $R^2$")
    ax.set_xticks(group_x)
    ax.set_xticklabels([label for label, _ in conditions])
    ax.set_ylim(0.18, 0.92)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    clean_axis(ax)
    fig.tight_layout()
    save_pdf(fig, "fig4_local_pool.pdf")
    manifest.append(
        {
            "figure": "fig4_local_pool.pdf",
            "claim": "local pools are sufficient under keep and nearly necessary under ablation",
            "artifact_id": "tinyimagenet_mechanism_summary",
            "source_file": str((RESULTS_DIR / rel).relative_to(PROJECT_DIR)),
        }
    )


def figure_5(manifest: list[dict[str, str]]) -> None:
    main_rel = "v8_mechanism_claim_synthesis/v8_tinyimagenet_10k_main_table.csv"
    repair_rel = "v9_tinyimagenet_farband_teacher_10k_summary/farband_teacher_10k_summary.csv"
    rows = read_csv(main_rel)
    repair_rows = read_csv(repair_rel)

    specs = [
        ("DCT top110\n+ const\n(111 cols)", pick(rows, model="DCT top110"), BLUE),
        ("Toric PJ\n220", pick(rows, model="Toric/PJ top220"), ORANGE),
        ("Mixed\nhard r4", pick(rows, model="Mixed residual-DCT32 hard r4"), TEAL),
        ("Mixed\nsoft", pick(rows, model="Mixed residual-DCT32 soft w=1"), PURPLE),
    ]
    repair = repair_rows[0]

    labels = [spec[0] for spec in specs] + ["Far-band\nrepair"]
    colors = [spec[2] for spec in specs] + [RED]
    final = [value(spec[1], "final") for spec in specs] + [value(repair, "full")]
    visible = [value(spec[1], "visible") for spec in specs] + [value(repair, "visible")]
    far = [value(spec[1], "radial_band_6_9") for spec in specs] + [value(repair, "r6_9")]

    fig, ax = plt.subplots(figsize=(7.15, 3.15))
    x = np.arange(len(labels))
    width = 0.24
    ax.bar(x - width, final, width, color=colors, alpha=0.95, label="Full")
    ax.bar(x, visible, width, color=colors, alpha=0.55, label="Visible")
    ax.bar(x + width, far, width, color=colors, alpha=0.25, label="Far band")
    ax.set_title("TinyImageNet far-shell readouts")
    ax.set_ylabel(r"Task $R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 0.94)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    clean_axis(ax)
    fig.tight_layout()
    save_pdf(fig, "fig5_far_shell.pdf")
    for artifact_id, rel in [
        ("tinyimagenet_main_table", main_rel),
        ("farband_teacher_summary", repair_rel),
    ]:
        manifest.append(
            {
                "figure": "fig5_far_shell.pdf",
                "claim": "full, visible, and far-band behavior separate; far-band repair is supervised",
                "artifact_id": artifact_id,
                "source_file": str((RESULTS_DIR / rel).relative_to(PROJECT_DIR)),
            }
        )


def figure_6(manifest: list[dict[str, str]]) -> None:
    rel = "v11e_cifar10_ordinary_classification_relpos_medium_3seed_5k/real_vision_aggregate.csv"
    rows = read_csv(rel)
    specs = [
        ("Axis", "axis_additive", TEAL),
        ("No pos", "no_pos_constant", GRAY),
        ("Table", "relative_2d_table", BLUE),
        ("Shuffled\nPJ", "toric_PJ_R2_coord_shuffle", LIGHT_GRAY),
        ("PJ-J2", "toric_PJ_R2", ORANGE),
        ("Toric-J0", "toric_order0", PURPLE),
    ]
    labels = [spec[0] for spec in specs]
    means = []
    stds = []
    colors = []
    for _, basis, color in specs:
        row = pick(rows, basis=basis)
        means.append(value(row, "score_mean"))
        stds.append(value(row, "score_std"))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(7.15, 2.9))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="#333333", linewidth=0.4)
    for xpos, mean, std in zip(x, means, stds):
        ax.text(
            xpos,
            mean + std + 0.0012,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#333333",
        )
    ax.set_title("Zoomed CIFAR10 classification boundary")
    ax.set_ylabel("Accuracy (zoomed)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.54, 0.61)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    clean_axis(ax)
    fig.tight_layout()
    save_pdf(fig, "fig6_classification_boundary.pdf")
    manifest.append(
        {
            "figure": "fig6_classification_boundary.pdf",
            "claim": "small-model classification is a negative boundary for scalar local geometry",
            "artifact_id": "classification_aggregate",
            "source_file": str((RESULTS_DIR / rel).relative_to(PROJECT_DIR)),
        }
    )


def main() -> None:
    configure_style()
    manifest: list[dict[str, str]] = []
    figure_1(manifest)
    figure_2(manifest)
    figure_3(manifest)
    figure_4(manifest)
    figure_5(manifest)
    figure_6(manifest)
    make_manifest(manifest)
    print(f"wrote 6 figures and manifest to {FIG_DIR}")


if __name__ == "__main__":
    main()
