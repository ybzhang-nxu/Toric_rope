from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("results")
OUT = ROOT / "v7_mechanism_figures"
MAIN_CSV = ROOT / "v7_strong_dataset_summary" / "v7_main_results.csv"
CONTROL_CSV = ROOT / "v7_strong_dataset_summary" / "v7_control_results.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str] | dict[str, object], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def r4(value: float) -> float:
    return round(value, 4) if math.isfinite(value) else value


def first(rows: list[dict[str, str]], **kwargs: str) -> dict[str, str]:
    for row in rows:
        if all(row.get(k) == v for k, v in kwargs.items()):
            return row
    raise KeyError(kwargs)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def family_row(path: Path, selector_key: str | None = None, selector_value: str | None = None) -> dict[str, str]:
    rows = read_rows(path)
    if selector_key is None:
        return rows[0]
    return first(rows, **{selector_key: selector_value or ""})


def build_mechanism_rows(main_rows: list[dict[str, str]], control_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def main(dataset: str, basis: str) -> dict[str, str]:
        return first(main_rows, dataset=dataset, basis=basis)

    def control(dataset: str, basis: str, group: str, condition: str) -> dict[str, str]:
        return first(control_rows, dataset=dataset, basis=basis, group=group, condition=condition)

    c100_pure_basis = "table_informed_toric_PJ_R0_top220"
    c100_pure_main = main("cifar100", c100_pure_basis)
    c100_pure_keep = control("cifar100", c100_pure_basis, "early_l0_l1_l2_all_heads", "keep r<=4")
    c100_pure_ablate = control("cifar100", c100_pure_basis, "early_l0_l1_l2_all_heads", "ablate r<=4")
    rows.append(
        {
            "label": "CIFAR100 Toric/PJ",
            "dataset": "cifar100",
            "basis": c100_pure_basis,
            "policy": "standard",
            "full": f(c100_pure_main, "final_mean"),
            "visible": f(c100_pure_main, "visible_mean"),
            "zero": f(c100_pure_main, "zero_mean"),
            "local_keep": f(c100_pure_keep, "score_mean"),
            "local_ablate": f(c100_pure_ablate, "score_mean"),
            "all_local_ablate": float("nan"),
            "main_local_pool": "L0-L2 all heads",
            "claim": "clean Toric/PJ local mechanism; full extrapolation has mild harm",
        }
    )

    c100_mixed = family_row(
        ROOT / "v8_cifar100_mixed_component_mechanism_summary" / "component_mechanism_family_key_metrics.csv"
    )
    rows.append(
        {
            "label": "CIFAR100 mixed",
            "dataset": "cifar100",
            "basis": "mixed_toric_PJ_R0_top220_residual_dct_top32",
            "policy": "standard",
            "full": f(c100_mixed, "full_mean"),
            "visible": f(c100_mixed, "radial4_mean"),
            "zero": f(c100_mixed, "zero_mean"),
            "local_keep": f(c100_mixed, "l0123_keep_mean"),
            "local_ablate": f(c100_mixed, "l0123_ablate_mean"),
            "all_local_ablate": f(c100_mixed, "l012345_ablate_mean"),
            "main_local_pool": "L0-L3 all heads",
            "claim": "DCT-like full stability while preserving distributed Toric/PJ local pool",
        }
    )

    c10_mixed = family_row(
        ROOT / "v7_cifar10_mixed_component_mechanism_summary" / "component_mechanism_family_key_metrics.csv"
    )
    rows.append(
        {
            "label": "CIFAR10 mixed",
            "dataset": "cifar10",
            "basis": "mixed_toric_PJ_R0_top220_residual_dct_top32",
            "policy": "standard",
            "full": f(c10_mixed, "full_mean"),
            "visible": f(c10_mixed, "radial4_mean"),
            "zero": f(c10_mixed, "zero_mean"),
            "local_keep": f(c10_mixed, "l0123_keep_mean"),
            "local_ablate": f(c10_mixed, "l0123_ablate_mean"),
            "all_local_ablate": float("nan"),
            "main_local_pool": "L0-L3 all heads",
            "claim": "mixed basis rescues full gap and gives near-zero early-pool ablation",
        }
    )

    svhn_mixed = family_row(
        ROOT / "v7_svhn_component_mechanism_summary" / "component_mechanism_family_key_metrics.csv",
        "family",
        "mixed_residual_dct32",
    )
    rows.append(
        {
            "label": "SVHN mixed",
            "dataset": "svhn",
            "basis": "mixed_toric_PJ_R0_top220_residual_dct_top32",
            "policy": "standard",
            "full": f(svhn_mixed, "full_mean"),
            "visible": f(svhn_mixed, "radial4_mean"),
            "zero": f(svhn_mixed, "zero_mean"),
            "local_keep": f(svhn_mixed, "early_keep_mean"),
            "local_ablate": f(svhn_mixed, "early_ablate_mean"),
            "all_local_ablate": float("nan"),
            "main_local_pool": "L0-L2 all heads",
            "claim": "mixed basis is near relative-table full and keeps early local pool",
        }
    )

    stl_mixed = family_row(
        ROOT / "v7_stl10_trainradial_component_mechanism_summary" / "component_mechanism_family_key_metrics.csv"
    )
    rows.append(
        {
            "label": "STL10 train-radial",
            "dataset": "stl10",
            "basis": "mixed_toric_PJ_R0_top220_residual_dct_top32_trainradial_r4",
            "policy": "train-radial r=4",
            "full": f(stl_mixed, "full_mean"),
            "visible": f(stl_mixed, "radial4_mean"),
            "zero": f(stl_mixed, "zero_mean"),
            "local_keep": f(stl_mixed, "l0123_keep_mean"),
            "local_ablate": f(stl_mixed, "l0123_ablate_mean"),
            "all_local_ablate": f(stl_mixed, "l012345_ablate_mean"),
            "main_local_pool": "L0-L3 all heads",
            "claim": "explicit radial-local training rescues full and local-pool ablation exhausts to zero",
        }
    )

    for row in rows:
        full = f(row, "full")
        visible = f(row, "visible")
        zero = f(row, "zero")
        keep = f(row, "local_keep")
        ablate = f(row, "local_ablate")
        all_ablate = f(row, "all_local_ablate")
        row["full_minus_zero"] = r4(full - zero)
        row["full_minus_visible"] = r4(full - visible)
        row["keep_minus_full"] = r4(keep - full)
        row["local_ablate_drop"] = r4(full - ablate)
        row["local_ablate_gain_vs_zero"] = r4(ablate - zero)
        row["all_local_ablate_gain_vs_zero"] = r4(all_ablate - zero)
        for key in [
            "full",
            "visible",
            "zero",
            "local_keep",
            "local_ablate",
            "all_local_ablate",
        ]:
            row[key] = r4(f(row, key))
    return rows


def build_boundary_rows(main_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    selected = [
        ("C100 relative", "cifar100", "relative_2d_table"),
        ("C100 DCT", "cifar100", "dct_top110"),
        ("C100 Toric/PJ", "cifar100", "table_informed_toric_PJ_R0_top220"),
        ("C100 mixed", "cifar100", "mixed_toric_PJ_R0_top220_residual_dct_top32"),
        ("C10 Toric/PJ", "cifar10", "table_informed_toric_PJ_R0_top110"),
        ("C10 mixed", "cifar10", "mixed_toric_PJ_R0_top220_residual_dct_top32"),
        ("SVHN pure", "svhn", "table_informed_toric_PJ_R0_top220"),
        ("SVHN mixed", "svhn", "mixed_toric_PJ_R0_top220_residual_dct_top32"),
        ("STL10 train-radial", "stl10", "mixed_toric_PJ_R0_top220_residual_dct_top32_trainradial_r4"),
    ]
    rows = []
    for label, dataset, basis in selected:
        row = first(main_rows, dataset=dataset, basis=basis)
        rows.append(
            {
                "label": label,
                "dataset": dataset,
                "basis": basis,
                "full": r4(f(row, "final_mean")),
                "visible": r4(f(row, "visible_mean")),
                "zero": r4(f(row, "zero_mean")),
                "full_minus_zero": r4(f(row, "full_minus_zero_mean")),
                "full_minus_visible": r4(f(row, "full_minus_visible_mean")),
            }
        )
    return rows


def figure_main_results(rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    ax.bar(x - width, [f(row, "full") for row in rows], width=width, color="#4c78a8", label="full")
    ax.bar(x, [f(row, "visible") for row in rows], width=width, color="#54a24b", label="visible/local")
    ax.bar(x + width, [f(row, "zero") for row in rows], width=width, color="#bab0ac", label="zero-bias")
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("reconstruction R2")
    ax.set_title("V7 cross-dataset reconstruction: full, local, and zero-bias controls")
    ax.legend(loc="lower right")
    fig.savefig(OUT / "figure_1_cross_dataset_reconstruction_controls.png", dpi=200)
    plt.close(fig)


def figure_local_pool(rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.21
    fig, ax = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    ax.bar(x - 1.5 * width, [f(row, "zero") for row in rows], width=width, color="#bab0ac", label="zero-bias")
    ax.bar(x - 0.5 * width, [f(row, "local_ablate") for row in rows], width=width, color="#e45756", label="local-pool ablate")
    ax.bar(x + 0.5 * width, [f(row, "full") for row in rows], width=width, color="#4c78a8", label="full")
    ax.bar(x + 1.5 * width, [f(row, "local_keep") for row in rows], width=width, color="#54a24b", label="local-pool keep")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("reconstruction R2")
    ax.set_title("V7 local-pool causality: sufficiency and necessity")
    ax.legend(loc="lower right", ncols=2)
    fig.savefig(OUT / "figure_2_local_pool_causality.png", dpi=200)
    plt.close(fig)


def figure_boundary_gap(rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    gaps = [f(row, "full_minus_visible") for row in rows]
    colors = ["#4c78a8" if gap >= -0.02 else "#e45756" for gap in gaps]
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.bar(np.arange(len(rows)), gaps, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(-0.02, color="#777", linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(rows)), labels, rotation=24, ha="right")
    ax.set_ylabel("full - visible/local R2")
    ax.set_title("V7 boundary extrapolation gap")
    fig.savefig(OUT / "figure_3_boundary_extrapolation_gap.png", dpi=200)
    plt.close(fig)


def figure_mechanism_drop(rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    drop = [f(row, "local_ablate_drop") for row in rows]
    residual = [f(row, "local_ablate_gain_vs_zero") for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, drop, width=width, color="#f58518", label="full - local ablate")
    ax.bar(x + width / 2, residual, width=width, color="#72b7b2", label="local ablate - zero")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("R2 difference")
    ax.set_title("V7 mechanism necessity: drop and remaining compensation")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "figure_4_mechanism_drop_and_residual.png", dpi=200)
    plt.close(fig)


def write_report(mechanism_rows: list[dict[str, object]], boundary_rows: list[dict[str, object]]) -> None:
    strongest = max(mechanism_rows, key=lambda row: f(row, "local_ablate_drop"))
    stable = [row for row in boundary_rows if f(row, "full_minus_visible") >= -0.02]
    text = "\n".join(
        [
            "# V7 Mechanism Figures",
            "",
            "Generated from v7 strong-dataset summary and component-intervention summaries.",
            "",
            "## Files",
            "",
            "- `figure_1_cross_dataset_reconstruction_controls.png`: full / visible-local / zero-bias controls.",
            "- `figure_2_local_pool_causality.png`: local-pool keep and ablate evidence.",
            "- `figure_3_boundary_extrapolation_gap.png`: full minus visible-local boundary gap.",
            "- `figure_4_mechanism_drop_and_residual.png`: ablation drop and remaining compensation.",
            "- `mechanism_claim_table.csv`: compact table behind the mechanism claims.",
            "- `boundary_gap_table.csv`: compact table behind the boundary extrapolation figure.",
            "",
            "## Reading",
            "",
            (
                f"- Strongest local-pool drop in this cross-dataset table: {strongest['label']} "
                f"with drop={float(strongest['local_ablate_drop']):.4f}."
            ),
            (
                f"- Boundary-stable rows at full-visible >= -0.02: "
                f"{', '.join(str(row['label']) for row in stable)}."
            ),
            "- Mixed residual-DCT is the dominant full-performance pattern; Toric/PJ-style local heads remain causal.",
            "- STL10 needs explicit train-radial policy, but after rescue its all-local ablation exhausts to zero.",
            "",
        ]
    )
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    main_rows = read_rows(MAIN_CSV)
    control_rows = read_rows(CONTROL_CSV)
    mechanism_rows = build_mechanism_rows(main_rows, control_rows)
    boundary_rows = build_boundary_rows(main_rows)

    write_rows(OUT / "mechanism_claim_table.csv", mechanism_rows)
    write_rows(OUT / "boundary_gap_table.csv", boundary_rows)
    figure_main_results(boundary_rows)
    figure_local_pool(mechanism_rows)
    figure_boundary_gap(boundary_rows)
    figure_mechanism_drop(mechanism_rows)
    write_report(mechanism_rows, boundary_rows)
    summary = {
        "out": str(OUT),
        "mechanism_rows": len(mechanism_rows),
        "boundary_rows": len(boundary_rows),
        "figures": [
            "figure_1_cross_dataset_reconstruction_controls.png",
            "figure_2_local_pool_causality.png",
            "figure_3_boundary_extrapolation_gap.png",
            "figure_4_mechanism_drop_and_residual.png",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
