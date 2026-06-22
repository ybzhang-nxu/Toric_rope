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
class BaselineSpec:
    key: str
    dataset: str
    seed: int
    result_csv: Path
    intervention_csv: Path | None = None


@dataclass(frozen=True)
class FunctionSpec:
    key: str
    baseline_key: str
    role: str
    label: str
    train_csv: Path
    selection_basis: str
    heads_label: str


BASELINES = [
    BaselineSpec(
        key="mnist_s426",
        dataset="mnist",
        seed=426,
        result_csv=Path("results/v4_offset_holdout_classification_mnist_radial_causal_10k_start5000/offset_holdout_results.csv"),
        intervention_csv=Path("results/v4_component_intervention_eval_mnist_l0_subsets_final/component_interventions.csv"),
    ),
    BaselineSpec(
        key="mnist_s526",
        dataset="mnist",
        seed=526,
        result_csv=Path("results/v5_mnist_seed526_full_top110_10k_start5000/offset_holdout_results.csv"),
        intervention_csv=Path("results/v5_component_eval_seed526_final_layer0_individual/component_interventions.csv"),
    ),
    BaselineSpec(
        key="mnist_s626",
        dataset="mnist",
        seed=626,
        result_csv=Path("results/v5_mnist_seed626_full_top110_10k_start5000/offset_holdout_results.csv"),
        intervention_csv=Path("results/v5_component_eval_seed626_final_layer0_individual/component_interventions.csv"),
    ),
    BaselineSpec(
        key="rotated_s426",
        dataset="rotated-mnist",
        seed=426,
        result_csv=Path("results/v5_rotated_mnist_full_top110_10k_start5000/offset_holdout_results.csv"),
        intervention_csv=Path("results/v5_component_eval_rotated_mnist_final_layer0_individual/component_interventions.csv"),
    ),
    BaselineSpec(
        key="affine_s426",
        dataset="affine-mnist",
        seed=426,
        result_csv=Path("results/v5_affine_mnist_full_top110_10k_start5000/offset_holdout_results.csv"),
        intervention_csv=Path("results/v5_component_eval_affine_mnist_final_layer0_individual/component_interventions.csv"),
    ),
]


FUNCTIONS = [
    FunctionSpec(
        key="mnist_s426_l0_top1",
        baseline_key="mnist_s426",
        role="primary_top1",
        label="MNIST s426 L0 top1",
        train_csv=Path("results/v5_mnist_train_component_l0h3_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking",
        heads_label="L0H3",
    ),
    FunctionSpec(
        key="mnist_s426_l0_top2",
        baseline_key="mnist_s426",
        role="primary_top2",
        label="MNIST s426 L0 top2",
        train_csv=Path("results/v5_mnist_train_component_l0h36_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking",
        heads_label="L0H3+H6",
    ),
    FunctionSpec(
        key="mnist_s426_l0_top3",
        baseline_key="mnist_s426",
        role="primary_top3",
        label="MNIST s426 L0 top3",
        train_csv=Path("results/v5_mnist_train_component_l0h361_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking",
        heads_label="L0H3+H6+H1",
    ),
    FunctionSpec(
        key="mnist_s526_l0_top2",
        baseline_key="mnist_s526",
        role="primary_top2",
        label="MNIST s526 L0 top2",
        train_csv=Path("results/v5_mnist_train_component_seed526_l0h53_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking",
        heads_label="L0H5+H3",
    ),
    FunctionSpec(
        key="mnist_s526_l0_top3",
        baseline_key="mnist_s526",
        role="primary_top3",
        label="MNIST s526 L0 top3",
        train_csv=Path("results/v5_mnist_train_component_seed526_l0h537_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking",
        heads_label="L0H5+H3+H7",
    ),
    FunctionSpec(
        key="mnist_s626_l0_top2",
        baseline_key="mnist_s626",
        role="primary_top2",
        label="MNIST s626 L0 top2",
        train_csv=Path("results/v5_mnist_train_component_seed626_l0h24_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking",
        heads_label="L0H2+H4",
    ),
    FunctionSpec(
        key="mnist_s626_l0_top3_h6",
        baseline_key="mnist_s626",
        role="primary_top3",
        label="MNIST s626 L0 top3/H6",
        train_csv=Path("results/v5_mnist_train_component_seed626_l0h246_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking + train-time auxiliary",
        heads_label="L0H2+H4+H6",
    ),
    FunctionSpec(
        key="mnist_s626_l0_top3_h1",
        baseline_key="mnist_s626",
        role="primary_top3_alt",
        label="MNIST s626 L0 top3/H1",
        train_csv=Path("results/v5_mnist_train_component_seed626_l0h241_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking + auxiliary sanity",
        heads_label="L0H2+H4+H1",
    ),
    FunctionSpec(
        key="rotated_s426_l0_top3",
        baseline_key="rotated_s426",
        role="primary_top3",
        label="Rotated s426 L0 top3",
        train_csv=Path("results/v5_rotated_mnist_train_component_l0h347_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking",
        heads_label="L0H3+H4+H7",
    ),
    FunctionSpec(
        key="rotated_s426_comp",
        baseline_key="rotated_s426",
        role="conditional_compensation",
        label="Rotated s426 L1/L2 comp",
        train_csv=Path("results/v5_rotated_mnist_train_component_comp_l1h6_l2h3_10k_start5000/offset_holdout_results.csv"),
        selection_basis="post-L0-ablate compensation eval",
        heads_label="L1H6+L2H3",
    ),
    FunctionSpec(
        key="rotated_s426_cond",
        baseline_key="rotated_s426",
        role="conditional_exhaustion",
        label="Rotated s426 L0all+comp ablate",
        train_csv=Path("results/v5_rotated_mnist_train_component_cond_l0all_l1h6_l2h3_10k_start5000/offset_holdout_results.csv"),
        selection_basis="conditional ablate after L0 pool removal",
        heads_label="L0 all + L1H6+L2H3",
    ),
    FunctionSpec(
        key="affine_s426_l0_top5",
        baseline_key="affine_s426",
        role="primary_top5",
        label="Affine s426 L0 top5",
        train_csv=Path("results/v5_affine_mnist_train_component_l0h31476_10k_start5000/offset_holdout_results.csv"),
        selection_basis="eval L0 r<=4 ranking",
        heads_label="L0H3+H1+H4+H7+H6",
    ),
    FunctionSpec(
        key="affine_s426_comp",
        baseline_key="affine_s426",
        role="conditional_compensation",
        label="Affine s426 L1 comp",
        train_csv=Path("results/v5_affine_mnist_train_component_comp_l1h2_l1h4_10k_start5000/offset_holdout_results.csv"),
        selection_basis="post-L0-ablate compensation eval",
        heads_label="L1H2+H4",
    ),
    FunctionSpec(
        key="affine_s426_l0all",
        baseline_key="affine_s426",
        role="pool_exhaustion",
        label="Affine s426 L0all ablate",
        train_csv=Path("results/v5_affine_mnist_train_component_l0all_10k_start5000/offset_holdout_results.csv"),
        selection_basis="remove whole L0 local pool",
        heads_label="L0 all",
    ),
    FunctionSpec(
        key="affine_s426_cond_l1h24",
        baseline_key="affine_s426",
        role="conditional_reallocation",
        label="Affine s426 L0all+L1H2/H4 ablate",
        train_csv=Path("results/v5_affine_mnist_train_component_cond_l0all_l1h2_l1h4_10k_start5000/offset_holdout_results.csv"),
        selection_basis="conditional ablate of visible L1 compensation heads",
        heads_label="L0 all + L1H2+H4",
    ),
    FunctionSpec(
        key="affine_s426_cond_l1h2456",
        baseline_key="affine_s426",
        role="conditional_pool_sanity",
        label="Affine s426 L0all+L1H2/H4/H5/H6 ablate",
        train_csv=Path("results/v5_affine_mnist_train_component_cond_l0all_l1h2456_10k_start5000/offset_holdout_results.csv"),
        selection_basis="conditional ablate of visible plus reallocated L1 compensation heads",
        heads_label="L0 all + L1H2+H4+H5+H6",
    ),
    FunctionSpec(
        key="affine_s426_l0_l1_all",
        baseline_key="affine_s426",
        role="pool_exhaustion",
        label="Affine s426 L0all+L1all ablate",
        train_csv=Path("results/v5_affine_mnist_train_component_l0all_l1all_10k_start5000/offset_holdout_results.csv"),
        selection_basis="remove L0 and L1 local pools",
        heads_label="L0 all + L1 all",
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


def row_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def fmt(value: float, digits: int = 4) -> float:
    if not math.isfinite(value):
        return value
    return round(value, digits)


def parse_head_param(value: str) -> tuple[int, int, float] | None:
    match = HEAD_PARAM_RE.search(value)
    if not match:
        return None
    return int(match.group("layer")), int(match.group("head")), float(match.group("radius"))


def parse_spec_heads(value: str) -> list[tuple[int, int]]:
    heads: list[tuple[int, int]] = []
    for item in value.split(","):
        if ":" not in item:
            continue
        layer_text, head_text = item.split(":", 1)
        try:
            layer = int(layer_text.strip().removeprefix("L").removeprefix("l"))
        except ValueError:
            continue
        for raw in head_text.replace("|", "+").split("+"):
            raw = raw.strip().removeprefix("H").removeprefix("h")
            if not raw:
                continue
            try:
                heads.append((layer, int(raw)))
            except ValueError:
                continue
    return heads


def load_baseline(spec: BaselineSpec) -> dict[str, object]:
    if not spec.result_csv.exists():
        raise FileNotFoundError(spec.result_csv)
    rows = read_csv(spec.result_csv)
    if not rows:
        raise ValueError(f"empty baseline csv: {spec.result_csv}")
    row = rows[0]
    out: dict[str, object] = {
        "baseline_key": spec.key,
        "dataset": spec.dataset,
        "seed": spec.seed,
        "baseline_final": row_float(row, "final_score"),
        "baseline_best": row_float(row, "best_score"),
        "baseline_zero": row_float(row, "final_zero_bias_score"),
        "baseline_visible": row_float(row, "final_visible_only_score"),
    }
    if spec.intervention_csv and spec.intervention_csv.exists():
        out.update(load_eval_ranking(spec.intervention_csv))
    return out


def load_eval_ranking(path: Path) -> dict[str, object]:
    rows = read_csv(path)
    head_rows: list[dict[str, object]] = []
    for row in rows:
        if row.get("eval_mode") != "head_radial_ablate":
            continue
        parsed = parse_head_param(row.get("eval_param", ""))
        if parsed is None:
            continue
        layer, head, radius = parsed
        if abs(radius - 4.0) > 1e-6:
            continue
        drop_pt = -100.0 * row_float(row, "delta_vs_full")
        head_rows.append({"layer": layer, "head": head, "drop_pt": drop_pt})
    layer0 = sorted([row for row in head_rows if row["layer"] == 0], key=lambda row: float(row["drop_pt"]), reverse=True)
    all_heads = sorted(head_rows, key=lambda row: float(row["drop_pt"]), reverse=True)
    return {
        "eval_top_l0_heads": "+".join(f"H{row['head']}" for row in layer0[:5]),
        "eval_top_l0_drop_pt": "+".join(f"{float(row['drop_pt']):.2f}" for row in layer0[:5]),
        "eval_top_all_heads": "+".join(f"L{row['layer']}H{row['head']}" for row in all_heads[:5]),
        "eval_top_all_drop_pt": "+".join(f"{float(row['drop_pt']):.2f}" for row in all_heads[:5]),
        "eval_head_drop_lookup": {(int(row["layer"]), int(row["head"])): float(row["drop_pt"]) for row in head_rows},
    }


def selected_eval_ranks(selected_heads: list[tuple[int, int]], lookup: dict[tuple[int, int], float]) -> str:
    if not selected_heads or not lookup:
        return ""
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for (layer, head), drop in lookup.items():
        by_layer.setdefault(layer, []).append((head, drop))
    rank_lookup: dict[tuple[int, int], int] = {}
    for layer, values in by_layer.items():
        for rank, (head, _drop) in enumerate(sorted(values, key=lambda item: item[1], reverse=True), start=1):
            rank_lookup[(layer, head)] = rank
    items: list[str] = []
    for layer, head in selected_heads:
        rank = rank_lookup.get((layer, head))
        drop = lookup.get((layer, head))
        if rank is None or drop is None:
            items.append(f"L{layer}H{head}:na")
        else:
            items.append(f"L{layer}H{head}:r{rank}/drop{drop:.2f}")
    return "; ".join(items)


def classify_sufficiency(keep_gap_pt: float) -> str:
    if not math.isfinite(keep_gap_pt):
        return "not_measured"
    if keep_gap_pt <= 0.75:
        return "near_full"
    if keep_gap_pt <= 1.75:
        return "moderate_gap"
    return "weak_or_partial"


def classify_necessity(ablate_drop_pt: float) -> str:
    if not math.isfinite(ablate_drop_pt):
        return "not_measured"
    if ablate_drop_pt >= 2.0:
        return "strong"
    if ablate_drop_pt >= 1.0:
        return "moderate"
    return "weak_or_compensated"


def row_for_function(spec: FunctionSpec, baselines: dict[str, dict[str, object]]) -> dict[str, object]:
    if not spec.train_csv.exists():
        raise FileNotFoundError(spec.train_csv)
    baseline = baselines[spec.baseline_key]
    rows = read_csv(spec.train_csv)
    by_mode = {row["train_component_mode"]: row for row in rows}
    keep = by_mode.get("keep")
    ablate = by_mode.get("ablate")
    baseline_final = float(baseline["baseline_final"])
    baseline_zero = float(baseline["baseline_zero"])
    keep_final = row_float(keep, "final_score") if keep else float("nan")
    keep_visible = row_float(keep, "final_visible_only_score") if keep else float("nan")
    keep_zero = row_float(keep, "final_zero_bias_score") if keep else float("nan")
    ablate_final = row_float(ablate, "final_score") if ablate else float("nan")
    ablate_visible = row_float(ablate, "final_visible_only_score") if ablate else float("nan")
    ablate_zero = row_float(ablate, "final_zero_bias_score") if ablate else float("nan")
    keep_gap_pt = 100.0 * (baseline_final - keep_final) if math.isfinite(keep_final) else float("nan")
    ablate_drop_pt = 100.0 * (baseline_final - ablate_final) if math.isfinite(ablate_final) else float("nan")
    ablate_gain_over_zero_pt = 100.0 * (ablate_final - ablate_zero) if math.isfinite(ablate_final) else float("nan")
    selected_heads = parse_spec_heads(keep["train_component_spec"] if keep else ablate["train_component_spec"] if ablate else "")
    lookup = baseline.get("eval_head_drop_lookup", {})
    eval_rank_summary = selected_eval_ranks(selected_heads, lookup if isinstance(lookup, dict) else {})
    return {
        "function_key": spec.key,
        "dataset": baseline["dataset"],
        "seed": baseline["seed"],
        "role": spec.role,
        "label": spec.label,
        "heads": spec.heads_label,
        "selection_basis": spec.selection_basis,
        "baseline_final": fmt(baseline_final),
        "baseline_zero": fmt(baseline_zero),
        "eval_top_l0_heads": baseline.get("eval_top_l0_heads", ""),
        "eval_top_all_heads": baseline.get("eval_top_all_heads", ""),
        "selected_eval_ranks": eval_rank_summary,
        "keep_final": fmt(keep_final),
        "keep_visible": fmt(keep_visible),
        "keep_zero": fmt(keep_zero),
        "keep_gap_pt": fmt(keep_gap_pt, 2),
        "keep_sufficiency": classify_sufficiency(keep_gap_pt),
        "ablate_final": fmt(ablate_final),
        "ablate_visible": fmt(ablate_visible),
        "ablate_zero": fmt(ablate_zero),
        "ablate_drop_pt": fmt(ablate_drop_pt, 2),
        "ablate_gain_over_zero_pt": fmt(ablate_gain_over_zero_pt, 2),
        "ablate_necessity": classify_necessity(ablate_drop_pt),
        "train_csv": str(spec.train_csv),
    }


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["role"]), []).append(row)
    out: list[dict[str, object]] = []
    for role, items in sorted(groups.items()):
        keep_gaps = [float(row["keep_gap_pt"]) for row in items if isinstance(row["keep_gap_pt"], (int, float)) and math.isfinite(float(row["keep_gap_pt"]))]
        ablate_drops = [float(row["ablate_drop_pt"]) for row in items if isinstance(row["ablate_drop_pt"], (int, float)) and math.isfinite(float(row["ablate_drop_pt"]))]
        out.append(
            {
                "role": role,
                "n": len(items),
                "mean_keep_gap_pt": fmt(float(np.mean(keep_gaps)) if keep_gaps else float("nan"), 2),
                "min_keep_gap_pt": fmt(float(np.min(keep_gaps)) if keep_gaps else float("nan"), 2),
                "max_keep_gap_pt": fmt(float(np.max(keep_gaps)) if keep_gaps else float("nan"), 2),
                "mean_ablate_drop_pt": fmt(float(np.mean(ablate_drops)) if ablate_drops else float("nan"), 2),
                "min_ablate_drop_pt": fmt(float(np.min(ablate_drops)) if ablate_drops else float("nan"), 2),
                "max_ablate_drop_pt": fmt(float(np.max(ablate_drops)) if ablate_drops else float("nan"), 2),
            }
        )
    return out


def plot_alignment(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["label"]) for row in rows]
    keep_gap = [float(row["keep_gap_pt"]) if isinstance(row["keep_gap_pt"], (int, float)) and math.isfinite(float(row["keep_gap_pt"])) else np.nan for row in rows]
    ablate_drop = [float(row["ablate_drop_pt"]) if isinstance(row["ablate_drop_pt"], (int, float)) and math.isfinite(float(row["ablate_drop_pt"])) else np.nan for row in rows]
    y = np.arange(len(rows), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11.5, max(5.5, 0.42 * len(rows))), constrained_layout=True)
    ax.barh(y - 0.17, keep_gap, height=0.32, label="baseline - keep final (pt)", color="#4c78a8")
    ax.barh(y + 0.17, ablate_drop, height=0.32, label="baseline - ablate final (pt)", color="#f58518")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axvline(1.0, color="#888", linewidth=0.8, linestyle="--")
    ax.axvline(2.0, color="#555", linewidth=0.8, linestyle=":")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("points")
    ax.set_title("Train-Time Function Alignment: Sufficiency and Necessity")
    ax.grid(True, axis="x", alpha=0.22)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, object]], aggregate: list[dict[str, object]], output_dir: Path) -> None:
    def display_value(value: object, *, digits: int = 4) -> str:
        if not isinstance(value, (int, float)):
            return str(value)
        parsed = float(value)
        if not math.isfinite(parsed):
            return "NA"
        return f"{parsed:.{digits}f}"

    def display_pt(value: object) -> str:
        if not isinstance(value, (int, float)):
            return str(value)
        parsed = float(value)
        if not math.isfinite(parsed):
            return "NA"
        return str(value)

    lines = [
        "# V5 Train-Time Head Function Alignment",
        "",
        "This report joins eval-time head ranking with train-time keep/ablate runs.",
        "The unit of comparison is a function set, not a fixed head id.",
        "",
        "## Function Sets",
        "",
        "| dataset | seed | role | heads | baseline | keep final | keep gap pt | ablate final | ablate drop pt |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        keep_final = row["keep_final"]
        ablate_final = row["ablate_final"]
        lines.append(
            "| "
            + f"{row['dataset']} | {row['seed']} | {row['role']} | {row['heads']} | "
            + f"{float(row['baseline_final']):.4f} | "
            + display_value(keep_final)
            + f" | {display_pt(row['keep_gap_pt'])} | "
            + display_value(ablate_final)
            + f" | {display_pt(row['ablate_drop_pt'])} |"
        )
    lines += [
        "",
        "## Role Aggregate",
        "",
        "| role | n | mean keep gap pt | mean ablate drop pt |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['role']} | {row['n']} | "
            f"{display_pt(row['mean_keep_gap_pt'])} | {display_pt(row['mean_ablate_drop_pt'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "1. MNIST seed-specific L0 top-ranked sets transfer at the function level:",
        "   top head ids change across seeds, but top2/top3 keep runs remain close to each seed's full baseline.",
        "2. Fixed head ids are the wrong invariant; eval-time ranking plus train-time keep/ablate is the stable alignment unit.",
        "3. Rotated-MNIST follows a cascade pattern: L0 top heads are sufficient, then L1H6/L2H3 become conditional compensation heads.",
        "4. Affine-MNIST follows a pool pattern: fixed L1H2/H4 are not necessary, but the broader L1 pool is the last strong local compensation pool.",
        "",
        "## Files",
        "",
        f"- `{(output_dir / 'train_time_function_alignment.csv').as_posix()}`",
        f"- `{(output_dir / 'role_aggregate.csv').as_posix()}`",
        f"- `{(output_dir / 'train_time_function_alignment.png').as_posix()}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines = {spec.key: load_baseline(spec) for spec in BASELINES}
    rows = [row_for_function(spec, baselines) for spec in FUNCTIONS]
    aggregate = aggregate_rows(rows)
    write_csv(output_dir / "train_time_function_alignment.csv", rows)
    write_csv(output_dir / "role_aggregate.csv", aggregate)
    plot_alignment(output_dir / "train_time_function_alignment.png", rows)
    write_report(output_dir / "REPORT.md", rows, aggregate, output_dir)
    summary = {
        "output_dir": str(output_dir),
        "rows": len(rows),
        "function_csv": str(output_dir / "train_time_function_alignment.csv"),
        "role_aggregate_csv": str(output_dir / "role_aggregate.csv"),
        "figure": str(output_dir / "train_time_function_alignment.png"),
        "report": str(output_dir / "REPORT.md"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/v5_train_time_function_alignment")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
