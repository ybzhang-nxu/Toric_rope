from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Iterable


ROOT = Path("results")
OUT = ROOT / "v7_svhn_component_mechanism_summary"

MODEL_SPECS = [
    {
        "family": "pure_toric_pj_top220",
        "seed": 443,
        "model": "pure_toric_pj_top220_seed443",
        "path": ROOT / "v7_component_eval_svhn_top220_seed443_layer_head_l012_10k" / "component_interventions.csv",
    },
    {
        "family": "pure_toric_pj_top220",
        "seed": 543,
        "model": "pure_toric_pj_top220_seed543",
        "path": ROOT / "v7_component_eval_svhn_top220_seed543_layer_head_l012_10k" / "component_interventions.csv",
    },
    {
        "family": "pure_toric_pj_top220",
        "seed": 643,
        "model": "pure_toric_pj_top220_seed643",
        "path": ROOT / "v7_component_eval_svhn_top220_seed643_layer_head_l012_10k" / "component_interventions.csv",
    },
    {
        "family": "mixed_residual_dct32",
        "seed": 460,
        "model": "mixed_residual_dct32_seed460",
        "path": ROOT / "v7_component_eval_svhn_mixed_residual_dct32_seed460_layer_head_l012_10k" / "component_interventions.csv",
    },
    {
        "family": "mixed_residual_dct32",
        "seed": 560,
        "model": "mixed_residual_dct32_seed560",
        "path": ROOT / "v7_component_eval_svhn_mixed_residual_dct32_seed560_layer_head_l012_10k" / "component_interventions.csv",
    },
    {
        "family": "mixed_residual_dct32",
        "seed": 660,
        "model": "mixed_residual_dct32_seed660",
        "path": ROOT / "v7_component_eval_svhn_mixed_residual_dct32_seed660_layer_head_l012_10k" / "component_interventions.csv",
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def find_row(rows: list[dict[str, str]], mode: str, param: str = "") -> dict[str, str]:
    for row in rows:
        if row["eval_mode"] == mode and row["eval_param"] == param:
            return row
    raise KeyError((mode, param))


def early_pool_param() -> str:
    heads = "+".join(f"H{i}" for i in range(8))
    return f"L0{heads}+L1{heads}+L2{heads},r<=4"


def drop_vs_full(row: dict[str, str]) -> float:
    return -as_float(row, "delta_vs_full")


def add_summary_row(
    out: list[dict[str, object]],
    *,
    spec: dict[str, object],
    section: str,
    label: str,
    row: dict[str, str],
    note: str = "",
) -> None:
    score = as_float(row, "score")
    delta = as_float(row, "delta_vs_full")
    gain = as_float(row, "gain_vs_zero")
    out.append(
        {
            "family": spec["family"],
            "seed": spec["seed"],
            "model": spec["model"],
            "section": section,
            "label": label,
            "score": score,
            "delta_vs_full": delta,
            "drop_vs_full": -delta,
            "gain_vs_zero": gain,
            "note": note,
        }
    )


def model_summary(spec: dict[str, object], rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for mode, param, label, note in [
        ("full", "", "full", ""),
        ("zero_bias", "", "zero_bias", ""),
        ("radial_truncate", "r<=4", "radial_truncate r<=4", "bounded/local score"),
        ("head_subset_keep", early_pool_param(), "L0-L2 all-head keep r<=4", "early local pool sufficiency"),
        ("head_subset_ablate", early_pool_param(), "L0-L2 all-head ablate r<=4", "early local pool necessity"),
    ]:
        add_summary_row(out, spec=spec, section="overview", label=label, row=find_row(rows, mode, param), note=note)

    for layer in range(6):
        for mode, suffix in [("layer_radial_ablate", "ablate"), ("layer_radial_keep", "keep")]:
            param = f"layer={layer},r<=4"
            add_summary_row(
                out,
                spec=spec,
                section=f"layer_{suffix}",
                label=f"L{layer} {suffix} r<=4",
                row=find_row(rows, mode, param),
            )

    ablate_rows = [r for r in rows if r["eval_mode"] == "head_radial_ablate"]
    keep_rows = [r for r in rows if r["eval_mode"] == "head_radial_keep"]
    ablate_rows = sorted(ablate_rows, key=drop_vs_full, reverse=True)
    keep_rows = sorted(keep_rows, key=lambda r: as_float(r, "score"), reverse=True)
    for rank, row in enumerate(ablate_rows[:8], start=1):
        add_summary_row(
            out,
            spec=spec,
            section="top_head_ablate",
            label=f"#{rank} {row['eval_param']}",
            row=row,
            note="ranked by drop_vs_full",
        )
    for rank, row in enumerate(keep_rows[:8], start=1):
        add_summary_row(
            out,
            spec=spec,
            section="top_head_keep",
            label=f"#{rank} {row['eval_param']}",
            row=row,
            note="ranked by score",
        )
    return out


def best_row(rows: Iterable[dict[str, str]], key: str) -> dict[str, str]:
    items = list(rows)
    if key == "drop":
        return max(items, key=drop_vs_full)
    if key == "score":
        return max(items, key=lambda r: as_float(r, "score"))
    raise ValueError(key)


def key_metrics(spec: dict[str, object], rows: list[dict[str, str]]) -> dict[str, object]:
    full = find_row(rows, "full")
    zero = find_row(rows, "zero_bias")
    radial4 = find_row(rows, "radial_truncate", "r<=4")
    early_keep = find_row(rows, "head_subset_keep", early_pool_param())
    early_ablate = find_row(rows, "head_subset_ablate", early_pool_param())
    layer_ablate = best_row(
        (r for r in rows if r["eval_mode"] == "layer_radial_ablate" and r["eval_param"].endswith("r<=4")),
        "drop",
    )
    layer_keep = best_row(
        (r for r in rows if r["eval_mode"] == "layer_radial_keep" and r["eval_param"].endswith("r<=4")),
        "score",
    )
    head_ablate = best_row((r for r in rows if r["eval_mode"] == "head_radial_ablate"), "drop")
    head_keep = best_row((r for r in rows if r["eval_mode"] == "head_radial_keep"), "score")
    full_score = as_float(full, "score")
    zero_score = as_float(zero, "score")
    early_keep_score = as_float(early_keep, "score")
    early_ablate_score = as_float(early_ablate, "score")
    radial4_score = as_float(radial4, "score")
    return {
        "family": spec["family"],
        "seed": spec["seed"],
        "model": spec["model"],
        "full": full_score,
        "zero": zero_score,
        "full_minus_zero": full_score - zero_score,
        "radial4": radial4_score,
        "radial4_minus_full": radial4_score - full_score,
        "early_keep": early_keep_score,
        "early_keep_minus_full": early_keep_score - full_score,
        "early_ablate": early_ablate_score,
        "early_ablate_drop": full_score - early_ablate_score,
        "early_ablate_gain_vs_zero": early_ablate_score - zero_score,
        "strongest_layer_ablate": layer_ablate["eval_param"],
        "strongest_layer_ablate_drop": drop_vs_full(layer_ablate),
        "best_layer_keep": layer_keep["eval_param"],
        "best_layer_keep_score": as_float(layer_keep, "score"),
        "strongest_head_ablate": head_ablate["eval_param"],
        "strongest_head_ablate_drop": drop_vs_full(head_ablate),
        "best_head_keep": head_keep["eval_param"],
        "best_head_keep_score": as_float(head_keep, "score"),
        "best_head_keep_gain_vs_zero": as_float(head_keep, "gain_vs_zero"),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def aggregate_key_metrics(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_cols = [
        "full",
        "zero",
        "full_minus_zero",
        "radial4",
        "radial4_minus_full",
        "early_keep",
        "early_keep_minus_full",
        "early_ablate",
        "early_ablate_drop",
        "early_ablate_gain_vs_zero",
        "strongest_layer_ablate_drop",
        "best_layer_keep_score",
        "strongest_head_ablate_drop",
        "best_head_keep_score",
        "best_head_keep_gain_vs_zero",
    ]
    out: list[dict[str, object]] = []
    for family in sorted({str(r["family"]) for r in seed_rows}):
        group = [r for r in seed_rows if r["family"] == family]
        row: dict[str, object] = {"family": family, "n": len(group), "seeds": ",".join(str(r["seed"]) for r in group)}
        for col in numeric_cols:
            vals = [float(r[col]) for r in group]
            row[f"{col}_mean"] = mean(vals)
            row[f"{col}_std"] = std(vals)
        out.append(row)
    return out


def aggregate_comparable_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    comparable = {"overview", "layer_ablate", "layer_keep"}
    keys = sorted(
        {
            (str(r["family"]), str(r["section"]), str(r["label"]))
            for r in rows
            if str(r["section"]) in comparable
        }
    )
    for family, section, label in keys:
        group = [r for r in rows if r["family"] == family and r["section"] == section and r["label"] == label]
        row: dict[str, object] = {
            "family": family,
            "section": section,
            "label": label,
            "n": len(group),
            "seeds": ",".join(str(r["seed"]) for r in group),
        }
        for col in ["score", "delta_vs_full", "drop_vs_full", "gain_vs_zero"]:
            vals = [float(r[col]) for r in group]
            row[f"{col}_mean"] = mean(vals)
            row[f"{col}_std"] = std(vals)
        out.append(row)
    return out


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def report_key_family_rows(family_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "family": row["family"],
            "n": row["n"],
            "full": f"{float(row['full_mean']):.4f} +/- {float(row['full_std']):.4f}",
            "zero": f"{float(row['zero_mean']):.4f} +/- {float(row['zero_std']):.4f}",
            "radial4": f"{float(row['radial4_mean']):.4f} +/- {float(row['radial4_std']):.4f}",
            "early_keep": f"{float(row['early_keep_mean']):.4f} +/- {float(row['early_keep_std']):.4f}",
            "early_ablate": f"{float(row['early_ablate_mean']):.4f} +/- {float(row['early_ablate_std']):.4f}",
            "early_ablate_drop": f"{float(row['early_ablate_drop_mean']):.4f} +/- {float(row['early_ablate_drop_std']):.4f}",
            "top_head_drop": f"{float(row['strongest_head_ablate_drop_mean']):.4f} +/- {float(row['strongest_head_ablate_drop_std']):.4f}",
            "best_head_keep": f"{float(row['best_head_keep_score_mean']):.4f} +/- {float(row['best_head_keep_score_std']):.4f}",
        }
        for row in family_rows
    ]


def compact_seed_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "family": row["family"],
            "seed": row["seed"],
            "full": row["full"],
            "zero": row["zero"],
            "radial4": row["radial4"],
            "early_keep": row["early_keep"],
            "early_ablate": row["early_ablate"],
            "top_head_ablate": row["strongest_head_ablate"],
            "top_head_drop": row["strongest_head_ablate_drop"],
            "best_head_keep": row["best_head_keep"],
            "best_head_keep_score": row["best_head_keep_score"],
        }
        for row in seed_rows
    ]


def row_by_family(family_rows: list[dict[str, object]], family: str) -> dict[str, object]:
    for row in family_rows:
        if row["family"] == family:
            return row
    raise KeyError(family)


def write_report(
    *,
    summary_rows: list[dict[str, object]],
    seed_rows: list[dict[str, object]],
    family_rows: list[dict[str, object]],
    aggregate_rows: list[dict[str, object]],
) -> None:
    pure = row_by_family(family_rows, "pure_toric_pj_top220")
    mixed = row_by_family(family_rows, "mixed_residual_dct32")
    overview_rows = [
        row
        for row in aggregate_rows
        if row["section"] == "overview"
        and row["label"]
        in {
            "full",
            "zero_bias",
            "radial_truncate r<=4",
            "L0-L2 all-head keep r<=4",
            "L0-L2 all-head ablate r<=4",
        }
    ]
    text = "\n".join(
        [
            "# SVHN V7 Component Mechanism Summary",
            "",
            "Inputs:",
            "",
            "- pure_toric_pj_top220 seeds: 443,543,643",
            "- mixed_residual_dct32 seeds: 460,560,660",
            "",
            "## Family Key Metrics",
            "",
            markdown_table(
                report_key_family_rows(family_rows),
                [
                    "family",
                    "n",
                    "full",
                    "zero",
                    "radial4",
                    "early_keep",
                    "early_ablate",
                    "early_ablate_drop",
                    "top_head_drop",
                    "best_head_keep",
                ],
            ),
            "",
            "## Seed Key Metrics",
            "",
            markdown_table(
                compact_seed_rows(seed_rows),
                [
                    "family",
                    "seed",
                    "full",
                    "zero",
                    "radial4",
                    "early_keep",
                    "early_ablate",
                    "top_head_ablate",
                    "top_head_drop",
                    "best_head_keep",
                    "best_head_keep_score",
                ],
            ),
            "",
            "## Comparable Overview Aggregate",
            "",
            markdown_table(
                overview_rows,
                ["family", "label", "score_mean", "score_std", "delta_vs_full_mean", "gain_vs_zero_mean"],
            ),
            "",
            "## Reading",
            "",
            (
                "- Pure Toric/PJ keeps a high local ceiling across seeds: "
                f"radial r<=4 = {float(pure['radial4_mean']):.4f} +/- {float(pure['radial4_std']):.4f}, "
                f"L0-L2 keep = {float(pure['early_keep_mean']):.4f} +/- {float(pure['early_keep_std']):.4f}; "
                f"but full is only {float(pure['full_mean']):.4f} +/- {float(pure['full_std']):.4f}."
            ),
            (
                "- Mixed residual-DCT keeps the same local ceiling while preserving full evaluation: "
                f"full = {float(mixed['full_mean']):.4f} +/- {float(mixed['full_std']):.4f}, "
                f"radial r<=4 = {float(mixed['radial4_mean']):.4f} +/- {float(mixed['radial4_std']):.4f}."
            ),
            (
                "- Early L0-L2 all-head ablation causes large drops in both families, "
                f"pure drop = {float(pure['early_ablate_drop_mean']):.4f} +/- {float(pure['early_ablate_drop_std']):.4f}, "
                f"mixed drop = {float(mixed['early_ablate_drop_mean']):.4f} +/- {float(mixed['early_ablate_drop_std']):.4f}."
            ),
            "- The cross-seed result supports the mechanism claim: Toric/PJ supplies a strong early local geometry pool, while residual-DCT stabilizes full heldout boundary behavior.",
            "",
        ]
    )
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        rows = read_rows(Path(spec["path"]))
        summary_rows.extend(model_summary(spec, rows))
        seed_rows.append(key_metrics(spec, rows))

    family_rows = aggregate_key_metrics(seed_rows)
    aggregate_rows = aggregate_comparable_rows(summary_rows)

    summary_fields = [
        "family",
        "seed",
        "model",
        "section",
        "label",
        "score",
        "delta_vs_full",
        "drop_vs_full",
        "gain_vs_zero",
        "note",
    ]
    seed_fields = list(seed_rows[0])
    family_fields = list(family_rows[0])
    aggregate_fields = list(aggregate_rows[0])

    summary_csv = OUT / "component_mechanism_summary.csv"
    seed_csv = OUT / "component_mechanism_seed_key_metrics.csv"
    family_csv = OUT / "component_mechanism_family_key_metrics.csv"
    aggregate_csv = OUT / "component_mechanism_aggregate.csv"
    write_rows(summary_csv, summary_rows, summary_fields)
    write_rows(seed_csv, seed_rows, seed_fields)
    write_rows(family_csv, family_rows, family_fields)
    write_rows(aggregate_csv, aggregate_rows, aggregate_fields)
    write_report(
        summary_rows=summary_rows,
        seed_rows=seed_rows,
        family_rows=family_rows,
        aggregate_rows=aggregate_rows,
    )
    summary = {
        "summary_csv": str(summary_csv),
        "seed_key_metrics_csv": str(seed_csv),
        "family_key_metrics_csv": str(family_csv),
        "aggregate_csv": str(aggregate_csv),
        "readme": str(OUT / "README.md"),
        "models": [spec["model"] for spec in MODEL_SPECS],
        "summary_rows": len(summary_rows),
        "seed_rows": len(seed_rows),
        "family_rows": len(family_rows),
        "aggregate_rows": len(aggregate_rows),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
