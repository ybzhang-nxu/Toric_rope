from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Iterable


ROOT = Path("results")
OUT = ROOT / "v8_cifar100_mixed_component_mechanism_summary"
BASIS = "mixed_toric_PJ_R0_top220_residual_dct_top32"

MODEL_SPECS = [
    {
        "seed": 426,
        "model": f"{BASIS}_seed426",
        "path": ROOT / "v8_component_eval_cifar100_mixed_residual_dct32_seed426_layer_head_l0123_allheads_10k" / "component_interventions.csv",
        "late_path": ROOT / "v8_component_eval_cifar100_mixed_residual_dct32_seed426_late_compensation_subsets_10k" / "component_interventions.csv",
    },
    {
        "seed": 526,
        "model": f"{BASIS}_seed526",
        "path": ROOT / "v8_component_eval_cifar100_mixed_residual_dct32_seed526_layer_head_l0123_allheads_10k" / "component_interventions.csv",
        "late_path": ROOT / "v8_component_eval_cifar100_mixed_residual_dct32_seed526_late_compensation_subsets_10k" / "component_interventions.csv",
    },
    {
        "seed": 626,
        "model": f"{BASIS}_seed626",
        "path": ROOT / "v8_component_eval_cifar100_mixed_residual_dct32_seed626_layer_head_l0123_allheads_10k" / "component_interventions.csv",
        "late_path": ROOT / "v8_component_eval_cifar100_mixed_residual_dct32_seed626_late_compensation_subsets_10k" / "component_interventions.csv",
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


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def find_row(rows: list[dict[str, str]], mode: str, param: str = "") -> dict[str, str]:
    for row in rows:
        if row["eval_mode"] == mode and row["eval_param"] == param:
            return row
    raise KeyError((mode, param))


def subset_param(layers: Iterable[int]) -> str:
    heads = "+".join(f"H{i}" for i in range(8))
    label = "+".join(f"L{layer}{heads}" for layer in layers)
    return f"{label},r<=4"


def drop(row: dict[str, str]) -> float:
    return -f(row, "delta_vs_full")


def best_row(rows: Iterable[dict[str, str]], key: str) -> dict[str, str]:
    items = list(rows)
    if key == "drop":
        return max(items, key=drop)
    if key == "score":
        return max(items, key=lambda row: f(row, "score"))
    raise ValueError(key)


def summary_row(spec: dict[str, object], section: str, label: str, row: dict[str, str]) -> dict[str, object]:
    delta = f(row, "delta_vs_full")
    return {
        "seed": spec["seed"],
        "model": spec["model"],
        "section": section,
        "label": label,
        "score": f(row, "score"),
        "delta_vs_full": delta,
        "drop_vs_full": -delta,
        "gain_vs_zero": f(row, "gain_vs_zero"),
    }


def selected_summary_rows(spec: dict[str, object], rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for mode, param, label in [
        ("full", "", "full"),
        ("zero_bias", "", "zero_bias"),
        ("radial_truncate", "r<=4", "radial_truncate r<=4"),
        ("radial_truncate", "r<=5", "radial_truncate r<=5"),
        ("head_subset_keep", subset_param([0, 1, 2]), "L0-L2 all-head keep r<=4"),
        ("head_subset_ablate", subset_param([0, 1, 2]), "L0-L2 all-head ablate r<=4"),
        ("head_subset_keep", subset_param([0, 1, 2, 3]), "L0-L3 all-head keep r<=4"),
        ("head_subset_ablate", subset_param([0, 1, 2, 3]), "L0-L3 all-head ablate r<=4"),
    ]:
        out.append(summary_row(spec, "overview", label, find_row(rows, mode, param)))

    for layer in range(6):
        for mode, suffix in [("layer_radial_ablate", "ablate"), ("layer_radial_keep", "keep")]:
            param = f"layer={layer},r<=4"
            out.append(summary_row(spec, f"layer_{suffix}", f"L{layer} {suffix} r<=4", find_row(rows, mode, param)))

    ablate_heads = sorted([r for r in rows if r["eval_mode"] == "head_radial_ablate"], key=drop, reverse=True)
    keep_heads = sorted([r for r in rows if r["eval_mode"] == "head_radial_keep"], key=lambda r: f(r, "score"), reverse=True)
    for rank, row in enumerate(ablate_heads[:8], start=1):
        out.append(summary_row(spec, "top_head_ablate", f"#{rank} {row['eval_param']}", row))
    for rank, row in enumerate(keep_heads[:8], start=1):
        out.append(summary_row(spec, "top_head_keep", f"#{rank} {row['eval_param']}", row))
    return out


def selected_compensation_rows(spec: dict[str, object], rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out = []
    for mode, param, label in [
        ("head_subset_keep", subset_param([4]), "L4 all-head keep r<=4"),
        ("head_subset_ablate", subset_param([4]), "L4 all-head ablate r<=4"),
        ("head_subset_keep", subset_param([5]), "L5 all-head keep r<=4"),
        ("head_subset_ablate", subset_param([5]), "L5 all-head ablate r<=4"),
        ("head_subset_keep", subset_param([4, 5]), "L4-L5 all-head keep r<=4"),
        ("head_subset_ablate", subset_param([4, 5]), "L4-L5 all-head ablate r<=4"),
        ("head_subset_keep", subset_param([3, 4, 5]), "L3-L5 all-head keep r<=4"),
        ("head_subset_ablate", subset_param([3, 4, 5]), "L3-L5 all-head ablate r<=4"),
        ("head_subset_keep", subset_param([0, 1, 2, 3, 4]), "L0-L4 all-head keep r<=4"),
        ("head_subset_ablate", subset_param([0, 1, 2, 3, 4]), "L0-L4 all-head ablate r<=4"),
        ("head_subset_keep", subset_param([0, 1, 2, 3, 4, 5]), "L0-L5 all-head keep r<=4"),
        ("head_subset_ablate", subset_param([0, 1, 2, 3, 4, 5]), "L0-L5 all-head ablate r<=4"),
    ]:
        out.append(summary_row(spec, "late_compensation", label, find_row(rows, mode, param)))
    return out


def key_metrics(spec: dict[str, object], rows: list[dict[str, str]], late_rows: list[dict[str, str]]) -> dict[str, object]:
    full = find_row(rows, "full")
    zero = find_row(rows, "zero_bias")
    radial4 = find_row(rows, "radial_truncate", "r<=4")
    radial5 = find_row(rows, "radial_truncate", "r<=5")
    pool_l012_keep = find_row(rows, "head_subset_keep", subset_param([0, 1, 2]))
    pool_l012_ablate = find_row(rows, "head_subset_ablate", subset_param([0, 1, 2]))
    pool_l0123_keep = find_row(rows, "head_subset_keep", subset_param([0, 1, 2, 3]))
    pool_l0123_ablate = find_row(rows, "head_subset_ablate", subset_param([0, 1, 2, 3]))
    pool_l45_keep = find_row(late_rows, "head_subset_keep", subset_param([4, 5]))
    pool_l45_ablate = find_row(late_rows, "head_subset_ablate", subset_param([4, 5]))
    pool_l345_keep = find_row(late_rows, "head_subset_keep", subset_param([3, 4, 5]))
    pool_l345_ablate = find_row(late_rows, "head_subset_ablate", subset_param([3, 4, 5]))
    pool_l01234_keep = find_row(late_rows, "head_subset_keep", subset_param([0, 1, 2, 3, 4]))
    pool_l01234_ablate = find_row(late_rows, "head_subset_ablate", subset_param([0, 1, 2, 3, 4]))
    pool_l012345_keep = find_row(late_rows, "head_subset_keep", subset_param([0, 1, 2, 3, 4, 5]))
    pool_l012345_ablate = find_row(late_rows, "head_subset_ablate", subset_param([0, 1, 2, 3, 4, 5]))
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
    full_score = f(full, "score")
    zero_score = f(zero, "score")
    return {
        "seed": spec["seed"],
        "model": spec["model"],
        "full": full_score,
        "zero": zero_score,
        "full_minus_zero": full_score - zero_score,
        "radial4": f(radial4, "score"),
        "radial5": f(radial5, "score"),
        "l012_keep": f(pool_l012_keep, "score"),
        "l012_keep_minus_full": f(pool_l012_keep, "score") - full_score,
        "l012_ablate": f(pool_l012_ablate, "score"),
        "l012_ablate_drop": full_score - f(pool_l012_ablate, "score"),
        "l012_ablate_gain_vs_zero": f(pool_l012_ablate, "score") - zero_score,
        "l0123_keep": f(pool_l0123_keep, "score"),
        "l0123_keep_minus_full": f(pool_l0123_keep, "score") - full_score,
        "l0123_ablate": f(pool_l0123_ablate, "score"),
        "l0123_ablate_drop": full_score - f(pool_l0123_ablate, "score"),
        "l0123_ablate_gain_vs_zero": f(pool_l0123_ablate, "score") - zero_score,
        "l45_keep": f(pool_l45_keep, "score"),
        "l45_keep_gain_vs_zero": f(pool_l45_keep, "score") - zero_score,
        "l45_ablate": f(pool_l45_ablate, "score"),
        "l45_ablate_drop": full_score - f(pool_l45_ablate, "score"),
        "l345_keep": f(pool_l345_keep, "score"),
        "l345_keep_gain_vs_zero": f(pool_l345_keep, "score") - zero_score,
        "l345_ablate": f(pool_l345_ablate, "score"),
        "l345_ablate_drop": full_score - f(pool_l345_ablate, "score"),
        "l01234_keep": f(pool_l01234_keep, "score"),
        "l01234_ablate": f(pool_l01234_ablate, "score"),
        "l01234_ablate_drop": full_score - f(pool_l01234_ablate, "score"),
        "l01234_ablate_gain_vs_zero": f(pool_l01234_ablate, "score") - zero_score,
        "l012345_keep": f(pool_l012345_keep, "score"),
        "l012345_ablate": f(pool_l012345_ablate, "score"),
        "l012345_ablate_drop": full_score - f(pool_l012345_ablate, "score"),
        "l012345_ablate_gain_vs_zero": f(pool_l012345_ablate, "score") - zero_score,
        "strongest_layer_ablate": layer_ablate["eval_param"],
        "strongest_layer_ablate_drop": drop(layer_ablate),
        "best_layer_keep": layer_keep["eval_param"],
        "best_layer_keep_score": f(layer_keep, "score"),
        "strongest_head_ablate": head_ablate["eval_param"],
        "strongest_head_ablate_drop": drop(head_ablate),
        "best_head_keep": head_keep["eval_param"],
        "best_head_keep_score": f(head_keep, "score"),
        "best_head_keep_gain_vs_zero": f(head_keep, "gain_vs_zero"),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric = [
        "full",
        "zero",
        "full_minus_zero",
        "radial4",
        "radial5",
        "l012_keep",
        "l012_keep_minus_full",
        "l012_ablate",
        "l012_ablate_drop",
        "l012_ablate_gain_vs_zero",
        "l0123_keep",
        "l0123_keep_minus_full",
        "l0123_ablate",
        "l0123_ablate_drop",
        "l0123_ablate_gain_vs_zero",
        "l45_keep",
        "l45_keep_gain_vs_zero",
        "l45_ablate",
        "l45_ablate_drop",
        "l345_keep",
        "l345_keep_gain_vs_zero",
        "l345_ablate",
        "l345_ablate_drop",
        "l01234_keep",
        "l01234_ablate",
        "l01234_ablate_drop",
        "l01234_ablate_gain_vs_zero",
        "l012345_keep",
        "l012345_ablate",
        "l012345_ablate_drop",
        "l012345_ablate_gain_vs_zero",
        "strongest_layer_ablate_drop",
        "best_layer_keep_score",
        "strongest_head_ablate_drop",
        "best_head_keep_score",
        "best_head_keep_gain_vs_zero",
    ]
    out: dict[str, object] = {"basis": BASIS, "n": len(rows), "seeds": ",".join(str(r["seed"]) for r in rows)}
    for key in numeric:
        vals = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = mean(vals)
        out[f"{key}_std"] = std(vals)
    return out


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table(rows: list[dict[str, object]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def compact_seed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "seed": row["seed"],
            "full": row["full"],
            "zero": row["zero"],
            "radial4": row["radial4"],
            "l012_keep": row["l012_keep"],
            "l012_ablate": row["l012_ablate"],
            "l0123_keep": row["l0123_keep"],
            "l0123_ablate": row["l0123_ablate"],
            "l45_keep": row["l45_keep"],
            "l012345_ablate": row["l012345_ablate"],
            "top_head_ablate": row["strongest_head_ablate"],
            "top_head_drop": row["strongest_head_ablate_drop"],
            "best_head_keep": row["best_head_keep"],
            "best_head_keep_score": row["best_head_keep_score"],
        }
        for row in rows
    ]


def write_report(seed_rows: list[dict[str, object]], family: dict[str, object]) -> None:
    family_rows = [
        {
            "basis": BASIS,
            "n": family["n"],
            "full": f"{float(family['full_mean']):.4f} +/- {float(family['full_std']):.4f}",
            "zero": f"{float(family['zero_mean']):.4f} +/- {float(family['zero_std']):.4f}",
            "radial4": f"{float(family['radial4_mean']):.4f} +/- {float(family['radial4_std']):.4f}",
            "l012_keep": f"{float(family['l012_keep_mean']):.4f} +/- {float(family['l012_keep_std']):.4f}",
            "l012_ablate": f"{float(family['l012_ablate_mean']):.4f} +/- {float(family['l012_ablate_std']):.4f}",
            "l0123_keep": f"{float(family['l0123_keep_mean']):.4f} +/- {float(family['l0123_keep_std']):.4f}",
            "l0123_ablate": f"{float(family['l0123_ablate_mean']):.4f} +/- {float(family['l0123_ablate_std']):.4f}",
            "l45_keep": f"{float(family['l45_keep_mean']):.4f} +/- {float(family['l45_keep_std']):.4f}",
            "l012345_ablate": f"{float(family['l012345_ablate_mean']):.4f} +/- {float(family['l012345_ablate_std']):.4f}",
            "top_head_drop": f"{float(family['strongest_head_ablate_drop_mean']):.4f} +/- {float(family['strongest_head_ablate_drop_std']):.4f}",
        }
    ]
    text = "\n".join(
        [
            "# CIFAR100 Mixed Residual-DCT Component Mechanism Summary",
            "",
            "Inputs: mixed residual-DCT seeds 426,526,626.",
            "",
            "## Family Key Metrics",
            "",
            table(
                family_rows,
                [
                    "basis",
                    "n",
                    "full",
                    "zero",
                    "radial4",
                    "l012_keep",
                    "l012_ablate",
                    "l0123_keep",
                    "l0123_ablate",
                    "l45_keep",
                    "l012345_ablate",
                    "top_head_drop",
                ],
            ),
            "",
            "## Seed Key Metrics",
            "",
            table(
                compact_seed_rows(seed_rows),
                [
                    "seed",
                    "full",
                    "zero",
                    "radial4",
                    "l012_keep",
                    "l012_ablate",
                    "l0123_keep",
                    "l0123_ablate",
                    "l45_keep",
                    "l012345_ablate",
                    "top_head_ablate",
                    "top_head_drop",
                    "best_head_keep",
                    "best_head_keep_score",
                ],
            ),
            "",
            "## Reading",
            "",
            (
                "- L0-L2 all-head keep is sufficient: "
                f"{float(family['l012_keep_mean']):.4f} +/- {float(family['l012_keep_std']):.4f}, "
                f"near full {float(family['full_mean']):.4f} +/- {float(family['full_std']):.4f}."
            ),
            (
                "- L0-L3 all-head ablate is strongly necessary but not zero-exhaustive: "
                f"{float(family['l0123_ablate_mean']):.4f} +/- {float(family['l0123_ablate_std']):.4f}, "
                f"zero is {float(family['zero_mean']):.4f} +/- {float(family['zero_std']):.4f}."
            ),
            (
                "- The residual is mostly late local compensation: L4-L5 keep is "
                f"{float(family['l45_keep_mean']):.4f} +/- {float(family['l45_keep_std']):.4f}, "
                "while L0-L5 all-head ablate nearly exhausts to zero: "
                f"{float(family['l012345_ablate_mean']):.4f} +/- {float(family['l012345_ablate_std']):.4f}."
            ),
            (
                "- CIFAR100 mixed residual-DCT keeps the Toric/PJ-style early local pool, while leaving a larger residual compensation pool than CIFAR10 mixed."
            ),
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
        late_rows = read_rows(Path(spec["late_path"]))
        summary_rows.extend(selected_summary_rows(spec, rows))
        summary_rows.extend(selected_compensation_rows(spec, late_rows))
        seed_rows.append(key_metrics(spec, rows, late_rows))
    family = aggregate(seed_rows)

    summary_csv = OUT / "component_mechanism_summary.csv"
    seed_csv = OUT / "component_mechanism_seed_key_metrics.csv"
    family_csv = OUT / "component_mechanism_family_key_metrics.csv"
    write_rows(summary_csv, summary_rows, list(summary_rows[0]))
    write_rows(seed_csv, seed_rows, list(seed_rows[0]))
    write_rows(family_csv, [family], list(family))
    write_report(seed_rows, family)
    summary = {
        "summary_csv": str(summary_csv),
        "seed_key_metrics_csv": str(seed_csv),
        "family_key_metrics_csv": str(family_csv),
        "readme": str(OUT / "README.md"),
        "models": [spec["model"] for spec in MODEL_SPECS],
        "summary_rows": len(summary_rows),
        "seed_rows": len(seed_rows),
        "family_rows": 1,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
