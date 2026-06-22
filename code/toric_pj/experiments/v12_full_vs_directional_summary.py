from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "" or value.lower() == "nan":
        return float("nan")
    return float(value)


def _fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def _fmt_sci(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.2e}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pair_rows(
    rows: Iterable[dict[str, str]],
    *,
    protocol: str,
    same_budget: bool,
) -> list[dict[str, object]]:
    selected = [
        r
        for r in rows
        if r["protocol"] == protocol
        and r["basis_family"] in {"full", "directional"}
        and r["order"] in {"1", "2"}
    ]
    out: list[dict[str, object]] = []
    if same_budget:
        key_fields = ["frequency_source", "fit_target", "order", "atom_budget"]
    else:
        key_fields = ["frequency_source", "fit_target", "order"]

    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
    for row in selected:
        key = tuple(row[f] for f in key_fields)
        grouped.setdefault(key, {})[row["basis_family"]] = row

    for key in sorted(grouped):
        pair = grouped[key]
        if "full" not in pair or "directional" not in pair:
            continue
        full = pair["full"]
        directional = pair["directional"]
        full_target = _float(full, "target_fit_r2_mean")
        dir_target = _float(directional, "target_fit_r2_mean")
        full_table = _float(full, "table_fit_r2_mean")
        dir_table = _float(directional, "table_fit_r2_mean")
        full_cond = _float(full, "condition_number_mean")
        dir_cond = _float(directional, "condition_number_mean")
        full_coeff = _float(full, "coeff_norm_mean")
        dir_coeff = _float(directional, "coeff_norm_mean")
        out.append(
            {
                "protocol": protocol,
                "comparison": "matched_atom" if same_budget else "nested_same_centers",
                "frequency_source": full["frequency_source"],
                "fit_target": full["fit_target"],
                "order": int(full["order"]),
                "directional_atom_budget": int(float(directional["atom_budget"])),
                "full_atom_budget": int(float(full["atom_budget"])),
                "directional_num_features": _float(directional, "num_features_mean"),
                "full_num_features": _float(full, "num_features_mean"),
                "directional_target_fit_r2": dir_target,
                "full_target_fit_r2": full_target,
                "delta_target_fit_r2_full_minus_directional": full_target - dir_target,
                "directional_table_fit_r2": dir_table,
                "full_table_fit_r2": full_table,
                "delta_table_fit_r2_full_minus_directional": full_table - dir_table,
                "directional_condition_number": dir_cond,
                "full_condition_number": full_cond,
                "condition_ratio_full_over_directional": full_cond / dir_cond if dir_cond else float("nan"),
                "directional_coeff_norm": dir_coeff,
                "full_coeff_norm": full_coeff,
                "coeff_norm_ratio_full_over_directional": full_coeff / dir_coeff if dir_coeff else float("nan"),
            }
        )
    return out


def _load_downstream(j0_path: Path, j1j2_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in (j0_path, j1j2_path):
        if path.exists():
            rows.extend(_read_csv(path))
    keep = []
    names = {
        "v12_matched_axis_residual_toric_full_J0_atoms108",
        "v12_matched_axis_residual_toric_full_J1_atoms108",
        "v12_matched_axis_residual_toric_full_J2_atoms108",
    }
    for row in rows:
        if row.get("basis") in names:
            keep.append(row)
    order = {
        "v12_matched_axis_residual_toric_full_J0_atoms108": 0,
        "v12_matched_axis_residual_toric_full_J1_atoms108": 1,
        "v12_matched_axis_residual_toric_full_J2_atoms108": 2,
    }
    return sorted(keep, key=lambda r: order.get(r["basis"], 99))


def _short_basis_name(name: str) -> str:
    if "_J0_" in name:
        return "full multivariate J0"
    if "_J1_" in name:
        return "full multivariate J1"
    if "_J2_" in name:
        return "full multivariate J2"
    return name


def _table(rows: list[dict[str, object]], *, target: str, budget: int) -> str:
    subset = [
        r
        for r in rows
        if r["fit_target"] == target
        and r["directional_atom_budget"] == budget
        and r["full_atom_budget"] == budget
    ]
    lines = [
        "| source | order | dir R2 | full R2 | delta | dir cond | full cond | cond ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in subset:
        lines.append(
            "| {source} | J{order} | {dir_r2} | {full_r2} | {delta} | {dir_cond} | {full_cond} | {ratio} |".format(
                source=row["frequency_source"],
                order=row["order"],
                dir_r2=_fmt(float(row["directional_target_fit_r2"])),
                full_r2=_fmt(float(row["full_target_fit_r2"])),
                delta=_fmt(float(row["delta_target_fit_r2_full_minus_directional"])),
                dir_cond=_fmt_sci(float(row["directional_condition_number"])),
                full_cond=_fmt_sci(float(row["full_condition_number"])),
                ratio=_fmt_sci(float(row["condition_ratio_full_over_directional"])),
            )
        )
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    source_projection: Path,
    matched_rows: list[dict[str, object]],
    nested_rows: list[dict[str, object]],
    downstream_rows: list[dict[str, str]],
) -> None:
    matched_deltas = [float(r["delta_target_fit_r2_full_minus_directional"]) for r in matched_rows]
    positive = sum(1 for v in matched_deltas if v > 0)
    mean_delta = sum(matched_deltas) / len(matched_deltas)
    lines: list[str] = []
    lines.append("# V12 E10 Full Multivariate Versus Directional Jets")
    lines.append("")
    lines.append("Date: 2026-06-21")
    lines.append("")
    lines.append("Source projection artifact:")
    lines.append("")
    lines.append("```text")
    lines.append(str(source_projection))
    lines.append("```")
    lines.append("")
    lines.append("This audit extracts the full-vs-directional comparison already present in the E1 teacher jet-order projection run. No model is retrained.")
    lines.append("")
    lines.append("## Main Matched-Atom Result")
    lines.append("")
    lines.append(f"Across matched-atom J1/J2 rows, full multivariate jets improve target-fit R2 over directional jets in {positive}/{len(matched_rows)} comparisons. The mean delta is {_fmt(mean_delta)}.")
    lines.append("")
    lines.append("Full-table target, atom budget 108:")
    lines.append("")
    lines.append(_table(matched_rows, target="full_table", budget=108))
    lines.append("")
    lines.append("Axis-plus-residual target, atom budget 108:")
    lines.append("")
    lines.append(_table(matched_rows, target="axis_plus_residual", budget=108))
    lines.append("")
    lines.append("Full-table target, atom budget 54:")
    lines.append("")
    lines.append(_table(matched_rows, target="full_table", budget=54))
    lines.append("")
    lines.append("Axis-plus-residual target, atom budget 54:")
    lines.append("")
    lines.append(_table(matched_rows, target="axis_plus_residual", budget=54))
    lines.append("")
    lines.append("## Nested Same-Center Note")
    lines.append("")
    lines.append("The nested rows compare the same center schedule but not the same atom budget: directional nested rows use 109 nominal atoms and full nested rows use 73. They are useful for conditioning diagnostics, but the matched-atom table above is the fairer E10 comparison.")
    lines.append("")
    nested_deltas = [float(r["delta_target_fit_r2_full_minus_directional"]) for r in nested_rows]
    tol = 1e-4
    nested_positive = sum(1 for v in nested_deltas if v > tol)
    nested_tied = sum(1 for v in nested_deltas if abs(v) <= tol)
    nested_mean = sum(nested_deltas) / len(nested_deltas)
    lines.append(
        f"In nested same-center rows, full multivariate jets improve target-fit R2 in {nested_positive}/{len(nested_rows)} comparisons and tie within {tol:g} in {nested_tied}/{len(nested_rows)} comparisons; the mean delta is {_fmt(nested_mean)}."
    )
    lines.append("")
    if downstream_rows:
        lines.append("## Downstream Boundary")
        lines.append("")
        lines.append("The projection audit should be read together with the matched downstream confirmation. Full multivariate J1/J2 project better than directional J1/J2, but the natural CIFAR10 downstream task still prefers the matched J0 row:")
        lines.append("")
        lines.append("| basis | n | score mean | score std | final mean | teacher-init R2 note |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for row in downstream_rows:
            note = ""
            if row["basis"].endswith("J0_atoms108"):
                note = "0.8397 from E1 confirm3"
            elif row["basis"].endswith("J1_atoms108"):
                note = "0.8155 from J1/J2 confirm3"
            elif row["basis"].endswith("J2_atoms108"):
                note = "0.7991 from J1/J2 confirm3"
            lines.append(
                "| {basis} | {n} | {score} | {std} | {final} | {note} |".format(
                    basis=_short_basis_name(row["basis"]),
                    n=row["n"],
                    score=_fmt(float(row["score_mean"])),
                    std=_fmt(float(row["score_std"])),
                    final=_fmt(float(row["final_score_mean"])),
                    note=note,
                )
            )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Full multivariate jets are a better projection family than directional jets for the learned teacher table under matched atom budgets.")
    lines.append("- This supports retaining the full multivariate PJ construction as a theory/diagnostic object.")
    lines.append("- It does not create a monotonic downstream jet-order claim: the matched downstream confirmation still shows J0 > J1 > J2 for CIFAR10 reconstruction.")
    lines.append("- The paper should separate projection expressivity from downstream utility.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("```text")
    lines.append("matched_atom_full_vs_directional.csv")
    lines.append("nested_same_center_full_vs_directional.csv")
    lines.append("REPORT.md")
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V12 E10 full-vs-directional jet projection comparisons.")
    parser.add_argument(
        "--projection-aggregate",
        type=Path,
        default=Path("MetricToric/results/v12_e1_teacher_jetorder_projection_cifar10_10k/projection_aggregate.csv"),
    )
    parser.add_argument(
        "--j0-downstream-aggregate",
        type=Path,
        default=Path("MetricToric/results/v12_e1_downstream_confirm3_cifar10_10k/student_aggregate.csv"),
    )
    parser.add_argument(
        "--j1j2-downstream-aggregate",
        type=Path,
        default=Path("MetricToric/results/v12_e1_j1j2_downstream_confirm3_cifar10_10k/student_aggregate.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MetricToric/results/v12_e10_full_vs_directional_projection"),
    )
    args = parser.parse_args()

    rows = _read_csv(args.projection_aggregate)
    matched = _pair_rows(rows, protocol="matched", same_budget=True)
    nested = _pair_rows(rows, protocol="nested", same_budget=False)
    downstream = _load_downstream(args.j0_downstream_aggregate, args.j1j2_downstream_aggregate)

    fields = [
        "protocol",
        "comparison",
        "frequency_source",
        "fit_target",
        "order",
        "directional_atom_budget",
        "full_atom_budget",
        "directional_num_features",
        "full_num_features",
        "directional_target_fit_r2",
        "full_target_fit_r2",
        "delta_target_fit_r2_full_minus_directional",
        "directional_table_fit_r2",
        "full_table_fit_r2",
        "delta_table_fit_r2_full_minus_directional",
        "directional_condition_number",
        "full_condition_number",
        "condition_ratio_full_over_directional",
        "directional_coeff_norm",
        "full_coeff_norm",
        "coeff_norm_ratio_full_over_directional",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "matched_atom_full_vs_directional.csv", matched, fields)
    _write_csv(args.output_dir / "nested_same_center_full_vs_directional.csv", nested, fields)
    _write_report(
        args.output_dir / "REPORT.md",
        source_projection=args.projection_aggregate,
        matched_rows=matched,
        nested_rows=nested,
        downstream_rows=downstream,
    )
    print(
        {
            "output_dir": str(args.output_dir),
            "matched_rows": len(matched),
            "nested_rows": len(nested),
            "report": str(args.output_dir / "REPORT.md"),
        }
    )


if __name__ == "__main__":
    main()
