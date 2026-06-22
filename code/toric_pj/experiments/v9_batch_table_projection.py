from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace

from toric_pj.experiments.v4_table_projection import run as run_single_projection


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
    else:
        path = spec
        label = Path(path).parent.name or Path(path).stem
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("_")
    if not label:
        label = "input"
    return label, Path(path)


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tagged_rows(path: Path, *, label: str, input_path: Path, source_output_dir: Path) -> list[dict[str, object]]:
    rows = read_csv_rows(path)
    for row in rows:
        row["source_label"] = label
        row["source_input"] = str(input_path)
        row["source_output_dir"] = str(source_output_dir)
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    used_labels: set[str] = set()
    inputs: list[tuple[str, Path]] = []
    for idx, spec in enumerate(args.input):
        label, input_path = parse_input_spec(spec)
        base_label = label
        suffix = 2
        while label in used_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        used_labels.add(label)
        inputs.append((label, input_path))

    summaries: list[dict[str, object]] = []
    combined_projection_rows: list[dict[str, object]] = []
    combined_residual_rows: list[dict[str, object]] = []
    combined_random_rows: list[dict[str, object]] = []
    combined_aggregate_rows: list[dict[str, object]] = []

    for label, input_path in inputs:
        source_output_dir = output_dir / "per_input" / label
        aggregate_path = source_output_dir / "projection_aggregate.csv"
        if args.skip_existing and aggregate_path.exists():
            summary_path = source_output_dir / "table_projection_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                summary = {"input": str(input_path), "output_dir": str(source_output_dir), "skipped": True}
        else:
            summary = run_single_projection(
                SimpleNamespace(
                    input=str(input_path),
                    grid_side=args.grid_side,
                    feature_budgets=args.feature_budgets,
                    fit_targets=args.fit_targets,
                    variants=args.variants,
                    mlp_steps=args.mlp_steps,
                    seed=args.seed,
                    device=args.device,
                    output_dir=str(source_output_dir),
                )
            )
        summary["source_label"] = label
        summary["source_input"] = str(input_path)
        summaries.append(summary)

        combined_projection_rows.extend(
            tagged_rows(source_output_dir / "projection_results.csv", label=label, input_path=input_path, source_output_dir=source_output_dir)
        )
        combined_residual_rows.extend(
            tagged_rows(source_output_dir / "residual_projection_results.csv", label=label, input_path=input_path, source_output_dir=source_output_dir)
        )
        combined_random_rows.extend(
            tagged_rows(source_output_dir / "random_matched_radius_controls.csv", label=label, input_path=input_path, source_output_dir=source_output_dir)
        )
        combined_aggregate_rows.extend(
            tagged_rows(aggregate_path, label=label, input_path=input_path, source_output_dir=source_output_dir)
        )

    write_csv_rows(output_dir / "projection_results.csv", combined_projection_rows)
    write_csv_rows(output_dir / "residual_projection_results.csv", combined_residual_rows)
    write_csv_rows(output_dir / "random_matched_radius_controls.csv", combined_random_rows)
    write_csv_rows(output_dir / "projection_aggregate.csv", combined_aggregate_rows)

    summary = {
        "output_dir": str(output_dir),
        "num_inputs": len(inputs),
        "num_projection_rows": len(combined_projection_rows),
        "num_residual_rows": len(combined_residual_rows),
        "num_random_rows": len(combined_random_rows),
        "num_aggregate_rows": len(combined_aggregate_rows),
        "inputs": summaries,
    }
    (output_dir / "table_projection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, combined_aggregate_rows, summary)
    return summary


def write_report(output_dir: Path, aggregate_rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# V9 Batch Table Projection",
        "",
        f"Inputs: {summary['num_inputs']}",
        f"Aggregate rows: {summary['num_aggregate_rows']}",
        "",
        "## Best Residual Fits",
        "",
        "| source | variant | target | budget | table R2 | residual R2 |",
        "|---|---|---|---:|---:|---:|",
    ]
    sortable = []
    for row in aggregate_rows:
        try:
            residual_r2 = float(row.get("residual_fit_r2_mean", "nan"))
        except ValueError:
            residual_r2 = float("nan")
        sortable.append((residual_r2, row))
    for _, row in sorted(sortable, key=lambda item: item[0], reverse=True)[:40]:
        lines.append(
            "| "
            + f"{row.get('source_label', '')} | {row.get('variant', '')} | {row.get('fit_target', '')} | "
            + f"{row.get('feature_budget', '')} | {float(row.get('table_fit_r2_mean', 'nan')):.4f} | "
            + f"{float(row.get('residual_fit_r2_mean', 'nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `projection_results.csv`",
            "- `projection_aggregate.csv`",
            "- `residual_projection_results.csv`",
            "- `random_matched_radius_controls.csv`",
            "- `per_input/*/REPORT.md`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V9 batch table projection over multiple bias tables.")
    parser.add_argument("--input", nargs="+", required=True, help="Input specs: label=path or path.")
    parser.add_argument("--grid-side", type=int, default=None)
    parser.add_argument("--feature-budgets", type=str, default="33,55,221")
    parser.add_argument("--fit-targets", type=str, default="full_table,oblique_residual,axis_plus_residual")
    parser.add_argument(
        "--variants",
        type=str,
        default="axis_full_29,topk_dct,topk_dft,random_spectral_atoms_matched_radius,table_informed_toric_PJ_R0,table_informed_toric_PJ_R2,axis_plus_toric_residual_R0,axis_plus_toric_residual_R2",
    )
    parser.add_argument("--mlp-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="results/v9_table_projection")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
