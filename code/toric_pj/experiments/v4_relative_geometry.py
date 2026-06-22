from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from toric_pj.diagnostics.basis_projection import default_device
from toric_pj.diagnostics.relative_table_geometry import (
    aggregate_geometry_rows,
    geometry_rows,
    load_bias_npz,
    save_plots,
    spectral_peak_rows,
    write_csv,
    write_geometry_report,
)


def collect_inputs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            out.extend(sorted(path.rglob("bias_tables.npz")))
        elif path.name == "bias_tables.npz":
            out.append(path)
        else:
            raise ValueError(f"expected a bias_tables.npz file or directory: {path}")
    if not out:
        raise ValueError("no bias_tables.npz inputs found")
    return out


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = collect_inputs(args.input)
    all_rows: list[dict[str, object]] = []
    all_peaks: list[dict[str, object]] = []
    first_bundle = None
    first_metadata: dict[str, object] | None = None
    for path in inputs:
        bundle, metadata = load_bias_npz(path, device=device)
        metadata = dict(metadata)
        metadata.setdefault("bias_npz", str(path))
        basis = str(metadata.get("basis", path.parent.name))
        dataset = str(metadata.get("dataset", "unknown"))
        task = str(metadata.get("task", "unknown"))
        seed = int(metadata.get("seed", -1))
        all_rows.extend(geometry_rows(bundle.tables, basis=basis, dataset=dataset, task=task, seed=seed, topk=args.topk_metrics))
        all_peaks.extend(
            spectral_peak_rows(
                bundle.tables,
                basis=basis,
                dataset=dataset,
                task=task,
                seed=seed,
                topk=args.topk_peaks,
            )
        )
        if first_bundle is None:
            first_bundle = bundle
            first_metadata = metadata
    aggregate_rows = aggregate_geometry_rows(all_rows)
    write_csv(output_dir / "geometry_metrics.csv", all_rows)
    write_csv(output_dir / "geometry_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "spectral_peaks.csv", all_peaks)
    if first_bundle is not None:
        save_plots(output_dir, first_bundle.tables, first_bundle.residual_tables)
    report_metadata = {
        "inputs": len(inputs),
        "device": str(device),
        "topk_metrics": args.topk_metrics,
        "topk_peaks": args.topk_peaks,
    }
    if first_metadata:
        report_metadata.update({f"first_{key}": value for key, value in first_metadata.items()})
    write_geometry_report(output_dir, report_metadata, aggregate_rows)
    summary = {
        "inputs": [str(path) for path in inputs],
        "output_dir": str(output_dir),
        "num_metric_rows": len(all_rows),
        "num_peak_rows": len(all_peaks),
        "num_aggregate_rows": len(aggregate_rows),
    }
    (output_dir / "relative_geometry_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exported V4 relative-bias geometry.")
    parser.add_argument("--input", nargs="+", required=True, help="bias_tables.npz file(s) or directories containing them.")
    parser.add_argument("--output-dir", type=str, default="results/v4_relative_table_geometry")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--topk-metrics", type=int, default=5)
    parser.add_argument("--topk-peaks", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
