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

from toric_pj.experiments.v12_phase_a_utils import parse_csv_list, summarize_groups, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V12 E7 conditioning and identifiability audit.")
    parser.add_argument("--inputs", required=True, help="Comma-separated v12 result directories.")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--ridge-grid", type=str, default="1e-12,1e-10,1e-8,1e-6,1e-4,1e-2")
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v12_e7_conditioning_audit")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def first_present(row: dict[str, object], keys: list[str]) -> float:
    for key in keys:
        value = f(row, key)
        if math.isfinite(value):
            return value
    return float("nan")


def normalize_condition_rows(input_dir: Path) -> list[dict[str, object]]:
    candidates = [
        "projection_results.csv",
        "frequency_source_results.csv",
        "compactified_chart_results.csv",
        "sector_recovery_results.csv",
    ]
    out: list[dict[str, object]] = []
    for name in candidates:
        for row in read_csv(input_dir / name):
            item = dict(row)
            item["source_dir"] = str(input_dir)
            item["source_file"] = name
            item["condition_number"] = first_present(row, ["condition_number", "design_condition"])
            item["effective_rank"] = first_present(row, ["effective_rank"])
            item["coeff_norm"] = first_present(row, ["coeff_norm"])
            item["fit_score"] = first_present(
                row,
                [
                    "table_fit_r2",
                    "target_fit_r2",
                    "visible_table_r2",
                    "r2_full",
                    "r2_vis_clean",
                ],
            )
            item["deployment_gap"] = first_present(row, ["full_visible_gap", "final_extrapolation_gain_mean"])
            item["tail_rms"] = first_present(row, ["heldout_rms", "ext_outer_rms", "extended_outer_pred_rms"])
            item["basis_label"] = "|".join(
                str(row.get(key, ""))
                for key in [
                    "protocol",
                    "source",
                    "frequency_source",
                    "basis_family",
                    "model",
                    "phase_chart",
                    "jet_chart",
                    "order",
                ]
                if str(row.get(key, "")) != ""
            )
            out.append(item)
    return out


def collect_ridge_paths(input_dir: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for path in input_dir.glob("**/ridge_path.csv"):
        for row in read_csv(path):
            item = dict(row)
            item["source_dir"] = str(input_dir)
            item["source_file"] = str(path.relative_to(input_dir))
            out.append(item)
    return out


def frequency_stability_rows(condition_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
    for row in condition_rows:
        raw = row.get("omegas", "")
        if not raw:
            continue
        try:
            arr = np.array(json.loads(str(raw)), dtype=np.float64)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            continue
        key = (
            str(row.get("source_dir", "")),
            str(row.get("fit_target", "")),
            str(row.get("source", row.get("frequency_source", ""))),
            str(row.get("order", "")),
        )
        groups.setdefault(key, []).append(arr)
    out: list[dict[str, object]] = []
    for (source_dir, fit_target, source, order), arrays in groups.items():
        radii = np.concatenate([np.linalg.norm(arr, axis=1) for arr in arrays])
        angles = np.concatenate([np.arctan2(arr[:, 1], arr[:, 0]) for arr in arrays])
        bins = np.histogram(angles, bins=12, range=(-math.pi, math.pi))[0].astype(np.float64)
        probs = bins / max(1.0, bins.sum())
        nz = probs[probs > 0]
        entropy = float(-(nz * np.log(nz)).sum() / math.log(12)) if nz.size else float("nan")
        out.append(
            {
                "source_dir": source_dir,
                "fit_target": fit_target,
                "source": source,
                "order": order,
                "n_rows": len(arrays),
                "n_omegas": int(sum(arr.shape[0] for arr in arrays)),
                "radius_mean": float(radii.mean()),
                "radius_std": float(radii.std()),
                "angle_entropy": entropy,
            }
        )
    return out


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dirs = [Path(item) for item in parse_csv_list(args.inputs)]

    condition_rows: list[dict[str, object]] = []
    ridge_rows: list[dict[str, object]] = []
    for input_dir in input_dirs:
        condition_rows.extend(normalize_condition_rows(input_dir))
        ridge_rows.extend(collect_ridge_paths(input_dir))
    aggregate = summarize_groups(
        condition_rows,
        keys=["source_file", "basis_label"],
        numeric=["condition_number", "effective_rank", "coeff_norm", "fit_score", "deployment_gap", "tail_rms"],
    )
    stability = frequency_stability_rows(condition_rows)

    write_csv(output_dir / "conditioning_results.csv", condition_rows)
    write_csv(output_dir / "conditioning_aggregate.csv", aggregate)
    write_csv(output_dir / "ridge_path.csv", ridge_rows)
    write_csv(output_dir / "bootstrap_frequency_stability.csv", stability)
    plot_scatter(
        condition_rows,
        x_key="condition_number",
        y_key="tail_rms",
        path=output_dir / "condition_vs_tail.pdf",
        xlabel="condition number",
        ylabel="tail RMS",
        logx=True,
    )
    plot_scatter(
        condition_rows,
        x_key="coeff_norm",
        y_key="deployment_gap",
        path=output_dir / "coeff_norm_vs_deployment_gap.pdf",
        xlabel="coefficient norm",
        ylabel="deployment gap",
        logx=True,
    )
    plot_ridge_path(ridge_rows, output_dir / "ridge_path_fit.pdf")

    summary = {
        "inputs": [str(item) for item in input_dirs],
        "output_dir": str(output_dir),
        "num_condition_rows": len(condition_rows),
        "num_ridge_rows": len(ridge_rows),
        "num_frequency_stability_rows": len(stability),
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, aggregate)
    return summary


def plot_scatter(
    rows: list[dict[str, object]],
    *,
    x_key: str,
    y_key: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    logx: bool = False,
) -> None:
    vals = [(f(row, x_key), f(row, y_key), str(row.get("source_file", ""))) for row in rows]
    vals = [(x, y, label) for x, y, label in vals if math.isfinite(x) and math.isfinite(y)]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    labels = sorted({label for _, _, label in vals})
    for label in labels:
        group = [(x, y) for x, y, item_label in vals if item_label == label]
        if not group:
            continue
        ax.scatter([x for x, _ in group], [y for _, y in group], s=12, alpha=0.5, label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ridge_path(rows: list[dict[str, object]], path: Path) -> None:
    vals = [(f(row, "ridge"), first_present(row, ["train_r2", "target_fit_r2", "table_fit_r2"])) for row in rows]
    vals = [(x, y) for x, y in vals if math.isfinite(x) and math.isfinite(y)]
    if not vals:
        return
    grouped: dict[float, list[float]] = {}
    for x, y in vals:
        grouped.setdefault(x, []).append(y)
    xs = sorted(grouped)
    means = [float(np.mean(grouped[x])) for x in xs]
    stds = [float(np.std(grouped[x])) for x in xs]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("ridge")
    ax.set_ylabel("mean train R2")
    ax.set_title("Ridge path summary")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V12 E7 Conditioning Audit",
        "",
        f"Condition rows: {summary['num_condition_rows']}",
        f"Ridge rows: {summary['num_ridge_rows']}",
        "",
        "| source | label | n | cond | erank | coeff norm | fit | gap | tail |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate[:80]:
        lines.append(
            "| "
            + f"{row['source_file']} | {row['basis_label']} | {int(row['n'])} | "
            + f"{float(row.get('condition_number_mean', float('nan'))):.2e} | "
            + f"{float(row.get('effective_rank_mean', float('nan'))):.2f} | "
            + f"{float(row.get('coeff_norm_mean', float('nan'))):.2e} | "
            + f"{float(row.get('fit_score_mean', float('nan'))):.4f} | "
            + f"{float(row.get('deployment_gap_mean', float('nan'))):.4f} | "
            + f"{float(row.get('tail_rms_mean', float('nan'))):.4f} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `conditioning_results.csv`",
            "- `conditioning_aggregate.csv`",
            "- `ridge_path.csv`",
            "- `bootstrap_frequency_stability.csv`",
            "- `condition_vs_tail.pdf`",
            "- `coeff_norm_vs_deployment_gap.pdf`",
            "- `ridge_path_fit.pdf`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()

