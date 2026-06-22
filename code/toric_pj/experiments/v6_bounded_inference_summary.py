#!/usr/bin/env python3
"""Summarize V6 CIFAR10 compact bounded-inference controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_RUNS = {
    "seed426_teacher426_base": "results/v6_cifar10_top110_from_classification_teacher_seed426_best_10k",
    "seed426_teacher526_base": "results/v6_cifar10_top110_from_classification_teacher_seed526_best_trainseed426_base_10k",
    "seed526_teacher426_base": "results/v6_cifar10_top110_from_classification_teacher_seed426_best_trainseed526_base_10k",
    "seed526_teacher526_base": "results/v6_cifar10_top110_from_classification_teacher_seed526_best_10k",
    "seed526_teacher626_base": "results/v6_cifar10_top110_from_classification_teacher_seed626_best_trainseed526_base_10k",
    "seed527_teacher526_base": "results/v6_cifar10_top110_from_classification_teacher_seed526_best_trainseed527_base_10k",
    "seed528_teacher526_base": "results/v6_cifar10_top110_from_classification_teacher_seed526_best_trainseed528_base_10k",
    "seed626_teacher626_base": "results/v6_cifar10_top110_from_classification_teacher_seed626_best_10k",
    "seed526_teacher526_strong": "results/v6_cifar10_top110_from_classification_teacher_seed526_best_h10_tail30_const_10k",
    "seed526_teacher526_noreg": "results/v6_cifar10_top110_from_classification_teacher_seed526_best_noreg_10k",
    "seed626_teacher526_base": "results/v6_cifar10_top110_from_classification_teacher_seed526_best_trainseed626_base_10k",
}

BOUNDED_MODES = {"visible_only", "heldout_clamp", "radial_decay", "radial_truncate", "radial_band"}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def parse_run_specs(values: list[str]) -> dict[str, Path]:
    if not values:
        return {label: Path(path) for label, path in DEFAULT_RUNS.items()}
    runs: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            path = Path(item)
            runs[path.name] = path
            continue
        label, raw_path = item.split("=", 1)
        runs[label.strip()] = Path(raw_path.strip())
    return runs


def read_run(label: str, path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_path = path / "offset_holdout_results.csv"
    controls_path = path / "offset_holdout_eval_controls.csv"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    if not controls_path.exists():
        raise FileNotFoundError(controls_path)

    results = pd.read_csv(result_path)
    controls = pd.read_csv(controls_path)
    result_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    for _, row in results.iterrows():
        seed = int(row["seed"])
        run_controls = controls[controls["seed"].astype(int) == seed].copy()
        if run_controls.empty:
            continue
        run_controls["eval_param"] = run_controls["eval_param"].fillna("")
        bounded = run_controls[run_controls["eval_mode"].isin(BOUNDED_MODES)].sort_values(
            "score", ascending=False
        )
        if bounded.empty:
            best_bounded = None
        else:
            best_bounded = bounded.iloc[0]

        def control_score(mode: str, param: str = "") -> float:
            matches = run_controls[
                (run_controls["eval_mode"] == mode) & (run_controls["eval_param"].astype(str) == param)
            ]
            if matches.empty:
                return float("nan")
            return float(matches.iloc[0]["score"])

        final_full = float(row["final_score"])
        visible = float(row["final_visible_only_score"])
        zero = float(row["final_zero_bias_score"])
        heldout_clamp = control_score("heldout_clamp")
        best_bounded_score = float(best_bounded["score"]) if best_bounded is not None else float("nan")
        best_bounded_label = (
            f"{best_bounded['eval_mode']} {best_bounded['eval_param']}".strip()
            if best_bounded is not None
            else ""
        )
        result_rows.append(
            {
                "run_label": label,
                "run_dir": str(path),
                "train_seed": seed,
                "teacher_seed": int(row["teacher_seed"]) if "teacher_seed" in row else "",
                "best_score": float(row["best_score"]),
                "best_step": int(row["best_step"]),
                "final_full": final_full,
                "visible_only": visible,
                "zero_bias": zero,
                "heldout_clamp": heldout_clamp,
                "best_bounded_control": best_bounded_label,
                "best_bounded_score": best_bounded_score,
                "best_bounded_minus_full": best_bounded_score - final_full,
                "best_bounded_minus_zero": best_bounded_score - zero,
                "full_visible_gain": float(row["final_extrapolation_gain"]),
                "heldout_bias_rms": float(row["heldout_bias_rms"]),
                "visible_teacher_init_r2_mean": float(row.get("visible_teacher_init_r2_mean", float("nan"))),
                "passes_0645_bounded": bool(best_bounded_score >= 0.645),
                "passes_0645_full": bool(final_full >= 0.645),
            }
        )

        for _, control in run_controls.iterrows():
            control_rows.append(
                {
                    "run_label": label,
                    "run_dir": str(path),
                    "train_seed": seed,
                    "teacher_seed": int(row["teacher_seed"]) if "teacher_seed" in row else "",
                    "eval_mode": control["eval_mode"],
                    "eval_param": control["eval_param"],
                    "score": float(control["score"]),
                    "loss": float(control["loss"]),
                    "bounded_mode": bool(control["eval_mode"] in BOUNDED_MODES),
                }
            )
    return result_rows, control_rows


def write_report(output_dir: Path, summary: pd.DataFrame, controls: pd.DataFrame) -> None:
    lines: list[str] = [
        "# V6 Bounded-Inference Compact Summary",
        "",
        "## Main Runs",
        "",
        "| run | teacher | train | full | zero | visible | heldout_clamp | best bounded | bounded-full | heldout RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for _, row in summary.sort_values(["train_seed", "teacher_seed", "run_label"]).iterrows():
        lines.append(
            "| {run_label} | {teacher_seed} | {train_seed} | {final_full:.4f} | {zero_bias:.4f} | "
            "{visible_only:.4f} | {heldout_clamp:.4f} | {best_bounded_control} {best_bounded_score:.4f} | "
            "{best_bounded_minus_full:+.4f} | {heldout_bias_rms:.4f} |".format(**row)
        )

    successful = summary[summary["passes_0645_bounded"]].copy()
    lines.extend(
        [
            "",
            "## Successful Bounded Controls",
            "",
            "| run | best bounded | full | gap |",
            "|---|---|---:|---:|",
        ]
    )
    if successful.empty:
        lines.append("| none |  |  |  |")
    else:
        for _, row in successful.sort_values("best_bounded_score", ascending=False).iterrows():
            lines.append(
                "| {run_label} | {best_bounded_control} {best_bounded_score:.4f} | "
                "{final_full:.4f} | {best_bounded_minus_full:+.4f} |".format(**row)
            )

    top_controls = controls[controls["bounded_mode"]].sort_values("score", ascending=False).head(12)
    lines.extend(
        [
            "",
            "## Top Bounded Eval Controls",
            "",
            "| run | mode | param | score |",
            "|---|---|---|---:|",
        ]
    )
    for _, row in top_controls.iterrows():
        lines.append(
            f"| {row['run_label']} | {row['eval_mode']} | {row['eval_param']} | {float(row['score']):.4f} |"
        )

    output_dir.joinpath("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(output_dir: Path, summary: pd.DataFrame) -> None:
    setup_style()
    plot_rows = summary.sort_values(["train_seed", "teacher_seed", "run_label"]).reset_index(drop=True)
    labels = [
        f"t{int(row.teacher_seed)}-s{int(row.train_seed)}"
        + ("*" if str(row.run_label).endswith("_noreg") else "")
        + ("+" if str(row.run_label).endswith("_strong") else "")
        for row in plot_rows.itertuples()
    ]
    x = list(range(len(plot_rows)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12.8, 4.9), constrained_layout=True)
    ax.bar(
        [item - width / 2 for item in x],
        plot_rows["final_full"],
        width=width,
        color="#6f7f8f",
        label="full",
    )
    ax.bar(
        [item + width / 2 for item in x],
        plot_rows["best_bounded_score"],
        width=width,
        color="#2b8a78",
        label="best bounded",
    )
    ax.axhline(0.645, color="#a33f3f", linestyle="--", linewidth=1.2, label="0.645 threshold")
    ax.axhline(0.6430, color="#8a6f2a", linestyle=":", linewidth=1.2, label="relative 3-seed mean")
    ax.set_ylim(0.625, 0.649)
    ax.set_ylabel("CIFAR10 accuracy")
    ax.set_title("V6 CIFAR10 compact: bounded/local signal vs full extrapolation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(ncol=4, loc="upper left", frameon=False)
    for idx, row in plot_rows.iterrows():
        if bool(row["passes_0645_bounded"]):
            ax.text(
                idx + width / 2,
                float(row["best_bounded_score"]) + 0.00035,
                "pass",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#1d6f5f",
            )
    fig.savefig(output_dir / "figure_bounded_full_vs_control.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec as label=path. If omitted, use known V6 compact teacher-transfer runs.",
    )
    parser.add_argument("--output-dir", default="results/v6_cifar10_bounded_inference_summary")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, object]] = []
    all_controls: list[dict[str, object]] = []
    for label, path in parse_run_specs(args.run).items():
        result_rows, control_rows = read_run(label, path)
        all_results.extend(result_rows)
        all_controls.extend(control_rows)

    summary = pd.DataFrame(all_results)
    controls = pd.DataFrame(all_controls)
    summary.to_csv(output_dir / "bounded_inference_summary.csv", index=False)
    controls.to_csv(output_dir / "bounded_inference_controls.csv", index=False)
    write_report(output_dir, summary, controls)
    write_figure(output_dir, summary)
    payload = {
        "output_dir": str(output_dir),
        "runs": int(summary.shape[0]),
        "successful_bounded_runs": int(summary["passes_0645_bounded"].sum()),
        "successful_full_runs": int(summary["passes_0645_full"].sum()),
        "best_bounded_score": float(summary["best_bounded_score"].max()),
        "figure": str(output_dir / "figure_bounded_full_vs_control.png"),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
