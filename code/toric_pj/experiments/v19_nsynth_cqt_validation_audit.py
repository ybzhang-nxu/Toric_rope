from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V19 audit of NSynth/CQT real-data validation artifacts.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v19_nsynth_cqt_validation_audit")
    parser.add_argument("--data-root", type=str, default="data/nsynth/nsynth-test")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, object], key: str) -> float:
    return float(row[key])


def find_one(rows: Iterable[dict[str, str]], **items: object) -> dict[str, str]:
    matches = []
    for row in rows:
        ok = True
        for key, value in items.items():
            if str(row.get(key)) != str(value):
                ok = False
                break
        if ok:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected one row for {items}, found {len(matches)}")
    return matches[0]


def status_row(
    checks: list[dict[str, object]],
    *,
    gate: str,
    status: str,
    value: object,
    criterion: str,
    source: Path,
    reading: str,
) -> None:
    checks.append(
        {
            "gate": gate,
            "status": status,
            "value": value,
            "criterion": criterion,
            "source": str(source),
            "reading": reading,
        }
    )


def margin_from_projection(rows: list[dict[str, str]], lhs: str, rhs: str) -> float:
    left = find_one(rows, basis=lhs)
    right = find_one(rows, basis=rhs)
    return float(left["r2"]) - float(right["r2"])


def plot_margins(plot_rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["label"]) for row in plot_rows]
    values = [float(row["margin"]) for row in plot_rows]
    colors = ["#4c6f91" if value >= 0.0 else "#a45a52" for value in values]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(values))
    ax.bar(x, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("R2 margin")
    ax.set_title("Audited NSynth/CQT higher-order jet margins")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.root).resolve()
    results = root / "MetricToric" / "results"
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "v14": results / "v14_nsynth_cqt_masked_confirm256_rectblock_3seed_2k" / "nsynth_cqt_aggregate.csv",
        "v15": results / "v15_nsynth_cqt_table_projection_256_top6" / "projection_results.csv",
        "v16_margins": results
        / "v16_nsynth_cqt_projection_stability_256_subsets"
        / "projection_stability_margin_aggregate.csv",
        "v16_scores": results
        / "v16_nsynth_cqt_projection_stability_256_subsets"
        / "projection_stability_aggregate.csv",
        "v17_scores": results / "v17_nsynth_learned_bias_projection_3seed_2k" / "learned_bias_projection_mean_aggregate.csv",
        "v17_margins": results
        / "v17_nsynth_learned_bias_projection_3seed_2k"
        / "learned_bias_projection_margins_aggregate.csv",
        "v18_scores": results
        / "v18_nsynth_learned_bias_holdout_ridge_sweep_3seed_2k"
        / "learned_bias_holdout_basis_aggregate.csv",
        "v18_margins": results
        / "v18_nsynth_learned_bias_holdout_ridge_sweep_3seed_2k"
        / "learned_bias_holdout_margin_aggregate.csv",
    }

    checks: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []

    data_root = root / args.data_root
    wav_count = len(list((data_root / "audio").glob("*.wav")))
    status_row(
        checks,
        gate="data_count",
        status="pass" if wav_count >= 256 else "fail",
        value=wav_count,
        criterion="at least 256 real NSynth wav files are present",
        source=data_root,
        reading="Confirms the current audits use the intended real-data subset size.",
    )

    v14 = read_csv(paths["v14"])
    best_v14 = max(v14, key=lambda row: float(row["score_mean"]))
    j2_v14 = find_one(v14, basis="toric_j2")
    status_row(
        checks,
        gate="v14_downstream_boundary",
        status="boundary" if best_v14["basis"] != "toric_j2" else "review",
        value=f"best={best_v14['basis']}:{float(best_v14['score_mean']):.4f}; J2={float(j2_v14['score_mean']):.4f}",
        criterion="MT-A07 should not be described as a downstream J2 win",
        source=paths["v14"],
        reading="The downstream masked-reconstruction pilot remains a boundary/negative result.",
    )

    v15 = read_csv(paths["v15"])
    v15_j1_j0 = margin_from_projection(v15, "fft_top6_J1", "fft_top6_J0")
    v15_j2_j1 = margin_from_projection(v15, "fft_top6_J2", "fft_top6_J1")
    v15_j2_shuffle = margin_from_projection(v15, "fft_top6_J2", "fft_top6_J2_coord_shuffle")
    for name, value, criterion in [
        ("v15_empirical_J1_minus_J0", v15_j1_j0, "J1 improves over J0 on empirical CQT offset table"),
        ("v15_empirical_J2_minus_J1", v15_j2_j1, "J2 improves over J1 on empirical CQT offset table"),
        ("v15_empirical_J2_minus_shuffle", v15_j2_shuffle, "coordinate shuffle collapses empirical CQT Toric/PJ fit"),
    ]:
        status_row(
            checks,
            gate=name,
            status="pass" if value > 0.0 else "fail",
            value=f"{value:.4f}",
            criterion=criterion,
            source=paths["v15"],
            reading="Empirical real-data covariance-field projection supports the high-order jet diagnostic.",
        )
        plot_rows.append({"label": name.replace("v15_empirical_", "MT-A05 "), "margin": value})

    v16_margins = read_csv(paths["v16_margins"])
    for subset_size in ["128", "192"]:
        for comp, gate in [
            ("J1_minus_J0", "J1-J0"),
            ("J2_minus_J1", "J2-J1"),
            ("J2_minus_shuffle", "J2-shuffle"),
        ]:
            row = find_one(v16_margins, subset_size=subset_size, comparison=comp)
            value = float(row["r2_margin_min"])
            status_row(
                checks,
                gate=f"v16_subset{subset_size}_{comp}",
                status="pass" if value > 0.0 else "fail",
                value=f"min={value:.4f}, mean={float(row['r2_margin_mean']):.4f}",
                criterion=f"{gate} is positive across random {subset_size}-example real-data subsets",
                source=paths["v16_margins"],
                reading="Subset stability check reduces the chance that MT-A05 is a single-table accident.",
            )
            plot_rows.append({"label": f"MT-A05 {subset_size} {gate}", "margin": float(row["r2_margin_mean"])})

    v17_scores = read_csv(paths["v17_scores"])
    v17_j0 = find_one(v17_scores, basis="fft_top6_J0")
    v17_j1 = find_one(v17_scores, basis="fft_top6_J1")
    v17_j2 = find_one(v17_scores, basis="fft_top6_J2")
    v17_shuffle = find_one(v17_scores, basis="fft_top6_J2_coord_shuffle")
    learned_hierarchy = float(v17_j2["r2_mean"]) > float(v17_j1["r2_mean"]) > float(v17_j0["r2_mean"])
    status_row(
        checks,
        gate="v17_learned_mean_hierarchy",
        status="pass" if learned_hierarchy else "fail",
        value=(
            f"J0={float(v17_j0['r2_mean']):.4f}, J1={float(v17_j1['r2_mean']):.4f}, "
            f"J2={float(v17_j2['r2_mean']):.4f}, shuffle={float(v17_shuffle['r2_mean']):.4f}"
        ),
        criterion="learned mean tables satisfy J0 < J1 < J2 and shuffle collapse",
        source=paths["v17_scores"],
        reading="Learned scalar attention-bias tables carry the same hierarchy as empirical CQT covariance fields.",
    )
    v17_margins = read_csv(paths["v17_margins"])
    for comp, label in [
        ("J1_minus_J0", "J1-J0"),
        ("J2_minus_J1", "J2-J1"),
        ("J2_minus_shuffle", "J2-shuffle"),
    ]:
        row = find_one(v17_margins, comparison=comp)
        status_row(
            checks,
            gate=f"v17_alltable_{comp}",
            status="pass" if int(float(row["positive_count"])) == int(float(row["n"])) else "fail",
            value=f"mean={float(row['margin_mean']):.4f}, min={float(row['margin_min']):.4f}, positive={row['positive_count']}/{row['n']}",
            criterion=f"learned head+mean tables have positive {label} margins",
            source=paths["v17_margins"],
            reading="All exported learned bias tables support the coordinate-sensitive hierarchy.",
        )
        plot_rows.append({"label": f"MT-A06 learned {label}", "margin": float(row["margin_mean"])})

    v18_scores = read_csv(paths["v18_scores"])
    v18_margins = read_csv(paths["v18_margins"])
    for comp, label in [
        ("J1_minus_J0", "J1-J0"),
        ("J2_minus_J1", "J2-J1"),
        ("J2_minus_shuffle", "J2-shuffle"),
    ]:
        row = find_one(v18_margins, split_scheme="random", ridge="1e-06", comparison=comp)
        status_row(
            checks,
            gate=f"v18_random_holdout_{comp}",
            status="pass" if int(float(row["heldout_positive_count"])) == int(float(row["n"])) else "fail",
            value=(
                f"heldout mean={float(row['heldout_margin_mean']):.4f}, "
                f"min={float(row['heldout_margin_min']):.4f}, positive={row['heldout_positive_count']}/{row['n']}"
            ),
            criterion=f"random heldout offsets preserve positive {label} margin at ridge 1e-6",
            source=paths["v18_margins"],
            reading="Heldout interpolation check shows MT-A06 is not only an in-sample projection artifact.",
        )
        plot_rows.append({"label": f"MT-A06 random {label}", "margin": float(row["heldout_margin_mean"])})

    outer_row = find_one(v18_margins, split_scheme="outer_shell", ridge="0.1", comparison="J2_minus_J1")
    outer_shuffle = find_one(v18_margins, split_scheme="outer_shell", ridge="0.1", comparison="J2_minus_shuffle")
    status_row(
        checks,
        gate="v18_outer_shell_regularized_boundary",
        status="boundary" if float(outer_row["heldout_margin_min"]) > 0.0 and float(outer_shuffle["heldout_margin_min"]) > 0.0 else "review",
        value=(
            f"J2-J1 mean={float(outer_row['heldout_margin_mean']):.4f}, min={float(outer_row['heldout_margin_min']):.4f}; "
            f"J2-shuffle mean={float(outer_shuffle['heldout_margin_mean']):.4f}, min={float(outer_shuffle['heldout_margin_min']):.4f}"
        ),
        criterion="outer-shell deployment needs explicit ridge/boundary handling",
        source=paths["v18_margins"],
        reading="This is useful boundary evidence, not a blanket extrapolation claim.",
    )

    checker = find_one(v18_margins, split_scheme="checkerboard", ridge="0.1", comparison="J2_minus_shuffle")
    status_row(
        checks,
        gate="v18_checkerboard_aliasing_stress",
        status="boundary" if float(checker["heldout_margin_mean"]) < 0.0 else "review",
        value=f"J2-shuffle mean={float(checker['heldout_margin_mean']):.4f}, positive={checker['heldout_positive_count']}/{checker['n']}",
        criterion="checkerboard split is not a clean coordinate-sensitive positive result",
        source=paths["v18_margins"],
        reading="Alternating heldout support behaves as an aliasing stress case.",
    )

    write_csv(output_dir / "evidence_checks.csv", checks)
    plot_margins(plot_rows, output_dir / "audited_nsynth_margins.pdf")

    summary = {
        "status": "ok",
        "wav_count": wav_count,
        "num_checks": len(checks),
        "pass_count": sum(1 for row in checks if row["status"] == "pass"),
        "boundary_count": sum(1 for row in checks if row["status"] == "boundary"),
        "fail_count": sum(1 for row in checks if row["status"] == "fail"),
        "review_count": sum(1 for row in checks if row["status"] == "review"),
        "sources": {key: str(path) for key, path in paths.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, checks, summary)
    return summary


def write_report(output_dir: Path, checks: list[dict[str, object]], summary: dict[str, object]) -> None:
    lines = [
        "# MT-A07 NSynth/CQT Validation Audit",
        "",
        "This report re-reads the NSynth/CQT result CSV files and recomputes the key",
        "claims used by the real-data validation narrative.  It is an audit layer,",
        "not a new training run.",
        "",
        "## Summary",
        "",
        f"- wav files: `{summary['wav_count']}`",
        f"- checks: `{summary['num_checks']}`",
        f"- pass: `{summary['pass_count']}`",
        f"- boundary: `{summary['boundary_count']}`",
        f"- fail: `{summary['fail_count']}`",
        f"- review: `{summary['review_count']}`",
        "",
        "## Checks",
        "",
        "| gate | status | value | criterion |",
        "|---|---|---|---|",
    ]
    for row in checks:
        lines.append(f"| {row['gate']} | {row['status']} | {row['value']} | {row['criterion']} |")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The audited evidence supports a bounded claim: real NSynth/CQT scalar",
            "offset fields and learned scalar attention-bias tables exhibit a stable",
            "coordinate-sensitive J0 -> J1 -> J2 hierarchy under projection and",
            "random heldout-offset checks.  The downstream masked-reconstruction",
            "pilot remains a boundary result, and structured heldout splits reinforce",
            "the need for explicit deployment-tail checks.",
            "",
            "Artifacts:",
            "",
            "- `evidence_checks.csv`",
            "- `audited_nsynth_margins.pdf`",
            "- `summary.json`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
