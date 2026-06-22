from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from toric_pj.diagnostics.relative_table_geometry import load_bias_npz, relative_grid
from toric_pj.experiments.v12_phase_a_utils import (
    axis_j0_basis,
    fit_with_ridge_grid,
    generic_omegas,
    omegas_json,
    parse_csv_list,
    predict,
    r2_score,
    random_matched_omegas,
    select_omegas_from_source,
    shuffled_omegas,
    summarize_groups,
    table_target,
    toric_j0_basis,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V12 E2 frequency-source controls.")
    parser.add_argument("--teacher-bias", required=True)
    parser.add_argument("--order", type=int, default=0)
    parser.add_argument("--real-atoms", type=int, default=108)
    parser.add_argument(
        "--sources",
        type=str,
        default="axial_only,fixed_grid,random_matched,learned_mixed,table_informed,table_informed_shuffled",
    )
    parser.add_argument("--learned-restarts", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--fit-targets", type=str, default="full_table,oblique_residual,axis_plus_residual")
    parser.add_argument("--ridge-grid", type=str, default="1e-8,1e-6,1e-4,1e-2")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v12_e2_frequency_source_projection_cifar10_10k")
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def angular_entropy(omegas: torch.Tensor, *, bins: int = 12) -> float:
    if omegas.numel() == 0:
        return float("nan")
    angles = torch.atan2(omegas[:, 1], omegas[:, 0])
    idx = torch.floor(((angles + math.pi) / (2.0 * math.pi) * bins).clamp(0, bins - 1e-6)).to(torch.long)
    hist = torch.bincount(idx, minlength=bins).to(torch.float64)
    probs = hist / hist.sum().clamp_min(1.0)
    nz = probs[probs > 0]
    entropy = -torch.sum(nz * torch.log(nz)) / math.log(float(bins))
    return float(entropy.detach().cpu())


def learned_correlation_omegas(
    target: torch.Tensor,
    d: torch.Tensor,
    *,
    k: int,
    seed: int,
    restarts: int,
) -> torch.Tensor:
    device, dtype = target.device, target.dtype
    n_candidates = max(k, min(2048, k * max(1, restarts)))
    candidates = generic_omegas(n_candidates, device=device, dtype=dtype, seed=seed)
    y = (target.reshape(-1) - target.mean()).to(torch.float64)
    y = y / torch.linalg.norm(y).clamp_min(1e-12)
    phase = d.to(torch.float64) @ candidates.to(torch.float64).T
    cos = torch.cos(phase)
    sin = torch.sin(phase)
    cos = cos / torch.linalg.norm(cos, dim=0, keepdim=True).clamp_min(1e-12)
    sin = sin / torch.linalg.norm(sin, dim=0, keepdim=True).clamp_min(1e-12)
    score = (cos.T @ y).square() + (sin.T @ y).square()
    _, idx = torch.topk(score, k=min(k, candidates.shape[0]))
    return candidates[idx].to(dtype)


def source_omegas(
    source: str,
    target: torch.Tensor,
    d: torch.Tensor,
    *,
    k: int,
    seed: int,
    learned_restarts: int,
) -> torch.Tensor:
    if source == "axial_only":
        return torch.empty(0, 2, device=target.device, dtype=target.dtype)
    if source in {"fixed_grid", "fixed"}:
        return select_omegas_from_source("generic", target, k=k, seed=seed)
    if source == "random_matched":
        return select_omegas_from_source("random_matched", target, k=k, seed=seed)
    if source == "learned_mixed":
        return learned_correlation_omegas(target, d, k=k, seed=seed, restarts=learned_restarts)
    if source == "table_informed":
        return select_omegas_from_source("table_informed", target, k=k, seed=seed)
    if source == "table_informed_shuffled":
        return shuffled_omegas(select_omegas_from_source("table_informed", target, k=k, seed=seed), seed=seed)
    raise ValueError(f"unknown source: {source}")


def run(args: argparse.Namespace) -> dict[str, object]:
    if int(args.order) != 0:
        raise ValueError("E2 frequency-source controls currently enforce J0 only; set --order 0.")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, metadata = load_bias_npz(Path(args.teacher_bias), device=device)
    tables = bundle.tables.to(torch.float64)
    side = (int(tables.shape[-1]) + 1) // 2
    d = relative_grid(side, device=device, dtype=torch.float64)
    k = max(1, (int(args.real_atoms) - 1) // 2)
    sources = parse_csv_list(args.sources)
    fit_targets = parse_csv_list(args.fit_targets)
    ridge_grid = parse_float_list(args.ridge_grid)

    rows: list[dict[str, object]] = []
    ridge_rows: list[dict[str, object]] = []
    for layer in range(tables.shape[0]):
        for head in range(tables.shape[1]):
            table = tables[layer, head]
            for fit_target in fit_targets:
                target, base = table_target(table, fit_target)
                for source in sources:
                    for seed_idx in range(int(args.seeds)):
                        source_seed = int(args.seed + 1000 * layer + 37 * head + 101 * seed_idx)
                        omegas = source_omegas(
                            source,
                            target,
                            d,
                            k=k,
                            seed=source_seed,
                            learned_restarts=int(args.learned_restarts),
                        )
                        if source == "axial_only":
                            matrix = axis_j0_basis(d, name="axial_only").matrix
                            num_centers = 0
                        else:
                            matrix = toric_j0_basis(d, omegas, name=f"{source}_J0_atoms{args.real_atoms}").matrix
                            num_centers = int(omegas.shape[0])
                        fit, path = fit_with_ridge_grid(matrix, target.reshape(-1), ridge_grid=ridge_grid)
                        pred_target = predict(matrix, fit.coeff, fit.column_norms).reshape_as(table)
                        pred_table = pred_target + base
                        row = {
                            "dataset": metadata.get("dataset", "unknown"),
                            "task": metadata.get("task", "unknown"),
                            "teacher_basis": metadata.get("basis", "unknown"),
                            "teacher_seed": metadata.get("seed", -1),
                            "layer": layer,
                            "head": head,
                            "fit_target": fit_target,
                            "source": source,
                            "source_seed": source_seed,
                            "order": 0,
                            "real_atoms_requested": int(args.real_atoms),
                            "num_centers": num_centers,
                            "num_features": int(matrix.shape[1]),
                            "selected_ridge": fit.ridge,
                            "target_fit_r2": fit.r2,
                            "table_fit_r2": r2_score(pred_table, table),
                            "condition_number": fit.condition_number,
                            "effective_rank": fit.effective_rank,
                            "coeff_norm": fit.coeff_norm,
                            "frequency_direction_entropy": angular_entropy(omegas),
                            "omegas": omegas_json(omegas),
                        }
                        rows.append(row)
                        for item in path:
                            item.update(
                                {
                                    "layer": layer,
                                    "head": head,
                                    "fit_target": fit_target,
                                    "source": source,
                                    "source_seed": source_seed,
                                    "num_features": int(matrix.shape[1]),
                                    "num_centers": num_centers,
                                }
                            )
                            ridge_rows.append(item)

    aggregate = summarize_groups(
        rows,
        keys=["source", "fit_target"],
        numeric=[
            "target_fit_r2",
            "table_fit_r2",
            "condition_number",
            "effective_rank",
            "coeff_norm",
            "frequency_direction_entropy",
            "num_features",
            "num_centers",
        ],
    )
    write_csv(output_dir / "frequency_source_results.csv", rows)
    write_csv(output_dir / "frequency_source_aggregate.csv", aggregate)
    write_csv(output_dir / "ridge_path.csv", ridge_rows)
    plot_pareto(aggregate, output_dir / "frequency_source_pareto.pdf")

    summary = {
        "teacher_bias": args.teacher_bias,
        "metadata": metadata,
        "output_dir": str(output_dir),
        "num_rows": len(rows),
        "num_ridge_rows": len(ridge_rows),
        "sources": sources,
        "seeds": int(args.seeds),
        "real_atoms": int(args.real_atoms),
        "centers_for_j0": int(k),
        "learned_mixed_note": "Phase-A learned_mixed uses correlation-selected random candidate frequencies, not downstream gradient training.",
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, aggregate)
    return summary


def plot_pareto(aggregate: list[dict[str, object]], path: Path) -> None:
    vals = [row for row in aggregate if row["fit_target"] == "axis_plus_residual"]
    if not vals:
        vals = aggregate
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    x = [float(row.get("num_features_mean", np.nan)) for row in vals]
    y = [float(row.get("table_fit_r2_mean", np.nan)) for row in vals]
    labels = [str(row["source"]) for row in vals]
    ax.scatter(x, y, s=45)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("features")
    ax.set_ylabel("table fit R2")
    ax.set_title("Frequency-source projection control")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# V12 E2 Frequency-Source Controls",
        "",
        f"Teacher: `{summary['teacher_bias']}`",
        f"Rows: {summary['num_rows']}",
        "",
        "Note: `learned_mixed` is a low-cost Phase-A correlation-selected random candidate control.",
        "",
        "| source | target | n | table R2 | target R2 | entropy | cond | coeff norm | features |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| "
            + f"{row['source']} | {row['fit_target']} | {int(row['n'])} | "
            + f"{float(row.get('table_fit_r2_mean', float('nan'))):.4f} | "
            + f"{float(row.get('target_fit_r2_mean', float('nan'))):.4f} | "
            + f"{float(row.get('frequency_direction_entropy_mean', float('nan'))):.4f} | "
            + f"{float(row.get('condition_number_mean', float('nan'))):.2e} | "
            + f"{float(row.get('coeff_norm_mean', float('nan'))):.2e} | "
            + f"{float(row.get('num_features_mean', float('nan'))):.1f} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `frequency_source_results.csv`",
            "- `frequency_source_aggregate.csv`",
            "- `ridge_path.csv`",
            "- `frequency_source_pareto.pdf`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
