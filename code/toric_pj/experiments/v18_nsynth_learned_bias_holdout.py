from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import Basis, condition_number
from toric_pj.experiments.v15_nsynth_cqt_table_projection import (
    dct_basis,
    dct_top_indices,
    full_jet_basis,
    offset_grid,
    top_fft_omegas,
)
from toric_pj.experiments.v17_nsynth_learned_bias_projection import find_inputs, row_float, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V18 heldout-offset diagnostics for learned NSynth bias tables.")
    parser.add_argument("--input", nargs="+", required=True, help="bias_tables.npz files or directories containing them.")
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v18_nsynth_learned_bias_holdout")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-freqs", type=int, default=6)
    parser.add_argument("--dct-atoms", type=int, default=73)
    parser.add_argument("--ridges", type=str, default="1e-8,1e-6,1e-4")
    parser.add_argument("--schemes", type=str, default="random,outer_shell,checkerboard")
    parser.add_argument("--random-holdout-frac", type=float, default=0.2)
    parser.add_argument("--split-seeds", type=str, default="1729,1730,1731")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def split_masks(
    d: torch.Tensor,
    *,
    scheme: str,
    seed: int,
    holdout_frac: float,
    time_tokens: int,
    freq_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = d.shape[0]
    if scheme == "random":
        gen = torch.Generator(device=d.device)
        gen.manual_seed(int(seed))
        order = torch.randperm(n, device=d.device, generator=gen)
        n_holdout = max(1, min(n - 1, int(round(float(holdout_frac) * n))))
        heldout = torch.zeros(n, device=d.device, dtype=torch.bool)
        heldout[order[:n_holdout]] = True
    elif scheme == "outer_shell":
        max_t = float(max(time_tokens - 1, 1))
        max_f = float(max(freq_tokens - 1, 1))
        heldout = (d[:, 0].abs() > 0.72 * max_t) | (d[:, 1].abs() > 0.72 * max_f)
    elif scheme == "checkerboard":
        coords = d.to(torch.long)
        heldout = ((coords[:, 0] + coords[:, 1] + int(seed)) % 2) == 0
    else:
        raise ValueError(f"unknown split scheme: {scheme}")
    train = ~heldout
    if int(train.sum().detach().cpu()) == 0 or int(heldout.sum().detach().cpu()) == 0:
        raise ValueError(f"split {scheme} produced an empty train or heldout set")
    return train, heldout


def visible_table_for_selection(table: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    flat = table.reshape(-1)
    fill = flat[train_mask].mean()
    visible = flat.clone()
    visible[~train_mask] = fill
    return visible.reshape_as(table)


def make_bases(
    table: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    top_freqs: int,
    dct_atoms: int,
    shuffle_seed: int,
) -> list[Basis]:
    height, width = table.shape
    time_tokens = (height + 1) // 2
    freq_tokens = (width + 1) // 2
    d = offset_grid(time_tokens, freq_tokens, table.device)
    selection_table = visible_table_for_selection(table, train_mask)
    centered_selection = selection_table - selection_table.reshape(-1)[train_mask].mean()
    omegas = top_fft_omegas(centered_selection, k=top_freqs)
    dct_indices_13 = dct_top_indices(centered_selection, k=12)
    dct_indices_full = dct_top_indices(centered_selection, k=dct_atoms - 1)
    gen = torch.Generator(device=table.device)
    gen.manual_seed(int(shuffle_seed))
    shuffled = d[torch.randperm(d.shape[0], device=table.device, generator=gen)]
    return [
        Basis("constant", torch.ones((d.shape[0], 1), device=table.device, dtype=torch.float32), ["const"], [0]),
        full_jet_basis(d, omegas, order=0, time_tokens=time_tokens, freq_tokens=freq_tokens, name=f"fft_top{top_freqs}_J0"),
        full_jet_basis(d, omegas, order=1, time_tokens=time_tokens, freq_tokens=freq_tokens, name=f"fft_top{top_freqs}_J1"),
        full_jet_basis(d, omegas, order=2, time_tokens=time_tokens, freq_tokens=freq_tokens, name=f"fft_top{top_freqs}_J2"),
        full_jet_basis(
            shuffled,
            omegas,
            order=2,
            time_tokens=time_tokens,
            freq_tokens=freq_tokens,
            name=f"fft_top{top_freqs}_J2_coord_shuffle",
        ),
        dct_basis(d, time_tokens, freq_tokens, dct_indices_13, name="dct_top13"),
        dct_basis(d, time_tokens, freq_tokens, dct_indices_full, name=f"dct_top{dct_atoms}"),
    ]


def subset_scores(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    y = target[mask]
    p = pred[mask]
    mse = torch.mean((p - y).square())
    var = torch.mean((y - y.mean()).square()).clamp_min(1e-30)
    r2 = 1.0 - mse / var
    return float(r2.detach().cpu()), float(mse.detach().cpu())


def fit_split_projection(
    basis: Basis,
    target: torch.Tensor,
    *,
    train_mask: torch.Tensor,
    heldout_mask: torch.Tensor,
    ridge: float,
) -> dict[str, object]:
    y = target.reshape(-1, 1).to(device=basis.matrix.device, dtype=basis.matrix.dtype)
    x_raw = basis.matrix.to(device=basis.matrix.device, dtype=basis.matrix.dtype)
    norms = torch.linalg.norm(x_raw[train_mask], dim=0).clamp_min(1e-12)
    x = x_raw / norms.reshape(1, -1)
    x_train = x[train_mask]
    y_train = y[train_mask]
    gram = x_train.T @ x_train
    rhs = x_train.T @ y_train
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coeff = torch.linalg.solve(gram + float(ridge) * eye, rhs)
    pred = x @ coeff
    train_r2, train_mse = subset_scores(pred, y, train_mask)
    heldout_r2, heldout_mse = subset_scores(pred, y, heldout_mask)
    full_mask = torch.ones_like(train_mask)
    full_r2, full_mse = subset_scores(pred, y, full_mask)
    return {
        "basis": basis.name,
        "num_features": int(basis.matrix.shape[1]),
        "ridge": float(ridge),
        "train_r2": train_r2,
        "heldout_r2": heldout_r2,
        "full_r2": full_r2,
        "train_mse": train_mse,
        "heldout_mse": heldout_mse,
        "full_mse": full_mse,
        "condition_train": condition_number(x_train),
        "coeff_norm": float(torch.linalg.norm(coeff / norms.reshape(-1, 1)).detach().cpu()),
    }


def row_key(row: dict[str, object]) -> tuple[str, str, str, int, float, str]:
    return (
        str(row["source"]),
        str(row["table_id"]),
        str(row["split_scheme"]),
        int(row["split_seed"]),
        float(row["ridge"]),
        str(row["basis"]),
    )


def margin_rows(rows: list[dict[str, object]], *, top_freqs: int, dct_atoms: int) -> list[dict[str, object]]:
    comparisons = [
        ("J1_minus_J0", f"fft_top{top_freqs}_J1", f"fft_top{top_freqs}_J0"),
        ("J2_minus_J1", f"fft_top{top_freqs}_J2", f"fft_top{top_freqs}_J1"),
        ("J2_minus_J0", f"fft_top{top_freqs}_J2", f"fft_top{top_freqs}_J0"),
        ("J2_minus_shuffle", f"fft_top{top_freqs}_J2", f"fft_top{top_freqs}_J2_coord_shuffle"),
        (f"DCT{dct_atoms}_minus_J2", f"dct_top{dct_atoms}", f"fft_top{top_freqs}_J2"),
    ]
    by_key = {row_key(row): row for row in rows}
    groups = sorted(
        {
            (
                str(row["source"]),
                str(row["table_id"]),
                str(row["split_scheme"]),
                int(row["split_seed"]),
                float(row["ridge"]),
            )
            for row in rows
        }
    )
    out: list[dict[str, object]] = []
    for source, table_id, split_scheme, split_seed, ridge in groups:
        for name, lhs, rhs in comparisons:
            left = by_key.get((source, table_id, split_scheme, split_seed, ridge, lhs))
            right = by_key.get((source, table_id, split_scheme, split_seed, ridge, rhs))
            if left is None or right is None:
                continue
            out.append(
                {
                    "source": source,
                    "table_id": table_id,
                    "split_scheme": split_scheme,
                    "split_seed": split_seed,
                    "ridge": ridge,
                    "comparison": name,
                    "train_margin": row_float(left, "train_r2") - row_float(right, "train_r2"),
                    "heldout_margin": row_float(left, "heldout_r2") - row_float(right, "heldout_r2"),
                    "full_margin": row_float(left, "full_r2") - row_float(right, "full_r2"),
                    "lhs_heldout_r2": row_float(left, "heldout_r2"),
                    "rhs_heldout_r2": row_float(right, "heldout_r2"),
                }
            )
    return out


def aggregate_margins(margins: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for row in margins:
        key = (str(row["split_scheme"]), float(row["ridge"]), str(row["comparison"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for (split_scheme, ridge, comparison), items in groups.items():
        train = np.array([row_float(row, "train_margin") for row in items], dtype=np.float64)
        heldout = np.array([row_float(row, "heldout_margin") for row in items], dtype=np.float64)
        full = np.array([row_float(row, "full_margin") for row in items], dtype=np.float64)
        out.append(
            {
                "split_scheme": split_scheme,
                "ridge": ridge,
                "comparison": comparison,
                "n": len(items),
                "train_margin_mean": float(np.nanmean(train)),
                "heldout_margin_mean": float(np.nanmean(heldout)),
                "full_margin_mean": float(np.nanmean(full)),
                "heldout_margin_std": float(np.nanstd(heldout)),
                "heldout_margin_min": float(np.nanmin(heldout)),
                "heldout_margin_max": float(np.nanmax(heldout)),
                "heldout_positive_count": int(np.sum(heldout > 0.0)),
            }
        )
    out.sort(key=lambda row: (str(row["split_scheme"]), float(row["ridge"]), str(row["comparison"])))
    return out


def aggregate_basis_scores(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["split_scheme"]), float(row["ridge"]), str(row["basis"])), []).append(row)
    out: list[dict[str, object]] = []
    for (split_scheme, ridge, basis), items in groups.items():
        train = np.array([row_float(row, "train_r2") for row in items], dtype=np.float64)
        heldout = np.array([row_float(row, "heldout_r2") for row in items], dtype=np.float64)
        full = np.array([row_float(row, "full_r2") for row in items], dtype=np.float64)
        out.append(
            {
                "split_scheme": split_scheme,
                "ridge": ridge,
                "basis": basis,
                "n": len(items),
                "train_r2_mean": float(np.nanmean(train)),
                "heldout_r2_mean": float(np.nanmean(heldout)),
                "full_r2_mean": float(np.nanmean(full)),
                "heldout_r2_std": float(np.nanstd(heldout)),
                "heldout_r2_min": float(np.nanmin(heldout)),
                "heldout_r2_max": float(np.nanmax(heldout)),
            }
        )
    out.sort(key=lambda row: (str(row["split_scheme"]), float(row["ridge"]), -row_float(row, "heldout_r2_mean")))
    return out


def plot_default_margin(margin_agg: list[dict[str, object]], path: Path, *, default_ridge: float) -> None:
    selected = [
        row
        for row in margin_agg
        if abs(float(row["ridge"]) - float(default_ridge)) <= max(1e-15, abs(default_ridge) * 1e-6)
        and str(row["comparison"]) in {"J1_minus_J0", "J2_minus_J1", "J2_minus_shuffle"}
    ]
    if not selected:
        return
    labels = [f"{row['split_scheme']}\n{row['comparison']}" for row in selected]
    values = [row_float(row, "heldout_margin_mean") for row in selected]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#496f9f")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Heldout R2 margin")
    ax.set_title("Heldout-offset margins for learned NSynth scalar-bias tables")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    output_dir: Path,
    *,
    rows: list[dict[str, object]],
    score_agg: list[dict[str, object]],
    margin_agg: list[dict[str, object]],
    default_ridge: float,
    top_freqs: int,
    dct_atoms: int,
) -> None:
    j0_name = f"fft_top{top_freqs}_J0"
    j1_name = f"fft_top{top_freqs}_J1"
    j2_name = f"fft_top{top_freqs}_J2"
    shuffle_name = f"fft_top{top_freqs}_J2_coord_shuffle"
    dct_name = f"dct_top{dct_atoms}"
    lines = [
        "# V18 NSynth Learned-Bias Holdout Projection",
        "",
        "Heldout-offset diagnostics for learned NSynth/CQT scalar relative-bias tables.",
        "For each split, frequency centers and DCT atoms are selected from the training",
        "offset window only; coefficients are fit on training offsets and scored on heldout offsets.",
        "",
        "## Heldout Scores At Default Ridge",
        "",
        f"Default ridge: `{default_ridge:g}`.",
        "",
        "| split | basis | n | heldout R2 mean | std | min | max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    default_scores = [
        row
        for row in score_agg
        if abs(float(row["ridge"]) - float(default_ridge)) <= max(1e-15, abs(default_ridge) * 1e-6)
        and str(row["basis"]) in {j0_name, j1_name, j2_name, shuffle_name, dct_name}
    ]
    for row in default_scores:
        lines.append(
            "| "
            + f"{row['split_scheme']} | {row['basis']} | {int(row['n'])} | "
            + f"{row_float(row, 'heldout_r2_mean'):.4f} | {row_float(row, 'heldout_r2_std'):.4f} | "
            + f"{row_float(row, 'heldout_r2_min'):.4f} | {row_float(row, 'heldout_r2_max'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Heldout Margins",
            "",
            "| split | ridge | comparison | n | heldout mean | heldout min | positive |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in margin_agg:
        if str(row["comparison"]) not in {"J1_minus_J0", "J2_minus_J1", "J2_minus_shuffle"}:
            continue
        lines.append(
            "| "
            + f"{row['split_scheme']} | {float(row['ridge']):.0e} | {row['comparison']} | {int(row['n'])} | "
            + f"{row_float(row, 'heldout_margin_mean'):.4f} | {row_float(row, 'heldout_margin_min'):.4f} | "
            + f"{int(row['heldout_positive_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "A stable positive heldout J2-J1 margin would strengthen the V17 result by",
            "showing that the high-order jet contribution is not only an in-window",
            "projection artifact.  Random and checkerboard splits test interpolation;",
            "outer-shell tests a harder boundary/extrapolation setting.",
            "",
            "Artifacts:",
            "",
            "- `learned_bias_holdout_rows.csv`",
            "- `learned_bias_holdout_margins.csv`",
            "- `learned_bias_holdout_basis_aggregate.csv`",
            "- `learned_bias_holdout_margin_aggregate.csv`",
            "- `learned_bias_holdout_margins.pdf`",
            "- `summary.json`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    inputs = find_inputs(args.input)
    ridges = parse_float_list(args.ridges)
    schemes = parse_str_list(args.schemes)
    split_seeds = parse_int_list(args.split_seeds)
    rows: list[dict[str, object]] = []
    for path in inputs:
        data = np.load(path, allow_pickle=True)
        tables = torch.tensor(np.asarray(data["tables"]), device=device, dtype=torch.float32)
        table_items: list[tuple[str, torch.Tensor]] = [("mean", tables.mean(dim=(0, 1)))]
        for layer in range(tables.shape[0]):
            for head in range(tables.shape[1]):
                table_items.append((f"L{layer}H{head}", tables[layer, head]))
        for table_id, table in table_items:
            height, width = table.shape
            time_tokens = (height + 1) // 2
            freq_tokens = (width + 1) // 2
            d = offset_grid(time_tokens, freq_tokens, device)
            target = table.reshape(-1, 1)
            target_std = float(table.reshape(-1).std(unbiased=False).detach().cpu())
            for scheme in schemes:
                seeds = split_seeds if scheme == "random" else [0]
                for split_seed in seeds:
                    train_mask, heldout_mask = split_masks(
                        d,
                        scheme=scheme,
                        seed=split_seed,
                        holdout_frac=float(args.random_holdout_frac),
                        time_tokens=time_tokens,
                        freq_tokens=freq_tokens,
                    )
                    bases = make_bases(
                        table,
                        train_mask,
                        top_freqs=int(args.top_freqs),
                        dct_atoms=int(args.dct_atoms),
                        shuffle_seed=1729 + int(split_seed),
                    )
                    for basis in bases:
                        for ridge in ridges:
                            row = fit_split_projection(
                                basis,
                                target,
                                train_mask=train_mask,
                                heldout_mask=heldout_mask,
                                ridge=ridge,
                            )
                            row.update(
                                {
                                    "source": str(path),
                                    "table_id": table_id,
                                    "split_scheme": scheme,
                                    "split_seed": int(split_seed),
                                    "train_count": int(train_mask.sum().detach().cpu()),
                                    "heldout_count": int(heldout_mask.sum().detach().cpu()),
                                    "target_std": target_std,
                                }
                            )
                            rows.append(row)
    margins = margin_rows(rows, top_freqs=int(args.top_freqs), dct_atoms=int(args.dct_atoms))
    score_agg = aggregate_basis_scores(rows)
    margin_agg = aggregate_margins(margins)
    write_csv(output_dir / "learned_bias_holdout_rows.csv", rows)
    write_csv(output_dir / "learned_bias_holdout_margins.csv", margins)
    write_csv(output_dir / "learned_bias_holdout_basis_aggregate.csv", score_agg)
    write_csv(output_dir / "learned_bias_holdout_margin_aggregate.csv", margin_agg)
    default_ridge = 1e-6 if any(abs(item - 1e-6) <= 1e-15 for item in ridges) else ridges[0]
    plot_default_margin(margin_agg, output_dir / "learned_bias_holdout_margins.pdf", default_ridge=default_ridge)
    summary = {
        "status": "ok",
        "inputs": [str(path) for path in inputs],
        "top_freqs": int(args.top_freqs),
        "dct_atoms": int(args.dct_atoms),
        "ridges": ridges,
        "schemes": schemes,
        "split_seeds": split_seeds,
        "num_rows": len(rows),
        "num_margins": len(margins),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(
        output_dir,
        rows=rows,
        score_agg=score_agg,
        margin_agg=margin_agg,
        default_ridge=default_ridge,
        top_freqs=int(args.top_freqs),
        dct_atoms=int(args.dct_atoms),
    )
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
