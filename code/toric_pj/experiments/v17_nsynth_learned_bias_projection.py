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

from toric_pj.experiments.v15_nsynth_cqt_table_projection import (
    Basis,
    dct_basis,
    dct_top_indices,
    fit_projection,
    full_jet_basis,
    offset_grid,
    top_fft_omegas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V17 projection diagnostics for learned NSynth scalar-bias tables.")
    parser.add_argument("--input", nargs="+", required=True, help="bias_tables.npz files or directories containing them.")
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v17_nsynth_learned_bias_projection")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--top-freqs", type=int, default=6)
    parser.add_argument("--dct-atoms", type=int, default=73)
    parser.add_argument("--ridge", type=float, default=1e-6)
    return parser.parse_args()


def find_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            out.extend(sorted(path.rglob("bias_tables.npz")))
        elif path.name == "bias_tables.npz" and path.exists():
            out.append(path)
        else:
            raise ValueError(f"expected bias_tables.npz or directory: {path}")
    if not out:
        raise ValueError("no bias_tables.npz inputs found")
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def row_float(row: dict[str, object], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def project_table(
    table: torch.Tensor,
    *,
    top_freqs: int,
    dct_atoms: int,
    ridge: float,
    table_id: str,
    source: str,
) -> list[dict[str, object]]:
    height, width = table.shape
    time_tokens = (height + 1) // 2
    freq_tokens = (width + 1) // 2
    d = offset_grid(time_tokens, freq_tokens, table.device)
    centered = table - table.mean()
    if float(centered.square().mean().detach().cpu()) <= 1e-20:
        return [
            {
                "source": source,
                "table_id": table_id,
                "basis": "degenerate_zero_variance",
                "num_features": 0,
                "r2": float("nan"),
                "mse": 0.0,
                "condition": float("nan"),
                "coeff_norm": 0.0,
                "target_std": 0.0,
                "order0_energy": 0.0,
                "order1_energy": 0.0,
                "order2_energy": 0.0,
            }
        ]
    omegas = top_fft_omegas(centered, k=top_freqs)
    dct_indices_13 = dct_top_indices(centered, k=12)
    dct_indices_full = dct_top_indices(centered, k=dct_atoms - 1)
    gen = torch.Generator(device=table.device)
    gen.manual_seed(1729)
    shuffled = d[torch.randperm(d.shape[0], device=table.device, generator=gen)]
    bases = [
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
    rows = []
    target_std = float(centered.std(unbiased=False).detach().cpu())
    target_norm = float(torch.linalg.norm(centered).detach().cpu())
    for basis in bases:
        row = fit_projection(basis, centered.reshape(-1), ridge=ridge)
        row.update({"source": source, "table_id": table_id, "target_std": target_std, "target_norm": target_norm})
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if str(row["table_id"]) == "mean":
            groups.setdefault(str(row["basis"]), []).append(row)
    out = []
    for basis, items in groups.items():
        values = np.array([row_float(row, "r2") for row in items], dtype=np.float64)
        out.append(
            {
                "basis": basis,
                "n": len(items),
                "r2_mean": float(np.nanmean(values)),
                "r2_std": float(np.nanstd(values)),
                "r2_min": float(np.nanmin(values)),
                "r2_max": float(np.nanmax(values)),
            }
        )
    out.sort(key=lambda row: row_float(row, "r2_mean"), reverse=True)
    return out


def margin_rows(rows: list[dict[str, object]], *, top_freqs: int, dct_atoms: int) -> list[dict[str, object]]:
    comparisons = [
        ("J1_minus_J0", f"fft_top{top_freqs}_J1", f"fft_top{top_freqs}_J0"),
        ("J2_minus_J1", f"fft_top{top_freqs}_J2", f"fft_top{top_freqs}_J1"),
        ("J2_minus_J0", f"fft_top{top_freqs}_J2", f"fft_top{top_freqs}_J0"),
        ("J2_minus_shuffle", f"fft_top{top_freqs}_J2", f"fft_top{top_freqs}_J2_coord_shuffle"),
        (f"DCT{dct_atoms}_minus_J2", f"dct_top{dct_atoms}", f"fft_top{top_freqs}_J2"),
    ]
    by_key = {(str(row["source"]), str(row["table_id"]), str(row["basis"])): row for row in rows}
    keys = sorted({(str(row["source"]), str(row["table_id"])) for row in rows})
    out = []
    for source, table_id in keys:
        for name, lhs, rhs in comparisons:
            left = by_key.get((source, table_id, lhs))
            right = by_key.get((source, table_id, rhs))
            if left is None or right is None:
                continue
            out.append(
                {
                    "source": source,
                    "table_id": table_id,
                    "comparison": name,
                    "r2_margin": row_float(left, "r2") - row_float(right, "r2"),
                    "lhs_r2": row_float(left, "r2"),
                    "rhs_r2": row_float(right, "r2"),
                }
            )
    return out


def aggregate_margins(margins: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in margins:
        groups.setdefault(str(row["comparison"]), []).append(row)
    out = []
    for comparison, items in groups.items():
        values = np.array([row_float(row, "r2_margin") for row in items], dtype=np.float64)
        out.append(
            {
                "comparison": comparison,
                "n": len(items),
                "margin_mean": float(np.nanmean(values)),
                "margin_std": float(np.nanstd(values)),
                "margin_min": float(np.nanmin(values)),
                "margin_max": float(np.nanmax(values)),
                "positive_count": int(np.sum(values > 0.0)),
            }
        )
    out.sort(key=lambda row: str(row["comparison"]))
    return out


def plot_mean_rows(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if str(row["table_id"]) == "mean"]
    if not selected:
        return
    labels = [f"{Path(str(row['source'])).parent.name}:{row['basis']}" for row in selected]
    values = [row_float(row, "r2") for row in selected]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#765f91")
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Projection R2")
    ax.set_title("Projection of learned NSynth scalar-bias teacher tables")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    output_dir: Path,
    rows: list[dict[str, object]],
    margins: list[dict[str, object]],
    margin_agg: list[dict[str, object]],
) -> None:
    mean_rows = [row for row in rows if str(row["table_id"]) == "mean"]
    lines = [
        "# V17 NSynth Learned Bias Projection",
        "",
        "Projection diagnostics for learned scalar attention-bias tables exported from V14.",
        "",
        "## Mean Tables",
        "",
        "| source | basis | R2 | target std | condition |",
        "|---|---|---:|---:|---:|",
    ]
    for row in mean_rows:
        lines.append(
            "| "
            + f"{Path(str(row['source'])).parent.name} | {row['basis']} | "
            + f"{row_float(row, 'r2'):.4f} | {row_float(row, 'target_std'):.4e} | {row_float(row, 'condition'):.2e} |"
        )
    lines.extend(
        [
            "",
            "## Mean-Table Margins",
            "",
            "| source | comparison | margin | lhs R2 | rhs R2 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in margins:
        if str(row["table_id"]) != "mean":
            continue
        lines.append(
            "| "
            + f"{Path(str(row['source'])).parent.name} | {row['comparison']} | "
            + f"{row_float(row, 'r2_margin'):.4f} | {row_float(row, 'lhs_r2'):.4f} | {row_float(row, 'rhs_r2'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## All-Table Margin Aggregate",
            "",
            "This aggregate includes every exported head table plus the mean table.",
            "",
            "| comparison | n | mean | std | min | max | positive |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in margin_agg:
        lines.append(
            "| "
            + f"{row['comparison']} | {int(row['n'])} | {row_float(row, 'margin_mean'):.4f} | "
            + f"{row_float(row, 'margin_std'):.4f} | {row_float(row, 'margin_min'):.4f} | "
            + f"{row_float(row, 'margin_max'):.4f} | {int(row['positive_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This diagnostic asks whether a learned relative-table scalar-bias teacher",
            "contains the same high-order structure seen in empirical CQT offset-table",
            "projection.  It is sensitive to whether the downstream teacher actually",
            "learned a nontrivial positional table.",
            "",
            "Artifacts:",
            "",
            "- `learned_bias_projection_rows.csv`",
            "- `learned_bias_projection_margins.csv`",
            "- `learned_bias_projection_margins_aggregate.csv`",
            "- `learned_bias_projection_mean_r2.pdf`",
            "- `summary.json`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    inputs = find_inputs(args.input)
    for path in inputs:
        data = np.load(path, allow_pickle=True)
        tables = torch.tensor(np.asarray(data["tables"]), device=device, dtype=torch.float32)
        mean_table = tables.mean(dim=(0, 1))
        rows.extend(
            project_table(
                mean_table,
                top_freqs=args.top_freqs,
                dct_atoms=args.dct_atoms,
                ridge=args.ridge,
                table_id="mean",
                source=str(path),
            )
        )
        for layer in range(tables.shape[0]):
            for head in range(tables.shape[1]):
                rows.extend(
                    project_table(
                        tables[layer, head],
                        top_freqs=args.top_freqs,
                        dct_atoms=args.dct_atoms,
                        ridge=args.ridge,
                        table_id=f"L{layer}H{head}",
                        source=str(path),
                    )
                )
    margins = margin_rows(rows, top_freqs=int(args.top_freqs), dct_atoms=int(args.dct_atoms))
    agg = aggregate(rows)
    margin_agg = aggregate_margins(margins)
    write_csv(output_dir / "learned_bias_projection_rows.csv", rows)
    write_csv(output_dir / "learned_bias_projection_margins.csv", margins)
    write_csv(output_dir / "learned_bias_projection_mean_aggregate.csv", agg)
    write_csv(output_dir / "learned_bias_projection_margins_aggregate.csv", margin_agg)
    plot_mean_rows(rows, output_dir / "learned_bias_projection_mean_r2.pdf")
    summary = {
        "status": "ok",
        "inputs": [str(path) for path in inputs],
        "top_freqs": int(args.top_freqs),
        "dct_atoms": int(args.dct_atoms),
        "ridge": float(args.ridge),
        "num_rows": len(rows),
        "num_margins": len(margins),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, rows, margins, margin_agg)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
