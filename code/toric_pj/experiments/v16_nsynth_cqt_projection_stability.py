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

from toric_pj.experiments.v14_nsynth_cqt_masked import compute_cqt, discover_nsynth_records, patchify
from toric_pj.experiments.v15_nsynth_cqt_table_projection import (
    Basis,
    dct_basis,
    dct_top_indices,
    empirical_offset_table,
    fit_projection,
    full_jet_basis,
    offset_grid,
    top_fft_omegas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V16 NSynth CQT projection stability over real-data subsets.")
    parser.add_argument("--data-root", type=str, default="data/nsynth/nsynth-test")
    parser.add_argument("--output-dir", type=str, default="MetricToric/results/v16_nsynth_cqt_projection_stability")
    parser.add_argument("--cache-dir", type=str, default="MetricToric/results/v14_nsynth_cqt_masked_confirm256_rectblock_3seed_2k/cqt_cache")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--min-samples", type=int, default=128)
    parser.add_argument("--subset-sizes", type=str, default="128,192,256")
    parser.add_argument("--subset-seeds", type=str, default="101,202,303,404,505")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--cqt-bins", type=int, default=84)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--fmin-note", type=str, default="C1")
    parser.add_argument("--time-frames", type=int, default=128)
    parser.add_argument("--patch-time", type=int, default=4)
    parser.add_argument("--patch-freq", type=int, default=6)
    parser.add_argument("--top-freqs", type=int, default=6)
    parser.add_argument("--dct-atoms", type=int, default=73)
    parser.add_argument("--ridge", type=float, default=1e-8)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_all_patches(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, dict[str, object]]:
    records = discover_nsynth_records(Path(args.data_root), max_samples=args.max_samples)
    if len(records) < args.min_samples:
        raise RuntimeError(f"found {len(records)} NSynth wav files, need at least {args.min_samples}")
    cache_dir = Path(args.cache_dir)
    specs = [compute_cqt(record, args, cache_dir) for record in records]
    data = torch.tensor(np.stack(specs), dtype=torch.float32, device=device)
    mean = data.mean(dim=(0, 1), keepdim=True)
    std = data.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    patches, time_tokens, freq_tokens = patchify((data - mean) / std, args.patch_time, args.patch_freq)
    stats = {
        "num_records": len(records),
        "time_tokens": int(time_tokens),
        "freq_tokens": int(freq_tokens),
        "patch_dim": int(patches.shape[-1]),
        "mean_global": float(mean.mean().detach().cpu()),
        "std_global": float(std.mean().detach().cpu()),
    }
    return patches, stats


def subset_indices(total: int, size: int, seed: int, device: torch.device) -> torch.Tensor:
    if size >= total:
        return torch.arange(total, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randperm(total, device=device, generator=gen)[:size]


def run_subset(
    patches: torch.Tensor,
    *,
    subset_size: int,
    subset_seed: int,
    time_tokens: int,
    freq_tokens: int,
    top_freqs: int,
    dct_atoms: int,
    ridge: float,
) -> list[dict[str, object]]:
    idx = subset_indices(patches.shape[0], subset_size, subset_seed, patches.device)
    subset = patches[idx]
    table = empirical_offset_table(subset, time_tokens, freq_tokens)
    d = offset_grid(time_tokens, freq_tokens, patches.device)
    omegas = top_fft_omegas(table, k=top_freqs)
    dct_indices_13 = dct_top_indices(table, k=12)
    dct_indices_full = dct_top_indices(table, k=dct_atoms - 1)
    gen = torch.Generator(device=patches.device)
    gen.manual_seed(subset_seed + 17)
    shuffled = d[torch.randperm(d.shape[0], device=patches.device, generator=gen)]
    bases = [
        Basis("constant", torch.ones((d.shape[0], 1), device=patches.device, dtype=torch.float32), ["const"], [0]),
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
    for basis in bases:
        row = fit_projection(basis, table.reshape(-1), ridge=ridge)
        row.update(
            {
                "subset_size": int(subset_size),
                "subset_seed": int(subset_seed),
                "actual_subset_size": int(idx.numel()),
            }
        )
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, object]], keys: list[str], numeric: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    out = []
    for key_values, items in sorted(groups.items()):
        item: dict[str, object] = {key: value for key, value in zip(keys, key_values)}
        item["n"] = len(items)
        for name in numeric:
            values = np.array([float(row[name]) for row in items], dtype=np.float64)
            item[f"{name}_mean"] = float(np.nanmean(values))
            item[f"{name}_std"] = float(np.nanstd(values))
            item[f"{name}_min"] = float(np.nanmin(values))
            item[f"{name}_max"] = float(np.nanmax(values))
        out.append(item)
    return out


def margin_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (int(row["subset_size"]), int(row["subset_seed"]), str(row["basis"])): row
        for row in rows
    }
    comparisons = [
        ("J1_minus_J0", "fft_top6_J1", "fft_top6_J0"),
        ("J2_minus_J1", "fft_top6_J2", "fft_top6_J1"),
        ("J2_minus_J0", "fft_top6_J2", "fft_top6_J0"),
        ("J2_minus_shuffle", "fft_top6_J2", "fft_top6_J2_coord_shuffle"),
        ("DCT73_minus_J2", "dct_top73", "fft_top6_J2"),
        ("DCT13_minus_J0", "dct_top13", "fft_top6_J0"),
    ]
    out = []
    subset_keys = sorted({(int(row["subset_size"]), int(row["subset_seed"])) for row in rows})
    for subset_size, subset_seed in subset_keys:
        for name, lhs, rhs in comparisons:
            left = by_key.get((subset_size, subset_seed, lhs))
            right = by_key.get((subset_size, subset_seed, rhs))
            if left is None or right is None:
                continue
            out.append(
                {
                    "subset_size": subset_size,
                    "subset_seed": subset_seed,
                    "comparison": name,
                    "lhs": lhs,
                    "rhs": rhs,
                    "r2_margin": float(left["r2"]) - float(right["r2"]),
                    "lhs_r2": float(left["r2"]),
                    "rhs_r2": float(right["r2"]),
                }
            )
    return out


def plot_margins(margins_agg: list[dict[str, object]], path: Path) -> None:
    rows = [
        row
        for row in margins_agg
        if int(row["subset_size"]) != 256
        and str(row["comparison"]) in {"J1_minus_J0", "J2_minus_J1", "J2_minus_shuffle", "DCT73_minus_J2"}
    ]
    labels = [f"{row['subset_size']}:{row['comparison']}" for row in rows]
    values = [float(row["r2_margin_mean"]) for row in rows]
    errors = [float(row["r2_margin_std"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(values))
    ax.bar(x, values, yerr=errors, color="#4f7787", capsize=3)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Projection R2 margin")
    ax.set_title("NSynth CQT projection margins across real-data subsets")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: dict[str, object],
    projection_agg: list[dict[str, object]],
    margins_agg: list[dict[str, object]],
) -> None:
    lines = [
        "# V16 NSynth CQT Projection Stability",
        "",
        f"Records loaded: {summary['num_records']}",
        f"Subset sizes: {summary['subset_sizes']}",
        f"Subset seeds: {summary['subset_seeds']}",
        "",
        "## Projection R2 Aggregate",
        "",
        "| subset | basis | n | R2 mean | R2 std | R2 min | R2 max |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in projection_agg:
        lines.append(
            "| "
            + f"{int(row['subset_size'])} | {row['basis']} | {int(row['n'])} | "
            + f"{float(row['r2_mean']):.4f} | {float(row['r2_std']):.4f} | "
            + f"{float(row['r2_min']):.4f} | {float(row['r2_max']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Key Margins",
            "",
            "| subset | comparison | n | margin mean | margin std | margin min | margin max |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in margins_agg:
        if str(row["comparison"]) not in {"J1_minus_J0", "J2_minus_J1", "J2_minus_shuffle", "DCT73_minus_J2"}:
            continue
        lines.append(
            "| "
            + f"{int(row['subset_size'])} | {row['comparison']} | {int(row['n'])} | "
            + f"{float(row['r2_margin_mean']):.4f} | {float(row['r2_margin_std']):.4f} | "
            + f"{float(row['r2_margin_min']):.4f} | {float(row['r2_margin_max']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This is a stability check for the V15 empirical offset-table projection result.",
            "Each subset recomputes the empirical CQT offset table and table-informed FFT",
            "frequencies.  It is still a projection diagnostic, not a downstream task score.",
            "",
            "Artifacts:",
            "",
            "- `projection_stability_rows.csv`",
            "- `projection_stability_aggregate.csv`",
            "- `projection_stability_margins.csv`",
            "- `projection_stability_margin_aggregate.csv`",
            "- `projection_stability_margins.pdf`",
            "- `summary.json`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    patches, stats = load_all_patches(args, device)
    total = int(patches.shape[0])
    subset_sizes = parse_int_list(args.subset_sizes)
    subset_seeds = parse_int_list(args.subset_seeds)
    rows: list[dict[str, object]] = []
    for subset_size in subset_sizes:
        seeds = subset_seeds if subset_size < total else [subset_seeds[0]]
        for subset_seed in seeds:
            rows.extend(
                run_subset(
                    patches,
                    subset_size=subset_size,
                    subset_seed=subset_seed,
                    time_tokens=int(stats["time_tokens"]),
                    freq_tokens=int(stats["freq_tokens"]),
                    top_freqs=args.top_freqs,
                    dct_atoms=args.dct_atoms,
                    ridge=args.ridge,
                )
            )
            write_csv(output_dir / "projection_stability_rows.csv", rows)
    projection_agg = aggregate(rows, ["subset_size", "basis"], ["r2"])
    margins = margin_rows(rows)
    margins_agg = aggregate(margins, ["subset_size", "comparison"], ["r2_margin"])
    write_csv(output_dir / "projection_stability_aggregate.csv", projection_agg)
    write_csv(output_dir / "projection_stability_margins.csv", margins)
    write_csv(output_dir / "projection_stability_margin_aggregate.csv", margins_agg)
    plot_margins(margins_agg, output_dir / "projection_stability_margins.pdf")
    summary = {
        "status": "ok",
        "data_root": args.data_root,
        "device": str(device),
        "cache_dir": args.cache_dir,
        "subset_sizes": subset_sizes,
        "subset_seeds": subset_seeds,
        "top_freqs": int(args.top_freqs),
        "dct_atoms": int(args.dct_atoms),
        "ridge": float(args.ridge),
        **stats,
        "num_rows": len(rows),
        "num_margin_rows": len(margins),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, summary, projection_agg, margins_agg)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
