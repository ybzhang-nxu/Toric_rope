from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import default_device, normalize_columns


@dataclass(frozen=True)
class LangBasis:
    name: str
    matrix: torch.Tensor
    groups: list[str]


def make_hierarchical_displacements(
    *,
    num_tokens: int,
    segment_size: int,
    max_lag: int,
    samples_per_lag: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    lags = []
    segs = []
    same = []
    boundary = []
    within_delta = []
    for lag in range(max_lag + 1):
        low = lag
        high = num_tokens
        q = torch.randint(low=low, high=high, size=(samples_per_lag,), device=device, generator=gen)
        k = q - lag
        q_seg = q // segment_size
        k_seg = k // segment_size
        q_within = q % segment_size
        k_within = k % segment_size
        lags.append(torch.full((samples_per_lag,), lag, device=device, dtype=torch.float64))
        seg_delta = (q_seg - k_seg).to(torch.float64)
        segs.append(seg_delta)
        same.append((seg_delta == 0).to(torch.float64))
        boundary.append(((q_within < k_within) & (lag > 0)).to(torch.float64))
        within_delta.append((q_within - k_within).to(torch.float64))
    return {
        "d_token": torch.cat(lags),
        "d_segment": torch.cat(segs),
        "same_segment": torch.cat(same),
        "cross_boundary": torch.cat(boundary),
        "d_within": torch.cat(within_delta),
    }


def language_teacher(coords: dict[str, torch.Tensor], *, max_lag: int, segment_size: int) -> torch.Tensor:
    d_token = coords["d_token"]
    d_segment = coords["d_segment"]
    same_segment = coords["same_segment"]
    cross_boundary = coords["cross_boundary"]
    x = d_token / float(max_lag)
    s = d_segment / max(float(max_lag // segment_size), 1.0)
    segment_phase = 2.0 * math.pi * d_segment / 4.0
    local_phase = 2.0 * math.pi * d_token / float(segment_size)
    mixed_phase = 0.025 * d_token + 0.75 * d_segment
    return (
        -0.55 * x
        + 0.22 * same_segment
        - 0.18 * cross_boundary
        + 0.18 * torch.cos(segment_phase)
        + 0.08 * torch.cos(local_phase)
        + 0.10 * (0.7 * x + 0.4 * s) * torch.cos(mixed_phase)
    )


def build_basis(
    name: str,
    coords: dict[str, torch.Tensor],
    *,
    max_lag: int,
    segment_size: int,
    shuffle_segment: bool = False,
    seed: int = 0,
) -> LangBasis:
    device = coords["d_token"].device
    d_token = coords["d_token"]
    d_segment = coords["d_segment"].clone()
    if shuffle_segment:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 77)
        d_segment = d_segment[torch.randperm(d_segment.numel(), device=device, generator=gen)]
    same_segment = (d_segment == 0).to(torch.float64)
    cross_boundary = coords["cross_boundary"]
    x = d_token / float(max_lag)
    s = d_segment / max(float(max_lag // segment_size), 1.0)

    cols: list[torch.Tensor] = []
    groups: list[str] = []

    def add(col: torch.Tensor, group: str) -> None:
        cols.append(col)
        groups.append(group)

    add(torch.ones_like(x), "const")
    add(-x, "token_affine")
    for freq in [0.012, 0.025, 2.0 * math.pi / segment_size]:
        add(torch.cos(freq * d_token), "token_fourier")
        add(torch.sin(freq * d_token), "token_fourier")

    if name in {"token_plus_segment", "hierarchical_toric", "hierarchical_lc", "segment_shuffle"}:
        add(same_segment, "segment")
        add(cross_boundary, "segment")
        add(-s, "segment")
        for freq in [2.0 * math.pi / 4.0, 0.75]:
            add(torch.cos(freq * d_segment), "segment")
            add(torch.sin(freq * d_segment), "segment")

    if name in {"hierarchical_toric", "hierarchical_lc", "segment_shuffle"}:
        for omega_s in [0.75, 1.3]:
            ph = 0.025 * d_token + omega_s * d_segment
            add(torch.cos(ph), "toric_mixed")
            add(torch.sin(ph), "toric_mixed")
            jet = 0.7 * x + 0.4 * s
            add(jet * torch.cos(ph), "toric_mixed")
            add(jet * torch.sin(ph), "toric_mixed")

    if name == "hierarchical_lc":
        L = float(max_lag)
        raw = d_token + 0.35 * d_segment * segment_size
        phi = L * torch.asinh(raw / L)
        beta = raw / torch.sqrt(raw.square() + L**2)
        for omega in [0.012, 0.025]:
            add(beta * torch.cos(omega * phi), "LC")
            add(beta * torch.sin(omega * phi), "LC")
            add(beta.square() * torch.cos(omega * phi), "LC")
            add(beta.square() * torch.sin(omega * phi), "LC")

    matrix, _ = normalize_columns(torch.stack(cols, dim=1))
    return LangBasis(name=name, matrix=matrix, groups=groups)


def fit_basis(basis: LangBasis, target: torch.Tensor, ridge: float) -> dict[str, object]:
    y = target.reshape(-1, 1)
    x = basis.matrix
    gram = x.T @ x
    rhs = x.T @ y
    eye = torch.eye(gram.shape[0], device=x.device, dtype=x.dtype)
    coeff = torch.linalg.solve(gram + ridge * eye, rhs)
    pred = x @ coeff
    mse = torch.mean((pred - y).square())
    var = torch.mean((y - y.mean()).square()).clamp_min(1e-30)
    r2 = 1.0 - mse / var
    group_energy, group_loo = group_diagnostics(basis, coeff, target)
    return {
        "basis": basis.name,
        "mse": float(mse.detach().cpu()),
        "r2": float(r2.detach().cpu()),
        "num_features": basis.matrix.shape[1],
        "group_energy": group_energy,
        "group_loo": group_loo,
        "top_group_energy": max(group_energy, key=group_energy.get),
        "top_group_loo": max(group_loo, key=group_loo.get),
    }


def group_diagnostics(basis: LangBasis, coeff: torch.Tensor, target: torch.Tensor) -> tuple[dict[str, float], dict[str, float]]:
    pred = basis.matrix @ coeff
    y = target.reshape(-1, 1)
    denom = torch.linalg.norm(pred).clamp_min(1e-30)
    base_mse = torch.mean((pred - y).square())
    energy: dict[str, float] = {}
    loo: dict[str, float] = {}
    for group in sorted(set(basis.groups)):
        idx = torch.tensor([i for i, item in enumerate(basis.groups) if item == group], device=target.device)
        contrib = basis.matrix[:, idx] @ coeff[idx]
        energy[group] = float((torch.linalg.norm(contrib) / denom).detach().cpu())
        loo[group] = float((torch.mean((pred - contrib - y).square()) - base_mse).detach().cpu())
    return energy, loo


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coords = make_hierarchical_displacements(
        num_tokens=args.num_tokens,
        segment_size=args.segment_size,
        max_lag=args.max_lag,
        samples_per_lag=args.samples_per_lag,
        seed=args.seed,
        device=device,
    )
    target = language_teacher(coords, max_lag=args.max_lag, segment_size=args.segment_size)
    basis_names = ["token_lag", "token_plus_segment", "hierarchical_toric", "hierarchical_lc", "segment_shuffle"]
    rows = []
    for name in basis_names:
        basis = build_basis(
            name,
            coords,
            max_lag=args.max_lag,
            segment_size=args.segment_size,
            shuffle_segment=name == "segment_shuffle",
            seed=args.seed,
        )
        result = fit_basis(basis, target, ridge=args.ridge)
        rows.append(
            {
                **{key: value for key, value in result.items() if not isinstance(value, dict)},
                "group_energy": json.dumps(result["group_energy"], sort_keys=True),
                "group_loo": json.dumps(result["group_loo"], sort_keys=True),
            }
        )

    csv_path = output_dir / "language_hierarchical_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = output_dir / "language_hierarchical_r2.png"
    plot_results(rows, plot_path)
    summary = {
        "device": str(device),
        "num_samples": int(target.numel()),
        "num_tokens": args.num_tokens,
        "segment_size": args.segment_size,
        "max_lag": args.max_lag,
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "language_hierarchical_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["basis"]) for row in rows]
    r2 = [float(row["r2"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(np.arange(len(labels)), r2, color="#6f5e9c")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R^2")
    ax.set_title("Hierarchical Byte-like Language Coordinate Probe")
    for i, value in enumerate(r2):
        ax.text(i, min(value + 0.02, 1.02), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hierarchical byte-like language coordinate probe.")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--segment-size", type=int, default=128)
    parser.add_argument("--max-lag", type=int, default=512)
    parser.add_argument("--samples-per-lag", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage5_language")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
