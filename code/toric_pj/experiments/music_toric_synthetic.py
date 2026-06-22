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
class MusicBasis:
    name: str
    matrix: torch.Tensor
    groups: list[str]


def make_music_displacements(
    *,
    bars: int,
    beats_per_bar: int,
    voices: int,
    max_lag: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    events = bars * beats_per_bar
    rows = []
    for lag in range(max_lag + 1):
        q = torch.arange(lag, events, device=device, dtype=torch.long)
        k = q - lag
        q_voice = torch.arange(voices, device=device, dtype=torch.long)
        k_voice = torch.arange(voices, device=device, dtype=torch.long)
        q_grid, qv_grid, kv_grid = torch.meshgrid(q, q_voice, k_voice, indexing="ij")
        k_grid = q_grid - lag
        rows.append(
            torch.stack(
                [
                    torch.full_like(q_grid.reshape(-1), lag),
                    (q_grid.reshape(-1) // beats_per_bar) - (k_grid.reshape(-1) // beats_per_bar),
                    (q_grid.reshape(-1) % beats_per_bar - k_grid.reshape(-1) % beats_per_bar) % beats_per_bar,
                    qv_grid.reshape(-1) - kv_grid.reshape(-1),
                    (qv_grid.reshape(-1) == kv_grid.reshape(-1)).to(torch.long),
                ],
                dim=1,
            )
        )
    data = torch.cat(rows, dim=0).to(torch.float64)
    return {
        "d_event": data[:, 0],
        "d_bar": data[:, 1],
        "d_beat": data[:, 2],
        "d_voice": data[:, 3],
        "same_voice": data[:, 4],
    }


def music_teacher(coords: dict[str, torch.Tensor], *, max_lag: int, beats_per_bar: int) -> torch.Tensor:
    d_event = coords["d_event"]
    d_bar = coords["d_bar"]
    d_beat = coords["d_beat"]
    same_voice = coords["same_voice"]
    x = d_event / float(max_lag)
    b = d_bar / max(float(max_lag // beats_per_bar), 1.0)
    beat_phase = 2.0 * math.pi * d_beat / float(beats_per_bar)
    bar_phase = 2.0 * math.pi * d_bar / 4.0
    oblique_phase = 0.09 * d_event + 0.7 * d_bar
    jet_coord = 0.8 * x + 0.35 * b
    return (
        -0.35 * x
        + 0.45 * torch.cos(beat_phase)
        + 0.20 * same_voice * torch.cos(beat_phase)
        + 0.25 * torch.cos(bar_phase)
        + 0.12 * jet_coord * torch.cos(oblique_phase)
    )


def build_basis(
    name: str,
    coords: dict[str, torch.Tensor],
    *,
    max_lag: int,
    beats_per_bar: int,
    shuffle_beat: bool = False,
    shuffle_bar: bool = False,
    seed: int = 0,
) -> MusicBasis:
    device = coords["d_event"].device
    d_event = coords["d_event"]
    d_bar = coords["d_bar"].clone()
    d_beat = coords["d_beat"].clone()
    d_voice = coords["d_voice"]
    same_voice = coords["same_voice"]
    if shuffle_beat:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        d_beat = d_beat[torch.randperm(d_beat.numel(), device=device, generator=gen)]
    if shuffle_bar:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 101)
        d_bar = d_bar[torch.randperm(d_bar.numel(), device=device, generator=gen)]

    x = d_event / float(max_lag)
    b = d_bar / max(float(max_lag // beats_per_bar), 1.0)
    columns: list[torch.Tensor] = []
    groups: list[str] = []

    def add(col: torch.Tensor, group: str) -> None:
        columns.append(col)
        groups.append(group)

    add(torch.ones_like(x), "const")
    add(-x, "affine")
    for freq in [0.09, 2.0 * math.pi / beats_per_bar, 2.0 * math.pi / (4.0 * beats_per_bar)]:
        add(torch.cos(freq * d_event), "event_fourier")
        add(torch.sin(freq * d_event), "event_fourier")

    if name in {"1d_plus_voice", "toric_event_bar", "toric_music", "toric_music_lc", "toric_music_beat_shuffle", "toric_music_bar_shuffle"}:
        add(same_voice, "voice")
        add(d_voice, "voice")

    if name in {"toric_event_bar", "toric_music", "toric_music_lc", "toric_music_beat_shuffle", "toric_music_bar_shuffle"}:
        add(-b, "affine")
        for freq_bar in [0.7, 2.0 * math.pi / 4.0]:
            ph = 0.09 * d_event + freq_bar * d_bar
            add(torch.cos(ph), "toric_FJ")
            add(torch.sin(ph), "toric_FJ")
            jet = 0.8 * x + 0.35 * b
            add(jet * torch.cos(ph), "toric_FJ")
            add(jet * torch.sin(ph), "toric_FJ")

    if name in {"toric_music", "toric_music_lc", "toric_music_beat_shuffle", "toric_music_bar_shuffle"}:
        beat_phase = 2.0 * math.pi * d_beat / float(beats_per_bar)
        add(torch.cos(beat_phase), "beat_cyclic")
        add(torch.sin(beat_phase), "beat_cyclic")
        add(same_voice * torch.cos(beat_phase), "voice_beat")
        add(same_voice * torch.sin(beat_phase), "voice_beat")

    if name == "toric_music_lc":
        L = float(max_lag)
        raw = 0.8 * d_event + 0.35 * d_bar * beats_per_bar
        phi = L * torch.asinh(raw / L)
        beta = raw / torch.sqrt(raw.square() + L**2)
        for omega in [0.035, 0.07]:
            add(beta * torch.cos(omega * phi), "LC")
            add(beta * torch.sin(omega * phi), "LC")
            add(beta.square() * torch.cos(omega * phi), "LC")
            add(beta.square() * torch.sin(omega * phi), "LC")

    matrix, _ = normalize_columns(torch.stack(columns, dim=1))
    return MusicBasis(name=name, matrix=matrix, groups=groups)


def fit_basis(basis: MusicBasis, target: torch.Tensor, ridge: float) -> dict[str, object]:
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


def group_diagnostics(basis: MusicBasis, coeff: torch.Tensor, target: torch.Tensor) -> tuple[dict[str, float], dict[str, float]]:
    pred = basis.matrix @ coeff
    y = target.reshape(-1, 1)
    base_mse = torch.mean((pred - y).square())
    denom = torch.linalg.norm(pred).clamp_min(1e-30)
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
    coords = make_music_displacements(
        bars=args.bars,
        beats_per_bar=args.beats_per_bar,
        voices=args.voices,
        max_lag=args.max_lag,
        device=device,
    )
    target = music_teacher(coords, max_lag=args.max_lag, beats_per_bar=args.beats_per_bar)
    basis_names = [
        "1d_lag",
        "1d_plus_voice",
        "toric_event_bar",
        "toric_music",
        "toric_music_lc",
        "toric_music_beat_shuffle",
        "toric_music_bar_shuffle",
    ]
    rows = []
    for name in basis_names:
        basis = build_basis(
            name,
            coords,
            max_lag=args.max_lag,
            beats_per_bar=args.beats_per_bar,
            shuffle_beat=name == "toric_music_beat_shuffle",
            shuffle_bar=name == "toric_music_bar_shuffle",
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

    csv_path = output_dir / "music_toric_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = output_dir / "music_toric_r2.png"
    plot_results(rows, plot_path)
    summary = {
        "device": str(device),
        "num_samples": int(target.numel()),
        "bars": args.bars,
        "beats_per_bar": args.beats_per_bar,
        "voices": args.voices,
        "max_lag": args.max_lag,
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": rows,
    }
    summary_path = output_dir / "music_toric_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["basis"]) for row in rows]
    r2 = [float(row["r2"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(len(labels)), r2, color="#3b7ea1")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R^2")
    ax.set_title("Synthetic Music Toric Coordinate Probe")
    for i, value in enumerate(r2):
        ax.text(i, min(value + 0.02, 1.02), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic music toric coordinate probe.")
    parser.add_argument("--bars", type=int, default=64)
    parser.add_argument("--beats-per-bar", type=int, default=8)
    parser.add_argument("--voices", type=int, default=2)
    parser.add_argument("--max-lag", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/stage5_music")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
