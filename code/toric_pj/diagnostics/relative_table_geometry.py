from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from toric_pj.diagnostics.basis_projection import Basis, directional_jet_basis, normalize_columns, toric_fourier_basis
from toric_pj.diagnostics.direction_alignment import normalize_direction


EPS = 1e-12


@dataclass
class RelativeGeometryBundle:
    tables: torch.Tensor
    axis_tables: torch.Tensor
    residual_tables: torch.Tensor
    dx_values: torch.Tensor
    dy_values: torch.Tensor


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pairwise_deltas(side: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    coords = torch.arange(side, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    return (positions[:, None, :] - positions[None, :, :]).reshape(-1, 2)


def relative_grid(side: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    vals = torch.arange(-(side - 1), side, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(vals, vals, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def coeff_to_pairwise_bias(basis_matrix: torch.Tensor, coeff: torch.Tensor, n_positions: int) -> torch.Tensor:
    """Return bias as [layer, head, query, key]."""
    if coeff.ndim != 3:
        raise ValueError("coeff must have shape [layer, head, feature]")
    bias_flat = torch.einsum("nf,lhf->lhn", basis_matrix.to(coeff.device, coeff.dtype), coeff)
    return bias_flat.reshape(coeff.shape[0], coeff.shape[1], n_positions, n_positions)


def pairwise_to_relative_table(pairwise_bias: torch.Tensor, side: int) -> torch.Tensor:
    """Average [*, query, key] bias values onto [*, dx, dy]."""
    device = pairwise_bias.device
    dtype = pairwise_bias.dtype
    deltas = pairwise_deltas(side, device=device, dtype=dtype).to(torch.long)
    table_side = 2 * side - 1
    flat_index = (deltas[:, 0] + side - 1) * table_side + (deltas[:, 1] + side - 1)
    values = pairwise_bias.reshape(*pairwise_bias.shape[:-2], side * side * side * side)
    table_flat = torch.zeros(*values.shape[:-1], table_side * table_side, device=device, dtype=dtype)
    counts = torch.zeros(table_side * table_side, device=device, dtype=dtype)
    table_flat.scatter_add_(-1, flat_index.expand(*values.shape[:-1], -1), values)
    counts.scatter_add_(0, flat_index, torch.ones_like(flat_index, dtype=dtype))
    table_flat = table_flat / counts.clamp_min(1.0)
    return table_flat.reshape(*values.shape[:-1], table_side, table_side)


def relative_table_to_pairwise(table: torch.Tensor, side: int) -> torch.Tensor:
    """Gather [*, dx, dy] table values back to [*, query, key]."""
    device = table.device
    dtype = table.dtype
    deltas = pairwise_deltas(side, device=device, dtype=dtype).to(torch.long)
    table_side = 2 * side - 1
    flat_index = (deltas[:, 0] + side - 1) * table_side + (deltas[:, 1] + side - 1)
    flat = table.reshape(*table.shape[:-2], table_side * table_side)
    values = torch.gather(flat, -1, flat_index.expand(*flat.shape[:-1], -1))
    return values.reshape(*table.shape[:-2], side * side, side * side)


def coeff_to_relative_tables(
    basis_matrix: torch.Tensor,
    coeff: torch.Tensor,
    *,
    side: int,
) -> RelativeGeometryBundle:
    n_positions = side * side
    pairwise = coeff_to_pairwise_bias(basis_matrix, coeff, n_positions)
    tables = pairwise_to_relative_table(pairwise, side)
    axis_tables = axial_projection(tables)
    residual = tables - axis_tables
    vals = torch.arange(-(side - 1), side, device=tables.device, dtype=tables.dtype)
    return RelativeGeometryBundle(
        tables=tables,
        axis_tables=axis_tables,
        residual_tables=residual,
        dx_values=vals,
        dy_values=vals,
    )


def axial_projection(table: torch.Tensor) -> torch.Tensor:
    """Best full axial-additive projection u(dx)+v(dy) on a complete grid."""
    row_mean = table.mean(dim=-1, keepdim=True)
    col_mean = table.mean(dim=-2, keepdim=True)
    overall = table.mean(dim=(-2, -1), keepdim=True)
    return row_mean + col_mean - overall


def gauge_versions(table: torch.Tensor) -> dict[str, torch.Tensor]:
    centered = table - table.mean(dim=(-2, -1), keepdim=True)
    norm = torch.linalg.norm(centered.reshape(*centered.shape[:-2], -1), dim=-1).clamp_min(EPS)
    normalized = centered / norm[..., None, None]
    return {"raw": table, "centered": centered, "normalized": normalized}


def _hessian_terms(table: torch.Tensor, boundary: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if boundary == "interior_only":
        hxx_full = table[..., 2:, :] - 2.0 * table[..., 1:-1, :] + table[..., :-2, :]
        hyy_full = table[..., :, 2:] - 2.0 * table[..., :, 1:-1] + table[..., :, :-2]
        hxx = table[..., 2:, 1:-1] - 2.0 * table[..., 1:-1, 1:-1] + table[..., :-2, 1:-1]
        hyy = table[..., 1:-1, 2:] - 2.0 * table[..., 1:-1, 1:-1] + table[..., 1:-1, :-2]
        hxy = (table[..., 2:, 2:] - table[..., 2:, :-2] - table[..., :-2, 2:] + table[..., :-2, :-2]) / 4.0
        lap = hxx + hyy
        return hxx_full, hyy_full, hxy, lap
    if boundary != "reflect_padding":
        raise ValueError(f"unknown boundary mode: {boundary}")
    shape = table.shape
    padded = F.pad(table.reshape(-1, shape[-2], shape[-1]).unsqueeze(1), (1, 1, 1, 1), mode="reflect").squeeze(1)
    padded = padded.reshape(*shape[:-2], shape[-2] + 2, shape[-1] + 2)
    center = padded[..., 1:-1, 1:-1]
    hxx = padded[..., 2:, 1:-1] - 2.0 * center + padded[..., :-2, 1:-1]
    hyy = padded[..., 1:-1, 2:] - 2.0 * center + padded[..., 1:-1, :-2]
    hxy = (padded[..., 2:, 2:] - padded[..., 2:, :-2] - padded[..., :-2, 2:] + padded[..., :-2, :-2]) / 4.0
    lap = hxx + hyy
    return hxx, hyy, hxy, lap


def _energy(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum(dim=(-2, -1))


def dct_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(size, device=device, dtype=dtype)
    k = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    mat = torch.cos(math.pi / float(size) * (n + 0.5) * k)
    mat[0] *= math.sqrt(1.0 / float(size))
    if size > 1:
        mat[1:] *= math.sqrt(2.0 / float(size))
    return mat


def dct2(table: torch.Tensor) -> torch.Tensor:
    size = table.shape[-1]
    mat = dct_matrix(size, device=table.device, dtype=table.dtype)
    flat = table.reshape(-1, size, size)
    out = torch.matmul(mat, torch.matmul(flat, mat.T))
    return out.reshape(*table.shape)


def spectral_metrics(table: torch.Tensor, *, topk: int = 5) -> dict[str, float]:
    centered = table - table.mean()
    fft = torch.fft.fft2(centered)
    power = fft.abs().square()
    total_with_dc = power.sum().clamp_min(EPS)
    dc = power[0, 0]
    power_no_dc = power.clone()
    power_no_dc[0, 0] = 0.0
    total = power_no_dc.sum().clamp_min(EPS)
    probs = (power_no_dc.reshape(-1) / total).clamp_min(EPS)
    entropy = -torch.sum(probs * torch.log(probs)) / math.log(float(max(2, probs.numel() - 1)))
    k = min(int(topk), probs.numel())
    top_mass = torch.topk(power_no_dc.reshape(-1), k=k).values.sum() / total

    dct_power = dct2(centered).square()
    dct_power_no_dc = dct_power.clone()
    dct_power_no_dc[0, 0] = 0.0
    dct_total = dct_power_no_dc.sum().clamp_min(EPS)
    top_dct = torch.topk(dct_power_no_dc.reshape(-1), k=k).values.sum() / dct_total

    freq = torch.fft.fftfreq(table.shape[-1], device=table.device, dtype=table.dtype)
    fx, fy = torch.meshgrid(freq, freq, indexing="ij")
    nonzero = (fx != 0) | (fy != 0)
    oblique = (fx != 0) & (fy != 0)
    low = (torch.sqrt(fx.square() + fy.square()) <= 0.20) & nonzero
    mid = (torch.sqrt(fx.square() + fy.square()) > 0.20) & (torch.sqrt(fx.square() + fy.square()) <= 0.35)
    high = (torch.sqrt(fx.square() + fy.square()) > 0.35)

    flipped = torch.flip(torch.flip(centered, dims=(-1,)), dims=(-2,))
    even = 0.5 * (centered + flipped)
    odd = 0.5 * (centered - flipped)
    even_energy = even.square().sum()
    odd_energy = odd.square().sum()
    signal_energy = centered.square().sum().clamp_min(EPS)

    conj_error = torch.max(torch.abs(fft - torch.conj(torch.flip(torch.flip(fft, dims=(-1,)), dims=(-2,)))))
    fft_scale = torch.max(torch.abs(fft)).clamp_min(EPS)
    return {
        "dc_mass_ratio": float((dc / total_with_dc).detach().cpu()),
        "spectral_entropy": float(entropy.detach().cpu()),
        "topk_spectral_mass": float(top_mass.detach().cpu()),
        "topk_dct_mass": float(top_dct.detach().cpu()),
        "diagonal_atom_ratio": float((power_no_dc[oblique].sum() / total).detach().cpu()),
        "low_radial_mass": float((power_no_dc[low].sum() / total).detach().cpu()),
        "mid_radial_mass": float((power_no_dc[mid].sum() / total).detach().cpu()),
        "high_radial_mass": float((power_no_dc[high].sum() / total).detach().cpu()),
        "even_spectral_mass": float((even_energy / signal_energy).detach().cpu()),
        "odd_spectral_mass": float((odd_energy / signal_energy).detach().cpu()),
        "complex_pair_consistency": float((1.0 - conj_error / fft_scale).detach().cpu()),
    }


def geometry_metrics_for_table(
    table: torch.Tensor,
    *,
    gauge: str,
    boundary: str,
    topk: int = 5,
) -> dict[str, float]:
    axis = axial_projection(table)
    residual = table - axis
    hxx, hyy, hxy, lap = _hessian_terms(table, boundary)
    e_xx = _energy(hxx)
    e_yy = _energy(hyy)
    e_xy = _energy(hxy)
    e_lap = _energy(lap)
    norm = table.square().sum().clamp_min(EPS)
    e_axial = axis.square().sum()
    e_obl = residual.square().sum()
    out = {
        "gauge": gauge,
        "boundary": boundary,
        "bias_mean": float(table.mean().detach().cpu()),
        "bias_std": float(table.std(unbiased=False).detach().cpu()),
        "bias_norm": float(torch.linalg.norm(table).detach().cpu()),
        "E_xx": float(e_xx.detach().cpu()),
        "E_yy": float(e_yy.detach().cpu()),
        "E_xy": float(e_xy.detach().cpu()),
        "E_lap": float(e_lap.detach().cpu()),
        "mixed_ratio": float((e_xy / (e_xx + e_yy + e_xy).clamp_min(EPS)).detach().cpu()),
        "E_axial": float(e_axial.detach().cpu()),
        "E_obl": float(e_obl.detach().cpu()),
        "obl_ratio": float((e_obl / norm).detach().cpu()),
    }
    out.update(spectral_metrics(table, topk=topk))
    return out


def geometry_rows(
    tables: torch.Tensor,
    *,
    basis: str,
    dataset: str,
    task: str,
    seed: int,
    topk: int = 5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer in range(tables.shape[0]):
        for head in range(tables.shape[1]):
            versions = gauge_versions(tables[layer, head])
            for gauge, table in versions.items():
                for boundary in ["interior_only", "reflect_padding"]:
                    metrics = geometry_metrics_for_table(table, gauge=gauge, boundary=boundary, topk=topk)
                    rows.append(
                        {
                            "dataset": dataset,
                            "task": task,
                            "basis": basis,
                            "seed": seed,
                            "layer": layer,
                            "head": head,
                            **metrics,
                        }
                    )
    return rows


def aggregate_geometry_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_keys = [
        key
        for key in rows[0].keys()
        if key not in {"dataset", "task", "basis", "seed", "layer", "head", "gauge", "boundary"}
        and isinstance(rows[0][key], (int, float))
    ] if rows else []
    groups: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["dataset"]), str(row["task"]), str(row["basis"]), str(row["gauge"]), str(row["boundary"])),
            [],
        ).append(row)
    out: list[dict[str, object]] = []
    for (dataset, task, basis, gauge, boundary), values in sorted(groups.items()):
        item: dict[str, object] = {
            "dataset": dataset,
            "task": task,
            "basis": basis,
            "gauge": gauge,
            "boundary": boundary,
            "n": len(values),
        }
        for key in numeric_keys:
            vals = np.array([float(row[key]) for row in values], dtype=np.float64)
            item[f"{key}_mean"] = float(vals.mean())
            item[f"{key}_std"] = float(vals.std())
        out.append(item)
    return out


def top_spectral_peaks(
    table: torch.Tensor,
    *,
    basis: str,
    dataset: str,
    task: str,
    seed: int,
    layer: int,
    head: int,
    topk: int = 8,
) -> list[dict[str, object]]:
    centered = table - table.mean()
    fft = torch.fft.fft2(centered)
    power = fft.abs().square()
    power[0, 0] = 0.0
    vals, idx = torch.topk(power.reshape(-1), k=min(topk, power.numel()))
    freq = torch.fft.fftfreq(table.shape[-1], device=table.device, dtype=table.dtype) * (2.0 * math.pi)
    rows: list[dict[str, object]] = []
    total = power.sum().clamp_min(EPS)
    for rank, (value, flat_idx) in enumerate(zip(vals, idx), start=1):
        ix = int(flat_idx // table.shape[-1])
        iy = int(flat_idx % table.shape[-1])
        coeff = fft[ix, iy]
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "basis": basis,
                "seed": seed,
                "layer": layer,
                "head": head,
                "rank": rank,
                "omega_x": float(freq[ix].detach().cpu()),
                "omega_y": float(freq[iy].detach().cpu()),
                "power": float(value.detach().cpu()),
                "mass": float((value / total).detach().cpu()),
                "phase": float(torch.angle(coeff).detach().cpu()),
            }
        )
    return rows


def topk_fft_reconstruction(table: torch.Tensor, *, topk: int = 5) -> torch.Tensor:
    centered = table - table.mean(dim=(-2, -1), keepdim=True)
    shape = centered.shape
    flat = centered.reshape(-1, shape[-2], shape[-1])
    recon = []
    for item in flat:
        fft = torch.fft.fft2(item)
        power = fft.abs().square()
        power[0, 0] = 0.0
        keep = torch.zeros_like(power, dtype=torch.bool)
        _, idx = torch.topk(power.reshape(-1), k=min(topk, power.numel()))
        keep.reshape(-1)[idx] = True
        # Keep conjugate partners implicitly by adding the flipped index mask.
        keep = keep | torch.flip(torch.flip(keep, dims=(-1,)), dims=(-2,))
        filtered = torch.where(keep, fft, torch.zeros_like(fft))
        recon.append(torch.fft.ifft2(filtered).real + item.mean())
    return torch.stack(recon, dim=0).reshape_as(table)


def spectral_peak_rows(
    tables: torch.Tensor,
    *,
    basis: str,
    dataset: str,
    task: str,
    seed: int,
    topk: int = 8,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer in range(tables.shape[0]):
        for head in range(tables.shape[1]):
            rows.extend(
                top_spectral_peaks(
                    tables[layer, head],
                    basis=basis,
                    dataset=dataset,
                    task=task,
                    seed=seed,
                    layer=layer,
                    head=head,
                    topk=topk,
                )
            )
    return rows


def save_npz(
    output_dir: Path,
    bundle: RelativeGeometryBundle,
    *,
    metadata: dict[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "bias_tables.npz"
    np.savez_compressed(
        path,
        tables=bundle.tables.detach().cpu().numpy(),
        axis_tables=bundle.axis_tables.detach().cpu().numpy(),
        residual_tables=bundle.residual_tables.detach().cpu().numpy(),
        dx_values=bundle.dx_values.detach().cpu().numpy(),
        dy_values=bundle.dy_values.detach().cpu().numpy(),
        metadata=np.array([metadata], dtype=object),
    )
    return path


def _save_heatmap(path: Path, matrices: list[tuple[str, torch.Tensor]], *, cmap: str = "coolwarm") -> None:
    if not matrices:
        return
    cols = min(4, len(matrices))
    rows = int(math.ceil(len(matrices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, (title, mat) in zip(axes.reshape(-1), matrices):
        arr = mat.detach().cpu().numpy()
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("on")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_plots(output_dir: Path, tables: torch.Tensor, residual: torch.Tensor, *, max_items: int = 8) -> None:
    matrices: list[tuple[str, torch.Tensor]] = []
    residual_mats: list[tuple[str, torch.Tensor]] = []
    fft_mats: list[tuple[str, torch.Tensor]] = []
    dct_mats: list[tuple[str, torch.Tensor]] = []
    count = 0
    for layer in range(tables.shape[0]):
        for head in range(tables.shape[1]):
            if count >= max_items:
                break
            table = tables[layer, head] - tables[layer, head].mean()
            _, _, hxy, _ = _hessian_terms(table, "interior_only")
            matrices.append((f"L{layer} H{head} Hxy", hxy))
            residual_mats.append((f"L{layer} H{head} residual", residual[layer, head]))
            fft_mats.append((f"L{layer} H{head} FFT", torch.fft.fftshift(torch.fft.fft2(table).abs().log1p())))
            dct_mats.append((f"L{layer} H{head} DCT", dct2(table).abs().log1p()))
            count += 1
        if count >= max_items:
            break
    _save_heatmap(output_dir / "mixed_hessian_heatmaps.png", matrices)
    _save_heatmap(output_dir / "axial_residual_heatmaps.png", residual_mats)
    _save_heatmap(output_dir / "fft_spectrum_grid.png", fft_mats, cmap="magma")
    _save_heatmap(output_dir / "dct_spectrum_grid.png", dct_mats, cmap="magma")


def write_geometry_report(output_dir: Path, metadata: dict[str, object], aggregate_rows: list[dict[str, object]]) -> None:
    lines = [
        "# V4 Relative Table Geometry Report",
        "",
        "Metadata:",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Aggregate Geometry",
            "",
            "| basis | gauge | boundary | n | obl ratio | mixed ratio | spectral entropy | top-5 FFT mass | top-5 DCT mass |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| "
            + f"{row['basis']} | {row['gauge']} | {row['boundary']} | {int(row['n'])} | "
            + f"{float(row.get('obl_ratio_mean', 0.0)):.4f} | "
            + f"{float(row.get('mixed_ratio_mean', 0.0)):.4f} | "
            + f"{float(row.get('spectral_entropy_mean', 0.0)):.4f} | "
            + f"{float(row.get('topk_spectral_mass_mean', 0.0)):.4f} | "
            + f"{float(row.get('topk_dct_mass_mean', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Geometry is reported for raw, centered, and normalized gauges.",
            "- Main-text curvature should use `interior_only`; `reflect_padding` is a sanity check.",
            "- `obl_ratio` is the energy share outside the best full axial-additive projection.",
            "",
            "Artifacts:",
            "",
            "- `bias_tables.npz`",
            "- `geometry_metrics.csv`",
            "- `geometry_aggregate.csv`",
            "- `spectral_peaks.csv`",
            "- `mixed_hessian_heatmaps.png`",
            "- `fft_spectrum_grid.png`",
            "- `dct_spectrum_grid.png`",
            "- `axial_residual_heatmaps.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_geometry_artifacts(
    output_dir: Path,
    *,
    basis_matrix: torch.Tensor,
    coeff: torch.Tensor,
    side: int,
    metadata: dict[str, object],
    topk_metrics: int = 5,
    topk_peaks: int = 8,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = coeff_to_relative_tables(basis_matrix, coeff, side=side)
    npz_path = save_npz(output_dir, bundle, metadata=metadata)
    rows = geometry_rows(
        bundle.tables,
        basis=str(metadata.get("basis", "unknown")),
        dataset=str(metadata.get("dataset", "unknown")),
        task=str(metadata.get("task", "unknown")),
        seed=int(metadata.get("seed", -1)),
        topk=topk_metrics,
    )
    aggregate_rows = aggregate_geometry_rows(rows)
    peaks = spectral_peak_rows(
        bundle.tables,
        basis=str(metadata.get("basis", "unknown")),
        dataset=str(metadata.get("dataset", "unknown")),
        task=str(metadata.get("task", "unknown")),
        seed=int(metadata.get("seed", -1)),
        topk=topk_peaks,
    )
    write_csv(output_dir / "geometry_metrics.csv", rows)
    write_csv(output_dir / "geometry_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "spectral_peaks.csv", peaks)
    save_plots(output_dir, bundle.tables, bundle.residual_tables)
    write_geometry_report(output_dir, metadata, aggregate_rows)
    return {
        "bias_table_npz": str(npz_path),
        "geometry_metrics": str(output_dir / "geometry_metrics.csv"),
        "geometry_aggregate": str(output_dir / "geometry_aggregate.csv"),
        "spectral_peaks": str(output_dir / "spectral_peaks.csv"),
        "report": str(output_dir / "REPORT.md"),
    }


def load_bias_npz(path: Path, *, device: torch.device | None = None) -> tuple[RelativeGeometryBundle, dict[str, object]]:
    arr = np.load(path, allow_pickle=True)
    device = device or torch.device("cpu")
    tables = torch.tensor(arr["tables"], device=device, dtype=torch.float32)
    axis = torch.tensor(arr["axis_tables"], device=device, dtype=torch.float32)
    residual = torch.tensor(arr["residual_tables"], device=device, dtype=torch.float32)
    dx_values = torch.tensor(arr["dx_values"], device=device, dtype=torch.float32)
    dy_values = torch.tensor(arr["dy_values"], device=device, dtype=torch.float32)
    metadata = dict(arr["metadata"][0]) if "metadata" in arr else {}
    return RelativeGeometryBundle(tables, axis, residual, dx_values, dy_values), metadata


def fit_linear_basis(matrix: torch.Tensor, target: torch.Tensor, *, ridge: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    target_vec = target.reshape(-1, 1).to(matrix.device, matrix.dtype)
    mat, _ = normalize_columns(matrix)
    mat64 = mat.to(torch.float64)
    target64 = target_vec.to(torch.float64)
    try:
        coeff64 = torch.linalg.pinv(mat64, rtol=max(ridge, 1e-12)) @ target64
    except torch.linalg.LinAlgError:
        coeff64 = torch.linalg.lstsq(mat64, target64, rcond=max(ridge, 1e-10)).solution
    if not torch.isfinite(coeff64).all():
        coeff64 = torch.linalg.pinv(mat64, rtol=max(ridge, 1e-8)) @ target64
    pred64 = mat64 @ coeff64
    if not torch.isfinite(pred64).all():
        coeff64 = torch.zeros(mat64.shape[1], 1, device=mat64.device, dtype=mat64.dtype)
        pred64 = mat64 @ coeff64
    pred = pred64.reshape_as(target).to(target.dtype)
    coeff = coeff64.to(matrix.dtype)
    mse = torch.mean((pred64.reshape_as(target64.reshape_as(target).to(torch.float64)) - target.to(torch.float64)).square())
    var = torch.mean((target.to(torch.float64) - target.to(torch.float64).mean()).square()).clamp_min(EPS)
    r2 = 1.0 - mse / var
    r2_value = float(r2.detach().cpu()) if torch.isfinite(r2) else float("nan")
    return pred, coeff, float(mse.detach().cpu()), r2_value


def full_axial_basis(side: int, *, device: torch.device, dtype: torch.dtype) -> Basis:
    d = relative_grid(side, device=device, dtype=dtype)
    vals = torch.arange(-(side - 1), side, device=device, dtype=dtype)
    cols: list[torch.Tensor] = []
    labels: list[str] = []
    for value in vals:
        cols.append((d[:, 0] == value).to(dtype))
        labels.append(f"dx_{int(value.item())}")
    for value in vals[:-1]:
        cols.append((d[:, 1] == value).to(dtype))
        labels.append(f"dy_{int(value.item())}")
    matrix = torch.stack(cols, dim=1)
    return Basis("axis_full_29", matrix, labels, [0] * len(labels))


def axis_lowrank_basis(side: int, *, device: torch.device, dtype: torch.dtype) -> Basis:
    from toric_pj.diagnostics.basis_projection import axis_additive_fourier_basis

    d = relative_grid(side, device=device, dtype=dtype)
    return axis_additive_fourier_basis(d, [0.78, 0.58], name="axis_lowrank_5")


def _canonical_fft_frequencies(size: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    freq = torch.fft.fftfreq(size, device=device, dtype=dtype) * (2.0 * math.pi)
    fx, fy = torch.meshgrid(freq, freq, indexing="ij")
    ix = torch.arange(size, device=device)
    iy = torch.arange(size, device=device)
    gx, gy = torch.meshgrid(ix, iy, indexing="ij")
    canonical = ((gx > 0) | ((gx == 0) & (gy > 0))) & ~((gx == 0) & (gy == 0))
    return fx, fy, canonical


def top_dft_omegas(table: torch.Tensor, *, k: int, window: bool = False) -> torch.Tensor:
    centered = table - table.mean()
    if window:
        size = table.shape[-1]
        win = torch.hann_window(size, periodic=False, device=table.device, dtype=table.dtype)
        centered = centered * win[:, None] * win[None, :]
    fft = torch.fft.fft2(centered)
    power = fft.abs().square()
    fx, fy, canonical = _canonical_fft_frequencies(table.shape[-1], device=table.device, dtype=table.dtype)
    masked = power.masked_fill(~canonical, -1.0)
    vals, idx = torch.topk(masked.reshape(-1), k=max(1, min(k, int(canonical.sum().item()))))
    del vals
    ix = idx // table.shape[-1]
    iy = idx % table.shape[-1]
    return torch.stack([fx[ix, iy], fy[ix, iy]], dim=1)


def random_matched_radius_omegas(table: torch.Tensor, *, k: int, seed: int = 0) -> torch.Tensor:
    top = top_dft_omegas(table, k=k)
    fx, fy, canonical = _canonical_fft_frequencies(table.shape[-1], device=table.device, dtype=table.dtype)
    candidates = torch.stack([fx[canonical], fy[canonical]], dim=1)
    radii = torch.linalg.norm(candidates, dim=1)
    top_radii = torch.linalg.norm(top, dim=1)
    gen = torch.Generator(device=table.device)
    gen.manual_seed(seed)
    chosen: list[torch.Tensor] = []
    for radius in top_radii:
        nearest = torch.argsort(torch.abs(radii - radius))[: max(4, min(16, candidates.shape[0]))]
        pick = nearest[torch.randint(0, nearest.numel(), (1,), device=table.device, generator=gen)]
        chosen.append(candidates[pick].reshape(2))
    return torch.stack(chosen, dim=0) if chosen else candidates[:0]


def fourier_atom_basis(side: int, omegas: torch.Tensor, *, name: str) -> Basis:
    d = relative_grid(side, device=omegas.device, dtype=omegas.dtype)
    return toric_fourier_basis(d, list(omegas), name=name)


def table_informed_pj_basis(side: int, omegas: torch.Tensor, *, order: int, name: str) -> Basis:
    d = relative_grid(side, device=omegas.device, dtype=omegas.dtype)
    ex = torch.tensor([1.0, 0.0], device=omegas.device, dtype=omegas.dtype)
    ey = torch.tensor([0.0, 1.0], device=omegas.device, dtype=omegas.dtype)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=omegas.device, dtype=omegas.dtype)).reshape(-1)
    oblique = normalize_direction(torch.tensor([[1.0, -0.65]], device=omegas.device, dtype=omegas.dtype)).reshape(-1)
    orders = list(range(order + 1))
    return directional_jet_basis(d, list(omegas), [ex, ey, diag, oblique], orders, scale=float(side), name=name)


def fixed_toric_pj_basis(side: int, *, device: torch.device, dtype: torch.dtype) -> Basis:
    omega_a = torch.tensor([0.78, 0.42], device=device, dtype=dtype)
    omega_b = torch.tensor([0.62, -0.58], device=device, dtype=dtype)
    omega_c = torch.tensor([0.35, 0.91], device=device, dtype=dtype)
    return table_informed_pj_basis(side, torch.stack([omega_a, omega_b, omega_c]), order=2, name="fixed_toric_PJ_R2")
