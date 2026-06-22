from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import Basis, normalize_columns
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.diagnostics.relative_table_geometry import axial_projection, dct2, top_dft_omegas


EPS = 1e-12


@dataclass
class LinearFit:
    ridge: float
    mse: float
    r2: float
    coeff_norm: float
    condition_number: float
    effective_rank: float
    singular_values: list[float]
    coeff: torch.Tensor
    pred: torch.Tensor
    column_norms: torch.Tensor


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def signed_grid(radius: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vals = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(vals, vals, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


def rectangular_mask(d: torch.Tensor, radius: int) -> torch.Tensor:
    return (d[:, 0].abs() <= radius) & (d[:, 1].abs() <= radius)


def shell_mask(d: torch.Tensor, inner_radius: float, outer_radius: float | None = None) -> torch.Tensor:
    r = torch.linalg.norm(d, dim=1)
    mask = r > float(inner_radius)
    if outer_radius is not None:
        mask = mask & (r <= float(outer_radius))
    return mask


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.reshape(-1).to(torch.float64)
    target = target.reshape(-1).to(torch.float64)
    mse = torch.mean((pred - target).square())
    var = torch.mean((target - target.mean()).square()).clamp_min(EPS)
    return float((1.0 - mse / var).detach().cpu())


def rms(pred: torch.Tensor, target: torch.Tensor | None = None) -> float:
    pred = pred.reshape(-1).to(torch.float64)
    if target is None:
        return float(torch.sqrt(torch.mean(pred.square())).detach().cpu())
    target = target.reshape(-1).to(torch.float64)
    return float(torch.sqrt(torch.mean((pred - target).square())).detach().cpu())


def max_abs(x: torch.Tensor) -> float:
    return float(torch.max(torch.abs(x.reshape(-1))).detach().cpu())


def singular_diagnostics(matrix: torch.Tensor) -> tuple[float, float, list[float]]:
    if matrix.numel() == 0 or matrix.shape[1] == 0:
        return float("nan"), float("nan"), []
    mat, _ = normalize_columns(matrix.to(torch.float64))
    try:
        s = torch.linalg.svdvals(mat)
    except torch.linalg.LinAlgError:
        return float("inf"), float("nan"), []
    s_cpu = [float(v.detach().cpu()) for v in s]
    s_pos = s.clamp_min(EPS)
    condition = float((s_pos.max() / s_pos.min()).detach().cpu())
    probs = s_pos / s_pos.sum().clamp_min(EPS)
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(EPS)))
    effective_rank = float(torch.exp(entropy).detach().cpu())
    return condition, effective_rank, s_cpu


def fit_linear(train_matrix: torch.Tensor, train_target: torch.Tensor, *, ridge: float) -> LinearFit:
    mat, norms = normalize_columns(train_matrix.to(torch.float64))
    target = train_target.reshape(-1, 1).to(torch.float64)
    gram = mat.T @ mat
    rhs = mat.T @ target
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    try:
        coeff = torch.linalg.solve(gram + float(ridge) * eye, rhs)
    except torch.linalg.LinAlgError:
        coeff = torch.linalg.lstsq(gram + float(ridge) * eye, rhs).solution
    pred = mat @ coeff
    mse = torch.mean((pred - target).square())
    r2 = r2_score(pred.reshape(-1), target.reshape(-1))
    cond, erank, s = singular_diagnostics(train_matrix)
    return LinearFit(
        ridge=float(ridge),
        mse=float(mse.detach().cpu()),
        r2=r2,
        coeff_norm=float(torch.linalg.norm(coeff).detach().cpu()),
        condition_number=cond,
        effective_rank=erank,
        singular_values=s,
        coeff=coeff,
        pred=pred.reshape(-1),
        column_norms=norms.to(torch.float64),
    )


def predict(matrix: torch.Tensor, coeff: torch.Tensor, column_norms: torch.Tensor | None = None) -> torch.Tensor:
    if column_norms is None:
        mat, _ = normalize_columns(matrix.to(torch.float64))
    else:
        norms = column_norms.to(device=matrix.device, dtype=torch.float64).clamp_min(EPS)
        mat = matrix.to(torch.float64) / norms
    return (mat @ coeff.to(torch.float64)).reshape(-1)


def fit_with_ridge_grid(
    train_matrix: torch.Tensor,
    train_target: torch.Tensor,
    *,
    ridge_grid: Sequence[float],
) -> tuple[LinearFit, list[dict[str, object]]]:
    path: list[dict[str, object]] = []
    best: LinearFit | None = None
    for ridge in ridge_grid:
        fit = fit_linear(train_matrix, train_target, ridge=float(ridge))
        path.append(
            {
                "ridge": float(ridge),
                "train_r2": fit.r2,
                "train_mse": fit.mse,
                "coeff_norm": fit.coeff_norm,
                "condition_number": fit.condition_number,
                "effective_rank": fit.effective_rank,
            }
        )
        if best is None or fit.r2 > best.r2 + 1e-10 or (
            abs(fit.r2 - best.r2) <= 1e-10 and fit.coeff_norm < best.coeff_norm
        ):
            best = fit
    assert best is not None
    return best, path


def phase_coordinates(d: torch.Tensor, *, chart: str, scale: float) -> torch.Tensor:
    if chart == "raw":
        return d
    if chart in {"asinh", "lc"}:
        return float(scale) * torch.asinh(d / float(scale))
    if chart == "log":
        return torch.sign(d) * float(scale) * torch.log1p(torch.abs(d) / float(scale))
    raise ValueError(f"unknown phase chart: {chart}")


def jet_coordinates(d: torch.Tensor, *, chart: str, scale: float) -> torch.Tensor:
    if chart == "raw":
        return d / float(scale)
    if chart in {"bounded", "asinh", "lc"}:
        return torch.asinh(d / float(scale))
    if chart == "tanh":
        return torch.tanh(d / float(scale))
    raise ValueError(f"unknown jet chart: {chart}")


def axis_j0_basis(d: torch.Tensor, *, omega: torch.Tensor | None = None, name: str = "axis_j0") -> Basis:
    if omega is None:
        omega = torch.tensor([0.73, 0.51], device=d.device, dtype=d.dtype)
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    for axis in [0, 1]:
        ph = d[:, axis] * omega[axis].to(d.device, d.dtype)
        cols.extend([torch.cos(ph), torch.sin(ph)])
        labels.extend([f"axis{axis}_cos", f"axis{axis}_sin"])
    return Basis(name, torch.stack(cols, dim=1), labels, [0] * len(labels))


def toric_j0_basis(
    d: torch.Tensor,
    omegas: torch.Tensor,
    *,
    phase_chart: str = "raw",
    chart_scale: float = 1.0,
    include_const: bool = True,
    name: str = "toric_j0",
) -> Basis:
    coords = phase_coordinates(d, chart=phase_chart, scale=chart_scale)
    cols: list[torch.Tensor] = []
    labels: list[str] = []
    orders: list[int] = []
    if include_const:
        cols.append(torch.ones(d.shape[0], device=d.device, dtype=d.dtype))
        labels.append("const")
        orders.append(0)
    for idx, omega in enumerate(omegas.to(d.device, d.dtype)):
        ph = coords @ omega
        cols.extend([torch.cos(ph), torch.sin(ph)])
        labels.extend([f"w{idx}_j0_cos", f"w{idx}_j0_sin"])
        orders.extend([0, 0])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def full_multijet_basis(
    d: torch.Tensor,
    omegas: torch.Tensor,
    *,
    order: int,
    scale: float,
    phase_chart: str = "raw",
    jet_chart: str = "raw",
    include_const: bool = True,
    name: str = "full_multijet",
) -> Basis:
    phase_d = phase_coordinates(d, chart=phase_chart, scale=scale)
    jet_d = jet_coordinates(d, chart=jet_chart, scale=scale)
    cols: list[torch.Tensor] = []
    labels: list[str] = []
    orders: list[int] = []
    if include_const:
        cols.append(torch.ones(d.shape[0], device=d.device, dtype=d.dtype))
        labels.append("const")
        orders.append(0)
    for omega_idx, omega in enumerate(omegas.to(d.device, d.dtype)):
        ph = phase_d @ omega
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)
        for total in range(order + 1):
            for rx in range(total + 1):
                ry = total - rx
                if total == 0:
                    poly = torch.ones_like(cos_ph)
                else:
                    poly = jet_d[:, 0].pow(rx) * jet_d[:, 1].pow(ry)
                cols.extend([poly * cos_ph, poly * sin_ph])
                labels.extend([f"w{omega_idx}_j{rx}{ry}_cos", f"w{omega_idx}_j{rx}{ry}_sin"])
                orders.extend([total, total])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def directional_jet_basis(
    d: torch.Tensor,
    omegas: torch.Tensor,
    *,
    directions: torch.Tensor,
    order: int,
    scale: float,
    phase_chart: str = "raw",
    jet_chart: str = "raw",
    include_const: bool = True,
    name: str = "directional_jet",
) -> Basis:
    phase_d = phase_coordinates(d, chart=phase_chart, scale=scale)
    jet_d = jet_coordinates(d, chart=jet_chart, scale=scale)
    dirs = normalize_direction(directions.to(d.device, d.dtype))
    cols: list[torch.Tensor] = []
    labels: list[str] = []
    orders: list[int] = []
    if include_const:
        cols.append(torch.ones(d.shape[0], device=d.device, dtype=d.dtype))
        labels.append("const")
        orders.append(0)
    for omega_idx, omega in enumerate(omegas.to(d.device, d.dtype)):
        ph = phase_d @ omega
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)
        cols.extend([cos_ph, sin_ph])
        labels.extend([f"w{omega_idx}_j0_cos", f"w{omega_idx}_j0_sin"])
        orders.extend([0, 0])
        for direction_idx, direction in enumerate(dirs):
            coord = jet_d @ direction
            for item_order in range(1, order + 1):
                poly = coord.pow(item_order)
                cols.extend([poly * cos_ph, poly * sin_ph])
                labels.extend(
                    [
                        f"w{omega_idx}_u{direction_idx}_j{item_order}_cos",
                        f"w{omega_idx}_u{direction_idx}_j{item_order}_sin",
                    ]
                )
                orders.extend([item_order, item_order])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def default_directions(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return normalize_direction(
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -0.65]],
            device=device,
            dtype=dtype,
        )
    )


def generic_omegas(k: int, *, device: torch.device, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    if k <= 0:
        return torch.empty(0, 2, device=device, dtype=dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=k + 1, device=device, dtype=dtype)[:-1]
    angles = angles + 0.17
    radii_base = torch.tensor([0.42, 0.67, 0.93, 1.18, 1.43], device=device, dtype=dtype)
    radii = radii_base[torch.arange(k, device=device) % radii_base.numel()]
    jitter = 0.03 * torch.randn(k, 2, generator=gen, device=device, dtype=dtype)
    out = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1) + jitter
    axis_like = (out[:, 0].abs() < 0.05) | (out[:, 1].abs() < 0.05)
    out[axis_like, 0] += 0.11
    out[axis_like, 1] += 0.07
    return out


def random_matched_omegas(reference: torch.Tensor, *, seed: int) -> torch.Tensor:
    if reference.numel() == 0:
        return reference.clone()
    device, dtype = reference.device, reference.dtype
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    radii = torch.linalg.norm(reference, dim=1)
    angles = 2.0 * math.pi * torch.rand(reference.shape[0], generator=gen, device=device, dtype=dtype)
    out = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)
    return out


def shuffled_omegas(reference: torch.Tensor, *, seed: int) -> torch.Tensor:
    if reference.shape[0] <= 1:
        return -reference
    gen = torch.Generator(device=reference.device)
    gen.manual_seed(int(seed))
    perm = torch.randperm(reference.shape[0], generator=gen, device=reference.device)
    out = reference[perm].clone()
    signs = torch.where(torch.arange(out.shape[0], device=out.device)[:, None] % 2 == 0, 1.0, -1.0).to(out.dtype)
    return out * signs


def top_dct_indices(table: torch.Tensor, *, k: int) -> torch.Tensor:
    coeff = dct2(table - table.mean()).square()
    coeff[0, 0] = 0.0
    vals, idx = torch.topk(coeff.reshape(-1), k=min(max(1, k), coeff.numel()))
    del vals
    return torch.stack([idx // table.shape[-1], idx % table.shape[-1]], dim=1)


def dct_basis_from_indices(
    d: torch.Tensor,
    *,
    train_radius: int,
    indices: torch.Tensor,
    include_const: bool = True,
    name: str = "dct_matched",
) -> Basis:
    size = 2 * int(train_radius) + 1
    x = d[:, 0] + float(train_radius)
    y = d[:, 1] + float(train_radius)
    cols: list[torch.Tensor] = []
    labels: list[str] = []
    if include_const:
        cols.append(torch.ones(d.shape[0], device=d.device, dtype=d.dtype))
        labels.append("const")
    for rank, (kx, ky) in enumerate(indices.tolist(), start=1):
        col = torch.cos(math.pi / float(size) * (x + 0.5) * int(kx)) * torch.cos(
            math.pi / float(size) * (y + 0.5) * int(ky)
        )
        cols.append(col)
        labels.append(f"dct{rank}_kx{int(kx)}_ky{int(ky)}")
    return Basis(name, torch.stack(cols, dim=1), labels, [0] * len(labels))


def table_target(table: torch.Tensor, fit_target: str) -> tuple[torch.Tensor, torch.Tensor]:
    axis = axial_projection(table)
    residual = table - axis
    if fit_target == "full_table":
        return table, torch.zeros_like(table)
    if fit_target == "oblique_residual":
        return residual, torch.zeros_like(table)
    if fit_target == "axis_plus_residual":
        return residual, axis
    raise ValueError(f"unknown fit target: {fit_target}")


def crop_visible_table(table: torch.Tensor, visible_radius: int) -> torch.Tensor:
    center = table.shape[-1] // 2
    r = int(visible_radius)
    return table[..., center - r : center + r + 1, center - r : center + r + 1]


def select_omegas_from_source(
    source: str,
    target_table: torch.Tensor,
    *,
    k: int,
    seed: int,
) -> torch.Tensor:
    device, dtype = target_table.device, target_table.dtype
    if k <= 0:
        return torch.empty(0, 2, device=device, dtype=dtype)
    if source in {"generic", "fixed_grid", "fixed"}:
        return generic_omegas(k, device=device, dtype=dtype, seed=seed)
    if source in {"table_informed", "top_dft"}:
        return top_dft_omegas(target_table, k=k)
    if source == "random_matched":
        return random_matched_omegas(top_dft_omegas(target_table, k=k), seed=seed)
    if source in {"table_informed_shuffled", "shuffled"}:
        return shuffled_omegas(top_dft_omegas(target_table, k=k), seed=seed)
    raise ValueError(f"unknown frequency source: {source}")


def omegas_json(omegas: torch.Tensor) -> str:
    return json.dumps([[float(x), float(y)] for x, y in omegas.detach().cpu().tolist()])


def summarize_groups(
    rows: list[dict[str, object]],
    *,
    keys: Sequence[str],
    numeric: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    out: list[dict[str, object]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        item: dict[str, object] = {key: value for key, value in zip(keys, key_values)}
        item["n"] = len(group)
        for name in numeric:
            vals = []
            for row in group:
                try:
                    value = float(row.get(name, "nan"))
                except (TypeError, ValueError):
                    value = float("nan")
                if math.isfinite(value):
                    vals.append(value)
            if vals:
                arr = np.array(vals, dtype=np.float64)
                item[f"{name}_mean"] = float(arr.mean())
                item[f"{name}_std"] = float(arr.std())
                item[f"{name}_min"] = float(arr.min())
                item[f"{name}_max"] = float(arr.max())
            else:
                item[f"{name}_mean"] = float("nan")
                item[f"{name}_std"] = float("nan")
                item[f"{name}_min"] = float("nan")
                item[f"{name}_max"] = float("nan")
        out.append(item)
    return out
