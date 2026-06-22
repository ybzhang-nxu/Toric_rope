from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from toric_pj.diagnostics.basis_projection import Basis, directional_jet_basis, normalize_columns, toric_fourier_basis
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.diagnostics.relative_table_geometry import (
    axial_projection,
    dct2,
    fit_linear_basis,
    load_bias_npz,
    relative_table_to_pairwise,
    write_csv,
)
from toric_pj.experiments.real_digits_probe import make_positions, pairwise_d, relative_2d_table_basis
from toric_pj.experiments.v3_digits_transformer_scaling import PRUNED_REAL_DIGITS_GROUPS, prune_basis
from toric_pj.experiments.v3_real_vision_scaling import (
    VISION_DATASETS,
    aggregate,
    build_patch_bases,
    load_vision_dataset,
    normalize_patches,
    patchify,
    plot_results,
    read_csv,
    train_task,
)
from toric_pj.diagnostics.basis_projection import default_device
from toric_pj.experiments.v12_phase_a_utils import (
    default_directions as v12_default_directions,
    directional_jet_basis as v12_directional_jet_basis,
    full_multijet_basis as v12_full_multijet_basis,
    generic_omegas as v12_generic_omegas,
    random_matched_omegas as v12_random_matched_omegas,
    shuffled_omegas as v12_shuffled_omegas,
    toric_j0_basis as v12_toric_j0_basis,
)


DEFAULT_TEACHER = (
    "results/v4_cifar10_bias_export_10k/bias_exports/"
    "cifar10_reconstruction_relative_2d_table_seed477_steps10000/bias_tables.npz"
)


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _relative_d(side: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(-(side - 1), side, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def _teacher_target(tables: torch.Tensor, target: str) -> torch.Tensor:
    if target == "full":
        return tables
    if target == "residual":
        return tables - axial_projection(tables)
    raise ValueError(f"unknown teacher target: {target}")


def _mean_dct_power(tables: torch.Tensor) -> torch.Tensor:
    centered = tables - tables.mean(dim=(-2, -1), keepdim=True)
    coeff = dct2(centered)
    power = coeff.square().mean(dim=tuple(range(coeff.ndim - 2)))
    power[0, 0] = 0.0
    return power


def top_dct_indices_from_teacher(tables: torch.Tensor, *, k: int) -> torch.Tensor:
    power = _mean_dct_power(tables)
    _, idx = torch.topk(power.reshape(-1), k=min(k, power.numel()))
    return torch.stack([idx // power.shape[-1], idx % power.shape[-1]], dim=1)


def top_dft_omegas_from_teacher(tables: torch.Tensor, *, k: int) -> torch.Tensor:
    centered = tables - tables.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fft2(centered)
    power = fft.abs().square().mean(dim=tuple(range(fft.ndim - 2)))
    size = power.shape[-1]
    freq = torch.fft.fftfreq(size, device=tables.device, dtype=tables.dtype) * (2.0 * math.pi)
    fx, fy = torch.meshgrid(freq, freq, indexing="ij")
    ix = torch.arange(size, device=tables.device)
    iy = torch.arange(size, device=tables.device)
    gx, gy = torch.meshgrid(ix, iy, indexing="ij")
    canonical = ((gx > 0) | ((gx == 0) & (gy > 0))) & ~((gx == 0) & (gy == 0))
    masked = power.masked_fill(~canonical, -1.0)
    _, idx = torch.topk(masked.reshape(-1), k=max(1, min(k, int(canonical.sum().item()))))
    px = idx // size
    py = idx % size
    return torch.stack([fx[px, py], fy[px, py]], dim=1)


def dct_pairwise_basis(d: torch.Tensor, side: int, indices: torch.Tensor, *, name: str) -> Basis:
    size = 2 * side - 1
    x = d[:, 0] + side - 1
    y = d[:, 1] + side - 1
    cols = [torch.ones_like(x)]
    labels = ["const"]
    orders = [0]
    for rank, (kx, ky) in enumerate(indices.detach().cpu().tolist(), start=1):
        col = torch.cos(math.pi / float(size) * (x + 0.5) * int(kx)) * torch.cos(
            math.pi / float(size) * (y + 0.5) * int(ky)
        )
        cols.append(col)
        labels.append(f"dct{rank}_kx{int(kx)}_ky{int(ky)}")
        orders.append(0)
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def full_axial_pairwise_basis(d: torch.Tensor, side: int, *, name: str = "axis_full_29") -> Basis:
    values = torch.arange(-(side - 1), side, device=d.device, dtype=d.dtype)
    cols: list[torch.Tensor] = []
    labels: list[str] = []
    for value in values:
        cols.append((d[:, 0] == value).to(d.dtype))
        labels.append(f"dx_{int(value.item())}")
    for value in values[:-1]:
        cols.append((d[:, 1] == value).to(d.dtype))
        labels.append(f"dy_{int(value.item())}")
    return Basis(name, torch.stack(cols, dim=1), labels, [0] * len(labels))


def axis_fourier_j0_basis(d: torch.Tensor, *, num_centers: int, seed: int, name: str) -> Basis:
    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17031)
    base = torch.linspace(0.17, math.pi - 0.17, steps=max(1, int(num_centers)), dtype=torch.float64)
    jitter = (torch.rand(base.shape, generator=generator, dtype=torch.float64) - 0.5) * 0.04
    freqs = (base + jitter).clamp(0.05, math.pi - 0.05).to(device=d.device, dtype=d.dtype)
    for idx, omega in enumerate(freqs):
        axis = idx % 2
        phase = d[:, axis] * omega
        cols.extend([torch.cos(phase), torch.sin(phase)])
        labels.extend([f"axis{axis}_w{idx}_cos", f"axis{axis}_w{idx}_sin"])
    return Basis(name, torch.stack(cols, dim=1), labels, [0] * len(labels))


def table_informed_pj_pairwise_basis(d: torch.Tensor, side: int, omegas: torch.Tensor, *, order: int, name: str) -> Basis:
    ex = torch.tensor([1.0, 0.0], device=d.device, dtype=d.dtype)
    ey = torch.tensor([0.0, 1.0], device=d.device, dtype=d.dtype)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1)
    oblique = normalize_direction(torch.tensor([[1.0, -0.65]], device=d.device, dtype=d.dtype)).reshape(-1)
    return directional_jet_basis(
        d,
        [omega.to(device=d.device, dtype=d.dtype) for omega in omegas],
        [ex, ey, diag, oblique],
        list(range(order + 1)),
        scale=float(side),
        name=name,
    )


def chart_scale_from_token(value: str | None, *, default: float) -> float:
    if value is None or value == "":
        return float(default)
    return float(value.replace("p", "."))


def radial_chart_coordinates(d: torch.Tensor, *, chart: str, scale: float) -> torch.Tensor:
    scale = float(scale)
    radius = torch.linalg.norm(d, dim=-1, keepdim=True)
    if chart in {"lc", "asinh"}:
        warped_radius = scale * torch.asinh(radius / scale)
    elif chart == "log":
        warped_radius = scale * torch.log1p(radius / scale)
    else:
        raise ValueError(f"unknown radial chart: {chart}")
    factor = torch.where(radius > 0, warped_radius / radius.clamp_min(1e-12), torch.ones_like(radius))
    return d * factor


def table_informed_chart_pj_pairwise_basis(
    d: torch.Tensor,
    side: int,
    omegas: torch.Tensor,
    *,
    order: int,
    chart: str,
    scale: float,
    name: str,
) -> Basis:
    ex = torch.tensor([1.0, 0.0], device=d.device, dtype=d.dtype)
    ey = torch.tensor([0.0, 1.0], device=d.device, dtype=d.dtype)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=d.device, dtype=d.dtype)).reshape(-1)
    oblique = normalize_direction(torch.tensor([[1.0, -0.65]], device=d.device, dtype=d.dtype)).reshape(-1)
    directions = [ex, ey, diag, oblique]
    order_list = sorted(set(range(order + 1)))
    phi = radial_chart_coordinates(d, chart=chart, scale=scale)

    cols = [torch.ones(d.shape[0], device=d.device, dtype=d.dtype)]
    labels = ["const"]
    orders = [0]
    for omega_idx, omega in enumerate(omegas):
        ph = phi @ omega.to(device=d.device, dtype=d.dtype)
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)
        if 0 in order_list:
            cols.extend([cos_ph, sin_ph])
            labels.extend([f"{chart}_w{omega_idx}_r0_cos", f"{chart}_w{omega_idx}_r0_sin"])
            orders.extend([0, 0])
        for direction_idx, direction in enumerate(directions):
            coord = (phi @ direction) / max(float(scale), 1e-6)
            for item_order in order_list:
                if item_order == 0:
                    continue
                poly = coord.pow(item_order)
                cols.extend([poly * cos_ph, poly * sin_ph])
                labels.extend(
                    [
                        f"{chart}_w{omega_idx}_u{direction_idx}_r{item_order}_cos",
                        f"{chart}_w{omega_idx}_u{direction_idx}_r{item_order}_sin",
                    ]
                )
                orders.extend([item_order, item_order])
    return Basis(name, torch.stack(cols, dim=1), labels, orders)


def without_const(basis: Basis) -> Basis:
    keep = [idx for idx, label in enumerate(basis.labels) if label != "const"]
    if not keep:
        return basis
    idx = torch.tensor(keep, device=basis.matrix.device, dtype=torch.long)
    return Basis(
        basis.name,
        basis.matrix[:, idx],
        [basis.labels[i] for i in keep],
        [basis.orders[i] for i in keep],
    )


def concat_bases(name: str, bases: list[Basis]) -> Basis:
    matrix = torch.cat([basis.matrix for basis in bases], dim=1)
    labels: list[str] = []
    orders: list[int] = []
    for basis in bases:
        labels.extend([f"{basis.name}:{label}" for label in basis.labels])
        orders.extend(basis.orders)
    return Basis(name, matrix, labels, orders)


def orthonormalize_basis(basis: Basis, *, name: str) -> Basis:
    matrix = basis.matrix.to(dtype=torch.float32)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diag = torch.diagonal(r)
    signs = torch.where(diag < 0, -torch.ones_like(diag), torch.ones_like(diag))
    q = q * signs.unsqueeze(0)
    labels = [f"orth:{idx}:{label}" for idx, label in enumerate(basis.labels)]
    return Basis(name, q, labels, list(basis.orders))


def v12_per_center_atoms(order: int, family: str) -> int:
    if family == "full":
        return 2 * ((int(order) + 1) * (int(order) + 2) // 2)
    if family == "directional":
        return 2 + 8 * int(order)
    raise ValueError(f"unknown v12 family: {family}")


def v12_teacher_omegas(
    teacher_tables: torch.Tensor,
    *,
    source: str,
    target: str,
    k: int,
    seed: int,
) -> torch.Tensor:
    if k <= 0:
        return torch.empty(0, 2, device=teacher_tables.device, dtype=teacher_tables.dtype)
    target_tables = _teacher_target(teacher_tables, target)
    source = {"table": "table_informed", "table_shuffled": "table_informed_shuffled", "fixed": "generic"}.get(source, source)
    if source in {"table_informed", "top_dft"}:
        return top_dft_omegas_from_teacher(target_tables, k=k)
    if source in {"generic", "fixed_grid"}:
        return v12_generic_omegas(k, device=teacher_tables.device, dtype=teacher_tables.dtype, seed=seed)
    if source == "random_matched":
        ref = top_dft_omegas_from_teacher(target_tables, k=k)
        return v12_random_matched_omegas(ref, seed=seed)
    if source == "table_informed_shuffled":
        ref = top_dft_omegas_from_teacher(target_tables, k=k)
        return v12_shuffled_omegas(ref, seed=seed)
    raise ValueError(f"unknown v12 frequency source: {source}")


def v12_toric_basis_from_omegas(
    d: torch.Tensor,
    side: int,
    omegas: torch.Tensor,
    *,
    family: str,
    order: int,
    name: str,
    include_const: bool = True,
) -> Basis:
    scale = float(max(1, side - 1))
    if family == "full":
        return v12_full_multijet_basis(
            d,
            omegas,
            order=order,
            scale=scale,
            include_const=include_const,
            name=name,
        )
    if family == "directional":
        return v12_directional_jet_basis(
            d,
            omegas,
            directions=v12_default_directions(device=d.device, dtype=d.dtype),
            order=order,
            scale=scale,
            include_const=include_const,
            name=name,
        )
    raise ValueError(f"unknown v12 family: {family}")


def v12_fair_student_basis(
    variant: str,
    d: torch.Tensor,
    side: int,
    teacher_tables: torch.Tensor,
    *,
    seed: int,
) -> Basis | None:
    match = re.fullmatch(
        r"v12_(nested|matched)_(table|axis_residual)_toric_(full|directional)_J(\d+)_(k|atoms)(\d+)(?:_(generic|table|random_matched|fixed|table_shuffled))?",
        variant,
    )
    if match is None:
        return None
    protocol = match.group(1)
    target_mode = match.group(2)
    family = match.group(3)
    order = int(match.group(4))
    size_kind = match.group(5)
    size_value = int(match.group(6))
    source = match.group(7) or "table"

    if protocol == "nested":
        if size_kind != "k":
            raise ValueError(f"nested v12 variants use explicit k: {variant}")
        k = size_value
    elif protocol == "matched":
        if size_kind != "atoms":
            raise ValueError(f"matched v12 variants use explicit atoms: {variant}")
        fixed_features = full_axial_pairwise_basis(d, side).matrix.shape[1] if target_mode == "axis_residual" else 1
        remaining = max(1, size_value - fixed_features)
        k = max(1, remaining // v12_per_center_atoms(order, family))
    else:
        raise ValueError(f"unknown v12 protocol: {protocol}")

    target = "residual" if target_mode == "axis_residual" else "full"
    omegas = v12_teacher_omegas(teacher_tables, source=source, target=target, k=k, seed=seed)
    toric = v12_toric_basis_from_omegas(
        d,
        side,
        omegas,
        family=family,
        order=order,
        include_const=target_mode != "axis_residual",
        name=f"{variant}_toric",
    )
    if target_mode == "axis_residual":
        axis = full_axial_pairwise_basis(d, side)
        return concat_bases(variant, [axis, toric])
    return Basis(variant, toric.matrix, toric.labels, toric.orders)


def v12_frequency_source_basis(
    variant: str,
    d: torch.Tensor,
    side: int,
    teacher_tables: torch.Tensor,
    *,
    seed: int,
) -> Basis | None:
    match = re.fullmatch(r"v12_freqsrc_(axis|fixed|random|table|table_shuffled)_J0_atoms(\d+)(?:_(full|residual))?", variant)
    if match is None:
        return None
    source = match.group(1)
    budget = int(match.group(2))
    target = match.group(3) or "full"
    if source == "axis":
        k = max(1, (budget - 1) // 2)
        return axis_fourier_j0_basis(d, num_centers=k, seed=seed, name=variant)
    k = max(1, (budget - 1) // 2)
    source_map = {
        "fixed": "generic",
        "random": "random_matched",
        "table": "table",
        "table_shuffled": "table_shuffled",
    }
    omegas = v12_teacher_omegas(teacher_tables, source=source_map[source], target=target, k=k, seed=seed)
    basis = v12_toric_j0_basis(d, omegas, include_const=True, name=variant)
    return Basis(variant, basis.matrix, basis.labels, basis.orders)


def build_teacher_bases(
    *,
    side: int,
    device: torch.device,
    teacher_tables: torch.Tensor,
    variants: list[str],
    include_shuffle: bool,
    seed: int,
) -> list[Basis]:
    positions = make_positions(side, device)
    d = pairwise_d(positions).reshape(-1, 2).to(torch.float32)
    standard = {basis.name: basis for basis in build_patch_bases(side, device, include_shuffle=include_shuffle, seed=seed)}
    bases: list[Basis] = []
    for variant in variants:
        if variant.startswith("orth_"):
            base_variant = variant[len("orth_") :]
            base = build_teacher_bases(
                side=side,
                device=device,
                teacher_tables=teacher_tables,
                variants=[base_variant],
                include_shuffle=include_shuffle,
                seed=seed,
            )[0]
            bases.append(orthonormalize_basis(base, name=variant))
            continue
        if variant in standard:
            bases.append(standard[variant])
            continue
        v12_basis = v12_fair_student_basis(variant, d, side, teacher_tables, seed=seed)
        if v12_basis is not None:
            bases.append(v12_basis)
            continue
        v12_freq_basis = v12_frequency_source_basis(variant, d, side, teacher_tables, seed=seed)
        if v12_freq_basis is not None:
            bases.append(v12_freq_basis)
            continue
        if variant.startswith("mixed_"):
            match = re.fullmatch(
                r"mixed_(lc|asinh|log)_toric_PJ_R(\d+)_top(\d+)(?:_L([0-9p.]+))?_(residual_)?dct_top(\d+)",
                variant,
            )
            if match is not None:
                chart = match.group(1)
                order = int(match.group(2))
                pj_budget = int(match.group(3))
                scale = chart_scale_from_token(match.group(4), default=float(side))
                dct_prefix = "residual_dct_top" if match.group(5) else "dct_top"
                dct_budget = int(match.group(6))
                pj_scale_token = f"_L{match.group(4)}" if match.group(4) else ""
                pj = build_teacher_bases(
                    side=side,
                    device=device,
                    teacher_tables=teacher_tables,
                    variants=[f"table_informed_{chart}_toric_PJ_R{order}_top{pj_budget}{pj_scale_token}"],
                    include_shuffle=include_shuffle,
                    seed=seed,
                )[0]
                dct = build_teacher_bases(
                    side=side,
                    device=device,
                    teacher_tables=teacher_tables,
                    variants=[f"{dct_prefix}{dct_budget}"],
                    include_shuffle=include_shuffle,
                    seed=seed,
                )[0]
                bases.append(concat_bases(variant, [pj, without_const(dct)]))
                continue
        if variant.startswith("mixed_toric_PJ_R"):
            match = re.fullmatch(r"mixed_toric_PJ_R(\d+)_top(\d+)_(residual_)?dct_top(\d+)", variant)
            if match is None:
                raise ValueError(f"cannot parse variant: {variant}")
            order = int(match.group(1))
            pj_budget = int(match.group(2))
            dct_prefix = "residual_dct_top" if match.group(3) else "dct_top"
            dct_budget = int(match.group(4))
            pj = build_teacher_bases(
                side=side,
                device=device,
                teacher_tables=teacher_tables,
                variants=[f"table_informed_toric_PJ_R{order}_top{pj_budget}"],
                include_shuffle=include_shuffle,
                seed=seed,
            )[0]
            dct = build_teacher_bases(
                side=side,
                device=device,
                teacher_tables=teacher_tables,
                variants=[f"{dct_prefix}{dct_budget}"],
                include_shuffle=include_shuffle,
                seed=seed,
            )[0]
            bases.append(concat_bases(variant, [pj, without_const(dct)]))
            continue
        if variant.startswith("dct_top") or variant.startswith("residual_dct_top"):
            budget = int(re.search(r"top(\d+)", variant).group(1))  # type: ignore[union-attr]
            target = "residual" if variant.startswith("residual_") else "full"
            indices = top_dct_indices_from_teacher(_teacher_target(teacher_tables, target), k=budget)
            bases.append(dct_pairwise_basis(d, side, indices, name=variant))
            continue
        if variant.startswith("dft_R0_top") or variant.startswith("residual_dft_R0_top"):
            budget = int(re.search(r"top(\d+)", variant).group(1))  # type: ignore[union-attr]
            target = "residual" if variant.startswith("residual_") else "full"
            k = max(1, (budget - 1) // 2)
            omegas = top_dft_omegas_from_teacher(_teacher_target(teacher_tables, target), k=k)
            bases.append(toric_fourier_basis(d, list(omegas), name=variant))
            continue
        if variant.startswith("table_informed_lc_toric_PJ_R") or variant.startswith(
            "table_informed_asinh_toric_PJ_R"
        ) or variant.startswith("table_informed_log_toric_PJ_R"):
            match = re.fullmatch(r"table_informed_(lc|asinh|log)_toric_PJ_R(\d+)_top(\d+)(?:_L([0-9p.]+))?", variant)
            if match is None:
                raise ValueError(f"cannot parse variant: {variant}")
            chart = match.group(1)
            order = int(match.group(2))
            budget = int(match.group(3))
            scale = chart_scale_from_token(match.group(4), default=float(side))
            per_atom = {0: 2, 1: 10, 2: 18}.get(order, 18)
            k = max(1, (budget - 1) // per_atom)
            omegas = top_dft_omegas_from_teacher(teacher_tables, k=k)
            bases.append(
                table_informed_chart_pj_pairwise_basis(
                    d,
                    side,
                    omegas,
                    order=order,
                    chart=chart,
                    scale=scale,
                    name=variant,
                )
            )
            continue
        if variant.startswith("table_informed_toric_PJ_R"):
            match = re.search(r"R(\d+)_top(\d+)", variant)
            if match is None:
                raise ValueError(f"cannot parse variant: {variant}")
            order = int(match.group(1))
            budget = int(match.group(2))
            per_atom = {0: 2, 1: 10, 2: 18}.get(order, 18)
            k = max(1, (budget - 1) // per_atom)
            omegas = top_dft_omegas_from_teacher(teacher_tables, k=k)
            bases.append(table_informed_pj_pairwise_basis(d, side, omegas, order=order, name=variant))
            continue
        if variant.startswith("axis_plus_toric_residual_R"):
            match = re.search(r"R(\d+)_top(\d+)", variant)
            if match is None:
                raise ValueError(f"cannot parse variant: {variant}")
            order = int(match.group(1))
            budget = int(match.group(2))
            axis = full_axial_pairwise_basis(d, side)
            per_atom = {0: 2, 1: 10, 2: 18}.get(order, 18)
            k = max(1, (budget - axis.matrix.shape[1]) // per_atom)
            residual = _teacher_target(teacher_tables, "residual")
            omegas = top_dft_omegas_from_teacher(residual, k=k)
            pj = without_const(table_informed_pj_pairwise_basis(d, side, omegas, order=order, name=f"{variant}_pj"))
            bases.append(concat_bases(variant, [axis, pj]))
            continue
        raise ValueError(f"unknown V4-C student basis: {variant}")
    return bases


def teacher_init_coeff(
    basis: Basis,
    teacher_tables: torch.Tensor,
    *,
    side: int,
    depth: int,
    n_heads: int,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    if teacher_tables.shape[0] < depth or teacher_tables.shape[1] < n_heads:
        raise ValueError(
            f"teacher has {teacher_tables.shape[0]} layers/{teacher_tables.shape[1]} heads, "
            f"but student requests {depth}/{n_heads}"
        )
    target_tables = teacher_tables[:depth, :n_heads].to(device=basis.matrix.device, dtype=torch.float32)
    pairwise = relative_table_to_pairwise(target_tables, side)
    coeff = torch.zeros(depth, n_heads, basis.matrix.shape[1], device=basis.matrix.device, dtype=torch.float32)
    rows: list[dict[str, object]] = []
    matrix = basis.matrix.to(device=basis.matrix.device, dtype=torch.float32)
    for layer in range(depth):
        for head in range(n_heads):
            _, item_coeff, mse, r2 = fit_linear_basis(matrix, pairwise[layer, head])
            coeff[layer, head] = item_coeff.reshape(-1).to(coeff.dtype)
            rows.append(
                {
                    "basis": basis.name,
                    "layer": layer,
                    "head": head,
                    "teacher_init_mse": mse,
                    "teacher_init_r2": r2,
                    "num_features": int(basis.matrix.shape[1]),
                }
            )
    return coeff, rows


def write_v4_report(output_dir: Path, summary: dict[str, object], aggregate_rows: list[dict[str, object]], init_rows: list[dict[str, object]]) -> None:
    init_by_basis: dict[str, list[float]] = {}
    for row in init_rows:
        init_by_basis.setdefault(str(row["basis"]), []).append(float(row["teacher_init_r2"]))
    lines = [
        "# V4-C Metric Toric PJ Student Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        f"- Dataset: {summary['dataset']}",
        f"- Teacher bias: `{summary['teacher_bias']}`",
        f"- Patch size / grid side: {summary['patch_size']} / {summary['grid_side']}",
        f"- Depth / dim / heads: {summary['depth']} / {summary['dim']} / {summary['n_heads']}",
        f"- Steps: {summary['steps']}",
        f"- Teacher init: {summary['teacher_init']}",
        "",
        "## Aggregate",
        "",
        "| basis | n | score mean | best mean | final mean | train score | features | init R2 mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        basis = str(row["basis"])
        init_vals = init_by_basis.get(basis, [])
        init_mean = float(np.mean(init_vals)) if init_vals else float("nan")
        lines.append(
            "| "
            + f"{basis} | {int(row['n'])} | {float(row['score_mean']):.4f} | "
            + f"{float(row.get('best_score_mean', row['score_mean'])):.4f} | "
            + f"{float(row.get('final_score_mean', row['score_mean'])):.4f} | "
            + f"{float(row['train_score_mean']):.4f} | {int(row['num_features'])} | {init_mean:.4f} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `student_results.csv`",
            "- `student_aggregate.csv`",
            "- `student_curves.csv`",
            "- `teacher_init_fits.csv`",
            "- `basis_accuracy_boxplot.png`",
            "- `train_test_curves.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    bundle, teacher_metadata = load_bias_npz(Path(args.teacher_bias), device=torch.device("cpu"))
    teacher_tables_cpu = bundle.tables.float()
    train_images, train_y, test_images, test_y = load_vision_dataset(
        dataset=args.dataset,
        root=Path(args.data_root),
        device=device,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
    train_x, grid_side, patch_dim = patchify(train_images, args.patch_size)
    test_x, test_grid_side, _ = patchify(test_images, args.patch_size)
    if test_grid_side != grid_side:
        raise ValueError("train/test grid side mismatch")
    teacher_side = int(teacher_metadata.get("grid_side", (teacher_tables_cpu.shape[-1] + 1) // 2))
    if teacher_side != grid_side:
        raise ValueError(f"teacher grid side {teacher_side} != data grid side {grid_side}")
    train_x, test_x = normalize_patches(train_x, test_x)

    variants = parse_list(args.student_bases)
    bases = build_teacher_bases(
        side=grid_side,
        device=device,
        teacher_tables=teacher_tables_cpu.to(device=device),
        variants=variants,
        include_shuffle=args.include_shuffle,
        seed=args.seed,
    )
    tasks = parse_list(args.tasks)
    rows: list[dict[str, object]] = read_csv(output_dir / "student_results.csv") if args.resume else []
    curves: list[dict[str, object]] = read_csv(output_dir / "student_curves.csv") if args.resume else []
    init_rows: list[dict[str, object]] = read_csv(output_dir / "teacher_init_fits.csv") if args.resume else []
    completed = {
        (str(row.get("dataset")), str(row.get("task")), str(row.get("basis")), int(float(row.get("seed", -1))))
        for row in rows
    }
    if args.seed_basis_order:
        seed_basis_order = parse_list(args.seed_basis_order)
        basis_seed_indices = {name: idx for idx, name in enumerate(seed_basis_order)}
    else:
        basis_seed_indices = {basis.name: idx for idx, basis in enumerate(bases)}

    runs_done = 0
    wall_start = time.time()
    for basis_idx, basis in enumerate(bases):
        init_coeff = None
        basis_init_rows: list[dict[str, object]] = []
        if args.teacher_init:
            init_coeff, basis_init_rows = teacher_init_coeff(
                basis,
                teacher_tables_cpu.to(device=device),
                side=grid_side,
                depth=args.depth,
                n_heads=args.n_heads,
            )
            if not args.resume or not any(str(row.get("basis")) == basis.name for row in init_rows):
                init_rows.extend(basis_init_rows)
                write_csv(output_dir / "teacher_init_fits.csv", init_rows)
        init_r2_mean = float(np.mean([float(row["teacher_init_r2"]) for row in basis_init_rows])) if basis_init_rows else float("nan")
        for task_idx, task in enumerate(tasks):
            for seed_idx in range(args.seed_start_idx, args.seeds):
                basis_seed_idx = basis_seed_indices.get(basis.name, basis_idx)
                seed = args.seed + 100 * seed_idx + 17 * basis_seed_idx + 1009 * task_idx
                if (args.dataset, task, basis.name, seed) in completed:
                    continue
                if args.max_runs is not None and runs_done >= args.max_runs:
                    break
                row, task_curves = train_task(
                    basis,
                    task=task,
                    train_x=train_x,
                    train_y=train_y,
                    test_x=test_x,
                    test_y=test_y,
                    patch_dim=patch_dim,
                    args=args,
                    seed=seed,
                    init_coeff=init_coeff,
                )
                row.update(
                    {
                        "teacher_bias": args.teacher_bias,
                        "teacher_basis": teacher_metadata.get("basis", "unknown"),
                        "teacher_seed": teacher_metadata.get("seed", -1),
                        "teacher_init": bool(args.teacher_init),
                        "teacher_init_r2_mean": init_r2_mean,
                    }
                )
                rows.append(row)
                curves.extend(task_curves)
                completed.add((args.dataset, task, basis.name, seed))
                runs_done += 1
                aggregate_rows = aggregate(rows)
                write_csv(output_dir / "student_results.csv", rows)
                write_csv(output_dir / "student_aggregate.csv", aggregate_rows)
                write_csv(output_dir / "student_curves.csv", curves)
                partial = {
                    "device": str(device),
                    "dataset": args.dataset,
                    "teacher_bias": args.teacher_bias,
                    "teacher_init": args.teacher_init,
                    "patch_size": args.patch_size,
                    "grid_side": grid_side,
                    "patch_dim": patch_dim,
                    "depth": args.depth,
                    "dim": args.dim,
                    "n_heads": args.n_heads,
                    "steps": args.steps,
                    "seeds": args.seeds,
                    "tasks": tasks,
                    "bases": [basis.name for basis in bases],
                    "runs_done_this_invocation": runs_done,
                    "wall_sec": time.time() - wall_start,
                    "status": "partial",
                }
                (output_dir / "student_summary.partial.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
            if args.max_runs is not None and runs_done >= args.max_runs:
                break
        if args.max_runs is not None and runs_done >= args.max_runs:
            break

    aggregate_rows = aggregate(rows)
    write_csv(output_dir / "student_results.csv", rows)
    write_csv(output_dir / "student_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "student_curves.csv", curves)
    write_csv(output_dir / "teacher_init_fits.csv", init_rows)
    plot_results(output_dir, aggregate_rows, curves)
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() and device.type == "cuda" else 0
    summary = {
        "device": str(device),
        "dataset": args.dataset,
        "teacher_bias": args.teacher_bias,
        "teacher_metadata": teacher_metadata,
        "teacher_init": args.teacher_init,
        "patch_size": args.patch_size,
        "grid_side": grid_side,
        "patch_dim": patch_dim,
        "depth": args.depth,
        "dim": args.dim,
        "n_heads": args.n_heads,
        "steps": args.steps,
        "seeds": args.seeds,
        "score_mode": args.score_mode,
        "tasks": tasks,
        "bases": [basis.name for basis in bases],
        "train_count": int(train_x.shape[0]),
        "test_count": int(test_x.shape[0]),
        "wall_sec": time.time() - wall_start,
        "peak_cuda_memory_bytes": peak_mem,
        "runs_done_this_invocation": runs_done,
    }
    summary_path = output_dir / "student_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_v4_report(output_dir, summary, aggregate_rows, init_rows)
    summary["summary"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4-C table-informed Metric Toric PJ students.")
    parser.add_argument("--teacher-bias", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-init", action="store_true")
    parser.add_argument("--student-bases", type=str, default="dct_top33,dct_top55,table_informed_toric_PJ_R0_top55,table_informed_toric_PJ_R0_top110,axis_plus_toric_residual_R0_top55,axis_plus_toric_residual_R2_top55")
    parser.add_argument("--dataset", choices=VISION_DATASETS, default="cifar10")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--tasks", type=str, default="reconstruction")
    parser.add_argument("--mask-rate", type=float, default=0.35)
    parser.add_argument("--lambda-recon", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-schedule", choices=["cosine", "cosine_hold", "constant"], default="cosine_hold")
    parser.add_argument("--lr-decay-steps", type=int, default=5000)
    parser.add_argument("--lr-min-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-subset", type=int, default=4096)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--include-shuffle", action="store_true", default=True)
    parser.add_argument("--seed-start-idx", type=int, default=0)
    parser.add_argument("--seed-basis-order", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-mode", choices=["final", "best"], default="best")
    parser.add_argument("--export-bias-every", type=int, default=0)
    parser.add_argument("--bias-ablation-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=426)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v4_metric_toric_pj_cifar10_10k")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
