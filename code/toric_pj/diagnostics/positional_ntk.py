from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from toric_pj.diagnostics.basis_projection import normalize_columns, phase
from toric_pj.diagnostics.direction_alignment import normalize_direction


@dataclass(frozen=True)
class TangentMeta:
    sector: str
    order: int
    source: str
    label: str


@dataclass(frozen=True)
class TangentBank:
    features: torch.Tensor
    metas: list[TangentMeta]


def build_projected_pj_tangent_bank(
    d: torch.Tensor,
    *,
    radius: int,
    omegas: Iterable[torch.Tensor],
    directions: Iterable[torch.Tensor],
    max_order: int,
    include_spectral_tangents: bool,
    include_lc: bool = True,
    lc_omega: float = 0.55,
) -> TangentBank:
    columns: list[torch.Tensor] = []
    metas: list[TangentMeta] = []
    scale = float(radius)

    def add(column: torch.Tensor, sector: str, order: int, source: str, label: str) -> None:
        columns.append(column)
        metas.append(TangentMeta(sector=sector, order=order, source=source, label=label))

    norm_dirs = [
        normalize_direction(direction.to(device=d.device, dtype=d.dtype).reshape(1, -1)).reshape(-1)
        for direction in directions
    ]
    omega_list = [omega.to(device=d.device, dtype=d.dtype) for omega in omegas]

    for omega_idx, omega in enumerate(omega_list):
        ph = phase(d, omega)
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)
        add(cos_ph, "FJ", 0, "amplitude", f"FJ_w{omega_idx}_r0_cos")
        add(sin_ph, "FJ", 0, "amplitude", f"FJ_w{omega_idx}_r0_sin")
        if include_spectral_tangents:
            for axis in range(d.shape[-1]):
                scaled_axis = d[:, axis] / scale
                add(-scaled_axis * sin_ph, "FJ", 1, "spectral", f"FJ_w{omega_idx}_domega{axis}_cos")
                add(scaled_axis * cos_ph, "FJ", 1, "spectral", f"FJ_w{omega_idx}_domega{axis}_sin")
        for dir_idx, direction in enumerate(norm_dirs):
            coord = (d @ direction) / scale
            for order in range(1, max_order + 1):
                poly = coord.pow(order)
                add(poly * cos_ph, "FJ", order, "amplitude", f"FJ_w{omega_idx}_u{dir_idx}_r{order}_cos")
                add(poly * sin_ph, "FJ", order, "amplitude", f"FJ_w{omega_idx}_u{dir_idx}_r{order}_sin")
                if include_spectral_tangents:
                    for axis in range(d.shape[-1]):
                        scaled_axis = d[:, axis] / scale
                        tangent_order = order + 1
                        add(
                            -scaled_axis * poly * sin_ph,
                            "FJ",
                            tangent_order,
                            "spectral",
                            f"FJ_w{omega_idx}_u{dir_idx}_r{order}_domega{axis}_cos",
                        )
                        add(
                            scaled_axis * poly * cos_ph,
                            "FJ",
                            tangent_order,
                            "spectral",
                            f"FJ_w{omega_idx}_u{dir_idx}_r{order}_domega{axis}_sin",
                        )

    add(torch.ones(d.shape[0], device=d.device, dtype=d.dtype), "affine", 0, "amplitude", "affine_const")
    for dir_idx, direction in enumerate(norm_dirs):
        add(-(d @ direction) / scale, "affine", 1, "amplitude", f"affine_u{dir_idx}")

    if include_lc:
        for dir_idx, direction in enumerate(norm_dirs):
            raw = d @ direction
            phi = scale * torch.asinh(raw / scale)
            beta = raw / torch.sqrt(raw.square() + scale**2)
            ph = float(lc_omega) * phi
            cos_ph = torch.cos(ph)
            sin_ph = torch.sin(ph)
            add(cos_ph, "LC", 0, "amplitude", f"LC_u{dir_idx}_r0_cos")
            add(sin_ph, "LC", 0, "amplitude", f"LC_u{dir_idx}_r0_sin")
            for order in range(1, max_order + 1):
                poly = beta.pow(order)
                add(poly * cos_ph, "LC", order, "amplitude", f"LC_u{dir_idx}_r{order}_cos")
                add(poly * sin_ph, "LC", order, "amplitude", f"LC_u{dir_idx}_r{order}_sin")

    features, _ = normalize_columns(torch.stack(columns, dim=1))
    return TangentBank(features=features, metas=metas)


def kernel_from_features(features: torch.Tensor) -> torch.Tensor:
    return features @ features.T


def kernel_target_alignment(kernel: torch.Tensor, target: torch.Tensor) -> float:
    yy = target.reshape(-1, 1) @ target.reshape(1, -1)
    numerator = torch.sum(kernel * yy)
    denominator = torch.linalg.norm(kernel).clamp_min(1e-30) * torch.linalg.norm(yy).clamp_min(1e-30)
    return float((numerator / denominator).detach().cpu())


def effective_rank(kernel: torch.Tensor, eps: float = 1e-12) -> float:
    evals = torch.linalg.eigvalsh(kernel).clamp_min(0)
    total = evals.sum()
    if total <= eps:
        return 0.0
    probs = evals / total
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(eps)))
    return float(torch.exp(entropy).detach().cpu())


def matrix_condition(features: torch.Tensor, eps: float = 1e-12) -> float:
    sv = torch.linalg.svdvals(features)
    if sv[-1] <= eps:
        return float("inf")
    return float((sv[0] / sv[-1]).detach().cpu())


def grouped_kernel_metrics(
    bank: TangentBank,
    targets: dict[str, torch.Tensor],
    *,
    group_by: str,
) -> list[dict[str, object]]:
    if group_by not in {"sector", "order", "source"}:
        raise ValueError("group_by must be sector, order, or source")
    total_kernel = kernel_from_features(bank.features)
    total_norm = torch.linalg.norm(total_kernel).clamp_min(1e-30)
    groups = sorted({str(getattr(meta, group_by)) for meta in bank.metas})
    rows: list[dict[str, object]] = []
    for group in groups:
        idx = torch.tensor(
            [i for i, meta in enumerate(bank.metas) if str(getattr(meta, group_by)) == group],
            device=bank.features.device,
        )
        features = bank.features[:, idx]
        kernel = kernel_from_features(features)
        row: dict[str, object] = {
            "group_by": group_by,
            "group": group,
            "num_features": int(features.shape[1]),
            "kernel_energy": float((torch.linalg.norm(kernel) / total_norm).detach().cpu()),
        }
        for target_name, target in targets.items():
            row[f"align:{target_name}"] = kernel_target_alignment(kernel, target)
        rows.append(row)
    return rows
