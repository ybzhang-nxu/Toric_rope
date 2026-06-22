from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from toric_pj.diagnostics.direction_alignment import normalize_direction


@dataclass
class Basis:
    name: str
    matrix: torch.Tensor
    labels: list[str]
    orders: list[int]


@dataclass
class FitResult:
    basis_name: str
    mse: float
    r2: float
    condition: float
    coeff_norm: float
    top_energy_order: int | None
    top_loo_order: int | None
    order_energy: dict[int, float]
    order_loo: dict[int, float]


def default_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grid_2d(
    radius: int,
    *,
    signed: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    device = default_device(str(device)) if device is not None else default_device()
    if signed:
        values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    else:
        values = torch.arange(0, radius + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


def normalize_columns(matrix: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    norms = torch.linalg.norm(matrix, dim=0).clamp_min(eps)
    return matrix / norms, norms


def condition_number(matrix: torch.Tensor, eps: float = 1e-14) -> float:
    if matrix.shape[1] == 0:
        return float("nan")
    singular_values = torch.linalg.svdvals(matrix)
    s_min = torch.min(singular_values)
    s_max = torch.max(singular_values)
    if s_min <= eps:
        return float("inf")
    return float((s_max / s_min).detach().cpu())


def fit_basis(
    basis: Basis,
    target: torch.Tensor,
    *,
    ridge: float = 1e-10,
    normalize: bool = True,
) -> FitResult:
    target = target.reshape(-1, 1)
    matrix = basis.matrix
    if normalize:
        matrix, _ = normalize_columns(matrix)

    gram = matrix.T @ matrix
    rhs = matrix.T @ target
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coeff = torch.linalg.solve(gram + ridge * eye, rhs)
    pred = matrix @ coeff
    residual = pred - target
    mse = torch.mean(residual.square())
    variance = torch.mean((target - target.mean()).square()).clamp_min(1e-30)
    r2 = 1.0 - mse / variance

    order_energy, order_loo = _order_diagnostics(matrix, coeff, target, basis.orders)
    top_energy = _top_positive(order_energy)
    top_loo = _top_positive(order_loo)

    return FitResult(
        basis_name=basis.name,
        mse=float(mse.detach().cpu()),
        r2=float(r2.detach().cpu()),
        condition=condition_number(matrix),
        coeff_norm=float(torch.linalg.norm(coeff).detach().cpu()),
        top_energy_order=top_energy,
        top_loo_order=top_loo,
        order_energy=order_energy,
        order_loo=order_loo,
    )


def _order_diagnostics(
    matrix: torch.Tensor,
    coeff: torch.Tensor,
    target: torch.Tensor,
    orders: Sequence[int],
) -> tuple[dict[int, float], dict[int, float]]:
    pred = matrix @ coeff
    denom = torch.linalg.norm(pred).clamp_min(1e-30)
    base_mse = torch.mean((pred - target).square())
    energy: dict[int, float] = {}
    loo: dict[int, float] = {}
    for order in sorted(set(orders)):
        idx = torch.tensor(
            [i for i, item_order in enumerate(orders) if item_order == order],
            device=matrix.device,
            dtype=torch.long,
        )
        if idx.numel() == 0:
            continue
        contrib = matrix[:, idx] @ coeff[idx]
        energy[order] = float((torch.linalg.norm(contrib) / denom).detach().cpu())
        mse_without = torch.mean((pred - contrib - target).square())
        loo[order] = float((mse_without - base_mse).detach().cpu())
    return energy, loo


def _top_positive(values: dict[int, float]) -> int | None:
    if not values:
        return None
    return max(values, key=lambda key: values[key])


def phase(d: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    return d @ omega.to(device=d.device, dtype=d.dtype)


def toric_fourier_basis(
    d: torch.Tensor,
    omegas: Iterable[torch.Tensor],
    *,
    include_const: bool = True,
    name: str = "toric_fourier",
) -> Basis:
    columns: list[torch.Tensor] = []
    labels: list[str] = []
    orders: list[int] = []
    if include_const:
        columns.append(torch.ones(d.shape[0], device=d.device, dtype=d.dtype))
        labels.append("const")
        orders.append(0)
    for idx, omega in enumerate(omegas):
        ph = phase(d, omega)
        columns.extend([torch.cos(ph), torch.sin(ph)])
        labels.extend([f"cos_w{idx}", f"sin_w{idx}"])
        orders.extend([0, 0])
    return Basis(name=name, matrix=torch.stack(columns, dim=1), labels=labels, orders=orders)


def axis_additive_fourier_basis(
    d: torch.Tensor,
    axis_freqs: Sequence[float],
    *,
    include_const: bool = True,
    name: str = "axis_additive",
) -> Basis:
    columns: list[torch.Tensor] = []
    labels: list[str] = []
    orders: list[int] = []
    if include_const:
        columns.append(torch.ones(d.shape[0], device=d.device, dtype=d.dtype))
        labels.append("const")
        orders.append(0)
    for axis, freq in enumerate(axis_freqs):
        ph = d[:, axis] * float(freq)
        columns.extend([torch.cos(ph), torch.sin(ph)])
        labels.extend([f"axis{axis}_cos", f"axis{axis}_sin"])
        orders.extend([0, 0])
    return Basis(name=name, matrix=torch.stack(columns, dim=1), labels=labels, orders=orders)


def separable_product_fourier_basis(
    d: torch.Tensor,
    omega: torch.Tensor,
    *,
    include_const: bool = True,
    name: str = "separable_product",
) -> Basis:
    # cos(a+b) and sin(a+b) written as products of axis phases.
    omega = omega.to(device=d.device, dtype=d.dtype)
    ax = d[:, 0] * omega[0]
    ay = d[:, 1] * omega[1]
    columns: list[torch.Tensor] = []
    labels: list[str] = []
    orders: list[int] = []
    if include_const:
        columns.append(torch.ones(d.shape[0], device=d.device, dtype=d.dtype))
        labels.append("const")
        orders.append(0)
    columns.extend(
        [
            torch.cos(ax) * torch.cos(ay),
            torch.sin(ax) * torch.sin(ay),
            torch.sin(ax) * torch.cos(ay),
            torch.cos(ax) * torch.sin(ay),
        ]
    )
    labels.extend(["cosx_cosy", "sinx_siny", "sinx_cosy", "cosx_siny"])
    orders.extend([0, 0, 0, 0])
    return Basis(name=name, matrix=torch.stack(columns, dim=1), labels=labels, orders=orders)


def directional_jet_basis(
    d: torch.Tensor,
    omegas: Iterable[torch.Tensor],
    directions: Iterable[torch.Tensor],
    orders_to_include: Iterable[int],
    *,
    scale: float,
    include_const: bool = True,
    name: str = "directional_jet",
) -> Basis:
    columns: list[torch.Tensor] = []
    labels: list[str] = []
    orders: list[int] = []
    if include_const:
        columns.append(torch.ones(d.shape[0], device=d.device, dtype=d.dtype))
        labels.append("const")
        orders.append(0)

    directions = [
        normalize_direction(direction.to(device=d.device, dtype=d.dtype).reshape(1, -1)).reshape(-1)
        for direction in directions
    ]
    order_list = sorted(set(int(order) for order in orders_to_include))
    for omega_idx, omega in enumerate(omegas):
        ph = phase(d, omega)
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)
        if 0 in order_list:
            columns.extend([cos_ph, sin_ph])
            labels.extend([f"w{omega_idx}_r0_cos", f"w{omega_idx}_r0_sin"])
            orders.extend([0, 0])
        for direction_idx, direction in enumerate(directions):
            coord = (d @ direction) / float(scale)
            for order in order_list:
                if order == 0:
                    continue
                poly = coord.pow(order)
                columns.extend([poly * cos_ph, poly * sin_ph])
                labels.extend(
                    [
                        f"w{omega_idx}_u{direction_idx}_r{order}_cos",
                        f"w{omega_idx}_u{direction_idx}_r{order}_sin",
                    ]
                )
                orders.extend([order, order])
    return Basis(name=name, matrix=torch.stack(columns, dim=1), labels=labels, orders=orders)


def full_multijet_basis(
    d: torch.Tensor,
    omega: torch.Tensor,
    *,
    max_order: int,
    scale: float,
    include_const: bool = True,
    name: str = "full_multijet",
) -> Basis:
    columns: list[torch.Tensor] = []
    labels: list[str] = []
    orders: list[int] = []
    if include_const:
        columns.append(torch.ones(d.shape[0], device=d.device, dtype=d.dtype))
        labels.append("const")
        orders.append(0)
    ph = phase(d, omega)
    cos_ph = torch.cos(ph)
    sin_ph = torch.sin(ph)
    for total in range(max_order + 1):
        for rx in range(total + 1):
            ry = total - rx
            if total == 0:
                columns.extend([cos_ph, sin_ph])
                labels.extend(["r00_cos", "r00_sin"])
                orders.extend([0, 0])
                continue
            poly = (d[:, 0] / float(scale)).pow(rx) * (d[:, 1] / float(scale)).pow(ry)
            columns.extend([poly * cos_ph, poly * sin_ph])
            labels.extend([f"r{rx}{ry}_cos", f"r{rx}{ry}_sin"])
            orders.extend([total, total])
    return Basis(name=name, matrix=torch.stack(columns, dim=1), labels=labels, orders=orders)

