from __future__ import annotations

import torch

from toric_pj.diagnostics.basis_projection import Basis, fit_basis, phase


def collision_target(d: torch.Tensor, omega: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    omega = omega.to(device=d.device, dtype=d.dtype)
    v = v.to(device=d.device, dtype=d.dtype)
    return (torch.cos(phase(d, omega + eps * v)) - torch.cos(phase(d, omega))) / eps


def two_frequency_basis(d: torch.Tensor, omega: torch.Tensor, v: torch.Tensor, eps: float) -> Basis:
    omega = omega.to(device=d.device, dtype=d.dtype)
    v = v.to(device=d.device, dtype=d.dtype)
    ph0 = phase(d, omega)
    ph1 = phase(d, omega + eps * v)
    matrix = torch.stack([torch.cos(ph0), torch.sin(ph0), torch.cos(ph1), torch.sin(ph1)], dim=1)
    return Basis(
        name="two_close_frequency",
        matrix=matrix,
        labels=["cos_w", "sin_w", "cos_w_eps", "sin_w_eps"],
        orders=[0, 0, 0, 0],
    )


def first_jet_limit_basis(d: torch.Tensor, omega: torch.Tensor, v: torch.Tensor) -> Basis:
    omega = omega.to(device=d.device, dtype=d.dtype)
    v = v.to(device=d.device, dtype=d.dtype)
    ph = phase(d, omega)
    directional_lag = d @ v
    matrix = torch.stack([-directional_lag * torch.sin(ph)], dim=1)
    return Basis(name="explicit_first_jet_limit", matrix=matrix, labels=["-vd_sin_w"], orders=[1])


def collision_curve(
    d: torch.Tensor,
    omega: torch.Tensor,
    v: torch.Tensor,
    eps_values: list[float],
    *,
    ridge: float = 1e-10,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for eps in eps_values:
        target = collision_target(d, omega, v, eps)
        two = fit_basis(two_frequency_basis(d, omega, v, eps), target, ridge=ridge)
        jet = fit_basis(first_jet_limit_basis(d, omega, v), target, ridge=ridge)
        rows.append(
            {
                "eps": float(eps),
                "two_freq_condition": two.condition,
                "two_freq_mse": two.mse,
                "two_freq_coeff_norm": two.coeff_norm,
                "jet_condition": jet.condition,
                "jet_mse": jet.mse,
                "jet_coeff_norm": jet.coeff_norm,
            }
        )
    return rows

