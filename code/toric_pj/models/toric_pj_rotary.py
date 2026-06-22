from __future__ import annotations

import torch


def nilpotent_shift(order: int, *, device: torch.device, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Nilpotent Jordan shift with N**(order + 1) = 0."""
    size = order + 1
    matrix = torch.zeros((size, size), device=device, dtype=dtype)
    for idx in range(order):
        matrix[idx, idx + 1] = 1.0
    return matrix


def exact_unipotent_action(
    displacement: torch.Tensor,
    direction: torch.Tensor,
    *,
    order: int,
    scale: float = 1.0,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Compute prod_a (I + u_a N / scale) ** d_a for integer displacement."""
    if displacement.ndim != 1:
        raise ValueError("displacement must be a 1D tensor")
    if direction.shape != displacement.shape:
        raise ValueError("direction must match displacement")
    device = displacement.device
    n = nilpotent_shift(order, device=device, dtype=dtype)
    eye = torch.eye(order + 1, device=device, dtype=dtype)
    action = eye.clone()
    for d_a, u_a in zip(displacement.tolist(), direction.to(device=device, dtype=torch.float64).tolist()):
        generator = eye + (float(u_a) / float(scale)) * n
        action = action @ torch.linalg.matrix_power(generator, int(d_a))
    return action


def exact_toric_pj_action(
    displacement: torch.Tensor,
    omega: torch.Tensor,
    direction: torch.Tensor,
    *,
    order: int,
    scale: float = 1.0,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Exact directional toric PJ relative action z**d prod_a(I+u_a N)**d_a."""
    omega = omega.to(device=displacement.device, dtype=torch.float64)
    phase = torch.sum(displacement.to(torch.float64) * omega)
    character = torch.exp(1j * phase).to(dtype=dtype)
    return character * exact_unipotent_action(
        displacement,
        direction,
        order=order,
        scale=scale,
        dtype=dtype,
    )


def semisimple_character(displacement: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    phase = torch.sum(displacement.to(torch.float64) * omega.to(device=displacement.device, dtype=torch.float64))
    return torch.exp(1j * phase)
