from __future__ import annotations

import torch


def normalize_direction(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return a unit vector along the last dimension."""
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)


def abs_cosine_alignment(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Absolute cosine alignment for unoriented spectral / jet directions."""
    a_unit = normalize_direction(a, eps=eps)
    b_unit = normalize_direction(b, eps=eps)
    return torch.abs(torch.sum(a_unit * b_unit, dim=-1))

