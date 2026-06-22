from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ToricPJConfig:
    n_heads: int
    n_dims: int
    n_freqs: int = 4
    n_dirs: int = 4
    max_order: int = 2
    lengths: tuple[float, ...] | None = None
    use_lc: bool = True
    use_affine: bool = True
    use_damping: bool = False


class ToricPJBias(nn.Module):
    """Scalar Toric Fourier-jet / affine / LC relative attention bias.

    Input displacements have shape ``(..., n_dims)`` and output has shape
    ``(..., n_heads)``. The module is intentionally a scalar-bias component,
    not an exact rotary Q/K feature transform.
    """

    def __init__(self, config: ToricPJConfig) -> None:
        super().__init__()
        self.config = config
        if config.lengths is None:
            lengths = torch.ones(config.n_dims, dtype=torch.float32)
        else:
            if len(config.lengths) != config.n_dims:
                raise ValueError("lengths must match n_dims")
            lengths = torch.tensor(config.lengths, dtype=torch.float32)
        self.register_buffer("lengths", lengths)
        self.directional_scale = float(torch.linalg.norm(lengths).item())

        self.gate_logits = nn.Parameter(torch.zeros(config.n_heads, 3))
        self.omega = nn.Parameter(0.02 * torch.randn(config.n_heads, config.n_freqs, config.n_dims))
        self.raw_dirs = nn.Parameter(torch.randn(config.n_heads, config.n_dirs, config.n_dims))

        coeff_shape = (config.n_heads, config.n_freqs, config.n_dirs, config.max_order + 1)
        self.fj_cos = nn.Parameter(0.01 * torch.randn(coeff_shape))
        self.fj_sin = nn.Parameter(0.01 * torch.randn(coeff_shape))

        self.affine_bias = nn.Parameter(torch.zeros(config.n_heads))
        self.affine_slope = nn.Parameter(torch.zeros(config.n_heads, config.n_dims))

        self.lc_cos = nn.Parameter(0.01 * torch.randn(coeff_shape))
        self.lc_sin = nn.Parameter(0.01 * torch.randn(coeff_shape))

        if config.use_damping:
            self.raw_damping = nn.Parameter(torch.zeros(config.n_heads, config.n_freqs, config.n_dims))
        else:
            self.register_parameter("raw_damping", None)

    @property
    def normalized_dirs(self) -> torch.Tensor:
        return F.normalize(self.raw_dirs, p=2, dim=-1, eps=1e-12)

    @property
    def gate_values(self) -> torch.Tensor:
        return F.softplus(self.gate_logits)

    @property
    def gate_shares(self) -> torch.Tensor:
        gates = self.gate_values
        return gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(self, displacements: torch.Tensor) -> torch.Tensor:
        branches = self.branch_outputs(displacements)
        gates = self.gate_values
        gate_shape = (self.config.n_heads,) + (1,) * (branches["fj"].ndim - 1)
        return (
            gates[:, 0].reshape(gate_shape) * branches["fj"]
            + gates[:, 1].reshape(gate_shape) * branches["affine"]
            + gates[:, 2].reshape(gate_shape) * branches["lc"]
        ).movedim(0, -1)

    def branch_outputs(self, displacements: torch.Tensor) -> Mapping[str, torch.Tensor]:
        d = displacements.to(device=self.omega.device, dtype=self.omega.dtype)
        if d.shape[-1] != self.config.n_dims:
            raise ValueError(f"expected last dimension {self.config.n_dims}, got {d.shape[-1]}")

        fj = self._fourier_jet(d)
        affine = self._affine(d) if self.config.use_affine else torch.zeros_like(fj)
        lc = self._lc(d) if self.config.use_lc else torch.zeros_like(fj)
        return {"fj": fj, "affine": affine, "lc": lc}

    def _fourier_jet(self, d: torch.Tensor) -> torch.Tensor:
        phase = torch.einsum("...d,hfd->...hf", d, self.omega)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        damping = self._damping(d)
        dirs = self.normalized_dirs
        coord = torch.einsum("...d,hqd->...hq", d, dirs) / max(self.directional_scale, 1e-12)
        powers = self._powers(coord)
        weighted_cos = torch.einsum("hfqr,...hqr->...hfq", self.fj_cos, powers)
        weighted_sin = torch.einsum("hfqr,...hqr->...hfq", self.fj_sin, powers)
        atoms = (weighted_cos * cos_phase.unsqueeze(-1) + weighted_sin * sin_phase.unsqueeze(-1))
        atoms = atoms * damping.unsqueeze(-1)
        return atoms.sum(dim=(-1, -2)).movedim(-1, 0)

    def _affine(self, d: torch.Tensor) -> torch.Tensor:
        scaled = d / self.lengths.to(device=d.device, dtype=d.dtype).clamp_min(1e-12)
        value = self.affine_bias - torch.einsum("...d,hd->...h", scaled, self.affine_slope)
        return value.movedim(-1, 0)

    def _lc(self, d: torch.Tensor) -> torch.Tensor:
        lengths = self.lengths.to(device=d.device, dtype=d.dtype).clamp_min(1e-12)
        phi = lengths * torch.asinh(d / lengths)
        phase = torch.einsum("...d,hfd->...hf", phi, self.omega)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        dirs = self.normalized_dirs
        raw_coord = torch.einsum("...d,hqd->...hq", d, dirs)
        beta = raw_coord / torch.sqrt(raw_coord.square() + self.directional_scale**2)
        powers = self._powers(beta)
        weighted_cos = torch.einsum("hfqr,...hqr->...hfq", self.lc_cos, powers)
        weighted_sin = torch.einsum("hfqr,...hqr->...hfq", self.lc_sin, powers)
        atoms = weighted_cos * cos_phase.unsqueeze(-1) + weighted_sin * sin_phase.unsqueeze(-1)
        return atoms.sum(dim=(-1, -2)).movedim(-1, 0)

    def _damping(self, d: torch.Tensor) -> torch.Tensor:
        if self.raw_damping is None:
            shape = (*d.shape[:-1], self.config.n_heads, self.config.n_freqs)
            return torch.ones(shape, device=d.device, dtype=d.dtype)
        damping = F.softplus(self.raw_damping)
        lengths = self.lengths.to(device=d.device, dtype=d.dtype).clamp_min(1e-12)
        scaled = d / lengths
        exponent = torch.einsum("...d,hfd->...hf", scaled, damping)
        return torch.exp(-exponent)

    def _powers(self, coord: torch.Tensor) -> torch.Tensor:
        powers = [torch.ones_like(coord)]
        for order in range(1, self.config.max_order + 1):
            powers.append(coord.pow(order))
        return torch.stack(powers, dim=-1)
