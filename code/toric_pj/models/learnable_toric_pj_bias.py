from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LearnableSpectralPJConfig:
    n_dims: int
    n_freqs: int = 1
    n_dirs: int = 1
    max_order: int = 2
    scale: float = 1.0
    include_affine: bool = True
    include_lc: bool = True
    learn_omega: bool = True
    learn_dirs: bool = True
    learn_lc_scale: bool = False
    lc_min_scale: float = 1e-3


class LearnableSpectralPJKernel(nn.Module):
    """Scalar spectral-jet kernel with learnable frequencies and directions.

    This module is intentionally a scalar relative-bias kernel. It is not an
    exact Q/K rotary feature transform. It is designed for v2 controlled
    experiments where the spectral geometry itself is learned.
    """

    def __init__(
        self,
        config: LearnableSpectralPJConfig,
        *,
        init_omega: torch.Tensor,
        init_dirs: torch.Tensor,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        device = device if device is not None else init_omega.device
        omega = init_omega.to(device=device, dtype=dtype).reshape(config.n_freqs, config.n_dims)
        dirs = init_dirs.to(device=device, dtype=dtype).reshape(config.n_dirs, config.n_dims)
        if omega.shape != (config.n_freqs, config.n_dims):
            raise ValueError("init_omega shape must be (n_freqs, n_dims)")
        if dirs.shape != (config.n_dirs, config.n_dims):
            raise ValueError("init_dirs shape must be (n_dirs, n_dims)")

        self.omega = nn.Parameter(omega.clone(), requires_grad=config.learn_omega)
        self.raw_dirs = nn.Parameter(dirs.clone(), requires_grad=config.learn_dirs)
        self.const = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.order0_cos = nn.Parameter(0.01 * torch.randn(config.n_freqs, device=device, dtype=dtype))
        self.order0_sin = nn.Parameter(0.01 * torch.randn(config.n_freqs, device=device, dtype=dtype))
        if config.max_order > 0:
            shape = (config.n_freqs, config.n_dirs, config.max_order)
            self.jet_cos = nn.Parameter(0.01 * torch.randn(shape, device=device, dtype=dtype))
            self.jet_sin = nn.Parameter(0.01 * torch.randn(shape, device=device, dtype=dtype))
            self.lc_cos = nn.Parameter(0.01 * torch.randn(shape, device=device, dtype=dtype))
            self.lc_sin = nn.Parameter(0.01 * torch.randn(shape, device=device, dtype=dtype))
        else:
            self.register_parameter("jet_cos", None)
            self.register_parameter("jet_sin", None)
            self.register_parameter("lc_cos", None)
            self.register_parameter("lc_sin", None)
        if config.include_affine:
            self.affine_slope = nn.Parameter(torch.zeros(config.n_dims, device=device, dtype=dtype))
        else:
            self.register_parameter("affine_slope", None)
        raw_lc = torch.log(torch.expm1(torch.tensor(float(config.scale), device=device, dtype=dtype).clamp_min(1e-6)))
        self.raw_lc_scale = nn.Parameter(raw_lc, requires_grad=config.learn_lc_scale)

    @property
    def normalized_dirs(self) -> torch.Tensor:
        return F.normalize(self.raw_dirs, p=2, dim=-1, eps=1e-12)

    @property
    def lc_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_lc_scale) + float(self.config.lc_min_scale)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        d = d.to(device=self.omega.device, dtype=self.omega.dtype)
        if d.shape[-1] != self.config.n_dims:
            raise ValueError(f"expected last dimension {self.config.n_dims}, got {d.shape[-1]}")
        output = self.const.expand(d.shape[:-1]).clone()
        phase = torch.einsum("...d,fd->...f", d, self.omega)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        output = output + torch.einsum("...f,f->...", cos_phase, self.order0_cos)
        output = output + torch.einsum("...f,f->...", sin_phase, self.order0_sin)

        if self.config.max_order > 0:
            dirs = self.normalized_dirs
            coord = torch.einsum("...d,qd->...q", d, dirs) / max(float(self.config.scale), 1e-12)
            for order in range(1, self.config.max_order + 1):
                poly = coord.pow(order)
                output = output + torch.einsum("...f,...q,fq->...", cos_phase, poly, self.jet_cos[:, :, order - 1])
                output = output + torch.einsum("...f,...q,fq->...", sin_phase, poly, self.jet_sin[:, :, order - 1])

        if self.config.include_affine and self.affine_slope is not None:
            output = output - torch.einsum("...d,d->...", d / max(float(self.config.scale), 1e-12), self.affine_slope)

        if self.config.include_lc and self.config.max_order > 0 and self.lc_cos is not None and self.lc_sin is not None:
            scale = self.lc_scale.clamp_min(1e-12)
            dirs = self.normalized_dirs
            raw = torch.einsum("...d,qd->...q", d, dirs)
            beta = raw / torch.sqrt(raw.square() + scale.square())
            phi = scale * torch.asinh(raw / scale)
            # Directional LC uses the mean spectral magnitude as a 1D phase rate.
            omega_mag = torch.linalg.norm(self.omega, dim=-1)
            lc_phase = torch.einsum("...q,f->...fq", phi, omega_mag)
            lc_cos_phase = torch.cos(lc_phase)
            lc_sin_phase = torch.sin(lc_phase)
            for order in range(1, self.config.max_order + 1):
                poly = beta.pow(order)
                output = output + torch.einsum(
                    "...fq,...q,fq->...", lc_cos_phase, poly, self.lc_cos[:, :, order - 1]
                )
                output = output + torch.einsum(
                    "...fq,...q,fq->...", lc_sin_phase, poly, self.lc_sin[:, :, order - 1]
                )
        return output

    def coefficient_parameters(self) -> list[nn.Parameter]:
        spectral_ids = {id(self.omega), id(self.raw_dirs), id(self.raw_lc_scale)}
        return [param for param in self.parameters() if id(param) not in spectral_ids]

    def geometry_parameters(self) -> list[nn.Parameter]:
        params = []
        if self.omega.requires_grad:
            params.append(self.omega)
        if self.raw_dirs.requires_grad:
            params.append(self.raw_dirs)
        if self.raw_lc_scale.requires_grad:
            params.append(self.raw_lc_scale)
        return params
