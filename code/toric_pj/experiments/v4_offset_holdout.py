from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from toric_pj.diagnostics.basis_projection import Basis, default_device
from toric_pj.diagnostics.relative_table_geometry import (
    fit_linear_basis,
    load_bias_npz,
    pairwise_to_relative_table,
    relative_table_to_pairwise,
    write_geometry_artifacts,
)
from toric_pj.experiments.v3_digits_transformer_scaling import autocast_context, write_csv
from toric_pj.experiments.v3_real_vision_scaling import (
    VISION_DATASETS,
    VisionRelPosTransformer,
    evaluate_classifier,
    evaluate_reconstruction,
    load_vision_dataset,
    normalize_patches,
    patchify,
    read_csv,
    sample_batch,
    vision_num_classes,
)
from toric_pj.experiments.v4_metric_toric_pj import DEFAULT_TEACHER, build_teacher_bases, parse_list


def offset_masks(side: int, radius: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.arange(side, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    d = positions[:, None, :] - positions[None, :, :]
    pairwise_visible = (d[..., 0].abs() <= radius) & (d[..., 1].abs() <= radius)

    vals = torch.arange(-(side - 1), side, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")
    table_visible = (gx.abs() <= radius) & (gy.abs() <= radius)
    return pairwise_visible, table_visible


def radial_offset_masks(side: int, radius: float, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.arange(side, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    d = positions[:, None, :] - positions[None, :, :]
    pairwise_visible = torch.sqrt(d[..., 0].square() + d[..., 1].square()) <= float(radius)

    vals = torch.arange(-(side - 1), side, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")
    table_visible = torch.sqrt(gx.square() + gy.square()) <= float(radius)
    return pairwise_visible, table_visible


def soft_radial_offset_weights(
    side: int,
    radius: float,
    *,
    width: float,
    floor: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(float(width), 1e-6)
    floor = min(max(float(floor), 0.0), 1.0)

    coords = torch.arange(side, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    d = positions[:, None, :] - positions[None, :, :]
    pair_radius = torch.sqrt(d[..., 0].square() + d[..., 1].square())
    pair_excess = (pair_radius - float(radius)).clamp_min(0.0)
    pair_weight = floor + (1.0 - floor) * torch.exp(-pair_excess / width)

    vals = torch.arange(-(side - 1), side, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")
    table_radius = torch.sqrt(gx.square() + gy.square())
    table_excess = (table_radius - float(radius)).clamp_min(0.0)
    table_weight = floor + (1.0 - floor) * torch.exp(-table_excess / width)
    return pair_weight, table_weight


def radial_offset_bin_indices(side: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    coords = torch.arange(side, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    d = positions[:, None, :] - positions[None, :, :]
    pair_radius = torch.sqrt(d[..., 0].square() + d[..., 1].square())
    pair_bins = torch.ceil(pair_radius).long()

    vals = torch.arange(-(side - 1), side, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")
    table_radius = torch.sqrt(gx.square() + gy.square())
    table_bins = torch.ceil(table_radius).long()
    max_bin = int(table_bins.max().detach().cpu())
    return pair_bins, table_bins, max_bin


def angular_offset_bin_indices(side: int, angular_bins: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    angular_bins = max(1, int(angular_bins))

    coords = torch.arange(side, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    d = positions[:, None, :] - positions[None, :, :]
    pair_angle = torch.atan2(d[..., 1], d[..., 0])
    pair_bins = torch.floor((pair_angle + math.pi) / (2.0 * math.pi) * angular_bins).long().clamp(0, angular_bins - 1)

    vals = torch.arange(-(side - 1), side, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")
    table_angle = torch.atan2(gy, gx)
    table_bins = torch.floor((table_angle + math.pi) / (2.0 * math.pi) * angular_bins).long().clamp(0, angular_bins - 1)
    return pair_bins, table_bins


def initial_learned_chart_logits(
    args: argparse.Namespace,
    max_bin: int,
    *,
    angular_bins: int = 1,
    device: torch.device,
) -> torch.nn.Parameter:
    floor = min(max(float(args.train_learned_chart_floor), 0.0), 0.999)
    bins = torch.arange(max_bin + 1, device=device, dtype=torch.float32)
    init = str(args.train_learned_chart_init)
    if init == "ones":
        weights = torch.ones_like(bins)
    elif init == "hard":
        weights = (bins <= float(args.train_learned_chart_radius)).to(dtype=torch.float32)
    elif init == "soft":
        width = max(float(args.train_learned_chart_width), 1e-6)
        excess = (bins - float(args.train_learned_chart_radius)).clamp_min(0.0)
        weights = floor + (1.0 - floor) * torch.exp(-excess / width)
    else:
        raise ValueError(f"unknown learned chart init: {init}")

    angular_bins = max(1, int(angular_bins))
    if angular_bins > 1:
        weights = weights[:, None].expand(max_bin + 1, angular_bins).clone()

    unit = ((weights - floor) / max(1.0 - floor, 1e-6)).clamp(1e-4, 1.0 - 1e-4)
    logits = torch.logit(unit)
    return torch.nn.Parameter(logits)


def learned_chart_pairwise_weights(
    logits: torch.Tensor,
    *,
    pairwise_bins: torch.Tensor,
    pairwise_angular_bins: torch.Tensor | None = None,
    pairwise_visible: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    floor = min(max(float(args.train_learned_chart_floor), 0.0), 0.999)
    bin_weights = floor + (1.0 - floor) * torch.sigmoid(logits)
    if logits.ndim == 1:
        weights = bin_weights[pairwise_bins]
    else:
        if pairwise_angular_bins is None:
            raise ValueError("angular learned chart requires pairwise angular bins")
        weights = bin_weights[pairwise_bins, pairwise_angular_bins]
    return pairwise_visible.to(device=weights.device, dtype=weights.dtype) * weights


def learned_chart_regularization_terms(
    logits: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    if logits is None:
        zero = torch.zeros(())
        return {
            "chart_angular_smoothness_mse": zero,
            "chart_radial_smoothness_mse": zero,
            "chart_monotonic_mse": zero,
        }

    floor = min(max(float(args.train_learned_chart_floor), 0.0), 0.999)
    weights = floor + (1.0 - floor) * torch.sigmoid(logits.float())
    if weights.ndim == 1:
        radial_diff = weights[1:] - weights[:-1]
        angular_smoothness = weights.new_zeros(())
    else:
        radial_diff = weights[1:, :] - weights[:-1, :]
        angular_diff = weights - torch.roll(weights, shifts=1, dims=1)
        angular_smoothness = angular_diff.square().mean()

    radial_smoothness = radial_diff.square().mean() if radial_diff.numel() else weights.new_zeros(())
    monotonic = radial_diff.clamp_min(0.0).square().mean() if radial_diff.numel() else weights.new_zeros(())
    return {
        "chart_angular_smoothness_mse": angular_smoothness,
        "chart_radial_smoothness_mse": radial_smoothness,
        "chart_monotonic_mse": monotonic,
    }


def learned_chart_weight_stats(
    logits: torch.Tensor | None,
    *,
    pairwise_bins: torch.Tensor | None,
    pairwise_angular_bins: torch.Tensor | None = None,
    pairwise_visible: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    if logits is None or pairwise_bins is None:
        return {
            "learned_chart_weight_mean": float("nan"),
            "learned_chart_weight_r_le_4_mean": float("nan"),
            "learned_chart_weight_r_4_5_mean": float("nan"),
            "learned_chart_weight_r_5_6_mean": float("nan"),
            "learned_chart_weight_angular_std_mean": float("nan"),
            "learned_chart_weight_angular_std_r_4_6_mean": float("nan"),
        }
    with torch.no_grad():
        weights = learned_chart_pairwise_weights(
            logits,
            pairwise_bins=pairwise_bins,
            pairwise_angular_bins=pairwise_angular_bins,
            pairwise_visible=pairwise_visible,
            args=args,
        ).float()
        visible = pairwise_visible.to(device=weights.device, dtype=torch.bool)
        bins = pairwise_bins.to(device=weights.device)
        floor = min(max(float(args.train_learned_chart_floor), 0.0), 0.999)
        bin_weights = floor + (1.0 - floor) * torch.sigmoid(logits.float())
        if bin_weights.ndim == 1:
            angular_std_mean = float("nan")
            angular_std_r_4_6 = float("nan")
        else:
            angular_std = bin_weights.std(dim=1, unbiased=False)
            angular_std_mean = float(angular_std.mean().detach().cpu())
            upper = min(bin_weights.shape[0], 7)
            angular_std_r_4_6 = float(angular_std[5:upper].mean().detach().cpu()) if upper > 5 else float("nan")

        def masked_mean(mask: torch.Tensor) -> float:
            full_mask = visible & mask
            if not bool(full_mask.any()):
                return float("nan")
            return float(weights[full_mask].mean().detach().cpu())

        return {
            "learned_chart_weight_mean": masked_mean(torch.ones_like(visible, dtype=torch.bool)),
            "learned_chart_weight_r_le_4_mean": masked_mean(bins <= 4),
            "learned_chart_weight_r_4_5_mean": masked_mean((bins > 4) & (bins <= 5)),
            "learned_chart_weight_r_5_6_mean": masked_mean((bins > 5) & (bins <= 6)),
            "learned_chart_weight_angular_std_mean": angular_std_mean,
            "learned_chart_weight_angular_std_r_4_6_mean": angular_std_r_4_6,
        }


CONTINUATION_LEARNED_POLICIES = {"learned_radial_coordinate", "local_continuation"}


def initial_continuation_logits(
    args: argparse.Namespace,
    max_bin: int,
    *,
    device: torch.device,
) -> torch.nn.Parameter:
    floor = min(max(float(args.train_continuation_floor), 0.0), 0.999)
    radius = float(args.train_continuation_radius)
    width = max(float(args.train_continuation_width), 1e-6)
    bins = torch.arange(max_bin + 1, device=device, dtype=torch.float32)
    excess = (bins - radius).clamp_min(0.0)
    weights = floor + (1.0 - floor) * torch.exp(-excess / width)
    weights = torch.where(bins <= radius, torch.ones_like(weights), weights)
    unit = ((weights - floor) / max(1.0 - floor, 1e-6)).clamp(1e-4, 1.0 - 1e-4)
    return torch.nn.Parameter(torch.logit(unit))


def continuation_pairwise_weights(
    logits: torch.Tensor,
    *,
    pairwise_bins: torch.Tensor,
    pairwise_visible: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    floor = min(max(float(args.train_continuation_floor), 0.0), 0.999)
    bin_weights = floor + (1.0 - floor) * torch.sigmoid(logits)
    weights = bin_weights[pairwise_bins]
    local = pairwise_bins.to(device=weights.device, dtype=torch.float32) <= float(args.train_continuation_radius)
    weights = torch.where(local, torch.ones_like(weights), weights)
    return pairwise_visible.to(device=weights.device, dtype=weights.dtype) * weights


def continuation_weight_stats(
    logits: torch.Tensor | None,
    *,
    pairwise_bins: torch.Tensor | None,
    pairwise_visible: torch.Tensor,
    pairwise_train_weights: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    if args.train_continuation_policy == "none" or pairwise_bins is None:
        return {
            "continuation_weight_mean": float("nan"),
            "continuation_weight_r_le_4_mean": float("nan"),
            "continuation_weight_r_4_5_mean": float("nan"),
            "continuation_weight_r_5_6_mean": float("nan"),
            "continuation_weight_r_gt_6_mean": float("nan"),
        }

    with torch.no_grad():
        if logits is not None:
            weights = continuation_pairwise_weights(
                logits,
                pairwise_bins=pairwise_bins,
                pairwise_visible=pairwise_visible,
                args=args,
            ).float()
        elif pairwise_train_weights is not None:
            weights = pairwise_train_weights.to(device=pairwise_visible.device, dtype=torch.float32)
        else:
            weights = pairwise_visible.to(dtype=torch.float32)

        visible = pairwise_visible.to(device=weights.device, dtype=torch.bool)
        bins = pairwise_bins.to(device=weights.device)

        def masked_mean(mask: torch.Tensor) -> float:
            full_mask = visible & mask
            if not bool(full_mask.any()):
                return float("nan")
            return float(weights[full_mask].mean().detach().cpu())

        return {
            "continuation_weight_mean": masked_mean(torch.ones_like(visible, dtype=torch.bool)),
            "continuation_weight_r_le_4_mean": masked_mean(bins <= 4),
            "continuation_weight_r_4_5_mean": masked_mean((bins > 4) & (bins <= 5)),
            "continuation_weight_r_5_6_mean": masked_mean((bins > 5) & (bins <= 6)),
            "continuation_weight_r_gt_6_mean": masked_mean(bins > 6),
        }


def continuation_config_fields(args: argparse.Namespace) -> dict[str, object]:
    return {
        "train_continuation_policy": args.train_continuation_policy,
        "train_continuation_radius": args.train_continuation_radius,
        "train_continuation_width": args.train_continuation_width,
        "train_continuation_floor": args.train_continuation_floor,
        "train_continuation_lr_mult": args.train_continuation_lr_mult,
        "train_continuation_weight_decay": args.train_continuation_weight_decay,
    }


def v9_regularizer_config_fields(args: argparse.Namespace) -> dict[str, object]:
    return {
        "boundary_value_match_weight": args.boundary_value_match_weight,
        "boundary_derivative_match_weight": args.boundary_derivative_match_weight,
        "far_laplacian_weight": args.far_laplacian_weight,
        "far_mixed_hessian_weight": args.far_mixed_hessian_weight,
        "far_spectral_energy_weight": args.far_spectral_energy_weight,
        "radial_tail_tv_weight": args.radial_tail_tv_weight,
        "local_identity_weight": args.local_identity_weight,
        "far_band_teacher_mse_weight": args.far_band_teacher_mse_weight,
        "far_band_teacher_range": args.far_band_teacher_range,
        "far_spectral_cutoff": args.far_spectral_cutoff,
        "boundary_band_width": args.boundary_band_width,
    }


def mask_teacher_tables(tables: torch.Tensor, table_visible: torch.Tensor) -> torch.Tensor:
    mask = table_visible.to(device=tables.device, dtype=tables.dtype)
    return tables * mask


def bias_overrides_from_model(
    model: VisionRelPosTransformer,
    *,
    pairwise_visible: torch.Tensor | None,
) -> list[torch.Tensor]:
    overrides: list[torch.Tensor] = []
    mask = None
    if pairwise_visible is not None:
        mask = pairwise_visible.to(device=model.basis_matrix.device, dtype=model.basis_matrix.dtype)
    for block in model.blocks:
        bias = torch.einsum("nf,hf->hn", model.basis_matrix, block.coeff).reshape(
            block.n_heads,
            model.n_positions,
            model.n_positions,
        )
        if mask is not None:
            bias = bias * mask.unsqueeze(0)
        overrides.append(bias)
    return overrides


def zero_bias_overrides(model: VisionRelPosTransformer) -> list[torch.Tensor]:
    return [
        torch.zeros(block.n_heads, model.n_positions, model.n_positions, device=model.basis_matrix.device)
        for block in model.blocks
    ]


def eval_control_bias_overrides(
    model: VisionRelPosTransformer,
    *,
    mode: str,
    holdout_radius: int,
    radial_decay_gamma: float | None = None,
    radial_keep_min: float | None = None,
    radial_keep_max: float | None = None,
) -> list[torch.Tensor]:
    side = int(math.sqrt(float(model.n_positions)))
    full = torch.stack(bias_overrides_from_model(model, pairwise_visible=None), dim=0).float()
    tables = pairwise_to_relative_table(full, side)
    vals = torch.arange(-(side - 1), side, device=tables.device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")

    if mode == "heldout_clamp":
        shift = side - 1
        idx_x = (vals.clamp(-holdout_radius, holdout_radius).long() + shift).clamp(0, 2 * side - 2)
        idx_y = (vals.clamp(-holdout_radius, holdout_radius).long() + shift).clamp(0, 2 * side - 2)
        controlled = tables[..., idx_x[:, None], idx_y[None, :]]
    elif mode == "radial_decay":
        gamma = 1.0 if radial_decay_gamma is None else float(radial_decay_gamma)
        excess = torch.maximum(gx.abs() - float(holdout_radius), gy.abs() - float(holdout_radius)).clamp_min(0.0)
        controlled = tables * torch.exp(-gamma * excess).to(device=tables.device, dtype=tables.dtype)
    elif mode == "radial_truncate":
        if radial_keep_max is None:
            raise ValueError("radial_truncate requires radial_keep_max")
        radius = torch.sqrt(gx.square() + gy.square())
        keep = radius <= float(radial_keep_max)
        controlled = tables * keep.to(device=tables.device, dtype=tables.dtype)
    elif mode == "radial_band":
        if radial_keep_min is None or radial_keep_max is None:
            raise ValueError("radial_band requires radial_keep_min and radial_keep_max")
        radius = torch.sqrt(gx.square() + gy.square())
        keep = (radius >= float(radial_keep_min)) & (radius < float(radial_keep_max))
        controlled = tables * keep.to(device=tables.device, dtype=tables.dtype)
    else:
        raise ValueError(f"unknown eval control mode: {mode}")

    pairwise = relative_table_to_pairwise(controlled, side)
    return [pairwise[layer] for layer in range(pairwise.shape[0])]


def component_radial_bias_overrides(
    model: VisionRelPosTransformer,
    *,
    mode: str,
    radius: float,
    layer: int,
    head: int | None = None,
) -> list[torch.Tensor]:
    side = int(math.sqrt(float(model.n_positions)))
    full = torch.stack(bias_overrides_from_model(model, pairwise_visible=None), dim=0).float()
    tables = pairwise_to_relative_table(full, side)
    vals = torch.arange(-(side - 1), side, device=tables.device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")
    keep = torch.sqrt(gx.square() + gy.square()) <= float(radius)
    keep = keep.to(device=tables.device, dtype=tables.dtype)
    n_layers, n_heads = tables.shape[:2]
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"layer index out of range: {layer}")
    if head is not None and (head < 0 or head >= n_heads):
        raise ValueError(f"head index out of range: {head}")

    if mode == "layer_radial_ablate":
        controlled = tables.clone()
        controlled[layer] = controlled[layer] * (1.0 - keep)
    elif mode == "layer_radial_keep":
        controlled = torch.zeros_like(tables)
        controlled[layer] = tables[layer] * keep
    elif mode == "head_radial_ablate":
        if head is None:
            raise ValueError("head_radial_ablate requires head")
        controlled = tables.clone()
        controlled[layer, head] = controlled[layer, head] * (1.0 - keep)
    elif mode == "head_radial_keep":
        if head is None:
            raise ValueError("head_radial_keep requires head")
        controlled = torch.zeros_like(tables)
        controlled[layer, head] = tables[layer, head] * keep
    else:
        raise ValueError(f"unknown component radial mode: {mode}")

    pairwise = relative_table_to_pairwise(controlled, side)
    return [pairwise[item] for item in range(pairwise.shape[0])]


def parse_component_spec(value: str) -> list[tuple[int, list[int]]]:
    components: list[tuple[int, list[int]]] = []
    if not value:
        return components
    for raw_component in value.split(","):
        raw_component = raw_component.strip()
        if not raw_component:
            continue
        if ":" not in raw_component:
            raise ValueError(f"component spec item must be layer:head+head, got {raw_component!r}")
        raw_layer, raw_heads = raw_component.split(":", 1)
        layer = int(raw_layer.strip().removeprefix("L").removeprefix("l"))
        heads = [
            int(item.strip().removeprefix("H").removeprefix("h"))
            for item in raw_heads.replace("|", "+").split("+")
            if item.strip()
        ]
        if not heads:
            raise ValueError(f"component spec item has no heads: {raw_component!r}")
        components.append((layer, heads))
    return components


def path_text_token(value: str) -> str:
    out = []
    for char in value:
        if char.isalnum():
            out.append(char)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "none"


def path_short_text_token(value: str, *, max_chars: int = 48) -> str:
    token = path_text_token(value)
    if len(token) <= max_chars:
        return token
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{token[:max_chars].rstrip('_')}_sha{digest}"


def train_component_bias_overrides(
    model: VisionRelPosTransformer,
    *,
    pairwise_visible: torch.Tensor,
    pairwise_train_weights: torch.Tensor | None = None,
    args: argparse.Namespace,
) -> list[torch.Tensor]:
    base_mask = pairwise_visible if pairwise_train_weights is None else pairwise_train_weights
    base = bias_overrides_from_model(model, pairwise_visible=base_mask)
    mode = str(args.train_component_mode)
    if mode == "full":
        return base

    components = parse_component_spec(args.train_component_spec)
    if not components:
        raise ValueError("train component mode requires --train-component-spec")

    side = int(math.sqrt(float(model.n_positions)))
    radial_pairwise_visible, _ = radial_offset_masks(
        side,
        float(args.train_component_radius),
        device=model.basis_matrix.device,
    )
    radial_mask = radial_pairwise_visible.to(device=model.basis_matrix.device, dtype=model.basis_matrix.dtype)

    if mode == "keep":
        controlled = [torch.zeros_like(item) for item in base]
        for layer, heads in components:
            if layer < 0 or layer >= len(base):
                raise ValueError(f"layer index out of range: {layer}")
            for head in heads:
                if head < 0 or head >= base[layer].shape[0]:
                    raise ValueError(f"head index out of range: {head}")
                controlled[layer][head] = base[layer][head] * radial_mask
        return controlled

    if mode == "ablate":
        controlled = [item.clone() for item in base]
        for layer, heads in components:
            if layer < 0 or layer >= len(base):
                raise ValueError(f"layer index out of range: {layer}")
            for head in heads:
                if head < 0 or head >= base[layer].shape[0]:
                    raise ValueError(f"head index out of range: {head}")
                controlled[layer][head] = controlled[layer][head] * (1.0 - radial_mask)
        return controlled

    raise ValueError(f"unknown train component mode: {mode}")


def parse_float_list(value: str) -> list[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_ranges(value: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    if not value:
        return ranges
    for item in value.split(","):
        text = item.strip().replace(":", "-")
        if not text:
            continue
        parts = text.split("-")
        if len(parts) != 2:
            raise ValueError(f"range must be lo-hi, got {item!r}")
        lo = float(parts[0].strip())
        hi = float(parts[1].strip())
        if hi <= lo:
            raise ValueError(f"range upper bound must exceed lower bound, got {item!r}")
        ranges.append((lo, hi))
    return ranges


def teacher_init_coeff_visible(
    basis: Basis,
    teacher_tables: torch.Tensor,
    *,
    pairwise_visible: torch.Tensor,
    side: int,
    depth: int,
    n_heads: int,
    ridge: float,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    if teacher_tables.shape[0] < depth or teacher_tables.shape[1] < n_heads:
        raise ValueError(
            f"teacher has {teacher_tables.shape[0]} layers/{teacher_tables.shape[1]} heads, "
            f"but student requests {depth}/{n_heads}"
        )
    target_tables = teacher_tables[:depth, :n_heads].to(device=basis.matrix.device, dtype=torch.float32)
    pairwise = relative_table_to_pairwise(target_tables, side)
    visible_flat = pairwise_visible.reshape(-1).to(device=basis.matrix.device)
    matrix = basis.matrix.to(device=basis.matrix.device, dtype=torch.float32)[visible_flat]
    coeff = torch.zeros(depth, n_heads, basis.matrix.shape[1], device=basis.matrix.device, dtype=torch.float32)
    rows: list[dict[str, object]] = []
    for layer in range(depth):
        for head in range(n_heads):
            target = pairwise[layer, head].reshape(-1)[visible_flat]
            _, item_coeff, mse, r2 = fit_linear_basis(matrix, target, ridge=ridge)
            coeff[layer, head] = item_coeff.reshape(-1).to(coeff.dtype)
            rows.append(
                {
                    "basis": basis.name,
                    "layer": layer,
                    "head": head,
                    "visible_teacher_init_mse": mse,
                    "visible_teacher_init_r2": r2,
                    "visible_teacher_init_ridge": ridge,
                    "num_features": int(basis.matrix.shape[1]),
                }
            )
    return coeff, rows


def bias_energy_stats(model: VisionRelPosTransformer, pairwise_visible: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        full = torch.stack(bias_overrides_from_model(model, pairwise_visible=None), dim=0).float()
        visible = pairwise_visible.to(device=full.device, dtype=torch.bool)
        heldout = ~visible
        visible_vals = full[..., visible]
        heldout_vals = full[..., heldout]
        return {
            "visible_bias_rms": float(torch.sqrt(torch.mean(visible_vals.square())).detach().cpu()),
            "heldout_bias_rms": float(torch.sqrt(torch.mean(heldout_vals.square())).detach().cpu()),
            "heldout_bias_abs_mean": float(heldout_vals.abs().mean().detach().cpu()),
            "heldout_pair_frac": float(heldout.float().mean().detach().cpu()),
        }


def bias_regularization_terms(
    model: VisionRelPosTransformer,
    *,
    pairwise_visible: torch.Tensor,
    table_visible: torch.Tensor,
    holdout_radius: int,
    radial_tail_gamma: float,
    teacher_tables: torch.Tensor | None = None,
    boundary_band_width: float = 1.0,
    far_spectral_cutoff: float = 0.25,
    far_band_teacher_range: str = "6-9",
) -> dict[str, torch.Tensor]:
    full = torch.stack(bias_overrides_from_model(model, pairwise_visible=None), dim=0).float()
    visible = pairwise_visible.to(device=full.device, dtype=torch.bool)
    heldout = ~visible
    heldout_mse = full[..., heldout].square().mean()

    side = int(math.sqrt(float(model.n_positions)))
    tables = pairwise_to_relative_table(full, side)
    table_mask = table_visible.to(device=tables.device, dtype=torch.bool)
    boundary_terms: list[torch.Tensor] = []

    row_cross = table_mask[1:, :] != table_mask[:-1, :]
    if bool(row_cross.any()):
        row_diff = tables[..., 1:, :] - tables[..., :-1, :]
        boundary_terms.append(row_diff[..., row_cross].square().mean())

    col_cross = table_mask[:, 1:] != table_mask[:, :-1]
    if bool(col_cross.any()):
        col_diff = tables[..., :, 1:] - tables[..., :, :-1]
        boundary_terms.append(col_diff[..., col_cross].square().mean())

    if boundary_terms:
        boundary_mse = torch.stack(boundary_terms).mean()
    else:
        boundary_mse = full.new_zeros(())

    vals = torch.arange(-(side - 1), side, device=tables.device, dtype=torch.float32)
    gx, gy = torch.meshgrid(vals, vals, indexing="ij")
    excess = torch.maximum(gx.abs() - float(holdout_radius), gy.abs() - float(holdout_radius)).clamp_min(0.0)
    tail = excess > 0
    if bool(tail.any()):
        decay = torch.exp(-float(radial_tail_gamma) * excess).to(device=tables.device, dtype=tables.dtype)
        removed = (1.0 - decay)[tail]
        radial_tail_mse = (tables[..., tail] * removed).square().mean()
    else:
        radial_tail_mse = full.new_zeros(())

    radius = torch.sqrt(gx.square() + gy.square())
    local = radius <= float(holdout_radius)
    euclidean_boundary_terms: list[torch.Tensor] = []

    row_cross = local[1:, :] != local[:-1, :]
    if bool(row_cross.any()):
        row_diff = tables[..., 1:, :] - tables[..., :-1, :]
        euclidean_boundary_terms.append(row_diff[..., row_cross].square().mean())

    col_cross = local[:, 1:] != local[:, :-1]
    if bool(col_cross.any()):
        col_diff = tables[..., :, 1:] - tables[..., :, :-1]
        euclidean_boundary_terms.append(col_diff[..., col_cross].square().mean())

    if euclidean_boundary_terms:
        boundary_value_match_mse = torch.stack(euclidean_boundary_terms).mean()
    else:
        boundary_value_match_mse = full.new_zeros(())

    band_width = max(float(boundary_band_width), 1e-6)
    derivative_terms: list[torch.Tensor] = []
    if tables.shape[-2] >= 3:
        row_second = tables[..., 2:, :] - 2.0 * tables[..., 1:-1, :] + tables[..., :-2, :]
        row_band = (radius[1:-1, :] - float(holdout_radius)).abs() <= band_width
        if bool(row_band.any()):
            derivative_terms.append(row_second[..., row_band].square().mean())
    if tables.shape[-1] >= 3:
        col_second = tables[..., :, 2:] - 2.0 * tables[..., :, 1:-1] + tables[..., :, :-2]
        col_band = (radius[:, 1:-1] - float(holdout_radius)).abs() <= band_width
        if bool(col_band.any()):
            derivative_terms.append(col_second[..., col_band].square().mean())
    if derivative_terms:
        boundary_derivative_match_mse = torch.stack(derivative_terms).mean()
    else:
        boundary_derivative_match_mse = full.new_zeros(())

    inner_radius = radius[1:-1, 1:-1]
    far_inner = inner_radius > float(holdout_radius)
    if bool(far_inner.any()):
        center = tables[..., 1:-1, 1:-1]
        laplacian = (
            tables[..., 2:, 1:-1]
            + tables[..., :-2, 1:-1]
            + tables[..., 1:-1, 2:]
            + tables[..., 1:-1, :-2]
            - 4.0 * center
        )
        far_laplacian_energy = laplacian[..., far_inner].square().mean()
        mixed_hessian = (
            tables[..., 2:, 2:]
            - tables[..., 2:, :-2]
            - tables[..., :-2, 2:]
            + tables[..., :-2, :-2]
        ) * 0.25
        far_mixed_hessian_energy = mixed_hessian[..., far_inner].square().mean()
    else:
        far_laplacian_energy = full.new_zeros(())
        far_mixed_hessian_energy = full.new_zeros(())

    far = radius > float(holdout_radius)
    if bool(far.any()):
        far_mask = far.to(device=tables.device, dtype=tables.dtype)
        far_count = far_mask.sum().clamp_min(1.0)
        far_mean = (tables * far_mask).sum(dim=(-2, -1), keepdim=True) / far_count
        centered_far = (tables - far_mean) * far_mask
        spectrum = torch.fft.rfft2(centered_far, norm="ortho")
        fx = torch.fft.fftfreq(tables.shape[-2], device=tables.device, dtype=torch.float32)
        fy = torch.fft.rfftfreq(tables.shape[-1], device=tables.device, dtype=torch.float32)
        freq_radius = torch.sqrt(fx[:, None].square() + fy[None, :].square())
        high_freq = freq_radius >= float(far_spectral_cutoff)
        if bool(high_freq.any()):
            far_spectral_energy = spectrum.abs().square()[..., high_freq].mean()
        else:
            far_spectral_energy = full.new_zeros(())
    else:
        far_spectral_energy = full.new_zeros(())

    radial_bins = torch.ceil(radius).long()
    shell_means: list[torch.Tensor] = []
    shell_energy: list[torch.Tensor] = []
    min_bin = int(math.floor(float(holdout_radius))) + 1
    max_bin = int(radial_bins.max().detach().cpu())
    for radial_bin in range(min_bin, max_bin + 1):
        shell = radial_bins == radial_bin
        if bool(shell.any()):
            shell_values = tables[..., shell]
            shell_means.append(shell_values.mean(dim=-1))
            shell_energy.append(shell_values.square().mean(dim=-1))
    if len(shell_means) >= 2:
        shell_mean_stack = torch.stack(shell_means, dim=-1)
        radial_tail_tv = (shell_mean_stack[..., 1:] - shell_mean_stack[..., :-1]).abs().mean()
        shell_energy_stack = torch.stack(shell_energy, dim=-1)
        radial_tail_monotonic_mse = (
            shell_energy_stack[..., 1:] - shell_energy_stack[..., :-1]
        ).clamp_min(0.0).square().mean()
    else:
        radial_tail_tv = full.new_zeros(())
        radial_tail_monotonic_mse = full.new_zeros(())

    if teacher_tables is not None:
        target = teacher_tables[: tables.shape[0], : tables.shape[1]].to(device=tables.device, dtype=tables.dtype)
        if target.shape[-2:] != tables.shape[-2:]:
            raise ValueError(f"teacher table shape {tuple(target.shape[-2:])} != student table shape {tuple(tables.shape[-2:])}")
        local_identity_mse = (tables[..., local] - target[..., local]).square().mean()
        far_band = torch.zeros_like(radius, dtype=torch.bool)
        for lo, hi in parse_float_ranges(str(far_band_teacher_range)):
            far_band |= (radius >= float(lo)) & (radius < float(hi))
        if bool(far_band.any()):
            far_band_teacher_mse = (tables[..., far_band] - target[..., far_band]).square().mean()
        else:
            far_band_teacher_mse = full.new_zeros(())
    else:
        local_identity_mse = full.new_zeros(())
        far_band_teacher_mse = full.new_zeros(())

    return {
        "heldout_bias_mse": heldout_mse,
        "boundary_smoothness_mse": boundary_mse,
        "radial_tail_mse": radial_tail_mse,
        "boundary_value_match_mse": boundary_value_match_mse,
        "boundary_derivative_match_mse": boundary_derivative_match_mse,
        "far_laplacian_energy": far_laplacian_energy,
        "far_mixed_hessian_energy": far_mixed_hessian_energy,
        "far_spectral_energy": far_spectral_energy,
        "radial_tail_tv": radial_tail_tv,
        "radial_tail_monotonic_mse": radial_tail_monotonic_mse,
        "local_identity_mse": local_identity_mse,
        "far_band_teacher_mse": far_band_teacher_mse,
    }


def coeff_l2_term(model: VisionRelPosTransformer) -> torch.Tensor:
    coeff = torch.stack([block.coeff for block in model.blocks], dim=0)
    return coeff.square().mean()


def regularization_scale(step: int, args: argparse.Namespace) -> float:
    start = max(0, int(args.reg_start_step))
    if step < start:
        return 0.0
    if args.reg_schedule == "constant":
        return 1.0
    ramp_steps = max(1, int(args.reg_ramp_steps))
    progress = min(1.0, float(step - start + 1) / float(ramp_steps))
    if args.reg_schedule == "linear":
        return progress
    if args.reg_schedule == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * progress)
    raise ValueError(f"unknown reg schedule: {args.reg_schedule}")


def active_radial_tail_gamma(args: argparse.Namespace) -> float:
    return float(args.radial_tail_gamma) if float(args.radial_tail_weight) != 0.0 else 0.0


def train_soft_radial_anneal_progress(step: int, args: argparse.Namespace) -> float:
    ramp_steps = max(0, int(args.train_soft_radial_anneal_ramp_steps))
    if args.train_soft_radial_visible_radius is None or ramp_steps <= 0:
        return 1.0
    start = max(0, int(args.train_soft_radial_anneal_start_step))
    if step < start:
        return 0.0
    return min(1.0, float(step - start) / float(ramp_steps))


def pairwise_train_weights_for_step(
    step: int,
    *,
    pairwise_train_weights: torch.Tensor | None,
    pairwise_train_weights_start: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    if pairwise_train_weights is None or pairwise_train_weights_start is None:
        return pairwise_train_weights
    progress = train_soft_radial_anneal_progress(step, args)
    if progress <= 0.0:
        return pairwise_train_weights_start
    if progress >= 1.0:
        return pairwise_train_weights
    return pairwise_train_weights_start + (pairwise_train_weights - pairwise_train_weights_start) * progress


def train_policy_pairwise_weights(
    step: int,
    *,
    pairwise_visible: torch.Tensor,
    pairwise_train_weights: torch.Tensor | None,
    pairwise_train_weights_start: torch.Tensor | None,
    learned_chart_logits: torch.Tensor | None,
    continuation_logits: torch.Tensor | None,
    pairwise_radial_bins: torch.Tensor | None,
    pairwise_angular_bins: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor | None:
    if continuation_logits is not None:
        if pairwise_radial_bins is None:
            raise ValueError("continuation policy requires pairwise radial bins")
        return continuation_pairwise_weights(
            continuation_logits,
            pairwise_bins=pairwise_radial_bins,
            pairwise_visible=pairwise_visible,
            args=args,
        )
    if learned_chart_logits is not None:
        if pairwise_radial_bins is None:
            raise ValueError("learned chart policy requires pairwise radial bins")
        return learned_chart_pairwise_weights(
            learned_chart_logits,
            pairwise_bins=pairwise_radial_bins,
            pairwise_angular_bins=pairwise_angular_bins,
            pairwise_visible=pairwise_visible,
            args=args,
        )
    return pairwise_train_weights_for_step(
        step,
        pairwise_train_weights=pairwise_train_weights,
        pairwise_train_weights_start=pairwise_train_weights_start,
        args=args,
    )


def path_float_token(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def train_radial_visible_radius_value(args: argparse.Namespace) -> float:
    if args.train_radial_visible_radius is None:
        return -1.0
    return float(args.train_radial_visible_radius)


def train_soft_radial_visible_radius_value(args: argparse.Namespace) -> float:
    if args.train_soft_radial_visible_radius is None:
        return -1.0
    return float(args.train_soft_radial_visible_radius)


def row_float_key(row: dict[str, object], key: str, default: float = -1.0) -> float:
    value = row.get(key, default)
    if value is None or value == "":
        return default
    return float(value)


def mean_optional_float(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row.get(key, float("nan"))) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def state_dict_to_cpu(model: VisionRelPosTransformer) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def export_config_slug(args: argparse.Namespace) -> str:
    train_radius = ""
    if args.train_radial_visible_radius is not None:
        train_radius = f"_trainr{path_float_token(args.train_radial_visible_radius)}"
    soft_radius = ""
    if args.train_soft_radial_visible_radius is not None:
        soft_radius = (
            f"_softtrainr{path_float_token(args.train_soft_radial_visible_radius)}"
            f"w{path_float_token(args.train_soft_radial_width)}"
            f"f{path_float_token(args.train_soft_radial_floor)}"
        )
        if int(args.train_soft_radial_anneal_ramp_steps) > 0:
            soft_radius += (
                f"_anneals{int(args.train_soft_radial_anneal_start_step)}"
                f"r{int(args.train_soft_radial_anneal_ramp_steps)}"
            )
    train_component = ""
    if args.train_component_mode != "full":
        train_component = (
            f"_tc{path_text_token(args.train_component_mode)}"
            f"_tcs{path_short_text_token(args.train_component_spec)}"
            f"_tcr{path_float_token(args.train_component_radius)}"
        )
    learned_chart = ""
    if args.train_learned_chart_policy != "none":
        learned_chart = (
            f"_lc{path_text_token(args.train_learned_chart_policy)}"
            f"i{path_text_token(args.train_learned_chart_init)}"
            f"r{path_float_token(args.train_learned_chart_radius)}"
            f"w{path_float_token(args.train_learned_chart_width)}"
            f"f{path_float_token(args.train_learned_chart_floor)}"
            f"a{int(args.train_learned_chart_angular_bins)}"
            f"lr{path_float_token(args.train_learned_chart_lr_mult)}"
            f"as{path_float_token(args.train_learned_chart_angular_smoothness_weight)}"
            f"rs{path_float_token(args.train_learned_chart_radial_smoothness_weight)}"
            f"m{path_float_token(args.train_learned_chart_monotonic_weight)}"
        )
    continuation = ""
    if args.train_continuation_policy != "none":
        continuation = (
            f"_cont{path_text_token(args.train_continuation_policy)}"
            f"r{path_float_token(args.train_continuation_radius)}"
            f"w{path_float_token(args.train_continuation_width)}"
            f"f{path_float_token(args.train_continuation_floor)}"
            f"lr{path_float_token(args.train_continuation_lr_mult)}"
        )
    v9_reg = ""
    v9_reg_values = (
        args.boundary_value_match_weight,
        args.boundary_derivative_match_weight,
        args.far_laplacian_weight,
        args.far_mixed_hessian_weight,
        args.far_spectral_energy_weight,
        args.radial_tail_tv_weight,
        args.local_identity_weight,
        args.far_band_teacher_mse_weight,
    )
    if any(float(value) != 0.0 for value in v9_reg_values):
        v9_reg = (
            f"_v9bv{path_float_token(args.boundary_value_match_weight)}"
            f"bd{path_float_token(args.boundary_derivative_match_weight)}"
            f"fl{path_float_token(args.far_laplacian_weight)}"
            f"fm{path_float_token(args.far_mixed_hessian_weight)}"
            f"fs{path_float_token(args.far_spectral_energy_weight)}"
            f"tv{path_float_token(args.radial_tail_tv_weight)}"
            f"li{path_float_token(args.local_identity_weight)}"
            f"fbt{path_float_token(args.far_band_teacher_mse_weight)}"
            f"r{path_short_text_token(args.far_band_teacher_range, max_chars=16)}"
        )
    return (
        f"h{path_float_token(args.heldout_bias_mse_weight)}"
        f"_b{path_float_token(args.boundary_smoothness_weight)}"
        f"_cl2{path_float_token(args.coeff_l2_weight)}"
        f"_tail{path_float_token(args.radial_tail_weight)}"
        f"_g{path_float_token(active_radial_tail_gamma(args))}"
        f"_{args.reg_schedule}"
        f"_s{int(args.reg_start_step)}"
        f"_r{int(args.reg_ramp_steps)}"
        f"{train_radius}"
        f"{soft_radius}"
        f"{train_component}"
        f"{learned_chart}"
        f"{continuation}"
        f"{v9_reg}"
    )


def evaluate_holdout_task(
    model: VisionRelPosTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    args: argparse.Namespace,
    seed: int,
    bias_overrides: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    if args.task == "classification":
        return evaluate_classifier(model, x, y, args=args, bias_overrides=bias_overrides)
    if args.task == "reconstruction":
        return evaluate_reconstruction(model, x, args=args, seed=seed, bias_overrides=bias_overrides)
    raise ValueError(f"unknown task: {args.task}")


def train_holdout_task(
    basis: Basis,
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    patch_dim: int,
    pairwise_visible: torch.Tensor,
    pairwise_train_weights: torch.Tensor | None,
    pairwise_train_weights_start: torch.Tensor | None,
    pairwise_radial_bins: torch.Tensor | None,
    pairwise_angular_bins: torch.Tensor | None,
    table_visible: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    init_coeff: torch.Tensor | None,
    teacher_tables: torch.Tensor | None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=train_x.device)
    gen.manual_seed(seed + 77)
    model = VisionRelPosTransformer(
        basis,
        n_positions=train_x.shape[1],
        patch_dim=patch_dim,
        dim=args.dim,
        n_heads=args.n_heads,
        depth=args.depth,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        n_classes=vision_num_classes(args.dataset),
    ).to(train_x.device)
    if init_coeff is not None:
        if tuple(init_coeff.shape) != (len(model.blocks), args.n_heads, basis.matrix.shape[1]):
            raise ValueError(
                "init_coeff must have shape "
                f"{(len(model.blocks), args.n_heads, basis.matrix.shape[1])}, got {tuple(init_coeff.shape)}"
            )
        with torch.no_grad():
            for layer, block in enumerate(model.blocks):
                block.coeff.copy_(init_coeff[layer].to(device=block.coeff.device, dtype=block.coeff.dtype))

    learned_chart_logits = None
    if args.train_learned_chart_policy == "radial_bins":
        _, _, max_bin = radial_offset_bin_indices(int(math.sqrt(float(train_x.shape[1]))), device=train_x.device)
        learned_chart_logits = initial_learned_chart_logits(args, max_bin, device=train_x.device)
    elif args.train_learned_chart_policy == "radial_angular_bins":
        _, _, max_bin = radial_offset_bin_indices(int(math.sqrt(float(train_x.shape[1]))), device=train_x.device)
        learned_chart_logits = initial_learned_chart_logits(
            args,
            max_bin,
            angular_bins=args.train_learned_chart_angular_bins,
            device=train_x.device,
        )
    elif args.train_learned_chart_policy != "none":
        raise ValueError(f"unknown learned chart policy: {args.train_learned_chart_policy}")

    continuation_logits = None
    if args.train_continuation_policy in CONTINUATION_LEARNED_POLICIES:
        _, _, max_bin = radial_offset_bin_indices(int(math.sqrt(float(train_x.shape[1]))), device=train_x.device)
        continuation_logits = initial_continuation_logits(args, max_bin, device=train_x.device)
    elif args.train_continuation_policy not in {"none", "low_curvature_tail", "boundary_matching"}:
        raise ValueError(f"unknown continuation policy: {args.train_continuation_policy}")

    param_groups = [{"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}]
    if learned_chart_logits is not None:
        param_groups.append(
            {
                "params": [learned_chart_logits],
                "lr": args.lr * float(args.train_learned_chart_lr_mult),
                "weight_decay": args.train_learned_chart_weight_decay,
            }
        )
    if continuation_logits is not None:
        param_groups.append(
            {
                "params": [continuation_logits],
                "lr": args.lr * float(args.train_continuation_lr_mult),
                "weight_decay": args.train_continuation_weight_decay,
            }
        )
    opt = torch.optim.AdamW(param_groups)
    decay_steps = max(1, args.lr_decay_steps or args.steps)
    if args.lr_schedule == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    elif args.lr_schedule == "cosine_hold":
        min_ratio = args.lr_min_ratio

        def lr_lambda(step: int) -> float:
            progress = min(float(step), float(decay_steps)) / float(decay_steps)
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=decay_steps, eta_min=args.lr * args.lr_min_ratio)

    curves: list[dict[str, object]] = []
    best_score = -1e9
    best_step = -1
    best_loss = float("nan")
    best_visible_score = float("nan")
    best_state_dict: dict[str, torch.Tensor] | None = None
    wall_start = time.time()
    metric_name = "accuracy" if args.task == "classification" else "masked_patch_r2"
    for step in range(args.steps):
        batch, labels = sample_batch(train_x, train_y, batch_size=args.batch_size, generator=gen)
        opt.zero_grad(set_to_none=True)
        step_train_weights = train_policy_pairwise_weights(
            step,
            pairwise_visible=pairwise_visible,
            pairwise_train_weights=pairwise_train_weights,
            pairwise_train_weights_start=pairwise_train_weights_start,
            learned_chart_logits=learned_chart_logits,
            continuation_logits=continuation_logits,
            pairwise_radial_bins=pairwise_radial_bins,
            pairwise_angular_bins=pairwise_angular_bins,
            args=args,
        )
        train_overrides = train_component_bias_overrides(
            model,
            pairwise_visible=pairwise_visible,
            pairwise_train_weights=step_train_weights,
            args=args,
        )
        with autocast_context(train_x.device, args.amp):
            if args.task == "classification":
                logits = model.forward_classification(batch, bias_overrides=train_overrides)
                task_loss = F.cross_entropy(logits.float(), labels)
            else:
                mask = torch.rand(batch.shape[:2], device=batch.device, generator=gen) < args.mask_rate
                pred = model.forward_reconstruction(batch, mask, bias_overrides=train_overrides)
                task_loss = torch.mean((pred[mask] - batch[mask]).square())
        reg_terms = bias_regularization_terms(
            model,
            pairwise_visible=pairwise_visible,
            table_visible=table_visible,
            holdout_radius=args.holdout_radius,
            radial_tail_gamma=args.radial_tail_gamma,
            teacher_tables=teacher_tables,
            boundary_band_width=args.boundary_band_width,
            far_spectral_cutoff=args.far_spectral_cutoff,
            far_band_teacher_range=args.far_band_teacher_range,
        )
        chart_reg_terms = learned_chart_regularization_terms(learned_chart_logits, args)
        if learned_chart_logits is None:
            chart_reg_terms = {key: task_loss.new_zeros(()) for key in chart_reg_terms}
        reg_scale = regularization_scale(step, args)
        heldout_weight = args.heldout_bias_mse_weight * reg_scale
        boundary_weight = args.boundary_smoothness_weight * reg_scale
        coeff_l2_weight = args.coeff_l2_weight * reg_scale
        radial_tail_weight = args.radial_tail_weight * reg_scale
        boundary_value_weight = args.boundary_value_match_weight * reg_scale
        boundary_derivative_weight = args.boundary_derivative_match_weight * reg_scale
        far_laplacian_weight = args.far_laplacian_weight * reg_scale
        far_mixed_hessian_weight = args.far_mixed_hessian_weight * reg_scale
        far_spectral_energy_weight = args.far_spectral_energy_weight * reg_scale
        radial_tail_tv_weight = args.radial_tail_tv_weight * reg_scale
        local_identity_weight = args.local_identity_weight * reg_scale
        far_band_teacher_mse_weight = args.far_band_teacher_mse_weight * reg_scale
        chart_angular_smoothness_weight = args.train_learned_chart_angular_smoothness_weight * reg_scale
        chart_radial_smoothness_weight = args.train_learned_chart_radial_smoothness_weight * reg_scale
        chart_monotonic_weight = args.train_learned_chart_monotonic_weight * reg_scale
        coeff_l2 = coeff_l2_term(model)
        loss = (
            task_loss
            + heldout_weight * reg_terms["heldout_bias_mse"]
            + boundary_weight * reg_terms["boundary_smoothness_mse"]
            + coeff_l2_weight * coeff_l2
            + radial_tail_weight * reg_terms["radial_tail_mse"]
            + boundary_value_weight * reg_terms["boundary_value_match_mse"]
            + boundary_derivative_weight * reg_terms["boundary_derivative_match_mse"]
            + far_laplacian_weight * reg_terms["far_laplacian_energy"]
            + far_mixed_hessian_weight * reg_terms["far_mixed_hessian_energy"]
            + far_spectral_energy_weight * reg_terms["far_spectral_energy"]
            + radial_tail_tv_weight * reg_terms["radial_tail_tv"]
            + local_identity_weight * reg_terms["local_identity_mse"]
            + far_band_teacher_mse_weight * reg_terms["far_band_teacher_mse"]
            + chart_angular_smoothness_weight * chart_reg_terms["chart_angular_smoothness_mse"]
            + chart_radial_smoothness_weight * chart_reg_terms["chart_radial_smoothness_mse"]
            + chart_monotonic_weight * chart_reg_terms["chart_monotonic_mse"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            eval_train_weights = train_policy_pairwise_weights(
                step,
                pairwise_visible=pairwise_visible,
                pairwise_train_weights=pairwise_train_weights,
                pairwise_train_weights_start=pairwise_train_weights_start,
                learned_chart_logits=learned_chart_logits,
                continuation_logits=continuation_logits,
                pairwise_radial_bins=pairwise_radial_bins,
                pairwise_angular_bins=pairwise_angular_bins,
                args=args,
            )
            train_metric = evaluate_holdout_task(
                model,
                train_x[: min(args.eval_subset, train_x.shape[0])],
                train_y[: min(args.eval_subset, train_y.shape[0])],
                args=args,
                seed=seed + step,
            )
            full_metric = evaluate_holdout_task(model, test_x, test_y, args=args, seed=seed + 1000 + step)
            visible_metric = evaluate_holdout_task(
                model,
                test_x,
                test_y,
                args=args,
                seed=seed + 1000 + step,
                bias_overrides=train_component_bias_overrides(
                    model,
                    pairwise_visible=pairwise_visible,
                    pairwise_train_weights=eval_train_weights,
                    args=args,
                ),
            )
            if full_metric["score"] > best_score:
                best_score = full_metric["score"]
                best_step = step
                best_loss = full_metric["loss"]
                best_visible_score = visible_metric["score"]
                if args.export_checkpoint:
                    best_state_dict = state_dict_to_cpu(model)
            curves.append(
                {
                    "dataset": args.dataset,
                    "task": args.task,
                    "basis": basis.name,
                    "seed": seed,
                    "train_radial_visible_radius": train_radial_visible_radius_value(args),
                    "train_soft_radial_visible_radius": train_soft_radial_visible_radius_value(args),
                    "train_soft_radial_width": args.train_soft_radial_width,
                    "train_soft_radial_floor": args.train_soft_radial_floor,
                    "train_soft_radial_anneal_start_step": args.train_soft_radial_anneal_start_step,
                    "train_soft_radial_anneal_ramp_steps": args.train_soft_radial_anneal_ramp_steps,
                    "train_soft_radial_anneal_progress": train_soft_radial_anneal_progress(step, args),
                    "train_learned_chart_policy": args.train_learned_chart_policy,
                    "train_learned_chart_init": args.train_learned_chart_init,
                    "train_learned_chart_radius": args.train_learned_chart_radius,
                    "train_learned_chart_width": args.train_learned_chart_width,
                    "train_learned_chart_floor": args.train_learned_chart_floor,
                    "train_learned_chart_angular_bins": args.train_learned_chart_angular_bins,
                    "train_learned_chart_lr_mult": args.train_learned_chart_lr_mult,
                    "train_learned_chart_weight_decay": args.train_learned_chart_weight_decay,
                    "train_learned_chart_angular_smoothness_weight": args.train_learned_chart_angular_smoothness_weight,
                    "train_learned_chart_radial_smoothness_weight": args.train_learned_chart_radial_smoothness_weight,
                    "train_learned_chart_monotonic_weight": args.train_learned_chart_monotonic_weight,
                    **continuation_config_fields(args),
                    **v9_regularizer_config_fields(args),
                    **learned_chart_weight_stats(
                        learned_chart_logits,
                        pairwise_bins=pairwise_radial_bins,
                        pairwise_angular_bins=pairwise_angular_bins,
                        pairwise_visible=pairwise_visible,
                        args=args,
                    ),
                    **continuation_weight_stats(
                        continuation_logits,
                        pairwise_bins=pairwise_radial_bins,
                        pairwise_visible=pairwise_visible,
                        pairwise_train_weights=eval_train_weights,
                        args=args,
                    ),
                    "train_component_mode": args.train_component_mode,
                    "train_component_spec": args.train_component_spec,
                    "train_component_radius": args.train_component_radius,
                    "step": step,
                    "metric": metric_name,
                    "train_score": train_metric["score"],
                    "full_bias_score": full_metric["score"],
                    "visible_only_score": visible_metric["score"],
                    "full_bias_loss": full_metric["loss"],
                    "visible_only_loss": visible_metric["loss"],
                    "reg_scale": reg_scale,
                    "effective_heldout_bias_mse_weight": heldout_weight,
                    "effective_boundary_smoothness_weight": boundary_weight,
                    "effective_coeff_l2_weight": coeff_l2_weight,
                    "effective_radial_tail_weight": radial_tail_weight,
                    "effective_boundary_value_match_weight": boundary_value_weight,
                    "effective_boundary_derivative_match_weight": boundary_derivative_weight,
                    "effective_far_laplacian_weight": far_laplacian_weight,
                    "effective_far_mixed_hessian_weight": far_mixed_hessian_weight,
                    "effective_far_spectral_energy_weight": far_spectral_energy_weight,
                    "effective_radial_tail_tv_weight": radial_tail_tv_weight,
                    "effective_local_identity_weight": local_identity_weight,
                    "effective_far_band_teacher_mse_weight": far_band_teacher_mse_weight,
                    "effective_chart_angular_smoothness_weight": chart_angular_smoothness_weight,
                    "effective_chart_radial_smoothness_weight": chart_radial_smoothness_weight,
                    "effective_chart_monotonic_weight": chart_monotonic_weight,
                    "coeff_l2_weight": args.coeff_l2_weight,
                    "radial_tail_weight": args.radial_tail_weight,
                    "radial_tail_gamma": active_radial_tail_gamma(args),
                    "radial_tail_mse": float(reg_terms["radial_tail_mse"].detach().cpu()),
                    "boundary_jump_mse": float(reg_terms["boundary_value_match_mse"].detach().cpu()),
                    "boundary_derivative_jump_mse": float(reg_terms["boundary_derivative_match_mse"].detach().cpu()),
                    "far_laplacian_energy": float(reg_terms["far_laplacian_energy"].detach().cpu()),
                    "far_mixed_hessian_energy": float(reg_terms["far_mixed_hessian_energy"].detach().cpu()),
                    "far_spectral_energy": float(reg_terms["far_spectral_energy"].detach().cpu()),
                    "radial_tail_tv": float(reg_terms["radial_tail_tv"].detach().cpu()),
                    "radial_tail_monotonic_mse": float(reg_terms["radial_tail_monotonic_mse"].detach().cpu()),
                    "local_identity_mse": float(reg_terms["local_identity_mse"].detach().cpu()),
                    "far_band_teacher_mse": float(reg_terms["far_band_teacher_mse"].detach().cpu()),
                    "chart_angular_smoothness_mse": float(
                        chart_reg_terms["chart_angular_smoothness_mse"].detach().cpu()
                    ),
                    "chart_radial_smoothness_mse": float(chart_reg_terms["chart_radial_smoothness_mse"].detach().cpu()),
                    "chart_monotonic_mse": float(chart_reg_terms["chart_monotonic_mse"].detach().cpu()),
                    "coeff_l2": float(coeff_l2.detach().cpu()),
                    "elapsed_sec": time.time() - wall_start,
                }
            )

    full_final = evaluate_holdout_task(model, test_x, test_y, args=args, seed=seed + 3000)
    final_train_weights = train_policy_pairwise_weights(
        args.steps - 1,
        pairwise_visible=pairwise_visible,
        pairwise_train_weights=pairwise_train_weights,
        pairwise_train_weights_start=pairwise_train_weights_start,
        learned_chart_logits=learned_chart_logits,
        continuation_logits=continuation_logits,
        pairwise_radial_bins=pairwise_radial_bins,
        pairwise_angular_bins=pairwise_angular_bins,
        args=args,
    )
    visible_final = evaluate_holdout_task(
        model,
        test_x,
        test_y,
        args=args,
        seed=seed + 3000,
        bias_overrides=train_component_bias_overrides(
            model,
            pairwise_visible=pairwise_visible,
            pairwise_train_weights=final_train_weights,
            args=args,
        ),
    )
    zero_final = evaluate_holdout_task(
        model,
        test_x,
        test_y,
        args=args,
        seed=seed + 3000,
        bias_overrides=zero_bias_overrides(model),
    )
    eval_rows: list[dict[str, object]] = []

    def add_eval_row(mode: str, metric: dict[str, float], *, param: str = "") -> None:
        eval_rows.append(
            {
                "dataset": args.dataset,
                "task": args.task,
                "basis": basis.name,
                "seed": seed,
                "holdout_radius": args.holdout_radius,
                "train_radial_visible_radius": train_radial_visible_radius_value(args),
                "train_soft_radial_visible_radius": train_soft_radial_visible_radius_value(args),
                "train_soft_radial_width": args.train_soft_radial_width,
                "train_soft_radial_floor": args.train_soft_radial_floor,
                "train_soft_radial_anneal_start_step": args.train_soft_radial_anneal_start_step,
                "train_soft_radial_anneal_ramp_steps": args.train_soft_radial_anneal_ramp_steps,
                "train_learned_chart_policy": args.train_learned_chart_policy,
                "train_learned_chart_init": args.train_learned_chart_init,
                "train_learned_chart_radius": args.train_learned_chart_radius,
                "train_learned_chart_width": args.train_learned_chart_width,
                "train_learned_chart_floor": args.train_learned_chart_floor,
                "train_learned_chart_angular_bins": args.train_learned_chart_angular_bins,
                "train_learned_chart_lr_mult": args.train_learned_chart_lr_mult,
                "train_learned_chart_weight_decay": args.train_learned_chart_weight_decay,
                "train_learned_chart_angular_smoothness_weight": args.train_learned_chart_angular_smoothness_weight,
                "train_learned_chart_radial_smoothness_weight": args.train_learned_chart_radial_smoothness_weight,
                "train_learned_chart_monotonic_weight": args.train_learned_chart_monotonic_weight,
                **continuation_config_fields(args),
                **v9_regularizer_config_fields(args),
                "train_component_mode": args.train_component_mode,
                "train_component_spec": args.train_component_spec,
                "train_component_radius": args.train_component_radius,
                "heldout_bias_mse_weight": args.heldout_bias_mse_weight,
                "boundary_smoothness_weight": args.boundary_smoothness_weight,
                "coeff_l2_weight": args.coeff_l2_weight,
                "radial_tail_weight": args.radial_tail_weight,
                "radial_tail_gamma": active_radial_tail_gamma(args),
                "reg_schedule": args.reg_schedule,
                "reg_start_step": args.reg_start_step,
                "reg_ramp_steps": args.reg_ramp_steps,
                "eval_mode": mode,
                "eval_param": param,
                "score": metric["score"],
                "loss": metric["loss"],
            }
        )

    add_eval_row("full", full_final)
    add_eval_row("visible_only", visible_final)
    add_eval_row("zero_bias", zero_final)
    eval_modes = set(parse_list(args.eval_control_modes))
    if "heldout_clamp" in eval_modes:
        clamp_final = evaluate_holdout_task(
            model,
            test_x,
            test_y,
            args=args,
            seed=seed + 3000,
            bias_overrides=eval_control_bias_overrides(
                model,
                mode="heldout_clamp",
                holdout_radius=args.holdout_radius,
            ),
        )
        add_eval_row("heldout_clamp", clamp_final)
    if "radial_decay" in eval_modes:
        for gamma in parse_float_list(args.eval_radial_decay_gammas):
            decay_final = evaluate_holdout_task(
                model,
                test_x,
                test_y,
                args=args,
                seed=seed + 3000,
                bias_overrides=eval_control_bias_overrides(
                    model,
                    mode="radial_decay",
                    holdout_radius=args.holdout_radius,
                    radial_decay_gamma=gamma,
                ),
            )
            add_eval_row("radial_decay", decay_final, param=f"gamma={gamma:g}")
    if "radial_truncate" in eval_modes:
        for radius in parse_float_list(args.eval_radial_truncate_radii):
            truncate_final = evaluate_holdout_task(
                model,
                test_x,
                test_y,
                args=args,
                seed=seed + 3000,
                bias_overrides=eval_control_bias_overrides(
                    model,
                    mode="radial_truncate",
                    holdout_radius=args.holdout_radius,
                    radial_keep_max=radius,
                ),
            )
            add_eval_row("radial_truncate", truncate_final, param=f"r<={radius:g}")
    if "radial_band" in eval_modes:
        for lo, hi in parse_float_ranges(args.eval_radial_band_ranges):
            band_final = evaluate_holdout_task(
                model,
                test_x,
                test_y,
                args=args,
                seed=seed + 3000,
                bias_overrides=eval_control_bias_overrides(
                    model,
                    mode="radial_band",
                    holdout_radius=args.holdout_radius,
                    radial_keep_min=lo,
                    radial_keep_max=hi,
                ),
            )
            add_eval_row("radial_band", band_final, param=f"{lo:g}<=r<{hi:g}")
    for mode in ("layer_radial_ablate", "layer_radial_keep"):
        if mode in eval_modes:
            for radius in parse_float_list(args.eval_layer_radial_radii):
                for layer in range(len(model.blocks)):
                    metric = evaluate_holdout_task(
                        model,
                        test_x,
                        test_y,
                        args=args,
                        seed=seed + 3000,
                        bias_overrides=component_radial_bias_overrides(
                            model,
                            mode=mode,
                            radius=radius,
                            layer=layer,
                        ),
                    )
                    add_eval_row(mode, metric, param=f"layer={layer},r<={radius:g}")
    for mode in ("head_radial_ablate", "head_radial_keep"):
        if mode in eval_modes:
            for radius in parse_float_list(args.eval_head_radial_radii):
                for layer, block in enumerate(model.blocks):
                    for head in range(block.n_heads):
                        metric = evaluate_holdout_task(
                            model,
                            test_x,
                            test_y,
                            args=args,
                            seed=seed + 3000,
                            bias_overrides=component_radial_bias_overrides(
                                model,
                                mode=mode,
                                radius=radius,
                                layer=layer,
                                head=head,
                            ),
                        )
                        add_eval_row(mode, metric, param=f"layer={layer},head={head},r<={radius:g}")
    if best_step < 0:
        best_score = full_final["score"]
        best_loss = full_final["loss"]
        best_visible_score = visible_final["score"]
        best_step = args.steps - 1
        if args.export_checkpoint:
            best_state_dict = state_dict_to_cpu(model)
    stats = model.bias_stats()
    energy = bias_energy_stats(model, pairwise_visible)
    final_reg_terms = bias_regularization_terms(
        model,
        pairwise_visible=pairwise_visible,
        table_visible=table_visible,
        holdout_radius=args.holdout_radius,
        radial_tail_gamma=args.radial_tail_gamma,
        teacher_tables=teacher_tables,
        boundary_band_width=args.boundary_band_width,
        far_spectral_cutoff=args.far_spectral_cutoff,
        far_band_teacher_range=args.far_band_teacher_range,
    )
    final_chart_reg_terms = learned_chart_regularization_terms(learned_chart_logits, args)
    row: dict[str, object] = {
        "dataset": args.dataset,
        "task": args.task,
        "basis": basis.name,
        "seed": seed,
        "metric": metric_name,
        "score_mode": args.score_mode,
        "score": best_score if args.score_mode == "best" else full_final["score"],
        "loss": best_loss if args.score_mode == "best" else full_final["loss"],
        "best_score": best_score,
        "best_loss": best_loss,
        "best_step": best_step,
        "best_visible_only_score": best_visible_score,
        "final_score": full_final["score"],
        "final_loss": full_final["loss"],
        "final_visible_only_score": visible_final["score"],
        "final_visible_only_loss": visible_final["loss"],
        "final_zero_bias_score": zero_final["score"],
        "final_zero_bias_loss": zero_final["loss"],
        "final_extrapolation_gain": full_final["score"] - visible_final["score"],
        "num_features": basis.matrix.shape[1],
        "param_count": sum(param.numel() for param in model.parameters()),
        "wall_sec": time.time() - wall_start,
        "holdout_radius": args.holdout_radius,
        "train_radial_visible_radius": train_radial_visible_radius_value(args),
        "train_soft_radial_visible_radius": train_soft_radial_visible_radius_value(args),
        "train_soft_radial_width": args.train_soft_radial_width,
        "train_soft_radial_floor": args.train_soft_radial_floor,
        "train_soft_radial_anneal_start_step": args.train_soft_radial_anneal_start_step,
        "train_soft_radial_anneal_ramp_steps": args.train_soft_radial_anneal_ramp_steps,
        "train_learned_chart_policy": args.train_learned_chart_policy,
        "train_learned_chart_init": args.train_learned_chart_init,
        "train_learned_chart_radius": args.train_learned_chart_radius,
        "train_learned_chart_width": args.train_learned_chart_width,
        "train_learned_chart_floor": args.train_learned_chart_floor,
        "train_learned_chart_angular_bins": args.train_learned_chart_angular_bins,
        "train_learned_chart_lr_mult": args.train_learned_chart_lr_mult,
        "train_learned_chart_weight_decay": args.train_learned_chart_weight_decay,
        "train_learned_chart_angular_smoothness_weight": args.train_learned_chart_angular_smoothness_weight,
        "train_learned_chart_radial_smoothness_weight": args.train_learned_chart_radial_smoothness_weight,
        "train_learned_chart_monotonic_weight": args.train_learned_chart_monotonic_weight,
        **continuation_config_fields(args),
        **v9_regularizer_config_fields(args),
        "train_component_mode": args.train_component_mode,
        "train_component_spec": args.train_component_spec,
        "train_component_radius": args.train_component_radius,
        "heldout_bias_mse_weight": args.heldout_bias_mse_weight,
        "boundary_smoothness_weight": args.boundary_smoothness_weight,
        "coeff_l2_weight": args.coeff_l2_weight,
        "radial_tail_weight": args.radial_tail_weight,
        "radial_tail_gamma": active_radial_tail_gamma(args),
        "reg_schedule": args.reg_schedule,
        "reg_start_step": args.reg_start_step,
        "reg_ramp_steps": args.reg_ramp_steps,
        "final_heldout_bias_mse": float(final_reg_terms["heldout_bias_mse"].detach().cpu()),
        "final_boundary_smoothness_mse": float(final_reg_terms["boundary_smoothness_mse"].detach().cpu()),
        "final_radial_tail_mse": float(final_reg_terms["radial_tail_mse"].detach().cpu()),
        "final_boundary_jump_mse": float(final_reg_terms["boundary_value_match_mse"].detach().cpu()),
        "final_boundary_derivative_jump_mse": float(final_reg_terms["boundary_derivative_match_mse"].detach().cpu()),
        "final_far_laplacian_energy": float(final_reg_terms["far_laplacian_energy"].detach().cpu()),
        "final_far_mixed_hessian_energy": float(final_reg_terms["far_mixed_hessian_energy"].detach().cpu()),
        "final_far_spectral_energy": float(final_reg_terms["far_spectral_energy"].detach().cpu()),
        "final_radial_tail_tv": float(final_reg_terms["radial_tail_tv"].detach().cpu()),
        "final_radial_tail_monotonic_mse": float(final_reg_terms["radial_tail_monotonic_mse"].detach().cpu()),
        "final_local_identity_mse": float(final_reg_terms["local_identity_mse"].detach().cpu()),
        "final_far_band_teacher_mse": float(final_reg_terms["far_band_teacher_mse"].detach().cpu()),
        "final_chart_angular_smoothness_mse": float(
            final_chart_reg_terms["chart_angular_smoothness_mse"].detach().cpu()
        ),
        "final_chart_radial_smoothness_mse": float(final_chart_reg_terms["chart_radial_smoothness_mse"].detach().cpu()),
        "final_chart_monotonic_mse": float(final_chart_reg_terms["chart_monotonic_mse"].detach().cpu()),
        **learned_chart_weight_stats(
            learned_chart_logits,
            pairwise_bins=pairwise_radial_bins,
            pairwise_angular_bins=pairwise_angular_bins,
            pairwise_visible=pairwise_visible,
            args=args,
        ),
        **continuation_weight_stats(
            continuation_logits,
            pairwise_bins=pairwise_radial_bins,
            pairwise_visible=pairwise_visible,
            pairwise_train_weights=final_train_weights,
            args=args,
        ),
        **energy,
        **stats,
    }
    artifact_name = (
        f"{args.dataset}_{args.task}_{basis.name}_seed{seed}_holdoutR{args.holdout_radius}"
        f"_steps{args.steps}_{export_config_slug(args)}"
    )
    artifact_metadata = {
        "dataset": args.dataset,
        "task": args.task,
        "basis": basis.name,
        "seed": seed,
        "steps": args.steps,
        "best_step": best_step,
        "score": row["score"],
        "final_score": row["final_score"],
        "grid_side": int(math.sqrt(float(model.n_positions))),
        "n_layers": len(model.blocks),
        "n_heads": args.n_heads,
        "num_features": int(model.basis_matrix.shape[1]),
        "holdout_radius": args.holdout_radius,
        "train_radial_visible_radius": train_radial_visible_radius_value(args),
        "train_soft_radial_visible_radius": train_soft_radial_visible_radius_value(args),
        "train_soft_radial_width": args.train_soft_radial_width,
        "train_soft_radial_floor": args.train_soft_radial_floor,
        "train_soft_radial_anneal_start_step": args.train_soft_radial_anneal_start_step,
        "train_soft_radial_anneal_ramp_steps": args.train_soft_radial_anneal_ramp_steps,
        "train_learned_chart_policy": args.train_learned_chart_policy,
        "train_learned_chart_init": args.train_learned_chart_init,
        "train_learned_chart_radius": args.train_learned_chart_radius,
        "train_learned_chart_width": args.train_learned_chart_width,
        "train_learned_chart_floor": args.train_learned_chart_floor,
        "train_learned_chart_angular_bins": args.train_learned_chart_angular_bins,
        "train_learned_chart_lr_mult": args.train_learned_chart_lr_mult,
        "train_learned_chart_weight_decay": args.train_learned_chart_weight_decay,
        "train_learned_chart_angular_smoothness_weight": args.train_learned_chart_angular_smoothness_weight,
        "train_learned_chart_radial_smoothness_weight": args.train_learned_chart_radial_smoothness_weight,
        "train_learned_chart_monotonic_weight": args.train_learned_chart_monotonic_weight,
        **continuation_config_fields(args),
        **v9_regularizer_config_fields(args),
        "train_component_mode": args.train_component_mode,
        "train_component_spec": args.train_component_spec,
        "train_component_radius": args.train_component_radius,
        "heldout_bias_mse_weight": args.heldout_bias_mse_weight,
        "boundary_smoothness_weight": args.boundary_smoothness_weight,
        "coeff_l2_weight": args.coeff_l2_weight,
        "radial_tail_weight": args.radial_tail_weight,
        "radial_tail_gamma": active_radial_tail_gamma(args),
        "reg_schedule": args.reg_schedule,
        "reg_start_step": args.reg_start_step,
        "reg_ramp_steps": args.reg_ramp_steps,
    }
    if args.export_checkpoint:
        checkpoint_dir = Path(args.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{artifact_name}.pt"
        torch.save(
            {
                "metadata": artifact_metadata,
                "args": vars(args),
                "row": row,
                "final_model_state_dict": state_dict_to_cpu(model),
                "best_model_state_dict": best_state_dict,
                "basis_matrix": model.basis_matrix.detach().float().cpu(),
                "basis_name": basis.name,
                "basis_orders": list(basis.orders),
            },
            checkpoint_path,
        )
        row["checkpoint"] = str(checkpoint_path)
    if args.export_bias:
        coeff = torch.stack([block.coeff.detach().float().cpu() for block in model.blocks], dim=0)
        basis_matrix = model.basis_matrix.detach().float().cpu()
        export_dir = Path(args.output_dir) / "bias_exports" / artifact_name
        artifacts = write_geometry_artifacts(
            export_dir,
            basis_matrix=basis_matrix,
            coeff=coeff,
            side=int(math.sqrt(float(model.n_positions))),
            metadata=artifact_metadata,
        )
        row.update({f"bias_{key}": value for key, value in artifacts.items()})
    return row, curves, eval_rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["dataset"]),
                str(row.get("task", "reconstruction")),
                str(row["basis"]),
                int(float(row.get("holdout_radius", 0))),
                row_float_key(row, "train_radial_visible_radius"),
                row_float_key(row, "train_soft_radial_visible_radius"),
                row_float_key(row, "train_soft_radial_width", 0.0),
                row_float_key(row, "train_soft_radial_floor", 0.0),
                int(float(row.get("train_soft_radial_anneal_start_step", 0))),
                int(float(row.get("train_soft_radial_anneal_ramp_steps", 0))),
                str(row.get("train_learned_chart_policy", "none")),
                str(row.get("train_learned_chart_init", "soft")),
                row_float_key(row, "train_learned_chart_radius", 4.0),
                row_float_key(row, "train_learned_chart_width", 1.0),
                row_float_key(row, "train_learned_chart_floor", 0.0),
                int(float(row.get("train_learned_chart_angular_bins", 1))),
                row_float_key(row, "train_learned_chart_lr_mult", 1.0),
                row_float_key(row, "train_learned_chart_weight_decay", 0.0),
                row_float_key(row, "train_learned_chart_angular_smoothness_weight", 0.0),
                row_float_key(row, "train_learned_chart_radial_smoothness_weight", 0.0),
                row_float_key(row, "train_learned_chart_monotonic_weight", 0.0),
                str(row.get("train_continuation_policy", "none")),
                row_float_key(row, "train_continuation_radius", 4.0),
                row_float_key(row, "train_continuation_width", 1.0),
                row_float_key(row, "train_continuation_floor", 0.0),
                row_float_key(row, "train_continuation_lr_mult", 1.0),
                row_float_key(row, "train_continuation_weight_decay", 0.0),
                str(row.get("train_component_mode", "full")),
                str(row.get("train_component_spec", "")),
                row_float_key(row, "train_component_radius", 0.0),
                float(row.get("heldout_bias_mse_weight", 0.0)),
                float(row.get("boundary_smoothness_weight", 0.0)),
                float(row.get("coeff_l2_weight", 0.0)),
                float(row.get("radial_tail_weight", 0.0)),
                float(row.get("radial_tail_gamma", 0.0)),
                float(row.get("boundary_value_match_weight", 0.0)),
                float(row.get("boundary_derivative_match_weight", 0.0)),
                float(row.get("far_laplacian_weight", 0.0)),
                float(row.get("far_mixed_hessian_weight", 0.0)),
                float(row.get("far_spectral_energy_weight", 0.0)),
                float(row.get("radial_tail_tv_weight", 0.0)),
                float(row.get("local_identity_weight", 0.0)),
                float(row.get("far_band_teacher_mse_weight", 0.0)),
                str(row.get("far_band_teacher_range", "6-9")),
                float(row.get("far_spectral_cutoff", 0.25)),
                float(row.get("boundary_band_width", 1.0)),
                str(row.get("reg_schedule", "constant")),
                int(float(row.get("reg_start_step", 0))),
                int(float(row.get("reg_ramp_steps", 0))),
                float(row.get("visible_teacher_init_ridge", 0.0)),
            ),
            [],
        ).append(row)
    out: list[dict[str, object]] = []
    for (
        dataset,
        task,
        basis,
        holdout_radius,
        train_radial_visible_radius,
        train_soft_radial_visible_radius,
        train_soft_radial_width,
        train_soft_radial_floor,
        train_soft_radial_anneal_start_step,
        train_soft_radial_anneal_ramp_steps,
        train_learned_chart_policy,
        train_learned_chart_init,
        train_learned_chart_radius,
        train_learned_chart_width,
        train_learned_chart_floor,
        train_learned_chart_angular_bins,
        train_learned_chart_lr_mult,
        train_learned_chart_weight_decay,
        train_learned_chart_angular_smoothness_weight,
        train_learned_chart_radial_smoothness_weight,
        train_learned_chart_monotonic_weight,
        train_continuation_policy,
        train_continuation_radius,
        train_continuation_width,
        train_continuation_floor,
        train_continuation_lr_mult,
        train_continuation_weight_decay,
        train_component_mode,
        train_component_spec,
        train_component_radius,
        heldout_weight,
        boundary_weight,
        coeff_l2_weight,
        radial_tail_weight,
        radial_tail_gamma,
        boundary_value_match_weight,
        boundary_derivative_match_weight,
        far_laplacian_weight,
        far_mixed_hessian_weight,
        far_spectral_energy_weight,
        radial_tail_tv_weight,
        local_identity_weight,
        far_band_teacher_mse_weight,
        far_band_teacher_range,
        far_spectral_cutoff,
        boundary_band_width,
        reg_schedule,
        reg_start_step,
        reg_ramp_steps,
        init_ridge,
    ), values in sorted(groups.items()):
        score = np.array([float(row["score"]) for row in values])
        final = np.array([float(row["final_score"]) for row in values])
        visible = np.array([float(row["final_visible_only_score"]) for row in values])
        gain = np.array([float(row["final_extrapolation_gain"]) for row in values])
        out.append(
            {
                "dataset": dataset,
                "task": task,
                "basis": basis,
                "n": len(values),
                "holdout_radius": holdout_radius,
                "train_radial_visible_radius": train_radial_visible_radius,
                "train_soft_radial_visible_radius": train_soft_radial_visible_radius,
                "train_soft_radial_width": train_soft_radial_width,
                "train_soft_radial_floor": train_soft_radial_floor,
                "train_soft_radial_anneal_start_step": train_soft_radial_anneal_start_step,
                "train_soft_radial_anneal_ramp_steps": train_soft_radial_anneal_ramp_steps,
                "train_learned_chart_policy": train_learned_chart_policy,
                "train_learned_chart_init": train_learned_chart_init,
                "train_learned_chart_radius": train_learned_chart_radius,
                "train_learned_chart_width": train_learned_chart_width,
                "train_learned_chart_floor": train_learned_chart_floor,
                "train_learned_chart_angular_bins": train_learned_chart_angular_bins,
                "train_learned_chart_lr_mult": train_learned_chart_lr_mult,
                "train_learned_chart_weight_decay": train_learned_chart_weight_decay,
                "train_learned_chart_angular_smoothness_weight": train_learned_chart_angular_smoothness_weight,
                "train_learned_chart_radial_smoothness_weight": train_learned_chart_radial_smoothness_weight,
                "train_learned_chart_monotonic_weight": train_learned_chart_monotonic_weight,
                "train_continuation_policy": train_continuation_policy,
                "train_continuation_radius": train_continuation_radius,
                "train_continuation_width": train_continuation_width,
                "train_continuation_floor": train_continuation_floor,
                "train_continuation_lr_mult": train_continuation_lr_mult,
                "train_continuation_weight_decay": train_continuation_weight_decay,
                "train_component_mode": train_component_mode,
                "train_component_spec": train_component_spec,
                "train_component_radius": train_component_radius,
                "score_mean": float(score.mean()),
                "score_std": float(score.std()),
                "final_score_mean": float(final.mean()),
                "final_visible_only_mean": float(visible.mean()),
                "final_extrapolation_gain_mean": float(gain.mean()),
                "heldout_bias_rms_mean": float(np.mean([float(row["heldout_bias_rms"]) for row in values])),
                "visible_bias_rms_mean": float(np.mean([float(row["visible_bias_rms"]) for row in values])),
                "final_heldout_bias_mse_mean": float(np.mean([float(row.get("final_heldout_bias_mse", 0.0)) for row in values])),
                "final_boundary_smoothness_mse_mean": float(
                    np.mean([float(row.get("final_boundary_smoothness_mse", 0.0)) for row in values])
                ),
                "final_radial_tail_mse_mean": float(np.mean([float(row.get("final_radial_tail_mse", 0.0)) for row in values])),
                "final_boundary_jump_mse_mean": float(
                    np.mean([float(row.get("final_boundary_jump_mse", 0.0)) for row in values])
                ),
                "final_boundary_derivative_jump_mse_mean": float(
                    np.mean([float(row.get("final_boundary_derivative_jump_mse", 0.0)) for row in values])
                ),
                "final_far_laplacian_energy_mean": float(
                    np.mean([float(row.get("final_far_laplacian_energy", 0.0)) for row in values])
                ),
                "final_far_mixed_hessian_energy_mean": float(
                    np.mean([float(row.get("final_far_mixed_hessian_energy", 0.0)) for row in values])
                ),
                "final_far_spectral_energy_mean": float(
                    np.mean([float(row.get("final_far_spectral_energy", 0.0)) for row in values])
                ),
                "final_radial_tail_tv_mean": float(
                    np.mean([float(row.get("final_radial_tail_tv", 0.0)) for row in values])
                ),
                "final_local_identity_mse_mean": float(
                    np.mean([float(row.get("final_local_identity_mse", 0.0)) for row in values])
                ),
                "final_far_band_teacher_mse_mean": float(
                    np.mean([float(row.get("final_far_band_teacher_mse", 0.0)) for row in values])
                ),
                "final_chart_angular_smoothness_mse_mean": float(
                    np.mean([float(row.get("final_chart_angular_smoothness_mse", 0.0)) for row in values])
                ),
                "final_chart_radial_smoothness_mse_mean": float(
                    np.mean([float(row.get("final_chart_radial_smoothness_mse", 0.0)) for row in values])
                ),
                "final_chart_monotonic_mse_mean": float(
                    np.mean([float(row.get("final_chart_monotonic_mse", 0.0)) for row in values])
                ),
                "learned_chart_weight_mean": mean_optional_float(values, "learned_chart_weight_mean"),
                "learned_chart_weight_r_le_4_mean": mean_optional_float(values, "learned_chart_weight_r_le_4_mean"),
                "learned_chart_weight_r_4_5_mean": mean_optional_float(values, "learned_chart_weight_r_4_5_mean"),
                "learned_chart_weight_r_5_6_mean": mean_optional_float(values, "learned_chart_weight_r_5_6_mean"),
                "learned_chart_weight_angular_std_mean": mean_optional_float(values, "learned_chart_weight_angular_std_mean"),
                "learned_chart_weight_angular_std_r_4_6_mean": mean_optional_float(
                    values, "learned_chart_weight_angular_std_r_4_6_mean"
                ),
                "continuation_weight_mean": mean_optional_float(values, "continuation_weight_mean"),
                "continuation_weight_r_le_4_mean": mean_optional_float(values, "continuation_weight_r_le_4_mean"),
                "continuation_weight_r_4_5_mean": mean_optional_float(values, "continuation_weight_r_4_5_mean"),
                "continuation_weight_r_5_6_mean": mean_optional_float(values, "continuation_weight_r_5_6_mean"),
                "continuation_weight_r_gt_6_mean": mean_optional_float(values, "continuation_weight_r_gt_6_mean"),
                "heldout_bias_mse_weight": heldout_weight,
                "boundary_smoothness_weight": boundary_weight,
                "coeff_l2_weight": coeff_l2_weight,
                "radial_tail_weight": radial_tail_weight,
                "radial_tail_gamma": radial_tail_gamma,
                "boundary_value_match_weight": boundary_value_match_weight,
                "boundary_derivative_match_weight": boundary_derivative_match_weight,
                "far_laplacian_weight": far_laplacian_weight,
                "far_mixed_hessian_weight": far_mixed_hessian_weight,
                "far_spectral_energy_weight": far_spectral_energy_weight,
                "radial_tail_tv_weight": radial_tail_tv_weight,
                "local_identity_weight": local_identity_weight,
                "far_band_teacher_mse_weight": far_band_teacher_mse_weight,
                "far_band_teacher_range": far_band_teacher_range,
                "far_spectral_cutoff": far_spectral_cutoff,
                "boundary_band_width": boundary_band_width,
                "reg_schedule": reg_schedule,
                "reg_start_step": reg_start_step,
                "reg_ramp_steps": reg_ramp_steps,
                "visible_teacher_init_ridge": init_ridge,
                "num_features": int(values[0]["num_features"]),
            }
        )
    return out


def aggregate_eval_controls(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["dataset"]),
                str(row.get("task", "reconstruction")),
                str(row["basis"]),
                int(float(row.get("holdout_radius", 0))),
                row_float_key(row, "train_radial_visible_radius"),
                row_float_key(row, "train_soft_radial_visible_radius"),
                row_float_key(row, "train_soft_radial_width", 0.0),
                row_float_key(row, "train_soft_radial_floor", 0.0),
                int(float(row.get("train_soft_radial_anneal_start_step", 0))),
                int(float(row.get("train_soft_radial_anneal_ramp_steps", 0))),
                str(row.get("train_learned_chart_policy", "none")),
                str(row.get("train_learned_chart_init", "soft")),
                row_float_key(row, "train_learned_chart_radius", 4.0),
                row_float_key(row, "train_learned_chart_width", 1.0),
                row_float_key(row, "train_learned_chart_floor", 0.0),
                int(float(row.get("train_learned_chart_angular_bins", 1))),
                row_float_key(row, "train_learned_chart_lr_mult", 1.0),
                row_float_key(row, "train_learned_chart_weight_decay", 0.0),
                row_float_key(row, "train_learned_chart_angular_smoothness_weight", 0.0),
                row_float_key(row, "train_learned_chart_radial_smoothness_weight", 0.0),
                row_float_key(row, "train_learned_chart_monotonic_weight", 0.0),
                str(row.get("train_continuation_policy", "none")),
                row_float_key(row, "train_continuation_radius", 4.0),
                row_float_key(row, "train_continuation_width", 1.0),
                row_float_key(row, "train_continuation_floor", 0.0),
                row_float_key(row, "train_continuation_lr_mult", 1.0),
                row_float_key(row, "train_continuation_weight_decay", 0.0),
                str(row.get("train_component_mode", "full")),
                str(row.get("train_component_spec", "")),
                row_float_key(row, "train_component_radius", 0.0),
                float(row.get("heldout_bias_mse_weight", 0.0)),
                float(row.get("boundary_smoothness_weight", 0.0)),
                float(row.get("coeff_l2_weight", 0.0)),
                float(row.get("radial_tail_weight", 0.0)),
                float(row.get("radial_tail_gamma", 0.0)),
                float(row.get("boundary_value_match_weight", 0.0)),
                float(row.get("boundary_derivative_match_weight", 0.0)),
                float(row.get("far_laplacian_weight", 0.0)),
                float(row.get("far_mixed_hessian_weight", 0.0)),
                float(row.get("far_spectral_energy_weight", 0.0)),
                float(row.get("radial_tail_tv_weight", 0.0)),
                float(row.get("local_identity_weight", 0.0)),
                float(row.get("far_band_teacher_mse_weight", 0.0)),
                str(row.get("far_band_teacher_range", "6-9")),
                float(row.get("far_spectral_cutoff", 0.25)),
                float(row.get("boundary_band_width", 1.0)),
                str(row.get("reg_schedule", "constant")),
                int(float(row.get("reg_start_step", 0))),
                int(float(row.get("reg_ramp_steps", 0))),
                str(row.get("eval_mode", "")),
                str(row.get("eval_param", "")),
            ),
            [],
        ).append(row)
    out: list[dict[str, object]] = []
    for (
        dataset,
        task,
        basis,
        holdout_radius,
        train_radial_visible_radius,
        train_soft_radial_visible_radius,
        train_soft_radial_width,
        train_soft_radial_floor,
        train_soft_radial_anneal_start_step,
        train_soft_radial_anneal_ramp_steps,
        train_learned_chart_policy,
        train_learned_chart_init,
        train_learned_chart_radius,
        train_learned_chart_width,
        train_learned_chart_floor,
        train_learned_chart_angular_bins,
        train_learned_chart_lr_mult,
        train_learned_chart_weight_decay,
        train_learned_chart_angular_smoothness_weight,
        train_learned_chart_radial_smoothness_weight,
        train_learned_chart_monotonic_weight,
        train_continuation_policy,
        train_continuation_radius,
        train_continuation_width,
        train_continuation_floor,
        train_continuation_lr_mult,
        train_continuation_weight_decay,
        train_component_mode,
        train_component_spec,
        train_component_radius,
        heldout_weight,
        boundary_weight,
        coeff_l2_weight,
        radial_tail_weight,
        radial_tail_gamma,
        boundary_value_match_weight,
        boundary_derivative_match_weight,
        far_laplacian_weight,
        far_mixed_hessian_weight,
        far_spectral_energy_weight,
        radial_tail_tv_weight,
        local_identity_weight,
        far_band_teacher_mse_weight,
        far_band_teacher_range,
        far_spectral_cutoff,
        boundary_band_width,
        reg_schedule,
        reg_start_step,
        reg_ramp_steps,
        eval_mode,
        eval_param,
    ), values in sorted(groups.items()):
        score = np.array([float(row["score"]) for row in values])
        loss = np.array([float(row["loss"]) for row in values])
        out.append(
            {
                "dataset": dataset,
                "task": task,
                "basis": basis,
                "holdout_radius": holdout_radius,
                "train_radial_visible_radius": train_radial_visible_radius,
                "train_soft_radial_visible_radius": train_soft_radial_visible_radius,
                "train_soft_radial_width": train_soft_radial_width,
                "train_soft_radial_floor": train_soft_radial_floor,
                "train_soft_radial_anneal_start_step": train_soft_radial_anneal_start_step,
                "train_soft_radial_anneal_ramp_steps": train_soft_radial_anneal_ramp_steps,
                "train_learned_chart_policy": train_learned_chart_policy,
                "train_learned_chart_init": train_learned_chart_init,
                "train_learned_chart_radius": train_learned_chart_radius,
                "train_learned_chart_width": train_learned_chart_width,
                "train_learned_chart_floor": train_learned_chart_floor,
                "train_learned_chart_angular_bins": train_learned_chart_angular_bins,
                "train_learned_chart_lr_mult": train_learned_chart_lr_mult,
                "train_learned_chart_weight_decay": train_learned_chart_weight_decay,
                "train_learned_chart_angular_smoothness_weight": train_learned_chart_angular_smoothness_weight,
                "train_learned_chart_radial_smoothness_weight": train_learned_chart_radial_smoothness_weight,
                "train_learned_chart_monotonic_weight": train_learned_chart_monotonic_weight,
                "train_continuation_policy": train_continuation_policy,
                "train_continuation_radius": train_continuation_radius,
                "train_continuation_width": train_continuation_width,
                "train_continuation_floor": train_continuation_floor,
                "train_continuation_lr_mult": train_continuation_lr_mult,
                "train_continuation_weight_decay": train_continuation_weight_decay,
                "train_component_mode": train_component_mode,
                "train_component_spec": train_component_spec,
                "train_component_radius": train_component_radius,
                "heldout_bias_mse_weight": heldout_weight,
                "boundary_smoothness_weight": boundary_weight,
                "coeff_l2_weight": coeff_l2_weight,
                "radial_tail_weight": radial_tail_weight,
                "radial_tail_gamma": radial_tail_gamma,
                "boundary_value_match_weight": boundary_value_match_weight,
                "boundary_derivative_match_weight": boundary_derivative_match_weight,
                "far_laplacian_weight": far_laplacian_weight,
                "far_mixed_hessian_weight": far_mixed_hessian_weight,
                "far_spectral_energy_weight": far_spectral_energy_weight,
                "radial_tail_tv_weight": radial_tail_tv_weight,
                "local_identity_weight": local_identity_weight,
                "far_band_teacher_mse_weight": far_band_teacher_mse_weight,
                "far_band_teacher_range": far_band_teacher_range,
                "far_spectral_cutoff": far_spectral_cutoff,
                "boundary_band_width": boundary_band_width,
                "reg_schedule": reg_schedule,
                "reg_start_step": reg_start_step,
                "reg_ramp_steps": reg_ramp_steps,
                "eval_mode": eval_mode,
                "eval_param": eval_param,
                "n": len(values),
                "score_mean": float(score.mean()),
                "score_std": float(score.std()),
                "loss_mean": float(loss.mean()),
                "loss_std": float(loss.std()),
            }
        )
    return out


def write_report(output_dir: Path, summary: dict[str, object], aggregate_rows: list[dict[str, object]]) -> None:
    lines = [
        "# V4-E Offset Holdout Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        f"- Dataset: {summary['dataset']}",
        f"- Task: {summary.get('task', 'reconstruction')}",
        f"- Teacher bias: `{summary['teacher_bias']}`",
        f"- Teacher-init ridge: {summary['teacher_init_ridge']}",
        f"- Heldout bias MSE weight: {summary['heldout_bias_mse_weight']}",
        f"- Boundary smoothness weight: {summary['boundary_smoothness_weight']}",
        f"- Coeff L2 weight: {summary.get('coeff_l2_weight', 0.0)}",
        f"- Radial tail weight/gamma: {summary.get('radial_tail_weight', 0.0)} / {summary.get('radial_tail_gamma', 0.0)}",
        f"- Regularization schedule: {summary['reg_schedule']}",
        f"- Regularization start/ramp: {summary['reg_start_step']} / {summary['reg_ramp_steps']}",
        f"- Eval control modes: {summary.get('eval_control_modes', '')}",
        f"- Eval radial decay gammas: {summary.get('eval_radial_decay_gammas', '')}",
        f"- Eval radial truncate radii: {summary.get('eval_radial_truncate_radii', '')}",
        f"- Eval radial band ranges: {summary.get('eval_radial_band_ranges', '')}",
        f"- Eval layer/head radial radii: {summary.get('eval_layer_radial_radii', '')} / {summary.get('eval_head_radial_radii', '')}",
        f"- Export checkpoint: {summary.get('export_checkpoint', False)}",
        f"- Holdout radius: {summary['holdout_radius']}",
        f"- Train radial visible radius: {summary.get('train_radial_visible_radius', -1.0)}",
        (
            "- Train soft radial radius/width/floor: "
            f"{summary.get('train_soft_radial_visible_radius', -1.0)} / "
            f"{summary.get('train_soft_radial_width', 1.0)} / "
            f"{summary.get('train_soft_radial_floor', 0.0)}"
        ),
        (
            "- Train soft radial anneal start/ramp: "
            f"{summary.get('train_soft_radial_anneal_start_step', 0)} / "
            f"{summary.get('train_soft_radial_anneal_ramp_steps', 0)}"
        ),
        (
            "- Train learned chart: "
            f"{summary.get('train_learned_chart_policy', 'none')} "
            f"init={summary.get('train_learned_chart_init', 'soft')} "
            f"r={summary.get('train_learned_chart_radius', 4.0)} "
            f"w={summary.get('train_learned_chart_width', 1.0)} "
            f"floor={summary.get('train_learned_chart_floor', 0.0)} "
            f"angular_bins={summary.get('train_learned_chart_angular_bins', 1)} "
            f"lr_mult={summary.get('train_learned_chart_lr_mult', 1.0)} "
            f"ang_smooth={summary.get('train_learned_chart_angular_smoothness_weight', 0.0)} "
            f"rad_smooth={summary.get('train_learned_chart_radial_smoothness_weight', 0.0)} "
            f"mono={summary.get('train_learned_chart_monotonic_weight', 0.0)}"
        ),
        (
            "- Train continuation: "
            f"{summary.get('train_continuation_policy', 'none')} "
            f"r={summary.get('train_continuation_radius', 4.0)} "
            f"w={summary.get('train_continuation_width', 1.0)} "
            f"floor={summary.get('train_continuation_floor', 0.0)} "
            f"lr_mult={summary.get('train_continuation_lr_mult', 1.0)}"
        ),
        (
            "- V9 regularizers: "
            f"boundary_value={summary.get('boundary_value_match_weight', 0.0)} "
            f"boundary_derivative={summary.get('boundary_derivative_match_weight', 0.0)} "
            f"far_laplacian={summary.get('far_laplacian_weight', 0.0)} "
            f"far_mixed_hessian={summary.get('far_mixed_hessian_weight', 0.0)} "
            f"far_spectral={summary.get('far_spectral_energy_weight', 0.0)} "
            f"tail_tv={summary.get('radial_tail_tv_weight', 0.0)} "
            f"local_identity={summary.get('local_identity_weight', 0.0)} "
            f"far_band_teacher={summary.get('far_band_teacher_mse_weight', 0.0)} "
            f"far_band_range={summary.get('far_band_teacher_range', '6-9')}"
        ),
        (
            "- Train component: "
            f"{summary.get('train_component_mode', 'full')} "
            f"{summary.get('train_component_spec', '')} "
            f"r={summary.get('train_component_radius', -1.0)}"
        ),
        f"- Basis/train visible pair fractions: {summary.get('basis_visible_pair_fraction', '')} / {summary.get('visible_pair_fraction', '')}",
        f"- Steps: {summary['steps']}",
        f"- Seeds: {summary['seeds']}",
        "",
        "## Aggregate",
        "",
        "| task | basis | n | features | R | train r | soft r | soft w | soft floor | anneal start | anneal ramp | learned chart | chart r | chart w | chart a | component | comp r | h-wt | b-wt | coeff-l2 | tail-wt | tail-gamma | reg | start | ramp | score mean | score std | final full | final visible-only | full-visible gain | heldout bias rms | chart r<=4 | chart 4<r<=5 | chart ang std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + f"{row.get('task', 'reconstruction')} | "
            + f"{row['basis']} | {int(row['n'])} | {int(row['num_features'])} | "
            + f"{int(row['holdout_radius'])} | "
            + f"{float(row.get('train_radial_visible_radius', -1.0)):.4g} | "
            + f"{float(row.get('train_soft_radial_visible_radius', -1.0)):.4g} | "
            + f"{float(row.get('train_soft_radial_width', 1.0)):.4g} | "
            + f"{float(row.get('train_soft_radial_floor', 0.0)):.4g} | "
            + f"{int(float(row.get('train_soft_radial_anneal_start_step', 0)))} | "
            + f"{int(float(row.get('train_soft_radial_anneal_ramp_steps', 0)))} | "
            + f"{row.get('train_learned_chart_policy', 'none')}:{row.get('train_learned_chart_init', '')} | "
            + f"{float(row.get('train_learned_chart_radius', 4.0)):.4g} | "
            + f"{float(row.get('train_learned_chart_width', 1.0)):.4g} | "
            + f"{int(float(row.get('train_learned_chart_angular_bins', 1)))} | "
            + f"{row.get('train_component_mode', 'full')}:{row.get('train_component_spec', '')} | "
            + f"{float(row.get('train_component_radius', -1.0)):.4g} | "
            + f"{float(row['heldout_bias_mse_weight']):.4g} | "
            + f"{float(row['boundary_smoothness_weight']):.4g} | "
            + f"{float(row.get('coeff_l2_weight', 0.0)):.4g} | "
            + f"{float(row.get('radial_tail_weight', 0.0)):.4g} | "
            + f"{float(row.get('radial_tail_gamma', 0.0)):.4g} | "
            + f"{row['reg_schedule']} | "
            + f"{int(row['reg_start_step'])} | "
            + f"{int(row['reg_ramp_steps'])} | "
            + f"{float(row['score_mean']):.4f} | {float(row['score_std']):.4f} | "
            + f"{float(row['final_score_mean']):.4f} | "
            + f"{float(row['final_visible_only_mean']):.4f} | "
            + f"{float(row['final_extrapolation_gain_mean']):.4f} | "
            + f"{float(row['heldout_bias_rms_mean']):.4f} | "
            + f"{float(row.get('learned_chart_weight_r_le_4_mean', float('nan'))):.4f} | "
            + f"{float(row.get('learned_chart_weight_r_4_5_mean', float('nan'))):.4f} | "
            + f"{float(row.get('learned_chart_weight_angular_std_mean', float('nan'))):.4f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Training keeps the attention graph unchanged but masks positional-bias contributions for held-out offsets.",
            "- `final full` evaluates the learned compact function on all offsets.",
            "- `final visible-only` keeps held-out offset bias at zero during eval; in component mode it also applies the train-time component constraint.",
            "- Positive full-visible gain means the learned functional bias uses extrapolated held-out offsets beneficially.",
            "- Extra final-time controls, when enabled, are written to `offset_holdout_eval_controls.csv`.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.train_soft_radial_anneal_ramp_steps > 0 and args.train_soft_radial_visible_radius is None:
        raise ValueError("--train-soft-radial-anneal-ramp-steps requires --train-soft-radial-visible-radius")
    if args.train_learned_chart_policy != "none" and args.train_radial_visible_radius is not None:
        raise ValueError("--train-learned-chart-policy is intended for train/eval policy and should not be combined with hard train-radial")
    if args.train_continuation_policy != "none":
        if args.train_learned_chart_policy != "none":
            raise ValueError("--train-continuation-policy and --train-learned-chart-policy are separate policy paths")
        if args.train_radial_visible_radius is not None or args.train_soft_radial_visible_radius is not None:
            raise ValueError("--train-continuation-policy should not be combined with train-radial or train-soft-radial")
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
    train_x, test_x = normalize_patches(train_x, test_x)
    teacher_side = int(teacher_metadata.get("grid_side", (teacher_tables_cpu.shape[-1] + 1) // 2))
    if teacher_side != grid_side:
        raise ValueError(f"teacher grid side {teacher_side} != data grid side {grid_side}")
    basis_pairwise_visible, basis_table_visible = offset_masks(grid_side, args.holdout_radius, device=device)
    pairwise_visible = basis_pairwise_visible
    table_visible = basis_table_visible
    if args.train_radial_visible_radius is not None:
        radial_pairwise_visible, radial_table_visible = radial_offset_masks(
            grid_side,
            args.train_radial_visible_radius,
            device=device,
        )
        pairwise_visible = pairwise_visible & radial_pairwise_visible
        table_visible = table_visible & radial_table_visible
    pairwise_radial_bins = None
    pairwise_angular_bins = None
    if args.train_learned_chart_policy != "none" or args.train_continuation_policy != "none":
        pairwise_radial_bins, _, _ = radial_offset_bin_indices(grid_side, device=device)
        if args.train_learned_chart_policy == "radial_angular_bins":
            pairwise_angular_bins, _ = angular_offset_bin_indices(
                grid_side,
                args.train_learned_chart_angular_bins,
                device=device,
            )
    pairwise_train_weights = None
    pairwise_train_weights_start = None
    if args.train_soft_radial_visible_radius is not None:
        soft_pairwise, _ = soft_radial_offset_weights(
            grid_side,
            args.train_soft_radial_visible_radius,
            width=args.train_soft_radial_width,
            floor=args.train_soft_radial_floor,
            device=device,
        )
        pairwise_train_weights = pairwise_visible.to(device=device, dtype=torch.float32) * soft_pairwise
        if args.train_soft_radial_anneal_ramp_steps > 0:
            hard_pairwise, _ = radial_offset_masks(
                grid_side,
                args.train_soft_radial_visible_radius,
                device=device,
            )
            pairwise_train_weights_start = pairwise_visible.to(device=device, dtype=torch.float32) * hard_pairwise.to(
                device=device,
                dtype=torch.float32,
            )
    if args.train_continuation_policy == "low_curvature_tail":
        hard_pairwise, _ = radial_offset_masks(
            grid_side,
            args.train_continuation_radius,
            device=device,
        )
        pairwise_train_weights = pairwise_visible.to(device=device, dtype=torch.float32) * hard_pairwise.to(
            device=device,
            dtype=torch.float32,
        )
    elif args.train_continuation_policy == "boundary_matching":
        soft_pairwise, _ = soft_radial_offset_weights(
            grid_side,
            args.train_continuation_radius,
            width=args.train_continuation_width,
            floor=args.train_continuation_floor,
            device=device,
        )
        pairwise_train_weights = pairwise_visible.to(device=device, dtype=torch.float32) * soft_pairwise
    visible_teacher_cpu = mask_teacher_tables(teacher_tables_cpu, basis_table_visible.to(device=teacher_tables_cpu.device))

    variants = parse_list(args.student_bases)
    bases = build_teacher_bases(
        side=grid_side,
        device=device,
        teacher_tables=visible_teacher_cpu.to(device=device),
        variants=variants,
        include_shuffle=False,
        seed=args.seed,
    )
    rows: list[dict[str, object]] = read_csv(output_dir / "offset_holdout_results.csv") if args.resume else []
    curves: list[dict[str, object]] = read_csv(output_dir / "offset_holdout_curves.csv") if args.resume else []
    eval_control_rows: list[dict[str, object]] = read_csv(output_dir / "offset_holdout_eval_controls.csv") if args.resume else []
    init_rows: list[dict[str, object]] = read_csv(output_dir / "visible_teacher_init_fits.csv") if args.resume else []
    completed = {
        (
            str(row.get("dataset")),
            str(row.get("task", "reconstruction")),
            str(row.get("basis")),
            int(float(row.get("seed", -1))),
            int(float(row.get("holdout_radius", args.holdout_radius))),
            row_float_key(row, "train_radial_visible_radius"),
            row_float_key(row, "train_soft_radial_visible_radius"),
            row_float_key(row, "train_soft_radial_width", 0.0),
            row_float_key(row, "train_soft_radial_floor", 0.0),
            int(float(row.get("train_soft_radial_anneal_start_step", 0))),
            int(float(row.get("train_soft_radial_anneal_ramp_steps", 0))),
            str(row.get("train_learned_chart_policy", "none")),
            str(row.get("train_learned_chart_init", "soft")),
            row_float_key(row, "train_learned_chart_radius", 4.0),
            row_float_key(row, "train_learned_chart_width", 1.0),
            row_float_key(row, "train_learned_chart_floor", 0.0),
            int(float(row.get("train_learned_chart_angular_bins", 1))),
            row_float_key(row, "train_learned_chart_lr_mult", 1.0),
            row_float_key(row, "train_learned_chart_weight_decay", 0.0),
            row_float_key(row, "train_learned_chart_angular_smoothness_weight", 0.0),
            row_float_key(row, "train_learned_chart_radial_smoothness_weight", 0.0),
            row_float_key(row, "train_learned_chart_monotonic_weight", 0.0),
            str(row.get("train_continuation_policy", "none")),
            row_float_key(row, "train_continuation_radius", 4.0),
            row_float_key(row, "train_continuation_width", 1.0),
            row_float_key(row, "train_continuation_floor", 0.0),
            row_float_key(row, "train_continuation_lr_mult", 1.0),
            row_float_key(row, "train_continuation_weight_decay", 0.0),
            str(row.get("train_component_mode", "full")),
            str(row.get("train_component_spec", "")),
            row_float_key(row, "train_component_radius", 0.0),
            float(row.get("heldout_bias_mse_weight", 0.0)),
            float(row.get("boundary_smoothness_weight", 0.0)),
            float(row.get("coeff_l2_weight", 0.0)),
            float(row.get("radial_tail_weight", 0.0)),
            float(row.get("radial_tail_gamma", 0.0)),
            float(row.get("boundary_value_match_weight", 0.0)),
            float(row.get("boundary_derivative_match_weight", 0.0)),
            float(row.get("far_laplacian_weight", 0.0)),
            float(row.get("far_mixed_hessian_weight", 0.0)),
            float(row.get("far_spectral_energy_weight", 0.0)),
            float(row.get("radial_tail_tv_weight", 0.0)),
            float(row.get("local_identity_weight", 0.0)),
            float(row.get("far_band_teacher_mse_weight", 0.0)),
            str(row.get("far_band_teacher_range", "6-9")),
            float(row.get("far_spectral_cutoff", 0.25)),
            float(row.get("boundary_band_width", 1.0)),
            str(row.get("reg_schedule", "constant")),
            int(float(row.get("reg_start_step", 0))),
            int(float(row.get("reg_ramp_steps", 0))),
            float(row.get("visible_teacher_init_ridge", args.teacher_init_ridge)),
        )
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
            init_coeff, basis_init_rows = teacher_init_coeff_visible(
                basis,
                teacher_tables_cpu.to(device=device),
                pairwise_visible=pairwise_visible,
                side=grid_side,
                depth=args.depth,
                n_heads=args.n_heads,
                ridge=args.teacher_init_ridge,
            )
            for init_row in basis_init_rows:
                init_row["train_radial_visible_radius"] = train_radial_visible_radius_value(args)
                init_row["train_soft_radial_visible_radius"] = train_soft_radial_visible_radius_value(args)
                init_row["train_soft_radial_width"] = args.train_soft_radial_width
                init_row["train_soft_radial_floor"] = args.train_soft_radial_floor
                init_row["train_soft_radial_anneal_start_step"] = args.train_soft_radial_anneal_start_step
                init_row["train_soft_radial_anneal_ramp_steps"] = args.train_soft_radial_anneal_ramp_steps
                init_row["train_learned_chart_policy"] = args.train_learned_chart_policy
                init_row["train_learned_chart_init"] = args.train_learned_chart_init
                init_row["train_learned_chart_radius"] = args.train_learned_chart_radius
                init_row["train_learned_chart_width"] = args.train_learned_chart_width
                init_row["train_learned_chart_floor"] = args.train_learned_chart_floor
                init_row["train_learned_chart_angular_bins"] = args.train_learned_chart_angular_bins
                init_row["train_learned_chart_lr_mult"] = args.train_learned_chart_lr_mult
                init_row["train_learned_chart_weight_decay"] = args.train_learned_chart_weight_decay
                init_row["train_learned_chart_angular_smoothness_weight"] = args.train_learned_chart_angular_smoothness_weight
                init_row["train_learned_chart_radial_smoothness_weight"] = args.train_learned_chart_radial_smoothness_weight
                init_row["train_learned_chart_monotonic_weight"] = args.train_learned_chart_monotonic_weight
                init_row.update(continuation_config_fields(args))
                init_row.update(v9_regularizer_config_fields(args))
            if not args.resume or not any(
                str(row.get("basis")) == basis.name
                and float(row.get("visible_teacher_init_ridge", args.teacher_init_ridge)) == float(args.teacher_init_ridge)
                and row_float_key(row, "train_radial_visible_radius") == train_radial_visible_radius_value(args)
                and row_float_key(row, "train_soft_radial_visible_radius") == train_soft_radial_visible_radius_value(args)
                and row_float_key(row, "train_soft_radial_width", 0.0) == float(args.train_soft_radial_width)
                and row_float_key(row, "train_soft_radial_floor", 0.0) == float(args.train_soft_radial_floor)
                for row in init_rows
            ):
                init_rows.extend(basis_init_rows)
                write_csv(output_dir / "visible_teacher_init_fits.csv", init_rows)
        init_r2_mean = float(np.mean([float(row["visible_teacher_init_r2"]) for row in basis_init_rows])) if basis_init_rows else float("nan")
        for seed_idx in range(args.seed_start_idx, args.seeds):
            basis_seed_idx = basis_seed_indices.get(basis.name, basis_idx)
            seed = args.seed + 100 * seed_idx + 17 * basis_seed_idx
            run_key = (
                args.dataset,
                args.task,
                basis.name,
                seed,
                args.holdout_radius,
                train_radial_visible_radius_value(args),
                train_soft_radial_visible_radius_value(args),
                args.train_soft_radial_width,
                args.train_soft_radial_floor,
                args.train_soft_radial_anneal_start_step,
                args.train_soft_radial_anneal_ramp_steps,
                args.train_learned_chart_policy,
                args.train_learned_chart_init,
                args.train_learned_chart_radius,
                args.train_learned_chart_width,
                args.train_learned_chart_floor,
                args.train_learned_chart_angular_bins,
                args.train_learned_chart_lr_mult,
                args.train_learned_chart_weight_decay,
                args.train_learned_chart_angular_smoothness_weight,
                args.train_learned_chart_radial_smoothness_weight,
                args.train_learned_chart_monotonic_weight,
                args.train_continuation_policy,
                args.train_continuation_radius,
                args.train_continuation_width,
                args.train_continuation_floor,
                args.train_continuation_lr_mult,
                args.train_continuation_weight_decay,
                args.train_component_mode,
                args.train_component_spec,
                args.train_component_radius,
                args.heldout_bias_mse_weight,
                args.boundary_smoothness_weight,
                args.coeff_l2_weight,
                args.radial_tail_weight,
                active_radial_tail_gamma(args),
                args.boundary_value_match_weight,
                args.boundary_derivative_match_weight,
                args.far_laplacian_weight,
                args.far_mixed_hessian_weight,
                args.far_spectral_energy_weight,
                args.radial_tail_tv_weight,
                args.local_identity_weight,
                args.far_band_teacher_mse_weight,
                args.far_band_teacher_range,
                args.far_spectral_cutoff,
                args.boundary_band_width,
                args.reg_schedule,
                args.reg_start_step,
                args.reg_ramp_steps,
                args.teacher_init_ridge,
            )
            if run_key in completed:
                continue
            if args.max_runs is not None and runs_done >= args.max_runs:
                break
            row, task_curves, task_eval_rows = train_holdout_task(
                basis,
                train_x=train_x,
                train_y=train_y,
                test_x=test_x,
                test_y=test_y,
                patch_dim=patch_dim,
                pairwise_visible=pairwise_visible,
                pairwise_train_weights=pairwise_train_weights,
                pairwise_train_weights_start=pairwise_train_weights_start,
                pairwise_radial_bins=pairwise_radial_bins,
                pairwise_angular_bins=pairwise_angular_bins,
                table_visible=table_visible,
                args=args,
                seed=seed,
                init_coeff=init_coeff,
                teacher_tables=teacher_tables_cpu.to(device=device),
            )
            row.update(
                {
                    "teacher_bias": args.teacher_bias,
                    "teacher_basis": teacher_metadata.get("basis", "unknown"),
                    "teacher_seed": teacher_metadata.get("seed", -1),
                    "visible_teacher_init_r2_mean": init_r2_mean,
                    "visible_teacher_init_ridge": args.teacher_init_ridge,
                }
            )
            for eval_row in task_eval_rows:
                eval_row.update(
                    {
                        "teacher_bias": args.teacher_bias,
                        "teacher_basis": teacher_metadata.get("basis", "unknown"),
                        "teacher_seed": teacher_metadata.get("seed", -1),
                        "visible_teacher_init_r2_mean": init_r2_mean,
                        "visible_teacher_init_ridge": args.teacher_init_ridge,
                    }
                )
            rows.append(row)
            curves.extend(task_curves)
            eval_control_rows.extend(task_eval_rows)
            completed.add(run_key)
            runs_done += 1
            aggregate_rows = aggregate(rows)
            eval_control_aggregate_rows = aggregate_eval_controls(eval_control_rows)
            write_csv(output_dir / "offset_holdout_results.csv", rows)
            write_csv(output_dir / "offset_holdout_aggregate.csv", aggregate_rows)
            write_csv(output_dir / "offset_holdout_curves.csv", curves)
            write_csv(output_dir / "offset_holdout_eval_controls.csv", eval_control_rows)
            write_csv(output_dir / "offset_holdout_eval_controls_aggregate.csv", eval_control_aggregate_rows)
            partial = {
                "device": str(device),
                "dataset": args.dataset,
                "task": args.task,
                "teacher_bias": args.teacher_bias,
                "holdout_radius": args.holdout_radius,
                "train_radial_visible_radius": train_radial_visible_radius_value(args),
                "train_soft_radial_visible_radius": train_soft_radial_visible_radius_value(args),
                "train_soft_radial_width": args.train_soft_radial_width,
                "train_soft_radial_floor": args.train_soft_radial_floor,
                "train_soft_radial_anneal_start_step": args.train_soft_radial_anneal_start_step,
                "train_soft_radial_anneal_ramp_steps": args.train_soft_radial_anneal_ramp_steps,
                "train_learned_chart_policy": args.train_learned_chart_policy,
                "train_learned_chart_init": args.train_learned_chart_init,
                "train_learned_chart_radius": args.train_learned_chart_radius,
                "train_learned_chart_width": args.train_learned_chart_width,
                "train_learned_chart_floor": args.train_learned_chart_floor,
                "train_learned_chart_angular_bins": args.train_learned_chart_angular_bins,
                "train_learned_chart_lr_mult": args.train_learned_chart_lr_mult,
                "train_learned_chart_weight_decay": args.train_learned_chart_weight_decay,
                "train_learned_chart_angular_smoothness_weight": args.train_learned_chart_angular_smoothness_weight,
                "train_learned_chart_radial_smoothness_weight": args.train_learned_chart_radial_smoothness_weight,
                "train_learned_chart_monotonic_weight": args.train_learned_chart_monotonic_weight,
                **continuation_config_fields(args),
                **v9_regularizer_config_fields(args),
                "train_component_mode": args.train_component_mode,
                "train_component_spec": args.train_component_spec,
                "train_component_radius": args.train_component_radius,
                "heldout_bias_mse_weight": args.heldout_bias_mse_weight,
                "boundary_smoothness_weight": args.boundary_smoothness_weight,
                "coeff_l2_weight": args.coeff_l2_weight,
                "radial_tail_weight": args.radial_tail_weight,
                "radial_tail_gamma": active_radial_tail_gamma(args),
                "reg_schedule": args.reg_schedule,
                "reg_start_step": args.reg_start_step,
                "reg_ramp_steps": args.reg_ramp_steps,
                "eval_control_modes": args.eval_control_modes,
                "eval_radial_decay_gammas": args.eval_radial_decay_gammas,
                "eval_radial_truncate_radii": args.eval_radial_truncate_radii,
                "eval_radial_band_ranges": args.eval_radial_band_ranges,
                "eval_layer_radial_radii": args.eval_layer_radial_radii,
                "eval_head_radial_radii": args.eval_head_radial_radii,
                "basis_visible_pair_fraction": float(basis_pairwise_visible.float().mean().detach().cpu()),
                "visible_pair_fraction": float(pairwise_visible.float().mean().detach().cpu()),
                "steps": args.steps,
                "seeds": args.seeds,
                "bases": [basis.name for basis in bases],
                "runs_done_this_invocation": runs_done,
                "wall_sec": time.time() - wall_start,
                "status": "partial",
            }
            (output_dir / "offset_holdout_summary.partial.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
        if args.max_runs is not None and runs_done >= args.max_runs:
            break

    aggregate_rows = aggregate(rows)
    eval_control_aggregate_rows = aggregate_eval_controls(eval_control_rows)
    write_csv(output_dir / "offset_holdout_results.csv", rows)
    write_csv(output_dir / "offset_holdout_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "offset_holdout_curves.csv", curves)
    write_csv(output_dir / "offset_holdout_eval_controls.csv", eval_control_rows)
    write_csv(output_dir / "offset_holdout_eval_controls_aggregate.csv", eval_control_aggregate_rows)
    write_csv(output_dir / "visible_teacher_init_fits.csv", init_rows)
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() and device.type == "cuda" else 0
    summary = {
        "device": str(device),
        "dataset": args.dataset,
        "task": args.task,
        "teacher_bias": args.teacher_bias,
        "teacher_metadata": teacher_metadata,
        "teacher_init": args.teacher_init,
        "teacher_init_ridge": args.teacher_init_ridge,
        "patch_size": args.patch_size,
        "grid_side": grid_side,
        "patch_dim": patch_dim,
        "depth": args.depth,
        "dim": args.dim,
        "n_heads": args.n_heads,
        "steps": args.steps,
        "seeds": args.seeds,
        "score_mode": args.score_mode,
        "holdout_radius": args.holdout_radius,
        "train_radial_visible_radius": train_radial_visible_radius_value(args),
        "train_soft_radial_visible_radius": train_soft_radial_visible_radius_value(args),
        "train_soft_radial_width": args.train_soft_radial_width,
        "train_soft_radial_floor": args.train_soft_radial_floor,
        "train_soft_radial_anneal_start_step": args.train_soft_radial_anneal_start_step,
        "train_soft_radial_anneal_ramp_steps": args.train_soft_radial_anneal_ramp_steps,
        "train_learned_chart_policy": args.train_learned_chart_policy,
        "train_learned_chart_init": args.train_learned_chart_init,
        "train_learned_chart_radius": args.train_learned_chart_radius,
        "train_learned_chart_width": args.train_learned_chart_width,
        "train_learned_chart_floor": args.train_learned_chart_floor,
        "train_learned_chart_angular_bins": args.train_learned_chart_angular_bins,
        "train_learned_chart_lr_mult": args.train_learned_chart_lr_mult,
        "train_learned_chart_weight_decay": args.train_learned_chart_weight_decay,
        "train_learned_chart_angular_smoothness_weight": args.train_learned_chart_angular_smoothness_weight,
        "train_learned_chart_radial_smoothness_weight": args.train_learned_chart_radial_smoothness_weight,
        "train_learned_chart_monotonic_weight": args.train_learned_chart_monotonic_weight,
        **continuation_config_fields(args),
        **v9_regularizer_config_fields(args),
        "train_component_mode": args.train_component_mode,
        "train_component_spec": args.train_component_spec,
        "train_component_radius": args.train_component_radius,
        "heldout_bias_mse_weight": args.heldout_bias_mse_weight,
        "boundary_smoothness_weight": args.boundary_smoothness_weight,
        "coeff_l2_weight": args.coeff_l2_weight,
        "radial_tail_weight": args.radial_tail_weight,
        "radial_tail_gamma": active_radial_tail_gamma(args),
        "reg_schedule": args.reg_schedule,
        "reg_start_step": args.reg_start_step,
        "reg_ramp_steps": args.reg_ramp_steps,
        "eval_control_modes": args.eval_control_modes,
        "eval_radial_decay_gammas": args.eval_radial_decay_gammas,
        "eval_radial_truncate_radii": args.eval_radial_truncate_radii,
        "eval_radial_band_ranges": args.eval_radial_band_ranges,
        "eval_layer_radial_radii": args.eval_layer_radial_radii,
        "eval_head_radial_radii": args.eval_head_radial_radii,
        "export_checkpoint": args.export_checkpoint,
        "basis_visible_pair_fraction": float(basis_pairwise_visible.float().mean().detach().cpu()),
        "visible_pair_fraction": float(pairwise_visible.float().mean().detach().cpu()),
        "train_count": int(train_x.shape[0]),
        "test_count": int(test_x.shape[0]),
        "wall_sec": time.time() - wall_start,
        "peak_cuda_memory_bytes": peak_mem,
        "bases": [basis.name for basis in bases],
        "runs_done_this_invocation": runs_done,
    }
    summary_path = output_dir / "offset_holdout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, summary, aggregate_rows)
    summary["summary"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4-E bias-parameter offset holdout.")
    parser.add_argument("--teacher-bias", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--task", choices=["reconstruction", "classification"], default="reconstruction")
    parser.add_argument("--teacher-init", action="store_true")
    parser.add_argument("--teacher-init-ridge", type=float, default=1e-4)
    parser.add_argument(
        "--student-bases",
        type=str,
        default="relative_2d_table,dct_top33,table_informed_toric_PJ_R0_top110,axis_plus_toric_residual_R0_top55",
    )
    parser.add_argument("--dataset", choices=VISION_DATASETS, default="cifar10")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--holdout-radius", type=int, default=4)
    parser.add_argument(
        "--train-radial-visible-radius",
        type=float,
        default=None,
        help="Optional Euclidean radius intersected into the train/eval visible positional-bias mask.",
    )
    parser.add_argument(
        "--train-soft-radial-visible-radius",
        type=float,
        default=None,
        help="Optional Euclidean radius for a soft train/eval positional-bias mask.",
    )
    parser.add_argument(
        "--train-soft-radial-width",
        type=float,
        default=1.0,
        help="Exponential falloff width for --train-soft-radial-visible-radius.",
    )
    parser.add_argument(
        "--train-soft-radial-floor",
        type=float,
        default=0.0,
        help="Minimum mask weight for --train-soft-radial-visible-radius.",
    )
    parser.add_argument(
        "--train-soft-radial-anneal-start-step",
        type=int,
        default=0,
        help="Start step for annealing from a hard radial mask to the soft radial mask.",
    )
    parser.add_argument(
        "--train-soft-radial-anneal-ramp-steps",
        type=int,
        default=0,
        help="Number of steps used to anneal from a hard radial mask to the soft radial mask. Zero disables annealing.",
    )
    parser.add_argument(
        "--train-learned-chart-policy",
        choices=["none", "radial_bins", "radial_angular_bins"],
        default="none",
        help="Optional learnable train/eval chart policy for positional-bias weights.",
    )
    parser.add_argument(
        "--train-learned-chart-init",
        choices=["soft", "hard", "ones"],
        default="soft",
        help="Initialization for --train-learned-chart-policy.",
    )
    parser.add_argument("--train-learned-chart-radius", type=float, default=4.0)
    parser.add_argument("--train-learned-chart-width", type=float, default=1.0)
    parser.add_argument("--train-learned-chart-floor", type=float, default=0.0)
    parser.add_argument(
        "--train-learned-chart-angular-bins",
        type=int,
        default=8,
        help="Number of angular sectors for radial_angular_bins learned chart policy.",
    )
    parser.add_argument(
        "--train-learned-chart-lr-mult",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for learned chart policy parameters.",
    )
    parser.add_argument(
        "--train-learned-chart-weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay for learned chart policy parameters.",
    )
    parser.add_argument(
        "--train-learned-chart-angular-smoothness-weight",
        type=float,
        default=0.0,
        help="Regularization weight for cyclic angular smoothness of learned chart weights.",
    )
    parser.add_argument(
        "--train-learned-chart-radial-smoothness-weight",
        type=float,
        default=0.0,
        help="Regularization weight for adjacent radial-shell smoothness of learned chart weights.",
    )
    parser.add_argument(
        "--train-learned-chart-monotonic-weight",
        type=float,
        default=0.0,
        help="Regularization weight penalizing learned chart weights that increase with radius.",
    )
    parser.add_argument(
        "--train-continuation-policy",
        choices=["none", "learned_radial_coordinate", "low_curvature_tail", "boundary_matching", "local_continuation"],
        default="none",
        help="V9 local-continuation train/eval policy for scalar-bias offset extrapolation.",
    )
    parser.add_argument("--train-continuation-radius", type=float, default=4.0)
    parser.add_argument("--train-continuation-width", type=float, default=1.0)
    parser.add_argument("--train-continuation-floor", type=float, default=0.0)
    parser.add_argument(
        "--train-continuation-lr-mult",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for learned V9 continuation weights.",
    )
    parser.add_argument(
        "--train-continuation-weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay for learned V9 continuation weights.",
    )
    parser.add_argument(
        "--train-component-mode",
        choices=["full", "keep", "ablate"],
        default="full",
        help="Optional train-time component control applied to positional-bias overrides.",
    )
    parser.add_argument(
        "--train-component-spec",
        type=str,
        default="",
        help="Component list such as 0:3+6+1 or L0:H3+H6+H1 for train-time keep/ablate.",
    )
    parser.add_argument(
        "--train-component-radius",
        type=float,
        default=4.0,
        help="Euclidean offset radius used by train-time component keep/ablate.",
    )
    parser.add_argument("--heldout-bias-mse-weight", type=float, default=0.0)
    parser.add_argument("--boundary-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--coeff-l2-weight", type=float, default=0.0)
    parser.add_argument("--radial-tail-weight", type=float, default=0.0)
    parser.add_argument("--radial-tail-gamma", type=float, default=2.0)
    parser.add_argument("--boundary-value-match-weight", type=float, default=0.0)
    parser.add_argument("--boundary-derivative-match-weight", type=float, default=0.0)
    parser.add_argument("--far-laplacian-weight", type=float, default=0.0)
    parser.add_argument("--far-mixed-hessian-weight", type=float, default=0.0)
    parser.add_argument("--far-spectral-energy-weight", type=float, default=0.0)
    parser.add_argument("--radial-tail-tv-weight", type=float, default=0.0)
    parser.add_argument("--local-identity-weight", type=float, default=0.0)
    parser.add_argument(
        "--far-band-teacher-mse-weight",
        type=float,
        default=0.0,
        help="Teacher-table MSE weight on selected far radial bands, e.g. 6-9.",
    )
    parser.add_argument(
        "--far-band-teacher-range",
        type=str,
        default="6-9",
        help="Comma-separated Euclidean radius ranges for far-band teacher MSE, e.g. 5-6,6-9.",
    )
    parser.add_argument(
        "--far-spectral-cutoff",
        type=float,
        default=0.25,
        help="Normalized frequency cutoff for V9 far-shell spectral energy.",
    )
    parser.add_argument(
        "--boundary-band-width",
        type=float,
        default=1.0,
        help="Euclidean shell half-width used by V9 boundary derivative diagnostics.",
    )
    parser.add_argument("--reg-schedule", choices=["constant", "linear", "cosine"], default="constant")
    parser.add_argument("--reg-start-step", type=int, default=0)
    parser.add_argument("--reg-ramp-steps", type=int, default=0)
    parser.add_argument(
        "--eval-control-modes",
        type=str,
        default="",
        help=(
            "Comma-separated final-time controls: heldout_clamp,radial_decay,radial_truncate,radial_band,"
            "layer_radial_ablate,layer_radial_keep,head_radial_ablate,head_radial_keep."
        ),
    )
    parser.add_argument("--eval-radial-decay-gammas", type=str, default="0.5,1,2")
    parser.add_argument("--eval-radial-truncate-radii", type=str, default="2,3,4,5,6")
    parser.add_argument("--eval-radial-band-ranges", type=str, default="0-2,2-4,4-6,6-9")
    parser.add_argument("--eval-layer-radial-radii", type=str, default="2,4")
    parser.add_argument("--eval-head-radial-radii", type=str, default="2")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--mask-rate", type=float, default=0.35)
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
    parser.add_argument("--seed-start-idx", type=int, default=0)
    parser.add_argument("--seed-basis-order", type=str, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-mode", choices=["final", "best"], default="best")
    parser.add_argument("--export-bias", action="store_true")
    parser.add_argument("--export-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=426)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v4_offset_holdout_cifar10_10k")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
