from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision import datasets, transforms

from toric_pj.diagnostics.basis_projection import (
    Basis,
    axis_additive_fourier_basis,
    default_device,
    directional_jet_basis,
    normalize_columns,
    toric_fourier_basis,
)
from toric_pj.diagnostics.direction_alignment import normalize_direction
from toric_pj.experiments.real_digits_probe import make_positions, pairwise_d, raster_1d_basis, relative_2d_table_basis
from toric_pj.experiments.v3_digits_transformer_scaling import (
    PRUNED_REAL_DIGITS_GROUPS,
    RelPosBlock,
    autocast_context,
    label_group,
    prune_basis,
    write_csv,
)

VISION_DATASETS = [
    "mnist",
    "rotated-mnist",
    "affine-mnist",
    "fashion-mnist",
    "cifar10",
    "cifar100",
    "svhn",
    "stl10",
    "tiny-imagenet",
]
VISION_NUM_CLASSES = {
    "mnist": 10,
    "rotated-mnist": 10,
    "affine-mnist": 10,
    "fashion-mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "svhn": 10,
    "stl10": 10,
    "tiny-imagenet": 200,
}


def vision_num_classes(dataset: str) -> int:
    name = dataset.lower()
    if name not in VISION_NUM_CLASSES:
        raise ValueError(f"unknown dataset: {dataset}")
    return VISION_NUM_CLASSES[name]


def apply_mnist_variant(images: torch.Tensor, *, dataset: str, split: str) -> torch.Tensor:
    name = dataset.lower()
    if name not in {"rotated-mnist", "affine-mnist"}:
        return images
    count = int(images.shape[0])
    if count == 0:
        return images
    gen = torch.Generator(device="cpu")
    split_offset = 0 if split == "train" else 100_000
    variant_offset = 0 if name == "rotated-mnist" else 10_000
    gen.manual_seed(20_260_609 + split_offset + variant_offset)
    if name == "rotated-mnist":
        angles = (torch.rand(count, generator=gen) * 2.0 - 1.0) * 30.0
        scale = torch.ones(count)
        translate = torch.zeros(count, 2)
    else:
        angles = (torch.rand(count, generator=gen) * 2.0 - 1.0) * 20.0
        scale = 0.9 + torch.rand(count, generator=gen) * 0.2
        translate = (torch.rand(count, 2, generator=gen) * 2.0 - 1.0) * 3.0

    device = images.device
    dtype = images.dtype
    radians = angles.to(device=device, dtype=dtype) * (math.pi / 180.0)
    scale = scale.to(device=device, dtype=dtype)
    translate = translate.to(device=device, dtype=dtype)
    cos = torch.cos(radians) / scale
    sin = torch.sin(radians) / scale
    theta = torch.zeros(count, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = translate[:, 0] * (2.0 / max(int(images.shape[-1]) - 1, 1))
    theta[:, 1, 2] = translate[:, 1] * (2.0 / max(int(images.shape[-2]) - 1, 1))

    chunks = []
    for start in range(0, count, 4096):
        stop = min(start + 4096, count)
        grid = F.affine_grid(theta[start:stop], images[start:stop].shape, align_corners=False)
        chunks.append(
            F.grid_sample(
                images[start:stop],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    return torch.cat(chunks, dim=0)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, *, split: str, transform: transforms.Compose) -> None:
        self.root = root / "tiny-imagenet-200"
        self.split = split
        self.transform = transform
        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(
                f"missing TinyImageNet data at {self.root}; expected wnids.txt. "
                "Download and extract tiny-imagenet-200.zip under data/."
            )
        wnids = [line.strip() for line in wnids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        self.samples: list[tuple[Path, int]] = []
        if split == "train":
            for wnid in wnids:
                image_dir = self.root / "train" / wnid / "images"
                for image_path in sorted(image_dir.glob("*.JPEG")):
                    self.samples.append((image_path, self.class_to_idx[wnid]))
        elif split == "val":
            annotations = self.root / "val" / "val_annotations.txt"
            if not annotations.exists():
                raise FileNotFoundError(f"missing TinyImageNet val annotations: {annotations}")
            for line in annotations.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                image_name, wnid = parts[0], parts[1]
                self.samples.append((self.root / "val" / "images" / image_name, self.class_to_idx[wnid]))
        else:
            raise ValueError(f"unknown TinyImageNet split: {split}")
        if not self.samples:
            raise ValueError(f"empty TinyImageNet split: {split}")
        self.samples = self._round_robin_by_label(self.samples)

    @staticmethod
    def _round_robin_by_label(samples: list[tuple[Path, int]]) -> list[tuple[Path, int]]:
        by_label: dict[int, list[Path]] = {}
        for path, label in samples:
            by_label.setdefault(label, []).append(path)
        labels = sorted(by_label)
        out: list[tuple[Path, int]] = []
        max_len = max(len(paths) for paths in by_label.values())
        for idx in range(max_len):
            for label in labels:
                paths = by_label[label]
                if idx < len(paths):
                    out.append((paths[idx], label))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            x = self.transform(image.convert("RGB"))
        return x, label


def load_vision_dataset(
    *,
    dataset: str,
    root: Path,
    device: torch.device,
    train_limit: int | None,
    test_limit: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    transform = transforms.ToTensor()
    resize32_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    name = dataset.lower()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if name in {"mnist", "rotated-mnist", "affine-mnist"}:
            train_ds = datasets.MNIST(root=str(root), train=True, download=False, transform=transform)
            test_ds = datasets.MNIST(root=str(root), train=False, download=False, transform=transform)
        elif name == "fashion-mnist":
            train_ds = datasets.FashionMNIST(root=str(root), train=True, download=False, transform=transform)
            test_ds = datasets.FashionMNIST(root=str(root), train=False, download=False, transform=transform)
        elif name == "cifar10":
            train_ds = datasets.CIFAR10(root=str(root), train=True, download=False, transform=transform)
            test_ds = datasets.CIFAR10(root=str(root), train=False, download=False, transform=transform)
        elif name == "cifar100":
            train_ds = datasets.CIFAR100(root=str(root), train=True, download=False, transform=transform)
            test_ds = datasets.CIFAR100(root=str(root), train=False, download=False, transform=transform)
        elif name == "svhn":
            train_ds = datasets.SVHN(root=str(root), split="train", download=False, transform=transform)
            test_ds = datasets.SVHN(root=str(root), split="test", download=False, transform=transform)
        elif name == "stl10":
            train_ds = datasets.STL10(root=str(root), split="train", download=False, transform=resize32_transform)
            test_ds = datasets.STL10(root=str(root), split="test", download=False, transform=resize32_transform)
        elif name == "tiny-imagenet":
            train_ds = TinyImageNetDataset(root, split="train", transform=resize32_transform)
            test_ds = TinyImageNetDataset(root, split="val", transform=resize32_transform)
        else:
            raise ValueError(f"unknown dataset: {dataset}")

    def stack(ds, limit: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        count = len(ds) if limit is None else min(int(limit), len(ds))
        xs = []
        ys = []
        for idx in range(count):
            x, y = ds[idx]
            xs.append(x)
            label = int(y)
            if name == "svhn" and label == 10:
                label = 0
            ys.append(label)
        return torch.stack(xs, dim=0).to(device=device, dtype=torch.float32), torch.tensor(ys, device=device, dtype=torch.long)

    train_x, train_y = stack(train_ds, train_limit)
    test_x, test_y = stack(test_ds, test_limit)
    train_x = apply_mnist_variant(train_x, dataset=name, split="train")
    test_x = apply_mnist_variant(test_x, dataset=name, split="test")
    return train_x, train_y, test_x, test_y


def patchify(images: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int, int]:
    if images.shape[-1] % patch_size != 0 or images.shape[-2] % patch_size != 0:
        raise ValueError("image height/width must be divisible by patch size")
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size).transpose(1, 2)
    grid_h = images.shape[-2] // patch_size
    grid_w = images.shape[-1] // patch_size
    if grid_h != grid_w:
        raise ValueError("current basis builder expects square patch grids")
    return patches, grid_h, patches.shape[-1]


def normalize_patches(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=(0, 1), keepdim=True)
    std = train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (test - mean) / std


def no_pos_basis(d: torch.Tensor) -> Basis:
    return Basis("no_pos_constant", torch.ones((d.shape[0], 1), device=d.device, dtype=d.dtype), ["const"], [0])


def build_patch_bases(side: int, device: torch.device, *, include_shuffle: bool, seed: int) -> list[Basis]:
    positions = make_positions(side, device)
    d = pairwise_d(positions).reshape(-1, 2).to(torch.float32)
    omega_a = torch.tensor([0.78, 0.42], device=device, dtype=torch.float32)
    omega_b = torch.tensor([0.62, -0.58], device=device, dtype=torch.float32)
    omega_c = torch.tensor([0.35, 0.91], device=device, dtype=torch.float32)
    ex = torch.tensor([1.0, 0.0], device=device, dtype=torch.float32)
    ey = torch.tensor([0.0, 1.0], device=device, dtype=torch.float32)
    diag = normalize_direction(torch.tensor([[1.0, 1.0]], device=device, dtype=torch.float32)).reshape(-1)
    oblique = normalize_direction(torch.tensor([[1.0, -0.65]], device=device, dtype=torch.float32)).reshape(-1)
    toric_pj = directional_jet_basis(
        d,
        [omega_a, omega_b, omega_c],
        [ex, ey, diag, oblique],
        [0, 1, 2],
        scale=float(side),
        name="toric_PJ_R2",
    )
    bases = [
        no_pos_basis(d),
        raster_1d_basis(d, side),
        axis_additive_fourier_basis(d, [0.78, 0.58], name="axis_additive"),
        toric_fourier_basis(d, [omega_a, omega_b, omega_c], name="toric_order0"),
        toric_pj,
        prune_basis(toric_pj, PRUNED_REAL_DIGITS_GROUPS, name="pruned_toric_PJ"),
        relative_2d_table_basis(d),
    ]
    if include_shuffle:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        shuffled = d[torch.randperm(d.shape[0], device=device, generator=gen)]
        bases.append(
            directional_jet_basis(
                shuffled,
                [omega_a, omega_b, omega_c],
                [ex, ey, diag, oblique],
                [0, 1, 2],
                scale=float(side),
                name="toric_PJ_R2_coord_shuffle",
            )
        )
    return bases


class VisionRelPosTransformer(nn.Module):
    def __init__(
        self,
        basis: Basis,
        *,
        n_positions: int,
        patch_dim: int,
        dim: int,
        n_heads: int,
        depth: int,
        ffn_mult: int,
        dropout: float,
        n_classes: int,
    ) -> None:
        super().__init__()
        matrix, _ = normalize_columns(basis.matrix.to(dtype=torch.float32))
        self.register_buffer("basis_matrix", matrix)
        self.basis_name = basis.name
        self.orders = list(basis.orders)
        self.n_positions = n_positions
        self.patch_dim = patch_dim
        self.input = nn.Linear(patch_dim + 1, dim)
        self.blocks = nn.ModuleList(
            [
                RelPosBlock(
                    dim=dim,
                    n_heads=n_heads,
                    n_features=matrix.shape[1],
                    n_positions=n_positions,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.reconstruct = nn.Linear(dim, patch_dim)
        self.classifier = nn.Linear(dim, n_classes)

    def encode(
        self,
        patches: torch.Tensor,
        visible: torch.Tensor,
        bias_overrides: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = torch.cat([patches, visible.to(patches.dtype).unsqueeze(-1)], dim=-1)
        x = self.input(x)
        for idx, block in enumerate(self.blocks):
            bias_override = None if bias_overrides is None else bias_overrides[idx]
            x = block(x, self.basis_matrix, bias_override=bias_override)
        return x

    def forward_reconstruction(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor,
        bias_overrides: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        visible_patches = patches.masked_fill(mask.unsqueeze(-1), 0.0)
        encoded = self.encode(visible_patches, ~mask, bias_overrides=bias_overrides)
        return self.reconstruct(encoded)

    def forward_classification(self, patches: torch.Tensor, bias_overrides: list[torch.Tensor] | None = None) -> torch.Tensor:
        visible = torch.ones(patches.shape[:2], device=patches.device, dtype=torch.bool)
        encoded = self.encode(patches, visible, bias_overrides=bias_overrides)
        return self.classifier(encoded.mean(dim=1))

    def bias_stats(self) -> dict[str, float]:
        with torch.no_grad():
            coeff = torch.stack([block.coeff.detach() for block in self.blocks], dim=0)
            flat = coeff.reshape(-1, coeff.shape[-1])
            denom = torch.linalg.norm(flat).clamp_min(1e-12)
            out: dict[str, float] = {"coeff_norm": float(torch.linalg.norm(coeff).detach().cpu())}
            for order in sorted(set(self.orders)):
                idx = torch.tensor([i for i, item in enumerate(self.orders) if item == order], device=coeff.device, dtype=torch.long)
                if idx.numel() > 0:
                    out[f"order{order}_coeff_share"] = float((torch.linalg.norm(flat[:, idx]) / denom).detach().cpu())
            return out


def sample_batch(x: torch.Tensor, y: torch.Tensor, *, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device, generator=generator)
    return x[idx], y[idx]


def evaluate_classifier(
    model: VisionRelPosTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    args: argparse.Namespace,
    bias_overrides: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    model.eval()
    losses = []
    correct = []
    with torch.no_grad():
        for start in range(0, x.shape[0], args.eval_batch_size):
            batch = x[start : start + args.eval_batch_size]
            labels = y[start : start + args.eval_batch_size]
            with autocast_context(batch.device, args.amp):
                logits = model.forward_classification(batch, bias_overrides=bias_overrides)
                loss = F.cross_entropy(logits.float(), labels)
            losses.append(loss)
            correct.append((torch.argmax(logits, dim=-1) == labels).float())
    model.train()
    return {"loss": float(torch.stack(losses).mean().detach().cpu()), "score": float(torch.cat(correct).mean().detach().cpu())}


def evaluate_reconstruction(
    model: VisionRelPosTransformer,
    x: torch.Tensor,
    *,
    args: argparse.Namespace,
    seed: int,
    bias_overrides: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    model.eval()
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    preds = []
    targets = []
    losses = []
    with torch.no_grad():
        for start in range(0, x.shape[0], args.eval_batch_size):
            batch = x[start : start + args.eval_batch_size]
            mask = torch.rand(batch.shape[:2], device=batch.device, generator=gen) < args.mask_rate
            with autocast_context(batch.device, args.amp):
                pred = model.forward_reconstruction(batch, mask, bias_overrides=bias_overrides)
                loss = torch.mean((pred[mask] - batch[mask]).square())
            losses.append(loss.float())
            preds.append(pred[mask].float())
            targets.append(batch[mask].float())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    mse = torch.mean((pred_all - target_all).square())
    var = torch.mean((target_all - target_all.mean(dim=0, keepdim=True)).square()).clamp_min(1e-30)
    model.train()
    return {"loss": float(torch.stack(losses).mean().detach().cpu()), "score": float((1.0 - mse / var).detach().cpu())}


def _shuffle_tables(table: torch.Tensor, *, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=table.device)
    gen.manual_seed(seed)
    flat = table.reshape(-1, table.shape[-2] * table.shape[-1])
    shuffled = []
    for row in flat:
        shuffled.append(row[torch.randperm(row.numel(), device=table.device, generator=gen)])
    return torch.stack(shuffled, dim=0).reshape_as(table)


def _bias_overrides_from_table(table: torch.Tensor, *, side: int, device: torch.device) -> list[torch.Tensor]:
    from toric_pj.diagnostics.relative_table_geometry import relative_table_to_pairwise

    pairwise = relative_table_to_pairwise(table.to(device=device), side)
    return [pairwise[layer].detach() for layer in range(pairwise.shape[0])]


def bias_utility_rows(
    model: VisionRelPosTransformer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    args: argparse.Namespace,
    task: str,
    basis: str,
    seed: int,
) -> list[dict[str, object]]:
    from toric_pj.diagnostics.relative_table_geometry import (
        axial_projection,
        coeff_to_pairwise_bias,
        pairwise_to_relative_table,
        topk_fft_reconstruction,
    )

    side = int(math.sqrt(float(model.n_positions)))
    coeff = torch.stack([block.coeff.detach().float() for block in model.blocks], dim=0)
    pairwise = coeff_to_pairwise_bias(model.basis_matrix.detach().float(), coeff, model.n_positions)
    tables = pairwise_to_relative_table(pairwise, side)
    axis = axial_projection(tables)
    residual = tables - axis
    modes = {
        "normal": None,
        "zero_bias": torch.zeros_like(tables),
        "shuffle_bias": _shuffle_tables(tables, seed=seed + 11),
        "axis_only_bias": axis,
        "obl_residual_only_bias": residual,
        "axis_plus_obl_residual": axis + residual,
        "axis_plus_shuffled_residual": axis + _shuffle_tables(residual, seed=seed + 23),
        "topk_spectrum_only_bias": topk_fft_reconstruction(tables, topk=5),
    }
    rows: list[dict[str, object]] = []
    normal_score = None
    for mode, table in modes.items():
        overrides = None if table is None else _bias_overrides_from_table(table, side=side, device=x.device)
        if task in {"classification", "multitask"}:
            metric = evaluate_classifier(model, x, y, args=args, bias_overrides=overrides)
        else:
            metric = evaluate_reconstruction(model, x, args=args, seed=seed + 4000, bias_overrides=overrides)
        if mode == "normal":
            normal_score = float(metric["score"])
        delta = float(normal_score - float(metric["score"])) if normal_score is not None else 0.0
        rows.append(
            {
                "dataset": args.dataset,
                "task": task,
                "basis": basis,
                "seed": seed,
                "mode": mode,
                "score": metric["score"],
                "loss": metric["loss"],
                "delta_from_normal": delta,
            }
        )
    return rows


def train_task(
    basis: Basis,
    *,
    task: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    patch_dim: int,
    args: argparse.Namespace,
    seed: int,
    init_coeff: torch.Tensor | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
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
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    best_train_score = float("nan")
    best_train_loss = float("nan")
    wall_start = time.time()
    for step in range(args.steps):
        batch, labels = sample_batch(train_x, train_y, batch_size=args.batch_size, generator=gen)
        opt.zero_grad(set_to_none=True)
        with autocast_context(train_x.device, args.amp):
            if task == "classification":
                logits = model.forward_classification(batch)
                loss = F.cross_entropy(logits.float(), labels)
            elif task == "multitask":
                mask = torch.rand(batch.shape[:2], device=batch.device, generator=gen) < args.mask_rate
                pred = model.forward_reconstruction(batch, mask)
                logits = model.forward_classification(batch)
                loss = F.cross_entropy(logits.float(), labels) + args.lambda_recon * torch.mean((pred[mask] - batch[mask]).square())
            else:
                mask = torch.rand(batch.shape[:2], device=batch.device, generator=gen) < args.mask_rate
                pred = model.forward_reconstruction(batch, mask)
                loss = torch.mean((pred[mask] - batch[mask]).square())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            if task in {"classification", "multitask"}:
                train_metric = evaluate_classifier(
                    model,
                    train_x[: min(args.eval_subset, train_x.shape[0])],
                    train_y[: min(args.eval_subset, train_y.shape[0])],
                    args=args,
                )
                test_metric = evaluate_classifier(model, test_x, test_y, args=args)
                metric_name = "accuracy"
            else:
                train_metric = evaluate_reconstruction(model, train_x[: min(args.eval_subset, train_x.shape[0])], args=args, seed=seed + step)
                test_metric = evaluate_reconstruction(model, test_x, args=args, seed=seed + 1000 + step)
                metric_name = "masked_patch_r2"
            if test_metric["score"] > best_score:
                best_score = test_metric["score"]
                best_step = step
                best_loss = test_metric["loss"]
                best_train_score = train_metric["score"]
                best_train_loss = train_metric["loss"]
            curves.append(
                {
                    "dataset": args.dataset,
                    "task": task,
                    "basis": basis.name,
                    "seed": seed,
                    "step": step,
                    "metric": metric_name,
                    "train_score": train_metric["score"],
                    "test_score": test_metric["score"],
                    "train_loss": train_metric["loss"],
                    "test_loss": test_metric["loss"],
                    "elapsed_sec": time.time() - wall_start,
                }
            )
    if task in {"classification", "multitask"}:
        final_train = evaluate_classifier(
            model,
            train_x[: min(args.eval_subset, train_x.shape[0])],
            train_y[: min(args.eval_subset, train_y.shape[0])],
            args=args,
        )
        final_test = evaluate_classifier(model, test_x, test_y, args=args)
        metric_name = "accuracy"
    else:
        final_train = evaluate_reconstruction(model, train_x[: min(args.eval_subset, train_x.shape[0])], args=args, seed=seed + 2000)
        final_test = evaluate_reconstruction(model, test_x, args=args, seed=seed + 3000)
        metric_name = "masked_patch_r2"
    if best_step < 0:
        best_score = final_test["score"]
        best_loss = final_test["loss"]
        best_train_score = final_train["score"]
        best_train_loss = final_train["loss"]
        best_step = args.steps - 1
    use_best = args.score_mode == "best"
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    stats = base_model.bias_stats()
    row: dict[str, object] = {
        "dataset": args.dataset,
        "task": task,
        "basis": basis.name,
        "seed": seed,
        "metric": metric_name,
        "score_mode": args.score_mode,
        "score": best_score if use_best else final_test["score"],
        "loss": best_loss if use_best else final_test["loss"],
        "train_score": best_train_score if use_best else final_train["score"],
        "train_loss": best_train_loss if use_best else final_train["loss"],
        "final_score": final_test["score"],
        "final_loss": final_test["loss"],
        "final_train_score": final_train["score"],
        "final_train_loss": final_train["loss"],
        "best_score": best_score,
        "best_loss": best_loss,
        "best_train_score": best_train_score,
        "best_train_loss": best_train_loss,
        "best_step": best_step,
        "num_features": basis.matrix.shape[1],
        "param_count": sum(param.numel() for param in model.parameters()),
        "wall_sec": time.time() - wall_start,
        **stats,
    }
    if args.export_bias_every and args.export_bias_every > 0:
        from toric_pj.diagnostics.relative_table_geometry import write_geometry_artifacts

        coeff = torch.stack([block.coeff.detach().float().cpu() for block in base_model.blocks], dim=0)
        basis_matrix = base_model.basis_matrix.detach().float().cpu()
        export_dir = (
            Path(args.output_dir)
            / "bias_exports"
            / f"{args.dataset}_{task}_{basis.name}_seed{seed}_steps{args.steps}"
        )
        metadata = {
            "dataset": args.dataset,
            "task": task,
            "basis": basis.name,
            "seed": seed,
            "steps": args.steps,
            "best_step": best_step,
            "score": row["score"],
            "final_score": row["final_score"],
            "train_score": row["train_score"],
            "grid_side": int(math.sqrt(float(base_model.n_positions))),
            "n_layers": len(base_model.blocks),
            "n_heads": args.n_heads,
            "num_features": int(basis_matrix.shape[1]),
        }
        artifacts = write_geometry_artifacts(
            export_dir,
            basis_matrix=basis_matrix,
            coeff=coeff,
            side=int(math.sqrt(float(base_model.n_positions))),
            metadata=metadata,
        )
        if args.bias_ablation_eval:
            utility = bias_utility_rows(
                base_model,
                test_x,
                test_y,
                args=args,
                task=task,
                basis=basis.name,
                seed=seed,
            )
            write_csv(export_dir / "bias_utility.csv", utility)
            artifacts["bias_utility"] = str(export_dir / "bias_utility.csv")
        row.update({f"bias_{key}": value for key, value in artifacts.items()})
    return row, curves


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["task"]), str(row["basis"])), []).append(row)
    out = []
    for (dataset, task, basis), values in sorted(groups.items()):
        scores = np.array([float(row["score"]) for row in values])
        best_scores = np.array([float(row.get("best_score", row["score"])) for row in values])
        final_scores = np.array([float(row.get("final_score", row["score"])) for row in values])
        out.append(
            {
                "dataset": dataset,
                "task": task,
                "basis": basis,
                "n": len(values),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "score_best": float(scores.max()),
                "best_score_mean": float(best_scores.mean()),
                "final_score_mean": float(final_scores.mean()),
                "train_score_mean": float(np.mean([float(row["train_score"]) for row in values])),
                "loss_mean": float(np.mean([float(row["loss"]) for row in values])),
                "num_features": int(values[0]["num_features"]),
                "param_count": int(values[0]["param_count"]),
                "wall_sec_max": float(max(float(row["wall_sec"]) for row in values)),
            }
        )
    return out


def plot_results(output_dir: Path, aggregate_rows: list[dict[str, object]], curves: list[dict[str, object]]) -> None:
    tasks = sorted({str(row["task"]) for row in aggregate_rows})
    fig, axes = plt.subplots(max(1, len(tasks)), 1, figsize=(12, max(4, 4 * len(tasks))), squeeze=False)
    for ax, task in zip(axes.reshape(-1), tasks):
        vals = [row for row in aggregate_rows if row["task"] == task]
        x = np.arange(len(vals))
        y = [float(row["score_mean"]) for row in vals]
        err = [float(row["score_std"]) for row in vals]
        ax.bar(x, y, yerr=err, color="#6c6f95")
        ax.set_xticks(x, [str(row["basis"]) for row in vals], rotation=25, ha="right")
        ax.set_title(task)
        ax.set_ylabel("score")
    fig.tight_layout()
    fig.savefig(output_dir / "basis_accuracy_boxplot.png", dpi=180)
    plt.close(fig)

    if curves:
        fig, ax = plt.subplots(figsize=(12, 5))
        for key in sorted({(row["task"], row["basis"]) for row in curves}):
            vals = [row for row in curves if (row["task"], row["basis"]) == key]
            grouped: dict[int, list[float]] = {}
            for row in vals:
                grouped.setdefault(int(row["step"]), []).append(float(row["test_score"]))
            steps = sorted(grouped)
            ax.plot(steps, [float(np.mean(grouped[step])) for step in steps], label=f"{key[0]} {key[1]}")
        ax.set_xlabel("step")
        ax.set_ylabel("test score")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "train_test_curves.png", dpi=180)
        plt.close(fig)


def write_report(output_dir: Path, summary: dict[str, object], aggregate_rows: list[dict[str, object]]) -> None:
    lines = [
        "# V3-D Real Vision Scaling Report",
        "",
        "Environment:",
        "",
        f"- Device: {summary['device']}",
        f"- Dataset: {summary['dataset']}",
        f"- Mode: {summary['mode']}",
        f"- Patch size / grid side: {summary['patch_size']} / {summary['grid_side']}",
        f"- Depth / dim / heads: {summary['depth']} / {summary['dim']} / {summary['n_heads']}",
        f"- Steps: {summary['steps']}",
        f"- Seeds: {summary['seeds']}",
        f"- Score mode: {summary.get('score_mode', 'final')}",
        f"- LR schedule: {summary.get('lr_schedule', 'cosine')} / decay steps {summary.get('lr_decay_steps', summary['steps'])}",
        f"- Wall seconds: {summary['wall_sec']:.2f}",
        "",
        "## Aggregate",
        "",
        "| dataset | task | basis | n | score mean | score std | best mean | final mean | train score | features | params |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            "| "
            + f"{row['dataset']} | {row['task']} | {row['basis']} | {int(row['n'])} | "
            + f"{float(row['score_mean']):.4f} | {float(row['score_std']):.4f} | "
            + f"{float(row.get('best_score_mean', row['score_mean'])):.4f} | "
            + f"{float(row.get('final_score_mean', row['score_mean'])):.4f} | "
            + f"{float(row['train_score_mean']):.4f} | "
            + f"{int(row['num_features'])} | {int(row['param_count'])} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Classification scores are accuracy.",
            "- Reconstruction scores are masked-patch R2.",
            "- `score mean` follows the selected score mode; `best mean` and `final mean` are shown separately when available.",
            "- `relative_2d_table` is the high-capacity upper-bound style baseline.",
            "",
            "Artifacts:",
            "",
            "- `real_vision_results.csv`",
            "- `real_vision_aggregate.csv`",
            "- `real_vision_curves.csv`",
            "- `basis_accuracy_boxplot.png`",
            "- `train_test_curves.png`",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def data_status(args: argparse.Namespace) -> dict[str, object]:
    device = default_device(args.device)
    root = Path(args.data_root)
    rows = []
    for dataset in ["mnist", "cifar10"]:
        train_x, train_y, test_x, test_y = load_vision_dataset(
            dataset=dataset,
            root=root,
            device=device,
            train_limit=1,
            test_limit=1,
        )
        rows.append(
            {
                "dataset": dataset,
                "train_available": True,
                "test_available": True,
                "sample_shape": tuple(train_x[0].shape),
                "sample_label": int(train_y[0].detach().cpu()),
                "test_label": int(test_y[0].detach().cpu()),
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {"device": str(device), "data_root": str(root.resolve()), "rows": rows}
    (output_dir / "real_vision_data_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "data-status":
        return data_status(args)
    device = default_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
    bases = build_patch_bases(grid_side, device, include_shuffle=args.include_shuffle, seed=args.seed)
    selected_names = set(parse_list(args.bases)) if args.bases != "all" else None
    if args.mode == "smoke" and selected_names is None:
        selected_names = {"no_pos_constant", "toric_order0", "toric_PJ_R2", "relative_2d_table"}
    selected_bases = [basis for basis in bases if selected_names is None or basis.name in selected_names]
    if args.seed_basis_order:
        seed_basis_order = parse_list(args.seed_basis_order)
        basis_seed_indices = {name: idx for idx, name in enumerate(seed_basis_order)}
    else:
        basis_seed_indices = {basis.name: idx for idx, basis in enumerate(selected_bases)}
    tasks = parse_list(args.tasks)
    rows: list[dict[str, object]] = read_csv(output_dir / "real_vision_results.csv") if args.resume else []
    curves: list[dict[str, object]] = read_csv(output_dir / "real_vision_curves.csv") if args.resume else []
    completed = {
        (str(row.get("dataset")), str(row.get("task")), str(row.get("basis")), int(float(row.get("seed", -1))))
        for row in rows
    }
    runs_done = 0
    wall_start = time.time()
    for basis_idx, basis in enumerate(selected_bases):
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
                )
                rows.append(row)
                curves.extend(task_curves)
                completed.add((args.dataset, task, basis.name, seed))
                runs_done += 1
                aggregate_rows = aggregate(rows)
                write_csv(output_dir / "real_vision_results.csv", rows)
                write_csv(output_dir / "real_vision_aggregate.csv", aggregate_rows)
                write_csv(output_dir / "real_vision_curves.csv", curves)
                partial = {
                    "device": str(device),
                    "dataset": args.dataset,
                    "mode": args.mode,
                    "patch_size": args.patch_size,
                    "grid_side": grid_side,
                    "patch_dim": patch_dim,
                    "depth": args.depth,
                    "dim": args.dim,
                    "n_heads": args.n_heads,
                    "steps": args.steps,
                    "seeds": args.seeds,
                    "score_mode": args.score_mode,
                    "lr_schedule": args.lr_schedule,
                    "lr_decay_steps": args.lr_decay_steps or args.steps,
                    "lr_min_ratio": args.lr_min_ratio,
                    "tasks": tasks,
                    "bases": [basis.name for basis in selected_bases],
                    "seed_start_idx": args.seed_start_idx,
                    "seed_basis_order": args.seed_basis_order,
                    "max_runs": args.max_runs,
                    "resume": args.resume,
                    "runs_done_this_invocation": runs_done,
                    "train_count": int(train_x.shape[0]),
                    "test_count": int(test_x.shape[0]),
                    "wall_sec": time.time() - wall_start,
                    "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
                    if torch.cuda.is_available() and device.type == "cuda"
                    else 0,
                    "rows": rows,
                    "aggregate_rows": aggregate_rows,
                    "curves": curves,
                    "status": "partial",
                }
                (output_dir / "real_vision_summary.partial.json").write_text(json.dumps(partial, indent=2), encoding="utf-8")
            if args.max_runs is not None and runs_done >= args.max_runs:
                break
        if args.max_runs is not None and runs_done >= args.max_runs:
            break
    aggregate_rows = aggregate(rows)
    write_csv(output_dir / "real_vision_results.csv", rows)
    write_csv(output_dir / "real_vision_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "real_vision_curves.csv", curves)
    plot_results(output_dir, aggregate_rows, curves)
    peak_mem = 0
    if torch.cuda.is_available() and device.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated(device))
    summary = {
        "device": str(device),
        "dataset": args.dataset,
        "mode": args.mode,
        "patch_size": args.patch_size,
        "grid_side": grid_side,
        "patch_dim": patch_dim,
        "depth": args.depth,
        "dim": args.dim,
        "n_heads": args.n_heads,
        "steps": args.steps,
        "seeds": args.seeds,
        "score_mode": args.score_mode,
        "lr_schedule": args.lr_schedule,
        "lr_decay_steps": args.lr_decay_steps or args.steps,
        "lr_min_ratio": args.lr_min_ratio,
        "tasks": tasks,
        "bases": [basis.name for basis in selected_bases],
        "seed_start_idx": args.seed_start_idx,
        "seed_basis_order": args.seed_basis_order,
        "max_runs": args.max_runs,
        "resume": args.resume,
        "runs_done_this_invocation": runs_done,
        "train_count": int(train_x.shape[0]),
        "test_count": int(test_x.shape[0]),
        "wall_sec": time.time() - wall_start,
        "peak_cuda_memory_bytes": peak_mem,
        "rows": rows,
        "aggregate_rows": aggregate_rows,
        "curves": curves,
    }
    summary_path = output_dir / "real_vision_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    write_report(output_dir, summary, aggregate_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V3-D MNIST/CIFAR real vision scaling.")
    parser.add_argument("--mode", choices=["data-status", "smoke", "main", "overnight"], default="smoke")
    parser.add_argument("--dataset", choices=VISION_DATASETS, default="mnist")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--ffn-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--tasks", type=str, default="classification,reconstruction")
    parser.add_argument("--bases", type=str, default="all")
    parser.add_argument("--mask-rate", type=float, default=0.35)
    parser.add_argument("--lambda-recon", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr-schedule", choices=["cosine", "cosine_hold", "constant"], default="cosine")
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--lr-min-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=None)
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
    parser.add_argument("--score-mode", choices=["final", "best"], default="final")
    parser.add_argument("--export-bias-every", type=int, default=0)
    parser.add_argument("--bias-ablation-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=426)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/v3_real_vision")
    args = parser.parse_args()
    if args.depth is None:
        args.depth = {"data-status": 1, "smoke": 2, "main": 4, "overnight": 6}[args.mode]
    if args.dim is None:
        args.dim = {"data-status": 64, "smoke": 128, "main": 256, "overnight": 384}[args.mode]
    if args.n_heads is None:
        args.n_heads = {"data-status": 4, "smoke": 4, "main": 8, "overnight": 8}[args.mode]
    if args.batch_size is None:
        args.batch_size = {"data-status": 128, "smoke": 256, "main": 1024 if args.dataset == "mnist" else 512, "overnight": 512}[args.mode]
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.steps is None:
        args.steps = {"data-status": 0, "smoke": 300, "main": 20000, "overnight": 50000}[args.mode]
    if args.seeds is None:
        args.seeds = {"data-status": 1, "smoke": 1, "main": 3, "overnight": 5}[args.mode]
    if args.eval_every is None:
        args.eval_every = max(50, args.steps // 40) if args.steps > 0 else 1
    if args.mode == "smoke":
        if args.train_limit is None:
            args.train_limit = 4096 if args.dataset == "mnist" else 8192
        if args.test_limit is None:
            args.test_limit = 2048
    return args


def main() -> None:
    result = run(parse_args())
    printable = {key: value for key, value in result.items() if key not in {"rows", "aggregate_rows", "curves"}}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
