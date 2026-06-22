from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import transforms

from toric_pj.experiments.v3_real_vision_scaling import (
    TinyImageNetDataset,
    load_vision_dataset,
    patchify,
    vision_num_classes,
)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def label_summary(labels: torch.Tensor) -> dict[str, object]:
    cpu_labels = labels.detach().cpu()
    unique = torch.unique(cpu_labels)
    return {
        "min": int(cpu_labels.min().item()),
        "max": int(cpu_labels.max().item()),
        "unique_count": int(unique.numel()),
        "first_20": [int(item) for item in cpu_labels[:20].tolist()],
    }


def tensor_summary(images: torch.Tensor) -> dict[str, object]:
    cpu_images = images.detach().float().cpu()
    return {
        "shape": list(cpu_images.shape),
        "dtype": str(images.dtype),
        "device": str(images.device),
        "min": float(cpu_images.min().item()),
        "max": float(cpu_images.max().item()),
        "mean": float(cpu_images.mean().item()),
        "std": float(cpu_images.std().item()),
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    resize32 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    train_ds = TinyImageNetDataset(data_root, split="train", transform=resize32)
    val_ds = TinyImageNetDataset(data_root, split="val", transform=resize32)
    train_images, train_y, val_images, val_y = load_vision_dataset(
        dataset="tiny-imagenet",
        root=data_root,
        device=device,
        train_limit=args.train_limit,
        test_limit=args.val_limit,
    )
    train_patches, train_grid_side, train_patch_dim = patchify(train_images, args.patch_size)
    val_patches, val_grid_side, val_patch_dim = patchify(val_images, args.patch_size)
    if train_grid_side != val_grid_side:
        raise ValueError("train/val patch grid mismatch")
    if train_patch_dim != val_patch_dim:
        raise ValueError("train/val patch dim mismatch")

    cuda_memory = {}
    if device.type == "cuda":
        cuda_memory = {
            "allocated_mib": float(torch.cuda.memory_allocated(device) / 1024**2),
            "reserved_mib": float(torch.cuda.memory_reserved(device) / 1024**2),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        }

    return {
        "dataset": "tiny-imagenet",
        "data_root": str(data_root.resolve()),
        "tiny_imagenet_root": str((data_root / "tiny-imagenet-200").resolve()),
        "device": str(device),
        "num_classes": vision_num_classes("tiny-imagenet"),
        "split_counts": {
            "train": len(train_ds),
            "val": len(val_ds),
        },
        "sample_limits": {
            "train": int(args.train_limit),
            "val": int(args.val_limit),
        },
        "train_images": tensor_summary(train_images),
        "val_images": tensor_summary(val_images),
        "train_labels": label_summary(train_y),
        "val_labels": label_summary(val_y),
        "patchify": {
            "patch_size": int(args.patch_size),
            "grid_side": int(train_grid_side),
            "num_patches": int(train_patches.shape[1]),
            "patch_dim": int(train_patch_dim),
            "train_shape": list(train_patches.shape),
            "val_shape": list(val_patches.shape),
        },
        "cuda_memory": cuda_memory,
    }


def write_summary(report: dict[str, object], output_dir: Path) -> None:
    patchify_report = report["patchify"]
    train_labels = report["train_labels"]
    val_labels = report["val_labels"]
    lines = [
        "# V8 TinyImageNet Loader Smoke",
        "",
        f"- dataset: {report['dataset']}",
        f"- data_root: {report['data_root']}",
        f"- device: {report['device']}",
        f"- num_classes: {report['num_classes']}",
        f"- split_counts: train={report['split_counts']['train']}, val={report['split_counts']['val']}",
        f"- train_images: {report['train_images']['shape']}",
        f"- val_images: {report['val_images']['shape']}",
        (
            "- train_labels: "
            f"min={train_labels['min']}, max={train_labels['max']}, "
            f"unique={train_labels['unique_count']}, first_20={train_labels['first_20']}"
        ),
        (
            "- val_labels: "
            f"min={val_labels['min']}, max={val_labels['max']}, "
            f"unique={val_labels['unique_count']}, first_20={val_labels['first_20']}"
        ),
        (
            "- patchify: "
            f"patch_size={patchify_report['patch_size']}, "
            f"grid_side={patchify_report['grid_side']}, "
            f"num_patches={patchify_report['num_patches']}, "
            f"patch_dim={patchify_report['patch_dim']}"
        ),
        f"- cuda_memory: {report['cuda_memory']}",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a V8 TinyImageNet loader smoke test.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/v8_tinyimagenet_loader_smoke")
    parser.add_argument("--train-limit", type=int, default=256)
    parser.add_argument("--val-limit", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report = build_report(args)
    (output_dir / "loader_smoke_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary(report, output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
