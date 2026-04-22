from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset

from trainmodel import (
    _partition_items_by_case_ids,
    create_dataset_dict,
    create_model,
    get_transforms,
    load_checkpoint,
    resolve_cases_data_dir,
    resolve_device,
    resolve_split_case_lists,
)


def _autocast_context(device: torch.device):
    return nullcontext()


def resolve_checkpoint_path(checkpoint_path: str | None, checkpoint_dir: str | None) -> Path:
    # Prefer an explicit checkpoint path; otherwise fall back to the newest checkpoint in the directory.
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        return path

    if not checkpoint_dir:
        raise ValueError("Provide --checkpoint-path or --checkpoint-dir")

    cdir = Path(checkpoint_dir)
    candidates = sorted(cdir.glob("checkpoint_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")
    return candidates[-1]


def load_trained_model(checkpoint_path: str, device: torch.device):
    # Recreate architecture, restore weights, and switch to inference mode.
    model = create_model(device)
    _ = load_checkpoint(checkpoint_path, model, device, optimizer=None)
    model.eval()
    return model


def build_internal_split_loader(
    data_dir: str,
    split_name: str,
    include_label: bool,
    split_policy: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    split_file: str | None = None,
    num_workers: int = 0,
):
    # Always derive evaluation/inference sets from the same split logic used in training.
    train_dir = resolve_cases_data_dir(data_dir)
    all_labeled_files = create_dataset_dict(str(train_dir), include_label=True)
    split_case_lists = resolve_split_case_lists(
        all_labeled_files,
        split_file=split_file,
        split_policy=split_policy,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    _, val_files, test_files, missing_case_ids = _partition_items_by_case_ids(all_labeled_files, split_case_lists)
    if missing_case_ids:
        print(f"Warning: {len(missing_case_ids)} case(s) from split file were missing on disk and ignored.")

    if split_name == "val":
        selected_files = val_files
    elif split_name == "test":
        selected_files = test_files
    else:
        raise ValueError(f"Unsupported split_name '{split_name}'. Use 'val' or 'test'.")

    if not selected_files:
        raise RuntimeError(f"Selected split '{split_name}' is empty.")

    transforms = get_transforms(include_label=include_label, for_training=False)

    ds = Dataset(selected_files, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return selected_files, loader


def _compute_batch_dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    # Compute soft-to-hard Dice on thresholded sigmoid predictions.
    preds = (torch.sigmoid(logits) > 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * target).sum(dim=dims)
    denominator = preds.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return float(dice.mean().item())


@torch.no_grad()
def run_inference(
    model,
    data_items: list[dict[str, Any]],
    loader: DataLoader,
    device: torch.device,
    output_dir: str,
) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for idx, batch in enumerate(loader):
        image = batch["image"].to(device)
        with _autocast_context(device):
            pred = model(image)

        # Save probabilities to preserve confidence for downstream threshold tuning.
        prob_volume = torch.sigmoid(pred).cpu().numpy()[0, 0].astype(np.float32)
        case_id = data_items[idx].get("case_id", f"case_{idx:04d}")
        output_path = out_dir / f"{case_id}_prediction_probability.nii.gz"

        nib.save(nib.Nifti1Image(prob_volume, np.eye(4)), str(output_path))
        saved_paths.append(output_path)

    return saved_paths


@torch.no_grad()
def evaluate(config: dict[str, Any],) -> dict[str, Any]:
    device = resolve_device(str(config["device"]))
    checkpoint = resolve_checkpoint_path(config.get("checkpoint_path"), config.get("checkpoint_dir"))
    model = load_trained_model(str(checkpoint), device)

    _, loader = build_internal_split_loader(
        data_dir=str(config["data_dir"]),
        split_name=str(config.get("eval_split", "val")),
        include_label=True,
        split_policy=str(config.get("split_policy", "train_val_only")),
        train_ratio=float(config.get("train_ratio", 0.8)),
        val_ratio=float(config.get("val_ratio", 0.2)),
        seed=int(config.get("seed", 67)),
        split_file=config.get("use_split_file"),
        num_workers=int(config["num_workers"]),)

    had_labels = False
    total_dice = 0.0
    num_batches = 0
    for batch in loader:
        # Skip unlabeled batches defensively if a mixed loader is ever provided.
        if "label" not in batch:
            continue
        had_labels = True

        image = batch["image"].to(device)
        label = batch["label"].to(device)

        with _autocast_context(device):
            pred = model(image)
        total_dice += _compute_batch_dice(pred, label)
        num_batches += 1

    if not had_labels:
        raise RuntimeError("No labels found in internal test split. Evaluate mode requires labels.")

    dice_score = total_dice / max(1, num_batches)
    print(f"Evaluation Dice: {dice_score:.4f}")

    return {
        "device": str(device),
        "checkpoint": str(checkpoint),
        "dice": dice_score,
    }


def inference(config: dict[str, Any]) -> dict[str, Any]:
    device = resolve_device(str(config["device"]))
    checkpoint = resolve_checkpoint_path(config.get("checkpoint_path"), config.get("checkpoint_dir"))
    model = load_trained_model(str(checkpoint), device)

    data_items, loader = build_internal_split_loader(
        # Inference targets test cases and does not require labels.
        data_dir=str(config["data_dir"]),
        split_name="test",
        include_label=False,
        split_policy=str(config.get("split_policy", "train_val_test")),
        train_ratio=float(config.get("train_ratio", 0.75)),
        val_ratio=float(config.get("val_ratio", 0.15)),
        seed=int(config.get("seed", 67)),
        split_file=config.get("use_split_file"),
        num_workers=int(config["num_workers"]),
    )

    saved = run_inference(
        model=model,
        data_items=data_items,
        loader=loader,
        device=device,
        output_dir=str(config["output_dir"]),
    )

    print(f"Saved {len(saved)} prediction file(s) to {config['output_dir']}")
    return {
        "device": str(device),
        "checkpoint": str(checkpoint),
        "saved_files": [str(p) for p in saved],
    }