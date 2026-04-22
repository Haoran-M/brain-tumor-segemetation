from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data.dataset import Dataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd
from monai.transforms.utility.dictionary import EnsureChannelFirstd, Lambdad, ToTensord

from loadmodel import load_trained_model
from trainmodel import (
    _partition_items_by_case_ids,
    create_dataset_dict,
    robust_clip_and_zscore_image,
    resolve_cases_data_dir,
    resolve_device,
    resolve_split_case_lists,)


DEFAULT_OUTPUT_DIR = Path("outputs/new_visualizations")
DEFAULT_SLICES_PER_PLANE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize predictions for a specific case or random test-split cases")
    parser.add_argument("--model", required=True, help="Checkpoint path")
    parser.add_argument("--data", required=True, help="Path to folder containing all case folders")
    parser.add_argument("--split-path", default=None, help="Path to split JSON file (required for random mode)")
    parser.add_argument("--case-id", default=None, help="Exact case ID to visualize directly from data")
    parser.add_argument("--i", type=int, default=1, help="Number of random test cases to visualize")
    return parser


def _binarize_label(label: Any) -> Any:
    if isinstance(label, torch.Tensor):
        return (label > 0).to(dtype=torch.float32)
    label_array = np.asarray(label)
    return (label_array > 0).astype(np.float32)


def _deterministic_transform(include_label: bool) -> Compose:
    if include_label:
        # Match eval preprocessing without random augmentation.
        return Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(
                    keys=["image", "label"],
                    axcodes="RAS",
                    labels=(("L", "R"), ("P", "A"), ("I", "S")),
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Lambdad(keys=["label"], func=_binarize_label),
                Lambdad(keys=["image"], func=robust_clip_and_zscore_image),
                ToTensord(keys=["image", "label"]),
            ]
        )

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(
                keys=["image"],
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            Lambdad(keys=["image"], func=robust_clip_and_zscore_image),
            ToTensord(keys=["image"]),
        ]
    )


def _sample_plane_indices(
    shape: tuple[int, int, int],
    axis: int,
    num_slices: int,
    gt_volume: np.ndarray | None,) -> list[int]:
    axis_size = shape[axis]
    if axis_size <= 0:
        return [0]

    if gt_volume is not None:
        reduce_axes = tuple(i for i in range(3) if i != axis)
        profile = gt_volume.sum(axis=reduce_axes)
        # Prefer slices that actually contain tumor so the PNGs show informative regions.
        positive_indices = np.where(profile > 0)[0].tolist()
    else:
        positive_indices = []

    candidate_indices = positive_indices if positive_indices else list(range(axis_size))
    sample_count = max(1, min(num_slices, len(candidate_indices)))
    selected = random.sample(candidate_indices, k=sample_count)
    return sorted(selected)


def _extract_plane_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    if plane == "axial":
        return volume[:, :, index]
    if plane == "coronal":
        return volume[:, index, :]
    if plane == "sagittal":
        return volume[index, :, :]
    raise ValueError(f"Unsupported plane '{plane}'")


def _compute_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    # Binary Dice overlap for case-level summary.
    pred_binary = (pred_mask > 0.5).astype(np.float32)
    target_binary = (target_mask > 0.5).astype(np.float32)
    intersection = float((pred_binary * target_binary).sum())
    denominator = float(pred_binary.sum() + target_binary.sum())
    return (2.0 * intersection + 1e-5) / (denominator + 1e-5)


def _save_png(
    image: np.ndarray,
    pred_mask: np.ndarray,
    case_id: str,
    output_path: Path,
    label: np.ndarray | None = None,
    slices_per_plane: int = 3,
    dice_score: float | None = None,) -> None:

    flair = image[2]
    pred = pred_mask[0]
    gt = label[0] if label is not None else None

    plane_to_axis = {"axial": 2, "coronal": 1, "sagittal": 0}
    # Lay out each plane as image + label/prediction/overlay so the comparisons stay easy to read.
    plane_indices = {
        plane: _sample_plane_indices(flair.shape, axis, slices_per_plane, gt)
        for plane, axis in plane_to_axis.items()
    }

    rows = max(len(indices) for indices in plane_indices.values())
    columns_per_plane = 4 if gt is not None else 3
    total_columns = columns_per_plane * len(plane_to_axis)

    fig, axes = plt.subplots(rows, total_columns, figsize=(4 * total_columns, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(rows):
        for plane_idx, plane in enumerate(("axial", "coronal", "sagittal")):
            base_col = plane_idx * columns_per_plane
            current_indices = plane_indices[plane]

            for col_offset in range(columns_per_plane):
                axes[row, base_col + col_offset].axis("off")

            if row >= len(current_indices):
                continue

            slice_idx = current_indices[row]
            image_slice = _extract_plane_slice(flair, plane, slice_idx)
            pred_slice = _extract_plane_slice(pred, plane, slice_idx)

            axes[row, base_col].imshow(image_slice, cmap="gray")
            axes[row, base_col].set_title(f"{plane} idx={slice_idx}")
            axes[row, base_col].axis("off")

            if gt is not None:
                gt_slice = _extract_plane_slice(gt, plane, slice_idx)

                axes[row, base_col + 1].imshow(gt_slice, cmap="gray", vmin=0, vmax=1)
                axes[row, base_col + 1].set_title("Ground Truth")
                axes[row, base_col + 1].axis("off")

                axes[row, base_col + 2].imshow(pred_slice, cmap="gray", vmin=0, vmax=1)
                axes[row, base_col + 2].set_title("Prediction")
                axes[row, base_col + 2].axis("off")

                axes[row, base_col + 3].imshow(image_slice, cmap="gray")
                axes[row, base_col + 3].imshow(pred_slice, cmap="jet", alpha=0.35, vmin=0, vmax=1)
                axes[row, base_col + 3].contour(gt_slice, levels=[0.5], colors="lime", linewidths=0.7)
                axes[row, base_col + 3].set_title("Overlay")
                axes[row, base_col + 3].axis("off")
            else:
                axes[row, base_col + 1].imshow(pred_slice, cmap="gray", vmin=0, vmax=1)
                axes[row, base_col + 1].set_title("Prediction")
                axes[row, base_col + 1].axis("off")

                axes[row, base_col + 2].imshow(image_slice, cmap="gray")
                axes[row, base_col + 2].imshow(pred_slice, cmap="jet", alpha=0.35, vmin=0, vmax=1)
                axes[row, base_col + 2].set_title("Overlay")
                axes[row, base_col + 2].axis("off")

    title = f"Case visualization (random axial/coronal/sagittal): {case_id}"
    if dice_score is not None:
        title += f" | Dice: {dice_score:.4f}"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def run_visualization(
    model_path: str,
    data_path: str,
    split_path: str | None,
    num_cases: int,
    case_id: str | None,) -> dict[str, Any]:

    if num_cases <= 0:
        raise ValueError(f"--i must be >= 1. Got {num_cases}")

    device = resolve_device("auto")

    checkpoint = Path(model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint}")

    model = load_trained_model(str(checkpoint), device)

    data_dir = resolve_cases_data_dir(data_path)
    all_cases = create_dataset_dict(str(data_dir), include_label=True, require_label=False)

    selection_mode: str
    selected_cases: list[dict[str, Any]]
    split_file: Path | None = None

    if case_id is not None:
        # Direct mode: visualize a specific case id.
        case_lookup = {str(item.get("case_id", "")): item for item in all_cases}
        if case_id not in case_lookup:
            raise RuntimeError(f"Requested case_id not found in data directory: {case_id}")
        selected_cases = [case_lookup[case_id]]
        selection_mode = "direct_case"
    else:
        # Random mode: sample cases from the test split definition.
        if split_path is None:
            raise ValueError("Provide --case-id for direct mode, or provide --split-path for random test-split mode.")

        split_file = Path(split_path)
        if not split_file.exists():
            raise FileNotFoundError(f"Split file does not exist: {split_file}")

        split_case_lists = resolve_split_case_lists(
            all_cases,
            split_file=str(split_file),
            split_policy="train_val_test",
            seed=67,
            train_ratio=0.75,
            val_ratio=0.15,
        )

        _, _, test_cases, missing_case_ids = _partition_items_by_case_ids(all_cases, split_case_lists)
        if missing_case_ids:
            print(f"Warning: {len(missing_case_ids)} case(s) from split file were missing on disk and ignored.")

        if not test_cases:
            raise RuntimeError("No test split cases found.")

        selected_count = min(num_cases, len(test_cases))
        selected_cases = random.sample(test_cases, k=selected_count)
        selection_mode = "random_test_split"

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    case_summaries: list[dict[str, Any]] = []

    for selected_case in selected_cases:
        has_label = "label" in selected_case
        ds = Dataset([selected_case], transform=_deterministic_transform(include_label=has_label))
        batch = cast(dict[str, torch.Tensor], ds[0])

        image = cast(torch.Tensor, batch["image"]).unsqueeze(0).to(device)
        logits = model(image)
        pred = (torch.sigmoid(logits) > 0.5).float()

        image_np = image.detach().cpu().numpy()[0]
        pred_np = pred.detach().cpu().numpy()[0]
        label_np = None
        dice_score: float | None = None
        if has_label:
            # Compute Dice only when ground truth is present.
            label_np = cast(torch.Tensor, batch["label"]).unsqueeze(0).detach().cpu().numpy()[0]
            dice_score = _compute_dice(pred_np[0], label_np[0])

        selected_case_id = str(selected_case.get("case_id", "unknown_case"))
        output_path = output_dir / f"{selected_case_id}_test_visual.png"

        _save_png(
            image=image_np,
            pred_mask=pred_np,
            case_id=selected_case_id,
            output_path=output_path,
            label=label_np,
            slices_per_plane=DEFAULT_SLICES_PER_PLANE,
            dice_score=dice_score,
        )

        print(f"Saved PNG: {output_path}")
        if dice_score is not None:
            print(f"Case Dice: {dice_score:.4f}")

        saved_paths.append(str(output_path))
        case_summaries.append(
            {
                "case_id": selected_case_id,
                "has_label": has_label,
                "dice": dice_score,
            }
        )

    return {
        "device": str(device),
        "checkpoint": str(checkpoint),
        "selection_mode": selection_mode,
        "split": "test" if selection_mode == "random_test_split" else None,
        "split_file": str(split_file) if split_file is not None else None,
        "case_id_requested": case_id,
        "data_dir": str(data_dir),
        "num_cases_requested": num_cases,
        "num_cases_visualized": len(selected_cases),
        "cases": case_summaries,
        "png_files": saved_paths,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_visualization(
        model_path=args.model,
        data_path=args.data,
        split_path=args.split_path,
        num_cases=args.i,
        case_id=args.case_id,
    )

    print("Run complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
