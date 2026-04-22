from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import glob
import random
import time
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import grad_scaler, autocast_mode
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.losses.dice import DiceLoss
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import CropForegroundd, RandCropByPosNegLabeld, SpatialPadd
from monai.transforms.intensity.dictionary import RandShiftIntensityd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, RandFlipd, RandRotate90d
from monai.transforms.utility.dictionary import EnsureChannelFirstd, Lambdad, ToTensord
from monai.metrics.meandice import DiceMetric
from monai.transforms.post.array import Activations, AsDiscrete


SPLIT_RECORDS_DIRNAME = "splits"
SPLIT_RECORD_FILENAME = "split_data.json"
DEFAULT_LR = 1e-4
# Tumor misses are weighted a bit higher than background mistakes so the model does not ignore small lesions.
FP_BACKGROUND_WEIGHT = 4
FN_TUMOR_WEIGHT = 3
DICE_LOSS_WEIGHT = 1.0
BCE_LOSS_WEIGHT = 1.5


def resolve_cases_data_dir(data_dir: str | Path) -> Path:
    return Path(data_dir)


def _validate_three_way_ratios(train_ratio: float, val_ratio: float) -> None:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1 (exclusive). Got {train_ratio}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1 (exclusive). Got {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio + val_ratio must be < 1. Got {train_ratio + val_ratio:.3f}")


def _extract_unique_case_ids(files: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("case_id", "")) for item in files if item.get("case_id")})


def _split_case_ids(
    case_ids: list[str],
    split_policy: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,) -> dict[str, list[str]]:
    if not case_ids:
        raise RuntimeError("No discovered cases available for splitting.")

    shuffled_case_ids = list(case_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled_case_ids)

    # Keep each split policy explicit so CLI modes map to predictable partitions.
    if split_policy == "full_train":
        return {"train_cases": shuffled_case_ids, "val_cases": [], "test_cases": []}

    if split_policy == "train_val_only":
        if len(shuffled_case_ids) < 2:
            raise RuntimeError("Need at least 2 cases for train/validation split.")
        train_end = int(round(len(shuffled_case_ids) * train_ratio))
        train_end = max(1, min(len(shuffled_case_ids) - 1, train_end))
        return {
            "train_cases": shuffled_case_ids[:train_end],
            "val_cases": shuffled_case_ids[train_end:],
            "test_cases": [],
        }

    if split_policy == "train_val_test":
        _validate_three_way_ratios(train_ratio, val_ratio)
        if len(shuffled_case_ids) < 3:
            raise RuntimeError("Need at least 3 cases for train/validation/test split.")

        train_end = int(round(len(shuffled_case_ids) * train_ratio))
        train_end = max(1, min(len(shuffled_case_ids) - 2, train_end))

        val_end = int(round(len(shuffled_case_ids) * (train_ratio + val_ratio)))
        val_end = max(train_end + 1, min(len(shuffled_case_ids) - 1, val_end))

        train_cases = shuffled_case_ids[:train_end]
        val_cases = shuffled_case_ids[train_end:val_end]
        test_cases = shuffled_case_ids[val_end:]
        if not train_cases or not val_cases or not test_cases:
            raise RuntimeError("Split produced an empty partition. Adjust ratios or dataset size.")
        return {
            "train_cases": train_cases,
            "val_cases": val_cases,
            "test_cases": test_cases,
        }

    raise ValueError(f"Unsupported split_policy '{split_policy}'")


def _partition_items_by_case_ids(
    files: list[dict[str, Any]],
    split_case_lists: dict[str, list[str]],) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    # Align split case ids with the files currently present on disk.
    case_lookup = {str(item["case_id"]): item for item in files if "case_id" in item}

    missing_case_ids: list[str] = []

    def _resolve(case_list_key: str) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        for case_id in split_case_lists.get(case_list_key, []):
            item = case_lookup.get(case_id)
            if item is None:
                missing_case_ids.append(case_id)
                continue
            resolved.append(item)
        return resolved

    train_files = _resolve("train_cases")
    val_files = _resolve("val_cases")
    test_files = _resolve("test_cases")

    return train_files, val_files, test_files, missing_case_ids


def save_split_record(
    checkpoint_dir: str,
    split_case_lists: dict[str, list[str]],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    split_policy: str,
    data_dir: str,) -> Path:
    output_dir = Path(checkpoint_dir) / SPLIT_RECORDS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep only the latest split JSON to avoid replaying stale splits.
    output_path = output_dir / SPLIT_RECORD_FILENAME
    for existing_json in output_dir.glob("*.json"):
        if existing_json == output_path:
            continue
        existing_json.unlink(missing_ok=True)

    payload = {
        "run_context": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": max(0.0, 1.0 - train_ratio - val_ratio),
            "split_policy": split_policy,
            "data_dir": str(Path(data_dir).resolve()),
        },
        "cases_all": sorted(
            set(split_case_lists.get("train_cases", []))
            | set(split_case_lists.get("val_cases", []))
            | set(split_case_lists.get("test_cases", []))
        ),
        "train_cases": split_case_lists.get("train_cases", []),
        "val_cases": split_case_lists.get("val_cases", []),
        "test_cases": split_case_lists.get("test_cases", []),
        "counts": {
            "all": len(
                set(split_case_lists.get("train_cases", []))
                | set(split_case_lists.get("val_cases", []))
                | set(split_case_lists.get("test_cases", []))
            ),
            "train": len(split_case_lists.get("train_cases", [])),
            "val": len(split_case_lists.get("val_cases", [])),
            "test": len(split_case_lists.get("test_cases", [])),
        },
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_split_case_lists(split_file: str) -> dict[str, list[str]]:
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "train_cases": list(payload.get("train_cases", [])),
        "val_cases": list(payload.get("val_cases", [])),
        "test_cases": list(payload.get("test_cases", [])),
    }


def resolve_split_case_lists(
    files: list[dict[str, Any]],
    *,
    split_file: str | None,
    split_policy: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,) -> dict[str, list[str]]:
    if split_file:
        return load_split_case_lists(split_file)

    return _split_case_ids(
        _extract_unique_case_ids(files),
        split_policy=split_policy,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    value = device_arg.lower().strip()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA but CUDA is not available.")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{device_arg}'. Use one of: auto, cuda, cpu")


def _find_single(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Missing file for pattern: {pattern}")
    return matches[0]


def create_dataset_dict(
    data_path: str,
    include_label: bool = True,
    require_label: bool = True,) -> list[dict[str, Any]]:
    data_root = Path(data_path)
    if not data_root.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    data: list[dict[str, Any]] = []
    for patient_path in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        # Each BraTS case should provide all four MRI modalities.
        t1n = _find_single(str(patient_path / "*t1n.nii.gz"))
        t1c = _find_single(str(patient_path / "*t1c.nii.gz"))
        t2f = _find_single(str(patient_path / "*t2f.nii.gz"))
        t2w = _find_single(str(patient_path / "*t2w.nii.gz"))

        item: dict[str, Any] = {
            "image": [t1n, t1c, t2f, t2w],
            "case_id": patient_path.name,
        }

        if include_label:
            seg_files = glob.glob(str(patient_path / "*seg.nii.gz"))
            if not seg_files:
                if require_label:
                    continue
            else:
                item["label"] = seg_files[0]

        data.append(item)

    if not data:
        raise RuntimeError(f"No patient folders found under: {data_path}")

    return data


def _binarize_label(label: Any) -> Any:
    if isinstance(label, torch.Tensor):
        return (label > 0).to(dtype=torch.float32)

    label_array = np.asarray(label)
    return (label_array > 0).astype(np.float32)


def robust_clip_and_zscore_image(image: Any, lower_pct: float = 0.5, upper_pct: float = 99.5) -> Any:
    image_np = np.asarray(image, dtype=np.float32).copy()
    if image_np.ndim < 4:
        return image_np

    # Normalize each modality channel on foreground voxels only.
    for channel_index in range(image_np.shape[0]):
        channel = image_np[channel_index]
        nonzero_mask = channel != 0
        if not np.any(nonzero_mask):
            continue

        foreground = channel[nonzero_mask]
        lower = float(np.percentile(foreground, lower_pct))
        upper = float(np.percentile(foreground, upper_pct))
        clipped_foreground = np.clip(foreground, lower, upper)

        mean = float(clipped_foreground.mean())
        std = float(clipped_foreground.std())
        if std <= 1e-8:
            continue

        channel_out = channel.copy()
        channel_out[nonzero_mask] = (clipped_foreground - mean) / std
        image_np[channel_index] = channel_out

    return image_np


def _image_robust_normalization() -> Lambdad:
    return Lambdad(keys=["image"], func=robust_clip_and_zscore_image)


def get_transforms(include_label: bool = True, for_training: bool = True) -> Compose:
    if include_label:
        if not for_training:
            # Deterministic preprocessing for validation/evaluation.
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
                    _image_robust_normalization(),
                    ToTensord(keys=["image", "label"]),
                ]
            )

        # Training path adds augmentation after deterministic preprocessing.
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
                _image_robust_normalization(),
                SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
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
            _image_robust_normalization(),
            ToTensord(keys=["image"]),
        ]
    )


def build_train_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 67,
    split_policy: str = "train_val_only",
    checkpoint_dir: str = "checkpoints",
    split_file: str | None = None,) -> tuple[DataLoader, DataLoader | None, Path | None]:
    train_dir = resolve_cases_data_dir(data_dir)

    all_labeled_files = create_dataset_dict(str(train_dir), include_label=True)
    # Split by case id so each case stays in a single partition.
    split_case_lists = resolve_split_case_lists(
        all_labeled_files,
        split_file=split_file,
        split_policy=split_policy,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    if split_file:
        split_record_path: Path | None = Path(split_file)
    else:
        split_record_path = save_split_record(
            checkpoint_dir=checkpoint_dir,
            split_case_lists=split_case_lists,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_policy=split_policy,
            data_dir=data_dir,
        )

    train_files, val_files, test_files, missing_case_ids = _partition_items_by_case_ids(
        all_labeled_files,
        split_case_lists,
    )

    if missing_case_ids:
        print(
            f"Warning: {len(missing_case_ids)} case(s) from split record were not found on disk and were ignored."
        )

    if not train_files:
        raise RuntimeError("Training split is empty after filtering available files.")

    if split_policy == "train_val_only" and not val_files:
        raise RuntimeError("Validation split is empty for train_val_only policy.")

    if split_policy == "train_val_test" and (not val_files or not test_files):
        raise RuntimeError("Validation/test split is empty for train_val_test policy.")

    print(
        f"Loaded {len(all_labeled_files)} labeled cases -> "
        f"train: {len(train_files)} | val: {len(val_files)} | test: {len(test_files)}"
    )

    train_ds = Dataset(train_files, transform=get_transforms(include_label=True, for_training=True))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader: DataLoader | None = None
    if val_files:
        val_ds = Dataset(val_files, transform=get_transforms(include_label=True, for_training=False))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, split_record_path


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
                diff_z // 2,
                diff_z - diff_z // 2,
            ],
        )

        return self.conv(torch.cat([x2, x1], dim=1))


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1):
        super().__init__()

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.dropout = nn.Dropout3d(p=0.2)

        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)

        self.outc = nn.Conv3d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder-decoder path with skip connections.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.dropout(self.down3(x3))

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.outc(x)


def create_model(device: torch.device) -> nn.Module:
    return UNet3D(in_channels=4, out_channels=1).to(device)


class WeightedDiceBCELoss(nn.Module):
    def __init__(
        self,
        fp_background_weight: float = FP_BACKGROUND_WEIGHT,
        fn_tumor_weight: float = FN_TUMOR_WEIGHT,
        dice_weight: float = DICE_LOSS_WEIGHT,
        bce_weight: float = BCE_LOSS_WEIGHT,):
        super().__init__()
        self.fp_background_weight = fp_background_weight
        self.fn_tumor_weight = fn_tumor_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(sigmoid=True, squared_pred=True, smooth_nr=1e-5, smooth_dr=1e-5)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()

        # Combine overlap and voxelwise terms for more stable optimization.
        dice_component = self.dice_loss(logits, target)
        bce_raw = self.bce_loss(logits, target)

        # Use separate voxel weights so missed tumor voxels and background false positives are balanced differently.
        voxel_weights = torch.where(
            target > 0.5,
            torch.full_like(target, self.fn_tumor_weight),
            torch.full_like(target, self.fp_background_weight),
        )
        weighted_bce = (bce_raw * voxel_weights).sum() / voxel_weights.sum().clamp_min(1.0)

        return self.dice_weight * dice_component + self.bce_weight * weighted_bce


def setup_training_objects(model: nn.Module, lr: float, device: torch.device):
    # Centralize optimizer/loss/scaler construction for fresh and resumed runs.
    criterion = WeightedDiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if device.type == "cuda":
        scaler = grad_scaler.GradScaler()
    else:        scaler = None

    return criterion, optimizer, scaler


def _get_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_mode: str,
    patience: int,
    factor: float,
    min_lr: float,) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_mode == "none":
        return None
    if scheduler_mode != "reduce_on_plateau":
        raise ValueError(f"Unsupported lr_scheduler '{scheduler_mode}'")

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=patience,
        factor=factor,
        min_lr=min_lr,
    )


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return autocast_mode.autocast("cuda")
    return nullcontext()


def _format_duration(total_seconds: float) -> str:
    rounded_seconds = int(round(total_seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    total_batches = len(train_loader)

    progress_bar = tqdm(
        train_loader,
        total=total_batches,
        desc="Train",
        unit="case",
        leave=False,
        dynamic_ncols=True,)

    for batch_index, batch in enumerate(progress_bar, start=1):
        images = batch["image"].to(device)
        target = batch["label"].to(device)

        optimizer.zero_grad()
        with _autocast_context(device):
            outputs = model(images)
            loss = criterion(outputs, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_value = float(loss.item())
        total_loss += loss_value
        progress_bar.set_postfix(loss=f"{loss_value:.4f}", seen=f"{batch_index}/{total_batches}")

        # Aggregate train Dice from thresholded predictions.
        train_outputs = [cast(torch.Tensor, post_trans(i)) for i in outputs.detach()]
        dice_metric(y_pred=train_outputs, y=target)

    avg_loss = total_loss / max(1, len(train_loader))
    train_dice = cast(torch.Tensor, dice_metric.aggregate()).item()
    return avg_loss, train_dice


@torch.no_grad()
def evaluate_on_labeled_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,) -> float:
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    has_label = False

    for batch in data_loader:
        if "label" not in batch:
            continue
        has_label = True

        images = batch["image"].to(device)
        target = batch["label"].to(device)
        with _autocast_context(device):
            outputs = model(images)
        eval_outputs = [cast(torch.Tensor, post_trans(i)) for i in outputs]
        dice_metric(y_pred=eval_outputs, y=target)

    if not has_label:
        return float("nan")

    return cast(torch.Tensor, dice_metric.aggregate()).item()


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,) -> Path:
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"checkpoint_epoch_{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        path,
    )
    return path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,) -> int:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", 0))


def train(config: dict[str, Any]) -> dict[str, Any]:
    training_start_time = time.perf_counter()

    set_seed(int(config["seed"]))
    device = resolve_device(str(config["device"]))

    split_policy = str(config.get("split_policy", "train_val_only"))

    train_loader, val_loader, split_record_path = build_train_dataloaders(
        data_dir=str(config["data_dir"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        train_ratio=float(config.get("train_ratio", 0.8)),
        val_ratio=float(config.get("val_ratio", 0.2)),
        seed=int(config["seed"]),
        split_policy=split_policy,
        checkpoint_dir=str(config.get("checkpoint_dir", "checkpoints")),
        split_file=cast(str | None, config.get("use_split_file")),
    )

    configured_lr_value = config.get("lr")
    configured_lr = float(configured_lr_value) if configured_lr_value is not None else None
    # Use default LR only when caller did not provide one.
    initial_lr = configured_lr if configured_lr is not None else DEFAULT_LR

    model = create_model(device)
    criterion, optimizer, scaler = setup_training_objects(model, initial_lr, device)
    scheduler_mode = str(config.get("lr_scheduler", "reduce_on_plateau"))
    scheduler = create_scheduler(
        optimizer,
        scheduler_mode=scheduler_mode,
        patience=int(config.get("lr_patience", 3)),
        factor=float(config.get("lr_factor", 0.5)),
        min_lr=float(config.get("lr_min", 1e-7)),
    )
    print(
        "Loss: WeightedDiceBCE "
        f"(dice_w={DICE_LOSS_WEIGHT:.2f}, bce_w={BCE_LOSS_WEIGHT:.2f}, "
        f"fp_background_w={FP_BACKGROUND_WEIGHT:.2f}, fn_tumor_w={FN_TUMOR_WEIGHT:.2f})"
    )

    if scheduler is None:
        print("LR scheduler: disabled")
    else:
        print(
            "LR scheduler: ReduceLROnPlateau "
            f"(patience={int(config.get('lr_patience', 3))}, "
            f"factor={float(config.get('lr_factor', 0.5))}, "
            f"min_lr={float(config.get('lr_min', 1e-7)):.6g})"
        )

    start_epoch = 0
    resume_from = config.get("resume_from")
    if resume_from:
        # Resume model and optimizer state to continue the same training trajectory.
        start_epoch = load_checkpoint(str(resume_from), model, device, optimizer, scheduler)
        resumed_lr = _get_optimizer_lr(optimizer)
        if configured_lr is None:
            print(f"Resumed optimizer LR from checkpoint: {resumed_lr:.6g}")
        else:
            _set_optimizer_lr(optimizer, configured_lr)
            print(f"Overriding resumed optimizer LR with CLI --lr: {configured_lr:.6g}")
    else:
        if configured_lr is None:
            print(f"Starting fresh with default LR: {initial_lr:.6g}")
        else:
            print(f"Starting fresh with CLI --lr: {initial_lr:.6g}")

    num_epochs = int(config["epochs"])
    checkpoint_dir = str(config["checkpoint_dir"])
    scheduler_skip_notice_printed = False

    last_checkpoint: Path | None = None
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.perf_counter()

        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metric_for_scheduler: float | None = None

        if config.get("run_eval_after_train", False) and val_loader is not None:
            val_dice = evaluate_on_labeled_loader(model, val_loader, device)
            if np.isnan(val_dice):
                print("Internal validation split has no labels. Skipping Dice metric.")
            else:
                print(f"Validation Dice: {val_dice:.4f}")
                val_metric_for_scheduler = float(val_dice)

        if scheduler is not None:
            # Scheduler updates require a validation metric.
            if val_metric_for_scheduler is not None:
                lr_before = _get_optimizer_lr(optimizer)
                scheduler.step(val_metric_for_scheduler)
                lr_after = _get_optimizer_lr(optimizer)
                if lr_after != lr_before:
                    print(f"Scheduler reduced LR: {lr_before:.6g} -> {lr_after:.6g}")
            elif not scheduler_skip_notice_printed:
                print("LR scheduler is enabled but no validation metric is available; skipping scheduler updates.")
                scheduler_skip_notice_printed = True

        last_checkpoint = save_checkpoint(checkpoint_dir, epoch + 1, model, optimizer, train_loss, scheduler)

        epoch_time_seconds = time.perf_counter() - epoch_start_time
        total_time_seconds = time.perf_counter() - training_start_time
        current_lr = _get_optimizer_lr(optimizer)
        print(
            f"Epoch {epoch + 1}/{start_epoch + num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"LR: {current_lr:.6g} | "
            f"Epoch Time: {_format_duration(epoch_time_seconds)} | "
            f"Total Time: {_format_duration(total_time_seconds)}"
        )

    training_time_seconds = time.perf_counter() - training_start_time
    print(f"Training completed in {_format_duration(training_time_seconds)}")

    return {
        "checkpoint_path": str(last_checkpoint) if last_checkpoint else None,
        "training_time": _format_duration(training_time_seconds),
        "split_record_path": str(split_record_path) if split_record_path else None,
    }