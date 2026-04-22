from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from loadmodel import evaluate
from trainmodel import resolve_cases_data_dir, resolve_device, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain MRI segmentation pipeline (local, CUDA-aware)")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true", help="Use all data for training only")
    mode_group.add_argument(
        "--test",
        action="store_true",
        help="Train with train/val/test split and evaluate on test split after training",
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the exact folder containing BraTS case folders",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional learning rate. On resume, this overrides checkpoint LR only when provided.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["reduce_on_plateau", "none"],
        default="reduce_on_plateau",
        help="LR scheduler mode. Default uses ReduceLROnPlateau on validation Dice.",
    )
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--train-ratio", type=float, default=0.75, help="Train split ratio from discovered cases")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio from discovered cases")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--use-split-file", default=None, help="Optional split record JSON for replay testing")

    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--resume-from", default=None, help="Optional checkpoint path to resume training")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    # Validate data layout early so training/eval failures are easier to diagnose.
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    cases_dir = resolve_cases_data_dir(data_dir)
    if not cases_dir.exists() or not cases_dir.is_dir():
        raise FileNotFoundError(f"Resolved case directory does not exist: {cases_dir}")

    if not any(path.is_dir() for path in cases_dir.iterdir()):
        raise RuntimeError(f"No case folders found under: {cases_dir}")

    if args.test:
        if not (0.0 < args.train_ratio < 1.0):
            raise ValueError(f"--train-ratio must be between 0 and 1 (exclusive). Got {args.train_ratio}")
        if not (0.0 < args.val_ratio < 1.0):
            raise ValueError(f"--val-ratio must be between 0 and 1 (exclusive). Got {args.val_ratio}")
        if args.train_ratio + args.val_ratio >= 1.0:
            raise ValueError(
                f"--train-ratio + --val-ratio must be < 1 for --test. Got {args.train_ratio + args.val_ratio:.3f}"
            )

    if args.lr is not None and args.lr <= 0.0:
        raise ValueError(f"--lr must be > 0 when provided. Got {args.lr}")
    if args.lr_patience < 0:
        raise ValueError(f"--lr-patience must be >= 0. Got {args.lr_patience}")
    if not (0.0 < args.lr_factor < 1.0):
        raise ValueError(f"--lr-factor must be between 0 and 1 (exclusive). Got {args.lr_factor}")
    if args.lr_min < 0.0:
        raise ValueError(f"--lr-min must be >= 0. Got {args.lr_min}")


def _build_base_config(args: argparse.Namespace, device: str) -> dict[str, Any]:
    # Keep a single config payload so train/evaluate use the same runtime settings.
    return {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "lr_patience": args.lr_patience,
        "lr_factor": args.lr_factor,
        "lr_min": args.lr_min,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_workers": args.num_workers,
        "device": device,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_path": args.checkpoint_path,
        "resume_from": args.resume_from,
        "use_split_file": args.use_split_file,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    config = _build_base_config(args, str(device))

    # Pick the split policy based on the run mode: train-only, train/val, or train/val/test.
    if args.train:
        config["split_policy"] = "full_train"
        config["run_eval_after_train"] = False
        result = train(config)
    elif args.test:
        config["split_policy"] = "train_val_test"
        config["run_eval_after_train"] = True
        train_result = train(config)

        eval_config = dict(config)
        eval_config["checkpoint_path"] = train_result.get("checkpoint_path") or config.get("checkpoint_path")
        eval_config["eval_split"] = "test"
        eval_config["use_split_file"] = config.get("use_split_file") or train_result.get("split_record_path")
        eval_result = evaluate(eval_config)

        result = {
            **train_result,
            "test_dice": eval_result.get("dice"),
            "test_checkpoint": eval_result.get("checkpoint"),
        }
    else:
        # Default mode trains with a deterministic train/val split and then evaluates on val.
        config["split_policy"] = "train_val_only"
        config["train_ratio"] = 0.8
        config["val_ratio"] = 0.2
        config["run_eval_after_train"] = False
        train_result = train(config)

        eval_config = dict(config)
        eval_config["checkpoint_path"] = train_result.get("checkpoint_path") or config.get("checkpoint_path")
        eval_config["eval_split"] = "val"
        eval_config["use_split_file"] = config.get("use_split_file") or train_result.get("split_record_path")
        eval_result = evaluate(eval_config)

        result = {
            **train_result,
            "validation_dice": eval_result.get("dice"),
            "validation_checkpoint": eval_result.get("checkpoint"),
        }

    print("Run complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()