"""Skull-strip BraTS-PED data with GPU-only SynthStrip via Docker.

This script requires a CUDA-enabled SynthStrip image and exits immediately if
GPU activation fails.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
from nibabel import Nifti1Image
import numpy as np


DEFAULT_SYNTHSTRIP_GPU_IMAGE = "synthstrip-cuda-local:0.1"
SPLITS = ("training", "validation")


def _to_docker_unix(path: Path) -> str:
    return path.resolve().as_posix()


def _find_single(pattern: str) -> Path:
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching '{pattern}', found {len(matches)}: {matches}"
        )
    return Path(matches[0])


def apply_mask(source_path: Path, mask_data: np.ndarray, out_path: Path) -> None:
    image: Nifti1Image = nib.load(str(source_path))  # type: ignore[assignment]
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.shape != mask_data.shape:
        raise ValueError(
            f"Shape mismatch: {source_path.name} has shape {data.shape} but mask has {mask_data.shape}."
        )
    # Preserve original affine/header while applying the brain mask.
    masked = data * mask_data.astype(np.float32)
    nib.save(nib.Nifti1Image(masked, image.affine, image.header), str(out_path))


def run_synthstrip_docker(
    patient_dir: Path,
    out_dir: Path,
    t1n_name: str,
    image: str = DEFAULT_SYNTHSTRIP_GPU_IMAGE,
) -> Path:
    # Mount input read-only, write outputs to the case folder, and enable GPU access in Docker.
    input_mount = f"{_to_docker_unix(patient_dir)}:/input:ro"
    output_mount = f"{_to_docker_unix(out_dir)}:/output"

    cmd = [
        "docker", "run", "--rm",
        "-v", input_mount,
        "-v", output_mount,
    ]

    cmd += ["--gpus", "all"]

    cmd += [
        image,
        "-i", f"/input/{t1n_name}",
        "-o", f"/output/{t1n_name}",
        "-m", "/output/brain_mask.nii.gz",
    ]

    cmd += ["-g"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"SynthStrip failed (exit {result.returncode})\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )

    return out_dir / "brain_mask.nii.gz"


def check_docker_gpu_runtime() -> tuple[bool, str]:
    result = subprocess.run(
        [
            "docker", "run", "--rm", "--gpus", "all",
            "nvidia/cuda:12.0.0-base-ubuntu20.04",
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )

    out = result.stdout.strip()
    if result.returncode == 0 and out:
        return True, out
    return False, result.stderr.strip() or "(none)"


def check_synthstrip_image_cuda(image: str) -> tuple[bool, str, str]:
    result = subprocess.run(
        [
            "docker", "run", "--rm", "--gpus", "all",
            "--entrypoint", "python",
            image,
            "-c",
            (
                "import torch; "
                "print(f'cuda_available={torch.cuda.is_available()}'); "
                "print(f'device_count={torch.cuda.device_count()}'); "
                "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'device_name=none')"
            ),
        ],
        capture_output=True,
        text=True,
    )

    out = result.stdout.strip()
    if result.returncode == 0 and "cuda_available=True" in out:
        device_name = "unknown"
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if lines:
            device_name = lines[-1]
        return True, device_name, out
    return False, "none", f"stdout: {out or '(none)'}\nstderr: {result.stderr.strip() or '(none)'}"


def process_patient(patient_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use t1n as the SynthStrip driver volume and reuse its mask for other modalities.
    t1n_path = _find_single(str(patient_dir / "*t1n.nii.gz"))
    t1c_path = _find_single(str(patient_dir / "*t1c.nii.gz"))
    t2f_path = _find_single(str(patient_dir / "*t2f.nii.gz"))
    t2w_path = _find_single(str(patient_dir / "*t2w.nii.gz"))

    mask_path = run_synthstrip_docker(
        patient_dir=patient_dir,
        out_dir=out_dir,
        t1n_name=t1n_path.name,
    )

    mask_image: Nifti1Image = nib.load(str(mask_path))  # type: ignore[assignment]
    mask_data = np.asarray(mask_image.dataobj) > 0

    for src in (t1c_path, t2f_path, t2w_path):
        apply_mask(src, mask_data, out_dir / src.name)

    for seg in glob.glob(str(patient_dir / "*seg.nii.gz")):
        seg_path = Path(seg)
        shutil.copy2(seg, out_dir / seg_path.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Skull-strip BraTS-PED data (GPU-only).")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    data_root = Path(args.data_dir)
    out_root = Path(args.output_dir)

    if not data_root.exists():
        print(f"ERROR: data directory does not exist: {data_root}", file=sys.stderr)
        sys.exit(1)

    runtime_ok, runtime_info = check_docker_gpu_runtime()
    # Fail fast when Docker GPU runtime is not available.
    if not runtime_ok:
        print("ERROR: GPU activation failed at Docker runtime check.", file=sys.stderr)
        print(runtime_info, file=sys.stderr)
        sys.exit(1)

    cuda_ok, device_name, cuda_info = check_synthstrip_image_cuda(DEFAULT_SYNTHSTRIP_GPU_IMAGE)
    if not cuda_ok:
        print(
            "ERROR: GPU activation failed inside synthstrip-cuda-local:0.1.\n"
            "Build/fix the CUDA image and retry, e.g.:\n"
            "  powershell -ExecutionPolicy Bypass -File src/helper/build_synthstrip_gpu_image.ps1",
            file=sys.stderr,
        )
        print(cuda_info, file=sys.stderr)
        sys.exit(1)

    print(f"[device] GPU active: {device_name}")

    total_ok = 0
    total_err = 0

    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        patient_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])

        for patient_dir in patient_dirs:
            out_dir = out_root / split / patient_dir.name
            # Skip cases that already have all four stripped modalities so reruns can resume safely.
            existing = [
                *glob.glob(str(out_dir / "*t1n.nii.gz")),
                *glob.glob(str(out_dir / "*t1c.nii.gz")),
                *glob.glob(str(out_dir / "*t2f.nii.gz")),
                *glob.glob(str(out_dir / "*t2w.nii.gz")),
            ]
            if out_dir.exists() and len(existing) == 4:
                total_ok += 1
                continue

            try:
                process_patient(patient_dir=patient_dir, out_dir=out_dir)
                total_ok += 1
            except Exception as exc:  # pylint: disable=broad-except
                print(f"  [ERROR] {patient_dir.name}: {exc}", file=sys.stderr)
                total_err += 1

    print(f"\nFinished — {total_ok} succeeded, {total_err} failed.")
    if total_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
