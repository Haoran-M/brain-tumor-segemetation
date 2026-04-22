from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def install(requirements_file: Path, upgrade_pip: bool, cuda: bool) -> None:
    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    # Optional pip self-upgrade to avoid installer/version conflicts.
    if upgrade_pip:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install project baseline dependencies first.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])

    if cuda:
        # Replace CPU wheels with CUDA-enabled PyTorch wheels.
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu130",
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Install project dependencies from requirements.txt")
    parser.add_argument(
        "--requirements",
        default=str(Path(__file__).resolve().parent.parent / "requirements.txt"),
        help="Path to requirements file",
    )
    parser.add_argument("--upgrade-pip", action="store_true", help="Upgrade pip before installing")
    parser.add_argument("--cuda", action="store_true", help="Install CUDA-enabled PyTorch (cu130)")
    args = parser.parse_args()

    install(Path(args.requirements), args.upgrade_pip, args.cuda)
    print("Dependency installation completed.")


if __name__ == "__main__":
    main()