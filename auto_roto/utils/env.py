"""
Environment Verification
========================

Ensures the correct conda environment is active before running.
"""

import os
import sys

# Required conda environment name
REQUIRED_ENV = "autoroto"


def check_conda_environment(required_env: str = None, exit_on_fail: bool = True) -> bool:
    """
    Verify we're running in the correct conda environment.

    Args:
        required_env: Expected environment name (default: REQUIRED_ENV)
        exit_on_fail: If True, exit with error message on mismatch

    Returns:
        True if environment matches, False otherwise (if exit_on_fail=False)
    """
    if required_env is None:
        required_env = REQUIRED_ENV

    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    if current_env != required_env:
        if exit_on_fail:
            print("\n" + "=" * 60)
            print("WRONG CONDA ENVIRONMENT")
            print("=" * 60)
            print(f"\n  Current environment: {current_env or '(none/base)'}")
            print(f"  Required environment: {required_env}")
            print(f"\n  Please activate the correct environment:")
            print(f"    conda activate {required_env}")
            print("\n" + "=" * 60)
            sys.exit(1)
        return False

    return True


def check_dependencies() -> None:
    """
    Check and report missing dependencies with installation instructions.

    Raises:
        SystemExit if critical dependencies are missing
    """
    missing = []

    try:
        import torch
    except ImportError:
        missing.append(("torch", "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"))

    try:
        import numpy
    except ImportError:
        missing.append(("numpy", "pip install numpy"))

    try:
        import cv2
    except ImportError:
        missing.append(("opencv-python", "pip install opencv-python"))

    if missing:
        print("\n" + "=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        for pkg, cmd in missing:
            print(f"\n  {pkg}:")
            print(f"    {cmd}")
        print("\n" + "=" * 60)
        sys.exit(1)


def get_current_environment() -> str:
    """
    Get the current conda environment name.

    Returns:
        Environment name or empty string if not in conda
    """
    return os.environ.get("CONDA_DEFAULT_ENV", "")


def is_correct_environment(required_env: str = None) -> bool:
    """
    Check if current environment matches required.

    Args:
        required_env: Expected environment name

    Returns:
        True if matching
    """
    if required_env is None:
        required_env = REQUIRED_ENV
    return get_current_environment() == required_env
