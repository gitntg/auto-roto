"""
GPU Memory Management
=====================

Utilities for managing GPU memory between pipeline stages.
"""

import gc
import logging

logger = logging.getLogger("AutoRoto.GPU")


def clear_gpu_memory(sync: bool = True) -> None:
    """
    Clear GPU memory between stages.

    This is essential when running multiple models sequentially,
    as VRAM is limited and models need to be unloaded.

    Args:
        sync: If True, synchronize CUDA before clearing
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if sync:
                torch.cuda.synchronize()
    except ImportError:
        pass

    gc.collect()
    logger.debug("GPU memory cleared")


def get_gpu_memory_info() -> dict:
    """
    Get current GPU memory usage information.

    Returns:
        Dictionary with memory info or empty dict if CUDA unavailable
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
            }
    except ImportError:
        pass
    return {}


def log_gpu_memory(prefix: str = "") -> None:
    """
    Log current GPU memory usage.

    Args:
        prefix: Optional prefix for the log message
    """
    info = get_gpu_memory_info()
    if info:
        allocated_mb = info["allocated"] / 1024 / 1024
        reserved_mb = info["reserved"] / 1024 / 1024
        msg = f"GPU Memory: {allocated_mb:.1f}MB allocated, {reserved_mb:.1f}MB reserved"
        if prefix:
            msg = f"{prefix}: {msg}"
        logger.debug(msg)


def move_model_to_cpu(model) -> None:
    """
    Move a PyTorch model to CPU to free GPU memory.

    Args:
        model: PyTorch model to move
    """
    try:
        model.cpu()
        clear_gpu_memory()
    except Exception:
        pass


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device(preferred: str = "cuda") -> str:
    """
    Get the best available device.

    Args:
        preferred: Preferred device ("cuda" or "cpu")

    Returns:
        Device string ("cuda" or "cpu")
    """
    if preferred == "cuda" and is_cuda_available():
        return "cuda"
    return "cpu"
