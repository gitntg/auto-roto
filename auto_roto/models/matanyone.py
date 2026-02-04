"""
MatAnyone Model Wrapper
=======================

Adapter for MatAnyone v1 (Matting Anything) video matting.

MatAnyone uses temporal consistency for video matting, producing
high-quality alpha mattes from SAM masks.

Note: This module was previously named mam2 but renamed for clarity
as it uses MatAnyone v1, not v2.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("AutoRoto.Models.MatAnyone")


@dataclass
class MatAnyoneAdapter:
    """Adapter object holding MatAnyone model and inference function."""
    backend: str
    model: Optional[Any] = None
    infer_fn: Optional[Callable[[Any, np.ndarray, np.ndarray], Any]] = None
    processor: Optional[Any] = None
    device: Optional[str] = None


def resolve_repo_path(repo_path: str) -> Path:
    """Resolve repository path (relative to script or absolute)."""
    path = Path(repo_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).parent.parent.parent / path).resolve()


def resolve_checkpoint_path(checkpoint_path: str) -> Path:
    """Resolve checkpoint path (relative to script or absolute)."""
    path = Path(checkpoint_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).parent.parent.parent / path).resolve()


def resolve_device(device: str, log: logging.Logger = None) -> str:
    """Resolve compute device, falling back to CPU if CUDA unavailable."""
    if not device:
        return device
    try:
        import torch
    except ImportError:
        return device
    if device == "cuda" and not torch.cuda.is_available():
        if log:
            log.warning("CUDA not available, falling back to CPU.")
        return "cpu"
    return device


def normalize_alpha(alpha: Any) -> np.ndarray:
    """Normalize alpha output to float32 [0, 1] range."""
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(alpha, torch.Tensor):
        alpha = alpha.detach().cpu().numpy()

    alpha = np.asarray(alpha)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0] if alpha.shape[2] == 1 else alpha.mean(axis=2)
    alpha = alpha.astype(np.float32)

    max_val = float(alpha.max()) if alpha.size else 1.0
    if max_val > 1.0:
        if max_val <= 255.0:
            alpha = alpha / 255.0
        elif max_val <= 65535.0:
            alpha = alpha / 65535.0
        else:
            alpha = alpha / max_val if max_val > 0 else alpha

    return np.clip(alpha, 0.0, 1.0)


def load_matanyone_v1_adapter(
    repo_path: Path,
    checkpoint_path: Path,
    device: str,
    log: logging.Logger,
    mem_every: int = 3,
    max_mem_frames: int = 10,
    top_k: int = 50,
    use_long_term: bool = True,
    max_internal_size: int = -1
) -> MatAnyoneAdapter:
    """
    Load MatAnyone v1 model and create inference adapter.

    Args:
        repo_path: Path to MatAnyone repository
        checkpoint_path: Path to model checkpoint
        device: Compute device
        log: Logger instance
        mem_every: Memory frame interval (lower = better quality, slower)
        max_mem_frames: Max memory frames (higher = better quality, more VRAM)
        top_k: Top-k memory matching (higher = more accurate)
        use_long_term: Enable long-term memory for temporal consistency
        max_internal_size: Max internal processing size (-1 = full resolution)

    Returns:
        Configured MatAnyoneAdapter
    """
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        from matanyone.inference.inference_core import InferenceCore
        from matanyone.model.matanyone import MatAnyone
        from matanyone.utils.device import get_default_device
    except Exception as exc:
        raise ImportError(
            "Unable to import MatAnyone v1 modules. "
            "Ensure the repo contains matanyone/ and its dependencies."
        ) from exc

    log.info("Detected MatAnyone v1 repo; using InferenceCore backend.")

    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MatAnyone checkpoint not found: {checkpoint_path}")

    log.info(f"Loading MatAnyone from LOCAL checkpoint: {checkpoint_path}")

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    config_dir = str(repo_path / "matanyone" / "config")
    initialize_config_dir(version_base='1.3.2', config_dir=config_dir)
    cfg = compose(config_name="eval_matanyone_config")

    with open_dict(cfg):
        cfg['weights'] = str(checkpoint_path)
        cfg['mem_every'] = mem_every
        cfg['max_mem_frames'] = max_mem_frames
        cfg['top_k'] = top_k
        cfg['use_long_term'] = use_long_term
        cfg['max_internal_size'] = max_internal_size
        if use_long_term:
            cfg['long_term']['max_mem_frames'] = max_mem_frames

    log.info(f"Quality settings: mem_every={mem_every}, "
             f"max_mem_frames={max_mem_frames}, "
             f"top_k={top_k}, long_term={use_long_term}, "
             f"max_internal_size={max_internal_size}")

    # Load model
    if device and device != "cpu":
        model = MatAnyone(cfg, single_object=True).to(device).eval()
        model_weights = torch.load(str(checkpoint_path), map_location=device)
    else:
        # CPU path: load model on CPU, map weights to CPU
        model = MatAnyone(cfg, single_object=True).cpu().eval()
        model_weights = torch.load(str(checkpoint_path), map_location="cpu")

    model.load_weights(model_weights)
    log.info("✓ Loaded MatAnyone from local checkpoint")

    if not device or device == "auto":
        device = str(get_default_device())

    model = model.to(device).eval()
    processor = InferenceCore(model, cfg=model.cfg, device=device)

    return MatAnyoneAdapter(
        backend="matanyone_v1",
        model=model,
        processor=processor,
        device=device,
    )


def load_matanyone_adapter(
    repo_path: str = "./MatAnyone",
    checkpoint_path: str = "./checkpoints/matanyone.pth",
    device: str = "cuda",
    logger: logging.Logger = None,
    **kwargs
) -> MatAnyoneAdapter:
    """
    Load MatAnyone adapter with automatic backend detection.

    Args:
        repo_path: Path to MatAnyone repository
        checkpoint_path: Path to model checkpoint
        device: Compute device
        logger: Optional logger instance
        **kwargs: Additional parameters for v1 adapter (mem_every, etc.)

    Returns:
        Configured MatAnyoneAdapter
    """
    log = logger or logging.getLogger("MatAnyone")

    repo = resolve_repo_path(repo_path)
    checkpoint = resolve_checkpoint_path(checkpoint_path)
    device = resolve_device(device, log)

    if not repo.exists():
        raise FileNotFoundError(
            f"MatAnyone repo not found at {repo}. "
            "Clone it or pass --repo to point at the checkout."
        )

    # Check for MatAnyone v1 structure
    if (repo / "matanyone").exists():
        return load_matanyone_v1_adapter(
            repo, checkpoint, device, log,
            mem_every=kwargs.get('mem_every', 3),
            max_mem_frames=kwargs.get('max_mem_frames', 10),
            top_k=kwargs.get('top_k', 50),
            use_long_term=kwargs.get('use_long_term', True),
            max_internal_size=kwargs.get('max_internal_size', -1)
        )

    # Fallback to legacy inference_utils approach
    sys.path.insert(0, str(repo))

    try:
        import importlib
        inference_utils = importlib.import_module("inference_utils")
    except Exception as exc:
        raise ImportError(
            "Unable to import MatAnyone inference utils. "
            "Update load_matanyone_adapter() with the correct import path."
        ) from exc

    load_model = getattr(inference_utils, "load_model", None)
    inference_one_image = getattr(inference_utils, "inference_one_image", None)

    if load_model is None or inference_one_image is None:
        raise ImportError(
            "MatAnyone inference utilities not found. "
            "Expected load_model() and inference_one_image() in inference_utils."
        )

    log.info(f"Loading MatAnyone checkpoint: {checkpoint}")

    try:
        model = load_model(str(checkpoint), device=device)
    except TypeError:
        model = load_model(str(checkpoint), device)

    return MatAnyoneAdapter(
        backend="mam2",
        model=model,
        infer_fn=inference_one_image,
        device=device,
    )


def infer_alpha(
    model: Any,
    infer_fn: Callable[[Any, np.ndarray, np.ndarray], Any],
    image: np.ndarray,
    guidance: np.ndarray
) -> np.ndarray:
    """
    Run inference to produce alpha matte.

    Args:
        model: Loaded model
        infer_fn: Inference function
        image: RGB image
        guidance: Guidance mask

    Returns:
        Alpha matte (H, W), float32 [0, 1]
    """
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        with torch.no_grad():
            alpha = infer_fn(model, image, guidance)
    else:
        alpha = infer_fn(model, image, guidance)

    return normalize_alpha(alpha)


def process_matanyone_v1_sequence(
    adapter: MatAnyoneAdapter,
    frame_files: List[Path],
    alpha_files: List[Path],
    alpha_output_dir: Path,
    warmup: int = 5,
    erode: int = 3,
    dilate: int = 5,
    output_format: str = "exr",
    bit_depth: int = 16,
    log: logging.Logger = None
) -> int:
    """
    Process a frame sequence using MatAnyone v1 InferenceCore.

    Args:
        adapter: MatAnyone adapter
        frame_files: List of frame file paths
        alpha_files: List of SAM mask file paths
        alpha_output_dir: Output directory for alpha mattes
        warmup: Number of warmup frames
        erode: Erosion iterations
        dilate: Dilation iterations
        output_format: Output format (exr, png, tiff)
        bit_depth: Output bit depth
        log: Logger instance

    Returns:
        0 on success, 1 on error
    """
    log = log or logger

    try:
        import torch
        from matanyone.utils.inference_utils import gen_dilate, gen_erosion
        from matanyone.utils.device import safe_autocast
    except Exception as exc:
        log.error(f"MatAnyone v1 dependencies unavailable: {exc}")
        return 1

    if not frame_files:
        log.error("No frames available for MatAnyone v1 inference.")
        return 1
    if not alpha_files:
        log.error("No SAM masks available for MatAnyone v1 inference.")
        return 1

    log.info("MatAnyone v1 uses only the first SAM mask as initialization.")

    # Load first mask
    from auto_roto.io.alpha import load_alpha
    mask_np = load_alpha(alpha_files[0])
    mask_uint8 = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)

    # Apply morphological operations
    if dilate > 0:
        mask_uint8 = gen_dilate(mask_uint8, dilate, dilate)
    if erode > 0:
        mask_uint8 = gen_erosion(mask_uint8, erode, erode)

    device = adapter.device or "cuda"
    processor = adapter.processor
    if processor is None:
        log.error("MatAnyone v1 processor not initialized.")
        return 1

    # Process sequence
    alpha_output_dir.mkdir(parents=True, exist_ok=True)

    # Load all frames
    frames = []
    for frame_path in frame_files:
        img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            log.warning(f"Cannot read frame: {frame_path}")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img.astype(np.float32))

    if not frames:
        log.error("No frames loaded.")
        return 1

    # Initialize processor with first frame and mask
    first_frame = frames[0]
    h, w = first_frame.shape[:2]

    # Process remaining frames
    from auto_roto.io.alpha import save_alpha

    # Object IDs for single object matting
    objects = [1]

    # Prepare mask tensor (2D: H, W)
    mask_tensor = torch.from_numpy(mask_uint8).float().to(device)
    mask_tensor = mask_tensor / 255.0 if mask_tensor.max() > 1.0 else mask_tensor

    log.info(f"Processing {len(frames)} frames with {warmup} warmup frames")

    for idx, frame in enumerate(frames):
        # Convert frame to tensor: (3, H, W) normalized to [0, 1]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(device)
        frame_tensor = frame_tensor / 255.0 if frame_tensor.max() > 1.0 else frame_tensor

        with safe_autocast():
            if idx == 0:
                # First frame: encode mask into memory
                processor.step(frame_tensor, mask_tensor, objects=objects)
                # Get first frame prediction
                out_prob = processor.step(frame_tensor, first_frame_pred=True)
            elif idx <= warmup:
                # Warmup frames: reinitialize as first frame prediction
                out_prob = processor.step(frame_tensor, first_frame_pred=True)
            else:
                # Normal propagation
                out_prob = processor.step(frame_tensor)

            # Convert output probability to mask using MatAnyone's method
            alpha_mask = processor.output_prob_to_mask(out_prob)

        # Convert to numpy
        alpha = alpha_mask.detach().cpu().numpy()
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        # Save output
        frame_name = frame_files[idx].stem
        if output_format == "exr":
            out_path = alpha_output_dir / f"{frame_name}.exr"
        else:
            out_path = alpha_output_dir / f"{frame_name}.{output_format}"

        save_alpha(out_path, alpha, bit_depth=bit_depth)

        if (idx + 1) % 10 == 0:
            log.info(f"Processed frame {idx + 1}/{len(frames)}")

    log.info(f"Processed {len(frames)} frames with MatAnyone v1")
    return 0


# Backward compatibility aliases
Mam2Adapter = MatAnyoneAdapter
load_mam2_adapter = load_matanyone_adapter
