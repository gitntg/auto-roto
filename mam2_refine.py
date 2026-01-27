#!/usr/bin/env python
"""
MAM2-BASED ALPHA REFINEMENT
===========================

Wrapper for MatAnyone2 (Matting Anything 2) to refine SAM3 masks,
with a temporary MatAnyone v1 backend until MatAnyone2 code is released.

USAGE:
    python mam2_refine.py --sam-mask ./sam/alpha --frames ./frames \
        --output ./mam2_output --checkpoint ./checkpoints/mam2.pth

NOTES:
- This script expects a MatAnyone2 repo checkout (default: ./MatAnyone2).
- If you pass --repo ./MatAnyone (v1), it uses MatAnyone v1 inference core.
- The import path may vary by repo version. Update `load_mam2_adapter()` if needed.
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import sys
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Any
import importlib

import cv2
import numpy as np


@dataclass
class Mam2Config:
    sam_mask: str
    frames: str
    output: str
    checkpoint: str
    repo: str
    device: str
    output_format: str
    bit_depth: int
    mask_threshold: float
    allow_fallback: bool
    ma1_warmup: int
    ma1_erode: int
    ma1_dilate: int
    verbose: bool


@dataclass
class Mam2Adapter:
    backend: str
    model: Optional[Any] = None
    infer_fn: Optional[Callable[[Any, np.ndarray, np.ndarray], Any]] = None
    processor: Optional[Any] = None
    device: Optional[str] = None


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("Mam2Refine")


def resolve_repo_path(repo_path: str) -> Path:
    path = Path(repo_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).parent / path).resolve()


def resolve_checkpoint_path(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).parent / path).resolve()


def resolve_device(device: str, logger: logging.Logger) -> str:
    if not device:
        return device
    try:
        import torch
    except ImportError:
        return device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        return "cpu"
    return device


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read frame: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] > 3:
        image = image[:, :, :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0] if mask.shape[2] == 1 else mask.mean(axis=2)
    if mask.dtype == np.uint16:
        mask = mask.astype(np.float32) / 65535.0
    elif mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = mask.astype(np.float32)
        max_val = float(mask.max()) if mask.size else 1.0
        if max_val > 1.0:
            denom = max_val if max_val > 255.0 else 255.0
            mask = mask / denom
    return np.clip(mask, 0.0, 1.0)


def normalize_alpha(alpha: Any) -> np.ndarray:
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
            denom = max_val if max_val > 0 else 1.0
            alpha = alpha / denom

    return np.clip(alpha, 0.0, 1.0)


def save_alpha(alpha: np.ndarray, path: Path, bit_depth: int = 16):
    ext = path.suffix.lower()

    if ext == '.exr':
        try:
            import OpenEXR
            import Imath

            h, w = alpha.shape
            header = OpenEXR.Header(w, h)

            if bit_depth == 32:
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                data = alpha.astype(np.float32)
            else:
                pixel_type = Imath.PixelType(Imath.PixelType.HALF)
                data = alpha.astype(np.float16)

            header['channels'] = {'A': Imath.Channel(pixel_type)}

            exr = OpenEXR.OutputFile(str(path), header)
            exr.writePixels({'A': data.tobytes()})
            exr.close()
            return

        except ImportError:
            pass

        if cv2.imwrite(str(path), alpha.astype(np.float32)):
            return

        path = path.with_suffix('.png')

    if bit_depth == 16:
        alpha_int = (alpha * 65535).astype(np.uint16)
    else:
        alpha_int = (alpha * 255).astype(np.uint8)

    cv2.imwrite(str(path), alpha_int)


def build_file_list(directory: Path, extensions: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def load_matanyone_v1_adapter(
    repo_path: Path,
    checkpoint_path: Path,
    device: str,
    logger: logging.Logger
) -> Mam2Adapter:
    sys.path.insert(0, str(repo_path))

    try:
        from matanyone.utils.get_default_model import get_matanyone_model
        from matanyone.inference.inference_core import InferenceCore
    except Exception as exc:
        raise ImportError(
            "Unable to import MatAnyone v1 modules. "
            "Ensure the repo contains matanyone/ and its dependencies."
        ) from exc

    logger.info("Detected MatAnyone v1 repo; using InferenceCore backend.")
    logger.info(f"Loading MatAnyone v1 checkpoint: {checkpoint_path}")

    model = get_matanyone_model(str(checkpoint_path), device=device)
    processor = InferenceCore(model, cfg=model.cfg, device=device)

    return Mam2Adapter(
        backend="matanyone_v1",
        model=model,
        processor=processor,
        device=device,
    )


def load_mam2_adapter(
    repo_path: Path,
    checkpoint_path: Path,
    device: str,
    logger: logging.Logger
) -> Mam2Adapter:
    if not repo_path.exists():
        raise FileNotFoundError(
            f"MatAnyone2 repo not found at {repo_path}. "
            "Clone it or pass --mam2-repo to point at the checkout."
        )

    if (repo_path / "matanyone").exists():
        return load_matanyone_v1_adapter(repo_path, checkpoint_path, device, logger)

    sys.path.insert(0, str(repo_path))

    try:
        inference_utils = importlib.import_module("inference_utils")
    except Exception as exc:
        raise ImportError(
            "Unable to import MatAnyone2 inference_utils. "
            "Update load_mam2_adapter() with the correct import path."
        ) from exc

    load_model = getattr(inference_utils, "load_model", None)
    inference_one_image = getattr(inference_utils, "inference_one_image", None)

    if load_model is None or inference_one_image is None:
        raise ImportError(
            "MatAnyone2 inference utilities not found. "
            "Expected load_model() and inference_one_image() in inference_utils."
        )

    logger.info(f"Loading MatAnyone2 checkpoint: {checkpoint_path}")

    try:
        model = load_model(str(checkpoint_path), device=device)
    except TypeError:
        model = load_model(str(checkpoint_path), device)

    return Mam2Adapter(
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
    adapter: Mam2Adapter,
    frame_files: List[Path],
    alpha_files: List[Path],
    alpha_output_dir: Path,
    config: Mam2Config,
    logger: logging.Logger,
) -> int:
    try:
        import torch
        from matanyone.utils.inference_utils import gen_dilate, gen_erosion
        from matanyone.utils.device import safe_autocast
    except Exception as exc:
        logger.error(f"MatAnyone v1 dependencies unavailable: {exc}")
        return 1

    if not frame_files:
        logger.error("No frames available for MatAnyone v1 inference.")
        return 1
    if not alpha_files:
        logger.error("No SAM masks available for MatAnyone v1 inference.")
        return 1

    logger.info("MatAnyone v1 uses only the first SAM mask as initialization.")

    mask_np = load_mask(alpha_files[0])
    mask_uint8 = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)

    if config.ma1_dilate > 0:
        mask_uint8 = gen_dilate(mask_uint8, config.ma1_dilate, config.ma1_dilate)
    if config.ma1_erode > 0:
        mask_uint8 = gen_erosion(mask_uint8, config.ma1_erode, config.ma1_erode)

    device = adapter.device or "cuda"
    processor = adapter.processor
    if processor is None:
        logger.error("MatAnyone v1 processor was not initialized.")
        return 1

    mask_tensor = torch.from_numpy(mask_uint8).float().to(device)
    objects = [1]

    warmup = max(config.ma1_warmup, 0)
    frames_for_inference = [frame_files[0]] * warmup + frame_files

    logger.info(f"Processing {len(frame_files)} frames with MatAnyone v1 (warmup={warmup}).")

    for ti, frame_path in enumerate(frames_for_inference):
        rgb = load_rgb(frame_path)
        image = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)

        with safe_autocast():
            if ti == 0:
                output_prob = processor.step(image, mask_tensor, objects=objects)
                output_prob = processor.step(image, first_frame_pred=True)
            else:
                if ti <= warmup:
                    output_prob = processor.step(image, first_frame_pred=True)
                else:
                    output_prob = processor.step(image)

        if ti < warmup:
            continue

        frame_idx = ti - warmup
        alpha = processor.output_prob_to_mask(output_prob)
        alpha_np = alpha.unsqueeze(2).detach().cpu().numpy()
        alpha_np = normalize_alpha(alpha_np)

        output_path = alpha_output_dir / f"mam2.{frame_idx:04d}.{config.output_format}"
        save_alpha(alpha_np, output_path, config.bit_depth)

        if frame_idx % 10 == 0:
            logger.info(f"Processed frame {frame_idx}")

    logger.info("MatAnyone v1 refinement complete.")
    return 0


def process_sequence(config: Mam2Config, logger: logging.Logger) -> int:
    alpha_dir = Path(config.sam_mask)
    frames_dir = Path(config.frames)
    output_dir = Path(config.output)

    output_dir.mkdir(parents=True, exist_ok=True)
    alpha_output_dir = output_dir / "alpha"
    alpha_output_dir.mkdir(exist_ok=True)

    alpha_files = build_file_list(alpha_dir, (".exr", ".png", ".tif", ".tiff"))
    frame_files = build_file_list(frames_dir, (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"))

    if not alpha_files:
        logger.error(f"No SAM masks found in {alpha_dir}")
        return 1
    if not frame_files:
        logger.error(f"No frames found in {frames_dir}")
        return 1

    if len(alpha_files) != len(frame_files):
        logger.warning(
            "Mask/frame count mismatch: "
            f"{len(alpha_files)} masks vs {len(frame_files)} frames. "
            "Processing aligned pairs by index."
        )

    repo_path = resolve_repo_path(config.repo)
    checkpoint_path = resolve_checkpoint_path(config.checkpoint)
    device = resolve_device(config.device, logger)

    adapter: Optional[Mam2Adapter] = None
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        if not config.allow_fallback:
            return 1
        logger.warning("Falling back to SAM masks (pass-through).")
    else:
        try:
            adapter = load_mam2_adapter(repo_path, checkpoint_path, device, logger)
        except Exception as exc:
            logger.error(str(exc))
            if not config.allow_fallback:
                return 1
            logger.warning("Falling back to SAM masks (pass-through).")

    if adapter and adapter.backend == "matanyone_v1":
        return process_matanyone_v1_sequence(
            adapter,
            frame_files,
            alpha_files,
            alpha_output_dir,
            config,
            logger,
        )

    model = adapter.model if adapter else None
    infer_fn = adapter.infer_fn if adapter else None

    logger.info(f"Processing {min(len(alpha_files), len(frame_files))} frames...")

    for idx, (alpha_path, frame_path) in enumerate(zip(alpha_files, frame_files)):
        rgb = load_rgb(frame_path)
        sam_mask = load_mask(alpha_path)
        guidance = (sam_mask > config.mask_threshold).astype(np.float32)

        if model is None or infer_fn is None:
            alpha_pred = sam_mask
        else:
            alpha_pred = infer_alpha(model, infer_fn, rgb, guidance)

        output_path = alpha_output_dir / f"mam2.{idx:04d}.{config.output_format}"
        save_alpha(alpha_pred, output_path, config.bit_depth)

        if idx % 10 == 0:
            logger.info(f"Processed frame {idx}")

    logger.info(f"MatAnyone2 refinement complete. Output: {output_dir}")
    return 0


def parse_args() -> Mam2Config:
    parser = argparse.ArgumentParser(
        description="MatAnyone2-based alpha refinement wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  %(prog)s --sam-mask ./output/alpha --frames ./frames --output ./output/03_mam2_output
  %(prog)s --sam-mask ./output/alpha --frames ./frames --output ./output/03_mam2_output \
      --checkpoint ./checkpoints/mam2.pth --repo ./MatAnyone2
        """
    )

    parser.add_argument("--sam-mask", required=True, help="Path to SAM alpha masks")
    parser.add_argument("--frames", required=True, help="Path to RGB frames")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--checkpoint", default="./checkpoints/mam2.pth", help="MatAnyone2 checkpoint path")
    parser.add_argument("--repo", default="MatAnyone2", help="MatAnyone2 repo path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--format", default="exr", choices=["exr", "png", "tiff"],
                        help="Output format")
    parser.add_argument("--bit-depth", type=int, default=16, choices=[8, 16, 32],
                        help="Output bit depth")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="Threshold for SAM guidance mask binarization")
    parser.add_argument("--ma1-warmup", type=int, default=10,
                        help="MatAnyone v1 warmup frames (default: 10)")
    parser.add_argument("--ma1-erode", type=int, default=0,
                        help="MatAnyone v1 erosion radius for first mask (default: 0)")
    parser.add_argument("--ma1-dilate", type=int, default=0,
                        help="MatAnyone v1 dilation radius for first mask (default: 0)")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Allow pass-through output if MatAnyone2 import fails")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    return Mam2Config(
        sam_mask=args.sam_mask,
        frames=args.frames,
        output=args.output,
        checkpoint=args.checkpoint,
        repo=args.repo,
        device=args.device,
        output_format=args.format,
        bit_depth=args.bit_depth,
        mask_threshold=args.mask_threshold,
        ma1_warmup=args.ma1_warmup,
        ma1_erode=args.ma1_erode,
        ma1_dilate=args.ma1_dilate,
        allow_fallback=args.allow_fallback,
        verbose=args.verbose,
    )


def main():
    config = parse_args()
    logger = setup_logging(config.verbose)
    sys.exit(process_sequence(config, logger))


if __name__ == "__main__":
    main()
