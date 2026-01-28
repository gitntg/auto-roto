#!/usr/bin/env python
"""
MATANYONE1 ALPHA REFINEMENT
===========================

Wrapper for MatAnyone v1 (Matting Anything 1) to refine SAM3 masks.
Legacy filename retained for pipeline compatibility.

USAGE:
    python mam2_refine.py --sam-mask ./sam/alpha --frames ./frames \
        --output ./matanyone1_output --checkpoint ./checkpoints/matanyone.pth

NOTES:
- This script expects a MatAnyone repo checkout (default: ./MatAnyone).
- The adapter uses MatAnyone v1 inference when the repo contains matanyone/.
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
    # Quality parameters
    mem_every: int = 3          # Memory frame interval (lower = better quality, slower)
    max_mem_frames: int = 10    # Max memory frames (higher = better quality, more VRAM)
    top_k: int = 50             # Top-k memory matching (higher = more accurate)
    use_long_term: bool = True  # Enable long-term memory for better temporal consistency
    max_internal_size: int = -1 # Max internal processing size (-1 = no resize, full resolution)


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
    return logging.getLogger("MatAnyone1Refine")


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


def load_rgb(path: Path, normalize: bool = True) -> np.ndarray:
    """Load RGB image.
    
    Args:
        path: Path to image file
        normalize: If True, return [0,1] range. If False, return [0,255] range.
                   MatAnyone expects [0,255] and normalizes internally.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read frame: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] > 3:
        image = image[:, :, :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 in [0, 255] range first
    if image.dtype == np.uint16:
        image = (image.astype(np.float32) / 65535.0) * 255.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32)  # Already 0-255
    else:
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image = image * 255.0
    
    if normalize:
        return np.clip(image / 255.0, 0.0, 1.0)
    else:
        return np.clip(image, 0.0, 255.0)


def load_mask(path: Path) -> np.ndarray:
    path_str = str(path)
    ext = path.suffix.lower()

    # Handle EXR files
    if ext == '.exr':
        # Try OpenEXR first
        try:
            import OpenEXR
            import Imath
            exr = OpenEXR.InputFile(path_str)
            header = exr.header()
            dw = header['dataWindow']
            h = dw.max.y - dw.min.y + 1
            w = dw.max.x - dw.min.x + 1
            
            # Try different channel names
            channels = list(header['channels'].keys())
            channel_name = None
            for name in ['A', 'Y', 'R', 'G', 'B']:
                if name in channels:
                    channel_name = name
                    break
            if channel_name is None and channels:
                channel_name = channels[0]
            
            if channel_name:
                pt = header['channels'][channel_name].type
                if pt == Imath.PixelType(Imath.PixelType.FLOAT):
                    dtype = np.float32
                elif pt == Imath.PixelType(Imath.PixelType.HALF):
                    dtype = np.float16
                else:
                    dtype = np.float32
                
                data = exr.channel(channel_name)
                mask = np.frombuffer(data, dtype=dtype).reshape(h, w).astype(np.float32)
                return np.clip(mask, 0.0, 1.0)
        except ImportError:
            pass  # OpenEXR not installed, try cv2
        except Exception as e:
            pass  # OpenEXR failed, try cv2
        
        # Try cv2 with EXR support (requires OPENCV_IO_ENABLE_OPENEXR=1)
        mask = cv2.imread(path_str, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if mask is not None:
            if mask.ndim == 3:
                # Try alpha channel first, then first channel
                if mask.shape[2] == 4:
                    mask = mask[:, :, 3]
                else:
                    mask = mask[:, :, 0]
            mask = mask.astype(np.float32)
            if mask.max() > 1.0:
                mask = mask / mask.max()
            return np.clip(mask, 0.0, 1.0)
        
        raise FileNotFoundError(
            f"Failed to read EXR mask: {path}. "
            f"Install OpenEXR: pip install OpenEXR"
        )

    # Fallback to cv2 for non-EXR
    mask = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
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
    logger: logging.Logger,
    config: Mam2Config = None
) -> Mam2Adapter:
    """Load MatAnyone v1 model and create inference adapter.
    
    This uses the official MatAnyone InferenceCore for video matting.
    Quality parameters from config are applied to the model.
    """
    # Ensure repo is in path
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

    logger.info("Detected MatAnyone v1 repo; using InferenceCore backend.")

    import torch
    
    # LOCAL checkpoint ONLY - no HuggingFace
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MatAnyone checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading MatAnyone from LOCAL checkpoint: {checkpoint_path}")
    
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict
    
    # Clear any existing Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    config_dir = str(repo_path / "matanyone" / "config")
    initialize_config_dir(version_base='1.3.2', config_dir=config_dir)
    cfg = compose(config_name="eval_matanyone_config")
    
    with open_dict(cfg):
        cfg['weights'] = str(checkpoint_path)
        # Apply quality parameters if config provided
        if config is not None:
            cfg['mem_every'] = config.mem_every
            cfg['max_mem_frames'] = config.max_mem_frames
            cfg['top_k'] = config.top_k
            cfg['use_long_term'] = config.use_long_term
            cfg['max_internal_size'] = config.max_internal_size
            if config.use_long_term:
                cfg['long_term']['max_mem_frames'] = config.max_mem_frames
            logger.info(f"Quality settings: mem_every={config.mem_every}, "
                       f"max_mem_frames={config.max_mem_frames}, "
                       f"top_k={config.top_k}, long_term={config.use_long_term}, "
                       f"max_internal_size={config.max_internal_size}")
    
    # Load model
    if device and device != "cpu":
        model = MatAnyone(cfg, single_object=True).to(device).eval()
        model_weights = torch.load(str(checkpoint_path), map_location=device)
    else:
        model = MatAnyone(cfg, single_object=True).cuda().eval()
        model_weights = torch.load(str(checkpoint_path))
    
    model.load_weights(model_weights)
    logger.info(f"✓ Loaded MatAnyone from local checkpoint")

    # Resolve device
    if not device or device == "auto":
        device = str(get_default_device())
    
    model = model.to(device).eval()
    
    # Create inference processor
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
    logger: logging.Logger,
    config: Mam2Config = None
) -> Mam2Adapter:
    if not repo_path.exists():
        raise FileNotFoundError(
            f"MatAnyone repo not found at {repo_path}. "
            "Clone it or pass --repo to point at the checkout."
        )

    if (repo_path / "matanyone").exists():
        return load_matanyone_v1_adapter(repo_path, checkpoint_path, device, logger, config)

    sys.path.insert(0, str(repo_path))

    try:
        inference_utils = importlib.import_module("inference_utils")
    except Exception as exc:
        raise ImportError(
            "Unable to import MatAnyone inference utils. "
            "Update load_mam2_adapter() with the correct import path."
        ) from exc

    load_model = getattr(inference_utils, "load_model", None)
    inference_one_image = getattr(inference_utils, "inference_one_image", None)

    if load_model is None or inference_one_image is None:
        raise ImportError(
            "MatAnyone inference utilities not found. "
            "Expected load_model() and inference_one_image() in inference_utils."
        )

    logger.info(f"Loading MatAnyone checkpoint: {checkpoint_path}")

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
    """Process a frame sequence using MatAnyone v1 InferenceCore.
    
    IMPORTANT: MatAnyone expects:
    - Images in [0, 255] range, then normalized to [0, 1] internally via (image / 255.)
    - First-frame mask in [0, 255] uint8 format
    - Uses warmup frames to stabilize temporal consistency
    """
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

    # Load first mask and convert to [0, 255] uint8 format
    mask_np = load_mask(alpha_files[0])
    mask_uint8 = np.clip(mask_np * 255.0, 0, 255).astype(np.uint8)

    # Apply morphological operations (expects uint8 input)
    if config.ma1_dilate > 0:
        mask_uint8 = gen_dilate(mask_uint8, config.ma1_dilate, config.ma1_dilate)
    if config.ma1_erode > 0:
        mask_uint8 = gen_erosion(mask_uint8, config.ma1_erode, config.ma1_erode)

    device = adapter.device or "cuda"
    processor = adapter.processor
    if processor is None:
        logger.error("MatAnyone v1 processor was not initialized.")
        return 1

    # Mask tensor stays in [0, 255] range - InferenceCore normalizes it
    mask_tensor = torch.from_numpy(mask_uint8).float().to(device)
    objects = [1]

    warmup = max(config.ma1_warmup, 0)
    frames_for_inference = [frame_files[0]] * warmup + frame_files

    logger.info(f"Processing {len(frame_files)} frames with MatAnyone v1 (warmup={warmup}).")

    # Use inference_mode for better performance and memory efficiency
    with torch.inference_mode():
        for ti, frame_path in enumerate(frames_for_inference):
            # CRITICAL: Load image in [0, 255] range (normalize=False)
            # MatAnyone normalizes internally: image = (image / 255.).float()
            rgb = load_rgb(frame_path, normalize=False)
            
            # Convert to tensor: HWC -> CHW, then normalize to [0, 1]
            image = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)
            image = image / 255.0  # Normalize to [0, 1] as MatAnyone expects

            with safe_autocast():
                if ti == 0:
                    # First call: encode the mask
                    output_prob = processor.step(image, mask_tensor, objects=objects)
                    # Second call: first frame prediction
                    output_prob = processor.step(image, first_frame_pred=True)
                else:
                    if ti <= warmup:
                        # Warmup frames: reinit as first frame
                        output_prob = processor.step(image, first_frame_pred=True)
                    else:
                        # Normal inference
                        output_prob = processor.step(image)

            # Skip warmup frames for output
            if ti < warmup:
                continue

            frame_idx = ti - warmup
            alpha = processor.output_prob_to_mask(output_prob)
            alpha_np = alpha.unsqueeze(2).detach().cpu().numpy()
            alpha_np = normalize_alpha(alpha_np)

            output_path = alpha_output_dir / f"ma1.{frame_idx:04d}.{config.output_format}"
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
            adapter = load_mam2_adapter(repo_path, checkpoint_path, device, logger, config)
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

        output_path = alpha_output_dir / f"ma1.{idx:04d}.{config.output_format}"
        save_alpha(alpha_pred, output_path, config.bit_depth)

        if idx % 10 == 0:
            logger.info(f"Processed frame {idx}")

    logger.info(f"MatAnyone1 refinement complete. Output: {output_dir}")
    return 0


def verify_matanyone_setup(
    repo_path: Path,
    checkpoint_path: Path,
    device: str,
    logger: logging.Logger
) -> bool:
    """Verify MatAnyone setup is correct and model can be loaded.
    
    Returns True if setup is valid, False otherwise.
    """
    logger.info("=" * 60)
    logger.info("MATANYONE SETUP VERIFICATION")
    logger.info("=" * 60)
    
    # Check repo path
    logger.info(f"Repo path: {repo_path}")
    if not repo_path.exists():
        logger.error(f"  ✗ Repo not found at {repo_path}")
        return False
    logger.info("  ✓ Repo directory exists")
    
    # Check for matanyone module
    matanyone_module = repo_path / "matanyone"
    if not matanyone_module.exists():
        logger.error(f"  ✗ matanyone/ module not found in {repo_path}")
        return False
    logger.info("  ✓ matanyone/ module found")
    
    # Check checkpoint
    logger.info(f"Checkpoint: {checkpoint_path}")
    if not checkpoint_path.exists():
        logger.warning(f"  ⚠ Checkpoint not found (will try HuggingFace)")
    else:
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✓ Checkpoint exists ({size_mb:.1f} MB)")
    
    # Check device
    logger.info(f"Device: {device}")
    try:
        import torch
        if device == "cuda":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"  ✓ CUDA available: {gpu_name}")
            else:
                logger.warning("  ⚠ CUDA not available, will use CPU")
        else:
            logger.info(f"  ✓ Using device: {device}")
    except ImportError:
        logger.error("  ✗ PyTorch not installed")
        return False
    
    # Try importing MatAnyone
    logger.info("Testing MatAnyone imports...")
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    
    try:
        from matanyone.inference.inference_core import InferenceCore
        from matanyone.model.matanyone import MatAnyone
        from matanyone.utils.device import get_default_device, safe_autocast
        from matanyone.utils.inference_utils import gen_dilate, gen_erosion
        logger.info("  ✓ All MatAnyone modules imported successfully")
    except ImportError as e:
        logger.error(f"  ✗ Import failed: {e}")
        return False
    
    # Try loading the model
    logger.info("Testing model loading...")
    try:
        # First try HuggingFace
        try:
            model = MatAnyone.from_pretrained("PeiqingYang/MatAnyone")
            logger.info("  ✓ Model loaded from HuggingFace")
        except Exception as hf_exc:
            logger.debug(f"  HuggingFace failed: {hf_exc}")
            if checkpoint_path.exists():
                logger.info("  Trying local checkpoint...")
                # This will use Hydra config
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
                
                model = MatAnyone(cfg, single_object=True)
                model_weights = torch.load(str(checkpoint_path), map_location='cpu')
                model.load_weights(model_weights)
                logger.info("  ✓ Model loaded from local checkpoint")
            else:
                raise hf_exc
        
        # Move to device and create processor
        resolved_device = device if device != "auto" else str(get_default_device())
        model = model.to(resolved_device).eval()
        processor = InferenceCore(model, cfg=model.cfg, device=resolved_device)
        logger.info(f"  ✓ InferenceCore created on {resolved_device}")
        
        # Clean up
        del processor
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        logger.error(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("=" * 60)
    logger.info("✓ MATANYONE SETUP VERIFIED SUCCESSFULLY")
    logger.info("=" * 60)
    return True


def parse_args() -> Mam2Config:
    parser = argparse.ArgumentParser(
        description="MatAnyone1-based alpha refinement wrapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process frames with SAM masks
  %(prog)s --sam-mask ./output/alpha --frames ./frames --output ./output/03_matanyone1_output
  
  # With explicit checkpoint and repo paths
  %(prog)s --sam-mask ./output/alpha --frames ./frames --output ./output/03_matanyone1_output \\
      --checkpoint ./checkpoints/matanyone.pth --repo ./MatAnyone
  
  # Verify setup only (no processing)
  %(prog)s --verify --repo ./MatAnyone --checkpoint ./checkpoints/matanyone.pth
        """
    )

    parser.add_argument("--sam-mask", help="Path to SAM alpha masks (required unless --verify)")
    parser.add_argument("--frames", help="Path to RGB frames (required unless --verify)")
    parser.add_argument("--output", help="Output directory (required unless --verify)")
    parser.add_argument("--checkpoint", default="./checkpoints/matanyone.pth", help="MatAnyone checkpoint path")
    parser.add_argument("--repo", default="./MatAnyone", help="MatAnyone repo path")
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
    # Quality parameters
    parser.add_argument("--mem-every", type=int, default=3,
                        help="Memory frame interval - lower = better quality, slower (default: 3)")
    parser.add_argument("--max-mem-frames", type=int, default=10,
                        help="Max memory frames - higher = better quality, more VRAM (default: 10)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k memory matching - higher = more accurate (default: 50)")
    parser.add_argument("--use-long-term", action="store_true", default=True,
                        help="Enable long-term memory for better temporal consistency (default: True)")
    parser.add_argument("--no-long-term", action="store_false", dest="use_long_term",
                        help="Disable long-term memory")
    parser.add_argument("--max-internal-size", type=int, default=-1,
                        help="Max internal processing size (-1 = full resolution, no downscaling)")
    parser.add_argument("--allow-fallback", action="store_true",
                        help="Allow pass-through output if MatAnyone import fails")
    parser.add_argument("--verify", action="store_true",
                        help="Verify MatAnyone setup only (no processing)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Handle verify mode - doesn't need sam-mask, frames, output
    if args.verify:
        return args, True  # Return args and verify flag
    
    # Validate required args for processing mode
    if not args.sam_mask or not args.frames or not args.output:
        parser.error("--sam-mask, --frames, and --output are required unless using --verify")

    config = Mam2Config(
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
        # Quality parameters
        mem_every=args.mem_every,
        max_mem_frames=args.max_mem_frames,
        top_k=args.top_k,
        use_long_term=args.use_long_term,
        max_internal_size=args.max_internal_size,
    )
    return config, False


def main():
    result = parse_args()
    
    # Handle verify mode
    if isinstance(result, tuple):
        config_or_args, is_verify = result
        if is_verify:
            args = config_or_args
            logger = setup_logging(args.verbose)
            repo_path = resolve_repo_path(args.repo)
            checkpoint_path = resolve_checkpoint_path(args.checkpoint)
            success = verify_matanyone_setup(repo_path, checkpoint_path, args.device, logger)
            sys.exit(0 if success else 1)
        else:
            config = config_or_args
    else:
        config = result
    
    logger = setup_logging(config.verbose)
    sys.exit(process_sequence(config, logger))


if __name__ == "__main__":
    main()
