#!/usr/bin/env python
"""
HAIR-SPECIFIC ALPHA REFINEMENT
==============================

Uses ViTMatte (Vision Transformer Matting) to refine alpha mattes with
focus on fine details like hair, fur, and semi-transparent edges.

Unlike depth-based refinement, matting networks are specifically trained
to handle:
  - Individual hair strands
  - Semi-transparent regions
  - Fine edge details
  - Color-based boundary detection

WORKFLOW:
    1. Run auto_roto.py to get initial SAM2 masks
    2. Run hair_refine.py to enhance hair/fine detail edges

USAGE:
    # Basic refinement
    python hair_refine.py --alpha ./output/alpha/ --video video.mp4 --output ./refined/

    # With custom trimap dilation (larger = more refinement area)
    python hair_refine.py --alpha ./output/alpha/ --video video.mp4 --output ./refined/ --dilate 30

    # Save debug visualizations
    python hair_refine.py --alpha ./output/alpha/ --video video.mp4 --output ./refined/ --debug
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Generator, List
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)


# ==============================================================================
# TRIMAP GENERATION
# ==============================================================================

def generate_trimap(
    mask: np.ndarray,
    erode_size: int = 10,
    dilate_size: int = 20
) -> np.ndarray:
    """
    Generate a trimap from a binary/soft mask.

    Trimap has 3 regions:
        - 0: Definite background
        - 128: Unknown (where matting happens - includes hair edges)
        - 255: Definite foreground

    Args:
        mask: Input mask (H, W), values 0-1
        erode_size: Erosion kernel size for foreground
        dilate_size: Dilation kernel size for unknown region

    Returns:
        Trimap (H, W), values 0, 128, 255
    """
    import cv2

    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Threshold to binary
    _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    # Erode for definite foreground
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_size, erode_size)
    )
    foreground = cv2.erode(binary, erode_kernel, iterations=1)

    # Dilate for definite background boundary
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
    )
    dilated = cv2.dilate(binary, dilate_kernel, iterations=1)

    # Create trimap
    trimap = np.zeros_like(mask_uint8)
    trimap[dilated == 255] = 128  # Unknown region
    trimap[foreground == 255] = 255  # Definite foreground
    # Background stays 0

    return trimap


def generate_trimap_from_soft_mask(
    mask: np.ndarray,
    fg_threshold: float = 0.9,
    bg_threshold: float = 0.1,
    dilate_unknown: int = 15
) -> np.ndarray:
    """
    Generate trimap from a soft mask using thresholds.

    Better for masks that already have soft edges (like SAM2 output).

    Args:
        mask: Input soft mask (H, W), values 0-1
        fg_threshold: Values above this are definite foreground
        bg_threshold: Values below this are definite background
        dilate_unknown: Additional dilation for unknown region
    """
    import cv2

    trimap = np.full(mask.shape, 128, dtype=np.uint8)  # Start with unknown

    # Definite foreground and background
    trimap[mask >= fg_threshold] = 255
    trimap[mask <= bg_threshold] = 0

    # Optionally dilate the unknown region
    if dilate_unknown > 0:
        unknown_mask = (trimap == 128).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_unknown, dilate_unknown)
        )
        unknown_dilated = cv2.dilate(unknown_mask, kernel, iterations=1)

        # Only expand into foreground/background, don't shrink unknown
        trimap[unknown_dilated == 1] = 128

    return trimap


# ==============================================================================
# VITMATTE MATTING NETWORK
# ==============================================================================

class ViTMatteRefiner:
    """
    Hair/fine-detail alpha refinement using ViTMatte.

    ViTMatte is a Vision Transformer-based matting network that excels at:
    - Fine hair strands
    - Semi-transparent regions
    - Detailed edge refinement
    """

    MODEL_ID = "hustvl/vitmatte-small-composition-1k"

    def __init__(
        self,
        model_size: str = "small",
        device: str = None,
        logger: logging.Logger = None
    ):
        self.model_size = model_size
        self.logger = logger or logging.getLogger("ViTMatte")

        # Determine device
        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load ViTMatte model from HuggingFace."""
        import torch

        self.logger.info(f"Loading ViTMatte ({self.model_size})...")

        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            # Select model based on size
            if self.model_size == "small":
                model_id = "hustvl/vitmatte-small-composition-1k"
            elif self.model_size == "base":
                model_id = "hustvl/vitmatte-base-composition-1k"
            else:
                model_id = "hustvl/vitmatte-small-composition-1k"

            self.processor = VitMatteImageProcessor.from_pretrained(model_id)
            self.model = VitMatteForImageMatting.from_pretrained(model_id)
            self.model = self.model.to(self.device).eval()

            self.logger.info("ViTMatte loaded successfully")

        except ImportError as e:
            self.logger.error(f"Failed to import transformers: {e}")
            self.logger.error("Install with: pip install transformers")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load ViTMatte: {e}")
            raise

    def refine(
        self,
        image: np.ndarray,
        trimap: np.ndarray
    ) -> np.ndarray:
        """
        Refine alpha matte using ViTMatte.

        Args:
            image: RGB image (H, W, 3), values 0-255 uint8 or 0-1 float
            trimap: Trimap (H, W), values 0, 128, 255

        Returns:
            Refined alpha matte (H, W), values 0-1
        """
        import torch
        import cv2
        from PIL import Image

        # Store original dimensions
        orig_h, orig_w = image.shape[:2]

        # Convert image to PIL
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image

        pil_image = Image.fromarray(image_uint8)
        pil_trimap = Image.fromarray(trimap)

        # Process inputs
        inputs = self.processor(
            images=pil_image,
            trimaps=pil_trimap,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract alpha
        alpha = outputs.alphas[0, 0].cpu().numpy()

        # Resize back to original dimensions if needed (ViTMatte may pad input)
        if alpha.shape[0] != orig_h or alpha.shape[1] != orig_w:
            alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Ensure proper range
        alpha = np.clip(alpha, 0, 1).astype(np.float32)

        return alpha

    def refine_with_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        erode_size: int = 10,
        dilate_size: int = 25
    ) -> np.ndarray:
        """
        Convenience method: generate trimap from mask and refine.

        Args:
            image: RGB image
            mask: Soft mask from SAM2
            erode_size: Erosion for definite foreground
            dilate_size: Dilation for unknown region (larger = more hair capture)
        """
        trimap = generate_trimap(mask, erode_size, dilate_size)
        return self.refine(image, trimap)

    def release(self) -> None:
        """Release model and free GPU memory."""
        try:
            import torch
            import gc
            
            if self.model is not None:
                try:
                    self.model = self.model.to('cpu')
                except Exception:
                    pass
                del self.model
                del self.processor
                self.model = None
                self.processor = None
                self.logger.debug("ViTMatteRefiner model released")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except Exception:
            pass

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.release()
        return False


# ==============================================================================
# FILE I/O (reuse from depth_refine.py patterns)
# ==============================================================================

def _read_exr_alpha(filepath: Path) -> np.ndarray:
    """Read alpha channel from EXR file using OpenEXR module."""
    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(filepath))
        header = exr_file.header()

        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Try to read alpha channel, fall back to Y (luminance) or R
        channels = header['channels'].keys()

        if 'A' in channels:
            channel = 'A'
        elif 'Y' in channels:
            channel = 'Y'
        elif 'R' in channels:
            channel = 'R'
        else:
            channel = list(channels)[0]

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        data = exr_file.channel(channel, pt)

        alpha = np.frombuffer(data, dtype=np.float32).reshape((height, width))
        return alpha

    except Exception as e:
        print(f"Error reading EXR {filepath}: {e}")
        return None


def load_alpha_sequence(alpha_dir: str) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
    """Load alpha images from directory."""
    import cv2

    alpha_path = Path(alpha_dir)

    # Find all image files
    extensions = {'.exr', '.png', '.tif', '.tiff', '.jpg', '.jpeg'}
    files = []
    for ext in extensions:
        files.extend(alpha_path.glob(f"*{ext}"))
        files.extend(alpha_path.glob(f"*{ext.upper()}"))

    files = sorted(set(files))

    for idx, filepath in enumerate(files):
        # Read alpha - use OpenEXR for EXR files
        if filepath.suffix.lower() == '.exr':
            alpha = _read_exr_alpha(filepath)
        else:
            alpha = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)

        if alpha is None:
            continue

        # Handle different formats
        if len(alpha.shape) == 3:
            if alpha.shape[2] == 4:
                alpha = alpha[:, :, 3]
            else:
                alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)

        # Normalize to 0-1
        if alpha.dtype == np.uint8:
            alpha = alpha.astype(np.float32) / 255.0
        elif alpha.dtype == np.uint16:
            alpha = alpha.astype(np.float32) / 65535.0
        else:
            alpha = alpha.astype(np.float32)

        yield idx, alpha, filepath


def save_alpha(alpha: np.ndarray, path: Path, bit_depth: int = 16):
    """Save alpha matte to file."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()

    if ext == '.exr':
        try:
            import OpenEXR
            import Imath

            h, w = alpha.shape
            header = OpenEXR.Header(w, h)

            # Single channel alpha
            header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

            exr = OpenEXR.OutputFile(str(path), header)
            exr.writePixels({'Y': alpha.astype(np.float32).tobytes()})
            exr.close()
        except ImportError:
            # Fallback to PNG
            path = path.with_suffix('.png')
            alpha_uint16 = (alpha * 65535).astype(np.uint16)
            cv2.imwrite(str(path), alpha_uint16)
    else:
        if bit_depth == 16:
            alpha_int = (alpha * 65535).astype(np.uint16)
        else:
            alpha_int = (alpha * 255).astype(np.uint8)
        cv2.imwrite(str(path), alpha_int)


def extract_frames(video_path: str, output_dir: Path) -> int:
    """Extract frames from video."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = output_dir / f"frame.{idx:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        idx += 1

    cap.release()
    return idx


# ==============================================================================
# HAIR REFINEMENT PIPELINE
# ==============================================================================

class HairRefinementPipeline:
    """
    Pipeline for hair-specific alpha refinement using ViTMatte.
    """

    def __init__(
        self,
        alpha_dir: str,
        output_dir: str,
        video_path: str = None,
        frames_dir: str = None,
        model_size: str = "small",
        erode_size: int = 10,
        dilate_size: int = 25,
        output_format: str = "exr",
        bit_depth: int = 16,
        save_trimap: bool = False,
        save_comparison: bool = False,
        logger: logging.Logger = None
    ):
        self.alpha_dir = Path(alpha_dir)
        self.output_dir = Path(output_dir)
        self.video_path = video_path
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.model_size = model_size
        self.erode_size = erode_size
        self.dilate_size = dilate_size
        self.output_format = output_format
        self.bit_depth = bit_depth
        self.save_trimap = save_trimap
        self.save_comparison = save_comparison
        self.logger = logger or logging.getLogger("HairRefine")

        self._refiner = None

    @property
    def refiner(self) -> ViTMatteRefiner:
        """Lazy load refiner."""
        if self._refiner is None:
            self._refiner = ViTMatteRefiner(
                model_size=self.model_size,
                logger=self.logger
            )
        return self._refiner

    def run(self):
        """Run the hair refinement pipeline."""
        import cv2

        self.logger.info("=" * 60)
        self.logger.info("Hair-Specific Alpha Refinement (ViTMatte)")
        self.logger.info("=" * 60)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        refined_dir = self.output_dir / "refined"
        refined_dir.mkdir(exist_ok=True)

        if self.save_trimap:
            trimap_dir = self.output_dir / "trimap"
            trimap_dir.mkdir(exist_ok=True)

        if self.save_comparison:
            compare_dir = self.output_dir / "comparison"
            compare_dir.mkdir(exist_ok=True)

        # Get frames
        frames_path = None
        if self.frames_dir and self.frames_dir.exists():
            frames_path = self.frames_dir
            self.logger.info(f"Using existing frames from: {frames_path}")
        elif self.video_path:
            self.logger.info("Extracting frames from video...")
            frames_path = self.output_dir / "temp_frames"
            num_frames = extract_frames(self.video_path, frames_path)
            self.logger.info(f"Extracted {num_frames} frames")
        else:
            self.logger.error("Need frames_dir or video_path for matting")
            return

        # Load frame files
        frame_files = sorted(frames_path.glob("*.jpg")) + sorted(frames_path.glob("*.png"))

        # Process each alpha
        self.logger.info(f"Loading alpha mattes from: {self.alpha_dir}")
        self.logger.info(f"Trimap settings: erode={self.erode_size}, dilate={self.dilate_size}")

        processed = 0
        for idx, alpha, alpha_path in load_alpha_sequence(str(self.alpha_dir)):
            self.logger.info(f"Processing frame {idx}: {alpha_path.name}")

            # Find corresponding frame
            if idx < len(frame_files):
                frame = cv2.imread(str(frame_files[idx]))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                self.logger.warning(f"No frame for index {idx}, skipping")
                continue

            # Resize alpha to match frame if needed
            if alpha.shape[:2] != frame_rgb.shape[:2]:
                alpha = cv2.resize(alpha, (frame_rgb.shape[1], frame_rgb.shape[0]))

            # Generate trimap
            trimap = generate_trimap(alpha, self.erode_size, self.dilate_size)

            if self.save_trimap:
                trimap_path = trimap_dir / f"trimap.{idx:04d}.png"
                cv2.imwrite(str(trimap_path), trimap)

            # Refine with ViTMatte
            refined = self.refiner.refine(frame_rgb, trimap)

            # Save refined alpha
            output_path = refined_dir / f"refined.{idx:04d}.{self.output_format}"
            save_alpha(refined, output_path, self.bit_depth)

            # Save comparison
            if self.save_comparison:
                self._save_comparison(
                    frame_rgb, alpha, trimap, refined,
                    compare_dir / f"compare.{idx:04d}.jpg"
                )

            processed += 1
            if processed % 10 == 0:
                self.logger.info(f"  Processed {processed} frames...")

        self.logger.info("=" * 60)
        self.logger.info("Hair Refinement Complete!")
        self.logger.info(f"Output: {refined_dir}")
        self.logger.info("=" * 60)

    def _save_comparison(
        self,
        image: np.ndarray,
        original: np.ndarray,
        trimap: np.ndarray,
        refined: np.ndarray,
        path: Path
    ):
        """Save side-by-side comparison visualization."""
        import cv2

        h, w = image.shape[:2]

        # Resize for visualization
        scale = min(1.0, 1920 / (w * 4))
        new_w = int(w * scale)
        new_h = int(h * scale)

        def resize(img):
            return cv2.resize(img, (new_w, new_h))

        # Convert to BGR for display
        img_small = resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Alpha overlays
        original_overlay = img_small.copy()
        original_alpha = resize(original)
        original_overlay[:, :, 2] = np.clip(
            original_overlay[:, :, 2] + original_alpha * 100, 0, 255
        ).astype(np.uint8)

        trimap_vis = resize(cv2.cvtColor(trimap, cv2.COLOR_GRAY2BGR))

        refined_overlay = img_small.copy()
        refined_alpha = resize(refined)
        refined_overlay[:, :, 1] = np.clip(
            refined_overlay[:, :, 1] + refined_alpha * 100, 0, 255
        ).astype(np.uint8)

        # Combine
        comparison = np.hstack([
            original_overlay,
            trimap_vis,
            refined_overlay,
            resize((refined[..., None] * image).astype(np.uint8))
        ])

        cv2.imwrite(str(path), comparison)

    def release(self) -> None:
        """Release all resources and free GPU memory."""
        if self._refiner is not None:
            self._refiner.release()
            self._refiner = None
        
        # Clear GPU memory
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except ImportError:
            pass

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.release()
        return False


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hair-specific alpha refinement using ViTMatte",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic refinement
    python hair_refine.py --alpha ./output/alpha/ --video video.mp4 --output ./refined/

    # More aggressive hair capture (larger unknown region)
    python hair_refine.py --alpha ./output/alpha/ --video video.mp4 --output ./refined/ --dilate 40

    # Save debug visualizations
    python hair_refine.py --alpha ./output/alpha/ --video video.mp4 --output ./refined/ --debug
        """
    )

    # Required
    parser.add_argument(
        "--alpha", "-a",
        required=True,
        help="Directory containing alpha mattes to refine"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for refined mattes"
    )

    # Input source (one required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", "-v",
        help="Video file to extract frames from"
    )
    input_group.add_argument(
        "--frames", "-f",
        help="Directory containing extracted frames"
    )

    # Trimap settings
    parser.add_argument(
        "--erode", "-e",
        type=int,
        default=10,
        help="Erosion size for definite foreground (default: 10)"
    )
    parser.add_argument(
        "--dilate", "-d",
        type=int,
        default=25,
        help="Dilation size for unknown region - larger captures more hair (default: 25)"
    )

    # Model settings
    parser.add_argument(
        "--model-size",
        choices=["small", "base"],
        default="small",
        help="ViTMatte model size (default: small)"
    )

    # Output settings
    parser.add_argument(
        "--format",
        choices=["exr", "png", "tiff"],
        default="exr",
        help="Output format (default: exr)"
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=[8, 16],
        default=16,
        help="Bit depth for PNG/TIFF output (default: 16)"
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save trimap and comparison visualizations"
    )
    parser.add_argument(
        "--save-trimap",
        action="store_true",
        help="Save generated trimaps"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = HairRefinementPipeline(
        alpha_dir=args.alpha,
        output_dir=args.output,
        video_path=args.video,
        frames_dir=args.frames,
        model_size=args.model_size,
        erode_size=args.erode,
        dilate_size=args.dilate,
        output_format=args.format,
        bit_depth=args.bit_depth,
        save_trimap=args.save_trimap or args.debug,
        save_comparison=args.debug
    )

    pipeline.run()


if __name__ == "__main__":
    main()
