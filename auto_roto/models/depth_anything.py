"""
Depth Anything 3 Model Wrapper
==============================

Wrapper for Depth Anything 3 monocular depth estimation.

Outputs relative depth maps (not metric) which is perfect for
edge detection and alpha refinement.

Models:
    - small: DA3-Small (fastest, lower quality)
    - base: DA3-Base (balanced)
    - large: DA3Mono-Large (best for hair detail)
    - nested-large: DA3NESTED-GIANT-LARGE (higher resolution)
    - nested-base: DA3NESTED-Base (higher resolution, faster)
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("AutoRoto.Models.DepthAnything")

# Resolution constants
DEPTH_DEFAULT_TILE_SIZE = 2048
DEPTH_MIN_TILE_SIZE = 1024
DEPTH_MAX_PROCESS_RES = 8192


class DepthEstimator:
    """
    Wrapper for Depth Anything 3 monocular depth estimation.

    Outputs relative depth maps (not metric) which is perfect for
    edge detection and refinement.
    """

    MODEL_NAME_BY_SIZE = {
        "small": "depth-anything/DA3-Small",
        "base": "depth-anything/DA3-Base",
        "large": "depth-anything/DA3Mono-Large",
        "nested-large": "depth-anything/DA3NESTED-GIANT-LARGE",
        "nested-base": "depth-anything/DA3NESTED-Base",
    }

    MAX_PROCESS_RES = 8192

    def __init__(
        self,
        model_size: str = 'large',
        device: str = 'cuda',
        logger: logging.Logger = None,
        process_res: Optional[int] = None,
        process_method: str = "upper",
        norm_percentiles: Tuple[float, float] = (2.0, 98.0),
        use_confidence: bool = False
    ):
        """
        Initialize Depth Anything 3 estimator.

        Args:
            model_size: Model size (small, base, large, nested-large, nested-base)
            device: Compute device (cuda, cpu)
            logger: Optional logger instance
            process_res: Processing resolution (None = auto)
            process_method: Resize method ("upper" or "lower" bound)
            norm_percentiles: Percentiles for visualization normalization
            use_confidence: Extract confidence maps
        """
        self.model_size = model_size
        self.device = device
        self.logger = logger or logging.getLogger("DepthEstimator")

        self.process_res = process_res
        self.process_method = process_method
        self.norm_percentiles = norm_percentiles
        self.use_confidence = use_confidence

        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Depth Anything 3 model."""
        model_name = self.MODEL_NAME_BY_SIZE.get(
            self.model_size, self.MODEL_NAME_BY_SIZE["large"]
        )
        self.logger.info(f"Loading Depth Anything 3 ({model_name})...")

        # Add Depth-Anything-3 repo to path if it exists locally
        script_dir = Path(__file__).parent.parent.parent
        depth_repo = script_dir / "Depth-Anything-3"
        if depth_repo.exists() and str(depth_repo) not in sys.path:
            sys.path.insert(0, str(depth_repo))

        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()

        self.logger.info("Depth Anything 3 loaded successfully")

    def estimate(
        self,
        image: np.ndarray,
        process_res: int = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate depth from RGB image.

        Args:
            image: RGB image (H, W, 3), uint8 or float
            process_res: Processing resolution override

        Returns:
            Depth map (H, W), float32, normalized 0-1
            - Foreground (close) = LOW values
            - Background (far) = HIGH values

            If use_confidence is True, returns tuple: (depth, confidence)
        """
        import torch
        import cv2

        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        h, w = image.shape[:2]

        # Determine process_res
        if process_res is None:
            if self.process_res is not None:
                process_res = min(self.process_res, self.MAX_PROCESS_RES)
            else:
                process_res = min(max(h, w), self.MAX_PROCESS_RES)
        else:
            process_res = min(process_res, self.MAX_PROCESS_RES)

        process_method = f"{self.process_method}_bound_resize"

        self.logger.debug(
            f"Estimating depth at process_res={process_res}, method={process_method} "
            f"(image: {w}x{h})"
        )

        with torch.no_grad():
            prediction = self.model.inference(
                [image],
                process_res=process_res,
                process_res_method=process_method,
            )
        depth = prediction.depth[0]

        # Extract confidence if requested
        confidence = None
        if self.use_confidence and hasattr(prediction, 'conf') and prediction.conf is not None:
            confidence = prediction.conf[0]
            if confidence.shape != (h, w):
                confidence = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_LINEAR)
            confidence = confidence.astype(np.float32)

        # Resize depth to match input
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.use_confidence and confidence is not None:
            return depth.astype(np.float32), confidence
        return depth.astype(np.float32)

    def estimate_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Estimate depth for a batch of images."""
        return [self.estimate(img) for img in images]

    def estimate_high_res(
        self,
        image: np.ndarray,
        tile_size: int = None,
        overlap: int = None
    ) -> np.ndarray:
        """
        Estimate depth at higher resolution using tiled processing.

        Args:
            image: RGB image (H, W, 3), uint8
            tile_size: Size of each tile (default: 2048)
            overlap: Overlap between tiles (default: tile_size // 2)

        Returns:
            High-resolution depth map (H, W), float32
        """
        import cv2

        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        h, w = image.shape[:2]

        # Determine tile size
        if tile_size is None:
            if self.process_res is not None and self.process_res >= DEPTH_MIN_TILE_SIZE:
                tile_size = min(self.process_res, DEPTH_DEFAULT_TILE_SIZE)
            else:
                tile_size = DEPTH_DEFAULT_TILE_SIZE

        if overlap is None:
            overlap = tile_size // 2

        self.logger.debug(f"High-res depth: tile_size={tile_size}, overlap={overlap}")

        # If image is smaller than tile size, use regular estimation
        if h <= tile_size and w <= tile_size:
            return self.estimate(image)

        # Create output arrays
        depth_sum = np.zeros((h, w), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)

        def create_weight_mask(th, tw):
            """Create cosine-feathered weight mask."""
            feather = min(overlap, th // 2, tw // 2)
            if feather <= 0:
                return np.ones((th, tw), dtype=np.float32)

            def cosine_ramp(length, feather_size):
                ramp = np.ones(length, dtype=np.float32)
                if feather_size > 0 and length > 2 * feather_size:
                    t = np.linspace(0, np.pi / 2, feather_size)
                    fade_in = np.sin(t) ** 2
                    ramp[:feather_size] = fade_in
                    ramp[-feather_size:] = fade_in[::-1]
                return ramp

            ramp_h = cosine_ramp(th, feather)
            ramp_w = cosine_ramp(tw, feather)
            return np.outer(ramp_h, ramp_w).astype(np.float32)

        # Process tiles
        step = tile_size - overlap

        for y in range(0, h, step):
            for x in range(0, w, step):
                y1, y2 = y, min(y + tile_size, h)
                x1, x2 = x, min(x + tile_size, w)

                tile = image[y1:y2, x1:x2]
                tile_h, tile_w = tile.shape[:2]

                # Pad tile if at edge
                if tile_h < tile_size or tile_w < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:tile_h, :tile_w] = tile
                    if tile_h < tile_size:
                        mirror_h = min(tile_h, tile_size - tile_h)
                        padded[tile_h:tile_h + mirror_h, :tile_w] = tile[tile_h - mirror_h:tile_h, :][::-1]
                    if tile_w < tile_size:
                        mirror_w = min(tile_w, tile_size - tile_w)
                        padded[:tile_h, tile_w:tile_w + mirror_w] = tile[:, tile_w - mirror_w:tile_w][:, ::-1]
                    tile = padded

                # Estimate depth
                result = self.estimate(tile)
                tile_depth = result[0] if isinstance(result, tuple) else result

                if tile_depth.shape != (tile_size, tile_size):
                    tile_depth = cv2.resize(tile_depth, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

                tile_depth = tile_depth[:tile_h, :tile_w]
                weight = create_weight_mask(tile_h, tile_w)

                depth_sum[y1:y2, x1:x2] += tile_depth * weight
                weight_sum[y1:y2, x1:x2] += weight

        weight_sum = np.maximum(weight_sum, 1e-8)
        return (depth_sum / weight_sum).astype(np.float32)

    def release(self) -> None:
        """Release model and free GPU memory."""
        if self.model is not None:
            try:
                self.model = self.model.to('cpu')
            except Exception:
                pass
            del self.model
            self.model = None
            self.logger.debug("DepthEstimator model released")

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
