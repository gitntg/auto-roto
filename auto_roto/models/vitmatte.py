"""
ViTMatte Model Wrapper
======================

Wrapper for ViTMatte (Vision Transformer for Image Matting).

ViTMatte is a Vision Transformer-based matting model that excels
at resolving fine detail (hair, fur, transparency) given a good trimap.

Models:
    - small: hustvl/vitmatte-small-composition-1k
    - base: hustvl/vitmatte-base-composition-1k
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("AutoRoto.Models.ViTMatte")


class ViTMatteRefiner:
    """
    Runs ViTMatte on RGB + Trimap to produce final alpha matte.

    ViTMatte is a Vision Transformer-based matting model that excels
    at resolving fine detail (hair, fur, transparency) given a good trimap.
    """

    MODEL_IDS = {
        'small': 'hustvl/vitmatte-small-composition-1k',
        'base': 'hustvl/vitmatte-base-composition-1k',
    }

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        max_resolution: int = 8192,
        logger: logging.Logger = None
    ):
        """
        Initialize ViTMatte refiner.

        Args:
            model_size: Model size ("small" or "base")
            device: Compute device (cuda, cpu)
            max_resolution: Maximum dimension for processing
            logger: Optional logger instance
        """
        self.model_size = model_size
        self.device = device
        self.max_resolution = max_resolution
        self.logger = logger or logging.getLogger("ViTMatte")

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load ViTMatte model from HuggingFace."""
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        model_id = self.MODEL_IDS.get(self.model_size, self.MODEL_IDS['base'])

        self.logger.info(f"Loading ViTMatte ({self.model_size}) from {model_id}...")

        self.processor = VitMatteImageProcessor.from_pretrained(model_id)
        self.model = VitMatteForImageMatting.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()

        self.logger.info("ViTMatte loaded successfully")

    def refine(
        self,
        rgb: np.ndarray,
        trimap: np.ndarray
    ) -> np.ndarray:
        """
        Run ViTMatte to produce alpha matte.

        Args:
            rgb: RGB image (H, W, 3), uint8 or float
            trimap: Trimap (H, W), uint8 with values 0, 128, 255

        Returns:
            Alpha matte (H, W), float32 0-1
        """
        import torch
        from PIL import Image
        import cv2

        # Ensure correct formats
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        if len(rgb.shape) == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        h_orig, w_orig = rgb.shape[:2]

        # Ensure trimap is single channel uint8
        if len(trimap.shape) == 3:
            trimap = trimap[:, :, 0]
        trimap = trimap.astype(np.uint8)

        # Resize for memory efficiency if needed
        max_dim = max(h_orig, w_orig)
        scale_factor = 1.0
        if max_dim > self.max_resolution:
            scale_factor = self.max_resolution / max_dim
            new_h = int(h_orig * scale_factor)
            new_w = int(w_orig * scale_factor)

            self.logger.info(f"Resizing {w_orig}x{h_orig} -> {new_w}x{new_h} for processing")

            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        h, w = rgb.shape[:2]

        # Convert to PIL
        rgb_pil = Image.fromarray(rgb)
        trimap_pil = Image.fromarray(trimap, mode='L')

        # Process inputs
        inputs = self.processor(
            images=rgb_pil,
            trimaps=trimap_pil,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        alpha = outputs.alphas

        # Handle size mismatch
        if alpha.shape[-2:] != (h, w):
            alpha = torch.nn.functional.interpolate(
                alpha,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

        alpha = alpha.squeeze().cpu().numpy()

        # Upscale back to original resolution
        if scale_factor < 1.0:
            self.logger.debug(f"Upscaling alpha back to {w_orig}x{h_orig}")
            alpha = cv2.resize(alpha, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return np.clip(alpha, 0, 1).astype(np.float32)

    def release(self):
        """Release GPU memory."""
        try:
            if self.model is not None:
                del self.model
                del self.processor
                self.model = None
                self.processor = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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


# Convenience function for creating ViTMatteRefiner from config
def create_vitmatte_refiner(config=None, **kwargs) -> ViTMatteRefiner:
    """
    Create ViTMatteRefiner from config or keyword arguments.

    Args:
        config: Optional ViTMatteConfig object
        **kwargs: Override any config values

    Returns:
        Configured ViTMatteRefiner instance
    """
    if config is not None:
        return ViTMatteRefiner(
            model_size=kwargs.get('model_size', config.model_size),
            device=kwargs.get('device', config.device),
            max_resolution=kwargs.get('max_resolution', config.max_resolution),
        )
    return ViTMatteRefiner(**kwargs)
