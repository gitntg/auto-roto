"""
SAM3 Segmenter
==============

Wrapper for SAM3 (Segment Anything Model 3) with built-in text prompting.

SAM3 has native Promptable Concept Segmentation (PCS) supporting 270k+ concepts,
eliminating the need for separate detection (GroundingDINO).

Requirements:
    - ultralytics >= 8.3.237
    - sam3.pt checkpoint
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger("AutoRoto.Models.SAM3")

# SAM3 inference resolution limits
SAM3_DEFAULT_IMGSZ = 1024
SAM3_MAX_AUTO_IMGSZ = 2048
SAM3_MIN_IMGSZ = 640


class SAM3Segmenter:
    """
    SAM3 segmenter with built-in text prompting via Ultralytics.

    SAM3 (released November 2025) has native Promptable Concept Segmentation (PCS)
    which allows direct text-to-segmentation without needing Grounding DINO.

    Features:
        - Built-in text prompting (no Grounding DINO needed)
        - Video tracking with temporal consistency
        - 270k+ trained concepts
        - Single-step inference pipeline

    Requirements:
        - ultralytics >= 8.3.237
        - sam3.pt checkpoint (download from HuggingFace after approval)
    """

    def __init__(
        self,
        model_path: str = "sam3.pt",
        device: str = "cuda",
        compile_model: bool = True,
        half_precision: bool = True,
        logger: logging.Logger = None,
        imgsz: int = 1024,
        conf: float = 0.25,
        retina_masks: bool = True,
        max_det: int = 100
    ):
        """
        Initialize SAM3 segmenter.

        Args:
            model_path: Path to sam3.pt checkpoint
            device: Compute device (cuda, cpu)
            compile_model: Use torch.compile for speed
            half_precision: Use FP16 inference
            logger: Optional logger instance
            imgsz: Processing resolution (0 = auto from input)
            conf: Confidence threshold (0.0-1.0)
            retina_masks: High-resolution mask output
            max_det: Maximum detections per frame
        """
        self.model_path = model_path
        self.device = device
        self.compile_model = compile_model
        self.half_precision = half_precision
        self.logger = logger or logging.getLogger("SAM3")

        # Inference settings
        self._imgsz_setting = imgsz
        self._resolved_imgsz = None
        self.conf = conf
        self.retina_masks = retina_masks
        self.max_det = max_det

        # Predictors (lazy loaded)
        self._image_predictor = None
        self._video_predictor = None

        self._verify_checkpoint()

    def _resolve_imgsz(self, image_shape: tuple) -> int:
        """Resolve imgsz from image dimensions."""
        if self._imgsz_setting > 0:
            return self._imgsz_setting

        h, w = image_shape[:2]
        max_dim = max(h, w)
        resolved = min(max_dim, SAM3_MAX_AUTO_IMGSZ)

        self.logger.info(f"Auto imgsz: input={w}x{h}, using imgsz={resolved}")
        return resolved

    @property
    def imgsz(self) -> int:
        """Get resolved imgsz (may be 0 if not yet resolved)."""
        return self._resolved_imgsz if self._resolved_imgsz else self._imgsz_setting

    def _verify_checkpoint(self):
        """Verify SAM3 checkpoint exists."""
        possible_paths = [
            Path(self.model_path),
            Path("checkpoints/sam3") / Path(self.model_path).name,
            Path("checkpoints") / self.model_path,
            Path.home() / ".cache" / "sam3" / self.model_path,
        ]

        for p in possible_paths:
            if p.exists():
                self.model_path = str(p)
                self.logger.info(f"Found SAM3 checkpoint: {p}")
                return

        self.logger.warning(
            f"SAM3 checkpoint '{self.model_path}' not found.\n"
            "Download from HuggingFace (requires approval):\n"
            "  1. Request access at https://huggingface.co/facebook/sam3\n"
            "  2. Download sam3.pt after approval\n"
            "  3. Place in working directory or checkpoints/sam3/"
        )

    @property
    def image_predictor(self):
        """Lazy-load image predictor."""
        if self._image_predictor is None:
            self.logger.info("Loading SAM3 image predictor...")
            self.logger.info(f"  imgsz={self.imgsz}, conf={self.conf}, retina_masks={self.retina_masks}")
            try:
                from ultralytics.models.sam import SAM3SemanticPredictor

                overrides = dict(
                    conf=self.conf,
                    imgsz=self.imgsz,
                    retina_masks=self.retina_masks,
                    max_det=self.max_det,
                    task="segment",
                    mode="predict",
                    model=self.model_path,
                    half=self.half_precision,
                    device=self.device,
                )
                self._image_predictor = SAM3SemanticPredictor(overrides=overrides)
                self.logger.info("SAM3 image predictor loaded")
            except ImportError as e:
                raise ImportError(
                    "SAM3 requires ultralytics >= 8.3.237. Install with:\n"
                    "  pip install -U ultralytics>=8.3.237"
                ) from e
        return self._image_predictor

    @property
    def video_predictor(self):
        """Lazy-load video predictor."""
        if self._video_predictor is None:
            self.logger.info("Loading SAM3 video predictor...")
            self.logger.info(f"  imgsz={self.imgsz}, conf={self.conf}, retina_masks={self.retina_masks}")
            try:
                from ultralytics.models.sam import SAM3VideoSemanticPredictor

                overrides = dict(
                    conf=self.conf,
                    imgsz=self.imgsz,
                    retina_masks=self.retina_masks,
                    max_det=self.max_det,
                    task="segment",
                    mode="predict",
                    model=self.model_path,
                    half=self.half_precision,
                    device=self.device,
                    save=False,
                )
                self._video_predictor = SAM3VideoSemanticPredictor(overrides=overrides)
                self.logger.info("SAM3 video predictor loaded")
            except ImportError as e:
                raise ImportError(
                    "SAM3 requires ultralytics >= 8.3.237. Install with:\n"
                    "  pip install -U ultralytics>=8.3.237"
                ) from e
        return self._video_predictor

    def segment_image_with_text(
        self,
        image: np.ndarray,
        text_prompts: List[str]
    ) -> List[np.ndarray]:
        """
        Segment image using text prompts.

        Args:
            image: RGB image (numpy array)
            text_prompts: List of text descriptions (e.g., ["person", "dog"])

        Returns:
            List of binary masks for each detected concept
        """
        self.logger.info(f"Segmenting with text prompts: {text_prompts}")

        if self._resolved_imgsz is None:
            self._resolved_imgsz = self._resolve_imgsz(image.shape)

        self.image_predictor.set_image(image)
        results = self.image_predictor(text=text_prompts)

        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    masks.append(mask.cpu().numpy())

        self.logger.info(f"Found {len(masks)} masks")
        return masks

    def segment_image_with_box(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:
        """
        Segment image using bounding box prompts.

        Args:
            image: RGB image (numpy array)
            boxes: List of bounding boxes as (x1, y1, x2, y2)

        Returns:
            List of binary masks for each box
        """
        self.logger.info(f"Segmenting with {len(boxes)} box prompts")

        self.image_predictor.set_image(image)
        bboxes = np.array(boxes, dtype=np.float32)
        results = self.image_predictor(bboxes=bboxes)

        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    masks.append(mask.cpu().numpy())

        self.logger.info(f"Found {len(masks)} masks from box prompts")
        return masks

    def segment_image_with_points(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        labels: List[int]
    ) -> List[np.ndarray]:
        """
        Segment image using point prompts.

        Args:
            image: RGB image (numpy array)
            points: List of (x, y) coordinates
            labels: List of labels (1 = foreground, 0 = background)

        Returns:
            List of binary masks
        """
        self.logger.info(f"Segmenting with {len(points)} point prompts")

        self.image_predictor.set_image(image)
        points_np = np.array(points, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int32)
        results = self.image_predictor(points=points_np, labels=labels_np)

        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    masks.append(mask.cpu().numpy())

        self.logger.info(f"Found {len(masks)} masks from point prompts")
        return masks

    def segment_image_with_text_and_points(
        self,
        image: np.ndarray,
        text_prompts: List[str],
        points: List[Tuple[int, int]],
        labels: List[int]
    ) -> List[np.ndarray]:
        """
        Segment image using BOTH text prompts AND point prompts together.

        This allows refinement of text-based segmentation with include/exclude points.
        Points with label=1 indicate foreground (include), label=0 indicate background (exclude).

        Args:
            image: RGB image (numpy array)
            text_prompts: List of text descriptions (e.g., ["person", "dog"])
            points: List of (x, y) coordinates for refinement
            labels: List of labels (1 = foreground/include, 0 = background/exclude)

        Returns:
            List of binary masks refined by both text and point prompts
        """
        self.logger.info(f"Segmenting with text prompts: {text_prompts}")
        self.logger.info(f"  + {len(points)} refinement points ({sum(labels)} include, {len(labels) - sum(labels)} exclude)")

        if self._resolved_imgsz is None:
            self._resolved_imgsz = self._resolve_imgsz(image.shape)

        self.image_predictor.set_image(image)

        # Convert points and labels to numpy arrays
        points_np = np.array(points, dtype=np.float32) if points else None
        labels_np = np.array(labels, dtype=np.int32) if labels else None

        # Call predictor with both text and points
        results = self.image_predictor(
            text=text_prompts,
            points=points_np,
            labels=labels_np
        )

        masks = []
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    masks.append(mask.cpu().numpy())

        self.logger.info(f"Found {len(masks)} masks from combined text+point prompts")
        return masks

    def segment_video_with_box(
        self,
        video_path: str,
        boxes: List[Tuple[int, int, int, int]],
        output_masks: bool = True
    ):
        """
        Segment video using bounding box prompts with temporal tracking.

        Args:
            video_path: Path to video file or frames directory
            boxes: List of bounding boxes as (x1, y1, x2, y2)
            output_masks: Whether to yield masks for each frame

        Yields:
            (frame_idx, masks) tuples
        """
        self.logger.info(f"Processing video with box prompts: {video_path}")
        self.logger.info(f"Boxes: {boxes}")

        bboxes = np.array(boxes, dtype=np.float32)
        results = self.video_predictor(
            source=video_path,
            bboxes=bboxes,
            stream=True
        )

        frame_idx = 0
        for result in results:
            if output_masks and result.masks is not None:
                masks_data = result.masks.data.cpu().numpy()

                if len(masks_data) > 0:
                    combined = np.zeros(masks_data.shape[-2:], dtype=np.float32)
                    for mask in masks_data:
                        combined = np.maximum(combined, mask.astype(np.float32))
                    yield frame_idx, combined
                else:
                    yield frame_idx, None
            else:
                yield frame_idx, None

            frame_idx += 1

            if frame_idx % 10 == 0:
                self.logger.info(f"  Processed frame {frame_idx}")

    def segment_video_with_text(
        self,
        video_path: str,
        text_prompts: List[str],
        output_masks: bool = True
    ):
        """
        Segment video using text prompts with temporal tracking.

        Args:
            video_path: Path to video file or frames directory
            text_prompts: List of text descriptions
            output_masks: Whether to yield masks for each frame

        Yields:
            (frame_idx, masks) tuples
        """
        path = Path(video_path)

        if path.is_dir():
            self.logger.info(f"Processing frames directory: {video_path}")
            yield from self.segment_frames_with_text(video_path, text_prompts)
            return

        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Text prompts: {text_prompts}")

        results = self.video_predictor(
            source=video_path,
            text=text_prompts,
            stream=True
        )

        frame_idx = 0
        for result in results:
            if output_masks and result.masks is not None:
                masks_data = result.masks.data.cpu().numpy()

                if len(masks_data) > 0:
                    combined = np.zeros(masks_data.shape[-2:], dtype=np.float32)
                    for mask in masks_data:
                        combined = np.maximum(combined, mask.astype(np.float32))
                    yield frame_idx, combined
                else:
                    yield frame_idx, None
            else:
                yield frame_idx, None

            frame_idx += 1

            if frame_idx % 10 == 0:
                self.logger.info(f"  Processed frame {frame_idx}")

    def segment_frames_with_text(
        self,
        frames_dir: str,
        text_prompts: List[str]
    ):
        """
        Segment a directory of frames using text prompts.

        Args:
            frames_dir: Path to directory containing frame images
            text_prompts: List of text descriptions

        Yields:
            (frame_idx, mask) tuples
        """
        import cv2

        frames_path = Path(frames_dir)

        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        frame_files = []
        for ext in extensions:
            frame_files.extend(sorted(frames_path.glob(ext)))

        if not frame_files:
            raise ValueError(f"No image files found in {frames_dir}")

        self.logger.info(f"Processing {len(frame_files)} frames from {frames_dir}")

        for idx, frame_path in enumerate(frame_files):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                self.logger.warning(f"Could not read frame: {frame_path}")
                yield idx, None
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = self.segment_image_with_text(frame, text_prompts)

            if masks:
                combined = np.zeros(frame.shape[:2], dtype=np.float32)
                for mask in masks:
                    combined = np.maximum(combined, mask.astype(np.float32))
                yield idx, combined
            else:
                yield idx, None

            if idx % 10 == 0:
                self.logger.info(f"  Processed frame {idx}/{len(frame_files)}")

    def release(self) -> None:
        """Release models and free GPU memory."""
        try:
            if self._image_predictor is not None:
                del self._image_predictor
                self._image_predictor = None

            if self._video_predictor is not None:
                del self._video_predictor
                self._video_predictor = None

            self.logger.debug("SAM3Segmenter models released")

            import torch
            import gc
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
