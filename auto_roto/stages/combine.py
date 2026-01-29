"""
Combine Stage
=============

Matte combination stage for the auto-roto pipeline.

Combines alpha mattes with RGB frames to produce final output.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from auto_roto.stages.base import PipelineStage, StageContext, StageResult, StageRegistry
from auto_roto.config.matte import MatteCombineConfig

logger = logging.getLogger("AutoRoto.Stages.Combine")


@StageRegistry.register
class CombineStage(PipelineStage):
    """
    Matte combination stage.

    Combines alpha mattes with RGB frames to produce:
    - Final alpha mattes (with core erosion)
    - RGBA composites
    - Preview composites

    Outputs:
        - alpha/: Final alpha mattes
        - rgba/: RGBA composites with premultiplied alpha
        - preview/: Preview composites on checkerboard
    """

    def __init__(
        self,
        config: MatteCombineConfig = None,
        logger: logging.Logger = None
    ):
        """
        Initialize combine stage.

        Args:
            config: MatteCombineConfig instance (uses defaults if None)
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.config = config or MatteCombineConfig()

    @property
    def name(self) -> str:
        return "combine"

    @property
    def description(self) -> str:
        return "Matte Combination"

    def validate_inputs(self, context: StageContext) -> Optional[str]:
        """Validate that alpha mattes and frames exist."""
        # Check for alpha output from vitmatte (preferred) or sam
        alpha_dir = context.get_alpha_dir("vitmatte")
        if not alpha_dir:
            alpha_dir = context.get_alpha_dir("sam")

        if not alpha_dir or not alpha_dir.exists():
            return "No alpha output found from previous stages"

        # Check for frames
        frames_dir = context.frames_dir or context.input_path
        if not frames_dir.exists():
            return f"Frames directory not found: {frames_dir}"

        return None

    def run(self, context: StageContext) -> StageResult:
        """Execute matte combination."""
        import cv2

        from auto_roto.io.alpha import load_alpha, save_alpha
        from auto_roto.matte.combiner import MatteCombiner

        # Setup output directory
        output_dir = context.output_dir / "04_combine_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        alpha_out_dir = output_dir / "alpha"
        rgba_dir = output_dir / "rgba"
        preview_dir = output_dir / "preview"

        alpha_out_dir.mkdir(exist_ok=True)
        rgba_dir.mkdir(exist_ok=True)
        preview_dir.mkdir(exist_ok=True)

        # Get input directories
        alpha_dir = context.get_alpha_dir("vitmatte")
        if not alpha_dir:
            alpha_dir = context.get_alpha_dir("sam")

        frames_dir = context.frames_dir or context.input_path

        # Get output format
        output_format = context.config.get("output_format", "exr")
        bit_depth = context.config.get("bit_depth", 16)

        # Initialize combiner
        combiner = MatteCombiner(
            config=self.config,
            logger=self._logger
        )

        # Get frame files
        frame_files = sorted(
            list(frames_dir.glob("*.png")) +
            list(frames_dir.glob("*.jpg")) +
            list(frames_dir.glob("*.jpeg"))
        )

        # Get alpha files
        alpha_files = sorted(
            list(alpha_dir.glob("*.exr")) +
            list(alpha_dir.glob("*.png"))
        )

        if not frame_files or not alpha_files:
            return StageResult.failure_result(
                ValueError("Missing input files"),
                message="No frame or alpha files found"
            )

        min_count = min(len(frame_files), len(alpha_files))
        self._logger.info(f"Processing {min_count} frames")

        frame_count = 0

        for idx in range(min_count):
            # Load inputs
            frame = cv2.imread(str(frame_files[idx]))
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            alpha = load_alpha(alpha_files[idx])

            # Process frame
            result = combiner.combine(rgb, alpha)

            # Save outputs
            # Alpha
            alpha_path = alpha_out_dir / f"alpha.{idx:04d}.{output_format}"
            save_alpha(alpha_path, result["alpha"], bit_depth=bit_depth)

            # RGBA
            rgba_path = rgba_dir / f"rgba.{idx:04d}.{output_format}"
            self._save_rgba(rgba_path, result["rgba"], output_format, bit_depth)

            # Preview
            preview_path = preview_dir / f"preview.{idx:04d}.png"
            self._save_preview(preview_path, result["preview"])

            frame_count = idx + 1

            if idx % 10 == 0:
                self._logger.info(f"  Frame {idx}/{min_count}")

        self._logger.info(f"Combined {frame_count} frames")

        return StageResult.success_result(
            output_dir=output_dir,
            message=f"Combined {frame_count} frames",
            outputs={
                "alpha_dir": str(alpha_out_dir),
                "rgba_dir": str(rgba_dir),
                "preview_dir": str(preview_dir),
                "frame_count": frame_count
            }
        )

    def _save_rgba(
        self,
        path: Path,
        rgba: np.ndarray,
        output_format: str,
        bit_depth: int
    ):
        """Save RGBA image."""
        import cv2

        if output_format == "exr":
            # OpenEXR format
            try:
                import OpenEXR
                import Imath

                h, w = rgba.shape[:2]
                header = OpenEXR.Header(w, h)
                header['compression'] = Imath.Compression(Imath.Compression.DWAA_COMPRESSION)

                # Convert to float32
                rgba_f = rgba.astype(np.float32)
                if rgba_f.max() > 1.0:
                    rgba_f = rgba_f / 255.0

                # Split channels
                r = rgba_f[:, :, 0].tobytes()
                g = rgba_f[:, :, 1].tobytes()
                b = rgba_f[:, :, 2].tobytes()
                a = rgba_f[:, :, 3].tobytes()

                out = OpenEXR.OutputFile(str(path), header)
                out.writePixels({'R': r, 'G': g, 'B': b, 'A': a})
                out.close()

            except ImportError:
                # Fallback to cv2
                cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        elif output_format == "png":
            # PNG format
            if bit_depth == 16:
                rgba_16 = (rgba * 65535).astype(np.uint16) if rgba.max() <= 1.0 else rgba.astype(np.uint16)
                cv2.imwrite(str(path), cv2.cvtColor(rgba_16, cv2.COLOR_RGBA2BGRA))
            else:
                rgba_8 = (rgba * 255).astype(np.uint8) if rgba.max() <= 1.0 else rgba.astype(np.uint8)
                cv2.imwrite(str(path), cv2.cvtColor(rgba_8, cv2.COLOR_RGBA2BGRA))

        else:
            # Default: convert to 8-bit
            rgba_8 = (rgba * 255).astype(np.uint8) if rgba.max() <= 1.0 else rgba.astype(np.uint8)
            cv2.imwrite(str(path), cv2.cvtColor(rgba_8, cv2.COLOR_RGBA2BGRA))

    def _save_preview(self, path: Path, preview: np.ndarray):
        """Save preview image."""
        import cv2

        # Convert to 8-bit if needed
        if preview.max() <= 1.0:
            preview = (preview * 255).astype(np.uint8)

        # Convert RGB to BGR for cv2
        if preview.ndim == 3 and preview.shape[2] == 3:
            preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(path), preview)
