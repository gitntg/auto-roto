"""
Interactive SAM3 → MatAnyone Pipeline
======================================

Interactive workflow for high-quality video matting:
1. SAM3 segments first frame with text prompt
2. User reviews and refines with include/exclude points
3. MatAnyone processes entire video with approved mask

MatAnyone uses Consistent Memory Propagation - only needs first-frame mask.
"""

import logging
import signal
import time
import tempfile
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("AutoRoto.Pipelines.InteractiveMatAnyone")


class UserCancelledError(Exception):
    """Raised when user cancels the pipeline (ESC, Ctrl+C, Q)."""
    pass


class InteractiveMatAnyonePipeline:
    """
    Interactive SAM3 → MatAnyone pipeline.

    Workflow:
        1. Extract first frame from video
        2. Run SAM3 with text prompt
        3. Show result for user review
        4. Allow refinement with include/exclude points
        5. Run MatAnyone on entire video with approved mask

    Usage:
        pipeline = InteractiveMatAnyonePipeline(prompt="person")
        pipeline.run(input_path="video.mp4", output_dir="./output")
    """

    def __init__(
        self,
        prompt: str = "person",
        warmup: int = 10,
        erode_kernel: int = 10,
        dilate_kernel: int = 10,
        device: str = "cuda",
        output_format: str = "exr",
        bit_depth: int = 16,
        matanyone_repo: str = "./MatAnyone",
        matanyone_checkpoint: str = "./checkpoints/matanyone.pth",
        logger: logging.Logger = None
    ):
        """
        Initialize interactive pipeline.

        Args:
            prompt: Text prompt for SAM3 segmentation
            warmup: MatAnyone warmup iterations (10 for 1080p, 1 for 512p)
            erode_kernel: Erosion kernel size for mask (15 for 1080p, 4 for 512p)
            dilate_kernel: Dilation kernel size for mask (15 for 1080p, 4 for 512p)
            device: Compute device (cuda/cpu)
            output_format: Output format (exr/png)
            bit_depth: Bit depth for output
            matanyone_repo: Path to MatAnyone repository
            matanyone_checkpoint: Path to MatAnyone checkpoint
            logger: Optional logger instance
        """
        self.prompt = prompt
        self.warmup = warmup
        self.erode_kernel = erode_kernel
        self.dilate_kernel = dilate_kernel
        self.device = device
        self.output_format = output_format
        self.bit_depth = bit_depth
        self.matanyone_repo = Path(matanyone_repo)
        self.matanyone_checkpoint = Path(matanyone_checkpoint)
        self._logger = logger or logging.getLogger("InteractiveMatAnyone")

    def run(
        self,
        input_path: str,
        output_dir: str,
        auto_resolution_settings: bool = True,
        verbose: bool = False
    ) -> bool:
        """
        Run the interactive pipeline.

        Args:
            input_path: Input video path
            output_dir: Output directory
            auto_resolution_settings: Auto-adjust warmup/erode/dilate based on resolution
            verbose: Verbose output

        Returns:
            True if successful
        """
        import cv2

        # Setup signal handler for clean Ctrl+C exit
        self._cancelled = False
        original_sigint = signal.getsignal(signal.SIGINT)

        def sigint_handler(signum, frame):
            self._cancelled = True
            print("\n\nCancelled by user (Ctrl+C)")
            raise UserCancelledError("Interrupted by Ctrl+C")

        signal.signal(signal.SIGINT, sigint_handler)

        start_time = time.time()
        input_path = Path(input_path)
        output_path = Path(output_dir)

        try:
            output_path.mkdir(parents=True, exist_ok=True)

            self._logger.info("=" * 60)
            self._logger.info("INTERACTIVE SAM3 → MATANYONE PIPELINE")
            self._logger.info("=" * 60)
            self._logger.info(f"Input: {input_path}")
            self._logger.info(f"Output: {output_dir}")
            self._logger.info(f"Prompt: {self.prompt}")
            self._logger.info("(Press Ctrl+C or Q to cancel at any time)")

            # =====================================================================
            # STEP 1: Extract first frame
            # =====================================================================
            self._logger.info("\n[Step 1] Extracting first frame...")

            first_frame, fps, frame_count, frame_size = self._extract_first_frame(input_path)
            if first_frame is None:
                self._logger.error("Failed to extract first frame")
                return False

            h, w = first_frame.shape[:2]
            self._logger.info(f"  Video: {w}x{h}, {fps} fps, {frame_count} frames")

            # Auto-adjust settings based on resolution
            if auto_resolution_settings:
                min_dim = min(h, w)
                if min_dim <= 576:  # Low res
                    self.warmup = 1
                    self.erode_kernel = 4
                    self.dilate_kernel = 4
                    self._logger.info(f"  Low-res detected: warmup={self.warmup}, erode={self.erode_kernel}, dilate={self.dilate_kernel}")
                else:  # High res
                    self.warmup = 10
                    self.erode_kernel = 15
                    self.dilate_kernel = 15
                    self._logger.info(f"  High-res detected: warmup={self.warmup}, erode={self.erode_kernel}, dilate={self.dilate_kernel}")

            # =====================================================================
            # STEP 2: Interactive SAM3 refinement loop
            # =====================================================================
            self._logger.info("\n[Step 2] Interactive first-frame segmentation...")

            approved_mask = self._interactive_sam3_loop(first_frame, output_path)
            if approved_mask is None:
                self._logger.info("Pipeline cancelled by user")
                return False

            # Save approved mask
            mask_path = output_path / "first_frame_mask.png"
            cv2.imwrite(str(mask_path), (approved_mask * 255).astype(np.uint8))
            self._logger.info(f"  Approved mask saved: {mask_path}")

            # =====================================================================
            # STEP 3: Run MatAnyone
            # =====================================================================
            self._logger.info("\n[Step 3] Running MatAnyone...")

            success = self._run_matanyone(input_path, mask_path, output_path)

            # Summary
            total_time = time.time() - start_time

            self._logger.info("\n" + "=" * 60)
            self._logger.info("PIPELINE COMPLETE" if success else "PIPELINE FAILED")
            self._logger.info("=" * 60)
            self._logger.info(f"Total time: {total_time:.1f}s")
            self._logger.info(f"Output: {output_path}")

            return success

        except (UserCancelledError, KeyboardInterrupt):
            print("\n")
            self._logger.info("=" * 60)
            self._logger.info("PIPELINE CANCELLED BY USER")
            self._logger.info("=" * 60)
            return False

        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint)
            # Clean up any OpenCV windows
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _extract_first_frame(
        self,
        input_path: Path
    ) -> Tuple[Optional[np.ndarray], float, int, Tuple[int, int]]:
        """Extract first frame from video."""
        import cv2

        if input_path.is_dir():
            # Frame sequence
            frame_files = sorted(
                list(input_path.glob("*.png")) +
                list(input_path.glob("*.jpg")) +
                list(input_path.glob("*.jpeg"))
            )
            if not frame_files:
                return None, 0, 0, (0, 0)

            frame = cv2.imread(str(frame_files[0]))
            if frame is None:
                return None, 0, 0, (0, 0)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb, 24.0, len(frame_files), frame.shape[:2]

        else:
            # Video file
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                return None, 0, 0, (0, 0)

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            ret, frame = cap.read()
            cap.release()

            if not ret:
                return None, 0, 0, (0, 0)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb, fps, frame_count, frame.shape[:2]

    def _interactive_sam3_loop(
        self,
        first_frame: np.ndarray,
        output_path: Path
    ) -> Optional[np.ndarray]:
        """
        Interactive SAM3 refinement loop.

        Returns approved mask or None if cancelled.
        """
        import cv2

        from auto_roto.models.sam3 import SAM3Segmenter

        # Initialize SAM3
        sam3 = SAM3Segmenter(
            model_path="sam3.pt",
            device=self.device,
            logger=self._logger
        )

        try:
            # Initial segmentation with text prompt
            prompts = self.prompt.split(".")
            self._logger.info(f"  Running SAM3 with prompt: {prompts}")

            masks = sam3.segment_image_with_text(first_frame, prompts)

            if not masks:
                self._logger.error("  SAM3 returned no masks")
                return None

            # Combine masks
            current_mask = np.zeros(first_frame.shape[:2], dtype=np.float32)
            for m in masks:
                current_mask = np.maximum(current_mask, m.astype(np.float32))

            # Refinement points (accumulated)
            include_points = []
            exclude_points = []

            while True:
                # Save and show preview
                preview = self._create_preview(first_frame, current_mask)
                preview_path = output_path / "preview_first_frame.png"
                cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

                # Open preview
                self._open_preview(preview_path)

                # Get user choice
                choice = self._get_user_choice()

                if choice == "accept":
                    self._logger.info("  Mask approved!")
                    return current_mask

                elif choice == "add_points":
                    # Interactive point selection
                    new_include, new_exclude = self._interactive_point_selection(
                        first_frame, current_mask
                    )

                    if new_include or new_exclude:
                        include_points.extend(new_include)
                        exclude_points.extend(new_exclude)

                        self._logger.info(
                            f"  Total points: {len(include_points)} include, "
                            f"{len(exclude_points)} exclude"
                        )

                        # Re-run SAM3 with BOTH text prompts AND points together
                        # This allows proper include/exclude refinement
                        all_points = include_points + exclude_points
                        all_labels = [1] * len(include_points) + [0] * len(exclude_points)

                        if all_points:
                            # Segment with combined text + point prompts
                            refined_masks = sam3.segment_image_with_text_and_points(
                                first_frame, prompts, all_points, all_labels
                            )

                            if refined_masks:
                                # Use the refined masks (text + points combined)
                                current_mask = np.zeros(first_frame.shape[:2], dtype=np.float32)
                                for m in refined_masks:
                                    current_mask = np.maximum(current_mask, m.astype(np.float32))

                                self._logger.info("  Mask updated with text + point refinements")
                            else:
                                self._logger.warning("  No masks returned from refinement, keeping current")
                    else:
                        self._logger.info("  No points added")

                elif choice == "quit":
                    self._logger.info("  User cancelled")
                    return None

        finally:
            sam3.release()

    def _create_preview(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create preview with mask overlay."""
        # Green overlay for mask
        overlay = frame.copy().astype(np.float32)
        green = np.array([0, 255, 0], dtype=np.float32)

        mask_3ch = mask[:, :, np.newaxis]
        overlay = overlay * (1 - mask_3ch * alpha) + green * mask_3ch * alpha

        return overlay.astype(np.uint8)

    def _open_preview(self, preview_path: Path):
        """Open preview image in default viewer."""
        import platform

        system = platform.system()

        try:
            if system == "Windows":
                subprocess.Popen(["start", "", str(preview_path)], shell=True)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", str(preview_path)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(preview_path)])

            self._logger.info(f"  Preview opened: {preview_path}")
        except Exception as e:
            self._logger.warning(f"  Could not open preview: {e}")
            self._logger.info(f"  Please open manually: {preview_path}")

    def _get_user_choice(self) -> str:
        """Get user choice from terminal."""
        print("\n" + "=" * 50)
        print("REVIEW FIRST-FRAME MASK")
        print("=" * 50)
        print("  [A] Accept - proceed to MatAnyone")
        print("  [P] Add points - refine with include/exclude points")
        print("  [Q] Quit - cancel pipeline (or Ctrl+C / ESC)")
        print("=" * 50)

        while True:
            try:
                choice = input("Your choice (A/P/Q): ").strip().upper()

                if choice in ("A", "ACCEPT"):
                    return "accept"
                elif choice in ("P", "POINTS", "ADD"):
                    return "add_points"
                elif choice in ("Q", "QUIT", "EXIT", ""):
                    # Empty string can happen if user presses Ctrl+C during input
                    return "quit"
                else:
                    print("Invalid choice. Please enter A, P, or Q.")

            except (KeyboardInterrupt, EOFError):
                # Handle Ctrl+C or Ctrl+D during input
                print("\n")
                return "quit"

    def _interactive_point_selection(
        self,
        frame: np.ndarray,
        current_mask: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Interactive point selection using OpenCV window.

        Left-click: Include point
        Right-click: Exclude point
        Enter/Space: Done
        Escape: Cancel

        Returns:
            Tuple of (include_points, exclude_points)
        """
        import cv2

        include_points = []
        exclude_points = []

        # Create display image with current mask overlay
        display = self._create_preview(frame, current_mask, alpha=0.3).copy()
        original_display = display.copy()

        window_name = "Point Selection (LEFT=include, RIGHT=exclude, ENTER=done, ESC=cancel)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, frame.shape[1]), min(720, frame.shape[0]))

        def mouse_callback(event, x, y, flags, param):
            nonlocal display

            if event == cv2.EVENT_LBUTTONDOWN:
                # Include point (green)
                include_points.append((x, y))
                cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(display, (x, y), 8, (255, 255, 255), 2)
                cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

            elif event == cv2.EVENT_RBUTTONDOWN:
                # Exclude point (red)
                exclude_points.append((x, y))
                cv2.circle(display, (x, y), 8, (255, 0, 0), -1)
                cv2.circle(display, (x, y), 8, (255, 255, 255), 2)
                cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

        print("\nPoint Selection:")
        print("  LEFT-CLICK  = Add include point (green)")
        print("  RIGHT-CLICK = Add exclude point (red)")
        print("  R           = Reset all points")
        print("  ENTER/SPACE = Done selecting")
        print("  ESC/Q       = Cancel")

        try:
            while True:
                key = cv2.waitKey(100) & 0xFF  # 100ms timeout for responsiveness

                # Check if window was closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    include_points.clear()
                    exclude_points.clear()
                    break

                if key in (13, 32):  # Enter or Space
                    break
                elif key in (27, ord('q'), ord('Q')):  # Escape or Q
                    include_points.clear()
                    exclude_points.clear()
                    break
                elif key == ord('r') or key == ord('R'):  # Reset
                    display = original_display.copy()
                    include_points.clear()
                    exclude_points.clear()
                    cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                    print("  Points reset")

        except KeyboardInterrupt:
            include_points.clear()
            exclude_points.clear()

        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        return include_points, exclude_points

    def _run_matanyone(
        self,
        input_path: Path,
        mask_path: Path,
        output_path: Path
    ) -> bool:
        """Run MatAnyone on the video."""
        try:
            # Try to import and run MatAnyone directly
            return self._run_matanyone_library(input_path, mask_path, output_path)
        except ImportError:
            # Fall back to subprocess
            self._logger.info("  Running MatAnyone via subprocess...")
            return self._run_matanyone_subprocess(input_path, mask_path, output_path)

    def _run_matanyone_library(
        self,
        input_path: Path,
        mask_path: Path,
        output_path: Path
    ) -> bool:
        """Run MatAnyone using library import."""
        import sys

        # Add MatAnyone to path
        matanyone_path = self.matanyone_repo
        if matanyone_path.exists():
            sys.path.insert(0, str(matanyone_path))

        from matanyone import InferenceCore

        self._logger.info(f"  Warmup: {self.warmup}")
        self._logger.info(f"  Erode kernel: {self.erode_kernel}")
        self._logger.info(f"  Dilate kernel: {self.dilate_kernel}")

        processor = InferenceCore("PeiqingYang/MatAnyone")

        foreground_path, alpha_path = processor.process_video(
            input_path=str(input_path),
            mask_path=str(mask_path),
            output_path=str(output_path),
            n_warmup=self.warmup,
            r_erode=self.erode_kernel,
            r_dilate=self.dilate_kernel,
            save_image=True
        )

        self._logger.info(f"  Alpha output: {alpha_path}")
        self._logger.info(f"  Foreground output: {foreground_path}")

        return True

    def _run_matanyone_subprocess(
        self,
        input_path: Path,
        mask_path: Path,
        output_path: Path
    ) -> bool:
        """Run MatAnyone as subprocess."""
        script_path = self.matanyone_repo / "inference_matanyone.py"

        if not script_path.exists():
            self._logger.error(f"MatAnyone script not found: {script_path}")
            return False

        cmd = [
            sys.executable,
            str(script_path),
            "-i", str(input_path),
            "-m", str(mask_path),
            "-o", str(output_path),
            "-w", str(self.warmup),
            "-e", str(self.erode_kernel),
            "-d", str(self.dilate_kernel),
            "--save_image"
        ]

        if self.matanyone_checkpoint.exists():
            cmd.extend(["-c", str(self.matanyone_checkpoint)])

        self._logger.info(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            self._logger.error(f"  MatAnyone failed: {e.stderr}")
            return False
