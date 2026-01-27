#!/usr/bin/env python
"""
TEMPORAL COHERENCE MODULE
=========================

Maintains consistent alpha values across frames to prevent flickering.

Problems this module solves:
1. Frame-to-frame alpha flickering at edges
2. Temporal aliasing during motion
3. Inconsistent edge positions between frames
4. Jittery semi-transparent regions

Techniques used:
1. OPTICAL FLOW ESTIMATION
   - Track motion between frames
   - Warp alpha to align with motion
   - Motion-compensated filtering

2. TEMPORAL MEDIAN FILTERING
   - Remove outlier frames
   - Preserve edges while smoothing
   - Adaptive window based on motion

3. EDGE CONFIDENCE PROPAGATION
   - Track edge certainty across frames
   - Smooth only uncertain edges
   - Preserve consistently detected edges

4. KEYFRAME ANCHORING
   - Use high-confidence frames as anchors
   - Interpolate between keyframes
   - Prevent long-term drift

Author: AUTO-ROTO v5 Enhancement
License: MIT
"""

import os

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional, Dict, Generator
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque


@dataclass
class TemporalConfig:
    """Configuration for temporal smoothing."""

    # Optical flow
    use_optical_flow: bool = True
    flow_quality: str = "high"         # low, medium, high
    flow_window_size: int = 21         # Window for flow computation

    # Temporal filtering
    temporal_window: int = 5           # Frames to consider (odd number)
    temporal_sigma: float = 1.0        # Gaussian weight sigma
    motion_threshold: float = 0.5      # Motion threshold for adaptive window

    # Edge confidence
    edge_confidence_threshold: float = 0.8   # High-confidence edges preserved
    confidence_decay: float = 0.95           # How confidence decays over frames

    # Keyframe anchoring
    use_keyframes: bool = True
    keyframe_interval: int = 30        # Frames between automatic keyframes
    keyframe_weight: float = 2.0       # Extra weight for keyframes

    # Anti-flicker
    max_frame_delta: float = 0.1       # Max allowed change between frames
    flicker_suppression: float = 0.5   # 0-1, how much to suppress flicker


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("TemporalSmooth")


class OpticalFlowEstimator:
    """
    Estimate motion between frames using optical flow.

    Provides motion vectors for warping alpha mattes to
    align temporally adjacent frames.
    """

    def __init__(self, quality: str = "high"):
        self.quality = quality

        # Configure Farneback parameters based on quality
        if quality == "low":
            self.farneback_params = dict(
                pyr_scale=0.5, levels=2, winsize=11,
                iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )
        elif quality == "medium":
            self.farneback_params = dict(
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=5, poly_n=5, poly_sigma=1.2, flags=0
            )
        else:  # high
            self.farneback_params = dict(
                pyr_scale=0.5, levels=5, winsize=21,
                iterations=7, poly_n=7, poly_sigma=1.5, flags=0
            )

    def compute_flow(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> np.ndarray:
        """
        Compute optical flow from frame1 to frame2.

        Args:
            frame1: First frame (grayscale or RGB)
            frame2: Second frame (grayscale or RGB)

        Returns:
            Flow field (H, W, 2) with (dx, dy) motion vectors
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = frame1

        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = frame2

        # Ensure uint8
        if gray1.dtype != np.uint8:
            gray1 = (np.clip(gray1, 0, 1) * 255).astype(np.uint8)
        if gray2.dtype != np.uint8:
            gray2 = (np.clip(gray2, 0, 1) * 255).astype(np.uint8)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, **self.farneback_params
        )

        return flow

    def warp_image(
        self,
        image: np.ndarray,
        flow: np.ndarray
    ) -> np.ndarray:
        """
        Warp image using optical flow.

        Args:
            image: Image to warp (H, W) or (H, W, C)
            flow: Flow field (H, W, 2)

        Returns:
            Warped image
        """
        h, w = flow.shape[:2]

        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Add flow to get new coordinates
        new_x = x_coords + flow[:, :, 0]
        new_y = y_coords + flow[:, :, 1]

        # Remap image
        warped = cv2.remap(
            image.astype(np.float32),
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return warped

    def compute_motion_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """Compute motion magnitude from flow field."""
        magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        return magnitude


class TemporalMedianFilter:
    """
    Apply temporal median filtering to alpha sequence.

    Median filtering removes temporal outliers while
    preserving edges better than mean filtering.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.half_window = window_size // 2

    def filter_sequence(
        self,
        alphas: List[np.ndarray],
        flows: List[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Apply temporal median filter to alpha sequence.

        Args:
            alphas: List of alpha mattes
            flows: Optional list of optical flows for motion compensation

        Returns:
            Filtered alpha sequence
        """
        n_frames = len(alphas)
        filtered = []

        for i in range(n_frames):
            # Gather window of frames
            window_frames = []

            for j in range(-self.half_window, self.half_window + 1):
                frame_idx = i + j

                # Handle boundaries
                if frame_idx < 0:
                    frame_idx = 0
                elif frame_idx >= n_frames:
                    frame_idx = n_frames - 1

                frame = alphas[frame_idx].copy()

                # Motion compensate if flows available
                if flows is not None and j != 0:
                    # Warp frame to align with current frame
                    if j < 0:
                        # Need to warp backward
                        for k in range(frame_idx, i):
                            if k < len(flows):
                                frame = self._warp_with_flow(frame, flows[k])
                    else:
                        # Need to warp forward (reverse flow)
                        for k in range(i, frame_idx):
                            if k < len(flows):
                                frame = self._warp_with_flow(
                                    frame, -flows[k]  # Reverse flow direction
                                )

                window_frames.append(frame)

            # Stack and take median
            stacked = np.stack(window_frames, axis=0)
            median_frame = np.median(stacked, axis=0)

            filtered.append(median_frame.astype(np.float32))

        return filtered

    def _warp_with_flow(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp image using flow."""
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        new_x = x_coords + flow[:, :, 0]
        new_y = y_coords + flow[:, :, 1]
        return cv2.remap(
            image.astype(np.float32),
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )


class EdgeConfidenceTracker:
    """
    Track edge confidence across frames.

    High-confidence edges (consistently detected) are preserved,
    while uncertain edges get more temporal smoothing.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        decay_rate: float = 0.95
    ):
        self.confidence_threshold = confidence_threshold
        self.decay_rate = decay_rate
        self.confidence_map = None

    def update(
        self,
        alpha: np.ndarray,
        prev_alpha: np.ndarray = None,
        flow: np.ndarray = None
    ) -> np.ndarray:
        """
        Update confidence map with new alpha.

        Returns:
            Edge confidence map (H, W), values 0-1
        """
        # Detect edges in current frame
        edges = self._detect_edges(alpha)

        if self.confidence_map is None:
            # Initialize confidence
            self.confidence_map = edges.copy()
        else:
            # Warp previous confidence if flow available
            if flow is not None:
                warped_confidence = self._warp_image(self.confidence_map, flow)
            else:
                warped_confidence = self.confidence_map

            # Decay old confidence
            warped_confidence *= self.decay_rate

            # Boost confidence where edges are consistent
            if prev_alpha is not None:
                prev_edges = self._detect_edges(prev_alpha)

                # Warp previous edges
                if flow is not None:
                    prev_edges = self._warp_image(prev_edges, flow)

                # Consistency: both frames have edge at same location
                consistency = edges * prev_edges
                warped_confidence = np.maximum(warped_confidence, consistency)

            # Update with current edges
            self.confidence_map = np.maximum(warped_confidence, edges * 0.5)
            self.confidence_map = np.clip(self.confidence_map, 0, 1)

        return self.confidence_map

    def get_smoothing_weights(self) -> np.ndarray:
        """
        Get per-pixel smoothing weights.

        High confidence = less smoothing, low confidence = more smoothing.
        """
        if self.confidence_map is None:
            return None

        # Invert: high confidence = low weight (less smoothing)
        weights = 1 - self.confidence_map
        return weights

    def _detect_edges(self, alpha: np.ndarray) -> np.ndarray:
        """Detect edges in alpha."""
        # Sobel gradients
        gx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(gx**2 + gy**2)
        edges = edges / (edges.max() + 1e-8)
        return edges

    def _warp_image(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp image using flow."""
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        new_x = x_coords + flow[:, :, 0]
        new_y = y_coords + flow[:, :, 1]
        return cv2.remap(
            image.astype(np.float32),
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )


class KeyframeAnchor:
    """
    Use keyframes to anchor temporal smoothing.

    Keyframes are high-confidence frames that serve as
    reference points to prevent drift.
    """

    def __init__(
        self,
        keyframe_interval: int = 30,
        keyframe_weight: float = 2.0
    ):
        self.keyframe_interval = keyframe_interval
        self.keyframe_weight = keyframe_weight
        self.keyframes: Dict[int, np.ndarray] = {}
        self.keyframe_confidences: Dict[int, float] = {}

    def add_keyframe(
        self,
        frame_idx: int,
        alpha: np.ndarray,
        confidence: float = 1.0
    ):
        """Add a keyframe."""
        self.keyframes[frame_idx] = alpha.copy()
        self.keyframe_confidences[frame_idx] = confidence

    def is_keyframe(self, frame_idx: int) -> bool:
        """Check if frame should be a keyframe."""
        return frame_idx % self.keyframe_interval == 0

    def get_interpolation_weights(
        self,
        frame_idx: int,
        n_frames: int
    ) -> Tuple[Optional[int], Optional[int], float]:
        """
        Get keyframe interpolation info.

        Returns:
            (prev_keyframe_idx, next_keyframe_idx, blend_weight)
            blend_weight is 0 at prev_keyframe, 1 at next_keyframe
        """
        # Find surrounding keyframes
        prev_kf = None
        next_kf = None

        for kf_idx in sorted(self.keyframes.keys()):
            if kf_idx <= frame_idx:
                prev_kf = kf_idx
            elif kf_idx > frame_idx and next_kf is None:
                next_kf = kf_idx
                break

        if prev_kf is None and next_kf is None:
            return None, None, 0.5

        if prev_kf is None:
            return None, next_kf, 1.0

        if next_kf is None:
            return prev_kf, None, 0.0

        # Compute blend weight
        distance = next_kf - prev_kf
        if distance == 0:
            blend = 0.5
        else:
            blend = (frame_idx - prev_kf) / distance

        return prev_kf, next_kf, blend

    def anchor_alpha(
        self,
        frame_idx: int,
        alpha: np.ndarray,
        flow_to_prev_kf: np.ndarray = None,
        flow_to_next_kf: np.ndarray = None
    ) -> np.ndarray:
        """
        Anchor alpha to surrounding keyframes.

        Blends alpha towards keyframes to prevent drift.
        """
        prev_kf, next_kf, blend = self.get_interpolation_weights(frame_idx, 0)

        if prev_kf is None and next_kf is None:
            return alpha

        anchored = alpha.copy()

        # Weight to apply to keyframe influence
        kf_influence = 0.2  # How much keyframes pull the result

        if prev_kf is not None and prev_kf in self.keyframes:
            prev_alpha = self.keyframes[prev_kf]
            # Warp to current frame if flow available
            if flow_to_prev_kf is not None:
                prev_alpha = self._warp_image(prev_alpha, flow_to_prev_kf)

            # Blend towards prev keyframe
            weight = (1 - blend) * kf_influence
            anchored = anchored * (1 - weight) + prev_alpha * weight

        if next_kf is not None and next_kf in self.keyframes:
            next_alpha = self.keyframes[next_kf]
            # Warp to current frame if flow available
            if flow_to_next_kf is not None:
                next_alpha = self._warp_image(next_alpha, flow_to_next_kf)

            # Blend towards next keyframe
            weight = blend * kf_influence
            anchored = anchored * (1 - weight) + next_alpha * weight

        return anchored.astype(np.float32)

    def _warp_image(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp image using flow."""
        h, w = image.shape[:2]
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        new_x = x_coords + flow[:, :, 0]
        new_y = y_coords + flow[:, :, 1]
        return cv2.remap(
            image.astype(np.float32),
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )


class AntiFlicker:
    """
    Suppress temporal flicker in alpha values.

    Limits how much alpha can change between frames,
    especially at semi-transparent edges.
    """

    def __init__(
        self,
        max_delta: float = 0.1,
        suppression: float = 0.5
    ):
        self.max_delta = max_delta
        self.suppression = suppression
        self.prev_alpha = None

    def process(
        self,
        alpha: np.ndarray,
        motion_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Apply anti-flicker processing.

        Args:
            alpha: Current alpha
            motion_mask: Optional mask where motion is occurring (less suppression)

        Returns:
            Flicker-suppressed alpha
        """
        if self.prev_alpha is None:
            self.prev_alpha = alpha.copy()
            return alpha

        # Compute delta from previous frame
        delta = alpha - self.prev_alpha

        # Limit delta
        max_delta = self.max_delta
        if motion_mask is not None:
            # Allow more change where there's motion
            max_delta = max_delta * (1 + motion_mask * 2)

        # Clip delta
        limited_delta = np.clip(delta, -max_delta, max_delta)

        # Blend based on suppression strength
        smoothed = self.prev_alpha + limited_delta * self.suppression

        # Store for next frame
        self.prev_alpha = smoothed.copy()

        return np.clip(smoothed, 0, 1).astype(np.float32)


class TemporalSmoother:
    """
    Main temporal smoothing class combining all techniques.
    """

    def __init__(
        self,
        config: TemporalConfig = None,
        logger: logging.Logger = None
    ):
        self.config = config or TemporalConfig()
        self.logger = logger or logging.getLogger("TemporalSmoother")

        # Initialize components
        self.flow_estimator = OpticalFlowEstimator(self.config.flow_quality)
        self.median_filter = TemporalMedianFilter(self.config.temporal_window)
        self.confidence_tracker = EdgeConfidenceTracker(
            self.config.edge_confidence_threshold,
            self.config.confidence_decay
        )
        self.keyframe_anchor = KeyframeAnchor(
            self.config.keyframe_interval,
            self.config.keyframe_weight
        )
        self.anti_flicker = AntiFlicker(
            self.config.max_frame_delta,
            self.config.flicker_suppression
        )

    def process_sequence(
        self,
        alphas: List[np.ndarray],
        frames: List[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Process entire alpha sequence for temporal coherence.

        Args:
            alphas: List of alpha mattes
            frames: Optional list of RGB frames for optical flow

        Returns:
            Temporally smoothed alpha sequence
        """
        n_frames = len(alphas)
        self.logger.info(f"Processing {n_frames} frames for temporal coherence...")

        # Step 1: Compute optical flows if frames available
        flows = []
        if frames is not None and self.config.use_optical_flow:
            self.logger.info("Computing optical flows...")
            for i in range(n_frames - 1):
                flow = self.flow_estimator.compute_flow(frames[i], frames[i + 1])
                flows.append(flow)
                if i % 50 == 0:
                    self.logger.debug(f"  Computed flow for frame {i}")

        # Step 2: Identify and store keyframes
        if self.config.use_keyframes:
            self.logger.info("Setting up keyframes...")
            for i in range(n_frames):
                if self.keyframe_anchor.is_keyframe(i):
                    self.keyframe_anchor.add_keyframe(i, alphas[i])

        # Step 3: Apply temporal median filter
        self.logger.info("Applying temporal median filter...")
        filtered_alphas = self.median_filter.filter_sequence(
            alphas,
            flows if flows else None
        )

        # Step 4: Apply edge confidence weighting
        self.logger.info("Applying edge confidence weighting...")
        smoothed_alphas = []

        for i in range(n_frames):
            alpha = filtered_alphas[i]
            prev_alpha = filtered_alphas[i - 1] if i > 0 else None
            flow = flows[i - 1] if flows and i > 0 else None

            # Update confidence
            confidence = self.confidence_tracker.update(alpha, prev_alpha, flow)

            # Get smoothing weights (high confidence = less smoothing)
            smoothing_weights = self.confidence_tracker.get_smoothing_weights()

            if smoothing_weights is not None and prev_alpha is not None:
                # Blend with previous frame based on confidence
                # Low confidence edges get more temporal blending
                blended = alpha * (1 - smoothing_weights * 0.3) + prev_alpha * (smoothing_weights * 0.3)
                alpha = blended

            smoothed_alphas.append(alpha)

        # Step 5: Apply keyframe anchoring
        if self.config.use_keyframes:
            self.logger.info("Applying keyframe anchoring...")
            anchored_alphas = []

            for i in range(n_frames):
                anchored = self.keyframe_anchor.anchor_alpha(
                    i, smoothed_alphas[i]
                )
                anchored_alphas.append(anchored)

            smoothed_alphas = anchored_alphas

        # Step 6: Apply anti-flicker
        self.logger.info("Applying anti-flicker...")
        final_alphas = []

        for i in range(n_frames):
            # Get motion magnitude for this frame
            motion_mask = None
            if flows and i < len(flows):
                motion_mag = self.flow_estimator.compute_motion_magnitude(flows[i])
                motion_mask = motion_mag / (motion_mag.max() + 1e-8)

            deflickered = self.anti_flicker.process(smoothed_alphas[i], motion_mask)
            final_alphas.append(deflickered)

        self.logger.info("Temporal smoothing complete!")
        return final_alphas


# ==============================================================================
# STREAMING PROCESSOR
# ==============================================================================

class StreamingTemporalProcessor:
    """
    Process frames in a streaming manner for memory efficiency.

    Uses a sliding window instead of loading all frames at once.
    """

    def __init__(
        self,
        config: TemporalConfig = None,
        logger: logging.Logger = None
    ):
        self.config = config or TemporalConfig()
        self.logger = logger or logging.getLogger("StreamingTemporal")

        self.window_size = config.temporal_window if config else 5
        self.half_window = self.window_size // 2

        # Sliding window buffers
        self.alpha_buffer: deque = deque(maxlen=self.window_size)
        self.frame_buffer: deque = deque(maxlen=self.window_size)
        self.flow_buffer: deque = deque(maxlen=self.window_size - 1)

        # Components
        self.flow_estimator = OpticalFlowEstimator(
            self.config.flow_quality if config else "high"
        )
        self.anti_flicker = AntiFlicker(
            self.config.max_frame_delta if config else 0.1,
            self.config.flicker_suppression if config else 0.5
        )

        self.frame_count = 0
        self.prev_alpha = None

    def process_frame(
        self,
        alpha: np.ndarray,
        rgb_frame: np.ndarray = None
    ) -> Optional[np.ndarray]:
        """
        Process a single frame in streaming mode.

        Returns smoothed alpha for frame (frame_count - half_window),
        or None if still filling buffer.
        """
        # Add to buffers
        self.alpha_buffer.append(alpha.copy())

        if rgb_frame is not None:
            self.frame_buffer.append(rgb_frame)

            # Compute flow to previous frame
            if len(self.frame_buffer) >= 2:
                flow = self.flow_estimator.compute_flow(
                    self.frame_buffer[-2],
                    self.frame_buffer[-1]
                )
                self.flow_buffer.append(flow)

        self.frame_count += 1

        # Wait until buffer is full
        if len(self.alpha_buffer) < self.window_size:
            return None

        # Process center frame of window
        center_idx = self.half_window
        alphas_list = list(self.alpha_buffer)

        # Motion-compensated median
        flows_list = list(self.flow_buffer) if self.flow_buffer else None
        smoothed = self._temporal_median(alphas_list, flows_list, center_idx)

        # Anti-flicker
        motion_mask = None
        if flows_list and center_idx < len(flows_list):
            motion_mag = self.flow_estimator.compute_motion_magnitude(
                flows_list[center_idx]
            )
            motion_mask = motion_mag / (motion_mag.max() + 1e-8)

        result = self.anti_flicker.process(smoothed, motion_mask)

        return result

    def flush(self) -> List[np.ndarray]:
        """
        Flush remaining frames in buffer.

        Call after all frames processed.
        """
        results = []

        # Process remaining frames with smaller window
        while len(self.alpha_buffer) > 0:
            alphas_list = list(self.alpha_buffer)
            center_idx = min(self.half_window, len(alphas_list) - 1)

            smoothed = self._temporal_median(alphas_list, None, center_idx)
            result = self.anti_flicker.process(smoothed, None)
            results.append(result)

            self.alpha_buffer.popleft()

        return results

    def _temporal_median(
        self,
        alphas: List[np.ndarray],
        flows: List[np.ndarray],
        center_idx: int
    ) -> np.ndarray:
        """Compute motion-compensated temporal median."""
        if len(alphas) == 1:
            return alphas[0]

        # Simple median without motion compensation
        stacked = np.stack(alphas, axis=0)
        median = np.median(stacked, axis=0)

        return median.astype(np.float32)


# ==============================================================================
# PIPELINE
# ==============================================================================

class TemporalSmoothPipeline:
    """
    Pipeline for batch temporal smoothing of alpha sequences.
    """

    def __init__(
        self,
        config: TemporalConfig = None,
        output_format: str = "exr",
        bit_depth: int = 16,
        verbose: bool = False
    ):
        self.config = config or TemporalConfig()
        self.output_format = output_format
        self.bit_depth = bit_depth
        self.logger = setup_logging(verbose)
        self.smoother = TemporalSmoother(self.config, self.logger)

    def process_sequence(
        self,
        alpha_dir: Path,
        output_dir: Path,
        frames_dir: Path = None
    ):
        """
        Process alpha sequence for temporal coherence.

        Args:
            alpha_dir: Directory with alpha mattes
            output_dir: Output directory
            frames_dir: Optional RGB frames for optical flow
        """
        import cv2

        output_dir.mkdir(parents=True, exist_ok=True)
        smoothed_dir = output_dir / "alpha"
        smoothed_dir.mkdir(exist_ok=True)

        # Load all alphas
        alpha_files = sorted(list(alpha_dir.glob("*.exr")) +
                           list(alpha_dir.glob("*.png")))

        self.logger.info(f"Loading {len(alpha_files)} alpha mattes...")

        alphas = []
        for path in alpha_files:
            alpha = self._load_alpha(path)
            alphas.append(alpha)

        # Load frames if available
        frames = None
        if frames_dir and frames_dir.exists():
            frame_files = sorted(list(frames_dir.glob("*.png")) +
                               list(frames_dir.glob("*.jpg")))

            self.logger.info(f"Loading {len(frame_files)} RGB frames...")

            frames = []
            for path in frame_files[:len(alphas)]:
                frame = cv2.imread(str(path))
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

        # Process
        smoothed = self.smoother.process_sequence(alphas, frames)

        # Save
        self.logger.info("Saving smoothed alphas...")
        for i, alpha in enumerate(smoothed):
            output_path = smoothed_dir / f"smoothed.{i:04d}.{self.output_format}"
            self._save_alpha(alpha, output_path)

        self.logger.info(f"Output saved to: {smoothed_dir}")

    def _load_alpha(self, path: Path) -> np.ndarray:
        """Load alpha from file."""
        import cv2

        if path.suffix.lower() == '.exr':
            try:
                import OpenEXR
                import Imath

                exr = OpenEXR.InputFile(str(path))
                header = exr.header()
                dw = header['dataWindow']
                w = dw.max.x - dw.min.x + 1
                h = dw.max.y - dw.min.y + 1

                channels = list(header['channels'].keys())
                channel = 'A' if 'A' in channels else channels[0]

                pt = Imath.PixelType(Imath.PixelType.FLOAT)
                data = exr.channel(channel, pt)
                return np.frombuffer(data, dtype=np.float32).reshape((h, w))
            except ImportError:
                pass

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read: {path}")

        if len(img.shape) == 3:
            img = img[:, :, -1]

        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        return img.astype(np.float32)

    def _save_alpha(self, alpha: np.ndarray, path: Path):
        """Save alpha to file."""
        import cv2

        if path.suffix.lower() == '.exr':
            try:
                import OpenEXR
                import Imath

                h, w = alpha.shape
                header = OpenEXR.Header(w, h)
                header['channels'] = {'A': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

                exr = OpenEXR.OutputFile(str(path), header)
                exr.writePixels({'A': alpha.astype(np.float32).tobytes()})
                exr.close()
                return
            except ImportError:
                pass

            if cv2.imwrite(str(path), alpha.astype(np.float32)):
                return

            path = path.with_suffix('.png')

        if self.bit_depth == 16:
            alpha_int = (alpha * 65535).astype(np.uint16)
        else:
            alpha_int = (alpha * 255).astype(np.uint8)

        cv2.imwrite(str(path), alpha_int)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Temporal coherence for alpha matte sequences"
    )

    parser.add_argument("--alpha", "-a", required=True,
                       help="Directory with alpha mattes")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory")
    parser.add_argument("--frames", "-f",
                       help="Directory with RGB frames for optical flow")

    # Temporal settings
    parser.add_argument("--window", type=int, default=5,
                       help="Temporal window size (default: 5)")
    parser.add_argument("--keyframe-interval", type=int, default=30,
                       help="Frames between keyframes (default: 30)")
    parser.add_argument("--max-delta", type=float, default=0.1,
                       help="Max frame-to-frame change (default: 0.1)")

    # Flow settings
    parser.add_argument("--no-flow", action="store_true",
                       help="Disable optical flow")
    parser.add_argument("--flow-quality", default="high",
                       choices=["low", "medium", "high"])

    # Output
    parser.add_argument("--format", default="exr",
                       choices=["exr", "png", "tiff"])
    parser.add_argument("--bit-depth", type=int, default=16,
                       choices=[8, 16, 32])

    parser.add_argument("--verbose", "-v", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    config = TemporalConfig(
        use_optical_flow=not args.no_flow,
        flow_quality=args.flow_quality,
        temporal_window=args.window,
        keyframe_interval=args.keyframe_interval,
        max_frame_delta=args.max_delta
    )

    pipeline = TemporalSmoothPipeline(
        config=config,
        output_format=args.format,
        bit_depth=args.bit_depth,
        verbose=args.verbose
    )

    pipeline.process_sequence(
        alpha_dir=Path(args.alpha),
        output_dir=Path(args.output),
        frames_dir=Path(args.frames) if args.frames else None
    )


if __name__ == "__main__":
    main()
