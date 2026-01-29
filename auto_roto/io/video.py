"""
Video Reader
============

Read video files or image sequences into frames.
"""

import logging
from pathlib import Path
from typing import List, Iterator, Optional
import re
import glob

import numpy as np

logger = logging.getLogger("AutoRoto.IO.Video")

# Bit depth conversion constant
BIT_DEPTH_16_TO_8_DIVISOR = 256


def normalize_frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Normalize a frame to 8-bit RGB format.

    Handles:
        - 16-bit to 8-bit conversion
        - Float to uint8 conversion
        - Grayscale to RGB conversion
        - BGRA/BGR to RGB conversion

    Args:
        frame: Input frame (numpy array)

    Returns:
        RGB uint8 numpy array
    """
    import cv2

    # Convert to 8-bit if needed
    if frame.dtype == np.uint16:
        frame = (frame / BIT_DEPTH_16_TO_8_DIVISOR).astype(np.uint8)
    elif frame.dtype in (np.float32, np.float64):
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

    # Convert to RGB
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


class VideoReader:
    """
    Read video files or image sequences into frames.

    Supports:
        - Video files: .mp4, .mov, .avi, .mkv
        - Image sequences: frame.####.exr, frame_####.png, etc.
    """

    SUPPORTED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf'}
    SUPPORTED_IMAGE = {'.exr', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.dpx'}

    def __init__(self, path: str, logger: logging.Logger = None):
        """
        Initialize video reader.

        Args:
            path: Path to video file, image file, or directory
            logger: Optional logger instance
        """
        self.path = Path(path)
        self.logger = logger or logging.getLogger("VideoReader")
        self.frames: List[Path] = []
        self.fps: float = 24.0
        self.width: int = 0
        self.height: int = 0
        self._is_sequence = False
        self._cap = None

        self._detect_input_type()

    def _detect_input_type(self):
        """Detect if input is video file or image sequence."""
        if self.path.is_file():
            ext = self.path.suffix.lower()
            if ext in self.SUPPORTED_VIDEO:
                self._init_video()
            elif ext in self.SUPPORTED_IMAGE:
                self._init_sequence_from_file()
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        elif self.path.is_dir():
            self._init_sequence_from_dir()
        else:
            # Try pattern matching for sequences
            self._init_sequence_from_pattern()

    def _init_video(self):
        """Initialize from video file."""
        import cv2

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Video: {self.path.name}")
        self.logger.info(f"  Resolution: {self.width}x{self.height}")
        self.logger.info(f"  FPS: {self.fps:.2f}")
        self.logger.info(f"  Frames: {frame_count}")

    def _init_sequence_from_dir(self):
        """Initialize from directory of images."""
        self._is_sequence = True

        # Find all image files
        for ext in self.SUPPORTED_IMAGE:
            self.frames.extend(sorted(self.path.glob(f"*{ext}")))
            self.frames.extend(sorted(self.path.glob(f"*{ext.upper()}")))

        self.frames = sorted(set(self.frames))

        if not self.frames:
            raise ValueError(f"No image files found in: {self.path}")

        # Get dimensions from first frame
        self._read_first_frame_info()

    def _init_sequence_from_file(self):
        """Initialize sequence from single file (find related frames)."""
        self._is_sequence = True

        # Extract pattern from filename
        name = self.path.stem
        ext = self.path.suffix
        parent = self.path.parent

        # Match common patterns: name.0001.ext, name_0001.ext, name0001.ext
        patterns = [
            r'^(.+?)[\._]?(\d+)$',
        ]

        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                prefix = match.group(1)
                # Find all related frames
                for f in sorted(parent.glob(f"{prefix}*{ext}")):
                    if re.match(rf'^{re.escape(prefix)}[\._]?\d+$', f.stem):
                        self.frames.append(f)
                break

        if not self.frames:
            self.frames = [self.path]

        self._read_first_frame_info()

    def _init_sequence_from_pattern(self):
        """Initialize from glob pattern."""
        self._is_sequence = True

        # Replace # with glob pattern
        pattern = str(self.path).replace('#', '[0-9]')
        self.frames = sorted([Path(f) for f in glob.glob(pattern)])

        if not self.frames:
            raise ValueError(f"No files match pattern: {self.path}")

        self._read_first_frame_info()

    def _read_first_frame_info(self):
        """Read first frame to get dimensions."""
        import cv2

        frame = cv2.imread(str(self.frames[0]), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise IOError(f"Cannot read: {self.frames[0]}")

        self.height, self.width = frame.shape[:2]

        self.logger.info(f"Sequence: {self.frames[0].parent.name}/")
        self.logger.info(f"  Resolution: {self.width}x{self.height}")
        self.logger.info(f"  Frames: {len(self.frames)}")

    def __len__(self) -> int:
        """Return number of frames."""
        import cv2
        if self._is_sequence:
            return len(self.frames)
        else:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) if self._cap else 0)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames as numpy arrays (RGB)."""
        import cv2

        if self._is_sequence:
            for frame_path in self.frames:
                frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                if frame is None:
                    self.logger.warning(f"Skipping unreadable frame: {frame_path}")
                    continue
                yield normalize_frame_to_rgb(frame)
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame(self, idx: int) -> Optional[np.ndarray]:
        """
        Get specific frame by index.

        Args:
            idx: Frame index (0-based)

        Returns:
            RGB frame or None if not available
        """
        import cv2

        if self._is_sequence:
            if 0 <= idx < len(self.frames):
                frame = cv2.imread(str(self.frames[idx]), cv2.IMREAD_UNCHANGED)
                if frame is not None:
                    return normalize_frame_to_rgb(frame)
                return frame
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_frame_path(self, idx: int) -> Optional[Path]:
        """
        Get path to frame file by index (for sequences only).

        Args:
            idx: Frame index

        Returns:
            Path to frame file or None
        """
        if self._is_sequence and 0 <= idx < len(self.frames):
            return self.frames[idx]
        return None

    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
