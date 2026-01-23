"""
Pytest configuration and shared fixtures for AUTO-ROTO tests.

This module provides:
- Temporary directory fixtures
- Sample data generation (frames, alphas, depths, optical flow)
- Configuration fixtures
- Mocked model fixtures
- Performance monitoring fixtures
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Tuple
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

# ==============================================================================
# SESSION-LEVEL FIXTURES (run once per test session)
# ==============================================================================

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def sample_video_path(test_data_dir) -> Path:
    """Path to sample test video (creates a synthetic one if needed)."""
    video_path = test_data_dir / "sample_test.mp4"

    if not video_path.exists():
        pytest.skip("Test video not available (optional)")

    return video_path


# ==============================================================================
# FUNCTION-LEVEL FIXTURES (run before each test)
# ==============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create and cleanup a temporary directory."""
    tmpdir = tempfile.mkdtemp(prefix="auto_roto_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_output_dir(temp_dir) -> Path:
    """Create output directory structure for pipeline tests."""
    output_dir = temp_dir / "output"
    (output_dir / "alpha").mkdir(parents=True)
    (output_dir / "preview").mkdir(parents=True)
    (output_dir / "depth").mkdir(parents=True)
    return output_dir


# ==============================================================================
# SAMPLE DATA FIXTURES
# ==============================================================================

@pytest.fixture
def sample_frame_512x512() -> np.ndarray:
    """Generate a 512x512 RGB test frame."""
    frame = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_frame_1080p() -> np.ndarray:
    """Generate a 1920x1080 RGB test frame."""
    frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_frame_with_content() -> np.ndarray:
    """Generate a test frame with distinct regions (not random noise)."""
    frame = np.ones((512, 512, 3), dtype=np.uint8) * 128  # Gray background

    # Add a red square (object)
    frame[100:300, 100:300] = [255, 0, 0]

    # Add some green (background)
    frame[50:150, 350:450] = [0, 255, 0]

    return frame


@pytest.fixture
def sample_alpha_512x512() -> np.ndarray:
    """Generate a 512x512 test alpha map (binary)."""
    alpha = np.zeros((512, 512), dtype=np.float32)
    alpha[100:400, 100:400] = 1.0  # Solid square
    return alpha


@pytest.fixture
def sample_alpha_soft() -> np.ndarray:
    """Generate a 512x512 test alpha map with soft edges."""
    alpha = np.zeros((512, 512), dtype=np.float32)

    # Create soft edge using distance transform
    cy, cx = 256, 256
    y, x = np.ogrid[:512, :512]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Soft falloff
    alpha = np.exp(-(dist / 100)**2)
    return np.clip(alpha, 0, 1).astype(np.float32)


@pytest.fixture
def sample_alpha_with_hair() -> np.ndarray:
    """Generate an alpha map with fine detail regions (hair-like)."""
    alpha = np.zeros((512, 512), dtype=np.float32)

    # Solid core
    alpha[150:350, 150:350] = 1.0

    # Hair strands (semi-transparent lines)
    for x in range(200, 300, 10):
        for y in range(100, 150):
            alpha[y, x] = 0.5 + 0.3 * np.sin(y * 0.1)

    return np.clip(alpha, 0, 1).astype(np.float32)


@pytest.fixture
def sample_depth_map() -> np.ndarray:
    """Generate a 512x512 depth map."""
    depth = np.random.rand(512, 512).astype(np.float32)
    return depth


@pytest.fixture
def sample_depth_with_edge() -> np.ndarray:
    """Generate depth map with a sharp edge (useful for testing edge detection)."""
    depth = np.ones((512, 512), dtype=np.float32) * 0.5
    depth[:, 256:] = 0.8  # Step discontinuity in middle
    return depth


@pytest.fixture
def sample_alpha_sequence(sample_alpha_512x512) -> list:
    """Generate a sequence of 10 alpha frames."""
    sequence = []
    for i in range(10):
        # Slight variation between frames (for testing temporal stability)
        alpha = sample_alpha_512x512.copy()
        alpha += np.random.randn(512, 512) * 0.01  # Small noise
        alpha = np.clip(alpha, 0, 1)
        sequence.append(alpha.astype(np.float32))
    return sequence


@pytest.fixture
def sample_alpha_sequence_flickering() -> list:
    """Generate a sequence with significant flicker (for testing temporal smoothing)."""
    sequence = []
    base_alpha = np.zeros((256, 256), dtype=np.float32)
    base_alpha[50:200, 50:200] = 1.0

    for i in range(20):
        # Alternate between two significantly different states
        alpha = base_alpha.copy()
        if i % 2 == 0:
            alpha += np.random.rand(256, 256) * 0.3
        else:
            alpha -= np.random.rand(256, 256) * 0.3

        alpha = np.clip(alpha, 0, 1)
        sequence.append(alpha.astype(np.float32))

    return sequence


@pytest.fixture
def sample_optical_flow() -> np.ndarray:
    """Generate a sample optical flow field (2 channels: dx, dy)."""
    flow = np.random.randn(512, 512, 2).astype(np.float32) * 2.0  # Small motion
    return flow


@pytest.fixture
def sample_optical_flow_with_motion() -> np.ndarray:
    """Generate optical flow with distinct motion regions."""
    flow = np.zeros((512, 512, 2), dtype=np.float32)

    # Left half moves right
    flow[:256, :, 0] = 5.0

    # Right half moves down
    flow[256:, :, 1] = 5.0

    return flow


@pytest.fixture
def sample_trimap() -> np.ndarray:
    """Generate a trimap (3-valued: 0=bg, 128=unknown, 255=fg)."""
    trimap = np.zeros((512, 512), dtype=np.uint8)

    # Foreground (255)
    trimap[150:350, 150:350] = 255

    # Unknown region around boundary
    trimap[130:370, 130:370] = 128
    trimap[150:350, 150:350] = 255  # Restore foreground

    return trimap


# ==============================================================================
# CONFIGURATION FIXTURES
# ==============================================================================

@pytest.fixture
def depth_config():
    """Fixture for DepthRefineConfig."""
    from depth_refine import DepthRefineConfig

    return DepthRefineConfig(
        input_dir="./test_input",
        output_dir="./test_output",
        depth_model="small",
        depth_version="v3",
        device="cpu",
        verbose=False
    )


@pytest.fixture
def matte_config():
    """Fixture for MatteCombineConfig."""
    from matte_combine import MatteCombineConfig

    return MatteCombineConfig(
        core_erosion=5,
        core_feather=2,
        detail_blend=0.8,
        despill_enabled=True,
        despill_strength=0.5,
        premultiply=True
    )


@pytest.fixture
def temporal_config():
    """Fixture for TemporalConfig."""
    from temporal_smooth import TemporalConfig

    return TemporalConfig(
        use_optical_flow=True,
        flow_quality="high",
        temporal_window=5,
        motion_threshold=0.5,
        use_keyframes=True
    )


# ==============================================================================
# MOCK FIXTURES
# ==============================================================================

@pytest.fixture
def mock_depth_estimator(monkeypatch):
    """Mock DepthEstimator that returns synthetic depth."""
    def mock_estimate(frame: np.ndarray) -> np.ndarray:
        """Return synthetic depth matching frame shape."""
        assert frame.ndim == 3 and frame.shape[2] == 3
        return np.random.rand(frame.shape[0], frame.shape[1]).astype(np.float32)

    return mock_estimate


@pytest.fixture
def mock_sam2_predictor(monkeypatch):
    """Mock SAM2 predictor that returns synthetic masks."""
    class MockPredictor:
        def __init__(self):
            self.inference_state = {}

        def init_state(self, video_path):
            self.inference_state = {"video_path": video_path}

        def add_new_prompt(self, frame_idx, box=None, point=None):
            return {"mask": np.zeros((1024, 1024), dtype=np.uint8)}

        def propagate_in_video(self, predictor):
            return [np.zeros((1024, 1024), dtype=np.uint8) for _ in range(10)]

    return MockPredictor()


# ==============================================================================
# PERFORMANCE MONITORING FIXTURES
# ==============================================================================

@pytest.fixture
def timer():
    """Simple timer for performance tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.end_time = time.perf_counter()

        @property
        def elapsed(self) -> float:
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0.0

    return Timer()


@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    import psutil
    import os

    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = None
            self.peak_memory = None
            self.samples = []

        def start(self):
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory

        def sample(self):
            current = self.process.memory_info().rss / 1024 / 1024
            self.samples.append(current)
            self.peak_memory = max(self.peak_memory, current)

        @property
        def growth_mb(self) -> float:
            if self.start_memory and self.peak_memory:
                return self.peak_memory - self.start_memory
            return 0.0

        @property
        def growth_pct(self) -> float:
            if self.start_memory and self.peak_memory:
                return (self.peak_memory - self.start_memory) / self.start_memory * 100
            return 0.0

    return MemoryTracker()


# ==============================================================================
# ASSERTION HELPERS
# ==============================================================================

def assert_alpha_valid(alpha: np.ndarray, name: str = "alpha"):
    """Assert that alpha map is valid."""
    assert alpha.ndim == 2, f"{name} should be 2D, got {alpha.ndim}D"
    assert alpha.dtype in [np.float32, np.float64], f"{name} dtype should be float, got {alpha.dtype}"
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0, \
        f"{name} range [{alpha.min()}, {alpha.max()}] outside [0, 1]"


def assert_frame_valid(frame: np.ndarray, name: str = "frame"):
    """Assert that RGB frame is valid."""
    assert frame.ndim == 3 and frame.shape[2] == 3, \
        f"{name} should be (H, W, 3), got {frame.shape}"
    assert frame.dtype == np.uint8, f"{name} dtype should be uint8, got {frame.dtype}"


def assert_depth_valid(depth: np.ndarray, name: str = "depth"):
    """Assert that depth map is valid."""
    assert depth.ndim == 2, f"{name} should be 2D, got {depth.ndim}D"
    assert depth.dtype in [np.float32, np.float64], f"{name} dtype should be float, got {depth.dtype}"


def assert_shape_consistent(shape1: Tuple, shape2: Tuple, msg: str = ""):
    """Assert that two shapes match."""
    assert shape1 == shape2, f"Shape mismatch: {shape1} != {shape2}. {msg}"


# Export assertion helpers for use in tests
pytest.helpers = {
    'assert_alpha_valid': assert_alpha_valid,
    'assert_frame_valid': assert_frame_valid,
    'assert_depth_valid': assert_depth_valid,
    'assert_shape_consistent': assert_shape_consistent,
}


# ==============================================================================
# PARAMETRIZATION HELPERS
# ==============================================================================

# Common test parameter sets
FRAME_SIZES = [
    pytest.param((512, 512), id="512x512"),
    pytest.param((1024, 1024), id="1K"),
    pytest.param((1920, 1080), id="1080p"),
    pytest.param((3840, 2160), id="4K"),
]

SAM_MODELS = [
    pytest.param("tiny", id="tiny"),
    pytest.param("small", id="small"),
    pytest.param("base_plus", id="base_plus"),
    pytest.param("large", id="large"),
]

DEPTH_MODELS = [
    pytest.param("small", id="small"),
    pytest.param("base", id="base"),
    pytest.param("large", id="large"),
]

QUALITY_PRESETS = [
    pytest.param("draft", id="draft"),
    pytest.param("standard", id="standard"),
    pytest.param("high", id="high"),
    pytest.param("ultra", id="ultra"),
]
