# AUTO-ROTO V5: Comprehensive Testing Strategy & Analysis

## Executive Summary

The AUTO-ROTO project is a sophisticated multi-stage VFX rotoscoping pipeline with **zero existing test coverage**. This document provides a complete testing assessment and prioritized recommendations for building a robust, maintainable test suite.

**Project Size:** 10,347 lines of production code across 7 core modules
**Test Files Found:** 0 dedicated test modules
**Test Configuration:** No pytest.ini, setup.cfg, or test framework configuration
**Current Testing Approach:** Manual ad-hoc testing via test output directories

---

## 1. TESTING ASSESSMENT

### 1.1 Does the Project Have Test Files?

**Finding: NO - Zero test files exist**

- No `test_*.py` or `*_test.py` files in main project directory
- Test files found only in external dependencies (GroundingDINO, SAM2 demos)
- Test data directories exist (`test_input/`, `test_output/`, `refined_test/`) but contain only fixture data, not automated tests
- Current validation is manual inspection of output EXR/PNG files

### 1.2 Pytest/Test Configuration

**Finding: No test framework configuration**

- No `pytest.ini` file
- No `setup.cfg` with test configuration
- No `pyproject.toml` with pytest settings
- No `conftest.py` fixture setup
- Dependencies in `requirements.txt` do not include pytest, unittest, or testing frameworks

### 1.3 Assertion Patterns in Codebase

**Finding: Minimal defensive programming, inconsistent error handling**

#### Error Handling Observed:
```python
# auto_roto.py (8 error cases)
raise ValueError(f"Unsupported file format: {ext}")
raise IOError(f"Cannot open video: {self.path}")
raise ValueError(f"No image files found: {self.path}")
raise FileNotFoundError("GroundingDINO config not found")
raise RuntimeError("GroundingDINO not available")
raise ValueError("No prompt specified...")

# depth_refine.py (6 error cases)
raise IOError(f"Cannot read depth map: {depth_path}")
raise IOError(f"Cannot read depth: {filepath}")
# except ImportError / AttributeError blocks with logging only

# matte_combine.py (7 methods with validation)
# CoreMatte.extract() - validates alpha shape/range
# Blend mode enum with discrete type safety
```

#### Patterns by Module:

| Module | Lines | Try/Except | Raise | Data Validation | Assessment |
|--------|-------|-----------|-------|-----------------|------------|
| auto_roto.py | 1544 | 5 | 7 | Minimal | Entry point has basic I/O validation |
| depth_refine.py | 1615 | 6 | 3 | Weak | Sparse error handling, no shape validation |
| vitmatte_refine.py | 1575 | 4 | 2 | Weak | Limited input validation |
| edge_refine.py | 900 | 3 | 1 | Weak | Assumes valid data flow |
| temporal_smooth.py | 1036 | 4 | 0 | Minimal | Motion estimation unchecked |
| matte_combine.py | 1084 | 2 | 0 | Type enum | BlendMode validation only |
| full_pipeline_v5.py | 743 | 2 | 0 | Config dataclass | Orchestrator assumes stage outputs valid |

**Data Validation Gaps:**
- No shape consistency checks between pipeline stages
- No dtype validation (float32 vs float64, uint8, etc.)
- No range validation (alpha values should be 0-1, not outside)
- No null/None checks for optional parameters
- Motion vectors from optical flow unchecked for NaN/Inf
- EXR/image read operations don't validate channel counts

---

## 2. CRITICAL TEST CASES FOR VFX PIPELINE

### 2.1 Category 1: Core Segmentation (auto_roto.py)

**Purpose:** SAM2 video segmentation with temporal coherence

```python
# UNIT TESTS
test_frame_reader_mp4_valid()
    # Assert: Extracts all frames at correct dimensions
    # Assert: Frame count matches video duration
    # Assert: dtype is uint8, shape (H, W, 3)

test_frame_reader_png_sequence()
    # Assert: Reads PNG sequence in order
    # Assert: Handles numeric sorting (frame001, frame100)
    # Assert: Returns frames in ascending order

test_frame_reader_invalid_formats()
    # Assert: ValueError on .txt input
    # Assert: IOError on missing file
    # Assert: IOError on corrupted MP4

test_grounding_dino_text_prompt()
    # Assert: Returns valid bbox [x1, y1, x2, y2]
    # Assert: Box coordinates within frame bounds
    # Assert: Confidence scores 0-1

test_grounding_dino_multi_prompt()
    # Assert: "person.dog.car" returns 3 separate detections
    # Assert: No overlapping boxes (or expected overlap)
    # Assert: Consistent detection across similar frames

test_sam2_single_frame()
    # Assert: Generates binary mask from box prompt
    # Assert: Mask dtype is uint8, values {0, 1}
    # Assert: Mask shape matches frame shape

test_sam2_video_propagation()
    # Assert: Temporal coherence between frames
    # Assert: Object stays tracked despite occlusion
    # Assert: Mask boundary stays consistent

test_alpha_refinement_valid()
    # Assert: Refines binary mask to soft alpha (0-1)
    # Assert: Alpha range is [0.0, 1.0]
    # Assert: Soft edges present in hair/transparency

test_alpha_refinement_no_object()
    # Assert: Returns all-zero alpha for empty frame
    # Assert: Graceful handling of non-existent masks

# INTEGRATION TESTS
test_end_to_end_person_detection()
    # Setup: Load test video with person
    # Execute: Full SAM2 pipeline with text prompt
    # Assert: Alpha sequence generated
    # Assert: All frames have output files
    # Assert: Output directory structure correct

test_temporal_consistency()
    # Assert: Alpha values don't flicker frame-to-frame
    # Assert: Max delta between adjacent frames < threshold
```

### 2.2 Category 2: Depth Refinement (depth_refine.py)

**Purpose:** Depth Anything V3 edge-guided alpha refinement

```python
# UNIT TESTS
test_depth_estimator_initialization()
    # Assert: Model loads without CUDA errors
    # Assert: Device assignment (CPU/GPU) correct

test_depth_estimation_shape()
    # Setup: 1920x1080 RGB frame
    # Assert: Output depth map 1920x1080 (same resolution)
    # Assert: dtype is float32
    # Assert: Range approximately [0.0, 1.0] or [0, 255]

test_depth_edge_detection()
    # Setup: Synthetic frame with sharp edge
    # Assert: Depth edges detected at object boundary
    # Assert: Edge magnitude correlates with alpha discontinuity

test_depth_guided_alpha_refinement()
    # Setup: Binary mask + depth map
    # Assert: Refined alpha smoother than input
    # Assert: Fine details (hair) better defined
    # Assert: Alpha range still [0, 1]

test_depth_nested_model_architecture()
    # Assert: V3 nested encoder properly initialized
    # Assert: Multi-scale depth predictions merged correctly
    # Assert: Output resolution matches input

# INTEGRATION TESTS
test_depth_refinement_pipeline()
    # Setup: Alpha sequence + video
    # Execute: Full depth refinement
    # Assert: All frames processed
    # Assert: Output has same frame count as input

test_depth_quality_improvement()
    # Setup: Test frame with fine details (hair, lace)
    # Assert: Refined alpha smoother than coarse SAM2 mask
    # Assert: Detail preservation verified visually
```

### 2.3 Category 3: ViTMatte Refinement (vitmatte_refine.py)

**Purpose:** True alpha matting with Vision Transformer

```python
# UNIT TESTS
test_trimap_generation()
    # Setup: Binary mask
    # Assert: Trimap has 3 regions: 0 (background), 128 (unknown), 255 (foreground)
    # Assert: Unknown region width configurable
    # Assert: Regions don't overlap

test_trimap_adaptive_width()
    # Setup: Mask + motion estimates
    # Assert: Wider unknown region in high-motion areas
    # Assert: Narrow region in static areas
    # Assert: Width stays within configured bounds

test_vitmatte_model_inference()
    # Setup: RGB frame + trimap
    # Assert: Alpha output matches input shape
    # Assert: Alpha range [0, 1]
    # Assert: Inference completes in reasonable time

test_vitmatte_hair_detail()
    # Setup: Frame with visible hair/fur strands
    # Assert: Semi-transparent strands detected (0 < alpha < 1)
    # Assert: Not all-or-nothing binary values

test_vitmatte_transparency_preservation()
    # Setup: Glass/water with transparency
    # Assert: Semi-transparent regions correctly valued
    # Assert: Not collapsed to 0 or 1

# INTEGRATION TESTS
test_vitmatte_full_pipeline()
    # Setup: SAM2 masks + video frames
    # Execute: Adaptive trimap -> ViTMatte -> refined alpha
    # Assert: Output alpha smoother than SAM2 only
    # Assert: Hair/fine details preserved

test_vitmatte_performance()
    # Assert: Processes 24fps stream in real-time with optimization
    # Assert: Memory stable (no leak) across 1000+ frames
```

### 2.4 Category 4: Edge Refinement (edge_refine.py)

**Purpose:** Subpixel edge processing and despill

```python
# UNIT TESTS
test_edge_detection_sobel()
    # Setup: Synthetic sharp edge
    # Assert: Sobel correctly identifies edge pixels
    # Assert: Edge map gradient range appropriate

test_edge_detection_laplacian()
    # Setup: Smooth vs. sharp transitions
    # Assert: Sharp edges detected, smooth gradients ignored
    # Assert: Laplacian extrema at edge centers

test_guided_filter()
    # Setup: Noisy alpha map + guidance image
    # Assert: Smooths within same regions
    # Assert: Preserves guidance boundaries
    # Assert: Radius parameter affects smoothness

test_color_keying()
    # Setup: Frame with spill color
    # Assert: Spill regions identified by hue similarity
    # Assert: Despill strength configurable
    # Assert: Doesn't affect on-object colors

test_subpixel_edge_refinement()
    # Setup: Alpha with hard edge
    # Assert: Edge antialiasing applied
    # Assert: Gradient smoothness improved
    # Assert: Subpixel accuracy verified

# INTEGRATION TESTS
test_edge_refinement_sequence()
    # Setup: Alpha sequence
    # Execute: Full edge refinement pipeline
    # Assert: All frames processed consistently
    # Assert: No artifacts between frames
```

### 2.5 Category 5: Temporal Coherence (temporal_smooth.py)

**Purpose:** Optical flow and anti-flicker

```python
# UNIT TESTS
test_optical_flow_computation()
    # Setup: Two adjacent frames
    # Assert: Flow map same size as frames
    # Assert: Flow vectors reasonable magnitude
    # Assert: No NaN/Inf values

test_optical_flow_quality_presets()
    # Assert: low quality faster than high
    # Assert: high quality produces smoother flow
    # Assert: All presets produce valid output

test_motion_warping()
    # Setup: Alpha map + optical flow
    # Assert: Warped alpha same shape as input
    # Assert: Warping boundary handled correctly
    # Assert: No black/undefined regions

test_temporal_median_filter()
    # Setup: Alpha sequence with frame [50ms spike, 51ms spike]
    # Assert: Spikes removed by median filter
    # Assert: Smooth frames unchanged
    # Assert: Window size affects smoothness

test_edge_confidence_propagation()
    # Setup: Confidence map + edge positions
    # Assert: High-confidence edges preserved
    # Assert: Low-confidence edges smoothed
    # Assert: Confidence values in [0, 1]

test_keyframe_anchoring()
    # Setup: Alpha sequence with keyframes marked
    # Assert: Keyframes never modified
    # Assert: Interpolation between keyframes works
    # Assert: No long-term drift in values

test_flicker_suppression()
    # Setup: Flickering alpha sequence (frame delta > 0.1)
    # Assert: Frame deltas reduced below threshold
    # Assert: Suppression strength adjustable
    # Assert: Motion preserved (not over-smoothed)

# INTEGRATION TESTS
test_temporal_smoothing_sequence()
    # Setup: Flickering alpha sequence
    # Execute: Full temporal smoothing pipeline
    # Assert: Output smoother than input
    # Assert: Frame count unchanged
    # Assert: No frames dropped

test_motion_robustness()
    # Setup: Video with fast motion/cuts
    # Assert: Handles motion boundaries without artifacts
    # Assert: Keyframe anchoring prevents drift
```

### 2.6 Category 6: Matte Combination (matte_combine.py)

**Purpose:** Multi-layer matte compositing and despill

```python
# UNIT TESTS
test_core_matte_extraction()
    # Setup: Soft alpha map [0, 1]
    # Assert: Core eroded correctly
    # Assert: Core is 100% solid (all 1s) except boundary
    # Assert: Erosion depth configurable

test_core_matte_feathering()
    # Setup: Core matte with hard boundary
    # Assert: Feather applied at boundary
    # Assert: Smooth transition from core to unknown
    # Assert: Feather width configurable

test_detail_matte_blending()
    # Setup: Core matte + detail matte + blend strength
    # Assert: Output is weighted blend
    # Assert: Blend strength 0 -> core only
    # Assert: Blend strength 1 -> detail matte

test_soft_edge_generation()
    # Setup: Alpha map
    # Assert: Soft edge band identified
    # Assert: Width configurable
    # Assert: Gamma curve applied correctly

test_blend_modes_over()
    # Setup: Two alphas + OVER blend mode
    # Assert: Follows Porter-Duff OVER semantics
    # Assert: Result range [0, 1]

test_blend_modes_under()
    # Assert: UNDER is reverse of OVER
    # Assert: Commutative with alpha values swapped

test_blend_modes_max_min():
    # Setup: Two alphas
    # Assert: MAX returns max(a1, a2)
    # Assert: MIN returns min(a1, a2)

test_premultiply_conversion()
    # Setup: Straight alpha + color
    # Assert: Premultiply: RGB *= alpha
    # Assert: Unpremultiply: RGB /= alpha (no divide by zero)
    # Assert: Reversible (multiply then divide -> original)

test_despill_complementary()
    # Setup: Frame with green spill
    # Assert: Green suppressed from edges
    # Assert: Red/blue preserved
    # Assert: Strength adjustable

test_linear_workflow()
    # Setup: sRGB frame + config linear_workflow=True
    # Assert: Converted to linear space for math
    # Assert: Converted back to sRGB for output
    # Assert: Colors match expected after round-trip

# INTEGRATION TESTS
test_matte_combination_pipeline()
    # Setup: Multiple matte layers
    # Execute: Full combination pipeline
    # Assert: Output single alpha map
    # Assert: Range [0, 1]

test_despill_on_real_footage()
    # Setup: Video with color spill
    # Assert: Spill significantly reduced
    # Assert: Original colors preserved

test_premultiply_preservation()
    # Setup: Full RGBA + premultiply config
    # Assert: Alpha channel preserved exactly
    # Assert: Color channel properly premultiplied
```

### 2.7 Category 7: Full Pipeline (full_pipeline_v5.py)

**Purpose:** Orchestration and stage integration

```python
# INTEGRATION TESTS
test_full_pipeline_draft_quality()
    # Setup: Short test video
    # Execute: full_pipeline_v5.py with quality=draft
    # Assert: All stages complete
    # Assert: Final alpha sequence present
    # Assert: Execution time < 5 minutes

test_full_pipeline_standard_quality()
    # Setup: Standard test video
    # Execute: full_pipeline_v5.py with quality=standard
    # Assert: All 6 stages complete
    # Assert: Quality better than draft
    # Assert: Execution time reasonable for production

test_full_pipeline_high_quality()
    # Setup: High-quality test footage
    # Execute: full_pipeline_v5.py with quality=high
    # Assert: Final output superior to standard
    # Assert: All stages use larger models

test_pipeline_skip_stages()
    # Execute: Skip depth, ViTMatte, edge, temporal separately
    # Assert: Remaining stages work correctly
    # Assert: Output valid but lower quality when skipped

test_pipeline_keep_intermediate()
    # Execute: full_pipeline_v5.py --keep-intermediate
    # Assert: Intermediate outputs preserved
    # Assert: Can inspect per-stage results

test_pipeline_stage_failure_handling()
    # Setup: Invalid input or corrupted intermediate
    # Assert: Graceful error messages
    # Assert: Cleanup temporary files
    # Assert: Partial output saved if possible

test_pipeline_memory_stability()
    # Setup: 2000-frame sequence
    # Assert: Memory doesn't grow unbounded
    # Assert: Per-frame garbage collection works
    # Assert: Model inference cached efficiently

test_pipeline_output_structure()
    # Execute: Full pipeline
    # Assert: output/alpha/ contains EXR sequence
    # Assert: output/preview/ contains JPG previews
    # Assert: output/depth/ contains depth maps (if requested)
    # Assert: All files properly named

test_pipeline_batch_processing()
    # Setup: 5 different input videos
    # Execute: batch_roto.py on all
    # Assert: All complete successfully
    # Assert: Outputs don't interfere
    # Assert: Memory released between jobs

# DATA VALIDATION TESTS
test_dtype_consistency_across_pipeline()
    # Assert: SAM2 output uint8 binary mask
    # Assert: Alpha refinement produces float32 [0, 1]
    # Assert: Depth maps are float32
    # Assert: Final EXR export preserves dtype

test_shape_consistency_across_pipeline()
    # Setup: 1920x1080 input
    # Assert: All intermediate outputs 1920x1080
    # Assert: No unexpected downsampling/upsampling
    # Assert: Batch dimension consistent

test_temporal_shape_consistency()
    # Setup: 300-frame input
    # Assert: All outputs have 300 frames
    # Assert: Frame order preserved
    # Assert: No dropped/duplicated frames
```

---

## 3. TESTING GAPS & RISKS

### 3.1 Critical Gaps (High Priority)

| Gap | Risk | Impact | Difficulty |
|-----|------|--------|------------|
| **No input validation** | Corrupted data silently propagates | Pipeline fails midway, user can't debug | Medium |
| **No shape/dtype checks** | Dimension mismatches between stages | Matrix operation failures, cryptic errors | Low |
| **No output verification** | Bad alphas shipped to VFX artists | Unusable footage, rework required | Low |
| **No temporal validation** | Frame count mismatches between stages | Missing frames in output sequence | Medium |
| **No memory testing** | Memory leaks on long sequences | OOM crashes on feature films | High |
| **No performance baselines** | Regression goes undetected | Processing becomes slower silently | Medium |
| **No error recovery** | Crashes lose all intermediate work | Restart entire pipeline from scratch | High |

### 3.2 Gap by Module

**auto_roto.py (Entry Point)**
- No validation that prompt returns valid detections
- No check for empty SAM2 masks
- No handling of partially initialized SAM2 (model corrupted)
- No timeouts on inference (hung process)

**depth_refine.py**
- No shape compatibility check (depth vs. alpha dimensions)
- Assumes depth model initialization succeeds silently
- No handling of depth NaN regions
- No gradient validation (depth might be constant/flat)

**vitmatte_refine.py**
- No validation that trimap has correct regions
- Assumes ViTMatte model always converges
- No check for alpha outside [0, 1]
- No detection of failed inference (all-zero alpha)

**edge_refine.py**
- No validation of edge detection thresholds
- Assumes guided filter produces valid output
- No check for color space consistency
- No despill validation (color channels might go negative)

**temporal_smooth.py**
- **Biggest gap: Optical flow unchecked for NaN/Inf**
- No validation of motion magnitude (might be unrealistic)
- No keyframe integrity checks
- No detection of flow failure (zero flow everywhere)

**matte_combine.py**
- No shape validation for blend operations
- No check for premultiply division by zero
- No validation of color range after linear conversion
- No detection of invalid blend mode

**full_pipeline_v5.py**
- No stage output validation before feeding to next stage
- No check that intermediate directories exist
- No cleanup on failure (orphaned temp files)
- No progress persistence (can't resume)

---

## 4. TESTING IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Set up test infrastructure and unit test framework

```bash
# Step 1: Install test framework
pip install pytest pytest-cov pytest-xdist pytest-timeout

# Step 2: Create pytest configuration
# pytest.ini or pyproject.toml

# Step 3: Create test fixtures (test_fixtures.py)
# - Test data (small video, images)
# - Mock models
# - Sample configs

# Step 4: Write unit tests for utility functions
# - Frame reading/writing
# - I/O operations
# - Data format conversions
```

**Deliverables:**
- pytest.ini with config
- conftest.py with shared fixtures
- test_utils.py (50 tests, ~8 hours)

### Phase 2: Core Module Tests (Weeks 3-5)

**Goal:** 70% unit test coverage for 6 core modules

```bash
# test_auto_roto.py (100 tests)
# test_depth_refine.py (80 tests)
# test_vitmatte_refine.py (70 tests)
# test_edge_refine.py (60 tests)
# test_temporal_smooth.py (80 tests)
# test_matte_combine.py (90 tests)
```

**Deliverables:**
- Unit tests for all major functions
- Mocked deep learning models (inference)
- Parametrized tests for quality presets
- ~480 tests total

### Phase 3: Integration Tests (Weeks 6-8)

**Goal:** End-to-end pipeline validation

```bash
# test_integration_pipeline.py
# - test_full_pipeline_draft_quality()
# - test_full_pipeline_standard_quality()
# - test_temporal_consistency()
# - test_batch_processing()
# - test_memory_stability()
```

**Deliverables:**
- Full pipeline integration tests (50 tests)
- Test fixtures with real video data
- CI/CD integration

### Phase 4: Performance & Regression (Weeks 9-10)

**Goal:** Establish baselines and prevent regressions

```bash
# test_performance.py
# - Benchmark SAM2 inference time
# - Measure peak memory usage
# - Track alpha quality metrics

# test_regression.py
# - Fixed test cases for known edge cases
# - Before/after comparison
```

**Deliverables:**
- Performance benchmarks
- Regression test suite
- CI performance tracking

---

## 5. RECOMMENDED TEST COVERAGE PRIORITIES

### Priority Tier 1: Critical Path (Week 1-2)

**Why:** These tests prevent catastrophic failures

```python
# File: tests/test_data_validation.py

def test_frame_shape_validation():
    """Ensure consistent dimensions across pipeline."""
    # All outputs must match input resolution
    # Prevent dimension mismatch bugs

def test_alpha_range_validation():
    """Verify alpha stays in [0, 1]."""
    # Catch NaN, Inf, out-of-range values
    # Most common issue in matting pipelines

def test_frame_count_consistency():
    """Ensure no frames dropped/duplicated."""
    # Temporal sequences must stay synchronized

def test_dtype_consistency():
    """Binary masks, alpha, depth dtypes correct."""
    # uint8 vs float32 bugs cause silent failures

def test_pipeline_stage_failures():
    """Handle graceful errors at each stage."""
    # Don't cascade failures
```

**Estimated Coverage Impact:** 40% critical bugs caught

### Priority Tier 2: Quality Validation (Week 3-4)

**Why:** These tests ensure output quality, not just validity

```python
# File: tests/test_quality_metrics.py

def test_alpha_edge_quality():
    """Edges are antialiased, not binary."""
    # Visual inspection of soft edges
    # Use SSIM or gradient comparison

def test_temporal_stability():
    """No frame-to-frame flicker."""
    # Max delta < threshold between frames
    # Most complained-about artifact by artists

def test_hair_detail_preservation():
    """Fine details not lost in refinement."""
    # ViTMatte specifically improves hair
    # Verify with feature detection

def test_despill_effectiveness():
    """Spill color suppressed without color shifts."""
    # Check hue preservation
    # Saturation reduction in spill regions

def test_memory_stability():
    """No memory leaks on long sequences."""
    # Monitor RSS memory over 1000+ frames
    # Critical for feature film processing
```

**Estimated Coverage Impact:** 35% quality issues caught

### Priority Tier 3: Edge Cases (Week 5-6)

**Why:** Handle unusual but valid inputs gracefully

```python
# File: tests/test_edge_cases.py

def test_fast_motion():
    """Optical flow with extreme motion."""
    # Check for motion vector artifacts

def test_occlusion_and_reveal():
    """Object partially hidden then revealed."""
    # SAM2 temporal memory stability

def test_small_objects():
    """Objects smaller than kernel size."""
    # Erosion/dilation edge cases

def test_transparency_objects():
    """Glass, water, semi-transparent surfaces."""
    # ViTMatte trimap accuracy

def test_single_frame():
    """Input of exactly 1 frame."""
    # No temporal operations available

def test_extreme_resolutions():
    """4K, 8K, non-standard aspect ratios."""
    # Memory and inference scaling

def test_color_space_edge_cases():
    """Extreme colors, very dark/bright frames."""
    # Depth estimation robustness
```

**Estimated Coverage Impact:** 15% edge case bugs caught

### Priority Tier 4: Performance (Week 7-8)

**Why:** Prevent regressions and set expectations

```python
# File: tests/test_performance.py

def test_inference_speed_baselines():
    """SAM2, ViTMatte, Depth inference times."""
    # Establish baseline for each model size

def test_memory_peak_usage():
    """Peak memory per quality preset."""
    # Warn if crosses expected thresholds

def test_throughput_fps():
    """Frames per second for full pipeline."""
    # Track regressions in optimization

def test_batch_parallelization():
    """Speedup with multi-GPU processing."""
    # Ensure scaling is efficient
```

**Estimated Coverage Impact:** 10% performance regressions caught

---

## 6. TEST INFRASTRUCTURE SETUP

### 6.1 Pytest Configuration

**File: C:/Users/nicho/Documents/auto_roto_scripts/auto_roto_v2/auto_roto/pytest.ini**

```ini
[pytest]
# Directory containing tests
testpaths = tests

# Minimum Python version
minversion = 7.0

# Output options
addopts =
    -v
    --tb=short
    --strict-markers
    --cov=.
    --cov-report=html
    --cov-report=term-missing
    --cov-report=xml
    -n auto
    --timeout=300

# Test discovery patterns
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers for test categorization
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, multiple modules)
    slow: Tests taking > 10 seconds
    performance: Performance benchmarks
    quality: Output quality validation
    gpu: Tests requiring GPU
    memory: Memory stability tests
```

### 6.2 Conftest Fixtures

**File: C:/Users/nicho/Documents/auto_roto_scripts/auto_roto_v2/auto_roto/tests/conftest.py**

```python
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for each test."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture
def sample_frame():
    """Generate a 512x512 test frame."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

@pytest.fixture
def sample_alpha():
    """Generate a 512x512 test alpha map."""
    return np.random.rand(512, 512).astype(np.float32)

@pytest.fixture
def sample_depth():
    """Generate a 512x512 depth map."""
    return np.random.rand(512, 512).astype(np.float32)

@pytest.fixture(scope="session")
def test_video_path():
    """Path to test video (must exist)."""
    return Path("test_input/sample.mp4")

@pytest.fixture
def mock_depth_model(monkeypatch):
    """Mock Depth Anything V3 inference."""
    def mock_estimate(frame):
        return np.random.rand(frame.shape[0], frame.shape[1]).astype(np.float32)
    return mock_estimate
```

### 6.3 Test Data Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_data/
│   ├── frame_512x512.png    # Test frame
│   ├── video_100frames.mp4  # Short test video
│   ├── masks/
│   │   ├── frame_001.png    # SAM2 output
│   │   └── ...
│   └── expected_outputs/    # Golden reference files
├── test_auto_roto.py
├── test_depth_refine.py
├── test_vitmatte_refine.py
├── test_edge_refine.py
├── test_temporal_smooth.py
├── test_matte_combine.py
├── test_full_pipeline_v5.py
├── test_integration.py
└── test_performance.py
```

---

## 7. SPECIFIC IMPLEMENTATION EXAMPLES

### 7.1 Unit Test Example: Shape Validation

```python
# tests/test_data_validation.py

import pytest
import numpy as np
from auto_roto import FrameReader
from depth_refine import DepthEstimator, DepthGuidedRefiner

class TestShapeConsistency:
    """Test that shapes stay consistent across pipeline."""

    def test_frame_reader_output_shape(self, sample_video_path):
        """FrameReader outputs consistent shape."""
        reader = FrameReader(sample_video_path)
        frames = list(reader.get_frames())

        # All frames same shape
        expected_shape = (1080, 1920, 3)
        for i, frame in enumerate(frames):
            assert frame.shape == expected_shape, \
                f"Frame {i} shape {frame.shape} != {expected_shape}"

    def test_alpha_refinement_preserves_shape(self, sample_alpha):
        """Alpha refinement doesn't change dimensions."""
        from auto_roto import AlphaRefiner

        H, W = sample_alpha.shape
        refiner = AlphaRefiner()
        refined = refiner.refine(sample_alpha)

        assert refined.shape == (H, W), \
            f"Output shape {refined.shape} != input {(H, W)}"

    @pytest.mark.parametrize("model_size", ["small", "base", "large"])
    def test_depth_output_matches_input_resolution(self, model_size, sample_frame):
        """Depth Anything preserves input resolution."""
        estimator = DepthEstimator(model_size)
        depth = estimator.estimate(sample_frame)

        expected_shape = sample_frame.shape[:2]  # H, W
        assert depth.shape == expected_shape, \
            f"Depth {depth.shape} != frame {sample_frame.shape[:2]}"

    def test_temporal_smoothing_preserves_sequence_length(self, sample_alpha_sequence):
        """Temporal smoothing returns same number of frames."""
        from temporal_smooth import TemporalSmoother

        n_frames = len(sample_alpha_sequence)
        smoother = TemporalSmoother()
        smoothed = smoother.smooth(sample_alpha_sequence)

        assert len(smoothed) == n_frames, \
            f"Output {len(smoothed)} frames != input {n_frames}"
```

### 7.2 Integration Test Example: Full Pipeline

```python
# tests/test_full_pipeline_v5.py

import pytest
from pathlib import Path
from full_pipeline_v5 import FullPipelineV5, PipelineConfig

class TestFullPipelineV5:
    """Test the complete V5 pipeline."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_draft_quality(self, test_video_path, temp_output_dir):
        """Full pipeline executes successfully in draft mode."""
        config = PipelineConfig(
            input_path=str(test_video_path),
            output_dir=temp_output_dir,
            prompt="person",
            quality="draft"
        )

        pipeline = FullPipelineV5(config)
        results = pipeline.run()

        # Verify outputs exist
        alpha_dir = Path(temp_output_dir) / "alpha"
        assert alpha_dir.exists(), "Alpha output directory missing"

        alpha_files = list(alpha_dir.glob("*.exr"))
        assert len(alpha_files) > 0, "No alpha output files generated"

    @pytest.mark.integration
    def test_pipeline_stage_outputs_valid_shapes(self, test_video_path, temp_output_dir):
        """Each stage produces correctly shaped outputs."""
        from full_pipeline_v5 import FullPipelineV5

        config = PipelineConfig(
            input_path=str(test_video_path),
            output_dir=temp_output_dir,
            quality="standard"
        )

        pipeline = FullPipelineV5(config)

        # Run just SAM2
        sam2_output = pipeline.run_stage_sam2()
        assert sam2_output is not None

        # Verify shape consistency
        frame_h, frame_w = pipeline.frame_dimensions
        for i, alpha in enumerate(sam2_output):
            assert alpha.shape == (frame_h, frame_w), \
                f"SAM2 frame {i} wrong shape: {alpha.shape}"

    @pytest.mark.memory
    @pytest.mark.slow
    def test_pipeline_memory_stability(self, long_video_path, temp_output_dir):
        """Pipeline doesn't leak memory on long sequences."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        config = PipelineConfig(
            input_path=str(long_video_path),  # 1000 frames
            output_dir=temp_output_dir,
            quality="draft",
            skip_vitmatte=True,  # Reduce memory
            skip_temporal=True
        )

        pipeline = FullPipelineV5(config)

        # Track memory over execution
        memory_samples = []

        def on_frame(frame_idx):
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_samples.append(memory_mb)

        pipeline.run(on_frame_callback=on_frame)

        # Memory shouldn't grow linearly
        early_avg = np.mean(memory_samples[:100])
        late_avg = np.mean(memory_samples[-100:])
        growth_pct = (late_avg - early_avg) / early_avg * 100

        assert growth_pct < 20, \
            f"Memory grew {growth_pct}% (early: {early_avg}MB, late: {late_avg}MB)"
```

### 7.3 Quality Test Example: Temporal Stability

```python
# tests/test_quality_metrics.py

import pytest
import numpy as np
from typing import List

class TestTemporalStability:
    """Test temporal consistency (no flicker)."""

    def test_alpha_frame_delta_within_threshold(self, alpha_sequence: List[np.ndarray]):
        """Adjacent frames don't differ too much."""
        max_delta_threshold = 0.1  # 10% max change

        for i in range(len(alpha_sequence) - 1):
            curr_alpha = alpha_sequence[i]
            next_alpha = alpha_sequence[i+1]

            # Per-pixel delta
            delta = np.abs(curr_alpha - next_alpha)
            max_delta = np.max(delta)

            assert max_delta < max_delta_threshold, \
                f"Frame {i}->{i+1} delta {max_delta:.3f} exceeds threshold {max_delta_threshold}"

    def test_temporal_smoothing_reduces_flicker(self, flickering_sequence, smoother):
        """Temporal smoothing reduces frame deltas."""
        smoothed = smoother.smooth(flickering_sequence)

        # Compute average frame delta (flicker metric)
        orig_deltas = []
        for i in range(len(flickering_sequence) - 1):
            delta = np.mean(np.abs(
                flickering_sequence[i] - flickering_sequence[i+1]
            ))
            orig_deltas.append(delta)

        smoothed_deltas = []
        for i in range(len(smoothed) - 1):
            delta = np.mean(np.abs(smoothed[i] - smoothed[i+1]))
            smoothed_deltas.append(delta)

        # Smoothing must reduce flicker
        orig_avg = np.mean(orig_deltas)
        smooth_avg = np.mean(smoothed_deltas)

        assert smooth_avg < orig_avg, \
            f"Smoothing increased flicker: {orig_avg:.3f} -> {smooth_avg:.3f}"

    @pytest.mark.quality
    def test_hair_detail_preserved_after_refinement(self, frame_with_hair, coarse_alpha, refined_alpha):
        """ViTMatte refinement preserves fine hair details."""
        # Use high-pass filter to detect fine details
        from scipy import ndimage

        # Details: areas where coarse and refined differ
        detail_regions = np.abs(coarse_alpha - refined_alpha) > 0.05

        # Check that refined alpha has semi-transparent values in detail regions
        detail_alpha_values = refined_alpha[detail_regions]

        # Should have range, not all 0 or 1
        has_semi_transparent = np.any((detail_alpha_values > 0.1) & (detail_alpha_values < 0.9))
        assert has_semi_transparent, "No semi-transparent values in refined hair regions"
```

---

## 8. CONTINUOUS INTEGRATION SETUP

### 8.1 GitHub Actions Workflow

**File: .github/workflows/tests.yml**

```yaml
name: Tests

on:
  push:
    branches: [master, develop]
  pull_request:
    branches: [master]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov pytest-xdist pytest-timeout

      - name: Run unit tests
        run: |
          pytest tests/ -m "unit" -v --cov --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

      - name: Run integration tests
        if: github.event_name == 'pull_request'
        run: |
          pytest tests/ -m "integration" -v
```

---

## 9. SUMMARY TABLE: Testing Gaps by Module

| Module | Lines | Current Coverage | Critical Gaps | Priority Tests | Est. Time |
|--------|-------|------------------|--------------|----------------|-----------|
| auto_roto.py | 1544 | 0% | I/O validation, SAM2 checks, prompt validation | 60 tests | 16h |
| depth_refine.py | 1615 | 0% | Shape validation, depth quality, edge detection | 70 tests | 18h |
| vitmatte_refine.py | 1575 | 0% | Trimap validation, alpha range, detail preservation | 60 tests | 16h |
| edge_refine.py | 900 | 0% | Edge quality, despill validation, blur amount | 50 tests | 14h |
| temporal_smooth.py | 1036 | 0% | Optical flow validation, flicker suppression | 70 tests | 18h |
| matte_combine.py | 1084 | 0% | Blend mode correctness, premultiply precision | 60 tests | 16h |
| full_pipeline_v5.py | 743 | 0% | Stage output validation, memory stability | 40 tests | 12h |
| **TOTAL** | **10,347** | **0%** | 7 critical categories | **410 tests** | **110h** |

**Estimated effort breakdown:**
- Infrastructure setup: 16h
- Unit tests (Tier 1 & 2): 80h
- Integration tests: 20h
- Performance tests: 8h
- Documentation: 6h
- **Total: ~130 hours (~3-4 weeks with 1 FTE)**

---

## 10. NEXT STEPS

1. **Week 1:** Install pytest, create pytest.ini, set up conftest.py fixtures
2. **Week 2:** Create 50 data validation tests (highest priority)
3. **Week 3-4:** Create 250 unit tests for 6 core modules
4. **Week 5-6:** Create 100 integration tests for full pipeline
5. **Week 7:** Set up CI/CD with GitHub Actions
6. **Ongoing:** Maintain 70%+ coverage, add tests with each bug fix

---

## Appendix: Test File Template

```python
# tests/test_module_name.py
"""
Unit tests for module_name.py

Test organization:
- TestDataValidation: Input/output shape, dtype, range
- TestFunctionality: Core logic correctness
- TestEdgeCases: Unusual but valid inputs
- TestIntegration: Multi-function workflows
"""

import pytest
import numpy as np
from pathlib import Path
from module_name import MyClass, my_function

class TestDataValidation:
    """Test input/output data contract."""

    def test_input_shape_validation(self):
        """Function validates input shape."""
        pass

    def test_output_dtype_consistency(self):
        """Output dtype matches specification."""
        pass

class TestFunctionality:
    """Test core logic."""

    @pytest.mark.parametrize("input,expected", [
        (np.zeros((100, 100)), np.zeros((100, 100))),
        # More cases...
    ])
    def test_identity_on_zeros(self, input, expected):
        """Function returns expected output."""
        result = my_function(input)
        np.testing.assert_array_almost_equal(result, expected)

class TestEdgeCases:
    """Test boundary conditions."""

    def test_single_pixel_input(self):
        """Function handles 1x1 input."""
        pass

    def test_very_large_input(self):
        """Function handles 4K input without overflow."""
        pass

class TestIntegration:
    """Test with adjacent modules."""

    @pytest.mark.integration
    def test_with_upstream_output(self, upstream_fixture):
        """Function works with output from previous stage."""
        pass
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-23
**Coverage Assessment:** Comprehensive testing gap analysis complete
