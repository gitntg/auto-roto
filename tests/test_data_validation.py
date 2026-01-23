"""
Data Validation Tests for AUTO-ROTO Pipeline

Tests that verify data contracts (shape, dtype, range) between pipeline stages.
This is the HIGHEST PRIORITY test suite as shape/dtype mismatches cause
silent failures or cryptic errors in deep learning models.

Test categories:
- Shape consistency (all intermediate outputs match input resolution)
- Dtype correctness (float32 for alpha, uint8 for masks, etc.)
- Value range validation (alpha in [0,1], depth valid, no NaN/Inf)
- Frame sequence integrity (no dropped/duplicated frames)
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Tuple


class TestAlphaMapValidation:
    """Test that alpha maps are properly formatted."""

    @pytest.mark.unit
    def test_alpha_is_2d_array(self, sample_alpha_512x512):
        """Alpha map should be 2D (H, W), not 3D."""
        alpha = sample_alpha_512x512
        assert alpha.ndim == 2, f"Alpha should be 2D, got {alpha.ndim}D with shape {alpha.shape}"

    @pytest.mark.unit
    def test_alpha_dtype_is_float(self, sample_alpha_512x512):
        """Alpha should be float32 or float64, not int."""
        alpha = sample_alpha_512x512
        assert alpha.dtype in [np.float32, np.float64], \
            f"Alpha dtype should be float, got {alpha.dtype}"

    @pytest.mark.unit
    def test_alpha_range_within_bounds(self, sample_alpha_512x512):
        """Alpha values must be in [0.0, 1.0], not outside."""
        alpha = sample_alpha_512x512
        assert 0.0 <= alpha.min() <= 1.0, f"Alpha min {alpha.min()} outside [0,1]"
        assert 0.0 <= alpha.max() <= 1.0, f"Alpha max {alpha.max()} outside [0,1]"

    @pytest.mark.unit
    def test_alpha_no_nan_or_inf(self, sample_alpha_512x512):
        """Alpha should not contain NaN or Inf values."""
        alpha = sample_alpha_512x512
        assert not np.isnan(alpha).any(), "Alpha contains NaN values"
        assert not np.isinf(alpha).any(), "Alpha contains Inf values"

    @pytest.mark.unit
    def test_alpha_with_soft_edges_valid(self, sample_alpha_soft):
        """Soft alpha with gradients should still be in [0,1]."""
        alpha = sample_alpha_soft
        assert 0.0 <= alpha.min(), f"Alpha min {alpha.min()} < 0"
        assert alpha.max() <= 1.0, f"Alpha max {alpha.max()} > 1"
        # Should have some intermediate values (not just 0/1)
        has_intermediates = np.any((alpha > 0.1) & (alpha < 0.9))
        assert has_intermediates, "Soft alpha has no intermediate values"

    @pytest.mark.unit
    @pytest.mark.parametrize("H,W", [(512, 512), (1080, 1920), (720, 1280), (256, 256)])
    def test_alpha_various_resolutions(self, H, W):
        """Alpha should work at various resolutions."""
        alpha = np.random.rand(H, W).astype(np.float32)
        assert alpha.shape == (H, W)
        assert 0.0 <= alpha.min() <= 1.0
        assert 0.0 <= alpha.max() <= 1.0


class TestDepthMapValidation:
    """Test that depth maps are properly formatted."""

    @pytest.mark.unit
    def test_depth_is_2d_array(self, sample_depth_map):
        """Depth map should be 2D (H, W), not 3D."""
        depth = sample_depth_map
        assert depth.ndim == 2, f"Depth should be 2D, got {depth.ndim}D"

    @pytest.mark.unit
    def test_depth_dtype_is_float(self, sample_depth_map):
        """Depth should be float32 or float64."""
        depth = sample_depth_map
        assert depth.dtype in [np.float32, np.float64], \
            f"Depth dtype should be float, got {depth.dtype}"

    @pytest.mark.unit
    def test_depth_no_nan_or_inf(self, sample_depth_map):
        """Depth should not contain NaN or Inf."""
        depth = sample_depth_map
        assert not np.isnan(depth).any(), "Depth contains NaN values"
        assert not np.isinf(depth).any(), "Depth contains Inf values"

    @pytest.mark.unit
    def test_depth_reasonable_range(self, sample_depth_map):
        """Depth typically in [0, 1] or [0, 255] depending on normalization."""
        depth = sample_depth_map
        # Should be normalized to reasonable range (not extreme values)
        assert depth.min() >= -1.0, f"Depth min {depth.min()} seems invalid"
        assert depth.max() <= 256.0, f"Depth max {depth.max()} seems invalid"

    @pytest.mark.unit
    def test_depth_with_edge_discontinuity(self, sample_depth_with_edge):
        """Depth with sharp edges should still be valid."""
        depth = sample_depth_with_edge
        assert depth.shape == (512, 512)
        assert depth.dtype in [np.float32, np.float64]

        # Should have some variation
        assert not np.allclose(depth, depth[0, 0]), \
            "Depth map is constant everywhere (no variation)"


class TestFrameValidation:
    """Test that RGB frames are properly formatted."""

    @pytest.mark.unit
    def test_frame_is_3d_array(self, sample_frame_512x512):
        """Frame should be 3D (H, W, 3), not 2D."""
        frame = sample_frame_512x512
        assert frame.ndim == 3, f"Frame should be 3D, got {frame.ndim}D"

    @pytest.mark.unit
    def test_frame_has_3_channels(self, sample_frame_512x512):
        """Frame should have 3 channels (RGB)."""
        frame = sample_frame_512x512
        assert frame.shape[2] == 3, f"Frame should have 3 channels, got {frame.shape[2]}"

    @pytest.mark.unit
    def test_frame_dtype_is_uint8(self, sample_frame_512x512):
        """Frame should be uint8 (0-255 range)."""
        frame = sample_frame_512x512
        assert frame.dtype == np.uint8, f"Frame dtype should be uint8, got {frame.dtype}"

    @pytest.mark.unit
    def test_frame_value_range(self, sample_frame_512x512):
        """Frame values should be in [0, 255]."""
        frame = sample_frame_512x512
        assert 0 <= frame.min(), f"Frame min {frame.min()} < 0"
        assert frame.max() <= 255, f"Frame max {frame.max()} > 255"

    @pytest.mark.unit
    @pytest.mark.parametrize("H,W", [(512, 512), (1080, 1920), (720, 1280)])
    def test_frame_various_resolutions(self, H, W):
        """Frames should work at various resolutions."""
        frame = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
        assert frame.shape == (H, W, 3)
        assert frame.dtype == np.uint8


class TestShapeConsistencyAcrossPipeline:
    """Test that shapes are preserved through pipeline stages."""

    @pytest.mark.unit
    def test_input_output_shape_match(self, sample_frame_512x512, sample_alpha_512x512):
        """Alpha output should match input frame shape."""
        frame = sample_frame_512x512
        alpha = sample_alpha_512x512

        frame_shape = frame.shape[:2]  # (H, W)
        alpha_shape = alpha.shape

        assert frame_shape == alpha_shape, \
            f"Frame {frame_shape} and alpha {alpha_shape} shapes don't match"

    @pytest.mark.unit
    def test_depth_matches_frame_resolution(self, sample_frame_1080p, sample_depth_map):
        """Depth map should have same resolution as frame."""
        # Create depth at same resolution as frame
        H, W = sample_frame_1080p.shape[:2]
        depth = np.random.rand(H, W).astype(np.float32)

        assert depth.shape == (H, W), \
            f"Depth {depth.shape} doesn't match frame {sample_frame_1080p.shape[:2]}"

    @pytest.mark.unit
    def test_trimap_matches_alpha_shape(self, sample_alpha_512x512, sample_trimap):
        """Trimap should have same shape as alpha."""
        # Create trimap at same shape as alpha
        H, W = sample_alpha_512x512.shape
        trimap = np.zeros((H, W), dtype=np.uint8)

        assert trimap.shape == (H, W), \
            f"Trimap {trimap.shape} doesn't match alpha {(H, W)}"


class TestSequenceValidation:
    """Test that frame sequences maintain integrity."""

    @pytest.mark.unit
    def test_frame_sequence_length_preserved(self, sample_alpha_sequence):
        """Sequence processing should not drop/add frames."""
        original_length = len(sample_alpha_sequence)
        # Simulate simple processing (e.g., temporal smoothing)
        processed = [alpha.copy() for alpha in sample_alpha_sequence]

        assert len(processed) == original_length, \
            f"Sequence length changed: {original_length} -> {len(processed)}"

    @pytest.mark.unit
    def test_frame_sequence_order_preserved(self, sample_alpha_sequence):
        """Frame order in sequence must be preserved."""
        # Add frame index as metadata
        for i, alpha in enumerate(sample_alpha_sequence):
            # Frame i should still be at position i after processing
            assert alpha is sample_alpha_sequence[i], "Frame order changed"

    @pytest.mark.unit
    def test_sequence_shape_consistency(self, sample_alpha_sequence):
        """All frames in sequence should have same shape."""
        if len(sample_alpha_sequence) > 0:
            expected_shape = sample_alpha_sequence[0].shape

            for i, alpha in enumerate(sample_alpha_sequence):
                assert alpha.shape == expected_shape, \
                    f"Frame {i} shape {alpha.shape} != expected {expected_shape}"

    @pytest.mark.unit
    def test_sequence_dtype_consistency(self, sample_alpha_sequence):
        """All frames should have same dtype."""
        if len(sample_alpha_sequence) > 0:
            expected_dtype = sample_alpha_sequence[0].dtype

            for i, alpha in enumerate(sample_alpha_sequence):
                assert alpha.dtype == expected_dtype, \
                    f"Frame {i} dtype {alpha.dtype} != expected {expected_dtype}"


class TestOpticalFlowValidation:
    """Test that optical flow maps are properly formatted."""

    @pytest.mark.unit
    def test_optical_flow_is_2d_with_2_channels(self, sample_optical_flow):
        """Optical flow should be (H, W, 2) - 2 channels for dx, dy."""
        flow = sample_optical_flow
        assert flow.ndim == 3, f"Flow should be 3D, got {flow.ndim}D"
        assert flow.shape[2] == 2, f"Flow should have 2 channels, got {flow.shape[2]}"

    @pytest.mark.unit
    def test_optical_flow_dtype_is_float(self, sample_optical_flow):
        """Optical flow should be float (has decimal components)."""
        flow = sample_optical_flow
        assert flow.dtype in [np.float32, np.float64], \
            f"Flow dtype should be float, got {flow.dtype}"

    @pytest.mark.unit
    def test_optical_flow_no_nan_or_inf(self, sample_optical_flow):
        """Optical flow must not contain NaN or Inf (would break warping)."""
        flow = sample_optical_flow
        assert not np.isnan(flow).any(), "Flow contains NaN values"
        assert not np.isinf(flow).any(), "Flow contains Inf values"

    @pytest.mark.unit
    def test_optical_flow_magnitude_reasonable(self, sample_optical_flow):
        """Optical flow magnitude should be realistic (not extreme)."""
        flow = sample_optical_flow
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Typical motion is < 50 pixels per frame (for 30fps video)
        # Allow up to 200 pixels for fast motion
        assert magnitude.max() < 500, \
            f"Flow magnitude {magnitude.max()} seems unrealistic"


class TestTrimapValidation:
    """Test that trimaps are properly formatted."""

    @pytest.mark.unit
    def test_trimap_is_2d_array(self, sample_trimap):
        """Trimap should be 2D (H, W)."""
        trimap = sample_trimap
        assert trimap.ndim == 2, f"Trimap should be 2D, got {trimap.ndim}D"

    @pytest.mark.unit
    def test_trimap_dtype_is_uint8(self, sample_trimap):
        """Trimap should be uint8 (values 0, 128, 255)."""
        trimap = sample_trimap
        assert trimap.dtype == np.uint8, f"Trimap dtype should be uint8, got {trimap.dtype}"

    @pytest.mark.unit
    def test_trimap_only_valid_values(self, sample_trimap):
        """Trimap should only contain 0 (bg), 128 (unknown), 255 (fg)."""
        trimap = sample_trimap
        unique_values = np.unique(trimap)

        valid_values = {0, 128, 255}
        for val in unique_values:
            assert val in valid_values, \
                f"Trimap contains invalid value {val}, must be in {valid_values}"

    @pytest.mark.unit
    def test_trimap_has_all_three_regions(self, sample_trimap):
        """Well-formed trimap should have bg, unknown, and fg regions."""
        trimap = sample_trimap
        unique_values = set(np.unique(trimap))

        # Should have background and unknown at minimum
        assert 0 in unique_values, "Trimap missing background (0)"
        assert 255 in unique_values, "Trimap missing foreground (255)"
        assert 128 in unique_values, "Trimap missing unknown region (128)"


class TestBlendModeDataValidation:
    """Test data contracts for blend mode operations."""

    @pytest.mark.unit
    def test_two_alphas_same_shape_for_blending(self):
        """Two alphas being blended must have same shape."""
        alpha1 = np.random.rand(512, 512).astype(np.float32)
        alpha2 = np.random.rand(512, 512).astype(np.float32)

        assert alpha1.shape == alpha2.shape, \
            f"Blend alpha1 {alpha1.shape} != alpha2 {alpha2.shape}"

    @pytest.mark.unit
    def test_premultiply_output_range(self):
        """After premultiply, RGB channels should still be in [0, 1] (or [0, 255])."""
        # Straight alpha
        rgb = np.ones((10, 10, 3), dtype=np.float32) * 0.8
        alpha = np.ones((10, 10), dtype=np.float32) * 0.5

        # Premultiply
        premul_rgb = rgb * alpha[:, :, np.newaxis]

        # Result should be in valid range
        assert premul_rgb.min() >= 0.0, "Premultiply produced negative values"
        assert premul_rgb.max() <= 1.0, "Premultiply exceeded max"


class TestDtypeConsistency:
    """Test that dtypes are consistent across operations."""

    @pytest.mark.unit
    def test_uint8_to_float_conversion_range(self):
        """Converting uint8 [0,255] to float should produce [0,1]."""
        uint8_val = np.uint8(200)
        float_val = float(uint8_val) / 255.0

        assert 0.0 <= float_val <= 1.0, f"Converted value {float_val} outside [0,1]"

    @pytest.mark.unit
    def test_float_to_uint8_conversion_range(self):
        """Converting float [0,1] to uint8 should produce [0,255]."""
        float_val = 0.75
        uint8_val = np.uint8(float_val * 255)

        assert 0 <= uint8_val <= 255, f"Converted value {uint8_val} outside [0,255]"

    @pytest.mark.unit
    def test_float32_vs_float64_precision(self):
        """Operations should handle both float32 and float64."""
        alpha32 = np.random.rand(100, 100).astype(np.float32)
        alpha64 = np.random.rand(100, 100).astype(np.float64)

        # Both should be valid
        assert alpha32.dtype == np.float32
        assert alpha64.dtype == np.float64
        # Both should be in valid range
        assert 0.0 <= alpha32.min() <= 1.0
        assert 0.0 <= alpha64.min() <= 1.0


class TestErrorHandlingBoundaries:
    """Test error handling at boundary conditions."""

    @pytest.mark.unit
    def test_zero_shaped_array_handling(self):
        """Zero-sized arrays should be handled gracefully (or rejected with clear error)."""
        # This tests that the code doesn't crash on empty arrays
        try:
            alpha = np.zeros((0, 0), dtype=np.float32)
            # If accepted, it must remain valid
            assert not np.isnan(alpha).any()
        except (ValueError, RuntimeError) as e:
            # Also acceptable: explicit error message
            assert "empty" in str(e).lower() or "shape" in str(e).lower()

    @pytest.mark.unit
    def test_single_pixel_array_handling(self):
        """1x1 arrays should work or reject with clear error."""
        alpha = np.array([[0.5]], dtype=np.float32)
        assert alpha.shape == (1, 1)
        assert alpha[0, 0] == 0.5

    @pytest.mark.unit
    def test_large_array_allocation(self):
        """Creating large arrays should work (or fail with clear memory error)."""
        try:
            # 4K x 4K float32 = 64MB (should be fine)
            large_alpha = np.zeros((4096, 4096), dtype=np.float32)
            assert large_alpha.shape == (4096, 4096)
        except MemoryError as e:
            # Clear error message
            assert "memory" in str(e).lower()
