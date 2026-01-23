"""
AUTO-ROTO Test Suite

This package contains comprehensive tests for the AUTO-ROTO v5 rotoscoping pipeline.

Test organization:
- test_data_validation.py: Input/output validation and data contracts
- test_auto_roto.py: SAM2 segmentation module
- test_depth_refine.py: Depth Anything V3 refinement module
- test_vitmatte_refine.py: ViTMatte alpha matting module
- test_edge_refine.py: Edge processing module
- test_temporal_smooth.py: Temporal coherence module
- test_matte_combine.py: Matte combination module
- test_full_pipeline_v5.py: Full pipeline integration
- test_performance.py: Performance benchmarks and regressions
- test_quality_metrics.py: Output quality validation

Run all tests:
    pytest tests/ -v

Run specific test category:
    pytest tests/ -m unit -v        # Unit tests only
    pytest tests/ -m integration -v # Integration tests only
    pytest tests/ -m quality -v     # Quality tests only

Run with coverage:
    pytest tests/ --cov=. --cov-report=html

For detailed information, see TESTING_STRATEGY.md
"""

__version__ = "1.0.0"
