# AUTO-ROTO Technical Context

> **Version:** 5.0
> **Updated:** 2026-01-23
> **Status:** Production-Ready

This document provides technical context for developers working on AUTO-ROTO. For user documentation, see [README.md](README.md).

---

## Architecture Overview

AUTO-ROTO is a multi-stage video rotoscoping pipeline that chains specialized AI models for different aspects of alpha matte generation.

### Design Philosophy

1. **Sequential GPU usage** - Models run one at a time to fit in typical VRAM (~12-24GB)
2. **Lazy model loading** - Models load on first use, not at import
3. **Dataclass configuration** - All modules use typed `@dataclass` configs
4. **Intermediate caching** - Each stage outputs can be reused independently
5. **Production output** - 16/32-bit EXR with proper alpha handling

### V5 Pipeline Stages

```
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│   SAM2     │──▸│   Depth    │──▸│  ViTMatte  │──▸│   Edge     │──▸│  Temporal  │──▸│  Combine   │
│ Segment    │   │  Refine    │   │  Refine    │   │  Refine    │   │  Smooth    │   │  Mattes    │
└────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘   └────────────┘
     │                │                │                │                │                │
     ▼                ▼                ▼                ▼                ▼                ▼
  Binary/soft     Depth-aware      True alpha       Subpixel       Flicker-free      Final
   masks         edge quality      with hair         edges         consistency      composite
```

---

## Module Details

### Stage 1: SAM2 Segmentation (`auto_roto.py`)

**Purpose:** Generate initial segmentation masks with temporal consistency

**Key Classes:**
- `RotoConfig` - Pipeline configuration
- `VideoReader` - Handles video files and image sequences
- `FrameWriter` - Multi-format output (EXR, PNG, TIFF)
- `AlphaRefiner` - Binary mask to soft alpha conversion
- `SAM2Segmenter` - SAM2 video predictor wrapper
- `GroundingDINODetector` - Text-to-bounding-box detection
- `AutoRotoPipeline` - Main orchestration

**Technical Notes:**
- Uses SAM2's temporal memory bank for frame-to-frame consistency
- Bi-directional propagation handles occlusions
- Multi-object tracking with consistent IDs
- Text prompts via GroundingDINO, or interactive/box selection

### Stage 2: Depth Refinement (`depth_refine.py`)

**Purpose:** Refine edges using monocular depth cues

**Key Insight:** Depth discontinuities indicate object boundaries

**Key Classes:**
- `DepthRefineConfig` - Refinement configuration
- `DepthEstimator` - Depth Anything V2/V3 wrapper
- `DepthGuidedRefiner` - Edge refinement logic
- `DepthRefinePipeline` - Orchestration

**Technical Notes:**
- Supports Depth Anything V2 and V3 (nested architecture)
- Hard edge sharpening where depth changes abruptly
- Soft edge preservation where depth is continuous
- Separate GPU pass to avoid VRAM conflicts with SAM2

### Stage 3: ViTMatte Refinement (`vitmatte_refine.py`)

**Purpose:** True alpha matting for hair and fine details

**Key Innovation:** Geometric-aware trimap synthesis from depth map

**Key Classes:**
- `TrimapConfig` - Adaptive trimap settings
- `TrimapSynthesizer` - Builds foreground/background/unknown regions
- `ViTMatteRefiner` - HuggingFace ViTMatte integration
- `GeometricMatteRefiner` - Full pipeline orchestration

**Technical Notes:**
- Adaptive trimap synthesis with unknown zone width 20-60px
- Depth variance analysis for zone width adjustment
- High-pass filtering (Laplacian) for geometric strand detection
- 32-bit float processing for sub-pixel precision
- Motion-aware trimap expansion (optional)

**Configuration Parameters:**
- `base_unknown_width`: Minimum unknown zone (default: 20px)
- `max_unknown_width`: Maximum unknown zone (default: 60px)
- `erosion_depth`: Foreground erosion iterations (default: 25)

### Stage 4: Edge Refinement (`edge_refine.py`)

**Purpose:** Professional VFX-grade edge processing

**Key Techniques:**
- Subpixel edge detection (Laplacian-of-Gaussian)
- Color difference keying (Primatte/Keylight-style)
- Morphological gradient refinement
- Premultiplied alpha operations
- Gamma-correct blending

**Technical Notes:**
- Edge sampling from source RGB for color keying
- Despill via complementary color suppression
- Core/edge/soft-edge decomposition

### Stage 5: Temporal Smoothing (`temporal_smooth.py`)

**Purpose:** Prevent frame-to-frame flickering

**Key Techniques:**
- Optical flow estimation for motion tracking
- Temporal median filtering with adaptive windows
- Edge confidence propagation
- Keyframe anchoring to prevent drift

**Configuration Parameters:**
- `temporal_window`: Frames for smoothing (3-9)
- `keyframe_interval`: Keyframe anchor frequency

### Stage 6: Matte Combination (`matte_combine.py`)

**Purpose:** Multi-layer matte architecture and final compositing

**Architecture:**
- **Core Matte**: 100% solid interior
- **Detail Matte**: Hair, transparency, fine edges
- **Soft Edge Matte**: Anti-aliased boundaries

**Blend Modes:** OVER, UNDER, MAX, MIN, ADD, SCREEN, MULTIPLY, DIFFERENCE, AVERAGE

---

## Configuration

### Quality Presets

| Preset | SAM Model | Depth Model | Temporal Window | Edge Softness |
|--------|-----------|-------------|-----------------|---------------|
| draft | tiny | small | 3 | 0.5 |
| standard | base_plus | base | 5 | 1.0 |
| high | large | large | 7 | 1.5 |
| ultra | large | large | 9 | 2.0 |

### Model VRAM Requirements

| Model | SAM2 | Depth | ViTMatte | Total Peak |
|-------|------|-------|----------|------------|
| tiny/small | ~4GB | ~2GB | ~2GB | ~4GB |
| base | ~7GB | ~4GB | ~2GB | ~7GB |
| large | ~12GB | ~8GB | ~2GB | ~12GB |

*Note: Models run sequentially, not simultaneously*

---

## Dependencies

### Python Packages

```
numpy>=1.24.0          # Array operations
opencv-python>=4.8.0   # Image processing
Pillow>=10.0.0         # Image I/O
scipy>=1.11.0          # Scientific computing
tqdm>=4.65.0           # Progress bars
rich>=13.0.0           # Rich CLI output
torch>=2.5.1           # PyTorch with CUDA
torchvision>=0.20.1    # Vision utilities
transformers>=4.30.0   # HuggingFace (ViTMatte)
huggingface_hub>=0.19.0
imageio>=2.31.0        # Image sequence I/O
imageio-ffmpeg>=0.4.8  # FFmpeg integration
```

### External Repositories

| Repo | Purpose | Installation |
|------|---------|--------------|
| sam2 | SAM2 segmentation | `pip install -e sam2_repo/` |
| GroundingDINO | Text detection | `pip install -e GroundingDINO/` |
| Depth-Anything-V2 | Depth estimation v2 | Clone to `Depth-Anything-V2/` |
| depth-anything-3 | Depth estimation v3 | Clone to `depth-anything-3/` |

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Model too large | Use smaller model size |
| GroundingDINO not found | Not installed | Use box/interactive mode |
| OpenEXR unavailable | Missing system lib | `pip install OpenEXR` |
| Flickering | No temporal smoothing | Enable temporal stage |
| Poor hair edges | ViTMatte skipped | Enable ViTMatte stage |

---

## File Structure

```
auto_roto/
├── auto_roto.py           # Stage 1: SAM2 segmentation
├── depth_refine.py        # Stage 2: Depth refinement
├── vitmatte_refine.py     # Stage 3: ViTMatte matting
├── edge_refine.py         # Stage 4: Edge refinement
├── temporal_smooth.py     # Stage 5: Temporal smoothing
├── matte_combine.py       # Stage 6: Matte combination
├── hair_refine.py         # Optional: Hair-specific refinement
├── full_pipeline_v5.py    # V5 orchestrator (recommended)
├── full_pipeline.py       # Legacy 2-stage (deprecated)
├── batch_roto.py          # Batch processing
├── auto_roto_nuke.py      # Nuke integration
├── requirements.txt       # Python dependencies
├── install.sh             # Installation script
├── README.md              # User documentation
├── CONTEXT.md             # This file
├── SOURCEOFTRUTH.md       # Scientific matting reference
├── docs/                  # Additional documentation
│   └── plans/             # Design documents
├── sam2_repo/             # SAM2 (cloned)
├── GroundingDINO/         # GroundingDINO (cloned)
├── Depth-Anything-V2/     # Depth v2 (cloned)
└── depth-anything-3/      # Depth v3 (cloned)
```

---

## Development Notes

### Adding a New Stage

1. Create module with `*Config` dataclass and `*Pipeline` class
2. Add stage flag to `PipelineConfig` in `full_pipeline_v5.py`
3. Add execution in `run_pipeline()` function
4. Update quality presets if needed

### Testing Changes

The `test_input/` directory contains sample frames for testing. Test output directories (gitignored) capture intermediate results.

### GPU Memory Management

Each stage should:
1. Load model on first frame
2. Process all frames
3. Call `clear_gpu_memory()` when done

```python
def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
```

---

## References

- [SAM2 Paper](https://arxiv.org/abs/2408.00714) - Segment Anything in Images and Videos
- [Depth Anything V2](https://arxiv.org/abs/2406.09414) - Fine-Tuned Depth Estimation
- [ViTMatte](https://arxiv.org/abs/2305.15272) - Vision Transformer for Image Matting
- [SOURCEOFTRUTH.md](SOURCEOFTRUTH.md) - Scientific foundation for trimap synthesis
