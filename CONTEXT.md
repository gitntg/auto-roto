# AUTO-ROTO Context Summary
> Generated: 2026-01-21
> Version: 2.0
> Status: Production-Ready

---

## Project Overview

**AUTO-ROTO** is a production-grade automatic rotoscoping pipeline built for VFX artists. It generates high-quality alpha mattes from video clips using state-of-the-art AI models.

### Core Purpose
- Automate rotoscoping workflows for VFX production
- Generate production-quality alpha mattes (not binary masks)
- Support Nuke/After Effects/Flame integration via 16-bit EXR output
- Handle complex scenarios: hair, fine details, motion blur

### Target Users
- VFX compositors and artists
- Post-production facilities
- Anyone needing clean mattes from video

---

## Architecture

### Pipeline Flow
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE (full_pipeline.py)                     │
├─────────────────────────────────┬───────────────────────────────────────────┤
│       STAGE 1: SAM2             │        STAGE 2: DEPTH REFINEMENT          │
│      (auto_roto.py)             │         (depth_refine.py)                 │
│                                 │                                           │
│  Input Video/Sequence           │    Rough Masks + Video                    │
│         ↓                       │           ↓                               │
│  Frame Extraction               │    Depth Anything V2                      │
│         ↓                       │           ↓                               │
│  ┌─────────────────────┐        │    Depth Edge Detection                   │
│  │  GroundingDINO      │ OR     │           ↓                               │
│  │  (text prompt)      │        │    Edge Comparison                        │
│  └─────────────────────┘        │    • Hard edges → Sharpen                 │
│  ┌─────────────────────┐        │    • Soft regions → Preserve              │
│  │  Interactive        │        │           ↓                               │
│  │  (point/box)        │        │    Guided Filter + Refinement             │
│  └─────────────────────┘        │           ↓                               │
│         ↓                       │    Final Alpha Mattes                     │
│  SAM 2.1 Video Predictor        │                                           │
│  • Temporal memory              │                                           │
│  • Multi-object tracking        │                                           │
│  • Bi-directional propagation   │                                           │
│         ↓                       │                                           │
│  Rough Masks (binary/soft)      │                                           │
│         ↓                       │                                           │
│  Alpha Refinement               │                                           │
│         ↓                       │                                           │
│  EXR Export (16-bit float)      │                                           │
└─────────────────────────────────┴───────────────────────────────────────────┘
```

### Key Components

| File | Purpose | Lines of Code |
|------|---------|---------------|
| `auto_roto.py` | Main SAM2 segmentation pipeline | ~1,520 |
| `depth_refine.py` | Depth-guided alpha refinement | ~920 |
| `full_pipeline.py` | Two-pass orchestrator | ~310 |
| `batch_roto.py` | Batch processing multiple videos | ~415 |
| `auto_roto_nuke.py` | Nuke integration module | ~370 |

---

## Technical Decisions

### 1. Two-Pass Architecture
**Decision**: Run SAM2 and Depth Anything V2 sequentially, not together.

**Rationale**:
- Both models are VRAM-intensive (~12GB+ each for large models)
- Cannot fit simultaneously in typical GPU memory
- Allows caching depth maps for reuse
- Enables iterative refinement with different settings

### 2. SAM2.1 as Core Segmentation Model
**Decision**: Use SAM2 (Segment Anything Model 2) over alternatives.

**Rationale**:
- Temporal memory bank prevents frame-to-frame flickering
- Multi-object tracking with consistent IDs
- Bi-directional propagation handles occlusions
- State-of-the-art quality for video segmentation

### 3. GroundingDINO for Text-Based Detection
**Decision**: Use GroundingDINO for converting text prompts to bounding boxes.

**Rationale**:
- Zero-shot object detection from text descriptions
- No pre-training required for custom objects
- Works well with common VFX subjects (person, car, etc.)
- Seamless integration with SAM2's box prompts

### 4. Depth Anything V2 for Refinement
**Decision**: Use monocular depth estimation to guide alpha refinement.

**Rationale**:
- Depth discontinuities indicate object boundaries
- Helps preserve soft edges (hair, motion blur) where depth is continuous
- Separates overlapping objects by depth ordering
- Improves edge quality beyond what SAM2 alone provides

### 5. EXR as Primary Output Format
**Decision**: Default to 16-bit OpenEXR output.

**Rationale**:
- Industry standard for VFX (Nuke, Flame, Resolve)
- Preserves full dynamic range
- Proper handling of semi-transparent edges
- No banding artifacts in subtle gradients

### 6. Lazy Model Loading
**Decision**: Load models on first use, not at import time.

**Rationale**:
- Faster script startup
- Only loads models actually needed
- Better memory management
- Cleaner error messages for missing dependencies

---

## Configuration Options

### SAM2 Models
| Model | VRAM | Speed (1080p) | Quality |
|-------|------|---------------|---------|
| tiny | ~4GB | ~45 FPS | Good |
| small | ~5GB | ~35 FPS | Better |
| base_plus | ~7GB | ~25 FPS | Great |
| large | ~12GB | ~15 FPS | Best |

### Depth Models
| Model | VRAM | Speed | Use Case |
|-------|------|-------|----------|
| small | ~2GB | Fast | Quick preview |
| base | ~4GB | Medium | Production |
| large | ~8GB | Slower | Maximum quality |

### Quality Presets (full_pipeline.py)
| Preset | SAM Model | Depth Model |
|--------|-----------|-------------|
| draft | tiny | small |
| standard | base_plus | base |
| high | large | large |
| fast | small | small |

---

## Input/Output

### Supported Input Formats
- **Video**: MP4, MOV, AVI, MKV, WebM, MXF
- **Image Sequences**: EXR, PNG, TIFF, DPX, JPEG

### Output Structure
```
output/
├── alpha/           # Alpha-only channels (primary output)
│   ├── roto.0001.exr
│   ├── roto.0002.exr
│   └── ...
├── rgba/            # Full RGBA images (optional)
│   ├── roto.0001.exr
│   ├── roto.0002.exr
│   └── ...
├── preview/         # Quick preview JPGs (optional)
│   ├── roto.0001.jpg
│   └── ...
└── depth/           # Depth maps (if saved)
    ├── depth.0001.png
    └── ...
```

---

## Dependencies

### Core Dependencies
```
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
scipy>=1.11.0
tqdm>=4.65.0
rich>=13.0.0
matplotlib>=3.7.0
requests>=2.31.0
huggingface_hub>=0.19.0
imageio>=2.31.0
imageio-ffmpeg>=0.4.8
```

### PyTorch (install separately)
```
torch>=2.5.1
torchvision>=0.20.1
```
Install with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### External Repositories (install from source)
1. **SAM2**: `git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .`
2. **GroundingDINO** (optional): `git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO && pip install -e .`
3. **Depth Anything V2** (optional): Can use HuggingFace Transformers or install from source

### Optional Dependencies
- `OpenEXR>=3.2.0` - Full EXR support (requires system libraries)
- `opencv-contrib-python>=4.8.0` - Better guided filter support

---

## Usage Patterns

### Basic Usage
```bash
# Text prompt detection
python auto_roto.py --input video.mp4 --prompt "person" --output ./output

# Multiple objects
python auto_roto.py --input video.mp4 --prompt "person.dog.car" --output ./output

# Interactive box selection
python auto_roto.py --input video.mp4 --interactive --output ./output
```

### Full Pipeline (Recommended)
```bash
# High quality with depth refinement
python full_pipeline.py --input video.mp4 --prompt "person" --output ./output

# Fast mode
python full_pipeline.py --input video.mp4 --prompt "person" --output ./output --fast
```

### Two-Pass Workflow (Manual)
```bash
# Pass 1: SAM2 segmentation
python auto_roto.py --input video.mp4 --prompt "person" --output ./sam_out --no-refine

# Pass 2: Depth refinement
python depth_refine.py --alpha ./sam_out/alpha/ --video video.mp4 --output ./final
```

### Batch Processing
```bash
# Process folder of videos
python batch_roto.py --input ./videos/ --prompt "person" --output ./output/

# Use config file
python batch_roto.py --config batch_config.json
```

---

## Nuke Integration

### Installation
1. Copy `auto_roto_nuke.py` to `~/.nuke/`
2. Add to `menu.py`:
   ```python
   import auto_roto_nuke
   auto_roto_nuke.add_to_menu()
   ```

### Usage in Nuke
- Select Read node with video/sequence
- Run: `Nodes > AUTO-ROTO > Process Selected`
- Or use keyboard shortcut: `Ctrl+Shift+R`

---

## Key Classes

### auto_roto.py
- `RotoConfig` - Configuration dataclass for pipeline settings
- `VideoReader` - Reads video files and image sequences
- `FrameWriter` - Writes alpha/RGBA/preview outputs
- `AlphaRefiner` - Refines binary masks to smooth alphas
- `SAM2Segmenter` - Wrapper for SAM2 video predictor
- `GroundingDINODetector` - Text-to-bbox detection
- `AutoRotoPipeline` - Main orchestration class

### depth_refine.py
- `DepthRefineConfig` - Configuration for depth refinement
- `DepthEstimator` - Depth Anything V2 wrapper
- `DepthGuidedRefiner` - Alpha refinement using depth
- `DepthRefinePipeline` - Depth refinement orchestrator

### batch_roto.py
- `RotoJob` - Single job dataclass with status tracking
- `BatchProcessor` - Multi-video processing with reports

---

## Error Handling

### Common Issues & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Model too large | Use smaller model (--sam-model small) |
| GroundingDINO not found | Not installed | Use --box or --interactive instead |
| OpenEXR not available | Missing system lib | Install libopenexr-dev (Ubuntu) |
| Flickering between frames | Poor temporal consistency | Use larger model, ensure backward propagation |
| Poor edge quality | Insufficient refinement | Increase --refine-iterations, --edge-softness |

---

## Future Roadmap

- [ ] Nuke plugin for in-app processing
- [ ] GUI interface (Gradio/Streamlit)
- [ ] Deep matting models (RVM, ViTMatte) integration
- [ ] Cloud processing option
- [ ] Real-time preview mode

---

## License

MIT License - Free for commercial use.

---

## Acknowledgments

- [SAM2](https://github.com/facebookresearch/sam2) by Meta AI
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA Research
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) by Depth Anything team
