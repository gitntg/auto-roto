# AUTO-ROTO: Production-Grade Automatic Rotoscoping

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Automatic rotoscoping that generates production-quality alpha mattes from video clips.**

Built on SAM2 (Segment Anything Model 2.1) with ViTMatte for fine detail matting, Depth Anything V3 for edge refinement, and professional VFX quality enhancements.

---
Path to full : F:/3DActive/CarCrash/reel-new/renders/scene-01nfp/
 python full_pipeline_v5.py --input ./test_input --prompt "person" --detail-points --output ./output --quality ultra --refiner ma1
## Features

- **Text-based detection** - "person", "car", "dog" -> instant roto via GroundingDINO
- **ViTMatte integration** - True alpha matting for hair and fine details
- **Depth-guided refinement** - Depth Anything V2/V3 for edge quality
- **Temporal consistency** - Optical flow-based anti-flicker
- **Multi-object support** - Track multiple objects simultaneously
- **Production output** - 16-bit EXR sequences for Nuke/After Effects

### Supported Formats

**Input:** MP4, MOV, MKV, AVI, WebM, MXF, EXR, PNG, TIFF, DPX, JPEG sequences
**Output:** EXR (16/32-bit), PNG, TIFF with proper alpha channel

---

## Quick Start

### Installation

```bash
# Clone this repo
git clone <repo-url>
cd auto_roto

# Run installation script
chmod +x install.sh
./install.sh
```

Or manual install:

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install SAM2
git clone https://github.com/facebookresearch/sam2.git sam2_repo
cd sam2_repo && pip install -e . && cd ..

# Install GroundingDINO (optional, for text prompts)
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && cd ..

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Recommended: Full V5 Pipeline
python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output

# With quality preset
python full_pipeline_v5.py --input video.mp4 --prompt "person" --quality high

# SAM2 only (fastest)
python auto_roto.py --input video.mp4 --prompt "person" --output ./output

# Multiple objects
python auto_roto.py --input video.mp4 --prompt "person.dog.car" --output ./output

# Interactive mode (click to select)
python auto_roto.py --input video.mp4 --interactive --output ./output
```

---

## V5 Pipeline

The recommended pipeline chains multiple refinement stages:

```
SAM2 -> Depth -> ViTMatte -> Edge Refine -> Temporal Smooth -> Final
```

### Quality Presets

| Preset | SAM Model | Depth Model | Temporal Window | Use Case |
|--------|-----------|-------------|-----------------|----------|
| draft | tiny | small | 3 frames | Quick preview |
| standard | base_plus | base | 5 frames | Production (default) |
| high | large | large | 7 frames | Hero shots |
| ultra | large | large | 9 frames | Maximum quality |

### Usage

```bash
# Standard quality (default)
python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output

# High quality
python full_pipeline_v5.py --input video.mp4 --prompt "person" --quality high

# Skip temporal smoothing (faster)
python full_pipeline_v5.py --input video.mp4 --prompt "person" --skip-temporal

# Keep intermediate files for debugging
python full_pipeline_v5.py --input video.mp4 --prompt "person" --keep-intermediate
```

---

## Architecture

```
                           V5 PIPELINE (full_pipeline_v5.py)
                                      |
    +------------------------------------------------------------------+
    |                                                                  |
    v                                                                  v
+-------------------+        +-------------------+        +-------------------+
|   STAGE 1: SAM2   |  --->  |  STAGE 2: DEPTH   |  --->  | STAGE 3: VITMATTE |
|  (auto_roto.py)   |        | (depth_refine.py) |        |(vitmatte_refine.py|
|                   |        |                   |        |                   |
| - Text/box prompt |        | - Depth Anything  |        | - Adaptive trimap |
| - SAM2 temporal   |        |   V2 or V3        |        | - True alpha solve|
| - Multi-object    |        | - Edge sharpening |        | - Hair/fine detail|
+-------------------+        +-------------------+        +-------------------+
                                      |
                                      v
+-------------------+        +-------------------+        +-------------------+
| STAGE 4: EDGE     |  --->  | STAGE 5: TEMPORAL |  --->  | STAGE 6: COMBINE  |
| (edge_refine.py)  |        |(temporal_smooth.py|        |(matte_combine.py) |
|                   |        |                   |        |                   |
| - Subpixel edges  |        | - Optical flow    |        | - Core/detail/soft|
| - Color keying    |        | - Anti-flicker    |        | - Despill         |
| - Despill         |        | - Keyframe anchor |        | - Final composite |
+-------------------+        +-------------------+        +-------------------+
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `auto_roto.py` | SAM2 video segmentation with GroundingDINO |
| `depth_refine.py` | Depth Anything V2/V3 edge refinement |
| `vitmatte_refine.py` | ViTMatte alpha matting with adaptive trimap |
| `edge_refine.py` | Professional subpixel edge processing |
| `temporal_smooth.py` | Optical flow-based temporal coherence |
| `matte_combine.py` | Multi-layer matte combination |
| `full_pipeline_v5.py` | V5 orchestrator (recommended entry point) |

---

## Module Reference

### SAM2 Segmentation (`auto_roto.py`)

Core segmentation using SAM2's temporal memory for consistent tracking.

```bash
python auto_roto.py --input video.mp4 --prompt "person" --output ./output
python auto_roto.py --input video.mp4 --box "100,100,500,400" --output ./output
python auto_roto.py --input video.mp4 --interactive --output ./output
```

**Key options:**
- `--sam-model`: tiny, small, base_plus, large (default: large)
- `--no-refine`: Output binary masks (faster)
- `--no-backward`: Disable backward propagation

### Depth Refinement (`depth_refine.py`)

Depth-guided edge refinement using Depth Anything V2 or V3.

```bash
python depth_refine.py --alpha ./output/alpha/ --video video.mp4 --output ./refined
```

**Key options:**
- `--depth-model`: small, base, large
- `--depth-version`: v2, v3 (default: v3)
- `--edge-threshold`: Depth gradient threshold (default: 0.1)

### ViTMatte Refinement (`vitmatte_refine.py`)

True alpha matting using Vision Transformer with depth-aware trimap synthesis.

```bash
python vitmatte_refine.py --alpha ./output/alpha/ --video video.mp4 --output ./vitmatte
```

**Key options:**
- `--adaptive`: Enable adaptive trimap width (default)
- `--motion-aware`: Expand trimap in motion areas

### Batch Processing (`batch_roto.py`)

Process multiple videos with a config file.

```bash
python batch_roto.py --input ./videos/ --prompt "person" --output ./output/
python batch_roto.py --config batch_config.json
```

---

## Output Structure

```
output/
├── alpha/           # Alpha-only channels (primary output)
│   ├── roto.0001.exr
│   ├── roto.0002.exr
│   └── ...
├── rgba/            # Full RGBA images
├── preview/         # Quick preview JPGs
└── depth/           # Depth maps (if saved)
```

---

## Nuke Integration

### Setup

1. Copy `auto_roto_nuke.py` to `~/.nuke/`
2. Add to `menu.py`:
   ```python
   import auto_roto_nuke
   auto_roto_nuke.add_to_menu()
   ```

### Usage

- Select Read node with video/sequence
- Run: `Nodes > AUTO-ROTO > Process Selected`

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Use `--sam-model small` or `--quality draft` |
| GroundingDINO not found | Use `--box` or `--interactive` instead of `--prompt` |
| OpenEXR not available | Install: `pip install OpenEXR` (requires system libs) |
| Flickering | Use `--quality high` or ensure temporal smoothing enabled |
| Poor hair edges | Ensure ViTMatte stage is not skipped |

---

## Performance

| Model | VRAM | Speed (1080p) | Quality |
|-------|------|---------------|---------|
| tiny | ~4GB | ~45 FPS | Good |
| small | ~5GB | ~35 FPS | Better |
| base_plus | ~7GB | ~25 FPS | Great |
| large | ~12GB | ~15 FPS | Best |

*Benchmarks on RTX 4090, varies by video complexity*

---

## Roadmap

- [ ] Nuke plugin for in-app processing
- [ ] GUI interface (Gradio/Streamlit)
- [ ] Cloud processing option
- [ ] Real-time preview mode

---

## License

MIT License - use freely in commercial projects.

---

## Acknowledgments

- [SAM2](https://github.com/facebookresearch/sam2) by Meta AI
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA Research
- [Depth Anything](https://github.com/DepthAnything/Depth-Anything-V2) by Depth Anything team
- [ViTMatte](https://huggingface.co/docs/transformers/model_doc/vitmatte) via HuggingFace Transformers

---

**Built for VFX artists who need reliable auto-roto.**
