# AUTO-ROTO: Production-Grade Automatic Rotoscoping

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Bulletproof automatic rotoscoping that generates perfect alpha mattes from complex video clips.**

Built on SAM2 (Segment Anything Model 2.1) with optional GroundingDINO for text-based object detection, plus production-quality alpha refinement for hair, fine details, and motion blur.

---

## 🎯 Why This Exists

ComfyUI and other auto-roto solutions often fail because:
- They use outdated models with poor temporal consistency
- Output binary masks instead of proper alpha mattes
- No edge refinement for hair/fine details
- Frame-by-frame processing causes flickering

**AUTO-ROTO solves these problems** by combining:
- **SAM 2.1** with temporal memory for rock-solid consistency across frames
- **GroundingDINO** for automatic object detection via text prompts
- **Alpha matting refinement** for proper semi-transparent edges
- **Production-grade output** (16-bit EXR sequences for Nuke/After Effects)

---

## 📋 Features

### Core Features
- ✅ **Text-based detection** - "person", "car", "dog" → instant roto
- ✅ **Temporal consistency** - SAM2's memory prevents flickering
- ✅ **Multi-object support** - Track multiple objects simultaneously
- ✅ **Alpha refinement** - Proper hair/edge handling, not binary masks
- ✅ **Interactive mode** - Click to select objects on first frame

### Input/Output
- ✅ Video files (MP4, MOV, MKV, AVI, WebM, MXF)
- ✅ Image sequences (EXR, PNG, TIFF, DPX, JPEG)
- ✅ 16-bit EXR output for Nuke/Flame
- ✅ PNG/TIFF with proper alpha channel
- ✅ Preview JPGs for quick review

### Performance
- ✅ GPU acceleration (CUDA)
- ✅ Model compilation via `torch.compile` for 2-3x speedup
- ✅ Batch processing for sequences

---

## 🚀 Quick Start

### 1. Installation

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
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..

# Install GroundingDINO (optional, for text prompts)
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO && pip install -e . && cd ..

# Install Depth Anything V2 support (optional, for depth refinement)
pip install transformers
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

# Install other dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# RECOMMENDED: Full pipeline (SAM2 + Depth Refinement)
python full_pipeline.py --input video.mp4 --prompt "person" --output ./output

# SAM2 only (faster, still good quality)
python auto_roto.py --input video.mp4 --prompt "person" --output ./output

# Multiple objects
python auto_roto.py --input video.mp4 --prompt "person.dog.car" --output ./output

# Interactive selection (click to draw boxes)
python auto_roto.py --input video.mp4 --interactive --output ./output
```

### 3. Two-Pass Workflow (Sequential GPU Usage)

Since SAM2 and Depth Anything V2 can't fit in VRAM simultaneously:

```bash
# Pass 1: SAM2 segmentation (releases GPU when done)
python auto_roto.py --input video.mp4 --prompt "person" --output ./sam_out --no-refine

# Pass 2: Depth-guided refinement
python depth_refine.py --alpha ./sam_out/alpha/ --video video.mp4 --output ./final
```

Or use the automated full pipeline:
```bash
python full_pipeline.py --input video.mp4 --prompt "person" --output ./output
```

---

## 🌊 Depth-Guided Refinement

Depth Anything V2 provides a second refinement pass that significantly improves edge quality:

### How It Works

```
SAM2 Mask → Depth Map → Compare Edges → Refine Alpha
                ↓
    Depth discontinuity = object edge
    Continuous depth = preserve softness
```

**Key benefits:**
- **Hair/fine detail**: Depth helps identify where soft edges should be
- **Overlapping objects**: Depth ordering separates layers
- **Motion blur**: Continuous depth regions preserve natural blur
- **Hard edges**: Sharp depth changes get sharpened

### Usage

```bash
# On existing alpha mattes
python depth_refine.py --alpha ./output/alpha/ --video input.mp4 --output ./refined

# With pre-computed depth maps
python depth_refine.py --alpha ./output/alpha/ --depth ./depth_maps/ --output ./refined

# Adjust refinement strength
python depth_refine.py --alpha ./output/alpha/ --video input.mp4 \
    --edge-threshold 0.15 --blend-strength 0.7 --output ./refined
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--edge-threshold` | 0.1 | Depth gradient threshold for hard edges |
| `--blend-strength` | 0.5 | How strongly to apply refinement |
| `--no-sharpen` | off | Disable edge sharpening |
| `--no-soften` | off | Disable soft edge preservation |
| `--depth-model` | large | Model size: small, base, large |

---

### Command Line Options

```
Input/Output:
  --input, -i      Input video file or image sequence directory
  --output, -o     Output directory (default: ./output)

Prompt Types (choose one):
  --prompt, -p     Text prompt for detection (use . to separate multiple)
  --box, -b        Box coordinates: x1,y1,x2,y2
  --point          Point coordinates: x,y
  --interactive    Draw boxes interactively on first frame

SAM2 Settings:
  --sam-model      Model size: tiny, small, base_plus, large (default: large)
  --no-backward    Disable backward propagation

Detection Settings:
  --detection-threshold  GroundingDINO detection confidence (default: 0.3)
  --text-threshold       GroundingDINO text similarity threshold (default: 0.25)

Alpha Refinement:
  --no-refine           Disable alpha refinement (output binary masks)
  --refine-iterations   Refinement iterations (default: 3)
  --edge-softness       Edge softness 0-5 (default: 1.0)

Output Settings:
  --format         Output format: exr, png, tiff (default: exr)
  --bit-depth      Bit depth: 8, 16, 32 (default: 16)
  --no-rgb         Don't output RGBA files
  --no-preview     Don't generate preview images

Performance:
  --device         Device: cuda or cpu (default: cuda)
  --no-compile     Disable torch.compile optimization
```

### Output Structure

```
output/
├── alpha/           # Alpha-only channels
│   ├── roto.0001.exr
│   ├── roto.0002.exr
│   └── ...
├── rgba/            # Full RGBA images
│   ├── roto.0001.exr
│   ├── roto.0002.exr
│   └── ...
└── preview/         # Quick preview images
    ├── roto.0001.jpg
    ├── roto.0002.jpg
    └── ...
```

---

## 🔧 Nuke Integration

### Reading EXR Sequences

```python
# In Nuke, use Read node:
# File: /path/to/output/rgba/roto.####.exr
# Frame Range: 1-100 (adjust to your range)
```

### Gizmo for Quick Import

```tcl
# autoroto_import.gizmo
Group {
 name AutoRotoImport
 tile_color 0x7f00ffff
 
 inputs 0
 
 knobs {
  filepath {/path/to/output/rgba/roto.####.exr}
 }
 
 Read {
  file "\[value filepath]"
  format "1920 1080 0 0 1920 1080 1 HD"
  name Read1
 }
 
 Premult {
  name Premult1
 }
 
 Output {
  name Output1
 }
}
```

---

## 🎨 Tips for Best Results

### Text Prompts
- Be specific: "person in red shirt" > "person"
- Use multiple prompts: "person.background.shadow"
- Common objects work best: person, car, dog, cat, etc.

### For Hair/Fine Details
```bash
# Increase refinement iterations and softness
python auto_roto.py --input video.mp4 --prompt "person" \
    --refine-iterations 5 --edge-softness 1.5 --output ./output
```

### For Fast Objects
```bash
# Use backward propagation to handle motion blur
python auto_roto.py --input video.mp4 --prompt "car" --output ./output
# (backward propagation is enabled by default)
```

### For Complex Scenes
```bash
# Use larger model for better accuracy
python auto_roto.py --input video.mp4 --prompt "person" \
    --sam-model large --output ./output
```

### For Speed (Lower Quality)
```bash
# Use tiny model
python auto_roto.py --input video.mp4 --prompt "person" \
    --sam-model tiny --no-refine --format png --output ./output
```

---

## 🏗️ Architecture

```
                        FULL PIPELINE (full_pipeline.py)
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
        ▼                                                       ▼
┌───────────────────┐                               ┌───────────────────┐
│   STAGE 1: SAM2   │                               │  STAGE 2: DEPTH   │
│   (auto_roto.py)  │ ────── GPU memory clear ───── │ (depth_refine.py) │
└───────────────────┘                               └───────────────────┘
        │                                                       │
        ▼                                                       ▼
Input Video/Sequence                                     Rough Masks
       │                                                       │
       ▼                                                       ▼
┌─────────────────┐                               ┌─────────────────────┐
│ Frame Extraction │                               │ Depth Anything V2   │
└────────┬────────┘                               │   (per frame)       │
         │                                        └──────────┬──────────┘
         ▼                                                   │
┌─────────────────┐     ┌──────────────────┐                 ▼
│  GroundingDINO  │ OR  │  Interactive     │      ┌─────────────────────┐
│  (text prompt)  │     │  (point/box)     │      │   Depth Edge Map    │
└────────┬────────┘     └────────┬─────────┘      └──────────┬──────────┘
         │                       │                           │
         └───────────┬───────────┘                           ▼
                     ▼                            ┌─────────────────────┐
          ┌─────────────────┐                     │  Edge Comparison    │
          │     SAM 2.1     │                     │  • Hard edges       │
          │ (video predict) │                     │  • Soft regions     │
          │                 │                     │  • Alignment fix    │
          │ • Temporal mem  │                     └──────────┬──────────┘
          │ • Multi-object  │                                │
          │ • Bi-direction  │                                ▼
          └────────┬────────┘                     ┌─────────────────────┐
                   │                              │   Guided Filter     │
                   ▼                              │   + Edge Refine     │
          ┌─────────────────┐                     └──────────┬──────────┘
          │  Rough Masks    │                                │
          │  (binary/soft)  │ ◄──────────────────────────────┘
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  EXR Export     │
          │  (16-bit float) │
          └─────────────────┘
```

---

## ⚠️ Troubleshooting

### "CUDA out of memory"
```bash
# Use smaller model
python auto_roto.py --input video.mp4 --prompt "person" --sam-model small

# Or process at lower resolution (resize video first)
ffmpeg -i video.mp4 -vf scale=1280:-1 video_small.mp4
```

### "GroundingDINO not found"
```bash
# Text prompts require GroundingDINO
# Install it or use box/point prompts instead
python auto_roto.py --input video.mp4 --box "100,100,500,400"
```

### "OpenEXR not available"
```bash
# EXR output falls back to OpenCV (limited)
# For full EXR support, install OpenEXR:
sudo apt-get install libopenexr-dev
pip install OpenEXR
```

### Flickering between frames
```bash
# Ensure backward propagation is enabled (default)
# Use larger model for better temporal consistency
python auto_roto.py --input video.mp4 --prompt "person" --sam-model large
```

### Poor edge quality
```bash
# Increase refinement
python auto_roto.py --input video.mp4 --prompt "person" \
    --refine-iterations 5 --edge-softness 2.0
```

---

## 📊 Performance

| Model | VRAM | Speed (1080p) | Quality |
|-------|------|---------------|---------|
| tiny | ~4GB | ~45 FPS | Good |
| small | ~5GB | ~35 FPS | Better |
| base_plus | ~7GB | ~25 FPS | Great |
| large | ~12GB | ~15 FPS | Best |

*Benchmarks on RTX 4090, may vary based on video complexity*

---

## 🔮 Roadmap

- [ ] Nuke plugin for in-app processing
- [ ] GUI interface (Gradio/Streamlit)
- [ ] Deep matting models (RVM, ViTMatte) for even better alpha
- [ ] Batch processing multiple videos
- [ ] Cloud processing option

---

## 📜 License

MIT License - use freely in commercial projects.

---

## 🙏 Acknowledgments

- [SAM2](https://github.com/facebookresearch/sam2) by Meta AI
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) by IDEA Research
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) for inspiration

---

**Built for VFX artists who need reliable auto-roto. No more hand-painting frames for simple shots.**
