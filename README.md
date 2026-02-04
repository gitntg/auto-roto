# AUTO-ROTO

Production-grade automatic rotoscoping using SAM3, Depth Anything V3, ViTMatte, and MatAnyone.

## Features

- **SAM3 Segmentation** - Text prompts with 270k+ concepts, no separate detector needed
- **Depth-Guided Refinement** - Depth Anything V3 for edge quality
- **ViTMatte Alpha Matting** - True alpha for hair and fine details
- **MatAnyone Video Matting** - Temporal consistency for video sequences
- **Interactive Workflow** - Review and refine masks before processing
- **Production Output** - 16/32-bit EXR sequences

## Installation

```bash
# Create conda environment
conda create -n autoroto python=3.10
conda activate autoroto

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Interactive Mode (Recommended for Video)

The interactive workflow lets you review and refine the first-frame mask before processing the entire video with MatAnyone:

```bash
# Basic interactive workflow
python -m auto_roto interactive --input video.mp4 --prompt "person"

# With custom MatAnyone settings
python -m auto_roto interactive --input video.mp4 --prompt "person" --warmup 10 --erode 15 --dilate 15

# Multiple prompts (use . separator)
python -m auto_roto interactive --input video.mp4 --prompt "person.dog"
```

**Interactive Workflow:**
1. SAM3 segments first frame with your text prompt
2. Preview window opens - review the mask
3. Choose: **Accept** (Enter), **Add Points** (A), or **Quit** (Q)
4. If refining: click to add include (left-click) / exclude (right-click) points
5. Once approved, MatAnyone processes the entire video

**Resolution-Based Auto Settings:**
| Resolution | Warmup | Erode | Dilate |
|------------|--------|-------|--------|
| Low (≤576p) | 1 | 4 | 4 |
| High (>576p) | 10 | 15 | 15 |

Use `--no-auto-settings` to disable auto-adjustment.

### Full Pipeline

For batch processing with ViTMatte or MatAnyone:

```bash
# Full pipeline with ViTMatte (default)
python -m auto_roto pipeline --input video.mp4 --prompt "person" --output ./output

# With quality preset
python -m auto_roto pipeline --input video.mp4 --prompt "person" --quality high

# Use MatAnyone instead of ViTMatte (per-frame refinement)
python -m auto_roto pipeline --input video.mp4 --prompt "person" --refiner ma1

# MatAnyone temporal mode: refine frame 1 only, propagate to all frames
python -m auto_roto pipeline --input video.mp4 --prompt "person" --use-matanyone

# MatAnyone temporal mode with custom settings
python -m auto_roto pipeline --input video.mp4 --prompt "person" --use-matanyone --matanyone-mem-every 2
```

### Individual Commands

```bash
# SAM only (fastest, no refinement)
python -m auto_roto sam --input video.mp4 --prompt "person" --output ./output

# Depth estimation only
python -m auto_roto depth --input ./frames --output ./output

# Show version
python -m auto_roto version
```

### Python Library

```python
from auto_roto import FullPipeline, InteractiveMatAnyonePipeline
from auto_roto.config.pipeline import PipelineConfig

# Full pipeline with ViTMatte
pipeline = FullPipeline(prompts=["person"], quality="high")
pipeline.run(input_path="video.mp4", output_dir="./output")

# Full pipeline with MatAnyone temporal propagation
config = PipelineConfig(
    prompt="person",
    quality="high",
    use_matanyone=True,
    matanyone_mem_every=3,
    matanyone_max_mem_frames=10
)
pipeline = FullPipeline(config=config)
pipeline.run(input_path="video.mp4", output_dir="./output")

# Interactive pipeline
interactive = InteractiveMatAnyonePipeline(prompt="person")
interactive.run(input_path="video.mp4", output_dir="./output")
```

## Quality Presets

| Preset | Depth Model | Use Case |
|--------|-------------|----------|
| draft | small | Quick preview |
| standard | base | Production (default) |
| high | large | Hero shots |
| ultra | large | Maximum quality |

## Alpha Refiners

| Refiner | Best For | Flag |
|---------|----------|------|
| ViTMatte | Hair, fine details, static shots | `--refiner vitmatte` (default) |
| MatAnyone (per-frame) | Temporal consistency, video | `--refiner ma1` |
| MatAnyone (temporal) | Best quality video, refine once propagate all | `--use-matanyone` |
| Interactive | Video with manual first-frame review | `interactive` command |

## Pipeline Stages

**Standard Pipeline:**
```
SAM3 → Depth → ViTMatte/MatAnyone → Combine → Final
```

**MatAnyone Temporal Mode (`--use-matanyone`):**
```
Frame 1:     SAM3 → Depth → ViTMatte → Combine → Best alpha
All frames:  MatAnyone (propagate frame 1 alpha) → Final
```

1. **SAM3** - Segment objects using text/box/point prompts
2. **Depth** - Estimate depth maps with Depth Anything V3
3. **ViTMatte** or **MatAnyone** - Refine alpha
4. **Combine** - Produce final RGBA output with despill
5. **MatAnyone** (temporal mode) - Propagate frame 1 alpha to all frames

## Output Structure

```
output/
├── final/
│   ├── alpha/      # Alpha mattes (EXR)
│   ├── rgba/       # RGBA composites
│   ├── preview/    # Preview images
│   └── depth/      # Depth maps
```

**Intermediate directories (with `--keep-intermediate`):**
```
output/
├── 01_sam_output/alpha/        # SAM masks
├── 02_depth_output/depth/      # Depth maps
├── 03_vitmatte_output/alpha/   # ViTMatte refined alpha
├── 04_combine_output/alpha/    # Combined output
├── 05_matanyone_output/alpha/  # MatAnyone propagated (temporal mode)
└── final/
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ~12GB VRAM (large models)

## License

MIT
