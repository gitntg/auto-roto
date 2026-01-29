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

# Use MatAnyone instead of ViTMatte
python -m auto_roto pipeline --input video.mp4 --prompt "person" --refiner ma1
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

# Full pipeline
pipeline = FullPipeline(prompts=["person"], quality="high")
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
| MatAnyone | Temporal consistency, video | `--refiner ma1` |
| Interactive | Video with manual first-frame review | `interactive` command |

## Pipeline Stages

```
SAM3 → Depth → ViTMatte/MatAnyone → Combine → Final
```

1. **SAM3** - Segment objects using text/box/point prompts
2. **Depth** - Estimate depth maps with Depth Anything V3
3. **ViTMatte** or **MatAnyone** - Refine alpha
4. **Combine** - Produce final RGBA output with despill

## Output Structure

```
output/
├── final/
│   ├── alpha/      # Alpha mattes (EXR)
│   ├── rgba/       # RGBA composites
│   ├── preview/    # Preview images
│   └── depth/      # Depth maps
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ~12GB VRAM (large models)

## License

MIT
