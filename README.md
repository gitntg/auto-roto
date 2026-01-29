# AUTO-ROTO

Production-grade automatic rotoscoping using SAM3, Depth Anything V3, and ViTMatte.

# AUTO-ROTO

Production-grade automatic rotoscoping using SAM3, Depth Anything V3, and ViTMatte.

## Features

- **SAM3 Segmentation** - Text prompts with 270k+ concepts, no separate detector needed
- **Depth-Guided Refinement** - Depth Anything V3 for edge quality
- **ViTMatte Alpha Matting** - True alpha for hair and fine details
- **MatAnyone Video Matting** - Temporal consistency for video sequences
- **Production Output** - 16/32-bit EXR sequences

## Installation

```bash
# Create conda environment
conda create -n autoroto python=3.10
conda activate autoroto

# Create conda environment
conda create -n autoroto python=3.10
conda activate autoroto

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line
## Usage

### Command Line

```bash
# Full pipeline (recommended)
python -m auto_roto pipeline --input video.mp4 --prompt "person" --output ./output
# Full pipeline (recommended)
python -m auto_roto pipeline --input video.mp4 --prompt "person" --output ./output

# With quality preset
python -m auto_roto pipeline --input video.mp4 --prompt "person" --quality high
python -m auto_roto pipeline --input video.mp4 --prompt "person" --quality high

# SAM only (fastest)
python -m auto_roto sam --input video.mp4 --prompt "person" --output ./output
# SAM only (fastest)
python -m auto_roto sam --input video.mp4 --prompt "person" --output ./output

# Depth estimation only
python -m auto_roto depth --input ./frames --output ./output
```

### Python Library

```python
from auto_roto import FullPipeline

pipeline = FullPipeline(prompts=["person"], quality="high")
pipeline.run(input_path="video.mp4", output_dir="./output")
```

## Quality Presets
# Depth estimation only
python -m auto_roto depth --input ./frames --output ./output
```

### Python Library

```python
from auto_roto import FullPipeline

pipeline = FullPipeline(prompts=["person"], quality="high")
pipeline.run(input_path="video.mp4", output_dir="./output")
```

## Quality Presets

| Preset | Depth Model | Use Case |
|--------|-------------|----------|
| draft | small | Quick preview |
| standard | base | Production (default) |
| high | large | Hero shots |
| ultra | large | Maximum quality |

## Pipeline Stages

```
SAM3 → Depth → ViTMatte/MatAnyone → Combine → Final
```

1. **SAM3** - Segment objects using text/box/point prompts
2. **Depth** - Estimate depth maps with Depth Anything V3
3. **ViTMatte** or **MatAnyone** - Refine alpha (see below)
4. **Combine** - Produce final RGBA output with despill

### Alpha Refiners

| Refiner | Best For | Flag |
|---------|----------|------|
| ViTMatte | Hair, fine details, static shots | `--refiner vitmatte` (default) |
| MatAnyone | Temporal consistency, video | `--refiner ma1` |

```bash
# Use MatAnyone instead of ViTMatte
python -m auto_roto pipeline --input video.mp4 --prompt "person" --refiner ma1
```

## Output

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
MIT
