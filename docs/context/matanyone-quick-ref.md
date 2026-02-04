# MatAnyone Quick Reference

## One-Liner Commands

### Basic Usage
```bash
# Simplest MatAnyone usage (recommended for most videos)
python -m auto_roto pipeline --input video.mp4 --prompt "person" --use-matanyone
```

### Quality Presets
```bash
# Draft quality (fastest)
python -m auto_roto pipeline --input video.mp4 --prompt "person" --use-matanyone --quality draft

# Standard quality (balanced)
python -m auto_roto pipeline --input video.mp4 --prompt "person" --use-matanyone --quality standard

# High quality (best)
python -m auto_roto pipeline --input video.mp4 --prompt "person" --use-matanyone --quality high
```

### Custom Memory Settings
```bash
# Higher quality, more VRAM
python -m auto_roto pipeline --input video.mp4 --prompt "person" \
  --use-matanyone --matanyone-mem-every 2 --matanyone-max-mem-frames 15

# Lower VRAM usage
python -m auto_roto pipeline --input video.mp4 --prompt "person" \
  --use-matanyone --matanyone-mem-every 5 --matanyone-max-mem-frames 5
```

## Configuration Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use-matanyone` | bool | false | Enable MatAnyone temporal mode |
| `--matanyone-mem-every` | int | 3 | Memory update interval (lower=better, slower) |
| `--matanyone-max-mem-frames` | int | 10 | Max frames in memory (higher=better, more VRAM) |
| `--matanyone-warmup` | int | 5 | Warmup frames for stability |
| `--matanyone-erode` | int | 3 | Mask erosion iterations |
| `--matanyone-dilate` | int | 5 | Mask dilation iterations |

## When to Use MatAnyone

### ✅ Good For
- Video sequences (>10 frames)
- Consistent subjects (person/object stays in frame)
- Need temporal coherence
- Speed is important (10-20x faster)

### ❌ Not Ideal For
- Single frames
- Highly varying content (subject appears/disappears)
- Need independent per-frame results

## Workflow

```
Input Video
    ↓
SAM3 Segmentation (all frames)
    ↓
Depth Estimation (all frames)
    ↓
[IF --use-matanyone]
    ↓
MatAnyone Stage:
  1. Refine frame 0 only (with ViTMatte or MA1)
  2. Propagate to all frames (temporal memory)
    ↓
[ELSE]
    ↓
ViTMatte Stage:
  - Refine each frame independently
    ↓
Combine Stage (depth + alpha)
    ↓
Output Alpha Mattes
```

## Performance Guide

### Recommended Settings by Use Case

#### Fast Preview (30 FPS, 1080p)
```bash
--use-matanyone --quality draft \
--matanyone-mem-every 5 --matanyone-max-mem-frames 5
```

#### Balanced Production (24 FPS, 2K)
```bash
--use-matanyone --quality standard \
--matanyone-mem-every 3 --matanyone-max-mem-frames 10
```

#### High Quality Final (24 FPS, 4K)
```bash
--use-matanyone --quality high \
--matanyone-mem-every 2 --matanyone-max-mem-frames 15
```

#### VRAM Constrained (<8GB)
```bash
--use-matanyone --quality draft \
--matanyone-mem-every 5 --matanyone-max-mem-frames 3
```

## Troubleshooting

### Out of Memory
**Symptom**: CUDA OOM error
**Solution**: Reduce `--matanyone-max-mem-frames` or increase `--matanyone-mem-every`

### Temporal Artifacts
**Symptom**: Flickering or inconsistent edges
**Solution**:
- Increase `--matanyone-max-mem-frames`
- Decrease `--matanyone-mem-every`
- Increase `--matanyone-warmup`

### Slow Processing
**Symptom**: Slower than expected
**Solution**:
- Use `--quality draft`
- Increase `--matanyone-mem-every`
- Decrease `--matanyone-max-mem-frames`

## Interactive Setup Alternative

If you prefer interactive configuration:
```bash
python -m auto_roto interactive --input video.mp4 --prompt "person"
```

This launches an interactive workflow that asks for MatAnyone settings.

## Python API

For programmatic use:
```python
from auto_roto.pipelines.full_pipeline import FullPipeline
from auto_roto.config.pipeline import PipelineConfig

config = PipelineConfig(
    prompt="person",
    use_matanyone=True,
    matanyone_mem_every=3,
    matanyone_max_mem_frames=10,
    quality="high"
)

pipeline = FullPipeline(config=config)
pipeline.run(input_path="video.mp4", output_dir="./output")
```

## Files Modified for MatAnyone

- `auto_roto/cli/pipeline.py`: CLI arguments
- `auto_roto/config/pipeline.py`: Configuration schema
- `auto_roto/pipelines/full_pipeline.py`: Pipeline integration
- `auto_roto/stages/matanyone.py`: Stage implementation (NEW)
- `auto_roto/models/matanyone.py`: Model wrapper (NEW)

## Key Implementation Details

### Frame 0 Refinement
MatAnyone refines only the first frame using the selected refiner:
- `--refiner vitmatte`: Use ViTMatte for frame 0 (default)
- `--refiner ma1`: Use MatAnyone-1 for frame 0

### Temporal Propagation
After refining frame 0, MatAnyone propagates the alpha to all frames using:
- Previous frame alphas
- Memory bank of key frames
- Temporal consistency loss

### Memory Management
- Memory updated every `mem_every` frames
- Max `max_mem_frames` stored
- Old frames evicted FIFO
- GPU memory efficient

## Comparison: MatAnyone vs ViTMatte

| Aspect | MatAnyone | ViTMatte |
|--------|-----------|----------|
| Speed | 10-20x faster | Baseline |
| VRAM | Constant + buffer | Constant per frame |
| Temporal coherence | Excellent | None (independent) |
| Per-frame quality | Good | Excellent |
| Best for | Videos | Single frames |

## Advanced Tips

### Chain with Other Tools
```bash
# Extract frames first
ffmpeg -i video.mp4 frames/%04d.png

# Process with MatAnyone
python -m auto_roto pipeline --input frames/ --prompt "person" --use-matanyone

# Composite with original
ffmpeg -i video.mp4 -i output/alpha_%04d.png ...
```

### Batch Processing
```bash
for video in *.mp4; do
    python -m auto_roto pipeline --input "$video" --prompt "person" \
      --use-matanyone --output "output/${video%.mp4}"
done
```

### GPU Selection
```bash
CUDA_VISIBLE_DEVICES=0 python -m auto_roto pipeline \
  --input video.mp4 --prompt "person" --use-matanyone
```

## Related Commands

### Legacy Mode (No MatAnyone)
```bash
python -m auto_roto pipeline --input video.mp4 --prompt "person"
```

### Interactive Mode
```bash
python -m auto_roto interactive --input video.mp4 --prompt "person"
```

### SAM Only
```bash
python -m auto_roto sam --input video.mp4 --prompt "person"
```

## Support

For issues or questions:
1. Check git status: `git status`
2. Review recent commits: `git log --oneline -5`
3. See README.md for examples
4. Check docs/context/ for detailed context
