# AUTO-ROTO

Automatic rotoscoping pipeline: SAM3 segmentation, Depth Anything V3, ViTMatte alpha matting, and MatAnyone temporal propagation.

## Installation

```bash
conda create -n autoroto python=3.10
conda activate autoroto
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

### Interactive Mode (Recommended)

Review and refine the first-frame mask before processing the full video:

```bash
python -m auto_roto interactive --input video.mp4 --prompt "person"

# Cinema preset (depth + ViTMatte refinement before MatAnyone)
python -m auto_roto interactive --input video.mp4 --prompt "person" --preset cinema

# Multiple prompts (use . separator)
python -m auto_roto interactive --input video.mp4 --prompt "person.dog"
```

**Interactive Workflow:**
1. SAM3 segments first frame with your text prompt
2. Preview opens - review the mask
3. Refine with the SAM3 menu:
   - **[A] Accept** - approve mask
   - **[P] Add prompt** - add text prompts (e.g. "hair", "clothing")
   - **[S] More sensitive** - lower detection threshold
   - **[F] Fill holes** - fill interior mask gaps
4. *(Cinema preset: Depth + ViTMatte refine the mask automatically)*
5. **Final review** before MatAnyone:
   - **[A] Accept** - proceed to temporal propagation
   - **[M] Manual refine** - open the mask editor (brush/eraser/bezier tools)
   - **[Q] Quit** - cancel
6. MatAnyone propagates the approved mask to all frames

### Manual Mask Editor

When you choose **[M] Manual refine** at the final review step, an OpenCV drawing window opens. This is for fixing details that AI segmentation misses (thin objects, drawstrings, accessories separated from the body).

**Tools:**

| Tool | Key | Mouse |
|------|-----|-------|
| Brush (add to mask) | `B` | Left-click + drag |
| Eraser (remove from mask) | `E` | Left-click + drag |
| Bezier/Pen (smooth curves) | `P` | Left-click to place points, right-click to complete |

**Controls:**

| Key | Action |
|-----|--------|
| `[` / `]` | Brush size down/up |
| `1`-`5` | Brush presets (5, 10, 20, 40, 80 px) |
| `Z` | Undo |
| `R` | Reset all edits |
| `H` | Toggle help overlay |
| `Enter` | Accept edits |
| `Esc` | Cancel (discard edits) |

Edits are non-destructive (separate drawing layer merged on accept). The bezier tool uses B-spline interpolation for smooth curves through your control points.

### Full Pipeline (Batch)

```bash
# Standard quality with ViTMatte
python -m auto_roto pipeline --input video.mp4 --prompt "person" --output ./output

# High quality
python -m auto_roto pipeline --input video.mp4 --prompt "person" --quality high

# MatAnyone temporal mode
python -m auto_roto pipeline --input video.mp4 --prompt "person" --use-matanyone
```

### Individual Commands

```bash
python -m auto_roto sam --input video.mp4 --prompt "person" --output ./output
python -m auto_roto depth --input ./frames --output ./output
```

## Quality Presets

| Preset | Use Case |
|--------|----------|
| draft | Quick preview |
| standard | Production (default) |
| high | Final delivery |
| ultra | Maximum quality |

## Pipeline Stages

```
Standard:     SAM3 → Depth → ViTMatte → Combine → EXR
Temporal:     SAM3 → Depth → ViTMatte → Combine → MatAnyone → EXR
Interactive:  SAM3 → [Review] → Cinema(optional) → [Manual Edit] → MatAnyone → EXR
```

## Output

```
output/final/
├── alpha/    # Alpha mattes (EXR, 16/32-bit float)
├── rgba/     # RGBA composites (EXR)
├── preview/  # Preview PNGs
└── depth/    # Depth maps
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- ~12GB VRAM (large models)
