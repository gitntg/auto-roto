# AUTO-ROTO: Claude Code Project Configuration

This file ensures consistent understanding across all Claude Code conversations.

## Quick Reference

| Item | Value |
|------|-------|
| **Python Command** | `python` (NOT `python3`) |
| **Environment** | `conda activate autoroto` or `autoroto_env` venv |
| **Main Entry** | `python -m auto_roto` or `python auto_roto.py` |
| **Test Command** | `python -m pytest tests/` |

---

## The Pipeline Architecture

```
Input Video/Sequence
       │
       ▼
┌─────────────────┐
│ Frame Extraction│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     SAM3        │  ◄── Text prompt → Semantic mask (knows "what")
│ (Segmentation)  │      Output: Binary core mask
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Depth Anything V3  │  ◄── Geometric guide (knows "where in 3D space")
│  (Depth Estimation) │      Output: Relative depth map
└────────┬────────────┘
         │
         ▼
┌─────────────────┐
│ Trimap Synthesis│  ◄── Combines SAM + Depth into Unknown zone
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    ViTMatte     │  ◄── Solves matting equation for alpha
│ (Alpha Matting) │      Output: Fractional alpha (0.0-1.0)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Manual Editor  │  ◄── Human-in-the-loop (optional, interactive only)
│ (Mask Refinement│      Brush, eraser, bezier tools
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MatAnyone     │  ◄── Temporal consistency (optional)
│ (Video Matting) │      Output: Temporally coherent alpha sequence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   EXR Export    │  ◄── 16/32-bit float preservation
└─────────────────┘
```

---

## Model Roles (DO NOT CONFUSE)

### SAM3 (Segment Anything Model 3)
- **Role**: Semantic segmentation - "knows WHAT the object is"
- **Input**: Image + text prompt (e.g., "person")
- **Output**: Binary mask (0 or 1) - the "core matte"
- **Limitation**: Produces HARD edges only, no transparency
- **Via**: Ultralytics (`ultralytics>=8.3.237`)

### Depth Anything V3
- **Role**: Geometric guide - "knows WHERE edges are in 3D space"
- **Input**: RGB image
- **Output**: Relative depth map (float values)
- **Use**: Creates "depth spikes" that indicate hair/fine detail locations
- **Critical**: Use HIGH-PASS FILTER (Laplacian) on depth, NOT absolute thresholds

### ViTMatte
- **Role**: Alpha matte solver - "solves the matting equation"
- **Input**: RGB image + Trimap (White=FG, Black=BG, Gray=Unknown)
- **Output**: Fractional alpha values (0.0-1.0)
- **Critical Requirements**:
  - Trimap "Unknown" zone must be **20-40 pixels wide** (NOT thin lines)
  - Needs patch context (16x16 or 32x32) to calculate gradients
  - Process in **Linear color space** (32-bit float)

### MatAnyone
- **Role**: Temporal video matting - "ensures frame-to-frame consistency"
- **Input**: Video + first-frame mask
- **Output**: Temporally coherent alpha sequence
- **Use**: Prevents flicker, maintains edge consistency across frames
- **Settings**: warmup, erode, dilate, mem_every

### MaskRefinementEditor (Human-in-the-Loop)
- **Role**: Manual mask correction - "fixes what AI misses"
- **Input**: RGB frame + current mask (from SAM3 or cinema refinement)
- **Output**: Refined mask with user additions/subtractions merged in
- **Tools**: Brush (add), Eraser (remove), Bezier/Pen (smooth curves)
- **When**: Interactive mode, after all AI processing, before MatAnyone
- **Code**: `auto_roto/utils/mask_editor.py`

---

## The Matting Equation

**This is the fundamental physics. Claude MUST understand this.**

```
I = α·F + (1-α)·B
```

Where:
- `I` = Observed pixel color (what the camera captured)
- `F` = True foreground color
- `B` = True background color
- `α` = Alpha/transparency (0.0 to 1.0)

**The Problem**: 3 unknowns (F, B, α) but only 1 known (I) per pixel.

**Why Models Fail**: They try to CLASSIFY pixels (α=0 or α=1) instead of SOLVING for α.

**The Truth**: A pixel of hair is often **60% background / 40% foreground**. The alpha MUST remain fractional.

---

## Trimap Requirements (Critical Numbers)

| Parameter | Value | Why |
|-----------|-------|-----|
| Unknown width | 20-40 px | ViTMatte attention needs context |
| Core erosion | 10 px | Ensure definite foreground |
| Search dilation | 25-50 px | Max hair strand distance from body |
| Depth variance zone | 0.4-0.6 | Mark as "Unknown", not foreground |

**The "Leash" Rule**: Hair detail can ONLY exist within ~50px of the SAM body mask. Anything further is background noise.

---

## Code Conventions

### Python Environment
```bash
# Use this - NOT python3
python -m auto_roto pipeline --input video.mp4 --prompt "person"

# Test syntax
python -m py_compile auto_roto.py

# Run tests
python -m pytest tests/ -v
```

### File Formats
- **Input**: MP4, MOV, image sequences (PNG, JPG, EXR)
- **Output**: Always EXR (16/32-bit float) to preserve alpha precision
- **Never**: Save alpha to 8-bit PNG (crushes subtle values)

### Quality Presets
| Preset | Use Case |
|--------|----------|
| draft | Quick preview, testing |
| standard | Normal production |
| high | Final delivery |
| ultra | Maximum quality, slow |

---

## Common Pitfalls (Avoid These)

1. **Hard edges instead of soft mattes**
   - Cause: Trimap "Unknown" zone too thin
   - Fix: Dilate unknown region to 20-40px

2. **Background noise in trimap**
   - Cause: Depth edges not constrained to body area
   - Fix: Apply "leash" - `bitwise_and(depth_edges, dilated_sam_mask)`

3. **Wispy hair disappearing**
   - Cause: Saving to 8-bit or thresholding alpha
   - Fix: Keep 32-bit float, never use `alpha > 0.5`

4. **Temporal flicker**
   - Cause: Per-frame processing without temporal model
   - Fix: Use MatAnyone temporal mode

5. **Wrong python command**
   - Cause: Using `python3` on Windows
   - Fix: Always use `python`

---

## Directory Structure

```
auto_roto/
├── auto_roto.py               # Main CLI entry point
├── requirements.txt           # Dependencies
├── CLAUDE.md                  # This file (project context)
├── SOURCEOFTRUTH.md           # Deep technical reference
├── README.md                  # User documentation
├── tests/                     # pytest tests
├── auto_roto/
│   ├── pipelines/
│   │   ├── interactive_matanyone.py  # Interactive workflow + final review gate
│   │   └── full_pipeline.py          # Batch pipeline
│   ├── utils/
│   │   ├── mask_editor.py            # MaskRefinementEditor (brush/eraser/bezier)
│   │   └── ...
│   ├── models/                # SAM3, DepthAnything, MatAnyone wrappers
│   ├── stages/                # Pipeline stage implementations
│   ├── refiners/              # Alpha refinement (geometric, depth expansion)
│   ├── config/                # Configuration dataclasses
│   ├── cli/                   # CLI argument parsing
│   └── io/                    # File I/O (EXR, depth, alpha)
├── .claude/                   # Claude Code configuration
├── Depth-Anything-V2/         # Depth model (bundled)
├── depth-anything-3/          # DA3 (bundled)
└── checkpoints/               # Model weights
```

---

## For Claude: Behavioral Guidelines

1. **Always use `python`, never `python3`** on this Windows project
2. **Reference SOURCEOFTRUTH.md** for deep technical questions about matting
3. **The models have distinct roles** - don't confuse SAM (segmentation) with ViTMatte (matting)
4. **Alpha is a ratio, not a classification** - values between 0 and 1 are expected and correct
5. **Test changes** with `python -m py_compile` before committing
6. **Preserve float precision** - never suggest 8-bit outputs for alpha
