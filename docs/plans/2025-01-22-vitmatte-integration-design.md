# ViTMatte Integration Design for Hair Detail Extraction

**Date:** 2025-01-22
**Status:** Ready for Implementation
**Branch:** feature/alpha-quality-v5

---

## Problem Statement

Current `depth_refine.py` attempts to detect and extend hair edges using heuristics:
- Depth-based foreground/background separation
- Texture analysis (Laplacian)
- Color matching
- Morphological operations

**Result:** "Alpha turned down to 50% and edge extended sitting on top of smeared hair"

The approach **guesses alpha values** from heuristics rather than solving for them. This produces uniform semi-transparent halos instead of actual hair strand detail.

---

## Solution: Geometric-Guided Image Matting

Follow the SOURCEOFTRUTH.md approach:

1. Use SAM2 mask as "Core Foreground"
2. Use Depth high-pass to find "Geometric Strands"
3. Build a **Trimap** (Foreground/Background/Unknown)
4. Let **ViTMatte** solve for actual alpha values

```
SAM2 Mask + Depth High-Pass → Trimap → ViTMatte → Final Alpha
     ↓              ↓              ↓
  "Core FG"    "Unknown Zone"   "Solve for α"
   (white)        (gray)         (0.0-1.0)
```

---

## Trimap Synthesis

### WHITE (255) - Definite Foreground
- Eroded SAM2 mask (shrink by 5-10px)
- 100% guaranteed subject pixels
- ViTMatte outputs α=1.0

### BLACK (0) - Definite Background
- Far from SAM2 boundary AND background-level depth
- 100% guaranteed non-subject pixels
- ViTMatte outputs α=0.0

### GRAY (128) - Unknown Zone
Where hair lives. Found via:

```python
1. Find SAM2 boundary (edge pixels)
2. Apply HIGH-PASS FILTER to depth map (Laplacian)
3. High-pass reveals "depth spikes" - sudden depth changes
4. Mark as UNKNOWN any pixel that:
   - Has depth spike (high-pass > threshold) AND
   - Is within 30-50px of SAM2 boundary AND
   - Has foreground-like depth (not background plane)
```

### Constraint: Geometric Continuity
A strand must have continuous depth path back to scalp. Isolated "floating" depth noise is filtered (rejects wires/debris).

---

## ViTMatte Integration

### Model Selection
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `hustvl/vitmatte-small-composition-1k` | ~90MB | Fast | Good |
| `hustvl/vitmatte-base-composition-1k` | ~350MB | Medium | Better |

### Usage
```python
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-base-composition-1k")
model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-base-composition-1k")

inputs = processor(images=rgb_image, trimaps=trimap, return_tensors="pt")
outputs = model(**inputs)
alpha = outputs.alphas
```

### Memory
ViTMatte is lightweight (~350MB). Can coexist with Depth Anything V2 in VRAM.

---

## Implementation Architecture

### New File: `vitmatte_refine.py`

```python
class TrimapSynthesizer:
    """Builds trimap from SAM2 mask + Depth map"""

    def __init__(self, config: TrimapConfig):
        self.core_erosion = config.core_erosion      # px to erode for definite FG
        self.unknown_radius = config.unknown_radius  # px from boundary for unknown
        self.depth_threshold = config.depth_threshold
        self.highpass_threshold = config.highpass_threshold

    def synthesize(self, sam_mask, depth, rgb=None) -> np.ndarray:
        """
        Returns trimap: 0=background, 128=unknown, 255=foreground
        """
        pass

class ViTMatteRefiner:
    """Runs ViTMatte on RGB + Trimap"""

    def __init__(self, model_size="base", device="cuda"):
        self.model = ...
        self.processor = ...

    def refine(self, rgb, trimap) -> np.ndarray:
        """Returns alpha matte (H, W), float32 0-1"""
        pass

class GeometricMatteRefiner:
    """Full pipeline: SAM + Depth → Trimap → ViTMatte → Alpha"""

    def __init__(self, config):
        self.trimap_synth = TrimapSynthesizer(config.trimap)
        self.vitmatte = ViTMatteRefiner(config.model_size)

    def process_frame(self, rgb, sam_mask, depth) -> np.ndarray:
        trimap = self.trimap_synth.synthesize(sam_mask, depth, rgb)
        alpha = self.vitmatte.refine(rgb, trimap)
        return alpha
```

### Pipeline Integration

**Option A:** New stage in v5 pipeline (after depth, before edge)
```
SAM2 → Depth → ViTMatte Refine → Edge → Temporal → Combine
```

**Option B:** Replace depth_refine.py for hair-focused shots
```
SAM2 → ViTMatte Refine (uses depth internally) → Edge → Temporal
```

Recommend **Option A** for flexibility.

---

## Test Strategy

### Phase 1: Validate Trimap
- Generate trimap from 3 test frames in `test_input/`
- Visual inspection: Does unknown zone capture hair strands?
- Save trimap visualization (white/gray/black)

### Phase 2: Validate ViTMatte
- Run ViTMatte on RGB + Trimap
- Compare to current `test_output_refined_v10/`
- Check: Are hair strands properly resolved?

### Phase 3: Tune & Iterate
- Adjust high-pass threshold
- Adjust erosion/dilation sizes
- Test on more footage

---

## Test Data

**Input:** `test_input/`
- `denoised-floatingCarCrash_125344.png`
- `denoised-floatingCarCrash_125345.png`
- `denoised-floatingCarCrash_125346.png`

**Current Output:** `test_output_refined_v10/`
- Shows stunt performer on wires
- Depth map clearly shows hair strands (bright yellow tendrils)
- Current refinement not capturing them

**Challenge:** Must capture hair but reject wire/cable artifacts

---

## Success Criteria

1. Hair strands visible in final alpha (not clipped at scalp)
2. Strand transparency varies naturally (not uniform 50% gray)
3. Wires/cables NOT included in matte
4. Core body remains solid (no erosion of torso/limbs)
5. Temporal consistency maintained across frames

---

## Dependencies

```
transformers>=4.30.0  # For ViTMatte
torch>=2.0.0
numpy
opencv-python
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `vitmatte_refine.py` | CREATE | New ViTMatte integration module |
| `full_pipeline_v5.py` | MODIFY | Add ViTMatte stage option |
| `requirements.txt` | MODIFY | Add transformers if missing |

---

## Implementation Tasks

1. Create `vitmatte_refine.py` with TrimapSynthesizer class
2. Implement depth high-pass trimap synthesis
3. Add ViTMatteRefiner class with HuggingFace integration
4. Create GeometricMatteRefiner orchestration class
5. Test trimap generation on 3 test frames
6. Test full ViTMatte pipeline on test frames
7. Compare results to current output
8. Integrate into v5 pipeline (optional stage)
9. Add CLI arguments for ViTMatte options
