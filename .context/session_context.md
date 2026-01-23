# Auto-Roto Session Context

**Captured:** 2026-01-22 23:05 UTC
**Branch:** `feature/alpha-quality-v5`
**Version:** 5.1.0

---

## Project Overview

AI-driven semantic segmentation and matte extractor for VFX rotoscoping using:
- **SAM2** for semantic segmentation
- **Depth Anything V2** for geometry
- **ViTMatte** for alpha refinement

## SOURCEOFTRUTH Implementation (Lines 77-187)

### Hard Numbers Applied

| Parameter | Old Value | SOURCEOFTRUTH Value |
|-----------|-----------|---------------------|
| Core erosion | 5px | **10px** |
| Unknown dilation | 80px | **25px** |
| Hair boost | 40px | 15px |
| Soft trimap | On | **Off** (hard 128) |

### Depth Confidence Intervals

```
Core Body:     Depth > 0.8  → Definite FG (255)
Background:    Depth < 0.2  → Definite BG (0)
Matting Zone:  0.4 - 0.6    → Unknown (128)
```

### Golden Rule Implementation

```python
# SOURCEOFTRUTH Golden Rule (vitmatte_refine.py lines 395-506)
core_mask = erode(sam_mask, 10px)       # Definite foreground
wide_search = dilate(sam_mask, 25px)    # Context window
# Everything between = Unknown (128)
```

## Key Files

| File | Purpose |
|------|---------|
| `vitmatte_refine.py` | ViTMatte + Scientific Trimap |
| `SOURCEOFTRUTH.md` | Design specification |
| `depth_refine.py` | Depth-based refinement |
| `full_pipeline_v5.py` | V5 orchestration |

## Benchmark Results

| Method | Hair Soft Pixels | vs SAM2 |
|--------|------------------|---------|
| SAM2 Original | 14,063 | baseline |
| SCIENTIFIC ViTMatte | 18,389 | **+31%** |
| Depth Refine v10 | 34,867 | +148% |

**Unknown Zone Width:** 35px (meets 25-40px requirement)

## Session Changes

1. ✅ Implemented scientific trimap mode (`TrimapConfig.scientific_mode = True`)
2. ✅ Added linear color space conversion functions
3. ✅ Added depth confidence interval parameters
4. ✅ Added Guided Filter post-processing
5. ✅ Cleaned up repo (removed 830MB test outputs, 12 debug scripts)

## New: Adaptive Trimap Mode (v5.2)

Integrated **Formulaic Distance Approach** + **Motion-Aware Trimap**:

| Feature | Description |
|---------|-------------|
| `--adaptive-mode` | Replace fixed dilation with depth gradient-based variance |
| `--motion-aware` | Expand unknown zone based on optical flow |

### Adaptive Mode Formula
```
dynamic_threshold = base_px + (max_px * complexity_map) + (weight * motion_factor)
```

- Smooth regions (shoulders): ~2px unknown
- Complex regions (hair): up to 60px unknown
- Motion areas: additional expansion for blur

### CLI Examples
```bash
# Adaptive mode only
python vitmatte_refine.py ... --adaptive-mode

# Adaptive + Motion-aware
python vitmatte_refine.py ... --adaptive-mode --motion-aware

# Custom parameters
python vitmatte_refine.py ... --adaptive-mode \
  --adaptive-base 3 --adaptive-max 50 \
  --motion-aware --motion-weight 25
```

## Pending Work

- [ ] Install `opencv-contrib-python` for Guided Filter
- [ ] Tune depth intervals for specific footage
- [ ] Consider ViTMatte + Depth Refine hybrid approach
- [ ] Test adaptive mode on hair-heavy footage

## CLI Quick Reference

```bash
# Scientific mode (default)
python vitmatte_refine.py \
  --sam-mask ./test_output/alpha/ \
  --depth ./test_output_refined_v10/depth/ \
  --frames ./test_input/ \
  --output ./output/ \
  --save-trimap --verbose

# Custom depth intervals
python vitmatte_refine.py ... \
  --depth-unknown-low 0.3 \
  --depth-unknown-high 0.7

# Legacy mode (pre-SOURCEOFTRUTH)
python vitmatte_refine.py ... --legacy-mode
```

---

*Context fingerprint: `sourceoftruth-scientific-trimap-v1`*
