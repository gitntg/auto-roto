# AUTO-ROTO V5 Pipeline Context

**Captured**: 2026-01-22
**Branch**: `feature/alpha-quality-v5`
**Version**: 5.0.0

---

## Project Summary

AUTO-ROTO is a production-grade automatic rotoscoping pipeline that generates professional-quality alpha mattes from video clips. V5 adds professional VFX compositing techniques for significantly improved edge quality.

## V5 Enhancement Overview

```
Pipeline Flow:
SAM2 → Depth Refine → Edge Refine → Temporal Smooth → Matte Combine → Final
```

### New Modules

| Module | Purpose |
|--------|---------|
| `edge_refine.py` | Subpixel edges, color keying, despill, premultiplied alpha |
| `temporal_smooth.py` | Optical flow, temporal filtering, anti-flicker |
| `matte_combine.py` | Multi-layer mattes, blend modes, light wrap |
| `full_pipeline_v5.py` | Integrated pipeline with quality presets |

---

## Key Technical Decisions

### 1. Multi-Layer Matte Architecture
- **Core Matte**: 100% solid interior, eroded from edge
- **Detail Matte**: Semi-transparent hair/fine detail
- **Soft Edge Matte**: Anti-aliased boundary transitions
- **Rationale**: Matches Nuke/Flame professional workflows

### 2. Subpixel Edge Detection
- 4x upscaling with Laplacian-of-Gaussian
- Provides anti-aliased edges without artifacts
- Configurable precision level

### 3. Temporal Coherence
- Farneback optical flow for motion tracking
- Motion-compensated median filtering
- Keyframe anchoring prevents drift
- Max-delta limiting prevents flicker

### 4. Premultiplied Alpha
- Industry standard for compositing
- Proper OVER/SCREEN operations
- Edge color preservation

---

## Quality Presets

| Preset | SAM Model | Depth Model | Temporal Window | Use Case |
|--------|-----------|-------------|-----------------|----------|
| draft | tiny | small | 3 | Quick preview |
| standard | base_plus | base | 5 | Production |
| high | large | large | 7 | Hero shots |
| ultra | large | large | 9 | Maximum |

---

## Current Test Sequence

**CarCrash Sequence**
- Path: `/mnt/f/3DActive/CarCrash/reel-new/renders/scene-01nfp`
- Frames: 125344-125603 (262 total)
- Pattern: `denoised-floatingCarCrash_######.png`
- Status: Ready for testing

---

## Usage Commands

```bash
# Basic v5 pipeline
python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output

# High quality
python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output --quality high

# CarCrash sequence
python full_pipeline_v5.py \
    --input "/mnt/f/3DActive/CarCrash/reel-new/renders/scene-01nfp" \
    --prompt "car" \
    --output ./carcrash_v5 \
    --quality high \
    --verbose

# Custom settings
python full_pipeline_v5.py --input video.mp4 --prompt "person" --output ./output \
    --edge-softness 1.5 --core-shrink 5 --despill 0.7
```

---

## Git State

```
827608a docs: Add v5 pipeline documentation to README
249798c feat: Add professional VFX quality modules for v5 pipeline
069b29a Initial commit: Auto-Roto pipeline
```

---

## Next Steps

1. Activate conda environment with PyTorch
2. Run v5 pipeline on CarCrash sequence
3. Tune parameters for specific footage
4. Consider `--with-hair` for subjects with hair

---

## Session Notes

- Virtual environment `autoroto_env` exists but needs PyTorch
- User should use their conda environment instead
- All new modules pass Python syntax validation
- README updated with v5 documentation
