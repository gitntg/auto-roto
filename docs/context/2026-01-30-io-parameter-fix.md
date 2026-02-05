# Context: I/O Parameter Order Fix & Depth Normalization Removal

**Date:** 2026-01-30
**Branch:** feature/MA2-integration
**Session Type:** Bug Fix & Refactoring

---

## Problem Summary

Two pipeline stages (`depth` and `combine`) were failing with:
```
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'ndarray'
```

## Root Cause

**Parameter order mismatch** between I/O function definitions and call sites:

| Function | Defined As | Called As |
|----------|------------|-----------|
| `save_depth_float` | `(depth, filepath)` | `(filepath, depth)` |
| `save_alpha` | `(alpha, filepath, ...)` | `(filepath, alpha, ...)` |

## Resolution

Changed I/O functions to use conventional `(filepath, data)` order (matching `cv2.imwrite`, standard file I/O patterns).

### Files Modified

1. **`auto_roto/io/depth.py`**
   - `save_depth_float(filepath, depth)` - swapped parameter order

2. **`auto_roto/io/alpha.py`**
   - `save_alpha(filepath, alpha, bit_depth)` - swapped parameter order

3. **`auto_roto/stages/depth.py`**
   - Removed `_normalize_depth()` method and calls
   - Removed unused `numpy` import
   - Now saves raw DA3 output (unnormalized)

4. **`auto_roto/models/matanyone.py`**
   - Updated `save_alpha` call to new parameter order

---

## Design Decision: Raw Depth Preservation

**Decision:** Remove depth normalization, save raw Depth Anything 3 output.

**Rationale:**
- DA3 outputs **relative depth** (not metric)
- Percentile normalization would corrupt relative depth relationships
- Raw values preserve model's intended depth representation
- Normalization can be applied downstream if needed for visualization

**Impact:**
- Depth EXR files now contain raw DA3 float32 values
- Downstream stages consuming depth should handle unnormalized values

---

## Depth Anything 3 Usage Notes

The implementation follows DA3 best practices:
- Uses official API: `DepthAnything3.from_pretrained(model_name)`
- Inference via `model.inference([image], process_res=..., process_res_method=...)`
- Supports models: small, base, large, nested-large, nested-base
- Stores depth as float32 EXR (single Y channel) for precision

---

## Call Sites Verified

### `save_depth_float`
- `stages/depth.py:124` ✓
- `stages/depth.py:155` ✓

### `save_alpha`
- `stages/combine.py:153` ✓
- `stages/vitmatte.py:184` ✓
- `models/matanyone.py:431` ✓ (fixed)

---

## Testing Notes

After this fix, the depth and combine stages should execute without TypeError. The pipeline will:
1. Run DA3 depth estimation
2. Save raw depth maps as float32 EXR
3. Continue to SAM/ViTMatte stages
4. Combine final outputs correctly
