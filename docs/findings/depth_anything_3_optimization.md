# Depth Anything 3 Optimization Findings for Auto-Roto

## 1. The "Nested Model" Trap
The default configuration used `depth-model="nested-large"`.
*   **Mechanism**: It runs the **Giant** (Any-View) model to get a base depth map, and the **Metric-Large** model to get a scale factor. It then simply *multiplies* the Giant output by that scale.
*   **Issue**: The **Giant** model (`da3-giant`) uses extremely deep layers (`out_layers: [19, 27, 33, 39]`) and is trained for multi-view consistency. This makes it "hallucinate" smooth, solid geometry—great for 3D reconstruction but **bad for rotoscoping hair**, as it actively suppresses "noise" like stray hairs.
*   **Detail Loss**: The **Metric-Large** model uses shallower layers (`[4, 11, 17, 23]`) and a standard DPT head, which preserves far more local texture detail. By using "Nested," the fine detail from the Metric model was discarded, using only its global scale.

## 2. Resolution Caps
In `depth_refine.py`, resolution limits were hard-coded:
```python
MAX_PROCESS_RES_BY_MODEL = {
    "large": 1024,
    "nested-large": 2048,
}
```
*   `nested-large` was preferred because it allowed 2048px, but the underlying model architecture (Giant) blurred the details.
*   The `large` model (mapped to `DA3Mono-Large`) was artificially capped at 1024px. The `DA3` architecture (ViT-Large) can theoretically handle higher resolutions if VRAM permits.

## 3. Optimization Strategy
To improve fine detail preservation (hair strands):

1.  **Switch to `DA3Mono-Large`**: Use the `large` profile instead of `nested-large`. This uses the `da3mono-large` model which extracts features from much earlier layers (Layer 4 vs Layer 19), preserving high-frequency details.
2.  **Uncap Resolution**: Allow the `large` model to run at higher resolutions (2048px or native) to match `nested-large` capabilities.
3.  **RGB-Guided Trimap**: Implement trimap generation that uses RGB edges/gradients in addition to depth edges, ensuring that high-contrast visual features (hair against background) are included in the unknown region for matting even if the depth map is smooth.
