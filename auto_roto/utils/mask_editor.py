"""
Mask Refinement Editor
======================

Interactive OpenCV-based mask editor with brush, eraser, and bezier curve tools.
Used as a Human-In-The-Loop step before MatAnyone temporal propagation.

Performance strategy:
  - Full-res drawing layer for precision
  - Downscaled display compositing (~1600x900) for 4K+ content
  - Two-tier rendering: expensive mask composite is cached; cursor/overlays
    are drawn cheaply on a copy each frame

Usage:
    editor = MaskRefinementEditor(rgb_frame, current_mask)
    result = editor.run()
    if result is not None:
        # User accepted - result is the refined mask (H, W) float32
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("AutoRoto.Utils.MaskEditor")


class MaskRefinementEditor:
    """
    Interactive mask editor with brush, eraser, and bezier tools.

    Tools:
        - Brush (B): Left-click + drag to add to mask
        - Eraser (E): Left-click + drag to remove from mask
        - Bezier/Pen (P): Left-click to place control points, right-click to complete

    Keyboard:
        B/E/P   - Switch tools
        D       - Toggle soft brush (Gaussian feather)
        [ / ]   - Brush size down/up (+-5px)
        1-5     - Brush size presets (5, 10, 20, 40, 80)
        Z       - Undo
        R       - Reset all edits
        H       - Toggle help overlay
        Enter   - Accept (completes pending bezier first)
        Esc/Q   - Cancel (discard edits)
    """

    MAX_UNDO = 20
    DISPLAY_MAX_W = 1600
    DISPLAY_MAX_H = 900

    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        window_name: str = "Mask Refinement",
    ):
        self.base_image = image.copy()
        self.original_mask = np.squeeze(mask).astype(np.float32)
        self.drawing_layer = np.zeros(self.original_mask.shape, dtype=np.float32)
        self.window_name = window_name

        # Tool state
        self.current_tool = "brush"
        self.brush_size = 10
        self.soft_brush = False
        self.is_drawing = False
        self.last_point: Optional[Tuple[int, int]] = None

        # Track stroke bounding box for region-local soft brush
        self._stroke_bbox: Optional[Tuple[int, int, int, int]] = None  # x1,y1,x2,y2

        # Bezier state
        self.bezier_points: List[Tuple[int, int]] = []

        # UI state
        self.mouse_pos = (0, 0)
        self.show_help = True
        self.undo_stack: List[np.ndarray] = []

        # Precompute display scaling for performance on 4K+
        h, w = self.base_image.shape[:2]
        self._full_h, self._full_w = h, w
        self._scale = min(self.DISPLAY_MAX_W / w, self.DISPLAY_MAX_H / h, 1.0)
        self._disp_w = int(w * self._scale)
        self._disp_h = int(h * self._scale)

        # Downscale base image once (uint8 RGB at display resolution)
        if self._scale < 1.0:
            self._disp_base = cv2.resize(
                self.base_image, (self._disp_w, self._disp_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            self._disp_base = self.base_image.copy()

        # Two-tier rendering:
        #   _disp_composite = cached mask overlay (expensive, rebuild only on mask changes)
        #   Each frame: copy _disp_composite, draw cursor/bezier/status (cheap)
        self._disp_composite: Optional[np.ndarray] = None
        self._composite_dirty = True  # Needs rebuild

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _disp_to_full(self, x: int, y: int) -> Tuple[int, int]:
        """Convert display coordinates to full-resolution coordinates."""
        if self._scale >= 1.0:
            return x, y
        return int(x / self._scale), int(y / self._scale)

    def _disp_brush_size(self) -> int:
        """Brush size in display pixels."""
        return max(1, int(self.brush_size * self._scale))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Optional[np.ndarray]:
        """
        Open the editor window and run the event loop.

        Returns:
            Merged mask (float32, 0-1, shape H,W) if accepted, or None if cancelled.
        """
        self._setup_window()
        self._rebuild_composite()
        self._render_frame()

        logger.info("Mask editor opened - draw additions or press Enter to accept")

        try:
            while True:
                key = cv2.waitKey(30) & 0xFF

                # Check if window was closed
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        return None
                except cv2.error:
                    return None

                result = self._handle_key(key)
                if result == "accept":
                    logger.info("Mask edits accepted")
                    return self._get_merged_mask()
                elif result == "cancel":
                    logger.info("Mask edits cancelled")
                    return None

                # Rebuild expensive composite only when mask data changed
                if self._composite_dirty:
                    self._rebuild_composite()
                    self._composite_dirty = False

                # Lightweight render (cursor, overlays) every frame
                self._render_frame()

        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self._disp_w, self._disp_h)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def _handle_key(self, key: int) -> Optional[str]:
        if key == 255:
            return None

        # Accept
        if key in (13, 10):
            if self.bezier_points:
                self._complete_bezier_stroke()
                return None
            return "accept"

        # Cancel
        if key in (27, ord("q"), ord("Q")):
            return "cancel"

        # Tool switching
        if key in (ord("b"), ord("B")):
            self.current_tool = "brush"
            self.bezier_points.clear()
        elif key in (ord("e"), ord("E")):
            self.current_tool = "eraser"
            self.bezier_points.clear()
        elif key in (ord("p"), ord("P")):
            self.current_tool = "bezier"

        # Soft brush toggle
        elif key in (ord("d"), ord("D")):
            self.soft_brush = not self.soft_brush

        # Brush size
        elif key == ord("["):
            self.brush_size = max(1, self.brush_size - 5)
        elif key == ord("]"):
            self.brush_size = min(200, self.brush_size + 5)
        elif key == ord("1"):
            self.brush_size = 5
        elif key == ord("2"):
            self.brush_size = 10
        elif key == ord("3"):
            self.brush_size = 20
        elif key == ord("4"):
            self.brush_size = 40
        elif key == ord("5"):
            self.brush_size = 80

        # Undo
        elif key in (ord("z"), ord("Z")):
            self._pop_undo()

        # Reset
        elif key in (ord("r"), ord("R")):
            self._push_undo()
            self.drawing_layer[:] = 0
            self.bezier_points.clear()
            self._composite_dirty = True

        # Help toggle
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help

        return None

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        self.mouse_pos = (x, y)

        if self.current_tool == "bezier":
            self._mouse_bezier(event, x, y)
        else:
            self._mouse_brush_eraser(event, x, y)

    def _mouse_brush_eraser(self, event: int, x: int, y: int):
        fx, fy = self._disp_to_full(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self._push_undo()
            self.is_drawing = True
            self.last_point = (fx, fy)
            self._stroke_bbox = (fx, fy, fx, fy)
            self._draw_stroke(fx, fy)
            self._composite_dirty = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.last_point is not None:
                self._draw_stroke(fx, fy, from_point=self.last_point)
                self.last_point = (fx, fy)
                self._expand_stroke_bbox(fx, fy)
                self._composite_dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing:
                if self.soft_brush and self._stroke_bbox is not None:
                    self._apply_feather_region(self._stroke_bbox)
                    self._composite_dirty = True
                self.is_drawing = False
                self.last_point = None
                self._stroke_bbox = None

    def _mouse_bezier(self, event: int, x: int, y: int):
        fx, fy = self._disp_to_full(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.bezier_points.append((fx, fy))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.bezier_points:
                self._complete_bezier_stroke()

    # ------------------------------------------------------------------
    # Drawing operations (full resolution)
    # ------------------------------------------------------------------

    def _draw_stroke(self, x: int, y: int, from_point: Optional[Tuple[int, int]] = None):
        value = 1.0 if self.current_tool == "brush" else -1.0
        thickness = max(1, self.brush_size)

        if from_point is not None:
            cv2.line(self.drawing_layer, from_point, (x, y), value, thickness)
        else:
            cv2.circle(self.drawing_layer, (x, y), thickness // 2, value, -1)

    def _expand_stroke_bbox(self, x: int, y: int):
        """Grow the stroke bounding box to include point (x, y)."""
        if self._stroke_bbox is None:
            self._stroke_bbox = (x, y, x, y)
        else:
            x1, y1, x2, y2 = self._stroke_bbox
            self._stroke_bbox = (min(x1, x), min(y1, y), max(x2, x), max(y2, y))

    def _apply_feather_region(self, bbox: Tuple[int, int, int, int]):
        """Apply Gaussian blur ONLY within the stroke bounding box + padding."""
        k = max(3, (self.brush_size // 2) | 1)
        sigma = k / 3.0
        pad = self.brush_size + k  # padding around stroke

        x1, y1, x2, y2 = bbox
        # Expand by pad + brush radius, clamp to image bounds
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(self._full_w, x2 + pad)
        ry2 = min(self._full_h, y2 + pad)

        region = self.drawing_layer[ry1:ry2, rx1:rx2]
        if region.size > 0:
            self.drawing_layer[ry1:ry2, rx1:rx2] = cv2.GaussianBlur(
                region, (k, k), sigma,
            )

    def _complete_bezier_stroke(self):
        if len(self.bezier_points) < 2:
            self.bezier_points.clear()
            return

        self._push_undo()

        curve = self._generate_bezier_curve(self.bezier_points)
        thickness = max(1, self.brush_size)

        for i in range(len(curve) - 1):
            pt1 = tuple(curve[i])
            pt2 = tuple(curve[i + 1])
            cv2.line(self.drawing_layer, pt1, pt2, 1.0, thickness)

        if self.soft_brush:
            # Compute bbox of the entire curve
            xs = curve[:, 0]
            ys = curve[:, 1]
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            self._apply_feather_region(bbox)

        self.bezier_points.clear()
        self._composite_dirty = True

    def _generate_bezier_curve(
        self, points: List[Tuple[int, int]], num_samples: int = 300
    ) -> np.ndarray:
        if len(points) < 2:
            return np.array(points, dtype=np.int32)

        points_arr = np.array(points, dtype=np.float64)

        if len(points) == 2:
            t = np.linspace(0, 1, num_samples)
            x = points_arr[0, 0] + t * (points_arr[1, 0] - points_arr[0, 0])
            y = points_arr[0, 1] + t * (points_arr[1, 1] - points_arr[0, 1])
            return np.column_stack([x, y]).astype(np.int32)

        try:
            from scipy.interpolate import splprep, splev
            k = min(3, len(points) - 1)
            tck, u = splprep([points_arr[:, 0], points_arr[:, 1]], s=0, k=k)
            u_new = np.linspace(0, 1, num_samples)
            x_new, y_new = splev(u_new, tck)
            return np.column_stack([x_new, y_new]).astype(np.int32)
        except Exception:
            return points_arr.astype(np.int32)

    # ------------------------------------------------------------------
    # Two-tier display rendering
    # ------------------------------------------------------------------

    def _rebuild_composite(self):
        """
        EXPENSIVE: Rebuild the cached mask-over-frame composite at display resolution.
        Only called when drawing layer actually changes.
        """
        # Combine masks at full res, then downscale once
        if self._scale < 1.0:
            combined_full = np.clip(
                self.original_mask + self.drawing_layer, 0.0, 1.0,
            )
            combined = cv2.resize(
                combined_full, (self._disp_w, self._disp_h),
                interpolation=cv2.INTER_AREA,
            )
        else:
            combined = np.clip(
                self.original_mask + self.drawing_layer, 0.0, 1.0,
            )

        # Green overlay on downscaled frame
        base = self._disp_base.astype(np.float32)
        mask_3ch = combined[:, :, np.newaxis]
        composited = base * (1.0 - mask_3ch * 0.5) + 255.0 * mask_3ch * np.array([0, 0.5, 0])
        self._disp_composite = composited.astype(np.uint8)

    def _render_frame(self):
        """
        CHEAP: Copy cached composite, draw cursor/bezier/overlays, show.
        Called every frame (~30ms). Cost: array copy + a few cv2 draw calls.
        """
        display = self._disp_composite.copy()

        # Cursor circle (display coords)
        cx, cy = self.mouse_pos
        radius = max(1, self._disp_brush_size() // 2)
        cv2.circle(display, (cx, cy), radius, (255, 255, 255), 1)

        # Bezier preview (convert full-res points to display coords)
        if self.current_tool == "bezier" and self.bezier_points:
            for pt in self.bezier_points:
                dx, dy = int(pt[0] * self._scale), int(pt[1] * self._scale)
                cv2.circle(display, (dx, dy), 4, (0, 255, 255), -1)
                cv2.circle(display, (dx, dy), 4, (255, 255, 255), 1)

            cursor_full = self._disp_to_full(*self.mouse_pos)
            preview_pts = self.bezier_points + [cursor_full]
            if len(preview_pts) >= 2:
                curve = self._generate_bezier_curve(preview_pts, num_samples=100)
                disp_curve = (curve * self._scale).astype(np.int32)
                cv2.polylines(display, [disp_curve], False, (0, 255, 255), 1)

        # Status bar
        self._draw_status_bar(display)

        if self.show_help:
            self._draw_help_overlay(display)

        cv2.imshow(self.window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

    def _draw_status_bar(self, display: np.ndarray):
        tool_name = self.current_tool.upper()
        soft_str = " SOFT" if self.soft_brush else ""
        text = (
            f"  Tool: {tool_name}{soft_str}  |  Size: {self.brush_size}px  |  "
            f"Undos: {len(self.undo_stack)}  |  H=Help  Enter=Accept  Esc=Cancel"
        )

        bar_h = 30
        if display.shape[0] < bar_h:
            return

        overlay = display[:bar_h, :].astype(np.float32)
        display[:bar_h, :] = (overlay * 0.4).astype(np.uint8)

        cv2.putText(
            display, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def _draw_help_overlay(self, display: np.ndarray):
        help_lines = [
            "=== MASK REFINEMENT ===",
            "",
            "Tools:",
            "  [B] Brush  - paint to ADD",
            "  [E] Eraser - paint to REMOVE",
            "  [P] Pen    - bezier curves",
            "  [D] Soft   - toggle feathered edges",
            "",
            "Pen Tool:",
            "  Left-click  = place point",
            "  Right-click = complete curve",
            "",
            "Brush Size:",
            "  [ / ] = smaller / larger",
            "  1-5   = presets (5,10,20,40,80)",
            "",
            "Actions:",
            "  Z     = Undo",
            "  R     = Reset all edits",
            "  Enter = Accept changes",
            "  Esc/Q = Cancel (discard)",
            "  H     = Toggle this help",
        ]

        box_w, box_h = 280, len(help_lines) * 20 + 20
        y_start = 40
        overlay_region = display[y_start : y_start + box_h, 10 : 10 + box_w]
        if overlay_region.size > 0:
            blended = (overlay_region.astype(np.float32) * 0.3).astype(np.uint8)
            display[y_start : y_start + box_h, 10 : 10 + box_w] = blended

        y = y_start + 18
        for line in help_lines:
            cv2.putText(
                display, line, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
            )
            y += 20

    # ------------------------------------------------------------------
    # Undo / state
    # ------------------------------------------------------------------

    def _push_undo(self):
        self.undo_stack.append(self.drawing_layer.copy())
        if len(self.undo_stack) > self.MAX_UNDO:
            self.undo_stack.pop(0)

    def _pop_undo(self):
        if self.undo_stack:
            self.drawing_layer = self.undo_stack.pop()
            self._composite_dirty = True

    def _get_merged_mask(self) -> np.ndarray:
        """Return the original mask merged with user edits, always (H, W)."""
        return np.clip(self.original_mask + self.drawing_layer, 0.0, 1.0)
