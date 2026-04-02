"""
Visualization utilities for the ReFloorBRUSNIKA pipeline.

Extracted from main.py. Generates summary images and room/OCR overlays
using matplotlib and OpenCV.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRect
import numpy as np


# ============================================================
# DESIGN SYSTEM
# ============================================================

class VisualizationTheme:
    """Colors and styling constants for visualization output."""

    FIG_BG = 'white'
    AXES_EDGE = '#cccccc'
    TITLE_COLOR = '#1a1a1a'
    CAPTION_COLOR = '#666666'

    ROOM_FILLS: List[str] = [
        '#ddeeff', '#ddf5e8', '#fff8dc',
        '#fde8e8', '#ede8f8', '#e0f4f4',
        '#fdeedd', '#e8f5e8', '#fce8f4',
    ]
    ROOM_BORDERS: List[str] = [
        '#2980b9', '#27ae60', '#c8960c',
        '#c0392b', '#7d3c98', '#148f77',
        '#ca6f1e', '#1e8449', '#943126',
    ]

    WALL_OUTER_CV2 = (50, 130, 80)
    WALL_INNER_CV2 = (100, 100, 100)
    WALL_OUTER_SHADOW_CV2 = (20, 60, 35)
    WALL_INNER_SHADOW_CV2 = (55, 55, 55)
    WALL_LABEL_BG_OUTER = (232, 245, 237)
    WALL_LABEL_BG_INNER = (238, 238, 238)
    WINDOW_TINT_CV2 = (185, 225, 225)
    WINDOW_BORDER_CV2 = (0, 140, 140)
    DOOR_BORDER_CV2 = (175, 95, 55)
    GAP_BORDER_CV2 = (170, 155, 45)

    WALL_OUTER_MPL = '#2e7d52'
    WALL_INNER_MPL = '#646464'
    DOOR_MPL = '#af6035'
    WINDOW_MPL = '#007070'
    OPENING_MPL = '#8a7800'
    OCR_NUMERIC_MPL = '#1a5c32'
    OCR_TEXT_MPL = '#555555'

    WATERMARK = 'ReFloor · Brusnika'


# ============================================================
# COORDINATE HELPERS
# ============================================================

def cm_to_px(val_cm: float, pixels_to_cm: float) -> float:
    """Convert centimeters to pixels."""
    return val_cm / pixels_to_cm


def room_points_to_px(room, pixels_to_cm: float) -> List[Tuple[float, float]]:
    """Convert room points from cm to pixels."""
    return [
        (cm_to_px(p.x, pixels_to_cm), cm_to_px(p.y, pixels_to_cm))
        for p in room.points
    ]


def parse_ocr_area_m2(text: str) -> Optional[float]:
    """Parse OCR text to extract area in square meters."""
    cleaned = text.strip().replace(',', '.').replace('²', '2')
    match = re.search(
        r'(\d{1,4}(?:\.\d{1,3})?)\s*(?:м2|m2|кв\.?м|sq\.?m)?',
        cleaned, re.IGNORECASE,
    )
    if match:
        value = float(match.group(1))
        if 1.0 <= value <= 9999.0:
            return value
    return None


def format_wall_length(length_cm: float) -> str:
    """Format wall length as human-readable string."""
    if length_cm >= 100:
        return f'{length_cm / 100:.2f}m'
    return f'{int(length_cm)}cm'


# ============================================================
# MATPLOTLIB HELPERS
# ============================================================

def _axes_clean(ax: plt.Axes, title: str = '') -> None:
    """Clean up matplotlib axes: remove ticks, thin spines, set title."""
    for spine in ax.spines.values():
        spine.set_edgecolor(VisualizationTheme.AXES_EDGE)
        spine.set_linewidth(0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(
            title,
            fontsize=9,
            color=VisualizationTheme.TITLE_COLOR,
            fontfamily='sans-serif',
            fontweight='normal',
            loc='left',
            pad=5,
        )


def _watermark(ax: plt.Axes) -> None:
    """Add watermark text to a matplotlib axes."""
    ax.text(
        0.995, 0.015, VisualizationTheme.WATERMARK,
        transform=ax.transAxes,
        fontsize=4.9, color='#aaaaaa',
        ha='right', va='bottom',
        fontfamily='sans-serif', fontstyle='italic',
    )


# ============================================================
# WALL SEGMENT ACCESSOR
# ============================================================

def _seg_coords(seg) -> Tuple[float, float, float, float]:
    """Extract (x1, y1, x2, y2) from any wall segment type.

    Supports WallSegment (.bbox.x1), WallRectangle (.x1), and stubs.
    """
    if hasattr(seg, 'bbox') and hasattr(seg.bbox, 'x1'):
        return seg.bbox.x1, seg.bbox.y1, seg.bbox.x2, seg.bbox.y2
    return seg.x1, seg.y1, seg.x2, seg.y2


# ============================================================
# CV2 OVERLAY BUILDER
# ============================================================

def build_wall_overlay(
    img: np.ndarray,
    segments: List[Any],  # WallRectangle / _SegmentStub list
    openings_unified: List[Any] = None,  # core.models.Opening list
    yolo_doors: List[Any] = None,
    yolo_windows: List[Any] = None,
    algo_windows: List[Any] = None,
    gap_openings: List[Any] = None,
    door_arcs: List[Any] = None,
    measurements: Optional[Any] = None,
    pixels_to_cm: float = 2.54,
) -> np.ndarray:
    """
    Build a CV2 overlay image with walls, openings, and arcs.

    Args:
        img: Original BGR image
        segments: Wall segments with x1, y1, x2, y2 attributes
        openings_unified: Unified Opening objects (core.models)
        yolo_doors: Legacy YOLO door detections
        yolo_windows: Legacy YOLO window detections
        algo_windows: Algorithmic window DetectedWindow objects
        gap_openings: Legacy gap Opening objects
        door_arcs: Detected door arc objects
        measurements: Optional RoomMeasurements for outer wall classification
        pixels_to_cm: Pixels to cm scale factor

    Returns:
        BGR overlay image
    """
    from core.models import OpeningType as UOpeningType

    overlay = img.copy()
    pixel_scale = measurements.pixel_scale if measurements else pixels_to_cm / 100.0
    outer_ids: set = set()
    if measurements:
        outer_ids = {w['id'] for w in measurements.walls if w.get('is_outer')}

    # Draw wall segments as centerlines
    for i, seg in enumerate(segments or []):
        sx1, sy1, sx2, sy2 = _seg_coords(seg)
        is_outer = i in outer_ids
        # Compute centerline from axis-aligned bbox
        if (sx2 - sx1) >= (sy2 - sy1):  # wider than tall → horizontal
            cy = int((sy1 + sy2) / 2)
            pt1 = (int(sx1), cy)
            pt2 = (int(sx2), cy)
        else:  # taller than wide → vertical
            cx = int((sx1 + sx2) / 2)
            pt1 = (cx, int(sy1))
            pt2 = (cx, int(sy2))
        shadow = VisualizationTheme.WALL_OUTER_SHADOW_CV2 if is_outer else VisualizationTheme.WALL_INNER_SHADOW_CV2
        fg = VisualizationTheme.WALL_OUTER_CV2 if is_outer else VisualizationTheme.WALL_INNER_CV2
        cv2.line(overlay, pt1, pt2, shadow, 6)
        cv2.line(overlay, pt1, pt2, fg, 3)

        length_px = math.hypot(sx2 - sx1, sy2 - sy1)
        if length_px < 20:
            continue

        length_cm = length_px * pixel_scale * 100
        txt = format_wall_length(length_cm)
        mx = int((sx1 + sx2) / 2)
        my = int((sy1 + sy2) / 2)
        is_h = abs(sx2 - sx1) > abs(sy2 - sy1)
        ox, oy = (0, -12) if is_h else (12, 3)

        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        p = 2
        rx1 = max(0, mx + ox - p)
        ry1 = max(0, my + oy - th - p)
        rx2 = min(img.shape[1], mx + ox + tw + p)
        ry2 = min(img.shape[0], my + oy + bl + p)

        sub = overlay[ry1:ry2, rx1:rx2]
        if sub.size:
            bg = (
                VisualizationTheme.WALL_LABEL_BG_OUTER
                if is_outer else VisualizationTheme.WALL_LABEL_BG_INNER
            )
            bg_layer = np.full_like(sub, bg)
            cv2.addWeighted(sub, 0.15, bg_layer, 0.85, 0, sub)

        cv2.putText(
            overlay, txt, (mx + ox, my + oy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (35, 35, 35), 1, cv2.LINE_AA,
        )

    # Draw unified openings (new pipeline path)
    if openings_unified:
        for op in openings_unified:
            x1, y1, x2, y2 = op.bbox.to_int_tuple()
            if op.opening_type == UOpeningType.DOOR:
                color = VisualizationTheme.DOOR_BORDER_CV2
            elif op.opening_type == UOpeningType.WINDOW:
                sub = overlay[y1:y2, x1:x2]
                if sub.size:
                    tint = np.full_like(sub, VisualizationTheme.WINDOW_TINT_CV2)
                    cv2.addWeighted(sub, 0.55, tint, 0.45, 0, sub)
                color = VisualizationTheme.WINDOW_BORDER_CV2
            else:
                color = VisualizationTheme.GAP_BORDER_CV2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    # Legacy paths (for backward compat with old pipeline results)
    for win in (algo_windows or []):
        x, y, ww, hh = win.x, win.y, win.w, win.h
        sub = overlay[y:y + hh, x:x + ww]
        if sub.size:
            tint = np.full_like(sub, VisualizationTheme.WINDOW_TINT_CV2)
            cv2.addWeighted(sub, 0.55, tint, 0.45, 0, sub)
        cv2.rectangle(overlay, (x, y), (x + ww, y + hh),
                      VisualizationTheme.WINDOW_BORDER_CV2, 2)

    for d in (yolo_doors or []):
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      VisualizationTheme.DOOR_BORDER_CV2, 2)

    for w in (yolo_windows or []):
        x1, y1, x2, y2 = map(int, w.bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      VisualizationTheme.WINDOW_BORDER_CV2, 2)

    for op in (gap_openings or []):
        if getattr(op, 'is_valid', True):
            cv2.rectangle(overlay, (op.x1, op.y1), (op.x2, op.y2),
                          VisualizationTheme.GAP_BORDER_CV2, 2)

    # Draw door arcs
    if door_arcs:
        try:
            from detection.doordetector import AlgorithmicDoorArcDetector
            for arc in door_arcs:
                AlgorithmicDoorArcDetector.draw_arc_on_image(
                    overlay, arc,
                    color=VisualizationTheme.DOOR_BORDER_CV2,
                    thickness=3,
                    draw_hinge=True,
                    draw_label=False,
                )
        except Exception:
            pass

    return overlay


# ============================================================
# INDIVIDUAL OVERLAY BUILDERS (for debug grid panels)
# ============================================================

def build_wall_bbox_overlay(
    img: np.ndarray,
    segments: List[Any],
    measurements: Optional[Any] = None,
) -> np.ndarray:
    """Draw wall bounding boxes only — no labels, no openings.

    Green rectangles for outer walls, gray for inner walls.
    """
    overlay = img.copy()
    fill_layer = np.zeros_like(img)
    outer_ids: set = set()
    if measurements:
        outer_ids = {w['id'] for w in measurements.walls if w.get('is_outer')}

    for i, seg in enumerate(segments or []):
        is_outer = i in outer_ids
        color = VisualizationTheme.WALL_OUTER_CV2 if is_outer else VisualizationTheme.WALL_INNER_CV2
        sx1, sy1, sx2, sy2 = _seg_coords(seg)
        x1, y1, x2, y2 = int(sx1), int(sy1), int(sx2), int(sy2)
        cv2.rectangle(fill_layer, (x1, y1), (x2, y2), color, -1)

    alpha = 0.35
    mask = np.any(fill_layer > 0, axis=2)
    overlay[mask] = cv2.addWeighted(img, 1 - alpha, fill_layer, alpha, 0)[mask]

    return overlay


def build_wall_mask_overlay(
    img: np.ndarray,
    wall_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Blend wall_mask as a semi-transparent green overlay on the original image."""
    if wall_mask is None:
        return img.copy()

    overlay = img.copy()
    # Ensure mask is single-channel and same size
    mask = wall_mask
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Create green tint where mask > 0
    tint = np.zeros_like(overlay)
    tint[:, :] = (40, 180, 80)  # BGR green

    wall_pixels = mask > 0
    alpha = 0.45
    overlay[wall_pixels] = cv2.addWeighted(
        overlay[wall_pixels], 1 - alpha,
        tint[wall_pixels], alpha, 0,
    )

    return overlay


def build_doors_overlay(
    img: np.ndarray,
    openings: List[Any],
    door_arcs: List[Any] = None,
) -> np.ndarray:
    """Draw only door detections (boxes + arcs) on the original image."""
    from core.models import OpeningType as UOpeningType

    overlay = img.copy()

    for op in (openings or []):
        if op.opening_type != UOpeningType.DOOR:
            continue
        x1, y1, x2, y2 = op.bbox.to_int_tuple()
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      VisualizationTheme.DOOR_BORDER_CV2, 2)
        # Label with confidence
        label = f'{op.confidence:.0%}'
        if op.source:
            label = f'{op.source} {label}'
        cv2.putText(overlay, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    VisualizationTheme.DOOR_BORDER_CV2, 1, cv2.LINE_AA)

    # Draw door arcs
    if door_arcs:
        try:
            from detection.doordetector import AlgorithmicDoorArcDetector
            for arc in door_arcs:
                AlgorithmicDoorArcDetector.draw_arc_on_image(
                    overlay, arc,
                    color=VisualizationTheme.DOOR_BORDER_CV2,
                    thickness=3,
                    draw_hinge=True,
                    draw_label=False,
                )
        except Exception:
            pass

    return overlay


def build_windows_overlay(
    img: np.ndarray,
    openings: List[Any],
) -> np.ndarray:
    """Draw only window detections (tint + boxes) on the original image."""
    from core.models import OpeningType as UOpeningType

    overlay = img.copy()

    for op in (openings or []):
        if op.opening_type != UOpeningType.WINDOW:
            continue
        x1, y1, x2, y2 = op.bbox.to_int_tuple()
        # Tint window area
        sub = overlay[y1:y2, x1:x2]
        if sub.size:
            tint = np.full_like(sub, VisualizationTheme.WINDOW_TINT_CV2)
            cv2.addWeighted(sub, 0.55, tint, 0.45, 0, sub)
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      VisualizationTheme.WINDOW_BORDER_CV2, 2)
        # Label with confidence
        label = f'{op.confidence:.0%}'
        if op.source:
            label = f'{op.source} {label}'
        cv2.putText(overlay, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    VisualizationTheme.WINDOW_BORDER_CV2, 1, cv2.LINE_AA)

    return overlay


# ============================================================
# SUMMARY RENDERER (2×3 GRID)
# ============================================================

class FloorplanVisualizer:
    """
    Generates visualization outputs for the floorplan pipeline.

    Produces a 2x3 matplotlib summary image and a room/OCR overlay.
    """

    def __init__(self, pixels_to_cm: float = 2.54) -> None:
        self.pixels_to_cm = pixels_to_cm

    def render_summary(
        self,
        img_bgr: np.ndarray,
        wall_segments: List[Any],
        openings: List[Any],
        door_arcs: List[Any],
        wall_mask: Optional[np.ndarray],
        outline_mask: Optional[np.ndarray],
        measurements: Optional[Any],
        output_path: str,
        n_walls: int = 0,
        n_doors: int = 0,
        n_windows: int = 0,
        n_arcs: int = 0,
    ) -> None:
        """
        Render 2x3 debug summary image.

        Layout:
            Original     | Wall BBoxes   | Wall Mask Overlay
            Doors Only   | Windows Only  | All Combined

        Args:
            img_bgr: Original BGR image
            wall_segments: Wall segments for overlay
            openings: Unified Opening objects
            door_arcs: Detected door arc objects
            wall_mask: Wall binary mask (for mask overlay panel)
            outline_mask: Wall outline mask
            measurements: Optional room measurements
            output_path: Output PNG path
            n_walls, n_doors, n_windows, n_arcs: Counts for subtitle
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        dpi = 140

        fig = plt.figure(
            figsize=(24, 14), dpi=dpi,
            facecolor=VisualizationTheme.FIG_BG,
        )
        gs = GridSpec(
            2, 3, figure=fig,
            left=0.015, right=0.985,
            top=0.910, bottom=0.025,
            hspace=0.08, wspace=0.025,
        )

        n_gaps = sum(
            1 for o in (openings or [])
            if hasattr(o, 'opening_type') and o.opening_type.value == 'gap'
        )
        area_str = f'{measurements.area_m2:.1f} m\u00b2' if measurements else '\u2014'

        fig.text(
            0.5, 0.970, 'Floor Plan Analysis',
            fontsize=14, color=VisualizationTheme.TITLE_COLOR,
            fontfamily='sans-serif', fontweight='semibold',
            ha='center', va='top',
        )
        fig.text(
            0.5, 0.945,
            f'{n_walls} walls  \u00b7  {n_doors} doors  \u00b7  {n_windows} windows'
            f'  \u00b7  {n_gaps} gaps  \u00b7  {n_arcs} swing arcs  \u00b7  {area_str}',
            fontsize=9, color=VisualizationTheme.CAPTION_COLOR,
            fontfamily='sans-serif', ha='center', va='top',
        )

        # ── Row 1 ─────────────────────────────────────────────────
        # Panel 1: Original
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(img_rgb, interpolation='bilinear')
        _axes_clean(ax, 'Original')
        _watermark(ax)

        # Panel 2: Wall BBoxes
        ax = fig.add_subplot(gs[0, 1])
        wall_bbox_img = build_wall_bbox_overlay(
            img_bgr, wall_segments, measurements)
        ax.imshow(cv2.cvtColor(wall_bbox_img, cv2.COLOR_BGR2RGB),
                  interpolation='bilinear')
        _axes_clean(ax, f'Wall Bounding Boxes ({n_walls})')
        _watermark(ax)

        # Panel 3: Wall Mask Overlay
        ax = fig.add_subplot(gs[0, 2])
        # Prefer outline_mask if wall_mask is None
        mask_to_show = wall_mask if wall_mask is not None else outline_mask
        mask_overlay_img = build_wall_mask_overlay(img_bgr, mask_to_show)
        ax.imshow(cv2.cvtColor(mask_overlay_img, cv2.COLOR_BGR2RGB),
                  interpolation='bilinear')
        _axes_clean(ax, 'Wall Mask Overlay')
        _watermark(ax)

        # ── Row 2 ─────────────────────────────────────────────────
        # Panel 4: Doors Only
        ax = fig.add_subplot(gs[1, 0])
        doors_img = build_doors_overlay(img_bgr, openings, door_arcs)
        ax.imshow(cv2.cvtColor(doors_img, cv2.COLOR_BGR2RGB),
                  interpolation='bilinear')
        _axes_clean(ax, f'Doors ({n_doors})  +  Arcs ({n_arcs})')
        _watermark(ax)

        # Panel 5: Windows Only
        ax = fig.add_subplot(gs[1, 1])
        windows_img = build_windows_overlay(img_bgr, openings)
        ax.imshow(cv2.cvtColor(windows_img, cv2.COLOR_BGR2RGB),
                  interpolation='bilinear')
        _axes_clean(ax, f'Windows ({n_windows})')
        _watermark(ax)

        # Panel 6: All Combined
        ax = fig.add_subplot(gs[1, 2])
        combined = build_wall_overlay(
            img_bgr,
            segments=wall_segments,
            openings_unified=openings,
            door_arcs=door_arcs,
            measurements=measurements,
            pixels_to_cm=self.pixels_to_cm,
        )
        ax.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB),
                  interpolation='bilinear')
        _axes_clean(ax, 'All Combined')
        _watermark(ax)

        fig.savefig(
            output_path, dpi=dpi, bbox_inches='tight',
            facecolor=VisualizationTheme.FIG_BG,
            edgecolor='none', pad_inches=0.015,
        )
        plt.close(fig)

    def render_room_ocr_overlay(
        self,
        img_bgr: np.ndarray,
        rooms: List[Any],
        openings: List[Any],
        ocr_labels: Optional[List[Tuple[str, int, int, int, int]]],
        output_path: str,
        door_arcs: Optional[List[Any]] = None,
    ) -> None:
        """
        Render room + OCR overlay image.

        Args:
            img_bgr: Original BGR image
            rooms: Room objects with .points attribute
            openings: Opening objects (core.models or legacy)
            ocr_labels: List of (text, x1, y1, x2, y2) OCR results
            output_path: Output PNG path
            door_arcs: Optional door arc objects for arc drawing
        """
        from core.models import OpeningType as UOpeningType

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]
        dpi = 150

        fig, ax = plt.subplots(
            1, 1,
            figsize=(w / dpi + 0.3, h / dpi + 0.5),
            dpi=dpi, facecolor=VisualizationTheme.FIG_BG,
        )
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)

        fig.text(
            0.01, 0.98, 'Rooms & OCR labels',
            fontsize=8, color=VisualizationTheme.TITLE_COLOR,
            fontfamily='sans-serif', fontweight='semibold', va='top',
        )

        ax.imshow(img_rgb, interpolation='bilinear')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        _axes_clean(ax)
        _watermark(ax)

        # Draw rooms
        for i, room in enumerate(rooms or []):
            fill = VisualizationTheme.ROOM_FILLS[i % len(VisualizationTheme.ROOM_FILLS)]
            border = VisualizationTheme.ROOM_BORDERS[i % len(VisualizationTheme.ROOM_BORDERS)]
            pts_px = room_points_to_px(room, self.pixels_to_cm)
            if len(pts_px) < 3:
                continue

            poly = MplPolygon(
                pts_px, closed=True,
                facecolor=fill, alpha=0.28,
                edgecolor=border, linewidth=1.6,
                zorder=2,
            )
            ax.add_patch(poly)

            cx = sum(p[0] for p in pts_px) / len(pts_px)
            cy = sum(p[1] for p in pts_px) / len(pts_px)
            ax.text(
                cx, cy, room.name,
                fontsize=6, color=border,
                fontfamily='sans-serif', fontweight='semibold',
                ha='center', va='center',
                bbox=dict(
                    boxstyle='round,pad=0.22',
                    facecolor='white', alpha=0.80,
                    edgecolor=border, linewidth=0.6,
                ),
                zorder=5,
            )

        # Draw openings
        for opening in (openings or []):
            # Support both unified Opening and legacy floorplanexporter.Opening
            if hasattr(opening, 'bbox'):
                ox = cm_to_px(opening.bbox.center_x, self.pixels_to_cm)
                oy = cm_to_px(opening.bbox.center_y, self.pixels_to_cm)
                otype = opening.opening_type
                is_door = otype == UOpeningType.DOOR
                is_window = otype == UOpeningType.WINDOW
            elif hasattr(opening, 'center'):
                ox = cm_to_px(opening.center.x, self.pixels_to_cm)
                oy = cm_to_px(opening.center.y, self.pixels_to_cm)
                from floorplanexporter import OpeningType as LegacyOT
                is_door = opening.opening_type == LegacyOT.DOOR
                is_window = opening.opening_type == LegacyOT.WINDOW
            else:
                continue

            if is_door:
                color, marker = VisualizationTheme.DOOR_MPL, 's'
            elif is_window:
                color, marker = VisualizationTheme.WINDOW_MPL, 'D'
            else:
                color, marker = VisualizationTheme.OPENING_MPL, 'o'
            ax.plot(ox, oy, marker, color=color, markersize=4.5,
                    markeredgecolor='white', markeredgewidth=0.5,
                    alpha=0.85, zorder=4)

        # Draw OCR labels
        if ocr_labels:
            for text, x1, y1, x2, y2 in ocr_labels:
                is_num = parse_ocr_area_m2(text) is not None
                ec = VisualizationTheme.OCR_NUMERIC_MPL if is_num else VisualizationTheme.OCR_TEXT_MPL
                ax.add_patch(MplRect(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=0.6 if is_num else 0.35,
                    edgecolor=ec, facecolor='none',
                    linestyle='-' if is_num else ':',
                    zorder=3,
                ))
                ax.text(
                    x1 + 1, y1 - 2, text,
                    fontsize=4.5, color=ec, va='bottom',
                    fontweight='semibold' if is_num else 'normal',
                    fontfamily='sans-serif', zorder=4,
                )

        # Draw door arcs
        if door_arcs:
            from matplotlib.patches import Arc as MplArc
            for arc in door_arcs:
                cx_a, cy_a = arc.center
                r = arc.radius_px
                arc_patch = MplArc(
                    (cx_a, cy_a), r * 2, r * 2,
                    angle=0,
                    theta1=arc.start_angle_deg,
                    theta2=arc.end_angle_deg,
                    color=VisualizationTheme.DOOR_MPL,
                    linewidth=1.9,
                    linestyle='--',
                    zorder=6,
                )
                ax.add_patch(arc_patch)
                hx, hy = arc.hinge
                ax.plot(hx, hy, 'd', color=VisualizationTheme.DOOR_MPL,
                        markersize=5.5, markeredgecolor='white',
                        markeredgewidth=0.6, zorder=7)

        fig.savefig(
            output_path, dpi=dpi, bbox_inches='tight',
            facecolor=VisualizationTheme.FIG_BG,
            edgecolor='none', pad_inches=0.015,
        )
        plt.close(fig)
