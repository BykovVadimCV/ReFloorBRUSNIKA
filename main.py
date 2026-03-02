"""
================================================================================
СИСТЕМА АВТОМАТИЧЕСКОГО АНАЛИЗА ПЛАНИРОВОК КВАРТИР «БРУСНИКА»
Модуль: Главный оркестратор и визуализация (main.py)
Версия: 5.3 — Clean Visuals Edition + Door Arc Detection
Автор: BykovVadimCV
Дата:  23 февраля 2026 г.

Описание пайплайна:
  1. Загрузка изображения планировки формата «Брусника»
  2. Детекция стен по цвету #8E8780 (VersatileWallDetector)
  3. Детекция дверей и окон (YOLODoorWindowDetector + AlgorithmicWindowDetector)
  4. Геометрический поиск проёмов (GeometricOpeningDetector)
  4.5 Детекция дуг открывания дверей (AlgorithmicDoorArcDetector)  ← NEW
  5. OCR + калибровка масштаба по площадям помещений (RoomSizeAnalyzer)
  6. Измерение длин стен по комнатам — пиксели и метры
  7. Экспорт в SweetHome3D (.sh3d) и OBJ
  8. Визуализации (matplotlib, белый фон, минималистичный стиль)

Цветовые коды формата «Брусника»:
  Стены         #8E8780  (общий с круглыми метками — фильтруются морфологией)
  Мебель        #E2DFDD  (R > 0x8E — исключается из расчётов)
  Окна          #E2DFDD  (два параллельных отрезка + белый зазор ½ толщины)
  Двери         #E2DFDD  (линия/прямоугольник + дуга открывания)
  Метки площади чёрные цифры внутри помещений

Принципы:
  - Single Responsibility: каждый класс решает одну задачу
  - Все цвета, шрифты и константы — только в классе Theme
  - Сбой одного этапа не прерывает обработку изображения
================================================================================
"""

import cv2
import numpy as np
import argparse
import os
import re
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import warnings

# ====================== WARNING SUPPRESSION ======================
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*"
)
# ================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRect

from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TaskProgressColumn, TimeElapsedColumn,
)
from rich.text import Text

from detection.walldetector import VersatileWallDetector, WallRectangle
from detection.furnituredetector import YOLODoorWindowDetector, YOLOResults
from detection.windowdetector import AlgorithmicWindowDetector
from detection.space import RoomSizeAnalyzer, RoomMeasurements
from detection.gap import GeometricOpeningDetector
from floorplanexporter import (
    SweetHome3DExporter, Point, Wall, Room, Opening, OpeningType,
    DebugVisualizer, SCALE_FACTOR,
)

# === DOOR ARC INTEGRATION — Import ===
from detection.doordetector import AlgorithmicDoorArcDetector

logger = logging.getLogger("main")
_console = Console()


# ============================================================
# Дизайн-система — цвета, шрифты, константы
# ============================================================

class Theme:
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

    WATERMARK = 'ReFloor · Brusnika v5.3'
    PIPELINE_STAGES: List[str] = [
        'loading',
        'detecting walls',
        'detecting openings',
        'detecting door arcs',       # === DOOR ARC INTEGRATION — new stage ===
        'measuring rooms',
        'exporting',
        'rendering',
    ]


# ============================================================
# Вспомогательные функции координат
# ============================================================

def cm_to_px(val_cm: float, pixels_to_cm: float) -> float:
    return val_cm / pixels_to_cm


def room_points_to_px(
    room: Room, pixels_to_cm: float
) -> List[Tuple[float, float]]:
    return [
        (cm_to_px(p.x, pixels_to_cm), cm_to_px(p.y, pixels_to_cm))
        for p in room.points
    ]


def polygon_area_from_tuples(pts: List[Tuple[float, float]]) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def point_in_polygon(
    px: float, py: float, pts: List[Tuple[float, float]]
) -> bool:
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def parse_ocr_area_m2(text: str) -> Optional[float]:
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
    if length_cm >= 100:
        return f'{length_cm / 100:.2f}m'
    return f'{int(length_cm)}cm'


def _iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _are_parallel_and_close(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    gap_px: float = 6.0,
) -> bool:
    """
    True when two axis-aligned boxes share the same orientation and their
    parallel edges are within gap_px of each other with >50% length overlap.
    Catches doubled detections of the same wall gap that IoU alone misses
    because the boxes sit side-by-side rather than on top of each other.
    """
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    a_horiz = aw >= ah
    b_horiz = bw >= bh
    if a_horiz != b_horiz:
        return False
    if a_horiz:
        v_gap = max(a[1], b[1]) - min(a[3], b[3])
        h_overlap = min(a[2], b[2]) - max(a[0], b[0])
        return v_gap < gap_px and h_overlap > max(aw, bw) * 0.5
    else:
        h_gap = max(a[0], b[0]) - min(a[2], b[2])
        v_overlap = min(a[3], b[3]) - max(a[1], b[1])
        return h_gap < gap_px and v_overlap > max(ah, bh) * 0.5


def deduplicate_boxes(
    yolo_results: Optional[object],
    color_results: Dict,
    iou_thresh: float = 0.30,
    parallel_gap_px: float = 6.0,
) -> None:
    """
    Collect every door/window bbox from every source, sort largest-first,
    then drop anything that either overlaps (IoU >= iou_thresh) or is a
    parallel near-duplicate of an already-kept box.  Writes back in-place.
    Largest box wins — no source has priority.
    """
    tagged: List[Tuple] = []

    if yolo_results:
        for d in (yolo_results.doors or []):
            b = tuple(float(v) for v in d.bbox)
            tagged.append(((b[2]-b[0])*(b[3]-b[1]), b, 'yd', d))
        for w in (yolo_results.windows or []):
            b = tuple(float(v) for v in w.bbox)
            tagged.append(((b[2]-b[0])*(b[3]-b[1]), b, 'yw', w))

    for aw in color_results.get('algo_windows', []):
        b = (float(aw.x), float(aw.y),
             float(aw.x + aw.w), float(aw.y + aw.h))
        tagged.append((float(aw.w * aw.h), b, 'aw', aw))

    for op in color_results.get('openings', []):
        if getattr(op, 'is_valid', True):
            b = (float(op.x1), float(op.y1),
                 float(op.x2), float(op.y2))
            tagged.append(((b[2]-b[0])*(b[3]-b[1]), b, 'op', op))

    if not tagged:
        return

    tagged.sort(key=lambda t: t[0], reverse=True)

    kept_boxes: list = []
    survivors: set = set()
    for i, (_, box, _, _) in enumerate(tagged):
        if not any(
            _iou_xyxy(box, kb) >= iou_thresh
            or _are_parallel_and_close(box, kb, parallel_gap_px)
            for kb in kept_boxes
        ):
            kept_boxes.append(box)
            survivors.add(i)

    if yolo_results:
        yolo_results.doors   = [t[3] for i, t in enumerate(tagged) if i in survivors and t[2] == 'yd']
        yolo_results.windows = [t[3] for i, t in enumerate(tagged) if i in survivors and t[2] == 'yw']
    color_results['algo_windows'] = [t[3] for i, t in enumerate(tagged) if i in survivors and t[2] == 'aw']
    color_results['openings']     = [t[3] for i, t in enumerate(tagged) if i in survivors and t[2] == 'op']


# ============================================================
# Утилиты оформления осей matplotlib
# ============================================================

def _axes_clean(ax: plt.Axes, title: str = '') -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(Theme.AXES_EDGE)
        spine.set_linewidth(0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(
            title,
            fontsize=9,
            color=Theme.TITLE_COLOR,
            fontfamily='sans-serif',
            fontweight='normal',
            loc='left',
            pad=5,
        )


def _watermark(ax: plt.Axes) -> None:
    ax.text(
        0.995, 0.015, Theme.WATERMARK,
        transform=ax.transAxes,
        fontsize=4.9, color='#aaaaaa',
        ha='right', va='bottom',
        fontfamily='sans-serif', fontstyle='italic',
    )


# ============================================================
# _build_wall_overlay — CV2-разметка поверх оригинала
# ============================================================

def _build_wall_overlay(
    img: np.ndarray,
    yolo_results: Optional[YOLOResults],
    color_results: Dict,
    measurements: Optional[RoomMeasurements],
    pixels_to_cm: float,
) -> np.ndarray:
    overlay = img.copy()
    segments = color_results.get('segments', [])
    pixel_scale = (
        measurements.pixel_scale if measurements else pixels_to_cm / 100.0
    )
    outer_ids: set = set()
    if measurements:
        outer_ids = {w['id'] for w in measurements.walls if w.get('is_outer')}

    for i, seg in enumerate(segments):
        is_outer = i in outer_ids
        pt1 = (int(seg.x1), int(seg.y1))
        pt2 = (int(seg.x2), int(seg.y2))
        shadow = Theme.WALL_OUTER_SHADOW_CV2 if is_outer else Theme.WALL_INNER_SHADOW_CV2
        fg = Theme.WALL_OUTER_CV2 if is_outer else Theme.WALL_INNER_CV2
        cv2.line(overlay, pt1, pt2, shadow, 6)
        cv2.line(overlay, pt1, pt2, fg, 3)

        length_px = math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)
        if length_px < 20:
            continue

        length_cm = length_px * pixel_scale * 100
        txt = format_wall_length(length_cm)
        mx = int((seg.x1 + seg.x2) / 2)
        my = int((seg.y1 + seg.y2) / 2)
        is_h = abs(seg.x2 - seg.x1) > abs(seg.y2 - seg.y1)
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
                Theme.WALL_LABEL_BG_OUTER
                if is_outer else Theme.WALL_LABEL_BG_INNER
            )
            bg_layer = np.full_like(sub, bg)
            cv2.addWeighted(sub, 0.15, bg_layer, 0.85, 0, sub)

        cv2.putText(
            overlay, txt, (mx + ox, my + oy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (35, 35, 35), 1, cv2.LINE_AA,
        )

    for win in color_results.get('algo_windows', []):
        x, y, ww, hh = win.x, win.y, win.w, win.h
        sub = overlay[y:y + hh, x:x + ww]
        if sub.size:
            tint = np.full_like(sub, Theme.WINDOW_TINT_CV2)
            cv2.addWeighted(sub, 0.55, tint, 0.45, 0, sub)
        cv2.rectangle(overlay, (x, y), (x + ww, y + hh),
                      Theme.WINDOW_BORDER_CV2, 2)

    if yolo_results and yolo_results.doors:
        for d in yolo_results.doors:
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2),
                          Theme.DOOR_BORDER_CV2, 2)

    if yolo_results and yolo_results.windows:
        for w in yolo_results.windows:
            x1, y1, x2, y2 = map(int, w.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2),
                          Theme.WINDOW_BORDER_CV2, 2)

    for op in color_results.get('openings', []):
        if getattr(op, 'is_valid', True):
            cv2.rectangle(overlay, (op.x1, op.y1), (op.x2, op.y2),
                          Theme.GAP_BORDER_CV2, 2)

    # === DOOR ARC INTEGRATION — Step 2.4: Draw detected swing arcs ===
    for arc in color_results.get('door_arcs', []):
        AlgorithmicDoorArcDetector.draw_arc_on_image(
            overlay, arc,
            color=Theme.DOOR_BORDER_CV2,
            thickness=3,
            draw_hinge=True,
            draw_label=False,
        )

    return overlay


# ============================================================
# render_summary — 2×2 белая сетка (оригинал / оверлей / маска / комнаты)
# ============================================================

def render_summary(
        img_bgr: np.ndarray,
        yolo_results: Optional[YOLOResults],
        color_results: Dict,
        measurements: Optional[RoomMeasurements],
        pixels_to_cm: float,
        output_path: str,
) -> None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    dpi = 160

    fig = plt.figure(figsize=(17.8, 13.4), dpi=dpi, facecolor=Theme.FIG_BG)

    gs = GridSpec(2, 2, figure=fig,
                  left=0.025, right=0.975,
                  top=0.905, bottom=0.055,
                  hspace=0.10, wspace=0.030)

    n_walls = len(color_results.get('segments', []))
    n_doors = len(yolo_results.doors) if yolo_results else 0
    n_windows = len(color_results.get('algo_windows', []))
    n_arcs = len(color_results.get('door_arcs', []))     # === DOOR ARC INTEGRATION ===
    area_str = f'{measurements.area_m2:.1f} m²' if measurements else '—'

    fig.text(
        0.5, 0.965, 'Floor Plan Analysis',
        fontsize=13.5, color=Theme.TITLE_COLOR,
        fontfamily='sans-serif', fontweight='semibold',
        ha='center', va='top',
    )
    # === DOOR ARC INTEGRATION — show arc count in subtitle ===
    fig.text(
        0.5, 0.938,
        f'{n_walls} walls  ·  {n_doors} doors  ·  {n_windows} windows'
        f'  ·  {n_arcs} swing arcs  ·  {area_str}',
        fontsize=8.8, color=Theme.CAPTION_COLOR,
        fontfamily='sans-serif', ha='center', va='top',
    )

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_mask = fig.add_subplot(gs[1, 0])
    ax_rooms = fig.add_subplot(gs[1, 1])

    ax_orig.imshow(img_rgb, interpolation='bilinear')
    _axes_clean(ax_orig, 'Original')
    _watermark(ax_orig)

    overlay = _build_wall_overlay(
        img_bgr, yolo_results, color_results, measurements, pixels_to_cm,
    )
    ax_overlay.imshow(
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), interpolation='bilinear'
    )
    _axes_clean(ax_overlay, 'Walls & Openings')
    _watermark(ax_overlay)

    mask = color_results.get(
        'outline_mask', np.zeros(img_bgr.shape[:2], np.uint8)
    )
    ax_mask.imshow(mask, cmap='binary', interpolation='nearest')
    _axes_clean(ax_mask, 'Wall Structure Mask')
    _watermark(ax_mask)

    ax_rooms.imshow(img_rgb, interpolation='bilinear', alpha=0.36)
    ax_rooms.set_xlim(0, w)
    ax_rooms.set_ylim(h, 0)
    _axes_clean(ax_rooms, 'Detected Rooms')
    _watermark(ax_rooms)

    fig.savefig(
        output_path, dpi=dpi, bbox_inches='tight',
        facecolor=Theme.FIG_BG, edgecolor='none', pad_inches=0.015,
    )
    plt.close(fig)


# ============================================================
# render_room_ocr_overlay — комнаты + OCR (отдельный файл)
# ============================================================

def render_room_ocr_overlay(
    img_bgr: np.ndarray,
    rooms: List[Room],
    openings: List[Opening],
    ocr_labels: Optional[List[Tuple[str, int, int, int, int]]],
    pixels_to_cm: float,
    output_path: str,
    door_arcs: Optional[List] = None,    # === DOOR ARC INTEGRATION — optional param ===
) -> None:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    dpi = 150

    fig, ax = plt.subplots(
        1, 1,
        figsize=(w / dpi + 0.3, h / dpi + 0.5),
        dpi=dpi, facecolor=Theme.FIG_BG,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)

    fig.text(
        0.01, 0.98, 'Rooms & OCR labels',
        fontsize=8, color=Theme.TITLE_COLOR,
        fontfamily='sans-serif', fontweight='semibold', va='top',
    )

    ax.imshow(img_rgb, interpolation='bilinear')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    _axes_clean(ax)
    _watermark(ax)

    for i, room in enumerate(rooms):
        fill = Theme.ROOM_FILLS[i % len(Theme.ROOM_FILLS)]
        border = Theme.ROOM_BORDERS[i % len(Theme.ROOM_BORDERS)]
        pts_px = room_points_to_px(room, pixels_to_cm)
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

    for opening in openings:
        ox = cm_to_px(opening.center.x, pixels_to_cm)
        oy = cm_to_px(opening.center.y, pixels_to_cm)
        if opening.opening_type == OpeningType.DOOR:
            color, marker = Theme.DOOR_MPL, 's'
        elif opening.opening_type == OpeningType.WINDOW:
            color, marker = Theme.WINDOW_MPL, 'D'
        else:
            color, marker = Theme.OPENING_MPL, 'o'
        ax.plot(ox, oy, marker, color=color, markersize=4.5,
                markeredgecolor='white', markeredgewidth=0.5,
                alpha=0.85, zorder=4)

    if ocr_labels:
        for text, x1, y1, x2, y2 in ocr_labels:
            is_num = parse_ocr_area_m2(text) is not None
            ec = Theme.OCR_NUMERIC_MPL if is_num else Theme.OCR_TEXT_MPL
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

    # === DOOR ARC INTEGRATION — Draw arcs on room/OCR overlay ===
    if door_arcs:
        # Convert BGR overlay drawing to matplotlib: draw arcs as
        # matplotlib Arc patches for clean vector rendering
        from matplotlib.patches import Arc as MplArc
        for arc in door_arcs:
            cx, cy = arc.center
            r = arc.radius_px
            # matplotlib Arc uses diameter, angles in degrees
            # Note: matplotlib y-axis is inverted in imshow, but we set
            # ylim(h, 0) so angles work correctly with sign flip
            arc_patch = MplArc(
                (cx, cy), r * 2, r * 2,
                angle=0,
                theta1=arc.start_angle_deg,
                theta2=arc.end_angle_deg,
                color=Theme.DOOR_MPL,
                linewidth=1.9,
                linestyle='--',
                zorder=6,
            )
            ax.add_patch(arc_patch)
            # Hinge marker
            hx, hy = arc.hinge
            ax.plot(hx, hy, 'd', color=Theme.DOOR_MPL, markersize=5.5,
                    markeredgecolor='white', markeredgewidth=0.4,
                    zorder=7)

    fig.savefig(
        output_path, dpi=dpi, bbox_inches='tight',
        facecolor=Theme.FIG_BG, edgecolor='none',
    )
    plt.close(fig)


# ============================================================
# calibrate_scale_factor
# ============================================================

def calibrate_scale_factor(
    rooms: List[Room],
    ocr_labels: List[Tuple[str, int, int, int, int]],
    pixels_to_cm: float,
    current_scale_factor: float,
) -> Optional[float]:
    if not rooms or not ocr_labels:
        return None
    parsed: List[Tuple[float, float, float]] = []
    for text, x1, y1, x2, y2 in ocr_labels:
        area = parse_ocr_area_m2(text)
        if area is not None:
            parsed.append((area, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    if not parsed:
        return None
    m_per_px = pixels_to_cm / 100.0
    ratios: List[float] = []
    for room in rooms:
        pts_px = room_points_to_px(room, pixels_to_cm)
        if len(pts_px) < 3:
            continue
        matched = [
            a for a, cx, cy in parsed
            if point_in_polygon(cx, cy, pts_px)
        ]
        if len(matched) != 1:
            continue
        area_px2 = polygon_area_from_tuples(pts_px)
        if area_px2 < 1.0:
            continue
        ratios.append(math.sqrt(area_px2 * (m_per_px ** 2) / matched[0]))
    if not ratios:
        return None
    return float(np.median(ratios))


# ============================================================
# _SegmentStub
# ============================================================

class _SegmentStub:
    def __init__(
        self,
        x1: int, y1: int, x2: int, y2: int,
        thickness_mean: float = 0.0,
    ) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.thickness_mean = thickness_mean


# ============================================================
# HybridWallAnalyzer
# ============================================================

class HybridWallAnalyzer:

    def __init__(
        self,
        yolo_model_path: str,
        wall_color_hex: str = '8E8780',
        color_tolerance: int = 3,
        yolo_conf_threshold: float = 0.5,
        device: str = 'cpu',
        auto_wall_color: bool = False,
        enable_measurements: bool = True,
        edge_distance_ratio: float = 0.15,
        debug_ocr: bool = False,
        min_window_len: int = 15,
        edge_expansion_tolerance: int = 8,
        max_edge_expansion_px: int = 10,
        wall_height_cm: float = 243.84,
        min_wall_thickness_px: float = 5.0,
        pixels_to_cm: float = 2.54,
        enable_obj_export: bool = True,
        obj_include_floor: bool = True,
        obj_include_ceiling: bool = False,
        scale_factor: float = SCALE_FACTOR,
        wall_detection_mode: str = 'hybrid',
        structure_method: str = 'lines',
        output_dir: str = 'output_v5',           # === DOOR ARC INTEGRATION — needed for debug path ===
    ) -> None:
        self.ocr_model = self._init_doctr()
        self.pixels_to_cm = pixels_to_cm
        self.scale_factor = scale_factor

        self.yolo_detector = YOLODoorWindowDetector(
            model_path=yolo_model_path,
            conf_threshold=yolo_conf_threshold,
            device=device,
        )
        self.color_detector = VersatileWallDetector(
            wall_color_hex=wall_color_hex,
            color_tolerance=color_tolerance,
            auto_wall_color=auto_wall_color,
            min_rect_area_ratio=0.0002,
            rect_visual_gap_px=3,
            ocr_model=self.ocr_model,
            edge_expansion_tolerance=edge_expansion_tolerance,
            max_edge_expansion_px=max_edge_expansion_px,
            wall_detection_mode=wall_detection_mode,
            structure_method=structure_method,
        )
        self.algo_window_detector = AlgorithmicWindowDetector(
            min_window_len=min_window_len,
            ocr_model=self.ocr_model,
        )
        self.gap_detector = GeometricOpeningDetector(
            max_extension_length=300,
            small_rect_aspect_threshold=2.0,
            small_rect_min_side=3,
            merge_distance=8,
            yolo_iou_threshold=0.0,
        )
        # Relaxed detector used during YOLO-targeted re-detection.
        # yolo_iou_threshold=0.0 prevents the detector from suppressing gaps
        # that coincide with a YOLO box — which is exactly what we need here,
        # since we're searching *inside* those boxes.
        # merge_distance is wider to bridge scan-line breaks caused by
        # painting the door leaf white.
        self.gap_detector_targeted = GeometricOpeningDetector(
            max_extension_length=300,
            small_rect_aspect_threshold=2.5,
            small_rect_min_side=3,
            merge_distance=8,
            yolo_iou_threshold=0.0,
        )

        # === DOOR ARC INTEGRATION — Step 2.1: Instantiate arc detector ===
        self.arc_detector = AlgorithmicDoorArcDetector(
            min_arc_radius_m=0.6,
            max_arc_radius_m=1.2,
            attachment_proximity_px=12,
            enable_furniture_filter=True,
            debug=False,                                              # set True during first tests
            debug_output_dir=os.path.join(output_dir, 'debug'),       # creates subfolder if needed
        )

        self.enable_measurements = enable_measurements
        self.size_analyzer: Optional[RoomSizeAnalyzer] = (
            RoomSizeAnalyzer(
                edge_distance_ratio=edge_distance_ratio,
                debug_ocr=debug_ocr,
            )
            if enable_measurements else None
        )
        self.sh3d_exporter = SweetHome3DExporter(
            min_wall_thickness=min_wall_thickness_px,
            pixels_to_cm=pixels_to_cm,
            wall_height_cm=wall_height_cm,
            scale_factor=scale_factor,
        )

    @staticmethod
    def _init_doctr():
        try:
            from doctr.models import ocr_predictor
            return ocr_predictor(pretrained=True)
        except Exception:
            return None

    def _rectangles_to_segments(
        self, rectangles: List[WallRectangle],
    ) -> List[_SegmentStub]:
        if not rectangles:
            return []
        h_rects = [
            r for r in rectangles
            if r.width >= r.height and r.width > 0 and r.height > 0
        ]
        v_rects = [
            r for r in rectangles
            if r.width < r.height and r.width > 0 and r.height > 0
        ]
        segs: List[_SegmentStub] = []
        if h_rects:
            segs.extend(self._merge_aligned(h_rects, True))
        if v_rects:
            segs.extend(self._merge_aligned(v_rects, False))
        return segs

    def _merge_aligned(
        self, rects: List[WallRectangle], is_h: bool,
    ) -> List[_SegmentStub]:
        tol, gap_tol = 5, 3
        groups: List[List[WallRectangle]] = []
        for r in rects:
            placed = False
            for g in groups:
                align_val = r.y2 if is_h else r.x2
                ref_vals = [rr.y2 if is_h else rr.x2 for rr in g]
                if any(abs(align_val - rv) <= tol for rv in ref_vals):
                    g.append(r)
                    placed = True
                    break
            if not placed:
                groups.append([r])
        segs: List[_SegmentStub] = []
        for g in groups:
            g.sort(key=lambda r: r.x1 if is_h else r.y1)
            cur: List[WallRectangle] = []
            for r in g:
                if not cur:
                    cur.append(r)
                    continue
                gap = (r.x1 - cur[-1].x2) if is_h else (r.y1 - cur[-1].y2)
                if gap <= gap_tol:
                    cur.append(r)
                else:
                    segs.append(self._build_seg(cur, is_h))
                    cur = [r]
            if cur:
                segs.append(self._build_seg(cur, is_h))
        return segs

    @staticmethod
    def _build_seg(
        rects: List[WallRectangle], is_h: bool,
    ) -> _SegmentStub:
        x1 = min(r.x1 for r in rects)
        x2 = max(r.x2 for r in rects)
        y1 = min(r.y1 for r in rects)
        y2 = max(r.y2 for r in rects)
        avg_t = sum(r.height if is_h else r.width for r in rects) / len(rects)
        if is_h:
            return _SegmentStub(int(x1), int(y2), int(x2), int(y2), float(avg_t))
        return _SegmentStub(int(x2), int(y1), int(x2), int(y2), float(avg_t))

    def _segments_to_dicts(
        self, segments: List[_SegmentStub],
    ) -> List[Dict]:
        return [
            {
                'id': i,
                'x1': int(s.x1), 'y1': int(s.y1),
                'x2': int(s.x2), 'y2': int(s.y2),
                'length_pixels': float(math.hypot(s.x2 - s.x1, s.y2 - s.y1)),
                'thickness': float(s.thickness_mean),
            }
            for i, s in enumerate(segments)
        ]

    def _rects_for_sh3d(
        self, rectangles: List[WallRectangle],
    ) -> List[Dict]:
        return self.color_detector.rectangles_to_dict_list(rectangles)

    @staticmethod
    def _doors_for_sh3d(
        yolo_results: Optional[YOLOResults],
    ) -> Optional[List[Dict]]:
        if not yolo_results or not yolo_results.doors:
            return None
        return [
            {'x1': float(d.bbox[0]), 'y1': float(d.bbox[1]),
             'x2': float(d.bbox[2]), 'y2': float(d.bbox[3])}
            for d in yolo_results.doors
        ]

    @staticmethod
    def _algo_windows_for_sh3d(algo_wins) -> List[Dict]:
        return [
            {'x': int(w.x), 'y': int(w.y),
             'width': int(w.w), 'height': int(w.h)}
            for w in algo_wins
        ]

    @staticmethod
    def _yolo_windows_for_sh3d(
        yolo_results: Optional[YOLOResults],
    ) -> List[Dict]:
        if not yolo_results or not yolo_results.windows:
            return []
        return [
            {'x1': float(w.bbox[0]), 'y1': float(w.bbox[1]),
             'x2': float(w.bbox[2]), 'y2': float(w.bbox[3])}
            for w in yolo_results.windows
        ]

    def _all_windows_for_sh3d(
        self,
        yolo_results: Optional[YOLOResults],
        algo_wins,
    ) -> Optional[List[Dict]]:
        out = self._algo_windows_for_sh3d(algo_wins)
        out.extend(self._yolo_windows_for_sh3d(yolo_results))
        return out or None

    def _gap_doors_for_sh3d(self, openings) -> List[Dict]:
        return [
            {'x1': int(o.x1), 'y1': int(o.y1),
             'x2': int(o.x2), 'y2': int(o.y2)}
            for o in openings if getattr(o, 'is_valid', True)
        ]

    # ------------------------------------------------------------------
    # YOLO-door → gap-door resolution
    def _resolve_yolo_doors_to_gaps(
        self,
        img: np.ndarray,
        color_results: Dict,
        yolo_results: Optional[YOLOResults],
        pixel_scale: Optional[float],
        match_margin_px: int = 25,
    ) -> List[Dict]:
        """
        Replace raw YOLO door bboxes with precisely-located wall-gap bboxes.

        Algorithm
        ---------
        Phase 1 — match from existing gap openings.
            For each YOLO door, check whether any gap opening already found by
            GeometricOpeningDetector has its centre inside the YOLO bbox
            expanded by ``match_margin_px``.  Greedy: closest centre wins;
            each gap may be claimed by at most one YOLO door.

        Phase 2 — paint + re-detect for any still-unmatched YOLO doors.
            Paint each unmatched YOLO bbox white on a copy of the image so the
            wall detector sees a solid opening there.  Re-run wall detection
            and then the relaxed ``gap_detector_targeted`` restricted to those
            areas.  Attempt matching again with the same spatial rule.

        Phase 3 — fallback.
            Any YOLO door still unmatched keeps its raw bbox so no detection is
            silently dropped.

        Standalone gap openings not claimed by any YOLO door (passageways,
        archways, etc.) are preserved unchanged.

        Returns
        -------
        A flat list of door rect dicts ready for ``OpeningProcessor``.
        Every rect has keys ``x1, y1, x2, y2``.
        """
        existing_gaps = [
            o for o in color_results.get('openings', [])
            if getattr(o, 'is_valid', True)
        ]
        yolo_doors = yolo_results.doors if yolo_results else []

        # Indices of gaps that have been claimed by a YOLO door
        claimed_gap_indices: Set[int] = set()
        # yolo_door_index → resolved rect dict
        matched: Dict[int, Dict] = {}

        def _expanded(door, margin: int) -> Tuple[float, float, float, float]:
            x1, y1, x2, y2 = door.bbox
            return x1 - margin, y1 - margin, x2 + margin, y2 + margin

        def _gap_center(gap) -> Tuple[float, float]:
            return (gap.x1 + gap.x2) / 2.0, (gap.y1 + gap.y2) / 2.0

        def _center_inside(cx, cy, ex1, ey1, ex2, ey2) -> bool:
            return ex1 <= cx <= ex2 and ey1 <= cy <= ey2

        def _gap_to_rect(gap) -> Dict:
            return {
                'x1': int(gap.x1), 'y1': int(gap.y1),
                'x2': int(gap.x2), 'y2': int(gap.y2),
            }

        # ── Phase 1: match from existing gap openings ────────────────────
        for yi, ydoor in enumerate(yolo_doors):
            bx1, by1, bx2, by2 = ydoor.bbox
            ycx = (bx1 + bx2) / 2.0
            ycy = (by1 + by2) / 2.0
            ex1, ey1, ex2, ey2 = _expanded(ydoor, match_margin_px)

            candidates = []
            for gi, gap in enumerate(existing_gaps):
                if gi in claimed_gap_indices:
                    continue
                gcx, gcy = _gap_center(gap)
                if _center_inside(gcx, gcy, ex1, ey1, ex2, ey2):
                    dist = math.hypot(gcx - ycx, gcy - ycy)
                    candidates.append((dist, gi, gap))

            if candidates:
                candidates.sort(key=lambda t: t[0])
                _, best_gi, best_gap = candidates[0]
                claimed_gap_indices.add(best_gi)
                matched[yi] = _gap_to_rect(best_gap)
                logger.debug(
                    "Phase1: YOLO door #%d matched gap gi=%d dist=%.1f px",
                    yi, best_gi, candidates[0][0],
                )

        # ── Phase 2: paint white + re-detect for unmatched YOLO doors ────
        unmatched_indices = [yi for yi in range(len(yolo_doors)) if yi not in matched]

        if unmatched_indices:
            logger.debug(
                "Phase2: %d YOLO door(s) unmatched — running targeted re-detection",
                len(unmatched_indices),
            )
            img_w = img.copy()
            for yi in unmatched_indices:
                x1, y1, x2, y2 = yolo_doors[yi].bbox
                # Paint the door leaf area white so the wall colour detector
                # sees a gap in the wall rather than a filled rectangle.
                img_w[y1:y2, x1:x2] = 255

            try:
                _, new_rects, _ = self.color_detector.process(img_w)

                # Pass yolo_results=None so the targeted detector does NOT
                # suppress openings that overlap the (now-whitened) YOLO areas.
                new_gaps = self.gap_detector_targeted.detect_openings(
                    rectangles=new_rects,
                    yolo_results=None,
                    pixel_scale=pixel_scale,
                    image=img_w,
                )
                new_gaps = [g for g in new_gaps if getattr(g, 'is_valid', True)]
                logger.debug(
                    "Phase2: targeted detector found %d gap(s)", len(new_gaps)
                )

                # Match new gaps to unmatched YOLO doors
                used_new: Set[int] = set()
                for yi in list(unmatched_indices):
                    ydoor = yolo_doors[yi]
                    bx1, by1, bx2, by2 = ydoor.bbox
                    ycx = (bx1 + bx2) / 2.0
                    ycy = (by1 + by2) / 2.0
                    ex1, ey1, ex2, ey2 = _expanded(ydoor, match_margin_px)

                    candidates = []
                    for gi, gap in enumerate(new_gaps):
                        if gi in used_new:
                            continue
                        gcx, gcy = _gap_center(gap)
                        if _center_inside(gcx, gcy, ex1, ey1, ex2, ey2):
                            dist = math.hypot(gcx - ycx, gcy - ycy)
                            candidates.append((dist, gi, gap))

                    if candidates:
                        candidates.sort(key=lambda t: t[0])
                        _, best_gi, best_gap = candidates[0]
                        used_new.add(best_gi)
                        matched[yi] = _gap_to_rect(best_gap)
                        unmatched_indices.remove(yi)
                        logger.debug(
                            "Phase2: YOLO door #%d matched new gap gi=%d dist=%.1f px",
                            yi, best_gi, candidates[0][0],
                        )

            except Exception as exc:
                logger.warning(
                    "Targeted gap re-detection failed (non-fatal): %s", exc
                )

        # ── Phase 3: fallback — keep raw YOLO bbox ───────────────────────
        for yi in unmatched_indices:
            ydoor = yolo_doors[yi]
            matched[yi] = {
                'x1': float(ydoor.bbox[0]), 'y1': float(ydoor.bbox[1]),
                'x2': float(ydoor.bbox[2]), 'y2': float(ydoor.bbox[3]),
            }
            logger.debug("Phase3: YOLO door #%d kept as raw fallback bbox", yi)

        # ── Assemble final door rect list ─────────────────────────────────
        # Resolved YOLO doors (gap-located or raw fallback), in original order
        result: List[Dict] = [matched[yi] for yi in range(len(yolo_doors))]

        # Standalone gap openings not claimed by any YOLO door are passageways
        # or openings with no corresponding YOLO detection — keep them all.
        for gi, gap in enumerate(existing_gaps):
            if gi not in claimed_gap_indices:
                result.append(_gap_to_rect(gap))

        return result

    # ------------------------------------------------------------------
    # Overlap suppression helpers
    def _do_sh3d_export(
        self,
        img: np.ndarray,
        color_results: Dict,
        yolo_results: Optional[YOLOResults],
        measurements: Optional[RoomMeasurements],
        output_dir: str,
        base_name: str,
    ) -> Tuple[Optional[str], List[Room], List, float]:
        rects = color_results.get('rectangles')
        if not rects:
            return None, [], [], self.pixels_to_cm

        sh3d_dir = os.path.join(output_dir, 'sh3d')
        os.makedirs(sh3d_dir, exist_ok=True)
        sh3d_path = os.path.join(sh3d_dir, f'{base_name}.sh3d')

        ptc = self.pixels_to_cm
        if measurements and measurements.pixel_scale:
            ptc = measurements.pixel_scale * 100.0
            self.sh3d_exporter.update_scale(ptc)

        door_rects = self._resolve_yolo_doors_to_gaps(
            img, color_results, yolo_results,
            pixel_scale=measurements.pixel_scale if measurements else None,
        ) or None
        win_rects = self._all_windows_for_sh3d(
            yolo_results, color_results.get('algo_windows', []),
        )
        ocr_labels = self.color_detector.get_ocr_texts_with_coords(img)

        # === DOOR ARC INTEGRATION — Step 2.3: Pass arcs to exporter ===
        door_arcs = color_results.get('door_arcs', [])

        rooms = self.sh3d_exporter.export_to_sh3d(
            wall_rectangles=self._rects_for_sh3d(rects),
            output_path=sh3d_path,
            door_rectangles=door_rects or None,
            window_rectangles=win_rects,
            original_image=None,
            ocr_labels=None,
            debug_image_path=None,
        )
        return sh3d_path, rooms or [], ocr_labels or [], ptc

    @staticmethod
    def _refine_living_space(
        measurements: RoomMeasurements,
        yolo_results: Optional[YOLOResults],
    ) -> None:
        if not yolo_results or not yolo_results.doors:
            return
        ab = getattr(measurements, 'apartment_boundary', None)
        if ab is None:
            return
        ab_gray = (
            cv2.cvtColor(ab, cv2.COLOR_BGR2GRAY) if ab.ndim == 3 else ab
        )
        interior = ab_gray > 0
        h, w = interior.shape
        remove = np.zeros_like(interior, dtype=bool)
        for det in yolo_results.doors:
            x1, y1, x2, y2 = det.bbox
            x1, x2 = sorted((max(0, int(x1)), min(w, int(math.ceil(x2)))))
            y1, y2 = sorted((max(0, int(y1)), min(h, int(math.ceil(y2)))))
            if x2 <= x1 or y2 <= y1:
                continue
            sub = interior[y1:y2, x1:x2]
            if sub.size and (sub.size - int(sub.sum())) / sub.size > 0.02:
                remove[y1:y2, x1:x2] = True
        if remove.any():
            new_ab = ab_gray.copy()
            new_ab[remove] = 0
            measurements.apartment_boundary = new_ab
            ps = float(getattr(measurements, 'pixel_scale', 0) or 0)
            if ps > 0:
                measurements.area_m2 = float((new_ab > 0).sum()) * (ps ** 2)

    def process_image(
        self,
        img_path: str,
        output_dir: str,
        run_color_detection: bool = True,
        progress: Optional[Progress] = None,
        task_id=None,
    ) -> Dict:
        base_name = Path(img_path).stem

        def _advance(stage: str) -> None:
            if progress is not None and task_id is not None:
                progress.advance(task_id)
                progress.update(
                    task_id,
                    description=f'[dim]{base_name}[/dim]  {stage}',
                )

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'Cannot load image: {img_path}')
        _advance('detecting walls')

        yolo_results: Optional[YOLOResults] = self.yolo_detector.detect(
            img, verbose=False,
        )

        color_results: Dict = {}
        measurements: Optional[RoomMeasurements] = None

        if run_color_detection:
            wall_mask_raw, rectangles, outline_mask = (
                self.color_detector.process(img)
            )
            door_bboxes = (
                [d.bbox for d in yolo_results.doors]
                if yolo_results and yolo_results.doors else []
            )
            ocr_bboxes = self.color_detector.get_ocr_bboxes(img)

            algo_windows = self.algo_window_detector.detect(
                img, outline_mask,
                exclude_bboxes=door_bboxes,
                ocr_bboxes=ocr_bboxes,
            )
            algo_win_mask = self.algo_window_detector.get_mask(
                img.shape, algo_windows,
            )
            _advance('detecting openings')

            wall_mask = wall_mask_raw.copy()
            if np.count_nonzero(algo_win_mask):
                kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                expanded = cv2.dilate(algo_win_mask, kern, iterations=1)
                wall_mask[expanded > 0] = 255

            segments = self._rectangles_to_segments(rectangles)
            color_results = {
                'wall_mask': wall_mask,
                'segments': segments,
                'rectangles': rectangles,
                'outline_mask': outline_mask,
                'openings': [],
                'algo_windows': algo_windows,
                'algo_win_mask': algo_win_mask,
                'door_arcs': [],           # === DOOR ARC INTEGRATION — initialize key ===
            }

            if rectangles:
                try:
                    color_results['openings'] = (
                        self.gap_detector.detect_openings(
                            rectangles=rectangles,
                            yolo_results=yolo_results,
                        )
                    )
                except Exception:
                    pass

            if self.enable_measurements and segments:
                _advance('measuring rooms')
                try:
                    measurements = self.size_analyzer.process(
                        img_path,
                        self._segments_to_dicts(segments),
                        output_dir,
                        wall_mask=outline_mask,
                        original_image=img,
                    )
                except Exception:
                    pass

            if rectangles:
                try:
                    ps = measurements.pixel_scale if measurements else None
                    combined = (
                        list(yolo_results.doors)
                        if yolo_results and yolo_results.doors else []
                    )
                    color_results['openings'] = (
                        self.gap_detector.detect_openings(
                            rectangles=rectangles,
                            yolo_results=yolo_results,
                            pixel_scale=ps,
                            image=img,
                            door_window_boxes=combined,
                        )
                    )
                except Exception:
                    pass

            # ====================== DETECT DOOR SWING ARCS ======================
            # === DOOR ARC INTEGRATION — Step 2.2: Call the arc detector ===
            _advance('detecting door arcs')
            ptc_used = self.pixels_to_cm
            if measurements and measurements.pixel_scale:
                ptc_used = measurements.pixel_scale * 100.0
            try:
                color_results['door_arcs'] = self.arc_detector.detect(
                    img=img,
                    wall_outline_mask=outline_mask,
                    candidate_gaps=color_results.get('openings', []),
                    yolo_doors=yolo_results.doors if yolo_results else None,
                    pixel_scale=ptc_used / 100.0,       # convert cm/px → m/px
                    furniture_mask=color_results.get('furniture_mask'),
                    ocr_boxes=(
                        self.color_detector.get_ocr_bboxes(img)
                        if hasattr(self.color_detector, 'get_ocr_bboxes')
                        else None
                    ),
                    base_name=base_name,
                )
                logger.info(
                    "Door arcs detected: %d", len(color_results['door_arcs'])
                )
            except Exception as e:
                logger.warning("Door arc detection failed (non-fatal): %s", e)
                color_results['door_arcs'] = []
            # =====================================================================

            # Single deduplication pass: drop overlapping and parallel-close boxes
            deduplicate_boxes(yolo_results, color_results)

        if measurements and color_results:
            self._refine_living_space(measurements, yolo_results)

        sh3d_path: Optional[str] = None
        rooms: List[Room] = []
        ocr_labels: List = []
        ptc_used = self.pixels_to_cm
        _advance('exporting')

        if color_results.get('rectangles'):
            sh3d_path, rooms, ocr_labels, ptc_used = self._do_sh3d_export(
                img, color_results, yolo_results,
                measurements, output_dir, base_name,
            )

        if sh3d_path and rooms and ocr_labels:
            new_sf = calibrate_scale_factor(
                rooms, ocr_labels, ptc_used, self.scale_factor,
            )
            if new_sf is not None and abs(new_sf - self.scale_factor) > 1e-4:
                try:
                    os.remove(sh3d_path)
                except OSError:
                    pass
                self.scale_factor = new_sf
                self.sh3d_exporter.xml_generator.scale_factor = new_sf
                sh3d_path, rooms, _, _ = self._do_sh3d_export(
                    img, color_results, yolo_results,
                    measurements, output_dir, base_name,
                )

        if sh3d_path:
            color_results['sh3d_path'] = sh3d_path

        _advance('rendering')

        if run_color_detection and color_results:
            render_summary(
                img, yolo_results, color_results, measurements,
                self.pixels_to_cm,
                os.path.join(output_dir, f'{base_name}_summary.png'),
            )

        if rooms:
            render_room_ocr_overlay(
                img, rooms, [], ocr_labels, ptc_used,
                os.path.join(output_dir, f'{base_name}_rooms_ocr.png'),
                door_arcs=color_results.get('door_arcs', []),   # === DOOR ARC INTEGRATION ===
            )

        _advance('done')

        return {
            'yolo': yolo_results,
            'color': color_results,
            'measurements': measurements,
        }


# ============================================================
# CLI
# ============================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Floor Plan Analysis System v5.3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input', required=True,
                        help='Single image or directory')
    parser.add_argument('--output', default='output_v5',
                        help='Output directory (default: output_v5)')
    parser.add_argument('--yolo-model', required=True,
                        help='Path to YOLOv9 .pt weights')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                        help='YOLO confidence threshold (default: 0.25)')
    parser.add_argument('--device', default='cpu',
                        help='cpu | cuda | mps (default: cpu)')
    parser.add_argument('--wall-color', default='8E8780',
                        help='Brusnika wall hex color (default: 8E8780)')
    parser.add_argument('--color-tolerance', type=int, default=3,
                        help='Per-channel color tolerance (default: 3)')
    parser.add_argument('--auto-wall-color', action='store_true',
                        help='Auto-detect wall color from image')
    parser.add_argument('--skip-color', action='store_true',
                        help='Skip color-based wall detection')
    parser.add_argument('--wall-mode', default='hybrid',
                        choices=['color', 'structure', 'hybrid', 'auto'],
                        help='Wall detection mode (default: hybrid)')
    parser.add_argument('--structure-method', default='lines',
                        choices=['lines', 'edges', 'contours'],
                        help='Structural detection method (default: lines)')

    parser.add_argument('--disable-measurements', action='store_true',
                        help='Disable OCR room measurement')
    parser.add_argument('--edge-distance-ratio', type=float, default=0.15)
    parser.add_argument('--debug-ocr', action='store_true')
    parser.add_argument('--min-window-len', type=int, default=15)
    parser.add_argument('--edge-expansion-tolerance', type=int, default=8)
    parser.add_argument('--max-edge-expansion', type=int, default=10)
    parser.add_argument('--wall-height', type=float, default=243.84,
                        help='Wall height cm for 3D export (default: 243.84)')
    parser.add_argument('--min-wall-thickness', type=float, default=5.0)
    parser.add_argument('--pixels-to-cm', type=float, default=2.54)
    parser.add_argument('--scale-factor', type=float, default=SCALE_FACTOR)
    parser.add_argument('--disable-obj', action='store_true')
    parser.add_argument('--no-floor', action='store_true')
    parser.add_argument('--include-ceiling', action='store_true')

    # === DOOR ARC INTEGRATION — CLI flag for debug mode ===
    parser.add_argument('--debug-arcs', action='store_true',
                        help='Save door arc detection debug images')

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    inp = Path(args.input)
    if inp.is_file():
        image_files = [str(inp)]
    elif inp.is_dir():
        image_files = sorted(
            str(f)
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp')
            for f in inp.glob(ext)
        )
    else:
        raise ValueError(f'Path not found: {args.input}')
    if not image_files:
        raise ValueError(f'No images found at: {args.input}')

    analyzer = HybridWallAnalyzer(
        yolo_model_path=args.yolo_model,
        wall_color_hex=args.wall_color,
        color_tolerance=args.color_tolerance,
        yolo_conf_threshold=args.yolo_conf,
        device=args.device,
        auto_wall_color=args.auto_wall_color,
        enable_measurements=not args.disable_measurements,
        edge_distance_ratio=args.edge_distance_ratio,
        debug_ocr=args.debug_ocr,
        min_window_len=args.min_window_len,
        edge_expansion_tolerance=args.edge_expansion_tolerance,
        max_edge_expansion_px=args.max_edge_expansion,
        wall_height_cm=args.wall_height,
        min_wall_thickness_px=args.min_wall_thickness,
        pixels_to_cm=args.pixels_to_cm,
        enable_obj_export=not args.disable_obj,
        obj_include_floor=not args.no_floor,
        obj_include_ceiling=args.include_ceiling,
        scale_factor=args.scale_factor,
        wall_detection_mode=args.wall_mode,
        structure_method=args.structure_method,
        output_dir=args.output,                    # === DOOR ARC INTEGRATION ===
    )

    # === DOOR ARC INTEGRATION — apply --debug-arcs flag ===
    if args.debug_arcs:
        analyzer.arc_detector.debug = True

    n_stages = len(Theme.PIPELINE_STAGES)

    _console.print()
    _console.print(Text('  Floor Plan Analysis', style='italic'))
    _console.print()

    ok, fail = 0, 0

    with Progress(
        TextColumn('  {task.description}', justify='left', style='default'),
        BarColumn(
            bar_width=24,
            complete_style='white',
            finished_style='dim',
            pulse_style='dim white',
        ),
        TaskProgressColumn(style='dim'),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
        expand=False,
    ) as progress:
        for path in image_files:
            name = Path(path).stem
            task_id = progress.add_task(
                f'[dim]{name}[/dim]  loading',
                total=n_stages,
            )
            try:
                res = analyzer.process_image(
                    path, args.output,
                    run_color_detection=not args.skip_color,
                    progress=progress,
                    task_id=task_id,
                )
                c = res.get('color', {})
                n_walls = len(c.get('segments', []))
                n_arcs = len(c.get('door_arcs', []))    # === DOOR ARC INTEGRATION ===
                suffix = f'{n_walls} walls'
                if n_arcs > 0:
                    suffix += f'  {n_arcs} arcs'         # === DOOR ARC INTEGRATION ===
                if c.get('sh3d_path'):
                    suffix += '  sh3d ✓'
                progress.update(
                    task_id,
                    description=f'[dim]{name}[/dim]  {suffix}',
                )
                ok += 1
            except Exception as e:
                progress.update(
                    task_id,
                    description=f'[dim]{name}[/dim]  [red]✗ {e}[/red]',
                )
                fail += 1

    _console.print()
    _console.print(
        f'  [dim]{ok}/{ok + fail} completed  ·  {args.output}/[/dim]'
    )
    _console.print()


if __name__ == '__main__':
    main()