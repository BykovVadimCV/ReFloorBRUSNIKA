"""
Clean pipeline orchestrator for the ReFloorBRUSNIKA floorplan pipeline.

Replaces the monolithic HybridWallAnalyzer with a clean, linear pipeline
that calls each stage once and passes typed data between stages.

Pipeline stages:
  1. Wall Detection
  2. Opening Detection (YOLO + geometric + algorithmic + door arcs)
  3. Room Measurement + Scale Calibration (OCR)
  4. SH3D Export
  5. Visualization
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.config import PipelineConfig
from core.models import (
    BBox,
    DetectionResult,
    Opening,
    OpeningType,
    Room,
    WallSegment,
)
from detection.walls import (
    ColorWallDetector,
    _build_enclosed_regions,
    build_grey_wall_image,
    debug_enclosed_regions,
    find_unenclosed_wall_rects,
    paint_unenclosed_walls_grey,
    refine_wall_mask_by_enclosed_regions,
    snap_walls,
    wall_rectangle_to_segment,
)
from detection.openings import OpeningDetectionPipeline
from floorplan_export.sh3d import SH3DExporter
from floorplan_export.visualization import FloorplanVisualizer

logger = logging.getLogger(__name__)


# ============================================================
# SCALE CALIBRATION HELPERS  (ported from legacy main.py)
# ============================================================

_AREA_PATTERN = re.compile(
    r'^(\d{1,3}[.,]\d{1,2})\s*(?:m²|м²|m2|м2|кв\.?\s*м\.?)?$',
    re.IGNORECASE,
)

_LENGTH_PATTERN = re.compile(
    r'^(\d{1,3}[.,]\d{1,2})$',
)


def _deskew_image(img: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """Rotate image so that dominant lines snap to X/Y axes.

    Uses Canny + HoughLinesP to detect line segments, collects their
    angles relative to horizontal/vertical, and rotates by the median
    deviation.  Only corrects small skew (up to *max_angle* degrees).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=80, minLineLength=40, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 2 and abs(dy) < 2:
            continue
        angle_deg = np.degrees(np.arctan2(dy, dx))
        # Snap to nearest axis: 0 (horizontal) or 90 (vertical)
        # Deviation from horizontal
        dev_h = angle_deg % 90
        if dev_h > 45:
            dev_h -= 90
        angles.append(dev_h)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.05 or abs(median_angle) > max_angle:
        return img

    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    # Compute new bounding size to avoid clipping corners
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
    logger.info("Deskew: rotated %.2f deg", median_angle)
    return rotated


def _parse_ocr_area_m2(text: str) -> Optional[float]:
    """
    Parse an OCR word to a room area in m².

    Accepts formats: '12,4', '12.4', '12,4 м²', '52.0 m2', etc.
    Returns None if the text does not match.
    """
    match = _AREA_PATTERN.match(text.strip())
    if match:
        try:
            return float(match.group(1).replace(',', '.'))
        except ValueError:
            pass
    return None


def _parse_ocr_length_m(text: str) -> Optional[float]:
    """Parse an OCR word as a wall length in metres (e.g. '3.92', '2,55')."""
    match = _LENGTH_PATTERN.match(text.strip())
    if match:
        try:
            val = float(match.group(1).replace(',', '.'))
            if 0.3 <= val <= 30.0:          # sane wall length range
                return val
        except ValueError:
            pass
    return None


def _calibrate_scale_from_wall_lengths(
    img: np.ndarray,
    walls: List,
    normal_labels: List[Tuple],
    rotated_labels: List[Tuple],
    max_dist: float = 80.0,
    debug_dir: Optional[str] = None,
) -> Optional[float]:
    """Match rotated wall-length labels to walls and compute pixels_to_cm.

    Receives pre-collected OCR labels from both orientations (collected
    in the pre-stage before wall detection).  Filters to NEW numerical
    labels only, un-rotates coordinates, and matches to walls.
    """
    h, w = img.shape[:2]

    if not rotated_labels:
        return None

    # --- 1. Existing numerical labels (normal orientation) ----------------
    existing_nums: set = set()
    for item in (normal_labels or []):
        text = str(item[0]).strip()
        val = _parse_ocr_length_m(text)
        if val is not None:
            existing_nums.add(round(val, 2))

    # --- 2. Filter to NEW numerical labels only ---------------------------
    new_labels: List[Tuple[float, float, float]] = []  # (length_m, cx_orig, cy_orig)
    for item in rotated_labels:
        if len(item) < 5:
            continue
        text, rx1, ry1, rx2, ry2 = str(item[0]), item[1], item[2], item[3], item[4]
        val = _parse_ocr_length_m(text)
        if val is None:
            continue
        if round(val, 2) in existing_nums:
            continue

        # Un-rotate: CW-rotated coords (rx, ry) -> original (ox, oy)
        # CW rotation: original (ox, oy) -> rotated (h-1-oy, ox)
        # Inverse:     rotated (rx, ry) -> original (ry, h-1-rx)
        rcx = (rx1 + rx2) / 2.0
        rcy = (ry1 + ry2) / 2.0
        ox = rcy
        oy = h - 1 - rcx
        new_labels.append((val, ox, oy))

    if not new_labels:
        logger.info("Wall-length calibration: no new rotated labels found")
        return None

    logger.info("Wall-length calibration: found %d rotated labels: %s",
                len(new_labels),
                [(f"{v:.2f}m", int(cx), int(cy)) for v, cx, cy in new_labels])

    # --- 4. Match each label to nearest wall line between perp cuts -------
    #
    # Key ideas:
    #  a) Collinear walls of similar thickness are merged into one
    #     continuous "wall line", bridging over door gaps.
    #  b) Perpendicular cuts are found along the full merged line.
    #  c) Thickness changes (wall expansion at corners) are also cuts.
    #  d) The two cuts bracketing the label define the measured segment.

    def _pt_to_seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def _merge_collinear(seed, all_walls, axis_tol=10.0, th_ratio=2.5):
        """Find all walls collinear with *seed* and of similar thickness.

        Returns (merged_lo, merged_hi, ref_thickness, member_walls)
        where lo/hi are the span along the major axis.
        Walls with thickness > th_ratio * seed.thickness are excluded.
        """
        is_h = seed.is_horizontal
        seed_th = min(seed.bbox.width, seed.bbox.height)
        seed_axis_pos = seed.bbox.center_y if is_h else seed.bbox.center_x

        members = []
        for w in all_walls:
            if w.is_horizontal != is_h:
                continue
            w_th = min(w.bbox.width, w.bbox.height)
            w_axis_pos = w.bbox.center_y if is_h else w.bbox.center_x
            # Must be on the same line (thin-axis position close)
            if abs(w_axis_pos - seed_axis_pos) > axis_tol:
                continue
            # Similar thickness (skip much thicker walls at corners)
            if w_th > seed_th * th_ratio and seed_th > 0:
                continue
            members.append(w)

        if not members:
            members = [seed]

        if is_h:
            lo = min(w.bbox.x1 for w in members)
            hi = max(w.bbox.x2 for w in members)
        else:
            lo = min(w.bbox.y1 for w in members)
            hi = max(w.bbox.y2 for w in members)

        return lo, hi, seed_th, members

    def _find_perp_cuts(line_lo, line_hi, is_h, thin_pos, thin_half,
                        all_walls, tol=5.0, min_extend=25.0,
                        min_total_extend=30.0,
                        grey_snap=15.0, diag_tol=4):
        """Positions along the wall line where perpendicular walls cross.

        A perpendicular wall counts as a primary cut only if:
        - It extends beyond the wall's thin-axis band by at least
          *min_extend* px on at least one side, AND
        - The total extension on both sides >= *min_total_extend*.
        This filters out T-junction stubs that only extend on one side.

        Short perpendicular segments ("grey lines") that fail these
        thresholds are collected separately.  If a grey line is within
        *grey_snap* px of a primary cut, the cut is snapped to the
        grey line's exact position for better precision.

        Grey lines that appear in clusters (2+ within 50px) are also
        accepted as independent cuts (window boundary markers).

        *diag_tol*: a wall is considered perpendicular even if its thin
        dimension is up to this many pixels (allows slight diagonal tilt).
        """
        primary = []
        grey = []
        for other in all_walls:
            ob = other.bbox
            if is_h:
                if ob.width > ob.height + diag_tol:
                    continue
            else:
                if ob.height > ob.width + diag_tol:
                    continue
            if is_h:
                if ob.y1 > thin_pos + thin_half + tol:
                    continue
                if ob.y2 < thin_pos - thin_half - tol:
                    continue
                ext_a = (thin_pos - thin_half) - ob.y1
                ext_b = ob.y2 - (thin_pos + thin_half)
                cut = ob.center_x
                if line_lo - tol <= cut <= line_hi + tol:
                    if (max(ext_a, ext_b) >= min_extend
                            and (ext_a + ext_b) >= min_total_extend):
                        primary.append(cut)
                    else:
                        grey.append(cut)
            else:
                if ob.x1 > thin_pos + thin_half + tol:
                    continue
                if ob.x2 < thin_pos - thin_half - tol:
                    continue
                ext_a = (thin_pos - thin_half) - ob.x1
                ext_b = ob.x2 - (thin_pos + thin_half)
                cut = ob.center_y
                if line_lo - tol <= cut <= line_hi + tol:
                    if (max(ext_a, ext_b) >= min_extend
                            and (ext_a + ext_b) >= min_total_extend):
                        primary.append(cut)
                    else:
                        grey.append(cut)

        # Snap primary cuts to nearby grey lines for precision.
        snapped = []
        for pc in primary:
            best_grey = None
            best_d = grey_snap + 1
            for gc in grey:
                d = abs(pc - gc)
                if d < best_d:
                    best_d = d
                    best_grey = gc
            snapped.append(best_grey if best_grey is not None and best_d <= grey_snap else pc)

        # Grey lines that appear in clusters (2+ within cluster_dist)
        # indicate window boundaries and should be independent cuts.
        cluster_dist = 50.0
        grey_cuts = set()
        for i, g1 in enumerate(grey):
            for j, g2 in enumerate(grey):
                if i != j and abs(g1 - g2) <= cluster_dist:
                    grey_cuts.add(g1)
                    grey_cuts.add(g2)
                    break

        all_cuts = set(snapped) | grey_cuts
        return sorted(all_cuts), set(primary), set(grey), grey_cuts

    ratios: List[float] = []
    # (lcx, lcy, seg_start, seg_end, is_h, wall, length_m, px_per_cm, cuts)
    matches: List[Tuple] = []
    for length_m, lcx, lcy in new_labels:
        # Find nearest wall by perpendicular distance to centerline
        best_wall = None
        best_dist = float('inf')
        for wall in walls:
            (ax, ay), (bx, by) = wall.centerline
            d = _pt_to_seg_dist(lcx, lcy, ax, ay, bx, by)
            if d < best_dist:
                best_dist = d
                best_wall = wall

        if best_wall is None or best_dist > max_dist:
            continue

        is_h = best_wall.is_horizontal

        # Merge collinear walls of similar thickness (bridge over doors)
        line_lo, line_hi, ref_th, members = _merge_collinear(best_wall, walls)

        # Thin-axis centre and half-thickness for perpendicular overlap check
        thin_pos = best_wall.bbox.center_y if is_h else best_wall.bbox.center_x
        thin_half = ref_th / 2.0

        # Find perpendicular cuts along the full merged line
        cuts, dbg_primary, dbg_grey, dbg_grey_used = _find_perp_cuts(
            line_lo, line_hi, is_h, thin_pos, thin_half, walls)

        # Project label onto wall's major axis
        label_pos = lcx if is_h else lcy

        # Boundaries = wall-line endpoints + perpendicular cuts
        boundaries = sorted(set([line_lo, line_hi] + cuts))

        # Find the two boundaries that bracket the label position
        lo_bound = line_lo
        hi_bound = line_hi
        for b in boundaries:
            if b <= label_pos and b >= lo_bound:
                lo_bound = b
            if b >= label_pos and b <= hi_bound:
                hi_bound = b

        seg_len_px = abs(hi_bound - lo_bound)
        if seg_len_px < 5:
            continue

        px_per_cm = seg_len_px / (length_m * 100.0)
        logger.info("  label %.2fm -> wall line seg=[%.0f..%.0f] len=%.0fpx  "
                    "(%d members)  => px/cm=%.4f",
                    length_m, lo_bound, hi_bound, seg_len_px,
                    len(members), px_per_cm)
        ratios.append(px_per_cm)
        matches.append((lcx, lcy, lo_bound, hi_bound, is_h,
                         best_wall, length_m, px_per_cm, cuts,
                         line_lo, line_hi,
                         dbg_primary, dbg_grey, dbg_grey_used))

    # --- Debug: save original image with matches --------------------------
    if debug_dir:
        dbg_orig = img.copy()
        # Draw all walls faintly
        for wall in walls:
            b = wall.bbox
            cv2.rectangle(dbg_orig,
                          (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
                          (200, 200, 200), 1)
        # Draw matched segments + labels
        for (lcx, lcy, seg_lo, seg_hi, is_h, wall, length_m, ppcm,
             cuts, m_lo, m_hi, m_primary, m_grey, m_grey_used) in matches:
            wb = wall.bbox
            # Full wall in thin gray
            cv2.rectangle(dbg_orig,
                          (int(wb.x1), int(wb.y1)), (int(wb.x2), int(wb.y2)),
                          (180, 180, 180), 1)
            # Merged wall line extent in cyan dotted
            if is_h:
                cy = int(wb.center_y)
                for px_x in range(int(m_lo), int(m_hi), 6):
                    cv2.line(dbg_orig, (px_x, cy), (min(px_x + 3, int(m_hi)), cy),
                             (200, 200, 0), 1)
            else:
                cx = int(wb.center_x)
                for px_y in range(int(m_lo), int(m_hi), 6):
                    cv2.line(dbg_orig, (cx, px_y), (cx, min(px_y + 3, int(m_hi))),
                             (200, 200, 0), 1)
            # Matched segment in thick green
            if is_h:
                cy = int(wb.center_y)
                cv2.line(dbg_orig, (int(seg_lo), cy), (int(seg_hi), cy),
                         (0, 220, 0), 3)
            else:
                cx = int(wb.center_x)
                cv2.line(dbg_orig, (cx, int(seg_lo)), (cx, int(seg_hi)),
                         (0, 220, 0), 3)
            # Draw cut positions colour-coded by type
            for c in cuts:
                if c in m_primary:
                    col, marker = (255, 0, 255), cv2.MARKER_CROSS  # magenta +
                elif c in m_grey_used:
                    col, marker = (0, 200, 200), cv2.MARKER_CROSS  # yellow +
                else:
                    col, marker = (0, 200, 200), cv2.MARKER_CROSS
                if is_h:
                    cv2.drawMarker(dbg_orig, (int(c), int(wb.center_y)),
                                   col, marker, 10, 1)
                else:
                    cv2.drawMarker(dbg_orig, (int(wb.center_x), int(c)),
                                   col, marker, 10, 1)
            # Unused grey lines in dark yellow (for reference)
            for g in m_grey:
                if g not in m_grey_used and g not in set(cuts):
                    if is_h:
                        cv2.drawMarker(dbg_orig, (int(g), int(wb.center_y)),
                                       (0, 140, 140), cv2.MARKER_TILTED_CROSS,
                                       6, 1)
                    else:
                        cv2.drawMarker(dbg_orig, (int(wb.center_x), int(g)),
                                       (0, 140, 140), cv2.MARKER_TILTED_CROSS,
                                       6, 1)
            # Label marker in red
            cv2.circle(dbg_orig, (int(lcx), int(lcy)), 6, (0, 0, 255), -1)
            # Line connecting label to segment midpoint
            if is_h:
                mid = (int((seg_lo + seg_hi) / 2), int(wb.center_y))
            else:
                mid = (int(wb.center_x), int((seg_lo + seg_hi) / 2))
            cv2.line(dbg_orig, (int(lcx), int(lcy)), mid, (0, 0, 255), 1)
            # Text annotation
            txt = f"{length_m:.2f}m -> {seg_hi - seg_lo:.0f}px  px/cm={ppcm:.3f}"
            cv2.putText(dbg_orig, txt, (int(lcx) + 8, int(lcy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # Draw unmatched labels in orange
        matched_set = {(int(m[0]), int(m[1])) for m in matches}
        for val, ox, oy in new_labels:
            if (int(ox), int(oy)) not in matched_set:
                cv2.circle(dbg_orig, (int(ox), int(oy)), 6, (0, 140, 255), -1)
                cv2.putText(dbg_orig, f"{val:.2f}m (no match)",
                            (int(ox) + 8, int(oy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
        # Result annotation
        if ratios:
            median_val = float(np.median(ratios))
            cv2.putText(dbg_orig,
                        f"Wall-length calib: median px/cm = {median_val:.4f}  "
                        f"({len(ratios)} matches)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, "_wl_matches.png"), dbg_orig)

    if not ratios:
        return None

    # --- Outlier rejection: discard ratios >30% from median ----------------
    med = float(np.median(ratios))
    filtered = [r for r in ratios if abs(r - med) / med < 0.30]
    if not filtered:
        filtered = ratios  # keep all if everything would be rejected
    result = float(np.median(filtered))
    if len(filtered) < len(ratios):
        logger.info("Wall-length calibration: rejected %d/%d outlier(s)",
                    len(ratios) - len(filtered), len(ratios))
    logger.info("Wall-length calibration: median pixels_to_cm = %.4f  (%d matches)",
                result, len(filtered))
    return result


def _polygon_area_from_tuples(pts: List[Tuple[float, float]]) -> float:
    """Shoelace formula — polygon area from a list of (x, y) tuples."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def _point_in_polygon(x: float, y: float, pts: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(pts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _validate_room_areas(
    rooms: list,
    ocr_text_labels: list,
    pixels_to_cm: float,
    scale_factor: float,
    tolerance_ratio: float = 0.30,
) -> list:
    """Compare each room's polygon area to OCR area labels inside it.

    Returns a list of dicts:
      {room_name, computed_area_m2, ocr_area_m2, ratio, status}
    where ``status`` is ``"ok"`` or ``"mismatch"``.
    """
    m_per_px = pixels_to_cm / 100.0
    reports: list = []
    for room in rooms:
        pts_px = [(p.x / pixels_to_cm, p.y / pixels_to_cm) for p in room.points]
        area_px2 = _polygon_area_from_tuples(pts_px)
        computed_m2 = area_px2 * (m_per_px ** 2)

        # Find OCR area labels whose centre falls inside this room polygon
        matched_area: Optional[float] = None
        for label in ocr_text_labels:
            text, lx1, ly1, lx2, ly2 = label[0], label[1], label[2], label[3], label[4]
            parsed = _parse_ocr_area_m2(text)
            if parsed is None:
                continue
            cx_label = (lx1 + lx2) / 2.0
            cy_label = (ly1 + ly2) / 2.0
            # OCR coords are in pixels; room points are in cm — convert label to cm
            cx_cm = cx_label * pixels_to_cm
            cy_cm = cy_label * pixels_to_cm
            pts_cm = [(p.x, p.y) for p in room.points]
            if _point_in_polygon(cx_cm, cy_cm, pts_cm):
                matched_area = parsed
                break  # take first match

        if matched_area is not None:
            ratio = computed_m2 / matched_area if matched_area > 0 else 0
            status = "ok" if abs(ratio - 1.0) <= tolerance_ratio else "mismatch"
            if status == "mismatch":
                logger.debug(
                    "Room %s area mismatch: computed=%.1f m², OCR=%.1f m² (ratio=%.2f)",
                    room.name, computed_m2, matched_area, ratio,
                )
            reports.append({
                "room_name": room.name,
                "computed_area_m2": round(computed_m2, 2),
                "ocr_area_m2": matched_area,
                "ratio": round(ratio, 3),
                "status": status,
            })
    return reports


# ============================================================
# WALL CLEANUP
# ============================================================


def cleanup_walls(walls: List[WallSegment],
                  min_thickness: float = 3.0,
                  min_length: float = 8.0,
                  thickness_snap: float = 2.0,
                  endpoint_snap: float = 5.0,
                  ) -> List[WallSegment]:
    """Post-process walls to remove noise and fix alignment issues.

    1. Remove zero-length walls.
    2. Remove walls thinner than *min_thickness* px.
    3. Remove very short walls that aren't connected to anything.
    4. Merge overlapping co-axial walls into one.
    5. Snap parallel wall thicknesses to the nearest *thickness_snap*.
    6. Snap endpoints of collinear/perpendicular walls within *endpoint_snap*.
    """
    if not walls:
        return walls

    # --- 1. Remove zero-length and degenerate walls -------------------------
    clean: List[WallSegment] = []
    for w in walls:
        length = max(w.bbox.width, w.bbox.height)
        thickness = min(w.bbox.width, w.bbox.height)
        if length < 0.5:
            continue  # zero-length
        if thickness < min_thickness:
            continue  # sub-pixel noise
        clean.append(w)

    # --- 2. Remove short orphan walls (not touching any other wall) ---------
    def _touches(a: WallSegment, b: WallSegment, margin: float = 3.0) -> bool:
        """True if bboxes overlap or are within margin."""
        return not (a.bbox.x2 + margin < b.bbox.x1 or
                    b.bbox.x2 + margin < a.bbox.x1 or
                    a.bbox.y2 + margin < b.bbox.y1 or
                    b.bbox.y2 + margin < a.bbox.y1)

    kept: List[WallSegment] = []
    for i, w in enumerate(clean):
        length = max(w.bbox.width, w.bbox.height)
        if length >= min_length:
            kept.append(w)
            continue
        # Short wall — keep only if it touches a longer wall
        has_neighbor = any(
            _touches(w, clean[j]) and max(clean[j].bbox.width, clean[j].bbox.height) >= min_length
            for j in range(len(clean)) if j != i
        )
        if has_neighbor:
            kept.append(w)
    clean = kept

    # --- 3. Merge overlapping co-axial walls --------------------------------
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(clean):
            j = i + 1
            while j < len(clean):
                wi = clean[i]
                wj = clean[j]
                # Both same orientation?
                if wi.is_horizontal != wj.is_horizontal:
                    j += 1
                    continue
                if wi.is_horizontal:
                    # Centerlines close on Y?
                    cy_i = wi.bbox.center_y
                    cy_j = wj.bbox.center_y
                    if abs(cy_i - cy_j) > max(wi.thickness, wj.thickness) * 0.6:
                        j += 1
                        continue
                    # X ranges overlap?
                    if wi.bbox.x2 < wj.bbox.x1 - endpoint_snap or wj.bbox.x2 < wi.bbox.x1 - endpoint_snap:
                        j += 1
                        continue
                    # Merge
                    nx1 = min(wi.bbox.x1, wj.bbox.x1)
                    nx2 = max(wi.bbox.x2, wj.bbox.x2)
                    ny1 = min(wi.bbox.y1, wj.bbox.y1)
                    ny2 = max(wi.bbox.y2, wj.bbox.y2)
                else:
                    cx_i = wi.bbox.center_x
                    cx_j = wj.bbox.center_x
                    if abs(cx_i - cx_j) > max(wi.thickness, wj.thickness) * 0.6:
                        j += 1
                        continue
                    if wi.bbox.y2 < wj.bbox.y1 - endpoint_snap or wj.bbox.y2 < wi.bbox.y1 - endpoint_snap:
                        j += 1
                        continue
                    nx1 = min(wi.bbox.x1, wj.bbox.x1)
                    nx2 = max(wi.bbox.x2, wj.bbox.x2)
                    ny1 = min(wi.bbox.y1, wj.bbox.y1)
                    ny2 = max(wi.bbox.y2, wj.bbox.y2)

                new_wall = WallSegment(
                    id=wi.id,
                    bbox=BBox(x1=nx1, y1=ny1, x2=nx2, y2=ny2),
                    thickness=float(min(nx2 - nx1, ny2 - ny1)),
                    is_structural=wi.is_structural or wj.is_structural,
                    is_outer=wi.is_outer or wj.is_outer,
                )
                clean[i] = new_wall
                clean.pop(j)
                merged = True
                continue  # re-check i against remaining
            i += 1

    # --- 4. Quantize thickness to nearest snap value ------------------------
    if thickness_snap > 0:
        for i, w in enumerate(clean):
            th = w.thickness
            snapped_th = round(th / thickness_snap) * thickness_snap
            snapped_th = max(snapped_th, thickness_snap)  # never go to 0
            if abs(snapped_th - th) < thickness_snap * 0.6:
                # Adjust the bbox's thin dimension symmetrically
                if w.is_horizontal:
                    cy = w.bbox.center_y
                    half = snapped_th / 2
                    clean[i] = WallSegment(
                        id=w.id,
                        bbox=BBox(x1=w.bbox.x1, y1=cy - half, x2=w.bbox.x2, y2=cy + half),
                        thickness=snapped_th,
                        is_structural=w.is_structural, is_outer=w.is_outer,
                    )
                else:
                    cx = w.bbox.center_x
                    half = snapped_th / 2
                    clean[i] = WallSegment(
                        id=w.id,
                        bbox=BBox(x1=cx - half, y1=w.bbox.y1, x2=cx + half, y2=w.bbox.y2),
                        thickness=snapped_th,
                        is_structural=w.is_structural, is_outer=w.is_outer,
                    )

    # --- 5. Snap endpoints to nearby walls ----------------------------------
    for i in range(len(clean)):
        wi = clean[i]
        for j in range(len(clean)):
            if i == j:
                continue
            wj = clean[j]

            if wi.is_horizontal and wj.is_vertical:
                # Snap right end of H wall to V wall's x-center
                cx_j = wj.bbox.center_x
                if abs(wi.bbox.x2 - cx_j) <= endpoint_snap:
                    ov = min(wi.bbox.y2, wj.bbox.y2) - max(wi.bbox.y1, wj.bbox.y1)
                    if ov > 0:
                        clean[i] = WallSegment(
                            id=wi.id,
                            bbox=BBox(x1=wi.bbox.x1, y1=wi.bbox.y1, x2=cx_j, y2=wi.bbox.y2),
                            thickness=wi.thickness, is_structural=wi.is_structural,
                            is_outer=wi.is_outer,
                        )
                        wi = clean[i]
                # Snap left end
                if abs(wi.bbox.x1 - cx_j) <= endpoint_snap:
                    ov = min(wi.bbox.y2, wj.bbox.y2) - max(wi.bbox.y1, wj.bbox.y1)
                    if ov > 0:
                        clean[i] = WallSegment(
                            id=wi.id,
                            bbox=BBox(x1=cx_j, y1=wi.bbox.y1, x2=wi.bbox.x2, y2=wi.bbox.y2),
                            thickness=wi.thickness, is_structural=wi.is_structural,
                            is_outer=wi.is_outer,
                        )
                        wi = clean[i]

            elif wi.is_vertical and wj.is_horizontal:
                cy_j = wj.bbox.center_y
                if abs(wi.bbox.y2 - cy_j) <= endpoint_snap:
                    ov = min(wi.bbox.x2, wj.bbox.x2) - max(wi.bbox.x1, wj.bbox.x1)
                    if ov > 0:
                        clean[i] = WallSegment(
                            id=wi.id,
                            bbox=BBox(x1=wi.bbox.x1, y1=wi.bbox.y1, x2=wi.bbox.x2, y2=cy_j),
                            thickness=wi.thickness, is_structural=wi.is_structural,
                            is_outer=wi.is_outer,
                        )
                        wi = clean[i]
                if abs(wi.bbox.y1 - cy_j) <= endpoint_snap:
                    ov = min(wi.bbox.x2, wj.bbox.x2) - max(wi.bbox.x1, wj.bbox.x1)
                    if ov > 0:
                        clean[i] = WallSegment(
                            id=wi.id,
                            bbox=BBox(x1=wi.bbox.x1, y1=cy_j, x2=wi.bbox.x2, y2=wi.bbox.y2),
                            thickness=wi.thickness, is_structural=wi.is_structural,
                            is_outer=wi.is_outer,
                        )
                        wi = clean[i]

    # --- 6. Re-assign IDs ---------------------------------------------------
    result = []
    for i, w in enumerate(clean):
        result.append(WallSegment(
            id=f"wall_{i}",
            bbox=w.bbox,
            thickness=w.thickness,
            is_structural=w.is_structural,
            is_outer=w.is_outer,
            outline=w.outline,
        ))

    logger.info("Wall cleanup: %d -> %d walls", len(walls), len(result))
    return result


# ============================================================
# RESULT TYPE
# ============================================================

@dataclass
class PipelineResult:
    """
    Full result from a pipeline run on one image.

    Contains all detected geometry plus paths to output files.
    """
    # Detected geometry
    walls: List[WallSegment] = field(default_factory=list)
    openings: List[Opening] = field(default_factory=list)
    door_arcs: List[Any] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)

    # Masks
    wall_mask: Optional[np.ndarray] = None
    outline_mask: Optional[np.ndarray] = None

    # Scale calibration
    pixel_scale: Optional[float] = None  # meters per pixel
    pixels_to_cm: float = 2.54

    # OCR labels
    ocr_labels: List[Any] = field(default_factory=list)

    # Output paths
    sh3d_path: Optional[str] = None
    summary_path: Optional[str] = None
    rooms_ocr_path: Optional[str] = None

    # Room area validation reports
    room_area_reports: List[Any] = field(default_factory=list)

    # Legacy raw results (for backward compat)
    raw_measurements: Optional[Any] = None
    raw_yolo_results: Optional[Any] = None

    @property
    def n_walls(self) -> int:
        return len(self.walls)

    @property
    def n_doors(self) -> int:
        return sum(1 for o in self.openings if o.opening_type == OpeningType.DOOR)

    @property
    def n_windows(self) -> int:
        return sum(1 for o in self.openings if o.opening_type == OpeningType.WINDOW)

    @property
    def n_gaps(self) -> int:
        return sum(1 for o in self.openings if o.opening_type == OpeningType.GAP)

    def to_api_dict(self) -> Dict:
        """Serialize for API response."""
        return {
            'n_walls': self.n_walls,
            'n_doors': self.n_doors,
            'n_windows': self.n_windows,
            'n_gaps': self.n_gaps,
            'n_rooms': len(self.rooms),
            'n_door_arcs': len(self.door_arcs),
            'pixel_scale': self.pixel_scale,
            'pixels_to_cm': self.pixels_to_cm,
            'sh3d_path': self.sh3d_path,
        }


# ============================================================
# U-NET SUBPROCESS ISOLATION
# ============================================================

def _unet_worker(img_path, result_path, checkpoint_path, device, img_size,
                 confidence, min_wall_area, min_door_area, min_window_area):
    """Worker function for subprocess — runs U-Net inference.

    Reads input image from *img_path* (numpy .npy file), writes results
    to *result_path* (pickle file).  If inference crashes, the subprocess
    dies and no result file is written — the parent detects this.
    """
    import pickle
    try:
        img_rgb = np.load(img_path)
        from detect_unet import run_unet_pipeline
        result = run_unet_pipeline(
            image_rgb=img_rgb,
            checkpoint_path=checkpoint_path,
            device=device,
            img_size=img_size,
            confidence=confidence,
            min_wall_area=min_wall_area,
            min_door_area=min_door_area,
            min_window_area=min_window_area,
        )
        if result is not None:
            out = {
                "wall_bboxes": result["wall_bboxes"],
                "door_bboxes": result["door_bboxes"],
                "window_bboxes": result["window_bboxes"],
                "wall_mask": result["wall_mask"],
            }
            with open(result_path, "wb") as f:
                pickle.dump(out, f)
    except Exception as e:
        import pickle
        with open(result_path, "wb") as f:
            pickle.dump({"error": str(e)}, f)


def _run_unet_subprocess(img_rgb, checkpoint_path, device, img_size,
                         confidence, min_wall_area, min_door_area,
                         min_window_area, timeout=120):
    """Run U-Net in a subprocess so crashes don't kill the main process.

    Uses temp files for IPC instead of multiprocessing.Manager (which
    spawns its own subprocess and fails on Windows with rich imports).

    Returns the result dict or None on failure/crash/timeout.
    """
    import multiprocessing as mp
    import pickle
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="unet_")
    img_path = os.path.join(tmp_dir, "input.npy")
    result_path = os.path.join(tmp_dir, "result.pkl")

    try:
        np.save(img_path, img_rgb)

        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_unet_worker,
            args=(img_path, result_path, checkpoint_path, device, img_size,
                  confidence, min_wall_area, min_door_area, min_window_area),
        )
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            logger.warning("U-Net subprocess timed out after %ds, killing", timeout)
            proc.kill()
            proc.join(5)
            return None

        if proc.exitcode != 0:
            logger.warning("U-Net subprocess crashed with exit code %s "
                           "(0x%08X) — falling back to color pipeline",
                           proc.exitcode,
                           proc.exitcode & 0xFFFFFFFF if proc.exitcode < 0
                           else proc.exitcode)
            return None

        if not os.path.exists(result_path):
            logger.warning("U-Net subprocess returned no results")
            return None

        with open(result_path, "rb") as f:
            result = pickle.load(f)

        if "error" in result:
            logger.warning("U-Net subprocess error: %s", result["error"])
            return None

        return result

    except Exception as e:
        logger.warning("U-Net subprocess setup failed: %s", e)
        return None
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# PIPELINE
# ============================================================

class FloorplanPipeline:
    """
    Clean, linear pipeline for floorplan processing.

    Each stage runs once and passes results forward.
    No redundant detection calls. Explicit error handling per stage.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

        # Stage 1: Wall detection
        self.wall_detector = ColorWallDetector(self.config)

        # Stage 2: Opening detection
        self.opening_detector = OpeningDetectionPipeline(self.config)

        # Stage 3: Room measurement (lazy init, requires OCR)
        self._room_analyzer: Optional[Any] = None

        # Stage 4: Export
        self.exporter = SH3DExporter(self.config)

        # Stage 5: Visualization
        self.visualizer = FloorplanVisualizer(self.config.pixels_to_cm)

        # YOLO detector (shared with opening pipeline)
        self._yolo: Optional[Any] = None

        self._initialized = False

    def initialize(self) -> None:
        """
        Load all models (YOLO, OCR).

        Call this once before processing images to avoid reloading
        models on each call.
        """
        if self._initialized:
            return

        # Initialize OCR
        ocr_model = self._init_ocr()

        # Initialize wall detector with shared OCR model
        self.wall_detector.initialize(ocr_model=ocr_model)

        # Initialize YOLO
        if Path(self.config.yolo_weights_path).exists():
            try:
                from detection.furnituredetector import YOLODoorWindowDetector
                self._yolo = YOLODoorWindowDetector(
                    model_path=self.config.yolo_weights_path,
                    conf_threshold=self.config.yolo_confidence,
                    iou_threshold=self.config.yolo_iou_threshold,
                    device=self.config.yolo_device,
                    max_door_area_fraction=self.config.yolo_max_area_fraction,
                )
                logger.info("YOLO model loaded from %s", self.config.yolo_weights_path)
            except Exception as e:
                logger.warning("YOLO initialization failed (non-fatal): %s", e)
        else:
            logger.warning("YOLO weights not found at %s", self.config.yolo_weights_path)

        # Initialize opening detector with YOLO
        self.opening_detector.initialize(yolo_model=self._yolo)

        # Initialize room analyzer
        if self.config.enable_room_measurement:
            try:
                from detection.space import RoomSizeAnalyzer
                self._room_analyzer = RoomSizeAnalyzer(
                    edge_distance_ratio=0.15,
                )
                # Reuse OCR model to avoid loading it twice
                if ocr_model is not None:
                    self._room_analyzer.ocr_model = ocr_model
            except Exception as e:
                logger.warning("Room analyzer initialization failed: %s", e)

        self._initialized = True
        logger.info("Pipeline initialized")

    def process(
        self,
        image_path: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Process a single floorplan image through all pipeline stages.

        Args:
            image_path: Path to input image file
            output_dir: Directory for output files
            progress_callback: Optional callback ``f(stage_name: str)``
                called at the start of each pipeline stage so external
                progress bars can be updated in real time.

        Returns:
            PipelineResult with all detection results and output paths
        """
        def _notify(stage: str) -> None:
            if progress_callback is not None:
                progress_callback(stage)
        if not self._initialized:
            self.initialize()

        base_name = Path(image_path).stem
        os.makedirs(output_dir, exist_ok=True)

        result = PipelineResult()

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        # ── Pre-stage 0: Deskew — snap lines to X/Y axis ─────────────
        img = _deskew_image(img)

        # ── Pre-stage: OCR text erasure before wall detection ─────────
        # Run OCR in both orientations, collect ALL text bboxes, then
        # erase them from the image so text doesn't pollute wall detection.
        _notify("OCR text collection")
        logger.info("[%s] Pre-stage: OCR text collection & erasure", base_name)

        # Normal orientation OCR
        ocr_text_labels_early = self._get_ocr_text_labels(img)

        # Rotated (CW) OCR — detect vertical wall-length labels
        # Build img_clean (single copy) — erase normal-OCR bboxes, then
        # rotate for vertical label OCR, then erase rotated bboxes too.
        # This replaces two separate img.copy() calls with one.
        h_img, w_img = img.shape[:2]
        img_clean = img.copy()
        pad = 3
        for item in (ocr_text_labels_early or []):
            if len(item) < 5:
                continue
            rx1, ry1 = int(item[1]), int(item[2])
            rx2, ry2 = int(item[3]), int(item[4])
            cv2.rectangle(img_clean,
                          (max(0, rx1 - pad), max(0, ry1 - pad)),
                          (min(w_img, rx2 + pad), min(h_img, ry2 + pad)),
                          (255, 255, 255), -1)
        # Rotate the already-erased image for vertical label detection
        rotated_img = cv2.rotate(img_clean, cv2.ROTATE_90_CLOCKWISE)
        rotated_ocr_labels = self._get_ocr_text_labels(rotated_img)
        del rotated_img   # free ~75MB

        # Un-rotate rotated bbox coords back to original image space
        rotated_bboxes_orig = []  # (x1, y1, x2, y2) in original coords
        for item in (rotated_ocr_labels or []):
            if len(item) < 5:
                continue
            rx1, ry1 = int(item[1]), int(item[2])
            rx2, ry2 = int(item[3]), int(item[4])
            # CW inverse: rotated (rx, ry) -> original (ry, h-1-rx)
            ox1 = min(ry1, ry2)
            oy1 = h_img - 1 - max(rx1, rx2)
            ox2 = max(ry1, ry2)
            oy2 = h_img - 1 - min(rx1, rx2)
            rotated_bboxes_orig.append((ox1, oy1, ox2, oy2))

        # Erase rotated text regions on img_clean too
        for (ox1, oy1, ox2, oy2) in rotated_bboxes_orig:
            cv2.rectangle(img_clean,
                          (max(0, ox1 - pad), max(0, oy1 - pad)),
                          (min(w_img, ox2 + pad), min(h_img, oy2 + pad)),
                          (255, 255, 255), -1)

        n_erased = len([i for i in (ocr_text_labels_early or []) if len(i) >= 5])
        n_erased += len(rotated_bboxes_orig)
        logger.info("[%s] Erased %d text regions before wall detection",
                    base_name, n_erased)

        # ── Stage 0: Try U-Net → wall mask + door/window bboxes ─────────
        _notify("U-Net detection")
        unet_detection = self._try_unet_detection(img_clean)
        raw_rectangles = None
        unet_door_bboxes: list = []
        unet_window_bboxes: list = []
        used_unet_grey = False
        has_unet_openings = False

        if unet_detection is not None:
            unet_wall_mask, unet_door_bboxes, unet_window_bboxes = unet_detection

            # ── Solid vs hollow wall check ────────────────────────────
            # Solid walls: the original image is already dark where the
            # U-Net predicts walls.  Hollow walls: the original image is
            # white (gap between double outline lines) where U-Net predicts.
            # If ≥ solid_wall_threshold of the mask pixels are non-white,
            # the walls are solid → skip grey pipeline, use color detection.
            WHITE_THRESH = 240  # pixel brightness considered "white"
            gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask_px = unet_wall_mask > 0
            total_mask_px = int(np.sum(mask_px))
            if total_mask_px > 0:
                non_white_px = int(np.sum(gray_orig[mask_px] < WHITE_THRESH))
                solid_ratio = non_white_px / total_mask_px
            else:
                solid_ratio = 0.0
            del gray_orig

            is_solid = solid_ratio >= self.config.solid_wall_threshold
            logger.info(
                "[%s] Wall-mask non-white ratio: %.1f%% (threshold %.0f%%) → %s",
                base_name, solid_ratio * 100,
                self.config.solid_wall_threshold * 100,
                "solid" if is_solid else "hollow",
            )

            if is_solid:
                logger.info(
                    "[%s] Solid-wall image — skipping grey pipeline, "
                    "using direct color detection",
                    base_name,
                )
                # Null out unet_detection to skip grey wall pipeline,
                # but keep door/window bboxes for opening detection.
                has_unet_openings = bool(unet_door_bboxes or unet_window_bboxes)
                unet_detection = None  # fall through to color path below

        if unet_detection is not None:
            unet_wall_mask, unet_door_bboxes, unet_window_bboxes = unet_detection

            # ── Stage 1 (U-Net path): enclosed regions → grey image → color detector
            _notify("detecting walls (U-Net hybrid)")
            logger.info("[%s] Stage 1: U-Net hybrid wall detection", base_name)

            num_labels, labels = _build_enclosed_regions(img)
            logger.info("[%s] Found %d enclosed regions", base_name, num_labels - 1)

            grey_img = build_grey_wall_image(
                original_bgr=img,
                labels=labels,
                num_labels=num_labels,
                wall_mask=unet_wall_mask,
                coverage_threshold=0.50,
            )

            # Add unenclosed wall rects (walls the U-Net sees but
            # enclosed-region analysis missed due to outline gaps)
            unenclosed_rects = find_unenclosed_wall_rects(
                img, unet_wall_mask,
                min_area=200, merge_gap=15, min_merged_area=1000)
            if unenclosed_rects:
                grey_img = paint_unenclosed_walls_grey(
                    grey_img, unenclosed_rects, grey_value=160)
                logger.info("[%s] Added %d unenclosed wall rects to grey image",
                            base_name, len(unenclosed_rects))

            grey_path = os.path.join(output_dir, f"{base_name}_grey_walls.png")
            cv2.imwrite(grey_path, grey_img)
            logger.info("[%s] Grey wall image saved: %s", base_name, grey_path)

            debug_path = os.path.join(output_dir, f"{base_name}_enclosed_regions.png")
            debug_enclosed_regions(
                original_img=img,
                wall_mask=unet_wall_mask,
                output_path=debug_path,
            )

            del img_clean  # free ~75MB

            GREY_WALL_RGB = (160, 160, 160)
            detection = self.wall_detector.detect_with_color(grey_img, GREY_WALL_RGB)
            raw_rectangles = self.wall_detector._last_rectangles
            del grey_img

            used_unet_grey = True
            logger.info("[%s] U-Net hybrid: %d walls from grey image",
                        base_name, len(detection.walls))

            result.walls = detection.walls
            result.wall_mask = detection.wall_mask
            result.outline_mask = detection.outline_mask

        if not used_unet_grey:
            # ── Stage 1 (fallback): color-based wall detection ───────────
            _notify("detecting walls")
            logger.info("[%s] Stage 1: Color-based wall detection", base_name)
            detection, raw_rectangles = self._run_wall_detection(img_clean)
            del img_clean  # free ~75MB

            result.walls = detection.walls
            result.wall_mask = detection.wall_mask
            result.outline_mask = detection.outline_mask

        # ── Stage 2: Opening Detection (UNIFIED: gap + U-Net with deduplication) ─────
        _notify("detecting openings")
        logger.info("[%s] Stage 2: Opening detection (gap priority + U-Net supplement)", base_name)

        ocr_bboxes = self._get_ocr_bboxes(img)

        # ALWAYS run traditional gap/algorithmic + YOLO detection (gap doors take priority)
        yolo_results = None
        if self._yolo is not None:
            try:
                yolo_results = self._yolo.detect(img)
                result.raw_yolo_results = yolo_results
            except Exception as e:
                logger.warning("YOLO detection failed: %s", e)

        pixel_scale = self.config.pixels_to_m

        openings, door_arcs = self._run_opening_detection(
            img=img,
            walls=result.walls,
            wall_mask=result.wall_mask,
            outline_mask=result.outline_mask,
            wall_rectangles=raw_rectangles,
            yolo_results=yolo_results,
            ocr_bboxes=ocr_bboxes,
            pixel_scale=pixel_scale,
            base_name=base_name,
        )

        # If U-Net doors/windows available, add non-overlapping ones (gap doors priority)
        if unet_door_bboxes or unet_window_bboxes:
            unet_openings = self._unet_bboxes_to_openings(
                unet_door_bboxes, unet_window_bboxes
            )
            # Priority openings for deduplication:
            # - U-Net DOOR deduped against traditional GAP or DOOR
            # - U-Net WINDOW deduped against traditional WINDOW
            priority_for_door = [o for o in openings if o.opening_type in (OpeningType.GAP, OpeningType.DOOR)]
            priority_for_window = [o for o in openings if o.opening_type == OpeningType.WINDOW]

            filtered_unet_openings = []
            overlapping_count = 0
            for uo in unet_openings:
                if uo.opening_type == OpeningType.DOOR:
                    priority_list = priority_for_door
                else:  # WINDOW
                    priority_list = priority_for_window
                overlaps = any(
                    self._bbox_iou(uo.bbox, po.bbox) > 0.3
                    for po in priority_list
                )
                if overlaps:
                    overlapping_count += 1
                else:
                    filtered_unet_openings.append(uo)

            if filtered_unet_openings:
                openings.extend(filtered_unet_openings)
                logger.info(
                    "[%s] U-Net dedup: kept %d/%d U-Net openings (removed %d overlapping with gap doors)",
                    base_name, len(filtered_unet_openings), len(unet_openings), overlapping_count
                )
            else:
                logger.info("[%s] All U-Net openings were overlapping with gap doors — discarded", base_name)

        # Now assign to result (door_arcs always from gap detection)
        result.openings = openings
        result.door_arcs = door_arcs

        # ── Stage 3: Room Measurement + Scale Calibration (primary) ───────
        _notify("OCR / scale calibration")
        logger.info("[%s] Stage 3: Room measurement", base_name)
        measurements, ocr_labels, pixels_to_cm = self._run_room_measurement(
            image_path=image_path,
            img=img,
            walls=result.walls,
            outline_mask=result.outline_mask,
            output_dir=output_dir,
        )

        result.raw_measurements = measurements
        result.ocr_labels = ocr_labels
        result.pixels_to_cm = pixels_to_cm

        if measurements:
            ps = measurements.pixel_scale
            if ps and ps > 0:
                pixels_to_cm = ps * 100.0
                if 0.1 < pixels_to_cm < 20.0:
                    result.pixel_scale = ps
                    result.pixels_to_cm = pixels_to_cm
                    self.config = self.config.copy_with(pixels_to_cm=pixels_to_cm)
                    self.exporter.update_scale(pixels_to_cm)
                    self.visualizer.pixels_to_cm = pixels_to_cm
                else:
                    logger.warning("Room-area scale out of bounds (%.4f), ignoring",
                                   pixels_to_cm)

        # Reuse OCR labels collected in pre-stage (no re-run needed).
        logger.info("[%s] Stage 3b: OCR text label collection (reusing pre-stage)",
                    base_name)
        ocr_text_labels = ocr_text_labels_early
        if ocr_text_labels:
            ocr_labels = ocr_text_labels
            result.ocr_labels = ocr_labels

        # ── Stage 3b2: Wall-length scale calibration (rotated labels) ────
        # Uses rotated OCR labels collected in pre-stage (no re-run).
        logger.info("[%s] Stage 3b2: Wall-length scale calibration", base_name)
        wl_ptcm = _calibrate_scale_from_wall_lengths(
            img=img,
            walls=result.walls,
            normal_labels=ocr_text_labels,
            rotated_labels=rotated_ocr_labels,
            debug_dir=output_dir,
        )
        if wl_ptcm is not None and 0.1 < wl_ptcm < 20.0:
            logger.info("[%s] Wall-length calibration: pixels_to_cm %.4f -> %.4f",
                        base_name, result.pixels_to_cm, wl_ptcm)
            result.pixels_to_cm = wl_ptcm
            result.pixel_scale = wl_ptcm / 100.0
            self.config = self.config.copy_with(pixels_to_cm=wl_ptcm)
            self.exporter.update_scale(wl_ptcm)
            self.visualizer.pixels_to_cm = wl_ptcm

        # ── Stage 3c: Wall cleanup ────────────────────────────────────
        logger.info("[%s] Stage 3c: Wall cleanup", base_name)
        result.walls = cleanup_walls(result.walls)

        # ── Stage 4: SH3D Export ───────────────────────────────────────
        _notify("exporting SH3D")
        logger.info("[%s] Stage 4: SH3D export", base_name)
        sh3d_path = os.path.join(output_dir, f"{base_name}.sh3d")
        debug_img_path = os.path.join(output_dir, f"{base_name}_debug.png")

        rooms = self._run_export(
            walls=result.walls,
            openings=result.openings,
            output_path=sh3d_path,
            original_image=img,
            ocr_labels=ocr_labels,
            debug_image_path=debug_img_path,
            wall_mask=result.wall_mask,
        )

        result.rooms = rooms
        result.sh3d_path = sh3d_path if os.path.exists(sh3d_path) else None

        # ── Stage 4b: Secondary scale calibration from room polygons ───────
        # Stacked-label rule (highest priority): if a room contains exactly one
        # integer numeric label with a smaller decimal label directly above it
        # and no other integers in the room, the integer is the room area in m².
        # Falls back to standard room-polygon calibration.
        if result.rooms and ocr_text_labels:
            logger.info("[%s] Stage 4b: Scale calibration from rooms", base_name)
            new_sf = self._calibrate_scale_from_stacked_labels(
                ocr_text_labels=ocr_text_labels,
                rooms=result.rooms,
                pixels_to_cm=result.pixels_to_cm,
                current_scale_factor=self.config.sh3d_scale_factor,
            )
            if new_sf is None:
                new_sf = self._calibrate_scale_from_rooms(
                    rooms=result.rooms,
                    ocr_text_labels=ocr_text_labels,
                    pixels_to_cm=result.pixels_to_cm,
                    current_scale_factor=self.config.sh3d_scale_factor,
                )
            if new_sf is not None and abs(new_sf - self.config.sh3d_scale_factor) > 1e-4:
                # Update the XML generator's scale_factor
                inner_exp = getattr(self.exporter, '_inner', None)
                if inner_exp is not None:
                    inner_exp.xml_generator.scale_factor = new_sf
                self.config = self.config.copy_with(sh3d_scale_factor=new_sf)

                # Remove the first export and redo with corrected scale
                try:
                    os.remove(sh3d_path)
                except OSError:
                    pass
                rooms = self._run_export(
                    walls=result.walls,
                    openings=result.openings,
                    output_path=sh3d_path,
                    original_image=img,
                    ocr_labels=ocr_labels,
                    debug_image_path=debug_img_path,
                    wall_mask=result.wall_mask,
                )
                result.rooms = rooms
                result.sh3d_path = sh3d_path if os.path.exists(sh3d_path) else None

        # ── Stage 4c: Room area validation ────────────────────────────────
        if result.rooms and ocr_text_labels:
            result.room_area_reports = _validate_room_areas(
                rooms=result.rooms,
                ocr_text_labels=ocr_text_labels,
                pixels_to_cm=result.pixels_to_cm,
                scale_factor=self.config.sh3d_scale_factor,
                tolerance_ratio=self.config.room_area_tolerance_ratio,
            )

        # ── Stage 5: Visualization ─────────────────────────────────────
        _notify("rendering")
        logger.info("[%s] Stage 5: Visualization", base_name)
        summary_path = os.path.join(output_dir, f"{base_name}_summary.png")
        rooms_ocr_path = os.path.join(output_dir, f"{base_name}_rooms_ocr.png")

        # Build wall segments for overlay from result.walls (unified WallSegments).
        # raw_rectangles may be None (e.g. U-Net path), but result.walls is
        # always populated.  _walls_to_stubs converts WallSegment → stub with
        # flat .x1/.y1/.x2/.y2 attrs that the visualizer expects.
        wall_stubs = self._walls_to_stubs(result.walls)

        self._render_summary(
            img=img,
            wall_stubs=wall_stubs,
            openings=result.openings,
            door_arcs=result.door_arcs,
            wall_mask=result.wall_mask,
            outline_mask=result.outline_mask,
            measurements=measurements,
            output_path=summary_path,
        )

        if result.rooms:
            self._render_rooms_ocr(
                img=img,
                rooms=result.rooms,
                openings=[],  # Empty — legacy rooms use legacy opening format
                ocr_labels=ocr_labels,
                door_arcs=result.door_arcs,
                output_path=rooms_ocr_path,
            )

        result.summary_path = summary_path if os.path.exists(summary_path) else None
        result.rooms_ocr_path = rooms_ocr_path if os.path.exists(rooms_ocr_path) else None

        # ── Cleanup: free heavy intermediate data ─────────────────────
        result.wall_mask = None
        result.outline_mask = None
        result.raw_yolo_results = None
        result.raw_measurements = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info(
            "[%s] Done: %d walls, %d doors, %d windows, %d rooms",
            base_name, result.n_walls, result.n_doors,
            result.n_windows, len(result.rooms),
        )

        return result

    # ============================================================
    # PRIVATE STAGE IMPLEMENTATIONS
    # ============================================================

    def _try_unet_detection(
        self, img: np.ndarray
    ) -> Optional[tuple]:
        """Try U-Net detection, return (wall_mask, door_bboxes, window_bboxes).

        wall_mask is binary uint8 (0/255) used for enclosed-region voting.
        door_bboxes / window_bboxes are lists of (x1,y1,x2,y2) tuples from
        the U-Net post-processing pipeline — identical to the old U-Net branch.

        Returns None if U-Net is unavailable or produces an empty mask.
        U-Net inference runs in a subprocess to isolate potential crashes.
        """
        # enable_unet previously gated U-Net completely; now U-Net always
        # attempts so both detection methods are always combined.
        # It still fails gracefully when checkpoint / deps are absent.
        ckpt = self.config.unet_checkpoint_path
        if not Path(ckpt).exists():
            logger.info("U-Net checkpoint not found at %s, skipping", ckpt)
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            from detect_unet import run_unet_subprocess as _detect_unet_subprocess
        except ImportError as exc:
            logger.info("U-Net dependencies not available (%s), skipping", exc)
            return None
        unet_result = _detect_unet_subprocess(
            image_rgb=img_rgb,
            checkpoint_path=ckpt,
            device=self.config.unet_device,
            img_size=self.config.unet_img_size,
            confidence=self.config.unet_confidence,
            min_wall_area=self.config.unet_min_wall_area,
            min_door_area=self.config.unet_min_door_area,
            min_window_area=self.config.unet_min_window_area,
        )

        if unet_result is None:
            return None

        wall_mask = unet_result.get("wall_mask")
        if wall_mask is None or not wall_mask.any():
            logger.warning("U-Net returned empty wall mask")
            return None

        wall_mask = (wall_mask > 0).astype(np.uint8) * 255
        door_bboxes = unet_result.get("door_bboxes", [])
        window_bboxes = unet_result.get("window_bboxes", [])

        logger.info(
            "U-Net: %d wall px, %d doors, %d windows",
            int(np.sum(wall_mask > 0)), len(door_bboxes), len(window_bboxes),
        )
        return wall_mask, door_bboxes, window_bboxes

    @staticmethod
    def _unet_bboxes_to_walls(
        wall_bboxes: list,
    ) -> List[WallSegment]:
        """Convert U-Net wall bboxes to unified WallSegment list."""
        walls = []
        for i, (x1, y1, x2, y2) in enumerate(wall_bboxes):
            w = x2 - x1
            h = y2 - y1
            thickness = float(min(w, h))
            walls.append(WallSegment(
                id=f"unet_wall_{i}",
                bbox=BBox(x1=float(x1), y1=float(y1),
                          x2=float(x2), y2=float(y2)),
                thickness=thickness,
                is_structural=True,
                is_outer=False,
            ))
        return walls

    @staticmethod
    def _unet_bboxes_to_openings(
        door_bboxes: list,
        window_bboxes: list,
    ) -> List[Opening]:
        """Convert U-Net door/window bboxes to unified Opening list."""
        openings = []
        for x1, y1, x2, y2 in door_bboxes:
            openings.append(Opening(
                bbox=BBox(x1=float(x1), y1=float(y1),
                          x2=float(x2), y2=float(y2)),
                opening_type=OpeningType.DOOR,
                confidence=1.0,
                source="unet",
            ))
        for x1, y1, x2, y2 in window_bboxes:
            openings.append(Opening(
                bbox=BBox(x1=float(x1), y1=float(y1),
                          x2=float(x2), y2=float(y2)),
                opening_type=OpeningType.WINDOW,
                confidence=1.0,
                source="unet",
            ))
        return openings

    @staticmethod
    def _bbox_iou(bbox_a: BBox, bbox_b: BBox) -> float:
        """Compute Intersection-over-Union (IoU) between two BBox objects.
        Returns 0.0 if no overlap.
        """
        x1 = max(bbox_a.x1, bbox_b.x1)
        y1 = max(bbox_a.y1, bbox_b.y1)
        x2 = min(bbox_a.x2, bbox_b.x2)
        y2 = min(bbox_a.y2, bbox_b.y2)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (bbox_a.x2 - bbox_a.x1) * (bbox_a.y2 - bbox_a.y1)
        area_b = (bbox_b.x2 - bbox_b.x1) * (bbox_b.y2 - bbox_b.y1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _run_wall_detection(self, img: np.ndarray) -> tuple:
        """
        Run wall detection and return (DetectionResult, raw_rectangles).

        Calls the adapter's detect() method so that auto-wall-color detection
        and wider filter tolerances are applied when cfg.auto_wall_color=True.
        """
        try:
            # CRITICAL: Call detect() not inner.process() directly.
            # This ensures auto-color-detection is applied.
            detection = self.wall_detector.detect(img, config=self.config)

            # Retrieve cached rectangles for downstream use
            rectangles = self.wall_detector._last_rectangles

            return detection, rectangles

        except Exception as e:
            logger.error("Wall detection failed: %s", e, exc_info=True)
            h, w = img.shape[:2]
            return DetectionResult(
                walls=[],
                wall_mask=np.zeros((h, w), dtype=np.uint8),
                outline_mask=np.zeros((h, w), dtype=np.uint8),
            ), []

    def _run_opening_detection(
        self,
        img: np.ndarray,
        walls: List[WallSegment],
        wall_mask: Optional[np.ndarray],
        outline_mask: Optional[np.ndarray],
        wall_rectangles: List,
        yolo_results: Optional[Any],
        ocr_bboxes: Optional[List],
        pixel_scale: float,
        base_name: str,
    ) -> Tuple[List[Opening], List[Any]]:
        """
        Run opening detection and door arc detection.

        Returns (openings, door_arcs)
        """
        openings: List[Opening] = []
        door_arcs: List[Any] = []

        effective_mask = outline_mask if outline_mask is not None else wall_mask

        try:
            openings = self.opening_detector.detect(
                image=img,
                walls=walls,
                wall_mask=effective_mask,
                wall_rectangles=wall_rectangles,
                yolo_results=yolo_results,
                ocr_bboxes=ocr_bboxes,
                pixel_scale=pixel_scale,
                base_name=base_name,
            )
        except Exception as e:
            logger.warning("Opening detection failed (non-fatal): %s", e, exc_info=True)

        # Run arc detector separately (returns DetectedDoorArc, not in openings list)
        if self.config.enable_door_arcs and self.opening_detector._arc_detector is not None:
            try:
                from detection.gap import Opening as GapOpening
                geo_gaps = [
                    o for o in openings
                    if o.opening_type == OpeningType.GAP
                ]
                raw_gaps = []
                for i, g in enumerate(geo_gaps):
                    raw_gaps.append(GapOpening(
                        id=i,
                        x1=int(g.bbox.x1), y1=int(g.bbox.y1),
                        x2=int(g.bbox.x2), y2=int(g.bbox.y2),
                        width=int(g.bbox.width), height=int(g.bbox.height),
                        area=int(g.bbox.area),
                        opening_type=g.opening_type.value,
                        confidence=g.confidence,
                        source_rects=[], yolo_matches=[],
                    ))

                door_arcs = self.opening_detector._arc_detector.detect(
                    img=img,
                    wall_outline_mask=effective_mask,
                    candidate_gaps=raw_gaps,
                    yolo_doors=yolo_results.doors if yolo_results else None,
                    pixel_scale=pixel_scale,
                    ocr_boxes=ocr_bboxes,
                    base_name=base_name,
                )
            except Exception as e:
                logger.warning("Door arc extraction failed (non-fatal): %s", e)

        return openings, door_arcs

    def _get_ocr_bboxes(self, img: np.ndarray) -> List:
        """Get OCR text bounding boxes from the wall detector's OCR model."""
        try:
            inner = self.wall_detector.get_inner()
            if hasattr(inner, 'get_ocr_bboxes'):
                return inner.get_ocr_bboxes(img)
        except Exception:
            pass
        return []

    def _run_room_measurement(
        self,
        image_path: str,
        img: np.ndarray,
        walls: List[WallSegment],
        outline_mask: Optional[np.ndarray],
        output_dir: str,
    ) -> Tuple[Optional[Any], List, float]:
        """
        Run OCR-based room measurement.

        Returns (measurements, ocr_labels, pixels_to_cm)
        """
        pixels_to_cm = self.config.pixels_to_cm
        ocr_labels: List = []
        measurements = None

        if self._room_analyzer is None or not walls:
            return measurements, ocr_labels, pixels_to_cm

        try:
            segments_dicts = []
            for w in walls:
                # Convert WallSegment centerline to the dict format expected by RoomSizeAnalyzer
                cl = w.centerline
                segments_dicts.append({
                    'id': hash(w.id) % 100000,
                    'x1': cl[0][0], 'y1': cl[0][1],
                    'x2': cl[1][0], 'y2': cl[1][1],
                    'is_outer': w.is_outer,
                })

            measurements = self._room_analyzer.process(
                image_path,
                segments_dicts,
                output_dir,
                wall_mask=outline_mask,
                original_image=img,
            )

            if measurements:
                # Extract OCR labels for visualization as (text, x1, y1, x2, y2) tuples
                ocr_labels = [
                    (r['text'], *r['bbox'])
                    for r in (measurements.room_areas or [])
                ]

        except Exception as e:
            logger.warning("Room measurement failed (non-fatal): %s", e, exc_info=True)

        return measurements, ocr_labels, pixels_to_cm

    def _run_export(
        self,
        walls: List[WallSegment],
        openings: List[Opening],
        output_path: str,
        original_image: Optional[np.ndarray],
        ocr_labels: List,
        debug_image_path: str,
        wall_mask: Optional[np.ndarray],
    ) -> List[Room]:
        """Run SH3D export and return detected rooms."""
        try:
            return self.exporter.export(
                walls=walls,
                openings=openings,
                output_path=output_path,
                original_image=original_image,
                ocr_labels=ocr_labels if ocr_labels else None,
                debug_image_path=debug_image_path,
                wall_mask=wall_mask,
            )
        except Exception as e:
            logger.error("SH3D export failed: %s", e, exc_info=True)
            return []

    def _render_summary(
        self,
        img: np.ndarray,
        wall_stubs: List,
        openings: List[Opening],
        door_arcs: List,
        wall_mask: Optional[np.ndarray],
        outline_mask: Optional[np.ndarray],
        measurements: Optional[Any],
        output_path: str,
    ) -> None:
        """Render 2x3 debug summary image."""
        try:
            n_doors = sum(1 for o in openings if o.opening_type == OpeningType.DOOR)
            n_windows = sum(1 for o in openings if o.opening_type == OpeningType.WINDOW)

            self.visualizer.render_summary(
                img_bgr=img,
                wall_segments=wall_stubs,
                openings=openings,
                door_arcs=door_arcs,
                wall_mask=wall_mask,
                outline_mask=outline_mask,
                measurements=measurements,
                output_path=output_path,
                n_walls=len(wall_stubs),
                n_doors=n_doors,
                n_windows=n_windows,
                n_arcs=len(door_arcs),
            )
        except Exception as e:
            logger.warning("Summary rendering failed (non-fatal): %s", e, exc_info=True)

    def _render_rooms_ocr(
        self,
        img: np.ndarray,
        rooms: List[Room],
        openings: List,
        ocr_labels: List,
        door_arcs: List,
        output_path: str,
    ) -> None:
        """Render room + OCR overlay."""
        try:
            self.visualizer.render_room_ocr_overlay(
                img_bgr=img,
                rooms=rooms,
                openings=openings,
                ocr_labels=ocr_labels,
                output_path=output_path,
                door_arcs=door_arcs,
            )
        except Exception as e:
            logger.warning("Room OCR rendering failed (non-fatal): %s", e, exc_info=True)

    @staticmethod
    def _walls_to_stubs(walls: List[WallSegment]) -> List:
        """Convert unified WallSegment list to stubs for visualization.

        The visualizer expects flat .x1/.y1/.x2/.y2 attributes on each
        object.  WallSegment stores them under .bbox, so we create thin
        wrappers.
        """
        class _Stub:
            __slots__ = ('x1', 'y1', 'x2', 'y2')
        stubs = []
        for w in walls:
            s = _Stub()
            s.x1 = w.bbox.x1
            s.y1 = w.bbox.y1
            s.x2 = w.bbox.x2
            s.y2 = w.bbox.y2
            stubs.append(s)
        return stubs

    def _rectangles_to_stubs(self, rectangles: List) -> List:
        """
        Convert WallRectangle list to stub objects for visualization.

        Returns objects with .x1, .y1, .x2, .y2 attributes.
        """
        # WallRectangle already has these attributes directly
        return rectangles

    def _get_ocr_text_labels(self, img: np.ndarray) -> List[Tuple]:
        """
        Return all OCR words with their pixel bounding boxes.

        Format: List of (text, x1, y1, x2, y2).
        Reuses the OCR model already loaded in the wall detector.
        """
        try:
            inner = self.wall_detector.get_inner()
            if hasattr(inner, 'get_ocr_texts_with_coords'):
                return inner.get_ocr_texts_with_coords(img)
        except Exception as e:
            logger.warning("OCR text label extraction failed (non-fatal): %s", e)
        return []

    def _calibrate_scale_from_stacked_labels(
        self,
        ocr_text_labels: List[Tuple],
        rooms: List[Room],
        pixels_to_cm: float,
        current_scale_factor: float,
    ) -> Optional[float]:
        """
        Stacked-label scale calibration (highest priority).

        Rule: if a room contains exactly one integer label (room number, above)
        with a decimal label directly below it whose value is larger, and no
        other integer labels in the room — treat the decimal below as the room
        area in m² and compute scale_factor with the same formula as
        _calibrate_scale_from_rooms.

        Pattern:   [integer]   ← room number, above, smaller
                   [decimal]   ← room area, below, larger

        Returns the new scale_factor, or None if the pattern is not found.
        """
        if not rooms or not ocr_text_labels:
            return None

        # Parse every OCR label into value + type
        all_labels = []
        for item in ocr_text_labels:
            if len(item) < 5:
                continue
            text = str(item[0]).strip()
            x1, y1, x2, y2 = item[1], item[2], item[3], item[4]
            clean = text.replace(',', '.').replace(' ', '')
            try:
                val = float(clean)
            except ValueError:
                continue
            # Integer: no decimal separator in the original text
            is_int = ('.' not in text and ',' not in text)
            all_labels.append({
                'val': val, 'is_int': is_int,
                'cx': (x1 + x2) / 2.0, 'cy': (y1 + y2) / 2.0,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            })

        if not all_labels:
            return None

        integer_labels = [lb for lb in all_labels if lb['is_int']]
        decimal_labels = [lb for lb in all_labels if not lb['is_int']]

        # For each integer label, find a qualifying decimal label directly below it.
        # stacked_pairs: integer index → decimal label (the area)
        stacked_pairs: dict = {}

        for i, a in enumerate(integer_labels):
            for b in decimal_labels:
                # b must be below a (larger y = lower on image)
                if b['cy'] <= a['cy']:
                    continue
                vert_gap = b['cy'] - a['cy']
                label_h = max(b['y2'] - b['y1'], a['y2'] - a['y1'], 1.0)
                if vert_gap > label_h * 3.5:
                    continue
                # Horizontally aligned
                horiz_dist = abs(a['cx'] - b['cx'])
                label_w = max(b['x2'] - b['x1'], a['x2'] - a['x1'], 1.0)
                if horiz_dist > label_w * 2.0:
                    continue
                # Decimal below must be numerically larger than integer above
                if b['val'] <= a['val']:
                    continue
                # Decimal must be in a sane room-area range
                if not (1.0 <= b['val'] <= 300.0):
                    continue
                stacked_pairs[i] = b
                break

        if not stacked_pairs:
            return None

        m_per_px = pixels_to_cm / 100.0
        ratios: List[float] = []

        for room in rooms:
            pts_px = [
                (p.x / pixels_to_cm, p.y / pixels_to_cm)
                for p in room.points
            ]
            if len(pts_px) < 3:
                continue
            area_px2 = _polygon_area_from_tuples(pts_px)
            if area_px2 < 1.0:
                continue

            # Collect all integer labels inside this room
            ints_in_room = [
                (i, lb) for i, lb in enumerate(integer_labels)
                if _point_in_polygon(lb['cx'], lb['cy'], pts_px)
            ]
            # Rule: exactly one integer in the room, and it must be a stacked one
            if len(ints_in_room) != 1:
                continue
            idx, _ = ints_in_room[0]
            if idx not in stacked_pairs:
                continue

            area_m2 = stacked_pairs[idx]['val']
            ratios.append(math.sqrt(area_px2 * (m_per_px ** 2) / area_m2))

        if not ratios:
            return None

        new_sf = float(np.median(ratios))
        logger.info(
            "Stacked-label scale calibration from %d room(s): sf %.4f -> %.4f",
            len(ratios), current_scale_factor, new_sf,
        )
        return new_sf

    def _calibrate_scale_from_rooms(
        self,
        rooms: List[Room],
        ocr_text_labels: List[Tuple],
        pixels_to_cm: float,
        current_scale_factor: float,
    ) -> Optional[float]:
        """
        Compute a corrected scale_factor by matching detected room polygon
        areas against OCR-detected room area labels placed inside them.

        This is the secondary calibration pass (mirrors calibrate_scale_factor()
        from the legacy main.py).  The primary pass sets pixels_to_cm from the
        total apartment area; this pass adjusts the XML scale_factor so that
        individual room polygons match the area labels printed on the plan.

        The returned value is the median of sqrt(computed_area / ocr_area) over
        all rooms where exactly one area label falls inside the polygon.  When
        pixels_to_cm is already correct this evaluates to ≈ 1.0; when the
        primary calibration was skipped it absorbs the full correction.

        Returns the new scale_factor, or None if calibration cannot proceed.
        """
        if not rooms or not ocr_text_labels:
            return None

        # Parse area values from every OCR word
        parsed: List[Tuple[float, float, float]] = []  # (area_m2, cx, cy)
        for item in ocr_text_labels:
            if len(item) < 5:
                continue
            text, x1, y1, x2, y2 = item[0], item[1], item[2], item[3], item[4]
            area = _parse_ocr_area_m2(str(text))
            if area is not None and 1.0 <= area <= 200.0:
                parsed.append((area, (x1 + x2) / 2.0, (y1 + y2) / 2.0))

        if not parsed:
            return None

        m_per_px = pixels_to_cm / 100.0
        ratios: List[float] = []

        for room in rooms:
            # Convert room points from cm back to pixel space
            pts_px = [
                (p.x / pixels_to_cm, p.y / pixels_to_cm)
                for p in room.points
            ]
            if len(pts_px) < 3:
                continue

            area_px2 = _polygon_area_from_tuples(pts_px)
            if area_px2 < 1.0:
                continue

            # Collect a ratio for every label whose centre lands in this room.
            # Using all matched pairs (not just rooms with exactly one label)
            # gives a more robust median across the whole floor plan.
            for area_val, cx, cy in parsed:
                if _point_in_polygon(cx, cy, pts_px):
                    ratios.append(math.sqrt(area_px2 * (m_per_px ** 2) / area_val))

        if not ratios:
            return None

        new_sf = float(np.median(ratios))
        logger.info(
            "Scale calibration from %d room(s): sf %.4f → %.4f",
            len(ratios), current_scale_factor, new_sf,
        )
        return new_sf

    def _init_ocr(self) -> Optional[Any]:
        """Initialize docTR OCR model."""
        try:
            from doctr.models import ocr_predictor
            model = ocr_predictor(pretrained=True)
            logger.info("OCR model loaded")
            return model
        except Exception as e:
            logger.warning("OCR initialization failed (non-fatal): %s", e)
            return None
