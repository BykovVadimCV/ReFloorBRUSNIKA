"""OCR-driven scale calibration: convert pixel measurements to real-world units."""

from __future__ import annotations

import logging
import math
import re
from typing import List, Tuple

import cv2
import numpy as np

from core.config import PipelineConfig
from core.geometry import iou_xyxy
from core.ocr_utils import (
    is_numeric_ocr_label as _is_numeric_ocr_label,
    parse_area_m2 as _parse_ocr_area_m2,
    parse_length_m as _parse_ocr_length_m,
)
from core.models import WallSegment

logger = logging.getLogger(__name__)


def _calibrate_scale_from_wall_lengths(
    img: np.ndarray,
    walls: List,
    normal_labels: List[Tuple],
    rotated_labels: List[Tuple],
    max_dist: float = 80.0,
    debug_dir: Optional[str] = None,
    rotated_labels_in_image_coords: bool = False,
) -> Optional[float]:
    """Match rotated wall-length labels to walls and compute pixels_to_cm.

    Receives pre-collected OCR labels from both orientations (collected
    in the pre-stage before wall detection).  Filters to NEW numerical
    labels only, un-rotates coordinates, and matches to walls.

    Parameters
    ----------
    rotated_labels_in_image_coords : bool
        When True, ``rotated_labels`` bboxes are already in current image
        coords (caller has un-rotated them) — use bbox centre directly.
        When False (legacy), bboxes are in 90° CW rotated-image coords —
        un-rotate using image height ``h``.
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

        if rotated_labels_in_image_coords:
            # Already in image coords — bbox centre is the label position
            ox = (rx1 + rx2) / 2.0
            oy = (ry1 + ry2) / 2.0
        else:
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


