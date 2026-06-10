"""Fuses YOLO doors, geometric gaps, and morphological swing arcs into canonical doors."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from detection.door_arc import DetectedDoorArc

logger = logging.getLogger("DoorFusion")


def _iou_xyxy(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class FusedDoor:
    """Canonical door produced by DoorFusion. Consumed by the SH3D exporter."""
    id: int
    hinge: Tuple[int, int]            # exact pixel hinge from arc or inferred
    center: Tuple[int, int]
    width_px: int
    wall_normal_deg: float
    direction: str                    # "left" | "right"
    swing_clockwise: bool
    confidence: float                 # 0.0–1.0
    matched_gap_idx: int = -1
    source: str = "arc"               # "arc" | "yolo_gap" | "yolo_only"


class DoorFusion:
    """Combines YOLO doors, gaps, and arcs into FusedDoor objects."""

    def __init__(
        self,
        gap_radius_tolerance: float = 0.12,
        hinge_snap_px: int = 18,
        yolo_iou_threshold: float = 0.15,
        yolo_center_dist_threshold: int = 60,
        perpendicular_angle_tolerance: float = 30.0,
    ):
        self.gap_radius_tolerance = gap_radius_tolerance
        self.hinge_snap_px = hinge_snap_px
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_center_dist_threshold = yolo_center_dist_threshold
        self.perpendicular_angle_tolerance = perpendicular_angle_tolerance

    def fuse(
        self,
        yolo_doors: List[Any],
        gaps: List[Any],
        arcs: List[DetectedDoorArc],
        wall_segments: List[Any],
        rooms: Optional[List[Any]] = None,
    ) -> List[FusedDoor]:
        """Run the 4-phase fusion algorithm and return canonical doors sorted by id."""
        norm_yolo = self._normalise_yolo(yolo_doors)
        norm_gaps = self._normalise_gaps(gaps)

        fused: List[FusedDoor] = []
        used_yolo: set = set()
        used_gap: set = set()
        next_id = 0

        # Phase 1: Tight YOLO ↔ Gap Matching
        yolo_gap_matches: Dict[int, int] = {}
        if norm_yolo and norm_gaps:
            yolo_areas = []
            for yi, y in enumerate(norm_yolo):
                area = (y[2] - y[0]) * (y[3] - y[1])
                yolo_areas.append((area, yi))
            yolo_areas.sort(reverse=True)

            assigned_gaps: set = set()
            for _, yi in yolo_areas:
                ybox = norm_yolo[yi]
                ycx = (ybox[0] + ybox[2]) / 2
                ycy = (ybox[1] + ybox[3]) / 2
                expanded = (ybox[0] - 40, ybox[1] - 40, ybox[2] + 40, ybox[3] + 40)

                best_score = -1.0
                best_gi = -1
                for gi, g in enumerate(norm_gaps):
                    if gi in assigned_gaps:
                        continue
                    gx1, gy1 = g[0], g[1]
                    gx2, gy2 = g[0] + g[2], g[1] + g[3]
                    gap_box = (gx1, gy1, gx2, gy2)

                    iou = _iou_xyxy(expanded, gap_box)
                    gcx = (gx1 + gx2) / 2
                    gcy = (gy1 + gy2) / 2
                    dist = math.hypot(ycx - gcx, ycy - gcy)

                    if iou > self.yolo_iou_threshold or dist < self.yolo_center_dist_threshold:
                        dist_score = max(0.0, 1.0 - dist / 80.0)
                        score = 0.55 * dist_score + 0.45 * iou
                        if score > best_score:
                            best_score = score
                            best_gi = gi

                if best_gi >= 0:
                    yolo_gap_matches[yi] = best_gi
                    assigned_gaps.add(best_gi)

        # Phase 2: Arc Triangulation (primary truth)
        used_arcs: set = set()
        if arcs:
            for arc in arcs:
                matched_gi = -1
                if norm_gaps:
                    for gi, g in enumerate(norm_gaps):
                        gap_length = max(g[2], g[3])
                        if gap_length > 0:
                            radius_diff = abs(arc.radius_px - gap_length) / gap_length
                        else:
                            radius_diff = 1.0

                        if radius_diff >= self.gap_radius_tolerance:
                            continue

                        gx1, gy1 = g[0], g[1]
                        gx2, gy2 = g[0] + g[2], g[1] + g[3]
                        gap_endpoints = [
                            (gx1, gy1), (gx2, gy1), (gx1, gy2), (gx2, gy2),
                            (gx1, (gy1 + gy2) / 2), (gx2, (gy1 + gy2) / 2),
                            ((gx1 + gx2) / 2, gy1), ((gx1 + gx2) / 2, gy2),
                        ]

                        hinge_near = any(
                            math.hypot(arc.hinge[0] - ep[0], arc.hinge[1] - ep[1]) < self.hinge_snap_px
                            for ep in gap_endpoints
                        )
                        if not hinge_near:
                            continue

                        tol = 10
                        if (gx1 - tol <= arc.center[0] <= gx2 + tol and
                                gy1 - tol <= arc.center[1] <= gy2 + tol):
                            matched_gi = gi
                            break

                matched_yi = -1
                if norm_yolo:
                    best_ydist = float('inf')
                    for yi, y in enumerate(norm_yolo):
                        ycx = (y[0] + y[2]) / 2
                        ycy = (y[1] + y[3]) / 2
                        d = math.hypot(arc.center[0] - ycx, arc.center[1] - ycy)
                        if d < best_ydist:
                            best_ydist = d
                            matched_yi = yi

                # Opening geometry. The arc's circle centre is the physical
                # hinge; the wall-attached contour endpoint (arc.hinge) is the
                # far jamb. The wall opening is the segment between them, so
                # its width equals the arc RADIUS (the door leaf), not 2r.
                # When the arc matched a geometric gap, the gap's bbox is the
                # most precise wall-aligned geometry — adopt it outright.
                hinge_pt = (int(arc.center[0]), int(arc.center[1]))
                jamb_pt = (int(arc.hinge[0]), int(arc.hinge[1]))
                if matched_gi >= 0:
                    g = norm_gaps[matched_gi]
                    door_center = (int(g[0] + g[2] / 2), int(g[1] + g[3] / 2))
                    door_width = int(max(g[2], g[3]))
                    wall_normal = 90.0 if g[2] >= g[3] else 0.0
                else:
                    door_center = (
                        int(round((hinge_pt[0] + jamb_pt[0]) / 2)),
                        int(round((hinge_pt[1] + jamb_pt[1]) / 2)),
                    )
                    door_width = int(round(arc.radius_px))
                    wall_normal = arc.wall_normal_deg

                fd = FusedDoor(
                    id=next_id,
                    hinge=hinge_pt,
                    center=door_center,
                    width_px=door_width,
                    wall_normal_deg=wall_normal,
                    direction=arc.direction,
                    swing_clockwise=arc.swing_clockwise,
                    confidence=arc.confidence,
                    matched_gap_idx=matched_gi,
                    source="arc",
                )
                fused.append(fd)
                next_id += 1
                used_arcs.add(arc.id)
                if matched_yi >= 0:
                    used_yolo.add(matched_yi)
                if matched_gi >= 0:
                    used_gap.add(matched_gi)

        # Phase 3: Fallback for doors without arc
        for yi, gi in yolo_gap_matches.items():
            if yi in used_yolo or gi in used_gap:
                continue

            ybox = norm_yolo[yi]
            g = norm_gaps[gi]

            gx1, gy1 = g[0], g[1]
            gx2, gy2 = g[0] + g[2], g[1] + g[3]
            gap_cx = (gx1 + gx2) / 2
            gap_cy = (gy1 + gy2) / 2
            gap_w = g[2]
            gap_h = g[3]

            is_horizontal_gap = gap_w >= gap_h
            if is_horizontal_gap:
                ycx = (ybox[0] + ybox[2]) / 2
                if abs(gx1 - ycx) < abs(gx2 - ycx):
                    hinge = (int(gx1), int(gap_cy))
                else:
                    hinge = (int(gx2), int(gap_cy))
                wall_normal = 90.0 if is_horizontal_gap else 0.0
            else:
                ycy = (ybox[1] + ybox[3]) / 2
                if abs(gy1 - ycy) < abs(gy2 - ycy):
                    hinge = (int(gap_cx), int(gy1))
                else:
                    hinge = (int(gap_cx), int(gy2))
                wall_normal = 0.0

            ycx = (ybox[0] + ybox[2]) / 2
            ycy = (ybox[1] + ybox[3]) / 2
            cross = (gap_cx - hinge[0]) * (ycy - hinge[1]) - (gap_cy - hinge[1]) * (ycx - hinge[0])
            direction = "left" if cross > 0 else "right"
            swing_cw = cross <= 0

            fd = FusedDoor(
                id=next_id,
                hinge=hinge,
                center=(int(gap_cx), int(gap_cy)),
                width_px=int(max(gap_w, gap_h)),
                wall_normal_deg=wall_normal,
                direction=direction,
                swing_clockwise=swing_cw,
                confidence=0.5,
                matched_gap_idx=gi,
                source="yolo_gap",
            )
            fused.append(fd)
            next_id += 1
            used_yolo.add(yi)
            used_gap.add(gi)

        # YOLO-only doors (no gap match, no arc)
        for yi, y in enumerate(norm_yolo):
            if yi in used_yolo:
                continue
            ycx = int((y[0] + y[2]) / 2)
            ycy = int((y[1] + y[3]) / 2)
            yw = int(y[2] - y[0])
            yh = int(y[3] - y[1])
            is_horiz = yw >= yh
            width = max(yw, yh)

            fd = FusedDoor(
                id=next_id,
                hinge=(int(y[0]), ycy) if is_horiz else (ycx, int(y[1])),
                center=(ycx, ycy),
                width_px=width,
                wall_normal_deg=90.0 if is_horiz else 0.0,
                direction="right",
                swing_clockwise=True,
                confidence=0.3,
                matched_gap_idx=-1,
                source="yolo_only",
            )
            fused.append(fd)
            next_id += 1

        # Phase 4: Sort and return
        fused.sort(key=lambda d: d.id)
        logger.info("DoorFusion: %d fused doors (%d from arcs, %d from yolo+gap, %d yolo-only)",
                     len(fused),
                     sum(1 for f in fused if f.source == "arc"),
                     sum(1 for f in fused if f.source == "yolo_gap"),
                     sum(1 for f in fused if f.source == "yolo_only"))
        return fused

    @staticmethod
    def _normalise_gaps(
        candidate_gaps: Optional[List[Any]],
    ) -> List[Tuple[int, int, int, int]]:
        """Convert heterogeneous gap objects to (x, y, w, h) tuples."""
        if not candidate_gaps:
            return []
        result = []
        for g in candidate_gaps:
            if isinstance(g, (list, tuple)) and len(g) >= 4:
                result.append((int(g[0]), int(g[1]), int(g[2]), int(g[3])))
            elif hasattr(g, "x") and hasattr(g, "y") and hasattr(g, "w") and hasattr(g, "h"):
                result.append((int(g.x), int(g.y), int(g.w), int(g.h)))
            elif hasattr(g, "x1") and hasattr(g, "y1") and hasattr(g, "x2") and hasattr(g, "y2"):
                result.append((int(g.x1), int(g.y1), int(g.x2 - g.x1), int(g.y2 - g.y1)))
            elif hasattr(g, "bbox"):
                b = g.bbox
                result.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        return result

    @staticmethod
    def _normalise_yolo(
        yolo_doors: Optional[List[Any]],
    ) -> List[Tuple[int, int, int, int]]:
        """Convert YOLO detection objects to (x1, y1, x2, y2) tuples."""
        if not yolo_doors:
            return []
        result = []
        for d in yolo_doors:
            if isinstance(d, (list, tuple)) and len(d) >= 4:
                result.append((int(d[0]), int(d[1]), int(d[2]), int(d[3])))
            elif hasattr(d, "x1"):
                result.append((int(d.x1), int(d.y1), int(d.x2), int(d.y2)))
            elif hasattr(d, "bbox"):
                b = d.bbox
                result.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
            elif hasattr(d, "xyxy"):
                b = d.xyxy
                result.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        return result
