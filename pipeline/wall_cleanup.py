"""Post-detection wall cleanup: drop noise, snap thicknesses, fix alignment."""

from __future__ import annotations

import logging
import math
import re
from typing import List

import numpy as np

from core.models import BBox, WallSegment

logger = logging.getLogger(__name__)


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

