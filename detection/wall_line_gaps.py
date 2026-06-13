"""Wall-line break detection — robust replacement for pairwise ray casting.

The previous :class:`detection.gap.GapDetector` found openings by casting rays
between *every pair* of wall rectangles, so any empty span between two
unconnected rectangles became a candidate doorway.  Because walls are
rect-decomposed into many pieces, that produced dense, duplicated detections,
and it could not distinguish a genuine break *inside one continuous wall* from
the empty space where a wall simply *ends* (a corridor mouth or T/L junction) —
the latter were wrongly "plugged" with doors.

This detector reframes the problem around reconstructed **wall lines**:

1. Group the axis-aligned wall rectangles into collinear wall lines (the real
   physical wall, not the decomposition pieces).
2. Within a line, the empty intervals *between consecutive wall runs* are the
   only doorway candidates.  An interval is bounded by wall material on both
   sides by construction; the unbounded ends of a line are never candidates, so
   wall terminations no longer masquerade as doorways.
3. Validate each break by real-world width, emptiness (white content), and the
   absence of a perpendicular wall crossing it (a junction, not a passage).
4. Every break gets its host line's thickness, so opening thickness is
   consistent per wall instead of varying with ray geometry.

The public surface (:meth:`WallLineGapDetector.detect_openings`) mirrors
``GapDetector.detect_openings`` so it is a drop-in within
:mod:`detection.openings`, returning :class:`detection.gap.Opening` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from detection.gap import Opening


@dataclass
class _Rect:
    """Normalised axis-aligned wall rectangle used internally."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


class WallLineGapDetector:
    """Detect doorway-sized breaks within continuous, axis-aligned wall lines."""

    def __init__(
        self,
        min_gap_width_m: float = 0.30,
        max_gap_width_m: float = 1.65,
        merge_distance: int = 8,
        band_overlap_ratio: float = 0.5,
        max_non_white_ratio: float = 0.45,
        white_threshold: int = 240,
        run_join_tol: int = 3,
        max_perp_cross_ratio: float = 0.6,
        squareness_ratio: float = 2.0,
        confidence: float = 0.7,
        **_ignored,
    ) -> None:
        """
        Args:
            min_gap_width_m / max_gap_width_m: plausible doorway width range.
            merge_distance: collinear breaks closer than this (px) are merged.
            band_overlap_ratio: min perpendicular overlap (fraction of the
                thinner rect) for two rects to join the same wall line.
            max_non_white_ratio: max fraction of non-white pixels allowed in a
                break ROI before it is rejected as obstructed.
            white_threshold: grey level at/above which a pixel counts as white.
            run_join_tol: rectangles within this px gap along the line are the
                same continuous wall run (no doorway between them).
            max_perp_cross_ratio: if a perpendicular wall fills more than this
                fraction of a break it is treated as a junction and the break is
                cut to its largest free sub-segment.
            squareness_ratio: a rect with major/minor side ratio below this is
                treated as a junction block and considered for *both*
                orientations.
            confidence: confidence assigned to detected breaks.
        """
        self.min_gap_width_m = float(min_gap_width_m)
        self.max_gap_width_m = float(max_gap_width_m)
        self.merge_distance = int(merge_distance)
        self.band_overlap_ratio = float(band_overlap_ratio)
        self.max_non_white_ratio = float(max_non_white_ratio)
        self.white_threshold = int(white_threshold)
        self.run_join_tol = int(run_join_tol)
        self.max_perp_cross_ratio = float(max_perp_cross_ratio)
        self.squareness_ratio = float(squareness_ratio)
        self.confidence = float(confidence)

        # Kept for API compatibility with GapDetector consumers / visualisers.
        self.last_deleted_openings: List[Opening] = []

    # ------------------------------------------------------------------
    # Rectangle normalisation and orientation pools
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(rectangles: Sequence) -> List[_Rect]:
        out: List[_Rect] = []
        for r in rectangles:
            try:
                x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            except AttributeError:
                continue
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            out.append(_Rect(x1, y1, x2, y2))
        return out

    def _orientation_pools(
        self, rects: List[_Rect]
    ) -> Tuple[List[_Rect], List[_Rect]]:
        """Split rects into horizontal-capable and vertical-capable pools.

        Strongly elongated rects join only their own orientation; near-square
        junction blocks join both so they can flank doorways on either axis.
        """
        horiz: List[_Rect] = []
        vert: List[_Rect] = []
        for r in rects:
            w, h = r.w, r.h
            longer, shorter = max(w, h), min(w, h)
            square = longer <= self.squareness_ratio * shorter
            if w >= h or square:
                horiz.append(r)
            if h >= w or square:
                vert.append(r)
        return horiz, vert

    # ------------------------------------------------------------------
    # Wall-line grouping
    # ------------------------------------------------------------------

    def _group_into_lines(
        self, rects: List[_Rect], horizontal: bool
    ) -> List[List[_Rect]]:
        """Cluster rects whose perpendicular bands overlap into wall lines."""
        n = len(rects)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if self._same_band(rects[i], rects[j], horizontal):
                    union(i, j)

        groups: dict = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(rects[i])
        return list(groups.values())

    def _same_band(self, a: _Rect, b: _Rect, horizontal: bool) -> bool:
        """True if a and b sit on the same wall line (perpendicular overlap)."""
        if horizontal:
            lo, hi = max(a.y1, b.y1), min(a.y2, b.y2)
            overlap = hi - lo
            thinner = min(a.h, b.h)
        else:
            lo, hi = max(a.x1, b.x1), min(a.x2, b.x2)
            overlap = hi - lo
            thinner = min(a.w, b.w)
        if overlap <= 0 or thinner <= 0:
            return False
        return overlap / float(thinner) >= self.band_overlap_ratio

    # ------------------------------------------------------------------
    # Break extraction within a line
    # ------------------------------------------------------------------

    def _breaks_in_line(
        self, members: List[_Rect], horizontal: bool
    ) -> List[Tuple[int, int, int, int, float]]:
        """Return (x1, y1, x2, y2, thickness) for each interior break in a line.

        Consecutive rectangles closer than ``run_join_tol`` along the line are
        first merged into continuous wall *runs*; a break is the empty interval
        between two adjacent runs.
        """
        if len(members) < 2:
            return []

        if horizontal:
            members = sorted(members, key=lambda r: r.x1)
            runs = self._merge_runs(members, axis="x")
        else:
            members = sorted(members, key=lambda r: r.y1)
            runs = self._merge_runs(members, axis="y")

        breaks: List[Tuple[int, int, int, int, float]] = []
        for left, right in zip(runs, runs[1:]):
            if horizontal:
                gap_lo, gap_hi = left["x2"], right["x1"]
                if gap_hi - gap_lo <= 0:
                    continue
                # Perpendicular band shared by both flanking runs.
                band_lo = max(left["y1"], right["y1"])
                band_hi = min(left["y2"], right["y2"])
                if band_hi - band_lo <= 0:
                    # Flanks misaligned — fall back to averaged centre band.
                    cy = (left["cy"] + right["cy"]) / 2.0
                    t = min(left["t"], right["t"])
                    band_lo, band_hi = cy - t / 2.0, cy + t / 2.0
                thickness = float(band_hi - band_lo)
                breaks.append(
                    (int(gap_lo), int(band_lo), int(gap_hi), int(band_hi), thickness)
                )
            else:
                gap_lo, gap_hi = left["y2"], right["y1"]
                if gap_hi - gap_lo <= 0:
                    continue
                band_lo = max(left["x1"], right["x1"])
                band_hi = min(left["x2"], right["x2"])
                if band_hi - band_lo <= 0:
                    cx = (left["cx"] + right["cx"]) / 2.0
                    t = min(left["t"], right["t"])
                    band_lo, band_hi = cx - t / 2.0, cx + t / 2.0
                thickness = float(band_hi - band_lo)
                breaks.append(
                    (int(band_lo), int(gap_lo), int(band_hi), int(gap_hi), thickness)
                )
        return breaks

    def _merge_runs(self, members: List[_Rect], axis: str) -> List[dict]:
        """Collapse rects that touch/overlap along ``axis`` into wall runs."""
        runs: List[dict] = []
        for r in members:
            if axis == "x":
                lo, hi = r.x1, r.x2
            else:
                lo, hi = r.y1, r.y2
            if runs and lo - runs[-1]["hi"] <= self.run_join_tol:
                run = runs[-1]
                run["hi"] = max(run["hi"], hi)
                run["x1"] = min(run["x1"], r.x1)
                run["y1"] = min(run["y1"], r.y1)
                run["x2"] = max(run["x2"], r.x2)
                run["y2"] = max(run["y2"], r.y2)
                run["_members"].append(r)
            else:
                runs.append(
                    {
                        "lo": lo,
                        "hi": hi,
                        "x1": r.x1,
                        "y1": r.y1,
                        "x2": r.x2,
                        "y2": r.y2,
                        "_members": [r],
                    }
                )
        # Per-run summary stats used when emitting breaks.
        for run in runs:
            members_ = run["_members"]
            run["cx"] = sum(m.cx for m in members_) / len(members_)
            run["cy"] = sum(m.cy for m in members_) / len(members_)
            if axis == "x":
                run["t"] = float(np.median([m.h for m in members_]))
            else:
                run["t"] = float(np.median([m.w for m in members_]))
        return runs

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _gap_is_empty(self, box: Tuple[int, int, int, int], image: np.ndarray) -> bool:
        """True if the break ROI is predominantly white (unobstructed)."""
        if image is None:
            return True
        if image.ndim == 3:
            h, w = image.shape[:2]
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            h, w = image.shape
            gray = image
        x1 = max(0, box[0])
        y1 = max(0, box[1])
        x2 = min(w, box[2])
        y2 = min(h, box[3])
        if x2 <= x1 or y2 <= y1:
            return False
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        non_white = np.count_nonzero(roi < self.white_threshold)
        return (non_white / roi.size) <= self.max_non_white_ratio

    def _largest_free_segment(
        self,
        gap_lo: int,
        gap_hi: int,
        crossings: List[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Subtract perpendicular crossings; return the widest free interval."""
        if not crossings:
            return (gap_lo, gap_hi)
        cuts = sorted((max(gap_lo, c0), min(gap_hi, c1)) for c0, c1 in crossings)
        free: List[Tuple[int, int]] = []
        cur = gap_lo
        for c0, c1 in cuts:
            if c1 <= c0:
                continue
            if c0 > cur:
                free.append((cur, c0))
            cur = max(cur, c1)
        if cur < gap_hi:
            free.append((cur, gap_hi))
        if not free:
            return None
        return max(free, key=lambda s: s[1] - s[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_openings(
        self,
        rectangles: Sequence,
        yolo_results=None,
        pixel_scale: Optional[float] = None,
        image: Optional[np.ndarray] = None,
        door_window_boxes: Optional[List] = None,
        visualize: bool = False,
        vis_output_path: Optional[str] = None,
        deleted_openings: Optional[List] = None,
    ) -> List[Opening]:
        """Detect doorway breaks within continuous wall lines.

        Signature mirrors ``GapDetector.detect_openings``; ``yolo_results``,
        ``door_window_boxes`` and the visualisation args are accepted for
        compatibility but unused (corroboration/classification happens in
        :mod:`detection.openings`).
        """
        self.last_deleted_openings = []
        rects = self._normalise(rectangles)
        if len(rects) < 2:
            return []

        # Metric width bounds in pixels (pixel_scale is metres per pixel).
        if pixel_scale and pixel_scale > 0:
            min_gap_px = self.min_gap_width_m / pixel_scale
            max_gap_px = self.max_gap_width_m / pixel_scale
        else:
            min_gap_px = 1.0
            max_gap_px = float("inf")

        horiz_pool, vert_pool = self._orientation_pools(rects)

        raw_breaks: List[Tuple[Tuple[int, int, int, int], float, bool]] = []
        for horizontal, pool, perp_pool in (
            (True, horiz_pool, vert_pool),
            (False, vert_pool, horiz_pool),
        ):
            for line in self._group_into_lines(pool, horizontal):
                for x1, y1, x2, y2, thickness in self._breaks_in_line(line, horizontal):
                    box = self._validate_break(
                        x1, y1, x2, y2, thickness, horizontal,
                        perp_pool, min_gap_px, max_gap_px, image,
                    )
                    if box is not None:
                        raw_breaks.append((box, thickness, horizontal))

        merged = self._merge_collinear(raw_breaks)

        openings: List[Opening] = []
        for idx, (box, _thickness, _horizontal) in enumerate(merged):
            bx1, by1, bx2, by2 = box
            openings.append(
                Opening(
                    id=idx,
                    x1=bx1,
                    y1=by1,
                    x2=bx2,
                    y2=by2,
                    width=bx2 - bx1,
                    height=by2 - by1,
                    area=(bx2 - bx1) * (by2 - by1),
                    opening_type="unknown",
                    confidence=self.confidence,
                    source_rects=[],
                    yolo_matches=[],
                    is_valid=True,
                    status="valid",
                )
            )

        if deleted_openings is not None:
            deleted_openings.clear()
            deleted_openings.extend(self.last_deleted_openings)
        return openings

    def _validate_break(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        thickness: float,
        horizontal: bool,
        perp_pool: List[_Rect],
        min_gap_px: float,
        max_gap_px: float,
        image: Optional[np.ndarray],
    ) -> Optional[Tuple[int, int, int, int]]:
        """Apply width, crossing and emptiness checks; return final box or None."""
        # Along-line span and perpendicular band.
        if horizontal:
            gap_lo, gap_hi = x1, x2
            band_lo, band_hi = y1, y2
        else:
            gap_lo, gap_hi = y1, y2
            band_lo, band_hi = x1, x2

        # Cut out any perpendicular wall crossing the break.
        crossings: List[Tuple[int, int]] = []
        for p in perp_pool:
            if horizontal:
                # A vertical wall crosses if it spans the line band and lies
                # within the gap along x.
                if p.y1 <= band_hi and p.y2 >= band_lo and p.x2 > gap_lo and p.x1 < gap_hi:
                    crossings.append((p.x1, p.x2))
            else:
                if p.x1 <= band_hi and p.x2 >= band_lo and p.y2 > gap_lo and p.y1 < gap_hi:
                    crossings.append((p.y1, p.y2))

        crossed_px = sum(
            min(gap_hi, c1) - max(gap_lo, c0)
            for c0, c1 in crossings
            if min(gap_hi, c1) > max(gap_lo, c0)
        )
        if (gap_hi - gap_lo) > 0 and crossed_px / float(gap_hi - gap_lo) > self.max_perp_cross_ratio:
            seg = self._largest_free_segment(gap_lo, gap_hi, crossings)
            if seg is None:
                return None
            gap_lo, gap_hi = seg

        gap_len = gap_hi - gap_lo
        if gap_len < min_gap_px or gap_len > max_gap_px:
            return None

        if horizontal:
            box = (gap_lo, band_lo, gap_hi, band_hi)
        else:
            box = (band_lo, gap_lo, band_hi, gap_hi)

        if not self._gap_is_empty(box, image):
            return None
        return box

    def _merge_collinear(
        self,
        breaks: List[Tuple[Tuple[int, int, int, int], float, bool]],
    ) -> List[Tuple[Tuple[int, int, int, int], float, bool]]:
        """Merge breaks that overlap or sit within ``merge_distance`` of each other."""
        if not breaks:
            return []
        remaining = list(breaks)
        merged: List[Tuple[Tuple[int, int, int, int], float, bool]] = []
        used = [False] * len(remaining)
        for i in range(len(remaining)):
            if used[i]:
                continue
            box_i, t_i, h_i = remaining[i]
            x1, y1, x2, y2 = box_i
            changed = True
            while changed:
                changed = False
                for j in range(len(remaining)):
                    if used[j] or j == i:
                        continue
                    box_j, _t_j, h_j = remaining[j]
                    if h_j != h_i:
                        continue
                    if self._boxes_near((x1, y1, x2, y2), box_j):
                        x1 = min(x1, box_j[0])
                        y1 = min(y1, box_j[1])
                        x2 = max(x2, box_j[2])
                        y2 = max(y2, box_j[3])
                        used[j] = True
                        changed = True
            used[i] = True
            merged.append(((x1, y1, x2, y2), t_i, h_i))
        return merged

    def _boxes_near(
        self,
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> bool:
        """True if boxes overlap or are within merge_distance on both axes."""
        gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
        return gap_x <= self.merge_distance and gap_y <= self.merge_distance
