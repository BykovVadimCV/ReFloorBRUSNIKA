"""
Wall detection adapter for the ReFloorBRUSNIKA pipeline.

Wraps VersatileWallDetector from the existing walldetector.py,
converting its output to unified core.models types.
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.config import PipelineConfig
from core.models import BBox, DetectionResult, WallSegment
from detection.base import WallDetector
from detection.walldetector import VersatileWallDetector, WallRectangle, Theme

import logging

_logger = logging.getLogger(__name__)


def _build_enclosed_regions(
    original_img: np.ndarray,
    black_threshold: int = 60,
    white_threshold: int = 220,
) -> Tuple[int, np.ndarray]:
    """Find enclosed white regions — paint-bucket style flood fill.

    Barriers = anything that is NOT near-white. This handles:
    - Black outlines (grayscale < black_threshold)
    - Colored wall fills (red, pink, etc. — not white)
    - Any non-white content (text, furniture icons, etc.)

    Only near-white pixels (all channels >= white_threshold) are fillable.
    Flood-fill from borders marks exterior white as background.
    Remaining white = enclosed interior regions.

    Returns (num_labels, labels) from connectedComponents.
    Label 0 = background/barriers, labels 1..N = enclosed regions.
    """
    h, w = original_img.shape[:2]

    if len(original_img.shape) == 3:
        # A pixel is "white" only if ALL channels are >= white_threshold
        is_white = np.all(original_img >= white_threshold, axis=2)
    else:
        is_white = original_img >= white_threshold

    not_black = is_white.astype(np.uint8) * 255

    # Flood-fill from all border pixels to mark the background
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood = not_black.copy()
    for x in range(w):
        if flood[0, x] == 255:
            cv2.floodFill(flood, padded, (x, 0), 128)
        if flood[h - 1, x] == 255:
            cv2.floodFill(flood, padded, (x, h - 1), 128)
    for y in range(h):
        if flood[y, 0] == 255:
            cv2.floodFill(flood, padded, (0, y), 128)
        if flood[y, w - 1] == 255:
            cv2.floodFill(flood, padded, (w - 1, y), 128)

    # Enclosed = pixels still 255 (background became 128, outlines are 0)
    enclosed = (flood == 255).astype(np.uint8) * 255

    return cv2.connectedComponents(enclosed)


def refine_wall_mask_by_enclosed_regions(
    wall_mask: np.ndarray,
    original_img: np.ndarray,
    coverage_threshold: float = 0.50,
    black_threshold: int = 60,
) -> np.ndarray:
    """Refine the wall mask using enclosed-region analysis.

    Algorithm:
      1. Find enclosed white regions bounded by non-white outlines.
      2. For each region, compute wall-mask coverage ratio.
         - If coverage >= threshold → fill entire region with wall mask.
         - If coverage < threshold  → erase wall mask pixels in that region.
    """
    num_labels, labels = _build_enclosed_regions(
        original_img, black_threshold=black_threshold,
    )

    _logger.debug(
        "Enclosed-region analysis: %d regions found", num_labels - 1
    )

    refined = wall_mask.copy()
    wall_bin = (wall_mask > 0)

    for label_id in range(1, num_labels):
        region_mask = (labels == label_id)
        region_area = int(np.sum(region_mask))
        if region_area == 0:
            continue

        wall_in_region = int(np.sum(wall_bin & region_mask))
        coverage = wall_in_region / region_area

        if coverage >= coverage_threshold:
            refined[region_mask] = 255
        else:
            refined[region_mask] = 0

    _logger.info(
        "Wall mask refined: %d enclosed regions processed",
        num_labels - 1,
    )
    return refined


def build_grey_wall_image(
    original_bgr: np.ndarray,
    labels: np.ndarray,
    num_labels: int,
    wall_mask: np.ndarray,
    coverage_threshold: float = 0.50,
    min_region_area: int = 50,
    grey_value: int = 160,
    black_threshold: int = 60,
    expand_px: int = 2,
) -> np.ndarray:
    """Build a clean grey image: wall regions filled grey, rest white.

    For each enclosed region, checks U-Net wall mask coverage.
    If >= threshold, the entire region is marked as wall.
    Result: white canvas with grey wall fills + original black outlines.
    Ready for VersatileWallDetector in color mode.
    """
    h, w = original_bgr.shape[:2]
    wall_bin = wall_mask > 0

    # Vote per region
    wall_region_mask = np.zeros((h, w), dtype=np.uint8)
    for label_id in range(1, num_labels):
        region = labels == label_id
        area = int(np.sum(region))
        if area < min_region_area:
            continue
        coverage = int(np.sum(wall_bin & region)) / area
        if coverage >= coverage_threshold:
            wall_region_mask[region] = 255

    # Expand wall regions to cover outline edges
    if expand_px > 0:
        kernel = np.ones((2 * expand_px + 1, 2 * expand_px + 1), np.uint8)
        wall_region_mask = cv2.dilate(wall_region_mask, kernel, iterations=1)

    # White canvas, fill walls grey
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[wall_region_mask > 0] = grey_value

    # Re-draw original black outlines on top
    if len(original_bgr.shape) == 3:
        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = original_bgr
    black_pixels = gray < black_threshold
    if len(original_bgr.shape) == 3:
        out[black_pixels] = original_bgr[black_pixels]
    else:
        out[black_pixels] = gray[black_pixels][:, None]

    return out


def debug_enclosed_regions(
    original_img: np.ndarray,
    wall_mask: np.ndarray,
    output_path: str,
    coverage_threshold: float = 0.50,
    black_threshold: int = 60,
) -> None:
    """Save a debug image with each enclosed region colored uniquely.

    Green = wall region (≥50% coverage, will be filled).
    Distinct hue per non-wall region. Coverage % at centroid.
    """
    num_labels, labels = _build_enclosed_regions(
        original_img, black_threshold=black_threshold,
    )
    wall_bin = (wall_mask > 0)

    if len(original_img.shape) == 3:
        base = original_img.copy()
    else:
        base = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    color_layer = np.zeros_like(base)

    for label_id in range(1, num_labels):
        region_mask = (labels == label_id)
        region_area = int(np.sum(region_mask))
        if region_area == 0:
            continue

        wall_in_region = int(np.sum(wall_bin & region_mask))
        coverage = wall_in_region / region_area
        is_wall = coverage >= coverage_threshold

        # Distinct color per region via golden-ratio hue spacing
        hue = int((label_id * 137.508) % 180)
        if is_wall:
            hsv_color = np.uint8([[[60, 200, 220]]])   # green
        else:
            hsv_color = np.uint8([[[hue, 180, 200]]])   # unique hue
        bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        color_layer[region_mask] = bgr

        # Coverage text at centroid
        ys, xs = np.where(region_mask)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        label_text = f"{coverage:.0%}"
        cv2.putText(base, label_text, (cx - 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(base, label_text, (cx - 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    mask_any = np.any(color_layer > 0, axis=2)
    alpha = 0.4
    base[mask_any] = cv2.addWeighted(base, 1 - alpha, color_layer, alpha, 0)[mask_any]

    # Overlay wall mask in red, semi-transparent
    wall_overlay = np.zeros_like(base)
    wall_overlay[wall_bin] = (0, 0, 255)  # red in BGR
    wall_alpha = 0.45
    base[wall_bin] = cv2.addWeighted(
        base, 1 - wall_alpha, wall_overlay, wall_alpha, 0,
    )[wall_bin]

    cv2.imwrite(output_path, base)
    _logger.info("Enclosed regions debug image saved: %s", output_path)


# ─────────────────────────────────────────────────────────────
# Unenclosed wall rectangle detection
# ─────────────────────────────────────────────────────────────

def _extract_wall_outlines(
    img_bgr: np.ndarray,
    wall_mask: np.ndarray,
    black_threshold: int = 80,
    dilate_mask: int = 10,
    close_kernel: int = 5,
    close_iterations: int = 2,
) -> np.ndarray:
    """Extract black outline pixels within the U-Net wall region.

    1. Find dark pixels (grayscale < black_threshold).
    2. Dilate U-Net mask to catch outline edges near predicted boundaries.
    3. Intersect: keep only dark pixels inside dilated mask.
    4. Morphological close to bridge small outline gaps.

    Returns binary mask (H, W) uint8: 255 = wall outline, 0 = other.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    dark_pixels = (gray < black_threshold).astype(np.uint8) * 255

    if dilate_mask > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_mask * 2 + 1, dilate_mask * 2 + 1))
        expanded_mask = cv2.dilate(wall_mask, kernel, iterations=1)
    else:
        expanded_mask = wall_mask

    outlines = cv2.bitwise_and(dark_pixels, expanded_mask)

    if close_kernel > 0 and close_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        outlines = cv2.morphologyEx(outlines, cv2.MORPH_CLOSE, kernel,
                                     iterations=close_iterations)
    return outlines


def _enclosed_regions_from_mask(boundary_mask: np.ndarray):
    """Find enclosed regions using a binary mask as boundaries.

    boundary_mask: (H, W) uint8, 255 = barrier, 0 = fillable.
    Returns (num_labels, labels) from connectedComponents.
    Label 0 = background/barriers, 1..N = enclosed regions.
    """
    h, w = boundary_mask.shape[:2]
    fillable = (boundary_mask == 0).astype(np.uint8) * 255

    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood = fillable.copy()

    for x in range(w):
        if flood[0, x] == 255:
            cv2.floodFill(flood, padded, (x, 0), 128)
        if flood[h - 1, x] == 255:
            cv2.floodFill(flood, padded, (x, h - 1), 128)
    for y in range(h):
        if flood[y, 0] == 255:
            cv2.floodFill(flood, padded, (0, y), 128)
        if flood[y, w - 1] == 255:
            cv2.floodFill(flood, padded, (w - 1, y), 128)

    enclosed = (flood == 255).astype(np.uint8) * 255
    return cv2.connectedComponents(enclosed)


def _largest_rectangle_in_mask(mask_bool: np.ndarray):
    """Find the largest axis-aligned rectangle of True pixels (O(H*W))."""
    h, w = mask_bool.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best_coords = None

    for i in range(h):
        row = mask_bool[i]
        heights = heights + row.astype(np.int32)
        heights[~row] = 0

        stack = []
        j = 0
        while j <= w:
            cur_h = heights[j] if j < w else 0
            if not stack or cur_h >= heights[stack[-1]]:
                stack.append(j)
                j += 1
            else:
                top = stack.pop()
                height = heights[top]
                if height == 0:
                    continue
                width_val = j if not stack else j - stack[-1] - 1
                area = int(height * width_val)
                if area > best_area:
                    best_area = area
                    y2 = i
                    y1 = i - height + 1
                    x2 = j - 1
                    x1 = (stack[-1] + 1) if stack else 0
                    best_coords = (x1, y1, x2, y2)

    if best_coords is None or best_area <= 0:
        return None
    return (*best_coords, best_area)


def _extract_rects_from_mask(
    mask: np.ndarray,
    min_area: int = 200,
) -> List[Tuple[int, int, int, int]]:
    """Greedy largest-rectangle extraction from a binary mask."""
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    work = (cleaned > 0).copy()
    bboxes = []

    while True:
        res = _largest_rectangle_in_mask(work)
        if res is None:
            break
        x1, y1, x2, y2, area = res
        if area < min_area:
            break
        bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        work[y1:y2 + 1, x1:x2 + 1] = False
        if not work.any():
            break

    return bboxes


def find_unenclosed_wall_rects(
    img_bgr: np.ndarray,
    wall_mask: np.ndarray,
    min_area: int = 200,
    merge_gap: int = 15,
    min_merged_area: int = 1000,
) -> List[Tuple[int, int, int, int]]:
    """Find axis-aligned bounding boxes for wall regions not in enclosed spaces.

    1. Extract wall outlines (black lines within U-Net mask).
    2. Find enclosed regions from those outlines.
    3. Identify wall-mask pixels NOT covered by any enclosed region.
    4. Greedy largest-rectangle extraction on uncovered wall pixels.
    5. Merge nearby rects (dilate by merge_gap), re-extract, filter small.

    Returns list of (x1, y1, x2, y2) tuples.
    """
    h, w = wall_mask.shape

    # Wall outlines = black pixels inside/near U-Net mask
    outlines = _extract_wall_outlines(img_bgr, wall_mask)

    # Enclosed regions from outlines
    n_enc, labels = _enclosed_regions_from_mask(outlines)
    covered = np.zeros((h, w), dtype=bool)
    for i in range(1, n_enc):
        covered |= (labels == i)

    _logger.info("Outline-based enclosed regions: %d, covered %.1f%% of wall mask",
                 n_enc - 1,
                 np.sum(covered & (wall_mask > 0)) * 100 / max(1, np.sum(wall_mask > 0)))

    # Uncovered wall pixels
    uncovered = ((wall_mask > 0) & ~covered).astype(np.uint8) * 255

    # Greedy rectangle extraction
    raw_bboxes = _extract_rects_from_mask(uncovered, min_area=min_area)
    _logger.info("Unenclosed wall rects: %d raw", len(raw_bboxes))

    if not raw_bboxes:
        return []

    # Merge nearby rects: paint → dilate → connected components → re-extract
    canvas = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in raw_bboxes:
        canvas[y1:y2, x1:x2] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (merge_gap * 2 + 1, merge_gap * 2 + 1))
    dilated = cv2.dilate(canvas, kernel, iterations=1)

    n_cc, cc_labels = cv2.connectedComponents(dilated)
    merged = []
    for cc_id in range(1, n_cc):
        ys, xs = np.where(cc_labels == cc_id)
        cx1, cy1 = int(xs.min()), int(ys.min())
        cx2, cy2 = int(xs.max()) + 1, int(ys.max()) + 1
        region = canvas[cy1:cy2, cx1:cx2]
        for sx1, sy1, sx2, sy2 in _extract_rects_from_mask(region, min_area=min_merged_area):
            merged.append((cx1 + sx1, cy1 + sy1, cx1 + sx2, cy1 + sy2))

    _logger.info("Unenclosed wall rects: %d after merge+filter (gap=%d, min=%d)",
                 len(merged), merge_gap, min_merged_area)

    return merged


def paint_unenclosed_walls_grey(
    grey_img: np.ndarray,
    rects: List[Tuple[int, int, int, int]],
    grey_value: int = 160,
    black_threshold: int = 60,
) -> np.ndarray:
    """Paint unenclosed wall rectangles grey, preserving black outlines."""
    result = grey_img.copy()
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result
    for x1, y1, x2, y2 in rects:
        # Fill grey, but preserve existing black outline pixels
        roi = gray[y1:y2, x1:x2]
        not_black = roi >= black_threshold
        if len(result.shape) == 3:
            result[y1:y2, x1:x2][not_black] = grey_value
        else:
            result[y1:y2, x1:x2][not_black] = grey_value
    return result


def detect_dominant_wall_color(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Detect the dominant wall color in a floorplan image using K-means
    clustering + spatial coherence scoring.

    Strategy:
      1. Cluster all pixels into K color groups.
      2. Exclude near-white clusters (background) and near-black clusters (text).
      3. For each surviving cluster, compute spatial coherence (largest CC area).
      4. Rank by weighted combination of pixel count + coherence.

    Returns (R, G, B).
    """
    # Delegate to the improved method on ColorBasedWallDetector
    from detection.walldetector import ColorBasedWallDetector
    _det = ColorBasedWallDetector(auto_wall_color=True)
    return _det._detect_wall_color_auto(image)


def wall_rectangle_to_segment(rect: WallRectangle) -> WallSegment:
    """
    Convert a WallRectangle (from walldetector.py) to a unified WallSegment.

    Args:
        rect: WallRectangle from VersatileWallDetector

    Returns:
        Unified WallSegment with BBox
    """
    bbox = BBox(
        x1=float(rect.x1),
        y1=float(rect.y1),
        x2=float(rect.x2),
        y2=float(rect.y2),
    )
    thickness = float(min(rect.width, rect.height))
    return WallSegment(
        id=str(rect.id),
        bbox=bbox,
        thickness=thickness,
        is_structural=True,
        outline=rect.outline if rect.outline else None,
    )


def wall_segment_to_dict(seg: WallSegment) -> dict:
    """
    Convert a WallSegment back to a dict usable by the exporter.

    Preserves the 'x1', 'y1', 'x2', 'y2' format expected by
    floorplanexporter.py / parse_rectangle().
    """
    return seg.to_dict()


class ColorWallDetector(WallDetector):
    """
    Color-based wall detector adapter.

    Wraps VersatileWallDetector, converts WallRectangle output to
    unified WallSegment/DetectionResult format.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._inner: Optional[VersatileWallDetector] = None
        self._last_rectangles: List[WallRectangle] = []  # Cache for pipeline

    def _ensure_inner(self, ocr_model=None) -> VersatileWallDetector:
        """Lazily initialize the inner VersatileWallDetector."""
        if self._inner is None:
            cfg = self.config
            self._inner = VersatileWallDetector(
                wall_color_hex=cfg.wall_color_hex,
                color_tolerance=cfg.color_tolerance,
                auto_wall_color=False,  # color override handled in detect()
                wall_detection_mode=cfg.wall_mode,
                structure_method=cfg.structure_method,
                ocr_model=ocr_model,
            )
        return self._inner

    def initialize(self, ocr_model=None) -> None:
        """
        Explicitly initialize the inner detector.

        Call this before first use when you have an OCR model to share.

        Args:
            ocr_model: Optional docTR OCR model instance
        """
        self._inner = VersatileWallDetector(
            wall_color_hex=self.config.wall_color_hex,
            color_tolerance=self.config.color_tolerance,
            auto_wall_color=False,  # color override handled in detect()
            wall_detection_mode=self.config.wall_mode,
            structure_method=self.config.structure_method,
            ocr_model=ocr_model,
        )

    @property
    def name(self) -> str:
        return "color_wall_detector"

    def detect(
        self,
        image: np.ndarray,
        config: Optional[PipelineConfig] = None,
    ) -> DetectionResult:
        """
        Run wall detection on an image.

        Args:
            image: Input image in BGR format
            config: Optional config override

        Returns:
            DetectionResult with walls, wall_mask, and outline_mask
        """
        cfg = config or self.config
        inner = self._ensure_inner()

        if cfg.auto_wall_color:
            r, g, b = detect_dominant_wall_color(image)
            inner.wall_rgb = (r, g, b)
            # Use auto_wall_color=True so filter_walls_by_color takes the
            # wider-tolerance _filter_walls_auto path.  The color is already
            # set, but _detect_wall_color_auto will simply re-confirm it.
            inner.auto_wall_color = True

        wall_mask, rectangles, outline_mask = inner.process(image)

        # Cache rectangles for downstream use (opening detection, export)
        self._last_rectangles = rectangles

        walls = [wall_rectangle_to_segment(r) for r in rectangles]

        return DetectionResult(
            walls=walls,
            openings=[],
            wall_mask=wall_mask,
            outline_mask=outline_mask,
            source=self.name,
        )

    def detect_with_color(
        self,
        image: np.ndarray,
        wall_rgb: Tuple[int, int, int],
    ) -> DetectionResult:
        """Run wall detection with a forced wall color (no auto-detection).

        Used for the grey-image pipeline where wall color is known (160,160,160).
        """
        inner = self._ensure_inner()
        inner.wall_rgb = wall_rgb
        inner.auto_wall_color = True  # use wider-tolerance path

        wall_mask, rectangles, outline_mask = inner.process(image)
        self._last_rectangles = rectangles
        walls = [wall_rectangle_to_segment(r) for r in rectangles]

        return DetectionResult(
            walls=walls,
            openings=[],
            wall_mask=wall_mask,
            outline_mask=outline_mask,
            source="unet_grey_wall_detector",
        )

    @property
    def ocr_model(self):
        """Access the OCR model from the inner detector."""
        if self._inner is not None:
            return self._inner.ocr_model
        return None

    def get_inner(self) -> VersatileWallDetector:
        """
        Get the underlying VersatileWallDetector.

        Useful for passing to other modules that need the raw detector
        (e.g., for OCR text extraction).
        """
        return self._ensure_inner()


# ============================================================
# Wall Snapping — bridge gaps & align wall segments (pixel space)
# ============================================================
# Mirrors the logic in floorplanexporter.py but operates on
# WallSegment / BBox objects in pixel coordinates, before export.

# Tolerances (pixels)
_AXIS_SNAP_PX = 8.0      # align walls that share an axis within this
_GAP_BRIDGE_PX = 20.0    # extend endpoints to bridge gaps up to this
_CORNER_SNAP_PX = 10.0   # snap nearby endpoints together


def _seg_is_horizontal(seg: WallSegment) -> bool:
    """True if the segment's bounding box is wider than tall."""
    return seg.bbox.width >= seg.bbox.height


def _seg_centerline(seg: WallSegment, attr: str) -> float:
    """Return the centerline value along the given axis.

    attr='y' → (y1+y2)/2   (for horizontal walls)
    attr='x' → (x1+x2)/2   (for vertical walls)
    """
    if attr == 'y':
        return (seg.bbox.y1 + seg.bbox.y2) / 2.0
    return (seg.bbox.x1 + seg.bbox.x2) / 2.0


def _seg_length(seg: WallSegment) -> float:
    return max(seg.bbox.width, seg.bbox.height)


def snap_walls(
    walls: List[WallSegment],
    axis_tolerance: float = _AXIS_SNAP_PX,
    gap_tolerance: float = _GAP_BRIDGE_PX,
    corner_tolerance: float = _CORNER_SNAP_PX,
) -> List[WallSegment]:
    """Apply snapping to a list of WallSegments in pixel space.

    Steps (same order as floorplanexporter):
      1. Axis alignment — group walls on similar axes, average them
      2. Endpoint extension — bridge small gaps between walls
      3. Corner snapping — snap nearby endpoints together

    Modifies bbox coords in place and returns the same list.
    """
    _align_to_shared_axes(walls, axis_tolerance)
    _extend_endpoints(walls, gap_tolerance, iterations=2)
    _snap_corners(walls, corner_tolerance)
    return walls


def _align_to_shared_axes(
    walls: List[WallSegment], tolerance: float
) -> None:
    """Group walls on similar horizontal/vertical axes and align them."""
    h_walls = [w for w in walls if _seg_is_horizontal(w)]
    v_walls = [w for w in walls if not _seg_is_horizontal(w)]

    for wall_list, attr in ((h_walls, 'y'), (v_walls, 'x')):
        groups: List[dict] = []
        for w in wall_list:
            val = _seg_centerline(w, attr)
            placed = False
            for g in groups:
                if abs(val - g['val']) < tolerance:
                    g['walls'].append(w)
                    total_len = sum(_seg_length(ww) for ww in g['walls'])
                    if total_len > 0:
                        g['val'] = sum(
                            _seg_centerline(ww, attr) * _seg_length(ww)
                            for ww in g['walls']
                        ) / total_len
                    placed = True
                    break
            if not placed:
                groups.append({'walls': [w], 'val': val})

        for g in groups:
            v = g['val']
            for w in g['walls']:
                thickness = min(w.bbox.width, w.bbox.height)
                half = thickness / 2.0
                if attr == 'y':
                    w.bbox.y1 = v - half
                    w.bbox.y2 = v + half
                else:
                    w.bbox.x1 = v - half
                    w.bbox.x2 = v + half


def _extend_endpoints(
    walls: List[WallSegment], max_gap: float, iterations: int = 2
) -> None:
    """Extend wall endpoints along their axis to bridge gaps."""
    for _ in range(iterations):
        for wall in walls:
            if _seg_length(wall) < 1.0:
                continue
            _try_extend(wall, 'start', walls, max_gap)
            _try_extend(wall, 'end', walls, max_gap)


def _try_extend(
    wall: WallSegment,
    endpoint: str,
    all_walls: List[WallSegment],
    max_gap: float,
) -> None:
    """Try to extend one endpoint of *wall* to meet a neighbor."""
    is_h = _seg_is_horizontal(wall)
    b = wall.bbox

    if is_h:
        # horizontal wall: extend x1 (start) or x2 (end)
        pt_x = b.x1 if endpoint == 'start' else b.x2
        pt_y = (b.y1 + b.y2) / 2.0
    else:
        # vertical wall: extend y1 (start) or y2 (end)
        pt_x = (b.x1 + b.x2) / 2.0
        pt_y = b.y1 if endpoint == 'start' else b.y2

    best_target = None
    best_gap = max_gap

    for other in all_walls:
        if other.id == wall.id:
            continue
        ob = other.bbox
        other_h = _seg_is_horizontal(other)

        if is_h:
            if other_h:
                # Parallel horizontal: check if on same axis
                other_cy = (ob.y1 + ob.y2) / 2.0
                half = (min(b.height, b.width) + min(ob.height, ob.width)) / 2.0
                if abs(pt_y - other_cy) > half:
                    continue
                if endpoint == 'start' and ob.x2 < pt_x:
                    gap = pt_x - ob.x2
                    target = ob.x2
                elif endpoint == 'end' and ob.x1 > pt_x:
                    gap = ob.x1 - pt_x
                    target = ob.x1
                else:
                    continue
            else:
                # Perpendicular: T-junction — extend to vertical wall's axis
                other_cx = (ob.x1 + ob.x2) / 2.0
                half = (min(b.height, b.width) + min(ob.height, ob.width)) / 2.0
                if not (ob.y1 - half <= pt_y <= ob.y2 + half):
                    continue
                gap = abs(pt_x - other_cx) - half
                target = other_cx
        else:
            if not other_h:
                # Parallel vertical
                other_cx = (ob.x1 + ob.x2) / 2.0
                half = (min(b.height, b.width) + min(ob.height, ob.width)) / 2.0
                if abs(pt_x - other_cx) > half:
                    continue
                if endpoint == 'start' and ob.y2 < pt_y:
                    gap = pt_y - ob.y2
                    target = ob.y2
                elif endpoint == 'end' and ob.y1 > pt_y:
                    gap = ob.y1 - pt_y
                    target = ob.y1
                else:
                    continue
            else:
                # Perpendicular: T-junction — extend to horizontal wall's axis
                other_cy = (ob.y1 + ob.y2) / 2.0
                half = (min(b.height, b.width) + min(ob.height, ob.width)) / 2.0
                if not (ob.x1 - half <= pt_x <= ob.x2 + half):
                    continue
                gap = abs(pt_y - other_cy) - half
                target = other_cy

        if 0 <= gap < best_gap:
            best_gap = gap
            best_target = target

    if best_target is not None:
        if is_h:
            if endpoint == 'start':
                wall.bbox.x1 = best_target
            else:
                wall.bbox.x2 = best_target
        else:
            if endpoint == 'start':
                wall.bbox.y1 = best_target
            else:
                wall.bbox.y2 = best_target


def _snap_corners(
    walls: List[WallSegment], snap_distance: float
) -> None:
    """Snap nearby endpoints of perpendicular walls together."""
    import math

    def _endpoints(w):
        b = w.bbox
        is_h = _seg_is_horizontal(w)
        if is_h:
            return [
                (w, 'x1', b.x1, (b.y1 + b.y2) / 2),
                (w, 'x2', b.x2, (b.y1 + b.y2) / 2),
            ]
        else:
            return [
                (w, 'y1', (b.x1 + b.x2) / 2, b.y1),
                (w, 'y2', (b.x1 + b.x2) / 2, b.y2),
            ]

    eps = []
    for w in walls:
        eps.extend(_endpoints(w))

    for i, (w1, attr1, x1, y1) in enumerate(eps):
        for w2, attr2, x2, y2 in eps[i + 1:]:
            if w1.id == w2.id:
                continue
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist > snap_distance:
                continue
            # Snap shorter wall's endpoint to longer wall's endpoint
            if _seg_length(w1) >= _seg_length(w2):
                target_x, target_y = x1, y1
            else:
                target_x, target_y = x2, y2

            for w, attr in ((w1, attr1), (w2, attr2)):
                is_h = _seg_is_horizontal(w)
                if is_h:
                    setattr(w.bbox, attr, target_x)
                else:
                    setattr(w.bbox, attr, target_y)
