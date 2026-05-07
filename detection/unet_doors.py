"""
detection/unet_doors.py — Door detection via the 4-class U-Net (epoch_040.pth).

Pipeline
--------
1. Run the 4-class segmentation U-Net (background / wall / window / door).
2. Extract the door-class mask (class index 3).
3. Connected-component labelling → per-blob door masks.
4. For each blob:
   a. Rect decomposition (rect_decompose.decompose) on the blob's sub-mask
      with coverage_stop=0.70 and max_spill=0.05.  Produces tight axis-aligned
      rectangles that precisely follow the door-mask shape.
   b. Wall-interruption splitting (fallback when rect decompose is disabled):
      if a candidate bbox is significantly crossed by a wall-mask band, split
      it into two sub-bboxes at the band boundary.
5. Size / thickness filtering: discard sub-bboxes that are too small or too
   thin to be plausible door openings.
6. Gap snapping: for each surviving bbox search the geometric gap list for the
   best matching gap and *snap* to that gap's exact position and extent.
   If no gap matches within the search radius, keep the raw bbox as-is.

The result is a list of Opening objects with source="unet_door" that feed
into the deduplication step of OpeningDetectionPipeline alongside any
YOLO or geometric detections.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.models import BBox, Opening, OpeningType, WallSegment

logger = logging.getLogger(__name__)

# Door class index in the 4-class checkpoint
_DOOR_CLASS = 3

# Confidence assigned to U-Net door openings (lower than YOLO/geo to let
# higher-quality sources win in deduplication).
_UNET_DOOR_CONFIDENCE = 0.60


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _find_runs(mask_1d: np.ndarray, min_length: int = 1) -> List[Tuple[int, int]]:
    """
    Return (start, end) pairs for every True run of length >= *min_length*
    in the 1-D boolean array *mask_1d*.  Indices are half-open [start, end).
    """
    runs: List[Tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask_1d):
        if v and not in_run:
            in_run = True
            start = i
        elif not v and in_run:
            in_run = False
            if i - start >= min_length:
                runs.append((start, i))
    if in_run and len(mask_1d) - start >= min_length:
        runs.append((start, len(mask_1d)))
    return runs


def _extract_door_blobs(
    door_mask: np.ndarray,
    min_area: int = 300,
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Connected components on a binary door mask.

    Returns a list of (bbox, sub_mask) pairs where:
    - bbox is (x1, y1, x2, y2) in image coordinates
    - sub_mask is a cropped uint8 binary mask (255=door) for that blob only,
      sized to the bbox dimensions
    """
    if door_mask is None or not np.any(door_mask):
        return []

    # Light morph-close to merge fragmented predictions
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(
        (door_mask > 0).astype(np.uint8) * 255,
        cv2.MORPH_CLOSE, k, iterations=1,
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    blobs: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bbox = (x, y, x + bw, y + bh)
        # Crop the label mask to this blob's bbox (255 where this component)
        sub_mask = ((labels[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255)
        blobs.append((bbox, sub_mask))

    return blobs


def _rect_decompose_blob(
    sub_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    coverage_stop: float = 0.70,
    max_spill: float = 0.05,
    max_overlap: float = 0.50,
    bleed_weight: float = 15.0,
    initial_bleed_weight: float = 15.0,
    bleed_decay: float = 0.0,
    rect_penalty: float = 0.02,
    max_grid_dim: int = 200,
    axis_only: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    Run rect_decompose on a single door blob's sub-mask and return a list of
    (x1, y1, x2, y2) bboxes in the original image coordinate space.

    *sub_mask* is a uint8 H×W binary mask (255=door) cropped to the blob bbox.
    *bbox* is the (x1, y1, x2, y2) offset of sub_mask within the full image.
    """
    try:
        import rect_decompose as rd
    except ImportError:
        logger.debug("rect_decompose not available — using blob bbox directly")
        return [bbox]

    if sub_mask is None or not np.any(sub_mask):
        return [bbox]

    mask01 = (sub_mask > 0).astype(np.uint8)

    try:
        rects = rd.decompose(
            mask01,
            angle_steps          = 2 if axis_only else 12,   # 0° and 90° only
            rect_penalty         = rect_penalty,
            max_grid_dim         = max_grid_dim,
            coverage_stop        = coverage_stop,
            bleed_weight         = bleed_weight,
            initial_bleed_weight = initial_bleed_weight,
            bleed_decay          = bleed_decay,
            axis_only            = axis_only,
            axis_gap             = 0.0,
            max_overlap          = max_overlap,
            max_spill_fraction   = max_spill,
            refine               = True,
            verbose              = False,
        )
    except Exception as exc:
        logger.warning("Door rect decompose failed for blob %s: %s", bbox, exc)
        return [bbox]

    if not rects:
        return [bbox]

    # Convert each Rect's corner points back to image coordinates
    ox, oy = bbox[0], bbox[1]
    result: List[Tuple[int, int, int, int]] = []
    for r in rects:
        corners = r.corners()   # ndarray shape (4,2)
        xs = corners[:, 0] + ox
        ys = corners[:, 1] + oy
        x1, y1 = int(np.floor(xs.min())), int(np.floor(ys.min()))
        x2, y2 = int(np.ceil(xs.max())),  int(np.ceil(ys.max()))
        result.append((x1, y1, x2, y2))

    return result


def _split_by_wall(
    door_bbox: Tuple[int, int, int, int],
    wall_mask: np.ndarray,
    wall_band_ratio: float = 0.50,
    min_band_thickness_px: int = 5,
) -> List[Tuple[int, int, int, int]]:
    """
    Fallback splitter: splits *door_bbox* wherever the wall mask has a
    continuous band spanning >= *wall_band_ratio* of the bbox width/height
    and >= *min_band_thickness_px* pixels thick.

    Returns [door_bbox] when no qualifying band is found.
    """
    x1, y1, x2, y2 = door_bbox
    if wall_mask is None:
        return [door_bbox]

    H, W = wall_mask.shape[:2]
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(W, x2), min(H, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return [door_bbox]

    region = (wall_mask[cy1:cy2, cx1:cx2] > 0).astype(np.float32)
    rh, rw = region.shape

    for axis in ('y', 'x'):
        if axis == 'y':
            frac = region.sum(axis=1) / max(1.0, float(rw))
            span = rh
        else:
            frac = region.sum(axis=0) / max(1.0, float(rh))
            span = rw

        bands = _find_runs(frac >= wall_band_ratio, min_length=min_band_thickness_px)
        if not bands:
            continue

        pieces: List[Tuple[int, int]] = []
        prev_end = 0
        for band_start, band_end in bands:
            if band_start > prev_end:
                pieces.append((prev_end, band_start))
            prev_end = band_end
        if prev_end < span:
            pieces.append((prev_end, span))

        if axis == 'y':
            sub_bboxes = [(x1, cy1 + s, x2, cy1 + e) for s, e in pieces if e > s]
        else:
            sub_bboxes = [(cx1 + s, y1, cx1 + e, y2) for s, e in pieces if e > s]

        if len(sub_bboxes) >= 2:
            logger.debug(
                "Wall-split door bbox %s → %d pieces on %s-axis",
                door_bbox, len(sub_bboxes), axis,
            )
            return sub_bboxes

    return [door_bbox]


def _is_bbox_valid(
    bbox: Tuple[int, int, int, int],
    min_area: int = 200,
    min_thickness_px: int = 12,
) -> bool:
    """
    Return True when the bbox meets minimum area and thickness requirements.

    *min_thickness_px* guards against hair-thin slivers produced by
    rect-decomposition or wall-splitting.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    if w * h < min_area:
        return False
    if min(w, h) < min_thickness_px:
        return False
    return True


def _bbox_to_door_wall_segment(bbox: BBox, idx: int) -> WallSegment:
    """
    Build a door-wall WallSegment from a door bbox.

    The shorter bbox dimension becomes the wall thickness (= the wall the door
    sits in); the longer dimension becomes the door opening width (= wall
    length).  The centerline runs along the middle of the longer axis.
    """
    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    w, h = x2 - x1, y2 - y1
    if w >= h:                       # horizontal door
        thickness = h
        cy = (y1 + y2) / 2.0
        outline = [(int(x1), int(cy)), (int(x2), int(cy))]
    else:                            # vertical door
        thickness = w
        cx = (x1 + x2) / 2.0
        outline = [(int(cx), int(y1)), (int(cx), int(y2))]
    return WallSegment(
        id=f"door-wall-{idx:04d}",
        bbox=bbox,
        thickness=float(thickness),
        is_structural=False,
        is_outer=False,
        is_diagonal=False,
        is_door_wall=True,
        outline=outline,
    )


def _snap_to_gap(
    door_bbox: Tuple[int, int, int, int],
    geo_gaps: List[Opening],
    match_margin: int = 40,
) -> Tuple[BBox, bool]:
    """
    Find the nearest geometric gap whose centre falls inside the door bbox
    (expanded by *match_margin*).  Return the gap's bbox if found (snapped),
    or the original door bbox if not.

    Returns (bbox, was_snapped).
    """
    dx1, dy1, dx2, dy2 = door_bbox
    dcx = (dx1 + dx2) / 2.0
    dcy = (dy1 + dy2) / 2.0

    ex1 = dx1 - match_margin
    ey1 = dy1 - match_margin
    ex2 = dx2 + match_margin
    ey2 = dy2 + match_margin

    best_dist = float("inf")
    best_gap: Optional[Opening] = None

    for gap in geo_gaps:
        gcx = gap.bbox.center_x
        gcy = gap.bbox.center_y
        if ex1 <= gcx <= ex2 and ey1 <= gcy <= ey2:
            dist = math.hypot(gcx - dcx, gcy - dcy)
            if dist < best_dist:
                best_dist = dist
                best_gap = gap

    if best_gap is not None:
        return best_gap.bbox, True

    return BBox(float(dx1), float(dy1), float(dx2), float(dy2)), False


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class UNetDoorDetector:
    """
    Detect doors using the 4-class U-Net checkpoint (epoch_040.pth).

    Usage::

        det = UNetDoorDetector(ckpt_path="weights/epoch_040.pth")
        det.initialize()
        doors, door_mask = det.detect(img_bgr, geo_gaps, wall_mask=wall_mask)
    """

    def __init__(
        self,
        ckpt_path: str = "weights/epoch_040.pth",
        img_size: int = 1024,
        confidence: float = 0.5,
        min_door_area: int = 300,
        match_margin: int = 40,
        device: Optional[str] = None,
        # Rect decomposition
        use_rect_decompose: bool = True,
        door_rect_coverage_stop: float = 0.70,
        door_rect_max_spill: float = 0.05,
        door_rect_max_overlap: float = 0.50,
        door_rect_bleed_weight: float = 15.0,
        door_rect_initial_bleed_weight: float = 15.0,
        door_rect_bleed_decay: float = 0.0,
        door_rect_penalty: float = 0.02,
        door_rect_max_grid_dim: int = 200,
        door_rect_axis_only: bool = True,
        # Wall-split fallback parameters
        wall_band_ratio: float = 0.50,
        min_wall_band_px: int = 5,
        # Post-decompose validity
        min_piece_area: int = 200,
        min_piece_thickness_px: int = 12,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.img_size = img_size
        self.confidence = confidence
        self.min_door_area = min_door_area
        self.match_margin = match_margin
        self.device = device
        self.use_rect_decompose = use_rect_decompose
        self.door_rect_coverage_stop = door_rect_coverage_stop
        self.door_rect_max_spill = door_rect_max_spill
        self.door_rect_max_overlap = door_rect_max_overlap
        self.door_rect_bleed_weight = door_rect_bleed_weight
        self.door_rect_initial_bleed_weight = door_rect_initial_bleed_weight
        self.door_rect_bleed_decay = door_rect_bleed_decay
        self.door_rect_penalty = door_rect_penalty
        self.door_rect_max_grid_dim = door_rect_max_grid_dim
        self.door_rect_axis_only = door_rect_axis_only
        self.wall_band_ratio = wall_band_ratio
        self.min_wall_band_px = min_wall_band_px
        self.min_piece_area = min_piece_area
        self.min_piece_thickness_px = min_piece_thickness_px

        self._model = None
        self._num_classes: int = 0
        self._torch_device = None
        # Populated during detect() — valid pieces after decompose/filter,
        # before gap snapping.  Consumed by the rect overlay debug image.
        self._debug_rect_bboxes: List[Tuple[int, int, int, int]] = []
        # Door-wall WallSegments built from snapped bboxes — injected into
        # result.walls so the SH3D exporter has a precise parent wall for
        # each door and never needs to guess or create a synthetic wall.
        self._debug_door_wall_segments: List[WallSegment] = []

    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """Load the model.  Returns True on success."""
        ckpt = Path(self.ckpt_path)
        if not ckpt.is_absolute():
            ckpt = Path.cwd() / ckpt
        if not ckpt.exists():
            logger.warning("UNetDoorDetector: checkpoint not found: %s", ckpt)
            return False
        try:
            import torch
            from detect_unet import build_model_from_checkpoint

            dev_str = self.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
            self._torch_device = __import__("torch").device(dev_str)
            self._model, self._num_classes = build_model_from_checkpoint(
                str(ckpt), self._torch_device
            )
            self._model.eval()
            logger.info(
                "UNetDoorDetector: loaded %s  (%d classes)",
                ckpt.name, self._num_classes,
            )
            return True
        except Exception as exc:
            logger.warning("UNetDoorDetector: load failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    def detect(
        self,
        img_bgr: np.ndarray,
        geo_gaps: List[Opening],
        wall_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[Opening], Optional[np.ndarray]]:
        """
        Run inference and return ``(door_openings, door_mask)``.

        Parameters
        ----------
        img_bgr:
            Input image in BGR format (original resolution).
        geo_gaps:
            Geometric gap openings used for bbox snapping.
        wall_mask:
            Optional binary wall mask (uint8, 255=wall) at the same
            resolution as *img_bgr*.  Used as the fallback wall-split
            splitter when rect decompose is disabled.

        Returns
        -------
        door_openings:
            List of Opening objects (source="unet_door").
        door_mask:
            uint8 H×W array (255 = door pixel) at original resolution —
            useful for debug visualisation.
        """
        if self._model is None:
            return [], None

        door_mask = self._run_inference(img_bgr)
        if door_mask is None:
            return [], None

        blobs = _extract_door_blobs(door_mask, min_area=self.min_door_area)
        if not blobs:
            logger.info("UNetDoorDetector: no door blobs found after inference")
            return [], door_mask

        doors: List[Opening] = []
        snapped_count = 0
        decomposed_count = 0
        filtered_count = 0
        self._debug_rect_bboxes = []          # reset for this run
        self._debug_door_wall_segments = []   # reset for this run

        for blob_bbox, sub_mask in blobs:
            # ── 1. Get candidate bboxes from this blob ─────────────────
            if self.use_rect_decompose:
                # Rect decomposition: tight rectangles that follow the mask
                # shape with only 5% bleed allowed.
                candidates = _rect_decompose_blob(
                    sub_mask,
                    blob_bbox,
                    coverage_stop        = self.door_rect_coverage_stop,
                    max_spill            = self.door_rect_max_spill,
                    max_overlap          = self.door_rect_max_overlap,
                    bleed_weight         = self.door_rect_bleed_weight,
                    initial_bleed_weight = self.door_rect_initial_bleed_weight,
                    bleed_decay          = self.door_rect_bleed_decay,
                    rect_penalty         = self.door_rect_penalty,
                    max_grid_dim         = self.door_rect_max_grid_dim,
                    axis_only            = self.door_rect_axis_only,
                )
                if len(candidates) > 1:
                    decomposed_count += 1
                    logger.debug(
                        "Door blob %s → %d rects via decompose",
                        blob_bbox, len(candidates),
                    )
            else:
                # Fallback: wall-band splitting
                candidates = _split_by_wall(
                    blob_bbox,
                    wall_mask,
                    wall_band_ratio      = self.wall_band_ratio,
                    min_band_thickness_px = self.min_wall_band_px,
                )

            for piece in candidates:
                # ── 2. Size / thickness filter ──────────────────────────
                if not _is_bbox_valid(
                    piece,
                    min_area         = self.min_piece_area,
                    min_thickness_px = self.min_piece_thickness_px,
                ):
                    filtered_count += 1
                    logger.debug("UNetDoorDetector: dropped thin/tiny piece %s", piece)
                    continue

                # Record the valid decomposed piece before snapping
                self._debug_rect_bboxes.append(piece)

                # ── 3. Snap to nearest geometric gap ────────────────────
                snapped_bbox, was_snapped = _snap_to_gap(
                    piece, geo_gaps, match_margin=self.match_margin
                )
                if was_snapped:
                    snapped_count += 1

                # ── 4. Build door-wall segment from final bbox ───────────
                # The door wall gives the SH3D exporter a pre-built parent
                # wall so ParentWallFinder always finds an exact match —
                # no synthetic wall creation or thickness guessing needed.
                dw_seg = _bbox_to_door_wall_segment(
                    snapped_bbox, len(self._debug_door_wall_segments)
                )
                self._debug_door_wall_segments.append(dw_seg)

                doors.append(Opening(
                    bbox=snapped_bbox,
                    opening_type=OpeningType.DOOR,
                    confidence=_UNET_DOOR_CONFIDENCE,
                    source="unet_door",
                ))

        logger.info(
            "UNetDoorDetector: %d blobs → %d decomposed → %d filtered → "
            "%d openings (%d snapped to gaps)",
            len(blobs), decomposed_count, filtered_count,
            len(doors), snapped_count,
        )
        return doors, door_mask

    # ------------------------------------------------------------------
    def _run_inference(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Run the 4-class U-Net and return a binary door mask (uint8, 255=door)
        at the original image resolution.
        """
        try:
            import torch
            from detect_unet import get_val_transform

            h_orig, w_orig = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            transform = get_val_transform(self.img_size)
            inp = transform(image=img_rgb)["image"].unsqueeze(0).to(self._torch_device)

            with torch.no_grad():
                logits = self._model(inp)
                probs = torch.softmax(logits, dim=1)
                conf_map, pred_idx = probs.max(dim=1)

            conf_np = conf_map.squeeze(0).cpu().numpy()
            pred_np = pred_idx.squeeze(0).cpu().numpy().copy()
            pred_np[conf_np < self.confidence] = 0   # uncertain → background

            # Resize to original resolution
            pred_full = cv2.resize(
                pred_np.astype(np.uint8), (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            )

            # Extract door class only (valid when checkpoint has ≥ 4 classes)
            if self._num_classes < 4:
                logger.warning(
                    "UNetDoorDetector: checkpoint only has %d classes "
                    "(door=class 3 missing)", self._num_classes
                )
                return None

            door_mask = (pred_full == _DOOR_CLASS).astype(np.uint8) * 255
            pct = 100.0 * int(np.count_nonzero(door_mask)) / door_mask.size
            logger.info("UNetDoorDetector: %.2f%% door pixels", pct)
            return door_mask

        except Exception as exc:
            logger.warning("UNetDoorDetector inference failed: %s", exc)
            return None
