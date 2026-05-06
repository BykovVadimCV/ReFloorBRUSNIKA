"""
detection/unet_doors.py — Door detection via the 4-class U-Net (epoch_040.pth).

Pipeline
--------
1. Run the 4-class segmentation U-Net (background / wall / window / door).
2. Extract the door-class mask (class index 3).
3. Connected-component labelling → raw door candidate bboxes.
4. For each candidate, search the geometric gap list for the best matching
   gap and *snap* the candidate bbox to that gap (adopt the gap's exact
   position and extent along the wall).  If no gap matches within the
   search radius, keep the raw U-Net bbox as-is.

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

from core.models import BBox, Opening, OpeningType

logger = logging.getLogger(__name__)

# Door class index in the 4-class checkpoint
_DOOR_CLASS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_door_bboxes(
    door_mask: np.ndarray,
    min_area: int = 300,
) -> List[Tuple[int, int, int, int]]:
    """Connected components on a binary door mask → (x1,y1,x2,y2) bboxes."""
    if door_mask is None or not np.any(door_mask):
        return []

    # Light morph-close to merge fragmented predictions
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(
        (door_mask > 0).astype(np.uint8) * 255,
        cv2.MORPH_CLOSE, k, iterations=1,
    )

    n, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    bboxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bboxes.append((x, y, x + bw, y + bh))

    return bboxes


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
        doors, door_mask = det.detect(img_bgr, geo_gaps)
    """

    def __init__(
        self,
        ckpt_path: str = "weights/epoch_040.pth",
        img_size: int = 1024,
        confidence: float = 0.5,
        min_door_area: int = 300,
        match_margin: int = 40,
        device: Optional[str] = None,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.img_size = img_size
        self.confidence = confidence
        self.min_door_area = min_door_area
        self.match_margin = match_margin
        self.device = device

        self._model = None
        self._num_classes: int = 0
        self._torch_device = None

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
    ) -> Tuple[List[Opening], Optional[np.ndarray]]:
        """
        Run inference and return ``(door_openings, door_mask)``.

        ``door_mask`` is a uint8 H×W array (255 = door pixel) at the
        original image resolution — useful for debug visualisation.

        Each returned Opening has ``source="unet_door"`` and is snapped
        to the nearest geometric gap when one exists within
        ``self.match_margin`` pixels; otherwise the raw U-Net bbox is
        used.
        """
        if self._model is None:
            return [], None

        door_mask = self._run_inference(img_bgr)
        if door_mask is None:
            return [], None

        raw_bboxes = _extract_door_bboxes(door_mask, min_area=self.min_door_area)
        if not raw_bboxes:
            logger.info("UNetDoorDetector: no door blobs found after inference")
            return [], door_mask

        doors: List[Opening] = []
        snapped_count = 0

        for bbox_tuple in raw_bboxes:
            snapped_bbox, was_snapped = _snap_to_gap(
                bbox_tuple, geo_gaps, match_margin=self.match_margin
            )
            if was_snapped:
                snapped_count += 1
            doors.append(Opening(
                bbox=snapped_bbox,
                opening_type=OpeningType.DOOR,
                confidence=0.75,
                source="unet_door",
            ))

        logger.info(
            "UNetDoorDetector: %d door blobs → %d openings (%d snapped to gaps)",
            len(raw_bboxes), len(doors), snapped_count,
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
