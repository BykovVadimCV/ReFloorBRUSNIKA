#!/usr/bin/env python3
"""
detect_unet.py — U-Net semantic segmentation → bounding box extraction.

Loads a trained U-Net checkpoint, runs inference on floorplan images,
post-processes the predicted semantic masks to extract individual
bounding boxes for walls, doors, and windows, then outputs annotated
images and optional JSON.

Usage
-----
    python detect_unet.py \
        --checkpoint  best_unet.pth \
        --images      path/to/images/ \
        --save-dir    results/ \
        --save-json
"""

import argparse
import json
import os
import sys
import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Config — must match training
# ═══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = ["background", "wall", "window", "door"]
NUM_CLASSES = 4
ENCODER = "resnet34"

PALETTE_RGB = np.array([
    [0, 0, 0],        # 0 background
    [200, 0, 0],      # 1 wall
    [0, 200, 0],      # 2 window
    [0, 160, 200],    # 3 door
], dtype=np.uint8)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Visualization colors (BGR for OpenCV)
COLOR_WALL = (0, 180, 0)       # green
COLOR_WINDOW = (200, 120, 0)   # blue-ish
COLOR_DOOR = (200, 160, 0)     # cyan


# ═══════════════════════════════════════════════════════════════════════════════
# Model helpers (same as test.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = NUM_CLASSES, encoder: str = ENCODER):
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


def load_checkpoint(model, ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        epoch = state.get("epoch", "?")
        miou = state.get("best_val_iou", None)
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def get_val_transform(img_size: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def collect_images(directory: str) -> List[str]:
    return sorted(
        p for p in glob.glob(os.path.join(directory, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Greedy largest-rectangle decomposition
# (ported from walldetector.py:745-936, standalone — no class deps)
# ═══════════════════════════════════════════════════════════════════════════════

def _largest_rectangle_in_mask(
    mask_bool: np.ndarray,
) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Find the largest axis-aligned rectangle of True pixels.

    Uses a histogram-based stack algorithm (O(H*W) per call).
    Returns (x1, y1, x2, y2, area) or None if no rectangle found.
    """
    h, w = mask_bool.shape
    heights = np.zeros(w, dtype=np.int32)

    best_area = 0
    best_coords = None

    for i in range(h):
        row = mask_bool[i]
        heights = heights + row.astype(np.int32)
        heights[~row] = 0

        stack: List[int] = []
        j = 0
        while j <= w:
            cur_h = heights[j] if j < w else 0  # sentinel
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


def build_unified_wall_structure(
    wall_bboxes: List[Tuple[int, int, int, int]],
    door_bboxes: List[Tuple[int, int, int, int]],
    window_bboxes: List[Tuple[int, int, int, int]],
    img_shape: Tuple[int, ...],
    gap_close: int = 11,
    min_area: int = 100,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """Merge walls + doors + windows into one binary shape, bridge gaps,
    then decompose back into minimal rectangles.

    1. Paint all wall/door/window bboxes onto a single binary canvas.
    2. Morphological close to bridge small gaps at junctions.
    3. Greedy largest-rectangle decomposition → minimal wall bboxes.

    Returns (wall_bboxes, wall_mask) — updated.
    """
    h, w = img_shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)

    # Paint everything that is structurally "wall"
    for x1, y1, x2, y2 in wall_bboxes:
        canvas[y1:y2, x1:x2] = 1
    for x1, y1, x2, y2 in door_bboxes:
        canvas[y1:y2, x1:x2] = 1
    for x1, y1, x2, y2 in window_bboxes:
        canvas[y1:y2, x1:x2] = 1

    # Bridge small gaps with morphological close
    if gap_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (gap_close, gap_close))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k)

    # Decompose back into minimal rectangles
    new_bboxes = extract_wall_bboxes(canvas, min_area=min_area)

    return new_bboxes, canvas


def extract_wall_bboxes(
    wall_mask: np.ndarray,
    min_area: int = 200,
) -> List[Tuple[int, int, int, int]]:
    """
    Decompose a binary wall mask into individual axis-aligned bounding boxes.

    Uses greedy largest-rectangle extraction: repeatedly finds and erases the
    largest rectangle until remaining rectangles are below min_area.
    """
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(wall_mask * 255, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    work = (cleaned > 0).copy()
    bboxes: List[Tuple[int, int, int, int]] = []

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


# ═══════════════════════════════════════════════════════════════════════════════
# Wall post-processing
# ═══════════════════════════════════════════════════════════════════════════════

def remove_fringe_walls(
    bboxes: List[Tuple[int, int, int, int]],
    img_shape: Tuple[int, int],
    edge_margin: int = 10,
    min_thickness: int = 8,
) -> List[Tuple[int, int, int, int]]:
    """
    Remove wall bboxes that sit on the very edge of the image and are thin.
    These are typically artifacts from the segmentation model seeing the
    image boundary as a wall-like edge.
    """
    h, w = img_shape[:2]
    kept = []
    for x1, y1, x2, y2 in bboxes:
        bw = x2 - x1
        bh = y2 - y1
        thickness = min(bw, bh)

        on_edge = (y1 <= edge_margin or y2 >= h - edge_margin or
                   x1 <= edge_margin or x2 >= w - edge_margin)

        if on_edge and thickness < min_thickness:
            continue
        kept.append((x1, y1, x2, y2))
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing filters for doors / windows
# ═══════════════════════════════════════════════════════════════════════════════

def _bbox_iou(a: Tuple[int, int, int, int],
              b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _bbox_contained(inner: Tuple[int, int, int, int],
                    outer: Tuple[int, int, int, int],
                    threshold: float = 0.85) -> bool:
    """True if `inner` is mostly contained within `outer`."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_inner = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if area_inner == 0:
        return True
    return inter / area_inner >= threshold


def remove_nested_openings(
    bboxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    """Remove openings that sit inside larger openings of the same class."""
    if len(bboxes) <= 1:
        return bboxes

    # Sort by area descending
    by_area = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    for i, box in enumerate(by_area):
        nested = False
        for larger in by_area[:i]:
            if _bbox_contained(box, larger, threshold=0.8):
                nested = True
                break
        if not nested:
            keep.append(box)
    return keep


def remove_cross_class_nested(
    doors: List[Tuple[int, int, int, int]],
    windows: List[Tuple[int, int, int, int]],
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """Remove doors nested in windows and vice versa — keep the larger one."""
    all_openings = [(b, "door") for b in doors] + [(b, "window") for b in windows]
    all_openings.sort(key=lambda t: (t[0][2]-t[0][0])*(t[0][3]-t[0][1]), reverse=True)

    discard = set()
    for i in range(len(all_openings)):
        if i in discard:
            continue
        for j in range(i + 1, len(all_openings)):
            if j in discard:
                continue
            if _bbox_contained(all_openings[j][0], all_openings[i][0], threshold=0.7):
                discard.add(j)

    kept_doors = [b for idx, (b, cls) in enumerate(all_openings)
                  if idx not in discard and cls == "door"]
    kept_windows = [b for idx, (b, cls) in enumerate(all_openings)
                    if idx not in discard and cls == "window"]
    return kept_doors, kept_windows


def touches_wall_mask(
    bbox: Tuple[int, int, int, int],
    wall_binary: np.ndarray,
    margin: int = 5,
) -> bool:
    """Check if a bbox border overlaps with wall pixels within `margin` px."""
    x1, y1, x2, y2 = bbox
    h, w = wall_binary.shape[:2]

    # Expand bbox by margin and check for wall pixels along each edge
    for side_slice in [
        # top edge
        wall_binary[max(0, y1-margin):y1+margin, x1:x2],
        # bottom edge
        wall_binary[max(0, y2-margin):min(h, y2+margin), x1:x2],
        # left edge
        wall_binary[y1:y2, max(0, x1-margin):x1+margin],
        # right edge
        wall_binary[y1:y2, max(0, x2-margin):min(w, x2+margin)],
    ]:
        if side_slice.size > 0 and np.any(side_slice):
            return True
    return False


def wall_contact_sides(
    bbox: Tuple[int, int, int, int],
    wall_binary: np.ndarray,
    margin: int = 5,
    min_contact_ratio: float = 0.25,
) -> Tuple[bool, bool, bool, bool]:
    """Return (top, bottom, left, right) booleans for which sides touch walls."""
    x1, y1, x2, y2 = bbox
    h, w = wall_binary.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    sides = []
    for region, length in [
        (wall_binary[max(0, y1-margin):y1+margin, x1:x2], bw),    # top
        (wall_binary[max(0, y2-margin):min(h, y2+margin), x1:x2], bw),  # bottom
        (wall_binary[y1:y2, max(0, x1-margin):x1+margin], bh),    # left
        (wall_binary[y1:y2, max(0, x2-margin):min(w, x2+margin)], bh),  # right
    ]:
        if region.size == 0 or length == 0:
            sides.append(False)
        else:
            contact_px = np.sum(region > 0)
            # Normalize by the edge length
            sides.append(contact_px / length >= min_contact_ratio)

    return tuple(sides)


def filter_standalone_openings(
    bboxes: List[Tuple[int, int, int, int]],
    wall_binary: np.ndarray,
    margin: int = 10,
    min_contact_ratio: float = 0.20,
) -> List[Tuple[int, int, int, int]]:
    """Remove openings that don't have substantial wall contact on any side."""
    kept = []
    for b in bboxes:
        sides = wall_contact_sides(b, wall_binary, margin=margin,
                                   min_contact_ratio=min_contact_ratio)
        if any(sides):
            kept.append(b)
    return kept


def bridge_wall_gaps(
    wall_bboxes: List[Tuple[int, int, int, int]],
    window_bboxes: List[Tuple[int, int, int, int]],
    max_gap: int = 25,
) -> List[Tuple[int, int, int, int]]:
    """
    Extend wall bboxes to bridge small gaps to nearby walls or windows.

    For each wall, check if another wall or window is within max_gap pixels
    along the same axis (collinear). If so, extend the wall to close the gap.
    Two passes to propagate extensions.
    """
    targets = list(wall_bboxes) + list(window_bboxes)

    for _pass in range(2):
        result = [list(b) for b in wall_bboxes]

        for i, w in enumerate(result):
            x1, y1, x2, y2 = w
            ww = x2 - x1
            wh = y2 - y1
            is_horizontal = ww > wh

            for tx1, ty1, tx2, ty2 in targets:
                if (tx1, ty1, tx2, ty2) == tuple(w):
                    continue

                if is_horizontal:
                    overlap = min(y2, ty2) - max(y1, ty1)
                    min_span = min(wh, ty2 - ty1)
                    if min_span <= 0 or overlap < min_span * 0.2:
                        continue
                    if 0 < tx1 - x2 <= max_gap:
                        result[i][2] = tx1
                    if 0 < x1 - tx2 <= max_gap:
                        result[i][0] = tx2
                else:
                    overlap = min(x2, tx2) - max(x1, tx1)
                    min_span = min(ww, tx2 - tx1)
                    if min_span <= 0 or overlap < min_span * 0.2:
                        continue
                    if 0 < ty1 - y2 <= max_gap:
                        result[i][3] = ty1
                    if 0 < y1 - ty2 <= max_gap:
                        result[i][1] = ty2

        wall_bboxes = [tuple(b) for b in result]
        targets = list(wall_bboxes) + list(window_bboxes)

    return wall_bboxes


def snap_wall_endpoints(
    wall_bboxes: List[Tuple[int, int, int, int]],
    max_gap: int = 15,
) -> List[Tuple[int, int, int, int]]:
    """Extend wall endpoints to close small gaps at corners and T-junctions.

    For each wall, check both perpendicular and collinear neighbours.
    - **Perpendicular (L / T)**: if the end of wall A is within *max_gap* of
      the side of wall B (and they overlap on the perpendicular axis),
      extend A so it touches B.
    - **Collinear**: if two co-axial walls nearly touch end-to-end, extend the
      shorter gap side to meet.

    Two passes so extensions propagate once.
    """
    for _pass in range(2):
        result = [list(b) for b in wall_bboxes]
        n = len(result)

        for i in range(n):
            x1, y1, x2, y2 = result[i]
            w_i = x2 - x1
            h_i = y2 - y1
            is_h = w_i > h_i  # horizontal

            for j in range(n):
                if i == j:
                    continue
                jx1, jy1, jx2, jy2 = result[j]

                if is_h:
                    # Wall i is horizontal.
                    # Check perpendicular (vertical) wall j:
                    #   j must overlap i on Y axis
                    ov = min(y2, jy2) - max(y1, jy1)
                    min_span = min(h_i, jy2 - jy1)
                    if min_span > 0 and ov >= min_span * 0.3:
                        # right end of i near left edge of j
                        if 0 < jx1 - x2 <= max_gap:
                            result[i][2] = jx1
                        # left end of i near right edge of j
                        if 0 < x1 - jx2 <= max_gap:
                            result[i][0] = jx2
                        # right end of i near right edge of j (T-junction)
                        if 0 < jx2 - x2 <= max_gap and abs(jx2 - x2) < max_gap:
                            # only if i ends inside j's x-span
                            if jx1 <= x2 <= jx2:
                                result[i][2] = jx2
                        if 0 < x1 - jx1 <= max_gap and jx1 <= x1 <= jx2:
                            result[i][0] = jx1
                else:
                    # Wall i is vertical.
                    ov = min(x2, jx2) - max(x1, jx1)
                    min_span = min(w_i, jx2 - jx1)
                    if min_span > 0 and ov >= min_span * 0.3:
                        if 0 < jy1 - y2 <= max_gap:
                            result[i][3] = jy1
                        if 0 < y1 - jy2 <= max_gap:
                            result[i][1] = jy2
                        if 0 < jy2 - y2 <= max_gap and jy1 <= y2 <= jy2:
                            result[i][3] = jy2
                        if 0 < y1 - jy1 <= max_gap and jy1 <= y1 <= jy2:
                            result[i][1] = jy1

        wall_bboxes = [tuple(b) for b in result]

    return wall_bboxes


def close_parallel_wall_gaps(
    wall_bboxes: List[Tuple[int, int, int, int]],
    max_gap: int = 8,
) -> List[Tuple[int, int, int, int]]:
    """Close small perpendicular gaps between parallel walls.

    Two horizontal walls side-by-side with a small vertical gap: extend both
    edges toward each other so they meet in the middle.  Same for vertical
    walls with a small horizontal gap.  Walls are only adjusted where they
    overlap on the major axis, so neither protrudes beyond its original extent.
    """
    result = [list(b) for b in wall_bboxes]
    n = len(result)

    for i in range(n):
        x1i, y1i, x2i, y2i = result[i]
        wi = x2i - x1i
        hi = y2i - y1i
        is_h_i = wi > hi

        for j in range(i + 1, n):
            x1j, y1j, x2j, y2j = result[j]
            wj = x2j - x1j
            hj = y2j - y1j
            is_h_j = wj > hj

            if is_h_i != is_h_j:
                continue  # not parallel

            if is_h_i:
                # Both horizontal — check Y gap, require X overlap
                x_ov = min(x2i, x2j) - max(x1i, x1j)
                min_len = min(wi, wj)
                if min_len <= 0 or x_ov < min_len * 0.3:
                    continue

                gap = y1j - y2i  # j is below i
                if 0 < gap <= max_gap:
                    mid = (y2i + y1j) // 2
                    result[i][3] = mid + 1  # y2 of i
                    result[j][1] = mid      # y1 of j
                    continue

                gap = y1i - y2j  # i is below j
                if 0 < gap <= max_gap:
                    mid = (y2j + y1i) // 2
                    result[j][3] = mid + 1
                    result[i][1] = mid
            else:
                # Both vertical — check X gap, require Y overlap
                y_ov = min(y2i, y2j) - max(y1i, y1j)
                min_len = min(hi, hj)
                if min_len <= 0 or y_ov < min_len * 0.3:
                    continue

                gap = x1j - x2i  # j is right of i
                if 0 < gap <= max_gap:
                    mid = (x2i + x1j) // 2
                    result[i][2] = mid + 1
                    result[j][0] = mid
                    continue

                gap = x1i - x2j  # i is right of j
                if 0 < gap <= max_gap:
                    mid = (x2j + x1i) // 2
                    result[j][2] = mid + 1
                    result[i][0] = mid

    return [tuple(b) for b in result]


def subtract_windows_from_walls(
    wall_bboxes: List[Tuple[int, int, int, int]],
    window_bboxes: List[Tuple[int, int, int, int]],
    min_overlap: int = 5,
) -> List[Tuple[int, int, int, int]]:
    """
    If a wall bbox overlaps a window bbox, trim the wall so it stops at
    the window edge. Splits the wall into remaining fragments if needed.
    """
    result = list(wall_bboxes)

    for wx1, wy1, wx2, wy2 in window_bboxes:
        next_result = []
        for bx1, by1, bx2, by2 in result:
            # Check overlap
            ix1 = max(bx1, wx1)
            iy1 = max(by1, wy1)
            ix2 = min(bx2, wx2)
            iy2 = min(by2, wy2)

            if ix2 - ix1 < min_overlap or iy2 - iy1 < min_overlap:
                # No meaningful overlap — keep wall as-is
                next_result.append((bx1, by1, bx2, by2))
                continue

            # Wall overlaps window — split wall into up to 4 fragments
            # (parts of the wall bbox that don't overlap the window)
            bw = bx2 - bx1
            bh = by2 - by1
            is_horizontal = bw > bh

            if is_horizontal:
                # Keep left fragment
                if bx1 < wx1:
                    next_result.append((bx1, by1, wx1, by2))
                # Keep right fragment
                if bx2 > wx2:
                    next_result.append((wx2, by1, bx2, by2))
            else:
                # Keep top fragment
                if by1 < wy1:
                    next_result.append((bx1, by1, bx2, wy1))
                # Keep bottom fragment
                if by2 > wy2:
                    next_result.append((bx1, wy2, bx2, by2))

        result = next_result

    return result


def validate_doors(
    door_bboxes: List[Tuple[int, int, int, int]],
    wall_binary: np.ndarray,
    max_aspect: float = 12.0,
    margin: int = 10,
) -> List[Tuple[int, int, int, int]]:
    """
    Validate doors using gap.py-inspired checks:
      1. Aspect ratio: reject extremely elongated (>max_aspect)
      2. Wall contact: must touch walls on at least 1 side
      3. Dead-end rejection: must NOT touch walls on all 4 sides (enclosed)
    """
    valid = []
    for box in door_bboxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / min(w, h)
        if aspect > max_aspect:
            continue

        # Wall contact check
        top, bottom, left, right = wall_contact_sides(
            box, wall_binary, margin=margin)
        n_sides = sum([top, bottom, left, right])

        # Must touch at least 1 wall, reject fully enclosed (4 sides)
        if n_sides == 0 or n_sides == 4:
            continue

        valid.append(box)
    return valid


def validate_windows(
    window_bboxes: List[Tuple[int, int, int, int]],
    wall_binary: np.ndarray,
    min_aspect: float = 1.5,
    max_aspect: float = 20.0,
    margin: int = 10,
) -> List[Tuple[int, int, int, int]]:
    """
    Validate windows:
      1. Aspect ratio: must be somewhat elongated
      2. Must touch wall on at least 1 side
    """
    valid = []
    for box in window_bboxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / min(w, h)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        if not touches_wall_mask(box, wall_binary, margin=margin):
            continue

        valid.append(box)
    return valid


def clip_openings_to_enclosed_regions(
    opening_bboxes: List[Tuple[int, int, int, int]],
    image_rgb: np.ndarray,
    min_cover: float = 0.80,
    outline_threshold: int = 80,
    gap_bridge: int = 3,
) -> List[Tuple[int, int, int, int]]:
    """Clip opening bboxes to enclosed outline regions they fully contain.

    For each opening bbox, find every enclosed white region (between
    black outlines) whose pixels are at least *min_cover* (80%) inside
    the bbox.  The opening is then replaced by the smallest bounding
    box that covers all qualifying regions.
    """
    if not opening_bboxes:
        return []

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    outline = (gray < outline_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (gap_bridge, gap_bridge))
    outline_closed = cv2.dilate(outline, kernel, iterations=1)
    white_closed = 255 - outline_closed

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        white_closed, connectivity=4)

    # Pre-compute region bboxes and areas
    region_info = []  # (lbl, rx1, ry1, rx2, ry2, area)
    for lbl in range(1, n_labels):
        rx = stats[lbl, cv2.CC_STAT_LEFT]
        ry = stats[lbl, cv2.CC_STAT_TOP]
        rw = stats[lbl, cv2.CC_STAT_WIDTH]
        rh = stats[lbl, cv2.CC_STAT_HEIGHT]
        r_area = stats[lbl, cv2.CC_STAT_AREA]
        if r_area > 0:
            region_info.append((lbl, rx, ry, rx + rw, ry + rh, r_area))

    result = []
    for ox1, oy1, ox2, oy2 in opening_bboxes:
        if ox2 <= ox1 or oy2 <= oy1:
            continue

        # Find all regions entirely (>=min_cover) inside this bbox
        clip_x1, clip_y1, clip_x2, clip_y2 = None, None, None, None

        for lbl, rx1, ry1, rx2, ry2, r_area in region_info:
            # Quick bbox overlap check
            ix1 = max(ox1, rx1)
            iy1 = max(oy1, ry1)
            ix2 = min(ox2, rx2)
            iy2 = min(oy2, ry2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            # Count region pixels inside the opening bbox
            covered = int((labels[iy1:iy2, ix1:ix2] == lbl).sum())
            if r_area > 0 and covered / r_area >= min_cover:
                if clip_x1 is None:
                    clip_x1, clip_y1 = rx1, ry1
                    clip_x2, clip_y2 = rx2, ry2
                else:
                    clip_x1 = min(clip_x1, rx1)
                    clip_y1 = min(clip_y1, ry1)
                    clip_x2 = max(clip_x2, rx2)
                    clip_y2 = max(clip_y2, ry2)

        if clip_x1 is not None:
            result.append((clip_x1, clip_y1, clip_x2, clip_y2))
        else:
            result.append((ox1, oy1, ox2, oy2))

    return result


def merge_wall_bboxes(
    bboxes: List[Tuple[int, int, int, int]],
    img_shape: Tuple[int, int],
) -> List[np.ndarray]:
    """
    Merge overlapping/adjacent wall bboxes into continuous rectilinear shapes.

    Paints all bboxes as filled rectangles onto a blank mask, then extracts
    contours — adjacent rectangles fuse into single shapes with only
    horizontal/vertical edges.
    """
    if not bboxes:
        return []

    h, w = img_shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, -1)

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


# ═══════════════════════════════════════════════════════════════════════════════
# Connected-component extraction for doors / windows
# ═══════════════════════════════════════════════════════════════════════════════

def extract_opening_bboxes(
    class_mask: np.ndarray,
    min_area: int = 100,
) -> List[Tuple[int, int, int, int]]:
    """
    Extract bounding boxes from a binary class mask via connected components.
    """
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(class_mask * 255, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    bboxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num_labels):  # skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((x, y, x + w, y + h))

    return bboxes


def merge_nearby_windows(
    window_bboxes: List[Tuple[int, int, int, int]],
    wall_mask: np.ndarray,
    max_gap: int = 20,
) -> List[Tuple[int, int, int, int]]:
    """Merge nearby window bboxes that have no wall between them.

    For each pair of windows within *max_gap* pixels, check whether
    the gap region between them contains any wall pixels.  If not,
    merge them into one bbox.  Repeats until no more merges possible.
    """
    merged = list(window_bboxes)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(merged):
            j = i + 1
            while j < len(merged):
                ax1, ay1, ax2, ay2 = merged[i]
                bx1, by1, bx2, by2 = merged[j]

                # Check if they're aligned (overlap on one axis) and
                # close on the other axis
                # Vertical alignment: x ranges overlap
                x_overlap = min(ax2, bx2) - max(ax1, bx1)
                # Horizontal alignment: y ranges overlap
                y_overlap = min(ay2, by2) - max(ay1, by1)

                can_merge = False
                if x_overlap > 0:
                    # Vertically stacked — gap on y axis
                    gap = max(ay1, by1) - min(ay2, by2)
                    if 0 <= gap <= max_gap:
                        # Check wall pixels in gap region
                        gx1 = max(ax1, bx1)
                        gx2 = min(ax2, bx2)
                        gy1 = min(ay2, by2)
                        gy2 = max(ay1, by1)
                        if gy2 > gy1 and gx2 > gx1:
                            wall_in_gap = wall_mask[gy1:gy2, gx1:gx2].sum()
                            can_merge = wall_in_gap == 0
                        else:
                            can_merge = True  # touching
                elif y_overlap > 0:
                    # Horizontally adjacent — gap on x axis
                    gap = max(ax1, bx1) - min(ax2, bx2)
                    if 0 <= gap <= max_gap:
                        gx1 = min(ax2, bx2)
                        gx2 = max(ax1, bx1)
                        gy1 = max(ay1, by1)
                        gy2 = min(ay2, by2)
                        if gx2 > gx1 and gy2 > gy1:
                            wall_in_gap = wall_mask[gy1:gy2, gx1:gx2].sum()
                            can_merge = wall_in_gap == 0
                        else:
                            can_merge = True  # touching

                if can_merge:
                    # Union of both bboxes
                    merged[i] = (min(ax1, bx1), min(ay1, by1),
                                 max(ax2, bx2), max(ay2, by2))
                    merged.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    return merged


def refine_walls_by_outline(
    image_rgb: np.ndarray,
    wall_mask: np.ndarray,
    fill_threshold: float = 0.5,
    outline_threshold: int = 80,
    gap_bridge: int = 25,
) -> np.ndarray:
    """Refine wall mask using enclosed regions from the original image.

    1. Threshold image to find dark outlines.
    2. Dilate outlines to bridge single-side openings (doorways etc.),
       then flood-fill from image border to identify exterior.
    3. Everything not exterior and not outline = interior region.
    4. For each interior region: if >= fill_threshold of its pixels are
       wall_mask, fill the whole region; otherwise clear wall pixels.
    5. Wall pixels outside any interior region are kept unchanged.
    """
    h, w = wall_mask.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Dark pixels = outlines
    _, outline_bin = cv2.threshold(gray, outline_threshold, 255,
                                   cv2.THRESH_BINARY_INV)

    # Small close to seal hairline cracks
    k_small = np.ones((3, 3), np.uint8)
    outline_bin = cv2.morphologyEx(outline_bin, cv2.MORPH_CLOSE, k_small,
                                   iterations=1)

    # Dilate outlines to bridge gaps (doorway-width openings) so that
    # regions open on one side become enclosed.
    k_bridge = np.ones((gap_bridge, gap_bridge), np.uint8)
    outline_closed = cv2.dilate(outline_bin, k_bridge, iterations=1)

    # Flood-fill from the border to mark exterior on the closed version.
    # connectedComponents on the white (non-outline) areas: the component
    # touching the border is exterior, everything else is interior.
    white_closed = cv2.bitwise_not(outline_closed)
    num_labels, labels = cv2.connectedComponentsWithStats(
        white_closed, connectivity=4
    )[:2]

    # Find which labels touch the image border → exterior
    border_labels = set()
    border_labels.update(labels[0, :].flat)       # top row
    border_labels.update(labels[h - 1, :].flat)   # bottom row
    border_labels.update(labels[:, 0].flat)        # left col
    border_labels.update(labels[:, w - 1].flat)    # right col
    border_labels.discard(0)  # 0 = outline itself

    # Now find interior regions using the ORIGINAL (non-dilated) outlines
    # so region boundaries are precise.
    white_orig = cv2.bitwise_not(outline_bin)
    num_orig, labels_orig, stats_orig = cv2.connectedComponentsWithStats(
        white_orig, connectivity=4
    )[:3]

    # Map each original-resolution label to interior/exterior:
    # sample its centroid in the closed-outline label map.
    refined = wall_mask.copy()

    # Debug visualisation: each interior region gets a unique colour
    debug_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Generate distinct colours via golden-angle hue spacing
    _region_colors = []
    for i in range(num_orig):
        hue = int((i * 137.508) % 180)
        _region_colors.append(
            cv2.cvtColor(np.uint8([[[hue, 200, 220]]]),
                         cv2.COLOR_HSV2BGR)[0, 0].tolist()
        )

    for lbl in range(1, num_orig):
        region_area = stats_orig[lbl, cv2.CC_STAT_AREA]
        if region_area < 4:
            continue

        region = (labels_orig == lbl)

        # Check if this region is exterior: either its centroid maps to
        # a border component in the dilated version, OR the region itself
        # touches the image border in the original resolution.
        ry = stats_orig[lbl, cv2.CC_STAT_TOP]
        rx = stats_orig[lbl, cv2.CC_STAT_LEFT]
        rh = stats_orig[lbl, cv2.CC_STAT_HEIGHT]
        rw = stats_orig[lbl, cv2.CC_STAT_WIDTH]
        touches_border = (ry <= 1 or rx <= 1 or
                          ry + rh >= h - 1 or rx + rw >= w - 1)
        if touches_border:
            continue  # exterior region — leave wall pixels as-is

        cy = ry + rh // 2
        cx = rx + rw // 2
        cy = min(cy, h - 1)
        cx = min(cx, w - 1)
        closed_lbl = labels[cy, cx]
        if closed_lbl in border_labels:
            continue  # exterior region — leave wall pixels as-is

        # This is an interior region — colour it on the debug image
        debug_img[region] = _region_colors[lbl]

        # Interior region: apply 50% rule
        wall_in_region = int((wall_mask[region] > 0).sum())
        if region_area == 0:
            continue
        ratio = wall_in_region / region_area
        if ratio >= fill_threshold:
            refined[region] = 1   # fill entire region
        else:
            refined[region] = 0   # clear wall pixels

    return refined, debug_img


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(PALETTE_RGB):
        rgb[mask == cls_id] = color
    return rgb


def overlay_prediction(image_rgb: np.ndarray, mask: np.ndarray,
                       alpha: float = 0.45) -> np.ndarray:
    from PIL import Image
    h_orig, w_orig = image_rgb.shape[:2]
    colour_rgb = colorize_mask(mask)
    colour_pil = Image.fromarray(colour_rgb).resize(
        (w_orig, h_orig), resample=Image.NEAREST)
    mask_pil = Image.fromarray(mask.astype(np.uint8)).resize(
        (w_orig, h_orig), resample=Image.NEAREST)

    base_pil = Image.fromarray(image_rgb)
    blended = Image.blend(base_pil, colour_pil, alpha=alpha)

    fg_mask = np.array(mask_pil) != 0
    result = np.array(base_pil)
    result[fg_mask] = np.array(blended)[fg_mask]
    return result


def draw_bboxes(
    image_bgr: np.ndarray,
    wall_bboxes: List[Tuple[int, int, int, int]],
    door_bboxes: List[Tuple[int, int, int, int]],
    window_bboxes: List[Tuple[int, int, int, int]],
    wall_contours: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Draw labeled bounding boxes (and optional wall contours) on a copy."""
    vis = image_bgr.copy()

    if wall_contours is not None:
        # Draw filled contours with transparency
        overlay = vis.copy()
        cv2.drawContours(overlay, wall_contours, -1, COLOR_WALL, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        cv2.drawContours(vis, wall_contours, -1, COLOR_WALL, 2, cv2.LINE_AA)
    else:
        for x1, y1, x2, y2 in wall_bboxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_WALL, 2, cv2.LINE_AA)

    for x1, y1, x2, y2 in window_bboxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_WINDOW, 2, cv2.LINE_AA)
        cv2.putText(vis, "window", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WINDOW, 1, cv2.LINE_AA)

    for x1, y1, x2, y2 in door_bboxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_DOOR, 2, cv2.LINE_AA)
        cv2.putText(vis, "door", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DOOR, 1, cv2.LINE_AA)

    return vis


def display_results(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_vis_rgb: np.ndarray,
    fname: str,
) -> None:
    """Show original, mask, and bbox visualization side-by-side."""
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    h_orig, w_orig = image_rgb.shape[:2]
    colour_rgb = colorize_mask(mask)
    colour_resized = np.array(
        Image.fromarray(colour_rgb).resize((w_orig, h_orig), resample=Image.NEAREST)
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(fname, fontsize=12)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(colour_resized)
    axes[1].set_title("Segmentation mask")
    axes[1].axis("off")

    axes[2].imshow(bbox_vis_rgb)
    axes[2].set_title(f"Extracted bboxes")
    axes[2].axis("off")

    patches = [
        mpatches.Patch(color=tuple(c / 255 for c in PALETTE_RGB[i]),
                       label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Reusable pipeline entry point (called from pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════

_cached_model = None
_cached_ckpt_path = None
_cached_device = None


def _get_or_load_model(checkpoint_path: str, device: torch.device):
    """Load model once and cache it for subsequent calls."""
    global _cached_model, _cached_ckpt_path, _cached_device
    if (_cached_model is not None
            and _cached_ckpt_path == checkpoint_path
            and _cached_device == device):
        return _cached_model
    model = build_model()
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    _cached_model = model
    _cached_ckpt_path = checkpoint_path
    _cached_device = device
    return model


def release_model():
    """Free the cached U-Net model to reclaim RAM."""
    global _cached_model, _cached_ckpt_path, _cached_device
    _cached_model = None
    _cached_ckpt_path = None
    _cached_device = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_unet_pipeline(
    image_rgb: np.ndarray,
    checkpoint_path: str,
    device: Optional[str] = None,
    img_size: int = 1024,
    confidence: float = 0.5,
    min_wall_area: int = 100,
    min_door_area: int = 300,
    min_window_area: int = 300,
) -> Optional[dict]:
    """Run U-Net segmentation + post-processing on a single image.

    Returns a dict with keys:
        wall_bboxes:    List[(x1,y1,x2,y2)]
        wall_contours:  List[np.ndarray]   (merged contour polygons)
        wall_mask:      np.ndarray (H,W) uint8 binary
        door_bboxes:    List[(x1,y1,x2,y2)]
        window_bboxes:  List[(x1,y1,x2,y2)]
        pred_mask:      np.ndarray (H,W) uint8, class indices

    Returns None if the model cannot be loaded or inference fails.
    """
    dev = torch.device(device if device else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        model = _get_or_load_model(checkpoint_path, dev)
    except Exception as e:
        logger.warning("U-Net model load failed: %s", e)
        return None

    h_orig, w_orig = image_rgb.shape[:2]
    transform = get_val_transform(img_size)

    augmented = transform(image=image_rgb)
    inp = augmented["image"].unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        max_prob, pred_idx = probs.max(dim=1)
        pred = pred_idx.squeeze(0).cpu().numpy()
        low_conf = max_prob.squeeze(0).cpu().numpy() < confidence
        pred[low_conf] = 0

    del inp, logits, probs, max_prob, pred_idx

    pred_full = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig),
                           interpolation=cv2.INTER_NEAREST)

    wall_mask = (pred_full == 1).astype(np.uint8)
    window_mask = (pred_full == 2).astype(np.uint8)
    door_mask = (pred_full == 3).astype(np.uint8)

    _close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, _close_k)

    _inv = (1 - wall_mask).astype(np.uint8)
    _n, _lbls, _stats, _ = cv2.connectedComponentsWithStats(_inv, 8)
    for _l in range(1, _n):
        if _stats[_l, cv2.CC_STAT_AREA] <= 1000:
            wall_mask[_lbls == _l] = 1

    wall_mask, _ = refine_walls_by_outline(image_rgb, wall_mask)

    wall_bboxes = extract_wall_bboxes(wall_mask, min_area=min_wall_area)
    window_bboxes = extract_opening_bboxes(window_mask,
                                            min_area=min_window_area)
    door_bboxes = extract_opening_bboxes(door_mask, min_area=min_door_area)

    window_bboxes = merge_nearby_windows(window_bboxes, wall_mask, max_gap=20)

    wall_binary = wall_mask * 255
    wall_bboxes = remove_fringe_walls(wall_bboxes, image_rgb.shape)
    wall_bboxes = bridge_wall_gaps(wall_bboxes, window_bboxes, max_gap=15)
    wall_bboxes = snap_wall_endpoints(wall_bboxes, max_gap=15)
    wall_bboxes = close_parallel_wall_gaps(wall_bboxes, max_gap=8)
    wall_bboxes = subtract_windows_from_walls(wall_bboxes, window_bboxes)
    wall_contours = merge_wall_bboxes(wall_bboxes, image_rgb.shape)

    door_bboxes = remove_nested_openings(door_bboxes)
    window_bboxes = remove_nested_openings(window_bboxes)
    door_bboxes, window_bboxes = remove_cross_class_nested(
        door_bboxes, window_bboxes)
    door_bboxes = filter_standalone_openings(door_bboxes, wall_binary)
    window_bboxes = filter_standalone_openings(window_bboxes, wall_binary)
    door_bboxes = validate_doors(door_bboxes, wall_binary)
    window_bboxes = validate_windows(window_bboxes, wall_binary)

    window_bboxes = clip_openings_to_enclosed_regions(window_bboxes,
                                                       image_rgb)
    door_bboxes = clip_openings_to_enclosed_regions(door_bboxes, image_rgb)

    return {
        "wall_bboxes": wall_bboxes,
        "wall_contours": wall_contours,
        "wall_mask": wall_mask,
        "door_bboxes": door_bboxes,
        "window_bboxes": window_bboxes,
        "pred_mask": pred_full,
    }


import logging
logger = logging.getLogger(__name__)


def is_outline_style(image_rgb: np.ndarray, wall_bboxes, white_threshold=200):
    """Check if wall bboxes sit on white/near-white background (outline style)
    vs solid non-white color (solid style).

    Returns True for outline-style plans, False for solid/color plans.
    """
    if not wall_bboxes:
        return False

    ratios = []
    for x1, y1, x2, y2 in wall_bboxes:
        region = image_rgb[y1:y2, x1:x2]
        if region.size == 0:
            continue
        gray = np.mean(region, axis=2)
        white_ratio = np.mean(gray > white_threshold)
        ratios.append(white_ratio)

    if not ratios:
        return False

    median_white = np.median(ratios)
    # Outline-style: walls are drawn as thin lines on white background,
    # so most of the bbox interior is white.
    return median_white > 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    from PIL import Image
    parser = argparse.ArgumentParser(
        description="U-Net floorplan detection: semantic mask -> bounding boxes",
    )
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--images", "-i", required=True,
                        help="Directory containing input images")
    parser.add_argument("--img-size", type=int, default=1024,
                        help="Model input size (default: 1024)")
    parser.add_argument("--save-dir", "-s", default="unet_results",
                        help="Output directory for results (default: unet_results)")
    parser.add_argument("--save-json", action="store_true",
                        help="Save per-image JSON with bbox coordinates")
    parser.add_argument("--min-wall-area", type=int, default=100,
                        help="Min rectangle area for walls (default: 500)")
    parser.add_argument("--min-door-area", type=int, default=300,
                        help="Min blob area for doors (default: 100)")
    parser.add_argument("--min-window-area", type=int, default=300,
                        help="Min blob area for windows (default: 300)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Min softmax probability to assign a class; "
                             "below this the pixel becomes background (default: 0.5)")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip matplotlib display")
    parser.add_argument("--device", default=None,
                        help="Force device (auto-detected if omitted)")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model()
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # ── Images ────────────────────────────────────────────────────────────
    image_paths = collect_images(args.images)
    if not image_paths:
        print(f"No images found in '{args.images}'")
        sys.exit(1)
    print(f"Found {len(image_paths)} image(s) in '{args.images}'")

    # ── Output dir ────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving results to '{args.save_dir}'")

    # ── Transform ─────────────────────────────────────────────────────────
    transform = get_val_transform(args.img_size)

    # ── Inference + post-processing loop ──────────────────────────────────
    for idx, img_path in enumerate(image_paths, 1):
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]

        # Read image
        try:
            raw_rgb = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"  [{idx}/{len(image_paths)}] SKIP: {fname} -- {e}")
            continue

        h_orig, w_orig = raw_rgb.shape[:2]

        # Preprocess + inference
        augmented = transform(image=raw_rgb)
        inp = augmented["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_idx = probs.max(dim=1)
            pred = pred_idx.squeeze(0).cpu().numpy()
            # Zero out low-confidence pixels (set to background)
            low_conf = max_prob.squeeze(0).cpu().numpy() < args.confidence
            pred[low_conf] = 0

        # Resize prediction to original image size
        pred_full = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig),
                               interpolation=cv2.INTER_NEAREST)

        # Extract per-class binary masks
        wall_mask = (pred_full == 1).astype(np.uint8)
        window_mask = (pred_full == 2).astype(np.uint8)
        door_mask = (pred_full == 3).astype(np.uint8)

        # Morphological close: fill small dents / concavities in wall edges
        _close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, _close_k)

        # Fill small holes (<=1000px) enclosed by wall mask
        _inv = (1 - wall_mask).astype(np.uint8)
        _n, _lbls, _stats, _ = cv2.connectedComponentsWithStats(_inv, 8)
        for _l in range(1, _n):
            if _stats[_l, cv2.CC_STAT_AREA] <= 1000:
                wall_mask[_lbls == _l] = 1

        # Refine wall mask using image outlines: find enclosed white
        # regions in the original image, fill those with >=50% wall
        # coverage, clear wall pixels from the rest.
        wall_mask, regions_debug = refine_walls_by_outline(raw_rgb, wall_mask)

        # Post-process: extract raw bounding boxes
        wall_bboxes = extract_wall_bboxes(wall_mask, min_area=args.min_wall_area)
        window_bboxes = extract_opening_bboxes(window_mask,
                                                min_area=args.min_window_area)
        door_bboxes = extract_opening_bboxes(door_mask,
                                              min_area=args.min_door_area)

        # Merge nearby windows with no wall between them
        window_bboxes = merge_nearby_windows(window_bboxes, wall_mask,
                                             max_gap=20)

        raw_counts = (len(wall_bboxes), len(door_bboxes), len(window_bboxes))

        # Wall post-processing
        wall_binary = wall_mask * 255
        wall_bboxes = remove_fringe_walls(wall_bboxes, raw_rgb.shape)
        wall_bboxes = bridge_wall_gaps(wall_bboxes, window_bboxes, max_gap=15)
        wall_bboxes = subtract_windows_from_walls(wall_bboxes, window_bboxes)
        wall_contours = merge_wall_bboxes(wall_bboxes, raw_rgb.shape)

        # Opening filters: remove nested (same-class and cross-class)
        door_bboxes = remove_nested_openings(door_bboxes)
        window_bboxes = remove_nested_openings(window_bboxes)
        door_bboxes, window_bboxes = remove_cross_class_nested(
            door_bboxes, window_bboxes)

        # Remove standalone openings (require substantial wall contact)
        door_bboxes = filter_standalone_openings(door_bboxes, wall_binary)
        window_bboxes = filter_standalone_openings(window_bboxes, wall_binary)

        # Validate geometry
        door_bboxes = validate_doors(door_bboxes, wall_binary)
        window_bboxes = validate_windows(window_bboxes, wall_binary)

        # Clip openings to enclosed outline regions they fill >=80%
        window_bboxes = clip_openings_to_enclosed_regions(window_bboxes,
                                                           raw_rgb)
        door_bboxes = clip_openings_to_enclosed_regions(door_bboxes,
                                                         raw_rgb)

        # Per-class pixel stats
        total_px = pred_full.size
        wall_pct = np.sum(wall_mask) / total_px * 100
        window_pct = np.sum(window_mask) / total_px * 100
        door_pct = np.sum(door_mask) / total_px * 100

        print(f"  [{idx}/{len(image_paths)}] {fname}  ({w_orig}x{h_orig})")
        print(f"    Pixels: wall:{wall_pct:.1f}%  window:{window_pct:.1f}%  "
              f"door:{door_pct:.1f}%")
        print(f"    Walls: {raw_counts[0]} raw -> {len(wall_bboxes)} trimmed -> {len(wall_contours)} merged")
        print(f"    Doors: {raw_counts[1]} raw -> {len(door_bboxes)} filtered")
        print(f"    Windows: {raw_counts[2]} raw -> {len(window_bboxes)} filtered")

        # ── Save overlay ──────────────────────────────────────────────────
        overlay = overlay_prediction(raw_rgb, pred)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.save_dir, f"overlay_{stem}.png"),
                    overlay_bgr)

        # ── Save enclosed-regions debug image ────────────────────────────
        cv2.imwrite(os.path.join(args.save_dir, f"regions_{stem}.png"),
                    regions_debug)

        # ── Save bbox visualization ───────────────────────────────────────
        img_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        bbox_vis = draw_bboxes(img_bgr, wall_bboxes, door_bboxes, window_bboxes,
                               wall_contours=wall_contours)
        cv2.imwrite(os.path.join(args.save_dir, f"bboxes_{stem}.png"), bbox_vis)

        # ── Optional JSON ─────────────────────────────────────────────────
        if args.save_json:
            result = {
                "image": fname,
                "image_size": [w_orig, h_orig],
                "walls": [
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                     "thickness": min(x2 - x1, y2 - y1)}
                    for x1, y1, x2, y2 in wall_bboxes
                ],
                "wall_contours": [
                    cnt.reshape(-1, 2).tolist()
                    for cnt in wall_contours
                ],
                "doors": [
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    for x1, y1, x2, y2 in door_bboxes
                ],
                "windows": [
                    {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    for x1, y1, x2, y2 in window_bboxes
                ],
                "stats": {
                    "wall_px_pct": round(wall_pct, 2),
                    "window_px_pct": round(window_pct, 2),
                    "door_px_pct": round(door_pct, 2),
                    "num_walls": len(wall_bboxes),
                    "num_doors": len(door_bboxes),
                    "num_windows": len(window_bboxes),
                },
            }
            json_path = os.path.join(args.save_dir, f"bboxes_{stem}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=lambda o: int(o))

        # ── Display ───────────────────────────────────────────────────────
        if not args.no_display:
            bbox_vis_rgb = cv2.cvtColor(bbox_vis, cv2.COLOR_BGR2RGB)
            display_results(raw_rgb, pred, bbox_vis_rgb, fname)

    print(f"\nDone. Results saved to '{os.path.abspath(args.save_dir)}'")


def _subprocess_entry():
    """Entry point when invoked as a subprocess by run_unet_subprocess().

    Reads JSON config from stdin, runs the full pipeline in this process,
    writes results as JSON to stdout, then exits (freeing all memory).
    """
    import base64
    import os

    # Redirect fd 1 (C-level stdout) to fd 2 (stderr) so that YOLO/torch
    # C-extension prints don't contaminate the JSON we write at the end.
    # Must happen before any imports that trigger native prints (YOLO, torch).
    real_stdout_fd = os.dup(1)
    os.dup2(2, 1)
    real_stdout = os.fdopen(real_stdout_fd, 'wb')
    sys.stdout = sys.stderr

    raw = sys.stdin.buffer.read()
    cfg = json.loads(raw)

    # Decode image from temp file
    img_path = cfg["image_path"]
    image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    result = run_unet_pipeline(
        image_rgb=image_rgb,
        checkpoint_path=cfg["checkpoint_path"],
        device=cfg.get("device"),
        img_size=cfg.get("img_size", 1024),
        confidence=cfg.get("confidence", 0.5),
        min_wall_area=cfg.get("min_wall_area", 100),
        min_door_area=cfg.get("min_door_area", 300),
        min_window_area=cfg.get("min_window_area", 300),
    )

    if result is None:
        real_stdout.write(json.dumps({"status": "error"}).encode())
        real_stdout.flush()
        return

    # Encode wall_mask as base64 PNG
    _, png_bytes = cv2.imencode(".png", result["wall_mask"])
    mask_b64 = base64.b64encode(png_bytes.tobytes()).decode()

    out = {
        "status": "ok",
        "wall_bboxes": [list(map(int, b)) for b in result["wall_bboxes"]],
        "door_bboxes": [list(map(int, b)) for b in result["door_bboxes"]],
        "window_bboxes": [list(map(int, b)) for b in result["window_bboxes"]],
        "wall_mask_b64": mask_b64,
    }
    real_stdout.write(json.dumps(out, default=lambda o: int(o)).encode())
    real_stdout.flush()


def run_unet_subprocess(
    image_rgb: np.ndarray,
    checkpoint_path: str,
    device: Optional[str] = None,
    img_size: int = 1024,
    confidence: float = 0.5,
    min_wall_area: int = 100,
    min_door_area: int = 300,
    min_window_area: int = 300,
) -> Optional[dict]:
    """Run U-Net pipeline in an isolated subprocess to avoid OOM.

    Same interface as run_unet_pipeline() but spawns a child process
    so that all PyTorch memory is freed by the OS on exit.

    Returns dict with wall_bboxes, door_bboxes, window_bboxes, wall_mask
    or None on failure.
    """
    import base64
    import subprocess
    import tempfile

    # Save image to temp file (child reads it)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    try:
        cv2.imwrite(tmp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        cfg = json.dumps({
            "image_path": tmp_path,
            "checkpoint_path": checkpoint_path,
            "device": device,
            "img_size": img_size,
            "confidence": confidence,
            "min_wall_area": min_wall_area,
            "min_door_area": min_door_area,
            "min_window_area": min_window_area,
        })

        script_path = os.path.abspath(__file__)
        proc = subprocess.run(
            [sys.executable, script_path, "--subprocess-mode"],
            input=cfg.encode(),
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent stderr (debug prints visible)
            timeout=300,
        )

        if proc.returncode != 0:
            logger.warning("U-Net subprocess failed (rc=%d)", proc.returncode)
            return None

        out = json.loads(proc.stdout)
        if out.get("status") != "ok":
            return None

        # Decode wall_mask from base64 PNG
        png_bytes = base64.b64decode(out["wall_mask_b64"])
        wall_mask = cv2.imdecode(
            np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        return {
            "wall_bboxes": [tuple(b) for b in out["wall_bboxes"]],
            "door_bboxes": [tuple(b) for b in out["door_bboxes"]],
            "window_bboxes": [tuple(b) for b in out["window_bboxes"]],
            "wall_mask": wall_mask,
            "wall_contours": [],  # not serialized, regenerated if needed
            "pred_mask": None,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    if "--subprocess-mode" in sys.argv:
        _subprocess_entry()
    else:
        main()
