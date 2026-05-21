"""Wall rectangle data type plus shared mask-cleanup utilities."""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from detection.wall_params import WallParams


@dataclass
class WallRectangle:
    """Axis-oriented rectangle approximating part of the wall mask."""

    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    area: int
    outline: List[Tuple[int, int]]


def morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """Close small gaps, open thin connections, drop tiny components."""
    kernel_small = np.ones(
        (WallParams.MORPH_KERNEL_SIZE, WallParams.MORPH_KERNEL_SIZE), np.uint8
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel_small,
        iterations=WallParams.MORPH_CLOSE_ITERATIONS,
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel_small,
        iterations=WallParams.MORPH_OPEN_ITERATIONS,
    )
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > WallParams.MORPH_COMPONENT_MIN_AREA:
            cleaned[labels == i] = 255
    return cleaned


def refine_wall_mask_by_enclosed_regions(
    wall_mask: np.ndarray,
    img: np.ndarray,
    fill_threshold: float = 0.50,
    black_threshold: int = 60,
    min_region_area: int = 200,
) -> np.ndarray:
    """Snap the wall mask to enclosed regions bounded by black outlines.

    For each interior region bounded by dark outlines: if at least
    *fill_threshold* of its area was wall-colored, fill it entirely; otherwise
    clear stray wall pixels from it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape[:2]

    outline_pixels = (gray < black_threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    outline_pixels = cv2.dilate(outline_pixels, kernel, iterations=1)

    non_outline = cv2.bitwise_not(outline_pixels)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        non_outline, connectivity=4
    )

    border_labels = set()
    border_labels.update(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[h - 1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, w - 1]))
    border_labels.discard(0)

    refined = wall_mask.copy()
    wall_binary = (wall_mask > 0)

    for label_id in range(1, num_labels):
        if label_id in border_labels:
            continue

        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_region_area:
            continue

        region_mask = (labels == label_id)
        wall_pixels_in_region = np.count_nonzero(wall_binary & region_mask)
        coverage = wall_pixels_in_region / area

        if coverage >= fill_threshold:
            refined[region_mask] = 255
        else:
            refined[region_mask] = 0

    return refined
