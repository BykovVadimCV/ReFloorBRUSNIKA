"""Image preprocessing helpers: deskew and floorplan-extent cropping."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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


def _crop_to_floorplan(
    img: np.ndarray,
    padding: int = 20,
    min_fill: float = 0.01,
) -> np.ndarray:
    """
    Auto-crop the image to the bounding box of the largest ink structure.

    Mirrors the logic in utils/crop_floorplan.py:
      1. Convert to grayscale.
      2. Gaussian blur + Otsu binarise (auto-invert so ink = 255).
      3. Morphological close to connect wall outlines.
      4. Find the largest connected component that passes min_fill.
      5. Crop with padding.

    If detection fails (no component found, or component covers nearly
    the entire image), the original image is returned unchanged.

    Args:
        img: BGR image (as loaded by cv2.imread).
        padding: Extra pixels added around the detected bounding box.
        min_fill: Minimum component area / image area ratio to accept.

    Returns:
        Cropped BGR image, or the original if cropping fails.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu auto-threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure ink (dark structures) = 255, background = 0
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    # Close small gaps so wall outlines connect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        logger.debug("_crop_to_floorplan: no foreground components — skipping crop")
        return img

    image_area = img.shape[0] * img.shape[1]
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
    ranked = np.argsort(areas)[::-1]

    chosen_bbox = None
    for rank_idx in ranked:
        label_id = rank_idx + 1
        area = int(areas[rank_idx])
        if area / image_area < min_fill:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        chosen_bbox = (x, y, w, h)
        break

    if chosen_bbox is None:
        logger.debug("_crop_to_floorplan: largest component below min_fill — skipping crop")
        return img

    x, y, w, h = chosen_bbox

    # Skip crop when the component already covers almost the full image
    # (>= 95% in both dimensions) to avoid pointless padding
    h_img, w_img = img.shape[:2]
    if w >= 0.95 * w_img and h >= 0.95 * h_img:
        logger.debug("_crop_to_floorplan: component fills image — skipping crop")
        return img

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)

    cropped = img[y1:y2, x1:x2]
    logger.info(
        "_crop_to_floorplan: cropped %dx%d → %dx%d (bbox x=%d y=%d w=%d h=%d)",
        w_img, h_img, cropped.shape[1], cropped.shape[0], x, y, w, h,
    )
    return cropped


