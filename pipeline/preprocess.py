"""Image preprocessing helpers: deskew and floorplan-extent cropping."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _upscale_to_min_long_side(
    img: np.ndarray, min_long_side: int = 1200
) -> np.ndarray:
    """Upscale *img* so its longest side is at least *min_long_side* px.

    Detection (U-Net, LSD, OCR) is unreliable on small inputs, so any image
    whose longest dimension is below the threshold is enlarged — preserving
    aspect ratio — before anything else touches it.  Images already at or above
    the threshold are returned unchanged (we never downscale here).
    Cubic interpolation keeps thin wall strokes reasonably crisp.
    """
    if img is None or img.size == 0 or min_long_side <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side >= min_long_side:
        return img
    scale = min_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    logger.info(
        "Upscale: %dx%d → %dx%d (longest side %d → %d px, ×%.2f)",
        w, h, new_w, new_h, long_side, max(new_w, new_h), scale,
    )
    return upscaled


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
    Auto-crop the image to the bounding box of the biggest ink cluster.

    Steps:
      1. Convert to grayscale.
      2. Gaussian blur + Otsu binarise (auto-invert so ink = 255).
      3. Light close to connect broken wall outlines, then connected components.
      4. Seed with the largest significant component, then iteratively absorb
         every other significant component whose bbox is within ``margin`` of the
         growing union — so detached drawing parts (balconies, fixtures,
         annotations) join the crop while a far-separated legend / title block
         stays out.  This crops to the whole drawing cluster, not just the
         largest connected wall loop.
      5. Crop the union bbox with padding.
      6. Pad to a square with white space.

    If detection fails (no component found, or it covers nearly the entire
    image), the original image is returned unchanged.

    Args:
        img: BGR image (as loaded by cv2.imread).
        padding: Extra pixels added around the detected bounding box.
        min_fill: Minimum seed component area / image area ratio to accept.

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

    # Light close so a wall outline broken by small gaps stays one component.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        logger.debug("_crop_to_floorplan: no foreground components — skipping crop")
        return img

    h_img, w_img = img.shape[:2]
    image_area = h_img * w_img

    # Keep only components above a small speckle floor — pure noise must never
    # drag the crop outward.
    speckle_floor = max(1, int(image_area * 0.0002))
    comps = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < speckle_floor:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        comps.append((area, x, y, x + w, y + h))

    if not comps:
        logger.debug("_crop_to_floorplan: no significant components — skipping crop")
        return img

    # Seed with the largest component; it must clear min_fill.
    comps.sort(key=lambda c: c[0], reverse=True)
    seed = comps[0]
    if seed[0] / image_area < min_fill:
        logger.debug("_crop_to_floorplan: largest component below min_fill — skipping crop")
        return img

    ux1, uy1, ux2, uy2 = seed[1], seed[2], seed[3], seed[4]

    # Absorb nearby significant components into the union bbox.  ``margin`` is
    # the max whitespace gap (per axis) tolerated between drawing parts.
    margin = max(8, int(round(min(h_img, w_img) * 0.04)))
    pool = comps[1:]
    changed = True
    while changed:
        changed = False
        remaining = []
        for c in pool:
            _, cx1, cy1, cx2, cy2 = c
            gap_x = max(0, ux1 - cx2, cx1 - ux2)
            gap_y = max(0, uy1 - cy2, cy1 - uy2)
            if gap_x <= margin and gap_y <= margin:
                ux1, uy1 = min(ux1, cx1), min(uy1, cy1)
                ux2, uy2 = max(ux2, cx2), max(uy2, cy2)
                changed = True
            else:
                remaining.append(c)
        pool = remaining

    x, y = ux1, uy1
    w, h = ux2 - ux1, uy2 - uy1

    # Skip crop when the cluster already covers almost the full image
    # (>= 95% in both dimensions) to avoid pointless padding
    if w >= 0.95 * w_img and h >= 0.95 * h_img:
        logger.debug("_crop_to_floorplan: cluster fills image — skipping crop")
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

    # Pad to square with white background
    ch, cw = cropped.shape[:2]
    side = max(ch, cw)
    pad_top = (side - ch) // 2
    pad_left = (side - cw) // 2
    square = np.full((side, side, cropped.shape[2]), 255, dtype=cropped.dtype)
    square[pad_top:pad_top + ch, pad_left:pad_left + cw] = cropped
    logger.info("_crop_to_floorplan: padded to %dx%d square", side, side)
    return square


