"""
Gap Bridge Utility v2
=====================
Closes wall gaps in floorplan images using U-Net wall segmentation.

Instead of trying to detect and bridge gaps with Hough lines (brittle),
this approach uses the U-Net wall mask — which recognizes walls semantically —
as the boundary for enclosed region detection.

Algorithm:
  1. Run U-Net segmentation → binary wall mask (wall=255, else=0)
  2. Apply morphological closing to bridge small gaps in the mask
  3. Invert the mask: wall pixels become barriers, non-wall = fillable
  4. Flood-fill from image borders to mark exterior
  5. Remaining unfilled interior = enclosed regions
  6. Visualize: original, raw enclosed regions, U-Net mask,
     closed mask, enclosed regions from closed mask

Usage:
  python utils/gap_bridge.py input.png [--checkpoint weights/epoch_040.pth]
                             [--kernel 7] [--iterations 3]
                             [--output out.png] [--debug]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────
# U-Net wall mask
# ─────────────────────────────────────────────────────────────

def get_unet_wall_mask(image_rgb: np.ndarray, checkpoint_path: str) -> np.ndarray:
    """Run U-Net and return binary wall mask (H, W), 255=wall, 0=other."""
    import torch
    from detect_unet import build_model, load_checkpoint, get_val_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    h_orig, w_orig = image_rgb.shape[:2]
    transform = get_val_transform(1024)
    augmented = transform(image=image_rgb)
    inp = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        max_prob, pred_idx = probs.max(dim=1)
        pred = pred_idx.squeeze(0).cpu().numpy()
        low_conf = max_prob.squeeze(0).cpu().numpy() < 0.5
        pred[low_conf] = 0

    del inp, logits, probs, max_prob, pred_idx

    pred_full = cv2.resize(pred.astype(np.uint8), (w_orig, h_orig),
                           interpolation=cv2.INTER_NEAREST)
    wall_mask = (pred_full == 1).astype(np.uint8) * 255
    return wall_mask


# ─────────────────────────────────────────────────────────────
# Enclosed region detection from a binary mask
# ─────────────────────────────────────────────────────────────

def find_enclosed_regions_from_mask(wall_mask: np.ndarray):
    """Find enclosed regions using a binary wall mask as boundaries.

    wall_mask: (H, W) uint8, 255 = wall/barrier, 0 = fillable space.

    Returns (num_labels, labels) like cv2.connectedComponents.
    Label 0 = background/barriers, 1..N = enclosed regions.
    """
    h, w = wall_mask.shape[:2]

    # Fillable = non-wall pixels
    fillable = (wall_mask == 0).astype(np.uint8) * 255

    # Flood-fill from all border pixels to mark exterior
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

    # Enclosed = still 255 (exterior became 128, walls are 0)
    enclosed = (flood == 255).astype(np.uint8) * 255

    return cv2.connectedComponents(enclosed)


def find_enclosed_from_original(img_bgr: np.ndarray, white_threshold: int = 220):
    """Original approach: enclosed regions from raw image (for comparison)."""
    from detection.walls import _build_enclosed_regions
    return _build_enclosed_regions(img_bgr, white_threshold=white_threshold)


# ─────────────────────────────────────────────────────────────
# Morphological gap closing
# ─────────────────────────────────────────────────────────────

def close_wall_mask(wall_mask: np.ndarray, kernel_size: int = 7,
                    iterations: int = 3) -> np.ndarray:
    """Apply morphological closing to bridge small gaps in the wall mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    closed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel,
                              iterations=iterations)
    return closed


# ─────────────────────────────────────────────────────────────
# Black-line extraction guided by U-Net mask
# ─────────────────────────────────────────────────────────────

def extract_wall_outlines(
    img_bgr: np.ndarray,
    wall_mask: np.ndarray,
    black_threshold: int = 80,
    dilate_mask: int = 10,
) -> np.ndarray:
    """Extract black outline pixels that fall within the U-Net wall region.

    1. Find all "dark" pixels in the original image (grayscale < black_threshold).
    2. Dilate the U-Net wall mask slightly so we don't miss outline edges
       that sit just outside the predicted wall region.
    3. Keep only dark pixels that overlap with the dilated mask.

    Returns a binary mask (H, W) uint8: 255 = wall outline, 0 = other.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    dark_pixels = (gray < black_threshold).astype(np.uint8) * 255

    # Dilate U-Net mask to catch outline pixels near predicted wall edges
    if dilate_mask > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_mask * 2 + 1, dilate_mask * 2 + 1))
        expanded_mask = cv2.dilate(wall_mask, kernel, iterations=1)
    else:
        expanded_mask = wall_mask

    # Intersection: dark pixels within wall regions
    outlines = cv2.bitwise_and(dark_pixels, expanded_mask)

    return outlines


def extract_and_close_outlines(
    img_bgr: np.ndarray,
    wall_mask: np.ndarray,
    black_threshold: int = 80,
    dilate_mask: int = 10,
    close_kernel: int = 5,
    close_iterations: int = 2,
) -> np.ndarray:
    """Extract wall outlines, then close small gaps in the outline.

    Combines extract_wall_outlines() with morphological closing on the
    resulting thin outline to bridge small discontinuities.
    """
    outlines = extract_wall_outlines(img_bgr, wall_mask,
                                     black_threshold, dilate_mask)

    if close_kernel > 0 and close_iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        outlines = cv2.morphologyEx(outlines, cv2.MORPH_CLOSE, kernel,
                                     iterations=close_iterations)
    return outlines


# ─────────────────────────────────────────────────────────────
# Gap detection and bridging
# ─────────────────────────────────────────────────────────────

def _compute_outline_thickness(outline_mask: np.ndarray) -> int:
    """Estimate median outline thickness via distance transform.

    For each outline pixel, distanceTransform gives the distance to the
    nearest non-outline pixel (i.e. the edge).  Pixels at the center of a
    thick line have distance ≈ half_width.  The 80th percentile × 2 gives a
    robust estimate of typical wall-outline width.
    """
    dt = cv2.distanceTransform(outline_mask, cv2.DIST_L2, 5)
    vals = dt[outline_mask > 0]
    if len(vals) == 0:
        return 3
    return max(2, int(round(np.percentile(vals, 80) * 2)))


def _snap_to_outline(px: int, py: int, outline_mask: np.ndarray,
                     radius: int = 12):
    """Return the nearest outline pixel to (px, py) within radius."""
    h, w = outline_mask.shape
    y0 = max(0, py - radius)
    y1 = min(h, py + radius + 1)
    x0 = max(0, px - radius)
    x1 = min(w, px + radius + 1)
    roi = outline_mask[y0:y1, x0:x1]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return px, py
    dists = (xs - (px - x0)) ** 2 + (ys - (py - y0)) ** 2
    idx = int(np.argmin(dists))
    return x0 + int(xs[idx]), y0 + int(ys[idx])


def _find_gaps_for_blob(
    blob_mask: np.ndarray,
    outline_mask: np.ndarray,
    dilation: int = 20,
    check_radius: int = 10,
    min_coverage: float = 0.4,
    max_gap_px: float = 80.0,
) -> list:
    """Walk the contour around a dilated blob, find gaps in the outline.

    Returns list of (gap_length, (x1,y1), (x2,y2)) for each gap found.
    (x1,y1) and (x2,y2) are the contour points at the gap edges (the last
    point with outline before the gap, and the first after).
    """
    h, w = outline_mask.shape

    # Dilate blob so the contour sits outside the blob, near the outline
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
    dilated = cv2.dilate(blob_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    pts = contour.squeeze()
    if pts.ndim < 2 or len(pts) < 10:
        return []

    n = len(pts)

    # For each contour point, check if an outline pixel exists nearby
    near_outline = np.zeros(n, dtype=bool)
    for i in range(n):
        px, py = int(pts[i][0]), int(pts[i][1])
        y0 = max(0, py - check_radius)
        y1 = min(h, py + check_radius + 1)
        x0 = max(0, px - check_radius)
        x1 = min(w, px + check_radius + 1)
        if np.any(outline_mask[y0:y1, x0:x1] > 0):
            near_outline[i] = True

    coverage = np.sum(near_outline) / n
    if coverage < min_coverage:
        return []  # not enough outline around this blob

    # Rotate array so it starts at an outline point
    first = int(np.argmax(near_outline))
    if not near_outline[first]:
        return []
    ro = np.roll(near_outline, -first)
    rp = np.roll(pts, -first, axis=0)

    # Walk linearly, find gap segments (consecutive non-outline points)
    gaps = []
    in_gap = False
    gap_start = -1
    for i in range(n):
        if not ro[i]:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                # Gap from gap_start to i-1
                p1 = rp[gap_start - 1]   # last outline point before gap
                p2 = rp[i]               # first outline point after gap
                dist = float(np.sqrt(
                    (float(p1[0]) - float(p2[0])) ** 2 +
                    (float(p1[1]) - float(p2[1])) ** 2))
                if dist <= max_gap_px:
                    gaps.append((
                        dist,
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                    ))
                in_gap = False

    return gaps


def _dedup_bridges(candidates: list, min_dist: int = 15) -> list:
    """Remove near-duplicate bridge candidates (endpoints within min_dist)."""
    filtered = []
    for c in candidates:
        dist, p1, p2 = c
        is_dup = False
        for f in filtered:
            _, fp1, fp2 = f
            d_aa = np.sqrt((p1[0] - fp1[0]) ** 2 + (p1[1] - fp1[1]) ** 2)
            d_ab = np.sqrt((p1[0] - fp2[0]) ** 2 + (p1[1] - fp2[1]) ** 2)
            d_ba = np.sqrt((p2[0] - fp1[0]) ** 2 + (p2[1] - fp1[1]) ** 2)
            d_bb = np.sqrt((p2[0] - fp2[0]) ** 2 + (p2[1] - fp2[1]) ** 2)
            if ((d_aa < min_dist and d_bb < min_dist) or
                    (d_ab < min_dist and d_ba < min_dist)):
                is_dup = True
                break
        if not is_dup:
            filtered.append(c)
    return filtered


def find_and_draw_bridges(
    outline_mask: np.ndarray,
    wall_mask: np.ndarray,
    max_gap: float = 80.0,
    min_blob_area: int = 100,
    min_coverage: float = 0.4,
    dilation: int = 20,
    check_radius: int = 10,
) -> tuple:
    """Find gaps around uncovered wall blobs and draw bridges.

    Algorithm:
      1. Find enclosed regions from the current outline mask.
      2. Identify wall-mask pixels NOT covered by any enclosed region.
      3. Group uncovered pixels into connected blobs.
      4. For each blob whose contour is mostly (≥min_coverage) surrounded by
         outline, find the gaps.
      5. Snap bridge endpoints to nearest actual outline pixel.
      6. Sort candidates by length (shortest first).
      7. Greedily draw each bridge if it improves wall-mask coverage.

    Returns (bridged_outline, drawn_bridges, thickness).
    """
    h, w = outline_mask.shape

    # Current enclosed regions
    n_enc, labels = find_enclosed_regions_from_mask(outline_mask)
    covered = np.zeros((h, w), dtype=bool)
    for i in range(1, n_enc):
        covered |= (labels == i)
    wall_bool = wall_mask > 0

    total_wall = int(np.sum(wall_bool))
    initial_covered = int(np.sum(covered & wall_bool))
    print(f"  Initial wall coverage: {initial_covered}/{total_wall} "
          f"({initial_covered * 100 / max(1, total_wall):.1f}%)")

    # Uncovered wall → connected blobs
    uncovered = wall_bool & ~covered
    n_blobs, blob_labels = cv2.connectedComponents(
        uncovered.astype(np.uint8) * 255)
    print(f"  Uncovered wall blobs: {n_blobs - 1}")

    # Outline thickness
    thickness = _compute_outline_thickness(outline_mask)
    print(f"  Estimated outline thickness: {thickness} px")

    # Collect candidate bridges from all blobs
    candidates = []
    for blob_id in range(1, n_blobs):
        blob = (blob_labels == blob_id).astype(np.uint8) * 255
        area = int(np.sum(blob > 0))
        if area < min_blob_area:
            continue

        gaps = _find_gaps_for_blob(
            blob, outline_mask, dilation=dilation,
            check_radius=check_radius, min_coverage=min_coverage,
            max_gap_px=max_gap)

        for dist, p1, p2 in gaps:
            # Snap to nearest actual outline pixel
            p1s = _snap_to_outline(p1[0], p1[1], outline_mask)
            p2s = _snap_to_outline(p2[0], p2[1], outline_mask)
            final_dist = np.sqrt(
                (p1s[0] - p2s[0]) ** 2 + (p1s[1] - p2s[1]) ** 2)
            if final_dist > 2:  # skip degenerate
                candidates.append((final_dist, p1s, p2s))

    # Deduplicate and sort shortest first
    candidates = _dedup_bridges(candidates)
    candidates.sort(key=lambda c: c[0])
    print(f"  Candidate bridges: {len(candidates)}")

    # Greedily draw bridges that improve wall coverage
    bridged = outline_mask.copy()
    current_covered = covered.copy()
    drawn = []

    for dist, (x1, y1), (x2, y2) in candidates:
        # Tentatively draw
        test = bridged.copy()
        cv2.line(test, (x1, y1), (x2, y2), 255, thickness, cv2.LINE_AA)

        # Check if coverage improved
        n_new, labels_new = find_enclosed_regions_from_mask(test)
        new_covered = np.zeros((h, w), dtype=bool)
        for i in range(1, n_new):
            new_covered |= (labels_new == i)

        old_wall_cov = int(np.sum(current_covered & wall_bool))
        new_wall_cov = int(np.sum(new_covered & wall_bool))

        if new_wall_cov > old_wall_cov:
            bridged = test
            current_covered = new_covered
            drawn.append((x1, y1, x2, y2))
            print(f"    Bridge ({x1},{y1})->({x2},{y2}) len={dist:.0f}px: "
                  f"coverage +{new_wall_cov - old_wall_cov} px "
                  f"({new_wall_cov * 100 / max(1, total_wall):.1f}%)")

    final_covered = int(np.sum(current_covered & wall_bool))
    print(f"  Final wall coverage: {final_covered}/{total_wall} "
          f"({final_covered * 100 / max(1, total_wall):.1f}%) "
          f"— {len(drawn)} bridge(s) drawn")

    return bridged, drawn, thickness


# ─────────────────────────────────────────────────────────────
# Greedy largest-rectangle decomposition (same as detect_unet.py)
# ─────────────────────────────────────────────────────────────

def _largest_rectangle_in_mask(mask_bool: np.ndarray):
    """Find the largest axis-aligned rectangle of True pixels.

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


def extract_wall_bboxes_from_mask(
    wall_mask: np.ndarray,
    min_area: int = 200,
) -> list:
    """Greedy largest-rectangle extraction from a binary mask.

    Repeatedly finds and erases the largest axis-aligned rectangle
    until remaining rectangles are below min_area.

    Returns list of (x1, y1, x2, y2) tuples.
    """
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
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


# ─────────────────────────────────────────────────────────────
# Grey-fill uncovered wall rectangles
# ─────────────────────────────────────────────────────────────

def _merge_and_reextract(
    bboxes: list,
    img_shape: tuple,
    merge_gap: int = 15,
    min_area: int = 400,
) -> list:
    """Merge nearby bounding boxes into continuous shapes, then re-extract.

    1. Paint all bboxes onto a binary canvas.
    2. Dilate by merge_gap to bridge nearby rectangles.
    3. Find connected components on the dilated canvas.
    4. For each component, re-run greedy largest-rectangle on the ORIGINAL
       (undilated) pixels within that component's bounding box.
    5. Filter out rectangles below min_area.

    Returns list of (x1, y1, x2, y2) tuples.
    """
    h, w = img_shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        canvas[y1:y2, x1:x2] = 255

    if not np.any(canvas):
        return []

    # Dilate to merge nearby rects
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (merge_gap * 2 + 1, merge_gap * 2 + 1))
    dilated = cv2.dilate(canvas, kernel, iterations=1)

    # Connected components on dilated
    n_cc, cc_labels = cv2.connectedComponents(dilated)

    merged = []
    for cc_id in range(1, n_cc):
        # Bounding box of this component
        ys, xs = np.where(cc_labels == cc_id)
        cx1, cy1, cx2, cy2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

        # Extract from original (undilated) canvas within this region
        region = canvas[cy1:cy2, cx1:cx2]
        sub_bboxes = extract_wall_bboxes_from_mask(region, min_area=min_area)

        for sx1, sy1, sx2, sy2 in sub_bboxes:
            merged.append((cx1 + sx1, cy1 + sy1, cx1 + sx2, cy1 + sy2))

    return merged


def find_unenclosed_wall_rects(
    wall_mask: np.ndarray,
    outline_mask: np.ndarray,
    min_area: int = 200,
    merge_gap: int = 15,
    min_merged_area: int = 1000,
) -> list:
    """Find axis-aligned bounding boxes for uncovered wall-mask regions.

    Pipeline:
      1. Find enclosed regions from outline mask.
      2. Mask out covered wall pixels → uncovered wall mask.
      3. Greedy extract axis-aligned rectangles from uncovered mask.
      4. Merge nearby rectangles into continuous shapes (dilate by merge_gap).
      5. Re-extract rectangles from merged shapes.
      6. Filter out small rectangles (< min_merged_area).

    Returns list of dicts:
      {bbox: (x1,y1,x2,y2), area, width, height}
    """
    h, w = wall_mask.shape

    # Enclosed regions from outline
    n_enc, labels = find_enclosed_regions_from_mask(outline_mask)
    covered = np.zeros((h, w), dtype=bool)
    for i in range(1, n_enc):
        covered |= (labels == i)

    # Uncovered wall pixels
    uncovered = ((wall_mask > 0) & ~covered).astype(np.uint8) * 255

    # Greedy rectangle extraction
    raw_bboxes = extract_wall_bboxes_from_mask(uncovered, min_area=min_area)
    print(f"    Raw rectangles: {len(raw_bboxes)}")

    # Merge nearby and re-extract
    merged_bboxes = _merge_and_reextract(
        raw_bboxes, (h, w), merge_gap=merge_gap, min_area=min_merged_area)
    print(f"    After merge+filter (gap={merge_gap}, min={min_merged_area}): "
          f"{len(merged_bboxes)}")

    results = []
    for x1, y1, x2, y2 in merged_bboxes:
        bw = x2 - x1
        bh = y2 - y1
        results.append({
            "bbox": (x1, y1, x2, y2),
            "area": bw * bh,
            "width": bw,
            "height": bh,
        })

    return results


def paint_wall_rects_grey(
    img_bgr: np.ndarray,
    wall_rects: list,
    grey_value: int = 180,
) -> np.ndarray:
    """Paint the wall rectangles grey on a copy of the image."""
    result = img_bgr.copy()
    for wr in wall_rects:
        x1, y1, x2, y2 = wr["bbox"]
        result[y1:y2, x1:x2] = grey_value
    return result


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

def colorize_regions(num_labels: int, labels: np.ndarray,
                     min_area: int = 50) -> np.ndarray:
    """Create an RGB image with each enclosed region in a unique color."""
    from matplotlib.colors import hsv_to_rgb

    h, w = labels.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id in range(1, num_labels):
        region_mask = labels == label_id
        area = int(np.sum(region_mask))
        if area < min_area:
            continue

        hue = (label_id * 0.618033988749895) % 1.0
        rgb = hsv_to_rgb([hue, 0.7, 0.9])
        canvas[region_mask] = [int(c * 255) for c in rgb]

    return canvas


def blend(base_rgb: np.ndarray, overlay_rgb: np.ndarray,
          alpha: float = 0.5) -> np.ndarray:
    """Blend overlay onto base where overlay is non-black."""
    result = base_rgb.copy()
    mask = np.any(overlay_rgb > 0, axis=2)
    result[mask] = (base_rgb[mask] * (1 - alpha) +
                    overlay_rgb[mask] * alpha).astype(np.uint8)
    return result


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def process(
    input_path: str,
    checkpoint_path: str = "weights/epoch_040.pth",
    black_threshold: int = 80,
    dilate_mask: int = 10,
    close_outline_kernel: int = 5,
    close_outline_iter: int = 2,
    max_gap: float = 80.0,
    min_blob_area: int = 100,
    min_coverage: float = 0.4,
    dilation: int = 20,
    check_radius: int = 10,
    merge_gap: int = 15,
    min_merged_area: int = 1000,
    output_path: str = None,
    debug: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        sys.exit(f"Cannot read image: {input_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    # 1. U-Net wall mask
    print(f"Running U-Net ({checkpoint_path})...")
    wall_mask = get_unet_wall_mask(img_rgb, checkpoint_path)
    wall_px = int(np.sum(wall_mask > 0))
    print(f"  Wall mask: {wall_px} wall pixels ({wall_px * 100 / (h * w):.1f}%)")

    # 2. Extract black outlines guided by U-Net mask
    print("Extracting wall outlines...")
    outlines = extract_and_close_outlines(
        img_bgr, wall_mask, black_threshold, dilate_mask,
        close_outline_kernel, close_outline_iter)
    outline_px = int(np.sum(outlines > 0))
    print(f"  Outlines: {outline_px} px")

    # 3. Enclosed regions BEFORE bridging
    n_before, labels_before = find_enclosed_regions_from_mask(outlines)
    print(f"  Enclosed regions before bridging: {n_before - 1}")

    # 4. Find and draw bridges
    print("Finding gaps and drawing bridges...")
    bridged, bridges, thickness = find_and_draw_bridges(
        outlines, wall_mask, max_gap=max_gap,
        min_blob_area=min_blob_area, min_coverage=min_coverage,
        dilation=dilation, check_radius=check_radius)

    # 5. Enclosed regions AFTER bridging
    n_after, labels_after = find_enclosed_regions_from_mask(bridged)
    print(f"  Enclosed regions after bridging: {n_after - 1}")

    # 6. Find uncovered wall rectangles → grey
    print("Extracting unenclosed wall rectangles (greedy largest-rect)...")
    wall_rects = find_unenclosed_wall_rects(
        wall_mask, bridged, min_area=min_blob_area,
        merge_gap=merge_gap, min_merged_area=min_merged_area)
    grey_img = paint_wall_rects_grey(img_bgr, wall_rects, grey_value=180)
    grey_rgb = cv2.cvtColor(grey_img, cv2.COLOR_BGR2RGB)
    total_grey_px = sum(wr["area"] for wr in wall_rects)
    print(f"  Found {len(wall_rects)} wall rectangles "
          f"({total_grey_px} px total)")
    for i, wr in enumerate(wall_rects):
        x1, y1, x2, y2 = wr["bbox"]
        print(f"    [{i}] ({x1},{y1})-({x2},{y2})  "
              f"{wr['width']}x{wr['height']}  area={wr['area']}px")

    # ── Visualize: 3 rows x 3 cols ──
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(
        f"Gap Bridging: {Path(input_path).name}\n"
        f"Before: {n_before-1} regions | After: {n_after-1} regions | "
        f"{len(bridges)} bridge(s) | {len(wall_rects)} grey wall(s)",
        fontsize=14, fontweight="bold"
    )

    # Row 1: Original | Outlines + U-Net mask | Uncovered wall blobs
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Image")

    overlay1 = np.zeros((h, w, 3), dtype=np.uint8)
    overlay1[wall_mask > 0] = [150, 0, 0]
    overlay1[outlines > 0] = [0, 255, 0]
    axes[0, 1].imshow(blend(img_rgb, overlay1, 0.6))
    axes[0, 1].set_title("Wall Outlines (green) + U-Net Mask (red)")

    wall_bool = wall_mask > 0
    covered_before = np.zeros((h, w), dtype=bool)
    for i in range(1, n_before):
        covered_before |= (labels_before == i)
    uncovered = wall_bool & ~covered_before
    overlay2 = np.zeros((h, w, 3), dtype=np.uint8)
    overlay2[uncovered] = [255, 200, 0]
    overlay2[outlines > 0] = [0, 255, 0]
    axes[0, 2].imshow(blend(img_rgb, overlay2, 0.6))
    axes[0, 2].set_title(f"Uncovered Wall ({int(np.sum(uncovered))} px, yellow)")

    # Row 2: Bridges | Enclosed before | Enclosed after
    bridge_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    bridge_overlay[outlines > 0] = [0, 200, 0]
    for x1, y1, x2, y2 in bridges:
        cv2.line(bridge_overlay, (x1, y1), (x2, y2), (0, 255, 255),
                 thickness, cv2.LINE_AA)
    axes[1, 0].imshow(blend(img_rgb, bridge_overlay, 0.7))
    axes[1, 0].set_title(f"Bridges ({len(bridges)}, cyan lines)")

    regions_before_rgb = colorize_regions(n_before, labels_before)
    axes[1, 1].imshow(blend(img_rgb, regions_before_rgb, 0.55))
    axes[1, 1].set_title(f"Enclosed — Before ({n_before-1})")

    regions_after_rgb = colorize_regions(n_after, labels_after)
    axes[1, 2].imshow(blend(img_rgb, regions_after_rgb, 0.55))
    axes[1, 2].set_title(f"Enclosed — After ({n_after-1})")

    # Row 3: Grey walls on image | Grey wall blobs highlighted | Final composite
    axes[2, 0].imshow(grey_rgb)
    axes[2, 0].set_title(f"Unenclosed Walls → Grey ({len(wall_rects)})")

    # Highlight the wall rects in magenta on original
    highlight = np.zeros((h, w, 3), dtype=np.uint8)
    for wr in wall_rects:
        x1, y1, x2, y2 = wr["bbox"]
        highlight[y1:y2, x1:x2] = [255, 0, 255]
    axes[2, 1].imshow(blend(img_rgb, highlight, 0.7))
    axes[2, 1].set_title(f"Wall Rects (magenta, {len(wall_rects)})")

    # Final composite: enclosed regions + grey walls + bridges
    final = grey_rgb.copy()
    # Draw bridges on the final image
    for x1, y1, x2, y2 in bridges:
        cv2.line(final, (x1, y1), (x2, y2), (0, 0, 0),
                 thickness, cv2.LINE_AA)
    # Overlay enclosed regions
    regions_after_rgb2 = colorize_regions(n_after, labels_after)
    final = blend(final, regions_after_rgb2, 0.35)
    axes[2, 2].imshow(final)
    axes[2, 2].set_title("Final: Grey Walls + Bridges + Regions")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        out = str(Path(input_path).with_stem(
            Path(input_path).stem + "_gap_analysis"))
        out = str(Path(out).with_suffix(".png"))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.show()

    # Print region areas after bridging
    print("\n--- Region areas (after bridging) ---")
    areas = []
    for label_id in range(1, n_after):
        area = int(np.sum(labels_after == label_id))
        areas.append((label_id, area))
    areas.sort(key=lambda x: -x[1])
    for label_id, area in areas:
        pct = area * 100 / (h * w)
        if pct > 0.1:
            print(f"  Region {label_id}: {area:>8} px ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Test U-Net-guided wall gap bridging for enclosed region detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input floorplan image")
    parser.add_argument("--checkpoint", default="weights/epoch_040.pth",
                        help="U-Net checkpoint path")
    parser.add_argument("--black", "-b", type=int, default=80,
                        help="Black pixel threshold (grayscale < this = dark)")
    parser.add_argument("--dilate", type=int, default=10,
                        help="Pixels to dilate U-Net mask before intersecting outlines")
    parser.add_argument("--close-k", type=int, default=5,
                        help="Outline morphological closing kernel size")
    parser.add_argument("--close-i", type=int, default=2,
                        help="Outline morphological closing iterations")
    parser.add_argument("--max-gap", type=float, default=80.0,
                        help="Maximum bridge length in pixels")
    parser.add_argument("--min-blob", type=int, default=100,
                        help="Minimum uncovered wall blob area to consider")
    parser.add_argument("--min-coverage", type=float, default=0.4,
                        help="Minimum outline coverage around blob (0.0-1.0)")
    parser.add_argument("--dilation", type=int, default=20,
                        help="Blob dilation radius for contour search")
    parser.add_argument("--check-radius", type=int, default=10,
                        help="Radius for outline proximity check along contour")
    parser.add_argument("--merge-gap", type=int, default=15,
                        help="Max pixel gap to merge nearby wall rects")
    parser.add_argument("--min-merged", type=int, default=1000,
                        help="Min area for merged wall rects (filters small fragments)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output image path")
    parser.add_argument("--debug", action="store_true",
                        help="Extra debug output")
    args = parser.parse_args()

    process(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        black_threshold=args.black,
        dilate_mask=args.dilate,
        close_outline_kernel=args.close_k,
        close_outline_iter=args.close_i,
        max_gap=args.max_gap,
        min_blob_area=args.min_blob,
        min_coverage=args.min_coverage,
        dilation=args.dilation,
        check_radius=args.check_radius,
        merge_gap=args.merge_gap,
        min_merged_area=args.min_merged,
        output_path=args.output,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
