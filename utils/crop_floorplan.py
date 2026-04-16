#!/usr/bin/env python3
"""
extract_floorplan.py — Magic-wand style floor plan extractor.

Finds the largest connected "ink" structure in a document image,
crops a tight bounding box around it, and saves the result.

Usage:
    python extract_floorplan.py input.png output.png [options]

Options:
    --threshold     Binarisation threshold 0-255 (default: auto via Otsu)
    --padding       Extra pixels to add around the bounding box (default: 20)
    --invert        Invert the image before processing (useful for dark backgrounds)
    --min-fill      Minimum fill ratio 0-1 to reject noise regions (default: 0.01)
    --debug         Save a debug image showing all detected components
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    # Convert to BGR if needed (e.g. RGBA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def binarise(gray: np.ndarray, threshold: int | None, invert: bool) -> np.ndarray:
    """Return a binary mask where foreground (ink) = 255."""
    # Mild denoise before thresholding
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if threshold is None:
        # Otsu auto-threshold
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    if invert:
        binary = cv2.bitwise_not(binary)

    # We want the dark structures on light background → ink should be 255
    # If the image background is mostly white (high mean), the ink is dark → invert
    if not invert and np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    return binary


def find_largest_component(binary: np.ndarray, min_fill: float
                           ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Run connected-components analysis and return:
      - mask of the largest component (same size as binary)
      - bounding box (x, y, w, h)
    """
    # Close small gaps so walls connect properly
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    if num_labels <= 1:
        raise RuntimeError("No foreground components found — try adjusting --threshold or --invert.")

    image_area = binary.shape[0] * binary.shape[1]

    # stats columns: CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA
    # Label 0 = background — skip it
    areas = stats[1:, cv2.CC_STAT_AREA]
    ranked = np.argsort(areas)[::-1]  # descending

    chosen_label = None
    chosen_bbox = None
    for rank_idx in ranked:
        label_id = rank_idx + 1  # +1 because we skipped background
        area = areas[rank_idx]
        if area / image_area < min_fill:
            continue  # skip tiny noise
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        chosen_label = label_id
        chosen_bbox = (x, y, w, h)
        break

    if chosen_label is None:
        raise RuntimeError(
            "Largest component is below min-fill ratio. "
            "Try a smaller --min-fill value or check --invert."
        )

    mask = (labels == chosen_label).astype(np.uint8) * 255
    return mask, chosen_bbox


def crop_with_padding(img: np.ndarray, bbox: tuple[int, int, int, int],
                      padding: int) -> np.ndarray:
    x, y, w, h = bbox
    h_img, w_img = img.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    return img[y1:y2, x1:x2]


def save_debug(img: np.ndarray, binary: np.ndarray,
               mask: np.ndarray, bbox: tuple, out_path: str) -> None:
    """Overlay the component mask and bounding box on original for inspection."""
    debug = img.copy()
    # Green tint on the selected component
    green_overlay = np.zeros_like(debug)
    green_overlay[:, :, 1] = mask  # green channel
    debug = cv2.addWeighted(debug, 0.7, green_overlay, 0.3, 0)
    # Draw bounding box
    x, y, w, h = bbox
    cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imwrite(out_path, debug)
    print(f"  Debug image saved → {out_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract the largest structure (floor plan) from a document image."
    )
    p.add_argument("input",  help="Path to input image")
    p.add_argument("output", help="Path for cropped output image")
    p.add_argument("--threshold", type=int, default=None,
                   help="Manual binarisation threshold 0-255 (default: Otsu auto)")
    p.add_argument("--padding", type=int, default=20,
                   help="Padding in pixels around the bounding box (default: 20)")
    p.add_argument("--invert", action="store_true",
                   help="Invert image before processing (for dark/black backgrounds)")
    p.add_argument("--min-fill", type=float, default=0.01,
                   help="Minimum area ratio 0-1 to accept a component (default: 0.01)")
    p.add_argument("--debug", action="store_true",
                   help="Save a debug overlay image alongside the output")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading  : {args.input}")
    img = load_image(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Binarising …")
    binary = binarise(gray, args.threshold, args.invert)

    print("Finding largest connected component …")
    mask, bbox = find_largest_component(binary, args.min_fill)
    x, y, w, h = bbox
    print(f"  Bounding box : x={x}, y={y}, w={w}, h={h}  (area={w*h:,} px²)")

    if args.debug:
        debug_path = str(Path(args.output).with_suffix("")) + "_debug.png"
        save_debug(img, binary, mask, bbox, debug_path)

    print(f"Cropping with {args.padding}px padding …")
    cropped = crop_with_padding(img, bbox, args.padding)

    cv2.imwrite(args.output, cropped)
    print(f"Saved    : {args.output}  ({cropped.shape[1]}×{cropped.shape[0]} px)")


if __name__ == "__main__":
    main()