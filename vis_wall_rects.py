#!/usr/bin/env python3
"""
vis_wall_rects.py — diagnostic overlay: wall mask vs rect-decompose output.

Colour coding
─────────────
  WHITE  : pixel is wall mask  AND covered by a rect
  GREEN  : pixel is wall mask  but NOT covered by any rect
  RED    : pixel is covered by a rect  but NOT in wall mask
  BLACK  : everything else

rect_max_overlap is forced to 1.0 so rectangles may overlap freely
(no overlap penalty is applied).

Usage
─────
  python vis_wall_rects.py <image_path> [output_path]

  <image_path>   path to the floorplan image (BGR, any OpenCV-readable format)
  [output_path]  where to save the result
                 (default: <stem>_wall_rect_diag.png, same directory as input)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Make sure repo root is on the path when called from a different CWD
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rasterise_polygons(polygons: list, H: int, W: int) -> np.ndarray:
    """Fill all polygon interiors into a uint8 mask (255 = inside)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polygons:
        corners = np.asarray(poly, dtype=np.int32)
        cv2.fillPoly(mask, [corners], 255)
    return mask


def _build_diag_image(wall_mask: np.ndarray, rect_mask: np.ndarray) -> np.ndarray:
    """
    Build the 3-channel diagnostic image.

    Parameters
    ----------
    wall_mask : H×W uint8, non-zero = wall pixel
    rect_mask : H×W uint8, non-zero = covered by at least one rect

    Returns
    -------
    BGR image (H×W×3 uint8):
        (255,255,255) white  — wall ∩ rect
        (  0,255,  0) green  — wall only
        (  0,  0,255) red    — rect only
        (  0,  0,  0) black  — neither
    """
    wall_bin = wall_mask > 0
    rect_bin = rect_mask > 0

    diag = np.zeros((*wall_mask.shape, 3), dtype=np.uint8)

    # White: wall AND rect
    both = wall_bin & rect_bin
    diag[both] = (255, 255, 255)

    # Green: wall only  (BGR order)
    wall_only = wall_bin & ~rect_bin
    diag[wall_only] = (0, 255, 0)

    # Red: rect only
    rect_only = rect_bin & ~wall_bin
    diag[rect_only] = (0, 0, 255)

    return diag


def _print_stats(wall_mask: np.ndarray, rect_mask: np.ndarray) -> None:
    wall_bin = wall_mask > 0
    rect_bin = rect_mask > 0

    n_both      = int(np.count_nonzero(wall_bin & rect_bin))
    n_wall_only = int(np.count_nonzero(wall_bin & ~rect_bin))
    n_rect_only = int(np.count_nonzero(rect_bin & ~wall_bin))
    total_wall  = n_both + n_wall_only
    total_rect  = n_both + n_rect_only

    print()
    print("Pixel breakdown:")
    print(f"  White  (wall ∩ rect)   : {n_both:>9,}  "
          f"({100 * n_both      / max(1, total_wall):.1f}% of all wall px)")
    print(f"  Green  (wall only)     : {n_wall_only:>9,}  "
          f"({100 * n_wall_only / max(1, total_wall):.1f}% of all wall px)")
    print(f"  Red    (rect only)     : {n_rect_only:>9,}  "
          f"({100 * n_rect_only / max(1, total_rect):.1f}% of all rect px)")
    print()
    print(f"  Wall coverage by rects : {100 * n_both / max(1, total_wall):.2f}%")
    print(f"  Rect pixels outside wall: {100 * n_rect_only / max(1, total_rect):.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: image not found: {img_path}")
        sys.exit(1)

    out_path = (
        Path(sys.argv[2])
        if len(sys.argv) >= 3
        else img_path.parent / (img_path.stem + "_wall_rect_diag.png")
    )

    # ── Load image ───────────────────────────────────────────────────────
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Error: could not read image: {img_path}")
        sys.exit(1)

    H, W = img_bgr.shape[:2]
    print(f"Image  : {img_path}  ({W} × {H} px)")

    # ── Config — force rect_max_overlap=1.0 (free overlap) ──────────────
    from core.config import PipelineConfig
    cfg = PipelineConfig(rect_max_overlap=1.0)

    # ── Run detector ─────────────────────────────────────────────────────
    from detection.rect_walls import RectWallDetector
    det = RectWallDetector(cfg)
    det.initialize()

    print("Running RectWallDetector …")
    result = det.detect(img_bgr)

    wall_mask = result.wall_mask          # H×W uint8
    polygons  = result.wall_rect_polygons # list[ndarray]  Nx2 float32

    if wall_mask is None:
        print("Error: detector returned no wall mask — cannot build diagnostic image.")
        sys.exit(1)

    wall_px = int(np.count_nonzero(wall_mask))
    print(f"Wall mask pixels : {wall_px:,}")
    print(f"Rectangles found : {len(polygons)}")

    # ── Rasterise rectangles ─────────────────────────────────────────────
    rect_mask = _rasterise_polygons(polygons, H, W)

    # ── Build & save diagnostic image ────────────────────────────────────
    diag = _build_diag_image(wall_mask, rect_mask)
    _print_stats(wall_mask, rect_mask)

    cv2.imwrite(str(out_path), diag)
    print(f"\nSaved : {out_path}")


if __name__ == "__main__":
    main()
