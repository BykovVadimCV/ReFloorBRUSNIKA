"""
detection/brusnika.py — Brusnika-format detection gate + color-derived wall mask.

Brusnika apartment floorplans are CAD/Illustrator exports with a *fixed* colour
palette (see ``brusnika_examples/BRUSNIKA_FORMAT_REPORT.md``):

    #8F8880  warm gray   — walls (solid fill) AND every thin line (furniture
                           outlines, dimensions, glazing bars, door leaves)
    #E3E0DD  light beige — furniture fills
    #484848  dark gray   — room/area labels
    #AAAAAA  light gray   — balcony railings
    #EE3D24  red-orange   — balcony / loggia floor hatch

This module provides two things, and *only* these two — everything downstream
(rect decomposition, snapping, opening detection, SH3D export) is the existing
pipeline, unchanged:

    is_brusnika_image(img, cfg)       — a strict, verifiable format gate.
    build_brusnika_wall_mask(img, cfg) — a colour + morphology wall mask that
                                         feeds RectWallDetector.detect(wall_mask=…)
                                         in place of the binary U-Net.

Design rationale
----------------
Walls are the only *thick solid* #8F8880 regions, but that exact gray is also
the colour of all thin line-art.  Colour thresholding therefore selects walls
*plus* clutter; the separation is geometric (a morphological opening removes
strokes thinner than a wall while leaving wall bodies intact).  Furniture beige,
balcony red and railing gray are different colours and are excluded implicitly
by the colour match.

The gate is intentionally strict: a false positive (a non-Brusnika plan routed
through this branch) is the dangerous direction, so both a colour-area test and
a dominant-colour test must agree before the branch activates.  When the gate
returns False the caller passes ``wall_mask=None`` and the original U-Net path
runs byte-for-byte unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Parse a 'RRGGBB' hex string (no '#') to an (R, G, B) tuple."""
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _wall_color_frac(img_bgr: np.ndarray, rgb: Tuple[int, int, int],
                     tol: int) -> float:
    """Fraction of pixels within ±tol (per channel, L-inf) of the wall colour.

    Cheap, fully vectorised — this is the first gate stage, so non-Brusnika
    images bail out here before the more expensive dominant-colour pass runs.
    """
    b, g, r = int(rgb[2]), int(rgb[1]), int(rgb[0])
    diff = np.abs(img_bgr.astype(np.int16) - np.array([b, g, r], np.int16))
    hit = (diff.max(axis=2) <= tol)
    return float(hit.mean())


def is_brusnika_image(img_bgr: np.ndarray, config) -> bool:
    """Return True only when the image is verifiably a Brusnika-format plan.

    Two independent criteria must BOTH hold (logical AND):

      1. Colour-area: a meaningful fraction of pixels match the Brusnika wall
         gray #8F8880.  Empirically ≥10 % on Brusnika plans vs ≤1.4 % on every
         non-Brusnika sample tested — checked first because it is cheap.
      2. Dominant-colour: the auto-detected dominant wall colour is within a
         tight distance of #8F8880.  Empirically 0–1 on Brusnika vs ≥46 on
         every non-Brusnika sample (the connected gray skeleton wins K-means).

    The conjunction makes a false positive extremely unlikely, which is the
    behaviour the pipeline needs — the branch must never hijack a generic plan.
    """
    if img_bgr is None or img_bgr.size == 0:
        return False
    if not getattr(config, "enable_brusnika_branch", True):
        return False

    wall_rgb = _hex_to_rgb(getattr(config, "brusnika_wall_color_hex", "8F8880"))
    min_frac = float(getattr(config, "brusnika_gate_min_wallgray_frac", 0.03))
    max_dist = int(getattr(config, "brusnika_gate_max_color_dist", 12))
    area_tol = int(getattr(config, "brusnika_gate_color_tol", 18))

    # ── Stage 1 (cheap): colour-area test ──────────────────────────────
    frac = _wall_color_frac(img_bgr, wall_rgb, area_tol)
    if frac < min_frac:
        logger.debug(
            "Brusnika gate: wall-gray fraction %.3f < %.3f → not Brusnika",
            frac, min_frac,
        )
        return False

    # ── Stage 2 (only on survivors): dominant-colour confirmation ──────
    try:
        from detection.walls import detect_dominant_wall_color
        r, g, b = detect_dominant_wall_color(img_bgr)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Brusnika gate: dominant-colour detection failed (%s) "
                       "— treating as not Brusnika", exc)
        return False

    dist = max(abs(r - wall_rgb[0]), abs(g - wall_rgb[1]), abs(b - wall_rgb[2]))
    is_brusnika = dist <= max_dist
    logger.info(
        "Brusnika gate: wall-gray frac=%.3f (≥%.3f ✓), dominant RGB=(%d,%d,%d) "
        "dist=%d (≤%d %s) → %s",
        frac, min_frac, r, g, b, dist, max_dist,
        "✓" if is_brusnika else "✗",
        "BRUSNIKA" if is_brusnika else "not Brusnika",
    )
    return is_brusnika


def _remove_circular_badges(mask: np.ndarray, config,
                            min_area: Optional[int] = None) -> np.ndarray:
    """Erase the solid apartment-count badge disk(s) from the wall mask.

    Brusnika plans stamp a filled #8F8880 circle (with the apartment's room
    count inside) in open space.  It is the same gray as walls, survives the
    opening, and would otherwise become a spurious wall blob — the rect_walls
    path has no circle removal of its own.

    A filled disk has a unique, empirically-stable connected-component
    signature that distinguishes it from every real wall feature:

        aspect  ≈ 1.0   (square bounding box)   — walls/stubs are elongated
        circ    ≥ 0.80  (4πA/P²)                — walls are < 0.35
        extent  ≈ π/4 ≈ 0.78  (A / bbox-area)   — solid square columns are ≈1.0

    Only components matching ALL three (within tolerance) and above a minimum
    area are removed, so walls, door stubs and square structural columns are
    never touched.
    """
    if not getattr(config, "brusnika_remove_badge", True):
        return mask
    min_circ = float(getattr(config, "brusnika_badge_min_circularity", 0.80))
    min_aspect = float(getattr(config, "brusnika_badge_min_aspect", 0.80))
    ext_lo, ext_hi = getattr(config, "brusnika_badge_extent_range", (0.60, 0.85))
    if min_area is None:
        min_area = int(getattr(config, "brusnika_badge_min_area_px", 800))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    removed = 0
    out = mask.copy()
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        aspect = min(bw, bh) / max(1, max(bw, bh))
        if aspect < min_aspect:
            continue
        extent = area / float(max(1, bw * bh))
        if not (ext_lo <= extent <= ext_hi):
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        per = cv2.arcLength(cnts[0], True)
        circ = 4.0 * np.pi * area / (per * per) if per > 0 else 0.0
        if circ < min_circ:
            continue
        out[labels == i] = 0
        removed += 1
        logger.info(
            "Brusnika: removed apartment-badge disk at (%d,%d) "
            "area=%d circ=%.2f extent=%.2f aspect=%.2f",
            x, y, area, circ, extent, aspect,
        )
    return out


def build_brusnika_wall_mask(img_bgr: np.ndarray, config) -> np.ndarray:
    """Build a binary wall mask (0/255) from the Brusnika colour palette.

    Steps (report §4):
      1. Colour-match the wall gray #8F8880 → selects walls + all same-gray
         line-art.  Furniture beige, balcony red, railing gray, dark text are
         different colours and are excluded implicitly.
      2. Morphological OPEN → removes strokes thinner than a wall (furniture
         outlines, glazing bars, dimension lines, door leaves) while leaving
         solid wall bodies intact.
      3. Drop tiny connected components (specks, isolated symbol fragments).

    The result is handed to ``RectWallDetector.detect(wall_mask=…)``, which then
    runs the identical morph-close / door-text filter / cap / rect_decompose
    chain the U-Net mask would have gone through.
    """
    wall_rgb = _hex_to_rgb(getattr(config, "brusnika_wall_color_hex", "8F8880"))
    tol = int(getattr(config, "brusnika_color_tol", 22))
    open_it = int(getattr(config, "brusnika_open_iterations", 1))

    h, w = img_bgr.shape[:2]

    # All kernel/area defaults were tuned at ~929 px width; scale them to the
    # actual image so the opening removes line-art (not thin walls) and the
    # area floors stay consistent across input resolutions.  Length-like
    # quantities scale with width, area-like quantities with width².
    ref_w = float(getattr(config, "brusnika_ref_width_px", 929))
    s = max(0.3, w / ref_w) if ref_w > 0 else 1.0
    open_k = max(3, int(round(int(getattr(config, "brusnika_open_kernel", 5)) * s)))
    if open_k % 2 == 0:
        open_k += 1
    min_area = int(int(getattr(config, "brusnika_min_component_area_px", 80)) * s * s)
    badge_min_area = int(int(getattr(config, "brusnika_badge_min_area_px", 800)) * s * s)

    # ── 1. Colour match (L-inf ball around the wall gray) ──────────────
    b, g, r = int(wall_rgb[2]), int(wall_rgb[1]), int(wall_rgb[0])
    diff = np.abs(img_bgr.astype(np.int16) - np.array([b, g, r], np.int16))
    mask = (diff.max(axis=2) <= tol).astype(np.uint8) * 255
    px_color = int(np.count_nonzero(mask))

    # ── 2. Morphological opening — strip thin line-art, keep wall bodies ─
    if open_k >= 2 and open_it > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_it)
    px_opened = int(np.count_nonzero(mask))

    # ── 3. Remove tiny components (specks / symbol fragments) ──────────
    if min_area > 0 and px_opened > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[i] = True
        mask = np.where(keep[labels], 255, 0).astype(np.uint8)

    # ── 4. Remove the apartment-count badge disk (same gray as walls) ──
    mask = _remove_circular_badges(mask, config, min_area=badge_min_area)
    px_final = int(np.count_nonzero(mask))

    logger.info(
        "Brusnika wall mask: colour-match %d px (%.1f%%) → open(k=%d,i=%d) "
        "%d px → drop<%dpx %d px (%.1f%% of image)",
        px_color, 100.0 * px_color / (h * w), open_k, open_it,
        px_opened, min_area, px_final, 100.0 * px_final / (h * w),
    )
    return mask
