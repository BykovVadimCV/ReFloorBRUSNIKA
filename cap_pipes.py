#!/usr/bin/env python3
"""
cap_pipes.py — Close open-ended pipe structures in binary floor plan images.

Pipes are hollow: two parallel white walls, black interior, open at one/both ends.

Algorithm:
  1. Skeletonize white region; find wall-tip endpoints.
  2. For each endpoint on a long-enough wall, estimate wall direction from
     the LOCAL skeleton path (first few steps), not the full trace.
  3. Shoot a ray perpendicular to the wall, skipping the source wall's own
     white pixels, crossing the black channel, landing on the opposing wall.
  4. If that sequence completes within max_pipe_width px → draw white cap.

Usage:
    python cap_pipes.py input.png [output.png] [options]
"""

import sys, argparse
import cv2
import numpy as np
from skimage.morphology import skeletonize


# ── skeleton helpers ────────────────────────────────────────────────────────────

def _trace_skel(skel, r0, c0, max_steps):
    H, W = skel.shape
    visited = {(r0, c0)}
    path    = [(r0, c0)]
    cur     = (r0, c0)
    for _ in range(max_steps):
        cr, cc = cur
        nbrs = [(cr+dr, cc+dc)
                for dr in (-1,0,1) for dc in (-1,0,1)
                if not (dr==0 and dc==0)
                and 0<=cr+dr<H and 0<=cc+dc<W
                and skel[cr+dr,cc+dc]
                and (cr+dr,cc+dc) not in visited]
        if not nbrs or len(nbrs) > 1:
            break
        visited.add(nbrs[0]); path.append(nbrs[0]); cur = nbrs[0]
    return path


def _local_wall_dir(path, local_steps=15):
    """
    Estimate wall direction using the first `local_steps` steps from the endpoint.
    Returns unit (dr, dc) pointing OUTWARD from the wall (toward the opening).
    """
    n = min(local_steps, len(path) - 1)
    if n < 2:
        return None
    # path[0] = endpoint; path[n] = a point along the wall interior
    d = np.array(path[0], float) - np.array(path[n], float)   # endpoint - interior
    norm = np.linalg.norm(d)
    return d / norm if norm > 1e-6 else None


# ── ray caster ──────────────────────────────────────────────────────────────────

def _ray_hit(binary, dist_black, r0, c0, perp_r, perp_c,
             max_pipe_width, min_pipe_r, max_wall_skip=12):
    """
    Shoot from (r0,c0) in direction (perp_r, perp_c).
    Returns (hit_r, hit_c) or None.
    Phases: skip source-wall white → cross black channel → land on opposing white.
    """
    H, W = binary.shape
    in_source = True
    white_skip = 0
    traversed  = False

    for step in range(1, max_pipe_width + 1):
        nr = int(round(r0 + perp_r * step))
        nc = int(round(c0 + perp_c * step))
        if not (0 <= nr < H and 0 <= nc < W):
            return None

        is_white = binary[nr, nc] > 0

        if in_source:
            if is_white:
                white_skip += 1
                if white_skip > max_wall_skip:
                    return None
                continue
            else:
                in_source = False   # entered the channel

        if is_white:
            return (nr, nc) if traversed else None

        if dist_black[nr, nc] >= min_pipe_r:
            traversed = True

    return None


# ── main detector ───────────────────────────────────────────────────────────────

def find_caps(binary,
              min_wall_len   = 30,
              max_pipe_width = 50,
              min_pipe_r     = 3.0,
              look_ahead     = 80,
              local_steps    = 15,
              cap_thickness  = None,
              merge_radius   = 6):
    """
    Parameters
    ----------
    binary          : uint8 ndarray, 0/255
    min_wall_len    : min skeleton-path length to qualify as a pipe wall
    max_pipe_width  : max gap searched for the opposing wall
    min_pipe_r      : min dist_transform(black) to confirm a real channel
    look_ahead      : how far to trace skeleton (for path-length check)
    local_steps     : how many first steps to use for direction estimation
    cap_thickness   : drawn line thickness (None → auto)
    merge_radius    : endpoints within this many px of an already-capped hit
                      are also skipped (prevents double-capping same opening)

    Returns list of dict with 'p1', 'p2' (col,row) and 'thickness'.
    """
    dist_black = cv2.distanceTransform(
        (binary == 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)

    skel = skeletonize(binary > 0).astype(np.uint8)
    k    = np.ones((3, 3), np.uint8); k[1, 1] = 0
    nc_  = cv2.filter2D(skel, -1, k)
    endpoints = np.argwhere((skel > 0) & (nc_ == 1))

    caps          = []
    capped_exact  = set()          # exact endpoint coords
    capped_hits   = []             # (r, c) of successful hit locations

    def _near_capped_hit(r, c):
        return any(abs(r - hr) + abs(c - hc) <= merge_radius
                   for hr, hc in capped_hits)

    for (r, c) in endpoints:
        if (r, c) in capped_exact or _near_capped_hit(r, c):
            continue

        path = _trace_skel(skel, r, c, look_ahead)
        if len(path) < min_wall_len:
            continue

        d = _local_wall_dir(path, local_steps)
        if d is None:
            continue
        dr, dc = d

        # The perpendicular direction (into the pipe opening)
        # is rotated 90° from the wall direction.
        # We try both signs.
        found = False
        for sign in (1, -1):
            perp_r = sign * (-dc)
            perp_c = sign *   dr

            hit = _ray_hit(binary, dist_black, r, c,
                           perp_r, perp_c,
                           max_pipe_width, min_pipe_r)
            if hit is None:
                continue

            hr, hc = hit
            if _near_capped_hit(hr, hc):
                continue

            seg_len = max(1, int(np.hypot(hr - r, hc - c)))

            # ≥60 % of segment must be black (confirms channel, not wall bulge)
            black_count = sum(
                1 for t in range(1, seg_len)
                if binary[int(round(r  + (hr - r ) * t / seg_len)),
                          int(round(c  + (hc - c ) * t / seg_len))] == 0
            )
            if black_count < seg_len * 0.6:
                continue

            thick = cap_thickness if cap_thickness is not None \
                    else max(2, seg_len // 6)

            caps.append({'p1': (c, r), 'p2': (hc, hr), 'thickness': thick})
            capped_exact.add((r, c))
            capped_hits.append((r, c))
            capped_hits.append((hr, hc))
            found = True
            break

    return caps


def apply_caps(binary, caps):
    result = binary.copy()
    for cap in caps:
        cv2.line(result, cap['p1'], cap['p2'], 255, cap['thickness'])
    return result


def binarize_outlines(img_bgr_or_gray: np.ndarray, threshold: int = 100) -> np.ndarray:
    """
    Convert a floorplan image to a binary mask suitable for find_caps().

    Dark wall outlines (≤ threshold) → 255 (white = pipe walls).
    Light background  (> threshold)  → 0   (black = channel/interior).

    Accepts BGR colour or single-channel grayscale input.
    """
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def cap_pipes(img_path, out_path=None, threshold=100, **kwargs):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    binary = binarize_outlines(img, threshold)
    caps   = find_caps(binary, **kwargs)
    result = apply_caps(binary, caps)
    if out_path:
        cv2.imwrite(out_path, result)
    return result, caps, binary


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--threshold",      type=int,   default=100)
    ap.add_argument("--min-wall",       type=int,   default=30)
    ap.add_argument("--max-pipe-width", type=int,   default=50)
    ap.add_argument("--min-pipe-r",     type=float, default=3.0)
    ap.add_argument("--look-ahead",     type=int,   default=80)
    ap.add_argument("--local-steps",    type=int,   default=15)
    ap.add_argument("--cap-thickness",  type=int,   default=None)
    ap.add_argument("--merge-radius",   type=int,   default=6)
    ap.add_argument("--debug",          action="store_true")
    args = ap.parse_args()

    out = args.output or args.input.replace(".png", "_capped.png")
    result, caps = cap_pipes(
        args.input, out,
        threshold      = args.threshold,
        min_wall_len   = args.min_wall,
        max_pipe_width = args.max_pipe_width,
        min_pipe_r     = args.min_pipe_r,
        look_ahead     = args.look_ahead,
        local_steps    = args.local_steps,
        cap_thickness  = args.cap_thickness,
        merge_radius   = args.merge_radius,
    )

    print(f"Detected {len(caps)} cap(s):")
    for i, cap in enumerate(caps, 1):
        x1,y1 = cap['p1']; x2,y2 = cap['p2']
        orient = "HORIZ" if abs(x2-x1) >= abs(y2-y1) else "VERT"
        print(f"  [{i}] ({x1},{y1})→({x2},{y2})  "
              f"len={int(np.hypot(x2-x1,y2-y1))}px  {orient}  thick={cap['thickness']}px")
    print(f"Saved → {out}")

    if args.debug:
        img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, args.threshold, 255, cv2.THRESH_BINARY)
        dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for cap in caps:
            cv2.line(dbg, cap['p1'], cap['p2'], (0, 255, 0), cap['thickness'] + 4)
        dbg_path = out.replace(".png", "_debug.png")
        cv2.imwrite(dbg_path, dbg)
        print(f"Debug  → {dbg_path}")
