"""SH3D export sanity checker.

Parses every ``*.sh3d`` in a results directory and validates the exported
geometry against physical-plausibility bands. Written after the 2026-07-15
post_training audit, where the PNG overlays looked fine while the exports
carried 36 m-wide apartments, 34 cm-tall walls, and 3.8 m-wide doors.

Usage:
    python tools/check_sh3d.py <results_dir> [--wall-height 243.84]

Exit code 0 when every file passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import zipfile
from xml.etree import ElementTree as ET

# Plausibility bands (metres / centimetres)
EXTENT_LONG_SIDE_M = (4.0, 30.0)
DOOR_WIDTH_M = (0.5, 1.3)
WINDOW_WIDTH_M = (0.4, 3.5)
MIN_WALL_THICKNESS_CM = 3.0
MIN_WALL_LENGTH_CM = 20.0
MAX_WALL_THICKNESS_CM = 100.0


def check_file(path: str, wall_height_cm: float) -> list:
    """Return a list of failure strings for one .sh3d file."""
    fails = []
    try:
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("Home.xml").decode("utf-8", "replace"))
    except Exception as exc:
        return [f"cannot parse: {exc}"]

    walls = root.findall(".//wall")
    rooms = root.findall(".//room")
    dws = root.findall(".//doorOrWindow")

    if not walls:
        return ["no walls exported"]

    # 1. Heights: absolute cm, identical everywhere.
    root_h = float(root.get("wallHeight", "0"))
    if abs(root_h - wall_height_cm) > 0.01:
        fails.append(f"root wallHeight {root_h:.2f} != {wall_height_cm}")
    bad_h = sorted({round(float(w.get("height", "0")), 2) for w in walls
                    if abs(float(w.get("height", "0")) - wall_height_cm) > 0.01})
    if bad_h:
        fails.append(f"wall heights scaled/wrong: {bad_h[:4]}")

    # 2. Extent: long side inside the plausible band.
    xs = [float(w.get(k)) for w in walls for k in ("xStart", "xEnd")]
    ys = [float(w.get(k)) for w in walls for k in ("yStart", "yEnd")]
    long_m = max(max(xs) - min(xs), max(ys) - min(ys)) / 100.0
    if not (EXTENT_LONG_SIDE_M[0] <= long_m <= EXTENT_LONG_SIDE_M[1]):
        fails.append(f"plan long side {long_m:.1f} m outside {EXTENT_LONG_SIDE_M}")

    # 3. Wall geometry hygiene.
    thin = sharp = fat = 0
    for w in walls:
        t = float(w.get("thickness", "0"))
        length = math.hypot(
            float(w.get("xEnd")) - float(w.get("xStart")),
            float(w.get("yEnd")) - float(w.get("yStart")),
        )
        thin += t < MIN_WALL_THICKNESS_CM
        sharp += length < MIN_WALL_LENGTH_CM
        fat += t > MAX_WALL_THICKNESS_CM
    if thin:
        fails.append(f"{thin} wall(s) thinner than {MIN_WALL_THICKNESS_CM} cm")
    if sharp:
        fails.append(f"{sharp} wall(s) shorter than {MIN_WALL_LENGTH_CM} cm")
    if fat:
        fails.append(f"{fat} wall(s) thicker than {MAX_WALL_THICKNESS_CM} cm")

    # 4. Opening sizes.
    for d in dws:
        wm = float(d.get("width", "0")) / 100.0
        is_door = "door" in (d.get("catalogId") or "")
        lo, hi = DOOR_WIDTH_M if is_door else WINDOW_WIDTH_M
        if not (lo <= wm <= hi):
            kind = "door" if is_door else "window"
            fails.append(f"{kind} width {wm:.2f} m outside [{lo}, {hi}]")

    # 5. Room names: no OCR shrapnel.
    for r in rooms:
        name = (r.get("name") or "").strip()
        letters = sum(1 for ch in name if ch.isalpha())
        is_area = ("м²" in name or "m2" in name.lower())
        if name and letters <= 2 and not is_area and not name.lower().startswith("room"):
            fails.append(f"garbage room name {name!r}")

    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--wall-height", type=float, default=243.84,
                    help="expected absolute wall height in cm")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "*.sh3d")))
    if not files:
        print(f"no .sh3d files in {args.results_dir}")
        return 1

    total_fails = 0
    for path in files:
        fails = check_file(path, args.wall_height)
        status = "PASS" if not fails else "FAIL"
        print(f"{status}  {os.path.basename(path)}")
        for f in fails:
            print(f"      - {f}")
        total_fails += len(fails)

    print(f"\n{len(files)} file(s), {total_fails} failure(s)")
    return 1 if total_fails else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
