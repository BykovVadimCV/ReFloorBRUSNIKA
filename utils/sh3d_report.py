"""Print a geometry report for an .sh3d file: openings, walls, connectivity.

Usage:  python utils/sh3d_report.py <file.sh3d> [file2.sh3d ...]
"""

import math
import sys
import zipfile
import xml.etree.ElementTree as ET


def report(path: str) -> None:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("Home.xml").decode("utf-8")
    root = ET.fromstring(xml)

    print(f"\n=== {path} ===")

    openings = root.findall("doorOrWindow")
    print(f"openings: {len(openings)}")
    for o in openings:
        name = o.get("name")
        x, y = float(o.get("x", 0)), float(o.get("y", 0))
        w = float(o.get("width", 0))
        d = float(o.get("depth", 0))
        ang = float(o.get("angle", 0))
        wt = float(o.get("wallThickness", 0))
        elev = float(o.get("elevation", 0) or 0)
        print(f"  {name:<7} pos=({x:7.1f},{y:7.1f})  w={w:6.1f} d={d:5.2f} "
              f"angle={ang:6.3f} wallThk={wt:5.3f} elev={elev:5.1f}")

    walls = root.findall("wall")
    n_conn = sum(1 for w in walls if w.get("wallAtStart") or w.get("wallAtEnd"))
    # Endpoint-proximity connectivity (within 1.0 units)
    eps = 1.0
    pts = []
    for w in walls:
        pts.append((float(w.get("xStart")), float(w.get("yStart")),
                    float(w.get("xEnd")), float(w.get("yEnd")),
                    float(w.get("thickness", 0))))
    isolated = 0
    for i, (x1, y1, x2, y2, t) in enumerate(pts):
        touches = False
        for j, (a1, b1, a2, b2, t2) in enumerate(pts):
            if i == j:
                continue
            for px, py in ((x1, y1), (x2, y2)):
                for qx, qy in ((a1, b1), (a2, b2)):
                    if math.hypot(px - qx, py - qy) <= max(eps, (t + t2) / 2):
                        touches = True
        if not touches:
            isolated += 1
    thks = sorted(t for *_xy, t in pts)
    print(f"walls: {len(walls)}  declared-connected: {n_conn}  "
          f"endpoint-isolated: {isolated}")
    if thks:
        print(f"wall thickness: min={thks[0]:.2f} med={thks[len(thks)//2]:.2f} "
              f"max={thks[-1]:.2f}")
    rooms = root.findall("room")
    print(f"rooms: {len(rooms)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
