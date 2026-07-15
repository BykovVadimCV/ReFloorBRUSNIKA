"""A/B harness for the binary wall U-Net input-size experiment.

Runs the full pipeline once per arm (baseline = dynamic (H+W)//2 input, then one
arm per --size N using main.py's --unet-fixed-size flag), into a separate output
directory per arm, and diffs the per-plan results.

Must be run on a machine where the full pipeline works: E2E dies with a native
SIGILL inside core/rect_decompose.py on the CPU dev box.  Use --device cuda.

    python tools/ab_unet_size.py --input input --size 563 --device cuda

Add --skip-run to re-diff already-generated arm directories without re-running
the pipeline.  Stdlib only, so it needs nothing beyond the pipeline's own deps.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Room names the OCR stage is known to emit that are areas or junk rather than
# real names ("11.8 m2", "h=3.30", "-", "N").  Used only to count *usable* names.
_JUNK_NAME = re.compile(r"^\s*(-+|[\d.,]+\s*(m2|м2)?|h\s*=.*|[a-zA-Zа-яА-Я]{1,2})\s*$", re.I)

_RE_RECTS = re.compile(r"Rects from decompose\s*:\s*(\d+)\s*\(coverage\s*([\d.]+)%")
_RE_MASK = re.compile(r"Final filtered mask px\s*:\s*(\d+)")
_RE_STRAIGHT = re.compile(r"Near-axis straightening.*?:\s*(\d+)\s*wall")
_RE_STOP = re.compile(r"STOP REASON:\s*(.+)")


def _poly_area(points: list[tuple[float, float]]) -> float:
    """Shoelace area of a room polygon, in the exported coordinate units."""
    a = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def read_sh3d(path: Path) -> dict:
    """Pull per-plan counts out of a .sh3d's Home.xml."""
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("Home.xml"))

    rooms = root.findall("room")
    named = 0
    area = 0.0
    for r in rooms:
        name = (r.get("name") or "").strip()
        if name and not _JUNK_NAME.match(name):
            named += 1
        pts = [(float(p.get("x")), float(p.get("y"))) for p in r.findall("point")]
        if len(pts) >= 3:
            area += _poly_area(pts)

    doors = windows = 0
    for o in root.findall("doorOrWindow"):
        tag = f"{o.get('catalogId', '')} {o.get('name', '')}".lower()
        if "window" in tag:
            windows += 1
        else:
            doors += 1

    return {
        "walls": len(root.findall("wall")),
        "rooms": len(rooms),
        "named_rooms": named,
        "doors": doors,
        "windows": windows,
        "room_area": round(area, 1),
    }


def read_rectlog(path: Path) -> dict:
    """Pull decomposition stats out of a <plan>_rectdecompose.log."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    if m := _RE_RECTS.search(text):
        out["rects"] = int(m.group(1))
        out["coverage"] = float(m.group(2))
    if m := _RE_MASK.search(text):
        out["mask_px"] = int(m.group(1))
    if m := _RE_STRAIGHT.search(text):
        out["straightened"] = int(m.group(1))
    if m := _RE_STOP.search(text):
        out["stop_reason"] = m.group(1).strip()
    return out


def collect(out_dir: Path) -> dict[str, dict]:
    """Metrics for every plan the pipeline exported into out_dir."""
    plans: dict[str, dict] = {}
    for sh3d in sorted(out_dir.glob("*.sh3d")):
        plan = sh3d.stem
        try:
            row = read_sh3d(sh3d)
        except Exception as exc:  # a corrupt/partial export shouldn't kill the diff
            plans[plan] = {"error": str(exc)}
            continue
        row.update(read_rectlog(out_dir / f"{plan}_rectdecompose.log"))
        plans[plan] = row
    return plans


def run_arm(input_dir: str, out_dir: Path, device: str, size: int | None) -> bool:
    """Invoke main.py for one arm.  size=None is the dynamic-input baseline."""
    cmd = [
        sys.executable, "main.py",
        "--input", input_dir,
        "--output", str(out_dir),
        "--device", device,
        "--debug",  # the rectdecompose logs we diff are debug artifacts
    ]
    if size is not None:
        cmd += ["--unet-fixed-size", str(size)]

    label = "baseline (dynamic)" if size is None else f"fixed {size}"
    print(f"\n=== arm: {label} -> {out_dir} ===", flush=True)
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        print(f"!! arm '{label}' exited {proc.returncode} "
              f"(132/-4 = the known rect_decompose SIGILL; are you on the GPU box?)")
        return False
    return True


# Columns we diff, and whether "up" is the direction we hope to see.
_METRICS = ["walls", "rects", "coverage", "mask_px", "rooms", "named_rooms",
            "doors", "windows", "straightened", "room_area"]


def diff_table(base: dict[str, dict], arms: dict[str, dict[str, dict]]) -> None:
    plans = sorted(set(base) | {p for a in arms.values() for p in a},
                   key=lambda s: (len(s), s))

    for arm_name, arm in arms.items():
        print(f"\n{'=' * 78}\n  {arm_name}  vs  baseline\n{'=' * 78}")
        header = f"{'plan':>6}  " + "  ".join(f"{m:>12}" for m in _METRICS)
        print(header)
        print("-" * len(header))
        for plan in plans:
            b, a = base.get(plan, {}), arm.get(plan, {})
            cells = []
            for m in _METRICS:
                bv, av = b.get(m), a.get(m)
                if bv is None and av is None:
                    cells.append(f"{'-':>12}")
                elif bv is None or av is None:
                    cells.append(f"{str(av if bv is None else bv):>12}")
                else:
                    d = av - bv
                    sign = "+" if d > 0 else ""
                    cells.append(f"{av:>7}{sign}{d:>4}" if d else f"{av:>7}{'':>5}")
            print(f"{plan:>6}  " + "  ".join(cells))

        # Totals give the one-line verdict for the arm.
        print("-" * len(header))
        tot = []
        for m in _METRICS:
            bs = sum(base.get(p, {}).get(m, 0) or 0 for p in plans)
            as_ = sum(arm.get(p, {}).get(m, 0) or 0 for p in plans)
            d = as_ - bs
            sign = "+" if d > 0 else ""
            tot.append(f"{as_:>7.0f}{sign}{d:>4.0f}" if d else f"{as_:>7.0f}{'':>5}")
        print(f"{'TOTAL':>6}  " + "  ".join(tot))


def write_csv(path: Path, base: dict, arms: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["plan", "arm"] + _METRICS + ["stop_reason"])
        for arm_name, arm in [("baseline", base)] + list(arms.items()):
            for plan in sorted(arm, key=lambda s: (len(s), s)):
                row = arm[plan]
                w.writerow([plan, arm_name]
                           + [row.get(m, "") for m in _METRICS]
                           + [row.get("stop_reason", "")])
    print(f"\nper-plan CSV: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="input", help="image dir (default: input)")
    ap.add_argument("--size", type=int, nargs="+", default=[563],
                    metavar="N",
                    help="fixed U-Net input size(s) to test (default: 563, the "
                         "pooled optimum from analysis_out/unet_fixed_size_best.json)")
    ap.add_argument("--device", default="cuda", help="cpu | cuda | 0 (default: cuda)")
    ap.add_argument("--work", default="ab_unet", metavar="DIR",
                    help="parent dir for the arm outputs (default: ab_unet)")
    ap.add_argument("--skip-run", action="store_true",
                    help="don't re-run the pipeline; just re-diff existing arm dirs")
    args = ap.parse_args()

    work = REPO / args.work
    base_dir = work / "baseline"
    arm_dirs = {f"fixed_{n}": (work / f"fixed_{n}", n) for n in args.size}

    if not args.skip_run:
        work.mkdir(parents=True, exist_ok=True)
        if not run_arm(args.input, base_dir, args.device, None):
            sys.exit(1)
        for name, (d, n) in arm_dirs.items():
            if not run_arm(args.input, d, args.device, n):
                sys.exit(1)

    base = collect(base_dir)
    if not base:
        sys.exit(f"no .sh3d found in {base_dir} — run without --skip-run first")
    arms = {name: collect(d) for name, (d, _) in arm_dirs.items()}

    diff_table(base, arms)
    write_csv(work / "ab_unet_size.csv", base, arms)


if __name__ == "__main__":
    main()
