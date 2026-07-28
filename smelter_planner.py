#!/usr/bin/env python3
"""
Factorio smelter layout planner.

Input : a text map of the buildable area
          .   free tile
          #   blocked tile (ore patch, water, existing structures)
          I   iron-ore belt entry point   (on map edge, feeds inward)
          C   copper-ore belt entry point
          L   coal belt entry point (optional; if absent, coal is assumed
              pre-mixed on the ore belts' second lane)
Output: ASCII layout + placement stats + importable Factorio blueprint string.

Approach (heuristic, not exhaustive -- true optimality is NP-hard):
  1. Pack standard 11-wide double-sided smelting cells into maximal free
     rectangles (histogram-based largest-rectangle search, greedy stamping).
  2. Route each ore entry to its cells with A* over (x, y, direction) states.
     Underground belts are modelled as jump moves (span 2..5) that may cross
     occupied tiles, with a small cost penalty so surface belts are preferred.
  3. Chain multiple cells per resource with splitters at each cell's intake.

Cell anatomy (width 11, ore flows top -> bottom):
  col 0  output belt (plates, side A)
  col 1  inserters furnace -> belt
  col 2-3 stone furnaces (2x2), side A
  col 4  inserters belt -> furnace
  col 5  input belt (ore on one lane, coal on the other)
  col 6  inserters belt -> furnace
  col 7-8 stone furnaces, side B
  col 9  inserters furnace -> belt
  col 10 output belt (plates, side B)

A cell of height H holds 2 * (H // 2) furnaces. 48 stone furnaces
(24 per side, H = 48) saturate one yellow belt of ore -> one belt of plates.
"""

from __future__ import annotations
import argparse
import base64
import heapq
import json
import sys
import zlib
from dataclasses import dataclass, field

# Directions: Factorio blueprint convention (0=N, 2=E, 4=S, 6=W)
N, E, S, W = 0, 2, 4, 6
DXY = {N: (0, -1), E: (1, 0), S: (0, 1), W: (-1, 0)}
DIR_CHAR = {N: "^", E: ">", S: "v", W: "<"}
UNDER_MAX_SPAN = 5   # yellow underground: exit up to 5 tiles from entrance
UNDER_MIN_SPAN = 2

CELL_W = 11          # standard double-sided smelting cell width
CELL_GAP = 1         # spacing between stamped cells
CELL_MIN_H = 8       # below this a cell isn't worth the routing overhead
CELL_HEADER = 2      # rows above furnaces reserved for intake splitter/turn

RESOURCE_NAMES = {"I": "iron ore", "C": "copper ore", "L": "coal"}


@dataclass
class Entity:
    name: str
    x: float            # blueprint position (entity centre)
    y: float
    direction: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class Cell:
    x: int              # top-left tile of the cell footprint
    y: int
    h: int              # total footprint height (header + furnace rows)
    resource: str

    @property
    def furnace_rows(self) -> int:
        return (self.h - CELL_HEADER) // 2

    @property
    def furnaces(self) -> int:
        return self.furnace_rows * 2

    @property
    def intake(self) -> tuple[int, int]:
        """Tile where the input belt must be delivered (top of col 5)."""
        return (self.x + 5, self.y)


class Planner:
    def __init__(self, grid: list[list[str]]):
        self.h = len(grid)
        self.w = len(grid[0])
        self.blocked = [[c == "#" for c in row] for row in grid]
        self.entries: list[tuple[int, int, str]] = []
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c in RESOURCE_NAMES:
                    self.entries.append((x, y, c))
        # canvas: what occupies each tile after planning ('' = empty)
        self.canvas: list[list[str]] = [["" for _ in range(self.w)]
                                        for _ in range(self.h)]
        self.entities: list[Entity] = []
        self.cells: list[Cell] = []
        self.reserved: set[tuple[int, int]] = set()
        for y in range(min(2, self.h)):
            for x in range(self.w):
                self.reserved.add((x, y))
        for ex, ey, _res in self.entries:
            for yy in range(max(0, ey - 1), min(self.h, ey + 2)):
                for x in range(self.w):
                    self.reserved.add((x, yy))

    # ------------------------------------------------------------------ util
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def free(self, x: int, y: int) -> bool:
        return (self.in_bounds(x, y) and not self.blocked[y][x]
                and self.canvas[y][x] == "")

    def free_build(self, x: int, y: int) -> bool:
        return self.free(x, y) and (x, y) not in self.reserved

    def occupy(self, x: int, y: int, tag: str) -> None:
        self.canvas[y][x] = tag

    # ---------------------------------------------------- rectangle packing
    def _maximal_rectangles(self) -> list[tuple[int, int, int, int]]:
        """All maximal free rectangles via the histogram method.
        Returns (x, y, w, h), largest area first."""
        heights = [0] * self.w
        rects = []
        for y in range(self.h):
            for x in range(self.w):
                heights[x] = heights[x] + 1 if self.free_build(x, y) else 0
            stack: list[tuple[int, int]] = []   # (start_x, height)
            for x in range(self.w + 1):
                cur = heights[x] if x < self.w else 0
                start = x
                while stack and stack[-1][1] >= cur:
                    sx, sh = stack.pop()
                    rects.append((sx, y - sh + 1, x - sx, sh))
                    start = sx
                if not stack or cur > stack[-1][1]:
                    stack.append((start, cur))
        rects = [r for r in rects if r[2] >= CELL_W and r[3] >= CELL_MIN_H]
        rects.sort(key=lambda r: r[2] * r[3], reverse=True)
        return rects

    def pack_cells(self, max_cells: int | None = None) -> None:
        """Greedily stamp smelting cells into the largest free rectangles."""
        placed = 0
        while max_cells is None or placed < max_cells:
            rects = self._maximal_rectangles()
            if not rects:
                break
            rx, ry, rw, rh = rects[0]
            h = min(rh, CELL_HEADER + 48)      # cap at belt-saturating size
            h = CELL_HEADER + ((h - CELL_HEADER) // 2) * 2
            if h < CELL_MIN_H:
                break
            n_fit = (rw + CELL_GAP) // (CELL_W + CELL_GAP)
            if n_fit == 0:
                break
            for i in range(n_fit):
                if max_cells is not None and placed >= max_cells:
                    break
                cx = rx + i * (CELL_W + CELL_GAP)
                cell = Cell(cx, ry, h, resource="")
                self._stamp_cell(cell)
                self.cells.append(cell)
                placed += 1
        # assign each cell to the nearest ore entry's resource, so each
        # resource's cells form a contiguous district near its belt
        ore_entries = [e for e in self.entries if e[2] in ("I", "C")]
        if not ore_entries:
            ore_entries = [(0, 0, "I")]
        for cell in self.cells:
            ix, iy = cell.intake
            nearest = min(ore_entries,
                          key=lambda e: abs(e[0] - ix) + abs(e[1] - iy))
            cell.resource = nearest[2]

    def _stamp_cell(self, c: Cell) -> None:
        top = c.y + CELL_HEADER
        for row in range(c.furnace_rows):
            fy = top + row * 2
            for fx0 in (c.x + 2, c.x + 7):
                for dy in range(2):
                    for dx in range(2):
                        self.occupy(fx0 + dx, fy + dy, "F")
                self.entities.append(
                    Entity("stone-furnace", fx0 + 1.0, fy + 1.0))
            # inserters (blueprint direction points TOWARD pickup)
            self._place_inserter(c.x + 1, fy, pickup=E)   # furnace -> belt A
            self._place_inserter(c.x + 4, fy, pickup=E)   # input belt -> furnace A
            self._place_inserter(c.x + 6, fy, pickup=W)   # input belt -> furnace B
            self._place_inserter(c.x + 9, fy, pickup=W)   # furnace -> belt B
        # belts: input col 5 runs S the full height; outputs cols 0,10 run S
        for yy in range(c.y, c.y + c.h):
            self._place_belt(c.x + 5, yy, S)
        for col in (c.x + 0, c.x + 10):
            for yy in range(c.y + CELL_HEADER - 1, c.y + c.h):
                self._place_belt(col, yy, S)
        # header: intake splitter feeding col 5, pass-through continuing E
        # (splitter itself is placed during routing so the chain lines up)

    def _place_belt(self, x: int, y: int, d: int) -> None:
        self.occupy(x, y, DIR_CHAR[d])
        self.entities.append(Entity("transport-belt", x + 0.5, y + 0.5, d))

    def _place_underground(self, x: int, y: int, d: int, io: str) -> None:
        self.occupy(x, y, "U" if io == "input" else "D")
        self.entities.append(Entity("underground-belt", x + 0.5, y + 0.5, d,
                                    {"type": io}))

    def _place_inserter(self, x: int, y: int, pickup: int) -> None:
        self.occupy(x, y, "i")
        self.entities.append(Entity("inserter", x + 0.5, y + 0.5, pickup))

    def _place_splitter(self, x: int, y: int, d: int) -> None:
        # 2 tiles wide perpendicular to facing; anchor at (x, y) and (x+1, y)
        self.occupy(x, y, "S")
        self.occupy(x + 1, y, "S")
        self.entities.append(Entity("splitter", x + 1.0, y + 0.5, d))

    # ------------------------------------------------------------- routing
    def route(self, sx: int, sy: int, gx: int, gy: int) -> bool:
        """A* from (sx, sy) to (gx, gy); writes belts/undergrounds. The goal
        tile is expected to already contain the cell's input belt."""
        start_dirs = [d for d in (N, E, S, W)]
        openq: list[tuple[int, int, int, int]] = []
        best: dict[tuple[int, int, int], int] = {}
        came: dict[tuple[int, int, int],
                   tuple[tuple[int, int, int], str]] = {}

        def hcost(x: int, y: int) -> int:
            return abs(x - gx) + abs(y - gy)

        for d in start_dirs:
            st = (sx, sy, d)
            best[st] = 0
            heapq.heappush(openq, (hcost(sx, sy), 0, sx, sy, d))

        goal_state = None
        while openq:
            f, g, x, y, d = heapq.heappop(openq)
            if best.get((x, y, d), 1 << 30) < g:
                continue
            if (x, y) == (gx, gy):
                goal_state = (x, y, d)
                break
            for nd in (N, E, S, W):
                if (nd + 4) % 8 == d:          # no immediate reversal
                    continue
                dx, dy = DXY[nd]
                nx, ny = x + dx, y + dy
                # surface belt step
                if (self.free(nx, ny) or (nx, ny) == (gx, gy)):
                    ng = g + 1
                    st = (nx, ny, nd)
                    if ng < best.get(st, 1 << 30):
                        best[st] = ng
                        came[st] = ((x, y, d), "belt")
                        heapq.heappush(
                            openq, (ng + hcost(nx, ny), ng, nx, ny, nd))
                # underground jump (straight, same new direction)
                for span in range(UNDER_MIN_SPAN, UNDER_MAX_SPAN + 1):
                    ex, ey = x + dx * (span + 1), y + dy * (span + 1)
                    if not self.in_bounds(ex, ey):
                        break
                    if not (self.free(nx, ny) and
                            (self.free(ex, ey) or (ex, ey) == (gx, gy))):
                        continue
                    ng = g + span + 3          # penalty: prefer surface
                    st = (ex, ey, nd)
                    if ng < best.get(st, 1 << 30):
                        best[st] = ng
                        came[st] = ((x, y, d), f"under{span}")
                        heapq.heappush(
                            openq, (ng + hcost(ex, ey), ng, ex, ey, nd))
        if goal_state is None:
            return False
        # walk back and place
        steps = []
        st = goal_state
        while st in came:
            prev, kind = came[st]
            steps.append((prev, st, kind))
            st = prev
        for (px, py, pd), (cx_, cy_, cd), kind in reversed(steps):
            if kind == "belt":
                if (cx_, cy_) != (gx, gy):
                    self._place_belt(cx_, cy_, cd)
            else:
                span = int(kind[5:])
                dx, dy = DXY[cd]
                self._place_underground(px + dx, py + dy, cd, "input")
                if (cx_, cy_) != (gx, gy):
                    self._place_underground(cx_, cy_, cd, "output")
        return True

    def route_all(self) -> list[str]:
        """Route every ore entry to its assigned cells; chain with splitters."""
        log = []
        by_res: dict[str, list[Cell]] = {}
        for c in self.cells:
            by_res.setdefault(c.resource, []).append(c)
        for ex, ey, res in self.entries:
            targets = by_res.get(res, [])
            if not targets:
                log.append(f"{RESOURCE_NAMES[res]} entry at ({ex},{ey}): "
                           f"no cells assigned")
                continue
            remaining = list(targets)
            src = (ex, ey)
            while remaining:
                remaining.sort(key=lambda c: abs(c.intake[0] - src[0])
                               + abs(c.intake[1] - src[1]))
                cell = remaining.pop(0)
                gx, gy = cell.intake
                ok = self.route(src[0], src[1], gx, gy)
                if not ok and src != (ex, ey):
                    ok = self.route(ex, ey, gx, gy)   # retry from the entry
                tag = "ok" if ok else "FAILED (no path)"
                log.append(
                    f"{RESOURCE_NAMES[res]}: ({src[0]},{src[1]}) -> "
                    f"cell@({cell.x},{cell.y}) [{cell.furnaces} furnaces] {tag}")
                if not ok:
                    continue
                if remaining:
                    spx, spy = gx, gy - 1
                    if self.free(spx, spy) and self.free(spx + 1, spy):
                        self._place_splitter(spx, spy, S)
                        src = (spx + 1, spy)
                    else:
                        src = (gx, gy)
            by_res[res] = []   # consumed; second entry of same res gets none
        return log

    # ------------------------------------------------------------- output
    def render(self) -> str:
        out = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                if self.blocked[y][x]:
                    row.append("#")
                elif self.canvas[y][x]:
                    row.append(self.canvas[y][x])
                else:
                    row.append(".")
            out.append("".join(row))
        for ex, ey, res in self.entries:
            line = list(out[ey])
            line[ex] = res
            out[ey] = "".join(line)
        return "\n".join(out)

    def stats(self) -> str:
        by_res: dict[str, int] = {}
        for c in self.cells:
            by_res[c.resource] = by_res.get(c.resource, 0) + c.furnaces
        lines = [f"cells placed: {len(self.cells)}"]
        for res, n in sorted(by_res.items()):
            belts = n / 48.0
            lines.append(f"  {RESOURCE_NAMES.get(res, res)}: {n} stone "
                         f"furnaces ({belts:.2f} yellow belts of plates)")
        lines.append(f"entities: {len(self.entities)}")
        return "\n".join(lines)

    def blueprint(self) -> str:
        ents = []
        for i, e in enumerate(self.entities, 1):
            d = {"entity_number": i, "name": e.name,
                 "position": {"x": e.x, "y": e.y}}
            if e.direction:
                d["direction"] = e.direction
            d.update(e.extra)
            ents.append(d)
        bp = {"blueprint": {
            "item": "blueprint",
            "label": "Auto smelter (planner)",
            "entities": ents,
            "icons": [{"signal": {"type": "item", "name": "stone-furnace"},
                       "index": 1}],
            "version": 281479275675648}}
        raw = json.dumps(bp, separators=(",", ":")).encode()
        return "0" + base64.b64encode(zlib.compress(raw, 9)).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description="Factorio smelter planner")
    ap.add_argument("map", help="path to map text file")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="cap on number of smelting cells")
    ap.add_argument("--blueprint", metavar="FILE",
                    help="write importable blueprint string here")
    ap.add_argument("--layout", metavar="FILE",
                    help="write ASCII layout here (default: stdout)")
    args = ap.parse_args()

    with open(args.map) as f:
        rows = [line.rstrip("\n") for line in f if line.strip("\n")]
    width = max(len(r) for r in rows)
    grid = [list(r.ljust(width, ".")) for r in rows]

    p = Planner(grid)
    p.pack_cells(args.max_cells)
    log = p.route_all()

    layout = p.render()
    report = (layout + "\n\n--- routing ---\n" + "\n".join(log)
              + "\n\n--- stats ---\n" + p.stats() + "\n")
    if args.layout:
        with open(args.layout, "w") as f:
            f.write(report)
    else:
        print(report)
    if args.blueprint:
        with open(args.blueprint, "w") as f:
            f.write(p.blueprint() + "\n")
        print(f"blueprint written to {args.blueprint}", file=sys.stderr)


if __name__ == "__main__":
    main()
