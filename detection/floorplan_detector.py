import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import math
import os
from collections import defaultdict


# ════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════

@dataclass
class LineSegment:
    """A detected line segment in vector form."""
    x1: float
    y1: float
    x2: float
    y2: float
    angle: float
    length: float
    thickness: float = 1.0
    group_id: int = -1
    gap_positions: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def is_horizontal(self) -> bool:
        return self.angle < 5 or self.angle > 175

    @property
    def is_vertical(self) -> bool:
        return 85 < self.angle < 95

    def angle_diff(self, other: 'LineSegment') -> float:
        diff = abs(self.angle - other.angle)
        return min(diff, 180 - diff)

    def is_parallel_to(self, other: 'LineSegment', tolerance: float = 5.0) -> bool:
        return self.angle_diff(other) < tolerance

    def perpendicular_distance_to(self, other: 'LineSegment') -> float:
        angle_rad = math.radians(self.angle)
        nx = -math.sin(angle_rad)
        ny = math.cos(angle_rad)
        dx = other.midpoint[0] - self.midpoint[0]
        dy = other.midpoint[1] - self.midpoint[1]
        return abs(dx * nx + dy * ny)

    def project_point(self, px: float, py: float) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return 0.0
        return ((px - self.x1) * dx + (py - self.y1) * dy) / length

    def collinear_overlap(self, other: 'LineSegment') -> float:
        p1 = self.project_point(self.x1, self.y1)
        p2 = self.project_point(self.x2, self.y2)
        p3 = self.project_point(other.x1, other.y1)
        p4 = self.project_point(other.x2, other.y2)
        a1, a2 = min(p1, p2), max(p1, p2)
        b1, b2 = min(p3, p4), max(p3, p4)
        overlap = max(0, min(a2, b2) - max(a1, b1))
        shorter = min(a2 - a1, b2 - b1)
        return overlap / (shorter + 1e-6)


@dataclass
class WallSegment:
    """A wall = pair of parallel line segments."""
    id: int
    line_a: LineSegment
    line_b: LineSegment
    thickness: float
    orientation: str
    bbox: Tuple[int, int, int, int]
    center_line: Tuple[Tuple[float, float], Tuple[float, float]]
    mask: Optional[np.ndarray] = None
    gap_positions: List[Tuple[float, float]] = field(default_factory=list)
    is_structural: bool = True
    is_exterior: bool = False
    thickness_class: str = "normal"


@dataclass
class Door:
    """A detected door."""
    position: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    orientation: str
    wall_id: int = -1
    confidence: float = 0.0
    gap_width: float = 0.0
    detection_method: str = ""
    is_exterior_door: bool = False


class DebugVisualizer:
    def __init__(self, output_dir: str, enabled: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        self.step_counter = 0
        # Ordered list of (name, path) for grid compilation
        self._saved: List[Tuple[str, str]] = []
        if enabled:
            os.makedirs(os.path.join(output_dir, "stages"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "details"), exist_ok=True)

    def save(self, name: str, image: np.ndarray, subdir: str = "stages"):
        if not self.enabled:
            return
        self.step_counter += 1
        path = os.path.join(self.output_dir, subdir, f"{self.step_counter:02d}_{name}.png")
        cv2.imwrite(path, image)
        self._saved.append((name, path))
        print(f"  [DEBUG] {path}")

    def save_log(self, filename: str, lines: List[str]):
        if not self.enabled:
            return
        path = os.path.join(self.output_dir, "details", filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def compile_grid(
        self,
        output_filename: str = "debug_grid.png",
        thumb_w: int = 400,
        thumb_h: int = 300,
        cols: int = 4,
        bg_color: Tuple[int, int, int] = (30, 30, 30),
        label_color: Tuple[int, int, int] = (220, 220, 220),
        padding: int = 8,
    ) -> Optional[str]:
        """
        Compiles every saved debug image into a single grid PNG.

        Parameters
        ----------
        output_filename : name of the output file (placed in output_dir)
        thumb_w / thumb_h : cell thumbnail size in pixels
        cols : number of columns in the grid
        bg_color : RGB background / gutter colour
        label_color : RGB colour for the step labels
        padding : gutter width in pixels around each cell
        """
        if not self.enabled or not self._saved:
            return None

        label_h = 20  # pixels reserved for the step label below each thumb
        cell_w = thumb_w + padding * 2
        cell_h = thumb_h + padding * 2 + label_h

        n = len(self._saved)
        rows = math.ceil(n / cols)

        grid_w = cell_w * cols
        grid_h = cell_h * rows

        grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        font_thick = 1

        for idx, (name, path) in enumerate(self._saved):
            img = cv2.imread(path)
            if img is None:
                # Placeholder for missing / unreadable images
                img = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                cv2.putText(img, "N/A", (thumb_w // 2 - 15, thumb_h // 2),
                            font, 0.8, (80, 80, 80), 1)
            else:
                # Convert grayscale saves to BGR for uniform display
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            row, col = divmod(idx, cols)
            x0 = col * cell_w + padding
            y0 = row * cell_h + padding

            grid[y0:y0 + thumb_h, x0:x0 + thumb_w] = img

            # Step label
            label = f"{idx + 1:02d}. {name}"
            text_y = y0 + thumb_h + label_h - 5
            # Shadow for readability
            cv2.putText(grid, label, (x0 + 1, text_y + 1), font, font_scale, (0, 0, 0), font_thick + 1)
            cv2.putText(grid, label, (x0, text_y), font, font_scale, label_color, font_thick)

        out_path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(out_path, grid)
        print(f"\n  [DEBUG GRID] {out_path}  ({cols}×{rows}, {n} images)")
        return out_path


# ════════════════════════════════════════════════════════════════
# Main detector v4.2
# ════════════════════════════════════════════════════════════════

class FloorplanDetector:
    def __init__(
            self,
            wall_thickness_range: Tuple[int, int] = (4, 35),
            min_wall_length: int = 40,
            door_min_gap: int = 15,
            door_max_gap: int = 45,
            debug_dir: str = "debug_output",
            debug: bool = True,
    ):
        self.wall_thickness_range = wall_thickness_range
        self.min_wall_length = min_wall_length
        self.door_min_gap = door_min_gap
        self.door_max_gap = door_max_gap
        self.dbg = DebugVisualizer(debug_dir, enabled=debug)

        self._thickness_clusters: Dict[str, float] = {}
        self._dominant_thickness: float = 0.0
        self._interior_mask: Optional[np.ndarray] = None

    # ────────────────────────────────────────────────────────────
    # STAGE 1-4: Preprocessing, segments, aggregation
    # ────────────────────────────────────────────────────────────

    def _load_and_binarize(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.dbg.save("original", img)

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 41, 10
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        self.dbg.save("binary", binary)
        return img, gray, binary

    def _deskew(self, img: np.ndarray, gray: np.ndarray,
                binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        lines = cv2.HoughLinesP(binary, 1, np.pi / 720, 80, minLineLength=60, maxLineGap=10)
        if lines is None:
            return img, gray, binary, 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 15 or abs(angle) > 165:
                dev = angle if abs(angle) < 15 else (angle - 180 if angle > 0 else angle + 180)
                angles.append((dev, length))

        if not angles:
            return img, gray, binary, 0.0

        devs = np.array([d for d, _ in angles])
        weights = np.array([l for _, l in angles])
        sorted_idx = np.argsort(devs)
        cumsum = np.cumsum(weights[sorted_idx])
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        skew = devs[sorted_idx[min(median_idx, len(devs) - 1)]]

        if abs(skew) < 0.1 or abs(skew) > 5.0:
            return img, gray, binary, 0.0

        h, w = binary.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), skew, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        gray = cv2.warpAffine(gray, M, (w, h), borderValue=255)
        binary = cv2.warpAffine(binary, M, (w, h), borderValue=0)
        self.dbg.save("deskewed", binary)
        return img, gray, binary, skew

    def _detect_segments(self, gray: np.ndarray, binary: np.ndarray) -> List[LineSegment]:
        segments = []
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            lines, widths, _, _ = lsd.detect(gray)
        except:
            lsd = cv2.createLineSegmentDetector()
            lines, widths, _, _ = lsd.detect(gray)

        if lines is None:
            return segments

        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 8:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            thickness = float(widths[i][0]) if widths is not None else 1.0
            segments.append(LineSegment(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                angle=angle, length=length, thickness=thickness
            ))

        vis = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        for seg in segments:
            color = (0, 255, 0) if seg.is_horizontal else ((255, 0, 0) if seg.is_vertical else (0, 255, 255))
            cv2.line(vis, (int(seg.x1), int(seg.y1)), (int(seg.x2), int(seg.y2)), color, 1)
        self.dbg.save("raw_segments", vis)
        return segments

    def _aggregate_with_gaps(self, segments: List[LineSegment],
                             binary: np.ndarray) -> List[LineSegment]:
        if not segments:
            return []

        angle_groups: Dict[int, List[LineSegment]] = defaultdict(list)
        for seg in segments:
            if seg.length < 15:
                continue
            angle_bin = int(round(seg.angle / 5) * 5) % 180
            angle_groups[angle_bin].append(seg)

        result = []
        for angle_bin, group in angle_groups.items():
            merged = self._merge_collinear_fixed(group, angle_bin)
            result.extend(merged)

        vis = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        for seg in result:
            color = (0, 255, 0) if seg.is_horizontal else ((255, 0, 0) if seg.is_vertical else (0, 255, 255))
            cv2.line(vis, (int(seg.x1), int(seg.y1)), (int(seg.x2), int(seg.y2)), color, 2)
            for gx, gy in seg.gap_positions:
                cv2.circle(vis, (int(gx), int(gy)), 4, (0, 0, 255), -1)
        self.dbg.save("aggregated_with_gaps", vis)
        return result

    def _merge_collinear_fixed(self, segments: List[LineSegment], angle_bin: int) -> List[LineSegment]:
        if not segments:
            return []

        angle_rad = math.radians(angle_bin)
        dir_x, dir_y = math.cos(angle_rad), math.sin(angle_rad)
        norm_x, norm_y = -dir_y, dir_x

        def perp_coord(seg):
            return seg.midpoint[0] * norm_x + seg.midpoint[1] * norm_y

        def along_coord(seg):
            p1 = seg.x1 * dir_x + seg.y1 * dir_y
            p2 = seg.x2 * dir_x + seg.y2 * dir_y
            return min(p1, p2), max(p1, p2)

        sorted_segs = sorted(segments, key=perp_coord)
        groups: List[List[LineSegment]] = []
        current_group = [sorted_segs[0]]
        group_perp_for_grouping = perp_coord(sorted_segs[0])

        for seg in sorted_segs[1:]:
            seg_perp = perp_coord(seg)
            if abs(seg_perp - group_perp_for_grouping) <= 3.0:
                current_group.append(seg)
            else:
                groups.append(current_group)
                current_group = [seg]
                group_perp_for_grouping = seg_perp
        groups.append(current_group)

        result = []
        for group in groups:
            if not group:
                continue
            group_perp = np.mean([perp_coord(s) for s in group])
            group_sorted = sorted(group, key=lambda s: along_coord(s)[0])

            cur_start, cur_end = along_coord(group_sorted[0])
            cur_gaps = []
            cur_thickness = group_sorted[0].thickness
            total_length = group_sorted[0].length

            for seg in group_sorted[1:]:
                seg_start, seg_end = along_coord(seg)
                gap = seg_start - cur_end

                if gap <= 50:
                    if gap >= 12:
                        gap_along = (cur_end + seg_start) / 2
                        gap_x = gap_along * dir_x + group_perp * norm_x
                        gap_y = gap_along * dir_y + group_perp * norm_y
                        cur_gaps.append((gap_x, gap_y))
                    cur_end = max(cur_end, seg_end)
                    cur_thickness = (cur_thickness * total_length + seg.thickness * seg.length) / (
                                total_length + seg.length)
                    total_length += seg.length
                else:
                    merged = self._create_merged_segment(cur_start, cur_end, group_perp, dir_x, dir_y, norm_x, norm_y,
                                                         angle_bin, cur_thickness, cur_gaps)
                    if merged.length >= self.min_wall_length * 0.5:
                        result.append(merged)
                    cur_start, cur_end = seg_start, seg_end
                    cur_gaps = []
                    cur_thickness = seg.thickness
                    total_length = seg.length

            merged = self._create_merged_segment(cur_start, cur_end, group_perp, dir_x, dir_y, norm_x, norm_y,
                                                 angle_bin, cur_thickness, cur_gaps)
            if merged.length >= self.min_wall_length * 0.5:
                result.append(merged)
        return result

    def _create_merged_segment(self, along_start, along_end, perp, dir_x, dir_y, norm_x, norm_y, angle, thickness,
                               gaps):
        x1 = along_start * dir_x + perp * norm_x
        y1 = along_start * dir_y + perp * norm_y
        x2 = along_end * dir_x + perp * norm_x
        y2 = along_end * dir_y + perp * norm_y
        return LineSegment(x1=x1, y1=y1, x2=x2, y2=y2, angle=angle % 180, length=along_end - along_start,
                           thickness=thickness, gap_positions=gaps.copy())

    # ────────────────────────────────────────────────────────────
    # STAGE 5: Bimodal thickness estimation
    # ────────────────────────────────────────────────────────────

    def _estimate_thickness_bimodal(self, segments: List[LineSegment], binary: np.ndarray) -> Dict[str, float]:
        thicknesses = []
        h, w = binary.shape

        for seg in segments:
            if seg.length < 30:
                continue
            samples = []
            for t in [0.25, 0.5, 0.75]:
                px = seg.x1 + t * (seg.x2 - seg.x1)
                py = seg.y1 + t * (seg.y2 - seg.y1)
                angle_rad = math.radians(seg.angle + 90)
                perp_dx, perp_dy = math.cos(angle_rad), math.sin(angle_rad)

                count = 0
                for direction in [1, -1]:
                    for step in range(1, 30):
                        sx = int(px + direction * step * perp_dx)
                        sy = int(py + direction * step * perp_dy)
                        if 0 <= sx < w and 0 <= sy < h and binary[sy, sx] > 0:
                            count += 1
                        else:
                            break
                cx, cy = int(px), int(py)
                if 0 <= cx < w and 0 <= cy < h and binary[cy, cx] > 0:
                    count += 1
                if count >= 3:
                    samples.append(count)
            if samples:
                thicknesses.append(np.median(samples))

        if not thicknesses:
            default = (self.wall_thickness_range[0] + self.wall_thickness_range[1]) / 2
            self._thickness_clusters = {'thin': default * 0.7, 'normal': default, 'thick': default * 1.4}
            self._dominant_thickness = default
            return self._thickness_clusters

        thicknesses = np.array(thicknesses)

        bins = np.arange(self.wall_thickness_range[0] - 1, self.wall_thickness_range[1] + 2, 2.0)
        hist, edges = np.histogram(thicknesses, bins=bins)

        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 2:
                peaks.append((hist[i], (edges[i] + edges[i + 1]) / 2))

        peaks.sort(reverse=True)

        if len(peaks) >= 2:
            t1, t2 = peaks[0][1], peaks[1][1]
            thin_t = min(t1, t2)
            thick_t = max(t1, t2)
            normal_t = (thin_t + thick_t) / 2
        else:
            normal_t = np.median(thicknesses)
            thin_t = normal_t * 0.7
            thick_t = normal_t * 1.4

        self._thickness_clusters = {
            'thin': thin_t,
            'normal': normal_t,
            'thick': thick_t
        }
        self._dominant_thickness = normal_t

        print(f"  Thickness clusters: thin={thin_t:.1f}, normal={normal_t:.1f}, thick={thick_t:.1f}")
        return self._thickness_clusters

    def _classify_wall_thickness(self, thickness: float) -> str:
        thin = self._thickness_clusters.get('thin', 8)
        thick = self._thickness_clusters.get('thick', 14)

        if thickness < (thin + self._dominant_thickness) / 2:
            return 'thin'
        elif thickness > (thick + self._dominant_thickness) / 2:
            return 'thick'
        return 'normal'

    # ────────────────────────────────────────────────────────────
    # STAGE 6: Wall pairing
    # ────────────────────────────────────────────────────────────

    def _pair_walls_optimal(self, segments: List[LineSegment], binary: np.ndarray) -> List[WallSegment]:
        min_thick, max_thick = self.wall_thickness_range
        valid_segments = [s for s in segments if s.length >= self.min_wall_length * 0.4]

        if len(valid_segments) < 2:
            return []

        n = len(valid_segments)
        INF = 1e9
        cost_matrix = np.full((n, n), INF)

        for i in range(n):
            for j in range(i + 1, n):
                seg_a, seg_b = valid_segments[i], valid_segments[j]

                if not seg_a.is_parallel_to(seg_b, tolerance=5.0):
                    continue

                perp_dist = seg_a.perpendicular_distance_to(seg_b)
                if perp_dist < min_thick or perp_dist > max_thick:
                    continue

                overlap = seg_a.collinear_overlap(seg_b)
                if overlap < 0.25:
                    continue

                if not self._validate_wall_multisample(binary, seg_a, seg_b):
                    continue

                thickness_class = self._classify_wall_thickness(perp_dist)
                cluster_center = self._thickness_clusters.get(thickness_class, self._dominant_thickness)
                thickness_penalty = abs(perp_dist - cluster_center) / (cluster_center + 1)
                overlap_bonus = 1.0 - overlap
                cost = thickness_penalty * 0.5 + overlap_bonus

                cost_matrix[i, j] = cost
                cost_matrix[j, i] = cost

        walls = []
        wall_id = 0
        used = set()

        pairs = [(cost_matrix[i, j], i, j) for i in range(n) for j in range(i + 1, n) if cost_matrix[i, j] < INF]
        pairs.sort(key=lambda x: x[0])

        for cost, i, j in pairs:
            if i in used or j in used:
                continue
            wall = self._create_wall(wall_id, valid_segments[i], valid_segments[j], binary)
            if wall is not None:
                walls.append(wall)
                wall_id += 1
                used.add(i)
                used.add(j)

        print(f"  Paired {len(walls)} walls from {len(valid_segments)} segments")

        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for wall in walls:
            x, y, bw, bh = wall.bbox
            color = (0, 255, 0) if wall.orientation == 'horizontal' else (
                (255, 0, 0) if wall.orientation == 'vertical' else (0, 255, 255))
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(vis, f"W{wall.id} t={wall.thickness:.0f} {wall.thickness_class[0]}", (x, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        self.dbg.save("paired_walls", vis)
        return walls

    def _validate_wall_multisample(self, binary: np.ndarray, seg_a: LineSegment, seg_b: LineSegment) -> bool:
        h, w = binary.shape
        angle_rad = math.radians(seg_a.angle)
        dir_x, dir_y = math.cos(angle_rad), math.sin(angle_rad)
        perp_x, perp_y = -math.sin(angle_rad), math.cos(angle_rad)

        p1, p2 = seg_a.project_point(seg_a.x1, seg_a.y1), seg_a.project_point(seg_a.x2, seg_a.y2)
        p3, p4 = seg_a.project_point(seg_b.x1, seg_b.y1), seg_a.project_point(seg_b.x2, seg_b.y2)
        a1, a2 = min(p1, p2), max(p1, p2)
        b1, b2 = min(p3, p4), max(p3, p4)
        overlap_start, overlap_end = max(a1, b1), min(a2, b2)

        if overlap_end - overlap_start < 20:
            return False

        perp_a = seg_a.midpoint[0] * perp_x + seg_a.midpoint[1] * perp_y
        perp_b = seg_b.midpoint[0] * perp_x + seg_b.midpoint[1] * perp_y
        perp_center = (perp_a + perp_b) / 2
        perp_dist = abs(perp_a - perp_b)
        half_dist = perp_dist / 2 + 2

        edge_densities, mid_densities, total_densities = [], [], []

        for t in np.linspace(0.1, 0.9, 7):
            along = overlap_start + t * (overlap_end - overlap_start)
            cx = along * dir_x + perp_center * perp_x
            cy = along * dir_y + perp_center * perp_y

            samples = []
            for s in np.linspace(-half_dist, half_dist, max(5, int(perp_dist))):
                sx, sy = int(cx + s * perp_x), int(cy + s * perp_y)
                if 0 <= sx < w and 0 <= sy < h:
                    samples.append(1.0 if binary[sy, sx] > 0 else 0.0)
                else:
                    samples.append(0.0)

            if len(samples) < 3:
                continue

            samples = np.array(samples)
            n_s = len(samples)
            edge_size = max(1, n_s // 4)
            edge_density = (np.mean(samples[:edge_size]) + np.mean(samples[-edge_size:])) / 2
            mid_start, mid_end = n_s // 4, 3 * n_s // 4
            mid_density = np.mean(samples[mid_start:mid_end]) if mid_end > mid_start else 0

            edge_densities.append(edge_density)
            mid_densities.append(mid_density)
            total_densities.append(np.mean(samples))

        if not edge_densities:
            return False

        avg_edge = np.mean(edge_densities)
        avg_mid = np.mean(mid_densities)
        avg_total = np.mean(total_densities)

        if avg_total > 0.6:
            return True
        if avg_edge > 0.25 and avg_edge > avg_mid * 1.1:
            return True
        if avg_edge > 0.2 and avg_total > 0.1:
            return True
        if avg_mid > avg_edge * 1.5 and avg_mid > 0.3:
            return False
        return avg_total > 0.08

    def _get_wall_polygon(self, seg_a: LineSegment, seg_b: LineSegment) -> Optional[np.ndarray]:
        angle_rad = math.radians(seg_a.angle)
        dir_x, dir_y = math.cos(angle_rad), math.sin(angle_rad)
        norm_x, norm_y = -math.sin(angle_rad), math.cos(angle_rad)

        projs = [
            (seg_a.x1 * dir_x + seg_a.y1 * dir_y, seg_a.x1, seg_a.y1, 'a'),
            (seg_a.x2 * dir_x + seg_a.y2 * dir_y, seg_a.x2, seg_a.y2, 'a'),
            (seg_b.x1 * dir_x + seg_b.y1 * dir_y, seg_b.x1, seg_b.y1, 'b'),
            (seg_b.x2 * dir_x + seg_b.y2 * dir_y, seg_b.x2, seg_b.y2, 'b'),
        ]
        projs.sort(key=lambda x: x[0])

        if projs[1][3] == projs[2][3]:
            return None

        start_proj, end_proj = projs[1][0], projs[2][0]
        if end_proj - start_proj < 10:
            return None

        perp_a = seg_a.midpoint[0] * norm_x + seg_a.midpoint[1] * norm_y
        perp_b = seg_b.midpoint[0] * norm_x + seg_b.midpoint[1] * norm_y

        corners = []
        for along in [start_proj, end_proj]:
            for perp in [perp_a, perp_b]:
                corners.append([int(along * dir_x + perp * norm_x), int(along * dir_y + perp * norm_y)])

        return np.array([corners[0], corners[2], corners[3], corners[1]], dtype=np.int32)

    def _create_wall(self, wall_id: int, seg_a: LineSegment, seg_b: LineSegment, binary: np.ndarray) -> Optional[
        WallSegment]:
        pts = self._get_wall_polygon(seg_a, seg_b)
        if pts is None:
            return None

        x_min, x_max = max(0, pts[:, 0].min()), min(binary.shape[1] - 1, pts[:, 0].max())
        y_min, y_max = max(0, pts[:, 1].min()), min(binary.shape[0] - 1, pts[:, 1].max())
        bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        center = ((seg_a.midpoint[0] + seg_b.midpoint[0]) / 2, (seg_a.midpoint[1] + seg_b.midpoint[1]) / 2)
        angle_rad = math.radians(seg_a.angle)
        half_len = min(seg_a.length, seg_b.length) / 2
        p1 = (center[0] - half_len * math.cos(angle_rad), center[1] - half_len * math.sin(angle_rad))
        p2 = (center[0] + half_len * math.cos(angle_rad), center[1] + half_len * math.sin(angle_rad))

        orientation = 'horizontal' if seg_a.is_horizontal else ('vertical' if seg_a.is_vertical else 'diagonal')
        thickness = seg_a.perpendicular_distance_to(seg_b)
        thickness_class = self._classify_wall_thickness(thickness)
        gaps = seg_a.gap_positions + seg_b.gap_positions

        return WallSegment(
            id=wall_id, line_a=seg_a, line_b=seg_b,
            thickness=thickness, orientation=orientation,
            bbox=bbox, center_line=(p1, p2),
            gap_positions=gaps, thickness_class=thickness_class
        )

    # ────────────────────────────────────────────────────────────
    # STAGE 7: Network filtering
    # ────────────────────────────────────────────────────────────

    def _filter_walls_network(self, walls: List[WallSegment], binary: np.ndarray) -> List[WallSegment]:
        if len(walls) <= 2:
            return walls

        tolerance = max(self._dominant_thickness * 0.5, 10)

        n = len(walls)
        adjacency: Dict[int, Set[int]] = defaultdict(set)

        for i in range(n):
            for j in range(i + 1, n):
                if self._walls_connect_extended(walls[i], walls[j], tolerance):
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        visited = set()
        components: List[Set[int]] = []
        for start in range(n):
            if start in visited:
                continue
            component = set()
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(adjacency[node] - visited)
            components.append(component)

        largest = max(components, key=len) if components else set()
        structural = set()

        for i, wall in enumerate(walls):
            degree = len(adjacency[i])
            wall_length = max(wall.line_a.length, wall.line_b.length)

            if i in largest and len(largest) >= 2:
                structural.add(i)
            elif degree >= 1:
                structural.add(i)
            elif wall_length > self.min_wall_length * 1.5:
                structural.add(i)
            elif wall.thickness_class in ['normal', 'thick'] and wall_length > self.min_wall_length:
                structural.add(i)

        filtered = [w for i, w in enumerate(walls) if i in structural]

        if len(filtered) < len(walls) * 0.4 and len(walls) > 3:
            print(f"  Warning: filter too aggressive, keeping all {len(walls)} walls")
            filtered = walls

        print(f"  Network filter: {len(walls)} → {len(filtered)} walls")

        vis = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
        for i, wall in enumerate(walls):
            x, y, bw, bh = wall.bbox
            color = (0, 255, 0) if i in structural else (0, 0, 180)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(vis, f"d={len(adjacency[i])}", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        for i in range(n):
            for j in adjacency[i]:
                if j > i:
                    c1, c2 = walls[i].center_line, walls[j].center_line
                    p1 = (int((c1[0][0] + c1[1][0]) / 2), int((c1[0][1] + c1[1][1]) / 2))
                    p2 = (int((c2[0][0] + c2[1][0]) / 2), int((c2[0][1] + c2[1][1]) / 2))
                    cv2.line(vis, p1, p2, (100, 100, 100), 1)
        self.dbg.save("wall_network", vis)

        return filtered

    def _walls_connect_extended(self, wa: WallSegment, wb: WallSegment, tolerance: float) -> bool:
        t = int(tolerance)
        ax, ay, aw, ah = wa.bbox
        bx, by, bw, bh = wb.bbox

        a_ext = (ax - t, ay - t, ax + aw + t, ay + ah + t)
        b_ext = (bx - t, by - t, bx + bw + t, by + bh + t)

        if a_ext[0] > b_ext[2] or b_ext[0] > a_ext[2] or a_ext[1] > b_ext[3] or b_ext[1] > a_ext[3]:
            return False

        if wa.orientation != wb.orientation:
            return True

        ends_a = [(wa.line_a.x1, wa.line_a.y1), (wa.line_a.x2, wa.line_a.y2)]
        ends_b = [(wb.line_a.x1, wb.line_a.y1), (wb.line_a.x2, wb.line_a.y2)]

        for ea in ends_a:
            for eb in ends_b:
                if math.hypot(ea[0] - eb[0], ea[1] - eb[1]) < tolerance * 2:
                    if wa.line_a.perpendicular_distance_to(wb.line_a) < tolerance:
                        return True
        return False

    # ────────────────────────────────────────────────────────────
    # STAGE 8: Generate interior mask
    # ────────────────────────────────────────────────────────────

    def _generate_wall_masks(self, walls: List[WallSegment], binary: np.ndarray) -> np.ndarray:
        h, w = binary.shape
        combined = np.zeros((h, w), dtype=np.uint8)

        for wall in walls:
            pts = self._get_wall_polygon(wall.line_a, wall.line_b)
            if pts is not None:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                wall.mask = mask
                combined = cv2.bitwise_or(combined, mask)

        self._interior_mask = self._compute_interior_mask(binary, combined)

        self.dbg.save("wall_masks", combined)
        if self._interior_mask is not None:
            self.dbg.save("interior_mask", self._interior_mask)

        return combined

    def _compute_interior_mask(self, binary: np.ndarray, wall_mask: np.ndarray) -> Optional[np.ndarray]:
        h, w = binary.shape

        structure = cv2.bitwise_or(binary, wall_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(structure, cv2.MORPH_CLOSE, kernel, iterations=2)

        empty = cv2.bitwise_not(closed)

        contours, _ = cv2.findContours(empty, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

        interior = np.zeros((h, w), dtype=np.uint8)
        if len(contours_sorted) > 1:
            for cnt in contours_sorted[1:]:
                if cv2.contourArea(cnt) > 500:
                    cv2.drawContours(interior, [cnt], -1, 255, -1)

        return interior

    # ────────────────────────────────────────────────────────────
    # STAGE 9: Door detection
    # ────────────────────────────────────────────────────────────

    def _detect_doors(self, walls: List[WallSegment], binary: np.ndarray) -> List[Door]:
        doors = []
        h, w = binary.shape

        for wall in walls:
            for gap_x, gap_y in wall.gap_positions:
                gx, gy = int(gap_x), int(gap_y)
                if not (0 <= gx < w and 0 <= gy < h):
                    continue

                gap_width = self._measure_gap_width_correct(binary, gx, gy, wall)

                if not (self.door_min_gap <= gap_width <= self.door_max_gap):
                    continue

                if not self._validate_door_topology(gx, gy, wall, binary):
                    continue

                door = self._create_door(wall, gx, gy, gap_width, 'gap_preserved', 0.8)
                if door:
                    doors.append(door)

        for wall in walls:
            gap_doors = self._scan_wall_for_gaps(wall, binary)
            doors.extend(gap_doors)

        doors = self._deduplicate_doors(doors)

        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for door in doors:
            x, y, dw, dh = door.bounding_box
            color = (0, 255, 0) if door.confidence > 0.6 else (0, 200, 255)
            cv2.rectangle(vis, (x, y), (x + dw, y + dh), color, 2)
            cv2.circle(vis, door.position, 4, color, -1)
            label = f"{door.detection_method[:3]}"
            if door.is_exterior_door:
                label += " EXT"
            cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        self.dbg.save("doors_detected", vis)

        print(f"  Detected {len(doors)} doors")
        return doors

    def _measure_gap_width_correct(self, binary: np.ndarray, x: int, y: int, wall: WallSegment) -> float:
        h, w = binary.shape

        if wall.orientation == 'horizontal':
            dx, dy = 1, 0
        elif wall.orientation == 'vertical':
            dx, dy = 0, 1
        else:
            angle_rad = math.radians(wall.line_a.angle)
            dx, dy = math.cos(angle_rad), math.sin(angle_rad)

        max_search = 60
        half_thick = int(wall.thickness / 2) + 2

        def find_edge(start_x, start_y, step_dx, step_dy):
            for step in range(1, max_search + 1):
                px = int(start_x + step * step_dx)
                py = int(start_y + step * step_dy)

                if not (0 <= px < w and 0 <= py < h):
                    return None

                has_ink = False
                for offset in range(-half_thick, half_thick + 1):
                    if wall.orientation == 'horizontal':
                        check_x, check_y = px, py + offset
                    elif wall.orientation == 'vertical':
                        check_x, check_y = px + offset, py
                    else:
                        perp_angle = math.radians(wall.line_a.angle + 90)
                        check_x = int(px + offset * math.cos(perp_angle))
                        check_y = int(py + offset * math.sin(perp_angle))

                    if 0 <= check_x < w and 0 <= check_y < h:
                        if binary[check_y, check_x] > 0:
                            has_ink = True
                            break

                if has_ink:
                    return step

            return None

        dist_pos = find_edge(x, y, dx, dy)
        dist_neg = find_edge(x, y, -dx, -dy)

        if dist_pos is None or dist_neg is None:
            return 0.0

        return float(dist_pos + dist_neg)

    def _validate_door_topology(self, x: int, y: int, wall: WallSegment, binary: np.ndarray) -> bool:
        h, w = binary.shape

        if wall.orientation == 'horizontal':
            perp_dx, perp_dy = 0, 1
        elif wall.orientation == 'vertical':
            perp_dx, perp_dy = 1, 0
        else:
            angle_rad = math.radians(wall.line_a.angle + 90)
            perp_dx, perp_dy = math.cos(angle_rad), math.sin(angle_rad)

        check_dist = int(wall.thickness) + 10

        def check_empty_space(cx, cy, direction):
            empty_count = 0
            total_count = 0

            for step in range(5, check_dist):
                px = int(cx + direction * step * perp_dx)
                py = int(cy + direction * step * perp_dy)

                if not (0 <= px < w and 0 <= py < h):
                    return False, False

                total_count += 1
                if binary[py, px] == 0:
                    empty_count += 1

            if total_count == 0:
                return False, False

            is_empty = empty_count / total_count > 0.7

            is_interior = False
            if self._interior_mask is not None:
                sample_x = int(x + direction * check_dist * perp_dx)
                sample_y = int(y + direction * check_dist * perp_dy)
                if 0 <= sample_x < w and 0 <= sample_y < h:
                    is_interior = self._interior_mask[sample_y, sample_x] > 0

            return is_empty, is_interior

        empty_pos, interior_pos = check_empty_space(x, y, 1)
        empty_neg, interior_neg = check_empty_space(x, y, -1)

        if empty_pos and empty_neg and interior_pos and interior_neg:
            return True

        if empty_pos and empty_neg and (interior_pos != interior_neg):
            return True

        if self._interior_mask is None and empty_pos and empty_neg:
            return True

        return False

    def _scan_wall_for_gaps(self, wall: WallSegment, binary: np.ndarray) -> List[Door]:
        doors = []
        h, w = binary.shape

        p1, p2 = wall.center_line
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if length < 30:
            return doors

        num_samples = int(length)
        dx = (p2[0] - p1[0]) / num_samples
        dy = (p2[1] - p1[1]) / num_samples

        in_gap = False
        gap_start_idx = 0
        positions = []

        for i in range(num_samples):
            px, py = p1[0] + i * dx, p1[1] + i * dy
            positions.append((px, py))
            has_ink = self._check_wall_ink(binary, int(px), int(py), wall)

            if not has_ink and not in_gap:
                in_gap = True
                gap_start_idx = i
            elif has_ink and in_gap:
                in_gap = False
                gap_width = i - gap_start_idx

                if self.door_min_gap <= gap_width <= self.door_max_gap:
                    center_idx = (gap_start_idx + i) // 2
                    gx, gy = int(positions[center_idx][0]), int(positions[center_idx][1])

                    if self._validate_door_topology(gx, gy, wall, binary):
                        door = self._create_door(wall, gx, gy, float(gap_width), 'gap_scan', 0.6)
                        if door:
                            doors.append(door)

        return doors

    def _check_wall_ink(self, binary: np.ndarray, x: int, y: int, wall: WallSegment) -> bool:
        h, w = binary.shape
        half_thick = int(wall.thickness / 2) + 1

        if wall.orientation == 'horizontal':
            for dy in range(-half_thick, half_thick + 1):
                ny = y + dy
                if 0 <= ny < h and 0 <= x < w and binary[ny, x] > 0:
                    return True
        elif wall.orientation == 'vertical':
            for dx in range(-half_thick, half_thick + 1):
                nx = x + dx
                if 0 <= y < h and 0 <= nx < w and binary[y, nx] > 0:
                    return True
        else:
            angle_rad = math.radians(wall.line_a.angle + 90)
            for step in range(-half_thick, half_thick + 1):
                nx = int(x + step * math.cos(angle_rad))
                ny = int(y + step * math.sin(angle_rad))
                if 0 <= nx < w and 0 <= ny < h and binary[ny, nx] > 0:
                    return True
        return False

    def _create_door(self, wall: WallSegment, gx: int, gy: int, gap_width: float, method: str,
                     confidence: float) -> Optional[Door]:
        half_gap = int(gap_width / 2) + 3

        if wall.orientation == 'horizontal':
            bbox = (gx - half_gap, wall.bbox[1] - 2, int(gap_width) + 6, wall.bbox[3] + 4)
        elif wall.orientation == 'vertical':
            bbox = (wall.bbox[0] - 2, gy - half_gap, wall.bbox[2] + 4, int(gap_width) + 6)
        else:
            bbox = (gx - half_gap, gy - half_gap, int(gap_width) + 6, int(gap_width) + 6)

        is_exterior = wall.thickness_class == 'thick'

        return Door(
            position=(gx, gy), bounding_box=bbox,
            orientation=f'in_{wall.orientation}_wall',
            wall_id=wall.id, confidence=confidence,
            gap_width=gap_width, detection_method=method,
            is_exterior_door=is_exterior
        )

    def _deduplicate_doors(self, doors: List[Door], min_dist: int = 25) -> List[Door]:
        if len(doors) <= 1:
            return doors
        doors.sort(key=lambda d: d.confidence, reverse=True)
        keep = [True] * len(doors)
        for i in range(len(doors)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(doors)):
                if not keep[j]:
                    continue
                if math.hypot(doors[i].position[0] - doors[j].position[0],
                              doors[i].position[1] - doors[j].position[1]) < min_dist:
                    keep[j] = False
        return [d for d, k in zip(doors, keep) if k]

    # ────────────────────────────────────────────────────────────
    # STAGE 10: Annotation
    # ────────────────────────────────────────────────────────────

    def _annotate(self, img: np.ndarray, walls: List[WallSegment], doors: List[Door],
                  wall_mask: np.ndarray) -> np.ndarray:
        annotated = img.copy()
        wall_colored = np.zeros_like(annotated)
        wall_colored[wall_mask > 0] = (255, 180, 0)
        cv2.addWeighted(wall_colored, 0.35, annotated, 1.0, 0, annotated)

        for wall in walls:
            x, y, bw, bh = wall.bbox
            color = (255, 150, 0) if wall.orientation == 'horizontal' else (
                (0, 150, 255) if wall.orientation == 'vertical' else (150, 150, 0))
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(annotated, f"W{wall.id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        for i, door in enumerate(doors):
            x, y, dw, dh = door.bounding_box
            color = (0, 200, 0) if not door.is_exterior_door else (200, 0, 200)
            cv2.rectangle(annotated, (x, y), (x + dw, y + dh), color, 2)
            cv2.circle(annotated, door.position, 5, color, -1)
            cv2.putText(annotated, f"D{i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        self.dbg.save("final_annotated", annotated)
        return annotated

    # ════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ════════════════════════════════════════════════════════════

    def process(self, image_path: str, output_path: str = None) -> dict:
        print(f"{'=' * 60}")
        print(f"FLOORPLAN DETECTOR v4.2 — {image_path}")
        print(f"{'=' * 60}")

        print("\n[1] Loading and binarizing...")
        img, gray, binary = self._load_and_binarize(image_path)
        print(f"  Size: {img.shape[1]}×{img.shape[0]}")

        print("\n[2] Deskewing...")
        img, gray, binary, skew = self._deskew(img, gray, binary)

        print("\n[3] Detecting line segments...")
        segments = self._detect_segments(gray, binary)
        print(f"  Found {len(segments)} segments")

        print("\n[4] Aggregating segments...")
        aggregated = self._aggregate_with_gaps(segments, binary)
        print(f"  {len(aggregated)} segments, {sum(len(s.gap_positions) for s in aggregated)} gaps")

        print("\n[5] Estimating wall thickness (bimodal)...")
        self._estimate_thickness_bimodal(aggregated, binary)

        print("\n[6] Pairing walls...")
        walls = self._pair_walls_optimal(aggregated, binary)

        print("\n[7] Filtering wall network...")
        walls = self._filter_walls_network(walls, binary)

        print("\n[8] Generating masks...")
        wall_mask = self._generate_wall_masks(walls, binary)

        print("\n[9] Detecting doors...")
        doors = self._detect_doors(walls, binary)

        print("\n[10] Creating annotation...")
        annotated = self._annotate(img, walls, doors, wall_mask)

        if output_path is None:
            output_path = image_path.rsplit('.', 1)[0] + '_annotated.png'
        cv2.imwrite(output_path, annotated)
        print(f"\n  Saved: {output_path}")

        # ── Compile all debug images into one grid ──────────────
        print("\n[11] Compiling debug grid...")
        self.dbg.compile_grid(
            output_filename="debug_grid.png",
            thumb_w=400,
            thumb_h=300,
            cols=4,
        )

        print(f"\n{'=' * 60}")
        print(f"RESULTS: {len(walls)} walls, {len(doors)} doors")
        print(f"{'=' * 60}")

        return {'walls': walls, 'doors': doors, 'annotated_image': annotated, 'wall_mask': wall_mask}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Floorplan Detector v4.2')
    parser.add_argument('image', help='Path to floorplan image')
    parser.add_argument('--output', '-o', help='Output path')
    parser.add_argument('--debug-dir', '-d', default='debug_output')
    parser.add_argument('--no-debug', action='store_true')
    # Grid layout tweaks (optional CLI overrides)
    parser.add_argument('--grid-cols', type=int, default=4, help='Columns in debug grid (default 4)')
    parser.add_argument('--grid-thumb-w', type=int, default=400, help='Thumbnail width in grid (default 400)')
    parser.add_argument('--grid-thumb-h', type=int, default=300, help='Thumbnail height in grid (default 300)')
    args = parser.parse_args()

    detector = FloorplanDetector(debug_dir=args.debug_dir, debug=not args.no_debug)

    # Patch grid params from CLI before running
    _orig_process = detector.process
    def _patched_process(image_path, output_path=None):
        result = _orig_process(image_path, output_path)
        # compile_grid already called inside process(); re-run with CLI params if different from defaults
        if args.grid_cols != 4 or args.grid_thumb_w != 400 or args.grid_thumb_h != 300:
            detector.dbg.compile_grid(
                output_filename="debug_grid.png",
                thumb_w=args.grid_thumb_w,
                thumb_h=args.grid_thumb_h,
                cols=args.grid_cols,
            )
        return result
    detector.process = _patched_process

    detector.process(args.image, args.output)


if __name__ == '__main__':
    main()
