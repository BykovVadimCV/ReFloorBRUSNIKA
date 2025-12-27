"""
Быков Вадим Олегович - 19.12.2025

Модуль генерации floorplan JSON - v1.2

ПАЙПЛАЙН:
- Построение контура квартиры из apartment_boundary
- Коррекция контура с учётом дверных bbox (outer-door fix, 2% rule)
- Генерация стен по контуру
- Вырезание участков стен под дверные проёмы (YOLO)
- Добавление внутренних стен с фильтрацией дубликатов
- Подключение внутренних стен к контуру
- Объединение близких внутренних стен
- Кластеризация вершин в общие corners
- Генерация JSON в формате Blueprint3D

СТЕК:
- OpenCV: обработка контуров
- NumPy: векторные операции
- UUID: генерация идентификаторов

Последнее изменение - 26.12.25 - Door bbox 2% rule in outline
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from space import RoomMeasurements
from detection.furnituredetector import YOLOResults


# ------------------------------------------------------------------ #
# BASIC GEOMETRY
# ------------------------------------------------------------------ #

@dataclass
class Pt:
    """Точка в 2D пространстве"""
    x: float
    y: float


@dataclass
class Segment:
    """Отрезок между двумя точками"""
    p1: Pt
    p2: Pt


@dataclass
class OutlineEdge:
    """Ребро контура с направлением"""
    p1: Pt
    p2: Pt
    dirx: float
    diry: float
    length: float


def _segment_dir(p1: Pt, p2: Pt) -> Tuple[float, float, float]:
    """
    Вычисление направления и длины отрезка

    Возвращает:
        Кортеж (dx, dy, length)
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    L = float(np.hypot(dx, dy))
    if L == 0:
        return 0.0, 0.0, 0.0
    return dx / L, dy / L, L


def _project_point_to_edge(p: Pt, edge: OutlineEdge) -> Tuple[Pt, float, float]:
    """
    Проекция точки на отрезок

    Параметры:
        p: точка для проекции
        edge: ребро для проекции

    Возвращает:
        Кортеж (projected_point, s_clamped, distance)
    """
    vx = p.x - edge.p1.x
    vy = p.y - edge.p1.y
    s = vx * edge.dirx + vy * edge.diry
    s_clamped = max(0.0, min(edge.length, s))
    px = edge.p1.x + edge.dirx * s_clamped
    py = edge.p1.y + edge.diry * s_clamped
    dist = float(np.hypot(p.x - px, p.y - py))
    return Pt(px, py), s_clamped, dist


def _angle_between(dx1: float, dy1: float, dx2: float, dy2: float) -> float:
    """
    Угол между двумя векторами в радианах

    Возвращает:
        Угол в диапазоне [0, π]
    """
    dot = dx1 * dx2 + dy1 * dy2
    n1 = float(np.hypot(dx1, dy1))
    n2 = float(np.hypot(dx2, dy2))
    if n1 == 0 or n2 == 0:
        return 0.0
    c = dot / (n1 * n2)
    c = max(-1.0, min(1.0, c))
    return float(np.arccos(c))


# ------------------------------------------------------------------ #
# CORNER CLUSTERING
# ------------------------------------------------------------------ #

@dataclass
class _Node:
    """Узел для кластеризации вершин"""
    x: float
    y: float
    count: int = 1


class _CornerCluster:
    """Кластеризация точек в вершины (corners)"""

    def __init__(self, elevation_m: float, merge_tol_m: float):
        """
        Инициализация кластеризатора

        Параметры:
            elevation_m: высота этажа в метрах
            merge_tol_m: толерантность слияния в метрах
        """
        self.elevation_m = float(elevation_m)
        self.merge_tol_m = float(merge_tol_m)
        self.nodes: List[_Node] = []
        self.node_to_corner_id: List[str] = []
        self.corners: Dict[str, Dict[str, float]] = {}

    def add_point(self, x_m: float, y_m: float) -> int:
        """
        Добавление точки в кластер

        Параметры:
            x_m, y_m: координаты в метрах

        Возвращает:
            Индекс узла
        """
        tol2 = self.merge_tol_m * self.merge_tol_m
        for i, n in enumerate(self.nodes):
            dx = x_m - n.x
            dy = y_m - n.y
            if dx * dx + dy * dy <= tol2:
                new_count = n.count + 1
                n.x = (n.x * n.count + x_m) / new_count
                n.y = (n.y * n.count + y_m) / new_count
                n.count = new_count
                return i
        self.nodes.append(_Node(x=x_m, y=y_m, count=1))
        return len(self.nodes) - 1

    def finalize(self) -> None:
        """Финализация кластеров с генерацией UUID"""
        import uuid
        self.node_to_corner_id = []
        self.corners = {}
        for n in self.nodes:
            cid = str(uuid.uuid4())
            self.node_to_corner_id.append(cid)
            self.corners[cid] = {
                "x": n.x * 100.0,  # м -> см
                "y": n.y * 100.0,
                "elevation": self.elevation_m,
            }

    def corner_id_for_node(self, idx: int) -> str:
        """Получение corner_id для индекса узла"""
        return self.node_to_corner_id[idx]


# ------------------------------------------------------------------ #
# FLOORPLAN BUILDER
# ------------------------------------------------------------------ #

class FloorplanBuilder:
    """Построитель floorplan JSON из measurements и YOLO результатов"""

    def __init__(
        self,
        wall_height_m: float = 3.0,
        node_merge_tol_pixels: float = 1.5,
        min_wall_length_m: float = 0.25,
        min_door_width_m: float = 0.6,
        parallel_dist_factor: float = 1.5,
        parallel_angle_deg: float = 15.0,
        perp_dist_factor: float = 1.5,
        perp_angle_deg: float = 30.0,
        base_wall_thickness_m: float = 0.2,
        merge_inner_angle_deg: float = 10.0,
        merge_inner_dist_factor: float = 0.8,
    ):
        """
        Инициализация построителя floorplan

        Параметры:
            wall_height_m: высота стен в метрах
            node_merge_tol_pixels: толерантность слияния вершин в пикселях
            min_wall_length_m: минимальная длина стены в метрах
            min_door_width_m: минимальная ширина двери в метрах
            parallel_dist_factor: фактор расстояния для параллельных стен
            parallel_angle_deg: порог угла для параллельности (градусы)
            perp_dist_factor: фактор расстояния для перпендикулярных стен
            perp_angle_deg: порог угла для перпендикулярности (градусы)
            base_wall_thickness_m: базовая толщина стены в метрах
            merge_inner_angle_deg: порог угла для слияния внутренних стен (градусы)
            merge_inner_dist_factor: фактор расстояния для слияния внутренних стен
        """
        self.wall_height_m = float(wall_height_m)
        self.node_merge_tol_pixels = float(node_merge_tol_pixels)
        self.min_wall_length_m = float(min_wall_length_m)
        self.min_door_width_m = float(min_door_width_m)
        self.parallel_dist_factor = float(parallel_dist_factor)
        self.parallel_angle_rad = np.deg2rad(parallel_angle_deg)
        self.perp_dist_factor = float(perp_dist_factor)
        self.perp_angle_rad = np.deg2rad(perp_angle_deg)
        self.base_wall_thickness_m = float(base_wall_thickness_m)
        self.merge_inner_angle_rad = np.deg2rad(merge_inner_angle_deg)
        self.merge_inner_dist_factor = float(merge_inner_dist_factor)

    def build_floorplan(
        self,
        measurements: RoomMeasurements,
        yolo_results: Optional[YOLOResults] = None,
        room_name: str = "A New Room",
    ) -> Dict[str, Any]:
        """
        Построение floorplan JSON

        Параметры:
            measurements: результаты измерений помещения
            yolo_results: результаты YOLO детекции (опционально)
            room_name: название комнаты

        Возвращает:
            Словарь с floorplan в формате Blueprint3D
        """
        if measurements is None:
            raise ValueError("FloorplanBuilder: 'measurements' is None")
        if not getattr(measurements, "walls", None):
            raise ValueError("FloorplanBuilder: 'measurements.walls' is empty")
        if not getattr(measurements, "pixel_scale", None):
            raise ValueError("FloorplanBuilder: 'measurements.pixel_scale' is missing")
        if measurements.pixel_scale <= 0:
            raise ValueError("FloorplanBuilder: pixel_scale must be > 0")

        walls_data = measurements.walls
        pixel_scale = float(measurements.pixel_scale)

        # Преобразование пикселей в метры
        xs_px, ys_px = [], []
        for w in walls_data:
            xs_px.extend([float(w["x1"]), float(w["x2"])])
            ys_px.extend([float(w["y1"]), float(w["y2"])])
        cx_px = (min(xs_px) + max(xs_px)) / 2.0
        cy_px = (min(ys_px) + max(ys_px)) / 2.0

        def px_to_m(x_px: float, y_px: float) -> Pt:
            return Pt(
                x=(x_px - cx_px) * pixel_scale,
                y=(cy_px - y_px) * pixel_scale,
            )

        # --- NEW: скорректированная маска контура с учётом дверей (2% правило) ---
        boundary_mask_override = self._refine_apartment_boundary_with_doors(
            measurements=measurements,
            yolo_results=yolo_results,
            non_interior_threshold=0.02,  # >2% не-интерьерных пикселей → bbox вырезаем
        )
        # --------------------------------------------------------------------------

        # Построение полигона контура
        poly_pts_m = self._build_outline_polygon(
            measurements,
            px_to_m,
            boundary_mask_override=boundary_mask_override,
        )

        # Построение outline edges
        outline_edges: List[OutlineEdge] = []
        for i in range(len(poly_pts_m)):
            p1 = poly_pts_m[i]
            p2 = poly_pts_m[(i + 1) % len(poly_pts_m)]
            dx, dy, L = _segment_dir(p1, p2)
            if L > 0:
                outline_edges.append(OutlineEdge(p1, p2, dx, dy, L))

        # Вырезание дверных проёмов из outline
        door_segments_m: List[Segment] = []
        if yolo_results:
            for det in yolo_results.doors:
                x1, y1, x2, y2 = det.bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                half_w = abs(x2 - x1) / 2.0
                half_h = abs(y2 - y1) / 2.0
                c_m = px_to_m(cx, cy)
                half_w_m = half_w * pixel_scale
                half_h_m = half_h * pixel_scale
                box_half = max(half_w_m, half_h_m)
                if box_half >= self.min_door_width_m / 2.0:
                    door_segments_m.append(Segment(
                        Pt(c_m.x - box_half, c_m.y - box_half),
                        Pt(c_m.x + box_half, c_m.y + box_half)
                    ))

        outline_segments = self._cut_outline_edges_by_doors(outline_edges, door_segments_m)

        # Добавление внутренних стен
        inner_walls_m: List[Segment] = []
        for w in walls_data:
            if w.get("is_outer", False):
                continue
            p1 = px_to_m(float(w["x1"]), float(w["y1"]))
            p2 = px_to_m(float(w["x2"]), float(w["y2"]))
            L = float(np.hypot(p2.x - p1.x, p2.y - p1.y))
            if L >= self.min_wall_length_m:
                inner_walls_m.append(Segment(p1, p2))

        # Фильтрация и обработка внутренних стен
        filtered = self._remove_inner_walls_duplicating_outline(
            inner_walls_m, outline_edges
        )
        snapped = self._snap_perpendicular_inner_to_outline(
            filtered, outline_edges
        )
        merged = self._merge_inner_segments(
            snapped,
            self.base_wall_thickness_m * self.merge_inner_dist_factor,
            self.merge_inner_angle_rad
        )

        # Объединение всех сегментов
        all_segments = outline_segments + merged

        # Кластеризация вершин
        cluster = _CornerCluster(
            elevation_m=3.0,
            merge_tol_m=self.node_merge_tol_pixels * pixel_scale
        )

        edges_nodes: List[Tuple[int, int]] = []
        for seg in all_segments:
            n1 = cluster.add_point(seg.p1.x, seg.p1.y)
            n2 = cluster.add_point(seg.p2.x, seg.p2.y)
            edges_nodes.append((n1, n2))

        # Узлы для room polygon
        room_node_indices: List[int] = []
        last_idx: Optional[int] = None
        for p in poly_pts_m:
            ni = cluster.add_point(p.x, p.y)
            if ni != last_idx:
                room_node_indices.append(ni)
                last_idx = ni
        if len(room_node_indices) >= 3 and room_node_indices[0] == room_node_indices[-1]:
            room_node_indices = room_node_indices[:-1]

        cluster.finalize()
        corners_out = cluster.corners

        # Генерация walls
        walls_out: List[Dict[str, Any]] = []
        used_pairs = set()
        for n1, n2 in edges_nodes:
            if n1 == n2:
                continue
            c1 = cluster.corner_id_for_node(n1)
            c2 = cluster.corner_id_for_node(n2)
            key = tuple(sorted((c1, c2)))
            if key in used_pairs:
                continue
            used_pairs.add(key)
            c1d = corners_out[c1]
            c2d = corners_out[c2]
            walls_out.append({
                "corner1": c1,
                "corner2": c2,
                "frontTexture": {
                    "url": "rooms/textures/wallmap.png",
                    "stretch": True,
                    "scale": 0,
                },
                "backTexture": {
                    "url": "rooms/textures/wallmap.png",
                    "stretch": True,
                    "scale": 0,
                },
                "wallType": "STRAIGHT",
                "a": {"x": c1d["x"], "y": c1d["y"]},
                "b": {"x": c2d["x"], "y": c2d["y"]},
            })

        # Генерация rooms
        rooms: Dict[str, Dict[str, str]] = {}
        if len(room_node_indices) >= 3:
            room_corner_ids: List[str] = []
            last_cid: Optional[str] = None
            for ni in room_node_indices:
                cid = cluster.corner_id_for_node(ni)
                if cid != last_cid:
                    room_corner_ids.append(cid)
                    last_cid = cid
            if len(room_corner_ids) >= 3:
                room_key = ",".join(room_corner_ids)
                rooms[room_key] = {"name": room_name}

        # Финальный JSON
        floorplan_dict: Dict[str, Any] = {
            "floorplan": {
                "version": "0.0.2a",
                "corners": corners_out,
                "walls": walls_out,
                "rooms": rooms,
                "wallTextures": [],
                "floorTextures": {},
                "newFloorTextures": {},
                "carbonSheet": {
                    "url": "",
                    "transparency": 1,
                    "x": 0,
                    "y": 0,
                    "anchorX": 0,
                    "anchorY": 0,
                    "width": 0.01,
                    "height": 0.01,
                },
            },
            "items": [],
        }
        return floorplan_dict

    # ------------------------------------------------------------------ #
    # CONTOUR REFINEMENT WITH DOORS (2% RULE)
    # ------------------------------------------------------------------ #

    def _refine_apartment_boundary_with_doors(
        self,
        measurements: RoomMeasurements,
        yolo_results: Optional[YOLOResults],
        non_interior_threshold: float = 0.02,  # 2 %
    ) -> Optional[np.ndarray]:
        """
        Строит скорректированную маску apartment_boundary для построения контура.

        Правило: для каждого bbox двери считаем долю пикселей, которые НЕ
        принадлежат интерьеру (interior = apartment_boundary > 0).
        Если доля > non_interior_threshold (по умолчанию 2 %), весь bbox
        двери считается не-интерьером и вырезается из маски.

        Возвращает:
            - новую маску (2D ndarray), если были изменения
            - None, если изменений нет (использовать исходную mask)
        """
        if yolo_results is None or not getattr(yolo_results, "doors", None):
            return None

        mask = getattr(measurements, "apartment_boundary", None)
        if mask is None:
            return None

        # Приводим к одноканальной маске
        if mask.ndim == 3:
            ab_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            ab_gray = mask

        interior = ab_gray > 0  # "интерьер" = всё, что внутри apartment_boundary
        h, w = interior.shape[:2]

        remove_mask = np.zeros_like(interior, dtype=bool)

        for det in yolo_results.doors:
            x1, y1, x2, y2 = det.bbox

            # Нормализуем и переводим в int
            x1, x2 = sorted((int(np.floor(x1)), int(np.ceil(x2))))
            y1, y2 = sorted((int(np.floor(y1)), int(np.ceil(y2))))

            # Ограничиваем рамками изображения
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w,     x2))   # правая граница как end-index
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h,     y2))   # нижняя граница как end-index

            if x2 <= x1 or y2 <= y1:
                continue

            sub = interior[y1:y2, x1:x2]
            total_px = sub.size
            if total_px == 0:
                continue

            interior_px = int(sub.sum())
            non_interior_px = total_px - interior_px
            frac_non_interior = non_interior_px / float(total_px)

            # Если >2% пикселей bbox НЕ в interior, то весь bbox не-интерьер
            if frac_non_interior > non_interior_threshold:
                remove_mask[y1:y2, x1:x2] = True

        if not remove_mask.any():
            return None

        new_ab = ab_gray.copy()
        new_ab[remove_mask] = 0
        return new_ab

    # ------------------------------------------------------------------ #
    # OUTLINE POLYGON
    # ------------------------------------------------------------------ #

    def _build_outline_polygon(
        self,
        measurements: RoomMeasurements,
        px_to_m,
        boundary_mask_override: Optional[np.ndarray] = None,
    ) -> List[Pt]:
        """
        Построение полигона контура из apartment_boundary.

        Если boundary_mask_override не None, используется он
        (например, после коррекции маски дверями).
        """
        # Выбор маски
        mask = boundary_mask_override
        if mask is None:
            mask = getattr(measurements, "apartment_boundary", None)

        if mask is not None:
            try:
                if mask.ndim == 3:
                    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                else:
                    mask_gray = mask.copy()
                _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(
                    mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    peri = cv2.arcLength(cnt, True)
                    epsilon = 0.01 * peri
                    approx = cv2.approxPolyDP(cnt, epsilon, True)

                    pts: List[Pt] = []
                    last: Optional[Tuple[float, float]] = None
                    for pt in approx:
                        x_px = float(pt[0][0])
                        y_px = float(pt[0][1])
                        p = px_to_m(x_px, y_px)
                        if last is None or abs(p.x - last[0]) > 1e-6 or abs(p.y - last[1]) > 1e-6:
                            pts.append(p)
                            last = (p.x, p.y)

                    if len(pts) >= 3:
                        if abs(pts[0].x - pts[-1].x) < 1e-6 and abs(pts[0].y - pts[-1].y) < 1e-6:
                            pts = pts[:-1]
                        return pts
            except Exception:
                pass

        # Fallback: прямоугольник по bounding box стен
        walls_data = measurements.walls
        xs_px = [float(w["x1"]) for w in walls_data] + [float(w["x2"]) for w in walls_data]
        ys_px = [float(w["y1"]) for w in walls_data] + [float(w["y2"]) for w in walls_data]
        min_xp, max_xp = min(xs_px), max(xs_px)
        min_yp, max_yp = min(ys_px), max(ys_px)

        tl = px_to_m(min_xp, max_yp)
        bl = px_to_m(min_xp, min_yp)
        br = px_to_m(max_xp, min_yp)
        tr = px_to_m(max_xp, max_yp)
        return [bl, tl, tr, br]

    # ------------------------------------------------------------------ #
    # OUTLINE CUTTING / INNER WALL PROCESSING
    # ------------------------------------------------------------------ #

    def _cut_outline_edges_by_doors(
        self,
        edges: List[OutlineEdge],
        door_segs: List[Segment]
    ) -> List[Segment]:
        """
        Вырезание дверных проёмов из outline edges

        Возвращает:
            Список сегментов после вырезания
        """
        result: List[Segment] = []
        for edge in edges:
            cuts: List[Tuple[float, float]] = [(0.0, edge.length)]
            for ds in door_segs:
                for door_pt in [ds.p1, ds.p2]:
                    _, s, dist = _project_point_to_edge(door_pt, edge)
                    if dist < self.base_wall_thickness_m:
                        cuts.append((max(0.0, s - self.min_door_width_m / 2.0),
                                     min(edge.length, s + self.min_door_width_m / 2.0)))

            cuts.sort()
            merged_cuts: List[Tuple[float, float]] = []
            for c in cuts:
                if not merged_cuts or c[0] > merged_cuts[-1][1]:
                    merged_cuts.append(c)
                else:
                    merged_cuts[-1] = (merged_cuts[-1][0], max(merged_cuts[-1][1], c[1]))

            for i in range(len(merged_cuts)):
                s0, s1 = merged_cuts[i]
                if s1 - s0 >= self.min_wall_length_m:
                    px1 = Pt(edge.p1.x + edge.dirx * s0, edge.p1.y + edge.diry * s0)
                    px2 = Pt(edge.p1.x + edge.dirx * s1, edge.p1.y + edge.diry * s1)
                    result.append(Segment(px1, px2))
        return result

    def _remove_inner_walls_duplicating_outline(
        self,
        inner_segs: List[Segment],
        outline_edges: List[OutlineEdge]
    ) -> List[Segment]:
        """
        Удаление внутренних стен, дублирующих outline

        Возвращает:
            Отфильтрованный список внутренних стен
        """
        filtered: List[Segment] = []
        thr = self.base_wall_thickness_m * self.parallel_dist_factor

        for seg in inner_segs:
            dx, dy, L = _segment_dir(seg.p1, seg.p2)
            if L < self.min_wall_length_m:
                continue

            is_dup = False
            for oe in outline_edges:
                angle = _angle_between(dx, dy, oe.dirx, oe.diry)
                if angle > self.parallel_angle_rad and angle < (np.pi - self.parallel_angle_rad):
                    continue

                _, _, d1 = _project_point_to_edge(seg.p1, oe)
                _, _, d2 = _project_point_to_edge(seg.p2, oe)
                if d1 < thr and d2 < thr:
                    is_dup = True
                    break

            if not is_dup:
                filtered.append(seg)

        return filtered

    def _snap_perpendicular_inner_to_outline(
        self,
        inner_segs: List[Segment],
        outline_edges: List[OutlineEdge]
    ) -> List[Segment]:
        """
        Привязка перпендикулярных внутренних стен к outline

        Возвращает:
            Список стен с привязкой
        """
        result: List[Segment] = []
        thr = self.base_wall_thickness_m * self.perp_dist_factor

        for seg in inner_segs:
            dx, dy, L = _segment_dir(seg.p1, seg.p2)
            if L < self.min_wall_length_m:
                continue

            best_oe_p1, best_d_p1 = None, 999999.0
            best_oe_p2, best_d_p2 = None, 999999.0

            for oe in outline_edges:
                angle = _angle_between(dx, dy, oe.dirx, oe.diry)
                perp = abs(angle - np.pi / 2.0) < self.perp_angle_rad

                if perp:
                    proj1, _, d1 = _project_point_to_edge(seg.p1, oe)
                    if d1 < thr and d1 < best_d_p1:
                        best_d_p1 = d1
                        best_oe_p1 = proj1

                    proj2, _, d2 = _project_point_to_edge(seg.p2, oe)
                    if d2 < thr and d2 < best_d_p2:
                        best_d_p2 = d2
                        best_oe_p2 = proj2

            new_p1 = best_oe_p1 if best_oe_p1 else seg.p1
            new_p2 = best_oe_p2 if best_oe_p2 else seg.p2
            new_L = float(np.hypot(new_p2.x - new_p1.x, new_p2.y - new_p1.y))

            if new_L >= self.min_wall_length_m:
                result.append(Segment(new_p1, new_p2))

        return result

    def _merge_inner_segments(
        self,
        segments: List[Segment],
        dist_thr: float,
        angle_thr: float,
    ) -> List[Segment]:
        """
        Объединение близких параллельных внутренних стен

        Возвращает:
            Список объединённых сегментов
        """
        if len(segments) < 2:
            return segments

        def can_merge(a: Segment, b: Segment) -> Tuple[bool, Segment]:
            vx = a.p2.x - a.p1.x
            vy = a.p2.y - a.p1.y
            L = float(np.hypot(vx, vy))
            if L == 0:
                return False, a
            ux, uy = vx / L, vy / L

            wx = b.p2.x - b.p1.x
            wy = b.p2.y - b.p1.y
            M = float(np.hypot(wx, wy))
            if M == 0:
                return False, a

            ang = _angle_between(vx, vy, wx, wy)
            if ang > angle_thr:
                return False, a

            # Проверка расстояния концов b до прямой a
            for p in (b.p1, b.p2):
                rx = p.x - a.p1.x
                ry = p.y - a.p1.y
                dist = abs(rx * (-uy) + ry * ux)
                if dist > dist_thr:
                    return False, a

            # Проекция всех концов на прямую a
            s_vals = []
            for p in (a.p1, a.p2, b.p1, b.p2):
                rx = p.x - a.p1.x
                ry = p.y - a.p1.y
                s_vals.append(rx * ux + ry * uy)
            s_min = min(s_vals)
            s_max = max(s_vals)
            if (s_max - s_min) < 1e-3:
                return False, a

            new_p1 = Pt(a.p1.x + ux * s_min, a.p1.y + uy * s_min)
            new_p2 = Pt(a.p1.x + ux * s_max, a.p1.y + uy * s_max)
            return True, Segment(new_p1, new_p2)

        segs = segments[:]
        changed = True
        while changed and len(segs) > 1:
            changed = False
            used = [False] * len(segs)
            new_segs: List[Segment] = []
            for i in range(len(segs)):
                if used[i]:
                    continue
                cur = segs[i]
                merged_any = False
                for j in range(i + 1, len(segs)):
                    if used[j]:
                        continue
                    ok, merged_seg = can_merge(cur, segs[j])
                    if ok:
                        used[j] = True
                        cur = merged_seg
                        merged_any = True
                        changed = True
                used[i] = True
                if merged_any:
                    if np.hypot(cur.p2.x - cur.p1.x, cur.p2.y - cur.p1.y) >= self.min_wall_length_m:
                        new_segs.append(cur)
                else:
                    new_segs.append(cur)
            segs = new_segs
        return segs