"""
Быков Вадим Олегович - 15.12.25

Генератор синтетических датасетов для распознавания дверей и окон на планах формата "Брусника" - v1.2

УЛУЧШЕНИЯ В v1.2:
- Реалистичное разнообразие окон (многосекционные, балконные двери, разные стили рам)
- Окна с 1-4 секциями/створками
- Различные типы окон: стандартные, балконные двери, широкие, узкие
- Улучшенная визуализация рам и перемычек

СПЕЦИФИКАЦИЯ ФОРМАТА "БРУСНИКА":
- Стены: цвет #8E8780 (RGB: 142, 135, 128), толщина 12-30 пикселей
- Мебель: цвет #E2DFDD (RGB: 226, 223, 221), различные формы (прямоугольники, овалы, круги)
- Окна: две параллельные линии цвета #E2DFDD с белым промежутком между ними
- Двери: прямоугольник цвета #E2DFDD + дуга открывания
- Фон: белый #FFFFFF

ПАЙПЛАЙН:
1. Создание планировки помещения (комнаты, коридор);
2. Рисование стен цветом #8E8780;
3. Размещение мебели разных форм цветом #E2DFDD;
4. Добавление окон на внешних стенах;
5. Добавление дверей с дугами на внутренних стенах;
6. Базовые аугментации (поворот, blur, шум);
7. Экспорт в формате YOLO.

Классы:
  0 - door
  1 - window

Последнее изменение - 18.12.25 - Улучшено разнообразие окон
"""

import os
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np


WALL_COLOR = (128, 135, 142)
FURNITURE_COLOR = (221, 223, 226)
WINDOW_COLOR = WALL_COLOR
BG_COLOR = (255, 255, 255)
LINE_TYPE = cv2.LINE_AA


@dataclass
class Wall:
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: int
    is_external: bool


@dataclass
class Room:
    x1: int
    y1: int
    x2: int
    y2: int
    room_type: str


@dataclass
class OccupiedRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    region_type: str


class BrusnikaFloorPlanGenerator:
    def __init__(self, img_size: int = 1024, seed: int = 0, super_sampling: int = 2):
        self.out_size = img_size
        self.super_sampling = super_sampling
        self.size = img_size * super_sampling
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)
        self.window_length_ratios = (0.035, 0.05, 0.07, 0.09)
        self.door_length_ratios = (0.04, 0.05, 0.06)
        self.bg_color = BG_COLOR
        self.fake_door_chair_prob = 0.7
        self.fake_door_chair_count = (1, 3)
        self.occupied_regions: List[OccupiedRegion] = []

    def generate_dataset(self, n_images: int, out_dir: str, wall_thickness_range: Tuple[int, int] = (12, 30)):
        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for i in range(n_images):
            base_ext_thickness = self.rng.randint(wall_thickness_range[0], wall_thickness_range[1])
            ext_wall_thickness = base_ext_thickness * self.super_sampling
            int_wall_thickness = int(ext_wall_thickness * self.rng.uniform(0.5, 0.7))

            hi_res_img, labels = self.generate_single_plan(ext_wall_thickness, int_wall_thickness)

            img = cv2.resize(
                hi_res_img,
                (self.out_size, self.out_size),
                interpolation=cv2.INTER_AREA,
            )

            if self.rng.random() < 0.15:
                img, labels = self._apply_rotation(img, labels)

            if self.rng.random() < 0.3:
                img = self._apply_augmentations(img)

            fname = f"{i:05d}"
            cv2.imwrite(os.path.join(img_dir, fname + ".png"), img)

            with open(os.path.join(lbl_dir, fname + ".txt"), "w", encoding="utf-8") as f:
                for cls_id, cx, cy, w, h in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def generate_single_plan(
        self,
        ext_wall_thickness: int,
        int_wall_thickness: int,
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        self.occupied_regions = []

        base_bg = self.rng.randint(248, 255)
        self.bg_color = (base_bg, base_bg, base_bg)
        img = np.full((self.size, self.size, 3), self.bg_color, dtype=np.uint8)

        margin = int(self.size * self.rng.uniform(0.15, 0.25))
        outer_left = margin
        outer_top = margin
        outer_right = self.size - margin
        outer_bottom = self.size - margin

        walls, rooms = self._create_layout(
            outer_left, outer_top, outer_right, outer_bottom, ext_wall_thickness, int_wall_thickness
        )

        self._draw_walls(img, walls)

        self._add_furniture(img, rooms, walls)

        labels: List[Tuple[int, float, float, float, float]] = []

        external_walls = [w for w in walls if w.is_external]
        for w in external_walls:
            length = max(abs(w.x2 - w.x1), abs(w.y2 - w.y1))
            n_windows = self.rng.randint(1, max(2, int(length / 200)))
            for _ in range(n_windows):
                if self.rng.random() < 0.7:
                    box = self._place_window(img, w)
                    if box:
                        labels.append((1, *box))

        internal_walls = [w for w in walls if not w.is_external]
        door_spans: Dict[int, List[Tuple[int, int]]] = {}

        for w in internal_walls:
            length = max(abs(w.x2 - w.x1), abs(w.y2 - w.y1))
            if length < int_wall_thickness * 3:
                continue
            n_doors = 1 if length < 250 * self.super_sampling else self.rng.randint(1, 2)
            spans_for_wall = door_spans.setdefault(id(w), [])
            for _ in range(n_doors):
                if self.rng.random() < 0.8:
                    box = self._place_door(img, w, spans_for_wall)
                    if box:
                        labels.append((0, *box))

        if not any(lbl[0] == 1 for lbl in labels) and external_walls:
            w = self.rng.choice(external_walls)
            box = self._place_window(img, w)
            if box:
                labels.append((1, *box))

        if not any(lbl[0] == 0 for lbl in labels) and internal_walls:
            suitable_walls = [
                w for w in internal_walls
                if max(abs(w.x2 - w.x1), abs(w.y2 - w.y1)) >= int_wall_thickness * 3
            ]
            if suitable_walls:
                w = self.rng.choice(suitable_walls)
                spans_for_wall = door_spans.setdefault(id(w), [])
                box = self._place_door(img, w, spans_for_wall)
                if box:
                    labels.append((0, *box))

        self._add_annotations(img, rooms)

        return img, labels

    def _create_layout(
        self,
        left: int,
        top: int,
        right: int,
        bottom: int,
        ext_thickness: int,
        int_thickness: int,
    ) -> Tuple[List[Wall], List[Room]]:
        walls: List[Wall] = []
        rooms: List[Room] = []

        walls.append(Wall(left, top, right, top, ext_thickness, True))
        walls.append(Wall(left, bottom, right, bottom, ext_thickness, True))
        walls.append(Wall(left, top, left, bottom, ext_thickness, True))
        walls.append(Wall(right, top, right, bottom, ext_thickness, True))

        width = right - left
        height = bottom - top

        layout_type = self.rng.choice(['horizontal', 'vertical'])

        if layout_type == 'horizontal':
            corridor_width = self.rng.randint(int(width * 0.15), int(width * 0.22))
            corridor_x = left + corridor_width
            walls.append(Wall(corridor_x, top, corridor_x, bottom, int_thickness, False))

            bathroom_height = self.rng.randint(int(height * 0.18), int(height * 0.28))
            bathroom_y = top + bathroom_height
            walls.append(Wall(left, bathroom_y, corridor_x, bathroom_y, int_thickness, False))

            kitchen_height = self.rng.randint(int(height * 0.30), int(height * 0.40))
            kitchen_y = bottom - kitchen_height
            walls.append(Wall(corridor_x, kitchen_y, right, kitchen_y, int_thickness, False))

            kitchen_width = self.rng.randint(int(width * 0.25), int(width * 0.35))
            kitchen_x = corridor_x + kitchen_width
            walls.append(Wall(kitchen_x, kitchen_y, kitchen_x, bottom, int_thickness, False))

            if self.rng.random() < 0.7:
                div_y = top + self.rng.randint(int(height * 0.35), int(height * 0.60))
                walls.append(Wall(corridor_x, div_y, kitchen_x, div_y, int_thickness, False))

            rooms.append(Room(left, top, corridor_x, bathroom_y, "bathroom"))
            rooms.append(Room(left, bathroom_y, corridor_x, bottom, "corridor"))
            rooms.append(Room(corridor_x, kitchen_y, kitchen_x, bottom, "kitchen"))
            rooms.append(Room(corridor_x, top, right, kitchen_y, "living"))

        else:
            corridor_height = self.rng.randint(int(height * 0.15), int(height * 0.22))
            corridor_y = top + corridor_height
            walls.append(Wall(left, corridor_y, right, corridor_y, int_thickness, False))

            bathroom_width = self.rng.randint(int(width * 0.18), int(width * 0.28))
            bathroom_x = left + bathroom_width
            walls.append(Wall(bathroom_x, top, bathroom_x, corridor_y, int_thickness, False))

            kitchen_width = self.rng.randint(int(width * 0.30), int(width * 0.40))
            kitchen_x = left + kitchen_width
            walls.append(Wall(kitchen_x, corridor_y, kitchen_x, bottom, int_thickness, False))

            if self.rng.random() < 0.7:
                div_x = left + self.rng.randint(int(width * 0.45), int(width * 0.70))
                walls.append(Wall(div_x, corridor_y, div_x, bottom, int_thickness, False))

            rooms.append(Room(left, top, bathroom_x, corridor_y, "bathroom"))
            rooms.append(Room(bathroom_x, top, right, corridor_y, "corridor"))
            rooms.append(Room(left, corridor_y, kitchen_x, bottom, "kitchen"))
            rooms.append(Room(kitchen_x, corridor_y, right, bottom, "living"))

        return walls, rooms

    def _draw_walls(self, img: np.ndarray, walls: List[Wall]):
        for w in walls:
            half = w.thickness // 2
            x1, x2 = min(w.x1, w.x2), max(w.x1, w.x2)
            y1, y2 = min(w.y1, w.y2), max(w.y1, w.y2)

            if w.x1 == w.x2:
                pt1 = (x1 - half, y1)
                pt2 = (x1 + half, y2)
            else:
                pt1 = (x1, y1 - half)
                pt2 = (x2, y1 + half)

            cv2.rectangle(img, pt1, pt2, WALL_COLOR, -1, lineType=LINE_TYPE)

    def _check_collision(self, x1: int, y1: int, x2: int, y2: int, margin: int = 5) -> bool:
        for region in self.occupied_regions:
            if not (x2 + margin < region.x1 or x1 - margin > region.x2 or
                    y2 + margin < region.y1 or y1 - margin > region.y2):
                return True
        return False

    def _add_occupied_region(self, x1: int, y1: int, x2: int, y2: int, region_type: str):
        self.occupied_regions.append(OccupiedRegion(
            min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), region_type
        ))

    def _add_bathroom_furniture(self, img: np.ndarray, room: Room):
        rw = room.x2 - room.x1
        rh = room.y2 - room.y1
        if rw < 80 or rh < 80:
            return

        if rw >= rh:
            tub_len = int(rw * self.rng.uniform(0.45, 0.75))
            tub_depth = int(rh * self.rng.uniform(0.35, 0.5))
            x1 = self.rng.randint(room.x1 + 8, room.x2 - 8 - tub_len)
            y1 = room.y1 + 5
            x2 = x1 + tub_len
            y2 = y1 + tub_depth
        else:
            tub_len = int(rh * self.rng.uniform(0.45, 0.75))
            tub_depth = int(rw * self.rng.uniform(0.35, 0.5))
            x1 = room.x1 + 5
            y1 = self.rng.randint(room.y1 + 8, room.y2 - 8 - tub_len)
            x2 = x1 + tub_depth
            y2 = y1 + tub_len

        cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, lineType=LINE_TYPE)
        cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, lineType=LINE_TYPE)
        self._add_occupied_region(x1, y1, x2, y2, "furniture")

        inner_pad = max(3, int(min(tub_len, tub_depth) * 0.08))
        cv2.rectangle(
            img,
            (x1 + inner_pad, y1 + inner_pad),
            (x2 - inner_pad, y2 - inner_pad),
            self.bg_color,
            -1,
            lineType=LINE_TYPE,
        )
        cv2.rectangle(
            img,
            (x1 + inner_pad, y1 + inner_pad),
            (x2 - inner_pad, y2 - inner_pad),
            WALL_COLOR,
            1,
            lineType=LINE_TYPE,
        )

        base = min(rw, rh)
        sink_w = int(base * 0.18)
        sink_h = int(base * 0.14)
        sx1 = room.x2 - sink_w - 8
        sy1 = room.y2 - sink_h - 8
        sx2 = sx1 + sink_w
        sy2 = sy1 + sink_h
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), FURNITURE_COLOR, -1, lineType=LINE_TYPE)
        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), WALL_COLOR, 1, lineType=LINE_TYPE)
        self._add_occupied_region(sx1, sy1, sx2, sy2, "furniture")

        cv2.circle(
            img,
            (sx1 + sink_w // 2, sy1 + sink_h // 2),
            max(2, int(min(sink_w, sink_h) * 0.18)),
            self.bg_color,
            -1,
            lineType=LINE_TYPE,
        )
        cv2.circle(
            img,
            (sx1 + sink_w // 2, sy1 + sink_h // 2),
            max(1, int(min(sink_w, sink_h) * 0.07)),
            WALL_COLOR,
            1,
            lineType=LINE_TYPE,
        )

        wc_r = int(base * 0.06)
        wc_cx = room.x1 + 12 + wc_r
        wc_cy = room.y2 - 12 - wc_r
        cv2.circle(img, (wc_cx, wc_cy), wc_r, FURNITURE_COLOR, -1, lineType=LINE_TYPE)
        cv2.circle(img, (wc_cx, wc_cy), wc_r, WALL_COLOR, 1, lineType=LINE_TYPE)
        self._add_occupied_region(wc_cx - wc_r, wc_cy - wc_r, wc_cx + wc_r, wc_cy + wc_r, "furniture")

        cv2.rectangle(
            img,
            (wc_cx - wc_r // 2, wc_cy - wc_r * 2),
            (wc_cx + wc_r // 2, wc_cy - wc_r),
            FURNITURE_COLOR,
            -1,
            lineType=LINE_TYPE,
        )
        cv2.rectangle(
            img,
            (wc_cx - wc_r // 2, wc_cy - wc_r * 2),
            (wc_cx + wc_r // 2, wc_cy - wc_r),
            WALL_COLOR,
            1,
            lineType=LINE_TYPE,
        )
        self._add_occupied_region(wc_cx - wc_r // 2, wc_cy - wc_r * 2, wc_cx + wc_r // 2, wc_cy - wc_r, "furniture")

    def _add_kitchen_furniture(self, img: np.ndarray, room: Room):
        rw = room.x2 - room.x1
        rh = room.y2 - room.y1

        if rw < 120 or rh < 120:
            return

        depth = int(min(rw, rh) * self.rng.uniform(0.08, 0.13))
        depth = max(depth, 24)

        side = self.rng.choice(["top", "bottom", "left", "right"])

        if side in ("top", "bottom"):
            usable_len = rw - depth * 2
            if usable_len <= depth * 2:
                return

            band_len = int(usable_len * self.rng.uniform(0.6, 0.95))
            x1 = self.rng.randint(room.x1 + depth, room.x1 + depth + usable_len - band_len)
            x2 = x1 + band_len

            if side == "top":
                y1 = room.y1 + 5
                y2 = y1 + depth
            else:
                y2 = room.y2 - 5
                y1 = y2 - depth

            cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, lineType=LINE_TYPE)
            cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, lineType=LINE_TYPE)
            self._add_occupied_region(x1, y1, x2, y2, "furniture")

            segments: List[List[int]] = [[x1, x2]]

            n_gaps = self.rng.randint(0, 2)
            for _ in range(n_gaps):
                if not segments:
                    break
                idx = self.rng.randrange(len(segments))
                sx1, sx2 = segments[idx]
                if sx2 - sx1 < depth * 3:
                    continue

                gap_w = self.rng.randint(int(depth * 0.7), int(depth * 1.3))

                left = sx1 + depth
                right = sx2 - depth - gap_w

                if left >= right:
                    continue

                gx1 = self.rng.randint(left, right)
                gx2 = gx1 + gap_w

                cv2.rectangle(
                    img, (gx1 + 1, y1 + 1), (gx2 - 1, y2 - 1),
                    self.bg_color, -1, lineType=LINE_TYPE
                )

                new_segments: List[List[int]] = []
                if gx1 - sx1 > depth * 1.5:
                    new_segments.append([sx1, gx1])
                if sx2 - gx2 > depth * 1.5:
                    new_segments.append([gx2, sx2])
                segments.pop(idx)
                segments.extend(new_segments)

            if segments:
                sx1, sx2 = self.rng.choice(segments)
                stove_w = min(int(depth * self.rng.uniform(0.9, 1.3)), sx2 - sx1 - 6)
                if stove_w > depth // 2:
                    st_x1 = self.rng.randint(sx1 + 3, sx2 - stove_w - 3)
                    st_x2 = st_x1 + stove_w
                    st_y1 = y1 + 3
                    st_y2 = y2 - 3

                    cv2.rectangle(img, (st_x1, st_y1), (st_x2, st_y2),
                                  FURNITURE_COLOR, -1, lineType=LINE_TYPE)
                    cv2.rectangle(img, (st_x1, st_y1), (st_x2, st_y2),
                                  WALL_COLOR, 1, lineType=LINE_TYPE)

                    n_cols = 2
                    n_rows = 2
                    cell_w = (st_x2 - st_x1) / (n_cols + 1)
                    cell_h = (st_y2 - st_y1) / (n_rows + 1)
                    r = int(min(cell_w, cell_h) * 0.35)
                    for i in range(n_cols):
                        for j in range(n_rows):
                            cx = int(st_x1 + (i + 1) * cell_w)
                            cy = int(st_y1 + (j + 1) * cell_h)
                            cv2.circle(img, (cx, cy), r, WALL_COLOR, 1, lineType=LINE_TYPE)

        else:
            usable_len = rh - depth * 2
            if usable_len <= depth * 2:
                return

            band_len = int(usable_len * self.rng.uniform(0.6, 0.95))
            y1 = self.rng.randint(room.y1 + depth,
                                  room.y1 + depth + usable_len - band_len)
            y2 = y1 + band_len

            if side == "left":
                x1 = room.x1 + 5
                x2 = x1 + depth
            else:
                x2 = room.x2 - 5
                x1 = x2 - depth

            cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, lineType=LINE_TYPE)
            cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, lineType=LINE_TYPE)
            self._add_occupied_region(x1, y1, x2, y2, "furniture")

            segments: List[List[int]] = [[y1, y2]]

            n_gaps = self.rng.randint(0, 2)
            for _ in range(n_gaps):
                if not segments:
                    break
                idx = self.rng.randrange(len(segments))
                sy1, sy2 = segments[idx]
                if sy2 - sy1 < depth * 3:
                    continue

                gap_h = self.rng.randint(int(depth * 0.7), int(depth * 1.3))

                left = sy1 + depth
                right = sy2 - depth - gap_h

                if left >= right:
                    continue

                gy1 = self.rng.randint(left, right)
                gy2 = gy1 + gap_h

                cv2.rectangle(
                    img, (x1 + 1, gy1 + 1), (x2 - 1, gy2 - 1),
                    self.bg_color, -1, lineType=LINE_TYPE
                )

                new_segments: List[List[int]] = []
                if gy1 - sy1 > depth * 1.5:
                    new_segments.append([sy1, gy1])
                if sy2 - gy2 > depth * 1.5:
                    new_segments.append([gy2, sy2])
                segments.pop(idx)
                segments.extend(new_segments)

            if segments:
                sy1, sy2 = self.rng.choice(segments)
                stove_h = min(int(depth * self.rng.uniform(0.9, 1.3)), sy2 - sy1 - 6)
                if stove_h > depth // 2:
                    st_y1 = self.rng.randint(sy1 + 3, sy2 - stove_h - 3)
                    st_y2 = st_y1 + stove_h
                    st_x1 = x1 + 3
                    st_x2 = x2 - 3

                    cv2.rectangle(img, (st_x1, st_y1), (st_x2, st_y2),
                                  FURNITURE_COLOR, -1, lineType=LINE_TYPE)
                    cv2.rectangle(img, (st_x1, st_y1), (st_x2, st_y2),
                                  WALL_COLOR, 1, lineType=LINE_TYPE)

                    n_cols = 2
                    n_rows = 2
                    cell_w = (st_x2 - st_x1) / (n_cols + 1)
                    cell_h = (st_y2 - st_y1) / (n_rows + 1)
                    r = int(min(cell_w, cell_h) * 0.35)
                    for i in range(n_cols):
                        for j in range(n_rows):
                            cx = int(st_x1 + (i + 1) * cell_w)
                            cy = int(st_y1 + (j + 1) * cell_h)
                            cv2.circle(img, (cx, cy), r, WALL_COLOR, 1, lineType=LINE_TYPE)

    def _draw_rounded_rect(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        fill_color: Tuple[int, int, int],
        radius: int,
        border_color: Optional[Tuple[int, int, int]] = None,
        border_th: int = 1,
    ):
        if x2 <= x1 or y2 <= y1:
            return

        w = x2 - x1
        h = y2 - y1

        radius = int(radius)
        if radius <= 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1, lineType=LINE_TYPE)
            if border_color is not None and border_th > 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border_th, lineType=LINE_TYPE)
            return

        radius = min(radius, w // 2, h // 2)
        if radius <= 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1, lineType=LINE_TYPE)
            if border_color is not None and border_th > 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border_th, lineType=LINE_TYPE)
            return

        cv2.rectangle(
            img,
            (x1 + radius, y1),
            (x2 - radius, y2),
            fill_color,
            -1,
            lineType=LINE_TYPE,
        )
        cv2.rectangle(
            img,
            (x1, y1 + radius),
            (x2, y2 - radius),
            fill_color,
            -1,
            lineType=LINE_TYPE,
        )

        cv2.circle(img, (x1 + radius, y1 + radius), radius, fill_color, -1, lineType=LINE_TYPE)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, fill_color, -1, lineType=LINE_TYPE)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, fill_color, -1, lineType=LINE_TYPE)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, fill_color, -1, lineType=LINE_TYPE)

        if border_color is not None and border_th > 0:
            cv2.line(
                img,
                (x1 + radius, y1),
                (x2 - radius, y1),
                border_color,
                border_th,
                lineType=LINE_TYPE,
            )
            cv2.line(
                img,
                (x1 + radius, y2),
                (x2 - radius, y2),
                border_color,
                border_th,
                lineType=LINE_TYPE,
            )
            cv2.line(
                img,
                (x1, y1 + radius),
                (x1, y2 - radius),
                border_color,
                border_th,
                lineType=LINE_TYPE,
            )
            cv2.line(
                img,
                (x2, y1 + radius),
                (x2, y2 - radius),
                border_color,
                border_th,
                lineType=LINE_TYPE,
            )

            cv2.ellipse(
                img,
                (x1 + radius, y1 + radius),
                (radius, radius),
                0,
                180,
                270,
                border_color,
                border_th,
                lineType=LINE_TYPE,
            )
            cv2.ellipse(
                img,
                (x2 - radius, y1 + radius),
                (radius, radius),
                0,
                270,
                360,
                border_color,
                border_th,
                lineType=LINE_TYPE,
            )
            cv2.ellipse(
                img,
                (x1 + radius, y2 - radius),
                (radius, radius),
                0,
                90,
                180,
                border_color,
                border_th,
                lineType=LINE_TYPE,
            )
            cv2.ellipse(
                img,
                (x2 - radius, y2 - radius),
                (radius, radius),
                0,
                0,
                90,
                border_color,
                border_th,
                lineType=LINE_TYPE,
            )

    def _add_door_like_chair(self, img: np.ndarray, room: Room):
        rw = room.x2 - room.x1
        rh = room.y2 - room.y1

        if rw < 80 or rh < 80:
            return

        margin = int(min(rw, rh) * 0.12)

        max_w = rw - 2 * margin - 2
        max_h = rh - 2 * margin - 2
        if max_w <= 4 or max_h <= 4:
            return

        shape_variant = self.rng.random()
        if shape_variant < 0.33:
            len_range = (0.01, 0.06)
            thick_ratio_range = (0.30, 0.90)
        elif shape_variant < 0.66:
            len_range = (0.045, 0.09)
            thick_ratio_range = (0.05, 0.25)
        else:
            len_range = (0.02, 0.07)
            thick_ratio_range = (0.10, 1.20)

        door_like_len = int(self.size * self.rng.uniform(*len_range))
        door_like_thick = int(door_like_len * self.rng.uniform(*thick_ratio_range))
        door_like_thick = max(3, door_like_thick)

        orientation = self.rng.choice(["vertical", "horizontal"])

        if orientation == "vertical":
            rect_h = door_like_len
            rect_w = door_like_thick
        else:
            rect_w = door_like_len
            rect_h = door_like_thick

        size_jitter = self.rng.uniform(0.7, 1.4)
        rect_w = max(3, int(rect_w * size_jitter))
        rect_h = max(3, int(rect_h * size_jitter))

        scale = min(max_w / rect_w, max_h / rect_h, 1.0)
        if scale < 1.0:
            rect_w = max(3, int(rect_w * scale))
            rect_h = max(3, int(rect_h * scale))
        if rect_w < 3 or rect_h < 3:
            return

        for _ in range(10):
            x_min = room.x1 + margin + rect_w // 2
            x_max = room.x2 - margin - rect_w // 2
            y_min = room.y1 + margin + rect_h // 2
            y_max = room.y2 - margin - rect_h // 2
            if x_max <= x_min or y_max <= y_min:
                return

            cx = self.rng.randint(x_min, x_max)
            cy = self.rng.randint(y_min, y_max)

            x1 = cx - rect_w // 2
            x2 = cx + rect_w // 2
            y1 = cy - rect_h // 2
            y2 = cy + rect_h // 2

            if self._check_collision(x1, y1, x2, y2, margin=10):
                continue

            corner_radius = int(min(rect_w, rect_h) * self.rng.uniform(0.15, 0.40))

            self._draw_rounded_rect(
                img,
                x1,
                y1,
                x2,
                y2,
                FURNITURE_COLOR,
                corner_radius,
                border_color=WALL_COLOR,
                border_th=1,
            )
            self._add_occupied_region(x1, y1, x2, y2, "furniture")

            if self.rng.random() < 0.55:
                if orientation == "vertical":
                    attach_left = self.rng.random() < 0.5

                    big_w = int(rect_w * self.rng.uniform(1.5, 3.0))
                    big_h = int(rect_h * self.rng.uniform(1.0, 1.5))

                    bcy = (y1 + y2) // 2
                    offset = int((rect_h + big_h) * 0.15 * (self.rng.random() - 0.5))
                    bcy = max(room.y1 + margin + big_h // 2,
                              min(room.y2 - margin - big_h // 2, bcy + offset))
                    by1 = bcy - big_h // 2
                    by2 = bcy + big_h // 2

                    if attach_left:
                        bx2 = x1
                        bx1 = bx2 - big_w
                    else:
                        bx1 = x2
                        bx2 = bx1 + big_w
                else:
                    attach_top = self.rng.random() < 0.5

                    big_h = int(rect_h * self.rng.uniform(1.5, 3.0))
                    big_w = int(rect_w * self.rng.uniform(1.0, 1.5))

                    bcx = (x1 + x2) // 2
                    offset = int((rect_w + big_w) * 0.15 * (self.rng.random() - 0.5))
                    bcx = max(room.x1 + margin + big_w // 2,
                              min(room.x2 - margin - big_w // 2, bcx + offset))
                    bx1 = bcx - big_w // 2
                    bx2 = bcx + big_w // 2

                    if attach_top:
                        by2 = y1
                        by1 = by2 - big_h
                    else:
                        by1 = y2
                        by2 = by1 + big_h

                if (bx1 >= room.x1 + margin and
                        bx2 <= room.x2 - margin and
                        by1 >= room.y1 + margin and
                        by2 <= room.y2 - margin and
                        not self._check_collision(bx1, by1, bx2, by2, margin=10)):
                    big_corner_radius = int(
                        min(bx2 - bx1, by2 - by1) * self.rng.uniform(0.15, 0.35)
                    )
                    self._draw_rounded_rect(
                        img,
                        bx1,
                        by1,
                        bx2,
                        by2,
                        FURNITURE_COLOR,
                        big_corner_radius,
                        border_color=WALL_COLOR,
                        border_th=1,
                    )
                    self._add_occupied_region(bx1, by1, bx2, by2, "furniture")

            detail_mode = self.rng.random()
            seat_w = x2 - x1
            seat_h = y2 - y1

            if seat_w > 10 and seat_h > 10:
                if detail_mode < 0.45:
                    pad_x = int(seat_w * self.rng.uniform(0.15, 0.30))
                    pad_y = int(seat_h * self.rng.uniform(0.15, 0.30))
                    ix1 = x1 + pad_x
                    iy1 = y1 + pad_y
                    ix2 = x2 - pad_x
                    iy2 = y2 - pad_y
                    if ix2 - ix1 > 4 and iy2 - iy1 > 4:
                        inner_radius = int(
                            min(ix2 - ix1, iy2 - iy1) * self.rng.uniform(0.25, 0.5)
                        )
                        self._draw_rounded_rect(
                            img,
                            ix1,
                            iy1,
                            ix2,
                            iy2,
                            self.bg_color,
                            inner_radius,
                            border_color=WALL_COLOR,
                            border_th=1,
                        )
                elif detail_mode < 0.9:
                    n_stripes = self.rng.randint(2, 4)
                    if seat_w >= seat_h:
                        step = seat_w / (n_stripes + 1)
                        for i in range(1, n_stripes + 1):
                            sx = int(x1 + step * i)
                            cv2.line(
                                img,
                                (sx, y1 + 2),
                                (sx, y2 - 2),
                                WALL_COLOR,
                                1,
                                lineType=LINE_TYPE,
                            )
                    else:
                        step = seat_h / (n_stripes + 1)
                        for i in range(1, n_stripes + 1):
                            sy = int(y1 + step * i)
                            cv2.line(
                                img,
                                (x1 + 2, sy),
                                (x2 - 2, sy),
                                WALL_COLOR,
                                1,
                                lineType=LINE_TYPE,
                            )

            if self.rng.random() < 0.2:
                if orientation == "vertical":
                    hinge_on_left = self.rng.random() < 0.5
                    hinge_x = x1 if hinge_on_left else x2
                    hinge_y = (y1 + y2) // 2
                    radius = int((y2 - y1) * self.rng.uniform(0.8, 1.1))

                    if hinge_on_left:
                        start_angle, end_angle = -60, 60
                    else:
                        start_angle, end_angle = 120, 240
                    cv2.ellipse(
                        img,
                        (hinge_x, hinge_y),
                        (radius, radius),
                        0,
                        start_angle,
                        end_angle,
                        WALL_COLOR,
                        1,
                        lineType=LINE_TYPE,
                    )
                else:
                    hinge_on_top = self.rng.random() < 0.5
                    hinge_x = (x1 + x2) // 2
                    hinge_y = y1 if hinge_on_top else y2
                    radius = int((x2 - x1) * self.rng.uniform(0.8, 1.1))

                    if hinge_on_top:
                        start_angle, end_angle = 210, 330
                    else:
                        start_angle, end_angle = 30, 150
                    cv2.ellipse(
                        img,
                        (hinge_x, hinge_y),
                        (radius, radius),
                        0,
                        start_angle,
                        end_angle,
                        WALL_COLOR,
                        1,
                        lineType=LINE_TYPE,
                    )
            break

    def _add_furniture(self, img: np.ndarray, rooms: List[Room], walls: List[Wall]):
        for room in rooms:
            rw = room.x2 - room.x1
            rh = room.y2 - room.y1

            if rw < 80 or rh < 80:
                continue

            if room.room_type == "kitchen":
                self._add_kitchen_furniture(img, room)
            elif room.room_type == "bathroom":
                self._add_bathroom_furniture(img, room)

            if room.room_type == "bathroom":
                n_items = self.rng.randint(0, 2)
            elif room.room_type == "kitchen":
                n_items = self.rng.randint(1, 3)
            elif room.room_type == "living":
                n_items = self.rng.randint(4, 8)
            else:
                n_items = self.rng.randint(0, 3)

            margin = 40

            for _ in range(n_items):
                for attempt in range(10):
                    shape_type = self.rng.choice(['rectangle', 'ellipse', 'circle'])

                    if shape_type == 'rectangle':
                        max_w = max(20, int(rw * 0.40))
                        max_h = max(20, int(rh * 0.40))
                        w = self.rng.randint(int(rw * 0.15), max_w)
                        h = self.rng.randint(int(rh * 0.15), max_h)

                        x_max = room.x2 - margin - w
                        x_min = room.x1 + margin
                        if x_max <= x_min:
                            continue

                        y_max = room.y2 - margin - h
                        y_min = room.y1 + margin
                        if y_max <= y_min:
                            continue

                        x = self.rng.randint(x_min, x_max)
                        y = self.rng.randint(y_min, y_max)

                        if self._check_collision(x, y, x + w, y + h):
                            continue

                        cv2.rectangle(img, (x, y), (x + w, y + h), FURNITURE_COLOR, -1, lineType=LINE_TYPE)
                        cv2.rectangle(img, (x, y), (x + w, y + h), WALL_COLOR, 1, lineType=LINE_TYPE)
                        self._add_occupied_region(x, y, x + w, y + h, "furniture")
                        break

                    elif shape_type == 'ellipse':
                        max_a = max(15, int(rw * 0.25))
                        max_b = max(15, int(rh * 0.25))
                        a = self.rng.randint(int(rw * 0.10), max_a)
                        b = self.rng.randint(int(rh * 0.10), max_b)

                        cx_max = room.x2 - margin - a
                        cx_min = room.x1 + margin + a
                        if cx_max <= cx_min:
                            continue

                        cy_max = room.y2 - margin - b
                        cy_min = room.y1 + margin + b
                        if cy_max <= cy_min:
                            continue

                        cx = self.rng.randint(cx_min, cx_max)
                        cy = self.rng.randint(cy_min, cy_max)

                        if self._check_collision(cx - a, cy - b, cx + a, cy + b):
                            continue

                        cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, FURNITURE_COLOR, -1, lineType=LINE_TYPE)
                        cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, WALL_COLOR, 1, lineType=LINE_TYPE)
                        self._add_occupied_region(cx - a, cy - b, cx + a, cy + b, "furniture")
                        break

                    else:
                        max_r = max(10, int(min(rw, rh) * 0.18))
                        r = self.rng.randint(int(min(rw, rh) * 0.08), max_r)

                        cx_max = room.x2 - margin - r
                        cx_min = room.x1 + margin + r
                        if cx_max <= cx_min:
                            continue

                        cy_max = room.y2 - margin - r
                        cy_min = room.y1 + margin + r
                        if cy_max <= cy_min:
                            continue

                        cx = self.rng.randint(cx_min, cx_max)
                        cy = self.rng.randint(cy_min, cy_max)

                        if self._check_collision(cx - r, cy - r, cx + r, cy + r):
                            continue

                        cv2.circle(img, (cx, cy), r, FURNITURE_COLOR, -1, lineType=LINE_TYPE)
                        cv2.circle(img, (cx, cy), r, WALL_COLOR, 1, lineType=LINE_TYPE)
                        self._add_occupied_region(cx - r, cy - r, cx + r, cy + r, "furniture")
                        break

            if room.room_type in ("living", "kitchen", "corridor"):
                if self.rng.random() < self.fake_door_chair_prob:
                    n_fake = self.rng.randint(
                        self.fake_door_chair_count[0],
                        self.fake_door_chair_count[1],
                    )
                    for _ in range(n_fake):
                        self._add_door_like_chair(img, room)

    def _add_radiator(
        self,
        img: np.ndarray,
        horizontal: bool,
        axis_start: int,
        axis_end: int,
        center_coord: int,
        wall_thickness: int,
    ):
        length = axis_end - axis_start
        if length < wall_thickness * 2 or wall_thickness < 4:
            return

        rad_len = int(length * self.rng.uniform(0.7, 0.9))
        rad_height = max(3, int(wall_thickness * self.rng.uniform(0.6, 0.9)))
        gap = max(3, int(wall_thickness * self.rng.uniform(0.2, 0.6)))

        if horizontal:
            interior_dir = 1 if center_coord < self.size // 2 else -1
            cx = (axis_start + axis_end) // 2
            x1 = max(0, cx - rad_len // 2)
            x2 = min(self.size - 1, cx + rad_len // 2)

            if interior_dir == 1:
                y1 = center_coord + wall_thickness // 2 + gap
                y2 = y1 + rad_height
            else:
                y2 = center_coord - wall_thickness // 2 - gap
                y1 = y2 - rad_height

            y1 = max(0, y1)
            y2 = min(self.size - 1, y2)
            if y2 - y1 < 3 or x2 - x1 < 5:
                return

            if self._check_collision(x1, y1, x2, y2):
                return

            cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, lineType=LINE_TYPE)
            cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, lineType=LINE_TYPE)
            self._add_occupied_region(x1, y1, x2, y2, "radiator")

            n_fins = self.rng.randint(4, 10)
            step = (x2 - x1) / (n_fins + 1)
            for i in range(1, n_fins + 1):
                fx = int(x1 + step * i)
                cv2.line(img, (fx, y1 + 1), (fx, y2 - 1), WALL_COLOR, 1, lineType=LINE_TYPE)

        else:
            interior_dir = 1 if center_coord < self.size // 2 else -1
            cy = (axis_start + axis_end) // 2
            y1 = max(0, cy - rad_len // 2)
            y2 = min(self.size - 1, cy + rad_len // 2)

            if interior_dir == 1:
                x1 = center_coord + wall_thickness // 2 + gap
                x2 = x1 + rad_height
            else:
                x2 = center_coord - wall_thickness // 2 - gap
                x1 = x2 - rad_height

            x1 = max(0, x1)
            x2 = min(self.size - 1, x2)
            if x2 - x1 < 3 or y2 - y1 < 5:
                return

            if self._check_collision(x1, y1, x2, y2):
                return

            cv2.rectangle(img, (x1, y1), (x2, y2), FURNITURE_COLOR, -1, lineType=LINE_TYPE)
            cv2.rectangle(img, (x1, y1), (x2, y2), WALL_COLOR, 1, lineType=LINE_TYPE)
            self._add_occupied_region(x1, y1, x2, y2, "radiator")

            n_fins = self.rng.randint(4, 10)
            step = (y2 - y1) / (n_fins + 1)
            for i in range(1, n_fins + 1):
                fy = int(y1 + step * i)
                cv2.line(img, (x1 + 1, fy), (x2 - 1, fy), WALL_COLOR, 1, lineType=LINE_TYPE)

    def _place_window(self, img: np.ndarray, w: Wall) -> Optional[Tuple[float, float, float, float]]:
        """
        Размещение окна на внешней стене.

        Типы:
          - standard        — основной вариант, 1–3 секции, средняя длина
          - wide            — длинные панорамные окна, 2–4 секции
          - narrow          — узкие/короткие окна (санузлы и т.п.)
          - balcony_door    — балконные двери с нижней глухой частью
          - dead_end        — окно в торце/«глухом» участке стены

        Стиль рам:
          - double_line  — две близкие параллельные линии (доминирующий вариант)
          - single_line  — одна линия
          - thick_frame  — две центральные линии + ещё две у краёв стены (толстое окно)
          - gap_only     — просто прорезь (редко)

        Для большинства наружных окон дополнительно рисуется радиатор.
        """
        pad = max(20, int(self.size * 0.01))
        is_horizontal = abs(w.x2 - w.x1) > abs(w.y2 - w.y1)

        # Тип окна: стандартные доминируют
        window_type = self.rng.choices(
            ['standard', 'balcony_door', 'wide', 'narrow', 'dead_end'],
            weights=[0.60, 0.12, 0.15, 0.09, 0.04]
        )[0]

        # Параметры длины и секций
        if window_type == 'balcony_door':
            length_ratios = (0.05, 0.07, 0.09, 0.12)
            num_segments = self.rng.choice([1, 1, 2, 2, 3])
            has_bottom_section = self.rng.random() < 0.7
        elif window_type == 'wide':
            length_ratios = (0.07, 0.09, 0.11, 0.14, 0.16)
            num_segments = self.rng.choice([2, 2, 3, 3, 4])
            has_bottom_section = False
        elif window_type == 'narrow':
            length_ratios = (0.02, 0.03, 0.04)
            num_segments = self.rng.choice([1, 1, 1, 2])
            has_bottom_section = False
        elif window_type == 'dead_end':
            length_ratios = (0.04, 0.06, 0.08)
            num_segments = 1
            has_bottom_section = False
        else:  # standard
            length_ratios = self.window_length_ratios
            num_segments = self.rng.choice([1, 1, 2, 2, 3])
            has_bottom_section = False

        typical_lengths = [int(self.size * r) for r in length_ratios]

        # Стиль рам: thick_frame редкий, но даёт толстое окно
        frame_style = self.rng.choices(
            ['double_line', 'single_line', 'thick_frame', 'gap_only'],
            weights=[0.65, 0.20, 0.10, 0.05]
        )[0]

        if frame_style == 'thick_frame':
            line_thickness = self.rng.randint(2, 3)
            gap_width = self.rng.randint(4, 10)
        else:
            line_thickness = self.rng.randint(1, 2)
            gap_width = self.rng.randint(2, 8)

        mullion_thickness = max(1, line_thickness - 1)

        # Цвета рам: мебельный цвет доминирует
        use_different_colors = self.rng.random() < 0.2

        def lerp_color(c1, c2, t):
            return (
                int(c1[0] + (c2[0] - c1[0]) * t),
                int(c1[1] + (c2[1] - c1[1]) * t),
                int(c1[2] + (c2[2] - c1[2]) * t),
            )

        if use_different_colors:
            t1 = self.rng.random()
            t2 = self.rng.random()
            frame_color1 = lerp_color(FURNITURE_COLOR, WALL_COLOR, t1)
            frame_color2 = lerp_color(FURNITURE_COLOR, WALL_COLOR, t2)
        else:
            t = self.rng.random()
            base_color = lerp_color(FURNITURE_COLOR, WALL_COLOR, t)
            frame_color1 = base_color
            frame_color2 = base_color

        # Смещение рам относительно толщины стены
        offset_mode = self.rng.choices(
            ['center', 'inner', 'outer'],
            weights=[0.75, 0.18, 0.07]
        )[0]

        if is_horizontal:
            # ----------------- ГОРИЗОНТАЛЬНАЯ СТЕНА -----------------
            axis_start, axis_end = min(w.x1, w.x2), max(w.x1, w.x2)
            wall_center_y = (w.y1 + w.y2) // 2
            half = w.thickness // 2
            y_top = wall_center_y - half
            y_bottom = wall_center_y + half

            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None

            length_candidates = [L for L in typical_lengths if w.thickness * 3 <= L <= avail * 0.9]
            if length_candidates:
                win_len = self.rng.choice(length_candidates)
            else:
                min_win_len = max(int(w.thickness * 3), int(avail * 0.3))
                max_win_len = int(avail * 0.9)
                if max_win_len < min_win_len:
                    return None
                win_len = self.rng.randint(min_win_len, max_win_len)

            offset_range = avail - win_len
            if offset_range < 0:
                return None

            for _ in range(10):
                start = axis_start + pad + self.rng.randint(0, offset_range)
                end = start + win_len

                # Центр рам (с возможным смещением внутрь/наружу)
                frame_center_y = wall_center_y
                interior_dir = 1 if wall_center_y < self.size // 2 else -1
                shift = int(w.thickness * 0.25)
                if offset_mode == 'inner':
                    frame_center_y += interior_dir * shift
                elif offset_mode == 'outer':
                    frame_center_y -= interior_dir * shift

                offset = (gap_width + line_thickness) // 2
                line1_y = int(frame_center_y - offset - line_thickness // 2)
                line2_y = int(frame_center_y + offset + line_thickness // 2)

                y1 = line1_y - line_thickness // 2
                y2 = line2_y + line_thickness // 2

                # Для thick_frame расширяем bbox до всей толщины стены
                if frame_style == 'thick_frame':
                    y1 = min(y1, y_top)
                    y2 = max(y2, y_bottom)

                # Вылет для dead_end
                extend = int(w.thickness * self.rng.uniform(0.6, 1.0))
                if window_type == 'dead_end':
                    if self.rng.random() < 0.5:
                        start_ext, end_ext = start - extend, end
                        close_x = start
                    else:
                        start_ext, end_ext = start, end + extend
                        close_x = end
                else:
                    start_ext, end_ext = start, end

                if self._check_collision(start_ext, y1, end_ext, y2, margin=15):
                    continue

                # Прорезаем стену
                cv2.rectangle(
                    img,
                    (start_ext, y_top),
                    (end_ext, y_bottom),
                    self.bg_color,
                    -1,
                    lineType=LINE_TYPE,
                )

                # Рисуем рамы по стилю
                if frame_style == 'gap_only':
                    if self.rng.random() < 0.8:
                        cv2.rectangle(
                            img,
                            (start_ext, y_top),
                            (end_ext, y_bottom),
                            frame_color1,
                            1,
                            lineType=LINE_TYPE,
                        )
                elif frame_style == 'single_line':
                    cv2.line(
                        img,
                        (start_ext, frame_center_y),
                        (end_ext, frame_center_y),
                        frame_color1,
                        line_thickness,
                        lineType=LINE_TYPE,
                    )
                else:
                    # Центральная пара линий
                    cv2.line(
                        img,
                        (start_ext, line1_y),
                        (end_ext, line1_y),
                        frame_color1,
                        line_thickness,
                        lineType=LINE_TYPE,
                    )
                    cv2.line(
                        img,
                        (start_ext, line2_y),
                        (end_ext, line2_y),
                        frame_color2,
                        line_thickness,
                        lineType=LINE_TYPE,
                    )

                    # Дополнительные линии на краях стены -> "толстое" окно
                    if frame_style == 'thick_frame':
                        outer_y1 = y_top + max(1, line_thickness // 2)
                        outer_y2 = y_bottom - max(1, line_thickness // 2)
                        if outer_y2 > outer_y1:
                            cv2.line(
                                img,
                                (start_ext, outer_y1),
                                (end_ext, outer_y1),
                                frame_color1,
                                line_thickness,
                                lineType=LINE_TYPE,
                            )
                            cv2.line(
                                img,
                                (start_ext, outer_y2),
                                (end_ext, outer_y2),
                                frame_color1,
                                line_thickness,
                                lineType=LINE_TYPE,
                            )

                if window_type == 'dead_end' and frame_style != 'gap_only':
                    cv2.line(
                        img,
                        (close_x, line1_y),
                        (close_x, line2_y),
                        frame_color1,
                        line_thickness,
                        lineType=LINE_TYPE,
                    )

                # Импосты
                if num_segments > 1 and frame_style != 'gap_only':
                    segment_width = win_len / num_segments
                    mullion_color = frame_color1
                    for i in range(1, num_segments):
                        mullion_x = int(start + segment_width * i)
                        cv2.line(
                            img,
                            (mullion_x, line1_y + 1),
                            (mullion_x, line2_y - 1),
                            mullion_color,
                            mullion_thickness,
                            lineType=LINE_TYPE,
                        )

                # Нижняя глухая часть балконной двери
                if has_bottom_section and window_type == 'balcony_door':
                    section_height = int((line2_y - line1_y) * self.rng.uniform(0.25, 0.35))
                    section_y = line2_y - line_thickness // 2 - section_height
                    cv2.rectangle(
                        img,
                        (start + 2, section_y),
                        (end - 2, line2_y - 1),
                        FURNITURE_COLOR,
                        -1,
                        lineType=LINE_TYPE,
                    )
                    cv2.line(
                        img,
                        (start, section_y),
                        (end, section_y),
                        frame_color1,
                        mullion_thickness,
                        lineType=LINE_TYPE,
                    )

                x_min, x_max = min(start_ext, end_ext), max(start_ext, end_ext)
                self._add_occupied_region(x_min, y1, x_max, y2, "window")

                # Радиатор под окном (кроме балконных дверей)
                if window_type != 'balcony_door' and self.rng.random() < 0.85:
                    self._add_radiator(
                        img,
                        horizontal=True,
                        axis_start=start,
                        axis_end=end,
                        center_coord=wall_center_y,
                        wall_thickness=w.thickness,
                    )

                return self._points_to_yolo(
                    [(x_min, y1), (x_max, y1), (x_min, y2), (x_max, y2)]
                )

            return None

        else:
            # ----------------- ВЕРТИКАЛЬНАЯ СТЕНА -----------------
            axis_start, axis_end = min(w.y1, w.y2), max(w.y1, w.y2)
            wall_center_x = (w.x1 + w.x2) // 2
            half = w.thickness // 2
            x_left = wall_center_x - half
            x_right = wall_center_x + half

            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None

            length_candidates = [L for L in typical_lengths if w.thickness * 3 <= L <= avail * 0.9]
            if length_candidates:
                win_len = self.rng.choice(length_candidates)
            else:
                min_win_len = max(int(w.thickness * 3), int(avail * 0.3))
                max_win_len = int(avail * 0.9)
                if max_win_len < min_win_len:
                    return None
                win_len = self.rng.randint(min_win_len, max_win_len)

            offset_range = avail - win_len
            if offset_range < 0:
                return None

            for _ in range(10):
                start = axis_start + pad + self.rng.randint(0, offset_range)
                end = start + win_len

                frame_center_x = wall_center_x
                interior_dir = 1 if wall_center_x < self.size // 2 else -1
                shift = int(w.thickness * 0.25)
                if offset_mode == 'inner':
                    frame_center_x += interior_dir * shift
                elif offset_mode == 'outer':
                    frame_center_x -= interior_dir * shift

                offset = (gap_width + line_thickness) // 2
                line1_x = int(frame_center_x - offset - line_thickness // 2)
                line2_x = int(frame_center_x + offset + line_thickness // 2)

                x1 = line1_x - line_thickness // 2
                x2 = line2_x + line_thickness // 2

                if frame_style == 'thick_frame':
                    x1 = min(x1, x_left)
                    x2 = max(x2, x_right)

                extend = int(w.thickness * self.rng.uniform(0.6, 1.0))
                if window_type == 'dead_end':
                    if self.rng.random() < 0.5:
                        start_ext, end_ext = start - extend, end
                        close_y = start
                    else:
                        start_ext, end_ext = start, end + extend
                        close_y = end
                else:
                    start_ext, end_ext = start, end

                if self._check_collision(x1, start_ext, x2, end_ext, margin=15):
                    continue

                cv2.rectangle(
                    img,
                    (x_left, start_ext),
                    (x_right, end_ext),
                    self.bg_color,
                    -1,
                    lineType=LINE_TYPE,
                )

                if frame_style == 'gap_only':
                    if self.rng.random() < 0.8:
                        cv2.rectangle(
                            img,
                            (x_left, start_ext),
                            (x_right, end_ext),
                            frame_color1,
                            1,
                            lineType=LINE_TYPE,
                        )
                elif frame_style == 'single_line':
                    cv2.line(
                        img,
                        (frame_center_x, start_ext),
                        (frame_center_x, end_ext),
                        frame_color1,
                        line_thickness,
                        lineType=LINE_TYPE,
                    )
                else:
                    cv2.line(
                        img,
                        (line1_x, start_ext),
                        (line1_x, end_ext),
                        frame_color1,
                        line_thickness,
                        lineType=LINE_TYPE,
                    )
                    cv2.line(
                        img,
                        (line2_x, start_ext),
                        (line2_x, end_ext),
                        frame_color2,
                        line_thickness,
                        lineType=LINE_TYPE,
                    )

                    if frame_style == 'thick_frame':
                        outer_x1 = x_left + max(1, line_thickness // 2)
                        outer_x2 = x_right - max(1, line_thickness // 2)
                        if outer_x2 > outer_x1:
                            cv2.line(
                                img,
                                (outer_x1, start_ext),
                                (outer_x1, end_ext),
                                frame_color1,
                                line_thickness,
                                lineType=LINE_TYPE,
                            )
                            cv2.line(
                                img,
                                (outer_x2, start_ext),
                                (outer_x2, end_ext),
                                frame_color1,
                                line_thickness,
                                lineType=LINE_TYPE,
                            )

                if window_type == 'dead_end' and frame_style != 'gap_only':
                    cv2.line(
                        img,
                        (line1_x, close_y),
                        (line2_x, close_y),
                        frame_color1,
                        line_thickness,
                        lineType=LINE_TYPE,
                    )

                if num_segments > 1 and frame_style != 'gap_only':
                    segment_height = win_len / num_segments
                    mullion_color = frame_color1
                    for i in range(1, num_segments):
                        mullion_y = int(start + segment_height * i)
                        cv2.line(
                            img,
                            (line1_x + 1, mullion_y),
                            (line2_x - 1, mullion_y),
                            mullion_color,
                            mullion_thickness,
                            lineType=LINE_TYPE,
                        )

                y_min, y_max = min(start_ext, end_ext), max(start_ext, end_ext)
                self._add_occupied_region(x1, y_min, x2, y_max, "window")

                if window_type != 'balcony_door' and self.rng.random() < 0.85:
                    self._add_radiator(
                        img,
                        horizontal=False,
                        axis_start=start,
                        axis_end=end,
                        center_coord=wall_center_x,
                        wall_thickness=w.thickness,
                    )

                return self._points_to_yolo(
                    [(x1, y_min), (x2, y_min), (x1, y_max), (x2, y_max)]
                )

            return None

    def _place_door(
        self,
        img: np.ndarray,
        w: Wall,
        occupied_spans: List[Tuple[int, int]],
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Размещение двери на внутренней стене.

        Типы дверей:
          - standard     — основной вариант, ширина по self.door_length_ratios, дуга ~90°
          - wide         — чуть шире стандартной
          - narrow       — более узкая (кладовые, с/у)
          - no_arc       — дверь без дуги (редко используется)
          - double_leaf  — визуально более широкая дверь (одна дуга, но шире проём)

        Доминирует стандартная дверь с дугой, остальные типы добавляют разнообразие.
        """
        # Тип двери: стандартная — доминирующий случай
        door_type = self.rng.choices(
            ['standard', 'wide', 'narrow', 'no_arc', 'double_leaf'],
            weights=[0.6, 0.15, 0.1, 0.1, 0.05]
        )[0]

        # Толщина полотна двери
        base_th_min, base_th_max = 2, 7
        if door_type == 'wide' or door_type == 'double_leaf':
            base_th_max = 11
        elif door_type == 'narrow':
            base_th_max = 6
        door_thickness = self.rng.randint(base_th_min, base_th_max)

        # Толщина дуги (как раньше)
        arc_thickness = 1

        # Масштаб длины двери относительно базовых ratios
        length_scale = 1.0
        if door_type == 'wide':
            length_scale = self.rng.uniform(1.2, 1.5)
        elif door_type == 'narrow':
            length_scale = self.rng.uniform(0.7, 0.9)
        elif door_type == 'double_leaf':
            length_scale = self.rng.uniform(1.3, 1.6)

        # Угол открывания: для стандартных дверей стремится к 90°
        if door_type == 'no_arc':
            # дуга не рисуется, но геометрия полотна по‑прежнему слегка открыта
            open_angle_range = (25, 45)
        else:
            open_angle_range = (70, 105)

        pad = max(40, int(w.thickness * 2.5))
        is_horizontal = abs(w.x2 - w.x1) > abs(w.y2 - w.y1)

        if is_horizontal:
            # -------- ДВЕРЬ НА ГОРИЗОНТАЛЬНОЙ СТЕНЕ --------
            axis_start, axis_end = min(w.x1, w.x2), max(w.x1, w.x2)
            y_center = (w.y1 + w.y2) // 2

            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None

            # Кандидаты по длине двери
            length_candidates: List[int] = []
            for r in self.door_length_ratios:
                L = int(self.size * r * length_scale)
                if w.thickness * 2 <= L <= avail * 0.9:
                    length_candidates.append(L)

            if length_candidates:
                door_len = self.rng.choice(length_candidates)
            else:
                min_door_len = max(int(w.thickness * 2), int(avail * 0.25))
                max_door_len = int(avail * 0.6)
                if max_door_len < min_door_len:
                    return None
                door_len = self.rng.randint(min_door_len, max_door_len)

            for _ in range(10):
                usable = avail - door_len
                if usable <= 0:
                    return None
                edge_span = int(usable * 0.4)
                if edge_span <= 0:
                    edge_span = usable

                side = self.rng.choice(["start", "end"])
                if side == "start":
                    left_bound = axis_start + pad
                    right_bound = min(axis_start + pad + edge_span, axis_end - pad - door_len)
                else:
                    right_bound = axis_end - pad - door_len
                    left_bound = max(axis_end - pad - door_len - edge_span, axis_start + pad)

                if left_bound > right_bound:
                    continue

                gap_start = self.rng.randint(left_bound, right_bound)
                gap_end = gap_start + door_len

                # Проверка, не пересекаемся ли с уже размещёнными дверями
                overlap = False
                for s, e in occupied_spans:
                    if not (gap_end <= s or gap_start >= e):
                        overlap = True
                        break
                if overlap:
                    continue

                half = w.thickness // 2
                y1 = y_center - half
                y2 = y_center + half

                hinge_left = self.rng.choice([True, False])
                open_up = self.rng.choice([True, False])
                swing_angle = self.rng.randint(*open_angle_range)

                if hinge_left:
                    hinge_pos = (gap_start, y_center)
                    closed_angle = 0
                    open_angle = -swing_angle if open_up else swing_angle
                else:
                    hinge_pos = (gap_end, y_center)
                    closed_angle = 180
                    open_angle = 180 + swing_angle if open_up else 180 - swing_angle

                open_rad = np.deg2rad(open_angle)
                open_pos = (
                    hinge_pos[0] + int(door_len * np.cos(open_rad)),
                    hinge_pos[1] + int(door_len * np.sin(open_rad)),
                )

                perp_angle = open_rad + np.pi / 2
                half_thick = door_thickness / 2
                dx = int(half_thick * np.cos(perp_angle))
                dy = int(half_thick * np.sin(perp_angle))

                door_corners = np.array(
                    [
                        [hinge_pos[0] - dx, hinge_pos[1] - dy],
                        [hinge_pos[0] + dx, hinge_pos[1] + dy],
                        [open_pos[0] + dx, open_pos[1] + dy],
                        [open_pos[0] - dx, open_pos[1] - dy],
                    ],
                    dtype=np.int32,
                )

                # Bounding‑box двери
                all_x = [p[0] for p in door_corners] + [gap_start, gap_end, hinge_pos[0], open_pos[0]]
                all_y = [p[1] for p in door_corners] + [y1, y2, hinge_pos[1], open_pos[1]]
                door_x1 = min(all_x)
                door_y1 = min(all_y)
                door_x2 = max(all_x)
                door_y2 = max(all_y)

                if self._check_collision(door_x1, door_y1, door_x2, door_y2, margin=20):
                    continue

                occupied_spans.append((gap_start, gap_end))

                # Прорезь в стене под дверь
                cv2.rectangle(img, (gap_start, y1), (gap_end, y2), self.bg_color, -1, lineType=LINE_TYPE)

                # Полотно двери
                cv2.fillPoly(img, [door_corners], BG_COLOR, lineType=LINE_TYPE)
                cv2.polylines(
                    img,
                    [door_corners],
                    True,
                    WALL_COLOR,
                    1,
                    lineType=LINE_TYPE,
                )

                # Дуга открывания (кроме no_arc)
                arc_start = min(closed_angle, open_angle)
                arc_end = max(closed_angle, open_angle)
                if door_type != 'no_arc':
                    cv2.ellipse(
                        img,
                        hinge_pos,
                        (door_len, door_len),
                        0,
                        arc_start,
                        arc_end,
                        WALL_COLOR,
                        arc_thickness,
                        lineType=LINE_TYPE,
                    )

                closed_rad = np.deg2rad(closed_angle)
                closed_pos = (
                    hinge_pos[0] + int(door_len * np.cos(closed_rad)),
                    hinge_pos[1] + int(door_len * np.sin(closed_rad)),
                )

                # Для "двустворчатой" двери можно слегка подчеркнуть ширину
                if door_type == 'double_leaf':
                    mid_x = (hinge_pos[0] + closed_pos[0]) // 2
                    mid_y = (hinge_pos[1] + closed_pos[1]) // 2
                    cv2.circle(img, (mid_x, mid_y), 2, WALL_COLOR, -1, lineType=LINE_TYPE)

                self._add_occupied_region(door_x1, door_y1, door_x2, door_y2, "door")

                all_pts = [
                    (gap_start, y1),
                    (gap_start, y2),
                    (gap_end, y1),
                    (gap_end, y2),
                    hinge_pos,
                    open_pos,
                    closed_pos,
                ]
                return self._points_to_yolo(all_pts)

            return None

        else:
            # -------- ДВЕРЬ НА ВЕРТИКАЛЬНОЙ СТЕНЕ --------
            axis_start, axis_end = min(w.y1, w.y2), max(w.y1, w.y2)
            x_center = (w.x1 + w.x2) // 2

            avail = axis_end - axis_start - 2 * pad
            if avail <= w.thickness * 2:
                return None

            length_candidates: List[int] = []
            for r in self.door_length_ratios:
                L = int(self.size * r * length_scale)
                if w.thickness * 2 <= L <= avail * 0.9:
                    length_candidates.append(L)

            if length_candidates:
                door_len = self.rng.choice(length_candidates)
            else:
                min_door_len = max(int(w.thickness * 2), int(avail * 0.25))
                max_door_len = int(avail * 0.6)
                if max_door_len < min_door_len:
                    return None
                door_len = self.rng.randint(min_door_len, max_door_len)

            for _ in range(10):
                usable = avail - door_len
                if usable <= 0:
                    return None
                edge_span = int(usable * 0.4)
                if edge_span <= 0:
                    edge_span = usable

                side = self.rng.choice(["start", "end"])
                if side == "start":
                    top_bound = axis_start + pad
                    bottom_bound = min(axis_start + pad + edge_span, axis_end - pad - door_len)
                else:
                    bottom_bound = axis_end - pad - door_len
                    top_bound = max(axis_end - pad - door_len - edge_span, axis_start + pad)

                if top_bound > bottom_bound:
                    continue

                gap_start = self.rng.randint(top_bound, bottom_bound)
                gap_end = gap_start + door_len

                overlap = False
                for s, e in occupied_spans:
                    if not (gap_end <= s or gap_start >= e):
                        overlap = True
                        break
                if overlap:
                    continue

                half = w.thickness // 2
                x1 = x_center - half
                x2 = x_center + half

                hinge_top = self.rng.choice([True, False])
                open_right = self.rng.choice([True, False])
                swing_angle = self.rng.randint(*open_angle_range)

                if hinge_top:
                    hinge_pos = (x_center, gap_start)
                    closed_angle = 90
                    open_angle = 90 - swing_angle if open_right else 90 + swing_angle
                else:
                    hinge_pos = (x_center, gap_end)
                    closed_angle = 270
                    open_angle = 270 + swing_angle if open_right else 270 - swing_angle

                open_rad = np.deg2rad(open_angle)
                open_pos = (
                    hinge_pos[0] + int(door_len * np.cos(open_rad)),
                    hinge_pos[1] + int(door_len * np.sin(open_rad)),
                )

                perp_angle = open_rad + np.pi / 2
                half_thick = door_thickness / 2
                dx = int(half_thick * np.cos(perp_angle))
                dy = int(half_thick * np.sin(perp_angle))

                door_corners = np.array(
                    [
                        [hinge_pos[0] - dx, hinge_pos[1] - dy],
                        [hinge_pos[0] + dx, hinge_pos[1] + dy],
                        [open_pos[0] + dx, open_pos[1] + dy],
                        [open_pos[0] - dx, open_pos[1] - dy],
                    ],
                    dtype=np.int32,
                )

                all_x = [p[0] for p in door_corners] + [x1, x2, hinge_pos[0], open_pos[0]]
                all_y = [p[1] for p in door_corners] + [gap_start, gap_end, hinge_pos[1], open_pos[1]]
                door_x1 = min(all_x)
                door_y1 = min(all_y)
                door_x2 = max(all_x)
                door_y2 = max(all_y)

                if self._check_collision(door_x1, door_y1, door_x2, door_y2, margin=20):
                    continue

                occupied_spans.append((gap_start, gap_end))

                cv2.rectangle(img, (x1, gap_start), (x2, gap_end), self.bg_color, -1, lineType=LINE_TYPE)

                cv2.fillPoly(img, [door_corners], BG_COLOR, lineType=LINE_TYPE)
                cv2.polylines(
                    img,
                    [door_corners],
                    True,
                    WALL_COLOR,
                    1,
                    lineType=LINE_TYPE,
                )

                arc_start = min(closed_angle, open_angle)
                arc_end = max(closed_angle, open_angle)
                if door_type != 'no_arc':
                    cv2.ellipse(
                        img,
                        hinge_pos,
                        (door_len, door_len),
                        0,
                        arc_start,
                        arc_end,
                        WALL_COLOR,
                        arc_thickness,
                        lineType=LINE_TYPE,
                    )

                closed_rad = np.deg2rad(closed_angle)
                closed_pos = (
                    hinge_pos[0] + int(door_len * np.cos(closed_rad)),
                    hinge_pos[1] + int(door_len * np.sin(closed_rad)),
                )

                if door_type == 'double_leaf':
                    mid_x = (hinge_pos[0] + closed_pos[0]) // 2
                    mid_y = (hinge_pos[1] + closed_pos[1]) // 2
                    cv2.circle(img, (mid_x, mid_y), 2, WALL_COLOR, -1, lineType=LINE_TYPE)

                self._add_occupied_region(door_x1, door_y1, door_x2, door_y2, "door")

                all_pts = [
                    (x1, gap_start),
                    (x2, gap_start),
                    (x1, gap_end),
                    (x2, gap_end),
                    hinge_pos,
                    open_pos,
                    closed_pos,
                ]
                return self._points_to_yolo(all_pts)

            return None

    def _points_to_yolo(
        self,
        pts: List[Tuple[int, int]],
        pad: int = 0,
    ) -> Optional[Tuple[float, float, float, float]]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        x_min = max(0, min(xs) - pad)
        x_max = min(self.size - 1, max(xs) + pad)
        y_min = max(0, min(ys) - pad)
        y_max = min(self.size - 1, max(ys) + pad)

        if x_max <= x_min or y_max <= y_min:
            return None

        cx = (x_min + x_max) / 2.0 / self.size
        cy = (y_min + y_max) / 2.0 / self.size
        w = (x_max - x_min) / self.size
        h = (y_max - y_min) / self.size

        return (
            min(max(cx, 0.0), 1.0),
            min(max(cy, 0.0), 1.0),
            min(max(w, 0.0), 1.0),
            min(max(h, 0.0), 1.0),
        )

    def _apply_rotation(
        self,
        img: np.ndarray,
        labels: List[Tuple[int, float, float, float, float]],
    ) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        size = img.shape[0]
        angle = self.rng.uniform(-15, 15)
        center = (size // 2, size // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (size, size), borderValue=self.bg_color)

        rotated_labels: List[Tuple[int, float, float, float, float]] = []
        angle_rad = np.deg2rad(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for cls_id, cx, cy, w, h in labels:
            cx_px = cx * size
            cy_px = cy * size
            w_px = w * size
            h_px = h * size

            corners = [
                (cx_px - w_px / 2, cy_px - h_px / 2),
                (cx_px + w_px / 2, cy_px - h_px / 2),
                (cx_px + w_px / 2, cy_px + h_px / 2),
                (cx_px - w_px / 2, cy_px + h_px / 2),
            ]

            rotated_corners = []
            for x, y in corners:
                x_rel = x - center[0]
                y_rel = y - center[1]
                x_rot = x_rel * cos_a - y_rel * sin_a
                y_rot = x_rel * sin_a + y_rel * cos_a
                rotated_corners.append((x_rot + center[0], y_rot + center[1]))

            xs = [c[0] for c in rotated_corners]
            ys = [c[1] for c in rotated_corners]
            x_min = max(0, min(xs))
            x_max = min(size, max(xs))
            y_min = max(0, min(ys))
            y_max = min(size, max(ys))

            new_cx = (x_min + x_max) / 2 / size
            new_cy = (y_min + y_max) / 2 / size
            new_w = (x_max - x_min) / size
            new_h = (y_max - y_min) / size

            if new_w > 0.01 and new_h > 0.01:
                rotated_labels.append(
                    (
                        cls_id,
                        min(max(new_cx, 0.0), 1.0),
                        min(max(new_cy, 0.0), 1.0),
                        min(max(new_w, 0.0), 1.0),
                        min(max(new_h, 0.0), 1.0),
                    )
                )

        return rotated_img, rotated_labels

    def _apply_augmentations(self, img: np.ndarray) -> np.ndarray:
        if self.rng.random() < 0.5:
            ksize = self.rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), self.rng.uniform(0.3, 0.7))

        if self.rng.random() < 0.3:
            noise_std = self.rng.uniform(2, 8)
            noise = self.nprng.normal(0, noise_std, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if self.rng.random() < 0.5:
            alpha = self.rng.uniform(0.97, 1.03)
            beta = self.rng.uniform(-5, 5)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if self.rng.random() < 0.4:
            q = self.rng.randint(70, 95)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            ok, encimg = cv2.imencode(".jpg", img, encode_param)
            if ok:
                img = cv2.imdecode(encimg, 1)

        return img

    def _add_annotations(self, img: np.ndarray, rooms: List[Room]):
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = 0.0007 * self.size
        thickness = max(1, int(self.size * 0.0012))

        for room in rooms:
            cx = (room.x1 + room.x2) // 2
            cy = (room.y1 + room.y2) // 2
            if room.room_type == "living":
                area = self.rng.uniform(15.0, 50.0)
            elif room.room_type == "kitchen":
                area = self.rng.uniform(8.0, 20.0)
            elif room.room_type == "bathroom":
                area = self.rng.uniform(3.0, 8.0)
            else:
                area = self.rng.uniform(4.0, 15.0)
            txt = f"{area:.1f}".replace(".", ",")
            cv2.putText(
                img,
                txt,
                (cx, cy),
                font,
                base_scale,
                WALL_COLOR,
                thickness,
                lineType=LINE_TYPE,
            )

        corridor_rooms = [r for r in rooms if r.room_type == "corridor"]
        if corridor_rooms:
            r = corridor_rooms[0]
            cx = (r.x1 + r.x2) // 2
            cy = (r.y1 + r.y2) // 2
            radius = int(self.size * 0.04)
            cv2.circle(img, (cx, cy), radius, WALL_COLOR, -1, lineType=LINE_TYPE)

            apt_num = str(self.rng.randint(1, 4))
            text_scale = base_scale * 1.4
            text_size = cv2.getTextSize(apt_num, font, text_scale, thickness)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            cv2.putText(
                img,
                apt_num,
                (text_x, text_y),
                font,
                text_scale,
                self.bg_color,
                thickness,
                lineType=LINE_TYPE,
            )

            total_area = self.rng.uniform(50.0, 200.0)
            m2_text = f"{total_area:.1f}".replace(".", ",") + " м²"
            m2_scale = base_scale * 0.9
            m2_size = cv2.getTextSize(m2_text, font, m2_scale, thickness)[0]
            tx = cx - m2_size[0] // 2
            ty = cy + radius + m2_size[1] + int(self.size * 0.008)
            cv2.putText(
                img,
                m2_text,
                (tx, ty),
                font,
                m2_scale,
                WALL_COLOR,
                thickness,
                lineType=LINE_TYPE,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генератор синтетических датасетов для распознавания дверей и окон на планах формата 'Брусника' - v1.2"
    )
    parser.add_argument("--output", type=str, default="brusnika_dataset", help="Каталог для датасета")
    parser.add_argument("--count", type=int, default=500, help="Количество генерируемых изображений")
    parser.add_argument("--seed", type=int, default=42, help="Сид случайных чисел")
    parser.add_argument("--size", type=int, default=1024, help="Размер изображений в пикселях")
    parser.add_argument("--min-wall-thickness", type=int, default=12, help="Минимальная толщина стены")
    parser.add_argument("--max-wall-thickness", type=int, default=30, help="Максимальная толщина стены")

    args = parser.parse_args()

    gen = BrusnikaFloorPlanGenerator(img_size=args.size, seed=args.seed)
    gen.generate_dataset(args.count, args.output, (args.min_wall_thickness, args.max_wall_thickness))

    print(f"Датасет 'Брусника' сгенерирован: {args.count} изображений")
    print(f"Каталог: {args.output}/")
    print(f"  - images/: изображения планов")
    print(f"  - labels/: YOLO разметка")
    print(f"Параметры:")
    print(f"  - Размер: {args.size}x{args.size}")
    print(f"  - Толщина стен: {args.min_wall_thickness}-{args.max_wall_thickness} пикселей")
    print(f"  - Классы: 0=door, 1=window")