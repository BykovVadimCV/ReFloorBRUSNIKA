"""
Улучшенный экспортёр планировок в формат OBJ
Версия: 2.0
Автор: Расширение для системы анализа планировок
Дата: 02.06.2026

Экспортирует планировки как 3D модели OBJ, совместимые с форматом Sweet Home 3D.
Решает критические проблемы, выявленные при сравнительном анализе:
- Y-up система координат (промышленный стандарт)
- Вершинные нормали для правильного освещения
- Плоскость пола для полной модели
- Раздельные группы граней для лучшего контроля материалов
- Корректное назначение материалов
"""

import os
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Point3D:
    """
    3D точка в системе координат Y-up

    Атрибуты:
        x: Координата X (горизонтальная ось)
        y: Координата Y (ВЫСОТА - вертикальная ось)
        z: Координата Z (глубина)
    """
    x: float
    y: float  # ВЫСОТА (вертикальная ось)
    z: float


@dataclass
class Normal3D:
    """
    3D вектор нормали

    Атрибуты:
        x: Компонента X нормали
        y: Компонента Y нормали
        z: Компонента Z нормали
    """
    x: float
    y: float
    z: float


class ImprovedOBJExporter:
    """
    Экспортёр OBJ, совместимый с Sweet Home 3D.
    Использует систему координат Y-up с корректными нормалями и группировкой.

    Основные возможности:
    - Правильная система координат (Y-вверх)
    - Вершинные нормали для реалистичного освещения
    - Опциональные плоскости пола и потолка
    - Группировка граней по функциональным частям стен
    - MTL материалы для визуального представления
    """

    def __init__(self,
                 pixels_to_cm: float = 2.54,
                 wall_height_cm: float = 243.84,
                 include_floor: bool = True,
                 include_ceiling: bool = False):
        """
        Инициализация улучшенного OBJ экспортёра.

        Параметры:
            pixels_to_cm: Коэффициент преобразования пикселей в сантиметры
            wall_height_cm: Высота стен в сантиметрах (по оси Y)
            include_floor: Включить плоскость пола
            include_ceiling: Включить плоскость потолка
        """
        self.pixels_to_cm = pixels_to_cm
        self.wall_height_cm = wall_height_cm
        self.include_floor = include_floor
        self.include_ceiling = include_ceiling

        # Счётчики элементов
        self.vertex_count = 0
        self.normal_count = 0

        # Хранилища данных
        self.vertices = []  # Список вершин
        self.normals = []   # Список нормалей
        self.faces = []     # Список граней

        # Стандартные нормали (система Y-up)
        self.NORMAL_UP = Normal3D(0.0, 1.0, 0.0)      # +Y (потолок/верх)
        self.NORMAL_DOWN = Normal3D(0.0, -1.0, 0.0)   # -Y (пол/низ)
        self.NORMAL_RIGHT = Normal3D(1.0, 0.0, 0.0)   # +X (право)
        self.NORMAL_LEFT = Normal3D(-1.0, 0.0, 0.0)   # -X (лево)
        self.NORMAL_FORWARD = Normal3D(0.0, 0.0, 1.0) # +Z (вперёд)
        self.NORMAL_BACK = Normal3D(0.0, 0.0, -1.0)   # -Z (назад)

    def export_from_segments(self,
                            segments: List,
                            output_path: str = "floorplan.obj") -> str:
        """
        Экспорт планировки из сегментов стен.

        Параметры:
            segments: Список объектов сегментов стен с атрибутами x1, y1, x2, y2, thickness_mean
            output_path: Путь к выходному файлу

        Возвращает:
            Путь к созданному OBJ файлу
        """
        self._reset()

        # Вычисление ограничивающей рамки для пола/потолка
        bounds = self._calculate_bounds(segments)

        # Добавление стандартных нормалей
        self._add_standard_normals()

        # Добавление пола при необходимости
        if self.include_floor and bounds:
            self._add_floor(bounds)

        # Добавление стен
        for i, seg in enumerate(segments):
            self._add_wall_segment(seg, i)

        # Добавление потолка при необходимости
        if self.include_ceiling and bounds:
            self._add_ceiling(bounds)

        # Запись файлов
        self._write_obj_file(output_path)
        self._write_mtl_file(output_path.replace('.obj', '.mtl'))

        return output_path

    def _reset(self):
        """Сброс всех счётчиков и данных"""
        self.vertex_count = 0
        self.normal_count = 0
        self.vertices = []
        self.normals = []
        self.faces = []

    def _calculate_bounds(self, segments: List) -> Optional[Tuple[float, float, float, float]]:
        """
        Вычисление ограничивающей рамки всех сегментов в сантиметрах

        Параметры:
            segments: Список сегментов стен

        Возвращает:
            Кортеж (min_x, min_z, max_x, max_z) в сантиметрах или None
        """
        if not segments:
            return None

        min_x = min_z = float('inf')
        max_x = max_z = float('-inf')

        for seg in segments:
            # Преобразование в сантиметры
            x1 = seg.x1 * self.pixels_to_cm
            z1 = seg.y1 * self.pixels_to_cm
            x2 = seg.x2 * self.pixels_to_cm
            z2 = seg.y2 * self.pixels_to_cm

            thickness = seg.thickness_mean * self.pixels_to_cm if hasattr(seg, 'thickness_mean') else 15.0

            # Расширение на толщину стен
            min_x = min(min_x, x1 - thickness, x2 - thickness)
            max_x = max(max_x, x1 + thickness, x2 + thickness)
            min_z = min(min_z, z1 - thickness, z2 - thickness)
            max_z = max(max_z, z1 + thickness, z2 + thickness)

        # Добавление отступа
        margin = 20.0  # 20 см отступ
        return (min_x - margin, min_z - margin, max_x + margin, max_z + margin)

    def _add_standard_normals(self):
        """Добавление 6 стандартных нормалей, выровненных по осям"""
        self.normals.append("# Стандартные нормали (система координат Y-up)")
        self._add_normal(self.NORMAL_UP)      # 1: вверх
        self._add_normal(self.NORMAL_DOWN)    # 2: вниз
        self._add_normal(self.NORMAL_RIGHT)   # 3: вправо
        self._add_normal(self.NORMAL_LEFT)    # 4: влево
        self._add_normal(self.NORMAL_FORWARD) # 5: вперёд
        self._add_normal(self.NORMAL_BACK)    # 6: назад
        self.normals.append("")

    def _add_normal(self, normal: Normal3D) -> int:
        """
        Добавление нормали и возврат её индекса

        Параметры:
            normal: Объект Normal3D

        Возвращает:
            Индекс добавленной нормали (1-based)
        """
        self.normals.append(f"vn {normal.x:.6f} {normal.y:.6f} {normal.z:.6f}")
        self.normal_count += 1
        return self.normal_count

    def _add_floor(self, bounds: Tuple[float, float, float, float]):
        """
        Добавление плоскости пола

        Параметры:
            bounds: Границы (min_x, min_z, max_x, max_z)
        """
        min_x, min_z, max_x, max_z = bounds

        self.faces.append("g ground_1")
        self.faces.append("usemtl ground_1")

        # Четыре угла на уровне Y=0
        v1 = self._add_vertex(Point3D(min_x, 0.0, min_z))
        v2 = self._add_vertex(Point3D(min_x, 0.0, max_z))
        v3 = self._add_vertex(Point3D(max_x, 0.0, max_z))
        v4 = self._add_vertex(Point3D(max_x, 0.0, min_z))

        # Два треугольника с нормалью ВВЕРХ (индекс 1)
        self.faces.append(f"f {v1}//1 {v2}//1 {v3}//1")
        self.faces.append(f"f {v1}//1 {v3}//1 {v4}//1")
        self.faces.append("")

    def _add_ceiling(self, bounds: Tuple[float, float, float, float]):
        """
        Добавление плоскости потолка

        Параметры:
            bounds: Границы (min_x, min_z, max_x, max_z)
        """
        min_x, min_z, max_x, max_z = bounds

        self.faces.append("g ceiling_1")
        self.faces.append("usemtl ceiling_1")

        # Четыре угла на уровне Y=wall_height
        v1 = self._add_vertex(Point3D(min_x, self.wall_height_cm, min_z))
        v2 = self._add_vertex(Point3D(min_x, self.wall_height_cm, max_z))
        v3 = self._add_vertex(Point3D(max_x, self.wall_height_cm, max_z))
        v4 = self._add_vertex(Point3D(max_x, self.wall_height_cm, min_z))

        # Два треугольника с нормалью ВНИЗ (индекс 2) - направлены вниз
        self.faces.append(f"f {v1}//2 {v4}//2 {v3}//2")
        self.faces.append(f"f {v1}//2 {v3}//2 {v2}//2")
        self.faces.append("")

    def _add_wall_segment(self, seg, index: int):
        """
        Добавление сегмента стены с корректной группировкой и нормалями.
        Создаёт отдельные группы для низа, сторон и верха стены.

        Параметры:
            seg: Объект сегмента стены с координатами и толщиной
            index: Индекс сегмента для именования групп
        """
        # Преобразование в см и координаты Y-up
        x1 = seg.x1 * self.pixels_to_cm
        z1 = seg.y1 * self.pixels_to_cm  # старая Y становится Z
        x2 = seg.x2 * self.pixels_to_cm
        z2 = seg.y2 * self.pixels_to_cm  # старая Y становится Z

        thickness = seg.thickness_mean * self.pixels_to_cm if hasattr(seg, 'thickness_mean') else 15.0

        # Вычисление перпендикулярного смещения для толщины
        dx = x2 - x1
        dz = z2 - z1
        length = math.sqrt(dx*dx + dz*dz)

        if length < 0.001:
            return

        # Нормализованный перпендикулярный вектор
        perp_x = -dz / length
        perp_z = dx / length

        # Смещение на половину толщины
        half_t = thickness / 2.0

        # Четыре угла внизу (Y=0)
        p1 = Point3D(x1 + perp_x * half_t, 0.0, z1 + perp_z * half_t)
        p2 = Point3D(x2 + perp_x * half_t, 0.0, z2 + perp_z * half_t)
        p3 = Point3D(x2 - perp_x * half_t, 0.0, z2 - perp_z * half_t)
        p4 = Point3D(x1 - perp_x * half_t, 0.0, z1 - perp_z * half_t)

        # Четыре угла вверху (Y=wall_height)
        p5 = Point3D(x1 + perp_x * half_t, self.wall_height_cm, z1 + perp_z * half_t)
        p6 = Point3D(x2 + perp_x * half_t, self.wall_height_cm, z2 + perp_z * half_t)
        p7 = Point3D(x2 - perp_x * half_t, self.wall_height_cm, z2 - perp_z * half_t)
        p8 = Point3D(x1 - perp_x * half_t, self.wall_height_cm, z1 - perp_z * half_t)

        # Добавление вершин
        v1 = self._add_vertex(p1)
        v2 = self._add_vertex(p2)
        v3 = self._add_vertex(p3)
        v4 = self._add_vertex(p4)
        v5 = self._add_vertex(p5)
        v6 = self._add_vertex(p6)
        v7 = self._add_vertex(p7)
        v8 = self._add_vertex(p8)

        # Вычисление пользовательских нормалей для наклонных стен
        # Нормаль для стороны 1 (p1-p2-p6-p5)
        side1_normal = self._calculate_face_normal(p1, p2, p6)
        n1 = self._add_normal(side1_normal)

        # Нормаль для стороны 2 (p3-p4-p8-p7) - противоположное направление
        side2_normal = Normal3D(-side1_normal.x, side1_normal.y, -side1_normal.z)
        n2 = self._add_normal(side2_normal)

        # Вычисление торцевых нормалей
        end1_dx = p4.x - p1.x
        end1_dz = p4.z - p1.z
        end1_len = math.sqrt(end1_dx*end1_dx + end1_dz*end1_dz)
        if end1_len > 0.001:
            end1_normal = Normal3D(-end1_dz/end1_len, 0.0, end1_dx/end1_len)
        else:
            end1_normal = self.NORMAL_BACK
        n3 = self._add_normal(end1_normal)

        end2_normal = Normal3D(-end1_normal.x, end1_normal.y, -end1_normal.z)
        n4 = self._add_normal(end2_normal)

        # ГРУППА 1: Нижняя грань
        self.faces.append(f"g wall_{index+1}_bottom")
        self.faces.append("usemtl wall_material")
        self.faces.append(f"f {v4}//2 {v3}//2 {v2}//2")
        self.faces.append(f"f {v4}//2 {v2}//2 {v1}//2")

        # ГРУППА 2: Сторона 1 (передняя грань)
        self.faces.append(f"g wall_{index+1}_side1")
        self.faces.append("usemtl wall_material")
        self.faces.append(f"f {v1}//n1 {v2}//n1 {v6}//n1 {v5}//n1".replace('n1', str(n1)))

        # ГРУППА 3: Сторона 2 (задняя грань)
        self.faces.append(f"g wall_{index+1}_side2")
        self.faces.append("usemtl wall_material")
        self.faces.append(f"f {v3}//n2 {v4}//n2 {v8}//n2 {v7}//n2".replace('n2', str(n2)))

        # ГРУППА 4: Торец 1
        self.faces.append(f"g wall_{index+1}_end1")
        self.faces.append("usemtl wall_material")
        self.faces.append(f"f {v4}//n3 {v1}//n3 {v5}//n3 {v8}//n3".replace('n3', str(n3)))

        # ГРУППА 5: Торец 2
        self.faces.append(f"g wall_{index+1}_end2")
        self.faces.append("usemtl wall_material")
        self.faces.append(f"f {v2}//n4 {v3}//n4 {v7}//n4 {v6}//n4".replace('n4', str(n4)))

        # ГРУППА 6: Верхняя грань
        self.faces.append(f"g wall_{index+1}_top")
        self.faces.append("usemtl wall_material")
        self.faces.append(f"f {v5}//1 {v6}//1 {v7}//1")
        self.faces.append(f"f {v5}//1 {v7}//1 {v8}//1")
        self.faces.append("")

    def _calculate_face_normal(self, p1: Point3D, p2: Point3D, p3: Point3D) -> Normal3D:
        """
        Вычисление вектора нормали для грани, определённой 3 точками (CCW порядок обхода)

        Параметры:
            p1, p2, p3: Три точки грани

        Возвращает:
            Нормализованный вектор нормали
        """
        # Векторы
        v1 = Point3D(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)
        v2 = Point3D(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z)

        # Векторное произведение
        nx = v1.y * v2.z - v1.z * v2.y
        ny = v1.z * v2.x - v1.x * v2.z
        nz = v1.x * v2.y - v1.y * v2.x

        # Нормализация
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length < 0.001:
            return Normal3D(0.0, 1.0, 0.0)

        return Normal3D(nx/length, ny/length, nz/length)

    def _add_vertex(self, point: Point3D) -> int:
        """
        Добавление вершины и возврат её индекса (1-based)

        Параметры:
            point: 3D точка

        Возвращает:
            Индекс вершины (начиная с 1)
        """
        self.vertices.append(f"v {point.x:.4f} {point.y:.4f} {point.z:.4f}")
        self.vertex_count += 1
        return self.vertex_count

    def _write_obj_file(self, output_path: str):
        """
        Запись OBJ файла

        Параметры:
            output_path: Путь для сохранения файла
        """
        if not output_path.endswith('.obj'):
            output_path += '.obj'

        mtl_name = os.path.basename(output_path).replace('.obj', '.mtl')

        with open(output_path, 'w') as f:
            # Заголовок
            f.write("# \n")
            f.write("# Экспорт планировки в OBJ - совместимый с Sweet Home 3D\n")
            f.write("# Сгенерировано системой анализа планировок v2.0\n")
            f.write("# Система координат Y-up (промышленный стандарт)\n")
            f.write("# \n")
            f.write(f"mtllib {mtl_name}\n\n")

            # Вершины
            for v in self.vertices:
                f.write(v + "\n")
            f.write("\n")

            # Нормали
            for n in self.normals:
                f.write(n + "\n")
            f.write("\n")

            # Грани
            for face in self.faces:
                f.write(face + "\n")

    def _write_mtl_file(self, mtl_path: str):
        """
        Запись файла материалов MTL, совместимого с Sweet Home 3D

        Параметры:
            mtl_path: Путь для сохранения файла материалов
        """
        with open(mtl_path, 'w') as f:
            f.write("# Библиотека материалов\n")
            f.write("# Совместимая с Sweet Home 3D\n\n")

            # Материал пола
            f.write("newmtl ground_1\n")
            f.write("Kd 0.75 0.75 0.75\n")
            f.write("Ks 0.0 0.0 0.0\n\n")

            # Материал стен
            f.write("newmtl wall_material\n")
            f.write("Kd 0.9 0.9 0.9\n")
            f.write("Ks 0.0 0.0 0.0\n\n")

            # Материал потолка
            f.write("newmtl ceiling_1\n")
            f.write("Kd 0.95 0.95 0.95\n")
            f.write("Ks 0.0 0.0 0.0\n\n")

            # Белый материал для будущих дверных/оконных рам
            f.write("newmtl white\n")
            f.write("Kd 1.0 1.0 1.0\n")
            f.write("Ks 0.3 0.3 0.3\n")