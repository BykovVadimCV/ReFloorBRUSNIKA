"""
SmartFloorPlanGenerator — Professional-Grade Configurable Procedural Engine
Version: 5.0
Author: Bykov Vadim Olegovich
Date: 2026-01-28
Architecture: Plugin-based strategy system with Shapely geometry, theme engine,
              constraint solver, vector export, semantic furniture placement, and multi-format output.

Changes v5.0:
  - Doors: no swing arcs, smaller opening angles (10-45 deg)
  - Windows: 5 visual sub-types (single/double/triple/french/slit), wider size range
  - Per-image color jitter within same style family
  - Two new layout strategies (asymmetric, courtyard)
  - More overall variety while keeping YOLO bbox accuracy

Dependencies:
    pip install opencv-python numpy shapely networkx pulp svgwrite ezdxf pydantic pyyaml scipy noise kiwisolver
"""
from __future__ import annotations

import os
import sys
import math
import json
import random
import argparse
import importlib
import pkgutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    List, Tuple, Optional, Dict, Set, Any, Union, Type, Callable, Sequence
)
from enum import Enum, auto
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict

import cv2
import numpy as np

# ── Geometry ──────────────────────────────────────────────────────────────────
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString,
    Point, box as shapely_box, GeometryCollection
)
from shapely.ops import unary_union, split, nearest_points
from shapely.affinity import scale, rotate, translate
import shapely.validation

# ── Graph / Solver ────────────────────────────────────────────────────────────
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import gaussian_filter

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

try:
    import kiwisolver
    HAS_KIWI = True
except ImportError:
    HAS_KIWI = False

# ── Vector / CAD export ──────────────────────────────────────────────────────
try:
    import svgwrite
    HAS_SVG = True
except ImportError:
    HAS_SVG = False

try:
    import ezdxf
    HAS_DXF = True
except ImportError:
    HAS_DXF = False

# ── Config ────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field, validator
import yaml

# ── Perlin / Simplex noise ────────────────────────────────────────────────────
try:
    from noise import pnoise2, snoise2
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & THEME SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class ExportFormat(str, Enum):
    PNG = "png"
    SVG = "svg"
    DXF = "dxf"
    JSON = "json"


class BuildingType(str, Enum):
    APARTMENT = "apartment"
    VILLA = "villa"
    STUDIO = "studio"
    OFFICE = "office"
    LOFT = "loft"
    TOWNHOUSE = "townhouse"


class StylePreset(str, Enum):
    MODERN_MINIMAL = "modern_minimal"
    CLASSIC_BLUEPRINT = "classic_blueprint"
    VINTAGE_SEPIA = "vintage_sepia"
    INDUSTRIAL_LOFT = "industrial_loft"
    SCANDINAVIAN = "scandinavian"
    JAPANESE_MINIMAL = "japanese_minimal"
    MEDITERRANEAN = "mediterranean"
    NIGHT_MODE = "night_mode"
    CUSTOM = "custom"


# ── Color helpers ─────────────────────────────────────────────────────────────
def hsl_to_bgr(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """Convert HSL (h 0-360, s/l 0-1) → BGR tuple for OpenCV."""
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (
        int((b + m) * 255),
        int((g + m) * 255),
        int((r + m) * 255),
    )


def hex_to_bgr(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _jitter_color(color: Tuple[int, int, int], rng: random.Random,
                   max_delta: int = 15) -> Tuple[int, int, int]:
    """Slightly randomize a BGR color within ±max_delta per channel."""
    return tuple(
        max(0, min(255, c + rng.randint(-max_delta, max_delta)))
        for c in color
    )


# ── Theme dictionary ─────────────────────────────────────────────────────────
@dataclass
class RoomStyle:
    fill: Tuple[int, int, int]
    stroke: Tuple[int, int, int]
    stroke_width: int = 1
    hatch_pattern: Optional[str] = None   # "diagonal", "cross", "dots", None
    hatch_spacing: int = 12
    hatch_color: Optional[Tuple[int, int, int]] = None
    texture: Optional[str] = None         # "wood", "tile", "concrete", None
    label_color: Optional[Tuple[int, int, int]] = None


@dataclass
class ThemeSpec:
    name: str
    background: Tuple[int, int, int] = (255, 255, 255)
    wall_color: Tuple[int, int, int] = (128, 135, 142)
    wall_ext_color: Tuple[int, int, int] = (80, 85, 90)
    furniture_fill: Tuple[int, int, int] = (221, 223, 226)
    furniture_stroke: Tuple[int, int, int] = (128, 135, 142)
    window_color: Tuple[int, int, int] = (128, 135, 142)
    door_color: Tuple[int, int, int] = (128, 135, 142)
    balcony_stripe: Tuple[int, int, int] = (0, 0, 255)
    label_color: Tuple[int, int, int] = (128, 135, 142)
    accent_public: Tuple[int, int, int] = (240, 248, 255)
    accent_private: Tuple[int, int, int] = (245, 240, 255)
    accent_service: Tuple[int, int, int] = (240, 255, 240)
    accent_circulation: Tuple[int, int, int] = (255, 255, 240)
    room_styles: Dict[str, RoomStyle] = field(default_factory=dict)
    shadow_offset: int = 0
    shadow_color: Tuple[int, int, int] = (200, 200, 200)
    line_type: int = cv2.LINE_AA

    def room_fill(self, room_type: str) -> Tuple[int, int, int]:
        if room_type in self.room_styles:
            return self.room_styles[room_type].fill
        return self.background

    def room_stroke(self, room_type: str) -> Tuple[int, int, int]:
        if room_type in self.room_styles:
            return self.room_styles[room_type].stroke
        return self.wall_color

    def jittered_copy(self, rng: random.Random, intensity: int = 12) -> "ThemeSpec":
        """Return a copy with all colors slightly randomized."""
        t = deepcopy(self)
        t.background = _jitter_color(t.background, rng, intensity // 2)
        t.wall_color = _jitter_color(t.wall_color, rng, intensity)
        t.wall_ext_color = _jitter_color(t.wall_ext_color, rng, intensity)
        t.furniture_fill = _jitter_color(t.furniture_fill, rng, intensity)
        t.furniture_stroke = _jitter_color(t.furniture_stroke, rng, intensity)
        t.window_color = _jitter_color(t.window_color, rng, intensity)
        t.door_color = _jitter_color(t.door_color, rng, intensity)
        t.label_color = _jitter_color(t.label_color, rng, intensity)
        t.accent_public = _jitter_color(t.accent_public, rng, intensity)
        t.accent_private = _jitter_color(t.accent_private, rng, intensity)
        t.accent_service = _jitter_color(t.accent_service, rng, intensity)
        t.accent_circulation = _jitter_color(t.accent_circulation, rng, intensity)
        for rt, rs in t.room_styles.items():
            rs.fill = _jitter_color(rs.fill, rng, intensity)
            rs.stroke = _jitter_color(rs.stroke, rng, intensity)
            if rs.hatch_color:
                rs.hatch_color = _jitter_color(rs.hatch_color, rng, intensity)
        return t


# ── Built-in themes ──────────────────────────────────────────────────────────
_THEMES: Dict[str, ThemeSpec] = {}


def _register_themes():
    # Modern minimal
    _THEMES["modern_minimal"] = ThemeSpec(
        name="modern_minimal",
        background=(255, 255, 255),
        wall_color=(160, 160, 160),
        wall_ext_color=(90, 90, 90),
        furniture_fill=(230, 230, 235),
        furniture_stroke=(160, 160, 160),
        accent_public=hsl_to_bgr(35, 0.15, 0.95),
        accent_private=hsl_to_bgr(220, 0.15, 0.95),
        accent_service=hsl_to_bgr(140, 0.12, 0.94),
        accent_circulation=hsl_to_bgr(60, 0.08, 0.96),
        room_styles={
            "living": RoomStyle(fill=hsl_to_bgr(35, 0.15, 0.95),
                                stroke=(160, 160, 160)),
            "kitchen": RoomStyle(fill=hsl_to_bgr(45, 0.12, 0.93),
                                 stroke=(160, 160, 160)),
            "bedroom": RoomStyle(fill=hsl_to_bgr(220, 0.15, 0.95),
                                 stroke=(160, 160, 160)),
            "bathroom": RoomStyle(fill=hsl_to_bgr(190, 0.18, 0.93),
                                  stroke=(160, 160, 160),
                                  hatch_pattern="dots", hatch_spacing=8,
                                  hatch_color=(180, 200, 210)),
            "corridor": RoomStyle(fill=hsl_to_bgr(60, 0.08, 0.96),
                                  stroke=(160, 160, 160)),
            "dining": RoomStyle(fill=hsl_to_bgr(25, 0.14, 0.94),
                                stroke=(160, 160, 160)),
            "home_office": RoomStyle(fill=hsl_to_bgr(200, 0.12, 0.94),
                                     stroke=(160, 160, 160)),
            "laundry": RoomStyle(fill=hsl_to_bgr(170, 0.10, 0.94),
                                 stroke=(160, 160, 160)),
            "walk_in_closet": RoomStyle(fill=hsl_to_bgr(280, 0.08, 0.95),
                                        stroke=(160, 160, 160)),
            "terrace": RoomStyle(fill=hsl_to_bgr(100, 0.15, 0.92),
                                 stroke=(160, 160, 160),
                                 hatch_pattern="diagonal", hatch_spacing=10,
                                 hatch_color=hsl_to_bgr(100, 0.10, 0.85)),
            "gym": RoomStyle(fill=hsl_to_bgr(0, 0.08, 0.95),
                             stroke=(160, 160, 160)),
            "atrium": RoomStyle(fill=hsl_to_bgr(120, 0.20, 0.90),
                                stroke=(160, 160, 160)),
            "winter_garden": RoomStyle(fill=hsl_to_bgr(110, 0.25, 0.88),
                                       stroke=(120, 160, 120)),
        },
    )

    # Classic blueprint
    _THEMES["classic_blueprint"] = ThemeSpec(
        name="classic_blueprint",
        background=(160, 80, 20),
        wall_color=(255, 255, 255),
        wall_ext_color=(255, 255, 255),
        furniture_fill=(180, 110, 40),
        furniture_stroke=(255, 255, 255),
        window_color=(255, 255, 255),
        door_color=(255, 255, 255),
        balcony_stripe=(255, 200, 100),
        label_color=(255, 255, 255),
        accent_public=(160, 80, 20),
        accent_private=(160, 80, 20),
        accent_service=(160, 80, 20),
        accent_circulation=(160, 80, 20),
        room_styles={
            rt: RoomStyle(fill=(160, 80, 20), stroke=(255, 255, 255))
            for rt in [
                "living", "kitchen", "bedroom", "bathroom", "corridor",
                "dining", "home_office", "laundry", "walk_in_closet",
                "terrace", "gym", "atrium", "winter_garden",
            ]
        },
    )

    # Vintage sepia
    _THEMES["vintage_sepia"] = ThemeSpec(
        name="vintage_sepia",
        background=(230, 225, 210),
        wall_color=(90, 75, 55),
        wall_ext_color=(60, 50, 35),
        furniture_fill=(210, 200, 180),
        furniture_stroke=(90, 75, 55),
        window_color=(90, 75, 55),
        door_color=(90, 75, 55),
        label_color=(90, 75, 55),
        balcony_stripe=(120, 100, 70),
        room_styles={
            rt: RoomStyle(fill=(230, 225, 210), stroke=(90, 75, 55),
                          hatch_pattern="cross" if rt == "bathroom" else None,
                          hatch_spacing=14,
                          hatch_color=(180, 170, 150))
            for rt in [
                "living", "kitchen", "bedroom", "bathroom", "corridor",
                "dining", "home_office", "laundry", "walk_in_closet",
                "terrace", "gym", "atrium", "winter_garden",
            ]
        },
    )

    # Industrial loft
    _THEMES["industrial_loft"] = ThemeSpec(
        name="industrial_loft",
        background=(235, 230, 225),
        wall_color=(70, 65, 60),
        wall_ext_color=(40, 38, 35),
        furniture_fill=(200, 195, 185),
        furniture_stroke=(70, 65, 60),
        label_color=(70, 65, 60),
        room_styles={
            "living": RoomStyle(fill=(235, 230, 225), stroke=(70, 65, 60),
                                texture="concrete"),
            "kitchen": RoomStyle(fill=(225, 220, 210), stroke=(70, 65, 60),
                                 texture="concrete"),
            "bedroom": RoomStyle(fill=(230, 225, 218), stroke=(70, 65, 60),
                                 texture="wood"),
            "bathroom": RoomStyle(fill=(220, 225, 230), stroke=(70, 65, 60),
                                  texture="tile"),
        },
    )

    # Scandinavian
    _THEMES["scandinavian"] = ThemeSpec(
        name="scandinavian",
        background=(252, 250, 248),
        wall_color=(180, 175, 168),
        wall_ext_color=(120, 115, 108),
        furniture_fill=(240, 235, 228),
        furniture_stroke=(180, 175, 168),
        room_styles={
            "living": RoomStyle(fill=hsl_to_bgr(40, 0.08, 0.97),
                                stroke=(180, 175, 168)),
            "bedroom": RoomStyle(fill=hsl_to_bgr(210, 0.06, 0.97),
                                 stroke=(180, 175, 168)),
            "kitchen": RoomStyle(fill=hsl_to_bgr(80, 0.06, 0.96),
                                 stroke=(180, 175, 168)),
        },
    )

    # Japanese minimal
    _THEMES["japanese_minimal"] = ThemeSpec(
        name="japanese_minimal",
        background=(245, 240, 230),
        wall_color=(100, 90, 70),
        wall_ext_color=(60, 55, 40),
        furniture_fill=(220, 210, 195),
        furniture_stroke=(100, 90, 70),
        room_styles={
            "living": RoomStyle(fill=(245, 240, 230), stroke=(100, 90, 70),
                                texture="wood"),
            "bedroom": RoomStyle(fill=(240, 235, 225), stroke=(100, 90, 70),
                                 texture="wood"),
        },
    )

    # Mediterranean
    _THEMES["mediterranean"] = ThemeSpec(
        name="mediterranean",
        background=(250, 248, 240),
        wall_color=(140, 120, 90),
        wall_ext_color=(100, 80, 55),
        furniture_fill=(235, 225, 210),
        furniture_stroke=(140, 120, 90),
        room_styles={
            "living": RoomStyle(fill=hsl_to_bgr(30, 0.20, 0.94),
                                stroke=(140, 120, 90)),
            "kitchen": RoomStyle(fill=hsl_to_bgr(200, 0.15, 0.92),
                                 stroke=(140, 120, 90),
                                 texture="tile"),
            "terrace": RoomStyle(fill=hsl_to_bgr(40, 0.25, 0.90),
                                 stroke=(140, 120, 90),
                                 texture="tile"),
        },
    )

    # Night mode
    _THEMES["night_mode"] = ThemeSpec(
        name="night_mode",
        background=(35, 35, 40),
        wall_color=(180, 180, 190),
        wall_ext_color=(200, 200, 210),
        furniture_fill=(55, 55, 65),
        furniture_stroke=(180, 180, 190),
        window_color=(180, 180, 190),
        door_color=(180, 180, 190),
        label_color=(200, 200, 210),
        balcony_stripe=(80, 80, 200),
        shadow_offset=3,
        shadow_color=(20, 20, 25),
        room_styles={
            rt: RoomStyle(fill=(35, 35, 40), stroke=(180, 180, 190))
            for rt in [
                "living", "kitchen", "bedroom", "bathroom", "corridor",
                "dining", "home_office", "laundry", "walk_in_closet",
                "terrace", "gym", "atrium", "winter_garden",
            ]
        },
    )


_register_themes()


# ── Pydantic config model ────────────────────────────────────────────────────
class FloorPlanConfig(BaseModel):
    """Master configuration — can be loaded from YAML / CLI / dict."""
    style: StylePreset = StylePreset.MODERN_MINIMAL
    building_type: BuildingType = BuildingType.APARTMENT
    max_rooms: int = Field(6, ge=1, le=20)
    min_rooms: int = Field(3, ge=1, le=10)
    allow_non_rect: bool = True
    palette_name: str = "modern_minimal"
    export_formats: List[ExportFormat] = [ExportFormat.PNG]
    img_size: int = Field(1024, ge=256, le=8192)
    super_sampling: int = Field(2, ge=1, le=4)
    seed: int = 42
    count: int = 10
    output_dir: str = "smart_dataset"
    multi_style: bool = False
    style_rotation_mode: str = "random"
    style_themes: Optional[List[str]] = None
    style_weights: Optional[List[float]] = None
    vary_strategies: bool = True
    vary_wall_thickness: bool = True
    vary_room_counts: bool = True
    vary_balcony: bool = True
    wall_thickness_range: Tuple[int, int] = (7, 30)
    balcony_probability: float = Field(0.7, ge=0.0, le=1.0)
    augmentation_probability: float = Field(0.3, ge=0.0, le=1.0)
    rotation_probability: float = Field(0.15, ge=0.0, le=1.0)
    cultural_preset: Optional[str] = None
    multi_story: bool = False
    num_floors: int = Field(1, ge=1, le=5)
    generate_metadata_json: bool = True
    constraint_solver: str = "proportional"
    strategies: List[str] = [
        "central", "linear", "radial", "bsp", "voronoi",
        "graph", "open_plan", "asymmetric", "courtyard"
    ]
    room_types_enabled: List[str] = [
        "living", "kitchen", "bedroom", "bathroom", "corridor",
        "dining", "home_office", "gym", "laundry",
        "walk_in_closet", "terrace", "winter_garden", "atrium",
    ]
    semantic_furniture: bool = True
    enable_augmentations: bool = Field(True, description="Enable visual corruptions/augmentations")
    color_jitter_intensity: int = Field(12, ge=0, le=40,
                                         description="Per-image color jitter within same style")

    aug_rotation_range: float = 5.0
    aug_scale_range: float = 0.1
    aug_brightness_range: float = 0.15
    aug_blur_prob: float = 0.4
    aug_noise_prob: float = 0.5
    aug_jpeg_prob: float = 0.5

    class Config:
        use_enum_values = True

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "FloorPlanConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_theme(self) -> ThemeSpec:
        name = self.palette_name
        if name in _THEMES:
            return deepcopy(_THEMES[name])
        return deepcopy(_THEMES.get("modern_minimal", ThemeSpec(name="fallback")))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ENUMERATIONS & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class Zone(Enum):
    PUBLIC = auto()
    PRIVATE = auto()
    SERVICE = auto()
    CIRCULATION = auto()


class RoomPriority(Enum):
    PRIMARY = 3
    SECONDARY = 2
    TERTIARY = 1


class Orientation(Enum):
    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270


GOLDEN_RATIO = 1.618
MIN_ROOM_ASPECT = 0.4
MAX_ROOM_ASPECT = 2.5


@dataclass
class RoomGeo:
    polygon: Polygon
    room_type: str
    zone: Zone = Zone.PUBLIC
    priority: RoomPriority = RoomPriority.SECONDARY
    needs_window: bool = True
    id: int = field(default_factory=lambda: random.randint(0, 10_000_000))
    floor: int = 0

    @property
    def x1(self) -> int:
        return int(self.polygon.bounds[0])

    @property
    def y1(self) -> int:
        return int(self.polygon.bounds[1])

    @property
    def x2(self) -> int:
        return int(self.polygon.bounds[2])

    @property
    def y2(self) -> int:
        return int(self.polygon.bounds[3])

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.polygon.area

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        c = self.polygon.centroid
        return int(c.x), int(c.y)

    @property
    def is_rectangular(self) -> bool:
        coords = list(self.polygon.exterior.coords)
        return len(coords) == 5

    def as_legacy_room(self) -> "Room":
        return Room(
            x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
            room_type=self.room_type, zone=self.zone,
            priority=self.priority, needs_window=self.needs_window,
            id=self.id,
        )


@dataclass
class Room:
    x1: int
    y1: int
    x2: int
    y2: int
    room_type: str
    zone: Zone = Zone.PUBLIC
    priority: RoomPriority = RoomPriority.SECONDARY
    needs_window: bool = True
    id: int = field(default_factory=lambda: random.randint(0, 100000))
    floor: int = 0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1, self.height)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def to_polygon(self) -> Polygon:
        return shapely_box(self.x1, self.y1, self.x2, self.y2)

    def to_geo(self) -> RoomGeo:
        return RoomGeo(
            polygon=self.to_polygon(),
            room_type=self.room_type,
            zone=self.zone,
            priority=self.priority,
            needs_window=self.needs_window,
            id=self.id,
            floor=self.floor,
        )


@dataclass
class Wall:
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: int
    is_external: bool
    is_guardrail: bool = False
    is_balcony_interface: bool = False

    def to_linestring(self) -> LineString:
        return LineString([(self.x1, self.y1), (self.x2, self.y2)])

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class Balcony:
    x1: int
    y1: int
    x2: int
    y2: int
    side: str
    archetype: str
    interface_wall: Optional[Wall] = None

    def to_polygon(self) -> Polygon:
        return shapely_box(
            min(self.x1, self.x2), min(self.y1, self.y2),
            max(self.x1, self.x2), max(self.y1, self.y2),
        )


@dataclass
class OccupiedRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    region_type: str


@dataclass
class ApartmentBrief:
    style: str
    num_rooms: int
    has_balcony: bool
    circulation_type: str
    light_priority: str
    building_type: str = "apartment"
    cultural_preset: Optional[str] = None
    floor: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GEOMETRY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class GeometryEngine:
    @staticmethod
    def make_rect(x1, y1, x2, y2) -> Polygon:
        return shapely_box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    @staticmethod
    def make_l_shape(outer, cutout) -> Polygon:
        base = shapely_box(*outer)
        cut = shapely_box(*cutout)
        result = base.difference(cut)
        if result.is_empty or not result.is_valid:
            return base
        return result

    @staticmethod
    def make_u_shape(outer, cutout) -> Polygon:
        return GeometryEngine.make_l_shape(outer, cutout)

    @staticmethod
    def subdivide_polygon(poly, direction, ratio):
        bx1, by1, bx2, by2 = poly.bounds
        if direction == "vertical":
            split_x = bx1 + (bx2 - bx1) * ratio
            line = LineString([(split_x, by1 - 1), (split_x, by2 + 1)])
        else:
            split_y = by1 + (by2 - by1) * ratio
            line = LineString([(bx1 - 1, split_y), (bx2 + 1, split_y)])
        parts = split(poly, line)
        geoms = [g for g in parts.geoms if isinstance(g, Polygon) and g.area > 1]
        if len(geoms) >= 2:
            return geoms[0], geoms[1]
        return poly, Polygon()

    @staticmethod
    def polygon_to_int_coords(poly):
        coords = list(poly.exterior.coords)
        return [(int(x), int(y)) for x, y in coords]

    @staticmethod
    def polygon_bounds_int(poly):
        b = poly.bounds
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])

    @staticmethod
    def rooms_overlap(r1, r2, tolerance=1.0):
        return r1.buffer(-tolerance).intersects(r2.buffer(-tolerance))

    @staticmethod
    def clip_to_boundary(room, boundary):
        return room.intersection(boundary)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CONSTRAINT SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class ConstraintSolver:
    ROOM_CONFIGS: Dict[str, Dict[str, Any]] = {
        "living": {
            "zone": Zone.PUBLIC, "priority": RoomPriority.PRIMARY,
            "min_area_ratio": 0.20, "max_area_ratio": 0.40,
            "aspect_range": (0.7, 1.5), "needs_window": True, "margin_factor": 1.2,
        },
        "kitchen": {
            "zone": Zone.SERVICE, "priority": RoomPriority.SECONDARY,
            "min_area_ratio": 0.10, "max_area_ratio": 0.22,
            "aspect_range": (0.6, 1.8), "needs_window": True, "margin_factor": 1.0,
        },
        "bedroom": {
            "zone": Zone.PRIVATE, "priority": RoomPriority.PRIMARY,
            "min_area_ratio": 0.12, "max_area_ratio": 0.28,
            "aspect_range": (0.7, 1.4), "needs_window": True, "margin_factor": 1.1,
        },
        "bathroom": {
            "zone": Zone.SERVICE, "priority": RoomPriority.TERTIARY,
            "min_area_ratio": 0.05, "max_area_ratio": 0.12,
            "aspect_range": (0.5, 2.0), "needs_window": False, "margin_factor": 0.8,
        },
        "corridor": {
            "zone": Zone.CIRCULATION, "priority": RoomPriority.TERTIARY,
            "min_area_ratio": 0.06, "max_area_ratio": 0.18,
            "aspect_range": (0.2, 5.0), "needs_window": False, "margin_factor": 0.6,
        },
        "dining": {
            "zone": Zone.PUBLIC, "priority": RoomPriority.SECONDARY,
            "min_area_ratio": 0.08, "max_area_ratio": 0.18,
            "aspect_range": (0.7, 1.5), "needs_window": True, "margin_factor": 1.0,
        },
        "home_office": {
            "zone": Zone.PRIVATE, "priority": RoomPriority.SECONDARY,
            "min_area_ratio": 0.06, "max_area_ratio": 0.15,
            "aspect_range": (0.6, 1.6), "needs_window": True, "margin_factor": 1.0,
        },
        "gym": {
            "zone": Zone.PRIVATE, "priority": RoomPriority.TERTIARY,
            "min_area_ratio": 0.06, "max_area_ratio": 0.14,
            "aspect_range": (0.7, 1.5), "needs_window": True, "margin_factor": 1.0,
        },
        "laundry": {
            "zone": Zone.SERVICE, "priority": RoomPriority.TERTIARY,
            "min_area_ratio": 0.03, "max_area_ratio": 0.08,
            "aspect_range": (0.5, 2.0), "needs_window": False, "margin_factor": 0.7,
        },
        "walk_in_closet": {
            "zone": Zone.PRIVATE, "priority": RoomPriority.TERTIARY,
            "min_area_ratio": 0.03, "max_area_ratio": 0.08,
            "aspect_range": (0.6, 1.6), "needs_window": False, "margin_factor": 0.7,
        },
        "terrace": {
            "zone": Zone.PUBLIC, "priority": RoomPriority.TERTIARY,
            "min_area_ratio": 0.05, "max_area_ratio": 0.15,
            "aspect_range": (0.4, 3.0), "needs_window": False, "margin_factor": 0.8,
        },
        "winter_garden": {
            "zone": Zone.PUBLIC, "priority": RoomPriority.SECONDARY,
            "min_area_ratio": 0.05, "max_area_ratio": 0.12,
            "aspect_range": (0.7, 1.5), "needs_window": True, "margin_factor": 1.1,
        },
        "atrium": {
            "zone": Zone.PUBLIC, "priority": RoomPriority.PRIMARY,
            "min_area_ratio": 0.10, "max_area_ratio": 0.25,
            "aspect_range": (0.8, 1.3), "needs_window": True, "margin_factor": 1.3,
        },
    }

    ADJACENCY_MATRIX = {
        ("living", "kitchen"): 0.9,
        ("living", "dining"): 0.95,
        ("living", "corridor"): 0.7,
        ("living", "bedroom"): 0.3,
        ("living", "bathroom"): -0.5,
        ("living", "terrace"): 0.8,
        ("living", "winter_garden"): 0.7,
        ("kitchen", "dining"): 0.9,
        ("kitchen", "bathroom"): 0.4,
        ("kitchen", "laundry"): 0.6,
        ("bedroom", "bathroom"): 0.6,
        ("bedroom", "walk_in_closet"): 0.85,
        ("bedroom", "corridor"): 0.5,
        ("corridor", "bathroom"): 0.8,
        ("corridor", "kitchen"): 0.6,
        ("home_office", "living"): 0.5,
        ("home_office", "bedroom"): 0.4,
        ("gym", "bathroom"): 0.5,
        ("atrium", "living"): 0.8,
        ("atrium", "corridor"): 0.7,
    }

    def __init__(self, rng, method="proportional"):
        self.rng = rng
        self.method = method

    def allocate_areas(self, room_types, total_area, brief):
        if self.method == "pulp" and HAS_PULP:
            return self._solve_pulp(room_types, total_area, brief)
        return self._solve_proportional(room_types, total_area, brief)

    def _solve_proportional(self, room_types, total_area, brief):
        weights = {}
        for rt in room_types:
            cfg = self.ROOM_CONFIGS.get(rt, self.ROOM_CONFIGS["corridor"])
            mid = (cfg["min_area_ratio"] + cfg["max_area_ratio"]) / 2
            jitter = self.rng.uniform(0.85, 1.15)
            weights[rt] = mid * cfg["margin_factor"] * jitter
        if brief.style == "compact":
            for rt in weights:
                if self.ROOM_CONFIGS.get(rt, {}).get("zone") == Zone.CIRCULATION:
                    weights[rt] *= 0.7
        elif brief.style == "spacious":
            for rt in weights:
                if self.ROOM_CONFIGS.get(rt, {}).get("zone") == Zone.PUBLIC:
                    weights[rt] *= 1.2
        total_w = sum(weights.values())
        return {rt: (w / total_w) * total_area for rt, w in weights.items()}

    def _solve_pulp(self, room_types, total_area, brief):
        prob = pulp.LpProblem("RoomAllocation", pulp.LpMinimize)
        vars_ = {}
        for rt in room_types:
            cfg = self.ROOM_CONFIGS.get(rt, self.ROOM_CONFIGS["corridor"])
            lo = cfg["min_area_ratio"] * total_area
            hi = cfg["max_area_ratio"] * total_area
            vars_[rt] = pulp.LpVariable(rt, lowBound=lo, upBound=hi)
        prob += pulp.lpSum(vars_.values()) == total_area
        devs = []
        for rt in room_types:
            cfg = self.ROOM_CONFIGS.get(rt, self.ROOM_CONFIGS["corridor"])
            mid = ((cfg["min_area_ratio"] + cfg["max_area_ratio"]) / 2) * total_area
            d = pulp.LpVariable(f"dev_{rt}", lowBound=0)
            prob += vars_[rt] - mid <= d
            prob += mid - vars_[rt] <= d
            devs.append(d)
        prob += pulp.lpSum(devs)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        result = {}
        for rt in room_types:
            val = vars_[rt].varValue
            result[rt] = val if val is not None else total_area / len(room_types)
        return result

    def get_adjacency_score(self, r1, r2):
        key = (r1, r2) if (r1, r2) in self.ADJACENCY_MATRIX else (r2, r1)
        return self.ADJACENCY_MATRIX.get(key, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LAYOUT STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

class LayoutStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        ...


_STRATEGY_REGISTRY: Dict[str, Type[LayoutStrategy]] = {}


def register_strategy(cls):
    _STRATEGY_REGISTRY[cls.name] = cls
    return cls


# ── 5a. Central corridor ─────────────────────────────────────────────────────
@register_strategy
class CentralCorridorStrategy(LayoutStrategy):
    name = "central"

    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = solver.rng
        walls, rooms = [], []

        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        is_horizontal = width > height * 1.2

        available = list(config.room_types_enabled)
        essentials = ["corridor", "bathroom"]
        primary_choices = [r for r in available if r not in essentials and
                           solver.ROOM_CONFIGS.get(r, {}).get("zone") in (Zone.PUBLIC, Zone.PRIVATE)]
        service_choices = [r for r in available if r not in essentials and
                           solver.ROOM_CONFIGS.get(r, {}).get("zone") == Zone.SERVICE
                           and r != "bathroom"]

        if is_horizontal:
            corridor_height = int(height * rng.uniform(0.15, 0.22))
            corridor_y = top + int(height * rng.uniform(0.35, 0.50))

            walls.append(Wall(left, corridor_y, right, corridor_y, int_th, False))
            walls.append(Wall(left, corridor_y + corridor_height, right,
                              corridor_y + corridor_height, int_th, False))
            rooms.append(Room(left, corridor_y, right, corridor_y + corridor_height,
                              "corridor", Zone.CIRCULATION, RoomPriority.TERTIARY, False))

            kitchen_width = int(width * rng.uniform(0.35, 0.45))
            walls.append(Wall(left + kitchen_width, top, left + kitchen_width, corridor_y, int_th, False))

            top_left_type = rng.choice(service_choices) if service_choices else "kitchen"
            top_right_type = rng.choice(primary_choices) if primary_choices else "living"

            rooms.append(Room(left, top, left + kitchen_width, corridor_y,
                              top_left_type,
                              solver.ROOM_CONFIGS.get(top_left_type, {}).get("zone", Zone.SERVICE),
                              solver.ROOM_CONFIGS.get(top_left_type, {}).get("priority", RoomPriority.SECONDARY)))
            rooms.append(Room(left + kitchen_width, top, right, corridor_y,
                              top_right_type,
                              solver.ROOM_CONFIGS.get(top_right_type, {}).get("zone", Zone.PUBLIC),
                              solver.ROOM_CONFIGS.get(top_right_type, {}).get("priority", RoomPriority.PRIMARY)))

            bottom_start = corridor_y + corridor_height
            bathroom_width = int(width * rng.uniform(0.20, 0.30))
            walls.append(Wall(left + bathroom_width, bottom_start,
                              left + bathroom_width, bottom, int_th, False))

            rooms.append(Room(left, bottom_start, left + bathroom_width, bottom,
                              "bathroom", Zone.SERVICE, RoomPriority.TERTIARY, False))

            remaining = width - bathroom_width
            if brief.num_rooms >= 2 and remaining > width * 0.4:
                bedroom_split = left + bathroom_width + int(remaining * 0.5)
                walls.append(Wall(bedroom_split, bottom_start, bedroom_split, bottom, int_th, False))

                bot_left = rng.choice([r for r in primary_choices if r != top_right_type] or ["bedroom"])
                bot_right = rng.choice(primary_choices or ["living"])

                rooms.append(Room(left + bathroom_width, bottom_start, bedroom_split, bottom,
                                  bot_left,
                                  solver.ROOM_CONFIGS.get(bot_left, {}).get("zone", Zone.PRIVATE),
                                  solver.ROOM_CONFIGS.get(bot_left, {}).get("priority", RoomPriority.PRIMARY)))
                rooms.append(Room(bedroom_split, bottom_start, right, bottom,
                                  bot_right,
                                  solver.ROOM_CONFIGS.get(bot_right, {}).get("zone", Zone.PUBLIC),
                                  solver.ROOM_CONFIGS.get(bot_right, {}).get("priority", RoomPriority.SECONDARY)))
            else:
                rooms.append(Room(left + bathroom_width, bottom_start, right, bottom,
                                  "bedroom", Zone.PRIVATE, RoomPriority.PRIMARY))
        else:
            corridor_width = int(width * rng.uniform(0.15, 0.22))
            corridor_x = left + int(width * rng.uniform(0.35, 0.50))

            walls.append(Wall(corridor_x, top, corridor_x, bottom, int_th, False))
            walls.append(Wall(corridor_x + corridor_width, top,
                              corridor_x + corridor_width, bottom, int_th, False))
            rooms.append(Room(corridor_x, top, corridor_x + corridor_width, bottom,
                              "corridor", Zone.CIRCULATION, RoomPriority.TERTIARY, False))

            bathroom_height = int(height * rng.uniform(0.22, 0.32))
            walls.append(Wall(left, top + bathroom_height, corridor_x,
                              top + bathroom_height, int_th, False))
            rooms.append(Room(left, top, corridor_x, top + bathroom_height,
                              "bathroom", Zone.SERVICE, RoomPriority.TERTIARY, False))

            left_bottom_type = rng.choice(service_choices or ["kitchen"])
            rooms.append(Room(left, top + bathroom_height, corridor_x, bottom,
                              left_bottom_type,
                              solver.ROOM_CONFIGS.get(left_bottom_type, {}).get("zone", Zone.SERVICE),
                              solver.ROOM_CONFIGS.get(left_bottom_type, {}).get("priority", RoomPriority.SECONDARY)))

            right_start = corridor_x + corridor_width
            living_height = int(height * rng.uniform(0.45, 0.60))
            walls.append(Wall(right_start, top + living_height, right,
                              top + living_height, int_th, False))

            rt_top = rng.choice(primary_choices or ["living"])
            rt_bot = rng.choice([r for r in primary_choices if r != rt_top] or ["bedroom"])

            rooms.append(Room(right_start, top, right, top + living_height,
                              rt_top,
                              solver.ROOM_CONFIGS.get(rt_top, {}).get("zone", Zone.PUBLIC),
                              solver.ROOM_CONFIGS.get(rt_top, {}).get("priority", RoomPriority.PRIMARY)))
            rooms.append(Room(right_start, top + living_height, right, bottom,
                              rt_bot,
                              solver.ROOM_CONFIGS.get(rt_bot, {}).get("zone", Zone.PRIVATE),
                              solver.ROOM_CONFIGS.get(rt_bot, {}).get("priority", RoomPriority.PRIMARY)))

        return walls, rooms


# ── 5b. Linear flow ──────────────────────────────────────────────────────────
@register_strategy
class LinearFlowStrategy(LayoutStrategy):
    name = "linear"

    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = solver.rng
        walls, rooms = [], []

        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        is_horizontal = width > height
        zone_splits = self._calculate_splits(rng, brief)

        if is_horizontal:
            current_x = left
            zone_types = self._pick_zone_types(rng, config, brief)
            for i, (ratio, zone_info) in enumerate(zip(zone_splits, zone_types)):
                z_width = int(width * ratio)
                next_x = min(current_x + z_width, right)
                if i < len(zone_splits) - 1:
                    walls.append(Wall(next_x, top, next_x, bottom, int_th, False))
                if isinstance(zone_info, list):
                    split_y = top + int(height * rng.uniform(0.35, 0.55))
                    walls.append(Wall(current_x, split_y, next_x, split_y, int_th, False))
                    rooms.append(Room(current_x, top, next_x, split_y,
                                      zone_info[0],
                                      solver.ROOM_CONFIGS.get(zone_info[0], {}).get("zone", Zone.SERVICE),
                                      solver.ROOM_CONFIGS.get(zone_info[0], {}).get("priority", RoomPriority.TERTIARY),
                                      solver.ROOM_CONFIGS.get(zone_info[0], {}).get("needs_window", False)))
                    rooms.append(Room(current_x, split_y, next_x, bottom,
                                      zone_info[1],
                                      solver.ROOM_CONFIGS.get(zone_info[1], {}).get("zone", Zone.CIRCULATION),
                                      solver.ROOM_CONFIGS.get(zone_info[1], {}).get("priority", RoomPriority.TERTIARY),
                                      solver.ROOM_CONFIGS.get(zone_info[1], {}).get("needs_window", False)))
                else:
                    cfg = solver.ROOM_CONFIGS.get(zone_info, {})
                    rooms.append(Room(current_x, top, next_x, bottom,
                                      zone_info,
                                      cfg.get("zone", Zone.PUBLIC),
                                      cfg.get("priority", RoomPriority.PRIMARY),
                                      cfg.get("needs_window", True)))
                current_x = next_x
        else:
            current_y = top
            zone_types = self._pick_zone_types(rng, config, brief)
            for i, (ratio, zone_info) in enumerate(zip(zone_splits, zone_types)):
                z_height = int(height * ratio)
                next_y = min(current_y + z_height, bottom)
                if i < len(zone_splits) - 1:
                    walls.append(Wall(left, next_y, right, next_y, int_th, False))
                if isinstance(zone_info, list):
                    split_x = left + int(width * rng.uniform(0.40, 0.60))
                    walls.append(Wall(split_x, current_y, split_x, next_y, int_th, False))
                    for j, rt in enumerate(zone_info):
                        cfg = solver.ROOM_CONFIGS.get(rt, {})
                        if j == 0:
                            rooms.append(Room(left, current_y, split_x, next_y,
                                              rt, cfg.get("zone", Zone.SERVICE),
                                              cfg.get("priority", RoomPriority.SECONDARY),
                                              cfg.get("needs_window", True)))
                        else:
                            rooms.append(Room(split_x, current_y, right, next_y,
                                              rt, cfg.get("zone", Zone.CIRCULATION),
                                              cfg.get("priority", RoomPriority.TERTIARY),
                                              cfg.get("needs_window", False)))
                else:
                    cfg = solver.ROOM_CONFIGS.get(zone_info, {})
                    rooms.append(Room(left, current_y, right, next_y,
                                      zone_info, cfg.get("zone", Zone.PUBLIC),
                                      cfg.get("priority", RoomPriority.PRIMARY),
                                      cfg.get("needs_window", True)))
                current_y = next_y

        return walls, rooms

    def _calculate_splits(self, rng, brief):
        if brief.style == "compact":
            base = [0.18, 0.22, 0.35, 0.25]
        elif brief.style == "spacious":
            base = [0.15, 0.20, 0.40, 0.25]
        else:
            base = [0.17, 0.21, 0.37, 0.25]
        result = [v * rng.uniform(0.9, 1.1) for v in base]
        total = sum(result)
        return [v / total for v in result]

    def _pick_zone_types(self, rng, config, brief):
        available = list(config.room_types_enabled)
        types = []
        bath = "bathroom" if "bathroom" in available else rng.choice(available)
        corr = "corridor" if "corridor" in available else rng.choice(available)
        types.append([bath, corr])
        service_opts = [r for r in available if
                        ConstraintSolver.ROOM_CONFIGS.get(r, {}).get("zone") == Zone.SERVICE
                        and r not in ("bathroom",)]
        types.append(rng.choice(service_opts) if service_opts else "kitchen")
        public_opts = [r for r in available if
                       ConstraintSolver.ROOM_CONFIGS.get(r, {}).get("zone") == Zone.PUBLIC]
        types.append(rng.choice(public_opts) if public_opts else "living")
        private_opts = [r for r in available if
                        ConstraintSolver.ROOM_CONFIGS.get(r, {}).get("zone") == Zone.PRIVATE]
        types.append(rng.choice(private_opts) if private_opts else "bedroom")
        return types


# ── 5c. Radial hub ───────────────────────────────────────────────────────────
@register_strategy
class RadialHubStrategy(LayoutStrategy):
    name = "radial"

    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = solver.rng
        walls, rooms = [], []

        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        margin_x = int(width * rng.uniform(0.25, 0.35))
        margin_y = int(height * rng.uniform(0.25, 0.35))
        living_left = left + margin_x
        living_top = top + margin_y

        walls.append(Wall(left, living_top, right, living_top, int_th, False))

        available = list(config.room_types_enabled)
        service_types = [r for r in available if
                         ConstraintSolver.ROOM_CONFIGS.get(r, {}).get("zone") == Zone.SERVICE]
        circ_types = [r for r in available if
                      ConstraintSolver.ROOM_CONFIGS.get(r, {}).get("zone") == Zone.CIRCULATION]

        top_split = left + int(width * rng.uniform(0.35, 0.50))
        walls.append(Wall(top_split, top, top_split, living_top, int_th, False))

        r1 = rng.choice(service_types) if service_types else "bathroom"
        r2 = rng.choice(circ_types) if circ_types else "corridor"

        rooms.append(Room(left, top, top_split, living_top, r1,
                          solver.ROOM_CONFIGS.get(r1, {}).get("zone", Zone.SERVICE),
                          solver.ROOM_CONFIGS.get(r1, {}).get("priority", RoomPriority.TERTIARY),
                          solver.ROOM_CONFIGS.get(r1, {}).get("needs_window", False)))
        rooms.append(Room(top_split, top, right, living_top, r2,
                          solver.ROOM_CONFIGS.get(r2, {}).get("zone", Zone.CIRCULATION),
                          solver.ROOM_CONFIGS.get(r2, {}).get("priority", RoomPriority.TERTIARY),
                          solver.ROOM_CONFIGS.get(r2, {}).get("needs_window", False)))

        walls.append(Wall(living_left, living_top, living_left, bottom, int_th, False))
        left_split = living_top + int((bottom - living_top) * rng.uniform(0.40, 0.55))
        walls.append(Wall(left, left_split, living_left, left_split, int_th, False))

        r3 = rng.choice([r for r in service_types if r != r1] or ["kitchen"])
        r4 = rng.choice([r for r in available if
                         ConstraintSolver.ROOM_CONFIGS.get(r, {}).get("zone") == Zone.PRIVATE] or ["bedroom"])

        rooms.append(Room(left, living_top, living_left, left_split, r3,
                          solver.ROOM_CONFIGS.get(r3, {}).get("zone", Zone.SERVICE),
                          solver.ROOM_CONFIGS.get(r3, {}).get("priority", RoomPriority.SECONDARY)))
        rooms.append(Room(left, left_split, living_left, bottom, r4,
                          solver.ROOM_CONFIGS.get(r4, {}).get("zone", Zone.PRIVATE),
                          solver.ROOM_CONFIGS.get(r4, {}).get("priority", RoomPriority.PRIMARY)))

        rooms.append(Room(living_left, living_top, right, bottom,
                          "living", Zone.PUBLIC, RoomPriority.PRIMARY))

        return walls, rooms


# ── 5d. BSP Recursive ────────────────────────────────────────────────────────
@register_strategy
class BSPRecursiveStrategy(LayoutStrategy):
    name = "bsp"
    MIN_CELL_SIZE = 0.08

    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        left, top, right, bottom = bounds
        rng = solver.rng
        walls, rooms = [], []

        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        cells = self._bsp_split(left, top, right, bottom, rng,
                                max_depth=brief.num_rooms + 1,
                                min_size=int((right - left) * self.MIN_CELL_SIZE))

        available = list(config.room_types_enabled)
        needed = min(len(cells), brief.num_rooms + 2)
        cells.sort(key=lambda c: (c[2] - c[0]) * (c[3] - c[1]), reverse=True)

        room_assignments = self._assign_rooms(cells[:needed], available, solver, rng)

        for (cx1, cy1, cx2, cy2), rt in room_assignments:
            cfg = solver.ROOM_CONFIGS.get(rt, solver.ROOM_CONFIGS.get("corridor", {}))
            rooms.append(Room(cx1, cy1, cx2, cy2, rt,
                              cfg.get("zone", Zone.PUBLIC),
                              cfg.get("priority", RoomPriority.SECONDARY),
                              cfg.get("needs_window", True)))

        for i, (cx1, cy1, cx2, cy2) in enumerate(cells[:needed]):
            for j, (dx1, dy1, dx2, dy2) in enumerate(cells[:needed]):
                if i == j:
                    continue
                if abs(cx2 - dx1) < 3 and max(cy1, dy1) < min(cy2, dy2):
                    walls.append(Wall(cx2, max(cy1, dy1), cx2, min(cy2, dy2), int_th, False))
                if abs(cy2 - dy1) < 3 and max(cx1, dx1) < min(cx2, dx2):
                    walls.append(Wall(max(cx1, dx1), cy2, min(cx2, dx2), cy2, int_th, False))

        return walls, rooms

    def _bsp_split(self, x1, y1, x2, y2, rng, max_depth=4, min_size=80, depth=0):
        w, h = x2 - x1, y2 - y1
        if depth >= max_depth or w < min_size * 2 or h < min_size * 2:
            return [(x1, y1, x2, y2)]
        if w > h * 1.3:
            direction = "vertical"
        elif h > w * 1.3:
            direction = "horizontal"
        else:
            direction = rng.choice(["vertical", "horizontal"])
        ratio = rng.uniform(0.35, 0.65)
        if direction == "vertical":
            s = x1 + int(w * ratio)
            if s - x1 < min_size or x2 - s < min_size:
                return [(x1, y1, x2, y2)]
            return (self._bsp_split(x1, y1, s, y2, rng, max_depth, min_size, depth + 1) +
                    self._bsp_split(s, y1, x2, y2, rng, max_depth, min_size, depth + 1))
        else:
            s = y1 + int(h * ratio)
            if s - y1 < min_size or y2 - s < min_size:
                return [(x1, y1, x2, y2)]
            return (self._bsp_split(x1, y1, x2, s, rng, max_depth, min_size, depth + 1) +
                    self._bsp_split(x1, s, x2, y2, rng, max_depth, min_size, depth + 1))

    def _assign_rooms(self, cells, available, solver, rng):
        assignments = []
        priority_order = ["living", "bedroom", "kitchen", "bathroom", "corridor"]
        priority_order = [r for r in priority_order if r in available]
        extras = [r for r in available if r not in priority_order]
        rng.shuffle(extras)
        full_order = priority_order + extras
        for i, cell in enumerate(cells):
            rt = full_order[i] if i < len(full_order) else rng.choice(available)
            assignments.append((cell, rt))
        return assignments


# ── 5e. Voronoi ──────────────────────────────────────────────────────────────
@register_strategy
class VoronoiZoneStrategy(LayoutStrategy):
    name = "voronoi"

    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = solver.rng
        walls, rooms = [], []

        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        n_points = min(brief.num_rooms + 2, 8)
        points = []
        for _ in range(n_points):
            px = rng.uniform(left + width * 0.1, right - width * 0.1)
            py = rng.uniform(top + height * 0.1, bottom - height * 0.1)
            points.append([px, py])
        points = np.array(points)

        mirrored = np.vstack([
            points,
            np.column_stack([2 * left - points[:, 0], points[:, 1]]),
            np.column_stack([2 * right - points[:, 0], points[:, 1]]),
            np.column_stack([points[:, 0], 2 * top - points[:, 1]]),
            np.column_stack([points[:, 0], 2 * bottom - points[:, 1]]),
        ])

        try:
            vor = Voronoi(mirrored)
        except Exception:
            return BSPRecursiveStrategy().generate(bounds, ext_th, int_th, solver, brief, config)

        boundary = shapely_box(left, top, right, bottom)
        available = list(config.room_types_enabled)
        room_order = ["living", "kitchen", "bedroom", "bathroom", "corridor"]
        room_order = [r for r in room_order if r in available]
        extras = [r for r in available if r not in room_order]
        rng.shuffle(extras)
        room_order += extras

        cell_idx = 0
        for i in range(n_points):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 in region or len(region) < 3:
                continue
            vertices = [vor.vertices[v] for v in region]
            try:
                poly = Polygon(vertices)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                clipped = poly.intersection(boundary)
                if clipped.is_empty or clipped.area < 100:
                    continue
            except Exception:
                continue
            bx1, by1, bx2, by2 = [int(v) for v in clipped.bounds]
            bx1, by1 = max(bx1, left), max(by1, top)
            bx2, by2 = min(bx2, right), min(by2, bottom)
            rt = room_order[cell_idx % len(room_order)]
            cfg = solver.ROOM_CONFIGS.get(rt, {})
            rooms.append(Room(bx1, by1, bx2, by2, rt,
                              cfg.get("zone", Zone.PUBLIC),
                              cfg.get("priority", RoomPriority.SECONDARY),
                              cfg.get("needs_window", True)))
            cell_idx += 1

        for i, r1 in enumerate(rooms):
            for j, r2 in enumerate(rooms):
                if i >= j:
                    continue
                if abs(r1.x2 - r2.x1) < 5:
                    y_start = max(r1.y1, r2.y1)
                    y_end = min(r1.y2, r2.y2)
                    if y_end > y_start:
                        walls.append(Wall(r1.x2, y_start, r1.x2, y_end, int_th, False))
                if abs(r1.y2 - r2.y1) < 5:
                    x_start = max(r1.x1, r2.x1)
                    x_end = min(r1.x2, r2.x2)
                    if x_end > x_start:
                        walls.append(Wall(x_start, r1.y2, x_end, r1.y2, int_th, False))

        if not rooms:
            return BSPRecursiveStrategy().generate(bounds, ext_th, int_th, solver, brief, config)

        return walls, rooms


# ── 5f. Graph-based ──────────────────────────────────────────────────────────
@register_strategy
class GraphBasedStrategy(LayoutStrategy):
    name = "graph"

    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = solver.rng
        walls, rooms = [], []

        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        G = nx.Graph()
        available = list(config.room_types_enabled)
        n_rooms = min(brief.num_rooms + 2, len(available), 8)

        selected = []
        for rt in ["living", "corridor", "bathroom"]:
            if rt in available:
                selected.append(rt)
        remaining = [r for r in available if r not in selected]
        rng.shuffle(remaining)
        while len(selected) < n_rooms and remaining:
            selected.append(remaining.pop())

        for rt in selected:
            G.add_node(rt)

        for i, r1 in enumerate(selected):
            for j, r2 in enumerate(selected):
                if i >= j:
                    continue
                score = solver.get_adjacency_score(r1, r2)
                if score > 0:
                    G.add_edge(r1, r2, weight=score)

        centrality = nx.degree_centrality(G)
        sorted_rooms = sorted(selected, key=lambda r: centrality.get(r, 0), reverse=True)

        n_cols = max(2, int(math.ceil(math.sqrt(n_rooms))))
        n_rows = max(2, int(math.ceil(n_rooms / n_cols)))
        cell_w = width // n_cols
        cell_h = height // n_rows

        placed = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if placed >= len(sorted_rooms):
                    break
                rt = sorted_rooms[placed]
                cfg = solver.ROOM_CONFIGS.get(rt, {})
                cx1 = left + col * cell_w
                cy1 = top + row * cell_h
                cx2 = left + (col + 1) * cell_w if col < n_cols - 1 else right
                cy2 = top + (row + 1) * cell_h if row < n_rows - 1 else bottom
                rooms.append(Room(cx1, cy1, cx2, cy2, rt,
                                  cfg.get("zone", Zone.PUBLIC),
                                  cfg.get("priority", RoomPriority.SECONDARY),
                                  cfg.get("needs_window", True)))
                placed += 1

        for row in range(n_rows):
            for col in range(n_cols):
                if col < n_cols - 1:
                    wx = left + (col + 1) * cell_w
                    walls.append(Wall(wx, top + row * cell_h, wx,
                                      top + (row + 1) * cell_h if row < n_rows - 1 else bottom,
                                      int_th, False))
                if row < n_rows - 1:
                    wy = top + (row + 1) * cell_h
                    walls.append(Wall(left + col * cell_w, wy,
                                      left + (col + 1) * cell_w if col < n_cols - 1 else right,
                                      wy, int_th, False))

        return walls, rooms


# ── 5g. Open plan ────────────────────────────────────────────────────────────
@register_strategy
class OpenPlanStrategy(LayoutStrategy):
    name = "open_plan"

    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = solver.rng
        walls, rooms = [], []

        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        open_ratio = rng.uniform(0.60, 0.70)
        corner = rng.choice(["top-left", "top-right", "bottom-left", "bottom-right"])

        priv_w = int(width * (1 - open_ratio))
        priv_h = int(height * rng.uniform(0.45, 0.65))

        if "left" in corner:
            priv_x1, priv_x2 = left, left + priv_w
            open_x1, open_x2 = left + priv_w, right
        else:
            priv_x1, priv_x2 = right - priv_w, right
            open_x1, open_x2 = left, right - priv_w

        if "top" in corner:
            priv_y1, priv_y2 = top, top + priv_h
        else:
            priv_y1, priv_y2 = bottom - priv_h, bottom

        walls.append(Wall(priv_x2 if "left" in corner else priv_x1,
                          priv_y1, priv_x2 if "left" in corner else priv_x1,
                          priv_y2, int_th, False))

        mid_y = priv_y1 + int((priv_y2 - priv_y1) * rng.uniform(0.45, 0.55))
        walls.append(Wall(priv_x1, mid_y, priv_x2, mid_y, int_th, False))

        rooms.append(Room(priv_x1, priv_y1, priv_x2, mid_y,
                          "bedroom", Zone.PRIVATE, RoomPriority.PRIMARY))
        rooms.append(Room(priv_x1, mid_y, priv_x2, priv_y2,
                          "bathroom", Zone.SERVICE, RoomPriority.TERTIARY, False))

        if "top" in corner:
            rooms.append(Room(open_x1, top, open_x2, bottom,
                              "living", Zone.PUBLIC, RoomPriority.PRIMARY))
            if "left" in corner:
                rooms.append(Room(left, priv_y2, left + priv_w, bottom,
                                  rng.choice(["kitchen", "dining"]),
                                  Zone.SERVICE, RoomPriority.SECONDARY))
                walls.append(Wall(left + priv_w, priv_y2, left + priv_w, bottom, int_th, False))
            else:
                rooms.append(Room(right - priv_w, priv_y2, right, bottom,
                                  rng.choice(["kitchen", "dining"]),
                                  Zone.SERVICE, RoomPriority.SECONDARY))
                walls.append(Wall(right - priv_w, priv_y2, right - priv_w, bottom, int_th, False))
        else:
            rooms.append(Room(open_x1, top, open_x2, bottom,
                              "living", Zone.PUBLIC, RoomPriority.PRIMARY))
            if "left" in corner:
                rooms.append(Room(left, top, left + priv_w, priv_y1,
                                  rng.choice(["kitchen", "dining"]),
                                  Zone.SERVICE, RoomPriority.SECONDARY))
                walls.append(Wall(left + priv_w, top, left + priv_w, priv_y1, int_th, False))
            else:
                rooms.append(Room(right - priv_w, top, right, priv_y1,
                                  rng.choice(["kitchen", "dining"]),
                                  Zone.SERVICE, RoomPriority.SECONDARY))
                walls.append(Wall(right - priv_w, top, right - priv_w, priv_y1, int_th, False))

        return walls, rooms


# ── 5h. Asymmetric strategy (NEW) ────────────────────────────────────────────
@register_strategy
class AsymmetricStrategy(LayoutStrategy):
    """
    Irregular room sizes — one dominant room + several smaller ones
    arranged along two sides. Creates more organic, less grid-like layouts.
    """
    name = "asymmetric"

    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = solver.rng
        walls, rooms = [], []

        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        available = list(config.room_types_enabled)

        # Dominant room takes 35-55% of one axis
        dominant_ratio = rng.uniform(0.35, 0.55)
        orientation = rng.choice(["left-dominant", "right-dominant",
                                   "top-dominant", "bottom-dominant"])

        if "left" in orientation or "right" in orientation:
            dom_w = int(width * dominant_ratio)
            if "left" in orientation:
                dom_x1, dom_x2 = left, left + dom_w
                rem_x1, rem_x2 = left + dom_w, right
            else:
                dom_x1, dom_x2 = right - dom_w, right
                rem_x1, rem_x2 = left, right - dom_w

            walls.append(Wall(dom_x2 if "left" in orientation else dom_x1,
                              top,
                              dom_x2 if "left" in orientation else dom_x1,
                              bottom, int_th, False))

            rooms.append(Room(dom_x1, top, dom_x2, bottom,
                              "living", Zone.PUBLIC, RoomPriority.PRIMARY))

            # Split remainder into 2-4 rooms vertically
            n_sub = rng.randint(2, max(2, min(4, brief.num_rooms)))
            sub_types = []
            for rt in ["kitchen", "bedroom", "bathroom", "corridor", "home_office"]:
                if rt in available and len(sub_types) < n_sub:
                    sub_types.append(rt)
            while len(sub_types) < n_sub:
                sub_types.append(rng.choice(available))

            ratios = [rng.uniform(0.7, 1.3) for _ in range(n_sub)]
            total_r = sum(ratios)
            ratios = [r / total_r for r in ratios]

            cur_y = top
            for idx, (ratio, rt) in enumerate(zip(ratios, sub_types)):
                next_y = cur_y + int(height * ratio) if idx < n_sub - 1 else bottom
                next_y = min(next_y, bottom)
                if idx < n_sub - 1:
                    walls.append(Wall(rem_x1, next_y, rem_x2, next_y, int_th, False))
                cfg = solver.ROOM_CONFIGS.get(rt, {})
                rooms.append(Room(rem_x1, cur_y, rem_x2, next_y, rt,
                                  cfg.get("zone", Zone.SERVICE),
                                  cfg.get("priority", RoomPriority.SECONDARY),
                                  cfg.get("needs_window", True)))
                cur_y = next_y
        else:
            dom_h = int(height * dominant_ratio)
            if "top" in orientation:
                dom_y1, dom_y2 = top, top + dom_h
                rem_y1, rem_y2 = top + dom_h, bottom
            else:
                dom_y1, dom_y2 = bottom - dom_h, bottom
                rem_y1, rem_y2 = top, bottom - dom_h

            walls.append(Wall(left,
                              dom_y2 if "top" in orientation else dom_y1,
                              right,
                              dom_y2 if "top" in orientation else dom_y1,
                              int_th, False))

            rooms.append(Room(left, dom_y1, right, dom_y2,
                              "living", Zone.PUBLIC, RoomPriority.PRIMARY))

            n_sub = rng.randint(2, max(2, min(4, brief.num_rooms)))
            sub_types = []
            for rt in ["bedroom", "kitchen", "bathroom", "corridor"]:
                if rt in available and len(sub_types) < n_sub:
                    sub_types.append(rt)
            while len(sub_types) < n_sub:
                sub_types.append(rng.choice(available))

            ratios = [rng.uniform(0.7, 1.3) for _ in range(n_sub)]
            total_r = sum(ratios)
            ratios = [r / total_r for r in ratios]

            cur_x = left
            for idx, (ratio, rt) in enumerate(zip(ratios, sub_types)):
                next_x = cur_x + int(width * ratio) if idx < n_sub - 1 else right
                next_x = min(next_x, right)
                if idx < n_sub - 1:
                    walls.append(Wall(next_x, rem_y1, next_x, rem_y2, int_th, False))
                cfg = solver.ROOM_CONFIGS.get(rt, {})
                rooms.append(Room(cur_x, rem_y1, next_x, rem_y2, rt,
                                  cfg.get("zone", Zone.SERVICE),
                                  cfg.get("priority", RoomPriority.SECONDARY),
                                  cfg.get("needs_window", True)))
                cur_x = next_x

        return walls, rooms


# ── 5i. Courtyard strategy (NEW) ─────────────────────────────────────────────
@register_strategy
class CourtyardStrategy(LayoutStrategy):
    """
    Rooms arranged around a central courtyard/atrium void.
    The void is drawn as a special room with hatching.
    """
    name = "courtyard"

    def generate(self, bounds, ext_th, int_th, solver, brief, config):
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        rng = solver.rng
        walls, rooms = [], []

        walls.extend([
            Wall(left, top, right, top, ext_th, True),
            Wall(left, bottom, right, bottom, ext_th, True),
            Wall(left, top, left, bottom, ext_th, True),
            Wall(right, top, right, bottom, ext_th, True),
        ])

        # Courtyard in center
        cx_margin = rng.uniform(0.25, 0.35)
        cy_margin = rng.uniform(0.25, 0.35)
        court_x1 = left + int(width * cx_margin)
        court_x2 = right - int(width * cx_margin)
        court_y1 = top + int(height * cy_margin)
        court_y2 = bottom - int(height * cy_margin)

        # Courtyard walls
        walls.append(Wall(court_x1, court_y1, court_x2, court_y1, int_th, False))
        walls.append(Wall(court_x1, court_y2, court_x2, court_y2, int_th, False))
        walls.append(Wall(court_x1, court_y1, court_x1, court_y2, int_th, False))
        walls.append(Wall(court_x2, court_y1, court_x2, court_y2, int_th, False))

        rooms.append(Room(court_x1, court_y1, court_x2, court_y2,
                          "atrium" if "atrium" in config.room_types_enabled else "living",
                          Zone.PUBLIC, RoomPriority.PRIMARY))

        available = [r for r in config.room_types_enabled if r != "atrium"]

        # Top strip
        rt_top = rng.choice(["living", "dining"] if "living" in available else available)
        rooms.append(Room(left, top, right, court_y1, rt_top,
                          solver.ROOM_CONFIGS.get(rt_top, {}).get("zone", Zone.PUBLIC),
                          solver.ROOM_CONFIGS.get(rt_top, {}).get("priority", RoomPriority.PRIMARY)))

        # Bottom strip — split into 2
        bot_split = left + int(width * rng.uniform(0.40, 0.60))
        walls.append(Wall(bot_split, court_y2, bot_split, bottom, int_th, False))
        rt_bl = rng.choice(["kitchen", "laundry"] if "kitchen" in available else available)
        rt_br = rng.choice(["bathroom"] if "bathroom" in available else available)
        rooms.append(Room(left, court_y2, bot_split, bottom, rt_bl,
                          solver.ROOM_CONFIGS.get(rt_bl, {}).get("zone", Zone.SERVICE),
                          RoomPriority.SECONDARY))
        rooms.append(Room(bot_split, court_y2, right, bottom, rt_br,
                          solver.ROOM_CONFIGS.get(rt_br, {}).get("zone", Zone.SERVICE),
                          RoomPriority.TERTIARY, False))

        # Left strip
        rt_left = rng.choice(["bedroom", "home_office"] if "bedroom" in available else available)
        rooms.append(Room(left, court_y1, court_x1, court_y2, rt_left,
                          solver.ROOM_CONFIGS.get(rt_left, {}).get("zone", Zone.PRIVATE),
                          RoomPriority.PRIMARY))

        # Right strip
        rt_right = rng.choice(["bedroom", "corridor"] if "bedroom" in available else available)
        rooms.append(Room(court_x2, court_y1, right, court_y2, rt_right,
                          solver.ROOM_CONFIGS.get(rt_right, {}).get("zone", Zone.PRIVATE),
                          RoomPriority.SECONDARY))

        return walls, rooms


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TEXTURE & HATCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TextureEngine:
    def __init__(self, rng, size):
        self.rng = rng
        self.size = size

    def apply_room_fill(self, img, room, style):
        x1, y1, x2, y2 = room.x1, room.y1, room.x2, room.y2
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return

        cv2.rectangle(img, (x1, y1), (x2, y2), style.fill, -1, cv2.LINE_AA)

        if style.texture and HAS_NOISE:
            self._apply_perlin_texture(img, x1, y1, x2, y2, style.texture, style.fill)

        if style.hatch_pattern:
            hatch_col = style.hatch_color or style.stroke
            if style.hatch_pattern == "diagonal":
                self._hatch_diagonal(img, x1, y1, x2, y2, style.hatch_spacing, hatch_col)
            elif style.hatch_pattern == "cross":
                self._hatch_cross(img, x1, y1, x2, y2, style.hatch_spacing, hatch_col)
            elif style.hatch_pattern == "dots":
                self._hatch_dots(img, x1, y1, x2, y2, style.hatch_spacing, hatch_col)

    def _apply_perlin_texture(self, img, x1, y1, x2, y2, tex_type, base_color):
        h, w = y2 - y1, x2 - x1
        if h < 4 or w < 4:
            return
        scale_map = {"wood": 0.02, "tile": 0.05, "concrete": 0.03}
        noise_scale = scale_map.get(tex_type, 0.03)
        intensity = 12
        offset_x = self.rng.uniform(0, 1000)
        offset_y = self.rng.uniform(0, 1000)
        for dy in range(0, h, 2):
            for dx in range(0, w, 2):
                n = pnoise2((dx + offset_x) * noise_scale,
                            (dy + offset_y) * noise_scale, octaves=3)
                delta = int(n * intensity)
                py, px = y1 + dy, x1 + dx
                if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                    pixel = img[py, px].astype(np.int16)
                    pixel = np.clip(pixel + delta, 0, 255).astype(np.uint8)
                    img[py, px] = pixel
                    if py + 1 < img.shape[0] and px + 1 < img.shape[1]:
                        img[py:py + 2, px:px + 2] = pixel

    def _hatch_diagonal(self, img, x1, y1, x2, y2, spacing, color):
        for offset in range(-max(x2 - x1, y2 - y1), max(x2 - x1, y2 - y1), spacing):
            p1 = (x1, y1 + offset)
            p2 = (x1 + (y2 - y1), y1 + offset + (y2 - y1))
            cv2.clipLine((y1, x1, y2 - y1, x2 - x1), p1, p2)
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

    def _hatch_cross(self, img, x1, y1, x2, y2, spacing, color):
        for x in range(x1, x2, spacing):
            cv2.line(img, (x, y1), (x, y2), color, 1, cv2.LINE_AA)
        for y in range(y1, y2, spacing):
            cv2.line(img, (x1, y), (x2, y), color, 1, cv2.LINE_AA)

    def _hatch_dots(self, img, x1, y1, x2, y2, spacing, color):
        for x in range(x1 + spacing // 2, x2, spacing):
            for y in range(y1 + spacing // 2, y2, spacing):
                cv2.circle(img, (x, y), max(1, spacing // 6), color, -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SEMANTIC FURNITURE PLACER
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticFurniturePlacer:
    CLEARANCE_RATIOS = {
        "bathroom": 0.15, "kitchen": 0.12, "living": 0.10,
        "bedroom": 0.12, "corridor": 0.05, "dining": 0.10,
        "home_office": 0.10, "gym": 0.08, "laundry": 0.10,
        "walk_in_closet": 0.08, "terrace": 0.05,
        "winter_garden": 0.08, "atrium": 0.06,
    }

    DENSITY = {
        "bathroom": (2, 4), "kitchen": (1, 2), "living": (4, 8),
        "bedroom": (2, 5), "corridor": (0, 2), "dining": (2, 4),
        "home_office": (2, 4), "gym": (1, 3), "laundry": (1, 2),
        "walk_in_closet": (1, 3), "terrace": (1, 3),
        "winter_garden": (1, 3), "atrium": (0, 2),
    }

    def __init__(self, rng, size, theme, semantic=True):
        self.rng = rng
        self.size = size
        self.theme = theme
        self.semantic = semantic
        self.occupied: List[OccupiedRegion] = []

    def reset(self):
        self.occupied = []

    def add_occupied(self, x1, y1, x2, y2, region_type):
        self.occupied.append(OccupiedRegion(
            min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), region_type
        ))

    def check_collision(self, x1, y1, x2, y2, margin=5):
        for r in self.occupied:
            if not (x2 + margin < r.x1 or x1 - margin > r.x2 or
                    y2 + margin < r.y1 or y1 - margin > r.y2):
                return True
        return False

    def place_furniture(self, img, rooms, walls):
        for room in rooms:
            if room.width < 80 or room.height < 80:
                continue
            if room.room_type == "kitchen":
                self._place_kitchen_suite(img, room)
            elif room.room_type == "bathroom":
                self._place_bathroom_suite(img, room)

            density = self.DENSITY.get(room.room_type, (1, 3))
            n_items = self.rng.randint(*density)
            clearance = self.CLEARANCE_RATIOS.get(room.room_type, 0.1)
            margin = int(min(room.width, room.height) * clearance)

            for _ in range(n_items):
                self._place_single_furniture(img, room, margin)

            if room.room_type in ("living", "kitchen", "corridor", "dining", "home_office"):
                if self.rng.random() < 0.7:
                    for _ in range(self.rng.randint(1, 3)):
                        self._place_decorative_element(img, room, margin)

    def _place_single_furniture(self, img, room, margin):
        fill = self.theme.furniture_fill
        stroke = self.theme.furniture_stroke
        for _ in range(10):
            shape = self.rng.choice(["rectangle", "ellipse", "circle"])
            rw, rh = room.width, room.height
            if shape == "rectangle":
                w = self.rng.randint(int(rw * 0.15), max(20, int(rw * 0.40)))
                h = self.rng.randint(int(rh * 0.15), max(20, int(rh * 0.40)))
                x_range = room.x2 - margin - w - (room.x1 + margin)
                y_range = room.y2 - margin - h - (room.y1 + margin)
                if x_range <= 0 or y_range <= 0:
                    continue
                x = room.x1 + margin + self.rng.randint(0, x_range)
                y = room.y1 + margin + self.rng.randint(0, y_range)
                if self.check_collision(x, y, x + w, y + h):
                    continue
                cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1, cv2.LINE_AA)
                cv2.rectangle(img, (x, y), (x + w, y + h), stroke, 1, cv2.LINE_AA)
                self.add_occupied(x, y, x + w, y + h, "furniture")
                break
            elif shape == "ellipse":
                a = self.rng.randint(int(rw * 0.10), max(15, int(rw * 0.25)))
                b = self.rng.randint(int(rh * 0.10), max(15, int(rh * 0.25)))
                cx_range = room.x2 - margin - a - (room.x1 + margin + a)
                cy_range = room.y2 - margin - b - (room.y1 + margin + b)
                if cx_range <= 0 or cy_range <= 0:
                    continue
                cx = room.x1 + margin + a + self.rng.randint(0, cx_range)
                cy = room.y1 + margin + b + self.rng.randint(0, cy_range)
                if self.check_collision(cx - a, cy - b, cx + a, cy + b):
                    continue
                cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, fill, -1, cv2.LINE_AA)
                cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, stroke, 1, cv2.LINE_AA)
                self.add_occupied(cx - a, cy - b, cx + a, cy + b, "furniture")
                break
            else:
                r = self.rng.randint(int(min(rw, rh) * 0.08),
                                     max(10, int(min(rw, rh) * 0.18)))
                cx_range = room.x2 - margin - r - (room.x1 + margin + r)
                cy_range = room.y2 - margin - r - (room.y1 + margin + r)
                if cx_range <= 0 or cy_range <= 0:
                    continue
                cx = room.x1 + margin + r + self.rng.randint(0, cx_range)
                cy = room.y1 + margin + r + self.rng.randint(0, cy_range)
                if self.check_collision(cx - r, cy - r, cx + r, cy + r):
                    continue
                cv2.circle(img, (cx, cy), r, fill, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), r, stroke, 1, cv2.LINE_AA)
                self.add_occupied(cx - r, cy - r, cx + r, cy + r, "furniture")
                break

    def _place_kitchen_suite(self, img, room):
        rw, rh = room.width, room.height
        if rw < 120 or rh < 120:
            return
        fill = self.theme.furniture_fill
        stroke = self.theme.furniture_stroke
        depth = max(24, int(min(rw, rh) * self.rng.uniform(0.08, 0.13)))
        side = self.rng.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            usable = rw - depth * 2
            if usable <= depth * 2:
                return
            band_len = int(usable * self.rng.uniform(0.6, 0.95))
            if band_len < depth * 3:
                return
            x1 = self.rng.randint(room.x1 + depth, room.x2 - depth - band_len)
            x2 = x1 + band_len
            if side == "top":
                y1, y2 = room.y1 + 5, room.y1 + 5 + depth
            else:
                y2, y1 = room.y2 - 5, room.y2 - 5 - depth
            cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), stroke, 1, cv2.LINE_AA)
            self.add_occupied(x1, y1, x2, y2, "furniture")
        else:
            usable = rh - depth * 2
            if usable <= depth * 2:
                return
            band_len = int(usable * self.rng.uniform(0.6, 0.95))
            if band_len < depth * 3:
                return
            y1 = self.rng.randint(room.y1 + depth, room.y2 - depth - band_len)
            y2 = y1 + band_len
            if side == "left":
                x1, x2 = room.x1 + 5, room.x1 + 5 + depth
            else:
                x2, x1 = room.x2 - 5, room.x2 - 5 - depth
            cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), stroke, 1, cv2.LINE_AA)
            self.add_occupied(x1, y1, x2, y2, "furniture")

    def _place_bathroom_suite(self, img, room):
        rw, rh = room.width, room.height
        if rw < 80 or rh < 80:
            return
        fill = self.theme.furniture_fill
        stroke = self.theme.furniture_stroke
        bg = self.theme.background

        if rw >= rh:
            tub_len = int(rw * self.rng.uniform(0.45, 0.75))
            tub_depth = int(rh * self.rng.uniform(0.35, 0.5))
            x1 = self.rng.randint(room.x1 + 8, room.x2 - 8 - tub_len)
            y1 = room.y1 + 5
            x2, y2 = x1 + tub_len, y1 + tub_depth
        else:
            tub_len = int(rh * self.rng.uniform(0.45, 0.75))
            tub_depth = int(rw * self.rng.uniform(0.35, 0.5))
            x1 = room.x1 + 5
            y1 = self.rng.randint(room.y1 + 8, room.y2 - 8 - tub_len)
            x2, y2 = x1 + tub_depth, y1 + tub_len

        cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), stroke, 1, cv2.LINE_AA)
        self.add_occupied(x1, y1, x2, y2, "furniture")

        pad = max(3, int(min(x2 - x1, y2 - y1) * 0.08))
        cv2.rectangle(img, (x1 + pad, y1 + pad), (x2 - pad, y2 - pad), bg, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1 + pad, y1 + pad), (x2 - pad, y2 - pad), stroke, 1, cv2.LINE_AA)

        # Sink
        base = min(rw, rh)
        sw, sh = int(base * 0.18), int(base * 0.14)
        sx1 = room.x2 - sw - 8
        sy1 = room.y2 - sh - 8
        cv2.rectangle(img, (sx1, sy1), (sx1 + sw, sy1 + sh), fill, -1, cv2.LINE_AA)
        cv2.rectangle(img, (sx1, sy1), (sx1 + sw, sy1 + sh), stroke, 1, cv2.LINE_AA)
        self.add_occupied(sx1, sy1, sx1 + sw, sy1 + sh, "furniture")

        # Toilet
        wc_r = int(base * 0.06)
        wc_cx, wc_cy = room.x1 + 12 + wc_r, room.y2 - 12 - wc_r
        cv2.circle(img, (wc_cx, wc_cy), wc_r, fill, -1, cv2.LINE_AA)
        cv2.circle(img, (wc_cx, wc_cy), wc_r, stroke, 1, cv2.LINE_AA)
        cv2.rectangle(img, (wc_cx - wc_r // 2, wc_cy - wc_r * 2),
                      (wc_cx + wc_r // 2, wc_cy - wc_r), fill, -1, cv2.LINE_AA)
        cv2.rectangle(img, (wc_cx - wc_r // 2, wc_cy - wc_r * 2),
                      (wc_cx + wc_r // 2, wc_cy - wc_r), stroke, 1, cv2.LINE_AA)
        self.add_occupied(wc_cx - wc_r, wc_cy - wc_r * 2,
                          wc_cx + wc_r, wc_cy + wc_r, "furniture")

    def _place_decorative_element(self, img, room, margin):
        fill = self.theme.furniture_fill
        stroke = self.theme.furniture_stroke
        elem_len = int(self.size * self.rng.uniform(0.02, 0.07))
        elem_thick = max(3, int(elem_len * self.rng.uniform(0.10, 1.20)))
        orientation = self.rng.choice(["vertical", "horizontal"])
        if orientation == "vertical":
            rect_h, rect_w = elem_len, elem_thick
        else:
            rect_w, rect_h = elem_len, elem_thick

        rw, rh = room.width, room.height
        s = min((rw - 2 * margin) / max(1, rect_w),
                (rh - 2 * margin) / max(1, rect_h), 1.0)
        rect_w, rect_h = max(3, int(rect_w * s)), max(3, int(rect_h * s))

        for _ in range(10):
            x_max = room.x2 - margin - rect_w
            x_min = room.x1 + margin
            y_max = room.y2 - margin - rect_h
            y_min = room.y1 + margin
            if x_max <= x_min or y_max <= y_min:
                return
            x = self.rng.randint(x_min, x_max)
            y = self.rng.randint(y_min, y_max)
            if self.check_collision(x, y, x + rect_w, y + rect_h, margin=10):
                continue
            cv2.rectangle(img, (x, y), (x + rect_w, y + rect_h), fill, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x + rect_w, y + rect_h), stroke, 1, cv2.LINE_AA)
            self.add_occupied(x, y, x + rect_w, y + rect_h, "furniture")
            break


# ═══════════════════════════════════════════════════════════════════════════════
# 8. VECTOR & CAD EXPORTERS
# ═══════════════════════════════════════════════════════════════════════════════

class SVGExporter:
    def __init__(self, theme, size):
        self.theme = theme
        self.size = size

    def export(self, path, walls, rooms, balconies):
        if not HAS_SVG:
            return

        def bgr_to_css(c):
            return f"rgb({c[2]},{c[1]},{c[0]})"

        dwg = svgwrite.Drawing(path, size=(self.size, self.size))
        for room in rooms:
            style = self.theme.room_styles.get(room.room_type, None)
            fill_col = bgr_to_css(style.fill) if style else bgr_to_css(self.theme.background)
            dwg.add(dwg.rect(insert=(room.x1, room.y1), size=(room.width, room.height),
                             fill=fill_col, stroke=bgr_to_css(self.theme.wall_color), stroke_width=1))
            cx, cy = room.center
            dwg.add(dwg.text(room.room_type, insert=(cx, cy), text_anchor="middle",
                             font_size="10px", fill=bgr_to_css(self.theme.label_color)))
        for w in walls:
            stroke_w = 1 if w.is_guardrail else max(1, w.thickness // 4)
            color = bgr_to_css(self.theme.wall_ext_color if w.is_external else self.theme.wall_color)
            dwg.add(dwg.line(start=(w.x1, w.y1), end=(w.x2, w.y2),
                             stroke=color, stroke_width=stroke_w))
        for b in balconies:
            dwg.add(dwg.rect(insert=(min(b.x1, b.x2), min(b.y1, b.y2)),
                             size=(abs(b.x2 - b.x1), abs(b.y2 - b.y1)),
                             fill="none", stroke=bgr_to_css(self.theme.balcony_stripe),
                             stroke_width=1, stroke_dasharray="4,2"))
        dwg.save()


class DXFExporter:
    def __init__(self, theme, size):
        self.theme = theme
        self.size = size

    def export(self, path, walls, rooms, balconies):
        if not HAS_DXF:
            return
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        doc.layers.add("WALLS_EXT", color=7)
        doc.layers.add("WALLS_INT", color=3)
        doc.layers.add("ROOMS", color=1)
        doc.layers.add("BALCONIES", color=5)
        doc.layers.add("LABELS", color=2)
        for room in rooms:
            points = [(room.x1, room.y1), (room.x2, room.y1),
                      (room.x2, room.y2), (room.x1, room.y2), (room.x1, room.y1)]
            msp.add_lwpolyline(points, dxfattribs={"layer": "ROOMS"})
            cx, cy = room.center
            msp.add_text(room.room_type, dxfattribs={"layer": "LABELS", "height": 8}).set_placement((cx, cy))
        for w in walls:
            layer = "WALLS_EXT" if w.is_external else "WALLS_INT"
            msp.add_line((w.x1, w.y1), (w.x2, w.y2), dxfattribs={"layer": layer})
        for b in balconies:
            pts = [(b.x1, b.y1), (b.x2, b.y1), (b.x2, b.y2), (b.x1, b.y2), (b.x1, b.y1)]
            msp.add_lwpolyline(pts, dxfattribs={"layer": "BALCONIES"})
        doc.saveas(path)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. METADATA / JSON EXPORTER
# ═══════════════════════════════════════════════════════════════════════════════

class MetadataExporter:
    def __init__(self, theme, solver):
        self.theme = theme
        self.solver = solver

    def export(self, path, rooms, walls, balconies, brief, config):
        data = {
            "generator_version": "5.0",
            "config": {
                "style": config.style,
                "building_type": config.building_type,
                "palette": config.palette_name,
                "img_size": config.img_size,
            },
            "brief": {
                "style": brief.style,
                "num_rooms": brief.num_rooms,
                "has_balcony": brief.has_balcony,
                "circulation_type": brief.circulation_type,
                "light_priority": brief.light_priority,
            },
            "rooms": [],
            "adjacencies": [],
            "balconies": [],
            "total_area": sum(r.area for r in rooms),
        }
        for r in rooms:
            data["rooms"].append({
                "id": r.id, "type": r.room_type,
                "zone": r.zone.name if isinstance(r.zone, Zone) else str(r.zone),
                "bounds": [r.x1, r.y1, r.x2, r.y2],
                "area": r.area, "center": list(r.center), "floor": r.floor,
            })
        for i, r1 in enumerate(rooms):
            for j, r2 in enumerate(rooms):
                if i >= j:
                    continue
                p1 = shapely_box(r1.x1, r1.y1, r1.x2, r1.y2)
                p2 = shapely_box(r2.x1, r2.y1, r2.x2, r2.y2)
                if p1.buffer(3).intersects(p2):
                    score = self.solver.get_adjacency_score(r1.room_type, r2.room_type)
                    data["adjacencies"].append({
                        "room_a": r1.id, "room_b": r2.id,
                        "type_a": r1.room_type, "type_b": r2.room_type,
                        "score": score,
                    })
        for b in balconies:
            data["balconies"].append({
                "bounds": [b.x1, b.y1, b.x2, b.y2],
                "side": b.side, "archetype": b.archetype,
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

class ThemeRenderer:
    def __init__(self, theme, size, rng, stroke_scale=1.0):
        self.theme = theme
        self.size = size
        self.rng = rng
        self.stroke_scale = stroke_scale
        self.texture_engine = TextureEngine(rng, size)

    def _th(self, thickness):
        return max(1, int(thickness * self.stroke_scale))

    def fill_rooms(self, img, rooms):
        for room in rooms:
            style = self.theme.room_styles.get(room.room_type, None)
            if style:
                self.texture_engine.apply_room_fill(img, room, style)
            else:
                fill = {
                    Zone.PUBLIC: self.theme.accent_public,
                    Zone.PRIVATE: self.theme.accent_private,
                    Zone.SERVICE: self.theme.accent_service,
                    Zone.CIRCULATION: self.theme.accent_circulation,
                }.get(room.zone, self.theme.background)
                cv2.rectangle(img, (room.x1, room.y1), (room.x2, room.y2),
                              fill, -1, cv2.LINE_AA)

    def draw_walls(self, img, walls):
        for w in walls:
            if w.is_guardrail:
                cv2.line(img, (w.x1, w.y1), (w.x2, w.y2),
                         self.theme.wall_color, self._th(2), cv2.LINE_AA)
            else:
                half = w.thickness // 2
                x1, x2 = min(w.x1, w.x2), max(w.x1, w.x2)
                y1, y2 = min(w.y1, w.y2), max(w.y1, w.y2)
                color = self.theme.wall_ext_color if w.is_external else self.theme.wall_color
                if self.theme.shadow_offset > 0:
                    so = self.theme.shadow_offset
                    shadow_col = self.theme.shadow_color
                    if w.x1 == w.x2:
                        pt1 = (x1 - half + so, y1 + so)
                        pt2 = (x1 + half + so, y2 + so)
                    else:
                        pt1 = (x1 + so, y1 - half + so)
                        pt2 = (x2 + so, y1 + half + so)
                    cv2.rectangle(img, pt1, pt2, shadow_col, -1, cv2.LINE_AA)
                if w.x1 == w.x2:
                    pt1, pt2 = (x1 - half, y1), (x1 + half, y2)
                else:
                    pt1, pt2 = (x1, y1 - half), (x2, y1 + half)
                cv2.rectangle(img, pt1, pt2, color, -1, cv2.LINE_AA)

    def draw_balcony_floor(self, img, balconies):
        for b in balconies:
            x1, x2 = min(b.x1, b.x2), max(b.x1, b.x2)
            y1, y2 = min(b.y1, b.y2), max(b.y1, b.y2)
            step = int(self.size * 0.005)
            th = self._th(1)
            for y in range(y1, y2, step):
                cv2.line(img, (x1, y), (x2, y), self.theme.balcony_stripe, th, cv2.LINE_AA)

    def add_annotations(self, img, rooms):
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_scale = 0.0007 * self.size
        thickness = max(1, self._th(int(self.size * 0.0012)))
        label_color = self.theme.label_color
        for room in rooms:
            if room.room_type == "balcony":
                continue
            cx, cy = room.center
            name_scale = base_scale * 0.6
            name = room.room_type.replace("_", " ").title()
            (tw, th), _ = cv2.getTextSize(name, font, name_scale, max(1, thickness - 1))
            name_x = cx - tw // 2
            name_y = cy - 5
            cv2.putText(img, name, (name_x, name_y), font, name_scale,
                        label_color, max(1, thickness - 1), cv2.LINE_AA)
            area_txt = f"{self.rng.uniform(4.0, 50.0):.1f}".replace(".", ",")
            (tw2, th2), _ = cv2.getTextSize(area_txt, font, base_scale, thickness)
            cv2.putText(img, area_txt, (cx - tw2 // 2, cy + th2 + 5), font, base_scale,
                        label_color, thickness, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE ROTATOR
# ═══════════════════════════════════════════════════════════════════════════════

class StyleRotator:
    def __init__(self, mode="random", themes=None, weights=None,
                 strategy_pool=None, rng=None):
        self.mode = mode
        self.rng = rng or random.Random(42)
        self.cycle_index = 0
        self.themes = themes or list(_THEMES.keys())
        if weights and len(weights) == len(self.themes):
            self.weights = weights
        else:
            self.weights = [1.0] * len(self.themes)
        self.strategy_pool = strategy_pool or list(_STRATEGY_REGISTRY.keys())

    def next_theme(self):
        if self.mode == "cycle":
            name = self.themes[self.cycle_index % len(self.themes)]
            self.cycle_index += 1
        elif self.mode == "weighted":
            name = self.rng.choices(self.themes, weights=self.weights, k=1)[0]
        else:
            name = self.rng.choice(self.themes)
        return deepcopy(_THEMES.get(name, _THEMES["modern_minimal"]))

    def next_strategy_name(self):
        return self.rng.choice(self.strategy_pool)

    def next_building_type(self):
        return self.rng.choice(["apartment", "villa", "studio", "office", "loft"])

    def next_room_count_range(self):
        ranges = [(1, 3), (2, 4), (3, 6), (4, 8), (5, 10)]
        return self.rng.choice(ranges)

    def next_wall_thickness(self):
        presets = [(5, 15), (7, 30), (15, 40)]
        return self.rng.choice(presets)

    def next_balcony_prob(self):
        return self.rng.choice([0.0, 0.3, 0.5, 0.7, 1.0])


# ═══════════════════════════════════════════════════════════════════════════════
# 11. MAIN GENERATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SmartFloorPlanGenerator:
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = FloorPlanConfig(**kwargs)
        self.config = config
        self.out_size = config.img_size
        self.super_sampling = config.super_sampling
        self.size = config.img_size * config.super_sampling
        self.rng = random.Random(config.seed)
        self.nprng = np.random.default_rng(config.seed)
        self.door_swing_arc_probability = 0.5

        self.theme = config.get_theme()
        self.solver = ConstraintSolver(self.rng, config.constraint_solver)
        self.furniture_placer = SemanticFurniturePlacer(
            self.rng, self.size, self.theme, config.semantic_furniture)
        self.renderer = ThemeRenderer(self.theme, self.size, self.rng)
        self.texture_engine = TextureEngine(self.rng, self.size)

        self.strategies: Dict[str, LayoutStrategy] = {}
        for name in config.strategies:
            if name in _STRATEGY_REGISTRY:
                self.strategies[name] = _STRATEGY_REGISTRY[name]()
        if not self.strategies:
            self.strategies["central"] = CentralCorridorStrategy()

        self.svg_exporter = SVGExporter(self.theme, self.size)
        self.dxf_exporter = DXFExporter(self.theme, self.size)
        self.metadata_exporter = MetadataExporter(self.theme, self.solver)

        self.style_rotator: Optional[StyleRotator] = None
        if config.multi_style:
            self.style_rotator = StyleRotator(
                mode=config.style_rotation_mode,
                themes=config.style_themes,
                weights=config.style_weights,
                strategy_pool=list(self.strategies.keys()) if not config.vary_strategies
                else list(_STRATEGY_REGISTRY.keys()),
                rng=self.rng,
            )

        # ── Window sub-types for variety ─────────────────────────────
        # Expanded length ratios for wider range
        self.window_length_ratios = (
            0.005, 0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035,
            0.040, 0.045, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100, 0.120
        )
        self.door_length_ratios = (0.04, 0.05, 0.06)

    def generate_brief(self):
        styles = ["compact", "spacious", "linear", "square"]
        style_weights = [0.3, 0.25, 0.25, 0.2]

        if self.style_rotator and self.config.vary_strategies:
            circ_type = self.style_rotator.next_strategy_name()
            if circ_type in _STRATEGY_REGISTRY and circ_type not in self.strategies:
                self.strategies[circ_type] = _STRATEGY_REGISTRY[circ_type]()
        else:
            circ_types = list(self.strategies.keys())
            circ_type = self.rng.choice(circ_types)

        return ApartmentBrief(
            style=self.rng.choices(styles, style_weights)[0],
            num_rooms=self.rng.choices(
                list(range(self.config.min_rooms, self.config.max_rooms + 1)),
                [1.0] * (self.config.max_rooms - self.config.min_rooms + 1),
            )[0],
            has_balcony=self.rng.random() < self.config.balcony_probability,
            circulation_type=circ_type,
            light_priority=self.rng.choice(["living", "bedroom", "balanced"]),
            building_type=self.config.building_type,
            cultural_preset=self.config.cultural_preset,
        )

    def _apply_style_for_image(self):
        overrides = {}
        if not self.style_rotator:
            # Still apply color jitter even without style rotation
            if self.config.color_jitter_intensity > 0:
                self.theme = self.config.get_theme().jittered_copy(
                    self.rng, self.config.color_jitter_intensity)
                self._rebuild_with_theme()
            return overrides

        self.theme = self.style_rotator.next_theme()
        # Apply per-image color jitter within the chosen style
        if self.config.color_jitter_intensity > 0:
            self.theme = self.theme.jittered_copy(
                self.rng, self.config.color_jitter_intensity)
        overrides["theme"] = self.theme.name

        self._rebuild_with_theme()

        if self.config.vary_strategies:
            strat_name = self.style_rotator.next_strategy_name()
            if strat_name in _STRATEGY_REGISTRY and strat_name not in self.strategies:
                self.strategies[strat_name] = _STRATEGY_REGISTRY[strat_name]()
            overrides["strategy"] = strat_name

        if self.config.vary_room_counts:
            min_r, max_r = self.style_rotator.next_room_count_range()
            self.config.min_rooms = min_r
            self.config.max_rooms = max_r
            overrides["room_range"] = (min_r, max_r)

        if self.config.vary_balcony:
            self.config.balcony_probability = self.style_rotator.next_balcony_prob()
            overrides["balcony_prob"] = self.config.balcony_probability

        return overrides

    def _rebuild_with_theme(self):
        """Rebuild renderer, furniture placer, exporters with current theme."""
        self.renderer = ThemeRenderer(self.theme, self.size, self.rng)
        self.furniture_placer = SemanticFurniturePlacer(
            self.rng, self.size, self.theme, self.config.semantic_furniture)
        self.svg_exporter = SVGExporter(self.theme, self.size)
        self.dxf_exporter = DXFExporter(self.theme, self.size)
        self.metadata_exporter = MetadataExporter(self.theme, self.solver)

    # ── Dataset generation ────────────────────────────────────────────────────
    def generate_dataset(self, n_images=None, out_dir=None, wall_thickness_range=None):
        n_images = n_images or self.config.count
        out_dir = out_dir or self.config.output_dir
        wall_thickness_range = wall_thickness_range or self.config.wall_thickness_range

        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
        svg_dir = os.path.join(out_dir, "svg")
        dxf_dir = os.path.join(out_dir, "dxf")
        meta_dir = os.path.join(out_dir, "metadata")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        export_formats = [ExportFormat(f) if isinstance(f, str) else f
                          for f in self.config.export_formats]

        if ExportFormat.SVG in export_formats:
            os.makedirs(svg_dir, exist_ok=True)
        if ExportFormat.DXF in export_formats:
            os.makedirs(dxf_dir, exist_ok=True)
        if ExportFormat.JSON in export_formats or self.config.generate_metadata_json:
            os.makedirs(meta_dir, exist_ok=True)

        style_counts: Dict[str, int] = defaultdict(int)
        strategy_counts: Dict[str, int] = defaultdict(int)

        for i in range(n_images):
            overrides = self._apply_style_for_image()

            if self.config.vary_wall_thickness and self.style_rotator:
                wall_thickness_range = self.style_rotator.next_wall_thickness()

            base_ext = self.rng.randint(*wall_thickness_range)
            ext_th = base_ext * self.super_sampling
            int_th = int(ext_th * self.rng.uniform(0.5, 0.7))

            result = self.generate_single_plan(ext_th, int_th)

            hi_res_img = result["image"]
            labels = result["labels"]
            walls = result["walls"]
            rooms = result["rooms"]
            balconies = result["balconies"]
            brief = result["brief"]

            style_counts[self.theme.name] += 1
            strategy_counts[brief.circulation_type] += 1

            img = cv2.resize(hi_res_img, (self.out_size, self.out_size),
                             interpolation=cv2.INTER_AREA)

            if self.rng.random() < self.config.rotation_probability:
                img, labels = self._apply_rotation(img, labels)

            if self.rng.random() < self.config.augmentation_probability:
                img = self._apply_augmentations(img)

            fname = f"{i:05d}"
            cv2.imwrite(os.path.join(img_dir, fname + ".png"), img)

            with open(os.path.join(lbl_dir, fname + ".txt"), "w", encoding="utf-8") as f:
                for cls_id, cx, cy, w, h in labels:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            if ExportFormat.SVG in export_formats:
                scale_f = self.out_size / self.size
                scaled_walls = [
                    Wall(int(w.x1 * scale_f), int(w.y1 * scale_f),
                         int(w.x2 * scale_f), int(w.y2 * scale_f),
                         max(1, int(w.thickness * scale_f)),
                         w.is_external, w.is_guardrail, w.is_balcony_interface)
                    for w in walls
                ]
                scaled_rooms = [
                    Room(int(r.x1 * scale_f), int(r.y1 * scale_f),
                         int(r.x2 * scale_f), int(r.y2 * scale_f),
                         r.room_type, r.zone, r.priority, r.needs_window, r.id)
                    for r in rooms
                ]
                scaled_balconies = [
                    Balcony(int(b.x1 * scale_f), int(b.y1 * scale_f),
                            int(b.x2 * scale_f), int(b.y2 * scale_f),
                            b.side, b.archetype)
                    for b in balconies
                ]
                self.svg_exporter.export(os.path.join(svg_dir, fname + ".svg"),
                                         scaled_walls, scaled_rooms, scaled_balconies)

            if ExportFormat.DXF in export_formats:
                scale_f = self.out_size / self.size
                scaled_walls_d = [
                    Wall(int(w.x1 * scale_f), int(w.y1 * scale_f),
                         int(w.x2 * scale_f), int(w.y2 * scale_f),
                         max(1, int(w.thickness * scale_f)),
                         w.is_external, w.is_guardrail, w.is_balcony_interface)
                    for w in walls
                ]
                scaled_rooms_d = [
                    Room(int(r.x1 * scale_f), int(r.y1 * scale_f),
                         int(r.x2 * scale_f), int(r.y2 * scale_f),
                         r.room_type, r.zone, r.priority, r.needs_window, r.id)
                    for r in rooms
                ]
                scaled_balc_d = [
                    Balcony(int(b.x1 * scale_f), int(b.y1 * scale_f),
                            int(b.x2 * scale_f), int(b.y2 * scale_f),
                            b.side, b.archetype)
                    for b in balconies
                ]
                self.dxf_exporter.export(os.path.join(dxf_dir, fname + ".dxf"),
                                          scaled_walls_d, scaled_rooms_d, scaled_balc_d)

            if ExportFormat.JSON in export_formats or self.config.generate_metadata_json:
                meta_path = os.path.join(meta_dir, fname + ".json")
                self.metadata_exporter.export(meta_path, rooms, walls, balconies, brief, self.config)
                if overrides:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    meta["style_overrides"] = overrides
                    meta["actual_theme"] = self.theme.name
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)

            if (i + 1) % 100 == 0 or i == n_images - 1:
                print(f"  [{i + 1}/{n_images}] theme={self.theme.name}")

        if self.style_rotator:
            print(f"\n── Style distribution across {n_images} images ──")
            for name, count in sorted(style_counts.items()):
                pct = 100 * count / n_images
                bar = "█" * int(pct / 2)
                print(f"  {name:<22s} {count:>5d} ({pct:5.1f}%) {bar}")
            print(f"\n── Strategy distribution ──")
            for name, count in sorted(strategy_counts.items()):
                pct = 100 * count / n_images
                bar = "█" * int(pct / 2)
                print(f"  {name:<22s} {count:>5d} ({pct:5.1f}%) {bar}")

    # ── Single plan generation ────────────────────────────────────────────────
    def generate_single_plan(self, ext_wall_thickness, int_wall_thickness):
        self.furniture_placer.reset()
        bg = self.theme.background
        img = np.full((self.size, self.size, 3), bg, dtype=np.uint8)

        stroke_scale = self.rng.uniform(0.5, 2.5) if self.config.enable_augmentations else 1.0
        self.renderer = ThemeRenderer(self.theme, self.size, self.rng, stroke_scale=stroke_scale)

        margin = int(self.size * self.rng.uniform(0.12, 0.28))
        bounds = (margin, margin, self.size - margin, self.size - margin)

        brief = self.generate_brief()
        strategy_name = brief.circulation_type
        strategy = self.strategies.get(strategy_name, list(self.strategies.values())[0])

        walls, rooms = strategy.generate(
            bounds, ext_wall_thickness, int_wall_thickness,
            self.solver, brief, self.config,
        )

        balconies: List[Balcony] = []
        if brief.has_balcony:
            walls, rooms, balconies = self._integrate_balconies(
                walls, rooms, ext_wall_thickness
            )

        # Render
        self.renderer.fill_rooms(img, rooms)
        self.renderer.draw_balcony_floor(img, balconies)
        for b in balconies:
            x1, x2 = min(b.x1, b.x2), max(b.x1, b.x2)
            y1, y2 = min(b.y1, b.y2), max(b.y1, b.y2)
            self.furniture_placer.add_occupied(x1, y1, x2, y2, "balcony")

        self.renderer.draw_walls(img, walls)
        self.furniture_placer.place_furniture(img, rooms, walls)

        labels: List[Tuple[int, float, float, float, float]] = []

        # Balcony features
        balcony_labels = self._draw_balcony_features(img, balconies, int_wall_thickness)
        labels.extend(balcony_labels)

        # Windows
        external_walls = [w for w in walls if w.is_external and not w.is_guardrail]
        for w in external_walls:
            length = max(abs(w.x2 - w.x1), abs(w.y2 - w.y1))
            n_windows = self.rng.randint(1, max(2, int(length / 200)))
            for _ in range(n_windows):
                if self.rng.random() < 0.7:
                    box = self._place_window(img, w)
                    if box:
                        labels.append((1, *box))

        # Doors — NO SWING ARC, smaller angle
        internal_walls = [w for w in walls
                          if not w.is_external and not w.is_balcony_interface]
        door_spans: Dict[int, List[Tuple[int, int]]] = {}
        for w in internal_walls:
            length = max(abs(w.x2 - w.x1), abs(w.y2 - w.y1))
            if length < int_wall_thickness * 3:
                continue
            n_doors = 1 if length < 250 * self.super_sampling else self.rng.randint(1, 2)
            spans = door_spans.setdefault(id(w), [])
            for _ in range(n_doors):
                if self.rng.random() < 0.8:
                    box = self._place_door(img, w, spans)
                    if box:
                        labels.append((0, *box))

        self.renderer.add_annotations(img, rooms)

        if self.config.multi_story and self.config.num_floors > 1:
            self._draw_staircase_symbol(img, rooms)

        return {
            "image": img, "labels": labels, "walls": walls,
            "rooms": rooms, "balconies": balconies, "brief": brief,
        }

    def _apply_augmentations(self, img, labels=None):
        if not self.config.enable_augmentations:
            return img

        h, w = img.shape[:2]

        if self.rng.random() < 0.8:
            brightness_factor = 1.0 + self.rng.uniform(
                -self.config.aug_brightness_range, self.config.aug_brightness_range)
            img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

        if self.rng.random() < self.config.aug_blur_prob:
            k = self.rng.choice([3, 5, 7])
            sigma = self.rng.uniform(0.5, 2.0)
            img = cv2.GaussianBlur(img, (k, k), sigma)

        if self.rng.random() < self.config.aug_noise_prob:
            sigma = self.rng.uniform(5, 25)
            gauss = np.random.normal(0, sigma, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)

        if self.rng.random() < self.config.aug_jpeg_prob:
            quality = self.rng.randint(10, 80)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            if result:
                img = cv2.imdecode(encimg, 1)

        return img

    # ── Multi-story ───────────────────────────────────────────────────────────
    def generate_multi_story(self, ext_th, int_th):
        floors = []
        for floor_idx in range(self.config.num_floors):
            result = self.generate_single_plan(ext_th, int_th)
            for r in result["rooms"]:
                r.floor = floor_idx
            result["floor"] = floor_idx
            floors.append(result)
        return floors

    def _draw_staircase_symbol(self, img, rooms):
        corridors = [r for r in rooms if r.zone == Zone.CIRCULATION]
        if not corridors:
            return
        room = corridors[0]
        cx, cy = room.center
        stair_w = min(room.width // 3, 60)
        stair_h = min(room.height // 3, 80)
        x1, y1 = cx - stair_w // 2, cy - stair_h // 2
        n_steps = 6
        step_h = stair_h // n_steps
        for i in range(n_steps):
            sy = y1 + i * step_h
            cv2.rectangle(img, (x1, sy), (x1 + stair_w, sy + step_h),
                          self.theme.furniture_fill, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, sy), (x1 + stair_w, sy + step_h),
                          self.theme.wall_color, 1, cv2.LINE_AA)
        arrow_x = x1 + stair_w // 2
        cv2.arrowedLine(img, (arrow_x, y1 + stair_h),
                        (arrow_x, y1), self.theme.wall_color, 2,
                        cv2.LINE_AA, tipLength=0.3)

    # ── Balcony integration ───────────────────────────────────────────────────
    def _integrate_balconies(self, walls, rooms, ext_th):
        balconies = []
        new_walls = list(walls)
        candidates = [r for r in rooms if r.room_type in
                      ("living", "kitchen", "bedroom", "dining", "terrace")]
        if not candidates:
            return new_walls, rooms, balconies
        selected = self.rng.sample(candidates, k=min(len(candidates), self.rng.randint(1, 2)))
        for room in selected:
            wall, side = self._find_external_wall(room, new_walls)
            if not wall:
                continue
            dim = room.height if side in ("left", "right") else room.width
            depth = int(self.size * (0.08 if dim > self.size * 0.35 else 0.12))
            balcony, interface = self._carve_balcony(room, side, depth, ext_th)
            if balcony:
                new_walls.append(interface)
                balconies.append(balcony)
                new_walls = self._convert_to_guardrail(new_walls, wall, balcony, side)
        return new_walls, rooms, balconies

    def _find_external_wall(self, room, walls):
        for side in ("left", "right", "top", "bottom"):
            for w in walls:
                if not w.is_external or w.is_guardrail:
                    continue
                if self._wall_matches_side(w, room, side):
                    return w, side
        return None, None

    def _wall_matches_side(self, wall, room, side):
        tol = 20
        if side == "left" and wall.x1 == wall.x2 and abs(wall.x1 - room.x1) < tol:
            return max(wall.y1, wall.y2) >= room.y1 and min(wall.y1, wall.y2) <= room.y2
        if side == "right" and wall.x1 == wall.x2 and abs(wall.x1 - room.x2) < tol:
            return max(wall.y1, wall.y2) >= room.y1 and min(wall.y1, wall.y2) <= room.y2
        if side == "top" and wall.y1 == wall.y2 and abs(wall.y1 - room.y1) < tol:
            return max(wall.x1, wall.x2) >= room.x1 and min(wall.x1, wall.x2) <= room.x2
        if side == "bottom" and wall.y1 == wall.y2 and abs(wall.y1 - room.y2) < tol:
            return max(wall.x1, wall.x2) >= room.x1 and min(wall.x1, wall.x2) <= room.x2
        return False

    def _carve_balcony(self, room, side, depth, ext_th):
        if side == "left":
            old = room.x1
            room.x1 += depth
            bx1, by1, bx2, by2 = old, room.y1, room.x1, room.y2
            iface = Wall(room.x1, room.y1, room.x1, room.y2, ext_th, False, is_balcony_interface=True)
        elif side == "right":
            old = room.x2
            room.x2 -= depth
            bx1, by1, bx2, by2 = room.x2, room.y1, old, room.y2
            iface = Wall(room.x2, room.y1, room.x2, room.y2, ext_th, False, is_balcony_interface=True)
        elif side == "top":
            old = room.y1
            room.y1 += depth
            bx1, by1, bx2, by2 = room.x1, old, room.x2, room.y1
            iface = Wall(room.x1, room.y1, room.x2, room.y1, ext_th, False, is_balcony_interface=True)
        elif side == "bottom":
            old = room.y2
            room.y2 -= depth
            bx1, by1, bx2, by2 = room.x1, room.y2, room.x2, old
            iface = Wall(room.x1, room.y2, room.x2, room.y2, ext_th, False, is_balcony_interface=True)
        else:
            return None, None
        arch = "large_linear" if max(bx2 - bx1, by2 - by1) > self.size * 0.3 else "compact_box"
        return Balcony(bx1, by1, bx2, by2, side, arch, iface), iface

    def _convert_to_guardrail(self, walls, target, balcony, side):
        if target in walls:
            walls.remove(target)
        if side in ("left", "right"):
            w_s, w_e = min(target.y1, target.y2), max(target.y1, target.y2)
            b_s, b_e = min(balcony.y1, balcony.y2), max(balcony.y1, balcony.y2)
            if w_s < b_s:
                walls.append(Wall(target.x1, w_s, target.x2, b_s, target.thickness, True))
            walls.append(Wall(target.x1, b_s, target.x2, b_e, 2, True, is_guardrail=True))
            if w_e > b_e:
                walls.append(Wall(target.x1, b_e, target.x2, w_e, target.thickness, True))
        else:
            w_s, w_e = min(target.x1, target.x2), max(target.x1, target.x2)
            b_s, b_e = min(balcony.x1, balcony.x2), max(balcony.x1, balcony.x2)
            if w_s < b_s:
                walls.append(Wall(w_s, target.y1, b_s, target.y2, target.thickness, True))
            walls.append(Wall(b_s, target.y1, b_e, target.y2, 2, True, is_guardrail=True))
            if w_e > b_e:
                walls.append(Wall(b_e, target.y1, w_e, target.y2, target.thickness, True))
        return walls

    def _draw_balcony_features(self, img, balconies, int_th):
        labels = []
        for b in balconies:
            w = b.interface_wall
            if not w:
                continue
            is_vert = abs(w.x1 - w.x2) < abs(w.y1 - w.y2)
            start, end = ((min(w.y1, w.y2), max(w.y1, w.y2)) if is_vert
                          else (min(w.x1, w.x2), max(w.x1, w.x2)))
            pad = int_th * 2
            start, end = start + pad, end - pad
            length = end - start
            if length <= 0:
                continue

            elements = []
            if b.archetype == "large_linear":
                d_len = int(length * 0.2)
                w_len = int(length * 0.4)
                elements.append(("door", start, start + d_len))
                mid = start + d_len + int(length * 0.1)
                elements.append(("window", mid, mid + w_len))
                elements.append(("door", end - d_len, end))
            else:
                w_len = int(length * 0.5)
                d_len = int(length * 0.3)
                if self.rng.random() < 0.5:
                    elements.append(("window", start, start + w_len))
                    elements.append(("door", start + w_len + int(length * 0.05),
                                     start + w_len + int(length * 0.05) + d_len))
                else:
                    elements.append(("door", start, start + d_len))
                    elements.append(("window", start + d_len + int(length * 0.05),
                                     start + d_len + int(length * 0.05) + w_len))

            for el_type, s, e in elements:
                box = self._draw_interface_element(img, w, s, e, el_type, b.side)
                if box:
                    labels.append((0 if el_type == "door" else 1, *box))

        return labels

    def _draw_interface_element(self, img, wall, start, end, el_type, side):
        """Draw door/window on balcony interface wall.
        Doors have small opening angle; swing arc drawn probabilistically."""
        th = wall.thickness
        half_th = th // 2
        is_vert = abs(wall.x1 - wall.x2) < abs(wall.y1 - wall.y2)
        bg = self.theme.background
        wc = self.theme.wall_color

        if is_vert:
            cx = wall.x1
            p1, p2 = (cx - half_th, start), (cx + half_th, end)
        else:
            cy = wall.y1
            p1, p2 = (start, cy - half_th), (end, cy + half_th)

        cv2.rectangle(img, p1, p2, bg, -1)

        line_th = self.rng.randint(1, 2)
        off = max(1, half_th // 8)

        if el_type == "window":
            if is_vert:
                cv2.line(img, (cx - off, start), (cx - off, end), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (cx + off, start), (cx + off, end), wc, line_th, cv2.LINE_AA)
            else:
                cv2.line(img, (start, cy - off), (end, cy - off), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (start, cy + off), (end, cy + off), wc, line_th, cv2.LINE_AA)
        else:
            # Door: draw slab at small angle
            door_len = end - start
            swing_angle = self.rng.uniform(10, 45)
            if is_vert:
                swing_dir = 1 if side == "left" else -1
                orad = np.deg2rad(swing_angle * swing_dir)
                hp = (cx, start)
                op = (cx + int(door_len * 0.7 * np.sin(orad)),
                      start + int(door_len * 0.7 * np.cos(0)))
                cv2.line(img, hp, op, wc, line_th + 1, cv2.LINE_AA)

                # Probabilistic swing arc
                if self.rng.random() < self.door_swing_arc_probability:
                    arc_radius = int(door_len * 0.6)
                    if swing_dir == 1:
                        sa, ea = -int(swing_angle), 0
                    else:
                        sa, ea = 180, 180 + int(swing_angle)
                    cv2.ellipse(img, hp, (arc_radius, arc_radius),
                                0, sa, ea, wc, 1, cv2.LINE_AA)
            else:
                swing_dir = 1 if side == "top" else -1
                orad = np.deg2rad(swing_angle * swing_dir)
                hp = (start, cy)
                op = (start + int(door_len * 0.7),
                      cy + int(door_len * 0.7 * np.sin(orad)))
                cv2.line(img, hp, op, wc, line_th + 1, cv2.LINE_AA)

                # Probabilistic swing arc
                if self.rng.random() < self.door_swing_arc_probability:
                    arc_radius = int(door_len * 0.6)
                    if swing_dir == 1:
                        sa, ea = 90 - int(swing_angle), 90
                    else:
                        sa, ea = 270, 270 + int(swing_angle)
                    cv2.ellipse(img, hp, (arc_radius, arc_radius),
                                0, sa, ea, wc, 1, cv2.LINE_AA)

        return self._points_to_yolo([p1, p2])

    # ── Windows — VARIED VISUAL TYPES ─────────────────────────────────────────
    def _place_window(self, img, w):
        pad = max(20, int(self.size * 0.01))
        is_horiz = abs(w.x2 - w.x1) > abs(w.y2 - w.y1)
        bg = self.theme.background
        wc = self.theme.window_color

        # Pick window visual sub-type
        window_subtype = self.rng.choices(
            ["single", "double", "triple", "french", "slit"],
            weights=[0.30, 0.25, 0.15, 0.15, 0.15]
        )[0]

        # Size selection based on sub-type
        if window_subtype == "slit":
            length_ratios = (0.005, 0.008, 0.010, 0.015)
        elif window_subtype in ("double", "triple"):
            length_ratios = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
        elif window_subtype == "french":
            length_ratios = (0.05, 0.06, 0.07, 0.08)
        else:  # single
            length_ratios = self.window_length_ratios

        typical_lengths = [int(self.size * r) for r in length_ratios]
        line_th = self.rng.randint(1, 2)
        gap_w = self.rng.randint(1, 8)

        if is_horiz:
            axis_s, axis_e = min(w.x1, w.x2), max(w.x1, w.x2)
            center_y = (w.y1 + w.y2) // 2
            half = w.thickness // 2
            avail = axis_e - axis_s - 2 * pad
            if avail <= w.thickness * 2:
                return None

            min_win = max(int(w.thickness * 3), int(avail * 0.15))
            max_win = int(avail * 0.9)
            if min_win > max_win:
                min_win = max(10, max_win - 10)
            if min_win > max_win or max_win <= 0:
                return None

            cands = [L for L in typical_lengths if min_win <= L <= max_win]
            win_len = self.rng.choice(cands) if cands else self.rng.randint(min_win, max_win)

            off_range = avail - win_len
            if off_range < 0:
                return None

            for _ in range(10):
                start = axis_s + pad + self.rng.randint(0, max(0, off_range))
                end = start + win_len
                offset = (gap_w + line_th) // 2
                l1y, l2y = center_y - offset, center_y + offset
                y1, y2 = center_y - half, center_y + half

                if self.furniture_placer.check_collision(start, y1, end, y2, 15):
                    continue

                # Clear the wall area
                cv2.rectangle(img, (start, y1), (end, y2), bg, -1, cv2.LINE_AA)

                # Draw based on sub-type
                self._draw_window_subtype(img, window_subtype, True,
                                           start, end, l1y, l2y, y1, y2,
                                           center_y, wc, line_th)

                self.furniture_placer.add_occupied(start, y1, end, y2, "window")
                self._add_radiator(img, True, start, end, center_y, w.thickness)
                return self._points_to_yolo([(start, y1), (end, y2)])
        else:
            axis_s, axis_e = min(w.y1, w.y2), max(w.y1, w.y2)
            center_x = (w.x1 + w.x2) // 2
            half = w.thickness // 2
            avail = axis_e - axis_s - 2 * pad
            if avail <= w.thickness * 2:
                return None

            min_win = max(int(w.thickness * 3), int(avail * 0.15))
            max_win = int(avail * 0.9)
            if min_win > max_win:
                min_win = max(10, max_win - 10)
            if min_win > max_win or max_win <= 0:
                return None

            cands = [L for L in typical_lengths if min_win <= L <= max_win]
            win_len = self.rng.choice(cands) if cands else self.rng.randint(min_win, max_win)

            off_range = avail - win_len
            if off_range < 0:
                return None

            for _ in range(10):
                start = axis_s + pad + self.rng.randint(0, max(0, off_range))
                end = start + win_len
                offset = (gap_w + line_th) // 2
                l1x, l2x = center_x - offset, center_x + offset
                x1, x2 = center_x - half, center_x + half

                if self.furniture_placer.check_collision(x1, start, x2, end, 15):
                    continue

                cv2.rectangle(img, (x1, start), (x2, end), bg, -1, cv2.LINE_AA)

                self._draw_window_subtype(img, window_subtype, False,
                                           start, end, l1x, l2x, x1, x2,
                                           center_x, wc, line_th)

                self.furniture_placer.add_occupied(x1, start, x2, end, "window")
                self._add_radiator(img, False, start, end, center_x, w.thickness)
                return self._points_to_yolo([(x1, start), (x2, end)])

        return None

    def _draw_window_subtype(self, img, subtype, is_horiz,
                              start, end, l1, l2, bound1, bound2,
                              center, wc, line_th):
        """
        Draw window visual based on sub-type.
        For horiz windows: l1/l2 are y-coords of the two lines, start/end are x range.
        For vert windows: l1/l2 are x-coords, start/end are y range.
        """
        win_len = end - start

        if is_horiz:
            if subtype == "slit":
                # Single thin line in the middle
                mid_y = (l1 + l2) // 2
                cv2.line(img, (start, mid_y), (end, mid_y), wc, line_th, cv2.LINE_AA)
            elif subtype == "single":
                # Two parallel lines
                cv2.line(img, (start, l1), (end, l1), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (start, l2), (end, l2), wc, line_th, cv2.LINE_AA)
            elif subtype == "double":
                # Two parallel lines + vertical mullion in center
                cv2.line(img, (start, l1), (end, l1), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (start, l2), (end, l2), wc, line_th, cv2.LINE_AA)
                mid_x = (start + end) // 2
                cv2.line(img, (mid_x, l1), (mid_x, l2), wc, line_th, cv2.LINE_AA)
            elif subtype == "triple":
                # Two parallel lines + 2 vertical mullions (3 panes)
                cv2.line(img, (start, l1), (end, l1), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (start, l2), (end, l2), wc, line_th, cv2.LINE_AA)
                third = win_len // 3
                cv2.line(img, (start + third, l1), (start + third, l2), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (start + 2 * third, l1), (start + 2 * third, l2), wc, line_th, cv2.LINE_AA)
            elif subtype == "french":
                # Cross pattern: 2 horizontal + 1 vertical + 1 horizontal mid
                cv2.line(img, (start, l1), (end, l1), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (start, l2), (end, l2), wc, line_th, cv2.LINE_AA)
                mid_x = (start + end) // 2
                cv2.line(img, (mid_x, bound1), (mid_x, bound2), wc, line_th, cv2.LINE_AA)
                mid_y = (l1 + l2) // 2
                cv2.line(img, (start, mid_y), (end, mid_y), wc, max(1, line_th - 1), cv2.LINE_AA)
        else:
            # Vertical window — l1/l2 are x-coords, start/end are y range
            if subtype == "slit":
                mid_x = (l1 + l2) // 2
                cv2.line(img, (mid_x, start), (mid_x, end), wc, line_th, cv2.LINE_AA)
            elif subtype == "single":
                cv2.line(img, (l1, start), (l1, end), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (l2, start), (l2, end), wc, line_th, cv2.LINE_AA)
            elif subtype == "double":
                cv2.line(img, (l1, start), (l1, end), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (l2, start), (l2, end), wc, line_th, cv2.LINE_AA)
                mid_y = (start + end) // 2
                cv2.line(img, (l1, mid_y), (l2, mid_y), wc, line_th, cv2.LINE_AA)
            elif subtype == "triple":
                cv2.line(img, (l1, start), (l1, end), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (l2, start), (l2, end), wc, line_th, cv2.LINE_AA)
                third = win_len // 3
                cv2.line(img, (l1, start + third), (l2, start + third), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (l1, start + 2 * third), (l2, start + 2 * third), wc, line_th, cv2.LINE_AA)
            elif subtype == "french":
                cv2.line(img, (l1, start), (l1, end), wc, line_th, cv2.LINE_AA)
                cv2.line(img, (l2, start), (l2, end), wc, line_th, cv2.LINE_AA)
                mid_y = (start + end) // 2
                cv2.line(img, (bound1, mid_y), (bound2, mid_y), wc, line_th, cv2.LINE_AA)
                mid_x = (l1 + l2) // 2
                cv2.line(img, (mid_x, start), (mid_x, end), wc, max(1, line_th - 1), cv2.LINE_AA)

    # ── Doors — NO SWING ARC, SMALLER ANGLE ──────────────────────────────────
    def _place_door(self, img, w, occupied_spans):
        """Place a door as a rotated rectangle slab. Swing arc drawn probabilistically."""
        door_th = self.rng.randint(2, 7)
        pad = max(40, int(w.thickness * 2.5))
        is_horiz = abs(w.x2 - w.x1) > abs(w.y2 - w.y1)
        bg = self.theme.background
        wc = self.theme.door_color

        if is_horiz:
            axis_s, axis_e = min(w.x1, w.x2), max(w.x1, w.x2)
            center_y = (w.y1 + w.y2) // 2
            avail = axis_e - axis_s - 2 * pad
            if avail <= w.thickness * 2:
                return None
            door_len = int(self.size * self.rng.choice(self.door_length_ratios))
            if door_len > avail * 0.9:
                return None

            for _ in range(10):
                usable = avail - door_len
                if usable <= 0:
                    return None
                gap_s = axis_s + pad + self.rng.randint(0, usable)
                gap_e = gap_s + door_len

                if any(not (gap_e <= s or gap_s >= e) for s, e in occupied_spans):
                    continue

                half = w.thickness // 2
                y1, y2 = center_y - half, center_y + half

                # Door opening angle: small (10-45 degrees)
                swing = self.rng.uniform(10, 90)
                hinge_left = self.rng.choice([True, False])
                open_up = self.rng.choice([True, False])

                if hinge_left:
                    hp = (gap_s, center_y)
                    angle_deg = -swing if open_up else swing
                else:
                    hp = (gap_e, center_y)
                    angle_deg = 180 + swing if open_up else 180 - swing

                orad = np.deg2rad(angle_deg)
                op = (hp[0] + int(door_len * np.cos(orad)),
                      hp[1] + int(door_len * np.sin(orad)))

                perp = orad + np.pi / 2
                dx, dy = int(door_th / 2 * np.cos(perp)), int(door_th / 2 * np.sin(perp))
                corners = np.array([
                    [hp[0] - dx, hp[1] - dy], [hp[0] + dx, hp[1] + dy],
                    [op[0] + dx, op[1] + dy], [op[0] - dx, op[1] - dy],
                ], dtype=np.int32)

                all_x = [p[0] for p in corners] + [gap_s, gap_e]
                all_y = [p[1] for p in corners] + [y1, y2]
                bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

                if self.furniture_placer.check_collision(*bbox, 20):
                    continue

                occupied_spans.append((gap_s, gap_e))

                # Clear the wall gap
                cv2.rectangle(img, (gap_s, y1), (gap_e, y2), bg, -1, cv2.LINE_AA)

                # Draw door slab (filled polygon)
                cv2.fillPoly(img, [corners], bg, cv2.LINE_AA)
                cv2.polylines(img, [corners], True, wc, 1, cv2.LINE_AA)

                # Probabilistically draw swing arc
                if self.rng.random() < self.door_swing_arc_probability:
                    arc_radius = int(door_len * 0.7)
                    if open_up:
                        start_angle = 0 if hinge_left else 180 - swing
                        end_angle = swing if hinge_left else 180
                        if not hinge_left:
                            start_angle, end_angle = 180, 180 + swing
                    else:
                        if hinge_left:
                            start_angle, end_angle = 360 - swing, 360
                        else:
                            start_angle, end_angle = 180 - swing, 180

                    # Simplified: compute arc from 0 to swing degrees
                    if hinge_left and open_up:
                        sa, ea = -int(swing), 0
                    elif hinge_left and not open_up:
                        sa, ea = 0, int(swing)
                    elif not hinge_left and open_up:
                        sa, ea = 180, 180 + int(swing)
                    else:
                        sa, ea = 180 - int(swing), 180

                    cv2.ellipse(img, hp, (arc_radius, arc_radius),
                                0, sa, ea, wc, 1, cv2.LINE_AA)

                self.furniture_placer.add_occupied(*bbox, "door")
                return self._points_to_yolo([(gap_s, y1), (gap_e, y2), op])
        else:
            axis_s, axis_e = min(w.y1, w.y2), max(w.y1, w.y2)
            center_x = (w.x1 + w.x2) // 2
            avail = axis_e - axis_s - 2 * pad
            if avail <= w.thickness * 2:
                return None
            door_len = int(self.size * self.rng.choice(self.door_length_ratios))
            if door_len > avail * 0.9:
                return None

            for _ in range(10):
                usable = avail - door_len
                if usable <= 0:
                    return None
                gap_s = axis_s + pad + self.rng.randint(0, usable)
                gap_e = gap_s + door_len

                if any(not (gap_e <= s or gap_s >= e) for s, e in occupied_spans):
                    continue

                half = w.thickness // 2
                x1, x2 = center_x - half, center_x + half

                swing = self.rng.uniform(10, 45)
                hinge_top = self.rng.choice([True, False])
                open_right = self.rng.choice([True, False])

                if hinge_top:
                    hp = (center_x, gap_s)
                    angle_deg = 90 - swing if open_right else 90 + swing
                else:
                    hp = (center_x, gap_e)
                    angle_deg = 270 + swing if open_right else 270 - swing

                orad = np.deg2rad(angle_deg)
                op = (hp[0] + int(door_len * np.cos(orad)),
                      hp[1] + int(door_len * np.sin(orad)))

                perp = orad + np.pi / 2
                dx, dy = int(door_th / 2 * np.cos(perp)), int(door_th / 2 * np.sin(perp))
                corners = np.array([
                    [hp[0] - dx, hp[1] - dy], [hp[0] + dx, hp[1] + dy],
                    [op[0] + dx, op[1] + dy], [op[0] - dx, op[1] - dy],
                ], dtype=np.int32)

                all_x = [p[0] for p in corners] + [x1, x2]
                all_y = [p[1] for p in corners] + [gap_s, gap_e]
                bbox = (min(all_x), min(all_y), max(all_x), max(all_y))

                if self.furniture_placer.check_collision(*bbox, 20):
                    continue

                occupied_spans.append((gap_s, gap_e))

                cv2.rectangle(img, (x1, gap_s), (x2, gap_e), bg, -1, cv2.LINE_AA)
                cv2.fillPoly(img, [corners], bg, cv2.LINE_AA)
                cv2.polylines(img, [corners], True, wc, 1, cv2.LINE_AA)

                # Probabilistically draw swing arc
                if self.rng.random() < self.door_swing_arc_probability:
                    arc_radius = int(door_len * 0.7)
                    if hinge_top and open_right:
                        sa, ea = 90 - int(swing), 90
                    elif hinge_top and not open_right:
                        sa, ea = 90, 90 + int(swing)
                    elif not hinge_top and open_right:
                        sa, ea = 270, 270 + int(swing)
                    else:
                        sa, ea = 270 - int(swing), 270

                    cv2.ellipse(img, hp, (arc_radius, arc_radius),
                                0, sa, ea, wc, 1, cv2.LINE_AA)

                self.furniture_placer.add_occupied(*bbox, "door")
                return self._points_to_yolo([(x1, gap_s), (x2, gap_e), op])

        return None

    def _add_radiator(self, img, horizontal, axis_start, axis_end,
                      center_coord, wall_th):
        length = axis_end - axis_start
        if length < wall_th * 2 or wall_th < 4:
            return
        fill = self.theme.furniture_fill
        stroke = self.theme.furniture_stroke
        rad_len = int(length * self.rng.uniform(0.7, 0.9))
        rad_h = max(3, int(wall_th * self.rng.uniform(0.6, 0.9)))
        gap = max(3, int(wall_th * self.rng.uniform(0.2, 0.6)))

        if horizontal:
            idir = 1 if center_coord < self.size // 2 else -1
            cx = (axis_start + axis_end) // 2
            x1 = max(0, cx - rad_len // 2)
            x2 = min(self.size - 1, cx + rad_len // 2)
            if idir == 1:
                y1 = center_coord + wall_th // 2 + gap
                y2 = y1 + rad_h
            else:
                y2 = center_coord - wall_th // 2 - gap
                y1 = y2 - rad_h
            y1, y2 = max(0, y1), min(self.size - 1, y2)
            if y2 - y1 < 3 or x2 - x1 < 5:
                return
            if self.furniture_placer.check_collision(x1, y1, x2, y2):
                return
            cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), stroke, 1, cv2.LINE_AA)
            self.furniture_placer.add_occupied(x1, y1, x2, y2, "radiator")
            n_fins = self.rng.randint(4, 10)
            step = (x2 - x1) / (n_fins + 1)
            for k in range(1, n_fins + 1):
                fx = int(x1 + step * k)
                cv2.line(img, (fx, y1 + 1), (fx, y2 - 1), stroke, 1, cv2.LINE_AA)
        else:
            idir = 1 if center_coord < self.size // 2 else -1
            cy = (axis_start + axis_end) // 2
            y1 = max(0, cy - rad_len // 2)
            y2 = min(self.size - 1, cy + rad_len // 2)
            if idir == 1:
                x1 = center_coord + wall_th // 2 + gap
                x2 = x1 + rad_h
            else:
                x2 = center_coord - wall_th // 2 - gap
                x1 = x2 - rad_h
            x1, x2 = max(0, x1), min(self.size - 1, x2)
            if x2 - x1 < 3 or y2 - y1 < 5:
                return
            if self.furniture_placer.check_collision(x1, y1, x2, y2):
                return
            cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), stroke, 1, cv2.LINE_AA)
            self.furniture_placer.add_occupied(x1, y1, x2, y2, "radiator")
            n_fins = self.rng.randint(4, 10)
            step = (y2 - y1) / (n_fins + 1)
            for k in range(1, n_fins + 1):
                fy = int(y1 + step * k)
                cv2.line(img, (x1 + 1, fy), (x2 - 1, fy), stroke, 1, cv2.LINE_AA)

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _points_to_yolo(self, pts, pad=0):
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

    def _apply_rotation(self, img, labels):
        size = img.shape[0]
        angle = self.rng.uniform(-15, 15)
        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        bg = self.theme.background
        rotated = cv2.warpAffine(img, M, (size, size), borderValue=bg)
        a_rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(a_rad), np.sin(a_rad)
        new_labels = []
        for cls_id, cx, cy, w, h in labels:
            cx_px, cy_px = cx * size, cy * size
            xr, yr = cx_px - center[0], cy_px - center[1]
            x_rot = xr * cos_a - yr * sin_a + center[0]
            y_rot = xr * sin_a + yr * cos_a + center[1]
            new_labels.append((cls_id, x_rot / size, y_rot / size, w, h))
        return rotated, new_labels


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SmartFloorPlanGenerator v5.0 — Professional Procedural Engine"
    )
    parser.add_argument("--output", type=str, default="smart_dataset")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--super-sampling", type=int, default=2)
    parser.add_argument("--style", type=str, default="modern_minimal",
                        choices=[s.value for s in StylePreset])
    parser.add_argument("--building-type", type=str, default="apartment",
                        choices=[b.value for b in BuildingType])
    parser.add_argument("--multi-style", action="store_true")
    parser.add_argument("--style-mode", type=str, default="random",
                        choices=["random", "cycle", "weighted"])
    parser.add_argument("--style-themes", nargs="+", default=None)
    parser.add_argument("--style-weights", nargs="+", type=float, default=None)
    parser.add_argument("--vary-all", action="store_true")
    parser.add_argument("--export", nargs="+", default=["png"],
                        choices=["png", "svg", "dxf", "json"])
    parser.add_argument("--strategies", nargs="+",
                        default=["central", "linear", "radial", "bsp",
                                 "voronoi", "graph", "open_plan",
                                 "asymmetric", "courtyard"])
    parser.add_argument("--min-rooms", type=int, default=3)
    parser.add_argument("--max-rooms", type=int, default=6)
    parser.add_argument("--multi-story", action="store_true")
    parser.add_argument("--floors", type=int, default=1)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--solver", type=str, default="proportional",
                        choices=["proportional", "pulp"])
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--color-jitter", type=int, default=12,
                        help="Per-image color jitter intensity (0-40)")

    args = parser.parse_args()

    if args.config:
        config = FloorPlanConfig.from_yaml(args.config)
    else:
        config = FloorPlanConfig(
            style=args.style,
            building_type=args.building_type,
            max_rooms=args.max_rooms,
            min_rooms=args.min_rooms,
            palette_name=args.style,
            export_formats=args.export,
            img_size=args.size,
            super_sampling=args.super_sampling,
            seed=args.seed,
            count=args.count,
            output_dir=args.output,
            strategies=args.strategies,
            multi_story=args.multi_story,
            num_floors=args.floors,
            constraint_solver=args.solver,
            generate_metadata_json=not args.no_metadata,
            multi_style=args.multi_style,
            style_rotation_mode=args.style_mode,
            style_themes=args.style_themes,
            style_weights=args.style_weights,
            vary_strategies=args.vary_all or args.multi_style,
            vary_wall_thickness=args.vary_all,
            vary_room_counts=args.vary_all,
            vary_balcony=args.vary_all,
            color_jitter_intensity=args.color_jitter,
        )

    gen = SmartFloorPlanGenerator(config=config)
    gen.generate_dataset()

    print(f"✓ Generated {config.count} floor plans → {config.output_dir}/")
    print(f"  Theme: {config.palette_name}")
    print(f"  Strategies: {', '.join(config.strategies)}")
    print(f"  Formats: {', '.join(str(f) for f in config.export_formats)}")
    if config.multi_story:
        print(f"  Floors: {config.num_floors}")


if __name__ == "__main__":
    main()