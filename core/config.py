"""
Centralized configuration for the ReFloorBRUSNIKA pipeline.

All thresholds, tolerances, scales, and feature flags are defined here.
Eliminates scattered magic numbers across detection and export modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """
    Master configuration for the floorplan processing pipeline.

    All numeric constants previously scattered across Theme classes
    and module-level variables are consolidated here.
    """

    # ============================================================
    # SCALE
    # ============================================================

    # Pixels to centimeters conversion factor (may be updated by OCR calibration)
    pixels_to_cm: float = 2.54

    # Standard apartment wall height in centimeters
    wall_height_cm: float = 243.84

    # Scale factor for SweetHome3D export
    sh3d_scale_factor: float = 2.2

    # ============================================================
    # WALL DETECTION
    # ============================================================

    # Brusnika-specific wall color (hex, without '#')
    wall_color_hex: str = "8E8780"

    # BGR values for wall color matching (OpenCV uses BGR)
    wall_color_bgr: tuple = (128, 135, 142)  # BGR order for #8E8780

    # Auto-detect wall color from image (overrides wall_color_hex)
    auto_wall_color: bool = True

    # Tolerance for color-based wall detection (RGB per-channel ±)
    color_tolerance: int = 15

    # Minimum wall thickness in pixels to be considered structural
    min_wall_thickness_px: float = 5.0

    # Furniture/decoration color (excluded from walls)
    furniture_color_hex: str = "E2DFDD"

    # Morphological kernel sizes for wall detection
    morph_close_kernel_size: int = 7
    morph_open_kernel_size: int = 3

    # Minimum area (pixels) for a wall rectangle to be considered valid
    min_wall_area_px: int = 100

    # ============================================================
    # YOLO DETECTION
    # ============================================================

    # Confidence threshold for YOLO detections
    yolo_confidence: float = 0.25

    # IoU threshold for YOLO NMS
    yolo_iou_threshold: float = 0.45

    # Maximum fraction of image area a single YOLO detection may occupy
    # (filters out whole-corridor false positives)
    yolo_max_area_fraction: float = 0.05

    # ============================================================
    # OPENING DETECTION
    # ============================================================

    # Minimum opening width in meters
    min_opening_width_m: float = 0.25

    # Maximum opening width in meters
    max_opening_width_m: float = 1.65

    # Minimum opening width in pixels (derived from meters at default scale)
    @property
    def min_opening_width_px(self) -> float:
        return (self.min_opening_width_m * 100) / self.pixels_to_cm

    # Maximum opening width in pixels
    @property
    def max_opening_width_px(self) -> float:
        return (self.max_opening_width_m * 100) / self.pixels_to_cm

    # Gap detection: white content ratio threshold (ratio of white pixels in gap)
    gap_white_ratio_threshold: float = 0.30

    # Gap detection: minimum overlap with wall to be considered valid
    gap_min_wall_touch_ratio: float = 0.3

    # Gap deduplication: merge gaps closer than this (pixels)
    gap_merge_distance_px: float = 20.0

    # ============================================================
    # DOOR DETECTION
    # ============================================================

    # Default door height in centimeters
    door_height_cm: float = 208.5975

    # Enable door arc detection
    enable_door_arcs: bool = True

    # Minimum door arc radius in pixels
    min_arc_radius_px: float = 15.0

    # Maximum door arc radius in pixels
    max_arc_radius_px: float = 200.0

    # Minimum arc angle span in degrees to consider a valid door arc
    min_arc_angle_span_deg: float = 60.0

    # ============================================================
    # WINDOW DETECTION
    # ============================================================

    # Default window height in centimeters
    window_height_cm: float = 173.98999

    # Window elevation above floor in centimeters
    window_elevation_cm: float = 46.989998

    # ============================================================
    # TOLERANCES & ALIGNMENT
    # ============================================================

    # Wall endpoint snapping distance (cm)
    snap_tolerance_cm: float = 40.0

    # Axis snapping tolerance (cm) - align walls to shared axes
    axis_snap_tolerance_cm: float = 15.0

    # T-junction snapping tolerance (cm)
    t_junction_snap_tolerance_cm: float = 30.0

    # Search radius for finding parent wall of an opening (cm)
    opening_wall_proximity_cm: float = 50.0

    # Wall alignment tolerance in pixels
    wall_alignment_tolerance_px: float = 5.0

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    # IoU threshold for duplicate detection removal
    iou_threshold: float = 0.30

    # Maximum gap (pixels) between parallel boxes to consider as duplicates
    parallel_gap_px: float = 6.0

    # ============================================================
    # ROOM DETECTION
    # ============================================================

    # Enable room area measurement via OCR
    enable_room_measurement: bool = True

    # Minimum room area in pixels to be considered a valid room
    min_room_area_px: int = 1000

    # Flood fill border step for exterior detection
    flood_fill_border_step: int = 10

    # ============================================================
    # EXPORT
    # ============================================================

    # Wall overlap at junctions (cm)
    wall_overlap_cm: float = 2.0

    # Wall extension factor (fraction of wall length added at each end)
    wall_extension_factor: float = 0.05

    # Centerline search radius for opening-to-wall matching (cm)
    centerline_search_radius_cm: float = 150.0

    # Opening axis snap tolerance (cm)
    opening_axis_snap_tolerance_cm: float = 30.0

    # Opening extension tolerance (cm)
    opening_extension_tolerance_cm: float = 30.0

    # Minimum wall thickness in cm for export
    min_wall_thickness_cm: float = 2.0

    # ============================================================
    # FEATURE FLAGS
    # ============================================================

    # Enable debug output and visualizations
    debug_mode: bool = False

    # Enable door arc visualization in summary image
    debug_arcs: bool = False

    # Wall detection mode: "color" | "structure" | "outline" | "hybrid" | "auto"
    # "auto" classifies each image and picks color/outline/structure accordingly.
    wall_mode: str = "auto"

    # Structure detection method: "lines" | "edges" | "contours"
    structure_method: str = "lines"

    # ============================================================
    # PATHS
    # ============================================================

    # Path to YOLO weights file
    yolo_weights_path: str = "weights/best.pt"

    # YOLO device: "cpu" | "cuda" | "0" | "1" etc.
    yolo_device: str = "cpu"

    # ============================================================
    # U-NET DETECTION
    # ============================================================

    # Path to U-Net checkpoint (.pth)
    unet_checkpoint_path: str = "weights/epoch_040.pth"

    # U-Net model input size
    unet_img_size: int = 1024

    # U-Net confidence threshold (softmax probability)
    unet_confidence: float = 0.5

    # U-Net device: None = auto, "cpu", "cuda"
    unet_device: Optional[str] = None

    # Enable U-Net detection (outline-style fallback).
    # Disabled by default: U-Net inference can trigger 0xC000001D
    # (Illegal Instruction) on some CPU/library combinations.
    enable_unet: bool = False

    # Minimum areas for U-Net bbox extraction
    unet_min_wall_area: int = 100
    unet_min_door_area: int = 300
    unet_min_window_area: int = 300

    # ============================================================
    # PLAUSIBILITY VALIDATION
    # ============================================================

    min_plausible_wall_thickness_cm: float = 5.0
    max_plausible_wall_thickness_cm: float = 40.0
    min_plausible_wall_length_cm: float = 10.0
    min_plausible_opening_width_cm: float = 40.0
    max_plausible_opening_width_cm: float = 200.0
    max_door_depth_cm: float = 15.0
    max_window_depth_cm: float = 15.0
    room_area_tolerance_ratio: float = 0.30
    room_gap_close_max_cm: float = 15.0

    # Fraction of U-Net wall mask pixels that must be non-white in the
    # original image to classify walls as "solid" (filled) rather than
    # "hollow" (double-line outlines).  When the fraction exceeds this
    # threshold the grey-fill pipeline is skipped and direct color
    # detection is used instead.  Range 0.0–1.0.
    solid_wall_threshold: float = 0.50

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        assert self.pixels_to_cm > 0, "pixels_to_cm must be positive"
        assert self.wall_height_cm > 0, "wall_height_cm must be positive"
        assert 0 < self.yolo_confidence < 1, "yolo_confidence must be in (0, 1)"
        assert self.min_opening_width_m < self.max_opening_width_m, (
            "min_opening_width must be less than max_opening_width"
        )

    @classmethod
    def from_dict(cls, d: dict) -> PipelineConfig:
        """Create a config from a dictionary (e.g., from CLI args)."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def copy_with(self, **kwargs) -> PipelineConfig:
        """Create a copy of this config with specific fields overridden."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)

    @property
    def pixels_to_m(self) -> float:
        """Conversion factor: pixels to meters."""
        return self.pixels_to_cm / 100.0

    @property
    def m_to_pixels(self) -> float:
        """Conversion factor: meters to pixels."""
        return 100.0 / self.pixels_to_cm if self.pixels_to_cm > 0 else 0.0
