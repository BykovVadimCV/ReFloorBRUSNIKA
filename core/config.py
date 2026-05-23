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

    # Per-image cap on door/gap opening widths: max_allowed = multiplier ×
    # median detected door width. 1.3 leaves room for legitimate wider
    # entries (French doors) while suppressing over-extended geometric gaps.
    opening_max_door_multiplier: float = 1.3

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
    unet_min_door_area: int = 600
    unet_min_window_area: int = 300

    # Gap-snap search margin for U-Net door bboxes (pixels)
    unet_door_match_margin: int = 40

    # Wall-interruption splitting: fraction of bbox span that must be wall
    # pixels in a row/column for it to count as a "wall band" dividing the door
    unet_door_wall_band_ratio: float = 0.50

    # Minimum wall band thickness (pixels) to trigger a split
    unet_door_min_wall_band_px: int = 5

    # Minimum area (pixels) for a post-split sub-bbox to be kept
    unet_door_min_piece_area: int = 200

    # Minimum thickness of the short side (pixels) for a post-split piece;
    # filters hair-thin slivers that are not real door openings
    unet_door_min_piece_thickness_px: int = 12

    # ============================================================
    # DOOR RECT DECOMPOSITION  (U-Net door mask → precise bboxes)
    # ============================================================

    # Enable rect-decompose pass on the U-Net door mask.
    # When True, the raw blob bboxes are replaced by tight decomposed rects.
    enable_door_rect_decompose: bool = True

    # Coverage stop: greedy decomposer runs until this fraction of door-mask
    # pixels are covered.  70% is intentionally low — doors are small and a
    # single well-placed rectangle usually suffices.
    door_rect_coverage_stop: float = 0.70

    # Maximum fraction of a door rect's pixels that may bleed outside the
    # door mask (tight: 2% noise-only tolerance).
    door_rect_max_spill: float = 0.02

    # Number of angle steps tested per greedy iteration.
    # Doors are always axis-aligned, so we only need 0°/90°.
    door_rect_axis_only: bool = True

    # Remaining rect_decompose tunables for doors (generous defaults —
    # doors are small so the decomposer is fast regardless).
    door_rect_penalty:              float = 0.0
    door_rect_max_grid_dim:         int   = 200
    door_rect_bleed_weight:         float = 15.0
    door_rect_initial_bleed_weight: float = 15.0
    door_rect_bleed_decay:          float = 0.0
    door_rect_max_overlap:          float = 0.5

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

    # ============================================================
    # RECTANGLE-DECOMPOSITION WALL DETECTION  (new pipeline)
    # ============================================================

    # Master switch — when True, use the rect-decomposition wall detector
    # (epoch_20.pth + rect_decompose) instead of the legacy color/hollow
    # path. The legacy path remains available for fallback.
    use_rect_walls: bool = True

    # Binary wall U-Net checkpoint
    rect_unet_ckpt_path: str = "weights/epoch_20.pth"
    rect_unet_img_size:  int = 1024

    # Diagonal-presence test (LSD on the wall mask)
    rect_axis_tol_deg:    float = 3.0     # angles within ±this of 0/90 = axis
    rect_diag_lsd_ratio:  float = 0.05    # fraction of off-axis LSD length
    rect_diag_lsd_min_len: float = 25.0   # ignore LSD segments shorter than this

    # rect_decompose parameters (defaults requested by the user)
    rect_angle_steps:          int   = 12
    rect_penalty:              float = 0.0
    rect_max_grid_dim:         int   = 200
    rect_coverage_stop:        float = 0.93
    rect_bleed_weight:         float = 15.0
    rect_initial_bleed_weight: float = 15.0
    rect_bleed_decay:          float = 0.0
    rect_axis_gap:             float = 0.0
    rect_max_overlap:          float = 1.0

    # Endpoint snapping (tiny nudge to close perimeter — NOT wall merging)
    rect_snap_gap_factor:       float = 0.3   # gap ≤ factor × thickness (was 2.0)
    rect_snap_gap_floor:        float = 5.0   # but never below this many pixels
    rect_snap_angle_deg:        float = 5.0   # walls within this angle are "parallel" → skip
    rect_snap_max_extend_frac:  float = 0.10  # max endpoint shift as fraction of wall length

    # Hard spill cap: any candidate rect whose non-wall-pixel fraction exceeds
    # this is rejected regardless of its score.  0.15 = max 15% bleed.
    rect_max_spill: float = 0.15

    # Cap (morph-close) kernel size applied to the raw U-Net wall mask
    # before enclosed-space analysis and rect decomposition.  Bridges small
    # gaps in wall coverage so the decomposer sees a connected mask.
    rect_cap_kernel_size: int = 9
    rect_cap_iters:       int = 2

    # Enclosed-space refinement: fraction of an enclosed region's pixels that
    # must overlap the wall mask for the region to be filled as wall.
    # 0.25 matches test_cap.py default (was 0.50).
    rect_enclosed_overlap_threshold: float = 0.25

    # Skeleton cap enclosed-space fill: maximum area (pixels) of a background
    # region that the cap algorithm is allowed to fill as wall.
    # Rooms are 5 000–100 000+ px; genuine construction gaps are tiny.
    # The fill also requires the region to be NEWLY enclosed (not pre-existing
    # room interior) and to contain no numeric OCR label.
    cap_fill_max_area_px: int = 5000

    # Door + OCR-text wall-mask filter (mirrors test_cap.py)
    # When True, after the cap step the wall mask is filtered to remove
    # door/window pixel regions (from epoch_040, dilated by door_margin and
    # with wall pixels preserved) and OCR text bbox regions (dilated by
    # text_margin).  The filtered mask is what rect_decompose sees.
    enable_wall_door_text_filter: bool = True
    wall_filter_door_margin: int = 10
    wall_filter_text_margin: int = 5
    wall_filter_include_windows: bool = True

    # Door/window diagonal filter — openings whose parent wall has angle
    # deviating from 0°/90° by more than this are dropped (see #1 in spec:
    # diagonals not allowed for openings).
    opening_axis_tol_deg: float = 3.0

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
