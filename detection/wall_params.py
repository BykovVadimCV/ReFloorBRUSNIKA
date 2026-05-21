"""Constants and thresholds used by the wall detectors."""

from typing import Tuple


class WallParams:
    """Centralized thresholds, kernel sizes, and tunables for wall detection."""

    # Morphology
    MORPH_KERNEL_SIZE: int = 3
    MORPH_CLOSE_ITERATIONS: int = 2
    MORPH_OPEN_ITERATIONS: int = 1
    MORPH_COMPONENT_MIN_AREA: int = 5

    # HSV color tolerance
    HSV_H_TOLERANCE: int = 4
    HSV_S_TOLERANCE: int = 15
    HSV_V_TOLERANCE: int = 12

    # Rectangle edge matching
    EDGE_MATCH_MIN_FRACTION: float = 0.5

    # Rectangle decomposition
    RECT_VISUAL_GAP_DEFAULT: int = 3

    # Auto wall-color detection via K-means
    KMEANS_K: int = 8
    KMEANS_MAX_SAMPLES: int = 2_000_000
    KMEANS_MIN_FRACTION: float = 0.02
    KMEANS_ITERATIONS: int = 10
    KMEANS_EPSILON: float = 1.0

    # Auto: brightness filtering in LAB L*
    AUTO_L_MIN: int = 15          # exclude near-black (text, thin lines)
    AUTO_L_MAX: int = 210         # exclude near-white (background, floor fill)

    # Auto: spatial coherence
    AUTO_COHERENCE_MIN_COMPONENT_AREA: int = 500
    AUTO_COHERENCE_WEIGHT: float = 0.6
    AUTO_AREA_WEIGHT: float = 0.4

    # Auto: adaptive HSV tolerances
    AUTO_HSV_H_TOLERANCE: int = 6
    AUTO_HSV_S_TOLERANCE: int = 25
    AUTO_HSV_V_TOLERANCE: int = 20
    AUTO_RGB_TOLERANCE: int = 18

    # OCR cleanup
    OCR_INPAINT_RADIUS_DEFAULT: int = 3
    OCR_BBOX_EXPANSION_DEFAULT: int = 2
    OCR_DILATE_MIN_KERNEL: int = 1

    # Circle-marker removal via Hough
    HOUGH_DP: int = 1
    HOUGH_MIN_DIST: int = 20
    HOUGH_PARAM1: int = 50
    HOUGH_PARAM2: int = 20
    HOUGH_MIN_RADIUS: int = 25
    HOUGH_MAX_RADIUS: int = 100
    HOUGH_CIRCLE_EXPANSION: int = 5
    HOUGH_BLUR_KERNEL: Tuple[int, int] = (5, 5)

    # Geometric constraints
    RECT_STACK_SENTINEL: int = 0

    # Structural detection
    STRUCT_MIN_LINE_LENGTH: int = 5
    STRUCT_HOUGH_THRESHOLD: int = 30
    STRUCT_HOUGH_MAX_GAP: int = 20
    STRUCT_CANNY_LOW: int = 20
    STRUCT_CANNY_HIGH: int = 100
    STRUCT_CANNY_EDGES_LOW: int = 10
    STRUCT_CANNY_EDGES_HIGH: int = 60
    STRUCT_DILATE_KERNEL: int = 3
    STRUCT_DILATE_ITERATIONS: int = 1
    STRUCT_MIN_WALL_AREA_RATIO: float = 0.0001
    ADAPTIVE_BLOCK_SIZE: int = 11
    ADAPTIVE_C: int = 2

    # Hybrid / auto fusion
    HYBRID_COLOR_COVERAGE_THRESHOLD: float = 0.02
    AUTO_COLOR_COVERAGE_THRESHOLD: float = 0.015
    AUTO_FUSION_CLOSE_KERNEL: int = 3

    # Furniture filtering
    STRUCT_MIN_LINE_RATIO: float = 0.005
    FURNITURE_MIN_ASPECT_RATIO: float = 1.2
    MAX_FURNITURE_AREA_RATIO: float = 0.005
    ANGLE_TOLERANCE_DEG: float = 5.0
    FURNITURE_MAX_FILL_RATIO: float = 0.70
    RECT_MIN_ASPECT_RATIO: float = 1.5
    RECT_MIN_AREA_RATIO: float = 0.0003
    RECT_CONNECTIVITY_MARGIN_PX: int = 15

    # Parallel-wall detection
    PARALLEL_DISTANCE_MIN_PX: int = 4
    PARALLEL_DISTANCE_MAX_PX: int = 30
    PARALLEL_ANGLE_TOLERANCE_DEG: float = 3.0

    # Interior-object removal
    ROOM_FLOOD_MIN_AREA_RATIO: float = 0.005
    ROOM_FLOOD_ERODE_SIZE: int = 3

    # Hybrid fallback
    STRUCT_MAX_COMPONENTS_BEFORE_FALLBACK: int = 200

    # Outline / BTI-style walls
    CLASSIFIER_BLACK_GRAY_THRESH: int = 80
    CLASSIFIER_OUTLINE_BLACK_RATIO_MIN: float = 0.005
    CLASSIFIER_OUTLINE_BLACK_RATIO_MAX: float = 0.18
    CLASSIFIER_FILLED_COVERAGE_MIN: float = 0.015
    CLASSIFIER_THIN_COMPONENT_RATIO_MIN: float = 0.25

    OUTLINE_BLACK_THRESH: int = 90
    OUTLINE_ADAPTIVE_BLOCK: int = 25
    OUTLINE_ADAPTIVE_C: int = 10
    OUTLINE_USE_ADAPTIVE: bool = True

    # Text/digit filtering
    TEXT_FILTER_MAX_AREA_RATIO: float = 0.0015
    TEXT_FILTER_MIN_HEIGHT_RATIO: float = 0.005
    TEXT_FILTER_MAX_HEIGHT_RATIO: float = 0.04
    TEXT_FILTER_MAX_ASPECT: float = 4.0
    TEXT_FILTER_MIN_FILL: float = 0.15
    TEXT_FILTER_MAX_FILL: float = 0.85
    TEXT_FILTER_CLUSTER_GAP_PX: int = 15
    TEXT_FILTER_MIN_CLUSTER_SIZE: int = 2

    # Dimension-line filtering
    DIMLINE_MAX_THIN_DIM_PX: int = 3
    DIMLINE_MIN_ASPECT: float = 8.0
    DIMLINE_MAX_AREA_RATIO: float = 0.002
    DIMLINE_CONNECTIVITY_RADIUS_PX: int = 5

    # Outline-wall fill
    OUTLINE_FILL_CLOSE_KERNEL: int = 9
    OUTLINE_FILL_DIRECTIONAL_KERNEL_LENGTH: int = 11
    OUTLINE_FILL_DIRECTIONAL_KERNEL_WIDTH: int = 3
    OUTLINE_FILL_CLOSE_ITERATIONS: int = 2
    OUTLINE_MIN_WALL_AREA_RATIO: float = 0.0003

    # Outline detector final cleanup
    OUTLINE_FINAL_OPEN_KERNEL: int = 3
    OUTLINE_FINAL_OPEN_ITERATIONS: int = 1
    OUTLINE_MIN_FILLED_COMPONENT_RATIO: float = 0.0005
    OUTLINE_MIN_RECTANGULARITY: float = 0.3
    OUTLINE_ROOM_HOLE_MIN_RATIO: float = 0.003
