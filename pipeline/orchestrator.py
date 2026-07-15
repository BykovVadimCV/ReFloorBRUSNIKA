"""FloorplanPipeline class — orchestrates wall/opening/room detection and SH3D export."""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.config import PipelineConfig
from core.geometry import iou_xyxy
from core.ocr_utils import (
    is_numeric_ocr_label as _is_numeric_ocr_label,
    parse_area_m2 as _parse_ocr_area_m2,
    parse_length_m as _parse_ocr_length_m,
)
from core.models import (
    BBox,
    DetectionResult,
    Opening,
    OpeningType,
    Room,
    WallSegment,
)
from detection.walls import (
    ColorWallDetector,
    refine_wall_mask_by_enclosed_regions,
    snap_walls,
    wall_rectangle_to_segment,
)
from detection.openings import (
    OpeningPipeline,
    average_wall_thickness,
    filter_back_to_back_doors,
)
from floorplan_export.sh3d import SH3DExporter
from floorplan_export.visualization import FloorplanVisualizer

logger = logging.getLogger(__name__)

from pipeline.preprocess import (
    _deskew_image,
    _crop_to_floorplan,
    _upscale_to_min_long_side,
    _pad_image,
)
from pipeline.scale_calibration import (
    _calibrate_scale_from_wall_lengths,
    _polygon_area_from_tuples,
    _point_in_polygon,
    _validate_room_areas,
)
from pipeline.wall_cleanup import cleanup_walls
from pipeline.unet_subprocess import _unet_worker, _run_unet_subprocess
from pipeline.result import PipelineResult

logger = logging.getLogger(__name__)


class FloorplanPipeline:
    """
    Clean, linear pipeline for floorplan processing.

    Each stage runs once and passes results forward.
    No redundant detection calls. Explicit error handling per stage.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

        # Stage 1: Wall detection
        self.wall_detector = ColorWallDetector(self.config)

        # Stage 2: Opening detection
        self.opening_detector = OpeningPipeline(self.config)

        # Stage 3: Room measurement (lazy init, requires OCR)
        self._room_analyzer: Optional[Any] = None

        # Stage 4: Export
        self.exporter = SH3DExporter(self.config)

        # Stage 5: Visualization
        self.visualizer = FloorplanVisualizer(self.config.pixels_to_cm)

        # YOLO detector (shared with opening pipeline)
        self._yolo: Optional[Any] = None

        self._initialized = False

    def initialize(self) -> None:
        """
        Load all models (YOLO, OCR).

        Call this once before processing images to avoid reloading
        models on each call.
        """
        if self._initialized:
            return

        # Initialize OCR
        ocr_model = self._init_ocr()

        # Initialize wall detector with shared OCR model
        self.wall_detector.initialize(ocr_model=ocr_model)

        # Initialize YOLO
        if Path(self.config.yolo_weights_path).exists():
            try:
                from detection.furnituredetector import YOLODoorWindowDetector
                self._yolo = YOLODoorWindowDetector(
                    model_path=self.config.yolo_weights_path,
                    conf_threshold=self.config.yolo_confidence,
                    iou_threshold=self.config.yolo_iou_threshold,
                    device=self.config.yolo_device,
                    max_door_area_fraction=self.config.yolo_max_area_fraction,
                    window_conf_threshold=getattr(
                        self.config, "yolo_window_confidence", None),
                    door_conf_threshold=getattr(
                        self.config, "yolo_door_confidence", None),
                )
                logger.info("YOLO model loaded from %s", self.config.yolo_weights_path)
            except Exception as e:
                logger.warning("YOLO initialization failed (non-fatal): %s", e)
        else:
            logger.warning("YOLO weights not found at %s", self.config.yolo_weights_path)

        # Initialize opening detector with YOLO
        self.opening_detector.initialize(yolo_model=self._yolo)

        # Initialize room analyzer
        if self.config.enable_room_measurement:
            try:
                from detection.space import RoomSizeAnalyzer
                self._room_analyzer = RoomSizeAnalyzer(
                    edge_distance_ratio=0.15,
                )
                # Reuse OCR model to avoid loading it twice
                if ocr_model is not None:
                    self._room_analyzer.ocr_model = ocr_model
            except Exception as e:
                logger.warning("Room analyzer initialization failed: %s", e)

        self._initialized = True
        logger.info("Pipeline initialized")

    def process(
        self,
        image_path: str,
        output_dir: str,
        progress_callback: Optional[Callable] = None,
    ) -> PipelineResult:
        """
        Process a single floorplan image through all pipeline stages.

        Args:
            image_path: Path to input image file
            output_dir: Directory for output files
            progress_callback: Optional callback ``f(stage_name: str)``
                called at the start of each pipeline stage so external
                progress bars can be updated in real time.

        Returns:
            PipelineResult with all detection results and output paths
        """
        def _notify(stage: str) -> None:
            if progress_callback is not None:
                progress_callback(stage)

        # _is_numeric_ocr_label is imported at module level from core.ocr_utils

        if not self._initialized:
            self.initialize()

        base_name = Path(image_path).stem
        os.makedirs(output_dir, exist_ok=True)

        result = PipelineResult()

        # ── Pre-stage 0a: crop to floorplan extent ────────────────────
        # This is the ONLY step that runs before the diagonal pre-processor.
        # It trims legend / margins so that subsequent U-Net and LSD stages
        # work on a clean, tightly-bounded floorplan.
        _notify("crop to floorplan")
        _raw_img = cv2.imread(image_path)
        if _raw_img is None:
            raise ValueError(f"Cannot read input image: {image_path}")
        # ── Pre-stage 0 (pre-everything): upscale small inputs ────────────
        # Any image whose longest side is below the threshold is enlarged
        # (aspect-preserving) before crop / deskew / OCR / U-Net so detection
        # has enough resolution to work with.  Scale calibration runs later on
        # this same enlarged image, so metric measurements stay consistent.
        _raw_img = _upscale_to_min_long_side(
            _raw_img,
            getattr(self.config, "upscale_min_long_side_px", 1200),
        )
        _cropped_img = _crop_to_floorplan(_raw_img)
        if _cropped_img.shape != _raw_img.shape:
            logger.info(
                "[%s] Pre-stage 0a: cropped %dx%d → %dx%d",
                base_name,
                _raw_img.shape[1], _raw_img.shape[0],
                _cropped_img.shape[1], _cropped_img.shape[0],
            )
        del _raw_img

        # ── Pre-stage 0a2: pad floorplan with a white border ──────────
        # Runs AFTER crop on purpose: crop trims to the ink bbox, so any margin
        # added earlier would just be trimmed straight back off.  Framing the
        # cropped drawing in white gives perimeter walls background context on
        # both sides so the U-Net stops dropping them (see input 12).  Mirrors
        # the deskew step: the padded image is physically resaved to
        # <input_dir>/padded/<basename> and that pixel buffer feeds everything
        # downstream.
        _cropped_img = _pad_image(
            _cropped_img,
            ratio=getattr(self.config, "pad_border_ratio", 0.06),
            min_px=getattr(self.config, "pad_border_min_px", 40),
        )
        try:
            _pad_dir = os.path.join(
                os.path.dirname(image_path) or ".", "padded")
            os.makedirs(_pad_dir, exist_ok=True)
            _pad_path = os.path.join(_pad_dir, os.path.basename(image_path))
            cv2.imwrite(_pad_path, _cropped_img)
            logger.info("[%s] Pre-stage 0a2: padded image saved → %s",
                        base_name, _pad_path)
        except Exception as _pad_exc:
            logger.warning("[%s] Pre-stage 0a2: could not save padded image: %s",
                           base_name, _pad_exc)

        # ── Pre-stage 0b: diagonal pre-processor ──────────────────────
        # Handles small-skew correction, OCR, U-Net, LSD clustering, and
        # deskewing.  Writes the result to <input_dir>/deskewed/<basename>.
        _notify("diagonal pre-processor (OCR + U-Net + deskew)")
        try:
            from detection.diagonal import preprocess_floorplan
        except Exception as _imp_exc:
            raise RuntimeError(
                "detection.diagonal.preprocess_floorplan unavailable: "
                f"{_imp_exc}"
            )

        unet_ckpt = getattr(self.config, 'unet_ckpt_path',
                             'weights/epoch_040.pth')
        diag_out_dir = os.path.join(
            os.path.dirname(image_path) or ".", "deskewed")
        diag = preprocess_floorplan(
            image_path=image_path,
            unet_ckpt_path=unet_ckpt,
            output_dir=diag_out_dir,
            _preloaded_img=_cropped_img,
        )
        del _cropped_img

        deskewed_path = diag["deskewed_path"]
        ocr_text_labels_early = list(diag.get("normal_labels")  or [])
        rotated_ocr_labels    = list(diag.get("rotated_labels") or [])
        deskew_angle          = diag.get("deskew_angle")

        logger.info(
            "[%s] Pre-stage 0: deskewed_path=%s deskew=%s "
            "normal_labels=%d rotated_labels=%d",
            base_name, deskewed_path,
            f"{deskew_angle:.3f}deg" if deskew_angle is not None else "none",
            len(ocr_text_labels_early), len(rotated_ocr_labels),
        )

        # All downstream stages operate on the pre-processed image.
        img = cv2.imread(deskewed_path)
        if img is None:
            raise ValueError(
                f"Cannot load pre-processed image: {deskewed_path}")
        h_img, w_img = img.shape[:2]

        # ── Pre-stage: text erasure (uses the OCR labels above) ───────
        # Erase ONLY numeric text regions from a copy of `img` so that text
        # doesn't pollute wall detection. Non-numeric labels (room names like
        # "Кухня", "Ванная") are left in place.  No OCR pass — labels come
        # straight from the pre-processor.
        logger.info(
            "[%s] Pre-stage: text erasure (numeric labels only)",
            base_name,
        )
        img_clean = img.copy()
        shrink = 3  # inset each OCR bbox by this many pixels before erasing
        n_erased = 0
        for _src in (ocr_text_labels_early, rotated_ocr_labels):
            for item in _src:
                if len(item) < 5:
                    continue
                # Only erase numeric labels (area measurements)
                if not _is_numeric_ocr_label(str(item[0])):
                    continue
                rx1, ry1 = int(item[1]) + shrink, int(item[2]) + shrink
                rx2, ry2 = int(item[3]) - shrink, int(item[4]) - shrink
                if rx2 <= rx1 or ry2 <= ry1:
                    continue  # bbox too small after shrink — skip
                cv2.rectangle(
                    img_clean,
                    (max(0, rx1), max(0, ry1)),
                    (min(w_img, rx2), min(h_img, ry2)),
                    (255, 255, 255), -1,
                )
                n_erased += 1

        logger.info(
            "[%s] Erased %d numeric text regions before wall detection",
            base_name, n_erased,
        )

        # ── Stage 0: Wall detection ─────────────────────────────────────
        # The new (default) path uses the binary wall U-Net (epoch_20.pth)
        # plus rectangle decomposition. The legacy color/hollow path stays
        # available behind ``config.use_rect_walls=False`` for fallback.
        raw_rectangles = None
        unet_door_bboxes: list = []
        unet_window_bboxes: list = []
        detection = None
        unet_detection = None

        if self.config.use_rect_walls:
            _notify("rect-wall detection")
            logger.info("[%s] Stage 0: Rectangle-decomposition wall detection",
                        base_name)
            try:
                from detection.rect_walls import (
                    RectWallDetector, rasterise_wall_segments,
                )
                # Build OCR bbox list for enclosed-space refinement.
                # Only numeric labels (e.g. "12.5", "8 м²") are kept —
                # room names like "Кухня" are excluded so they cannot
                # prevent structural cavities from being filled as wall.
                _ocr_bboxes_for_rect: list = []
                for _src in (ocr_text_labels_early, rotated_ocr_labels):
                    for _item in _src:
                        if len(_item) >= 5 and _is_numeric_ocr_label(str(_item[0])):
                            _ocr_bboxes_for_rect.append((
                                str(_item[0]),
                                int(_item[1]), int(_item[2]),
                                int(_item[3]), int(_item[4]),
                            ))
                _rect_det = RectWallDetector(self.config)
                _rect_det.initialize()
                _enclosed_dbg_path = (
                    os.path.join(output_dir, f"{base_name}_enclosed_spaces.png")
                    if self.config.debug_mode else None
                )
                _rectdecompose_log = (
                    os.path.join(output_dir, f"{base_name}_rectdecompose.log")
                    if self.config.debug_mode else None
                )
                # ── Brusnika branch (gated) ──────────────────────────────
                # When the image verifiably matches the Brusnika palette, build
                # the wall mask from colour + morphology and hand it to the
                # rect detector in place of the binary U-Net.  When the gate is
                # False this stays None and detect() runs the U-Net exactly as
                # before — the original path is left untouched.
                _brusnika_mask = None
                if self.config.enable_brusnika_branch:
                    try:
                        from detection.brusnika import (
                            is_brusnika_image, build_brusnika_wall_mask,
                        )
                        if is_brusnika_image(img_clean, self.config):
                            _brusnika_mask = build_brusnika_wall_mask(
                                img_clean, self.config)
                            logger.info(
                                "[%s] Brusnika format detected — using "
                                "colour-derived wall mask (%d px)",
                                base_name,
                                int(np.count_nonzero(_brusnika_mask)),
                            )
                    except Exception as _bexc:
                        logger.warning(
                            "[%s] Brusnika branch failed (%s) — falling back "
                            "to U-Net wall mask", base_name, _bexc,
                        )
                        _brusnika_mask = None

                _rd_res = _rect_det.detect(
                    img_clean,
                    wall_mask=_brusnika_mask,
                    ocr_bboxes=_ocr_bboxes_for_rect,
                    debug_img_path=_enclosed_dbg_path,
                    log_file_path=_rectdecompose_log,
                )
                result.walls             = _rd_res.walls
                result.wall_mask         = _rd_res.wall_mask
                result.raw_wall_mask     = _rd_res.raw_wall_mask
                result.wall_rect_polygons = _rd_res.wall_rect_polygons
                # outline_mask = rect raster OR processed U-Net mask so
                # room-finding has extra ink to bridge corner gaps.
                _rect_rast = rasterise_wall_segments(_rd_res.walls,
                                                      img_clean.shape[:2])
                _rast = _rect_rast.copy()
                if _rd_res.wall_mask is not None:
                    _rast = cv2.bitwise_or(_rast, _rd_res.wall_mask)
                result.outline_mask = _rast
                logger.info(
                    "[%s] Rect-wall detector: %d walls (%d diagonal)",
                    base_name, len(result.walls),
                    sum(1 for w in result.walls if w.is_diagonal),
                )

                # ── Save wall mask debug images ────────────────────────
                # Raw: straight U-Net argmax output, no post-processing.
                _raw_mask = (_rd_res.raw_wall_mask
                             if _rd_res.raw_wall_mask is not None
                             else np.zeros((2, 2), np.uint8))
                cv2.imwrite(
                    os.path.join(output_dir, f"{base_name}_mask_raw.png"),
                    _raw_mask,
                )
                # Filtered: after cap (morph-close) + enclosed-space
                # refinement — the actual input to rect_decompose.
                _filt_mask = (_rd_res.wall_mask
                              if _rd_res.wall_mask is not None
                              else np.zeros((2, 2), np.uint8))
                cv2.imwrite(
                    os.path.join(output_dir, f"{base_name}_mask_filtered.png"),
                    _filt_mask,
                )
            except Exception as exc:
                logger.exception(
                    "[%s] Rect-wall detection failed (%s) — falling back to "
                    "legacy U-Net/colour path", base_name, exc,
                )
                # Fall through to the legacy path with normal vars.
                # NOTE: img_clean must stay alive here — the legacy fallback
                # below still needs it; it is freed after wall detection.
                result.walls = []

        # Legacy path runs only when the new detector is disabled or
        # produced no walls.
        _legacy_walls_active = (not self.config.use_rect_walls
                                or not result.walls)

        if _legacy_walls_active:
            _notify("U-Net detection")
            unet_detection = self._try_unet_detection(img_clean)

        if _legacy_walls_active and unet_detection is not None:
            unet_wall_mask, unet_door_bboxes, unet_window_bboxes = unet_detection

            # U-Net door/window bboxes are ALWAYS used for opening detection
            # regardless of --enable-unet.  Wall detection from U-Net is only
            # activated when --enable-unet is explicitly set.
            if not self.config.enable_unet:
                logger.info(
                    "[%s] U-Net doors collected (%d doors, %d windows); "
                    "U-Net wall pipeline skipped (--enable-unet not set)",
                    base_name, len(unet_door_bboxes), len(unet_window_bboxes),
                )
                unet_detection = None

        if _legacy_walls_active:
            # ── Stage 1 (fallback): colour-based wall detection ──────────
            _notify("detecting walls")
            logger.info("[%s] Stage 1: Colour-based wall detection", base_name)
            detection, raw_rectangles = self._run_wall_detection(img_clean)

            result.walls = detection.walls
            result.wall_mask = detection.wall_mask
            result.outline_mask = detection.outline_mask

        # Free the cleaned image; downstream stages use `img`
        # (un-erased deskewed image) for opening detection.
        try:
            del img_clean
        except NameError:
            pass


        # ── Stage 1b: Room Measurement + Scale Calibration ────────────────
        # Runs BEFORE opening detection so the gap/arc detectors' metric
        # filters (min/max opening width in meters, arc radius bounds) use
        # the calibrated scale instead of the config default.
        _notify("OCR / scale calibration")
        logger.info("[%s] Stage 1b: Room measurement", base_name)
        measurements, ocr_labels, pixels_to_cm = self._run_room_measurement(
            image_path=image_path,
            img=img,
            walls=result.walls,
            outline_mask=result.outline_mask,
            output_dir=output_dir,
        )

        result.raw_measurements = measurements
        result.ocr_labels = ocr_labels
        result.pixels_to_cm = pixels_to_cm

        if measurements:
            ps = measurements.pixel_scale
            if ps and ps > 0:
                pixels_to_cm = ps * 100.0
                if 0.1 < pixels_to_cm < 20.0:
                    result.pixel_scale = ps
                    result.pixels_to_cm = pixels_to_cm
                    self.config = self.config.copy_with(pixels_to_cm=pixels_to_cm)
                    self.exporter.update_scale(pixels_to_cm)
                    self.visualizer.pixels_to_cm = pixels_to_cm
                else:
                    logger.warning("Room-area scale out of bounds (%.4f), ignoring",
                                   pixels_to_cm)

        # Reuse OCR labels collected in pre-stage (no re-run needed).
        logger.info("[%s] Stage 1b2: OCR text label collection (reusing pre-stage)",
                    base_name)
        ocr_text_labels = ocr_text_labels_early
        if ocr_text_labels:
            ocr_labels = ocr_text_labels
            result.ocr_labels = ocr_labels

        # ── Stage 1b3: Wall-length scale calibration (rotated labels) ────
        # Uses rotated OCR labels collected in pre-stage (no re-run).
        logger.info("[%s] Stage 1b3: Wall-length scale calibration", base_name)
        wl_ptcm = _calibrate_scale_from_wall_lengths(
            img=img,
            walls=result.walls,
            normal_labels=ocr_text_labels,
            rotated_labels=rotated_ocr_labels,
            debug_dir=output_dir,
            rotated_labels_in_image_coords=True,
        )
        if wl_ptcm is not None and 0.1 < wl_ptcm < 20.0:
            logger.info("[%s] Wall-length calibration: pixels_to_cm %.4f -> %.4f",
                        base_name, result.pixels_to_cm, wl_ptcm)
            result.pixels_to_cm = wl_ptcm
            result.pixel_scale = wl_ptcm / 100.0
            self.config = self.config.copy_with(pixels_to_cm=wl_ptcm)
            self.exporter.update_scale(wl_ptcm)
            self.visualizer.pixels_to_cm = wl_ptcm

        # ── Stage 2: Opening Detection (UNIFIED: gap + U-Net with deduplication) ─────
        _notify("detecting openings")

        openings: List[Opening] = []
        door_arcs: List[Any] = []
        ocr_bboxes: List = []

        if not self.config.enable_opening_detection:
            logger.info("[%s] Stage 2: Opening detection disabled (enable_opening_detection=False)", base_name)
        else:
            logger.info("[%s] Stage 2: Opening detection (gap priority + U-Net supplement)", base_name)

            # Build ocr_bboxes from the labels collected pre-deskew
            # (no second OCR pass).  Only numeric labels are kept so that
            # non-numeric room names cannot suppress door/wall detection.
            for _src in (ocr_text_labels_early, rotated_ocr_labels):
                for item in _src:
                    if len(item) < 5:
                        continue
                    if not _is_numeric_ocr_label(str(item[0])):
                        continue
                    ocr_bboxes.append((
                        str(item[0]),
                        int(item[1]), int(item[2]),
                        int(item[3]), int(item[4]),
                    ))

            # ALWAYS run traditional gap/algorithmic + YOLO detection (gap doors take priority)
            yolo_results = None
            if self._yolo is not None:
                try:
                    yolo_results = self._yolo.detect(img)
                    result.raw_yolo_results = yolo_results
                except Exception as e:
                    logger.warning("YOLO detection failed: %s", e)

            pixel_scale = self.config.pixels_to_m

            # ── Solid-wall / no-U-Net path: gap + U-Net combined ─────────────
            openings, door_arcs = self._run_opening_detection(
                img=img,
                walls=result.walls,
                wall_mask=result.wall_mask,
                outline_mask=result.outline_mask,
                wall_rectangles=raw_rectangles,
                yolo_results=yolo_results,
                ocr_bboxes=ocr_bboxes,
                pixel_scale=pixel_scale,
                base_name=base_name,
                output_dir=output_dir,
                wall_rect_polygons=result.wall_rect_polygons,
            )

            # Supplement with U-Net doors that don't overlap gap detections
            if unet_door_bboxes or unet_window_bboxes:
                unet_openings = self._unet_bboxes_to_openings(
                    unet_door_bboxes, unet_window_bboxes
                )
                priority_for_door = [o for o in openings if o.opening_type in (OpeningType.GAP, OpeningType.DOOR)]
                priority_for_window = [o for o in openings if o.opening_type == OpeningType.WINDOW]

                filtered_unet = []
                overlapping_count = 0
                for uo in unet_openings:
                    priority_list = priority_for_door if uo.opening_type == OpeningType.DOOR else priority_for_window
                    if any(self._bbox_iou(uo.bbox, po.bbox) > 0.3 for po in priority_list):
                        overlapping_count += 1
                    else:
                        filtered_unet.append(uo)

                if filtered_unet:
                    openings.extend(filtered_unet)
                    logger.info(
                        "[%s] U-Net supplement: added %d/%d openings "
                        "(%d overlapped gap/door detections)",
                        base_name, len(filtered_unet), len(unet_openings), overlapping_count,
                    )

        # ── Stage 2a: Spatial filter — no back-to-back doors ───────────
        # Runs after every door source (gap, YOLO, U-Net, supplement) has been
        # merged so doors stacked on the same doorway are removed once, here.
        if openings:
            openings = filter_back_to_back_doors(
                openings,
                separation_factor=self.config.min_door_separation_factor,
                tiny_room_factor=getattr(self.config, "door_tiny_room_factor", 1.0),
                min_along_overlap=getattr(self.config, "door_tiny_room_min_overlap", 0.6),
            )

        result.openings = openings
        result.door_arcs = door_arcs

        # ── Stage 2a2: Wall-attachment filter ───────────────────────────
        # A real door/window straddles a wall (or the gap between two wall
        # segments), so its center is never far from a wall centerline.
        # Detections floating in the middle of a room are furniture false
        # positives; they must go BEFORE Stage 2b so the diagonal-drop list
        # (reused later to seal doorways for room detection) stays clean.
        if (self.config.enable_opening_wall_attachment
                and result.openings and result.walls):
            _n_un, _unattached = self._drop_unattached_openings(
                openings=result.openings,
                walls=result.walls,
                max_dist_factor=self.config.opening_attach_max_dist_factor,
            )
            if _n_un:
                result.dropped_unattached_openings = _unattached
                logger.info(
                    "[%s] Stage 2a2: dropped %d floating opening(s) with no "
                    "wall within reach", base_name, _n_un,
                )

        # ── Stage 2b: Drop openings on diagonal walls ───────────────────
        # Doors and windows must sit on axis-aligned walls. Any opening
        # whose nearest wall (within opening_wall_proximity_cm) is diagonal
        # is removed entirely — relocating them onto axis walls would be
        # wrong and the YOLO/arc detectors are themselves axis-strict.
        if self.config.use_rect_walls and result.openings:
            _drop_n, _dropped = self._drop_diagonal_openings(
                openings=result.openings,
                walls=result.walls,
                tol_deg=self.config.opening_axis_tol_deg,
            )
            if _drop_n:
                result.dropped_diagonal_openings = _dropped
                logger.info(
                    "[%s] Stage 2b: dropped %d opening(s) on diagonal walls "
                    "(see result.dropped_diagonal_openings for details)",
                    base_name, _drop_n,
                )

        # ── Stage 3c: Wall cleanup ────────────────────────────────────
        logger.info("[%s] Stage 3c: Wall cleanup", base_name)
        result.walls = cleanup_walls(result.walls)

        # ── Stage 3d: Inject door walls ───────────────────────────────
        # Door-wall segments (is_door_wall=True) are thin WallSegments whose
        # geometry matches each detected door opening exactly.  They are added
        # AFTER cleanup so they are not filtered/merged by cleanup_walls.
        # The SH3D exporter routes them as non-structural extra walls so
        # ParentWallFinder always finds an exact parent — no synthetic wall
        # creation or thickness guessing in the exporter.
        #
        # One door-wall is built for EVERY door opening (YOLO, gap, U-Net),
        # all at the apartment-average wall thickness so doors render solid and
        # consistent (not thin/awkward) and each fits its opening exactly.
        _door_openings = [
            o for o in result.openings
            if o.opening_type == OpeningType.DOOR
        ]
        if _door_openings:
            from detection.unet_doors import _bbox_to_door_wall_segment
            _avg_t = average_wall_thickness(result.walls)
            _door_walls = [
                _bbox_to_door_wall_segment(
                    o.bbox, i, walls=result.walls, thickness=_avg_t,
                )
                for i, o in enumerate(_door_openings)
            ]
            result.walls = result.walls + _door_walls
            logger.info(
                "[%s] Stage 3d: injected %d door-wall segment(s) "
                "at avg thickness %.1f px",
                base_name, len(_door_walls), _avg_t,
            )

        # ── Stage 3e: Inject window walls ─────────────────────────────
        # Mirror of Stage 3d for windows: build a thin non-structural wall
        # matching each window opening so the SH3D exporter always has an exact
        # parent wall to host the window object.  Unlike door walls these may
        # overlap existing structural walls — windows sit ON exterior walls.
        # Built AFTER cleanup (like door walls) so they are not merged/filtered.
        _window_openings = [
            o for o in result.openings
            if o.opening_type == OpeningType.WINDOW
        ]
        logger.info(
            "[%s] Stage 3e: %d window opening(s) detected before host-wall check",
            base_name, len(_window_openings),
        )
        if _window_openings:
            from detection.unet_doors import bbox_to_window_wall_segment
            _window_walls = []
            _hostless_ids = set()
            for i, o in enumerate(_window_openings):
                _seg = bbox_to_window_wall_segment(o.bbox, i, walls=result.walls)
                if _seg is None:
                    # No same-orientation structural wall near the window —
                    # almost certainly a false detection; drop the window
                    # rather than synthesising a free-floating wall panel.
                    _hostless_ids.add(o.id)
                    continue
                _window_walls.append(_seg)
            if _hostless_ids:
                result.openings = [
                    o for o in result.openings if o.id not in _hostless_ids
                ]
                logger.info(
                    "[%s] Stage 3e: dropped %d window(s) with no host wall",
                    base_name, len(_hostless_ids),
                )
            result.walls = result.walls + _window_walls
            logger.info(
                "[%s] Stage 3e: injected %d window-wall segment(s)",
                base_name, len(_window_walls),
            )

        # ── Stage 4: SH3D Export ───────────────────────────────────────
        _notify("exporting SH3D")
        logger.info("[%s] Stage 4: SH3D export", base_name)
        sh3d_path = os.path.join(output_dir, f"{base_name}.sh3d")
        debug_img_path = os.path.join(output_dir, f"{base_name}_debug.png")

        rooms = self._run_export(
            walls=result.walls,
            openings=result.openings,
            output_path=sh3d_path,
            original_image=img,
            ocr_labels=ocr_labels,
            debug_image_path=debug_img_path,
            wall_mask=result.wall_mask,
            enclosed_labels=None,
            seal_only_openings=getattr(result, 'dropped_diagonal_openings', None),
        )

        result.rooms = rooms
        result.sh3d_path = sh3d_path if os.path.exists(sh3d_path) else None

        # ── Stage 4b: Secondary scale calibration from room polygons ───────
        # Stacked-label rule (highest priority): if a room contains exactly one
        # integer numeric label with a smaller decimal label directly above it
        # and no other integers in the room, the integer is the room area in m².
        # Falls back to standard room-polygon calibration.
        if result.rooms and ocr_text_labels:
            logger.info("[%s] Stage 4b: Scale calibration from rooms", base_name)
            new_sf = self._calibrate_scale_from_stacked_labels(
                ocr_text_labels=ocr_text_labels,
                rooms=result.rooms,
                pixels_to_cm=result.pixels_to_cm,
                current_scale_factor=self.config.sh3d_scale_factor,
            )
            if new_sf is None:
                new_sf = self._calibrate_scale_from_rooms(
                    rooms=result.rooms,
                    ocr_text_labels=ocr_text_labels,
                    pixels_to_cm=result.pixels_to_cm,
                    current_scale_factor=self.config.sh3d_scale_factor,
                )
            if new_sf is not None and abs(new_sf - self.config.sh3d_scale_factor) > 1e-4:
                # Update the XML generator's scale_factor
                inner_exp = getattr(self.exporter, '_inner', None)
                if inner_exp is not None:
                    inner_exp.xml_generator.scale_factor = new_sf
                self.config = self.config.copy_with(sh3d_scale_factor=new_sf)

                # Remove the first export and redo with corrected scale
                try:
                    os.remove(sh3d_path)
                except OSError:
                    pass
                rooms = self._run_export(
                    walls=result.walls,
                    openings=result.openings,
                    output_path=sh3d_path,
                    original_image=img,
                    ocr_labels=ocr_labels,
                    debug_image_path=debug_img_path,
                    wall_mask=result.wall_mask,
                    enclosed_labels=None,
                    seal_only_openings=getattr(result, 'dropped_diagonal_openings', None),
                )
                result.rooms = rooms
                result.sh3d_path = sh3d_path if os.path.exists(sh3d_path) else None

        # ── Stage 4b2: Scale sanity gate ───────────────────────────────────
        # Exported plan-space cm = px * pixels_to_cm / sh3d_scale_factor.
        # When every calibration source fails (common on BTI scans where room
        # extraction fails), stale defaults ship physically impossible plans
        # (observed: 36 m-wide apartments with 3.8 m doors, and 5 m-wide ones).
        # Clamp by re-deriving sf against an assumed long side as last resort.
        if self.config.enable_scale_sanity_gate and result.walls:
            xs = [c for w in result.walls for c in (w.bbox.x1, w.bbox.x2)]
            ys = [c for w in result.walls for c in (w.bbox.y1, w.bbox.y2)]
            ext_px = float(max(max(xs) - min(xs), max(ys) - min(ys)))
            sf_cur = self.config.sh3d_scale_factor
            ext_m = ext_px * result.pixels_to_cm / sf_cur / 100.0
            lo = self.config.scale_sanity_min_extent_m
            hi = self.config.scale_sanity_max_extent_m
            if ext_px > 0 and not (lo <= ext_m <= hi):
                target_m = self.config.scale_sanity_fallback_extent_m
                new_sf = ext_px * result.pixels_to_cm / (target_m * 100.0)
                logger.warning(
                    "[%s] Scale sanity gate: plan long side %.1f m outside "
                    "[%.1f, %.1f] m (pixels_to_cm=%.4f, sf=%.4f) — forcing "
                    "sf -> %.4f (assumed %.1f m long side)",
                    base_name, ext_m, lo, hi, result.pixels_to_cm, sf_cur,
                    new_sf, target_m,
                )
                inner_exp = getattr(self.exporter, '_inner', None)
                if inner_exp is not None:
                    inner_exp.xml_generator.scale_factor = new_sf
                self.config = self.config.copy_with(sh3d_scale_factor=new_sf)
                try:
                    os.remove(sh3d_path)
                except OSError:
                    pass
                rooms = self._run_export(
                    walls=result.walls,
                    openings=result.openings,
                    output_path=sh3d_path,
                    original_image=img,
                    ocr_labels=ocr_labels,
                    debug_image_path=debug_img_path,
                    wall_mask=result.wall_mask,
                    enclosed_labels=None,
                    seal_only_openings=getattr(result, 'dropped_diagonal_openings', None),
                )
                result.rooms = rooms
                result.sh3d_path = sh3d_path if os.path.exists(sh3d_path) else None
            else:
                logger.info(
                    "[%s] Scale sanity gate: plan long side %.1f m OK "
                    "(pixels_to_cm=%.4f, sf=%.4f)",
                    base_name, ext_m, result.pixels_to_cm, sf_cur,
                )

        # ── Stage 4c: Room area validation (diagnostic only) ──────────────
        if result.rooms and ocr_text_labels:
            result.room_area_reports = _validate_room_areas(
                rooms=result.rooms,
                ocr_text_labels=ocr_text_labels,
                pixels_to_cm=result.pixels_to_cm,
                scale_factor=self.config.sh3d_scale_factor,
                tolerance_ratio=self.config.room_area_tolerance_ratio,
            )
            if result.room_area_reports:
                _n_mismatch = sum(
                    1 for r in result.room_area_reports
                    if r.get("status") == "mismatch"
                )
                if _n_mismatch:
                    logger.info(
                        "[%s] Stage 4c: %d/%d rooms have area mismatch",
                        base_name, _n_mismatch, len(result.room_area_reports),
                    )

        # ── Stage 5: Visualization ─────────────────────────────────────
        _notify("rendering")
        logger.info("[%s] Stage 5: Visualization", base_name)
        summary_path = os.path.join(output_dir, f"{base_name}_summary.png")
        rooms_ocr_path = os.path.join(output_dir, f"{base_name}_rooms_ocr.png")

        # Build wall segments for overlay from result.walls (unified WallSegments).
        # raw_rectangles may be None (e.g. U-Net path), but result.walls is
        # always populated.  _walls_to_stubs converts WallSegment → stub with
        # flat .x1/.y1/.x2/.y2 attrs that the visualizer expects.
        wall_stubs = self._walls_to_stubs(result.walls)

        self._render_summary(
            img=img,
            wall_stubs=wall_stubs,
            openings=result.openings,
            door_arcs=result.door_arcs,
            raw_wall_mask=result.raw_wall_mask,
            outline_mask=result.outline_mask,
            measurements=measurements,
            output_path=summary_path,
        )

        # rooms_ocr is always generated — works fine with an empty rooms list
        # (just shows the image with OCR labels and no room polygons).
        self._render_rooms_ocr(
            img=img,
            rooms=result.rooms or [],
            openings=[],
            ocr_labels=ocr_labels,
            door_arcs=result.door_arcs,
            output_path=rooms_ocr_path,
        )

        # Rooms + openings overlay — only meaningful when rooms exist
        if result.rooms:
            rooms_openings_path = os.path.join(output_dir, f"{base_name}_rooms_openings.png")
            self._render_rooms_with_openings(
                img=img,
                rooms=result.rooms,
                openings=result.openings,
                door_arcs=result.door_arcs,
                output_path=rooms_openings_path,
            )

        result.summary_path = summary_path if os.path.exists(summary_path) else None
        result.rooms_ocr_path = rooms_ocr_path if os.path.exists(rooms_ocr_path) else None

        # ── Cleanup: free heavy intermediate data ─────────────────────
        result.wall_mask = None
        result.outline_mask = None
        result.raw_wall_mask = None
        result.raw_yolo_results = None
        result.raw_measurements = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info(
            "[%s] Done: %d walls, %d doors, %d windows, %d rooms",
            base_name, result.n_walls, result.n_doors,
            result.n_windows, len(result.rooms),
        )

        return result

    # ============================================================
    # PRIVATE STAGE IMPLEMENTATIONS
    # ============================================================

    def _try_unet_detection(
        self, img: np.ndarray
    ) -> Optional[tuple]:
        """Try U-Net detection, return (wall_mask, door_bboxes, window_bboxes).

        wall_mask is binary uint8 (0/255) used for enclosed-region voting.
        door_bboxes / window_bboxes are lists of (x1,y1,x2,y2) tuples from
        the U-Net post-processing pipeline — identical to the old U-Net branch.

        Returns None if U-Net is unavailable or produces an empty mask.
        U-Net inference runs in a subprocess to isolate potential crashes.
        """
        # enable_unet previously gated U-Net completely; now U-Net always
        # attempts so both detection methods are always combined.
        # It still fails gracefully when checkpoint / deps are absent.
        ckpt = self.config.unet_checkpoint_path
        if not Path(ckpt).exists():
            logger.info("U-Net checkpoint not found at %s, skipping", ckpt)
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Dynamic U-Net input size: average of image dimensions
        h, w = img.shape[:2]
        unet_img_size = (h + w) // 2

        try:
            from detection.unet_inference import run_unet_subprocess as _detect_unet_subprocess
        except ImportError as exc:
            logger.info("U-Net dependencies not available (%s), skipping", exc)
            return None
        unet_result = _detect_unet_subprocess(
            image_rgb=img_rgb,
            checkpoint_path=ckpt,
            device=self.config.unet_device,
            img_size=unet_img_size,
            confidence=self.config.unet_confidence,
            min_wall_area=self.config.unet_min_wall_area,
            min_door_area=self.config.unet_min_door_area,
            min_window_area=self.config.unet_min_window_area,
        )
        logger.info(
            "U-Net subprocess: img_size=%d px (dynamic: (%d+%d)//2)",
            unet_img_size, h, w
        )

        if unet_result is None:
            return None

        wall_mask = unet_result.get("wall_mask")
        if wall_mask is None or not wall_mask.any():
            logger.warning("U-Net returned empty wall mask")
            return None

        wall_mask = (wall_mask > 0).astype(np.uint8) * 255
        door_bboxes = unet_result.get("door_bboxes", [])
        window_bboxes = unet_result.get("window_bboxes", [])

        logger.info(
            "U-Net: %d wall px, %d doors, %d windows",
            int(np.sum(wall_mask > 0)), len(door_bboxes), len(window_bboxes),
        )
        return wall_mask, door_bboxes, window_bboxes

    @staticmethod
    def _unet_bboxes_to_walls(
        wall_bboxes: list,
    ) -> List[WallSegment]:
        """Convert U-Net wall bboxes to unified WallSegment list."""
        walls = []
        for i, (x1, y1, x2, y2) in enumerate(wall_bboxes):
            w = x2 - x1
            h = y2 - y1
            thickness = float(min(w, h))
            walls.append(WallSegment(
                id=f"unet_wall_{i}",
                bbox=BBox(x1=float(x1), y1=float(y1),
                          x2=float(x2), y2=float(y2)),
                thickness=thickness,
                is_structural=True,
                is_outer=False,
            ))
        return walls

    @staticmethod
    def _unet_bboxes_to_openings(
        door_bboxes: list,
        window_bboxes: list,
    ) -> List[Opening]:
        """Convert U-Net door/window bboxes to unified Opening list."""
        openings = []
        for x1, y1, x2, y2 in door_bboxes:
            openings.append(Opening(
                bbox=BBox(x1=float(x1), y1=float(y1),
                          x2=float(x2), y2=float(y2)),
                opening_type=OpeningType.DOOR,
                confidence=1.0,
                source="unet",
            ))
        for x1, y1, x2, y2 in window_bboxes:
            openings.append(Opening(
                bbox=BBox(x1=float(x1), y1=float(y1),
                          x2=float(x2), y2=float(y2)),
                opening_type=OpeningType.WINDOW,
                confidence=1.0,
                source="unet",
            ))
        return openings

    @staticmethod
    def _bbox_iou(bbox_a: BBox, bbox_b: BBox) -> float:
        """Compute Intersection-over-Union (IoU) between two BBox objects.
        Returns 0.0 if no overlap.
        """
        x1 = max(bbox_a.x1, bbox_b.x1)
        y1 = max(bbox_a.y1, bbox_b.y1)
        x2 = min(bbox_a.x2, bbox_b.x2)
        y2 = min(bbox_a.y2, bbox_b.y2)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (bbox_a.x2 - bbox_a.x1) * (bbox_a.y2 - bbox_a.y1)
        area_b = (bbox_b.x2 - bbox_b.x1) * (bbox_b.y2 - bbox_b.y1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _drop_unattached_openings(
        openings: list, walls: List[WallSegment], max_dist_factor: float = 1.0,
    ) -> Tuple[int, List[Opening]]:
        """
        Mutate ``openings`` in place: remove every opening whose center is
        farther from ALL wall centerlines than ``max_dist_factor`` times the
        opening's own long side.

        A genuine door straddles its wall, and a door in an undetected wall
        gap still sits within ~half a door-width of the flanking segments'
        endpoints, so legitimate openings measure well under 1.0x. Furniture
        false positives float in room interiors, several widths away.
        Angle-agnostic: diagonal walls attach via their outline centerline.

        Returns ``(n_dropped, dropped_openings)``.
        """
        if not openings or not walls:
            return 0, []

        def _seg_dist(px: float, py: float,
                      a: Tuple[float, float], b: Tuple[float, float]) -> float:
            ax, ay = a; bx, by = b
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            if l2 < 1e-9:
                return math.hypot(px - ax, py - ay)
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
            return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

        wall_lines = []
        for w in walls:
            if w.outline and len(w.outline) >= 2:
                p1 = (float(w.outline[0][0]), float(w.outline[0][1]))
                p2 = (float(w.outline[1][0]), float(w.outline[1][1]))
            else:
                p1, p2 = w.centerline
                p1 = (float(p1[0]), float(p1[1]))
                p2 = (float(p2[0]), float(p2[1]))
            wall_lines.append((p1, p2))

        kept: List[Opening] = []
        dropped_list: List[Opening] = []
        for op in openings:
            cx, cy = op.bbox.center
            long_side = max(op.bbox.width, op.bbox.height)
            reach = max_dist_factor * long_side
            best_d = min(_seg_dist(cx, cy, p1, p2) for p1, p2 in wall_lines)
            if best_d > reach:
                logger.debug(
                    "unattached opening dropped: type=%s center=(%.0f,%.0f) "
                    "nearest wall %.0fpx away (reach %.0fpx)",
                    op.opening_type, cx, cy, best_d, reach,
                )
                dropped_list.append(op)
            else:
                kept.append(op)

        if dropped_list:
            openings.clear()
            openings.extend(kept)
        return len(dropped_list), dropped_list

    @staticmethod
    def _drop_diagonal_openings(
        openings: list, walls: List[WallSegment], tol_deg: float = 3.0,
    ) -> Tuple[int, List[Opening]]:
        """
        Mutate ``openings`` in place: remove every opening whose nearest
        wall is diagonal (angle deviates from 0°/90° by more than tol_deg).
        Distance is measured center-to-centerline.

        Returns ``(n_dropped, dropped_openings)`` so callers can expose the
        dropped list for diagnostics without re-running the filter.
        """
        if not openings or not walls:
            return 0, []

        def _seg_dist(px: float, py: float,
                      a: Tuple[float, float], b: Tuple[float, float]) -> float:
            ax, ay = a; bx, by = b
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            if l2 < 1e-9:
                return math.hypot(px - ax, py - ay)
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
            cx, cy = ax + t * dx, ay + t * dy
            return math.hypot(px - cx, py - cy)

        # Pre-compute centerline endpoints + diagonal flag per wall
        wall_geo = []
        for w in walls:
            if w.outline and len(w.outline) >= 2:
                p1 = (float(w.outline[0][0]), float(w.outline[0][1]))
                p2 = (float(w.outline[1][0]), float(w.outline[1][1]))
            else:
                # axis-aligned legacy WallSegment — derive from bbox
                p1, p2 = w.centerline
                p1 = (float(p1[0]), float(p1[1]))
                p2 = (float(p2[0]), float(p2[1]))
            wall_geo.append((w.is_diagonal, p1, p2))

        kept: List[Opening] = []
        dropped_list: List[Opening] = []
        for op in openings:
            cx, cy = op.bbox.center
            best_d = float('inf'); best_diag = False
            for is_diag, p1, p2 in wall_geo:
                d = _seg_dist(cx, cy, p1, p2)
                if d < best_d:
                    best_d = d
                    best_diag = is_diag
            if best_diag:
                dropped_list.append(op)
            else:
                kept.append(op)

        if dropped_list:
            openings.clear()
            openings.extend(kept)
        return len(dropped_list), dropped_list

    def _run_wall_detection(self, img: np.ndarray) -> tuple:
        """
        Run wall detection and return (DetectionResult, raw_rectangles).

        Calls the adapter's detect() method so that auto-wall-color detection
        and wider filter tolerances are applied when cfg.auto_wall_color=True.
        """
        try:
            # CRITICAL: Call detect() not inner.process() directly.
            # This ensures auto-color-detection is applied.
            detection = self.wall_detector.detect(img, config=self.config)

            # Retrieve cached rectangles for downstream use
            rectangles = self.wall_detector._last_rectangles

            return detection, rectangles

        except Exception as e:
            logger.error("Wall detection failed: %s", e, exc_info=True)
            h, w = img.shape[:2]
            return DetectionResult(
                walls=[],
                wall_mask=np.zeros((h, w), dtype=np.uint8),
                outline_mask=np.zeros((h, w), dtype=np.uint8),
            ), []

    def _run_opening_detection(
        self,
        img: np.ndarray,
        walls: List[WallSegment],
        wall_mask: Optional[np.ndarray],
        outline_mask: Optional[np.ndarray],
        wall_rectangles: List,
        yolo_results: Optional[Any],
        ocr_bboxes: Optional[List],
        pixel_scale: float,
        base_name: str,
        output_dir: Optional[str] = None,
        wall_rect_polygons: Optional[List] = None,
    ) -> Tuple[List[Opening], List[Any]]:
        """
        Run opening detection and door arc detection.

        Returns (openings, door_arcs)
        """
        openings: List[Opening] = []
        door_arcs: List[Any] = []

        effective_mask = outline_mask if outline_mask is not None else wall_mask

        try:
            openings = self.opening_detector.detect(
                image=img,
                walls=walls,
                wall_mask=effective_mask,
                wall_rectangles=wall_rectangles,
                yolo_results=yolo_results,
                ocr_bboxes=ocr_bboxes,
                pixel_scale=pixel_scale,
                base_name=base_name,
            )
        except Exception as e:
            logger.warning("Opening detection failed (non-fatal): %s", e, exc_info=True)

        # Run arc detector separately (returns DetectedDoorArc, not in openings list)
        if self.config.enable_door_arcs and self.opening_detector._arc_detector is not None:
            try:
                from detection.gap import Opening as GapOpening
                # Doorway candidates for arc matching: wall-line breaks are now
                # promoted to DOOR (see OpeningPipeline.detect Step 6), so feed
                # both doors and any residual gaps as candidate openings.
                geo_gaps = [
                    o for o in openings
                    if o.opening_type in (OpeningType.GAP, OpeningType.DOOR)
                ]
                raw_gaps = []
                for i, g in enumerate(geo_gaps):
                    raw_gaps.append(GapOpening(
                        id=i,
                        x1=int(g.bbox.x1), y1=int(g.bbox.y1),
                        x2=int(g.bbox.x2), y2=int(g.bbox.y2),
                        width=int(g.bbox.width), height=int(g.bbox.height),
                        area=int(g.bbox.area),
                        opening_type=g.opening_type.value,
                        confidence=g.confidence,
                        source_rects=[], yolo_matches=[],
                    ))

                door_arcs = self.opening_detector._arc_detector.detect(
                    img=img,
                    wall_outline_mask=effective_mask,
                    candidate_gaps=raw_gaps,
                    yolo_doors=yolo_results.doors if yolo_results else None,
                    pixel_scale=pixel_scale,
                    ocr_boxes=ocr_bboxes,
                    base_name=base_name,
                )
            except Exception as e:
                logger.warning("Door arc extraction failed (non-fatal): %s", e)

        # ── Door debug images ──────────────────────────────────────────
        if output_dir:
            try:
                unet_door_mask = getattr(
                    self.opening_detector, '_debug_unet_door_mask', None
                )
                self._save_door_debug(
                    img=img,
                    openings=openings,
                    output_dir=output_dir,
                    base_name=base_name,
                    unet_door_mask=unet_door_mask,
                )
            except Exception as _dde:
                logger.warning("Door debug image failed (non-fatal): %s", _dde)

            # ── Rect decompose overlay ─────────────────────────────────
            try:
                _unet_det = getattr(self.opening_detector, '_unet_door_detector', None)
                door_rect_bboxes = list(getattr(_unet_det, '_debug_rect_bboxes', []))
                self._save_rect_overlay(
                    img=img,
                    wall_rect_polygons=wall_rect_polygons or [],
                    door_rect_bboxes=door_rect_bboxes,
                    output_dir=output_dir,
                    base_name=base_name,
                )
            except Exception as _roe:
                logger.warning("Rect overlay image failed (non-fatal): %s", _roe)

        return openings, door_arcs

    def _save_door_debug(
        self,
        img: np.ndarray,
        openings: List[Opening],
        output_dir: str,
        base_name: str,
        unet_door_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Write two door debug images:

        ``*_door_mask.png``
            U-Net door-class pixel mask (uint8, 255=door) when available;
            otherwise a white-filled rectangle mask of accepted doors.

        ``*_door_boxes.png``
            Original image annotated with:
              - GREEN  solid rect  = accepted door (in final openings)
              - RED    solid rect  = rejected YOLO candidate (detected but
                                    not in final output after dedup/fusion)
              - CYAN   solid rect  = U-Net door bboxes (source="unet_door")
            Confidence score and source shown above each box.
        """
        H, W = img.shape[:2]

        # Accepted doors from the final opening list
        accepted = [o for o in openings if o.opening_type == OpeningType.DOOR]

        # Raw YOLO door candidates stored by detect() before dedup/fusion
        yolo_raw = getattr(self.opening_detector, '_debug_yolo_doors_raw', [])

        # ── 1. Door mask ───────────────────────────────────────────────
        # Prefer the raw U-Net pixel mask; fall back to bbox rectangles.
        if unet_door_mask is not None and unet_door_mask.shape[:2] == (H, W):
            mask = unet_door_mask.copy()
        else:
            mask = np.zeros((H, W), dtype=np.uint8)
            for o in accepted:
                x1, y1, x2, y2 = (int(o.bbox.x1), int(o.bbox.y1),
                                   int(o.bbox.x2), int(o.bbox.y2))
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_door_mask.png"), mask)

        # ── 2. Door bounding box image ─────────────────────────────────
        ann = img.copy()

        def _iou_any(bbox: BBox, others: List[Opening]) -> float:
            """Max IoU of bbox against a list of openings."""
            a = bbox.to_xyxy()
            return max(
                (iou_xyxy(a, o.bbox.to_xyxy()) for o in others),
                default=0.0,
            )

        # Rejected = raw YOLO doors with low overlap against accepted set
        REJECT_IOU_THR = 0.30
        rejected = [
            o for o in yolo_raw
            if _iou_any(o.bbox, accepted) < REJECT_IOU_THR
        ]

        # Draw rejected YOLO candidates (red)
        for o in rejected:
            x1, y1, x2, y2 = (int(o.bbox.x1), int(o.bbox.y1),
                               int(o.bbox.x2), int(o.bbox.y2))
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 0, 220), 2)
            lbl = f"{o.confidence:.2f} yolo-rej"
            cv2.putText(ann, lbl, (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1,
                        cv2.LINE_AA)

        # Draw accepted openings — colour by source
        unet_accepted = [o for o in accepted if getattr(o, 'source', '') == 'unet_door']
        other_accepted = [o for o in accepted if getattr(o, 'source', '') != 'unet_door']

        for o in other_accepted:
            x1, y1, x2, y2 = (int(o.bbox.x1), int(o.bbox.y1),
                               int(o.bbox.x2), int(o.bbox.y2))
            cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 200, 0), 2)
            src = getattr(o, 'source', '')
            lbl = f"{o.confidence:.2f} {src}"
            cv2.putText(ann, lbl, (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1,
                        cv2.LINE_AA)

        # U-Net door detections in cyan
        for o in unet_accepted:
            x1, y1, x2, y2 = (int(o.bbox.x1), int(o.bbox.y1),
                               int(o.bbox.x2), int(o.bbox.y2))
            cv2.rectangle(ann, (x1, y1), (x2, y2), (220, 200, 0), 2)
            lbl = f"{o.confidence:.2f} unet"
            cv2.putText(ann, lbl, (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 200, 0), 1,
                        cv2.LINE_AA)

        logger.info(
            "[%s] Door debug: %d accepted (%d unet), %d rejected YOLO candidates",
            base_name, len(accepted), len(unet_accepted), len(rejected),
        )
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_door_boxes.png"), ann
        )

    def _save_rect_overlay(
        self,
        img: np.ndarray,
        wall_rect_polygons: List,
        door_rect_bboxes: List,
        output_dir: str,
        base_name: str,
    ) -> None:
        """
        Write ``*_rect_overlay.png`` — the original floorplan with
        rect-decompose rectangles drawn on top.

        Colour coding
        -------------
        Walls  (filled + outline): semi-transparent **teal**  (BGR 160,120,0)
        Doors  (filled + outline): semi-transparent **orange** (BGR 0,100,230)

        Each rectangle is filled at 40% opacity then outlined at full opacity
        so individual rects remain distinguishable.
        """
        ann = img.copy()
        canvas = img.copy()

        # ── Wall polygons (4×2 float32 each) ──────────────────────────
        WALL_FILL    = (160, 120,   0)   # teal fill
        WALL_OUTLINE = (255, 200,   0)   # bright teal outline

        for poly in wall_rect_polygons:
            pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [pts], WALL_FILL)

        # ── Door rect bboxes ──────────────────────────────────────────
        DOOR_FILL    = (  0, 100, 230)   # orange fill
        DOOR_OUTLINE = (  0,  50, 255)   # vivid orange outline

        for x1, y1, x2, y2 in door_rect_bboxes:
            pts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32
            ).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [pts], DOOR_FILL)

        # Blend filled canvas over original
        ann = cv2.addWeighted(canvas, 0.40, img, 0.60, 0)

        # ── Outlines (fully opaque, drawn on blended result) ──────────
        for poly in wall_rect_polygons:
            pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(ann, [pts], isClosed=True, color=WALL_OUTLINE, thickness=1)

        for x1, y1, x2, y2 in door_rect_bboxes:
            cv2.rectangle(ann, (x1, y1), (x2, y2), DOOR_OUTLINE, thickness=2)

        logger.info(
            "[%s] Rect overlay: %d wall rects, %d door rects",
            base_name, len(wall_rect_polygons), len(door_rect_bboxes),
        )
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_rect_overlay.png"), ann)

    def _get_ocr_bboxes(self, img: np.ndarray) -> List:
        """Get OCR text bounding boxes from the wall detector's OCR model."""
        try:
            inner = self.wall_detector.get_inner()
            if hasattr(inner, 'get_ocr_bboxes'):
                return inner.get_ocr_bboxes(img)
        except Exception:
            pass
        return []

    def _run_room_measurement(
        self,
        image_path: str,
        img: np.ndarray,
        walls: List[WallSegment],
        outline_mask: Optional[np.ndarray],
        output_dir: str,
    ) -> Tuple[Optional[Any], List, float]:
        """
        Run OCR-based room measurement.

        Returns (measurements, ocr_labels, pixels_to_cm)
        """
        pixels_to_cm = self.config.pixels_to_cm
        ocr_labels: List = []
        measurements = None

        if self._room_analyzer is None or not walls:
            return measurements, ocr_labels, pixels_to_cm

        try:
            segments_dicts = []
            for w in walls:
                # Convert WallSegment centerline to the dict format expected by RoomSizeAnalyzer
                cl = w.centerline
                segments_dicts.append({
                    'id': hash(w.id) % 100000,
                    'x1': cl[0][0], 'y1': cl[0][1],
                    'x2': cl[1][0], 'y2': cl[1][1],
                    'is_outer': w.is_outer,
                })

            measurements = self._room_analyzer.process(
                image_path,
                segments_dicts,
                output_dir,
                wall_mask=outline_mask,
                original_image=img,
                save_debug=self.config.debug_mode,
            )

            if measurements:
                # Extract OCR labels for visualization as (text, x1, y1, x2, y2) tuples
                ocr_labels = [
                    (r['text'], *r['bbox'])
                    for r in (measurements.room_areas or [])
                ]

        except Exception as e:
            logger.warning("Room measurement failed (non-fatal): %s", e, exc_info=True)

        return measurements, ocr_labels, pixels_to_cm

    def _run_export(
        self,
        walls: List[WallSegment],
        openings: List[Opening],
        output_path: str,
        original_image: Optional[np.ndarray],
        ocr_labels: List,
        debug_image_path: str,
        wall_mask: Optional[np.ndarray],
        enclosed_labels: Optional[np.ndarray] = None,
        seal_only_openings: Optional[List[Opening]] = None,
    ) -> List[Room]:
        """Run SH3D export and return detected rooms."""
        try:
            return self.exporter.export(
                walls=walls,
                openings=openings,
                output_path=output_path,
                original_image=original_image,
                ocr_labels=ocr_labels if ocr_labels else None,
                debug_image_path=debug_image_path,
                wall_mask=wall_mask,
                enclosed_labels=enclosed_labels,
                seal_only_openings=seal_only_openings,
            )
        except Exception as e:
            logger.error("SH3D export failed: %s", e, exc_info=True)
            return []

    def _render_summary(
        self,
        img: np.ndarray,
        wall_stubs: List,
        openings: List[Opening],
        door_arcs: List,
        raw_wall_mask: Optional[np.ndarray],
        outline_mask: Optional[np.ndarray],
        measurements: Optional[Any],
        output_path: str,
    ) -> None:
        """Render 2x3 debug summary image."""
        try:
            n_doors = sum(1 for o in openings if o.opening_type == OpeningType.DOOR)
            n_windows = sum(1 for o in openings if o.opening_type == OpeningType.WINDOW)

            self.visualizer.render_summary(
                img_bgr=img,
                wall_segments=wall_stubs,
                openings=openings,
                door_arcs=door_arcs,
                wall_mask=raw_wall_mask,
                outline_mask=outline_mask,
                measurements=measurements,
                output_path=output_path,
                n_walls=len(wall_stubs),
                n_doors=n_doors,
                n_windows=n_windows,
                n_arcs=len(door_arcs),
            )
        except Exception as e:
            logger.warning("Summary rendering failed (non-fatal): %s", e, exc_info=True)

    def _render_rooms_ocr(
        self,
        img: np.ndarray,
        rooms: List[Room],
        openings: List,
        ocr_labels: List,
        door_arcs: List,
        output_path: str,
    ) -> None:
        """Render room + OCR overlay."""
        try:
            self.visualizer.render_room_ocr_overlay(
                img_bgr=img,
                rooms=rooms,
                openings=openings,
                ocr_labels=ocr_labels,
                output_path=output_path,
                door_arcs=door_arcs,
            )
        except Exception as e:
            logger.warning("Room OCR rendering failed (non-fatal): %s", e, exc_info=True)

    def _render_rooms_with_openings(
        self,
        img: np.ndarray,
        rooms: List[Room],
        openings: List[Opening],
        door_arcs: List,
        output_path: str,
    ) -> None:
        """Render rooms polygons overlaid with all openings (doors + windows)."""
        try:
            self.visualizer.render_rooms_with_openings(
                img_bgr=img,
                rooms=rooms,
                openings=openings,
                door_arcs=door_arcs,
                output_path=output_path,
                pixels_to_cm=self.config.pixels_to_cm,
            )
        except Exception as e:
            logger.warning("Rooms+openings rendering failed (non-fatal): %s", e, exc_info=True)

    @staticmethod
    def _walls_to_stubs(walls: List[WallSegment]) -> List:
        """Convert unified WallSegment list to stubs for visualization.

        The visualizer expects flat .x1/.y1/.x2/.y2 attributes on each
        object.  WallSegment stores them under .bbox, so we create thin
        wrappers.
        """
        class _Stub:
            __slots__ = ('x1', 'y1', 'x2', 'y2')
        stubs = []
        for w in walls:
            s = _Stub()
            s.x1 = w.bbox.x1
            s.y1 = w.bbox.y1
            s.x2 = w.bbox.x2
            s.y2 = w.bbox.y2
            stubs.append(s)
        return stubs

    def _rectangles_to_stubs(self, rectangles: List) -> List:
        """
        Convert WallRectangle list to stub objects for visualization.

        Returns objects with .x1, .y1, .x2, .y2 attributes.
        """
        # WallRectangle already has these attributes directly
        return rectangles

    def _get_ocr_text_labels(self, img: np.ndarray) -> List[Tuple]:
        """
        Return all OCR words with their pixel bounding boxes.

        Format: List of (text, x1, y1, x2, y2).
        Reuses the OCR model already loaded in the wall detector.
        """
        try:
            inner = self.wall_detector.get_inner()
            if hasattr(inner, 'get_ocr_texts_with_coords'):
                return inner.get_ocr_texts_with_coords(img)
        except Exception as e:
            logger.warning("OCR text label extraction failed (non-fatal): %s", e)
        return []

    def _calibrate_scale_from_stacked_labels(
        self,
        ocr_text_labels: List[Tuple],
        rooms: List[Room],
        pixels_to_cm: float,
        current_scale_factor: float,
    ) -> Optional[float]:
        """
        Stacked-label scale calibration (highest priority).

        Rule: if a room contains exactly one integer label (room number, above)
        with a decimal label directly below it whose value is larger, and no
        other integer labels in the room — treat the decimal below as the room
        area in m² and compute scale_factor with the same formula as
        _calibrate_scale_from_rooms.

        Pattern:   [integer]   ← room number, above, smaller
                   [decimal]   ← room area, below, larger

        Returns the new scale_factor, or None if the pattern is not found.
        """
        if not rooms or not ocr_text_labels:
            return None

        # Parse every OCR label into value + type
        all_labels = []
        for item in ocr_text_labels:
            if len(item) < 5:
                continue
            text = str(item[0]).strip()
            x1, y1, x2, y2 = item[1], item[2], item[3], item[4]
            clean = text.replace(',', '.').replace(' ', '')
            try:
                val = float(clean)
            except ValueError:
                continue
            # Integer: no decimal separator in the original text
            is_int = ('.' not in text and ',' not in text)
            all_labels.append({
                'val': val, 'is_int': is_int,
                'cx': (x1 + x2) / 2.0, 'cy': (y1 + y2) / 2.0,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            })

        if not all_labels:
            return None

        integer_labels = [lb for lb in all_labels if lb['is_int']]
        decimal_labels = [lb for lb in all_labels if not lb['is_int']]

        # For each integer label, find a qualifying decimal label directly below it.
        # stacked_pairs: integer index → decimal label (the area)
        stacked_pairs: dict = {}

        for i, a in enumerate(integer_labels):
            for b in decimal_labels:
                # b must be below a (larger y = lower on image)
                if b['cy'] <= a['cy']:
                    continue
                vert_gap = b['cy'] - a['cy']
                label_h = max(b['y2'] - b['y1'], a['y2'] - a['y1'], 1.0)
                if vert_gap > label_h * 3.5:
                    continue
                # Horizontally aligned
                horiz_dist = abs(a['cx'] - b['cx'])
                label_w = max(b['x2'] - b['x1'], a['x2'] - a['x1'], 1.0)
                if horiz_dist > label_w * 2.0:
                    continue
                # Decimal below must be numerically larger than integer above
                if b['val'] <= a['val']:
                    continue
                # Decimal must be in a sane room-area range
                if not (1.0 <= b['val'] <= 300.0):
                    continue
                stacked_pairs[i] = b
                break

        if not stacked_pairs:
            return None

        m_per_px = pixels_to_cm / 100.0
        ratios: List[float] = []

        for room in rooms:
            pts_px = [
                (p.x / pixels_to_cm, p.y / pixels_to_cm)
                for p in room.points
            ]
            if len(pts_px) < 3:
                continue
            area_px2 = _polygon_area_from_tuples(pts_px)
            if area_px2 < 1.0:
                continue

            # Collect all integer labels inside this room
            ints_in_room = [
                (i, lb) for i, lb in enumerate(integer_labels)
                if _point_in_polygon(lb['cx'], lb['cy'], pts_px)
            ]
            # Rule: exactly one integer in the room, and it must be a stacked one
            if len(ints_in_room) != 1:
                continue
            idx, _ = ints_in_room[0]
            if idx not in stacked_pairs:
                continue

            area_m2 = stacked_pairs[idx]['val']
            ratios.append(math.sqrt(area_px2 * (m_per_px ** 2) / area_m2))

        if not ratios:
            return None

        new_sf = float(np.median(ratios))
        logger.info(
            "Stacked-label scale calibration from %d room(s): sf %.4f -> %.4f",
            len(ratios), current_scale_factor, new_sf,
        )
        return new_sf

    def _calibrate_scale_from_rooms(
        self,
        rooms: List[Room],
        ocr_text_labels: List[Tuple],
        pixels_to_cm: float,
        current_scale_factor: float,
    ) -> Optional[float]:
        """
        Compute a corrected scale_factor by matching detected room polygon
        areas against OCR-detected room area labels placed inside them.

        This is the secondary calibration pass (mirrors calibrate_scale_factor()
        from the legacy main.py).  The primary pass sets pixels_to_cm from the
        total apartment area; this pass adjusts the XML scale_factor so that
        individual room polygons match the area labels printed on the plan.

        The returned value is the median of sqrt(computed_area / ocr_area) over
        all rooms where exactly one area label falls inside the polygon.  When
        pixels_to_cm is already correct this evaluates to ≈ 1.0; when the
        primary calibration was skipped it absorbs the full correction.

        Returns the new scale_factor, or None if calibration cannot proceed.
        """
        if not rooms or not ocr_text_labels:
            return None

        # Parse area values from every OCR word
        parsed: List[Tuple[float, float, float]] = []  # (area_m2, cx, cy)
        for item in ocr_text_labels:
            if len(item) < 5:
                continue
            text, x1, y1, x2, y2 = item[0], item[1], item[2], item[3], item[4]
            area = _parse_ocr_area_m2(str(text))
            if area is not None and 1.0 <= area <= 200.0:
                parsed.append((area, (x1 + x2) / 2.0, (y1 + y2) / 2.0))

        if not parsed:
            return None

        m_per_px = pixels_to_cm / 100.0
        ratios: List[float] = []

        for room in rooms:
            # Convert room points from cm back to pixel space
            pts_px = [
                (p.x / pixels_to_cm, p.y / pixels_to_cm)
                for p in room.points
            ]
            if len(pts_px) < 3:
                continue

            area_px2 = _polygon_area_from_tuples(pts_px)
            if area_px2 < 1.0:
                continue

            # Collect a ratio for every label whose centre lands in this room.
            # Using all matched pairs (not just rooms with exactly one label)
            # gives a more robust median across the whole floor plan.
            for area_val, cx, cy in parsed:
                if _point_in_polygon(cx, cy, pts_px):
                    ratios.append(math.sqrt(area_px2 * (m_per_px ** 2) / area_val))

        if not ratios:
            return None

        new_sf = float(np.median(ratios))
        logger.info(
            "Scale calibration from %d room(s): sf %.4f → %.4f",
            len(ratios), current_scale_factor, new_sf,
        )
        return new_sf

    def _init_ocr(self) -> Optional[Any]:
        """Initialize docTR OCR model."""
        try:
            from doctr.models import ocr_predictor
            model = ocr_predictor(pretrained=True)
            logger.info("OCR model loaded")
            return model
        except Exception as e:
            logger.warning("OCR initialization failed (non-fatal): %s", e)
            return None
