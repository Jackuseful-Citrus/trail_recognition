"""Piece recognition directly in an aspect-preserving source grayscale image."""

import math

import puzzle_config as cfg
from a4_projective_mapper import A4ProjectiveMapper
from puzzle_geometry import PieceObservation, polygon_area, polygon_centroid
from puzzle_perf import PERF_STATS
from puzzle_vision import (
    _blob_component_boundary,
    _blob_value,
    _convex_hull_points,
    _native_gray_sanity,
    _ordered_contour_polygon,
    _piece_center_contour_threshold,
    background_difference_threshold,
)
from source_divider_detector import (
    SourceDividerTracker,
    SourceScanlineMask,
    detect_source_divider,
    estimate_source_half,
)


def _polygon_area_px(points):
    total = 0.0
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        total += point[0] * other[1] - other[0] * point[1]
    return abs(total) * 0.5


def _background_stats_from_a4(gray_array, scanline_mask):
    bins = max(
        8,
        min(
            256,
            int(getattr(cfg, "PIECE_BACKGROUND_HISTOGRAM_BINS", 64)),
        ),
    )
    counts = [0 for _ in range(bins)]
    sums = [0 for _ in range(bins)]
    stride = max(
        1, int(getattr(cfg, "PIECE_BACKGROUND_SAMPLE_STRIDE", 4))
    )
    sample_count = 0
    # Both divider-separated halves participate.  Percentiles below the
    # median prioritise their dark background and reject sparse white pieces.
    for rows in (scanline_mask.top_rows, scanline_mask.bottom_rows):
        for y in range(0, scanline_mask.height, stride):
            span = rows[y]
            if span is None:
                continue
            row = gray_array[y]
            for x in range(span[0], span[1] + 1, stride):
                value = max(0, min(255, int(row[x])))
                index = min(bins - 1, value * bins // 256)
                counts[index] += 1
                sums[index] += value
                sample_count += 1

    def percentile(fraction):
        if sample_count <= 0:
            return 0.0
        target = max(1, int(sample_count * fraction + 0.5))
        cumulative = 0
        for index, count in enumerate(counts):
            cumulative += count
            if cumulative >= target:
                if count:
                    return float(sums[index]) / count
                return (index + 0.5) * 256.0 / bins
        return 255.0

    background = percentile(0.45)
    background_high = percentile(0.60)
    return {
        "background_gray": background,
        "background_high_gray": background_high,
        "background_spread_gray": max(0.0, background_high - background),
        "sample_count": sample_count,
    }


def _source_threshold(background_stats, threshold):
    mode = getattr(cfg, "PIECE_SEGMENTATION_MODE", "fixed")
    if threshold is not None:
        return max(0, min(255, int(threshold))), "background_delta_override"
    if (
        mode == "background_delta"
        and background_stats["sample_count"]
        >= int(getattr(cfg, "PIECE_BACKGROUND_MIN_SAMPLES", 96))
    ):
        return (
            background_difference_threshold(
                background_stats,
                cfg.PIECE_BACKGROUND_DELTA_GRAY,
                cfg.PIECE_BACKGROUND_NOISE_MARGIN_GRAY,
                cfg.PIECE_BACKGROUND_MAX_DELTA_GRAY,
            ),
            "background_delta",
        )
    return int(cfg.WHITE_GRAY_THRESHOLD), (
        "background_delta_fallback" if mode == "background_delta" else "fixed_native"
    )


def _touches_source_boundary(rect, rows, margin=1):
    x0 = int(rect[0])
    y0 = int(rect[1])
    x1 = x0 + int(rect[2]) - 1
    y1 = y0 + int(rect[3]) - 1
    for y in range(max(0, y0), min(len(rows), y1 + 1)):
        span = rows[y]
        if span is None:
            return True
        if x0 <= span[0] + margin or x1 >= span[1] - margin:
            return True
    return False


def _empty_diagnostics(gray_image, mapper, divider, generation, reason):
    diagnostics = {
        "rectified": gray_image,
        "source_work_image": gray_image,
        "backend": "source_projective",
        "reason": reason,
        "generation": generation,
        "rotation_corr_calls": 0,
        "raw_contours": 0,
        "rejected": {},
        "trace_failures": {},
        "detected_vertex_counts": [],
        "divider_detected": bool(divider and divider.get("detected", False)),
        "divider_strip_detected": bool(
            divider and divider.get("detected", False)
        ),
        "divider_source": "source_midband",
        "divider_reason": (
            divider.get("reason", reason) if divider else reason
        ),
        "divider_y_mm": (
            divider.get("divider_y_mm", cfg.DIVIDER_Y_MM)
            if divider
            else cfg.DIVIDER_Y_MM
        ),
        "divider_slope_mm": (
            divider.get("slope_mm", 0.0) if divider else 0.0
        ),
        "divider_coverage": (
            divider.get("coverage", 0.0) if divider else 0.0
        ),
        "divider_residual_px": (
            divider.get("residual_px", 0.0) if divider else 0.0
        ),
        "divider_contrast": (
            divider.get("contrast", 0.0) if divider else 0.0
        ),
        "divider_confidence": (
            divider.get("confidence", 0.0) if divider else 0.0
        ),
        "divider_hits": (
            divider.get("hit_count", 0) if divider else 0
        ),
        "divider_samples": (
            divider.get("sample_count", 0) if divider else 0
        ),
        "divider_thickness_px": 0.0,
        "divider_mask_half_width_px": 0,
        "divider_masked_pixels": 0,
        "condition_metric": (
            mapper.condition_metric if mapper is not None else float("inf")
        ),
        "a4_corners_source_px": (
            list(mapper.a4_polygon_source_px)
            if mapper is not None and mapper.valid
            else []
        ),
        "recognition_width": int(gray_image.width()),
        "recognition_height": int(gray_image.height()),
    }
    if (
        mapper is not None
        and mapper.valid
        and divider is not None
        and divider.get("detected", False)
    ):
        diagnostics["divider_left_px"] = mapper.a4_mm_to_source_px(
            (0.0, divider.get("left_y_mm", divider["divider_y_mm"]))
        )
        diagnostics["divider_right_px"] = mapper.a4_mm_to_source_px(
            (
                mapper.a4_width_mm,
                divider.get("right_y_mm", divider["divider_y_mm"]),
            )
        )
    return diagnostics


def detect_pieces_from_source_projective_image(
    gray_image,
    mapper,
    divider,
    region="source",
    threshold=None,
    collect_sanity=False,
    source_side=None,
    scanline_mask=None,
    generation=0,
):
    """Detect source-image components, then project every boundary point.

    No call in this function performs image rectification.  ``contour_px`` is
    always expressed in the supplied source work image coordinates.
    """
    if region != "source":
        raise ValueError("source-projective detector only supports region=source")
    if mapper is None or not mapper.valid:
        return [], _empty_diagnostics(
            gray_image, mapper, divider, generation, "invalid_mapper"
        )
    if divider is None or not divider.get("detected", False):
        return [], _empty_diagnostics(
            gray_image, mapper, divider, generation, "divider_required"
        )
    if source_side not in ("top", "bottom"):
        return [], _empty_diagnostics(
            gray_image, mapper, divider, generation, "source_half_required"
        )
    width = int(gray_image.width())
    height = int(gray_image.height())
    if width != mapper.source_width or height != mapper.source_height:
        raise ValueError("source work image and mapper size mismatch")
    gray_array = gray_image.to_numpy_ref()
    if scanline_mask is None:
        scanline_mask = SourceScanlineMask(mapper, divider, source_side)
    background_stats = _background_stats_from_a4(gray_array, scanline_mask)
    piece_threshold, threshold_mode = _source_threshold(
        background_stats, threshold
    )
    contour_threshold = max(
        piece_threshold,
        int(getattr(cfg, "PIECE_CONTOUR_MIN_GRAY_THRESHOLD", 0)),
    )
    mask_started = PERF_STATS.mark()
    masked_pixels = scanline_mask.blacken_outside_source(gray_array)
    PERF_STATS.add_stage("source_mask_ms", mask_started)
    bbox = mapper.a4_bbox_source_px
    pixel_area_per_mm2 = _polygon_area_px(
        mapper.a4_polygon_source_px
    ) / (mapper.a4_width_mm * mapper.a4_height_mm)
    min_pixels = max(
        16,
        int(cfg.MIN_PIECE_AREA_MM2 * pixel_area_per_mm2 * 0.30),
    )
    blob_started = PERF_STATS.mark()
    blobs = gray_image.find_blobs(
        [(piece_threshold, 255)],
        roi=bbox,
        x_stride=1,
        y_stride=1,
        pixels_threshold=min_pixels,
        area_threshold=min_pixels,
        merge=False,
    )
    PERF_STATS.add_stage("source_blob_ms", blob_started)
    observations = []
    details = []
    rejected = {"area": 0, "border": 0, "polygon": 0, "mapping": 0}
    trace_failures = {}
    boundary_steps = 0
    pixel_reads = 0
    component_bound_count = 0
    component_candidate_count = 0
    contour_component_candidate_count = 0
    polygon_failure_rects = []
    boundary_failure_reasons = {}
    raw_rects = []
    for blob in blobs:
        rect = tuple(_blob_value(blob, "rect", None))
        raw_rects.append(rect)
        if _touches_source_boundary(rect, scanline_mask.source_rows):
            rejected["border"] += 1
            continue
        center = (
            float(_blob_value(blob, "cx", 5)),
            float(_blob_value(blob, "cy", 6)),
        )
        blob_pixels = float(_blob_value(blob, "pixels", 4))
        adaptive_background = (
            background_stats["background_gray"]
            if background_stats["sample_count"]
            >= int(getattr(cfg, "PIECE_BACKGROUND_MIN_SAMPLES", 96))
            else None
        )
        (
            piece_contour_threshold,
            center_gray,
            contour_mode,
        ) = _piece_center_contour_threshold(
            gray_array,
            center,
            adaptive_background,
            piece_threshold,
            contour_threshold,
        )
        boundary_started = PERF_STATS.mark()
        boundary_px, trace = _blob_component_boundary(
            gray_array,
            rect,
            center,
            blob_pixels,
            piece_threshold,
            piece_contour_threshold,
        )
        PERF_STATS.add_stage("source_boundary_ms", boundary_started)
        boundary_steps += trace.get("boundary_steps", 0)
        pixel_reads += trace.get("pixel_reads", 0)
        component_candidate_count += trace.get("component_candidates", 0)
        contour_component_candidate_count += trace.get(
            "contour_component_candidates", 0
        )
        if trace.get("reason") == "identity_ok":
            component_bound_count += 1
        if not trace.get("ok", False):
            reason = trace.get("reason", "unknown")
            trace_failures[reason] = trace_failures.get(reason, 0) + 1
            boundary_failure_reasons[reason] = (
                boundary_failure_reasons.get(reason, 0) + 1
            )
            rejected["polygon"] += 1
            polygon_failure_rects.append(rect)
            continue
        project_started = PERF_STATS.mark()
        boundary_mm = []
        projection_failed = False
        for point in boundary_px:
            mapped = mapper.source_px_to_a4_mm(point)
            if mapped is None:
                projection_failed = True
                break
            boundary_mm.append(mapped)
        PERF_STATS.add_stage("boundary_project_ms", project_started)
        if projection_failed or len(boundary_mm) < cfg.MIN_POLYGON_VERTICES:
            rejected["mapping"] += 1
            continue
        fit_started = PERF_STATS.mark()
        fit_diagnostics = {}
        polygon_mm = _ordered_contour_polygon(
            boundary_mm, fit_diagnostics
        )
        if (
            cfg.FORCE_CONVEX_CONTOURS
            or (
                polygon_mm is not None
                and len(_convex_hull_points(polygon_mm)) == len(polygon_mm)
            )
        ):
            stable_polygon = _ordered_contour_polygon(
                _convex_hull_points(boundary_mm)
            )
            if stable_polygon is not None:
                polygon_mm = stable_polygon
        PERF_STATS.add_stage("polygon_fit_mm_ms", fit_started)
        if polygon_mm is None:
            rejected["polygon"] += 1
            polygon_failure_rects.append(rect)
            trace_failures["fit_invalid"] = (
                trace_failures.get("fit_invalid", 0) + 1
            )
            continue
        area_mm2 = polygon_area(polygon_mm)
        if (
            area_mm2 < cfg.MIN_PIECE_AREA_MM2
            or area_mm2 > cfg.MAX_PIECE_AREA_MM2
        ):
            rejected["area"] += 1
            continue
        identity_error = float(trace.get("blob_pixel_error_ratio", 0.0))
        confidence = max(0.0, min(1.0, 1.0 - 0.5 * identity_error))
        try:
            observation = PieceObservation(
                "",
                boundary_px,
                polygon_mm,
                centroid_mm=polygon_centroid(polygon_mm),
                area_mm2=area_mm2,
                confidence=confidence,
                calibration_generation=generation,
            )
        except ValueError:
            rejected["polygon"] += 1
            trace_failures["piece_invalid"] = (
                trace_failures.get("piece_invalid", 0) + 1
            )
            continue
        details.append(
            (
                observation,
                center_gray,
                piece_contour_threshold,
                contour_mode,
            )
        )
    details.sort(key=lambda item: item[0].area_mm2, reverse=True)
    details = details[: cfg.MAX_PIECE_COUNT]
    observations = [item[0] for item in details]
    center_grays = [item[1] for item in details]
    contour_thresholds = [item[2] for item in details]
    contour_threshold_modes = [item[3] for item in details]
    sanity = None
    if collect_sanity:
        sanity = {
            "native": _native_gray_sanity(gray_image),
            "rotation_return": "not_called",
            "find_min_pixels": min_pixels,
            "blob_rects": raw_rects,
        }
    diagnostics = _empty_diagnostics(
        gray_image, mapper, divider, generation, "ok"
    )
    diagnostics.update(
        {
            "source_side": source_side,
            "source_polygon_px": scanline_mask.source_polygon_px,
            "divider_left_px": scanline_mask.divider_left_px,
            "divider_right_px": scanline_mask.divider_right_px,
            "threshold": float(piece_threshold),
            "contour_threshold": float(contour_threshold),
            "threshold_mode": threshold_mode,
            "segmentation_mode": getattr(
                cfg, "PIECE_SEGMENTATION_MODE", "fixed"
            ),
            "background_gray": background_stats["background_gray"],
            "background_high_gray": background_stats[
                "background_high_gray"
            ],
            "background_spread_gray": background_stats[
                "background_spread_gray"
            ],
            "background_sample_count": background_stats["sample_count"],
            "threshold_delta_gray": (
                float(piece_threshold) - background_stats["background_gray"]
            ),
            "raw_contours": len(blobs),
            "raw_blob_rects": raw_rects,
            "rejected": rejected,
            "trace_failures": trace_failures,
            "rectified_border_black_px": 0,
            "source_masked_pixels": masked_pixels,
            "component_bound_count": component_bound_count,
            "component_candidate_count": component_candidate_count,
            "contour_component_candidate_count": (
                contour_component_candidate_count
            ),
            "boundary_primary_ok": component_bound_count,
            "boundary_fallback_used": 0,
            "boundary_fallback_ordered_ok": 0,
            "boundary_failure_reason": boundary_failure_reasons,
            "polygon_failure_rects": polygon_failure_rects,
            "polygon_reverse_retry_count": 0,
            "polygon_unrefined_retry_count": 0,
            "boundary_steps": boundary_steps,
            "pixel_reads": pixel_reads,
            "detected_vertex_counts": [
                len(piece.polygon_mm) for piece in observations
            ],
            "center_grays": center_grays,
            "contour_thresholds": contour_thresholds,
            "contour_threshold_modes": contour_threshold_modes,
            "gray_sanity": sanity,
            "recognition_width": width,
            "recognition_height": height,
            "work_corners_px": list(mapper.a4_polygon_source_px),
        }
    )
    expected_corners_mm = (
        (0.0, 0.0),
        (mapper.a4_width_mm, 0.0),
        (mapper.a4_width_mm, mapper.a4_height_mm),
        (0.0, mapper.a4_height_mm),
    )
    corner_errors = []
    for source_point, expected_mm in zip(
        mapper.a4_polygon_source_px, expected_corners_mm
    ):
        mapped = mapper.source_px_to_a4_mm(source_point)
        if mapped is None:
            corner_errors.append(float("inf"))
        else:
            corner_errors.append(
                math.sqrt(
                    (mapped[0] - expected_mm[0]) ** 2
                    + (mapped[1] - expected_mm[1]) ** 2
                )
            )
    roundtrip_points = list(mapper.a4_polygon_source_px)
    center_source = mapper.a4_mm_to_source_px(
        (0.5 * mapper.a4_width_mm, 0.5 * mapper.a4_height_mm)
    )
    if center_source is not None:
        roundtrip_points.append(center_source)
    diagnostics["map_sanity"] = {
        "corner_error_mm": max(corner_errors),
        "roundtrip_max_px": max(
            mapper.roundtrip_error_px(point)
            for point in roundtrip_points
        ),
        "nonfinite": 0,
    }
    return observations, diagnostics


class SourceProjectiveRecognition:
    """Stateful source-projective calibration/divider/half coordinator."""

    __slots__ = (
        "mapper",
        "generation",
        "divider_tracker",
        "divider",
        "source_side",
        "source_half_state",
        "scanline_mask",
        "_corner_key",
    )

    def __init__(self):
        self.mapper = None
        self.generation = None
        self.divider_tracker = SourceDividerTracker()
        self.divider = None
        self.source_side = None
        self.source_half_state = None
        self.scanline_mask = None
        self._corner_key = None

    def reset(self, generation=None):
        self.mapper = None
        self.generation = generation
        self.divider_tracker.reset(generation)
        self.divider = None
        self.source_side = None
        self.source_half_state = None
        self.scanline_mask = None
        self._corner_key = None

    def _update_mapper(self, corners_source_px, width, height, generation):
        corner_key = tuple(
            int(round(float(value) * 16.0))
            for point in corners_source_px
            for value in point
        )
        if generation != self.generation:
            self.reset(generation)
        if self.mapper is not None and corner_key == self._corner_key:
            return False
        build_started = PERF_STATS.mark()
        self.mapper = A4ProjectiveMapper(
            corners_source_px,
            width,
            height,
            cfg.A4_WIDTH_MM,
            cfg.A4_HEIGHT_MM,
        )
        PERF_STATS.add_stage("a4_map_build_ms", build_started)
        self._corner_key = corner_key
        self.scanline_mask = None
        return True

    def detect(
        self,
        gray_image,
        corners_source_px,
        generation,
        threshold=None,
        collect_sanity=False,
    ):
        width = int(gray_image.width())
        height = int(gray_image.height())
        self._update_mapper(
            corners_source_px, width, height, generation
        )
        if self.mapper is None or not self.mapper.valid:
            return [], _empty_diagnostics(
                gray_image,
                self.mapper,
                None,
                generation,
                "invalid_mapper",
            )
        divider_started = PERF_STATS.mark()
        detected_divider = detect_source_divider(gray_image, self.mapper)
        self.divider = self.divider_tracker.update(
            detected_divider, generation
        )
        PERF_STATS.add_stage("divider_detect_ms", divider_started)
        if not self.divider.get("detected", False):
            return [], _empty_diagnostics(
                gray_image,
                self.mapper,
                self.divider,
                generation,
                "divider_required",
            )
        neutral_mask = SourceScanlineMask(
            self.mapper, self.divider, "top"
        )
        background_stats = _background_stats_from_a4(
            gray_image.to_numpy_ref(), neutral_mask
        )
        provisional_threshold, _mode = _source_threshold(
            background_stats, threshold
        )
        half_state = estimate_source_half(
            gray_image,
            self.mapper,
            self.divider,
            provisional_threshold,
        )
        self.source_half_state = half_state
        if self.source_side is None and half_state["valid"]:
            self.source_side = half_state["side"]
        if self.source_side is None:
            diagnostics = _empty_diagnostics(
                gray_image,
                self.mapper,
                self.divider,
                generation,
                half_state["reason"],
            )
            diagnostics["source_half"] = half_state
            return [], diagnostics
        # Rebuild after every accepted mapper/divider update; ordinary pixel
        # loops then use only row spans, never point-in-polygon tests.
        self.scanline_mask = SourceScanlineMask(
            self.mapper, self.divider, self.source_side
        )
        observations, diagnostics = (
            detect_pieces_from_source_projective_image(
                gray_image,
                self.mapper,
                self.divider,
                region="source",
                threshold=threshold,
                collect_sanity=collect_sanity,
                source_side=self.source_side,
                scanline_mask=self.scanline_mask,
                generation=generation,
            )
        )
        diagnostics["source_half"] = half_state
        diagnostics["source_half_selected"] = self.source_side
        return observations, diagnostics
