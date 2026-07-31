"""Divider detection and scanline masks in the unrectified source image."""

import math

import puzzle_config as cfg


def _source_divider_median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return 0.5 * (
        float(ordered[middle - 1]) + float(ordered[middle])
    )


def _source_divider_gray_at(gray_array, width, height, point):
    x = max(0, min(width - 1, int(round(point[0]))))
    y = max(0, min(height - 1, int(round(point[1]))))
    return float(gray_array[y][x])


def _source_divider_point_line_distance(point, start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 1e-9:
        return float("inf")
    return abs(
        dx * (start[1] - point[1])
        - (start[0] - point[0]) * dy
    ) / length


def detect_source_divider(gray_image, mapper):
    """Detect the physical bright divider without perspective resampling."""
    result = {
        "detected": False,
        "reason": "invalid_mapper",
        "divider_y_mm": float(cfg.DIVIDER_Y_MM),
        "slope_mm": 0.0,
        "coverage": 0.0,
        "contrast": 0.0,
        "residual_px": 0.0,
        "rms_residual_px": 0.0,
        "thickness_mm": 0.0,
        "confidence": 0.0,
        "hit_count": 0,
        "sample_count": 0,
        "points_source_px": [],
    }
    if mapper is None or not mapper.valid:
        return result
    gray_array = gray_image.to_numpy_ref()
    width = int(gray_image.width())
    height = int(gray_image.height())
    sample_count = max(
        5, int(getattr(cfg, "DIVIDER_LINE_SAMPLE_COUNT", 25))
    )
    result["sample_count"] = sample_count
    half_range = float(
        getattr(cfg, "DIVIDER_SEARCH_HALF_RANGE_MM", 8.0)
    )
    search_step_mm = max(
        0.35,
        float(
            getattr(cfg, "SOURCE_DIVIDER_SEARCH_STEP_MM", 0.75)
        ),
    )
    search_steps = max(3, int(2.0 * half_range / search_step_mm + 0.5))
    offsets = [
        -half_range + 2.0 * half_range * index / search_steps
        for index in range(search_steps + 1)
    ]
    minimum_gray = float(
        getattr(
            cfg,
            "PIECE_DIVIDER_MIN_GRAY",
            getattr(cfg, "DIVIDER_LINE_MIN_GRAY", 70),
        )
    )
    minimum_contrast = float(
        getattr(
            cfg,
            "DIVIDER_LINE_MIN_CONTRAST_GRAY",
            35,
        )
    )
    max_thickness = float(
        getattr(cfg, "PIECE_DIVIDER_MAX_THICKNESS_MM", 5.0)
    )
    hits = []
    contrasts = []
    thicknesses = []
    source_points = []
    for sample_index in range(sample_count):
        fraction = (
            0.04
            + 0.92 * sample_index / max(1, sample_count - 1)
        )
        x_mm = fraction * mapper.a4_width_mm
        values = []
        points = []
        for offset in offsets:
            point = mapper.a4_mm_to_source_px(
                (x_mm, cfg.DIVIDER_Y_MM + offset)
            )
            if point is None:
                continue
            points.append(point)
            values.append(
                _source_divider_gray_at(
                    gray_array, width, height, point
                )
            )
        if len(values) < 5:
            continue
        background_count = max(2, len(values) // 3)
        background = _source_divider_median(
            sorted(values)[:background_count]
        )
        peak_index = max(range(len(values)), key=lambda index: values[index])
        peak = values[peak_index]
        contrast = peak - background
        if peak < minimum_gray or contrast < minimum_contrast:
            continue
        run_threshold = background + max(6.0, 0.45 * minimum_contrast)
        run_start = peak_index
        run_end = peak_index
        while run_start > 0 and values[run_start - 1] >= run_threshold:
            run_start -= 1
        while run_end + 1 < len(values) and values[run_end + 1] >= run_threshold:
            run_end += 1
        thickness_mm = (
            (run_end - run_start + 1) * search_step_mm
        )
        if thickness_mm > max_thickness + search_step_mm:
            continue
        weight_total = 0.0
        offset_total = 0.0
        for value_index in range(run_start, run_end + 1):
            weight = max(1.0, values[value_index] - background)
            weight_total += weight
            offset_total += offsets[value_index] * weight
        offset_mm = offset_total / max(1.0, weight_total)
        y_mm = cfg.DIVIDER_Y_MM + offset_mm
        source_point = mapper.a4_mm_to_source_px((x_mm, y_mm))
        if source_point is None:
            continue
        hits.append((x_mm, y_mm))
        source_points.append(source_point)
        contrasts.append(contrast)
        thicknesses.append(thickness_mm)
    result["hit_count"] = len(hits)
    result["coverage"] = float(len(hits)) / sample_count
    result["points_source_px"] = source_points
    if len(hits) < 2:
        result["reason"] = "insufficient_hits"
        return result
    mean_x = sum(point[0] for point in hits) / len(hits)
    mean_y = sum(point[1] for point in hits) / len(hits)
    xx = sum((point[0] - mean_x) ** 2 for point in hits)
    xy = sum(
        (point[0] - mean_x) * (point[1] - mean_y)
        for point in hits
    )
    slope_per_mm = xy / xx if xx > 1e-9 else 0.0
    intercept = mean_y - slope_per_mm * mean_x
    left_y = intercept
    right_y = intercept + slope_per_mm * mapper.a4_width_mm
    left_source = mapper.a4_mm_to_source_px((0.0, left_y))
    right_source = mapper.a4_mm_to_source_px(
        (mapper.a4_width_mm, right_y)
    )
    if left_source is None or right_source is None:
        result["reason"] = "projection_failed"
        return result
    residuals = [
        _source_divider_point_line_distance(
            point, left_source, right_source
        )
        for point in source_points
    ]
    max_residual = max(residuals)
    rms_residual = math.sqrt(
        sum(value * value for value in residuals) / len(residuals)
    )
    slope_mm = right_y - left_y
    contrast = sum(contrasts) / len(contrasts)
    thickness = _source_divider_median(thicknesses)
    result.update(
        {
            "divider_y_mm": mean_y,
            "intercept_y_mm": intercept,
            "slope_mm": slope_mm,
            "left_y_mm": left_y,
            "right_y_mm": right_y,
            "contrast": contrast,
            "residual_px": max_residual,
            "rms_residual_px": rms_residual,
            "thickness_mm": thickness,
        }
    )
    minimum_coverage = float(
        getattr(
            cfg,
            "DIVIDER_LINE_MIN_COVERAGE",
            getattr(cfg, "PIECE_DIVIDER_MIN_COVERAGE", 0.70),
        )
    )
    max_residual_allowed = float(
        getattr(
            cfg,
            "DIVIDER_LINE_MAX_RESIDUAL_PX",
            getattr(cfg, "PIECE_DIVIDER_MAX_RESIDUAL_PX", 2.5),
        )
    )
    max_slope = float(
        getattr(
            cfg,
            "DIVIDER_LINE_MAX_SLOPE_MM",
            getattr(cfg, "PIECE_DIVIDER_MAX_SLOPE_MM", 5.0),
        )
    )
    if result["coverage"] < minimum_coverage:
        result["reason"] = "coverage"
    elif contrast < minimum_contrast:
        result["reason"] = "contrast"
    elif max_residual > max_residual_allowed:
        result["reason"] = "residual"
    elif abs(slope_mm) > max_slope:
        result["reason"] = "slope"
    elif thickness > max_thickness + search_step_mm:
        result["reason"] = "thickness"
    else:
        result["detected"] = True
        result["reason"] = "ok"
        coverage_score = min(1.0, result["coverage"] / max(0.01, minimum_coverage))
        contrast_score = min(1.0, contrast / max(1.0, minimum_contrast * 1.5))
        residual_score = max(
            0.0, 1.0 - rms_residual / max(0.01, max_residual_allowed)
        )
        result["confidence"] = (
            0.45 * coverage_score
            + 0.35 * contrast_score
            + 0.20 * residual_score
        )
    return result


class SourceDividerTracker:
    """Hold a real divider briefly, but never replace it with a nominal line."""

    __slots__ = ("generation", "misses", "state_value")

    def __init__(self):
        self.reset()

    def reset(self, generation=None):
        self.generation = generation
        self.misses = 0
        self.state_value = None

    def update(self, detected, generation):
        if self.generation != generation:
            self.reset(generation)
        if detected is not None and detected.get("detected", False):
            alpha = (
                1.0
                if self.state_value is None
                else float(getattr(cfg, "DIVIDER_TRACK_ALPHA", 0.35))
            )
            next_state = dict(detected)
            if self.state_value is not None:
                for name in (
                    "divider_y_mm",
                    "intercept_y_mm",
                    "slope_mm",
                    "left_y_mm",
                    "right_y_mm",
                ):
                    next_state[name] = self.state_value[name] + alpha * (
                        detected[name] - self.state_value[name]
                    )
            next_state["held"] = False
            next_state["generation"] = generation
            self.state_value = next_state
            self.misses = 0
        else:
            self.misses += 1
            hold = max(
                0,
                int(
                    getattr(
                        cfg,
                        "SOURCE_PROJECTIVE_DIVIDER_HOLD_MISSES",
                        2,
                    )
                ),
            )
            if self.state_value is None or self.misses > hold:
                self.state_value = None
        if self.state_value is None:
            return {
                "detected": False,
                "reason": (
                    detected.get("reason", "missing")
                    if detected is not None
                    else "missing"
                ),
                "misses": self.misses,
                "generation": generation,
            }
        result = dict(self.state_value)
        result["held"] = self.misses > 0
        result["misses"] = self.misses
        return result


def _polygon_rows(points, width, height):
    rows = [None for _ in range(height)]
    if points is None or len(points) < 3:
        return rows
    minimum_y = max(0, int(math.floor(min(point[1] for point in points))))
    maximum_y = min(
        height - 1, int(math.ceil(max(point[1] for point in points)))
    )
    for y in range(minimum_y, maximum_y + 1):
        scan_y = y + 0.5
        intersections = []
        for index, point in enumerate(points):
            other = points[(index + 1) % len(points)]
            y0 = float(point[1])
            y1 = float(other[1])
            if (y0 <= scan_y < y1) or (y1 <= scan_y < y0):
                ratio = (scan_y - y0) / (y1 - y0)
                intersections.append(
                    float(point[0])
                    + ratio * (float(other[0]) - float(point[0]))
                )
        if len(intersections) < 2:
            continue
        intersections.sort()
        x0 = max(0, int(math.ceil(intersections[0])))
        x1 = min(width - 1, int(math.floor(intersections[-1])))
        if x1 >= x0:
            rows[y] = (x0, x1)
    return rows


class SourceScanlineMask:
    """Precomputed A4 and half-page spans for every source-image row."""

    __slots__ = (
        "width",
        "height",
        "a4_rows",
        "top_rows",
        "bottom_rows",
        "source_rows",
        "source_side",
        "source_polygon_px",
        "divider_left_px",
        "divider_right_px",
    )

    def __init__(self, mapper, divider, source_side):
        self.width = mapper.source_width
        self.height = mapper.source_height
        self.source_side = source_side
        self.a4_rows = _polygon_rows(
            mapper.a4_polygon_source_px, self.width, self.height
        )
        left_y = float(divider["left_y_mm"])
        right_y = float(divider["right_y_mm"])
        margin = max(
            0.0,
            0.5 * float(divider.get("thickness_mm", 0.0))
            + float(getattr(cfg, "PIECE_DIVIDER_MASK_MARGIN_MM", 1.5)),
        )
        top_mm = [
            (0.0, 0.0),
            (mapper.a4_width_mm, 0.0),
            (mapper.a4_width_mm, max(0.0, right_y - margin)),
            (0.0, max(0.0, left_y - margin)),
        ]
        bottom_mm = [
            (0.0, min(mapper.a4_height_mm, left_y + margin)),
            (
                mapper.a4_width_mm,
                min(mapper.a4_height_mm, right_y + margin),
            ),
            (mapper.a4_width_mm, mapper.a4_height_mm),
            (0.0, mapper.a4_height_mm),
        ]
        top_px = [mapper.a4_mm_to_source_px(point) for point in top_mm]
        bottom_px = [mapper.a4_mm_to_source_px(point) for point in bottom_mm]
        self.top_rows = _polygon_rows(top_px, self.width, self.height)
        self.bottom_rows = _polygon_rows(bottom_px, self.width, self.height)
        self.source_rows = (
            self.top_rows if source_side == "top" else self.bottom_rows
        )
        self.source_polygon_px = top_px if source_side == "top" else bottom_px
        self.divider_left_px = mapper.a4_mm_to_source_px((0.0, left_y))
        self.divider_right_px = mapper.a4_mm_to_source_px(
            (mapper.a4_width_mm, right_y)
        )

    def blacken_outside_source(self, gray_array):
        masked = 0
        for y in range(self.height):
            span = self.source_rows[y]
            row = gray_array[y]
            if span is None:
                for x in range(self.width):
                    if int(row[x]) != 0:
                        row[x] = 0
                        masked += 1
                continue
            x0, x1 = span
            for x in range(x0):
                if int(row[x]) != 0:
                    row[x] = 0
                    masked += 1
            for x in range(x1 + 1, self.width):
                if int(row[x]) != 0:
                    row[x] = 0
                    masked += 1
        return masked


def _sample_half(gray_array, rows, stride, threshold):
    bright = 0
    samples = 0
    for y in range(0, len(rows), stride):
        span = rows[y]
        if span is None:
            continue
        row = gray_array[y]
        for x in range(span[0], span[1] + 1, stride):
            samples += 1
            if int(row[x]) >= threshold:
                bright += 1
    return bright, samples


def estimate_source_half(
    gray_image,
    mapper,
    divider,
    threshold,
    scanline_mask=None,
):
    """Choose the half with clearly greater bright-fragment area."""
    neutral_mask = (
        scanline_mask
        if scanline_mask is not None
        else SourceScanlineMask(mapper, divider, "top")
    )
    gray_array = gray_image.to_numpy_ref()
    stride = max(
        1, int(getattr(cfg, "SOURCE_HALF_SAMPLE_STRIDE", 3))
    )
    bright_top, samples_top = _sample_half(
        gray_array, neutral_mask.top_rows, stride, threshold
    )
    bright_bottom, samples_bottom = _sample_half(
        gray_array, neutral_mask.bottom_rows, stride, threshold
    )
    ratio_top = float(bright_top) / max(1, samples_top)
    ratio_bottom = float(bright_bottom) / max(1, samples_bottom)
    maximum = max(ratio_top, ratio_bottom)
    difference = abs(ratio_top - ratio_bottom)
    confidence = difference / max(1e-6, maximum)
    minimum_confidence = float(
        getattr(cfg, "SOURCE_HALF_MIN_CONFIDENCE", 0.18)
    )
    minimum_bright = max(
        1, int(getattr(cfg, "SOURCE_HALF_MIN_BRIGHT_SAMPLES", 8))
    )
    valid = (
        max(bright_top, bright_bottom) >= minimum_bright
        and confidence >= minimum_confidence
    )
    return {
        "valid": valid,
        "side": (
            "top" if ratio_top > ratio_bottom else "bottom"
        ) if valid else None,
        "bright_top": bright_top,
        "bright_bottom": bright_bottom,
        "ratio_top": ratio_top,
        "ratio_bottom": ratio_bottom,
        "confidence": confidence,
        "reason": "ok" if valid else "ambiguous_bright_area",
    }
