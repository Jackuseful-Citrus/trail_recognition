"""CanMV-native piece recognition without OpenCV or NumPy."""

import math
import os

import puzzle_config as cfg
from puzzle_perf import PERF_STATS
from puzzle_geometry import (
    PieceObservation,
    edge_lengths,
    ensure_clockwise,
    polygon_area,
    polygon_centroid,
    polygon_is_convex,
    polygon_is_simple,
    remove_near_collinear_vertices,
)


def _vision_exitpoint(counter, interval=256):
    """Let CanMV IDE Stop interrupt long pure-Python pixel loops."""
    if counter % interval != 0:
        return
    exitpoint = getattr(os, "exitpoint", None)
    if exitpoint is not None:
        exitpoint()


def _reduce_polygon_to_limit(points, maximum):
    result = list(points)
    while len(result) > maximum:
        best_index = None
        best_importance = None
        for index in range(len(result)):
            prev = result[(index - 1) % len(result)]
            cur = result[index]
            nxt = result[(index + 1) % len(result)]
            triangle_twice_area = abs(
                (cur[0] - prev[0]) * (nxt[1] - prev[1])
                - (cur[1] - prev[1]) * (nxt[0] - prev[0])
            )
            if best_importance is None or triangle_twice_area < best_importance:
                best_importance = triangle_twice_area
                best_index = index
        del result[best_index]
    return result


def _rdp_open(points, tolerance):
    """Iterative Douglas-Peucker simplification for one open point chain."""
    if len(points) <= 2:
        return list(points)
    keep = bytearray(len(points))
    keep[0] = 1
    keep[-1] = 1
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        a = points[start]
        b = points[end]
        farthest = None
        maximum = -1.0
        for index in range(start + 1, end):
            distance = _point_line_distance(points[index], a, b)
            if distance > maximum:
                maximum = distance
                farthest = index
        if farthest is not None and maximum > tolerance:
            keep[farthest] = 1
            stack.append((start, farthest))
            stack.append((farthest, end))
    return [
        point for index, point in enumerate(points) if keep[index]
    ]


def _deduplicate_ordered_points(points):
    result = []
    for point in points:
        value = (float(point[0]), float(point[1]))
        if result and value == result[-1]:
            continue
        result.append(value)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result


def _simplify_closed_contour(points, tolerance):
    """Simplify a closed ordered contour while preserving concave vertices."""
    values = _deduplicate_ordered_points(points)
    if len(values) <= cfg.MAX_POLYGON_VERTICES:
        return values
    canonical_index = min(
        range(len(values)),
        key=lambda index: (
            values[index][1],
            values[index][0],
        ),
    )
    values = (
        values[canonical_index:]
        + values[:canonical_index]
    )
    anchor = values[0]
    split = max(
        range(1, len(values)),
        key=lambda index: (
            (values[index][0] - anchor[0]) ** 2
            + (values[index][1] - anchor[1]) ** 2
        ),
    )
    first = _rdp_open(values[: split + 1], tolerance)
    second = _rdp_open(
        values[split:] + [values[0]], tolerance
    )
    return _deduplicate_ordered_points(
        first[:-1] + second[:-1]
    )


def _fit_line(points):
    """Return a total-least-squares line and maximum residual."""
    if len(points) < 2:
        return None
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    xx = 0.0
    yy = 0.0
    xy = 0.0
    for x, y in points:
        dx = x - mean_x
        dy = y - mean_y
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
    if xx + yy <= 1e-12:
        return None
    angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
    direction = (math.cos(angle), math.sin(angle))
    maximum = 0.0
    for point in points:
        residual = abs(
            direction[0] * (mean_y - point[1])
            - direction[1] * (mean_x - point[0])
        )
        maximum = max(maximum, residual)
    return ((mean_x, mean_y), direction, maximum)


def _infinite_line_intersection(line_a, line_b):
    point_a, direction_a, _ = line_a
    point_b, direction_b, _ = line_b
    denominator = (
        direction_a[0] * direction_b[1]
        - direction_a[1] * direction_b[0]
    )
    if abs(denominator) <= 1e-8:
        return None
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    scale = (
        dx * direction_b[1] - dy * direction_b[0]
    ) / denominator
    return (
        point_a[0] + scale * direction_a[0],
        point_a[1] + scale * direction_a[1],
    )


def _refine_vertices_with_lines(contour, polygon):
    """Fit each polygon side and intersect adjacent fitted lines."""
    if len(polygon) < 3:
        return list(polygon)
    groups = [[] for _ in polygon]
    for point in contour:
        distances = [
            (
                _point_line_distance(
                    point,
                    polygon[index],
                    polygon[(index + 1) % len(polygon)],
                ),
                index,
            )
            for index in range(len(polygon))
        ]
        distances.sort()
        # A vertex belongs equally to two adjacent sides. Excluding numerical
        # ties keeps the fitted lines independent of ring start/direction.
        if (
            len(distances) > 1
            and abs(distances[0][0] - distances[1][0])
            <= cfg.GEOMETRY_EPSILON_MM
        ):
            continue
        nearest = distances[0][1]
        groups[nearest].append(point)
    lines = []
    for index, points in enumerate(groups):
        fitted = (
            _fit_line(points)
            if len(points) >= cfg.LINE_FIT_MIN_POINTS
            else None
        )
        if (
            fitted is None
            or fitted[2] > cfg.LINE_FIT_MAX_ERROR_MM
        ):
            a = polygon[index]
            b = polygon[(index + 1) % len(polygon)]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length <= 1e-9:
                return list(polygon)
            fitted = (
                a,
                (dx / length, dy / length),
                0.0,
            )
        lines.append(fitted)
    result = []
    # Line fitting should reduce raster jitter, not move a hand-cut vertex by
    # several millimetres and thereby change the puzzle geometry.
    maximum_shift = cfg.LINE_REFINE_MAX_SHIFT_MM
    for index, original in enumerate(polygon):
        point = _infinite_line_intersection(
            lines[(index - 1) % len(lines)],
            lines[index],
        )
        if point is None:
            result.append(original)
            continue
        dx = point[0] - original[0]
        dy = point[1] - original[1]
        if math.sqrt(dx * dx + dy * dy) > maximum_shift:
            result.append(original)
        else:
            result.append(point)
    original_area = polygon_area(polygon)
    refined_area = polygon_area(result)
    if (
        original_area <= 1e-9
        or abs(refined_area - original_area) / original_area
        > cfg.LINE_REFINE_MAX_AREA_CHANGE_RATIO
    ):
        return list(polygon)
    return result


def _finalize_fitted_polygon(
    contour_mm, polygon, refine_lines=True
):
    """Optionally refine, then validate an ordered simplified polygon."""
    if refine_lines:
        polygon = _refine_vertices_with_lines(
            contour_mm, polygon
        )
    polygon = remove_near_collinear_vertices(
        polygon,
        tolerance_deg=cfg.VERTEX_COLLINEAR_ANGLE_TOLERANCE_DEG,
        min_edge_mm=cfg.VERTEX_MERGE_DISTANCE_MM,
    )
    if len(polygon) > cfg.MAX_POLYGON_VERTICES:
        polygon = _reduce_polygon_to_limit(
            polygon, cfg.MAX_POLYGON_VERTICES
        )
        # Reduction can turn a multi-point reflection notch into one exactly
        # collinear residual point.  Run the same guarded cleanup once more so
        # that residual does not become a false physical vertex.
        polygon = remove_near_collinear_vertices(
            polygon,
            tolerance_deg=(
                cfg.VERTEX_COLLINEAR_ANGLE_TOLERANCE_DEG
            ),
            min_edge_mm=cfg.VERTEX_MERGE_DISTANCE_MM,
        )
    if not (
        cfg.MIN_POLYGON_VERTICES
        <= len(polygon)
        <= cfg.MAX_POLYGON_VERTICES
    ):
        return None
    if not polygon_is_simple(polygon):
        return None
    if min(edge_lengths(polygon)) <= cfg.GEOMETRY_EPSILON_MM:
        return None
    if polygon_area(polygon) < cfg.MIN_PIECE_AREA_MM2:
        return None
    return ensure_clockwise(polygon)


def _ordered_contour_polygon_once(
    points_mm, tolerance, refine_lines=True
):
    """Run one direction-sensitive closed-contour polygon fit."""
    cleanup_limit = (
        cfg.MAX_POLYGON_VERTICES
        + cfg.VERTEX_CLEANUP_EXTRA_VERTICES
    )
    polygon = []
    for _ in range(10):
        polygon = _simplify_closed_contour(
            points_mm, tolerance
        )
        if len(polygon) <= cleanup_limit:
            break
        tolerance *= 1.35
    if len(polygon) > cleanup_limit:
        polygon = _reduce_polygon_to_limit(
            polygon, cleanup_limit
        )
    return _finalize_fitted_polygon(
        points_mm,
        polygon,
        refine_lines=refine_lines,
    )


def _ordered_contour_polygon(
    points_mm, fit_diagnostics=None, tolerance_mm=None
):
    """Fit a validated 3..5 vertex polygon from an ordered outer contour."""
    tolerance = max(
        0.05,
        cfg.CONTOUR_DP_TOLERANCE_MM
        if tolerance_mm is None
        else float(tolerance_mm),
    )
    polygon = _ordered_contour_polygon_once(
        points_mm, tolerance
    )
    if polygon is not None:
        if fit_diagnostics is not None:
            fit_diagnostics["polygon_fit_method"] = "forward"
            fit_diagnostics["polygon_fit_reverse_used"] = False
            fit_diagnostics["polygon_fit_unrefined_used"] = False
        return polygon

    # Closed Douglas-Peucker fitting can be direction-sensitive at the chosen
    # ring seam. Retrying the identical measured contour in reverse order
    # changes neither its pixels nor its convexity and is safer than forcing a
    # convex hull or changing the threshold.
    reversed_points = list(reversed(points_mm))
    polygon = _ordered_contour_polygon_once(
        reversed_points, tolerance
    )
    if polygon is not None:
        if fit_diagnostics is not None:
            fit_diagnostics["polygon_fit_method"] = "reverse"
            fit_diagnostics["polygon_fit_reverse_used"] = True
            fit_diagnostics["polygon_fit_unrefined_used"] = False
        return polygon

    # A valid Douglas-Peucker polygon can occasionally be made invalid when
    # neighbouring least-squares lines intersect on the wrong side of a short
    # or acute edge. In that case retain the measured simplified vertices.
    # This does not invent a convex hull, relax geometry validation, or change
    # segmentation; the same simple-polygon and area checks still apply.
    polygon = _ordered_contour_polygon_once(
        points_mm, tolerance, refine_lines=False
    )
    reverse_used = False
    if polygon is None:
        polygon = _ordered_contour_polygon_once(
            reversed_points, tolerance, refine_lines=False
        )
        reverse_used = polygon is not None
    if fit_diagnostics is not None:
        if polygon is None:
            method = "invalid"
        elif reverse_used:
            method = "reverse_unrefined"
        else:
            method = "forward_unrefined"
        fit_diagnostics["polygon_fit_method"] = method
        fit_diagnostics["polygon_fit_reverse_used"] = reverse_used
        fit_diagnostics["polygon_fit_unrefined_used"] = (
            polygon is not None
        )
    return polygon


def _blob_value(blob, method_name, tuple_index=None):
    """Read a CanMV blob method, with tuple-index compatibility."""
    method = getattr(blob, method_name, None)
    if method is not None:
        return method()
    if tuple_index is None:
        raise AttributeError("blob has no {}".format(method_name))
    return blob[tuple_index]


def _diagnostic_scalar(value):
    if isinstance(value, (tuple, list)):
        return "|".join(str(item) for item in value)
    return str(value).replace(",", ";")


def _native_gray_sanity(gray_image):
    """Read native image health after find_blobs without changing its input."""
    result = {
        "format": "na",
        "width": int(gray_image.width()),
        "height": int(gray_image.height()),
        "min": "na",
        "max": "na",
        "mean": "na",
        "median": "na",
        "p00": "na",
        "pc": "na",
        "pt": "na",
        "pb": "na",
        "error": "",
    }
    errors = []
    try:
        format_method = getattr(gray_image, "format", None)
        if format_method is not None:
            value = (
                format_method()
                if callable(format_method)
                else format_method
            )
            result["format"] = _diagnostic_scalar(value)
    except Exception as exc:
        errors.append("format:{}".format(str(exc)))
    try:
        stats = gray_image.get_statistics()
        for name in ("min", "max", "mean", "median"):
            method = getattr(stats, name, None)
            if method is not None:
                value = method() if callable(method) else method
                result[name] = _diagnostic_scalar(value)
    except Exception as exc:
        errors.append("statistics:{}".format(str(exc)))
    width = result["width"]
    height = result["height"]
    samples = {
        "p00": (min(2, width - 1), min(2, height - 1)),
        "pc": (width // 2, height // 2),
        "pt": (width // 2, height // 4),
        "pb": (width // 2, 3 * height // 4),
    }
    for name, point in samples.items():
        try:
            result[name] = _diagnostic_scalar(
                gray_image.get_pixel(point[0], point[1])
            )
        except Exception as exc:
            errors.append("{}:{}".format(name, str(exc)))
    result["error"] = "|".join(errors).replace(",", ";")
    return result


def _array_gray_region_sanity(gray_array, roi, threshold):
    """Measure the shared array in one ROI after native blob extraction."""
    array_height = int(gray_array.shape[0])
    array_width = int(gray_array.shape[1])
    x0 = max(0, int(roi[0]))
    y0 = max(0, int(roi[1]))
    x1 = min(array_width, x0 + max(0, int(roi[2])))
    y1 = min(array_height, y0 + max(0, int(roi[3])))
    minimum = 255
    maximum = 0
    total = 0
    count = 0
    bright = 0
    bright_x0 = x1
    bright_y0 = y1
    bright_x1 = -1
    bright_y1 = -1
    for y in range(y0, y1):
        _vision_exitpoint(y - y0, interval=16)
        row = gray_array[y]
        for x in range(x0, x1):
            value = max(0, min(255, int(row[x])))
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            total += value
            count += 1
            if value >= threshold:
                bright += 1
                bright_x0 = min(bright_x0, x)
                bright_y0 = min(bright_y0, y)
                bright_x1 = max(bright_x1, x)
                bright_y1 = max(bright_y1, y)
    if count <= 0:
        minimum = 0
    bbox = (
        (
            bright_x0,
            bright_y0,
            bright_x1 - bright_x0 + 1,
            bright_y1 - bright_y0 + 1,
        )
        if bright > 0
        else None
    )
    return {
        "min": minimum,
        "max": maximum,
        "mean": float(total) / count if count else 0.0,
        "pixels": count,
        "bright": bright,
        "bright_bbox": bbox,
    }


def estimate_background_gray(
    gray_array,
    roi,
    sample_stride=4,
    histogram_bins=64,
):
    """Estimate the dominant A4 background from a sparsely sampled ROI.

    The physical lower half is empty during acquisition and remains mostly
    background during placement. Median and the 60th percentile are therefore
    robust to black divider pixels and to the target rectangle covering up to
    roughly 36 percent of the lower A4 half.
    """
    array_height = int(gray_array.shape[0])
    array_width = int(gray_array.shape[1])
    x0 = max(0, int(roi[0]))
    y0 = max(0, int(roi[1]))
    x1 = min(
        array_width, x0 + max(0, int(roi[2]))
    )
    y1 = min(
        array_height, y0 + max(0, int(roi[3]))
    )
    stride = max(1, int(sample_stride))
    bin_count = max(8, min(256, int(histogram_bins)))
    counts = [0 for _ in range(bin_count)]
    sums = [0 for _ in range(bin_count)]
    sample_count = 0
    for y in range(y0, y1, stride):
        _vision_exitpoint(y - y0, interval=16)
        row = gray_array[y]
        for x in range(x0, x1, stride):
            value = max(0, min(255, int(row[x])))
            index = min(
                bin_count - 1,
                value * bin_count // 256,
            )
            counts[index] += 1
            sums[index] += value
            sample_count += 1

    def percentile(fraction):
        if sample_count <= 0:
            return 0.0
        target = max(
            1, int(sample_count * fraction + 0.5)
        )
        cumulative = 0
        for index, count in enumerate(counts):
            cumulative += count
            if cumulative >= target:
                if count > 0:
                    return float(sums[index]) / count
                return (index + 0.5) * 256.0 / bin_count
        return 255.0

    background = percentile(0.50)
    upper_background = percentile(0.60)
    return {
        "background_gray": background,
        "background_high_gray": upper_background,
        "background_spread_gray": max(
            0.0, upper_background - background
        ),
        "sample_count": sample_count,
        "roi": (x0, y0, x1 - x0, y1 - y0),
    }


def background_difference_threshold(
    background_stats,
    minimum_delta_gray,
    noise_margin_gray,
    maximum_delta_gray,
):
    """Convert a robust background estimate into a bright-piece threshold."""
    background = float(
        background_stats.get("background_gray", 0.0)
    )
    spread = max(
        0.0,
        float(
            background_stats.get(
                "background_spread_gray", 0.0
            )
        ),
    )
    delta = max(
        float(minimum_delta_gray),
        spread + float(noise_margin_gray),
    )
    delta = min(float(maximum_delta_gray), delta)
    threshold = int(background + delta + 0.5)
    return max(0, min(250, threshold))


def blacken_rectified_border(gray_array, border_px):
    """Set an outer rectified-image safety band to black in place."""
    height = int(gray_array.shape[0])
    width = int(gray_array.shape[1])
    border = max(
        0,
        min(
            int(border_px),
            max(0, (min(width, height) - 1) // 2),
        ),
    )
    if border <= 0:
        return 0
    for y in range(height):
        row = gray_array[y]
        if y < border or y >= height - border:
            for x in range(width):
                row[x] = 0
        else:
            for x in range(border):
                row[x] = 0
                row[width - 1 - x] = 0
    return border


def _median_scalar(values):
    """Return a small-list median without NumPy/statistics dependencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return 0.5 * (
        float(ordered[middle - 1])
        + float(ordered[middle])
    )


def detect_rectified_divider_strip(
    gray_array,
    nominal_y_mm=None,
    margin_x_px=0,
):
    """Find a thin bright separator near the rectified A4 midpoint.

    Only sparse vertical scans are used.  A candidate must be thin at every
    sampled column, cover most of the page width, and fit one near-horizontal
    line.  This runs after perspective correction and before ``find_blobs``.
    """
    height = int(gray_array.shape[0])
    width = int(gray_array.shape[1])
    nominal_mm = (
        cfg.DIVIDER_Y_MM
        if nominal_y_mm is None
        else max(
            0.0,
            min(cfg.A4_HEIGHT_MM, float(nominal_y_mm)),
        )
    )
    result = {
        "enabled": bool(
            getattr(
                cfg,
                "PIECE_DIVIDER_DETECTION_ENABLED",
                False,
            )
        ),
        "detected": False,
        "line_found": False,
        "reason": "",
        "divider_y_mm": nominal_mm,
        "center_y_px": 0.0,
        "intercept_px": 0.0,
        "slope_px_per_x": 0.0,
        "slope_mm": 0.0,
        "coverage": 0.0,
        "residual_px": 0.0,
        "thickness_px": 0.0,
        "threshold": 0,
        "background_gray": 0.0,
        "sample_count": 0,
        "hit_count": 0,
        "inlier_count": 0,
        "search_rows": (0, 0),
    }
    if not result["enabled"]:
        result["reason"] = "disabled"
        return result
    if width < 8 or height < 8:
        result["reason"] = "image_size"
        return result

    pixels_per_mm_y = float(height - 1) / cfg.A4_HEIGHT_MM
    nominal_y_px = nominal_mm * pixels_per_mm_y
    result["center_y_px"] = nominal_y_px
    search_half_mm = float(
        getattr(
            cfg,
            "PIECE_DIVIDER_SEARCH_HALF_RANGE_MM",
            cfg.DIVIDER_SEARCH_HALF_RANGE_MM,
        )
    )
    search_half_px = max(
        2, int(search_half_mm * pixels_per_mm_y + 0.5)
    )
    search_start = max(
        1, int(nominal_y_px + 0.5) - search_half_px
    )
    search_end = min(
        height - 2,
        int(nominal_y_px + 0.5) + search_half_px,
    )
    result["search_rows"] = (search_start, search_end)
    if search_end - search_start < 4:
        result["reason"] = "search_band"
        return result

    edge_inset = max(
        int(margin_x_px),
        int(0.05 * float(width - 1) + 0.5),
    )
    x_start = min(width - 2, max(1, edge_inset))
    x_end = max(
        x_start + 1,
        min(width - 2, width - 1 - edge_inset),
    )
    if x_end - x_start < 4:
        result["reason"] = "search_width"
        return result

    background_stats = estimate_background_gray(
        gray_array,
        (
            x_start,
            search_start,
            x_end - x_start + 1,
            search_end - search_start + 1,
        ),
        sample_stride=2,
        histogram_bins=32,
    )
    background_gray = float(
        background_stats.get("background_gray", 0.0)
    )
    bright_threshold = max(
        float(
            getattr(cfg, "PIECE_DIVIDER_MIN_GRAY", 50)
        ),
        background_gray
        + float(
            getattr(
                cfg,
                "PIECE_DIVIDER_MIN_CONTRAST_GRAY",
                20,
            )
        ),
    )
    bright_threshold = max(
        0, min(255, int(bright_threshold + 0.5))
    )
    result["threshold"] = bright_threshold
    result["background_gray"] = background_gray

    requested_samples = max(
        7,
        int(getattr(cfg, "PIECE_DIVIDER_SAMPLE_COUNT", 31)),
    )
    sample_count = min(
        requested_samples, x_end - x_start + 1
    )
    result["sample_count"] = sample_count
    maximum_thickness_px = max(
        1,
        int(
            float(
                getattr(
                    cfg,
                    "PIECE_DIVIDER_MAX_THICKNESS_MM",
                    5.0,
                )
            )
            * pixels_per_mm_y
            + 0.5
        ),
    )
    samples = []
    for index in range(sample_count):
        _vision_exitpoint(index, interval=8)
        x = int(
            x_start
            + (x_end - x_start)
            * index
            / max(1, sample_count - 1)
            + 0.5
        )
        runs = []
        run_start = None
        run_sum = 0.0
        run_count = 0
        for y in range(search_start, search_end + 2):
            value = (
                int(gray_array[y][x])
                if y <= search_end
                else -1
            )
            if value >= bright_threshold:
                if run_start is None:
                    run_start = y
                    run_sum = 0.0
                    run_count = 0
                run_sum += value
                run_count += 1
                continue
            if run_start is None:
                continue
            run_end = y - 1
            thickness = run_end - run_start + 1
            # A run clipped by either scan edge is usually a large fragment,
            # not a thin separator whose complete thickness is observable.
            if (
                run_start > search_start
                and run_end < search_end
                and thickness <= maximum_thickness_px
            ):
                center_y = 0.5 * (run_start + run_end)
                runs.append(
                    (
                        abs(center_y - nominal_y_px),
                        -(run_sum / max(1, run_count)),
                        center_y,
                        float(thickness),
                    )
                )
            run_start = None
            run_sum = 0.0
            run_count = 0
        if runs:
            best = min(runs)
            samples.append((float(x), best[2], best[3]))

    result["hit_count"] = len(samples)
    minimum_coverage = float(
        getattr(cfg, "PIECE_DIVIDER_MIN_COVERAGE", 0.72)
    )
    minimum_hits = max(
        5, int(math.ceil(sample_count * minimum_coverage))
    )
    if len(samples) < minimum_hits:
        result["reason"] = "coverage"
        result["coverage"] = float(len(samples)) / sample_count
        return result

    span = float(max(1, x_end - x_start))
    left = [
        item
        for item in samples
        if item[0] <= x_start + 0.30 * span
    ]
    right = [
        item
        for item in samples
        if item[0] >= x_start + 0.70 * span
    ]
    if len(left) < 2 or len(right) < 2:
        result["reason"] = "span"
        result["coverage"] = float(len(samples)) / sample_count
        return result
    sample_span = (
        max(item[0] for item in samples)
        - min(item[0] for item in samples)
    ) / span
    if sample_span < minimum_coverage:
        result["reason"] = "span"
        result["coverage"] = float(len(samples)) / sample_count
        return result

    left_x = _median_scalar([item[0] for item in left])
    right_x = _median_scalar([item[0] for item in right])
    left_y = _median_scalar([item[1] for item in left])
    right_y = _median_scalar([item[1] for item in right])
    if right_x - left_x <= 1.0:
        result["reason"] = "span"
        return result
    slope = (right_y - left_y) / (right_x - left_x)
    intercept = _median_scalar(
        [item[1] - slope * item[0] for item in samples]
    )
    maximum_residual_px = float(
        getattr(
            cfg,
            "PIECE_DIVIDER_MAX_RESIDUAL_PX",
            2.5,
        )
    )
    inliers = [
        item
        for item in samples
        if abs(item[1] - (intercept + slope * item[0]))
        <= maximum_residual_px
    ]
    result["inlier_count"] = len(inliers)
    result["coverage"] = float(len(inliers)) / sample_count
    if len(inliers) < minimum_hits:
        result["reason"] = "residual"
        return result

    mean_x = sum(item[0] for item in inliers) / len(inliers)
    mean_y = sum(item[1] for item in inliers) / len(inliers)
    denominator = sum(
        (item[0] - mean_x) * (item[0] - mean_x)
        for item in inliers
    )
    if denominator > 1e-9:
        slope = sum(
            (item[0] - mean_x) * (item[1] - mean_y)
            for item in inliers
        ) / denominator
    intercept = mean_y - slope * mean_x
    residual_px = sum(
        abs(item[1] - (intercept + slope * item[0]))
        for item in inliers
    ) / len(inliers)
    center_x = 0.5 * float(width - 1)
    center_y = intercept + slope * center_x
    if abs(center_y - nominal_y_px) > search_half_px + 1.0:
        result["reason"] = "position"
        return result
    slope_mm = slope * float(width - 1) / pixels_per_mm_y
    thickness_px = max(item[2] for item in inliers)
    result.update(
        {
            "line_found": True,
            "divider_y_mm": center_y / pixels_per_mm_y,
            "center_y_px": center_y,
            "intercept_px": intercept,
            "slope_px_per_x": slope,
            "slope_mm": slope_mm,
            "residual_px": residual_px,
            "thickness_px": thickness_px,
        }
    )
    if abs(slope_mm) > float(
        getattr(cfg, "PIECE_DIVIDER_MAX_SLOPE_MM", 5.0)
    ):
        result["reason"] = "slope"
        return result
    result["detected"] = True
    result["reason"] = "ok"
    return result


def mask_rectified_divider_strip(gray_array, divider):
    """Blacken the fitted separator and its interpolation halo in place."""
    if not divider.get("detected", False):
        return {"half_width_px": 0, "masked_pixels": 0}
    height = int(gray_array.shape[0])
    width = int(gray_array.shape[1])
    pixels_per_mm_y = float(height - 1) / cfg.A4_HEIGHT_MM
    half_width_px = max(
        1,
        int(
            math.ceil(
                0.5 * float(divider.get("thickness_px", 1.0))
                + float(
                    getattr(
                        cfg,
                        "PIECE_DIVIDER_MASK_MARGIN_MM",
                        1.5,
                    )
                )
                * pixels_per_mm_y
            )
        ),
    )
    intercept = float(divider.get("intercept_px", 0.0))
    slope = float(divider.get("slope_px_per_x", 0.0))
    masked_pixels = 0
    for x in range(width):
        _vision_exitpoint(x, interval=64)
        center_y = intercept + slope * x
        y_start = max(0, int(math.floor(center_y - half_width_px)))
        y_end = min(
            height - 1,
            int(math.ceil(center_y + half_width_px)),
        )
        for y in range(y_start, y_end + 1):
            gray_array[y][x] = 0
            masked_pixels += 1
    return {
        "half_width_px": half_width_px,
        "masked_pixels": masked_pixels,
    }


def trace_ordered_boundary(
    gray_array,
    rect,
    threshold,
    max_steps=None,
):
    """Trace one ordered outer boundary with the Moore-neighbour algorithm.

    Pixels outside ``rect`` or the array are treated as background.  The return
    value is ``(points, diagnostics)``; a failed trace returns an empty point
    list and an explicit reason so callers can fail closed.
    """
    array_height = int(gray_array.shape[0])
    array_width = int(gray_array.shape[1])
    x0 = max(0, int(rect[0]))
    y0 = max(0, int(rect[1]))
    x1 = min(array_width, x0 + max(1, int(rect[2])))
    y1 = min(array_height, y0 + max(1, int(rect[3])))
    width = x1 - x0
    height = y1 - y0
    diagnostics = {
        "ok": False,
        "reason": "",
        "pixel_reads": 0,
        "boundary_steps": 0,
        "max_steps": 0,
    }
    if width <= 0 or height <= 0:
        diagnostics["reason"] = "empty_bbox"
        return [], diagnostics

    def foreground(x, y):
        if x < x0 or x >= x1 or y < y0 or y >= y1:
            return False
        diagnostics["pixel_reads"] += 1
        return int(gray_array[y][x]) >= threshold

    start = None
    for y in range(y0, y1):
        _vision_exitpoint(y - y0, interval=16)
        for x in range(x0, x1):
            if foreground(x, y):
                start = (x, y)
                break
        if start is not None:
            break
    if start is None:
        diagnostics["reason"] = "no_foreground"
        return [], diagnostics

    # Clockwise order in image coordinates (positive y points down).
    directions = (
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    )
    if max_steps is None:
        max_steps = max(
            cfg.BOUNDARY_TRACE_MIN_POINTS * 2,
            2
            * (width + height)
            * cfg.BOUNDARY_TRACE_MAX_STEP_FACTOR,
        )
    max_steps = max(1, int(max_steps))
    diagnostics["max_steps"] = max_steps

    current = start
    backtrack = (start[0] - 1, start[1])
    first_next = None
    result = [start]

    for step in range(max_steps):
        _vision_exitpoint(step)
        relative = (
            backtrack[0] - current[0],
            backtrack[1] - current[1],
        )
        try:
            backtrack_index = directions.index(relative)
        except ValueError:
            diagnostics["reason"] = "invalid_backtrack"
            break

        next_point = None
        preceding = backtrack
        for offset in range(1, 9):
            direction = directions[
                (backtrack_index + offset) % 8
            ]
            candidate = (
                current[0] + direction[0],
                current[1] + direction[1],
            )
            if foreground(candidate[0], candidate[1]):
                next_point = candidate
                break
            preceding = candidate
        if next_point is None:
            diagnostics["reason"] = "isolated_pixel"
            break

        if first_next is None:
            first_next = next_point
        elif current == start and next_point == first_next:
            diagnostics["ok"] = True
            diagnostics["reason"] = "ok"
            break

        result.append(next_point)
        current = next_point
        backtrack = preceding
        diagnostics["boundary_steps"] += 1
    else:
        diagnostics["reason"] = "max_steps"

    if len(result) > 1 and result[-1] == result[0]:
        result.pop()
    result = _deduplicate_ordered_points(result)
    if diagnostics["ok"] and len(result) < 3:
        diagnostics["ok"] = False
        diagnostics["reason"] = "too_short"
    PERF_STATS.increment(
        "pixel_reads", diagnostics["pixel_reads"]
    )
    PERF_STATS.increment(
        "boundary_steps", diagnostics["boundary_steps"]
    )
    return [
        (float(point[0]), float(point[1])) for point in result
    ], diagnostics


class _ComponentMaskRow:
    __slots__ = ("data", "offset")

    def __init__(self, data, offset):
        self.data = data
        self.offset = int(offset)

    def __getitem__(self, x):
        return self.data[self.offset + int(x)]


class _ComponentMask:
    """Minimal 2-D bytearray view accepted by trace_ordered_boundary."""

    __slots__ = ("data", "shape", "width")

    def __init__(self, data, width, height):
        self.data = data
        self.width = int(width)
        self.shape = (int(height), int(width))

    def __getitem__(self, y):
        return _ComponentMaskRow(
            self.data, int(y) * self.width
        )


_COMPONENT_NEIGHBORS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def _flood_component_mask(
    gray_array,
    x0,
    y0,
    width,
    height,
    seed_index,
    threshold,
    allowed_mask=None,
    output_mask=None,
):
    """Return the 8-connected component containing one local seed."""
    component = (
        output_mask
        if output_mask is not None
        else bytearray(width * height)
    )
    component[seed_index] = 1
    stack = [seed_index]
    count = 0
    sum_x = 0
    sum_y = 0
    processed = 0
    while stack:
        processed += 1
        _vision_exitpoint(processed)
        local_index = stack.pop()
        local_y = local_index // width
        local_x = local_index - local_y * width
        count += 1
        sum_x += x0 + local_x
        sum_y += y0 + local_y
        for dx, dy in _COMPONENT_NEIGHBORS:
            nx = local_x + dx
            ny = local_y + dy
            if (
                nx < 0
                or nx >= width
                or ny < 0
                or ny >= height
            ):
                continue
            neighbor_index = ny * width + nx
            if component[neighbor_index]:
                continue
            if (
                allowed_mask is not None
                and not allowed_mask[neighbor_index]
            ):
                continue
            if int(gray_array[y0 + ny][x0 + nx]) < threshold:
                continue
            component[neighbor_index] = 1
            stack.append(neighbor_index)
    return component, count, sum_x, sum_y


def _select_component_seed(
    gray_array,
    x0,
    y0,
    width,
    height,
    threshold,
    center,
    expected_pixels=None,
    allowed_mask=None,
):
    """Choose the component matching native Blob size/centre evidence."""
    visited = bytearray(width * height)
    best_seed = None
    best_score = None
    best_count = 0
    best_center = None
    candidate_count = 0
    diagonal2 = max(1.0, float(width * width + height * height))
    expected = (
        max(1.0, float(expected_pixels))
        if expected_pixels is not None
        else None
    )
    for local_index in range(width * height):
        _vision_exitpoint(local_index)
        if visited[local_index]:
            continue
        local_y = local_index // width
        local_x = local_index - local_y * width
        if (
            allowed_mask is not None
            and not allowed_mask[local_index]
        ):
            visited[local_index] = 1
            continue
        if int(gray_array[y0 + local_y][x0 + local_x]) < threshold:
            visited[local_index] = 1
            continue
        _component, count, sum_x, sum_y = _flood_component_mask(
            gray_array,
            x0,
            y0,
            width,
            height,
            local_index,
            threshold,
            allowed_mask=allowed_mask,
            output_mask=visited,
        )
        candidate_count += 1
        component_center = (
            float(sum_x) / max(1, count),
            float(sum_y) / max(1, count),
        )
        center_distance2 = (
            (component_center[0] - float(center[0])) ** 2
            + (component_center[1] - float(center[1])) ** 2
        )
        if expected is None:
            # The high-threshold white core should be the largest component
            # inside the already selected low-threshold Blob component.
            score = (
                -float(count),
                center_distance2 / diagonal2,
            )
        else:
            score = (
                abs(float(count) - expected) / expected
                + 0.5 * center_distance2 / diagonal2,
                center_distance2 / diagonal2,
            )
        if best_score is None or score < best_score:
            best_score = score
            best_seed = local_index
            best_count = count
            best_center = component_center
    return {
        "seed_index": best_seed,
        "count": best_count,
        "center": best_center,
        "candidate_count": candidate_count,
    }


def _blob_component_boundary(
    gray_array,
    rect,
    center,
    blob_pixels,
    discovery_threshold,
    contour_threshold,
):
    """Bind a native low-threshold Blob to only its own white core.

    Bounding boxes may overlap even when native Blobs are distinct. Rebuilding
    the discovery-threshold components and matching their pixel count/centroid
    prevents two overlapping boxes from tracing the same high-threshold piece.
    """
    array_height = int(gray_array.shape[0])
    array_width = int(gray_array.shape[1])
    x0 = max(0, int(rect[0]))
    y0 = max(0, int(rect[1]))
    x1 = min(array_width, x0 + max(1, int(rect[2])))
    y1 = min(array_height, y0 + max(1, int(rect[3])))
    width = x1 - x0
    height = y1 - y0
    diagnostics = {
        "ok": False,
        "reason": "",
        "pixel_reads": 0,
        "boundary_steps": 0,
        "component_candidates": 0,
        "blob_component_pixels": 0,
        "contour_component_candidates": 0,
        "contour_component_pixels": 0,
        "blob_pixel_error_ratio": 1.0,
    }
    if width <= 0 or height <= 0:
        diagnostics["reason"] = "identity_empty_bbox"
        return [], diagnostics

    selected_blob = _select_component_seed(
        gray_array,
        x0,
        y0,
        width,
        height,
        discovery_threshold,
        center,
        expected_pixels=blob_pixels,
    )
    diagnostics["component_candidates"] = selected_blob[
        "candidate_count"
    ]
    if selected_blob["seed_index"] is None:
        diagnostics["reason"] = "identity_no_blob_component"
        return [], diagnostics
    blob_mask, blob_count, _sum_x, _sum_y = (
        _flood_component_mask(
            gray_array,
            x0,
            y0,
            width,
            height,
            selected_blob["seed_index"],
            discovery_threshold,
        )
    )
    diagnostics["blob_component_pixels"] = blob_count
    diagnostics["blob_pixel_error_ratio"] = abs(
        float(blob_count) - max(1.0, float(blob_pixels))
    ) / max(1.0, float(blob_pixels))

    selected_contour = _select_component_seed(
        gray_array,
        x0,
        y0,
        width,
        height,
        contour_threshold,
        center,
        allowed_mask=blob_mask,
    )
    diagnostics["contour_component_candidates"] = (
        selected_contour["candidate_count"]
    )
    if selected_contour["seed_index"] is None:
        diagnostics["reason"] = "identity_no_contour_component"
        return [], diagnostics
    contour_mask, contour_count, _sum_x, _sum_y = (
        _flood_component_mask(
            gray_array,
            x0,
            y0,
            width,
            height,
            selected_contour["seed_index"],
            contour_threshold,
            allowed_mask=blob_mask,
        )
    )
    diagnostics["contour_component_pixels"] = contour_count
    local_mask = _ComponentMask(contour_mask, width, height)
    boundary, trace = trace_ordered_boundary(
        local_mask,
        (0, 0, width, height),
        1,
    )
    diagnostics["ok"] = bool(trace["ok"])
    diagnostics["reason"] = (
        "identity_ok"
        if trace["ok"]
        else "identity_{}".format(trace["reason"])
    )
    diagnostics["pixel_reads"] = trace["pixel_reads"]
    diagnostics["boundary_steps"] = trace["boundary_steps"]
    if not trace["ok"]:
        return [], diagnostics
    return [
        (point[0] + x0, point[1] + y0)
        for point in boundary
    ], diagnostics


def _cross(origin, a, b):
    return (
        (a[0] - origin[0]) * (b[1] - origin[1])
        - (a[1] - origin[1]) * (b[0] - origin[0])
    )


def _convex_hull_points(points):
    """Andrew monotone-chain convex hull without NumPy/OpenCV."""
    ordered = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(ordered) <= 2:
        return ordered
    lower = []
    for point in ordered:
        while (
            len(lower) >= 2
            and _cross(lower[-2], lower[-1], point) <= 0.0
        ):
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while (
            len(upper) >= 2
            and _cross(upper[-2], upper[-1], point) <= 0.0
        ):
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _point_line_distance(point, a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 1e-9:
        return math.sqrt(
            (point[0] - a[0]) * (point[0] - a[0])
            + (point[1] - a[1]) * (point[1] - a[1])
        )
    return abs(
        dx * (a[1] - point[1])
        - (a[0] - point[0]) * dy
    ) / length


def _extract_canmv_polygon(
    gray_array,
    blob,
    discovery_threshold,
    threshold,
    pixels_per_mm_x,
    pixels_per_mm_y,
):
    """Extract one identity-bound CanMV component as a millimetre polygon."""
    rect = tuple(_blob_value(blob, "rect", None))
    center = (
        float(_blob_value(blob, "cx", 5)),
        float(_blob_value(blob, "cy", 6)),
    )
    blob_pixels = float(_blob_value(blob, "pixels", 4))
    contour_started = PERF_STATS.mark()
    boundary_px, trace_diagnostics = _blob_component_boundary(
        gray_array,
        rect,
        center,
        blob_pixels,
        discovery_threshold,
        threshold,
    )
    trace_diagnostics["boundary_primary_ok"] = bool(
        trace_diagnostics["ok"]
    )
    trace_diagnostics["boundary_fallback_used"] = False
    trace_diagnostics["boundary_fallback_ordered_ok"] = False
    trace_diagnostics["boundary_failure_reason"] = (
        ""
        if trace_diagnostics["ok"]
        else trace_diagnostics["reason"]
    )
    # A raw bounding-box fallback would reintroduce the exact ambiguity this
    # identity binding prevents, so failures remain fail-closed.
    trace_diagnostics["fallback"] = False
    PERF_STATS.add_stage("contour_ms", contour_started)
    if (
        not trace_diagnostics["ok"]
        or len(boundary_px) < cfg.MIN_POLYGON_VERTICES
    ):
        if not trace_diagnostics["boundary_failure_reason"]:
            trace_diagnostics["boundary_failure_reason"] = (
                "ordered_boundary_too_short"
            )
        return None, boundary_px, trace_diagnostics

    fit_started = PERF_STATS.mark()
    boundary_mm = [
        (
            point[0] / pixels_per_mm_x,
            point[1] / pixels_per_mm_y,
        )
        for point in boundary_px
    ]
    polygon_mm = _ordered_contour_polygon(
        boundary_mm, trace_diagnostics
    )
    if (
        cfg.FORCE_CONVEX_CONTOURS
        or (
            polygon_mm is not None
            and polygon_is_convex(polygon_mm)
        )
    ):
        stable = _ordered_contour_polygon(
            _convex_hull_points(boundary_mm)
        )
        if stable is not None:
            polygon_mm = stable
    PERF_STATS.add_stage("polygon_fit_ms", fit_started)
    if polygon_mm is None:
        return None, boundary_px, trace_diagnostics
    return polygon_mm, boundary_px, trace_diagnostics


def _piece_center_contour_threshold(
    gray_array,
    center,
    background_gray,
    discovery_threshold,
    fallback_threshold,
):
    """Return a lightweight per-piece contour threshold and diagnostics.

    Blob discovery has already established the component identity at the low
    threshold.  A small median sample around that Blob centroid therefore
    adapts only the white-core boundary trace and cannot add search candidates.
    """
    radius = max(
        0,
        int(
            getattr(
                cfg,
                "PIECE_CONTOUR_CENTER_SAMPLE_RADIUS_PX",
                2,
            )
        ),
    )
    height = int(gray_array.shape[0])
    width = int(gray_array.shape[1])
    center_x = max(
        0, min(width - 1, int(round(float(center[0]))))
    )
    center_y = max(
        0, min(height - 1, int(round(float(center[1]))))
    )
    samples = []
    for y in range(
        max(0, center_y - radius),
        min(height, center_y + radius + 1),
    ):
        for x in range(
            max(0, center_x - radius),
            min(width, center_x + radius + 1),
        ):
            samples.append(float(gray_array[y][x]))
    center_gray = _median_scalar(samples)
    fallback = max(0, min(255, int(fallback_threshold)))
    if not getattr(
        cfg, "PIECE_ADAPTIVE_CONTOUR_THRESHOLD_ENABLED", False
    ):
        return fallback, center_gray, "fixed"
    if background_gray is None:
        return fallback, center_gray, "fallback_no_background"
    minimum_contrast = float(
        getattr(
            cfg,
            "PIECE_CONTOUR_CENTER_MIN_CONTRAST_GRAY",
            40,
        )
    )
    if (
        center_gray < float(discovery_threshold) + minimum_contrast
        or center_gray <= float(background_gray) + minimum_contrast
    ):
        return fallback, center_gray, "fallback_low_contrast"
    alpha = max(
        0.0,
        min(
            1.0,
            float(
                getattr(
                    cfg,
                    "PIECE_CONTOUR_ADAPTIVE_ALPHA",
                    0.42,
                )
            ),
        ),
    )
    minimum = max(
        int(discovery_threshold),
        int(
            getattr(
                cfg,
                "PIECE_CONTOUR_ADAPTIVE_MIN_GRAY",
                85,
            )
        ),
    )
    maximum = max(
        minimum,
        min(
            255,
            int(
                getattr(
                    cfg,
                    "PIECE_CONTOUR_ADAPTIVE_MAX_GRAY",
                    140,
                )
            ),
        ),
    )
    adaptive = int(
        float(background_gray)
        + alpha * (center_gray - float(background_gray))
        + 0.5
    )
    return (
        max(minimum, min(maximum, adaptive)),
        center_gray,
        "adaptive",
    )


def detect_pieces_from_canmv_image(
    gray_image,
    corners_px,
    source_frame_size,
    region="upper",
    threshold=None,
    divider_y_mm=None,
    collect_sanity=False,
):
    """Detect pieces with CanMV v1.6 native image APIs, without ``cv2``.

    ``gray_image`` is a resized grayscale working image.  The calibrated A4
    corners are scaled from ``source_frame_size`` and perspective-corrected by
    the firmware's native ``rotation_corr`` implementation.
    """
    work_width = int(gray_image.width())
    work_height = int(gray_image.height())
    source_width = int(source_frame_size[0])
    source_height = int(source_frame_size[1])
    if work_width < 2 or work_height < 2:
        raise ValueError("invalid CanMV work image size")
    if source_width < 2 or source_height < 2:
        raise ValueError("invalid source frame size")
    scale_x = float(work_width - 1) / float(source_width - 1)
    scale_y = float(work_height - 1) / float(source_height - 1)
    work_corners = [
        (
            int(round(float(point[0]) * scale_x)),
            int(round(float(point[1]) * scale_y)),
        )
        for point in corners_px
    ]
    rectify_started = PERF_STATS.mark()
    original_gray_image = gray_image
    corrected = gray_image.rotation_corr(corners=work_corners)
    if corrected is None:
        rotation_return = "none"
    elif corrected is original_gray_image:
        rotation_return = "self"
    elif (
        hasattr(corrected, "width")
        and hasattr(corrected, "height")
        and hasattr(corrected, "find_blobs")
    ):
        gray_image = corrected
        rotation_return = "new"
    else:
        rotation_return = "invalid:{}".format(
            type(corrected).__name__
        )
    PERF_STATS.add_stage("rectify_ms", rectify_started)

    pixels_per_mm_x = float(work_width - 1) / cfg.A4_WIDTH_MM
    pixels_per_mm_y = float(work_height - 1) / cfg.A4_HEIGHT_MM
    divider_hint_y_mm = (
        cfg.DIVIDER_Y_MM
        if divider_y_mm is None
        else max(
            0.0,
            min(cfg.A4_HEIGHT_MM, float(divider_y_mm)),
        )
    )
    safety_border_px = max(
        0,
        int(
            getattr(
                cfg,
                "PIECE_RECTIFIED_BORDER_BLACK_PX",
                0,
            )
        ),
    )
    margin_x = max(
        1,
        safety_border_px,
        int(cfg.DETECTION_BORDER_MARGIN_MM * pixels_per_mm_x + 0.5),
    )
    margin_y = max(
        1,
        safety_border_px,
        int(cfg.DETECTION_BORDER_MARGIN_MM * pixels_per_mm_y + 0.5),
    )
    gray_array = gray_image.to_numpy_ref()
    safety_border_px = blacken_rectified_border(
        gray_array, safety_border_px
    )
    divider_probe = detect_rectified_divider_strip(
        gray_array,
        nominal_y_mm=divider_hint_y_mm,
        margin_x_px=margin_x,
    )
    divider_mask = {
        "half_width_px": 0,
        "masked_pixels": 0,
    }
    if divider_probe.get("detected", False):
        active_divider_y_mm = float(
            divider_probe["divider_y_mm"]
        )
        divider_source = "rectified_strip"
        divider_mask = mask_rectified_divider_strip(
            gray_array, divider_probe
        )
        intercept = float(divider_probe["intercept_px"])
        slope = float(divider_probe["slope_px_per_x"])
        left_y = intercept + slope * margin_x
        right_y = intercept + slope * (
            work_width - 1 - margin_x
        )
        half_width = int(divider_mask["half_width_px"])
        upper_end = max(
            2,
            int(math.floor(min(left_y, right_y) - half_width)),
        )
        lower_start = min(
            work_height - 2,
            int(math.ceil(max(left_y, right_y) + half_width + 1)),
        )
    else:
        active_divider_y_mm = divider_hint_y_mm
        divider_source = (
            "a4_hint"
            if divider_y_mm is not None
            else "nominal_fallback"
        )
        nominal_divider = int(
            active_divider_y_mm * pixels_per_mm_y + 0.5
        )
        fallback_gap_px = max(
            1,
            int(
                float(
                    getattr(
                        cfg,
                        "PIECE_DIVIDER_FALLBACK_GAP_MM",
                        2.0,
                    )
                )
                * pixels_per_mm_y
                + 0.5
            ),
        )
        upper_end = max(2, nominal_divider - fallback_gap_px)
        lower_start = min(
            work_height - 2,
            nominal_divider + fallback_gap_px,
        )
    segmentation_mode = getattr(
        cfg, "PIECE_SEGMENTATION_MODE", "fixed"
    )
    background_stats = {
        "background_gray": 0.0,
        "background_high_gray": 0.0,
        "background_spread_gray": 0.0,
        "sample_count": 0,
        "roi": (0, 0, 0, 0),
    }
    if segmentation_mode == "background_delta":
        calibration_start = min(
            work_height - margin_y,
            lower_start + margin_y,
        )
        background_stats = estimate_background_gray(
            gray_array,
            (
                margin_x,
                calibration_start,
                max(1, work_width - 2 * margin_x),
                max(
                    1,
                    work_height
                    - margin_y
                    - calibration_start,
                ),
            ),
            sample_stride=cfg.PIECE_BACKGROUND_SAMPLE_STRIDE,
            histogram_bins=(
                cfg.PIECE_BACKGROUND_HISTOGRAM_BINS
            ),
        )
    if (
        threshold is None
        and segmentation_mode == "background_delta"
        and background_stats["sample_count"]
        >= cfg.PIECE_BACKGROUND_MIN_SAMPLES
    ):
        piece_threshold = background_difference_threshold(
            background_stats,
            cfg.PIECE_BACKGROUND_DELTA_GRAY,
            cfg.PIECE_BACKGROUND_NOISE_MARGIN_GRAY,
            cfg.PIECE_BACKGROUND_MAX_DELTA_GRAY,
        )
        threshold_mode = "background_delta"
    elif threshold is None:
        piece_threshold = int(cfg.WHITE_GRAY_THRESHOLD)
        threshold_mode = (
            "background_delta_fallback"
            if segmentation_mode == "background_delta"
            else "fixed_native"
        )
    else:
        piece_threshold = max(
            0, min(255, int(threshold))
        )
        threshold_mode = (
            "background_delta_override"
            if segmentation_mode == "background_delta"
            else "fixed_native"
        )
    contour_threshold = piece_threshold
    if segmentation_mode == "background_delta":
        contour_threshold = max(
            contour_threshold,
            max(
                0,
                min(
                    255,
                    int(
                        getattr(
                            cfg,
                            "PIECE_CONTOUR_MIN_GRAY_THRESHOLD",
                            0,
                        )
                    ),
                ),
            ),
        )
    if region == "upper":
        detection_regions = [(margin_y, upper_end)]
    elif region == "lower":
        detection_regions = [
            (lower_start, work_height - margin_y)
        ]
    elif region == "full":
        # Keep the white divider out of both searches. A piece crossing it is
        # intentionally reconsidered at the next placement check.
        detection_regions = [
            (margin_y, upper_end),
            (lower_start, work_height - margin_y),
        ]
    else:
        raise ValueError(
            "unknown CanMV detection region {}".format(region)
        )
    min_pixels = max(
        24,
        int(
            cfg.MIN_PIECE_AREA_MM2
            * pixels_per_mm_x
            * pixels_per_mm_y
            * 0.45
        ),
    )
    blob_regions = []
    blob_started = PERF_STATS.mark()
    for region_start, region_end in detection_regions:
        roi = (
            margin_x,
            region_start,
            max(1, work_width - 2 * margin_x),
            max(1, region_end - region_start),
        )
        blobs = gray_image.find_blobs(
            [(piece_threshold, 255)],
            roi=roi,
            x_stride=1,
            y_stride=1,
            pixels_threshold=min_pixels,
            area_threshold=min_pixels,
            merge=False,
        )
        for blob in blobs:
            blob_regions.append(
                (blob, region_start, region_end)
            )
    PERF_STATS.add_stage("blob_ms", blob_started)
    gray_sanity = None
    if collect_sanity:
        upper_roi = (
            margin_x,
            margin_y,
            max(1, work_width - 2 * margin_x),
            max(1, upper_end - margin_y),
        )
        lower_roi = (
            margin_x,
            lower_start,
            max(1, work_width - 2 * margin_x),
            max(1, work_height - margin_y - lower_start),
        )
        gray_sanity = {
            "native": _native_gray_sanity(gray_image),
            "upper": _array_gray_region_sanity(
                gray_array, upper_roi, piece_threshold
            ),
            "lower": _array_gray_region_sanity(
                gray_array, lower_roi, piece_threshold
            ),
            "rotation_return": rotation_return,
            "find_min_pixels": min_pixels,
            "blob_rects": [
                tuple(_blob_value(blob, "rect", None))
                for blob, _region_start, _region_end in blob_regions
            ],
        }
    observation_details = []
    rejected = {
        "area": 0,
        "border": 0,
        "polygon": 0,
    }
    boundary_steps = 0
    pixel_reads = 0
    fallback_count = 0
    primary_boundary_ok_count = 0
    ordered_fallback_ok_count = 0
    polygon_reverse_retry_count = 0
    polygon_unrefined_retry_count = 0
    component_bound_count = 0
    component_candidate_count = 0
    contour_component_candidate_count = 0
    polygon_failure_rects = []
    boundary_failure_reasons = {}
    trace_failures = {}
    for blob, region_start, region_end in blob_regions:
        rect = tuple(_blob_value(blob, "rect", None))
        blob_center = (
            float(_blob_value(blob, "cx", 5)),
            float(_blob_value(blob, "cy", 6)),
        )
        blob_pixels = float(_blob_value(blob, "pixels", 4))
        area_mm2 = blob_pixels / (
            pixels_per_mm_x * pixels_per_mm_y
        )
        if (
            area_mm2 < cfg.MIN_PIECE_AREA_MM2
            or area_mm2 > cfg.MAX_PIECE_AREA_MM2
        ):
            rejected["area"] += 1
            continue
        if (
            rect[0] <= margin_x
            or rect[1] <= region_start
            or rect[0] + rect[2] >= work_width - margin_x
            or rect[1] + rect[3] >= region_end
        ):
            rejected["border"] += 1
            continue
        adaptive_background = (
            background_stats["background_gray"]
            if background_stats["sample_count"]
            >= cfg.PIECE_BACKGROUND_MIN_SAMPLES
            else None
        )
        (
            piece_contour_threshold,
            center_gray,
            contour_threshold_mode,
        ) = _piece_center_contour_threshold(
            gray_array,
            blob_center,
            adaptive_background,
            piece_threshold,
            contour_threshold,
        )
        polygon_mm, boundary_px, trace_diagnostics = (
            _extract_canmv_polygon(
                gray_array,
                blob,
                piece_threshold,
                piece_contour_threshold,
                pixels_per_mm_x,
                pixels_per_mm_y,
            )
        )
        trace_diagnostics["center_gray"] = center_gray
        trace_diagnostics["contour_threshold_used"] = (
            piece_contour_threshold
        )
        trace_diagnostics["contour_threshold_mode"] = (
            contour_threshold_mode
        )
        boundary_steps += trace_diagnostics.get(
            "boundary_steps", 0
        )
        pixel_reads += trace_diagnostics.get("pixel_reads", 0)
        component_candidate_count += trace_diagnostics.get(
            "component_candidates", 0
        )
        contour_component_candidate_count += (
            trace_diagnostics.get(
                "contour_component_candidates", 0
            )
        )
        if trace_diagnostics.get("reason") == "identity_ok":
            component_bound_count += 1
        if trace_diagnostics.get("boundary_primary_ok", False):
            primary_boundary_ok_count += 1
        if trace_diagnostics.get("fallback", False):
            fallback_count += 1
        if trace_diagnostics.get(
            "boundary_fallback_ordered_ok", False
        ):
            ordered_fallback_ok_count += 1
        if trace_diagnostics.get(
            "polygon_fit_reverse_used", False
        ):
            polygon_reverse_retry_count += 1
        if trace_diagnostics.get(
            "polygon_fit_unrefined_used", False
        ):
            polygon_unrefined_retry_count += 1
        failure_reason = trace_diagnostics.get(
            "boundary_failure_reason", ""
        )
        if failure_reason:
            boundary_failure_reasons[failure_reason] = (
                boundary_failure_reasons.get(failure_reason, 0) + 1
            )
        if not trace_diagnostics.get("ok", False):
            reason = trace_diagnostics.get("reason", "unknown")
            trace_failures[reason] = (
                trace_failures.get(reason, 0) + 1
            )
        if polygon_mm is None:
            rejected["polygon"] += 1
            polygon_failure_rects.append(rect)
            if trace_diagnostics.get("ok", False):
                trace_failures["fit_invalid"] = (
                    trace_failures.get("fit_invalid", 0) + 1
                )
            continue
        polygon_area_mm2 = polygon_area(polygon_mm)
        hull_pixels = max(
            1.0,
            polygon_area_mm2
            * pixels_per_mm_x
            * pixels_per_mm_y,
        )
        convexity = max(
            0.0, min(1.0, blob_pixels / hull_pixels)
        )
        confidence = max(
            0.0,
            min(1.0, 0.60 * convexity + 0.40),
        )
        try:
            observation = PieceObservation(
                "",
                boundary_px,
                polygon_mm,
                centroid_mm=polygon_centroid(polygon_mm),
                area_mm2=polygon_area_mm2,
                confidence=confidence,
            )
        except ValueError:
            rejected["polygon"] += 1
            trace_failures["piece_invalid"] = (
                trace_failures.get("piece_invalid", 0) + 1
            )
            continue
        observation_details.append(
            (
                observation,
                center_gray,
                piece_contour_threshold,
                contour_threshold_mode,
            )
        )

    observation_details.sort(
        key=lambda item: item[0].area_mm2, reverse=True
    )
    if len(observation_details) > cfg.MAX_PIECE_COUNT:
        observation_details = observation_details[
            : cfg.MAX_PIECE_COUNT
        ]
    observations = [item[0] for item in observation_details]
    center_grays = [item[1] for item in observation_details]
    contour_thresholds = [item[2] for item in observation_details]
    contour_threshold_modes = [
        item[3] for item in observation_details
    ]
    detected_vertex_counts = [
        len(piece.polygon_mm) for piece in observations
    ]
    diagnostics = {
        "rectified": gray_image,
        "mask": None,
        "divider_y_mm": active_divider_y_mm,
        "divider_detected": bool(
            divider_probe.get("detected", False)
            or divider_y_mm is not None
        ),
        "divider_strip_detected": bool(
            divider_probe.get("detected", False)
        ),
        "divider_source": divider_source,
        "divider_reason": divider_probe.get("reason", ""),
        "divider_coverage": divider_probe.get("coverage", 0.0),
        "divider_residual_px": divider_probe.get(
            "residual_px", 0.0
        ),
        "divider_slope_mm": divider_probe.get("slope_mm", 0.0),
        "divider_thickness_px": divider_probe.get(
            "thickness_px", 0.0
        ),
        "divider_threshold": divider_probe.get("threshold", 0),
        "divider_background_gray": divider_probe.get(
            "background_gray", 0.0
        ),
        "divider_hits": divider_probe.get("hit_count", 0),
        "divider_samples": divider_probe.get("sample_count", 0),
        "divider_mask_half_width_px": divider_mask.get(
            "half_width_px", 0
        ),
        "divider_masked_pixels": divider_mask.get(
            "masked_pixels", 0
        ),
        "divider_search_rows": divider_probe.get(
            "search_rows", (0, 0)
        ),
        "threshold": float(piece_threshold),
        "contour_threshold": float(contour_threshold),
        "center_grays": center_grays,
        "contour_thresholds": contour_thresholds,
        "contour_threshold_modes": contour_threshold_modes,
        "threshold_mode": threshold_mode,
        "segmentation_mode": segmentation_mode,
        "background_gray": background_stats[
            "background_gray"
        ],
        "background_high_gray": background_stats[
            "background_high_gray"
        ],
        "background_spread_gray": background_stats[
            "background_spread_gray"
        ],
        "background_sample_count": background_stats[
            "sample_count"
        ],
        "threshold_delta_gray": (
            float(piece_threshold)
            - background_stats["background_gray"]
            if background_stats["sample_count"] > 0
            else 0.0
        ),
        "raw_contours": len(blob_regions),
        "rejected": rejected,
        "rectified_border_black_px": safety_border_px,
        "detection_end_row": upper_end,
        "detection_regions": detection_regions,
        "region": region,
        "backend": "canmv_image",
        "work_corners_px": work_corners,
        "boundary_steps": boundary_steps,
        "pixel_reads": pixel_reads,
        "boundary_fallback_count": fallback_count,
        "boundary_primary_ok": primary_boundary_ok_count,
        "boundary_fallback_used": fallback_count,
        "boundary_fallback_ordered_ok": (
            ordered_fallback_ok_count
        ),
        "component_bound_count": component_bound_count,
        "component_candidate_count": component_candidate_count,
        "contour_component_candidate_count": (
            contour_component_candidate_count
        ),
        "polygon_reverse_retry_count": (
            polygon_reverse_retry_count
        ),
        "polygon_unrefined_retry_count": (
            polygon_unrefined_retry_count
        ),
        "polygon_failure_rects": polygon_failure_rects,
        "boundary_failure_reason": boundary_failure_reasons,
        "trace_failures": trace_failures,
        "detected_vertex_counts": detected_vertex_counts,
        "gray_sanity": gray_sanity,
    }
    return observations, diagnostics
