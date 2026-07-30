"""Vision adapters for desktop OpenCV and CanMV's native ``image`` module.

The desktop path keeps OpenCV for repeatable photo regression.  The K230 path
uses only firmware-standard ``image.Image`` operations plus pure Python contour
geometry, because some CanMV v1.6 images do not include the optional ``cv2``
module.
"""

import math
import os

import puzzle_config as cfg
from puzzle_perf import PERF_STATS
from puzzle_geometry import (
    PieceObservation,
    edge_lengths,
    ensure_clockwise,
    point_in_polygon,
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


def mm_to_rectified_px(point_mm):
    return (
        float(point_mm[0]) * cfg.RECTIFIED_PX_PER_MM,
        float(point_mm[1]) * cfg.RECTIFIED_PX_PER_MM,
    )


def rectified_px_to_mm(point_px):
    return (
        float(point_px[0]) / cfg.RECTIFIED_PX_PER_MM,
        float(point_px[1]) / cfg.RECTIFIED_PX_PER_MM,
    )


def make_perspective_transform(corners_px, cv2_module, np_module):
    """Build the camera-to-standard-A4 homography."""
    src = np_module.array(corners_px, dtype=np_module.float32)
    dst = np_module.array(
        [
            (0.0, 0.0),
            (cfg.RECTIFIED_WIDTH_PX - 1.0, 0.0),
            (cfg.RECTIFIED_WIDTH_PX - 1.0, cfg.RECTIFIED_HEIGHT_PX - 1.0),
            (0.0, cfg.RECTIFIED_HEIGHT_PX - 1.0),
        ],
        dtype=np_module.float32,
    )
    return cv2_module.getPerspectiveTransform(src, dst)


def rectify_gray(
    gray_array,
    corners_px,
    cv2_module,
    np_module,
    perspective_matrix=None,
):
    matrix = perspective_matrix
    if matrix is None:
        matrix = make_perspective_transform(
            corners_px, cv2_module, np_module
        )
    return cv2_module.warpPerspective(
        gray_array,
        matrix,
        (cfg.RECTIFIED_WIDTH_PX, cfg.RECTIFIED_HEIGHT_PX),
        flags=cv2_module.INTER_LINEAR,
        borderMode=cv2_module.BORDER_CONSTANT,
        borderValue=0,
    )


def _find_divider_row(binary_mask, nominal_row):
    half_range = int(
        cfg.DIVIDER_SEARCH_HALF_RANGE_MM * cfg.RECTIFIED_PX_PER_MM + 0.5
    )
    height = binary_mask.shape[0]
    width = binary_mask.shape[1]
    start = max(0, nominal_row - half_range)
    end = min(height - 1, nominal_row + half_range)
    best_row = nominal_row
    best_count = 0
    # Avoid np.sum axis compatibility differences between NumPy and ulab.
    for y in range(start, end + 1):
        count = 0
        row = binary_mask[y]
        for x in range(width):
            if int(row[x]) != 0:
                count += 1
        if count > best_count:
            best_count = count
            best_row = y
    # A divider should span most of A4. Otherwise keep the configured location.
    if best_count < int(width * 0.55):
        return nominal_row, False
    return best_row, True


def _contour_points(contour):
    """Convert OpenCV/ulab contour layouts (N,1,2) or (N,2) to tuples."""
    result = []
    for item in contour:
        try:
            if len(item) == 1:
                item = item[0]
        except TypeError:
            pass
        result.append((float(item[0]), float(item[1])))
    return result


def _points_array(points, np_module):
    return np_module.array(points, dtype=np_module.float32)


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


def _finalize_fitted_polygon(contour_mm, polygon):
    """Refine and validate an already simplified ordered polygon."""
    polygon = _refine_vertices_with_lines(contour_mm, polygon)
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


def _ordered_contour_polygon(points_mm):
    """Fit a validated 3..5 vertex polygon from an ordered outer contour."""
    tolerance = max(0.05, cfg.CONTOUR_DP_TOLERANCE_MM)
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
    return _finalize_fitted_polygon(points_mm, polygon)


def _extract_polygon(contour, cv2_module, np_module):
    del np_module

    def approximate(source):
        perimeter = float(cv2_module.arcLength(source, True))
        epsilon = (
            cfg.CONTOUR_DP_TOLERANCE_MM
            * cfg.RECTIFIED_PX_PER_MM
        )
        approx = None
        cleanup_limit = (
            cfg.MAX_POLYGON_VERTICES
            + cfg.VERTEX_CLEANUP_EXTRA_VERTICES
        )
        for _ in range(10):
            approx = cv2_module.approxPolyDP(
                source, epsilon, True
            )
            if len(approx) <= cleanup_limit:
                break
            epsilon *= 1.35
        source_mm = [
            rectified_px_to_mm(point)
            for point in _contour_points(source)
        ]
        polygon_mm = [
            rectified_px_to_mm(point)
            for point in _contour_points(approx)
        ]
        return (
            _finalize_fitted_polygon(
                source_mm, polygon_mm
            ),
            bool(cv2_module.isContourConvex(approx)),
        )

    polygon, simplified_is_convex = approximate(contour)
    # Raster shadows can make a physically convex hand-cut piece locally
    # jagged.  A hull is used only after the ordered fit itself proves convex;
    # a meaningful concavity therefore remains the primary model.
    if (
        cfg.FORCE_CONVEX_CONTOURS
        or (
            polygon is not None
            and simplified_is_convex
        )
    ):
        stable, _ = approximate(
            cv2_module.convexHull(contour)
        )
        if stable is not None:
            polygon = stable
    return polygon


def detect_pieces_from_gray(
    gray_array,
    corners_px,
    cv2_module,
    np_module,
    perspective_matrix=None,
):
    """Rectify, segment, and return ``(observations, diagnostics)``."""
    rectified = rectify_gray(
        gray_array,
        corners_px,
        cv2_module,
        np_module,
        perspective_matrix=perspective_matrix,
    )

    threshold_mode = cfg.THRESHOLD_MODE
    threshold_flags = cv2_module.THRESH_BINARY
    threshold_value = cfg.WHITE_GRAY_THRESHOLD
    if threshold_mode == "otsu":
        threshold_flags |= cv2_module.THRESH_OTSU
    try:
        used_threshold, mask = cv2_module.threshold(
            rectified, threshold_value, 255, threshold_flags
        )
    except Exception:
        # Fixed threshold is the explicit no-silent-failure fallback.
        used_threshold, mask = cv2_module.threshold(
            rectified,
            cfg.WHITE_GRAY_THRESHOLD,
            255,
            cv2_module.THRESH_BINARY,
        )
        threshold_mode = "fixed_fallback"

    nominal_divider = int(cfg.DIVIDER_Y_MM * cfg.RECTIFIED_PX_PER_MM + 0.5)
    divider_row, divider_detected = _find_divider_row(mask, nominal_divider)
    detection_end = max(
        1, divider_row - int(2.0 * cfg.RECTIFIED_PX_PER_MM + 0.5)
    )
    # Keep divider/lower region out of contour extraction. cv2 drawing avoids a
    # slow Python pixel loop on the K230.
    cv2_module.rectangle(
        mask,
        (0, detection_end),
        (mask.shape[1] - 1, mask.shape[0] - 1),
        0,
        cv2_module.FILLED,
    )

    kernel = cv2_module.getStructuringElement(
        cv2_module.MORPH_RECT,
        (cfg.MORPH_KERNEL_PX, cfg.MORPH_KERNEL_PX),
    )
    if cfg.MORPH_OPEN_ITERATIONS > 0:
        mask = cv2_module.morphologyEx(
            mask,
            cv2_module.MORPH_OPEN,
            kernel,
            iterations=cfg.MORPH_OPEN_ITERATIONS,
        )
    if cfg.MORPH_CLOSE_ITERATIONS > 0:
        mask = cv2_module.morphologyEx(
            mask,
            cv2_module.MORPH_CLOSE,
            kernel,
            iterations=cfg.MORPH_CLOSE_ITERATIONS,
        )

    contour_result = cv2_module.findContours(
        mask, cv2_module.RETR_EXTERNAL, cv2_module.CHAIN_APPROX_SIMPLE
    )
    contours = contour_result[-2]
    margin_px = cfg.DETECTION_BORDER_MARGIN_MM * cfg.RECTIFIED_PX_PER_MM
    observations = []
    rejected = {
        "area": 0,
        "border": 0,
        "polygon": 0,
    }
    for contour in contours:
        contour_area_px = float(cv2_module.contourArea(contour))
        area_mm2 = contour_area_px / (
            cfg.RECTIFIED_PX_PER_MM * cfg.RECTIFIED_PX_PER_MM
        )
        if (
            area_mm2 < cfg.MIN_PIECE_AREA_MM2
            or area_mm2 > cfg.MAX_PIECE_AREA_MM2
        ):
            rejected["area"] += 1
            continue
        x, y, width, height = cv2_module.boundingRect(contour)
        if (
            x <= margin_px
            or y <= margin_px
            or x + width >= cfg.RECTIFIED_WIDTH_PX - margin_px
            or y + height >= detection_end - margin_px
        ):
            rejected["border"] += 1
            continue
        polygon_mm = _extract_polygon(contour, cv2_module, np_module)
        if polygon_mm is None:
            rejected["polygon"] += 1
            continue

        hull = cv2_module.convexHull(contour)
        hull_area_px = max(1.0, float(cv2_module.contourArea(hull)))
        convexity = min(1.0, contour_area_px / hull_area_px)
        vertex_confidence = 1.0 if 3 <= len(polygon_mm) <= 5 else 0.0
        confidence = max(
            0.0,
            min(1.0, 0.55 * convexity + 0.35 * vertex_confidence + 0.10),
        )
        contour_px = _contour_points(contour)
        observations.append(
            PieceObservation(
                "",
                contour_px,
                polygon_mm,
                centroid_mm=polygon_centroid(polygon_mm),
                area_mm2=polygon_area(polygon_mm),
                confidence=confidence,
            )
        )

    observations.sort(key=lambda piece: piece.area_mm2, reverse=True)
    if len(observations) > cfg.MAX_PIECE_COUNT:
        observations = observations[: cfg.MAX_PIECE_COUNT]
    diagnostics = {
        "rectified": rectified,
        "mask": mask,
        "divider_y_mm": divider_row / cfg.RECTIFIED_PX_PER_MM,
        "divider_detected": divider_detected,
        "threshold": float(used_threshold),
        "threshold_mode": threshold_mode,
        "raw_contours": len(contours),
        "rejected": rejected,
        "detection_end_row": detection_end,
    }
    return observations, diagnostics


def _blob_value(blob, method_name, tuple_index=None):
    """Read a CanMV blob method, with tuple-index compatibility."""
    method = getattr(blob, method_name, None)
    if method is not None:
        return method()
    if tuple_index is None:
        raise AttributeError("blob has no {}".format(method_name))
    return blob[tuple_index]


def _pixel_is_white(gray_array, x, y, threshold):
    return int(gray_array[y][x]) >= threshold


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


def trace_ordered_boundary(
    gray_array,
    rect,
    threshold,
    max_steps=None,
):
    """Trace one ordered outer boundary with the Moore-neighbour algorithm.

    Pixels outside ``rect`` or the array are treated as background.  The return
    value is ``(points, diagnostics)``; a failed trace returns an empty point
    list and an explicit reason so callers can fail closed or use the measured
    legacy fallback.
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


def _nearest_white_pixel(
    gray_array,
    x0,
    y0,
    width,
    height,
    center_x,
    center_y,
    threshold,
):
    center_x = max(x0, min(x0 + width - 1, int(center_x)))
    center_y = max(y0, min(y0 + height - 1, int(center_y)))
    if _pixel_is_white(gray_array, center_x, center_y, threshold):
        return center_x, center_y
    best = None
    best_distance = None
    for y in range(y0, y0 + height):
        _vision_exitpoint(y - y0, interval=16)
        row = gray_array[y]
        for x in range(x0, x0 + width):
            if int(row[x]) < threshold:
                continue
            distance = (
                (x - center_x) * (x - center_x)
                + (y - center_y) * (y - center_y)
            )
            if best_distance is None or distance < best_distance:
                best = (x, y)
                best_distance = distance
    return best


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


def _component_boundary(
    gray_array,
    rect,
    center,
    threshold,
):
    """Flood-isolate one component, then Moore-trace its ordered boundary."""
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
        "component_pixels": 0,
    }
    if width <= 0 or height <= 0:
        diagnostics["reason"] = "fallback_empty_bbox"
        return [], diagnostics

    seed = _nearest_white_pixel(
        gray_array,
        x0,
        y0,
        width,
        height,
        center[0],
        center[1],
        threshold,
    )
    if seed is None:
        diagnostics["reason"] = "fallback_no_seed"
        return [], diagnostics

    # One byte per bounding-box pixel is predictable and much smaller than a
    # Python set of coordinate tuples on MicroPython.
    visited = bytearray(width * height)
    seed_index = (seed[1] - y0) * width + (seed[0] - x0)
    visited[seed_index] = 1
    stack = [seed_index]
    neighbor_offsets = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    processed = 0
    while stack:
        processed += 1
        _vision_exitpoint(processed)
        local_index = stack.pop()
        local_y = local_index // width
        local_x = local_index - local_y * width
        x = x0 + local_x
        y = y0 + local_y
        if not _pixel_is_white(gray_array, x, y, threshold):
            continue
        diagnostics["component_pixels"] += 1

        for dx, dy in neighbor_offsets:
            nx = x + dx
            ny = y + dy
            if (
                nx < x0
                or nx >= x1
                or ny < y0
                or ny >= y1
                or not _pixel_is_white(
                    gray_array, nx, ny, threshold
                )
            ):
                continue
            neighbor_index = (ny - y0) * width + (nx - x0)
            if not visited[neighbor_index]:
                visited[neighbor_index] = 1
                stack.append(neighbor_index)
    local_mask = _ComponentMask(visited, width, height)
    boundary, trace = trace_ordered_boundary(
        local_mask,
        (0, 0, width, height),
        1,
    )
    diagnostics["ok"] = bool(trace["ok"])
    diagnostics["reason"] = (
        "fallback_ok"
        if trace["ok"]
        else "fallback_{}".format(trace["reason"])
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


def _polygon_perimeter(points):
    total = 0.0
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        dx = other[0] - point[0]
        dy = other[1] - point[1]
        total += math.sqrt(dx * dx + dy * dy)
    return total


def _simplify_convex_polygon(points, epsilon):
    """Remove hull stair-steps while preserving significant short corners."""
    result = list(points)
    while len(result) > cfg.MIN_POLYGON_VERTICES:
        best_index = None
        best_distance = None
        for index, point in enumerate(result):
            distance = _point_line_distance(
                point,
                result[(index - 1) % len(result)],
                result[(index + 1) % len(result)],
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        if best_distance is None or best_distance > epsilon:
            break
        del result[best_index]
    if len(result) > cfg.MAX_POLYGON_VERTICES:
        result = _reduce_polygon_to_limit(
            result, cfg.MAX_POLYGON_VERTICES
        )
    return result


def _extract_canmv_polygon(
    gray_array,
    blob,
    threshold,
    pixels_per_mm_x,
    pixels_per_mm_y,
):
    """Extract one CanMV connected component as a millimetre polygon."""
    rect = tuple(_blob_value(blob, "rect", None))
    center = (
        float(_blob_value(blob, "cx", 5)),
        float(_blob_value(blob, "cy", 6)),
    )
    contour_started = PERF_STATS.mark()
    boundary_px, trace_diagnostics = trace_ordered_boundary(
        gray_array, rect, threshold
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
    if (
        not trace_diagnostics["ok"]
        or len(boundary_px) < cfg.BOUNDARY_TRACE_MIN_POINTS
    ):
        if not cfg.ENABLE_BOUNDARY_FLOOD_FALLBACK:
            PERF_STATS.add_stage(
                "contour_ms", contour_started
            )
            return None, boundary_px, trace_diagnostics
        boundary_px, fallback_diagnostics = _component_boundary(
            gray_array, rect, center, threshold
        )
        trace_diagnostics["fallback"] = True
        trace_diagnostics["boundary_fallback_used"] = True
        trace_diagnostics["boundary_fallback_ordered_ok"] = bool(
            fallback_diagnostics["ok"]
        )
        trace_diagnostics["ok"] = bool(
            fallback_diagnostics["ok"]
        )
        trace_diagnostics["reason"] = fallback_diagnostics["reason"]
        trace_diagnostics["boundary_failure_reason"] = (
            ""
            if fallback_diagnostics["ok"]
            else fallback_diagnostics["reason"]
        )
        trace_diagnostics["pixel_reads"] += fallback_diagnostics[
            "pixel_reads"
        ]
        trace_diagnostics["boundary_steps"] += fallback_diagnostics[
            "boundary_steps"
        ]
        PERF_STATS.increment("boundary_fallback_count")
    else:
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
    polygon_mm = _ordered_contour_polygon(boundary_mm)
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


def detect_pieces_from_canmv_image(
    gray_image,
    corners_px,
    source_frame_size,
    region="upper",
    threshold=None,
    divider_y_mm=None,
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
    gray_image.rotation_corr(corners=work_corners)
    PERF_STATS.add_stage("rectify_ms", rectify_started)

    pixels_per_mm_x = float(work_width - 1) / cfg.A4_WIDTH_MM
    pixels_per_mm_y = float(work_height - 1) / cfg.A4_HEIGHT_MM
    active_divider_y_mm = (
        cfg.DIVIDER_Y_MM
        if divider_y_mm is None
        else max(
            0.0,
            min(cfg.A4_HEIGHT_MM, float(divider_y_mm)),
        )
    )
    nominal_divider = int(
        active_divider_y_mm * pixels_per_mm_y + 0.5
    )
    upper_end = max(
        2,
        nominal_divider - int(2.0 * pixels_per_mm_y + 0.5),
    )
    lower_start = min(
        work_height - 2,
        nominal_divider
        + int(2.0 * pixels_per_mm_y + 0.5),
    )
    margin_x = max(
        1,
        int(cfg.DETECTION_BORDER_MARGIN_MM * pixels_per_mm_x + 0.5),
    )
    margin_y = max(
        1,
        int(cfg.DETECTION_BORDER_MARGIN_MM * pixels_per_mm_y + 0.5),
    )
    gray_array = gray_image.to_numpy_ref()
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
    observations = []
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
    boundary_failure_reasons = {}
    trace_failures = {}
    for blob, region_start, region_end in blob_regions:
        rect = tuple(_blob_value(blob, "rect", None))
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
        polygon_mm, boundary_px, trace_diagnostics = (
            _extract_canmv_polygon(
                gray_array,
                blob,
                piece_threshold,
                pixels_per_mm_x,
                pixels_per_mm_y,
            )
        )
        boundary_steps += trace_diagnostics.get(
            "boundary_steps", 0
        )
        pixel_reads += trace_diagnostics.get("pixel_reads", 0)
        if trace_diagnostics.get("boundary_primary_ok", False):
            primary_boundary_ok_count += 1
        if trace_diagnostics.get("fallback", False):
            fallback_count += 1
        if trace_diagnostics.get(
            "boundary_fallback_ordered_ok", False
        ):
            ordered_fallback_ok_count += 1
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
        observations.append(observation)

    observations.sort(key=lambda piece: piece.area_mm2, reverse=True)
    if len(observations) > cfg.MAX_PIECE_COUNT:
        observations = observations[: cfg.MAX_PIECE_COUNT]
    detected_vertex_counts = [
        len(piece.polygon_mm) for piece in observations
    ]
    diagnostics = {
        "rectified": gray_image,
        "mask": None,
        "divider_y_mm": active_divider_y_mm,
        "divider_detected": divider_y_mm is not None,
        "threshold": float(piece_threshold),
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
        "boundary_failure_reason": boundary_failure_reasons,
        "trace_failures": trace_failures,
        "detected_vertex_counts": detected_vertex_counts,
    }
    return observations, diagnostics


def build_polygon_scanlines(
    polygon_mm,
    width,
    height,
    sample_stride=2,
):
    """Precompute sampled x intervals for a millimetre-space polygon."""
    width = int(width)
    height = int(height)
    stride = max(1, int(sample_stride))
    if width <= 1 or height <= 1 or len(polygon_mm) < 3:
        return {
            "width": width,
            "height": height,
            "sample_stride": stride,
            "lines": {},
            "sample_count": 0,
        }
    pixels_per_mm_x = float(width - 1) / cfg.A4_WIDTH_MM
    pixels_per_mm_y = float(height - 1) / cfg.A4_HEIGHT_MM
    polygon_px = [
        (
            point[0] * pixels_per_mm_x,
            point[1] * pixels_per_mm_y,
        )
        for point in polygon_mm
    ]
    min_y = max(
        0, int(min(point[1] for point in polygon_px))
    )
    max_y = min(
        height - 1,
        int(max(point[1] for point in polygon_px) + 1.0),
    )
    lines = {}
    sample_count = 0
    for y in range(min_y, max_y + 1, stride):
        intersections = []
        scan_y = float(y)
        for index, a in enumerate(polygon_px):
            b = polygon_px[(index + 1) % len(polygon_px)]
            if abs(b[1] - a[1]) <= 1e-9:
                continue
            if not (
                (a[1] <= scan_y < b[1])
                or (b[1] <= scan_y < a[1])
            ):
                continue
            ratio = (scan_y - a[1]) / (b[1] - a[1])
            intersections.append(
                a[0] + ratio * (b[0] - a[0])
            )
        intersections.sort()
        intervals = []
        for index in range(0, len(intersections) - 1, 2):
            start_x = max(
                0, int(math.ceil(intersections[index]))
            )
            end_x = min(
                width - 1,
                int(math.floor(intersections[index + 1])),
            )
            if start_x > end_x:
                continue
            intervals.append((start_x, end_x))
            sample_count += (
                (end_x - start_x) // stride + 1
            )
        if intervals:
            lines[y] = intervals
    return {
        "width": width,
        "height": height,
        "sample_stride": stride,
        "lines": lines,
        "sample_count": sample_count,
    }


def polygon_white_coverage_scanlines(
    gray_array,
    target_scanlines,
    threshold=None,
):
    """Evaluate one cached target without point-in-polygon calls."""
    if threshold is None:
        threshold = cfg.WHITE_GRAY_THRESHOLD
    stride = target_scanlines["sample_stride"]
    sample_count = 0
    foreground_count = 0
    loop_counter = 0
    for y, intervals in target_scanlines["lines"].items():
        row = gray_array[y]
        for start_x, end_x in intervals:
            for x in range(start_x, end_x + 1, stride):
                loop_counter += 1
                _vision_exitpoint(loop_counter)
                sample_count += 1
                if int(row[x]) >= threshold:
                    foreground_count += 1
    coverage = (
        float(foreground_count) / float(sample_count)
        if sample_count > 0
        else 0.0
    )
    return {
        "coverage_ratio": coverage,
        "sample_count": sample_count,
        "foreground_count": foreground_count,
    }


def polygon_white_coverage(
    gray_array,
    polygon_mm,
    threshold=None,
    sample_stride=2,
):
    """Compatibility wrapper around cached scanline coverage."""
    height = int(gray_array.shape[0])
    width = int(gray_array.shape[1])
    target_scanlines = build_polygon_scanlines(
        polygon_mm,
        width,
        height,
        sample_stride=sample_stride,
    )
    return polygon_white_coverage_scanlines(
        gray_array,
        target_scanlines,
        threshold=threshold,
    )["coverage_ratio"]
