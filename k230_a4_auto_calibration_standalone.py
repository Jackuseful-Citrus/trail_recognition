#!/usr/bin/env python3
"""Self-contained CanMV K230 automatic A4 calibration test."""
# No project-local imports are required; this file can be edited and
# run directly in CanMV IDE.

FRAME_WIDTH = 800
FRAME_HEIGHT = 480
A4_DETECT_WIDTH = 320
A4_DETECT_HEIGHT = 192
A4_RECT_EDGE_THRESHOLD = 7000
A4_MIN_FRAME_AREA_RATIO = 0.12
A4_MAX_FRAME_AREA_RATIO = 0.94
A4_MIN_WIDTH_HEIGHT_RATIO = 0.40
A4_MAX_WIDTH_HEIGHT_RATIO = 2.40
A4_EXPECTED_WIDTH_HEIGHT_RATIO = 210.0 / 297.0
A4_MIN_SIDE_PX = 38.0
A4_MAX_CENTER_OFFSET_RATIO = 0.48
A4_MIN_IMAGE_EDGE_MARGIN_PX = 3
A4_TOP_SIDE = "auto"
A4_DARK_THRESHOLD = 135
A4_MAX_INSIDE_GRAY = 178.0
A4_DARK_BLOB_MIN_AREA_RATIO = 0.03
A4_EDGE_PROBE_OFFSET_PX = 8.0
A4_EDGE_PROBE_SAMPLES = 9
A4_INTERNAL_EDGE_SIMILAR_GRAY_DELTA = 24.0
A4_INTERNAL_EDGE_DARK_RATIO_MAX = 0.67
A4_INTERNAL_EDGE_MIN_SAMPLES = 5
A4_LOCK_REQUIRED_FRAMES = 3
A4_HOLD_MISSED_FRAMES = 15
A4_SLOW_SMOOTH_ALPHA = 0.38
A4_STATUS_PRINT_EVERY_N_FRAMES = 15
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DIVIDER_Y_MM = 148.5
DIVIDER_SEARCH_HALF_RANGE_MM = 8.0
ENABLE_DYNAMIC_DIVIDER = True
A4_REQUIRE_DIVIDER_FOR_LOCK = True
DIVIDER_LINE_SAMPLE_COUNT = 25
DIVIDER_LINE_MIN_GRAY = 70
DIVIDER_LINE_MIN_CONTRAST_GRAY = 35
DIVIDER_LINE_MIN_COVERAGE = 0.70
DIVIDER_LINE_MAX_RESIDUAL_PX = 2.5
DIVIDER_LINE_MAX_SLOPE_MM = 3.0
DIVIDER_TRACK_ALPHA = 0.35
WHITE_GRAY_THRESHOLD = 180
A4_LOCK_DEADBAND_PX = 3.0
A4_RELOCK_MOTION_PX = 10.0
A4_RELOCK_CONFIRM_FRAMES = 3
LOOP_IDLE_MS = 5
AUTO_STOP_SECONDS = 0
MAX_FRAME_COUNT = 0

"""Native CanMV automatic A4 boundary detection and corner tracking."""

import math



def _distance(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _polygon_area(points):
    total = 0.0
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        total += (
            float(point[0]) * float(other[1])
            - float(other[0]) * float(point[1])
        )
    return abs(total) * 0.5


def order_a4_corners(points):
    """Return distinct quadrilateral corners in TL, TR, BR, BL order."""
    if points is None or len(points) != 4:
        return None
    values = [
        (float(point[0]), float(point[1])) for point in points
    ]
    sums = [point[0] + point[1] for point in values]
    differences = [point[0] - point[1] for point in values]
    ordered = [
        values[sums.index(min(sums))],
        values[differences.index(max(differences))],
        values[sums.index(max(sums))],
        values[differences.index(min(differences))],
    ]
    unique = set(
        (int(round(point[0])), int(round(point[1])))
        for point in ordered
    )
    if len(unique) != 4 or _polygon_area(ordered) <= 1.0:
        return None
    return ordered


def _quad_point(corners, u, v):
    tl, tr, br, bl = corners
    return (
        (1.0 - u) * (1.0 - v) * tl[0]
        + u * (1.0 - v) * tr[0]
        + u * v * br[0]
        + (1.0 - u) * v * bl[0],
        (1.0 - u) * (1.0 - v) * tl[1]
        + u * (1.0 - v) * tr[1]
        + u * v * br[1]
        + (1.0 - u) * v * bl[1],
    )


def _unit_square_transform(corners):
    """Return a projective unit-square-to-quadrilateral transform."""
    x0, y0 = corners[0]
    x1, y1 = corners[1]
    x2, y2 = corners[2]
    x3, y3 = corners[3]
    dx1 = x1 - x2
    dx2 = x3 - x2
    dy1 = y1 - y2
    dy2 = y3 - y2
    projective_x = x0 - x1 + x2 - x3
    projective_y = y0 - y1 + y2 - y3
    if (
        abs(projective_x) <= 1e-9
        and abs(projective_y) <= 1e-9
    ):
        g = 0.0
        h = 0.0
    else:
        denominator = dx1 * dy2 - dx2 * dy1
        if abs(denominator) <= 1e-9:
            return None
        g = (
            projective_x * dy2
            - dx2 * projective_y
        ) / denominator
        h = (
            dx1 * projective_y
            - projective_x * dy1
        ) / denominator
    return (
        x1 - x0 + g * x1,
        x3 - x0 + h * x3,
        x0,
        y1 - y0 + g * y1,
        y3 - y0 + h * y3,
        y0,
        g,
        h,
    )


def _projective_point(transform, u, v):
    denominator = (
        transform[6] * u + transform[7] * v + 1.0
    )
    if abs(denominator) <= 1e-9:
        return None
    return (
        (
            transform[0] * u
            + transform[1] * v
            + transform[2]
        )
        / denominator,
        (
            transform[3] * u
            + transform[4] * v
            + transform[5]
        )
        / denominator,
    )


def _projective_quad_point(corners, u, v):
    transform = _unit_square_transform(corners)
    if transform is None:
        return None
    return _projective_point(transform, u, v)


def _inside_gray(gray_image, corners):
    values = []
    for v in (0.18, 0.36, 0.64, 0.82):
        for u in (0.22, 0.50, 0.78):
            point = _quad_point(corners, u, v)
            x = max(
                0,
                min(gray_image.width() - 1, int(round(point[0]))),
            )
            y = max(
                0,
                min(gray_image.height() - 1, int(round(point[1]))),
            )
            value = gray_image.get_pixel(x, y)
            if isinstance(value, tuple):
                value = value[0]
            values.append(float(value))
    values.sort()
    # Median resists white fragments and the horizontal divider.
    middle = len(values) // 2
    return 0.5 * (values[middle - 1] + values[middle])


def _gray_at(gray_image, x, y):
    value = gray_image.get_pixel(int(round(x)), int(round(y)))
    if isinstance(value, tuple):
        value = value[0]
    return float(value)


def _median(values):
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


def _divider_line_probe(gray_image, corners, inside_gray):
    """Estimate divider v(u) inside one physically ordered A4 candidate."""
    nominal_fraction = (
        DIVIDER_Y_MM / A4_HEIGHT_MM
    )
    result = {
        "detected": False,
        "line_found": False,
        "reject_reason": "",
        "center_fraction": nominal_fraction,
        "left_fraction": nominal_fraction,
        "right_fraction": nominal_fraction,
        "divider_y_mm": DIVIDER_Y_MM,
        "slope_mm": 0.0,
        "coverage": 0.0,
        "residual_px": 0.0,
        "confidence": 0.0,
    }
    if not ENABLE_DYNAMIC_DIVIDER:
        return result
    transform = _unit_square_transform(corners)
    if transform is None:
        result["reject_reason"] = "projective_transform"
        return result
    side_pixels = 0.5 * (
        _distance(corners[0], corners[3])
        + _distance(corners[1], corners[2])
    )
    if side_pixels <= 4.0:
        return result
    pixel_fraction = 1.0 / side_pixels
    half_range_fraction = (
        DIVIDER_SEARCH_HALF_RANGE_MM
        / A4_HEIGHT_MM
    )
    scan_steps = max(
        2,
        int(
            half_range_fraction
            / pixel_fraction
            + 0.5
        ),
    )
    sample_count = max(
        7, int(DIVIDER_LINE_SAMPLE_COUNT)
    )
    minimum_bright_gray = max(
        float(DIVIDER_LINE_MIN_GRAY),
        float(inside_gray)
        + float(DIVIDER_LINE_MIN_CONTRAST_GRAY),
    )
    samples = []
    for index in range(sample_count):
        u = 0.06 + 0.88 * index / max(
            1, sample_count - 1
        )
        best_value = -1.0
        scan_values = []
        for offset in range(-scan_steps, scan_steps + 1):
            fraction = (
                nominal_fraction
                + offset * pixel_fraction
            )
            if fraction <= 0.0 or fraction >= 1.0:
                continue
            point = _projective_point(transform, u, fraction)
            if point is None:
                continue
            value = _gray_at(
                gray_image, point[0], point[1]
            )
            scan_values.append((fraction, value))
            best_value = max(best_value, value)
        if best_value >= minimum_bright_gray:
            # Use the centre of a multi-pixel painted/taped line rather than
            # whichever bright edge happened to be visited first.
            peak_fractions = [
                item[0]
                for item in scan_values
                if item[1] >= max(
                    minimum_bright_gray,
                    best_value - 5.0,
                )
            ]
            best_fraction = _median(peak_fractions)
            samples.append((u, best_fraction))

    minimum_hits = max(
        5,
        int(
            sample_count
            * DIVIDER_LINE_MIN_COVERAGE
            + 0.5
        ),
    )
    if len(samples) < minimum_hits:
        result["reject_reason"] = "coverage"
        return result
    left = [item for item in samples if item[0] <= 0.35]
    right = [item for item in samples if item[0] >= 0.65]
    if len(left) < 2 or len(right) < 2:
        result["reject_reason"] = "span"
        return result
    left_u = _median([item[0] for item in left])
    right_u = _median([item[0] for item in right])
    if right_u - left_u <= 0.2:
        result["reject_reason"] = "span"
        return result
    left_v = _median([item[1] for item in left])
    right_v = _median([item[1] for item in right])
    slope = (right_v - left_v) / (right_u - left_u)
    intercept = _median(
        [item[1] - slope * item[0] for item in samples]
    )
    inliers = [
        item
        for item in samples
        if abs(
            item[1] - (intercept + slope * item[0])
        )
        * side_pixels
        <= DIVIDER_LINE_MAX_RESIDUAL_PX
    ]
    if len(inliers) < minimum_hits:
        result["reject_reason"] = "residual"
        return result

    # One least-squares refinement after the robust median initialization.
    mean_u = sum(item[0] for item in inliers) / len(inliers)
    mean_v = sum(item[1] for item in inliers) / len(inliers)
    denominator = sum(
        (item[0] - mean_u) * (item[0] - mean_u)
        for item in inliers
    )
    if denominator > 1e-9:
        slope = sum(
            (item[0] - mean_u) * (item[1] - mean_v)
            for item in inliers
        ) / denominator
    intercept = mean_v - slope * mean_u
    residual_px = sum(
        abs(item[1] - (intercept + slope * item[0]))
        * side_pixels
        for item in inliers
    ) / len(inliers)
    center_fraction = intercept + 0.5 * slope
    if (
        abs(center_fraction - nominal_fraction)
        > half_range_fraction + 2.0 * pixel_fraction
    ):
        result["reject_reason"] = "position"
        return result
    left_fraction = intercept
    right_fraction = intercept + slope
    coverage = float(len(inliers)) / sample_count
    residual_quality = max(
        0.0,
        1.0
        - residual_px
        / max(0.1, DIVIDER_LINE_MAX_RESIDUAL_PX),
    )
    slope_mm = (
        (right_fraction - left_fraction)
        * A4_HEIGHT_MM
    )
    result.update(
        {
            "line_found": True,
            "center_fraction": center_fraction,
            "left_fraction": left_fraction,
            "right_fraction": right_fraction,
            "divider_y_mm": (
                center_fraction * A4_HEIGHT_MM
            ),
            "slope_mm": slope_mm,
            "coverage": coverage,
            "residual_px": residual_px,
            "confidence": coverage * residual_quality,
        }
    )
    if abs(slope_mm) > DIVIDER_LINE_MAX_SLOPE_MM:
        result["reject_reason"] = "slope"
        return result
    result["detected"] = True
    return result


def _internal_edge_probe(gray_image, corners):
    """Return evidence that a proposed edge lies inside the A4 surface.

    A half of a landscape A4 cut by the physical divider has the same aspect
    ratio as a complete A4 rotated by 90 degrees.  Geometry therefore cannot
    reject it.  Probe symmetric points on both sides of every candidate edge:
    if both sides repeatedly contain the same dark paper tone, that edge is an
    internal divider rather than the outside paper boundary.
    """
    center_x = sum(point[0] for point in corners) * 0.25
    center_y = sum(point[1] for point in corners) * 0.25
    sample_count = max(1, int(A4_EDGE_PROBE_SAMPLES))
    offset = float(A4_EDGE_PROBE_OFFSET_PX)
    ratios = []
    sample_counts = []
    for edge_index, start in enumerate(corners):
        end = corners[(edge_index + 1) % len(corners)]
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        length = math.sqrt(dx * dx + dy * dy)
        if length <= 1.0:
            ratios.append(0.0)
            sample_counts.append(0)
            continue

        normal_x = -dy / length
        normal_y = dx / length
        midpoint_x = 0.5 * (float(start[0]) + float(end[0]))
        midpoint_y = 0.5 * (float(start[1]) + float(end[1]))
        # Select the normal pointing away from the quadrilateral centre.
        if (
            normal_x * (center_x - midpoint_x)
            + normal_y * (center_y - midpoint_y)
            > 0.0
        ):
            normal_x = -normal_x
            normal_y = -normal_y

        paper_samples = 0
        same_surface_samples = 0
        for sample_index in range(sample_count):
            fraction = float(sample_index + 1) / float(
                sample_count + 1
            )
            edge_x = float(start[0]) + fraction * dx
            edge_y = float(start[1]) + fraction * dy
            outside_x = edge_x + offset * normal_x
            outside_y = edge_y + offset * normal_y
            inside_x = edge_x - offset * normal_x
            inside_y = edge_y - offset * normal_y
            if (
                outside_x < 0.0
                or outside_y < 0.0
                or outside_x > gray_image.width() - 1
                or outside_y > gray_image.height() - 1
                or inside_x < 0.0
                or inside_y < 0.0
                or inside_x > gray_image.width() - 1
                or inside_y > gray_image.height() - 1
            ):
                continue
            inside_gray = _gray_at(gray_image, inside_x, inside_y)
            outside_gray = _gray_at(
                gray_image, outside_x, outside_y
            )
            if inside_gray > A4_MAX_INSIDE_GRAY:
                continue
            paper_samples += 1
            if (
                outside_gray <= A4_MAX_INSIDE_GRAY
                and abs(outside_gray - inside_gray)
                <= A4_INTERNAL_EDGE_SIMILAR_GRAY_DELTA
            ):
                same_surface_samples += 1

        sample_counts.append(paper_samples)
        if paper_samples:
            ratios.append(
                float(same_surface_samples) / float(paper_samples)
            )
        else:
            ratios.append(0.0)

    internal_edges = [
        edge_index
        for edge_index, ratio in enumerate(ratios)
        if (
            sample_counts[edge_index]
            >= A4_INTERNAL_EDGE_MIN_SAMPLES
            and ratio >= A4_INTERNAL_EDGE_DARK_RATIO_MAX
        )
    ]
    return {
        "internal": bool(internal_edges),
        "edge_ratios": ratios,
        "edge_sample_counts": sample_counts,
        "internal_edge_indices": internal_edges,
    }


def _count_bright_samples(gray_image, corners, side):
    count = 0
    if side in ("left", "right"):
        u_values = (
            (0.12, 0.24, 0.36)
            if side == "left"
            else (0.64, 0.76, 0.88)
        )
        v_values = (0.15, 0.30, 0.45, 0.60, 0.75, 0.88)
    else:
        u_values = (0.15, 0.30, 0.45, 0.60, 0.75, 0.88)
        v_values = (
            (0.12, 0.24, 0.36)
            if side == "top"
            else (0.64, 0.76, 0.88)
        )
    for v in v_values:
        for u in u_values:
            point = _quad_point(corners, u, v)
            x = max(
                0,
                min(gray_image.width() - 1, int(round(point[0]))),
            )
            y = max(
                0,
                min(gray_image.height() - 1, int(round(point[1]))),
            )
            value = gray_image.get_pixel(x, y)
            if isinstance(value, tuple):
                value = value[0]
            if int(value) >= WHITE_GRAY_THRESHOLD:
                count += 1
    return count


def _physical_corner_order(gray_image, image_corners, ratio):
    """Orient image corners so A4 y=0 is the half containing fragments."""
    tl, tr, br, bl = image_corners
    configured = A4_TOP_SIDE
    if ratio > 1.0:
        if configured in ("left", "right"):
            top_side = configured
        else:
            left_count = _count_bright_samples(
                gray_image, image_corners, "left"
            )
            right_count = _count_bright_samples(
                gray_image, image_corners, "right"
            )
            top_side = (
                "left" if left_count >= right_count else "right"
            )
        if top_side == "left":
            # Physical TL,TR,BR,BL after a 90-degree image rotation.
            return [bl, tl, tr, br], "landscape_top_left"
        return [tr, br, bl, tl], "landscape_top_right"

    if configured in ("top", "bottom"):
        top_side = configured
    else:
        top_count = _count_bright_samples(
            gray_image, image_corners, "top"
        )
        bottom_count = _count_bright_samples(
            gray_image, image_corners, "bottom"
        )
        top_side = "top" if top_count >= bottom_count else "bottom"
    if top_side == "bottom":
        return [br, bl, tl, tr], "portrait_top_bottom"
    return [tl, tr, br, bl], "portrait_top_top"


def _reject(diagnostics, reason):
    rejected = diagnostics["rejected"]
    rejected[reason] = rejected.get(reason, 0) + 1
    return None


def _score_candidate(
    gray_image,
    corners,
    magnitude,
    source,
    diagnostics,
):
    corners = order_a4_corners(corners)
    if corners is None:
        return _reject(diagnostics, "corners")

    edge_margin = A4_MIN_IMAGE_EDGE_MARGIN_PX
    if any(
        point[0] <= edge_margin
        or point[1] <= edge_margin
        or point[0] >= gray_image.width() - 1 - edge_margin
        or point[1] >= gray_image.height() - 1 - edge_margin
        for point in corners
    ):
        return _reject(diagnostics, "touches_edge")

    width = 0.5 * (
        _distance(corners[0], corners[1])
        + _distance(corners[3], corners[2])
    )
    height = 0.5 * (
        _distance(corners[0], corners[3])
        + _distance(corners[1], corners[2])
    )
    if min(width, height) < A4_MIN_SIDE_PX or height <= 1.0:
        return _reject(diagnostics, "side")
    ratio = width / height
    if (
        ratio < A4_MIN_WIDTH_HEIGHT_RATIO
        or ratio > A4_MAX_WIDTH_HEIGHT_RATIO
    ):
        return _reject(diagnostics, "aspect")

    frame_area = float(gray_image.width() * gray_image.height())
    area_ratio = _polygon_area(corners) / frame_area
    if (
        area_ratio < A4_MIN_FRAME_AREA_RATIO
        or area_ratio > A4_MAX_FRAME_AREA_RATIO
    ):
        return _reject(diagnostics, "area")

    center_x = sum(point[0] for point in corners) * 0.25
    center_y = sum(point[1] for point in corners) * 0.25
    offset_x = (
        center_x - 0.5 * gray_image.width()
    ) / gray_image.width()
    offset_y = (
        center_y - 0.5 * gray_image.height()
    ) / gray_image.height()
    center_offset = math.sqrt(
        offset_x * offset_x + offset_y * offset_y
    )
    if center_offset > A4_MAX_CENTER_OFFSET_RATIO:
        return _reject(diagnostics, "center")

    inside_gray = _inside_gray(gray_image, corners)
    if inside_gray > A4_MAX_INSIDE_GRAY:
        return _reject(diagnostics, "brightness")

    portrait_error = abs(
        math.log(
            max(1e-6, ratio)
            / A4_EXPECTED_WIDTH_HEIGHT_RATIO
        )
    )
    landscape_error = abs(
        math.log(
            max(1e-6, ratio)
            * A4_EXPECTED_WIDTH_HEIGHT_RATIO
        )
    )
    aspect_error = min(portrait_error, landscape_error)
    physical_corners, orientation = _physical_corner_order(
        gray_image, corners, ratio
    )
    divider = _divider_line_probe(
        gray_image, physical_corners, inside_gray
    )
    edge_probe = _internal_edge_probe(gray_image, corners)
    if edge_probe["internal"] and not divider["detected"]:
        return _reject(diagnostics, "internal_edge")
    if edge_probe["internal"]:
        # A slightly inset min_corners/find_rects edge can leave dark paper on
        # both sides and look internal.  A continuous centre divider across
        # the configured coverage is stronger evidence that this is the full
        # A4 sheet.  A divider-generated half-sheet has no second centre line,
        # so it still fails closed above.
        diagnostics["divider_rescued_internal_edge"] = (
            diagnostics.get("divider_rescued_internal_edge", 0) + 1
        )
    if A4_REQUIRE_DIVIDER_FOR_LOCK and not divider["detected"]:
        reason = (
            "divider_slope"
            if divider.get("line_found", False)
            else "divider"
        )
        return _reject(diagnostics, reason)
    physical_corners = list(physical_corners)
    darkness_loss = inside_gray / 255.0
    magnitude_bonus = min(0.20, float(magnitude) / 80000.0)
    source_penalty = 0.0 if source == "rect" else 0.07
    score = (
        2.20 * aspect_error
        + 0.55 * center_offset
        + 0.18 * (1.0 - area_ratio)
        + 0.32 * darkness_loss
        + source_penalty
        - magnitude_bonus
    )
    confidence = max(0.0, min(1.0, 1.0 - score / 1.4))
    return {
        "corners_work_px": physical_corners,
        "image_corners_work_px": corners,
        "score": score,
        "confidence": confidence,
        "area_ratio": area_ratio,
        "aspect_ratio": ratio,
        "inside_gray": inside_gray,
        "edge_outside_dark_ratios": edge_probe["edge_ratios"],
        "edge_probe_sample_counts": edge_probe[
            "edge_sample_counts"
        ],
        "source": source,
        "magnitude": float(magnitude),
        "orientation": orientation,
        "divider_detected": divider["detected"],
        "divider_y_mm": divider["divider_y_mm"],
        "divider_slope_mm": divider["slope_mm"],
        "divider_coverage": divider["coverage"],
        "divider_residual_px": divider["residual_px"],
        "divider_confidence": divider["confidence"],
    }


def _rect_candidates(gray_image, diagnostics):
    candidates = []
    try:
        rectangles = gray_image.find_rects(
            threshold=A4_RECT_EDGE_THRESHOLD
        )
        diagnostics["raw_rects"] = len(rectangles)
        for rectangle in rectangles:
            candidate = _score_candidate(
                gray_image,
                rectangle.corners(),
                rectangle.magnitude(),
                "rect",
                diagnostics,
            )
            if candidate is not None:
                candidates.append(candidate)
    except Exception as exc:
        if "IDE interrupt" in str(exc):
            raise
        diagnostics["rect_error"] = str(exc)
    return candidates


def _dark_blob_candidates(gray_image, diagnostics):
    candidates = []
    image_area = gray_image.width() * gray_image.height()
    minimum = max(
        100,
        int(image_area * A4_DARK_BLOB_MIN_AREA_RATIO),
    )
    try:
        blobs = gray_image.find_blobs(
            [(0, A4_DARK_THRESHOLD)],
            x_stride=2,
            y_stride=2,
            pixels_threshold=minimum,
            area_threshold=minimum,
            merge=True,
            margin=4,
        )
        diagnostics["raw_dark_blobs"] = len(blobs)
        for blob in blobs:
            contour_corners = getattr(blob, "corners", None)
            if contour_corners is None:
                raw_corners = blob.min_corners()
                blob_source = "dark_blob_min_box"
            else:
                raw_corners = (
                    contour_corners()
                    if callable(contour_corners)
                    else contour_corners
                )
                blob_source = "dark_blob_contour"
            candidate = _score_candidate(
                gray_image,
                raw_corners,
                float(blob.pixels()),
                blob_source,
                diagnostics,
            )
            if candidate is not None:
                candidates.append(candidate)
    except Exception as exc:
        if "IDE interrupt" in str(exc):
            raise
        diagnostics["blob_error"] = str(exc)
    return candidates


def detect_a4_boundary(gray_image, source_frame_size):
    """Return the best A4 quadrilateral in source-frame pixel coordinates."""
    diagnostics = {
        "raw_rects": 0,
        "raw_dark_blobs": 0,
        "rect_error": "",
        "blob_error": "",
        "valid_candidates": 0,
        "divider_rescued_internal_edge": 0,
        "rejected": {},
    }
    candidates = _rect_candidates(gray_image, diagnostics)
    # Avoid a second full connected-component pass when find_rects already
    # produced a valid A4. Dark blobs are the fallback for blurred/weak edges.
    if not candidates:
        candidates.extend(
            _dark_blob_candidates(gray_image, diagnostics)
        )
    diagnostics["valid_candidates"] = len(candidates)
    if not candidates:
        return None, diagnostics
    candidates.sort(key=lambda item: item["score"])
    best = candidates[0]

    scale_x = float(source_frame_size[0] - 1) / float(
        gray_image.width() - 1
    )
    scale_y = float(source_frame_size[1] - 1) / float(
        gray_image.height() - 1
    )
    best["corners_px"] = [
        (point[0] * scale_x, point[1] * scale_y)
        for point in best["corners_work_px"]
    ]
    return best, diagnostics


class A4BoundaryTracker:
    """Confirm the initial A4 boundary, then retain an immutable calibration."""

    def __init__(self):
        self.corners_px = None
        self.confidence = 0.0
        self.source = ""
        self.orientation = ""
        self.valid_frames = 0
        self.missed_frames = 0
        self.motion_px = 0.0
        self.locked = False
        self.frozen = False
        self.relock_confirm_count = 0
        self.divider_y_mm = DIVIDER_Y_MM
        self.divider_slope_mm = 0.0
        self.divider_confidence = 0.0
        self.divider_detected = False

    def update(self, candidate):
        if self.frozen:
            return self.state()
        if candidate is None:
            self.missed_frames += 1
            # Lock acquisition requires truly consecutive valid frames.
            # Holding through short misses is allowed only after lock.
            if not self.locked:
                self.corners_px = None
                self.valid_frames = 0
                self.confidence = 0.0
                self.source = ""
                self.orientation = ""
                self.motion_px = 0.0
                self.relock_confirm_count = 0
                self.divider_y_mm = DIVIDER_Y_MM
                self.divider_slope_mm = 0.0
                self.divider_confidence = 0.0
                self.divider_detected = False
            elif self.missed_frames > A4_HOLD_MISSED_FRAMES:
                self.corners_px = None
                self.valid_frames = 0
                self.locked = False
                self.confidence = 0.0
                self.source = ""
                self.orientation = ""
                self.motion_px = 0.0
                self.relock_confirm_count = 0
                self.divider_y_mm = DIVIDER_Y_MM
                self.divider_slope_mm = 0.0
                self.divider_confidence = 0.0
                self.divider_detected = False
            return self.state()

        detected = candidate["corners_px"]
        divider_calibration_acquired = (
            candidate.get("divider_detected", False)
            and not self.divider_detected
        )
        self.missed_frames = 0
        if self.corners_px is None:
            self.corners_px = [
                (float(point[0]), float(point[1]))
                for point in detected
            ]
            self.motion_px = 0.0
            self.valid_frames = 1
            self.relock_confirm_count = 0
        else:
            distances = [
                _distance(current, observed)
                for current, observed in zip(
                    self.corners_px, detected
                )
            ]
            self.motion_px = max(distances)
            if self.motion_px <= A4_LOCK_DEADBAND_PX:
                # A fixed camera/A4 should not feed sub-deadband jitter into
                # every later perspective transform.
                if divider_calibration_acquired:
                    self.corners_px = [
                        (float(point[0]), float(point[1]))
                        for point in detected
                    ]
                self.valid_frames += 1
                self.relock_confirm_count = 0
            elif self.motion_px <= A4_RELOCK_MOTION_PX:
                alpha = A4_SLOW_SMOOTH_ALPHA
                self.corners_px = [
                    (
                        current[0]
                        + alpha * (observed[0] - current[0]),
                        current[1]
                        + alpha * (observed[1] - current[1]),
                    )
                    for current, observed in zip(
                        self.corners_px, detected
                    )
                ]
                self.valid_frames += 1
                self.relock_confirm_count = 0
            else:
                self.relock_confirm_count += 1
            if (
                self.motion_px > A4_RELOCK_MOTION_PX
                and self.relock_confirm_count
                >= A4_RELOCK_CONFIRM_FRAMES
            ):
                self.corners_px = [
                    (float(point[0]), float(point[1]))
                    for point in detected
                ]
                self.valid_frames = 1
                self.relock_confirm_count = 0

        self.confidence = candidate["confidence"]
        self.source = candidate["source"]
        self.orientation = candidate.get("orientation", "")
        if candidate.get("divider_detected", False):
            observed_divider = candidate.get(
                "divider_y_mm", DIVIDER_Y_MM
            )
            observed_slope = candidate.get(
                "divider_slope_mm", 0.0
            )
            alpha = (
                1.0
                if not self.divider_detected
                else DIVIDER_TRACK_ALPHA
            )
            self.divider_y_mm += alpha * (
                observed_divider - self.divider_y_mm
            )
            self.divider_slope_mm += alpha * (
                observed_slope - self.divider_slope_mm
            )
            self.divider_confidence = candidate.get(
                "divider_confidence", 0.0
            )
            self.divider_detected = True
        self.locked = (
            self.valid_frames >= A4_LOCK_REQUIRED_FRAMES
        )
        return self.state()

    def freeze(self):
        """Make the first locked calibration immutable until process restart."""
        if self.locked and self.corners_px is not None:
            self.frozen = True
            self.motion_px = 0.0
            self.missed_frames = 0
            self.relock_confirm_count = 0
        return self.state()

    def state(self):
        return {
            "corners_px": self.corners_px,
            "locked": self.locked,
            "frozen": self.frozen,
            "confidence": self.confidence,
            "source": self.source,
            "orientation": self.orientation,
            "valid_frames": self.valid_frames,
            "missed_frames": self.missed_frames,
            "motion_px": self.motion_px,
            "relock_confirm_count": self.relock_confirm_count,
            "divider_y_mm": self.divider_y_mm,
            "divider_slope_mm": self.divider_slope_mm,
            "divider_confidence": self.divider_confidence,
            "divider_detected": self.divider_detected,
        }

#!/usr/bin/env python3
"""CanMV K230 standalone A4 recognition test entrypoint.

This entrypoint intentionally runs only the current A4 boundary detector and
tracker.  It does not import piece recognition, geometry planning, or placement
code.  Hardware imports stay inside ``main`` so desktop tests can import the
small formatting and drawing helpers.
"""

import gc
import os
import time



# This test is automatic-only: it never loads configured A4 corner points.
# Keep detecting after the initial three-frame lock so moving or replacing the
# sheet automatically updates/reacquires the calibration.
AUTO_CALIBRATE_A4 = True
FREEZE_AFTER_LOCK = False
DETECT_EVERY_N_FRAMES = 1

WHITE = (235, 235, 235)
GRAY = (150, 150, 150)
RED = (255, 65, 65)
GREEN = (70, 240, 100)
YELLOW = (255, 210, 40)
CYAN = (80, 220, 255)


def _ms_now():
    if hasattr(time, "ticks_ms"):
        return int(time.ticks_ms())
    return int(time.time() * 1000)


def _ms_delta(newer, older):
    if hasattr(time, "ticks_diff"):
        return int(time.ticks_diff(newer, older))
    if newer >= older:
        return newer - older
    return max(1, newer)


def _sleep_ms(milliseconds):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(milliseconds)
    else:
        time.sleep(milliseconds / 1000.0)


def _format_corners(corners):
    if not corners:
        return "none"
    return "|".join(
        "{:.0f}:{:.0f}".format(point[0], point[1])
        for point in corners
    )


def _format_rejections(diagnostics):
    rejected = diagnostics.get("rejected", {})
    if not rejected:
        return "none"
    return "|".join(
        "{}:{}".format(name, count)
        for name, count in sorted(rejected.items())
    )


def _divider_points(corners, divider_y_mm):
    """Return the two display endpoints of the A4 horizontal divider."""
    if corners is None or len(corners) != 4:
        return None
    fraction = max(
        0.0,
        min(1.0, float(divider_y_mm) / float(A4_HEIGHT_MM)),
    )
    top_left, top_right, bottom_right, bottom_left = corners
    return (
        (
            top_left[0] + fraction * (bottom_left[0] - top_left[0]),
            top_left[1] + fraction * (bottom_left[1] - top_left[1]),
        ),
        (
            top_right[0] + fraction * (bottom_right[0] - top_right[0]),
            top_right[1] + fraction * (bottom_right[1] - top_right[1]),
        ),
    )


def _draw_text(frame, x, y, text, color=WHITE, size=18):
    try:
        frame.draw_string_advanced(
            int(x),
            int(y),
            int(size),
            str(text),
            color=color,
        )
    except Exception:
        try:
            frame.draw_string(int(x), int(y), str(text), color=color)
        except Exception:
            pass


def _draw_quad(frame, corners, color, thickness=3):
    if corners is None:
        return
    for index, point in enumerate(corners):
        other = corners[(index + 1) % len(corners)]
        frame.draw_line(
            int(point[0]),
            int(point[1]),
            int(other[0]),
            int(other[1]),
            color=color,
            thickness=thickness,
        )


def _draw_calibration(frame, candidate, state):
    if candidate is not None:
        _draw_quad(frame, candidate["corners_px"], YELLOW, thickness=2)

    corners = state.get("corners_px")
    if corners is None:
        return
    color = GREEN if state["locked"] else YELLOW
    _draw_quad(frame, corners, color, thickness=3)

    divider = _divider_points(
        corners,
        state.get("divider_y_mm", DIVIDER_Y_MM),
    )
    if divider is not None:
        frame.draw_line(
            int(divider[0][0]),
            int(divider[0][1]),
            int(divider[1][0]),
            int(divider[1][1]),
            color=CYAN,
            thickness=2,
        )

    for label, point in zip(("TL", "TR", "BR", "BL"), corners):
        x = int(point[0])
        y = int(point[1])
        try:
            frame.draw_cross(x, y, color=color, size=8, thickness=2)
        except Exception:
            pass
        _draw_text(
            frame,
            max(0, min(frame.width() - 30, x + 6)),
            max(42, min(frame.height() - 22, y - 18)),
            label,
            color,
            16,
        )


def _print_status(
    frame_index,
    state,
    candidate,
    diagnostics,
    fps,
    event="STATUS",
):
    print(
        "A4_TEST_{},frame={},locked={},frozen={},candidate={},"
        "confidence={:.2f},valid_frames={},missed_frames={},"
        "motion_px={:.1f},source={},orientation={},divider={},"
        "divider_y_mm={:.1f},divider_slope_mm={:.1f},"
        "corners={},rects={},dark_blobs={},valid_candidates={},"
        "rejected={},fps={:.1f}".format(
            event,
            frame_index,
            int(state.get("locked", False)),
            int(state.get("frozen", False)),
            int(candidate is not None),
            state.get("confidence", 0.0),
            state.get("valid_frames", 0),
            state.get("missed_frames", 0),
            state.get("motion_px", 0.0),
            state.get("source", ""),
            state.get("orientation", ""),
            int(state.get("divider_detected", False)),
            state.get("divider_y_mm", DIVIDER_Y_MM),
            state.get("divider_slope_mm", 0.0),
            _format_corners(state.get("corners_px")),
            diagnostics.get("raw_rects", 0),
            diagnostics.get("raw_dark_blobs", 0),
            diagnostics.get("valid_candidates", 0),
            _format_rejections(diagnostics),
            fps,
        )
    )


def _stop_requested(start_ms, frame_index):
    if AUTO_STOP_SECONDS > 0:
        if _ms_delta(_ms_now(), start_ms) >= AUTO_STOP_SECONDS * 1000:
            return True
    if MAX_FRAME_COUNT > 0 and frame_index >= MAX_FRAME_COUNT:
        return True
    return False


def main():
    from media.display import Display
    from media.media import MediaManager
    from media.sensor import Sensor

    sensor = None
    tracker = A4BoundaryTracker()
    state = tracker.state()
    diagnostics = {}
    candidate = None
    frame_index = 0
    fps = 0.0
    start_ms = _ms_now()
    last_locked = False

    try:
        sensor = Sensor()
        sensor.reset()
        sensor.set_hmirror(True)
        sensor.set_vflip(True)
        sensor.set_framesize(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
        )
        sensor.set_pixformat(Sensor.RGB565)

        Display.init(
            Display.ST7701,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            to_ide=True,
        )
        MediaManager.init()
        sensor.run()

        print(
            "A4_TEST_START,frame={}x{},detect={}x{},"
            "auto_calibrate={},lock_frames={},freeze_after_lock={},"
            "bundle={}".format(
                FRAME_WIDTH,
                FRAME_HEIGHT,
                A4_DETECT_WIDTH,
                A4_DETECT_HEIGHT,
                int(AUTO_CALIBRATE_A4),
                A4_LOCK_REQUIRED_FRAMES,
                int(FREEZE_AFTER_LOCK),
                "single"
                if True
                else "modules",
            )
        )

        while True:
            os.exitpoint()
            if _stop_requested(start_ms, frame_index):
                print(
                    "A4_TEST_STOP,reason=configured_limit,frame={}".format(
                        frame_index
                    )
                )
                break

            frame_start = _ms_now()
            frame = sensor.snapshot()
            if frame is None:
                state = tracker.update(None)
                if frame_index % A4_STATUS_PRINT_EVERY_N_FRAMES == 0:
                    print(
                        "A4_TEST_ERROR,frame={},reason=snapshot_none".format(
                            frame_index
                        )
                    )
                frame_index += 1
                continue

            error = None
            try:
                should_detect = (
                    not state.get("frozen", False)
                    and frame_index % max(1, DETECT_EVERY_N_FRAMES) == 0
                )
                if should_detect:
                    boundary_gray = frame.to_grayscale(
                        x_size=A4_DETECT_WIDTH,
                        y_size=A4_DETECT_HEIGHT,
                    )
                    if boundary_gray is None:
                        raise RuntimeError("to_grayscale returned None")
                    candidate, diagnostics = detect_a4_boundary(
                        boundary_gray,
                        (FRAME_WIDTH, FRAME_HEIGHT),
                    )
                    state = tracker.update(candidate)
                    if (
                        FREEZE_AFTER_LOCK
                        and state["locked"]
                        and not state.get("frozen", False)
                    ):
                        state = tracker.freeze()
            except Exception as exc:
                if "IDE interrupt" in str(exc):
                    raise
                error = str(exc)
                state = tracker.state()

            elapsed = max(1, _ms_delta(_ms_now(), frame_start))
            instant_fps = 1000.0 / elapsed
            fps = (
                instant_fps
                if fps <= 0.0
                else 0.85 * fps + 0.15 * instant_fps
            )

            if state["locked"] and not last_locked:
                _print_status(
                    frame_index,
                    state,
                    candidate,
                    diagnostics,
                    fps,
                    event="LOCK",
                )
            elif not state["locked"] and last_locked:
                _print_status(
                    frame_index,
                    state,
                    candidate,
                    diagnostics,
                    fps,
                    event="LOST",
                )
            elif frame_index % A4_STATUS_PRINT_EVERY_N_FRAMES == 0:
                _print_status(
                    frame_index,
                    state,
                    candidate,
                    diagnostics,
                    fps,
                )

            _draw_calibration(frame, candidate, state)
            status_color = GREEN if state["locked"] else YELLOW
            _draw_text(
                frame,
                8,
                6,
                "{}  C:{:.2f}  F:{}  FPS:{:.1f}".format(
                    "A4 LOCK" if state["locked"] else "A4 SEARCH",
                    state.get("confidence", 0.0),
                    state.get("valid_frames", 0),
                    fps,
                ),
                status_color,
                22,
            )
            if error:
                _draw_text(
                    frame,
                    8,
                    34,
                    "ERROR: {}".format(error[:68]),
                    RED,
                    16,
                )
            else:
                _draw_text(
                    frame,
                    8,
                    34,
                    "R:{} B:{} V:{} D:{} Y:{:.1f}".format(
                        diagnostics.get("raw_rects", 0),
                        diagnostics.get("raw_dark_blobs", 0),
                        diagnostics.get("valid_candidates", 0),
                        int(state.get("divider_detected", False)),
                        state.get("divider_y_mm", DIVIDER_Y_MM),
                    ),
                    GRAY,
                    16,
                )
            Display.show_image(frame)

            last_locked = state["locked"]
            frame_index += 1
            if frame_index % 30 == 0:
                gc.collect()
            if LOOP_IDLE_MS > 0:
                _sleep_ms(LOOP_IDLE_MS)

    except KeyboardInterrupt:
        print("A4_TEST_STOP,reason=user,frame={}".format(frame_index))
    except BaseException as exc:
        reason = str(exc)
        if "IDE interrupt" in reason:
            print(
                "A4_TEST_STOP,reason=ide_interrupt,frame={}".format(
                    frame_index
                )
            )
        else:
            print(
                "A4_TEST_FATAL,frame={},reason={}".format(
                    frame_index,
                    reason.replace(",", ";"),
                )
            )
            return 1
    finally:
        if sensor is not None:
            try:
                sensor.stop()
            except Exception as exc:
                print("A4_TEST_CLEANUP_WARNING,sensor={}".format(exc))
        try:
            Display.deinit()
        except Exception as exc:
            print("A4_TEST_CLEANUP_WARNING,display={}".format(exc))
        try:
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            _sleep_ms(100)
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception as exc:
            print("A4_TEST_CLEANUP_WARNING,media={}".format(exc))
        gc.collect()
    return 0


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    main()
