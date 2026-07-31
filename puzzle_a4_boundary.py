"""Native CanMV automatic A4 boundary detection and corner tracking."""

import math

import puzzle_config as cfg


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


def project_a4_mm_to_frame(point_mm, corners):
    """Project one physical A4 millimetre point into the camera frame."""
    if corners is None or len(corners) != 4:
        return None
    return _projective_quad_point(
        corners,
        float(point_mm[0]) / float(cfg.A4_WIDTH_MM),
        float(point_mm[1]) / float(cfg.A4_HEIGHT_MM),
    )


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
        cfg.DIVIDER_Y_MM / cfg.A4_HEIGHT_MM
    )
    result = {
        "detected": False,
        "line_found": False,
        "reject_reason": "",
        "center_fraction": nominal_fraction,
        "left_fraction": nominal_fraction,
        "right_fraction": nominal_fraction,
        "divider_y_mm": cfg.DIVIDER_Y_MM,
        "slope_mm": 0.0,
        "coverage": 0.0,
        "residual_px": 0.0,
        "confidence": 0.0,
    }
    if not cfg.ENABLE_DYNAMIC_DIVIDER:
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
        cfg.DIVIDER_SEARCH_HALF_RANGE_MM
        / cfg.A4_HEIGHT_MM
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
        7, int(cfg.DIVIDER_LINE_SAMPLE_COUNT)
    )
    minimum_bright_gray = max(
        float(cfg.DIVIDER_LINE_MIN_GRAY),
        float(inside_gray)
        + float(cfg.DIVIDER_LINE_MIN_CONTRAST_GRAY),
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
            * cfg.DIVIDER_LINE_MIN_COVERAGE
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
        <= cfg.DIVIDER_LINE_MAX_RESIDUAL_PX
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
        / max(0.1, cfg.DIVIDER_LINE_MAX_RESIDUAL_PX),
    )
    slope_mm = (
        (right_fraction - left_fraction)
        * cfg.A4_HEIGHT_MM
    )
    result.update(
        {
            "line_found": True,
            "center_fraction": center_fraction,
            "left_fraction": left_fraction,
            "right_fraction": right_fraction,
            "divider_y_mm": (
                center_fraction * cfg.A4_HEIGHT_MM
            ),
            "slope_mm": slope_mm,
            "coverage": coverage,
            "residual_px": residual_px,
            "confidence": coverage * residual_quality,
        }
    )
    if abs(slope_mm) > cfg.DIVIDER_LINE_MAX_SLOPE_MM:
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
    sample_count = max(1, int(cfg.A4_EDGE_PROBE_SAMPLES))
    offset = float(cfg.A4_EDGE_PROBE_OFFSET_PX)
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
            if inside_gray > cfg.A4_MAX_INSIDE_GRAY:
                continue
            paper_samples += 1
            if (
                outside_gray <= cfg.A4_MAX_INSIDE_GRAY
                and abs(outside_gray - inside_gray)
                <= cfg.A4_INTERNAL_EDGE_SIMILAR_GRAY_DELTA
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
            >= cfg.A4_INTERNAL_EDGE_MIN_SAMPLES
            and ratio >= cfg.A4_INTERNAL_EDGE_DARK_RATIO_MAX
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
            if int(value) >= cfg.WHITE_GRAY_THRESHOLD:
                count += 1
    return count


def _physical_corner_order(gray_image, image_corners, ratio):
    """Orient image corners so A4 y=0 is the half containing fragments."""
    tl, tr, br, bl = image_corners
    configured = cfg.A4_TOP_SIDE
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


def _corner_debug_text(points):
    if points is None:
        return "none"
    try:
        values = []
        for point in list(points)[:6]:
            try:
                values.append(
                    "{:.0f}:{:.0f}".format(
                        float(point[0]), float(point[1])
                    )
                )
            except Exception:
                values.append(str(point).replace(",", ":"))
        return "{}@{}".format(len(points), "/".join(values))
    except Exception:
        return str(points).replace(",", ":")


def _reject(
    diagnostics, reason, source="unknown", corners=None, detail=""
):
    rejected = diagnostics["rejected"]
    rejected[reason] = rejected.get(reason, 0) + 1
    records = diagnostics.get("candidate_rejections")
    if records is not None and len(records) < 12:
        records.append(
            "{}:{}:{}:{}".format(
                source,
                reason,
                _corner_debug_text(corners),
                detail or "na",
            )
        )
    return None


def _score_candidate(
    gray_image,
    corners,
    magnitude,
    source,
    diagnostics,
):
    raw_corners = corners
    corners = order_a4_corners(corners)
    if corners is None:
        return _reject(
            diagnostics, "corners", source, raw_corners
        )

    edge_margin = cfg.A4_MIN_IMAGE_EDGE_MARGIN_PX
    if any(
        point[0] <= edge_margin
        or point[1] <= edge_margin
        or point[0] >= gray_image.width() - 1 - edge_margin
        or point[1] >= gray_image.height() - 1 - edge_margin
        for point in corners
    ):
        return _reject(
            diagnostics, "touches_edge", source, corners
        )

    width = 0.5 * (
        _distance(corners[0], corners[1])
        + _distance(corners[3], corners[2])
    )
    height = 0.5 * (
        _distance(corners[0], corners[3])
        + _distance(corners[1], corners[2])
    )
    if min(width, height) < cfg.A4_MIN_SIDE_PX or height <= 1.0:
        return _reject(
            diagnostics,
            "side",
            source,
            corners,
            "w{:.1f}_h{:.1f}".format(width, height),
        )
    ratio = width / height
    if (
        ratio < cfg.A4_MIN_WIDTH_HEIGHT_RATIO
        or ratio > cfg.A4_MAX_WIDTH_HEIGHT_RATIO
    ):
        return _reject(
            diagnostics,
            "aspect",
            source,
            corners,
            "ratio{:.3f}".format(ratio),
        )

    frame_area = float(gray_image.width() * gray_image.height())
    area_ratio = _polygon_area(corners) / frame_area
    if (
        area_ratio < cfg.A4_MIN_FRAME_AREA_RATIO
        or area_ratio > cfg.A4_MAX_FRAME_AREA_RATIO
    ):
        return _reject(
            diagnostics,
            "area",
            source,
            corners,
            "ratio{:.3f}".format(area_ratio),
        )

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
    if center_offset > cfg.A4_MAX_CENTER_OFFSET_RATIO:
        return _reject(
            diagnostics,
            "center",
            source,
            corners,
            "offset{:.3f}".format(center_offset),
        )

    inside_gray = _inside_gray(gray_image, corners)
    if inside_gray > cfg.A4_MAX_INSIDE_GRAY:
        return _reject(
            diagnostics,
            "brightness",
            source,
            corners,
            "inside{:.1f}".format(inside_gray),
        )

    portrait_error = abs(
        math.log(
            max(1e-6, ratio)
            / cfg.A4_EXPECTED_WIDTH_HEIGHT_RATIO
        )
    )
    landscape_error = abs(
        math.log(
            max(1e-6, ratio)
            * cfg.A4_EXPECTED_WIDTH_HEIGHT_RATIO
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
        return _reject(
            diagnostics,
            "internal_edge",
            source,
            corners,
        )
    if edge_probe["internal"]:
        # A slightly inset min_corners/find_rects edge can leave dark paper on
        # both sides and look internal.  A continuous centre divider across
        # the configured coverage is stronger evidence that this is the full
        # A4 sheet.  A divider-generated half-sheet has no second centre line,
        # so it still fails closed above.
        diagnostics["divider_rescued_internal_edge"] = (
            diagnostics.get("divider_rescued_internal_edge", 0) + 1
        )
    if cfg.A4_REQUIRE_DIVIDER_FOR_LOCK and not divider["detected"]:
        reason = (
            "divider_slope"
            if divider.get("line_found", False)
            else "divider"
        )
        return _reject(
            diagnostics,
            reason,
            source,
            corners,
            "coverage{:.2f}_slope{:.2f}".format(
                divider.get("coverage", 0.0),
                divider.get("slope_mm", 0.0),
            ),
        )
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
            threshold=cfg.A4_RECT_EDGE_THRESHOLD
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
        int(image_area * cfg.A4_DARK_BLOB_MIN_AREA_RATIO),
    )
    try:
        blobs = gray_image.find_blobs(
            [(0, cfg.A4_DARK_THRESHOLD)],
            x_stride=2,
            y_stride=2,
            pixels_threshold=minimum,
            area_threshold=minimum,
            merge=True,
            margin=4,
        )
        diagnostics["raw_dark_blobs"] = len(blobs)
        for blob in blobs:
            attempts = []
            contour_corners = getattr(blob, "corners", None)
            if contour_corners is not None:
                raw_corners = (
                    contour_corners()
                    if callable(contour_corners)
                    else contour_corners
                )
                attempts.append(
                    ("dark_blob_contour", raw_corners)
                )
                diagnostics["dark_blob_contour_corner_counts"].append(
                    len(raw_corners) if raw_corners is not None else 0
                )
            minimum_corners = getattr(blob, "min_corners", None)
            if minimum_corners is not None:
                raw_minimum = (
                    minimum_corners()
                    if callable(minimum_corners)
                    else minimum_corners
                )
                diagnostics["dark_blob_min_corner_counts"].append(
                    len(raw_minimum) if raw_minimum is not None else 0
                )
                if not attempts or raw_minimum != attempts[0][1]:
                    attempts.append(
                        ("dark_blob_min_box", raw_minimum)
                    )
            blob_rect = getattr(blob, "rect", None)
            if blob_rect is not None:
                rect = blob_rect() if callable(blob_rect) else blob_rect
                if rect is not None and len(rect) >= 4:
                    rect_x = float(rect[0])
                    rect_y = float(rect[1])
                    rect_width = float(rect[2])
                    rect_height = float(rect[3])
                    raw_rect_box = [
                        (rect_x, rect_y),
                        (rect_x + rect_width - 1.0, rect_y),
                        (
                            rect_x + rect_width - 1.0,
                            rect_y + rect_height - 1.0,
                        ),
                        (rect_x, rect_y + rect_height - 1.0),
                    ]
                    diagnostics["dark_blob_rects"].append(
                        tuple(rect[:4])
                    )
                    if all(
                        raw_rect_box != existing[1]
                        for existing in attempts
                    ):
                        attempts.append(
                            ("dark_blob_rect_box", raw_rect_box)
                        )
            for attempt_index, (blob_source, raw_corners) in enumerate(
                attempts
            ):
                try:
                    candidate = _score_candidate(
                        gray_image,
                        raw_corners,
                        float(blob.pixels()),
                        blob_source,
                        diagnostics,
                    )
                except Exception as exc:
                    if "IDE interrupt" in str(exc):
                        raise
                    diagnostics["dark_blob_attempt_errors"].append(
                        "{}:{}".format(
                            blob_source,
                            str(exc).replace(",", ";"),
                        )
                    )
                    continue
                if candidate is not None:
                    if attempt_index > 0:
                        diagnostics["dark_blob_fallback_sources"].append(
                            blob_source
                        )
                        if blob_source == "dark_blob_min_box":
                            diagnostics[
                                "dark_blob_min_corner_fallbacks"
                            ] += 1
                    candidates.append(candidate)
                    break
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
        "dark_blob_contour_corner_counts": [],
        "dark_blob_min_corner_counts": [],
        "dark_blob_min_corner_fallbacks": 0,
        "dark_blob_fallback_sources": [],
        "dark_blob_rects": [],
        "dark_blob_attempt_errors": [],
        "candidate_rejections": [],
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
    """Confirm an A4 boundary and optionally continue tracking/relocking it."""

    def __init__(self, continuous=False):
        self.continuous = bool(continuous)
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
        self.divider_y_mm = cfg.DIVIDER_Y_MM
        self.divider_slope_mm = 0.0
        self.divider_confidence = 0.0
        self.divider_detected = False
        self.candidate_history = []
        self.lock_spread_px = 0.0
        self.calibration_generation = 0
        self.relock_required = False

    def _clear_lock(self):
        self.corners_px = None
        self.valid_frames = 0
        self.locked = False
        self.confidence = 0.0
        self.source = ""
        self.orientation = ""
        self.motion_px = 0.0
        self.relock_confirm_count = 0
        self.divider_y_mm = cfg.DIVIDER_Y_MM
        self.divider_slope_mm = 0.0
        self.divider_confidence = 0.0
        self.divider_detected = False
        self.candidate_history = []
        self.lock_spread_px = 0.0

    def _continuous_update(self, candidate):
        if candidate is None:
            self.missed_frames += 1
            if not self.locked:
                self._clear_lock()
            elif self.missed_frames > cfg.A4_HOLD_MISSED_FRAMES:
                self._clear_lock()
                self.relock_required = True
            return self.state()

        detected = [
            (float(point[0]), float(point[1]))
            for point in candidate["corners_px"]
        ]
        self.missed_frames = 0
        self.confidence = candidate["confidence"]
        self.source = candidate["source"]
        self.orientation = candidate.get("orientation", "")
        if not self.locked:
            self.candidate_history.append(detected)
            required = max(1, int(cfg.A4_LOCK_REQUIRED_FRAMES))
            if len(self.candidate_history) > required:
                self.candidate_history.pop(0)
            median_corners = []
            for corner_index in range(4):
                xs = sorted(
                    sample[corner_index][0]
                    for sample in self.candidate_history
                )
                ys = sorted(
                    sample[corner_index][1]
                    for sample in self.candidate_history
                )
                middle = len(xs) // 2
                if len(xs) % 2:
                    median_corners.append((xs[middle], ys[middle]))
                else:
                    median_corners.append(
                        (
                            0.5 * (xs[middle - 1] + xs[middle]),
                            0.5 * (ys[middle - 1] + ys[middle]),
                        )
                    )
            self.lock_spread_px = max(
                _distance(sample[index], median_corners[index])
                for sample in self.candidate_history
                for index in range(4)
            )
            maximum_spread = float(
                getattr(cfg, "A4_LOCK_MAX_SPREAD_PX", 4.0)
            )
            if self.lock_spread_px > maximum_spread:
                self.candidate_history = [detected]
                median_corners = detected
                self.lock_spread_px = 0.0
            self.corners_px = median_corners
            self.valid_frames = len(self.candidate_history)
            self.motion_px = self.lock_spread_px
            if self.valid_frames >= required:
                self.locked = True
                self.calibration_generation += 1
                self.relock_required = False
                self.relock_confirm_count = 0
            return self.state()

        distances = [
            _distance(current, observed)
            for current, observed in zip(self.corners_px, detected)
        ]
        self.motion_px = max(distances)
        if self.motion_px <= cfg.A4_LOCK_DEADBAND_PX:
            self.relock_confirm_count = 0
        elif self.motion_px <= cfg.A4_RELOCK_MOTION_PX:
            alpha = float(cfg.A4_SLOW_SMOOTH_ALPHA)
            self.corners_px = [
                (
                    current[0] + alpha * (observed[0] - current[0]),
                    current[1] + alpha * (observed[1] - current[1]),
                )
                for current, observed in zip(self.corners_px, detected)
            ]
            self.relock_confirm_count = 0
        else:
            self.relock_confirm_count += 1
            if (
                self.relock_confirm_count
                >= int(cfg.A4_RELOCK_CONFIRM_FRAMES)
            ):
                self._clear_lock()
                self.candidate_history = [detected]
                self.corners_px = detected
                self.valid_frames = 1
                self.relock_required = True
                self.motion_px = max(distances)
        return self.state()

    def update(self, candidate):
        if self.continuous:
            return self._continuous_update(candidate)
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
                self.divider_y_mm = cfg.DIVIDER_Y_MM
                self.divider_slope_mm = 0.0
                self.divider_confidence = 0.0
                self.divider_detected = False
            elif self.missed_frames > cfg.A4_HOLD_MISSED_FRAMES:
                self.corners_px = None
                self.valid_frames = 0
                self.locked = False
                self.confidence = 0.0
                self.source = ""
                self.orientation = ""
                self.motion_px = 0.0
                self.relock_confirm_count = 0
                self.divider_y_mm = cfg.DIVIDER_Y_MM
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
            if self.motion_px <= cfg.A4_LOCK_DEADBAND_PX:
                # A fixed camera/A4 should not feed sub-deadband jitter into
                # every later perspective transform.
                if divider_calibration_acquired:
                    self.corners_px = [
                        (float(point[0]), float(point[1]))
                        for point in detected
                    ]
                self.valid_frames += 1
                self.relock_confirm_count = 0
            elif self.motion_px <= cfg.A4_RELOCK_MOTION_PX:
                alpha = cfg.A4_SLOW_SMOOTH_ALPHA
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
                self.motion_px > cfg.A4_RELOCK_MOTION_PX
                and self.relock_confirm_count
                >= cfg.A4_RELOCK_CONFIRM_FRAMES
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
                "divider_y_mm", cfg.DIVIDER_Y_MM
            )
            observed_slope = candidate.get(
                "divider_slope_mm", 0.0
            )
            alpha = (
                1.0
                if not self.divider_detected
                else cfg.DIVIDER_TRACK_ALPHA
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
            self.valid_frames >= cfg.A4_LOCK_REQUIRED_FRAMES
        )
        return self.state()

    def freeze(self):
        """Make the first locked calibration immutable until process restart."""
        if self.continuous:
            return self.state()
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
            "lock_spread_px": self.lock_spread_px,
            "calibration_generation": self.calibration_generation,
            "relock_required": self.relock_required,
        }
