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

    edge_margin = cfg.A4_MIN_IMAGE_EDGE_MARGIN_PX
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
    if min(width, height) < cfg.A4_MIN_SIDE_PX or height <= 1.0:
        return _reject(diagnostics, "side")
    ratio = width / height
    if (
        ratio < cfg.A4_MIN_WIDTH_HEIGHT_RATIO
        or ratio > cfg.A4_MAX_WIDTH_HEIGHT_RATIO
    ):
        return _reject(diagnostics, "aspect")

    frame_area = float(gray_image.width() * gray_image.height())
    area_ratio = _polygon_area(corners) / frame_area
    if (
        area_ratio < cfg.A4_MIN_FRAME_AREA_RATIO
        or area_ratio > cfg.A4_MAX_FRAME_AREA_RATIO
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
    if center_offset > cfg.A4_MAX_CENTER_OFFSET_RATIO:
        return _reject(diagnostics, "center")

    inside_gray = _inside_gray(gray_image, corners)
    if inside_gray > cfg.A4_MAX_INSIDE_GRAY:
        return _reject(diagnostics, "brightness")

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
        "source": source,
        "magnitude": float(magnitude),
        "orientation": orientation,
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
            candidate = _score_candidate(
                gray_image,
                blob.min_corners(),
                float(blob.pixels()),
                "dark_blob",
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
    """Adaptive smoothing that follows motion but suppresses edge jitter."""

    def __init__(self):
        self.corners_px = None
        self.confidence = 0.0
        self.source = ""
        self.orientation = ""
        self.valid_frames = 0
        self.missed_frames = 0
        self.motion_px = 0.0
        self.locked = False

    def update(self, candidate):
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
            elif self.missed_frames > cfg.A4_HOLD_MISSED_FRAMES:
                self.corners_px = None
                self.valid_frames = 0
                self.locked = False
                self.confidence = 0.0
                self.source = ""
                self.orientation = ""
                self.motion_px = 0.0
            return self.state()

        detected = candidate["corners_px"]
        self.missed_frames = 0
        if self.corners_px is None:
            self.corners_px = [
                (float(point[0]), float(point[1]))
                for point in detected
            ]
            self.motion_px = 0.0
            self.valid_frames = 1
        else:
            distances = [
                _distance(current, observed)
                for current, observed in zip(
                    self.corners_px, detected
                )
            ]
            self.motion_px = max(distances)
            if self.motion_px > cfg.A4_RESET_MOTION_PX:
                self.corners_px = [
                    (float(point[0]), float(point[1]))
                    for point in detected
                ]
                self.valid_frames = 1
            else:
                alpha = (
                    cfg.A4_FAST_SMOOTH_ALPHA
                    if self.motion_px >= cfg.A4_FAST_MOTION_PX
                    else cfg.A4_SLOW_SMOOTH_ALPHA
                )
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

        self.confidence = candidate["confidence"]
        self.source = candidate["source"]
        self.orientation = candidate.get("orientation", "")
        self.locked = (
            self.valid_frames >= cfg.A4_LOCK_REQUIRED_FRAMES
        )
        return self.state()

    def state(self):
        return {
            "corners_px": self.corners_px,
            "locked": self.locked,
            "confidence": self.confidence,
            "source": self.source,
            "orientation": self.orientation,
            "valid_frames": self.valid_frames,
            "missed_frames": self.missed_frames,
            "motion_px": self.motion_px,
        }
