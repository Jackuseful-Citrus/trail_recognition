"""Pure-Python geometry and initial-piece tracking for the A4 puzzle.

This module deliberately avoids OpenCV, NumPy, dataclasses, and CPython-only
features so the same planner runs unchanged on CanMV K230 MicroPython.
Coordinates passed to this module are always millimetres.
"""

import math

import puzzle_config as cfg
from puzzle_perf import PERF_STATS, ticks_diff, ticks_ms


EPS = 1e-7
GEOMETRY_COUNTERS = {
    "polygon_intersection_calls": 0,
    "aabb_reject_count": 0,
}
PLAN_DEBUG_STATE = None


def reset_geometry_counters():
    GEOMETRY_COUNTERS["polygon_intersection_calls"] = 0
    GEOMETRY_COUNTERS["aabb_reject_count"] = 0


def begin_plan_debug(planner, piece_count):
    """Start one low-frequency planner heartbeat session."""
    global PLAN_DEBUG_STATE
    if not getattr(cfg, "ENABLE_PLAN_DEBUG", False):
        PLAN_DEBUG_STATE = None
        return
    reset_geometry_counters()
    now = ticks_ms()
    PLAN_DEBUG_STATE = {
        "planner": str(planner),
        "stage": "dispatch",
        "started_ms": now,
        "last_report_ms": now,
        "piece_count": int(piece_count),
        "depth": 0,
        "states": 0,
        "expanded": 0,
        "nodes": 0,
        "work": 0,
        "best_score": None,
    }


def update_plan_debug(
    stage=None,
    depth=None,
    states=None,
    expanded=None,
    nodes=None,
    best_score=None,
):
    """Update heartbeat fields without reading the clock or printing."""
    state = PLAN_DEBUG_STATE
    if state is None:
        return
    if stage is not None:
        state["stage"] = str(stage)
    if depth is not None:
        state["depth"] = int(depth)
    if states is not None:
        state["states"] = int(states)
    if expanded is not None:
        state["expanded"] = int(expanded)
    if nodes is not None:
        state["nodes"] = int(nodes)
    if best_score is not None:
        state["best_score"] = float(best_score)


def plan_debug_heartbeat(force=False):
    """Print progress only when the configured wall-clock interval elapsed."""
    state = PLAN_DEBUG_STATE
    if state is None:
        return False
    now = ticks_ms()
    elapsed_since_report = ticks_diff(
        now, state["last_report_ms"]
    )
    interval_ms = max(
        250,
        int(getattr(cfg, "PLAN_DEBUG_INTERVAL_MS", 2000)),
    )
    if not force and elapsed_since_report < interval_ms:
        return False
    elapsed_ms = max(
        0, ticks_diff(now, state["started_ms"])
    )
    best_score = state["best_score"]
    best_text = (
        "na"
        if best_score is None
        else "{:.4f}".format(best_score)
    )
    print(
        "PLAN_DEBUG,planner={},stage={},elapsed_ms={},pieces={},"
        "depth={},states={},expanded={},nodes={},work={},"
        "best_score={},intersections={},aabb_rejects={}".format(
            state["planner"],
            state["stage"],
            elapsed_ms,
            state["piece_count"],
            state["depth"],
            state["states"],
            state["expanded"],
            state["nodes"],
            state["work"],
            best_text,
            GEOMETRY_COUNTERS["polygon_intersection_calls"],
            GEOMETRY_COUNTERS["aabb_reject_count"],
        )
    )
    state["last_report_ms"] = now
    return True


def end_plan_debug():
    """End the active heartbeat session without adding another log line."""
    global PLAN_DEBUG_STATE
    PLAN_DEBUG_STATE = None


class PieceObservation:
    """One detected physical puzzle piece."""

    __slots__ = (
        "piece_id",
        "contour_px",
        "polygon_mm",
        "centroid_mm",
        "area_mm2",
        "edge_lengths_mm",
        "interior_angles_deg",
        "current_orientation_deg",
        "confidence",
        "rotation_ambiguous",
        "centroid_fallback",
        "stable",
        "aabb_mm",
        "triangles_mm",
        "is_convex",
    )

    def __init__(
        self,
        piece_id,
        contour_px,
        polygon_mm,
        centroid_mm=None,
        area_mm2=None,
        edge_lengths_mm=None,
        interior_angles_deg=None,
        current_orientation_deg=None,
        confidence=0.0,
        rotation_ambiguous=None,
        centroid_fallback=False,
    ):
        self.piece_id = piece_id
        self.contour_px = contour_px or []
        self.polygon_mm = ensure_clockwise(polygon_mm)
        if not polygon_is_simple(self.polygon_mm):
            raise ValueError("piece polygon is not simple")
        self.aabb_mm = polygon_aabb(self.polygon_mm)
        self.is_convex = polygon_is_convex(self.polygon_mm)
        self.triangles_mm = triangulate_simple_polygon(
            self.polygon_mm
        )
        self.area_mm2 = (
            polygon_area(self.polygon_mm) if area_mm2 is None else float(area_mm2)
        )
        if centroid_mm is None:
            self.centroid_mm = polygon_centroid(self.polygon_mm)
        else:
            self.centroid_mm = (float(centroid_mm[0]), float(centroid_mm[1]))
        self.edge_lengths_mm = (
            edge_lengths(self.polygon_mm)
            if edge_lengths_mm is None
            else list(edge_lengths_mm)
        )
        self.interior_angles_deg = (
            interior_angles(self.polygon_mm)
            if interior_angles_deg is None
            else list(interior_angles_deg)
        )
        self.current_orientation_deg = (
            polygon_longest_edge_orientation(self.polygon_mm)
            if current_orientation_deg is None
            else float(current_orientation_deg)
        )
        self.confidence = float(confidence)
        self.rotation_ambiguous = (
            polygon_rotation_ambiguous(self.polygon_mm)
            if rotation_ambiguous is None
            else bool(rotation_ambiguous)
        )
        self.centroid_fallback = bool(centroid_fallback)
        self.stable = False


class PlanResult:
    """Planner output in physical A4 coordinates."""

    __slots__ = (
        "valid",
        "reason",
        "score",
        "operations",
        "target_polygons",
        "target_rect",
        "search_nodes",
        "mode",
        "max_vertex_error_mm",
        "fill_gap_mm2",
        "overlap_mm2",
        "outside_mm2",
        "seams",
        "plan_stats",
    )

    def __init__(
        self,
        valid=False,
        reason="NO VALID PLAN",
        score=None,
        operations=None,
        target_polygons=None,
        target_rect=None,
        search_nodes=0,
        mode="strict",
        max_vertex_error_mm=None,
        fill_gap_mm2=None,
        overlap_mm2=None,
        outside_mm2=None,
        seams=None,
        plan_stats=None,
    ):
        self.valid = bool(valid)
        self.reason = reason
        self.score = score
        self.operations = operations or []
        self.target_polygons = target_polygons or {}
        self.target_rect = target_rect
        self.search_nodes = int(search_nodes)
        self.mode = mode
        self.max_vertex_error_mm = max_vertex_error_mm
        self.fill_gap_mm2 = fill_gap_mm2
        self.overlap_mm2 = overlap_mm2
        self.outside_mm2 = outside_mm2
        self.seams = seams or []
        self.plan_stats = plan_stats or {}


def _point(p):
    return (float(p[0]), float(p[1]))


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (
        c[0] - a[0]
    )


def _distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


def _point_segment_distance(point, a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= EPS:
        return _distance(point, a)
    projection = (
        (point[0] - a[0]) * dx
        + (point[1] - a[1]) * dy
    ) / length_squared
    projection = max(0.0, min(1.0, projection))
    closest = (
        a[0] + projection * dx,
        a[1] + projection * dy,
    )
    return _distance(point, closest)


def polygon_signed_area(vertices):
    """Return the shoelace signed area (positive for CCW in x-right/y-up)."""
    n = len(vertices)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def polygon_area(vertices):
    """Return absolute polygon area."""
    return abs(polygon_signed_area(vertices))


def polygon_centroid(vertices):
    """Return the area-weighted polygon centroid.

    Raises ``ValueError`` for a degenerate polygon instead of silently using a
    bounding-box centre.
    """
    n = len(vertices)
    if n < 3:
        raise ValueError("polygon centroid needs at least three vertices")
    twice_area = 0.0
    sx = 0.0
    sy = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        twice_area += cross
        sx += (x0 + x1) * cross
        sy += (y0 + y1) * cross
    if abs(twice_area) <= EPS:
        raise ValueError("degenerate polygon has no area-weighted centroid")
    return (sx / (3.0 * twice_area), sy / (3.0 * twice_area))


def ensure_clockwise(vertices):
    """Return float vertices in clockwise order in a mathematical x/y frame."""
    result = [_point(p) for p in vertices]
    if polygon_signed_area(result) > 0.0:
        result.reverse()
    return result


def normalize_angle_deg(angle):
    """Normalize an angle to [-180, 180)."""
    value = (float(angle) + 180.0) % 360.0 - 180.0
    if value >= 180.0 - EPS:
        value = -180.0
    return value


def _angle_difference_deg(a, b, period=360.0):
    half = 0.5 * period
    return abs((a - b + half) % period - half)


def _merge_adjacent_polygon_vertices(points, index):
    """Split one short cyclic edge equally between its neighbours."""
    count = len(points)
    following = (index + 1) % count
    first = points[index]
    second = points[following]
    replacement = (
        0.5 * (first[0] + second[0]),
        0.5 * (first[1] + second[1]),
    )

    if following == 0:
        return [replacement] + points[1:index]
    return points[:index] + [replacement] + points[following + 1 :]


def _vertex_cleanup_candidate_is_safe(
    before,
    after,
    min_vertices,
    max_area_change_ratio,
):
    if len(after) < min_vertices:
        return False
    if any(
        _distance(after[index], after[(index + 1) % len(after)])
        <= EPS
        for index in range(len(after))
    ):
        return False
    before_signed = polygon_signed_area(before)
    after_signed = polygon_signed_area(after)
    if abs(after_signed) <= EPS or before_signed * after_signed <= 0.0:
        return False
    area_change_ratio = abs(
        abs(after_signed) - abs(before_signed)
    ) / max(abs(before_signed), EPS)
    if area_change_ratio > max_area_change_ratio:
        return False
    return polygon_is_simple(after)


def remove_near_collinear_vertices(
    vertices,
    tolerance_deg=None,
    min_edge_mm=None,
    min_vertices=3,
    max_collinear_offset_mm=None,
    max_area_change_ratio=None,
    max_passes=None,
):
    """Robustly merge duplicate corners and remove shallow line artefacts.

    A short cyclic edge is replaced by one fitted corner, rather than deleting
    whichever endpoint happens to be visited first.  A vertex is considered
    collinear only when both its angle and its physical offset from the
    neighbour-to-neighbour segment are small.  Every edit preserves winding,
    simplicity and most of the measured area.
    """
    if tolerance_deg is None:
        tolerance_deg = cfg.VERTEX_COLLINEAR_ANGLE_TOLERANCE_DEG
    if min_edge_mm is None:
        min_edge_mm = cfg.VERTEX_MERGE_DISTANCE_MM
    if max_collinear_offset_mm is None:
        max_collinear_offset_mm = (
            cfg.VERTEX_COLLINEAR_MAX_OFFSET_MM
        )
    if max_area_change_ratio is None:
        max_area_change_ratio = (
            cfg.VERTEX_CLEANUP_MAX_AREA_CHANGE_RATIO
        )
    if max_passes is None:
        max_passes = cfg.VERTEX_CLEANUP_MAX_PASSES
    points = [_point(p) for p in vertices]
    if len(points) <= min_vertices:
        return ensure_clockwise(points)

    for _ in range(max(0, int(max_passes))):
        if len(points) <= min_vertices:
            break
        changed = False
        n = len(points)

        # The geometric tie-breaker makes the choice independent of the
        # contour's cyclic start point and traversal direction.
        close_edges = []
        for index in range(n):
            following = (index + 1) % n
            length = _distance(points[index], points[following])
            if length < min_edge_mm:
                midpoint = (
                    0.5
                    * (points[index][0] + points[following][0]),
                    0.5
                    * (points[index][1] + points[following][1]),
                )
                close_edges.append(
                    (length, midpoint[1], midpoint[0], index)
                )
        close_edges.sort()
        for _, _, _, index in close_edges:
            candidate = _merge_adjacent_polygon_vertices(
                points,
                index,
            )
            if _vertex_cleanup_candidate_is_safe(
                points,
                candidate,
                min_vertices,
                max_area_change_ratio,
            ):
                points = candidate
                changed = True
                break
        if changed:
            continue

        collinear = []
        for i in range(n):
            prev = points[(i - 1) % n]
            cur = points[i]
            nxt = points[(i + 1) % n]
            v1 = (prev[0] - cur[0], prev[1] - cur[1])
            v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
            l1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
            l2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
            if l1 <= EPS or l2 <= EPS:
                continue
            cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
            angle = math.degrees(math.acos(cosine))
            angle_error = abs(180.0 - angle)
            offset = _point_segment_distance(cur, prev, nxt)
            if (
                angle_error <= tolerance_deg
                and offset <= max_collinear_offset_mm
            ):
                collinear.append(
                    (
                        offset,
                        angle_error,
                        cur[1],
                        cur[0],
                        i,
                    )
                )
        collinear.sort()
        for _, _, _, _, index in collinear:
            candidate = points[:index] + points[index + 1 :]
            if _vertex_cleanup_candidate_is_safe(
                points,
                candidate,
                min_vertices,
                max_area_change_ratio,
            ):
                points = candidate
                changed = True
                break
        if not changed:
            break
    return ensure_clockwise(points)


def edge_lengths(vertices):
    """Return cyclic edge lengths."""
    return [
        _distance(vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    ]


def interior_angles(vertices):
    """Return 0..360 degree interior angles for a simple polygon."""
    result = []
    n = len(vertices)
    orientation = (
        1.0 if polygon_signed_area(vertices) >= 0.0 else -1.0
    )
    for i in range(n):
        prev = vertices[(i - 1) % n]
        cur = vertices[i]
        nxt = vertices[(i + 1) % n]
        ax, ay = prev[0] - cur[0], prev[1] - cur[1]
        bx, by = nxt[0] - cur[0], nxt[1] - cur[1]
        la = math.sqrt(ax * ax + ay * ay)
        lb = math.sqrt(bx * bx + by * by)
        if la <= EPS or lb <= EPS:
            result.append(0.0)
            continue
        cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        angle = math.degrees(math.acos(cosine))
        cross = ax * by - ay * bx
        if cross * orientation > EPS:
            angle = 360.0 - angle
        result.append(angle)
    return result


def polygon_longest_edge_orientation(vertices):
    """Return the longest-edge axis angle in [-90, 90)."""
    lengths = edge_lengths(vertices)
    index = max(range(len(lengths)), key=lambda i: lengths[i])
    p0 = vertices[index]
    p1 = vertices[(index + 1) % len(vertices)]
    angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    return (angle + 90.0) % 180.0 - 90.0


def _canonical_cycle(values):
    if not values:
        return ()
    candidates = []
    n = len(values)
    for sequence in (values, list(reversed(values))):
        for offset in range(n):
            candidates.append(tuple(sequence[offset:] + sequence[:offset]))
    return min(candidates)


def polygon_shape_signature(vertices):
    """Return a rotation/reflection-invariant shape signature.

    The tuple contains vertex count, normalized edge sequence, interior angles,
    and a compactness value. It is suitable for matching, not identity hashing.
    """
    lengths = edge_lengths(vertices)
    perimeter = sum(lengths)
    if perimeter <= EPS:
        normalized = [0.0 for _ in lengths]
    else:
        normalized = [round(value / perimeter, 4) for value in lengths]
    angles = [round(value / 180.0, 4) for value in interior_angles(vertices)]
    paired = [
        (normalized[i], angles[(i + 1) % len(angles)])
        for i in range(len(normalized))
    ]
    canonical = _canonical_cycle(paired)
    compactness = 0.0
    if perimeter > EPS:
        compactness = 4.0 * math.pi * polygon_area(vertices) / (perimeter * perimeter)
    flattened = []
    for edge_value, angle_value in canonical:
        flattened.extend((edge_value, angle_value))
    return tuple([len(vertices), round(compactness, 4)] + flattened)


def polygon_rotation_ambiguous(vertices, tolerance=0.018):
    """Detect repeated cyclic edge/angle structure (equivalent rotations)."""
    return polygon_symmetry_period_deg(vertices, tolerance) < 360.0 - EPS


def polygon_symmetry_period_deg(vertices, tolerance=0.018):
    """Return the smallest equivalent rotational period inferred from shape."""
    lengths = edge_lengths(vertices)
    perimeter = max(EPS, sum(lengths))
    angles = interior_angles(vertices)
    n = len(vertices)
    features = [(lengths[i] / perimeter, angles[(i + 1) % n] / 180.0) for i in range(n)]
    for shift in range(1, n):
        same = True
        for i in range(n):
            a = features[i]
            b = features[(i + shift) % n]
            if abs(a[0] - b[0]) > tolerance or abs(a[1] - b[1]) > tolerance:
                same = False
                break
        if same:
            return 360.0 * shift / n
    return 360.0


def transform_point(point, transform):
    cosine, sine, tx, ty = transform[:4]
    x, y = point
    return (cosine * x - sine * y + tx, sine * x + cosine * y + ty)


def transform_polygon(vertices, transform):
    """Apply a rigid transform to every polygon vertex."""
    return [transform_point(point, transform) for point in vertices]


def compose_transforms(after, before):
    """Return the transform equivalent to ``after(before(point))``."""
    ca, sa, tax, tay = after[:4]
    cb, sb, tbx, tby = before[:4]
    cosine = ca * cb - sa * sb
    sine = sa * cb + ca * sb
    tx = ca * tbx - sa * tby + tax
    ty = sa * tbx + ca * tby + tay
    angle = normalize_angle_deg(math.degrees(math.atan2(sine, cosine)))
    return (cosine, sine, tx, ty, angle)


def polygon_aabb(polygon):
    """Return ``(min_x, min_y, max_x, max_y)`` in millimetres."""
    return (
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    )


def _aabb_positive_overlap(aabb_a, aabb_b):
    epsilon = cfg.GEOMETRY_EPSILON_MM
    return (
        min(aabb_a[2], aabb_b[2])
        - max(aabb_a[0], aabb_b[0])
        > epsilon
        and min(aabb_a[3], aabb_b[3])
        - max(aabb_a[1], aabb_b[1])
        > epsilon
    )


def polygon_is_convex(points):
    """Return whether a simple polygon has one consistent turn direction."""
    sign = 0
    for index in range(len(points)):
        value = _cross(
            points[index],
            points[(index + 1) % len(points)],
            points[(index + 2) % len(points)],
        )
        if abs(value) <= EPS:
            continue
        current = 1 if value > 0.0 else -1
        if sign and current != sign:
            return False
        sign = current
    return sign != 0


def _point_on_segment(point, a, b):
    return (
        abs(_cross(a, b, point)) <= EPS
        and min(a[0], b[0]) - EPS
        <= point[0]
        <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS
        <= point[1]
        <= max(a[1], b[1]) + EPS
    )


def _segments_intersect(a0, a1, b0, b1):
    c0 = _cross(a0, a1, b0)
    c1 = _cross(a0, a1, b1)
    c2 = _cross(b0, b1, a0)
    c3 = _cross(b0, b1, a1)
    if (
        ((c0 > EPS and c1 < -EPS) or (c0 < -EPS and c1 > EPS))
        and ((c2 > EPS and c3 < -EPS) or (c2 < -EPS and c3 > EPS))
    ):
        return True
    return (
        (abs(c0) <= EPS and _point_on_segment(b0, a0, a1))
        or (abs(c1) <= EPS and _point_on_segment(b1, a0, a1))
        or (abs(c2) <= EPS and _point_on_segment(a0, b0, b1))
        or (abs(c3) <= EPS and _point_on_segment(a1, b0, b1))
    )


def polygon_is_simple(points):
    """Reject repeated, degenerate, or self-intersecting polygon rings."""
    if len(points) < 3 or polygon_area(points) <= EPS:
        return False
    for index, point in enumerate(points):
        if _distance(
            point, points[(index + 1) % len(points)]
        ) <= EPS:
            return False
    count = len(points)
    for first in range(count):
        a0 = points[first]
        a1 = points[(first + 1) % count]
        for second in range(first + 1, count):
            if (
                second == first
                or second == (first + 1) % count
                or first == (second + 1) % count
            ):
                continue
            b0 = points[second]
            b1 = points[(second + 1) % count]
            if _segments_intersect(a0, a1, b0, b1):
                return False
    return True


def _point_in_triangle(point, triangle):
    a, b, c = triangle
    c0 = _cross(a, b, point)
    c1 = _cross(b, c, point)
    c2 = _cross(c, a, point)
    has_negative = c0 < -EPS or c1 < -EPS or c2 < -EPS
    has_positive = c0 > EPS or c1 > EPS or c2 > EPS
    return not (has_negative and has_positive)


def triangulate_simple_polygon(points):
    """Triangulate a 3..N vertex simple polygon using ear clipping."""
    polygon = [_point(point) for point in points]
    if not polygon_is_simple(polygon):
        raise ValueError("cannot triangulate non-simple polygon")
    orientation = 1.0 if polygon_signed_area(polygon) > 0.0 else -1.0
    remaining = list(range(len(polygon)))
    triangles = []
    guard = 0
    while len(remaining) > 3:
        guard += 1
        if guard > len(polygon) * len(polygon):
            raise ValueError("ear clipping did not converge")
        ear_found = False
        for position, current in enumerate(remaining):
            previous = remaining[(position - 1) % len(remaining)]
            following = remaining[(position + 1) % len(remaining)]
            triangle = (
                polygon[previous],
                polygon[current],
                polygon[following],
            )
            if (
                orientation
                * _cross(
                    triangle[0], triangle[1], triangle[2]
                )
                <= EPS
            ):
                continue
            contains_vertex = False
            for other in remaining:
                if other in (previous, current, following):
                    continue
                if _point_in_triangle(polygon[other], triangle):
                    contains_vertex = True
                    break
            if contains_vertex:
                continue
            if polygon_area(triangle) > EPS:
                triangles.append(list(triangle))
            del remaining[position]
            ear_found = True
            break
        if not ear_found:
            raise ValueError("polygon has no valid ear")
    final_triangle = [polygon[index] for index in remaining]
    if polygon_area(final_triangle) > EPS:
        triangles.append(final_triangle)
    triangulated_area = sum(
        polygon_area(triangle) for triangle in triangles
    )
    source_area = polygon_area(polygon)
    if abs(triangulated_area - source_area) > max(
        1e-5, source_area * 1e-6
    ):
        raise ValueError("triangulation area mismatch")
    return triangles


def _line_intersection(segment_start, segment_end, clip_start, clip_end):
    sx = segment_end[0] - segment_start[0]
    sy = segment_end[1] - segment_start[1]
    cx = clip_end[0] - clip_start[0]
    cy = clip_end[1] - clip_start[1]
    denominator = sx * cy - sy * cx
    if abs(denominator) <= EPS:
        return segment_end
    dx = clip_start[0] - segment_start[0]
    dy = clip_start[1] - segment_start[1]
    t = (dx * cy - dy * cx) / denominator
    return (segment_start[0] + t * sx, segment_start[1] + t * sy)


def convex_polygon_intersection(poly_a, poly_b):
    """Return the convex intersection polygon using Sutherland-Hodgman."""
    if not _aabb_positive_overlap(
        polygon_aabb(poly_a), polygon_aabb(poly_b)
    ):
        GEOMETRY_COUNTERS["aabb_reject_count"] += 1
        PERF_STATS.increment("aabb_reject_count")
        return []
    GEOMETRY_COUNTERS["polygon_intersection_calls"] += 1
    PERF_STATS.increment("polygon_intersection_calls")
    output = [_point(p) for p in poly_a]
    clip = [_point(p) for p in poly_b]
    if len(output) < 3 or len(clip) < 3:
        return []
    clip_sign = 1.0 if polygon_signed_area(clip) >= 0.0 else -1.0
    for i in range(len(clip)):
        cp1 = clip[i]
        cp2 = clip[(i + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = clip_sign * _cross(cp1, cp2, previous) >= -EPS
        for current in input_points:
            current_inside = clip_sign * _cross(cp1, cp2, current) >= -EPS
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, cp1, cp2))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, cp1, cp2))
            previous = current
            previous_inside = current_inside
    return output


def polygon_overlap_area(
    poly_a,
    poly_b,
    triangles_a=None,
    triangles_b=None,
):
    """Return overlap area for convex or concave simple polygons."""
    if not _aabb_positive_overlap(
        polygon_aabb(poly_a), polygon_aabb(poly_b)
    ):
        GEOMETRY_COUNTERS["aabb_reject_count"] += 1
        PERF_STATS.increment("aabb_reject_count")
        return 0.0
    if polygon_is_convex(poly_a) and polygon_is_convex(poly_b):
        intersection = convex_polygon_intersection(
            poly_a, poly_b
        )
        return (
            polygon_area(intersection)
            if len(intersection) >= 3
            else 0.0
        )
    if triangles_a is None:
        triangles_a = triangulate_simple_polygon(poly_a)
    if triangles_b is None:
        triangles_b = triangulate_simple_polygon(poly_b)
    overlap = 0.0
    for triangle_a in triangles_a:
        for triangle_b in triangles_b:
            intersection = convex_polygon_intersection(
                triangle_a, triangle_b
            )
            if len(intersection) >= 3:
                overlap += polygon_area(intersection)
    return overlap


def convex_hull(points):
    """Monotonic-chain convex hull."""
    unique = sorted(set((_point(p) for p in points)))
    if len(unique) <= 1:
        return unique

    def build(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2 and _cross(result[-2], result[-1], point) <= EPS:
                result.pop()
            result.append(point)
        return result

    lower = build(unique)
    upper = build(reversed(unique))
    return lower[:-1] + upper[:-1]


def _rotate_point(point, angle_rad):
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return (
        cosine * point[0] - sine * point[1],
        sine * point[0] + cosine * point[1],
    )


def minimum_area_rectangle(points):
    """Return an edge-aligned minimum-area rectangle descriptor."""
    hull = convex_hull(points)
    if len(hull) < 3:
        return None
    best = None
    for i in range(len(hull)):
        p0 = hull[i]
        p1 = hull[(i + 1) % len(hull)]
        edge_angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        rotated = [_rotate_point(point, -edge_angle) for point in hull]
        xs = [point[0] for point in rotated]
        ys = [point[1] for point in rotated]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        candidate = {
            "angle_deg": normalize_angle_deg(math.degrees(edge_angle)),
            "width": width,
            "height": height,
            "area": area,
            "bounds_rotated": (min_x, min_y, max_x, max_y),
        }
        if best is None or area < best["area"]:
            best = candidate
    return best


def _identity_transform():
    return (1.0, 0.0, 0.0, 0.0, 0.0)


def _choose_smallest_equivalent_rotation(rotation, polygon):
    period = polygon_symmetry_period_deg(polygon)
    if period >= 360.0 - EPS:
        return normalize_angle_deg(rotation)
    candidates = []
    steps = max(1, int(round(360.0 / period)))
    for index in range(steps):
        candidates.append(
            normalize_angle_deg(rotation + index * period)
        )
    return min(candidates, key=lambda value: abs(value))


def _signature_cost(signature_a, signature_b):
    if not signature_a or not signature_b or signature_a[0] != signature_b[0]:
        return 1e9
    size = min(len(signature_a), len(signature_b))
    if size <= 2:
        return 1e9
    total = abs(signature_a[1] - signature_b[1])
    for i in range(2, size):
        total += abs(signature_a[i] - signature_b[i])
    return total / (size - 1)


def match_piece_across_frames(previous, current):
    """Return ``(matches, cost)`` using shape, area, orientation, and distance."""
    previous_signature = polygon_shape_signature(previous.polygon_mm)
    current_signature = polygon_shape_signature(current.polygon_mm)
    vertex_count_delta = abs(
        len(previous.polygon_mm) - len(current.polygon_mm)
    )
    if vertex_count_delta == 0:
        shape_cost = _signature_cost(
            previous_signature, current_signature
        )
    elif vertex_count_delta <= cfg.TRACK_MAX_VERTEX_COUNT_DELTA:
        # Edge-by-edge signatures cannot be compared when one noisy corner was
        # split or merged. Compactness, area and rectified-A4 position remain
        # useful and do not require equal vertex counts.
        shape_cost = (
            abs(previous_signature[1] - current_signature[1])
            + cfg.TRACK_VERTEX_MISMATCH_SHAPE_PENALTY
            * vertex_count_delta
        )
    else:
        shape_cost = 1e9
    area_scale = max(EPS, previous.area_mm2, current.area_mm2)
    area_cost = abs(previous.area_mm2 - current.area_mm2) / area_scale
    distance = _distance(previous.centroid_mm, current.centroid_mm)
    distance_cost = min(1.5, distance / max(EPS, cfg.TRACK_MAX_DISTANCE_MM))
    angle_period = min(
        polygon_symmetry_period_deg(previous.polygon_mm),
        polygon_symmetry_period_deg(current.polygon_mm),
    )
    angle_cost = _angle_difference_deg(
        previous.current_orientation_deg,
        current.current_orientation_deg,
        period=angle_period,
    ) / 180.0
    cost = 0.55 * shape_cost + 0.20 * area_cost + 0.20 * distance_cost + 0.05 * angle_cost
    mismatch_geometry_ok = (
        vertex_count_delta == 0
        or (
            distance
            <= cfg.TRACK_VERTEX_MISMATCH_MAX_DISTANCE_MM
            and area_cost
            <= cfg.TRACK_VERTEX_MISMATCH_MAX_AREA_RATIO
        )
    )
    matches = (
        vertex_count_delta <= cfg.TRACK_MAX_VERTEX_COUNT_DELTA
        and distance <= cfg.TRACK_MAX_DISTANCE_MM
        and mismatch_geometry_ok
        and cost <= cfg.TRACK_SHAPE_COST_LIMIT
    )
    return matches, cost


class _Track:
    __slots__ = (
        "piece_id",
        "last",
        "history",
        "samples",
        "missed",
        "stable",
    )

    def __init__(self, piece_id, observation):
        self.piece_id = piece_id
        self.last = observation
        self.history = [
            (observation.centroid_mm, observation.current_orientation_deg)
        ]
        self.samples = [observation]
        self.missed = 0
        self.stable = False


def _representative_track_observation(track):
    """Choose a temporally supported polygon for planning and display."""
    if not track.samples:
        return track.last
    counts = {}
    for observation in track.samples:
        vertex_count = len(observation.polygon_mm)
        counts[vertex_count] = counts.get(vertex_count, 0) + 1
    modal_count = min(
        counts,
        key=lambda value: (-counts[value], value),
    )
    candidates = [
        observation
        for observation in track.samples
        if len(observation.polygon_mm) == modal_count
    ]
    areas = sorted(
        observation.area_mm2 for observation in candidates
    )
    median_area = areas[len(areas) // 2]
    candidates.sort(
        key=lambda observation: (
            abs(observation.area_mm2 - median_area),
            -observation.confidence,
        )
    )
    return candidates[0]


class PieceTracker:
    """Maintain shape-aware stable IDs during initial recognition."""

    def __init__(self, expected_count=None):
        self.expected_count = (
            None
            if expected_count is None
            else max(1, int(expected_count))
        )
        self.reset()

    def reset(self, expected_count=None):
        if expected_count is not None:
            self.expected_count = max(1, int(expected_count))
        self.tracks = []
        self.next_id = 1
        self.last_count = None

    def _new_track(self, observation):
        limit = (
            self.expected_count
            if self.expected_count is not None
            else cfg.MAX_PIECE_COUNT
        )
        reusable = [
            track
            for track in self.tracks
            if track.missed > cfg.TRACK_MAX_MISSED_FRAMES
        ]
        if reusable:
            track = max(reusable, key=lambda item: item.missed)
            observation.piece_id = track.piece_id
            track.last = observation
            track.history = [
                (
                    observation.centroid_mm,
                    observation.current_orientation_deg,
                )
            ]
            track.samples = [observation]
            track.missed = 0
            track.stable = False
            return track
        active_count = sum(
            1
            for track in self.tracks
            if track.missed <= cfg.TRACK_MAX_MISSED_FRAMES
        )
        if active_count >= limit:
            return None
        piece_id = "P{}".format(self.next_id)
        self.next_id += 1
        observation.piece_id = piece_id
        track = _Track(piece_id, observation)
        self.tracks.append(track)
        return track

    def update(self, observations):
        self.last_count = len(observations)
        active = [
            track
            for track in self.tracks
            if track.missed <= cfg.TRACK_MAX_MISSED_FRAMES
        ]
        candidates = []
        for track_index, track in enumerate(active):
            for obs_index, observation in enumerate(observations):
                matches, cost = match_piece_across_frames(track.last, observation)
                if matches:
                    candidates.append((cost, track_index, obs_index))
        candidates.sort(key=lambda item: item[0])

        used_tracks = set()
        used_observations = set()
        assignments = []
        for _, track_index, obs_index in candidates:
            if track_index in used_tracks or obs_index in used_observations:
                continue
            used_tracks.add(track_index)
            used_observations.add(obs_index)
            assignments.append((active[track_index], observations[obs_index]))

        for track in active:
            if track not in [pair[0] for pair in assignments]:
                track.missed += 1

        stable_representatives = {}
        for track, observation in assignments:
            observation.piece_id = track.piece_id
            moved = _distance(track.last.centroid_mm, observation.centroid_mm) > cfg.CENTER_STABLE_TOLERANCE_MM
            angle_period = min(
                polygon_symmetry_period_deg(track.last.polygon_mm),
                polygon_symmetry_period_deg(observation.polygon_mm),
            )
            rotated = _angle_difference_deg(
                track.last.current_orientation_deg,
                observation.current_orientation_deg,
                period=angle_period,
            ) > cfg.ANGLE_STABLE_TOLERANCE_DEG
            if moved or rotated:
                track.history = []
                track.samples = []
                track.stable = False
            track.history.append(
                (observation.centroid_mm, observation.current_orientation_deg)
            )
            track.samples.append(observation)
            if len(track.history) > cfg.STABLE_WINDOW_FRAMES:
                track.history.pop(0)
            if len(track.samples) > cfg.STABLE_WINDOW_FRAMES:
                track.samples.pop(0)
            track.last = observation
            track.missed = 0

            if len(track.history) >= cfg.REQUIRED_STABLE_FRAMES:
                centers = [sample[0] for sample in track.history]
                base_center = centers[0]
                center_spread = max(_distance(base_center, center) for center in centers)
                base_angle = track.history[0][1]
                angle_spread = max(
                    _angle_difference_deg(base_angle, sample[1], period=angle_period)
                    for sample in track.history
                )
                areas = sorted(
                    sample.area_mm2 for sample in track.samples
                )
                median_area = areas[len(areas) // 2]
                area_spread = max(
                    abs(area - median_area)
                    / max(EPS, median_area)
                    for area in areas
                )
                track.stable = (
                    center_spread <= cfg.CENTER_STABLE_TOLERANCE_MM
                    and angle_spread <= cfg.ANGLE_STABLE_TOLERANCE_DEG
                    and area_spread
                    <= cfg.AREA_STABLE_TOLERANCE_RATIO
                )
            observation.stable = track.stable
            if track.stable:
                representative = _representative_track_observation(
                    track
                )
                representative.piece_id = track.piece_id
                representative.stable = True
                stable_representatives[track.piece_id] = (
                    representative
                )

        for obs_index, observation in enumerate(observations):
            if obs_index not in used_observations:
                track = self._new_track(observation)
                if track is not None:
                    used_observations.add(obs_index)

        tracked_observations = [
            observation
            for observation in observations
            if observation.piece_id
        ]
        for index, observation in enumerate(tracked_observations):
            representative = stable_representatives.get(
                observation.piece_id
            )
            if representative is not None:
                tracked_observations[index] = representative
        observations = tracked_observations
        observations.sort(
            key=lambda observation: int(observation.piece_id[1:])
        )
        count_ready = (
            len(observations) == self.expected_count
            if self.expected_count is not None
            else (
                cfg.MIN_PIECE_COUNT
                <= len(observations)
                <= cfg.MAX_PIECE_COUNT
            )
        )
        all_stable = (
            count_ready
            and all(observation.stable for observation in observations)
        )
        return observations, all_stable
