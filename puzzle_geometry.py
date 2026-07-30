"""Pure-Python geometry, tracking, and rectangle assembly for the A4 puzzle.

This module deliberately avoids OpenCV, NumPy, dataclasses, and CPython-only
features so the same planner runs unchanged on CanMV K230 MicroPython.
Coordinates passed to this module are always millimetres.
"""

import math
import os

import puzzle_config as cfg
from puzzle_perf import PERF_STATS, ticks_diff, ticks_ms


EPS = 1e-7
GEOMETRY_COUNTERS = {
    "polygon_intersection_calls": 0,
    "aabb_reject_count": 0,
}


def reset_geometry_counters():
    GEOMETRY_COUNTERS["polygon_intersection_calls"] = 0
    GEOMETRY_COUNTERS["aabb_reject_count"] = 0


def geometry_counters_snapshot():
    return dict(GEOMETRY_COUNTERS)


def _geometry_exitpoint(counter, interval=32):
    """Keep long planner searches interruptible from CanMV IDE."""
    if counter % interval != 0:
        return
    exitpoint = getattr(os, "exitpoint", None)
    if exitpoint is not None:
        exitpoint()


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


def polygon_orientation(vertices):
    """Return ``CCW``, ``CW``, or ``DEGENERATE``."""
    area = polygon_signed_area(vertices)
    if area > EPS:
        return "CCW"
    if area < -EPS:
        return "CW"
    return "DEGENERATE"


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


def _infinite_segment_line_intersection(a0, a1, b0, b1):
    """Intersect the infinite lines through two directed segments."""
    adx = a1[0] - a0[0]
    ady = a1[1] - a0[1]
    bdx = b1[0] - b0[0]
    bdy = b1[1] - b0[1]
    denominator = adx * bdy - ady * bdx
    if abs(denominator) <= EPS:
        return None
    dx = b0[0] - a0[0]
    dy = b0[1] - a0[1]
    scale = (dx * bdy - dy * bdx) / denominator
    return (a0[0] + scale * adx, a0[1] + scale * ady)


def _merge_adjacent_polygon_vertices(points, index, max_extrapolation_mm):
    """Replace one short cyclic edge by one fitted corner."""
    count = len(points)
    following = (index + 1) % count
    previous = (index - 1) % count
    after = (following + 1) % count
    first = points[index]
    second = points[following]
    midpoint = (
        0.5 * (first[0] + second[0]),
        0.5 * (first[1] + second[1]),
    )

    # A small chamfer caused by a missing/reflective corner is best restored by
    # intersecting its two neighbouring long-edge directions.  Parallel lines
    # describe a duplicate point on one straight edge, for which the midpoint
    # is the stable answer.  The distance cap prevents noisy shallow lines from
    # producing a far-away corner.
    replacement = _infinite_segment_line_intersection(
        points[previous],
        first,
        second,
        points[after],
    )
    if (
        replacement is None
        or _distance(replacement, first) > max_extrapolation_mm
        or _distance(replacement, second) > max_extrapolation_mm
    ):
        replacement = midpoint

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
    max_extrapolation_mm = (
        cfg.VERTEX_MERGE_MAX_EXTRAPOLATION_MM
    )
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
                max_extrapolation_mm,
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


def rigid_transform_from_edge_pair(edge_a, edge_b):
    """Return a rigid transform mapping ``edge_b`` onto reversed ``edge_a``.

    The transform tuple is ``(cos, sin, tx, ty, angle_deg)``. With unequal edge
    lengths the midpoints coincide and endpoint errors are shared equally.
    """
    a0, a1 = _point(edge_a[0]), _point(edge_a[1])
    b0, b1 = _point(edge_b[0]), _point(edge_b[1])
    target_angle = math.atan2(a0[1] - a1[1], a0[0] - a1[0])
    source_angle = math.atan2(b1[1] - b0[1], b1[0] - b0[0])
    angle = target_angle - source_angle
    cosine = math.cos(angle)
    sine = math.sin(angle)
    mbx = 0.5 * (b0[0] + b1[0])
    mby = 0.5 * (b0[1] + b1[1])
    max_ = 0.5 * (a0[0] + a1[0])
    may = 0.5 * (a0[1] + a1[1])
    tx = max_ - (cosine * mbx - sine * mby)
    ty = may - (sine * mbx + cosine * mby)
    return (cosine, sine, tx, ty, normalize_angle_deg(math.degrees(angle)))


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


def _inverse_transform(transform):
    cosine, sine, tx, ty, angle = transform
    return (
        cosine,
        -sine,
        -(cosine * tx + sine * ty),
        -(-sine * tx + cosine * ty),
        normalize_angle_deg(-angle),
    )


def polygon_aabb(polygon):
    """Return ``(min_x, min_y, max_x, max_y)`` in millimetres."""
    return (
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    )


class EdgeDesc:
    """One immutable, millimetre-space polygon edge descriptor."""

    __slots__ = (
        "piece_index",
        "piece_id",
        "edge_index",
        "p0",
        "p1",
        "length",
        "direction_deg",
        "unit_direction",
        "inward_normal",
        "angle_at_p0",
        "angle_at_p1",
        "midpoint",
    )

    def __init__(
        self,
        piece_index,
        piece_id,
        edge_index,
        p0,
        p1,
        angle_at_p0,
        angle_at_p1,
    ):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.sqrt(dx * dx + dy * dy)
        ux = dx / max(EPS, length)
        uy = dy / max(EPS, length)
        self.piece_index = int(piece_index)
        self.piece_id = piece_id
        self.edge_index = int(edge_index)
        self.p0 = _point(p0)
        self.p1 = _point(p1)
        self.length = length
        self.direction_deg = normalize_angle_deg(
            math.degrees(math.atan2(dy, dx))
        )
        self.unit_direction = (ux, uy)
        self.inward_normal = (uy, -ux)
        self.angle_at_p0 = float(angle_at_p0)
        self.angle_at_p1 = float(angle_at_p1)
        self.midpoint = (
            0.5 * (p0[0] + p1[0]),
            0.5 * (p0[1] + p1[1]),
        )


class EdgeMatchCandidate:
    """Precomputed reverse-edge alignment between two different pieces."""

    __slots__ = (
        "piece_a",
        "edge_a",
        "piece_b",
        "edge_b",
        "reverse_mapping",
        "transform_b_to_a",
        "transform_a_to_b",
        "rotation_deg",
        "cos_theta",
        "sin_theta",
        "tx",
        "ty",
        "seam_length_error",
        "seam_relative_error",
        "endpoint_angle_error",
        "geometric_cost",
        "transformed_vertices_b",
        "transformed_aabb_b",
        "transformed_area_b",
        "transformed_triangles_b",
        "transformed_vertices_a",
        "transformed_aabb_a",
        "transformed_triangles_a",
        "optional_strip_cost",
    )

    def __init__(
        self,
        desc_a,
        desc_b,
        polygon_a,
        polygon_b,
    ):
        transform_b_to_a = rigid_transform_from_edge_pair(
            (desc_a.p0, desc_a.p1),
            (desc_b.p0, desc_b.p1),
        )
        transform_a_to_b = _inverse_transform(
            transform_b_to_a
        )
        transformed_b = transform_polygon(
            polygon_b, transform_b_to_a
        )
        transformed_a = transform_polygon(
            polygon_a, transform_a_to_b
        )
        length_error = abs(desc_a.length - desc_b.length)
        relative_error = length_error / max(
            EPS, desc_a.length, desc_b.length
        )
        endpoint_error = (
            abs(
                desc_a.angle_at_p0
                + desc_b.angle_at_p1
                - 180.0
            )
            + abs(
                desc_a.angle_at_p1
                + desc_b.angle_at_p0
                - 180.0
            )
        )
        self.piece_a = desc_a.piece_index
        self.edge_a = desc_a.edge_index
        self.piece_b = desc_b.piece_index
        self.edge_b = desc_b.edge_index
        self.reverse_mapping = True
        self.transform_b_to_a = transform_b_to_a
        self.transform_a_to_b = transform_a_to_b
        self.rotation_deg = transform_b_to_a[4]
        self.cos_theta = transform_b_to_a[0]
        self.sin_theta = transform_b_to_a[1]
        self.tx = transform_b_to_a[2]
        self.ty = transform_b_to_a[3]
        self.seam_length_error = length_error
        self.seam_relative_error = relative_error
        self.endpoint_angle_error = endpoint_error
        self.geometric_cost = (
            relative_error
            + 0.10
            * min(
                1.0,
                endpoint_error
                / max(
                    EPS,
                    cfg.SEAM_ENDPOINT_ANGLE_TOLERANCE_DEG,
                ),
            )
        )
        self.transformed_vertices_b = transformed_b
        self.transformed_aabb_b = polygon_aabb(
            transformed_b
        )
        self.transformed_area_b = polygon_area(
            transformed_b
        )
        self.transformed_triangles_b = (
            triangulate_simple_polygon(transformed_b)
        )
        self.transformed_vertices_a = transformed_a
        self.transformed_aabb_a = polygon_aabb(
            transformed_a
        )
        self.transformed_triangles_a = (
            triangulate_simple_polygon(transformed_a)
        )
        self.optional_strip_cost = None

    def other_piece(self, piece_index):
        if piece_index == self.piece_a:
            return self.piece_b
        if piece_index == self.piece_b:
            return self.piece_a
        return None

    def relative_transform_from_fixed(self, fixed_piece_index):
        if fixed_piece_index == self.piece_a:
            return self.transform_b_to_a
        if fixed_piece_index == self.piece_b:
            return self.transform_a_to_b
        return None


class EdgeCandidateGraph:
    """Indexes every compatible cross-piece edge pair exactly once."""

    __slots__ = (
        "edges",
        "candidates",
        "candidates_by_piece_pair",
        "candidates_by_edge",
        "candidate_count_by_open_edge",
        "raw_pair_count",
        "filtered_pair_count",
        "build_ms",
    )

    def __init__(self):
        self.edges = []
        self.candidates = []
        self.candidates_by_piece_pair = {}
        self.candidates_by_edge = {}
        self.candidate_count_by_open_edge = {}
        self.raw_pair_count = 0
        self.filtered_pair_count = 0
        self.build_ms = 0

    def for_edge(self, piece_index, edge_index):
        return self.candidates_by_edge.get(
            (piece_index, edge_index), ()
        )

    def for_piece_pair(self, piece_a, piece_b):
        key = (
            min(piece_a, piece_b),
            max(piece_a, piece_b),
        )
        return self.candidates_by_piece_pair.get(key, ())


def build_edge_candidate_graph(pieces, tolerant=False):
    """Build the complete seam candidate graph once before DFS."""
    started = ticks_ms()
    graph = EdgeCandidateGraph()
    edges_by_piece = []
    for piece_index, piece in enumerate(pieces):
        polygon = piece.polygon_mm
        angles = interior_angles(polygon)
        descriptors = []
        for edge_index, p0 in enumerate(polygon):
            descriptor = EdgeDesc(
                piece_index,
                piece.piece_id,
                edge_index,
                p0,
                polygon[(edge_index + 1) % len(polygon)],
                angles[edge_index],
                angles[(edge_index + 1) % len(polygon)],
            )
            descriptors.append(descriptor)
            graph.edges.append(descriptor)
        edges_by_piece.append(descriptors)

    for piece_a in range(len(pieces)):
        for piece_b in range(piece_a + 1, len(pieces)):
            pair_key = (piece_a, piece_b)
            pair_candidates = []
            for desc_a in edges_by_piece[piece_a]:
                for desc_b in edges_by_piece[piece_b]:
                    graph.raw_pair_count += 1
                    if not _edge_length_matches(
                        desc_a.length,
                        desc_b.length,
                        tolerant=tolerant,
                    ):
                        continue
                    candidate = EdgeMatchCandidate(
                        desc_a,
                        desc_b,
                        pieces[piece_a].polygon_mm,
                        pieces[piece_b].polygon_mm,
                    )
                    pair_candidates.append(candidate)
                    graph.candidates.append(candidate)
                    for key in (
                        (piece_a, desc_a.edge_index),
                        (piece_b, desc_b.edge_index),
                    ):
                        graph.candidates_by_edge.setdefault(
                            key, []
                        ).append(candidate)
            if pair_candidates:
                pair_candidates.sort(
                    key=lambda item: item.geometric_cost
                )
                graph.candidates_by_piece_pair[
                    pair_key
                ] = pair_candidates
    for key, candidates in graph.candidates_by_edge.items():
        candidates.sort(key=lambda item: item.geometric_cost)
        graph.candidate_count_by_open_edge[key] = len(
            candidates
        )
    graph.filtered_pair_count = len(graph.candidates)
    graph.build_ms = max(0, ticks_diff(ticks_ms(), started))
    PERF_STATS.add_stage(
        "candidate_graph_ms", elapsed_ms=graph.build_ms
    )
    PERF_STATS.increment(
        "candidate_pair_count_raw", graph.raw_pair_count
    )
    PERF_STATS.increment(
        "candidate_pair_count_filtered",
        graph.filtered_pair_count,
    )
    return graph


def point_in_polygon(point, polygon, include_boundary=True):
    """Ray-casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        cross = _cross(a, b, point)
        if abs(cross) <= EPS and min(a[0], b[0]) - EPS <= x <= max(
            a[0], b[0]
        ) + EPS and min(a[1], b[1]) - EPS <= y <= max(a[1], b[1]) + EPS:
            return include_boundary
        if (a[1] > y) != (b[1] > y):
            x_intersection = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < x_intersection:
                inside = not inside
    return inside


def aabb_overlap(aabb_a, aabb_b, tolerance=0.0):
    """Return whether two boxes overlap or touch within a linear tolerance."""
    tolerance = float(tolerance)
    return not (
        aabb_a[2] < aabb_b[0] - tolerance
        or aabb_b[2] < aabb_a[0] - tolerance
        or aabb_a[3] < aabb_b[1] - tolerance
        or aabb_b[3] < aabb_a[1] - tolerance
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


def polygons_overlap(poly_a, poly_b, tolerance_mm2=EPS):
    """Return whether convex polygons overlap by more than an area tolerance."""
    return polygon_overlap_area(poly_a, poly_b) > tolerance_mm2


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


def _score_rectangle_details(polygons):
    all_points = []
    sum_area = 0.0
    overlap_area = 0.0
    for i, polygon in enumerate(polygons):
        all_points.extend(polygon)
        sum_area += polygon_area(polygon)
        for j in range(i):
            overlap_area += polygon_overlap_area(polygon, polygons[j])
    rect = minimum_area_rectangle(all_points)
    if rect is None or rect["area"] <= EPS:
        return {"score": 1e9, "reason": "degenerate assembly", "rect": rect}

    long_side = max(rect["width"], rect["height"])
    short_side = min(rect["width"], rect["height"])
    dimension_error = 0.0
    if long_side < cfg.RECT_MIN_WIDTH_MM:
        dimension_error += (cfg.RECT_MIN_WIDTH_MM - long_side) / cfg.RECT_MIN_WIDTH_MM
    elif long_side > cfg.RECT_MAX_WIDTH_MM:
        dimension_error += (long_side - cfg.RECT_MAX_WIDTH_MM) / cfg.RECT_MAX_WIDTH_MM
    if short_side < cfg.RECT_MIN_HEIGHT_MM:
        dimension_error += (cfg.RECT_MIN_HEIGHT_MM - short_side) / cfg.RECT_MIN_HEIGHT_MM
    elif short_side > cfg.RECT_MAX_HEIGHT_MM:
        dimension_error += (short_side - cfg.RECT_MAX_HEIGHT_MM) / cfg.RECT_MAX_HEIGHT_MM

    hull = convex_hull(all_points)
    hull_area = polygon_area(hull)
    union_area_approx = max(0.0, sum_area - overlap_area)
    fill_gap = max(0.0, rect["area"] - union_area_approx)
    convex_gap = max(0.0, rect["area"] - hull_area)

    angle_rad = math.radians(rect["angle_deg"])
    rotated_hull = [_rotate_point(point, -angle_rad) for point in hull]
    min_x, min_y, max_x, max_y = rect["bounds_rotated"]
    boundary_distance = 0.0
    for x, y in rotated_hull:
        boundary_distance += min(abs(x - min_x), abs(x - max_x), abs(y - min_y), abs(y - max_y))
    boundary_distance /= max(1, len(rotated_hull))
    boundary_term = boundary_distance / max(EPS, short_side)

    score = (
        fill_gap / rect["area"]
        + 0.35 * convex_gap / rect["area"]
        + 2.0 * overlap_area / rect["area"]
        + 2.5 * dimension_error
        + 0.25 * boundary_term
    )
    reason = "ok"
    if dimension_error > EPS:
        reason = "target dimensions outside configured range"
    elif fill_gap > cfg.RECT_FILL_GAP_TOLERANCE_MM2:
        reason = "assembly leaves excessive gap"
    return {
        "score": score,
        "reason": reason,
        "rect": rect,
        "fill_gap_mm2": fill_gap,
        "overlap_mm2": overlap_area,
        "dimension_error": dimension_error,
    }


def score_rectangle_assembly(polygons):
    """Return the dimensionless rectangle assembly loss (lower is better)."""
    return _score_rectangle_details(polygons)["score"]


def _edge_length_matches(length_a, length_b, tolerant=False):
    difference = abs(length_a - length_b)
    relaxation = 1.5 if tolerant else 1.0
    absolute_limit = (
        cfg.SEAM_LENGTH_ABS_TOLERANCE_MM * relaxation
    )
    relative_limit = (
        cfg.SEAM_LENGTH_REL_TOLERANCE * relaxation
    )
    relative_error = difference / max(
        EPS, length_a, length_b
    )
    return (
        difference <= absolute_limit
        or relative_error <= relative_limit
    )


def _each_piece_has_outer_edge(polygons, rect, tolerance=None):
    angle_rad = math.radians(rect["angle_deg"])
    min_x, min_y, max_x, max_y = rect["bounds_rotated"]
    if tolerance is None:
        tolerance = cfg.OUTER_EDGE_TOLERANCE_MM
    for polygon in polygons:
        rotated = [_rotate_point(point, -angle_rad) for point in polygon]
        found = False
        for i in range(len(rotated)):
            a = rotated[i]
            b = rotated[(i + 1) % len(rotated)]
            if (
                abs(a[0] - min_x) <= tolerance
                and abs(b[0] - min_x) <= tolerance
            ) or (
                abs(a[0] - max_x) <= tolerance
                and abs(b[0] - max_x) <= tolerance
            ) or (
                abs(a[1] - min_y) <= tolerance
                and abs(b[1] - min_y) <= tolerance
            ) or (
                abs(a[1] - max_y) <= tolerance
                and abs(b[1] - max_y) <= tolerance
            ):
                found = True
                break
        if not found:
            return False
    return True


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


def _place_assembly_on_a4(pieces, placed_polygons, placed_transforms, rect):
    # First align the minimum-area rectangle with A4 x/y axes.
    align_angle = -rect["angle_deg"]
    angle_rad = math.radians(align_angle)
    align = (
        math.cos(angle_rad),
        math.sin(angle_rad),
        0.0,
        0.0,
        normalize_angle_deg(align_angle),
    )
    aligned = [transform_polygon(polygon, align) for polygon in placed_polygons]
    all_points = [point for polygon in aligned for point in polygon]
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    center = (0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
    tx = cfg.TARGET_CENTER_MM[0] - center[0]
    ty = cfg.TARGET_CENTER_MM[1] - center[1]
    placement = (1.0, 0.0, tx, ty, 0.0)
    global_transform = compose_transforms(placement, align)

    target_polygons = {}
    operations = []
    target_all = []
    for index, piece in enumerate(pieces):
        final_transform = compose_transforms(global_transform, placed_transforms[index])
        target_polygon = transform_polygon(piece.polygon_mm, final_transform)
        target_polygons[piece.piece_id] = target_polygon
        target_all.extend(target_polygon)
        target_center = polygon_centroid(target_polygon)
        rotation = _choose_smallest_equivalent_rotation(
            final_transform[4], piece.polygon_mm
        )
        operations.append(
            {
                "piece_id": piece.piece_id,
                "source_center_mm": piece.centroid_mm,
                "target_center_mm": target_center,
                "rotation_deg": rotation,
                "rotation_ambiguous": piece.rotation_ambiguous,
                "confidence": piece.confidence,
            }
        )

    target_min_x = min(point[0] for point in target_all)
    target_max_x = max(point[0] for point in target_all)
    target_min_y = min(point[1] for point in target_all)
    target_max_y = max(point[1] for point in target_all)
    margin = cfg.TARGET_MARGIN_MM
    if (
        target_min_x < margin
        or target_max_x > cfg.A4_WIDTH_MM - margin
        or target_min_y < cfg.DIVIDER_Y_MM + margin
        or target_max_y > cfg.A4_HEIGHT_MM - margin
    ):
        return None, None, None
    return (
        operations,
        target_polygons,
        (target_min_x, target_min_y, target_max_x, target_max_y),
    )


def _transform_from_vertex_direction(
    polygon,
    vertex_index,
    neighbor_index,
    target_vertex,
    target_direction,
):
    source = polygon[vertex_index]
    neighbor = polygon[neighbor_index]
    source_angle = math.atan2(
        neighbor[1] - source[1], neighbor[0] - source[0]
    )
    target_angle = math.atan2(
        target_direction[1], target_direction[0]
    )
    angle = target_angle - source_angle
    cosine = math.cos(angle)
    sine = math.sin(angle)
    tx = target_vertex[0] - (
        cosine * source[0] - sine * source[1]
    )
    ty = target_vertex[1] - (
        sine * source[0] + cosine * source[1]
    )
    return (
        cosine,
        sine,
        tx,
        ty,
        normalize_angle_deg(math.degrees(angle)),
    )


def _fixed_rectangle_polygon(width, height):
    return [
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    ]


def _outside_fixed_rectangle_area(polygon, rectangle):
    return max(
        0.0,
        polygon_area(polygon)
        - polygon_overlap_area(polygon, rectangle),
    )


def _boundary_contact_length(polygon, width, height, tolerance):
    total = 0.0
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        on_boundary = (
            max(abs(a[0]), abs(b[0])) <= tolerance
            or max(abs(a[0] - width), abs(b[0] - width))
            <= tolerance
            or max(abs(a[1]), abs(b[1])) <= tolerance
            or max(abs(a[1] - height), abs(b[1] - height))
            <= tolerance
        )
        if on_boundary:
            total += _distance(a, b)
    return total


def _boundary_anchor_candidates(polygon, width, height, rectangle):
    corners = [
        ((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))),
        ((width, 0.0), ((-1.0, 0.0), (0.0, 1.0))),
        ((width, height), ((-1.0, 0.0), (0.0, -1.0))),
        ((0.0, height), ((1.0, 0.0), (0.0, -1.0))),
    ]
    result = []
    seen = set()
    for vertex_index in range(len(polygon)):
        neighbors = (
            (vertex_index - 1) % len(polygon),
            (vertex_index + 1) % len(polygon),
        )
        for neighbor_index in neighbors:
            for corner_index, (corner, directions) in enumerate(corners):
                for direction_index, direction in enumerate(directions):
                    transform = _transform_from_vertex_direction(
                        polygon,
                        vertex_index,
                        neighbor_index,
                        corner,
                        direction,
                    )
                    candidate = transform_polygon(
                        polygon, transform
                    )
                    outside = _outside_fixed_rectangle_area(
                        candidate, rectangle
                    )
                    if outside > cfg.FIXED_RECT_MAX_OUTSIDE_MM2:
                        continue
                    center = polygon_centroid(candidate)
                    key = (
                        int(round(center[0] * 2.0)),
                        int(round(center[1] * 2.0)),
                        int(round(transform[4] * 2.0)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(
                        {
                            "polygon": candidate,
                            "transform": transform,
                            "outside": outside,
                            "anchor": {
                                "type": "boundary",
                                "vertex_index": vertex_index,
                                "neighbor_index": neighbor_index,
                                "corner_index": corner_index,
                                "direction_index": direction_index,
                            },
                        }
                    )
    return result


def _fixed_state_key(state):
    result = []
    for index, transform in enumerate(state["transforms"]):
        if transform is None:
            continue
        center = polygon_centroid(state["polygons"][index])
        result.extend(
            (
                index,
                int(round(center[0])),
                int(round(center[1])),
                int(round(transform[4])),
            )
        )
    return tuple(result)


def _fixed_partial_rank(
    state, rectangle, width, height
):
    outside = 0.0
    overlap = 0.0
    boundary = 0.0
    for position, index in enumerate(state["placed"]):
        polygon = state["polygons"][index]
        outside += _outside_fixed_rectangle_area(
            polygon, rectangle
        )
        boundary += _boundary_contact_length(
            polygon,
            width,
            height,
            cfg.FIXED_RECT_BOUNDARY_TOLERANCE_MM,
        )
        for earlier in state["placed"][:position]:
            overlap += polygon_overlap_area(
                polygon, state["polygons"][earlier]
            )
    return 5.0 * outside + 20.0 * overlap - 0.5 * boundary


def _fixed_complete_metrics(polygons, rectangle, width, height):
    outside = 0.0
    overlap = 0.0
    inside_sum = 0.0
    for index, polygon in enumerate(polygons):
        inside_area = polygon_overlap_area(polygon, rectangle)
        inside_sum += inside_area
        outside += max(0.0, polygon_area(polygon) - inside_area)
        for earlier in range(index):
            overlap += polygon_overlap_area(
                polygon, polygons[earlier]
            )
    rectangle_area = width * height
    gap = max(0.0, rectangle_area - (inside_sum - overlap))
    score = (
        outside + 5.0 * overlap + gap
    ) / max(EPS, rectangle_area)
    return {
        "score": score,
        "outside_mm2": outside,
        "overlap_mm2": overlap,
        "fill_gap_mm2": gap,
    }


def _max_corresponding_vertex_gap(polygons, width, height):
    maximum = 0.0
    for piece_index, polygon in enumerate(polygons):
        for point in polygon:
            nearest = min(
                abs(point[0]),
                abs(point[0] - width),
                abs(point[1]),
                abs(point[1] - height),
            )
            for other_index, other in enumerate(polygons):
                if other_index == piece_index:
                    continue
                for edge_index, edge_start in enumerate(other):
                    edge_end = other[
                        (edge_index + 1) % len(other)
                    ]
                    nearest = min(
                        nearest,
                        _point_segment_distance(
                            point, edge_start, edge_end
                        ),
                    )
            maximum = max(maximum, nearest)
    return maximum


def _fixed_edge_candidates(
    pieces,
    new_index,
    state,
    rectangle,
):
    result = []
    new_polygon = pieces[new_index].polygon_mm
    new_lengths = edge_lengths(new_polygon)
    for fixed_index in state["placed"]:
        fixed_polygon = state["polygons"][fixed_index]
        fixed_lengths = edge_lengths(fixed_polygon)
        for fixed_edge in range(len(fixed_polygon)):
            a0 = fixed_polygon[fixed_edge]
            a1 = fixed_polygon[(fixed_edge + 1) % len(fixed_polygon)]
            for new_edge in range(len(new_polygon)):
                if not _edge_length_matches(
                    fixed_lengths[fixed_edge],
                    new_lengths[new_edge],
                    tolerant=True,
                ):
                    continue
                b0 = new_polygon[new_edge]
                b1 = new_polygon[(new_edge + 1) % len(new_polygon)]
                transform = rigid_transform_from_edge_pair(
                    (a0, a1), (b0, b1)
                )
                candidate = transform_polygon(
                    new_polygon, transform
                )
                mapped_b0 = candidate[new_edge]
                mapped_b1 = candidate[
                    (new_edge + 1) % len(candidate)
                ]
                endpoint_error = max(
                    _distance(mapped_b0, a1),
                    _distance(mapped_b1, a0),
                )
                if (
                    endpoint_error
                    > cfg.CORRESPONDING_VERTEX_TOLERANCE_MM
                ):
                    continue
                fixed_center = polygon_centroid(fixed_polygon)
                new_center = polygon_centroid(candidate)
                if (
                    _cross(a0, a1, fixed_center)
                    * _cross(a0, a1, new_center)
                    >= -EPS
                ):
                    continue
                outside = _outside_fixed_rectangle_area(
                    candidate, rectangle
                )
                if outside > cfg.FIXED_RECT_MAX_OUTSIDE_MM2:
                    continue
                result.append(
                    {
                        "polygon": candidate,
                        "transform": transform,
                        "outside": outside,
                        "anchor": {
                            "type": "seam",
                            "piece_a_index": fixed_index,
                            "edge_a_index": fixed_edge,
                            "piece_b_index": new_index,
                            "edge_b_index": new_edge,
                            "endpoint_error_mm": endpoint_error,
                            "length_error_mm": abs(
                                fixed_lengths[fixed_edge]
                                - new_lengths[new_edge]
                            ),
                        },
                    }
                )
    return result


def _translate_fixed_plan_to_a4(
    pieces,
    polygons,
    transforms,
    width,
    height,
):
    translate_x = cfg.TARGET_CENTER_MM[0] - 0.5 * width
    translate_y = cfg.TARGET_CENTER_MM[1] - 0.5 * height
    placement = (
        1.0,
        0.0,
        translate_x,
        translate_y,
        0.0,
    )
    target_polygons = {}
    operations = []
    for index, piece in enumerate(pieces):
        final_transform = compose_transforms(
            placement, transforms[index]
        )
        target_polygon = transform_polygon(
            piece.polygon_mm, final_transform
        )
        target_polygons[piece.piece_id] = target_polygon
        operations.append(
            {
                "piece_id": piece.piece_id,
                "source_center_mm": piece.centroid_mm,
                "target_center_mm": polygon_centroid(
                    target_polygon
                ),
                "rotation_deg": _choose_smallest_equivalent_rotation(
                    final_transform[4], piece.polygon_mm
                ),
                "rotation_ambiguous": piece.rotation_ambiguous,
                "confidence": piece.confidence,
            }
        )
    target_rect = (
        translate_x,
        translate_y,
        translate_x + width,
        translate_y + height,
    )
    margin = cfg.TARGET_MARGIN_MM
    if (
        target_rect[0] < margin
        or target_rect[2] > cfg.A4_WIDTH_MM - margin
        or target_rect[1] < cfg.DIVIDER_Y_MM + margin
        or target_rect[3] > cfg.A4_HEIGHT_MM - margin
    ):
        return None, None, None
    return operations, target_polygons, target_rect


def _plan_fixed_rectangle(pieces, size_mm):
    width = float(size_mm[0])
    height = float(size_mm[1])
    rectangle = _fixed_rectangle_polygon(width, height)
    count = len(pieces)
    boundary_candidates = [
        _boundary_anchor_candidates(
            piece.polygon_mm, width, height, rectangle
        )
        for piece in pieces
    ]
    states = []
    for piece_index in range(count):
        for candidate in boundary_candidates[piece_index]:
            polygons = [None for _ in range(count)]
            transforms = [None for _ in range(count)]
            polygons[piece_index] = candidate["polygon"]
            transforms[piece_index] = candidate["transform"]
            states.append(
                {
                    "placed": (piece_index,),
                    "polygons": polygons,
                    "transforms": transforms,
                    "seams": [],
                    "anchors": [candidate["anchor"]],
                }
            )
    if not states:
        return PlanResult(
            reason="no boundary anchor for fixed rectangle",
            mode="fixed_tolerant",
        )

    nodes = len(states)
    work_counter = 0
    for _depth in range(1, count):
        expanded = []
        seen = set()
        for state in states:
            work_counter += 1
            _geometry_exitpoint(work_counter)
            for new_index in range(count):
                if new_index in state["placed"]:
                    continue
                candidates = list(
                    boundary_candidates[new_index]
                )
                candidates.extend(
                    _fixed_edge_candidates(
                        pieces,
                        new_index,
                        state,
                        rectangle,
                    )
                )
                for candidate in candidates:
                    work_counter += 1
                    _geometry_exitpoint(work_counter)
                    overlap = 0.0
                    for placed_index in state["placed"]:
                        overlap += polygon_overlap_area(
                            candidate["polygon"],
                            state["polygons"][placed_index],
                        )
                    if overlap > cfg.FIXED_RECT_MAX_OVERLAP_MM2:
                        continue
                    polygons = list(state["polygons"])
                    transforms = list(state["transforms"])
                    polygons[new_index] = candidate["polygon"]
                    transforms[new_index] = candidate["transform"]
                    anchor = candidate["anchor"]
                    seams = list(state["seams"])
                    anchors = list(state["anchors"])
                    if anchor["type"] == "seam":
                        seams.append(anchor)
                    else:
                        anchors.append(anchor)
                    new_state = {
                        "placed": tuple(
                            sorted(
                                state["placed"] + (new_index,)
                            )
                        ),
                        "polygons": polygons,
                        "transforms": transforms,
                        "seams": seams,
                        "anchors": anchors,
                    }
                    key = _fixed_state_key(new_state)
                    if key in seen:
                        continue
                    seen.add(key)
                    expanded.append(new_state)
        nodes += len(expanded)
        if not expanded:
            return PlanResult(
                reason="fixed rectangle beam exhausted",
                search_nodes=nodes,
                mode="fixed_tolerant",
            )
        expanded.sort(
            key=lambda state: _fixed_partial_rank(
                state, rectangle, width, height
            )
        )
        states = expanded[: cfg.FIXED_RECT_BEAM_WIDTH]

    best = None
    best_metrics = None
    for state in states:
        work_counter += 1
        _geometry_exitpoint(work_counter)
        metrics = _fixed_complete_metrics(
            state["polygons"], rectangle, width, height
        )
        if not _each_piece_has_outer_edge(
            state["polygons"],
            {
                "angle_deg": 0.0,
                "bounds_rotated": (
                    0.0,
                    0.0,
                    width,
                    height,
                ),
            },
            tolerance=cfg.CORRESPONDING_VERTEX_TOLERANCE_MM,
        ):
            continue
        if (
            best_metrics is None
            or metrics["score"] < best_metrics["score"]
        ):
            best = state
            best_metrics = metrics
    if best is None:
        return PlanResult(
            reason="no complete fixed rectangle candidate",
            search_nodes=nodes,
            mode="fixed_tolerant",
        )
    if (
        best_metrics["score"]
        > cfg.FIXED_RECT_SCORE_THRESHOLD
        or best_metrics["outside_mm2"]
        > cfg.FIXED_RECT_MAX_OUTSIDE_MM2
        or best_metrics["overlap_mm2"]
        > cfg.FIXED_RECT_MAX_OVERLAP_MM2
        or best_metrics["fill_gap_mm2"]
        > cfg.FIXED_RECT_MAX_GAP_MM2
    ):
        return PlanResult(
            reason=(
                "fixed rectangle metrics score={:.4f},outside={:.1f},"
                "overlap={:.1f},gap={:.1f}"
            ).format(
                best_metrics["score"],
                best_metrics["outside_mm2"],
                best_metrics["overlap_mm2"],
                best_metrics["fill_gap_mm2"],
            ),
            score=best_metrics["score"],
            search_nodes=nodes,
            mode="fixed_tolerant",
            fill_gap_mm2=best_metrics["fill_gap_mm2"],
        )

    operations, target_polygons, target_rect = (
        _translate_fixed_plan_to_a4(
            pieces,
            best["polygons"],
            best["transforms"],
            width,
            height,
        )
    )
    if operations is None:
        return PlanResult(
            reason="fixed target rectangle does not fit A4 lower region",
            score=best_metrics["score"],
            search_nodes=nodes,
            mode="fixed_tolerant",
        )
    max_vertex_error = 0.0
    if best["seams"]:
        max_vertex_error = max(
            seam["endpoint_error_mm"]
            for seam in best["seams"]
        )
    max_vertex_error = max(
        max_vertex_error,
        _max_corresponding_vertex_gap(
            best["polygons"], width, height
        ),
    )
    return PlanResult(
        valid=True,
        reason="ok",
        score=best_metrics["score"],
        operations=operations,
        target_polygons=target_polygons,
        target_rect=target_rect,
        search_nodes=nodes,
        mode="fixed_tolerant",
        max_vertex_error_mm=max_vertex_error,
        fill_gap_mm2=best_metrics["fill_gap_mm2"],
        overlap_mm2=best_metrics["overlap_mm2"],
        outside_mm2=best_metrics["outside_mm2"],
        seams=best["seams"],
    )


def _outer_first_axis_error_deg(a, b):
    angle = abs(
        math.degrees(
            math.atan2(b[1] - a[1], b[0] - a[0])
        )
    ) % 90.0
    return min(angle, 90.0 - angle)


def _outer_first_has_axis_edge(
    polygon,
    piece_index,
    seam_edges,
    axis_tolerance_deg,
):
    for edge_index, edge_start in enumerate(polygon):
        if (piece_index, edge_index) in seam_edges:
            continue
        edge_end = polygon[(edge_index + 1) % len(polygon)]
        if (
            _outer_first_axis_error_deg(
                edge_start, edge_end
            )
            <= axis_tolerance_deg
        ):
            return True
    return False


def _outer_first_root_candidates(pieces, tolerant):
    """Rank possible target-boundary edges.

    An unmatched edge is more likely to be external.  An edge touching an
    approximately right-angle vertex gets a secondary priority boost, but
    right angles are deliberately not mandatory: a cut can terminate exactly
    at a rectangle corner and split that 90-degree corner between two pieces.
    """
    candidates = []
    all_lengths = [
        edge_lengths(piece.polygon_mm) for piece in pieces
    ]
    all_angles = [
        interior_angles(piece.polygon_mm) for piece in pieces
    ]
    for piece_index, piece in enumerate(pieces):
        polygon = piece.polygon_mm
        for edge_index, length in enumerate(
            all_lengths[piece_index]
        ):
            compatible = 0
            for other_index in range(len(pieces)):
                if other_index == piece_index:
                    continue
                for other_length in all_lengths[other_index]:
                    if _edge_length_matches(
                        # Keep root ranking selective even when the later seam
                        # stage is tolerant.  The 20 mm endpoint allowance is
                        # far too broad for deciding whether an edge is likely
                        # to be an unmatched outside edge.
                        length, other_length, tolerant=False
                    ):
                        compatible += 1
            next_vertex = (edge_index + 1) % len(polygon)
            right_angle_error = min(
                abs(
                    all_angles[piece_index][edge_index]
                    - 90.0
                ),
                abs(
                    all_angles[piece_index][next_vertex]
                    - 90.0
                ),
            )
            # Piece polygons are clockwise.  Mapping the directed edge towards
            # negative x places its interior in positive y, i.e. below a
            # hypothesised top rectangle boundary.
            transform = _transform_from_vertex_direction(
                polygon,
                edge_index,
                next_vertex,
                (length, 0.0),
                (-1.0, 0.0),
            )
            placed = transform_polygon(polygon, transform)
            candidates.append(
                {
                    "piece_index": piece_index,
                    "edge_index": edge_index,
                    "polygon": placed,
                    "transform": transform,
                    "priority": (
                        min(compatible, 2),
                        right_angle_error,
                        -length,
                    ),
                }
            )
    candidates.sort(key=lambda item: item["priority"])
    return candidates


def _outer_first_partial_bounds_valid(
    polygons, placed_indices, slack
):
    points = [
        point
        for index in placed_indices
        for point in polygons[index]
    ]
    if not points:
        return False
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    maximum_side = max(
        cfg.RECT_MAX_WIDTH_MM, cfg.RECT_MAX_HEIGHT_MM
    )
    # The root is a hypothesised top outside edge at y=0. Correct assemblies
    # cannot extend materially above it. Along x the root may be only a middle
    # segment of the target side, so negative x remains legal.
    return (
        min_y >= -slack
        and max_y <= maximum_side + slack
        and max_x - min_x <= maximum_side + slack
    )


def _outer_first_complete_metrics(polygons):
    points = [point for polygon in polygons for point in polygon]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    width = max_x - min_x
    height = max_y - min_y
    long_side = max(width, height)
    short_side = min(width, height)
    if (
        long_side < cfg.RECT_MIN_WIDTH_MM
        or long_side > cfg.RECT_MAX_WIDTH_MM
        or short_side < cfg.RECT_MIN_HEIGHT_MM
        or short_side > cfg.RECT_MAX_HEIGHT_MM
    ):
        return None

    overlap = 0.0
    total_area = 0.0
    for index, polygon in enumerate(polygons):
        total_area += polygon_area(polygon)
        for earlier in range(index):
            overlap += polygon_overlap_area(
                polygon, polygons[earlier]
            )
    rectangle_area = width * height
    union_area = max(0.0, total_area - overlap)
    gap = max(0.0, rectangle_area - union_area)
    score = (
        gap + 2.0 * overlap
    ) / max(EPS, rectangle_area)
    return {
        "score": score,
        "rect": {
            "angle_deg": 0.0,
            "width": width,
            "height": height,
            "area": rectangle_area,
            "bounds_rotated": (
                min_x,
                min_y,
                max_x,
                max_y,
            ),
        },
        "fill_gap_mm2": gap,
        "overlap_mm2": overlap,
    }


def _corner_dimension_candidates(pieces, tolerant):
    """Infer plausible unknown rectangle sizes from corner-adjacent edges."""
    angle_tolerance = (
        cfg.OUTER_FIRST_TOLERANT_AXIS_TOLERANCE_DEG
        if tolerant
        else cfg.OUTER_FIRST_AXIS_TOLERANCE_DEG
    )
    adjacent_lengths = []
    for piece in pieces:
        polygon = piece.polygon_mm
        lengths = edge_lengths(polygon)
        angles = interior_angles(polygon)
        for vertex_index, angle in enumerate(angles):
            if abs(angle - 90.0) > angle_tolerance:
                continue
            adjacent_lengths.append(
                lengths[(vertex_index - 1) % len(polygon)]
            )
            adjacent_lengths.append(lengths[vertex_index])
    if not adjacent_lengths:
        return []

    # Dynamic subset sums avoid a 2^N enumeration when more than one noisy
    # right-angle candidate is present. Half-millimetre bins are finer than
    # the camera contour accuracy and cap the set at roughly 240 values.
    lengths = sorted(
        set(round(value * 2.0) / 2.0 for value in adjacent_lengths)
    )
    sums = {0.0}
    maximum_side = cfg.RECT_MAX_WIDTH_MM + (
        cfg.CORRESPONDING_VERTEX_TOLERANCE_MM
        if tolerant
        else cfg.OUTER_EDGE_TOLERANCE_MM
    )
    for length in lengths:
        additions = []
        for value in sums:
            candidate = round((value + length) * 2.0) / 2.0
            if candidate <= maximum_side:
                additions.append(candidate)
        sums.update(additions)

    side_values = set()
    for value in sums:
        if value < cfg.RECT_MIN_HEIGHT_MM - 10.0:
            continue
        side_values.add(value)
        # Fabrication and contour simplification often move a nominal cm-scale
        # edge by 1..3 mm. Rounded alternatives are hypotheses, not hard-coded
        # target dimensions, and compete by geometric score.
        side_values.add(round(value / 5.0) * 5.0)

    total_area = sum(piece.area_mm2 for piece in pieces)
    pairs = {}

    def add_pair(long_side, short_side):
        long_side = float(long_side)
        short_side = float(short_side)
        if short_side > long_side:
            long_side, short_side = short_side, long_side
        if (
            long_side < cfg.RECT_MIN_WIDTH_MM
            or long_side > cfg.RECT_MAX_WIDTH_MM
            or short_side < cfg.RECT_MIN_HEIGHT_MM
            or short_side > cfg.RECT_MAX_HEIGHT_MM
        ):
            return
        key = (
            int(round(long_side * 2.0)),
            int(round(short_side * 2.0)),
        )
        area_error = abs(
            long_side * short_side - total_area
        ) / max(EPS, total_area)
        evidence_error = min(
            abs(long_side - value) for value in side_values
        ) + min(
            abs(short_side - value) for value in side_values
        )
        rank = area_error * 20.0 + evidence_error
        previous = pairs.get(key)
        if previous is None or rank < previous[0]:
            pairs[key] = (rank, long_side, short_side)

    for value in side_values:
        if (
            cfg.RECT_MIN_WIDTH_MM
            <= value
            <= cfg.RECT_MAX_WIDTH_MM
        ):
            add_pair(value, total_area / max(EPS, value))
        if (
            cfg.RECT_MIN_HEIGHT_MM
            <= value
            <= cfg.RECT_MAX_HEIGHT_MM
        ):
            add_pair(total_area / max(EPS, value), value)
    # Also retain near-area pairs supported independently by two boundary
    # length sums; these cover small missing/overlapping contour regions.
    long_values = [
        value
        for value in side_values
        if cfg.RECT_MIN_WIDTH_MM
        <= value
        <= cfg.RECT_MAX_WIDTH_MM
    ]
    short_values = [
        value
        for value in side_values
        if cfg.RECT_MIN_HEIGHT_MM
        <= value
        <= cfg.RECT_MAX_HEIGHT_MM
    ]
    for long_side in long_values:
        for short_side in short_values:
            if (
                abs(long_side * short_side - total_area)
                / max(EPS, total_area)
                <= 0.12
            ):
                add_pair(long_side, short_side)

    ordered = sorted(pairs.values(), key=lambda item: item[0])
    result = []
    for _, long_side, short_side in ordered:
        result.append((long_side, short_side))
        if abs(long_side - short_side) > EPS:
            result.append((short_side, long_side))
    return result


def _corner_anchor_candidates(
    piece, piece_index, width, height, tolerant
):
    polygon = piece.polygon_mm
    angles = interior_angles(polygon)
    angle_tolerance = (
        cfg.OUTER_FIRST_TOLERANT_AXIS_TOLERANCE_DEG
        if tolerant
        else cfg.OUTER_FIRST_AXIS_TOLERANCE_DEG
    )
    rectangle = _fixed_rectangle_polygon(width, height)
    corners = [
        ((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))),
        ((width, 0.0), ((-1.0, 0.0), (0.0, 1.0))),
        ((width, height), ((-1.0, 0.0), (0.0, -1.0))),
        ((0.0, height), ((1.0, 0.0), (0.0, -1.0))),
    ]
    maximum_outside = (
        cfg.FIXED_RECT_MAX_OUTSIDE_MM2
        if tolerant
        else max(
            cfg.OVERLAP_AREA_TOLERANCE_MM2,
            cfg.RECT_FILL_GAP_TOLERANCE_MM2 * 0.25,
        )
    )
    result = []
    seen = set()
    for vertex_index, angle in enumerate(angles):
        if abs(angle - 90.0) > angle_tolerance:
            continue
        previous_index = (vertex_index - 1) % len(polygon)
        next_index = (vertex_index + 1) % len(polygon)
        for neighbor_index, other_index in (
            (previous_index, next_index),
            (next_index, previous_index),
        ):
            for corner_index, (corner, directions) in enumerate(
                corners
            ):
                for direction in directions:
                    transform = _transform_from_vertex_direction(
                        polygon,
                        vertex_index,
                        neighbor_index,
                        corner,
                        direction,
                    )
                    candidate = transform_polygon(
                        polygon, transform
                    )
                    if (
                        _outer_first_axis_error_deg(
                            candidate[vertex_index],
                            candidate[other_index],
                        )
                        > angle_tolerance
                    ):
                        continue
                    outside = _outside_fixed_rectangle_area(
                        candidate, rectangle
                    )
                    if outside > maximum_outside:
                        continue
                    center = polygon_centroid(candidate)
                    key = (
                        corner_index,
                        int(round(center[0] * 2.0)),
                        int(round(center[1] * 2.0)),
                        int(round(transform[4] * 2.0)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(
                        {
                            "piece_index": piece_index,
                            "corner_index": corner_index,
                            "polygon": candidate,
                            "transform": transform,
                            "outside": outside,
                            "anchor": {
                                "type": "boundary",
                                "piece_index": piece_index,
                                "corner_index": corner_index,
                                "vertex_index": vertex_index,
                                "neighbor_index": neighbor_index,
                            },
                        }
                    )
    result.sort(key=lambda item: item["outside"])
    return result


def _fixed_graph_edge_candidates(
    pieces,
    new_index,
    state,
    rectangle,
    candidate_graph,
):
    """Compose cached relative edge poses with placed world poses."""
    result = []
    seen = set()
    used_edges = set()
    for seam in state["seams"]:
        used_edges.add(
            (seam["piece_a_index"], seam["edge_a_index"])
        )
        used_edges.add(
            (seam["piece_b_index"], seam["edge_b_index"])
        )
    for fixed_index in state["placed"]:
        for match in candidate_graph.for_piece_pair(
            fixed_index, new_index
        ):
            if match.other_piece(fixed_index) != new_index:
                continue
            if fixed_index == match.piece_a:
                fixed_edge = match.edge_a
                new_edge = match.edge_b
                relative = match.transform_b_to_a
                relative_polygon = match.transformed_vertices_b
            else:
                fixed_edge = match.edge_b
                new_edge = match.edge_a
                relative = match.transform_a_to_b
                relative_polygon = match.transformed_vertices_a
            if (
                (fixed_index, fixed_edge) in used_edges
                or (new_index, new_edge) in used_edges
            ):
                continue
            fixed_transform = state["transforms"][fixed_index]
            transform = compose_transforms(
                fixed_transform, relative
            )
            candidate = transform_polygon(
                relative_polygon, fixed_transform
            )
            a0 = state["polygons"][fixed_index][fixed_edge]
            a1 = state["polygons"][fixed_index][
                (fixed_edge + 1)
                % len(state["polygons"][fixed_index])
            ]
            fixed_center = polygon_centroid(
                state["polygons"][fixed_index]
            )
            new_center = polygon_centroid(candidate)
            if (
                _cross(a0, a1, fixed_center)
                * _cross(a0, a1, new_center)
                >= -EPS
            ):
                continue
            outside = _outside_fixed_rectangle_area(
                candidate, rectangle
            )
            if outside > cfg.FIXED_RECT_MAX_OUTSIDE_MM2:
                continue
            key = (
                fixed_index,
                fixed_edge,
                new_edge,
                int(
                    round(
                        new_center[0]
                        / cfg.STATE_POSITION_QUANTIZATION_MM
                    )
                ),
                int(
                    round(
                        new_center[1]
                        / cfg.STATE_POSITION_QUANTIZATION_MM
                    )
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "polygon": candidate,
                    "transform": transform,
                    "outside": outside,
                    "candidate_cost": match.geometric_cost,
                    "anchor": {
                        "type": "seam",
                        "piece_a_index": fixed_index,
                        "edge_a_index": fixed_edge,
                        "piece_b_index": new_index,
                        "edge_b_index": new_edge,
                        "endpoint_error_mm": (
                            0.5 * match.seam_length_error
                        ),
                        "length_error_mm": (
                            match.seam_length_error
                        ),
                    },
                }
            )
    return result


def _plan_corner_hybrid_rectangle(
    pieces, tolerant, candidate_graph, plan_started_ms=None
):
    """Anchor reliable corner pieces, then attach non-corner pieces by seams."""
    mode = (
        "corner_outer_tolerant"
        if tolerant
        else "corner_outer_strict"
    )
    all_dimensions = _corner_dimension_candidates(
        pieces, tolerant
    )
    dimensions = all_dimensions[
        : max(1, cfg.MAX_RECTANGLE_HYPOTHESES)
    ]
    plan_stats = {
        "rectangle_hypothesis_raw_count": len(all_dimensions),
        "rectangle_hypothesis_count": len(dimensions),
        "rectangle_hypotheses_used": len(dimensions),
    }
    PERF_STATS.increment(
        "rectangle_hypothesis_count", len(dimensions)
    )
    if not dimensions:
        return PlanResult(
            reason="no near-right-angle dimension evidence",
            mode=mode,
            plan_stats=plan_stats,
        )
    count = len(pieces)
    best = None
    best_metrics = None
    nodes = 0
    limit_hit = False
    timed_out = False
    maximum_nodes = min(
        cfg.OUTER_FIRST_CORNER_MAX_SEARCH_NODES,
        cfg.MAX_DFS_NODES,
    )

    for width, height in dimensions:
        if (
            plan_started_ms is not None
            and ticks_diff(ticks_ms(), plan_started_ms)
            >= cfg.MAX_PLAN_TIME_MS
        ):
            timed_out = True
            limit_hit = True
            break
        rectangle = _fixed_rectangle_polygon(width, height)
        boundary_candidates = [
            _corner_anchor_candidates(
                piece, index, width, height, tolerant
            )
            for index, piece in enumerate(pieces)
        ]
        states = []
        for piece_index, candidates in enumerate(
            boundary_candidates
        ):
            for candidate in candidates:
                polygons = [None for _ in range(count)]
                transforms = [None for _ in range(count)]
                polygons[piece_index] = candidate["polygon"]
                transforms[piece_index] = candidate["transform"]
                states.append(
                    {
                        "placed": (piece_index,),
                        "polygons": polygons,
                        "transforms": transforms,
                        "seams": [],
                        "anchors": [candidate["anchor"]],
                        "corners": {
                            candidate["corner_index"]
                        },
                    }
                )
        if not states:
            continue
        states.sort(
            key=lambda state: _fixed_partial_rank(
                state, rectangle, width, height
            )
        )
        states = states[
            : cfg.OUTER_FIRST_CORNER_BEAM_WIDTH
        ]
        nodes += len(states)

        for _depth in range(1, count):
            expanded = []
            seen = set()
            for state in states:
                piece_options = []
                for new_index in range(count):
                    if new_index in state["placed"]:
                        continue
                    candidates = []
                    for candidate in boundary_candidates[new_index]:
                        if (
                            candidate["corner_index"]
                            not in state["corners"]
                        ):
                            candidates.append(candidate)
                    candidates.extend(
                        _fixed_graph_edge_candidates(
                            pieces,
                            new_index,
                            state,
                            rectangle,
                            candidate_graph,
                        )
                    )
                    candidates.sort(
                        key=lambda candidate: (
                            candidate["outside"],
                            candidate.get(
                                "candidate_cost", 0.0
                            ),
                        )
                    )
                    piece_options.append(
                        (len(candidates), new_index, candidates)
                    )
                # MRV is an ordering heuristic only. All pieces are retained,
                # so a noisy missing seam candidate cannot remove a solution.
                piece_options.sort(key=lambda item: item[0])
                for _option_count, new_index, candidates in piece_options:
                    for candidate in candidates[
                        : cfg.OUTER_FIRST_CORNER_CANDIDATES_PER_PIECE
                    ]:
                        nodes += 1
                        _geometry_exitpoint(nodes)
                        if (
                            nodes
                            >= maximum_nodes
                        ):
                            limit_hit = True
                            break
                        if (
                            plan_started_ms is not None
                            and ticks_diff(
                                ticks_ms(), plan_started_ms
                            )
                            >= cfg.MAX_PLAN_TIME_MS
                        ):
                            timed_out = True
                            limit_hit = True
                            break
                        overlap = 0.0
                        for placed_index in state["placed"]:
                            overlap += polygon_overlap_area(
                                candidate["polygon"],
                                state["polygons"][placed_index],
                            )
                        overlap_limit = (
                            cfg.FIXED_RECT_MAX_OVERLAP_MM2
                            if tolerant
                            else cfg.OVERLAP_AREA_TOLERANCE_MM2
                        )
                        if overlap > overlap_limit:
                            continue
                        polygons = list(state["polygons"])
                        transforms = list(state["transforms"])
                        polygons[new_index] = candidate["polygon"]
                        transforms[new_index] = candidate["transform"]
                        anchor = candidate["anchor"]
                        seams = list(state["seams"])
                        anchors = list(state["anchors"])
                        corners = set(state["corners"])
                        if anchor["type"] == "seam":
                            seams.append(anchor)
                        else:
                            anchors.append(anchor)
                            corners.add(candidate["corner_index"])
                        new_state = {
                            "placed": tuple(
                                sorted(
                                    state["placed"]
                                    + (new_index,)
                                )
                            ),
                            "polygons": polygons,
                            "transforms": transforms,
                            "seams": seams,
                            "anchors": anchors,
                            "corners": corners,
                        }
                        key = _fixed_state_key(new_state)
                        if key in seen:
                            continue
                        seen.add(key)
                        expanded.append(new_state)
                    if limit_hit:
                        break
                if limit_hit:
                    break
            if not expanded:
                states = []
                break
            expanded.sort(
                key=lambda state: _fixed_partial_rank(
                    state, rectangle, width, height
                )
            )
            states = expanded[
                : cfg.OUTER_FIRST_CORNER_BEAM_WIDTH
            ]
            if limit_hit:
                break

        for state in states:
            if len(state["placed"]) != count:
                continue
            metrics = _fixed_complete_metrics(
                state["polygons"],
                rectangle,
                width,
                height,
            )
            boundary_tolerance = (
                cfg.CORRESPONDING_VERTEX_TOLERANCE_MM
                if tolerant
                else cfg.OUTER_EDGE_TOLERANCE_MM
            )
            if not _each_piece_has_outer_edge(
                state["polygons"],
                {
                    "angle_deg": 0.0,
                    "bounds_rotated": (
                        0.0,
                        0.0,
                        width,
                        height,
                    ),
                },
                tolerance=boundary_tolerance,
            ):
                continue
            if (
                best_metrics is None
                or metrics["score"] < best_metrics["score"]
            ):
                best = state
                best_metrics = metrics
                best_width = width
                best_height = height
        if (
            best_metrics is not None
            and best_metrics["score"]
            <= cfg.FIXED_RECT_SCORE_THRESHOLD
        ):
            break
        if limit_hit:
            break

    if best is None:
        return PlanResult(
            reason=(
                "corner-outer plan time limit reached"
                if timed_out
                else "corner-outer search limit reached"
                if limit_hit
                else "no corner-outer assembly"
            ),
            search_nodes=nodes,
            mode=mode,
            plan_stats=plan_stats,
        )
    score_limit = (
        cfg.TOLERANT_RECTANGLE_SCORE_THRESHOLD
        if tolerant
        else cfg.FIXED_RECT_SCORE_THRESHOLD
    )
    gap_limit = (
        cfg.TOLERANT_MAX_FILL_GAP_RATIO
        * best_width
        * best_height
        if tolerant
        else cfg.RECT_FILL_GAP_TOLERANCE_MM2
    )
    if (
        best_metrics["score"] > score_limit
        or best_metrics["fill_gap_mm2"] > gap_limit
    ):
        return PlanResult(
            reason=(
                "corner-outer metrics score={:.4f},gap={:.1f}"
            ).format(
                best_metrics["score"],
                best_metrics["fill_gap_mm2"],
            ),
            score=best_metrics["score"],
            search_nodes=nodes,
            mode=mode,
            fill_gap_mm2=best_metrics["fill_gap_mm2"],
            plan_stats=plan_stats,
        )
    operations, target_polygons, target_rect = (
        _translate_fixed_plan_to_a4(
            pieces,
            best["polygons"],
            best["transforms"],
            best_width,
            best_height,
        )
    )
    if operations is None:
        return PlanResult(
            reason="corner-outer target does not fit A4",
            search_nodes=nodes,
            mode=mode,
            plan_stats=plan_stats,
        )
    maximum_error = _max_corresponding_vertex_gap(
        best["polygons"], best_width, best_height
    )
    return PlanResult(
        valid=True,
        reason="ok",
        score=best_metrics["score"],
        operations=operations,
        target_polygons=target_polygons,
        target_rect=target_rect,
        search_nodes=nodes,
        mode=mode,
        max_vertex_error_mm=maximum_error,
        fill_gap_mm2=best_metrics["fill_gap_mm2"],
        overlap_mm2=best_metrics["overlap_mm2"],
        outside_mm2=best_metrics["outside_mm2"],
        seams=best["seams"],
        plan_stats=plan_stats,
    )


def _outer_should_try_corner_first(pieces, tolerant):
    tolerance = (
        cfg.OUTER_FIRST_TOLERANT_AXIS_TOLERANCE_DEG
        if tolerant
        else cfg.OUTER_FIRST_AXIS_TOLERANCE_DEG
    )
    counts = []
    for piece in pieces:
        counts.append(
            sum(
                1
                for angle in interior_angles(piece.polygon_mm)
                if abs(angle - 90.0) <= tolerance
            )
        )
    pieces_with_corner = sum(1 for count in counts if count)
    # One clear corner per piece is highly informative. Several rectangular
    # strip pieces each expose four right angles, however, and are faster to
    # solve by their matching seams than by enumerating corner assignments.
    return (
        pieces_with_corner >= max(2, len(pieces) - 1)
        and sum(counts) <= 2 * len(pieces)
    )


def _outer_graph_state_candidates(
    pieces,
    candidate_graph,
    placed_indices,
    placed_polygons,
    placed_transforms,
    reserved_edges,
    seam_edges,
    axis_tolerance,
    endpoint_tolerance,
    overlap_tolerance,
    slack,
    state,
    plan_started_ms,
):
    """Expand one DFS state using only the precomputed seam graph."""
    candidates = []
    candidate_seen = set()
    maximum_nodes = min(
        cfg.OUTER_FIRST_MAX_SEARCH_NODES,
        cfg.MAX_DFS_NODES,
    )
    placed_set = set(placed_indices)
    for fixed_index in placed_indices:
        fixed_polygon = placed_polygons[fixed_index]
        fixed_transform = placed_transforms[fixed_index]
        for fixed_edge in range(len(fixed_polygon)):
            if (
                (fixed_index, fixed_edge) in seam_edges
                or (fixed_index, fixed_edge) in reserved_edges
            ):
                continue
            a0 = fixed_polygon[fixed_edge]
            a1 = fixed_polygon[
                (fixed_edge + 1) % len(fixed_polygon)
            ]
            for match in candidate_graph.for_edge(
                fixed_index, fixed_edge
            ):
                new_index = match.other_piece(fixed_index)
                if new_index is None or new_index in placed_set:
                    continue
                if fixed_index == match.piece_a:
                    if match.edge_a != fixed_edge:
                        continue
                    new_edge = match.edge_b
                    relative = match.transform_b_to_a
                    relative_polygon = match.transformed_vertices_b
                else:
                    if match.edge_b != fixed_edge:
                        continue
                    new_edge = match.edge_a
                    relative = match.transform_a_to_b
                    relative_polygon = match.transformed_vertices_a
                if (new_index, new_edge) in seam_edges:
                    continue

                state["nodes"] += 1
                _geometry_exitpoint(state["nodes"])
                if state["nodes"] >= maximum_nodes:
                    state["limit_hit"] = True
                    return candidates
                if (
                    ticks_diff(ticks_ms(), plan_started_ms)
                    >= cfg.MAX_PLAN_TIME_MS
                ):
                    state["timed_out"] = True
                    state["limit_hit"] = True
                    return candidates

                transform = compose_transforms(
                    fixed_transform, relative
                )
                candidate = transform_polygon(
                    relative_polygon, fixed_transform
                )
                endpoint_error = max(
                    _distance(candidate[new_edge], a1),
                    _distance(
                        candidate[
                            (new_edge + 1) % len(candidate)
                        ],
                        a0,
                    ),
                )
                if endpoint_error > endpoint_tolerance:
                    state["pruned_endpoint"] += 1
                    continue
                fixed_center = polygon_centroid(fixed_polygon)
                new_center = polygon_centroid(candidate)
                if (
                    _cross(a0, a1, fixed_center)
                    * _cross(a0, a1, new_center)
                    >= -EPS
                ):
                    state["pruned_side"] += 1
                    continue

                overlap = 0.0
                for existing_index in placed_indices:
                    overlap += polygon_overlap_area(
                        candidate,
                        placed_polygons[existing_index],
                    )
                    if overlap > overlap_tolerance:
                        break
                if overlap > overlap_tolerance:
                    state["pruned_overlap"] += 1
                    continue

                next_seams = set(seam_edges)
                next_seams.add((fixed_index, fixed_edge))
                next_seams.add((new_index, new_edge))
                if not _outer_first_has_axis_edge(
                    candidate,
                    new_index,
                    next_seams,
                    axis_tolerance,
                ):
                    state["pruned_boundary"] += 1
                    continue
                existing_valid = True
                for existing_index in placed_indices:
                    if not _outer_first_has_axis_edge(
                        placed_polygons[existing_index],
                        existing_index,
                        next_seams,
                        axis_tolerance,
                    ):
                        existing_valid = False
                        break
                if not existing_valid:
                    state["pruned_boundary"] += 1
                    continue

                placed_polygons[new_index] = candidate
                bounds_valid = _outer_first_partial_bounds_valid(
                    placed_polygons,
                    placed_indices + [new_index],
                    slack,
                )
                placed_polygons[new_index] = None
                if not bounds_valid:
                    state["pruned_dimension"] += 1
                    continue

                key = (
                    new_index,
                    int(
                        round(
                            new_center[0]
                            / cfg.STATE_POSITION_QUANTIZATION_MM
                        )
                    ),
                    int(
                        round(
                            new_center[1]
                            / cfg.STATE_POSITION_QUANTIZATION_MM
                        )
                    ),
                    int(
                        round(
                            transform[4]
                            / cfg.STATE_ANGLE_QUANTIZATION_DEG
                        )
                    ),
                    tuple(sorted(next_seams)),
                )
                if key in candidate_seen:
                    state["pruned_duplicate"] += 1
                    continue
                candidate_seen.add(key)
                remaining_edges = [
                    edge
                    for edge in range(len(candidate))
                    if (new_index, edge) not in next_seams
                ]
                if not remaining_edges:
                    state["pruned_boundary"] += 1
                    continue
                axis_error = min(
                    _outer_first_axis_error_deg(
                        candidate[edge],
                        candidate[
                            (edge + 1) % len(candidate)
                        ],
                    )
                    for edge in remaining_edges
                )
                candidates.append(
                    {
                        "new_index": new_index,
                        "fixed_index": fixed_index,
                        "fixed_edge": fixed_edge,
                        "new_edge": new_edge,
                        "polygon": candidate,
                        "transform": transform,
                        "endpoint_error": endpoint_error,
                        "length_error": match.seam_length_error,
                        "axis_error": axis_error,
                        "candidate_cost": match.geometric_cost,
                    }
                )

    # Prefer the piece with the fewest legal continuations (MRV), while
    # retaining all candidate pieces for completeness.
    option_counts = {}
    for candidate in candidates:
        index = candidate["new_index"]
        option_counts[index] = option_counts.get(index, 0) + 1
    candidates.sort(
        key=lambda item: (
            option_counts[item["new_index"]],
            item["candidate_cost"],
            item["endpoint_error"],
            item["axis_error"],
        )
    )
    return candidates


def plan_outer_first_rectangle(
    pieces, tolerant_fallback=True, _tolerant=False
):
    """Plan an unknown 2..4-piece rectangle from its outside edges first.

    Every root hypothesis aligns one observed edge with a target rectangle
    side.  During seam attachment, every placed piece must retain an unused
    horizontal/vertical edge that can become part of the final boundary.
    This encodes the on-site guarantee and prunes the orientation-free search
    while remaining independent of piece shape, count, and rectangle size.
    """
    if (
        len(pieces) < cfg.MIN_PIECE_COUNT
        or len(pieces) > cfg.MAX_PIECE_COUNT
    ):
        return PlanResult(
            reason="piece count {} outside {}..{}".format(
                len(pieces),
                cfg.MIN_PIECE_COUNT,
                cfg.MAX_PIECE_COUNT,
            ),
            mode="outer_first",
        )

    plan_started_ms = ticks_ms()
    reset_geometry_counters()
    candidate_graph = build_edge_candidate_graph(
        pieces, tolerant=_tolerant
    )
    search_stats = {
        "pruned_endpoint": 0,
        "pruned_side": 0,
        "pruned_overlap": 0,
        "pruned_boundary": 0,
        "pruned_dimension": 0,
        "pruned_duplicate": 0,
    }

    def finalize(plan):
        stats = dict(plan.plan_stats)
        stats.update(search_stats)
        stats.update(geometry_counters_snapshot())
        stats["candidate_pair_count_raw"] = (
            candidate_graph.raw_pair_count
        )
        stats["candidate_pair_count_filtered"] = (
            candidate_graph.filtered_pair_count
        )
        stats["candidate_graph_ms"] = candidate_graph.build_ms
        stats["dfs_nodes_expanded"] = plan.search_nodes
        stats["plan_ms"] = max(
            0, ticks_diff(ticks_ms(), plan_started_ms)
        )
        plan.plan_stats = stats
        PERF_STATS.add_stage(
            "plan_ms", elapsed_ms=stats["plan_ms"]
        )
        return plan

    mode = (
        "outer_first_tolerant"
        if _tolerant
        else "outer_first_strict"
    )

    corner_attempted = _outer_should_try_corner_first(
        pieces, tolerant=_tolerant
    )
    if corner_attempted:
        corner_plan = _plan_corner_hybrid_rectangle(
            pieces,
            tolerant=_tolerant,
            candidate_graph=candidate_graph,
            plan_started_ms=plan_started_ms,
        )
        if corner_plan.valid:
            return finalize(corner_plan)

    def failed(reason, score=None, nodes=0, fill_gap=None):
        if not corner_attempted:
            corner_plan = _plan_corner_hybrid_rectangle(
                pieces,
                tolerant=_tolerant,
                candidate_graph=candidate_graph,
                plan_started_ms=plan_started_ms,
            )
            if corner_plan.valid:
                corner_plan.search_nodes += nodes
                return finalize(corner_plan)
        if (
            not _tolerant
            and tolerant_fallback
            and cfg.ENABLE_TOLERANT_FALLBACK
        ):
            return plan_outer_first_rectangle(
                pieces,
                tolerant_fallback=False,
                _tolerant=True,
            )
        return finalize(
            PlanResult(
                reason=reason,
                score=score,
                search_nodes=nodes,
                mode=mode,
                fill_gap_mm2=fill_gap,
            )
        )

    count = len(pieces)
    source_polygons = [
        piece.polygon_mm for piece in pieces
    ]
    axis_tolerance = (
        cfg.OUTER_FIRST_TOLERANT_AXIS_TOLERANCE_DEG
        if _tolerant
        else cfg.OUTER_FIRST_AXIS_TOLERANCE_DEG
    )
    endpoint_tolerance = (
        cfg.CORRESPONDING_VERTEX_TOLERANCE_MM
        if _tolerant
        else cfg.EDGE_ENDPOINT_TOLERANCE_MM
    )
    overlap_tolerance = (
        cfg.FIXED_RECT_MAX_OVERLAP_MM2
        if _tolerant
        else cfg.OVERLAP_AREA_TOLERANCE_MM2
    )
    slack = cfg.OUTER_FIRST_PARTIAL_BOUND_SLACK_MM
    if _tolerant:
        slack += cfg.CORRESPONDING_VERTEX_TOLERANCE_MM

    best = {
        "score": 1e9,
        "metrics": None,
        "polygons": None,
        "transforms": None,
        "seams": None,
    }
    state = {
        "nodes": 0,
        "limit_hit": False,
        "exact_solution": False,
        "timed_out": False,
        "pruned_endpoint": 0,
        "pruned_side": 0,
        "pruned_overlap": 0,
        "pruned_boundary": 0,
        "pruned_dimension": 0,
        "pruned_duplicate": 0,
    }
    search_stats = state
    seen = [set() for _ in range(count + 1)]
    roots = _outer_first_root_candidates(
        pieces, tolerant=_tolerant
    )

    for root_number, root in enumerate(roots):
        if state["nodes"] >= min(
            cfg.OUTER_FIRST_MAX_SEARCH_NODES,
            cfg.MAX_DFS_NODES,
        ):
            state["limit_hit"] = True
            break
        if (
            ticks_diff(ticks_ms(), plan_started_ms)
            >= cfg.MAX_PLAN_TIME_MS
        ):
            state["timed_out"] = True
            state["limit_hit"] = True
            break
        state["nodes"] += 1
        _geometry_exitpoint(state["nodes"])
        root_index = root["piece_index"]
        root_edge = root["edge_index"]
        placed_polygons = [None for _ in range(count)]
        placed_transforms = [None for _ in range(count)]
        placed_polygons[root_index] = root["polygon"]
        placed_transforms[root_index] = root["transform"]
        reserved_edges = {(root_index, root_edge)}
        seam_edges = set()
        seam_stack = []

        if not _outer_first_partial_bounds_valid(
            placed_polygons, [root_index], slack
        ):
            continue

        def recurse(placed_indices):
            if (
                state["nodes"]
                >= min(
                    cfg.OUTER_FIRST_MAX_SEARCH_NODES,
                    cfg.MAX_DFS_NODES,
                )
            ):
                state["limit_hit"] = True
                return
            if (
                ticks_diff(ticks_ms(), plan_started_ms)
                >= cfg.MAX_PLAN_TIME_MS
            ):
                state["timed_out"] = True
                state["limit_hit"] = True
                return
            if len(placed_indices) == count:
                candidate_polygons = [
                    placed_polygons[index]
                    for index in range(count)
                ]
                metrics = _outer_first_complete_metrics(
                    candidate_polygons
                )
                if metrics is None:
                    return
                boundary_tolerance = (
                    cfg.CORRESPONDING_VERTEX_TOLERANCE_MM
                    if _tolerant
                    else cfg.OUTER_EDGE_TOLERANCE_MM
                )
                if not _each_piece_has_outer_edge(
                    candidate_polygons,
                    metrics["rect"],
                    tolerance=boundary_tolerance,
                ):
                    return
                if metrics["score"] < best["score"]:
                    best["score"] = metrics["score"]
                    best["metrics"] = metrics
                    best["polygons"] = [
                        list(polygon)
                        for polygon in candidate_polygons
                    ]
                    best["transforms"] = list(
                        placed_transforms
                    )
                    best["seams"] = [
                        dict(seam) for seam in seam_stack
                    ]
                    if metrics["score"] <= 1e-6:
                        state["exact_solution"] = True
                return

            candidates = _outer_graph_state_candidates(
                pieces,
                candidate_graph,
                placed_indices,
                placed_polygons,
                placed_transforms,
                reserved_edges,
                seam_edges,
                axis_tolerance,
                endpoint_tolerance,
                overlap_tolerance,
                slack,
                state,
                plan_started_ms,
            )
            for candidate in candidates[
                : cfg.OUTER_FIRST_BRANCH_LIMIT
            ]:
                new_index = candidate["new_index"]
                fixed_index = candidate["fixed_index"]
                fixed_edge = candidate["fixed_edge"]
                new_edge = candidate["new_edge"]
                placed_polygons[new_index] = candidate["polygon"]
                placed_transforms[
                    new_index
                ] = candidate["transform"]
                seam_edges.add((fixed_index, fixed_edge))
                seam_edges.add((new_index, new_edge))
                seam_stack.append(
                    {
                        "piece_a_index": fixed_index,
                        "edge_a_index": fixed_edge,
                        "piece_b_index": new_index,
                        "edge_b_index": new_edge,
                        "endpoint_error_mm": candidate[
                            "endpoint_error"
                        ],
                        "length_error_mm": candidate[
                            "length_error"
                        ],
                    }
                )
                next_indices = placed_indices + [new_index]
                state_key = []
                for index in sorted(next_indices):
                    center = polygon_centroid(
                        placed_polygons[index]
                    )
                    state_key.extend(
                        (
                            index,
                            int(round(center[0] * 2.0)),
                            int(round(center[1] * 2.0)),
                            int(
                                round(
                                    placed_transforms[index][4]
                                    * 2.0
                                )
                            ),
                        )
                    )
                state_key.extend(
                    ("s",)
                    + tuple(sorted(seam_edges))
                )
                state_key = tuple(state_key)
                if state_key not in seen[len(next_indices)]:
                    seen[len(next_indices)].add(state_key)
                    recurse(next_indices)
                seam_stack.pop()
                seam_edges.remove((fixed_index, fixed_edge))
                seam_edges.remove((new_index, new_edge))
                placed_polygons[new_index] = None
                placed_transforms[new_index] = None
                if state["limit_hit"]:
                    return
                if state["exact_solution"]:
                    return

        recurse([root_index])
        if state["limit_hit"] or state["exact_solution"]:
            break

    if best["metrics"] is None:
        reason = (
            "outer-first search limit reached"
            if state["limit_hit"]
            else "no outside-edge rectangle assembly"
        )
        return failed(reason, nodes=state["nodes"])

    score_threshold = (
        cfg.TOLERANT_RECTANGLE_SCORE_THRESHOLD
        if _tolerant
        else cfg.RECTANGLE_SCORE_THRESHOLD
    )
    if best["score"] > score_threshold:
        return failed(
            "best outer-first score {:.4f} exceeds {:.4f}".format(
                best["score"], score_threshold
            ),
            score=best["score"],
            nodes=state["nodes"],
            fill_gap=best["metrics"]["fill_gap_mm2"],
        )
    gap_tolerance = cfg.RECT_FILL_GAP_TOLERANCE_MM2
    if _tolerant:
        gap_tolerance = (
            cfg.TOLERANT_MAX_FILL_GAP_RATIO
            * best["metrics"]["rect"]["area"]
        )
    if best["metrics"]["fill_gap_mm2"] > gap_tolerance:
        return failed(
            "best outer-first gap {:.1f} exceeds {:.1f}".format(
                best["metrics"]["fill_gap_mm2"],
                gap_tolerance,
            ),
            score=best["score"],
            nodes=state["nodes"],
            fill_gap=best["metrics"]["fill_gap_mm2"],
        )

    operations, target_polygons, target_rect = (
        _place_assembly_on_a4(
            pieces,
            best["polygons"],
            best["transforms"],
            best["metrics"]["rect"],
        )
    )
    if operations is None:
        return failed(
            "target rectangle does not fit A4 lower region",
            score=best["score"],
            nodes=state["nodes"],
            fill_gap=best["metrics"]["fill_gap_mm2"],
        )
    maximum_error = 0.0
    if best["seams"]:
        maximum_error = max(
            seam["endpoint_error_mm"]
            for seam in best["seams"]
        )
    return finalize(
        PlanResult(
            valid=True,
            reason="ok",
            score=best["score"],
            operations=operations,
            target_polygons=target_polygons,
            target_rect=target_rect,
            search_nodes=state["nodes"],
            mode=mode,
            max_vertex_error_mm=maximum_error,
            fill_gap_mm2=best["metrics"]["fill_gap_mm2"],
            overlap_mm2=best["metrics"]["overlap_mm2"],
            outside_mm2=0.0,
            seams=best["seams"],
        )
    )


def plan_rectangle_assembly(
    pieces, tolerant_fallback=True, _tolerant=False
):
    """Search edge attachments and place a valid result in the A4 lower half."""
    if len(pieces) < cfg.MIN_PIECE_COUNT or len(pieces) > cfg.MAX_PIECE_COUNT:
        return PlanResult(
            reason="piece count {} outside {}..{}".format(
                len(pieces), cfg.MIN_PIECE_COUNT, cfg.MAX_PIECE_COUNT
            )
        )

    if (
        not _tolerant
        and cfg.TARGET_RECT_SIZE_MM is not None
    ):
        fixed_plan = _plan_fixed_rectangle(
            pieces, cfg.TARGET_RECT_SIZE_MM
        )
        if fixed_plan.valid:
            return fixed_plan

    mode = "tolerant" if _tolerant else "strict"

    def failed(reason, score=None, nodes=0, fill_gap=None):
        if (
            not _tolerant
            and tolerant_fallback
            and cfg.ENABLE_TOLERANT_FALLBACK
        ):
            return plan_rectangle_assembly(
                pieces,
                tolerant_fallback=False,
                _tolerant=True,
            )
        return PlanResult(
            reason=reason,
            score=score,
            search_nodes=nodes,
            mode=mode,
            fill_gap_mm2=fill_gap,
        )

    count = len(pieces)
    polygons = [piece.polygon_mm for piece in pieces]
    placed_polygons = [None for _ in range(count)]
    placed_transforms = [None for _ in range(count)]
    placed_polygons[0] = list(polygons[0])
    placed_transforms[0] = _identity_transform()
    used_edges = set()
    best = {
        "score": 1e9,
        "details": None,
        "polygons": None,
        "transforms": None,
        "seams": None,
    }
    state = {"nodes": 0, "limit_hit": False}
    seen = [set() for _ in range(count + 1)]

    seam_stack = []

    def recurse(placed_indices):
        if state["nodes"] >= cfg.MAX_SEARCH_NODES:
            state["limit_hit"] = True
            return
        state["nodes"] += 1
        _geometry_exitpoint(state["nodes"])
        depth = len(placed_indices)
        if depth == count:
            candidate_polygons = [placed_polygons[i] for i in range(count)]
            details = _score_rectangle_details(candidate_polygons)
            outer_tolerance = (
                cfg.CORRESPONDING_VERTEX_TOLERANCE_MM
                if _tolerant
                else cfg.OUTER_EDGE_TOLERANCE_MM
            )
            if not _each_piece_has_outer_edge(
                candidate_polygons,
                details["rect"],
                tolerance=outer_tolerance,
            ):
                return
            if details["score"] < best["score"]:
                best["score"] = details["score"]
                best["details"] = details
                best["polygons"] = [list(poly) for poly in candidate_polygons]
                best["transforms"] = list(placed_transforms)
                best["seams"] = [dict(seam) for seam in seam_stack]
            return

        unplaced = [i for i in range(count) if i not in placed_indices]
        for new_index in unplaced:
            new_polygon = polygons[new_index]
            new_lengths = edge_lengths(new_polygon)
            for fixed_index in placed_indices:
                fixed_polygon = placed_polygons[fixed_index]
                fixed_lengths = edge_lengths(fixed_polygon)
                for fixed_edge in range(len(fixed_polygon)):
                    if (fixed_index, fixed_edge) in used_edges:
                        continue
                    a0 = fixed_polygon[fixed_edge]
                    a1 = fixed_polygon[(fixed_edge + 1) % len(fixed_polygon)]
                    for new_edge in range(len(new_polygon)):
                        if not _edge_length_matches(
                            fixed_lengths[fixed_edge],
                            new_lengths[new_edge],
                            tolerant=_tolerant,
                        ):
                            continue
                        b0 = new_polygon[new_edge]
                        b1 = new_polygon[(new_edge + 1) % len(new_polygon)]
                        transform = rigid_transform_from_edge_pair((a0, a1), (b0, b1))
                        candidate = transform_polygon(new_polygon, transform)
                        mapped_b0 = candidate[new_edge]
                        mapped_b1 = candidate[(new_edge + 1) % len(candidate)]
                        endpoint_error_0 = _distance(mapped_b0, a1)
                        endpoint_error_1 = _distance(mapped_b1, a0)
                        endpoint_error = max(
                            endpoint_error_0, endpoint_error_1
                        )
                        endpoint_tolerance = (
                            cfg.CORRESPONDING_VERTEX_TOLERANCE_MM
                            if _tolerant
                            else cfg.EDGE_ENDPOINT_TOLERANCE_MM
                        )
                        if endpoint_error > endpoint_tolerance:
                            continue

                        fixed_center = polygon_centroid(fixed_polygon)
                        new_center = polygon_centroid(candidate)
                        side_fixed = _cross(a0, a1, fixed_center)
                        side_new = _cross(a0, a1, new_center)
                        if side_fixed * side_new >= -EPS:
                            continue

                        invalid_overlap = False
                        for existing_index in placed_indices:
                            if polygon_overlap_area(
                                candidate, placed_polygons[existing_index]
                            ) > cfg.OVERLAP_AREA_TOLERANCE_MM2:
                                invalid_overlap = True
                                break
                        if invalid_overlap:
                            continue

                        center = polygon_centroid(candidate)
                        key = (
                            new_index,
                            int(round(center[0] * 2.0)),
                            int(round(center[1] * 2.0)),
                            int(round(transform[4] * 2.0)),
                            tuple(sorted(used_edges)),
                        )
                        if key in seen[depth + 1]:
                            continue
                        seen[depth + 1].add(key)

                        placed_polygons[new_index] = candidate
                        placed_transforms[new_index] = transform
                        used_edges.add((fixed_index, fixed_edge))
                        used_edges.add((new_index, new_edge))
                        seam_stack.append(
                            {
                                "piece_a_index": fixed_index,
                                "edge_a_index": fixed_edge,
                                "piece_b_index": new_index,
                                "edge_b_index": new_edge,
                                "endpoint_error_mm": endpoint_error,
                                "length_error_mm": abs(
                                    fixed_lengths[fixed_edge]
                                    - new_lengths[new_edge]
                                ),
                            }
                        )
                        recurse(placed_indices + [new_index])
                        seam_stack.pop()
                        used_edges.remove((fixed_index, fixed_edge))
                        used_edges.remove((new_index, new_edge))
                        placed_polygons[new_index] = None
                        placed_transforms[new_index] = None

    recurse([0])
    if best["details"] is None:
        reason = "search limit reached" if state["limit_hit"] else "no compatible assembly"
        return failed(reason, nodes=state["nodes"])
    score_threshold = (
        cfg.TOLERANT_RECTANGLE_SCORE_THRESHOLD
        if _tolerant
        else cfg.RECTANGLE_SCORE_THRESHOLD
    )
    if best["score"] > score_threshold:
        return failed(
            "best rectangle score {:.4f} exceeds {:.4f}".format(
                best["score"], score_threshold
            ),
            score=best["score"],
            nodes=state["nodes"],
            fill_gap=best["details"]["fill_gap_mm2"],
        )
    gap_tolerance = cfg.RECT_FILL_GAP_TOLERANCE_MM2
    if _tolerant:
        gap_tolerance = (
            cfg.TOLERANT_MAX_FILL_GAP_RATIO
            * best["details"]["rect"]["area"]
        )
    if best["details"]["fill_gap_mm2"] > gap_tolerance:
        return failed(
            "best assembly gap {:.1f} mm2 exceeds {:.1f}".format(
                best["details"]["fill_gap_mm2"], gap_tolerance
            ),
            score=best["score"],
            nodes=state["nodes"],
            fill_gap=best["details"]["fill_gap_mm2"],
        )

    operations, targets, target_rect = _place_assembly_on_a4(
        pieces, best["polygons"], best["transforms"], best["details"]["rect"]
    )
    if operations is None:
        return failed(
            "target rectangle does not fit A4 lower region",
            score=best["score"],
            nodes=state["nodes"],
            fill_gap=best["details"]["fill_gap_mm2"],
        )
    max_vertex_error = 0.0
    if best["seams"]:
        max_vertex_error = max(
            seam["endpoint_error_mm"] for seam in best["seams"]
        )
    return PlanResult(
        valid=True,
        reason="ok",
        score=best["score"],
        operations=operations,
        target_polygons=targets,
        target_rect=target_rect,
        search_nodes=state["nodes"],
        mode=mode,
        max_vertex_error_mm=max_vertex_error,
        fill_gap_mm2=best["details"]["fill_gap_mm2"],
        overlap_mm2=best["details"]["overlap_mm2"],
        seams=best["seams"],
    )


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
    """Maintain shape-aware stable IDs and per-piece motion state."""

    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.last_count = None

    def _new_track(self, observation):
        piece_id = "P{}".format(self.next_id)
        self.next_id += 1
        observation.piece_id = piece_id
        track = _Track(piece_id, observation)
        self.tracks.append(track)
        return track

    def update(self, observations):
        count_changed = (
            self.last_count is not None
            and self.last_count != len(observations)
        )
        self.last_count = len(observations)
        active = [
            track
            for track in self.tracks
            if track.missed <= cfg.TRACK_MAX_MISSED_FRAMES
        ]
        if count_changed:
            for track in active:
                track.history = []
                track.samples = []
                track.stable = False
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
                track.stable = False
                track.history = []
                track.samples = []

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
                track.stable = (
                    center_spread <= cfg.CENTER_STABLE_TOLERANCE_MM
                    and angle_spread <= cfg.ANGLE_STABLE_TOLERANCE_DEG
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
                self._new_track(observation)

        for index, observation in enumerate(observations):
            representative = stable_representatives.get(
                observation.piece_id
            )
            if representative is not None:
                observations[index] = representative
        observations.sort(key=lambda observation: int(observation.piece_id[1:]))
        all_stable = (
            not count_changed
            and
            cfg.MIN_PIECE_COUNT <= len(observations) <= cfg.MAX_PIECE_COUNT
            and all(observation.stable for observation in observations)
        )
        return observations, all_stable
