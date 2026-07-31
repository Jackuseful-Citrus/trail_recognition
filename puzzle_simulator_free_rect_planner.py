"""Experimental free-size rectangle planner for one to four white pieces.

This module is intentionally isolated from ``puzzle_simulator_planner``.  It
reuses that backend's edge-match representation and rigid pose optimizer, but
owns its candidate shortlist, connected-tree search, time budget, scoring, and
fail-closed physical publication gates.  Overlap, gap, outside area, and
rectangle dimensions are never prefix hard gates; they are checked only on
exact complete proposals.
"""

import math
import os

import puzzle_config as cfg
from puzzle_geometry import (
    EPS,
    PlanResult,
    _choose_smallest_equivalent_rotation,
    _identity_transform,
    compose_transforms,
    convex_hull,
    minimum_area_rectangle,
    normalize_angle_deg,
    plan_debug_heartbeat,
    polygon_area,
    polygon_aabb,
    polygon_centroid,
    polygon_is_convex,
    polygon_overlap_area,
    triangulate_simple_polygon,
    transform_point,
    transform_polygon,
    update_plan_debug,
)
from puzzle_perf import PERF_STATS, ticks_diff, ticks_ms
from puzzle_simulator_planner import (
    _sim_align_edge,
    _sim_align_segment_midpoint,
    _sim_is_full_match,
    _sim_match_segments,
    _sim_optimize_pose_graph,
    simulator_candidate_matchings,
)


MODE = "simulator_free_rect_publish_best"
FIGURE2_DIRECT_MODE = "simulator_free_rect_figure2_direct"
FIGURE2_TEMPLATE_ORDER = (
    "TOP_LEFT",
    "RIGHT_TRIANGLE",
    "MIDDLE_LEFT",
    "BOTTOM_LEFT",
)
FIGURE2_TEMPLATE_POLYGONS = {
    "TOP_LEFT": (
        (0.0, 0.0),
        (20.0, 0.0),
        (36.0, 12.0),
        (0.0, 20.0),
    ),
    "RIGHT_TRIANGLE": (
        (20.0, 0.0),
        (100.0, 0.0),
        (100.0, 60.0),
    ),
    "MIDDLE_LEFT": (
        (0.0, 20.0),
        (36.0, 12.0),
        (76.0, 42.0),
        (0.0, 30.0),
    ),
    "BOTTOM_LEFT": (
        (0.0, 30.0),
        (76.0, 42.0),
        (100.0, 60.0),
        (0.0, 60.0),
    ),
}
FIGURE2_TEMPLATE_AREA_RATIOS = {
    "TOP_LEFT": 0.08,
    "RIGHT_TRIANGLE": 0.40,
    "MIDDLE_LEFT": 0.18,
    "BOTTOM_LEFT": 0.34,
}
FIGURE2_RECT_WIDTH_MM = 100.0
FIGURE2_RECT_HEIGHT_MM = 60.0
FIGURE2_RECT_AREA_MM2 = 6000.0


def _free_distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


def _free_edges(polygon):
    return [
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]


def _free_signed_area_twice(polygon):
    total = 0.0
    for index, point in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        total += (
            point[0] * following[1]
            - following[0] * point[1]
        )
    return total


def _free_area_equivalent_perimeter(area_mm2, aspect_ratio):
    """Return the perimeter of a rectangle with the given area/aspect."""
    aspect_ratio = max(1.0, float(aspect_ratio))
    root_area = math.sqrt(max(EPS, float(area_mm2)))
    root_aspect = math.sqrt(aspect_ratio)
    return 2.0 * root_area * (
        root_aspect + 1.0 / root_aspect
    )


def _free_dimension_ranges():
    return (
        float(getattr(cfg, "FREE_RECT_LONG_SIDE_MIN_MM", 90.0)),
        float(getattr(cfg, "FREE_RECT_LONG_SIDE_MAX_MM", 120.0)),
        float(getattr(cfg, "FREE_RECT_SHORT_SIDE_MIN_MM", 50.0)),
        float(getattr(cfg, "FREE_RECT_SHORT_SIDE_MAX_MM", 90.0)),
    )


def _free_feasible_perimeter_range(source_area_mm2):
    long_min, long_max, short_min, short_max = (
        _free_dimension_ranges()
    )
    area = max(EPS, float(source_area_mm2))
    feasible_short_min = max(short_min, area / max(EPS, long_max))
    feasible_short_max = min(short_max, area / max(EPS, long_min))
    if feasible_short_min <= feasible_short_max:
        candidates = [feasible_short_min, feasible_short_max]
        square_root = math.sqrt(area)
        candidates.append(
            max(
                feasible_short_min,
                min(feasible_short_max, square_root),
            )
        )
        perimeters = [
            2.0 * (short_side + area / short_side)
            for short_side in candidates
        ]
        return min(perimeters), max(perimeters)
    # The measured total area itself is outside the legal size envelope.  The
    # nearest range corners still provide a stable soft perimeter reference.
    perimeters = [
        2.0 * (long_side + short_side)
        for long_side in (long_min, long_max)
        for short_side in (short_min, short_max)
    ]
    return min(perimeters), max(perimeters)


def _free_dimension_prior(
    source_area_mm2, long_side, short_side
):
    long_min, long_max, short_min, short_max = (
        _free_dimension_ranges()
    )
    area = max(EPS, float(source_area_mm2))
    area_error = abs(long_side * short_side - area) / area
    range_penalty = _free_interval_penalty(
        long_side, long_min, long_max
    ) + _free_interval_penalty(
        short_side, short_min, short_max
    )
    feasible_short_min = max(short_min, area / max(EPS, long_max))
    feasible_short_max = min(short_max, area / max(EPS, long_min))
    nearest_penalty = 0.0
    if feasible_short_min <= feasible_short_max:
        candidate_shorts = (
            feasible_short_min,
            feasible_short_max,
            max(
                feasible_short_min,
                min(feasible_short_max, short_side),
            ),
        )
        nearest_penalty = min(
            (
                (long_side - area / candidate_short)
                / max(EPS, long_max)
            ) ** 2
            + (
                (short_side - candidate_short)
                / max(EPS, short_max)
            ) ** 2
            for candidate_short in candidate_shorts
        )
    else:
        nearest_penalty = range_penalty + area_error * area_error
    return {
        "area_prior_error": area_error,
        "dimension_range_penalty": range_penalty + nearest_penalty,
        "nearest_feasible_dimension_penalty": nearest_penalty,
    }


def _free_perimeter_context(polygons, source_area_mm2):
    """Cache source-edge facts reused by every assembly perimeter check."""
    edge_lengths = []
    winding_signs = []
    source_perimeter = 0.0
    for polygon in polygons:
        lengths = []
        for edge in _free_edges(polygon):
            length = _free_distance(edge[0], edge[1])
            lengths.append(length)
            source_perimeter += length
        edge_lengths.append(lengths)
        winding_signs.append(
            1.0
            if _free_signed_area_twice(polygon) >= 0.0
            else -1.0
        )

    expected_minimum, expected_maximum = (
        _free_feasible_perimeter_range(source_area_mm2)
    )
    return {
        "edge_lengths": edge_lengths,
        "winding_signs": winding_signs,
        "source_perimeter_mm": source_perimeter,
        "expected_perimeter_min_mm": expected_minimum,
        "expected_perimeter_max_mm": expected_maximum,
    }


def _free_add_centered_coverage(
    coverage,
    key,
    lower,
    upper,
    shared_length,
):
    first = float(lower)
    second = float(upper)
    lower = min(first, second)
    upper = max(first, second)
    available = max(0.0, upper - lower)
    used = min(available, max(0.0, float(shared_length)))
    if used <= EPS:
        return
    middle = 0.5 * (lower + upper)
    coverage.setdefault(key, []).append(
        (middle - 0.5 * used, middle + 0.5 * used)
    )


def _free_merged_interval_length(intervals):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    lower, upper = intervals[0]
    total = 0.0
    for next_lower, next_upper in intervals[1:]:
        if next_lower <= upper + EPS:
            upper = max(upper, next_upper)
        else:
            total += max(0.0, upper - lower)
            lower, upper = next_lower, next_upper
    return total + max(0.0, upper - lower)


def _free_exposed_perimeter_metrics(
    polygons,
    matches,
    transforms,
    source_area_mm2,
    context=None,
):
    """Estimate assembled exposed boundary without a polygon union.

    Selected seam fractions seed one-dimensional coverage on their two source
    edges.  A tiny all-edge scan then finds extra closing seams implied by the
    assembled poses (for example, the fourth side of a cyclic four-piece
    topology).  Coverage intervals are merged per physical edge before they
    are subtracted from the immutable sum of piece perimeters.
    """
    if context is None:
        context = _free_perimeter_context(
            polygons, source_area_mm2
        )
    assembled = [
        transform_polygon(polygon, transform)
        for polygon, transform in zip(polygons, transforms)
    ]
    coverage = {}
    selected_pairs = set()
    selected_shared_mm = 0.0
    edge_lengths = context["edge_lengths"]
    for match in matches:
        _, i, edge_i, j, edge_j, ia0, ia1, ja0, ja1 = match
        length_a = edge_lengths[i][edge_i]
        length_b = edge_lengths[j][edge_j]
        lower_a = min(ia0, ia1) * length_a
        upper_a = max(ia0, ia1) * length_a
        lower_b = min(ja0, ja1) * length_b
        upper_b = max(ja0, ja1) * length_b
        shared = min(upper_a - lower_a, upper_b - lower_b)
        _free_add_centered_coverage(
            coverage,
            (i, edge_i),
            lower_a,
            upper_a,
            shared,
        )
        _free_add_centered_coverage(
            coverage,
            (j, edge_j),
            lower_b,
            upper_b,
            shared,
        )
        selected_shared_mm += max(0.0, shared)
        selected_pairs.add(
            tuple(sorted(((i, edge_i), (j, edge_j))))
        )

    indexed_edges = []
    for piece_index, polygon in enumerate(assembled):
        winding = context["winding_signs"][piece_index]
        for edge_index, edge in enumerate(_free_edges(polygon)):
            a, b = edge
            length = edge_lengths[piece_index][edge_index]
            if length <= EPS:
                continue
            indexed_edges.append(
                (
                    piece_index,
                    edge_index,
                    a,
                    b,
                    length,
                    (b[0] - a[0]) / length,
                    (b[1] - a[1]) / length,
                    winding,
                )
            )

    distance_tolerance = max(
        0.0,
        float(
            getattr(
                cfg,
                "FREE_RECT_PERIMETER_SEAM_DISTANCE_MM",
                5.0,
            )
        ),
    )
    angle_tolerance = max(
        0.0,
        min(
            89.0,
            float(
                getattr(
                    cfg,
                    "FREE_RECT_PERIMETER_SEAM_ANGLE_DEG",
                    12.0,
                )
            ),
        ),
    )
    minimum_contact = max(
        EPS,
        float(
            getattr(
                cfg,
                "FREE_RECT_PERIMETER_MIN_CONTACT_MM",
                4.0,
            )
        ),
    )
    parallel_cosine = math.cos(math.radians(angle_tolerance))
    additional_contact_count = 0
    additional_shared_mm = 0.0
    for left in range(len(indexed_edges)):
        (
            i,
            edge_i,
            a,
            b,
            length_a,
            ax,
            ay,
            winding_a,
        ) = indexed_edges[left]
        for right in range(left + 1, len(indexed_edges)):
            (
                j,
                edge_j,
                c,
                d,
                length_b,
                bx,
                by,
                winding_b,
            ) = indexed_edges[right]
            if i == j:
                continue
            pair = tuple(sorted(((i, edge_i), (j, edge_j))))
            if pair in selected_pairs:
                continue
            # Canonicalize both contours to the same winding. Adjacent
            # physical boundaries must then run in opposite directions.
            if (
                (ax * bx + ay * by) * winding_a * winding_b
                > -parallel_cosine
            ):
                continue
            normal_ax = -ay
            normal_ay = ax
            normal_bx = -by
            normal_by = bx
            line_distances = (
                abs(
                    (c[0] - a[0]) * normal_ax
                    + (c[1] - a[1]) * normal_ay
                ),
                abs(
                    (d[0] - a[0]) * normal_ax
                    + (d[1] - a[1]) * normal_ay
                ),
                abs(
                    (a[0] - c[0]) * normal_bx
                    + (a[1] - c[1]) * normal_by
                ),
                abs(
                    (b[0] - c[0]) * normal_bx
                    + (b[1] - c[1]) * normal_by
                ),
            )
            if max(line_distances) > distance_tolerance:
                continue
            projected_b = (
                (c[0] - a[0]) * ax + (c[1] - a[1]) * ay,
                (d[0] - a[0]) * ax + (d[1] - a[1]) * ay,
            )
            lower_a = max(0.0, min(projected_b))
            upper_a = min(length_a, max(projected_b))
            projected_a = (
                (a[0] - c[0]) * bx + (a[1] - c[1]) * by,
                (b[0] - c[0]) * bx + (b[1] - c[1]) * by,
            )
            lower_b = max(0.0, min(projected_a))
            upper_b = min(length_b, max(projected_a))
            shared = min(
                upper_a - lower_a, upper_b - lower_b
            )
            if shared < minimum_contact:
                continue
            _free_add_centered_coverage(
                coverage,
                (i, edge_i),
                lower_a,
                upper_a,
                shared,
            )
            _free_add_centered_coverage(
                coverage,
                (j, edge_j),
                lower_b,
                upper_b,
                shared,
            )
            additional_contact_count += 1
            additional_shared_mm += shared

    covered_perimeter = sum(
        _free_merged_interval_length(intervals)
        for intervals in coverage.values()
    )
    exposed_perimeter = max(
        0.0,
        context["source_perimeter_mm"] - covered_perimeter,
    )
    expected_minimum = context["expected_perimeter_min_mm"]
    expected_maximum = context["expected_perimeter_max_mm"]
    deficit = max(0.0, expected_minimum - exposed_perimeter)
    excess = max(0.0, exposed_perimeter - expected_maximum)
    deficit_ratio = deficit / max(EPS, expected_minimum)
    excess_ratio = excess / max(EPS, expected_maximum)
    return {
        "source_perimeter_mm": context["source_perimeter_mm"],
        "covered_perimeter_mm": covered_perimeter,
        "internal_seam_length_mm": 0.5 * covered_perimeter,
        "selected_shared_length_mm": selected_shared_mm,
        "additional_shared_length_mm": additional_shared_mm,
        "additional_contact_count": additional_contact_count,
        "exposed_perimeter_mm": exposed_perimeter,
        "expected_perimeter_min_mm": expected_minimum,
        "expected_perimeter_max_mm": expected_maximum,
        "perimeter_deficit_ratio": deficit_ratio,
        "perimeter_excess_ratio": excess_ratio,
        "perimeter_error_ratio": deficit_ratio + excess_ratio,
    }


def _free_finite(value):
    value = float(value)
    return value == value and abs(value) < 1e300


def _free_transform_valid(transform):
    return (
        transform is not None
        and len(transform) >= 5
        and all(_free_finite(value) for value in transform[:5])
    )


def _free_polygon_valid(polygon):
    if polygon is None or len(polygon) < 3:
        return False
    for point in polygon:
        if (
            len(point) < 2
            or not _free_finite(point[0])
            or not _free_finite(point[1])
        ):
            return False
    return polygon_area(polygon) > EPS


def _free_figure2_layout_polygons(layout):
    if layout == "NORMAL":
        return FIGURE2_TEMPLATE_POLYGONS
    return {
        role: tuple(
            (
                FIGURE2_RECT_WIDTH_MM - point[0],
                point[1],
            )
            for point in FIGURE2_TEMPLATE_POLYGONS[role]
        )
        for role in FIGURE2_TEMPLATE_ORDER
    }


def _free_figure2_fit(observed, template, scale):
    """Fit one known contour with rotation and translation only.

    ``scale`` normalizes small A4 calibration size error for similarity
    measurement.  It is deliberately not included in the returned motion.
    """
    if len(observed) != len(template):
        return None
    count = len(template)
    observed_center = (
        sum(float(point[0]) for point in observed) / count,
        sum(float(point[1]) for point in observed) / count,
    )
    template_center = (
        sum(float(point[0]) for point in template) / count,
        sum(float(point[1]) for point in template) / count,
    )
    centered_observed = [
        (
            (float(point[0]) - observed_center[0]) * scale,
            (float(point[1]) - observed_center[1]) * scale,
        )
        for point in observed
    ]
    centered_template = [
        (
            float(point[0]) - template_center[0],
            float(point[1]) - template_center[1],
        )
        for point in template
    ]
    best = None
    for direction in (1, -1):
        for offset in range(count):
            ordered = [
                centered_observed[
                    (offset + direction * index) % count
                ]
                for index in range(count)
            ]
            dot = 0.0
            cross = 0.0
            for source, target in zip(
                ordered, centered_template
            ):
                dot += (
                    source[0] * target[0]
                    + source[1] * target[1]
                )
                cross += (
                    source[0] * target[1]
                    - source[1] * target[0]
                )
            angle = math.atan2(cross, dot)
            cosine = math.cos(angle)
            sine = math.sin(angle)
            squared = []
            for source, target in zip(
                ordered, centered_template
            ):
                dx = (
                    cosine * source[0]
                    - sine * source[1]
                    - target[0]
                )
                dy = (
                    sine * source[0]
                    + cosine * source[1]
                    - target[1]
                )
                squared.append(dx * dx + dy * dy)
            rms_mm = math.sqrt(sum(squared) / count)
            max_mm = math.sqrt(max(squared))
            key = (rms_mm, max_mm, direction, offset)
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "rms_mm": rms_mm,
                    "max_mm": max_mm,
                    "rotation_deg": normalize_angle_deg(
                        math.degrees(angle)
                    ),
                    "direction": direction,
                    "offset": offset,
                    "observed_center": observed_center,
                    "template_center": template_center,
                }
    return best


def _free_figure2_short_edge_variant(template):
    """Return the known Figure-2 contour after its 10 mm edge is merged."""
    count = len(template)
    index = min(
        range(count),
        key=lambda value: _free_distance(
            template[value], template[(value + 1) % count]
        ),
    )
    following = (index + 1) % count
    midpoint = (
        0.5 * (template[index][0] + template[following][0]),
        0.5 * (template[index][1] + template[following][1]),
    )
    if following == 0:
        return tuple([midpoint] + list(template[1:index]))
    return tuple(
        list(template[:index])
        + [midpoint]
        + list(template[following + 1 :])
    )


def _free_figure2_permutations(values):
    """Yield the 24 four-item assignments without depending on itertools."""
    for first in range(4):
        for second in range(4):
            if second == first:
                continue
            for third in range(4):
                if third == first or third == second:
                    continue
                for fourth in range(4):
                    if fourth in (first, second, third):
                        continue
                    yield (
                        values[first],
                        values[second],
                        values[third],
                        values[fourth],
                    )


def _free_figure2_assignment(pieces, source_area):
    if len(pieces) != 4:
        return None, "piece_count_not_four", None
    indexed = list(enumerate(pieces))
    triangles = [
        item for item in indexed
        if len(item[1].polygon_mm) == 3
    ]
    quads = [
        item for item in indexed
        if len(item[1].polygon_mm) == 4
    ]
    if len(triangles) != 1 or len(quads) != 3:
        return None, "expected_one_triangle_three_quads", None
    quads.sort(
        key=lambda item: (
            float(item[1].area_mm2),
            str(getattr(item[1], "piece_id", "")),
            item[0],
        )
    )
    assignment = {
        "TOP_LEFT": quads[0],
        "RIGHT_TRIANGLE": triangles[0],
        "MIDDLE_LEFT": quads[1],
        "BOTTOM_LEFT": quads[2],
    }
    ratio_errors = {}
    for role in FIGURE2_TEMPLATE_ORDER:
        ratio_errors[role] = abs(
            float(assignment[role][1].area_mm2)
            / source_area
            - FIGURE2_TEMPLATE_AREA_RATIOS[role]
        )
    maximum = max(ratio_errors.values())
    tolerance = float(
        getattr(
            cfg,
            "FREE_RECT_FIGURE2_AREA_RATIO_TOLERANCE",
            0.06,
        )
    )
    if maximum > tolerance:
        return (
            None,
            "area_ratio_error_{:.4f}_over_{:.4f}".format(
                maximum, tolerance
            ),
            ratio_errors,
        )
    return assignment, None, ratio_errors


def _free_figure2_match(pieces, source_area):
    if not bool(
        getattr(cfg, "FREE_RECT_FIGURE2_DIRECT_ENABLED", True)
    ):
        return None, "disabled"
    assignment, reason, ratio_errors = (
        _free_figure2_assignment(pieces, source_area)
    )
    if assignment is None:
        return None, reason
    scale = math.sqrt(FIGURE2_RECT_AREA_MM2 / source_area)
    layouts = []
    for layout in ("NORMAL", "MIRROR_X"):
        templates = _free_figure2_layout_polygons(layout)
        fits = {}
        for role in FIGURE2_TEMPLATE_ORDER:
            fits[role] = _free_figure2_fit(
                assignment[role][1].polygon_mm,
                templates[role],
                scale,
            )
        maximum_rms = max(
            fits[role]["rms_mm"]
            for role in FIGURE2_TEMPLATE_ORDER
        )
        maximum_vertex = max(
            fits[role]["max_mm"]
            for role in FIGURE2_TEMPLATE_ORDER
        )
        total_rms = sum(
            fits[role]["rms_mm"]
            for role in FIGURE2_TEMPLATE_ORDER
        )
        layouts.append(
            {
                "layout": layout,
                "templates": templates,
                "fits": fits,
                "maximum_rms_mm": maximum_rms,
                "maximum_vertex_mm": maximum_vertex,
                "total_rms_mm": total_rms,
            }
        )
    layouts.sort(
        key=lambda item: (
            item["maximum_rms_mm"],
            item["maximum_vertex_mm"],
            item["total_rms_mm"],
            item["layout"],
        )
    )
    match = layouts[0]
    rms_tolerance = float(
        getattr(
            cfg,
            "FREE_RECT_FIGURE2_RMS_TOLERANCE_MM",
            6.0,
        )
    )
    vertex_tolerance = float(
        getattr(
            cfg,
            "FREE_RECT_FIGURE2_MAX_VERTEX_TOLERANCE_MM",
            10.0,
        )
    )
    if (
        match["maximum_rms_mm"] > rms_tolerance
        or match["maximum_vertex_mm"] > vertex_tolerance
    ):
        return (
            None,
            "shape_error_rms_{:.2f}_max_{:.2f}".format(
                match["maximum_rms_mm"],
                match["maximum_vertex_mm"],
            ),
        )
    match["assignment"] = assignment
    match["area_ratio_errors"] = ratio_errors
    match["global_similarity_scale"] = scale
    return match, None


def _free_figure2_three_of_four_match(pieces, source_area):
    """Recover the fixed set when only its 10 mm-edge piece was simplified.

    The ordinary matcher remains authoritative whenever all four contours
    match.  This narrow fallback requires three independently valid template
    fits, then uses the known short-edge variant only to recover the pose and
    original centroid of the remaining MIDDLE_LEFT piece.
    """
    if not bool(
        getattr(cfg, "FREE_RECT_FIGURE2_DIRECT_ENABLED", True)
    ):
        return None, "disabled"
    if len(pieces) != 4 or source_area <= EPS:
        return None, "three_of_four_requires_four_pieces"

    area_tolerance = float(
        getattr(
            cfg,
            "FREE_RECT_FIGURE2_AREA_RATIO_TOLERANCE",
            0.06,
        )
    )
    rms_tolerance = float(
        getattr(
            cfg,
            "FREE_RECT_FIGURE2_RMS_TOLERANCE_MM",
            6.0,
        )
    )
    vertex_tolerance = float(
        getattr(
            cfg,
            "FREE_RECT_FIGURE2_MAX_VERTEX_TOLERANCE_MM",
            10.0,
        )
    )
    scale = math.sqrt(FIGURE2_RECT_AREA_MM2 / source_area)
    indexed = list(enumerate(pieces))
    best = None

    for ordered in _free_figure2_permutations(indexed):
        assignment = {
            role: ordered[index]
            for index, role in enumerate(FIGURE2_TEMPLATE_ORDER)
        }
        ratio_errors = {
            role: abs(
                float(assignment[role][1].area_mm2)
                / source_area
                - FIGURE2_TEMPLATE_AREA_RATIOS[role]
            )
            for role in FIGURE2_TEMPLATE_ORDER
        }
        if sum(
            1
            for role in FIGURE2_TEMPLATE_ORDER
            if ratio_errors[role] <= area_tolerance
        ) < 3:
            continue

        for layout in ("NORMAL", "MIRROR_X"):
            templates = _free_figure2_layout_polygons(layout)
            fits = {}
            matched_roles = []
            for role in FIGURE2_TEMPLATE_ORDER:
                fit = _free_figure2_fit(
                    assignment[role][1].polygon_mm,
                    templates[role],
                    scale,
                )
                fits[role] = fit
                if (
                    ratio_errors[role] <= area_tolerance
                    and fit is not None
                    and fit["rms_mm"] <= rms_tolerance
                    and fit["max_mm"] <= vertex_tolerance
                ):
                    matched_roles.append(role)

            if len(matched_roles) != 3:
                continue
            inferred_roles = [
                role
                for role in FIGURE2_TEMPLATE_ORDER
                if role not in matched_roles
            ]
            inferred_role = inferred_roles[0]
            if inferred_role != "MIDDLE_LEFT":
                continue

            inferred_fit = fits[inferred_role]
            if inferred_fit is None:
                recovery_template = _free_figure2_short_edge_variant(
                    templates[inferred_role]
                )
                inferred_fit = _free_figure2_fit(
                    assignment[inferred_role][1].polygon_mm,
                    recovery_template,
                    scale,
                )
            if inferred_fit is None:
                continue

            # Recover the true centroid that existed before the 10 mm edge was
            # collapsed.  The fit angle maps measured coordinates into the
            # fixed template, so apply its inverse to the centroid offset.
            fixed_center = polygon_centroid(
                templates[inferred_role]
            )
            fitted_center = inferred_fit["template_center"]
            dx = fixed_center[0] - fitted_center[0]
            dy = fixed_center[1] - fitted_center[1]
            angle = math.radians(inferred_fit["rotation_deg"])
            cosine = math.cos(angle)
            sine = math.sin(angle)
            observed_center = inferred_fit["observed_center"]
            inferred_fit["source_center_mm"] = (
                observed_center[0]
                + (cosine * dx + sine * dy) / scale,
                observed_center[1]
                + (-sine * dx + cosine * dy) / scale,
            )
            inferred_fit["short_edge_fit_cancelled"] = True
            fits[inferred_role] = inferred_fit

            maximum_rms = max(
                fits[role]["rms_mm"] for role in matched_roles
            )
            maximum_vertex = max(
                fits[role]["max_mm"] for role in matched_roles
            )
            total_rms = sum(
                fits[role]["rms_mm"] for role in matched_roles
            )
            key = (
                maximum_rms,
                maximum_vertex,
                total_rms,
                sum(ratio_errors[role] for role in matched_roles),
                layout,
            )
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "layout": layout,
                    "templates": templates,
                    "fits": fits,
                    "maximum_rms_mm": maximum_rms,
                    "maximum_vertex_mm": maximum_vertex,
                    "total_rms_mm": total_rms,
                    "assignment": assignment,
                    "area_ratio_errors": ratio_errors,
                    "global_similarity_scale": scale,
                    "matched_piece_count": 3,
                    "matched_roles": tuple(matched_roles),
                    "inferred_roles": (inferred_role,),
                    "short_edge_fit_cancelled": True,
                }

    if best is None:
        return None, "three_of_four_fixed_template_not_matched"
    best.pop("key", None)
    return best, None


def match_fixed_figure2_piece_set(pieces):
    """Return the reusable fixed Figure 2 match evaluation."""
    pieces = list(pieces)
    if len(pieces) != 4:
        return None, "piece_count_{}".format(len(pieces))
    source_area = sum(
        float(piece.area_mm2) for piece in pieces
    )
    if source_area <= EPS:
        return None, "source_area_not_positive"
    if any(
        not _free_polygon_valid(piece.polygon_mm)
        for piece in pieces
    ):
        return None, "invalid_source_polygon"
    match, reason = _free_figure2_match(pieces, source_area)
    if match is not None:
        match["matched_piece_count"] = 4
        match["matched_roles"] = FIGURE2_TEMPLATE_ORDER
        match["inferred_roles"] = ()
        match["short_edge_fit_cancelled"] = False
        return match, None
    recovered, recovered_reason = (
        _free_figure2_three_of_four_match(pieces, source_area)
    )
    if recovered is not None:
        return recovered, None
    return None, "{};{}".format(reason, recovered_reason)


def is_fixed_figure2_piece_set(pieces):
    """Return whether four observed polygons match the fixed Figure 2 cut."""
    match, _reason = match_fixed_figure2_piece_set(pieces)
    return match is not None


def _free_figure2_direct_result(
    pieces,
    source_area,
    match,
    state,
):
    origin_x = (
        float(cfg.TARGET_CENTER_MM[0])
        - 0.5 * FIGURE2_RECT_WIDTH_MM
    )
    origin_y = (
        float(cfg.TARGET_CENTER_MM[1])
        - 0.5 * FIGURE2_RECT_HEIGHT_MM
    )
    target_rect = (
        origin_x,
        origin_y,
        origin_x + FIGURE2_RECT_WIDTH_MM,
        origin_y + FIGURE2_RECT_HEIGHT_MM,
    )
    target_rectangle_polygon = (
        (target_rect[0], target_rect[1]),
        (target_rect[2], target_rect[1]),
        (target_rect[2], target_rect[3]),
        (target_rect[0], target_rect[3]),
    )
    operations = []
    target_polygons = {}
    target_list = []
    fit_records = []
    maximum_fit_error = 0.0
    for role in FIGURE2_TEMPLATE_ORDER:
        _index, piece = match["assignment"][role]
        fit = match["fits"][role]
        template = match["templates"][role]
        local_center = polygon_centroid(template)
        target_center = (
            origin_x + local_center[0],
            origin_y + local_center[1],
        )
        short_edge_fit_cancelled = bool(
            fit.get("short_edge_fit_cancelled", False)
        )
        source_center = fit.get(
            "source_center_mm", piece.centroid_mm
        )
        angle_deg = normalize_angle_deg(
            fit["rotation_deg"]
        )
        angle = math.radians(angle_deg)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        transform = (
            cosine,
            sine,
            target_center[0]
            - (
                cosine * source_center[0]
                - sine * source_center[1]
            ),
            target_center[1]
            - (
                sine * source_center[0]
                + cosine * source_center[1]
            ),
            angle_deg,
        )
        if short_edge_fit_cancelled:
            # The measured triangle is only the result of merging the real
            # 10 mm edge.  Restore the known four-corner target instead of
            # carrying that recognition artefact into the fixed plan.
            target_polygon = [
                (origin_x + point[0], origin_y + point[1])
                for point in template
            ]
        else:
            target_polygon = transform_polygon(
                piece.polygon_mm, transform
            )
        piece_id = getattr(
            piece, "piece_id", None
        ) or "P{}".format(_index + 1)
        target_polygons[piece_id] = target_polygon
        target_list.append(target_polygon)
        operations.append(
            {
                "piece_id": piece_id,
                "template_role": role,
                "source_center_mm": source_center,
                "target_center_mm": target_center,
                "rotation_deg": angle_deg,
                "rotation_ambiguous": bool(
                    getattr(piece, "rotation_ambiguous", False)
                ),
                "confidence": float(
                    getattr(piece, "confidence", 1.0)
                ),
            }
        )
        fit_records.append(
            {
                "template_role": role,
                "piece_id": piece_id,
                "area_ratio_error": match[
                    "area_ratio_errors"
                ][role],
                "rms_mm": fit["rms_mm"],
                "max_vertex_mm": fit["max_mm"],
                "rotation_deg": angle_deg,
                "short_edge_fit_cancelled": (
                    short_edge_fit_cancelled
                ),
            }
        )
        maximum_fit_error = max(
            maximum_fit_error, fit["max_mm"]
        )
        print(
            "FREE_FIXED_TEMPLATE_PIECE,role={},piece_id={},"
            "source_x={:.2f},source_y={:.2f},target_x={:.2f},"
            "target_y={:.2f},rotation_deg={:.2f},"
            "fit_rms_mm={:.2f},fit_max_mm={:.2f},"
            "short_edge_fit_cancelled={}".format(
                role,
                piece_id,
                source_center[0],
                source_center[1],
                target_center[0],
                target_center[1],
                angle_deg,
                fit["rms_mm"],
                fit["max_mm"],
                int(short_edge_fit_cancelled),
            )
        )

    overlap_mm2 = 0.0
    for left in range(len(target_list)):
        for right in range(left + 1, len(target_list)):
            overlap_mm2 += polygon_overlap_area(
                target_list[left], target_list[right]
            )
    inside_area = sum(
        polygon_overlap_area(
            polygon, target_rectangle_polygon
        )
        for polygon in target_list
    )
    outside_mm2 = max(0.0, source_area - inside_area)
    covered_inside_mm2 = max(
        0.0, inside_area - overlap_mm2
    )
    fill_gap_mm2 = max(
        0.0,
        FIGURE2_RECT_AREA_MM2 - covered_inside_mm2,
    )
    elapsed_ms = max(
        0, ticks_diff(ticks_ms(), state["started_ms"])
    )
    stats = _free_base_stats(state, source_area, [])
    stats.update(
        {
            "engine": "simulator-free-rectangle-figure2-direct-k230",
            "plan_ms": elapsed_ms,
            "fixed_template_matched": True,
            "fixed_template_matched_piece_count": match.get(
                "matched_piece_count", 4
            ),
            "fixed_template_inferred_roles": match.get(
                "inferred_roles", ()
            ),
            "short_edge_fit_cancelled": bool(
                match.get("short_edge_fit_cancelled", False)
            ),
            "fixed_template_layout": match["layout"],
            "enumeration_skipped": True,
            "safety_gates_applied": False,
            "global_similarity_scale": match[
                "global_similarity_scale"
            ],
            "maximum_template_rms_mm": match[
                "maximum_rms_mm"
            ],
            "maximum_template_vertex_mm": match[
                "maximum_vertex_mm"
            ],
            "maximum_area_ratio_error": max(
                match["area_ratio_errors"].values()
            ),
            "template_fits": fit_records,
            "long_side_mm": FIGURE2_RECT_WIDTH_MM,
            "short_side_mm": FIGURE2_RECT_HEIGHT_MM,
            "aspect_ratio": (
                FIGURE2_RECT_WIDTH_MM / FIGURE2_RECT_HEIGHT_MM
            ),
            "aspect_preferred": True,
            "aspect_range_penalty": 0.0,
            "actual_width_mm": FIGURE2_RECT_WIDTH_MM,
            "actual_height_mm": FIGURE2_RECT_HEIGHT_MM,
            "rect_area_mm2": FIGURE2_RECT_AREA_MM2,
            "overlap_mm2": overlap_mm2,
            "fill_gap_mm2": fill_gap_mm2,
            "outside_mm2": outside_mm2,
            "selected_match_count": 0,
            "selected_partial_match_count": 0,
            "selected_topology": "fixed_figure2_direct",
            "top_k": [],
        }
    )
    PERF_STATS.add_stage("plan_ms", elapsed_ms=elapsed_ms)
    print(
        "FREE_FIXED_TEMPLATE_BYPASS,enumeration=SKIPPED,"
        "safety_gates=SKIPPED,target_mm=100.0x60.0,"
        "target_center={:.1f}:{:.1f}".format(
            float(cfg.TARGET_CENTER_MM[0]),
            float(cfg.TARGET_CENTER_MM[1]),
        )
    )
    print(
        "FREE_FIXED_TEMPLATE_RESULT,valid=1,mode={},layout={},"
        "nodes=0,operations=4,plan_ms={}".format(
            FIGURE2_DIRECT_MODE,
            match["layout"],
            elapsed_ms,
        )
    )
    return PlanResult(
        valid=True,
        reason="fixed Figure 2 template direct plan",
        score=match["maximum_rms_mm"],
        operations=operations,
        target_polygons=target_polygons,
        target_rect=target_rect,
        search_nodes=0,
        mode=FIGURE2_DIRECT_MODE,
        max_vertex_error_mm=maximum_fit_error,
        fill_gap_mm2=fill_gap_mm2,
        overlap_mm2=overlap_mm2,
        outside_mm2=outside_mm2,
        seams=[],
        plan_stats=stats,
    )


def _free_raw_candidate_matchings(
    pieces,
    full_rel_tolerance=None,
    partial_enabled=True,
    partial_min=None,
    partial_max=None,
):
    """Generate compatible match tuples for one staged-search pass."""
    polygons = [
        piece.polygon_mm if hasattr(piece, "polygon_mm") else piece
        for piece in pieces
    ]
    indexed_edges = []
    for piece_index, polygon in enumerate(polygons):
        for edge_index, edge in enumerate(_free_edges(polygon)):
            indexed_edges.append((piece_index, edge_index, edge))

    relative_tolerance = float(
        getattr(cfg, "FREE_RECT_MATCH_REL_TOLERANCE", 0.12)
        if full_rel_tolerance is None
        else full_rel_tolerance
    )
    partial_min = float(
        getattr(cfg, "FREE_RECT_PARTIAL_MIN_RATIO", 0.22)
        if partial_min is None
        else partial_min
    )
    partial_max = float(
        getattr(cfg, "FREE_RECT_PARTIAL_MAX_RATIO", 0.88)
        if partial_max is None
        else partial_max
    )
    partial_penalty = float(
        getattr(cfg, "FREE_RECT_PARTIAL_MATCH_PENALTY", 0.15)
    )
    candidates = []
    for left in range(len(indexed_edges)):
        i, edge_i, edge_a = indexed_edges[left]
        length_a = _free_distance(edge_a[0], edge_a[1])
        if length_a <= EPS:
            continue
        for right in range(left + 1, len(indexed_edges)):
            j, edge_j, edge_b = indexed_edges[right]
            if i == j:
                continue
            length_b = _free_distance(edge_b[0], edge_b[1])
            if length_b <= EPS:
                continue
            relative_error = abs(length_a - length_b) / max(
                length_a, length_b
            )
            if relative_error < relative_tolerance:
                candidates.append(
                    (
                        relative_error,
                        i,
                        edge_i,
                        j,
                        edge_j,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                    )
                )
            if not partial_enabled:
                continue
            ratio = min(length_a, length_b) / max(
                length_a, length_b
            )
            if not partial_min <= ratio <= partial_max:
                continue
            if length_a > length_b:
                candidates.append(
                    (
                        partial_penalty,
                        i,
                        edge_i,
                        j,
                        edge_j,
                        0.0,
                        ratio,
                        0.0,
                        1.0,
                    )
                )
                candidates.append(
                    (
                        partial_penalty,
                        i,
                        edge_i,
                        j,
                        edge_j,
                        1.0 - ratio,
                        1.0,
                        0.0,
                        1.0,
                    )
                )
            else:
                candidates.append(
                    (
                        partial_penalty,
                        i,
                        edge_i,
                        j,
                        edge_j,
                        0.0,
                        1.0,
                        0.0,
                        ratio,
                    )
                )
                candidates.append(
                    (
                        partial_penalty,
                        i,
                        edge_i,
                        j,
                        edge_j,
                        0.0,
                        1.0,
                        1.0 - ratio,
                        1.0,
                    )
                )
    candidates.sort()
    return candidates


def _free_candidate_pair_key(candidate):
    return tuple(sorted((int(candidate[1]), int(candidate[3]))))


def _free_candidate_ranking_score(polygons, candidate):
    """Return an ordering score without changing the match tuple format."""
    _score, i, edge_i, j, edge_j = candidate[:5]
    edge_a = _free_edges(polygons[i])[edge_i]
    edge_b = _free_edges(polygons[j])[edge_j]
    length_a = _free_distance(edge_a[0], edge_a[1])
    length_b = _free_distance(edge_b[0], edge_b[1])
    relative_error = abs(length_a - length_b) / max(
        EPS, length_a, length_b
    )
    endpoint_error = _free_endpoint_angle_error(
        polygons, candidate
    )
    direction_penalty = 1.0 / max(1.0, length_a, length_b)
    if _sim_is_full_match(candidate):
        score = (
            relative_error
            + 0.35 * endpoint_error
            + 0.20 * direction_penalty
        )
    else:
        ratio = min(length_a, length_b) / max(
            EPS, length_a, length_b
        )
        ratio_quality = abs(ratio - 0.55)
        score = (
            float(
                getattr(
                    cfg, "FREE_RECT_PARTIAL_MATCH_PENALTY", 0.15
                )
            )
            + 0.20 * ratio_quality
            + 0.45 * endpoint_error
            + 0.20 * direction_penalty
        )
    return (
        score,
        relative_error,
        endpoint_error,
        candidate,
    )


def _free_candidate_metadata(polygons, candidate, ranking=None):
    """Cache candidate facts without changing the public tuple format."""
    _score, i, edge_i, j, edge_j = candidate[:5]
    segment_a0, segment_a1, segment_b0, segment_b1 = (
        _sim_match_segments(polygons, candidate)
    )
    length_a = _free_distance(segment_a0, segment_a1)
    length_b = _free_distance(segment_b0, segment_b1)
    return {
        "pair": _free_candidate_pair_key(candidate),
        "piece_a": i,
        "edge_a": edge_i,
        "piece_b": j,
        "edge_b": edge_j,
        "segment_a": (segment_a0, segment_a1),
        "segment_b": (segment_b0, segment_b1),
        "matched_length_mm": min(length_a, length_b),
        "full": bool(_sim_is_full_match(candidate)),
        "intervals": _free_candidate_edge_intervals(candidate),
        "relative_length_error": abs(length_a - length_b)
        / max(EPS, length_a, length_b),
        "endpoint_angle_error": _free_endpoint_angle_error(
            polygons, candidate
        ),
        "ranking": (
            _free_candidate_ranking_score(polygons, candidate)
            if ranking is None
            else ranking
        ),
    }


def _free_candidate_shortlist(
    candidates, polygons=None, return_details=False
):
    """Keep a deterministic full/partial reserve independently per pair."""
    candidates = list(candidates)
    if polygons is None:
        polygons = []

    ranking_cache = {}

    def ranking(candidate):
        if candidate not in ranking_cache:
            ranking_cache[candidate] = (
                _free_candidate_ranking_score(polygons, candidate)
                if polygons
                else (candidate[0], candidate)
            )
        return ranking_cache[candidate]

    grouped = {}
    for candidate in candidates:
        grouped.setdefault(
            _free_candidate_pair_key(candidate), []
        ).append(candidate)
    maximum_full = max(
        0, int(getattr(cfg, "FREE_RECT_PAIR_MAX_FULL", 8))
    )
    maximum_partial = max(
        0, int(getattr(cfg, "FREE_RECT_PAIR_MAX_PARTIAL", 4))
    )
    selected = []
    pair_counts = {}
    for pair in sorted(grouped):
        full = sorted(
            (
                item
                for item in grouped[pair]
                if _sim_is_full_match(item)
            ),
            key=ranking,
        )[:maximum_full]
        partial_groups = {}
        for item in grouped[pair]:
            if _sim_is_full_match(item):
                continue
            # A partial seam can align to either end of the longer edge.
            # Treat those two placements as one edge-pair hypothesis; keeping
            # only the marginally better endpoint made the shortlist flip
            # under sub-millimetre vision noise.
            partial_groups.setdefault(
                (item[1], item[2], item[3], item[4]), []
            ).append(item)
        ranked_partial_groups = []
        for edge_pair, values in partial_groups.items():
            values.sort(key=ranking)
            balanced_rank = ranking(
                values[1] if len(values) > 1 else values[0]
            )
            ranked_partial_groups.append(
                (
                    0 if len(values) > 1 else 1,
                    balanced_rank,
                    ranking(values[0]),
                    edge_pair,
                    values,
                )
            )
        ranked_partial_groups.sort(key=lambda item: item[:-1])
        partial = []
        for _paired, _balanced, _best, _edge_pair, values in (
            ranked_partial_groups
        ):
            remaining = maximum_partial - len(partial)
            if remaining <= 0:
                break
            partial.extend(values[:remaining])
        partial.sort(key=ranking)
        pair_counts[pair] = (len(full), len(partial))
        selected.extend(full)
        selected.extend(partial)
    safety_cap = max(
        1,
        int(
            getattr(
                cfg,
                "FREE_RECT_GLOBAL_CANDIDATE_SAFETY_CAP",
                96,
            )
        ),
    )
    selected.sort(
        key=lambda candidate: (
            _free_candidate_pair_key(candidate),
            0 if _sim_is_full_match(candidate) else 1,
            ranking(candidate),
            candidate,
        )
    )
    if len(selected) > safety_cap:
        selected = sorted(selected, key=ranking)[:safety_cap]
        selected.sort(
            key=lambda candidate: (
                _free_candidate_pair_key(candidate),
                0 if _sim_is_full_match(candidate) else 1,
                ranking(candidate),
                candidate,
            )
        )
        selected_set = set(selected)
        pair_counts = {}
        for pair in sorted(grouped):
            full = sum(
                1
                for item in grouped[pair]
                if item in selected_set and _sim_is_full_match(item)
            )
            partial = sum(
                1
                for item in grouped[pair]
                if item in selected_set
                and not _sim_is_full_match(item)
            )
            pair_counts[pair] = (full, partial)
    details = {
        "raw_count": len(candidates),
        "shortlisted_count": len(selected),
        "pair_counts": pair_counts,
        "candidate_pair_group_count": len(grouped),
        "candidate_cache": {
            candidate: _free_candidate_metadata(
                polygons, candidate, ranking(candidate)
            )
            for candidate in selected
        } if polygons else {},
    }
    return (selected, details) if return_details else selected


def _free_rect_candidate_matchings(
    pieces,
    full_rel_tolerance=None,
    partial_enabled=True,
    partial_min=None,
    partial_max=None,
    return_details=False,
):
    polygons = [
        piece.polygon_mm if hasattr(piece, "polygon_mm") else piece
        for piece in pieces
    ]
    raw_candidates = _free_raw_candidate_matchings(
        pieces,
        full_rel_tolerance=full_rel_tolerance,
        partial_enabled=partial_enabled,
        partial_min=partial_min,
        partial_max=partial_max,
    )
    return _free_candidate_shortlist(
        raw_candidates,
        polygons=polygons,
        return_details=return_details,
    )


def _free_labeled_spanning_trees(piece_count):
    """Return deterministic labelled trees using Prüfer sequences."""
    count = max(0, int(piece_count))
    if count <= 1:
        return [()]
    if count == 2:
        return [((0, 1),)]
    sequence_length = count - 2
    sequence_count = count ** sequence_length
    trees = []
    seen = set()
    for encoded in range(sequence_count):
        value = encoded
        sequence = []
        for _index in range(sequence_length):
            sequence.append(value % count)
            value //= count
        degree = [1] * count
        for node in sequence:
            degree[node] += 1
        edges = []
        for node in sequence:
            leaf = min(
                index
                for index in range(count)
                if degree[index] == 1
            )
            edges.append(tuple(sorted((leaf, node))))
            degree[leaf] -= 1
            degree[node] -= 1
        leaves = [
            index for index in range(count) if degree[index] == 1
        ]
        edges.append(tuple(sorted((leaves[0], leaves[1]))))
        tree = tuple(sorted(edges))
        if tree not in seen:
            seen.add(tree)
            trees.append(tree)
    trees.sort()
    return trees


def _free_candidates_by_pair(candidates):
    grouped = {}
    for candidate in candidates:
        grouped.setdefault(
            _free_candidate_pair_key(candidate), []
        ).append(candidate)

    def diversity_order(values):
        ordered = []
        left = 0
        right = len(values) - 1
        take_left = True
        while left <= right:
            if take_left:
                ordered.append(values[left])
                left += 1
            else:
                ordered.append(values[right])
                right -= 1
            take_left = not take_left
        return ordered

    for pair in grouped:
        full = [
            candidate
            for candidate in grouped[pair]
            if _sim_is_full_match(candidate)
        ]
        partial = [
            candidate
            for candidate in grouped[pair]
            if not _sim_is_full_match(candidate)
        ]
        partial_by_edges = {}
        partial_edge_order = []
        for candidate in partial:
            edge_pair = (
                candidate[1],
                candidate[2],
                candidate[3],
                candidate[4],
            )
            if edge_pair not in partial_by_edges:
                partial_by_edges[edge_pair] = []
                partial_edge_order.append(edge_pair)
            partial_by_edges[edge_pair].append(candidate)
        partial = []
        for edge_pair in partial_edge_order:
            partial.extend(
                diversity_order(partial_by_edges[edge_pair])
            )
        grouped[pair] = (
            diversity_order(full) + partial
        )
    return grouped


def _free_candidate_edge_intervals(candidate):
    _score, i, edge_i, j, edge_j, ia0, ia1, ja0, ja1 = candidate
    return (
        ((i, edge_i), (min(ia0, ia1), max(ia0, ia1))),
        ((j, edge_j), (min(ja0, ja1), max(ja0, ja1))),
    )


def _free_add_candidate_intervals(used_intervals, candidate):
    tolerance = max(
        0.0,
        float(
            getattr(
                cfg,
                "FREE_RECT_EDGE_INTERVAL_OVERLAP_TOLERANCE",
                0.03,
            )
        ),
    )
    next_intervals = {
        key: list(values) for key, values in used_intervals.items()
    }
    for key, interval in _free_candidate_edge_intervals(candidate):
        for existing in next_intervals.get(key, ()):
            overlap = min(interval[1], existing[1]) - max(
                interval[0], existing[0]
            )
            if overlap > tolerance:
                return None
        next_intervals.setdefault(key, []).append(interval)
        next_intervals[key].sort()
    return next_intervals


def _free_deadline_reached(deadline_ms):
    if isinstance(deadline_ms, (tuple, list)):
        return ticks_diff(ticks_ms(), deadline_ms[0]) >= int(
            deadline_ms[1]
        )
    return ticks_diff(ticks_ms(), deadline_ms) >= 0


def _free_one_tree_matching_sets(
    groups,
    allowed_partial_counts,
    state,
    deadline_ms,
):
    """Yield combinations for one tree; scheduling is handled by the caller."""
    minimum_allowed = min(allowed_partial_counts)
    maximum_allowed = max(allowed_partial_counts)

    def visit(
        group_index,
        selected,
        used_candidates,
        used_intervals,
        partial_count,
    ):
        if _free_deadline_reached(deadline_ms):
            return
        state["max_depth"] = max(
            state["max_depth"], group_index
        )
        if group_index >= len(groups):
            if partial_count in allowed_partial_counts:
                yield tuple(selected)
            return
        remaining = len(groups) - group_index - 1
        for candidate in groups[group_index][1]:
            if _free_deadline_reached(deadline_ms):
                return
            state["prefix_count"] += 1
            if candidate in used_candidates:
                state["prefix_pruned_candidate_reuse"] += 1
                continue
            next_partial = partial_count + (
                0 if _sim_is_full_match(candidate) else 1
            )
            if (
                next_partial > maximum_allowed
                or next_partial + remaining < minimum_allowed
            ):
                state["prefix_pruned_topology"] += 1
                continue
            next_intervals = _free_add_candidate_intervals(
                used_intervals, candidate
            )
            if next_intervals is None:
                state["interval_reuse_reject_count"] += 1
                continue
            selected.append(candidate)
            next_used = set(used_candidates)
            next_used.add(candidate)
            for result in visit(
                group_index + 1,
                selected,
                next_used,
                next_intervals,
                next_partial,
            ):
                yield result
            selected.pop()

    for matches in visit(0, [], set(), {}, 0):
        yield matches


def _free_partial_patterns(group_count, allowed_partial_counts):
    """Return deterministic full/partial placements for one labelled tree."""
    patterns = []
    for encoded in range(1 << max(0, int(group_count))):
        pattern = tuple(
            bool(encoded & (1 << index))
            for index in range(group_count)
        )
        if sum(
            1 for partial in pattern if partial
        ) in allowed_partial_counts:
            patterns.append(pattern)
    patterns.sort(
        key=lambda pattern: (
            sum(1 for partial in pattern if partial),
            pattern,
        )
    )
    return patterns


def _free_tree_matching_sets(
    candidates_by_pair,
    piece_count,
    allowed_partial_counts,
    state,
    deadline_ms,
    candidate_cache=None,
):
    """Round-robin labelled trees so an early tree cannot consume a pass."""
    candidate_cache = candidate_cache or {}

    def schedule_candidate_score(candidate):
        metadata = candidate_cache.get(candidate)
        if metadata is not None:
            ranking = metadata.get("ranking")
            if ranking:
                return float(ranking[0])
        return float(candidate[0])

    active = []
    for tree in _free_labeled_spanning_trees(piece_count):
        if _free_deadline_reached(deadline_ms):
            return
        state["trees_considered"] += 1
        groups = []
        for pair in tree:
            values = candidates_by_pair.get(pair, ())
            if not values:
                groups = []
                break
            groups.append((pair, values))
        if not groups and tree:
            continue
        groups.sort(key=lambda item: (len(item[1]), item[0]))
        for partial_pattern in _free_partial_patterns(
            len(groups), allowed_partial_counts
        ):
            scheduled_groups = []
            for group_index, group in enumerate(groups):
                values = [
                    candidate
                    for candidate in group[1]
                    if (
                        not _sim_is_full_match(candidate)
                    ) == partial_pattern[group_index]
                ]
                if not values:
                    scheduled_groups = []
                    break
                scheduled_groups.append((group[0], values))
            if not scheduled_groups and groups:
                continue
            partial_count = sum(
                1 for partial in partial_pattern if partial
            )
            combination_count = 1
            for scheduled_group in scheduled_groups:
                combination_count *= len(scheduled_group[1])
            schedule_priority = (
                0 if partial_count > 0 else 1,
                sum(
                    min(
                        schedule_candidate_score(candidate)
                        for candidate in group[1]
                    )
                    for group in scheduled_groups
                ),
                combination_count,
                tree,
                partial_pattern,
            )
            active.append(
                (
                    schedule_priority,
                    tree,
                    partial_pattern,
                    _free_one_tree_matching_sets(
                        scheduled_groups,
                        {partial_count},
                        state,
                        deadline_ms,
                    ),
                )
            )
            state["tree_schedule_count"] += 1

    quota = max(
        1,
        int(
            getattr(
                cfg, "FREE_RECT_TREE_ROUND_ROBIN_QUOTA", 16
            )
        ),
    )
    coverage_round = True
    while active and not _free_deadline_reached(deadline_ms):
        state["tree_round_robin_rounds"] += 1
        next_active = []
        round_quota = 1 if coverage_round else quota
        for priority, tree, partial_pattern, generator in active:
            if _free_deadline_reached(deadline_ms):
                return
            exhausted = False
            produced = 0
            while produced < round_quota:
                if _free_deadline_reached(deadline_ms):
                    return
                try:
                    matches = next(generator)
                except StopIteration:
                    exhausted = True
                    break
                produced += 1
                yield matches
            if not exhausted:
                next_active.append(
                    (priority, tree, partial_pattern, generator)
                )
        next_active.sort(key=lambda item: item[0])
        active = next_active
        coverage_round = False


def _free_time_expired(state):
    maximum = int(
        getattr(cfg, "FREE_RECT_MAX_PLAN_TIME_MS", 8000)
    )
    if maximum <= 0:
        return False
    if ticks_diff(ticks_ms(), state["started_ms"]) < maximum:
        return False
    state["timed_out"] = True
    return True


def _free_prefix_span(polygons, transforms, connected):
    points = []
    for index in range(len(polygons)):
        if not connected[index]:
            continue
        transform = transforms[index]
        if not _free_transform_valid(transform):
            return None
        points.extend(transform_polygon(polygons[index], transform))
    maximum_squared = 0.0
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            dx = points[right][0] - points[left][0]
            dy = points[right][1] - points[left][1]
            maximum_squared = max(
                maximum_squared, dx * dx + dy * dy
            )
    return math.sqrt(maximum_squared)


def _free_attach_transform(polygons, match, transforms, connected):
    _, i, _, j, _ = match[:5]
    ia, ib, ja, jb = _sim_match_segments(polygons, match)
    if connected[i] and not connected[j]:
        world_a = transform_point(ia, transforms[i])
        world_b = transform_point(ib, transforms[i])
        return j, _sim_align_edge(ja, jb, world_b, world_a)
    if connected[j] and not connected[i]:
        world_a = transform_point(ja, transforms[j])
        world_b = transform_point(jb, transforms[j])
        return i, _sim_align_edge(ia, ib, world_b, world_a)
    return None, None


def _free_topology_name(edge_count, full_count):
    partial_count = edge_count - full_count
    if edge_count == 0:
        return "single_piece"
    return "{}_full_{}_partial".format(
        full_count, partial_count
    )


def _free_topology_matching_sets(
    candidates,
    polygons,
    required_full,
    state,
):
    """Yield one deterministic rooted spanning-tree topology family."""
    piece_count = len(polygons)
    edge_count = max(0, piece_count - 1)
    prefix_seen = set()
    complete_seen = set()
    initial_connected = [False] * piece_count
    initial_connected[0] = True
    initial_transforms = [None] * piece_count
    initial_transforms[0] = _identity_transform()

    def visit(
        selected,
        selected_indices,
        used_edges,
        connected,
        transforms,
        full_count,
    ):
        if state["stop"] or _free_time_expired(state):
            state["stop"] = True
            return
        depth = len(selected)
        state["max_depth"] = max(state["max_depth"], depth)
        if depth == edge_count:
            if (
                full_count != required_full
                or not all(connected)
            ):
                state["prefix_pruned_topology"] += 1
                return
            signature = tuple(sorted(selected_indices))
            if signature in complete_seen:
                state["prefix_pruned_duplicate"] += 1
                return
            complete_seen.add(signature)
            yield tuple(selected)
            return

        remaining_after_add = edge_count - depth - 1
        for candidate_index in range(len(candidates)):
            if state["stop"] or _free_time_expired(state):
                state["stop"] = True
                return
            state["prefix_count"] += 1
            if candidate_index in selected_indices:
                state["prefix_pruned_candidate_reuse"] += 1
                continue
            match = candidates[candidate_index]
            is_full = _sim_is_full_match(match)
            next_full = full_count + (1 if is_full else 0)
            if (
                next_full > required_full
                or next_full + remaining_after_add
                < required_full
            ):
                state["prefix_pruned_topology"] += 1
                continue
            _, i, edge_i, j, edge_j = match[:5]
            if connected[i] == connected[j]:
                state["prefix_pruned_connectivity"] += 1
                continue
            if (
                (i, edge_i) in used_edges
                or (j, edge_j) in used_edges
            ):
                state["prefix_pruned_edge_reuse"] += 1
                continue

            new_piece, new_transform = _free_attach_transform(
                polygons, match, transforms, connected
            )
            if (
                new_piece is None
                or not _free_transform_valid(new_transform)
            ):
                state["prefix_pruned_invalid_geometry"] += 1
                continue
            next_connected = list(connected)
            next_connected[new_piece] = True
            next_transforms = list(transforms)
            next_transforms[new_piece] = new_transform
            span = _free_prefix_span(
                polygons, next_transforms, next_connected
            )
            if span is None:
                state["prefix_pruned_invalid_geometry"] += 1
                continue
            if span > float(
                getattr(cfg, "FREE_RECT_MAX_SPAN_MM", 170.0)
            ):
                state["prefix_pruned_span"] += 1
                continue

            next_indices = selected_indices + (
                candidate_index,
            )
            prefix_signature = tuple(sorted(next_indices))
            if prefix_signature in prefix_seen:
                state["prefix_pruned_duplicate"] += 1
                continue
            prefix_seen.add(prefix_signature)
            next_edges = set(used_edges)
            next_edges.add((i, edge_i))
            next_edges.add((j, edge_j))
            selected.append(match)
            for result in visit(
                selected,
                next_indices,
                next_edges,
                next_connected,
                next_transforms,
                next_full,
            ):
                yield result
            selected.pop()
            if state["stop"]:
                return

    for result in visit(
        [],
        (),
        set(),
        initial_connected,
        initial_transforms,
        0,
    ):
        yield result


def _free_rect_matching_sets(candidates, polygons, state):
    """Yield all supported full/partial connected trees round-robin."""
    piece_count = len(polygons)
    edge_count = max(0, piece_count - 1)
    if piece_count == 1:
        if not _free_time_expired(state):
            state["complete_matching_set_count"] = 1
            state["matching_topology_counts"][
                "single_piece"
            ] = 1
            yield ()
        return

    generators = []
    for required_full in range(edge_count, -1, -1):
        generators.append(
            (
                required_full,
                _free_topology_matching_sets(
                    candidates,
                    polygons,
                    required_full,
                    state,
                ),
            )
        )
    maximum = max(
        1,
        int(
            getattr(
                cfg, "FREE_RECT_MAX_COMPLETE_SETS", 6000
            )
        ),
    )
    while generators and not state["stop"]:
        active = []
        for required_full, generator in generators:
            if state["complete_matching_set_count"] >= maximum:
                state["limit_hit"] = True
                state["stop"] = True
                return
            if _free_time_expired(state):
                state["stop"] = True
                return
            try:
                matches = next(generator)
            except StopIteration:
                continue
            state["complete_matching_set_count"] += 1
            topology = _free_topology_name(
                edge_count, required_full
            )
            counts = state["matching_topology_counts"]
            counts[topology] = counts.get(topology, 0) + 1
            yield matches
            active.append((required_full, generator))
        generators = active


def _free_initial_transforms(
    polygons, matches, alignment="midpoint"
):
    adjacency = [[] for _ in polygons]
    for match in matches:
        _, i, _, j, _ = match[:5]
        adjacency[i].append((j, match, False))
        adjacency[j].append((i, match, True))
    transforms = [None] * len(polygons)
    transforms[0] = _identity_transform()
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor, match, reverse_sides in adjacency[current]:
            if transforms[neighbor] is not None:
                continue
            ia, ib, ja, jb = _sim_match_segments(
                polygons, match
            )
            if reverse_sides:
                ia, ib, ja, jb = ja, jb, ia, ib
            world_a = transform_point(ia, transforms[current])
            world_b = transform_point(ib, transforms[current])
            align = (
                _sim_align_edge
                if alignment == "endpoint"
                else _sim_align_segment_midpoint
            )
            transforms[neighbor] = align(
                ja, jb, world_b, world_a
            )
            stack.append(neighbor)
    if any(
        not _free_transform_valid(transform)
        for transform in transforms
    ):
        return None
    return transforms


def _free_vertex_angle(polygon, index):
    point = polygon[index % len(polygon)]
    previous = polygon[(index - 1) % len(polygon)]
    following = polygon[(index + 1) % len(polygon)]
    left = (
        previous[0] - point[0],
        previous[1] - point[1],
    )
    right = (
        following[0] - point[0],
        following[1] - point[1],
    )
    left_length = math.sqrt(
        left[0] * left[0] + left[1] * left[1]
    )
    right_length = math.sqrt(
        right[0] * right[0] + right[1] * right[1]
    )
    if left_length <= EPS or right_length <= EPS:
        return 180.0
    cosine = (
        left[0] * right[0] + left[1] * right[1]
    ) / (left_length * right_length)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _free_endpoint_angle(polygon, edge_index, fraction):
    if fraction <= 1e-6:
        return _free_vertex_angle(polygon, edge_index)
    if fraction >= 1.0 - 1e-6:
        return _free_vertex_angle(polygon, edge_index + 1)
    return None


def _free_endpoint_angle_error(polygons, match):
    _, i, edge_i, j, edge_j, ia0, ia1, ja0, ja1 = match
    pairs = (
        (
            _free_endpoint_angle(polygons[i], edge_i, ia0),
            _free_endpoint_angle(polygons[j], edge_j, ja1),
        ),
        (
            _free_endpoint_angle(polygons[i], edge_i, ia1),
            _free_endpoint_angle(polygons[j], edge_j, ja0),
        ),
    )
    error = 0.0
    for left, right in pairs:
        if left is None and right is None:
            continue
        if left is None or right is None:
            angle = right if left is None else left
            error += min(
                abs(angle - 90.0),
                abs(angle - 180.0),
            ) / 180.0
        else:
            total = left + right
            error += min(
                abs(total - 90.0),
                abs(total - 180.0),
            ) / 180.0
    return error / max(1, len(pairs))


def _free_seam_metrics(polygons, matches, transforms):
    records = []
    full_errors = []
    partial_errors = []
    endpoint_errors = []
    closure_distances = []
    typical_lengths = []
    maximum_length_error = 0.0
    for match in matches:
        a0, a1, b0, b1 = _sim_match_segments(
            polygons, match
        )
        length_a = _free_distance(a0, a1)
        length_b = _free_distance(b0, b1)
        length_error = abs(length_a - length_b)
        relative_error = length_error / max(
            EPS, length_a, length_b
        )
        partial = not _sim_is_full_match(match)
        if partial:
            partial_errors.append(
                max(relative_error, float(match[0]))
            )
        else:
            full_errors.append(relative_error)
        endpoint_error = _free_endpoint_angle_error(
            polygons, match
        )
        endpoint_errors.append(endpoint_error)
        maximum_length_error = max(
            maximum_length_error, length_error
        )
        _, i, edge_i, j, edge_j = match[:5]
        world_a0 = transform_point(a0, transforms[i])
        world_a1 = transform_point(a1, transforms[i])
        world_b0 = transform_point(b0, transforms[j])
        world_b1 = transform_point(b1, transforms[j])
        closure_distances.append(
            _free_distance(world_a0, world_b1)
        )
        closure_distances.append(
            _free_distance(world_a1, world_b0)
        )
        typical_lengths.append(0.5 * (length_a + length_b))
        records.append(
            {
                "piece_a_index": i,
                "edge_a_index": edge_i,
                "piece_b_index": j,
                "edge_b_index": edge_j,
                "partial": partial,
                "matched_length_a_mm": length_a,
                "matched_length_b_mm": length_b,
                "covered_length_mm": min(length_a, length_b),
                "length_error_mm": length_error,
                "relative_length_error": relative_error,
                "endpoint_angle_error": endpoint_error,
                "fractions": list(match[5:]),
            }
        )

    match_count = len(matches)
    full_mean = sum(full_errors) / max(1, len(full_errors))
    partial_mean = sum(partial_errors) / max(
        1, len(partial_errors)
    )
    partial_ratio = len(partial_errors) / max(1, match_count)
    endpoint_mean = sum(endpoint_errors) / max(
        1, len(endpoint_errors)
    )
    partial_count_term = partial_ratio * float(
        getattr(cfg, "FREE_RECT_PARTIAL_COUNT_PENALTY", 1.0)
    )
    seam_cost = (
        full_mean
        + partial_mean
        + partial_count_term
        + endpoint_mean
    ) / 4.0
    closure_error_mm = sum(closure_distances) / max(
        1, len(closure_distances)
    )
    typical_edge_mm = sum(typical_lengths) / max(
        1, len(typical_lengths)
    )
    closure_cost = closure_error_mm / max(
        EPS, len(polygons) * typical_edge_mm
    )
    return {
        "records": records,
        "max_length_error_mm": maximum_length_error,
        "full_relative_error": full_mean,
        "partial_match_error": partial_mean,
        "partial_ratio": partial_ratio,
        "endpoint_angle_error": endpoint_mean,
        "seam_cost": seam_cost,
        "closure_error_mm": closure_error_mm,
        "closure_cost": closure_cost,
    }


def _free_interval_penalty(value, lower, upper):
    if value < lower:
        return ((lower - value) / max(EPS, lower)) ** 2
    if value > upper:
        return ((value - upper) / max(EPS, upper)) ** 2
    return 0.0


def _free_build_piece_cache(pieces):
    cache = []
    for index, piece in enumerate(pieces):
        polygon = list(piece.polygon_mm)
        edges = _free_edges(polygon)
        cached_lengths = getattr(piece, "edge_lengths_mm", None)
        lengths = (
            list(cached_lengths)
            if cached_lengths is not None
            and len(cached_lengths) == len(edges)
            else [
                _free_distance(edge[0], edge[1]) for edge in edges
            ]
        )
        try:
            cached_triangles = getattr(piece, "triangles_mm", None)
            triangles = (
                [list(triangle) for triangle in cached_triangles]
                if cached_triangles
                else triangulate_simple_polygon(polygon)
            )
        except Exception:
            return None
        cache.append(
            {
                "piece_index": index,
                "polygon": polygon,
                "edges": edges,
                "edge_lengths": lengths,
                "vertex_angles": [
                    _free_vertex_angle(polygon, vertex)
                    for vertex in range(len(polygon))
                ],
                "centroid": (
                    piece.centroid_mm
                    if hasattr(piece, "centroid_mm")
                    else polygon_centroid(polygon)
                ),
                "area": float(
                    piece.area_mm2
                    if hasattr(piece, "area_mm2")
                    else polygon_area(polygon)
                ),
                "aabb": (
                    piece.aabb_mm
                    if hasattr(piece, "aabb_mm")
                    else polygon_aabb(polygon)
                ),
                "convex": bool(
                    piece.is_convex
                    if hasattr(piece, "is_convex")
                    else polygon_is_convex(polygon)
                ),
                "triangles": triangles,
                "source_perimeter": sum(lengths),
            }
        )
    return cache


def _free_selected_shared_length(
    polygons, matches, candidate_cache=None
):
    total = 0.0
    for match in matches:
        if candidate_cache is not None and match in candidate_cache:
            total += candidate_cache[match]["matched_length_mm"]
            continue
        a0, a1, b0, b1 = _sim_match_segments(polygons, match)
        total += min(
            _free_distance(a0, a1),
            _free_distance(b0, b1),
        )
    return total


def _free_aabb_overlap_ratio(polygons, source_area):
    overlap = 0.0
    boxes = [polygon_aabb(polygon) for polygon in polygons]
    for right in range(len(boxes)):
        for left in range(right):
            width = max(
                0.0,
                min(boxes[left][2], boxes[right][2])
                - max(boxes[left][0], boxes[right][0]),
            )
            height = max(
                0.0,
                min(boxes[left][3], boxes[right][3])
                - max(boxes[left][1], boxes[right][1]),
            )
            overlap += width * height
    return overlap / max(EPS, source_area)


def _free_cheap_complete_metrics(
    polygons,
    matches,
    transforms,
    source_area,
    perimeter_context,
    candidate_cache=None,
):
    assembled = [
        transform_polygon(polygon, transform)
        for polygon, transform in zip(polygons, transforms)
    ]
    if any(not _free_polygon_valid(polygon) for polygon in assembled):
        return None
    points = [point for polygon in assembled for point in polygon]
    rectangle = minimum_area_rectangle(points)
    if rectangle is None or rectangle["area"] <= EPS:
        return None
    long_side = max(rectangle["width"], rectangle["height"])
    short_side = min(rectangle["width"], rectangle["height"])
    dimension = _free_dimension_prior(
        source_area, long_side, short_side
    )
    hull_area = polygon_area(convex_hull(points))
    hull_gap_ratio = max(
        0.0, rectangle["area"] - hull_area
    ) / max(EPS, source_area)
    selected_shared = _free_selected_shared_length(
        polygons, matches, candidate_cache=candidate_cache
    )
    cheap_exposed = max(
        0.0,
        perimeter_context["source_perimeter_mm"]
        - 2.0 * selected_shared,
    )
    expected_minimum = perimeter_context[
        "expected_perimeter_min_mm"
    ]
    expected_maximum = perimeter_context[
        "expected_perimeter_max_mm"
    ]
    perimeter_error = (
        max(0.0, expected_minimum - cheap_exposed)
        / max(EPS, expected_minimum)
        + max(0.0, cheap_exposed - expected_maximum)
        / max(EPS, expected_maximum)
    )
    _evidence, missing_count, missing_ratio = (
        _free_outer_piece_evidence(
            assembled,
            rectangle,
            distance_tolerance=1.5
            * float(
                getattr(
                    cfg, "FREE_RECT_OUTER_EDGE_DISTANCE_MM", 6.0
                )
            ),
            angle_tolerance=1.5
            * float(
                getattr(
                    cfg, "FREE_RECT_OUTER_EDGE_ANGLE_DEG", 12.0
                )
            ),
            minimum_edge_mm=0.80
            * float(
                getattr(
                    cfg, "FREE_RECT_MIN_OBSERVED_EDGE_MM", 17.5
                )
            ),
        )
    )
    seam = _free_seam_metrics(polygons, matches, transforms)
    aabb_overlap = _free_aabb_overlap_ratio(
        assembled, source_area
    )
    cheap_score = (
        5.0 * dimension["area_prior_error"]
        + 4.0 * dimension["dimension_range_penalty"]
        + 3.0 * hull_gap_ratio
        + 2.0 * perimeter_error
        + 1.5 * missing_ratio
        + 1.0 * seam["seam_cost"]
        + 1.0 * aabb_overlap
    )
    return {
        "rectangle": rectangle,
        "long_side_mm": long_side,
        "short_side_mm": short_side,
        "area_prior_error": dimension["area_prior_error"],
        "dimension_range_penalty": dimension[
            "dimension_range_penalty"
        ],
        "hull_gap_ratio": hull_gap_ratio,
        "cheap_exposed_perimeter_mm": cheap_exposed,
        "cheap_perimeter_error": perimeter_error,
        "outer_piece_missing_approx": missing_count,
        "outer_piece_missing_ratio_approx": missing_ratio,
        "seam_cost": seam["seam_cost"],
        "aabb_overlap_ratio": aabb_overlap,
        "cheap_score": cheap_score,
    }


def _free_cheap_rank(item):
    metrics = item["cheap_metrics"]
    return (
        metrics["cheap_score"],
        metrics["area_prior_error"],
        metrics["hull_gap_ratio"],
        item["topology"],
        item["match_signature"],
    )


def _free_beam_limit_for_topology(partial_count):
    if partial_count == 1:
        return max(
            1,
            int(
                getattr(
                    cfg, "FREE_RECT_CHEAP_BEAM_ONE_PARTIAL", 24
                )
            ),
        )
    return max(
        1,
        int(
            getattr(
                cfg, "FREE_RECT_CHEAP_BEAM_PER_TOPOLOGY", 16
            )
        ),
    )


def _free_add_to_cheap_beam(beams, item):
    topology = item["topology"]
    beam = beams.setdefault(topology, [])
    beam.append(item)
    beam.sort(key=_free_cheap_rank)
    del beam[
        _free_beam_limit_for_topology(item["partial_count"]) :
    ]


def _free_merge_cheap_beams(beams):
    merged = []
    for topology in sorted(beams):
        merged.extend(beams[topology])
    merged.sort(key=_free_cheap_rank)
    maximum = max(
        1,
        int(getattr(cfg, "FREE_RECT_EXACT_BEAM_SIZE", 48)),
    )
    return merged[:maximum]


def _free_outer_piece_evidence(
    assembled,
    rectangle,
    distance_tolerance=None,
    angle_tolerance=None,
    minimum_edge_mm=None,
):
    angle = math.radians(-rectangle["angle_deg"])
    rotation = (
        math.cos(angle),
        math.sin(angle),
        0.0,
        0.0,
        normalize_angle_deg(math.degrees(angle)),
    )
    oriented = [
        transform_polygon(polygon, rotation)
        for polygon in assembled
    ]
    min_x, min_y, max_x, max_y = rectangle[
        "bounds_rotated"
    ]
    tolerance = float(
        getattr(cfg, "FREE_RECT_OUTER_EDGE_DISTANCE_MM", 6.0)
        if distance_tolerance is None
        else distance_tolerance
    )
    angle_tolerance = float(
        getattr(cfg, "FREE_RECT_OUTER_EDGE_ANGLE_DEG", 12.0)
        if angle_tolerance is None
        else angle_tolerance
    )
    minimum_edge_mm = float(
        getattr(cfg, "FREE_RECT_MIN_OBSERVED_EDGE_MM", 17.5)
        if minimum_edge_mm is None
        else minimum_edge_mm
    )
    parallel_cosine = math.cos(math.radians(angle_tolerance))
    evidence = []
    missing = 0
    for piece_index, polygon in enumerate(oriented):
        boundary_edges = []
        for edge_index, edge in enumerate(_free_edges(polygon)):
            a, b = edge
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length + EPS < minimum_edge_mm:
                continue
            side = None
            if (
                abs(a[0] - min_x) <= tolerance
                and abs(b[0] - min_x) <= tolerance
                and abs(dy) / max(EPS, length) >= parallel_cosine
            ):
                side = "left"
            elif (
                abs(a[0] - max_x) <= tolerance
                and abs(b[0] - max_x) <= tolerance
                and abs(dy) / max(EPS, length) >= parallel_cosine
            ):
                side = "right"
            elif (
                abs(a[1] - min_y) <= tolerance
                and abs(b[1] - min_y) <= tolerance
                and abs(dx) / max(EPS, length) >= parallel_cosine
            ):
                side = "top"
            elif (
                abs(a[1] - max_y) <= tolerance
                and abs(b[1] - max_y) <= tolerance
                and abs(dx) / max(EPS, length) >= parallel_cosine
            ):
                side = "bottom"
            if side is not None:
                boundary_edges.append(
                    {
                        "edge_index": edge_index,
                        "side": side,
                        "length_mm": length,
                    }
                )
        present = bool(boundary_edges)
        if not present:
            missing += 1
        evidence.append(
            {
                "piece_index": piece_index,
                "has_outer_edge": present,
                "boundary_edges": boundary_edges,
            }
        )
    return (
        evidence,
        missing,
        missing / max(1, len(assembled)),
    )


def _free_complete_metrics(
    polygons,
    matches,
    transforms,
    source_area_mm2,
    perimeter=None,
    piece_cache=None,
):
    assembled = [
        transform_polygon(polygon, transform)
        for polygon, transform in zip(polygons, transforms)
    ]
    if any(not _free_polygon_valid(polygon) for polygon in assembled):
        return None
    points = [
        point for polygon in assembled for point in polygon
    ]
    rectangle = minimum_area_rectangle(points)
    if (
        rectangle is None
        or rectangle["area"] <= EPS
        or not _free_finite(rectangle["area"])
    ):
        return None
    transformed_triangles = None
    if piece_cache is not None:
        transformed_triangles = [
            [
                transform_polygon(triangle, transforms[index])
                for triangle in piece_cache[index]["triangles"]
            ]
            for index in range(len(polygons))
        ]
    overlap_area = 0.0
    for index, polygon in enumerate(assembled):
        for earlier in range(index):
            overlap_area += polygon_overlap_area(
                polygon,
                assembled[earlier],
                (
                    transformed_triangles[index]
                    if transformed_triangles is not None
                    else None
                ),
                (
                    transformed_triangles[earlier]
                    if transformed_triangles is not None
                    else None
                ),
            )
    union_area_approx = max(
        0.0, source_area_mm2 - overlap_area
    )
    rectangle_area = rectangle["area"]
    hull_area = polygon_area(convex_hull(points))
    fill_gap = max(0.0, rectangle_area - union_area_approx)
    hull_gap = max(0.0, rectangle_area - hull_area)
    long_side = max(
        rectangle["width"], rectangle["height"]
    )
    short_side = min(
        rectangle["width"], rectangle["height"]
    )
    aspect_ratio = long_side / max(EPS, short_side)
    dimension_prior = _free_dimension_prior(
        source_area_mm2, long_side, short_side
    )
    outer_evidence, missing_count, missing_ratio = (
        _free_outer_piece_evidence(assembled, rectangle)
    )
    seam = _free_seam_metrics(
        polygons, matches, transforms
    )
    if perimeter is None:
        perimeter = _free_exposed_perimeter_metrics(
            polygons,
            matches,
            transforms,
            source_area_mm2,
        )
    source_area = max(EPS, source_area_mm2)
    metrics = {
        "source_area_mm2": source_area_mm2,
        "overlap_mm2": overlap_area,
        "overlap_ratio": overlap_area / source_area,
        "union_area_approx_mm2": union_area_approx,
        "rect_area_mm2": rectangle_area,
        "fill_gap_mm2": fill_gap,
        "fill_gap_ratio": fill_gap / source_area,
        "hull_area_mm2": hull_area,
        "hull_gap_mm2": hull_gap,
        "hull_gap_ratio": hull_gap / source_area,
        "area_prior_error": dimension_prior["area_prior_error"],
        "long_side_mm": long_side,
        "short_side_mm": short_side,
        "aspect_ratio": aspect_ratio,
        "aspect_preferred": (
            dimension_prior["dimension_range_penalty"] <= EPS
        ),
        "aspect_range_penalty": 0.0,
        "dimension_range_penalty": dimension_prior[
            "dimension_range_penalty"
        ],
        "nearest_feasible_dimension_penalty": dimension_prior[
            "nearest_feasible_dimension_penalty"
        ],
        "outer_piece_missing_count": missing_count,
        "outer_piece_missing_ratio": missing_ratio,
        "outer_piece_evidence": outer_evidence,
        "seam_cost": seam["seam_cost"],
        "closure_error_mm": seam["closure_error_mm"],
        "closure_cost": seam["closure_cost"],
        "full_relative_error": seam["full_relative_error"],
        "partial_match_error": seam["partial_match_error"],
        "partial_ratio": seam["partial_ratio"],
        "endpoint_angle_error": seam[
            "endpoint_angle_error"
        ],
        "source_perimeter_mm": perimeter[
            "source_perimeter_mm"
        ],
        "covered_perimeter_mm": perimeter[
            "covered_perimeter_mm"
        ],
        "internal_seam_length_mm": perimeter[
            "internal_seam_length_mm"
        ],
        "selected_shared_length_mm": perimeter[
            "selected_shared_length_mm"
        ],
        "additional_shared_length_mm": perimeter[
            "additional_shared_length_mm"
        ],
        "additional_contact_count": perimeter[
            "additional_contact_count"
        ],
        "exposed_perimeter_mm": perimeter[
            "exposed_perimeter_mm"
        ],
        "expected_perimeter_min_mm": perimeter[
            "expected_perimeter_min_mm"
        ],
        "expected_perimeter_max_mm": perimeter[
            "expected_perimeter_max_mm"
        ],
        "perimeter_deficit_ratio": perimeter[
            "perimeter_deficit_ratio"
        ],
        "perimeter_excess_ratio": perimeter[
            "perimeter_excess_ratio"
        ],
        "perimeter_error_ratio": perimeter[
            "perimeter_error_ratio"
        ],
    }
    cost = (
        float(getattr(cfg, "FREE_RECT_WEIGHT_OVERLAP", 10.0))
        * metrics["overlap_ratio"]
        + float(
            getattr(cfg, "FREE_RECT_WEIGHT_FILL_GAP", 6.0)
        )
        * metrics["fill_gap_ratio"]
        + float(
            getattr(cfg, "FREE_RECT_WEIGHT_HULL_GAP", 2.0)
        )
        * metrics["hull_gap_ratio"]
        + float(
            getattr(cfg, "FREE_RECT_WEIGHT_AREA_PRIOR", 3.0)
        )
        * metrics["area_prior_error"]
        + float(
            getattr(
                cfg, "FREE_RECT_WEIGHT_DIMENSION_RANGE", 3.0
            )
        )
        * metrics["dimension_range_penalty"]
        + float(
            getattr(cfg, "FREE_RECT_WEIGHT_OUTER_PIECE", 2.0)
        )
        * metrics["outer_piece_missing_ratio"]
        + float(
            getattr(cfg, "FREE_RECT_WEIGHT_SEAM", 1.0)
        )
        * metrics["seam_cost"]
        + float(
            getattr(cfg, "FREE_RECT_WEIGHT_CLOSURE", 1.0)
        )
        * metrics["closure_cost"]
        + float(
            getattr(cfg, "FREE_RECT_WEIGHT_PERIMETER", 12.0)
        )
        * metrics["perimeter_error_ratio"]
    )
    metrics["cost"] = cost
    return {
        "assembled": assembled,
        "rectangle": rectangle,
        "metrics": metrics,
        "seams": seam["records"],
        "max_length_error_mm": seam["max_length_error_mm"],
    }


def _free_physical_validity(metrics, target):
    failures = []
    if not (
        float(
            getattr(cfg, "FREE_RECT_PUBLISH_LONG_MIN_MM", 85.0)
        )
        <= metrics["long_side_mm"]
        <= float(
            getattr(cfg, "FREE_RECT_PUBLISH_LONG_MAX_MM", 125.0)
        )
    ):
        failures.append("long_side")
    if not (
        float(
            getattr(cfg, "FREE_RECT_PUBLISH_SHORT_MIN_MM", 45.0)
        )
        <= metrics["short_side_mm"]
        <= float(
            getattr(cfg, "FREE_RECT_PUBLISH_SHORT_MAX_MM", 95.0)
        )
    ):
        failures.append("short_side")
    if metrics["area_prior_error"] > float(
        getattr(cfg, "FREE_RECT_PUBLISH_AREA_ERROR_MAX", 0.15)
    ):
        failures.append("area_error")
    if metrics["overlap_ratio"] > float(
        getattr(
            cfg, "FREE_RECT_PUBLISH_OVERLAP_RATIO_MAX", 0.05
        )
    ):
        failures.append("overlap")
    if metrics["fill_gap_ratio"] > float(
        getattr(
            cfg, "FREE_RECT_PUBLISH_FILL_GAP_RATIO_MAX", 0.20
        )
    ):
        failures.append("fill_gap")
    if metrics["outer_piece_missing_count"] > int(
        getattr(cfg, "FREE_RECT_PUBLISH_OUTER_MISSING_MAX", 0)
    ):
        failures.append("outer_piece")
    if target is None:
        failures.append("target_fit")
    return not failures, tuple(failures)


def _free_is_strong_solution(proposal):
    metrics = proposal["metrics"]
    return bool(
        proposal.get("physical_valid", False)
        and metrics["area_prior_error"] <= 0.08
        and metrics["overlap_ratio"] <= 0.025
        and metrics["fill_gap_ratio"] <= 0.12
        and metrics["outer_piece_missing_count"] == 0
    )


def _free_target_candidate(
    pieces,
    polygons,
    transforms,
    base_rotation_deg,
    quarter_turn_deg,
):
    angle_deg = normalize_angle_deg(
        base_rotation_deg + quarter_turn_deg
    )
    angle = math.radians(angle_deg)
    rotation = (
        math.cos(angle),
        math.sin(angle),
        0.0,
        0.0,
        angle_deg,
    )
    oriented_transforms = [
        compose_transforms(rotation, transform)
        for transform in transforms
    ]
    oriented = [
        transform_polygon(polygon, transform)
        for polygon, transform in zip(
            polygons, oriented_transforms
        )
    ]
    points = [
        point for polygon in oriented for point in polygon
    ]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    center = (
        0.5 * (min_x + max_x),
        0.5 * (min_y + max_y),
    )
    translation = (
        1.0,
        0.0,
        float(cfg.TARGET_CENTER_MM[0]) - center[0],
        float(cfg.TARGET_CENTER_MM[1]) - center[1],
        0.0,
    )
    final_transforms = [
        compose_transforms(translation, transform)
        for transform in oriented_transforms
    ]
    target_list = [
        transform_polygon(polygon, transform)
        for polygon, transform in zip(polygons, final_transforms)
    ]
    target_points = [
        point for polygon in target_list for point in polygon
    ]
    target_rect = (
        min(point[0] for point in target_points),
        min(point[1] for point in target_points),
        max(point[0] for point in target_points),
        max(point[1] for point in target_points),
    )
    width = target_rect[2] - target_rect[0]
    height = target_rect[3] - target_rect[1]
    long_horizontal = width + EPS >= height
    margin = float(
        getattr(
            cfg,
            "FREE_RECT_TARGET_MARGIN_MM",
            getattr(cfg, "TARGET_MARGIN_MM", 10.0),
        )
    )
    lower_y = float(cfg.A4_HEIGHT_MM) * 0.5 + margin
    upper_y = float(cfg.A4_HEIGHT_MM) - margin
    fits = (
        target_rect[0] >= margin - EPS
        and target_rect[2]
        <= float(cfg.A4_WIDTH_MM) - margin + EPS
        and target_rect[1] >= lower_y - EPS
        and target_rect[3] <= upper_y + EPS
    )
    operations = []
    motion_cost = 0.0
    target_polygons = {}
    rotation_weight = float(
        getattr(
            cfg,
            "FREE_RECT_MOTION_ROTATION_WEIGHT_MM_PER_DEG",
            0.10,
        )
    )
    for index, piece in enumerate(pieces):
        piece_id = getattr(
            piece, "piece_id", None
        ) or "P{}".format(index + 1)
        target_polygon = target_list[index]
        target_center = polygon_centroid(target_polygon)
        source_center = piece.centroid_mm
        rotation_deg = _choose_smallest_equivalent_rotation(
            final_transforms[index][4], piece.polygon_mm
        )
        motion_cost += _free_distance(
            source_center, target_center
        ) + rotation_weight * abs(rotation_deg)
        target_polygons[piece_id] = target_polygon
        operations.append(
            {
                "piece_id": piece_id,
                "source_center_mm": source_center,
                "target_center_mm": target_center,
                "rotation_deg": rotation_deg,
                "rotation_ambiguous": bool(
                    getattr(piece, "rotation_ambiguous", False)
                ),
                "confidence": float(
                    getattr(piece, "confidence", 1.0)
                ),
            }
        )
    return {
        "quarter_turn_deg": quarter_turn_deg,
        "angle_deg": angle_deg,
        "long_horizontal": long_horizontal,
        "fits": fits,
        "motion_cost": motion_cost,
        "transforms": final_transforms,
        "target_list": target_list,
        "target_polygons": target_polygons,
        "target_rect": target_rect,
        "operations": operations,
    }


def _free_select_target(
    pieces,
    polygons,
    transforms,
    rectangle,
):
    base_rotation = -float(rectangle["angle_deg"])
    if rectangle["width"] + EPS < rectangle["height"]:
        base_rotation += 90.0
    candidates = []
    eligible = []
    for quarter_turn in (0.0, 90.0, 180.0, 270.0):
        candidate = _free_target_candidate(
            pieces,
            polygons,
            transforms,
            base_rotation,
            quarter_turn,
        )
        candidates.append(
            {
                "quarter_turn_deg": quarter_turn,
                "motion_cost": candidate["motion_cost"],
                "long_horizontal": candidate[
                    "long_horizontal"
                ],
                "fits": candidate["fits"],
            }
        )
        if (
            candidate["long_horizontal"]
            and candidate["fits"]
        ):
            eligible.append(candidate)
    if not eligible:
        return None, candidates
    eligible.sort(
        key=lambda item: (
            item["motion_cost"],
            item["quarter_turn_deg"],
        )
    )
    return eligible[0], candidates


def _free_match_signature(matches):
    return tuple(
        sorted(
            (
                match[1],
                match[2],
                match[3],
                match[4],
                tuple(match[5:]),
            )
            for match in matches
        )
    )


def _free_top_record(proposal):
    metrics = proposal["metrics"]
    return {
        "cost": metrics["cost"],
        "physical_valid": bool(
            proposal.get("physical_valid", False)
        ),
        "physical_failures": proposal.get(
            "physical_failures", ()
        ),
        "topology": proposal["topology"],
        "target_fits": proposal["target"] is not None,
        "long_side_mm": metrics["long_side_mm"],
        "short_side_mm": metrics["short_side_mm"],
        "aspect_ratio": metrics["aspect_ratio"],
        "aspect_preferred": metrics["aspect_preferred"],
        "aspect_range_penalty": metrics[
            "aspect_range_penalty"
        ],
        "rect_area_mm2": metrics["rect_area_mm2"],
        "overlap_mm2": metrics["overlap_mm2"],
        "overlap_ratio": metrics["overlap_ratio"],
        "fill_gap_mm2": metrics["fill_gap_mm2"],
        "fill_gap_ratio": metrics["fill_gap_ratio"],
        "hull_gap_mm2": metrics["hull_gap_mm2"],
        "hull_gap_ratio": metrics["hull_gap_ratio"],
        "area_prior_error": metrics["area_prior_error"],
        "dimension_range_penalty": metrics[
            "dimension_range_penalty"
        ],
        "outer_piece_missing_ratio": metrics[
            "outer_piece_missing_ratio"
        ],
        "exposed_perimeter_mm": metrics[
            "exposed_perimeter_mm"
        ],
        "expected_perimeter_min_mm": metrics[
            "expected_perimeter_min_mm"
        ],
        "expected_perimeter_max_mm": metrics[
            "expected_perimeter_max_mm"
        ],
        "perimeter_excess_ratio": metrics[
            "perimeter_excess_ratio"
        ],
        "perimeter_error_ratio": metrics[
            "perimeter_error_ratio"
        ],
        "additional_contact_count": metrics[
            "additional_contact_count"
        ],
        "seam_cost": metrics["seam_cost"],
        "closure_cost": metrics["closure_cost"],
        "selected_partial_match_count": proposal[
            "partial_count"
        ],
        "selected_seams": proposal["seams"],
    }


def _free_proposal_rank(proposal):
    """Physical validity precedes geometric cost; aspect is never privileged."""
    metrics = proposal["metrics"]
    return (
        0 if proposal.get("physical_valid", False) else 1,
        metrics["cost"],
        metrics["area_prior_error"],
        metrics["fill_gap_ratio"],
        metrics["overlap_ratio"],
        proposal["topology"],
        proposal["match_signature"],
    )


def _free_update_top(top, proposal):
    top.append(proposal)
    top.sort(key=_free_proposal_rank)
    maximum = max(
        1, int(getattr(cfg, "FREE_RECT_TOP_K", 5))
    )
    del top[maximum:]


def _free_progress(state, force=False):
    now = ticks_ms()
    interval = max(
        1,
        int(
            getattr(
                cfg, "FREE_RECT_PROGRESS_INTERVAL_MS", 1000
            )
        ),
    )
    elapsed = max(0, ticks_diff(now, state["started_ms"]))
    if (
        not force
        and ticks_diff(now, state["last_progress_ms"])
        < interval
    ):
        return
    best_cost = state.get("best_cost")
    best_aspect = state.get("best_aspect_ratio")
    print(
        "FREE_PLAN_PROGRESS,elapsed_ms={},complete_sets={},"
        "perimeter_passed={},perimeter_rejected={},"
        "best_cost={},best_aspect={},best_perimeter_error={}".format(
            elapsed,
            state["complete_matching_set_count"],
            state["perimeter_prefilter_passed_count"],
            state["perimeter_prefilter_rejected_count"],
            (
                "{:.6f}".format(best_cost)
                if best_cost is not None
                else "na"
            ),
            (
                "{:.3f}".format(best_aspect)
                if best_aspect is not None
                else "na"
            ),
            (
                "{:.3f}".format(
                    state["best_perimeter_error_ratio"]
                )
                if state["best_perimeter_error_ratio"] is not None
                else "na"
            ),
        )
    )
    state["last_progress_ms"] = now
    exitpoint = getattr(os, "exitpoint", None)
    if exitpoint is not None:
        exitpoint()
    plan_debug_heartbeat()


def _free_pass_definitions(edge_count):
    definitions = [
        {
            "name": "strict_full",
            "full_rel_tolerance": 0.12,
            "partial_enabled": False,
            "partial_min": 0.22,
            "partial_max": 0.88,
            "allowed_partial_counts": (0,),
            "budget_ms": int(
                getattr(cfg, "FREE_RECT_PASS_STRICT_FULL_MS", 1500)
            ),
        },
        {
            "name": "standard_t",
            "full_rel_tolerance": 0.12,
            "partial_enabled": True,
            "partial_min": 0.22,
            "partial_max": 0.88,
            "allowed_partial_counts": tuple(
                value for value in (0, 1) if value <= edge_count
            ),
            "budget_ms": int(
                getattr(cfg, "FREE_RECT_PASS_STANDARD_T_MS", 2500)
            ),
        },
        {
            "name": "relaxed_geometry",
            "full_rel_tolerance": 0.16,
            "partial_enabled": True,
            "partial_min": 0.18,
            "partial_max": 0.90,
            "allowed_partial_counts": tuple(
                value for value in (0, 1) if value <= edge_count
            ),
            "budget_ms": int(
                getattr(
                    cfg, "FREE_RECT_PASS_RELAXED_GEOMETRY_MS", 3000
                )
            ),
        },
        {
            "name": "multi_partial_fallback",
            "full_rel_tolerance": 0.20,
            "partial_enabled": True,
            "partial_min": 0.15,
            "partial_max": 0.92,
            "allowed_partial_counts": tuple(
                value
                for value in range(2, edge_count + 1)
            ),
            "budget_ms": int(
                getattr(
                    cfg, "FREE_RECT_PASS_MULTI_PARTIAL_MS", 3000
                )
            ),
        },
    ]
    return [
        definition
        for definition in definitions
        if definition["allowed_partial_counts"]
        and definition["budget_ms"] > 0
    ]


def _free_pair_counts_text(pair_counts):
    return "|".join(
        "P{}-P{}:{}/{}".format(
            pair[0] + 1,
            pair[1] + 1,
            pair_counts[pair][0],
            pair_counts[pair][1],
        )
        for pair in sorted(pair_counts)
    ) or "none"


def _free_pass_progress(
    pass_name,
    pass_started_ms,
    local,
    best=None,
    force=False,
):
    now = ticks_ms()
    interval = max(
        1,
        int(
            getattr(
                cfg, "FREE_RECT_PROGRESS_INTERVAL_MS", 1000
            )
        ),
    )
    if (
        not force
        and ticks_diff(now, local["last_progress_ms"]) < interval
    ):
        return
    print(
        "FREE_PASS_PROGRESS,name={},elapsed_ms={},trees={},"
        "tree_rounds={},cheap_complete={},beam_size={},exact_evaluated={},"
        "physical_valid={},best_cost={}".format(
            pass_name,
            max(0, ticks_diff(now, pass_started_ms)),
            local["trees"],
            local["tree_rounds"],
            local["cheap_complete"],
            local["beam_size"],
            local["exact_evaluated"],
            local["physical_valid"],
            (
                "{:.6f}".format(best["metrics"]["cost"])
                if best is not None
                else "na"
            ),
        )
    )
    local["last_progress_ms"] = now
    exitpoint = getattr(os, "exitpoint", None)
    if exitpoint is not None:
        exitpoint()
    plan_debug_heartbeat()


def _free_exact_proposal(
    item,
    pieces,
    polygons,
    piece_cache,
    source_area,
    perimeter_context,
    state,
):
    matches = item["matches"]
    transforms = item["transforms"]
    if len(matches) >= len(polygons):
        try:
            transforms = _sim_optimize_pose_graph(
                polygons, matches, transforms
            )
        except Exception:
            transforms = None
        state["pose_optimization_count"] += 1
        state["closed_graph_pose_optimizer_count"] += 1
    else:
        state["tree_pose_optimizer_skipped_count"] += 1
    if (
        transforms is None
        or any(
            not _free_transform_valid(transform)
            for transform in transforms
        )
    ):
        return None
    perimeter = _free_exposed_perimeter_metrics(
        polygons,
        matches,
        transforms,
        source_area,
        context=perimeter_context,
    )
    completed = _free_complete_metrics(
        polygons,
        matches,
        transforms,
        source_area,
        perimeter=perimeter,
        piece_cache=piece_cache,
    )
    if completed is None:
        return None
    state["geometrically_valid_complete_count"] += 1
    target, direction_candidates = _free_select_target(
        pieces,
        polygons,
        transforms,
        completed["rectangle"],
    )
    if target is not None:
        state["target_fit_complete_count"] += 1
    full_count = sum(
        1 for match in matches if _sim_is_full_match(match)
    )
    physical_valid, physical_failures = (
        _free_physical_validity(completed["metrics"], target)
    )
    proposal = {
        "matches": matches,
        "match_signature": item["match_signature"],
        "full_count": full_count,
        "partial_count": len(matches) - full_count,
        "topology": item["topology"],
        "optimized_transforms": transforms,
        "metrics": completed["metrics"],
        "seams": completed["seams"],
        "max_length_error_mm": completed[
            "max_length_error_mm"
        ],
        "target": target,
        "direction_candidates": direction_candidates,
        "physical_valid": physical_valid,
        "physical_failures": physical_failures,
        "pass_name": item["pass_name"],
    }
    proposal["strong"] = _free_is_strong_solution(proposal)
    return proposal


def _free_finalize_physical_result(
    best,
    state,
    source_area,
    candidates,
):
    elapsed_ms = max(
        0, ticks_diff(ticks_ms(), state["started_ms"])
    )
    metrics = best["metrics"]
    target = best["target"]
    state["selected_pass"] = best["pass_name"]
    stats = _free_base_stats(state, source_area, candidates)
    stats.update(
        {
            "plan_ms": elapsed_ms,
            "best_cost": metrics["cost"],
            "long_side_mm": metrics["long_side_mm"],
            "short_side_mm": metrics["short_side_mm"],
            "rect_area_mm2": metrics["rect_area_mm2"],
            "overlap_mm2": metrics["overlap_mm2"],
            "overlap_ratio": metrics["overlap_ratio"],
            "fill_gap_mm2": metrics["fill_gap_mm2"],
            "fill_gap_ratio": metrics["fill_gap_ratio"],
            "hull_gap_mm2": metrics["hull_gap_mm2"],
            "hull_gap_ratio": metrics["hull_gap_ratio"],
            "area_prior_error": metrics["area_prior_error"],
            "dimension_range_penalty": metrics[
                "dimension_range_penalty"
            ],
            "outer_piece_missing_count": metrics[
                "outer_piece_missing_count"
            ],
            "outer_piece_missing_ratio": metrics[
                "outer_piece_missing_ratio"
            ],
            "outer_piece_evidence": metrics[
                "outer_piece_evidence"
            ],
            "perimeter_error_ratio": metrics[
                "perimeter_error_ratio"
            ],
            "exposed_perimeter_mm": metrics[
                "exposed_perimeter_mm"
            ],
            "selected_match_count": len(best["matches"]),
            "selected_full_match_count": best["full_count"],
            "selected_partial_match_count": best[
                "partial_count"
            ],
            "selected_topology": best["topology"],
            "selected_seams": best["seams"],
            "selected_quarter_turn_deg": target[
                "quarter_turn_deg"
            ],
            "selected_motion_cost": target["motion_cost"],
            "physical_valid": True,
            "strong": bool(best.get("strong", False)),
            "top_k": [_free_top_record(best)],
        }
    )
    PERF_STATS.add_stage("plan_ms", elapsed_ms=elapsed_ms)
    print(
        "FREE_PLAN_RESULT,valid=1,elapsed_ms={},pass={},strong={},"
        "timed_out={},raw_candidates={},shortlisted_candidates={},"
        "trees_considered={},cheap_complete={},exact_evaluated={},"
        "pose_optimizations={},cost={:.6f},long_mm={:.2f},"
        "short_mm={:.2f},area_error={:.4f},overlap_ratio={:.4f},"
        "gap_ratio={:.4f},outer_missing={},perimeter_error={:.4f}".format(
            elapsed_ms,
            best["pass_name"],
            int(bool(best.get("strong", False))),
            int(bool(state["timed_out"])),
            state["raw_candidate_count"],
            state["shortlisted_candidate_count"],
            state["trees_considered"],
            state["cheap_complete_count"],
            state["exact_evaluated_count"],
            state["pose_optimization_count"],
            metrics["cost"],
            metrics["long_side_mm"],
            metrics["short_side_mm"],
            metrics["area_prior_error"],
            metrics["overlap_ratio"],
            metrics["fill_gap_ratio"],
            metrics["outer_piece_missing_count"],
            metrics["perimeter_error_ratio"],
        )
    )
    return PlanResult(
        valid=True,
        reason="physical rectangle candidate",
        score=metrics["cost"],
        operations=target["operations"],
        target_polygons=target["target_polygons"],
        target_rect=target["target_rect"],
        search_nodes=state["cheap_complete_count"],
        mode=MODE,
        max_vertex_error_mm=best["max_length_error_mm"],
        fill_gap_mm2=metrics["fill_gap_mm2"],
        overlap_mm2=metrics["overlap_mm2"],
        outside_mm2=0.0,
        seams=best["seams"],
        plan_stats=stats,
    )


def _free_plan_staged_generic(
    pieces,
    polygons,
    source_area,
    state,
):
    piece_cache = _free_build_piece_cache(pieces)
    if piece_cache is None:
        return _free_invalid_result(
            "invalid source polygon triangulation",
            state,
            source_area,
            [],
        )
    edge_count = max(0, len(polygons) - 1)
    perimeter_context = _free_perimeter_context(
        polygons, source_area
    )
    total_budget = max(
        1,
        int(getattr(cfg, "FREE_RECT_TOTAL_PLAN_TIME_MS", 10000)),
    )
    exact_cache = {}
    combination_cache = {}
    all_candidates = {}
    top = []
    best_physical = None
    best_invalid = None
    strong_found_ms = None
    strong_initial_cost = None

    for pass_definition in _free_pass_definitions(edge_count):
        total_elapsed = max(
            0, ticks_diff(ticks_ms(), state["started_ms"])
        )
        remaining = total_budget - total_elapsed
        if remaining <= 0:
            state["timed_out"] = True
            break
        pass_budget = min(
            remaining, max(1, pass_definition["budget_ms"])
        )
        pass_started = ticks_ms()
        candidates, details = _free_rect_candidate_matchings(
            pieces,
            full_rel_tolerance=pass_definition[
                "full_rel_tolerance"
            ],
            partial_enabled=pass_definition["partial_enabled"],
            partial_min=pass_definition["partial_min"],
            partial_max=pass_definition["partial_max"],
            return_details=True,
        )
        for candidate in candidates:
            all_candidates[candidate] = True
        state["raw_candidate_count"] += details["raw_count"]
        state["shortlisted_candidate_count"] += details[
            "shortlisted_count"
        ]
        state["candidate_pair_group_count"] = max(
            state["candidate_pair_group_count"],
            details["candidate_pair_group_count"],
        )
        full_count = sum(
            1 for candidate in candidates if _sim_is_full_match(candidate)
        )
        print(
            "FREE_CANDIDATES,pass={},raw={},shortlisted={},full={},"
            "partial={},pairs={},short_edge_repaired=source_stage".format(
                pass_definition["name"],
                details["raw_count"],
                details["shortlisted_count"],
                full_count,
                len(candidates) - full_count,
                _free_pair_counts_text(details["pair_counts"]),
            )
        )
        print(
            "FREE_PASS_START,name={},budget_ms={},candidate_count={},"
            "allowed_partial={},tree_round_robin_quota={}".format(
                pass_definition["name"],
                pass_budget,
                len(candidates),
                "|".join(
                    str(value)
                    for value in pass_definition[
                        "allowed_partial_counts"
                    ]
                ),
                int(
                    getattr(
                        cfg,
                        "FREE_RECT_TREE_ROUND_ROBIN_QUOTA",
                        16,
                    )
                ),
            )
        )
        local = {
            "trees": 0,
            "tree_rounds": 0,
            "cheap_complete": 0,
            "beam_size": 0,
            "exact_evaluated": 0,
            "physical_valid": 0,
            "last_progress_ms": pass_started,
        }
        tree_before = state["trees_considered"]
        rounds_before = state["tree_round_robin_rounds"]
        beams = {}
        cheap_seen = set()
        cheap_budget = max(1, int(pass_budget * 0.60))
        candidates_by_pair = _free_candidates_by_pair(candidates)
        candidate_cache = details.get("candidate_cache", {})
        for matches in _free_tree_matching_sets(
            candidates_by_pair,
            len(polygons),
            set(pass_definition["allowed_partial_counts"]),
            state,
            (pass_started, cheap_budget),
            candidate_cache=candidate_cache,
        ):
            signature = _free_match_signature(matches)
            if signature in cheap_seen:
                state["prefix_pruned_duplicate"] += 1
                continue
            cheap_seen.add(signature)
            cached = combination_cache.get(signature)
            if cached is None:
                transforms = _free_initial_transforms(
                    polygons, matches, alignment="midpoint"
                )
                if transforms is None:
                    state["prefix_pruned_invalid_geometry"] += 1
                    combination_cache[signature] = (None, None)
                    continue
                cheap_metrics = _free_cheap_complete_metrics(
                    polygons,
                    matches,
                    transforms,
                    source_area,
                    perimeter_context,
                    candidate_cache=candidate_cache,
                )
                combination_cache[signature] = (
                    transforms, cheap_metrics
                )
            else:
                transforms, cheap_metrics = cached
                state["cheap_cache_hit_count"] += 1
            if transforms is None or cheap_metrics is None:
                continue
            partial_count = sum(
                1
                for match in matches
                if not _sim_is_full_match(match)
            )
            topology = _free_topology_name(
                edge_count, edge_count - partial_count
            )
            item = {
                "matches": matches,
                "match_signature": signature,
                "transforms": transforms,
                "partial_count": partial_count,
                "topology": topology,
                "cheap_metrics": cheap_metrics,
                "pass_name": pass_definition["name"],
            }
            _free_add_to_cheap_beam(beams, item)
            local["cheap_complete"] += 1
            state["cheap_complete_count"] += 1
            state["complete_matching_set_count"] += 1
            counts = state["matching_topology_counts"]
            counts[topology] = counts.get(topology, 0) + 1
            if local["cheap_complete"] % 128 == 0:
                local["trees"] = (
                    state["trees_considered"] - tree_before
                )
                local["tree_rounds"] = (
                    state["tree_round_robin_rounds"]
                    - rounds_before
                )
                local["beam_size"] = sum(
                    len(beam) for beam in beams.values()
                )
                _free_pass_progress(
                    pass_definition["name"],
                    pass_started,
                    local,
                    best_physical,
                )

        local["trees"] = state["trees_considered"] - tree_before
        local["tree_rounds"] = (
            state["tree_round_robin_rounds"] - rounds_before
        )
        exact_beam = _free_merge_cheap_beams(beams)
        local["beam_size"] = len(exact_beam)
        enumeration_completed = not _free_deadline_reached(
            (pass_started, cheap_budget)
        )
        pass_best = None
        exact_completed = True
        for item in exact_beam:
            if _free_deadline_reached((pass_started, pass_budget)):
                exact_completed = False
                break
            total_elapsed = max(
                0, ticks_diff(ticks_ms(), state["started_ms"])
            )
            if total_elapsed >= total_budget:
                state["timed_out"] = True
                exact_completed = False
                break
            signature = item["match_signature"]
            if signature in exact_cache:
                proposal = exact_cache[signature]
            else:
                proposal = _free_exact_proposal(
                    item,
                    pieces,
                    polygons,
                    piece_cache,
                    source_area,
                    perimeter_context,
                    state,
                )
                exact_cache[signature] = proposal
                state["exact_evaluated_count"] += 1
                local["exact_evaluated"] += 1
            if proposal is None:
                continue
            _free_update_top(top, proposal)
            if (
                best_invalid is None
                or _free_proposal_rank(proposal)
                < _free_proposal_rank(best_invalid)
            ):
                best_invalid = proposal
            if proposal["physical_valid"]:
                state["physical_valid_count"] += 1
                local["physical_valid"] += 1
                if (
                    pass_best is None
                    or _free_proposal_rank(proposal)
                    < _free_proposal_rank(pass_best)
                ):
                    pass_best = proposal
                if (
                    best_physical is None
                    or _free_proposal_rank(proposal)
                    < _free_proposal_rank(best_physical)
                ):
                    best_physical = proposal
                    state["best_cost"] = proposal["metrics"]["cost"]
                    update_plan_debug(
                        stage="free_rect_physical",
                        best_score=state["best_cost"],
                        nodes=state["cheap_complete_count"],
                    )
                if proposal.get("strong", False):
                    if strong_found_ms is None:
                        strong_found_ms = ticks_ms()
                        strong_initial_cost = proposal["metrics"]["cost"]
            if strong_found_ms is not None:
                grace = max(
                    0,
                    int(
                        getattr(
                            cfg,
                            "FREE_RECT_STRONG_SOLUTION_GRACE_MS",
                            400,
                        )
                    ),
                )
                if ticks_diff(ticks_ms(), strong_found_ms) >= grace:
                    improvement = (
                        strong_initial_cost
                        - best_physical["metrics"]["cost"]
                    ) / max(EPS, strong_initial_cost)
                    minimum_improvement = float(
                        getattr(
                            cfg,
                            "FREE_RECT_STRONG_IMPROVEMENT_RATIO",
                            0.03,
                        )
                    )
                    if improvement < minimum_improvement:
                        state["strong_solution_early_exit"] = True
                        break
                    # A material improvement earns one new grace window.  If
                    # the cost then plateaus, the same rule exits promptly.
                    strong_found_ms = ticks_ms()
                    strong_initial_cost = best_physical["metrics"]["cost"]
            if local["exact_evaluated"] % 8 == 0:
                _free_pass_progress(
                    pass_definition["name"],
                    pass_started,
                    local,
                    best_physical,
                )

        reason = "completed"
        if state["timed_out"]:
            reason = "global_timeout"
        elif state["strong_solution_early_exit"]:
            reason = "strong_solution"
        elif (
            best_physical is not None
            and best_physical.get("strong", False)
        ):
            reason = "strong_solution"
        elif not enumeration_completed or not exact_completed:
            reason = "budget"
        _free_pass_progress(
            pass_definition["name"],
            pass_started,
            local,
            best_physical,
            force=True,
        )
        print(
            "FREE_PASS_END,name={},reason={},cheap_complete={},"
            "exact_evaluated={},physical_valid={}".format(
                pass_definition["name"],
                reason,
                local["cheap_complete"],
                local["exact_evaluated"],
                local["physical_valid"],
            )
        )
        state["pass_stats"].append(
            {
                "name": pass_definition["name"],
                "reason": reason,
                "budget_ms": pass_budget,
                "candidate_count": len(candidates),
                "trees": local["trees"],
                "tree_rounds": local["tree_rounds"],
                "cheap_complete": local["cheap_complete"],
                "beam_size": local["beam_size"],
                "exact_evaluated": local["exact_evaluated"],
                "physical_valid": local["physical_valid"],
            }
        )
        if best_physical is not None and (
            state["strong_solution_early_exit"]
            or best_physical.get("strong", False)
            or (enumeration_completed and exact_completed)
        ):
            break
        if state["timed_out"]:
            break

    candidates = sorted(all_candidates)
    if best_physical is not None:
        return _free_finalize_physical_result(
            best_physical,
            state,
            source_area,
            candidates,
        )
    state["best_invalid_proposal"] = best_invalid
    return _free_invalid_result(
        "no_physical_rectangle_candidate",
        state,
        source_area,
        candidates,
        top=top,
    )


def _free_base_stats(state, source_area, candidates):
    full_count = sum(
        1 for candidate in candidates
        if _sim_is_full_match(candidate)
    )
    return {
        "engine": "simulator-free-rectangle-k230",
        "validation": "publish_best",
        "candidate_count": len(candidates),
        "full_candidate_count": full_count,
        "partial_candidate_count": len(candidates) - full_count,
        "raw_candidate_count": state.get("raw_candidate_count", 0),
        "shortlisted_candidate_count": state.get(
            "shortlisted_candidate_count", len(candidates)
        ),
        "candidate_pair_group_count": state.get(
            "candidate_pair_group_count", 0
        ),
        "trees_considered": state.get("trees_considered", 0),
        "tree_schedule_count": state.get(
            "tree_schedule_count", 0
        ),
        "tree_round_robin_rounds": state.get(
            "tree_round_robin_rounds", 0
        ),
        "cheap_complete_count": state.get(
            "cheap_complete_count", 0
        ),
        "exact_evaluated_count": state.get(
            "exact_evaluated_count", 0
        ),
        "cheap_cache_hit_count": state.get(
            "cheap_cache_hit_count", 0
        ),
        "physical_valid_count": state.get(
            "physical_valid_count", 0
        ),
        "tree_pose_optimizer_skipped_count": state.get(
            "tree_pose_optimizer_skipped_count", 0
        ),
        "closed_graph_pose_optimizer_count": state.get(
            "closed_graph_pose_optimizer_count", 0
        ),
        "interval_reuse_reject_count": state.get(
            "interval_reuse_reject_count", 0
        ),
        "strong_solution_early_exit": bool(
            state.get("strong_solution_early_exit", False)
        ),
        "selected_pass": state.get("selected_pass"),
        "pass_stats": list(state.get("pass_stats", ())),
        "prefix_count": state["prefix_count"],
        "enumerated_prefix_count": state["prefix_count"],
        "max_depth": state["max_depth"],
        "complete_matching_set_count": state[
            "complete_matching_set_count"
        ],
        "matching_sets_evaluated": state[
            "complete_matching_set_count"
        ],
        "pose_optimization_count": state[
            "pose_optimization_count"
        ],
        "geometrically_valid_complete_count": state[
            "geometrically_valid_complete_count"
        ],
        "target_fit_complete_count": state[
            "target_fit_complete_count"
        ],
        "perimeter_prefilter_passed_count": state[
            "perimeter_prefilter_passed_count"
        ],
        "perimeter_prefilter_rejected_count": state[
            "perimeter_prefilter_rejected_count"
        ],
        "perimeter_postfit_rejected_count": state[
            "perimeter_postfit_rejected_count"
        ],
        "matching_topology_counts": dict(
            state["matching_topology_counts"]
        ),
        "prefix_pruned_candidate_reuse": state[
            "prefix_pruned_candidate_reuse"
        ],
        "prefix_pruned_edge_reuse": state[
            "prefix_pruned_edge_reuse"
        ],
        "prefix_pruned_connectivity": state[
            "prefix_pruned_connectivity"
        ],
        "prefix_pruned_topology": state[
            "prefix_pruned_topology"
        ],
        "prefix_pruned_duplicate": state[
            "prefix_pruned_duplicate"
        ],
        "prefix_pruned_invalid_geometry": state[
            "prefix_pruned_invalid_geometry"
        ],
        "prefix_pruned_span": state[
            "prefix_pruned_span"
        ],
        "timed_out": bool(state["timed_out"]),
        "limit_hit": bool(state["limit_hit"]),
        "source_area_mm2": source_area,
        "input_piece_area_mm2": source_area,
    }


def _free_invalid_result(
    reason,
    state,
    source_area,
    candidates,
    top=None,
):
    elapsed_ms = max(
        0, ticks_diff(ticks_ms(), state["started_ms"])
    )
    stats = _free_base_stats(state, source_area, candidates)
    stats["plan_ms"] = elapsed_ms
    stats["reason"] = reason
    stats["top_k"] = [
        _free_top_record(proposal)
        for proposal in (top or [])
    ]
    best_invalid = state.get("best_invalid_proposal")
    debug_invalid_enabled = bool(
        getattr(cfg, "FREE_RECT_ALLOW_INVALID_DEBUG_PROPOSAL", False)
    )
    stats["invalid_debug_proposal_enabled"] = debug_invalid_enabled
    stats["invalid_debug_proposal"] = (
        _free_top_record(best_invalid)
        if debug_invalid_enabled and best_invalid is not None
        else None
    )
    if best_invalid is not None:
        best_metrics = best_invalid["metrics"]
        stats.update(
            {
                "best_invalid_cost": best_metrics["cost"],
                "best_invalid_area_error": best_metrics[
                    "area_prior_error"
                ],
                "best_invalid_gap_ratio": best_metrics[
                    "fill_gap_ratio"
                ],
                "best_invalid_overlap_ratio": best_metrics[
                    "overlap_ratio"
                ],
                "best_invalid_outer_missing": best_metrics[
                    "outer_piece_missing_count"
                ],
                "best_invalid_failures": best_invalid.get(
                    "physical_failures", ()
                ),
            }
        )
    PERF_STATS.add_stage("plan_ms", elapsed_ms=elapsed_ms)
    print(
        "FREE_PLAN_INVALID,reason={},candidates={},max_depth={},"
        "complete_sets={},timed_out={},best_invalid_cost={},"
        "best_invalid_area_error={},best_invalid_gap_ratio={},"
        "best_invalid_overlap_ratio={},best_invalid_outer_missing={}".format(
            str(reason).replace(",", ";"),
            len(candidates),
            state["max_depth"],
            state["complete_matching_set_count"],
            1 if state["timed_out"] else 0,
            (
                "{:.6f}".format(best_invalid["metrics"]["cost"])
                if best_invalid is not None
                else "na"
            ),
            (
                "{:.4f}".format(
                    best_invalid["metrics"]["area_prior_error"]
                )
                if best_invalid is not None
                else "na"
            ),
            (
                "{:.4f}".format(
                    best_invalid["metrics"]["fill_gap_ratio"]
                )
                if best_invalid is not None
                else "na"
            ),
            (
                "{:.4f}".format(
                    best_invalid["metrics"]["overlap_ratio"]
                )
                if best_invalid is not None
                else "na"
            ),
            (
                best_invalid["metrics"]["outer_piece_missing_count"]
                if best_invalid is not None
                else "na"
            ),
        )
    )
    return PlanResult(
        reason=reason,
        search_nodes=state["complete_matching_set_count"],
        mode=MODE,
        plan_stats=stats,
    )


def _free_new_state(started_ms):
    return {
        "started_ms": started_ms,
        "last_progress_ms": started_ms,
        "prefix_count": 0,
        "max_depth": 0,
        "complete_matching_set_count": 0,
        "pose_optimization_count": 0,
        "geometrically_valid_complete_count": 0,
        "target_fit_complete_count": 0,
        "perimeter_prefilter_passed_count": 0,
        "perimeter_prefilter_rejected_count": 0,
        "perimeter_postfit_rejected_count": 0,
        "matching_topology_counts": {},
        "prefix_pruned_candidate_reuse": 0,
        "prefix_pruned_edge_reuse": 0,
        "prefix_pruned_connectivity": 0,
        "prefix_pruned_topology": 0,
        "prefix_pruned_duplicate": 0,
        "prefix_pruned_invalid_geometry": 0,
        "prefix_pruned_span": 0,
        "timed_out": False,
        "limit_hit": False,
        "stop": False,
        "best_cost": None,
        "best_aspect_ratio": None,
        "best_perimeter_error_ratio": None,
        "raw_candidate_count": 0,
        "shortlisted_candidate_count": 0,
        "candidate_pair_group_count": 0,
        "trees_considered": 0,
        "tree_schedule_count": 0,
        "tree_round_robin_rounds": 0,
        "cheap_complete_count": 0,
        "cheap_cache_hit_count": 0,
        "exact_evaluated_count": 0,
        "physical_valid_count": 0,
        "tree_pose_optimizer_skipped_count": 0,
        "closed_graph_pose_optimizer_count": 0,
        "interval_reuse_reject_count": 0,
        "strong_solution_early_exit": False,
        "selected_pass": None,
        "pass_stats": [],
    }


def _plan_simulator_free_rectangle(
    pieces, validation, fixed_template_evaluation=None
):
    started_ms = ticks_ms()
    state = _free_new_state(started_ms)
    pieces = list(pieces)
    source_area = sum(
        float(piece.area_mm2) for piece in pieces
    )
    candidates = []
    if validation != "publish_best":
        return _free_invalid_result(
            "free rectangle validation must be publish_best",
            state,
            source_area,
            candidates,
        )
    minimum = int(
        getattr(cfg, "FREE_RECT_MIN_PIECE_COUNT", 1)
    )
    maximum = int(
        getattr(cfg, "FREE_RECT_MAX_PIECE_COUNT", 4)
    )
    if not minimum <= len(pieces) <= maximum:
        return _free_invalid_result(
            "piece count {} outside {}..{}".format(
                len(pieces), minimum, maximum
            ),
            state,
            source_area,
            candidates,
        )
    if source_area <= EPS:
        return _free_invalid_result(
            "source piece area is not positive",
            state,
            source_area,
            candidates,
        )
    polygons = [piece.polygon_mm for piece in pieces]
    if any(not _free_polygon_valid(polygon) for polygon in polygons):
        return _free_invalid_result(
            "invalid source polygon",
            state,
            source_area,
            candidates,
        )
    identity_transforms = [_identity_transform()] * len(polygons)
    initial_connected = [True] * len(polygons)
    input_span = _free_prefix_span(
        polygons, identity_transforms, initial_connected
    )
    # Source pieces may be spread across the acquisition half, so only an
    # individual piece can trigger the catastrophic-span input check.
    for polygon in polygons:
        piece_span = _free_prefix_span(
            [polygon], [_identity_transform()], [True]
        )
        if piece_span is None or piece_span > float(
            getattr(cfg, "FREE_RECT_MAX_SPAN_MM", 170.0)
        ):
            return _free_invalid_result(
                "source polygon exceeds catastrophic span",
                state,
                source_area,
                candidates,
            )
    del identity_transforms, initial_connected, input_span

    if len(pieces) == 4:
        if fixed_template_evaluation is None:
            fixed_match, fixed_reason = (
                match_fixed_figure2_piece_set(pieces)
            )
        else:
            fixed_match, fixed_reason = (
                fixed_template_evaluation
            )
        if fixed_match is not None:
            print(
                "FREE_FIXED_TEMPLATE_CHECK,matched=1,layout={},"
                "matched_pieces={}/4,short_edge_fit_cancelled={},"
                "max_area_ratio_error={:.4f},"
                "max_fit_rms_mm={:.2f},max_fit_vertex_mm={:.2f}".format(
                    fixed_match["layout"],
                    fixed_match.get("matched_piece_count", 4),
                    int(
                        fixed_match.get(
                            "short_edge_fit_cancelled", False
                        )
                    ),
                    max(
                        fixed_match[
                            "area_ratio_errors"
                        ].values()
                    ),
                    fixed_match["maximum_rms_mm"],
                    fixed_match["maximum_vertex_mm"],
                )
            )
            return _free_figure2_direct_result(
                pieces,
                source_area,
                fixed_match,
                state,
            )
        print(
            "FREE_FIXED_TEMPLATE_CHECK,matched=0,reason={},"
            "action=FALLBACK_TO_ENUMERATION".format(
                str(fixed_reason).replace(",", ";")
            )
        )

    minimum_observed_edge = float(
        getattr(cfg, "FREE_RECT_MIN_OBSERVED_EDGE_MM", 17.5)
    )
    unresolved_edges = []
    for piece_index, polygon in enumerate(polygons):
        for edge_index, edge in enumerate(_free_edges(polygon)):
            length = _free_distance(edge[0], edge[1])
            if length + EPS < minimum_observed_edge:
                unresolved_edges.append(
                    (piece_index, edge_index, length)
                )
    if unresolved_edges:
        print(
            "FREE_INPUT_REJECT,reason=short_edge_unresolved,"
            "minimum_mm={:.2f},edges={}".format(
                minimum_observed_edge,
                "|".join(
                    "P{}-E{}:{:.2f}".format(
                        piece_index + 1,
                        edge_index,
                        length,
                    )
                    for piece_index, edge_index, length
                    in unresolved_edges
                ),
            )
        )
        return _free_invalid_result(
            "short_edge_unresolved",
            state,
            source_area,
            candidates,
        )

    print(
        "FREE_PLAN_START,pieces={},source_area_mm2={:.1f},"
        "search=staged_tree_beam,total_time_limit_ms={}".format(
            len(pieces),
            source_area,
            int(
                getattr(
                    cfg, "FREE_RECT_TOTAL_PLAN_TIME_MS", 10000
                )
            ),
        )
    )
    update_plan_debug(stage="free_rect_staged_start")
    return _free_plan_staged_generic(
        pieces,
        polygons,
        source_area,
        state,
    )

    # Legacy exhaustive implementation retained below for desktop A/B source
    # inspection only.  The staged return above is authoritative.
    candidates = _free_rect_candidate_matchings(pieces)
    full_count = sum(
        1 for candidate in candidates
        if _sim_is_full_match(candidate)
    )
    print(
        "FREE_PLAN_START,pieces={},source_area_mm2={:.1f},"
        "candidates={},full={},partial={},time_limit_ms={},"
        "preferred_aspect={:.2f}|{:.2f},"
        "perimeter_max_excess={:.3f}".format(
            len(pieces),
            source_area,
            len(candidates),
            full_count,
            len(candidates) - full_count,
            int(
                getattr(
                    cfg, "FREE_RECT_MAX_PLAN_TIME_MS", 8000
                )
            ),
            float(
                getattr(
                    cfg, "FREE_RECT_PREFERRED_ASPECT_MIN", 1.33
                )
            ),
            float(
                getattr(
                    cfg, "FREE_RECT_PREFERRED_ASPECT_MAX", 1.67
                )
            ),
            float(
                getattr(
                    cfg,
                    "FREE_RECT_MAX_PERIMETER_EXCESS_RATIO",
                    0.18,
                )
            ),
        )
    )
    if len(pieces) > 1 and not candidates:
        return _free_invalid_result(
            "no compatible edge candidates",
            state,
            source_area,
            candidates,
        )

    update_plan_debug(stage="free_rect_candidates")
    top = []
    best = None
    edge_count = max(0, len(polygons) - 1)
    perimeter_context = _free_perimeter_context(
        polygons, source_area
    )
    maximum_perimeter_excess = max(
        0.0,
        float(
            getattr(
                cfg,
                "FREE_RECT_MAX_PERIMETER_EXCESS_RATIO",
                0.18,
            )
        ),
    )
    for matches in _free_rect_matching_sets(
        candidates, polygons, state
    ):
        initial = _free_initial_transforms(
            polygons, matches
        )
        if initial is None:
            continue
        initial_perimeter = _free_exposed_perimeter_metrics(
            polygons,
            matches,
            initial,
            source_area,
            context=perimeter_context,
        )
        if (
            initial_perimeter["perimeter_excess_ratio"]
            > maximum_perimeter_excess
        ):
            state["perimeter_prefilter_rejected_count"] += 1
            _free_progress(state)
            continue
        state["perimeter_prefilter_passed_count"] += 1
        try:
            optimized = _sim_optimize_pose_graph(
                polygons, matches, initial
            )
        except Exception:
            optimized = None
        state["pose_optimization_count"] += 1
        if (
            optimized is None
            or any(
                not _free_transform_valid(transform)
                for transform in optimized
            )
        ):
            continue
        optimized_perimeter = _free_exposed_perimeter_metrics(
            polygons,
            matches,
            optimized,
            source_area,
            context=perimeter_context,
        )
        if (
            optimized_perimeter["perimeter_excess_ratio"]
            > maximum_perimeter_excess
        ):
            state["perimeter_postfit_rejected_count"] += 1
            _free_progress(state)
            continue
        completed = _free_complete_metrics(
            polygons,
            matches,
            optimized,
            source_area,
            perimeter=optimized_perimeter,
        )
        if completed is None:
            continue
        state["geometrically_valid_complete_count"] += 1
        target, direction_candidates = _free_select_target(
            pieces,
            polygons,
            optimized,
            completed["rectangle"],
        )
        if target is not None:
            state["target_fit_complete_count"] += 1
        full_selected = sum(
            1 for match in matches
            if _sim_is_full_match(match)
        )
        proposal = {
            "matches": matches,
            "match_signature": _free_match_signature(matches),
            "full_count": full_selected,
            "partial_count": len(matches) - full_selected,
            "topology": _free_topology_name(
                edge_count, full_selected
            ),
            "optimized_transforms": optimized,
            "metrics": completed["metrics"],
            "seams": completed["seams"],
            "max_length_error_mm": completed[
                "max_length_error_mm"
            ],
            "target": target,
            "direction_candidates": direction_candidates,
        }
        _free_update_top(top, proposal)
        if target is not None and (
            best is None
            or _free_proposal_rank(proposal)
            < _free_proposal_rank(best)
        ):
            best = proposal
            state["best_cost"] = proposal["metrics"]["cost"]
            state["best_aspect_ratio"] = proposal["metrics"][
                "aspect_ratio"
            ]
            state["best_perimeter_error_ratio"] = proposal[
                "metrics"
            ]["perimeter_error_ratio"]
            update_plan_debug(
                stage="free_rect_complete",
                best_score=state["best_cost"],
                nodes=state["complete_matching_set_count"],
            )
        _free_progress(state)

    if state["timed_out"]:
        _free_progress(state, force=True)
    if best is None:
        if (
            state["timed_out"]
            and state["complete_matching_set_count"] == 0
        ):
            reason = "no complete candidate before timeout"
        elif state["complete_matching_set_count"] == 0:
            reason = "no complete connected matching set"
        elif state["geometrically_valid_complete_count"] == 0:
            if (
                state["perimeter_prefilter_rejected_count"]
                + state["perimeter_postfit_rejected_count"]
                > 0
            ):
                reason = (
                    "all complete candidates rejected by geometry or "
                    "exposed perimeter"
                )
            else:
                reason = "all complete candidates geometrically invalid"
        else:
            reason = "no complete proposal fits A4 lower target region"
        return _free_invalid_result(
            reason,
            state,
            source_area,
            candidates,
            top=top,
        )

    elapsed_ms = max(
        0, ticks_diff(ticks_ms(), started_ms)
    )
    metrics = best["metrics"]
    target = best["target"]
    warnings = []
    if metrics["overlap_ratio"] > float(
        getattr(cfg, "FREE_RECT_WARN_OVERLAP_RATIO", 0.03)
    ):
        warnings.append("overlap")
    if metrics["fill_gap_ratio"] > float(
        getattr(cfg, "FREE_RECT_WARN_FILL_GAP_RATIO", 0.08)
    ):
        warnings.append("fill_gap")
    if metrics["hull_gap_ratio"] > float(
        getattr(cfg, "FREE_RECT_WARN_HULL_GAP_RATIO", 0.08)
    ):
        warnings.append("hull_gap")
    if metrics["dimension_range_penalty"] > 0.0:
        warnings.append("dimension_range")
    if not metrics["aspect_preferred"]:
        warnings.append("aspect_range")
    if metrics["outer_piece_missing_ratio"] > 0.0:
        warnings.append("outer_piece")
    if metrics["perimeter_error_ratio"] > float(
        getattr(
            cfg,
            "FREE_RECT_WARN_PERIMETER_ERROR_RATIO",
            0.08,
        )
    ):
        warnings.append("exposed_perimeter")
    if state["timed_out"]:
        warnings.append("timed_out_best_so_far")
    if state["limit_hit"]:
        warnings.append("complete_set_limit_best_so_far")
    reason = (
        "publish_best"
        if not warnings
        else "publish_best; warnings: {}".format(
            "|".join(warnings)
        )
    )
    stats = _free_base_stats(state, source_area, candidates)
    stats.update(
        {
            "plan_ms": elapsed_ms,
            "top_k": [
                _free_top_record(proposal)
                for proposal in top
            ],
            "best_cost": metrics["cost"],
            "long_side_mm": metrics["long_side_mm"],
            "short_side_mm": metrics["short_side_mm"],
            "aspect_ratio": metrics["aspect_ratio"],
            "aspect_preferred": metrics["aspect_preferred"],
            "aspect_range_penalty": metrics[
                "aspect_range_penalty"
            ],
            "rect_area_mm2": metrics["rect_area_mm2"],
            "overlap_mm2": metrics["overlap_mm2"],
            "overlap_ratio": metrics["overlap_ratio"],
            "fill_gap_mm2": metrics["fill_gap_mm2"],
            "fill_gap_ratio": metrics["fill_gap_ratio"],
            "hull_gap_mm2": metrics["hull_gap_mm2"],
            "hull_gap_ratio": metrics["hull_gap_ratio"],
            "area_prior_error": metrics["area_prior_error"],
            "dimension_range_penalty": metrics[
                "dimension_range_penalty"
            ],
            "outer_piece_missing_count": metrics[
                "outer_piece_missing_count"
            ],
            "outer_piece_missing_ratio": metrics[
                "outer_piece_missing_ratio"
            ],
            "outer_piece_evidence": metrics[
                "outer_piece_evidence"
            ],
            "source_perimeter_mm": metrics[
                "source_perimeter_mm"
            ],
            "covered_perimeter_mm": metrics[
                "covered_perimeter_mm"
            ],
            "internal_seam_length_mm": metrics[
                "internal_seam_length_mm"
            ],
            "selected_shared_length_mm": metrics[
                "selected_shared_length_mm"
            ],
            "additional_shared_length_mm": metrics[
                "additional_shared_length_mm"
            ],
            "additional_contact_count": metrics[
                "additional_contact_count"
            ],
            "exposed_perimeter_mm": metrics[
                "exposed_perimeter_mm"
            ],
            "expected_perimeter_min_mm": metrics[
                "expected_perimeter_min_mm"
            ],
            "expected_perimeter_max_mm": metrics[
                "expected_perimeter_max_mm"
            ],
            "perimeter_deficit_ratio": metrics[
                "perimeter_deficit_ratio"
            ],
            "perimeter_excess_ratio": metrics[
                "perimeter_excess_ratio"
            ],
            "perimeter_error_ratio": metrics[
                "perimeter_error_ratio"
            ],
            "seam_cost": metrics["seam_cost"],
            "closure_error_mm": metrics["closure_error_mm"],
            "closure_cost": metrics["closure_cost"],
            "selected_match_count": len(best["matches"]),
            "selected_full_match_count": best["full_count"],
            "selected_partial_match_count": best[
                "partial_count"
            ],
            "selected_topology": best["topology"],
            "selected_seams": best["seams"],
            "selected_quarter_turn_deg": target[
                "quarter_turn_deg"
            ],
            "selected_motion_cost": target["motion_cost"],
            "equivalent_direction_candidates": best[
                "direction_candidates"
            ],
            "warnings": warnings,
        }
    )
    PERF_STATS.add_stage("plan_ms", elapsed_ms=elapsed_ms)
    print(
        "FREE_PLAN_RESULT,valid=1,timed_out={},complete_sets={},"
        "cost={:.6f},long_mm={:.2f},short_mm={:.2f},"
        "aspect={:.3f},aspect_preferred={},"
        "rect_area_mm2={:.1f},overlap_mm2={:.1f},gap_mm2={:.1f},"
        "hull_gap_mm2={:.1f},area_error={:.6f},"
        "outer_missing={:.3f},exposed_perimeter_mm={:.1f},"
        "expected_perimeter_mm={:.1f}|{:.1f},"
        "perimeter_error={:.3f},additional_contacts={},"
        "perimeter_rejected={}|{},selected_partial={}".format(
            1 if state["timed_out"] else 0,
            state["complete_matching_set_count"],
            metrics["cost"],
            metrics["long_side_mm"],
            metrics["short_side_mm"],
            metrics["aspect_ratio"],
            1 if metrics["aspect_preferred"] else 0,
            metrics["rect_area_mm2"],
            metrics["overlap_mm2"],
            metrics["fill_gap_mm2"],
            metrics["hull_gap_mm2"],
            metrics["area_prior_error"],
            metrics["outer_piece_missing_ratio"],
            metrics["exposed_perimeter_mm"],
            metrics["expected_perimeter_min_mm"],
            metrics["expected_perimeter_max_mm"],
            metrics["perimeter_error_ratio"],
            metrics["additional_contact_count"],
            state["perimeter_prefilter_rejected_count"],
            state["perimeter_postfit_rejected_count"],
            best["partial_count"],
        )
    )
    return PlanResult(
        valid=True,
        reason=reason,
        score=metrics["cost"],
        operations=target["operations"],
        target_polygons=target["target_polygons"],
        target_rect=target["target_rect"],
        search_nodes=state["complete_matching_set_count"],
        mode=MODE,
        max_vertex_error_mm=best["max_length_error_mm"],
        fill_gap_mm2=metrics["fill_gap_mm2"],
        overlap_mm2=metrics["overlap_mm2"],
        outside_mm2=0.0,
        seams=best["seams"],
        plan_stats=stats,
    )


def plan_simulator_free_rectangle(
    pieces,
    validation="publish_best",
    fixed_template_evaluation=None,
):
    """Return the best complete rigid free-size rectangle proposal."""
    try:
        return _plan_simulator_free_rectangle(
            pieces,
            validation,
            fixed_template_evaluation=fixed_template_evaluation,
        )
    except Exception as exc:
        started_ms = ticks_ms()
        state = _free_new_state(started_ms)
        try:
            source_area = sum(
                float(piece.area_mm2) for piece in pieces
            )
        except Exception:
            source_area = 0.0
        reason = "fatal exception: {}".format(exc)
        return _free_invalid_result(
            reason, state, source_area, []
        )
