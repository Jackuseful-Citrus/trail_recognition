"""Experimental free-size rectangle planner for one to four white pieces.

This module is intentionally isolated from ``puzzle_simulator_planner``.  It
reuses that backend's edge-match representation and rigid pose optimizer, but
owns its candidate shortlist, connected-tree search, time budget, scoring, and
publish-best result semantics.  In particular, overlap, gap, outside area, and
rectangle dimensions are never prefix hard gates.
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
    polygon_centroid,
    polygon_overlap_area,
    transform_point,
    transform_polygon,
    update_plan_debug,
)
from puzzle_perf import PERF_STATS, ticks_diff, ticks_ms
from puzzle_simulator_planner import (
    _sim_align_edge,
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


def _free_raw_candidate_matchings(pieces):
    """Generate the simulator match tuples before any global shortlist."""
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
    )
    partial_min = float(
        getattr(cfg, "FREE_RECT_PARTIAL_MIN_RATIO", 0.22)
    )
    partial_max = float(
        getattr(cfg, "FREE_RECT_PARTIAL_MAX_RATIO", 0.88)
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


def _free_candidate_shortlist(candidates):
    """Reserve both full and partial matches, then fill by original cost."""
    maximum = max(
        1, int(getattr(cfg, "FREE_RECT_MAX_CANDIDATES", 80))
    )
    candidates = sorted(candidates)
    if len(candidates) <= maximum:
        return candidates
    full = [
        candidate
        for candidate in candidates
        if _sim_is_full_match(candidate)
    ]
    partial = [
        candidate
        for candidate in candidates
        if not _sim_is_full_match(candidate)
    ]
    full_reserve = min(
        len(full),
        max(
            0,
            int(
                getattr(
                    cfg, "FREE_RECT_MIN_FULL_SHORTLIST", 24
                )
            ),
        ),
        maximum,
    )
    partial_reserve = min(
        len(partial),
        max(
            0,
            int(
                getattr(
                    cfg, "FREE_RECT_MIN_PARTIAL_SHORTLIST", 40
                )
            ),
        ),
        maximum - full_reserve,
    )
    selected = full[:full_reserve] + partial[:partial_reserve]
    selected_set = set(selected)
    for candidate in candidates:
        if len(selected) >= maximum:
            break
        if candidate in selected_set:
            continue
        selected.append(candidate)
        selected_set.add(candidate)
    selected.sort()
    return selected


def _free_rect_candidate_matchings(pieces):
    """Reuse the fixed candidate semantics with a free-only class reserve."""
    simulator_shortlist = simulator_candidate_matchings(pieces)
    raw_candidates = _free_raw_candidate_matchings(pieces)
    # When the current simulator function did not truncate anything, use its
    # result verbatim. Complex inputs use the independent free-only shortlist
    # so a prolific class cannot remove the other class.
    if (
        len(raw_candidates)
        <= int(getattr(cfg, "SIMULATOR_MAX_CANDIDATES", 80))
        and simulator_shortlist == raw_candidates
        and len(raw_candidates)
        <= int(getattr(cfg, "FREE_RECT_MAX_CANDIDATES", 80))
    ):
        return simulator_shortlist
    return _free_candidate_shortlist(raw_candidates)


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


def _free_initial_transforms(polygons, matches):
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
            transforms[neighbor] = _sim_align_edge(
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


def _free_outer_piece_evidence(assembled, rectangle):
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
        getattr(
            cfg, "FREE_RECT_OUTER_EDGE_TOLERANCE_MM", 5.0
        )
    )
    evidence = []
    missing = 0
    for piece_index, polygon in enumerate(oriented):
        boundary_edges = []
        for edge_index, edge in enumerate(_free_edges(polygon)):
            a, b = edge
            side = None
            if (
                abs(a[0] - min_x) <= tolerance
                and abs(b[0] - min_x) <= tolerance
            ):
                side = "left"
            elif (
                abs(a[0] - max_x) <= tolerance
                and abs(b[0] - max_x) <= tolerance
            ):
                side = "right"
            elif (
                abs(a[1] - min_y) <= tolerance
                and abs(b[1] - min_y) <= tolerance
            ):
                side = "top"
            elif (
                abs(a[1] - max_y) <= tolerance
                and abs(b[1] - max_y) <= tolerance
            ):
                side = "bottom"
            if side is not None:
                boundary_edges.append(
                    {
                        "edge_index": edge_index,
                        "side": side,
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
    overlap_area = 0.0
    for index, polygon in enumerate(assembled):
        for earlier in range(index):
            overlap_area += polygon_overlap_area(
                polygon, assembled[earlier]
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
    aspect_minimum = float(
        getattr(cfg, "FREE_RECT_PREFERRED_ASPECT_MIN", 1.33)
    )
    aspect_maximum = float(
        getattr(cfg, "FREE_RECT_PREFERRED_ASPECT_MAX", 1.67)
    )
    if aspect_minimum > aspect_maximum:
        aspect_minimum, aspect_maximum = (
            aspect_maximum,
            aspect_minimum,
        )
    aspect_range_penalty = _free_interval_penalty(
        aspect_ratio, aspect_minimum, aspect_maximum
    )
    dimension_penalty = _free_interval_penalty(
        long_side,
        float(
            getattr(cfg, "FREE_RECT_LONG_SIDE_MIN_MM", 90.0)
        ),
        float(
            getattr(cfg, "FREE_RECT_LONG_SIDE_MAX_MM", 120.0)
        ),
    ) + _free_interval_penalty(
        short_side,
        float(
            getattr(cfg, "FREE_RECT_SHORT_SIDE_MIN_MM", 50.0)
        ),
        float(
            getattr(cfg, "FREE_RECT_SHORT_SIDE_MAX_MM", 90.0)
        ),
    )
    outer_evidence, missing_count, missing_ratio = (
        _free_outer_piece_evidence(assembled, rectangle)
    )
    seam = _free_seam_metrics(
        polygons, matches, transforms
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
        "area_prior_error": (
            abs(rectangle_area - source_area_mm2)
            / source_area
        ),
        "long_side_mm": long_side,
        "short_side_mm": short_side,
        "aspect_ratio": aspect_ratio,
        "aspect_preferred": aspect_range_penalty <= EPS,
        "aspect_range_penalty": aspect_range_penalty,
        "dimension_range_penalty": dimension_penalty,
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
    )
    metrics["cost"] = cost
    return {
        "assembled": assembled,
        "rectangle": rectangle,
        "metrics": metrics,
        "seams": seam["records"],
        "max_length_error_mm": seam["max_length_error_mm"],
    }


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
        (
            match[1],
            match[2],
            match[3],
            match[4],
            tuple(match[5:]),
        )
        for match in matches
    )


def _free_top_record(proposal):
    metrics = proposal["metrics"]
    return {
        "cost": metrics["cost"],
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
        "seam_cost": metrics["seam_cost"],
        "closure_cost": metrics["closure_cost"],
        "selected_partial_match_count": proposal[
            "partial_count"
        ],
        "selected_seams": proposal["seams"],
    }


def _free_proposal_rank(proposal):
    """Prefer the requested aspect band, then ordinary geometric quality."""
    metrics = proposal["metrics"]
    return (
        0 if metrics["aspect_preferred"] else 1,
        metrics["aspect_range_penalty"],
        metrics["cost"],
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
        "best_cost={},best_aspect={}".format(
            elapsed,
            state["complete_matching_set_count"],
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
        )
    )
    state["last_progress_ms"] = now
    exitpoint = getattr(os, "exitpoint", None)
    if exitpoint is not None:
        exitpoint()
    plan_debug_heartbeat()


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
    PERF_STATS.add_stage("plan_ms", elapsed_ms=elapsed_ms)
    print(
        "FREE_PLAN_INVALID,reason={},candidates={},max_depth={},"
        "complete_sets={},timed_out={}".format(
            str(reason).replace(",", ";"),
            len(candidates),
            state["max_depth"],
            state["complete_matching_set_count"],
            1 if state["timed_out"] else 0,
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

    candidates = _free_rect_candidate_matchings(pieces)
    full_count = sum(
        1 for candidate in candidates
        if _sim_is_full_match(candidate)
    )
    print(
        "FREE_PLAN_START,pieces={},source_area_mm2={:.1f},"
        "candidates={},full={},partial={},time_limit_ms={},"
        "preferred_aspect={:.2f}|{:.2f}".format(
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
    for matches in _free_rect_matching_sets(
        candidates, polygons, state
    ):
        initial = _free_initial_transforms(
            polygons, matches
        )
        if initial is None:
            continue
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
        completed = _free_complete_metrics(
            polygons,
            matches,
            optimized,
            source_area,
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
        "outer_missing={:.3f},selected_partial={}".format(
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
