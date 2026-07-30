"""K230-compatible port of the puzzle-vision-simulator assembly planner.

The matching and pose-graph behaviour follows
https://github.com/lvreng/puzzle-vision-simulator at revision
e9eb2e0fb945c348eedd0b0fa9258f5518d2892f.  The upstream implementation uses
NumPy and OpenCV and therefore cannot run on CanMV MicroPython.  This module
implements the same solver stages with the repository's pure-Python polygon
geometry:

* tolerant full-edge candidates;
* long-edge/short-edge T-junction candidates;
* connected matching-set enumeration;
* rigid pose propagation and global pose-graph refinement;
* global rectangularity scoring and target normalization.

The referenced repository did not contain a root license file at the pinned
revision.  This is consequently an independently written compatibility port,
not a vendored copy of the upstream source.
"""

import math
import os

import puzzle_config as cfg
from puzzle_geometry import (
    EPS,
    PlanResult,
    _choose_smallest_equivalent_rotation,
    _fixed_complete_metrics,
    _identity_transform,
    _normalize_pieces_for_known_target,
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


UPSTREAM_REVISION = "e9eb2e0fb945c348eedd0b0fa9258f5518d2892f"
SUPPORTED_CUT_MODES = (
    "auto",
    "common",
    "boundary_fan",
    "strips",
    "equal_rectangles",
    "t_junction",
    "corner",
    "concave",
    "sequential",
)


def _sim_distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


def _sim_edges(polygon):
    return [
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]


def _sim_interpolate(a, b, fraction):
    return (
        a[0] + (b[0] - a[0]) * fraction,
        a[1] + (b[1] - a[1]) * fraction,
    )


def _sim_match_segments(polygons, match):
    _, i, edge_i, j, edge_j, ia0, ia1, ja0, ja1 = match
    a, b = _sim_edges(polygons[i])[edge_i]
    c, d = _sim_edges(polygons[j])[edge_j]
    return (
        _sim_interpolate(a, b, ia0),
        _sim_interpolate(a, b, ia1),
        _sim_interpolate(c, d, ja0),
        _sim_interpolate(c, d, ja1),
    )


def _sim_candidate_shortlist(candidates):
    """Keep scored full/partial candidates without starving either class."""
    maximum = max(
        1, int(getattr(cfg, "SIMULATOR_MAX_CANDIDATES", 80))
    )
    candidates.sort()
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
        int(
            getattr(
                cfg,
                "SIMULATOR_MIN_FULL_SHORTLIST",
                max(1, maximum * 3 // 10),
            )
        ),
        maximum,
    )
    partial_reserve = min(
        len(partial),
        int(
            getattr(
                cfg,
                "SIMULATOR_MIN_PARTIAL_SHORTLIST",
                maximum // 2,
            )
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


def simulator_candidate_matchings(pieces, target_size_mm=None):
    """Return geometry-ranked full and partial edge candidates."""
    polygons = [
        piece.polygon_mm if hasattr(piece, "polygon_mm") else piece
        for piece in pieces
    ]
    target_size = tuple(
        target_size_mm
        or cfg.TARGET_RECT_SIZE_MM
        or (100.0, 60.0)
    )
    total_polygon_area = sum(
        polygon_area(polygon) for polygon in polygons
    )
    indexed_edges = []
    for piece_index, polygon in enumerate(polygons):
        for edge_index, edge in enumerate(_sim_edges(polygon)):
            indexed_edges.append((piece_index, edge_index, edge))

    candidates = []
    rel_tolerance = float(
        getattr(cfg, "SIMULATOR_MATCH_REL_TOLERANCE", 0.12)
    )
    partial_min = float(
        getattr(cfg, "SIMULATOR_PARTIAL_MIN_RATIO", 0.22)
    )
    partial_max = float(
        getattr(cfg, "SIMULATOR_PARTIAL_MAX_RATIO", 0.88)
    )
    partial_penalty = float(
        getattr(cfg, "SIMULATOR_PARTIAL_MATCH_PENALTY", 0.15)
    )

    def add_candidate(candidate):
        score = _sim_candidate_score(
            polygons,
            candidate,
            target_size,
            total_area=total_polygon_area,
        )
        candidates.append((score,) + candidate[1:])

    for left in range(len(indexed_edges)):
        i, edge_i, edge_a = indexed_edges[left]
        length_a = _sim_distance(edge_a[0], edge_a[1])
        if length_a <= EPS:
            continue
        for right in range(left + 1, len(indexed_edges)):
            j, edge_j, edge_b = indexed_edges[right]
            if i == j:
                continue
            length_b = _sim_distance(edge_b[0], edge_b[1])
            if length_b <= EPS:
                continue
            relative_error = abs(length_a - length_b) / max(
                length_a, length_b
            )
            if relative_error < rel_tolerance:
                add_candidate(
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
            if partial_min <= ratio <= partial_max:
                if length_a > length_b:
                    add_candidate(
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
                    add_candidate(
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
                    add_candidate(
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
                    add_candidate(
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
    return _sim_candidate_shortlist(candidates)


def _sim_is_full_match(match):
    return match[5:] == (0.0, 1.0, 0.0, 1.0)


def _sim_combinations(values, count, start=0, prefix=None):
    """Small MicroPython-safe replacement for itertools.combinations."""
    if prefix is None:
        prefix = []
    if count == 0:
        yield tuple(prefix)
        return
    last = len(values) - count
    for index in range(start, last + 1):
        prefix.append(values[index])
        for result in _sim_combinations(
            values, count - 1, index + 1, prefix
        ):
            yield result
        prefix.pop()


def _sim_valid_matching_set(combo, piece_count, cut_mode):
    used = set()
    degree = [0] * piece_count
    graph = [[] for _ in range(piece_count)]
    for match in combo:
        _, i, edge_i, j, edge_j = match[:5]
        if (i, edge_i) in used or (j, edge_j) in used:
            return False
        used.add((i, edge_i))
        used.add((j, edge_j))
        degree[i] += 1
        degree[j] += 1
        graph[i].append(j)
        graph[j].append(i)
    if any(value == 0 for value in degree):
        return False
    if (
        cut_mode == "common"
        and piece_count >= 3
        and any(value != 2 for value in degree)
    ):
        return False
    seen = set([0])
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor in graph[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == piece_count


def _sim_prefix_prune_reason(metrics, depth):
    """Apply progressively tighter physical gates to a connected prefix."""
    if metrics is None:
        return "dimension"
    multiplier = 1.0 + 0.5 * max(0, 3 - int(depth))
    dimension_limit = float(
        getattr(
            cfg,
            "SIMULATOR_PREFIX_DIMENSION_EXCESS_MAX_MM",
            30.0,
        )
    ) * multiplier
    outside_limit = float(
        getattr(
            cfg,
            "SIMULATOR_PREFIX_OUTSIDE_MAX_MM2",
            float(cfg.FIXED_RECT_MAX_OUTSIDE_MM2) * 2.0,
        )
    ) * multiplier
    overlap_limit = float(
        getattr(
            cfg,
            "SIMULATOR_PREFIX_OVERLAP_MAX_MM2",
            float(cfg.FIXED_RECT_MAX_OVERLAP_MM2) * 5.0,
        )
    ) * multiplier
    gap_limit = float(
        getattr(
            cfg,
            "SIMULATOR_PREFIX_GAP_MAX_MM2",
            float(cfg.FIXED_RECT_MAX_GAP_MM2) * 2.0,
        )
    ) * multiplier
    if metrics["dimension_excess_mm"] > dimension_limit:
        return "dimension"
    if metrics["outside_mm2"] > outside_limit:
        return "outside"
    if metrics["overlap_mm2"] > overlap_limit:
        return "overlap"
    if metrics["gap_lower_bound_mm2"] > gap_limit:
        return "gap"
    return None


def _sim_four_piece_pattern_sets(
    candidates,
    full_required,
    partial_required,
    polygons,
    target,
    state,
    topology_name,
):
    """Grow connected four-piece trees with geometry pruning per edge."""
    typed = {
        "full": [
            (index, match)
            for index, match in enumerate(candidates)
            if _sim_is_full_match(match)
        ],
        "partial": [
            (index, match)
            for index, match in enumerate(candidates)
            if not _sim_is_full_match(match)
        ],
    }
    total_area = sum(
        polygon_area(polygon) for polygon in polygons
    )
    topology_prefix_counts = state[
        "matching_topology_prefix_counts"
    ]
    topology_prefix_counts[topology_name] = 0
    topology_prefix_limit = max(
        1,
        int(
            getattr(
                cfg,
                "SIMULATOR_PREFIXES_PER_TOPOLOGY_MAX",
                600,
            )
        ),
    )
    prefix_seen = set()
    final_seen = set()
    max_time_ms = int(getattr(cfg, "MAX_PLAN_TIME_MS", 3000))

    def walk(
        full_remaining,
        partial_remaining,
        matches,
        match_ids,
        used_pieces,
        used_edges,
    ):
        if state.get("timed_out"):
            return
        depth = len(matches)
        if full_remaining <= 0 and partial_remaining <= 0:
            key = tuple(sorted(match_ids))
            if key in final_seen:
                return
            final_seen.add(key)
            combo = tuple(matches)
            if _sim_valid_matching_set(combo, 4, "auto"):
                yield combo
            return
        option_types = []
        if full_remaining > 0:
            option_types.append("full")
        if partial_remaining > 0:
            option_types.append("partial")
        options = []
        for match_type in option_types:
            options.extend(
                (match_id, match_type, match)
                for match_id, match in typed[match_type]
            )
        options.sort(key=lambda item: item[2][0])
        branch_limit = int(
            getattr(
                cfg,
                (
                    "SIMULATOR_MATCHING_ROOT_BRANCH_LIMIT"
                    if depth == 0
                    else "SIMULATOR_MATCHING_BRANCH_LIMIT"
                ),
                24 if depth == 0 else 10,
            )
        )
        branches = 0
        attempts = 0
        for match_id, match_type, match in options:
            if (
                topology_prefix_counts[topology_name]
                >= topology_prefix_limit
            ):
                state["matching_pruned_topology_budget"] += 1
                return
            state["matching_prefixes_considered"] += 1
            if (
                max_time_ms > 0
                and ticks_diff(
                    ticks_ms(), state["started_ms"]
                )
                >= max_time_ms
            ):
                state["timed_out"] = True
                return
            if match_id in match_ids:
                continue
            _, i, edge_i, j, edge_j = match[:5]
            if (
                (i, edge_i) in used_edges
                or (j, edge_j) in used_edges
            ):
                state["matching_pruned_edge_reuse"] += 1
                continue
            if depth > 0:
                i_used = i in used_pieces
                j_used = j in used_pieces
                if i_used == j_used:
                    state["matching_pruned_topology"] += 1
                    continue
            new_ids = match_ids + (match_id,)
            next_full = full_remaining - (
                1 if match_type == "full" else 0
            )
            next_partial = partial_remaining - (
                1 if match_type == "partial" else 0
            )
            prefix_key = (
                tuple(sorted(new_ids)),
                next_full,
                next_partial,
            )
            if prefix_key in prefix_seen:
                state["matching_pruned_duplicate"] += 1
                continue
            prefix_seen.add(prefix_key)
            attempts += 1
            if attempts > max(1, branch_limit) * 2:
                state["matching_pruned_branch_limit"] += 1
                break
            topology_prefix_counts[topology_name] += 1
            new_matches = matches + (match,)
            metric_key = tuple(sorted(new_ids))
            reason_cache = state[
                "_matching_prefix_reason_cache"
            ]
            if metric_key in reason_cache:
                reason = reason_cache[metric_key]
                state["matching_prefix_cache_hits"] += 1
            else:
                state["matching_prefixes_evaluated"] += 1
                metrics = _sim_prefix_metrics(
                    polygons,
                    new_matches,
                    target,
                    total_area=total_area,
                )
                reason = _sim_prefix_prune_reason(
                    metrics, depth + 1
                )
                if len(reason_cache) < int(
                    getattr(
                        cfg,
                        "SIMULATOR_PREFIX_CACHE_MAX",
                        512,
                    )
                ):
                    reason_cache[metric_key] = reason
            if reason is not None:
                state[
                    "matching_pruned_{}".format(reason)
                ] += 1
                continue
            branches += 1
            if branches > max(1, branch_limit):
                state["matching_pruned_branch_limit"] += 1
                break
            next_pieces = set(used_pieces)
            next_pieces.add(i)
            next_pieces.add(j)
            next_edges = set(used_edges)
            next_edges.add((i, edge_i))
            next_edges.add((j, edge_j))
            for result in walk(
                next_full,
                next_partial,
                new_matches,
                new_ids,
                next_pieces,
                next_edges,
            ):
                yield result

    for combo in walk(
        int(full_required),
        int(partial_required),
        (),
        (),
        set(),
        set(),
    ):
        yield combo


def _sim_matching_sets(
    candidates,
    piece_count,
    cut_mode,
    state,
    polygons=None,
    target=None,
):
    if piece_count == 1:
        yield ()
        return
    pair_count = (
        piece_count
        if (
            (cut_mode == "common" and piece_count >= 3)
            or (cut_mode == "concave" and piece_count >= 2)
        )
        else piece_count - 1
    )
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

    # Four-piece auto mode searches all spanning-tree full/partial mixes. Each
    # topology advances round-robin so an abundant 3-full family cannot consume
    # the global cap before 2F+1P, 1F+2P, and 3P receive search time.
    if (
        piece_count == 4
        and cut_mode == "auto"
        and polygons is not None
        and target is not None
    ):
        topology_specs = (
            ("3_full", 3, 0),
            ("2_full_1_partial", 2, 1),
            ("1_full_2_partial", 1, 2),
            ("3_partial", 0, 3),
        )
        generators = [
            (
                name,
                _sim_four_piece_pattern_sets(
                    candidates,
                    full_required,
                    partial_required,
                    polygons,
                    target,
                    state,
                    name,
                ),
            )
            for (
                name,
                full_required,
                partial_required,
            ) in topology_specs
        ]
        maximum = int(
            getattr(
                cfg, "SIMULATOR_MAX_MATCHING_SETS", 4000
            )
        )
        while generators and not state.get("timed_out"):
            active = []
            for name, generator in generators:
                if state["matching_sets_evaluated"] >= maximum:
                    state["limit_hit"] = True
                    return
                try:
                    combo = next(generator)
                except StopIteration:
                    continue
                state["matching_sets_evaluated"] += 1
                counts = state["matching_topology_counts"]
                counts[name] = counts.get(name, 0) + 1
                if state["matching_sets_evaluated"] % 32 == 0:
                    update_plan_debug(
                        stage="simulator_topology",
                        nodes=state[
                            "matching_sets_evaluated"
                        ],
                    )
                    exitpoint = getattr(os, "exitpoint", None)
                    if exitpoint is not None:
                        exitpoint()
                    plan_debug_heartbeat()
                yield combo
                active.append((name, generator))
            generators = active
        return

    def allowed():
        if cut_mode == "t_junction" and piece_count >= 3:
            for base in _sim_combinations(full, pair_count - 1):
                for part in partial:
                    yield base + (part,)
            return
        if cut_mode in (
            "common",
            "boundary_fan",
            "strips",
            "corner",
            "concave",
            "equal_rectangles",
            "sequential",
        ):
            for combo in _sim_combinations(full, pair_count):
                yield combo
            return
        for combo in _sim_combinations(full, pair_count):
            yield combo
        if pair_count > 0:
            for base in _sim_combinations(full, pair_count - 1):
                for part in partial:
                    yield base + (part,)

    maximum = int(
        getattr(cfg, "SIMULATOR_MAX_MATCHING_SETS", 4000)
    )
    max_time_ms = int(getattr(cfg, "MAX_PLAN_TIME_MS", 3000))
    for combo in allowed():
        if state["matching_sets_evaluated"] >= maximum:
            state["limit_hit"] = True
            return
        if (
            max_time_ms > 0
            and ticks_diff(ticks_ms(), state["started_ms"])
            >= max_time_ms
        ):
            state["timed_out"] = True
            return
        if not _sim_valid_matching_set(
            combo, piece_count, cut_mode
        ):
            continue
        state["matching_sets_evaluated"] += 1
        if state["matching_sets_evaluated"] % 32 == 0:
            update_plan_debug(
                stage="simulator_topology",
                nodes=state["matching_sets_evaluated"],
            )
            exitpoint = getattr(os, "exitpoint", None)
            if exitpoint is not None:
                exitpoint()
            plan_debug_heartbeat()
        yield combo


def _sim_align_edge(source_a, source_b, target_a, target_b):
    source_angle = math.atan2(
        source_b[1] - source_a[1],
        source_b[0] - source_a[0],
    )
    target_angle = math.atan2(
        target_b[1] - target_a[1],
        target_b[0] - target_a[0],
    )
    angle = target_angle - source_angle
    cosine = math.cos(angle)
    sine = math.sin(angle)
    mapped_x = cosine * source_a[0] - sine * source_a[1]
    mapped_y = sine * source_a[0] + cosine * source_a[1]
    return (
        cosine,
        sine,
        target_a[0] - mapped_x,
        target_a[1] - mapped_y,
        normalize_angle_deg(math.degrees(angle)),
    )


def _sim_vertex_angle(polygon, index):
    point = polygon[index % len(polygon)]
    previous = polygon[(index - 1) % len(polygon)]
    following = polygon[(index + 1) % len(polygon)]
    left = (previous[0] - point[0], previous[1] - point[1])
    right = (following[0] - point[0], following[1] - point[1])
    left_length = math.sqrt(left[0] * left[0] + left[1] * left[1])
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


def _sim_endpoint_angle(polygon, edge_index, fraction):
    if fraction <= 1e-6:
        return _sim_vertex_angle(polygon, edge_index)
    if fraction >= 1.0 - 1e-6:
        return _sim_vertex_angle(
            polygon, edge_index + 1
        )
    return None


def _sim_endpoint_angle_error(polygons, match):
    """Return normalized local endpoint-junction angle inconsistency."""
    _, i, edge_i, j, edge_j, ia0, ia1, ja0, ja1 = match
    pairs = (
        (
            _sim_endpoint_angle(polygons[i], edge_i, ia0),
            _sim_endpoint_angle(polygons[j], edge_j, ja1),
        ),
        (
            _sim_endpoint_angle(polygons[i], edge_i, ia1),
            _sim_endpoint_angle(polygons[j], edge_j, ja0),
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


def _sim_connected_layout(polygons, matches):
    """Rigidly place the one connected component described by a prefix."""
    used = set()
    adjacency = [[] for _ in polygons]
    for match in matches:
        _, i, _, j, _ = match[:5]
        used.add(i)
        used.add(j)
        adjacency[i].append((j, match, False))
        adjacency[j].append((i, match, True))
    if not used:
        return [], []
    transforms = [None] * len(polygons)
    root = min(used)
    transforms[root] = _identity_transform()
    stack = [root]
    while stack:
        i = stack.pop()
        for j, match, reverse_sides in adjacency[i]:
            if transforms[j] is not None:
                continue
            ia, ib, ja, jb = _sim_match_segments(
                polygons, match
            )
            if reverse_sides:
                ia, ib, ja, jb = ja, jb, ia, ib
            world_a = transform_point(ia, transforms[i])
            world_b = transform_point(ib, transforms[i])
            transforms[j] = _sim_align_edge(
                ja, jb, world_b, world_a
            )
            stack.append(j)
    if any(transforms[index] is None for index in used):
        return [], []
    indices = sorted(used)
    return indices, [
        transform_polygon(polygons[index], transforms[index])
        for index in indices
    ]


def _sim_orient_layout_to_target(assembled, target):
    rectangle = minimum_area_rectangle(
        [point for polygon in assembled for point in polygon]
    )
    if rectangle is None:
        return None
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
    points = [point for polygon in oriented for point in polygon]
    width = max(point[0] for point in points) - min(
        point[0] for point in points
    )
    height = max(point[1] for point in points) - min(
        point[1] for point in points
    )
    if (target[0] >= target[1]) != (width >= height):
        quarter_turn = (0.0, 1.0, 0.0, 0.0, 90.0)
        oriented = [
            transform_polygon(polygon, quarter_turn)
            for polygon in oriented
        ]
        points = [
            point for polygon in oriented for point in polygon
        ]
    low_x = min(point[0] for point in points)
    high_x = max(point[0] for point in points)
    low_y = min(point[1] for point in points)
    high_y = max(point[1] for point in points)
    translation = (
        1.0,
        0.0,
        -0.5 * (low_x + high_x),
        -0.5 * (low_y + high_y),
        0.0,
    )
    centered = [
        transform_polygon(polygon, translation)
        for polygon in oriented
    ]
    return centered, high_x - low_x, high_y - low_y


def _sim_prefix_metrics(polygons, matches, target, total_area=None):
    """Optimistic target metrics for one connected matching prefix."""
    indices, assembled = _sim_connected_layout(
        polygons, matches
    )
    if not assembled:
        return None
    oriented = _sim_orient_layout_to_target(
        assembled, target
    )
    if oriented is None:
        return None
    centered, width, height = oriented
    target_width = float(target[0])
    target_height = float(target[1])
    target_polygon = [
        (-0.5 * target_width, -0.5 * target_height),
        (0.5 * target_width, -0.5 * target_height),
        (0.5 * target_width, 0.5 * target_height),
        (-0.5 * target_width, 0.5 * target_height),
    ]
    sum_area = 0.0
    inside_sum = 0.0
    overlap = 0.0
    for index, polygon in enumerate(centered):
        area = polygon_area(polygon)
        sum_area += area
        inside_sum += polygon_overlap_area(
            polygon, target_polygon
        )
        for earlier in range(index):
            overlap += polygon_overlap_area(
                polygon, centered[earlier]
            )
    outside = max(0.0, sum_area - inside_sum)
    target_area = target_width * target_height
    if total_area is None:
        total_area = sum(
            polygon_area(polygon) for polygon in polygons
        )
    remaining_area = max(
        0.0,
        float(total_area)
        - sum(polygon_area(polygons[index]) for index in indices),
    )
    # Optimistic because overlapping inside pixels are not subtracted. It can
    # only miss a bad prefix, never reject a geometrically recoverable one.
    gap_lower_bound = max(
        0.0,
        target_area - min(target_area, inside_sum) - remaining_area,
    )
    dimension_excess = (
        max(0.0, width - target_width)
        + max(0.0, height - target_height)
    )
    return {
        "piece_count": len(indices),
        "width": width,
        "height": height,
        "dimension_excess_mm": dimension_excess,
        "outside_mm2": outside,
        "overlap_mm2": overlap,
        "gap_lower_bound_mm2": gap_lower_bound,
    }


def _sim_candidate_score(
    polygons, match, target, total_area=None
):
    """Rank an edge match by junction angle and local target feasibility."""
    metrics = _sim_prefix_metrics(
        polygons,
        (match,),
        target,
        total_area=total_area,
    )
    if metrics is None:
        return float(match[0]) + 1000.0
    target_area = max(EPS, float(target[0]) * float(target[1]))
    target_perimeter = max(
        EPS, 2.0 * (float(target[0]) + float(target[1]))
    )
    return (
        float(match[0])
        + _sim_endpoint_angle_error(polygons, match)
        * float(
            getattr(
                cfg, "SIMULATOR_CANDIDATE_ENDPOINT_WEIGHT", 0.08
            )
        )
        + metrics["outside_mm2"]
        / target_area
        * float(
            getattr(
                cfg, "SIMULATOR_CANDIDATE_OUTSIDE_WEIGHT", 2.0
            )
        )
        + metrics["overlap_mm2"]
        / target_area
        * float(
            getattr(
                cfg, "SIMULATOR_CANDIDATE_OVERLAP_WEIGHT", 3.0
            )
        )
        + metrics["dimension_excess_mm"]
        / target_perimeter
        * float(
            getattr(
                cfg, "SIMULATOR_CANDIDATE_DIMENSION_WEIGHT", 1.0
            )
        )
        + metrics["gap_lower_bound_mm2"]
        / target_area
        * float(
            getattr(
                cfg, "SIMULATOR_CANDIDATE_GAP_WEIGHT", 1.0
            )
        )
    )


def _sim_hull_perimeter(polygons):
    hull = convex_hull(
        [point for polygon in polygons for point in polygon]
    )
    if len(hull) < 2:
        return 0.0
    return sum(
        _sim_distance(hull[index], hull[(index + 1) % len(hull)])
        for index in range(len(hull))
    )


def _sim_assembly_score(polygons, matches, transforms, closure_error, target):
    assembled = [
        transform_polygon(polygon, transform)
        for polygon, transform in zip(polygons, transforms)
    ]
    all_points = [
        point for polygon in assembled for point in polygon
    ]
    rectangle = minimum_area_rectangle(all_points)
    if rectangle is None:
        return None
    overlap = 0.0
    sum_area = 0.0
    for index, polygon in enumerate(assembled):
        sum_area += polygon_area(polygon)
        for earlier in range(index):
            overlap += polygon_overlap_area(
                polygon, assembled[earlier]
            )
    union_area = max(0.0, sum_area - overlap)
    rectangle_area = rectangle["area"]
    fill_error = max(0.0, rectangle_area - union_area)
    width = max(rectangle["width"], rectangle["height"])
    height = min(rectangle["width"], rectangle["height"])
    target_width = max(target[0], target[1])
    target_height = min(target[0], target[1])
    aspect = width / max(EPS, height)
    expected_aspect = target_width / max(EPS, target_height)
    aspect_error = abs(
        math.log(max(aspect, EPS) / max(expected_aspect, EPS))
    )
    expected_area = target_width * target_height
    perimeter_error = abs(
        _sim_hull_perimeter(assembled)
        - 2.0 * (target_width + target_height)
    )
    match_error = sum(match[0] for match in matches) * 5000.0

    # Upstream scores pixel masks at 4 px/mm.  Apply those unit factors so
    # its relative weighting is retained while all local geometry stays in mm.
    score = (
        closure_error * 4.0 * 8.0
        + overlap * 16.0 * 12.0
        + fill_error * 16.0 * 8.0
        + abs(union_area - expected_area) * 16.0 * 4.0
        + abs(rectangle_area - expected_area) * 16.0 * 3.0
        + aspect_error * 80000.0
        + perimeter_error * 4.0 * 25.0
        + match_error
    )
    return score, transforms, assembled, rectangle


def _sim_assemble(polygons, matches, target):
    adjacency = [[] for _ in polygons]
    for match in matches:
        _, i, _, j, _ = match[:5]
        adjacency[i].append((j, match, False))
        adjacency[j].append((i, match, True))
    transforms = [None] * len(polygons)
    transforms[0] = _identity_transform()
    stack = [0]
    closure_error = 0.0
    while stack:
        i = stack.pop()
        for j, match, reverse_sides in adjacency[i]:
            ia, ib, ja, jb = _sim_match_segments(polygons, match)
            if reverse_sides:
                ia, ib, ja, jb = ja, jb, ia, ib
            world_a = transform_point(ia, transforms[i])
            world_b = transform_point(ib, transforms[i])
            proposed = _sim_align_edge(ja, jb, world_b, world_a)
            if transforms[j] is None:
                transforms[j] = proposed
                stack.append(j)
            else:
                existing = transform_polygon(
                    polygons[j], transforms[j]
                )
                alternative = transform_polygon(
                    polygons[j], proposed
                )
                closure_error += sum(
                    _sim_distance(a, b)
                    for a, b in zip(existing, alternative)
                ) / max(1, len(existing))
    if any(transform is None for transform in transforms):
        return None
    return _sim_assembly_score(
        polygons, matches, transforms, closure_error, target
    )


def _sim_solve_linear(matrix, vector):
    size = len(vector)
    augmented = [
        list(matrix[row]) + [vector[row]]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        if pivot != column:
            augmented[pivot], augmented[column] = (
                augmented[column],
                augmented[pivot],
            )
        divisor = augmented[column][column]
        for item in range(column, size + 1):
            augmented[column][item] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if abs(factor) <= 1e-18:
                continue
            for item in range(column, size + 1):
                augmented[row][item] -= (
                    factor * augmented[column][item]
                )
    return [augmented[row][size] for row in range(size)]


def _sim_optimize_pose_graph(polygons, matches, initial):
    if len(polygons) < 3 or not matches:
        return initial

    values = []
    for transform in initial[1:]:
        values.extend(
            [
                math.atan2(transform[1], transform[0]),
                transform[2],
                transform[3],
            ]
        )

    def unpack(packed):
        transforms = [initial[0]]
        for index in range(len(polygons) - 1):
            angle = packed[3 * index]
            transforms.append(
                (
                    math.cos(angle),
                    math.sin(angle),
                    packed[3 * index + 1],
                    packed[3 * index + 2],
                    normalize_angle_deg(math.degrees(angle)),
                )
            )
        return transforms

    def residual(packed):
        transforms = unpack(packed)
        result = []
        for match in matches:
            _, i, _, j, _ = match[:5]
            ia, ib, ja, jb = _sim_match_segments(
                polygons, match
            )
            left_a = transform_point(ia, transforms[i])
            left_b = transform_point(ib, transforms[i])
            right_a = transform_point(jb, transforms[j])
            right_b = transform_point(ja, transforms[j])
            result.extend(
                [
                    left_a[0] - right_a[0],
                    left_a[1] - right_a[1],
                    left_b[0] - right_b[0],
                    left_b[1] - right_b[1],
                ]
            )
        return result

    steps = int(
        getattr(cfg, "SIMULATOR_POSE_OPTIMIZATION_STEPS", 20)
    )
    for _ in range(max(0, steps)):
        base = residual(values)
        columns = []
        for variable in range(len(values)):
            step = 1e-5 if variable % 3 == 0 else 1e-3
            shifted = list(values)
            shifted[variable] += step
            shifted_result = residual(shifted)
            columns.append(
                [
                    (new - old) / step
                    for new, old in zip(shifted_result, base)
                ]
            )
        normal = [
            [0.0] * len(values) for _ in range(len(values))
        ]
        right = [0.0] * len(values)
        for row in range(len(values)):
            right[row] = -sum(
                columns[row][index] * base[index]
                for index in range(len(base))
            )
            for column in range(len(values)):
                normal[row][column] = sum(
                    columns[row][index] * columns[column][index]
                    for index in range(len(base))
                )
            normal[row][row] += 1e-9
        delta = _sim_solve_linear(normal, right)
        if delta is None:
            break
        for index, value in enumerate(delta):
            values[index] += value
        if math.sqrt(sum(value * value for value in delta)) < 1e-7:
            break
    return unpack(values)


def _sim_normalize_to_target(polygons, transforms, target_size):
    assembled = [
        transform_polygon(polygon, transform)
        for polygon, transform in zip(polygons, transforms)
    ]
    rectangle = minimum_area_rectangle(
        [point for polygon in assembled for point in polygon]
    )
    if rectangle is None:
        return None
    angle = -rectangle["angle_deg"]
    radians = math.radians(angle)
    rotate = (
        math.cos(radians),
        math.sin(radians),
        0.0,
        0.0,
        normalize_angle_deg(angle),
    )
    rotated = [
        transform_polygon(polygon, rotate)
        for polygon in assembled
    ]
    points = [point for polygon in rotated for point in polygon]
    width = max(point[0] for point in points) - min(
        point[0] for point in points
    )
    height = max(point[1] for point in points) - min(
        point[1] for point in points
    )
    target_landscape = target_size[0] >= target_size[1]
    if (target_landscape and width < height) or (
        not target_landscape and width > height
    ):
        quarter_turn = (0.0, 1.0, 0.0, 0.0, 90.0)
        rotate = compose_transforms(quarter_turn, rotate)
        rotated = [
            transform_polygon(polygon, quarter_turn)
            for polygon in rotated
        ]
        points = [point for polygon in rotated for point in polygon]
    center = (
        0.5
        * (
            min(point[0] for point in points)
            + max(point[0] for point in points)
        ),
        0.5
        * (
            min(point[1] for point in points)
            + max(point[1] for point in points)
        ),
    )
    translation = (
        1.0,
        0.0,
        cfg.TARGET_CENTER_MM[0] - center[0],
        cfg.TARGET_CENTER_MM[1] - center[1],
        0.0,
    )
    normalize = compose_transforms(translation, rotate)
    return [
        compose_transforms(normalize, transform)
        for transform in transforms
    ]


def _sim_equal_rectangle_transforms(polygons, target_size):
    count = len(polygons)
    if count == 4:
        cell_width = target_size[0] * 0.5
        cell_height = target_size[1] * 0.5
        slots = (
            (0.0, 0.0),
            (cell_width, 0.0),
            (0.0, cell_height),
            (cell_width, cell_height),
        )
    else:
        cell_width = target_size[0] / count
        cell_height = target_size[1]
        slots = tuple(
            (index * cell_width, 0.0) for index in range(count)
        )
    origin = (
        cfg.TARGET_CENTER_MM[0] - target_size[0] * 0.5,
        cfg.TARGET_CENTER_MM[1] - target_size[1] * 0.5,
    )
    transforms = []
    for polygon, slot in zip(polygons, slots):
        best = None
        for edge in _sim_edges(polygon):
            vector = (
                edge[1][0] - edge[0][0],
                edge[1][1] - edge[0][1],
            )
            angle = -math.atan2(vector[1], vector[0])
            rotation = (
                math.cos(angle),
                math.sin(angle),
                0.0,
                0.0,
                normalize_angle_deg(math.degrees(angle)),
            )
            rotated = transform_polygon(polygon, rotation)
            low_x = min(point[0] for point in rotated)
            low_y = min(point[1] for point in rotated)
            size_x = max(point[0] for point in rotated) - low_x
            size_y = max(point[1] for point in rotated) - low_y
            cost = abs(size_x - cell_width) + abs(
                size_y - cell_height
            )
            if best is None or cost < best[0]:
                best = (cost, rotation, low_x, low_y)
        destination = (
            origin[0] + slot[0],
            origin[1] + slot[1],
        )
        translation = (
            1.0,
            0.0,
            destination[0] - best[2],
            destination[1] - best[3],
            0.0,
        )
        transforms.append(
            compose_transforms(translation, best[1])
        )
    return transforms


def _sim_target_geometry(target_size):
    width, height = target_size
    x0 = cfg.TARGET_CENTER_MM[0] - width * 0.5
    y0 = cfg.TARGET_CENTER_MM[1] - height * 0.5
    return (
        [
            (x0, y0),
            (x0 + width, y0),
            (x0 + width, y0 + height),
            (x0, y0 + height),
        ],
        (x0, y0, x0 + width, y0 + height),
    )


def _sim_gate_failures(
    metrics,
    dimension_error,
    limit_multiplier=1.0,
    overlap_multiplier=None,
):
    if overlap_multiplier is None:
        overlap_multiplier = limit_multiplier
    failures = []
    gates = (
        (
            "outside_mm2",
            metrics["outside_mm2"],
            float(cfg.FIXED_RECT_MAX_OUTSIDE_MM2)
            * limit_multiplier,
        ),
        (
            "overlap_mm2",
            metrics["overlap_mm2"],
            float(cfg.FIXED_RECT_MAX_OVERLAP_MM2)
            * overlap_multiplier,
        ),
        (
            "fill_gap_mm2",
            metrics["fill_gap_mm2"],
            float(cfg.FIXED_RECT_MAX_GAP_MM2)
            * limit_multiplier,
        ),
        (
            "dimension_error_mm",
            dimension_error,
            float(cfg.FINAL_RECT_DIM_TOLERANCE_MM)
            * limit_multiplier,
        ),
    )
    for name, value, limit in gates:
        if value > limit:
            failures.append(
                "{}={:.1f}>{:.1f}".format(name, value, limit)
            )
    return failures


def _sim_local_gate_failures(metrics, dimension_error):
    return _sim_gate_failures(
        metrics, dimension_error, limit_multiplier=1.0
    )


def _sim_seam_records(polygons, matches):
    records = []
    max_error = 0.0
    for match in matches:
        a0, a1, b0, b1 = _sim_match_segments(polygons, match)
        length_a = _sim_distance(a0, a1)
        length_b = _sim_distance(b0, b1)
        error = abs(length_a - length_b)
        max_error = max(max_error, error)
        records.append(
            {
                "piece_a_index": match[1],
                "edge_a_index": match[2],
                "piece_b_index": match[3],
                "edge_b_index": match[4],
                "partial": not _sim_is_full_match(match),
                "length_error_mm": error,
                "relative_length_error": (
                    error / max(EPS, length_a, length_b)
                ),
                "candidate_quality_score": match[0],
                "fractions": list(match[5:]),
            }
        )
    return records, max_error


def _sim_add_matching_stats(stats, state):
    for name in (
        "matching_prefixes_considered",
        "matching_prefixes_evaluated",
        "matching_pruned_edge_reuse",
        "matching_pruned_topology",
        "matching_pruned_duplicate",
        "matching_pruned_branch_limit",
        "matching_pruned_topology_budget",
        "matching_prefix_cache_hits",
        "matching_pruned_dimension",
        "matching_pruned_outside",
        "matching_pruned_overlap",
        "matching_pruned_gap",
    ):
        stats[name] = state.get(name, 0)
    stats["matching_topology_counts"] = dict(
        state.get("matching_topology_counts", {})
    )
    stats["matching_topology_prefix_counts"] = dict(
        state.get("matching_topology_prefix_counts", {})
    )


def _sim_new_search_state(started_ms):
    return {
        "started_ms": started_ms,
        "matching_sets_evaluated": 0,
        "limit_hit": False,
        "timed_out": False,
        "matching_prefixes_considered": 0,
        "matching_prefixes_evaluated": 0,
        "matching_pruned_edge_reuse": 0,
        "matching_pruned_topology": 0,
        "matching_pruned_duplicate": 0,
        "matching_pruned_branch_limit": 0,
        "matching_pruned_topology_budget": 0,
        "matching_pruned_dimension": 0,
        "matching_pruned_outside": 0,
        "matching_pruned_overlap": 0,
        "matching_pruned_gap": 0,
        "matching_topology_counts": {},
        "matching_topology_prefix_counts": {},
        "matching_prefix_cache_hits": 0,
        "_matching_prefix_reason_cache": {},
    }


def plan_simulator_rectangle(
    pieces,
    target_size_mm=None,
    cut_mode=None,
    validation=None,
):
    """Solve one frozen piece set through the simulator-compatible backend."""
    started_ms = ticks_ms()
    target_size = tuple(
        target_size_mm
        or cfg.TARGET_RECT_SIZE_MM
        or (100.0, 60.0)
    )
    cut_mode = cut_mode or getattr(
        cfg, "SIMULATOR_PLANNER_CUT_MODE", "auto"
    )
    validation = validation or getattr(
        cfg, "SIMULATOR_PLANNER_VALIDATION", "local"
    )
    mode = "simulator_{}_{}".format(cut_mode, validation)
    if cut_mode not in SUPPORTED_CUT_MODES:
        return PlanResult(
            reason="unsupported simulator cut mode {}".format(cut_mode),
            mode=mode,
        )
    if validation not in ("local", "upstream"):
        return PlanResult(
            reason="simulator validation must be local or upstream",
            mode=mode,
        )
    if not cfg.MIN_PIECE_COUNT <= len(pieces) <= cfg.MAX_PIECE_COUNT:
        return PlanResult(
            reason="piece count {} outside {}..{}".format(
                len(pieces),
                cfg.MIN_PIECE_COUNT,
                cfg.MAX_PIECE_COUNT,
            ),
            mode=mode,
        )

    input_piece_area_mm2 = sum(
        piece.area_mm2 for piece in pieces
    )
    normalized_pieces, target_area_scale = (
        _normalize_pieces_for_known_target(
            pieces, target_size
        )
    )
    polygons = [
        piece.polygon_mm for piece in normalized_pieces
    ]
    candidates = simulator_candidate_matchings(
        polygons, target_size_mm=target_size
    )
    full_count = sum(
        1 for candidate in candidates if _sim_is_full_match(candidate)
    )
    state = _sim_new_search_state(started_ms)
    best = None
    best_matches = ()
    if cut_mode == "equal_rectangles":
        transforms = _sim_equal_rectangle_transforms(
            polygons, target_size
        )
    else:
        update_plan_debug(stage="simulator_candidates")
        for matches in _sim_matching_sets(
            candidates,
            len(polygons),
            cut_mode,
            state,
            polygons=polygons,
            target=target_size,
        ):
            assembled = _sim_assemble(
                polygons, matches, target_size
            )
            if assembled is None:
                continue
            if best is None or assembled[0] < best[0]:
                best = assembled
                best_matches = matches
                update_plan_debug(
                    best_score=assembled[0],
                    nodes=state["matching_sets_evaluated"],
                )
        if best is None:
            elapsed_ms = max(
                0, ticks_diff(ticks_ms(), started_ms)
            )
            if state["timed_out"]:
                reason = (
                    "simulator matching search timed out after {} sets"
                ).format(state["matching_sets_evaluated"])
            elif state["limit_hit"]:
                reason = (
                    "simulator matching-set limit reached after {} sets"
                ).format(state["matching_sets_evaluated"])
            elif not candidates:
                reason = "simulator found no compatible edge candidates"
            else:
                reason = (
                    "simulator found candidates but no connected "
                    "non-reusing assembly"
                )
            stats = {
                "engine": (
                    "lvreng/puzzle-vision-simulator-compatible-k230"
                ),
                "upstream_commit": UPSTREAM_REVISION,
                "cut_mode": cut_mode,
                "validation": validation,
                "plan_ms": elapsed_ms,
                "candidate_count": len(candidates),
                "full_candidate_count": full_count,
                "partial_candidate_count": (
                    len(candidates) - full_count
                ),
                "matching_sets_evaluated": (
                    state["matching_sets_evaluated"]
                ),
                "limit_hit": state["limit_hit"],
                "timed_out": state["timed_out"],
                "input_piece_area_mm2": input_piece_area_mm2,
                "target_area_mm2": target_size[0] * target_size[1],
                "target_area_scale": target_area_scale,
                "input_area_error_pct": (
                    100.0
                    * abs(
                        input_piece_area_mm2
                        - target_size[0] * target_size[1]
                    )
                    / max(
                        EPS, target_size[0] * target_size[1]
                    )
                ),
            }
            _sim_add_matching_stats(stats, state)
            PERF_STATS.add_stage("plan_ms", elapsed_ms=elapsed_ms)
            return PlanResult(
                reason=reason,
                search_nodes=state["matching_sets_evaluated"],
                mode=mode,
                plan_stats=stats,
            )
        transforms = _sim_optimize_pose_graph(
            polygons, best_matches, best[1]
        )
        transforms = _sim_normalize_to_target(
            polygons, transforms, target_size
        )
        if transforms is None:
            return PlanResult(
                reason="simulator could not normalize assembly rectangle",
                search_nodes=state["matching_sets_evaluated"],
                mode=mode,
            )

    target_list = [
        transform_polygon(polygon, transform)
        for polygon, transform in zip(polygons, transforms)
    ]
    rectangle, target_rect = _sim_target_geometry(target_size)
    metrics = _fixed_complete_metrics(
        target_list,
        rectangle,
        target_size[0],
        target_size[1],
    )
    points = [
        point for polygon in target_list for point in polygon
    ]
    actual_width = max(point[0] for point in points) - min(
        point[0] for point in points
    )
    actual_height = max(point[1] for point in points) - min(
        point[1] for point in points
    )
    dimension_error = max(
        abs(actual_width - target_size[0]),
        abs(actual_height - target_size[1]),
    )
    gate_failures = _sim_local_gate_failures(
        metrics, dimension_error
    )
    safety_multiplier = max(
        1.0,
        float(
            getattr(
                cfg,
                "SIMULATOR_UPSTREAM_SAFETY_GATE_MULTIPLIER",
                2.0,
            )
        ),
    )
    overlap_safety_multiplier = max(
        safety_multiplier,
        float(
            getattr(
                cfg,
                "SIMULATOR_UPSTREAM_OVERLAP_SAFETY_GATE_MULTIPLIER",
                safety_multiplier,
            )
        ),
    )
    safety_gate_failures = _sim_gate_failures(
        metrics,
        dimension_error,
        limit_multiplier=safety_multiplier,
        overlap_multiplier=overlap_safety_multiplier,
    )
    valid = (
        not safety_gate_failures
        if validation == "upstream"
        else not gate_failures
    )
    reason = "ok"
    if validation == "upstream" and safety_gate_failures:
        reason = (
            "upstream proposal rejected by physical safety gates: {}"
        ).format("; ".join(safety_gate_failures))
    elif validation == "upstream" and gate_failures:
        reason = "upstream proposal; local warnings: {}".format(
            "; ".join(gate_failures)
        )
    elif gate_failures:
        reason = "simulator proposal rejected by local gates: {}".format(
            "; ".join(gate_failures)
        )

    target_polygons = {}
    operations = []
    for index, piece in enumerate(pieces):
        piece_id = piece.piece_id or "P{}".format(index + 1)
        target_polygon = target_list[index]
        target_polygons[piece_id] = target_polygon
        operations.append(
            {
                "piece_id": piece_id,
                "source_center_mm": piece.centroid_mm,
                "target_center_mm": polygon_centroid(target_polygon),
                "rotation_deg": _choose_smallest_equivalent_rotation(
                    transforms[index][4], piece.polygon_mm
                ),
                "rotation_ambiguous": piece.rotation_ambiguous,
                "confidence": piece.confidence,
            }
        )
    seams, max_error = _sim_seam_records(
        polygons, best_matches
    )
    elapsed_ms = max(0, ticks_diff(ticks_ms(), started_ms))
    stats = {
        "engine": "lvreng/puzzle-vision-simulator-compatible-k230",
        "upstream_commit": UPSTREAM_REVISION,
        "cut_mode": cut_mode,
        "validation": validation,
        "plan_ms": elapsed_ms,
        "candidate_count": len(candidates),
        "full_candidate_count": full_count,
        "partial_candidate_count": len(candidates) - full_count,
        "matching_sets_evaluated": state["matching_sets_evaluated"],
        "selected_match_count": len(best_matches),
        "selected_partial_match_count": sum(
            1 for match in best_matches if not _sim_is_full_match(match)
        ),
        "limit_hit": state["limit_hit"],
        "timed_out": state["timed_out"],
        "upstream_score": best[0] if best is not None else 0.0,
        "dimension_error_mm": dimension_error,
        "actual_width_mm": actual_width,
        "actual_height_mm": actual_height,
        "local_gate_failures": gate_failures,
        "safety_gate_failures": safety_gate_failures,
        "safety_gate_multiplier": safety_multiplier,
        "overlap_safety_gate_multiplier": (
            overlap_safety_multiplier
        ),
        "input_piece_area_mm2": input_piece_area_mm2,
        "target_area_mm2": target_size[0] * target_size[1],
        "target_area_scale": target_area_scale,
        "input_area_error_pct": (
            100.0
            * abs(
                input_piece_area_mm2
                - target_size[0] * target_size[1]
            )
            / max(EPS, target_size[0] * target_size[1])
        ),
    }
    _sim_add_matching_stats(stats, state)
    PERF_STATS.add_stage("plan_ms", elapsed_ms=elapsed_ms)
    return PlanResult(
        valid=valid,
        reason=reason,
        score=metrics["score"],
        operations=operations,
        target_polygons=target_polygons,
        target_rect=target_rect,
        search_nodes=state["matching_sets_evaluated"],
        mode=mode,
        max_vertex_error_mm=max_error,
        fill_gap_mm2=metrics["fill_gap_mm2"],
        overlap_mm2=metrics["overlap_mm2"],
        outside_mm2=metrics["outside_mm2"],
        seams=seams,
        plan_stats=stats,
    )
