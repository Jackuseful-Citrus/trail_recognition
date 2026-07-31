"""K230-compatible edge matching and pose optimization primitives.

The matching and pose-graph behaviour follows
https://github.com/lvreng/puzzle-vision-simulator at revision
e9eb2e0fb945c348eedd0b0fa9258f5518d2892f.  The upstream implementation uses
NumPy and OpenCV and therefore cannot run on CanMV MicroPython.  This module
retains only the stages used by the active free-rectangle planner:

* tolerant full-edge candidates;
* long-edge/short-edge T-junction candidates;
* rigid edge alignment;
* global pose-graph refinement.

The referenced repository did not contain a root license file at the pinned
revision.  This is consequently an independently written compatibility port,
not a vendored copy of the upstream source.
"""

import math
import puzzle_config as cfg
from puzzle_geometry import (
    EPS,
    normalize_angle_deg,
    transform_point,
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


def simulator_candidate_matchings(pieces):
    """Return the upstream-compatible full and partial edge shortlist."""
    polygons = [
        piece.polygon_mm if hasattr(piece, "polygon_mm") else piece
        for piece in pieces
    ]
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
            if partial_min <= ratio <= partial_max:
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
    return candidates[
        : int(getattr(cfg, "SIMULATOR_MAX_CANDIDATES", 80))
    ]


def _sim_is_full_match(match):
    return match[5:] == (0.0, 1.0, 0.0, 1.0)


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
