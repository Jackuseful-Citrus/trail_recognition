"""Closed-loop per-piece placement monitoring for the K230 puzzle."""

import math
import os

import puzzle_config as cfg
from puzzle_geometry import (
    PieceObservation,
    edge_lengths,
    polygon_area,
    polygon_centroid,
    polygon_shape_signature,
    polygon_symmetry_period_deg,
)


def _placement_exitpoint(counter):
    if counter % 16 != 0:
        return
    function = getattr(os, "exitpoint", None)
    if function is not None:
        function()


def clone_piece(piece):
    return PieceObservation(
        piece.piece_id,
        list(piece.contour_px),
        list(piece.polygon_mm),
        centroid_mm=piece.centroid_mm,
        area_mm2=piece.area_mm2,
        edge_lengths_mm=piece.edge_lengths_mm,
        interior_angles_deg=piece.interior_angles_deg,
        current_orientation_deg=piece.current_orientation_deg,
        confidence=piece.confidence,
        rotation_ambiguous=piece.rotation_ambiguous,
        centroid_fallback=piece.centroid_fallback,
    )


def _distance(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(len(ordered) - 1, index))]


def _polygon_perimeter(polygon):
    return sum(edge_lengths(polygon))


def resample_closed_polygon(polygon, sample_count=32):
    """Uniformly sample a closed polygon by perimeter distance."""
    count = max(3, int(sample_count))
    points = [
        (float(point[0]), float(point[1])) for point in polygon
    ]
    if len(points) < 2:
        return []
    lengths = [
        _distance(points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    ]
    perimeter = sum(lengths)
    if perimeter <= 1e-9:
        return [points[0] for _ in range(count)]
    result = []
    edge_index = 0
    edge_start_distance = 0.0
    for sample_index in range(count):
        target_distance = perimeter * sample_index / count
        while (
            edge_index < len(lengths) - 1
            and edge_start_distance + lengths[edge_index]
            < target_distance
        ):
            edge_start_distance += lengths[edge_index]
            edge_index += 1
        a = points[edge_index]
        b = points[(edge_index + 1) % len(points)]
        edge_length = max(1e-9, lengths[edge_index])
        ratio = (
            target_distance - edge_start_distance
        ) / edge_length
        result.append(
            (
                a[0] + ratio * (b[0] - a[0]),
                a[1] + ratio * (b[1] - a[1]),
            )
        )
    return result


def _point_segment_distance(point, a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return _distance(point, a)
    ratio = (
        (point[0] - a[0]) * dx
        + (point[1] - a[1]) * dy
    ) / length_squared
    ratio = max(0.0, min(1.0, ratio))
    nearest = (a[0] + ratio * dx, a[1] + ratio * dy)
    return _distance(point, nearest)


def _point_polygon_boundary_distance(point, polygon):
    return min(
        _point_segment_distance(
            point,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
        )
        for index in range(len(polygon))
    )


def placement_contour_error(
    observed_polygon,
    target_polygon,
    sample_count=None,
):
    """Compare two fixed-pose boundaries without fitting away placement error."""
    if len(observed_polygon) < 3 or len(target_polygon) < 3:
        return None
    count = (
        cfg.PLACEMENT_CONTOUR_SAMPLE_COUNT
        if sample_count is None
        else max(3, int(sample_count))
    )
    observed_samples = resample_closed_polygon(
        observed_polygon, count
    )
    target_samples = resample_closed_polygon(
        target_polygon, count
    )
    distances = [
        _point_polygon_boundary_distance(point, target_polygon)
        for point in observed_samples
    ]
    distances.extend(
        _point_polygon_boundary_distance(point, observed_polygon)
        for point in target_samples
    )
    squared = sum(value * value for value in distances)
    return {
        "rms_mm": math.sqrt(squared / len(distances)),
        "p90_mm": _percentile(distances, 0.90),
        "p95_mm": _percentile(distances, 0.95),
        "max_mm": max(distances),
        "sample_count": count,
    }


def _normalised_resampled_shape(polygon, sample_count):
    samples = resample_closed_polygon(polygon, sample_count)
    if not samples:
        return []
    center_x = sum(point[0] for point in samples) / len(samples)
    center_y = sum(point[1] for point in samples) / len(samples)
    centered = [
        (point[0] - center_x, point[1] - center_y)
        for point in samples
    ]
    radius = math.sqrt(
        sum(x * x + y * y for x, y in centered)
        / len(centered)
    )
    if radius <= 1e-9:
        return []
    return [(x / radius, y / radius) for x, y in centered]


def _rotation_invariant_shape_error(polygon_a, polygon_b):
    count = cfg.PLACEMENT_CONTOUR_SAMPLE_COUNT
    first = _normalised_resampled_shape(polygon_a, count)
    second = _normalised_resampled_shape(polygon_b, count)
    if not first or not second:
        return None
    best = None
    for reverse in (False, True):
        base = list(reversed(second)) if reverse else second
        for shift in range(count):
            ordered = base[shift:] + base[:shift]
            dot = 0.0
            cross = 0.0
            for a, b in zip(first, ordered):
                dot += a[0] * b[0] + a[1] * b[1]
                cross += a[0] * b[1] - a[1] * b[0]
            angle = math.atan2(cross, dot)
            cosine = math.cos(angle)
            sine = math.sin(angle)
            squared = 0.0
            for a, b in zip(first, ordered):
                bx = cosine * b[0] + sine * b[1]
                by = -sine * b[0] + cosine * b[1]
                dx = a[0] - bx
                dy = a[1] - by
                squared += dx * dx + dy * dy
            rms = math.sqrt(squared / count)
            if best is None or rms < best:
                best = rms
    return best


def _compactness(polygon):
    perimeter = _polygon_perimeter(polygon)
    if perimeter <= 1e-9:
        return 0.0
    return (
        4.0 * math.pi * polygon_area(polygon)
        / (perimeter * perimeter)
    )


def _shape_cost(reference, observation):
    """Rigid-invariant identity cost with no vertex-count hard gate."""
    expected = polygon_shape_signature(reference.polygon_mm)
    actual = polygon_shape_signature(observation.polygon_mm)
    if not expected or not actual:
        return None
    resampled_loss = _rotation_invariant_shape_error(
        reference.polygon_mm, observation.polygon_mm
    )
    if resampled_loss is None:
        return None
    if expected[0] == actual[0]:
        feature_count = min(len(expected), len(actual))
        signature_loss = sum(
            abs(float(expected[index]) - float(actual[index]))
            for index in range(1, feature_count)
        ) / max(1, feature_count - 1)
    else:
        signature_loss = abs(
            _compactness(reference.polygon_mm)
            - _compactness(observation.polygon_mm)
        )
    area_ratio = max(
        1e-6,
        float(observation.area_mm2)
        / max(1e-6, float(reference.area_mm2)),
    )
    perimeter_ratio = max(
        1e-6,
        _polygon_perimeter(observation.polygon_mm)
        / max(1e-6, _polygon_perimeter(reference.polygon_mm)),
    )
    area_loss = min(1.5, abs(math.log(area_ratio)))
    perimeter_loss = min(1.5, abs(math.log(perimeter_ratio)))
    return (
        0.48 * resampled_loss
        + 0.22 * signature_loss
        + 0.20 * area_loss
        + 0.10 * perimeter_loss
    )


def match_remaining_pieces(
    references,
    observations,
    ignored_piece_ids=None,
):
    """Maximise shape-compatible assignments, then minimise their cost."""
    ignored = ignored_piece_ids or set()
    active_references = [
        piece
        for piece in references
        if piece.piece_id not in ignored
    ]
    pair_costs = []
    for reference in active_references:
        row = []
        for observation in observations:
            row.append(_shape_cost(reference, observation))
        pair_costs.append(row)

    best = {
        "count": -1,
        "cost": 1e9,
        "mapping": {},
        "nodes": 0,
    }
    used = set()
    mapping = {}

    def recurse(reference_index, count, total_cost):
        best["nodes"] += 1
        _placement_exitpoint(best["nodes"])
        if reference_index == len(active_references):
            if (
                count > best["count"]
                or (
                    count == best["count"]
                    and total_cost < best["cost"]
                )
            ):
                best["count"] = count
                best["cost"] = total_cost
                best["mapping"] = dict(mapping)
            return

        reference = active_references[reference_index]
        recurse(reference_index + 1, count, total_cost)
        for observation_index, observation in enumerate(
            observations
        ):
            if observation_index in used:
                continue
            cost = pair_costs[reference_index][
                observation_index
            ]
            if (
                cost is None
                or cost > cfg.PLACEMENT_SHAPE_COST_LIMIT
            ):
                continue
            used.add(observation_index)
            mapping[reference.piece_id] = observation
            recurse(
                reference_index + 1,
                count + 1,
                total_cost + cost,
            )
            del mapping[reference.piece_id]
            used.remove(observation_index)

    recurse(0, 0, 0.0)
    return best["mapping"], {
        "matched": max(0, best["count"]),
        "cost": best["cost"] if best["count"] > 0 else None,
        "nodes": best["nodes"],
    }


def cyclic_vertex_error(observed_polygon, target_polygon):
    """Compare actual and target vertices without fitting away pose error."""
    if len(observed_polygon) != len(target_polygon):
        contour_error = placement_contour_error(
            observed_polygon, target_polygon
        )
        if contour_error is None:
            return None
        return {
            "rms_mm": contour_error["rms_mm"],
            "max_mm": contour_error["max_mm"],
            "p90_mm": contour_error["p90_mm"],
            "p95_mm": contour_error["p95_mm"],
            "resampled": True,
        }
    count = len(observed_polygon)
    best = None
    for reversed_order in (False, True):
        base = (
            list(reversed(target_polygon))
            if reversed_order
            else list(target_polygon)
        )
        for shift in range(count):
            ordered = base[shift:] + base[:shift]
            squared = 0.0
            maximum = 0.0
            for observed, target in zip(
                observed_polygon, ordered
            ):
                dx = observed[0] - target[0]
                dy = observed[1] - target[1]
                distance = math.sqrt(dx * dx + dy * dy)
                squared += distance * distance
                maximum = max(maximum, distance)
            rms = math.sqrt(squared / count)
            if best is None or rms < best["rms_mm"]:
                best = {
                    "rms_mm": rms,
                    "max_mm": maximum,
                }
    return best


def _angle_difference_deg(a, b, period=360.0):
    period = max(1e-6, float(period))
    delta = (float(a) - float(b)) % period
    return min(delta, period - delta)


def _longest_edge_orientation_reliable(polygon):
    lengths = edge_lengths(polygon)
    if len(lengths) < 3:
        return False
    maximum = max(lengths)
    if maximum <= 1e-9:
        return False
    dominant = []
    remaining = []
    for index, length in enumerate(lengths):
        a = polygon[index]
        b = polygon[(index + 1) % len(polygon)]
        angle = math.degrees(
            math.atan2(b[1] - a[1], b[0] - a[0])
        ) % 180.0
        if (
            length
            >= maximum
            / cfg.PLACEMENT_ORIENTATION_LONGEST_EDGE_RATIO_MIN
        ):
            dominant.append(angle)
        else:
            remaining.append(length)
    base = dominant[0]
    if any(
        _angle_difference_deg(angle, base, period=180.0)
        > cfg.PLACEMENT_ORIENTATION_SAMPLE_SPREAD_MAX_DEG
        for angle in dominant[1:]
    ):
        return False
    if not remaining:
        return False
    return (
        maximum / max(1e-9, max(remaining))
        >= cfg.PLACEMENT_ORIENTATION_LONGEST_EDGE_RATIO_MIN
    )


def placement_pose_error_bound(reference, observation, operation):
    """Return the conservative rigid-pose bound when direction is reliable."""
    target_center = operation.get("target_center_mm")
    if target_center is None:
        return {
            "center_error_mm": None,
            "angle_error_deg": None,
            "radius_mm": None,
            "pose_error_bound_mm": None,
            "orientation_reliable": False,
        }
    center_error = _distance(
        observation.centroid_mm, target_center
    )
    radius = max(
        _distance(point, reference.centroid_mm)
        for point in reference.polygon_mm
    )
    period = polygon_symmetry_period_deg(reference.polygon_mm)
    target_orientation = (
        reference.current_orientation_deg
        + float(operation.get("rotation_deg", 0.0))
    )
    angle_error = _angle_difference_deg(
        observation.current_orientation_deg,
        target_orientation,
        period=period,
    )
    reliable = (
        not reference.rotation_ambiguous
        and not observation.rotation_ambiguous
        and _longest_edge_orientation_reliable(reference.polygon_mm)
        and _longest_edge_orientation_reliable(observation.polygon_mm)
    )
    bound = None
    if reliable:
        bound = center_error + 2.0 * radius * math.sin(
            math.radians(angle_error) / 2.0
        )
    return {
        "center_error_mm": center_error,
        "angle_error_deg": angle_error,
        "radius_mm": radius,
        "pose_error_bound_mm": bound,
        "orientation_reliable": reliable,
    }


def foreground_mask_from_gray(gray_array, threshold):
    """Copy a 2-D grayscale array into a compact stable binary snapshot."""
    height = int(gray_array.shape[0])
    width = int(gray_array.shape[1])
    result = bytearray(width * height)
    index = 0
    for y in range(height):
        row = gray_array[y]
        for x in range(width):
            result[index] = (
                1 if int(row[x]) >= int(threshold) else 0
            )
            index += 1
    return result


def _point_in_polygon(point, polygon):
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > point[1]) != (previous[1] > point[1]):
            denominator = previous[1] - current[1]
            if abs(denominator) <= 1e-12:
                denominator = 1e-12
            crossing_x = current[0] + (
                (point[1] - current[1])
                * (previous[0] - current[0])
                / denominator
            )
            if point[0] < crossing_x:
                inside = not inside
        previous = current
    return inside


def placement_delta_metrics(
    before_mask,
    after_mask,
    width,
    height,
    target_polygon,
    source_polygon,
    reference_area_mm2,
    envelope_mm=None,
):
    """Measure added target occupancy and source removal for one move."""
    width = int(width)
    height = int(height)
    expected_size = width * height
    if (
        width <= 1
        or height <= 1
        or len(before_mask) != expected_size
        or len(after_mask) != expected_size
    ):
        raise ValueError("placement delta mask dimensions differ")
    envelope = (
        cfg.PLACEMENT_DELTA_ENVELOPE_MM
        if envelope_mm is None
        else max(0.0, float(envelope_mm))
    )
    pixel_area = (
        cfg.A4_WIDTH_MM
        * cfg.A4_HEIGHT_MM
        / float(width * height)
    )
    target_samples = 0
    added_inside_target = 0
    added_total = 0
    added_outside_envelope = 0
    removed_inside_source = 0
    for y in range(height):
        point_y = (y + 0.5) * cfg.A4_HEIGHT_MM / height
        for x in range(width):
            point = (
                (x + 0.5) * cfg.A4_WIDTH_MM / width,
                point_y,
            )
            index = y * width + x
            in_target = _point_in_polygon(point, target_polygon)
            if in_target:
                target_samples += 1
            added = bool(after_mask[index]) and not bool(
                before_mask[index]
            )
            removed = bool(before_mask[index]) and not bool(
                after_mask[index]
            )
            if added:
                added_total += 1
                if in_target:
                    added_inside_target += 1
                elif (
                    _point_polygon_boundary_distance(
                        point, target_polygon
                    )
                    > envelope
                ):
                    added_outside_envelope += 1
            if (
                removed
                and source_polygon
                and _point_in_polygon(point, source_polygon)
            ):
                removed_inside_source += 1
    reference_area = max(1e-9, float(reference_area_mm2))
    return {
        "added_target_coverage": (
            float(added_inside_target) / max(1, target_samples)
        ),
        "added_area_ratio": (
            added_total * pixel_area / reference_area
        ),
        "added_spill_ratio": (
            float(added_outside_envelope) / max(1, added_total)
        ),
        "removed_source_ratio": (
            removed_inside_source * pixel_area / reference_area
        ),
        "added_pixel_count": added_total,
        "removed_source_pixel_count": removed_inside_source,
        "target_sample_count": target_samples,
    }


def _rect_sample_metrics(
    mask,
    width,
    height,
    rect,
    lower_foreground,
):
    x0, y0, x1, y1 = [float(value) for value in rect]
    inside_samples = 0
    inside_foreground = 0
    outside_envelope = 0
    envelope_x0 = x0 - cfg.FINAL_RECT_ENVELOPE_MM
    envelope_y0 = y0 - cfg.FINAL_RECT_ENVELOPE_MM
    envelope_x1 = x1 + cfg.FINAL_RECT_ENVELOPE_MM
    envelope_y1 = y1 + cfg.FINAL_RECT_ENVELOPE_MM
    for y in range(height):
        point_y = (y + 0.5) * cfg.A4_HEIGHT_MM / height
        for x in range(width):
            point_x = (x + 0.5) * cfg.A4_WIDTH_MM / width
            inside = (
                x0 <= point_x <= x1
                and y0 <= point_y <= y1
            )
            if inside:
                inside_samples += 1
                if mask[y * width + x]:
                    inside_foreground += 1
            if (
                mask[y * width + x]
                and point_y >= cfg.DIVIDER_Y_MM
                and not (
                    envelope_x0 <= point_x <= envelope_x1
                    and envelope_y0 <= point_y <= envelope_y1
                )
            ):
                outside_envelope += 1
    return {
        "rect": (x0, y0, x1, y1),
        "fill_ratio": (
            float(inside_foreground) / max(1, inside_samples)
        ),
        "spill_ratio": (
            float(outside_envelope)
            / max(1, lower_foreground)
        ),
    }


def final_rectangle_metrics(
    foreground_mask,
    width,
    height,
    target_rect,
    initial_total_piece_area,
):
    """Evaluate a whole final scene independently of piece vertex counts."""
    width = int(width)
    height = int(height)
    if len(foreground_mask) != width * height:
        raise ValueError("final scene mask dimensions differ")
    lower_foreground = 0
    min_x = width
    max_x = -1
    min_y = height
    max_y = -1
    for y in range(height):
        point_y = (y + 0.5) * cfg.A4_HEIGHT_MM / height
        if point_y < cfg.DIVIDER_Y_MM:
            continue
        for x in range(width):
            if foreground_mask[y * width + x]:
                lower_foreground += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
    pixel_area = (
        cfg.A4_WIDTH_MM
        * cfg.A4_HEIGHT_MM
        / float(width * height)
    )
    final_area = lower_foreground * pixel_area
    final_area_ratio = final_area / max(
        1e-9, float(initial_total_piece_area)
    )
    if lower_foreground:
        detected_width = (
            max_x - min_x + 1
        ) * cfg.A4_WIDTH_MM / width
        detected_height = (
            max_y - min_y + 1
        ) * cfg.A4_HEIGHT_MM / height
    else:
        detected_width = 0.0
        detected_height = 0.0

    x0, y0, x1, y1 = [float(value) for value in target_rect]
    target_width = x1 - x0
    target_height = y1 - y0
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    rect_candidates = [(x0, y0, x1, y1)]
    if abs(target_width - target_height) > 1e-9:
        rect_candidates.append(
            (
                center_x - 0.5 * target_height,
                center_y - 0.5 * target_width,
                center_x + 0.5 * target_height,
                center_y + 0.5 * target_width,
            )
        )
    sampled = [
        _rect_sample_metrics(
            foreground_mask,
            width,
            height,
            rect,
            lower_foreground,
        )
        for rect in rect_candidates
    ]
    best_rect = max(
        sampled,
        key=lambda item: (
            item["fill_ratio"],
            -item["spill_ratio"],
        ),
    )
    direct_errors = (
        abs(detected_width - target_width),
        abs(detected_height - target_height),
    )
    swapped_errors = (
        abs(detected_width - target_height),
        abs(detected_height - target_width),
    )
    if sum(swapped_errors) < sum(direct_errors):
        width_error, height_error = swapped_errors
        dimensions_swapped = True
    else:
        width_error, height_error = direct_errors
        dimensions_swapped = False
    valid = (
        best_rect["fill_ratio"] >= cfg.FINAL_RECT_FILL_MIN
        and cfg.FINAL_AREA_RATIO_MIN
        <= final_area_ratio
        <= cfg.FINAL_AREA_RATIO_MAX
        and width_error <= cfg.FINAL_RECT_DIM_TOLERANCE_MM
        and height_error <= cfg.FINAL_RECT_DIM_TOLERANCE_MM
        and best_rect["spill_ratio"] <= cfg.FINAL_RECT_SPILL_MAX
    )
    return {
        "fill_ratio": best_rect["fill_ratio"],
        "final_area_ratio": final_area_ratio,
        "detected_width_mm": detected_width,
        "detected_height_mm": detected_height,
        "width_error_mm": width_error,
        "height_error_mm": height_error,
        "dimensions_swapped": dimensions_swapped,
        "spill_ratio": best_rect["spill_ratio"],
        "selected_target_rect": best_rect["rect"],
        "lower_foreground_area_mm2": final_area,
        "lower_foreground_count": lower_foreground,
        "valid": valid,
    }


def final_rectangle_consensus(scene_metrics):
    """Require the configured number of valid samples in one stable burst."""
    samples = list(scene_metrics)
    required = cfg.FINAL_VERIFY_REQUIRED_PASSES
    pass_count = sum(
        1 for metrics in samples if metrics.get("valid", False)
    )
    result = {
        "valid": (
            len(samples) >= cfg.FINAL_VERIFY_SAMPLE_COUNT
            and pass_count >= required
        ),
        "sample_count": len(samples),
        "pass_count": pass_count,
        "required_passes": required,
        "samples": samples,
    }
    for name in (
        "fill_ratio",
        "final_area_ratio",
        "detected_width_mm",
        "detected_height_mm",
        "width_error_mm",
        "height_error_mm",
        "spill_ratio",
    ):
        values = sorted(
            float(metrics[name])
            for metrics in samples
            if metrics.get(name) is not None
        )
        result[name] = (
            values[len(values) // 2] if values else None
        )
    return result


class PlacementMonitor:
    """Freeze one plan and verify only its next operation after each move."""

    __slots__ = (
        "plan",
        "references",
        "reference_by_id",
        "operation_by_id",
        "order",
        "completed",
        "completion_order",
        "hit_counts",
        "visible_by_id",
        "check_index",
        "last_metrics",
        "last_observed_count",
        "last_match_count",
    )

    def __init__(self, pieces, plan):
        self.plan = plan
        self.references = [clone_piece(piece) for piece in pieces]
        self.reference_by_id = {
            piece.piece_id: piece for piece in self.references
        }
        self.operation_by_id = {
            operation["piece_id"]: dict(operation)
            for operation in plan.operations
        }
        self.order = [
            operation["piece_id"]
            for operation in plan.operations
        ]
        self.completed = set()
        self.completion_order = []
        self.hit_counts = {
            piece_id: 0 for piece_id in self.order
        }
        self.visible_by_id = {
            piece.piece_id: clone_piece(piece)
            for piece in self.references
        }
        self.check_index = 0
        self.last_metrics = {}
        self.last_observed_count = len(pieces)
        self.last_match_count = len(pieces)

    def _target_center(self, piece_id):
        for operation in self.plan.operations:
            if operation["piece_id"] == piece_id:
                return operation["target_center_mm"]
        return polygon_centroid(
            self.plan.target_polygons[piece_id]
        )

    def _best_next_observation(self, piece_id, observations):
        reference = self.reference_by_id[piece_id]
        target_center = self._target_center(piece_id)
        best = None
        for observation in observations:
            shape_cost = _shape_cost(reference, observation)
            if (
                shape_cost is None
                or shape_cost > cfg.PLACEMENT_SHAPE_COST_LIMIT
            ):
                continue
            center_error = _distance(
                observation.centroid_mm, target_center
            )
            candidate = (
                shape_cost
                + 0.15
                * min(
                    2.0,
                    center_error
                    / cfg.PLACEMENT_CENTER_COARSE_MAX_MM,
                ),
                observation,
                shape_cost,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            return None, None
        return best[1], best[2]

    def verify_next_piece(
        self,
        observations,
        delta_metrics=None,
    ):
        """Evaluate one stable sample without mutating completion state."""
        piece_id = self.next_piece_id()
        if piece_id is None:
            return {
                "piece_id": None,
                "placed": False,
                "reason": "all_pieces_already_confirmed",
                "method": "none",
                "matched_observation": None,
            }
        reference = self.reference_by_id[piece_id]
        target_polygon = self.plan.target_polygons[piece_id]
        operation = self.operation_by_id[piece_id]
        observation, shape_cost = self._best_next_observation(
            piece_id, observations
        )
        item = {
            "piece_id": piece_id,
            "matched": observation is not None,
            "matched_observation": observation,
            "shape_cost": shape_cost,
            "center_error_mm": None,
            "angle_error_deg": None,
            "pose_error_bound_mm": None,
            "orientation_reliable": False,
            "area_ratio": None,
            "contour_rms_mm": None,
            "contour_p90_mm": None,
            "contour_p95_mm": None,
            "target_coverage": None,
            "added_area_ratio": None,
            "spill_ratio": None,
            "removed_source_ratio": None,
            "source_removal_support": False,
            "placed_by_resampled_contour": False,
            "placed_by_pose_bound": False,
            "placed_by_delta_coverage": False,
            "placed": False,
            "method": "none",
            "reason": "no_matching_independent_contour",
        }
        if observation is not None:
            pose = placement_pose_error_bound(
                reference, observation, operation
            )
            contour = placement_contour_error(
                observation.polygon_mm, target_polygon
            )
            area_ratio = (
                observation.area_mm2
                / max(1e-9, reference.area_mm2)
            )
            item.update(pose)
            item["area_ratio"] = area_ratio
            if contour is not None:
                item["contour_rms_mm"] = contour["rms_mm"]
                item["contour_p90_mm"] = contour["p90_mm"]
                item["contour_p95_mm"] = contour["p95_mm"]
                item["placed_by_resampled_contour"] = (
                    pose["center_error_mm"]
                    <= cfg.PLACEMENT_CENTER_COARSE_MAX_MM
                    and cfg.PLACEMENT_AREA_RATIO_MIN
                    <= area_ratio
                    <= cfg.PLACEMENT_AREA_RATIO_MAX
                    and contour["rms_mm"]
                    <= cfg.PLACEMENT_CONTOUR_RMS_MAX_MM
                    and contour["p90_mm"]
                    <= cfg.PLACEMENT_CONTOUR_P90_MAX_MM
                    and contour["p95_mm"]
                    <= cfg.PLACEMENT_CONTOUR_P95_HARD_MAX_MM
                )
                item["placed_by_pose_bound"] = (
                    pose["pose_error_bound_mm"] is not None
                    and pose["pose_error_bound_mm"]
                    <= cfg.PLACEMENT_POSE_BOUND_MM
                )
                if item["placed_by_resampled_contour"]:
                    item["reason"] = "resampled_contour_pass"

        delta = delta_metrics or {}
        target_coverage = delta.get("added_target_coverage")
        added_area_ratio = delta.get("added_area_ratio")
        spill_ratio = delta.get("added_spill_ratio")
        removed_source_ratio = delta.get("removed_source_ratio")
        item["target_coverage"] = target_coverage
        item["added_area_ratio"] = added_area_ratio
        item["spill_ratio"] = spill_ratio
        item["removed_source_ratio"] = removed_source_ratio
        item["source_removal_support"] = (
            removed_source_ratio is not None
            and removed_source_ratio
            >= cfg.PLACEMENT_SOURCE_REMOVAL_MIN
        )
        if (
            target_coverage is not None
            and added_area_ratio is not None
            and spill_ratio is not None
        ):
            item["placed_by_delta_coverage"] = (
                target_coverage
                >= cfg.PLACEMENT_DELTA_TARGET_COVERAGE_MIN
                and cfg.PLACEMENT_DELTA_AREA_RATIO_MIN
                <= added_area_ratio
                <= cfg.PLACEMENT_DELTA_AREA_RATIO_MAX
                and spill_ratio <= cfg.PLACEMENT_DELTA_SPILL_MAX
            )
            if item["placed_by_delta_coverage"]:
                item["reason"] = "delta_coverage_pass"
            elif item["reason"] == "no_matching_independent_contour":
                item["reason"] = "delta_coverage_failed"
        item["placed"] = (
            item["placed_by_resampled_contour"]
            or item["placed_by_delta_coverage"]
        )
        if item["placed_by_resampled_contour"]:
            item["method"] = "resampled_contour"
        elif item["placed_by_delta_coverage"]:
            item["method"] = "delta_coverage"
        elif observation is not None:
            item["reason"] = "contour_and_delta_failed"
        return item

    @staticmethod
    def _median_metric(sample_results, name):
        values = [
            result[name]
            for result in sample_results
            if result.get(name) is not None
        ]
        if not values:
            return None
        values = sorted(float(value) for value in values)
        return values[len(values) // 2]

    def _commit_result(self, result, observed_count):
        piece_id = result["piece_id"]
        newly_completed = []
        if piece_id is None:
            self.last_observed_count = int(observed_count)
            self.last_match_count = 0
            return {
                "check_index": self.check_index,
                "newly_completed": [],
                "completed_count": len(self.completed),
                "total_count": len(self.order),
                "next_piece_id": None,
                "done": self.done(),
                "observed_count": int(observed_count),
                "matched_count": 0,
                "match_nodes": 0,
                "metrics": {},
            }
        placed = bool(result["placed"])
        self.hit_counts[piece_id] = (
            self.hit_counts[piece_id] + 1 if placed else 0
        )
        observation = result.get("matched_observation")
        if observation is not None:
            observation.piece_id = piece_id
            self.visible_by_id[piece_id] = observation
        if (
            placed
            and self.hit_counts[piece_id]
            >= cfg.PLACEMENT_REQUIRED_CHECKS
        ):
            self.completed.add(piece_id)
            self.completion_order.append(piece_id)
            newly_completed.append(piece_id)
            self.visible_by_id.pop(piece_id, None)
        public_metrics = {
            key: value
            for key, value in result.items()
            if key != "matched_observation"
        }
        self.last_metrics = {piece_id: public_metrics}
        self.last_observed_count = int(observed_count)
        self.last_match_count = int(result.get("matched", False))
        return {
            "check_index": self.check_index,
            "newly_completed": newly_completed,
            "completed_count": len(self.completed),
            "total_count": len(self.order),
            "next_piece_id": self.next_piece_id(),
            "done": self.done(),
            "observed_count": int(observed_count),
            "matched_count": self.last_match_count,
            "match_nodes": 0,
            "metrics": {piece_id: public_metrics},
        }

    def check(
        self,
        observations,
        coverages=None,
        delta_metrics=None,
    ):
        """Compatibility single-sample entrypoint; no global rematching."""
        self.check_index += 1
        if delta_metrics is None and coverages:
            piece_id = self.next_piece_id()
            supplied = coverages.get(piece_id)
            if isinstance(supplied, dict):
                delta_metrics = supplied
        result = self.verify_next_piece(
            observations, delta_metrics=delta_metrics
        )
        return self._commit_result(result, len(observations))

    def check_samples(self, samples):
        """Use medians and a required pass count across one stable burst."""
        self.check_index += 1
        sample_results = []
        observed_count = 0
        for sample in samples:
            observations = sample.get("observations", ())
            observed_count = max(observed_count, len(observations))
            sample_results.append(
                self.verify_next_piece(
                    observations,
                    delta_metrics=sample.get("delta_metrics"),
                )
            )
        if not sample_results:
            sample_results.append(
                self.verify_next_piece((), delta_metrics=None)
            )
        required = cfg.PLACEMENT_VERIFY_REQUIRED_PASSES
        pass_count = sum(
            1 for result in sample_results if result["placed"]
        )
        representative = sorted(
            sample_results,
            key=lambda result: (
                not result["placed"],
                result.get("contour_p90_mm")
                if result.get("contour_p90_mm") is not None
                else 1e9,
            ),
        )[0]
        aggregate = dict(representative)
        for name in (
            "center_error_mm",
            "angle_error_deg",
            "pose_error_bound_mm",
            "area_ratio",
            "contour_rms_mm",
            "contour_p90_mm",
            "contour_p95_mm",
            "target_coverage",
            "added_area_ratio",
            "spill_ratio",
            "removed_source_ratio",
        ):
            aggregate[name] = self._median_metric(
                sample_results, name
            )
        aggregate["sample_count"] = len(sample_results)
        aggregate["sample_pass_count"] = pass_count
        aggregate["required_passes"] = required
        aggregate["placed"] = (
            len(sample_results) >= required
            and pass_count >= required
        )
        if not aggregate["placed"]:
            aggregate["method"] = "none"
            aggregate["reason"] = "multi_sample_consensus_failed"
        result = self._commit_result(aggregate, observed_count)
        result["sample_results"] = [
            {
                key: value
                for key, value in sample.items()
                if key != "matched_observation"
            }
            for sample in sample_results
        ]
        return result

    def next_piece_id(self):
        for piece_id in self.order:
            if piece_id not in self.completed:
                return piece_id
        return None

    def done(self):
        return len(self.completed) == len(self.order)

    def visible_pieces(self):
        return [
            self.visible_by_id[piece_id]
            for piece_id in self.order
            if piece_id in self.visible_by_id
            and piece_id not in self.completed
        ]

    def state(self):
        return {
            "check_index": self.check_index,
            "completed": set(self.completed),
            "completion_order": list(self.completion_order),
            "completed_count": len(self.completed),
            "total_count": len(self.order),
            "next_piece_id": self.next_piece_id(),
            "done": self.done(),
            "visible_pieces": self.visible_pieces(),
            "metrics": dict(self.last_metrics),
            "observed_count": self.last_observed_count,
            "matched_count": self.last_match_count,
        }
