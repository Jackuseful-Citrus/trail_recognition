"""Closed-loop per-piece placement monitoring for the K230 puzzle."""

import math
import os

import puzzle_config as cfg
from puzzle_geometry import (
    PieceObservation,
    polygon_centroid,
    polygon_shape_signature,
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


def _shape_cost(reference, observation):
    expected = polygon_shape_signature(
        reference.polygon_mm
    )
    actual = polygon_shape_signature(
        observation.polygon_mm
    )
    if (
        not expected
        or not actual
        or expected[0] != actual[0]
    ):
        return None
    feature_count = min(len(expected), len(actual))
    feature_loss = 0.0
    for index in range(1, feature_count):
        feature_loss += abs(
            float(expected[index]) - float(actual[index])
        )
    feature_loss /= max(1, feature_count - 1)
    area_ratio = max(
        1e-6,
        float(observation.area_mm2)
        / max(1e-6, float(reference.area_mm2)),
    )
    area_loss = min(1.5, abs(math.log(area_ratio)))
    return 0.78 * feature_loss + 0.22 * area_loss


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
        return None
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


class PlacementMonitor:
    """Freeze one plan and retire pieces as target poses are confirmed."""

    __slots__ = (
        "plan",
        "references",
        "reference_by_id",
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

    def _filter_completed_observations(self, observations):
        result = []
        for observation in observations:
            belongs_to_completed = False
            for piece_id in self.completed:
                reference = self.reference_by_id[piece_id]
                cost = _shape_cost(reference, observation)
                if (
                    cost is None
                    or cost > cfg.PLACEMENT_SHAPE_COST_LIMIT
                ):
                    continue
                target = self._target_center(piece_id)
                dx = observation.centroid_mm[0] - target[0]
                dy = observation.centroid_mm[1] - target[1]
                if (
                    math.sqrt(dx * dx + dy * dy)
                    <= cfg.PLACEMENT_CENTER_TOLERANCE_MM
                ):
                    belongs_to_completed = True
                    break
            if not belongs_to_completed:
                result.append(observation)
        return result

    def check(self, observations, coverages=None):
        self.check_index += 1
        coverages = coverages or {}
        usable = self._filter_completed_observations(
            observations
        )
        mapping, diagnostics = match_remaining_pieces(
            self.references,
            usable,
            ignored_piece_ids=self.completed,
        )
        visible = {}
        metrics = {}
        newly_completed = []

        for piece_id, observation in mapping.items():
            observation.piece_id = piece_id
            visible[piece_id] = observation

        for piece_id in self.order:
            if piece_id in self.completed:
                continue
            observation = mapping.get(piece_id)
            target_polygon = self.plan.target_polygons[piece_id]
            placed = False
            item = {
                "matched": observation is not None,
                "coverage": float(coverages.get(piece_id, 0.0)),
                "center_error_mm": None,
                "rms_vertex_error_mm": None,
                "max_vertex_error_mm": None,
                "method": "none",
            }
            if observation is not None:
                target_center = self._target_center(piece_id)
                dx = (
                    observation.centroid_mm[0]
                    - target_center[0]
                )
                dy = (
                    observation.centroid_mm[1]
                    - target_center[1]
                )
                center_error = math.sqrt(dx * dx + dy * dy)
                vertex_error = cyclic_vertex_error(
                    observation.polygon_mm,
                    target_polygon,
                )
                item["center_error_mm"] = center_error
                if vertex_error is not None:
                    item["rms_vertex_error_mm"] = (
                        vertex_error["rms_mm"]
                    )
                    item["max_vertex_error_mm"] = (
                        vertex_error["max_mm"]
                    )
                    placed = (
                        center_error
                        <= cfg.PLACEMENT_CENTER_TOLERANCE_MM
                        and vertex_error["rms_mm"]
                        <= cfg.PLACEMENT_RMS_VERTEX_TOLERANCE_MM
                        and vertex_error["max_mm"]
                        <= cfg.PLACEMENT_MAX_VERTEX_TOLERANCE_MM
                    )
                    item["method"] = "vertices"

            # Once at least one piece is confirmed, touching pieces may merge
            # into one white component. Target-region occupancy is then a
            # conservative fallback even if that merged blob received a weak
            # shape assignment whose direct vertex pose cannot pass.
            if (
                not placed
                and self.completed
                and item["coverage"]
                >= cfg.PLACEMENT_TARGET_WHITE_COVERAGE
            ):
                placed = True
                item["method"] = "coverage"

            self.hit_counts[piece_id] = (
                self.hit_counts[piece_id] + 1
                if placed
                else 0
            )
            if (
                self.hit_counts[piece_id]
                >= cfg.PLACEMENT_REQUIRED_CHECKS
            ):
                self.completed.add(piece_id)
                self.completion_order.append(piece_id)
                newly_completed.append(piece_id)
                visible.pop(piece_id, None)
            metrics[piece_id] = item

        self.visible_by_id = visible
        self.last_metrics = metrics
        self.last_observed_count = len(observations)
        self.last_match_count = diagnostics["matched"]
        return {
            "check_index": self.check_index,
            "newly_completed": newly_completed,
            "completed_count": len(self.completed),
            "total_count": len(self.order),
            "next_piece_id": self.next_piece_id(),
            "done": self.done(),
            "observed_count": len(observations),
            "matched_count": diagnostics["matched"],
            "match_nodes": diagnostics["nodes"],
            "metrics": metrics,
        }

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
