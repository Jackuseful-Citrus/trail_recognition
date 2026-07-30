"""Adapt the upstream NumPy/OpenCV solver to the local ``PlanResult`` API."""

from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np

import puzzle_config as cfg
from puzzle_geometry import (
    PlanResult,
    normalize_angle_deg,
    plan_outer_first_rectangle,
    polygon_area,
    polygon_centroid,
    polygon_overlap_area,
)

from .upstream_loader import PINNED_COMMIT, load_upstream


# Upstream renders 10 cm as 400 pixels, hence 4 px/mm.
UPSTREAM_PIXELS_PER_MM = 4.0
SUPPORTED_CUT_MODES = {
    "auto",
    "common",
    "boundary_fan",
    "strips",
    "equal_rectangles",
    "t_junction",
    "corner",
    "concave",
    "sequential",
}


def _as_numpy_polygons(pieces: Iterable[object]) -> list[np.ndarray]:
    polygons = []
    for piece in pieces:
        polygon = np.asarray(piece.polygon_mm, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError("piece polygon must have shape (N, 2)")
        polygons.append(polygon * UPSTREAM_PIXELS_PER_MM)
    return polygons


def _transform_to_local_mm(
    upstream_transform: np.ndarray,
    target_shift_mm: np.ndarray,
) -> np.ndarray:
    """Convert ``H(px input -> px target)`` to ``H(mm -> local A4 mm)``."""
    result = np.eye(3, dtype=float)
    result[:2, :2] = upstream_transform[:2, :2]
    result[:2, 2] = (
        upstream_transform[:2, 2] / UPSTREAM_PIXELS_PER_MM
        + target_shift_mm
    )
    return result


def _apply_h(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points), dtype=float)]
    return (transform @ homogeneous.T).T[:, :2]


def _target_rectangle() -> tuple[list[tuple[float, float]], tuple[float, ...]]:
    width, height = cfg.TARGET_RECT_SIZE_MM or (100.0, 60.0)
    center_x, center_y = cfg.TARGET_CENTER_MM
    x0 = center_x - width * 0.5
    x1 = center_x + width * 0.5
    y0 = center_y - height * 0.5
    y1 = center_y + height * 0.5
    return (
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        (x0, y0, x1, y1),
    )


def _proposal_metrics(
    target_polygons: list[list[tuple[float, float]]],
) -> dict[str, float]:
    rectangle, target_rect = _target_rectangle()
    target_area = (
        (target_rect[2] - target_rect[0])
        * (target_rect[3] - target_rect[1])
    )
    areas = [polygon_area(polygon) for polygon in target_polygons]
    overlap = 0.0
    for left in range(len(target_polygons)):
        for right in range(left + 1, len(target_polygons)):
            overlap += polygon_overlap_area(
                target_polygons[left],
                target_polygons[right],
            )
    outside = sum(
        max(
            0.0,
            area - polygon_overlap_area(polygon, rectangle),
        )
        for area, polygon in zip(areas, target_polygons)
    )
    covered_inside = max(0.0, sum(areas) - outside - overlap)
    fill_gap = max(0.0, target_area - covered_inside)

    points = [
        point
        for polygon in target_polygons
        for point in polygon
    ]
    actual_width = max(point[0] for point in points) - min(
        point[0] for point in points
    )
    actual_height = max(point[1] for point in points) - min(
        point[1] for point in points
    )
    expected_width = target_rect[2] - target_rect[0]
    expected_height = target_rect[3] - target_rect[1]
    dimension_error = max(
        abs(actual_width - expected_width),
        abs(actual_height - expected_height),
    )
    score = (
        fill_gap / max(1.0, target_area)
        + overlap / max(1.0, target_area)
        + outside / max(1.0, target_area)
        + dimension_error / max(1.0, expected_width + expected_height)
    )
    return {
        "score": score,
        "fill_gap_mm2": fill_gap,
        "overlap_mm2": overlap,
        "outside_mm2": outside,
        "dimension_error_mm": dimension_error,
        "actual_width_mm": actual_width,
        "actual_height_mm": actual_height,
    }


def _local_gate_failures(metrics: dict[str, float]) -> list[str]:
    failures = []
    gates = (
        (
            "outside_mm2",
            cfg.FIXED_RECT_MAX_OUTSIDE_MM2,
        ),
        (
            "overlap_mm2",
            cfg.FIXED_RECT_MAX_OVERLAP_MM2,
        ),
        (
            "fill_gap_mm2",
            cfg.FIXED_RECT_MAX_GAP_MM2,
        ),
    )
    for name, limit in gates:
        if metrics[name] > limit:
            failures.append(
                "{}={:.1f}>{:.1f}".format(
                    name,
                    metrics[name],
                    limit,
                )
            )
    if metrics["dimension_error_mm"] > cfg.FINAL_RECT_DIM_TOLERANCE_MM:
        failures.append(
            "dimension_error_mm={:.1f}>{:.1f}".format(
                metrics["dimension_error_mm"],
                cfg.FINAL_RECT_DIM_TOLERANCE_MM,
            )
        )
    return failures


def _seam_records(upstream, polygons_px, matches) -> tuple[list[dict], float]:
    records = []
    max_error_mm = 0.0
    for match in matches:
        a0, a1, b0, b1 = upstream.match_segments(polygons_px, match)
        length_a = float(np.linalg.norm(a1 - a0))
        length_b = float(np.linalg.norm(b1 - b0))
        length_error_mm = abs(length_a - length_b) / UPSTREAM_PIXELS_PER_MM
        max_error_mm = max(max_error_mm, length_error_mm)
        records.append(
            {
                "piece_a_index": int(match[1]),
                "edge_a_index": int(match[2]),
                "piece_b_index": int(match[3]),
                "edge_b_index": int(match[4]),
                "partial": tuple(float(value) for value in match[5:])
                != (0.0, 1.0, 0.0, 1.0),
                "length_error_mm": length_error_mm,
                "relative_length_error": float(match[0]),
                "fractions": [float(value) for value in match[5:]],
            }
        )
    return records, max_error_mm


def plan_with_upstream(
    pieces,
    *,
    cut_mode: str = "auto",
    upstream_root=None,
    strict_revision: bool = True,
    validation: str = "local",
) -> PlanResult:
    """Run the upstream solver and return the local framework's result type.

    ``validation="local"`` is the safe default: an upstream proposal is only
    marked valid when it also satisfies this repository's fixed-rectangle
    geometry gates. ``validation="upstream"`` exposes the upstream decision
    for analysis and simulation, but should not drive a robot directly.
    """
    if cut_mode not in SUPPORTED_CUT_MODES:
        raise ValueError("unsupported cut mode: {}".format(cut_mode))
    if validation not in {"local", "upstream"}:
        raise ValueError("validation must be 'local' or 'upstream'")
    if not cfg.MIN_PIECE_COUNT <= len(pieces) <= cfg.MAX_PIECE_COUNT:
        return PlanResult(
            reason="piece count {} outside {}..{}".format(
                len(pieces),
                cfg.MIN_PIECE_COUNT,
                cfg.MAX_PIECE_COUNT,
            ),
            mode="upstream_bridge",
        )
    if (
        cfg.TARGET_RECT_SIZE_MM is not None
        and tuple(cfg.TARGET_RECT_SIZE_MM) != (100.0, 60.0)
    ):
        return PlanResult(
            reason=(
                "upstream solver is fixed to 100x60 mm; local target is {}"
            ).format(cfg.TARGET_RECT_SIZE_MM),
            mode="upstream_bridge",
        )

    upstream = load_upstream(
        upstream_root,
        strict_revision=strict_revision,
    )
    polygons_px = _as_numpy_polygons(pieces)
    started = time.perf_counter()
    try:
        upstream_transforms, matches = upstream.solve(
            polygons_px,
            cut_mode,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return PlanResult(
            reason="upstream solver failed: {}".format(exc),
            mode="upstream_bridge",
            plan_stats={
                "engine": "lvreng/puzzle-vision-simulator",
                "upstream_commit": PINNED_COMMIT,
                "cut_mode": cut_mode,
                "plan_ms": elapsed_ms,
            },
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    upstream_target_center_px = np.array(
        [
            (upstream.CANVAS_W - upstream.CARD_W) * 0.5
            + upstream.CARD_W * 0.5,
            upstream.TARGET_Y + upstream.CARD_H * 0.5,
        ],
        dtype=float,
    )
    target_shift_mm = (
        np.asarray(cfg.TARGET_CENTER_MM, dtype=float)
        - upstream_target_center_px / UPSTREAM_PIXELS_PER_MM
    )
    local_transforms = [
        _transform_to_local_mm(transform, target_shift_mm)
        for transform in upstream_transforms
    ]
    target_polygon_list = [
        [
            (float(point[0]), float(point[1]))
            for point in _apply_h(
                np.asarray(piece.polygon_mm, dtype=float),
                transform,
            )
        ]
        for piece, transform in zip(pieces, local_transforms)
    ]
    metrics = _proposal_metrics(target_polygon_list)
    gate_failures = _local_gate_failures(metrics)
    valid = validation == "upstream" or not gate_failures
    reason = (
        "ok"
        if valid
        else "upstream proposal rejected by local gates: {}".format(
            "; ".join(gate_failures)
        )
    )

    operations = []
    targets_by_id = {}
    for index, (piece, target_polygon, transform) in enumerate(
        zip(pieces, target_polygon_list, local_transforms)
    ):
        piece_id = piece.piece_id or "P{}".format(index + 1)
        targets_by_id[piece_id] = target_polygon
        rotation_deg = normalize_angle_deg(
            math.degrees(
                math.atan2(transform[1, 0], transform[0, 0])
            )
        )
        operations.append(
            {
                "piece_id": piece_id,
                "source_center_mm": piece.centroid_mm,
                "target_center_mm": polygon_centroid(target_polygon),
                "rotation_deg": rotation_deg,
                "rotation_ambiguous": piece.rotation_ambiguous,
                "confidence": piece.confidence,
                "matrix_3x3_mm": transform.tolist(),
            }
        )
    seams, max_vertex_error = _seam_records(
        upstream,
        polygons_px,
        matches,
    )
    candidate_count = len(upstream.candidate_matchings(polygons_px))
    _, target_rect = _target_rectangle()
    return PlanResult(
        valid=valid,
        reason=reason,
        score=metrics["score"],
        operations=operations,
        target_polygons=targets_by_id,
        target_rect=target_rect,
        mode="upstream_bridge_{}_{}".format(cut_mode, validation),
        max_vertex_error_mm=max_vertex_error,
        fill_gap_mm2=metrics["fill_gap_mm2"],
        overlap_mm2=metrics["overlap_mm2"],
        outside_mm2=metrics["outside_mm2"],
        seams=seams,
        plan_stats={
            "engine": "lvreng/puzzle-vision-simulator",
            "upstream_commit": PINNED_COMMIT,
            "cut_mode": cut_mode,
            "validation": validation,
            "plan_ms": elapsed_ms,
            "candidate_count": candidate_count,
            "selected_match_count": len(matches),
            "dimension_error_mm": metrics["dimension_error_mm"],
            "actual_width_mm": metrics["actual_width_mm"],
            "actual_height_mm": metrics["actual_height_mm"],
            "local_gate_failures": gate_failures,
        },
    )


def plan_with_upstream_then_outer_fallback(
    pieces,
    *,
    cut_mode: str = "auto",
    upstream_root=None,
    strict_revision: bool = True,
) -> PlanResult:
    """Use the upstream proposal when safe, otherwise use local outer-first."""
    hybrid_started = time.perf_counter()
    proposal = plan_with_upstream(
        pieces,
        cut_mode=cut_mode,
        upstream_root=upstream_root,
        strict_revision=strict_revision,
        validation="local",
    )
    if proposal.valid:
        proposal.plan_stats["hybrid_total_ms"] = (
            time.perf_counter() - hybrid_started
        ) * 1000.0
        return proposal
    fallback = plan_outer_first_rectangle(pieces)
    fallback.plan_stats = dict(fallback.plan_stats)
    fallback.plan_stats["bridge_upstream_proposal"] = {
        "valid": proposal.valid,
        "reason": proposal.reason,
        "score": proposal.score,
        "plan_ms": proposal.plan_stats.get("plan_ms"),
        "fill_gap_mm2": proposal.fill_gap_mm2,
        "overlap_mm2": proposal.overlap_mm2,
        "outside_mm2": proposal.outside_mm2,
    }
    fallback.plan_stats["hybrid_total_ms"] = (
        time.perf_counter() - hybrid_started
    ) * 1000.0
    fallback.mode = "bridge_outer_fallback/{}".format(fallback.mode)
    return fallback
