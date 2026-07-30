#!/usr/bin/env python3
"""K230 puzzle recognition using a fixed manual A4 calibration."""

import gc
import os
import time

import image
from media.display import Display
from media.media import MediaManager

import realtime_a4_config as cfg
from k230_puzzle_planner import (
    COLORS,
    GRAY,
    GREEN,
    RED,
    WHITE,
    YELLOW,
    _audit_runtime_api,
    _draw_box,
    _draw_polyline,
    _draw_text,
    _init_hardware,
    _ms_delta,
    _ms_now,
    _plan_key,
    _print_plan,
    _render_status,
    _sleep_ms,
    _stop_requested,
)
from puzzle_geometry import (
    PieceTracker,
    begin_plan_debug,
    end_plan_debug,
    plan_outer_first_rectangle,
    plan_rectangle_assembly,
    polygon_overlap_area,
)
from puzzle_simulator_planner import plan_simulator_rectangle
from puzzle_placement import PlacementMonitor
from puzzle_placement import (
    final_rectangle_consensus,
    final_rectangle_metrics,
    foreground_mask_from_gray,
    placement_delta_metrics,
)
from puzzle_perf import PERF_STATS
from puzzle_realtime_state import (
    MotionDetector,
    PieceCountConsensus,
    PlacementMotionState,
    a4_detection_interval,
    phase_allows_vision,
    operator_overlay_visibility,
    operator_status_line,
    plan_frozen_pieces,
    planning_input_integrity,
    placement_ui_key,
    periodic_output_due,
    should_render_ui,
    status_ui_key,
    top_right_thumbnail_rect,
)
from puzzle_vision import (
    background_difference_threshold,
    build_polygon_scanlines,
    detect_pieces_from_canmv_image,
    polygon_white_coverage_scanlines,
)
from puzzle_a4_boundary import (
    A4BoundaryTracker,
    detect_a4_boundary,
    project_a4_mm_to_frame,
)


PLACEMENT_PHASES = (
    "WAIT_FOR_MOTION",
    "MOVING",
    "POST_MOTION_SETTLE",
    "VERIFY_PLACEMENT",
    "FINAL_VERIFY",
    "COMPLETE",
)


def _planner_selection():
    backend = getattr(cfg, "PLANNER_BACKEND", "outer_first")
    if backend == "simulator":
        return "simulator", plan_simulator_rectangle, True
    prefer_unknown = (
        cfg.TARGET_RECT_SIZE_MM is None
        or cfg.PREFER_OUTER_FIRST_PLANNER
    )
    if prefer_unknown:
        return "outer_first", plan_outer_first_rectangle, True
    return "fixed_rectangle", plan_outer_first_rectangle, False


def _draw_quad(frame, corners, color, thickness=3):
    if corners is None:
        return
    for index, point in enumerate(corners):
        other = corners[(index + 1) % len(corners)]
        frame.draw_line(
            int(point[0]),
            int(point[1]),
            int(other[0]),
            int(other[1]),
            color=color,
            thickness=thickness,
        )


def _a4_mm_to_frame(point_mm, corners):
    projected = project_a4_mm_to_frame(
        point_mm, corners
    )
    if projected is None:
        # A4 locking already rejects a degenerate quadrilateral. Retain a
        # harmless fallback for manual emergency configurations.
        return float(corners[0][0]), float(corners[0][1])
    return projected


def _draw_piece_overlay(frame, pieces, corners):
    if corners is None:
        return
    for index, piece in enumerate(pieces):
        color = COLORS[index % len(COLORS)]
        polygon = [
            _a4_mm_to_frame(point, corners)
            for point in piece.polygon_mm
        ]
        _draw_quad(frame, polygon, color, thickness=2)
        center = _a4_mm_to_frame(piece.centroid_mm, corners)
        frame.draw_cross(
            int(center[0]),
            int(center[1]),
            size=6,
            thickness=2,
            color=color,
        )
        _draw_text(
            frame,
            int(center[0]) + 5,
            int(center[1]) - 15,
            piece.piece_id or "P?",
            color,
        )


def _draw_a4_operator_overlay(
    frame, corners, divider_y_mm=None
):
    if corners is None:
        return
    _draw_quad(frame, corners, GREEN, thickness=3)
    divider = (
        cfg.DIVIDER_Y_MM
        if divider_y_mm is None
        else float(divider_y_mm)
    )
    divider_left = _a4_mm_to_frame(
        (0.0, divider), corners
    )
    divider_right = _a4_mm_to_frame(
        (cfg.A4_WIDTH_MM, divider), corners
    )
    frame.draw_line(
        int(divider_left[0]),
        int(divider_left[1]),
        int(divider_right[0]),
        int(divider_right[1]),
        color=GRAY,
        thickness=2,
    )


def _draw_plan_target_overlay(
    frame,
    plan,
    corners,
    placement_state=None,
):
    if (
        plan is None
        or corners is None
        or not plan.valid
        or not plan.target_polygons
    ):
        return
    completed = set(
        placement_state.get("completed", ())
        if placement_state is not None
        else ()
    )
    next_piece_id = (
        placement_state.get("next_piece_id")
        if placement_state is not None
        else None
    )
    for operation in plan.operations:
        piece_id = operation["piece_id"]
        polygon = plan.target_polygons.get(piece_id)
        if not polygon:
            continue
        if piece_id in completed:
            color = GREEN
            thickness = 4
        elif piece_id == next_piece_id:
            color = YELLOW
            thickness = 4
        else:
            color = YELLOW
            thickness = 2
        points = [
            _a4_mm_to_frame(point, corners)
            for point in polygon
        ]
        _draw_polyline(
            frame, points, color, thickness=thickness
        )
        center = _a4_mm_to_frame(
            operation["target_center_mm"], corners
        )
        _draw_text(
            frame,
            int(center[0]) + 4,
            int(center[1]) - 13,
            "T:{}".format(piece_id),
            color,
        )


def _operator_status_color(
    phase, plan, error, motion_active
):
    if error:
        return RED
    if motion_active or phase == "MOVING":
        return YELLOW
    if phase == "COMPLETE":
        return GREEN
    if (
        plan is not None
        and plan.operations
        and not plan.valid
    ):
        return RED
    return WHITE


def _draw_operator_status_line(
    frame, corners, text, color
):
    if corners:
        x = max(
            4,
            min(
                frame.width() - 120,
                int(min(point[0] for point in corners)),
            ),
        )
        y = min(
            frame.height() - 18,
            int(max(point[1] for point in corners)) + 3,
        )
    else:
        x = 8
        y = frame.height() - 18
    # One-pixel dark shadow keeps short text readable on the grayscale feed.
    _draw_text(frame, x + 1, y + 1, text, (0, 0, 0))
    _draw_text(frame, x, y, text, color)


def _render_live_operator_view(
    canvas,
    source_frame,
    pieces,
    a4_state,
    plan,
    placement_state,
    phase,
    stable,
    motion_active,
    error,
    candidate=None,
):
    """Render the real grayscale feed with A4-space operator overlays."""
    base_error = None
    try:
        gray = source_frame.to_grayscale(
            x_size=canvas.width(),
            y_size=canvas.height(),
        )
        canvas.draw_image(gray, 0, 0, alpha=256)
    except Exception as exc:
        if "IDE interrupt" in str(exc):
            raise
        base_error = str(exc)
        canvas.clear()
        try:
            canvas.draw_image(
                source_frame, 0, 0, alpha=256
            )
        except Exception as fallback_exc:
            if "IDE interrupt" in str(fallback_exc):
                raise
            base_error = "{}; fallback={}".format(
                base_error, fallback_exc
            )

    corners = a4_state.get("corners_px")
    if candidate is not None:
        _draw_quad(
            canvas,
            candidate.get("corners_px"),
            YELLOW,
            thickness=2,
        )
    _draw_a4_operator_overlay(
        canvas,
        corners,
        a4_state.get("divider_y_mm", cfg.DIVIDER_Y_MM),
    )

    if getattr(
        cfg,
        "OPERATOR_HIDE_OVERLAYS_DURING_MOTION",
        True,
    ):
        visibility = operator_overlay_visibility(
            phase, motion_active
        )
    else:
        visibility = {
            "a4": True,
            "status": True,
            "pieces": True,
            "targets": True,
        }
    if visibility["pieces"]:
        visible_pieces = (
            placement_state.get("visible_pieces", ())
            if placement_state is not None
            else pieces
        )
        _draw_piece_overlay(
            canvas,
            visible_pieces,
            corners,
        )
    if visibility["targets"]:
        _draw_plan_target_overlay(
            canvas,
            plan,
            corners,
            placement_state,
        )

    completed_count = (
        placement_state.get("completed_count", 0)
        if placement_state is not None
        else 0
    )
    total_count = (
        placement_state.get("total_count", len(pieces))
        if placement_state is not None
        else len(pieces)
    )
    next_piece_id = (
        placement_state.get("next_piece_id")
        if placement_state is not None
        else None
    )
    status = operator_status_line(
        phase,
        len(pieces),
        stable=stable,
        plan_available=(
            plan is not None and bool(plan.operations)
        ),
        plan_valid=(
            plan is not None and plan.valid
        ),
        next_piece_id=next_piece_id,
        completed_count=completed_count,
        total_count=total_count,
        error=error or base_error,
    )
    _draw_operator_status_line(
        canvas,
        corners,
        status,
        _operator_status_color(
            phase, plan, error or base_error, motion_active
        ),
    )
    return base_error


def _draw_gray_work_thumbnail(
    canvas,
    gray_image,
    source_frame_index,
    threshold,
    contour_threshold,
):
    """Draw a rectified A4-only grayscale image at the top-right."""
    if (
        not cfg.SHOW_GRAY_WORK_THUMBNAIL
        or gray_image is None
    ):
        return None
    try:
        x, y, width, height, scale = (
            top_right_thumbnail_rect(
                canvas.width(),
                canvas.height(),
                gray_image.width(),
                gray_image.height(),
                cfg.GRAY_THUMBNAIL_MAX_WIDTH,
                cfg.GRAY_THUMBNAIL_MAX_HEIGHT,
                cfg.GRAY_THUMBNAIL_MARGIN_PX,
            )
        )
        canvas.draw_image(
            gray_image,
            x,
            y,
            x_scale=scale,
            y_scale=scale,
            alpha=256,
        )
        _draw_box(
            canvas,
            x - 1,
            y - 1,
            width + 2,
            height + 2,
            WHITE,
        )
        label = "A4 B:{} C:{} F:{}".format(
            threshold if threshold is not None else "-",
            (
                contour_threshold
                if contour_threshold is not None
                else "-"
            ),
            source_frame_index,
        )
        label_y = min(
            canvas.height() - 16,
            y + height + 3,
        )
        _draw_text(
            canvas,
            x + 1,
            label_y + 1,
            label,
            (0, 0, 0),
        )
        _draw_text(canvas, x, label_y, label, WHITE)
        return None
    except Exception as exc:
        if "IDE interrupt" in str(exc):
            raise
        return str(exc)


def _show_output_with_ide(
    output_image,
    frame_index,
    output_index,
    previous_ide_error,
    force_ide=False,
):
    """Submit to LCD and explicitly publish selected frames to IDE Preview."""
    display_started = PERF_STATS.mark()
    Display.show_image(output_image)
    PERF_STATS.add_stage("display_ms", display_started)
    PERF_STATS.increment("display_count")

    should_stream = (
        cfg.IDE_STREAM_ENABLED
        and (
            force_ide
            or periodic_output_due(
                output_index,
                cfg.IDE_STREAM_EVERY_N_OUTPUTS,
            )
        )
    )
    current_error = previous_ide_error
    if should_stream:
        ide_started = PERF_STATS.mark()
        try:
            sender = getattr(
                output_image, "compress_for_ide", None
            )
            if sender is None:
                raise RuntimeError(
                    "image missing compress_for_ide"
                )
            os.exitpoint()
            sender(quality=cfg.IDE_STREAM_QUALITY)
            os.exitpoint()
            PERF_STATS.increment("ide_stream_count")
            current_error = None
        except Exception as exc:
            if "IDE interrupt" in str(exc):
                raise
            current_error = str(exc)
        PERF_STATS.add_stage("ide_stream_ms", ide_started)

        if current_error and current_error != previous_ide_error:
            print(
                "IDE_STREAM_ERROR,frame={},reason={}".format(
                    frame_index,
                    current_error.replace(",", ";"),
                )
            )
        elif (
            current_error is None
            and previous_ide_error is not None
        ):
            print(
                "IDE_STREAM_RECOVERED,frame={}".format(
                    frame_index
                )
            )
    return output_index + 1, current_error


def _print_a4_lock(frame_index, state):
    corners_text = "|".join(
        "{:.0f}:{:.0f}".format(point[0], point[1])
        for point in state["corners_px"]
    )
    print(
        "A4_LOCK,frame={},source={},confidence={:.2f},"
        "motion_px={:.1f},orientation={},divider_y_mm={:.1f},"
        "divider_slope_mm={:.1f},divider_confidence={:.2f},"
        "divider_detected={},frozen={},"
        "corners={}".format(
            frame_index,
            state["source"],
            state["confidence"],
            state["motion_px"],
            state.get("orientation", ""),
            state.get("divider_y_mm", cfg.DIVIDER_Y_MM),
            state.get("divider_slope_mm", 0.0),
            state.get("divider_confidence", 0.0),
            int(state.get("divider_detected", False)),
            int(state.get("frozen", False)),
            corners_text,
        )
    )


def _manual_a4_state():
    """Return the immutable runtime view of the configured A4 calibration."""
    return {
        "corners_px": [
            (float(point[0]), float(point[1]))
            for point in cfg.A4_CORNERS_PX
        ],
        "locked": True,
        "frozen": True,
        "confidence": 1.0,
        "source": "manual",
        "orientation": "manual",
        "valid_frames": 0,
        "missed_frames": 0,
        "motion_px": 0.0,
        "divider_y_mm": float(cfg.DIVIDER_Y_MM),
        "divider_slope_mm": 0.0,
        "divider_confidence": 0.0,
        "divider_detected": False,
    }


def _preserve_corner_labels(candidate, previous_corners):
    """Prevent A4 physical orientation flipping after pieces move downward."""
    if candidate is None or previous_corners is None:
        return candidate
    corners = list(candidate["corners_px"])
    variants = []
    for base in (corners, list(reversed(corners))):
        for shift in range(4):
            variants.append(base[shift:] + base[:shift])
    best = min(
        variants,
        key=lambda variant: sum(
            (
                variant[index][0]
                - previous_corners[index][0]
            )
            ** 2
            + (
                variant[index][1]
                - previous_corners[index][1]
            )
            ** 2
            for index in range(4)
        ),
    )
    candidate["corners_px"] = best
    return candidate


def _placement_screen_point(point_mm):
    scale = 1.28
    return (
        int(18 + point_mm[0] * scale),
        int(48 + point_mm[1] * scale),
    )


def _piece_color(piece_id, reference_pieces):
    for index, piece in enumerate(reference_pieces):
        if piece.piece_id == piece_id:
            return COLORS[index % len(COLORS)]
    return WHITE


def _operation_for_piece(plan, piece_id):
    for operation in plan.operations:
        if operation["piece_id"] == piece_id:
            return operation
    return None


def _placement_phase_label(phase):
    return {
        "WAIT_FOR_MOTION": "WAITING FOR MOVE",
        "MOVING": "MOVING - VISION PAUSED",
        "POST_MOTION_SETTLE": "WAITING FOR STABLE IMAGE",
        "VERIFY_PLACEMENT": "VERIFYING PLACEMENT",
        "FINAL_VERIFY": "FINAL VERIFY",
        "COMPLETE": "COMPLETE",
    }.get(phase, phase)


def _placement_display_label(phase, placement_state):
    if phase != "WAIT_FOR_MOTION":
        return _placement_phase_label(phase)
    metrics = placement_state.get("metrics", {})
    if not metrics:
        return _placement_phase_label(phase)
    piece_id = next(iter(metrics))
    item = metrics[piece_id]
    if item.get("placed") and piece_id in placement_state["completed"]:
        return "PIECE ACCEPTED"
    return "PIECE NOT CONFIRMED"


def _render_placement_status(
    canvas,
    reference_pieces,
    plan,
    placement_state,
    phase,
    frame_index,
    fps,
    next_check_ms,
    error,
    divider_y_mm=None,
):
    """Render current and target contours on one full-A4 schematic."""
    divider = (
        cfg.DIVIDER_Y_MM
        if divider_y_mm is None
        else float(divider_y_mm)
    )
    canvas.clear()
    _draw_text(
        canvas, 12, 8, "K230 PUZZLE PLACEMENT", WHITE, 2
    )
    _draw_text(
        canvas,
        580,
        12,
        "F:{} FPS:{:.1f}".format(frame_index, fps),
        GRAY,
    )

    a4_top_left = _placement_screen_point((0.0, 0.0))
    a4_bottom_right = _placement_screen_point(
        (cfg.A4_WIDTH_MM, cfg.A4_HEIGHT_MM)
    )
    _draw_box(
        canvas,
        a4_top_left[0],
        a4_top_left[1],
        a4_bottom_right[0] - a4_top_left[0],
        a4_bottom_right[1] - a4_top_left[1],
        GRAY,
        thickness=2,
    )
    divider_a = _placement_screen_point(
        (0.0, divider)
    )
    divider_b = _placement_screen_point(
        (cfg.A4_WIDTH_MM, divider)
    )
    canvas.draw_line(
        divider_a[0],
        divider_a[1],
        divider_b[0],
        divider_b[1],
        color=GRAY,
        thickness=2,
    )
    _draw_text(canvas, 20, 431, "A4 / mm", GRAY)

    completed = placement_state["completed"]
    next_piece_id = placement_state["next_piece_id"]

    # Draw desired lower-half poses first. Frozen/last verified observations
    # remain visible while the event-driven motion gate pauses recognition.
    for piece in reference_pieces:
        piece_id = piece.piece_id
        target = plan.target_polygons.get(piece_id)
        if not target:
            continue
        if piece_id in completed:
            color = GREEN
            thickness = 3
        elif piece_id == next_piece_id:
            color = YELLOW
            thickness = 3
        else:
            color = GRAY
            thickness = 2
        points = [
            _placement_screen_point(point) for point in target
        ]
        _draw_polyline(
            canvas, points, color, thickness=thickness
        )
        operation = _operation_for_piece(plan, piece_id)
        if operation is not None:
            center = _placement_screen_point(
                operation["target_center_mm"]
            )
            _draw_text(
                canvas,
                center[0] + 3,
                center[1] - 12,
                "{}{}".format(
                    piece_id,
                    " OK" if piece_id in completed else "",
                ),
                color,
            )

    for piece in placement_state["visible_pieces"]:
        color = _piece_color(piece.piece_id, reference_pieces)
        points = [
            _placement_screen_point(point)
            for point in piece.polygon_mm
        ]
        _draw_polyline(canvas, points, color, thickness=2)
        center = _placement_screen_point(piece.centroid_mm)
        canvas.draw_cross(
            center[0],
            center[1],
            size=5,
            thickness=2,
            color=color,
        )
        _draw_text(
            canvas,
            center[0] + 4,
            center[1] - 12,
            piece.piece_id,
            color,
        )

    panel_x = 320
    status_color = GREEN if phase == "COMPLETE" else YELLOW
    _draw_text(
        canvas,
        panel_x,
        55,
        _placement_display_label(phase, placement_state),
        status_color,
        2,
    )
    _draw_text(
        canvas,
        panel_x,
        83,
        "DONE:{}/{}".format(
            placement_state["completed_count"],
            placement_state["total_count"],
        ),
        WHITE,
        2,
    )
    if phase == "COMPLETE":
        _draw_text(
            canvas, panel_x, 115, "PUZZLE COMPLETE", GREEN, 2
        )
    else:
        _draw_text(
            canvas,
            panel_x,
            115,
            "NEXT:{}".format(next_piece_id or "-"),
            YELLOW,
            2,
        )
        _draw_text(
            canvas,
            panel_x,
            145,
            "MOTION-TRIGGERED VERIFY",
            GRAY,
        )
    _draw_text(
        canvas,
        panel_x,
        171,
        "OBS:{} MATCH:{}".format(
            placement_state["observed_count"],
            placement_state["matched_count"],
        ),
        GRAY,
    )
    if error:
        _draw_text(
            canvas, panel_x, 198, str(error)[:48], RED
        )

    row_y = 225
    for piece in reference_pieces:
        piece_id = piece.piece_id
        operation = _operation_for_piece(plan, piece_id)
        if operation is None:
            continue
        if piece_id in completed:
            state_text = "DONE"
            color = GREEN
        elif piece_id == next_piece_id:
            state_text = "MOVE"
            color = YELLOW
        else:
            state_text = "WAIT"
            color = GRAY
        _draw_text(
            canvas,
            panel_x,
            row_y,
            "{} {}".format(piece_id, state_text),
            color,
            2,
        )
        _draw_text(
            canvas,
            panel_x + 110,
            row_y + 3,
            "T:{:.1f},{:.1f} R:{:+.1f}".format(
                operation["target_center_mm"][0],
                operation["target_center_mm"][1],
                operation["rotation_deg"],
            ),
            WHITE,
        )
        row_y += 48


def _render_planning_status(canvas, pieces, frame_index):
    canvas.clear()
    _draw_text(
        canvas, 160, 160, "PLANNING...", YELLOW, 3
    )
    _draw_text(
        canvas,
        185,
        215,
        "FROZEN PIECES:{}".format(len(pieces)),
        WHITE,
        2,
    )
    _draw_text(
        canvas,
        190,
        250,
        "FRAME:{}".format(frame_index),
        GRAY,
    )


def _print_placement_check(frame_index, result):
    metrics_map = result.get("metrics", {})
    for piece_id, metrics in metrics_map.items():
        print(
            "VERIFY_RESULT,frame={},id={},method={},"
            "center_error_mm={},pose_bound_mm={},"
            "area_ratio={},contour_rms_mm={},"
            "contour_p90_mm={},contour_p95_mm={},"
            "coverage={},spill_ratio={},placed={}".format(
                frame_index,
                piece_id,
                metrics.get("method", "unknown"),
                "{:.1f}".format(
                    metrics["center_error_mm"]
                )
                if metrics.get("center_error_mm") is not None
                else "na",
                (
                    "{:.2f}".format(
                        metrics["pose_error_bound_mm"]
                    )
                    if metrics.get("pose_error_bound_mm")
                    is not None
                    else "na"
                ),
                (
                    "{:.3f}".format(metrics["area_ratio"])
                    if metrics.get("area_ratio") is not None
                    else "na"
                ),
                (
                    "{:.2f}".format(metrics["contour_rms_mm"])
                    if metrics.get("contour_rms_mm") is not None
                    else "na"
                ),
                (
                    "{:.2f}".format(metrics["contour_p90_mm"])
                    if metrics.get("contour_p90_mm") is not None
                    else "na"
                ),
                (
                    "{:.2f}".format(metrics["contour_p95_mm"])
                    if metrics.get("contour_p95_mm") is not None
                    else "na"
                ),
                (
                    "{:.3f}".format(metrics["target_coverage"])
                    if metrics.get("target_coverage") is not None
                    else "na"
                ),
                (
                    "{:.3f}".format(metrics["spill_ratio"])
                    if metrics.get("spill_ratio") is not None
                    else "na"
                ),
                int(bool(metrics.get("placed"))),
            )
        )
        accepted = piece_id in result["newly_completed"]
        print(
            "{},frame={},id={},reason={},method={}".format(
                "PIECE_ACCEPTED" if accepted else "PIECE_REJECTED",
                frame_index,
                piece_id,
                metrics.get("reason", "unknown"),
                metrics.get("method", "none"),
            )
        )
    print(
        "PLACEMENT_CHECK,frame={},check={},observed={},matched={},"
        "completed={}/{},next={}".format(
            frame_index,
            result["check_index"],
            result["observed_count"],
            result["matched_count"],
            result["completed_count"],
            result["total_count"],
            result["next_piece_id"] or "none",
        )
    )


def _detect_frame_pieces(
    frame,
    corners_px,
    region,
    threshold,
    divider_y_mm=None,
    collect_sanity=False,
):
    piece_gray = frame.to_grayscale(
        x_size=cfg.REALTIME_PIECE_WORK_WIDTH,
        y_size=cfg.REALTIME_PIECE_WORK_HEIGHT,
    )
    return detect_pieces_from_canmv_image(
        piece_gray,
        corners_px,
        (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT),
        region=region,
        threshold=threshold,
        divider_y_mm=divider_y_mm,
        collect_sanity=collect_sanity,
    )


def _rectified_gray(frame, corners_px, width, height):
    """Return a perspective-corrected grayscale A4 image at one size."""
    gray = frame.to_grayscale(
        x_size=int(width),
        y_size=int(height),
    )
    scale_x = float(int(width) - 1) / float(cfg.FRAME_WIDTH - 1)
    scale_y = float(int(height) - 1) / float(
        cfg.FRAME_HEIGHT - 1
    )
    work_corners = [
        (
            int(round(float(point[0]) * scale_x)),
            int(round(float(point[1]) * scale_y)),
        )
        for point in corners_px
    ]
    corrected = gray.rotation_corr(corners=work_corners)
    if (
        corrected is not None
        and hasattr(corrected, "width")
        and hasattr(corrected, "height")
        and hasattr(corrected, "to_numpy_ref")
    ):
        gray = corrected
    return gray


def _new_motion_detector(divider_y_mm=None):
    divider = (
        cfg.DIVIDER_Y_MM
        if divider_y_mm is None
        else float(divider_y_mm)
    )
    divider_row = int(
        divider
        * (cfg.MOTION_SAMPLE_HEIGHT - 1)
        / cfg.A4_HEIGHT_MM
        + 0.5
    )
    ignore_rows = max(
        1,
        int(
            cfg.MOTION_DIVIDER_IGNORE_MM
            * (cfg.MOTION_SAMPLE_HEIGHT - 1)
            / cfg.A4_HEIGHT_MM
            + 0.5
        ),
    )
    return MotionDetector(
        cfg.MOTION_PIXEL_DIFF_THRESHOLD,
        cfg.MOTION_MEAN_ABS_DIFF_THRESHOLD,
        cfg.MOTION_CHANGED_PIXEL_RATIO,
        sample_stride=cfg.MOTION_SAMPLE_STRIDE,
        divider_rows=(
            divider_row - ignore_rows,
            divider_row + ignore_rows,
        ),
    )


def _new_placement_flow():
    return PlacementMotionState(
        cfg.MOTION_START_CONFIRM_FRAMES,
        cfg.MOTION_END_CONFIRM_FRAMES,
        cfg.POST_MOTION_STABLE_FRAMES,
    )


def _count_map_text(values):
    if not values:
        return "none"
    return "|".join(
        "{}:{}".format(name, values[name])
        for name in sorted(values)
    )


def _print_frozen_piece_geometry(frame_index, pieces):
    """Emit replayable millimetre polygons once per planning attempt."""
    for index, piece in enumerate(pieces):
        piece_id = piece.piece_id or "P{}".format(index + 1)
        vertices = "|".join(
            "{:.2f}:{:.2f}".format(point[0], point[1])
            for point in piece.polygon_mm
        )
        print(
            "PIECE_GEOMETRY,frame={},id={},area_mm2={:.1f},"
            "center_mm={:.2f}:{:.2f},vertices_mm={}".format(
                frame_index,
                piece_id,
                piece.area_mm2,
                piece.centroid_mm[0],
                piece.centroid_mm[1],
                vertices,
            )
        )


def _relaxed_piece_threshold(diagnostics):
    """Choose the low-contrast retry threshold in the same gray reference."""
    if (
        diagnostics.get("segmentation_mode")
        == "background_delta"
        and diagnostics.get("background_sample_count", 0)
        >= cfg.PIECE_BACKGROUND_MIN_SAMPLES
    ):
        return background_difference_threshold(
            {
                "background_gray": diagnostics.get(
                    "background_gray", 0.0
                ),
                "background_spread_gray": diagnostics.get(
                    "background_spread_gray", 0.0
                ),
            },
            cfg.PIECE_BACKGROUND_RELAXED_DELTA_GRAY,
            cfg.PIECE_BACKGROUND_NOISE_MARGIN_GRAY,
            cfg.PIECE_BACKGROUND_MAX_DELTA_GRAY,
        )
    return int(cfg.PIECE_LOW_GRAY_THRESHOLD)


def _print_piece_diagnostics(
    frame_index,
    region,
    pieces,
    diagnostics,
    retry_used,
    consensus_state=None,
):
    sanity = diagnostics.get("gray_sanity")
    if sanity is not None:
        native = sanity["native"]
        upper = sanity["upper"]
        lower = sanity["lower"]
        upper_bbox = upper.get("bright_bbox")
        lower_bbox = lower.get("bright_bbox")
        print(
            "GRAY_SANITY,frame={},format={},size={}x{},"
            "rotation_return={},min={},max={},mean={},median={},"
            "p00={},pc={},pt={},pb={},"
            "upper_min={},upper_max={},upper_mean={:.1f},"
            "upper_bright={},upper_bbox={},"
            "lower_min={},lower_max={},lower_mean={:.1f},"
            "lower_bright={},lower_bbox={},"
            "find_min_pixels={},blob_rects={},error={}".format(
                frame_index,
                native.get("format", "na"),
                native.get("width", 0),
                native.get("height", 0),
                sanity.get("rotation_return", "na"),
                native.get("min", "na"),
                native.get("max", "na"),
                native.get("mean", "na"),
                native.get("median", "na"),
                native.get("p00", "na"),
                native.get("pc", "na"),
                native.get("pt", "na"),
                native.get("pb", "na"),
                upper.get("min", 0),
                upper.get("max", 0),
                upper.get("mean", 0.0),
                upper.get("bright", 0),
                (
                    ":".join(str(value) for value in upper_bbox)
                    if upper_bbox is not None
                    else "none"
                ),
                lower.get("min", 0),
                lower.get("max", 0),
                lower.get("mean", 0.0),
                lower.get("bright", 0),
                (
                    ":".join(str(value) for value in lower_bbox)
                    if lower_bbox is not None
                    else "none"
                ),
                sanity.get("find_min_pixels", 0),
                (
                    "|".join(
                        ":".join(str(value) for value in rect)
                        for rect in sanity.get("blob_rects", ())
                    )
                    or "none"
                ),
                native.get("error", "") or "none",
            )
        )
    rejected = diagnostics.get("rejected", {})
    trace_failures = diagnostics.get("trace_failures", {})
    expected = "na"
    samples = "na"
    if consensus_state is not None:
        if consensus_state["expected_count"] is not None:
            expected = consensus_state["expected_count"]
        samples = "{}/{}".format(
            consensus_state["valid_samples"],
            consensus_state["settle_detections"],
        )
    print(
        "PIECE_DETECT,frame={},region={},segmentation={},"
        "threshold={},contour_threshold={},"
        "bg={:.1f},bg_high={:.1f},"
        "bg_spread={:.1f},delta={:.1f},bg_samples={},retry={},"
        "raw_blobs={},accepted={},rejected={},trace_failures={},"
        "border_black_px={},component_bound={},"
        "component_candidates={}|{},"
        "boundary_primary_ok={},boundary_fallback_used={},"
        "boundary_fallback_ordered_ok={},boundary_failure_reason={},"
        "polygon_reverse_retry={},polygon_unrefined_retry={},"
        "polygon_failure_rects={},"
        "boundary_steps={},pixel_reads={},"
        "raw_vertices={},vertices={},areas_mm2={},"
        "divider_y_mm={:.1f},divider_detected={},"
        "expected={},samples={}".format(
            frame_index,
            region,
            diagnostics.get("threshold_mode", "unknown"),
            int(diagnostics.get("threshold", 0)),
            int(
                diagnostics.get(
                    "contour_threshold",
                    diagnostics.get("threshold", 0),
                )
            ),
            diagnostics.get("background_gray", 0.0),
            diagnostics.get("background_high_gray", 0.0),
            diagnostics.get("background_spread_gray", 0.0),
            diagnostics.get("threshold_delta_gray", 0.0),
            diagnostics.get("background_sample_count", 0),
            int(bool(retry_used)),
            diagnostics.get("raw_contours", 0),
            len(pieces),
            _count_map_text(rejected),
            _count_map_text(trace_failures),
            diagnostics.get("rectified_border_black_px", 0),
            diagnostics.get("component_bound_count", 0),
            diagnostics.get("component_candidate_count", 0),
            diagnostics.get(
                "contour_component_candidate_count", 0
            ),
            diagnostics.get("boundary_primary_ok", 0),
            diagnostics.get("boundary_fallback_used", 0),
            diagnostics.get(
                "boundary_fallback_ordered_ok", 0
            ),
            _count_map_text(
                diagnostics.get("boundary_failure_reason", {})
            ),
            diagnostics.get("polygon_reverse_retry_count", 0),
            diagnostics.get("polygon_unrefined_retry_count", 0),
            (
                "|".join(
                    ":".join(str(value) for value in rect)
                    for rect in diagnostics.get(
                        "polygon_failure_rects", ()
                    )
                )
                or "none"
            ),
            diagnostics.get("boundary_steps", 0),
            diagnostics.get("pixel_reads", 0),
            "|".join(
                str(value)
                for value in diagnostics.get(
                    "detected_vertex_counts", []
                )
            )
            or "none",
            "|".join(
                str(len(piece.polygon_mm)) for piece in pieces
            )
            or "none",
            "|".join(
                "{:.0f}".format(piece.area_mm2)
                for piece in pieces
            )
            or "none",
            diagnostics.get(
                "divider_y_mm", cfg.DIVIDER_Y_MM
            ),
            int(
                diagnostics.get("divider_detected", False)
            ),
            expected,
            samples,
        )
    )


def _report_performance(frame_index):
    if not PERF_STATS.report_due(frame_index):
        return
    snapshot = PERF_STATS.window_snapshot(reset=False)
    print(PERF_STATS.format_report(frame_index))
    counters = snapshot["counters"]
    if counters:
        print(
            "[PERF_COUNT] frame={} {}".format(
                frame_index,
                " ".join(
                    "{}={}".format(name, counters[name])
                    for name in sorted(counters)
                ),
            )
        )
    PERF_STATS.window_snapshot(reset=True)


def main():
    sensor = None
    canvases = []
    _audit_runtime_api()
    auto_calibrate_a4 = bool(cfg.AUTO_CALIBRATE_A4)
    boundary_tracker = (
        A4BoundaryTracker() if auto_calibrate_a4 else None
    )
    a4_state = (
        boundary_tracker.state()
        if auto_calibrate_a4
        else _manual_a4_state()
    )
    piece_tracker = PieceTracker()
    frame_index = 0
    fps = 0.0
    start_ms = _ms_now()
    active_plan = None
    active_plan_key = None
    last_failed_plan_key = None
    last_stable = False
    last_piece_stable = False
    last_a4_locked = False
    last_pieces = []
    last_piece_detection_frame = -1000000
    last_boundary_diagnostics = {}
    canvas_index = 0
    a4_lock_frame = None
    phase = "ACQUIRE"
    placement_monitor = None
    reference_pieces = []
    initial_total_piece_area = 0.0
    placement_started_ms = 0
    last_placement_check_ms = 0
    placement_complete_logged = False
    target_scanlines = {}
    motion_detector = None
    placement_flow = None
    motion_metrics = {
        "mean_abs_diff": 0.0,
        "changed_ratio": 0.0,
        "motion": False,
        "scene_mean_abs_diff": 0.0,
        "scene_changed_ratio": 0.0,
        "scene_change": False,
    }
    before_foreground_mask = None
    verify_samples = []
    verify_started_frame = None
    last_motion_diagnostic_frame = -1000000
    last_rendered_state = None
    complete_displayed = False
    piece_count_consensus = PieceCountConsensus(
        (
            cfg.PLANNING_REQUIRED_PIECE_COUNT
            if cfg.PLANNING_REQUIRED_PIECE_COUNT is not None
            else cfg.MIN_PIECE_COUNT
        ),
        cfg.MAX_PIECE_COUNT,
        cfg.PIECE_COUNT_WINDOW_DETECTIONS,
        cfg.PIECE_COUNT_SETTLE_DETECTIONS,
        cfg.PIECE_COUNT_MIN_CONFIRMATIONS,
    )
    consensus_state = piece_count_consensus.state()
    tracker_expected_count = None
    bad_count_detections = 0
    piece_detection_count = 0
    last_piece_diagnostic_signature = None
    last_piece_diagnostics = {}
    last_input_integrity_signature = None
    pending_reason = "a4_unlocked"
    last_piece_gray = None
    last_piece_gray_frame = -1
    last_piece_gray_threshold = None
    last_piece_contour_threshold = None
    last_thumbnail_error = None
    last_operator_view_error = None
    ide_output_index = 0
    last_ide_stream_error = None
    PERF_STATS.enabled = bool(cfg.ENABLE_STAGE_TIMING)
    PERF_STATS.reset()

    try:
        sensor = _init_hardware()
        # The automatic acquisition preview transitions to this status canvas
        # after lock, so both canvases are prepared even when camera preview is
        # initially visible.
        canvases = [
            image.Image(
                cfg.FRAME_WIDTH,
                cfg.FRAME_HEIGHT,
                image.RGB565,
            ),
            image.Image(
                cfg.FRAME_WIDTH,
                cfg.FRAME_HEIGHT,
                image.RGB565,
            ),
        ]
        print(
            "START_REALTIME_A4,frame={}x{},boundary={}x{},"
            "piece_work={}x{},a4_every={},piece_every={},"
            "debug_camera={},planner={},"
            "placement_check_ms={},a4_hold_misses={},"
            "piece_segment={},piece_deltas={}|{},"
            "fixed_fallbacks={}|{},count_window={},"
            "count_settle={},gray_thumbnail={},gray_sanity={},"
            "ide_stream=explicit,ide_quality={},"
            "ide_every={},plan_debug={},"
            "plan_debug_ms={},dynamic_divider={},"
            "operator_view={},a4_calibration={}".format(
                cfg.FRAME_WIDTH,
                cfg.FRAME_HEIGHT,
                cfg.A4_DETECT_WIDTH,
                cfg.A4_DETECT_HEIGHT,
                cfg.REALTIME_PIECE_WORK_WIDTH,
                cfg.REALTIME_PIECE_WORK_HEIGHT,
                cfg.A4_DETECT_INTERVAL_ACQUIRE,
                cfg.PIECE_DETECT_EVERY_N_FRAMES,
                int(cfg.DEBUG_SHOW_CAMERA),
                _planner_selection()[0],
                cfg.PLACING_VERIFICATION_INTERVAL_MS,
                cfg.A4_HOLD_MISSED_FRAMES,
                cfg.PIECE_SEGMENTATION_MODE,
                cfg.PIECE_BACKGROUND_DELTA_GRAY,
                cfg.PIECE_BACKGROUND_RELAXED_DELTA_GRAY,
                cfg.WHITE_GRAY_THRESHOLD,
                cfg.PIECE_LOW_GRAY_THRESHOLD,
                cfg.PIECE_COUNT_WINDOW_DETECTIONS,
                cfg.PIECE_COUNT_SETTLE_DETECTIONS,
                int(cfg.SHOW_GRAY_WORK_THUMBNAIL),
                int(
                    getattr(
                        cfg,
                        "ENABLE_GRAY_SANITY_DIAGNOSTICS",
                        False,
                    )
                ),
                cfg.IDE_STREAM_QUALITY,
                cfg.IDE_STREAM_EVERY_N_OUTPUTS,
                int(cfg.ENABLE_PLAN_DEBUG),
                cfg.PLAN_DEBUG_INTERVAL_MS,
                int(
                    cfg.ENABLE_DYNAMIC_DIVIDER
                    and auto_calibrate_a4
                ),
                (
                    "live_grayscale"
                    if getattr(
                        cfg,
                        "LIVE_GRAYSCALE_OPERATOR_VIEW",
                        False,
                    )
                    else "schematic"
                ),
                (
                    "automatic_initial_lock_frozen"
                    if auto_calibrate_a4
                    else "manual_fixed"
                ),
            )
        )

        while True:
            os.exitpoint()
            if _stop_requested(start_ms, frame_index):
                print(
                    "STOP,reason=configured_limit,frame={}".format(
                        frame_index
                    )
                )
                break
            if (
                phase == "COMPLETE"
                and complete_displayed
                and not phase_allows_vision(phase)
            ):
                # Preserve the final canvas without snapshot, A4 tracking,
                # segmentation, coverage scans, or display writes.
                _sleep_ms(50)
                continue
            PERF_STATS.begin_frame(frame_index)
            frame_start = _ms_now()
            capture_started = PERF_STATS.mark()
            frame = sensor.snapshot()
            PERF_STATS.add_stage("capture_ms", capture_started)
            if frame is None:
                if (
                    frame_index
                    % cfg.A4_STATUS_PRINT_EVERY_N_FRAMES
                    == 0
                ):
                    print(
                        "DETECTION_ERROR,frame={},"
                        "reason=snapshot_none".format(frame_index)
                    )
                frame_index += 1
                PERF_STATS.end_frame()
                _report_performance(frame_index)
                continue

            error = None
            pieces = []
            stable = False
            candidate = None
            boundary_diagnostics = {}
            try:
                if auto_calibrate_a4:
                    a4_state = boundary_tracker.state()
                    boundary_interval = a4_detection_interval(
                        phase,
                        cfg.A4_DETECT_INTERVAL_ACQUIRE,
                        cfg.A4_DETECT_INTERVAL_PLACING,
                        a4_state["locked"],
                    )
                    boundary_due = (
                        boundary_interval is not None
                        and (
                            not a4_state["locked"]
                            or frame_index % boundary_interval == 0
                        )
                    )
                    if boundary_due:
                        a4_started = PERF_STATS.mark()
                        boundary_gray = frame.to_grayscale(
                            x_size=cfg.A4_DETECT_WIDTH,
                            y_size=cfg.A4_DETECT_HEIGHT,
                        )
                        candidate, boundary_diagnostics = (
                            detect_a4_boundary(
                                boundary_gray,
                                (
                                    cfg.FRAME_WIDTH,
                                    cfg.FRAME_HEIGHT,
                                ),
                            )
                        )
                        last_boundary_diagnostics = (
                            boundary_diagnostics
                        )
                        if phase in PLACEMENT_PHASES:
                            candidate = _preserve_corner_labels(
                                candidate,
                                a4_state["corners_px"],
                            )
                        a4_state = boundary_tracker.update(
                            candidate
                        )
                        if a4_state["locked"]:
                            a4_state = boundary_tracker.freeze()
                        PERF_STATS.add_stage(
                            "a4_detect_ms", a4_started
                        )
                    else:
                        boundary_diagnostics = (
                            last_boundary_diagnostics
                        )

                if a4_state["locked"] and phase == "ACQUIRE":
                    piece_due = (
                        frame_index - last_piece_detection_frame
                        >= cfg.PIECE_DETECT_EVERY_N_FRAMES
                    )
                    if piece_due:
                        piece_detection_count += 1
                        collect_gray_sanity = bool(
                            getattr(
                                cfg,
                                "ENABLE_GRAY_SANITY_DIAGNOSTICS",
                                False,
                            )
                            and (
                                piece_detection_count == 1
                                or piece_detection_count
                                % max(
                                    1,
                                    getattr(
                                        cfg,
                                        "GRAY_SANITY_EVERY_N_DETECTIONS",
                                        5,
                                    ),
                                )
                                == 0
                            )
                        )
                        pieces, piece_diagnostics = (
                            _detect_frame_pieces(
                                frame,
                                a4_state["corners_px"],
                                "upper",
                                None,
                                (
                                    a4_state["divider_y_mm"]
                                    if a4_state.get(
                                        "divider_detected",
                                        False,
                                    )
                                    else None
                                ),
                                collect_sanity=collect_gray_sanity,
                            )
                        )
                        retry_used = False
                        primary_threshold = int(
                            piece_diagnostics.get(
                                "threshold",
                                cfg.WHITE_GRAY_THRESHOLD,
                            )
                        )
                        relaxed_threshold = (
                            _relaxed_piece_threshold(
                                piece_diagnostics
                            )
                        )
                        previous_consensus = (
                            piece_count_consensus.state()
                        )
                        expected_count = previous_consensus[
                            "expected_count"
                        ]
                        probe_due = (
                            not previous_consensus["ready"]
                            and piece_detection_count
                            % max(
                                1,
                                cfg.PIECE_THRESHOLD_PROBE_EVERY_N_DETECTIONS,
                            )
                            == 0
                        )
                        retry_due = (
                            len(pieces) < cfg.MIN_PIECE_COUNT
                            or (
                                expected_count is not None
                                and len(pieces) < expected_count
                            )
                            or probe_due
                        )
                        if (
                            retry_due
                            and relaxed_threshold
                            < primary_threshold
                        ):
                            (
                                retry_pieces,
                                retry_diagnostics,
                            ) = _detect_frame_pieces(
                                frame,
                                a4_state["corners_px"],
                                "upper",
                                relaxed_threshold,
                                (
                                    a4_state["divider_y_mm"]
                                    if a4_state.get(
                                        "divider_detected",
                                        False,
                                    )
                                    else None
                                ),
                                collect_sanity=collect_gray_sanity,
                            )
                            if len(retry_pieces) > len(pieces):
                                pieces = retry_pieces
                                piece_diagnostics = (
                                    retry_diagnostics
                                )
                                retry_used = True
                        last_piece_diagnostics = piece_diagnostics
                        last_piece_gray = piece_diagnostics.get(
                            "rectified"
                        )
                        last_piece_gray_frame = frame_index
                        last_piece_gray_threshold = int(
                            piece_diagnostics.get(
                                "threshold",
                                cfg.WHITE_GRAY_THRESHOLD,
                            )
                        )
                        last_piece_contour_threshold = int(
                            piece_diagnostics.get(
                                "contour_threshold",
                                last_piece_gray_threshold,
                            )
                        )
                        raw_pieces = pieces
                        raw_piece_count = len(raw_pieces)
                        polygon_fit_incomplete = (
                            piece_diagnostics.get(
                                "rejected", {}
                            ).get("polygon", 0)
                            > 0
                        )
                        if (
                            tracker_expected_count is None
                            and not polygon_fit_incomplete
                        ):
                            consensus_state = (
                                piece_count_consensus.update(
                                    raw_piece_count
                                )
                            )
                            if consensus_state["ready"]:
                                tracker_expected_count = (
                                    consensus_state[
                                        "expected_count"
                                    ]
                                )
                                piece_tracker.reset(
                                    tracker_expected_count
                                )
                        if polygon_fit_incomplete:
                            tracker_stable = False
                            stable = False
                            # Show the current frame's accepted contours so a
                            # fit failure cannot leave stale, misleading
                            # overlays on screen. Planning remains fail-closed.
                            pieces = raw_pieces
                            pending_reason = (
                                "polygon_fit_incomplete"
                            )
                        elif (
                            tracker_expected_count is None
                            or raw_piece_count
                            != tracker_expected_count
                        ):
                            tracker_stable = False
                            stable = False
                            pieces = (
                                last_pieces
                                if tracker_expected_count is not None
                                else raw_pieces
                            )
                            if tracker_expected_count is None:
                                pending_reason = consensus_state[
                                    "reason"
                                ]
                            else:
                                bad_count_detections += 1
                                pending_reason = (
                                    "holding_incomplete_detection"
                                    if bad_count_detections
                                    <= cfg.BAD_COUNT_HOLD_DETECTIONS
                                    else "count_reacquire_pending"
                                )
                                if (
                                    bad_count_detections
                                    >= cfg.COUNT_REACQUIRE_FAILURES
                                ):
                                    piece_count_consensus.reset()
                                    consensus_state = (
                                        piece_count_consensus.state()
                                    )
                                    piece_tracker.reset()
                                    tracker_expected_count = None
                                    bad_count_detections = 0
                                    pending_reason = (
                                        "count_reacquire"
                                    )
                        else:
                            bad_count_detections = 0
                            pieces, tracker_stable = (
                                piece_tracker.update(raw_pieces)
                            )
                            stable = tracker_stable
                            pending_reason = (
                                "ready"
                                if tracker_stable
                                else "tracker_stabilizing"
                            )
                        diagnostic_signature = (
                            raw_piece_count,
                            int(
                                piece_diagnostics.get(
                                    "threshold", 0
                                )
                            ),
                            tuple(
                                sorted(
                                    piece_diagnostics.get(
                                        "rejected", {}
                                    ).items()
                                )
                            ),
                            tuple(
                                sorted(
                                    piece_diagnostics.get(
                                        "trace_failures", {}
                                    ).items()
                                )
                            ),
                            tuple(
                                piece_diagnostics.get(
                                    "detected_vertex_counts", []
                                )
                            ),
                            int(
                                round(
                                    piece_diagnostics.get(
                                        "divider_y_mm",
                                        cfg.DIVIDER_Y_MM,
                                    )
                                    * 2.0
                                )
                            ),
                            consensus_state["reason"],
                        )
                        diagnostic_due = (
                            retry_used
                            or diagnostic_signature
                            != last_piece_diagnostic_signature
                            or piece_detection_count
                            % max(
                                1,
                                cfg.PIECE_DIAGNOSTIC_PRINT_EVERY_N_DETECTIONS,
                            )
                            == 0
                        )
                        if diagnostic_due:
                            _print_piece_diagnostics(
                                frame_index,
                                "upper",
                                raw_pieces,
                                piece_diagnostics,
                                retry_used,
                                consensus_state,
                            )
                            last_piece_diagnostic_signature = (
                                diagnostic_signature
                            )
                        last_piece_detection_frame = frame_index
                        last_pieces = pieces
                        last_piece_stable = stable
                    else:
                        pieces = last_pieces
                        stable = last_piece_stable
                elif (
                    a4_state["locked"]
                    and phase in PLACEMENT_PHASES
                    and placement_monitor is not None
                ):
                    pieces = placement_monitor.visible_pieces()
                    stable = True
                    if phase != "COMPLETE":
                        motion_gray = _rectified_gray(
                            frame,
                            a4_state["corners_px"],
                            cfg.MOTION_SAMPLE_WIDTH,
                            cfg.MOTION_SAMPLE_HEIGHT,
                        )
                        motion_metrics = motion_detector.update(
                            motion_gray.to_numpy_ref()
                        )
                        if phase == "VERIFY_PLACEMENT":
                            if motion_metrics["motion"]:
                                phase = (
                                    placement_flow.verification_interrupted()
                                )
                                verify_samples = []
                                verify_started_frame = None
                                print(
                                    "MOTION_ACTIVE,frame={},"
                                    "mean_diff={:.2f},"
                                    "changed_ratio={:.3f},"
                                    "reason=verify_interrupted".format(
                                        frame_index,
                                        motion_metrics[
                                            "mean_abs_diff"
                                        ],
                                        motion_metrics[
                                            "changed_ratio"
                                        ],
                                    )
                                )
                        else:
                            motion_signal = motion_metrics["motion"]
                            motion_source = "adjacent"
                            if (
                                phase == "WAIT_FOR_MOTION"
                                and motion_metrics.get(
                                    "scene_change", False
                                )
                            ):
                                motion_signal = True
                                if not motion_metrics["motion"]:
                                    motion_source = "scene"
                            motion_state = placement_flow.update(
                                motion_signal,
                                frame_index,
                            )
                            phase = motion_state["phase"]
                            event = motion_state["event"]
                            if event == "MOTION_START":
                                print(
                                    "MOTION_START,frame={},"
                                    "mean_diff={:.2f},"
                                    "changed_ratio={:.3f},"
                                    "scene_mean_diff={:.2f},"
                                    "scene_changed_ratio={:.3f},"
                                    "source={}".format(
                                        motion_state[
                                            "motion_start_frame"
                                        ],
                                        motion_metrics[
                                            "mean_abs_diff"
                                        ],
                                        motion_metrics[
                                            "changed_ratio"
                                        ],
                                        motion_metrics.get(
                                            "scene_mean_abs_diff",
                                            0.0,
                                        ),
                                        motion_metrics.get(
                                            "scene_changed_ratio",
                                            0.0,
                                        ),
                                        motion_source,
                                    )
                                )
                            if motion_state["motion_ended"]:
                                print(
                                    "MOTION_END,frame={},"
                                    "stable_frames={}".format(
                                        frame_index,
                                        motion_state[
                                            "stable_after_motion_count"
                                        ],
                                    )
                                )
                            if event == "POST_MOTION_STABLE":
                                print(
                                    "POST_MOTION_STABLE,frame={},"
                                    "stable_frames={}".format(
                                        frame_index,
                                        motion_state[
                                            "stable_after_motion_count"
                                        ],
                                    )
                                )
                            if (
                                phase == "MOVING"
                                and frame_index
                                - last_motion_diagnostic_frame
                                >= cfg.A4_STATUS_PRINT_EVERY_N_FRAMES
                            ):
                                print(
                                    "MOTION_ACTIVE,frame={},"
                                    "mean_diff={:.2f},"
                                    "changed_ratio={:.3f}".format(
                                        frame_index,
                                        motion_metrics[
                                            "mean_abs_diff"
                                        ],
                                        motion_metrics[
                                            "changed_ratio"
                                        ],
                                    )
                                )
                                last_motion_diagnostic_frame = frame_index
                            if (
                                phase == "WAIT_FOR_MOTION"
                                and frame_index
                                - last_motion_diagnostic_frame
                                >= getattr(
                                    cfg,
                                    "MOTION_WAIT_DIAGNOSTIC_INTERVAL_FRAMES",
                                    60,
                                )
                            ):
                                print(
                                    "MOTION_WAIT,frame={},"
                                    "mean_diff={:.2f},"
                                    "changed_ratio={:.3f},"
                                    "scene_mean_diff={:.2f},"
                                    "scene_changed_ratio={:.3f}".format(
                                        frame_index,
                                        motion_metrics[
                                            "mean_abs_diff"
                                        ],
                                        motion_metrics[
                                            "changed_ratio"
                                        ],
                                        motion_metrics.get(
                                            "scene_mean_abs_diff",
                                            0.0,
                                        ),
                                        motion_metrics.get(
                                            "scene_changed_ratio",
                                            0.0,
                                        ),
                                    )
                                )
                                last_motion_diagnostic_frame = frame_index
                            watchdog_due = (
                                phase == "WAIT_FOR_MOTION"
                                and cfg.ENABLE_PLACEMENT_WATCHDOG
                                and _ms_delta(
                                    _ms_now(),
                                    last_placement_check_ms,
                                )
                                >= cfg.PLACING_VERIFICATION_INTERVAL_MS
                            )
                            if watchdog_due:
                                phase = "VERIFY_PLACEMENT"
                                placement_flow.phase = phase
                                verify_started_frame = frame_index
                                verify_samples = []
                                print(
                                    "VERIFY_START,frame={},"
                                    "trigger=watchdog".format(
                                        frame_index
                                    )
                                )
                    if phase == "VERIFY_PLACEMENT":
                        if verify_started_frame is None:
                            verify_started_frame = frame_index
                            verify_samples = []
                            print(
                                "VERIFY_START,frame={},trigger=motion,"
                                "id={}".format(
                                    frame_index,
                                    placement_monitor.next_piece_id()
                                    or "none",
                                )
                            )
                        observations, placement_diagnostics = (
                            _detect_frame_pieces(
                                frame,
                                a4_state["corners_px"],
                                "full",
                                None,
                                (
                                    a4_state["divider_y_mm"]
                                    if a4_state.get(
                                        "divider_detected",
                                        False,
                                    )
                                    else None
                                ),
                            )
                        )
                        last_piece_gray = (
                            placement_diagnostics.get("rectified")
                        )
                        last_piece_gray_frame = frame_index
                        last_piece_gray_threshold = int(
                            placement_diagnostics.get(
                                "threshold",
                                cfg.WHITE_GRAY_THRESHOLD,
                            )
                        )
                        last_piece_contour_threshold = int(
                            placement_diagnostics.get(
                                "contour_threshold",
                                last_piece_gray_threshold,
                            )
                        )
                        gray_array = placement_diagnostics[
                            "rectified"
                        ].to_numpy_ref()
                        after_mask = foreground_mask_from_gray(
                            gray_array, last_piece_gray_threshold
                        )
                        next_piece_id = (
                            placement_monitor.next_piece_id()
                        )
                        delta = None
                        if (
                            next_piece_id is not None
                            and before_foreground_mask is not None
                        ):
                            reference = (
                                placement_monitor.reference_by_id[
                                    next_piece_id
                                ]
                            )
                            delta = placement_delta_metrics(
                                before_foreground_mask,
                                after_mask,
                                cfg.REALTIME_PIECE_WORK_WIDTH,
                                cfg.REALTIME_PIECE_WORK_HEIGHT,
                                active_plan.target_polygons[
                                    next_piece_id
                                ],
                                reference.polygon_mm,
                                reference.area_mm2,
                            )
                        final_candidate = (
                            next_piece_id is None
                            or len(placement_monitor.completed) + 1
                            >= len(placement_monitor.order)
                        )
                        verify_sample_goal = (
                            max(
                                cfg.POST_MOTION_VERIFY_SAMPLES,
                                cfg.FINAL_VERIFY_SAMPLE_COUNT,
                            )
                            if final_candidate
                            else cfg.POST_MOTION_VERIFY_SAMPLES
                        )
                        final_metrics = None
                        if final_candidate:
                            final_metrics = final_rectangle_metrics(
                                after_mask,
                                cfg.REALTIME_PIECE_WORK_WIDTH,
                                cfg.REALTIME_PIECE_WORK_HEIGHT,
                                active_plan.target_rect,
                                initial_total_piece_area,
                            )
                        verify_samples.append(
                            {
                                "observations": observations,
                                "delta_metrics": delta,
                                "foreground_mask": after_mask,
                                "final_scene_metrics": final_metrics,
                            }
                        )
                        print(
                            "VERIFY_SAMPLE,frame={},sample={}/{},"
                            "id={},observed={},delta_coverage={},"
                            "delta_area_ratio={},delta_spill={}".format(
                                frame_index,
                                len(verify_samples),
                                verify_sample_goal,
                                next_piece_id or "none",
                                len(observations),
                                (
                                    "{:.3f}".format(
                                        delta[
                                            "added_target_coverage"
                                        ]
                                    )
                                    if delta is not None
                                    else "na"
                                ),
                                (
                                    "{:.3f}".format(
                                        delta["added_area_ratio"]
                                    )
                                    if delta is not None
                                    else "na"
                                ),
                                (
                                    "{:.3f}".format(
                                        delta[
                                            "added_spill_ratio"
                                        ]
                                    )
                                    if delta is not None
                                    else "na"
                                ),
                            )
                        )
                        if (
                            len(verify_samples)
                            >= verify_sample_goal
                        ):
                            placement_result = (
                                placement_monitor.check_samples(
                                    verify_samples
                                )
                            )
                            before_foreground_mask = verify_samples[
                                -1
                            ]["foreground_mask"]
                            last_placement_check_ms = _ms_now()
                            pieces = (
                                placement_monitor.visible_pieces()
                            )
                            _print_placement_check(
                                frame_index, placement_result
                            )
                            if placement_result["done"]:
                                phase = "FINAL_VERIFY"
                                placement_flow.phase = phase
                                final_result = (
                                    final_rectangle_consensus(
                                        [
                                            sample[
                                                "final_scene_metrics"
                                            ]
                                            for sample in verify_samples
                                            if sample[
                                                "final_scene_metrics"
                                            ]
                                            is not None
                                        ]
                                    )
                                )
                                print(
                                    "FINAL_SCENE_METRICS,frame={},"
                                    "fill_ratio={:.3f},"
                                    "area_ratio={:.3f},"
                                    "bbox_mm={:.1f}x{:.1f},"
                                    "width_error_mm={:.1f},"
                                    "height_error_mm={:.1f},"
                                    "spill_ratio={:.3f},passes={}/{},"
                                    "valid={}".format(
                                        frame_index,
                                        final_result["fill_ratio"],
                                        final_result[
                                            "final_area_ratio"
                                        ],
                                        final_result[
                                            "detected_width_mm"
                                        ],
                                        final_result[
                                            "detected_height_mm"
                                        ],
                                        final_result[
                                            "width_error_mm"
                                        ],
                                        final_result[
                                            "height_error_mm"
                                        ],
                                        final_result["spill_ratio"],
                                        final_result["pass_count"],
                                        final_result[
                                            "required_passes"
                                        ],
                                        int(final_result["valid"]),
                                    )
                                )
                                if final_result["valid"]:
                                    phase = "COMPLETE"
                                    placement_flow.verification_finished(
                                        complete=True
                                    )
                                    print(
                                        "FINAL_ACCEPTED,frame={},"
                                        "passes={}/{}".format(
                                            frame_index,
                                            final_result["pass_count"],
                                            final_result[
                                                "required_passes"
                                            ],
                                        )
                                    )
                                else:
                                    phase = (
                                        placement_flow.verification_finished()
                                    )
                                    print(
                                        "FINAL_REJECTED,frame={},"
                                        "passes={}/{}".format(
                                            frame_index,
                                            final_result["pass_count"],
                                            final_result[
                                                "required_passes"
                                            ],
                                        )
                                    )
                            else:
                                phase = (
                                    placement_flow.verification_finished()
                                )
                            if phase == "WAIT_FOR_MOTION":
                                motion_detector.accept_current_as_reference()
                            verify_samples = []
                            verify_started_frame = None
                        if phase == "COMPLETE":
                            if not placement_complete_logged:
                                placement_complete_logged = True
                                print(
                                    "PLACEMENT_COMPLETE,frame={},"
                                    "elapsed_ms={},count={}".format(
                                        frame_index,
                                        _ms_delta(
                                            _ms_now(),
                                            placement_started_ms,
                                        ),
                                        placement_result[
                                            "total_count"
                                        ],
                                    )
                                )
                elif phase == "ACQUIRE":
                    pieces = []
                    last_piece_stable = False
                    last_piece_detection_frame = -1000000
                    pending_reason = "a4_unlocked"
                    last_piece_gray = None
                    last_piece_gray_frame = -1
                    last_piece_gray_threshold = None
                    last_piece_contour_threshold = None
                elif placement_monitor is not None:
                    # Keep the frozen plan and most recent contour-only state
                    # while A4 tracking is temporarily lost.
                    pieces = placement_monitor.visible_pieces()
                    stable = True
            except Exception as exc:
                if "IDE interrupt" in str(exc):
                    raise
                error = str(exc)
                if auto_calibrate_a4:
                    a4_state = boundary_tracker.state()
                if (
                    phase in PLACEMENT_PHASES
                    and placement_monitor is not None
                ):
                    pieces = placement_monitor.visible_pieces()
                    stable = True
                else:
                    pieces = last_pieces
                    stable = False
                    last_piece_stable = False
                    pending_reason = "detection_error"

            if a4_state["locked"] and not last_a4_locked:
                a4_lock_frame = frame_index
                _print_a4_lock(frame_index, a4_state)
            elif not a4_state["locked"]:
                a4_lock_frame = None
                if last_a4_locked and phase == "ACQUIRE":
                    piece_count_consensus.reset()
                    piece_tracker.reset()
                    tracker_expected_count = None
                    bad_count_detections = 0
                    consensus_state = (
                        piece_count_consensus.state()
                    )
                    piece_detection_count = 0
                    last_piece_diagnostic_signature = None

            if phase == "ACQUIRE":
                if (
                    not stable
                    or error
                    or not a4_state["locked"]
                ):
                    active_plan = None
                    active_plan_key = None
                else:
                    key = _plan_key(pieces)
                    input_integrity = planning_input_integrity(
                        pieces,
                        cfg.TARGET_RECT_SIZE_MM,
                        polygon_overlap_area,
                        required_piece_count=(
                            cfg.PLANNING_REQUIRED_PIECE_COUNT
                        ),
                        area_ratio_min=(
                            cfg.PLANNING_INPUT_AREA_RATIO_MIN
                        ),
                        area_ratio_max=(
                            cfg.PLANNING_INPUT_AREA_RATIO_MAX
                        ),
                        max_pair_overlap_ratio=(
                            cfg.PLANNING_INPUT_MAX_PAIR_OVERLAP_RATIO
                        ),
                        rejected_border_blobs=(
                            last_piece_diagnostics.get(
                                "rejected", {}
                            ).get("border", 0)
                        ),
                        max_rejected_border_blobs=(
                            cfg.PLANNING_INPUT_MAX_BORDER_BLOBS
                        ),
                    )
                    integrity_signature = (
                        input_integrity["valid"],
                        input_integrity["failures"],
                        input_integrity["piece_count"],
                        int(
                            input_integrity["total_area_mm2"]
                            + 0.5
                        ),
                        int(
                            input_integrity[
                                "max_pair_overlap_ratio"
                            ]
                            * 1000.0
                            + 0.5
                        ),
                        input_integrity[
                            "rejected_border_blobs"
                        ],
                    )
                    if not input_integrity["valid"]:
                        active_plan = None
                        active_plan_key = None
                        pending_reason = "input_{}".format(
                            input_integrity["reason"]
                        )
                        if (
                            integrity_signature
                            != last_input_integrity_signature
                        ):
                            pair = input_integrity[
                                "max_pair_overlap_pair"
                            ]
                            print(
                                "PLANNING_INPUT_INVALID,frame={},"
                                "failures={},count={}/{},"
                                "total_area_mm2={:.1f},"
                                "target_area_mm2={},area_ratio={},"
                                "max_pair_overlap={:.3f},pair={},"
                                "border_blobs={}/{}".format(
                                    frame_index,
                                    "|".join(
                                        input_integrity["failures"]
                                    ),
                                    input_integrity["piece_count"],
                                    (
                                        input_integrity[
                                            "required_piece_count"
                                        ]
                                        if input_integrity[
                                            "required_piece_count"
                                        ]
                                        is not None
                                        else "any"
                                    ),
                                    input_integrity[
                                        "total_area_mm2"
                                    ],
                                    (
                                        "{:.1f}".format(
                                            input_integrity[
                                                "target_area_mm2"
                                            ]
                                        )
                                        if input_integrity[
                                            "target_area_mm2"
                                        ]
                                        is not None
                                        else "na"
                                    ),
                                    (
                                        "{:.3f}".format(
                                            input_integrity[
                                                "area_ratio"
                                            ]
                                        )
                                        if input_integrity[
                                            "area_ratio"
                                        ]
                                        is not None
                                        else "na"
                                    ),
                                    input_integrity[
                                        "max_pair_overlap_ratio"
                                    ],
                                    (
                                        "{}:{}".format(
                                            pair[0], pair[1]
                                        )
                                        if pair is not None
                                        else "none"
                                    ),
                                    input_integrity[
                                        "rejected_border_blobs"
                                    ],
                                    cfg.PLANNING_INPUT_MAX_BORDER_BLOBS,
                                )
                            )
                        last_input_integrity_signature = (
                            integrity_signature
                        )
                    elif key == last_failed_plan_key:
                        last_input_integrity_signature = None
                        pending_reason = (
                            "holding_repeated_failed_plan_input"
                        )
                    elif not last_stable or active_plan is None:
                        last_input_integrity_signature = None
                        plan_start_ms = _ms_now()
                        (
                            configured_planner,
                            unknown_planner,
                            prefer_unknown_planner,
                        ) = _planner_selection()
                        print(
                            "PLANNING_START,frame={},planner="
                            "{},count={}".format(
                                frame_index,
                                configured_planner,
                                len(pieces),
                            )
                        )
                        _print_frozen_piece_geometry(
                            frame_index, pieces
                        )
                        phase = "PLANNING"
                        planning_canvas = canvases[canvas_index]
                        canvas_index = 1 - canvas_index
                        render_started = PERF_STATS.mark()
                        _render_planning_status(
                            planning_canvas, pieces, frame_index
                        )
                        if getattr(
                            cfg,
                            "LIVE_GRAYSCALE_OPERATOR_VIEW",
                            False,
                        ):
                            last_operator_view_error = (
                                _render_live_operator_view(
                                    planning_canvas,
                                    frame,
                                    pieces,
                                    a4_state,
                                    None,
                                    None,
                                    "PLANNING",
                                    True,
                                    False,
                                    error,
                                    candidate,
                                )
                            )
                        thumbnail_error = (
                            _draw_gray_work_thumbnail(
                                planning_canvas,
                                last_piece_gray,
                                last_piece_gray_frame,
                                last_piece_gray_threshold,
                                last_piece_contour_threshold,
                            )
                        )
                        if (
                            thumbnail_error
                            and thumbnail_error
                            != last_thumbnail_error
                        ):
                            print(
                                "GRAY_THUMBNAIL_ERROR,frame={},"
                                "reason={}".format(
                                    frame_index,
                                    thumbnail_error.replace(
                                        ",", ";"
                                    ),
                                )
                            )
                        last_thumbnail_error = thumbnail_error
                        PERF_STATS.add_stage(
                            "render_ms", render_started
                        )
                        PERF_STATS.increment("render_count")
                        (
                            ide_output_index,
                            last_ide_stream_error,
                        ) = _show_output_with_ide(
                            planning_canvas,
                            frame_index,
                            ide_output_index,
                            last_ide_stream_error,
                            force_ide=True,
                        )
                        last_rendered_state = (
                            "PLANNING",
                            len(pieces),
                        )
                        begin_plan_debug(
                            configured_planner, len(pieces)
                        )
                        try:
                            routing = plan_frozen_pieces(
                                pieces,
                                cfg.TARGET_RECT_SIZE_MM,
                                plan_rectangle_assembly,
                                unknown_planner,
                                allow_unknown_fallback=(
                                    cfg.ENABLE_UNKNOWN_PLANNER_FALLBACK_AFTER_FIXED_FAILURE
                                ),
                                prefer_outer_first=(
                                    prefer_unknown_planner
                                ),
                                preferred_planner_name=(
                                    configured_planner
                                ),
                            )
                        finally:
                            end_plan_debug()
                        active_plan = routing["plan"]
                        if routing["fallback_used"]:
                            print(
                                "PLANNER_FALLBACK,frame={},from=fixed_rectangle,"
                                "to={},reason={}".format(
                                    frame_index,
                                    configured_planner,
                                    routing.get(
                                        "fixed_failure_reason",
                                        "unknown",
                                    ).replace(",", ";"),
                                )
                            )
                        active_plan_key = key
                        print(
                            "PLANNING_DONE,frame={},elapsed_ms={},"
                            "valid={},mode={},nodes={}".format(
                                frame_index,
                                _ms_delta(
                                    _ms_now(), plan_start_ms
                                ),
                                int(active_plan.valid),
                                active_plan.mode,
                                active_plan.search_nodes,
                            )
                        )
                        _print_plan(
                            active_plan, pieces, frame_index
                        )
                        if active_plan.valid:
                            last_failed_plan_key = None
                            placement_monitor = PlacementMonitor(
                                pieces, active_plan
                            )
                            reference_pieces = (
                                placement_monitor.references
                            )
                            initial_total_piece_area = sum(
                                piece.area_mm2
                                for piece in reference_pieces
                            )
                            phase = "WAIT_FOR_MOTION"
                            motion_detector = _new_motion_detector(
                                a4_state.get(
                                    "divider_y_mm",
                                    cfg.DIVIDER_Y_MM,
                                )
                            )
                            placement_flow = _new_placement_flow()
                            verify_samples = []
                            verify_started_frame = None
                            placement_started_ms = _ms_now()
                            last_placement_check_ms = (
                                placement_started_ms
                            )
                            target_scanlines = {}
                            for (
                                piece_id,
                                target_polygon,
                            ) in active_plan.target_polygons.items():
                                target_scanlines[piece_id] = (
                                    build_polygon_scanlines(
                                        target_polygon,
                                        cfg.REALTIME_PIECE_WORK_WIDTH,
                                        cfg.REALTIME_PIECE_WORK_HEIGHT,
                                        sample_stride=(
                                            cfg.PLACEMENT_COVERAGE_SAMPLE_STRIDE
                                        ),
                                    )
                                )
                            before_foreground_mask = None
                            if (
                                last_piece_gray is not None
                                and last_piece_gray_threshold
                                is not None
                            ):
                                before_foreground_mask = (
                                    foreground_mask_from_gray(
                                        last_piece_gray.to_numpy_ref(),
                                        last_piece_gray_threshold,
                                    )
                                )
                            pieces = (
                                placement_monitor.visible_pieces()
                            )
                            stable = True
                            print(
                                "PLACEMENT_START,frame={},count={},"
                                "trigger=motion,next={},"
                                "verify_samples={}".format(
                                    frame_index,
                                    len(reference_pieces),
                                    placement_monitor.next_piece_id(),
                                    cfg.POST_MOTION_VERIFY_SAMPLES,
                                )
                            )
                        else:
                            last_failed_plan_key = key
                            phase = "ACQUIRE"

            elapsed = max(1, _ms_delta(_ms_now(), frame_start))
            instant_fps = 1000.0 / elapsed
            fps = (
                instant_fps
                if fps <= 0.0
                else 0.85 * fps + 0.15 * instant_fps
            )

            if (
                frame_index % cfg.A4_STATUS_PRINT_EVERY_N_FRAMES
                == 0
            ):
                if error:
                    print(
                        "DETECTION_ERROR,frame={},reason={},"
                        "pending_reason={}".format(
                            frame_index,
                            error.replace(",", ";"),
                            pending_reason,
                        )
                    )
                elif not a4_state["locked"]:
                    print(
                        "A4_SEARCH,frame={},rects={},dark_blobs={},"
                        "candidates={},valid_frames={},missed={},"
                        "divider_rescues={},rejected={}".format(
                            frame_index,
                            boundary_diagnostics.get("raw_rects", 0),
                            boundary_diagnostics.get(
                                "raw_dark_blobs", 0
                            ),
                            boundary_diagnostics.get(
                                "valid_candidates", 0
                            ),
                            a4_state["valid_frames"],
                            a4_state["missed_frames"],
                            boundary_diagnostics.get(
                                "divider_rescued_internal_edge", 0
                            ),
                            "|".join(
                                "{}:{}".format(name, count)
                                for name, count in sorted(
                                    boundary_diagnostics.get(
                                        "rejected", {}
                                    ).items()
                                )
                            )
                            or "none",
                        )
                    )
                elif not stable:
                    print(
                        "PLAN_PENDING,frame={},count={},stable=0,"
                        "stable_pieces={},reason={},expected={},"
                        "count_samples={}/{},a4=1,"
                        "a4_motion_px={:.1f},fps={:.1f}".format(
                            frame_index,
                            len(pieces),
                            sum(
                                1
                                for piece in pieces
                                if piece.stable
                            ),
                            pending_reason,
                            (
                                consensus_state["expected_count"]
                                if consensus_state[
                                    "expected_count"
                                ]
                                is not None
                                else "none"
                            ),
                            consensus_state["valid_samples"],
                            consensus_state[
                                "settle_detections"
                            ],
                            a4_state["motion_px"],
                            fps,
                        )
                    )

            lock_preview_active = (
                a4_lock_frame is not None
                and frame_index - a4_lock_frame
                < cfg.A4_LOCK_PREVIEW_HOLD_FRAMES
            )
            show_camera = (
                phase == "ACQUIRE"
                and (
                    cfg.DEBUG_SHOW_CAMERA
                    or (
                        cfg.A4_AUTO_SEARCH_PREVIEW
                        and (
                            not a4_state["locked"]
                            or lock_preview_active
                        )
                    )
                )
            )
            next_check_ms = 0
            placement_state = None
            if (
                phase in PLACEMENT_PHASES
                and placement_monitor is not None
            ):
                next_check_ms = 0
                placement_state = placement_monitor.state()
                current_render_state = (
                    "placement",
                ) + placement_ui_key(
                    phase,
                    placement_state,
                    next_check_ms,
                    cfg.UI_COUNTDOWN_REFRESH_INTERVAL_MS,
                    error,
                ) + (last_piece_gray_frame,)
            else:
                current_render_state = (
                    "camera",
                    frame_index,
                ) if show_camera else (
                    "status",
                ) + status_ui_key(
                    phase,
                    a4_state["locked"],
                    stable,
                    len(pieces),
                    active_plan is not None
                    and active_plan.valid,
                    error,
                ) + (last_piece_gray_frame,)
            current_render_state += (
                int(
                    round(
                        a4_state.get(
                            "divider_y_mm",
                            cfg.DIVIDER_Y_MM,
                        )
                        * 2.0
                    )
                ),
                int(
                    round(
                        a4_state.get(
                            "divider_slope_mm", 0.0
                        )
                        * 2.0
                    )
                ),
            )
            ui_dirty = should_render_ui(
                last_rendered_state, current_render_state
            )
            render_due = (
                (
                    ui_dirty
                    or frame_index
                    % cfg.DISPLAY_EVERY_N_FRAMES
                    == 0
                )
                if getattr(
                    cfg,
                    "LIVE_GRAYSCALE_OPERATOR_VIEW",
                    False,
                )
                else (
                    frame_index
                    % cfg.DISPLAY_EVERY_N_FRAMES
                    == 0
                    if show_camera
                    else ui_dirty
                )
            )

            if render_due:
                render_started = PERF_STATS.mark()
                if getattr(
                    cfg,
                    "LIVE_GRAYSCALE_OPERATOR_VIEW",
                    False,
                ):
                    canvas = canvases[canvas_index]
                    canvas_index = 1 - canvas_index
                    operator_error = _render_live_operator_view(
                        canvas,
                        frame,
                        pieces,
                        a4_state,
                        active_plan,
                        placement_state,
                        phase,
                        stable,
                        bool(motion_metrics.get("motion")),
                        error,
                        candidate,
                    )
                    if (
                        operator_error
                        and operator_error
                        != last_operator_view_error
                    ):
                        print(
                            "OPERATOR_VIEW_ERROR,frame={},"
                            "reason={}".format(
                                frame_index,
                                operator_error.replace(",", ";"),
                            )
                        )
                    last_operator_view_error = operator_error
                    thumbnail_error = (
                        _draw_gray_work_thumbnail(
                            canvas,
                            last_piece_gray,
                            last_piece_gray_frame,
                            last_piece_gray_threshold,
                            last_piece_contour_threshold,
                        )
                    )
                    if (
                        thumbnail_error
                        and thumbnail_error
                        != last_thumbnail_error
                    ):
                        print(
                            "GRAY_THUMBNAIL_ERROR,frame={},"
                            "reason={}".format(
                                frame_index,
                                thumbnail_error.replace(",", ";"),
                            )
                        )
                    last_thumbnail_error = thumbnail_error
                    PERF_STATS.add_stage(
                        "render_ms", render_started
                    )
                    PERF_STATS.increment("render_count")
                    (
                        ide_output_index,
                        last_ide_stream_error,
                    ) = _show_output_with_ide(
                        canvas,
                        frame_index,
                        ide_output_index,
                        last_ide_stream_error,
                        force_ide=(
                            phase == "COMPLETE"
                            or (
                                ui_dirty
                                and active_plan is not None
                                and bool(active_plan.operations)
                            )
                        ),
                    )
                elif show_camera:
                    if candidate is not None:
                        _draw_quad(
                            frame,
                            candidate["corners_px"],
                            YELLOW,
                            thickness=2,
                        )
                    if a4_state["corners_px"] is not None:
                        _draw_quad(
                            frame,
                            a4_state["corners_px"],
                            GREEN
                            if a4_state["locked"]
                            else YELLOW,
                            thickness=3,
                        )
                    _draw_piece_overlay(
                        frame,
                        pieces,
                        a4_state["corners_px"],
                    )
                    label = (
                        "A4 LOCK"
                        if a4_state["locked"]
                        else "A4 SEARCH"
                    )
                    _draw_text(
                        frame,
                        10,
                        8,
                        "{} C:{:.2f} M:{:.1f}px P:{} "
                        "FPS:{:.1f}".format(
                            label,
                            a4_state["confidence"],
                            a4_state["motion_px"],
                            len(pieces),
                            fps,
                        ),
                        GREEN if a4_state["locked"] else YELLOW,
                        2,
                    )
                    if error:
                        _draw_text(
                            frame,
                            10,
                            38,
                            str(error)[:76],
                            RED,
                        )
                    PERF_STATS.add_stage(
                        "render_ms", render_started
                    )
                    PERF_STATS.increment("render_count")
                    (
                        ide_output_index,
                        last_ide_stream_error,
                    ) = _show_output_with_ide(
                        frame,
                        frame_index,
                        ide_output_index,
                        last_ide_stream_error,
                    )
                else:
                    canvas = canvases[canvas_index]
                    canvas_index = 1 - canvas_index
                    if (
                        phase in PLACEMENT_PHASES
                        and placement_monitor is not None
                    ):
                        _render_placement_status(
                            canvas,
                            reference_pieces,
                            active_plan,
                            placement_state,
                            phase,
                            frame_index,
                            fps,
                            next_check_ms,
                            error,
                            a4_state.get(
                                "divider_y_mm",
                                cfg.DIVIDER_Y_MM,
                            ),
                        )
                    else:
                        _render_status(
                            canvas,
                            pieces,
                            stable,
                            active_plan,
                            frame_index,
                            fps,
                            error,
                            divider_y_mm=a4_state.get(
                                "divider_y_mm",
                                cfg.DIVIDER_Y_MM,
                            ),
                        )
                        _draw_text(
                            canvas,
                            520,
                            38,
                            "A4:{} C:{:.2f} M:{:.1f}px "
                            "D:{:.1f} S:{:+.1f}".format(
                                "LOCK"
                                if a4_state["locked"]
                                else "SEARCH",
                                a4_state["confidence"],
                                a4_state["motion_px"],
                                a4_state.get(
                                    "divider_y_mm",
                                    cfg.DIVIDER_Y_MM,
                                ),
                                a4_state.get(
                                    "divider_slope_mm", 0.0
                                ),
                            ),
                            GREEN
                            if a4_state["locked"]
                            else YELLOW,
                        )
                    thumbnail_error = (
                        _draw_gray_work_thumbnail(
                            canvas,
                            last_piece_gray,
                            last_piece_gray_frame,
                            last_piece_gray_threshold,
                            last_piece_contour_threshold,
                        )
                    )
                    if (
                        thumbnail_error
                        and thumbnail_error
                        != last_thumbnail_error
                    ):
                        print(
                            "GRAY_THUMBNAIL_ERROR,frame={},"
                            "reason={}".format(
                                frame_index,
                                thumbnail_error.replace(",", ";"),
                            )
                        )
                    last_thumbnail_error = thumbnail_error
                    PERF_STATS.add_stage(
                        "render_ms", render_started
                    )
                    PERF_STATS.increment("render_count")
                    (
                        ide_output_index,
                        last_ide_stream_error,
                    ) = _show_output_with_ide(
                        canvas,
                        frame_index,
                        ide_output_index,
                        last_ide_stream_error,
                        force_ide=(phase == "COMPLETE"),
                    )
                last_rendered_state = current_render_state
                if phase == "COMPLETE":
                    complete_displayed = True
            else:
                PERF_STATS.increment("skipped_render_count")

            last_stable = stable
            last_a4_locked = a4_state["locked"]
            frame_index += 1
            PERF_STATS.end_frame()
            _report_performance(frame_index)
            if frame_index % 30 == 0:
                gc.collect()
            if cfg.LOOP_IDLE_MS > 0:
                _sleep_ms(cfg.LOOP_IDLE_MS)

    except KeyboardInterrupt:
        print("STOP,reason=user,frame={}".format(frame_index))
    except BaseException as exc:
        reason = str(exc)
        if "IDE interrupt" in reason:
            print(
                "STOP,reason=ide_interrupt,frame={}".format(
                    frame_index
                )
            )
        else:
            print(
                "FATAL,frame={},reason={}".format(
                    frame_index, reason.replace(",", ";")
                )
            )
            return 1
    finally:
        if sensor is not None:
            try:
                sensor.stop()
            except Exception as exc:
                print("CLEANUP_WARNING,sensor={}".format(exc))
        try:
            Display.deinit()
        except Exception as exc:
            print("CLEANUP_WARNING,display={}".format(exc))
        try:
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            _sleep_ms(100)
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception as exc:
            print("CLEANUP_WARNING,media={}".format(exc))
        canvases = []
        gc.collect()
    return 0


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    main()
