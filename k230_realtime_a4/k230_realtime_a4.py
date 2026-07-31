#!/usr/bin/env python3
"""K230 puzzle recognition using a fixed manual A4 calibration."""

import gc
import os
import time

import image
from media.display import Display
from media.media import MediaManager

UART_COMMUNICATION_ENABLED = globals().get(
    "UART_COMMUNICATION_ENABLED", True
)
UART_IDE_DEBUG_MODE = not UART_COMMUNICATION_ENABLED
UART_IMPORT_ERROR = (
    "disabled_by_build" if UART_IDE_DEBUG_MODE else "none"
)
protocal_execute_plan = None
if UART_COMMUNICATION_ENABLED:
    try:
        from protocal_puzzle_motion import protocal_execute_plan
    except ImportError as exc:
        # The IDE single-file runner does not upload the optional UART modules.
        # Recognition and planning must remain usable without touching pins.
        UART_COMMUNICATION_ENABLED = False
        UART_IDE_DEBUG_MODE = True
        UART_IMPORT_ERROR = str(exc).replace(",", ";")

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
    polygon_overlap_area,
)
from puzzle_simulator_free_rect_planner import (
    match_fixed_figure2_piece_set,
    plan_simulator_free_rectangle,
)
from puzzle_placement import (
    clone_piece,
    final_foreground_mask_from_gray,
    final_region_white_metrics,
)
from puzzle_perf import PERF_STATS
from puzzle_realtime_state import (
    FinalCheckState,
    MotionDetector,
    PieceCountConsensus,
    a4_detection_interval,
    divider_overlay_endpoints,
    phase_allows_vision,
    operator_overlay_visibility,
    operator_status_line,
    planning_input_integrity_unless_fixed_template,
    periodic_output_due,
    recognition_gate_ready,
    should_render_ui,
    status_ui_key,
    top_right_thumbnail_rect,
    tracking_expected_piece_count,
)
from puzzle_vision import (
    background_difference_threshold,
    detect_pieces_from_canmv_image,
)
from puzzle_a4_boundary import (
    A4BoundaryTracker,
    detect_a4_boundary,
    project_a4_mm_to_frame,
)
from source_projective_piece_detector import (
    SourceProjectiveRecognition,
)


FINAL_PHASES = (
    "WAIT_FINAL_CHECK",
    "COMPLETE",
)
ACTIVE_PLANNER = "simulator_free_rect"
CYAN = (0, 255, 255)


def _source_projective_mode():
    return getattr(
        cfg, "PIECE_COORDINATE_MODE", "rectified_raster"
    ) == "source_projective"


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


def _draw_piece_overlay(
    frame, pieces, corners, label_prefix=""
):
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
            "{}{}".format(
                label_prefix, piece.piece_id or "P?"
            ),
            color,
        )


def _draw_source_raw_contours(
    frame, raw_pieces, corners, diagnostics
):
    """Draw raw boundaries from rectified or source work coordinates."""
    if corners is None or not raw_pieces:
        return
    work_width = max(
        2,
        int(
            (diagnostics or {}).get(
                "recognition_width",
                cfg.REALTIME_PIECE_WORK_WIDTH,
            )
        ),
    )
    work_height = max(
        2,
        int(
            (diagnostics or {}).get(
                "recognition_height",
                cfg.REALTIME_PIECE_WORK_HEIGHT,
            )
        ),
    )
    max_points = max(
        12,
        int(
            getattr(
                cfg, "DEBUG_RECTIFIED_RAW_MAX_POINTS", 240
            )
        ),
    )
    for piece in raw_pieces:
        boundary = piece.contour_px
        if not boundary:
            continue
        step = max(
            1,
            int(len(boundary) / float(max_points) + 0.999),
        )
        if (diagnostics or {}).get("backend") == "source_projective":
            scale_x = float(cfg.FRAME_WIDTH - 1) / float(work_width - 1)
            scale_y = float(cfg.FRAME_HEIGHT - 1) / float(work_height - 1)
            projected = [
                (float(point[0]) * scale_x, float(point[1]) * scale_y)
                for point in boundary[::step]
            ]
        else:
            projected = []
            for point in boundary[::step]:
                point_mm = (
                    float(point[0])
                    * cfg.A4_WIDTH_MM
                    / float(work_width - 1),
                    float(point[1])
                    * cfg.A4_HEIGHT_MM
                    / float(work_height - 1),
                )
                projected.append(
                    _a4_mm_to_frame(point_mm, corners)
                )
        _draw_polyline(
            frame, projected, CYAN, thickness=1
        )


def _draw_a4_operator_overlay(
    frame, corners, divider_state=None
):
    if corners is None:
        return
    _draw_quad(frame, corners, GREEN, thickness=3)
    endpoints = divider_overlay_endpoints(
        divider_state, cfg.A4_WIDTH_MM
    )
    # Never draw the nominal midpoint as if it were an observation.  Until a
    # piece-stage strip has actually been confirmed, the A4 outline is the
    # only calibration overlay shown.
    if endpoints is None:
        return
    divider_left = _a4_mm_to_frame(
        endpoints[0], corners
    )
    divider_right = _a4_mm_to_frame(
        endpoints[1], corners
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
):
    if (
        plan is None
        or corners is None
        or not plan.valid
        or not plan.target_polygons
    ):
        return
    for operation in plan.operations:
        piece_id = operation["piece_id"]
        polygon = plan.target_polygons.get(piece_id)
        if not polygon:
            continue
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
        _draw_text(
            frame,
            int(center[0]) + 4,
            int(center[1]) + 2,
            "R:{:+.1f}".format(operation["rotation_deg"]),
            color,
        )


def _operator_status_color(
    phase, plan, error, motion_active
):
    if error:
        return RED
    if motion_active:
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
    final_state,
    phase,
    stable,
    motion_active,
    error,
    candidate=None,
    divider_state=None,
    raw_pieces=None,
    piece_diagnostics=None,
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
        divider_state,
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
        _draw_piece_overlay(
            canvas,
            pieces,
            corners,
            label_prefix=(
                "S:" if phase in FINAL_PHASES else ""
            ),
        )
        if getattr(
            cfg, "DEBUG_DRAW_SOURCE_RAW_CONTOURS", False
        ):
            _draw_source_raw_contours(
                canvas,
                raw_pieces,
                corners,
                piece_diagnostics,
            )
    if visibility["targets"]:
        _draw_plan_target_overlay(
            canvas,
            plan,
            corners,
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
        error=error or base_error,
    )
    if (
        phase == "WAIT_FINAL_CHECK"
        and final_state is not None
    ):
        upper_ratio = final_state.get(
            "upper_remaining_ratio"
        )
        status = "WAIT CLEAR | LEFT:{} S:{}/{}".format(
            (
                "{:.0f}%".format(100.0 * upper_ratio)
                if upper_ratio is not None
                else "-"
            ),
            final_state.get("stable_frames", 0),
            final_state.get(
                "stable_frames_required",
                cfg.FINAL_TRIGGER_STABLE_FRAMES,
            ),
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
    raw_pieces=None,
    diagnostics=None,
):
    """Draw the exact grayscale raster consumed by piece recognition."""
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
        source_projective = (
            (diagnostics or {}).get("backend")
            == "source_projective"
        )
        if source_projective:
            for points, color, thickness in (
                (
                    (diagnostics or {}).get(
                        "a4_corners_source_px", ()
                    ),
                    GREEN,
                    2,
                ),
                (
                    (diagnostics or {}).get(
                        "source_polygon_px", ()
                    ),
                    YELLOW,
                    1,
                ),
            ):
                if points:
                    _draw_polyline(
                        canvas,
                        [
                            (
                                x + float(point[0]) * scale,
                                y + float(point[1]) * scale,
                            )
                            for point in points
                        ],
                        color,
                        thickness=thickness,
                    )
            divider_left = (diagnostics or {}).get(
                "divider_left_px"
            )
            divider_right = (diagnostics or {}).get(
                "divider_right_px"
            )
            if divider_left is not None and divider_right is not None:
                canvas.draw_line(
                    int(x + divider_left[0] * scale),
                    int(y + divider_left[1] * scale),
                    int(x + divider_right[0] * scale),
                    int(y + divider_right[1] * scale),
                    color=GRAY,
                    thickness=1,
                )
            for rect in (diagnostics or {}).get(
                "raw_blob_rects", ()
            ):
                _draw_box(
                    canvas,
                    int(x + rect[0] * scale),
                    int(y + rect[1] * scale),
                    max(1, int(rect[2] * scale)),
                    max(1, int(rect[3] * scale)),
                    WHITE,
                )
        if getattr(
            cfg, "DEBUG_DRAW_RECTIFIED_CONTOURS", False
        ):
            max_raw_points = max(
                12,
                int(
                    getattr(
                        cfg,
                        "DEBUG_RECTIFIED_RAW_MAX_POINTS",
                        240,
                    )
                ),
            )
            pixels_per_mm_x = float(gray_image.width() - 1) / cfg.A4_WIDTH_MM
            pixels_per_mm_y = float(gray_image.height() - 1) / cfg.A4_HEIGHT_MM
            work_corners = (diagnostics or {}).get(
                "a4_corners_source_px", ()
            )
            for index, piece in enumerate(raw_pieces or ()):
                raw_boundary = piece.contour_px
                if raw_boundary:
                    step = max(
                        1,
                        int(
                            len(raw_boundary)
                            / float(max_raw_points)
                            + 0.999
                        ),
                    )
                    raw_points = [
                        (
                            x + float(point[0]) * scale,
                            y + float(point[1]) * scale,
                        )
                        for point in raw_boundary[::step]
                    ]
                    _draw_polyline(
                        canvas,
                        raw_points,
                        CYAN,
                        thickness=1,
                    )
                if source_projective and work_corners:
                    fitted_source = [
                        project_a4_mm_to_frame(point, work_corners)
                        for point in piece.polygon_mm
                    ]
                    fitted_points = [
                        (x + point[0] * scale, y + point[1] * scale)
                        for point in fitted_source
                        if point is not None
                    ]
                    center_source = project_a4_mm_to_frame(
                        piece.centroid_mm, work_corners
                    )
                    center_x = (
                        x + center_source[0] * scale
                        if center_source is not None
                        else x
                    )
                    center_y = (
                        y + center_source[1] * scale
                        if center_source is not None
                        else y
                    )
                else:
                    fitted_points = [
                        (
                            x + float(point[0]) * pixels_per_mm_x * scale,
                            y + float(point[1]) * pixels_per_mm_y * scale,
                        )
                        for point in piece.polygon_mm
                    ]
                    center_x = (
                        x + piece.centroid_mm[0] * pixels_per_mm_x * scale
                    )
                    center_y = (
                        y + piece.centroid_mm[1] * pixels_per_mm_y * scale
                    )
                color = COLORS[index % len(COLORS)]
                _draw_polyline(
                    canvas,
                    fitted_points,
                    color,
                    thickness=2,
                )
                _draw_text(
                    canvas,
                    int(center_x) + 2,
                    int(center_y) - 7,
                    "{}v".format(len(piece.polygon_mm)),
                    color,
                )
        _draw_box(
            canvas,
            x - 1,
            y - 1,
            width + 2,
            height + 2,
            WHITE,
        )
        used_thresholds = (
            diagnostics.get("contour_thresholds", ())
            if diagnostics is not None
            else ()
        )
        contour_label = (
            "|".join(str(int(value)) for value in used_thresholds)
            if used_thresholds
            else (
                contour_threshold
                if contour_threshold is not None
                else "-"
            )
        )
        label = "A4 B:{} C:{} F:{}".format(
            threshold if threshold is not None else "-",
            contour_label,
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
        "lock_spread_px": 0.0,
        "calibration_generation": 1,
        "relock_required": False,
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


def _final_screen_point(point_mm):
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


def _final_phase_label(phase):
    return {
        "WAIT_FINAL_CHECK": "WAITING FOR LEFT CLEAR",
        "COMPLETE": "LEFT CLEAR - COMPLETE",
    }.get(phase, phase)


def _render_final_status(
    canvas,
    reference_pieces,
    plan,
    final_state,
    phase,
    frame_index,
    fps,
    error,
    divider_state=None,
):
    """Render every frozen S/T/R operation and whole-scene final state."""
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

    a4_top_left = _final_screen_point((0.0, 0.0))
    a4_bottom_right = _final_screen_point(
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
    divider_endpoints = divider_overlay_endpoints(
        divider_state, cfg.A4_WIDTH_MM
    )
    if divider_endpoints is not None:
        divider_a = _final_screen_point(
            divider_endpoints[0]
        )
        divider_b = _final_screen_point(
            divider_endpoints[1]
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

    for piece in reference_pieces:
        piece_id = piece.piece_id
        target = plan.target_polygons.get(piece_id)
        if not target:
            continue
        points = [
            _final_screen_point(point) for point in target
        ]
        _draw_polyline(
            canvas, points, YELLOW, thickness=2
        )
        operation = _operation_for_piece(plan, piece_id)
        if operation is not None:
            center = _final_screen_point(
                operation["target_center_mm"]
            )
            _draw_text(
                canvas,
                center[0] + 3,
                center[1] - 12,
                "T:{}".format(piece_id),
                YELLOW,
            )
            _draw_text(
                canvas,
                center[0] + 3,
                center[1] + 2,
                "R:{:+.1f}".format(
                    operation["rotation_deg"]
                ),
                YELLOW,
            )

    for piece in reference_pieces:
        color = _piece_color(piece.piece_id, reference_pieces)
        points = [
            _final_screen_point(point)
            for point in piece.polygon_mm
        ]
        _draw_polyline(canvas, points, color, thickness=2)
        center = _final_screen_point(piece.centroid_mm)
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
            "S:{}".format(piece.piece_id),
            color,
        )

    panel_x = 320
    status_color = GREEN if phase == "COMPLETE" else YELLOW
    _draw_text(
        canvas,
        panel_x,
        55,
        _final_phase_label(phase),
        status_color,
        2,
    )
    upper_ratio = final_state.get(
        "upper_remaining_ratio"
    )
    _draw_text(
        canvas,
        panel_x,
        83,
        "LEFT SOURCE:{}".format(
            (
                "{:.1f}%".format(100.0 * upper_ratio)
                if upper_ratio is not None
                else "-"
            ),
        ),
        WHITE,
    )
    if phase == "COMPLETE":
        _draw_text(
            canvas, panel_x, 115, "SOURCE CLEARED - PASS", GREEN, 2
        )
    else:
        _draw_text(
            canvas,
            panel_x,
            115,
            "STABLE:{}/{}".format(
                final_state.get("stable_frames", 0),
                final_state.get(
                    "stable_frames_required",
                    cfg.FINAL_TRIGGER_STABLE_FRAMES,
                ),
            ),
            YELLOW,
            2,
        )
    if error:
        _draw_text(
            canvas, panel_x, 178, str(error)[:48], RED
        )

    row_y = 210
    for piece in reference_pieces:
        piece_id = piece.piece_id
        operation = _operation_for_piece(plan, piece_id)
        if operation is None:
            continue
        _draw_text(
            canvas,
            panel_x,
            row_y,
            piece_id,
            _piece_color(piece_id, reference_pieces),
            1,
        )
        _draw_text(
            canvas,
            panel_x + 42,
            row_y,
            "S:{:.1f},{:.1f} T:{:.1f},{:.1f} R:{:+.1f}".format(
                operation["source_center_mm"][0],
                operation["source_center_mm"][1],
                operation["target_center_mm"][0],
                operation["target_center_mm"][1],
                operation["rotation_deg"],
            ),
            WHITE,
        )
        row_y += 35


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


def _projection_closure_diagnostics(
    pieces,
    corners_px,
    work_corners_px,
    work_width,
    work_height,
):
    """Compare full-frame projection with the rounded work-image mapping."""
    if not corners_px or not work_corners_px:
        return {"max_px": 0.0, "mean_px": 0.0, "samples": 0}
    residuals = []
    scale_x = float(cfg.FRAME_WIDTH - 1) / max(
        1.0, float(work_width - 1)
    )
    scale_y = float(cfg.FRAME_HEIGHT - 1) / max(
        1.0, float(work_height - 1)
    )
    for piece in pieces:
        for point_mm in list(piece.polygon_mm) + [piece.centroid_mm]:
            direct = project_a4_mm_to_frame(point_mm, corners_px)
            work = project_a4_mm_to_frame(
                point_mm, work_corners_px
            )
            if direct is None or work is None:
                continue
            work_in_source = (
                float(work[0]) * scale_x,
                float(work[1]) * scale_y,
            )
            residuals.append(
                ((direct[0] - work_in_source[0]) ** 2
                 + (direct[1] - work_in_source[1]) ** 2)
                ** 0.5
            )
    if not residuals:
        return {"max_px": 0.0, "mean_px": 0.0, "samples": 0}
    return {
        "max_px": max(residuals),
        "mean_px": sum(residuals) / len(residuals),
        "samples": len(residuals),
    }


def _detect_frame_pieces(
    frame,
    corners_px,
    region,
    threshold,
    divider_y_mm=None,
    collect_sanity=False,
    work_width=None,
    work_height=None,
    source_recognizer=None,
    calibration_generation=0,
    recognition_sample_id=None,
):
    if _source_projective_mode():
        work_width = int(cfg.SOURCE_PROJECTIVE_WORK_WIDTH)
        work_height = int(cfg.SOURCE_PROJECTIVE_WORK_HEIGHT)
        resize_started = PERF_STATS.mark()
        piece_gray = frame.to_grayscale(
            x_size=work_width,
            y_size=work_height,
        )
        PERF_STATS.add_stage("source_resize_ms", resize_started)
        scale_x = float(work_width - 1) / float(cfg.FRAME_WIDTH - 1)
        scale_y = float(work_height - 1) / float(cfg.FRAME_HEIGHT - 1)
        work_corners = [
            (
                float(point[0]) * scale_x,
                float(point[1]) * scale_y,
            )
            for point in corners_px
        ]
        if source_recognizer is None:
            source_recognizer = SourceProjectiveRecognition()
        pieces, diagnostics = source_recognizer.detect(
            piece_gray,
            work_corners,
            calibration_generation,
            threshold=threshold,
            collect_sanity=collect_sanity,
            sample_id=recognition_sample_id,
        )
        diagnostics["projection_closure"] = {
            "max_px": 0.0,
            "mean_px": 0.0,
            "samples": 0,
        }
        diagnostics["recognition_width"] = work_width
        diagnostics["recognition_height"] = work_height
        return pieces, diagnostics
    work_width = int(
        cfg.REALTIME_PIECE_WORK_WIDTH
        if work_width is None
        else work_width
    )
    work_height = int(
        cfg.REALTIME_PIECE_WORK_HEIGHT
        if work_height is None
        else work_height
    )
    piece_gray = frame.to_grayscale(
        x_size=work_width,
        y_size=work_height,
    )
    pieces, diagnostics = detect_pieces_from_canmv_image(
        piece_gray,
        corners_px,
        (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT),
        region=region,
        threshold=threshold,
        divider_y_mm=divider_y_mm,
        collect_sanity=collect_sanity,
    )
    diagnostics["projection_closure"] = (
        _projection_closure_diagnostics(
            pieces,
            corners_px,
            diagnostics.get("work_corners_px"),
            work_width,
            work_height,
        )
    )
    diagnostics["recognition_width"] = work_width
    diagnostics["recognition_height"] = work_height
    return pieces, diagnostics


def _transfer_piece_ids(reference_pieces, refined_pieces):
    """Transfer stable IDs to a same-frame higher-resolution observation."""
    if len(reference_pieces) != len(refined_pieces):
        return False
    remaining = list(refined_pieces)
    for reference in reference_pieces:
        if not remaining:
            return False
        best = min(
            remaining,
            key=lambda piece: (
                (piece.centroid_mm[0] - reference.centroid_mm[0]) ** 2
                + (piece.centroid_mm[1] - reference.centroid_mm[1]) ** 2
            ),
        )
        distance = (
            (best.centroid_mm[0] - reference.centroid_mm[0]) ** 2
            + (best.centroid_mm[1] - reference.centroid_mm[1]) ** 2
        ) ** 0.5
        if distance > cfg.CENTER_STABLE_TOLERANCE_MM:
            return False
        best.piece_id = reference.piece_id
        best.stable = True
        remaining.remove(best)
    refined_pieces.sort(
        key=lambda piece: int(piece.piece_id[1:])
    )
    return True


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


def _new_final_check_flow():
    return FinalCheckState(
        cfg.FINAL_TRIGGER_STABLE_FRAMES,
        cfg.FINAL_TRIGGER_UPPER_REMAINING_RATIO_MAX,
    )


def _final_scene_mask(
    gray_image,
    threshold,
    divider_y_mm,
):
    width = int(gray_image.width())
    border_px = max(
        1,
        int(
            cfg.PIECE_RECTIFIED_BORDER_BLACK_PX
            * width
            / float(cfg.REALTIME_PIECE_WORK_WIDTH)
            + 0.5
        ),
    )
    return final_foreground_mask_from_gray(
        gray_image.to_numpy_ref(),
        threshold,
        border_px=border_px,
        divider_y_mm=divider_y_mm,
        divider_margin_mm=cfg.MOTION_DIVIDER_IGNORE_MM,
    )


def _print_all_plan_operations(plan):
    """Emit the complete machine operation list once after a valid plan."""
    for operation in plan.operations:
        source = operation["source_center_mm"]
        target = operation["target_center_mm"]
        print(
            "OPERATION,piece_id={},template_role={},source_x={:.2f},"
            "source_y={:.2f},target_x={:.2f},"
            "target_y={:.2f},rotation_deg={:.2f}".format(
                operation["piece_id"],
                operation.get("template_role", "none"),
                source[0],
                source[1],
                target[0],
                target[1],
                operation["rotation_deg"],
            )
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
        native = sanity.get("native", {})
        upper = sanity.get("upper", {})
        lower = sanity.get("lower", {})
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
        "center_gray={},contour_used={},contour_mode={},"
        "projection_closure_px={:.2f}|{:.2f}|{},"
        "divider_y_mm={:.1f},divider_detected={},"
        "divider_strip={},divider_source={},divider_reason={},"
        "divider_threshold={},divider_bg={:.1f},"
        "divider_hits={}/{},divider_coverage={:.2f},"
        "divider_thickness_px={:.1f},divider_slope_mm={:.2f},"
        "divider_residual_px={:.2f},divider_mask_half_px={},"
        "divider_masked_pixels={},"
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
            "|".join(
                "{:.0f}".format(value)
                for value in diagnostics.get(
                    "center_grays", ()
                )
            )
            or "none",
            "|".join(
                str(int(value))
                for value in diagnostics.get(
                    "contour_thresholds", ()
                )
            )
            or "none",
            "|".join(
                str(value)
                for value in diagnostics.get(
                    "contour_threshold_modes", ()
                )
            )
            or "none",
            diagnostics.get("projection_closure", {}).get(
                "max_px", 0.0
            ),
            diagnostics.get("projection_closure", {}).get(
                "mean_px", 0.0
            ),
            diagnostics.get("projection_closure", {}).get(
                "samples", 0
            ),
            diagnostics.get(
                "divider_y_mm", cfg.DIVIDER_Y_MM
            ),
            int(
                diagnostics.get("divider_detected", False)
            ),
            int(
                diagnostics.get(
                    "divider_strip_detected", False
                )
            ),
            diagnostics.get("divider_source", "unknown"),
            diagnostics.get("divider_reason", "none") or "none",
            diagnostics.get("divider_threshold", 0),
            diagnostics.get("divider_background_gray", 0.0),
            diagnostics.get("divider_hits", 0),
            diagnostics.get("divider_samples", 0),
            diagnostics.get("divider_coverage", 0.0),
            diagnostics.get("divider_thickness_px", 0.0),
            diagnostics.get("divider_slope_mm", 0.0),
            diagnostics.get("divider_residual_px", 0.0),
            diagnostics.get("divider_mask_half_width_px", 0),
            diagnostics.get("divider_masked_pixels", 0),
            expected,
            samples,
        )
    )
    if diagnostics.get("backend") == "source_projective":
        map_sanity = diagnostics.get("map_sanity", {})
        corners_text = "|".join(
            "{:.1f}:{:.1f}".format(point[0], point[1])
            for point in diagnostics.get(
                "a4_corners_source_px", ()
            )
        ) or "none"
        print(
            "SOURCE_PROJECTIVE_A4,frame={},generation={},corners={},"
            "condition={:.3f},lock_spread_px={:.2f}".format(
                frame_index,
                diagnostics.get("generation", 0),
                corners_text,
                diagnostics.get("condition_metric", 0.0),
                diagnostics.get("a4_lock_spread_px", 0.0),
            )
        )
        print(
            "SOURCE_PIECE_DETECT,frame={},generation={},work={}x{},"
            "threshold={},raw_blobs={},accepted={},vertices={},"
            "areas_mm2={},rotation_corr_calls={},mask_mode={},"
            "masked_px={},mask_reused={},background_cached={},"
            "threshold_cached={}".format(
                frame_index,
                diagnostics.get("generation", 0),
                diagnostics.get("recognition_width", 0),
                diagnostics.get("recognition_height", 0),
                int(diagnostics.get("threshold", 0)),
                diagnostics.get("raw_contours", 0),
                len(pieces),
                "|".join(
                    str(len(piece.polygon_mm)) for piece in pieces
                ) or "none",
                "|".join(
                    "{:.0f}".format(piece.area_mm2)
                    for piece in pieces
                ) or "none",
                diagnostics.get("rotation_corr_calls", 0),
                diagnostics.get("source_mask_mode", "pending"),
                diagnostics.get("source_masked_pixels", 0),
                int(diagnostics.get("scanline_mask_reused", False)),
                int(diagnostics.get("background_cached", False)),
                int(diagnostics.get("threshold_cached", False)),
            )
        )
        print(
            "SOURCE_MAP_SANITY,frame={},corner_error_mm={:.6f},"
            "roundtrip_max_px={:.6f},nonfinite={},condition={:.3f}".format(
                frame_index,
                map_sanity.get("corner_error_mm", 0.0),
                map_sanity.get("roundtrip_max_px", 0.0),
                map_sanity.get("nonfinite", 0),
                diagnostics.get("condition_metric", 0.0),
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


class _CompletionLight:
    """One-shot completion indicator using the onboard WS2812 RGB LED."""

    def __init__(self):
        self.pixel = None
        self.ready = False
        self.activated = False
        self.is_on = False
        self.turned_on_ms = None

    def prepare(self):
        if not bool(
            getattr(cfg, "COMPLETION_LED_ENABLED", False)
        ):
            print("COMPLETION_LED_READY,status=DISABLED")
            return
        try:
            from machine import Pin
            import neopixel

            self.pixel = neopixel.NeoPixel(
                Pin(cfg.COMPLETION_LED_PIN), 1
            )
            self.pixel[0] = (0, 0, 0)
            self.pixel.write()
            self.ready = True
            print(
                "COMPLETION_LED_READY,status=READY,pin={},"
                "type=ws2812".format(cfg.COMPLETION_LED_PIN)
            )
        except Exception as exc:
            self.pixel = None
            self.ready = False
            print(
                "COMPLETION_LED_READY,status=UNAVAILABLE,"
                "reason={}".format(
                    str(exc).replace(",", ";")
                )
            )

    def show_complete(self, frame_index):
        if self.activated:
            return
        self.activated = True
        if not self.ready:
            print(
                "COMPLETION_LED,status=SKIPPED,frame={},"
                "reason=not_ready".format(frame_index)
            )
            return
        try:
            self.pixel[0] = tuple(cfg.COMPLETION_LED_COLOR)
            self.pixel.write()
            self.is_on = True
            self.turned_on_ms = _ms_now()
            print(
                "COMPLETION_LED,status=ON,frame={},"
                "color={}|{}|{},duration_ms={}".format(
                    frame_index,
                    cfg.COMPLETION_LED_COLOR[0],
                    cfg.COMPLETION_LED_COLOR[1],
                    cfg.COMPLETION_LED_COLOR[2],
                    cfg.COMPLETION_LED_DURATION_MS,
                )
            )
        except Exception as exc:
            print(
                "COMPLETION_LED,status=ERROR,frame={},"
                "reason={}".format(
                    frame_index,
                    str(exc).replace(",", ";"),
                )
            )

    def update(self, frame_index):
        if (
            not self.is_on
            or self.turned_on_ms is None
            or _ms_delta(_ms_now(), self.turned_on_ms)
            < cfg.COMPLETION_LED_DURATION_MS
        ):
            return
        try:
            self.pixel[0] = (0, 0, 0)
            self.pixel.write()
            print(
                "COMPLETION_LED,status=OFF,frame={},"
                "reason=duration_elapsed".format(frame_index)
            )
        except Exception as exc:
            print(
                "COMPLETION_LED,status=ERROR,frame={},"
                "reason={}".format(
                    frame_index,
                    str(exc).replace(",", ";"),
                )
            )
        self.is_on = False
        self.turned_on_ms = None

    def close(self):
        if self.pixel is not None:
            try:
                self.pixel[0] = (0, 0, 0)
                self.pixel.write()
            except Exception as exc:
                print(
                    "CLEANUP_WARNING,completion_led={}".format(
                        str(exc).replace(",", ";")
                    )
                )
        self.pixel = None
        self.ready = False
        self.is_on = False
        self.turned_on_ms = None


def main():
    sensor = None
    canvases = []
    completion_light = _CompletionLight()
    print(
        "UART_MODE,enabled={},ide_debug={},reason={}".format(
            int(UART_COMMUNICATION_ENABLED),
            int(UART_IDE_DEBUG_MODE),
            UART_IMPORT_ERROR,
        )
    )
    _audit_runtime_api()
    auto_calibrate_a4 = bool(cfg.AUTO_CALIBRATE_A4)
    source_projective = _source_projective_mode()
    freeze_source_a4 = bool(
        source_projective
        and getattr(
            cfg, "SOURCE_PROJECTIVE_FREEZE_A4_AFTER_LOCK", False
        )
    )
    boundary_tracker = (
        A4BoundaryTracker(continuous=source_projective)
        if auto_calibrate_a4
        else None
    )
    source_recognizer = (
        SourceProjectiveRecognition() if source_projective else None
    )
    a4_state = (
        boundary_tracker.state()
        if auto_calibrate_a4
        else _manual_a4_state()
    )
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
    last_raw_pieces = []
    last_piece_detection_frame = -1000000
    last_boundary_diagnostics = {}
    canvas_index = 0
    a4_lock_frame = None
    phase = "ACQUIRE"
    reference_pieces = []
    initial_total_piece_area = 0.0
    motion_detector = None
    final_check_flow = None
    motion_metrics = {
        "mean_abs_diff": 0.0,
        "changed_ratio": 0.0,
        "motion": False,
    }
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
    tracker_expected_count = tracking_expected_piece_count(
        cfg.PLANNING_REQUIRED_PIECE_COUNT,
        consensus_state,
    )
    piece_tracker = PieceTracker(tracker_expected_count)
    bad_count_detections = 0
    piece_detection_count = 0
    last_piece_diagnostic_signature = None
    last_piece_diagnostics = {}
    confirmed_divider = None
    last_input_integrity_signature = None
    pending_reason = "a4_unlocked"
    last_piece_gray = None
    last_piece_gray_frame = -1
    last_piece_gray_threshold = None
    last_piece_contour_threshold = None
    last_thumbnail_error = None
    last_operator_view_error = None
    debug_hold_announced = False
    final_refinement_done = False
    ide_output_index = 0
    last_ide_stream_error = None
    PERF_STATS.enabled = bool(cfg.ENABLE_STAGE_TIMING)
    PERF_STATS.reset()

    try:
        sensor = _init_hardware()
        completion_light.prepare()
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
            "START_REALTIME_A4,frame={}x{},camera={},boundary={}x{},"
            "a4_refine={}x{}|{},"
            "piece_work={}x{},piece_final={}x{},"
            "a4_acquire_every={},a4_locked_every={},piece_every={},"
            "debug_camera={},planner={},"
            "source_clear={:.2f}|{},a4_hold_misses={},"
            "piece_segment={},piece_deltas={}|{},"
            "fixed_fallbacks={}|{},count_window={},"
            "count_settle={},gray_thumbnail={},gray_sanity={},"
            "ide_stream=explicit,ide_quality={},"
            "ide_every={},plan_debug={},"
            "plan_debug_ms={},dynamic_divider={},"
            "piece_divider_strip={},"
            "piece_divider_gray={}|{},"
            "adaptive_contour={},adaptive_alpha={:.2f},"
            "recognition_debug_hold={},"
            "operator_view={},a4_calibration={}".format(
                cfg.FRAME_WIDTH,
                cfg.FRAME_HEIGHT,
                (
                    "grayscale"
                    if getattr(cfg, "CAMERA_GRAYSCALE", False)
                    else "rgb565"
                ),
                cfg.A4_DETECT_WIDTH,
                cfg.A4_DETECT_HEIGHT,
                cfg.A4_REFINE_WIDTH,
                cfg.A4_REFINE_HEIGHT,
                int(cfg.A4_FULL_RES_REFINE_ENABLED),
                (
                    cfg.SOURCE_PROJECTIVE_WORK_WIDTH
                    if source_projective
                    else cfg.REALTIME_PIECE_WORK_WIDTH
                ),
                (
                    cfg.SOURCE_PROJECTIVE_WORK_HEIGHT
                    if source_projective
                    else cfg.REALTIME_PIECE_WORK_HEIGHT
                ),
                (
                    cfg.SOURCE_PROJECTIVE_FINAL_WIDTH
                    if source_projective
                    else cfg.REALTIME_PIECE_FINAL_WIDTH
                ),
                (
                    cfg.SOURCE_PROJECTIVE_FINAL_HEIGHT
                    if source_projective
                    else cfg.REALTIME_PIECE_FINAL_HEIGHT
                ),
                cfg.A4_DETECT_INTERVAL_ACQUIRE,
                (
                    "off"
                    if freeze_source_a4
                    else cfg.A4_TRACK_EVERY_N_FRAMES
                    if source_projective
                    else "off"
                ),
                cfg.PIECE_DETECT_EVERY_N_FRAMES,
                int(cfg.DEBUG_SHOW_CAMERA),
                ACTIVE_PLANNER,
                cfg.FINAL_TRIGGER_UPPER_REMAINING_RATIO_MAX,
                cfg.FINAL_TRIGGER_STABLE_FRAMES,
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
                int(
                    getattr(
                        cfg,
                        "PIECE_DIVIDER_DETECTION_ENABLED",
                        False,
                    )
                ),
                getattr(cfg, "PIECE_DIVIDER_MIN_GRAY", 50),
                getattr(
                    cfg,
                    "PIECE_DIVIDER_MIN_CONTRAST_GRAY",
                    20,
                ),
                int(
                    getattr(
                        cfg,
                        "PIECE_ADAPTIVE_CONTOUR_THRESHOLD_ENABLED",
                        False,
                    )
                ),
                getattr(
                    cfg, "PIECE_CONTOUR_ADAPTIVE_ALPHA", 0.42
                ),
                int(
                    getattr(
                        cfg,
                        "DEBUG_RECOGNITION_HOLD_BEFORE_PLANNING",
                        False,
                    )
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
                    (
                        "automatic_initial_lock_frozen"
                        if freeze_source_a4
                        else "automatic_continuous_relock"
                        if source_projective
                        else "automatic_initial_lock_frozen"
                    )
                    if auto_calibrate_a4
                    else "manual_fixed"
                ),
            )
        )
        print(
            "RECOGNITION_COORDINATES,piece_coordinate_mode={},"
            "source_work={}x{},image_rectification={},"
            "coordinate_projective_mapping={},"
            "divider_detection={}".format(
                getattr(
                    cfg,
                    "PIECE_COORDINATE_MODE",
                    "rectified_raster",
                ),
                (
                    cfg.SOURCE_PROJECTIVE_WORK_WIDTH
                    if source_projective
                    else cfg.REALTIME_PIECE_WORK_WIDTH
                ),
                (
                    cfg.SOURCE_PROJECTIVE_WORK_HEIGHT
                    if source_projective
                    else cfg.REALTIME_PIECE_WORK_HEIGHT
                ),
                0 if source_projective else 1,
                1 if source_projective else 0,
                (
                    "source_midband"
                    if source_projective
                    else "rectified_strip"
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
                completion_light.update(frame_index)
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
                        a4_state["locked"],
                        (
                            cfg.A4_TRACK_EVERY_N_FRAMES
                            if source_projective
                            and not freeze_source_a4
                            else None
                        ),
                    )
                    boundary_due = (
                        boundary_interval is not None
                        and frame_index % boundary_interval == 0
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
                        boundary_diagnostics["full_refine_attempted"] = 0
                        boundary_diagnostics["full_refine_used"] = 0
                        if (
                            candidate is not None
                            and getattr(
                                cfg,
                                "A4_FULL_RES_REFINE_ENABLED",
                                False,
                            )
                        ):
                            boundary_diagnostics[
                                "full_refine_attempted"
                            ] = 1
                            refine_gray = frame.to_grayscale(
                                x_size=cfg.A4_REFINE_WIDTH,
                                y_size=cfg.A4_REFINE_HEIGHT,
                            )
                            (
                                refined_candidate,
                                refine_diagnostics,
                            ) = detect_a4_boundary(
                                refine_gray,
                                (
                                    cfg.FRAME_WIDTH,
                                    cfg.FRAME_HEIGHT,
                                ),
                            )
                            boundary_diagnostics[
                                "full_refine_candidates"
                            ] = refine_diagnostics.get(
                                "valid_candidates", 0
                            )
                            boundary_diagnostics[
                                "full_refine_rejected"
                            ] = refine_diagnostics.get(
                                "rejected", {}
                            )
                            if refined_candidate is not None:
                                refined_candidate["source"] = (
                                    "{}_full_refine".format(
                                        refined_candidate.get(
                                            "source", "unknown"
                                        )
                                    )
                                )
                                candidate = refined_candidate
                                boundary_diagnostics[
                                    "full_refine_used"
                                ] = 1
                            else:
                                # A coarse candidate is sufficient to trigger
                                # refinement, but never sufficient to define
                                # the final homography when refinement is
                                # enabled. Keep searching instead of locking
                                # coordinates known to be lower precision.
                                candidate = None
                        last_boundary_diagnostics = (
                            boundary_diagnostics
                        )
                        if (
                            phase in FINAL_PHASES
                            or (
                                source_projective
                                and a4_state.get("corners_px")
                                is not None
                            )
                        ):
                            candidate = _preserve_corner_labels(
                                candidate,
                                a4_state["corners_px"],
                            )
                        a4_state = boundary_tracker.update(
                            candidate
                        )
                        if a4_state["locked"] and (
                            not source_projective or freeze_source_a4
                        ):
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
                        not final_refinement_done
                        and frame_index
                        - last_piece_detection_frame
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
                                source_recognizer=source_recognizer,
                                calibration_generation=a4_state.get(
                                    "calibration_generation", 0
                                ),
                                recognition_sample_id=(
                                    piece_detection_count
                                ),
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
                                source_recognizer=source_recognizer,
                                calibration_generation=a4_state.get(
                                    "calibration_generation", 0
                                ),
                                recognition_sample_id=(
                                    piece_detection_count
                                ),
                            )
                            if len(retry_pieces) > len(pieces):
                                pieces = retry_pieces
                                piece_diagnostics = (
                                    retry_diagnostics
                                )
                                retry_used = True
                        piece_diagnostics["a4_lock_spread_px"] = (
                            a4_state.get("lock_spread_px", 0.0)
                        )
                        last_piece_diagnostics = piece_diagnostics
                        if piece_diagnostics.get(
                            "divider_strip_detected", False
                        ):
                            next_divider = {
                                "detected": True,
                                "divider_y_mm": float(
                                    piece_diagnostics[
                                        "divider_y_mm"
                                    ]
                                ),
                                "slope_mm": float(
                                    piece_diagnostics.get(
                                        "divider_slope_mm", 0.0
                                    )
                                ),
                                "frame": frame_index,
                                "confidence": float(
                                    piece_diagnostics.get(
                                        "divider_confidence", 0.0
                                    )
                                ),
                                "residual_px": float(
                                    piece_diagnostics.get(
                                        "divider_residual_px", 0.0
                                    )
                                ),
                            }
                            if (
                                confirmed_divider is None
                                or abs(
                                    next_divider["divider_y_mm"]
                                    - confirmed_divider[
                                        "divider_y_mm"
                                    ]
                                )
                                >= 0.25
                                or abs(
                                    next_divider["slope_mm"]
                                    - confirmed_divider["slope_mm"]
                                )
                                >= 0.25
                            ):
                                print(
                                    "DIVIDER_DISPLAY,frame={},"
                                    "detected=1,y_mm={:.2f},"
                                    "slope_mm={:.2f},source="
                                    "rectified_strip".format(
                                        frame_index,
                                        next_divider[
                                            "divider_y_mm"
                                        ],
                                        next_divider["slope_mm"],
                                    )
                                )
                            confirmed_divider = next_divider
                        elif source_projective:
                            confirmed_divider = None
                        if source_projective:
                            half_state = piece_diagnostics.get(
                                "source_half", {}
                            )
                            print(
                                "SOURCE_DIVIDER,frame={},generation={},"
                                "detected={},coverage={:.2f},contrast={:.1f},"
                                "residual_px={:.2f},mean_y_mm={:.2f},"
                                "slope_mm={:.2f},confidence={:.2f},held={},"
                                "confirmations={}/{},frozen={},detect_calls={}".format(
                                    frame_index,
                                    a4_state.get(
                                        "calibration_generation", 0
                                    ),
                                    int(
                                        piece_diagnostics.get(
                                            "divider_detected", False
                                        )
                                    ),
                                    piece_diagnostics.get(
                                        "divider_coverage", 0.0
                                    ),
                                    piece_diagnostics.get(
                                        "divider_contrast", 0.0
                                    ),
                                    piece_diagnostics.get(
                                        "divider_residual_px", 0.0
                                    ),
                                    piece_diagnostics.get(
                                        "divider_y_mm", cfg.DIVIDER_Y_MM
                                    ),
                                    piece_diagnostics.get(
                                        "divider_slope_mm", 0.0
                                    ),
                                    piece_diagnostics.get(
                                        "divider_confidence", 0.0
                                    ),
                                    int(
                                        bool(
                                            source_recognizer
                                            and source_recognizer.divider
                                            and source_recognizer.divider.get(
                                                "held", False
                                            )
                                        )
                                    ),
                                    piece_diagnostics.get(
                                        "divider_confirmations", 0
                                    ),
                                    piece_diagnostics.get(
                                        "divider_required_confirmations", 1
                                    ),
                                    int(
                                        piece_diagnostics.get(
                                            "divider_frozen", False
                                        )
                                    ),
                                    piece_diagnostics.get(
                                        "divider_detection_count", 0
                                    ),
                                )
                            )
                            print(
                                "SOURCE_HALF,side={},bright_a={},bright_b={},"
                                "confidence={:.2f},reason={},frozen={},"
                                "estimate_calls={},background_calls={},"
                                "mask_builds={}".format(
                                    piece_diagnostics.get(
                                        "source_half_selected",
                                        half_state.get("side", "none"),
                                    )
                                    or "none",
                                    half_state.get("bright_top", 0),
                                    half_state.get("bright_bottom", 0),
                                    half_state.get("confidence", 0.0),
                                    half_state.get(
                                        "reason",
                                        piece_diagnostics.get(
                                            "reason", "unknown"
                                        ),
                                    ),
                                    int(
                                        piece_diagnostics.get(
                                            "source_half_frozen", False
                                        )
                                    ),
                                    piece_diagnostics.get(
                                        "source_half_estimation_count", 0
                                    ),
                                    piece_diagnostics.get(
                                        "background_estimation_count", 0
                                    ),
                                    piece_diagnostics.get(
                                        "scanline_mask_build_count", 0
                                    ),
                                )
                            )
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
                        last_raw_pieces = raw_pieces
                        raw_piece_count = len(raw_pieces)
                        polygon_fit_incomplete = (
                            piece_diagnostics.get(
                                "rejected", {}
                            ).get("polygon", 0)
                            > 0
                        )
                        recognition_prerequisite_missing = (
                            source_projective
                            and (
                                not piece_diagnostics.get(
                                    "divider_detected", False
                                )
                                or not piece_diagnostics.get(
                                    "source_half_selected"
                                )
                                or (
                                    getattr(
                                        cfg,
                                        "SOURCE_PROJECTIVE_REQUIRED_SOURCE_HALF",
                                        None,
                                    )
                                    is not None
                                    and piece_diagnostics.get(
                                        "source_half_selected"
                                    )
                                    != getattr(
                                        cfg,
                                        "SOURCE_PROJECTIVE_REQUIRED_SOURCE_HALF",
                                        None,
                                    )
                                )
                            )
                        )
                        if (
                            not recognition_prerequisite_missing
                            and not polygon_fit_incomplete
                            and not consensus_state["ready"]
                        ):
                            consensus_state = (
                                piece_count_consensus.update(
                                    raw_piece_count
                                )
                            )
                            next_tracking_count = (
                                tracking_expected_piece_count(
                                    cfg.PLANNING_REQUIRED_PIECE_COUNT,
                                    consensus_state,
                                )
                            )
                            if (
                                next_tracking_count
                                != tracker_expected_count
                            ):
                                tracker_expected_count = (
                                    next_tracking_count
                                )
                                piece_tracker.reset(
                                    tracker_expected_count
                                )
                        if recognition_prerequisite_missing:
                            tracker_stable = False
                            stable = False
                            pieces = last_pieces
                            required_source_half = getattr(
                                cfg,
                                "SOURCE_PROJECTIVE_REQUIRED_SOURCE_HALF",
                                None,
                            )
                            if (
                                required_source_half is not None
                                and piece_diagnostics.get(
                                    "source_half_selected"
                                )
                                != required_source_half
                            ):
                                pending_reason = (
                                    "source_half_not_normalized"
                                )
                            else:
                                pending_reason = piece_diagnostics.get(
                                    "reason",
                                    "source_prerequisite_missing",
                                )
                        elif polygon_fit_incomplete:
                            tracker_stable = False
                            stable = False
                            if (
                                tracker_expected_count is not None
                                and not consensus_state["ready"]
                            ):
                                piece_tracker.reset(
                                    tracker_expected_count
                                )
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
                            elif not consensus_state["ready"]:
                                # A missing piece breaks the provisional
                                # fixed-count history. Count consensus and
                                # geometric tracking still continue together
                                # once all four pieces reappear.
                                piece_tracker.reset(
                                    tracker_expected_count
                                )
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
                                    tracker_expected_count = (
                                        tracking_expected_piece_count(
                                            cfg.PLANNING_REQUIRED_PIECE_COUNT,
                                            consensus_state,
                                        )
                                    )
                                    piece_tracker.reset(
                                        tracker_expected_count
                                    )
                                    bad_count_detections = 0
                                    pending_reason = (
                                        "count_reacquire"
                                    )
                        else:
                            bad_count_detections = 0
                            pieces, tracker_stable = (
                                piece_tracker.update(raw_pieces)
                            )
                            stable = recognition_gate_ready(
                                consensus_state, tracker_stable
                            )
                            pending_reason = (
                                "ready"
                                if stable
                                else (
                                    consensus_state["reason"]
                                    if not consensus_state["ready"]
                                    else "tracker_stabilizing"
                                )
                            )
                        if getattr(
                            cfg,
                            "DEBUG_DRAW_RECTIFIED_CONTOURS",
                            False,
                        ):
                            # In the recognition diagnostic build the main
                            # projected outline and the thumbnail must both
                            # describe this exact detection, not the tracker's
                            # modal historical representative.
                            pieces = raw_pieces
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
                            bool(
                                piece_diagnostics.get(
                                    "divider_strip_detected",
                                    False,
                                )
                            ),
                            piece_diagnostics.get(
                                "divider_source", "unknown"
                            ),
                            piece_diagnostics.get(
                                "divider_reason", ""
                            ),
                            int(
                                round(
                                    piece_diagnostics.get(
                                        "divider_coverage", 0.0
                                    )
                                    * 20.0
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
                    if (
                        stable
                        and not final_refinement_done
                        and not source_projective
                    ):
                        fixed_before_refine, _fixed_reason = (
                            match_fixed_figure2_piece_set(pieces)
                        )
                        if fixed_before_refine is not None:
                            # Fixed mode already knows the fourth contour once
                            # three pieces match.  Do not run the final contour
                            # fit again: it can collapse the real 10 mm edge on
                            # MIDDLE_LEFT and move its fitted centroid.
                            final_refinement_done = True
                            pending_reason = "fixed_template_ready"
                            matched_count = fixed_before_refine.get(
                                "matched_piece_count", 4
                            )
                            print(
                                "PIECE_REFINE,frame={},status="
                                "skipped_fixed_template,matched={}/4,"
                                "short_edge_fit_cancelled={}".format(
                                    frame_index,
                                    matched_count,
                                    int(
                                        fixed_before_refine.get(
                                            "short_edge_fit_cancelled",
                                            False,
                                        )
                                    ),
                                )
                            )
                    if (
                        stable
                        and not final_refinement_done
                        and not source_projective
                    ):
                        coarse_pieces = pieces
                        refined_pieces, refined_diagnostics = (
                            _detect_frame_pieces(
                                frame,
                                a4_state["corners_px"],
                                "upper",
                                last_piece_gray_threshold,
                                (
                                    a4_state["divider_y_mm"]
                                    if a4_state.get(
                                        "divider_detected",
                                        False,
                                    )
                                    else None
                                ),
                                collect_sanity=False,
                                work_width=(
                                    cfg.REALTIME_PIECE_FINAL_WIDTH
                                ),
                                work_height=(
                                    cfg.REALTIME_PIECE_FINAL_HEIGHT
                                ),
                                source_recognizer=source_recognizer,
                                calibration_generation=a4_state.get(
                                    "calibration_generation", 0
                                ),
                                recognition_sample_id=(
                                    piece_detection_count
                                ),
                            )
                        )
                        refined_polygon_rejected = (
                            refined_diagnostics.get(
                                "rejected", {}
                            ).get("polygon", 0)
                            > 0
                        )
                        refined_ids_ok = _transfer_piece_ids(
                            coarse_pieces, refined_pieces
                        )
                        final_refinement_done = (
                            not refined_polygon_rejected
                            and refined_ids_ok
                            and len(refined_pieces)
                            == tracker_expected_count
                        )
                        print(
                            "PIECE_REFINE,frame={},work={}x{},"
                            "coarse_count={},refined_count={},"
                            "ids={},polygon_rejected={},status={}".format(
                                frame_index,
                                cfg.REALTIME_PIECE_FINAL_WIDTH,
                                cfg.REALTIME_PIECE_FINAL_HEIGHT,
                                len(coarse_pieces),
                                len(refined_pieces),
                                int(refined_ids_ok),
                                int(refined_polygon_rejected),
                                (
                                    "accepted"
                                    if final_refinement_done
                                    else "retry_after_stability"
                                ),
                            )
                        )
                        if final_refinement_done:
                            pieces = refined_pieces
                            raw_pieces = refined_pieces
                            last_pieces = refined_pieces
                            last_raw_pieces = refined_pieces
                            last_piece_diagnostics = (
                                refined_diagnostics
                            )
                            last_piece_gray = (
                                refined_diagnostics.get("rectified")
                            )
                            last_piece_gray_frame = frame_index
                            last_piece_gray_threshold = int(
                                refined_diagnostics.get(
                                    "threshold",
                                    cfg.WHITE_GRAY_THRESHOLD,
                                )
                            )
                            last_piece_contour_threshold = int(
                                refined_diagnostics.get(
                                    "contour_threshold",
                                    last_piece_gray_threshold,
                                )
                            )
                            last_piece_stable = True
                            pending_reason = "refined_ready"
                        else:
                            pieces = coarse_pieces
                            stable = False
                            last_piece_stable = False
                            pending_reason = "refinement_failed"
                            piece_tracker.reset(
                                tracker_expected_count
                            )
                elif (
                    a4_state["locked"]
                    and phase in FINAL_PHASES
                    and final_check_flow is not None
                ):
                    pieces = reference_pieces
                    stable = True
                    if phase != "COMPLETE":
                        divider_y_mm = a4_state.get(
                            "divider_y_mm", cfg.DIVIDER_Y_MM
                        )
                        final_threshold = int(
                            last_piece_contour_threshold
                            if last_piece_contour_threshold
                            is not None
                            else cfg.PIECE_CONTOUR_MIN_GRAY_THRESHOLD
                        )
                        motion_gray = _rectified_gray(
                            frame,
                            a4_state["corners_px"],
                            cfg.MOTION_SAMPLE_WIDTH,
                            cfg.MOTION_SAMPLE_HEIGHT,
                        )
                        motion_metrics = motion_detector.update(
                            motion_gray.to_numpy_ref()
                        )
                        # The thumbnail is the exact A4 grayscale image used
                        # by the final trigger, never a display-only warp.
                        last_piece_gray = motion_gray
                        last_piece_gray_frame = frame_index
                        last_piece_gray_threshold = final_threshold
                        last_piece_contour_threshold = final_threshold
                        trigger_mask = _final_scene_mask(
                            motion_gray,
                            final_threshold,
                            divider_y_mm,
                        )
                        trigger_regions = final_region_white_metrics(
                            trigger_mask,
                            cfg.MOTION_SAMPLE_WIDTH,
                            cfg.MOTION_SAMPLE_HEIGHT,
                            initial_total_piece_area,
                            divider_y_mm=divider_y_mm,
                        )
                        if phase == "WAIT_FINAL_CHECK":
                            final_state = final_check_flow.update(
                                motion_metrics["motion"],
                                trigger_regions[
                                    "upper_remaining_ratio"
                                ],
                                trigger_regions["lower_area_ratio"],
                            )
                            phase = final_state["phase"]
                            if final_state["trigger_complete"]:
                                print(
                                    "FINAL_RESULT,status=PASS,frame={},"
                                    "reason=source_clear,"
                                    "upper_remaining_ratio={:.3f},"
                                    "lower_area_ratio={:.3f},"
                                    "stable_frames={}/{},"
                                    "mean_diff={:.2f},"
                                    "changed_ratio={:.3f}".format(
                                        frame_index,
                                        final_state[
                                            "upper_remaining_ratio"
                                        ],
                                        final_state[
                                            "lower_area_ratio"
                                        ],
                                        final_state["stable_frames"],
                                        final_state[
                                            "stable_frames_required"
                                        ],
                                        motion_metrics["mean_abs_diff"],
                                        motion_metrics["changed_ratio"],
                                    )
                                )
                                completion_light.show_complete(
                                    frame_index
                                )
                elif phase == "ACQUIRE":
                    pieces = []
                    final_refinement_done = False
                    last_piece_stable = False
                    last_piece_detection_frame = -1000000
                    pending_reason = "a4_unlocked"
                    last_piece_gray = None
                    last_piece_gray_frame = -1
                    last_piece_gray_threshold = None
                    last_piece_contour_threshold = None
                    last_raw_pieces = []
                elif final_check_flow is not None:
                    # A frozen plan never returns to piece detection.
                    pieces = reference_pieces
                    stable = True
            except Exception as exc:
                if "IDE interrupt" in str(exc):
                    raise
                error = str(exc)
                if auto_calibrate_a4:
                    a4_state = boundary_tracker.state()
                if (
                    phase in FINAL_PHASES
                    and final_check_flow is not None
                ):
                    pieces = reference_pieces
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
                confirmed_divider = None
                if last_a4_locked and phase == "ACQUIRE":
                    final_refinement_done = False
                    piece_count_consensus.reset()
                    consensus_state = (
                        piece_count_consensus.state()
                    )
                    tracker_expected_count = (
                        tracking_expected_piece_count(
                            cfg.PLANNING_REQUIRED_PIECE_COUNT,
                            consensus_state,
                        )
                    )
                    piece_tracker.reset(tracker_expected_count)
                    bad_count_detections = 0
                    piece_detection_count = 0
                    last_piece_diagnostic_signature = None
                    last_piece_diagnostics = {}
                    last_input_integrity_signature = None
                    active_plan = None
                    active_plan_key = None
                    last_failed_plan_key = None
                    if source_recognizer is not None:
                        source_recognizer.reset(
                            a4_state.get(
                                "calibration_generation", 0
                            )
                        )
                    last_pieces = []
                    last_raw_pieces = []
                    last_piece_gray = None
                    last_piece_gray_frame = -1
                    confirmed_divider = None
                    debug_hold_announced = False
                    pending_reason = "a4_relock"
                    print(
                        "A4_RELOCK_REQUIRED,frame={},generation={},"
                        "tracker_reset=1,count_consensus_reset=1".format(
                            frame_index,
                            a4_state.get(
                                "calibration_generation", 0
                            ),
                        )
                    )

            if phase == "ACQUIRE":
                debug_hold_active = (
                    bool(
                        getattr(
                            cfg,
                            "DEBUG_RECOGNITION_HOLD_BEFORE_PLANNING",
                            False,
                        )
                    )
                    and stable
                    and not error
                    and a4_state["locked"]
                )
                if debug_hold_active:
                    active_plan = None
                    active_plan_key = None
                    pending_reason = "recognition_debug_hold"
                    if not debug_hold_announced:
                        closure = last_piece_diagnostics.get(
                            "projection_closure", {}
                        )
                        print(
                            "RECOGNITION_DEBUG_HOLD,frame={},"
                            "phase=ACQUIRE,stable=1,count={},"
                            "raw=cyan,fitted=piece_color,"
                            "projection_closure_px={:.2f}|{:.2f}".format(
                                frame_index,
                                len(pieces),
                                closure.get("max_px", 0.0),
                                closure.get("mean_px", 0.0),
                            )
                        )
                        debug_hold_announced = True
                elif (
                    not stable
                    or error
                    or not a4_state["locked"]
                ):
                    active_plan = None
                    active_plan_key = None
                else:
                    key = _plan_key(pieces)
                    fixed_template_evaluation = (
                        match_fixed_figure2_piece_set(pieces)
                    )
                    fixed_template_match = (
                        fixed_template_evaluation[0] is not None
                    )
                    input_integrity = (
                        planning_input_integrity_unless_fixed_template(
                            fixed_template_match,
                            pieces,
                            None,
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
                    )
                    integrity_signature = (
                        ("fixed_figure2_direct",)
                        if fixed_template_match
                        else (
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
                    )
                    if (
                        input_integrity is not None
                        and not input_integrity["valid"]
                    ):
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
                        configured_planner = ACTIVE_PLANNER
                        if fixed_template_match:
                            print(
                                "PLANNING_INPUT_BYPASS,frame={},"
                                "reason=fixed_figure2_direct,"
                                "border_blobs={}".format(
                                    frame_index,
                                    last_piece_diagnostics.get(
                                        "rejected", {}
                                    ).get("border", 0),
                                )
                            )
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
                                    confirmed_divider,
                                    last_raw_pieces,
                                    last_piece_diagnostics,
                                )
                            )
                        thumbnail_error = (
                            _draw_gray_work_thumbnail(
                                planning_canvas,
                                last_piece_gray,
                                last_piece_gray_frame,
                                last_piece_gray_threshold,
                                last_piece_contour_threshold,
                                last_raw_pieces,
                                last_piece_diagnostics,
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
                            active_plan = (
                                plan_simulator_free_rectangle(
                                    pieces,
                                    fixed_template_evaluation=(
                                        fixed_template_evaluation
                                    ),
                                )
                            )
                        finally:
                            end_plan_debug()
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
                            reference_pieces = [
                                clone_piece(piece) for piece in pieces
                            ]
                            initial_total_piece_area = sum(
                                piece.area_mm2
                                for piece in reference_pieces
                            )
                            _print_all_plan_operations(active_plan)
                            if UART_COMMUNICATION_ENABLED:
                                protocal_execute_plan(active_plan)
                            else:
                                print(
                                    "PROTOCAL_EXECUTION_SKIPPED,"
                                    "reason={}".format(
                                        UART_IMPORT_ERROR
                                    )
                                )
                            phase = "WAIT_FINAL_CHECK"
                            motion_detector = _new_motion_detector(
                                a4_state.get(
                                    "divider_y_mm",
                                    cfg.DIVIDER_Y_MM,
                                )
                            )
                            final_check_flow = (
                                _new_final_check_flow()
                            )
                            pieces = reference_pieces
                            stable = True
                            print(
                                "FINAL_CHECK_START,frame={},count={},"
                                "initial_area_mm2={:.1f},"
                                "source_clear_max={:.2f},"
                                "stable_frames={}".format(
                                    frame_index,
                                    len(reference_pieces),
                                    initial_total_piece_area,
                                    cfg.FINAL_TRIGGER_UPPER_REMAINING_RATIO_MAX,
                                    cfg.FINAL_TRIGGER_STABLE_FRAMES,
                                )
                            )
                        else:
                            last_failed_plan_key = key
                            phase = "ACQUIRE"
                            final_refinement_done = False

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
                        "candidates={},full_refine={}|{},"
                        "valid_frames={},missed={},"
                        "divider_rescues={},blob_corners={}|{},"
                        "blob_fallbacks={},blob_attempt_errors={},"
                        "rect_error={},blob_error={},"
                        "rejected={},candidate_detail={}".format(
                            frame_index,
                            boundary_diagnostics.get("raw_rects", 0),
                            boundary_diagnostics.get(
                                "raw_dark_blobs", 0
                            ),
                            boundary_diagnostics.get(
                                "valid_candidates", 0
                            ),
                            boundary_diagnostics.get(
                                "full_refine_attempted", 0
                            ),
                            boundary_diagnostics.get(
                                "full_refine_used", 0
                            ),
                            a4_state["valid_frames"],
                            a4_state["missed_frames"],
                            boundary_diagnostics.get(
                                "divider_rescued_internal_edge", 0
                            ),
                            "|".join(
                                str(value)
                                for value in boundary_diagnostics.get(
                                    "dark_blob_contour_corner_counts", ()
                                )
                            )
                            or "none",
                            "|".join(
                                str(value)
                                for value in boundary_diagnostics.get(
                                    "dark_blob_min_corner_counts", ()
                                )
                            )
                            or "none",
                            "|".join(
                                boundary_diagnostics.get(
                                    "dark_blob_fallback_sources", ()
                                )
                            )
                            or "none",
                            "|".join(
                                boundary_diagnostics.get(
                                    "dark_blob_attempt_errors", ()
                                )
                            )
                            or "none",
                            boundary_diagnostics.get("rect_error", "")
                            or "none",
                            boundary_diagnostics.get("blob_error", "")
                            or "none",
                            "|".join(
                                "{}:{}".format(name, count)
                                for name, count in sorted(
                                    boundary_diagnostics.get(
                                        "rejected", {}
                                    ).items()
                                )
                            )
                            or "none",
                            "|".join(
                                boundary_diagnostics.get(
                                    "candidate_rejections", ()
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
            final_state = None
            if (
                phase in FINAL_PHASES
                and final_check_flow is not None
            ):
                final_state = final_check_flow.state()
                current_render_state = (
                    "placement",
                    phase,
                    final_state["stable_frames"],
                    int(
                        1000.0
                        * (
                            final_state[
                                "upper_remaining_ratio"
                            ]
                            if final_state[
                                "upper_remaining_ratio"
                            ]
                            is not None
                            else -1.0
                        )
                    ),
                    str(error or ""),
                    last_piece_gray_frame,
                )
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
                (
                    int(
                        round(
                            confirmed_divider[
                                "divider_y_mm"
                            ]
                            * 2.0
                        )
                    )
                    if confirmed_divider is not None
                    else -1
                ),
                (
                    int(
                        round(
                            confirmed_divider["slope_mm"]
                            * 2.0
                        )
                    )
                    if confirmed_divider is not None
                    else 0
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
                        final_state,
                        phase,
                        stable,
                        bool(motion_metrics.get("motion")),
                        error,
                        candidate,
                        confirmed_divider,
                        last_raw_pieces,
                        last_piece_diagnostics,
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
                            last_raw_pieces,
                            last_piece_diagnostics,
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
                        phase in FINAL_PHASES
                        and final_check_flow is not None
                    ):
                        _render_final_status(
                            canvas,
                            reference_pieces,
                            active_plan,
                            final_state,
                            phase,
                            frame_index,
                            fps,
                            error,
                            confirmed_divider,
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
                            divider_y_mm=(
                                confirmed_divider[
                                    "divider_y_mm"
                                ]
                                if confirmed_divider is not None
                                else None
                            ),
                        )
                        divider_status = (
                            "D:{:.1f} S:{:+.1f}".format(
                                confirmed_divider[
                                    "divider_y_mm"
                                ],
                                confirmed_divider["slope_mm"],
                            )
                            if confirmed_divider is not None
                            else "D:- S:-"
                        )
                        _draw_text(
                            canvas,
                            520,
                            38,
                            "A4:{} C:{:.2f} M:{:.1f}px {}".format(
                                "LOCK"
                                if a4_state["locked"]
                                else "SEARCH",
                                a4_state["confidence"],
                                a4_state["motion_px"],
                                divider_status,
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
                            last_raw_pieces,
                            last_piece_diagnostics,
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
        completion_light.close()
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
