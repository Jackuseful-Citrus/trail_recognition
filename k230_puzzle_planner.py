#!/usr/bin/env python3
"""CanMV K230 v1.6 automatic-A4 puzzle detector and geometry planner.

The board finds and locks the black A4 work surface at startup, then switches
from the acquisition preview to the normal status/result canvas automatically.
"""

import gc
import os
import time

import image
from media.display import Display
from media.media import MediaManager
from media.sensor import Sensor

import puzzle_config as cfg
from puzzle_a4_boundary import (
    A4BoundaryTracker,
    detect_a4_boundary,
)
from puzzle_geometry import (
    PieceTracker,
    begin_plan_debug,
    end_plan_debug,
    plan_rectangle_assembly,
)
from puzzle_vision import detect_pieces_from_canmv_image


COLORS = [
    (255, 210, 40),
    (70, 240, 100),
    (80, 170, 255),
    (230, 90, 230),
]
WHITE = (235, 235, 235)
GRAY = (115, 115, 115)
RED = (255, 65, 65)
GREEN = (70, 240, 100)
YELLOW = (255, 210, 40)
BLACK = (0, 0, 0)


def _ms_now():
    if hasattr(time, "ticks_ms"):
        return int(time.ticks_ms())
    return int(time.time() * 1000)


def _ms_delta(newer, older):
    if hasattr(time, "ticks_diff"):
        return int(time.ticks_diff(newer, older))
    if newer >= older:
        return newer - older
    return max(1, newer)


def _sleep_ms(milliseconds):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(milliseconds)
    else:
        time.sleep(milliseconds / 1000.0)


def _audit_runtime_api():
    if not hasattr(image, "Image"):
        raise RuntimeError("CanMV image.Image is unavailable")
    # CanMV's native-bound Image type does not expose its instance methods on
    # the class object, so class-level hasattr() produces false negatives.
    # The actual frame/working-image calls below are the authoritative check.
    print(
        "RUNTIME,firmware=CanMV-v1.6,backend=image-native,cv2=0,"
        "find_blobs=1,rotation_corr=1,to_numpy_ref=1,"
        "canvas=image.Image-RGB565,text=draw_string_advanced,"
        "api_check=call-time,bundle={}".format(
            "single"
            if getattr(cfg, "STANDALONE_BUILD", False)
            else "modules"
        )
    )


def _init_hardware():
    """Initialize in the proven test_camera.py order."""
    sensor = Sensor()
    sensor.reset()
    sensor.set_hmirror(True)
    sensor.set_vflip(True)
    sensor.set_framesize(
        width=cfg.FRAME_WIDTH, height=cfg.FRAME_HEIGHT
    )
    sensor.set_pixformat(Sensor.RGB565)

    Display.init(
        Display.ST7701,
        width=cfg.FRAME_WIDTH,
        height=cfg.FRAME_HEIGHT,
        to_ide=True,
    )
    MediaManager.init()
    sensor.run()
    return sensor


def _stop_requested(start_ms, frame_index):
    if cfg.AUTO_STOP_SECONDS > 0:
        if _ms_delta(_ms_now(), start_ms) >= cfg.AUTO_STOP_SECONDS * 1000:
            return True
    if cfg.MAX_FRAME_COUNT > 0 and frame_index >= cfg.MAX_FRAME_COUNT:
        return True
    return False


def _draw_text(canvas, x, y, text, color=WHITE, scale=1):
    canvas.draw_string_advanced(
        int(x),
        int(y),
        max(12, int(12 * scale)),
        str(text),
        color=color,
    )


def _draw_box(canvas, x, y, width, height, color=GRAY, thickness=1):
    canvas.draw_line(x, y, x + width, y, color=color, thickness=thickness)
    canvas.draw_line(
        x + width,
        y,
        x + width,
        y + height,
        color=color,
        thickness=thickness,
    )
    canvas.draw_line(
        x + width,
        y + height,
        x,
        y + height,
        color=color,
        thickness=thickness,
    )
    canvas.draw_line(x, y + height, x, y, color=color, thickness=thickness)


def _draw_polyline(canvas, points, color, thickness=2):
    if len(points) < 2:
        return
    for index in range(len(points)):
        a = points[index]
        b = points[(index + 1) % len(points)]
        canvas.draw_line(
            int(a[0]),
            int(a[1]),
            int(b[0]),
            int(b[1]),
            color=color,
            thickness=thickness,
        )


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


def _print_a4_lock(frame_index, state):
    corners_text = "|".join(
        "{:.0f}:{:.0f}".format(point[0], point[1])
        for point in state["corners_px"]
    )
    print(
        "A4_LOCK,frame={},source={},confidence={:.2f},"
        "orientation={},corners={}".format(
            frame_index,
            state["source"],
            state["confidence"],
            state.get("orientation", ""),
            corners_text,
        )
    )


def _source_screen_point(point_mm):
    origin_x = 18
    origin_y = 44
    scale = 1.40
    return (
        int(origin_x + point_mm[0] * scale),
        int(origin_y + point_mm[1] * scale),
    )


def _target_screen_point(point_mm, divider_y_mm=None):
    origin_x = 350
    origin_y = 44
    scale = 1.10
    divider = (
        cfg.DIVIDER_Y_MM
        if divider_y_mm is None
        else float(divider_y_mm)
    )
    return (
        int(origin_x + point_mm[0] * scale),
        int(origin_y + (point_mm[1] - divider) * scale),
    )


def _render_status(
    canvas,
    pieces,
    stable,
    plan,
    frame_index,
    fps,
    error,
    calibration_state=None,
    divider_y_mm=None,
):
    canvas.clear()
    _draw_text(canvas, 12, 10, "K230 A4 PUZZLE V1", WHITE, 2)
    _draw_text(
        canvas,
        575,
        12,
        "F:{} FPS:{:.1f}".format(frame_index, fps),
        GRAY,
    )

    # Current upper-half A4 schematic.
    source_width = int(cfg.A4_WIDTH_MM * 1.40)
    divider = (
        cfg.DIVIDER_Y_MM
        if divider_y_mm is None
        else float(divider_y_mm)
    )
    source_height = int(divider * 1.40)
    _draw_box(canvas, 18, 44, source_width, source_height, GRAY)
    _draw_text(canvas, 18, 263, "CURRENT / mm", GRAY)
    for index, piece in enumerate(pieces):
        color = COLORS[index % len(COLORS)]
        points = [
            _source_screen_point(point)
            for point in piece.polygon_mm
        ]
        _draw_polyline(canvas, points, color)
        center = _source_screen_point(piece.centroid_mm)
        canvas.draw_cross(
            center[0],
            center[1],
            size=6,
            thickness=2,
            color=color,
        )
        _draw_text(
            canvas,
            center[0] + 5,
            center[1] - 12,
            piece.piece_id,
            color,
        )

    # Target lower-half A4 schematic.
    target_width = int(cfg.A4_WIDTH_MM * 1.10)
    target_height = int(
        (cfg.A4_HEIGHT_MM - divider) * 1.10
    )
    _draw_box(canvas, 350, 44, target_width, target_height, GRAY)
    _draw_text(canvas, 350, 225, "TARGET / mm", GRAY)
    if plan is not None and plan.valid:
        min_x, min_y, max_x, max_y = plan.target_rect
        rect_a = _target_screen_point(
            (min_x, min_y), divider
        )
        rect_b = _target_screen_point(
            (max_x, max_y), divider
        )
        _draw_box(
            canvas,
            rect_a[0],
            rect_a[1],
            rect_b[0] - rect_a[0],
            rect_b[1] - rect_a[1],
            GRAY,
        )
        for index, piece in enumerate(pieces):
            color = COLORS[index % len(COLORS)]
            polygon = plan.target_polygons.get(piece.piece_id)
            if polygon:
                points = [
                    _target_screen_point(point, divider)
                    for point in polygon
                ]
                _draw_polyline(canvas, points, color)
                operation = None
                for item in plan.operations:
                    if item["piece_id"] == piece.piece_id:
                        operation = item
                        break
                if operation is not None:
                    centre = _target_screen_point(
                        operation["target_center_mm"],
                        divider,
                    )
                    canvas.draw_cross(
                        centre[0],
                        centre[1],
                        size=5,
                        thickness=2,
                        color=color,
                    )
                    _draw_text(
                        canvas,
                        centre[0] + 4,
                        centre[1] - 11,
                        piece.piece_id,
                        color,
                    )

    # State and operation table.
    if error:
        status = "DETECTION ERROR"
        status_color = RED
    elif calibration_state == "search":
        status = "A4 AUTO CALIBRATION"
        status_color = YELLOW
    elif not pieces:
        status = "NO PIECES DETECTED"
        status_color = YELLOW
    elif not stable:
        status = "UNSTABLE / MOVING"
        status_color = YELLOW
    elif plan is None:
        status = "DETECTING..."
        status_color = YELLOW
    elif not plan.valid:
        status = "NO VALID PLAN"
        status_color = RED
    else:
        status = "{} PLAN {:.4f} VMAX:{:.1f}".format(
            plan.mode.upper(),
            plan.score,
            plan.max_vertex_error_mm or 0.0,
        )
        status_color = GREEN
    _draw_text(canvas, 18, 292, status, status_color, 2)
    _draw_text(
        canvas,
        18,
        320,
        "PIECES:{} stable:{} frame:{}".format(
            len(pieces), int(stable), frame_index
        ),
        WHITE,
    )
    if error:
        _draw_text(canvas, 18, 340, str(error)[:72], RED)
    elif plan is not None and not plan.valid:
        _draw_text(canvas, 18, 340, plan.reason[:72], RED)

    if plan is not None and plan.valid:
        for index, operation in enumerate(plan.operations):
            x = 350 + (index % 2) * 220
            y = 275 + (index // 2) * 95
            color = COLORS[index % len(COLORS)]
            source = operation["source_center_mm"]
            target = operation["target_center_mm"]
            _draw_text(canvas, x, y, operation["piece_id"], color, 2)
            _draw_text(
                canvas,
                x,
                y + 22,
                "S:{:.1f},{:.1f}".format(source[0], source[1]),
                WHITE,
            )
            _draw_text(
                canvas,
                x,
                y + 39,
                "T:{:.1f},{:.1f}".format(target[0], target[1]),
                WHITE,
            )
            ambiguity = "*" if operation["rotation_ambiguous"] else ""
            _draw_text(
                canvas,
                x,
                y + 56,
                "R:{:+.1f}{} deg".format(
                    operation["rotation_deg"], ambiguity
                ),
                color,
            )
    _draw_text(
        canvas,
        18,
        457,
        "* rotation has equivalent symmetric solution",
        GRAY,
    )


def _print_plan(plan, pieces, frame_index):
    stats = plan.plan_stats
    if stats:
        print(
            "PLAN_PERF,frame={},time_ms={},nodes={},edge_pairs={},"
            "filtered={},intersections={},aabb_rejects={},"
            "rect_hypotheses={},input_area_mm2={},"
            "area_scale={}".format(
                frame_index,
                stats.get("plan_ms", 0),
                stats.get(
                    "dfs_nodes_expanded", plan.search_nodes
                ),
                stats.get("candidate_pair_count_raw", 0),
                stats.get(
                    "candidate_pair_count_filtered", 0
                ),
                stats.get("polygon_intersection_calls", 0),
                stats.get("aabb_reject_count", 0),
                stats.get(
                    "rectangle_hypothesis_count",
                    stats.get(
                        "corner_rectangle_hypothesis_count",
                        0,
                    ),
                ),
                (
                    "{:.1f}".format(
                        stats["input_piece_area_mm2"]
                    )
                    if stats.get("input_piece_area_mm2")
                    is not None
                    else "na"
                ),
                (
                    "{:.4f}".format(
                        stats["target_area_scale"]
                    )
                    if stats.get("target_area_scale")
                    is not None
                    else "na"
                ),
            )
        )
        if str(stats.get("engine", "")).startswith(
            "lvreng/puzzle-vision-simulator"
        ):
            print(
                "SIMULATOR_PLAN_PERF,frame={},cut_mode={},"
                "validation={},candidates={},full={},partial={},"
                "sets={},selected={},selected_partial={},"
                "limit_hit={},timed_out={},actual_size={}x{},"
                "dimension_error_mm={},local_gate_failures={}".format(
                    frame_index,
                    stats.get("cut_mode", "auto"),
                    stats.get("validation", "local"),
                    stats.get("candidate_count", 0),
                    stats.get("full_candidate_count", 0),
                    stats.get("partial_candidate_count", 0),
                    stats.get("matching_sets_evaluated", 0),
                    stats.get("selected_match_count", 0),
                    stats.get("selected_partial_match_count", 0),
                    int(bool(stats.get("limit_hit"))),
                    int(bool(stats.get("timed_out"))),
                    (
                        "{:.1f}".format(stats["actual_width_mm"])
                        if stats.get("actual_width_mm") is not None
                        else "na"
                    ),
                    (
                        "{:.1f}".format(stats["actual_height_mm"])
                        if stats.get("actual_height_mm") is not None
                        else "na"
                    ),
                    (
                        "{:.1f}".format(stats["dimension_error_mm"])
                        if stats.get("dimension_error_mm") is not None
                        else "na"
                    ),
                    "|".join(stats.get("local_gate_failures", ()))
                    or "none",
                )
            )
    if not plan.valid:
        print(
            "PLAN_INVALID,frame={},reason={}".format(
                frame_index, plan.reason.replace(",", ";")
            )
        )
        if (
            stats
            and stats.get("candidate_pair_count_raw")
            is not None
        ):
            if stats.get("limit_hit"):
                failure_class = "search_limit"
            elif stats.get("pruned_target_dimension", 0):
                failure_class = "target_geometry"
            elif stats.get("complete_state_count", 0):
                failure_class = "rectangle_gate"
            else:
                failure_class = "seam_connectivity"

            def stat_float(key):
                value = stats.get(key)
                return (
                    "{:.1f}".format(value)
                    if value is not None
                    else "na"
                )

            print(
                "PLAN_FAIL_DETAIL,frame={},class={},complete={},"
                "max_depth={},seam_pairs={},rect_range_reject={},"
                "input_area_mm2={},target_area_mm2={},"
                "area_error_pct={},area_scale={},"
                "target_size_reject={},closest_size={}x{},"
                "size_error_mm={},closest_gap_mm2={},"
                "pruned_boundary={},pruned_dimension={},"
                "corner_reason={},corner_nodes={},corner_depth={},"
                "corner_complete={},corner_overlap_reject={}".format(
                    frame_index,
                    failure_class,
                    stats.get("complete_state_count", 0),
                    stats.get("max_depth", 0),
                    stats.get(
                        "candidate_pair_count_filtered", 0
                    ),
                    stats.get("pruned_rect_range", 0),
                    stat_float("input_piece_area_mm2"),
                    stat_float("target_area_mm2"),
                    stat_float("input_area_error_pct"),
                    (
                        "{:.4f}".format(
                            stats["target_area_scale"]
                        )
                        if stats.get("target_area_scale")
                        is not None
                        else "na"
                    ),
                    stats.get(
                        "pruned_target_dimension", 0
                    ),
                    stat_float("closest_target_long_mm"),
                    stat_float("closest_target_short_mm"),
                    stat_float(
                        "closest_target_dimension_error_mm"
                    ),
                    stat_float("closest_target_gap_mm2"),
                    stats.get("pruned_boundary", 0),
                    stats.get("pruned_dimension", 0),
                    str(
                        stats.get(
                            "corner_failure_reason", "not_attempted"
                        )
                    ).replace(",", ";"),
                    stats.get("corner_search_nodes", 0),
                    stats.get("corner_max_depth", 0),
                    stats.get("corner_complete_state_count", 0),
                    stats.get("corner_pruned_overlap", 0),
                )
            )
        elif str(stats.get("engine", "")).startswith(
            "lvreng/puzzle-vision-simulator"
        ):
            if stats.get("timed_out") or stats.get("limit_hit"):
                failure_class = "search_limit"
            elif not stats.get("candidate_count"):
                failure_class = "no_edge_candidates"
            elif not stats.get("selected_match_count"):
                failure_class = "no_connected_topology"
            else:
                failure_class = "local_geometry_gate"
            print(
                "PLAN_FAIL_DETAIL,frame={},class={},candidates={},"
                "full={},partial={},sets={},selected={},"
                "selected_partial={},actual_size={}x{},"
                "dimension_error_mm={},local_gate_failures={}".format(
                    frame_index,
                    failure_class,
                    stats.get("candidate_count", 0),
                    stats.get("full_candidate_count", 0),
                    stats.get("partial_candidate_count", 0),
                    stats.get("matching_sets_evaluated", 0),
                    stats.get("selected_match_count", 0),
                    stats.get("selected_partial_match_count", 0),
                    stats.get("actual_width_mm", "na"),
                    stats.get("actual_height_mm", "na"),
                    stats.get("dimension_error_mm", "na"),
                    "|".join(stats.get("local_gate_failures", ()))
                    or "none",
                )
            )
        for piece in pieces:
            vertices = "|".join(
                "{:.1f}:{:.1f}".format(point[0], point[1])
                for point in piece.polygon_mm
            )
            print(
                "PLAN_INPUT,frame={},id={},area_mm2={:.1f},"
                "vertices={}".format(
                    frame_index,
                    piece.piece_id,
                    piece.area_mm2,
                    vertices,
                )
            )
        return
    target_width = (
        plan.target_rect[2] - plan.target_rect[0]
    )
    target_height = (
        plan.target_rect[3] - plan.target_rect[1]
    )
    print(
        "PLAN,frame={},stable=1,count={},mode={},score={:.4f},"
        "max_vertex_error_mm={:.1f},gap_mm2={:.1f},overlap_mm2={:.1f},"
        "outside_mm2={:.1f},target_w_mm={:.1f},target_h_mm={:.1f},"
        "nodes={}".format(
            frame_index,
            len(pieces),
            plan.mode,
            plan.score,
            plan.max_vertex_error_mm or 0.0,
            plan.fill_gap_mm2 or 0.0,
            plan.overlap_mm2 or 0.0,
            plan.outside_mm2 or 0.0,
            target_width,
            target_height,
            plan.search_nodes,
        )
    )
    for operation in plan.operations:
        source = operation["source_center_mm"]
        target = operation["target_center_mm"]
        print(
            "PIECE,id={},sx_mm={:.1f},sy_mm={:.1f},tx_mm={:.1f},"
            "ty_mm={:.1f},rot_deg={:.1f},ambiguous={},confidence={:.2f}".format(
                operation["piece_id"],
                source[0],
                source[1],
                target[0],
                target[1],
                operation["rotation_deg"],
                int(operation["rotation_ambiguous"]),
                operation["confidence"],
            )
        )
    print("PLAN_END")


def _plan_key(pieces):
    values = []
    for piece in pieces:
        values.append(
            (
                piece.piece_id,
                int(round(piece.centroid_mm[0] * 2.0)),
                int(round(piece.centroid_mm[1] * 2.0)),
                int(round(piece.current_orientation_deg * 2.0)),
                int(round(piece.area_mm2 * 2.0)),
                tuple(
                    (
                        int(round(point[0] * 2.0)),
                        int(round(point[1] * 2.0)),
                    )
                    for point in piece.polygon_mm
                ),
            )
        )
    return tuple(values)


def main():
    sensor = None
    canvases = []
    _audit_runtime_api()
    tracker = PieceTracker()
    boundary_tracker = A4BoundaryTracker()
    frame_index = 0
    fps = 0.0
    start_ms = _ms_now()
    active_plan = None
    active_plan_key = None
    last_pieces = []
    last_stable = False
    last_a4_locked = not cfg.AUTO_CALIBRATE_A4
    canvas_index = 0
    a4_lock_frame = None
    a4_state = {
        "corners_px": (
            None
            if cfg.AUTO_CALIBRATE_A4
            else list(cfg.A4_CORNERS_PX)
        ),
        "locked": not cfg.AUTO_CALIBRATE_A4,
        "confidence": 1.0 if not cfg.AUTO_CALIBRATE_A4 else 0.0,
        "source": "manual" if not cfg.AUTO_CALIBRATE_A4 else "",
        "orientation": "manual" if not cfg.AUTO_CALIBRATE_A4 else "",
        "valid_frames": 0,
        "missed_frames": 0,
        "motion_px": 0.0,
    }

    try:
        sensor = _init_hardware()
        # Automatic acquisition begins with camera preview and transitions to
        # these complete status canvases after A4 lock.
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
            "START,frame_width={},frame_height={},work={}x{},"
            "backend=image-native,debug_camera={},auto_a4={},"
            "plan_debug={},plan_debug_ms={}".format(
                cfg.FRAME_WIDTH,
                cfg.FRAME_HEIGHT,
                cfg.CANMV_WORK_WIDTH,
                cfg.CANMV_WORK_HEIGHT,
                int(cfg.DEBUG_SHOW_CAMERA),
                int(cfg.AUTO_CALIBRATE_A4),
                int(cfg.ENABLE_PLAN_DEBUG),
                cfg.PLAN_DEBUG_INTERVAL_MS,
            )
        )

        while True:
            os.exitpoint()
            if _stop_requested(start_ms, frame_index):
                print("STOP,reason=configured_limit,frame={}".format(frame_index))
                break
            frame_start = _ms_now()
            frame = sensor.snapshot()
            if frame is None:
                if frame_index % cfg.PENDING_PRINT_EVERY_N_FRAMES == 0:
                    print(
                        "DETECTION_ERROR,frame={},reason=snapshot_none".format(
                            frame_index
                        )
                    )
                frame_index += 1
                continue

            error = None
            pieces = []
            stable = False
            candidate = None
            boundary_diagnostics = {}
            try:
                if cfg.AUTO_CALIBRATE_A4 and not a4_state["locked"]:
                    boundary_gray = frame.to_grayscale(
                        x_size=cfg.A4_DETECT_WIDTH,
                        y_size=cfg.A4_DETECT_HEIGHT,
                    )
                    candidate, boundary_diagnostics = (
                        detect_a4_boundary(
                            boundary_gray,
                            (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT),
                        )
                    )
                    a4_state = boundary_tracker.update(candidate)

                if a4_state["locked"]:
                    gray_image = frame.to_grayscale(
                        x_size=cfg.CANMV_WORK_WIDTH,
                        y_size=cfg.CANMV_WORK_HEIGHT,
                    )
                    if gray_image is None:
                        raise RuntimeError(
                            "to_grayscale returned None"
                        )
                    pieces, diagnostics = (
                        detect_pieces_from_canmv_image(
                            gray_image,
                            a4_state["corners_px"],
                            (
                                cfg.FRAME_WIDTH,
                                cfg.FRAME_HEIGHT,
                            ),
                        )
                    )
                    pieces, stable = tracker.update(pieces)
                    last_pieces = pieces
                else:
                    tracker.update([])
                    pieces = []
                    last_pieces = []
            except Exception as exc:
                if "IDE interrupt" in str(exc):
                    raise
                error = str(exc)
                pieces = (
                    last_pieces if a4_state["locked"] else []
                )
                stable = False

            if a4_state["locked"] and not last_a4_locked:
                a4_lock_frame = frame_index
                _print_a4_lock(frame_index, a4_state)
            elif not a4_state["locked"]:
                a4_lock_frame = None

            # Invalidate stale instructions immediately on motion/error.
            if not stable or error or not a4_state["locked"]:
                active_plan = None
                active_plan_key = None
            else:
                key = _plan_key(pieces)
                # Replan on the unstable->stable transition. Tracker motion
                # invalidates the plan immediately; sub-threshold jitter must
                # not rerun the beam search every frame.
                if not last_stable or active_plan is None:
                    begin_plan_debug(
                        "fixed_rectangle"
                        if cfg.TARGET_RECT_SIZE_MM is not None
                        else "outer_first",
                        len(pieces),
                    )
                    try:
                        active_plan = plan_rectangle_assembly(
                            pieces
                        )
                    finally:
                        end_plan_debug()
                    active_plan_key = key
                    _print_plan(active_plan, pieces, frame_index)

            elapsed = max(1, _ms_delta(_ms_now(), frame_start))
            instant_fps = 1000.0 / elapsed
            fps = instant_fps if fps <= 0.0 else 0.85 * fps + 0.15 * instant_fps

            if error and frame_index % cfg.PENDING_PRINT_EVERY_N_FRAMES == 0:
                print(
                    "DETECTION_ERROR,frame={},reason={}".format(
                        frame_index, error.replace(",", ";")
                    )
                )
            elif (
                not a4_state["locked"]
                and frame_index
                % cfg.A4_STATUS_PRINT_EVERY_N_FRAMES
                == 0
            ):
                print(
                    "A4_SEARCH,frame={},rects={},dark_blobs={},"
                    "candidates={},valid_frames={},rejected={}".format(
                        frame_index,
                        boundary_diagnostics.get("raw_rects", 0),
                        boundary_diagnostics.get(
                            "raw_dark_blobs", 0
                        ),
                        boundary_diagnostics.get(
                            "valid_candidates", 0
                        ),
                        a4_state["valid_frames"],
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
            elif (
                not stable
                and frame_index % cfg.PENDING_PRINT_EVERY_N_FRAMES == 0
            ):
                print(
                    "PLAN_PENDING,frame={},count={},stable=0,"
                    "a4={}".format(
                        frame_index,
                        len(pieces),
                        int(a4_state["locked"]),
                    )
                )

            if frame_index % cfg.DISPLAY_EVERY_N_FRAMES == 0:
                lock_preview_active = (
                    a4_lock_frame is not None
                    and frame_index - a4_lock_frame
                    < cfg.A4_LOCK_PREVIEW_HOLD_FRAMES
                )
                show_camera = (
                    cfg.DEBUG_SHOW_CAMERA
                    or (
                        cfg.A4_AUTO_SEARCH_PREVIEW
                        and (
                            not a4_state["locked"]
                            or lock_preview_active
                        )
                    )
                )
                if show_camera:
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
                    _draw_text(
                        frame,
                        10,
                        8,
                        "{} C:{:.2f} F:{}".format(
                            "A4 LOCK"
                            if a4_state["locked"]
                            else "A4 SEARCH",
                            a4_state["confidence"],
                            a4_state["valid_frames"],
                        ),
                        GREEN
                        if a4_state["locked"]
                        else YELLOW,
                        2,
                    )
                    Display.show_image(frame)
                else:
                    canvas = canvases[canvas_index]
                    canvas_index = 1 - canvas_index
                    _render_status(
                        canvas,
                        pieces,
                        stable,
                        active_plan,
                        frame_index,
                        fps,
                        error,
                        calibration_state=(
                            None
                            if a4_state["locked"]
                            else "search"
                        ),
                    )
                    Display.show_image(canvas)

            last_stable = stable
            last_a4_locked = a4_state["locked"]
            frame_index += 1
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
