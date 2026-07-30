#!/usr/bin/env python3
"""CanMV K230 standalone A4 recognition test entrypoint.

This entrypoint intentionally runs only the current A4 boundary detector and
tracker.  It does not import piece recognition, geometry planning, or placement
code.  Hardware imports stay inside ``main`` so desktop tests can import the
small formatting and drawing helpers.
"""

import gc
import os
import time

import puzzle_config as cfg
from puzzle_a4_boundary import A4BoundaryTracker, detect_a4_boundary


# This test is automatic-only: it never loads configured A4 corner points.
# Keep detecting after the initial three-frame lock so moving or replacing the
# sheet automatically updates/reacquires the calibration.
AUTO_CALIBRATE_A4 = True
FREEZE_AFTER_LOCK = False
DETECT_EVERY_N_FRAMES = 1

WHITE = (235, 235, 235)
GRAY = (150, 150, 150)
RED = (255, 65, 65)
GREEN = (70, 240, 100)
YELLOW = (255, 210, 40)
CYAN = (80, 220, 255)


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


def _format_corners(corners):
    if not corners:
        return "none"
    return "|".join(
        "{:.0f}:{:.0f}".format(point[0], point[1])
        for point in corners
    )


def _format_rejections(diagnostics):
    rejected = diagnostics.get("rejected", {})
    if not rejected:
        return "none"
    return "|".join(
        "{}:{}".format(name, count)
        for name, count in sorted(rejected.items())
    )


def _divider_points(corners, divider_y_mm):
    """Return the two display endpoints of the A4 horizontal divider."""
    if corners is None or len(corners) != 4:
        return None
    fraction = max(
        0.0,
        min(1.0, float(divider_y_mm) / float(cfg.A4_HEIGHT_MM)),
    )
    top_left, top_right, bottom_right, bottom_left = corners
    return (
        (
            top_left[0] + fraction * (bottom_left[0] - top_left[0]),
            top_left[1] + fraction * (bottom_left[1] - top_left[1]),
        ),
        (
            top_right[0] + fraction * (bottom_right[0] - top_right[0]),
            top_right[1] + fraction * (bottom_right[1] - top_right[1]),
        ),
    )


def _draw_text(frame, x, y, text, color=WHITE, size=18):
    try:
        frame.draw_string_advanced(
            int(x),
            int(y),
            int(size),
            str(text),
            color=color,
        )
    except Exception:
        try:
            frame.draw_string(int(x), int(y), str(text), color=color)
        except Exception:
            pass


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


def _draw_calibration(frame, candidate, state):
    if candidate is not None:
        _draw_quad(frame, candidate["corners_px"], YELLOW, thickness=2)

    corners = state.get("corners_px")
    if corners is None:
        return
    color = GREEN if state["locked"] else YELLOW
    _draw_quad(frame, corners, color, thickness=3)

    divider = _divider_points(
        corners,
        state.get("divider_y_mm", cfg.DIVIDER_Y_MM),
    )
    if divider is not None:
        frame.draw_line(
            int(divider[0][0]),
            int(divider[0][1]),
            int(divider[1][0]),
            int(divider[1][1]),
            color=CYAN,
            thickness=2,
        )

    for label, point in zip(("TL", "TR", "BR", "BL"), corners):
        x = int(point[0])
        y = int(point[1])
        try:
            frame.draw_cross(x, y, color=color, size=8, thickness=2)
        except Exception:
            pass
        _draw_text(
            frame,
            max(0, min(frame.width() - 30, x + 6)),
            max(42, min(frame.height() - 22, y - 18)),
            label,
            color,
            16,
        )


def _print_status(
    frame_index,
    state,
    candidate,
    diagnostics,
    fps,
    event="STATUS",
):
    print(
        "A4_TEST_{},frame={},locked={},frozen={},candidate={},"
        "confidence={:.2f},valid_frames={},missed_frames={},"
        "motion_px={:.1f},source={},orientation={},divider={},"
        "divider_y_mm={:.1f},divider_slope_mm={:.1f},"
        "corners={},rects={},dark_blobs={},valid_candidates={},"
        "rejected={},fps={:.1f}".format(
            event,
            frame_index,
            int(state.get("locked", False)),
            int(state.get("frozen", False)),
            int(candidate is not None),
            state.get("confidence", 0.0),
            state.get("valid_frames", 0),
            state.get("missed_frames", 0),
            state.get("motion_px", 0.0),
            state.get("source", ""),
            state.get("orientation", ""),
            int(state.get("divider_detected", False)),
            state.get("divider_y_mm", cfg.DIVIDER_Y_MM),
            state.get("divider_slope_mm", 0.0),
            _format_corners(state.get("corners_px")),
            diagnostics.get("raw_rects", 0),
            diagnostics.get("raw_dark_blobs", 0),
            diagnostics.get("valid_candidates", 0),
            _format_rejections(diagnostics),
            fps,
        )
    )


def _stop_requested(start_ms, frame_index):
    if cfg.AUTO_STOP_SECONDS > 0:
        if _ms_delta(_ms_now(), start_ms) >= cfg.AUTO_STOP_SECONDS * 1000:
            return True
    if cfg.MAX_FRAME_COUNT > 0 and frame_index >= cfg.MAX_FRAME_COUNT:
        return True
    return False


def main():
    from media.display import Display
    from media.media import MediaManager
    from media.sensor import Sensor

    sensor = None
    tracker = A4BoundaryTracker()
    state = tracker.state()
    diagnostics = {}
    candidate = None
    frame_index = 0
    fps = 0.0
    start_ms = _ms_now()
    last_locked = False

    try:
        sensor = Sensor()
        sensor.reset()
        sensor.set_hmirror(True)
        sensor.set_vflip(True)
        sensor.set_framesize(
            width=cfg.FRAME_WIDTH,
            height=cfg.FRAME_HEIGHT,
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

        print(
            "A4_TEST_START,frame={}x{},detect={}x{},"
            "auto_calibrate={},lock_frames={},freeze_after_lock={},"
            "bundle={}".format(
                cfg.FRAME_WIDTH,
                cfg.FRAME_HEIGHT,
                cfg.A4_DETECT_WIDTH,
                cfg.A4_DETECT_HEIGHT,
                int(AUTO_CALIBRATE_A4),
                cfg.A4_LOCK_REQUIRED_FRAMES,
                int(FREEZE_AFTER_LOCK),
                "single"
                if getattr(cfg, "STANDALONE_BUILD", False)
                else "modules",
            )
        )

        while True:
            os.exitpoint()
            if _stop_requested(start_ms, frame_index):
                print(
                    "A4_TEST_STOP,reason=configured_limit,frame={}".format(
                        frame_index
                    )
                )
                break

            frame_start = _ms_now()
            frame = sensor.snapshot()
            if frame is None:
                state = tracker.update(None)
                if frame_index % cfg.A4_STATUS_PRINT_EVERY_N_FRAMES == 0:
                    print(
                        "A4_TEST_ERROR,frame={},reason=snapshot_none".format(
                            frame_index
                        )
                    )
                frame_index += 1
                continue

            error = None
            try:
                should_detect = (
                    not state.get("frozen", False)
                    and frame_index % max(1, DETECT_EVERY_N_FRAMES) == 0
                )
                if should_detect:
                    boundary_gray = frame.to_grayscale(
                        x_size=cfg.A4_DETECT_WIDTH,
                        y_size=cfg.A4_DETECT_HEIGHT,
                    )
                    if boundary_gray is None:
                        raise RuntimeError("to_grayscale returned None")
                    candidate, diagnostics = detect_a4_boundary(
                        boundary_gray,
                        (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT),
                    )
                    state = tracker.update(candidate)
                    if (
                        FREEZE_AFTER_LOCK
                        and state["locked"]
                        and not state.get("frozen", False)
                    ):
                        state = tracker.freeze()
            except Exception as exc:
                if "IDE interrupt" in str(exc):
                    raise
                error = str(exc)
                state = tracker.state()

            elapsed = max(1, _ms_delta(_ms_now(), frame_start))
            instant_fps = 1000.0 / elapsed
            fps = (
                instant_fps
                if fps <= 0.0
                else 0.85 * fps + 0.15 * instant_fps
            )

            if state["locked"] and not last_locked:
                _print_status(
                    frame_index,
                    state,
                    candidate,
                    diagnostics,
                    fps,
                    event="LOCK",
                )
            elif not state["locked"] and last_locked:
                _print_status(
                    frame_index,
                    state,
                    candidate,
                    diagnostics,
                    fps,
                    event="LOST",
                )
            elif frame_index % cfg.A4_STATUS_PRINT_EVERY_N_FRAMES == 0:
                _print_status(
                    frame_index,
                    state,
                    candidate,
                    diagnostics,
                    fps,
                )

            _draw_calibration(frame, candidate, state)
            status_color = GREEN if state["locked"] else YELLOW
            _draw_text(
                frame,
                8,
                6,
                "{}  C:{:.2f}  F:{}  FPS:{:.1f}".format(
                    "A4 LOCK" if state["locked"] else "A4 SEARCH",
                    state.get("confidence", 0.0),
                    state.get("valid_frames", 0),
                    fps,
                ),
                status_color,
                22,
            )
            if error:
                _draw_text(
                    frame,
                    8,
                    34,
                    "ERROR: {}".format(error[:68]),
                    RED,
                    16,
                )
            else:
                _draw_text(
                    frame,
                    8,
                    34,
                    "R:{} B:{} V:{} D:{} Y:{:.1f}".format(
                        diagnostics.get("raw_rects", 0),
                        diagnostics.get("raw_dark_blobs", 0),
                        diagnostics.get("valid_candidates", 0),
                        int(state.get("divider_detected", False)),
                        state.get("divider_y_mm", cfg.DIVIDER_Y_MM),
                    ),
                    GRAY,
                    16,
                )
            Display.show_image(frame)

            last_locked = state["locked"]
            frame_index += 1
            if frame_index % 30 == 0:
                gc.collect()
            if cfg.LOOP_IDLE_MS > 0:
                _sleep_ms(cfg.LOOP_IDLE_MS)

    except KeyboardInterrupt:
        print("A4_TEST_STOP,reason=user,frame={}".format(frame_index))
    except BaseException as exc:
        reason = str(exc)
        if "IDE interrupt" in reason:
            print(
                "A4_TEST_STOP,reason=ide_interrupt,frame={}".format(
                    frame_index
                )
            )
        else:
            print(
                "A4_TEST_FATAL,frame={},reason={}".format(
                    frame_index,
                    reason.replace(",", ";"),
                )
            )
            return 1
    finally:
        if sensor is not None:
            try:
                sensor.stop()
            except Exception as exc:
                print("A4_TEST_CLEANUP_WARNING,sensor={}".format(exc))
        try:
            Display.deinit()
        except Exception as exc:
            print("A4_TEST_CLEANUP_WARNING,display={}".format(exc))
        try:
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            _sleep_ms(100)
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception as exc:
            print("A4_TEST_CLEANUP_WARNING,media={}".format(exc))
        gc.collect()
    return 0


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    main()
