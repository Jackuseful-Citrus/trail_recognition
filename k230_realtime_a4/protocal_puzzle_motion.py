"""Convert a frozen puzzle plan into simple blocking UART step commands."""

import math
import time

import protocal_uart2_config as protocal_cfg
from protocal_uart2 import (
    K230Uart2Link,
    PROTOCAL_ACTION_GRIP,
    PROTOCAL_ACTION_MOVE_X_ABS,
    PROTOCAL_ACTION_MOVE_Y_ABS,
    PROTOCAL_ACTION_ROTATE_REL,
)


class ProtocalMotionError(Exception):
    """The fixed calibration or generated motion program is invalid."""


def _sleep_ms(delay_ms):
    delay_ms = max(0, int(delay_ms))
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(delay_ms)
    else:
        time.sleep(delay_ms / 1000.0)


def _is_finite(value):
    try:
        return math.isfinite(value)
    except AttributeError:
        return value == value and abs(value) != float("inf")


def protocal_a4_to_machine(point_mm):
    if point_mm is None or len(point_mm) < 2:
        raise ProtocalMotionError("A4 point must contain x and y")
    a4_x = float(point_mm[0])
    a4_y = float(point_mm[1])
    if not _is_finite(a4_x) or not _is_finite(a4_y):
        raise ProtocalMotionError("A4 point must be finite")
    machine_x = (
        protocal_cfg.PROTOCAL_A4_TO_MACHINE_XX * a4_x
        + protocal_cfg.PROTOCAL_A4_TO_MACHINE_XY * a4_y
        + protocal_cfg.PROTOCAL_A4_TO_MACHINE_X_OFFSET_MM
    )
    machine_y = (
        protocal_cfg.PROTOCAL_A4_TO_MACHINE_YX * a4_x
        + protocal_cfg.PROTOCAL_A4_TO_MACHINE_YY * a4_y
        + protocal_cfg.PROTOCAL_A4_TO_MACHINE_Y_OFFSET_MM
    )
    if not _is_finite(machine_x) or not _is_finite(machine_y):
        raise ProtocalMotionError(
            "machine coordinate transform produced a non-finite value"
        )
    return float(machine_x), float(machine_y)


def _check_machine_point(point_mm, label):
    x_mm, y_mm = point_mm
    if not (
        protocal_cfg.PROTOCAL_MACHINE_X_MIN_MM
        <= x_mm
        <= protocal_cfg.PROTOCAL_MACHINE_X_MAX_MM
    ):
        raise ProtocalMotionError(
            "{} x={:.3f} outside [{:.3f}, {:.3f}]".format(
                label,
                x_mm,
                protocal_cfg.PROTOCAL_MACHINE_X_MIN_MM,
                protocal_cfg.PROTOCAL_MACHINE_X_MAX_MM,
            )
        )
    if not (
        protocal_cfg.PROTOCAL_MACHINE_Y_MIN_MM
        <= y_mm
        <= protocal_cfg.PROTOCAL_MACHINE_Y_MAX_MM
    ):
        raise ProtocalMotionError(
            "{} y={:.3f} outside [{:.3f}, {:.3f}]".format(
                label,
                y_mm,
                protocal_cfg.PROTOCAL_MACHINE_Y_MIN_MM,
                protocal_cfg.PROTOCAL_MACHINE_Y_MAX_MM,
            )
        )


def protocal_build_motion_program(plan):
    """Return the complete per-piece program after one full preflight."""
    if plan is None or not bool(getattr(plan, "valid", False)):
        raise ProtocalMotionError("a valid frozen plan is required")
    operations = list(getattr(plan, "operations", ()))
    if not operations:
        raise ProtocalMotionError("the plan contains no operations")

    program = []
    for index, operation in enumerate(operations):
        piece_id = operation.get(
            "piece_id", "P{}".format(index + 1)
        )
        source = protocal_a4_to_machine(
            operation["source_center_mm"]
        )
        target = protocal_a4_to_machine(
            operation["target_center_mm"]
        )
        _check_machine_point(source, "{} source".format(piece_id))
        _check_machine_point(target, "{} target".format(piece_id))

        rotation_deg = (
            float(operation["rotation_deg"])
            * float(protocal_cfg.PROTOCAL_ROTATION_SIGN)
        )
        if not _is_finite(rotation_deg):
            raise ProtocalMotionError(
                "{} rotation is not finite".format(piece_id)
            )
        if not (
            protocal_cfg.PROTOCAL_ROTATION_MIN_DEG
            <= rotation_deg
            <= protocal_cfg.PROTOCAL_ROTATION_MAX_DEG
        ):
            raise ProtocalMotionError(
                "{} rotation={:.3f} outside [{:.3f}, {:.3f}]".format(
                    piece_id,
                    rotation_deg,
                    protocal_cfg.PROTOCAL_ROTATION_MIN_DEG,
                    protocal_cfg.PROTOCAL_ROTATION_MAX_DEG,
                )
            )

        program.extend(
            (
                (
                    piece_id,
                    "source_x",
                    PROTOCAL_ACTION_MOVE_X_ABS,
                    source[0],
                ),
                (
                    piece_id,
                    "source_y",
                    PROTOCAL_ACTION_MOVE_Y_ABS,
                    source[1],
                ),
                (piece_id, "grip", PROTOCAL_ACTION_GRIP, 1.0),
            )
        )
        if abs(rotation_deg) > float(
            protocal_cfg.PROTOCAL_ROTATION_SKIP_EPSILON_DEG
        ):
            program.append(
                (
                    piece_id,
                    "rotate",
                    PROTOCAL_ACTION_ROTATE_REL,
                    rotation_deg,
                )
            )
        program.extend(
            (
                (
                    piece_id,
                    "target_x",
                    PROTOCAL_ACTION_MOVE_X_ABS,
                    target[0],
                ),
                (
                    piece_id,
                    "target_y",
                    PROTOCAL_ACTION_MOVE_Y_ABS,
                    target[1],
                ),
                (piece_id, "release", PROTOCAL_ACTION_GRIP, 0.0),
            )
        )
    return program


def _new_link():
    return K230Uart2Link(
        tx_pin=protocal_cfg.PROTOCAL_UART2_TX_PIN,
        rx_pin=protocal_cfg.PROTOCAL_UART2_RX_PIN,
        baudrate=protocal_cfg.PROTOCAL_UART_BAUDRATE,
        initial_sequence=protocal_cfg.PROTOCAL_INITIAL_SEQUENCE,
        status_timeout_ms=(
            protocal_cfg.PROTOCAL_STATUS_TIMEOUT_MS
        ),
        poll_delay_ms=protocal_cfg.PROTOCAL_POLL_DELAY_MS,
    )


def protocal_execute_motion_program(program, link=None):
    """Execute a preflighted program sequentially and block until complete."""
    commands = list(program)
    if not commands:
        raise ProtocalMotionError("motion program is empty")
    owns_link = link is None
    if link is None:
        link = _new_link()

    print(
        "PROTOCAL_EXECUTION_START,commands={},pieces={}".format(
            len(commands),
            len(set(command[0] for command in commands)),
        )
    )
    try:
        link.open()
        for index, command in enumerate(commands):
            piece_id, name, action, value = command
            print(
                "PROTOCAL_STEP,index={}/{},piece_id={},name={},"
                "action={},value={:.3f}".format(
                    index + 1,
                    len(commands),
                    piece_id,
                    name,
                    action,
                    value,
                )
            )
            link.send_step_and_wait(action, value)
        _sleep_ms(protocal_cfg.PROTOCAL_POST_PLAN_SETTLE_MS)
        print(
            "PROTOCAL_EXECUTION_DONE,commands={},tx_frames={},"
            "status_frames={}".format(
                len(commands),
                getattr(link, "tx_frames", len(commands)),
                getattr(link, "status_frames", 0),
            )
        )
        return {
            "executed": True,
            "commands": len(commands),
        }
    finally:
        if owns_link:
            link.close()


def protocal_execute_plan(plan, link=None):
    """Execute a valid plan when enabled, otherwise preserve manual mode."""
    if not bool(protocal_cfg.PROTOCAL_EXECUTION_ENABLED):
        print(
            "PROTOCAL_EXECUTION_SKIPPED,reason=disabled,"
            "config=protocal_uart2_config.py"
        )
        return {"executed": False, "commands": 0}
    program = protocal_build_motion_program(plan)
    return protocal_execute_motion_program(program, link=link)
