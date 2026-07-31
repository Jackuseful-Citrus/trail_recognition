"""Desktop validation for the minimal K230 UART2 protocol modules."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "k230_realtime_a4"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import protocal_uart2_config as protocal_cfg
from protocal_puzzle_motion import (
    protocal_build_motion_program,
    protocal_execute_motion_program,
)
from protocal_uart2 import (
    K230Uart2Link,
    PROTOCAL_ACTION_GRIP,
    PROTOCAL_ACTION_MOVE_X_ABS,
    PROTOCAL_ACTION_MOVE_Y_ABS,
    PROTOCAL_ACTION_ROTATE_REL,
    PROTOCAL_CMD_STEP_STATUS,
    PROTOCAL_STATUS_BUSY,
    PROTOCAL_STATUS_DONE,
    ProtocalFrameParser,
    protocal_build_frame,
    protocal_build_step_frame,
    protocal_crc16_ccitt_false,
)


def _status_frame(sequence, action, status, error_code=0):
    payload = bytes(
        (
            1,
            sequence,
            action,
            status,
            error_code,
            0,
            0,
            0,
        )
    )
    return protocal_build_frame(PROTOCAL_CMD_STEP_STATUS, payload)


class _FakeUart:
    def __init__(self):
        self.writes = []
        self.reads = []
        self.deinitialized = False

    def write(self, frame):
        frame = bytes(frame)
        self.writes.append(frame)
        sequence = frame[5]
        action = frame[6]
        replies = (
            _status_frame(sequence, action, PROTOCAL_STATUS_BUSY)
            + _status_frame(sequence, action, PROTOCAL_STATUS_DONE)
        )
        self.reads.extend(
            (replies[:5], replies[5:19], replies[19:])
        )
        return len(frame)

    def read(self):
        return self.reads.pop(0) if self.reads else None

    def deinit(self):
        self.deinitialized = True


class _FakeLink:
    def __init__(self):
        self.commands = []
        self.opened = False
        self.closed = False
        self.tx_frames = 0
        self.status_frames = 0

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def send_step_and_wait(self, action, value):
        self.commands.append((action, value))
        self.tx_frames += 1
        self.status_frames += 2


class _Plan:
    valid = True

    def __init__(self):
        self.operations = [
            {
                "piece_id": "P1",
                "source_center_mm": (10.0, 20.0),
                "target_center_mm": (30.0, 40.0),
                "rotation_deg": -25.0,
            }
        ]


class ProtocalUart2Tests(unittest.TestCase):
    def test_generated_runtime_treats_uart_modules_as_optional(self):
        source = (
            MODULE_DIR
            / "k230_realtime_a4_simulator_free_rect_standalone.py"
        ).read_text(encoding="utf-8")
        self.assertIn("except ImportError as exc:", source)
        self.assertIn("UART_COMMUNICATION_ENABLED = False", source)
        self.assertIn("UART_IDE_DEBUG_MODE = True", source)
        self.assertIn("if UART_COMMUNICATION_ENABLED:", source)
        self.assertIn(
            "reason=ide_debug_import_failed", source
        )

    def test_crc_and_documented_step_vector(self):
        body = bytes.fromhex(
            "10 08 01 01 01 01 00 00 C8 42"
        )
        self.assertEqual(protocal_crc16_ccitt_false(body), 0xBB68)
        self.assertEqual(
            protocal_build_step_frame(1, 1, 100.0),
            bytes.fromhex(
                "55 AA 10 08 01 01 01 01 00 00 C8 42 68 BB"
            ),
        )

    def test_parser_handles_noise_fragmentation_and_concatenation(self):
        busy = _status_frame(7, 3, PROTOCAL_STATUS_BUSY)
        done = _status_frame(7, 3, PROTOCAL_STATUS_DONE)
        parser = ProtocalFrameParser()
        self.assertEqual(parser.feed(b"noise\x55"), [])
        self.assertEqual(parser.feed(busy[1:8]), [])
        frames = parser.feed(busy[8:] + done)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][0], PROTOCAL_CMD_STEP_STATUS)
        self.assertEqual(frames[0][1][3], PROTOCAL_STATUS_BUSY)
        self.assertEqual(frames[1][1][3], PROTOCAL_STATUS_DONE)

    def test_link_waits_for_matching_busy_and_done(self):
        uart = _FakeUart()
        link = K230Uart2Link(
            uart=uart,
            initial_sequence=0,
            status_timeout_ms=100,
            poll_delay_ms=0,
        )
        status = link.send_step_and_wait(
            PROTOCAL_ACTION_MOVE_X_ABS, 100.0
        )
        self.assertEqual(status["status"], PROTOCAL_STATUS_DONE)
        self.assertEqual(len(uart.writes), 1)
        self.assertEqual(uart.writes[0][5], 1)
        self.assertEqual(link.tx_frames, 1)
        self.assertEqual(link.status_frames, 2)

    def test_motion_program_is_seven_steps_per_piece(self):
        original = (
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_XX,
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_XY,
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_X_OFFSET_MM,
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_YX,
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_YY,
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_Y_OFFSET_MM,
            protocal_cfg.PROTOCAL_ROTATION_SIGN,
        )
        try:
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_XX = 1.0
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_XY = 0.0
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_X_OFFSET_MM = 1.0
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_YX = 0.0
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_YY = 1.0
            protocal_cfg.PROTOCAL_A4_TO_MACHINE_Y_OFFSET_MM = 2.0
            protocal_cfg.PROTOCAL_ROTATION_SIGN = -1.0
            program = protocal_build_motion_program(_Plan())
        finally:
            (
                protocal_cfg.PROTOCAL_A4_TO_MACHINE_XX,
                protocal_cfg.PROTOCAL_A4_TO_MACHINE_XY,
                protocal_cfg.PROTOCAL_A4_TO_MACHINE_X_OFFSET_MM,
                protocal_cfg.PROTOCAL_A4_TO_MACHINE_YX,
                protocal_cfg.PROTOCAL_A4_TO_MACHINE_YY,
                protocal_cfg.PROTOCAL_A4_TO_MACHINE_Y_OFFSET_MM,
                protocal_cfg.PROTOCAL_ROTATION_SIGN,
            ) = original

        self.assertEqual(len(program), 7)
        self.assertEqual(
            [step[2] for step in program],
            [
                PROTOCAL_ACTION_MOVE_X_ABS,
                PROTOCAL_ACTION_MOVE_Y_ABS,
                PROTOCAL_ACTION_GRIP,
                PROTOCAL_ACTION_ROTATE_REL,
                PROTOCAL_ACTION_MOVE_X_ABS,
                PROTOCAL_ACTION_MOVE_Y_ABS,
                PROTOCAL_ACTION_GRIP,
            ],
        )
        self.assertEqual(program[0][3], 11.0)
        self.assertEqual(program[1][3], 22.0)
        self.assertEqual(program[3][3], 25.0)

        fake_link = _FakeLink()
        result = protocal_execute_motion_program(
            program, link=fake_link
        )
        self.assertTrue(result["executed"])
        self.assertTrue(fake_link.opened)
        self.assertFalse(fake_link.closed)
        self.assertEqual(len(fake_link.commands), 7)

    def test_zero_rotation_command_is_omitted(self):
        plan = _Plan()
        plan.operations[0]["rotation_deg"] = 0.0
        program = protocal_build_motion_program(plan)
        self.assertEqual(len(program), 6)
        self.assertNotIn(
            PROTOCAL_ACTION_ROTATE_REL,
            [step[2] for step in program],
        )


if __name__ == "__main__":
    unittest.main()
