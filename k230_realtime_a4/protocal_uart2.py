"""Minimal binary UART2 client for the K230/DM-board step protocol.

The module is deliberately small: one command is sent, then the caller blocks
until the matching BUSY/DONE/ERROR status arrives.  It performs no hot-plug,
heartbeat, retry, or reconnect handling.
"""

try:
    import ustruct as struct
except ImportError:  # CPython desktop validation
    import struct

import math
import time


PROTOCAL_VERSION = 1
PROTOCAL_FLAG_VALID = 0x01

PROTOCAL_CMD_STEP = 0x10
PROTOCAL_CMD_STEP_STATUS = 0x90

PROTOCAL_ACTION_MOVE_X_ABS = 1
PROTOCAL_ACTION_MOVE_Y_ABS = 2
PROTOCAL_ACTION_ROTATE_REL = 3
PROTOCAL_ACTION_GRIP = 4
PROTOCAL_ACTION_Z = 5

PROTOCAL_STATUS_BUSY = 0
PROTOCAL_STATUS_DONE = 1
PROTOCAL_STATUS_ERROR = 2

PROTOCAL_STEP_PAYLOAD_SIZE = 8
PROTOCAL_STATUS_PAYLOAD_SIZE = 8
PROTOCAL_MAX_PAYLOAD_SIZE = 32

_HEADER = b"\x55\xAA"

PROTOCAL_ERROR_NAMES = {
    0: "none",
    1: "bad_version",
    2: "invalid_flags",
    3: "unsupported_action",
    4: "invalid_value",
    5: "out_of_range",
    6: "executor_busy",
    7: "hardware_not_ready",
    8: "timeout",
    9: "sequence_conflict",
}


class ProtocalError(Exception):
    """Base error for the minimal UART protocol client."""


class ProtocalTimeoutError(ProtocalError):
    """The matching terminal status did not arrive before the deadline."""


class ProtocalRemoteError(ProtocalError):
    """The DM board returned an ERROR status."""

    def __init__(self, sequence, action, error_code):
        self.sequence = int(sequence)
        self.action = int(action)
        self.error_code = int(error_code)
        self.error_name = PROTOCAL_ERROR_NAMES.get(
            self.error_code, "unknown"
        )
        super().__init__(
            "remote error sequence={} action={} code={} ({})".format(
                self.sequence,
                self.action,
                self.error_code,
                self.error_name,
            )
        )


def _ticks_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.monotonic() * 1000.0)


def _ticks_diff(now, before):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(now, before)
    return int(now - before)


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


def protocal_crc16_ccitt_false(data):
    """Return CRC-16/CCITT-FALSE for a bytes-like iterable."""
    crc = 0xFFFF
    for value in data:
        crc ^= int(value) << 8
        for _bit in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def protocal_build_frame(command, payload):
    payload = bytes(payload)
    if len(payload) > PROTOCAL_MAX_PAYLOAD_SIZE:
        raise ValueError("payload is too large")
    body = bytes((int(command) & 0xFF, len(payload))) + payload
    crc = protocal_crc16_ccitt_false(body)
    return (
        _HEADER
        + body
        + bytes((crc & 0xFF, (crc >> 8) & 0xFF))
    )


def protocal_build_step_frame(sequence, action, value):
    sequence = int(sequence)
    action = int(action)
    value = float(value)
    if not 0 <= sequence <= 255:
        raise ValueError("sequence must be in 0..255")
    if not 1 <= action <= 5:
        raise ValueError("action must be in 1..5")
    if not _is_finite(value):
        raise ValueError("step value must be finite")
    payload = struct.pack(
        "<BBBBf",
        PROTOCAL_VERSION,
        sequence,
        action,
        PROTOCAL_FLAG_VALID,
        value,
    )
    return protocal_build_frame(PROTOCAL_CMD_STEP, payload)


class ProtocalFrameParser:
    """Incremental parser that accepts fragmented or concatenated frames."""

    def __init__(self):
        self.buffer = bytearray()
        self.frames = 0
        self.crc_errors = 0
        self.format_errors = 0
        self.discarded_bytes = 0

    def _find_header(self):
        limit = len(self.buffer) - 1
        for index in range(max(0, limit)):
            if (
                self.buffer[index] == 0x55
                and self.buffer[index + 1] == 0xAA
            ):
                return index
        return -1

    def feed(self, data):
        if data:
            self.buffer.extend(data)
        parsed = []
        while True:
            if len(self.buffer) < 2:
                return parsed

            header_index = self._find_header()
            if header_index < 0:
                keep_header_byte = self.buffer[-1] == 0x55
                discarded = len(self.buffer) - (
                    1 if keep_header_byte else 0
                )
                self.discarded_bytes += discarded
                if keep_header_byte:
                    self.buffer[:] = b"\x55"
                else:
                    self.buffer[:] = b""
                return parsed

            if header_index > 0:
                self.discarded_bytes += header_index
                del self.buffer[:header_index]

            if len(self.buffer) < 4:
                return parsed

            payload_length = int(self.buffer[3])
            if payload_length > PROTOCAL_MAX_PAYLOAD_SIZE:
                self.format_errors += 1
                del self.buffer[0]
                continue

            frame_length = 2 + 1 + 1 + payload_length + 2
            if len(self.buffer) < frame_length:
                return parsed

            frame = bytes(self.buffer[:frame_length])
            del self.buffer[:frame_length]
            received_crc = frame[-2] | (frame[-1] << 8)
            calculated_crc = protocal_crc16_ccitt_false(
                frame[2:-2]
            )
            if received_crc != calculated_crc:
                self.crc_errors += 1
                continue

            self.frames += 1
            parsed.append((frame[2], frame[4:-2]))


def protocal_parse_step_status(payload):
    if len(payload) != PROTOCAL_STATUS_PAYLOAD_SIZE:
        raise ValueError("step status payload length must be 8")
    return {
        "version": int(payload[0]),
        "sequence": int(payload[1]),
        "action": int(payload[2]),
        "status": int(payload[3]),
        "error_code": int(payload[4]),
        "reserved": bytes(payload[5:8]),
    }


class K230Uart2Link:
    """One-command-at-a-time UART2 link used by the puzzle executor."""

    def __init__(
        self,
        tx_pin=5,
        rx_pin=6,
        baudrate=115200,
        initial_sequence=0,
        status_timeout_ms=20000,
        poll_delay_ms=2,
        uart=None,
    ):
        self.tx_pin = int(tx_pin)
        self.rx_pin = int(rx_pin)
        self.baudrate = int(baudrate)
        self.sequence = int(initial_sequence) & 0xFF
        self.status_timeout_ms = max(1, int(status_timeout_ms))
        self.poll_delay_ms = max(0, int(poll_delay_ms))
        self.uart = uart
        self._owns_uart = uart is None
        self._fpioa = None
        self.parser = ProtocalFrameParser()
        self.tx_frames = 0
        self.tx_bytes = 0
        self.rx_bytes = 0
        self.status_frames = 0
        self.ignored_frames = 0

    @property
    def is_open(self):
        return self.uart is not None

    def open(self):
        if self.uart is not None:
            return self
        try:
            from machine import FPIOA, UART
        except ImportError:
            raise ProtocalError(
                "machine.FPIOA/UART is unavailable"
            )

        self._fpioa = FPIOA()
        self._fpioa.set_function(
            self.tx_pin, FPIOA.UART2_TXD
        )
        self._fpioa.set_function(
            self.rx_pin, FPIOA.UART2_RXD
        )
        self.uart = UART(
            UART.UART2,
            baudrate=self.baudrate,
            bits=UART.EIGHTBITS,
            parity=UART.PARITY_NONE,
            stop=UART.STOPBITS_ONE,
            timeout=0,
        )
        print(
            "PROTOCAL_UART_READY,uart=2,tx_pin={},rx_pin={},"
            "baudrate={}".format(
                self.tx_pin, self.rx_pin, self.baudrate
            )
        )
        return self

    def close(self):
        if self.uart is not None and self._owns_uart:
            try:
                self.uart.deinit()
            finally:
                self.uart = None
                self._fpioa = None

    def _next_sequence(self):
        self.sequence = (self.sequence + 1) & 0xFF
        return self.sequence

    def _read_frames(self):
        data = self.uart.read()
        if not data:
            return []
        self.rx_bytes += len(data)
        return self.parser.feed(data)

    def send_step_and_wait(
        self, action, value, timeout_ms=None
    ):
        self.open()
        sequence = self._next_sequence()
        action = int(action)
        value = float(value)
        frame = protocal_build_step_frame(
            sequence, action, value
        )
        written = self.uart.write(frame)
        if written is not None and int(written) != len(frame):
            raise ProtocalError(
                "short UART write: {}/{}".format(
                    written, len(frame)
                )
            )
        self.tx_frames += 1
        self.tx_bytes += len(frame)
        print(
            "PROTOCAL_TX,sequence={},action={},value={:.3f},"
            "bytes={}".format(
                sequence, action, value, len(frame)
            )
        )

        wait_ms = (
            self.status_timeout_ms
            if timeout_ms is None
            else max(1, int(timeout_ms))
        )
        started_ms = _ticks_ms()
        busy_reported = False
        while _ticks_diff(_ticks_ms(), started_ms) <= wait_ms:
            for command, payload in self._read_frames():
                if (
                    command != PROTOCAL_CMD_STEP_STATUS
                    or len(payload) != PROTOCAL_STATUS_PAYLOAD_SIZE
                ):
                    self.ignored_frames += 1
                    continue
                status = protocal_parse_step_status(payload)
                self.status_frames += 1
                if (
                    status["version"] != PROTOCAL_VERSION
                    or status["sequence"] != sequence
                    or status["action"] != action
                ):
                    self.ignored_frames += 1
                    continue

                status_code = status["status"]
                if status_code == PROTOCAL_STATUS_BUSY:
                    if not busy_reported:
                        print(
                            "PROTOCAL_RX,sequence={},action={},"
                            "status=BUSY".format(sequence, action)
                        )
                        busy_reported = True
                    continue
                if status_code == PROTOCAL_STATUS_DONE:
                    print(
                        "PROTOCAL_RX,sequence={},action={},"
                        "status=DONE".format(sequence, action)
                    )
                    return status
                if status_code == PROTOCAL_STATUS_ERROR:
                    print(
                        "PROTOCAL_RX,sequence={},action={},"
                        "status=ERROR,error_code={}".format(
                            sequence,
                            action,
                            status["error_code"],
                        )
                    )
                    raise ProtocalRemoteError(
                        sequence,
                        action,
                        status["error_code"],
                    )
                self.ignored_frames += 1
            _sleep_ms(self.poll_delay_ms)

        raise ProtocalTimeoutError(
            "status timeout sequence={} action={} timeout_ms={}".format(
                sequence, action, wait_ms
            )
        )
