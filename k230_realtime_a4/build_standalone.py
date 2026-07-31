#!/usr/bin/env python3
"""Build the active free-rectangle one-file CanMV deployment."""

import argparse
import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = (
    HERE
    / "k230_realtime_a4_simulator_free_rect_standalone.py"
)
NO_UART_OUTPUT = (
    HERE
    / "k230_realtime_a4_simulator_free_rect_no_uart_standalone.py"
)
DEBUG_OUTPUT = (
    HERE
    / "k230_realtime_a4_source_projective_recognition_debug_standalone.py"
)
SOURCE_PROJECTIVE_NO_UART_OUTPUT = (
    HERE
    / "k230_realtime_a4_simulator_free_rect_source_projective_no_uart_standalone.py"
)
SOURCE_PROJECTIVE_OPTIMIZED_NO_UART_OUTPUT = (
    HERE
    / "k230_realtime_a4_simulator_free_rect_source_projective_optimized_no_uart_standalone.py"
)
LOCAL_MODULES = {
    "puzzle_config",
    "puzzle_perf",
    "puzzle_geometry",
    "puzzle_simulator_planner",
    "puzzle_simulator_free_rect_planner",
    "puzzle_placement",
    "puzzle_realtime_state",
    "puzzle_vision",
    "k230_puzzle_planner",
    "realtime_a4_config",
    "realtime_a4_free_rect_config",
    "realtime_a4_recognition_debug_config",
    "realtime_free_rect_source_projective_no_uart_config",
    "realtime_free_rect_source_projective_optimized_no_uart_config",
    "puzzle_a4_boundary",
    "a4_projective_mapper",
    "source_divider_detector",
    "source_projective_piece_detector",
}

# These pre-existing helpers are intentionally equivalent and are overwritten
# by a later section in the flat standalone namespace.  Any new duplicate is
# a build error: unlike normal Python modules, concatenated private names are
# global and can silently change another module's runtime call signature.
ALLOWED_DUPLICATE_SYMBOLS = {
    "_cross",
    "_distance",
    "_draw_quad",
    "_print_a4_lock",
}

TIMESTAMPED_LOGGING_BLOCK = '''\
# Prefix every runtime log with milliseconds since this script started.
import time as _standalone_log_time

_STANDALONE_RAW_PRINT = print
_STANDALONE_TICKS_MS = getattr(
    _standalone_log_time, "ticks_ms", None
)
_STANDALONE_TICKS_DIFF = getattr(
    _standalone_log_time, "ticks_diff", None
)


def _standalone_log_now_ms():
    if _STANDALONE_TICKS_MS is not None:
        return int(_STANDALONE_TICKS_MS())
    return int(_standalone_log_time.time() * 1000.0)


_STANDALONE_LOG_STARTED_MS = _standalone_log_now_ms()


def _standalone_log_elapsed_ms():
    now_ms = _standalone_log_now_ms()
    if _STANDALONE_TICKS_DIFF is not None:
        elapsed_ms = int(
            _STANDALONE_TICKS_DIFF(
                now_ms, _STANDALONE_LOG_STARTED_MS
            )
        )
    else:
        elapsed_ms = now_ms - _STANDALONE_LOG_STARTED_MS
    return max(0, elapsed_ms)


def print(*values, **options):
    _STANDALONE_RAW_PRINT(
        "[T+{:07d}ms]".format(
            _standalone_log_elapsed_ms()
        ),
        *values,
        **options
    )
'''


DIRECT_UART2_PLAN_OUTPUT_BLOCK = '''\
_UART2_PLAN_OUTPUT = None
_UART2_PLAN_FPIOA = None
_UART2_PLAN_TX_PIN = 5
_UART2_PLAN_RX_PIN = 6
_UART2_PLAN_BAUDRATE = 115200


def _open_uart2_plan_output():
    """Lazily open the standalone-only four-line UART2 output."""
    global _UART2_PLAN_OUTPUT, _UART2_PLAN_FPIOA
    if _UART2_PLAN_OUTPUT is not None:
        return _UART2_PLAN_OUTPUT
    try:
        from machine import FPIOA, UART

        _UART2_PLAN_FPIOA = FPIOA()
        _UART2_PLAN_FPIOA.set_function(
            _UART2_PLAN_TX_PIN, FPIOA.UART2_TXD
        )
        _UART2_PLAN_FPIOA.set_function(
            _UART2_PLAN_RX_PIN, FPIOA.UART2_RXD
        )
        _UART2_PLAN_OUTPUT = UART(
            UART.UART2,
            baudrate=_UART2_PLAN_BAUDRATE,
            bits=UART.EIGHTBITS,
            parity=UART.PARITY_NONE,
            stop=UART.STOPBITS_ONE,
            timeout=0,
        )
        print(
            "UART2_PLAN_READY,tx_pin={},rx_pin={},baudrate={}".format(
                _UART2_PLAN_TX_PIN,
                _UART2_PLAN_RX_PIN,
                _UART2_PLAN_BAUDRATE,
            )
        )
        return _UART2_PLAN_OUTPUT
    except Exception as exc:
        _UART2_PLAN_OUTPUT = None
        print(
            "UART2_PLAN_ERROR,stage=open,reason={}".format(
                str(exc).replace(",", ";")
            )
        )
        return None


def _write_plan_operations_uart2(plan):
    """Write exactly four source/target/rotation records after Planning."""
    operations = list(getattr(plan, "operations", ()))
    if not getattr(plan, "valid", False) or len(operations) != 4:
        print(
            "UART2_PLAN_SKIPPED,valid={},operations={}".format(
                int(bool(getattr(plan, "valid", False))),
                len(operations),
            )
        )
        return False
    uart = _open_uart2_plan_output()
    if uart is None:
        return False
    try:
        for operation in operations:
            source = operation["source_center_mm"]
            target = operation["target_center_mm"]
            line = (
                "UART2_PLAN,piece_id={},source_x={:.2f},source_y={:.2f},"
                "target_x={:.2f},target_y={:.2f},rot={:.2f}\\r\\n"
            ).format(
                operation["piece_id"],
                source[0],
                source[1],
                target[0],
                target[1],
                operation["rotation_deg"],
            )
            uart.write(line.encode("ascii"))
        print("UART2_PLAN_SENT,records=4")
        return True
    except Exception as exc:
        print(
            "UART2_PLAN_ERROR,stage=write,reason={}".format(
                str(exc).replace(",", ";")
            )
        )
        return False
'''


def _inject_direct_uart2_plan_call(runtime_source):
    marker = "                            _print_all_plan_operations(active_plan)\n"
    replacement = (
        marker
        + "                            _write_plan_operations_uart2(active_plan)\n"
    )
    if runtime_source.count(marker) != 1:
        raise ValueError(
            "expected exactly one plan-operation output marker"
        )
    return runtime_source.replace(marker, replacement, 1)


def _assigned_names(source):
    tree = ast.parse(source)
    names = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id.isupper()
            ):
                names.append(target.id)
    return names


def _filtered_source(path, omit_main=False, omit_guard=False):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    omitted_lines = set()
    for node in ast.walk(tree):
        omit = False
        if isinstance(node, ast.Import):
            omit = any(
                alias.name in LOCAL_MODULES for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            omit = node.module in LOCAL_MODULES
        if omit:
            omitted_lines.update(
                range(node.lineno, node.end_lineno + 1)
            )
    for node in tree.body:
        if (
            omit_main
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
        ):
            omitted_lines.update(
                range(node.lineno, node.end_lineno + 1)
            )
        if (
            omit_guard
            and isinstance(node, ast.If)
            and "__name__" in ast.unparse(node.test)
        ):
            omitted_lines.update(
                range(node.lineno, node.end_lineno + 1)
            )
    return "\n".join(
        line
        for line_number, line in enumerate(
            source.splitlines(), 1
        )
        if line_number not in omitted_lines
    )


def _config_block(extra_override_paths=None):
    base_path = ROOT / "puzzle_config.py"
    override_paths = [
        HERE / "realtime_a4_config.py",
        HERE / "realtime_a4_free_rect_config.py",
    ]
    override_paths.extend(extra_override_paths or ())
    base = base_path.read_text(encoding="utf-8")
    base_names = _assigned_names(base)
    lines = [
        base,
        "class _StandaloneConfig:",
        "    pass",
        "",
        "cfg = _StandaloneConfig()",
    ]
    lines.extend(
        "cfg.{0} = {0}".format(name) for name in base_names
    )
    for override_path in override_paths:
        lines.append(_filtered_source(override_path))
        override_names = _assigned_names(
            override_path.read_text(encoding="utf-8")
        )
        lines.extend(
            "cfg.{0} = {0}".format(name)
            for name in override_names
        )
    lines.append("cfg.STANDALONE_BUILD = True")
    return "\n".join(lines)


def _validate_flat_namespace(source):
    tree = ast.parse(source)
    definitions = {}
    for node in tree.body:
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        definitions.setdefault(node.name, []).append(node.lineno)
    unexpected = {
        name: lines
        for name, lines in definitions.items()
        if len(lines) > 1 and name not in ALLOWED_DUPLICATE_SYMBOLS
    }
    if unexpected:
        details = ", ".join(
            "{}@{}".format(
                name, "|".join(str(line) for line in lines)
            )
            for name, lines in sorted(unexpected.items())
        )
        raise ValueError(
            "unsafe duplicate standalone symbols: {}".format(details)
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--recognition-debug",
        action="store_true",
        help=(
            "build the ACQUIRE-held raw/fitted contour diagnostic"
        ),
    )
    parser.add_argument(
        "--no-uart",
        action="store_true",
        help="build a one-file test runtime with UART execution disabled",
    )
    parser.add_argument(
        "--source-projective-no-uart",
        action="store_true",
        help=(
            "build the source-projective FreeRect production profile "
            "with UART execution disabled"
        ),
    )
    parser.add_argument(
        "--source-projective-optimized-no-uart",
        action="store_true",
        help=(
            "build the isolated staged source-projective FreeRect "
            "profile with UART execution disabled"
        ),
    )
    args = parser.parse_args(argv)
    selected_profiles = sum(
        int(value)
        for value in (
            args.recognition_debug,
            args.no_uart,
            args.source_projective_no_uart,
            args.source_projective_optimized_no_uart,
        )
    )
    if selected_profiles > 1:
        parser.error(
            "--recognition-debug, --no-uart, and "
            "--source-projective-no-uart, and "
            "--source-projective-optimized-no-uart are mutually exclusive"
        )
    output = args.output or (
        DEBUG_OUTPUT
        if args.recognition_debug
        else SOURCE_PROJECTIVE_OPTIMIZED_NO_UART_OUTPUT
        if args.source_projective_optimized_no_uart
        else SOURCE_PROJECTIVE_NO_UART_OUTPUT
        if args.source_projective_no_uart
        else NO_UART_OUTPUT
        if args.no_uart
        else OUTPUT
    )
    if args.recognition_debug:
        extra_override_paths = [
            HERE / "realtime_a4_recognition_debug_config.py"
        ]
    elif args.source_projective_optimized_no_uart:
        extra_override_paths = [
            HERE
            / "realtime_free_rect_source_projective_no_uart_config.py",
            HERE
            / "realtime_free_rect_source_projective_optimized_no_uart_config.py",
        ]
    elif args.source_projective_no_uart:
        extra_override_paths = [
            HERE
            / "realtime_free_rect_source_projective_no_uart_config.py"
        ]
    else:
        extra_override_paths = []
    sections = [
        "#!/usr/bin/env python3\n"
        '"""Generated simulator-backed realtime A4 CanMV planner."""\n'
        "# Generated by k230_realtime_a4/build_standalone.py.\n",
        TIMESTAMPED_LOGGING_BLOCK,
        _config_block(extra_override_paths),
        _filtered_source(ROOT / "puzzle_perf.py"),
        _filtered_source(ROOT / "puzzle_geometry.py"),
        _filtered_source(ROOT / "puzzle_simulator_planner.py"),
    ]
    sections.append(
        _filtered_source(
            ROOT / "puzzle_simulator_free_rect_planner.py"
        )
    )
    runtime_source = _filtered_source(HERE / "k230_realtime_a4.py")
    if args.source_projective_optimized_no_uart:
        runtime_source = _inject_direct_uart2_plan_call(
            runtime_source
        )
    sections.extend([
        _filtered_source(ROOT / "puzzle_placement.py"),
        _filtered_source(ROOT / "puzzle_realtime_state.py"),
        _filtered_source(ROOT / "puzzle_vision.py"),
        _filtered_source(ROOT / "puzzle_a4_boundary.py"),
        _filtered_source(ROOT / "a4_projective_mapper.py"),
        _filtered_source(ROOT / "source_divider_detector.py"),
        _filtered_source(ROOT / "source_projective_piece_detector.py"),
        _filtered_source(
            ROOT / "k230_puzzle_planner.py",
            omit_main=True,
            omit_guard=True,
        ),
        (
            "# Test-build invariant: never import or execute UART support.\n"
            "UART_COMMUNICATION_ENABLED = False"
            if (
                args.no_uart
                or args.source_projective_no_uart
                or args.source_projective_optimized_no_uart
            )
            else ""
        ),
        (
            DIRECT_UART2_PLAN_OUTPUT_BLOCK
            if args.source_projective_optimized_no_uart
            else ""
        ),
        runtime_source,
    ])
    bundled_source = "\n\n".join(sections) + "\n"
    _validate_flat_namespace(bundled_source)
    output.write_text(bundled_source, encoding="utf-8")
    print(
        "BUILT,output={},bytes={}".format(
            output, output.stat().st_size
        )
    )


if __name__ == "__main__":
    main()
