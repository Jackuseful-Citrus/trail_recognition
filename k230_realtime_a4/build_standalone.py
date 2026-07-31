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
    args = parser.parse_args(argv)
    selected_profiles = sum(
        int(value)
        for value in (
            args.recognition_debug,
            args.no_uart,
            args.source_projective_no_uart,
        )
    )
    if selected_profiles > 1:
        parser.error(
            "--recognition-debug, --no-uart, and "
            "--source-projective-no-uart are mutually exclusive"
        )
    output = args.output or (
        DEBUG_OUTPUT
        if args.recognition_debug
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
            if args.no_uart or args.source_projective_no_uart
            else ""
        ),
        _filtered_source(HERE / "k230_realtime_a4.py"),
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
