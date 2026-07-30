#!/usr/bin/env python3
"""Build the dependency-free CanMV A4 recognition test script."""

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "k230_a4_auto_calibration_standalone.py"
CONFIG_PATH = ROOT / "puzzle_config.py"
BOUNDARY_PATH = ROOT / "puzzle_a4_boundary.py"
ENTRYPOINT_PATH = ROOT / "k230_a4_recognition_test.py"

CONFIG_NAMES = {
    "FRAME_WIDTH",
    "FRAME_HEIGHT",
    "A4_DETECT_WIDTH",
    "A4_DETECT_HEIGHT",
    "A4_RECT_EDGE_THRESHOLD",
    "A4_MIN_FRAME_AREA_RATIO",
    "A4_MAX_FRAME_AREA_RATIO",
    "A4_MIN_WIDTH_HEIGHT_RATIO",
    "A4_MAX_WIDTH_HEIGHT_RATIO",
    "A4_EXPECTED_WIDTH_HEIGHT_RATIO",
    "A4_MIN_SIDE_PX",
    "A4_MAX_CENTER_OFFSET_RATIO",
    "A4_MIN_IMAGE_EDGE_MARGIN_PX",
    "A4_TOP_SIDE",
    "A4_DARK_THRESHOLD",
    "A4_MAX_INSIDE_GRAY",
    "A4_DARK_BLOB_MIN_AREA_RATIO",
    "A4_EDGE_PROBE_OFFSET_PX",
    "A4_EDGE_PROBE_SAMPLES",
    "A4_INTERNAL_EDGE_SIMILAR_GRAY_DELTA",
    "A4_INTERNAL_EDGE_DARK_RATIO_MAX",
    "A4_INTERNAL_EDGE_MIN_SAMPLES",
    "A4_LOCK_REQUIRED_FRAMES",
    "A4_HOLD_MISSED_FRAMES",
    "A4_SLOW_SMOOTH_ALPHA",
    "A4_STATUS_PRINT_EVERY_N_FRAMES",
    "A4_WIDTH_MM",
    "A4_HEIGHT_MM",
    "DIVIDER_Y_MM",
    "DIVIDER_SEARCH_HALF_RANGE_MM",
    "ENABLE_DYNAMIC_DIVIDER",
    "A4_REQUIRE_DIVIDER_FOR_LOCK",
    "DIVIDER_LINE_SAMPLE_COUNT",
    "DIVIDER_LINE_MIN_GRAY",
    "DIVIDER_LINE_MIN_CONTRAST_GRAY",
    "DIVIDER_LINE_MIN_COVERAGE",
    "DIVIDER_LINE_MAX_RESIDUAL_PX",
    "DIVIDER_LINE_MAX_SLOPE_MM",
    "DIVIDER_TRACK_ALPHA",
    "WHITE_GRAY_THRESHOLD",
    "A4_LOCK_DEADBAND_PX",
    "A4_RELOCK_MOTION_PX",
    "A4_RELOCK_CONFIRM_FRAMES",
    "LOOP_IDLE_MS",
    "AUTO_STOP_SECONDS",
    "MAX_FRAME_COUNT",
}

LOCAL_MODULES = {"puzzle_config", "puzzle_a4_boundary"}


def _assigned_name(node):
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return None
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return None
    return targets[0].id


def _config_block():
    source = CONFIG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = []
    found = set()
    for node in tree.body:
        name = _assigned_name(node)
        if name not in CONFIG_NAMES:
            continue
        segment = ast.get_source_segment(source, node)
        if segment is None:
            raise RuntimeError("cannot extract config assignment: {}".format(name))
        assignments.append(segment)
        found.add(name)
    missing = sorted(CONFIG_NAMES - found)
    if missing:
        raise RuntimeError(
            "missing A4 config values: {}".format(", ".join(missing))
        )
    return "\n".join(assignments)


def _without_local_imports(path):
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
            omitted_lines.update(range(node.lineno, node.end_lineno + 1))
    return "\n".join(
        line
        for line_number, line in enumerate(source.splitlines(), 1)
        if line_number not in omitted_lines
    )


def _inline_config_references(source):
    source = source.replace(
        'getattr(cfg, "STANDALONE_BUILD", False)',
        "True",
    )
    source = re.sub(r"\bcfg\.([A-Z][A-Z0-9_]*)", r"\1", source)
    if re.search(r"\bcfg\b", source):
        raise RuntimeError("standalone source still contains cfg references")
    return source


def build_source():
    sections = [
        "#!/usr/bin/env python3\n"
        '"""Self-contained CanMV K230 automatic A4 calibration test."""\n'
        "# No project-local imports are required; this file can be edited and\n"
        "# run directly in CanMV IDE.",
        _config_block(),
        _inline_config_references(
            _without_local_imports(BOUNDARY_PATH)
        ),
        _inline_config_references(
            _without_local_imports(ENTRYPOINT_PATH)
        ),
    ]
    source = "\n\n".join(sections) + "\n"
    ast.parse(source)
    forbidden = (
        "puzzle_vision",
        "puzzle_geometry",
        "puzzle_placement",
        "plan_rectangle_assembly",
        "detect_pieces",
        "import puzzle_config",
        "from puzzle_a4_boundary",
        "cfg.",
    )
    leaked = [name for name in forbidden if name in source]
    if leaked:
        raise RuntimeError(
            "non-A4 code leaked into standalone build: {}".format(
                ", ".join(leaked)
            )
        )
    return source


def main():
    source = build_source()
    OUTPUT.write_text(source, encoding="utf-8")
    print(
        "BUILT,output={},bytes={}".format(
            OUTPUT,
            OUTPUT.stat().st_size,
        )
    )


if __name__ == "__main__":
    main()
