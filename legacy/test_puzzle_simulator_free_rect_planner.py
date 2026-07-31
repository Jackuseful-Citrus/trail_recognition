import ast
import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import puzzle_config as cfg
from puzzle_geometry import (
    PieceObservation,
    polygon_area,
    polygon_centroid,
    transform_polygon,
)
import puzzle_simulator_free_rect_planner as free_planner
from puzzle_simulator_free_rect_planner import (
    plan_simulator_free_rectangle,
)
from puzzle_simulator_planner import plan_simulator_rectangle


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "frame_33_free_rect_regression.json"
)


def _rigid_transform(polygon, angle_deg, translation):
    angle = math.radians(angle_deg)
    return transform_polygon(
        polygon,
        (
            math.cos(angle),
            math.sin(angle),
            translation[0],
            translation[1],
            angle_deg,
        ),
    )


def _make_pieces(partition):
    angles = (17.0, -43.0, 71.0, -109.0)
    translations = (
        (20.0, 15.0),
        (135.0, 25.0),
        (80.0, 105.0),
        (155.0, 115.0),
    )
    pieces = []
    for index, polygon in enumerate(partition):
        transformed = _rigid_transform(
            polygon, angles[index], translations[index]
        )
        pieces.append(
            PieceObservation(
                "P{}".format(index + 1),
                [],
                transformed,
                confidence=1.0,
            )
        )
    return pieces


def _load_frame_33():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pieces = []
    for record in payload["pieces"]:
        pieces.append(
            PieceObservation(
                record["piece_id"],
                [],
                record["polygon_mm"],
                centroid_mm=record["centroid_mm"],
                area_mm2=record["area_mm2"],
                confidence=1.0,
            )
        )
    return payload, pieces


def _rounded_points(points):
    return sorted(
        (round(point[0], 6), round(point[1], 6))
        for point in points
    )


class FreeRectanglePlannerTests(unittest.TestCase):
    def setUp(self):
        self.saved = {}
        for name in (
            "FREE_RECT_MAX_COMPLETE_SETS",
            "FREE_RECT_MAX_PLAN_TIME_MS",
        ):
            self.saved[name] = getattr(cfg, name)
        cfg.FREE_RECT_MAX_COMPLETE_SETS = 1200
        cfg.FREE_RECT_MAX_PLAN_TIME_MS = 8000

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(cfg, name, value)

    def assert_rigid_operations(self, pieces, result):
        self.assertEqual(len(result.operations), len(pieces))
        target_rect = result.target_rect
        margin = cfg.FREE_RECT_TARGET_MARGIN_MM
        self.assertGreaterEqual(target_rect[0], margin - 1e-6)
        self.assertLessEqual(
            target_rect[2], cfg.A4_WIDTH_MM - margin + 1e-6
        )
        self.assertGreaterEqual(
            target_rect[1],
            cfg.A4_HEIGHT_MM * 0.5 + margin - 1e-6,
        )
        self.assertLessEqual(
            target_rect[3], cfg.A4_HEIGHT_MM - margin + 1e-6
        )
        self.assertGreaterEqual(
            target_rect[2] - target_rect[0],
            target_rect[3] - target_rect[1] - 1e-6,
        )
        for piece, operation in zip(pieces, result.operations):
            target = result.target_polygons[piece.piece_id]
            self.assertAlmostEqual(
                polygon_area(piece.polygon_mm),
                polygon_area(target),
                places=6,
            )
            self.assertAlmostEqual(
                operation["source_center_mm"][0],
                piece.centroid_mm[0],
                places=7,
            )
            self.assertAlmostEqual(
                operation["source_center_mm"][1],
                piece.centroid_mm[1],
                places=7,
            )
            target_center = polygon_centroid(target)
            self.assertAlmostEqual(
                operation["target_center_mm"][0],
                target_center[0],
                places=6,
            )
            self.assertAlmostEqual(
                operation["target_center_mm"][1],
                target_center[1],
                places=6,
            )
            angle = math.radians(operation["rotation_deg"])
            cosine = math.cos(angle)
            sine = math.sin(angle)
            reconstructed = []
            for x, y in piece.polygon_mm:
                dx = x - piece.centroid_mm[0]
                dy = y - piece.centroid_mm[1]
                reconstructed.append(
                    (
                        cosine * dx
                        - sine * dy
                        + target_center[0],
                        sine * dx
                        + cosine * dy
                        + target_center[1],
                    )
                )
            self.assertEqual(
                _rounded_points(reconstructed),
                _rounded_points(target),
            )

    def test_strict_rectangles_infer_all_required_sizes(self):
        cases = (
            (
                90.0,
                50.0,
                [[(0, 0), (90, 0), (90, 50), (0, 50)]],
            ),
            (
                100.0,
                60.0,
                [
                    [(0, 0), (42, 0), (42, 60), (0, 60)],
                    [(42, 0), (100, 0), (100, 60), (42, 60)],
                ],
            ),
            (
                110.0,
                70.0,
                [
                    [(0, 0), (45, 0), (45, 70), (0, 70)],
                    [(45, 0), (110, 0), (110, 30), (45, 30)],
                    [(45, 30), (110, 30), (110, 70), (45, 70)],
                ],
            ),
            (
                120.0,
                90.0,
                [
                    [(0, 0), (50, 0), (50, 90), (0, 90)],
                    [(50, 0), (120, 0), (120, 30), (50, 30)],
                    [(50, 30), (120, 30), (120, 60), (50, 60)],
                    [(50, 60), (120, 60), (120, 90), (50, 90)],
                ],
            ),
        )
        selected_topologies = []
        for long_side, short_side, partition in cases:
            pieces = _make_pieces(partition)
            result = plan_simulator_free_rectangle(pieces)
            self.assertTrue(result.valid, result.reason)
            self.assertAlmostEqual(
                result.plan_stats["long_side_mm"],
                long_side,
                places=5,
            )
            self.assertAlmostEqual(
                result.plan_stats["short_side_mm"],
                short_side,
                places=5,
            )
            self.assertEqual(
                result.plan_stats["complete_matching_set_count"]
                > 0,
                True,
            )
            self.assertLessEqual(
                len(result.plan_stats["top_k"]), 5
            )
            self.assert_rigid_operations(pieces, result)
            selected_topologies.append(
                result.plan_stats["selected_topology"]
            )
        self.assertEqual(selected_topologies[0], "single_piece")
        self.assertIn("0_partial", selected_topologies[1])
        self.assertIn("1_partial", selected_topologies[2])
        self.assertIn("1_partial", selected_topologies[3])

    def test_noisy_t_junction_is_scored_not_prefix_rejected(self):
        partition = [
            [(0, 0), (45, 0), (45, 70), (0, 70)],
            [(45, 0), (110, 0), (110, 30), (45, 30)],
            [(45, 30), (110, 30), (110, 70), (45, 70)],
        ]
        noise = (
            ((1.1, -0.7), (-1.2, 0.8), (0.9, -1.4), (-0.8, 1.2)),
            ((-0.9, 1.3), (1.2, -0.6), (-1.1, 0.9), (1.4, -1.2)),
            ((0.8, -1.1), (-1.4, 0.7), (1.3, -0.8), (-0.7, 1.4)),
        )
        noisy = []
        for polygon, offsets in zip(partition, noise):
            noisy.append(
                [
                    (
                        point[0] + offset[0],
                        point[1] + offset[1],
                    )
                    for point, offset in zip(polygon, offsets)
                ]
            )
        result = plan_simulator_free_rectangle(
            _make_pieces(noisy)
        )
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(
            result.plan_stats["complete_matching_set_count"], 0
        )
        self.assertGreater(result.overlap_mm2, 0.0)
        self.assertGreater(result.fill_gap_mm2, 0.0)
        self.assertGreater(result.score, 0.0)
        self.assertNotIn(
            "prefix_pruned_overlap", result.plan_stats
        )

    def test_frame_33_fixture_keeps_fixed_baseline_and_free_proposal(self):
        payload, pieces = _load_frame_33()
        baseline = payload["current_fixed_baseline"]
        fixed = plan_simulator_rectangle(
            pieces, validation="upstream"
        )
        self.assertEqual(fixed.valid, baseline["valid"])
        self.assertEqual(
            fixed.plan_stats["candidate_count"],
            baseline["candidate_count"],
        )
        self.assertEqual(
            fixed.plan_stats["matching_sets_evaluated"],
            baseline["matching_sets_evaluated"],
        )
        self.assertAlmostEqual(
            fixed.score, baseline["score"], places=10
        )
        self.assertAlmostEqual(
            fixed.overlap_mm2,
            baseline["overlap_mm2"],
            places=8,
        )
        self.assertEqual(
            list(fixed.target_rect), baseline["target_rect"]
        )

        cfg.FREE_RECT_MAX_COMPLETE_SETS = 400
        free = plan_simulator_free_rectangle(pieces)
        self.assertTrue(free.valid, free.reason)
        self.assertGreater(
            free.plan_stats["complete_matching_set_count"], 0
        )
        self.assertGreater(
            free.plan_stats["pose_optimization_count"], 0
        )
        self.assertNotIn(
            "prefix_pruned_overlap", free.plan_stats
        )
        self.assertTrue(free.operations)
        self.assertTrue(free.plan_stats["top_k"])

    def test_timeout_returns_best_so_far(self):
        pieces = _make_pieces(
            [
                [(0, 0), (42, 0), (42, 60), (0, 60)],
                [(42, 0), (100, 0), (100, 60), (42, 60)],
            ]
        )
        original_ticks = free_planner.ticks_ms
        calls = [0]

        def controlled_ticks():
            calls[0] += 1
            return 0 if calls[0] <= 5 else 10

        cfg.FREE_RECT_MAX_PLAN_TIME_MS = 5
        free_planner.ticks_ms = controlled_ticks
        try:
            result = plan_simulator_free_rectangle(pieces)
        finally:
            free_planner.ticks_ms = original_ticks
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.plan_stats["timed_out"])
        self.assertGreaterEqual(
            result.plan_stats["complete_matching_set_count"], 1
        )
        self.assertIn("timed_out_best_so_far", result.reason)

    def test_timeout_before_complete_is_invalid_and_explicit(self):
        pieces = _make_pieces(
            [
                [(0, 0), (42, 0), (42, 60), (0, 60)],
                [(42, 0), (100, 0), (100, 60), (42, 60)],
            ]
        )
        original_ticks = free_planner.ticks_ms
        calls = [0]

        def controlled_ticks():
            calls[0] += 1
            return 0 if calls[0] == 1 else 10

        cfg.FREE_RECT_MAX_PLAN_TIME_MS = 5
        free_planner.ticks_ms = controlled_ticks
        try:
            result = plan_simulator_free_rectangle(pieces)
        finally:
            free_planner.ticks_ms = original_ticks
        self.assertFalse(result.valid)
        self.assertTrue(result.plan_stats["timed_out"])
        self.assertEqual(
            result.reason,
            "no complete candidate before timeout",
        )

    def test_five_runs_are_deterministic(self):
        pieces = _make_pieces(
            [
                [(0, 0), (42, 0), (42, 60), (0, 60)],
                [(42, 0), (100, 0), (100, 60), (42, 60)],
            ]
        )
        snapshots = []
        logs = []
        for _ in range(5):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = plan_simulator_free_rectangle(pieces)
            stats = dict(result.plan_stats)
            stats.pop("plan_ms", None)
            snapshots.append(
                (
                    result.valid,
                    result.score,
                    result.reason,
                    stats,
                    result.operations,
                    result.target_polygons,
                )
            )
            logs.append(output.getvalue())
        for snapshot in snapshots[1:]:
            self.assertEqual(snapshot, snapshots[0])
        for log in logs[1:]:
            self.assertEqual(log, logs[0])

    def test_fixed_defaults_are_unchanged(self):
        self.assertEqual(cfg.PLANNER_BACKEND, "outer_first")
        self.assertEqual(cfg.TARGET_RECT_SIZE_MM, (100.0, 60.0))
        self.assertEqual(cfg.MAX_PLAN_TIME_MS, 3000)
        self.assertEqual(cfg.SIMULATOR_MAX_CANDIDATES, 80)
        self.assertEqual(cfg.SIMULATOR_MAX_MATCHING_SETS, 4000)
        self.assertEqual(cfg.MIN_PIECE_COUNT, 2)
        self.assertEqual(cfg.MAX_PIECE_COUNT, 4)

    def test_module_and_generated_bundle_are_micropython_safe(self):
        module_path = ROOT / "puzzle_simulator_free_rect_planner.py"
        source = module_path.read_text(encoding="utf-8")
        compile(source, str(module_path), "exec")
        tree = ast.parse(source)
        forbidden = {"numpy", "cv2", "dataclasses"}
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(
                    alias.name.split(".")[0]
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertTrue(forbidden.isdisjoint(imports))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "free_standalone.py"
            subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "k230_realtime_a4"
                        / "build_standalone.py"
                    ),
                    "--planner-backend",
                    "simulator_free_rect",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            bundle = output.read_text(encoding="utf-8")
            compile(bundle, str(output), "exec")
            self.assertIn(
                "cfg.PLANNER_BACKEND = 'simulator_free_rect'",
                bundle,
            )
            self.assertIn(
                "def plan_simulator_free_rectangle", bundle
            )
            bundle_tree = ast.parse(bundle)
            bundle_imports = set()
            for node in ast.walk(bundle_tree):
                if isinstance(node, ast.Import):
                    bundle_imports.update(
                        alias.name.split(".")[0]
                        for alias in node.names
                    )
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                ):
                    bundle_imports.add(
                        node.module.split(".")[0]
                    )
            self.assertTrue(
                forbidden.isdisjoint(bundle_imports)
            )


if __name__ == "__main__":
    unittest.main()
