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
    is_fixed_figure2_piece_set,
    match_fixed_figure2_piece_set,
    plan_simulator_free_rectangle,
)


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

    def test_exposed_perimeter_merges_selected_and_closing_seams(self):
        quadrants = [
            [(0, 0), (50, 0), (50, 30), (0, 30)],
            [(50, 0), (100, 0), (100, 30), (50, 30)],
            [(0, 30), (50, 30), (50, 60), (0, 60)],
            [(50, 30), (100, 30), (100, 60), (50, 60)],
        ]
        matches = (
            (0.0, 0, 1, 1, 3, 0.0, 1.0, 0.0, 1.0),
            (0.0, 0, 2, 2, 0, 0.0, 1.0, 0.0, 1.0),
            (0.0, 2, 1, 3, 3, 0.0, 1.0, 0.0, 1.0),
        )
        identity = [(1.0, 0.0, 0.0, 0.0, 0.0)] * 4
        metrics = free_planner._free_exposed_perimeter_metrics(
            quadrants,
            matches,
            identity,
            6000.0,
        )
        self.assertAlmostEqual(
            metrics["source_perimeter_mm"], 640.0, places=6
        )
        self.assertAlmostEqual(
            metrics["selected_shared_length_mm"], 110.0, places=6
        )
        self.assertAlmostEqual(
            metrics["additional_shared_length_mm"], 50.0, places=6
        )
        self.assertEqual(metrics["additional_contact_count"], 1)
        self.assertAlmostEqual(
            metrics["exposed_perimeter_mm"], 320.0, places=6
        )
        self.assertEqual(metrics["perimeter_error_ratio"], 0.0)

        local_piece = [(0, 0), (50, 0), (50, 30), (0, 30)]
        chain_polygons = [local_piece] * 4
        chain_matches = (
            (0.0, 0, 1, 1, 3, 0.0, 1.0, 0.0, 1.0),
            (0.0, 1, 1, 2, 3, 0.0, 1.0, 0.0, 1.0),
            (0.0, 2, 1, 3, 3, 0.0, 1.0, 0.0, 1.0),
        )
        chain_transforms = [
            (1.0, 0.0, 50.0 * index, 0.0, 0.0)
            for index in range(4)
        ]
        chain = free_planner._free_exposed_perimeter_metrics(
            chain_polygons,
            chain_matches,
            chain_transforms,
            6000.0,
        )
        self.assertAlmostEqual(
            chain["exposed_perimeter_mm"], 460.0, places=6
        )
        self.assertGreater(
            chain["perimeter_excess_ratio"],
            cfg.FREE_RECT_MAX_PERIMETER_EXCESS_RATIO,
        )

    def test_exact_figure2_skips_enumeration_and_uses_fixed_targets(self):
        pieces = _make_pieces(
            [
                free_planner.FIGURE2_TEMPLATE_POLYGONS[role]
                for role in free_planner.FIGURE2_TEMPLATE_ORDER
            ]
        )
        original_candidates = (
            free_planner._free_rect_candidate_matchings
        )

        def enumeration_must_not_run(_pieces):
            raise AssertionError("enumeration unexpectedly ran")

        free_planner._free_rect_candidate_matchings = (
            enumeration_must_not_run
        )
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                result = plan_simulator_free_rectangle(pieces)
        finally:
            free_planner._free_rect_candidate_matchings = (
                original_candidates
            )

        self.assertTrue(result.valid, result.reason)
        self.assertEqual(
            result.mode, free_planner.FIGURE2_DIRECT_MODE
        )
        self.assertEqual(result.search_nodes, 0)
        self.assertEqual(
            result.plan_stats["fixed_template_layout"], "NORMAL"
        )
        self.assertTrue(
            result.plan_stats["fixed_template_matched"]
        )
        self.assertTrue(result.plan_stats["enumeration_skipped"])
        self.assertFalse(
            result.plan_stats["safety_gates_applied"]
        )
        self.assertEqual(
            result.plan_stats["complete_matching_set_count"], 0
        )
        self.assertEqual(
            result.plan_stats["pose_optimization_count"], 0
        )
        self.assertEqual(
            [operation["template_role"] for operation in result.operations],
            list(free_planner.FIGURE2_TEMPLATE_ORDER),
        )
        expected_centers = {
            "TOP_LEFT": (68.6666666667, 204.0),
            "RIGHT_TRIANGLE": (128.3333333333, 215.0),
            "MIDDLE_LEFT": (88.1111111111, 221.7777777778),
            "BOTTOM_LEFT": (95.0392156863, 243.4117647059),
        }
        for operation in result.operations:
            expected = expected_centers[
                operation["template_role"]
            ]
            self.assertAlmostEqual(
                operation["target_center_mm"][0],
                expected[0],
                places=6,
            )
            self.assertAlmostEqual(
                operation["target_center_mm"][1],
                expected[1],
                places=6,
            )
        log = output.getvalue()
        self.assertIn(
            "FREE_FIXED_TEMPLATE_CHECK,matched=1,layout=NORMAL",
            log,
        )
        self.assertIn(
            "FREE_FIXED_TEMPLATE_BYPASS,enumeration=SKIPPED,"
            "safety_gates=SKIPPED",
            log,
        )
        self.assertIn(
            "FREE_FIXED_TEMPLATE_RESULT,valid=1", log
        )
        self.assertNotIn("FREE_PLAN_START", log)

    def test_three_fixed_pieces_restore_merged_ten_mm_edge(self):
        partition = []
        for role in free_planner.FIGURE2_TEMPLATE_ORDER:
            polygon = free_planner.FIGURE2_TEMPLATE_POLYGONS[role]
            if role == "MIDDLE_LEFT":
                polygon = (
                    free_planner._free_figure2_short_edge_variant(
                        polygon
                    )
                )
            partition.append(polygon)
        pieces = _make_pieces(partition)

        match, reason = match_fixed_figure2_piece_set(pieces)
        self.assertIsNotNone(match, reason)
        self.assertEqual(match["matched_piece_count"], 3)
        self.assertEqual(match["inferred_roles"], ("MIDDLE_LEFT",))
        self.assertTrue(match["short_edge_fit_cancelled"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = plan_simulator_free_rectangle(
                pieces,
                fixed_template_evaluation=(match, None),
            )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(
            result.mode, free_planner.FIGURE2_DIRECT_MODE
        )
        self.assertEqual(
            result.plan_stats["fixed_template_matched_piece_count"],
            3,
        )
        self.assertTrue(
            result.plan_stats["short_edge_fit_cancelled"]
        )
        middle_piece = match["assignment"]["MIDDLE_LEFT"][1]
        self.assertEqual(
            len(result.target_polygons[middle_piece.piece_id]), 4
        )
        self.assertNotEqual(
            tuple(
                operation["source_center_mm"]
                for operation in result.operations
                if operation["template_role"] == "MIDDLE_LEFT"
            )[0],
            middle_piece.centroid_mm,
        )
        self.assertIn("matched_pieces=3/4", output.getvalue())
        self.assertIn(
            "short_edge_fit_cancelled=1", output.getvalue()
        )
        self.assertNotIn("FREE_PLAN_START", output.getvalue())

    def test_frame_33_fixture_uses_direct_plan(self):
        _payload, pieces = _load_frame_33()
        self.assertTrue(is_fixed_figure2_piece_set(pieces))
        cfg.FREE_RECT_MAX_COMPLETE_SETS = 400
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            free = plan_simulator_free_rectangle(pieces)
        self.assertTrue(free.valid, free.reason)
        self.assertEqual(
            free.mode, free_planner.FIGURE2_DIRECT_MODE
        )
        self.assertEqual(
            free.plan_stats["fixed_template_layout"], "MIRROR_X"
        )
        self.assertEqual(
            free.plan_stats["complete_matching_set_count"], 0
        )
        self.assertEqual(
            free.plan_stats["pose_optimization_count"], 0
        )
        self.assertEqual(free.search_nodes, 0)
        self.assertTrue(free.plan_stats["enumeration_skipped"])
        self.assertFalse(
            free.plan_stats["safety_gates_applied"]
        )
        self.assertEqual(len(free.operations), 4)
        self.assertEqual(free.plan_stats["top_k"], [])
        self.assertIn(
            "FREE_FIXED_TEMPLATE_CHECK,matched=1,"
            "layout=MIRROR_X",
            output.getvalue(),
        )
        self.assertIn(
            "FREE_FIXED_TEMPLATE_BYPASS,enumeration=SKIPPED,"
            "safety_gates=SKIPPED",
            output.getvalue(),
        )

    def test_cached_fixed_match_is_reused_by_planner(self):
        _payload, pieces = _load_frame_33()
        evaluation = match_fixed_figure2_piece_set(pieces)
        self.assertIsNotNone(evaluation[0])
        original = free_planner._free_figure2_match

        def unexpected_recalculation(*_args, **_kwargs):
            raise AssertionError("fixed template match recalculated")

        free_planner._free_figure2_match = unexpected_recalculation
        try:
            result = plan_simulator_free_rectangle(
                pieces,
                fixed_template_evaluation=evaluation,
            )
        finally:
            free_planner._free_figure2_match = original
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(
            result.mode, free_planner.FIGURE2_DIRECT_MODE
        )

    def test_timed_board_geometry_prefers_requested_aspect_band(self):
        rows = [
            (
                "P1",
                3072.3,
                (156.82, 56.83),
                [
                    (113.11, 33.64),
                    (180.41, 107.47),
                    (194.66, 98.26),
                    (156.08, 12.84),
                ],
            ),
            (
                "P2",
                1401.7,
                (62.02, 87.95),
                [
                    (50.08, 105.11),
                    (82.86, 115.08),
                    (84.61, 98.70),
                    (57.87, 60.20),
                    (41.70, 57.40),
                ],
            ),
            (
                "P3",
                1107.5,
                (93.78, 41.55),
                [
                    (64.45, 34.08),
                    (106.53, 64.62),
                    (119.69, 54.44),
                    (91.63, 19.03),
                ],
            ),
            (
                "P4",
                495.0,
                (137.52, 105.20),
                [
                    (128.46, 126.59),
                    (152.13, 107.11),
                    (131.96, 81.89),
                ],
            ),
        ]
        pieces = [
            PieceObservation(
                piece_id,
                [],
                polygon,
                centroid_mm=center,
                area_mm2=area,
            )
            for piece_id, area, center, polygon in rows
        ]
        cfg.FREE_RECT_MAX_PLAN_TIME_MS = 0
        # The exposed-perimeter prefilter cheaply skips the early open chains;
        # retain enough matching sets to reach a closed, preferred-aspect set.
        cfg.FREE_RECT_MAX_COMPLETE_SETS = 3000
        with contextlib.redirect_stdout(io.StringIO()):
            result = plan_simulator_free_rectangle(pieces)
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.plan_stats["aspect_preferred"])
        self.assertGreaterEqual(
            result.plan_stats["aspect_ratio"],
            cfg.FREE_RECT_PREFERRED_ASPECT_MIN,
        )
        self.assertLessEqual(
            result.plan_stats["aspect_ratio"],
            cfg.FREE_RECT_PREFERRED_ASPECT_MAX,
        )

    def test_logged_open_chain_is_filtered_before_pose_optimization(self):
        rows = [
            (
                "P1",
                2912.5,
                (61.02, 91.52),
                [
                    (28.36, 46.09),
                    (57.68, 135.23),
                    (100.69, 116.97),
                    (42.96, 40.28),
                ],
            ),
            (
                "P2",
                1358.3,
                (100.20, 61.17),
                [
                    (67.95, 49.13),
                    (98.64, 81.44),
                    (131.52, 71.70),
                    (123.19, 55.77),
                    (81.54, 41.16),
                ],
            ),
            (
                "P3",
                1131.5,
                (134.39, 104.45),
                [
                    (105.99, 94.24),
                    (142.05, 128.36),
                    (157.39, 119.51),
                    (138.43, 82.55),
                ],
            ),
            (
                "P4",
                515.9,
                (102.00, 19.77),
                [
                    (78.48, 21.25),
                    (121.00, 33.20),
                    (106.53, 4.87),
                ],
            ),
        ]
        pieces = [
            PieceObservation(
                piece_id,
                [],
                polygon,
                centroid_mm=center,
                area_mm2=area,
            )
            for piece_id, area, center, polygon in rows
        ]
        cfg.FREE_RECT_MAX_PLAN_TIME_MS = 0
        cfg.FREE_RECT_MAX_COMPLETE_SETS = 6000
        with contextlib.redirect_stdout(io.StringIO()):
            result = plan_simulator_free_rectangle(
                pieces,
                fixed_template_evaluation=(None, "logged generic input"),
            )
        self.assertTrue(result.valid, result.reason)
        stats = result.plan_stats
        self.assertGreater(
            stats["perimeter_prefilter_rejected_count"], 5000
        )
        self.assertLess(stats["pose_optimization_count"], 150)
        self.assertGreater(stats["additional_contact_count"], 0)
        self.assertLessEqual(stats["perimeter_error_ratio"], 0.02)
        self.assertGreaterEqual(
            stats["exposed_perimeter_mm"],
            stats["expected_perimeter_min_mm"],
        )
        self.assertLessEqual(
            stats["exposed_perimeter_mm"],
            stats["expected_perimeter_max_mm"],
        )
        self.assertLess(stats["long_side_mm"], 105.0)
        self.assertLess(stats["short_side_mm"], 70.0)

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

    def test_removed_planner_modes_have_no_config_switches(self):
        for name in (
            "PLANNER_BACKEND",
            "TARGET_RECT_SIZE_MM",
            "MAX_PLAN_TIME_MS",
            "SIMULATOR_MAX_MATCHING_SETS",
            "PREFER_OUTER_FIRST_PLANNER",
            "FIXED_RECT_BEAM_WIDTH",
        ):
            self.assertFalse(hasattr(cfg, name), name)
        self.assertEqual(cfg.SIMULATOR_MAX_CANDIDATES, 80)
        self.assertEqual(cfg.MIN_PIECE_COUNT, 4)
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
                'ACTIVE_PLANNER = "simulator_free_rect"', bundle
            )
            self.assertIn(
                "def plan_simulator_free_rectangle", bundle
            )
            for removed in (
                "def plan_simulator_rectangle",
                "def plan_outer_first_rectangle",
                "def plan_rectangle_assembly",
                "def detect_pieces_from_gray",
                "class PlacementMonitor",
            ):
                self.assertNotIn(removed, bundle)
            self.assertIn(
                "FREE_FIXED_TEMPLATE_BYPASS,"
                "enumeration=SKIPPED",
                bundle,
            )
            self.assertIn(
                "template_role={}", bundle
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
