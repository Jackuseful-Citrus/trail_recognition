import ast
import contextlib
import io
import json
import math
from pathlib import Path
import unittest

import puzzle_config as cfg
from puzzle_geometry import (
    PieceObservation,
    _identity_transform,
    polygon_area,
    transform_point,
    transform_polygon,
)
import puzzle_simulator_free_rect_planner as planner
from puzzle_simulator_free_rect_planner import (
    FIGURE2_DIRECT_MODE,
    FIGURE2_TEMPLATE_ORDER,
    FIGURE2_TEMPLATE_POLYGONS,
    plan_simulator_free_rectangle,
)
from puzzle_simulator_planner import (
    _sim_align_edge,
    _sim_align_segment_midpoint,
)
import source_projective_piece_detector as source_detector


ROOT = Path(__file__).resolve().parent
BOARD_FIXTURE = (
    ROOT / "fixtures" / "non_figure2_board_short_edge_regression.json"
)
OPTIMIZED_ARTIFACT = (
    ROOT
    / "k230_realtime_a4"
    / "k230_realtime_a4_simulator_free_rect_source_projective_optimized_no_uart_standalone.py"
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
    return [
        PieceObservation(
            "P{}".format(index + 1),
            [],
            _rigid_transform(
                polygon, angles[index], translations[index]
            ),
            confidence=1.0,
        )
        for index, polygon in enumerate(partition)
    ]


def _plan_quiet(pieces, fixed_template_evaluation=None):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = plan_simulator_free_rectangle(
            pieces,
            fixed_template_evaluation=fixed_template_evaluation,
        )
    return result, output.getvalue()


def _strip_partition(long_side, short_side, count):
    result = []
    for index in range(count):
        left = long_side * index / count
        right = long_side * (index + 1) / count
        result.append(
            [
                (left, 0.0),
                (right, 0.0),
                (right, short_side),
                (left, short_side),
            ]
        )
    return result


def _board_tree_starvation_pieces():
    polygons = [
        [(15.48, 27.80), (20.55, 66.21), (102.36, 84.76), (95.44, 18.75)],
        [
            (130.76, 22.89),
            (116.57, 66.22),
            (143.31, 92.51),
            (181.07, 51.76),
            (148.13, 13.35),
        ],
        [(125.12, 109.74), (124.10, 129.31), (173.11, 119.41), (166.67, 96.84)],
        [(61.49, 129.37), (93.28, 122.30), (70.70, 85.59)],
    ]
    return [
        PieceObservation(
            "P{}".format(index + 1), [], polygon, confidence=1.0
        )
        for index, polygon in enumerate(polygons)
    ]


def _board_partial_endpoint_reserve_pieces():
    polygons = [
        [(15.85, 27.74), (19.92, 66.57), (102.33, 85.72), (95.46, 19.49)],
        [
            (131.04, 22.97),
            (116.68, 66.21),
            (144.80, 93.84),
            (181.37, 52.93),
            (148.42, 13.81),
        ],
        [(125.69, 109.82), (123.98, 129.95), (172.31, 120.15), (165.66, 96.14)],
        [(61.39, 129.90), (93.89, 122.88), (70.60, 86.21)],
    ]
    return [
        PieceObservation(
            "P{}".format(index + 1), [], polygon, confidence=1.0
        )
        for index, polygon in enumerate(polygons)
    ]


def _assert_publish_gates(testcase, result):
    testcase.assertTrue(result.valid, result.reason)
    stats = result.plan_stats
    testcase.assertGreaterEqual(
        stats["long_side_mm"], cfg.FREE_RECT_PUBLISH_LONG_MIN_MM
    )
    testcase.assertLessEqual(
        stats["long_side_mm"], cfg.FREE_RECT_PUBLISH_LONG_MAX_MM
    )
    testcase.assertGreaterEqual(
        stats["short_side_mm"], cfg.FREE_RECT_PUBLISH_SHORT_MIN_MM
    )
    testcase.assertLessEqual(
        stats["short_side_mm"], cfg.FREE_RECT_PUBLISH_SHORT_MAX_MM
    )
    testcase.assertLessEqual(
        stats["area_prior_error"], cfg.FREE_RECT_PUBLISH_AREA_ERROR_MAX
    )
    testcase.assertLessEqual(
        stats["overlap_ratio"], cfg.FREE_RECT_PUBLISH_OVERLAP_RATIO_MAX
    )
    testcase.assertLessEqual(
        stats["fill_gap_ratio"], cfg.FREE_RECT_PUBLISH_FILL_GAP_RATIO_MAX
    )
    testcase.assertEqual(stats["outer_piece_missing_count"], 0)


class SourcePolygonRefitTests(unittest.TestCase):
    def test_short_edge_retry_uses_same_boundary_and_keeps_area(self):
        boundary = [
            (0.0, 0.0),
            (25.0, 0.0),
            (50.0, 0.0),
            (75.0, 0.0),
            (100.0, 0.0),
            (100.0, 30.0),
            (100.0, 60.0),
            (75.0, 60.0),
            (50.0, 60.0),
            (25.0, 60.0),
            (0.0, 60.0),
            (0.0, 30.0),
        ]
        initial = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 60.0),
            (10.0, 60.0),
            (0.0, 55.0),
        ]
        repaired = [(0.0, 0.0), (100.0, 0.0), (100.0, 60.0), (0.0, 60.0)]
        original = source_detector._source_fit_polygon_candidate
        calls = []

        def fake_fit(points, tolerance, fit_diagnostics=None):
            calls.append((points, tolerance))
            return repaired

        source_detector._source_fit_polygon_candidate = fake_fit
        try:
            polygon, record = source_detector._source_refit_short_edge(
                boundary, initial
            )
        finally:
            source_detector._source_fit_polygon_candidate = original
        self.assertEqual(polygon, repaired)
        self.assertEqual(record["method"], "dp_retry_1")
        self.assertIs(calls[0][0], boundary)
        self.assertGreaterEqual(
            record["final_min_edge_mm"], cfg.FREE_RECT_MIN_OBSERVED_EDGE_MM
        )
        self.assertLess(
            abs(polygon_area(polygon) - polygon_area(initial))
            / polygon_area(initial),
            cfg.SOURCE_PROJECTIVE_REFIT_MAX_AREA_CHANGE_RATIO,
        )

    def test_unresolved_short_edge_is_retained_for_fixed_template(self):
        initial = [
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 60.0),
            (10.0, 60.0),
            (0.0, 55.0),
        ]
        original = source_detector._source_fit_polygon_candidate
        source_detector._source_fit_polygon_candidate = (
            lambda points, tolerance, fit_diagnostics=None: initial
        )
        try:
            polygon, record = source_detector._source_refit_short_edge(
                list(initial), initial
            )
        finally:
            source_detector._source_fit_polygon_candidate = original
        self.assertEqual(polygon, initial)
        self.assertEqual(record["method"], "unresolved_kept")
        self.assertEqual(record["final_vertices"], len(initial))
        self.assertAlmostEqual(
            record["final_min_edge_mm"],
            source_detector._source_polygon_min_edge_mm(initial),
        )

    def test_real_fixed_template_short_edge_reaches_direct_match(self):
        polygons = []
        methods = []
        for role in FIGURE2_TEMPLATE_ORDER:
            initial = list(FIGURE2_TEMPLATE_POLYGONS[role])
            polygon, record = source_detector._source_refit_short_edge(
                list(initial), initial
            )
            polygons.append(polygon)
            methods.append(record["method"])
        self.assertEqual(
            methods[FIGURE2_TEMPLATE_ORDER.index("MIDDLE_LEFT")],
            "unresolved_kept",
        )
        pieces = [
            PieceObservation(
                "P{}".format(index + 1),
                [],
                polygon,
                confidence=1.0,
            )
            for index, polygon in enumerate(polygons)
        ]
        result, log = _plan_quiet(pieces)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.mode, FIGURE2_DIRECT_MODE)
        self.assertIn("FREE_FIXED_TEMPLATE_CHECK,matched=1", log)
        self.assertNotIn("FREE_INPUT_REJECT", log)


class SearchPrimitiveTests(unittest.TestCase):
    def test_labeled_spanning_tree_counts(self):
        self.assertEqual(len(planner._free_labeled_spanning_trees(1)), 1)
        self.assertEqual(len(planner._free_labeled_spanning_trees(2)), 1)
        self.assertEqual(len(planner._free_labeled_spanning_trees(3)), 3)
        self.assertEqual(len(planner._free_labeled_spanning_trees(4)), 16)

    def test_shortlist_is_capped_per_piece_pair(self):
        pieces = _make_pieces(_strip_partition(100.0, 60.0, 4))
        candidates, details = planner._free_rect_candidate_matchings(
            pieces,
            full_rel_tolerance=0.20,
            partial_enabled=True,
            partial_min=0.15,
            partial_max=0.92,
            return_details=True,
        )
        self.assertLessEqual(len(candidates), 6 * (8 + 4))
        self.assertEqual(details["candidate_pair_group_count"], 6)
        self.assertEqual(len(details["candidate_cache"]), len(candidates))
        for full_count, partial_count in details["pair_counts"].values():
            self.assertLessEqual(full_count, cfg.FREE_RECT_PAIR_MAX_FULL)
            self.assertLessEqual(partial_count, cfg.FREE_RECT_PAIR_MAX_PARTIAL)

    def test_nonoverlapping_partial_intervals_share_one_edge(self):
        first = (0.15, 0, 1, 1, 3, 0.0, 0.40, 0.0, 1.0)
        touching = (0.15, 0, 1, 2, 3, 0.40, 1.0, 0.0, 1.0)
        overlapping = (0.15, 0, 1, 2, 3, 0.35, 0.80, 0.0, 1.0)
        occupied = planner._free_add_candidate_intervals({}, first)
        self.assertIsNotNone(occupied)
        self.assertIsNotNone(
            planner._free_add_candidate_intervals(occupied, touching)
        )
        self.assertIsNone(
            planner._free_add_candidate_intervals(occupied, overlapping)
        )

        state = planner._free_new_state(planner.ticks_ms())
        sets = list(
            planner._free_tree_matching_sets(
                {(0, 1): [first], (0, 2): [touching]},
                3,
                {2},
                state,
                (planner.ticks_ms(), 1000),
            )
        )
        self.assertEqual(len(sets), 1)
        fallback = planner._free_pass_definitions(3)[-1]
        self.assertEqual(fallback["name"], "multi_partial_fallback")
        self.assertEqual(fallback["allowed_partial_counts"], (2, 3))

    def test_midpoint_alignment_minimizes_two_endpoint_residual(self):
        source_a, source_b = (0.0, 0.0), (12.0, 0.0)
        target_a, target_b = (10.0, 5.0), (0.0, 5.0)
        endpoint = _sim_align_edge(
            source_a, source_b, target_a, target_b
        )
        midpoint = _sim_align_segment_midpoint(
            source_a, source_b, target_a, target_b
        )

        def squared_error(transform):
            mapped_a = transform_point(source_a, transform)
            mapped_b = transform_point(source_b, transform)
            return sum(
                (left[index] - right[index]) ** 2
                for left, right in (
                    (mapped_a, target_a),
                    (mapped_b, target_b),
                )
                for index in (0, 1)
            )

        self.assertLess(squared_error(midpoint), squared_error(endpoint))
        self.assertAlmostEqual(
            midpoint[0] * midpoint[0] + midpoint[1] * midpoint[1],
            1.0,
            places=12,
        )

    def test_board_tree_starvation_is_fixed_with_253_cheap_results(self):
        pieces = _board_tree_starvation_pieces()
        polygons = [piece.polygon_mm for piece in pieces]
        source_area = sum(piece.area_mm2 for piece in pieces)
        candidates, details = planner._free_rect_candidate_matchings(
            pieces,
            full_rel_tolerance=0.12,
            partial_enabled=True,
            partial_min=0.22,
            partial_max=0.88,
            return_details=True,
        )
        state = planner._free_new_state(planner.ticks_ms())
        context = planner._free_perimeter_context(
            polygons, source_area
        )
        beams = {}
        cheap_count = 0
        matching_sets = planner._free_tree_matching_sets(
            planner._free_candidates_by_pair(candidates),
            4,
            {0, 1},
            state,
            (planner.ticks_ms(), 100000),
            candidate_cache=details["candidate_cache"],
        )
        for matches in matching_sets:
            transforms = planner._free_initial_transforms(
                polygons, matches, alignment="midpoint"
            )
            cheap = planner._free_cheap_complete_metrics(
                polygons,
                matches,
                transforms,
                source_area,
                context,
                candidate_cache=details["candidate_cache"],
            )
            if cheap is None:
                continue
            cheap_count += 1
            partial_count = sum(
                1
                for match in matches
                if not planner._sim_is_full_match(match)
            )
            planner._free_add_to_cheap_beam(
                beams,
                {
                    "matches": matches,
                    "match_signature": planner._free_match_signature(
                        matches
                    ),
                    "transforms": transforms,
                    "partial_count": partial_count,
                    "topology": planner._free_topology_name(
                        3, 3 - partial_count
                    ),
                    "cheap_metrics": cheap,
                    "pass_name": "standard_t",
                },
            )
            if cheap_count >= 253:
                break

        cache = planner._free_build_piece_cache(pieces)
        valid = []
        for item in planner._free_merge_cheap_beams(beams):
            proposal = planner._free_exact_proposal(
                item,
                pieces,
                polygons,
                cache,
                source_area,
                context,
                state,
            )
            if proposal is not None and proposal["physical_valid"]:
                valid.append(proposal)
        self.assertEqual(cheap_count, 253)
        self.assertEqual(state["trees_considered"], 16)
        self.assertGreaterEqual(state["tree_schedule_count"], 64)
        self.assertTrue(valid)
        self.assertAlmostEqual(
            valid[0]["metrics"]["long_side_mm"], 111.01, places=1
        )
        self.assertAlmostEqual(
            valid[0]["metrics"]["short_side_mm"], 80.47, places=1
        )

    def test_board_partial_endpoint_is_kept_and_found_with_238_results(self):
        pieces = _board_partial_endpoint_reserve_pieces()
        polygons = [piece.polygon_mm for piece in pieces]
        source_area = sum(piece.area_mm2 for piece in pieces)
        candidates, details = planner._free_rect_candidate_matchings(
            pieces,
            full_rel_tolerance=0.12,
            partial_enabled=True,
            partial_min=0.22,
            partial_max=0.88,
            return_details=True,
        )
        target_partial = [
            candidate
            for candidate in candidates
            if candidate[1:5] == (0, 1, 3, 2)
            and candidate[5] == 0.0
        ]
        self.assertTrue(target_partial)

        state = planner._free_new_state(planner.ticks_ms())
        context = planner._free_perimeter_context(
            polygons, source_area
        )
        beams = {}
        cheap_count = 0
        matching_sets = planner._free_tree_matching_sets(
            planner._free_candidates_by_pair(candidates),
            4,
            {0, 1},
            state,
            (planner.ticks_ms(), 100000),
            candidate_cache=details["candidate_cache"],
        )
        for matches in matching_sets:
            transforms = planner._free_initial_transforms(
                polygons, matches, alignment="midpoint"
            )
            cheap = planner._free_cheap_complete_metrics(
                polygons,
                matches,
                transforms,
                source_area,
                context,
                candidate_cache=details["candidate_cache"],
            )
            if cheap is None:
                continue
            cheap_count += 1
            partial_count = sum(
                1
                for match in matches
                if not planner._sim_is_full_match(match)
            )
            planner._free_add_to_cheap_beam(
                beams,
                {
                    "matches": matches,
                    "match_signature": planner._free_match_signature(
                        matches
                    ),
                    "transforms": transforms,
                    "partial_count": partial_count,
                    "topology": planner._free_topology_name(
                        3, 3 - partial_count
                    ),
                    "cheap_metrics": cheap,
                    "pass_name": "standard_t",
                },
            )
            if cheap_count >= 238:
                break

        cache = planner._free_build_piece_cache(pieces)
        valid = []
        for item in planner._free_merge_cheap_beams(beams):
            proposal = planner._free_exact_proposal(
                item,
                pieces,
                polygons,
                cache,
                source_area,
                context,
                state,
            )
            if proposal is not None and proposal["physical_valid"]:
                valid.append(proposal)
        self.assertEqual(cheap_count, 238)
        self.assertEqual(state["trees_considered"], 16)
        self.assertTrue(valid)
        self.assertAlmostEqual(
            valid[0]["metrics"]["long_side_mm"], 112.04, places=1
        )
        self.assertAlmostEqual(
            valid[0]["metrics"]["short_side_mm"], 80.44, places=1
        )


class OptimizedPlannerTests(unittest.TestCase):
    def test_board_fixture_fails_closed_before_bad_publish(self):
        payload = json.loads(BOARD_FIXTURE.read_text(encoding="utf-8"))
        pieces = [
            PieceObservation(
                record["piece_id"], [], record["polygon_mm"], confidence=1.0
            )
            for record in payload["pieces"]
        ]
        result, log = _plan_quiet(pieces)
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "short_edge_unresolved")
        self.assertEqual(result.operations, [])
        self.assertEqual(result.plan_stats["pose_optimization_count"], 0)
        self.assertIn("FREE_INPUT_REJECT,reason=short_edge_unresolved", log)
        self.assertIn("FREE_FIXED_TEMPLATE_CHECK,matched=0", log)
        self.assertNotIn("FREE_PLAN_RESULT,valid=1", log)

    def test_required_free_rectangle_dimensions_and_rigid_outputs(self):
        cases = (
            (90.0, 50.0, 1),
            (90.0, 80.0, 2),
            (100.0, 60.0, 2),
            (110.0, 70.0, 3),
            (120.0, 50.0, 4),
            (120.0, 90.0, 4),
        )
        for long_side, short_side, count in cases:
            with self.subTest(size=(long_side, short_side), pieces=count):
                pieces = _make_pieces(
                    _strip_partition(long_side, short_side, count)
                )
                result, _log = _plan_quiet(
                    pieces,
                    fixed_template_evaluation=(None, "synthetic generic"),
                )
                _assert_publish_gates(self, result)
                self.assertLessEqual(
                    abs(result.plan_stats["long_side_mm"] - long_side), 6.0
                )
                self.assertLessEqual(
                    abs(result.plan_stats["short_side_mm"] - short_side), 6.0
                )
                self.assertEqual(len(result.operations), count)
                for piece in pieces:
                    target_polygon = result.target_polygons[
                        piece.piece_id
                    ]
                    self.assertAlmostEqual(
                        polygon_area(piece.polygon_mm),
                        polygon_area(target_polygon),
                        places=5,
                    )

    def test_tree_search_uses_beam_and_skips_pose_optimizer(self):
        pieces = _make_pieces(_strip_partition(100.0, 60.0, 4))
        result, log = _plan_quiet(
            pieces,
            fixed_template_evaluation=(None, "synthetic generic"),
        )
        _assert_publish_gates(self, result)
        stats = result.plan_stats
        self.assertEqual(stats["candidate_pair_group_count"], 6)
        self.assertGreater(stats["cheap_complete_count"], 100)
        self.assertLess(
            stats["exact_evaluated_count"], stats["cheap_complete_count"]
        )
        self.assertEqual(stats["pose_optimization_count"], 0)
        self.assertEqual(stats["closed_graph_pose_optimizer_count"], 0)
        self.assertEqual(
            stats["tree_pose_optimizer_skipped_count"],
            stats["exact_evaluated_count"],
        )
        self.assertIn("FREE_CANDIDATES,pass=strict_full", log)
        self.assertIn("FREE_PASS_START,name=strict_full", log)
        self.assertIn("FREE_PLAN_RESULT,valid=1", log)

    def test_standard_t_pass_finds_one_partial_rectangle(self):
        # P1's 90 mm right edge is shared by P2/P3 in two intervals.  P2 and
        # P3 also have one full 65 mm seam, so the assembly tree needs exactly
        # one partial edge and cannot be found by strict_full.
        partition = [
            [(0, 0), (45, 0), (45, 90), (0, 90)],
            [(45, 0), (110, 0), (110, 28), (45, 28)],
            [(45, 28), (110, 28), (110, 90), (45, 90)],
        ]
        result, log = _plan_quiet(
            _make_pieces(partition),
            fixed_template_evaluation=(None, "synthetic T junction"),
        )
        _assert_publish_gates(self, result)
        self.assertEqual(result.plan_stats["selected_pass"], "standard_t")
        self.assertEqual(
            result.plan_stats["selected_topology"], "1_full_1_partial"
        )
        self.assertAlmostEqual(result.plan_stats["long_side_mm"], 110.0)
        self.assertAlmostEqual(result.plan_stats["short_side_mm"], 90.0)
        self.assertIn("FREE_PASS_END,name=strict_full", log)
        self.assertIn("FREE_PASS_START,name=standard_t", log)

    def test_radial_sequential_and_concave_cut_families(self):
        families = {
            "radial_corner_triangles": [
                [(0, 0), (120, 0), (60, 45)],
                [(120, 0), (120, 90), (60, 45)],
                [(120, 90), (0, 90), (60, 45)],
                [(0, 90), (0, 0), (60, 45)],
            ],
            "sequential_t_junctions": [
                [(0, 0), (35, 0), (35, 90), (0, 90)],
                [(35, 0), (120, 0), (120, 30), (35, 30)],
                [(35, 30), (75, 30), (75, 90), (35, 90)],
                [(75, 30), (120, 30), (120, 90), (75, 90)],
            ],
            "concave_polyline": [
                [(0, 0), (60, 0), (60, 30), (30, 30), (0, 60)],
                [(60, 0), (100, 0), (100, 30), (60, 30)],
                [(30, 30), (60, 30), (100, 30), (100, 60), (0, 60)],
            ],
        }
        for name, partition in families.items():
            with self.subTest(family=name):
                result, _log = _plan_quiet(
                    _make_pieces(partition),
                    fixed_template_evaluation=(None, name),
                )
                _assert_publish_gates(self, result)
                self.assertEqual(result.plan_stats["pose_optimization_count"], 0)

    def test_five_runs_are_deterministic(self):
        pieces = _make_pieces(_strip_partition(90.0, 80.0, 2))
        snapshots = []
        for _index in range(5):
            result, _log = _plan_quiet(
                pieces,
                fixed_template_evaluation=(None, "synthetic generic"),
            )
            snapshots.append(
                (
                    result.plan_stats["selected_pass"],
                    result.plan_stats["selected_topology"],
                    result.plan_stats["selected_seams"],
                    round(result.score, 12),
                    [
                        (
                            operation["piece_id"],
                            round(operation["source_center_mm"][0], 8),
                            round(operation["source_center_mm"][1], 8),
                            round(operation["target_center_mm"][0], 8),
                            round(operation["target_center_mm"][1], 8),
                            round(operation["rotation_deg"], 8),
                        )
                        for operation in result.operations
                    ],
                    result.plan_stats["exact_evaluated_count"],
                    result.plan_stats["cheap_complete_count"],
                )
            )
        self.assertTrue(all(item == snapshots[0] for item in snapshots[1:]))

    def test_fixed_figure2_still_bypasses_generic_search(self):
        partition = [
            list(FIGURE2_TEMPLATE_POLYGONS[role])
            for role in FIGURE2_TEMPLATE_ORDER
        ]
        pieces = _make_pieces(partition)
        first, first_log = _plan_quiet(pieces)
        second, _second_log = _plan_quiet(pieces)
        self.assertTrue(first.valid, first.reason)
        self.assertEqual(first.mode, FIGURE2_DIRECT_MODE)
        self.assertTrue(first.plan_stats["enumeration_skipped"])
        self.assertEqual(first.plan_stats["complete_matching_set_count"], 0)
        self.assertEqual(first.plan_stats["pose_optimization_count"], 0)
        self.assertEqual(first.operations, second.operations)
        self.assertEqual(first.target_polygons, second.target_polygons)
        self.assertIn("FREE_FIXED_TEMPLATE_BYPASS", first_log)
        self.assertNotIn("search=staged_tree_beam", first_log)


class PhysicalGateTests(unittest.TestCase):
    def _metrics(self, polygons):
        source_area = sum(polygon_area(polygon) for polygon in polygons)
        complete = planner._free_complete_metrics(
            polygons,
            (),
            [_identity_transform()] * len(polygons),
            source_area,
        )
        self.assertIsNotNone(complete)
        return complete["metrics"]

    def test_bad_geometries_are_never_physically_publishable(self):
        scenarios = {
            "large_hole": [
                [(0, 0), (20, 0), (20, 20), (0, 20)],
                [(80, 0), (100, 0), (100, 20), (80, 20)],
                [(0, 40), (20, 40), (20, 60), (0, 60)],
                [(80, 40), (100, 40), (100, 60), (80, 60)],
            ],
            "severe_overlap": [
                [(0, 0), (100, 0), (100, 60), (0, 60)],
                [(0, 0), (100, 0), (100, 60), (0, 60)],
            ],
            "l_shape": [
                [(0, 0), (100, 0), (100, 20), (0, 20)],
                [(0, 20), (20, 20), (20, 60), (0, 60)],
            ],
            "internal_piece": [
                [(0, 0), (100, 0), (100, 20), (0, 20)],
                [(0, 40), (100, 40), (100, 60), (0, 60)],
                [(0, 20), (20, 20), (20, 40), (0, 40)],
                [(40, 20), (60, 20), (60, 40), (40, 40)],
            ],
            "dimension_out_of_range": [
                [(0, 0), (130, 0), (130, 50), (0, 50)],
            ],
        }
        for name, polygons in scenarios.items():
            with self.subTest(name=name):
                metrics = self._metrics(polygons)
                physical_valid, failures = planner._free_physical_validity(
                    metrics, object()
                )
                self.assertFalse(physical_valid)
                self.assertTrue(failures)
                if name == "internal_piece":
                    self.assertIn("outer_piece", failures)


class OptimizedStandaloneTests(unittest.TestCase):
    def test_direct_uart2_artifact_contains_optimized_backend(self):
        source = OPTIMIZED_ARTIFACT.read_text(encoding="utf-8")
        ast.parse(source)
        for marker in (
            "FREE_RECT_OPTIMIZED_STAGED_BUILD = True",
            "def _free_labeled_spanning_trees",
            "def _free_partial_patterns",
            "def _free_add_candidate_intervals",
            "def _sim_align_segment_midpoint",
            "def _free_plan_staged_generic",
            "FREE_RECT_PUBLISH_AREA_ERROR_MAX = 0.15",
            "FREE_RECT_TREE_ROUND_ROBIN_QUOTA = 16",
            "FREE_RECT_PAIR_MAX_PARTIAL = 4",
            "UART_COMMUNICATION_ENABLED = False",
            "def _open_uart2_plan_output",
            "def _write_plan_operations_uart2",
            "_UART2_PLAN_TX_PIN = 5",
            "_UART2_PLAN_RX_PIN = 6",
            "UART2_PLAN,piece_id={}",
            "_write_plan_operations_uart2(active_plan)",
        ):
            self.assertIn(marker, source)
        forbidden = ("numpy", "cv2", "scipy", "dataclass")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertFalse(
            [name for name in imported if name.split(".")[0] in forbidden]
        )
        self.assertIn("machine", imported)


if __name__ == "__main__":
    unittest.main()
