"""Regression tests for the pure-Python simulator-compatible planner."""

import math
import unittest

from puzzle_geometry import PieceObservation
from puzzle_geometry import polygon_centroid
from puzzle_perf import ticks_ms
from puzzle_simulator_planner import (
    UPSTREAM_REVISION,
    _sim_matching_sets,
    _sim_new_search_state,
    plan_simulator_rectangle,
    simulator_candidate_matchings,
)


def _scatter(polygons):
    angles = (17.0, -31.0, 48.0, -12.0)
    offsets = ((18.0, 12.0), (72.0, 18.0), (20.0, 82.0), (79.0, 88.0))
    pieces = []
    for index, polygon in enumerate(polygons):
        angle = math.radians(angles[index])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        offset = offsets[index]
        moved = [
            (
                cosine * point[0] - sine * point[1] + offset[0],
                sine * point[0] + cosine * point[1] + offset[1],
            )
            for point in polygon
        ]
        pieces.append(
            PieceObservation(
                "P{}".format(index + 1),
                [],
                moved,
                confidence=1.0,
            )
        )
    return pieces


def _upstream_fixture(mode):
    try:
        import numpy as np
        from puzzle_vision_simulator_bridge.upstream_loader import (
            load_upstream,
        )
    except (ImportError, OSError) as exc:
        raise unittest.SkipTest(str(exc))
    upstream = load_upstream()
    source = upstream.random_cut(
        np.random.default_rng(123), 4, mode
    )
    # Upstream uses 4 pixels per physical millimetre.
    return [
        [(float(point[0]) / 4.0, float(point[1]) / 4.0) for point in polygon]
        for polygon in source
    ]


class SimulatorCompatiblePlannerTests(unittest.TestCase):
    def test_candidate_shortlist_contains_t_junction_partial_match(self):
        polygons = [
            [(0.0, 0.0), (100.0, 0.0), (100.0, 20.0), (0.0, 20.0)],
            [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)],
        ]
        candidates = simulator_candidate_matchings(polygons)
        partial = [
            match
            for match in candidates
            if match[5:] != (0.0, 1.0, 0.0, 1.0)
        ]
        self.assertTrue(partial)
        self.assertTrue(
            any(
                (
                    abs(match[6] - 0.4) < 1e-9
                    or abs(match[8] - 0.4) < 1e-9
                )
                for match in partial
            )
        )
        # Partial candidates are no longer tied at a fixed 0.15 penalty.
        self.assertGreater(
            len(set(round(match[0], 6) for match in partial)),
            1,
        )

    def test_four_piece_auto_search_covers_every_partial_mix(self):
        polygons = [
            [
                (0.0, 0.0),
                (0.0, 60.0),
                (25.0, 60.0),
                (25.0, 0.0),
            ]
            for _ in range(4)
        ]
        candidates = []
        for index in range(3):
            candidates.append(
                (
                    0.01,
                    index,
                    2,
                    index + 1,
                    0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                (
                    0.20,
                    index,
                    2,
                    index + 1,
                    0,
                    0.0,
                    1.0,
                    0.0,
                    0.99,
                )
            )
        candidates.sort()
        state = _sim_new_search_state(ticks_ms())
        results = list(
            _sim_matching_sets(
                candidates,
                4,
                "auto",
                state,
                polygons=polygons,
                target=(100.0, 60.0),
            )
        )
        self.assertTrue(results)
        self.assertEqual(
            set(state["matching_topology_counts"]),
            {
                "3_full",
                "2_full_1_partial",
                "1_full_2_partial",
                "3_partial",
            },
        )

    def test_known_area_normalization_precedes_candidates(self):
        scaled = []
        for polygon in _upstream_fixture("boundary_fan"):
            center = polygon_centroid(polygon)
            scaled.append(
                [
                    (
                        center[0] + (point[0] - center[0]) * 1.028,
                        center[1] + (point[1] - center[1]) * 1.028,
                    )
                    for point in polygon
                ]
            )
        plan = plan_simulator_rectangle(
            _scatter(scaled),
            validation="local",
        )
        self.assertTrue(plan.valid, plan.reason)
        self.assertAlmostEqual(
            plan.plan_stats["target_area_scale"],
            1.0 / 1.028,
            places=6,
        )

    def test_all_upstream_cut_families_reassemble(self):
        modes = (
            "common",
            "boundary_fan",
            "strips",
            "equal_rectangles",
            "t_junction",
            "corner",
            "concave",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                pieces = _scatter(_upstream_fixture(mode))
                plan = plan_simulator_rectangle(
                    pieces,
                    cut_mode=mode,
                    validation="local",
                )
                self.assertTrue(plan.valid, plan.reason)
                self.assertEqual(len(plan.operations), 4)
                self.assertAlmostEqual(plan.fill_gap_mm2, 0.0, places=4)
                self.assertAlmostEqual(plan.overlap_mm2, 0.0, places=4)
                self.assertEqual(
                    plan.plan_stats["upstream_commit"],
                    UPSTREAM_REVISION,
                )
                if mode == "t_junction":
                    self.assertEqual(
                        plan.plan_stats[
                            "selected_partial_match_count"
                        ],
                        1,
                    )

    def test_upstream_validation_rejects_catastrophic_proposal(self):
        pieces = _scatter(
            [
                [(0.0, 0.0), (45.0, 0.0), (45.0, 60.0), (0.0, 60.0)],
                [(0.0, 0.0), (45.0, 0.0), (45.0, 60.0), (0.0, 60.0)],
            ]
        )
        local = plan_simulator_rectangle(
            pieces, validation="local"
        )
        proposal = plan_simulator_rectangle(
            pieces, validation="upstream"
        )
        self.assertFalse(local.valid)
        self.assertIn("local gates", local.reason)
        self.assertFalse(proposal.valid)
        self.assertIn("physical safety gates", proposal.reason)
        self.assertTrue(proposal.plan_stats["local_gate_failures"])
        self.assertTrue(proposal.plan_stats["safety_gate_failures"])

    def test_upstream_validation_normalizes_bounded_area_bias(self):
        pieces = _scatter(
            [
                [(0.0, 0.0), (47.0, 0.0), (47.0, 60.0), (0.0, 60.0)],
                [(0.0, 0.0), (47.0, 0.0), (47.0, 60.0), (0.0, 60.0)],
            ]
        )
        proposal = plan_simulator_rectangle(
            pieces, validation="upstream"
        )
        self.assertTrue(proposal.valid, proposal.reason)
        self.assertAlmostEqual(
            proposal.plan_stats["target_area_scale"],
            math.sqrt(6000.0 / 5640.0),
            places=6,
        )
        self.assertFalse(proposal.plan_stats["local_gate_failures"])
        self.assertFalse(
            proposal.plan_stats["safety_gate_failures"]
        )

    def test_upstream_accepts_confirmed_t_junction_raster_plan(self):
        # Board log 2026-07-30: the displayed topology and 103.5x59.7 mm
        # outline were confirmed correct. Raster vertex fitting leaves about
        # 86 mm2 overlap at the partial-edge junction, which is a warning but
        # not a reason to keep the realtime state machine in ACQUIRE.
        polygons = [
            [
                (77.32, 109.05),
                (129.16, 79.79),
                (86.99, 6.21),
            ],
            [
                (8.79, 39.01),
                (52.72, 129.44),
                (57.11, 99.30),
                (33.39, 25.71),
            ],
            [
                (156.40, 119.69),
                (191.55, 132.99),
                (196.82, 123.23),
                (138.83, 72.70),
            ],
            [
                (145.86, 46.10),
                (164.31, 58.51),
                (173.10, 39.90),
                (139.71, 28.37),
            ],
        ]
        pieces = [
            PieceObservation(
                "P{}".format(index + 1),
                [],
                polygon,
                confidence=1.0,
            )
            for index, polygon in enumerate(polygons)
        ]
        plan = plan_simulator_rectangle(
            pieces, validation="upstream"
        )
        self.assertTrue(plan.valid, plan.reason)
        self.assertLess(plan.plan_stats["dimension_error_mm"], 4.0)
        self.assertGreater(plan.overlap_mm2, 80.0)
        self.assertLess(plan.overlap_mm2, 100.0)
        self.assertIn("local warnings", plan.reason)
        self.assertFalse(plan.plan_stats["safety_gate_failures"])
        self.assertGreater(
            plan.plan_stats["matching_prefixes_evaluated"],
            plan.plan_stats["matching_sets_evaluated"],
        )
        self.assertLess(
            plan.plan_stats["matching_prefixes_evaluated"],
            2000,
        )
        for name in (
            "matching_pruned_dimension",
            "matching_pruned_outside",
            "matching_pruned_overlap",
            "matching_pruned_gap",
        ):
            self.assertGreater(plan.plan_stats[name], 0)

    def test_module_has_no_desktop_numeric_dependency(self):
        with open("puzzle_simulator_planner.py", encoding="utf-8") as source:
            text = source.read()
        self.assertNotIn("import numpy", text)
        self.assertNotIn("import cv2", text)


if __name__ == "__main__":
    unittest.main()
