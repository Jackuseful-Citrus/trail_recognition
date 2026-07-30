"""Regression tests for the pure-Python simulator-compatible planner."""

import math
import unittest

from puzzle_geometry import PieceObservation
from puzzle_simulator_planner import (
    UPSTREAM_REVISION,
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
                abs(match[0] - 0.15) < 1e-9
                and (
                    abs(match[6] - 0.4) < 1e-9
                    or abs(match[8] - 0.4) < 1e-9
                )
                for match in partial
            )
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

    def test_upstream_validation_exposes_but_labels_bad_proposal(self):
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
        self.assertTrue(proposal.valid)
        self.assertIn("local warnings", proposal.reason)
        self.assertTrue(proposal.plan_stats["local_gate_failures"])

    def test_module_has_no_desktop_numeric_dependency(self):
        with open("puzzle_simulator_planner.py", encoding="utf-8") as source:
            text = source.read()
        self.assertNotIn("import numpy", text)
        self.assertNotIn("import cv2", text)


if __name__ == "__main__":
    unittest.main()
