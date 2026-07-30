"""Regression coverage for the external-solver bridge."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import puzzle_config as cfg
from puzzle_geometry import PlanResult
from puzzle_vision import detect_pieces_from_gray

from .adapter import (
    plan_with_upstream,
    plan_with_upstream_then_outer_fallback,
)
from .upstream_loader import (
    PINNED_COMMIT,
    UpstreamUnavailableError,
    load_upstream,
)


class UpstreamBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.upstream = load_upstream()
        except UpstreamUnavailableError as exc:
            raise unittest.SkipTest(str(exc))
        gray = cv2.imread(cfg.OFFLINE_IMAGE, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise unittest.SkipTest("sample image unavailable")
        cls.pieces, _ = detect_pieces_from_gray(
            gray,
            cfg.OFFLINE_A4_CORNERS_PX,
            cv2,
            np,
        )
        for index, piece in enumerate(cls.pieces):
            piece.piece_id = "P{}".format(index + 1)

    def test_loader_uses_pinned_revision(self):
        self.assertEqual(
            self.upstream.__name__,
            "_puzzle_vision_simulator_{}".format(PINNED_COMMIT[:12]),
        )

    def test_upstream_result_maps_to_local_plan_schema(self):
        plan = plan_with_upstream(
            self.pieces,
            validation="upstream",
        )
        self.assertIsInstance(plan, PlanResult)
        self.assertTrue(plan.valid, plan.reason)
        self.assertEqual(len(plan.operations), len(self.pieces))
        self.assertEqual(set(plan.target_polygons), {"P1", "P2", "P3", "P4"})
        self.assertEqual(tuple(plan.target_rect), (55.0, 195.0, 155.0, 255.0))
        self.assertTrue(
            all(
                len(operation["matrix_3x3_mm"]) == 3
                for operation in plan.operations
            )
        )

    def test_local_gates_reject_noisy_fast_proposal(self):
        plan = plan_with_upstream(
            self.pieces,
            validation="local",
        )
        self.assertFalse(plan.valid)
        self.assertIn("local gates", plan.reason)
        self.assertTrue(plan.plan_stats["local_gate_failures"])

    def test_outer_fallback_preserves_safe_plan_contract(self):
        plan = plan_with_upstream_then_outer_fallback(self.pieces)
        self.assertTrue(plan.valid, plan.reason)
        self.assertTrue(plan.mode.startswith("bridge_outer_fallback/"))
        self.assertIn("bridge_upstream_proposal", plan.plan_stats)


if __name__ == "__main__":
    unittest.main()
