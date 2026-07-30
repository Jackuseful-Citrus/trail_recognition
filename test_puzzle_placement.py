"""Tests for frozen-plan, per-piece closed-loop placement monitoring."""

import math
import unittest

from puzzle_geometry import (
    PieceObservation,
    PlanResult,
    polygon_centroid,
    transform_polygon,
)
from puzzle_placement import (
    PlacementMonitor,
    cyclic_vertex_error,
    match_remaining_pieces,
)


def _transform(degrees, tx, ty):
    angle = math.radians(degrees)
    return (
        math.cos(angle),
        math.sin(angle),
        tx,
        ty,
        degrees,
    )


class PlacementMonitorTests(unittest.TestCase):
    def setUp(self):
        self.references = [
            PieceObservation(
                "P1",
                [],
                [(18, 20), (58, 23), (31, 55)],
                confidence=0.95,
            ),
            PieceObservation(
                "P2",
                [],
                [(82, 28), (127, 31), (119, 67), (88, 59)],
                confidence=0.94,
            ),
        ]
        self.target_polygons = {
            "P1": transform_polygon(
                self.references[0].polygon_mm,
                _transform(24.0, 65.0, 165.0),
            ),
            "P2": transform_polygon(
                self.references[1].polygon_mm,
                _transform(-37.0, 62.0, 245.0),
            ),
        }
        operations = []
        for piece in self.references:
            target_center = polygon_centroid(
                self.target_polygons[piece.piece_id]
            )
            operations.append(
                {
                    "piece_id": piece.piece_id,
                    "source_center_mm": piece.centroid_mm,
                    "target_center_mm": target_center,
                    "rotation_deg": 0.0,
                    "rotation_ambiguous": False,
                    "confidence": piece.confidence,
                }
            )
        self.plan = PlanResult(
            valid=True,
            reason="ok",
            score=0.0,
            operations=operations,
            target_polygons=self.target_polygons,
            target_rect=(55.0, 185.0, 155.0, 255.0),
            mode="test",
        )

    def test_shape_assignment_ignores_input_order(self):
        observations = [
            PieceObservation(
                "",
                [],
                transform_polygon(
                    self.references[1].polygon_mm,
                    _transform(73.0, 15.0, 94.0),
                ),
            ),
            PieceObservation(
                "",
                [],
                transform_polygon(
                    self.references[0].polygon_mm,
                    _transform(-51.0, 140.0, 70.0),
                ),
            ),
        ]
        mapping, diagnostics = match_remaining_pieces(
            self.references, observations
        )
        self.assertEqual(set(mapping), {"P1", "P2"})
        self.assertEqual(diagnostics["matched"], 2)
        self.assertEqual(
            len(mapping["P1"].polygon_mm), 3
        )
        self.assertEqual(
            len(mapping["P2"].polygon_mm), 4
        )

    def test_completed_piece_is_retired_then_coverage_finishes_next(self):
        monitor = PlacementMonitor(
            self.references, self.plan
        )
        correct_p1 = PieceObservation(
            "",
            [],
            self.target_polygons["P1"],
            confidence=0.93,
        )
        wrong_p2 = PieceObservation(
            "",
            [],
            transform_polygon(
                self.references[1].polygon_mm,
                _transform(12.0, 12.0, 78.0),
            ),
            confidence=0.91,
        )
        result = monitor.check([wrong_p2, correct_p1])
        self.assertEqual(result["newly_completed"], ["P1"])
        self.assertEqual(result["next_piece_id"], "P2")
        self.assertFalse(result["done"])
        self.assertEqual(
            [piece.piece_id for piece in monitor.visible_pieces()],
            ["P2"],
        )

        # Simulate P2 touching P1 and merging into one white component. With
        # one piece already confirmed, target-region coverage is the fallback.
        result = monitor.check(
            [], coverages={"P2": 0.90}
        )
        self.assertEqual(result["newly_completed"], ["P2"])
        self.assertTrue(result["done"])
        self.assertEqual(monitor.visible_pieces(), [])

    def test_pose_error_is_not_fitted_away(self):
        shifted = [
            (point[0] + 25.0, point[1])
            for point in self.target_polygons["P1"]
        ]
        error = cyclic_vertex_error(
            shifted, self.target_polygons["P1"]
        )
        self.assertAlmostEqual(error["rms_mm"], 25.0)
        self.assertAlmostEqual(error["max_mm"], 25.0)


if __name__ == "__main__":
    unittest.main()
