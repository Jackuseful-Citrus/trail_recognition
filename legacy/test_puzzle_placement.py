"""Tests for frozen-plan, per-piece closed-loop placement monitoring."""

import math
import unittest

import puzzle_config as cfg
from puzzle_geometry import (
    PieceObservation,
    PlanResult,
    polygon_centroid,
    transform_polygon,
)
from puzzle_placement import (
    PlacementMonitor,
    _shape_cost,
    cyclic_vertex_error,
    final_rectangle_consensus,
    final_rectangle_metrics,
    match_remaining_pieces,
    placement_contour_error,
    placement_delta_metrics,
    placement_pose_error_bound,
    resample_closed_polygon,
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

        # Simulate P2 touching P1 and merging into one white component.
        result = monitor.check(
            [],
            delta_metrics={
                "added_target_coverage": 0.90,
                "added_area_ratio": 0.96,
                "added_spill_ratio": 0.04,
                "removed_source_ratio": 0.62,
            },
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

    def test_vertex_count_change_keeps_shape_and_contour_matching(self):
        rectangle = PieceObservation(
            "P1",
            [],
            [(0, 0), (50, 0), (50, 30), (0, 30)],
            rotation_ambiguous=False,
        )
        redundant_five = PieceObservation(
            "",
            [],
            [(0, 0), (25, 0), (50, 0), (50, 30), (0, 30)],
            rotation_ambiguous=False,
        )
        self.assertIsNotNone(_shape_cost(rectangle, redundant_five))
        self.assertIsNotNone(_shape_cost(redundant_five, rectangle))
        error = cyclic_vertex_error(
            redundant_five.polygon_mm, rectangle.polygon_mm
        )
        self.assertIsNotNone(error)
        self.assertLess(error["rms_mm"], 0.01)
        operation = {
            "piece_id": "P1",
            "source_center_mm": rectangle.centroid_mm,
            "target_center_mm": rectangle.centroid_mm,
            "rotation_deg": 0.0,
            "rotation_ambiguous": False,
            "confidence": 1.0,
        }
        plan = PlanResult(
            valid=True,
            operations=[operation],
            target_polygons={"P1": rectangle.polygon_mm},
            target_rect=(0.0, 0.0, 50.0, 30.0),
        )
        verification = PlacementMonitor(
            [rectangle], plan
        ).verify_next_piece([redundant_five])
        self.assertTrue(verification["placed"])
        self.assertTrue(
            verification["placed_by_resampled_contour"]
        )
        self.assertTrue(verification["placed_by_pose_bound"])

    def test_resampled_contour_is_order_and_vertex_count_invariant(self):
        rectangle = [(0, 0), (50, 0), (50, 30), (0, 30)]
        redundant = [
            (50, 30),
            (25, 30),
            (0, 30),
            (0, 0),
            (50, 0),
        ]
        reversed_shifted = list(reversed(redundant[2:] + redundant[:2]))
        samples = resample_closed_polygon(rectangle, 32)
        self.assertEqual(len(samples), 32)
        for candidate in (redundant, reversed_shifted):
            error = placement_contour_error(rectangle, candidate)
            self.assertLess(error["rms_mm"], 0.01)
            self.assertLess(error["p95_mm"], 0.01)
        wrong = [(15, 15), (65, 15), (65, 45), (15, 45)]
        error = placement_contour_error(rectangle, wrong)
        self.assertGreater(error["rms_mm"], 8.0)

    def test_pose_error_bound_has_eight_mm_boundary(self):
        reference = PieceObservation(
            "P1",
            [],
            [(-30, -8), (30, -8), (24, 8), (-30, 8)],
            rotation_ambiguous=False,
        )
        operation = {
            "target_center_mm": (100.0, 200.0),
            "rotation_deg": 0.0,
        }
        inside_dx = 106.0 - reference.centroid_mm[0]
        target_dy = 200.0 - reference.centroid_mm[1]
        inside = PieceObservation(
            "",
            [],
            [
                (x + inside_dx, y + target_dy)
                for x, y in reference.polygon_mm
            ],
            current_orientation_deg=reference.current_orientation_deg,
            rotation_ambiguous=False,
        )
        inside_metrics = placement_pose_error_bound(
            reference, inside, operation
        )
        self.assertAlmostEqual(
            inside_metrics["pose_error_bound_mm"], 6.0, places=5
        )
        self.assertLessEqual(
            inside_metrics["pose_error_bound_mm"], 8.0
        )
        outside_dx = 109.0 - reference.centroid_mm[0]
        outside = PieceObservation(
            "",
            [],
            [
                (x + outside_dx, y + target_dy)
                for x, y in reference.polygon_mm
            ],
            current_orientation_deg=reference.current_orientation_deg,
            rotation_ambiguous=False,
        )
        outside_metrics = placement_pose_error_bound(
            reference, outside, operation
        )
        self.assertGreater(
            outside_metrics["pose_error_bound_mm"], 8.0
        )

        radius = max(
            math.dist(point, reference.centroid_mm)
            for point in reference.polygon_mm
        )

        def rotated_observation(angle_deg, center_error):
            angle = math.radians(angle_deg)
            cosine = math.cos(angle)
            sine = math.sin(angle)
            polygon = []
            for point in reference.polygon_mm:
                dx = point[0] - reference.centroid_mm[0]
                dy = point[1] - reference.centroid_mm[1]
                polygon.append(
                    (
                        100.0
                        + center_error
                        + cosine * dx
                        - sine * dy,
                        200.0 + sine * dx + cosine * dy,
                    )
                )
            return PieceObservation(
                "",
                [],
                polygon,
                current_orientation_deg=(
                    reference.current_orientation_deg + angle_deg
                ),
                rotation_ambiguous=False,
            )

        angled = placement_pose_error_bound(
            reference,
            rotated_observation(8.0, 2.0),
            operation,
        )
        expected_bound = 2.0 + 2.0 * radius * math.sin(
            math.radians(8.0) / 2.0
        )
        self.assertAlmostEqual(
            angled["pose_error_bound_mm"],
            expected_bound,
            places=5,
        )
        self.assertLess(expected_bound, 8.0)
        too_far = placement_pose_error_bound(
            reference,
            rotated_observation(14.0, 2.0),
            operation,
        )
        self.assertGreater(
            too_far["pose_error_bound_mm"], 8.0
        )

    def test_first_piece_can_pass_delta_coverage(self):
        monitor = PlacementMonitor(self.references, self.plan)
        result = monitor.check(
            [],
            delta_metrics={
                "added_target_coverage": 0.81,
                "added_area_ratio": 0.92,
                "added_spill_ratio": 0.08,
                "removed_source_ratio": None,
            },
        )
        self.assertEqual(result["newly_completed"], ["P1"])
        metrics = result["metrics"]["P1"]
        self.assertEqual(metrics["method"], "delta_coverage")
        self.assertFalse(metrics["source_removal_support"])

    def test_absolute_coverage_alone_never_confirms(self):
        monitor = PlacementMonitor(self.references, self.plan)
        result = monitor.check([], coverages={"P1": 0.99})
        self.assertEqual(result["newly_completed"], [])
        self.assertFalse(result["done"])

    def test_motion_delta_measures_target_add_source_remove_and_spill(self):
        width = 210
        height = 297
        before = bytearray(width * height)
        after = bytearray(width * height)
        for y in range(10, 30):
            for x in range(10, 40):
                before[y * width + x] = 1
        for y in range(200, 220):
            for x in range(100, 130):
                after[y * width + x] = 1
        metrics = placement_delta_metrics(
            before,
            after,
            width,
            height,
            [(100, 200), (130, 200), (130, 220), (100, 220)],
            [(10, 10), (40, 10), (40, 30), (10, 30)],
            600.0,
        )
        self.assertAlmostEqual(
            metrics["added_target_coverage"], 1.0
        )
        self.assertAlmostEqual(metrics["added_area_ratio"], 1.0)
        self.assertAlmostEqual(metrics["added_spill_ratio"], 0.0)
        self.assertAlmostEqual(
            metrics["removed_source_ratio"], 1.0
        )

    @staticmethod
    def _rectangle_mask(
        rect=(55, 190, 155, 250),
        width=210,
        height=297,
    ):
        mask = bytearray(width * height)
        for y in range(rect[1], rect[3]):
            for x in range(rect[0], rect[2]):
                mask[y * width + x] = 1
        return mask

    def test_final_rectangle_gate_accepts_fill_area_bbox_and_spill(self):
        target = (55.0, 190.0, 155.0, 250.0)
        metrics = final_rectangle_metrics(
            self._rectangle_mask(),
            210,
            297,
            target,
            6000.0,
        )
        self.assertTrue(metrics["valid"], metrics)
        self.assertAlmostEqual(metrics["fill_ratio"], 1.0)
        self.assertAlmostEqual(metrics["final_area_ratio"], 1.0)
        self.assertAlmostEqual(metrics["spill_ratio"], 0.0)
        self.assertAlmostEqual(metrics["detected_width_mm"], 100.0)
        self.assertAlmostEqual(metrics["detected_height_mm"], 60.0)

        relaxed_area = final_rectangle_metrics(
            self._rectangle_mask((65, 190, 155, 250)),
            210,
            297,
            target,
            6000.0,
        )
        self.assertAlmostEqual(
            relaxed_area["final_area_ratio"], 0.9
        )
        self.assertTrue(relaxed_area["valid"], relaxed_area)

        rotated = final_rectangle_metrics(
            self._rectangle_mask((75, 170, 135, 270)),
            210,
            297,
            target,
            6000.0,
        )
        self.assertTrue(rotated["valid"], rotated)
        self.assertTrue(rotated["dimensions_swapped"])

    def test_final_rectangle_rejects_hole_and_external_spill(self):
        width = 210
        height = 297
        target = (55.0, 190.0, 155.0, 250.0)
        hole = self._rectangle_mask()
        for y in range(205, 235):
            for x in range(75, 135):
                hole[y * width + x] = 0
        # Preserve total area with foreground elsewhere inside the 20 mm
        # envelope; fill must still reject the large central hole.
        for y in range(170, 185):
            for x in range(45, 165):
                hole[y * width + x] = 1
        hole_metrics = final_rectangle_metrics(
            hole, width, height, target, 6000.0
        )
        self.assertGreaterEqual(
            hole_metrics["final_area_ratio"],
            cfg.FINAL_AREA_RATIO_MIN,
        )
        self.assertFalse(hole_metrics["valid"])
        self.assertLess(
            hole_metrics["fill_ratio"], cfg.FINAL_RECT_FILL_MIN
        )

        spill = self._rectangle_mask()
        for y in range(190, 210):
            for x in range(55, 105):
                spill[y * width + x] = 0
        for y in range(190, 240):
            for x in range(0, 20):
                spill[y * width + x] = 1
        spill_metrics = final_rectangle_metrics(
            spill, width, height, target, 6000.0
        )
        self.assertFalse(spill_metrics["valid"])
        self.assertGreater(
            spill_metrics["spill_ratio"], cfg.FINAL_RECT_SPILL_MAX
        )

    def test_final_rectangle_consensus_requires_two_of_three(self):
        passing = {"valid": True}
        failing = {"valid": False}
        metric_defaults = {
            "fill_ratio": 0.8,
            "final_area_ratio": 1.0,
            "detected_width_mm": 100.0,
            "detected_height_mm": 60.0,
            "width_error_mm": 0.0,
            "height_error_mm": 0.0,
            "spill_ratio": 0.02,
        }
        passing.update(metric_defaults)
        failing.update(metric_defaults)
        accepted = final_rectangle_consensus(
            [passing, failing, passing]
        )
        self.assertTrue(accepted["valid"])
        self.assertEqual(accepted["pass_count"], 2)
        rejected = final_rectangle_consensus(
            [failing, passing, failing]
        )
        self.assertFalse(rejected["valid"])
        self.assertEqual(rejected["pass_count"], 1)
        too_few = final_rectangle_consensus([passing, passing])
        self.assertFalse(too_few["valid"])


if __name__ == "__main__":
    unittest.main()
