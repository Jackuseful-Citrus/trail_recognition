"""Desktop unit tests for the shared pure-Python geometry core."""

import math
import unittest

import puzzle_config as cfg
from puzzle_geometry import (
    PieceObservation,
    PieceTracker,
    _known_target_gate_reason,
    edge_lengths,
    interior_angles,
    match_piece_across_frames,
    normalize_angle_deg,
    plan_outer_first_rectangle,
    plan_rectangle_assembly,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
    polygon_orientation,
    polygon_shape_signature,
    polygon_signed_area,
    polygons_overlap,
    remove_near_collinear_vertices,
    rigid_transform_from_edge_pair,
    score_rectangle_assembly,
    transform_point,
    transform_polygon,
)


class GeometryTests(unittest.TestCase):
    def test_area_orientation_and_centroid(self):
        rectangle = [(0, 0), (100, 0), (100, 60), (0, 60)]
        self.assertAlmostEqual(polygon_signed_area(rectangle), 6000.0)
        self.assertAlmostEqual(polygon_area(rectangle), 6000.0)
        self.assertEqual(polygon_orientation(rectangle), "CCW")
        self.assertEqual(polygon_orientation(list(reversed(rectangle))), "CW")
        cx, cy = polygon_centroid(rectangle)
        self.assertAlmostEqual(cx, 50.0)
        self.assertAlmostEqual(cy, 30.0)

    def test_centroid_rejects_degenerate_polygon(self):
        with self.assertRaises(ValueError):
            polygon_centroid([(0, 0), (1, 0), (2, 0)])

    def test_collinear_removal(self):
        polygon = [(0, 0), (50, 0), (100, 0), (100, 60), (0, 60)]
        cleaned = remove_near_collinear_vertices(
            polygon, tolerance_deg=2.0, min_edge_mm=0.0
        )
        self.assertEqual(len(cleaned), 4)
        self.assertAlmostEqual(polygon_area(cleaned), 6000.0)

    def test_close_chamfer_vertices_are_fitted_to_one_corner(self):
        # A reflective/missing 3 mm corner appears as a short diagonal edge.
        polygon = [(0, 3), (3, 0), (40, 0), (40, 30), (0, 30)]
        cleaned = remove_near_collinear_vertices(polygon)
        self.assertEqual(len(cleaned), 4)
        self.assertAlmostEqual(polygon_area(cleaned), 1200.0)
        self.assertTrue(
            any(
                abs(x) < 1e-6 and abs(y) < 1e-6
                for x, y in cleaned
            )
        )

    def test_vertex_cleanup_is_independent_of_ring_order(self):
        polygon = [
            (0, 0),
            (20, 0.7),
            (40, 0),
            (40, 30),
            (0, 30),
        ]
        variants = []
        for source in (polygon, list(reversed(polygon))):
            for offset in range(len(source)):
                variants.append(source[offset:] + source[:offset])
        cleaned = [
            remove_near_collinear_vertices(value)
            for value in variants
        ]
        self.assertTrue(all(len(value) == 4 for value in cleaned))
        for value in cleaned:
            self.assertAlmostEqual(
                polygon_area(value),
                polygon_area(cleaned[0]),
                places=6,
            )

    def test_cleanup_preserves_real_short_corner_and_concavity(self):
        # The current prototype has a legitimate edge near 10 mm, shorter than
        # the contest's guaranteed 20 mm minimum but safely above the 7 mm
        # reflection-defect merge radius.
        real_short_edge = [
            (0, 0),
            (10, 0),
            (10, 20),
            (40, 20),
            (40, 40),
            (0, 40),
        ]
        concave = [
            (0, 0),
            (50, 0),
            (50, 40),
            (30, 20),
            (0, 40),
        ]
        self.assertEqual(
            len(remove_near_collinear_vertices(real_short_edge)),
            6,
        )
        self.assertEqual(
            len(remove_near_collinear_vertices(concave)),
            5,
        )
        self.assertAlmostEqual(
            polygon_area(
                remove_near_collinear_vertices(concave)
            ),
            polygon_area(concave),
        )

    def test_edges_and_angles(self):
        rectangle = [(0, 0), (100, 0), (100, 60), (0, 60)]
        self.assertEqual(
            [round(value) for value in edge_lengths(rectangle)],
            [100, 60, 100, 60],
        )
        for angle in interior_angles(rectangle):
            self.assertAlmostEqual(angle, 90.0)

    def test_normalize_angle(self):
        self.assertEqual(normalize_angle_deg(180), -180.0)
        self.assertEqual(normalize_angle_deg(540), -180.0)
        self.assertEqual(normalize_angle_deg(-181), 179.0)

    def test_rigid_edge_transform(self):
        edge_a = ((10, 10), (30, 10))
        edge_b = ((2, 5), (2, 25))
        transform = rigid_transform_from_edge_pair(edge_a, edge_b)
        mapped_b0 = transform_point(edge_b[0], transform)
        mapped_b1 = transform_point(edge_b[1], transform)
        self.assertAlmostEqual(mapped_b0[0], edge_a[1][0], places=6)
        self.assertAlmostEqual(mapped_b0[1], edge_a[1][1], places=6)
        self.assertAlmostEqual(mapped_b1[0], edge_a[0][0], places=6)
        self.assertAlmostEqual(mapped_b1[1], edge_a[0][1], places=6)

    def test_point_and_overlap(self):
        a = [(0, 0), (10, 0), (10, 10), (0, 10)]
        b = [(9, 2), (12, 2), (12, 5), (9, 5)]
        touching = [(10, 0), (20, 0), (20, 10), (10, 10)]
        self.assertTrue(point_in_polygon((5, 5), a))
        self.assertTrue(point_in_polygon((0, 5), a))
        self.assertFalse(point_in_polygon((11, 5), a))
        self.assertTrue(polygons_overlap(a, b))
        self.assertFalse(polygons_overlap(a, touching))

    def test_shape_signature_is_rigid_invariant(self):
        polygon = [(0, 0), (60, 0), (45, 30), (0, 20)]
        angle = math.radians(37)
        transform = (
            math.cos(angle),
            math.sin(angle),
            91.0,
            -23.0,
            37.0,
        )
        transformed = transform_polygon(polygon, transform)
        self.assertEqual(
            polygon_shape_signature(polygon),
            polygon_shape_signature(transformed),
        )

    def test_frame_matching_and_tracker_stability(self):
        polygon = [(0, 0), (50, 0), (35, 30), (0, 20)]
        previous = PieceObservation("", [], polygon, confidence=0.9)
        current = PieceObservation(
            "",
            [],
            [(x + 0.4, y - 0.3) for x, y in polygon],
            confidence=0.9,
        )
        matches, cost = match_piece_across_frames(previous, current)
        self.assertTrue(matches)
        self.assertLess(cost, cfg.TRACK_SHAPE_COST_LIMIT)

        tracker = PieceTracker()
        stable = False
        for _ in range(cfg.REQUIRED_STABLE_FRAMES + 1):
            observation = PieceObservation("", [], polygon, confidence=0.9)
            observations, stable = tracker.update([observation])
        # One piece is geometrically stable but the planner requires 2..4.
        self.assertFalse(stable)
        self.assertTrue(observations[0].stable)
        self.assertEqual(observations[0].piece_id, "P1")

    def test_tracker_tolerates_one_transient_extra_vertex(self):
        rectangle = [(0, 0), (40, 0), (40, 30), (0, 30)]
        chamfered = [
            (0, 0),
            (36, 0),
            (40, 4),
            (40, 30),
            (0, 30),
        ]
        fixed_triangle = [
            (85, 70),
            (125, 70),
            (105, 105),
        ]
        previous = PieceObservation("", [], rectangle)
        current = PieceObservation("", [], chamfered)
        matches, cost = match_piece_across_frames(
            previous, current
        )
        self.assertTrue(matches)
        self.assertLess(cost, cfg.TRACK_SHAPE_COST_LIMIT)

        tracker = PieceTracker()
        stable = False
        output = []
        varying = [
            rectangle,
            chamfered,
            rectangle,
            chamfered,
            rectangle,
            chamfered,
            rectangle,
            rectangle,
        ]
        for polygon in varying:
            output, stable = tracker.update(
                [
                    PieceObservation("", [], polygon),
                    PieceObservation("", [], fixed_triangle),
                ]
            )
        self.assertTrue(stable)
        self.assertEqual(
            [piece.piece_id for piece in output],
            ["P1", "P2"],
        )
        # Planning receives the modal four-corner observation, rather than
        # whichever noisy fit happened to arrive on the final frame.
        self.assertEqual(len(output[0].polygon_mm), 4)

    def test_tracker_holds_ids_and_history_through_count_dip(self):
        polygons = [
            [(0, 0), (40, 0), (35, 22), (0, 20)],
            [(60, 0), (92, 0), (88, 25), (62, 28)],
            [(0, 55), (35, 52), (18, 82)],
            [(65, 55), (105, 55), (103, 82), (66, 85)],
        ]
        tracker = PieceTracker(expected_count=4)
        output = []
        for _ in range(cfg.REQUIRED_STABLE_FRAMES + 1):
            output, stable = tracker.update(
                [
                    PieceObservation("", [], polygon)
                    for polygon in polygons
                ]
            )
        self.assertTrue(stable)
        original_ids = [piece.piece_id for piece in output]
        history_lengths = {
            track.piece_id: len(track.history)
            for track in tracker.tracks
        }

        partial, stable = tracker.update(
            [
                PieceObservation("", [], polygon)
                for polygon in polygons[:3]
            ]
        )
        self.assertFalse(stable)
        self.assertEqual(
            [piece.piece_id for piece in partial],
            original_ids[:3],
        )
        missing_track = next(
            track
            for track in tracker.tracks
            if track.piece_id == original_ids[3]
        )
        self.assertEqual(
            len(missing_track.history),
            history_lengths[original_ids[3]],
        )
        self.assertTrue(missing_track.stable)

        recovered, stable = tracker.update(
            [
                PieceObservation("", [], polygon)
                for polygon in polygons
            ]
        )
        self.assertTrue(stable)
        self.assertEqual(
            [piece.piece_id for piece in recovered], original_ids
        )
        self.assertLessEqual(tracker.next_id, 5)

    def test_rectangle_score(self):
        rectangle = [(0, 0), (100, 0), (100, 60), (0, 60)]
        self.assertAlmostEqual(score_rectangle_assembly([rectangle]), 0.0)

    def test_known_target_gate_rejects_logged_open_fan_plan(self):
        metrics = {
            "score": 0.2609,
            "fill_gap_mm2": 2197.3,
            "overlap_mm2": 0.0,
            "outside_mm2": 0.0,
        }
        reason = _known_target_gate_reason(
            metrics,
            105.3,
            80.0,
            (100.0, 60.0),
        )
        self.assertIsNotNone(reason)
        self.assertIn("actual=105.3x80.0", reason)
        self.assertIn("gap=2197.3", reason)

        accepted_metrics = {
            "score": 0.0318,
            "fill_gap_mm2": 85.7,
            "overlap_mm2": 4.2,
            "outside_mm2": 84.2,
        }
        self.assertIsNone(
            _known_target_gate_reason(
                accepted_metrics,
                100.0,
                60.0,
                (100.0, 60.0),
            )
        )

    def test_four_piece_rectangle_planning(self):
        source_polygons = [
            [(0, 0), (50, 30), (0, 60)],
            [(0, 0), (100, 0), (50, 30)],
            [(100, 0), (100, 60), (50, 30)],
            [(100, 60), (0, 60), (50, 30)],
        ]
        transforms = []
        for index, angle_deg in enumerate((20, -35, 75, 130)):
            angle = math.radians(angle_deg)
            transforms.append(
                (
                    math.cos(angle),
                    math.sin(angle),
                    20.0 + index * 35.0,
                    30.0 + index * 17.0,
                    angle_deg,
                )
            )
        pieces = []
        for index, polygon in enumerate(source_polygons):
            pieces.append(
                PieceObservation(
                    "P{}".format(index + 1),
                    [],
                    transform_polygon(polygon, transforms[index]),
                    confidence=0.95,
                )
            )
        plan = plan_rectangle_assembly(pieces)
        self.assertTrue(plan.valid, plan.reason)
        self.assertLess(plan.score, 1e-5)
        self.assertEqual(len(plan.operations), 4)
        self.assertEqual(len(plan.target_polygons), 4)
        min_x, min_y, max_x, max_y = plan.target_rect
        self.assertGreaterEqual(min_x, cfg.TARGET_MARGIN_MM)
        self.assertGreaterEqual(
            min_y, cfg.DIVIDER_Y_MM + cfg.TARGET_MARGIN_MM
        )
        self.assertLessEqual(max_x, cfg.A4_WIDTH_MM - cfg.TARGET_MARGIN_MM)
        self.assertLessEqual(max_y, cfg.A4_HEIGHT_MM - cfg.TARGET_MARGIN_MM)

    def test_known_target_corrects_small_common_scale_bias(self):
        source_polygons = [
            [(0, 0), (50, 30), (0, 60)],
            [(0, 0), (100, 0), (50, 30)],
            [(100, 0), (100, 60), (50, 30)],
            [(100, 60), (0, 60), (50, 30)],
        ]
        pieces = []
        for index, polygon in enumerate(source_polygons):
            center = polygon_centroid(polygon)
            scaled = [
                (
                    center[0] + (point[0] - center[0]) * 1.02,
                    center[1] + (point[1] - center[1]) * 1.02,
                )
                for point in polygon
            ]
            angle_deg = (20.0, -35.0, 75.0, 130.0)[index]
            angle = math.radians(angle_deg)
            pieces.append(
                PieceObservation(
                    "S{}".format(index + 1),
                    [],
                    transform_polygon(
                        scaled,
                        (
                            math.cos(angle),
                            math.sin(angle),
                            20.0 + index * 35.0,
                            30.0 + index * 17.0,
                            angle_deg,
                        ),
                    ),
                    confidence=0.95,
                )
            )
        plan = plan_outer_first_rectangle(
            pieces,
            target_size_mm=(100.0, 60.0),
        )
        self.assertTrue(plan.valid, plan.reason)
        self.assertAlmostEqual(
            plan.plan_stats["target_area_scale"],
            1.0 / 1.02,
            places=6,
        )
        width = plan.target_rect[2] - plan.target_rect[0]
        height = plan.target_rect[3] - plan.target_rect[1]
        self.assertEqual(
            sorted((round(width, 1), round(height, 1))),
            [60.0, 100.0],
        )

    def test_outer_first_unknown_sizes_and_piece_counts(self):
        cases = [
            (
                110.0,
                70.0,
                [
                    [(0, 0), (110, 0), (0, 70)],
                    [(110, 0), (110, 70), (0, 70)],
                ],
            ),
            (
                115.0,
                75.0,
                [
                    [(0, 0), (35, 0), (35, 75), (0, 75)],
                    [(35, 0), (75, 0), (75, 75), (35, 75)],
                    [(75, 0), (115, 0), (115, 75), (75, 75)],
                ],
            ),
            (
                115.0,
                85.0,
                [
                    [(0, 0), (57, 41), (0, 85)],
                    [(0, 0), (115, 0), (57, 41)],
                    [(115, 0), (115, 85), (57, 41)],
                    [(115, 85), (0, 85), (57, 41)],
                ],
            ),
        ]
        rotations = (23.0, -41.0, 76.0, 137.0)
        for expected_width, expected_height, polygons in cases:
            pieces = []
            for index, polygon in enumerate(polygons):
                angle = math.radians(rotations[index])
                transform = (
                    math.cos(angle),
                    math.sin(angle),
                    12.0 + index * 31.0,
                    18.0 + index * 22.0,
                    rotations[index],
                )
                pieces.append(
                    PieceObservation(
                        "U{}".format(index + 1),
                        [],
                        transform_polygon(polygon, transform),
                        confidence=0.9,
                    )
                )
            pieces.reverse()
            plan = plan_outer_first_rectangle(pieces)
            self.assertTrue(plan.valid, plan.reason)
            self.assertIn(
                plan.mode,
                ("corner_outer_strict", "outer_first_strict"),
            )
            self.assertLessEqual(plan.search_nodes, 200)
            self.assertEqual(len(plan.operations), len(pieces))
            width = plan.target_rect[2] - plan.target_rect[0]
            height = plan.target_rect[3] - plan.target_rect[1]
            self.assertEqual(
                sorted((round(width, 3), round(height, 3))),
                sorted((expected_width, expected_height)),
            )

        # This valid fan partition has no approximately right-angle vertex in
        # any individual piece: rectangle corners are split between pieces.
        fan_angles = [
            angle
            for polygon in cases[-1][2]
            for angle in interior_angles(polygon)
        ]
        self.assertTrue(
            all(abs(angle - 90.0) > 10.0 for angle in fan_angles)
        )

    def test_outer_first_tolerates_small_vertex_measurement_error(self):
        base = [
            [(0, 0), (57, 41), (0, 85)],
            [(0, 0), (115, 0), (57, 41)],
            [(115, 0), (115, 85), (57, 41)],
            [(115, 85), (0, 85), (57, 41)],
        ]
        offsets = [
            [(-1.0, 0.7), (0.8, -0.5), (-0.6, 1.2)],
            [(0.5, -0.9), (1.1, 0.6), (-0.7, -0.4)],
            [(-0.8, 0.5), (0.6, -1.1), (1.2, 0.3)],
            [(0.9, -0.6), (-1.2, 0.8), (0.4, -0.9)],
        ]
        rotations = (23.0, -41.0, 76.0, 137.0)
        pieces = []
        for index, polygon in enumerate(base):
            noisy = [
                (point[0] + delta[0], point[1] + delta[1])
                for point, delta in zip(polygon, offsets[index])
            ]
            angle = math.radians(rotations[index])
            pieces.append(
                PieceObservation(
                    "N{}".format(index + 1),
                    [],
                    transform_polygon(
                        noisy,
                        (
                            math.cos(angle),
                            math.sin(angle),
                            10.0 + index * 30.0,
                            20.0 + index * 17.0,
                            rotations[index],
                        ),
                    ),
                    confidence=0.9,
                )
            )
        pieces = [pieces[2], pieces[0], pieces[3], pieces[1]]
        plan = plan_outer_first_rectangle(pieces)
        self.assertTrue(plan.valid, plan.reason)
        self.assertEqual(plan.mode, "outer_first_tolerant")
        self.assertLessEqual(
            plan.search_nodes, cfg.OUTER_FIRST_MAX_SEARCH_NODES
        )
        self.assertLess(plan.max_vertex_error_mm, 2.0)
        self.assertLess(plan.score, 0.05)

    def test_outer_first_failure_explains_target_geometry_mismatch(self):
        pieces = [
            PieceObservation(
                "P1",
                [],
                [(18.5, 92.2), (82.6, 138.0), (54.5, 40.8)],
            ),
            PieceObservation(
                "P2",
                [],
                [
                    (90.5, 39.0),
                    (164.3, 53.2),
                    (169.6, 22.2),
                    (67.7, 18.6),
                ],
            ),
            PieceObservation(
                "P3",
                [],
                [
                    (90.5, 89.5),
                    (137.9, 98.4),
                    (170.5, 73.6),
                    (163.4, 63.8),
                ],
            ),
            PieceObservation(
                "P4",
                [],
                [
                    (151.1, 118.8),
                    (137.1, 135.6),
                    (173.1, 135.6),
                    (172.2, 115.3),
                ],
            ),
        ]
        plan = plan_outer_first_rectangle(
            pieces,
            target_size_mm=(100.0, 60.0),
        )
        self.assertFalse(plan.valid)
        self.assertIn(
            "complete candidates miss target", plan.reason
        )
        self.assertIn("closest=", plan.reason)
        self.assertGreater(
            plan.plan_stats["complete_state_count"], 0
        )
        self.assertGreater(
            plan.plan_stats["pruned_target_dimension"], 0
        )
        self.assertGreater(
            plan.plan_stats[
                "closest_target_dimension_error_mm"
            ],
            cfg.KNOWN_TARGET_DIMENSION_TOLERANCE_MM,
        )
        self.assertIn(
            "corner_failure_reason", plan.plan_stats
        )
        self.assertEqual(
            plan.plan_stats["target_area_mm2"], 6000.0
        )
        self.assertLess(
            plan.plan_stats["input_area_error_pct"], 6.0
        )


if __name__ == "__main__":
    unittest.main()
