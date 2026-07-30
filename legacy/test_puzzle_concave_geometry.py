"""AABB, triangulation, and concave-overlap geometry tests."""

import math
import unittest

from puzzle_geometry import (
    PieceObservation,
    aabb_overlap,
    geometry_counters_snapshot,
    polygon_aabb,
    polygon_area,
    polygon_is_convex,
    polygon_is_simple,
    polygon_overlap_area,
    reset_geometry_counters,
    transform_polygon,
    triangulate_simple_polygon,
)


class ConcaveGeometryTests(unittest.TestCase):
    def setUp(self):
        self.concave = [
            (0, 0),
            (6, 0),
            (6, 6),
            (3, 3),
            (0, 6),
        ]

    def test_aabb_separation_contact_overlap_and_containment(self):
        box = (0.0, 0.0, 5.0, 5.0)
        self.assertFalse(aabb_overlap(box, (6, 0, 8, 2)))
        self.assertTrue(aabb_overlap(box, (5, 1, 7, 3)))
        self.assertTrue(aabb_overlap(box, (4, 4, 8, 8)))
        self.assertTrue(aabb_overlap(box, (1, 1, 2, 2)))
        self.assertTrue(
            aabb_overlap(box, (5.04, 0, 8, 2), tolerance=0.05)
        )

    def test_rotated_aabb(self):
        square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        angle = math.radians(45.0)
        transformed = transform_polygon(
            square,
            (
                math.cos(angle),
                math.sin(angle),
                0.0,
                0.0,
                45.0,
            ),
        )
        aabb = polygon_aabb(transformed)
        extent = math.sqrt(2.0)
        self.assertAlmostEqual(aabb[0], -extent, places=7)
        self.assertAlmostEqual(aabb[2], extent, places=7)

    def test_ear_clipping_area_for_both_orientations(self):
        for polygon in (self.concave, list(reversed(self.concave))):
            triangles = triangulate_simple_polygon(polygon)
            self.assertEqual(len(triangles), 3)
            self.assertAlmostEqual(
                sum(polygon_area(item) for item in triangles),
                polygon_area(polygon),
                places=7,
            )
        self.assertFalse(polygon_is_convex(self.concave))

    def test_self_intersection_is_rejected(self):
        bow_tie = [(0, 0), (5, 5), (0, 5), (5, 0)]
        self.assertFalse(polygon_is_simple(bow_tie))
        with self.assertRaises(ValueError):
            triangulate_simple_polygon(bow_tie)
        with self.assertRaises(ValueError):
            PieceObservation("bad", [], bow_tie)

    def test_concave_overlap_area(self):
        covering = [(-1, -1), (7, -1), (7, 7), (-1, 7)]
        lower_left = [(0, 0), (3, 0), (3, 3), (0, 3)]
        self.assertAlmostEqual(
            polygon_overlap_area(self.concave, covering),
            polygon_area(self.concave),
            places=7,
        )
        self.assertAlmostEqual(
            polygon_overlap_area(self.concave, lower_left),
            9.0,
            places=7,
        )

    def test_shared_edge_has_zero_overlap(self):
        left = [(0, 0), (5, 0), (5, 5), (0, 5)]
        right = [(5, 0), (10, 0), (10, 5), (5, 5)]
        self.assertEqual(polygon_overlap_area(left, right), 0.0)

    def test_aabb_reject_skips_exact_intersection(self):
        reset_geometry_counters()
        first = [(0, 0), (4, 0), (4, 4), (0, 4)]
        second = [(10, 0), (14, 0), (14, 4), (10, 4)]
        self.assertEqual(polygon_overlap_area(first, second), 0.0)
        counters = geometry_counters_snapshot()
        self.assertGreaterEqual(counters["aabb_reject_count"], 1)
        self.assertEqual(counters["polygon_intersection_calls"], 0)


if __name__ == "__main__":
    unittest.main()
