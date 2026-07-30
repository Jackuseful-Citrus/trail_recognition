"""Ordered boundary tracing and concave-contour regression tests."""

import unittest

import cv2
import numpy as np

from puzzle_geometry import polygon_area, polygon_is_convex
from puzzle_vision import (
    _ordered_contour_polygon,
    trace_ordered_boundary,
)


class OrderedBoundaryTests(unittest.TestCase):
    def _filled(self, points, size=64):
        image = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(
            image,
            [np.array(points, dtype=np.int32)],
            255,
        )
        return image

    def _trace_polygon(self, points, size=64):
        image = self._filled(points, size=size)
        boundary, diagnostics = trace_ordered_boundary(
            image, (0, 0, size, size), 180
        )
        self.assertTrue(diagnostics["ok"], diagnostics)
        self.assertGreaterEqual(len(boundary), 3)
        self.assertLessEqual(
            max(
                abs(boundary[-1][0] - boundary[0][0]),
                abs(boundary[-1][1] - boundary[0][1]),
            ),
            1.0,
        )
        polygon = _ordered_contour_polygon(boundary)
        self.assertIsNotNone(polygon)
        return boundary, polygon, diagnostics

    def test_rectangle_and_triangle(self):
        for points in (
            [(4, 4), (48, 4), (48, 34), (4, 34)],
            [(5, 5), (52, 8), (22, 42)],
        ):
            _, polygon, diagnostics = self._trace_polygon(
                points
            )
            expected = abs(cv2.contourArea(
                np.array(points, dtype=np.float32)
            ))
            self.assertLess(
                abs(polygon_area(polygon) - expected)
                / expected,
                0.08,
            )
            self.assertLess(
                diagnostics["pixel_reads"],
                expected,
            )

    def test_convex_five_sided_piece(self):
        _, polygon, _ = self._trace_polygon(
            [(5, 8), (32, 3), (54, 18), (46, 47), (12, 50)]
        )
        self.assertTrue(polygon_is_convex(polygon))
        self.assertEqual(len(polygon), 5)

    def test_concave_five_sided_piece_keeps_notch(self):
        _, polygon, _ = self._trace_polygon(
            [(4, 4), (54, 4), (54, 50), (29, 27), (4, 50)]
        )
        self.assertEqual(len(polygon), 5)
        self.assertFalse(polygon_is_convex(polygon))
        hull = cv2.convexHull(
            np.array(polygon, dtype=np.float32)
        )
        self.assertGreater(
            cv2.contourArea(hull), polygon_area(polygon)
        )

    def test_image_edge_and_small_hole(self):
        image = self._filled(
            [(0, 0), (48, 0), (48, 44), (0, 44)]
        )
        cv2.rectangle(image, (17, 14), (25, 22), 0, -1)
        boundary, diagnostics = trace_ordered_boundary(
            image, (0, 0, 50, 46), 180
        )
        self.assertTrue(diagnostics["ok"], diagnostics)
        polygon = _ordered_contour_polygon(boundary)
        self.assertIsNotNone(polygon)
        self.assertGreater(polygon_area(polygon), 2000.0)

    def test_connected_noise_spur_is_simplified(self):
        image = self._filled(
            [(6, 6), (50, 6), (50, 42), (6, 42)]
        )
        image[5, 28] = 255
        boundary, diagnostics = trace_ordered_boundary(
            image, (0, 0, 60, 50), 180
        )
        self.assertTrue(diagnostics["ok"], diagnostics)
        polygon = _ordered_contour_polygon(boundary)
        self.assertIsNotNone(polygon)
        self.assertEqual(len(polygon), 4)

    def test_reflective_corner_chamfer_is_fitted_as_one_vertex(self):
        # A small exposed-metal patch can cut the white mask across a corner.
        # The resulting two close chamfer endpoints still describe one
        # physical rectangle vertex.
        _, polygon, _ = self._trace_polygon(
            [(6, 10), (10, 6), (50, 6), (50, 42), (6, 42)]
        )
        self.assertEqual(len(polygon), 4)
        self.assertLess(
            abs(polygon_area(polygon) - 44.0 * 36.0)
            / (44.0 * 36.0),
            0.03,
        )

    def test_small_reflection_bite_does_not_create_piece_vertex(self):
        # Simulate a narrow dark triangular bite in an otherwise straight
        # white edge.
        _, polygon, _ = self._trace_polygon(
            [
                (6, 6),
                (25, 6),
                (27, 10),
                (29, 6),
                (50, 6),
                (50, 42),
                (6, 42),
            ]
        )
        self.assertEqual(len(polygon), 4)

    def test_reversed_contour_has_equivalent_area(self):
        boundary, polygon, _ = self._trace_polygon(
            [(4, 4), (54, 4), (54, 50), (29, 27), (4, 50)]
        )
        reversed_polygon = _ordered_contour_polygon(
            list(reversed(boundary))
        )
        self.assertIsNotNone(reversed_polygon)
        self.assertAlmostEqual(
            polygon_area(polygon),
            polygon_area(reversed_polygon),
            places=5,
        )

    def test_single_pixel_fails_closed(self):
        image = np.zeros((20, 20), dtype=np.uint8)
        image[8, 9] = 255
        boundary, diagnostics = trace_ordered_boundary(
            image, (0, 0, 20, 20), 180
        )
        self.assertFalse(diagnostics["ok"])
        self.assertEqual(diagnostics["reason"], "isolated_pixel")
        self.assertEqual(len(boundary), 1)

    def test_maximum_step_guard(self):
        image = self._filled(
            [(3, 3), (52, 3), (52, 48), (3, 48)]
        )
        _, diagnostics = trace_ordered_boundary(
            image,
            (0, 0, 60, 60),
            180,
            max_steps=3,
        )
        self.assertFalse(diagnostics["ok"])
        self.assertEqual(diagnostics["reason"], "max_steps")


if __name__ == "__main__":
    unittest.main()
