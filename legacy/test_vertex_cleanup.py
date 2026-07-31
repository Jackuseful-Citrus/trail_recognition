"""Focused regression tests for the active physical-edge cleanup rules."""

import unittest

import puzzle_config as cfg
from puzzle_geometry import remove_near_collinear_vertices


class VertexCleanupTests(unittest.TestCase):
    def test_active_physical_thresholds(self):
        self.assertEqual(cfg.VERTEX_MERGE_DISTANCE_MM, 18.0)
        self.assertEqual(
            cfg.VERTEX_COLLINEAR_ANGLE_TOLERANCE_DEG,
            30.0,
        )

    def test_short_edge_is_split_equally_between_neighbours(self):
        polygon = [
            (0, 3),
            (3, 0),
            (40, 0),
            (40, 30),
            (0, 30),
        ]
        cleaned = remove_near_collinear_vertices(polygon)
        self.assertEqual(len(cleaned), 4)
        self.assertTrue(
            any(
                abs(x - 1.5) < 1e-6
                and abs(y - 1.5) < 1e-6
                for x, y in cleaned
            )
        )

    def test_150_degree_obtuse_vertex_is_smoothed(self):
        polygon = [
            (0, 0),
            (15, -4),
            (30, 0),
            (30, 30),
            (0, 30),
        ]
        cleaned = remove_near_collinear_vertices(
            polygon,
            tolerance_deg=30.0,
            min_edge_mm=0.0,
            max_collinear_offset_mm=4.0,
        )
        self.assertEqual(len(cleaned), 4)

    def test_logged_p1_five_vertices_reduce_to_four(self):
        polygon = [
            (112.35, 33.82),
            (172.81, 100.52),
            (182.82, 106.23),
            (194.22, 100.03),
            (153.01, 14.16),
        ]
        cleaned = remove_near_collinear_vertices(
            polygon,
            tolerance_deg=30.0,
            min_edge_mm=18.0,
            max_collinear_offset_mm=4.0,
            max_area_change_ratio=0.15,
        )
        self.assertEqual(len(cleaned), 4)
        self.assertTrue(
            any(
                abs(x - 177.815) < 1e-6
                and abs(y - 103.375) < 1e-6
                for x, y in cleaned
            )
        )


if __name__ == "__main__":
    unittest.main()
