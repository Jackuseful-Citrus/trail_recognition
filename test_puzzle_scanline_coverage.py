"""Tests for precomputed target-polygon coverage scanlines."""

import unittest

import numpy as np

import puzzle_config as cfg
from puzzle_geometry import point_in_polygon
from puzzle_vision import (
    build_polygon_scanlines,
    polygon_white_coverage_scanlines,
)


class ScanlineCoverageTests(unittest.TestCase):
    def _polygon_px(self, polygon, width, height):
        return [
            (
                point[0] * (width - 1) / cfg.A4_WIDTH_MM,
                point[1] * (height - 1) / cfg.A4_HEIGHT_MM,
            )
            for point in polygon
        ]

    def test_cached_rectangle_reports_foreground_counts(self):
        width, height = 210, 297
        polygon = [(40, 180), (110, 180), (110, 230), (40, 230)]
        gray = np.zeros((height, width), dtype=np.uint8)
        polygon_px = self._polygon_px(
            polygon, width, height
        )
        for y in range(height):
            for x in range(width):
                if point_in_polygon((x, y), polygon_px):
                    gray[y, x] = 240
        cache = build_polygon_scanlines(
            polygon, width, height, sample_stride=2
        )
        result = polygon_white_coverage_scanlines(gray, cache)
        self.assertGreater(cache["sample_count"], 500)
        self.assertEqual(
            result["sample_count"], cache["sample_count"]
        )
        self.assertGreater(result["coverage_ratio"], 0.98)

    def test_concave_polygon_keeps_multiple_intervals(self):
        polygon = [
            (30, 170),
            (100, 170),
            (100, 230),
            (70, 230),
            (70, 195),
            (55, 195),
            (55, 230),
            (30, 230),
        ]
        cache = build_polygon_scanlines(
            polygon, 210, 297, sample_stride=1
        )
        self.assertTrue(
            any(
                len(intervals) == 2
                for intervals in cache["lines"].values()
            )
        )

    def test_cache_is_reusable_across_frames(self):
        polygon = [(50, 180), (90, 180), (90, 220), (50, 220)]
        cache = build_polygon_scanlines(
            polygon, 210, 297, sample_stride=2
        )
        dark = np.zeros((297, 210), dtype=np.uint8)
        bright = np.full((297, 210), 255, dtype=np.uint8)
        first = polygon_white_coverage_scanlines(dark, cache)
        second = polygon_white_coverage_scanlines(bright, cache)
        self.assertEqual(first["coverage_ratio"], 0.0)
        self.assertEqual(second["coverage_ratio"], 1.0)
        self.assertEqual(
            first["sample_count"], second["sample_count"]
        )


if __name__ == "__main__":
    unittest.main()
