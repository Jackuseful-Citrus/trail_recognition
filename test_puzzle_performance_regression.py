"""Call-count regression gates for the supplied desktop photograph."""

import json
import unittest

import cv2
import numpy as np

import puzzle_config as cfg
from puzzle_geometry import (
    build_edge_candidate_graph,
    plan_outer_first_rectangle,
)
from puzzle_vision import detect_pieces_from_gray


class PerformanceRegressionTests(unittest.TestCase):
    def test_runtime_planner_reduces_baseline_search_work(self):
        gray = cv2.imread(
            cfg.OFFLINE_IMAGE, cv2.IMREAD_GRAYSCALE
        )
        pieces, _ = detect_pieces_from_gray(
            gray,
            cfg.OFFLINE_A4_CORNERS_PX,
            cv2,
            np,
        )
        for index, piece in enumerate(pieces):
            piece.piece_id = "P{}".format(index + 1)
        with open(
            "performance_baseline.json", "r", encoding="utf-8"
        ) as handle:
            baseline = json.load(handle)["desktop_sample"]

        graph = build_edge_candidate_graph(pieces)
        plan = plan_outer_first_rectangle(pieces)
        self.assertTrue(plan.valid, plan.reason)
        self.assertEqual(graph.raw_pair_count, 84)
        self.assertLessEqual(graph.filtered_pair_count, 5)
        self.assertLessEqual(
            plan.search_nodes,
            baseline["outer_first_dfs_nodes"],
        )
        self.assertLessEqual(
            plan.plan_stats["polygon_intersection_calls"],
            baseline["outer_first_polygon_intersections"] // 2,
        )
        self.assertLessEqual(
            plan.plan_stats["rectangle_hypothesis_count"],
            cfg.MAX_RECTANGLE_HYPOTHESES,
        )
        width = plan.target_rect[2] - plan.target_rect[0]
        height = plan.target_rect[3] - plan.target_rect[1]
        self.assertEqual(
            sorted((round(width, 1), round(height, 1))),
            [60.0, 100.0],
        )


if __name__ == "__main__":
    unittest.main()
