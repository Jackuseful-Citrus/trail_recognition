"""Call-count regression gates for the supplied desktop photograph."""

import json
import unittest

import cv2
import numpy as np

import puzzle_config as cfg
from puzzle_geometry import (
    build_edge_candidate_graph,
    geometry_counters_snapshot,
    plan_outer_first_rectangle,
    plan_rectangle_assembly,
    reset_geometry_counters,
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
        plan = plan_outer_first_rectangle(
            pieces,
            target_size_mm=cfg.TARGET_RECT_SIZE_MM,
        )
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

    def test_fixed_beam_reuses_incremental_rank_metrics(self):
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
            "performance_optimized.json", "r", encoding="utf-8"
        ) as handle:
            previous = json.load(handle)["desktop_sample"]

        reset_geometry_counters()
        plan = plan_rectangle_assembly(pieces)
        counters = geometry_counters_snapshot()
        self.assertTrue(plan.valid, plan.reason)
        self.assertLess(
            counters["polygon_intersection_calls"],
            previous["fixed_plan_polygon_intersections"],
        )
        self.assertAlmostEqual(plan.score, 0.031757453540141055)


if __name__ == "__main__":
    unittest.main()
