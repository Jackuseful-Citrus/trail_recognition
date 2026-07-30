"""One-time edge candidate graph and seam-tolerance tests."""

import json
import unittest

import puzzle_config as cfg
from puzzle_geometry import (
    PieceObservation,
    _edge_length_matches,
    build_edge_candidate_graph,
    rigid_transform_from_edge_pair,
    transform_point,
)


class EdgeCandidateGraphTests(unittest.TestCase):
    def _sample_pieces(self):
        with open(
            "offline_puzzle_result.json", encoding="utf-8"
        ) as handle:
            record = json.load(handle)
        return [
            PieceObservation(
                item["piece_id"],
                [],
                item["polygon_mm"],
            )
            for item in record["pieces"]
        ]

    def test_absolute_and_relative_seam_tolerances(self):
        self.assertTrue(_edge_length_matches(50.0, 54.0))
        self.assertTrue(_edge_length_matches(100.0, 105.0))
        self.assertFalse(_edge_length_matches(50.0, 55.0))
        self.assertFalse(_edge_length_matches(100.0, 108.0))

    def test_graph_only_pairs_different_pieces_once(self):
        pieces = self._sample_pieces()
        graph = build_edge_candidate_graph(pieces)
        edge_counts = [len(piece.polygon_mm) for piece in pieces]
        expected_raw = sum(
            edge_counts[i] * edge_counts[j]
            for i in range(len(pieces))
            for j in range(i)
        )
        self.assertEqual(graph.raw_pair_count, expected_raw)
        keys = [
            (
                candidate.piece_a,
                candidate.edge_a,
                candidate.piece_b,
                candidate.edge_b,
            )
            for candidate in graph.candidates
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(
            all(
                candidate.piece_a != candidate.piece_b
                for candidate in graph.candidates
            )
        )

    def test_sample_candidate_count_stays_small(self):
        graph = build_edge_candidate_graph(
            self._sample_pieces()
        )
        self.assertEqual(graph.raw_pair_count, 84)
        self.assertLessEqual(graph.filtered_pair_count, 6)
        self.assertGreaterEqual(graph.filtered_pair_count, 3)

    def test_final_tolerance_does_not_change_graph(self):
        pieces = self._sample_pieces()
        original_final = cfg.FINAL_VERTEX_TOLERANCE_MM
        original_legacy = cfg.CORRESPONDING_VERTEX_TOLERANCE_MM
        try:
            first = build_edge_candidate_graph(
                pieces
            ).filtered_pair_count
            cfg.FINAL_VERTEX_TOLERANCE_MM = 200.0
            cfg.CORRESPONDING_VERTEX_TOLERANCE_MM = 200.0
            second = build_edge_candidate_graph(
                pieces
            ).filtered_pair_count
        finally:
            cfg.FINAL_VERTEX_TOLERANCE_MM = original_final
            cfg.CORRESPONDING_VERTEX_TOLERANCE_MM = original_legacy
        self.assertEqual(first, second)

    def test_precomputed_transform_matches_direct_alignment(self):
        graph = build_edge_candidate_graph(
            self._sample_pieces()
        )
        self.assertTrue(graph.candidates)
        candidate = graph.candidates[0]
        desc_a = next(
            edge
            for edge in graph.edges
            if edge.piece_index == candidate.piece_a
            and edge.edge_index == candidate.edge_a
        )
        desc_b = next(
            edge
            for edge in graph.edges
            if edge.piece_index == candidate.piece_b
            and edge.edge_index == candidate.edge_b
        )
        direct = rigid_transform_from_edge_pair(
            (desc_a.p0, desc_a.p1),
            (desc_b.p0, desc_b.p1),
        )
        for actual, expected in zip(
            candidate.transform_b_to_a, direct
        ):
            self.assertAlmostEqual(actual, expected, places=8)
        mapped_b0 = transform_point(
            desc_b.p0, candidate.transform_b_to_a
        )
        mapped_b1 = transform_point(
            desc_b.p1, candidate.transform_b_to_a
        )
        self.assertAlmostEqual(
            mapped_b0[0] + mapped_b1[0],
            desc_a.p0[0] + desc_a.p1[0],
            places=7,
        )
        self.assertAlmostEqual(
            mapped_b0[1] + mapped_b1[1],
            desc_a.p0[1] + desc_a.p1[1],
            places=7,
        )

    def test_edge_and_piece_pair_indices(self):
        graph = build_edge_candidate_graph(
            self._sample_pieces()
        )
        for candidate in graph.candidates:
            self.assertIn(
                candidate,
                graph.for_piece_pair(
                    candidate.piece_a,
                    candidate.piece_b,
                ),
            )
            self.assertIn(
                candidate,
                graph.for_edge(
                    candidate.piece_a,
                    candidate.edge_a,
                ),
            )
            self.assertEqual(
                graph.candidate_count_by_open_edge[
                    (candidate.piece_a, candidate.edge_a)
                ],
                len(
                    graph.for_edge(
                        candidate.piece_a,
                        candidate.edge_a,
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
