"""Tests for optional card-pattern seam data and scoring."""

import unittest

import numpy as np

import puzzle_config as cfg
from puzzle_geometry import (
    PieceObservation,
    build_edge_candidate_graph,
)
from puzzle_image_strip import (
    EdgeImageStrip,
    apply_optional_strip_costs,
    build_non_background_mask,
    compute_edge_strip_cost,
    pixel_is_piece,
    sample_edge_image_strip,
)


def _strip(values, gradients=None):
    if gradients is None:
        gradients = [0.0 for _ in values]
    return EdgeImageStrip(
        values,
        gradients,
        [1 for _ in values],
        len(values),
        1,
    )


class EdgeImageStripTests(unittest.TestCase):
    def test_reverse_edge_alignment_has_low_cost(self):
        first = _strip([20, 70, 140, 210])
        reversed_match = _strip([210, 140, 70, 20])
        self.assertAlmostEqual(
            compute_edge_strip_cost(first, reversed_match),
            0.0,
        )

    def test_discontinuous_pattern_costs_more(self):
        first = _strip([20, 70, 140, 210])
        continuous = _strip([210, 140, 70, 20])
        discontinuous = _strip([0, 0, 0, 0])
        self.assertGreater(
            compute_edge_strip_cost(first, discontinuous),
            compute_edge_strip_cost(first, continuous),
        )

    def test_sampler_stays_inside_piece(self):
        gray = np.full((297, 210), 180, dtype=np.uint8)
        polygon = [(30, 30), (30, 80), (100, 80), (100, 30)]
        strip = sample_edge_image_strip(
            gray,
            polygon,
            0,
            strip_width_mm=3.0,
            sample_spacing_mm=1.0,
        )
        self.assertGreater(strip.sample_count, 40)
        self.assertTrue(
            all(
                value == 180
                for index, value in enumerate(
                    strip.gray_samples
                )
                if strip.valid_mask[index]
            )
        )

    def test_non_background_keeps_black_red_and_white_card_pixels(self):
        background = (20, 80, 120)
        image = np.array(
            [[background, (5, 5, 5), (210, 20, 20), (245, 245, 245)]],
            dtype=np.uint8,
        )
        mask = build_non_background_mask(
            image, background, distance_threshold=45
        )
        self.assertEqual(list(mask[0]), [0, 255, 255, 255])
        self.assertFalse(
            pixel_is_piece(
                background,
                mode="non_background_rgb",
                background_rgb=background,
                distance_threshold=45,
            )
        )

    def test_disabled_strip_scoring_does_not_change_geometry(self):
        pieces = [
            PieceObservation(
                "P1", [], [(0, 0), (0, 30), (40, 30), (40, 0)]
            ),
            PieceObservation(
                "P2", [], [(50, 0), (50, 30), (90, 30), (90, 0)]
            ),
        ]
        graph = build_edge_candidate_graph(pieces)
        before = [
            candidate.geometric_cost
            for candidate in graph.candidates
        ]
        original = cfg.ENABLE_IMAGE_STRIP_MATCHING
        try:
            cfg.ENABLE_IMAGE_STRIP_MATCHING = False
            applied = apply_optional_strip_costs(graph, {})
        finally:
            cfg.ENABLE_IMAGE_STRIP_MATCHING = original
        self.assertEqual(applied, 0)
        self.assertEqual(
            before,
            [
                candidate.geometric_cost
                for candidate in graph.candidates
            ],
        )


if __name__ == "__main__":
    unittest.main()
