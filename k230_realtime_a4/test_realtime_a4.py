"""Desktop tests for automatic A4 candidate selection and tracking."""

import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import realtime_a4_config as cfg
from puzzle_a4_boundary import (
    A4BoundaryTracker,
    detect_a4_boundary,
    order_a4_corners,
)
from puzzle_realtime_state import (
    PieceCountConsensus,
    RealtimePhaseState,
    a4_detection_interval,
    bottom_right_thumbnail_rect,
    phase_allows_vision,
    placement_ui_key,
    periodic_output_due,
    should_render_ui,
)


class _FakeRect:
    def __init__(self, corners, magnitude=30000):
        self._corners = corners
        self._magnitude = magnitude

    def corners(self):
        return self._corners

    def magnitude(self):
        return self._magnitude


class _FakeBoundaryImage:
    def __init__(self, array, rectangles):
        self.array = array
        self.rectangles = rectangles

    def width(self):
        return self.array.shape[1]

    def height(self):
        return self.array.shape[0]

    def get_pixel(self, x, y):
        return int(self.array[y, x])

    def find_rects(self, *, threshold):
        self.last_threshold = threshold
        return self.rectangles

    def find_blobs(self, *args, **kwargs):
        return []


class AutomaticA4Tests(unittest.TestCase):
    def setUp(self):
        self.corners = [
            (112, 14),
            (207, 17),
            (211, 178),
            (108, 175),
        ]

    def test_corner_order_is_tl_tr_br_bl(self):
        shuffled = [
            self.corners[2],
            self.corners[0],
            self.corners[3],
            self.corners[1],
        ]
        self.assertEqual(
            order_a4_corners(shuffled),
            [(float(x), float(y)) for x, y in self.corners],
        )

    def test_dark_portrait_a4_candidate_is_selected(self):
        gray = np.full(
            (cfg.A4_DETECT_HEIGHT, cfg.A4_DETECT_WIDTH),
            235,
            dtype=np.uint8,
        )
        # Candidate sampling is internal; a filled bounding region is enough
        # for this mildly perspective-distorted synthetic A4.
        gray[14:179, 108:212] = 35
        image = _FakeBoundaryImage(
            gray, [_FakeRect(self.corners)]
        )
        candidate, diagnostics = detect_a4_boundary(
            image, (800, 480)
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["source"], "rect")
        self.assertLess(candidate["inside_gray"], 80)
        self.assertGreater(candidate["confidence"], 0.45)
        self.assertEqual(diagnostics["valid_candidates"], 1)
        self.assertAlmostEqual(
            candidate["corners_px"][0][0],
            self.corners[0][0] * 799.0 / 319.0,
            places=4,
        )

    def test_tracker_locks_during_small_camera_jitter(self):
        tracker = A4BoundaryTracker()
        for frame in range(6):
            jitter = -1.0 if frame % 2 else 1.0
            candidate = {
                "corners_px": [
                    (
                        point[0] * 2.0 + jitter,
                        point[1] * 2.0 - jitter,
                    )
                    for point in self.corners
                ],
                "confidence": 0.85,
                "source": "rect",
            }
            state = tracker.update(candidate)
        self.assertTrue(state["locked"])
        self.assertLess(state["motion_px"], 4.0)
        for _ in range(cfg.A4_HOLD_MISSED_FRAMES):
            state = tracker.update(None)
            self.assertTrue(state["locked"])
        state = tracker.update(None)
        self.assertFalse(state["locked"])
        self.assertIsNone(state["corners_px"])

    def test_sparse_candidates_cannot_accumulate_a_false_lock(self):
        tracker = A4BoundaryTracker()
        candidate = {
            "corners_px": [
                (float(point[0]), float(point[1]))
                for point in self.corners
            ],
            "confidence": 0.85,
            "source": "rect",
            "orientation": "portrait_top_top",
        }
        for _ in range(8):
            state = tracker.update(candidate)
            self.assertFalse(state["locked"])
            state = tracker.update(None)
            self.assertFalse(state["locked"])
            self.assertEqual(state["valid_frames"], 0)

    def test_landscape_a4_is_supported_and_touching_frame_is_rejected(self):
        gray = np.full(
            (cfg.A4_DETECT_HEIGHT, cfg.A4_DETECT_WIDTH),
            230,
            dtype=np.uint8,
        )
        landscape = [
            (24, 23),
            (296, 25),
            (297, 181),
            (23, 179),
        ]
        gray[23:182, 23:298] = 30
        clipped_false_positive = [
            (0, 0),
            (145, 0),
            (145, 191),
            (0, 191),
        ]
        image = _FakeBoundaryImage(
            gray,
            [
                _FakeRect(clipped_false_positive, 50000),
                _FakeRect(landscape, 30000),
            ],
        )
        candidate, diagnostics = detect_a4_boundary(
            image, (800, 480)
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(
            candidate["orientation"], "landscape_top_left"
        )
        self.assertEqual(
            diagnostics["rejected"]["touches_edge"], 1
        )
        # For top-half-on-left orientation, physical TL is image BL.
        self.assertAlmostEqual(
            candidate["corners_work_px"][0][0], 23.0
        )
        self.assertAlmostEqual(
            candidate["corners_work_px"][0][1], 179.0
        )

    def test_landscape_physical_top_side_is_inferred_from_fragments(self):
        gray = np.full(
            (cfg.A4_DETECT_HEIGHT, cfg.A4_DETECT_WIDTH),
            230,
            dtype=np.uint8,
        )
        landscape = [
            (24, 23),
            (296, 25),
            (297, 181),
            (23, 179),
        ]
        gray[23:182, 23:298] = 30
        # Put white fragment samples only in the image-right half.  Automatic
        # orientation must therefore map image-right to physical A4 y=0.
        gray[45:105, 205:275] = 240
        image = _FakeBoundaryImage(
            gray, [_FakeRect(landscape, 30000)]
        )
        candidate, _ = detect_a4_boundary(image, (800, 480))
        self.assertIsNotNone(candidate)
        self.assertEqual(
            candidate["orientation"], "landscape_top_right"
        )
        # Physical TL for a right-side source half is image TR.
        self.assertAlmostEqual(
            candidate["corners_work_px"][0][0], 296.0
        )
        self.assertAlmostEqual(
            candidate["corners_work_px"][0][1], 25.0
        )


class RealtimeDisplayStateTests(unittest.TestCase):
    def _placement_state(self, completed=None, next_id="P1"):
        return {
            "completed": set(completed or ()),
            "next_piece_id": next_id,
            "observed_count": 2,
            "matched_count": 2,
            "visible_pieces": [],
        }

    def test_static_placing_state_skips_full_render(self):
        state = self._placement_state()
        key = placement_ui_key(
            "PLACING", state, 4200, 1000
        )
        same_bucket = placement_ui_key(
            "PLACING", state, 4001, 1000
        )
        self.assertFalse(should_render_ui(key, same_bucket))

    def test_countdown_or_result_change_requests_render(self):
        state = self._placement_state()
        initial = placement_ui_key(
            "PLACING", state, 4200, 1000
        )
        countdown_changed = placement_ui_key(
            "PLACING", state, 3900, 1000
        )
        completed_changed = placement_ui_key(
            "PLACING",
            self._placement_state({"P1"}, "P2"),
            3900,
            1000,
        )
        self.assertTrue(
            should_render_ui(initial, countdown_changed)
        )
        self.assertTrue(
            should_render_ui(
                countdown_changed, completed_changed
            )
        )

    def test_placing_uses_lower_a4_frequency(self):
        self.assertEqual(
            a4_detection_interval("ACQUIRE", 2, 8), 2
        )
        self.assertEqual(
            a4_detection_interval("PLACING", 2, 8), 8
        )
        self.assertIsNone(
            a4_detection_interval("COMPLETE", 2, 8)
        )

    def test_gray_thumbnail_fits_bottom_right_and_keeps_aspect(self):
        x, y, width, height, scale = (
            bottom_right_thumbnail_rect(
                800,
                480,
                240,
                336,
                128,
                180,
                8,
            )
        )
        self.assertEqual(x + width, 792)
        self.assertEqual(y + height, 472)
        self.assertLessEqual(width, 128)
        self.assertLessEqual(height, 180)
        self.assertAlmostEqual(width / height, 240 / 336, places=2)
        self.assertAlmostEqual(scale, min(128 / 240, 180 / 336))

    def test_explicit_ide_stream_is_rate_limited_but_sends_first(self):
        due = [
            index
            for index in range(6)
            if periodic_output_due(index, 2)
        ]
        self.assertEqual(due, [0, 2, 4])

    def test_complete_disables_vision_and_reset_restores_acquire(self):
        flow = RealtimePhaseState()
        flow.start_placing()
        flow.complete()
        self.assertFalse(flow.vision_enabled())
        self.assertFalse(phase_allows_vision("COMPLETE"))
        flow.reset()
        self.assertEqual(flow.phase, "ACQUIRE")
        self.assertTrue(flow.vision_enabled())

    def test_unknown_two_piece_count_becomes_ready(self):
        consensus = PieceCountConsensus(2, 4, 12, 5, 2)
        state = None
        for _ in range(5):
            state = consensus.update(2)
        self.assertTrue(state["ready"])
        self.assertEqual(state["expected_count"], 2)

    def test_intermittent_missing_piece_blocks_subset_plan(self):
        consensus = PieceCountConsensus(2, 4, 12, 5, 2)
        for count in (4, 3, 4, 3, 3):
            state = consensus.update(count)
        self.assertFalse(state["ready"])
        self.assertEqual(state["expected_count"], 4)
        self.assertEqual(
            state["reason"], "piece_count_mismatch"
        )
        state = consensus.update(4)
        self.assertTrue(state["ready"])

    def test_single_higher_count_is_held_until_resolved(self):
        consensus = PieceCountConsensus(2, 4, 6, 4, 2)
        for count in (4, 3, 3, 3):
            state = consensus.update(count)
        self.assertFalse(state["ready"])
        self.assertEqual(
            state["reason"], "higher_count_unconfirmed"
        )
        for _ in range(3):
            state = consensus.update(3)
        self.assertTrue(state["ready"])
        self.assertEqual(state["expected_count"], 3)


if __name__ == "__main__":
    unittest.main()
