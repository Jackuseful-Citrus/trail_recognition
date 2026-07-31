"""Desktop tests for automatic A4 candidate selection and tracking."""

import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import realtime_a4_config as cfg
from puzzle_geometry import (
    PieceObservation,
    polygon_overlap_area,
)
from puzzle_a4_boundary import (
    A4BoundaryTracker,
    _projective_quad_point,
    detect_a4_boundary,
    order_a4_corners,
    project_a4_mm_to_frame,
)
from puzzle_realtime_state import (
    MotionDetector,
    PieceCountConsensus,
    a4_detection_interval,
    bottom_right_thumbnail_rect,
    operator_overlay_visibility,
    operator_status_line,
    planning_input_integrity,
    planning_input_integrity_unless_fixed_template,
    periodic_output_due,
    recognition_gate_ready,
    top_right_thumbnail_rect,
    tracking_expected_piece_count,
)


class _FakeRect:
    def __init__(self, corners, magnitude=30000):
        self._corners = corners
        self._magnitude = magnitude

    def corners(self):
        return self._corners

    def magnitude(self):
        return self._magnitude


class _FakeBlob:
    def __init__(self, corners, min_corners, pixels=30000):
        self._corners = corners
        self._min_corners = min_corners
        self._pixels = pixels

    def corners(self):
        return self._corners

    def min_corners(self):
        return self._min_corners

    def pixels(self):
        return self._pixels


class _FakeBoundaryImage:
    def __init__(self, array, rectangles, blobs=None):
        self.array = array
        self.rectangles = rectangles
        self.blobs = list(blobs or ())

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
        return self.blobs


def _draw_projective_divider(
    gray,
    physical_corners,
    center_fraction=0.5,
    slope_fraction=0.0,
):
    for index in range(401):
        u = index / 400.0
        v = center_fraction + slope_fraction * (u - 0.5)
        point = _projective_quad_point(
            physical_corners, u, v
        )
        if point is None:
            continue
        x = int(round(point[0]))
        y = int(round(point[1]))
        gray[
            max(0, y - 1) : min(gray.shape[0], y + 2),
            max(0, x - 1) : min(gray.shape[1], x + 2),
        ] = 225


def _landscape_physical_corners(image_corners):
    tl, tr, br, bl = image_corners
    return [bl, tl, tr, br]


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
        _draw_projective_divider(gray, self.corners)
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
            self.corners[0][0]
            * 799.0
            / float(cfg.A4_DETECT_WIDTH - 1),
            places=4,
        )

    def test_a4_overlay_uses_projective_not_bilinear_mapping(self):
        corners = [
            (110.0, 431.0),
            (123.0, 73.0),
            (591.0, 73.0),
            (594.0, 431.0),
        ]
        point_mm = (
            cfg.A4_WIDTH_MM * 0.5,
            cfg.A4_HEIGHT_MM * 0.2,
        )
        expected = _projective_quad_point(
            corners, 0.5, 0.2
        )
        actual = project_a4_mm_to_frame(
            point_mm, corners
        )
        self.assertAlmostEqual(actual[0], expected[0])
        self.assertAlmostEqual(actual[1], expected[1])
        bilinear_y = (
            0.4 * corners[0][1]
            + 0.4 * corners[1][1]
            + 0.1 * corners[2][1]
            + 0.1 * corners[3][1]
        )
        self.assertGreater(
            abs(actual[1] - bilinear_y), 2.5
        )

    def test_projective_divider_validates_without_moving_corners(self):
        gray = np.full(
            (cfg.A4_DETECT_HEIGHT, cfg.A4_DETECT_WIDTH),
            235,
            dtype=np.uint8,
        )
        gray[14:179, 108:212] = 35
        _draw_projective_divider(
            gray, self.corners, center_fraction=0.52
        )
        candidate, _ = detect_a4_boundary(
            _FakeBoundaryImage(
                gray, [_FakeRect(self.corners)]
            ),
            (cfg.A4_DETECT_WIDTH, cfg.A4_DETECT_HEIGHT),
        )
        self.assertIsNotNone(candidate)
        self.assertTrue(candidate["divider_detected"])
        self.assertAlmostEqual(
            candidate["divider_y_mm"],
            0.52 * cfg.A4_HEIGHT_MM,
            delta=1.5,
        )
        self.assertAlmostEqual(
            candidate["divider_slope_mm"], 0.0, delta=1.0
        )
        self.assertGreater(candidate["divider_confidence"], 0.7)
        self.assertEqual(
            candidate["corners_work_px"],
            candidate["image_corners_work_px"],
        )

    def test_slanted_rectified_divider_rejects_bad_corners(self):
        gray = np.full(
            (cfg.A4_DETECT_HEIGHT, cfg.A4_DETECT_WIDTH),
            235,
            dtype=np.uint8,
        )
        gray[14:179, 108:212] = 35
        _draw_projective_divider(
            gray,
            self.corners,
            center_fraction=0.50,
            slope_fraction=0.04,
        )
        candidate, diagnostics = detect_a4_boundary(
            _FakeBoundaryImage(
                gray, [_FakeRect(self.corners)]
            ),
            (cfg.A4_DETECT_WIDTH, cfg.A4_DETECT_HEIGHT),
        )
        self.assertIsNone(candidate)
        self.assertEqual(
            diagnostics["rejected"]["divider_slope"], 1
        )

    def test_dark_blob_uses_contour_corners_not_min_area_box(self):
        gray = np.full(
            (cfg.A4_DETECT_HEIGHT, cfg.A4_DETECT_WIDTH),
            235,
            dtype=np.uint8,
        )
        gray[14:179, 108:212] = 35
        _draw_projective_divider(gray, self.corners)
        minimum_area_box = [
            (108, 14),
            (212, 14),
            (212, 179),
            (108, 179),
        ]
        candidate, diagnostics = detect_a4_boundary(
            _FakeBoundaryImage(
                gray,
                [],
                [
                    _FakeBlob(
                        self.corners,
                        minimum_area_box,
                    )
                ],
            ),
            (cfg.A4_DETECT_WIDTH, cfg.A4_DETECT_HEIGHT),
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["source"], "dark_blob_contour")
        self.assertEqual(
            candidate["image_corners_work_px"],
            [(float(x), float(y)) for x, y in self.corners],
        )
        self.assertEqual(diagnostics["raw_dark_blobs"], 1)

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

    def test_first_locked_calibration_is_immutable_after_freeze(self):
        tracker = A4BoundaryTracker()
        base = {
            "corners_px": [
                (float(point[0]), float(point[1]))
                for point in self.corners
            ],
            "confidence": 0.9,
            "source": "rect",
            "orientation": "portrait_top_top",
        }
        for _ in range(cfg.A4_LOCK_REQUIRED_FRAMES):
            state = tracker.update(base)
        self.assertTrue(state["locked"])
        state = tracker.freeze()
        frozen = list(state["corners_px"])
        self.assertTrue(state["frozen"])
        self.assertEqual(state["motion_px"], 0.0)

        moved = dict(base)
        moved["corners_px"] = [
            (point[0] + 40.0, point[1] - 25.0)
            for point in frozen
        ]
        state = tracker.update(moved)
        self.assertEqual(state["corners_px"], frozen)
        self.assertEqual(state["motion_px"], 0.0)
        self.assertTrue(state["frozen"])

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

    def test_locked_a4_uses_deadband_and_confirmed_relock(self):
        tracker = A4BoundaryTracker()
        base = {
            "corners_px": [
                (float(point[0]), float(point[1]))
                for point in self.corners
            ],
            "confidence": 0.9,
            "source": "rect",
        }
        for _ in range(cfg.A4_LOCK_REQUIRED_FRAMES):
            state = tracker.update(base)
        frozen = list(state["corners_px"])
        jitter = dict(base)
        jitter["corners_px"] = [
            (point[0] + 2.0, point[1] - 2.0)
            for point in frozen
        ]
        state = tracker.update(jitter)
        self.assertEqual(state["corners_px"], frozen)
        moved = dict(base)
        moved["corners_px"] = [
            (point[0] + 15.0, point[1])
            for point in frozen
        ]
        for attempt in range(cfg.A4_RELOCK_CONFIRM_FRAMES):
            state = tracker.update(moved)
            if attempt < cfg.A4_RELOCK_CONFIRM_FRAMES - 1:
                self.assertEqual(state["corners_px"], frozen)
        self.assertEqual(state["corners_px"], moved["corners_px"])

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
        _draw_projective_divider(
            gray, _landscape_physical_corners(landscape)
        )
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
        _draw_projective_divider(
            gray, _landscape_physical_corners(landscape)
        )
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

    def test_landscape_full_outline_beats_divider_half_outline(self):
        gray = np.full(
            (cfg.A4_DETECT_HEIGHT, cfg.A4_DETECT_WIDTH),
            235,
            dtype=np.uint8,
        )
        landscape = [
            (24, 23),
            (296, 25),
            (297, 181),
            (23, 179),
        ]
        # The full landscape paper is dark.  Its physical centre divider is
        # vertical in the camera image and, together with three outer edges,
        # produces a very strong false half-paper rectangle.
        gray[23:182, 23:298] = 35
        _draw_projective_divider(
            gray, _landscape_physical_corners(landscape)
        )
        divider_half = [
            (24, 23),
            (160, 24),
            (160, 180),
            (23, 179),
        ]
        image = _FakeBoundaryImage(
            gray,
            [
                _FakeRect(divider_half, 80000),
                _FakeRect(landscape, 500),
            ],
        )
        candidate, diagnostics = detect_a4_boundary(
            image, (800, 480)
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(
            candidate["image_corners_work_px"],
            [(float(x), float(y)) for x, y in landscape],
        )
        self.assertEqual(
            candidate["orientation"], "landscape_top_left"
        )
        self.assertEqual(
            diagnostics["rejected"]["internal_edge"], 1
        )

    def test_divider_half_without_full_outline_fails_closed(self):
        gray = np.full(
            (cfg.A4_DETECT_HEIGHT, cfg.A4_DETECT_WIDTH),
            235,
            dtype=np.uint8,
        )
        gray[23:182, 23:298] = 35
        divider_half = [
            (24, 23),
            (160, 24),
            (160, 180),
            (23, 179),
        ]
        image = _FakeBoundaryImage(
            gray, [_FakeRect(divider_half, 80000)]
        )
        candidate, diagnostics = detect_a4_boundary(
            image, (800, 480)
        )
        self.assertIsNone(candidate)
        self.assertEqual(
            diagnostics["rejected"]["internal_edge"], 1
        )

    def test_full_outline_divider_overrides_false_internal_edge(self):
        gray = np.full(
            (cfg.A4_DETECT_HEIGHT, cfg.A4_DETECT_WIDTH),
            235,
            dtype=np.uint8,
        )
        landscape = [
            (24, 23),
            (296, 25),
            (297, 181),
            (23, 179),
        ]
        gray[23:182, 23:298] = 35
        # Simulate a dark surround/inset outer-corner estimate.  The top-edge
        # probe sees the same dark tone on both sides and would otherwise
        # reject the real full sheet as an internal half-sheet boundary.
        gray[8:36, 23:298] = 35
        # The physical centre divider is vertical for this landscape camera
        # orientation and spans well above the configured 70% requirement.
        _draw_projective_divider(
            gray, _landscape_physical_corners(landscape)
        )
        candidate, diagnostics = detect_a4_boundary(
            _FakeBoundaryImage(
                gray, [_FakeRect(landscape, 30000)]
            ),
            (800, 480),
        )
        self.assertIsNotNone(candidate)
        self.assertTrue(candidate["divider_detected"])
        self.assertEqual(
            diagnostics["divider_rescued_internal_edge"], 1
        )
        self.assertNotIn("internal_edge", diagnostics["rejected"])


class RealtimeDisplayStateTests(unittest.TestCase):
    def test_operator_view_hides_piece_layers_during_motion(self):
        waiting = operator_overlay_visibility(
            "ACQUIRE", False
        )
        self.assertTrue(waiting["a4"])
        self.assertTrue(waiting["status"])
        self.assertTrue(waiting["pieces"])
        self.assertTrue(waiting["targets"])

        moving = operator_overlay_visibility(
            "ACQUIRE", True
        )
        self.assertTrue(moving["a4"])
        self.assertTrue(moving["status"])
        self.assertFalse(moving["pieces"])
        self.assertFalse(moving["targets"])
        final = operator_overlay_visibility(
            "WAIT_FINAL_CHECK", True
        )
        self.assertTrue(final["pieces"])
        self.assertTrue(final["targets"])

    def test_operator_status_line_is_short_and_state_specific(self):
        self.assertEqual(
            operator_status_line(
                "WAIT_FINAL_CHECK",
                4,
                plan_available=True,
                plan_valid=True,
            ),
            "WAIT FINAL CHECK",
        )
        self.assertEqual(
            operator_status_line(
                "ACQUIRE",
                4,
                stable=True,
                plan_available=True,
                plan_valid=False,
            ),
            "ACQUIRE | PLAN BLOCKED | P:4",
        )

    def test_locked_or_final_state_freezes_a4_updates(self):
        self.assertEqual(
            a4_detection_interval("ACQUIRE", 2), 2
        )
        self.assertIsNone(
            a4_detection_interval(
                "ACQUIRE", 2, locked=True
            )
        )
        self.assertIsNone(
            a4_detection_interval("WAIT_FINAL_CHECK", 2)
        )
        self.assertIsNone(
            a4_detection_interval("COMPLETE", 2)
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

    def test_gray_thumbnail_fits_top_right_and_keeps_aspect(self):
        x, y, width, height, scale = top_right_thumbnail_rect(
            800,
            480,
            240,
            336,
            128,
            180,
            8,
        )
        self.assertEqual(x + width, 792)
        self.assertEqual(y, 8)
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

    def test_fixed_count_tracking_runs_while_consensus_settles(self):
        consensus = PieceCountConsensus(4, 4, 12, 4, 2)
        state = consensus.state()
        self.assertEqual(
            tracking_expected_piece_count(4, state), 4
        )
        for _ in range(3):
            state = consensus.update(4)
            self.assertFalse(
                recognition_gate_ready(state, True)
            )
            self.assertEqual(
                tracking_expected_piece_count(4, state), 4
            )
        state = consensus.update(4)
        self.assertTrue(recognition_gate_ready(state, True))

    def test_production_disables_full_gray_sanity_scan(self):
        self.assertFalse(cfg.ENABLE_GRAY_SANITY_DIAGNOSTICS)

    def test_motion_detector_uses_both_mean_and_changed_ratio(self):
        detector = MotionDetector(18, 5.0, 0.20)
        base = np.zeros((10, 10), dtype=np.uint8)
        detector.update(base)
        one_pixel = base.copy()
        one_pixel[0, 0] = 255
        metrics = detector.update(one_pixel)
        self.assertFalse(metrics["motion"])
        broad = np.full((10, 10), 80, dtype=np.uint8)
        metrics = detector.update(broad)
        self.assertTrue(metrics["motion"])

    def test_planning_input_gate_rejects_missing_duplicate_and_border(self):
        polygons = [
            [(0, 0), (50, 0), (50, 30), (0, 30)],
            [(55, 0), (105, 0), (105, 30), (55, 30)],
            [(0, 35), (50, 35), (50, 65), (0, 65)],
            [(55, 35), (105, 35), (105, 65), (55, 65)],
        ]
        pieces = [
            PieceObservation(
                "P{}".format(index + 1), [], polygon
            )
            for index, polygon in enumerate(polygons)
        ]
        common = {
            "target_rect_size_mm": (100.0, 60.0),
            "overlap_area": polygon_overlap_area,
            "required_piece_count": 4,
            "area_ratio_min": 0.85,
            "area_ratio_max": 1.15,
            "max_pair_overlap_ratio": 0.20,
        }
        valid = planning_input_integrity(
            pieces,
            rejected_border_blobs=0,
            **common
        )
        self.assertTrue(valid["valid"])

        missing = planning_input_integrity(
            pieces[:3],
            rejected_border_blobs=0,
            **common
        )
        self.assertFalse(missing["valid"])
        self.assertIn("piece_count", missing["failures"])
        self.assertIn("total_area", missing["failures"])

        duplicate_pieces = list(pieces)
        duplicate_pieces[1] = PieceObservation(
            "P2", [], polygons[0]
        )
        duplicate = planning_input_integrity(
            duplicate_pieces,
            rejected_border_blobs=0,
            **common
        )
        self.assertFalse(duplicate["valid"])
        self.assertIn("pair_overlap", duplicate["failures"])
        self.assertAlmostEqual(
            duplicate["max_pair_overlap_ratio"], 1.0
        )

        border = planning_input_integrity(
            pieces,
            rejected_border_blobs=1,
            max_rejected_border_blobs=0,
            **common
        )
        self.assertFalse(border["valid"])
        self.assertIn("border_blob", border["failures"])

        bypass = planning_input_integrity_unless_fixed_template(
            True,
            pieces,
            None,
            polygon_overlap_area,
            required_piece_count=4,
            rejected_border_blobs=1,
            max_rejected_border_blobs=0,
        )
        self.assertIsNone(bypass)

if __name__ == "__main__":
    unittest.main()
