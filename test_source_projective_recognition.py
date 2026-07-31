"""Regression tests for the K230 source-projective recognition backend."""

import unittest

import cv2
import numpy as np

import puzzle_config as cfg
from a4_projective_mapper import A4ProjectiveMapper
from puzzle_a4_boundary import A4BoundaryTracker
from puzzle_a4_boundary import detect_a4_boundary
from puzzle_geometry import PieceObservation, PieceTracker
from source_divider_detector import (
    SourceScanlineMask,
    detect_source_divider,
)
from source_projective_piece_detector import (
    SourceProjectiveRecognition,
    detect_pieces_from_source_projective_image,
)


class _Blob:
    def __init__(self, rect, pixels, center):
        self._rect = rect
        self._pixels = pixels
        self._center = center

    def rect(self):
        return self._rect

    def pixels(self):
        return self._pixels

    def cx(self):
        return self._center[0]

    def cy(self):
        return self._center[1]


class _Statistics:
    def __init__(self, array):
        self.array = array

    def min(self):
        return int(self.array.min())

    def max(self):
        return int(self.array.max())

    def mean(self):
        return float(self.array.mean())

    def median(self):
        return float(np.median(self.array))


class _SourceGrayImage:
    """CanMV-style image facade intentionally lacking rotation_corr()."""

    def __init__(self, array):
        self.array = array

    def width(self):
        return int(self.array.shape[1])

    def height(self):
        return int(self.array.shape[0])

    def to_numpy_ref(self):
        return self.array

    def get_pixel(self, x, y):
        return int(self.array[int(y), int(x)])

    def format(self):
        return "GRAYSCALE"

    def get_statistics(self):
        return _Statistics(self.array)

    def find_blobs(
        self,
        thresholds,
        *,
        roi,
        x_stride,
        y_stride,
        pixels_threshold,
        area_threshold,
        merge,
    ):
        del x_stride, y_stride, merge
        low, high = thresholds[0]
        mask = np.zeros_like(self.array, dtype=np.uint8)
        x, y, width, height = roi
        region = self.array[y : y + height, x : x + width]
        mask[y : y + height, x : x + width] = (
            (region >= low) & (region <= high)
        ).astype(np.uint8)
        count, _labels, stats, centers = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        blobs = []
        for label in range(1, count):
            left, top, blob_width, blob_height, pixels = stats[label]
            if pixels < pixels_threshold or blob_width * blob_height < area_threshold:
                continue
            blobs.append(
                _Blob(
                    (
                        int(left),
                        int(top),
                        int(blob_width),
                        int(blob_height),
                    ),
                    int(pixels),
                    (
                        int(round(centers[label][0])),
                        int(round(centers[label][1])),
                    ),
                )
            )
        return blobs


class _BoundaryBlob:
    def __init__(
        self, contour_corners, minimum_corners, pixels, rect=None
    ):
        self._contour_corners = contour_corners
        self._minimum_corners = minimum_corners
        self._pixels = pixels
        self._rect = rect

    def corners(self):
        return self._contour_corners

    def min_corners(self):
        return self._minimum_corners

    def pixels(self):
        return self._pixels

    def rect(self):
        return self._rect


class _BoundaryImage(_SourceGrayImage):
    def __init__(self, array, blob):
        super().__init__(array)
        self.blob = blob

    def find_rects(self, *, threshold):
        del threshold
        return []

    def find_blobs(self, *args, **kwargs):
        del args, kwargs
        return [self.blob]


def _draw_mm_polygon(array, mapper, polygon_mm, value):
    points = np.array(
        [
            [int(round(point[0])), int(round(point[1]))]
            for point in (
                mapper.a4_mm_to_source_px(vertex)
                for vertex in polygon_mm
            )
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(array, [points], int(value))


def _scene(corners=None, include_divider=True):
    width = 640
    height = 384
    if corners is None:
        corners = [
            (106.0, 353.0),
            (112.0, 62.0),
            (495.0, 66.0),
            (491.0, 359.0),
        ]
    mapper = A4ProjectiveMapper(corners, width, height)
    array = np.full((height, width), 155, dtype=np.uint8)
    _draw_mm_polygon(
        array,
        mapper,
        [(0, 0), (210, 0), (210, 297), (0, 297)],
        25,
    )
    if include_divider:
        _draw_mm_polygon(
            array,
            mapper,
            [(0, 147.0), (210, 148.0), (210, 151.0), (0, 150.0)],
            105,
        )
    pieces = [
        [(18, 18), (58, 20), (35, 55)],
        [(75, 20), (116, 22), (114, 54), (78, 52)],
        [(135, 18), (180, 20), (190, 43), (164, 65), (132, 48)],
    ]
    for polygon in pieces:
        _draw_mm_polygon(array, mapper, polygon, 225)
    return array, mapper, pieces


class MapperTests(unittest.TestCase):
    def test_corner_correspondence_and_roundtrip_variants(self):
        variants = [
            [(10, 10), (220, 10), (220, 307), (10, 307)],
            [(18, 20), (225, 13), (215, 310), (8, 294)],
            [(300, 20), (315, 230), (18, 245), (8, 35)],
            [(620, 360), (620, 60), (90, 50), (80, 350)],
            [(80, 350), (90, 50), (620, 60), (620, 360)],
        ]
        expected = [(0, 0), (210, 0), (210, 297), (0, 297)]
        for corners in variants:
            mapper = A4ProjectiveMapper(corners, 640, 384)
            self.assertTrue(mapper.valid)
            for source, millimetres in zip(corners, expected):
                mapped = mapper.source_px_to_a4_mm(source)
                self.assertAlmostEqual(mapped[0], millimetres[0], places=6)
                self.assertAlmostEqual(mapped[1], millimetres[1], places=6)
                restored = mapper.a4_mm_to_source_px(millimetres)
                self.assertAlmostEqual(restored[0], source[0], places=6)
                self.assertAlmostEqual(restored[1], source[1], places=6)
            for point in ((100, 100), (320, 192), (500, 280)):
                self.assertLess(mapper.roundtrip_error_px(point), 1e-6)

    def test_singular_mapping_is_invalid(self):
        mapper = A4ProjectiveMapper(
            [(0, 0), (10, 0), (20, 0), (30, 0)], 640, 384
        )
        self.assertFalse(mapper.valid)
        self.assertIsNone(mapper.source_px_to_a4_mm((1, 1)))

    def test_dark_blob_invalid_contour_falls_back_to_min_corners(self):
        array, mapper, _pieces = _scene()
        # Some CanMV builds expose a flat four-scalar contour payload instead
        # of four (x, y) points.  That attempt must not prevent min_corners().
        invalid_contour = [1, 2, 3, 4]
        blob = _BoundaryBlob(
            invalid_contour,
            list(mapper.a4_polygon_source_px),
            pixels=int(0.45 * array.size),
        )
        candidate, diagnostics = detect_a4_boundary(
            _BoundaryImage(array, blob), (640, 384)
        )
        self.assertIsNotNone(candidate, diagnostics)
        self.assertEqual(candidate["source"], "dark_blob_min_box")
        self.assertEqual(diagnostics["dark_blob_contour_corner_counts"], [4])
        self.assertEqual(diagnostics["dark_blob_min_corner_counts"], [4])
        self.assertEqual(diagnostics["dark_blob_min_corner_fallbacks"], 1)
        self.assertEqual(len(diagnostics["dark_blob_attempt_errors"]), 1)

    def test_dark_blob_invalid_corner_apis_fall_back_to_blob_rect(self):
        array, mapper, _pieces = _scene()
        xs = [point[0] for point in mapper.a4_polygon_source_px]
        ys = [point[1] for point in mapper.a4_polygon_source_px]
        x0 = int(min(xs))
        y0 = int(min(ys))
        rect = (
            x0,
            y0,
            int(max(xs)) - x0 + 1,
            int(max(ys)) - y0 + 1,
        )
        invalid = [(1, 1), (2, 2), (3, 3)]
        blob = _BoundaryBlob(
            invalid,
            invalid,
            pixels=int(0.45 * array.size),
            rect=rect,
        )
        old_required = cfg.A4_REQUIRE_DIVIDER_FOR_LOCK
        cfg.A4_REQUIRE_DIVIDER_FOR_LOCK = False
        try:
            candidate, diagnostics = detect_a4_boundary(
                _BoundaryImage(array, blob), (640, 384)
            )
        finally:
            cfg.A4_REQUIRE_DIVIDER_FOR_LOCK = old_required
        self.assertIsNotNone(candidate, diagnostics)
        self.assertEqual(candidate["source"], "dark_blob_rect_box")
        self.assertEqual(
            diagnostics["dark_blob_fallback_sources"],
            ["dark_blob_rect_box"],
        )


class DividerAndPieceTests(unittest.TestCase):
    def setUp(self):
        self.saved = {
            "PIECE_SEGMENTATION_MODE": cfg.PIECE_SEGMENTATION_MODE,
            "PIECE_CONTOUR_MIN_GRAY_THRESHOLD": cfg.PIECE_CONTOUR_MIN_GRAY_THRESHOLD,
            "FORCE_CONVEX_CONTOURS": cfg.FORCE_CONVEX_CONTOURS,
            "SOURCE_PROJECTIVE_FREEZE_DIVIDER": cfg.SOURCE_PROJECTIVE_FREEZE_DIVIDER,
            "SOURCE_PROJECTIVE_FREEZE_SOURCE_HALF": cfg.SOURCE_PROJECTIVE_FREEZE_SOURCE_HALF,
            "SOURCE_PROJECTIVE_CACHE_BACKGROUND": cfg.SOURCE_PROJECTIVE_CACHE_BACKGROUND,
            "SOURCE_PROJECTIVE_MASK_MODE": cfg.SOURCE_PROJECTIVE_MASK_MODE,
        }
        cfg.PIECE_SEGMENTATION_MODE = "background_delta"
        cfg.PIECE_CONTOUR_MIN_GRAY_THRESHOLD = 100
        cfg.FORCE_CONVEX_CONTOURS = False

    def tearDown(self):
        for name, value in self.saved.items():
            setattr(cfg, name, value)

    def test_real_source_divider_is_detected(self):
        array, mapper, _pieces = _scene()
        divider = detect_source_divider(_SourceGrayImage(array), mapper)
        self.assertTrue(divider["detected"], divider)
        self.assertGreaterEqual(divider["coverage"], 0.70)
        self.assertLess(abs(divider["slope_mm"]), 3.0)
        self.assertAlmostEqual(divider["divider_y_mm"], 149.0, delta=2.0)

    def test_missing_divider_never_uses_nominal_fallback(self):
        array, mapper, _pieces = _scene(include_divider=False)
        divider = detect_source_divider(_SourceGrayImage(array), mapper)
        self.assertFalse(divider["detected"])
        recognizer = SourceProjectiveRecognition()
        observations, diagnostics = recognizer.detect(
            _SourceGrayImage(array.copy()),
            mapper.a4_polygon_source_px,
            generation=1,
        )
        self.assertEqual(observations, [])
        self.assertEqual(diagnostics["reason"], "divider_required")
        self.assertEqual(diagnostics["rotation_corr_calls"], 0)

    def test_short_midband_piece_and_low_contrast_line_are_rejected(self):
        array, mapper, _pieces = _scene(include_divider=False)
        _draw_mm_polygon(
            array,
            mapper,
            [(75, 143), (120, 143), (120, 153), (75, 153)],
            225,
        )
        divider = detect_source_divider(_SourceGrayImage(array), mapper)
        self.assertFalse(divider["detected"])

        low_contrast, mapper, _pieces = _scene(include_divider=False)
        _draw_mm_polygon(
            low_contrast,
            mapper,
            [(0, 147), (210, 148), (210, 151), (0, 150)],
            50,
        )
        divider = detect_source_divider(
            _SourceGrayImage(low_contrast), mapper
        )
        self.assertFalse(divider["detected"])

    def test_partly_occluded_slightly_sloped_divider_is_retained(self):
        array, mapper, _pieces = _scene()
        _draw_mm_polygon(
            array,
            mapper,
            [(88, 145), (112, 145), (112, 153), (88, 153)],
            25,
        )
        divider = detect_source_divider(_SourceGrayImage(array), mapper)
        self.assertTrue(divider["detected"], divider)
        self.assertGreaterEqual(divider["coverage"], 0.70)

    def test_three_four_five_vertex_pieces_are_projected_from_full_boundaries(self):
        array, mapper, expected = _scene()
        recognizer = SourceProjectiveRecognition()
        pieces, diagnostics = recognizer.detect(
            _SourceGrayImage(array.copy()),
            mapper.a4_polygon_source_px,
            generation=3,
            collect_sanity=True,
        )
        self.assertEqual(diagnostics["backend"], "source_projective")
        self.assertEqual(diagnostics["rotation_corr_calls"], 0)
        self.assertEqual(diagnostics["source_side"], "top")
        self.assertEqual(len(pieces), 3, diagnostics)
        self.assertEqual(
            sorted(len(piece.polygon_mm) for piece in pieces), [3, 4, 5]
        )
        self.assertTrue(all(piece.calibration_generation == 3 for piece in pieces))
        expected_centers = []
        for polygon in expected:
            points = np.array(polygon, dtype=np.float32)
            expected_centers.append(tuple(points.mean(axis=0)))
        measured_centers = sorted(piece.centroid_mm for piece in pieces)
        expected_centers = sorted(expected_centers)
        for measured, expected_center in zip(measured_centers, expected_centers):
            self.assertAlmostEqual(measured[0], expected_center[0], delta=5.0)
            self.assertAlmostEqual(measured[1], expected_center[1], delta=5.0)
        expected_areas = sorted(
            abs(cv2.contourArea(np.array(polygon, dtype=np.float32)))
            for polygon in expected
        )
        measured_areas = sorted(piece.area_mm2 for piece in pieces)
        for measured, expected_area in zip(measured_areas, expected_areas):
            self.assertAlmostEqual(
                measured, expected_area, delta=0.15 * expected_area
            )
        self.assertTrue(
            all(
                len(piece.edge_lengths_mm) == len(piece.polygon_mm)
                and min(piece.edge_lengths_mm) > 15.0
                for piece in pieces
            )
        )

    def test_corner_jitter_does_not_change_raw_source_contours(self):
        array, mapper, _pieces = _scene()
        divider = detect_source_divider(_SourceGrayImage(array), mapper)
        baseline_mask = SourceScanlineMask(mapper, divider, "top")
        baseline, _ = detect_pieces_from_source_projective_image(
            _SourceGrayImage(array.copy()),
            mapper,
            divider,
            source_side="top",
            scanline_mask=baseline_mask,
            generation=1,
        )
        for jitter in (1.0, 2.0, 4.0):
            shifted_corners = [
                (
                    point[0] + (jitter if index % 2 else -jitter),
                    point[1] + (jitter if index < 2 else -jitter),
                )
                for index, point in enumerate(mapper.a4_polygon_source_px)
            ]
            shifted = A4ProjectiveMapper(shifted_corners, 640, 384)
            shifted_mask = SourceScanlineMask(shifted, divider, "top")
            observations, _ = detect_pieces_from_source_projective_image(
                _SourceGrayImage(array.copy()),
                shifted,
                divider,
                source_side="top",
                scanline_mask=shifted_mask,
                generation=1,
            )
            self.assertEqual(
                [piece.contour_px for piece in observations],
                [piece.contour_px for piece in baseline],
            )

    def test_bbox_filter_matches_blacken_without_mutating_image(self):
        array, mapper, _pieces = _scene()
        divider = detect_source_divider(_SourceGrayImage(array), mapper)
        mask = SourceScanlineMask(mapper, divider, "top")
        blackened_array = array.copy()
        baseline, baseline_diagnostics = (
            detect_pieces_from_source_projective_image(
                _SourceGrayImage(blackened_array),
                mapper,
                divider,
                source_side="top",
                scanline_mask=mask,
                generation=1,
                mask_mode="blacken",
            )
        )
        bbox_array = array.copy()
        optimized, optimized_diagnostics = (
            detect_pieces_from_source_projective_image(
                _SourceGrayImage(bbox_array),
                mapper,
                divider,
                source_side="top",
                scanline_mask=mask,
                generation=1,
                mask_mode="bbox_filter",
            )
        )
        self.assertEqual(
            [piece.contour_px for piece in optimized],
            [piece.contour_px for piece in baseline],
        )
        self.assertFalse(np.array_equal(blackened_array, array))
        self.assertTrue(np.array_equal(bbox_array, array))
        self.assertGreater(
            baseline_diagnostics["source_masked_pixels"], 0
        )
        self.assertEqual(
            optimized_diagnostics["source_masked_pixels"], 0
        )
        self.assertEqual(
            optimized_diagnostics["source_mask_mode"], "bbox_filter"
        )

    def test_static_scene_freezes_and_reuses_calibration_work(self):
        cfg.SOURCE_PROJECTIVE_FREEZE_DIVIDER = True
        cfg.SOURCE_PROJECTIVE_FREEZE_SOURCE_HALF = True
        cfg.SOURCE_PROJECTIVE_CACHE_BACKGROUND = True
        cfg.SOURCE_PROJECTIVE_MASK_MODE = "bbox_filter"
        array, mapper, _pieces = _scene()
        recognizer = SourceProjectiveRecognition()

        first, first_diagnostics = recognizer.detect(
            _SourceGrayImage(array.copy()),
            mapper.a4_polygon_source_px,
            generation=4,
            sample_id=1,
        )
        same_frame, retry_diagnostics = recognizer.detect(
            _SourceGrayImage(array.copy()),
            mapper.a4_polygon_source_px,
            generation=4,
            sample_id=1,
        )
        second, second_diagnostics = recognizer.detect(
            _SourceGrayImage(array.copy()),
            mapper.a4_polygon_source_px,
            generation=4,
            sample_id=2,
        )
        third, third_diagnostics = recognizer.detect(
            _SourceGrayImage(array.copy()),
            mapper.a4_polygon_source_px,
            generation=4,
            sample_id=3,
        )

        self.assertEqual(first, [])
        self.assertEqual(same_frame, [])
        self.assertEqual(first_diagnostics["reason"], "divider_confirming")
        self.assertEqual(retry_diagnostics["divider_confirmations"], 1)
        self.assertEqual(retry_diagnostics["divider_detection_count"], 1)
        self.assertEqual(len(second), 3, second_diagnostics)
        self.assertEqual(len(third), 3, third_diagnostics)
        self.assertTrue(second_diagnostics["divider_frozen"])
        self.assertTrue(second_diagnostics["source_half_frozen"])
        self.assertEqual(third_diagnostics["divider_detection_count"], 2)
        self.assertEqual(third_diagnostics["source_half_estimation_count"], 1)
        self.assertEqual(third_diagnostics["background_estimation_count"], 1)
        self.assertEqual(third_diagnostics["scanline_mask_build_count"], 1)
        self.assertTrue(third_diagnostics["scanline_mask_reused"])
        self.assertTrue(third_diagnostics["background_cached"])
        self.assertTrue(third_diagnostics["threshold_cached"])


class RelockTests(unittest.TestCase):
    def test_continuous_tracker_can_freeze_after_initial_lock(self):
        tracker = A4BoundaryTracker(continuous=True)
        base_corners = [(10, 10), (210, 10), (210, 310), (10, 310)]
        old_required = cfg.A4_LOCK_REQUIRED_FRAMES
        old_spread = cfg.A4_LOCK_MAX_SPREAD_PX
        cfg.A4_LOCK_REQUIRED_FRAMES = 2
        cfg.A4_LOCK_MAX_SPREAD_PX = 2.0
        try:
            for offset in (0.5, -0.5):
                state = tracker.update(
                    {
                        "corners_px": [
                            (point[0] + offset, point[1])
                            for point in base_corners
                        ],
                        "confidence": 0.9,
                        "source": "test",
                    }
                )
            self.assertTrue(state["locked"])
            frozen = tracker.freeze()
            frozen_corners = list(frozen["corners_px"])
            moved = tracker.update(
                {
                    "corners_px": [
                        (point[0] + 30.0, point[1])
                        for point in base_corners
                    ],
                    "confidence": 0.9,
                    "source": "test",
                }
            )
        finally:
            cfg.A4_LOCK_REQUIRED_FRAMES = old_required
            cfg.A4_LOCK_MAX_SPREAD_PX = old_spread
        self.assertTrue(moved["frozen"])
        self.assertEqual(moved["corners_px"], frozen_corners)

    def test_continuous_tracker_invalidates_then_increments_generation(self):
        tracker = A4BoundaryTracker(continuous=True)
        base_corners = [(10, 10), (210, 10), (210, 310), (10, 310)]

        def candidate(offset):
            return {
                "corners_px": [
                    (point[0] + offset, point[1]) for point in base_corners
                ],
                "confidence": 0.9,
                "source": "test",
            }

        for offset in (0.5, -0.5, 0.0):
            state = tracker.update(candidate(offset))
        self.assertTrue(state["locked"])
        self.assertEqual(state["calibration_generation"], 1)
        for _ in range(cfg.A4_RELOCK_CONFIRM_FRAMES):
            state = tracker.update(candidate(20.0))
        self.assertFalse(state["locked"])
        self.assertTrue(state["relock_required"])
        for offset in (20.5, 19.5):
            state = tracker.update(candidate(offset))
        self.assertTrue(state["locked"])
        self.assertEqual(state["calibration_generation"], 2)
        self.assertFalse(state["relock_required"])

    def test_piece_tracker_never_mixes_calibration_generations(self):
        tracker = PieceTracker(expected_count=1)

        def observation(generation, offset=0.0):
            return PieceObservation(
                "",
                [],
                [
                    (10.0 + offset, 10.0),
                    (40.0 + offset, 10.0),
                    (40.0 + offset, 35.0),
                    (10.0 + offset, 35.0),
                ],
                calibration_generation=generation,
            )

        tracker.update([observation(1)])
        tracker.update([observation(1, 0.1)])
        pieces, stable = tracker.update([observation(2, 0.1)])
        self.assertFalse(stable)
        self.assertEqual(tracker.calibration_generation, 2)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(len(tracker.tracks[0].history), 1)
        self.assertEqual(pieces[0].calibration_generation, 2)


if __name__ == "__main__":
    unittest.main()
