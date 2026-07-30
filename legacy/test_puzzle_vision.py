"""Offline regression test for the user-supplied puzzle photograph."""

import unittest

import cv2
import numpy as np

import puzzle_config as cfg
from puzzle_geometry import (
    plan_outer_first_rectangle,
    plan_rectangle_assembly,
)
from puzzle_vision import (
    _finalize_fitted_polygon,
    _ordered_contour_polygon,
    _ordered_contour_polygon_once,
    background_difference_threshold,
    detect_pieces_from_canmv_image,
    detect_pieces_from_gray,
    estimate_background_gray,
    polygon_white_coverage,
)


class _FakeBlob:
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


class _FakeStatistics:
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


class _FakeCanMVGrayImage:
    """Minimal native-image facade for the no-cv2 board path."""

    def __init__(self, array):
        self.array = array

    def width(self):
        return self.array.shape[1]

    def height(self):
        return self.array.shape[0]

    def rotation_corr(self, *, corners):
        # The test supplies full-frame corners, so the expected warp is identity.
        expected = [
            (0, 0),
            (self.width() - 1, 0),
            (self.width() - 1, self.height() - 1),
            (0, self.height() - 1),
        ]
        if list(corners) != expected:
            raise AssertionError((corners, expected))
        return self

    def format(self):
        return "GRAYSCALE"

    def get_statistics(self):
        return _FakeStatistics(self.array)

    def get_pixel(self, x, y):
        return int(self.array[y, x])

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
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        result = []
        for label in range(1, count):
            left, top, blob_width, blob_height, pixels = stats[label]
            if (
                pixels < pixels_threshold
                or blob_width * blob_height < area_threshold
            ):
                continue
            result.append(
                _FakeBlob(
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
        return result

    def to_numpy_ref(self):
        return self.array


class VisionRegressionTests(unittest.TestCase):
    def test_self_intersecting_fit_is_rejected_before_observation(self):
        bow_tie = [
            (0, 0),
            (40, 30),
            (0, 30),
            (30, 0),
        ]
        self.assertIsNone(
            _finalize_fitted_polygon(bow_tie, bow_tie)
        )

    def test_background_threshold_tracks_global_brightness(self):
        dark = np.full((120, 160), 70, dtype=np.uint8)
        dark[58:62, :] = 12
        dark[78:120, :] = 190
        bright = np.full((120, 160), 120, dtype=np.uint8)
        bright[58:62, :] = 20
        bright[78:120, :] = 240

        dark_stats = estimate_background_gray(
            dark, (0, 0, 160, 120), sample_stride=2
        )
        bright_stats = estimate_background_gray(
            bright, (0, 0, 160, 120), sample_stride=2
        )
        dark_threshold = background_difference_threshold(
            dark_stats, 30, 12, 55
        )
        bright_threshold = background_difference_threshold(
            bright_stats, 30, 12, 55
        )

        self.assertAlmostEqual(
            dark_stats["background_gray"], 70.0, delta=2.0
        )
        self.assertAlmostEqual(
            bright_stats["background_gray"], 120.0, delta=2.0
        )
        self.assertEqual(
            bright_threshold - dark_threshold, 50
        )
        self.assertEqual(dark_threshold, 100)
        self.assertEqual(bright_threshold, 150)

    def test_canmv_native_detection_uses_live_divider_position(self):
        width = cfg.CANMV_WORK_WIDTH
        height = cfg.CANMV_WORK_HEIGHT
        gray = np.full((height, width), 25, dtype=np.uint8)
        corners = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        divider_y_mm = cfg.DIVIDER_Y_MM - 6.0
        _, diagnostics = detect_pieces_from_canmv_image(
            _FakeCanMVGrayImage(gray),
            corners,
            (width, height),
            divider_y_mm=divider_y_mm,
        )
        self.assertTrue(diagnostics["divider_detected"])
        self.assertAlmostEqual(
            diagnostics["divider_y_mm"],
            divider_y_mm,
        )
        pixels_per_mm_y = float(height - 1) / cfg.A4_HEIGHT_MM
        expected_end = max(
            2,
            int(divider_y_mm * pixels_per_mm_y + 0.5)
            - int(2.0 * pixels_per_mm_y + 0.5),
        )
        self.assertEqual(
            diagnostics["detection_end_row"],
            expected_end,
        )

    def test_canmv_background_delta_detects_under_two_exposures(self):
        width = cfg.CANMV_WORK_WIDTH
        height = cfg.CANMV_WORK_HEIGHT
        scale_x = float(width - 1) / cfg.A4_WIDTH_MM
        scale_y = float(height - 1) / cfg.A4_HEIGHT_MM

        def px(points_mm):
            return np.array(
                [
                    (
                        int(round(x * scale_x)),
                        int(round(y * scale_y)),
                    )
                    for x, y in points_mm
                ],
                dtype=np.int32,
            )

        polygons = [
            [(10, 12), (42, 12), (42, 42), (10, 42)],
            [(55, 18), (105, 18), (105, 55)],
            [(12, 72), (56, 72), (56, 101), (12, 101)],
            [(92, 76), (139, 76), (139, 111), (92, 111)],
        ]
        corners = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        old_mode = cfg.PIECE_SEGMENTATION_MODE
        cfg.PIECE_SEGMENTATION_MODE = "background_delta"
        try:
            results = []
            for background, foreground in ((80, 160), (125, 205)):
                gray = np.full(
                    (height, width),
                    background,
                    dtype=np.uint8,
                )
                divider = int(
                    cfg.DIVIDER_Y_MM * scale_y + 0.5
                )
                gray[
                    max(0, divider - 1) : divider + 2, :
                ] = 15
                for polygon in polygons:
                    cv2.fillPoly(
                        gray, [px(polygon)], foreground
                    )
                pieces, diagnostics = (
                    detect_pieces_from_canmv_image(
                        _FakeCanMVGrayImage(gray),
                        corners,
                        (width, height),
                    )
                )
                results.append((pieces, diagnostics))
        finally:
            cfg.PIECE_SEGMENTATION_MODE = old_mode

        for pieces, diagnostics in results:
            self.assertEqual(len(pieces), 4)
            self.assertEqual(
                diagnostics["threshold_mode"],
                "background_delta",
            )
            self.assertEqual(diagnostics["raw_contours"], 4)
            self.assertGreaterEqual(
                diagnostics["background_sample_count"],
                cfg.PIECE_BACKGROUND_MIN_SAMPLES,
            )
        self.assertEqual(
            int(results[1][1]["threshold"])
            - int(results[0][1]["threshold"]),
            45,
        )

    def test_background_delta_traces_white_edge_not_gray_shadow(self):
        width = cfg.CANMV_WORK_WIDTH
        height = cfg.CANMV_WORK_HEIGHT
        scale_x = float(width - 1) / cfg.A4_WIDTH_MM
        scale_y = float(height - 1) / cfg.A4_HEIGHT_MM

        def px(points_mm):
            return np.array(
                [
                    (
                        int(round(x * scale_x)),
                        int(round(y * scale_y)),
                    )
                    for x, y in points_mm
                ],
                dtype=np.int32,
            )

        gray = np.full((height, width), 20, dtype=np.uint8)
        true_polygon = [
            (40, 25),
            (90, 25),
            (90, 65),
            (40, 65),
        ]
        shadow_polygon = [
            (x + 7, y + 6) for x, y in true_polygon
        ]
        cv2.fillPoly(gray, [px(shadow_polygon)], 70)
        cv2.fillPoly(gray, [px(true_polygon)], 205)
        corners = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        old_mode = cfg.PIECE_SEGMENTATION_MODE
        old_contour_floor = getattr(
            cfg, "PIECE_CONTOUR_MIN_GRAY_THRESHOLD", 0
        )
        cfg.PIECE_SEGMENTATION_MODE = "background_delta"
        cfg.PIECE_CONTOUR_MIN_GRAY_THRESHOLD = 100
        try:
            pieces, diagnostics = (
                detect_pieces_from_canmv_image(
                    _FakeCanMVGrayImage(gray),
                    corners,
                    (width, height),
                )
            )
        finally:
            cfg.PIECE_SEGMENTATION_MODE = old_mode
            cfg.PIECE_CONTOUR_MIN_GRAY_THRESHOLD = (
                old_contour_floor
            )
        self.assertEqual(len(pieces), 1)
        self.assertLess(diagnostics["threshold"], 70)
        self.assertEqual(
            diagnostics["contour_threshold"], 100
        )
        self.assertAlmostEqual(
            pieces[0].centroid_mm[0], 65.0, delta=1.5
        )
        self.assertAlmostEqual(
            pieces[0].centroid_mm[1], 45.0, delta=1.5
        )
        self.assertAlmostEqual(
            pieces[0].area_mm2, 2000.0, delta=160.0
        )

    def test_canmv_threshold_override_recovers_dim_piece(self):
        width = cfg.CANMV_WORK_WIDTH
        height = cfg.CANMV_WORK_HEIGHT
        gray = np.zeros((height, width), dtype=np.uint8)
        scale_x = float(width - 1) / cfg.A4_WIDTH_MM
        scale_y = float(height - 1) / cfg.A4_HEIGHT_MM
        points = np.array(
            [
                (
                    int(round(x * scale_x)),
                    int(round(y * scale_y)),
                )
                for x, y in (
                    (30, 30),
                    (85, 30),
                    (85, 70),
                    (30, 70),
                )
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(gray, [points], 170)
        corners = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        primary, _ = detect_pieces_from_canmv_image(
            _FakeCanMVGrayImage(gray.copy()),
            corners,
            (width, height),
        )
        recovered, diagnostics = (
            detect_pieces_from_canmv_image(
                _FakeCanMVGrayImage(gray.copy()),
                corners,
                (width, height),
                threshold=165,
            )
        )
        self.assertEqual(primary, [])
        self.assertEqual(len(recovered), 1)
        self.assertEqual(diagnostics["threshold"], 165.0)

    def test_canmv_native_backend_needs_no_cv2_module(self):
        width = cfg.CANMV_WORK_WIDTH
        height = cfg.CANMV_WORK_HEIGHT
        gray = np.zeros((height, width), dtype=np.uint8)
        scale_x = float(width - 1) / cfg.A4_WIDTH_MM
        scale_y = float(height - 1) / cfg.A4_HEIGHT_MM

        def px(points_mm):
            return np.array(
                [
                    (
                        int(round(x * scale_x)),
                        int(round(y * scale_y)),
                    )
                    for x, y in points_mm
                ],
                dtype=np.int32,
            )

        polygons = [
            [(10, 12), (42, 12), (42, 42), (10, 42)],
            [(55, 18), (105, 18), (105, 55)],
            [(12, 72), (56, 72), (56, 101), (12, 101)],
            [(92, 76), (139, 76), (139, 111), (92, 111)],
        ]
        for polygon in polygons:
            cv2.fillPoly(gray, [px(polygon)], 255)

        native_image = _FakeCanMVGrayImage(gray)
        corners = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        pieces, diagnostics = detect_pieces_from_canmv_image(
            native_image,
            corners,
            (width, height),
        )
        self.assertEqual(len(pieces), 4)
        self.assertEqual(
            sorted(len(piece.polygon_mm) for piece in pieces),
            [3, 4, 4, 4],
        )
        self.assertEqual(diagnostics["backend"], "canmv_image")
        self.assertEqual(
            diagnostics["threshold_mode"], "fixed_native"
        )
        self.assertEqual(
            diagnostics["boundary_fallback_count"], 0
        )
        self.assertGreater(diagnostics["boundary_steps"], 0)
        self.assertGreater(diagnostics["pixel_reads"], 0)

    def test_canmv_gray_sanity_compares_native_and_shared_array(self):
        width = cfg.CANMV_WORK_WIDTH
        height = cfg.CANMV_WORK_HEIGHT
        gray = np.zeros((height, width), dtype=np.uint8)
        gray[24:64, 30:90] = 230
        image = _FakeCanMVGrayImage(gray)
        corners = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        _, diagnostics = detect_pieces_from_canmv_image(
            image,
            corners,
            (width, height),
            collect_sanity=True,
        )
        sanity = diagnostics["gray_sanity"]
        self.assertEqual(sanity["rotation_return"], "self")
        self.assertEqual(sanity["native"]["format"], "GRAYSCALE")
        self.assertEqual(sanity["native"]["max"], "230")
        self.assertEqual(sanity["upper"]["max"], 230)
        self.assertEqual(sanity["upper"]["bright"], 2400)
        self.assertEqual(
            sanity["upper"]["bright_bbox"],
            (30, 24, 60, 40),
        )
        self.assertEqual(sanity["lower"]["bright"], 0)

    def test_closed_contour_fit_retries_reverse_direction(self):
        mask = np.zeros((150, 150), dtype=np.uint8)
        cv2.fillPoly(
            mask,
            [
                np.array(
                    [(101, 78), (83, 87), (45, 23), (62, 53)],
                    dtype=np.int32,
                )
            ],
            255,
        )
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        contour = max(contours, key=cv2.contourArea)
        points = [
            (
                float(item[0][0]) * 0.88,
                float(item[0][1]) * 0.88,
            )
            for item in contour
        ]
        tolerance = max(0.05, cfg.CONTOUR_DP_TOLERANCE_MM)
        self.assertIsNone(
            _ordered_contour_polygon_once(points, tolerance)
        )
        diagnostics = {}
        recovered = _ordered_contour_polygon(
            points, diagnostics
        )
        self.assertIsNotNone(recovered)
        self.assertEqual(
            diagnostics["polygon_fit_method"], "reverse"
        )
        self.assertTrue(
            diagnostics["polygon_fit_reverse_used"]
        )

    def test_closed_contour_fit_keeps_valid_unrefined_polygon(self):
        mask = np.zeros((160, 160), dtype=np.uint8)
        cv2.fillPoly(
            mask,
            [
                np.array(
                    [(89, 82), (93, 114), (65, 120), (118, 66)],
                    dtype=np.int32,
                )
            ],
            255,
        )
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        contour = max(contours, key=cv2.contourArea)
        points = [
            (
                float(item[0][0]) * 0.88,
                float(item[0][1]) * 0.88,
            )
            for item in contour
        ]
        tolerance = max(0.05, cfg.CONTOUR_DP_TOLERANCE_MM)
        self.assertIsNone(
            _ordered_contour_polygon_once(points, tolerance)
        )
        self.assertIsNone(
            _ordered_contour_polygon_once(
                list(reversed(points)), tolerance
            )
        )
        diagnostics = {}
        recovered = _ordered_contour_polygon(
            points, diagnostics
        )
        self.assertIsNotNone(recovered)
        self.assertEqual(
            diagnostics["polygon_fit_method"],
            "forward_unrefined",
        )
        self.assertFalse(
            diagnostics["polygon_fit_reverse_used"]
        )
        self.assertTrue(
            diagnostics["polygon_fit_unrefined_used"]
        )

    def test_canmv_full_a4_mode_detects_upper_and_lower_pieces(self):
        width = cfg.CANMV_WORK_WIDTH
        height = cfg.CANMV_WORK_HEIGHT
        gray = np.zeros((height, width), dtype=np.uint8)
        scale_x = float(width - 1) / cfg.A4_WIDTH_MM
        scale_y = float(height - 1) / cfg.A4_HEIGHT_MM

        def px(points_mm):
            return np.array(
                [
                    (
                        int(round(x * scale_x)),
                        int(round(y * scale_y)),
                    )
                    for x, y in points_mm
                ],
                dtype=np.int32,
            )

        cv2.fillPoly(
            gray,
            [px([(18, 24), (72, 26), (39, 70)])],
            255,
        )
        cv2.fillPoly(
            gray,
            [
                px(
                    [
                        (96, 196),
                        (151, 198),
                        (146, 239),
                        (101, 235),
                    ]
                )
            ],
            255,
        )
        image = _FakeCanMVGrayImage(gray)
        corners = [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (0, height - 1),
        ]
        pieces, diagnostics = detect_pieces_from_canmv_image(
            image,
            corners,
            (width, height),
            region="full",
        )
        self.assertEqual(len(pieces), 2)
        self.assertEqual(diagnostics["region"], "full")
        self.assertEqual(len(diagnostics["detection_regions"]), 2)
        self.assertTrue(
            any(
                piece.centroid_mm[1] < cfg.DIVIDER_Y_MM
                for piece in pieces
            )
        )
        self.assertTrue(
            any(
                piece.centroid_mm[1] > cfg.DIVIDER_Y_MM
                for piece in pieces
            )
        )

        target = [
            (96, 196),
            (151, 198),
            (146, 239),
            (101, 235),
        ]
        coverage = polygon_white_coverage(gray, target)
        self.assertGreater(coverage, 0.90)

    def test_supplied_photo_detects_four_pieces(self):
        image = cv2.imread(cfg.OFFLINE_IMAGE, cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        pieces, diagnostics = detect_pieces_from_gray(
            image,
            cfg.OFFLINE_A4_CORNERS_PX,
            cv2,
            np,
        )
        self.assertEqual(len(pieces), 4)
        self.assertEqual(
            [len(piece.polygon_mm) for piece in pieces],
            [3, 4, 4, 4],
        )
        self.assertTrue(diagnostics["divider_detected"])
        self.assertTrue(
            all(3 <= len(piece.polygon_mm) <= 5 for piece in pieces)
        )
        self.assertTrue(
            all(
                piece.centroid_mm[1] < diagnostics["divider_y_mm"]
                for piece in pieces
            )
        )
        # Guards against the horizontal divider being accepted as a piece.
        self.assertTrue(
            all(piece.area_mm2 < cfg.MAX_PIECE_AREA_MM2 for piece in pieces)
        )
        for index, piece in enumerate(pieces):
            piece.piece_id = "P{}".format(index + 1)
        plan = plan_rectangle_assembly(pieces)
        self.assertTrue(plan.valid, plan.reason)
        self.assertEqual(plan.mode, "fixed_tolerant")
        self.assertLessEqual(
            plan.max_vertex_error_mm,
            cfg.CORRESPONDING_VERTEX_TOLERANCE_MM,
        )
        unknown_plan = plan_outer_first_rectangle(pieces)
        self.assertTrue(unknown_plan.valid, unknown_plan.reason)
        self.assertEqual(
            unknown_plan.mode, "corner_outer_strict"
        )
        target_width = (
            unknown_plan.target_rect[2]
            - unknown_plan.target_rect[0]
        )
        target_height = (
            unknown_plan.target_rect[3]
            - unknown_plan.target_rect[1]
        )
        self.assertEqual(
            sorted(
                (round(target_width, 1), round(target_height, 1))
            ),
            [60.0, 100.0],
        )
        self.assertLessEqual(
            unknown_plan.search_nodes,
            cfg.OUTER_FIRST_CORNER_MAX_SEARCH_NODES,
        )


if __name__ == "__main__":
    unittest.main()
