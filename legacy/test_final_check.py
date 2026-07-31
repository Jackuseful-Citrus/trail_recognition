"""Focused tests for the retained source-clear completion flow."""

import unittest

import puzzle_config as cfg
from puzzle_placement import (
    final_foreground_mask_from_gray,
    final_region_white_metrics,
)
from puzzle_realtime_state import (
    FinalCheckState,
    divider_overlay_endpoints,
    phase_allows_vision,
)


class FinalCheckTests(unittest.TestCase):
    WIDTH = 210
    HEIGHT = 297
    class _GrayRows(list):
        def __init__(self, rows):
            super().__init__(rows)
            self.shape = (len(rows), len(rows[0]))

    def test_final_mask_excludes_border_and_divider(self):
        gray = self._GrayRows([[255] * 6 for _ in range(6)])
        mask = final_foreground_mask_from_gray(
            gray,
            180,
            border_px=1,
            divider_y_mm=cfg.A4_HEIGHT_MM * 2.5 / 6.0,
        )
        self.assertEqual(sum(mask), 12)

    def test_final_region_metrics_split_source_and_target(self):
        mask = bytearray(4 * 4)
        mask[0:4] = b"\x01" * 4
        mask[12:16] = b"\x01" * 4
        metrics = final_region_white_metrics(
            mask,
            4,
            4,
            1000.0,
        )
        self.assertEqual(metrics["upper_foreground_count"], 4)
        self.assertEqual(metrics["lower_foreground_count"], 4)

    def test_source_clear_requires_consecutive_still_frames(self):
        flow = FinalCheckState(
            10,
            cfg.FINAL_TRIGGER_UPPER_REMAINING_RATIO_MAX,
        )
        for _ in range(9):
            state = flow.update(False, 0.0, 0.0)
        self.assertFalse(state["trigger_complete"])
        self.assertEqual(flow.phase, "WAIT_FINAL_CHECK")
        self.assertEqual(flow.stable_frames, 9)

        # A bright source frame or any motion breaks the consecutive streak.
        state = flow.update(False, 0.10, 2.0)
        self.assertEqual(flow.stable_frames, 0)
        self.assertFalse(state["source_clear"])
        state = flow.update(True, 0.0, 0.0)
        self.assertEqual(flow.stable_frames, 0)

        # The lower/target half is diagnostic only and cannot veto completion.
        for _ in range(10):
            state = flow.update(False, 0.0, 0.0)
        self.assertTrue(state["trigger_complete"])
        self.assertTrue(state["source_clear"])
        self.assertEqual(flow.phase, "COMPLETE")
        self.assertEqual(state["stable_frames"], 10)

    def test_final_phases_never_run_piece_detection(self):
        self.assertTrue(phase_allows_vision("WAIT_FINAL_CHECK"))
        self.assertFalse(phase_allows_vision("COMPLETE"))

    def test_divider_overlay_requires_confirmation_and_keeps_slope(self):
        self.assertIsNone(
            divider_overlay_endpoints(None, cfg.A4_WIDTH_MM)
        )
        self.assertIsNone(
            divider_overlay_endpoints(
                {
                    "detected": False,
                    "divider_y_mm": 148.5,
                    "slope_mm": 0.0,
                },
                cfg.A4_WIDTH_MM,
            )
        )
        self.assertEqual(
            divider_overlay_endpoints(
                {
                    "detected": True,
                    "divider_y_mm": 150.0,
                    "slope_mm": 4.0,
                },
                cfg.A4_WIDTH_MM,
            ),
            ((0.0, 148.0), (210.0, 152.0)),
        )


if __name__ == "__main__":
    unittest.main()
