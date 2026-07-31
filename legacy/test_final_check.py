"""Focused tests for source-clear completion and offline final metrics."""

import unittest

import puzzle_config as cfg
from puzzle_placement import (
    final_frame_pass,
    final_rectangle_consensus,
    final_rectangle_metrics,
)
from puzzle_realtime_state import (
    FinalCheckState,
    placement_phase_actions,
)


class FinalCheckTests(unittest.TestCase):
    WIDTH = 210
    HEIGHT = 297
    TARGET = (55.0, 190.0, 155.0, 250.0)

    @classmethod
    def _rectangle_mask(cls, rect):
        mask = bytearray(cls.WIDTH * cls.HEIGHT)
        for y in range(rect[1], rect[3]):
            for x in range(rect[0], rect[2]):
                mask[y * cls.WIDTH + x] = 1
        return mask

    def _metrics(self, rect):
        return final_rectangle_metrics(
            self._rectangle_mask(rect),
            self.WIDTH,
            self.HEIGHT,
            self.TARGET,
            6000.0,
        )

    def test_final_pass(self):
        metrics = self._metrics((55, 190, 155, 250))
        metrics["valid"] = final_frame_pass(metrics, 0.0)
        result = final_rectangle_consensus(
            [metrics, metrics, dict(metrics, valid=False)]
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["pass_count"], 2)

    def test_final_fail(self):
        metrics = self._metrics((15, 190, 115, 250))
        self.assertGreater(metrics["center_error_mm"], 15.0)
        metrics["valid"] = final_frame_pass(metrics, 0.0)
        result = final_rectangle_consensus(
            [metrics, metrics, dict(metrics, valid=True)]
        )
        self.assertFalse(result["valid"])
        self.assertEqual(result["pass_count"], 1)

    def test_final_width_height_swap(self):
        metrics = self._metrics((75, 170, 135, 270))
        self.assertTrue(metrics["dimensions_swapped"])
        self.assertTrue(final_frame_pass(metrics, 0.0))

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
        for phase in ("WAIT_FINAL_CHECK", "COMPLETE"):
            actions = placement_phase_actions(phase)
            self.assertFalse(actions["piece_detection"])
            self.assertFalse(actions["tracker_update"])


if __name__ == "__main__":
    unittest.main()
