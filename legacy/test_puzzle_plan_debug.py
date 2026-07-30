"""Tests for the low-frequency CanMV planning heartbeat."""

import contextlib
import io
import unittest
from unittest.mock import patch

import puzzle_config as cfg
import puzzle_geometry as geometry


class PlanDebugTests(unittest.TestCase):
    def setUp(self):
        self.original_enabled = cfg.ENABLE_PLAN_DEBUG
        self.original_interval = cfg.PLAN_DEBUG_INTERVAL_MS
        geometry.end_plan_debug()

    def tearDown(self):
        geometry.end_plan_debug()
        cfg.ENABLE_PLAN_DEBUG = self.original_enabled
        cfg.PLAN_DEBUG_INTERVAL_MS = self.original_interval

    def test_disabled_debug_is_a_noop(self):
        cfg.ENABLE_PLAN_DEBUG = False
        geometry.begin_plan_debug("fixed_rectangle", 4)
        self.assertIsNone(geometry.PLAN_DEBUG_STATE)
        self.assertFalse(geometry.plan_debug_heartbeat(force=True))

    def test_heartbeat_is_rate_limited_and_reports_progress(self):
        cfg.ENABLE_PLAN_DEBUG = True
        cfg.PLAN_DEBUG_INTERVAL_MS = 2000
        output = io.StringIO()
        with patch.object(
            geometry,
            "ticks_ms",
            side_effect=(1000, 2500, 3100),
        ):
            geometry.begin_plan_debug("fixed_rectangle", 4)
            geometry.update_plan_debug(
                stage="fixed_rank",
                depth=3,
                states=1200,
                expanded=4096,
                nodes=7777,
                best_score=0.1234,
            )
            with contextlib.redirect_stdout(output):
                self.assertFalse(
                    geometry.plan_debug_heartbeat()
                )
                self.assertTrue(
                    geometry.plan_debug_heartbeat()
                )

        line = output.getvalue().strip()
        self.assertEqual(line.count("PLAN_DEBUG,"), 1)
        self.assertIn("planner=fixed_rectangle", line)
        self.assertIn("stage=fixed_rank", line)
        self.assertIn("elapsed_ms=2100", line)
        self.assertIn("states=1200", line)
        self.assertIn("expanded=4096", line)
        self.assertIn("nodes=7777", line)
        self.assertIn("best_score=0.1234", line)


if __name__ == "__main__":
    unittest.main()
