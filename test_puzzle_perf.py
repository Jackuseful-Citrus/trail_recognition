"""Tests for optional low-overhead performance instrumentation."""

import unittest

from puzzle_perf import PerformanceStats


class PerformanceStatsTests(unittest.TestCase):
    def test_disabled_stats_are_noop(self):
        stats = PerformanceStats(enabled=False)
        self.assertEqual(stats.begin_frame(3), 0)
        stats.increment("pixel_reads", 5)
        stats.add_stage("capture_ms", elapsed_ms=7)
        self.assertEqual(stats.window_snapshot()["counters"], {})

    def test_stage_average_and_counter(self):
        stats = PerformanceStats(enabled=True)
        stats.begin_frame(1)
        stats.add_stage("capture_ms", elapsed_ms=4)
        stats.increment("pixel_reads", 9)
        stats.end_frame()
        snapshot = stats.window_snapshot()
        self.assertEqual(snapshot["frames"], 1)
        self.assertEqual(snapshot["averages_ms"]["capture_ms"], 4.0)
        self.assertEqual(snapshot["counters"]["pixel_reads"], 9)
        self.assertIn("[PERF]", stats.format_report())


if __name__ == "__main__":
    unittest.main()
