"""Lightweight CPython/CanMV performance counters for the puzzle pipeline."""

import time

import puzzle_config as cfg


def ticks_ms():
    function = getattr(time, "ticks_ms", None)
    if function is not None:
        return int(function())
    function = getattr(time, "perf_counter", None)
    if function is not None:
        return int(function() * 1000.0)
    return int(time.time() * 1000.0)


def ticks_diff(newer, older):
    function = getattr(time, "ticks_diff", None)
    if function is not None:
        return int(function(newer, older))
    return int(newer - older)


class PerformanceStats:
    """Accumulate named durations and counters without CPython-only features."""

    __slots__ = (
        "enabled",
        "frame_index",
        "frame_started_ms",
        "last_stages",
        "window_stage_ms",
        "window_stage_count",
        "counters",
        "window_frames",
        "boundary_fallback_count",
    )

    def __init__(self, enabled=None):
        if enabled is None:
            enabled = cfg.ENABLE_STAGE_TIMING
        self.enabled = bool(enabled)
        self.frame_index = -1
        self.frame_started_ms = 0
        self.last_stages = {}
        self.window_stage_ms = {}
        self.window_stage_count = {}
        self.counters = {}
        self.window_frames = 0
        self.boundary_fallback_count = 0

    def reset(self):
        enabled = self.enabled
        self.__init__(enabled=enabled)

    def begin_frame(self, frame_index):
        if not self.enabled:
            return 0
        self.frame_index = int(frame_index)
        self.frame_started_ms = ticks_ms()
        self.last_stages = {}
        return self.frame_started_ms

    def mark(self):
        return ticks_ms() if self.enabled else 0

    def add_stage(self, name, started_ms=None, elapsed_ms=None):
        if not self.enabled:
            return 0
        if elapsed_ms is None:
            elapsed_ms = ticks_diff(ticks_ms(), started_ms)
        elapsed_ms = max(0, int(elapsed_ms))
        self.last_stages[name] = (
            self.last_stages.get(name, 0) + elapsed_ms
        )
        self.window_stage_ms[name] = (
            self.window_stage_ms.get(name, 0) + elapsed_ms
        )
        self.window_stage_count[name] = (
            self.window_stage_count.get(name, 0) + 1
        )
        return elapsed_ms

    def increment(self, name, amount=1):
        if not self.enabled:
            return
        self.counters[name] = self.counters.get(name, 0) + int(amount)
        if name == "boundary_fallback_count":
            self.boundary_fallback_count += int(amount)

    def end_frame(self):
        if not self.enabled:
            return 0
        total = ticks_diff(ticks_ms(), self.frame_started_ms)
        self.add_stage("total_frame_ms", elapsed_ms=total)
        self.window_frames += 1
        return total

    def report_due(self, frame_index=None):
        if not self.enabled:
            return False
        if frame_index is None:
            frame_index = self.frame_index
        interval = max(1, int(cfg.TIMING_REPORT_INTERVAL_FRAMES))
        return self.window_frames > 0 and int(frame_index) % interval == 0

    def last_snapshot(self):
        return dict(self.last_stages)

    def window_snapshot(self, reset=False):
        averages = {}
        for name, total in self.window_stage_ms.items():
            averages[name] = float(total) / max(
                1, self.window_stage_count.get(name, 1)
            )
        result = {
            "frames": self.window_frames,
            "averages_ms": averages,
            "last_ms": dict(self.last_stages),
            "counters": dict(self.counters),
        }
        if reset:
            self.window_stage_ms = {}
            self.window_stage_count = {}
            self.counters = {}
            self.window_frames = 0
        return result

    def format_report(self, frame_index=None):
        if frame_index is None:
            frame_index = self.frame_index
        snapshot = self.window_snapshot(reset=False)
        averages = snapshot["averages_ms"]
        ordered = (
            "capture_ms",
            "a4_detect_ms",
            "source_resize_ms",
            "a4_map_build_ms",
            "divider_detect_ms",
            "source_mask_ms",
            "source_blob_ms",
            "source_boundary_ms",
            "boundary_project_ms",
            "polygon_fit_mm_ms",
            "rectify_ms",
            "segment_ms",
            "blob_ms",
            "contour_ms",
            "polygon_fit_ms",
            "candidate_graph_ms",
            "plan_ms",
            "coverage_scan_ms",
            "render_ms",
            "display_ms",
            "total_frame_ms",
        )
        fields = [
            "{}={:.1f}".format(name[:-3], averages[name])
            for name in ordered
            if name in averages
        ]
        return "[PERF] frame={} frames={} {}".format(
            int(frame_index),
            snapshot["frames"],
            " ".join(fields),
        )


PERF_STATS = PerformanceStats()
