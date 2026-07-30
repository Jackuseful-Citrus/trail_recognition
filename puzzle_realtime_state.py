"""Pure state/scheduling helpers shared by K230 runtime and desktop tests."""


def phase_allows_vision(phase):
    """COMPLETE retains the last display but performs no more vision work."""
    return phase != "COMPLETE"


def a4_detection_interval(
    phase,
    acquire_interval,
    placing_interval,
    locked=False,
):
    # The board and camera are fixed.  Once the initial A4 acquisition has
    # locked, keep that exact calibration instead of feeding detector jitter
    # into a different perspective transform every few frames.
    if locked:
        return None
    if phase in (
        "WAIT_FOR_MOTION",
        "MOVING",
        "POST_MOTION_SETTLE",
        "VERIFY_PLACEMENT",
        "FINAL_VERIFY",
        "PLACING",
    ):
        return None
    if phase == "COMPLETE":
        return None
    return max(1, int(acquire_interval))


def countdown_bucket(remaining_ms, refresh_interval_ms):
    """Quantize countdown display updates to remaining whole intervals."""
    remaining = max(0, int(remaining_ms))
    interval = max(1, int(refresh_interval_ms))
    if remaining == 0:
        return 0
    return (remaining + interval - 1) // interval


def _visible_piece_key(placement_state):
    result = []
    for piece in placement_state.get("visible_pieces", ()):
        center = piece.centroid_mm
        result.append(
            (
                piece.piece_id,
                int(round(center[0] * 2.0)),
                int(round(center[1] * 2.0)),
                len(piece.polygon_mm),
            )
        )
    return tuple(result)


def placement_ui_key(
    phase,
    placement_state,
    remaining_ms,
    refresh_interval_ms,
    error=None,
):
    """Return a compact key containing every dynamic placement UI field."""
    return (
        phase,
        tuple(sorted(placement_state.get("completed", ()))),
        placement_state.get("next_piece_id"),
        int(placement_state.get("observed_count", 0)),
        int(placement_state.get("matched_count", 0)),
        _visible_piece_key(placement_state),
        countdown_bucket(remaining_ms, refresh_interval_ms),
        str(error or ""),
    )


def status_ui_key(
    phase,
    locked,
    stable,
    piece_count,
    plan_valid,
    error=None,
):
    return (
        phase,
        bool(locked),
        bool(stable),
        int(piece_count),
        bool(plan_valid),
        str(error or ""),
    )


def should_render_ui(last_rendered_state, current_state):
    return last_rendered_state != current_state


def bottom_right_thumbnail_rect(
    canvas_width,
    canvas_height,
    source_width,
    source_height,
    max_width,
    max_height,
    margin,
):
    """Fit an aspect-preserving thumbnail against the bottom-right corner."""
    canvas_width = max(1, int(canvas_width))
    canvas_height = max(1, int(canvas_height))
    source_width = max(1, int(source_width))
    source_height = max(1, int(source_height))
    margin = max(0, int(margin))
    available_width = max(
        1, min(int(max_width), canvas_width - 2 * margin)
    )
    available_height = max(
        1, min(int(max_height), canvas_height - 2 * margin)
    )
    scale = min(
        float(available_width) / source_width,
        float(available_height) / source_height,
    )
    width = max(1, int(source_width * scale + 0.5))
    height = max(1, int(source_height * scale + 0.5))
    x = max(0, canvas_width - margin - width)
    y = max(0, canvas_height - margin - height)
    return (x, y, width, height, scale)


def periodic_output_due(output_index, every_n_outputs):
    """Return whether a throttled side-channel should publish this output."""
    interval = max(1, int(every_n_outputs))
    return max(0, int(output_index)) % interval == 0


def placement_phase_actions(phase):
    """Declare which expensive actions are permitted in each frozen-plan phase."""
    verify = phase in ("VERIFY_PLACEMENT", "FINAL_VERIFY")
    return {
        "motion_detection": phase
        in (
            "WAIT_FOR_MOTION",
            "MOVING",
            "POST_MOTION_SETTLE",
            "VERIFY_PLACEMENT",
            "FINAL_VERIFY",
        ),
        "piece_detection": verify,
        "tracker_update": False,
        "placement_check": verify,
        "a4_update": False,
    }


def plan_frozen_pieces(
    pieces,
    target_rect_size_mm,
    fixed_planner,
    unknown_planner,
    allow_unknown_fallback=False,
    prefer_outer_first=False,
    preferred_planner_name="outer_first",
):
    """Route one frozen input to the configured planner exactly once."""
    if target_rect_size_mm is None:
        return {
            "plan": unknown_planner(pieces),
            "planner": preferred_planner_name,
            "fallback_used": False,
        }
    if prefer_outer_first:
        return {
            "plan": unknown_planner(
                pieces,
                target_size_mm=target_rect_size_mm,
            ),
            "planner": preferred_planner_name,
            "fallback_used": False,
        }
    fixed_result = fixed_planner(pieces)
    if fixed_result.valid or not allow_unknown_fallback:
        return {
            "plan": fixed_result,
            "planner": "fixed_rectangle",
            "fallback_used": False,
        }
    return {
        "plan": unknown_planner(
            pieces,
            target_size_mm=target_rect_size_mm,
        ),
        "planner": preferred_planner_name,
        "fallback_used": True,
        "fixed_failure_reason": fixed_result.reason,
    }


class MotionDetector:
    """Low-cost adjacent-frame gray difference on sparse A4 samples."""

    __slots__ = (
        "pixel_threshold",
        "mean_threshold",
        "ratio_threshold",
        "sample_stride",
        "divider_start",
        "divider_end",
        "previous",
        "last_metrics",
    )

    def __init__(
        self,
        pixel_threshold,
        mean_threshold,
        ratio_threshold,
        sample_stride=1,
        divider_rows=None,
    ):
        self.pixel_threshold = int(pixel_threshold)
        self.mean_threshold = float(mean_threshold)
        self.ratio_threshold = float(ratio_threshold)
        self.sample_stride = max(1, int(sample_stride))
        if divider_rows is None:
            self.divider_start = -1
            self.divider_end = -1
        else:
            self.divider_start = int(divider_rows[0])
            self.divider_end = int(divider_rows[1])
        self.previous = None
        self.last_metrics = {
            "mean_abs_diff": 0.0,
            "changed_ratio": 0.0,
            "motion": False,
            "sample_count": 0,
        }

    def reset(self):
        self.previous = None
        self.last_metrics = {
            "mean_abs_diff": 0.0,
            "changed_ratio": 0.0,
            "motion": False,
            "sample_count": 0,
        }

    def update(self, gray_array):
        height = int(gray_array.shape[0])
        width = int(gray_array.shape[1])
        current = bytearray()
        for y in range(0, height, self.sample_stride):
            if self.divider_start <= y <= self.divider_end:
                continue
            row = gray_array[y]
            for x in range(0, width, self.sample_stride):
                current.append(int(row[x]))
        if self.previous is None or len(self.previous) != len(current):
            self.previous = current
            self.last_metrics = {
                "mean_abs_diff": 0.0,
                "changed_ratio": 0.0,
                "motion": False,
                "sample_count": len(current),
            }
            return dict(self.last_metrics)
        total = 0
        changed = 0
        for before, after in zip(self.previous, current):
            difference = abs(int(after) - int(before))
            total += difference
            if difference >= self.pixel_threshold:
                changed += 1
        count = max(1, len(current))
        mean_abs_diff = float(total) / count
        changed_ratio = float(changed) / count
        motion = (
            mean_abs_diff >= self.mean_threshold
            and changed_ratio >= self.ratio_threshold
        )
        self.previous = current
        self.last_metrics = {
            "mean_abs_diff": mean_abs_diff,
            "changed_ratio": changed_ratio,
            "motion": motion,
            "sample_count": len(current),
        }
        return dict(self.last_metrics)


class PlacementMotionState:
    """Event-driven move/settle/verify controller with a single verify pulse."""

    __slots__ = (
        "start_confirm_frames",
        "end_confirm_frames",
        "post_stable_frames",
        "phase",
        "motion_count",
        "stable_after_motion_count",
        "motion_start_frame",
        "motion_end_frame",
        "verify_trigger_count",
    )

    def __init__(
        self,
        start_confirm_frames,
        end_confirm_frames,
        post_stable_frames,
    ):
        self.start_confirm_frames = max(
            1, int(start_confirm_frames)
        )
        self.end_confirm_frames = max(1, int(end_confirm_frames))
        self.post_stable_frames = max(1, int(post_stable_frames))
        self.reset()

    def reset(self):
        self.phase = "WAIT_FOR_MOTION"
        self.motion_count = 0
        self.stable_after_motion_count = 0
        self.motion_start_frame = None
        self.motion_end_frame = None
        self.verify_trigger_count = 0

    def update(self, motion, frame_index):
        motion = bool(motion)
        event = None
        motion_ended = False
        trigger_verify = False
        if self.phase == "WAIT_FOR_MOTION":
            if motion:
                self.motion_count += 1
                if self.motion_count == 1:
                    self.motion_start_frame = int(frame_index)
                if self.motion_count >= self.start_confirm_frames:
                    self.phase = "MOVING"
                    self.stable_after_motion_count = 0
                    event = "MOTION_START"
            else:
                self.motion_count = 0
                self.motion_start_frame = None
        elif self.phase in ("MOVING", "POST_MOTION_SETTLE"):
            if motion:
                if self.phase == "POST_MOTION_SETTLE":
                    event = "MOTION_ACTIVE"
                self.phase = "MOVING"
                self.stable_after_motion_count = 0
                self.motion_end_frame = None
            else:
                self.stable_after_motion_count += 1
                if (
                    self.phase == "MOVING"
                ):
                    self.phase = "POST_MOTION_SETTLE"
                if (
                    self.motion_end_frame is None
                    and self.stable_after_motion_count
                    >= self.end_confirm_frames
                ):
                    self.motion_end_frame = int(frame_index)
                    event = "MOTION_END"
                    motion_ended = True
                if (
                    self.phase == "POST_MOTION_SETTLE"
                    and self.stable_after_motion_count
                    >= max(
                        self.end_confirm_frames,
                        self.post_stable_frames,
                    )
                ):
                    self.phase = "VERIFY_PLACEMENT"
                    self.verify_trigger_count += 1
                    event = "POST_MOTION_STABLE"
                    trigger_verify = True
        return {
            "phase": self.phase,
            "event": event,
            "motion_ended": motion_ended,
            "trigger_verify": trigger_verify,
            "motion_start_frame": self.motion_start_frame,
            "motion_end_frame": self.motion_end_frame,
            "stable_after_motion_count": (
                self.stable_after_motion_count
            ),
            "verify_trigger_count": self.verify_trigger_count,
        }

    def verification_finished(self, complete=False):
        self.motion_count = 0
        self.stable_after_motion_count = 0
        self.phase = "COMPLETE" if complete else "WAIT_FOR_MOTION"
        return self.phase

    def verification_interrupted(self):
        self.phase = "MOVING"
        self.motion_count = self.start_confirm_frames
        self.stable_after_motion_count = 0
        return self.phase


class PieceCountConsensus:
    """Prefer a repeatedly observed high count without assuming four pieces."""

    __slots__ = (
        "minimum",
        "maximum",
        "window_size",
        "settle_detections",
        "minimum_confirmations",
        "window",
        "expected_count",
        "ready",
        "reason",
        "valid_samples",
        "observed_max",
    )

    def __init__(
        self,
        minimum,
        maximum,
        window_size,
        settle_detections,
        minimum_confirmations,
    ):
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.window_size = max(1, int(window_size))
        self.settle_detections = max(
            1, int(settle_detections)
        )
        self.minimum_confirmations = max(
            1, int(minimum_confirmations)
        )
        self.reset()

    def reset(self):
        self.window = []
        self.expected_count = None
        self.ready = False
        self.reason = "count_consensus_empty"
        self.valid_samples = 0
        self.observed_max = 0

    def update(self, count):
        count = max(0, min(self.maximum, int(count)))
        self.window.append(count)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        frequencies = {}
        valid = []
        for value in self.window:
            if self.minimum <= value <= self.maximum:
                valid.append(value)
                frequencies[value] = (
                    frequencies.get(value, 0) + 1
                )
        self.valid_samples = len(valid)
        self.observed_max = max(valid) if valid else 0
        confirmed = [
            value
            for value, frequency in frequencies.items()
            if frequency >= self.minimum_confirmations
        ]
        self.expected_count = (
            max(confirmed) if confirmed else None
        )
        self.ready = False
        if count < self.minimum:
            self.reason = "piece_count_below_min"
        elif self.expected_count is None:
            self.reason = "count_unconfirmed"
        elif self.observed_max > self.expected_count:
            self.reason = "higher_count_unconfirmed"
        elif self.valid_samples < self.settle_detections:
            self.reason = "count_consensus_collecting"
        elif count != self.expected_count:
            self.reason = "piece_count_mismatch"
        else:
            self.reason = "count_consensus_ready"
            self.ready = True
        return self.state()

    def state(self):
        return {
            "ready": self.ready,
            "reason": self.reason,
            "current_count": (
                self.window[-1] if self.window else 0
            ),
            "expected_count": self.expected_count,
            "observed_max": self.observed_max,
            "valid_samples": self.valid_samples,
            "settle_detections": self.settle_detections,
            "window": tuple(self.window),
        }


class RealtimePhaseState:
    """Minimal resettable phase controller for runtime/simulation tests."""

    __slots__ = ("phase",)

    def __init__(self):
        self.phase = "ACQUIRE"

    def start_placing(self):
        self.phase = "PLACING"

    def complete(self):
        self.phase = "COMPLETE"

    def reset(self):
        self.phase = "ACQUIRE"

    def vision_enabled(self):
        return phase_allows_vision(self.phase)
