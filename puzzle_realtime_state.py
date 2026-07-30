"""Pure state/scheduling helpers shared by K230 runtime and desktop tests."""


def phase_allows_vision(phase):
    """COMPLETE retains the last display but performs no more vision work."""
    return phase != "COMPLETE"


def a4_detection_interval(phase, acquire_interval, placing_interval):
    if phase == "PLACING":
        return max(1, int(placing_interval))
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
