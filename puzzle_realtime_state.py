"""Pure state/scheduling helpers shared by K230 runtime and desktop tests."""


def divider_overlay_endpoints(divider_state, a4_width_mm):
    """Return confirmed left/right divider endpoints, or no display line.

    ``slope_mm`` is the total vertical change across the rectified A4 width,
    matching the piece-stage divider detector's diagnostics.
    """
    if (
        not divider_state
        or not divider_state.get("detected", False)
    ):
        return None
    center_y_mm = float(divider_state["divider_y_mm"])
    slope_mm = float(divider_state.get("slope_mm", 0.0))
    half_slope = 0.5 * slope_mm
    return (
        (0.0, center_y_mm - half_slope),
        (float(a4_width_mm), center_y_mm + half_slope),
    )


def phase_allows_vision(phase):
    """COMPLETE retains the last display but performs no more vision work."""
    return phase != "COMPLETE"


def a4_detection_interval(
    phase,
    acquire_interval,
    locked=False,
):
    # The board and camera are fixed.  Once the initial A4 acquisition has
    # locked, keep that exact calibration instead of feeding detector jitter
    # into a different perspective transform every few frames.
    if locked:
        return None
    if phase in ("WAIT_FINAL_CHECK", "COMPLETE"):
        return None
    return max(1, int(acquire_interval))


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


def top_right_thumbnail_rect(
    canvas_width,
    canvas_height,
    source_width,
    source_height,
    max_width,
    max_height,
    margin,
):
    """Fit an aspect-preserving thumbnail against the top-right corner."""
    x, _y, width, height, scale = bottom_right_thumbnail_rect(
        canvas_width,
        canvas_height,
        source_width,
        source_height,
        max_width,
        max_height,
        margin,
    )
    return (
        x,
        max(0, min(int(margin), int(canvas_height) - height)),
        width,
        height,
        scale,
    )


def periodic_output_due(output_index, every_n_outputs):
    """Return whether a throttled side-channel should publish this output."""
    interval = max(1, int(every_n_outputs))
    return max(0, int(output_index)) % interval == 0


def operator_overlay_visibility(phase, motion_active=False):
    """Return the live operator-view layers allowed in the current phase."""
    if phase in ("WAIT_FINAL_CHECK", "COMPLETE"):
        return {
            "a4": True,
            "status": True,
            "pieces": True,
            "targets": True,
        }
    moving = bool(motion_active)
    return {
        "a4": True,
        "status": True,
        "pieces": not moving,
        "targets": not moving,
    }


def operator_status_line(
    phase,
    piece_count,
    stable=False,
    plan_available=False,
    plan_valid=False,
    error=None,
):
    """Build one short line for the narrow strip below the camera-view A4."""
    if error:
        return "{} | ERROR".format(phase)
    if phase == "WAIT_FINAL_CHECK":
        return "WAIT FINAL CHECK"
    if phase == "COMPLETE":
        return "COMPLETE | PASS"
    if phase == "PLANNING":
        return "PLANNING | P:{}".format(piece_count)
    if plan_available and not plan_valid:
        return "ACQUIRE | PLAN BLOCKED | P:{}".format(
            piece_count
        )
    return "ACQUIRE | P:{} | {}".format(
        piece_count,
        "STABLE" if stable else "TRACKING",
    )


def planning_input_integrity(
    pieces,
    target_rect_size_mm,
    overlap_area,
    required_piece_count=None,
    area_ratio_min=0.85,
    area_ratio_max=1.15,
    max_pair_overlap_ratio=0.20,
    rejected_border_blobs=0,
    max_rejected_border_blobs=0,
):
    """Fail closed on incomplete or duplicate frozen planner input."""
    pieces = list(pieces)
    total_area = sum(float(piece.area_mm2) for piece in pieces)
    target_area = None
    area_ratio = None
    if target_rect_size_mm is not None:
        target_area = (
            float(target_rect_size_mm[0])
            * float(target_rect_size_mm[1])
        )
        if target_area > 0.0:
            area_ratio = total_area / target_area

    max_overlap_ratio = 0.0
    max_overlap_pair = None
    for left in range(len(pieces)):
        for right in range(left + 1, len(pieces)):
            overlap = float(
                overlap_area(
                    pieces[left].polygon_mm,
                    pieces[right].polygon_mm,
                )
            )
            smaller = max(
                1e-9,
                min(
                    float(pieces[left].area_mm2),
                    float(pieces[right].area_mm2),
                ),
            )
            ratio = overlap / smaller
            if ratio > max_overlap_ratio:
                max_overlap_ratio = ratio
                max_overlap_pair = (
                    pieces[left].piece_id
                    or "P{}".format(left + 1),
                    pieces[right].piece_id
                    or "P{}".format(right + 1),
                )

    failures = []
    if (
        required_piece_count is not None
        and len(pieces) != int(required_piece_count)
    ):
        failures.append("piece_count")
    if area_ratio is not None and (
        area_ratio < float(area_ratio_min)
        or area_ratio > float(area_ratio_max)
    ):
        failures.append("total_area")
    if max_overlap_ratio > float(max_pair_overlap_ratio):
        failures.append("pair_overlap")
    if int(rejected_border_blobs) > int(
        max_rejected_border_blobs
    ):
        failures.append("border_blob")
    return {
        "valid": not failures,
        "reason": failures[0] if failures else "ok",
        "failures": tuple(failures),
        "piece_count": len(pieces),
        "required_piece_count": required_piece_count,
        "total_area_mm2": total_area,
        "target_area_mm2": target_area,
        "area_ratio": area_ratio,
        "max_pair_overlap_ratio": max_overlap_ratio,
        "max_pair_overlap_pair": max_overlap_pair,
        "rejected_border_blobs": int(rejected_border_blobs),
    }


def planning_input_integrity_unless_fixed_template(
    fixed_template_match,
    pieces,
    target_rect_size_mm,
    overlap_area,
    **kwargs
):
    """Bypass generic frozen-input gates for the known four-piece set."""
    if fixed_template_match:
        return None
    return planning_input_integrity(
        pieces,
        target_rect_size_mm,
        overlap_area,
        **kwargs
    )


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
        if (
            self.previous is None
            or len(self.previous) != len(current)
        ):
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


class FinalCheckState:
    """Complete after the source half stays clear and still."""

    __slots__ = (
        "stable_frames_required",
        "upper_ratio_max",
        "phase",
        "stable_frames",
        "upper_remaining_ratio",
        "lower_area_ratio",
    )

    def __init__(
        self,
        stable_frames_required,
        upper_ratio_max,
    ):
        self.stable_frames_required = max(
            1, int(stable_frames_required)
        )
        self.upper_ratio_max = float(upper_ratio_max)
        self.phase = "WAIT_FINAL_CHECK"
        self.stable_frames = 0
        self.upper_remaining_ratio = None
        self.lower_area_ratio = None

    def update(
        self,
        motion,
        upper_remaining_ratio,
        lower_area_ratio,
    ):
        """Consume one source-clear sample without changing frozen references."""
        self.upper_remaining_ratio = float(
            upper_remaining_ratio
        )
        self.lower_area_ratio = float(lower_area_ratio)
        source_clear = (
            self.upper_remaining_ratio <= self.upper_ratio_max
        )
        trigger_complete = False
        if self.phase == "WAIT_FINAL_CHECK":
            if motion or not source_clear:
                self.stable_frames = 0
            else:
                self.stable_frames += 1
            if (
                self.stable_frames
                >= self.stable_frames_required
            ):
                self.phase = "COMPLETE"
                trigger_complete = True
        return {
            "phase": self.phase,
            "trigger_complete": trigger_complete,
            "source_clear": source_clear,
            "stable_frames": self.stable_frames,
            "stable_frames_required": (
                self.stable_frames_required
            ),
            "upper_remaining_ratio": (
                self.upper_remaining_ratio
            ),
            "lower_area_ratio": self.lower_area_ratio,
        }

    def state(self):
        return {
            "phase": self.phase,
            "source_clear": (
                self.upper_remaining_ratio is not None
                and self.upper_remaining_ratio
                <= self.upper_ratio_max
            ),
            "stable_frames": self.stable_frames,
            "stable_frames_required": (
                self.stable_frames_required
            ),
            "upper_remaining_ratio": (
                self.upper_remaining_ratio
            ),
            "lower_area_ratio": self.lower_area_ratio,
        }


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


def tracking_expected_piece_count(
    required_piece_count, consensus_state
):
    """Allow a fixed-count tracker to run while count consensus settles."""
    if required_piece_count is not None:
        return max(1, int(required_piece_count))
    if not consensus_state.get("ready", False):
        return None
    expected_count = consensus_state.get("expected_count")
    return (
        None
        if expected_count is None
        else max(1, int(expected_count))
    )


def recognition_gate_ready(consensus_state, tracker_stable):
    """Require both gates without forcing them to run sequentially."""
    return bool(
        consensus_state.get("ready", False)
        and tracker_stable
    )
