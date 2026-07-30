"""Central configuration for the K230 A4 puzzle planner.

The file is intentionally importable by both CPython and CanMV MicroPython.
All physical geometry uses millimetres; ``RECTIFIED_PX_PER_MM`` is the only
conversion factor used by the vision adapters.
"""

# Camera/display geometry. Lowering the frame size usually increases FPS.
FRAME_WIDTH = 800
FRAME_HEIGHT = 480
# Native CanMV processing image. Both dimensions are multiples of 16; its
# aspect ratio is close to A4 so native perspective correction retains detail
# without needing the optional cv2 firmware module.
CANMV_WORK_WIDTH = 320
CANMV_WORK_HEIGHT = 448

# Manual camera calibration, ordered TL, TR, BR, BL in the corrected 800x480
# camera image. Replace these four points after running the calibration overlay.
# The defaults are a centred A4-shaped placeholder, not a measured calibration.
A4_CORNERS_PX = [
    (230.0, 4.0),
    (570.0, 4.0),
    (570.0, 476.0),
    (230.0, 476.0),
]

# Automatic A4 calibration.  The manual points above are retained only as an
# emergency fallback/debug reference; normal board operation detects the black
# A4 work surface at startup and locks it after consecutive valid frames.
AUTO_CALIBRATE_A4 = True
A4_DETECT_WIDTH = 320
A4_DETECT_HEIGHT = 192
A4_RECT_EDGE_THRESHOLD = 7000
A4_MIN_FRAME_AREA_RATIO = 0.12
A4_MAX_FRAME_AREA_RATIO = 0.94
A4_MIN_WIDTH_HEIGHT_RATIO = 0.40
A4_MAX_WIDTH_HEIGHT_RATIO = 2.40
A4_EXPECTED_WIDTH_HEIGHT_RATIO = 210.0 / 297.0
A4_MIN_SIDE_PX = 38.0
A4_MAX_CENTER_OFFSET_RATIO = 0.48
A4_MIN_IMAGE_EDGE_MARGIN_PX = 3
A4_TOP_SIDE = "auto"
A4_DARK_THRESHOLD = 135
A4_MAX_INSIDE_GRAY = 178.0
A4_DARK_BLOB_MIN_AREA_RATIO = 0.03
# Reject a divider-generated half-paper rectangle by probing just inside and
# outside every proposed edge in the low-resolution A4 detection image.  A
# real outer edge changes from the dark A4 surface to the brighter background;
# an internal divider has the same paper surface on both sides.
A4_EDGE_PROBE_OFFSET_PX = 8.0
A4_EDGE_PROBE_SAMPLES = 9
A4_INTERNAL_EDGE_SIMILAR_GRAY_DELTA = 24.0
A4_INTERNAL_EDGE_DARK_RATIO_MAX = 0.67
A4_INTERNAL_EDGE_MIN_SAMPLES = 5
A4_LOCK_REQUIRED_FRAMES = 3
A4_HOLD_MISSED_FRAMES = 15
A4_SLOW_SMOOTH_ALPHA = 0.38
A4_FAST_SMOOTH_ALPHA = 0.82
A4_FAST_MOTION_PX = 5.0
A4_RESET_MOTION_PX = 85.0
A4_STATUS_PRINT_EVERY_N_FRAMES = 15
A4_AUTO_SEARCH_PREVIEW = True
A4_LOCK_PREVIEW_HOLD_FRAMES = 20

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
# Increase for finer contours at the cost of RAM and processing time.
RECTIFIED_PX_PER_MM = 1.5
RECTIFIED_WIDTH_PX = int(A4_WIDTH_MM * RECTIFIED_PX_PER_MM + 0.5)
RECTIFIED_HEIGHT_PX = int(A4_HEIGHT_MM * RECTIFIED_PX_PER_MM + 0.5)

# Nominal horizontal divider. Automatic correction is restricted to this band.
DIVIDER_Y_MM = 148.5
DIVIDER_SEARCH_HALF_RANGE_MM = 8.0
# The physical separator is also an internal calibration feature. Probe it in
# the unwarped A4 candidate, estimate its live position/slope, and use the
# slope to remove differential left/right corner error before rectification.
ENABLE_DYNAMIC_DIVIDER = True
A4_REQUIRE_DIVIDER_FOR_LOCK = True
DIVIDER_LINE_SAMPLE_COUNT = 25
DIVIDER_LINE_MIN_GRAY = 70
DIVIDER_LINE_MIN_CONTRAST_GRAY = 35
DIVIDER_LINE_MIN_COVERAGE = 0.70
DIVIDER_LINE_MAX_RESIDUAL_PX = 2.5
DIVIDER_LINE_MAX_SLOPE_MM = 3.0
DIVIDER_TRACK_ALPHA = 0.35

# White segmentation. Lowering the threshold detects dimmer white pieces but
# also admits more glare. ``fixed`` is the safe fallback on every frame.
THRESHOLD_MODE = "fixed"  # "fixed" or "otsu"
WHITE_GRAY_THRESHOLD = 180
# Board runtime may override this with ``background_delta``.  The desktop
# regression stays fixed so historical photo results remain reproducible.
PIECE_SEGMENTATION_MODE = "fixed"  # "fixed" or "background_delta"
PIECE_BACKGROUND_SAMPLE_STRIDE = 4
PIECE_BACKGROUND_HISTOGRAM_BINS = 64
PIECE_BACKGROUND_DELTA_GRAY = 30
PIECE_BACKGROUND_RELAXED_DELTA_GRAY = 20
PIECE_BACKGROUND_NOISE_MARGIN_GRAY = 12
PIECE_BACKGROUND_MAX_DELTA_GRAY = 55
PIECE_BACKGROUND_MIN_SAMPLES = 96
MORPH_KERNEL_PX = 3
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

# Piece filters. Lowering the minimum accepts smaller noise regions.
MIN_PIECE_AREA_MM2 = 220.0
MAX_PIECE_AREA_MM2 = 6500.0
MIN_PIECE_COUNT = 2
MAX_PIECE_COUNT = 4
DETECTION_BORDER_MARGIN_MM = 3.0

# Polygon extraction. Larger epsilon and angle tolerance remove more vertices.
POLYGON_APPROX_EPSILON = 0.022  # fraction of contour perimeter
COLLINEAR_ANGLE_TOLERANCE_DEG = 9.0
MIN_VALID_EDGE_MM = 18.0
MIN_POLYGON_VERTICES = 3
MAX_POLYGON_VERTICES = 5
# Robust vertex cleanup after contour simplification.  Contest edges are at
# least 20 mm long, while the current hand-cut prototype contains a real edge
# of about 10 mm.  A 7 mm merge radius therefore removes small reflection
# chamfers/duplicate corners while preserving both sets of legitimate edges.
VERTEX_MERGE_DISTANCE_MM = 7.0
VERTEX_MERGE_MAX_EXTRAPOLATION_MM = 14.0
VERTEX_COLLINEAR_ANGLE_TOLERANCE_DEG = 12.0
VERTEX_COLLINEAR_MAX_OFFSET_MM = 3.0
VERTEX_CLEANUP_MAX_AREA_CHANGE_RATIO = 0.08
VERTEX_CLEANUP_MAX_PASSES = 8
# Keep a few temporary defect vertices until the robust cleanup has had a
# chance to merge them; only then enforce the physical 3..5 vertex limit.
VERTEX_CLEANUP_EXTRA_VERTICES = 4
# Ordered-contour fitting.  These values are in physical millimetres after A4
# rectification, so they remain meaningful when the working resolution changes.
CONTOUR_DP_TOLERANCE_MM = 2.2
LINE_FIT_MAX_ERROR_MM = 2.5
LINE_REFINE_MAX_SHIFT_MM = 3.0
LINE_FIT_MIN_POINTS = 6
LINE_REFINE_MAX_AREA_CHANGE_RATIO = 0.03
MIN_EDGE_LENGTH_MM = 18.0
BOUNDARY_TRACE_MAX_STEP_FACTOR = 8
BOUNDARY_TRACE_MIN_POINTS = 12
ENABLE_BOUNDARY_FLOOD_FALLBACK = True
FORCE_CONVEX_CONTOURS = False

# Edge attachment. Larger values admit less exact seams and grow search time.
EDGE_LENGTH_ABS_TOLERANCE_MM = 3.0
EDGE_LENGTH_REL_TOLERANCE = 0.10
EDGE_ENDPOINT_TOLERANCE_MM = 3.0
OVERLAP_TOLERANCE_MM2 = 8.0
MAX_SEARCH_NODES = 30000

# Competition fallback for hand-cut pieces. Corresponding seam endpoints may
# differ by at most 20 mm; lowering this produces tighter but fewer solutions.
ENABLE_TOLERANT_FALLBACK = True
CORRESPONDING_VERTEX_TOLERANCE_MM = 20.0
TOLERANT_RECTANGLE_SCORE_THRESHOLD = 0.32
TOLERANT_MAX_FILL_GAP_RATIO = 0.30
# Seam measurement error is intentionally independent from the final 20 mm
# placement allowance in the competition rules.
SEAM_LENGTH_ABS_TOLERANCE_MM = 4.0
SEAM_LENGTH_REL_TOLERANCE = 0.05
SEAM_ENDPOINT_ANGLE_TOLERANCE_DEG = 35.0
FINAL_VERTEX_TOLERANCE_MM = 20.0
FINAL_CENTER_TOLERANCE_MM = 15.0
FINAL_ANGLE_TOLERANCE_DEG = 12.0

# Rectangle validation. Score is dimensionless; lower is stricter.
RECT_MIN_WIDTH_MM = 90.0
RECT_MAX_WIDTH_MM = 120.0
RECT_MIN_HEIGHT_MM = 50.0
RECT_MAX_HEIGHT_MM = 90.0
RECTANGLE_SCORE_THRESHOLD = 0.095
RECT_FILL_GAP_TOLERANCE_MM2 = 180.0
OUTER_EDGE_TOLERANCE_MM = 3.0

# Unknown on-site rectangle planner.  It anchors one likely outside edge to a
# rectangle axis before matching seams.  This uses the contest guarantee that
# every piece contributes at least one target-boundary edge and avoids the
# orientation-free search explosion on K230.
OUTER_FIRST_AXIS_TOLERANCE_DEG = 12.0
OUTER_FIRST_TOLERANT_AXIS_TOLERANCE_DEG = 22.0
OUTER_FIRST_PARTIAL_BOUND_SLACK_MM = 12.0
OUTER_FIRST_CORNER_MAX_SEARCH_NODES = 3000
OUTER_FIRST_CORNER_BEAM_WIDTH = 128
OUTER_FIRST_CORNER_CANDIDATES_PER_PIECE = 32
OUTER_FIRST_MAX_SEARCH_NODES = 1200
OUTER_FIRST_BRANCH_LIMIT = 48
MAX_RECTANGLE_HYPOTHESES = 12
MAX_DFS_NODES = 1200
MAX_PLAN_TIME_MS = 3000
STATE_POSITION_QUANTIZATION_MM = 0.5
STATE_ANGLE_QUANTIZATION_DEG = 0.5
# Low-overhead planner heartbeat. The shared default stays off so desktop
# benchmarks and non-realtime entrypoints remain quiet; the realtime CanMV
# profile enables it. Time is sampled only at existing search batch/exitpoint
# boundaries, never for every polygon intersection.
ENABLE_PLAN_DEBUG = False
PLAN_DEBUG_INTERVAL_MS = 2000

# Geometry tolerances.  Linear and area units are deliberately separate.
GEOMETRY_EPSILON_MM = 0.05
OVERLAP_AREA_TOLERANCE_MM2 = 8.0

# Final placement. Increase the margin to keep farther from paper/divider edges.
TARGET_CENTER_MM = (105.0, 225.0)
TARGET_MARGIN_MM = 10.0
# Known prototype target. Set to None for unknown on-site rectangle dimensions.
TARGET_RECT_SIZE_MM = (100.0, 60.0)
PREFER_OUTER_FIRST_PLANNER = False
ENABLE_UNKNOWN_PLANNER_FALLBACK_AFTER_FIXED_FAILURE = False

# Fixed-rectangle packing tolerances for hand-cut 100x60 mm prototypes.
FIXED_RECT_BEAM_WIDTH = 1200
FIXED_RECT_MAX_OUTSIDE_MM2 = 250.0
FIXED_RECT_MAX_OVERLAP_MM2 = 30.0
FIXED_RECT_MAX_GAP_MM2 = 220.0
FIXED_RECT_SCORE_THRESHOLD = 0.06
FIXED_RECT_BOUNDARY_TOLERANCE_MM = 5.0
# The fast outer-first search may use tolerant seam matching, but a known-size
# target is accepted only when its final bounding dimensions stay close to the
# configured prototype. This is a final-result gate, not a seam tolerance.
KNOWN_TARGET_DIMENSION_TOLERANCE_MM = 5.0
# A4 rectification and foreground expansion can bias every detected length by
# the same small factor. For a known complete partition, total piece area gives
# a robust global scale correction. Larger corrections are refused because
# they more likely indicate a missing/extra blob than calibration drift.
KNOWN_TARGET_MAX_AREA_SCALE_DELTA = 0.04

# Multi-frame stability. Increasing the window reduces false stable plans but
# delays output. Tolerances are maximum deviations within the stable window.
STABLE_WINDOW_FRAMES = 10
REQUIRED_STABLE_FRAMES = 7
CENTER_STABLE_TOLERANCE_MM = 2.0
ANGLE_STABLE_TOLERANCE_DEG = 2.0
AREA_STABLE_TOLERANCE_RATIO = 0.12
TRACK_MAX_DISTANCE_MM = 35.0
TRACK_SHAPE_COST_LIMIT = 0.42
TRACK_MAX_MISSED_FRAMES = 4
# Contour fitting may temporarily split one physical corner or merge a shallow
# one.  Cross-frame identity may therefore tolerate a small vertex-count
# difference when centre and area still agree.
TRACK_MAX_VERTEX_COUNT_DELTA = 2
TRACK_VERTEX_MISMATCH_MAX_DISTANCE_MM = 15.0
TRACK_VERTEX_MISMATCH_MAX_AREA_RATIO = 0.35
TRACK_VERTEX_MISMATCH_SHAPE_PENALTY = 0.10

# A4 lock deadband. Distances are measured in source-frame pixels.
A4_LOCK_DEADBAND_PX = 3.0
A4_RELOCK_MOTION_PX = 10.0
A4_RELOCK_CONFIRM_FRAMES = 3

# Initial count consensus and incomplete-detection hold policy. Values are
# measured in piece-detection attempts, not camera frames.
BAD_COUNT_HOLD_DETECTIONS = 3
COUNT_REACQUIRE_FAILURES = 6

# Closed-loop placement. All distances and areas are in A4 millimetres/mm².
PLACEMENT_SHAPE_COST_LIMIT = 0.34
PLACEMENT_CONTOUR_SAMPLE_COUNT = 32
PLACEMENT_CONTOUR_RMS_MAX_MM = 5.0
PLACEMENT_CONTOUR_P90_MAX_MM = 8.0
PLACEMENT_CONTOUR_P95_HARD_MAX_MM = 18.0
PLACEMENT_CENTER_COARSE_MAX_MM = 20.0
PLACEMENT_AREA_RATIO_MIN = 0.70
PLACEMENT_AREA_RATIO_MAX = 1.30
PLACEMENT_POSE_BOUND_MM = 8.0
PLACEMENT_ORIENTATION_LONGEST_EDGE_RATIO_MIN = 1.15
PLACEMENT_ORIENTATION_SAMPLE_SPREAD_MAX_DEG = 8.0
PLACEMENT_TARGET_WHITE_COVERAGE = 0.72
PLACEMENT_REQUIRED_CHECKS = 1
PLACEMENT_VERIFY_REQUIRED_PASSES = 2
PLACEMENT_COVERAGE_SAMPLE_STRIDE = 2
PLACEMENT_DELTA_TARGET_COVERAGE_MIN = 0.65
PLACEMENT_DELTA_AREA_RATIO_MIN = 0.55
PLACEMENT_DELTA_AREA_RATIO_MAX = 1.40
PLACEMENT_DELTA_SPILL_MAX = 0.15
PLACEMENT_SOURCE_REMOVAL_MIN = 0.40
PLACEMENT_DELTA_ENVELOPE_MM = 8.0

# Lightweight motion detection on the rectified A4 grayscale image.
MOTION_SAMPLE_WIDTH = 80
MOTION_SAMPLE_HEIGHT = 112
MOTION_SAMPLE_STRIDE = 3
MOTION_DIVIDER_IGNORE_MM = 3.0
MOTION_PIXEL_DIFF_THRESHOLD = 18
MOTION_MEAN_ABS_DIFF_THRESHOLD = 5.0
MOTION_CHANGED_PIXEL_RATIO = 0.035
MOTION_START_CONFIRM_FRAMES = 2
MOTION_END_CONFIRM_FRAMES = 4
POST_MOTION_STABLE_FRAMES = 4
POST_MOTION_VERIFY_SAMPLES = 3

# Final whole-rectangle validation. Linear values are millimetres.
FINAL_RECT_FILL_MIN = 0.75
FINAL_AREA_RATIO_MIN = 0.75
FINAL_AREA_RATIO_MAX = 1.25
FINAL_RECT_DIM_TOLERANCE_MM = 20.0
FINAL_RECT_ENVELOPE_MM = 20.0
FINAL_RECT_SPILL_MAX = 0.10
FINAL_VERIFY_SAMPLE_COUNT = 3
FINAL_VERIFY_REQUIRED_PASSES = 2

# Runtime/UI controls.
DEBUG_SHOW_CAMERA = False
DISPLAY_EVERY_N_FRAMES = 2
PENDING_PRINT_EVERY_N_FRAMES = 15
LOOP_IDLE_MS = 5
AUTO_STOP_SECONDS = 0
MAX_FRAME_COUNT = 0
A4_DETECT_INTERVAL_ACQUIRE = 2
A4_DETECT_INTERVAL_PLACING = 8
# Disabled by default: normal placement checks are motion-triggered. When
# enabled this low-frequency watchdog is diagnostic/recovery only.
PLACING_VERIFICATION_INTERVAL_MS = 30000
ENABLE_PLACEMENT_WATCHDOG = False
UI_COUNTDOWN_REFRESH_INTERVAL_MS = 1000

# Lightweight performance instrumentation.  Detailed collection and reporting
# are disabled by default so the instrumentation itself does not reduce FPS.
ENABLE_STAGE_TIMING = False
TIMING_REPORT_INTERVAL_FRAMES = 30

# Optional playing-card seam appearance scoring.  Geometry remains authoritative
# and this feature is disabled for the existing pure-colour task.
ENABLE_IMAGE_STRIP_MATCHING = False
IMAGE_STRIP_WIDTH_MM = 3.0
IMAGE_STRIP_SAMPLE_SPACING_MM = 1.0
IMAGE_STRIP_WEIGHT = 0.15
BACKGROUND_SEGMENTATION_MODE = "white"
# Used only when BACKGROUND_SEGMENTATION_MODE="non_background_rgb".
BACKGROUND_COLOR_RGB = (30, 70, 100)
BACKGROUND_COLOR_DISTANCE_THRESHOLD = 55.0

# Offline defaults. The supplied portrait photograph needs its own four-point
# calibration because it is not the K230 camera frame.
OFFLINE_IMAGE = "sample_puzzle.jpg"
OFFLINE_A4_CORNERS_PX = [
    (127.0, 94.0),
    (1105.0, 82.0),
    (1198.0, 1582.0),
    (1.0, 1570.0),
]
OFFLINE_OUTPUT_IMAGE = "offline_puzzle_result.png"
