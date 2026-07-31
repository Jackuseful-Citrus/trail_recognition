"""Central configuration for the K230 A4 puzzle planner.

The file is intentionally importable by both CPython and CanMV MicroPython.
All physical geometry uses millimetres; ``RECTIFIED_PX_PER_MM`` is the only
conversion factor used by the vision adapters.
"""

# Camera/display geometry. Lowering the frame size usually increases FPS.
FRAME_WIDTH = 800
FRAME_HEIGHT = 480
# Generic entrypoints retain the RGB565 camera stream. Realtime black/white
# puzzle profiles can request VICAP's grayscale/luma output directly.
CAMERA_GRAYSCALE = False
# Native CanMV processing image. Both dimensions are multiples of 16; its
# aspect ratio is close to A4 so native perspective correction retains detail
# without needing the optional cv2 firmware module.
CANMV_WORK_WIDTH = 320
CANMV_WORK_HEIGHT = 448

# Piece-coordinate backends.  The established rectified raster remains the
# default; recognition diagnostics can opt into source-projective coordinates.
PIECE_COORDINATE_MODE = "rectified_raster"
SOURCE_PROJECTIVE_WORK_WIDTH = 640
SOURCE_PROJECTIVE_WORK_HEIGHT = 384
SOURCE_PROJECTIVE_FINAL_WIDTH = 640
SOURCE_PROJECTIVE_FINAL_HEIGHT = 384
SOURCE_PROJECTIVE_MASK_OUTSIDE_A4 = True
SOURCE_PROJECTIVE_MASK_DIVIDER = True
SOURCE_PROJECTIVE_DIVIDER_REQUIRED = True
SOURCE_PROJECTIVE_DIVIDER_HOLD_MISSES = 2
SOURCE_PROJECTIVE_A4_RELOCK_ENABLED = True
SOURCE_PROJECTIVE_GENERATION_GUARD = True
# Optional production guard. ``None`` keeps recognition diagnostics able to
# observe either physical half; planner profiles may require a normalized side.
SOURCE_PROJECTIVE_REQUIRED_SOURCE_HALF = None
# Static-scene optimizations are opt-in so recognition diagnostics retain the
# continuously measured reference implementation.
SOURCE_PROJECTIVE_FREEZE_A4_AFTER_LOCK = False
SOURCE_PROJECTIVE_FREEZE_DIVIDER = False
SOURCE_PROJECTIVE_DIVIDER_CONFIRM_DETECTIONS = 2
SOURCE_PROJECTIVE_FREEZE_SOURCE_HALF = False
SOURCE_PROJECTIVE_CACHE_BACKGROUND = False
SOURCE_PROJECTIVE_MASK_MODE = "blacken"
SOURCE_PROJECTIVE_MIN_BOUNDARY_INSIDE_RATIO = 0.95

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
# Detect the A4 boundary in the original lightweight 320x192 work image.
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
A4_LOCK_MAX_SPREAD_PX = 4.0
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
SOURCE_DIVIDER_SEARCH_STEP_MM = 0.75
SOURCE_HALF_SAMPLE_STRIDE = 3
SOURCE_HALF_MIN_CONFIDENCE = 0.18
SOURCE_HALF_MIN_BRIGHT_SAMPLES = 8

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
# A low background-relative threshold is useful for finding every component,
# but its grey halo/shadow is not a physical puzzle edge. A value <= 0 keeps
# the discovery threshold for contour tracing; the realtime black-surface
# profile raises this floor independently.
PIECE_CONTOUR_MIN_GRAY_THRESHOLD = 0
# Optional per-piece high-threshold contour refinement. Blob discovery remains
# at the shared background-relative threshold; only the identity-bound white
# core is adapted from a small robust sample around that Blob's centroid.
PIECE_ADAPTIVE_CONTOUR_THRESHOLD_ENABLED = False
PIECE_CONTOUR_CENTER_SAMPLE_RADIUS_PX = 2
PIECE_CONTOUR_CENTER_MIN_CONTRAST_GRAY = 40
PIECE_CONTOUR_ADAPTIVE_ALPHA = 0.42
PIECE_CONTOUR_ADAPTIVE_MIN_GRAY = 85
PIECE_CONTOUR_ADAPTIVE_MAX_GRAY = 140
# Optional in-place black safety band after perspective correction. Realtime
# profiles can enable it to prevent interpolated exterior paper/table pixels
# from joining a physical piece to the A4 image boundary.
PIECE_RECTIFIED_BORDER_BLACK_PX = 0
# After A4 perspective correction, independently locate the physical white
# separator before any piece Blob search.  The detector samples a narrow band
# around the A4 midpoint and only accepts a thin bright run that crosses most
# of the rectified page, so an ordinary fragment cannot become the divider.
PIECE_DIVIDER_DETECTION_ENABLED = True
PIECE_DIVIDER_SEARCH_HALF_RANGE_MM = 8.0
PIECE_DIVIDER_SAMPLE_COUNT = 31
# The real divider is grey after the 240x336 warp (the supplied board log
# shows it at the piece threshold around 59).  Geometry provides the strong
# rejection here, so keep the brightness gate deliberately lower than the
# white-piece contour threshold.
PIECE_DIVIDER_MIN_GRAY = 50
PIECE_DIVIDER_MIN_CONTRAST_GRAY = 20
PIECE_DIVIDER_MIN_COVERAGE = 0.72
PIECE_DIVIDER_MAX_THICKNESS_MM = 5.0
PIECE_DIVIDER_MAX_RESIDUAL_PX = 2.5
PIECE_DIVIDER_MAX_SLOPE_MM = 5.0
# The detected painted/taped strip and a small anti-aliasing halo are erased
# from the same shared grayscale array later consumed by find_blobs.
PIECE_DIVIDER_MASK_MARGIN_MM = 1.5
PIECE_DIVIDER_FALLBACK_GAP_MM = 2.0
MORPH_KERNEL_PX = 3
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

# Piece filters. Lowering the minimum accepts smaller noise regions.
MIN_PIECE_AREA_MM2 = 220.0
MAX_PIECE_AREA_MM2 = 6500.0
MIN_PIECE_COUNT = 4
MAX_PIECE_COUNT = 4
DETECTION_BORDER_MARGIN_MM = 3.0

# Polygon extraction. Larger epsilon and angle tolerance remove more vertices.
POLYGON_APPROX_EPSILON = 0.022  # fraction of contour perimeter
COLLINEAR_ANGLE_TOLERANCE_DEG = 9.0
MIN_VALID_EDGE_MM = 18.0
MIN_POLYGON_VERTICES = 3
MAX_POLYGON_VERTICES = 5
# Robust vertex cleanup after contour simplification.  Every physical edge is
# longer than 20 mm, so a fitted edge below 18 mm is a raster/reflection
# artefact.  Merging replaces its two endpoints by their midpoint, assigning
# one half of the removed edge to each neighbouring edge.
VERTEX_MERGE_DISTANCE_MM = 18.0
# This is expressed as deviation from a straight 180-degree vertex: 30 degrees
# therefore smooths vertices whose measured interior angle is at least 150.
VERTEX_COLLINEAR_ANGLE_TOLERANCE_DEG = 30.0
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

# Source-projective-only guard for false raster corners.  The generic
# rectified detector keeps its established fit unchanged.
FREE_RECT_MIN_OBSERVED_EDGE_MM = 17.5
SOURCE_PROJECTIVE_REFIT_DP_MULTIPLIERS = (1.20, 1.40)
SOURCE_PROJECTIVE_REFIT_MAX_AREA_CHANGE_RATIO = 0.10
SOURCE_PROJECTIVE_REFIT_MAX_RMS_MM = 3.5

# Edge-match and pose-refinement primitives used by the active free-rectangle
# planner. The implementation is pure Python for CanMV MicroPython.
SIMULATOR_MATCH_REL_TOLERANCE = 0.12
SIMULATOR_PARTIAL_MIN_RATIO = 0.22
SIMULATOR_PARTIAL_MAX_RATIO = 0.88
SIMULATOR_PARTIAL_MATCH_PENALTY = 0.15
SIMULATOR_MAX_CANDIDATES = 80
SIMULATOR_POSE_OPTIMIZATION_STEPS = 20

# Active free-rectangle planner.
FREE_RECT_MIN_PIECE_COUNT = 1
FREE_RECT_MAX_PIECE_COUNT = 4
FREE_RECT_MATCH_REL_TOLERANCE = 0.12
FREE_RECT_PARTIAL_MIN_RATIO = 0.22
FREE_RECT_PARTIAL_MAX_RATIO = 0.88
FREE_RECT_PARTIAL_MATCH_PENALTY = 0.15
FREE_RECT_MAX_CANDIDATES = 80
FREE_RECT_MIN_FULL_SHORTLIST = 24
FREE_RECT_MIN_PARTIAL_SHORTLIST = 40
FREE_RECT_MAX_COMPLETE_SETS = 6000
FREE_RECT_MAX_PLAN_TIME_MS = 8000
FREE_RECT_MAX_SPAN_MM = 170.0
FREE_RECT_TOP_K = 5
FREE_RECT_PROGRESS_INTERVAL_MS = 1000

# Optimized generic free-rectangle search.  Fixed Figure 2 returns before any
# of these stages run.
FREE_RECT_PAIR_MAX_FULL = 8
FREE_RECT_PAIR_MAX_PARTIAL = 4
FREE_RECT_GLOBAL_CANDIDATE_SAFETY_CAP = 96
FREE_RECT_EDGE_INTERVAL_OVERLAP_TOLERANCE = 0.03
FREE_RECT_TREE_ROUND_ROBIN_QUOTA = 16
FREE_RECT_TOTAL_PLAN_TIME_MS = 10000
FREE_RECT_PASS_STRICT_FULL_MS = 1500
FREE_RECT_PASS_STANDARD_T_MS = 2500
FREE_RECT_PASS_RELAXED_GEOMETRY_MS = 3000
FREE_RECT_PASS_MULTI_PARTIAL_MS = 3000
FREE_RECT_CHEAP_BEAM_PER_TOPOLOGY = 16
FREE_RECT_CHEAP_BEAM_ONE_PARTIAL = 24
FREE_RECT_EXACT_BEAM_SIZE = 48
FREE_RECT_STRONG_SOLUTION_GRACE_MS = 400
FREE_RECT_STRONG_IMPROVEMENT_RATIO = 0.03

# Fixed Figure 2 fast path.  These thresholds only decide whether the four
# observed contours belong to the known cut set; once matched, no assembly
# enumeration or final safety gate is run.
FREE_RECT_FIGURE2_DIRECT_ENABLED = True
FREE_RECT_FIGURE2_AREA_RATIO_TOLERANCE = 0.06
FREE_RECT_FIGURE2_RMS_TOLERANCE_MM = 6.0
FREE_RECT_FIGURE2_MAX_VERTEX_TOLERANCE_MM = 10.0

# Soft physical priors for the on-site rectangle. None of these is a prefix
# rejection gate.
FREE_RECT_LONG_SIDE_MIN_MM = 90.0
FREE_RECT_LONG_SIDE_MAX_MM = 120.0
FREE_RECT_SHORT_SIDE_MIN_MM = 50.0
FREE_RECT_SHORT_SIDE_MAX_MM = 90.0
# Prefer plausible competition rectangles without making the aspect range a
# hard validity gate. If no in-range proposal exists, publish the proposal
# closest to this interval.
FREE_RECT_PREFERRED_ASPECT_MIN = 1.33
FREE_RECT_PREFERRED_ASPECT_MAX = 1.67
FREE_RECT_OUTER_EDGE_TOLERANCE_MM = 5.0
FREE_RECT_PARTIAL_COUNT_PENALTY = 1.0
# Exposed-boundary estimator. Selected seam fractions are reused directly;
# this small tolerance scan only discovers closing contacts that are not part
# of the spanning-tree match set. It is O(E^2) over at most 20 piece edges and
# does not construct a polygon union.
FREE_RECT_PERIMETER_SEAM_DISTANCE_MM = 5.0
FREE_RECT_PERIMETER_SEAM_ANGLE_DEG = 12.0
FREE_RECT_PERIMETER_MIN_CONTACT_MM = 4.0
# Reject an assembly before the expensive pose/area stages when its exposed
# boundary is far longer than any preferred-aspect rectangle with the same
# source area. The deliberately wide gate tolerates contour and area noise;
# the continuous score below performs the fine ranking.
FREE_RECT_MAX_PERIMETER_EXCESS_RATIO = 0.18

# Complete-proposal cost weights.
FREE_RECT_WEIGHT_OVERLAP = 10.0
FREE_RECT_WEIGHT_FILL_GAP = 6.0
FREE_RECT_WEIGHT_HULL_GAP = 2.0
FREE_RECT_WEIGHT_AREA_PRIOR = 3.0
FREE_RECT_WEIGHT_DIMENSION_RANGE = 3.0
FREE_RECT_WEIGHT_OUTER_PIECE = 2.0
FREE_RECT_WEIGHT_SEAM = 1.0
FREE_RECT_WEIGHT_CLOSURE = 1.0
FREE_RECT_WEIGHT_PERIMETER = 12.0

# Equivalent target directions are compared only after geometric ranking.
FREE_RECT_MOTION_ROTATION_WEIGHT_MM_PER_DEG = 0.10
FREE_RECT_TARGET_MARGIN_MM = 10.0

# Hard publish gates for the generic path.  These are intentionally separate
# from soft ranking and never apply to the fixed Figure 2 direct result.
FREE_RECT_PUBLISH_LONG_MIN_MM = 85.0
FREE_RECT_PUBLISH_LONG_MAX_MM = 125.0
FREE_RECT_PUBLISH_SHORT_MIN_MM = 45.0
FREE_RECT_PUBLISH_SHORT_MAX_MM = 95.0
FREE_RECT_PUBLISH_AREA_ERROR_MAX = 0.15
FREE_RECT_PUBLISH_OVERLAP_RATIO_MAX = 0.05
FREE_RECT_PUBLISH_FILL_GAP_RATIO_MAX = 0.20
FREE_RECT_PUBLISH_OUTER_MISSING_MAX = 0
FREE_RECT_ALLOW_INVALID_DEBUG_PROPOSAL = False
FREE_RECT_OUTER_EDGE_DISTANCE_MM = 6.0
FREE_RECT_OUTER_EDGE_ANGLE_DEG = 12.0

# These additional thresholds remain diagnostics after the physical hard gates;
# perimeter is deliberately not a first-version publication reject.
FREE_RECT_WARN_OVERLAP_RATIO = 0.03
FREE_RECT_WARN_FILL_GAP_RATIO = 0.08
FREE_RECT_WARN_HULL_GAP_RATIO = 0.08
FREE_RECT_WARN_PERIMETER_ERROR_RATIO = 0.08
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

# Frozen-input integrity gate before generic planning. The physical task always
# starts from exactly four pieces; the known template bypasses the other gates.
PLANNING_REQUIRED_PIECE_COUNT = 4
PLANNING_INPUT_AREA_RATIO_MIN = 0.85
PLANNING_INPUT_AREA_RATIO_MAX = 1.15
PLANNING_INPUT_MAX_PAIR_OVERLAP_RATIO = 0.20
PLANNING_INPUT_MAX_BORDER_BLOBS = 0

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

# Lightweight motion detection on the rectified A4 grayscale image.
MOTION_SAMPLE_WIDTH = 80
MOTION_SAMPLE_HEIGHT = 112
MOTION_SAMPLE_STRIDE = 3
MOTION_DIVIDER_IGNORE_MM = 3.0
MOTION_PIXEL_DIFF_THRESHOLD = 18
MOTION_MEAN_ABS_DIFF_THRESHOLD = 5.0
MOTION_CHANGED_PIXEL_RATIO = 0.035

# Runtime completion uses only the source half: after its white area remains at
# or below 3% of the frozen initial piece area for ten motion-free frames, all
# four source pieces are considered removed and the run completes. The lower
# target-half and rectangle metrics below remain available for offline
# diagnostics, but never veto runtime completion.
FINAL_TRIGGER_UPPER_REMAINING_RATIO_MAX = 0.03
FINAL_TRIGGER_STABLE_FRAMES = 10

# Runtime/UI controls.
DEBUG_SHOW_CAMERA = False
# Recognition-only debug builds may hold the state machine in ACQUIRE after
# stable detection and compare the traced boundary with the fitted polygon on
# the exact rectified image. Production profiles leave both switches disabled.
DEBUG_RECOGNITION_HOLD_BEFORE_PLANNING = False
DEBUG_DRAW_RECTIFIED_CONTOURS = False
DEBUG_RECTIFIED_RAW_MAX_POINTS = 240
DISPLAY_EVERY_N_FRAMES = 2
PENDING_PRINT_EVERY_N_FRAMES = 15
LOOP_IDLE_MS = 5
AUTO_STOP_SECONDS = 0
MAX_FRAME_COUNT = 0
A4_DETECT_INTERVAL_ACQUIRE = 2

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
