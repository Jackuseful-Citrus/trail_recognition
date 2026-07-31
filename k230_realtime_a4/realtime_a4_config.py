"""Configuration overrides for the realtime simulator-backed A4 runtime."""

from puzzle_config import *


# Automatically find and freeze the initial A4 boundary. The manual corners
# remain available as a fallback by setting this switch to False. They are in
# physical A4 order: TL, TR, BR, BL; in the current landscape camera view these
# labels appear as image BL, image TL, image TR, image BR.
AUTO_CALIBRATE_A4 = True
# The operator view and all recognition stages are grayscale. Ask VICAP for a
# grayscale/luma frame directly instead of converting every RGB565 snapshot on
# the CPU before A4 and piece processing.
CAMERA_GRAYSCALE = True
A4_CORNERS_PX = [
    (133.0, 441.0),
    (140.0, 78.0),
    (619.0, 83.0),
    (614.0, 449.0),
]
# The separator is the physical midpoint of the fixed A4 coordinate system.
# Do not let the white line participate in calibration or alter the split.
ENABLE_DYNAMIC_DIVIDER = False
A4_REQUIRE_DIVIDER_FOR_LOCK = False

# Show the fixed green calibration outline briefly at startup, then switch to
# the result canvas. Set DEBUG_SHOW_CAMERA only when the outline should remain
# visible continuously.
DEBUG_SHOW_CAMERA = False
A4_AUTO_SEARCH_PREVIEW = True
A4_LOCK_PREVIEW_HOLD_FRAMES = 20

# Operator display: keep the physical camera view as a grayscale background
# and project all A4/piece/target contours back onto that live image. During
# large motion, retain only the A4 reference and the short state-machine line.
LIVE_GRAYSCALE_OPERATOR_VIEW = True
OPERATOR_HIDE_OVERLAYS_DURING_MOTION = True

# CanMV-K230 V3.0 onboard programmable status light: one WS2812B-mini RGB
# pixel (D4) driven by GPIO35. Keep it off during acquisition/planning, then
# show a low-brightness green continuously after completion.
COMPLETION_LED_ENABLED = True
COMPLETION_LED_PIN = 35
COMPLETION_LED_COLOR = (0, 255, 0)
COMPLETION_LED_DURATION_MS = 3000

# Restore the original lightweight A4 path. The full-resolution refine remains
# available in the runtime but is disabled for this deployment.
A4_DETECT_WIDTH = 320
A4_DETECT_HEIGHT = 192
A4_FULL_RES_REFINE_ENABLED = False
A4_REFINE_WIDTH = 800
A4_REFINE_HEIGHT = 480
A4_RECT_EDGE_THRESHOLD = 7000

# Candidate geometry in the original 320x192 boundary image.
A4_MIN_FRAME_AREA_RATIO = 0.12
A4_MAX_FRAME_AREA_RATIO = 0.94
A4_MIN_WIDTH_HEIGHT_RATIO = 0.40
A4_MAX_WIDTH_HEIGHT_RATIO = 2.40
A4_EXPECTED_WIDTH_HEIGHT_RATIO = A4_WIDTH_MM / A4_HEIGHT_MM
A4_MIN_SIDE_PX = 38.0
A4_MAX_CENTER_OFFSET_RATIO = 0.48
# Reject false rectangles clipped by the camera image. At 320x192, 3 pixels
# correspond to about 8 pixels in the 800x480 source image.
A4_MIN_IMAGE_EDGE_MARGIN_PX = 3

# Infer which physical half contains the source fragments.  This removes the
# last installation-specific orientation setting: portrait/landscape placement
# and a 180-degree camera rotation are handled from the bright-fragment
# distribution.  Explicit "top"/"bottom"/"left"/"right" values remain
# available only as emergency overrides.
A4_TOP_SIDE = "auto"

# The work surface is black. This also enables a largest-dark-component
# fallback when the native edge rectangle detector misses one frame.
A4_DARK_THRESHOLD = 135
A4_MAX_INSIDE_GRAY = 178.0
A4_DARK_BLOB_MIN_AREA_RATIO = 0.03
# The internal divider can outline a half-sheet whose aspect ratio is again
# A-series-valid. Probe across each edge and reject it when dark paper
# continues outside the proposed boundary.
A4_EDGE_PROBE_OFFSET_PX = 8.0
A4_EDGE_PROBE_SAMPLES = 9
A4_INTERNAL_EDGE_SIMILAR_GRAY_DELTA = 24.0
A4_INTERNAL_EDGE_DARK_RATIO_MAX = 0.67
A4_INTERNAL_EDGE_MIN_SAMPLES = 5

# Dynamic corner tracker. Large motion follows quickly; sub-pixel edge jitter
# is smoothed more strongly. A few missed frames may reuse the last boundary.
A4_LOCK_REQUIRED_FRAMES = 3
# At one A4 check every two camera frames, 15 misses retain the calibrated
# corners for roughly one second at 30 FPS. This bridges short native
# find_rects dropouts without holding through a long obstruction.
A4_HOLD_MISSED_FRAMES = 15
A4_SLOW_SMOOTH_ALPHA = 0.38
A4_FAST_SMOOTH_ALPHA = 0.82
A4_FAST_MOTION_PX = 5.0
A4_RESET_MOTION_PX = 85.0

# Print boundary status without flooding the terminal.
A4_STATUS_PRINT_EVERY_N_FRAMES = 15

# Expensive stages run at independent rates after A4 lock. A4 tracking every
# second camera frame keeps up with light shake; polygon extraction every third
# frame is sufficient for stability while intermediate frames remain live.
A4_TRACK_EVERY_N_FRAMES = 2
PIECE_DETECT_EVERY_N_FRAMES = 3
PIECE_COUNT_WINDOW_DETECTIONS = 12
PIECE_COUNT_SETTLE_DETECTIONS = 4
PIECE_COUNT_MIN_CONFIRMATIONS = 2
PIECE_LOW_GRAY_THRESHOLD = 165
PIECE_THRESHOLD_PROBE_EVERY_N_DETECTIONS = 4
PIECE_DIAGNOSTIC_PRINT_EVERY_N_DETECTIONS = 5
# Full-array gray sanity scanning is diagnostic-only and expensive in
# MicroPython. Keep it off in production; the recognition-debug profile turns
# it back on explicitly.
ENABLE_GRAY_SANITY_DIAGNOSTICS = False
GRAY_SANITY_EVERY_N_DETECTIONS = 5
# After A4 rectification, calibrate the dominant green/dark paper gray level
# from the mostly empty lower half. The threshold follows global illumination
# instead of assuming that every white piece remains above gray 180.
PIECE_SEGMENTATION_MODE = "background_delta"
PIECE_BACKGROUND_SAMPLE_STRIDE = 4
PIECE_BACKGROUND_HISTOGRAM_BINS = 64
PIECE_BACKGROUND_DELTA_GRAY = 30
PIECE_BACKGROUND_RELAXED_DELTA_GRAY = 20
PIECE_BACKGROUND_NOISE_MARGIN_GRAY = 12
PIECE_BACKGROUND_MAX_DELTA_GRAY = 55
PIECE_BACKGROUND_MIN_SAMPLES = 96
# Discover components at background+30 (about 51 in the current lighting),
# then trace the white-paper boundary above the grey cast-shadow band.
PIECE_CONTOUR_MIN_GRAY_THRESHOLD = 100
# Adapt the identity-bound high-threshold contour independently for every
# discovered piece. This does not alter find_blobs or create more candidates.
PIECE_ADAPTIVE_CONTOUR_THRESHOLD_ENABLED = True
PIECE_CONTOUR_CENTER_SAMPLE_RADIUS_PX = 2
PIECE_CONTOUR_CENTER_MIN_CONTRAST_GRAY = 40
PIECE_CONTOUR_ADAPTIVE_ALPHA = 0.42
PIECE_CONTOUR_ADAPTIVE_MIN_GRAY = 85
PIECE_CONTOUR_ADAPTIVE_MAX_GRAY = 140
# Erase the perspective-interpolation fringe before native Blob discovery.
# The 5-pixel band is also excluded from background calibration/search ROIs.
PIECE_RECTIFIED_BORDER_BLACK_PX = 5
# The A4-lock detector intentionally does not depend on an internal line in
# this runtime profile.  Once the A4 is rectified, however, detect and erase a
# thin separator near its midpoint before native piece Blob discovery.
PIECE_DIVIDER_DETECTION_ENABLED = True
PIECE_DIVIDER_MIN_GRAY = 50
PIECE_DIVIDER_MIN_CONTRAST_GRAY = 20

# This competition assembly always starts from four physical pieces. Do not
# let a repeated incomplete three-piece observation become planner input.
PLANNING_REQUIRED_PIECE_COUNT = 4

# Repeated acquisition/tracking stays at 320x448 so temporal stability does
# not repeatedly pay for the final raster. Once the tracker is stable, one
# 480x672 pass (about 2.27 px/mm) replaces the coarse observations before
# planning. Planning still receives only simplified 3..5-vertex millimetre
# polygons, so its search cost is independent of these raster dimensions.
REALTIME_PIECE_WORK_WIDTH = 320
REALTIME_PIECE_WORK_HEIGHT = 448
REALTIME_PIECE_FINAL_WIDTH = 480
REALTIME_PIECE_FINAL_HEIGHT = 672

# Show the exact perspective-corrected grayscale image most recently consumed
# by piece segmentation. After stability this is the final 480x672 pass;
# rendering never runs a second thumbnail-only rotation_corr path.
SHOW_GRAY_WORK_THUMBNAIL = True
GRAY_THUMBNAIL_MAX_WIDTH = 128
GRAY_THUMBNAIL_MAX_HEIGHT = 180
GRAY_THUMBNAIL_MARGIN_PX = 8

# ``to_ide=True`` remains enabled as a low-cost display-mirror fallback. The
# explicit channel fixes IDE/VS Code Preview environments where that mirror is
# absent. CPU JPEG compression is throttled to protect recognition FPS.
IDE_STREAM_ENABLED = True
IDE_STREAM_EVERY_N_OUTPUTS = 2
IDE_STREAM_QUALITY = 50

# Dynamic A4 coordinates remove most camera motion. These slightly relaxed
# residual tolerances allow planning while a hand-held camera still jitters.
STABLE_WINDOW_FRAMES = 8
REQUIRED_STABLE_FRAMES = 4
CENTER_STABLE_TOLERANCE_MM = 4.0
ANGLE_STABLE_TOLERANCE_DEG = 8.0

# Planning can take much longer on MicroPython than on desktop CPython. Emit
# one compact progress heartbeat every two seconds so an IDE interrupt still
# leaves evidence of the active stage and search growth.
ENABLE_PLAN_DEBUG = True
PLAN_DEBUG_INTERVAL_MS = 2000
# A hand-cut corner can still move by several pixels between perspective
# corrections. Remove shallow raster corners consistently in both the
# 320x448 tracking pass and the final 480x672 pass. Every real edge is longer
# than 20 mm: merge sub-18 mm artefacts, and smooth interior angles >=150 deg.
CONTOUR_DP_TOLERANCE_MM = 3.0
VERTEX_MERGE_DISTANCE_MM = 18.0
VERTEX_COLLINEAR_ANGLE_TOLERANCE_DEG = 30.0
VERTEX_COLLINEAR_MAX_OFFSET_MM = 4.0
VERTEX_CLEANUP_MAX_AREA_CHANGE_RATIO = 0.15
