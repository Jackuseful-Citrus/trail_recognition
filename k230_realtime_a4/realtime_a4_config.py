"""Configuration overrides for shake-tolerant automatic A4 tracking."""

from puzzle_config import *


# Normal contest mode: show the camera automatically while acquiring A4, keep
# the green locked outline briefly, then switch to the result canvas without a
# manual configuration change.  Set DEBUG_SHOW_CAMERA only when a persistent
# raw-camera diagnostic view is explicitly needed.
DEBUG_SHOW_CAMERA = False
A4_AUTO_SEARCH_PREVIEW = True
A4_LOCK_PREVIEW_HOLD_FRAMES = 20

# A4 boundary detector uses a small aspect-preserving grayscale frame.
A4_DETECT_WIDTH = 320
A4_DETECT_HEIGHT = 192
A4_RECT_EDGE_THRESHOLD = 7000

# Candidate geometry in the 320x192 boundary image.
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
PIECE_COUNT_SETTLE_DETECTIONS = 8
PIECE_COUNT_MIN_CONFIRMATIONS = 2
PIECE_LOW_GRAY_THRESHOLD = 165
PIECE_THRESHOLD_PROBE_EVERY_N_DETECTIONS = 4
PIECE_DIAGNOSTIC_PRINT_EVERY_N_DETECTIONS = 5
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

# Lower-resolution real-time piece image. Roughly 1.14 px/mm is still adequate
# for the hand-cut 20 mm+ edges and reduces pixel traversal by about 44%.
REALTIME_PIECE_WORK_WIDTH = 240
REALTIME_PIECE_WORK_HEIGHT = 336

# Show the perspective-corrected grayscale image actually consumed by piece
# segmentation. It is fitted into the bottom-right corner of the 800x480
# contour canvas without allocating a second full-size image.
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

# At 240x336, a hand-cut corner can move by several pixels between perspective
# corrections. Remove shallow raster corners more consistently before temporal
# tracking; the contest still guarantees every real edge is at least 20 mm.
CONTOUR_DP_TOLERANCE_MM = 3.0
VERTEX_COLLINEAR_ANGLE_TOLERANCE_DEG = 18.0
VERTEX_COLLINEAR_MAX_OFFSET_MM = 4.0
VERTEX_CLEANUP_MAX_AREA_CHANGE_RATIO = 0.15
