"""Recognition-only overrides for the K230 contour diagnostic standalone."""

from realtime_a4_free_rect_config import *


# Freeze the state machine in ACQUIRE after the normal tracker declares all
# four pieces stable. The live overlay consequently has no ``S:`` prefix and
# planning cannot hide recognition behaviour behind a frozen geometry sample.
DEBUG_RECOGNITION_HOLD_BEFORE_PLANNING = True
DEBUG_DRAW_RECTIFIED_CONTOURS = True
DEBUG_DRAW_SOURCE_RAW_CONTOURS = True
ENABLE_GRAY_SANITY_DIAGNOSTICS = True

# The diagnostic must show the same raster that will be frozen for planning,
# not the cheaper 320x448 temporal-tracking preview.
REALTIME_PIECE_WORK_WIDTH = REALTIME_PIECE_FINAL_WIDTH
REALTIME_PIECE_WORK_HEIGHT = REALTIME_PIECE_FINAL_HEIGHT

# Refresh and report every detection so a 4/5-vertex transition is visible in
# both IDE Preview and the serial log. These are diagnostic costs only; the
# production standalone keeps its existing cadence.
DISPLAY_EVERY_N_FRAMES = 1
PIECE_DIAGNOSTIC_PRINT_EVERY_N_DETECTIONS = 1
GRAY_SANITY_EVERY_N_DETECTIONS = 1
DEBUG_RECTIFIED_RAW_MAX_POINTS = 240
