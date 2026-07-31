"""Production overrides for source-projective FreeRect planning without UART."""

from realtime_a4_free_rect_config import *


# Use the common source-projective vision backend while preserving the proven
# FreeRect planner configuration inherited above.
PIECE_COORDINATE_MODE = "source_projective"
SOURCE_PROJECTIVE_WORK_WIDTH = 640
SOURCE_PROJECTIVE_WORK_HEIGHT = 384
SOURCE_PROJECTIVE_FINAL_WIDTH = 640
SOURCE_PROJECTIVE_FINAL_HEIGHT = 384
SOURCE_PROJECTIVE_MASK_OUTSIDE_A4 = True
SOURCE_PROJECTIVE_MASK_DIVIDER = True
SOURCE_PROJECTIVE_DIVIDER_REQUIRED = True
SOURCE_PROJECTIVE_DIVIDER_HOLD_MISSES = 2
SOURCE_PROJECTIVE_A4_RELOCK_ENABLED = False
SOURCE_PROJECTIVE_GENERATION_GUARD = True
SOURCE_PROJECTIVE_FREEZE_A4_AFTER_LOCK = True
SOURCE_PROJECTIVE_FREEZE_DIVIDER = True
SOURCE_PROJECTIVE_DIVIDER_CONFIRM_DETECTIONS = 2
SOURCE_PROJECTIVE_FREEZE_SOURCE_HALF = True
SOURCE_PROJECTIVE_CACHE_BACKGROUND = True
SOURCE_PROJECTIVE_MASK_MODE = "bbox_filter"
SOURCE_PROJECTIVE_MIN_BOUNDARY_INSIDE_RATIO = 0.95

# FreeRect currently places the target in the lower A4 half. Refuse planning
# unless source-projective recognition has normalized the pieces to the top.
SOURCE_PROJECTIVE_REQUIRED_SOURCE_HALF = "top"

# Keep the verified low-cost A4 boundary detector. Divider and piece detection
# still share the aspect-preserving 640x384 source work image.
A4_DETECT_WIDTH = 320
A4_DETECT_HEIGHT = 192
A4_FULL_RES_REFINE_ENABLED = False
A4_LOCK_REQUIRED_FRAMES = 2
A4_LOCK_MAX_SPREAD_PX = 2.0

# This is a production planning build, not the ACQUIRE-held diagnostic profile.
DEBUG_RECOGNITION_HOLD_BEFORE_PLANNING = False
DEBUG_DRAW_RECTIFIED_CONTOURS = False
DEBUG_DRAW_SOURCE_RAW_CONTOURS = False
ENABLE_GRAY_SANITY_DIAGNOSTICS = False
ENABLE_STAGE_TIMING = False

# The camera, A4 sheet, and source pieces are fixed in this deployment. Two
# consecutive full detections retain an exposure-settling guard without the
# generic eight-frame tracking window or an idle frame between samples.
PIECE_DETECT_EVERY_N_FRAMES = 1
PIECE_COUNT_SETTLE_DETECTIONS = 2
STABLE_WINDOW_FRAMES = 2
REQUIRED_STABLE_FRAMES = 2

DISPLAY_EVERY_N_FRAMES = 2
PIECE_DIAGNOSTIC_PRINT_EVERY_N_DETECTIONS = 5
GRAY_SANITY_EVERY_N_DETECTIONS = 5

# The builder also writes a final global override immediately before the runtime
# section, so optional UART imports and physical execution remain impossible.
UART_COMMUNICATION_ENABLED = False
