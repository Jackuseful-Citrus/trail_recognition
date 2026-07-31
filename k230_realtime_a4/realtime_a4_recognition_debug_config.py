"""Recognition-only overrides for the K230 contour diagnostic standalone."""

from realtime_a4_free_rect_config import *


# Freeze the state machine in ACQUIRE after the normal tracker declares all
# four pieces stable. The live overlay consequently has no ``S:`` prefix and
# planning cannot hide recognition behaviour behind a frozen geometry sample.
DEBUG_RECOGNITION_HOLD_BEFORE_PLANNING = True
DEBUG_DRAW_RECTIFIED_CONTOURS = True
DEBUG_DRAW_SOURCE_RAW_CONTOURS = True
ENABLE_GRAY_SANITY_DIAGNOSTICS = True
ENABLE_STAGE_TIMING = True

# Recognition-only source-projective experiment.  The camera view remains
# 800x480, while all A4/divider/piece measurements share one aspect-preserving
# 640x384 grayscale image.  No perspective-resampled final pass is used.
PIECE_COORDINATE_MODE = "source_projective"
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
# Keep A4 boundary acquisition on the proven aspect-preserving 320x192
# detector raster.  On CanMV v1.6, native find_rects()/Blob.corners() at
# 640x384 produces many small rectangles and degenerate/touching dark-blob
# corners, while costing roughly 300 ms per call.  Only boundary candidates
# use this low-cost view; divider and piece recognition still share the real
# 640x384 SOURCE_PROJECTIVE work image and never call rotation_corr().
A4_DETECT_WIDTH = 320
A4_DETECT_HEIGHT = 192
A4_FULL_RES_REFINE_ENABLED = False

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
