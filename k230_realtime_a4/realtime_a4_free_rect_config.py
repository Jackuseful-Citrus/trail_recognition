"""Realtime profile for the experimental free-size rectangle backend."""

from realtime_a4_config import *


# Keep the established realtime camera, A4 calibration, contour fitting,
# tracking, and state-machine settings. Only planner selection and the physical
# piece-count/area envelope differ from the fixed 100x60 mm profile.
PLANNER_BACKEND = "simulator_free_rect"
TARGET_RECT_SIZE_MM = None
PLANNING_REQUIRED_PIECE_COUNT = None
MIN_PIECE_COUNT = 1
MAX_PIECE_COUNT = 4
MAX_PIECE_AREA_MM2 = 12000.0

