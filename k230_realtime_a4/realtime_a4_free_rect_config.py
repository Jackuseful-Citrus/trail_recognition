"""Realtime profile for the experimental free-size rectangle backend."""

from realtime_a4_config import *


# Keep the established realtime camera, A4 calibration, contour fitting,
# tracking, and state-machine settings. Only the physical area envelope differs
# from the shared profile.
PLANNING_REQUIRED_PIECE_COUNT = 4
MIN_PIECE_COUNT = 4
MAX_PIECE_COUNT = 4
MAX_PIECE_AREA_MM2 = 12000.0

# Figure 2 returns before enumeration. Give only the generic free-plan search
# a larger K230 budget so it can examine enough complete assemblies for the
# preferred rectangle aspect ranking to take effect.
FREE_RECT_MAX_PLAN_TIME_MS = 20000
