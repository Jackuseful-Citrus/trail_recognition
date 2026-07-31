"""Realtime profile for the experimental free-size rectangle backend."""

from realtime_a4_config import *


# Keep the established realtime camera, A4 calibration, contour fitting,
# tracking, and state-machine settings. Only the physical area envelope differs
# from the shared profile.
PLANNING_REQUIRED_PIECE_COUNT = 4
MIN_PIECE_COUNT = 4
MAX_PIECE_COUNT = 4
MAX_PIECE_AREA_MM2 = 12000.0

# Figure 2 returns before enumeration.  The generic non-100x60 mm path accepts
# a wider set of measured seams and has twice the former search/tolerance
# envelope.  Minimum gates are halved; maximum gates and resource limits are
# doubled.  Soft ranking preferences remain unchanged.
FREE_RECT_MATCH_REL_TOLERANCE = 0.24
FREE_RECT_PARTIAL_MIN_RATIO = 0.11
FREE_RECT_PARTIAL_MAX_RATIO = 0.94
FREE_RECT_MAX_CANDIDATES = 160
FREE_RECT_MIN_FULL_SHORTLIST = 48
FREE_RECT_MIN_PARTIAL_SHORTLIST = 80
FREE_RECT_MAX_COMPLETE_SETS = 12000
FREE_RECT_MAX_PLAN_TIME_MS = 40000
FREE_RECT_MAX_SPAN_MM = 340.0

FREE_RECT_OUTER_EDGE_TOLERANCE_MM = 10.0
FREE_RECT_PERIMETER_SEAM_DISTANCE_MM = 10.0
FREE_RECT_PERIMETER_SEAM_ANGLE_DEG = 24.0
FREE_RECT_PERIMETER_MIN_CONTACT_MM = 2.0
FREE_RECT_MAX_PERIMETER_EXCESS_RATIO = 0.36
FREE_RECT_TARGET_MARGIN_MM = 5.0
