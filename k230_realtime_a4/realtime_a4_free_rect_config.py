"""Realtime profile for the experimental free-size rectangle backend."""

from realtime_a4_config import *


# Keep the established realtime camera, A4 calibration, contour fitting,
# tracking, and state-machine settings.  This profile only specializes the
# FreeRect piece envelope, staged-search budget, and physical publication gates.
PLANNING_REQUIRED_PIECE_COUNT = 4
MIN_PIECE_COUNT = 4
MAX_PIECE_COUNT = 4
MAX_PIECE_AREA_MM2 = 12000.0

# Figure 2 returns before enumeration.  The generic path uses four bounded
# passes, a per-piece-pair shortlist, and a cheap-to-exact beam.  Do not restore
# the former single 40-second search with doubled geometry tolerances: that
# profile could publish a rectangle whose area exceeded the source by 44%.
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

# Generic-path publication is fail-closed.  Perimeter remains a soft score,
# while dimensions, area, overlap, fill, outer-edge evidence, and target fit
# are hard gates.
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
FREE_RECT_TARGET_MARGIN_MM = 10.0
