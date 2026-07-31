"""Isolated optimized source-projective FreeRect no-UART profile."""

from realtime_free_rect_source_projective_no_uart_config import *


# Build marker used to distinguish the experiment artifact in board logs.  The
# optimized planner values themselves are inherited from the reviewed common
# FreeRect profile so module tests and this standalone exercise the same code.
FREE_RECT_OPTIMIZED_STAGED_BUILD = True
