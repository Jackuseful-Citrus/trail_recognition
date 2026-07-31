"""Fixed settings for the minimal K230 UART2 motion link.

This file is intentionally separate from the vision configuration.  Upload it
beside the generated standalone script and edit the fixed calibration here
after measuring the real mechanism.
"""


# Keep execution disabled until the affine calibration below has been filled
# with values measured on the real machine.  The vision program continues its
# original manual-placement flow while this is False.
PROTOCAL_EXECUTION_ENABLED = False

# CanMV-K230 V3.0 UART2 pins.  They are connected to the DM-MC-Board02 UART7.
PROTOCAL_UART2_TX_PIN = 5
PROTOCAL_UART2_RX_PIN = 6
PROTOCAL_UART_BAUDRATE = 115200

# The first transmitted action uses sequence 1.  A new action increments the
# 8-bit sequence; this minimal version does not retransmit failed actions.
PROTOCAL_INITIAL_SEQUENCE = 0
PROTOCAL_STATUS_TIMEOUT_MS = 20000
PROTOCAL_POLL_DELAY_MS = 2
PROTOCAL_POST_PLAN_SETTLE_MS = 500

# Fixed A4-mm -> mechanism-mm affine transform:
#
#   machine_x = XX * a4_x + XY * a4_y + X_OFFSET
#   machine_y = YX * a4_x + YY * a4_y + Y_OFFSET
#
# The identity values are placeholders, not a completed machine calibration.
PROTOCAL_A4_TO_MACHINE_XX = 1.0
PROTOCAL_A4_TO_MACHINE_XY = 0.0
PROTOCAL_A4_TO_MACHINE_X_OFFSET_MM = 0.0
PROTOCAL_A4_TO_MACHINE_YX = 0.0
PROTOCAL_A4_TO_MACHINE_YY = 1.0
PROTOCAL_A4_TO_MACHINE_Y_OFFSET_MM = 0.0

# The planner and protocol both use clockwise-positive angles in the intended
# installation.  Set this to -1.0 after the low-speed direction test if the
# physical rotary actuator moves in the opposite direction.
PROTOCAL_ROTATION_SIGN = 1.0
# Avoid sending a zero-length relative move.  The DM executor can otherwise
# wait for a rotary move_active transition that never starts.
PROTOCAL_ROTATION_SKIP_EPSILON_DEG = 0.05

# Limits implemented by the current DM board single-step executor contract.
PROTOCAL_MACHINE_X_MIN_MM = 0.0
PROTOCAL_MACHINE_X_MAX_MM = 181.0
PROTOCAL_MACHINE_Y_MIN_MM = 0.0
PROTOCAL_MACHINE_Y_MAX_MM = 305.0
PROTOCAL_ROTATION_MIN_DEG = -180.0
PROTOCAL_ROTATION_MAX_DEG = 180.0
