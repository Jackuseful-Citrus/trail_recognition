"""Non-persistent CanMV K230 hardware-control probe.

This file is sent as an in-memory IDE script.  It does not access the camera,
filesystem, GPIO, motors, or other peripherals.
"""

import sys
import time


print("@@K230_PROBE_BEGIN")
print("@@K230_PYTHON_VERSION", sys.version)
print("@@K230_IMPLEMENTATION", sys.implementation)
for i in range(10):
    print("@@K230_HEARTBEAT", i)
    time.sleep_ms(200)
print("@@K230_PROBE_END")

# Remain interruptible so the host can prove that stop works after collecting
# the complete bounded heartbeat sequence.
while True:
    time.sleep_ms(1000)
