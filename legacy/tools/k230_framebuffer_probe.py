"""Safe synthetic framebuffer probe for CanMV IDE preview."""

import image
import time


frame = image.Image(320, 240, image.RGB565)
frame.clear()
frame.draw_rectangle(
    8, 8, 304, 224, color=(0, 255, 0), thickness=4, fill=False
)
frame.draw_string(24, 96, "K230 CODEX PROBE", scale=2, color=(255, 255, 255))
print("@@K230_FRAMEBUFFER_READY")

while True:
    frame.compress_for_ide()
    time.sleep_ms(100)
