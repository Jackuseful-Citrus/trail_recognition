import time

from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager


sensor = Sensor()

sensor.reset()


# 摄像头方向修正
sensor.set_hmirror(True)
sensor.set_vflip(True)


# 匹配屏幕
sensor.set_framesize(
    width=800,
    height=480
)

sensor.set_pixformat(
    Sensor.RGB565
)


Display.init(
    Display.ST7701,
    width=800,
    height=480,
    to_ide=True
)


MediaManager.init()


sensor.run()


while True:

    img = sensor.snapshot()

    Display.show_image(img)

    time.sleep_ms(10)