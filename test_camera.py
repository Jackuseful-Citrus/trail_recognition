"""Manual K230 camera/display smoke test.

The hardware imports intentionally live inside ``main`` so desktop unit-test
discovery can import this file without pretending that CanMV media APIs exist.
"""

_DEFAULT_WHITE = (235, 235, 235)


def _draw_text(canvas, x, y, text, color):
    try:
        canvas.draw_string_advanced(
            int(x),
            int(y),
            16,
            text,
            color=color,
        )
    except Exception:
        try:
            canvas.draw_string(int(x), int(y), text, color=color)
        except Exception:
            pass


def main():
    import time
    import os

    from media.display import Display
    from media.media import MediaManager
    from media.sensor import Sensor
    import image as image_module

    WHITE = getattr(image_module, "WHITE", _DEFAULT_WHITE)

    sensor = Sensor()
    sensor.reset()
    sensor.set_hmirror(True)
    sensor.set_vflip(True)
    sensor.set_framesize(width=800, height=480)
    sensor.set_pixformat(Sensor.RGB565)

    Display.init(
        Display.ST7701,
        width=800,
        height=480,
        to_ide=True,
    )
    MediaManager.init()
    sensor.run()

    while True:
        frame = sensor.snapshot()
        width = frame.width()
        height = frame.height()

        _draw_text(
            frame,
            8,
            8,
            f"w={width} h={height}",
            WHITE,
        )
        Display.show_image(frame)
        sender = getattr(frame, "compress_for_ide", None)
        if sender is not None:
            try:
                os.exitpoint()
                sender(quality=50)
                os.exitpoint()
            except Exception:
                pass
        time.sleep_ms(10)


if __name__ == "__main__":
    main()
