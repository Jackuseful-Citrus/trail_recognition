"""Manual K230 camera/display smoke test.

The hardware imports intentionally live inside ``main`` so desktop unit-test
discovery can import this file without pretending that CanMV media APIs exist.
"""

_A4_CORNERS_FALLBACK_PX = [
    (133.0, 441.0),
    (140.0, 78.0),
    (619.0, 83.0),
    (614.0, 449.0),
]

def _ordered_a4_rect(points):
    if len(points) != 4:
        return points

    top_left = min(points, key=lambda p: p[0] + p[1])
    bottom_right = max(points, key=lambda p: p[0] + p[1])
    top_right = max(points, key=lambda p: p[0] - p[1])
    bottom_left = min(points, key=lambda p: p[0] - p[1])

    # Preserve a deterministic TL-TR-BR-BL order for drawing and labeling.
    return [top_left, top_right, bottom_right, bottom_left]


_DEFAULT_WHITE = (235, 235, 235)
_DEFAULT_GREEN = (70, 240, 100)
_DEFAULT_YELLOW = (255, 210, 40)

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


def _load_a4_corners():
    try:
        import sys

        base_dir = __file__.rsplit("/", 1)[0]
        cfg_dir = base_dir + "/k230_realtime_a4"
        if cfg_dir not in sys.path:
            sys.path.insert(0, cfg_dir)
        from realtime_a4_config import A4_CORNERS_PX
        config_points = A4_CORNERS_PX
    except Exception:
        config_points = _A4_CORNERS_FALLBACK_PX

    try:
        points = [
            (float(x), float(y))
            for x, y in config_points
            if x is not None and y is not None
        ]

        if len(points) != 4:
            return points

        return _ordered_a4_rect(points)
    except Exception:
        return list(_A4_CORNERS_FALLBACK_PX)


def main():
    import time
    import os

    from media.display import Display
    from media.media import MediaManager
    from media.sensor import Sensor
    import image as image_module

    WHITE = getattr(image_module, "WHITE", _DEFAULT_WHITE)
    GREEN = getattr(image_module, "GREEN", _DEFAULT_GREEN)
    YELLOW = getattr(image_module, "YELLOW", _DEFAULT_YELLOW)

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

    a4_corners = _load_a4_corners()
    corner_labels = [
        ("TL", 0.0, 0.0),
        ("TR", 210.0, 0.0),
        ("BR", 210.0, 297.0),
        ("BL", 0.0, 297.0),
    ]
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
        cx = width // 2
        cy = height // 2
        frame.draw_cross(cx, cy, color=GREEN, size=10, thickness=2)
        _draw_text(
            frame,
            cx + 12,
            max(0, cy - 12),
            f"({cx},{cy})",
            GREEN,
        )

        if len(a4_corners) >= 4:
            # Connect the four points into a rectangle in TL-TR-BR-BL order.
            for index in range(4):
                x1, y1 = a4_corners[index]
                x2, y2 = a4_corners[(index + 1) % 4]
                frame.draw_line(
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                    color=GREEN,
                    thickness=2,
                )

            for index, (px, py) in enumerate(a4_corners):
                label, ox, oy = corner_labels[index]
                frame.draw_cross(
                    int(px),
                    int(py),
                    color=YELLOW,
                    size=8,
                    thickness=2,
                )
                _draw_text(
                    frame,
                    int(px) + 6,
                    int(py) + 6,
                    f"{label}  ({ox:.0f},{oy:.0f})",
                    YELLOW,
                )
                _draw_text(
                    frame,
                    int(px) + 6,
                    int(py) + 18,
                    f"pix({px:.0f},{py:.0f})",
                    YELLOW,
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
