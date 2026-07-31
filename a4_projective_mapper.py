"""Pure-Python projective mapping between source pixels and A4 millimetres.

The mapper deliberately transforms coordinates only.  It never resamples an
image, so it is safe to use in the K230 source-projective recognition path.
Corners always use the existing physical order TL, TR, BR, BL.
"""

import math

from puzzle_a4_boundary import _unit_square_transform


_MAPPER_EPSILON = 1e-10


def _finite(value):
    try:
        return not (math.isnan(value) or math.isinf(value))
    except AttributeError:
        return value == value and abs(value) != float("inf")


def _invert_3x3(matrix):
    a, b, c, d, e, f, g, h, i = matrix
    aa = e * i - f * h
    ab = c * h - b * i
    ac = b * f - c * e
    ad = f * g - d * i
    ae = a * i - c * g
    af = c * d - a * f
    ag = d * h - e * g
    ah = b * g - a * h
    ai = a * e - b * d
    determinant = a * aa + b * ad + c * ag
    if not _finite(determinant) or abs(determinant) <= _MAPPER_EPSILON:
        return None, determinant
    inverse_scale = 1.0 / determinant
    inverse = tuple(
        value * inverse_scale
        for value in (aa, ab, ac, ad, ae, af, ag, ah, ai)
    )
    if not all(_finite(value) for value in inverse):
        return None, determinant
    return inverse, determinant


def _unit_square_homography(corners):
    """Extend the project's canonical unit-square transform to 3x3."""
    transform = _unit_square_transform(corners)
    if transform is None:
        return None
    matrix = tuple(transform) + (1.0,)
    if not all(_finite(value) for value in matrix):
        return None
    return matrix


def _project(matrix, x, y):
    denominator = matrix[6] * x + matrix[7] * y + matrix[8]
    if not _finite(denominator) or abs(denominator) <= _MAPPER_EPSILON:
        return None
    projected_x = (
        matrix[0] * x + matrix[1] * y + matrix[2]
    ) / denominator
    projected_y = (
        matrix[3] * x + matrix[4] * y + matrix[5]
    ) / denominator
    if not (_finite(projected_x) and _finite(projected_y)):
        return None
    return projected_x, projected_y


class A4ProjectiveMapper:
    """Coordinate-only source-pixel <-> physical-A4 projector."""

    __slots__ = (
        "corners_source_px",
        "source_width",
        "source_height",
        "a4_width_mm",
        "a4_height_mm",
        "a4_polygon_source_px",
        "a4_bbox_source_px",
        "valid",
        "condition_metric",
        "_a4_to_source",
        "_source_to_a4_unit",
    )

    def __init__(
        self,
        corners_source_px,
        source_width,
        source_height,
        a4_width_mm=210.0,
        a4_height_mm=297.0,
    ):
        self.source_width = int(source_width)
        self.source_height = int(source_height)
        self.a4_width_mm = float(a4_width_mm)
        self.a4_height_mm = float(a4_height_mm)
        self.valid = False
        self.condition_metric = float("inf")
        self._a4_to_source = None
        self._source_to_a4_unit = None
        self.corners_source_px = []
        self.a4_polygon_source_px = []
        self.a4_bbox_source_px = (0, 0, 0, 0)
        if (
            corners_source_px is None
            or len(corners_source_px) != 4
            or self.source_width < 2
            or self.source_height < 2
            or self.a4_width_mm <= 0.0
            or self.a4_height_mm <= 0.0
        ):
            return
        corners = [
            (float(point[0]), float(point[1]))
            for point in corners_source_px
        ]
        if not all(
            _finite(value)
            for point in corners
            for value in point
        ):
            return
        forward = _unit_square_homography(corners)
        if forward is None:
            return
        inverse, determinant = _invert_3x3(forward)
        if inverse is None:
            return
        magnitude = max(abs(value) for value in forward)
        inverse_magnitude = max(abs(value) for value in inverse)
        condition = magnitude * inverse_magnitude
        if not _finite(condition):
            return
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        x0 = max(0, int(math.floor(min(xs))))
        y0 = max(0, int(math.floor(min(ys))))
        x1 = min(self.source_width - 1, int(math.ceil(max(xs))))
        y1 = min(self.source_height - 1, int(math.ceil(max(ys))))
        if x1 <= x0 or y1 <= y0:
            return
        self.corners_source_px = corners
        self.a4_polygon_source_px = list(corners)
        self.a4_bbox_source_px = (
            x0,
            y0,
            x1 - x0 + 1,
            y1 - y0 + 1,
        )
        self._a4_to_source = forward
        self._source_to_a4_unit = inverse
        self.condition_metric = condition
        self.valid = abs(determinant) > _MAPPER_EPSILON

    def a4_mm_to_source_px(self, point_mm):
        if not self.valid:
            return None
        u = float(point_mm[0]) / self.a4_width_mm
        v = float(point_mm[1]) / self.a4_height_mm
        return _project(self._a4_to_source, u, v)

    def source_px_to_a4_mm(self, point_px):
        if not self.valid:
            return None
        unit = _project(
            self._source_to_a4_unit,
            float(point_px[0]),
            float(point_px[1]),
        )
        if unit is None:
            return None
        result = (
            unit[0] * self.a4_width_mm,
            unit[1] * self.a4_height_mm,
        )
        if not (_finite(result[0]) and _finite(result[1])):
            return None
        return result

    def roundtrip_error_px(self, point_px):
        point_mm = self.source_px_to_a4_mm(point_px)
        if point_mm is None:
            return float("inf")
        restored = self.a4_mm_to_source_px(point_mm)
        if restored is None:
            return float("inf")
        dx = restored[0] - float(point_px[0])
        dy = restored[1] - float(point_px[1])
        return math.sqrt(dx * dx + dy * dy)
