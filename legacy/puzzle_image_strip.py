"""Optional non-background segmentation and seam image-strip scoring.

Geometry creates and filters candidates first. These helpers only provide an
optional ordering cost for the small surviving candidate set.
"""

import math

import puzzle_config as cfg
from puzzle_geometry import point_in_polygon


class EdgeImageStrip:
    __slots__ = (
        "gray_samples",
        "gradient_samples",
        "valid_mask",
        "sample_count",
        "along_count",
        "depth_count",
    )

    def __init__(
        self,
        gray_samples,
        gradient_samples,
        valid_mask,
        along_count,
        depth_count,
    ):
        self.gray_samples = list(gray_samples)
        self.gradient_samples = list(gradient_samples)
        self.valid_mask = bytearray(valid_mask)
        self.along_count = int(along_count)
        self.depth_count = int(depth_count)
        self.sample_count = sum(
            1 for value in self.valid_mask if value
        )


def pixel_is_piece(
    pixel,
    mode=None,
    background_rgb=None,
    distance_threshold=None,
):
    """Classify a grayscale/RGB pixel using an explicit segmentation mode."""
    if mode is None:
        mode = cfg.BACKGROUND_SEGMENTATION_MODE
    if mode == "white":
        if isinstance(pixel, (tuple, list)):
            gray = (
                0.299 * float(pixel[0])
                + 0.587 * float(pixel[1])
                + 0.114 * float(pixel[2])
            )
        else:
            gray = float(pixel)
        return gray >= cfg.WHITE_GRAY_THRESHOLD
    if mode != "non_background_rgb":
        raise ValueError(
            "unknown background segmentation mode {}".format(mode)
        )
    if background_rgb is None:
        background_rgb = cfg.BACKGROUND_COLOR_RGB
    if distance_threshold is None:
        distance_threshold = (
            cfg.BACKGROUND_COLOR_DISTANCE_THRESHOLD
        )
    if not isinstance(pixel, (tuple, list)) or len(pixel) < 3:
        raise ValueError(
            "non_background_rgb requires an RGB pixel"
        )
    squared = 0.0
    for channel in range(3):
        delta = (
            float(pixel[channel])
            - float(background_rgb[channel])
        )
        squared += delta * delta
    return squared >= float(distance_threshold) ** 2


def build_non_background_mask(
    rgb_array,
    background_rgb=None,
    distance_threshold=None,
):
    """Return rows of 0/255 bytes; card patterns remain part of a piece."""
    height = int(rgb_array.shape[0])
    width = int(rgb_array.shape[1])
    rows = []
    for y in range(height):
        row = bytearray(width)
        for x in range(width):
            pixel = rgb_array[y][x]
            value = (
                pixel
                if isinstance(pixel, (tuple, list))
                else tuple(int(channel) for channel in pixel)
            )
            if pixel_is_piece(
                value,
                mode="non_background_rgb",
                background_rgb=background_rgb,
                distance_threshold=distance_threshold,
            ):
                row[x] = 255
        rows.append(row)
    return rows


def sample_edge_image_strip(
    gray_array,
    polygon_mm,
    edge_index,
    strip_width_mm=None,
    sample_spacing_mm=None,
):
    """Sample an edge's inward strip in rectified A4 coordinates."""
    if strip_width_mm is None:
        strip_width_mm = cfg.IMAGE_STRIP_WIDTH_MM
    if sample_spacing_mm is None:
        sample_spacing_mm = cfg.IMAGE_STRIP_SAMPLE_SPACING_MM
    spacing = max(0.25, float(sample_spacing_mm))
    width_mm = max(spacing, float(strip_width_mm))
    height = int(gray_array.shape[0])
    width = int(gray_array.shape[1])
    p0 = polygon_mm[edge_index]
    p1 = polygon_mm[(edge_index + 1) % len(polygon_mm)]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 1e-9:
        return EdgeImageStrip([], [], [], 0, 0)
    ux = dx / length
    uy = dy / length
    # PieceObservation polygons are clockwise in mathematical coordinates.
    inward_x = uy
    inward_y = -ux
    along_count = max(1, int(length / spacing))
    depth_count = max(1, int(width_mm / spacing))
    pixels_per_mm_x = float(width - 1) / cfg.A4_WIDTH_MM
    pixels_per_mm_y = float(height - 1) / cfg.A4_HEIGHT_MM
    gray_samples = []
    gradient_samples = []
    valid_mask = bytearray(along_count * depth_count)
    output_index = 0
    for along_index in range(along_count):
        along = min(
            length - 0.25 * spacing,
            (along_index + 0.5) * length / along_count,
        )
        for depth_index in range(depth_count):
            depth = (depth_index + 0.5) * width_mm / depth_count
            point = (
                p0[0] + along * ux + depth * inward_x,
                p0[1] + along * uy + depth * inward_y,
            )
            px = int(round(point[0] * pixels_per_mm_x))
            py = int(round(point[1] * pixels_per_mm_y))
            valid = (
                1 <= px < width - 1
                and 1 <= py < height - 1
                and point_in_polygon(point, polygon_mm)
            )
            if valid:
                gray_value = int(gray_array[py][px])
                gradient = 0.5 * (
                    abs(
                        int(gray_array[py][px + 1])
                        - int(gray_array[py][px - 1])
                    )
                    + abs(
                        int(gray_array[py + 1][px])
                        - int(gray_array[py - 1][px])
                    )
                )
                valid_mask[output_index] = 1
            else:
                gray_value = 0
                gradient = 0.0
            gray_samples.append(gray_value)
            gradient_samples.append(gradient)
            output_index += 1
    return EdgeImageStrip(
        gray_samples,
        gradient_samples,
        valid_mask,
        along_count,
        depth_count,
    )


def compute_edge_strip_cost(strip_a, strip_b):
    """Compare edge B in reverse direction against edge A; lower is better."""
    if (
        strip_a.along_count <= 0
        or strip_a.along_count != strip_b.along_count
        or strip_a.depth_count != strip_b.depth_count
    ):
        return None
    gray_loss = 0.0
    gradient_loss = 0.0
    continuity_loss = 0.0
    count = 0
    depth_count = strip_a.depth_count
    for along in range(strip_a.along_count):
        reverse_along = strip_b.along_count - 1 - along
        for depth in range(depth_count):
            index_a = along * depth_count + depth
            index_b = reverse_along * depth_count + depth
            if (
                not strip_a.valid_mask[index_a]
                or not strip_b.valid_mask[index_b]
            ):
                continue
            gray_loss += abs(
                strip_a.gray_samples[index_a]
                - strip_b.gray_samples[index_b]
            ) / 255.0
            gradient_loss += min(
                1.0,
                abs(
                    strip_a.gradient_samples[index_a]
                    - strip_b.gradient_samples[index_b]
                )
                / 255.0,
            )
            if depth == 0:
                continuity_loss += abs(
                    strip_a.gray_samples[index_a]
                    - strip_b.gray_samples[index_b]
                ) / 255.0
            count += 1
    if count <= 0:
        return None
    normalizer = float(count)
    return (
        0.55 * gray_loss / normalizer
        + 0.30 * gradient_loss / normalizer
        + 0.15 * continuity_loss / max(
            1.0, float(strip_a.along_count)
        )
    )


def apply_optional_strip_costs(candidate_graph, strips_by_edge):
    """Attach optional costs without admitting a new geometric candidate."""
    if not cfg.ENABLE_IMAGE_STRIP_MATCHING:
        return 0
    applied = 0
    for candidate in candidate_graph.candidates:
        strip_a = strips_by_edge.get(
            (candidate.piece_a, candidate.edge_a)
        )
        strip_b = strips_by_edge.get(
            (candidate.piece_b, candidate.edge_b)
        )
        if strip_a is None or strip_b is None:
            continue
        cost = compute_edge_strip_cost(strip_a, strip_b)
        if cost is None:
            continue
        candidate.optional_strip_cost = cost
        candidate.geometric_cost += cfg.IMAGE_STRIP_WEIGHT * cost
        applied += 1
    for values in candidate_graph.candidates_by_piece_pair.values():
        values.sort(key=lambda item: item.geometric_cost)
    for values in candidate_graph.candidates_by_edge.values():
        values.sort(key=lambda item: item.geometric_cost)
    return applied
