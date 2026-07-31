"""Final source-clear detection helpers for the K230 puzzle."""

import puzzle_config as cfg
from puzzle_geometry import (
    PieceObservation,
)


def clone_piece(piece):
    return PieceObservation(
        piece.piece_id,
        list(piece.contour_px),
        list(piece.polygon_mm),
        centroid_mm=piece.centroid_mm,
        area_mm2=piece.area_mm2,
        edge_lengths_mm=piece.edge_lengths_mm,
        interior_angles_deg=piece.interior_angles_deg,
        current_orientation_deg=piece.current_orientation_deg,
        confidence=piece.confidence,
        rotation_ambiguous=piece.rotation_ambiguous,
        centroid_fallback=piece.centroid_fallback,
        calibration_generation=piece.calibration_generation,
    )


def final_foreground_mask_from_gray(
    gray_array,
    threshold,
    border_px=0,
    divider_y_mm=None,
    divider_margin_mm=0.0,
):
    """Build the final-scene mask while excluding fixed non-piece borders."""
    height = int(gray_array.shape[0])
    width = int(gray_array.shape[1])
    border = max(0, int(border_px))
    divider = (
        cfg.DIVIDER_Y_MM
        if divider_y_mm is None
        else float(divider_y_mm)
    )
    divider_margin = max(0.0, float(divider_margin_mm))
    result = bytearray(width * height)
    index = 0
    for y in range(height):
        point_y = (y + 0.5) * cfg.A4_HEIGHT_MM / height
        excluded_row = (
            y < border
            or y >= height - border
            or abs(point_y - divider) <= divider_margin
        )
        row = gray_array[y]
        for x in range(width):
            result[index] = (
                1
                if (
                    not excluded_row
                    and x >= border
                    and x < width - border
                    and int(row[x]) >= int(threshold)
                )
                else 0
            )
            index += 1
    return result


def final_region_white_metrics(
    foreground_mask,
    width,
    height,
    initial_total_piece_area,
    divider_y_mm=None,
):
    """Return upper/lower white areas relative to the frozen initial area."""
    width = int(width)
    height = int(height)
    if len(foreground_mask) != width * height:
        raise ValueError("final scene mask dimensions differ")
    divider = (
        cfg.DIVIDER_Y_MM
        if divider_y_mm is None
        else float(divider_y_mm)
    )
    upper_count = 0
    lower_count = 0
    for y in range(height):
        point_y = (y + 0.5) * cfg.A4_HEIGHT_MM / height
        row_start = y * width
        row_count = 0
        for x in range(width):
            if foreground_mask[row_start + x]:
                row_count += 1
        if point_y < divider:
            upper_count += row_count
        else:
            lower_count += row_count
    pixel_area = (
        cfg.A4_WIDTH_MM
        * cfg.A4_HEIGHT_MM
        / float(width * height)
    )
    initial_area = max(1e-9, float(initial_total_piece_area))
    upper_area = upper_count * pixel_area
    lower_area = lower_count * pixel_area
    return {
        "upper_white_area_mm2": upper_area,
        "lower_white_area_mm2": lower_area,
        "upper_remaining_ratio": upper_area / initial_area,
        "lower_area_ratio": lower_area / initial_area,
        "upper_foreground_count": upper_count,
        "lower_foreground_count": lower_count,
    }
