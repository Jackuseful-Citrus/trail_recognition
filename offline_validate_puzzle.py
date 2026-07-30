#!/usr/bin/env python3
"""Desktop validation for the shared A4 puzzle detector and planner."""

import argparse
import json
import os
import sys

import cv2
import numpy as np

import puzzle_config as cfg
from puzzle_geometry import plan_rectangle_assembly
from puzzle_vision import detect_pieces_from_gray, mm_to_rectified_px


COLORS = [
    (60, 220, 255),
    (90, 235, 90),
    (255, 150, 60),
    (220, 90, 230),
]


def _parse_corners(text):
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 8:
        raise argparse.ArgumentTypeError(
            "corners must be x1,y1,x2,y2,x3,y3,x4,y4"
        )
    return [
        (values[index], values[index + 1])
        for index in range(0, len(values), 2)
    ]


def _draw_polygon(canvas, polygon_mm, color, y_offset=0, label=None):
    points = []
    for point in polygon_mm:
        x, y = mm_to_rectified_px(point)
        points.append((int(round(x)), int(round(y + y_offset))))
    cv2.polylines(
        canvas,
        [np.array(points, dtype=np.int32)],
        True,
        color,
        2,
        cv2.LINE_AA,
    )
    if label:
        centre = np.mean(np.array(points, dtype=np.float32), axis=0)
        cv2.putText(
            canvas,
            label,
            (int(centre[0]) + 4, int(centre[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _render_result(diagnostics, pieces, plan):
    rectified = diagnostics["rectified"]
    current = cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
    divider_y = int(
        diagnostics["divider_y_mm"] * cfg.RECTIFIED_PX_PER_MM + 0.5
    )
    cv2.line(
        current,
        (0, divider_y),
        (current.shape[1] - 1, divider_y),
        (140, 140, 140),
        1,
    )
    for index, piece in enumerate(pieces):
        color = COLORS[index % len(COLORS)]
        _draw_polygon(current, piece.polygon_mm, color, label=piece.piece_id)
        cx, cy = mm_to_rectified_px(piece.centroid_mm)
        centre = (int(round(cx)), int(round(cy)))
        cv2.drawMarker(
            current,
            centre,
            color,
            cv2.MARKER_CROSS,
            14,
            2,
        )
        for vertex in piece.polygon_mm:
            vx, vy = mm_to_rectified_px(vertex)
            cv2.circle(
                current,
                (int(round(vx)), int(round(vy))),
                3,
                color,
                -1,
            )

    target = np.zeros_like(current)
    target[:] = (24, 24, 24)
    divider_px = int(cfg.DIVIDER_Y_MM * cfg.RECTIFIED_PX_PER_MM + 0.5)
    cv2.line(
        target,
        (0, divider_px),
        (target.shape[1] - 1, divider_px),
        (180, 180, 180),
        1,
    )
    if plan.valid:
        min_x, min_y, max_x, max_y = plan.target_rect
        rect_a = mm_to_rectified_px((min_x, min_y))
        rect_b = mm_to_rectified_px((max_x, max_y))
        cv2.rectangle(
            target,
            (int(round(rect_a[0])), int(round(rect_a[1]))),
            (int(round(rect_b[0])), int(round(rect_b[1]))),
            (105, 105, 105),
            1,
        )
        operation_by_id = {
            operation["piece_id"]: operation
            for operation in plan.operations
        }
        for index, piece in enumerate(pieces):
            color = COLORS[index % len(COLORS)]
            polygon = plan.target_polygons[piece.piece_id]
            _draw_polygon(target, polygon, color, label=piece.piece_id)
            operation = operation_by_id[piece.piece_id]
            cx, cy = mm_to_rectified_px(
                operation["target_center_mm"]
            )
            cv2.drawMarker(
                target,
                (int(round(cx)), int(round(cy))),
                color,
                cv2.MARKER_CROSS,
                14,
                2,
            )
            cv2.putText(
                target,
                "{} {:+.1f} deg".format(
                    piece.piece_id, operation["rotation_deg"]
                ),
                (8, 22 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
        status = "VALID {} score={:.4f} vmax={:.1f}mm".format(
            plan.mode.upper(),
            plan.score,
            plan.max_vertex_error_mm or 0.0,
        )
        status_color = (80, 240, 80)
    else:
        status = "NO VALID PLAN: {}".format(plan.reason)
        status_color = (40, 80, 255)
    cv2.putText(
        target,
        status,
        (8, target.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        status_color,
        1,
        cv2.LINE_AA,
    )
    return np.hstack((current, target))


def _print_structured(plan, pieces, frame=0):
    if not plan.valid:
        print("PLAN_INVALID,frame={},reason={}".format(frame, plan.reason))
        return
    print(
        "PLAN,frame={},stable=1,count={},mode={},score={:.4f},"
        "max_vertex_error_mm={:.1f},gap_mm2={:.1f},overlap_mm2={:.1f},"
        "outside_mm2={:.1f},nodes={}".format(
            frame,
            len(pieces),
            plan.mode,
            plan.score,
            plan.max_vertex_error_mm or 0.0,
            plan.fill_gap_mm2 or 0.0,
            plan.overlap_mm2 or 0.0,
            plan.outside_mm2 or 0.0,
            plan.search_nodes,
        )
    )
    for operation in plan.operations:
        source = operation["source_center_mm"]
        target = operation["target_center_mm"]
        print(
            "PIECE,id={},sx_mm={:.1f},sy_mm={:.1f},tx_mm={:.1f},"
            "ty_mm={:.1f},rot_deg={:.1f},ambiguous={},confidence={:.2f}".format(
                operation["piece_id"],
                source[0],
                source[1],
                target[0],
                target[1],
                operation["rotation_deg"],
                int(operation["rotation_ambiguous"]),
                operation["confidence"],
            )
        )
    print("PLAN_END")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate shared white-piece detection and geometry planning"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=cfg.OFFLINE_IMAGE,
        help="input photograph",
    )
    parser.add_argument(
        "--corners",
        type=_parse_corners,
        default=cfg.OFFLINE_A4_CORNERS_PX,
        help="TL,TR,BR,BL as x1,y1,...,x4,y4",
    )
    parser.add_argument(
        "--output",
        default=cfg.OFFLINE_OUTPUT_IMAGE,
        help="annotated output image",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        help="optional machine-readable observation/plan record",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open an interactive result window",
    )
    args = parser.parse_args(argv)

    source = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if source is None:
        print(
            "DETECTION_ERROR,reason=cannot read {}".format(args.image),
            file=sys.stderr,
        )
        return 2
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    try:
        pieces, diagnostics = detect_pieces_from_gray(
            gray, args.corners, cv2, np
        )
    except Exception as exc:
        print(
            "DETECTION_ERROR,reason={}".format(str(exc).replace(",", ";")),
            file=sys.stderr,
        )
        return 2

    for index, piece in enumerate(pieces):
        piece.piece_id = "P{}".format(index + 1)
        piece.stable = True

    print(
        "DETECTION,frame=0,count={},divider_y_mm={:.1f},"
        "divider_detected={},threshold={:.1f},mode={}".format(
            len(pieces),
            diagnostics["divider_y_mm"],
            int(diagnostics["divider_detected"]),
            diagnostics["threshold"],
            diagnostics["threshold_mode"],
        )
    )
    for piece in pieces:
        print(
            "OBS,id={},vertices={},cx_mm={:.1f},cy_mm={:.1f},"
            "area_mm2={:.1f},orientation_deg={:.1f},confidence={:.2f}".format(
                piece.piece_id,
                len(piece.polygon_mm),
                piece.centroid_mm[0],
                piece.centroid_mm[1],
                piece.area_mm2,
                piece.current_orientation_deg,
                piece.confidence,
            )
        )

    plan = plan_rectangle_assembly(pieces)
    _print_structured(plan, pieces)
    rendered = _render_result(diagnostics, pieces, plan)
    if not cv2.imwrite(args.output, rendered):
        print(
            "DETECTION_ERROR,reason=cannot write {}".format(args.output),
            file=sys.stderr,
        )
        return 2
    print("OUTPUT,image={}".format(os.path.abspath(args.output)))

    if args.json_output:
        record = {
            "image": os.path.abspath(args.image),
            "corners_px": args.corners,
            "divider_y_mm": diagnostics["divider_y_mm"],
            "pieces": [
                {
                    "piece_id": piece.piece_id,
                    "polygon_mm": piece.polygon_mm,
                    "centroid_mm": piece.centroid_mm,
                    "area_mm2": piece.area_mm2,
                    "edge_lengths_mm": piece.edge_lengths_mm,
                    "interior_angles_deg": piece.interior_angles_deg,
                    "current_orientation_deg": piece.current_orientation_deg,
                    "confidence": piece.confidence,
                }
                for piece in pieces
            ],
            "plan": {
                "valid": plan.valid,
                "reason": plan.reason,
                "score": plan.score,
                "mode": plan.mode,
                "max_vertex_error_mm": plan.max_vertex_error_mm,
                "fill_gap_mm2": plan.fill_gap_mm2,
                "overlap_mm2": plan.overlap_mm2,
                "outside_mm2": plan.outside_mm2,
                "search_nodes": plan.search_nodes,
                "plan_stats": plan.plan_stats,
                "operations": plan.operations,
                "target_polygons": plan.target_polygons,
                "seams": plan.seams,
            },
        }
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        print("OUTPUT,json={}".format(os.path.abspath(args.json_output)))

    if args.show:
        cv2.imshow("A4 puzzle validation: current | target", rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if plan.valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
