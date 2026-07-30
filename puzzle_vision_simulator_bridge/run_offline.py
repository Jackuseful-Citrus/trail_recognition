"""Run local A4 detection followed by the bridged upstream planner."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

import puzzle_config as cfg
from offline_validate_puzzle import _render_result
from puzzle_vision import detect_pieces_from_gray

from .adapter import (
    SUPPORTED_CUT_MODES,
    plan_with_upstream,
    plan_with_upstream_then_outer_fallback,
)


def _parse_corners(text: str):
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 8:
        raise argparse.ArgumentTypeError("corners require 8 comma-separated values")
    return [
        (values[index], values[index + 1])
        for index in range(0, 8, 2)
    ]


def _json_record(image: Path, corners, pieces, diagnostics, plan):
    return {
        "image": str(image.resolve()),
        "corners_px": corners,
        "engine": "lvreng/puzzle-vision-simulator bridge",
        "detection": {
            "divider_y_mm": diagnostics["divider_y_mm"],
            "divider_detected": diagnostics["divider_detected"],
            "threshold": diagnostics["threshold"],
            "threshold_mode": diagnostics["threshold_mode"],
            "raw_contours": diagnostics["raw_contours"],
            "rejected": diagnostics["rejected"],
        },
        "pieces": [
            {
                "piece_id": piece.piece_id,
                "polygon_mm": piece.polygon_mm,
                "centroid_mm": piece.centroid_mm,
                "area_mm2": piece.area_mm2,
                "confidence": piece.confidence,
            }
            for piece in pieces
        ],
        "plan": {
            "valid": plan.valid,
            "reason": plan.reason,
            "mode": plan.mode,
            "score": plan.score,
            "max_vertex_error_mm": plan.max_vertex_error_mm,
            "fill_gap_mm2": plan.fill_gap_mm2,
            "overlap_mm2": plan.overlap_mm2,
            "outside_mm2": plan.outside_mm2,
            "target_rect": plan.target_rect,
            "target_polygons": plan.target_polygons,
            "operations": plan.operations,
            "seams": plan.seams,
            "plan_stats": plan.plan_stats,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path(cfg.OFFLINE_IMAGE),
    )
    parser.add_argument(
        "--corners",
        type=_parse_corners,
        default=cfg.OFFLINE_A4_CORNERS_PX,
        help="TL,TR,BR,BL as x1,y1,...,x4,y4",
    )
    parser.add_argument(
        "--cut-mode",
        choices=sorted(SUPPORTED_CUT_MODES),
        default="auto",
        help="Use auto for real camera input unless topology is independently known.",
    )
    parser.add_argument(
        "--validation",
        choices=("local", "upstream"),
        default="local",
        help="Local keeps the current robot-safety geometry gates.",
    )
    parser.add_argument(
        "--fallback-outer",
        action="store_true",
        help="If the locally validated proposal fails, use local outer-first.",
    )
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "offline_bridge_result.png",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        type=Path,
        default=Path(__file__).resolve().parent / "offline_bridge_result.json",
    )
    args = parser.parse_args(argv)

    gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(
            "DETECTION_ERROR,reason=cannot read {}".format(args.image),
            file=sys.stderr,
        )
        return 2
    try:
        pieces, diagnostics = detect_pieces_from_gray(
            gray,
            args.corners,
            cv2,
            np,
        )
        for index, piece in enumerate(pieces):
            piece.piece_id = "P{}".format(index + 1)
            piece.stable = True
        if args.fallback_outer:
            if args.validation != "local":
                parser.error("--fallback-outer requires --validation local")
            plan = plan_with_upstream_then_outer_fallback(
                pieces,
                cut_mode=args.cut_mode,
                upstream_root=args.upstream_root,
            )
        else:
            plan = plan_with_upstream(
                pieces,
                cut_mode=args.cut_mode,
                upstream_root=args.upstream_root,
                validation=args.validation,
            )
    except Exception as exc:
        print(
            "PLAN_INVALID,reason={}".format(
                str(exc).replace(",", ";")
            ),
            file=sys.stderr,
        )
        return 3

    rendered = _render_result(diagnostics, pieces, plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rendered):
        print("OUTPUT_ERROR,image={}".format(args.output), file=sys.stderr)
        return 2
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_record(args.image, args.corners, pieces, diagnostics, plan),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "PLAN,valid={},mode={},time_ms={:.3f},gap_mm2={:.1f},"
        "overlap_mm2={:.1f},outside_mm2={:.1f}".format(
            int(plan.valid),
            plan.mode,
            float(
                plan.plan_stats.get(
                    "hybrid_total_ms",
                    plan.plan_stats.get("plan_ms", 0.0),
                )
            ),
            float(plan.fill_gap_mm2 or 0.0),
            float(plan.overlap_mm2 or 0.0),
            float(plan.outside_mm2 or 0.0),
        )
    )
    print("REASON,{}".format(plan.reason))
    print("OUTPUT,image={}".format(os.path.abspath(args.output)))
    print("OUTPUT,json={}".format(os.path.abspath(args.json_output)))
    return 0 if plan.valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
