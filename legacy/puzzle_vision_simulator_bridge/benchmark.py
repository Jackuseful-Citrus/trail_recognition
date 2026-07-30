"""Compare planners on the same locally detected real-photo polygons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np

import puzzle_config as cfg
from puzzle_geometry import (
    plan_outer_first_rectangle,
    plan_rectangle_assembly,
)
from puzzle_vision import detect_pieces_from_gray

from .adapter import (
    plan_with_upstream,
    plan_with_upstream_then_outer_fallback,
)


def _measure(callable_, rounds: int):
    durations = []
    result = None
    for _ in range(rounds):
        started = time.perf_counter()
        result = callable_()
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": statistics.median(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "rounds": rounds,
        "valid": result.valid,
        "mode": result.mode,
        "score": result.score,
        "fill_gap_mm2": result.fill_gap_mm2,
        "overlap_mm2": result.overlap_mm2,
        "outside_mm2": result.outside_mm2,
        "search_nodes": result.search_nodes,
        "reason": result.reason,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(cfg.OFFLINE_IMAGE),
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fixed-rounds", type=int, default=1)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.fixed_rounds < 1:
        parser.error("round counts must be positive")

    gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit("cannot read {}".format(args.image))
    pieces, _ = detect_pieces_from_gray(
        gray,
        cfg.OFFLINE_A4_CORNERS_PX,
        cv2,
        np,
    )
    for index, piece in enumerate(pieces):
        piece.piece_id = "P{}".format(index + 1)

    results = {
        "input": {
            "image": str(args.image.resolve()),
            "piece_count": len(pieces),
            "vertex_counts": [len(piece.polygon_mm) for piece in pieces],
            "timing_scope": "planner only; identical PieceObservation input",
        },
        "planners": {
            "upstream_bridge_local_validation": _measure(
                lambda: plan_with_upstream(
                    pieces,
                    upstream_root=args.upstream_root,
                    validation="local",
                ),
                args.rounds,
            ),
            "local_outer_first": _measure(
                lambda: plan_outer_first_rectangle(pieces),
                args.rounds,
            ),
            "upstream_then_outer_fallback": _measure(
                lambda: plan_with_upstream_then_outer_fallback(
                    pieces,
                    upstream_root=args.upstream_root,
                ),
                args.rounds,
            ),
            "local_fixed_default_dispatch": _measure(
                lambda: plan_rectangle_assembly(pieces),
                args.fixed_rounds,
            ),
        },
    }
    upstream_ms = results["planners"][
        "upstream_bridge_local_validation"
    ]["median_ms"]
    outer_ms = results["planners"]["local_outer_first"]["median_ms"]
    fixed_ms = results["planners"][
        "local_fixed_default_dispatch"
    ]["median_ms"]
    results["ratios"] = {
        "fixed_over_upstream": fixed_ms / max(1e-9, upstream_ms),
        "fixed_over_outer_first": fixed_ms / max(1e-9, outer_ms),
        "upstream_over_outer_first": upstream_ms / max(1e-9, outer_ms),
    }

    encoded = json.dumps(results, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

