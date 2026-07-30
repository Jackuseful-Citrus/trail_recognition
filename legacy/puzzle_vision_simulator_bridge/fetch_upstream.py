"""Fetch the exact upstream revision used by the bridge."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from .upstream_loader import (
    DEFAULT_UPSTREAM_ROOT,
    PINNED_COMMIT,
    _git_revision,
)


REPOSITORY = "https://github.com/lvreng/puzzle-vision-simulator.git"


def fetch(destination: Path = DEFAULT_UPSTREAM_ROOT) -> Path:
    """Clone the pinned revision into an empty bridge dependency directory."""
    destination = destination.expanduser().resolve()
    if destination.exists():
        revision = _git_revision(destination)
        if revision == PINNED_COMMIT and (destination / "puzzle_sim.py").is_file():
            return destination
        raise RuntimeError(
            "refusing to overwrite existing path {} (revision={})".format(
                destination,
                revision or "unknown",
            )
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            REPOSITORY,
            str(destination),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "--depth",
            "1",
            "origin",
            PINNED_COMMIT,
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "checkout",
            "--detach",
            PINNED_COMMIT,
        ],
        check=True,
    )
    if _git_revision(destination) != PINNED_COMMIT:
        raise RuntimeError("upstream checkout verification failed")
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_UPSTREAM_ROOT,
        help="empty destination directory",
    )
    args = parser.parse_args(argv)
    destination = fetch(args.destination)
    print("UPSTREAM_READY,path={},commit={}".format(destination, PINNED_COMMIT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
