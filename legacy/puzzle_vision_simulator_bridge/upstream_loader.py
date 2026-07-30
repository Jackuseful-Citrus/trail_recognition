"""Load the pinned upstream solver without copying it into the local source tree."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType


PINNED_COMMIT = "e9eb2e0fb945c348eedd0b0fa9258f5518d2892f"
UPSTREAM_ENV = "PUZZLE_VISION_SIMULATOR_ROOT"
DEFAULT_UPSTREAM_ROOT = (
    Path(__file__).resolve().parent
    / ".upstream"
    / "puzzle-vision-simulator"
)


class UpstreamUnavailableError(RuntimeError):
    """The pinned puzzle-vision-simulator checkout cannot be loaded."""


def resolve_upstream_root(upstream_root: str | os.PathLike | None = None) -> Path:
    """Resolve an explicit path, environment override, or the bridge cache."""
    if upstream_root is not None:
        return Path(upstream_root).expanduser().resolve()
    configured = os.environ.get(UPSTREAM_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_UPSTREAM_ROOT


def _git_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def load_upstream(
    upstream_root: str | os.PathLike | None = None,
    *,
    strict_revision: bool = True,
) -> ModuleType:
    """Import the upstream ``puzzle_sim.py`` from a verified checkout."""
    root = resolve_upstream_root(upstream_root)
    source = root / "puzzle_sim.py"
    if not source.is_file():
        raise UpstreamUnavailableError(
            "puzzle-vision-simulator is unavailable at {}. Run "
            "`python3 -m puzzle_vision_simulator_bridge.fetch_upstream` "
            "or set {}.".format(root, UPSTREAM_ENV)
        )

    revision = _git_revision(root)
    if strict_revision and revision != PINNED_COMMIT:
        raise UpstreamUnavailableError(
            "upstream revision mismatch at {}: expected {}, got {}".format(
                root,
                PINNED_COMMIT,
                revision or "unknown",
            )
        )

    module_name = "_puzzle_vision_simulator_{}".format(
        (revision or "unversioned")[:12]
    )
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise UpstreamUnavailableError(
            "cannot create an import specification for {}".format(source)
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module

