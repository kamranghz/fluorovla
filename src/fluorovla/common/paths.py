"""Location of the repository and of the (non-versioned) data and run directories.

Environment variables (all optional):
  FLUOROVLA_ROOT  repository root (default: two levels above ``src``)
  FLUOROVLA_DATA  data root holding ``raw`` and ``cache`` (default: ``<root>/data``)
  FLUOROVLA_RUNS  run-output root (default: ``<root>/runs``)
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("FLUOROVLA_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[3]


def data_root() -> Path:
    env = os.environ.get("FLUOROVLA_DATA")
    return Path(env) if env else repo_root() / "data"


def runs_root() -> Path:
    env = os.environ.get("FLUOROVLA_RUNS")
    return Path(env) if env else repo_root() / "runs"


def resolve_data(rel: str | os.PathLike) -> Path:
    """Resolve a data-root-relative path (e.g. ``cache/ct/s0011``)."""
    return data_root() / rel
