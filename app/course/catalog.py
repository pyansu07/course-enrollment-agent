"""Canonical course catalog loader.

The catalog (data/courses.json) is the system of record for course data.
Both the runtime app and the embedding indexer read from here, so there is
exactly one source of truth. In production this would be a real database.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

# data/courses.json lives at the repo root, two levels up from app/course/.
_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "courses.json"


def catalog_path() -> Path:
    """Resolve the catalog file path, allowing an env override for deployments."""
    override = os.environ.get("COURSES_CATALOG_PATH")
    return Path(override) if override else _DEFAULT_CATALOG_PATH


@lru_cache(maxsize=1)
def load_courses() -> list[dict]:
    """Load all courses from the catalog file (cached after first read)."""
    with catalog_path().open(encoding="utf-8") as f:
        return json.load(f)


def courses_by_id() -> dict[str, dict]:
    """Index the catalog by course id for fast lookups."""
    return {c["id"]: c for c in load_courses()}
