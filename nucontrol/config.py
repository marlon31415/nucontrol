"""Load and resolve nucontrol parameters from a YAML file (see ``params.yaml``).

One shared place so the CLI and the tests apply the *same* parameter file and the *same*
precedence rule: an explicit override (CLI flag / call argument) > a ``--config`` YAML value >
the built-in default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .scenario_query import DEFAULT_MAP_VERSION, ScenarioLoader

# The package/repo root (one level up from this module) — where params.yaml, pyproject.toml, etc.
# live. Generated output files are always written here (see cli.main).
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# The params.yaml shipped alongside the package (repo root, one level up from this file).
_SHIPPED_CONFIG = PACKAGE_ROOT / "params.yaml"

# Built-in defaults, mirrored by params.yaml. Grouped to match the YAML sections.
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "dataset": {
        "data_root": None,
        "map_root": None,
        "map_version": DEFAULT_MAP_VERSION,
        "splits": None,
        "log_split": "val",
        "max_workers": None,
    },
    "search": {
        "max_alternatives": 3,
        "min_lane_change_distance_m": 5.0,
        "divergence_max_distance_m": 50.0,
        "goal_time_s": 60.0,
        "min_goal_time_s": 0.0,
        "default_speed_mps": 10.0,
        "min_speed_mps": 2.0,
    },
    "io": {
        "tokens": None,
        "tokens_file": None,
        "output": "alternative_routes.jsonl",
        "append": False,
        "verbose": False,
    },
}

# Keyword arguments forwarded to ``search_alternative_routes`` (max_alternatives handled alongside).
SEARCH_KEYS = tuple(DEFAULTS["search"].keys())


def default_config_path() -> Optional[Path]:
    """Locate the parameter file to use when none is given explicitly.

    In order: the ``NUCONTROL_CONFIG`` environment variable, a ``params.yaml`` in the current
    working directory, then the ``params.yaml`` shipped with the package. Returns ``None`` if none
    of these exist (so the built-in defaults apply).
    """
    env = os.environ.get("NUCONTROL_CONFIG")
    if env:
        return Path(env)
    cwd = Path.cwd() / "params.yaml"
    if cwd.is_file():
        return cwd
    if _SHIPPED_CONFIG.is_file():
        return _SHIPPED_CONFIG
    return None


def resolve_config_path(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """The explicit path if given, otherwise the auto-discovered default (may be ``None``)."""
    if explicit is not None:
        return Path(explicit)
    return default_config_path()


def load_config(path: Optional[str | Path]) -> Dict[str, Any]:
    """Read a YAML parameter file into a dict; ``None`` path yields an empty config."""
    if path is None:
        return {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {path} must contain a YAML mapping at the top level."
        )
    return data


def section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Return a config section merged over its defaults."""
    return {**DEFAULTS[name], **(config.get(name) or {})}


def resolve(override: Any, sec: Dict[str, Any], key: str, default: Any) -> Any:
    """override (if not None) > config value (if present and not None) > default."""
    if override is not None:
        return override
    if key in sec and sec[key] is not None:
        return sec[key]
    return default


def make_loader(
    config: Optional[Dict[str, Any]] = None, **overrides: Any
) -> ScenarioLoader:
    """Build a :class:`ScenarioLoader` from the ``dataset`` section, honoring keyword overrides."""
    ds = section(config or {}, "dataset")
    return ScenarioLoader(
        data_root=resolve(overrides.get("data_root"), ds, "data_root", None),
        map_root=resolve(overrides.get("map_root"), ds, "map_root", None),
        map_version=resolve(
            overrides.get("map_version"), ds, "map_version", DEFAULT_MAP_VERSION
        ),
        include_splits=resolve(overrides.get("splits"), ds, "splits", None),
        log_split=resolve(overrides.get("log_split"), ds, "log_split", "val"),
        max_workers=resolve(overrides.get("max_workers"), ds, "max_workers", None),
    )


def search_kwargs(
    config: Optional[Dict[str, Any]] = None, **overrides: Any
) -> Dict[str, Any]:
    """Resolve the ``search`` section into kwargs for ``search_alternative_routes``."""
    se = section(config or {}, "search")
    return {
        k: resolve(overrides.get(k), se, k, DEFAULTS["search"][k]) for k in SEARCH_KEYS
    }
