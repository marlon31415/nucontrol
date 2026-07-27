"""Visual sanity check for route-lane extraction.

Loads one hand-picked token and writes one figure per route — the original route plus each alternative
— with the route lanes highlighted in blue over the surrounding map.

Requires a mounted nuPlan dataset; skips cleanly if it isn't available. Following nuPlan's own
convention, NUPLAN_DATA_ROOT is the dataset mount root.

Run as a test (writes PNGs to $NUCONTROL_VIZ_OUT or ./route_viz; set $NUCONTROL_CONFIG to apply a
params.yaml):

    pytest -s tests/test_route_visualization.py

Or standalone, optionally driven by a parameter file:

    python tests/test_route_visualization.py [TOKEN] [OUT_DIR]
    python tests/test_route_visualization.py --config params.yaml [TOKEN] [OUT_DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

# A token with a clean left/straight/right split ~20 m ahead of the ego (val14 split).
DEFAULT_TOKEN = "8eb8410b1e385101"

TEST_DIR = Path(__file__).resolve().parent
# The tests use their own config, independent from the repo-root params.yaml the CLI uses.
TEST_CONFIG = TEST_DIR / "test_params.yaml"
# Test artifacts are written under the tests folder.
OUTPUT_DIR = TEST_DIR / "output"


def _data_available() -> bool:
    root = os.environ.get("NUPLAN_DATA_ROOT")
    maps = os.environ.get("NUPLAN_MAPS_ROOT")
    return bool(root) and bool(maps) and os.path.isdir(root) and os.path.isdir(maps)


def _resolve_config(explicit: str | None) -> str:
    """The test config to use: an explicit path > NUCONTROL_CONFIG env > tests/test_params.yaml."""
    return explicit or os.environ.get("NUCONTROL_CONFIG") or str(TEST_CONFIG)


def _run(token: str, out_dir: str, config_path: str | None = None) -> list:
    """Load ``token`` and render its route figures, driven by the test config (see _resolve_config)."""
    from nucontrol.config import load_config, make_loader, search_kwargs
    from nucontrol.visualize import visualize_token

    config = load_config(_resolve_config(config_path))
    loader = make_loader(config)
    scenario = loader.load_scenario(token)
    return visualize_token(scenario, out_dir, **search_kwargs(config))


@pytest.mark.skipif(
    not _data_available(), reason="nuPlan dataset not mounted (set NUPLAN_*_ROOT)"
)
def test_visualize_routes():
    out_dir = os.environ.get(
        "NUCONTROL_VIZ_OUT", str(OUTPUT_DIR / "route_viz" / DEFAULT_TOKEN)
    )
    paths = _run(DEFAULT_TOKEN, out_dir)  # tests/test_params.yaml by default

    # Original route plus at least one alternative for this demo token.
    assert len(paths) >= 2
    for p in paths:
        assert os.path.isfile(p) and os.path.getsize(p) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token", nargs="?", default=DEFAULT_TOKEN)
    parser.add_argument(
        "out_dir", nargs="?", default=str(OUTPUT_DIR / "route_viz" / DEFAULT_TOKEN)
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML parameter file (default: tests/test_params.yaml).",
    )
    args = parser.parse_args()

    if not _data_available():
        sys.exit(
            "Set NUPLAN_DATA_ROOT and NUPLAN_MAPS_ROOT to a mounted nuPlan dataset first."
        )
    for p in _run(args.token, args.out_dir, args.config):
        print("wrote", p)
