"""Generate an exemplary alternative-route dataset (JSONL) for a hand-picked token.

Companion to ``test_route_visualization.py`` — same token, but instead of figures it produces the
actual dataset artifact the FlowDrive planner consumes: one JSONL row per alternative route, each
carrying the roadblock-id sequence, the new goal pose, and the meta tags. It then reads the file
back and validates every row, so it doubles as a round-trip check of the writer/reader and the
``search -> serialize`` path.

Requires a mounted nuPlan dataset; skips cleanly if it isn't available. Following nuPlan's own
convention, NUPLAN_DATA_ROOT is the dataset mount root (the loader finds the log DBs under
nuplan-v1.1/splits/trainval by default — the official split holding the val14 scenarios; set
NUCONTROL_SPLITS=test for test14, etc.):

    export NUPLAN_DATA_ROOT=/scratch/nuplan/dataset
    export NUPLAN_MAPS_ROOT=/scratch/nuplan/dataset/maps

Run as a test (writes the JSONL to $NUCONTROL_DATASET_OUT or a temp dir; set $NUCONTROL_CONFIG to
apply a params.yaml):

    pytest -s tests/test_generate_dataset.py

Or standalone, optionally driven by a parameter file:

    python tests/test_generate_dataset.py [TOKEN] [OUT_JSONL]
    python tests/test_generate_dataset.py --config params.yaml [TOKEN] [OUT_JSONL]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

# Same token as the visualization test: a clean left/straight/right split ~20 m ahead (val14).
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


def _generate(token: str, out_path: str, config_path: str | None = None) -> int:
    """Search alternatives for ``token`` and write JSONL rows, driven by the test config."""
    from nucontrol.config import load_config, make_loader, search_kwargs
    from nucontrol.route_search import search_alternative_routes
    from nucontrol.alternative_routes import write_alternative_routes

    config = load_config(_resolve_config(config_path))
    scenario = make_loader(config).load_scenario(token)
    routes = search_alternative_routes(scenario, **search_kwargs(config))
    return write_alternative_routes(routes, out_path)


@pytest.mark.skipif(
    not _data_available(), reason="nuPlan dataset not mounted (set NUPLAN_*_ROOT)"
)
def test_generate_dataset():
    from nucontrol.config import load_config, make_loader
    from nucontrol.alternative_routes import read_alternative_routes
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.environ.get(
        "NUCONTROL_DATASET_OUT", str(OUTPUT_DIR / "alternative_routes.jsonl")
    )
    n = _generate(DEFAULT_TOKEN, out_path)  # tests/test_params.yaml by default

    # The demo token diverges left/right off a straight-going original, so expect >= 1 row.
    assert n >= 1
    assert os.path.isfile(out_path) and os.path.getsize(out_path) > 0

    # Round-trip: read the file back and validate each row against the loaded scenario's map.
    scenario = make_loader(load_config(_resolve_config(None))).load_scenario(
        DEFAULT_TOKEN
    )
    map_api = scenario.map_api
    rows = list(read_alternative_routes(out_path))
    assert len(rows) == n

    print(f"\nWrote {n} alternative route(s) to {out_path}")
    for row in rows:
        assert row.token == DEFAULT_TOKEN
        assert len(row.route_ids) >= 1
        # Every roadblock id must resolve on the map (route is well-formed).
        for rid in row.route_ids:
            rb = map_api.get_map_object(
                rid, SemanticMapLayer.ROADBLOCK
            ) or map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            assert rb is not None, f"roadblock id {rid} does not resolve on the map"
        assert len(row.goal_position) == 3
        assert set(row.meta) >= {"divergence_distance_m", "goal_distance_m", "turn"}
        print(
            f"  {row.meta['turn']:>8}  {len(row.route_ids):>2} roadblocks  "
            f"goal_dist={row.meta['goal_distance_m']:.0f}m  "
            f"t={row.meta['est_travel_time_s']:.0f}s "
            f"({'reached' if row.meta['reached_goal_time'] else 'short'})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token", nargs="?", default=DEFAULT_TOKEN)
    parser.add_argument(
        "out_path", nargs="?", default=str(OUTPUT_DIR / "alternative_routes.jsonl")
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML parameter file (default: tests/test_params.yaml).",
    )
    args = parser.parse_args()
    os.makedirs(Path(args.out_path).parent, exist_ok=True)

    if not _data_available():
        sys.exit(
            "Set NUPLAN_DATA_ROOT and NUPLAN_MAPS_ROOT to a mounted nuPlan dataset first."
        )
    n = _generate(args.token, args.out_path, args.config)
    print(f"wrote {n} alternative route(s) to {args.out_path}")
