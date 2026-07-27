"""
Prove the per-lane-change runway constraint (``min_lane_change_distance_m``) actually changes
the alternative-route search, and visualize the difference.

The constraint (route_search.search_alternative_routes) drops a divergence branch when the ego
would need more lane-change runway to reach it than the distance left to the junction: a branch
needing ``N`` lane changes is kept only if ``N * min_lane_change_distance_m <= distance to the
junction``. Same-lane branches (``N == 0``) are always kept.

The test runs a hand-picked showcase token (``SHOWCASE_TOKEN``, found by the ``_find_biting_token``
scan over ``val14-alternative-routes.jsonl``) twice, with everything equal *except* the constraint:

    * OFF:  ``min_lane_change_distance_m = 0``   (every in-range branch kept)
    * ON:   ``min_lane_change_distance_m = CONSTRAINT_M``

For this token the ego sits right at a junction 3.77 m ahead: OFF returns straight + left + right;
ON returns only the straight branch (the two turns need a lane change with no runway left and are
dropped). It writes a side-by-side comparison figure (OFF | ON) into ``tests/output/`` with the
dropped branches dashed/outlined, so the effect is visible. Run standalone with ``scan`` to
rediscover a biting token over the JSONL instead of using the hard-coded one.

Requires a mounted nuPlan dataset; skips cleanly if it isn't available.

Run as a test (writes PNGs under tests/output/lane_change_constraint):

    pytest -s tests/test_lane_change_constraint.py

Or standalone, optionally forcing a specific token and/or constraint value:

    python tests/test_lane_change_constraint.py [TOKEN] [CONSTRAINT_M]
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless (HPC / no display)
import matplotlib.pyplot as plt
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from nucontrol.route_search import (
    extract_original_route,
    route_lanes,
    search_alternative_routes,
)
from nucontrol.visualize import _MAP_EDGE, _MAP_FILL, _exterior_xy

# --- Test knobs. ---------------------------------------------------------------------------------
# Hand-picked showcase token (found by the _find_biting_token scan below over the val14 JSONL).
# The ego sits right at a junction 3.77 m ahead offering straight / left / right. With the
# constraint OFF all three are returned; with it ON only the straight branch (N=0, same lane)
# survives — the left and right turns need a lane change the ego has no runway left for and are
# dropped. Pass a different token on the CLI, or "scan" to rediscover one, to override.
SHOWCASE_TOKEN = "eca32130c66755c1"
# Runway (m) required per lane change when the constraint is ON. Chosen larger than the shipped
# default so a single close turn-pocket branch (needing >=1 lane change within a few tens of metres)
# is reliably dropped, making the effect easy to see.
CONSTRAINT_M = 15.0
# max_alternatives kept high so a dropped branch shows up as a real difference rather than being
# masked by the cap.
MAX_ALTS = 8
# How many candidate tokens to try before giving up (each is a scenario load, so keep it bounded).
MAX_CANDIDATES = 40

TEST_DIR = Path(__file__).resolve().parent
TEST_CONFIG = TEST_DIR / "test_params.yaml"
OUTPUT_DIR = TEST_DIR / "output" / "lane_change_constraint"
ROUTES_JSONL = TEST_DIR.parent / "val14-alternative-routes.jsonl"

# Distinct colors for overlaid alternative routes.
_ALT_COLORS = [
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
]
_ORIG_FILL = "#1f77b4"


def _data_available() -> bool:
    root = os.environ.get("NUPLAN_DATA_ROOT")
    maps = os.environ.get("NUPLAN_MAPS_ROOT")
    return bool(root) and bool(maps) and os.path.isdir(root) and os.path.isdir(maps)


def _candidate_tokens() -> List[str]:
    """Tokens from the JSONL, ordered by how likely the constraint is to bite.

    Priority: tokens whose stored alternatives are many (wide, multi-lane junction) and diverge
    close to the ego (small divergence distance) — that is where a branch needs a lane change with
    little runway. The JSONL values only *rank* candidates; the search is re-run fresh per token.
    """
    if not ROUTES_JSONL.is_file():
        return []
    per_token: Dict[str, List[float]] = defaultdict(list)
    with ROUTES_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            div = float(rec.get("meta", {}).get("divergence_distance_m", math.inf))
            per_token[rec["token"]].append(div)
    # Sort: more alternatives first, then nearer divergence.
    ranked = sorted(
        per_token.items(),
        key=lambda kv: (-len(kv[1]), min(kv[1])),
    )
    return [tok for tok, _ in ranked[:MAX_CANDIDATES]]


def _route_key(alt) -> Tuple[str, ...]:
    """Identity of an alternative route for set comparison (its roadblock-id sequence)."""
    return tuple(alt.route_ids)


def _find_biting_token(loader, base_kwargs) -> Optional[dict]:
    """Return the first candidate where turning the constraint ON drops a branch found with it OFF.

    Result dict carries the loaded scenario, both alternative lists, and the dropped branches.
    """
    for token in _candidate_tokens():
        try:
            scenario = loader.load_scenario(token)
        except Exception as exc:  # missing token / wrong split — just skip it.
            print(f"skip {token}: {exc}")
            continue

        off = search_alternative_routes(
            scenario,
            max_alternatives=MAX_ALTS,
            min_lane_change_distance_m=0.0,
            **base_kwargs,
        )
        on = search_alternative_routes(
            scenario,
            max_alternatives=MAX_ALTS,
            min_lane_change_distance_m=CONSTRAINT_M,
            **base_kwargs,
        )
        on_keys = {_route_key(a) for a in on}
        dropped = [a for a in off if _route_key(a) not in on_keys]
        print(
            f"{token}: OFF={len(off)} alt  ON={len(on)} alt  dropped={len(dropped)}"
            + ("  <-- constraint bites" if dropped else "")
        )
        if dropped:
            return {"scenario": scenario, "off": off, "on": on, "dropped": dropped}
    return None


def _draw_panel(ax, scenario, alternatives, dropped_keys, title, view) -> None:
    """Draw one panel: grey map, original route (blue), each alternative in a distinct color."""
    map_api = scenario.map_api
    ego = scenario.initial_ego_state.rear_axle
    (xmin, xmax, ymin, ymax, cx, cy, view_radius) = view

    # Background: all lanes / connectors covering the view, grey.
    proximal = map_api.get_proximal_map_objects(
        Point2D(cx, cy),
        view_radius,
        [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
    )
    for layer_objs in proximal.values():
        for obj in layer_objs:
            xy = _exterior_xy(obj.polygon)
            if xy is None:
                continue
            ax.add_patch(
                MplPolygon(
                    list(zip(*xy)),
                    closed=True,
                    facecolor=_MAP_FILL,
                    edgecolor=_MAP_EDGE,
                    linewidth=0.4,
                    zorder=1,
                )
            )

    # Original route (context), light blue.
    orig_ids = extract_original_route(scenario)
    for lane in route_lanes(map_api, orig_ids):
        xy = _exterior_xy(lane.polygon)
        if xy is not None:
            ax.add_patch(
                MplPolygon(
                    list(zip(*xy)),
                    closed=True,
                    facecolor=_ORIG_FILL,
                    edgecolor="none",
                    alpha=0.25,
                    zorder=2,
                )
            )

    # Alternatives, one color each; dropped-by-constraint ones get a dashed heavy outline + tag.
    handles = []
    for i, alt in enumerate(alternatives):
        color = _ALT_COLORS[i % len(_ALT_COLORS)]
        is_dropped = _route_key(alt) in dropped_keys
        for lane in route_lanes(map_api, alt.route_ids):
            xy = _exterior_xy(lane.polygon)
            if xy is None:
                continue
            ax.add_patch(
                MplPolygon(
                    list(zip(*xy)),
                    closed=True,
                    facecolor=color,
                    edgecolor="black" if is_dropped else color,
                    linewidth=1.8 if is_dropped else 0.6,
                    linestyle="--" if is_dropped else "-",
                    alpha=0.55,
                    zorder=3,
                )
            )
        gx, gy, _ = alt.goal_position
        ax.plot(
            gx, gy, "*", color=color, markersize=16, markeredgecolor="black", zorder=6
        )
        turn = alt.meta.get("turn", "?")
        div = alt.meta.get("divergence_distance_m", "?")
        label = f"alt{i}: {turn} @ {div}m" + ("  (DROPPED)" if is_dropped else "")
        handles.append(Line2D([0], [0], color=color, lw=6, alpha=0.7, label=label))

    # Ego marker + heading arrow.
    ax.plot(
        ego.x,
        ego.y,
        "o",
        color="limegreen",
        markersize=11,
        markeredgecolor="black",
        zorder=7,
        label="ego",
    )
    arrow = max(6.0, 0.02 * view_radius)
    ax.arrow(
        ego.x,
        ego.y,
        arrow * math.cos(ego.heading),
        arrow * math.sin(ego.heading),
        head_width=0.4 * arrow,
        head_length=0.4 * arrow,
        fc="limegreen",
        ec="black",
        zorder=7,
    )
    handles.append(Line2D([0], [0], marker="o", color="limegreen", lw=0, label="ego"))

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(handles=handles, loc="upper right", fontsize=8)


def _shared_view(scenario, variants, pad_m: float = 25.0):
    """Compute one map extent covering the ego + every route across all variants, so panels align."""
    map_api = scenario.map_api
    ego = scenario.initial_ego_state.rear_axle
    xs: List[float] = [ego.x]
    ys: List[float] = [ego.y]
    for alts in variants:
        for alt in alts:
            for lane in route_lanes(map_api, alt.route_ids):
                xy = _exterior_xy(lane.polygon)
                if xy is not None:
                    xs.extend(xy[0])
                    ys.extend(xy[1])
            gx, gy, _ = alt.goal_position
            xs.append(gx)
            ys.append(gy)
    xmin, xmax = min(xs) - pad_m, max(xs) + pad_m
    ymin, ymax = min(ys) - pad_m, max(ys) + pad_m
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    view_radius = 0.5 * math.hypot(xmax - xmin, ymax - ymin) + pad_m
    return (xmin, xmax, ymin, ymax, cx, cy, view_radius)


def _render_comparison(result: dict, out_dir: str) -> str:
    """Write the OFF|ON side-by-side comparison PNG. Returns its path."""
    os.makedirs(out_dir, exist_ok=True)
    scenario = result["scenario"]
    off, on, dropped = result["off"], result["on"], result["dropped"]
    dropped_keys = {_route_key(a) for a in dropped}
    token = scenario.token

    view = _shared_view(scenario, [off, on])
    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    _draw_panel(
        axes[0],
        scenario,
        off,
        dropped_keys,
        title=f"{token}\nconstraint OFF (min_lane_change_distance_m=0):"
        f" {len(off)} alternatives",
        view=view,
    )
    _draw_panel(
        axes[1],
        scenario,
        on,
        dropped_keys,
        title=f"{token}\nconstraint ON (min_lane_change_distance_m={CONSTRAINT_M}):"
        f" {len(on)} alternatives  ({len(dropped)} dropped)",
        view=view,
    )
    fig.suptitle(
        "Per-lane-change runway constraint: branches needing a lane change with too little "
        "runway (dashed, black outline) are dropped on the right.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(out_dir, f"{token}_lane_change_constraint_off_vs_on.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _run(explicit_token: Optional[str] = None) -> dict:
    """Find (or, if forced, use) a biting token, render the comparison, and return the result."""
    from nucontrol.config import load_config, make_loader, search_kwargs

    config = load_config(os.environ.get("NUCONTROL_CONFIG") or str(TEST_CONFIG))
    loader = make_loader(config)

    # Everything except the two knobs we vary (max_alternatives, min_lane_change_distance_m).
    base_kwargs = search_kwargs(config)
    base_kwargs.pop("max_alternatives", None)
    base_kwargs.pop("min_lane_change_distance_m", None)

    if explicit_token:
        scenario = loader.load_scenario(explicit_token)
        off = search_alternative_routes(
            scenario,
            max_alternatives=MAX_ALTS,
            min_lane_change_distance_m=0.0,
            **base_kwargs,
        )
        on = search_alternative_routes(
            scenario,
            max_alternatives=MAX_ALTS,
            min_lane_change_distance_m=CONSTRAINT_M,
            **base_kwargs,
        )
        on_keys = {_route_key(a) for a in on}
        dropped = [a for a in off if _route_key(a) not in on_keys]
        result = {"scenario": scenario, "off": off, "on": on, "dropped": dropped}
    else:
        result = _find_biting_token(loader, base_kwargs)

    if result is None:
        pytest.skip(
            f"no candidate among {MAX_CANDIDATES} tokens had a branch dropped by the "
            f"constraint at min_lane_change_distance_m={CONSTRAINT_M}"
        )

    out = _render_comparison(result, str(OUTPUT_DIR))
    result["comparison_png"] = out
    return result


@pytest.mark.skipif(
    not _data_available(), reason="nuPlan dataset not mounted (set NUPLAN_*_ROOT)"
)
def test_lane_change_constraint_changes_routes():
    # Deterministic: use the hand-picked showcase token where the constraint is known to bite.
    result = _run(SHOWCASE_TOKEN)

    # The constraint dropped at least one branch that the unconstrained search found.
    assert result["dropped"], "expected the constraint to drop a branch"
    assert len(result["on"]) < len(result["off"])

    # Each dropped branch is a lane-change branch the ego is too close to (N>=1 and
    # N*CONSTRAINT_M > divergence distance); it must be absent from the constrained result.
    on_keys = {_route_key(a) for a in result["on"]}
    for a in result["dropped"]:
        assert _route_key(a) not in on_keys

    # The comparison figure was written.
    png = result["comparison_png"]
    assert os.path.isfile(png) and os.path.getsize(png) > 0
    print("wrote comparison:", png)


if __name__ == "__main__":
    if not _data_available():
        sys.exit(
            "Set NUPLAN_DATA_ROOT and NUPLAN_MAPS_ROOT to a mounted nuPlan dataset first."
        )
    # Default to the showcase token; "scan" rediscovers one from the JSONL; else use the given token.
    arg = sys.argv[1] if len(sys.argv) > 1 else SHOWCASE_TOKEN
    token = None if arg == "scan" else arg
    if len(sys.argv) > 2:
        # Rebind the module-level constant so the search functions pick up the override.
        CONSTRAINT_M = float(sys.argv[2])
    res = _run(token)
    print(f"token={res['scenario'].token}")
    print(f"  OFF: {len(res['off'])} alternatives")
    print(f"  ON : {len(res['on'])} alternatives  ({len(res['dropped'])} dropped)")
    for a in res["dropped"]:
        print(
            f"    dropped: {a.meta.get('turn')} @ {a.meta.get('divergence_distance_m')}m"
        )
    print("  comparison:", res["comparison_png"])
