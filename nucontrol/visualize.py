"""Visualize a scenario's map and highlight its **route lanes** in blue.

Used to sanity-check that route extraction is correct: for a given scenario token, render one
figure per route (the original route plus each alternative), drawing the surrounding map in grey
and the lanes that belong to the route's roadblocks in blue, with the ego and goal marked.

The point of the "route lanes" view: a route is stored as an ordered list of *roadblock* ids, but
what the planner ultimately drives on are the *lanes* inside those roadblocks (a lane is on-route
iff its parent roadblock is on-route). Painting those lanes blue over the raw map is the most
direct way to eyeball whether the extracted roadblock chain is connected, starts at the ego, and
leads to the goal.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless (HPC / no display)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from nucontrol.route_search import (
    extract_original_route,
    route_end_pose,
    route_lanes,
    search_alternative_routes,
)

Pose = Tuple[float, float, float]

# Colors.
_MAP_FILL = "#e8e8e8"
_MAP_EDGE = "#c4c4c4"
_ROUTE_FILL = "#1f77b4"
_ROUTE_EDGE = "#0b3d66"
_CENTERLINE = "#0b3d66"


def _exterior_xy(polygon) -> Optional[Tuple[List[float], List[float]]]:
    """Extract exterior ring x/y from a shapely (multi)polygon, or None."""
    geom = polygon
    if getattr(geom, "geom_type", "") == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    try:
        xs, ys = geom.exterior.xy
        return list(xs), list(ys)
    except Exception:
        return None


def plot_route(
    scenario: NuPlanScenario,
    roadblock_ids: Sequence[str],
    goal: Optional[Pose],
    title: str,
    out_path: str,
    pad_m: float = 25.0,
) -> str:
    """Render the map and highlight ``roadblock_ids``' lanes in blue, auto-framed to the route.

    The view is fit to the extent of the route lanes (plus the ego and goal), so a long route is
    not clipped by a fixed window. The grey background map is queried to cover that same extent.

    Args:
        scenario: Loaded nuPlan scenario (for map + ego).
        roadblock_ids: Route roadblock ids whose interior lanes are painted blue.
        goal: Optional ``(x, y, heading)`` goal pose to mark with a star.
        title: Figure title.
        out_path: PNG path to write.
        pad_m: Margin added around the route extent.

    Returns:
        ``out_path``.
    """
    map_api = scenario.map_api
    ego = scenario.initial_ego_state.rear_axle

    # --- route lanes + collect extent. ---
    lanes = route_lanes(map_api, list(roadblock_ids))
    xs: List[float] = [ego.x]
    ys: List[float] = [ego.y]
    lane_xy = []
    for lane in lanes:
        xy = _exterior_xy(lane.polygon)
        if xy is not None:
            lane_xy.append(xy)
            xs.extend(xy[0])
            ys.extend(xy[1])
    if goal is not None:
        xs.append(goal[0])
        ys.append(goal[1])

    xmin, xmax = min(xs) - pad_m, max(xs) + pad_m
    ymin, ymax = min(ys) - pad_m, max(ys) + pad_m
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    view_radius = 0.5 * math.hypot(xmax - xmin, ymax - ymin) + pad_m

    fig, ax = plt.subplots(figsize=(11, 11))

    # --- background: all lanes / lane-connectors covering the view, in grey. ---
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
                    linewidth=0.5,
                    zorder=1,
                )
            )

    # --- route lanes: fill blue + draw centerline. ---
    for xy in lane_xy:
        ax.add_patch(
            MplPolygon(
                list(zip(*xy)),
                closed=True,
                facecolor=_ROUTE_FILL,
                edgecolor=_ROUTE_EDGE,
                linewidth=0.6,
                alpha=0.55,
                zorder=2,
            )
        )
    for lane in lanes:
        path = lane.baseline_path.discrete_path
        ax.plot(
            [p.x for p in path],
            [p.y for p in path],
            color=_CENTERLINE,
            linewidth=1.0,
            zorder=3,
        )

    # --- ego marker + heading arrow. ---
    ax.plot(
        ego.x,
        ego.y,
        "o",
        color="limegreen",
        markersize=12,
        markeredgecolor="black",
        zorder=5,
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
        zorder=5,
    )

    # --- goal marker. ---
    if goal is not None:
        ax.plot(
            goal[0],
            goal[1],
            "*",
            color="red",
            markersize=22,
            markeredgecolor="black",
            zorder=6,
            label="goal",
        )

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def visualize_token(
    scenario: NuPlanScenario,
    out_dir: str,
    *,
    max_alternatives: int = 3,
    **search_kwargs: object,
) -> List[str]:
    """Write one PNG for the original route and one per alternative route of ``scenario``.

    Extra keyword arguments (``min_lane_change_distance_m``, ``divergence_max_distance_m``,
    ``goal_time_s``, ``min_goal_time_s``, ``default_speed_mps``, ``min_speed_mps``) are forwarded to
    the route search, so a parameter file drives the figures exactly as it drives the generated
    dataset. Returns the written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    token = scenario.token
    written: List[str] = []

    # Original route; goal marker at the end of the extracted route (comparable to alternatives).
    # Forward only the parameters extract_original_route accepts.
    orig_kwargs = {
        k: search_kwargs[k]
        for k in ("goal_time_s", "default_speed_mps", "min_speed_mps")
        if k in search_kwargs
    }
    orig_ids = extract_original_route(scenario, **orig_kwargs)
    orig_goal: Optional[Pose] = route_end_pose(scenario.map_api, orig_ids)
    p = os.path.join(out_dir, f"{token}_route_original.png")
    plot_route(
        scenario,
        orig_ids,
        orig_goal,
        title=f"{token} — ORIGINAL route ({len(orig_ids)} roadblocks)",
        out_path=p,
    )
    written.append(p)

    # Alternatives.
    alts = search_alternative_routes(
        scenario, max_alternatives=max_alternatives, **search_kwargs
    )
    for i, alt in enumerate(alts):
        m = alt.meta
        p = os.path.join(out_dir, f"{token}_route_alt{i}_{m['turn']}.png")
        title = (
            f"{token} — ALT {i} [{m['turn']}]  "
            f"div@{m['divergence_distance_m']}m  "
            f"goal {m['goal_distance_m']}m / {m['est_travel_time_s']}s"
            f"{'' if m['reached_goal_time'] else ' (short)'}"
        )
        plot_route(scenario, alt.route_ids, alt.goal_position, title, p)
        written.append(p)

    return written
