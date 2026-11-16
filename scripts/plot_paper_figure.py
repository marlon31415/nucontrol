"""Paper figure: one scene with the original route and all alternative routes overlaid.

The original route is painted light blue (as in the lane-change-constraint comparison figure), each
alternative in its own color, with a goal star per route and the ego footprint. Lanes shared by
several alternatives (but not by the original route) are filled with the first alternative's color
and hatched with the others', so a common prefix stays readable instead of turning into a muddy
overlap. Axes are hidden in favor of a scale bar, and the figure is written as both PDF (vector) and
PNG.

Requires a mounted nuPlan dataset (NUPLAN_DATA_ROOT / NUPLAN_MAPS_ROOT).

    python scripts/plot_paper_figure.py [TOKEN] [--config params.yaml] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42   # embed TrueType instead of Type 3
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif']  # .ttf only; Nimbus Roman is CFF/.otf and triggers a pdffonts type mismatch

matplotlib.use("Agg")  # headless (HPC / no display)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nucontrol.config import load_config, make_loader, search_kwargs  # noqa: E402
from nucontrol.route_search import (  # noqa: E402
    extract_original_route,
    route_end_pose,
    search_alternative_routes,
)
from nucontrol.visualize import _exterior_xy  # noqa: E402

# Pittsburgh grid scene with a clean right / straight / left split close to the ego (val14 split).
DEFAULT_TOKEN = "26c3cdb68bbd50e5"
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "output" / "paper_figure"

# Colors: same original-route blue and alternative palette as the lane-change-constraint figure.
_MAP_FILL = "#ececec"
_MAP_EDGE = "#cdcdcd"
_ORIG_COLOR = "#1f77b4"
_ORIG_ALPHA = 0.3
_ALT_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
_ALT_ALPHA = 0.6
_EGO_COLOR = "#222222"
# Ego chevron in vehicle frame (x forward, y left), in units of the marker size.
_EGO_CHEVRON = [(0.6, 0.0), (-0.4, 0.4), (-0.2, 0.0), (-0.4, -0.4)]


def _lanes_of(map_api, roadblock_id: str):
    """Interior lanes (or lane connectors) of one roadblock (or roadblock connector)."""
    rb = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
    if rb is None:
        rb = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
    return list(rb.interior_edges) if rb is not None else []


def _add_poly(ax, xy, **kwargs) -> None:
    ax.add_patch(MplPolygon(list(zip(*xy)), closed=True, **kwargs))


def _scale_bar(ax, xmin, xmax, ymin, ymax, fontsize: float) -> None:
    """Draw a round-numbered scale bar in the lower-left corner."""
    width = xmax - xmin
    length = next(
        (L for L in (10, 20, 25, 50, 100, 200) if L >= 0.15 * width), 200
    )
    x0 = xmin + 0.04 * width
    y0 = ymin + 0.04 * (ymax - ymin)
    ax.plot([x0, x0 + length], [y0, y0], color="black", lw=2.5, solid_capstyle="butt", zorder=10)
    ax.text(
        x0 + 0.5 * length,
        y0 + 0.012 * (ymax - ymin),
        f"{length} m",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        zorder=10,
    )


def plot_paper_figure(
    scenario,
    orig_ids: Sequence[str],
    alternatives: Sequence,
    out_stem: str,
    *,
    pad_m: float = 12.0,
    figsize: Tuple[float, float] = (5.0, 5.0),
    fontsize: float = 8.0,
    legend_loc: str = "below",
) -> List[str]:
    """Render the overlay figure to ``out_stem``.{pdf,png}. Returns the written paths."""
    map_api = scenario.map_api
    ego_state = scenario.initial_ego_state

    # Which routes each roadblock belongs to: -1 = original, i >= 0 = alternative i.
    owners: Dict[str, List[int]] = {}
    for rb in orig_ids:
        owners.setdefault(rb, []).append(-1)
    for i, alt in enumerate(alternatives):
        for rb in alt.route_roadblock_ids:
            owners.setdefault(rb, []).append(i)

    lane_xy: Dict[str, List] = {}
    xs: List[float] = []
    ys: List[float] = []
    for rb in owners:
        lane_xy[rb] = [xy for xy in (_exterior_xy(l.polygon) for l in _lanes_of(map_api, rb)) if xy]
        for xy in lane_xy[rb]:
            xs.extend(xy[0])
            ys.extend(xy[1])
    goals = [route_end_pose(map_api, list(orig_ids))] + [a.goal_position for a in alternatives]
    for g in goals:
        if g is not None:
            xs.append(g[0])
            ys.append(g[1])

    xmin, xmax = min(xs) - pad_m, max(xs) + pad_m
    ymin, ymax = min(ys) - pad_m, max(ys) + pad_m
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    view_radius = 0.5 * math.hypot(xmax - xmin, ymax - ymin) + pad_m

    fig, ax = plt.subplots(figsize=figsize)

    # Background map.
    proximal = map_api.get_proximal_map_objects(
        Point2D(cx, cy), view_radius, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    )
    for layer_objs in proximal.values():
        for obj in layer_objs:
            xy = _exterior_xy(obj.polygon)
            if xy is not None:
                _add_poly(ax, xy, facecolor=_MAP_FILL, edgecolor=_MAP_EDGE, linewidth=0.35, zorder=1)

    # Routes. The original route wins wherever it overlaps (the alternatives diverge from it).
    for rb, who in owners.items():
        for xy in lane_xy[rb]:
            if -1 in who:
                _add_poly(ax, xy, facecolor=_ORIG_COLOR, edgecolor="none", alpha=_ORIG_ALPHA, zorder=2)
                continue
            first = _ALT_COLORS[who[0] % len(_ALT_COLORS)]
            _add_poly(ax, xy, facecolor=first, edgecolor="none", alpha=_ALT_ALPHA, zorder=3)
            for j in who[1:]:
                _add_poly(
                    ax,
                    xy,
                    facecolor="none",
                    edgecolor=_ALT_COLORS[j % len(_ALT_COLORS)],
                    hatch="////",
                    linewidth=0,
                    alpha=0.9,
                    zorder=4,
                )

    # Goal stars.
    star_colors = [_ORIG_COLOR] + [_ALT_COLORS[i % len(_ALT_COLORS)] for i in range(len(alternatives))]
    for g, c in zip(goals, star_colors):
        if g is not None:
            ax.plot(g[0], g[1], "*", color=c, markersize=13, markeredgecolor="black", mew=0.8, zorder=8)

    # Ego: a heading chevron scaled to the view (the true footprint is a few pixels at this zoom).
    ego = ego_state.center
    size = 0.035 * max(xmax - xmin, ymax - ymin)
    c, s = math.cos(ego.heading), math.sin(ego.heading)
    shape = _EGO_CHEVRON
    chevron = (
        [ego.x + size * (u * c - v * s) for u, v in shape],
        [ego.y + size * (u * s + v * c) for u, v in shape],
    )
    _add_poly(ax, chevron, facecolor=_EGO_COLOR, edgecolor="white", linewidth=0.8, zorder=9)

    # Legend.
    handles = [Patch(facecolor=_ORIG_COLOR, alpha=_ORIG_ALPHA + 0.1, label="Original route")]
    for i, alt in enumerate(alternatives):
        turn = alt.meta.get("turn", "?")
        handles.append(
            Patch(
                facecolor=_ALT_COLORS[i % len(_ALT_COLORS)],
                alpha=_ALT_ALPHA,
                label=f"Alternative {i + 1} ({turn})",
            )
        )
    handles.append(
        Line2D(
            [0],
            [0],
            marker=_EGO_CHEVRON + [_EGO_CHEVRON[0]],  # same shape as on the map, heading right
            color=_EGO_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.6,
            lw=0,
            markersize=14,
            label="Ego vehicle",
        )
    )
    handles.append(
        Line2D(
            [0], [0], marker="*", color="white", markeredgecolor="black", lw=0, markersize=10, label="Goal"
        )
    )
    if legend_loc == "below":
        # Outside the map, so it can never hide a route or goal.
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=3,
            fontsize=fontsize,
            frameon=False,
            handlelength=1.6,
            columnspacing=1.2,
        )
    else:
        ax.legend(handles=handles, loc=legend_loc, fontsize=fontsize, framealpha=0.95)

    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    _scale_bar(ax, xmin, xmax, ymin, ymax, fontsize)
    fig.tight_layout(pad=0.2)

    written = []
    for ext in ("pdf", "png"):
        path = f"{out_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        written.append(path)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("token", nargs="?", default=DEFAULT_TOKEN)
    parser.add_argument("--config", default=str(REPO_ROOT / "params.yaml"), help="YAML parameter file.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--legend-loc",
        default="below",
        help='"below" (outside the map) or a matplotlib legend location such as "upper left".',
    )
    args = parser.parse_args()

    config = load_config(args.config)
    loader = make_loader(config)
    kwargs = search_kwargs(config)
    max_alternatives = kwargs.pop("max_alternatives")

    scenario = loader.load_scenario(args.token)
    orig_ids = extract_original_route(
        scenario, **{k: kwargs[k] for k in ("goal_time_s", "default_speed_mps", "min_speed_mps")}
    )
    alternatives = search_alternative_routes(scenario, max_alternatives=max_alternatives, **kwargs)
    for i, alt in enumerate(alternatives):
        m = alt.meta
        print(f"alt{i + 1}: {m['turn']} @ {m['divergence_distance_m']} m, goal {m['goal_distance_m']} m")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.token}_routes_overview")
    for p in plot_paper_figure(scenario, orig_ids, alternatives, stem, legend_loc=args.legend_loc):
        print("wrote", p)


if __name__ == "__main__":
    main()
