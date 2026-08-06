"""Search for alternative routes given a nuPlan scenario.

An **alternative route** is a path through the map's roadblock graph that diverges from the
scenario's original route at a junction close ahead of the ego (within
``divergence_max_distance_m``), then continues far enough that the ego needs roughly
``goal_time_s`` seconds of driving to reach a new goal position.

The original route is derived as a clean, connected roadblock chain straight from the expert
future trajectory via nuplan's :func:`get_roadblock_ids_from_trajectory` (no correction needed).
Roadblocks alternate between ``ROADBLOCK`` and ``ROADBLOCK_CONNECTOR`` segments;
``roadblock.outgoing_edges`` gives the next roadblock(s), so a junction with more than one
outgoing edge is a divergence point. A lane is "on route" iff its parent roadblock is on the
route, which is why routes are expressed at the roadblock level. Alternative routes are likewise
built by graph traversal, so they are connected by construction and need no post-hoc correction.

The result is a list of :class:`AlternativeRoute`, each carrying the full roadblock-id sequence
(ego -> goal) and the new goal pose. If the ego can only continue straight (no divergence within
the distance budget), an empty list is returned and the scenario should be skipped.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.utils import get_roadblock_ids_from_trajectory
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

# (x, y, heading) goal pose.
Pose = Tuple[float, float, float]


def _normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def extract_original_route(
    scenario: NuPlanScenario,
    *,
    goal_time_s: float = 60.0,
    default_speed_mps: float = 10.0,
    min_speed_mps: float = 2.0,
    max_roadblocks: int = 30,
    goal: Optional[Point2D] = None,
    goal_heading: Optional[float] = None,
    goal_radius_m: float = 5.0,
) -> List[str]:
    """Return the ego's original route as a clean, connected list of roadblock ids.

    The route is anchored at the roadblock the ego currently sits on (found from a single-point
    trajectory query, which is robust even at a standstill) and walked forward through the
    roadblock graph up to a ~``goal_time_s`` driving budget. At each junction the branch that lies
    on the scenario's stored **mission route** is preferred (straightest among ties); if the
    mission route has run out, the straightest continuation is taken.

    This is deliberately *not* taken from the expert's future trajectory alone: when the ego is
    stopped (e.g. waiting at a light) that trajectory covers a single roadblock, which both hides
    the intended route and — worse — leaves the on-route branch unknown, so the alternative search
    could not tell a genuine alternative from the original. Laying the route out along the stored
    mission with speed-limit timing fixes both. The result is connected and ego-aligned by
    construction, so it needs no correction. This is the route the alternative search diverges from.

    If ``goal`` is given, the walk targets actually **reaching** the goal's roadblock (resolved by
    proximity, within ``goal_radius_m``) instead of stopping purely on the ``goal_time_s`` time
    budget, which can otherwise cut the route off before a goal that's genuinely on it — and,
    unlike the plain time-budgeted walk, is guided by real graph distance rather than the stored
    route or heading alone:

    - ``goal_heading`` (recommended whenever available) rejects candidate roadblocks near ``goal``
      whose travel direction differs by more than 90 degrees. Without it, a location with two
      roadblocks representing opposite directions close together (e.g. both carriageways of a
      divided road) can resolve to the wrong (oncoming) one from a sub-meter difference in the
      query point — which then sends the whole walk chasing an unreachable or irrelevant target.

    - A bounded backward search from the goal (via ``incoming_edges``) computes each reachable
      roadblock's hop-distance to it. This runs **whenever a goal is given**, regardless of whether
      the goal's roadblock happens to be in the scenario's stored route: the stored route covers
      the whole scenario log and can legitimately be far longer than necessary to reach this
      particular goal (e.g. it continues on a big loop well beyond the near-term horizon), so
      on-route membership alone is not a reliable proxy for "the efficient way to get there".
    - At each fork, the forward walk picks whichever candidate has the *smallest* verified
      hop-distance to the goal, if any candidate has one — always making real, confirmed progress
      instead of blindly following the stored route's full length or guessing by heading. Only
      when no candidate is covered by the backward search (budget exhausted, or a genuine dead
      branch) does it fall back to the stored-route/heading preference used when no goal is given.
    - If ``goal`` can't be resolved to any roadblock within ``goal_radius_m`` (e.g. off-map), or no
      connecting path is found within ``max_roadblocks``, this falls back to the plain
      ``goal_time_s``-bounded walk (dead end via ``outgoing_edges`` breaks the loop as before).
    """
    map_api: AbstractMap = scenario.map_api
    ego_state = scenario.initial_ego_state
    ego_point = ego_state.rear_axle.point

    stored = [rid for rid in scenario.get_route_roadblock_ids() if rid]
    stored_set = set(stored)

    # Start roadblock: the block the ego actually sits on RIGHT NOW. This must be resolved by
    # proximity+heading (see _ego_current_roadblock), not by the stored route's first id: the
    # stored mission route begins where the route was originally laid out, which can be several
    # roadblocks BEHIND the ego, so anchoring there would search alternatives from a lane the ego
    # has already passed. The single-point trajectory query is only a fallback because it returns
    # nothing when the ego is stopped on a connector (a standstill at a junction).
    rb = _ego_current_roadblock(map_api, ego_state, stored_set)
    if rb is None:
        start_ids = get_roadblock_ids_from_trajectory(map_api, [ego_state])
        start_id = start_ids[0] if start_ids else (stored[0] if stored else None)
        rb = _get_roadblock(map_api, start_id) if start_id else None
    if rb is None:
        return list(stored)

    goal_roadblock_id = (
        _roadblock_id_near_point(map_api, goal, goal_radius_m, heading=goal_heading)
        if goal is not None
        else None
    )
    # Computed whenever a goal is given (not only when it's off the stored route — see docstring):
    # the stored route can be much longer than necessary to reach this specific goal, so on-route
    # membership alone can't be trusted to guide the walk efficiently toward it.
    goal_distances: Optional[Dict[str, int]] = None
    if goal_roadblock_id is not None:
        goal_distances = _backward_distances_to_goal(
            map_api, goal_roadblock_id, max_roadblocks=max_roadblocks
        )

    route_ids = [rb.id]
    visited = {rb.id}
    prev_lane = _nearest_lane(rb, ego_point)

    if goal_roadblock_id is not None and rb.id == goal_roadblock_id:
        return route_ids

    # Time already used up on the ego's current block (from the ego's arc position to its end).
    cum_time = 0.0
    if prev_lane is not None:
        covered = prev_lane.baseline_path.get_nearest_arc_length_from_position(
            ego_point
        )
        remaining = max(prev_lane.baseline_path.length - covered, 0.0)
        cum_time += remaining / _lane_speed(prev_lane, default_speed_mps, min_speed_mps)

    while len(route_ids) < max_roadblocks:
        # Without a goal, this reproduces the original goal_time_s-bounded walk exactly. With a
        # goal, the time budget no longer stops the walk early — it now runs until the goal's
        # roadblock is reached or the graph genuinely dead-ends (or the hard max_roadblocks cap).
        if goal_roadblock_id is None and cum_time >= goal_time_s:
            break
        candidates = [e for e in rb.outgoing_edges if e.id not in visited]
        if not candidates:
            break

        goal_directed = (
            [e for e in candidates if e.id in goal_distances]
            if goal_distances is not None
            else []
        )
        if goal_directed:
            # Verified progress toward the goal beats both the stored route (which can be far
            # longer than needed for this goal) and a heading guess: always take the candidate
            # with the smallest confirmed remaining hop-distance.
            nxt = min(goal_directed, key=lambda e: goal_distances[e.id])
        else:
            on_route = [e for e in candidates if e.id in stored_set]
            pool = on_route or candidates
            ref_heading = (
                prev_lane.baseline_path.discrete_path[-1].heading
                if prev_lane is not None
                else None
            )
            if ref_heading is None:
                nxt = pool[0]
            else:
                nxt = min(
                    pool, key=lambda e: _branch_misalignment(e, prev_lane, ref_heading)
                )
        lane = _straightest_lane(nxt, prev_lane)
        if lane is None:
            break
        route_ids.append(nxt.id)
        visited.add(nxt.id)
        cum_time += lane.baseline_path.length / _lane_speed(
            lane, default_speed_mps, min_speed_mps
        )
        rb, prev_lane = nxt, lane

        if nxt.id == goal_roadblock_id:
            break

    return route_ids


def route_end_pose(
    map_api: AbstractMap, roadblock_ids: List[str]
) -> Optional[Tuple[float, float, float]]:
    """Return an ``(x, y, heading)`` pose at the end of a route (its last lane's last point).

    Walks the roadblocks picking the straightest connected lane at each step, so the endpoint is
    a sensible "where does this route lead" marker for visualization.
    """
    prev_lane: Optional[LaneGraphEdgeMapObject] = None
    for rid in roadblock_ids:
        rb = _get_roadblock(map_api, rid)
        if rb is None:
            continue
        lane = _straightest_lane(rb, prev_lane)
        if lane is None:
            break
        prev_lane = lane
    if prev_lane is None:
        return None
    p = prev_lane.baseline_path.discrete_path[-1]
    return (p.x, p.y, p.heading)


def route_lanes(
    map_api: AbstractMap, roadblock_ids: List[str]
) -> List[LaneGraphEdgeMapObject]:
    """Expand a list of route roadblock ids into their interior **route lanes**.

    A route is expressed at the roadblock level; a lane is "on route" iff its parent roadblock is
    on the route. This resolves each roadblock id and returns all of its interior lanes, in route
    order (roadblocks that don't resolve are skipped).
    """
    lanes: List[LaneGraphEdgeMapObject] = []
    for rid in roadblock_ids:
        rb = _get_roadblock(map_api, rid)
        if rb is not None:
            lanes.extend(rb.interior_edges)
    return lanes


def search_alternative_routes(
    scenario: NuPlanScenario,
    max_alternatives: int = 3,
    *,
    min_lane_change_distance_m: float = 5.0,
    divergence_max_distance_m: float = 50.0,
    goal_time_s: float = 60.0,
    min_goal_time_s: float = 0.0,
    default_speed_mps: float = 10.0,
    min_speed_mps: float = 2.0,
) -> List["AlternativeRoute"]:
    """Find alternative routes for a scenario as a small route tree.

    All non-original branches at the **first** divergence junction ahead of the ego are returned
    (level 1) — every distinct option there is kept, even beyond ``max_alternatives``. If the first
    junction yields fewer than ``max_alternatives``, the result is topped up (level 2) from the next
    junction along each already-found alternative, in order, until the cap is reached. So the count
    is ``max(#first-junction options, min(max_alternatives, #reachable))``: more than
    ``max_alternatives`` only when the first junction alone provides them.

    Args:
        scenario: The loaded nuPlan scenario (ego states, map API, mission goal, route).
        max_alternatives: Target/cap for the total number of alternatives. First-junction options
            are always all kept even if they exceed this; it only bounds the level-2 top-up.
        min_lane_change_distance_m: Runway (m) required **per lane change**, applied per branch.
            Reaching a branch may need N lane changes (N=0 when the ego's current lane already feeds
            into it, e.g. go straight vs. turn right from a shared lane); the branch is kept only if
            ``N * min_lane_change_distance_m`` <= the distance to the junction. So same-lane branches
            (N=0) are always allowed, a single lane change needs this much room, and a branch that
            would take several lane changes is dropped unless the junction is far enough ahead to
            fit them. If every branch at a junction is dropped, the search moves to the next
            junction. 0 disables the floor (every branch at the nearest in-range junction is kept).
        divergence_max_distance_m: Soft upper bound (from 0m) on how far ahead of the ego the
            first divergence junction may be for alternatives to be searched.
        goal_time_s: Soft target driving time from the ego to the alternative goal (~1 min). If a
            branch dead-ends before this, the alternative is still emitted, with the achievable
            ``goal_distance_m`` / ``est_travel_time_s`` recorded in its meta.
        min_goal_time_s: Optional floor (default 0 = off) to drop trivially short spur branches
            that reach less than this many seconds of driving.
        default_speed_mps: Fallback speed for lanes with no speed limit, for time estimates.
        min_speed_mps: Lower clamp on lane speed to avoid division blow-ups on 0-limit lanes.

    Returns:
        Up to ``max_alternatives`` :class:`AlternativeRoute` objects (empty if the ego can only
        continue along the original route).
    """
    # Imported here to keep the module importable without the package installed as a whole.
    from .alternative_routes import AlternativeRoute

    map_api: AbstractMap = scenario.map_api
    ego_state = scenario.initial_ego_state
    ego_pose: StateSE2 = ego_state.rear_axle
    ego_point = ego_pose.point

    # Build the ego's original route as a clean, connected roadblock chain straight from the
    # expert future trajectory (nuplan's own extractor -> already connected, ego-aligned, no
    # correction needed). This defines both the ego's starting roadblock and the branch the
    # expert takes at each junction (the branch alternatives must differ from).
    route = extract_original_route(
        scenario,
        goal_time_s=goal_time_s,
        default_speed_mps=default_speed_mps,
        min_speed_mps=min_speed_mps,
    )
    if not route:
        return []

    # Resolve to objects, preserving order and dropping any that don't resolve.
    route_dict = {rid: _get_roadblock(map_api, rid) for rid in route}
    route_dict = {rid: rb for rid, rb in route_dict.items() if rb is not None}
    if not route_dict:
        return []
    route = list(route_dict.keys())
    starting_block = route_dict[route[0]]

    # How far along the starting roadblock the ego already is (arc length on its nearest lane).
    start_lane = _nearest_lane(starting_block, ego_point)
    ego_covered_m = (
        start_lane.baseline_path.get_nearest_arc_length_from_position(ego_point)
        if start_lane is not None
        else 0.0
    )

    # --- Find the FIRST divergence junction along the original route (nearest the ego, within the
    # distance budget). Every alternative branches from here; if there are too few, we top up from
    # the next junction along one of those branches (a small route tree, not a full enumeration). ---
    first_div: Optional[_Divergence] = None
    cum_dist = 0.0
    cum_time = 0.0
    prev_lane: Optional[LaneGraphEdgeMapObject] = None
    for i, rb_id in enumerate(route):
        rb = route_dict.get(rb_id) or _get_roadblock(map_api, rb_id)
        if rb is None:
            break
        lane = (
            start_lane
            if i == 0 and start_lane is not None
            else _straightest_lane(rb, prev_lane)
        )
        if lane is None:
            break
        block_len = lane.baseline_path.length
        remaining = block_len - ego_covered_m if i == 0 else block_len
        remaining = max(remaining, 0.0)
        block_time = remaining / _lane_speed(lane, default_speed_mps, min_speed_mps)

        cum_dist_end = cum_dist + remaining
        cum_time_end = cum_time + block_time
        next_id = route[i + 1] if i + 1 < len(route) else None
        alt_branches = [e for e in rb.outgoing_edges if e.id != next_id]

        if alt_branches and cum_dist_end <= divergence_max_distance_m:
            # Lane-change runway, applied per branch. Reaching a branch may require N lane changes
            # (N=0 when the ego's current lane already feeds into it, e.g. straight vs. turn right
            # from a shared lane). Each lane change needs ~min_lane_change_distance_m of runway, so a
            # branch is admissible only if N * min_lane_change_distance_m <= distance to the
            # junction. This drops turns/lane changes the ego is too close to still execute (a
            # single one near the junction, or several that would need more room than is left),
            # while always keeping same-lane branches (N=0) regardless of distance.
            branches = [
                b
                for b in alt_branches
                if (n := _lane_changes_to_branch(lane, rb, b)) is not None
                and n * min_lane_change_distance_m <= cum_dist_end
            ]
            if branches:
                first_div = _Divergence(
                    prefix_ids=route[: i + 1],
                    prefix_time_s=cum_time_end,
                    junction_lane=lane,
                    on_route_next_id=next_id,
                    branches=branches,
                    distance_m=cum_dist_end,
                )
                break

        cum_dist, cum_time = cum_dist_end, cum_time_end
        prev_lane = lane
        if cum_dist > divergence_max_distance_m:
            break

    if first_div is None:
        return []

    alternatives: List[AlternativeRoute] = []
    seen_finals: set[str] = set()

    # --- Level 1: one alternative per branch at the first junction. ALL are kept, even beyond
    # max_alternatives, since every distinct option at the first junction is wanted. ---
    level1: List[Tuple[List[str], List[_Step]]] = []
    for branch in first_div.branches:
        steps = _straightest_extension(
            branch,
            start_time_s=first_div.prefix_time_s,
            start_dist_m=first_div.distance_m,
            visited=set(first_div.prefix_ids),
            goal_time_s=goal_time_s,
            default_speed_mps=default_speed_mps,
            min_speed_mps=min_speed_mps,
        )
        alt = _build_alternative(
            token=scenario.token,
            prefix_ids=first_div.prefix_ids,
            junction_lane=first_div.junction_lane,
            branch=branch,
            steps=steps,
            divergence_distance_m=first_div.distance_m,
            goal_time_s=goal_time_s,
            default_speed_mps=default_speed_mps,
            min_speed_mps=min_speed_mps,
            min_goal_time_s=min_goal_time_s,
        )
        if alt is None:
            continue
        alternatives.append(alt)
        seen_finals.add(alt.route_ids[-1])
        level1.append((first_div.prefix_ids, steps))

    # --- Level 2 (fill only): if the first junction gave fewer than max_alternatives, top up from
    # the next junction along each already-found alternative, in order, until the cap is reached.
    # This never runs when the first junction alone already meets/exceeds max_alternatives. ---
    if len(alternatives) < max_alternatives:
        for prefix_ids, steps in level1:
            if len(alternatives) >= max_alternatives:
                break
            fork = _first_fork(steps, blocked_ids=set(prefix_ids))
            if fork is None:
                continue
            idx, junction, fork_branches = fork
            # Shared path from the ego up to and including the second-junction roadblock.
            j2_prefix = list(prefix_ids) + [s.rb.id for s in steps[: idx + 1]]
            j2_visited = set(j2_prefix)
            for fbranch in fork_branches:
                if len(alternatives) >= max_alternatives:
                    break
                steps2 = _straightest_extension(
                    fbranch,
                    start_time_s=junction.end_time,
                    start_dist_m=junction.end_dist,
                    visited=j2_visited,
                    goal_time_s=goal_time_s,
                    default_speed_mps=default_speed_mps,
                    min_speed_mps=min_speed_mps,
                )
                alt2 = _build_alternative(
                    token=scenario.token,
                    prefix_ids=j2_prefix,
                    junction_lane=junction.lane,
                    branch=fbranch,
                    steps=steps2,
                    divergence_distance_m=junction.end_dist,
                    goal_time_s=goal_time_s,
                    default_speed_mps=default_speed_mps,
                    min_speed_mps=min_speed_mps,
                    min_goal_time_s=min_goal_time_s,
                )
                if alt2 is None or alt2.route_ids[-1] in seen_finals:
                    continue
                alternatives.append(alt2)
                seen_finals.add(alt2.route_ids[-1])

    return alternatives


# --------------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------------


class _Divergence:
    """A junction where the ego could leave the original route."""

    __slots__ = (
        "prefix_ids",
        "prefix_time_s",
        "junction_lane",
        "on_route_next_id",
        "branches",
        "distance_m",
    )

    def __init__(
        self,
        prefix_ids: List[str],
        prefix_time_s: float,
        junction_lane: LaneGraphEdgeMapObject,
        on_route_next_id: Optional[str],
        branches: List[RoadBlockGraphEdgeMapObject],
        distance_m: float,
    ) -> None:
        self.prefix_ids = prefix_ids
        self.prefix_time_s = prefix_time_s
        self.junction_lane = junction_lane
        self.on_route_next_id = on_route_next_id
        self.branches = branches
        self.distance_m = distance_m


def _get_roadblock(
    map_api: AbstractMap, rb_id: str
) -> Optional[RoadBlockGraphEdgeMapObject]:
    """Resolve a roadblock or roadblock-connector by id."""
    rb = map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK)
    return rb or map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)


def _roadblock_id_near_point(
    map_api: AbstractMap,
    point: Point2D,
    radius_m: float,
    heading: Optional[float] = None,
    heading_thresh: float = math.pi / 2,
) -> Optional[str]:
    """Id of the roadblock/connector whose nearest interior-lane point is closest to ``point``.

    If ``heading`` is given, a candidate is only considered when its nearest-point lane heading is
    within ``heading_thresh`` (default 90 degrees) of it. Without this, plain proximity can't tell
    apart two roadblocks representing opposite travel directions at nearly the same location (e.g.
    the two carriageways of a divided road) — a sub-meter perturbation in ``point`` (such as the
    small shift between a raw trajectory point and its lane-projected counterpart) can then flip
    which one gets picked, silently resolving to the oncoming roadblock instead of the correct one.

    Returns ``None`` if nothing within ``radius_m`` matches (e.g. an off-map or badly mismapped
    goal, or every nearby candidate fails the heading check).
    """
    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    proximal = map_api.get_proximal_map_objects(
        point=point, radius=radius_m, layers=layers
    )
    candidates = (
        proximal[SemanticMapLayer.ROADBLOCK]
        + proximal[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )
    if not candidates:
        return None

    best_id: Optional[str] = None
    best_dist = math.inf
    for rb in candidates:
        for lane in rb.interior_edges:
            path = lane.baseline_path.discrete_path
            if not path:
                continue
            nearest = min(
                path, key=lambda p: (p.x - point.x) ** 2 + (p.y - point.y) ** 2
            )
            if (
                heading is not None
                and abs(_normalize_angle(nearest.heading - heading)) > heading_thresh
            ):
                continue
            dist = math.hypot(nearest.x - point.x, nearest.y - point.y)
            if dist < best_dist:
                best_dist, best_id = dist, rb.id
    return best_id


def _backward_distances_to_goal(
    map_api: AbstractMap,
    goal_roadblock_id: str,
    max_roadblocks: int = 30,
) -> Dict[str, int]:
    """Hop-distance to ``goal_roadblock_id`` (via forward/``outgoing_edges`` travel) per roadblock.

    Found by a bounded BFS *backward* from the goal over ``incoming_edges`` — cheaper than
    searching forward from every fork candidate, since it explores the graph once regardless of
    how many forks the eventual forward walk passes through. The returned distances let the
    forward walk always pick real, verified progress toward the goal (the neighbor with the
    smallest distance) instead of guessing by heading or stored-route membership alone. Bounded by
    ``max_roadblocks`` (same budget as the forward walk) so a goal in a huge or badly-connected
    local graph can't blow up the search; a goal with no path within that budget simply yields an
    incomplete map, so the forward walk falls back to its heading-based guess wherever a fork
    isn't covered.
    """
    goal_rb = _get_roadblock(map_api, goal_roadblock_id)
    if goal_rb is None:
        return {}

    distances = {goal_roadblock_id: 0}
    frontier = [goal_rb]
    depth = 0
    while frontier and len(distances) < max_roadblocks:
        depth += 1
        next_frontier = []
        for rb in frontier:
            for pred in getattr(rb, "incoming_edges", None) or []:
                if pred.id not in distances:
                    distances[pred.id] = depth
                    next_frontier.append(pred)
        frontier = next_frontier
    return distances


def _ego_current_roadblock(
    map_api: AbstractMap,
    ego_state,
    stored_set: set,
    *,
    radius_m: float = 2.0,
    heading_thresh: float = math.pi / 4,
    displacement_thresh: float = 3.0,
) -> Optional[RoadBlockGraphEdgeMapObject]:
    """Resolve the roadblock/connector the ego currently occupies, robust to standstills.

    Mirrors nuplan/flow_drive's ``get_current_roadblock_candidates``: among the roadblocks and
    connectors near the ego, keep those whose closest interior-lane point is both near the ego
    (< ``displacement_thresh`` m) and heading-aligned with it (< ``heading_thresh`` rad), then take
    the nearest — **preferring one on the stored mission route**. The heading test rejects an
    overlapping opposite-direction block, and preferring an on-route block plus the distance test
    pins the anchor to the ego's *current* block rather than one it has already driven through.
    Returns ``None`` only if nothing plausible is nearby (callers then fall back).
    """
    ego_pose: StateSE2 = ego_state.rear_axle
    ego_point = ego_pose.point
    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]

    proximal = map_api.get_proximal_map_objects(
        point=ego_point, radius=radius_m, layers=layers
    )
    candidates = (
        proximal[SemanticMapLayer.ROADBLOCK]
        + proximal[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )
    if not candidates:
        for layer in layers:
            rid, _ = map_api.get_distance_to_nearest_map_object(
                point=ego_point, layer=layer
            )
            rb = map_api.get_map_object(rid, layer) if rid else None
            if rb is not None:
                candidates.append(rb)

    best_on_route: Optional[RoadBlockGraphEdgeMapObject] = None
    best_on_route_disp = math.inf
    best_any: Optional[RoadBlockGraphEdgeMapObject] = None
    best_any_disp = math.inf
    for rb in candidates:
        disp, head = math.inf, math.inf
        for lane in rb.interior_edges:
            path = lane.baseline_path.discrete_path
            j = min(
                range(len(path)),
                key=lambda k: (path[k].x - ego_point.x) ** 2
                + (path[k].y - ego_point.y) ** 2,
            )
            d = math.hypot(path[j].x - ego_point.x, path[j].y - ego_point.y)
            if d < disp:
                disp = d
                head = abs(_normalize_angle(path[j].heading - ego_pose.heading))
        if disp >= displacement_thresh or head >= heading_thresh:
            continue
        if rb.id in stored_set:
            if disp < best_on_route_disp:
                best_on_route, best_on_route_disp = rb, disp
        elif disp < best_any_disp:
            best_any, best_any_disp = rb, disp

    return best_on_route if best_on_route is not None else best_any


def _lane_index(lane: LaneGraphEdgeMapObject) -> Optional[int]:
    """The lane's 1-indexed lateral slot within its roadblock, or None if it has none.

    Only proper lanes carry a lateral index; lane-connectors (interior edges of a roadblock
    connector) do not, so this returns None for them.
    """
    idx = getattr(lane, "index", None)
    return int(idx) if isinstance(idx, int) else None


def _lane_changes_to_branch(
    lane: LaneGraphEdgeMapObject,
    rb: RoadBlockGraphEdgeMapObject,
    branch: RoadBlockGraphEdgeMapObject,
) -> Optional[int]:
    """Number of lane changes for the ego (on ``lane``) to reach ``branch`` from roadblock ``rb``.

    An *entry lane* is an interior lane of ``rb`` that feeds directly into ``branch`` (one of its
    successor edges is an interior lane of the branch). The ego must be in an entry lane before the
    junction, so the count is the lateral lane-index distance from ``lane`` to the nearest entry
    lane — 0 if ``lane`` is itself an entry lane. Returns None if ``branch`` is not reachable from
    ``rb`` at all, or the lateral distance can't be determined (e.g. ``rb`` is a roadblock-connector
    whose interior connectors carry no lane index — inside a junction no lane change is possible).
    """
    branch_lane_ids = {e.id for e in branch.interior_edges}
    entry_lanes = [
        ln
        for ln in rb.interior_edges
        if any(e.id in branch_lane_ids for e in ln.outgoing_edges)
    ]
    if not entry_lanes:
        return None
    if any(ln.id == lane.id for ln in entry_lanes):
        return 0
    ego_idx = _lane_index(lane)
    entry_idxs = [j for j in (_lane_index(ln) for ln in entry_lanes) if j is not None]
    if ego_idx is None or not entry_idxs:
        return None
    return min(abs(ego_idx - j) for j in entry_idxs)


def _lane_speed(
    lane: LaneGraphEdgeMapObject, default_speed_mps: float, min_speed_mps: float
) -> float:
    speed = lane.speed_limit_mps or default_speed_mps
    return max(speed, min_speed_mps)


def _nearest_lane(
    roadblock: RoadBlockGraphEdgeMapObject, point: Point2D
) -> Optional[LaneGraphEdgeMapObject]:
    """Interior lane of ``roadblock`` whose baseline passes closest to ``point``."""
    best_lane, best_dist = None, math.inf
    for lane in roadblock.interior_edges:
        path = lane.baseline_path.discrete_path
        dist = min((p.x - point.x) ** 2 + (p.y - point.y) ** 2 for p in path)
        if dist < best_dist:
            best_lane, best_dist = lane, dist
    return best_lane


def _straightest_lane(
    roadblock: RoadBlockGraphEdgeMapObject,
    prev_lane: Optional[LaneGraphEdgeMapObject],
) -> Optional[LaneGraphEdgeMapObject]:
    """Interior lane best continuing ``prev_lane``'s heading (or the first lane)."""
    lanes = roadblock.interior_edges
    if not lanes:
        return None
    if prev_lane is None:
        return lanes[0]
    ref_heading = prev_lane.baseline_path.discrete_path[-1].heading
    return min(
        lanes,
        key=lambda ln: abs(
            _normalize_angle(ln.baseline_path.discrete_path[0].heading - ref_heading)
        ),
    )


class _Step:
    """One roadblock along an extended route, with the lane driven and the entry/exit time+dist.

    ``enter_time``/``end_time`` are measured from the ego; ``enter_dist``/``end_dist`` are the
    travel distance accumulated from the divergence junction the extension started at.
    """

    __slots__ = ("rb", "lane", "enter_time", "end_time", "enter_dist", "end_dist")

    def __init__(
        self,
        rb: RoadBlockGraphEdgeMapObject,
        lane: LaneGraphEdgeMapObject,
        enter_time: float,
        end_time: float,
        enter_dist: float,
        end_dist: float,
    ) -> None:
        self.rb = rb
        self.lane = lane
        self.enter_time = enter_time
        self.end_time = end_time
        self.enter_dist = enter_dist
        self.end_dist = end_dist


def _straightest_extension(
    branch: RoadBlockGraphEdgeMapObject,
    start_time_s: float,
    start_dist_m: float,
    visited: set,
    goal_time_s: float,
    default_speed_mps: float,
    min_speed_mps: float,
    max_depth: int = 300,
) -> List["_Step"]:
    """Extend from ``branch`` to a ~``goal_time_s`` route via straightest-first DFS.

    A greedy walk can dead-end into a short spur even when a long route exists; this DFS tries the
    straightest continuation first but backtracks on dead-ends, returning the first path that
    reaches ``goal_time_s`` (or the deepest one found if the local graph dead-ends earlier), as a
    list of :class:`_Step` from ``branch`` onward.
    """
    best_steps: List[_Step] = []
    best_end_time = start_time_s

    def dfs(
        rb: RoadBlockGraphEdgeMapObject,
        prev_lane: Optional[LaneGraphEdgeMapObject],
        cum_time: float,
        cum_dist: float,
        steps: List[_Step],
        local_visited: set,
    ) -> bool:
        nonlocal best_steps, best_end_time
        lane = _straightest_lane(rb, prev_lane)
        if lane is None:
            return False
        seg_len = lane.baseline_path.length
        end_time = cum_time + seg_len / _lane_speed(
            lane, default_speed_mps, min_speed_mps
        )
        end_dist = cum_dist + seg_len
        steps = steps + [_Step(rb, lane, cum_time, end_time, cum_dist, end_dist)]
        if end_time > best_end_time:
            best_end_time = end_time
            best_steps = steps
        if end_time >= goal_time_s or len(steps) >= max_depth:
            return end_time >= goal_time_s

        candidates = [e for e in rb.outgoing_edges if e.id not in local_visited]
        ref_heading = lane.baseline_path.discrete_path[-1].heading
        candidates.sort(key=lambda e: _branch_misalignment(e, lane, ref_heading))
        for nxt in candidates:
            if dfs(nxt, lane, end_time, end_dist, steps, local_visited | {nxt.id}):
                return True
        return False

    dfs(branch, None, start_time_s, start_dist_m, [], visited | {branch.id})
    return best_steps


def _first_fork(
    steps: List["_Step"], blocked_ids: set
) -> Optional[Tuple[int, "_Step", List[RoadBlockGraphEdgeMapObject]]]:
    """First step whose roadblock forks off the path (has an untaken, unvisited outgoing edge).

    Returns ``(index, step, alternative branches)`` for the earliest such junction, or ``None`` if
    the path never forks. ``blocked_ids`` are roadblocks already on the shared prefix.
    """
    path_ids = {s.rb.id for s in steps}
    for i, s in enumerate(steps):
        taken_next = steps[i + 1].rb.id if i + 1 < len(steps) else None
        forks = [
            e
            for e in s.rb.outgoing_edges
            if e.id != taken_next and e.id not in blocked_ids and e.id not in path_ids
        ]
        if forks:
            return i, s, forks
    return None


def _build_alternative(
    token: str,
    prefix_ids: List[str],
    junction_lane: LaneGraphEdgeMapObject,
    branch: RoadBlockGraphEdgeMapObject,
    steps: List["_Step"],
    divergence_distance_m: float,
    goal_time_s: float,
    default_speed_mps: float,
    min_speed_mps: float,
    min_goal_time_s: float,
):
    """Assemble an :class:`AlternativeRoute` from a prefix + extension, or ``None`` if unusable.

    The goal is placed on the final lane at the arc matching the leftover time budget; the travel
    distance to it is the distance up to that lane plus that in-lane arc. Reaching ~``goal_time_s``
    is a *soft* target: shorter paths are still emitted (tagged ``reached_goal_time=False``); the
    ``min_goal_time_s`` guard (0 by default) only trims trivially short spurs when raised.
    """
    from .alternative_routes import AlternativeRoute

    if not steps:
        return None
    total_time_s = steps[-1].end_time
    if total_time_s < min_goal_time_s:
        return None

    final = steps[-1]
    remaining_time_s = goal_time_s - final.enter_time
    goal_arc_m = min(
        max(
            remaining_time_s
            * _lane_speed(final.lane, default_speed_mps, min_speed_mps),
            0.0,
        ),
        final.lane.baseline_path.length,
    )
    goal = _pose_at_arc_length(final.lane, goal_arc_m)
    goal_distance_m = final.enter_dist + goal_arc_m
    ext_ids = [s.rb.id for s in steps]
    return AlternativeRoute(
        token=token,
        route_ids=list(prefix_ids) + ext_ids,
        goal_position=goal,
        meta={
            "divergence_distance_m": round(divergence_distance_m, 2),
            "goal_distance_m": round(goal_distance_m, 2),
            "est_travel_time_s": round(min(total_time_s, goal_time_s), 2),
            "reached_goal_time": bool(total_time_s >= goal_time_s),
            "turn": _turn_label(junction_lane, branch),
            "num_roadblocks": len(prefix_ids) + len(ext_ids),
        },
    )


def _branch_misalignment(
    rb: RoadBlockGraphEdgeMapObject,
    ref_lane: LaneGraphEdgeMapObject,
    ref_heading: float,
) -> float:
    """Absolute heading difference between ``rb``'s straightest lane and ``ref_heading``."""
    lane = _straightest_lane(rb, ref_lane)
    if lane is None:
        return math.pi
    return abs(
        _normalize_angle(lane.baseline_path.discrete_path[0].heading - ref_heading)
    )


def _pose_at_arc_length(lane: LaneGraphEdgeMapObject, arc: float) -> Pose:
    """Linear-interpolate the lane baseline's discrete path at arc length ``arc``."""
    path = lane.baseline_path.discrete_path
    if arc <= 0.0 or len(path) == 1:
        p = path[0]
        return (p.x, p.y, p.heading)
    acc = 0.0
    for a, b in zip(path, path[1:]):
        seg = math.hypot(b.x - a.x, b.y - a.y)
        if acc + seg >= arc:
            t = (arc - acc) / seg if seg > 0 else 0.0
            return (a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), b.heading)
        acc += seg
    p = path[-1]
    return (p.x, p.y, p.heading)


def _turn_label(
    junction_lane: LaneGraphEdgeMapObject,
    branch: RoadBlockGraphEdgeMapObject,
    threshold: float = math.pi / 8,
) -> str:
    """Classify a branch as ``left`` / ``right`` / ``straight`` vs. the junction heading.

    Uses the *net* direction of the branch (first-to-last baseline displacement) rather than its
    start heading, since a turn connector begins roughly aligned with the entry and only curves
    later — comparing start headings would label every branch "straight".
    """
    ref = junction_lane.baseline_path.discrete_path[-1].heading
    lane = branch.interior_edges[0] if branch.interior_edges else None
    if lane is None:
        return "unknown"
    path = lane.baseline_path.discrete_path
    net_heading = math.atan2(path[-1].y - path[0].y, path[-1].x - path[0].x)
    delta = _normalize_angle(net_heading - ref)
    if delta > threshold:
        return "left"
    if delta < -threshold:
        return "right"
    return "straight"
