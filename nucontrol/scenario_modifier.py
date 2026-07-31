"""Modify the routing of a loaded nuPlan scenario in place (as far as the planner cares).

A nuPlan planner only reads two things to know *where it should go*: the route roadblock ids
(``scenario.get_route_roadblock_ids()``) and the mission goal (``scenario.get_mission_goal()``).
This module lets you swap those out — e.g. to drive a counterfactual alternative route produced by
:func:`nucontrol.route_search.search_alternative_routes` — without touching the underlying
nuPlan database.

``NuPlanScenario`` is a frozen dataclass, so instead of mutating it we wrap it in a thin proxy that
delegates *everything* to the original object except the two routing accessors, which it overrides.
The proxy is a drop-in ``AbstractScenario`` and can be handed straight to the simulation / planner.

Usage::

    from nucontrol.scenario_query import load_scenario
    from nucontrol.route_search import search_alternative_routes
    from nucontrol.scenario_modifier import change_routing

    scenario = load_scenario(token)
    alt = search_alternative_routes(scenario)[0]
    rerouted = change_routing(scenario, route_ids=alt.route_ids, goal_position=alt.goal_position)
    # rerouted.get_route_roadblock_ids() -> alt.route_ids
    # rerouted.get_mission_goal()        -> StateSE2 at alt.goal_position
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

Pose = Union[StateSE2, Tuple[float, float, float]]


def _to_state_se2(goal: Pose) -> StateSE2:
    """Coerce a ``(x, y, heading)`` tuple (or an existing StateSE2) into a StateSE2."""
    if isinstance(goal, StateSE2):
        return goal
    x, y, heading = goal
    return StateSE2(x, y, heading)


class ModifiedRoutingScenario:
    """Proxy over a nuPlan scenario that overrides the route roadblock ids and mission goal.

    Every attribute/method access that is not one of the overridden routing accessors is forwarded
    to the wrapped scenario, so this object behaves exactly like the original everywhere else.
    """

    def __init__(
        self,
        scenario: AbstractScenario,
        route_roadblock_ids: Optional[Sequence[str]] = None,
        mission_goal: Optional[Pose] = None,
        scenario_name: Optional[str] = None,
        alt_goal_distance_m: Optional[float] = None,
    ) -> None:
        self._scenario = scenario
        self._route_roadblock_ids: Optional[List[str]] = (
            list(route_roadblock_ids) if route_roadblock_ids is not None else None
        )
        self._mission_goal: Optional[StateSE2] = (
            _to_state_se2(mission_goal) if mission_goal is not None else None
        )
        # Optional unique name so several alternatives of the same token do not collide on the
        # simulation output directory (which is keyed by scenario_name == token by default).
        self._scenario_name: Optional[str] = scenario_name
        # Total planned distance of this alternative route to its goal (meta.goal_distance_m). Read
        # by the ego_progress_along_expert_route metric to build a synthetic progress reference, since
        # alternative routes have no expert (logged-human) trajectory to compare against.
        self._alt_goal_distance_m: Optional[float] = alt_goal_distance_m

    # --- overridden identity ----------------------------------------------------------------
    @property
    def scenario_name(self) -> str:
        if self._scenario_name is not None:
            return self._scenario_name
        return self._scenario.scenario_name

    # --- alternative-route metadata ---------------------------------------------------------
    @property
    def alt_goal_distance_m(self) -> Optional[float]:
        """Planned distance [m] of this alternative route to its goal (``None`` if not provided).

        Defined as a real class property so normal attribute lookup finds it *before* ``__getattr__``
        delegates to the wrapped scenario; a plain nuPlan scenario has no such attribute, so metrics
        use ``getattr(scenario, "alt_goal_distance_m", None)`` and get ``None`` there.
        """
        return self._alt_goal_distance_m

    # --- overridden routing accessors -------------------------------------------------------
    def get_route_roadblock_ids(self) -> List[str]:
        if self._route_roadblock_ids is not None:
            return list(self._route_roadblock_ids)
        return self._scenario.get_route_roadblock_ids()

    def get_mission_goal(self) -> StateSE2:
        if self._mission_goal is not None:
            return self._mission_goal
        return self._scenario.get_mission_goal()

    # --- transparent delegation for everything else -----------------------------------------
    def __getattr__(self, name: str):
        # __getattr__ only fires for attributes not found normally, so the wrapped scenario's
        # methods/properties (map_api, token, initial_ego_state, get_*, ...) come through here.
        # Guard the proxy's own private attributes: during unpickling the instance exists before
        # its __dict__ is restored, so a lookup of ``_scenario`` (or any private field) must raise
        # AttributeError instead of recursing through getattr(self._scenario, ...) forever.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._scenario, name)

    # --- pickling (Ray sends scenarios across workers) --------------------------------------
    # Define state explicitly so pickle uses the proxy's own __dict__ and never routes __getstate__
    # / __setstate__ probing through __getattr__ into the wrapped scenario.
    def __getstate__(self) -> dict:
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ModifiedRoutingScenario({self._scenario!r})"


def change_routing(
    scenario: AbstractScenario,
    route_ids: Optional[Sequence[str]] = None,
    goal_position: Optional[Pose] = None,
    scenario_name: Optional[str] = None,
    alt_goal_distance_m: Optional[float] = None,
) -> ModifiedRoutingScenario:
    """Return a rerouted view of ``scenario`` with a new route and/or mission goal.

    Args:
        scenario: A loaded nuPlan scenario (e.g. from :func:`load_scenario`).
        route_ids: New ordered roadblock ids for the route. If ``None``, the original route is kept.
        goal_position: New mission goal as ``(x, y, heading)`` or a ``StateSE2``. If ``None``, the
            original mission goal is kept.
        scenario_name: Optional unique name for the rerouted view. Use this when several
            alternatives share the same token, so their simulation outputs (keyed by
            ``scenario_name``) do not overwrite each other. If ``None``, the original name is kept.
        alt_goal_distance_m: Planned distance [m] of this alternative route to its goal, exposed to
            the ego-progress metric as a reference. If ``None``, the metric keeps its expert path.

    Returns:
        A :class:`ModifiedRoutingScenario` drop-in that reports the new route/goal but otherwise
        behaves identically to ``scenario`` (safe to pass to the planner / simulation).
    """
    return ModifiedRoutingScenario(
        scenario,
        route_roadblock_ids=route_ids,
        mission_goal=goal_position,
        scenario_name=scenario_name,
        alt_goal_distance_m=alt_goal_distance_m,
    )
