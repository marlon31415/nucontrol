"""Expand a list of nuPlan scenarios into one rerouted scenario per alternative route.

Given the scenarios the simulation is about to run and a JSONL file of alternative routes
(produced by ``nucontrol-generate-routes``), this replaces each scenario whose token has
alternatives with one :class:`ModifiedRoutingScenario` per alternative. Each proxy carries a
unique ``scenario_name`` (``<token>_alt<i>``) so their simulation outputs do not collide.

This is an *alternatives-only* expansion: scenarios whose token has no alternative in the JSONL
are dropped, so the run covers strictly the counterfactual missions.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Union

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from .alternative_routes import AlternativeRoute, read_alternative_routes
from .scenario_modifier import change_routing


def load_alternatives_by_token(
    jsonl_path: Union[str, Path]
) -> Dict[str, List[AlternativeRoute]]:
    """Load alternative routes from JSONL, grouped by scenario token (order preserved)."""
    by_token: Dict[str, List[AlternativeRoute]] = defaultdict(list)
    for alt in read_alternative_routes(jsonl_path):
        by_token[alt.token].append(alt)
    return by_token


def expand_scenarios_with_alternatives(
    scenarios: Sequence[AbstractScenario],
    jsonl_path: Union[str, Path],
) -> List[AbstractScenario]:
    """Return one rerouted scenario per alternative route (alternatives-only).

    Args:
        scenarios: The scenarios the simulation would otherwise run.
        jsonl_path: JSONL of alternative routes from ``nucontrol-generate-routes``.

    Returns:
        A flattened list: for each input scenario whose token appears in the JSONL, one
        :class:`ModifiedRoutingScenario` per alternative (named ``<token>_alt<i>``). Scenarios
        whose token has no alternative are omitted.
    """
    by_token = load_alternatives_by_token(jsonl_path)
    expanded: List[AbstractScenario] = []
    for scenario in scenarios:
        alts = by_token.get(scenario.token)
        if not alts:
            continue
        for i, alt in enumerate(alts):
            expanded.append(
                change_routing(
                    scenario,
                    route_ids=alt.route_ids,
                    goal_position=alt.goal_position,
                    scenario_name=f"{scenario.token}_alt{i}",
                )
            )
    return expanded
