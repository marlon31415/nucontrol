"""Unit tests for ModifiedRoutingScenario.alt_goal_distance_m (no map/data needed).

The ego_progress_along_expert_route metric reads this attribute via
``getattr(scenario, "alt_goal_distance_m", None)`` to build a synthetic progress reference for
alternative routes; a plain nuPlan scenario must return None there so its behavior is unchanged.
"""

import unittest

from nucontrol.scenario_modifier import ModifiedRoutingScenario, change_routing


class _DummyScenario:
    """Minimal stand-in for a wrapped nuPlan scenario (only identity matters here)."""

    scenario_name = "tok"
    token = "tok"


class TestAltGoalDistance(unittest.TestCase):
    def test_stored_value_is_exposed(self) -> None:
        s = ModifiedRoutingScenario(_DummyScenario(), alt_goal_distance_m=670.54)
        self.assertEqual(s.alt_goal_distance_m, 670.54)

    def test_default_is_none(self) -> None:
        s = ModifiedRoutingScenario(_DummyScenario())
        self.assertIsNone(s.alt_goal_distance_m)

    def test_change_routing_threads_the_value(self) -> None:
        s = change_routing(_DummyScenario(), route_ids=["a", "b"], alt_goal_distance_m=5.0)
        self.assertEqual(s.alt_goal_distance_m, 5.0)

    def test_getattr_on_plain_scenario_is_none(self) -> None:
        # A plain nuPlan scenario has no such attribute -> metric's getattr default applies.
        self.assertIsNone(getattr(_DummyScenario(), "alt_goal_distance_m", None))


if __name__ == "__main__":
    unittest.main()
