"""Data model and JSONL (de)serialization for alternative routes.

Each generated alternative is stored as one JSONL row, mirroring the style used in
``route_description_generation`` (``lg_routing_data.jsonl``). Downstream, the FlowDrive
planner reads these rows to override the mission goal / route during simulation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass
class AlternativeRoute:
    """A single alternative route for a nuPlan scenario.

    Attributes:
        token: The scenario token this alternative belongs to.
        route_ids: Ordered roadblock (lane-group) ids describing the alternative route.
        goal_position: The new goal / mission-goal pose as ``(x, y, heading)`` in the
            global map frame.
        meta: Optional free-form metadata (e.g. length, search strategy, original goal).
    """

    token: str
    route_ids: List[str]
    goal_position: Tuple[float, float, float]
    meta: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        # Ensure goal_position is a plain list for JSON.
        d["goal_position"] = list(self.goal_position)
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "AlternativeRoute":
        return cls(
            token=d["token"],
            route_ids=list(d["route_ids"]),
            goal_position=tuple(d["goal_position"]),  # type: ignore[arg-type]
            meta=d.get("meta", {}),
        )


def write_alternative_routes(
    routes: Iterable[AlternativeRoute],
    output_path: str | Path,
    append: bool = False,
) -> int:
    """Write alternative routes to a JSONL file. Returns the number of rows written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with output_path.open(mode, encoding="utf-8") as f:
        for route in routes:
            f.write(json.dumps(route.to_json_dict()) + "\n")
            n += 1
    return n


def read_alternative_routes(input_path: str | Path) -> Iterator[AlternativeRoute]:
    """Yield alternative routes from a JSONL file."""
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield AlternativeRoute.from_json_dict(json.loads(line))
