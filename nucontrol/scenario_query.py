"""Load nuPlan scenarios by token via the vendored ``nuplan-devkit``.

We deliberately return nuPlan's native :class:`NuPlanScenario` objects rather than a
reduced view: the scenario dataclass already exposes everything the alternative-route
search needs (ego states, map API, mission goal, route roadblock ids, tracked objects,
...), so downstream code can query whatever it requires.

Speed comes straight from nuPlan's own logic. :meth:`NuPlanScenarioBuilder.get_scenarios`
opens a log DB only when its log name is in ``ScenarioFilter.log_names`` (see
``nuplan_scenario_builder._create_scenarios``). This is exactly how the simulation loads
val14 fast: it restricts to the ~1.4k val-split logs instead of the ~14.5k logs in the
whole ``trainval`` split. We do the same by passing ``log_names`` from the bundled official
split lists (``nucontrol/data/nuplan_{train,val,test}.json``), so building all val14
scenarios opens only the val logs and returns in seconds — no custom token scan or cache.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, cast

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)

# Default map version used across the FlowDrive workspace.
DEFAULT_MAP_VERSION = "nuplan-maps-v1.0"

# Maximum number of tokens per DB query, to stay within the SQL variable limit.
_MAX_TOKENS_PER_QUERY = 100000

# Default worker count for the parallel scenario build (opens/queries the log DBs).
_DEFAULT_MAX_WORKERS = 64

# Default split(s) to search for ``.db`` files when ``NUPLAN_DATA_ROOT`` is a dataset mount root
# rather than a split dir. The official nuPlan dataset ships only ``trainval`` and ``test``; the
# val14 benchmark scenarios live in ``trainval``.
_DEFAULT_SPLITS = ("trainval",)

# Official log-split lists shipped with the package (one log name per entry, no ``.db``), used as
# ``ScenarioFilter.log_names`` so only the split's logs are opened. Mirrors nuPlan's
# ``splitter.log_splits.{train,val,test}`` used by the simulation configs.
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_LOG_SPLIT_FILES = {
    "train": "nuplan_train.json",
    "val": "nuplan_val.json",
    "test": "nuplan_test.json",
}


def _load_log_names(log_split: Optional[str]) -> Optional[List[str]]:
    """Return the log names for a split, or ``None`` (no restriction) for ``None``/``"all"``.

    ``None`` / ``"all"`` means "open every log under the search dirs" — correct but slow; use it
    only when the tokens are not confined to one official split.
    """
    if log_split is None or log_split == "all":
        return None
    fname = _LOG_SPLIT_FILES.get(log_split)
    if fname is None:
        raise ValueError(
            f"unknown log_split {log_split!r}; expected one of "
            f"{sorted(_LOG_SPLIT_FILES)} or 'all'/None."
        )
    with open(os.path.join(_DATA_DIR, fname), "r", encoding="utf-8") as f:
        return list(json.load(f))


def _resolve_search_dirs(
    data_root: str, include_splits: Optional[List[str]] = None
) -> List[str]:
    """Resolve the directories that hold the log ``.db`` files under ``data_root``.

    Following nuPlan's own convention (see ``config/common/scenario_builder/nuplan.yaml``),
    ``NUPLAN_DATA_ROOT`` is the dataset *mount root* and the logs live under
    ``nuplan-v1.1/splits/<split>/``. This resolves both that layout and a directly-given split dir:

    - If ``data_root`` already holds ``.db`` files, use it as-is (it is a split dir).
    - Otherwise treat ``data_root`` as a mount root and return the requested split subdirectories
      under ``nuplan-v1.1/splits`` (default ``trainval``, the official split holding val14).
    - As a last resort, return every directory that contains a ``.db`` file anywhere below
      ``data_root``.
    """
    if glob.glob(os.path.join(data_root, "*.db")):
        return [data_root]

    splits_dir = os.path.join(data_root, "nuplan-v1.1", "splits")
    if os.path.isdir(splits_dir):
        splits = list(include_splits) if include_splits else list(_DEFAULT_SPLITS)
        dirs = [
            os.path.join(splits_dir, s)
            for s in splits
            if os.path.isdir(os.path.join(splits_dir, s))
        ]
        if dirs:
            return dirs

    # Unusual layout: find any directory below data_root that holds log DBs.
    found = sorted(glob.glob(os.path.join(data_root, "**", "*.db"), recursive=True))
    return sorted({os.path.dirname(p) for p in found})


class ScenarioLoader:
    """Loads nuPlan scenarios by token using nuPlan's own builder, restricted to a log split.

    Args:
        data_root: Path to the nuPlan data root. May be either a directory that directly holds
            the log ``.db`` files (e.g. a split dir), or a dataset mount root under which they
            live in ``nuplan-v1.1/splits/<split>/`` — both are handled. Defaults to the
            ``NUPLAN_DATA_ROOT`` environment variable.
        map_root: Path to the nuPlan maps root. Defaults to ``NUPLAN_MAPS_ROOT``.
        map_version: Map database version. Defaults to :data:`DEFAULT_MAP_VERSION`.
        include_splits: When ``data_root`` is a mount root, which split *directories* to search for
            ``.db`` files. ``None`` (default) uses ``trainval`` (the dir holding val14's logs).
        log_split: Which official log split to restrict loading to — ``"val"`` (default),
            ``"train"``, ``"test"``, or ``"all"``/``None`` for no restriction. This is what makes
            loading fast: only the split's logs are opened. For val14/test14 keep the matching
            split; use ``"all"`` only when tokens span splits (slow — opens every log).
        max_workers: Worker count for the parallel scenario build (default 64).
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        map_root: Optional[str] = None,
        map_version: str = DEFAULT_MAP_VERSION,
        include_splits: Optional[List[str]] = None,
        log_split: Optional[str] = "val",
        max_workers: Optional[int] = None,
    ) -> None:
        self.data_root = data_root or os.environ.get("NUPLAN_DATA_ROOT")
        self.map_root = map_root or os.environ.get("NUPLAN_MAPS_ROOT")
        if not self.data_root:
            raise ValueError(
                "data_root not provided and NUPLAN_DATA_ROOT is not set; "
                "cannot locate nuPlan log databases."
            )
        if not self.map_root:
            raise ValueError(
                "map_root not provided and NUPLAN_MAPS_ROOT is not set; "
                "cannot locate nuPlan maps."
            )
        self.map_version = map_version
        self.max_workers = max_workers or _DEFAULT_MAX_WORKERS

        # Allow selecting split(s) via env (e.g. NUCONTROL_SPLITS="test") when NUPLAN_DATA_ROOT is a
        # mount root; the explicit argument takes precedence.
        if include_splits is None:
            env_splits = os.environ.get("NUCONTROL_SPLITS")
            if env_splits:
                include_splits = [s.strip() for s in env_splits.split(",") if s.strip()]

        self.log_split = log_split
        self.search_dirs = _resolve_search_dirs(self.data_root, include_splits)
        if not self.search_dirs:
            raise ValueError(
                f"No nuPlan log '.db' directories found under {self.data_root!r}. Check that the "
                "dataset is mounted and that NUPLAN_DATA_ROOT points at the mount root (with logs "
                "under nuplan-v1.1/splits/<split>) or directly at a split dir; select a split other "
                "than the default 'trainval' via NUCONTROL_SPLITS (e.g. 'test')."
            )

    def load_scenarios(self, tokens: List[str]) -> List[NuPlanScenario]:
        """Load scenarios for the given tokens, preserving input order.

        Args:
            tokens: nuPlan lidarpc scenario tokens.

        Returns:
            The corresponding :class:`NuPlanScenario` objects, in the same order as ``tokens``.

        Raises:
            KeyError: If any requested token could not be found. When a token is missing, the most
                common cause is a ``log_split`` that does not contain it (e.g. loading test14 tokens
                with the default ``"val"`` split); set ``log_split`` accordingly or use ``"all"``.
        """
        if not tokens:
            return []

        log_names = _load_log_names(self.log_split)

        # Discover DB filenames under the search dirs (a cheap glob — the files are not opened here).
        # nuPlan opens a DB only when its log name is in ``log_names``, so restricting to the split's
        # logs is what keeps this fast.
        builder = NuPlanScenarioBuilder(
            self.data_root,
            self.map_root,
            None,  # type: ignore[arg-type]  # sensor_root typed str, but None (sensorless) is valid
            self.search_dirs,  # type: ignore[arg-type]  # list[dir] is a valid db_files load path
            self.map_version,
            max_workers=self.max_workers,
            verbose=False,
        )
        worker = SingleMachineParallelExecutor(
            use_process_pool=False, max_workers=self.max_workers
        )

        token_to_scenario: Dict[str, NuPlanScenario] = {}
        for start in range(0, len(tokens), _MAX_TOKENS_PER_QUERY):
            chunk = tokens[start : start + _MAX_TOKENS_PER_QUERY]
            scenario_filter = ScenarioFilter(
                scenario_types=None,
                scenario_tokens=chunk,  # type: ignore[arg-type]  # list[str] of tokens is accepted
                log_names=log_names,
                map_names=None,
                num_scenarios_per_type=None,
                limit_total_scenarios=None,
                timestamp_threshold_s=None,
                ego_displacement_minimum_m=None,
                expand_scenarios=False,
                remove_invalid_goals=False,
                shuffle=False,
            )
            for scenario in builder.get_scenarios(scenario_filter, worker):
                # get_scenarios is typed List[AbstractScenario]; the NuPlanScenarioBuilder always
                # yields NuPlanScenario instances at runtime.
                scenario = cast(NuPlanScenario, scenario)
                token_to_scenario[scenario.token] = scenario

        missing = [t for t in tokens if t not in token_to_scenario]
        if missing:
            raise KeyError(
                f"{len(missing)} token(s) not found in log split {self.log_split!r} under "
                f"{self.data_root!r}: {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
                + " (wrong log_split? set it to the split holding these tokens, or 'all')."
            )

        return [token_to_scenario[t] for t in tokens]

    def load_scenario(self, token: str) -> NuPlanScenario:
        """Load a single scenario by token."""
        return self.load_scenarios([token])[0]


# Module-level convenience API backed by a lazily-created default loader.
_default_loader: Optional[ScenarioLoader] = None


def _get_default_loader() -> ScenarioLoader:
    global _default_loader
    if _default_loader is None:
        _default_loader = ScenarioLoader()
    return _default_loader


def load_scenario(token: str) -> NuPlanScenario:
    """Load a single nuPlan scenario by token using the default loader."""
    return _get_default_loader().load_scenario(token)


def load_scenarios(tokens: List[str]) -> List[NuPlanScenario]:
    """Load multiple nuPlan scenarios by token using the default loader."""
    return _get_default_loader().load_scenarios(tokens)
