"""Command-line entry point for generating alternative routes.

Every tunable knob can be set in a YAML parameter file (see ``params.yaml``) and/or overridden on
the command line. Precedence is: explicit CLI flag > ``--config`` YAML value > built-in default.

    nucontrol-generate-routes --config params.yaml
    nucontrol-generate-routes --config params.yaml --max-alternatives 5
    nucontrol-generate-routes --tokens <t1> <t2> --output alternative_routes.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from .alternative_routes import AlternativeRoute, write_alternative_routes
from .config import (
    DEFAULTS,
    PACKAGE_ROOT,
    load_config,
    make_loader,
    resolve,
    resolve_config_path,
    search_kwargs,
)
from .route_search import search_alternative_routes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nucontrol-generate-routes",
        description="Query nuPlan scenarios by token and generate alternative routes (JSONL).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "YAML parameter file. If omitted, params.yaml is discovered automatically "
            "(NUCONTROL_CONFIG env > ./params.yaml > the shipped file). CLI flags override it."
        ),
    )

    # Input / output. CLI flags default to None so we can tell "not passed" from a real value and
    # fall back to the config file; the built-in defaults live in _DEFAULTS.
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--tokens", nargs="+", help="One or more nuPlan scenario tokens to process.")
    src.add_argument(
        "--tokens-file",
        type=Path,
        help=(
            "Path to a text file with one scenario token per line, or a nuPlan scenario_filter "
            "YAML (e.g. val14.yaml) with a 'scenario_tokens:' list."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL file name. Only the name is used; the file is always written to the "
             "package root (any directory component is ignored).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=None,
        help="Append to the output file instead of overwriting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Print a per-token summary (divergence distance, #alternatives, goals).",
    )

    # Dataset / loader.
    parser.add_argument("--data-root", default=None, help="nuPlan data root (else NUPLAN_DATA_ROOT).")
    parser.add_argument("--map-root", default=None, help="nuPlan maps root (else NUPLAN_MAPS_ROOT).")
    parser.add_argument("--map-version", default=None, help="Map database version.")
    parser.add_argument(
        "--splits", nargs="+", default=None, help="Split subdir(s) to search, e.g. trainval or test."
    )
    parser.add_argument(
        "--scan-threads", type=int, default=None, help="Threads for the token-location scan."
    )

    # Alternative-route search.
    parser.add_argument("--max-alternatives", type=int, default=None, help=(
        "Target number of alternatives per scenario. All options at the first junction are always "
        "kept (even if more than this); this only caps the second-junction top-up."
    ))
    parser.add_argument(
        "--divergence-max-distance-m", type=float, default=None,
        help="Max distance ahead of the ego (m) for the first divergence junction.",
    )
    parser.add_argument("--goal-time-s", type=float, default=None, help="Soft travel-time target (s).")
    parser.add_argument(
        "--min-goal-time-s", type=float, default=None, help="Min achievable travel time to keep (s)."
    )
    parser.add_argument(
        "--default-speed-mps", type=float, default=None, help="Fallback speed when a limit is missing."
    )
    parser.add_argument(
        "--min-speed-mps", type=float, default=None, help="Floor on per-lane speed used for timing."
    )
    return parser


def _read_tokens_file(path: Path) -> List[str]:
    """Read scenario tokens from a file.

    Supports two formats, picked by extension:
    - a nuPlan ``scenario_filter`` YAML (``.yaml``/``.yml``, e.g. ``val14.yaml``) with a
      ``scenario_tokens:`` list, or
    - a plain text file with one token per line (``#`` comments allowed).
    """
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        scenario_tokens = data.get("scenario_tokens") if isinstance(data, dict) else None
        if not scenario_tokens:
            raise ValueError(
                f"{path} has no non-empty 'scenario_tokens' list "
                "(expected a nuPlan scenario_filter YAML)."
            )
        return [str(t) for t in scenario_tokens]

    tokens: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                tokens.append(line)
    return tokens


def _resolve_tokens(args: argparse.Namespace, io_cfg: Dict[str, Any]) -> List[str]:
    if args.tokens:
        return list(args.tokens)
    if args.tokens_file:
        return _read_tokens_file(args.tokens_file)
    if io_cfg.get("tokens"):
        return list(io_cfg["tokens"])
    if io_cfg.get("tokens_file"):
        return _read_tokens_file(Path(io_cfg["tokens_file"]))
    return []


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # params.yaml is applied automatically (env NUCONTROL_CONFIG > ./params.yaml > shipped file);
    # --config overrides the discovery, and individual CLI flags override the file's values.
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    if config_path is not None:
        print(f"Using parameters from {config_path}")

    # Dataset + search parameters resolve identically to the tests (CLI > config > default).
    loader = make_loader(
        config,
        data_root=args.data_root,
        map_root=args.map_root,
        map_version=args.map_version,
        splits=args.splits,
        scan_threads=args.scan_threads,
    )
    kwargs = search_kwargs(
        config,
        max_alternatives=args.max_alternatives,
        divergence_max_distance_m=args.divergence_max_distance_m,
        goal_time_s=args.goal_time_s,
        min_goal_time_s=args.min_goal_time_s,
        default_speed_mps=args.default_speed_mps,
        min_speed_mps=args.min_speed_mps,
    )

    io = {**DEFAULTS["io"], **(config.get("io") or {})}
    # The output setting only controls the file *name*: generated routes are always written to the
    # package root, regardless of any directory component in --output / io.output or the CWD.
    output_name = Path(resolve(args.output, io, "output", DEFAULTS["io"]["output"])).name
    output = PACKAGE_ROOT / output_name
    append = bool(resolve(args.append, io, "append", DEFAULTS["io"]["append"]))
    verbose = bool(resolve(args.verbose, io, "verbose", DEFAULTS["io"]["verbose"]))

    tokens = _resolve_tokens(args, io)
    if not tokens:
        parser.error("no tokens given (use --tokens/--tokens-file or set io.tokens in --config).")

    # Batch-load all scenarios in a single pass (locating tokens is the slow part, so we do it once
    # for every token rather than once per token).
    print(f"Loading {len(tokens)} scenario(s) ...", flush=True)
    scenarios = loader.load_scenarios(tokens)

    all_routes: List[AlternativeRoute] = []
    n_skipped = 0
    for scenario in tqdm(scenarios, desc="scenarios"):
        token = scenario.token
        routes = search_alternative_routes(scenario, **kwargs)
        if not routes:
            n_skipped += 1
        all_routes.extend(routes)
        if verbose:
            if not routes:
                print(f"{token}: no alternative route (skipped)")
            for r in routes:
                gx, gy, gh = r.goal_position
                print(
                    f"{token}: {r.meta.get('turn')} div@{r.meta.get('divergence_distance_m')}m "
                    f"t~{r.meta.get('est_travel_time_s')}s "
                    f"{len(r.route_ids)} roadblocks goal=({gx:.1f},{gy:.1f},{gh:.2f})"
                )

    n = write_alternative_routes(all_routes, output, append=append)
    print(f"{n_skipped}/{len(tokens)} scenario(s) had no alternative route.")
    print(f"Wrote {n} alternative route(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
