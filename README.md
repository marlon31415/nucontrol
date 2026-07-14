# nucontrol

Query nuPlan scenarios by scenario token and search for **alternative routes**, so the
FlowDrive planner can be driven along counterfactual missions during closed-loop simulation.

## Idea

Given a scenario token, the tool:

1. Loads the corresponding nuPlan scenario via the `nuplan-devkit`.
2. Extracts the ego start state, map, and original route (roadblock ids / mission goal).
3. Searches for one or more **alternative routes** through the map graph
   (different roadblock sequences leading to a different goal position / mission goal).
4. Writes the alternatives to a JSONL file — one row per alternative — following the same
   style as `route_description_generation` (`lg_routing_data.jsonl`).

Each JSONL row has the shape:

```json
{
  "token": "<scenario_token>",
  "route_ids": ["<roadblock_id>", "..."],
  "goal_position": [x, y, heading],
  "meta": {
    "divergence_distance_m": 21.5,
    "goal_distance_m": 730.4,
    "est_travel_time_s": 60.0,
    "reached_goal_time": true,
    "turn": "left",
    "num_roadblocks": 17
  }
}
```

`meta` tags each alternative:
- `divergence_distance_m` — how far ahead of the ego the route leaves the original.
- `goal_distance_m` / `est_travel_time_s` — travel distance and estimated driving time to the goal.
- `reached_goal_time` — whether the branch reached the ~`goal_time_s` (~1 min) target before
  dead-ending. The 1-min goal is a *soft* target: shorter branches are still emitted with their
  achievable distance/time recorded here.
- `turn` — `left` / `right` / `straight` relative to the original route at the divergence.

Later, during FlowDrive inference, this file is read to override the mission goal / route
and simulate the ego driving the alternative route.

## Route lanes

A route is stored as an ordered list of **roadblock** ids, but what the ego actually drives on are
the **lanes** inside those roadblocks. Roadblocks alternate `ROADBLOCK` ↔ `ROADBLOCK_CONNECTOR`,
each holding one or more parallel `interior_edges` (lanes); a lane is *on route* iff its parent
roadblock is on the route. `route_lanes(map_api, roadblock_ids)` expands a route into those lanes,
and the visualization below paints them blue so you can eyeball whether extraction is correct
(connected, starts at the ego, leads to the goal).

## Checking route-lane extraction (visualization)

`tests/test_route_visualization.py` renders one figure per route — the original plus each
alternative — with the route lanes highlighted in blue over the grey surrounding map, the ego (green)
and the goal (red star). Each figure auto-frames to its own route:

```bash
export NUPLAN_DATA_ROOT=/scratch/nuplan/dataset
export NUPLAN_MAPS_ROOT=/scratch/nuplan/dataset/maps
# as a test (writes PNGs to $NUCONTROL_VIZ_OUT or a temp dir):
NUCONTROL_VIZ_OUT=./route_viz .venv/bin/python -m pytest -s tests/test_route_visualization.py
# or standalone for any token:
.venv/bin/python tests/test_route_visualization.py <token> ./route_viz
```

`8eb8410b1e385101` is a good demo token: a standstill scenario at an intersection with a clean
left / right split, exercising the standstill handling described below.

## How route search works

- The **original route** (`extract_original_route`) is a clean, connected roadblock chain anchored
  at the ego's current roadblock and walked forward along the scenario's stored **mission route**
  (straightest branch among ties) up to the ~`goal_time_s` budget. It is *not* taken from the
  expert future trajectory alone: at a standstill that trajectory covers a single roadblock, which
  would both hide the intended route and leave the on-route branch unknown — so an alternative
  labeled "straight" could actually *be* the original. Anchoring to the mission route fixes both.
  It also fixes the ego's starting roadblock and the branch the ego takes at each junction.
- Alternatives form a small **route tree**, not a single-junction fan-out:
  - **Level 1 — first junction.** Walking the original route from the ego up to
    `divergence_max_distance_m` (default 50 m), the *first* junction with a roadblock
    `outgoing_edge` other than the on-route next roadblock is the divergence point. **Every**
    non-original branch there becomes an alternative — all of them are kept, even if there are more
    than `max_alternatives`.
  - **Level 2 — fill.** If the first junction yields fewer than `max_alternatives`, the result is
    topped up from the **next junction along each already-found alternative** (in order): follow an
    alternative to its first onward fork and branch there too, until the cap is reached. This is
    what enables multi-turn routes like "right at the first junction, then left at the next".
  - So the count is `max(#first-junction options, min(max_alternatives, #reachable))` — more than
    `max_alternatives` only when the first junction alone provides them.
- Each branch (level 1 or 2) is extended by a **straightest-first DFS with backtracking** through
  `outgoing_edges`, accumulating `length / speed_limit_mps` until the ego would need ~`goal_time_s`
  (default 60 s) of driving; the goal pose is placed on the final lane at that arc length.
- Reaching ~`goal_time_s` is a *soft* target: branches that dead-end earlier are still emitted with
  their achievable `goal_distance_m` / `est_travel_time_s` (`min_goal_time_s`, default 0, only trims
  trivially short spurs when raised). Level-2 fills are deduplicated by final roadblock. Scenarios
  where the ego can only go straight yield no rows and are skipped.

Routes are connected by construction, so they need no post-hoc correction.

## Using it as a scenario modifier

Beyond writing JSONL, the package can reroute a **loaded** nuPlan scenario directly, so you can
inject an alternative mission wherever a scenario is used (e.g. closed-loop simulation). A planner
only reads the route roadblock ids and the mission goal to decide where to go, so `change_routing`
wraps the scenario in a transparent proxy that overrides just those two accessors and delegates
everything else (`map_api`, `token`, ego states, ...) to the original:

```python
from nucontrol import load_scenario, search_alternative_routes, change_routing

scenario = load_scenario(token)
alt = search_alternative_routes(scenario)[0]
rerouted = change_routing(scenario, route_ids=alt.route_ids, goal_position=alt.goal_position)

rerouted.get_route_roadblock_ids()  # -> alt.route_ids
rerouted.get_mission_goal()         # -> StateSE2 at alt.goal_position
```

`rerouted` is a drop-in `AbstractScenario` and can be handed straight to the simulation/planner.

## Package layout

```
nucontrol/
    scenario_query.py      # load nuPlan scenarios by token (NuPlanScenarioBuilder + ScenarioFilter)
    route_search.py        # search alternative routes; extract_original_route / route_lanes helpers
    scenario_modifier.py   # change_routing(): reroute a loaded scenario in place (proxy)
    visualize.py           # plot_route / visualize_token: map + route lanes highlighted in blue
    alternative_routes.py  # AlternativeRoute data model + JSONL (de)serialization
    cli.py                 # command-line entry point (--config params.yaml)
params.yaml                # all tunable parameters (dataset / search / io) in one place
tests/
    test_route_visualization.py  # renders one figure per route for a hand-picked token
    test_generate_dataset.py     # writes + round-trips an example JSONL dataset for that token
    test_params.yaml             # parameters for the tests (independent from params.yaml)
    output/                      # test artifacts land here (figures + example JSONL)
scripts/
    generate_alternative_routes.sh
```

## Environment

This package is **independent of `flow_drive_planner`**. It needs `nuplan-devkit` (for the map
API + scenario builder) but not its heavy torch/ray/bokeh/jupyter stack, so `nuplan-devkit` is
installed with `--no-deps` and only the geo/data libs it actually uses are pulled in.

```bash
cd nucontrol
uv venv --python 3.9 .venv
uv pip install --python .venv/bin/python --no-deps -e ../nuplan-devkit   # devkit, no heavy deps
uv pip install --python .venv/bin/python -e .                            # curated runtime deps
```

Point the tool at a mounted nuPlan dataset via env vars. Following nuPlan's own convention (see
`nuplan-devkit/.../scenario_builder/nuplan.yaml`), `NUPLAN_DATA_ROOT` is the dataset **mount
root** — the loader finds the log `.db` files under `nuplan-v1.1/splits/<split>/` for you:

```bash
export NUPLAN_DATA_ROOT=/scratch/nuplan/dataset
export NUPLAN_MAPS_ROOT=/scratch/nuplan/dataset/maps
```

The official nuPlan dataset ships only the `trainval` and `test` splits (there is no `val` split).
By default the loader uses `trainval`, which holds the val14 scenarios. Choose another split with
`NUCONTROL_SPLITS`, e.g. `NUCONTROL_SPLITS=test` for test14 or `NUCONTROL_SPLITS=mini` for the mini
set. Pointing `NUPLAN_DATA_ROOT` directly at a split dir (`…/splits/trainval`) also works and skips
the resolution step.

**Locating a token is the expensive part.** nuPlan finds a token by opening *every* log DB in the
split (~14.5k for `trainval`); on a SquashFS mount that file-open storm takes minutes and can't be
parallelized away (it's I/O-bound on the opens, not the query). So the loader instead does a cheap
raw `lidar_pc` lookup to find each token's owning DB, **caches** the `token → db` mapping to disk,
and then builds the scenario from just that DB. The first lookup of an uncached token pays the
one-time scan (a few minutes on a cold mount); every lookup after that — same token, any run — is
effectively instant (~0.01 s). The cache lives at `$NUCONTROL_CACHE_DIR` or
`~/.cache/nucontrol/token_index.sqlite`.

## Usage

```bash
nucontrol-generate-routes \
    --tokens <token1> <token2> ... \
    --output alternative_routes.jsonl \
    --verbose
# or from a file of tokens (one per line):
nucontrol-generate-routes --tokens-file tokens.txt --output alternative_routes.jsonl
# or straight from a nuPlan scenario_filter YAML (tokens read from its `scenario_tokens:` list):
nucontrol-generate-routes --tokens-file val14.yaml --output alternative_routes.jsonl
```

`--tokens-file` accepts two formats, picked by extension:
- a plain text file with **one token per line** (`#` comments allowed);
- a nuPlan **`scenario_filter` YAML** (`.yaml` / `.yml`, e.g. `val14.yaml`) — the tokens are read
  from its `scenario_tokens:` list, so you can point it directly at a FlowDrive eval split without
  extracting the tokens first. This is the natural way to generate alternatives for exactly the
  scenarios an eval config will simulate.

### Configuration (`params.yaml`)

Every tunable knob — dataset/loader, the alternative-route search, and I/O — lives in one YAML file
(`params.yaml`). It is **applied automatically**, so you normally just edit `params.yaml` and run:

```bash
nucontrol-generate-routes --tokens <token>
# or drive everything (tokens + output too) from the file:
nucontrol-generate-routes
```

The file to use is discovered as **`NUCONTROL_CONFIG` env > `./params.yaml` > the shipped file**, or
pointed at explicitly with `--config other.yaml`. Precedence for each value is **explicit CLI flag >
config file > built-in default**, so you can keep a standing setup in the file and still override one
value on the command line:

```bash
nucontrol-generate-routes --tokens <token> --max-alternatives 5
```

The file has three sections — `dataset` (`data_root`, `map_root`, `map_version`, `splits`,
`scan_threads`), `search` (`max_alternatives`, `divergence_max_distance_m`, `goal_time_s`,
`min_goal_time_s`, `default_speed_mps`, `min_speed_mps`), and `io` (`tokens`, `tokens_file`,
`output`, `append`, `verbose`). Any value left `null` falls back to the default / environment
variable. See `params.yaml` for the full annotated list.

**Output location.** `output` (and `--output`) only set the file *name* — the JSONL is always
written to the **package root** (the `nucontrol/` directory that holds `params.yaml`).
Any directory component is stripped and the current working directory is irrelevant, so repeated
runs always land in the same place. Pass just a name, e.g. `--output my_routes.jsonl` →
`nucontrol/my_routes.jsonl`.

The tests use their **own** parameter file, `tests/test_params.yaml`, independent from the top-level
`params.yaml` — so tweaking one never affects the other. `pytest` picks it up with no extra flags;
pass `--config other.yaml` (or set `NUCONTROL_CONFIG`) to point either the tests or the CLI at a
different file. Test artifacts (the figures and the example JSONL) are written under
`tests/output/`.

Pass all tokens together in one invocation: the loader locates and caches them in a single pass
(see the token-location note above), so a batch amortizes the one-time split scan and every already
-cached token loads instantly. Tune the scan parallelism with the `scan_threads` argument to
`ScenarioLoader` (default 64).
