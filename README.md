# rust_mc_vrp

Multi-criteria **Capacitated Vehicle Routing Problem (CVRP)** solver in Rust.

This branch currently:
- parses TSPLIB-like `.vrp` instances (Uchoa et al. X-set style),
- builds a **complete directed graph** with **two edge costs**,
- runs an **NSGA-II style** evolutionary search with **HGreX crossover** and a **route-aware mutation**.

## Problem model

- **Nodes**: `DIMENSION` total nodes including the depot.
- **Depot**: read from `DEPOT_SECTION` (terminated by `-1`). Only the first depot id is used.
- **Demands**: read from `DEMAND_SECTION`.
- **Capacity**: read from `CAPACITY`.

### Indexing convention

The `.vrp` files are typically **1-based** (node ids start at `1`).
Internally this project uses **0-based** ids:
- file node `1` → internal node `0`
- file node `k` → internal node `k-1`

### Edge costs / objectives

The instance is turned into a **complete directed graph** (edges `i -> j` for all `i != j`).
Each edge has two parameters:
1. **Distance**: Euclidean distance between `(x, y)` coordinates (objective 0)
2. **Secondary cost**: a random integer in `[1, 100]` stored as `f64` (objective 1)

The solver minimizes both objectives (Pareto optimization).

## Solution encoding

Each individual stores a **permutation of customers** (all nodes except the depot).
Routes are **implicit**: the permutation is split into consecutive segments by greedily packing
customers until adding the next one would exceed capacity.

## Algorithm

- **Selection**: tournament selection with rank + crowding distance (NSGA-II).
- **Crossover**: **HGreX (heuristic greedy crossover)**, edge-based and capacity-aware.
- **Mutation**: route-aware operator mix:
  - intra-route 2-opt (segment reversal)
  - relocate (remove one customer and insert elsewhere)
  - swap (prefers inter-route when possible)
- **Fitness**: sum of edge costs across all routes (including depot legs).
- **Convergence**: tracks 2D hypervolume of the archive and stops when it stalls.

## Input format

This parser expects a TSPLIB-like CVRP file containing (order can vary, extra header lines are ignored):

- `DIMENSION : <n>`
- `CAPACITY : <cap>`
- `NODE_COORD_SECTION`
- `DEMAND_SECTION`
- `DEPOT_SECTION`
- `EOF`

Example snippet:

```text
DIMENSION : 134
CAPACITY : 643
NODE_COORD_SECTION
1 113 195
2 377 239
...
DEMAND_SECTION
1 0
2 53
...
DEPOT_SECTION
1
-1
EOF
```

## Usage

Build:

```bash
cargo build --release
```

Run (defaults are set in `src/main.rs`):

```bash
cargo run --release -- --graph-path xset/X-n134-k13.vrp
```

Common flags:
- `--population-size <n>`
- `--p-crossover <0..1>`
- `--p-mutation <0..1>`
- `--runs <k>`
- `--output <path>`

## Notes / current limitations

- The graph is fully materialized (complete directed), so memory is \(O(n^2)\).
- The secondary edge cost is currently random; if you want a deterministic second objective, change
  how `Graph::new()` assigns `parameters.1`.