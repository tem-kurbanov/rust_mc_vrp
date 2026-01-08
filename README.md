# `rust_mc_vrp`

Multi-criteria **Capacitated Vehicle Routing Problem (CVRP)** solver implemented in Rust, using an **NSGA-II** style evolutionary algorithm.

The program:
- Loads a directed graph from a CSV file (each edge has **two cost parameters**).
- Randomly selects a **depot** and a set of **goal/customer nodes**, assigns random **demands**.
- Builds a complete graph over depot+goals (via shortest paths / multi-criteria costs).
- Runs NSGA-II to produce a population of candidate CVRP solutions.

## Quick start

Build:

```bash
cargo build --release
```

Run (example):

```bash
cargo run --release -- \
  --graph-path art/u500/graph_000_n500_c0_u1_seed85402.csv \
  --num-goals 200 \
  --max-demand 10 \
  --population-size 50 \
  --vehicle-capacity 50 \
  --p-crossover 0.5 \
  --p-mutation 0.1 \
  --runs 3
```

Write output to a file:

```bash
cargo run --release -- --output out.txt
```

## CLI options

Run `--help` to see all flags:

```bash
cargo run -- --help
```

Key flags (see `src/main.rs`):
- **`--graph-path` / `-g`**: Path to the input graph CSV.
- **`--num-goals` / `-n`**: Number of customer/goal nodes to sample (excluding depot).
- **`--max-demand` / `-m`**: Max demand assigned to each sampled goal (demands are uniform random in `1..=max_demand`).
- **`--population-size` / `-p`**: NSGA-II population size.
- **`--vehicle-capacity` / `-v`**: Vehicle capacity constraint for CVRP.
- **`--p-crossover`**: Crossover probability.
- **`--p-mutation`**: Mutation probability.
- **`--runs`**: Number of independent experiments to run (seeds are incremented per run).
- **`--seed`**: Base RNG seed (omit to use current time).
- **`--output`**: Write logs/results to a file instead of stdout.

## Input graph CSV format

The loader is implemented in `src/graph.rs`.

Expected format:
- **Line 1**: `num_nodes,num_edges`
- **Line 2**: header/ignored (present in the provided datasets)
- **Remaining lines**: one edge per line, with four comma-separated values:
  - `source,target,cost_1,cost_2`

Notes:
- Node ids are expected to be `0..num_nodes-1`.
- Each edge has two cost parameters; values are clamped to at least `1.0` in the parser.

## Output

For each run, the program prints:
- Selected depot id
- Selected goals and assigned demands
- NSGA parameters
- Solver progress / final population summary (from the NSGA implementation)

Use `--seed` to make runs reproducible.

## Repository layout

- `src/main.rs`: CLI + experiment harness (sampling depot/goals/demands, running multiple seeds)
- `src/graph.rs`: CSV graph loader
- `src/complete_graph.rs`: builds a complete graph over depot+goals
- `src/nsga.rs`: NSGA-II implementation for multi-criteria CVRP
- `art/`: example graph datasets
