use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

mod complete_graph;
mod graph;
mod nsga;

use complete_graph::CompleteGraph;
use graph::Graph;
use nsga::NSGA;

#[derive(Parser)]
#[command(author, version, about = "Solve CVRP using NSGA-II")]
struct Config {
    #[arg(
        short,
        long,
        default_value = "art/u500/graph_000_n500_c0_u1_seed85402.csv",
        help = "Path to graph CSV (required)"
    )]
    graph_path: String,

    #[arg(short, long, default_value = "200", help = "Number of goals to randomly select (required)")]
    num_goals: usize,

    #[arg(short, long, default_value = "10", help = "Maximum demand per goal (required)")]
    max_demand: u32,

    #[arg(short, long, default_value = "10", help = "Population size")]
    population_size: u32,

    #[arg(
        short,
        long,
        default_value = "50",
        help = "Vehicle capacity (default: 50)"
    )]
    vehicle_capacity: u32,

    #[arg(long, default_value = "0.5", help = "Crossover probability")]
    p_crossover: f64,

    #[arg(long, default_value = "0.1", help = "Mutation probability")]
    p_mutation: f64,

    #[arg(long, help = "Random seed")]
    seed: Option<u64>,

    #[arg(long, default_value = "5", help = "Number of experiments (independent runs) to execute")]
    runs: u32,

    #[arg(long, help = "Write program output to this file (otherwise stdout)")]
    output: Option<PathBuf>,
}

fn main() {
    // Parse named command-line flags
    let cfg = Config::parse();

    if cfg.runs == 0 {
        eprintln!("--runs must be >= 1");
        exit(1);
    }

    // Read original graph
    let original_graph = match Graph::new(&cfg.graph_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open graph '{}': {}", cfg.graph_path, e);
            exit(1);
        }
    };

    let num_nodes = original_graph.get_num_nodes() as usize;
    if num_nodes == 0 {
        eprintln!("Graph has no nodes");
        exit(1);
    }
    if cfg.num_goals + 1 > num_nodes {
        eprintln!(
            "Requested {} goals but graph only has {} nodes; need at least goals + depot",
            cfg.num_goals, num_nodes
        );
        exit(1);
    }

    // Determine vehicle capacity to pass to NSGA
    let vehicle_capacity: u32 = cfg.vehicle_capacity;

    let base_seed: u64 = cfg.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    if let Some(path) = &cfg.output {
        let file = match File::create(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create output file '{}': {}", path.display(), e);
                exit(1);
            }
        };
        let mut w = io::BufWriter::new(file);
        run_experiments(&cfg, &original_graph, num_nodes, vehicle_capacity, base_seed, &mut w);
        let _ = w.flush();
    } else {
        let stdout = io::stdout();
        let mut w = io::BufWriter::new(stdout.lock());
        run_experiments(&cfg, &original_graph, num_nodes, vehicle_capacity, base_seed, &mut w);
        let _ = w.flush();
    }
}

fn run_experiments<W: Write>(
    cfg: &Config,
    original_graph: &Graph,
    num_nodes: usize,
    vehicle_capacity: u32,
    base_seed: u64,
    w: &mut W,
) {
    for run_idx in 0..cfg.runs {
        let seed = base_seed + (run_idx as u64);
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

        writeln!(w, "=== Experiment {}/{} (seed={}) ===", run_idx + 1, cfg.runs, seed).ok();
        writeln!(w, "Graph: {}", cfg.graph_path).ok();
        writeln!(w, "Number of graph nodes: {}", num_nodes).ok();

        // Select a random depot
        let depot: u32 = rng.random_range(0..(num_nodes as u32));

        // Select unique goal node ids different from depot
        let mut unique_goal_nodes: BTreeSet<u32> = BTreeSet::new();
        while unique_goal_nodes.len() < cfg.num_goals {
            let candidate = rng.random_range(0..(num_nodes as u32));
            if candidate == depot {
                continue;
            }
            unique_goal_nodes.insert(candidate);
        }

        // Assign random demands to each unique goal node
        let mut selected_goals: BTreeSet<(u32, u32)> = BTreeSet::new();
        for node_id in unique_goal_nodes {
            let demand = if cfg.max_demand == 0 {
                0
            } else {
                rng.random_range(1..=cfg.max_demand)
            };
            selected_goals.insert((node_id, demand));
        }

        writeln!(w, "Selected depot node: {}", depot).ok();
        writeln!(w, "Selected {} goals (node -> demand):", selected_goals.len()).ok();
        for (n, d) in &selected_goals {
            writeln!(w, "  {} -> {}", n, d).ok();
        }
        writeln!(
            w,
            "NSGA params: population_size={}, vehicle_capacity={}, p_crossover={}, p_mutation={}",
            cfg.population_size, vehicle_capacity, cfg.p_crossover, cfg.p_mutation
        )
        .ok();

        // Build complete graph (multicriteria planning between depot and goals)
        let complete_graph = CompleteGraph::new(original_graph, depot, &selected_goals);

        // Create NSGA instance
        let nsga = NSGA::new_parallel(
            complete_graph,
            cfg.population_size,
            vehicle_capacity,
            cfg.p_crossover,
            cfg.p_mutation,
        );

        // Solve CVRP (solver logs to `w`)
        let _population = nsga.solve_capacitated_vrp_with_writer(w);
        writeln!(w).ok();
    }
}
