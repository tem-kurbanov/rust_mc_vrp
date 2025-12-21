use std::collections::BTreeSet;
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
    #[arg(short, long, default_value = "./art/c500/graph_007_n500_c1_u0_seed12968.csv", help = "Path to graph CSV (required)")]
    graph_path: String,

    #[arg(short, long, default_value = "20", help = "Number of goals to randomly select (required)")]
    num_goals: usize,

    #[arg(short, long, default_value = "10", help = "Maximum demand per goal (required)")]
    max_demand: u32,

    #[arg(short, long, default_value = "10", help = "Population size")]
    population_size: u32,

    #[arg(
        short,
        long,
        default_value = "10",
        help = "Vehicle capacity (default: 3 * max_demand if >0, else 10)"
    )]
    vehicle_capacity: u32,

    #[arg(long, default_value = "0.5", help = "Crossover probability")]
    p_crossover: f64,

    #[arg(long, default_value = "0.1", help = "Mutation probability")]
    p_mutation: f64,

    #[arg(long, help = "Random seed")]
    seed: Option<u64>,
}

fn main() {
    // Parse named command-line flags
    let cfg = Config::parse();

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

    // Setup RNG
    let mut rng: StdRng = match cfg.seed {
        Some(s) => SeedableRng::seed_from_u64(s),
        None => SeedableRng::seed_from_u64(10),
    };

    // Select a random depot
    let depot: u32 = rng.random_range(0..(num_nodes as u32));

    // Select unique goal node ids different from depot
    let mut selected_goals: BTreeSet<(u32, u32)> = BTreeSet::new();
    while selected_goals.len() < cfg.num_goals {
        let candidate = rng.random_range(0..(num_nodes as u32));
        if candidate == depot {
            continue;
        }
        // Random demand in 1..=max_demand
        let demand = if cfg.max_demand == 0 {
            0
        } else {
            rng.random_range(1..=cfg.max_demand)
        };
        selected_goals.insert((candidate, demand));
    }

    // Determine vehicle capacity to pass to NSGA
    let vehicle_capacity: u32 = cfg.vehicle_capacity;

    println!("Graph: {}", cfg.graph_path);
    println!("Number of graph nodes: {}", num_nodes);
    println!("Selected depot node: {}", depot);
    println!("Selected {} goals (node -> demand):", selected_goals.len());
    for (n, d) in &selected_goals {
        println!("  {} -> {}", n, d);
    }
    println!(
        "NSGA params: population_size={}, vehicle_capacity={}, p_crossover={}, p_mutation={}",
        cfg.population_size, vehicle_capacity, cfg.p_crossover, cfg.p_mutation
    );

    // Build complete graph (multicriteria planning between depot and goals)
    let complete_graph = CompleteGraph::new(&original_graph, depot, &selected_goals);

    // Create NSGA instance
    let nsga = NSGA::new_parallel(
        complete_graph,
        cfg.population_size,
        vehicle_capacity,
        cfg.p_crossover,
        cfg.p_mutation,
    );

    // Solve CVRP
    let solution = nsga.solve_capacitated_vrp();

    println!("Solution ({} chromosomes):", solution.len());
    for (i, chrom) in solution.iter().enumerate() {
        println!("Chromosome {}: {:?} {:?}", i, chrom.get_order_genes(), chrom.get_fitness_values());
    }
}
