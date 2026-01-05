//! Binary entrypoint for `rust_mc_vrp`.
//!
//! This crate currently focuses on solving **Capacitated VRP (CVRP)** instances provided in a
//! TSPLIB-like `.vrp` format (Uchoa et al. X-set style).
//!
//! The program:
//! - parses a `.vrp` instance into an internal complete directed graph with 2 edge parameters
//!   (distance + random secondary cost),
//! - runs an NSGA-II style evolutionary search on a permutation encoding of customers,
//! - logs convergence info and a final nondominated set summary.
//!
//! See `README.md` for usage and input format.

use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::exit;

use clap::Parser;

// mod complete_graph;
mod graph;
mod nsga;

use graph::Graph;
use nsga::NSGA;

#[derive(Parser)]
#[command(author, version, about = "Solve CVRP using NSGA-II")]
struct Config {
    /// Path to a TSPLIB-like `.vrp` file (Uchoa X-set style).
    #[arg(short, long, default_value = "xset/X-n819-k171.vrp")]
    graph_path: String,

    /// Number of individuals in the population.
    #[arg(short, long, default_value = "50")]
    population_size: u32,


    /// Crossover probability.
    #[arg(long, default_value = "0.5")]
    p_crossover: f64,

    /// Mutation probability.
    #[arg(long, default_value = "0.1")]
    p_mutation: f64,

    /// Number of independent runs to execute.
    #[arg(long, default_value = "1")]
    runs: u32,

    /// Write program output to this file (otherwise stdout).
    #[arg(long)]
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

    if let Some(path) = &cfg.output {
        let file = match File::create(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to create output file '{}': {}", path.display(), e);
                exit(1);
            }
        };
        let mut w = io::BufWriter::new(file);
        run_experiments(&cfg, &original_graph, num_nodes, &mut w);
        let _ = w.flush();
    } else {
        let stdout = io::stdout();
        let mut w = io::BufWriter::new(stdout.lock());
        run_experiments(&cfg, &original_graph, num_nodes, &mut w);
        let _ = w.flush();
    }
}

fn run_experiments<W: Write>(
    cfg: &Config,
    original_graph: &Graph,
    num_nodes: usize,
    w: &mut W,
) {
    for run_idx in 0..cfg.runs {
        writeln!(w, "=== Experiment {}/{} ===", run_idx + 1, cfg.runs).ok();
        writeln!(w, "Graph: {}", cfg.graph_path).ok();
        writeln!(w, "Number of graph nodes: {}", num_nodes).ok();


        writeln!(
            w,
            "NSGA params: population_size={}, p_crossover={}, p_mutation={}",
            cfg.population_size, cfg.p_crossover, cfg.p_mutation
        )
        .ok();

        // Create NSGA instance
        let nsga = NSGA::new(
            original_graph.clone(),
            cfg.population_size,
            cfg.p_crossover,
            cfg.p_mutation,
    );

        // Solve CVRP (solver logs to `w`)
        let _population = nsga.solve_capacitated_vrp_with_writer(w);
        writeln!(w).ok();
    }
}
