//! Binary entrypoint for `rust_mc_vrp`.
//!
//! Parses a `.vrp` or precomputed graph `.json`, runs NSGA-II, logs progress and final front.

use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::exit;

use clap::Parser;

use rust_mc_vrp::graph::Graph;
use rust_mc_vrp::nsga::NSGA;

#[derive(Parser)]
#[command(author, version, about = "Solve CVRP using NSGA-II")]
struct Config {
    /// Path to a TSPLIB-like `.vrp` file or a graph `.json` from `export_xset_graphs.py`.
    #[arg(short, long, default_value = "xset/X-n106-k14.vrp")]
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
    let cfg = Config::parse();

    if cfg.runs == 0 {
        eprintln!("--runs must be >= 1");
        exit(1);
    }

    let original_graph = if cfg.graph_path.to_ascii_lowercase().ends_with(".json") {
        match Graph::from_json_path(&cfg.graph_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to load JSON graph '{}': {}", cfg.graph_path, e);
                exit(1);
            }
        }
    } else {
        match Graph::new(&cfg.graph_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to open graph '{}': {}", cfg.graph_path, e);
                exit(1);
            }
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

        let run_seed = run_idx as u64;
        writeln!(
            w,
            "NSGA params: population_size={}, p_crossover={}, p_mutation={}, seed={}",
            cfg.population_size, cfg.p_crossover, cfg.p_mutation, run_seed
        )
        .ok();

        let mut nsga = NSGA::new(
            original_graph.clone(),
            cfg.population_size,
            cfg.p_crossover,
            cfg.p_mutation,
            run_seed,
        );

        let _outcome = nsga.solve_capacitated_vrp_with_writer(w);
        writeln!(w).ok();
    }
}
