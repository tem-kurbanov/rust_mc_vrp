use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// mod complete_graph;
mod graph;
mod nsga;

use graph::Graph;
use nsga::NSGA;

#[derive(Parser)]
#[command(author, version, about = "Solve CVRP using NSGA-II")]
struct Config {
    #[arg(short, long, default_value = "xset/X-n134-k13.vrp", help = "Path to graph CSV (required)")]
    graph_path: String,

    #[arg(short, long, default_value = "50", help = "Population size")]
    population_size: u32,


    #[arg(long, default_value = "0.5", help = "Crossover probability")]
    p_crossover: f64,

    #[arg(long, default_value = "0.1", help = "Mutation probability")]
    p_mutation: f64,

    #[arg(long, default_value = "1", help = "Number of experiments (independent runs) to execute")]
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
