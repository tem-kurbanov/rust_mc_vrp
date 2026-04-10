//! Batch-generate multi-objective CVRP label files (JSON in `.sol`) from graph JSON instances.
//!
//! Run: `cargo run --bin generate_labels --release -- --graphs-dir ... --output-dir ...`

use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::exit;

use clap::Parser;
use serde::Serialize;

use rust_mc_vrp::graph::Graph;
use rust_mc_vrp::nsga::NSGA;

#[derive(Parser)]
#[command(author, version, about = "Export NSGA Pareto-style solutions as JSON .sol label files")]
struct Config {
    /// Directory of graph JSON files (`*.json`).
    #[arg(long)]
    graphs_dir: PathBuf,

    /// Root output directory (one subfolder per instance stem).
    #[arg(long, default_value = "nsga_labels")]
    output_dir: PathBuf,

    /// Comma-separated population sizes.
    #[arg(long, default_value = "50,100")]
    pop_grid: String,

    /// Comma-separated crossover probabilities.
    #[arg(long, default_value = "0.5,0.7,0.9")]
    cross_grid: String,

    /// Comma-separated mutation probabilities.
    #[arg(long, default_value = "0.05,0.1,0.2")]
    mut_grid: String,

    /// Comma-separated RNG seeds.
    #[arg(long, default_value = "0,1,2")]
    seeds_grid: String,
}

#[derive(Serialize)]
struct SolutionFileJson {
    instance_stem: String,
    graph_source: String,
    depot: u32,
    capacity: u32,
    num_nodes: u32,
    population_size: u32,
    p_crossover: f64,
    p_mutation: f64,
    rng_seed: u64,
    generations: u32,
    converged_early: bool,
    num_solutions: usize,
    solutions: Vec<SolutionEntryJson>,
}

#[derive(Serialize)]
struct SolutionEntryJson {
    f0: f64,
    f1: f64,
    routes_customers_0based: Vec<Vec<u32>>,
}

fn main() {
    let cfg = Config::parse();
    if let Err(e) = run(&cfg) {
        eprintln!("{}", e);
        exit(1);
    }
}

fn parse_csv_u32(s: &str) -> Result<Vec<u32>, String> {
    s.split(',')
        .map(|t| t.trim().parse::<u32>().map_err(|e| e.to_string()))
        .collect()
}

fn parse_csv_u64(s: &str) -> Result<Vec<u64>, String> {
    s.split(',')
        .map(|t| t.trim().parse::<u64>().map_err(|e| e.to_string()))
        .collect()
}

fn parse_csv_f64(s: &str) -> Result<Vec<f64>, String> {
    s.split(',')
        .map(|t| t.trim().parse::<f64>().map_err(|e| e.to_string()))
        .collect()
}

fn param_token(v: f64) -> String {
    format!("{:.2}", v).replace('.', "p")
}

fn run(cfg: &Config) -> Result<(), String> {
    if !cfg.graphs_dir.is_dir() {
        return Err(format!("'{}' is not a directory", cfg.graphs_dir.display()));
    }

    let pops = parse_csv_u32(&cfg.pop_grid)?;
    let crosses = parse_csv_f64(&cfg.cross_grid)?;
    let muts = parse_csv_f64(&cfg.mut_grid)?;
    let seeds = parse_csv_u64(&cfg.seeds_grid)?;

    if pops.is_empty() || crosses.is_empty() || muts.is_empty() || seeds.is_empty() {
        return Err("parameter grids must be non-empty".into());
    }

    let mut json_files: Vec<PathBuf> = fs::read_dir(&cfg.graphs_dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("json"))
                .unwrap_or(false)
        })
        .collect();
    json_files.sort();

    if json_files.is_empty() {
        return Err(format!("No .json files in '{}'", cfg.graphs_dir.display()));
    }

    fs::create_dir_all(&cfg.output_dir).map_err(|e| e.to_string())?;

    for path in &json_files {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| format!("bad file name: {}", path.display()))?;

        let out_inst = cfg.output_dir.join(stem);
        fs::create_dir_all(&out_inst).map_err(|e| e.to_string())?;

        eprintln!("[generate_labels] {}", stem);

        for &pop in &pops {
            for &pc in &crosses {
                for &pm in &muts {
                    for &seed in &seeds {
                        let graph = Graph::from_json_path(path).map_err(|e| e.to_string())?;
                        let depot = graph.get_depot();
                        let capacity = graph.get_capacity();
                        let num_nodes = graph.get_num_nodes();
                        let mut nsga = NSGA::new(graph, pop, pc, pm, seed);
                        let mut quiet = io::BufWriter::new(io::sink());
                        let outcome = nsga.solve_capacitated_vrp_with_writer(&mut quiet);
                        let exports = nsga.pareto_unique_exports(&outcome.population);

                        let solutions: Vec<SolutionEntryJson> = exports
                            .into_iter()
                            .map(|e| SolutionEntryJson {
                                f0: e.f0,
                                f1: e.f1,
                                routes_customers_0based: e.routes_customers_0based,
                            })
                            .collect();

                        let payload = SolutionFileJson {
                            instance_stem: stem.to_string(),
                            graph_source: path.display().to_string(),
                            depot,
                            capacity,
                            num_nodes,
                            population_size: pop,
                            p_crossover: pc,
                            p_mutation: pm,
                            rng_seed: seed,
                            generations: outcome.generations,
                            converged_early: outcome.converged_early,
                            num_solutions: solutions.len(),
                            solutions,
                        };

                        let fname = format!(
                            "{}_{}_{}_{}_{}.sol",
                            stem,
                            pop,
                            param_token(pc),
                            param_token(pm),
                            seed
                        );
                        let out_path = out_inst.join(fname);
                        let json =
                            serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?;
                        fs::write(&out_path, json).map_err(|e| e.to_string())?;
                    }
                }
            }
        }
    }

    Ok(())
}
