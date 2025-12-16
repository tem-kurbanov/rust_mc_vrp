mod graph;
mod complete_graph;
mod nsga;

use graph::Graph;
use complete_graph::CompleteGraph;
use std::collections::{BTreeSet};

fn main() {
    let original_graph = Graph::new("./art/c100/graph_000_n100_c1_u0_seed99181.csv").unwrap();
    let complete_graph = CompleteGraph::new(&original_graph, 0, &BTreeSet::from([(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]));
    println!("Done");
}
