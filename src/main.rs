mod graph;
mod complete_graph;

use graph::Graph;
use complete_graph::CompleteGraph;
use std::collections::HashSet;

fn main() {
    let original_graph = Graph::new("./art/c100/graph_000_n100_c1_u0_seed99181.csv").unwrap();
    let complete_graph = CompleteGraph::new(&original_graph, 0, &HashSet::from([1, 2, 3, 4, 5]));
    println!("Done");
}
