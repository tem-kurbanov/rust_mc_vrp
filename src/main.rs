mod graph;
//mod shrink_graph;
//mod cover_graph;

use graph::Graph;

fn main() {
    let original_graph = Graph::new("./art/c100/graph_000_n100_c1_u0_seed99181.csv").unwrap();
    let num_nodes = original_graph.get_num_nodes();
    println!("Number of nodes: {}", num_nodes);
}
