use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;

struct Edge {
    source: u32,
    target: u32,
    parameters: (i32, i32),
    outgoing_edges: Vec<u32>,
    incoming_edges: Vec<u32>,
}

struct Graph {
    num_nodes: u32,
    num_edges: u32,
    num_parameters: u32,

    edges: Vec<Edge>,
    restructure_ids: HashMap<u32, u32>,
}

impl Graph {

    fn new(input_path: &str) -> Self {
        let file = File::open(input_path).unwrap();
        let reader = BufReader::new(file);
        
        
    }

    fn get_num_nodes(&self) -> u32 {
        self.num_nodes
    }

    fn get_num_edges(&self) -> u32 {
        self.num_edges
    }

    fn get_num_parameters(&self) -> u32 {
        self.num_parameters
    }

    fn get_edge_points(&self, edge_id: u32) -> (u32, u32) {
        (self.edges[edge_id as usize].source, self.edges[edge_id as usize].target)
    }

    fn get_edge_parameters(&self, edge_id: u32) -> &(i32, i32) {
        &self.edges[edge_id as usize].parameters
    }

    fn get_outgoing_edges(&self, edge_id: u32) -> &Vec<u32> {
        &self.edges[edge_id as usize].outgoing_edges
    }

    fn get_incoming_edges(&self, edge_id: u32) -> &Vec<u32> {
        &self.edges[edge_id as usize].incoming_edges
    }

    fn get_restructure_id(&self, node_id: u32) -> i32 {
        if self.restructure_ids.contains_key(&node_id) {
            self.restructure_ids[&node_id] as i32
        } else {
            -1
        }
    }

    fn augment_graph(&self, goals: &HashSet<u32>) {
    
    }





}