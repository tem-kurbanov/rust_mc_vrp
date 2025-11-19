use std::collections::HashSet;
use std::io::{self, Error, ErrorKind};

struct ShrinkEdge {
    id: u32,
    source: u32,
    target: u32,
    parameters: (i32, i32),
}

pub struct ShrinkGraph {
    num_nodes: u32,
    num_edges: u32,
    num_parameters: u32,

    edges: Vec<ShrinkEdge>,
    outgoing_edges: Vec<Vec<u32>>,
    incoming_edges: Vec<Vec<u32>>,
    
    depot: u32,
    goals: HashSet<u32>,
}

impl ShrinkGraph {
    
}