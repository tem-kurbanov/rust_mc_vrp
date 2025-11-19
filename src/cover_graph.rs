use std::collections::{HashMap, HashSet};
use std::rc::Weak;
use crate::graph::Graph;

struct CoverEdge {
    nodes: Vec<u32>,
    parameters: (i32, i32),
}

struct Label {
    node: u32,
    parameters: (i32, i32),
    parent: Option<Weak<Label>>,
    used_edge: u32,
}

pub struct CoverGraph {
    num_nodes: u32,
    num_edges: u32,
    num_parameters: u32,

    preceding_cover_graph: Option<Weak<CoverGraph>>,

    cover_nodes: HashSet<u32>,
    cover_edges: Vec<CoverEdge>,

    outgoing_cover_edges: HashMap<u32, HashMap<u32, Vec<u32>>>,
    incoming_cover_edges: HashMap<u32, HashMap<u32, Vec<u32>>>,

    goals: HashSet<u32>,
    depot: u32,

    original_edge_points: Vec<(u32, u32)>,
}

impl CoverGraph {
    pub fn new(original_graph: &Graph, depot: u32, goals: &HashSet<u32>) -> Self {
        let mut original_edge_points = Vec::with_capacity(original_graph.get_num_edges() as usize);
        for i in 0..original_graph.get_num_edges() {
            original_edge_points.push(original_graph.get_edge_points(i));
        }
        let preceding_cover_graph: Option<Weak<CoverGraph>> = None;
        let num_nodes = original_graph.get_num_nodes();
        let num_edges = original_graph.get_num_edges();
        let num_parameters = original_graph.get_num_parameters();
        let mut cover_nodes = HashSet::new();
        for i in 0..num_nodes {
            cover_nodes.insert(i);
        }
        let mut cover_edges = Vec::new();
        for i in 0..num_edges {
            cover_edges.push(CoverEdge {
                nodes: vec![original_graph.get_edge_points(i).0, original_graph.get_edge_points(i).1],
                parameters: original_graph.get_edge_parameters(i).clone(),
            });
        }

        let mut outgoing_cover_edges = HashMap::new();
        for i in 0..num_edges {
            let (first, second) = original_graph.get_edge_points(i);
            outgoing_cover_edges
                .entry(first)
                .or_insert_with(HashMap::new)
                .entry(second)
                .or_insert_with(Vec::new)
                .push(i);
        }
        let mut incoming_cover_edges = HashMap::new();
        for i in 0..num_edges {
            let (first, second) = original_graph.get_edge_points(i);
            incoming_cover_edges
                .entry(second)
                .or_insert_with(HashMap::new)
                .entry(first)
                .or_insert_with(Vec::new)
                .push(i);
        }

        let goals = goals.clone();
        let depot = depot;
        

        Self {
            num_nodes,
            num_edges,
            num_parameters,
            preceding_cover_graph,
            cover_nodes,
            cover_edges,
            outgoing_cover_edges,
            incoming_cover_edges,
            goals,
            depot,
            original_edge_points,
        }
    }

    pub fn successor(degree: u32, preceding_cover_graph: Option<Weak<CoverGraph>>) -> Self {

    }

    pub fn k_path_cover() {

    }

    pub fn get_goals(&self) -> &HashSet<u32> {
        &self.goals
    }

    pub fn get_depot(&self) -> u32 {
        self.depot
    }

    pub fn get_cover_nodes(&self) -> &HashSet<u32> {
        &self.cover_nodes
    }
    
    pub fn get_cover_edges(&self) -> &Vec<CoverEdge> {
        &self.cover_edges
    }

    pub fn get_outgoing_cover_edges(&self, node_id: u32) -> &HashMap<u32, Vec<u32>> {
        &self.outgoing_cover_edges[&node_id]
    }
    
    pub fn get_incoming_cover_edges(&self, node_id: u32) -> &HashMap<u32, Vec<u32>> {
        &self.incoming_cover_edges[&node_id]
    }

    pub fn get_original_edge_points(&self, edge_id: u32) -> &(u32, u32) {
        &self.original_edge_points[edge_id as usize]
    }

    pub fn get_num_nodes(&self) -> u32 {
        self.num_nodes
    }
    
    pub fn get_num_edges(&self) -> u32 {
        self.num_edges
    }

    fn link_paths(current_criteria: &(u32, u32), suffix_criteria: &(u32, u32)) -> (u32, u32) {
        (current_criteria.0 + suffix_criteria.0, current_criteria.1 + suffix_criteria.1)
    }

    fn check_domination(vector1: &(u32, u32), vector2: &(u32, u32)) -> bool {
        vector1.0 <= vector2.0 && vector1.1 <= vector2.1
    }

    fn t_kpc_mls(source: u32) -> HashMap<u32, HashSet<Label>> {
        
    }

    fn update_tset(tset: &mut HashSet<u32>, label: &Label) {
        
    }

    fn t_discard(closed_labels: &HashSet<Label>, tset: &HashSet<u32>, candidate: (u32, u32)) -> bool {
        
    }

    fn check_tdomination(t: &u32, v: &(u32, u32)) -> bool {

    }

}