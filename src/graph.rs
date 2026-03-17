//! TSPLIB-like CVRP instance parser and in-memory graph model.
//!
//! This module parses a subset of the Uchoa et al. X-set `.vrp` format:
//! - `DIMENSION`: number of nodes (including depot)
//! - `CAPACITY`: vehicle capacity
//! - `NODE_COORD_SECTION`: node coordinates as `id x y`
//! - `DEMAND_SECTION`: demands as `id demand`
//! - `DEPOT_SECTION`: depot id list terminated by `-1`
//!
//! ### Indexing convention
//! Files are typically **1-based** (node ids start at 1). Internally we use **0-based** ids.
//! So file id `1` becomes internal id `0`, etc.
//!
//! ### Edges and objectives
//! We materialize a **complete directed graph** (edges `i -> j` for all `i != j`).
//! Each edge has **two parameters**:
//! 1. Euclidean distance between node coordinates (objective 0)
//! 2. A random integer in `[1, 100]` stored as `f64` (objective 1 / secondary cost)
//!
//! The solver currently uses the first edge variant (`[0]`) for each `(i, j)`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{SystemTime, UNIX_EPOCH};
use std::fmt::{self, Debug};

/// Directed edge in the complete graph.
///
/// `parameters.0` is Euclidean distance, `parameters.1` is an additional random cost.
#[derive(Clone)]
pub struct Edge {
    id: u32,
    source: u32,
    target: u32,
    parameters: (f64, f64),
}

impl Edge {
    /// Stable id of this edge (dense, starting at 0 in creation order).
    pub fn get_id(&self) -> u32 {
        self.id
    }
    /// Source node id (0-based).
    pub fn get_source(&self) -> u32 {
        self.source
    }
    /// Target node id (0-based).
    pub fn get_target(&self) -> u32 {
        self.target
    }
    /// Returns `(distance, secondary_cost)`.
    pub fn get_parameters(&self) -> (f64, f64) {
        self.parameters
    }
}

/// Parsed CVRP instance and complete directed graph representation.
///
/// Nodes are indexed `0..num_nodes`, with `depot` being one of those ids.
pub struct Graph {
    num_nodes: u32,
    num_edges: u32,
    num_parameters: u32,
    demands: HashMap<u32, u32>,

    depot: u32,
    capacity: u32,

    goal_nodes:HashSet<u32>,

    edges: Vec<Edge>,
    outgoing_edges: HashMap<usize, Vec<u32>>,
    incoming_edges: HashMap<usize, Vec<u32>>,
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Self {
            num_nodes: self.num_nodes,
            num_edges: self.num_edges,
            num_parameters: self.num_parameters,
            demands: self.demands.clone(),
            depot: self.depot,
            capacity: self.capacity,
            goal_nodes: self.goal_nodes.clone(),
            edges: self.edges.clone(),
            outgoing_edges: self.outgoing_edges.clone().into_iter().map(|(k, v)| (k, v.clone())).collect(),
            incoming_edges: self.incoming_edges.clone().into_iter().map(|(k, v)| (k, v.clone())).collect(),
        }
    }
}

impl Debug for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph {{ num_nodes: {}, num_edges: {}, num_parameters: {}, demands: {:?}, depot: {}, capacity: {}, goal_nodes: {:?}, outgoing_edges: {:?}, incoming_edges: {:?}}}", self.num_nodes, self.num_edges, self.num_parameters, self.demands, self.depot, self.capacity, self.goal_nodes, self.outgoing_edges, self.incoming_edges)?;
        writeln!(f, "Nodes:")?;
        for (node, demand) in self.demands.iter() {
            writeln!(f, "Node {}: demand {}", node, demand)?;
        }
        Ok(())
    }
}

impl Graph {

    /// Parse a TSPLIB-like `.vrp` file and build a complete directed graph.
    ///
    /// Notes:
    /// - Input ids are converted from 1-based to 0-based.
    /// - `NODE_COORD_SECTION` is stored in a dense vector by node id, so ids may be read out of order.
    /// - Only the **first** depot id found is used.
    pub fn new(input_path: &str) -> io::Result<Self> {
        let num_parameters = 2;
        let file = File::open(input_path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Parse header fields
        let mut dimension = 0u32;
        let mut capacity = 0u32;
        let mut node_coords: Vec<(f64, f64)> = Vec::new();
        let mut demands: HashMap<u32, u32> = HashMap::new();
        let mut depot: Option<u32> = None;
        let mut coords_read = 0usize;

        let mut section = String::new();
        // Create RNG with time-based seed
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

        while let Some(line) = lines.next() {
            let line = line?;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            // Parse header fields
            if trimmed.starts_with("DIMENSION") {
                let parts: Vec<&str> = trimmed.split(':').collect();
                if parts.len() >= 2 {
                    dimension = parts[1].trim().parse().unwrap_or(0);
                }
            } else if trimmed.starts_with("CAPACITY") {
                let parts: Vec<&str> = trimmed.split(':').collect();
                if parts.len() >= 2 {
                    capacity = parts[1].trim().parse().unwrap_or(0);
                }
            } else if trimmed == "NODE_COORD_SECTION" {
                section = "NODE_COORD_SECTION".to_string();
            } else if trimmed == "DEMAND_SECTION" {
                section = "DEMAND_SECTION".to_string();
            } else if trimmed == "DEPOT_SECTION" {
                section = "DEPOT_SECTION".to_string();
            } else if trimmed == "EOF" {
                break;
            } else if section == "NODE_COORD_SECTION" {
                // Parse node coordinates: "id x y"
                // IDs in file are 1-based, convert to 0-based
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 3 {
                    let file_id: u32 = parts[0].parse().unwrap_or(0);
                    if file_id == 0 {
                        continue; // Skip invalid IDs
                    }
                    let id = file_id - 1; // Convert to 0-based (file IDs start at 1)
                    let x: f64 = parts[1].parse().unwrap_or(0.0);
                    let y: f64 = parts[2].parse().unwrap_or(0.0);
                    // Ensure we have enough space
                    if id as usize >= node_coords.len() {
                        node_coords.resize(id as usize + 1, (0.0, 0.0));
                    }
                    node_coords[id as usize] = (x, y);
                    coords_read += 1;
                }
            } else if section == "DEMAND_SECTION" {
                // Parse demands: "id demand"
                // IDs in file are 1-based, convert to 0-based
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let file_id: u32 = parts[0].parse().unwrap_or(0);
                    if file_id == 0 {
                        continue; // Skip invalid IDs
                    }
                    let id = file_id - 1; // Convert to 0-based (file IDs start at 1)
                    let demand: u32 = parts[1].parse().unwrap_or(0);
                    demands.insert(id, demand);
                }
            } else if section == "DEPOT_SECTION" {
                // Parse depot: "id" or "-1" to terminate
                // IDs in file are 1-based, convert to 0-based
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 1 {
                    let id_str = parts[0].trim();
                    if id_str == "-1" {
                        break;
                    } else if let Ok(file_id) = id_str.parse::<u32>() {
                        if file_id > 0 {
                            let id = file_id - 1; // Convert to 0-based (file IDs start at 1)
                            depot = Some(id);
                        }
                    }
                }
            }
        }

        // Ensure node_coords is properly sized (should have dimension nodes, 0-indexed)
        if node_coords.len() < dimension as usize {
            node_coords.resize(dimension as usize, (0.0, 0.0));
        }
        
        // Verify we read the expected number of node coordinates
        if coords_read != dimension as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Expected {} node coordinates but found {}", dimension, coords_read),
            ));
        }
        
        // Verify depot is valid
        if let Some(depot_id) = depot {
            if depot_id >= dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Depot ID {} is out of range (max: {})", depot_id, dimension - 1),
                ));
            }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "No depot specified in DEPOT_SECTION",
            ));
        }

        // Create complete graph: edges between all pairs of nodes
        let num_nodes = dimension;
        let mut edges = Vec::new();
        let mut outgoing_edges = HashMap::new();
        let mut incoming_edges = HashMap::new();
        let mut edge_id = 0u32;

        let mut goal_nodes = HashSet::new();
        for n in 1..num_nodes {
            goal_nodes.insert(n);
        }

        // Create edges for all pairs (complete graph)
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j {
                    // Calculate Euclidean distance
                    let (x1, y1) = node_coords[i as usize];
                    let (x2, y2) = node_coords[j as usize];
                    let distance = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt();

                    // Random number from 1 to 100
                    let random_param = rng.random_range(1..=100) as f64;

                    edges.push(Edge {
                        id: edge_id,
                        source: i,
                        target: j,
                        parameters: (distance, random_param),
                    });

                    outgoing_edges.entry(i as usize).or_insert(Vec::new()).push(edge_id);
                    incoming_edges.entry(j as usize).or_insert(Vec::new()).push(edge_id);
                    edge_id += 1;
                }
            }
        }

        let num_edges = edge_id;

        // Print all edges with their parameters in an orderly manner
        // println!("Edge parameters:");
        // println!("Edge Start\tEdge End\tParameter 1 (Distance)\tParameter 2 (Random)");
        // for edge in &edges {
        //     println!("{}\t\t{}\t\t{:.6}\t\t{:.0}", 
        //         edge.source, 
        //         edge.target, 
        //         edge.parameters.0, 
        //         edge.parameters.1
        //     );
        // }

        Ok(Graph {
            num_nodes,
            num_edges,
            num_parameters,
            demands,
            depot: depot.unwrap(),
            capacity,
            goal_nodes,
            edges,
            outgoing_edges,
            incoming_edges,
        })
    }

    pub fn new_subgraph(original_graph: &Graph, route: &Vec<usize>) -> Self {
        let num_parameters = original_graph.get_num_parameters();
        let capacity = original_graph.get_capacity();
        let depot = original_graph.get_depot();
        let mut demands = HashMap::new();
        let mut edges = Vec::new();
        let mut outgoing_edges: HashMap<usize, Vec<u32>> = HashMap::new();
        let mut incoming_edges: HashMap<usize, Vec<u32>> = HashMap::new();

        let mut goal_nodes = HashSet::new();

        for node in route {
            demands.insert((*node as u32) - 1, original_graph.get_demand((*node as u32) - 1));
            goal_nodes.insert((*node as u32) - 1);
        }

        demands.insert(depot, 0);
        let mut edge_id = 0;
        for edge in original_graph.get_edges() {
            let source = edge.get_source();
            let target = edge.get_target();
            if demands.contains_key(&source) && demands.contains_key(&target) {
                let mut match_edge = edge.clone();
                match_edge.id = edge_id;
                edges.push(match_edge);
                outgoing_edges.entry(edge.get_source() as usize).or_insert(Vec::new()).push(edge_id);
                incoming_edges.entry(edge.get_target() as usize).or_insert(Vec::new()).push(edge_id);
                edge_id += 1;
            }
        }

        return Graph {
            num_nodes: route.len() as u32 + 1,
            num_edges: edges.len() as u32,
            num_parameters,
            demands,
            depot,
            capacity,
            goal_nodes,
            edges,
            outgoing_edges,
            incoming_edges,
        };

    }

    /// Number of nodes (including depot).
    pub fn get_num_nodes(&self) -> u32 {
        self.num_nodes
    }

    pub fn get_num_edges(&self) -> u32 {
        self.num_edges
    }

    pub fn get_num_parameters(&self) -> u32 {
        self.num_parameters
    }

    /// Demand for a customer node id (0-based).
    ///
    /// Panics if the node id was not present in the `DEMAND_SECTION`.
    pub fn get_demand(&self, node_id: u32) -> u32 {
        self.demands[&node_id]
    }

    /// Depot node id (0-based).
    pub fn get_depot(&self) -> u32 {
        self.depot
    }

    /// Vehicle capacity from the instance file.
    pub fn get_capacity(&self) -> u32 {
        self.capacity
    }

    /// Returns `(source, target)` for the given edge id.
    pub fn get_edge_points(&self, edge_id: u32) -> (u32, u32) {
        (self.edges[edge_id as usize].source, self.edges[edge_id as usize].target)
    }

    /// Returns `(distance, secondary_cost)` for the given edge id.
    pub fn get_edge_parameters(&self, edge_id: u32) -> &(f64, f64) {
        &self.edges[edge_id as usize].parameters
    }

    pub fn get_outgoing_edges(&self, node_id: u32) -> &Vec<u32> {
        &self.outgoing_edges[&(node_id as usize)]
    }

    pub fn get_incoming_edges(&self, node_id: u32) -> &Vec<u32> {
        &self.incoming_edges[&(node_id as usize)]
    }

    /// All edges in the complete graph, in creation order.
    pub fn get_edges(&self) -> &Vec<Edge> {
        &self.edges
    }

    pub fn get_goal_nodes(&self) -> &HashSet<u32> {
        &self.goal_nodes
    }
}