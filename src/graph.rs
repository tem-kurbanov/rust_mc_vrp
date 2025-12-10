use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

struct Edge {
    id: u32,
    source: u32,
    target: u32,
    parameters: (i32, i32),
}

pub struct Graph {
    num_nodes: u32,
    num_edges: u32,
    num_parameters: u32,

    edges: Vec<Edge>,
    outgoing_edges: Vec<Vec<u32>>,
    incoming_edges: Vec<Vec<u32>>,
    restructure_ids: HashMap<u32, u32>,
}

impl Graph {

    pub fn new(input_path: &str) -> io::Result<Self> {
        let file = File::open(input_path).expect("Failed to open file");
        let mut reader = BufReader::new(file);

        let mut buf = String::new();
        reader.read_line(&mut buf)?;
        let mut buf = buf.trim().split(',');
        // read "100,280" to num_nodes and num_edges
        let num_nodes = buf.next().unwrap().parse::<u32>().unwrap();
        let num_edges = buf.next().unwrap().parse::<u32>().unwrap();

        let num_parameters = 2;

        let mut edges = Vec::with_capacity(num_edges as usize);

        let mut outgoing_edges: Vec<Vec<u32>> = vec![Vec::new(); num_nodes as usize];
        let mut incoming_edges: Vec<Vec<u32>> = vec![Vec::new(); num_nodes as usize];

        let mut buf = String::new();
        reader.read_line(&mut buf)?;

        let mut edge_id = 0;
        for line_result in reader.lines() {
            let line = line_result?;
            let parts = line.trim().split(',').map(|n| {
                let n = n.parse::<i32>().unwrap();
                n
            }).collect::<Vec<i32>>();

            let start = parts[0];
            let end = parts[1];
            let p1 = parts[2].max(1);
            let p2 = parts[3].max(1);

            edges.push(Edge{id: edge_id, source: start as u32, target: end as u32, parameters: (p1, p2)});
            outgoing_edges[start as usize].push(edge_id);
            incoming_edges[end as usize].push(edge_id);

            edge_id += 1;
        }

        let restructure_ids = HashMap::new();

        Ok(Self{
            num_nodes,
            num_edges,
            num_parameters,
            edges,
            outgoing_edges,
            incoming_edges,
            restructure_ids
        })
    }

    pub fn get_num_nodes(&self) -> u32 {
        self.num_nodes
    }

    pub fn get_num_edges(&self) -> u32 {
        self.num_edges
    }

    pub fn get_num_parameters(&self) -> u32 {
        self.num_parameters
    }

    pub fn get_edge_points(&self, edge_id: u32) -> (u32, u32) {
        (self.edges[edge_id as usize].source, self.edges[edge_id as usize].target)
    }

    pub fn get_edge_parameters(&self, edge_id: u32) -> &(i32, i32) {
        &self.edges[edge_id as usize].parameters
    }

    pub fn get_outgoing_edges(&self, node_id: u32) -> &Vec<u32> {
        &self.outgoing_edges[node_id as usize]
    }

    pub fn get_incoming_edges(&self, node_id: u32) -> &Vec<u32> {
        &self.incoming_edges[node_id as usize]
    }

    pub fn get_restructure_id(&self, node_id: u32) -> i32 {
        if self.restructure_ids.contains_key(&node_id) {
            self.restructure_ids[&node_id] as i32
        } else {
            -1
        }
    }

    pub fn augment_graph(&mut self, goals: &HashSet<u32>) {
        for g in goals {
            let new_id = self.num_nodes;

            for out_edge in &self.outgoing_edges[*g as usize] {
                self.edges[*out_edge as usize].source = new_id;
            }
            for in_edge in &self.incoming_edges[*g as usize] {
                self.edges[*in_edge as usize].target = new_id;
            }

            let new_out = self.outgoing_edges[*g as usize].to_vec();
            let new_in = self.incoming_edges[*g as usize].to_vec();

            self.outgoing_edges.push(new_out);
            self.incoming_edges.push(new_in);

            self.outgoing_edges[*g as usize] = vec![self.num_edges];
            self.incoming_edges[*g as usize] = vec![self.num_edges + 1];

            self.outgoing_edges[new_id as usize].push(self.num_edges + 1);
            self.incoming_edges[new_id as usize].push(self.num_edges);

            self.edges.push(Edge{id: self.num_edges, source: *g, target: new_id, parameters: (1, 1)});
            self.edges.push(Edge{id: self.num_edges + 1, source: new_id, target: *g, parameters: (1, 1)});

            self.num_nodes += 1;
            self.num_edges += 2;

            self.restructure_ids.insert(new_id, *g);
        }
    }
}