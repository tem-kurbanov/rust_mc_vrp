use std::collections::{HashMap, HashSet, BTreeSet};
use std::rc::{Rc};
use std::cmp::Ordering;

use crate::graph::Graph;


pub struct CoverEdge {
    id: u32,
    composite_nodes: Vec<u32>,
    composite_edges: Vec<u32>,
    parameters: (f64, f64),
}

impl CoverEdge {
    pub fn get_id(&self) -> u32 {
        self.id
    }
    pub fn get_composite_nodes(&self) -> &Vec<u32> {
        &self.composite_nodes
    }
    pub fn get_composite_edges(&self) -> &Vec<u32> {
        &self.composite_edges
    }
    pub fn get_parameters(&self) -> (f64, f64) {
        self.parameters
    }
}

struct Label {
    node: u32,
    parameters: (f64, f64),
    parent: Option<Rc<Label>>,
    used_edge: u32,
}

impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        // Compare all fields for true equality
        self.parameters == other.parameters
            && self.node == other.node
            && self.used_edge == other.used_edge
            && match (&self.parent, &other.parent) {
                (None, None) => true,
                (Some(p1), Some(p2)) => Rc::ptr_eq(p1, p2),
                _ => false,
            }
    }
}

impl Eq for Label {}

impl PartialOrd for Label {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Primary ordering by parameters (for lexicographic expansion)
        match self.parameters.partial_cmp(&other.parameters)? {
            Ordering::Equal => {
                // Break ties with node
                match self.node.cmp(&other.node) {
                    Ordering::Equal => {
                        // Break ties with used_edge
                        match self.used_edge.cmp(&other.used_edge) {
                            Ordering::Equal => {
                                // Break ties with parent (compare by pointer address)
                                match (&self.parent, &other.parent) {
                                    (None, None) => Some(Ordering::Equal),
                                    (None, Some(_)) => Some(Ordering::Less),
                                    (Some(_), None) => Some(Ordering::Greater),
                                    (Some(p1), Some(p2)) => {
                                        // Compare parent pointers by address for tie-breaking
                                        let addr1 = p1.as_ref() as *const Label as usize;
                                        let addr2 = p2.as_ref() as *const Label as usize;
                                        addr1.partial_cmp(&addr2)
                                    }
                                }
                            }
                            other => Some(other),
                        }
                    }
                    other => Some(other),
                }
            }
            other => Some(other),
        }
    }
}

impl Ord for Label {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap() // panics on NaN
    }
}

pub struct CompleteGraph<'a> {
    num_nodes: u32,
    num_edges: u32,
    num_parameters: u32,

    edges: Vec<CoverEdge>,
    outgoing_edges: HashMap<u32, Vec<u32>>,
    incoming_edges: HashMap<u32, Vec<u32>>,
    original_graph: &'a Graph,

    depot: u32,
    goals: BTreeSet<(u32, u32)>,
}

impl<'a> CompleteGraph<'a> {
    pub fn new(original_graph: &'a Graph, depot: u32, goals: &BTreeSet<(u32, u32)>) -> Self {
        // Use multicriteria planning between all goals and depot to construct the complete graph
        let num_nodes:u32 = (goals.len() as u32) + 1;

        let mut nodes: HashSet<u32> = HashSet::new();
        nodes.insert(depot);
        nodes.extend(goals.iter().map(|(node, _)| *node));
        let nodes = nodes;

        let mut num_edges = 0;
        let mut edges: Vec<CoverEdge> = Vec::new();
        let mut outgoing_edges: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut incoming_edges: HashMap<u32, Vec<u32>> = HashMap::new();

        for node in &nodes {
            let mut targets = nodes.clone();
            targets.remove(&node);
            let found_labels = mc_planning(original_graph, *node, &targets);

            for (target, labels) in found_labels {
                assert!(labels.len() > 0, "No labels found for node {} and target {}", node, target);
                for label in labels {
                    let edge_parameters = label.parameters;
                    let mut composite_nodes: Vec<u32> = Vec::new();
                    let mut composite_edges: Vec<u32> = Vec::new();
                    let mut current_label = Some(label);

                    while let Some(l) = current_label  {
                        composite_nodes.push(l.node);
                        composite_edges.push(l.used_edge);
                        current_label = l.parent.as_ref().map(Rc::clone);
                    }
                    composite_edges.pop();
                    composite_nodes.reverse();
                    composite_edges.reverse();
                    edges.push(CoverEdge {
                        id: num_edges,
                        composite_nodes,
                        composite_edges,
                        parameters: edge_parameters,
                    });
                    outgoing_edges.entry(*node).or_default().push(num_edges);
                    incoming_edges.entry(target).or_default().push(num_edges);
                    num_edges += 1;
                }
            }
        }

        Self{
            num_nodes,
            num_edges,
            num_parameters: 2,
            edges,
            outgoing_edges,
            incoming_edges,
            original_graph,
            depot,
            goals: goals.clone(),
        }
    }

    pub fn get_depot(&self) -> u32 {
        self.depot
    }

    pub fn get_goals(&self) -> BTreeSet<(u32, u32)> {
        self.goals.clone()
    }

    pub fn get_edges(&self) -> &Vec<CoverEdge> {
        &self.edges
    }

    pub fn get_num_edges(&self) -> u32 {
        self.num_edges
    }

    pub fn get_edge_parameters(&self, edge_id: u32) -> (f64, f64) {
        self.edges[edge_id as usize].get_parameters()
    }

    


    
}

type RcLabel = Rc<Label>;

fn mc_planning(original_graph: &Graph, source:u32, targets: &HashSet<u32>) -> HashMap<u32, BTreeSet<Rc<Label>>> {
        // Use standard MLS algorithm to find the shortest paths between the source and the rest of important nodes (depot, goals)

        let mut closed_labels: HashMap<u32, BTreeSet<RcLabel>> = HashMap::new();
        let mut open_labels: HashMap<u32, BTreeSet<RcLabel>> = HashMap::new();
        let mut to_expand: BTreeSet<RcLabel> = BTreeSet::new();

        let source_label: RcLabel = Rc::new(Label {node: source, parameters: (0.0, 0.0), parent: None, used_edge: 0});
        open_labels.insert(source, BTreeSet::from([Rc::clone(&source_label)]));
        to_expand.insert(Rc::clone(&source_label));

        let mut touched_nodes: HashSet<u32> = HashSet::new();

        while !to_expand.is_empty() {
            let current_label = to_expand.pop_first().unwrap();
            let current_node = current_label.node;

            open_labels.get_mut(&current_node).unwrap().remove(&current_label);
            closed_labels.entry(current_node).or_default().insert(Rc::clone(&current_label));

            'edge_loop: for edge in original_graph.get_outgoing_edges(current_node) {
                let next_label = extend_label(original_graph, &current_label, *edge);

                touched_nodes.insert(next_label.node);

                // Domination check against closed labels
                for l in closed_labels.get(&next_label.node).unwrap_or(&BTreeSet::new()) {
                    if dominates(l, &next_label) {
                        continue 'edge_loop;
                    }
                }

                // Domination check against open labels
                let mut to_remove: Vec<RcLabel> = Vec::new();
                for l in open_labels.get(&next_label.node).unwrap_or(&BTreeSet::new()) {
                    if dominates(l, &next_label) {
                        continue 'edge_loop;
                    }
                    else if dominates(&next_label, l) {
                        to_remove.push(Rc::clone(l));
                    }
                }
                for l in to_remove {
                    open_labels.get_mut(&next_label.node).unwrap().remove(&l);
                    to_expand.remove(&l);
                }

                open_labels.entry(next_label.node).or_default().insert(Rc::clone(&next_label));
                to_expand.insert(Rc::clone(&next_label));

            }
        }

        assert!(touched_nodes.contains(&427), "Touched nodes must contain 427");
        assert!(open_labels.contains_key(&427), "Open labels must contain 427");
        assert!(closed_labels.contains_key(&427), "Closed labels must contain 427");

        let mut result: HashMap<u32, BTreeSet<RcLabel>> = HashMap::new();
        for t in targets {
            result.insert(*t, closed_labels.get(&t).unwrap_or(&BTreeSet::new()).clone());
        }

        return result;

    }

    fn extend_label(original_graph: &Graph, label: &RcLabel, edge: u32) -> RcLabel {
        let next_node = original_graph.get_edge_points(edge).1;
        let next_edge_parameters = original_graph.get_edge_parameters(edge);
        let new_parameters = (label.parameters.0 + next_edge_parameters.0, label.parameters.1 + next_edge_parameters.1);
        Rc::new(Label {node: next_node, parameters: new_parameters, parent: Some(Rc::clone(label)), used_edge: edge})
    }

    fn dominates(label: &Rc<Label>, other: &Rc<Label>) -> bool {
        label.parameters.0 <= other.parameters.0 && label.parameters.1 <= other.parameters.1
    }