use std::collections::{HashMap, HashSet, BTreeSet};
use std::rc::{Rc, Weak};
use std::cmp::Ordering;

use crate::graph::Graph;


struct CoverEdge {
    id: u32,
    composite_nodes: Vec<u32>,
    composite_edges: Vec<u32>,
    parameters: (i32, i32),
}

struct Label {
    node: u32,
    parameters: (i32, i32),
    parent: Option<Rc<Label>>,
    used_edge: u32,
}

impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        self.parameters == other.parameters
    }
}

impl Eq for Label {}

impl PartialOrd for Label {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))  // delegates to Ord::cmp
    }
}

impl Ord for Label {
    fn cmp(&self, other: &Self) -> Ordering {
        self.parameters.cmp(&other.parameters)
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
    goals: HashSet<u32>,
}

impl<'a> CompleteGraph<'a> {
    pub fn new(original_graph: &'a Graph, depot: u32, goals: &HashSet<u32>) -> Self {
        // Use multicriteria planning between all goals and depot to construct the complete graph
        let num_nodes:u32 = (goals.len() as u32) + 1;

        let mut nodes: HashSet<u32> = HashSet::new();
        nodes.insert(depot);
        nodes.extend(goals.clone());
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

    

    

    


    
}

type RcLabel = Rc<Label>;

fn mc_planning(original_graph: &Graph, source:u32, targets: &HashSet<u32>) -> HashMap<u32, BTreeSet<Rc<Label>>> {
        // Use standard MLS algorithm to find the shortest paths between the source and the rest of important nodes (depot, goals)

        let mut closed_labels: HashMap<u32, BTreeSet<RcLabel>> = HashMap::new();
        let mut open_labels: HashMap<u32, BTreeSet<RcLabel>> = HashMap::new();
        let mut to_expand: BTreeSet<RcLabel> = BTreeSet::new();

        let source_label: RcLabel = Rc::new(Label {node: source, parameters: (0, 0), parent: None, used_edge: 0});
        open_labels.insert(source, BTreeSet::from([Rc::clone(&source_label)]));
        to_expand.insert(Rc::clone(&source_label));

        while !to_expand.is_empty() {
            let current_label = to_expand.pop_first().unwrap();
            let current_node = current_label.node;

            open_labels.get_mut(&current_node).unwrap().remove(&current_label);
            closed_labels.entry(current_node).or_default().insert(Rc::clone(&current_label));

            'edge_loop: for edge in original_graph.get_outgoing_edges(current_node) {
                let next_label = extend_label(original_graph, &current_label, *edge);


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
                    else if dominates(l, &next_label) {
                        to_remove.push(Rc::clone(l));
                    }
                }
                for l in to_remove {
                    open_labels.get_mut(&next_label.node).unwrap().remove(&l);
                }

                open_labels.entry(next_label.node).or_default().insert(Rc::clone(&next_label));
                to_expand.insert(Rc::clone(&next_label));

            }
        }

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