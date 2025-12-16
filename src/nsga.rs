use std::collections::{HashMap, BTreeSet};
use rand::{Rng};
use rand::seq::SliceRandom;
use std::collections::BTreeMap;

use crate::complete_graph::CompleteGraph;

struct Chromosome {
    order_genes: Vec<u32>,
    edge_genes: Vec<bool>,
    fitness_values: (f64, f64),
}

pub struct NSGA<'a> {
    complete_graph: CompleteGraph<'a>,
    id_to_order_index: HashMap<u32, u32>,
    order_index_to_id: HashMap<u32, u32>,

    goals: BTreeMap<u32, u32>,
    adjacency_matrix: Vec<Vec<Vec<u32>>>,
    population_size: u32,

    max_capacity: u32,

}

impl<'a> NSGA<'a> {
    pub fn new(complete_graph: CompleteGraph<'a>, population_size: u32, max_capacity: u32) -> Self {
        let mut id_to_order_index = HashMap::new();
        let mut order_index_to_id = HashMap::new();
        let mut goals: BTreeMap<u32, u32> = BTreeMap::new();

        let depot: u32 = complete_graph.get_depot();
        id_to_order_index.insert(depot, 0);
        order_index_to_id.insert(0, depot);

        let init_goals: BTreeSet<(u32, u32)> = complete_graph.get_goals();
        for (index, (node, demand)) in init_goals.iter().enumerate() {
            let order_index = (index as u32) + 1;
            id_to_order_index.insert(*node, order_index);
            order_index_to_id.insert(order_index, *node);
            goals.insert(order_index, *demand);
        }

        let mut adjacency_matrix = vec![vec![Vec::new(); goals.len() + 1]; goals.len() + 1];
        for edge in complete_graph.get_edges() {
            let node1 = edge.get_composite_nodes()[0];
            let node2 = edge.get_composite_nodes().last().unwrap().clone();
            let edge_id = edge.get_id();
            adjacency_matrix[node1 as usize][node2 as usize].push(edge_id);
        }

        Self { complete_graph, id_to_order_index, order_index_to_id, goals, adjacency_matrix, population_size, max_capacity }
    }

    pub fn solve_capacitate_vrp(&self) -> Vec<Chromosome> {
        let mut population = self.generate_initial_population();

        let num_iterations = 50;
        let mut current_iteration = 0;
        let mut converged = false;
        
        while current_iteration < num_iterations && !converged {
            let sorted_populations = self.sort_population(&mut population);
            let crowded_distances = self.calculate_crowded_distances(&sorted_populations);
            let mut new_population = self.select_population(sorted_populations, crowded_distances);
            new_population = self.crossover(new_population);
            new_population = self.mutation(new_population);
            converged = self.check_convergence(&new_population);
            current_iteration += 1;
            population = new_population;
        }

        population
    }

    fn generate_initial_population(&self) -> Vec<Chromosome> {
        // Generate random initial population according to the population size
        let mut population = Vec::new();
        let mut rng = rand::rng();
        for _ in 0..self.population_size {
            // Order node is a random permutation of ids from 1 to goals.len() + 1
            let mut order_genes:Vec<u32> = (1..(self.goals.len() + 1) as u32).collect();
            // Shuffle the order genes
            order_genes.shuffle(&mut rng);

            let mut edge_genes = vec![false; self.complete_graph.get_num_edges() as usize];
            for i in 0.. self.goals.len() + 1 {
                for j in 0.. self.goals.len() + 1 {
                    if i != j {
                        let edges = &self.adjacency_matrix[i][j];
                        // Select a random edge from the edges
                        let random_edge = edges[rng.random_range(0..edges.len())];
                        edge_genes[random_edge as usize] = true;
                    }
                }
            }
            // Compute the fitness values
            let fitness_values = self.evaluate(&order_genes, &edge_genes);

            population.push(Chromosome { order_genes, edge_genes, fitness_values });
        }

        population
    }

    fn calculate_crowded_distances(&self, &sorted_populations: &Vec<Vec<Chromosome>>) -> Vec<Vec<f64>> {
        
    }

    fn sort_population(&self, population: & mut Vec<Chromosome>) -> Vec<Vec<Chromosome>> {

    }

    fn select_population(&self, sorted_populations: Vec<Vec<Chromosome>>, crowded_distances: Vec<Vec<f64>>) -> Vec<Chromosome> {
        
    }

    fn check_convergence(&self, &population: &Vec<Chromosome>) -> bool {

    }

    fn crossover(&self, population: Vec<Chromosome>) -> Vec<Chromosome> {

    }

    fn mutation(&self, population: Vec<Chromosome>) -> Vec<Chromosome> {

    }

    fn evaluate(&self, order_genes: &Vec<u32>, edge_genes: &Vec<bool>) -> (f64, f64) {
        // Evaluate the fitness values
        let mut current_capacity: u32 = 0;
        let mut route_start_index: u32 = 0;
        let mut route_end_index: u32 = 0;

        let mut total_parameters = (0.0, 0.0);

        while route_start_index < order_genes.len() as u32 {
            while current_capacity <= self.max_capacity as u32 {
                let current_node = order_genes[route_end_index as usize];
                let node_demand = self.goals.get(&current_node).unwrap();
                
                current_capacity += node_demand;
                route_end_index += 1;
            }
            // Route starts at route_start_index and ends at route_end_index
            // Slice the involved nodes into a separate vector
            let slice = &order_genes[route_start_index as usize .. route_end_index as usize];
            let mut current_route = slice.to_vec();
            current_route.insert(0, 0);
            current_route.push(0);

            let mut current_parameters = (0.0, 0.0);
            for i in 0..current_route.len() - 1 {
                let current_node = current_route[i];
                let next_node = current_route[i + 1];
                let mut edge_id = 0;
                for edge in &self.adjacency_matrix[current_node as usize][next_node as usize] {
                    if edge_genes[*edge as usize] {
                        edge_id = *edge;
                        break;
                    }
                }
                let edge_parameters = self.complete_graph.get_edge_parameters(edge_id);
                current_parameters.0 += edge_parameters.0;
                current_parameters.1 += edge_parameters.1;
            }

            total_parameters.0 += current_parameters.0;
            total_parameters.1 += current_parameters.1;

            route_start_index = route_end_index;

        }

        return total_parameters;

    }
    
}
