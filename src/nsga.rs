use std::collections::{HashMap, BTreeSet};


use crate::complete_graph::CompleteGraph;

struct Chromosome {
    order_genes: Vec<u32>,
    edge_genes: Vec<u32>,
}

pub struct NSGA<'a> {
    complete_graph: CompleteGraph<'a>,
    id_to_order_index: HashMap<u32, u32>,
    order_index_to_id: HashMap<u32, u32>,

    goals: BTreeSet<(u32, u32)>,

}

impl<'a> NSGA<'a> {
    pub fn new(complete_graph: CompleteGraph<'a>) -> Self {
        let mut id_to_order_index = HashMap::new();
        let mut order_index_to_id = HashMap::new();
        let mut goals: BTreeSet<(u32, u32)> = BTreeSet::new();

        let depot: u32 = complete_graph.get_depot();
        id_to_order_index.insert(depot, 0);
        order_index_to_id.insert(0, depot);

        let init_goals: BTreeSet<(u32, u32)> = complete_graph.get_goals();
        for (index, (node, demand)) in init_goals.iter().enumerate() {
            let order_index = (index as u32) + 1;
            id_to_order_index.insert(*node, order_index);
            order_index_to_id.insert(order_index, *node);
            goals.insert((order_index, *demand));
        }

        Self { complete_graph, id_to_order_index, order_index_to_id, goals }
    }

    pub fn solve_capacitate_vrp(&self) -> Vec<Chromosome> {
        let mut population = self.generate_initial_population();

        let num_iterations = 50;
        let mut current_iteration = 0;
        let mut converged = false;
        
        while current_iteration < num_iterations && !converged {
            let sorted_populations = self.sort_population(&population);
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

    }

    fn calculate_crowded_distances(&self, &sorted_populations: &Vec<Vec<Chromosome>>) -> Vec<Vec<f64>> {
        
    }

    fn sort_population(&self, &population: &Vec<Chromosome>) -> Vec<Vec<Chromosome>> {

    }

    fn select_population(&self, sorted_populations: Vec<Vec<Chromosome>>, crowded_distances: Vec<Vec<f64>>) -> Vec<Chromosome> {
        
    }

    fn check_convergence(&self, &population: &Vec<Chromosome>) -> bool {

    }

    fn crossover(&self, population: Vec<Chromosome>) -> Vec<Chromosome> {

    }

    fn mutation(&self, population: Vec<Chromosome>) -> Vec<Chromosome> {

    }

    fn evaluate(&self, &chromosome: Chromosome) -> (f64, f64) {

    }
    
}
