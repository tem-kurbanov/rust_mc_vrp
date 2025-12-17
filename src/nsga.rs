use std::collections::{HashMap, HashSet, BTreeSet, BTreeMap};
use rand::{Rng};
use rand::seq::SliceRandom;
use std::cmp::Ordering;

use crate::complete_graph::CompleteGraph;

struct Chromosome {
    order_genes: Vec<u32>,
    edge_genes: Vec<bool>,
    fitness_values: (f64, f64),

    crowding_distance: f64,
    rank: usize,
}

impl Clone for Chromosome {
    fn clone(&self) -> Self {
        Self { order_genes: self.order_genes.clone(), edge_genes: self.edge_genes.clone(), fitness_values: self.fitness_values.clone(), crowding_distance: self.crowding_distance, rank: self.rank }
    }
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

    pub fn solve_capacitated_vrp(&self) -> Vec<Chromosome> {
        let mut population = self.generate_initial_population();
        let mut child_population = self.produce_child_population(&population);
        population.extend(child_population);

        let num_iterations = 50;
        let mut current_iteration = 0;
        let mut converged = false;
        
        while current_iteration < num_iterations && !converged {
            let mut sorted_populations = self.sort_population(&mut population);
            self.calculate_crowding_distances(&mut sorted_populations);

            let mut new_population = self.select_population(&sorted_populations);

            let mut child_population = self.produce_child_population(&new_population);
            new_population.extend(child_population);

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

            population.push(Chromosome { order_genes, edge_genes, fitness_values, crowding_distance: 0.0, rank: 0 });
        }

        population
    }

    fn produce_child_population(&self, parents: &Vec<Chromosome>) -> Vec<Chromosome> {
        let n = self.population_size as usize;
        let mut children = Vec::with_capacity(n);
        let mut rng = rand::rng();

        while children.len() < n {
            let p1 = self.tournament_select(parents);
            let p2 = self.tournament_select(parents);

            let (mut c1, mut c2) = if rng.random_bool(self.p_crossover) {
                self.crossover(p1, p2)          // returns 2 children
            } else {
                (p1.clone(), p2.clone())        // no crossover
            };

            self.mutate(&mut c1);
            self.mutate(&mut c2);

            // If your encoding needs repair, do it here
            // self.repair(&mut c1);
            // self.repair(&mut c2);

            // Evaluate objectives here or later in batch (but must happen before next sorting)
            // self.evaluate(&mut c1);
            // self.evaluate(&mut c2);

            children.push(c1);
            if children.len() < n {
                children.push(c2);
            }
        }

        children
    }

    fn tournament_select(&self, population: &'a Vec<Chromosome>) -> &'a Chromosome {

        let mut rng = rand::rng();
        let n = population.len();

        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);

        let a = &population[i];
        let b = &population[j];

        if a.rank < b.rank {
            a
        } else if a.rank > b.rank {
            b
        } else {
            // same rank → use crowding distance
            let da = a.crowding_distance;
            let db = b.crowding_distance;

            if da > db {
                a
            } else if da < db {
                b
            } else {
                // complete tie → random
                if rng.random_bool(0.5) { a } else { b }
            }
        }
    }


    fn calculate_crowding_distances(&self, population: &mut Vec<Chromosome>) {
        // Reset
        for ch in population.iter_mut() {
            ch.crowding_distance = 0.0;
        }

        // Group indices by rank (front)
        let mut fronts: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (idx, ch) in population.iter().enumerate() {
            fronts.entry(ch.rank).or_default().push(idx);
        }

        // Compute crowding per front
        for (_rank, front) in fronts {
            let m = front.len();

            if m == 0 {
                continue;
            }
            if m == 1 {
                population[front[0]].crowding_distance = f64::INFINITY;
                continue;
            }
            if m == 2 {
                population[front[0]].crowding_distance = f64::INFINITY;
                population[front[1]].crowding_distance = f64::INFINITY;
                continue;
            }

            // ---- Objective 0 ----
            {
                let mut ord = front.clone();
                ord.sort_by(|&a, &b| {
                    population[a].fitness_values.0.total_cmp(&population[b].fitness_values.0)
                });

                let minv = population[ord[0]].fitness_values.0;
                let maxv = population[ord[m - 1]].fitness_values.0;
                let denom = maxv - minv;

                population[ord[0]].crowding_distance = f64::INFINITY;
                population[ord[m - 1]].crowding_distance = f64::INFINITY;

                if denom > 0.0 {
                    for i in 1..m - 1 {
                        let k = ord[i];
                        if population[k].crowding_distance.is_infinite() {
                            continue;
                        }
                        let next = population[ord[i + 1]].fitness_values.0;
                        let prev = population[ord[i - 1]].fitness_values.0;
                        population[k].crowding_distance += (next - prev) / denom;
                    }
                }
            }

            // ---- Objective 1 ----
            {
                let mut ord = front.clone();
                ord.sort_by(|&a, &b| {
                    population[a].fitness_values.1.total_cmp(&population[b].fitness_values.1)
                });

                let minv = population[ord[0]].fitness_values.1;
                let maxv = population[ord[m - 1]].fitness_values.1;
                let denom = maxv - minv;

                population[ord[0]].crowding_distance = f64::INFINITY;
                population[ord[m - 1]].crowding_distance = f64::INFINITY;

                if denom > 0.0 {
                    for i in 1..m - 1 {
                        let k = ord[i];
                        if population[k].crowding_distance.is_infinite() {
                            continue;
                        }
                        let next = population[ord[i + 1]].fitness_values.1;
                        let prev = population[ord[i - 1]].fitness_values.1;
                        population[k].crowding_distance += (next - prev) / denom;
                    }
                }
            }
        }
    }

    fn sort_population(&self, population: &mut Vec<Chromosome>) -> Vec<Chromosome> {
        for ch in population.iter_mut() {
            ch.rank = 0;
        }
        // Sort population into several sets based on dominance of fitness values
        let mut sorted_population = Vec::new();
        let mut current_rank = 1;
        while !population.is_empty() {
            // Collect indices of non-dominated chromosomes
            let mut pareto_indices: Vec<usize> = Vec::new();
            let pop = population.as_slice();
            for i in 0..pop.len() {
                // Check if the chromosome is dominated by any other chromosome
                let mut dominated = false;
                for j in 0..pop.len() {
                    if i != j {
                        if Self::dominates(&pop[j], &pop[i]) {
                            dominated = true;
                            break;
                        }
                    }
                }
                // If the chromosome is not dominated by any other chromosome, add it to the Pareto set
                if !dominated {
                    pareto_indices.push(i);
                }
            }
            // Add the non-dominated chromosomes to a separate set and remove them from the population
            
            for i in pareto_indices {
                let mut chromosome = population.remove(i);
                chromosome.rank = current_rank;
                sorted_population.push(chromosome);
            }
            current_rank += 1;
        }

        sorted_population
    }

    fn select_population(&self, population: &Vec<Chromosome>) -> Vec<Chromosome> {
        let target = self.population_size as usize;
        let mut new_population = Vec::with_capacity(target);

        let mut current_rank: usize = 1; // rank 0 = unassigned

        while new_population.len() < target {
            // Collect indices of chromosomes with this rank
            let mut rank_indices: Vec<usize> = population
                .iter()
                .enumerate()
                .filter_map(|(i, ch)| if ch.rank == current_rank { Some(i) } else { None })
                .collect();

            assert!(!rank_indices.is_empty());

            let remaining = target - new_population.len();

            // Whole rank fits
            if rank_indices.len() <= remaining {
                new_population.extend(rank_indices.into_iter().map(|i| population[i].clone()));
                current_rank += 1;
                continue;
            }

            // Partial rank: take highest crowding distance within this rank
            rank_indices.sort_by(|&a, &b| {
                population[b]
                    .crowding_distance
                    .total_cmp(&population[a].crowding_distance)
            });

            new_population.extend(
                rank_indices
                    .into_iter()
                    .take(remaining)
                    .map(|i| population[i].clone()),
            );

            break; // filled
        }

        new_population
    }


    fn check_convergence(&self, &population: &Vec<Chromosome>) -> bool {

    }

    fn crossover(&self, p1: &Chromosome, p2: &Chromosome) -> (Chromosome, Chromosome) {

    }

    fn mutate(&self, chromosome: &mut Chromosome) {

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

    fn dominates(chromosome1: &Chromosome, chromosome2: &Chromosome) -> bool {
        chromosome1.fitness_values.0 <= chromosome2.fitness_values.0 && chromosome1.fitness_values.1 <= chromosome2.fitness_values.1
    }
    
}
