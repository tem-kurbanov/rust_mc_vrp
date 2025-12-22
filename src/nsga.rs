use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::process::exit;

use crate::complete_graph::CompleteGraph;

#[derive(Debug)]
pub struct Chromosome {
    order_genes: Vec<u32>,
    edge_genes: Vec<bool>,
    fitness_values: (f64, f64),

    crowding_distance: f64,
    rank: usize,
}

impl Clone for Chromosome {
    fn clone(&self) -> Self {
        Self {
            order_genes: self.order_genes.clone(),
            edge_genes: self.edge_genes.clone(),
            fitness_values: self.fitness_values.clone(),
            crowding_distance: self.crowding_distance,
            rank: self.rank,
        }
    }
}

impl Chromosome {
    pub fn get_order_genes(&self) -> &Vec<u32> {
        &self.order_genes
    }

    pub fn get_fitness_values(&self) -> (f64, f64) {
        self.fitness_values
    }
}

/// 2D hypervolume for **minimization** with a fixed reference point `r`
/// (r must be worse than all points you want to count: r0 >= f0 and r1 >= f1).
///
/// Expects `points` to be *any* set (may include dominated points); it will
/// compute the HV of the **nondominated envelope** in 2D.
pub fn hypervolume_2d_min(points: &[(f64, f64)], r: (f64, f64)) -> f64 {
    let (r0, r1) = r;

    // Filter invalid points and those outside the reference box (they contribute 0 or break math).
    let mut pts: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .filter(|(x, y)| *x <= r0 && *y <= r1)
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    // Sort by x ascending, then y ascending (stable tie-break doesn't matter much here).
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    // Sweep to keep only the decreasing "best-so-far" y envelope.
    // For minimization, as x increases, we want y to strictly decrease to be nondominated.
    let mut hv = 0.0;
    let mut prev_y = r1;

    for (x, y) in pts {
        if y >= prev_y {
            // dominated (or equal) in y by an earlier point with <= x
            continue;
        }
        // Rectangle from x..r0 and y..prev_y
        hv += (r0 - x) * (prev_y - y);
        prev_y = y;
        if prev_y <= f64::NEG_INFINITY {
            break;
        }
    }

    hv
}

/// Tracks hypervolume and reports convergence when it stalls for `window` gens.
/// Criterion: `hv_now - min(hv_history) < eps` over the last `window` values.
pub struct HvStall {
    reference: (f64, f64),
    window: usize,
    eps: f64,
    history: VecDeque<f64>,
}

impl HvStall {
    pub fn new(reference: (f64, f64), window: usize, eps: f64) -> Self {
        assert!(window >= 2);
        assert!(eps >= 0.0);
        Self {
            reference,
            window,
            eps,
            history: VecDeque::with_capacity(window),
        }
    }

    /// Push current nondominated set (or archive) and return (hv, converged?).
    pub fn update(&mut self, points: &[(f64, f64)]) -> (f64, bool) {
        let hv = hypervolume_2d_min(points, self.reference);

        if self.history.len() == self.window {
            self.history.pop_front();
        }
        self.history.push_back(hv);

        if self.history.len() < self.window {
            return (hv, false);
        }

        let min_hv = self.history.iter().copied().fold(f64::INFINITY, f64::min);

        let stalled = (hv - min_hv) < self.eps;
        (hv, stalled)
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

    p_crossover: f64,
    p_mutation: f64,
}

impl<'a> NSGA<'a> {
    pub fn new_parallel(
        complete_graph: CompleteGraph<'a>,
        population_size: u32,
        max_capacity: u32,
        p_crossover: f64,
        p_mutation: f64,
    ) -> Self {
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
            let node1 = id_to_order_index[&edge.get_composite_nodes()[0]];
            let node2 = id_to_order_index[&edge.get_composite_nodes().last().unwrap().clone()];
            let edge_id = edge.get_id();
            adjacency_matrix[node1 as usize][node2 as usize].push(edge_id);
        }

        Self {
            complete_graph,
            id_to_order_index,
            order_index_to_id,
            goals,
            adjacency_matrix,
            population_size,
            max_capacity,
            p_crossover,
            p_mutation,
        }
    }

    pub fn solve_capacitated_vrp(&self) -> Vec<Chromosome> {
        let mut population = self.generate_initial_population();
        let child_population = self.produce_child_population(&population);
        population.extend(child_population);

        let num_iterations = 500;

        // External archive of nondominated objective pairs (minimization)
        let mut archive_front: Vec<(f64, f64)> = Vec::new();

        let mut hv_stall = HvStall::new(
            /*reference*/ (1.0e9, 1.0e9),
            /*window*/ 30,
            /*eps*/ 1e-6,
        );

        let mut initial_hv: f64 = 0.0;

        for generation in 0..num_iterations {
            let fronts = self.sort_population(&mut population);
            self.calculate_crowding_distances(&mut population, &fronts);

            let mut new_population = self.select_population(&population, &fronts);

            let child_population = self.produce_child_population(&new_population);
            new_population.extend(child_population);

            // --- update archive from the NEW population (or from first front only) ---
            Self::update_archive_2d_min(
                &mut archive_front,
                new_population.iter().map(|c| c.fitness_values),
            );

            // --- convergence check on archive ---
            let (_hv, converged) = hv_stall.update(&archive_front);
            if initial_hv == 0.0 {
                initial_hv = _hv;
            }
            println!(
                "Generation {}: Archive size {}, HV: {:.6}",
                generation + 1,
                archive_front.len(),
                _hv - initial_hv
            );
            if converged {
                println!("Converged at generation {}", generation + 1);
                population = new_population;
                break;
            }

            population = new_population;
        }

        println!("Final archive ({} solutions):", archive_front.len());
        for (i, (obj1, obj2)) in archive_front.iter().enumerate() {
            println!("  Solution {}: ({:.3}, {:.3})", i + 1, obj1, obj2);
        }

        population
    }

    fn points_from_population(pop: &[Chromosome]) -> Vec<(f64, f64)> {
        pop.iter().map(|c| c.fitness_values).collect()
    }

    fn dominates(a: (f64, f64), b: (f64, f64)) -> bool {
        // minimization: a dominates b if it's <= in both and < in at least one
        (a.0 <= b.0 && a.1 <= b.1) && (a.0 < b.0 || a.1 < b.1)
    }

    fn nondominated_2d(points: &mut Vec<(f64, f64)>) {
        // O(n^2) but fine for typical archive sizes; can optimize later.
        let mut keep = Vec::with_capacity(points.len());

        'outer: for &p in points.iter() {
            for &q in points.iter() {
                if q != p && Self::dominates(q, p) {
                    continue 'outer; // p is dominated
                }
            }
            keep.push(p);
        }

        // Optional: remove duplicates
        keep.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        keep.dedup();

        *points = keep;
    }

    fn update_archive_2d_min(
        archive: &mut Vec<(f64, f64)>,
        current: impl IntoIterator<Item = (f64, f64)>,
    ) {
        archive.extend(current);
        // drop NaNs etc. early (HV code also filters, but do it once here)
        archive.retain(|(x, y)| x.is_finite() && y.is_finite());
        Self::nondominated_2d(archive);
    }

    fn generate_initial_population(&self) -> Vec<Chromosome> {
        // Generate random initial population according to the population size
        let mut population = Vec::new();
        let mut rng = rand::rng();
        for _ in 0..self.population_size {
            // Order node is a random permutation of ids from 1 to goals.len() + 1
            let mut order_genes: Vec<u32> = (1..(self.goals.len() + 1) as u32).collect();
            // Shuffle the order genes
            order_genes.shuffle(&mut rng);

            let mut edge_genes = vec![false; self.complete_graph.get_num_edges() as usize];
            for i in 0..=self.goals.len() {
                for j in 0..=self.goals.len() {
                    if i != j {
                        let edges = &self.adjacency_matrix[i][j];
                        if edges.is_empty() {
                            continue;
                        }
                        // Select a random edge from the edges
                        let random_edge = edges[0];
                        edge_genes[random_edge as usize] = true;
                    }
                }
            }
            // Compute the fitness values
            let fitness_values = self.evaluate(&order_genes, &edge_genes);

            population.push(Chromosome {
                order_genes,
                edge_genes,
                fitness_values,
                crowding_distance: 0.0,
                rank: 0,
            });
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
                self.crossover(p1, p2) // returns 2 children
            } else {
                (p1.clone(), p2.clone()) // no crossover
            };

            // Mutate each child independently with probability p_mutation
            if rng.random_bool(self.p_mutation) {
                self.mutate(&mut c1);
            }
            if rng.random_bool(self.p_mutation) {
                self.mutate(&mut c2);
            }

            c1.fitness_values = self.evaluate(&c1.order_genes, &c1.edge_genes);
            children.push(c1);
            if children.len() < n {
                c2.fitness_values = self.evaluate(&c2.order_genes, &c2.edge_genes);
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

    fn calculate_crowding_distances(&self, population: &mut [Chromosome], fronts: &[Vec<usize>]) {
        // Reset
        for ch in population.iter_mut() {
            ch.crowding_distance = 0.0;
        }

        // Compute crowding per front
        for front in fronts {
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
                    population[a]
                        .fitness_values
                        .0
                        .total_cmp(&population[b].fitness_values.0)
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
                    population[a]
                        .fitness_values
                        .1
                        .total_cmp(&population[b].fitness_values.1)
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

    /// Standard NSGA-II fast non-dominated sort (O(MN^2) with M=2 here).
    /// Assigns `rank` for each chromosome and returns fronts as indices into `population`.
    fn sort_population(&self, population: &mut [Chromosome]) -> Vec<Vec<usize>> {
        let n = population.len();
        for ch in population.iter_mut() {
            ch.rank = 0;
        }

        let mut domination_count: Vec<usize> = vec![0; n];
        let mut dominates: Vec<Vec<usize>> = vec![Vec::new(); n];

        // Pairwise dominance checks (symmetric loop cuts comparisons roughly in half)
        for p in 0..n {
            for q in (p + 1)..n {
                let fp = population[p].fitness_values;
                let fq = population[q].fitness_values;

                if Self::dominates(fp, fq) {
                    dominates[p].push(q);
                    domination_count[q] += 1;
                } else if Self::dominates(fq, fp) {
                    dominates[q].push(p);
                    domination_count[p] += 1;
                }
            }
        }

        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut first: Vec<usize> = Vec::new();
        for i in 0..n {
            if domination_count[i] == 0 {
                population[i].rank = 1;
                first.push(i);
            }
        }
        if first.is_empty() {
            // Shouldn't happen unless all fitnesses are NaN/inf in a weird way.
            // Fall back to single front.
            for i in 0..n {
                population[i].rank = 1;
                first.push(i);
            }
        }
        fronts.push(first);

        let mut front_idx = 0usize;
        while front_idx < fronts.len() {
            let mut next: Vec<usize> = Vec::new();
            for &p in &fronts[front_idx] {
                for &q in &dominates[p] {
                    domination_count[q] -= 1;
                    if domination_count[q] == 0 {
                        population[q].rank = front_idx + 2;
                        next.push(q);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            fronts.push(next);
            front_idx += 1;
        }

        fronts
    }

    fn select_population(&self, population: &[Chromosome], fronts: &[Vec<usize>]) -> Vec<Chromosome> {
        let target = self.population_size as usize;
        let mut new_population = Vec::with_capacity(target);

        for front in fronts {
            if new_population.len() >= target {
                break;
            }

            let remaining = target - new_population.len();
            if front.len() <= remaining {
                new_population.extend(front.iter().map(|&i| population[i].clone()));
                continue;
            }

            // Partial front: take highest crowding distance
            let mut ord = front.clone();
            ord.sort_by(|&a, &b| {
                population[b]
                    .crowding_distance
                    .total_cmp(&population[a].crowding_distance)
            });
            new_population.extend(ord.into_iter().take(remaining).map(|i| population[i].clone()));
            break;
        }

        new_population
    }

    fn crossover(&self, p1: &Chromosome, p2: &Chromosome) -> (Chromosome, Chromosome) {
        let c1 = self.ox1(&p1.order_genes, &p2.order_genes);
        let c2 = self.ox1(&p2.order_genes, &p1.order_genes);

        let mut child1 = p1.clone();
        child1.order_genes = c1;
        child1.fitness_values = (0.0, 0.0);
        child1.rank = 0;
        child1.crowding_distance = 0.0;

        let mut child2 = p2.clone();
        child2.order_genes = c2;
        child2.fitness_values = (0.0, 0.0);
        child2.rank = 0;
        child2.crowding_distance = 0.0;

        (child1, child2)
    }

    fn ox1<T: Copy + Eq + Hash>(&self, p1: &[T], p2: &[T]) -> Vec<T> {
        // Select crossover points
        let n = p1.len();
        let mut rng = rand::rng();

        // two cut points [a, b)
        let mut a = rng.random_range(0..n);
        let mut b = rng.random_range(0..n);
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        if a == b {
            b = (a + 1) % n;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
        }

        let mut child: Vec<Option<T>> = vec![None; n];

        // copy the slice from p1
        let mut used = HashSet::with_capacity(b - a);
        for i in a..b {
            let g = p1[i];
            child[i] = Some(g);
            used.insert(g);
        }

        // fill remaining slots from p2, starting at b, circularly
        let mut write = b % n;
        for k in 0..n {
            let g = p2[(b + k) % n];
            if used.contains(&g) {
                continue;
            }

            while child[write].is_some() {
                write = (write + 1) % n;
            }
            child[write] = Some(g);
            used.insert(g);
        }

        child.into_iter().map(|x| x.unwrap()).collect()
    }

    fn mutate(&self, chromosome: &mut Chromosome) {
        // inversion mutation
        let v = &mut chromosome.order_genes;
        let n = v.len();

        if n < 2 {
            return;
        }

        let mut rng = rand::rng();
        let mut i = rng.random_range(0..n);
        let mut j = rng.random_range(0..n);

        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        if i == j {
            return;
        }
        v[i..=j].reverse();
    }

    fn evaluate(&self, order_genes: &Vec<u32>, edge_genes: &Vec<bool>) -> (f64, f64) {
        // Evaluate the fitness values
        let mut route_start_index: u32 = 0;
        let n: u32 = order_genes.len() as u32;

        let mut total_parameters = (0.0, 0.0);

        let add_leg = |from: u32, to: u32, total: &mut (f64, f64)| {
            let mut edge_id = 0;
            let mut edge_found = false;
            for edge in &self.adjacency_matrix[from as usize][to as usize] {
                if edge_genes[*edge as usize] {
                    edge_id = *edge;
                    edge_found = true;
                    break;
                }
            }

            if !edge_found {
                println!(
                    "No edge from node {} to node {}",
                    self.order_index_to_id[&from],
                    self.order_index_to_id[&to]
                );
                exit(1);
            }

            let edge_parameters = self.complete_graph.get_edge_parameters(edge_id);
            total.0 += edge_parameters.0;
            total.1 += edge_parameters.1;
        };

        while route_start_index < n {
            let mut current_capacity: u32 = 0;
            let mut route_end_index: u32 = route_start_index;

            // Build a single vehicle route as a maximal prefix that still fits capacity.
            while route_end_index < n {
                let current_node = order_genes[route_end_index as usize];
                let node_demand = *self.goals.get(&current_node).unwrap();

                // Infeasible instance / chromosome for CVRP: a single customer exceeds capacity.
                // Return a dominated fitness so it won't survive selection.
                if node_demand > self.max_capacity {
                    return (f64::INFINITY, f64::INFINITY);
                }

                // Stop BEFORE exceeding capacity.
                if current_capacity + node_demand > self.max_capacity {
                    break;
                }

                current_capacity += node_demand;
                route_end_index += 1;
            }
            // Route is [route_start_index, route_end_index). Sum legs directly:
            // depot -> first -> ... -> last -> depot.
            let mut prev_node: u32 = 0;
            for idx in route_start_index..route_end_index {
                let node = order_genes[idx as usize];
                add_leg(prev_node, node, &mut total_parameters);
                prev_node = node;
            }
            add_leg(prev_node, 0, &mut total_parameters);

            route_start_index = route_end_index;
        }

        return total_parameters;
    }
}
