//! NSGA-II style multiobjective evolutionary solver for CVRP.
//!
//! ## Encoding
//! A solution (`Chromosome`) stores a permutation of customers in `order_genes`.
//! - Node id `0` is treated as the depot.
//! - Customers are `1..num_nodes-1`.
//! - Routes are implicit: the permutation is split into consecutive segments using the
//!   capacity constraint (`route_splits`).
//!
//! ## Objectives
//! The underlying instance graph provides 2 edge parameters:
//! - objective 0: Euclidean distance
//! - objective 1: seeded random secondary cost (see `graph.rs`)
//!
//! `evaluate()` computes the sum of both objectives across all routes (including depot legs).
//!
//! ## Variation operators
//! - Crossover: HGreX (edge-based greedy crossover), capacity-aware.
//! - Mutation: route-aware mutation (intra-route 2-opt, relocate, swap).

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::{HashSet, VecDeque};
use std::hash::Hash;
use std::io::Write;
use std::time::Instant;

use crate::graph::Graph;

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum InsertionStrategy {
    CheapRandom,
    BestDistance,
    BestSecondary,
    BestNormalized,
    BestWeighted,
}

impl InsertionStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CheapRandom => "cheap-random",
            Self::BestDistance => "best-distance",
            Self::BestSecondary => "best-secondary",
            Self::BestNormalized => "best-normalized",
            Self::BestWeighted => "best-weighted",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RunConfig {
    pub max_generations: usize,
    pub hv_window: usize,
    pub hv_eps: f64,
    pub log_every: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParetoSolution {
    pub fitness_values: (f64, f64),
    pub routes: Vec<Vec<u32>>,
}

#[derive(Clone, Debug)]
pub struct NsgaRunResult {
    pub pareto_front: Vec<ParetoSolution>,
    pub generations: usize,
    pub archive_size: usize,
    pub hv_final: f64,
    pub elapsed_ms: u128,
    pub converged: bool,
}

#[derive(Clone, Debug)]
struct ParentInfo {
    order: Vec<u32>,
    pos: Vec<Option<usize>>,
    route_end: Vec<usize>,
    route_starts: Vec<u32>,
}

#[derive(Clone, Debug)]
struct Chromosome {
    order_genes: Vec<u32>,
    fitness_values: (f64, f64),
    crowding_distance: f64,
    rank: usize,
}

#[derive(Clone, Copy, Debug)]
struct InsertionCandidate {
    route_index: Option<usize>,
    insert_index: usize,
    delta: (f64, f64),
}

/// 2D hypervolume for minimization with a fixed reference point `r`.
pub fn hypervolume_2d_min(points: &[(f64, f64)], r: (f64, f64)) -> f64 {
    let (r0, r1) = r;

    let mut pts: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .filter(|(x, y)| *x <= r0 && *y <= r1)
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    let mut hv = 0.0;
    let mut prev_y = r1;

    for (x, y) in pts {
        if y >= prev_y {
            continue;
        }
        hv += (r0 - x) * (prev_y - y);
        prev_y = y;
    }

    hv
}

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

pub struct NSGA {
    graph: Graph,
    adjacency_matrix: Vec<Vec<Vec<u32>>>,
    num_nodes: usize,
    population_size: u32,
    max_capacity: u32,
    p_crossover: f64,
    p_mutation: f64,
    mutation_inversion_pct: u32,
    mutation_relocate_pct: u32,
    mutation_swap_pct: u32,
    mutation_or_opt_pct: u32,
    mutation_destroy_repair_pct: u32,
    mutation_refine_sample_size: usize,
    init_random_pct: u32,
    init_best_insertion_pct: u32,
    init_insertion_strategy: InsertionStrategy,
    relocate_insertion_strategy: InsertionStrategy,
    init_weighted_weights: (f64, f64),
    relocate_weighted_weights: (f64, f64),
    nsga_seed: u64,
    run_config: RunConfig,
}

impl NSGA {
    pub fn new(
        graph: Graph,
        population_size: u32,
        p_crossover: f64,
        p_mutation: f64,
        init_random_pct: u32,
        init_best_insertion_pct: u32,
        init_insertion_strategy: InsertionStrategy,
        relocate_insertion_strategy: InsertionStrategy,
        mutation_inversion_pct: u32,
        mutation_relocate_pct: u32,
        mutation_swap_pct: u32,
        mutation_or_opt_pct: u32,
        mutation_destroy_repair_pct: u32,
        mutation_refine_sample_size: usize,
        nsga_seed: u64,
        run_config: RunConfig,
    ) -> Self {
        let num_nodes = graph.get_num_nodes();
        let max_capacity = graph.get_capacity();
        let mut adjacency_matrix = vec![vec![Vec::new(); num_nodes as usize]; num_nodes as usize];
        for edge in graph.get_edges() {
            adjacency_matrix[edge.get_source() as usize][edge.get_target() as usize]
                .push(edge.get_id());
        }

        Self {
            graph,
            adjacency_matrix,
            num_nodes: num_nodes as usize,
            population_size,
            max_capacity,
            p_crossover,
            p_mutation,
            mutation_inversion_pct,
            mutation_relocate_pct,
            mutation_swap_pct,
            mutation_or_opt_pct,
            mutation_destroy_repair_pct,
            mutation_refine_sample_size,
            init_random_pct,
            init_best_insertion_pct,
            init_insertion_strategy,
            relocate_insertion_strategy,
            init_weighted_weights: (0.8, 0.2),
            relocate_weighted_weights: (0.7, 0.3),
            nsga_seed,
            run_config,
        }
    }

    pub fn solve_with_writer<W: Write>(&self, w: &mut W) -> NsgaRunResult {
        let started_at = Instant::now();
        let mut rng = StdRng::seed_from_u64(self.nsga_seed);

        let mut population = self.generate_initial_population(&mut rng);
        let child_population = self.produce_child_population(&population, &mut rng);
        population.extend(child_population);

        let mut archive: Vec<Chromosome> = Vec::new();
        let mut hv_stall = HvStall::new((1.0e9, 1.0e9), self.run_config.hv_window, self.run_config.hv_eps);
        let mut completed_generations = 0usize;
        let mut final_hv = 0.0;
        let mut converged = false;

        for generation in 0..self.run_config.max_generations {
            let fronts = self.sort_population(&mut population);
            self.calculate_crowding_distances(&mut population, &fronts);

            let mut new_population = self.select_population(&population, &fronts);
            let child_population = self.produce_child_population(&new_population, &mut rng);
            new_population.extend(child_population);

            Self::update_archive(&mut archive, new_population.iter().cloned());

            let archive_points: Vec<(f64, f64)> = archive.iter().map(|c| c.fitness_values).collect();
            let (hv, stalled) = hv_stall.update(&archive_points);
            final_hv = hv;
            completed_generations = generation + 1;

            if self.run_config.log_every > 0
                && (completed_generations == 1
                    || completed_generations % self.run_config.log_every == 0
                    || stalled)
            {
                let _ = writeln!(
                    w,
                    "Generation {}: archive_size={}, hv={:.6}",
                    completed_generations,
                    archive.len(),
                    hv
                );
            }

            population = new_population;
            if stalled {
                converged = true;
                break;
            }
        }

        let pareto_front = self.archive_to_pareto_solutions(&archive);
        NsgaRunResult {
            archive_size: pareto_front.len(),
            pareto_front,
            generations: completed_generations,
            hv_final: final_hv,
            elapsed_ms: started_at.elapsed().as_millis(),
            converged,
        }
    }

    fn dominates(a: (f64, f64), b: (f64, f64)) -> bool {
        (a.0 <= b.0 && a.1 <= b.1) && (a.0 < b.0 || a.1 < b.1)
    }

    fn update_archive(archive: &mut Vec<Chromosome>, current: impl IntoIterator<Item = Chromosome>) {
        archive.extend(current);
        archive.retain(|ch| ch.fitness_values.0.is_finite() && ch.fitness_values.1.is_finite());
        archive.sort_by(|a, b| {
            a.fitness_values
                .0
                .total_cmp(&b.fitness_values.0)
                .then_with(|| a.fitness_values.1.total_cmp(&b.fitness_values.1))
        });
        archive.dedup_by(|a, b| {
            a.fitness_values.0.to_bits() == b.fitness_values.0.to_bits()
                && a.fitness_values.1.to_bits() == b.fitness_values.1.to_bits()
        });

        let snapshot = archive.clone();
        let mut keep = Vec::with_capacity(snapshot.len());
        'outer: for (idx, candidate) in snapshot.iter().enumerate() {
            for (other_idx, other) in snapshot.iter().enumerate() {
                if idx != other_idx && Self::dominates(other.fitness_values, candidate.fitness_values) {
                    continue 'outer;
                }
            }
            keep.push(candidate.clone());
        }
        *archive = keep;
    }

    fn archive_to_pareto_solutions(&self, archive: &[Chromosome]) -> Vec<ParetoSolution> {
        let mut out: Vec<ParetoSolution> = archive
            .iter()
            .map(|ch| ParetoSolution {
                fitness_values: ch.fitness_values,
                routes: self.decode_routes(&ch.order_genes),
            })
            .collect();
        out.sort_by(|a, b| {
            a.fitness_values
                .0
                .total_cmp(&b.fitness_values.0)
                .then_with(|| a.fitness_values.1.total_cmp(&b.fitness_values.1))
        });
        out
    }

    fn generate_initial_population<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<Chromosome> {
        let mut population = Vec::with_capacity(self.population_size as usize);
        let counts = self.allocate_initialization_counts();

        for _ in 0..counts.0 {
            let order_genes = self.generate_random_order(rng);
            let fitness_values = self.evaluate(&order_genes);
            population.push(Chromosome {
                order_genes,
                fitness_values,
                crowding_distance: 0.0,
                rank: 0,
            });
        }

        for _ in 0..counts.1 {
            let order_genes = self.generate_best_insertion_order(rng);
            let fitness_values = self.evaluate(&order_genes);
            population.push(Chromosome {
                order_genes,
                fitness_values,
                crowding_distance: 0.0,
                rank: 0,
            });
        }

        population.shuffle(rng);
        population
    }

    fn allocate_initialization_counts(&self) -> (usize, usize) {
        let total = self.population_size as usize;
        let exact_random = (self.init_random_pct as f64 * total as f64) / 100.0;
        let exact_best = (self.init_best_insertion_pct as f64 * total as f64) / 100.0;

        let mut random_count = exact_random.floor() as usize;
        let mut best_count = exact_best.floor() as usize;
        let assigned = random_count + best_count;

        if assigned < total {
            let random_frac = exact_random - random_count as f64;
            let best_frac = exact_best - best_count as f64;
            if best_frac > random_frac {
                best_count += total - assigned;
            } else {
                random_count += total - assigned;
            }
        }

        (random_count, best_count)
    }

    fn generate_random_order<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<u32> {
        let mut order_genes: Vec<u32> = (1..self.num_nodes as u32).collect();
        order_genes.shuffle(rng);
        order_genes
    }

    fn generate_best_insertion_order<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<u32> {
        let mut unvisited: Vec<u32> = (1..self.num_nodes as u32).collect();
        let mut routes: Vec<Vec<u32>> = Vec::new();
        let mut route_loads: Vec<u32> = Vec::new();

        while !unvisited.is_empty() {
            let (selected_node, selected_idx, selected) = if routes.is_empty() {
                let selected_idx = rng.random_range(0..unvisited.len());
                let selected_node = unvisited[selected_idx];
                (
                    selected_node,
                    selected_idx,
                    InsertionCandidate {
                        route_index: None,
                        insert_index: 0,
                        delta: self.singleton_route_delta(selected_node),
                    },
                )
            } else {
                self.best_initial_insertion(&unvisited, &routes, &route_loads, rng)
            };
            let node = unvisited.swap_remove(selected_idx);
            debug_assert_eq!(node, selected_node);
            let demand = self.graph.get_demand(node);

            match selected.route_index {
                Some(route_idx) => {
                    routes[route_idx].insert(selected.insert_index, node);
                    route_loads[route_idx] += demand;
                }
                None => {
                    routes.push(vec![node]);
                    route_loads.push(demand);
                }
            }
        }

        let mut order: Vec<u32> = routes.into_iter().flatten().collect();
        self.refine_with_sampled_reinsertion(
            &mut order,
            self.init_insertion_strategy,
            self.init_weighted_weights,
            12,
            rng,
        );
        order
    }

    fn best_initial_insertion<R: Rng + ?Sized>(
        &self,
        unvisited: &[u32],
        routes: &[Vec<u32>],
        route_loads: &[u32],
        rng: &mut R,
    ) -> (u32, usize, InsertionCandidate) {
        let mut scored_candidates: Vec<(u32, usize, InsertionCandidate)> = Vec::new();

        for (node_idx, &node) in unvisited.iter().enumerate() {
            let demand = self.graph.get_demand(node);
            for (route_idx, route) in routes.iter().enumerate() {
                if route_loads[route_idx] + demand > self.max_capacity {
                    continue;
                }
                for (insert_index, delta) in self.route_insertion_candidates(route, node) {
                    scored_candidates.push((
                        node,
                        node_idx,
                        InsertionCandidate {
                            route_index: Some(route_idx),
                            insert_index,
                            delta,
                        },
                    ));
                }
            }
            scored_candidates.push((
                node,
                node_idx,
                InsertionCandidate {
                    route_index: None,
                    insert_index: 0,
                    delta: self.singleton_route_delta(node),
                },
            ));
        }

        let deltas: Vec<(f64, f64)> = scored_candidates
            .iter()
            .map(|(_, _, candidate)| candidate.delta)
            .collect();
        let chosen_idx = Self::choose_candidate_index(
            &deltas,
            self.init_insertion_strategy,
            self.init_weighted_weights,
            rng,
        );
        scored_candidates[chosen_idx]
    }

    fn produce_child_population<R: Rng + ?Sized>(
        &self,
        parents: &[Chromosome],
        rng: &mut R,
    ) -> Vec<Chromosome> {
        let n = self.population_size as usize;
        let mut children = Vec::with_capacity(n);

        while children.len() < n {
            let p1 = self.tournament_select(parents, rng);
            let p2 = self.tournament_select(parents, rng);

            let (mut c1, mut c2) = if rng.random_bool(self.p_crossover) {
                self.crossover(p1, p2, rng)
            } else {
                (p1.clone(), p2.clone())
            };

            if rng.random_bool(self.p_mutation) {
                self.mutate(&mut c1, rng);
            }
            if rng.random_bool(self.p_mutation) {
                self.mutate(&mut c2, rng);
            }

            c1.fitness_values = self.evaluate(&c1.order_genes);
            children.push(c1);
            if children.len() < n {
                c2.fitness_values = self.evaluate(&c2.order_genes);
                children.push(c2);
            }
        }

        children
    }

    fn tournament_select<'a, R: Rng + ?Sized>(
        &self,
        population: &'a [Chromosome],
        rng: &mut R,
    ) -> &'a Chromosome {
        let n = population.len();
        let i = rng.random_range(0..n);
        let j = rng.random_range(0..n);

        let a = &population[i];
        let b = &population[j];

        if a.rank < b.rank {
            a
        } else if a.rank > b.rank {
            b
        } else if a.crowding_distance > b.crowding_distance {
            a
        } else if a.crowding_distance < b.crowding_distance {
            b
        } else if rng.random_bool(0.5) {
            a
        } else {
            b
        }
    }

    fn calculate_crowding_distances(&self, population: &mut [Chromosome], fronts: &[Vec<usize>]) {
        for ch in population.iter_mut() {
            ch.crowding_distance = 0.0;
        }

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

    fn sort_population(&self, population: &mut [Chromosome]) -> Vec<Vec<usize>> {
        let n = population.len();
        for ch in population.iter_mut() {
            ch.rank = 0;
        }

        let mut domination_count = vec![0usize; n];
        let mut dominates = vec![Vec::new(); n];

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
        let mut first = Vec::new();
        for i in 0..n {
            if domination_count[i] == 0 {
                population[i].rank = 1;
                first.push(i);
            }
        }
        if first.is_empty() {
            for i in 0..n {
                population[i].rank = 1;
                first.push(i);
            }
        }
        fronts.push(first);

        let mut front_idx = 0usize;
        while front_idx < fronts.len() {
            let mut next = Vec::new();
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

            let mut ord = front.clone();
            ord.sort_by(|&a, &b| {
                population[b]
                    .crowding_distance
                    .total_cmp(&population[a].crowding_distance)
            });
            new_population.extend(
                ord.into_iter()
                    .take(remaining)
                    .map(|i| population[i].clone()),
            );
            break;
        }

        new_population
    }

    fn crossover<R: Rng + ?Sized>(
        &self,
        p1: &Chromosome,
        p2: &Chromosome,
        rng: &mut R,
    ) -> (Chromosome, Chromosome) {
        let c1 = self.hgrex(&p1.order_genes, &p2.order_genes, rng);
        let c2 = self.hgrex(&p2.order_genes, &p1.order_genes, rng);

        let mut child1 = p1.clone();
        child1.order_genes = c1;
        self.refine_with_sampled_reinsertion(
            &mut child1.order_genes,
            InsertionStrategy::BestDistance,
            (1.0, 0.0),
            self.mutation_refine_sample_size,
            rng,
        );
        child1.fitness_values = (0.0, 0.0);
        child1.rank = 0;
        child1.crowding_distance = 0.0;

        let mut child2 = p2.clone();
        child2.order_genes = c2;
        self.refine_with_sampled_reinsertion(
            &mut child2.order_genes,
            InsertionStrategy::BestDistance,
            (1.0, 0.0),
            self.mutation_refine_sample_size,
            rng,
        );
        child2.fitness_values = (0.0, 0.0);
        child2.rank = 0;
        child2.crowding_distance = 0.0;

        (child1, child2)
    }

    fn edge_costs_or_zero(&self, from: u32, to: u32) -> (f64, f64) {
        if from == to {
            return (0.0, 0.0);
        }
        let edge_id = self.adjacency_matrix[from as usize][to as usize][0];
        *self.graph.get_edge_parameters(edge_id)
    }

    fn route_insertion_candidates(&self, route: &[u32], node: u32) -> Vec<(usize, (f64, f64))> {
        let mut candidates = Vec::with_capacity(route.len() + 1);
        for ins in 0..=route.len() {
            let prev = if ins == 0 { 0 } else { route[ins - 1] };
            let next = if ins == route.len() { 0 } else { route[ins] };
            candidates.push((ins, self.insertion_delta(prev, node, next)));
        }
        candidates
    }

    fn insertion_delta(&self, prev: u32, node: u32, next: u32) -> (f64, f64) {
        let prev_to_next = self.edge_costs_or_zero(prev, next);
        let prev_to_node = self.edge_costs_or_zero(prev, node);
        let node_to_next = self.edge_costs_or_zero(node, next);
        (
            prev_to_node.0 + node_to_next.0 - prev_to_next.0,
            prev_to_node.1 + node_to_next.1 - prev_to_next.1,
        )
    }

    fn singleton_route_delta(&self, node: u32) -> (f64, f64) {
        let depot_to = self.edge_costs_or_zero(0, node);
        let to_depot = self.edge_costs_or_zero(node, 0);
        (depot_to.0 + to_depot.0, depot_to.1 + to_depot.1)
    }

    fn choose_candidate_index<R: Rng + ?Sized>(
        deltas: &[(f64, f64)],
        strategy: InsertionStrategy,
        weights: (f64, f64),
        rng: &mut R,
    ) -> usize {
        assert!(!deltas.is_empty());
        if deltas.len() == 1 {
            return 0;
        }
        if strategy == InsertionStrategy::CheapRandom {
            return rng.random_range(0..deltas.len());
        }

        let min0 = deltas.iter().map(|delta| delta.0).fold(f64::INFINITY, f64::min);
        let max0 = deltas
            .iter()
            .map(|delta| delta.0)
            .fold(f64::NEG_INFINITY, f64::max);
        let min1 = deltas.iter().map(|delta| delta.1).fold(f64::INFINITY, f64::min);
        let max1 = deltas
            .iter()
            .map(|delta| delta.1)
            .fold(f64::NEG_INFINITY, f64::max);
        let range0 = max0 - min0;
        let range1 = max1 - min1;

        let mut best_index = 0usize;
        let mut best_score = f64::INFINITY;
        let mut best_delta = deltas[0];

        for (idx, delta) in deltas.iter().copied().enumerate() {
            let norm0 = if range0 > 0.0 {
                (delta.0 - min0) / range0
            } else {
                0.0
            };
            let norm1 = if range1 > 0.0 {
                (delta.1 - min1) / range1
            } else {
                0.0
            };

            let score = match strategy {
                InsertionStrategy::BestDistance => delta.0,
                InsertionStrategy::BestSecondary => delta.1,
                InsertionStrategy::BestNormalized => norm0 + norm1,
                InsertionStrategy::BestWeighted => weights.0 * norm0 + weights.1 * norm1,
                InsertionStrategy::CheapRandom => unreachable!(),
            };

            let is_better = score < best_score
                || (score == best_score
                    && (delta.0 < best_delta.0
                        || (delta.0 == best_delta.0
                            && (delta.1 < best_delta.1
                                || (delta.1 == best_delta.1 && idx < best_index)))));

            if is_better {
                best_index = idx;
                best_score = score;
                best_delta = delta;
            }
        }

        best_index
    }

    fn build_parent_info(&self, order: &[u32]) -> Option<ParentInfo> {
        let splits = self.route_splits(order)?;
        let mut pos = vec![None; self.num_nodes];
        for (i, &node) in order.iter().enumerate() {
            if (node as usize) < pos.len() {
                pos[node as usize] = Some(i);
            }
        }

        let mut route_end = vec![order.len(); order.len()];
        let mut route_starts = Vec::with_capacity(splits.len());
        for (s, e) in splits {
            if s < e && e <= order.len() {
                route_starts.push(order[s]);
                for value in route_end.iter_mut().take(e).skip(s) {
                    *value = e;
                }
            }
        }

        Some(ParentInfo {
            order: order.to_vec(),
            pos,
            route_end,
            route_starts,
        })
    }

    fn parent_next_candidate(
        &self,
        info: &ParentInfo,
        current: u32,
        visited: &[bool],
    ) -> Option<u32> {
        if current == 0 {
            for &s in &info.route_starts {
                if (s as usize) < visited.len() && !visited[s as usize] {
                    return Some(s);
                }
            }
            return None;
        }

        let idx = *info.pos.get(current as usize)?.as_ref()?;
        let end = *info.route_end.get(idx)?;
        for &node in &info.order[(idx + 1)..end] {
            if (node as usize) < visited.len() && !visited[node as usize] {
                return Some(node);
            }
        }
        None
    }

    fn parent_route_start_candidates(
        &self,
        info: &ParentInfo,
        visited: &[bool],
        remaining_capacity: u32,
    ) -> Vec<u32> {
        info.route_starts
            .iter()
            .copied()
            .filter(|&node| {
                (node as usize) < visited.len()
                    && !visited[node as usize]
                    && self.graph.get_demand(node) <= remaining_capacity
            })
            .collect()
    }

    fn append_extension_score(&self, current: u32, node: u32) -> f64 {
        let current_to_node = self.edge_costs_or_zero(current, node).0;
        let node_to_depot = self.edge_costs_or_zero(node, 0).0;
        let current_to_depot = self.edge_costs_or_zero(current, 0).0;
        current_to_node + node_to_depot - current_to_depot
    }

    fn choose_best_extension_candidate(&self, current: u32, candidates: &[u32]) -> Option<u32> {
        let mut best: Option<(u32, f64)> = None;
        for &node in candidates {
            let score = self.append_extension_score(current, node);
            match best {
                None => best = Some((node, score)),
                Some((best_node, best_score))
                    if score < best_score || (score == best_score && node < best_node) =>
                {
                    best = Some((node, score))
                }
                _ => {}
            }
        }
        best.map(|(node, _)| node)
    }

    fn hgrex<R: Rng + ?Sized>(&self, p1: &[u32], p2: &[u32], rng: &mut R) -> Vec<u32> {
        let n_customers = self.num_nodes.saturating_sub(1);
        if n_customers == 0 {
            return Vec::new();
        }

        let Some(info1) = self.build_parent_info(p1) else {
            return self.ox1(p1, p2, rng);
        };
        let Some(info2) = self.build_parent_info(p2) else {
            return self.ox1(p1, p2, rng);
        };

        let mut visited = vec![false; self.num_nodes];
        visited[0] = true;

        let mut child = Vec::with_capacity(n_customers);
        let mut current = 0u32;
        let mut used_capacity = 0u32;

        while child.len() < n_customers {
            let remaining = self.max_capacity.saturating_sub(used_capacity);

            let mut parent_candidates = Vec::new();
            if current == 0 {
                parent_candidates.extend(
                    self.parent_route_start_candidates(&info1, &visited, remaining)
                        .into_iter(),
                );
                parent_candidates.extend(
                    self.parent_route_start_candidates(&info2, &visited, remaining)
                        .into_iter(),
                );
            } else {
                if let Some(candidate) = self
                    .parent_next_candidate(&info1, current, &visited)
                    .filter(|&c| self.graph.get_demand(c) <= remaining)
                {
                    parent_candidates.push(candidate);
                }
                if let Some(candidate) = self
                    .parent_next_candidate(&info2, current, &visited)
                    .filter(|&c| self.graph.get_demand(c) <= remaining)
                {
                    parent_candidates.push(candidate);
                }
            }
            parent_candidates.sort_unstable();
            parent_candidates.dedup();

            let mut chosen = self.choose_best_extension_candidate(current, &parent_candidates);

            if chosen.is_none() {
                let mut fallback_candidates = Vec::new();
                for node in 1..(self.num_nodes as u32) {
                    if visited[node as usize] {
                        continue;
                    }
                    let demand = self.graph.get_demand(node);
                    if demand > remaining {
                        continue;
                    }
                    fallback_candidates.push(node);
                }
                chosen = self.choose_best_extension_candidate(current, &fallback_candidates);
            }

            let Some(next) = chosen else {
                current = 0;
                used_capacity = 0;
                continue;
            };

            visited[next as usize] = true;
            used_capacity += self.graph.get_demand(next);
            child.push(next);
            current = next;
        }

        child
    }

    fn ox1<T: Copy + Eq + Hash, R: Rng + ?Sized>(&self, p1: &[T], p2: &[T], rng: &mut R) -> Vec<T> {
        let n = p1.len();
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
        let mut used = HashSet::with_capacity(b - a);
        for i in a..b {
            let gene = p1[i];
            child[i] = Some(gene);
            used.insert(gene);
        }

        let mut write = b % n;
        for k in 0..n {
            let gene = p2[(b + k) % n];
            if used.contains(&gene) {
                continue;
            }
            while child[write].is_some() {
                write = (write + 1) % n;
            }
            child[write] = Some(gene);
            used.insert(gene);
        }

        child.into_iter().map(|value| value.unwrap()).collect()
    }

    fn mutate<R: Rng + ?Sized>(&self, chromosome: &mut Chromosome, rng: &mut R) {
        let v = &mut chromosome.order_genes;
        let n = v.len();
        if n < 2 {
            return;
        }

        let before = v.clone();
        let Some(splits) = self.route_splits(v) else {
            let mut i = rng.random_range(0..n);
            let mut j = rng.random_range(0..n);
            if i > j {
                std::mem::swap(&mut i, &mut j);
            }
            if i != j {
                v[i..=j].reverse();
            }
            return;
        };

        let op = rng.random_range(0..100);
        let inversion_threshold = self.mutation_inversion_pct;
        let relocate_threshold = inversion_threshold + self.mutation_relocate_pct;
        let swap_threshold = relocate_threshold + self.mutation_swap_pct;
        let or_opt_threshold = swap_threshold + self.mutation_or_opt_pct;
        let destroy_repair_threshold = or_opt_threshold + self.mutation_destroy_repair_pct;
        debug_assert_eq!(destroy_repair_threshold, 100);

        if op < inversion_threshold {
            let eligible: Vec<(usize, usize)> = splits
                .iter()
                .copied()
                .filter(|(s, e)| e.saturating_sub(*s) >= 2)
                .collect();

            if eligible.is_empty() {
                let i = rng.random_range(0..n);
                let mut j = rng.random_range(0..n);
                if j == i {
                    j = (j + 1) % n;
                }
                v.swap(i, j);
            } else {
                let (s, e) = eligible[rng.random_range(0..eligible.len())];
                let mut i = rng.random_range(s..e);
                let mut j = rng.random_range(s..e);
                if i > j {
                    std::mem::swap(&mut i, &mut j);
                }
                if i != j {
                    v[i..=j].reverse();
                }
            }
        } else if op < relocate_threshold {
            let src_r = rng.random_range(0..splits.len());
            let (src_s, src_e) = splits[src_r];
            let src_idx = rng.random_range(src_s..src_e);
            let node = v.remove(src_idx);

            let insert_index = self.best_reinsert_index_exact(
                v,
                node,
                self.relocate_insertion_strategy,
                self.relocate_weighted_weights,
                rng,
            );
            v.insert(insert_index, node);
        } else if op < swap_threshold {
            if splits.len() >= 2 {
                let r1 = rng.random_range(0..splits.len());
                let mut r2 = rng.random_range(0..splits.len());
                if r2 == r1 {
                    r2 = (r2 + 1) % splits.len();
                }
                let (s1, e1) = splits[r1];
                let (s2, e2) = splits[r2];
                let i = rng.random_range(s1..e1);
                let j = rng.random_range(s2..e2);
                v.swap(i, j);
            } else {
                let i = rng.random_range(0..n);
                let mut j = rng.random_range(0..n);
                if j == i {
                    j = (j + 1) % n;
                }
                v.swap(i, j);
            }
        } else if op < or_opt_threshold {
            self.apply_or_opt_mutation(v, &splits, rng);
        } else {
            self.apply_destroy_repair_mutation(v, rng);
        }

        if self.route_splits(v).is_none() {
            *v = before;
            return;
        }

        self.refine_with_sampled_reinsertion(
            v,
            self.relocate_insertion_strategy,
            self.relocate_weighted_weights,
            self.mutation_refine_sample_size,
            rng,
        );
    }

    fn best_reinsert_index_exact<R: Rng + ?Sized>(
        &self,
        order_after_removal: &[u32],
        node: u32,
        strategy: InsertionStrategy,
        weights: (f64, f64),
        rng: &mut R,
    ) -> usize {
        if strategy == InsertionStrategy::CheapRandom {
            return rng.random_range(0..=order_after_removal.len());
        }

        let mut candidate_scores = Vec::with_capacity(order_after_removal.len() + 1);
        for insert_index in 0..=order_after_removal.len() {
            let mut candidate = order_after_removal.to_vec();
            candidate.insert(insert_index, node);
            candidate_scores.push(self.evaluate(&candidate));
        }

        Self::choose_candidate_index(&candidate_scores, strategy, weights, rng)
    }

    fn best_reinsert_segment_index_exact<R: Rng + ?Sized>(
        &self,
        order_after_removal: &[u32],
        segment: &[u32],
        strategy: InsertionStrategy,
        weights: (f64, f64),
        rng: &mut R,
    ) -> usize {
        if segment.is_empty() {
            return 0;
        }
        if strategy == InsertionStrategy::CheapRandom {
            return rng.random_range(0..=order_after_removal.len());
        }

        let mut candidate_scores = Vec::with_capacity(order_after_removal.len() + 1);
        for insert_index in 0..=order_after_removal.len() {
            let mut candidate = order_after_removal.to_vec();
            candidate.splice(insert_index..insert_index, segment.iter().copied());
            candidate_scores.push(self.evaluate(&candidate));
        }

        Self::choose_candidate_index(&candidate_scores, strategy, weights, rng)
    }

    fn apply_or_opt_mutation<R: Rng + ?Sized>(
        &self,
        order: &mut Vec<u32>,
        splits: &[(usize, usize)],
        rng: &mut R,
    ) {
        let mut eligible = Vec::new();
        for &(start, end) in splits {
            let len = end.saturating_sub(start);
            if len >= 2 {
                for seg_len in 2..=len.min(3) {
                    eligible.push((start, end, seg_len));
                }
            }
        }

        if eligible.is_empty() {
            return;
        }

        let (route_start, route_end, segment_len) = eligible[rng.random_range(0..eligible.len())];
        let src_idx = rng.random_range(route_start..=(route_end - segment_len));
        let segment: Vec<u32> = order[src_idx..src_idx + segment_len].to_vec();
        order.drain(src_idx..src_idx + segment_len);

        let insert_index = self.best_reinsert_segment_index_exact(
            order,
            &segment,
            self.relocate_insertion_strategy,
            self.relocate_weighted_weights,
            rng,
        );
        order.splice(insert_index..insert_index, segment);
    }

    fn apply_destroy_repair_mutation<R: Rng + ?Sized>(&self, order: &mut Vec<u32>, rng: &mut R) {
        let n = order.len();
        if n < 4 {
            return;
        }

        let destroy_count = n.min(6).max(3);
        let remove_count = rng.random_range(3..=destroy_count);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(rng);
        indices.truncate(remove_count);
        indices.sort_unstable();

        let mut removed = Vec::with_capacity(remove_count);
        for idx in indices.into_iter().rev() {
            removed.push(order.remove(idx));
        }
        removed.reverse();
        removed.shuffle(rng);

        for node in removed {
            let insert_index = self.best_reinsert_index_exact(
                order,
                node,
                self.relocate_insertion_strategy,
                self.relocate_weighted_weights,
                rng,
            );
            order.insert(insert_index, node);
        }
    }

    fn refine_with_sampled_reinsertion<R: Rng + ?Sized>(
        &self,
        order: &mut Vec<u32>,
        strategy: InsertionStrategy,
        weights: (f64, f64),
        sample_size: usize,
        rng: &mut R,
    ) {
        if strategy == InsertionStrategy::CheapRandom || order.len() < 3 {
            return;
        }

        let mut sampled_nodes = order.clone();
        sampled_nodes.shuffle(rng);
        sampled_nodes.truncate(sample_size.min(sampled_nodes.len()));

        let mut current_score = self.evaluate(order);
        for node in sampled_nodes {
            let Some(original_index) = order.iter().position(|&value| value == node) else {
                continue;
            };

            order.remove(original_index);
            let best_index =
                self.best_reinsert_index_exact(order, node, strategy, weights, rng);
            order.insert(best_index, node);
            let candidate_score = self.evaluate(order);

            if Self::is_better_score(candidate_score, current_score, strategy, weights) {
                current_score = candidate_score;
            } else {
                order.remove(best_index);
                order.insert(original_index, node);
            }
        }
    }

    fn is_better_score(
        candidate: (f64, f64),
        current: (f64, f64),
        strategy: InsertionStrategy,
        weights: (f64, f64),
    ) -> bool {
        match strategy {
            InsertionStrategy::CheapRandom => false,
            InsertionStrategy::BestDistance => {
                candidate.0 < current.0 || (candidate.0 == current.0 && candidate.1 < current.1)
            }
            InsertionStrategy::BestSecondary => {
                candidate.1 < current.1 || (candidate.1 == current.1 && candidate.0 < current.0)
            }
            InsertionStrategy::BestNormalized | InsertionStrategy::BestWeighted => {
                let candidate_score = weights.0 * candidate.0 + weights.1 * candidate.1;
                let current_score = weights.0 * current.0 + weights.1 * current.1;
                candidate_score < current_score
                    || (candidate_score == current_score
                        && (candidate.0 < current.0
                            || (candidate.0 == current.0 && candidate.1 < current.1)))
            }
        }
    }

    fn decode_routes(&self, order_genes: &[u32]) -> Vec<Vec<u32>> {
        let Some(splits) = self.route_splits(order_genes) else {
            return Vec::new();
        };

        let depot_id = self.graph.get_depot() + 1;
        let mut out = Vec::with_capacity(splits.len());
        for (s, e) in splits {
            let mut route = Vec::with_capacity((e - s) + 2);
            route.push(depot_id);
            for &order_idx in &order_genes[s..e] {
                route.push(order_idx + 1);
            }
            route.push(depot_id);
            out.push(route);
        }
        out
    }

    fn route_splits(&self, order_genes: &[u32]) -> Option<Vec<(usize, usize)>> {
        let n = order_genes.len();
        let mut splits = Vec::new();
        let mut start = 0usize;

        while start < n {
            let mut cap = 0u32;
            let mut end = start;

            while end < n {
                let node = order_genes[end];
                let demand = self.graph.get_demand(node);
                if demand > self.max_capacity {
                    return None;
                }
                if cap + demand > self.max_capacity {
                    break;
                }
                cap += demand;
                end += 1;
            }

            if end == start {
                return None;
            }

            splits.push((start, end));
            start = end;
        }

        Some(splits)
    }

    fn evaluate(&self, order_genes: &[u32]) -> (f64, f64) {
        let mut route_start_index = 0u32;
        let n = order_genes.len() as u32;
        let mut total = (0.0, 0.0);

        while route_start_index < n {
            let mut current_capacity = 0u32;
            let mut route_end_index = route_start_index;

            while route_end_index < n {
                let current_node = order_genes[route_end_index as usize];
                let node_demand = self.graph.get_demand(current_node);
                if node_demand > self.max_capacity {
                    return (f64::INFINITY, f64::INFINITY);
                }
                if current_capacity + node_demand > self.max_capacity {
                    break;
                }
                current_capacity += node_demand;
                route_end_index += 1;
            }

            let mut prev_node = 0u32;
            for idx in route_start_index..route_end_index {
                let node = order_genes[idx as usize];
                let edge_parameters = self.edge_costs_or_zero(prev_node, node);
                total.0 += edge_parameters.0;
                total.1 += edge_parameters.1;
                prev_node = node;
            }
            let back = self.edge_costs_or_zero(prev_node, 0);
            total.0 += back.0;
            total.1 += back.1;

            route_start_index = route_end_index;
        }

        total
    }
}

impl NSGA {
    #[cfg(test)]
    fn test_choose_candidate_index(
        deltas: &[(f64, f64)],
        strategy: InsertionStrategy,
        weights: (f64, f64),
        seed: u64,
    ) -> usize {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::choose_candidate_index(deltas, strategy, weights, &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::{InsertionStrategy, NSGA, RunConfig};
    use crate::graph::Graph;

    #[test]
    fn insertion_strategy_scoring_prefers_expected_candidates() {
        let deltas = [(10.0, 5.0), (8.0, 30.0), (14.0, 1.0)];

        assert_eq!(
            NSGA::test_choose_candidate_index(
                &deltas,
                InsertionStrategy::BestDistance,
                (0.8, 0.2),
                1,
            ),
            1
        );
        assert_eq!(
            NSGA::test_choose_candidate_index(
                &deltas,
                InsertionStrategy::BestSecondary,
                (0.8, 0.2),
                1,
            ),
            2
        );
        assert_eq!(
            NSGA::test_choose_candidate_index(
                &deltas,
                InsertionStrategy::BestNormalized,
                (0.8, 0.2),
                1,
            ),
            0
        );
        assert_eq!(
            NSGA::test_choose_candidate_index(
                &deltas,
                InsertionStrategy::BestWeighted,
                (0.8, 0.2),
                1,
            ),
            1
        );
    }

    #[test]
    fn solver_is_deterministic_for_fixed_seeds() {
        let graph_a = Graph::new_with_seed("xset/X-n101-k25.vrp", 1).unwrap();
        let graph_b = Graph::new_with_seed("xset/X-n101-k25.vrp", 1).unwrap();
        let cfg = RunConfig {
            max_generations: 5,
            hv_window: 3,
            hv_eps: 1e-6,
            log_every: 0,
        };

        let nsga_a = NSGA::new(
            graph_a,
            12,
            0.5,
            0.1,
            0,
            100,
            InsertionStrategy::BestWeighted,
            InsertionStrategy::BestWeighted,
            55,
            30,
            15,
            0,
            0,
            8,
            11,
            cfg,
        );
        let nsga_b = NSGA::new(
            graph_b,
            12,
            0.5,
            0.1,
            0,
            100,
            InsertionStrategy::BestWeighted,
            InsertionStrategy::BestWeighted,
            55,
            30,
            15,
            0,
            0,
            8,
            11,
            cfg,
        );

        let mut sink_a = std::io::sink();
        let mut sink_b = std::io::sink();
        let result_a = nsga_a.solve_with_writer(&mut sink_a);
        let result_b = nsga_b.solve_with_writer(&mut sink_b);

        let front_a: Vec<(f64, f64)> = result_a
            .pareto_front
            .iter()
            .map(|solution| solution.fitness_values)
            .collect();
        let front_b: Vec<(f64, f64)> = result_b
            .pareto_front
            .iter()
            .map(|solution| solution.fitness_values)
            .collect();

        assert_eq!(front_a, front_b);
        assert_eq!(result_a.generations, result_b.generations);
        assert_eq!(result_a.archive_size, result_b.archive_size);
        assert_eq!(result_a.hv_final, result_b.hv_final);
    }
}
