//! NSGA-II style multiobjective evolutionary solver for CVRP.
//!
//! ## Encoding
//! A solution (`Chromosome`) stores a **permutation of customers** in `order_genes`.
//! - Node id `0` is treated as the **depot**.
//! - Customers are `1..num_nodes-1`.
//! - Routes are **implicit**: the permutation is split into consecutive segments using the
//!   capacity constraint (`route_splits`).
//!
//! ## Objectives
//! The underlying instance graph provides 2 edge parameters:
//! - objective 0: Euclidean distance
//! - objective 1: random secondary cost (see `graph.rs`)
//!
//! `evaluate()` computes the sum of both objectives across all routes (including depot legs).
//!
//! ## Variation operators
//! - **Crossover**: HGreX (heuristic greedy crossover), edge-based and capacity-aware.
//! - **Mutation**: route-aware mutation (intra-route 2-opt, relocate, swap).

use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::io::Write;

use crate::graph::Graph;

#[derive(Clone, Debug)]
struct ParentInfo {
    /// Parent permutation (customers only, no depot marker)
    order: Vec<u32>,
    /// For customer node id -> index in `order` (None for depot / absent)
    pos: Vec<Option<usize>>,
    /// For each index i in `order`, stores the exclusive end index of the route containing i
    route_end: Vec<usize>,
    /// Route start customer IDs (in parent order)
    route_starts: Vec<u32>,
}

#[derive(Debug)]
pub struct Chromosome {
    order_genes: Vec<u32>,
    /// For each ordered pair (u, v) (in the NSGA node-index space, incl depot=0),
    /// stores which candidate edge in `adjacency_matrix[u][v]` is chosen.
    /// Flattened row-major matrix of size `num_nodes * num_nodes`.
    fitness_values: (f64, f64),

    crowding_distance: f64,
    rank: usize,
}

impl Clone for Chromosome {
    fn clone(&self) -> Self {
        Self {
            order_genes: self.order_genes.clone(),
            fitness_values: self.fitness_values.clone(),
            crowding_distance: self.crowding_distance,
            rank: self.rank,
        }
    }
}

impl Chromosome {
    #[allow(dead_code)]
    pub fn get_order_genes(&self) -> &Vec<u32> {
        &self.order_genes
    }

    #[allow(dead_code)]
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

pub struct NSGA {
    graph: Graph,

    adjacency_matrix: HashMap<u32, HashMap<u32, Vec<u32>>>,
    num_nodes: usize,
    population_size: u32,

    max_capacity: u32,
    /// Optional hard upper bound on the number of vehicle routes.
    /// If `None`, the solver may use as many vehicles as needed (capacity-driven).
    /// If `Some(k)`, any chromosome that requires more than `k` routes is treated as infeasible.
    max_vehicles: Option<usize>,

    p_crossover: f64,
    p_mutation: f64,
}

impl NSGA {
    /// Create a solver instance from a parsed `Graph`.
    ///
    /// The graph is expected to be a complete directed graph with edge parameters `(f0, f1)`.
    pub fn new(
        graph: Graph,
        population_size: u32,
        p_crossover: f64,
        p_mutation: f64,
        max_vehicles: Option<usize>,
    ) -> Self {
        let mut id_to_order_index = HashMap::new();
        let mut order_index_to_id = HashMap::new();
        let num_nodes = graph.get_num_nodes();

        let depot: u32 = graph.get_depot();
        id_to_order_index.insert(depot, 0);
        order_index_to_id.insert(0, depot);

        let mut adjacency_matrix = HashMap::new();
        for edge in graph.get_edges() {
            let node1 = edge.get_source();
            let node2 = edge.get_target();
            adjacency_matrix.entry(node1).or_insert(HashMap::new()).entry(node2).or_insert(Vec::new()).push(edge.get_id());
        }

        let max_capacity = graph.get_capacity();
        Self {
            graph,
            adjacency_matrix,
            num_nodes: num_nodes as usize,
            population_size,
            max_capacity,
            max_vehicles,
            p_crossover,
            p_mutation,
        }
    }

    #[allow(dead_code)]
    pub fn solve_capacitated_vrp(&self) -> Vec<Chromosome> {
        let stdout = std::io::stdout();
        let mut w = std::io::BufWriter::new(stdout.lock());
        self.solve_capacitated_vrp_with_writer(&mut w)
    }

    /// Solve the CVRP and stream progress logs to `w`.
    pub fn solve_capacitated_vrp_with_writer<W: Write>(&self, w: &mut W) -> Vec<Chromosome> {
        let mut population = self.generate_initial_population();
        let child_population = self.produce_child_population(&population);
        population.extend(child_population);

        let num_iterations = 100000;

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
            writeln!(
                w,
                "Generation {}: Archive size {}, HV: {:.6}",
                generation + 1,
                archive_front.len(),
                _hv - initial_hv
            )
            .ok();
            if converged {
                writeln!(w, "Converged at generation {}", generation + 1).ok();
                population = new_population;
                break;
            }

            population = new_population;
        }

        // Print the actual nondominated solutions (rank 1) as decoded routes.
        let fronts = self.sort_population(&mut population);
        if let Some(first_front) = fronts.first() {
            // Deduplicate by exact fitness pairs so we don't print repeated solutions.
            let mut reps: Vec<usize> = first_front.clone();
            reps.sort_by(|&a, &b| {
                population[a]
                    .fitness_values
                    .0
                    .total_cmp(&population[b].fitness_values.0)
                    .then_with(|| {
                        population[a]
                            .fitness_values
                            .1
                            .total_cmp(&population[b].fitness_values.1)
                    })
            });

            let mut unique: Vec<usize> = Vec::new();
            let mut last_key: Option<(u64, u64)> = None;
            for idx in reps {
                let fv = population[idx].fitness_values;
                let key = (fv.0.to_bits(), fv.1.to_bits());
                if last_key == Some(key) {
                    continue;
                }
                last_key = Some(key);
                unique.push(idx);
            }

            let _ = writeln!(
                w,
                "Final nondominated solutions (rank 1): {} (unique by fitness)",
                unique.len()
            );
            for (k, &idx) in unique.iter().enumerate() {
                let ch = &population[idx];
                writeln!(
                    w,
                    "  Solution {}: ({:.3}, {:.3})",
                    k + 1,
                    ch.fitness_values.0,
                    ch.fitness_values.1
                )
                .ok();
                let routes = self.decode_routes(&ch.order_genes);
                for (ri, route) in routes.iter().enumerate() {
                    writeln!(w, "    Route {}: {:?}", ri + 1, route).ok();
                }
            }
        }

        population
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
            // Order node is a random permutation of ids from 1 to num_nodes
            let mut order_genes: Vec<u32> = Vec::new();
            for node in self.adjacency_matrix.keys() {
                if *node != self.graph.get_depot() {
                    order_genes.push(*node);
                }
            }
            // Shuffle the order genes
            order_genes.shuffle(&mut rng);

            // Compute the fitness values
            let fitness_values = self.evaluate(&order_genes);

            population.push(Chromosome {
                order_genes,
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

            c1.fitness_values = self.evaluate(&c1.order_genes);
            children.push(c1);
            if children.len() < n {
                c2.fitness_values = self.evaluate(&c2.order_genes);
                children.push(c2);
            }
        }

        children
    }

    fn tournament_select<'a>(&self, population: &'a [Chromosome]) -> &'a Chromosome {
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

    /// Apply crossover to produce two children.
    ///
    /// Uses HGreX in both directions to form two offspring.
    fn crossover(&self, p1: &Chromosome, p2: &Chromosome) -> (Chromosome, Chromosome) {
        // HGreX crossover (edge-based greedy using parent adjacencies, capacity-aware)
        let c1 = self.hgrex(&p1.order_genes, &p2.order_genes);
        let c2 = self.hgrex(&p2.order_genes, &p1.order_genes);

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

    fn edge_distance(&self, from: u32, to: u32) -> f64 {
        let edge_id = self.adjacency_matrix[&from][&to][0];
        self.graph.get_edge_parameters(edge_id).0
    }

    fn build_parent_info(&self, order: &[u32]) -> Option<ParentInfo> {
        let splits = self.route_splits(&order.to_vec())?;
        let mut pos: Vec<Option<usize>> = vec![None; self.num_nodes];
        for (i, &node) in order.iter().enumerate() {
            if (node as usize) < pos.len() {
                pos[node as usize] = Some(i);
            }
        }

        let mut route_end: Vec<usize> = vec![order.len(); order.len()];
        let mut route_starts: Vec<u32> = Vec::with_capacity(splits.len());
        for (s, e) in splits {
            if s < e && e <= order.len() {
                route_starts.push(order[s]);
                for i in s..e {
                    route_end[i] = e;
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
        visited: &HashMap<u32, bool>,
    ) -> Option<u32> {
        if current == 0 {
            // depot: choose the first unvisited route-start in this parent
            for &s in &info.route_starts {
                if visited.contains_key(&s) && !visited[&s] {
                    return Some(s);
                }
            }
            return None;
        }

        let idx = *info.pos.get(current as usize)?.as_ref()?;
        let end = *info.route_end.get(idx)?;

        // first unvisited after current within same parent route
        for &node in info.order[(idx + 1)..end].iter() {
            if visited.contains_key(&node) && !visited[&node] {
                return Some(node);
            }
        }
        None
    }

    /// HGreX (heuristic greedy crossover) for the permutation encoding.
    ///
    /// The child is built sequentially by:
    /// - taking each parent's implied "next customer" suggestion from the current node,
    /// - choosing the closer suggestion (objective 0) when feasible w.r.t remaining capacity,
    /// - falling back to the nearest feasible unvisited customer,
    /// - starting a new route when nothing fits.
    ///
    /// Returns a permutation of all customers (length `num_nodes-1`).
    fn hgrex(&self, p1: &[u32], p2: &[u32]) -> Vec<u32> {
        let n_customers = self.num_nodes.saturating_sub(1);
        if n_customers == 0 {
            return Vec::new();
        }

        // If we can't derive parent routes (infeasible), fall back to OX1.
        let Some(info1) = self.build_parent_info(p1) else {
            return self.ox1(p1, p2);
        };
        let Some(info2) = self.build_parent_info(p2) else {
            return self.ox1(p1, p2);
        };

        let mut visited: HashMap<u32, bool> = HashMap::new();
        visited.insert(self.graph.get_depot(), true); // depot
        for node in self.graph.get_goal_nodes() {
            visited.insert(*node, false);
        }

        let mut child: Vec<u32> = Vec::with_capacity(n_customers);
        let mut current: u32 = 0;
        let mut used_capacity: u32 = 0;

        while child.len() < n_customers {
            let remaining = self.max_capacity.saturating_sub(used_capacity);

            let c1 = self
                .parent_next_candidate(&info1, current, &visited)
                .filter(|&c| self.graph.get_demand(c) <= remaining);
            let c2 = self
                .parent_next_candidate(&info2, current, &visited)
                .filter(|&c| self.graph.get_demand(c) <= remaining);

            // pick best of parent candidates by distance
            let mut chosen: Option<u32> = None;
            match (c1, c2) {
                (Some(a), Some(b)) => {
                    let da = self.edge_distance(current, a);
                    let db = self.edge_distance(current, b);
                    chosen = Some(if da <= db { a } else { b });
                }
                (Some(a), None) => chosen = Some(a),
                (None, Some(b)) => chosen = Some(b),
                (None, None) => {}
            }

            // fallback: nearest feasible unvisited
            if chosen.is_none() {
                let mut best: Option<(u32, f64)> = None;
                for node in self.graph.get_goal_nodes() {
                    if visited[&*node] {
                        continue;
                    }
                    let d = self.graph.get_demand(*node);
                    if d > remaining {
                        continue;
                    }
                    let dist = self.edge_distance(current, *node);
                    match best {
                        None => best = Some((*node, dist)),
                        Some((_, best_dist)) if dist < best_dist => best = Some((*node, dist)),
                        _ => {}
                    }
                }
                chosen = best.map(|(n, _)| n);
            }

            // If nothing fits, start a new route (return to depot) and try again.
            let Some(next) = chosen else {
                current = 0;
                used_capacity = 0;
                continue;
            };

            // Append customer to permutation
            visited.insert(next, true);
            used_capacity += self.graph.get_demand(next);
            child.push(next);
            current = next;
        }

        child
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

    /// Route-aware mutation for the permutation encoding.
    ///
    /// Weighted operator mix:
    /// - intra-route 2-opt (segment reversal)
    /// - relocate (remove one customer and insert elsewhere)
    /// - swap (prefer inter-route when multiple routes exist)
    fn mutate(&self, chromosome: &mut Chromosome) {
        // Route-aware mutation for CVRP permutation encoding.
        //
        // Your routes are *implicit* (computed by `route_splits()` using capacity), so mutation
        // only needs to keep a valid permutation; feasibility is guarded by demands <= capacity.
        let v = &mut chromosome.order_genes;
        let n = v.len();
        if n < 2 {
            return;
        }

        let before = v.clone();
        let mut rng = rand::rng();

        // If we cannot compute route splits (should only happen if a demand > capacity),
        // fall back to a simple inversion.
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

        // Weighted choice of operator:
        // - ~55%: intra-route 2-opt (segment reversal)
        // - ~30%: relocate (1-0 insertion), route-aware
        // - ~15%: swap (can be inter-route)
        let op: u32 = rng.random_range(0..100);

        if op < 55 {
            // Intra-route inversion (2-opt) within one inferred route
            let eligible: Vec<(usize, usize)> = splits
                .iter()
                .copied()
                .filter(|(s, e)| e.saturating_sub(*s) >= 2)
                .collect();

            if eligible.is_empty() {
                // fallback: simple swap
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
        } else if op < 85 {
            // Relocate (remove one customer and insert elsewhere), biased by current route splits
            let src_r = rng.random_range(0..splits.len());
            let (src_s, src_e) = splits[src_r];
            let src_idx = rng.random_range(src_s..src_e);
            let node = v.remove(src_idx);

            // pick a target route (possibly same)
            let tgt_r = rng.random_range(0..splits.len());
            let (tgt_s, tgt_e) = splits[tgt_r];
            // insertion position is allowed at end (tgt_e)
            let mut ins = rng.random_range(tgt_s..=tgt_e);
            // account for the removal shifting indices
            if src_idx < ins {
                ins = ins.saturating_sub(1);
            }
            if ins > v.len() {
                ins = v.len();
            }
            v.insert(ins, node);
        } else {
            // Swap two customers (try inter-route if we have >= 2 routes)
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
        }

        // Safety: if mutation produced an infeasible ordering (shouldn't happen unless
        // a node demand exceeds capacity), revert.
        if self.route_splits(v).is_none() {
            *v = before;
        }
    }

    fn decode_routes(&self, order_genes: &[u32]) -> Vec<Vec<u32>> {
        // Decode routes in terms of original graph node IDs (including depot at start/end).
        let Some(splits) = self.route_splits(order_genes) else {
            return Vec::new();
        };

        let depot_id = self.graph.get_depot() + 1;
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(splits.len());
        for (s, e) in splits {
            let mut route: Vec<u32> = Vec::with_capacity((e - s) + 2);
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
        // Returns route segments as half-open ranges [start, end) over `order_genes`.
        // If a single customer demand exceeds capacity, returns None.
        let n = order_genes.len();
        let mut splits: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        let mut route_count: usize = 0;

        while start < n {
            let mut cap: u32 = 0;
            let mut end = start;

            while end < n {
                let node = order_genes[end];
                let d = self.graph.get_demand(node);
                if d > self.max_capacity {
                    return None;
                }
                if cap + d > self.max_capacity {
                    break;
                }
                cap += d;
                end += 1;
            }

            // With the guard above, end should always advance at least once when start < n.
            if end == start {
                return None;
            }

            splits.push((start, end));
            route_count += 1;

            if let Some(max_routes) = self.max_vehicles {
                if route_count > max_routes {
                    return None;
                }
            }

            start = end;
        }

        Some(splits)
    }

    fn legs_from_order(&self, order_genes: &[u32]) -> Option<Vec<Vec<(u32, u32)>>> {
        // Route legs including depot legs, grouped per route.
        let splits = self.route_splits(order_genes)?;
        let mut routes: Vec<Vec<(u32, u32)>> = Vec::with_capacity(splits.len());
        for (s, e) in splits {
            let mut legs: Vec<(u32, u32)> = Vec::new();
            let mut prev: u32 = 0;
            for &node in &order_genes[s..e] {
                legs.push((prev, node));
                prev = node;
            }
            legs.push((prev, 0));
            routes.push(legs);
        }
        Some(routes)
    }

    fn parent_leg_set(&self, parent: &Chromosome) -> Option<HashSet<(u32, u32)>> {
        let routes = self.legs_from_order(&parent.order_genes)?;
        let mut set: HashSet<(u32, u32)> = HashSet::new();
        for r in routes {
            for leg in r {
                set.insert(leg);
            }
        }
        Some(set)
    }


    /// Evaluate a permutation by greedily splitting into capacity-feasible routes and summing edge costs.
    ///
    /// Returns `(f0, f1)` where both are minimized.
    fn evaluate(&self, order_genes: &Vec<u32>) -> (f64, f64) {
        // Evaluate the fitness values
        let mut route_start_index: u32 = 0;
        let n: u32 = order_genes.len() as u32;

        let mut total_parameters = (0.0, 0.0);
        let mut routes_used: usize = 0;

        let add_leg = |from: u32, to: u32, total: &mut (f64, f64)| -> bool {
            let from_us = from as usize;
            let to_us = to as usize;
            let edge_id = self.adjacency_matrix[&(from_us as u32)][&(to_us as u32)][0];
            let edge_parameters = self.graph.get_edge_parameters(edge_id);
            total.0 += edge_parameters.0;
            total.1 += edge_parameters.1;
            true
        };

        while route_start_index < n {
            let mut current_capacity: u32 = 0;
            let mut route_end_index: u32 = route_start_index;

            // Build a single vehicle route as a maximal prefix that still fits capacity.
            while route_end_index < n {
                let current_node = order_genes[route_end_index as usize];
                let node_demand = self.graph.get_demand(current_node);

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
                if !add_leg(prev_node, node, &mut total_parameters) {
                    return (f64::INFINITY, f64::INFINITY);
                }
                prev_node = node;
            }
            if !add_leg(prev_node, 0, &mut total_parameters) {
                return (f64::INFINITY, f64::INFINITY);
            }

            routes_used += 1;
            if let Some(max_routes) = self.max_vehicles {
                if routes_used > max_routes {
                    return (f64::INFINITY, f64::INFINITY);
                }
            }

            route_start_index = route_end_index;
        }

        return total_parameters;
    }
}
