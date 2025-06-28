use rand::prelude::*;

use crate::nepre;
use crate::peptide::combined_fitness;
use crate::peptide::is_biologically_valid;
use crate::peptide::PeptideProblem;
use crate::problem::TSProblem;

#[derive(Clone, Debug)]
pub struct NeighCfg {
    pub pop_size: usize,
    pub crossover_p: f32,
    pub mutation_p: f32,
    pub smart_xover: bool,
    pub max_gens: usize,
}

impl Default for NeighCfg {
    fn default() -> Self {
        Self {
            pop_size: 400,
            crossover_p: 0.9,
            mutation_p: 0.25,
            smart_xover: true,
            max_gens: 500,
        }
    }
}
pub struct NeighbourGA<'a> {
    problem: &'a PeptideProblem,
    cfg: NeighCfg,
    rng: ThreadRng,
    population: Vec<Vec<u8>>,
    fitness: Vec<f32>,
}

impl<'a> NeighbourGA<'a> {
    pub fn new(problem: &'a PeptideProblem, cfg: NeighCfg) -> Self {
        let mut rng = thread_rng();
        let mut population = Vec::with_capacity(cfg.pop_size);
        for _ in 0..cfg.pop_size {
            population.push(PeptideProblem::random_individual(&mut rng));
        }
        let mut ga = Self {
            problem,
            cfg,
            rng,
            population,
            fitness: Vec::new(),
        };
        ga.evaluate();
        ga
    }

    pub fn run(&mut self) -> Vec<u8> {
        for _gen in 0..self.cfg.max_gens {
            self.step_generation();
        }
        self.best_individual().to_vec()
    }

    pub fn best(&self) -> (usize, f32) {
        self.fitness
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &f)| (i, f))
            .unwrap()
    }

    pub fn best_individual(&self) -> &[u8] {
        let (idx, _) = self.best();
        &self.population[idx]
    }

    fn step_generation(&mut self) {
        let mut next_pop = Vec::with_capacity(self.cfg.pop_size);

        while next_pop.len() < self.cfg.pop_size {
            let p1 = self.tournament_pick(3);
            let p2 = self.tournament_pick(3);

            let parent_a = &self.population[p1];
            let parent_b = &self.population[p2];

            let (mut child_a, mut child_b) = if self.rng.gen::<f32>() < self.cfg.crossover_p {
                if self.cfg.smart_xover {
                    (
                        smart_uniform(parent_a, parent_b, &mut self.rng),
                        smart_uniform(parent_b, parent_a, &mut self.rng),
                    )
                } else {
                    uniform_crossover(parent_a, parent_b, &mut self.rng)
                }
            } else {
                (parent_a.clone(), parent_b.clone())
            };

            mutate_all(&mut child_a, self.cfg.mutation_p, &mut self.rng);
            mutate_all(&mut child_b, self.cfg.mutation_p, &mut self.rng);

            PeptideProblem::repair(&mut child_a);
            PeptideProblem::repair(&mut child_b);

            if self.cfg.smart_xover && self.rng.gen::<f32>() < 0.20 {
                let lc_prob = if child_a.len() <= 5 { 0.60 } else { 0.20 };
                if self.rng.gen::<f32>() < lc_prob {
                    hill_climb_optimize(&mut child_a);
                }
            }
            if self.cfg.smart_xover && self.rng.gen::<f32>() < 0.20 {
                let lc_prob = if child_b.len() <= 5 { 0.60 } else { 0.20 };
                if self.rng.gen::<f32>() < lc_prob {
                    hill_climb_optimize(&mut child_b);
                }
            }

            // —--- Biological-plausibility filter —---
            if !is_biologically_valid(&child_a) {
                child_a = loop {
                    let mut cand = PeptideProblem::random_individual(&mut self.rng);
                    PeptideProblem::repair(&mut cand);
                    if is_biologically_valid(&cand) {
                        break cand;
                    }
                };
            }
            if !is_biologically_valid(&child_b) {
                child_b = loop {
                    let mut cand = PeptideProblem::random_individual(&mut self.rng);
                    PeptideProblem::repair(&mut cand);
                    if is_biologically_valid(&cand) {
                        break cand;
                    }
                };
            }
            // —--- end filter —---

            next_pop.push(child_a);
            if next_pop.len() < self.cfg.pop_size {
                next_pop.push(child_b);
            }
        }

        self.population = next_pop;
        self.evaluate();
        let (best_idx, _) = self.best();
        let elite = self.population[best_idx].clone();
        let pop_len = self.population.len();
        let elite_present = self.population.contains(&elite);
        if !elite_present {
            // ensure not already present
            let rnd_idx = self.rng.gen_range(0..pop_len);
            self.population[rnd_idx] = elite;
        }
    }

    fn tournament_pick(&mut self, k: usize) -> usize {
        let mut best_idx = self.rng.gen_range(0..self.population.len());
        let mut best_fit = self.fitness[best_idx];
        for _ in 1..k {
            let idx = self.rng.gen_range(0..self.population.len());
            if self.fitness[idx] < best_fit {
                best_idx = idx;
                best_fit = self.fitness[idx];
            }
        }
        best_idx
    }

    fn evaluate(&mut self) {
        self.fitness = self
            .population
            .iter()
            .map(|seq| self.fitness_of(seq))
            .collect();
    }

    fn fitness_of(&self, seq: &[u8]) -> f32 {
        combined_fitness(seq)
    }
}

fn hill_climb_optimize(seq: &mut [u8]) {
    let mut best_score = combined_fitness(seq);

    for pos in 0..seq.len() {
        let orig = seq[pos];
        let mut best_local = best_score;
        let mut best_aa = orig;

        // test the 19 alternative amino acids
        for aa in 0u8..20 {
            if aa == orig {
                continue;
            }
            seq[pos] = aa;

            // keep search inside biologically plausible space
            if !is_biologically_valid(seq) {
                continue;
            }

            let score = combined_fitness(seq);
            if score < best_local {
                best_local = score;
                best_aa = aa;
            }
        }

        // commit the best substitution found for this position
        seq[pos] = best_aa;
        best_score = best_local;
    }
}

fn uniform_crossover(a: &[u8], b: &[u8], rng: &mut ThreadRng) -> (Vec<u8>, Vec<u8>) {
    let mut child_a = a.to_vec();
    let mut child_b = b.to_vec();
    for i in 0..a.len() {
        if rng.gen::<bool>() {
            child_a[i] = b[i];
            child_b[i] = a[i];
        }
    }
    (child_a, child_b)
}

fn smart_uniform(parent_a: &[u8], parent_b: &[u8], rng: &mut ThreadRng) -> Vec<u8> {
    let len = parent_a.len();
    let mut child = parent_a.to_vec(); // start as clone of A (cheap)

    for i in 0..len {
        if parent_a[i] == parent_b[i] {
            continue;
        } // no choice to make

        // try allele from B
        let old = child[i];
        child[i] = parent_b[i];
        let fit_b = combined_fitness(&child);

        // keep A's allele
        child[i] = old;
        let fit_a = combined_fitness(&child);

        // choose the better allele (lower energy)
        if fit_b < fit_a {
            child[i] = parent_b[i];
        }
    }

    // randomise the first locus to keep diversity
    if rng.gen::<bool>() {
        child[0] = parent_b[0];
    }

    child
}

fn mutate_all(seq: &mut [u8], p: f32, rng: &mut ThreadRng) {
    if rng.gen::<f32>() < p {
        mutate_substitution(seq, rng);
    }
    if rng.gen::<f32>() < p {
        mutate_inversion(seq, rng);
    }
}

fn mutate_substitution(seq: &mut [u8], rng: &mut ThreadRng) {
    let idx = rng.gen_range(0..seq.len());
    seq[idx] = rng.gen_range(0..20) as u8;
}

fn mutate_inversion(seq: &mut [u8], rng: &mut ThreadRng) {
    if seq.len() < 3 {
        return;
    }
    let i = rng.gen_range(0..seq.len() - 1);
    let j = rng.gen_range(i + 1..seq.len());
    seq[i..=j].reverse();
}
