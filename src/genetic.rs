use crate::peptide::PeptideProblem;
use crate::problem::TSProblem;
use rand::{rngs::StdRng, Rng, SeedableRng};

pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub generations: usize,
    pub crossover_prob: f64,
    pub mutation_prob: f64,
    pub tournament_size: usize,
}

impl GeneticAlgorithm {
    pub fn run(&self, seed: u64) -> (Vec<u8>, Vec<(usize, f64, f64, f64)>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut population = self.initialize_population(&mut rng);
        let mut progress: Vec<(usize, f64, f64, f64)> = Vec::new();

        for i in 0..self.generations {
            population = self.evolve(&population, &mut rng);

            let fitnesses: Vec<f64> = population
                .iter()
                .map(|ind| PeptideProblem::fitness(ind))
                .collect();

            let min = *fitnesses
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let max = *fitnesses
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let avg = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

            progress.push((i, min, max, avg));
        }

        let best = self.get_best_solution(&population);
        (best, progress)
    }

    fn initialize_population<R: Rng>(&self, rng: &mut R) -> Vec<Vec<u8>> {
        (0..self.population_size)
            .map(|_| PeptideProblem::random_individual(rng))
            .collect()
    }

    fn evolve<R: Rng>(&self, population: &Vec<Vec<u8>>, rng: &mut R) -> Vec<Vec<u8>> {
        let mut new_population = Vec::new();

        while new_population.len() < self.population_size {
            let parent1 = self.tournament_selection(population, rng);
            let parent2 = self.tournament_selection(population, rng);
            let mut offspring = self.crossover(&parent1, &parent2, rng);
            self.mutate(&mut offspring, rng);
            new_population.push(offspring);
        }

        new_population
    }

    fn tournament_selection<R: Rng>(&self, population: &Vec<Vec<u8>>, rng: &mut R) -> Vec<u8> {
        let mut tournament = Vec::new();

        for _ in 0..self.tournament_size {
            let idx = rng.gen_range(0..population.len());
            tournament.push(&population[idx]);
        }

        let best = tournament
            .iter()
            .min_by(|a, b| {
                PeptideProblem::fitness(a)
                    .partial_cmp(&PeptideProblem::fitness(b))
                    .unwrap()
            })
            .unwrap();

        (*best).clone()
    }

    fn crossover<R: Rng>(&self, parent1: &Vec<u8>, parent2: &Vec<u8>, rng: &mut R) -> Vec<u8> {
        if rng.gen::<f64>() < self.crossover_prob {
            // Single point crossover
            let point = rng.gen_range(1..parent1.len().min(parent2.len()));
            let mut child = parent1[..point].to_vec();
            child.extend_from_slice(&parent2[point.min(parent2.len())..]);
            child
        } else {
            parent1.clone()
        }
    }

    fn mutate<R: Rng>(&self, individual: &mut Vec<u8>, rng: &mut R) {
        if rng.gen::<f64>() < self.mutation_prob {
            // Use one of the mutation operations randomly
            let r: f64 = rng.gen();

            if r < 0.35 {
                // Substitution mutation
                let pos = rng.gen_range(0..individual.len());
                let old = individual[pos];
                let mut new = rng.gen_range(0..20) as u8;
                while new == old {
                    new = rng.gen_range(0..20) as u8;
                }
                individual[pos] = new;
            } else if r < 0.55 && individual.len() < 16 {
                // Insertion mutation
                let pos = rng.gen_range(0..=individual.len());
                let aa = rng.gen_range(0..20) as u8;
                individual.insert(pos, aa);
            } else if r < 0.75 && individual.len() > 8 {
                // Deletion mutation
                let pos = rng.gen_range(0..individual.len());
                individual.remove(pos);
            } else if individual.len() >= 2 {
                // Swap mutation
                let p1 = rng.gen_range(0..individual.len());
                let mut p2 = rng.gen_range(0..individual.len());
                while p2 == p1 {
                    p2 = rng.gen_range(0..individual.len());
                }
                individual.swap(p1, p2);
            }
        }
    }

    fn get_best_solution(&self, population: &Vec<Vec<u8>>) -> Vec<u8> {
        population
            .iter()
            .min_by(|a, b| {
                PeptideProblem::fitness(a)
                    .partial_cmp(&PeptideProblem::fitness(b))
                    .unwrap()
            })
            .unwrap()
            .clone()
    }
}
