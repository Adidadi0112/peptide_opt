use crate::peptide::combined_fitness;
use crate::peptide::PeptideProblem;
use crate::problem::TSProblem;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Crossover {
    SinglePoint,
    Uniform,
}

pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub generations: usize,
    pub crossover_prob: f64,
    pub crossover: Crossover,
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
                .map(|ind| combined_fitness(ind) as f64)
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
                    .partial_cmp(&(combined_fitness(b) as f64))
                    .unwrap()
            })
            .unwrap();

        (*best).clone()
    }

    fn crossover<R: Rng>(&self, parent1: &Vec<u8>, parent2: &Vec<u8>, rng: &mut R) -> Vec<u8> {
        if rng.gen::<f64>() < self.crossover_prob {
            match self.crossover {
                Crossover::SinglePoint => {
                    // Single point crossover
                    let point = rng.gen_range(1..parent1.len().min(parent2.len()));
                    let mut child = parent1[..point].to_vec();
                    child.extend_from_slice(&parent2[point.min(parent2.len())..]);
                    child
                }
                Crossover::Uniform => {
                    // Uniform crossover with p=0.5
                    let mut child = Vec::with_capacity(parent1.len());
                    for i in 0..parent1.len().min(parent2.len()) {
                        if rng.gen::<f64>() < 0.5 {
                            child.push(parent1[i]);
                        } else {
                            child.push(parent2[i]);
                        }
                    }
                    child
                }
            }
        } else {
            parent1.clone()
        }
    }

    fn mutate<R: Rng>(&self, individual: &mut Vec<u8>, rng: &mut R) {
        if rng.gen::<f64>() < self.mutation_prob {
            // Use one of the mutation operations randomly (only fixed-length operations)
            let r: f64 = rng.gen();

            if r < 0.7 {
                // Substitution mutation
                let pos = rng.gen_range(0..individual.len());
                let old = individual[pos];
                let mut new = rng.gen_range(0..20) as u8;
                while new == old {
                    new = rng.gen_range(0..20) as u8;
                }
                individual[pos] = new;
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
                (PeptideProblem::fitness(a) as f64)
                    .partial_cmp(&(combined_fitness(b) as f64))
                    .unwrap()
            })
            .unwrap()
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_uniform_crossover() {
        let ga = GeneticAlgorithm {
            population_size: 10,
            generations: 1,
            crossover_prob: 1.0, // Always do crossover
            crossover: Crossover::Uniform,
            mutation_prob: 0.0, // No mutation for testing
            tournament_size: 2,
        };

        let parent1 = vec![0, 1, 2, 3, 4];
        let parent2 = vec![5, 6, 7, 8, 9];
        let mut rng = StdRng::seed_from_u64(42);

        let child = ga.crossover(&parent1, &parent2, &mut rng);

        // Check that child has same length as parents
        assert_eq!(child.len(), parent1.len());

        // Check that each position comes from either parent1 or parent2
        for (i, &value) in child.iter().enumerate() {
            assert!(value == parent1[i] || value == parent2[i]);
        }

        println!("Parent1: {:?}", parent1);
        println!("Parent2: {:?}", parent2);
        println!("Child:   {:?}", child);
    }

    #[test]
    fn test_single_point_crossover() {
        let ga = GeneticAlgorithm {
            population_size: 10,
            generations: 1,
            crossover_prob: 1.0, // Always do crossover
            crossover: Crossover::SinglePoint,
            mutation_prob: 0.0, // No mutation for testing
            tournament_size: 2,
        };

        let parent1 = vec![0, 1, 2, 3, 4];
        let parent2 = vec![5, 6, 7, 8, 9];
        let mut rng = StdRng::seed_from_u64(42);

        let child = ga.crossover(&parent1, &parent2, &mut rng);

        // Check that child has same length as parents
        assert_eq!(child.len(), parent1.len());

        println!("Parent1: {:?}", parent1);
        println!("Parent2: {:?}", parent2);
        println!("Child:   {:?}", child);
    }
}
