use crate::problem::TSProblem;
use rand::SeedableRng;
use std::collections::VecDeque;

pub struct TabuSearch<P: TSProblem> {
    pub iterations: usize,
    pub neigh_size: usize,
    pub tabu_len: usize,
    pub(crate) _phantom: std::marker::PhantomData<P>,
}

impl<P: TSProblem> TabuSearch<P> {
    pub fn run(&self, seed: u64) -> (P::Individ, Vec<(usize, f64)>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut best = P::random_individual(&mut rng);
        let mut curr = best.clone();
        let mut best_f = P::fitness(&best);

        // keeps last moves to avoid revisiting them
        let mut tabu: VecDeque<P::Move> = VecDeque::with_capacity(self.tabu_len);

        let mut trace = Vec::new();

        for it in 0..self.iterations {
            // generete neighbourhood
            let neigh = P::neighbourhood(&mut rng, &curr, self.neigh_size);

            // choose the best candidate that is not on tabu list
            let (mut chosen_ind, mut chosen_mv, mut chosen_f) = (None, None, f64::INFINITY);
            for (cand, mv) in neigh {
                if tabu.contains(&mv) && P::fitness(&cand) >= best_f {
                    continue; // skip this move because of tabu
                }
                let f = P::fitness(&cand);

                // aspiration (if tabu move is better than current best)
                let tabu_hit = tabu.contains(&mv);
                let aspiration = f + 1.0 < P::fitness(&curr);
                if tabu_hit && !aspiration {
                    continue;
                }

                if f < chosen_f {
                    chosen_ind = Some(cand); // candidate individual
                    chosen_mv = Some(mv); // candidate move
                    chosen_f = f; // candidate fitness
                }
            }

            if let (Some(ind), Some(mv)) = (chosen_ind, chosen_mv) {
                curr = ind;
                // update tabu list
                if tabu.len() == self.tabu_len {
                    tabu.pop_front();
                }
                tabu.push_back(mv);
            }

            // update global-best
            let curr_f = P::fitness(&curr);
            if curr_f < best_f {
                best = curr.clone();
                best_f = curr_f;
            }
            trace.push((it, best_f));

            // reheat
            if it % 10_000 == 0 && it != 0 {
                tabu.clear();
            }
        }
        (best, trace)
    }
}
