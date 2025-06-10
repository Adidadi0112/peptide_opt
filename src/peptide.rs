use crate::{
    data::{AA_LETTERS, BLOSUM62},
    problem::TSProblem,
};
use lazy_static::lazy_static;
use rand::Rng;

// muszę wrzucić GA i dedykowany GA dla tego problemu

pub fn aa_index(letter: u8) -> usize {
    AA_LETTERS
        .iter()
        .position(|&c| c == letter)
        .expect("undefined amino acid")
}
// w genetycznym wstawiamy blanki, które później do ocenny usuwamy. z nimi się łatwiej cossuje i mutuje
const MOTIF: &[u8] = b"GGAGGVGKS"; // więcej ni 10 motywów na przynajmniej 2-3 grupy do testowania

lazy_static! {
    // change motif into amino acid indexes
    static ref MOTIF_IDX: Vec<u8> = {
        MOTIF
            .iter()
            .map(|&c| aa_index(c) as u8)
            .collect()
    };
}

#[derive(Clone, PartialEq)]
// possible sequence modifications
pub enum Move {
    Swap { p1: usize, p2: usize },
    Subst { pos: usize, old: u8, new: u8 },
    Insert { pos: usize, aa: u8 },
    Delete { pos: usize, aa: u8 },
}

pub struct PeptideProblem;

impl PeptideProblem {
    // calculate the energy of a peptide sequence
    // based on the BLOSUM62 matrix and the motif
    fn energy(ind: &[u8]) -> i32 {
        ind.iter()
            .enumerate()
            .map(|(i, &aa)| {
                let a = aa as usize;
                let b = MOTIF_IDX[i % MOTIF_IDX.len()] as usize;
                -(BLOSUM62[a][b] as i32)
            })
            .sum()
    }
}

impl TSProblem for PeptideProblem {
    type Individ = Vec<u8>;
    type Move = Move;

    fn random_individual<R: Rng>(rng: &mut R) -> Self::Individ {
        let len = rng.gen_range(8..=14);
        (0..len).map(|_| rng.gen_range(0..20) as u8).collect()
    }

    fn fitness(ind: &Self::Individ) -> f64 {
        Self::energy(ind) as f64
    }

    fn neighbourhood<R: Rng>(
        rng: &mut R,
        ind: &Self::Individ,
        size: usize,
    ) -> Vec<(Self::Individ, Self::Move)> {
        let mut out = Vec::with_capacity(size);

        for _ in 0..size {
            let mut neigh = ind.clone();
            let r: f64 = rng.gen();

            if r < 0.35 {
                // ---------- SUBST ----------
                let pos = rng.gen_range(0..neigh.len());
                let old = neigh[pos];
                let mut new = rng.gen_range(0..20) as u8;
                while new == old {
                    new = rng.gen_range(0..20) as u8;
                }
                neigh[pos] = new;
                out.push((neigh, Move::Subst { pos, old, new }));
            } else if r < 0.55 && neigh.len() < 16 {
                // ---------- INSERT ----------
                let pos = rng.gen_range(0..=neigh.len());
                let aa = rng.gen_range(0..20) as u8;
                neigh.insert(pos, aa);
                out.push((neigh, Move::Insert { pos, aa }));
            } else if r < 0.75 && neigh.len() > 8 {
                // ---------- DELETE ----------
                let pos = rng.gen_range(0..neigh.len());
                let aa = neigh.remove(pos);
                out.push((neigh, Move::Delete { pos, aa }));
            } else {
                // ---------- SWAP ----------
                if neigh.len() >= 2 {
                    let p1 = rng.gen_range(0..neigh.len());
                    let mut p2 = rng.gen_range(0..neigh.len());
                    while p2 == p1 {
                        p2 = rng.gen_range(0..neigh.len());
                    }
                    neigh.swap(p1, p2);
                    out.push((neigh, Move::Swap { p1, p2 }));
                }
            }
        }
        out
    }

    fn apply_move(ind: &mut Self::Individ, mv: &Self::Move) {
        match *mv {
            Move::Subst { pos, new, .. } => ind[pos] = new,
            Move::Insert { pos, aa } => ind.insert(pos, aa),
            Move::Delete { pos, .. } => {
                ind.remove(pos);
            }
            Move::Swap { p1, p2 } => ind.swap(p1, p2),
        }
    }
}
