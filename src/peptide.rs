use crate::{
    data::{AA_LETTERS, BLOSUM62},
    problem::TSProblem,
};
use lazy_static::lazy_static;
use rand::Rng;

// muszę wrzucić GA i dedykowany GA dla tego problemu
// w genetycznym wstawiamy blanki, które później do oceny usuwamy. z nimi się łatwiej crossuje i mutuje

pub fn aa_index(letter: u8) -> usize {
    AA_LETTERS
        .iter()
        .position(|&c| c == letter)
        .expect("undefined amino acid")
}

// Multiple motifs for testing
pub const MOTIFS: [&[u8]; 13] = [
    b"GGAGGVGKS",
    b"RGD",                    // Cell adhesion motif
    b"KDEL",                   // ER retention signal
    b"PKKP",                   // Modified SH3 domain binding
    b"YPAF",                   // Modified sorting signal
    b"DPDGGDGMDDSD",           // Modified calcium binding
    b"CIGCINGSMRKSDWKNHKPWH",  // Modified zinc finger
    b"LPEKAYNLALGRCELMYSHKNL", // Modified leucine zipper
    b"HTH",                    // Helix-turn-helix
    b"YGRKKRRQRRR",            // HIV-1 Tat protein transduction domain
    b"RQIKIWFQNRRMKWKK",       // Antennapedia homeodomain
    b"AGYLLGKLGAALKG",         // Antimicrobial peptide
    b"KWRWKRWKK",              // Cell-penetrating peptide
];

// Default motif index to use if none specified
static mut CURRENT_MOTIF_IDX: usize = 0;

// Set which motif to use
pub fn set_motif(index: usize) {
    if index < MOTIFS.len() {
        unsafe {
            CURRENT_MOTIF_IDX = index;
        }
    }
}

lazy_static! {
    // All motifs converted to amino acid indices
    static ref MOTIF_INDICES: Vec<Vec<u8>> = {
        MOTIFS
            .iter()
            .map(|motif| {
                motif.iter()
                    .map(|&c| aa_index(c) as u8)
                    .collect()
            })
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

pub struct PeptideProblem {
    // No fields needed
}

// Global flag to determine whether to use best motif matching
static mut USE_BEST_MOTIF: bool = false;

// Public function to set the flag
pub fn set_use_best_motif(use_best: bool) {
    unsafe {
        USE_BEST_MOTIF = use_best;
    }
}

// Public function to get the flag value
pub fn get_use_best_motif() -> bool {
    unsafe { USE_BEST_MOTIF }
}

impl PeptideProblem {
    // calculate the energy of a peptide sequence
    // based on the BLOSUM62 matrix and the selected motif
    fn energy(ind: &[u8]) -> i32 {
        // Get the current motif index
        let motif_idx = unsafe { CURRENT_MOTIF_IDX };

        // Use the selected motif's indices
        let motif_indices = &MOTIF_INDICES[motif_idx];

        ind.iter()
            .enumerate()
            .map(|(i, &aa)| {
                let a = aa as usize;
                let b = motif_indices[i % motif_indices.len()] as usize;
                -(BLOSUM62[a][b] as i32)
            })
            .sum()
    }

    // Calculate energy using all motifs and return the best (minimum) value
    fn energy_best_motif(ind: &[u8]) -> i32 {
        (0..MOTIFS.len())
            .map(|motif_idx| {
                let motif_indices = &MOTIF_INDICES[motif_idx];

                ind.iter()
                    .enumerate()
                    .map(|(i, &aa)| {
                        let a = aa as usize;
                        let b = motif_indices[i % motif_indices.len()] as usize;
                        -(BLOSUM62[a][b] as i32)
                    })
                    .sum()
            })
            .min()
            .unwrap_or(0)
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
        // Use best_motif flag to decide which energy function to use
        let use_best = unsafe { USE_BEST_MOTIF };

        if use_best {
            Self::energy_best_motif(ind) as f64
        } else {
            Self::energy(ind) as f64
        }
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
