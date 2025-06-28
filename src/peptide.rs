use crate::nepre;
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

const HYDROPATHY: [f32; 20] = [
    1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7,
    4.2, -0.9, -1.3,
];

/// Returns `true` iff the peptide passes a few fast heuristics
/// that make it resemble a viable, soluble biological sequence.
///
/// Current rules (easy to tweak):
/// 1. Average hydropathy must be in -1.5 … +3.0  
/// 2. No forbidden adjacent pairs  (“CC” or “PP”)  
/// 3. No homopolymer run ≥ 4 identical residues
pub fn is_biologically_valid(seq: &[u8]) -> bool {
    if seq.is_empty() {
        return false;
    }

    // --- average hydropathy ---
    let avg_hydro: f32 =
        seq.iter().map(|&aa| HYDROPATHY[aa as usize]).sum::<f32>() / (seq.len() as f32);
    if !(-1.5..=3.0).contains(&avg_hydro) {
        return false;
    }

    // --- forbidden adjacent pairs ---
    // C = index 1, P = index 12 in AA_LETTERS
    for win in seq.windows(2) {
        if (win[0] == 1 && win[1] == 1) || (win[0] == 12 && win[1] == 12) {
            return false;
        }
    }

    // --- long homopolymers (≥4) ---
    let mut run = 1usize;
    for i in 1..seq.len() {
        if seq[i] == seq[i - 1] {
            run += 1;
            if run >= 4 {
                return false;
            }
        } else {
            run = 1;
        }
    }

    true
}

/// Combined energy  (lower = better).
/// Decides automatically whether to align against the *current motif*
/// or against *all motifs* (whichever `set_use_best_motif()` selected).
pub fn combined_fitness(seq: &[u8]) -> f32 {
    // --- BLOSUM term ---
    let blosum_e = if get_use_best_motif() {
        PeptideProblem::energy_best_motif(seq) as f32
    } else {
        PeptideProblem::energy(seq) as f32
    };

    // --- NEPRE term (pairwise neighbourhood energy) ---
    let nepre_e: f32 = seq.windows(2).map(|w| nepre::pair(w[0], w[1])).sum();

    blosum_e + nepre_e // we keep “minimise” convention
}

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

// Get current motif length
pub fn current_motif_len() -> usize {
    let motif_idx = unsafe { CURRENT_MOTIF_IDX };
    MOTIFS[motif_idx].len()
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
        let len = current_motif_len();
        (0..len).map(|_| rng.gen_range(0..20) as u8).collect()
    }

    fn fitness(ind: &Self::Individ) -> f64 {
        // Use combined_fitness which already handles the USE_BEST_MOTIF flag internally
        combined_fitness(ind) as f64
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

            if r < 0.7 {
                // ---------- SUBST ----------
                let pos = rng.gen_range(0..neigh.len());
                let old = neigh[pos];
                let mut new = rng.gen_range(0..20) as u8;
                while new == old {
                    new = rng.gen_range(0..20) as u8;
                }
                neigh[pos] = new;
                out.push((neigh, Move::Subst { pos, old, new }));
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
            Move::Swap { p1, p2 } => ind.swap(p1, p2),
            // Insert and Delete operations are no longer supported
            Move::Insert { .. } => panic!("Insert operation not supported with fixed length"),
            Move::Delete { .. } => panic!("Delete operation not supported with fixed length"),
        }
    }

    fn repair(ind: &mut Self::Individ) {
        let target_len = current_motif_len();

        // Ensure the individual has exactly the target length
        if ind.len() < target_len {
            // If too short, extend with random amino acids
            let mut rng = rand::thread_rng();
            while ind.len() < target_len {
                ind.push(rng.gen_range(0..20) as u8);
            }
        } else if ind.len() > target_len {
            // If too long, truncate
            ind.truncate(target_len);
        }
    }
}
