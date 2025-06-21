mod data;
mod genetic;
mod peptide;
mod problem;
mod tabu;

use clap::Parser;
use genetic::GeneticAlgorithm;
use peptide::PeptideProblem;
use tabu::TabuSearch;

use crate::problem::TSProblem;

#[derive(Parser)]
struct Args {
    /// RNG seed
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// default number of iterations
    #[arg(long, default_value_t = 20000)]
    iters: usize,
    #[arg(long)]
    genetic: bool,
    #[arg(long, default_value_t = 100)]
    pop_size: usize,
    #[arg(long, default_value_t = 100)]
    generations: usize,
    #[arg(long, default_value_t = 0.8)]
    crossover_prob: f64,
    #[arg(long, default_value_t = 0.2)]
    mutation_prob: f64,
    #[arg(long, default_value_t = 5)]
    tournament_size: usize,
    #[arg(long)]
    motif: Option<usize>,
    #[arg(long)]
    list_motifs: bool,
}

fn main() {
    let args = Args::parse();

    // Handle listing motifs
    if args.list_motifs {
        println!("Available motifs:");
        for (i, motif) in peptide::MOTIFS.iter().enumerate() {
            println!(
                "{}: {}",
                i,
                std::str::from_utf8(motif).unwrap_or("Invalid UTF-8")
            );
        }
        return;
    }

    // Set motif if specified
    if let Some(motif_idx) = args.motif {
        if motif_idx < peptide::MOTIFS.len() {
            peptide::set_motif(motif_idx);
            println!(
                "Using motif {}: {}",
                motif_idx,
                std::str::from_utf8(peptide::MOTIFS[motif_idx]).unwrap_or("Invalid UTF-8")
            );
        } else {
            eprintln!("Invalid motif index. Use --list-motifs to see available motifs.");
            return;
        }
    }

    if args.genetic {
        // Run genetic algorithm
        let ga = GeneticAlgorithm {
            population_size: args.pop_size,
            generations: args.generations,
            crossover_prob: args.crossover_prob,
            mutation_prob: args.mutation_prob,
            tournament_size: args.tournament_size,
        };

        let (best, progress) = ga.run(args.seed);

        print!("BEST (fitness={}): ", PeptideProblem::fitness(&best));
        for &aa in best.iter() {
            print!("{}", data::AA_LETTERS[aa as usize] as char);
        }
        println!();

        println!("Generation\tBest\tWorst\tAvg");
        for (gen, best_fit, worst_fit, avg_fit) in progress.iter().step_by(5) {
            println!("{}\t{:.2}\t{:.2}\t{:.2}", gen, best_fit, worst_fit, avg_fit);
        }
    } else {
        // Run tabu search
        let ts = TabuSearch::<PeptideProblem> {
            iterations: args.iters,
            neigh_size: 250,
            tabu_len: 20,
            _phantom: std::marker::PhantomData,
        };
        let (best, trace) = ts.run(args.seed);

        print!("BEST (fitness={}): ", PeptideProblem::fitness(&best));
        for &aa in best.iter() {
            print!("{}", data::AA_LETTERS[aa as usize] as char);
        }
        println!();

        for (it, best_fit) in trace.into_iter().step_by(1000) {
            println!("{it}\t{best_fit}");
        }
    }
    // example command line: cargo run -- --genetic --motif 2 --seed 42 --pop-size 150 --generations 50
}
