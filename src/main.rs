mod data;
mod ga_neighbour;
mod genetic;
mod nepre;
mod peptide;
mod problem;
mod tabu;

use clap::Parser;
use ga_neighbour::{NeighCfg, NeighbourGA};
use genetic::GeneticAlgorithm;
use peptide::combined_fitness;
use peptide::PeptideProblem;

#[derive(Parser)]
struct Args {
    /// Run every algorithm on every motif
    #[arg(long, default_value_t = false)]
    all: bool,

    /// RNG seed
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// default number of iterations / generations
    #[arg(long, default_value_t = 200)]
    generations: usize,

    /// population size
    #[arg(long, default_value_t = 400)]
    pop_size: usize,

    /// crossover probability
    #[arg(long, default_value_t = 0.9)]
    crossover_prob: f64,

    /// mutation probability
    #[arg(long, default_value_t = 0.3)]
    mutation_prob: f64,

    /// tournament size (GA)
    #[arg(long, default_value_t = 3)]
    tournament_size: usize,

    /// run only the chosen motif (index in MOTIFS)
    #[arg(long)]
    motif: Option<usize>,

    /// list available motifs and exit
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

    // Run both GA algorithms on all motifs for comparison
    println!("=== COMPARATIVE ANALYSIS: Normal GA vs NeighbourGA ===");
    println!("Running on all {} motifs\n", peptide::MOTIFS.len());

    // Results storage
    let mut normal_ga_results = Vec::new();
    let mut neighbour_ga_results = Vec::new();

    let motif_range: Vec<usize> = if let Some(m) = args.motif {
        vec![m]
    } else {
        (0..peptide::MOTIFS.len()).collect()
    };

    for motif_idx in motif_range {
        peptide::set_motif(motif_idx);
        let motif_str = std::str::from_utf8(peptide::MOTIFS[motif_idx]).unwrap_or("Invalid UTF-8");

        println!("=== MOTIF {}: {} ===", motif_idx, motif_str);

        // ============= NORMAL GA =============
        let ga = GeneticAlgorithm {
            population_size: args.pop_size,
            generations: args.generations,
            crossover_prob: args.crossover_prob,
            crossover: genetic::Crossover::SinglePoint,
            mutation_prob: args.mutation_prob,
            tournament_size: args.tournament_size,
        };

        let start_time = std::time::Instant::now();
        let (normal_best, _normal_progress) = ga.run(args.seed + motif_idx as u64);
        let normal_time = start_time.elapsed();
        let normal_fitness = combined_fitness(&normal_best);

        // ============= NEIGHBOUR GA =============
        let problem = PeptideProblem {};
        let neigh_cfg = NeighCfg {
            pop_size: args.pop_size,
            crossover_p: args.crossover_prob as f32,
            mutation_p: args.mutation_prob as f32,
            smart_xover: true,
            max_gens: args.generations,
        };

        let start_time = std::time::Instant::now();
        let mut neigh_ga = NeighbourGA::new(&problem, neigh_cfg);
        let neighbour_best = neigh_ga.run();
        let neighbour_time = start_time.elapsed();
        let neighbour_fitness = combined_fitness(&neighbour_best);

        // ============= RESULTS =============
        println!("Normal GA:");
        print!("  Best sequence (fitness={:.4}): ", normal_fitness);
        for &aa in normal_best.iter() {
            print!("{}", data::AA_LETTERS[aa as usize] as char);
        }
        println!("  (Time: {:.2}s)", normal_time.as_secs_f32());

        println!("NeighbourGA:");
        print!("  Best sequence (fitness={:.4}): ", neighbour_fitness);
        for &aa in neighbour_best.iter() {
            print!("{}", data::AA_LETTERS[aa as usize] as char);
        }
        println!("  (Time: {:.2}s)", neighbour_time.as_secs_f32());

        // Performance comparison
        // Performance comparison (lower fitness = better)
        let improvement = if normal_fitness < neighbour_fitness {
            format!(
                "Normal GA (lower by {:.4})",
                neighbour_fitness - normal_fitness
            )
        } else if neighbour_fitness < normal_fitness {
            format!(
                "NeighbourGA (lower by {:.4})",
                normal_fitness - neighbour_fitness
            )
        } else {
            "Tie".to_string()
        };
        println!("  Winner: {}\n", improvement);

        // Store results for summary
        normal_ga_results.push((
            motif_idx,
            motif_str.to_string(),
            normal_fitness,
            normal_time,
        ));
        neighbour_ga_results.push((
            motif_idx,
            motif_str.to_string(),
            neighbour_fitness,
            neighbour_time,
        ));
    }

    // ============= SUMMARY =============
    println!("=== SUMMARY ===");
    println!(
        "{:<3} {:<25} {:<15} {:<15} {:<20}",
        "ID", "Motif", "Normal GA", "NeighbourGA", "Better (min)"
    );
    println!("{}", "-".repeat(80));

    let mut normal_wins = 0;
    let mut neighbour_wins = 0;
    let mut ties = 0;

    for i in 0..normal_ga_results.len() {
        let (motif_idx, motif_name, normal_fit, _normal_time) = &normal_ga_results[i];
        let (_, _, neighbour_fit, _neighbour_time) = &neighbour_ga_results[i];

        let winner = if normal_fit < neighbour_fit {
            normal_wins += 1;
            "Normal GA"
        } else if neighbour_fit < normal_fit {
            neighbour_wins += 1;
            "NeighbourGA"
        } else {
            ties += 1;
            "Tie"
        };

        println!(
            "{:<3} {:<25} {:<15.4} {:<15.4} {:<20}",
            motif_idx,
            if motif_name.len() > 24 {
                &motif_name[..24]
            } else {
                motif_name
            },
            normal_fit,
            neighbour_fit,
            winner
        );
    }

    println!("\n=== OVERALL STATISTICS ===");
    println!("Normal GA wins:    {}", normal_wins);
    println!("NeighbourGA wins:  {}", neighbour_wins);
    println!("Ties:              {}", ties);

    let avg_normal_time: f32 = normal_ga_results
        .iter()
        .map(|(_, _, _, t)| t.as_secs_f32())
        .sum::<f32>()
        / normal_ga_results.len() as f32;
    let avg_neighbour_time: f32 = neighbour_ga_results
        .iter()
        .map(|(_, _, _, t)| t.as_secs_f32())
        .sum::<f32>()
        / neighbour_ga_results.len() as f32;

    println!("Average Normal GA time:    {:.2}s", avg_normal_time);
    println!("Average NeighbourGA time:  {:.2}s", avg_neighbour_time);

    // Best overall fitness for each algorithm
    let best_normal = normal_ga_results
        .iter()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();
    let best_neighbour = neighbour_ga_results
        .iter()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    println!(
        "\nBest Normal GA result: Motif {} ({}) with fitness {:.4}",
        best_normal.0, best_normal.1, best_normal.2
    );
    println!(
        "Best NeighbourGA result: Motif {} ({}) with fitness {:.4}",
        best_neighbour.0, best_neighbour.1, best_neighbour.2
    );
}
