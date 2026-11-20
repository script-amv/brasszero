//! Play random games with uniformly random legal moves and report statistics.
//!
//! Usage: `brass-random --games N --seed S`. Game `i` uses setup seed `S + i`; move choices
//! come from one RNG seeded with `S`, so a run is reproducible.

use std::process::ExitCode;
use std::time::Instant;

use brass_engine::flow::winners;
use brass_engine::state::new_game;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

/// Moves after which a game counts as stuck (no real game comes close).
const MAX_MOVES: usize = 2000;

fn parse_args() -> Result<(u64, u64), String> {
    let mut games = 100;
    let mut seed = 0;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| -> Result<u64, String> {
            args.next()
                .ok_or(format!("{name} needs a value"))?
                .parse()
                .map_err(|e| format!("{name}: {e}"))
        };
        match a.as_str() {
            "--games" => games = value("--games")?,
            "--seed" => seed = value("--seed")?,
            _ => return Err(format!("unknown argument {a}")),
        }
    }
    Ok((games, seed))
}

fn main() -> ExitCode {
    let (games, seed) = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}\nusage: brass-random --games N --seed S");
            return ExitCode::from(2);
        }
    };
    let mut rng = Pcg64::seed_from_u64(seed);
    let start = Instant::now();
    let (mut total_moves, mut max_moves) = (0usize, 0usize);
    let (mut vp_sum, mut vp_max, mut winner_vp_sum) = (0u64, 0u16, 0u64);
    for i in 0..games {
        let mut g = new_game(seed.wrapping_add(i));
        let mut moves = 0;
        while !g.is_over() {
            let legal = g.legal_moves();
            if legal.is_empty() || moves >= MAX_MOVES {
                eprintln!(
                    "game {i} (seed {}) stuck after {moves} moves",
                    seed.wrapping_add(i)
                );
                return ExitCode::FAILURE;
            }
            let m = &legal[rng.random_range(0..legal.len())];
            g.apply(m).expect("legal move applies");
            moves += 1;
        }
        total_moves += moves;
        max_moves = max_moves.max(moves);
        for ps in &g.players {
            vp_sum += ps.vp as u64;
            vp_max = vp_max.max(ps.vp);
        }
        winner_vp_sum += g.players[winners(&g)[0] as usize].vp as u64;
    }
    let secs = start.elapsed().as_secs_f64();
    let n = games.max(1) as f64;
    println!(
        "games:            {games} (seeds {seed}..{})",
        seed.wrapping_add(games)
    );
    println!("mean VP / player: {:.2}", vp_sum as f64 / (4.0 * n));
    println!("mean winner VP:   {:.2}", winner_vp_sum as f64 / n);
    println!("max VP:           {vp_max}");
    println!(
        "moves per game:   {:.1} mean, {max_moves} max",
        total_moves as f64 / n
    );
    println!("games per second: {:.1} ({secs:.2} s)", games as f64 / secs);
    ExitCode::SUCCESS
}
