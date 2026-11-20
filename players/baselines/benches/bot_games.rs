// SPDX-License-Identifier: Apache-2.0
//! Game throughput with the referee and the baseline bots (Stage 1 Task 8). Criterion reports
//! elements/s = games per second per core. Results: `engine/PERF.md`, "Stage 1 re-profile".
//!
//! - `raw_random`: no referee, the `brass-random` loop (`legal_moves()` + `apply()`).
//! - `referee_random`: 4 × `RandomAgent` through `play_game` (adds `observe()` per move).
//! - `referee_greedy`: 4 × `GreedyAgent` through `play_game`.

use std::hint::black_box;

use brass_bots::{Agent, GreedyAgent, RandomAgent, play_game};
use brass_engine::state::new_game;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

/// Games per benchmark iteration (fixed seeds, so every iteration does the same work).
const GAMES: u64 = 10;

/// The splitmix64 finalizer, as in `brass_ai.seeds`.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `GAMES` high-entropy deal seeds, derived like `brass_ai.seeds.game_seeds(0, GAMES)`.
fn seeds() -> Vec<u64> {
    let base = splitmix64(0);
    (0..GAMES).map(|i| splitmix64(base ^ i)).collect()
}

fn raw_random(seed: u64) -> usize {
    let mut g = new_game(seed);
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut moves = 0;
    while !g.is_over() {
        let legal = g.legal_moves();
        let m = &legal[rng.random_range(0..legal.len())];
        g.apply(m).expect("legal move applies");
        moves += 1;
    }
    moves
}

fn referee(seed: u64, agent: fn(u64) -> Box<dyn Agent>) -> usize {
    let mut agents: [Box<dyn Agent>; 4] = std::array::from_fn(|i| agent(seed ^ i as u64));
    let record = play_game(&mut agents, seed).expect("game finishes");
    record.moves.len()
}

fn bot_games(c: &mut Criterion) {
    let seeds = seeds();
    let mut group = c.benchmark_group("bot_games");
    group.throughput(Throughput::Elements(GAMES));
    group.sample_size(10);
    group.bench_function("raw_random", |b| {
        b.iter(|| {
            seeds
                .iter()
                .map(|&s| raw_random(black_box(s)))
                .sum::<usize>()
        })
    });
    group.bench_function("referee_random", |b| {
        b.iter(|| {
            seeds
                .iter()
                .map(|&s| referee(black_box(s), |s| Box::new(RandomAgent::new(s))))
                .sum::<usize>()
        })
    });
    group.bench_function("referee_greedy", |b| {
        b.iter(|| {
            seeds
                .iter()
                .map(|&s| referee(black_box(s), |s| Box::new(GreedyAgent::new(s))))
                .sum::<usize>()
        })
    });
    group.finish();
}

criterion_group!(benches, bot_games);
criterion_main!(benches);
