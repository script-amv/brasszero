// SPDX-License-Identifier: Apache-2.0
//! Random-game throughput (plan Task 10): uniformly random legal moves, single thread.
//! Criterion reports elements/s = games per second per core. Results: `engine/PERF.md`.

use std::hint::black_box;

use brass_engine::state::new_game;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

/// Games per benchmark iteration (fixed seeds, so every iteration does the same work).
const GAMES: u64 = 10;

fn play(seed: u64) -> usize {
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

fn random_game_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_games");
    group.throughput(Throughput::Elements(GAMES));
    group.sample_size(10);
    group.bench_function("random_game_throughput", |b| {
        b.iter(|| (0..GAMES).map(|s| play(black_box(s))).sum::<usize>())
    });
    group.finish();
}

criterion_group!(benches, random_game_throughput);
criterion_main!(benches);
