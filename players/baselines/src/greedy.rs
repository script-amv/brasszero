use brass_engine::observe::Observation;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

use crate::Agent;
use crate::heuristic::value;

/// Scores below the best by at most this much count as tied.
const TIE_EPS: f64 = 1e-9;

/// One-ply greedy: applies each legal move to one determinization and picks the move with the
/// best [`value`] for itself.
pub struct GreedyAgent {
    rng: Pcg64,
}

impl GreedyAgent {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Pcg64::seed_from_u64(seed),
        }
    }

    /// One score per legal move: `value` for `obs.me` after applying it to one determinization.
    pub fn scores(&mut self, obs: &Observation) -> Vec<f64> {
        let base = obs.determinize(self.rng.random::<u64>());
        obs.legal_moves
            .iter()
            .map(|m| {
                let mut g = base.clone();
                g.apply(m)
                    .expect("a legal move applies to a determinization of its observation");
                value(&g, obs.me)
            })
            .collect()
    }
}

impl Agent for GreedyAgent {
    fn name(&self) -> &str {
        "greedy"
    }

    fn act(&mut self, obs: &Observation) -> usize {
        let scores = self.scores(obs);
        let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ties: Vec<usize> = (0..scores.len())
            .filter(|&i| scores[i] >= best - TIE_EPS)
            .collect();
        ties[self.rng.random_range(0..ties.len())]
    }
}
