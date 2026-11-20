use brass_engine::observe::Observation;
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

use crate::Agent;

/// Picks a uniformly random legal move.
pub struct RandomAgent {
    rng: Pcg64,
}

impl RandomAgent {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Pcg64::seed_from_u64(seed),
        }
    }
}

impl Agent for RandomAgent {
    fn name(&self) -> &str {
        "random"
    }

    fn act(&mut self, obs: &Observation) -> usize {
        self.rng.random_range(0..obs.legal_moves.len())
    }
}
