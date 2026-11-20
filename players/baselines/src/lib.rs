//! Baseline bots and the referee. Agents only ever see an `Observation`.

pub mod greedy;
pub mod heuristic;
pub mod random;
pub mod referee;

use brass_engine::observe::Observation;

pub use greedy::GreedyAgent;
pub use random::RandomAgent;
pub use referee::{GameRecord, MAX_MOVES, RefereeError, play_game};

/// A player. It never receives the true game state, only its own observation.
/// Whatever seeds an agent must never be derivable into a game seed (derive it by one-way
/// hashing, not by an invertible mix of the deal seed), or the agent could replay the deal.
pub trait Agent {
    fn name(&self) -> &str;
    /// Index into `obs.legal_moves`. Called only for the player to move (`legal_moves` non-empty).
    fn act(&mut self, obs: &Observation) -> usize;
}
