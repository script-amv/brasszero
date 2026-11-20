use std::fmt;

use brass_engine::flow::ranking;
use brass_engine::moves::Move;
use brass_engine::state::new_game;

use crate::Agent;

/// Moves after which a game counts as stuck (no real game comes close).
pub const MAX_MOVES: usize = 2000;

#[derive(Debug, Clone, PartialEq)]
pub struct GameRecord {
    pub seed: u64,
    pub moves: Vec<Move>,
    pub ranking: [u8; 4],
    pub vp: [u16; 4],
}

#[derive(Debug, PartialEq)]
pub enum RefereeError {
    BadIndex {
        player: u8,
        index: usize,
        legal: usize,
    },
    TooManyMoves,
}

impl fmt::Display for RefereeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadIndex {
                player,
                index,
                legal,
            } => write!(
                f,
                "player {player} chose move index {index} but only {legal} moves are legal"
            ),
            Self::TooManyMoves => write!(f, "game exceeded {MAX_MOVES} moves"),
        }
    }
}

impl std::error::Error for RefereeError {}

/// Plays one game; agent `i` sits in seat `i`. The referee alone holds the `GameState`.
pub fn play_game(agents: &mut [Box<dyn Agent>; 4], seed: u64) -> Result<GameRecord, RefereeError> {
    let mut g = new_game(seed);
    let mut moves = Vec::new();
    while !g.is_over() {
        if moves.len() >= MAX_MOVES {
            return Err(RefereeError::TooManyMoves);
        }
        let p = g
            .current_player()
            .expect("unfinished game has a player to move");
        let obs = g.observe(p);
        let index = agents[p as usize].act(&obs);
        if index >= obs.legal_moves.len() {
            return Err(RefereeError::BadIndex {
                player: p,
                index,
                legal: obs.legal_moves.len(),
            });
        }
        let m = &obs.legal_moves[index];
        g.apply(m).expect("legal move applies");
        moves.push(m.clone());
    }
    let mut vp = [0u16; 4];
    for (v, ps) in vp.iter_mut().zip(&g.players) {
        *v = ps.vp;
    }
    Ok(GameRecord {
        seed,
        moves,
        ranking: ranking(&g),
        vp,
    })
}
