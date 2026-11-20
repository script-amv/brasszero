//! Moves, turn phases and rule errors.
//!
//! [`GameState::legal_moves`](crate::state::GameState::legal_moves) is the single source of
//! truth for what may be played; [`GameState::apply`](crate::state::GameState::apply) rejects
//! anything it does not list. Cards are identified by value: identical cards are
//! interchangeable, so they never produce duplicate moves.

use serde::{Deserialize, Serialize};

use crate::data::{Card, Industry, LinkId, PlayerId, SlotId};
use crate::resources::{BeerSource, ResourcePlan};

/// Where the current turn is within its action flow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    ChooseAction,
    SecondRail,
    DevelopMore,
    SellMore,
    BonusDevelop,
    /// `player` must pay `owed` of negative income by removing industry tiles.
    Shortfall {
        player: PlayerId,
        owed: u16,
    },
    GameOver,
}

/// One decision by the player to act.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Move {
    // Phase::ChooseAction: each consumes one action and discards `card` (Scout: 3 cards).
    Build {
        card: Card,
        industry: Industry,
        slot: SlotId,
        coal: ResourcePlan,
        iron: ResourcePlan,
    },
    Canal {
        card: Card,
        link: LinkId,
    },
    /// Continues in `Phase::SecondRail`.
    Rail {
        card: Card,
        link: LinkId,
        coal: ResourcePlan,
    },
    /// Continues in `Phase::DevelopMore`.
    Develop {
        card: Card,
        industry: Industry,
        iron: ResourcePlan,
    },
    /// Continues in `Phase::SellMore` (or `Phase::BonusDevelop`).
    Sell {
        card: Card,
        slot: SlotId,
        merchant: u8,
        beer: Vec<BeerSource>,
    },
    Loan {
        card: Card,
    },
    /// Sorted, distinct multiset of three cards.
    Scout {
        cards: [Card; 3],
    },
    Pass {
        card: Card,
    },
    // Follow-up phases.
    /// Brewery beer only, never merchant beer.
    SecondRail {
        link: LinkId,
        coal: ResourcePlan,
        beer: SlotId,
    },
    DevelopSecond {
        industry: Industry,
        iron: ResourcePlan,
    },
    SellAnother {
        slot: SlotId,
        merchant: u8,
        beer: Vec<BeerSource>,
    },
    BonusDevelop {
        industry: Industry,
    },
    /// `Phase::Shortfall`: remove one own tile for half its cost (rounded down).
    ShortfallRemove {
        slot: SlotId,
    },
    /// Ends the action; legal in `SecondRail`, `DevelopMore` and `SellMore`.
    EndAction,
}

/// Why [`GameState::apply`](crate::state::GameState::apply) refused a move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleError {
    /// The move is not in `legal_moves()`.
    IllegalMove,
    /// The game has ended; no move is legal.
    GameOver,
}

impl std::fmt::Display for RuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            RuleError::IllegalMove => "move is not legal in this state",
            RuleError::GameOver => "the game is over",
        })
    }
}

impl std::error::Error for RuleError {}
