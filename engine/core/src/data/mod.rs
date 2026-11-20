//! Static component data for 4-player Brass: Birmingham.
//!
//! Every value is transcribed from the physical components (rulebook, board, player mat)
//! and cross-checked against two independent implementations; disagreements are logged
//! in `RULES-QUESTIONS.md` and listed in `oracles/known_differences.toml`.

use serde::{Deserialize, Serialize};

pub mod board;
pub mod cards;
pub mod tiles;
pub mod tracks;

pub use board::*;
pub use cards::*;
pub use tiles::*;
pub use tracks::*;

/// Index into [`LOCATIONS`].
pub type LocId = u8;
/// Index into [`LINKS`].
pub type LinkId = u8;
/// Index into [`SLOTS`].
pub type SlotId = u8;
/// Seat index, 0..4.
pub type PlayerId = u8;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Industry {
    Cotton,
    Manufacturer,
    Pottery,
    Coal,
    Iron,
    Brewery,
}

impl Industry {
    pub const ALL: [Industry; 6] = [
        Industry::Cotton,
        Industry::Manufacturer,
        Industry::Pottery,
        Industry::Coal,
        Industry::Iron,
        Industry::Brewery,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Era {
    Canal,
    Rail,
}
