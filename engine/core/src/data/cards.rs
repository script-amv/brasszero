//! Location and industry cards.

use serde::{Deserialize, Serialize};

use super::LocId;
use super::board::loc::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IndustryCard {
    Iron,
    Coal,
    Pottery,
    Brewery,
    /// The two-icon card: build either a cotton mill or a manufacturer.
    CottonOrManufacturer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Card {
    Location(LocId),
    Industry(IndustryCard),
    WildLocation,
    WildIndustry,
}

/// Location cards in the 4-player deck: (town, copies). 41 cards.
// [C: BGG 6524612][C: BGG 6519129][FAQ: which-cards-are-removed-for-two-or-three-players]
// (docs/rules-reference.md §3.1, RQ-029).
const LOCATION_CARDS_4P: &[(LocId, u8)] = &[
    (BIRMINGHAM, 3),
    (COVENTRY, 3),
    (COALBROOKDALE, 3),
    (DERBY, 3),
    (STOKE_ON_TRENT, 3),
    (BELPER, 2),
    (LEEK, 2),
    (STONE, 2),
    (UTTOXETER, 2),
    (STAFFORD, 2),
    (BURTON_ON_TRENT, 2),
    (CANNOCK, 2),
    (DUDLEY, 2),
    (KIDDERMINSTER, 2),
    (WOLVERHAMPTON, 2),
    (WORCESTER, 2),
    (TAMWORTH, 1),
    (WALSALL, 1),
    (NUNEATON, 1),
    (REDDITCH, 1),
];

/// Industry cards in the 4-player deck: (card, copies). 23 cards.
// Same sources as above (§3.1); wild cards are a separate supply [R p.4] (§3.2).
const INDUSTRY_CARDS_4P: &[(IndustryCard, u8)] = &[
    (IndustryCard::Iron, 4),
    (IndustryCard::Coal, 3),
    (IndustryCard::Pottery, 3),
    (IndustryCard::Brewery, 5),
    (IndustryCard::CottonOrManufacturer, 8),
];

/// The unshuffled 64-card 4-player draw deck (wild cards are a separate supply).
pub fn deck_4p() -> Vec<Card> {
    let locs = LOCATION_CARDS_4P
        .iter()
        .flat_map(|&(l, n)| std::iter::repeat_n(Card::Location(l), n as usize));
    let inds = INDUSTRY_CARDS_4P
        .iter()
        .flat_map(|&(i, n)| std::iter::repeat_n(Card::Industry(i), n as usize));
    locs.chain(inds).collect()
}
