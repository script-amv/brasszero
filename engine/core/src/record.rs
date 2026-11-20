//! Game records: the initial deal ([`InitSpec`]), one action as played ([`ActionRecord`]),
//! the public-state snapshot for agents ([`PublicSnapshot`]), and the cross-engine comparison
//! views used by the oracle differential harness ([`DiffSnapshot`], [`legal_summary`]).
//!
//! The harness protocol is described in `oracles/protocol.md`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::data::{Card, Era, Industry, LINKS, LOCATIONS, MerchantTile, PlayerId, SLOTS};
use crate::moves::{Move, Phase, RuleError};
use crate::state::{GameState, PlacedTile, from_init};

/// Everything that is random about the start of a game, made explicit so a game can be
/// rebuilt from a record without an RNG (docs/rules-reference.md section 5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitSpec {
    /// Each player's 8-card starting hand.
    pub hands: [Vec<Card>; 4],
    /// The face-down card that starts each player's discard pile.
    pub discard_bottoms: [Card; 4],
    /// The draw deck after dealing (top of deck is the end of the vector).
    pub deck: Vec<Card>,
    /// Merchant tile per flattened merchant tile slot, in `MERCHANT_SPACES` order.
    pub merchant_tiles: Vec<MerchantTile>,
    /// Seat order on the turn-order track.
    pub turn_order: [PlayerId; 4],
    /// Seed that produced this setup; later-era shuffles derive from it (see `GameState`).
    pub rng_seed: u64,
}

/// The public part of one player's state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicPlayer {
    /// Cards in hand (the other cards are private; Wilds are counted below).
    pub hand_size: u8,
    /// Wild Location cards in hand. Public: they came from the public supply (Scout).
    pub wild_location_in_hand: u8,
    /// Wild Industry cards in hand. Public, like `wild_location_in_hand`.
    pub wild_industry_in_hand: u8,
    /// The visible discard pile, oldest first. In the Canal Era the face-down bottom card is
    /// excluded (see `has_face_down_discard`).
    pub discard: Vec<Card>,
    /// Whether a face-down card sits under the visible discards. Nobody, its owner included,
    /// may look at it (rules-reference section 5, setup item 7); the engine keeps it as
    /// `discard[0]` during the Canal Era. Always `false` in the Rail Era.
    pub has_face_down_discard: bool,
    pub mat_remaining: [[u8; 8]; 6],
    pub money: u16,
    pub income_space: u8,
    pub vp: u16,
    pub spent: u16,
}

/// Everything every player may see: all of `GameState` except hand contents, the draw deck's
/// order and contents, face-down discards and RNG internals. Money, discards, hand and deck
/// sizes are public (tournament setting, rules-reference "Spec, Information").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicSnapshot {
    pub era: Era,
    pub round: u8,
    pub turn_order: [PlayerId; 4],
    pub turn_idx: u8,
    pub actions_left: u8,
    pub phase: Phase,
    pub slots: Vec<Option<PlacedTile>>,
    pub links: Vec<Option<PlayerId>>,
    pub merchant_tiles: Vec<MerchantTile>,
    pub merchant_beer: Vec<bool>,
    pub coal_market: u8,
    pub iron_market: u8,
    pub deck_size: u8,
    pub wild_location_supply: u8,
    pub wild_industry_supply: u8,
    pub players: [PublicPlayer; 4],
    pub game_over: bool,
}

/// One full action as played: the `Phase::ChooseAction` move, its follow-ups (second rail,
/// second develop, further sales, the Gloucester develop) and the closing `EndAction`, plus any
/// `ShortfallRemove` moves the action's end of round forces (they are made by the shortfall
/// player, not `player`). A record ends when the game is back in `Phase::ChooseAction` or over
/// (see [`action_complete`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionRecord {
    /// The player taking the action (`current_player()` when the record starts).
    pub player: PlayerId,
    pub moves: Vec<Move>,
}

/// Whether the action in progress is finished: the game is over or waits for a new action.
pub fn action_complete(g: &GameState) -> bool {
    g.is_over() || g.phase == Phase::ChooseAction
}

/// Rebuild a game from its deal and action records, checking each record's player and that
/// each record is exactly one complete action.
pub fn replay(spec: &InitSpec, records: &[ActionRecord]) -> Result<GameState, RuleError> {
    let mut g = from_init(spec);
    for rec in records {
        if g.is_over() {
            return Err(RuleError::GameOver);
        }
        if g.phase != Phase::ChooseAction || g.current_player() != Some(rec.player) {
            return Err(RuleError::IllegalMove);
        }
        for (i, m) in rec.moves.iter().enumerate() {
            if i > 0 && action_complete(&g) {
                // Moves past the end of the action belong to the next record.
                return Err(RuleError::IllegalMove);
            }
            g.apply(m)?;
        }
        if !action_complete(&g) {
            return Err(RuleError::IllegalMove);
        }
    }
    Ok(g)
}

/// The state both engines of the differential harness can express, field for field
/// (`oracles/protocol.md`). Unlike [`PublicSnapshot`] it shows hidden information (hands,
/// face-down discards): both engines know it, and comparing it catches deal and draw bugs.
///
/// Left out, because the TS oracle has no counterpart: `phase` (the oracle resolves a whole
/// action at once, so snapshots are only compared between actions, where ours is always
/// `ChooseAction`), the deck order (the oracle's era-2 reshuffle uses its own RNG; the harness
/// re-deals it from ours), merchant tiles (fixed after setup) and RNG fields.
/// Hands and discard piles are sorted: order has no rules effect (hands are unordered, piles
/// are only ever shuffled), and the engines order the three Scout discards differently. Once the game is over the turn-position fields
/// (`round`, `turn_idx`, `actions_left`) are zeroed: they no longer mean anything and the two
/// engines leave different leftovers in them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffSnapshot {
    pub era: Era,
    /// Round within the era, from 1.
    pub round: u8,
    pub turn_order: [PlayerId; 4],
    pub turn_idx: u8,
    pub actions_left: u8,
    pub game_over: bool,
    pub money: [u16; 4],
    pub income_space: [u8; 4],
    pub vp: [u16; 4],
    pub spent: [u16; 4],
    pub hand_sizes: [u8; 4],
    pub hands: [Vec<Card>; 4],
    /// Each player's discard pile (sorted), including the Canal Era face-down card.
    pub discards: [Vec<Card>; 4],
    pub slots: Vec<Option<PlacedTile>>,
    pub links: Vec<Option<PlayerId>>,
    pub merchant_beer: Vec<bool>,
    pub coal_market: u8,
    pub iron_market: u8,
    pub mat_remaining: [[[u8; 8]; 6]; 4],
    pub deck_size: u8,
    pub wild_location_supply: u8,
    pub wild_industry_supply: u8,
}

/// The [`DiffSnapshot`] of `g`.
pub fn diff_snapshot(g: &GameState) -> DiffSnapshot {
    let over = g.is_over();
    let ps = &g.players;
    DiffSnapshot {
        era: g.era,
        round: if over { 0 } else { g.round },
        turn_order: g.turn_order,
        turn_idx: if over { 0 } else { g.turn_idx },
        actions_left: if over { 0 } else { g.actions_left },
        game_over: over,
        money: std::array::from_fn(|p| ps[p].money),
        income_space: std::array::from_fn(|p| ps[p].income_space),
        vp: std::array::from_fn(|p| ps[p].vp),
        spent: std::array::from_fn(|p| ps[p].spent),
        hand_sizes: std::array::from_fn(|p| ps[p].hand.len() as u8),
        hands: std::array::from_fn(|p| {
            let mut h = ps[p].hand.clone();
            h.sort_unstable();
            h
        }),
        discards: std::array::from_fn(|p| {
            let mut d = ps[p].discard.clone();
            d.sort_unstable();
            d
        }),
        slots: g.slots.clone(),
        links: g.links.clone(),
        merchant_beer: g.merchant_beer.clone(),
        coal_market: g.coal_market,
        iron_market: g.iron_market,
        mat_remaining: std::array::from_fn(|p| ps[p].mat_remaining),
        deck_size: g.deck.len() as u8,
        wild_location_supply: g.wild_location_supply,
        wild_industry_supply: g.wild_industry_supply,
    }
}

/// Kebab-case industry name (as serialized, and as the TS oracle spells it).
pub(crate) fn industry_name(ind: Industry) -> &'static str {
    match ind {
        Industry::Cotton => "cotton",
        Industry::Manufacturer => "manufacturer",
        Industry::Pottery => "pottery",
        Industry::Coal => "coal",
        Industry::Iron => "iron",
        Industry::Brewery => "brewery",
    }
}

/// `<locA>-<locB>`: the link's two printed endpoints, sorted (a farm `extra` is left out).
pub(crate) fn link_name(link: u8) -> String {
    let l = &LINKS[link as usize];
    let mut ends = [LOCATIONS[l.a as usize].id, LOCATIONS[l.b as usize].id];
    ends.sort_unstable();
    format!("{}-{}", ends[0], ends[1])
}

/// `<loc>:<industry>` of the tile in `slot` (the slot's location and the tile's industry).
fn tile_name(g: &GameState, slot: u8) -> String {
    let loc = LOCATIONS[SLOTS[slot as usize].loc as usize].id;
    let ind = g.slots[slot as usize]
        .as_ref()
        .map_or("empty", |t| industry_name(t.industry));
    format!("{loc}:{ind}")
}

/// The coarse, engine-independent key of the legal move `m` in `g` (see [`legal_summary`]).
pub fn summary_key(g: &GameState, m: &Move) -> String {
    let era_link = |link: u8| match g.era {
        Era::Canal => format!("canal:{}", link_name(link)),
        Era::Rail => format!("rail:{}", link_name(link)),
    };
    match m {
        Move::Build { industry, slot, .. } => format!(
            "build:{}:{}",
            LOCATIONS[SLOTS[*slot as usize].loc as usize].id,
            industry_name(*industry)
        ),
        Move::Canal { link, .. } | Move::Rail { link, .. } | Move::SecondRail { link, .. } => {
            era_link(*link)
        }
        Move::Develop { industry, .. }
        | Move::DevelopSecond { industry, .. }
        | Move::BonusDevelop { industry } => format!("develop:{}", industry_name(*industry)),
        Move::Sell { slot, .. } | Move::SellAnother { slot, .. } => {
            format!("sell:{}", tile_name(g, *slot))
        }
        Move::Loan { .. } => "loan".into(),
        Move::Scout { .. } => "scout".into(),
        Move::Pass { .. } => "pass".into(),
        Move::ShortfallRemove { slot } => format!("shortfall:{}", tile_name(g, *slot)),
        Move::EndAction => "end".into(),
    }
}

/// Coarse keys of the legal moves, comparable across engines that model moves differently
/// (cards, resource sources and follow-up structure are dropped):
///
/// - `Phase::ChooseAction`: `build:<loc>:<industry>`, `canal:<locA>-<locB>`,
///   `rail:<locA>-<locB>` (the first link), `develop:<industry>` (the first tile),
///   `sell:<loc>:<industry>` (the first sale), `loan`, `scout`, `pass`.
/// - Follow-up phases use the same families for the further link, tile or sale (the
///   Gloucester develop is `develop:<industry>`), plus `end` for `EndAction`;
///   `Phase::Shortfall` gives `shortfall:<loc>:<industry>`.
///
/// Locations are kebab-case ids; the two ends of a link key are sorted. Empty once the game
/// is over.
pub fn legal_summary(g: &GameState) -> BTreeSet<String> {
    g.legal_moves().iter().map(|m| summary_key(g, m)).collect()
}
