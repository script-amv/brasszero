//! Rules for stepping around known differences with the TS oracle (each one listed, with
//! its RQ, in `[ts]` of `oracles/known_differences.toml`; see `known.rs`).
//!
//! Each rule restricts our random choice to the moves the oracle can express, re-stating the
//! oracle's documented canonical choice in our own terms (no oracle code is copied).

use std::collections::BTreeSet;

use brass_engine::data::{Era, Industry, LINKS, LOCATIONS, LocId, PlayerId, SLOTS, tile};
use brass_engine::moves::{Move, Phase};
use brass_engine::network::has_board_presence;
use brass_engine::record::DiffSnapshot;
use brass_engine::state::{GameState, PlacedTile};

use crate::known::Known;

/// The oracle resolves negative income itself, removing tiles in a fixed order; our
/// players choose (RQ-004).
const SHORTFALL: &str = "shortfall-choice";
/// The oracle's Build picks the slot (opponent overbuild, else a free slot, else an own
/// overbuild) and only lets a player choose within that class (RQ-046).
const BUILD_SLOT: &str = "build-slot-choice";

/// The oracle also removes £0 tiles (Pottery II/IV) to cover negative income, gaining
/// nothing; ours never requires them (RQ-004). The game then cannot continue comparably.
const SHORTFALL_ZERO: &str = "shortfall-zero-value-tile";

/// The oracle's Sell takes one explicit Gloucester develop choice per action; a second
/// Gloucester bonus in the same action removes the first developable industry in industry
/// order (RQ-009: the player chooses).
const SECOND_BONUS: &str = "gloucester-second-bonus";

/// With nothing on the board, the oracle lets an Industry card overbuild an exhausted
/// opponent Coal Mine / Iron Works anywhere; ours opens only free slots (RQ-042).
const NO_TILES_OVERBUILD: &str = "no-tiles-exception-overbuild";

/// Differences handled structurally by the adapter.
pub const STRUCTURAL: &[&str] = &["shared-discard-pile"];

/// Harness rules implemented here, by `known_differences.toml` key.
pub const IMPLEMENTED: &[&str] = &[
    SHORTFALL,
    BUILD_SLOT,
    SHORTFALL_ZERO,
    SECOND_BONUS,
    NO_TILES_OVERBUILD,
];

/// Whether the snapshots after an action differ only by a known divergence that ends the
/// comparable part of the game (the harness then stops the game without a mismatch).
///
/// `shortfall-zero-value-tile`: some slots hold a £0 tile in ours and nothing in the oracle,
/// every other slot agrees, and nothing else differs except VP, which differs only when the
/// round end also ended the era (the removed tile and its link icons were not scored). The VP
/// gap is bounded: no player is behind in ours, and the total gap is at most the removed
/// tiles' VP plus their link icons on every link that could touch their location.
pub fn known_divergence(known: &mut Known, ours: &DiffSnapshot, theirs: &DiffSnapshot) -> bool {
    if !known.active(SHORTFALL_ZERO) {
        return false;
    }
    // Just after the Canal-to-Rail transition, or at the game end.
    let era_ended = ours.game_over
        || (ours.era == Era::Rail && (ours.round, ours.turn_idx, ours.actions_left) == (1, 0, 2));
    let rest_equal = DiffSnapshot {
        slots: theirs.slots.clone(),
        vp: theirs.vp,
        ..ours.clone()
    } == *theirs;
    let zero_tile = |t: &Option<PlacedTile>| {
        t.as_ref()
            .is_some_and(|t| tile(t.industry, t.level).cost_money == 0)
    };
    let pairs = || ours.slots.iter().zip(&theirs.slots);
    let removed_zero = pairs().any(|(a, b)| a != b);
    let only_zero = pairs().all(|(a, b)| a == b || (b.is_none() && zero_tile(a)));
    // Upper bound on what the removed £0 tiles scored in ours: their own VP plus their link
    // icons for each link ending at (or, for the farm link, passing through) the location.
    // Links are gone from the snapshot at era end, so count every incident link.
    let bound: u32 = (0..SLOTS.len())
        .filter(|&s| ours.slots[s] != theirs.slots[s])
        .filter_map(|s| {
            ours.slots[s]
                .as_ref()
                .map(|t| (SLOTS[s].loc, tile(t.industry, t.level)))
        })
        .map(|(loc, def)| {
            let incident = LINKS
                .iter()
                .filter(|l| l.a == loc || l.b == loc || l.extra == Some(loc))
                .count() as u32;
            def.vp as u32 + def.link_vp as u32 * incident
        })
        .sum();
    // The gap exists only when the era ended; it never favours the oracle.
    let gap_ok = (0..4).all(|p| {
        let gap = ours.vp[p].checked_sub(theirs.vp[p]);
        gap.is_some_and(|d| (era_ended || d == 0) && u32::from(d) <= bound)
    }) && (0..4)
        .map(|p| u32::from(ours.vp[p] - theirs.vp[p]))
        .sum::<u32>()
        <= bound;
    if rest_equal && removed_zero && only_zero && gap_ok {
        known.fire(SHORTFALL_ZERO);
        return true;
    }
    false
}

/// The moves of `legal` (legal in `g`, the action so far being `so_far`) that the oracle can
/// express. Never empty when `legal` is not: if a rule would remove every move, it is not
/// applied (and the oracle comparison reports what happens).
pub fn candidates(
    known: &mut Known,
    g: &GameState,
    so_far: &[Move],
    legal: Vec<Move>,
) -> Vec<Move> {
    let mut out = legal;
    if known.active(SECOND_BONUS)
        && g.phase == Phase::BonusDevelop
        && so_far
            .iter()
            .any(|m| matches!(m, Move::BonusDevelop { .. }))
        && out.len() > 1
    {
        // `legal_moves` lists the Gloucester develop in industry order.
        known.fire(SECOND_BONUS);
        out.truncate(1);
    }
    for (rule, keep) in [
        (SHORTFALL, shortfall_ok as fn(&GameState, &Move) -> bool),
        (BUILD_SLOT, build_slot_ok),
    ] {
        if !known.active(rule) {
            continue;
        }
        let kept: Vec<Move> = out.iter().filter(|m| keep(g, m)).cloned().collect();
        if kept.len() < out.len() && !kept.is_empty() {
            known.fire(rule);
            out = kept;
        }
    }
    out
}

/// Drop legal-summary keys that differ for a known reason.
pub fn filter_keys(
    known: &mut Known,
    g: &GameState,
    ours: &mut BTreeSet<String>,
    theirs: &mut BTreeSet<String>,
) {
    if !known.active(NO_TILES_OVERBUILD) {
        return;
    }
    let p = g.current_player().expect("live game");
    if has_board_presence(g, p) {
        return;
    }
    // Oracle-only builds of an exhausted opponent Coal Mine / Iron Works (outside any
    // network: the player has none).
    let extra: Vec<String> = theirs
        .difference(ours)
        .filter(|k| {
            [Industry::Coal, Industry::Iron].into_iter().any(|ind| {
                let Some(lvl) = g.players[p as usize].lowest_level(ind) else {
                    return false;
                };
                (0..LOCATIONS.len() as LocId).any(|loc| {
                    **k == format!(
                        "build:{}:{}",
                        LOCATIONS[loc as usize].id,
                        industry_name(ind)
                    ) && oracle_slot_class(g, p, loc, ind, lvl) == Some(SlotClass::Opponent)
                })
            })
        })
        .cloned()
        .collect();
    if !extra.is_empty() {
        known.fire(NO_TILES_OVERBUILD);
        for k in extra {
            theirs.remove(&k);
        }
    }
}

fn industry_name(ind: Industry) -> &'static str {
    match ind {
        Industry::Coal => "coal",
        Industry::Iron => "iron",
        _ => unreachable!("only coal and iron are overbuilt by opponents"),
    }
}

/// `Phase::Shortfall`: only the tile the oracle removes next: unflipped before flipped, then
/// lowest printed VP, then location id (alphabetical), then slot order within the location.
/// £0 tiles are skipped: ours never offers them (see `SHORTFALL_ZERO`).
fn shortfall_ok(g: &GameState, m: &Move) -> bool {
    let (Phase::Shortfall { player, .. }, Move::ShortfallRemove { slot }) = (g.phase, m) else {
        return true;
    };
    let key = |s: usize| {
        let t = g.slots[s].as_ref().expect("own tile");
        (
            t.flipped,
            tile(t.industry, t.level).vp,
            LOCATIONS[SLOTS[s].loc as usize].id,
            s,
        )
    };
    let next = (0..g.slots.len())
        .filter(|&s| {
            g.slots[s]
                .as_ref()
                .is_some_and(|t| t.owner == player && tile(t.industry, t.level).cost_money > 0)
        })
        .min_by_key(|&s| key(s));
    next == Some(*slot as usize)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SlotClass {
    Opponent,
    Free,
    Own,
}

/// The kind of slot the oracle builds `ind` (level `lvl`) on at `loc` for `p`.
fn oracle_slot_class(
    g: &GameState,
    p: PlayerId,
    loc: LocId,
    ind: Industry,
    lvl: u8,
) -> Option<SlotClass> {
    let here: Vec<usize> = (0..SLOTS.len())
        .filter(|&s| SLOTS[s].loc == loc && SLOTS[s].industries.contains(&ind))
        .collect();
    let own_at_loc = (0..SLOTS.len())
        .any(|s| SLOTS[s].loc == loc && g.slots[s].as_ref().is_some_and(|t| t.owner == p));
    let blocked = g.era == Era::Canal && own_at_loc;
    let lower = |s: usize, own: bool| {
        g.slots[s]
            .as_ref()
            .is_some_and(|t| (t.owner == p) == own && t.industry == ind && t.level < lvl)
    };
    let cubes_left = || {
        let market = if ind == Industry::Coal {
            g.coal_market
        } else {
            g.iron_market
        };
        market as u32
            + g.slots
                .iter()
                .flatten()
                .filter(|t| t.industry == ind)
                .map(|t| t.resources as u32)
                .sum::<u32>()
    };
    if !blocked
        && matches!(ind, Industry::Coal | Industry::Iron)
        && cubes_left() == 0
        && here.iter().any(|&s| lower(s, false))
    {
        return Some(SlotClass::Opponent);
    }
    if !blocked && here.iter().any(|&s| g.slots[s].is_none()) {
        return Some(SlotClass::Free);
    }
    here.iter()
        .any(|&s| lower(s, true))
        .then_some(SlotClass::Own)
}

/// Build: drop only the targets RQ-046 makes legal but the oracle cannot express:
/// - an own overbuild where the oracle would use a free slot or an opponent overbuild;
/// - a free slot where the oracle would overbuild an exhausted opponent tile;
/// - a free single-icon slot other than the first one (all are interchangeable).
///
/// Every other Build is kept, including any our engine should not offer (a two-icon slot
/// beside a free single-icon one, an opponent overbuild with cubes left, a second own tile
/// in a Canal Era location): the oracle then rejects it and the harness reports a mismatch.
fn build_slot_ok(g: &GameState, m: &Move) -> bool {
    let Move::Build { industry, slot, .. } = m else {
        return true;
    };
    let p = g.current_player().expect("live game");
    let lvl = g.players[p as usize]
        .lowest_level(*industry)
        .expect("legal build has a tile");
    let class = match &g.slots[*slot as usize] {
        None => SlotClass::Free,
        Some(t) if t.owner == p => SlotClass::Own,
        Some(_) => SlotClass::Opponent,
    };
    let loc = SLOTS[*slot as usize].loc;
    let oracle = oracle_slot_class(g, p, loc, *industry, lvl);
    match (class, oracle) {
        (SlotClass::Own, Some(SlotClass::Free | SlotClass::Opponent)) => false,
        (SlotClass::Free, Some(SlotClass::Opponent)) => false,
        (SlotClass::Free, Some(SlotClass::Free)) if SLOTS[*slot as usize].industries.len() == 1 => {
            let first_single = (0..SLOTS.len()).find(|&s| {
                SLOTS[s].loc == loc && SLOTS[s].industries == [*industry] && g.slots[s].is_none()
            });
            first_single == Some(*slot as usize)
        }
        _ => true,
    }
}
