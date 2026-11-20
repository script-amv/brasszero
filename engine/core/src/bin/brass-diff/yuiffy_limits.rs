//! Rules for stepping around known differences with the Yuiffy oracle (each one listed, with
//! its RQ, in `[yuiffy]` of `oracles/known_differences.toml`; see `known.rs`).
//!
//! Each rule is narrow: it applies only where the oracle provably cannot express what our RQs
//! decide, and states the oracle's behaviour in our own terms (no oracle code is copied).

use std::collections::BTreeSet;

use brass_engine::data::{
    Card, Era, Industry, LINKS, LOCATIONS, LocId, MERCHANT_SPACES, MerchantTile, PlayerId, SLOTS,
    income_level, tile,
};
use brass_engine::moves::{Move, Phase};
use brass_engine::network::{connected, has_board_presence};
use brass_engine::record::{ActionRecord, DiffSnapshot};
use brass_engine::state::{GameState, PlacedTile};
use serde_json::{Value, json};

use crate::known::Known;

/// Yuiffy's Build offers only the first connected Coal Mines in its build-location order,
/// not the closest (RQ-013, RQ-014). The adapter accepts a mine Yuiffy does not offer if
/// Yuiffy itself counts it as connected and stocked, and reports it as relaxed; this entry
/// is what lets the harness accept that report (`game.rs`, `unlisted_relaxation`).
const BUILD_COAL_ORDER: &str = "build-coal-order";

/// Yuiffy lets a player overbuild an exhausted opponent Coal Mine / Iron Works with a tile of
/// the same or a lower level; the new tile must be higher (RQ-024). The harness drops the
/// oracle-only build keys whose only target is such a tile.
const OVERBUILD_LEVEL: &str = "opponent-overbuild-level";

/// Yuiffy's shortfall resolution sets money to 0 and then keeps the whole value of the
/// removed tiles, without paying the rest of the debt from it (RQ-004: only the excess is
/// kept). The harness repairs the oracle's money when it differs by exactly that.
const SHORTFALL_MONEY: &str = "shortfall-excess-money";

/// Yuiffy resolves the shortfalls of the last Canal round (tile removal, VP loss) after the
/// era-end scoring; the rules settle them with the round's income, before scoring (RQ-021).
/// So Yuiffy scores the flipped tiles the shortfall removes, and takes the VP loss from the
/// scored total. The harness repairs the oracle's VP when it differs by exactly that.
const SHORTFALL_ORDER: &str = "canal-end-shortfall-order";

/// Yuiffy removes only its canal-only tiles at the end of the Canal Era, so Pottery I
/// (buildable in both eras) stays on the board; it is a level 1 tile and is removed
/// (RQ-018). The harness removes it from the oracle's board.
const POTTERY_I_KEPT: &str = "pottery-i-kept-at-canal-end";

/// With nothing on the board, Yuiffy lets an Industry card overbuild an exhausted opponent
/// Coal Mine / Iron Works anywhere; the exception opens free slots only (RQ-042, as for the
/// TS oracle). The harness drops those oracle-only build keys.
const NO_TILES_OVERBUILD: &str = "no-tiles-exception-overbuild";

/// Yuiffy takes iron from the chosen Iron Works in order, each until it is empty or the need
/// is met, so it cannot take part of the cubes of two works in one action; the rules let the
/// player take each cube from any works (RQ-047). The harness plays only iron plans that
/// leave at most one works partly used.
const IRON_SPLIT: &str = "iron-split-across-works";

/// Yuiffy's double rail only offers a second link on a rail-only line (no canal), and only
/// next to the first link or connected to one of the player's tiles; the rules allow any
/// line next to the player's network including the first link (RQ-020, RQ-035). The harness
/// plays only second links Yuiffy offers.
const SECOND_RAIL: &str = "double-rail-second-link";

/// Yuiffy asks for one Gloucester develop per Sell and repeats it for a second Gloucester
/// barrel; the player chooses each (RQ-009). The harness plays the same industry again.
const SECOND_BONUS: &str = "gloucester-second-bonus";

/// Yuiffy pools the barrels of every connected merchant tile that buys the industry for one
/// sale; a sale may use only the barrel of the merchant tile it sells to (RQ-016). The
/// harness drops the oracle-only sell keys that only such pooling makes possible.
const MERCHANT_BEER_POOL: &str = "sell-merchant-beer-pool";

/// Harness rules implemented here, by `known_differences.toml` key.
pub const IMPLEMENTED: &[&str] = &[
    BUILD_COAL_ORDER,
    OVERBUILD_LEVEL,
    SHORTFALL_MONEY,
    SHORTFALL_ORDER,
    POTTERY_I_KEPT,
    NO_TILES_OVERBUILD,
    IRON_SPLIT,
    SECOND_RAIL,
    SECOND_BONUS,
    MERCHANT_BEER_POOL,
];

/// Differences handled structurally: by the adapter (the shared discard pile) or by the
/// recorded vendor patch (`oracles/yuiffy-adapter/patches/`).
pub const STRUCTURAL: &[&str] = &[
    "shared-discard-pile",
    "manufacturer-ii-income",
    "manufacturer-viii-count",
    "rail-only-tiles",
    "rail-brewery-barrels",
];

/// No Yuiffy difference ends the comparable game: the ones that cannot be stepped around
/// are repaired (`resync`).
pub fn known_divergence(_known: &mut Known, _ours: &DiffSnapshot, _theirs: &DiffSnapshot) -> bool {
    false
}

/// The round end in an action record, measured on our engine: per player, what they owed
/// after paying income from cash (`owed`), the value of the tiles they removed (`raised`)
/// and their VP before the round end (`vp_before`); plus the flipped tiles removed (owner,
/// tile, location) and the links on the board when they were removed.
struct RoundEnd {
    owed: [u16; 4],
    raised: [u16; 4],
    vp_before: [u16; 4],
    removed_flipped: Vec<(PlayerId, PlacedTile, LocId)>,
    links: Vec<Option<PlayerId>>,
}

/// Replays `rec` from `before` to measure its round end (`None` if it ends no round).
fn round_end(before: &GameState, rec: &ActionRecord) -> Option<RoundEnd> {
    let (action, removes): (Vec<&Move>, Vec<&Move>) = rec
        .moves
        .iter()
        .partition(|m| !matches!(m, Move::ShortfallRemove { .. }));
    let mut after = before.clone();
    for m in &action {
        after.apply(m).ok()?;
    }
    let ended = after.round != before.round
        || after.era != before.era
        || after.game_over
        || matches!(after.phase, Phase::Shortfall { .. });
    if !ended {
        return None;
    }
    // The state after the action but before its round end: with one more action left and a
    // spare card in hand (an empty hand also ends the turn), the turn and so the round do
    // not end. The spare card plays no part in the recorded moves.
    let mut pre = before.clone();
    pre.actions_left += 1;
    pre.players[rec.player as usize]
        .hand
        .push(Card::Location(0));
    for m in &action {
        pre.apply(m).ok()?;
    }
    let owed = std::array::from_fn(|p| {
        let level = income_level(pre.players[p].income_space);
        (level.min(0).unsigned_abs() as u16).saturating_sub(pre.players[p].money)
    });
    let mut raised = [0u16; 4];
    let mut removed_flipped = Vec::new();
    let links = after.links.clone();
    for m in removes {
        if let Move::ShortfallRemove { slot } = m {
            let t = after.slots[*slot as usize].clone()?;
            raised[t.owner as usize] += tile(t.industry, t.level).cost_money as u16 / 2;
            if t.flipped {
                removed_flipped.push((t.owner, t, SLOTS[*slot as usize].loc));
            }
        }
        after.apply(m).ok()?;
    }
    Some(RoundEnd {
        owed,
        raised,
        vp_before: std::array::from_fn(|p| pre.players[p].vp),
        removed_flipped,
        links,
    })
}

/// Repairs the oracle after an action whose snapshots differ by exactly these known
/// differences (and nothing else), each predicted from our own state:
///
/// - `shortfall-excess-money`: a player who removed tiles has the smaller of their value and
///   the debt still owed after paying cash more money in the oracle;
/// - `canal-end-shortfall-order`: at the end of the Canal Era, the oracle's VP of each player
///   is what scoring before the shortfall gives: the tiles the shortfall removed score
///   (their VP for the owner, their link icons for the owners of adjacent links), then the
///   VP loss is taken from the total (never below 0);
/// - `pottery-i-kept-at-canal-end`: just after the Canal Era, the oracle still has Pottery I
///   tiles where ours has none.
///
/// Returns the `overwrite` request that sets the oracle's money and VP to ours and removes
/// those tiles; `None` unless the predictions explain every difference.
pub fn resync(
    known: &mut Known,
    before: &GameState,
    rec: &ActionRecord,
    ours: &DiffSnapshot,
    theirs: &DiffSnapshot,
) -> Option<Value> {
    let r = round_end(before, rec)?;
    let canal_end = before.era == Era::Canal && ours.era == Era::Rail;
    let mut patched = theirs.clone();
    let mut fired: Vec<&str> = Vec::new();

    if known.active(SHORTFALL_MONEY) {
        let expected: [u16; 4] =
            std::array::from_fn(|p| ours.money[p] + r.owed[p].min(r.raised[p]));
        if expected != ours.money && theirs.money == expected {
            patched.money = ours.money;
            fired.push(SHORTFALL_MONEY);
        }
    }
    if canal_end && known.active(SHORTFALL_ORDER) {
        let loss: [u16; 4] = std::array::from_fn(|p| r.owed[p].saturating_sub(r.raised[p]));
        let mut extra = [0u16; 4];
        for (owner, t, loc) in &r.removed_flipped {
            let def = tile(t.industry, t.level);
            extra[*owner as usize] += def.vp as u16;
            for (l, link_owner) in r.links.iter().enumerate() {
                let link = &LINKS[l];
                let touches = link.a == *loc || link.b == *loc || link.extra == Some(*loc);
                if let (Some(o), true) = (link_owner, touches) {
                    extra[*o as usize] += def.link_vp as u16;
                }
            }
        }
        let expected: [u16; 4] = std::array::from_fn(|p| {
            let after_loss = r.vp_before[p].saturating_sub(loss[p]);
            let scored = ours.vp[p].saturating_sub(after_loss);
            (r.vp_before[p] + scored + extra[p]).saturating_sub(loss[p])
        });
        if expected != ours.vp && theirs.vp == expected {
            patched.vp = ours.vp;
            fired.push(SHORTFALL_ORDER);
        }
    }
    let mut remove: Vec<usize> = Vec::new();
    if canal_end && known.active(POTTERY_I_KEPT) {
        for (s, (a, b)) in ours.slots.iter().zip(&theirs.slots).enumerate() {
            let pottery_i = b
                .as_ref()
                .is_some_and(|t| t.industry == Industry::Pottery && t.level == 1);
            if a.is_none() && pottery_i {
                patched.slots[s] = None;
                remove.push(s);
            }
        }
        if !remove.is_empty() {
            fired.push(POTTERY_I_KEPT);
        }
    }
    if fired.is_empty() || patched != *ours {
        return None;
    }
    for rule in fired {
        known.fire(rule);
    }
    Some(json!({
        "cmd": "overwrite",
        "money": ours.money,
        "vp": ours.vp,
        "remove_slots": remove,
    }))
}

/// A candidate rule's test: whether a move stays in play.
type Keep<'a> = dyn Fn(&Move) -> bool + 'a;

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
    let first_bonus = so_far.iter().find_map(|m| match m {
        Move::BonusDevelop { industry } => Some(*industry),
        _ => None,
    });
    let rules: [(&str, &Keep); 3] = [
        (IRON_SPLIT, &|m| iron_expressible(g, so_far, m)),
        (SECOND_RAIL, &|m| second_rail_expressible(g, so_far, m)),
        (SECOND_BONUS, &|m| match (m, first_bonus) {
            // Only a developable industry (its lowest tile on the mat has no lightbulb) other
            // than the first is ever dropped.
            (Move::BonusDevelop { industry }, Some(first)) => {
                let developable = g.players[g.current_player().expect("live game") as usize]
                    .lowest_level(*industry)
                    .is_some_and(|l| tile(*industry, l).developable);
                *industry == first || !developable
            }
            _ => true,
        }),
    ];
    for (rule, keep) in rules {
        if !known.active(rule) {
            continue;
        }
        let kept: Vec<Move> = out.iter().filter(|m| keep(m)).cloned().collect();
        if kept.len() < out.len() && !kept.is_empty() {
            known.fire(rule);
            out = kept;
        }
    }
    out
}

/// `iron-split-across-works`: the iron the move takes, added to the iron already taken in
/// this action (a first Develop), leaves at most one Iron Works partly used, measured
/// against the cubes it held when the action began.
fn iron_expressible(g: &GameState, so_far: &[Move], m: &Move) -> bool {
    let plan = match m {
        Move::Build { iron, .. }
        | Move::Develop { iron, .. }
        | Move::DevelopSecond { iron, .. } => iron,
        _ => return true,
    };
    let earlier = so_far.iter().filter_map(|m| match m {
        Move::Develop { iron, .. } => Some(iron),
        _ => None,
    });
    let mut taken: Vec<(usize, u8)> = Vec::new();
    for p in earlier.clone().chain([plan]) {
        for &(slot, n) in &p.from_tiles {
            match taken.iter_mut().find(|(s, _)| *s == slot as usize) {
                Some((_, t)) => *t += n,
                None => taken.push((slot as usize, n)),
            }
        }
    }
    let taken_before = |slot: usize| -> u8 {
        earlier
            .clone()
            .flat_map(|p| p.from_tiles.iter())
            .filter(|(s, _)| *s as usize == slot)
            .map(|(_, n)| n)
            .sum()
    };
    let held = |slot: usize| g.slots[slot].as_ref().map_or(0, |t| t.resources) + taken_before(slot);
    let partial = taken.iter().filter(|&&(slot, n)| n < held(slot)).count();
    // Only a well-formed plan is ever dropped (iron taken from Iron Works that held it, no
    // market iron): anything else stays in play for the oracle to judge.
    let market = earlier.clone().chain([plan]).any(|p| p.from_market > 0);
    let well_formed = !market
        && taken.iter().all(|&(slot, n)| {
            g.slots[slot]
                .as_ref()
                .is_some_and(|t| t.industry == Industry::Iron)
                && n <= held(slot)
        });
    !well_formed || partial <= 1
}

/// `double-rail-second-link`: whether the second rail link is one Yuiffy offers: on a
/// rail-only line, with at least two rail-only lines free before the first link was placed
/// (Yuiffy's double-rail precondition), and next to the first link or connected (without the
/// first link) to a location holding one of the player's tiles.
fn second_rail_expressible(g: &GameState, so_far: &[Move], m: &Move) -> bool {
    let Move::SecondRail { link, .. } = m else {
        return true;
    };
    let Some(first) = so_far.iter().find_map(|m| match m {
        Move::Rail { link, .. } => Some(*link as usize),
        _ => None,
    }) else {
        return true;
    };
    let rail_only = |l: usize| !LINKS[l].canal;
    let free_rail_only_before = (0..LINKS.len())
        .filter(|&l| rail_only(l) && (g.links[l].is_none() || l == first))
        .count();
    let locs = |l: usize| {
        let d = &LINKS[l];
        [Some(d.a), Some(d.b), d.extra].into_iter().flatten()
    };
    let second = *link as usize;
    let next_to_first = locs(second).any(|a| locs(first).any(|b| a == b));
    let p = g.current_player().expect("live game");
    let mut without_first = g.clone();
    without_first.links[first] = None;
    let own_locs: Vec<LocId> = (0..SLOTS.len())
        .filter(|&s| g.slots[s].as_ref().is_some_and(|t| t.owner == p))
        .map(|s| SLOTS[s].loc)
        .collect();
    let reaches_own =
        locs(second).any(|a| own_locs.iter().any(|&o| connected(&without_first, a, o)));
    let expressible =
        rail_only(second) && free_rail_only_before >= 2 && (next_to_first || reaches_own);
    // Only a link next to the player's network (own tiles' locations and own links' ends,
    // the first link included) is ever dropped: anything else stays in play.
    let network: Vec<LocId> = own_locs
        .iter()
        .copied()
        .chain(
            (0..LINKS.len())
                .filter(|&l| g.links[l] == Some(p))
                .flat_map(locs),
        )
        .collect();
    let next_to_network = locs(second).any(|a| network.contains(&a));
    !next_to_network || expressible
}

/// Drop legal-summary keys that differ for a known reason.
pub fn filter_keys(
    known: &mut Known,
    g: &GameState,
    ours: &mut BTreeSet<String>,
    theirs: &mut BTreeSet<String>,
) {
    let p = g.current_player().expect("live game");
    if known.active(MERCHANT_BEER_POOL) {
        let extra: Vec<String> = theirs
            .difference(ours)
            .filter(|k| only_pooled_merchant_beer(g, p, k))
            .cloned()
            .collect();
        if !extra.is_empty() {
            known.fire(MERCHANT_BEER_POOL);
            for k in extra {
                theirs.remove(&k);
            }
        }
    }
    for rule in [OVERBUILD_LEVEL, NO_TILES_OVERBUILD] {
        if !known.active(rule) {
            continue;
        }
        let extra: Vec<String> = theirs
            .difference(ours)
            .filter(|k| match only_opponent_overbuild(g, p, k) {
                Some(Target { higher_level }) if rule == OVERBUILD_LEVEL => !higher_level,
                Some(Target { higher_level }) => higher_level && !has_board_presence(g, p),
                None => false,
            })
            .cloned()
            .collect();
        if !extra.is_empty() {
            known.fire(rule);
            for k in extra {
                theirs.remove(&k);
            }
        }
    }
}

/// Whether `key` is `sell:<loc>:<industry>` and every unflipped tile of `p`'s it names can
/// be sold only by pooling merchant barrels: the brewery beer `p` may use plus one merchant
/// barrel is too little, but with the barrels of every connected merchant tile that buys
/// the industry it is enough.
fn only_pooled_merchant_beer(g: &GameState, p: PlayerId, key: &str) -> bool {
    let Some(rest) = key.strip_prefix("sell:") else {
        return false;
    };
    let Some((loc_id, ind_name)) = rest.split_once(':') else {
        return false;
    };
    let Some(ind) = [
        (Industry::Cotton, "cotton"),
        (Industry::Manufacturer, "manufacturer"),
        (Industry::Pottery, "pottery"),
    ]
    .into_iter()
    .find_map(|(i, n)| (n == ind_name).then_some(i)) else {
        return false;
    };
    let Some(loc) = (0..LOCATIONS.len() as LocId).find(|&l| LOCATIONS[l as usize].id == loc_id)
    else {
        return false;
    };
    let merchant_locs: Vec<LocId> = MERCHANT_SPACES
        .iter()
        .flat_map(|m| std::iter::repeat_n(m.loc, m.tile_slots as usize))
        .collect();
    let buys = |t: MerchantTile| {
        matches!(
            (t, ind),
            (MerchantTile::Any, _)
                | (MerchantTile::Cotton, Industry::Cotton)
                | (MerchantTile::Manufacturer, Industry::Manufacturer)
                | (MerchantTile::Pottery, Industry::Pottery)
        )
    };
    let merchants: Vec<usize> = (0..merchant_locs.len())
        .filter(|&m| buys(g.merchant_tiles[m]) && connected(g, loc, merchant_locs[m]))
        .collect();
    let barrels = merchants.iter().filter(|&&m| g.merchant_beer[m]).count() as u16;
    let brewery: u16 = (0..SLOTS.len())
        .filter_map(|s| g.slots[s].as_ref().map(|t| (s, t)))
        .filter(|(s, t)| {
            t.industry == Industry::Brewery
                && !t.flipped
                && (t.owner == p || connected(g, loc, SLOTS[*s].loc))
        })
        .map(|(_, t)| t.resources as u16)
        .sum();
    let tiles: Vec<&PlacedTile> = (0..SLOTS.len())
        .filter(|&s| SLOTS[s].loc == loc)
        .filter_map(|s| g.slots[s].as_ref())
        .filter(|t| t.owner == p && t.industry == ind && !t.flipped)
        .collect();
    !merchants.is_empty()
        && !tiles.is_empty()
        && tiles.iter().all(|t| {
            let need = tile(t.industry, t.level).beer_to_sell as u16;
            brewery + barrels.min(1) < need && brewery + barrels >= need
        })
}

/// The only target Yuiffy has for an oracle-only build key: an exhausted opponent tile.
struct Target {
    /// Some such opponent tile has a lower level than the player's next tile (a legal
    /// overbuild level); otherwise all are the same level or higher.
    higher_level: bool,
}

/// If `key` is `build:<loc>:coal|iron` and the only slots at `loc` Yuiffy could build it on
/// are exhausted opponent tiles of that industry (coal / iron gone from the board and the
/// market, as Yuiffy requires; no free slot for it and no own lower tile of it there), what
/// they are.
fn only_opponent_overbuild(g: &GameState, p: PlayerId, key: &str) -> Option<Target> {
    for (ind, name) in [(Industry::Coal, "coal"), (Industry::Iron, "iron")] {
        let Some(lvl) = g.players[p as usize].lowest_level(ind) else {
            continue;
        };
        for loc in 0..LOCATIONS.len() as LocId {
            if key != format!("build:{}:{name}", LOCATIONS[loc as usize].id) {
                continue;
            }
            let market = if ind == Industry::Coal {
                g.coal_market
            } else {
                g.iron_market
            };
            let depleted = market == 0
                && g.slots
                    .iter()
                    .flatten()
                    .all(|t| t.industry != ind || t.resources == 0);
            let here: Vec<Option<&PlacedTile>> = (0..SLOTS.len())
                .filter(|&s| SLOTS[s].loc == loc && SLOTS[s].industries.contains(&ind))
                .map(|s| g.slots[s].as_ref())
                .collect();
            let free = here.iter().any(|t| t.is_none());
            // An own lower tile of the industry would be a legal target of ours.
            let own_lower = here
                .iter()
                .flatten()
                .any(|t| t.owner == p && t.industry == ind && t.level < lvl);
            let opponent: Vec<&PlacedTile> = here
                .iter()
                .flatten()
                .filter(|t| t.owner != p && t.industry == ind)
                .copied()
                .collect();
            if !depleted || free || own_lower || opponent.is_empty() {
                return None;
            }
            return Some(Target {
                higher_level: opponent.iter().any(|t| t.level < lvl),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use brass_engine::record::diff_snapshot;
    use brass_engine::resources::ResourcePlan;
    use brass_engine::state::new_game;

    fn works(g: &mut GameState, slot: usize, cubes: u8) -> u8 {
        g.slots[slot] = Some(PlacedTile {
            owner: 1,
            industry: Industry::Iron,
            level: 1,
            flipped: false,
            resources: cubes,
        });
        slot as u8
    }

    fn develop(tiles: &[(u8, u8)], market: u8) -> Move {
        Move::Develop {
            card: Card::Location(0),
            industry: Industry::Cotton,
            iron: ResourcePlan {
                from_tiles: tiles.to_vec(),
                from_market: market,
            },
        }
    }

    fn develop_second(tiles: &[(u8, u8)]) -> Move {
        Move::DevelopSecond {
            industry: Industry::Cotton,
            iron: ResourcePlan {
                from_tiles: tiles.to_vec(),
                from_market: 0,
            },
        }
    }

    /// RQ-047: Yuiffy drains works in order, so at most one works may be left partly used,
    /// counted across both halves of a double Develop.
    #[test]
    fn iron_plans_with_two_partly_used_works_are_dropped() {
        let mut g = new_game(0);
        let (a, b) = (works(&mut g, 0, 4), works(&mut g, 1, 2));
        assert!(!iron_expressible(&g, &[], &develop(&[(a, 1), (b, 1)], 0)));
        assert!(iron_expressible(&g, &[], &develop(&[(a, 1), (b, 2)], 0)));
        assert!(iron_expressible(&g, &[], &develop(&[(a, 2)], 0)));
        // A plan using the market is left for the oracle to judge.
        assert!(iron_expressible(&g, &[], &develop(&[(a, 1), (b, 1)], 1)));
        // Double Develop: the first half took 1 cube from `a` (the state shows 3 left).
        g.slots[0].as_mut().unwrap().resources = 3;
        let first = [develop(&[(a, 1)], 0)];
        assert!(!iron_expressible(&g, &first, &develop_second(&[(b, 1)])));
        assert!(iron_expressible(&g, &first, &develop_second(&[(b, 2)])));
        assert!(iron_expressible(&g, &first, &develop_second(&[(a, 1)])));
    }

    /// The repair is never sent for an action that ends no round, whatever the difference.
    #[test]
    fn resync_needs_a_round_end() {
        let g = new_game(0);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../oracles/known_differences.toml");
        let mut known = Known::load(&path, "yuiffy", IMPLEMENTED, STRUCTURAL).unwrap();
        let p = g.current_player().unwrap();
        let card = g.players[p as usize].hand[0];
        let rec = ActionRecord {
            player: p,
            moves: vec![Move::Loan { card }],
        };
        let mut after = g.clone();
        after.apply(&rec.moves[0]).unwrap();
        let ours = diff_snapshot(&after);
        let mut theirs = ours.clone();
        theirs.money[p as usize] += 5;
        assert_eq!(resync(&mut known, &g, &rec, &ours, &theirs), None);
    }
}
