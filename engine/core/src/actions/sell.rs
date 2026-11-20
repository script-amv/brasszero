//! Sell (rules-reference section 9.4): flip unflipped Cotton, Manufacturer and Pottery tiles
//! connected to a merchant that buys them, consuming the beer shown on the tile.
//!
//! The first sale is `Move::Sell` (discards the card, enters `Phase::SellMore`); each further
//! sale is `Move::SellAnother`, and `Move::EndAction` stops. Taking the barrel beside the
//! merchant tile (optional, at most one per sale) collects that merchant's bonus. Gloucester's
//! free develop pauses in `Phase::BonusDevelop`, then returns to `Phase::SellMore`.
//! Sales without merchant beer name the lowest accepting merchant slot (RQ-044).
//! Rules questions: RQ-008 (Oxford spaces), RQ-009 (Gloucester), RQ-012 (0-beer tiles),
//! RQ-040 (income cap), RQ-041 (lightbulb Pottery).

use crate::actions::develop::{developable, remove_lowest};
use crate::actions::{actor, discard};
use crate::data::*;
use crate::moves::{Move, Phase};
use crate::network::Adjacency;
use crate::resources::{BeerSource, beer_plans_in, consume_beer, flip};
use crate::state::GameState;

// `merchant_space` maps flattened slots onto `MERCHANT_SPACES`: the spaces' tile slots must
// be exactly the 9 merchant tiles of a 4-player game.
const _: () = {
    let mut total = 0;
    let mut i = 0;
    while i < MERCHANT_SPACES.len() {
        total += MERCHANT_SPACES[i].tile_slots as usize;
        i += 1;
    }
    assert!(total == MERCHANT_TILES_4P.len());
};

/// The location and bonus of the merchant space holding flattened merchant tile slot `m`
/// (index into `GameState::merchant_tiles`).
fn merchant_space(m: u8) -> &'static MerchantSpaceDef {
    let mut rest = m;
    for sp in MERCHANT_SPACES {
        if rest < sp.tile_slots {
            return sp;
        }
        rest -= sp.tile_slots;
    }
    panic!("no merchant tile slot {m}");
}

/// Whether a merchant tile buys `ind`: `Any` buys all three, a blank buys nothing.
fn accepts(tile: MerchantTile, ind: Industry) -> bool {
    matches!(
        (tile, ind),
        (MerchantTile::Any, _)
            | (MerchantTile::Cotton, Industry::Cotton)
            | (MerchantTile::Manufacturer, Industry::Manufacturer)
            | (MerchantTile::Pottery, Industry::Pottery)
    )
}

/// Every `(slot, merchant, beer)` sale the actor may make now, in slot, merchant, beer order.
/// With `only = Some(slot)`, just the sales of that slot.
fn sales(g: &GameState, only: Option<SlotId>) -> Vec<(SlotId, u8, Vec<BeerSource>)> {
    let p = actor(g);
    let adj = Adjacency::of(g);
    let mut out = Vec::new();
    for (i, placed) in g.slots.iter().enumerate() {
        if only.is_some_and(|s| s as usize != i) {
            continue;
        }
        let Some(t) = placed.as_ref().filter(|t| {
            t.owner == p
                && !t.flipped
                && matches!(
                    t.industry,
                    Industry::Cotton | Industry::Manufacturer | Industry::Pottery
                )
        }) else {
            continue;
        };
        let loc = SLOTS[i].loc;
        let need = tile(t.industry, t.level).beer_to_sell;
        let dist = adj.distances(loc);
        let mut seen_plain = false;
        for (m, &mt) in g.merchant_tiles.iter().enumerate() {
            let m = m as u8;
            // The connection to the sold-to merchant is checked here, not by `beer_plans`.
            if !accepts(mt, t.industry) || dist[merchant_space(m).loc as usize] == u8::MAX {
                continue;
            }
            // Merchant beer fills a required barrel only, so a 0-beer tile never takes it.
            for beer in beer_plans_in(g, &adj, p, loc, need, Some(m)) {
                // Plans without merchant beer are identical for every merchant: offer them
                // once per tile, at the first (lowest) accepting slot (RQ-044).
                let uses_merchant = beer.iter().any(|b| matches!(b, BeerSource::Merchant(_)));
                if uses_merchant || !seen_plain {
                    out.push((i as SlotId, m, beer));
                }
            }
            seen_plain = true;
        }
    }
    out
}

/// `Phase::ChooseAction`: card × sale. `cards` is `distinct_cards` of the actor.
pub(crate) fn legal(g: &GameState, cards: &[Card]) -> Vec<Move> {
    moves(g, cards, None)
}

/// The Sell moves of `legal` for the tile in `slot` (for `apply`'s check).
pub(crate) fn legal_for(g: &GameState, cards: &[Card], slot: SlotId) -> Vec<Move> {
    moves(g, cards, Some(slot))
}

fn moves(g: &GameState, cards: &[Card], only: Option<SlotId>) -> Vec<Move> {
    let sales = sales(g, only);
    if sales.is_empty() {
        return Vec::new();
    }
    cards
        .iter()
        .flat_map(|&card| {
            sales.iter().map(move |(slot, merchant, beer)| Move::Sell {
                card,
                slot: *slot,
                merchant: *merchant,
                beer: beer.clone(),
            })
        })
        .collect()
}

/// `Phase::SellMore`: a further sale (`EndAction` is added by `legal.rs`).
pub(crate) fn legal_another(g: &GameState) -> Vec<Move> {
    sales(g, None)
        .into_iter()
        .map(|(slot, merchant, beer)| Move::SellAnother {
            slot,
            merchant,
            beer,
        })
        .collect()
}

/// `Phase::BonusDevelop`: the industry whose lowest tile is removed, free of iron.
pub(crate) fn legal_bonus_develop(g: &GameState) -> Vec<Move> {
    developable(g)
        .into_iter()
        .map(|industry| Move::BonusDevelop { industry })
        .collect()
}

/// Consume the beer, flip the tile, then collect the bonus if merchant beer was taken.
fn sell_one(g: &mut GameState, slot: SlotId, beer: &[BeerSource]) {
    let p = actor(g);
    consume_beer(g, beer);
    flip(g, slot);
    g.phase = Phase::SellMore;
    let Some(&BeerSource::Merchant(m)) = beer.iter().find(|b| matches!(b, BeerSource::Merchant(_)))
    else {
        return;
    };
    let ps = &mut g.players[p as usize];
    match merchant_space(m).bonus {
        // Skipped automatically when nothing is developable (RQ-009).
        MerchantBonus::Develop => {
            if !developable(g).is_empty() {
                g.phase = Phase::BonusDevelop;
            }
        }
        // Spaces, not levels; capped like every income gain (RQ-008, RQ-040).
        MerchantBonus::Income(n) => {
            ps.income_space = ps.income_space.saturating_add(n).min(MAX_INCOME_SPACE);
        }
        MerchantBonus::Vp(n) => ps.vp += n as u16,
        MerchantBonus::Money(n) => ps.money += n as u16,
    }
}

pub(crate) fn apply(g: &mut GameState, card: Card, slot: SlotId, beer: &[BeerSource]) {
    discard(g, actor(g), card);
    sell_one(g, slot, beer);
}

pub(crate) fn apply_another(g: &mut GameState, slot: SlotId, beer: &[BeerSource]) {
    sell_one(g, slot, beer);
}

pub(crate) fn apply_bonus_develop(g: &mut GameState, industry: Industry) {
    remove_lowest(g, actor(g), industry);
    g.phase = Phase::SellMore;
}
