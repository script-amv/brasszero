// SPDX-License-Identifier: Apache-2.0
//! The Sell action and merchant bonuses.
//! Rules: docs/rules-reference.md section 9.4 (Sell, merchant beer bonuses), section 3 (merchants
//! table), section 8.3 (beer); RULES-QUESTIONS.md RQ-008, RQ-009, RQ-012, RQ-040, RQ-041.

mod common;

use brass_engine::data::*;
use brass_engine::moves::{Move, Phase, RuleError};
use brass_engine::resources::BeerSource;
use brass_engine::state::GameState;
use common::*;

use Industry::{Brewery, Coal, Cotton, Manufacturer, Pottery};

const ANY: Card = Card::Location(loc::BIRMINGHAM);

// Flattened merchant tile slots (MERCHANT_SPACES order, 4 players).
const SHREWSBURY: u8 = 0;
const GLOUCESTER: u8 = 1;
const OXFORD: u8 = 3;
const WARRINGTON: u8 = 5;
const NOTTINGHAM: u8 = 7;

/// An empty game: every merchant blank, the acting player holds one card and has 2 actions.
fn setup() -> (GameState, PlayerId) {
    let mut g = empty_game();
    g.actions_left = 2;
    let p = g.current_player().expect("someone is to act");
    g.players[p as usize].hand = vec![ANY];
    g.merchant_tiles = vec![MerchantTile::Blank; 9];
    g.merchant_beer = vec![false; 9];
    (g, p)
}

/// Put `tile` (with its barrel) on merchant tile slot `m`.
fn merchant(g: &mut GameState, m: u8, tile: MerchantTile) {
    g.merchant_tiles[m as usize] = tile;
    g.merchant_beer[m as usize] = tile != MerchantTile::Blank;
}

fn sells(g: &GameState) -> Vec<Move> {
    g.legal_moves()
        .into_iter()
        .filter(|m| matches!(m, Move::Sell { .. } | Move::SellAnother { .. }))
        .collect()
}

fn sell(slot: SlotId, merchant: u8, beer: Vec<BeerSource>) -> Move {
    Move::Sell {
        card: ANY,
        slot,
        merchant,
        beer,
    }
}

fn another(slot: SlotId, merchant: u8, beer: Vec<BeerSource>) -> Move {
    Move::SellAnother {
        slot,
        merchant,
        beer,
    }
}

// 9.4 step 2: the tile must be connected to a merchant tile showing its industry; "Any" shows
// all three. Merchant locations are ordinary nodes, so a link to Oxford connects Birmingham.
#[test]
fn sell_requires_connection_to_accepting_merchant() {
    let (mut g, p) = setup();
    let cotton = place(&mut g, "birmingham", 0, p, Cotton, 1, 0);
    let brewery = place(&mut g, "burton-on-trent", 0, p, Brewery, 1, 1);
    merchant(&mut g, OXFORD, MerchantTile::Any);
    // Not connected: nothing to sell to.
    assert!(sells(&g).is_empty());
    link(&mut g, "birmingham", "oxford", p);
    // Connected: sell with brewery beer, or with the merchant's own barrel.
    assert_eq!(
        sells(&g),
        vec![
            sell(cotton, OXFORD, vec![BeerSource::Brewery(brewery)]),
            sell(cotton, OXFORD, vec![BeerSource::Merchant(OXFORD)]),
        ]
    );
    // A merchant that buys only manufactured goods or pottery does not take cotton.
    merchant(&mut g, OXFORD, MerchantTile::Manufacturer);
    assert!(sells(&g).is_empty());
    merchant(&mut g, OXFORD, MerchantTile::Pottery);
    assert!(sells(&g).is_empty());
    merchant(&mut g, OXFORD, MerchantTile::Cotton);
    assert_eq!(sells(&g).len(), 2);
    // A tile of another owner, or one already flipped, is not sold.
    g.slots[cotton as usize].as_mut().unwrap().owner = (p + 1) % 4;
    g.slots[brewery as usize].as_mut().unwrap().owner = (p + 1) % 4;
    assert!(sells(&g).is_empty());
    g.slots[cotton as usize].as_mut().unwrap().owner = p;
    g.slots[brewery as usize].as_mut().unwrap().owner = p;
    assert_eq!(sells(&g).len(), 2);
    g.slots[cotton as usize].as_mut().unwrap().flipped = true;
    assert!(sells(&g).is_empty());
}

// 9.4 / RQ-012-adjacent: a blank merchant tile buys nothing and holds no beer, though its
// location still joins the network.
#[test]
fn sell_blank_merchant_never_accepts() {
    let (mut g, p) = setup();
    place(&mut g, "birmingham", 0, p, Cotton, 1, 0);
    place(&mut g, "burton-on-trent", 0, p, Brewery, 1, 1);
    link(&mut g, "birmingham", "oxford", p);
    assert!(sells(&g).is_empty());
    // Two tiles in one location are separate merchants: only the buying one is offered.
    merchant(&mut g, OXFORD + 1, MerchantTile::Cotton);
    let offered = sells(&g);
    assert_eq!(offered.len(), 2);
    assert!(
        offered
            .iter()
            .all(|m| matches!(m, Move::Sell { merchant, .. } if *merchant == OXFORD + 1))
    );
    // Nothing to sell and no legal Sell: applying one is refused.
    let (mut g, _) = setup();
    assert_eq!(
        g.apply(&sell(0, OXFORD, vec![])),
        Err(RuleError::IllegalMove)
    );
}

// 9.4 steps 3-4: the sale consumes the beer shown on the tile, flips it and gains its income;
// an emptied Brewery flips too (its owner gains). Without merchant beer the merchant's barrel
// stays and there is no bonus.
#[test]
fn sell_consumes_beer_and_flips() {
    let (mut g, p) = setup();
    let cotton = place(&mut g, "birmingham", 0, p, Cotton, 1, 0);
    let o = (p + 1) % 4;
    let brewery = place(&mut g, "birmingham", 1, o, Brewery, 1, 1);
    merchant(&mut g, OXFORD, MerchantTile::Cotton);
    link(&mut g, "birmingham", "oxford", o);
    let (income, vp, money) = {
        let ps = &g.players[p as usize];
        (ps.income_space, ps.vp, ps.money)
    };
    let o_income = g.players[o as usize].income_space;
    let m = sell(cotton, OXFORD, vec![BeerSource::Brewery(brewery)]);
    assert!(g.legal_moves().contains(&m));
    g.apply(&m).unwrap();
    let t = g.slots[cotton as usize].as_ref().unwrap();
    assert!(t.flipped);
    let b = g.slots[brewery as usize].as_ref().unwrap();
    assert!(b.flipped && b.resources == 0);
    let ps = &g.players[p as usize];
    assert_eq!(ps.income_space, income + 5);
    assert_eq!((ps.vp, ps.money), (vp, money));
    assert_eq!(ps.discard.last(), Some(&ANY));
    assert!(ps.hand.is_empty());
    assert_eq!(
        g.players[o as usize].income_space,
        o_income + tile(Brewery, 1).income
    );
    assert!(g.merchant_beer[OXFORD as usize]);
    assert_eq!(g.phase, Phase::SellMore);
    assert_eq!(g.legal_moves(), vec![Move::EndAction]);
    g.apply(&Move::EndAction).unwrap();
    assert_eq!((g.phase, g.actions_left), (Phase::ChooseAction, 1));
}

// 9.4 / RQ-012: a tile showing no beer (Manufacturer III) sells with no beer, and never takes
// merchant beer or its bonus.
#[test]
fn sell_zero_beer_tile_needs_none() {
    let (mut g, p) = setup();
    let man = place(&mut g, "birmingham", 0, p, Manufacturer, 3, 0);
    merchant(&mut g, SHREWSBURY, MerchantTile::Manufacturer);
    link(&mut g, "birmingham", "oxford", p);
    link(&mut g, "birmingham", "worcester", p);
    link(&mut g, "worcester", "kidderminster", p);
    link(&mut g, "kidderminster", "coalbrookdale", p);
    link(&mut g, "coalbrookdale", "shrewsbury", p);
    assert_eq!(sells(&g), vec![sell(man, SHREWSBURY, vec![])]);
    let income = g.players[p as usize].income_space;
    g.apply(&sells(&g)[0]).unwrap();
    assert!(g.slots[man as usize].as_ref().unwrap().flipped);
    assert_eq!(g.players[p as usize].income_space, income + 4);
    assert_eq!(g.players[p as usize].vp, 0);
    assert!(g.merchant_beer[SHREWSBURY as usize]);
}

// 9.4 step 5: further tiles are sold in the same action with one card; flipped tiles are not
// offered again, and EndAction uses the one action.
#[test]
fn sell_another_in_same_action_one_card() {
    let (mut g, p) = setup();
    g.players[p as usize].hand = vec![ANY, ANY, Card::Location(loc::DERBY)];
    let a = place(&mut g, "birmingham", 0, p, Cotton, 1, 0);
    let b = place(&mut g, "birmingham", 1, p, Cotton, 1, 0);
    let brewery = place(&mut g, "burton-on-trent", 0, p, Brewery, 1, 2);
    merchant(&mut g, OXFORD, MerchantTile::Cotton);
    link(&mut g, "birmingham", "oxford", p);
    let beer = vec![BeerSource::Brewery(brewery)];
    g.apply(&sell(a, OXFORD, beer.clone())).unwrap();
    assert_eq!(g.players[p as usize].hand.len(), 2);
    assert_eq!(g.phase, Phase::SellMore);
    assert_eq!(g.actions_left, 2);
    // Only the unflipped tile; no card is part of the move.
    assert_eq!(
        sells(&g),
        vec![
            another(b, OXFORD, beer.clone()),
            another(b, OXFORD, vec![BeerSource::Merchant(OXFORD)]),
        ]
    );
    // A fresh Sell is not a SellMore move.
    assert_eq!(
        g.apply(&sell(b, OXFORD, beer.clone())),
        Err(RuleError::IllegalMove)
    );
    g.apply(&another(b, OXFORD, beer)).unwrap();
    assert_eq!(g.players[p as usize].hand.len(), 2);
    assert!(g.slots[brewery as usize].as_ref().unwrap().flipped);
    assert_eq!(g.legal_moves(), vec![Move::EndAction]);
    g.apply(&Move::EndAction).unwrap();
    assert_eq!((g.phase, g.actions_left), (Phase::ChooseAction, 1));
}

// 9.4 / RQ-009: taking the merchant barrel is optional; once taken the bonus is collected, and a
// barrel is used at most once, so a second sale to that merchant gets neither beer nor bonus.
#[test]
fn merchant_beer_grants_bonus_once() {
    let (mut g, p) = setup();
    let a = place(&mut g, "birmingham", 0, p, Cotton, 1, 0);
    let b = place(&mut g, "birmingham", 1, p, Cotton, 1, 0);
    let brewery = place(&mut g, "burton-on-trent", 0, p, Brewery, 1, 1);
    merchant(&mut g, SHREWSBURY, MerchantTile::Cotton);
    for (x, y) in [
        ("birmingham", "worcester"),
        ("worcester", "kidderminster"),
        ("kidderminster", "coalbrookdale"),
        ("coalbrookdale", "shrewsbury"),
    ] {
        link(&mut g, x, y, p);
    }
    let vp = g.players[p as usize].vp;
    let barrel = vec![BeerSource::Merchant(SHREWSBURY)];
    assert!(sells(&g).contains(&sell(a, SHREWSBURY, barrel.clone())));
    g.apply(&sell(a, SHREWSBURY, barrel)).unwrap();
    assert_eq!(g.players[p as usize].vp, vp + 4);
    assert!(!g.merchant_beer[SHREWSBURY as usize]);
    assert!(g.slots[brewery as usize].as_ref().unwrap().resources == 1);
    // The barrel is gone: only the brewery sells tile b, and there is no further bonus.
    let beer = vec![BeerSource::Brewery(brewery)];
    assert_eq!(sells(&g), vec![another(b, SHREWSBURY, beer.clone())]);
    g.apply(&another(b, SHREWSBURY, beer)).unwrap();
    assert_eq!(g.players[p as usize].vp, vp + 4);
}

// 9.4 / RQ-009 / RQ-041: Gloucester's merchant beer gives a free develop: no iron, the lowest
// tile of the chosen industry, never a lightbulb Pottery; then back to SellMore.
#[test]
fn gloucester_bonus_develop_free() {
    let (mut g, p) = setup();
    let cotton = place(&mut g, "worcester", 0, p, Cotton, 1, 0);
    merchant(&mut g, GLOUCESTER, MerchantTile::Cotton);
    link(&mut g, "gloucester", "worcester", p);
    let m = sell(cotton, GLOUCESTER, vec![BeerSource::Merchant(GLOUCESTER)]);
    assert!(g.legal_moves().contains(&m));
    let (money, iron) = (g.players[p as usize].money, g.iron_market);
    g.apply(&m).unwrap();
    assert_eq!(g.phase, Phase::BonusDevelop);
    // Pottery I is a lightbulb at the bottom of the mat: Pottery cannot be developed.
    assert_eq!(
        g.legal_moves(),
        [Cotton, Manufacturer, Coal, Industry::Iron, Brewery]
            .into_iter()
            .map(|industry| Move::BonusDevelop { industry })
            .collect::<Vec<_>>()
    );
    assert!(
        !g.legal_moves()
            .contains(&Move::BonusDevelop { industry: Pottery })
    );
    let before = g.players[p as usize].mat_remaining[Coal as usize];
    g.apply(&Move::BonusDevelop { industry: Coal }).unwrap();
    let after = g.players[p as usize].mat_remaining[Coal as usize];
    assert_eq!(after[0], before[0] - 1);
    assert_eq!(&after[1..], &before[1..]);
    assert_eq!((g.players[p as usize].money, g.iron_market), (money, iron));
    assert_eq!(g.phase, Phase::SellMore);
    assert_eq!(g.legal_moves(), vec![Move::EndAction]);
}

// RQ-009: with nothing developable the bonus is skipped and the phase never stalls.
#[test]
fn gloucester_bonus_skipped_when_nothing_developable() {
    let (mut g, p) = setup();
    let cotton = place(&mut g, "worcester", 0, p, Cotton, 1, 0);
    merchant(&mut g, GLOUCESTER, MerchantTile::Cotton);
    link(&mut g, "gloucester", "worcester", p);
    g.players[p as usize].mat_remaining = [[0; 8]; 6];
    let barrel = vec![BeerSource::Merchant(GLOUCESTER)];
    g.apply(&sell(cotton, GLOUCESTER, barrel)).unwrap();
    assert_eq!(g.phase, Phase::SellMore);
    assert_eq!(g.legal_moves(), vec![Move::EndAction]);
}

// 9.4 / RQ-008 / RQ-040: Oxford advances the income marker 2 spaces (on top of the tile's own
// income), capped at space 99.
#[test]
fn oxford_bonus_income() {
    let (mut g, p) = setup();
    let cotton = place(&mut g, "birmingham", 0, p, Cotton, 1, 0);
    merchant(&mut g, OXFORD, MerchantTile::Cotton);
    link(&mut g, "birmingham", "oxford", p);
    let barrel = vec![BeerSource::Merchant(OXFORD)];
    let income = g.players[p as usize].income_space;
    let mut capped = g.clone();
    g.apply(&sell(cotton, OXFORD, barrel.clone())).unwrap();
    assert_eq!(g.players[p as usize].income_space, income + 5 + 2);
    assert_eq!(g.phase, Phase::SellMore);

    capped.players[p as usize].income_space = MAX_INCOME_SPACE - 1;
    capped.apply(&sell(cotton, OXFORD, barrel)).unwrap();
    assert_eq!(capped.players[p as usize].income_space, MAX_INCOME_SPACE);
}

// 9.4: Warrington's bonus is £5; it is income, not spending.
#[test]
fn warrington_bonus_money_5() {
    let (mut g, p) = setup();
    let cotton = place(&mut g, "stoke-on-trent", 0, p, Cotton, 1, 0);
    merchant(&mut g, WARRINGTON, MerchantTile::Any);
    link(&mut g, "stoke-on-trent", "warrington", p);
    let money = g.players[p as usize].money;
    g.apply(&sell(
        cotton,
        WARRINGTON,
        vec![BeerSource::Merchant(WARRINGTON)],
    ))
    .unwrap();
    assert_eq!(g.players[p as usize].money, money + 5);
    assert_eq!(g.players[p as usize].spent, 0);
}

// RQ-044: without merchant beer every accepting merchant is equivalent, so the sale is offered
// once per tile and plan (at the lowest accepting slot); merchant-beer sales stay per slot.
#[test]
fn sell_without_merchant_beer_offered_once() {
    let (mut g, p) = setup();
    let cotton = place(&mut g, "birmingham", 0, p, Cotton, 1, 0);
    let brewery = place(&mut g, "burton-on-trent", 0, p, Brewery, 1, 1);
    // Two accepting merchants in different locations (Oxford, Shrewsbury) plus a second Oxford slot.
    merchant(&mut g, OXFORD, MerchantTile::Cotton);
    merchant(&mut g, OXFORD + 1, MerchantTile::Any);
    merchant(&mut g, SHREWSBURY, MerchantTile::Any);
    link(&mut g, "birmingham", "oxford", p);
    for (x, y) in [
        ("birmingham", "worcester"),
        ("worcester", "kidderminster"),
        ("kidderminster", "coalbrookdale"),
        ("coalbrookdale", "shrewsbury"),
    ] {
        link(&mut g, x, y, p);
    }
    let offered = sells(&g);
    let plain = vec![BeerSource::Brewery(brewery)];
    assert_eq!(
        offered
            .iter()
            .filter(|m| matches!(m, Move::Sell { beer, .. } if *beer == plain))
            .count(),
        1
    );
    assert!(offered.contains(&sell(cotton, SHREWSBURY, plain)));
    for m in [SHREWSBURY, OXFORD, OXFORD + 1] {
        assert!(offered.contains(&sell(cotton, m, vec![BeerSource::Merchant(m)])));
    }
    assert_eq!(offered.len(), 4);
}

// 9.4 / 8.3: a tile showing 2 beer may mix the merchant's barrel with a brewery barrel; both
// are consumed and the merchant bonus is collected.
#[test]
fn sell_two_beer_tile_mixes_merchant_and_brewery_beer() {
    let (mut g, p) = setup();
    let man = place(&mut g, "birmingham", 0, p, Manufacturer, 5, 0);
    let brewery = place(&mut g, "burton-on-trent", 0, p, Brewery, 1, 1);
    merchant(&mut g, OXFORD, MerchantTile::Manufacturer);
    link(&mut g, "birmingham", "oxford", p);
    let mixed = vec![BeerSource::Brewery(brewery), BeerSource::Merchant(OXFORD)];
    // One brewery barrel alone cannot cover 2 beer: the mixed plan is the only sale.
    assert_eq!(sells(&g), vec![sell(man, OXFORD, mixed.clone())]);
    let income = g.players[p as usize].income_space;
    g.apply(&sell(man, OXFORD, mixed)).unwrap();
    assert!(g.slots[man as usize].as_ref().unwrap().flipped);
    assert!(g.slots[brewery as usize].as_ref().unwrap().flipped);
    assert!(!g.merchant_beer[OXFORD as usize]);
    // Tile income (2 spaces) plus Oxford's 2 spaces; the emptied brewery is also p's.
    assert_eq!(
        g.players[p as usize].income_space,
        income + tile(Manufacturer, 5).income + 2 + tile(Brewery, 1).income
    );
}

// 9.4: Manufacturer and Pottery tiles sell to their own merchant tiles (and not to the other's).
#[test]
fn sell_manufacturer_and_pottery_to_matching_merchants() {
    for (ind, level, good, other) in [
        (
            Manufacturer,
            1,
            MerchantTile::Manufacturer,
            MerchantTile::Pottery,
        ),
        (
            Pottery,
            1,
            MerchantTile::Pottery,
            MerchantTile::Manufacturer,
        ),
    ] {
        let (mut g, p) = setup();
        let t = place(&mut g, "birmingham", 0, p, ind, level, 0);
        let brewery = place(&mut g, "burton-on-trent", 0, p, Brewery, 1, 1);
        link(&mut g, "birmingham", "oxford", p);
        merchant(&mut g, OXFORD, other);
        assert!(sells(&g).is_empty(), "{ind:?}");
        merchant(&mut g, OXFORD, good);
        assert_eq!(
            sells(&g),
            vec![
                sell(t, OXFORD, vec![BeerSource::Brewery(brewery)]),
                sell(t, OXFORD, vec![BeerSource::Merchant(OXFORD)]),
            ],
            "{ind:?}"
        );
        g.apply(&sell(t, OXFORD, vec![BeerSource::Merchant(OXFORD)]))
            .unwrap();
        assert!(g.slots[t as usize].as_ref().unwrap().flipped, "{ind:?}");
    }
}

// 9.4 step 5: a merchant bonus is also collected on a further sale in the same action
// (Warrington £5 on `SellAnother`, Nottingham 3 VP on `SellAnother`).
#[test]
fn bonus_taken_during_sell_another() {
    let (mut g, p) = setup();
    g.players[p as usize].hand = vec![ANY, ANY];
    let a = place(&mut g, "stoke-on-trent", 0, p, Cotton, 1, 0);
    let b = place(&mut g, "stoke-on-trent", 1, p, Cotton, 1, 0);
    let c = place(&mut g, "derby", 0, p, Cotton, 1, 0);
    let brewery = place(&mut g, "burton-on-trent", 0, p, Brewery, 1, 1);
    merchant(&mut g, WARRINGTON, MerchantTile::Cotton);
    merchant(&mut g, NOTTINGHAM, MerchantTile::Cotton);
    link(&mut g, "stoke-on-trent", "warrington", p);
    link(&mut g, "derby", "nottingham", p);
    let (money, vp) = (g.players[p as usize].money, g.players[p as usize].vp);
    g.apply(&sell(a, WARRINGTON, vec![BeerSource::Brewery(brewery)]))
        .unwrap();
    assert_eq!(
        (g.players[p as usize].money, g.players[p as usize].vp),
        (money, vp)
    );
    let m = another(b, WARRINGTON, vec![BeerSource::Merchant(WARRINGTON)]);
    assert!(sells(&g).contains(&m));
    g.apply(&m).unwrap();
    assert_eq!(g.players[p as usize].money, money + 5);
    assert_eq!(g.players[p as usize].spent, 0);
    let m = another(c, NOTTINGHAM, vec![BeerSource::Merchant(NOTTINGHAM)]);
    assert!(sells(&g).contains(&m));
    g.apply(&m).unwrap();
    assert_eq!(g.players[p as usize].vp, vp + 3);
    assert_eq!(g.phase, Phase::SellMore);
}

// 9.4 step 2 / RQ-016: each sale names one connected accepting merchant tile and may take at
// most that tile's barrel. Two connected merchants with barrels give a choice per sale, but a
// 2-beer tile cannot take both barrels in one sale.
#[test]
fn sale_chooses_one_merchant_and_takes_at_most_its_barrel() {
    let (mut g, p) = setup();
    let cotton = place(&mut g, "birmingham", 0, p, Cotton, 1, 0);
    merchant(&mut g, OXFORD, MerchantTile::Cotton);
    merchant(&mut g, SHREWSBURY, MerchantTile::Cotton);
    link(&mut g, "birmingham", "oxford", p);
    for (x, y) in [
        ("birmingham", "worcester"),
        ("worcester", "kidderminster"),
        ("kidderminster", "coalbrookdale"),
        ("coalbrookdale", "shrewsbury"),
    ] {
        link(&mut g, x, y, p);
    }
    let offered = sells(&g);
    assert_eq!(offered.len(), 2);
    for m in [OXFORD, SHREWSBURY] {
        assert!(offered.contains(&sell(cotton, m, vec![BeerSource::Merchant(m)])));
    }
    // Choosing Shrewsbury leaves Oxford's barrel untouched.
    g.apply(&sell(
        cotton,
        SHREWSBURY,
        vec![BeerSource::Merchant(SHREWSBURY)],
    ))
    .unwrap();
    assert!(g.merchant_beer[OXFORD as usize]);
    assert!(!g.merchant_beer[SHREWSBURY as usize]);

    // A 2-beer tile with no brewery: one merchant's single barrel cannot cover it.
    let (mut g, p) = setup();
    place(&mut g, "birmingham", 0, p, Manufacturer, 5, 0);
    merchant(&mut g, OXFORD, MerchantTile::Manufacturer);
    merchant(&mut g, OXFORD + 1, MerchantTile::Manufacturer);
    link(&mut g, "birmingham", "oxford", p);
    assert!(sells(&g).is_empty());
}
