// SPDX-License-Identifier: Apache-2.0
//! The Network action: canal, rail and double rail.
//! Rules: docs/rules-reference.md section 9.2 (Network), section 7 (network), section 8.1 and
//! 8.3 (coal and beer sourcing), section 6 (turn structure); RULES-QUESTIONS.md RQ-013, RQ-043.

mod common;

use brass_engine::data::*;
use brass_engine::moves::{Move, Phase, RuleError};
use brass_engine::resources::ResourcePlan;
use brass_engine::state::GameState;
use common::*;

use Industry::{Brewery, Coal, Cotton};

const ANY: Card = Card::Location(loc::BIRMINGHAM);

/// A fresh game whose acting player holds two identical cards (so the first action does not
/// empty the hand and end the turn), has `money` and 2 actions left.
fn setup(era: Era, money: u16) -> (GameState, PlayerId) {
    let mut g = empty_game();
    g.era = era;
    g.actions_left = 2;
    let p = g.current_player().expect("someone is to act");
    g.players[p as usize].hand = vec![ANY, ANY];
    g.players[p as usize].money = money;
    (g, p)
}

fn opp(p: PlayerId) -> PlayerId {
    (p + 1) % 4
}

fn sl(l: &str, i: usize) -> SlotId {
    let l = loc_id(l).expect("location");
    SLOTS
        .iter()
        .enumerate()
        .filter(|(_, s)| s.loc == l)
        .nth(i)
        .expect("slot")
        .0 as SlotId
}

/// Links of the legal `Canal` moves, ascending.
fn canal_links(g: &GameState) -> Vec<LinkId> {
    g.legal_moves()
        .into_iter()
        .filter_map(|m| match m {
            Move::Canal { link, .. } => Some(link),
            _ => None,
        })
        .collect()
}

/// The legal `Rail` moves for `link`.
fn rails(g: &GameState, link: LinkId) -> Vec<Move> {
    g.legal_moves()
        .into_iter()
        .filter(|m| matches!(m, Move::Rail { link: l, .. } if *l == link))
        .collect()
}

/// The legal `SecondRail` moves for `link`.
fn seconds(g: &GameState, link: LinkId) -> Vec<Move> {
    g.legal_moves()
        .into_iter()
        .filter(|m| matches!(m, Move::SecondRail { link: l, .. } if *l == link))
        .collect()
}

fn market(n: u8) -> ResourcePlan {
    ResourcePlan {
        from_tiles: vec![],
        from_market: n,
    }
}

fn from_tile(slot: SlotId) -> ResourcePlan {
    ResourcePlan {
        from_tiles: vec![(slot, 1)],
        from_market: 0,
    }
}

// ---------------------------------------------------------------- canal

// 9.2 Canal Era: a canal link costs £3, no coal, and must touch the player's network. Only canal
// lines qualify (Birmingham–Nuneaton is rail-only); opponents' links never extend the network.
#[test]
fn canal_costs_3_and_requires_adjacency() {
    let (mut g, p) = setup(Era::Canal, 20);
    place(&mut g, "birmingham", 1, p, Cotton, 1, 0);
    link(&mut g, "cannock", "walsall", opp(p));
    let offered = canal_links(&g);
    let expected: Vec<LinkId> = (0..LINKS.len() as LinkId)
        .filter(|&i| {
            let l = &LINKS[i as usize];
            l.canal
                && i != link_id("cannock", "walsall")
                && (l.a == loc::BIRMINGHAM || l.b == loc::BIRMINGHAM)
        })
        .collect();
    assert_eq!(offered, expected);
    assert!(!offered.contains(&link_id("birmingham", "nuneaton")));
    assert!(!offered.contains(&link_id("cannock", "stafford")));

    let l = link_id("birmingham", "dudley");
    g.apply(&Move::Canal { card: ANY, link: l }).unwrap();
    assert_eq!(g.links[l as usize], Some(p));
    let ps = &g.players[p as usize];
    assert_eq!((ps.money, ps.spent), (17, 3));
    assert_eq!(ps.hand, vec![ANY]);
    assert_eq!(ps.discard.last(), Some(&ANY));
    assert_eq!(g.actions_left, 1);
    assert_eq!(g.phase, Phase::ChooseAction);

    // The new link extends the network to Dudley, but a link elsewhere stays refused.
    let mut g2 = g.clone();
    g2.players[p as usize].hand = vec![ANY];
    assert!(canal_links(&g2).contains(&link_id("dudley", "kidderminster")));
    let far = Move::Canal {
        card: ANY,
        link: link_id("cannock", "stafford"),
    };
    assert_eq!(g2.apply(&far), Err(RuleError::IllegalMove));
}

// 9.2 canal cost: with less than £3 no canal link is offered.
#[test]
fn canal_needs_3_pounds() {
    let (g, _) = setup(Era::Canal, 2);
    assert!(canal_links(&g).is_empty());
}

// 9.2 exception: with no tiles or links on the board any unbuilt canal line may be built.
#[test]
fn canal_any_link_without_presence() {
    let (g, _) = setup(Era::Canal, 20);
    let all: Vec<LinkId> = (0..LINKS.len() as LinkId)
        .filter(|&i| LINKS[i as usize].canal)
        .collect();
    assert_eq!(canal_links(&g), all);
    // An opponent's presence does not matter; an own link ends the exception.
    let (mut g, p) = setup(Era::Canal, 20);
    link(&mut g, "cannock", "walsall", opp(p));
    assert_eq!(canal_links(&g).len(), all.len() - 1);
    link(&mut g, "leek", "stoke-on-trent", p);
    assert!(!canal_links(&g).contains(&link_id("birmingham", "dudley")));
}

// 9.2: each line holds one Link tile, and a Rail Era link must be a rail line (and the reverse
// in the Canal Era).
#[test]
fn rail_link_not_buildable_in_canal_era() {
    let (mut g, p) = setup(Era::Canal, 50);
    place(&mut g, "birmingham", 1, p, Cotton, 1, 0);
    let nun = link_id("birmingham", "nuneaton");
    assert!(!canal_links(&g).contains(&nun));
    assert!(
        g.legal_moves()
            .iter()
            .all(|m| !matches!(m, Move::Rail { .. }))
    );
    let m = Move::Rail {
        card: ANY,
        link: nun,
        coal: market(1),
    };
    assert_eq!(g.apply(&m), Err(RuleError::IllegalMove));

    // Rail Era: no canals, and a built line is gone.
    let (mut g, p) = setup(Era::Rail, 50);
    place(&mut g, "birmingham", 1, p, Cotton, 1, 0);
    link(&mut g, "birmingham", "oxford", opp(p));
    assert!(
        g.legal_moves()
            .iter()
            .all(|m| !matches!(m, Move::Canal { .. }))
    );
    assert!(rails(&g, link_id("birmingham", "oxford")).is_empty());
    assert!(!rails(&g, link_id("birmingham", "nuneaton")).is_empty());
}

// ---------------------------------------------------------------- rail

// 9.2 Rail Era: a rail link costs £5 + 1 coal, and the coal must be connected to the link after
// it is placed. A mine at the far end of the new link counts; with no coal in reach (and no
// merchant connection for the market) the link is not offered.
#[test]
fn rail_needs_coal_connected_after_placement() {
    let (mut g, p) = setup(Era::Rail, 20);
    place(&mut g, "birmingham", 1, p, Cotton, 1, 0);
    let mine = place(&mut g, "dudley", 0, opp(p), Coal, 1, 2);
    let dudley = link_id("birmingham", "dudley");
    let coventry = link_id("birmingham", "coventry");
    assert_eq!(
        rails(&g, dudley),
        vec![Move::Rail {
            card: ANY,
            link: dudley,
            coal: from_tile(mine)
        }]
    );
    assert!(rails(&g, coventry).is_empty());

    g.apply(&rails(&g, dudley)[0]).unwrap();
    assert_eq!(g.links[dudley as usize], Some(p));
    assert_eq!(g.slots[mine as usize].as_ref().unwrap().resources, 1);
    let ps = &g.players[p as usize];
    assert_eq!((ps.money, ps.spent), (15, 5));
    assert_eq!(g.phase, Phase::SecondRail);
    assert_eq!(g.actions_left, 2);
    assert!(g.legal_moves().contains(&Move::EndAction));
    g.apply(&Move::EndAction).unwrap();
    assert_eq!((g.phase, g.actions_left), (Phase::ChooseAction, 1));
}

// 9.2 / 7: the coal market is a coal source only for a link connected to a merchant location;
// it is priced per cube (£1 here: the market has 13 of 14 cubes) and added to the cost.
#[test]
fn rail_market_coal_priced_and_affordability_filtered() {
    let (mut g, p) = setup(Era::Rail, 5);
    place(&mut g, "birmingham", 1, p, Cotton, 1, 0);
    link(&mut g, "birmingham", "oxford", opp(p));
    let coventry = link_id("birmingham", "coventry");
    // £5 + £1 market coal: too expensive with £5.
    assert!(rails(&g, coventry).is_empty());
    g.players[p as usize].money = 6;
    assert_eq!(
        rails(&g, coventry),
        vec![Move::Rail {
            card: ANY,
            link: coventry,
            coal: market(1)
        }]
    );
    g.apply(&rails(&g, coventry)[0]).unwrap();
    assert_eq!(g.players[p as usize].money, 0);
    assert_eq!(g.players[p as usize].spent, 6);
    assert_eq!(g.coal_market, 12);
}

// ---------------------------------------------------------------- double rail

/// Own Birmingham tile, opponent's Birmingham–Oxford link (market access), a 2-cube opponent
/// mine in Birmingham and an own 1-barrel Brewery at Burton. First link Birmingham–Coventry,
/// second Coventry–Nuneaton.
fn double_rail_setup(money: u16) -> (GameState, PlayerId, SlotId, SlotId) {
    let (mut g, p) = setup(Era::Rail, money);
    place(&mut g, "birmingham", 1, p, Cotton, 1, 0);
    let mine = place(&mut g, "birmingham", 2, opp(p), Coal, 1, 2);
    let brewery = place(&mut g, "burton-on-trent", 1, p, Brewery, 1, 1);
    (g, p, mine, brewery)
}

// 9.2 rulings: each link's coal is checked after that link is placed. The second link may use
// the first for connection (the Birmingham mine is reachable from Coventry–Nuneaton only through
// Birmingham–Coventry); the first link may not use the second.
#[test]
fn double_rail_coal_checked_per_link() {
    let (mut g, _, mine, brewery) = double_rail_setup(20);
    let (first, second) = (
        link_id("birmingham", "coventry"),
        link_id("coventry", "nuneaton"),
    );
    // The second link is not offered as a first link: no coal reaches it yet.
    assert!(rails(&g, second).is_empty());
    g.apply(&rails(&g, first)[0]).unwrap();
    assert_eq!(g.slots[mine as usize].as_ref().unwrap().resources, 1);
    // Mine coal comes first, so there is no market alternative (RQ-013 / 8.1 ordering).
    assert_eq!(
        seconds(&g, second),
        vec![Move::SecondRail {
            link: second,
            coal: from_tile(mine),
            beer: brewery
        }]
    );

    // The first link never takes coal that only the second link reaches: a mine in Nuneaton.
    let (mut g, p2, _, _) = double_rail_setup(20);
    g.slots[sl("birmingham", 2) as usize] = None;
    let far_mine = place(&mut g, "nuneaton", 0, opp(p2), Coal, 1, 1);
    assert!(rails(&g, first).is_empty()); // no mine in reach, no market link
    link(&mut g, "birmingham", "oxford", opp(p2));
    assert_eq!(
        rails(&g, first),
        vec![Move::Rail {
            card: ANY,
            link: first,
            coal: market(1)
        }]
    );
    g.apply(&rails(&g, first)[0]).unwrap();
    assert_eq!(g.slots[far_mine as usize].as_ref().unwrap().resources, 1);
    // The second link's coal, though, is the Nuneaton mine at its own end.
    assert_eq!(seconds(&g, second).len(), 1);
    assert!(
        matches!(&seconds(&g, second)[0], Move::SecondRail { coal, .. } if coal == &from_tile(far_mine))
    );
}

// 9.2: only brewery beer pays for the second link: your own Brewery anywhere, or an opponent's
// Brewery connected to the second link once placed; never merchant beer.
#[test]
fn double_rail_needs_brewery_beer_not_merchant() {
    let (mut g, p, _, own) = double_rail_setup(30);
    let (first, second) = (
        link_id("birmingham", "coventry"),
        link_id("coventry", "nuneaton"),
    );
    // Merchant beer is available at the board's merchants; it must not count.
    assert!(g.merchant_beer.iter().any(|&b| b));
    g.slots[own as usize] = None;
    g.apply(&rails(&g, first)[0]).unwrap();
    assert!(seconds(&g, second).is_empty());
    assert_eq!(g.legal_moves(), vec![Move::EndAction]);

    // An opponent's Brewery: unconnected (Walsall) is no help; connected through the first link
    // (Birmingham) is; and so is one in the second link's own end (Nuneaton).
    let walsall = place(&mut g, "walsall", 0, opp(p), Brewery, 1, 1);
    assert!(seconds(&g, second).is_empty());
    let birm = place(&mut g, "birmingham", 3, opp(p), Brewery, 1, 1);
    let nun = place(&mut g, "nuneaton", 1, opp(p), Brewery, 1, 1);
    let beers: Vec<SlotId> = seconds(&g, second)
        .iter()
        .map(|m| match m {
            Move::SecondRail { beer, .. } => *beer,
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(beers, vec![birm.min(nun), birm.max(nun)]);
    assert!(!beers.contains(&walsall));

    // Taking the barrel flips the emptied Brewery and pays its owner income.
    let income = g.players[opp(p) as usize].income_space;
    let m = seconds(&g, second)
        .into_iter()
        .find(|m| matches!(m, Move::SecondRail { beer, .. } if *beer == birm))
        .unwrap();
    g.apply(&m).unwrap();
    let t = g.slots[birm as usize].as_ref().unwrap();
    assert!(t.flipped);
    assert!(g.players[opp(p) as usize].income_space > income);
    assert!(g.merchant_beer.iter().any(|&b| b));
}

// 9.2: a double rail costs £15 in all (£5 + £10) plus the coal; the second link is offered
// only if the remaining money covers its £10, and the action ends afterwards.
#[test]
fn double_rail_total_cost_15() {
    let (mut g, p, mine, brewery) = double_rail_setup(15);
    let (first, second) = (
        link_id("birmingham", "coventry"),
        link_id("coventry", "nuneaton"),
    );
    g.apply(&rails(&g, first)[0]).unwrap();
    assert_eq!(g.players[p as usize].money, 10);
    let m = seconds(&g, second)
        .pop()
        .expect("second link affordable with exactly £10");
    g.apply(&m).unwrap();
    let ps = &g.players[p as usize];
    assert_eq!((ps.money, ps.spent), (0, 15));
    assert_eq!(g.links[first as usize], Some(p));
    assert_eq!(g.links[second as usize], Some(p));
    assert_eq!(g.slots[mine as usize].as_ref().unwrap().resources, 0);
    assert!(g.slots[mine as usize].as_ref().unwrap().flipped);
    assert!(g.slots[brewery as usize].as_ref().unwrap().flipped);
    assert_eq!((g.phase, g.actions_left), (Phase::ChooseAction, 1));

    // One pound short: the first link is fine, the second is not offered.
    let (mut g, _, _, _) = double_rail_setup(14);
    g.apply(&rails(&g, first)[0]).unwrap();
    assert!(seconds(&g, second).is_empty());
    assert_eq!(g.legal_moves(), vec![Move::EndAction]);
}

// RQ-043: each player has 14 Link tiles. With none left no link is offered, and a double rail
// needs two tiles.
#[test]
fn link_tiles_run_out() {
    let spare: Vec<LinkId> = (0..LINKS.len() as LinkId)
        .filter(|&i| {
            let l = &LINKS[i as usize];
            ![loc::BIRMINGHAM, loc::COVENTRY, loc::NUNEATON].contains(&l.a)
                && ![loc::BIRMINGHAM, loc::COVENTRY, loc::NUNEATON].contains(&l.b)
        })
        .collect();
    let (mut g, p) = setup(Era::Canal, 20);
    for &i in &spare[..14] {
        g.links[i as usize] = Some(p);
    }
    assert!(canal_links(&g).is_empty());
    g.links[spare[0] as usize] = None;
    assert!(!canal_links(&g).is_empty());

    let (first, second) = (
        link_id("birmingham", "coventry"),
        link_id("coventry", "nuneaton"),
    );
    for owned_elsewhere in [12, 13] {
        let (mut g, p, _, _) = double_rail_setup(30);
        for &i in &spare[..owned_elsewhere] {
            g.links[i as usize] = Some(p);
        }
        g.apply(&rails(&g, first)[0]).unwrap();
        // 12 others + first = 13 placed: the 14th is still available; 13 + first: none left.
        assert_eq!(!seconds(&g, second).is_empty(), owned_elsewhere == 12);
    }
}

// RQ-043 adjacency: a farm brewery tile is the player's whole presence; the link that touches
// that farm (Kidderminster–Worcester, its third endpoint) is offered, a far link is not.
#[test]
fn canal_link_offered_at_farm_brewery_presence() {
    let (mut g, p) = setup(Era::Canal, 20);
    place(&mut g, "farm-south", 0, p, Brewery, 1, 1);
    let offered = canal_links(&g);
    assert_eq!(offered, vec![link_id("kidderminster", "worcester")]);
}

// 9.2 Rail Era no-presence exception: any rail line is offered, but it still needs coal.
#[test]
fn rail_no_presence_any_line_still_needs_coal() {
    let (mut g, p) = setup(Era::Rail, 20);
    // No coal on the board: only lines touching a merchant location (market coal) are offered.
    assert!(rails(&g, link_id("birmingham", "coventry")).is_empty());
    assert!(!rails(&g, link_id("birmingham", "oxford")).is_empty());
    let touches_merchant =
        |l: &LinkDef| MERCHANT_SPACES.iter().any(|m| m.loc == l.a || m.loc == l.b);
    for m in g.legal_moves() {
        if let Move::Rail { link, .. } = m {
            assert!(touches_merchant(&LINKS[link as usize]));
        }
    }
    // A mine at the far end of a remote link makes exactly the lines it touches offerable.
    place(&mut g, "stafford", 0, opp(p), Coal, 1, 2);
    let far = link_id("cannock", "stafford");
    assert!(!rails(&g, far).is_empty());
    assert!(rails(&g, link_id("birmingham", "coventry")).is_empty());
}

// 9.2 / 7: the second link's market coal is added to its £10; £10 left after the first link is
// not enough when the second link also needs market coal.
#[test]
fn second_rail_market_coal_counts_toward_cost() {
    let (mut g, p) = setup(Era::Rail, 16);
    place(&mut g, "birmingham", 1, p, Cotton, 1, 0);
    link(&mut g, "birmingham", "oxford", opp(p));
    place(&mut g, "burton-on-trent", 1, p, Brewery, 1, 1);
    let (first, second) = (
        link_id("birmingham", "coventry"),
        link_id("coventry", "nuneaton"),
    );
    // First link: £5 + £1 market coal, leaving £10. The market is then at £2 a cube, so the
    // second link needs £10 + £2.
    g.apply(&rails(&g, first)[0]).unwrap();
    assert_eq!(g.players[p as usize].money, 10);
    assert!(seconds(&g, second).is_empty());
    g.players[p as usize].money = 11;
    assert!(seconds(&g, second).is_empty());
    g.players[p as usize].money = 12;
    assert!(!seconds(&g, second).is_empty());
}
