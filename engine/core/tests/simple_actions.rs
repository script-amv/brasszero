// SPDX-License-Identifier: Apache-2.0
//! Move model, phases and the simple actions: Loan, Pass, Scout, Develop.
//! Rules: docs/rules-reference.md section 6 (turn structure, discards, Pass), section 3.2
//! (wild cards), section 9.3 (Develop), 9.5 (Loan), 9.6 (Scout); RULES-QUESTIONS.md RQ-002,
//! RQ-003, RQ-010, RQ-026, RQ-030.

mod common;

use std::collections::HashSet;

use brass_engine::data::*;
use brass_engine::moves::{Move, Phase, RuleError};
use brass_engine::network::iron_buy_cost;
use brass_engine::resources::ResourcePlan;
use brass_engine::state::GameState;
use common::*;

use Industry::{Coal, Iron, Pottery};

const BIRM: Card = Card::Location(loc::BIRMINGHAM);
const DERBY: Card = Card::Location(loc::DERBY);
const COAL_CARD: Card = Card::Industry(IndustryCard::Coal);
const IRON_CARD: Card = Card::Industry(IndustryCard::Iron);

/// The player to act. Panics if nobody is.
fn me(g: &GameState) -> PlayerId {
    g.current_player().expect("someone is to act")
}

/// Replace the acting player's hand.
fn set_hand(g: &mut GameState, cards: &[Card]) -> PlayerId {
    let p = me(g);
    g.players[p as usize].hand = cards.to_vec();
    p
}

/// A fresh game whose acting player has `cards` and 2 actions left.
fn game_with_hand(cards: &[Card]) -> (GameState, PlayerId) {
    let mut g = empty_game();
    g.actions_left = 2;
    let p = set_hand(&mut g, cards);
    (g, p)
}

fn mat(g: &GameState, p: PlayerId, ind: Industry, level: u8) -> u8 {
    g.players[p as usize].mat_remaining[ind as usize][level as usize - 1]
}

fn market_iron(n: u8) -> ResourcePlan {
    ResourcePlan {
        from_tiles: vec![],
        from_market: n,
    }
}

fn develop_industries(g: &GameState) -> HashSet<Industry> {
    g.legal_moves()
        .iter()
        .filter_map(|m| match m {
            Move::Develop { industry, .. } | Move::DevelopSecond { industry, .. } => {
                Some(*industry)
            }
            _ => None,
        })
        .collect()
}

fn has_scout(g: &GameState) -> bool {
    g.legal_moves()
        .iter()
        .any(|m| matches!(m, Move::Scout { .. }))
}

// ---------------------------------------------------------------- Loan (section 9.5)

// 9.5: take £30, income marker back 3 income levels onto the top space of the new level.
// Worked examples: space 10 (level 0) -> space 7; space 30 (level 10) -> space 24.
// Section 6: loan money is received, not spent, so `spent` is unchanged. Discards the card.
#[test]
fn loan_adds_30_and_drops_three_levels() {
    let (mut g, p) = game_with_hand(&[BIRM, DERBY]);
    g.apply(&Move::Loan { card: BIRM }).unwrap();
    let ps = &g.players[p as usize];
    assert_eq!(ps.money, START_MONEY + 30);
    assert_eq!(ps.income_space, 7);
    assert_eq!(ps.spent, 0);
    assert_eq!(ps.hand, vec![DERBY]);
    assert_eq!(ps.discard.last(), Some(&BIRM));
    // One action used, the same player acts again (2-action turn).
    assert_eq!(g.actions_left, 1);
    assert_eq!(me(&g), p);

    g.players[p as usize].income_space = 30;
    g.apply(&Move::Loan { card: DERBY }).unwrap();
    assert_eq!(g.players[p as usize].income_space, 24);
    assert_eq!(g.players[p as usize].money, START_MONEY + 60);
    // Second action done: the turn passes on.
    assert_ne!(me(&g), p);
}

// Review Focus 2, RQ-002: illegal if it takes income below level -10. Level -8 (space 2) is
// illegal; level -7 (space 3) is legal and lands exactly on -10 (space 0). Never clamps.
#[test]
fn loan_illegal_when_result_below_minus_10() {
    let (mut g, p) = game_with_hand(&[BIRM, DERBY]);
    g.players[p as usize].income_space = 2;
    assert_eq!(income_level(2), -8);
    assert!(
        !g.legal_moves()
            .iter()
            .any(|m| matches!(m, Move::Loan { .. }))
    );
    let before = g.clone();
    assert_eq!(
        g.apply(&Move::Loan { card: BIRM }),
        Err(RuleError::IllegalMove)
    );
    assert_eq!(g, before);

    g.players[p as usize].income_space = 3;
    assert_eq!(income_level(3), -7);
    g.apply(&Move::Loan { card: BIRM }).unwrap();
    assert_eq!(g.players[p as usize].income_space, 0);
}

// One Loan move per distinct card value.
#[test]
fn loan_offered_once_per_distinct_card() {
    let (g, _) = game_with_hand(&[BIRM, BIRM, DERBY]);
    let loans: Vec<Move> = g
        .legal_moves()
        .into_iter()
        .filter(|m| matches!(m, Move::Loan { .. }))
        .collect();
    // Ordered by card value.
    let mut want = vec![Move::Loan { card: BIRM }, Move::Loan { card: DERBY }];
    want.sort();
    assert_eq!(loans, want);
}

// ---------------------------------------------------------------- Pass (section 6)

// Section 6: passing still discards exactly one card face up and does nothing else.
#[test]
fn pass_discards_card() {
    let (mut g, p) = game_with_hand(&[BIRM, COAL_CARD]);
    let before = g.players[p as usize].clone();
    g.apply(&Move::Pass { card: COAL_CARD }).unwrap();
    let ps = &g.players[p as usize];
    assert_eq!(ps.hand, vec![BIRM]);
    assert_eq!(ps.discard.len(), before.discard.len() + 1);
    assert_eq!(ps.discard.last(), Some(&COAL_CARD));
    assert_eq!(ps.money, before.money);
    assert_eq!(ps.income_space, before.income_space);
    assert_eq!(ps.mat_remaining, before.mat_remaining);
    assert_eq!(g.actions_left, 1);
}

// ---------------------------------------------------------------- Scout (section 9.6)

// 9.6: discard 3 cards, take 1 Wild Location and 1 Wild Industry. RQ-026: the hand shrinks by
// exactly 1 and the wilds come from their own supplies, never the discard pile or deck.
#[test]
fn scout_gives_two_wilds_and_discards_three() {
    let (mut g, p) = game_with_hand(&[BIRM, DERBY, BIRM, IRON_CARD, COAL_CARD]);
    let deck_before = g.deck.clone();
    let discard_before = g.players[p as usize].discard.len();
    g.apply(&Move::Scout {
        cards: [BIRM, BIRM, IRON_CARD],
    })
    .unwrap();
    let ps = &g.players[p as usize];
    let mut hand = ps.hand.clone();
    hand.sort();
    let mut want = vec![DERBY, COAL_CARD, Card::WildLocation, Card::WildIndustry];
    want.sort();
    assert_eq!(hand, want);
    assert_eq!(ps.discard.len(), discard_before + 3);
    let mut new_discards = ps.discard[discard_before..].to_vec();
    new_discards.sort();
    assert_eq!(new_discards, vec![BIRM, BIRM, IRON_CARD]);
    assert_eq!(g.wild_location_supply, WILD_LOCATION_COUNT - 1);
    assert_eq!(g.wild_industry_supply, WILD_INDUSTRY_COUNT - 1);
    assert_eq!(g.deck, deck_before);
    assert_eq!(g.actions_left, 1);
}

// 9.6 and the FAQ can-i-scout-with-two-cards-left: Scout needs 3 cards in hand. The offered
// card triples are the distinct sorted multisets of the hand.
#[test]
fn scout_needs_three_cards_and_offers_distinct_multisets() {
    let (mut g, _) = game_with_hand(&[BIRM, DERBY]);
    assert!(!has_scout(&g));

    set_hand(&mut g, &[BIRM, BIRM, DERBY, DERBY]);
    let scouts: Vec<[Card; 3]> = g
        .legal_moves()
        .into_iter()
        .filter_map(|m| match m {
            Move::Scout { cards } => Some(cards),
            _ => None,
        })
        .collect();
    let mut bd = [BIRM, DERBY];
    bd.sort();
    let (a, b) = (bd[0], bd[1]);
    assert_eq!(scouts, vec![[a, a, b], [a, b, b]]);
}

// RQ-003: Scout is illegal while holding any Wild card, of either type, including using that
// wild as one of the three discards.
#[test]
fn scout_illegal_when_holding_wild() {
    for wild in [Card::WildLocation, Card::WildIndustry] {
        let (mut g, _) = game_with_hand(&[BIRM, DERBY, COAL_CARD, IRON_CARD, wild]);
        assert!(!has_scout(&g), "{wild:?}");
        let before = g.clone();
        assert_eq!(
            g.apply(&Move::Scout {
                cards: [BIRM, DERBY, COAL_CARD]
            }),
            Err(RuleError::IllegalMove)
        );
        assert_eq!(g, before);
    }
}

// Section 3.2 and section 6: a discarded Wild card returns to its own supply, not the discard
// pile, whatever the action.
#[test]
fn wild_discard_returns_to_supply() {
    let (mut g, p) = game_with_hand(&[Card::WildLocation, Card::WildIndustry]);
    g.wild_location_supply = 3;
    g.wild_industry_supply = 3;
    let discard_before = g.players[p as usize].discard.clone();
    let deck_before = g.deck.clone();

    g.apply(&Move::Pass {
        card: Card::WildLocation,
    })
    .unwrap();
    assert_eq!(g.wild_location_supply, 4);
    g.apply(&Move::Loan {
        card: Card::WildIndustry,
    })
    .unwrap();
    assert_eq!(g.wild_industry_supply, 4);
    assert_eq!(g.players[p as usize].discard, discard_before);
    // The turn is over and the emptied hand was refilled with exactly the top 8 deck cards, so
    // neither Wild stayed in (or returned to) the hand: both went back to their supplies.
    assert_eq!(
        g.players[p as usize].hand,
        deck_before[deck_before.len() - 8..]
    );
}

// ---------------------------------------------------------------- Develop (section 9.3)

// 9.3: remove the lowest-level tile of the chosen industry, consuming 1 iron. Iron from a works
// (any owner, no connection, section 8.2) is free; market iron costs money, which counts as
// spent (section 6).
#[test]
fn develop_removes_lowest_tile_and_costs_iron() {
    // From an opponent's iron works: no money changes hands.
    let (mut g, p) = game_with_hand(&[BIRM, DERBY]);
    let opp = (p + 1) % 4;
    let works = place(&mut g, "coalbrookdale", 0, opp, Iron, 1, 2);
    let iron = ResourcePlan {
        from_tiles: vec![(works, 1)],
        from_market: 0,
    };
    g.apply(&Move::Develop {
        card: BIRM,
        industry: Coal,
        iron,
    })
    .unwrap();
    assert_eq!(mat(&g, p, Coal, 1), 0);
    assert_eq!(mat(&g, p, Coal, 2), 2);
    assert_eq!(g.slots[works as usize].as_ref().unwrap().resources, 1);
    assert_eq!(g.players[p as usize].money, START_MONEY);
    assert_eq!(g.players[p as usize].spent, 0);
    assert_eq!(g.players[p as usize].discard.last(), Some(&BIRM));
    // The action continues: an optional second tile.
    assert_eq!(g.phase, Phase::DevelopMore);
    assert_eq!(g.actions_left, 2);

    // With no works on the board the iron is bought from the market.
    let (mut g, p) = game_with_hand(&[BIRM, DERBY]);
    let (cost, market_after) = iron_buy_cost(g.iron_market, 1);
    assert!(cost > 0);
    assert!(g.legal_moves().contains(&Move::Develop {
        card: DERBY,
        industry: Coal,
        iron: market_iron(1),
    }));
    g.apply(&Move::Develop {
        card: DERBY,
        industry: Coal,
        iron: market_iron(1),
    })
    .unwrap();
    assert_eq!(g.players[p as usize].money, START_MONEY - cost);
    assert_eq!(g.players[p as usize].spent, cost);
    assert_eq!(g.iron_market, market_after);
    assert_eq!(mat(&g, p, Coal, 1), 0);
}

// Section 6 / RQ-030: every action must be paid in full when taken. A player who cannot pay
// for market iron cannot develop.
#[test]
fn develop_requires_affording_market_iron() {
    let (mut g, p) = game_with_hand(&[BIRM]);
    g.players[p as usize].money = 0;
    assert!(develop_industries(&g).is_empty());
    let (cost, _) = iron_buy_cost(g.iron_market, 1);
    g.players[p as usize].money = cost;
    assert!(!develop_industries(&g).is_empty());
}

// 9.3: lightbulb tiles (Pottery I and III) cannot be developed, and the removed tile is always
// the lowest-level one, so a lightbulb at the bottom blocks Pottery entirely until it is built.
// Canal-only level 1 tiles and rail-only tiles may be developed (RQ-017).
#[test]
fn develop_lightbulb_pottery_not_offered() {
    let (mut g, p) = game_with_hand(&[BIRM]);
    // Fresh mat: Pottery I (lightbulb) is lowest.
    let inds = develop_industries(&g);
    assert!(!inds.contains(&Pottery));
    assert_eq!(inds.len(), 5);
    let before = g.clone();
    assert_eq!(
        g.apply(&Move::Develop {
            card: BIRM,
            industry: Pottery,
            iron: market_iron(1),
        }),
        Err(RuleError::IllegalMove)
    );
    assert_eq!(g, before);

    // Pottery I built: Pottery II is lowest and developable.
    g.players[p as usize].mat_remaining[Pottery as usize][0] = 0;
    assert!(develop_industries(&g).contains(&Pottery));
    g.apply(&Move::Develop {
        card: BIRM,
        industry: Pottery,
        iron: market_iron(1),
    })
    .unwrap();
    assert_eq!(mat(&g, p, Pottery, 2), 0);
    // Now Pottery III (lightbulb) is lowest: no second Pottery develop.
    assert!(!develop_industries(&g).contains(&Pottery));

    // An industry with no tiles left cannot be developed either.
    let (mut g, p) = game_with_hand(&[BIRM]);
    g.players[p as usize].mat_remaining[Iron as usize] = [0; 8];
    assert!(!develop_industries(&g).contains(&Iron));
}

// 9.3: Develop removes 1 or 2 tiles. In DevelopMore the only moves are a second tile
// (DevelopSecond, also 1 iron, any industry, the lowest tile at that moment) or EndAction.
#[test]
fn develop_second_is_optional_via_end_action() {
    // Declining the second tile.
    let (mut g, p) = game_with_hand(&[BIRM, DERBY]);
    g.apply(&Move::Develop {
        card: BIRM,
        industry: Coal,
        iron: market_iron(1),
    })
    .unwrap();
    let moves = g.legal_moves();
    assert!(moves.contains(&Move::EndAction));
    assert!(
        moves
            .iter()
            .all(|m| matches!(m, Move::DevelopSecond { .. } | Move::EndAction))
    );
    // The same industry again takes the next-lowest tile.
    assert!(moves.contains(&Move::DevelopSecond {
        industry: Coal,
        iron: market_iron(1),
    }));
    let money = g.players[p as usize].money;
    g.apply(&Move::EndAction).unwrap();
    assert_eq!(g.phase, Phase::ChooseAction);
    assert_eq!(g.actions_left, 1);
    assert_eq!(g.players[p as usize].money, money);
    assert_eq!(g.players[p as usize].hand, vec![DERBY]);

    // Taking the second tile, from the same industry: one card for both.
    let (mut g, p) = game_with_hand(&[BIRM, DERBY]);
    g.apply(&Move::Develop {
        card: BIRM,
        industry: Coal,
        iron: market_iron(1),
    })
    .unwrap();
    let spent_first = g.players[p as usize].spent;
    let (cost2, _) = iron_buy_cost(g.iron_market, 1);
    g.apply(&Move::DevelopSecond {
        industry: Coal,
        iron: market_iron(1),
    })
    .unwrap();
    assert_eq!(mat(&g, p, Coal, 1), 0);
    assert_eq!(mat(&g, p, Coal, 2), 1);
    assert_eq!(g.players[p as usize].spent, spent_first + cost2);
    assert_eq!(g.players[p as usize].hand, vec![DERBY]);
    assert_eq!(g.phase, Phase::ChooseAction);
    assert_eq!(g.actions_left, 1);
}

// ---------------------------------------------------------------- dispatch

// apply() accepts exactly the moves legal_moves() lists and leaves the state untouched on
// rejection.
#[test]
fn apply_rejects_move_not_in_legal_moves() {
    let (mut g, _) = game_with_hand(&[BIRM, DERBY]);
    let before = g.clone();
    let illegal = [
        // Card not in hand.
        Move::Loan { card: COAL_CARD },
        Move::Pass {
            card: Card::WildLocation,
        },
        // Follow-up moves outside their phase.
        Move::EndAction,
        Move::DevelopSecond {
            industry: Coal,
            iron: market_iron(1),
        },
        // Develop with a wrong iron plan (2 cubes for 1 tile).
        Move::Develop {
            card: BIRM,
            industry: Coal,
            iron: market_iron(2),
        },
    ];
    for m in &illegal {
        assert!(!g.legal_moves().contains(m), "{m:?}");
        assert_eq!(g.apply(m), Err(RuleError::IllegalMove), "{m:?}");
        assert_eq!(g, before, "{m:?}");
    }

    // A Scout triple the hand can pay for is rejected when it is not sorted.
    let (mut g3, _) = game_with_hand(&[BIRM, DERBY, BIRM]);
    let before3 = g3.clone();
    let sorted = Move::Scout {
        cards: [DERBY, BIRM, BIRM],
    };
    let unsorted = Move::Scout {
        cards: [BIRM, DERBY, BIRM],
    };
    assert!(g3.legal_moves().contains(&sorted));
    assert!(!g3.legal_moves().contains(&unsorted));
    assert_eq!(g3.apply(&unsorted), Err(RuleError::IllegalMove));
    assert_eq!(g3, before3);

    // In DevelopMore, ChooseAction moves are illegal.
    g.apply(&Move::Develop {
        card: BIRM,
        industry: Coal,
        iron: market_iron(1),
    })
    .unwrap();
    assert_eq!(
        g.apply(&Move::Pass { card: DERBY }),
        Err(RuleError::IllegalMove)
    );

    // Once the game is over nothing is legal.
    g.game_over = true;
    g.phase = Phase::GameOver;
    assert!(g.legal_moves().is_empty());
    assert_eq!(g.current_player(), None);
    assert_eq!(g.apply(&Move::EndAction), Err(RuleError::GameOver));
}

// Moves are deduplicated and deterministic across many random positions (duplicate cards in
// hand, Develop follow-ups, wild cards from Scouting).
#[test]
fn legal_moves_has_no_duplicates() {
    for seed in 0..20u64 {
        let mut g = brass_engine::state::new_game(seed);
        for step in 0..30usize {
            let moves = g.legal_moves();
            // Hands are refilled after each turn: no legal move before game end is a deadlock.
            assert!(!moves.is_empty(), "seed {seed} step {step}: no legal move");
            let set: HashSet<&Move> = moves.iter().collect();
            assert_eq!(set.len(), moves.len(), "seed {seed} step {step}");
            assert_eq!(moves, g.legal_moves(), "deterministic");
            let m = moves[(seed as usize * 7 + step * 13) % moves.len()].clone();
            g.apply(&m).unwrap();
        }
    }
}

// The player in Shortfall is the one to act (the phase itself is tested in tests/flow.rs).
#[test]
fn shortfall_phase_names_its_player() {
    let mut g = empty_game();
    g.phase = Phase::Shortfall { player: 2, owed: 3 };
    assert_eq!(g.current_player(), Some(2));
}

// Section 6 / RQ-030: actions resolve in sequence. A Develop that is unaffordable alone becomes
// legal in the second action once the first action (a Loan) has paid out.
#[test]
fn second_action_sees_first_actions_results() {
    let (mut g, p) = game_with_hand(&[BIRM, DERBY]);
    g.players[p as usize].money = 0;
    assert!(develop_industries(&g).is_empty());
    g.apply(&Move::Loan { card: BIRM }).unwrap();
    assert_eq!(me(&g), p);
    assert!(!develop_industries(&g).is_empty());
}
