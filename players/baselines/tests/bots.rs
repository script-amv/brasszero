// SPDX-License-Identifier: Apache-2.0
use brass_bots::heuristic::payments_left;
use brass_bots::{Agent, GreedyAgent, MAX_MOVES, RandomAgent, RefereeError, play_game};
use brass_engine::data::{Card, Era};
use brass_engine::flow;
use brass_engine::observe::Observation;
use brass_engine::state::{GameState, new_game};
use rand::{RngExt, SeedableRng};
use rand_pcg::Pcg64;

#[test]
fn random_agents_finish_games_reproducibly() {
    let mk = || -> [Box<dyn Agent>; 4] {
        std::array::from_fn(|i| Box::new(RandomAgent::new(i as u64)) as Box<dyn Agent>)
    };
    let a = play_game(&mut mk(), 0x9E37_79B9_7F4A_7C15).unwrap();
    let b = play_game(&mut mk(), 0x9E37_79B9_7F4A_7C15).unwrap();
    assert_eq!(a, b);
    assert!(a.moves.len() > 100 && a.moves.len() < MAX_MOVES);
    // Replaying the recorded moves reproduces the ranking.
    let mut g = new_game(a.seed);
    for m in &a.moves {
        g.apply(m).unwrap();
    }
    assert!(g.is_over());
    assert_eq!(flow::ranking(&g), a.ranking);
    for i in 0..4 {
        assert_eq!(g.players[i].vp, a.vp[i]);
    }
}

struct Bad;
impl Agent for Bad {
    fn name(&self) -> &str {
        "bad"
    }
    fn act(&mut self, o: &Observation) -> usize {
        o.legal_moves.len()
    }
}

#[test]
fn bad_index_is_an_error() {
    let mut agents: [Box<dyn Agent>; 4] = std::array::from_fn(|_| Box::new(Bad) as Box<dyn Agent>);
    assert!(matches!(
        play_game(&mut agents, 1),
        Err(RefereeError::BadIndex { .. })
    ));
}

/// Checks every call is for its own seat with moves available, then plays randomly.
struct Recorder {
    seat: u8,
    calls: usize,
    inner: RandomAgent,
}
impl Agent for Recorder {
    fn name(&self) -> &str {
        "recorder"
    }
    fn act(&mut self, o: &Observation) -> usize {
        assert_eq!(o.me, self.seat);
        assert!(!o.legal_moves.is_empty());
        self.calls += 1;
        self.inner.act(o)
    }
}

/// Agents are called only with their own observation, only on their turn.
#[test]
fn agents_only_see_their_own_observation() {
    let mut agents: [Box<dyn Agent>; 4] = std::array::from_fn(|i| {
        Box::new(Recorder {
            seat: i as u8,
            calls: 0,
            inner: RandomAgent::new(i as u64),
        }) as Box<dyn Agent>
    });
    let r = play_game(&mut agents, 7).unwrap();
    assert!(r.moves.len() > 100);
}

// ---------------------------------------------------------------- greedy

#[test]
fn payments_left_counts_remaining_income() {
    let mut g = new_game(0);
    assert_eq!(payments_left(&g), 15);
    g.era = Era::Rail;
    g.round = 1;
    assert_eq!(payments_left(&g), 7);
    g.round = 8;
    assert_eq!(payments_left(&g), 0);
}

/// `new_game(seed)` after `n` uniformly random legal moves.
fn random_play(seed: u64, n: usize) -> GameState {
    let mut g = new_game(seed);
    let mut rng = Pcg64::seed_from_u64(seed ^ 0x5EED);
    for _ in 0..n {
        let legal = g.legal_moves();
        g.apply(&legal[rng.random_range(0..legal.len())]).unwrap();
    }
    g
}

/// For seeds 0..4: `(g, h)` where `g` is 20 random moves in (40 often empties the Canal
/// deck, leaving no top card to swap) and `h` is `g` with a non-Wild
/// hand card of an opponent of the player to move swapped with the top deck card.
fn hidden_variants() -> Vec<(GameState, GameState)> {
    (0..4)
        .map(|seed| {
            let g = random_play(seed, 20);
            let p = g.current_player().unwrap();
            let q = ((p + 1) % 4) as usize;
            let top = *g.deck.last().expect("deck not empty after 20 moves");
            let i = g.players[q]
                .hand
                .iter()
                .position(|&c| !matches!(c, Card::WildLocation | Card::WildIndustry) && c != top)
                .expect("opponent holds a non-Wild card unlike the top deck card");
            let mut h = g.clone();
            let n = h.deck.len();
            std::mem::swap(&mut h.players[q].hand[i], &mut h.deck[n - 1]);
            assert_ne!(g, h);
            (g, h)
        })
        .collect()
}

#[test]
fn greedy_ignores_hidden_information() {
    for (g, h) in hidden_variants() {
        let p = g.current_player().unwrap();
        assert_eq!(
            GreedyAgent::new(7).scores(&g.observe(p)),
            GreedyAgent::new(7).scores(&h.observe(p))
        );
    }
}

#[test]
fn greedy_scores_one_per_legal_move() {
    for seed in 0..3 {
        let g = random_play(seed, 25);
        let obs = g.observe(g.current_player().unwrap());
        let scores = GreedyAgent::new(seed).scores(&obs);
        assert_eq!(scores.len(), obs.legal_moves.len());
        assert!(scores.iter().all(|s| s.is_finite()));
    }
}

#[test]
fn greedy_finishes_a_game() {
    let mut agents: [Box<dyn Agent>; 4] = [
        Box::new(GreedyAgent::new(1)),
        Box::new(RandomAgent::new(2)),
        Box::new(RandomAgent::new(3)),
        Box::new(RandomAgent::new(4)),
    ];
    assert_eq!(agents[0].name(), "greedy");
    let r = play_game(&mut agents, 0x0123_4567_89AB_CDEF).unwrap();
    assert!(r.moves.len() > 100);
}

/// SplitMix64 finaliser (seeds for evaluation deals).
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[test]
#[ignore] // release only: cargo test --release -p brass_bots -- --ignored
fn greedy_beats_random() {
    // 50 deals from splitmix64 of 0..50, 4 seat rotations each (200 games):
    // greedy strictly first in >= 75% of games.
    let mut wins = 0;
    let mut games = 0;
    for deal in 0..50u64 {
        let seed = splitmix64(deal);
        for seat in 0..4usize {
            let mut agents: [Box<dyn Agent>; 4] = std::array::from_fn(|i| {
                let s = splitmix64(seed ^ (i as u64 + 1));
                if i == seat {
                    Box::new(GreedyAgent::new(s)) as Box<dyn Agent>
                } else {
                    Box::new(RandomAgent::new(s)) as Box<dyn Agent>
                }
            });
            let r = play_game(&mut agents, seed).unwrap();
            games += 1;
            if r.ranking[seat] == 0 && r.ranking.iter().filter(|&&x| x == 0).count() == 1 {
                wins += 1;
            }
        }
    }
    let rate = wins as f64 / games as f64;
    println!(
        "greedy strictly first in {wins}/{games} games ({:.1}%)",
        rate * 100.0
    );
    assert!(rate >= 0.75, "greedy win rate {rate:.3} < 0.75");
}
