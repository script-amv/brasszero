//! Game state and seeded setup (docs/rules-reference.md section 5, 4 players).

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};

use crate::data::*;
use crate::moves::Phase;
use crate::record::InitSpec;

/// An industry tile on the board.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacedTile {
    pub owner: PlayerId,
    pub industry: Industry,
    pub level: u8,
    pub flipped: bool,
    /// Coal/iron/beer cubes still on the tile.
    pub resources: u8,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerState {
    pub hand: Vec<Card>,
    pub discard: Vec<Card>,
    /// Tiles left on the mat, indexed by `Industry as usize` then level index (level - 1).
    pub mat_remaining: [[u8; 8]; 6],
    pub money: u16,
    pub income_space: u8,
    pub vp: u16,
    /// Money spent this round (decides next round's turn order).
    pub spent: u16,
}

impl PlayerState {
    /// Lowest level of `ind` still on the mat, or `None` if none are left. Build and Develop
    /// always take this tile (rules-reference sections 9.1 and 9.3).
    pub fn lowest_level(&self, ind: Industry) -> Option<u8> {
        self.mat_remaining[ind as usize]
            .iter()
            .position(|&n| n > 0)
            .map(|i| i as u8 + 1)
    }
}

/// Full game state. All randomness is reproducible: `rng_seed` is the setup seed and
/// `reshuffles` counts deck reshuffles done so far; the era-2 reshuffle uses
/// [`GameState::reshuffle_rng`], a fresh `Pcg64` derived from both, so no RNG internals
/// need serializing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GameState {
    pub era: Era,
    pub round: u8,
    pub turn_order: [PlayerId; 4],
    pub turn_idx: u8,
    pub actions_left: u8,
    pub phase: Phase,
    /// One entry per `SLOTS` index.
    pub slots: Vec<Option<PlacedTile>>,
    /// One entry per `LINKS` index; the owner of the link tile.
    pub links: Vec<Option<PlayerId>>,
    /// Indexed by flattened merchant tile slot, in `MERCHANT_SPACES` order.
    pub merchant_tiles: Vec<MerchantTile>,
    /// Whether a beer barrel remains at each merchant tile slot.
    pub merchant_beer: Vec<bool>,
    pub coal_market: u8,
    pub iron_market: u8,
    /// Top of deck is the end of the vector.
    pub deck: Vec<Card>,
    pub wild_location_supply: u8,
    pub wild_industry_supply: u8,
    pub players: [PlayerState; 4],
    pub game_over: bool,
    pub rng_seed: u64,
    pub reshuffles: u8,
}

impl GameState {
    /// RNG for the next deck reshuffle. Deterministic in (`rng_seed`, `reshuffles`) and
    /// domain-separated from every game's setup stream (`Pcg64::seed_from_u64(seed)`): the
    /// seed is mixed with a constant tag through splitmix64, so game `s`'s reshuffle does not
    /// replay game `s + 1`'s setup.
    pub fn reshuffle_rng(&self) -> Pcg64 {
        const TAG: u64 = 0x5245_5348_5546_464C; // "RESHUFFL"
        let seed = splitmix64(splitmix64(self.rng_seed ^ TAG) ^ self.reshuffles as u64);
        Pcg64::seed_from_u64(seed)
    }
}

/// The splitmix64 finalizer: a bijective mix of a 64-bit value.
pub(crate) fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fresh_mat() -> [[u8; 8]; 6] {
    let mut mat = [[0u8; 8]; 6];
    for t in TILES {
        mat[t.industry as usize][t.level as usize - 1] = t.count;
    }
    mat
}

/// The random setup for `seed`: shuffle the deck, then the merchant tiles, then turn order.
pub fn init_spec(seed: u64) -> InitSpec {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut deck = deck_4p();
    deck.shuffle(&mut rng);
    let mut merchant_tiles = MERCHANT_TILES_4P.to_vec();
    merchant_tiles.shuffle(&mut rng);
    let mut turn_order: [PlayerId; 4] = [0, 1, 2, 3];
    turn_order.shuffle(&mut rng);

    let hands: [Vec<Card>; 4] = std::array::from_fn(|_| deck.split_off(deck.len() - HAND_SIZE));
    let discard_bottoms: [Card; 4] =
        std::array::from_fn(|_| deck.pop().expect("deck has 64 cards"));
    InitSpec {
        hands,
        discard_bottoms,
        deck,
        merchant_tiles,
        turn_order,
        rng_seed: seed,
    }
}

/// Build the opening state from an explicit deal.
pub fn from_init(spec: &InitSpec) -> GameState {
    let players: [PlayerState; 4] = std::array::from_fn(|i| PlayerState {
        hand: spec.hands[i].clone(),
        discard: vec![spec.discard_bottoms[i]],
        mat_remaining: fresh_mat(),
        money: START_MONEY,
        income_space: START_INCOME_SPACE,
        vp: 0,
        spent: 0,
    });
    GameState {
        era: Era::Canal,
        round: 1,
        turn_order: spec.turn_order,
        turn_idx: 0,
        // First round of the Canal era: 1 action each (rules-reference section 6).
        actions_left: 1,
        phase: Phase::ChooseAction,
        slots: vec![None; SLOTS.len()],
        links: vec![None; LINKS.len()],
        merchant_tiles: spec.merchant_tiles.clone(),
        merchant_beer: spec
            .merchant_tiles
            .iter()
            .map(|&t| t != MerchantTile::Blank)
            .collect(),
        coal_market: COAL_MARKET_START,
        iron_market: IRON_MARKET_START,
        deck: spec.deck.clone(),
        wild_location_supply: WILD_LOCATION_COUNT,
        wild_industry_supply: WILD_INDUSTRY_COUNT,
        players,
        game_over: false,
        rng_seed: spec.rng_seed,
        reshuffles: 0,
    }
}

/// A new game, fully determined by `seed`.
pub fn new_game(seed: u64) -> GameState {
    from_init(&init_spec(seed))
}

/// Check the state-local invariants of a game reachable by legal play. Used by the invariant
/// fuzzer (`engine/tests/invariants.rs`); not part of the stable API.
///
/// - Cards: the 64 deck cards are conserved across hands, discards and the deck; Wild cards
///   are only in hands or their supplies, with the supply sizes conserved (sections 3.1, 3.2).
/// - Tiles: per player and per (industry, level), mat tiles plus board tiles never exceed the
///   printed count (45 per player in total, section 4). Removed tiles (developed, overbuilt,
///   level-1 removal, shortfall) are not tracked in the state, so this is an upper bound.
/// - Links: at most `LINK_TILES` per player on the board, all of the current era's kind.
/// - Board: every tile sits in a slot that accepts its industry; a flipped tile holds no cubes.
/// - Markets: cubes within capacity; merchant beer only beside non-blank merchant tiles.
/// - Income space within the track. (Money is a `u16`, never negative by construction.)
/// - RQ-026: at the start of every 2-action turn every hand holds an even number of cards and
///   the acting player holds at least 2. (Hands of players who already moved in an era's
///   last round may be empty.)
#[doc(hidden)]
pub fn check_invariants(g: &GameState) -> Result<(), String> {
    // Cards.
    let mut cards: Vec<Card> = g.deck.clone();
    let (mut wild_loc, mut wild_ind) = (g.wild_location_supply, g.wild_industry_supply);
    for (p, ps) in g.players.iter().enumerate() {
        for &c in &ps.discard {
            if matches!(c, Card::WildLocation | Card::WildIndustry) {
                return Err(format!("player {p} has a Wild card in the discard pile"));
            }
        }
        cards.extend(&ps.discard);
        for &c in &ps.hand {
            match c {
                Card::WildLocation => wild_loc += 1,
                Card::WildIndustry => wild_ind += 1,
                _ => cards.push(c),
            }
        }
    }
    if g.deck
        .iter()
        .any(|c| matches!(c, Card::WildLocation | Card::WildIndustry))
    {
        return Err("Wild card in the deck".into());
    }
    let mut expected = deck_4p();
    expected.sort_unstable();
    cards.sort_unstable();
    if cards != expected {
        return Err(format!(
            "deck cards not conserved: {} cards in play, expected {}",
            cards.len(),
            expected.len()
        ));
    }
    if (wild_loc, wild_ind) != (WILD_LOCATION_COUNT, WILD_INDUSTRY_COUNT) {
        return Err(format!(
            "Wild cards not conserved: {wild_loc} location, {wild_ind} industry"
        ));
    }

    // Tiles.
    for (p, ps) in g.players.iter().enumerate() {
        for t in TILES {
            let on_board = g
                .slots
                .iter()
                .flatten()
                .filter(|b| b.owner as usize == p && b.industry == t.industry && b.level == t.level)
                .count();
            let on_mat = ps.mat_remaining[t.industry as usize][t.level as usize - 1] as usize;
            if on_board + on_mat > t.count as usize {
                return Err(format!(
                    "player {p}: {:?} {} has {on_mat} on the mat and {on_board} on the board, \
                     only {} exist",
                    t.industry, t.level, t.count
                ));
            }
        }
    }
    for (s, t) in g.slots.iter().enumerate() {
        let Some(t) = t else { continue };
        if !SLOTS[s].industries.contains(&t.industry) {
            return Err(format!(
                "slot {s} holds {:?}, which it does not accept",
                t.industry
            ));
        }
        if t.flipped && t.resources != 0 {
            return Err(format!(
                "flipped tile in slot {s} holds {} cubes",
                t.resources
            ));
        }
        if t.owner >= 4 {
            return Err(format!("slot {s} has owner {}", t.owner));
        }
    }

    // Links.
    for p in 0..4u8 {
        let n = g.links.iter().filter(|o| **o == Some(p)).count();
        if n > LINK_TILES {
            return Err(format!("player {p} has {n} links on the board"));
        }
    }
    for (i, o) in g.links.iter().enumerate() {
        let era_ok = match g.era {
            Era::Canal => LINKS[i].canal,
            Era::Rail => LINKS[i].rail,
        };
        if o.is_some() && !era_ok {
            return Err(format!("link {i} is not a {:?} Era link", g.era));
        }
    }

    // Markets and merchants.
    if g.coal_market as usize > COAL_MARKET_PRICES.len() {
        return Err(format!("coal market holds {}", g.coal_market));
    }
    if g.iron_market as usize > IRON_MARKET_PRICES.len() {
        return Err(format!("iron market holds {}", g.iron_market));
    }
    for (m, (&beer, &t)) in g.merchant_beer.iter().zip(&g.merchant_tiles).enumerate() {
        if beer && t == MerchantTile::Blank {
            return Err(format!("beer beside blank merchant tile slot {m}"));
        }
    }

    for (p, ps) in g.players.iter().enumerate() {
        if ps.income_space > MAX_INCOME_SPACE {
            return Err(format!("player {p} income space {}", ps.income_space));
        }
    }

    // RQ-026.
    if g.phase == Phase::ChooseAction && g.actions_left == 2 && !g.is_over() {
        let actor = g.turn_order[g.turn_idx as usize] as usize;
        for (p, ps) in g.players.iter().enumerate() {
            let n = ps.hand.len();
            if n % 2 != 0 || (p == actor && n < 2) {
                return Err(format!(
                    "RQ-026: player {p} holds {n} cards at the start of a 2-action turn \
                     (actor {actor})"
                ));
            }
        }
    }
    Ok(())
}
