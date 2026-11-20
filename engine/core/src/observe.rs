//! The agent-facing information-hiding API: [`GameState::observe`] shows one player exactly
//! what they may know, and [`Observation::determinize`] samples a full state consistent with
//! it (for search such as ISMCTS).
//!
//! What is hidden (rules-reference "Spec, Information", tournament setting): opponents' hand
//! cards (except Wilds, which came from the public supply), the draw deck, and each player's
//! face-down bottom discard, which nobody may look at, its owner included. Money, discards,
//! tiles, hand sizes and deck size are public.

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};

use crate::data::{Card, Era, PlayerId, deck_4p};
use crate::moves::{Move, Phase};
use crate::record::{PublicPlayer, PublicSnapshot};
use crate::state::{GameState, PlayerState, splitmix64};

/// What player `me` knows about the game.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub me: PlayerId,
    pub my_hand: Vec<Card>,
    pub public: PublicSnapshot,
    pub hand_sizes: [u8; 4],
    pub deck_size: u8,
    pub phase: Phase,
    /// Legal moves, only when `me` is the player to move (`current_player()`, which is the
    /// shortfall player during `Phase::Shortfall`); empty otherwise.
    pub legal_moves: Vec<Move>,
}

impl GameState {
    /// Whether each player's `discard[0]` is a face-down card. It is during the Canal Era;
    /// the Canal-to-Rail transition shuffles every discard into the deck, so the Rail Era has
    /// none (`flow::start_rail_era`).
    fn has_face_down_discards(&self) -> bool {
        self.era == Era::Canal
    }

    /// All public state (see [`PublicSnapshot`]).
    pub fn public_snapshot(&self) -> PublicSnapshot {
        let face_down = self.has_face_down_discards();
        PublicSnapshot {
            era: self.era,
            round: self.round,
            turn_order: self.turn_order,
            turn_idx: self.turn_idx,
            actions_left: self.actions_left,
            phase: self.phase,
            slots: self.slots.clone(),
            links: self.links.clone(),
            merchant_tiles: self.merchant_tiles.clone(),
            merchant_beer: self.merchant_beer.clone(),
            coal_market: self.coal_market,
            iron_market: self.iron_market,
            deck_size: self.deck.len() as u8,
            wild_location_supply: self.wild_location_supply,
            wild_industry_supply: self.wild_industry_supply,
            players: std::array::from_fn(|i| {
                let ps = &self.players[i];
                PublicPlayer {
                    hand_size: ps.hand.len() as u8,
                    wild_location_in_hand: ps
                        .hand
                        .iter()
                        .filter(|&&c| c == Card::WildLocation)
                        .count() as u8,
                    wild_industry_in_hand: ps
                        .hand
                        .iter()
                        .filter(|&&c| c == Card::WildIndustry)
                        .count() as u8,
                    discard: ps.discard[usize::from(face_down).min(ps.discard.len())..].to_vec(),
                    has_face_down_discard: face_down && !ps.discard.is_empty(),
                    mat_remaining: ps.mat_remaining,
                    money: ps.money,
                    income_space: ps.income_space,
                    vp: ps.vp,
                    spent: ps.spent,
                }
            }),
            game_over: self.game_over,
        }
    }

    /// Player `p`'s view of the game.
    pub fn observe(&self, p: PlayerId) -> Observation {
        let mut o = self.observe_without_moves(p);
        if self.current_player() == Some(p) {
            o.legal_moves = self.legal_moves();
        }
        o
    }

    /// [`GameState::observe`] with `legal_moves` left empty.
    fn observe_without_moves(&self, p: PlayerId) -> Observation {
        Observation {
            me: p,
            my_hand: self.players[p as usize].hand.clone(),
            public: self.public_snapshot(),
            hand_sizes: std::array::from_fn(|i| self.players[i].hand.len() as u8),
            deck_size: self.deck.len() as u8,
            phase: self.phase,
            legal_moves: Vec::new(),
        }
    }

    /// `self.observe(p).determinize(seed)`: see [`Observation::determinize`].
    pub fn determinize(&self, p: PlayerId, seed: u64) -> GameState {
        self.observe_without_moves(p).determinize(seed)
    }
}

impl Observation {
    /// A full state consistent with this observation, sampled from `seed`; deterministic in
    /// (observation, `seed`). It reads nothing but the observation, so it cannot leak the
    /// true hidden cards.
    ///
    /// Everything public and `me`'s hand are kept. The unseen cards are the full deck minus
    /// `me`'s non-Wild hand cards and every visible discard: the opponents' non-Wild hand
    /// cards, the draw deck and, in the Canal Era, all four face-down discards (`me`'s own
    /// included, since not even `me` has seen it). They are shuffled and dealt: each
    /// face-down position in seat order, then each opponent in seat order (their public
    /// Wilds first, then hidden cards up to their hand size), then the rest becomes the deck.
    /// The card multiset is conserved, so `check_invariants` holds.
    ///
    /// Future randomness is cut off from the true game: `rng_seed` (which seeds the era-2
    /// reshuffle) is derived from `seed` and `me`, so an agent cannot recover the true future
    /// deck order from it. The result is therefore not replayable against the original game's
    /// setup seed.
    pub fn determinize(&self, seed: u64) -> GameState {
        const SHUFFLE_TAG: u64 = 0x4445_5445_524D_494E; // "DETERMIN"
        const RESEED_TAG: u64 = 0x4445_5445_5253_4544; // "DETERSED"
        let me = self.me;
        let mix = |tag: u64| splitmix64(splitmix64(seed ^ tag) ^ me as u64);
        let mut rng = Pcg64::seed_from_u64(mix(SHUFFLE_TAG));
        let public = &self.public;
        let is_wild = |c: &Card| matches!(c, Card::WildLocation | Card::WildIndustry);

        // The unseen pool: every deck card not in my hand or a visible discard.
        let mut pool = deck_4p();
        let seen = self
            .my_hand
            .iter()
            .filter(|c| !is_wild(c))
            .chain(public.players.iter().flat_map(|pp| &pp.discard));
        for c in seen {
            let i = pool
                .iter()
                .position(|x| x == c)
                .expect("a seen card is in the deck list");
            pool.swap_remove(i);
        }
        pool.sort_unstable();
        let face_down = public
            .players
            .iter()
            .filter(|pp| pp.has_face_down_discard)
            .count();
        let hidden_hand_slots: usize = public
            .players
            .iter()
            .enumerate()
            .filter(|&(q, _)| q != me as usize)
            .map(|(_, pp)| {
                (pp.hand_size - pp.wild_location_in_hand - pp.wild_industry_in_hand) as usize
            })
            .sum();
        debug_assert_eq!(
            pool.len(),
            public.deck_size as usize + face_down + hidden_hand_slots,
            "the unseen pool matches the hidden positions"
        );
        pool.shuffle(&mut rng);

        let discards: [Vec<Card>; 4] = std::array::from_fn(|q| {
            let pp = &public.players[q];
            let mut d = Vec::with_capacity(pp.discard.len() + 1);
            if pp.has_face_down_discard {
                d.push(pool.pop().expect("pool holds a card per face-down discard"));
            }
            d.extend(&pp.discard);
            d
        });
        let hands: [Vec<Card>; 4] = std::array::from_fn(|q| {
            if q == me as usize {
                return self.my_hand.clone();
            }
            let pp = &public.players[q];
            let mut h = vec![Card::WildLocation; pp.wild_location_in_hand as usize];
            h.extend(std::iter::repeat_n(
                Card::WildIndustry,
                pp.wild_industry_in_hand as usize,
            ));
            while h.len() < pp.hand_size as usize {
                h.push(pool.pop().expect("pool holds a card per hidden hand slot"));
            }
            h
        });
        let mut hands = hands.into_iter();
        let mut discards = discards.into_iter();
        let players: [PlayerState; 4] = std::array::from_fn(|q| {
            let pp = &public.players[q];
            PlayerState {
                hand: hands.next().expect("four hands"),
                discard: discards.next().expect("four discards"),
                mat_remaining: pp.mat_remaining,
                money: pp.money,
                income_space: pp.income_space,
                vp: pp.vp,
                spent: pp.spent,
            }
        });

        GameState {
            era: public.era,
            round: public.round,
            turn_order: public.turn_order,
            turn_idx: public.turn_idx,
            actions_left: public.actions_left,
            phase: public.phase,
            slots: public.slots.clone(),
            links: public.links.clone(),
            merchant_tiles: public.merchant_tiles.clone(),
            merchant_beer: public.merchant_beer.clone(),
            coal_market: public.coal_market,
            iron_market: public.iron_market,
            deck: pool,
            wild_location_supply: public.wild_location_supply,
            wild_industry_supply: public.wild_industry_supply,
            players,
            game_over: public.game_over,
            rng_seed: mix(RESEED_TAG),
            // The only reshuffle is at the start of the Rail Era (`flow::start_rail_era`).
            reshuffles: (public.era == Era::Rail) as u8,
        }
    }
}
