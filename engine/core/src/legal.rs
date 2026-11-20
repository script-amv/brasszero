//! `legal_moves()`: dispatch by phase to the per-action modules in `actions/`.
//!
//! Each action module contributes its own moves; adding an action (Tasks 7–8) means adding
//! one line to the matching phase arm, never touching another action's code.

use crate::actions;
use crate::moves::{Move, Phase};
use crate::state::GameState;

impl GameState {
    /// Whether the game has ended (no move is legal).
    pub fn is_over(&self) -> bool {
        self.game_over || self.phase == Phase::GameOver
    }

    /// The player who must move now, or `None` once the game is over.
    pub fn current_player(&self) -> Option<crate::data::PlayerId> {
        match self.phase {
            _ if self.is_over() => None,
            Phase::Shortfall { player, .. } => Some(player),
            _ => Some(self.turn_order[self.turn_idx as usize]),
        }
    }

    /// The legal moves that could equal `m`: a subset of [`legal_moves`](Self::legal_moves)
    /// holding every legal move of `m`'s kind (and, for Build, Rail and Sell, of `m`'s
    /// industry and location, link, or slot). So `legal_candidates(m).contains(m)` exactly when
    /// `legal_moves().contains(m)`, at a fraction of the cost. Each arm mirrors the matching
    /// phase arm of `legal_moves`; a move kind not generated in the current phase gets none.
    pub(crate) fn legal_candidates(&self, m: &Move) -> Vec<Move> {
        if self.is_over() {
            return Vec::new();
        }
        let g = self;
        let cards = || actions::distinct_cards(g, actions::actor(g));
        match (self.phase, m) {
            (Phase::ChooseAction, Move::Build { industry, slot, .. }) => {
                match crate::data::SLOTS.get(*slot as usize) {
                    Some(s) => actions::build::legal_at(g, &cards(), *industry, s.loc),
                    None => Vec::new(),
                }
            }
            (Phase::ChooseAction, Move::Canal { link, .. } | Move::Rail { link, .. }) => {
                actions::network::legal_for(g, &cards(), *link)
            }
            (Phase::ChooseAction, Move::Develop { .. }) => actions::develop::legal(g, &cards()),
            (Phase::ChooseAction, Move::Sell { slot, .. }) => {
                actions::sell::legal_for(g, &cards(), *slot)
            }
            (Phase::ChooseAction, Move::Loan { .. }) => actions::loan::legal(g, &cards()),
            (Phase::ChooseAction, Move::Scout { .. }) => actions::scout::legal(g),
            (Phase::ChooseAction, Move::Pass { .. }) => actions::pass::legal(&cards()),
            (Phase::SecondRail, Move::SecondRail { .. }) => actions::network::legal_second(g),
            (Phase::DevelopMore, Move::DevelopSecond { .. }) => actions::develop::legal_second(g),
            (Phase::SellMore, Move::SellAnother { .. }) => actions::sell::legal_another(g),
            (Phase::SecondRail | Phase::DevelopMore | Phase::SellMore, Move::EndAction) => {
                vec![Move::EndAction]
            }
            (Phase::BonusDevelop, Move::BonusDevelop { .. }) => {
                actions::sell::legal_bonus_develop(g)
            }
            (Phase::Shortfall { .. }, Move::ShortfallRemove { .. }) => actions::shortfall::legal(g),
            _ => Vec::new(),
        }
    }

    /// Every legal move for [`current_player`](Self::current_player): deduplicated, in a
    /// deterministic order (action by action, then by card value and the action's own
    /// parameters). Empty once the game is over.
    pub fn legal_moves(&self) -> Vec<Move> {
        if self.is_over() {
            return Vec::new();
        }
        let g = self;
        let mut out = Vec::new();
        match self.phase {
            Phase::ChooseAction => {
                // Every card-discarding action generates per distinct card: sort the hand once.
                let cards = actions::distinct_cards(g, actions::actor(g));
                let parts = [
                    actions::build::legal(g, &cards),
                    actions::network::legal(g, &cards),
                    actions::develop::legal(g, &cards),
                    actions::sell::legal(g, &cards),
                    actions::loan::legal(g, &cards),
                    actions::scout::legal(g),
                    actions::pass::legal(&cards),
                ];
                // One allocation for the whole list instead of growing it per action.
                out.reserve_exact(parts.iter().map(Vec::len).sum());
                for part in parts {
                    out.extend(part);
                }
            }
            Phase::SecondRail => {
                out.extend(actions::network::legal_second(g));
                out.push(Move::EndAction);
            }
            Phase::DevelopMore => {
                out.extend(actions::develop::legal_second(g));
                out.push(Move::EndAction);
            }
            Phase::SellMore => {
                out.extend(actions::sell::legal_another(g));
                out.push(Move::EndAction);
            }
            Phase::BonusDevelop => {
                out.extend(actions::sell::legal_bonus_develop(g));
            }
            Phase::Shortfall { .. } => {
                out.extend(actions::shortfall::legal(g));
            }
            Phase::GameOver => unreachable!("handled by is_over"),
        }
        debug_assert!(
            {
                let mut s = out.clone();
                s.sort();
                s.dedup();
                s.len() == out.len()
            },
            "legal_moves produced duplicates"
        );
        out
    }
}
