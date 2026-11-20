//! `apply()`: validate against `legal_moves()` (via the cheaper, equivalent
//! `legal_candidates`), then dispatch to the per-action modules.

use crate::actions;
use crate::flow;
use crate::moves::{Move, Phase, RuleError};
use crate::state::GameState;

impl GameState {
    /// Play `m`. Rejects (leaving the state unchanged) any move not in
    /// [`legal_moves`](Self::legal_moves), and every move once the game is over.
    pub fn apply(&mut self, m: &Move) -> Result<(), RuleError> {
        if self.is_over() {
            return Err(RuleError::GameOver);
        }
        // Equivalent to `legal_moves().contains(m)` (see `legal_candidates`).
        if !self.legal_candidates(m).contains(m) {
            return Err(RuleError::IllegalMove);
        }
        self.apply_unchecked(m);
        Ok(())
    }

    /// [`apply_unchecked`](Self::apply_unchecked) for the invariant fuzzer, which checks that
    /// it matches `apply` on legal moves. Not part of the stable API.
    #[doc(hidden)]
    pub fn apply_unchecked_for_testing(&mut self, m: &Move) {
        self.apply_unchecked(m);
    }

    /// Play `m` without checking legality. The caller guarantees `m` is in `legal_moves()`;
    /// otherwise this may panic or corrupt the state.
    pub(crate) fn apply_unchecked(&mut self, m: &Move) {
        let g = self;
        match m {
            Move::Develop {
                card,
                industry,
                iron,
            } => actions::develop::apply(g, *card, *industry, iron),
            Move::DevelopSecond { industry, iron } => {
                actions::develop::apply_second(g, *industry, iron)
            }
            Move::Loan { card } => actions::loan::apply(g, *card),
            Move::Scout { cards } => actions::scout::apply(g, cards),
            Move::Pass { card } => actions::pass::apply(g, *card),
            Move::EndAction => {
                g.phase = Phase::ChooseAction;
                flow::end_action(g);
            }
            Move::Build {
                card,
                industry,
                slot,
                coal,
                iron,
            } => actions::build::apply(g, *card, *industry, *slot, coal, iron),
            Move::Canal { card, link } => actions::network::apply_canal(g, *card, *link),
            Move::Rail { card, link, coal } => actions::network::apply_rail(g, *card, *link, coal),
            Move::SecondRail { link, coal, beer } => {
                actions::network::apply_second(g, *link, coal, *beer)
            }
            Move::Sell {
                card, slot, beer, ..
            } => actions::sell::apply(g, *card, *slot, beer),
            Move::SellAnother { slot, beer, .. } => actions::sell::apply_another(g, *slot, beer),
            Move::BonusDevelop { industry } => actions::sell::apply_bonus_develop(g, *industry),
            Move::ShortfallRemove { slot } => actions::shortfall::apply(g, *slot),
        }
    }
}
