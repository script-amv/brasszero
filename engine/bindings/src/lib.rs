//! Python bindings for `brass_engine` (module `brass`).
mod observation_transport;
mod static_data;

use brass_bots::{Agent, GreedyAgent, RandomAgent};
use brass_engine::describe::{describe_move, render_observation};
use brass_engine::flow;
use brass_engine::moves::Move;
use brass_engine::observe::Observation as RustObservation;
use brass_engine::state::{GameState, check_invariants, new_game};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

fn json_err(e: serde_json::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn check_player(p: u8) -> PyResult<()> {
    if p < 4 {
        Ok(())
    } else {
        Err(PyValueError::new_err("player must be 0..3"))
    }
}

/// A 4-player Brass: Birmingham game.
#[pyclass(module = "brass", skip_from_py_object)]
#[derive(Clone)]
struct Game {
    state: GameState,
}

#[pymethods]
impl Game {
    #[new]
    fn new(seed: u64) -> Self {
        Game {
            state: new_game(seed),
        }
    }

    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Game> {
        let state: GameState = serde_json::from_str(s).map_err(json_err)?;
        // Corrupt state must be a ValueError, never a later panic.
        let checked = std::panic::catch_unwind(|| check_invariants(&state));
        match checked {
            Ok(Ok(())) => Ok(Game { state }),
            Ok(Err(e)) => Err(PyValueError::new_err(format!("invalid game state: {e}"))),
            Err(_) => Err(PyValueError::new_err("invalid game state")),
        }
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.state).map_err(json_err)
    }

    /// Seat to move, or None when the game is over.
    fn current_player(&self) -> Option<u8> {
        self.state.current_player()
    }

    /// JSON string of each legal `Move`.
    fn legal_moves(&self) -> PyResult<Vec<String>> {
        self.state
            .legal_moves()
            .iter()
            .map(|m| serde_json::to_string(m).map_err(json_err))
            .collect()
    }

    /// Apply a move given as JSON. Raises ValueError if malformed or illegal.
    fn apply(&mut self, move_json: &str) -> PyResult<()> {
        let m: Move = serde_json::from_str(move_json).map_err(json_err)?;
        self.state
            .apply(&m)
            .map_err(|e| PyValueError::new_err(format!("illegal move: {e:?}")))
    }

    /// JSON `Observation` for player `p` (no opponent hands, no deck order).
    fn observe(&self, p: u8) -> PyResult<String> {
        check_player(p)?;
        serde_json::to_string(&self.state.observe(p)).map_err(json_err)
    }

    /// A full game state consistent with player `p`'s observation, sampled from `seed`.
    fn determinize(&self, p: u8, seed: u64) -> PyResult<Game> {
        check_player(p)?;
        Ok(Game {
            state: self.state.determinize(p, seed),
        })
    }

    fn is_over(&self) -> bool {
        self.state.is_over()
    }

    /// Per-seat rank list (index = player, 0 = winner). Ties (VP, then income level, then
    /// money) share a rank and the next rank skips, e.g. `[0, 0, 2, 3]`.
    fn ranking(&self) -> Vec<u8> {
        flow::ranking(&self.state).to_vec()
    }

    /// Player `p`'s `Observation` (the only thing an agent may see).
    fn observation(&self, p: u8) -> PyResult<Observation> {
        check_player(p)?;
        Ok(Observation {
            obs: self.state.observe(p),
        })
    }

    /// Apply `legal_moves()[i]`. Raises ValueError if `i` is out of range or the game is over.
    fn apply_index(&mut self, i: usize) -> PyResult<()> {
        if self.state.is_over() {
            return Err(PyValueError::new_err("game is over"));
        }
        let moves = self.state.legal_moves();
        let m = moves.get(i).ok_or_else(|| {
            PyValueError::new_err(format!("move index {i} out of range ({})", moves.len()))
        })?;
        self.state
            .apply(m)
            .map_err(|e| PyValueError::new_err(format!("illegal move: {e:?}")))
    }

    /// Victory points per seat.
    fn vp(&self) -> Vec<u16> {
        self.state.players.iter().map(|p| p.vp).collect()
    }

    fn clone(&self) -> Game {
        Clone::clone(self)
    }

    fn __copy__(&self) -> Game {
        Clone::clone(self)
    }
}

/// What one player can see. Holds no `GameState`.
#[pyclass(module = "brass", frozen, skip_from_py_object)]
struct Observation {
    obs: RustObservation,
}

#[pymethods]
impl Observation {
    /// Strict sanitized transport bridge. The real referee remains the legality authority.
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Observation> {
        observation_transport::parse(s)
            .map(|obs| Observation { obs })
            .map_err(PyValueError::new_err)
    }

    /// Exactly the public snapshot, without private cards or setup/RNG state.
    fn public_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.obs.public).map_err(json_err)
    }

    #[getter]
    fn me(&self) -> u8 {
        self.obs.me
    }

    /// Move JSON, same order as `Game.legal_moves()` for this player.
    fn legal_moves(&self) -> PyResult<Vec<String>> {
        self.obs
            .legal_moves
            .iter()
            .map(|m| serde_json::to_string(m).map_err(json_err))
            .collect()
    }

    fn num_legal(&self) -> usize {
        self.obs.legal_moves.len()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.obs).map_err(json_err)
    }

    fn determinize(&self, seed: u64) -> Game {
        Game {
            state: self.obs.determinize(seed),
        }
    }

    /// Multi-line text of everything this player can see.
    fn render(&self) -> String {
        render_observation(&self.obs)
    }

    /// One-line description of legal move `i`. Raises IndexError if out of range.
    fn describe_move(&self, i: usize) -> PyResult<String> {
        self.obs
            .legal_moves
            .get(i)
            .map(describe_move)
            .ok_or_else(|| PyIndexError::new_err(format!("move index {i} out of range")))
    }
}

fn act_checked(agent: &mut dyn Agent, obs: &Observation) -> PyResult<usize> {
    if obs.obs.legal_moves.is_empty() {
        return Err(PyValueError::new_err(
            "no legal moves (game over or not this player's turn)",
        ));
    }
    Ok(agent.act(&obs.obs))
}

/// Uniformly random baseline bot.
#[pyclass(module = "brass")]
struct RandomBot {
    agent: RandomAgent,
}

#[pymethods]
impl RandomBot {
    #[new]
    fn new(seed: u64) -> Self {
        RandomBot {
            agent: RandomAgent::new(seed),
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.agent.name().to_string()
    }

    fn act(&mut self, obs: &Observation) -> PyResult<usize> {
        act_checked(&mut self.agent, obs)
    }
}

/// One-ply greedy heuristic baseline bot.
#[pyclass(module = "brass")]
struct GreedyBot {
    agent: GreedyAgent,
}

#[pymethods]
impl GreedyBot {
    #[new]
    fn new(seed: u64) -> Self {
        GreedyBot {
            agent: GreedyAgent::new(seed),
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.agent.name().to_string()
    }

    fn act(&mut self, obs: &Observation) -> PyResult<usize> {
        act_checked(&mut self.agent, obs)
    }

    /// Score per legal move (for the GUI's "AI thinking" panel).
    fn scores(&mut self, obs: &Observation) -> Vec<f64> {
        self.agent.scores(&obs.obs)
    }
}

#[pyfunction]
fn component_data() -> String {
    static_data::render()
}

#[pymodule]
fn brass(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(component_data, m)?)?;
    m.add_class::<Game>()?;
    m.add_class::<Observation>()?;
    m.add_class::<RandomBot>()?;
    m.add_class::<GreedyBot>()?;
    Ok(())
}
