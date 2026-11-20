//! Differential harness: plays random games on our engine and an oracle side by side and
//! reports every disagreement (plan Task 12; protocol in `oracles/protocol.md`).
//!
//! Usage: `brass-diff --oracle ts|yuiffy --games N --seed S [--adapter PATH] [--cases DIR]`.
//!
//! Game `i` is dealt by `init_spec(S + i)`. Before each action the coarse legal summaries are
//! compared; then a random complete action is chosen from our legal moves (a uniformly random
//! summary key, then a random move with that key, then random follow-ups), applied to both
//! engines, and the snapshots are compared (`game.rs`, `compare.rs`). Known differences
//! (`oracles/known_differences.toml`, one section per oracle) are stepped around by the
//! oracle's rule module (`limits.rs` selects it). An oracle that declares `sourcing_check` also
//! checks our explicit coal and iron sources against its canonical sourcing (`sourcing` in the
//! `apply` response). The first other mismatch of a game is written to
//! `<cases>/<seed>-<n>.json` (`case.rs`: deal, action history, both sides, a field-level diff)
//! and the run moves on to the next game. Exit status 1 if any mismatch was found.
//!
//! The brief's "with probability 0.1 pick a key the oracle offers that we don't" is subsumed:
//! any such key already fails the legal-summary comparison (and is written as a case), and it
//! cannot be played on our engine anyway.

mod case;
mod compare;
mod game;
mod known;
mod limits;
mod oracle;
mod ts_limits;
mod yuiffy_limits;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use brass_engine::state::init_spec;

use game::Outcome;
use known::Known;
use limits::Limits;
use oracle::Oracle;

struct Args {
    oracle: Limits,
    games: u64,
    seed: u64,
    adapter: Option<PathBuf>,
    cases: PathBuf,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        oracle: Limits::Ts,
        games: 200,
        seed: 1,
        adapter: None,
        cases: repo_root().join("oracles/cases"),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--oracle" => {
                a.oracle = match value("--oracle")?.as_str() {
                    "ts" => Limits::Ts,
                    "yuiffy" => Limits::Yuiffy,
                    o => return Err(format!("unsupported oracle {o:?} (`ts` or `yuiffy`)")),
                }
            }
            "--games" => {
                a.games = value("--games")?
                    .parse()
                    .map_err(|e| format!("--games: {e}"))?
            }
            "--seed" => {
                a.seed = value("--seed")?
                    .parse()
                    .map_err(|e| format!("--seed: {e}"))?
            }
            "--adapter" => a.adapter = Some(value("--adapter")?.into()),
            "--cases" => a.cases = value("--cases")?.into(),
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    Ok(a)
}

/// Start the oracle's adapter process (`--adapter` overrides its default path).
fn spawn_oracle(args: &Args) -> Result<Oracle, String> {
    match args.oracle {
        Limits::Ts => {
            let adapter = args
                .adapter
                .clone()
                .unwrap_or_else(|| repo_root().join("oracles/ts-adapter/adapter.mjs"));
            Oracle::spawn("node", &[adapter.display().to_string()])
        }
        // Built separately (AGPL, outside the workspace): `cargo build --release
        // --manifest-path oracles/yuiffy-adapter/Cargo.toml`.
        Limits::Yuiffy => {
            let adapter = args.adapter.clone().unwrap_or_else(|| {
                repo_root().join("oracles/yuiffy-adapter/target/release/yuiffy-adapter")
            });
            Oracle::spawn(&adapter.display().to_string(), &[])
        }
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "{e}\nusage: brass-diff --oracle ts|yuiffy --games N --seed S [--adapter PATH] [--cases DIR]"
            );
            return ExitCode::from(2);
        }
    };
    let known_path = repo_root().join("oracles/known_differences.toml");
    let mut known = match Known::load(
        &known_path,
        args.oracle.section(),
        args.oracle.implemented(),
        args.oracle.structural(),
    ) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let mut oracle = match spawn_oracle(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let start = Instant::now();
    let mut categories: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let (mut actions, mut complete, mut diverged) = (0usize, 0u64, 0u64);
    for i in 0..args.games {
        let seed = args.seed.wrapping_add(i);
        let spec = init_spec(seed);
        let mut history = Vec::new();
        let result = game::play_game(
            seed,
            &spec,
            &mut oracle,
            args.oracle,
            &mut known,
            &mut history,
        );
        actions += history.len();
        match result {
            Ok(Outcome::Complete) => complete += 1,
            Ok(Outcome::KnownDivergence) => diverged += 1,
            Err(m) => {
                case::write_case(&args.cases, seed, &spec, &history, &m);
                eprintln!(
                    "game {seed}: {} after {} actions",
                    m.category,
                    history.len()
                );
                for l in m.diff.iter().take(6) {
                    eprintln!("    {l}");
                }
                categories
                    .entry(m.category)
                    .or_default()
                    .push(format!("{seed}-{}", history.len()));
            }
        }
        if (i + 1) % 50 == 0 {
            eprintln!(
                "[{}/{}] {:.0} s, {} mismatching games",
                i + 1,
                args.games,
                start.elapsed().as_secs_f64(),
                i + 1 - complete - diverged
            );
        }
    }
    println!(
        "games: {} (seeds {}..{}), {actions} actions compared, {:.0} s",
        args.games,
        args.seed,
        args.seed.wrapping_add(args.games),
        start.elapsed().as_secs_f64()
    );
    println!(
        "  {complete} played to the end, {diverged} stopped at a known divergence, {} with a \
         mismatch",
        args.games - complete - diverged
    );
    println!("known differences ({}):", known_path.display());
    for l in known.report() {
        println!("{l}");
    }
    if categories.is_empty() {
        println!("untriaged mismatches: none");
        return ExitCode::SUCCESS;
    }
    println!(
        "untriaged mismatches by category (cases in {}):",
        args.cases.display()
    );
    for (c, games) in &categories {
        let shown: Vec<&String> = games.iter().take(5).collect();
        println!("  {:5}  {c}  e.g. {shown:?}", games.len());
    }
    ExitCode::FAILURE
}
