//! Mismatch case files: `<cases>/<seed>-<n>.json` (`oracles/protocol.md`).

use std::path::Path;

use brass_engine::record::{ActionRecord, InitSpec};
use serde_json::json;

use crate::compare::Mismatch;

/// Write the first mismatch of game `seed`: the deal, the `n` actions played before it, the
/// pending action, both sides and the field-level diff.
pub fn write_case(dir: &Path, seed: u64, spec: &InitSpec, history: &[ActionRecord], m: &Mismatch) {
    let n = history.len();
    let path = dir.join(format!("{seed}-{n}.json"));
    let case = json!({
        "seed": seed,
        "action_index": n,
        "category": m.category,
        "spec": spec,
        "history": history,
        "pending": m.pending,
        "ours": m.ours,
        "oracle": m.theirs,
        "diff": m.diff,
    });
    std::fs::create_dir_all(dir).expect("create cases dir");
    std::fs::write(&path, serde_json::to_string_pretty(&case).unwrap()).expect("write case");
}
