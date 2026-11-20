//! A JSON-lines oracle process (`oracles/protocol.md`): one request per line on its stdin,
//! one response per line on its stdout.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

pub struct Oracle {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Oracle {
    /// Start `program args..` (its stderr is passed through).
    pub fn spawn(program: &str, args: &[String]) -> Result<Self, String> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("cannot start {program}: {e}"))?;
        let stdin = BufWriter::new(child.stdin.take().expect("piped stdin"));
        let stdout = BufReader::new(child.stdout.take().expect("piped stdout"));
        Ok(Oracle {
            child,
            stdin,
            stdout,
        })
    }

    /// Send one request; `Ok(response)` for `{"ok":true,..}`, `Err(error)` for
    /// `{"ok":false,"error":..}`. Panics if the process dies or speaks garbage: the harness
    /// cannot continue without it.
    pub fn request(&mut self, req: &Value) -> Result<Value, String> {
        let line = serde_json::to_string(req).expect("request serializes");
        writeln!(self.stdin, "{line}")
            .and_then(|_| self.stdin.flush())
            .expect("oracle stdin closed");
        let mut resp = String::new();
        let n = self.stdout.read_line(&mut resp).expect("oracle stdout");
        assert!(n > 0, "oracle exited (request was {line:.200})");
        let v: Value = serde_json::from_str(&resp)
            .unwrap_or_else(|e| panic!("oracle response is not JSON ({e}): {resp:.200}"));
        if v["ok"] == json!(true) {
            Ok(v)
        } else {
            Err(v["error"]
                .as_str()
                .unwrap_or("(no error message)")
                .to_string())
        }
    }
}

impl Drop for Oracle {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
