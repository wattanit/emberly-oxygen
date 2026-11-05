//! Running the built binary, for the CLI's test files.

#![allow(dead_code)] // each test file uses its own subset

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub struct Run {
    pub code: i32,
    pub stdout: Vec<u8>,
    pub stderr: String,
    pub elapsed: Duration,
}

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn model_dir() -> String {
    let dir = root().join("models/smollm2-135m");
    assert!(dir.exists(), "{}: missing; run `cd oracle && uv run python download.py`", dir.display());
    dir.to_str().unwrap().to_owned()
}

/// Runs the binary with `args`, giving it `stdin` (empty when `None`).
pub fn run(args: &[&str], stdin: Option<&[u8]>) -> Run {
    let start = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_emberly-oxygen"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    input.write_all(stdin.unwrap_or_default()).unwrap();
    drop(input);
    let out = child.wait_with_output().unwrap();
    Run {
        code: out.status.code().expect("exited normally, not by a signal"),
        stdout: out.stdout,
        stderr: String::from_utf8(out.stderr).unwrap(),
        elapsed: start.elapsed(),
    }
}

pub fn generate(extra: &[&str]) -> Run {
    let model = model_dir();
    let mut args = vec!["generate", "--model", &model];
    args.extend(extra);
    run(&args, None)
}
