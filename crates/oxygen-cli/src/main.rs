//! The `emberly-oxygen` command (Spec §9). This file runs the command and maps
//! failures to exit codes (Spec §14): 1 for an error, with the message on
//! standard error; 2 for a usage error, which `clap` reports and exits with
//! itself. Text already streamed stays printed and the error follows it.

mod cli;

use std::io::Write;

use clap::Parser;

fn main() {
    let args = cli::Cli::parse();
    if let Err(message) = cli::run(args) {
        fail(&message);
    }
}

/// Reports an error on standard error and exits with code 1.
pub(crate) fn fail(message: &str) -> ! {
    // Nothing more can be done if standard error is gone too.
    let _ = writeln!(std::io::stderr(), "error: {message}");
    std::process::exit(1)
}
