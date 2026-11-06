//! Arguments and the generate command (Spec §9). The engine work is the
//! library's; this reads the prompt, routes text and log events, and reports.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use clap::{ArgAction, Args, Parser, Subcommand};
use emberly_oxygen::{generate, Model, Stats, Tokenizer};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt::writer::BoxMakeWriter;

/// A decoder-only LLM inference engine, built from scratch.
#[derive(Parser)]
#[command(name = "emberly-oxygen", version)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a greedy continuation of a prompt and stream it to standard output.
    Generate(GenerateArgs),
}

#[derive(Args)]
struct GenerateArgs {
    /// Model directory holding config.json, model.safetensors and tokenizer.json.
    #[arg(long, value_name = "DIR")]
    model: PathBuf,

    /// The most tokens to generate. Generation also stops at the end-of-sequence
    /// token, which is not printed. Zero checks the prompt and generates nothing.
    #[arg(long, value_name = "N")]
    max_new_tokens: usize,

    /// Print run statistics to standard error: model load time, prompt and
    /// generated token counts, time to first token (from receiving the prompt to
    /// choosing the first token, so the first token counts as prompt work) and
    /// generation throughput (the tokens after the first, per second after it).
    #[arg(long)]
    stats: bool,

    /// Log more, to standard error: -v run summary and timings, -vv each decode
    /// step, -vvv each layer within each step. By default only warnings and errors.
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,

    /// Write the log to this file instead of standard error.
    #[arg(long, value_name = "PATH")]
    log_file: Option<PathBuf>,

    /// The prompt. When omitted, all of standard input is read exactly as given,
    /// so a trailing newline (as `echo` adds) is part of the prompt.
    prompt: Option<String>,
}

pub fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Generate(args) => run_generate(args),
    }
}

fn run_generate(args: GenerateArgs) -> Result<(), String> {
    init_logging(args.verbose, args.log_file.as_deref())?;
    let prompt = match args.prompt {
        Some(prompt) => prompt,
        None => read_stdin()?,
    };

    let start = Instant::now();
    let model = Model::load(&args.model).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::load(&args.model).map_err(|e| e.to_string())?;
    let load_time = start.elapsed();

    let mut stdout = io::stdout().lock();
    let mut write = |text: &str| {
        if let Err(e) = stdout.write_all(text.as_bytes()).and_then(|()| stdout.flush()) {
            // Usually a closed pipe: nobody is reading, so stop at once.
            crate::fail(&format!("writing to standard output: {e}"));
        }
    };
    let generation = generate(&model, &tokenizer, &prompt, args.max_new_tokens, &mut write).map_err(|e| e.to_string())?;

    if args.stats {
        print_stats(load_time, &generation.stats);
    }
    Ok(())
}

/// Installs the subscriber that formats the library's events, with their span
/// path, to standard error or to `log_file` (FR-23, FR-24). The file is created
/// here, before any work.
fn init_logging(verbose: u8, log_file: Option<&Path>) -> Result<(), String> {
    let level = match verbose {
        0 => LevelFilter::WARN,
        1 => LevelFilter::INFO,
        2 => LevelFilter::DEBUG,
        _ => LevelFilter::TRACE,
    };
    let writer = match log_file {
        Some(path) => {
            let file = File::create(path).map_err(|e| format!("cannot create log file {}: {e}", path.display()))?;
            BoxMakeWriter::new(Mutex::new(file))
        }
        None => BoxMakeWriter::new(io::stderr),
    };
    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_writer(writer)
        .try_init()
        .map_err(|e| format!("cannot start logging: {e}"))
}

/// All of standard input as the prompt, byte for byte. Text that is not UTF-8
/// is an error rather than being repaired.
fn read_stdin() -> Result<String, String> {
    let mut bytes = Vec::new();
    io::stdin()
        .lock()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("reading standard input: {e}"))?;
    String::from_utf8(bytes).map_err(|e| {
        format!("standard input is not valid UTF-8 (at byte {})", e.utf8_error().valid_up_to())
    })
}

/// The `--stats` report (FR-21), on standard error.
fn print_stats(load_time: Duration, stats: &Stats) {
    let seconds = |d: Duration| format!("{:.3} s", d.as_secs_f64());
    let first = stats.time_to_first_token.map_or("-".into(), seconds);
    let rate = stats
        .generation_tokens_per_second()
        .map_or("- (fewer than two tokens)".into(), |r| format!("{r:.2} tokens/s"));
    let report = format!(
        "\n\n---------- stats ----------\n\
         model load:          {}\n\
         prompt tokens:       {}\n\
         generated tokens:    {}\n\
         time to first token: {first}\n\
         generation:          {rate}\n",
        seconds(load_time),
        stats.prompt_tokens,
        stats.generated_tokens,
    );
    let _ = io::stderr().write_all(report.as_bytes());
}
