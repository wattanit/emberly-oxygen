//! Arguments and the generate command (Spec §9). The engine work is the
//! library's; this reads the prompt, routes text and log events, and reports.

use std::fs::File;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::{ArgAction, Args, Parser, Subcommand};
use emberly_oxygen::{generate_with, Chat, Error, Model, Sampling, Stats, Tokenizer};
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
    /// Generate a continuation of a prompt and stream it to standard output: greedy by
    /// default, sampled with --temperature above 0.
    Generate(GenerateArgs),
    /// Chat with an instruct model in the terminal: one message per line, each reply
    /// streamed to standard output. Ends at end of input or a line that is exactly /exit.
    Chat(ChatArgs),
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

    /// Temperature: logits are divided by it. 0 (the default) is greedy selection, and
    /// then the other sampling options are ignored with a warning. Above 0 the run is
    /// sampled, and its seed is printed to standard error.
    #[arg(long, value_name = "T", default_value_t = 0.0)]
    temperature: f32,

    /// Sample only among the K highest-scoring tokens (ties with the K-th stay). 0 is off.
    #[arg(long, value_name = "K", default_value_t = 0)]
    top_k: usize,

    /// Sample only among the smallest set of most likely tokens whose probabilities add
    /// up to at least P, 0 < P <= 1. 1 is off.
    #[arg(long, value_name = "P", default_value_t = 1.0)]
    top_p: f32,

    /// Divide the positive scores and multiply the negative ones of every token already
    /// in the prompt or the output by R, as the reference library does; R > 0. 1 is off.
    #[arg(long, value_name = "R", default_value_t = 1.0)]
    repetition_penalty: f32,

    /// Seed of the random number generator. The same seed, prompt, settings and build
    /// give the same text; reproduction across releases is not promised. When sampling
    /// without one, a seed is taken from the clock and printed.
    #[arg(long, value_name = "N")]
    seed: Option<u64>,

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

/// The options of the chat command (Spec §9). The sampling defaults are the
/// instruct model's published example (FR-37).
#[derive(Args)]
struct ChatArgs {
    /// Model directory holding config.json, model.safetensors and tokenizer.json of an
    /// instruct model.
    #[arg(long, value_name = "DIR")]
    model: PathBuf,

    /// A system message to start the conversation with, instead of the model's default one.
    #[arg(long, value_name = "TEXT")]
    system: Option<String>,

    /// The most tokens in one reply. A reply also ends at the model's end-of-turn token,
    /// which is not printed. A turn whose conversation plus this many tokens would not
    /// fit the model's context is rejected and the chat continues.
    #[arg(long, value_name = "N", default_value_t = 256)]
    max_new_tokens: usize,

    /// Temperature: logits are divided by it. The default is 0.2, the instruct model's
    /// published example. 0 is greedy replies; the other sampling options are then ignored.
    #[arg(long, value_name = "T", default_value_t = 0.2)]
    temperature: f32,

    /// Sample only among the K highest-scoring tokens (ties with the K-th stay). 0 is off.
    #[arg(long, value_name = "K", default_value_t = 0)]
    top_k: usize,

    /// Sample only among the smallest set of most likely tokens whose probabilities add
    /// up to at least P, 0 < P <= 1. The default is 0.9, or 1 (off) when --temperature
    /// is 0 and this is not given.
    #[arg(long, value_name = "P")]
    top_p: Option<f32>,

    /// Divide the positive scores and multiply the negative ones of every token already
    /// in the conversation or the reply by R, as the reference library does; R > 0. 1 is off.
    #[arg(long, value_name = "R", default_value_t = 1.0)]
    repetition_penalty: f32,

    /// Seed of the random number generator for the whole conversation. When sampling
    /// without one, a seed is taken from the clock and printed once, before the first prompt.
    #[arg(long, value_name = "N")]
    seed: Option<u64>,

    /// Log more, to standard error: -v run summary and timings, -vv each decode
    /// step, -vvv each layer within each step. By default only warnings and errors.
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,

    /// Write the log to this file instead of standard error.
    #[arg(long, value_name = "PATH")]
    log_file: Option<PathBuf>,
}

pub fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Generate(args) => run_generate(args),
        Command::Chat(args) => run_chat(args),
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
    let mut sampling = Sampling {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
    };
    // Sampling always reports its seed, before any text, so the run can be replayed (FR-28).
    // An invalid setting is rejected by the library, which names it, before anything runs.
    if sampling.temperature > 0.0 && sampling.validate().is_ok() {
        let seed = sampling.seed.unwrap_or_else(clock_seed);
        sampling.seed = Some(seed);
        eprintln!("seed: {seed}");
    }
    let generation =
        generate_with(&model, &tokenizer, &prompt, args.max_new_tokens, &sampling, &mut write).map_err(|e| e.to_string())?;

    if args.stats {
        print_stats(load_time, &generation.stats);
    }
    Ok(())
}

fn run_chat(args: ChatArgs) -> Result<(), String> {
    init_logging(args.verbose, args.log_file.as_deref())?;
    let model = Model::load(&args.model).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::load(&args.model).map_err(|e| e.to_string())?;

    let mut sampling = Sampling {
        temperature: args.temperature,
        top_k: args.top_k,
        // Greedy chat asked for with --temperature 0 alone is not answered with a warning about
        // a top-p the user never set; one the user did set is still reported ignored (Spec §9).
        top_p: args.top_p.unwrap_or(if args.temperature == 0.0 { 1.0 } else { 0.9 }),
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
    };
    // The seed is reported once, before the first prompt marker, so the whole conversation can be replayed (FR-28).
    if sampling.temperature > 0.0 && sampling.validate().is_ok() {
        let seed = sampling.seed.unwrap_or_else(clock_seed);
        sampling.seed = Some(seed);
        eprintln!("seed: {seed}");
    }
    let mut chat = Chat::new(&model, &tokenizer, args.system, &sampling, args.max_new_tokens).map_err(|e| e.to_string())?;

    let mut stdout = io::stdout().lock();
    let mut stdin = io::stdin().lock();
    let mut line = String::new();
    loop {
        // The marker invites the next message; it is not a reply, so it goes to standard error (FR-38, FR-20).
        let mut stderr = io::stderr();
        let _ = stderr.write_all(b"> ");
        let _ = stderr.flush();
        line.clear();
        let read = stdin.read_line(&mut line).map_err(|e| format!("reading standard input: {e}"))?;
        if read == 0 {
            let _ = stderr.write_all(b"\n");
            return Ok(());
        }
        let message = line.trim_end_matches(['\n', '\r']);
        if message == "/exit" {
            return Ok(());
        }
        if message.trim().is_empty() {
            continue;
        }
        let mut write = |text: &str| {
            if let Err(e) = stdout.write_all(text.as_bytes()).and_then(|()| stdout.flush()) {
                // Usually a closed pipe: nobody is reading, so stop at once.
                crate::fail(&format!("writing to standard output: {e}"));
            }
        };
        match chat.turn(message, &mut write) {
            Ok(_) => write("\n"),
            // A turn that cannot fit is the user's to shorten: say so and go on (FR-36).
            Err(e @ Error::Rejected(_)) => eprintln!("error: {e}"),
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// A seed from the system clock, for a sampled run that was not given one.
fn clock_seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
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
