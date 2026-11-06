//! A minimal `tracing` subscriber for tests: it keeps every event with its
//! level, fields and the names of the spans it happened in. `tracing-subscriber`
//! is for the command-line tool only (Spec §15), so the library's tests use this.

use std::cell::RefCell;
use std::fmt::Debug;
use std::sync::Once;

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::level_filters::LevelFilter;
use tracing::{Event, Level, Metadata, Subscriber};

/// One event: its level, its fields as text, and the path of spans it was in,
/// outermost first, each as `name` or `name{field=value}`.
#[derive(Debug, Clone)]
pub struct Logged {
    pub level: Level,
    pub fields: Vec<(String, String)>,
    pub path: Vec<String>,
}

impl Logged {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    pub fn message(&self) -> &str {
        self.get("message").unwrap_or("")
    }
}

/// Runs `f` with a collector at `level` for this thread, and returns what `f`
/// returned with every event logged meanwhile. `None` collects nothing, which
/// is the same as running with no subscriber.
///
/// One subscriber, [`Router`], is installed globally once and hands each event
/// to the collector of the thread it happens on. Installing a subscriber per
/// test (`with_default`) is racy with tests on parallel threads: `tracing`
/// caches whether anyone is interested in a callsite, and a callsite first
/// reached on a thread with no subscriber while another thread is installing
/// one can be cached as uninteresting, so a test silently receives nothing
/// (seen as a debug test with no step events in about one run in ten).
///
/// The same can happen once here, to a callsite being registered on another
/// thread while the router is installed. So after installing, registrations
/// already under way are given time to finish, and every call rebuilds the
/// cache before running `f` (seen as a layer event without its layer span
/// about one run in 300).
pub fn collect<T>(level: Option<Level>, f: impl FnOnce() -> T) -> (T, Vec<Logged>) {
    let Some(level) = level else {
        return (f(), Vec::new());
    };
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        tracing::subscriber::set_global_default(Router).expect("no other global subscriber in tests");
        std::thread::sleep(std::time::Duration::from_millis(10));
    });
    tracing::callsite::rebuild_interest_cache();

    CURRENT.with(|c| *c.borrow_mut() = Some(Collector { level, spans: Vec::new(), stack: Vec::new(), events: Vec::new() }));
    let result = f();
    let collector = CURRENT.with(|c| c.borrow_mut().take()).expect("the collector is still there");
    (result, collector.events)
}

thread_local! {
    static CURRENT: RefCell<Option<Collector>> = const { RefCell::new(None) };
}

/// What one thread's test has logged.
struct Collector {
    level: Level,
    /// Every span created, as shown in a path; the ID is the index plus one.
    spans: Vec<String>,
    /// The spans entered and not yet left, innermost last.
    stack: Vec<u64>,
    events: Vec<Logged>,
}

fn with_current<R>(f: impl FnOnce(&mut Collector) -> R) -> Option<R> {
    CURRENT.with(|c| c.borrow_mut().as_mut().map(f))
}

#[derive(Default)]
struct Fields(Vec<(String, String)>);

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        self.0.push((field.name().into(), format!("{value:?}")));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.push((field.name().into(), value.into()));
    }
}

/// The one global subscriber of the test binary: interested in everything,
/// enabled only on a thread with a collector at a level that wants the event.
struct Router;

impl Subscriber for Router {
    fn register_callsite(&self, _: &'static Metadata<'static>) -> Interest {
        Interest::sometimes()
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        Some(LevelFilter::TRACE)
    }

    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        with_current(|c| *metadata.level() <= c.level).unwrap_or(false)
    }

    fn new_span(&self, attrs: &Attributes<'_>) -> Id {
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        let name = attrs.metadata().name();
        let shown = if fields.0.is_empty() {
            name.to_string()
        } else {
            let inner: Vec<String> = fields.0.iter().map(|(n, v)| format!("{n}={v}")).collect();
            format!("{name}{{{}}}", inner.join(","))
        };
        let id = with_current(|c| {
            c.spans.push(shown);
            c.spans.len() as u64
        });
        Id::from_u64(id.expect("spans are only created where enabled"))
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        with_current(|c| {
            let path = c.stack.iter().map(|&id| c.spans[id as usize - 1].clone()).collect();
            c.events.push(Logged { level: *event.metadata().level(), fields: fields.0, path });
        });
    }

    fn enter(&self, span: &Id) {
        with_current(|c| c.stack.push(span.into_u64()));
    }

    fn exit(&self, span: &Id) {
        with_current(|c| {
            let at = c.stack.iter().rposition(|&id| id == span.into_u64()).expect("exit of a span that was entered");
            c.stack.remove(at);
        });
    }
}
