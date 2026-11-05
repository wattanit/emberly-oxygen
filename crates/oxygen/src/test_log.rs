//! A minimal `tracing` subscriber for tests: it keeps every event with its
//! level, fields and the names of the spans it happened in. `tracing-subscriber`
//! is for the command-line tool only (Spec §15), so the library's tests use this.

use std::fmt::Debug;
use std::sync::Mutex;

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
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

/// Runs `f` with a collector at `level` installed for this thread, and returns
/// what `f` returned with every event logged meanwhile. `None` installs nothing.
pub fn collect<T>(level: Option<Level>, f: impl FnOnce() -> T) -> (T, Vec<Logged>) {
    let Some(level) = level else {
        return (f(), Vec::new());
    };
    let collector = std::sync::Arc::new(Collector { level, spans: Mutex::default(), stack: Mutex::default(), events: Mutex::default() });
    let result = tracing::subscriber::with_default(collector.clone(), f);
    let events = collector.events.lock().unwrap().clone();
    (result, events)
}

struct Collector {
    level: Level,
    /// Every span created, as shown in a path; the ID is the index plus one.
    spans: Mutex<Vec<String>>,
    /// The spans entered and not yet left, innermost last.
    stack: Mutex<Vec<u64>>,
    events: Mutex<Vec<Logged>>,
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

impl Subscriber for Collector {
    // Ask every time, so collectors at different levels on other test threads
    // do not decide for this one.
    fn register_callsite(&self, _: &'static Metadata<'static>) -> Interest {
        Interest::sometimes()
    }

    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        *metadata.level() <= self.level
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
        let mut spans = self.spans.lock().unwrap();
        spans.push(shown);
        Id::from_u64(spans.len() as u64)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let spans = self.spans.lock().unwrap();
        let path = self.stack.lock().unwrap().iter().map(|&id| spans[id as usize - 1].clone()).collect();
        self.events.lock().unwrap().push(Logged { level: *event.metadata().level(), fields: fields.0, path });
    }

    fn enter(&self, span: &Id) {
        self.stack.lock().unwrap().push(span.into_u64());
    }

    fn exit(&self, span: &Id) {
        let mut stack = self.stack.lock().unwrap();
        let at = stack.iter().rposition(|&id| id == span.into_u64()).expect("exit of a span that was entered");
        stack.remove(at);
    }
}
