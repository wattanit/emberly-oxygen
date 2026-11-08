//! The reported-only measurements on the real instruct model (FR-39): they are
//! made, finite and non-negative, and they do not depend on S-11 (they run on the
//! stored rendering, not the engine's).

mod common;

use common::compare::Verdict;
use common::fixtures::{discover_chat, require_model_dir, workspace_root};
use common::harness::report_chat;
use emberly_oxygen::{Model, Tokenizer};

#[test]
fn fr39_the_reported_figures_are_measured_on_the_real_model() {
    let root = workspace_root();
    let all = discover_chat(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}"));
    let fixtures = &all[0];
    let dir = root.join("models").join(&fixtures.chat.model_id);
    require_model_dir(&dir).unwrap_or_else(|e| panic!("{e}"));
    let model = Model::load(&dir).unwrap_or_else(|e| panic!("{e}"));
    let tokenizer = Tokenizer::load(&dir).unwrap_or_else(|e| panic!("{e}"));
    let c = fixtures.chat.conversations.iter().find(|c| c.id == "no-system").expect("conversation missing");

    let r = report_chat(&model, &tokenizer, fixtures, c).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((r.model_id.as_str(), r.conversation.as_str()), (fixtures.chat.model_id.as_str(), "no-system"));
    assert!(r.max_logit_error.is_finite() && r.max_logit_error > 0.0, "an f32 engine differs from f64: {}", r.max_logit_error);
    assert_eq!(r.reference_reply_len, c.reply_ids.len());
    assert!(r.engine_reply_len > 1, "the reply is generated up to the stored reply's limit, not cut short");
    assert!(matches!(r.reply, Verdict::Pass | Verdict::NearTie { .. } | Verdict::Fail { .. }));
}

#[test]
fn fr39_a_missing_logits_file_is_an_error_not_a_figure() {
    let root = workspace_root();
    let mut all = discover_chat(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}"));
    let fixtures = &mut all[0];
    let dir = root.join("models").join(&fixtures.chat.model_id);
    let model = Model::load(&dir).unwrap_or_else(|e| panic!("{e}"));
    let tokenizer = Tokenizer::load(&dir).unwrap_or_else(|e| panic!("{e}"));
    let mut c = fixtures.chat.conversations[0].clone();
    c.tensors_file = "absent.f64.safetensors".into();
    let error = report_chat(&model, &tokenizer, fixtures, &c).unwrap_err();
    assert!(error.contains("absent.f64.safetensors"), "{error}");
}

// FR-39: the reference's length is the stored reply's, and the reply is compared with it
// under S-4's rule: a stored reply cut to two tokens is reported as differing where it ends.
#[test]
fn fr39_a_shorter_stored_reply_is_reported_as_differing_where_it_ends() {
    let root = workspace_root();
    let all = discover_chat(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}"));
    let fixtures = &all[0];
    let dir = root.join("models").join(&fixtures.chat.model_id);
    let model = Model::load(&dir).unwrap_or_else(|e| panic!("{e}"));
    let tokenizer = Tokenizer::load(&dir).unwrap_or_else(|e| panic!("{e}"));
    let mut c = fixtures.chat.conversations.iter().find(|c| c.id == "no-system").unwrap().clone();
    c.reply_ids.truncate(2);
    c.reply_top2_gaps.truncate(2);
    let r = report_chat(&model, &tokenizer, fixtures, &c).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(r.reference_reply_len, 2);
    assert!(r.engine_reply_len > 2);
    assert!(matches!(r.reply, Verdict::Fail { position: Some(2), .. }), "{:?}", r.reply);
}
