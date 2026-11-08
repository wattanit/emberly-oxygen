"""Writes the expected adjusted distributions for sampling (FR-31, Spec section 10).

Run after generate.py, from this directory:  uv run python sampling_reference.py

For every prompt and every setting in sampling_settings.json it takes the f64
logits at the prompt's last position from the stored f64 fixtures (the model is
not run again), applies the reference library's own logit processors in the
order Specification section 8 fixes (repetition penalty, temperature, top-k,
top-p), and stores the softmax in float64 as <id>.sampling.safetensors, one
vocabulary-sized tensor per setting. The manifest gains a "sampling" entry with
the settings and, per prompt and setting, the margins to the cutoffs, so the
harness can tell a cutoff that rounding in the engine might cross.

Margins, in the units the engine's error is measured in:
  top_k_margin_logit   gap between the k-th and (k+1)-th highest logit after the
                       penalty, before temperature (null when top-k is off)
  top_p_margin_prob    the smaller of: how far the mass above the boundary token
                       is below p, and how far the mass including it is above p
                       (null when top-p is off)
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from download import MODEL_ID

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures" / MODEL_ID


def apply(setting: dict, ids: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """The library's processors in the fixed order; returns the distribution and margins."""
    scores = logits.clone().unsqueeze(0)
    temperature = setting.get("temperature", 1.0)
    top_k = setting.get("top_k", 0)
    top_p = setting.get("top_p", 1.0)
    penalty = setting.get("repetition_penalty", 1.0)
    margins = {"top_k_margin_logit": None, "top_p_margin_prob": None}

    if penalty != 1.0:
        scores = RepetitionPenaltyLogitsProcessor(penalty)(ids, scores)
    penalized = scores.clone()
    if temperature != 1.0:
        scores = TemperatureLogitsWarper(temperature)(ids, scores)
    if top_k > 0:
        # The library keeps the k highest, with ties at the k-th kept.
        top = penalized[0].topk(top_k + 1).values
        margins["top_k_margin_logit"] = (top[top_k - 1] - top[top_k]).item()
        scores = TopKLogitsWarper(top_k)(ids, scores)
    if top_p < 1.0:
        probs = scores[0].softmax(-1)
        sorted_probs = probs.sort(descending=True).values
        cum = sorted_probs.cumsum(-1)
        boundary = int((cum >= top_p).nonzero()[0])
        before = cum[boundary - 1].item() if boundary > 0 else 0.0
        margins["top_p_margin_prob"] = min(top_p - before, cum[boundary].item() - top_p)
        scores = TopPLogitsWarper(top_p)(ids, scores)
    return scores[0].softmax(-1), margins


def check_processors_keep_f64() -> None:
    """The processors must return float64 for a float64 input (start-up check).

    The check itself is shown able to fail: the same call on a float32 input
    must be rejected by assert_f64.
    """
    ids = torch.tensor([[1, 2, 3]])
    probe = torch.tensor([[2.0, -1.0, 0.5, 3.0, -2.0]], dtype=torch.float64)
    setting = {"temperature": 0.7, "top_k": 3, "top_p": 0.9, "repetition_penalty": 1.3}
    out, _ = apply(setting, ids, probe[0])
    assert_f64(out)
    narrowed, _ = apply(setting, ids, probe[0].float())
    try:
        assert_f64(narrowed)
    except AssertionError:
        return
    raise AssertionError("the dtype check accepted a float32 result: it cannot fail")


def assert_f64(t: torch.Tensor) -> None:
    assert t.dtype == torch.float64, f"processors returned {t.dtype}, not float64"


def main() -> int:
    manifest_path = FIXTURES / "manifest.json"
    if not manifest_path.is_file():
        print(f"error: {manifest_path} missing; run generate.py first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    settings = json.loads((Path(__file__).parent / "sampling_settings.json").read_text())["settings"]
    check_processors_keep_f64()

    margins = {}
    for prompt in manifest["prompts"]:
        tensors = load_file(FIXTURES / prompt["f64_tensors_file"])
        logits = tensors["logits"][-1]
        assert logits.dtype == torch.float64, f"{prompt['id']}: the f64 fixture is {logits.dtype}"
        ids = torch.tensor([prompt["token_ids"]])
        out, per_setting = {}, {}
        for setting in settings:
            dist, m = apply(setting, ids, logits)
            total = dist.sum().item()
            assert abs(total - 1.0) < 1e-12, f"{prompt['id']} {setting['name']}: sums to {total}"
            out[setting["name"]] = dist.contiguous()
            per_setting[setting["name"]] = {**m, "support": int((dist > 0).sum())}
        file = f"{prompt['id']}.sampling.safetensors"
        save_file(out, FIXTURES / file)
        prompt["sampling_file"] = file
        margins[prompt["id"]] = per_setting
        print(f"{prompt['id']:<18} supports " + " ".join(f"{k}={v['support']}" for k, v in per_setting.items()))

    manifest["sampling"] = {
        "settings": settings,
        "context": "the prompt's token ids; distribution taken at the prompt's last position from the f64 logits",
        "margins": margins,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
