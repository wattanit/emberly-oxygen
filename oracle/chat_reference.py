"""Generate the chat fixtures for the instruct model (Spec §10, FR-39).

For each conversation in conversations.json this renders the reference library's
chat template (with the generation prompt), encodes the rendered text, runs the
instruct model in float64 on CPU (the same reference as the base model's, see
reference_common.py), and stores the logits at every prompt position and a greedy
reply produced by this script's own argmax loop, stopping after <|im_end|>.

Output, in fixtures/smollm2-135m-instruct/: `chat.json` (no manifest.json, so the
harness's manifest discovery never judges this model by S-1 to S-4) and one
`<id>.f64.safetensors` per conversation holding `logits`.

usage: uv run python chat_reference.py [--out DIR]
"""

import argparse
import datetime
import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from download import MODELS, ROOT
from reference_common import check_float64_pure_ops, float64_pure_ops, machine, top2_gaps

MODEL_ID = "smollm2-135m-instruct"
MODEL_DIR = ROOT / "models" / MODEL_ID
REPLY_TOKENS = 64  # initial; tune with use


def ids_of(encoded) -> list[int]:
    """Token ids from whatever apply_chat_template(tokenize=True) returns."""
    return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def render(tok, messages: list[dict]) -> tuple[str, list[int]]:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    # The library's own tokenizing path must agree with encoding the text ourselves.
    library_ids = ids_of(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    assert ids == library_ids, "encoding the rendered text differs from apply_chat_template(tokenize=True)"
    return text, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "fixtures" / MODEL_ID)
    out_dir = parser.parse_args().out
    if not MODEL_DIR.joinpath("model.safetensors").is_file():
        print(f"error: model files missing in {MODEL_DIR}; run `uv run python download.py {MODEL_ID}`", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    conversations = json.loads((Path(__file__).parent / "conversations.json").read_text())

    records = []
    with float64_pure_ops():
        check_float64_pure_ops()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.float64, attn_implementation="eager"
        ).eval()
        assert next(model.parameters()).dtype == torch.float64
        assert next(model.parameters()).device.type == "cpu"
        eos = model.config.eos_token_id
        assert eos == tok.convert_tokens_to_ids("<|im_end|>") == 2

        for c in conversations:
            text, ids = render(tok, c["messages"])
            start = tok.convert_tokens_to_ids("<|im_start|>")
            assert ids[0] == start and ids.count(start) >= 2, "the markers must be single tokens"
            tensor = torch.tensor([ids])
            with torch.no_grad():
                logits = model(tensor, use_cache=False).logits[0]
                assert logits.dtype == torch.float64
                prompt_gaps = top2_gaps(logits)
                seq, reply, gaps, hit_eos = tensor, [], [], False
                last = logits[-1:]
                for _ in range(REPLY_TOKENS):
                    gaps.append(top2_gaps(last)[0])
                    token = int(torch.argmax(last[0]))
                    reply.append(token)
                    if token == eos:
                        hit_eos = True
                        break
                    seq = torch.cat([seq, torch.tensor([[token]])], dim=1)
                    last = model(seq, use_cache=False).logits[0, -1:]
            file = f"{c['id']}.f64.safetensors"
            save_file({"logits": logits.contiguous()}, out_dir / file)
            shown = reply[:-1] if hit_eos else reply
            records.append(
                {
                    "id": c["id"],
                    "messages": c["messages"],
                    "rendered": text,
                    "token_ids": ids,
                    "prompt_top2_gaps": prompt_gaps,
                    "reply_ids": reply,
                    "reply_text": tok.decode(shown),
                    "reply_top2_gaps": gaps,
                    "stopped_at_eos": hit_eos,
                    "tensors_file": file,
                }
            )
            print(f"{c['id']:<12} positions={len(ids):>4} reply={len(reply)} eos={hit_eos} {tok.decode(shown)!r}")
    # The library is the stock one again: the replacement must have been undone.
    try:
        check_float64_pure_ops()
    except AssertionError:
        pass
    else:
        raise AssertionError("the float64 replacement was not undone")

    config = model.config
    spec = MODELS[MODEL_ID]
    chat = {
        "model_id": MODEL_ID,
        "repo": spec["repo"],
        "commit": spec["revision"],
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "machine": machine(),
        "libraries": {n: version(n) for n in ["torch", "transformers", "huggingface_hub", "safetensors", "numpy"]},
        "tensors": {"logits": "[positions, vocab], float64, at every position of the rendered prompt"},
        "dims": {"vocab": config.vocab_size},
        "reply_tokens": REPLY_TOKENS,
        "eos_token_id": eos,
        "conversations": records,
    }
    (out_dir / "chat.json").write_text(json.dumps(chat, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
