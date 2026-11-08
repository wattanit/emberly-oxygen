"""Generate the expected outputs (fixtures) for every prompt (Spec §10).

Runs the model through Hugging Face transformers on CPU with eager attention,
twice: in float64, which is the reference the engine is judged against
(Requirements S-1 to S-4), and in float32, the library's default precision,
which the harness reports against but does not judge by. Both runs load the
same published weights. Per prompt and per run it stores token IDs, logits at
every position, input embeddings, the output of every decoder layer, the
final-norm output, and a greedy continuation produced by this script's own
argmax loop. The float32 files are `<id>.safetensors`; the float64 files are
`<id>.f64.safetensors`, kept in 64-bit, and their manifest fields start `f64_`.

The float64 run is slower and writes about twice the bytes: expect minutes, and
fixtures of about 2 GB.

usage: uv run python generate.py [--out DIR]
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from download import MODEL_DIR, MODEL_ID, REPO, REVISION

ROOT = Path(__file__).resolve().parent.parent
CONTINUATION_TOKENS = 32


def machine() -> dict:
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor()
    return {
        "cpu": cpu,
        "arch": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "torch_threads": torch.get_num_threads(),
    }


def top2_gaps(logits: torch.Tensor) -> list[float]:
    """Gap between the two highest logits at each row of [positions, vocab]."""
    top = logits.topk(2, dim=-1).values
    return (top[:, 0] - top[:, 1]).tolist()


def forward_all(model, ids: torch.Tensor, layer_outputs: list) -> dict:
    """One full forward pass; returns the tensors the fixtures store.

    The library's hidden-states tuple replaces the last layer's output with the
    final-norm output, so the layer outputs are captured with hooks instead and
    the final-norm output is stored under its own name.
    """
    layer_outputs.clear()
    out = model(ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states
    n = len(model.model.layers)
    assert len(hidden) == n + 1 and len(layer_outputs) == n
    # The hooks must agree with the library's own hidden states.
    for i in range(n - 1):
        assert torch.equal(layer_outputs[i], hidden[i + 1]), f"hook mismatch at layer {i}"
    return {
        "logits": out.logits[0],
        "embeddings": hidden[0][0],
        "layer_outputs": torch.stack([t[0] for t in layer_outputs]),
        "final_norm_output": hidden[-1][0],
    }


def run_precision(dtype: torch.dtype, prompts: list, tok, out_dir: Path, suffix: str) -> dict:
    """Run every prompt through the model in `dtype`; write one tensor file per prompt.

    Returns, per prompt ID, the fields the manifest records for this run.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=dtype, attn_implementation="eager"
    ).eval()
    # No device_map is given, so the model stays on CPU; check rather than assume.
    assert next(model.parameters()).dtype == dtype
    assert next(model.parameters()).device.type == "cpu"

    layer_outputs: list[torch.Tensor] = []
    for layer in model.model.layers:
        layer.register_forward_hook(
            lambda _m, _i, o: layer_outputs.append(o[0] if isinstance(o, tuple) else o)
        )
    eos = model.config.eos_token_id

    runs = {}
    for p in prompts:
        ids = torch.tensor([tok(p["text"])["input_ids"]])
        with torch.no_grad():
            tensors = forward_all(model, ids, layer_outputs)
            assert tensors["logits"].dtype == dtype
            prompt_gaps = top2_gaps(tensors["logits"])

            # Greedy continuation by our own argmax loop (not generate()), so the
            # top-two gap is known at every step. torch.argmax returns the lowest
            # index on an exact tie, as the engine does.
            seq = ids
            continuation, gaps, hit_eos = [], [], False
            last = tensors["logits"][-1:]
            for _ in range(CONTINUATION_TOKENS):
                gaps.append(top2_gaps(last)[0])
                token = int(torch.argmax(last[0]))
                continuation.append(token)
                if token == eos:
                    hit_eos = True
                    break
                seq = torch.cat([seq, torch.tensor([[token]])], dim=1)
                last = model(seq, use_cache=False).logits[0, -1:]

        file = f"{p['id']}{suffix}.safetensors"
        save_file({k: v.contiguous() for k, v in tensors.items()}, out_dir / file)
        runs[p["id"]] = {
            "ids": ids[0].tolist(),
            "prompt_top2_gaps": prompt_gaps,
            "continuation_ids": continuation,
            "continuation_text": tok.decode(continuation),
            "continuation_top2_gaps": gaps,
            "stopped_at_eos": hit_eos,
            "tensors_file": file,
        }
        print(
            f"{str(dtype).removeprefix('torch.'):<8} {p['id']:<18} positions={ids.shape[1]:>4} "
            f"continuation={len(continuation)} eos={hit_eos}"
        )
    return {"runs": runs, "config": model.config}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "fixtures" / MODEL_ID)
    out_dir = parser.parse_args().out
    if not MODEL_DIR.joinpath("model.safetensors").is_file():
        print(f"error: model files missing in {MODEL_DIR}; run `uv run python download.py`", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompts = json.loads((Path(__file__).parent / "prompts.json").read_text())

    f32 = run_precision(torch.float32, prompts, tok, out_dir, "")
    f64 = run_precision(torch.float64, prompts, tok, out_dir, ".f64")

    records = []
    for p in prompts:
        a, b = f32["runs"][p["id"]], f64["runs"][p["id"]]
        # Same tokenizer and prompt, so the IDs agree; the check costs nothing.
        assert a["ids"] == b["ids"]
        records.append(
            {
                "id": p["id"],
                "category": p["category"],
                "text": p["text"],
                "token_ids": a["ids"],
                "prompt_top2_gaps": a["prompt_top2_gaps"],
                "continuation_ids": a["continuation_ids"],
                "continuation_text": a["continuation_text"],
                "continuation_top2_gaps": a["continuation_top2_gaps"],
                "stopped_at_eos": a["stopped_at_eos"],
                "tensors_file": a["tensors_file"],
                "f64_tensors_file": b["tensors_file"],
                "f64_prompt_top2_gaps": b["prompt_top2_gaps"],
                "f64_continuation_ids": b["continuation_ids"],
                "f64_continuation_text": b["continuation_text"],
                "f64_continuation_top2_gaps": b["continuation_top2_gaps"],
                "f64_stopped_at_eos": b["stopped_at_eos"],
            }
        )

    config = f32["config"]
    manifest = {
        "model_id": MODEL_ID,
        "repo": REPO,
        "commit": REVISION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "machine": machine(),
        "libraries": {n: version(n) for n in ["torch", "transformers", "huggingface_hub", "safetensors", "numpy"]},
        "tensors": {
            "logits": "[positions, vocab]",
            "embeddings": "[positions, hidden]",
            "layer_outputs": "[layers, positions, hidden], the raw output of each decoder layer",
            "final_norm_output": "[positions, hidden], after the final RMSNorm",
        },
        "precisions": {
            "tensors_file": "float32, the library's default precision; reported against, never judged by",
            "f64_*": "float64, stored in 64-bit; the reference S-1 to S-4 are judged against",
        },
        "dims": {
            "layers": config.num_hidden_layers,
            "hidden": config.hidden_size,
            "vocab": config.vocab_size,
        },
        "continuation_tokens": CONTINUATION_TOKENS,
        "eos_token_id": config.eos_token_id,
        "prompts": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
