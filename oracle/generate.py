"""Generate the expected outputs (fixtures) for every prompt (Spec §10).

Runs the model through Hugging Face transformers in float32 on CPU with eager
attention, and stores per prompt: token IDs, logits at every position, input
embeddings, the output of every decoder layer, the final-norm output, and a
greedy continuation produced by this script's own argmax loop.

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "fixtures" / MODEL_ID)
    out_dir = parser.parse_args().out
    if not MODEL_DIR.joinpath("model.safetensors").is_file():
        print(f"error: model files missing in {MODEL_DIR}; run `uv run python download.py`", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    # No device_map is given, so the model stays on CPU; check rather than assume.
    assert next(model.parameters()).dtype == torch.float32
    assert next(model.parameters()).device.type == "cpu"

    layer_outputs: list[torch.Tensor] = []
    for layer in model.model.layers:
        layer.register_forward_hook(
            lambda _m, _i, o: layer_outputs.append(o[0] if isinstance(o, tuple) else o)
        )
    eos = model.config.eos_token_id

    records = []
    for p in json.loads((Path(__file__).parent / "prompts.json").read_text()):
        ids = torch.tensor([tok(p["text"])["input_ids"]])
        with torch.no_grad():
            tensors = forward_all(model, ids, layer_outputs)
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

        file = f"{p['id']}.safetensors"
        save_file({k: v.contiguous() for k, v in tensors.items()}, out_dir / file)
        records.append(
            {
                "id": p["id"],
                "category": p["category"],
                "text": p["text"],
                "token_ids": ids[0].tolist(),
                "prompt_top2_gaps": prompt_gaps,
                "continuation_ids": continuation,
                "continuation_text": tok.decode(continuation),
                "continuation_top2_gaps": gaps,
                "stopped_at_eos": hit_eos,
                "tensors_file": file,
            }
        )
        print(f"{p['id']:<18} positions={ids.shape[1]:>4} continuation={len(continuation)} eos={hit_eos}")

    config = model.config
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
        "dims": {
            "layers": config.num_hidden_layers,
            "hidden": config.hidden_size,
            "vocab": config.vocab_size,
        },
        "continuation_tokens": CONTINUATION_TOKENS,
        "eos_token_id": eos,
        "prompts": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
