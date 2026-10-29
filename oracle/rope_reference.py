"""Print the reference's rotary-embedding cosines and sines as Rust constants.

The engine's RoPE test (crates/oxygen/src/ops.rs) compares its f32 rotation
angles with these values: transformers' LlamaRotaryEmbedding in float32, for
SmolLM2-135M's head dimension 64 and rope_theta 100000, at positions 1 and
5000. Position 5000 is chosen because an f64 angle already differs from the
reference's f32 cosine there by 8e-5 (Spec §4: "a more precise angle is a
different angle").

usage: uv run python rope_reference.py
"""

import sys

import torch
import transformers
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from download import MODEL_DIR

POSITIONS = [1, 5000]


def main() -> int:
    if not (MODEL_DIR / "config.json").exists():
        print(f"error: {MODEL_DIR} is missing; run `uv run python download.py`", file=sys.stderr)
        return 1
    rotary = LlamaRotaryEmbedding(config=AutoConfig.from_pretrained(MODEL_DIR))
    x = torch.zeros(1, 1, 1, 64)
    print(f"// transformers {transformers.__version__}, torch {torch.__version__}")
    for position in POSITIONS:
        cos, sin = rotary(x, torch.tensor([[position]]))
        for name, table in (("COS", cos[0, 0, :32]), ("SIN", sin[0, 0, :32])):
            values = [f"{float(v):.9e}" for v in table]
            print(f"const POS{position}_{name}: [f32; 32] = [")
            for i in range(0, 32, 4):
                print("    " + ", ".join(values[i : i + 4]) + ",")
            print("];")
    return 0


if __name__ == "__main__":
    sys.exit(main())
