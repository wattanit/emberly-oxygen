"""Check prompts.json against the category rules (FR-3) with the reference tokenizer.

Prints prompt and token counts per category. Exits non-zero if a rule fails.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from download import MODEL_DIR

CATEGORIES = ["english", "thai", "code", "utf8-split", "long"]
LONG_TOKENS = (480, 544)  # about 512, the longest benchmark prompt
SHORT_TOKENS = (24, 40)  # about 32, the shortest benchmark prompt


def main() -> int:
    prompts = json.loads((Path(__file__).parent / "prompts.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    errors = []
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids):
        errors.append("prompt IDs are not unique")

    by_cat = defaultdict(list)
    for p in prompts:
        tokens = tok(p["text"])["input_ids"]
        # A token that alone decodes to U+FFFD carries a partial multi-byte character.
        partial = sum("�" in tok.decode([t]) for t in tokens)
        by_cat[p["category"]].append(p["id"])
        print(f"{p['category']:<11} {p['id']:<18} tokens={len(tokens):>4} partial-char tokens={partial:>4}")
        if p["category"] == "utf8-split" and partial == 0:
            errors.append(f"{p['id']}: no token splits a multi-byte character")
        if p["category"] == "long" and not LONG_TOKENS[0] <= len(tokens) <= LONG_TOKENS[1]:
            errors.append(f"{p['id']}: {len(tokens)} tokens is not about 512")
        if p["id"] == "english-short" and not SHORT_TOKENS[0] <= len(tokens) <= SHORT_TOKENS[1]:
            errors.append(f"{p['id']}: {len(tokens)} tokens is not about 32")

    for c in CATEGORIES:
        if len(by_cat[c]) < 2:
            errors.append(f"category {c} has fewer than two prompts")
    for c in by_cat:
        if c not in CATEGORIES:
            errors.append(f"unknown category {c}")

    for e in errors:
        print("error:", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
