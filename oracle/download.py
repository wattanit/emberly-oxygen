"""Download the pinned model files into models/<model-id>/.

This is the only step in the project that needs network access (Spec §10).
The repository is public, so the download is anonymous: a stale token in the
local Hugging Face cache would otherwise be sent and rejected.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# Pinned models. Changing a revision invalidates every stored fixture of that model.
MODELS = {
    "smollm2-135m": {
        "repo": "HuggingFaceTB/SmolLM2-135M",
        # Latest commit of the base model when pinned (2025-02-06).
        "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    },
    "smollm2-135m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
        # Latest commit of the instruct model when pinned (2026-11-08); Spec §3.
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        # Read by the oracle only (the chat template lives here); the engine never reads it.
        "oracle_files": ["tokenizer_config.json"],
    },
}
FILES = ["config.json", "model.safetensors", "tokenizer.json"]

ROOT = Path(__file__).resolve().parent.parent

# The base model, which the other scripts import by these names.
MODEL_ID = "smollm2-135m"
REPO = MODELS[MODEL_ID]["repo"]
REVISION = MODELS[MODEL_ID]["revision"]
MODEL_DIR = ROOT / "models" / MODEL_ID


def download(model_id: str) -> int:
    repo, revision = MODELS[model_id]["repo"], MODELS[model_id]["revision"]
    model_dir = ROOT / "models" / model_id
    info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        print(f"error: repository resolved to {info.sha}, expected {revision}", file=sys.stderr)
        return 1
    sizes = {s.rfilename: s.size for s in info.siblings}

    model_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in FILES + MODELS[model_id].get("oracle_files", []):
        path = Path(hf_hub_download(repo, name, revision=revision, local_dir=model_dir, token=False))
        if not path.is_file() or path.stat().st_size != sizes[name]:
            print(f"error: {name} is missing or has the wrong size", file=sys.stderr)
            return 1
        report[name] = path.stat().st_size

    print(json.dumps({"model_id": model_id, "repo": repo, "commit": revision, "files": report}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a pinned model into models/<model-id>/.")
    parser.add_argument("model", nargs="?", default=MODEL_ID, choices=sorted(MODELS))
    return download(parser.parse_args().model)


if __name__ == "__main__":
    sys.exit(main())
