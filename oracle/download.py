"""Download the pinned model files into models/<model-id>/.

This is the only step in the project that needs network access (Spec §10).
The repository is public, so the download is anonymous: a stale token in the
local Hugging Face cache would otherwise be sent and rejected.
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

MODEL_ID = "smollm2-135m"
REPO = "HuggingFaceTB/SmolLM2-135M"
# Latest commit of the base model when pinned (2025-02-06). Changing it
# invalidates every stored fixture.
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
FILES = ["config.json", "model.safetensors", "tokenizer.json"]

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / MODEL_ID


def main() -> int:
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        print(f"error: repository resolved to {info.sha}, expected {REVISION}", file=sys.stderr)
        return 1
    sizes = {s.rfilename: s.size for s in info.siblings}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in FILES:
        path = Path(
            hf_hub_download(REPO, name, revision=REVISION, local_dir=MODEL_DIR, token=False)
        )
        if not path.is_file() or path.stat().st_size != sizes[name]:
            print(f"error: {name} is missing or has the wrong size", file=sys.stderr)
            return 1
        report[name] = path.stat().st_size

    print(json.dumps({"model_id": MODEL_ID, "repo": REPO, "commit": REVISION, "files": report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
