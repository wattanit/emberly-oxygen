"""Generate the tokenizer's differential corpus (Spec §6, §10).

For a few thousand strings, stores what the reference tokenizer does: the
pre-tokenizer's pieces (in the byte-level alphabet) and the encoded token IDs.
The strings are the prompt set, hand-picked edge cases (including special-token text), and seeded random
mixtures of character pools chosen to stress the splitting pattern: Thai
including combining marks, contractions, every kind of white space, digits,
other-script numerals, emoji sequences and punctuation.

Output: fixtures/<model-id>/tokenizer-corpus.json (git-ignored like the other
fixtures). The same seed always produces the same strings.

usage: uv run python tokenizer_corpus.py
"""

import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer

from download import MODEL_DIR, MODEL_ID

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / MODEL_ID / "tokenizer-corpus.json"
SEED = 20261029
RANDOM_STRINGS = 4000

EDGE_CASES = [
    "", " ", "  ", "\n", "\n\n", " \n ", "\t\t x", "a", "a ", " a", "  a", "   a  ",
    "Hello world", "Hello  world", "Hello world  ", "Hello world \n",
    "it's", "IT'S", "I'm you're we've they'll he'd don't", "'s", "'", "''s", "'ss", "x'y",
    "it’s", "rock 'n' roll", "'re'll've",
    "a1", "1a", "123", "a123b", "12 34", "1,234.56", "x=1+2", "3.14",
    "١٢٣", "Ⅷ ² ½", "๑๒๓",
    "!!", "?!", " !!", "a!b", "a !b", "...", " ...", "(a)", "[b]", "a_b", "a-b", "a/b",
    " ", "a b", "　", "a　b", " ", "\u0085", "a​b",
    "ก", "ก็", "ได้", "สวัสดีครับ",
    "ผมชื่อสมชาย", "กำ", "ก์", "ะั",
    "你好世界", "こんにちは", "한국어",
    "café", "café", "ü", "ü", "مرحبا",
    "\U0001f600", "\U0001f468‍\U0001f469‍\U0001f467", "\U0001f1f9\U0001f1ed", "a\U0001f600b",
    "<|endoftext|>", "a<|im_start|>b", "<|im_start|>user\nhi<|im_end|>\n", "<|im_start|><|im_end|>",
    "<|endoftext", "<|im_start|", "<file_sep>x<filename>", "x <jupyter_code> y", "<issue_start>1<issue_comment>",
    "<reponame><repo_name>", "<|im_start|>\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35<|im_end|>",
    "\x04", "a\x04b", "\x06\x13\x14\x16\x1d", "x\x1dy z", "\U00040000", "a\U00040001b", "\U0010ffff",
    "def f(x):\n    return x + 1\n", "fmt.Println(\"hi\")\n\n\tfoo()\n", "\r\n", "a\r\nb", "\x00", "a\x00b", "\x7f",
]

POOLS = [
    list("abcdefghijklmnopqrstuvwxyz"),
    list("ABCXYZ"),
    list("0123456789"),
    [" "] * 6 + ["  ", "   "],
    ["\n", "\n\n", "\t", "\r\n", "\r", "\x0b", "\x0c"],
    [" ", "　", " ", " ", "\u0085", "​", " "],
    list("!?.,;:-_()[]{}<>/\\|@#$%^&*+=~`\"") + ["'"] * 3,
    ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "’s"],
    [chr(c) for c in range(0x0E01, 0x0E2F)],       # Thai consonants
    [chr(c) for c in (0x0E30, 0x0E31, 0x0E32, 0x0E33, 0x0E34, 0x0E35, 0x0E36, 0x0E37,
                      0x0E38, 0x0E39, 0x0E3A, 0x0E40, 0x0E41, 0x0E42, 0x0E43, 0x0E44,
                      0x0E45, 0x0E46, 0x0E47, 0x0E48, 0x0E49, 0x0E4A, 0x0E4B, 0x0E4C,
                      0x0E4D, 0x0E4E)],             # Thai vowels, tone marks, signs
    [chr(c) for c in range(0x0E50, 0x0E5A)],       # Thai digits
    list("你好世界こんにちは한국어"),
    list("éüñß̣́̈"),
    list("١٢Ⅷ²½①"),
    ["\U0001f600", "\U0001f44d\U0001f3fd", "\U0001f468‍\U0001f469‍\U0001f467", "\U0001f1fa\U0001f1f8", "❤️", "©"],
    list("مرحباשלוםनमस्ते"),
]


def random_string(rng: random.Random) -> str:
    n = rng.randint(1, 24)
    # Each string draws from a few pools so that interesting neighbours are common.
    chosen = rng.sample(POOLS, rng.randint(1, 4))
    return "".join(rng.choice(rng.choice(chosen)) for _ in range(n))


def main() -> int:
    path = MODEL_DIR / "tokenizer.json"
    if not path.exists():
        print(f"error: {path} is missing; run `uv run python download.py`", file=sys.stderr)
        return 1
    tokenizer = Tokenizer.from_file(str(path))
    prompts = [p["text"] for p in json.loads((Path(__file__).parent / "prompts.json").read_text())]

    rng = random.Random(SEED)
    texts = list(dict.fromkeys(EDGE_CASES + prompts + [random_string(rng) for _ in range(RANDOM_STRINGS)]))

    cases = []
    for text in texts:
        pieces = [p for p, _ in tokenizer.pre_tokenizer.pre_tokenize_str(text)]
        # Special-token text in a string is matched before splitting, as the engine must.
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        cases.append({"text": text, "pieces": pieces, "ids": ids})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": SEED, "cases": cases}, ensure_ascii=False))
    print(f"wrote {len(cases)} cases to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
