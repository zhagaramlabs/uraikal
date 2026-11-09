"""D4.web, D4.literature and fresh.web for every open tokenizer we know of, before the tokenizer release.

The roster is uraikal's D4 list plus the tokenizers the release plan said to add, and the release folder itself:
- Llama 3.1 and Aya Expanse were in the D4 list but skipped as gated. Llama loads now. For Aya, the gated repo's
  tokenizer.json is read from an ungated copy, checked against the gated file's sha256, which Hugging Face lists
  publicly.
- BrahmicTokenizer-131K is new.
- Krutrim's IndicSuperTokenizer (arXiv 2511.03237, since renamed MUTANT-Indic) has no download on Hugging Face or
  GitHub as of 2026-11-08. Krutrim-1's own tokenizer stands in. Its repo config won't load, so its tokenizer.json
  is read directly.
Every slice comes from data/bpb/eval.json, which is private and stays on our machines. Its D4 slices are the same
passages as data/d4_*.jsonl, so D4 numbers can be compared with results/d4.json. Any difference is reported.

    uv run --extra tokenizers python experiments/tokenizer_d4_fresh.py

Also scores r2-grapheme-64k, the whole-letter tokenizer without stem-suffix cuts that Ezhuthu was first built from.

Writes results/experiments/tokenizer_d4_fresh.json.
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import PreTrainedTokenizerFast

from uraikal.tokenizer_eval import DEFAULT_TOKENIZERS, load, measure

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/ezhuthu-64k"
AYA = "CohereLabs/aya-expanse-8b"
AYA_SHA256 = "c69a7ea6c0927dfac8c349186ebcf0466a4723c21cbdb2e850cf559f0bee92b8"  # the gated repo's tokenizer.json
FILES = {  # name: (repo to read tokenizer.json from, revision, sha256 it must have, or None)
    AYA: ("adamo1139/aya-expanse-8b-ungated", None, AYA_SHA256),
    "krutrim-ai-labs/Krutrim-1-instruct": ("krutrim-ai-labs/Krutrim-1-instruct", "ef1e55353589e1b53f3c79b9526666ac8901df11", None),
}
NAMES = DEFAULT_TOKENIZERS + ["theschoolofai/BrahmicTokenizer-131K", "krutrim-ai-labs/Krutrim-1-instruct"]
SLICES = ("D4.web", "D4.literature", "fresh.web")


def from_file(repo, revision, sha256):
    path = Path(hf_hub_download(repo, "tokenizer.json", revision=revision))
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256 in (None, got), (repo, got)
    return PreTrainedTokenizerFast(tokenizer_file=str(path)), {"read_from": repo, "tokenizer_json_sha256": got}


def main():
    held = json.load(open(ROOT / "data/bpb/eval.json", encoding="utf-8"))
    old = {(r["tokenizer"], r["slice"]): r for r in json.load(open(ROOT / "results/d4.json"))}
    out = {"slices": {s: len(held[s]) for s in SLICES}, "results": [], "skipped": {}, "differs_from_d4_json": []}
    api = HfApi()
    for name in NAMES + ["ezhuthu-64k", "r2-grapheme-64k"]:
        try:
            if name == "ezhuthu-64k":
                tok, source = load(str(RELEASE)), {"read_from": "release/ezhuthu-64k (r4-morph-64k)"}
            elif name == "r2-grapheme-64k":  # the whole-letter sibling without stem-suffix cuts, not released
                path = "data/tamil_tokenizer/r2-grapheme-64k.json"
                tok, source = PreTrainedTokenizerFast(tokenizer_file=str(ROOT / path)), {"read_from": path}
            elif name in FILES:
                tok, source = from_file(*FILES[name])
            else:
                tok, source = load(name), {}
            if name not in ("ezhuthu-64k", "r2-grapheme-64k"):
                source["revision"] = FILES.get(name, (0, None))[1] or api.model_info(FILES.get(name, (name,))[0]).sha
        except Exception as e:  # gated repos, missing files, remote code we won't run
            out["skipped"][name] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
            print("skipped", name, out["skipped"][name], file=sys.stderr, flush=True)
            continue
        row = {"tokenizer": name, "vocab_size": len(tok), **source}
        for s in SLICES:
            row[s] = measure(tok, held[s])
            prev = old.get((name, s))
            if prev and any(abs(prev[k] - row[s][k]) > 1e-9 for k in ("tokens_per_word", "letter_cut_words", "round_trip")):
                out["differs_from_d4_json"].append({"tokenizer": name, "slice": s,
                                                    "was": prev["tokens_per_word"], "now": row[s]["tokens_per_word"]})
        out["results"].append(row)
        print(f"{name:<40}{len(tok):>8}" + "".join(f"{row[s]['tokens_per_word']:>8.3f}" for s in SLICES)
              + f"{row['fresh.web']['letter_cut_words']:>8.1%}{row['fresh.web']['round_trip']:>8.1%}", flush=True)
        json.dump(out, open(ROOT / "results/experiments/tokenizer_d4_fresh.json", "w"), indent=1, ensure_ascii=False)
    print("differs from results/d4.json:", out["differs_from_d4_json"] or "nothing")


if __name__ == "__main__":
    main()
