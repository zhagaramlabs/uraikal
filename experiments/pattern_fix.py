"""Pattern or vocabulary: how much of Qwen3's Tamil cost is the split pattern, and how much the vocabulary?

Qwen3 -> Qwen3.5 changed the pattern and the vocabulary together. Here two byte-level BPE tokenizers
are trained on the same Tamil text, identical except for the split pattern (Qwen3's, then Qwen3.5's),
at several vocabulary sizes, and scored on the D4 slices with uraikal's own measure().
Training text: Sangraha verified Tamil shard data-0 (CC BY 4.0); D4.web comes from data-52.

    uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/pattern_fix.py

Writes results/experiments/pattern_fix.json, and the trained tokenizers to data/pattern_fix/.
Section 6 of the paper "Where open tokenizers break Tamil" reports it.
"""
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from uraikal.tokenizer_eval import WORD, load, measure

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SHARD = "verified/tam/data-0.parquet"
SIZES = [16_000, 32_000, 64_000, 128_000]


def pattern(name):
    split = json.loads(load(name).backend_tokenizer.to_str())["pre_tokenizer"]["pretokenizers"][0]
    return split["pattern"]["Regex"]


def untrained(regex):
    t = Tokenizer(models.BPE())
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(regex), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    return t


def floor(t, texts):
    """Tokens per word with an unlimited vocabulary: BPE never merges across pre-tokens."""
    pieces = sum(len(t.pre_tokenizer.pre_tokenize_str(x)) for x in texts)
    return pieces / sum(len(WORD.findall(x)) for x in texts)


def main():
    arms = {"qwen3-pattern": pattern("Qwen/Qwen3-8B"), "qwen3.5-pattern": pattern("Qwen/Qwen3.5-0.8B")}
    assert arms["qwen3-pattern"] != arms["qwen3.5-pattern"]
    docs = [x for x in pq.read_table(hf_hub_download("ai4bharat/sangraha", SHARD, repo_type="dataset"),
                                     columns=["text"]).column(0).to_pylist() if x]
    slices = {f"D4.{s}": [json.loads(line)["text"] for line in open(DATA / f"d4_{s}.jsonl", encoding="utf-8")]
              for s in ("web", "literature")}
    out = {"training": {"shard": SHARD, "documents": len(docs), "bytes": sum(len(d.encode()) for d in docs)},
           "patterns": arms, "results": []}
    print(out["training"], flush=True)
    for arm, regex in arms.items():
        for size in SIZES:
            t, start = untrained(regex), time.time()
            t.train_from_iterator(docs, trainers.BpeTrainer(
                vocab_size=size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
            t.save(str(DATA / f"pattern_fix/{arm}-{size // 1000}k.json"))
            tok = PreTrainedTokenizerFast(tokenizer_object=t)
            for name, texts in slices.items():
                out["results"].append({"arm": arm, "vocab_size": t.get_vocab_size(), "slice": name,
                                       "floor_tokens_per_word": floor(t, texts), **measure(tok, texts)})
                r = out["results"][-1]
                print(f"{arm} {size} {name}: {r['tokens_per_word']:.2f} tok/word (floor {r['floor_tokens_per_word']:.2f}),"
                      f" byte cuts {r['byte_cut_words']:.3f}, {time.time() - start:.0f}s", flush=True)
            json.dump(out, open(ROOT / "results/experiments/pattern_fix.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    (DATA / "pattern_fix").mkdir(parents=True, exist_ok=True)
    main()
