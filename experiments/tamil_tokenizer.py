"""Tamil tokenizer candidates that should top D4 fairly: step 1 of the lab's tokenizer plan (2026-11-02).

Every candidate normalises to NFC and splits words with Qwen3.5's mark-aware pattern, so no vowel sign
is ever cut off before training (P1 section 6):
- ta-char-64k: character-level BPE with byte fallback, Tamil only. Same text and size as pattern_fix's
  mark-aware 64k arm, so the two differ only in byte-level vs character-level units.
- mix-64k, mix-128k: the same recipe on 65% Tamil, 25% English and 10% code, so the vocabulary
  serves a real model. mix80-64k: 80% Tamil, 15% English, 5% code, to find what beats Agaram at 64k.
- qwen3.5+ta-32k: Qwen3.5's tokenizer with 32k merges learned on Tamil words only, continuing its
  own BPE. Every original token and merge is kept, so English tokenizes as before.
Scored on D4.web and D4.literature with uraikal's measure(), and by tokens per word on English and
romanized Tamil, where Agaram, Gemma 3 and Qwen3.5 are the baselines.

Text:
- Tamil: Sangraha verified shard data-0 (CC BY 4.0), as in pattern_fix; D4.web comes from data-52.
- English: FineWeb sample-10BT (ODC-By 1.0), read in row groups. Its first 2,000 documents are the
  English slice and are never trained on.
- Code: the Python source of this machine's standard library and the project's installed packages.
- Romanized Tamil slice: Dakshina v1.0 (Google, CC BY-SA 4.0), Tamil romanized dev sentences.

    uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/tamil_tokenizer.py

Writes results/experiments/tamil_tokenizer.json and the tokenizers to data/tamil_tokenizer/.
"""
import io
import json
import sysconfig
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from uraikal.text import tamil_share
from uraikal.tokenizer_eval import WORD, load, measure

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "tamil_tokenizer"
TAMIL = "verified/tam/data-0.parquet"
FINEWEB = "datasets/HuggingFaceFW/fineweb/sample/10BT/000_00000.parquet"
DAKSHINA = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
MIXES = {"mix": (0.25, 0.10), "mix80": (0.15, 0.05)}  # English and code shares of the text; Tamil is the rest
BASELINES = ["murai-labs/agaram-2.5b", "google/gemma-3-4b-it", "Qwen/Qwen3.5-0.8B"]


def pattern():
    split = json.loads(load("Qwen/Qwen3.5-0.8B").backend_tokenizer.to_str())["pre_tokenizer"]["pretokenizers"][0]
    return split["pattern"]["Regex"]


def english(n_bytes, n_slice=2000):
    """The English slice, then about n_bytes of training text from later documents."""
    slice_, train, size = [], [], 0
    with HfFileSystem().open(FINEWEB, "rb") as f:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1000, columns=["text"]):
            for text in batch.column(0).to_pylist():
                if len(slice_) < n_slice:
                    slice_.append(text)
                elif size < n_bytes:
                    train.append(text)
                    size += len(text.encode())
                else:
                    return slice_, train


def code(n_bytes):
    files = [p for key in ("stdlib", "purelib") for p in sorted(Path(sysconfig.get_paths()[key]).rglob("*.py"))]
    out, size = [], 0
    for p in files:
        if size >= n_bytes:
            break
        text = p.read_text(encoding="utf-8", errors="ignore")
        out.append(text)
        size += len(text.encode())
    return out


def romanized(n=2000):
    """Dakshina's Tamil romanized dev sentences; the 2 GB archive is streamed and only Tamil kept."""
    path = DATA / "dakshina/ta.romanized.rejoined.dev.roman.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DAKSHINA) as r, tarfile.open(fileobj=r, mode="r|") as tar:
            for m in tar:
                if m.name.endswith("/ta/romanized/" + path.name):
                    path.write_bytes(tar.extractfile(m).read())
                    break
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) >= n // 2, len(lines)
    return lines[:n]


def char_bpe(texts, size, regex):
    """Character-level BPE with byte fallback: a token can only end mid-character on an unseen character."""
    t = Tokenizer(models.BPE())
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Split(Regex(regex), behavior="isolated")
    t.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=size - 256, limit_alphabet=1000, show_progress=False))
    m = json.loads(t.to_str())["model"]
    vocab = {f"<0x{b:02X}>": b for b in range(256)}
    vocab.update({tok: i + 256 for tok, i in m["vocab"].items()})
    merges = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in m["merges"]]
    out = Tokenizer(models.BPE(vocab, merges, byte_fallback=True))
    out.normalizer, out.pre_tokenizer = t.normalizer, t.pre_tokenizer
    out.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    return out


def extend(base_name, texts, n_new):
    """Continue the base tokenizer's BPE on Tamil: base merges keep priority, new ones are appended.

    Each base token used by the Tamil text becomes one private-use character, so the fast Rust trainer
    can learn merges over base tokens; the merges are then mapped back to base token strings."""
    bt = load(base_name).backend_tokenizer
    spec = json.loads(bt.to_str())
    words = Counter()
    for text in texts:
        if bt.normalizer:
            text = bt.normalizer.normalize_str(text)
        words.update(p for p, _ in bt.pre_tokenizer.pre_tokenize_str(text))
    # Learn from Tamil words only: merges over the English inside Tamil documents would change English.
    words = Counter({w: n for w, n in words.items() if tamil_share(bt.decoder.decode([w])) > 0.5})
    char_of, token_of, encoded = {}, {}, {}
    for w in words:
        toks = [t.value for t in bt.model.tokenize(w)]
        for t in toks:
            if t not in char_of:
                c = chr(0xF0000 + len(char_of))
                char_of[t], token_of[c] = c, t
        encoded[w] = "".join(char_of[t] for t in toks)
    assert len(char_of) < 0xFFFE, len(char_of)  # supplementary private use area A
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t.train_from_iterator((" ".join([encoded[w]] * n) for w, n in words.items()), trainers.BpeTrainer(
        vocab_size=len(char_of) + n_new, initial_alphabet=list(token_of), limit_alphabet=len(token_of), show_progress=False))
    learned = [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in json.loads(t.to_str())["model"]["merges"]]
    vocab, merges = spec["model"]["vocab"], spec["model"]["merges"]
    as_str = isinstance(merges[0], str)
    # Special tokens missing from the model vocab get renumbered on load, onto our new ids; pin them.
    for a in spec.get("added_tokens", []):
        vocab.setdefault(a["content"], a["id"])
    next_id = max(vocab.values()) + 1
    for a, b in learned:
        a, b = "".join(token_of[c] for c in a), "".join(token_of[c] for c in b)
        merges.append(f"{a} {b}" if as_str else [a, b])
        if a + b not in vocab:
            vocab[a + b] = next_id
            next_id += 1
    return Tokenizer.from_str(json.dumps(spec))


def take(texts, n_bytes):
    out, size = [], 0
    for x in texts:
        if size >= n_bytes:
            break
        out.append(x)
        size += len(x.encode())
    return out


def tokens_per_word(tok, texts):
    return sum(len(tok(x, add_special_tokens=False)["input_ids"]) for x in texts) / sum(len(WORD.findall(x)) for x in texts)


def main():
    start = time.time()
    tamil = [x for x in pq.read_table(hf_hub_download("ai4bharat/sangraha", TAMIL, repo_type="dataset"),
                                      columns=["text"]).column(0).to_pylist() if x]
    tamil_bytes = sum(len(x.encode()) for x in tamil)
    need = lambda i: max(int(tamil_bytes * m[i] / (1 - sum(m))) for m in MIXES.values())
    eng_slice, eng = english(need(0))
    py = code(need(1))
    mixed = {name: tamil + take(eng, tamil_bytes * e / (1 - e - c)) + take(py, tamil_bytes * c / (1 - e - c))
             for name, (e, c) in MIXES.items()}
    slices = {f"D4.{s}": [json.loads(line)["text"] for line in open(DATA / f"d4_{s}.jsonl", encoding="utf-8")]
              for s in ("web", "literature")}
    other = {"english": eng_slice, "romanized": romanized()}
    out = {"training": {name: {"documents": len(x), "bytes": sum(len(d.encode()) for d in x)}
                        for name, x in (("tamil", tamil), ("english", eng), ("code", py))},
           "sources": {"tamil": f"ai4bharat/sangraha {TAMIL}", "english": FINEWEB, "code": "local Python stdlib + purelib",
                       "romanized": DAKSHINA},
           "mixes": {name: {"english": e, "code": c, "tamil": round(1 - e - c, 2)} for name, (e, c) in MIXES.items()},
           "results": []}
    print(out["training"], f"{time.time() - start:.0f}s", flush=True)

    def score(name, tok, kind):
        row = {"tokenizer": name, "kind": kind, "vocab_size": len(tok)}
        for s, texts in slices.items():
            row[s] = measure(tok, texts)
        for s, texts in other.items():
            row[s] = {"tokens_per_word": tokens_per_word(tok, texts), "passages": len(texts)}
        out["results"].append(row)
        print(f"{name}: web {row['D4.web']['tokens_per_word']:.2f}, lit {row['D4.literature']['tokens_per_word']:.2f},"
              f" byte cuts web/lit {row['D4.web']['byte_cut_words']:.4f}/{row['D4.literature']['byte_cut_words']:.4f},"
              f" english {row['english']['tokens_per_word']:.2f}, romanized {row['romanized']['tokens_per_word']:.2f},"
              f" {time.time() - start:.0f}s", flush=True)
        json.dump(out, open(ROOT / "results/experiments/tamil_tokenizer.json", "w"), indent=1, ensure_ascii=False)

    for name in BASELINES:
        score(name, load(name), "baseline")
    regex = pattern()
    for name, texts, size in (("ta-char-64k", tamil, 64_000), ("mix-64k", mixed["mix"], 64_000),
                              ("mix-128k", mixed["mix"], 128_000), ("mix80-64k", mixed["mix80"], 64_000)):
        t = char_bpe(texts, size, regex)
        t.save(str(OUT / f"{name}.json"))
        score(name, PreTrainedTokenizerFast(tokenizer_object=t), "candidate")
    t = extend("Qwen/Qwen3.5-0.8B", tamil, 32_000)
    t.save(str(OUT / "qwen3.5+ta-32k.json"))
    score("qwen3.5+ta-32k", PreTrainedTokenizerFast(tokenizer_object=t), "candidate")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
