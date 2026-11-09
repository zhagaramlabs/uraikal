"""Round 2 of the Tamil tokenizer plan: the levers that could beat Agaram by a wide margin.

Same base recipe as tamil_tokenizer.py (character-level BPE with byte fallback, Qwen3.5's mark-aware
pattern, NFC), on broader text, with new kinds of arm:
- bpe-64k, bpe-128k: the round 1 recipe on round 2's text, to separate the effect of the data.
- super-64k, super-128k: SuperBPE (Liu et al. 2025). Stage 1 is BPE to 56k (112k); stage 2 continues it on a
  0.3 GB sample, on chunks of words joined by single spaces, so tokens may cross word boundaries.
- grapheme-64k: BPE over whole letters (grapheme clusters), so no token boundary falls inside a letter.
- unigram-64k: a Unigram language model instead of BPE, trained on a 0.6 GB sample.
Text, 75% Tamil, 15% English, 5% code, 5% romanized Tamil:
- Tamil: Sangraha verified data-0 to data-4 (CC BY 4.0), Tamil Wikipedia 20231101 (CC BY-SA), and
  Project Madurai without the D4.literature lines and without the books that quote the Thirukkural.
- English: FineWeb sample-10BT after its first 2,000 documents (the English slice). Code: local Python.
- Romanized Tamil: Sangraha data-5 lines romanized word by word with Aksharantar's pairs (CC BY 4.0),
  leaving out its Dakshina-sourced pairs, with a rule-based fallback for unseen words.
Scored on D4.web and D4.literature with measure(), and by tokens per word on Thirukkural verse, English,
romanized Tamil (Dakshina dev), code (this repository and oosai, never trained on) and tool-call JSON.

    uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/tamil_tokenizer_r2.py

Writes results/experiments/tamil_tokenizer_r2.json and the tokenizers to data/tamil_tokenizer/r2-*.json.
"""
import json
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import regex
from huggingface_hub import hf_hub_download
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

import tamil_tokenizer as r1
from uraikal.classics import kural_text
from uraikal.text import clean, tamil_share
from uraikal.tokenizer_eval import load, measure

ROOT, DATA, OUT = r1.ROOT, r1.DATA, r1.OUT
SHARES = {"english": 0.15, "code": 0.05, "romanized": 0.05}  # Tamil is the rest
WIKI = ["20231101.ta/train-00000-of-00002.parquet", "20231101.ta/train-00001-of-00002.parquet"]
CLUSTER = regex.compile(r"\X")
rng = random.Random(0)


def sangraha(i):
    path = hf_hub_download("ai4bharat/sangraha", f"verified/tam/data-{i}.parquet", repo_type="dataset")
    return [x for x in pq.read_table(path, columns=["text"]).column(0).to_pylist() if x]


def wikipedia():
    return [x for f in WIKI for x in pq.read_table(hf_hub_download("wikimedia/wikipedia", f, repo_type="dataset"),
                                                   columns=["text"]).column(0).to_pylist() if x]


def madurai():
    """Project Madurai minus the D4.literature lines (prep.py's d4 sampled them) and the Thirukkural books."""
    held = {json.loads(line)["text"] for line in open(DATA / "d4_literature.jsonl", encoding="utf-8")}
    firsts = [re.sub(r"\s", "", json.loads(line)["line1"]) for line in open(DATA / "kural.jsonl", encoding="utf-8")][:20]
    out = []
    for doc in pq.read_table(DATA / "madurai/project-madurai.parquet", columns=["text"]).column(0).to_pylist():
        flat = re.sub(r"\s", "", doc or "")
        if not doc or sum(k in flat for k in firsts) >= 3:
            continue
        out.append("\n".join(line for line in doc.split("\n") if clean(line).strip() not in held))
    return out


VOWELS = dict(zip("அஆஇஈஉஊஎஏஐஒஓஔ", "a aa i ee u oo e e ai o o au".split()))
SIGNS = dict(zip("ாிீுூெேைொோௌ", "aa i ee u oo e e ai o o au".split()))
CONS = dict(zip("கஙசஞடணதநபமயரலவழளறனஜஷஸஹ", "k ng s nj d n th n p m y r l v zh l r n j sh s h".split()))


def rule_roman(word):
    out = []
    for i, c in enumerate(word):
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if c in CONS:
            out.append(CONS[c] + ("" if nxt in SIGNS or nxt == "்" else "a"))
        elif c != "்":
            out.append(SIGNS.get(c) or VOWELS.get(c) or ("h" if c == "ஃ" else c))
    return "".join(out)


def romanized_text(n_bytes):
    """Synthetic Tanglish: Sangraha data-5 lines, each Tamil word swapped for a romanization people use."""
    z = zipfile.ZipFile(hf_hub_download("ai4bharat/Aksharantar", "tam.zip", repo_type="dataset"))
    seen = defaultdict(Counter)
    with z.open("tam_train.json") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] != "Dakshina":  # Dakshina's romanized sentences are a test slice
                seen[r["native word"]][r["english word"]] += 1
    choices = {w: (list(c), list(c.values())) for w, c in seen.items()}

    def roman(word):
        core = word.strip(".,;:!?\"'()[]")
        if not core or tamil_share(core) == 0:
            return word
        r = rng.choices(*choices[core])[0] if core in choices else rule_roman(core)
        return word.replace(core, r)

    out, size = [], 0
    for doc in sangraha(5):
        for line in doc.split("\n"):
            if tamil_share(line) < 0.8 or len(line.split()) < 5:
                continue
            out.append(" ".join(roman(w) for w in line.split()))
            size += len(out[-1])
            if size >= n_bytes:
                return out, len(choices)
    return out, len(choices)


def pua(k):
    """The k-th private-use character, across planes 15 and 16."""
    return chr(0xF0000 + k) if k < 0xFFFE else chr(0x100000 + k - 0xFFFE)


def bpe_from(vocab, merges, pattern):
    t = Tokenizer(models.BPE(vocab, [tuple(m) for m in merges], byte_fallback=True))
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    t.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    return t


def spec(t):
    m = json.loads(t.to_str())["model"]
    return m["vocab"], [tuple(x.split(" ", 1)) if isinstance(x, str) else tuple(x) for x in m["merges"]]


def train_over_symbols(words, n_new):
    """Learn n_new BPE merges over words already written as private-use symbols, with their counts."""
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    alphabet = sorted({c for w in words for c in w})
    t.train_from_iterator((" ".join([w] * n) for w, n in words.items()), trainers.BpeTrainer(
        vocab_size=len(alphabet) + n_new, initial_alphabet=alphabet, limit_alphabet=len(alphabet), show_progress=False))
    return spec(t)[1]


def superword(t1, texts, n_new, pattern, super_pattern):
    """SuperBPE stage 2: continue t1's merges on chunks of words joined by single spaces.

    Stage 1 never merged across a word boundary, so applying its merges to a whole chunk gives the
    same tokens as word by word; stage 2's merges, ranked after them, are free to cross spaces."""
    p1, chunker = t1.pre_tokenizer, pre_tokenizers.Split(Regex(super_pattern), behavior="isolated")
    sym, tok_of, cache, chunks = {}, {}, {}, Counter()

    def symbols(piece):
        if piece not in cache:
            toks = [t.value for t in t1.model.tokenize(piece)]
            if any(t.startswith("<0x") for t in toks):
                cache[piece] = None  # byte-fallback pieces stay out of stage 2
            else:
                for t in toks:
                    if t not in sym:
                        sym[t] = pua(len(sym))
                        tok_of[sym[t]] = t
                cache[piece] = "".join(sym[t] for t in toks)
        return cache[piece]

    for text in texts:
        for chunk, _ in chunker.pre_tokenize_str(t1.normalizer.normalize_str(text)):
            parts = [symbols(p) for p, _ in p1.pre_tokenize_str(chunk)]
            if parts and None not in parts:
                chunks["".join(parts)] += 1
    vocab, merges = spec(t1)
    for a, b in train_over_symbols(chunks, n_new):
        a, b = "".join(tok_of[c] for c in a), "".join(tok_of[c] for c in b)
        merges.append((a, b))
        vocab.setdefault(a + b, len(vocab))
    return bpe_from(vocab, merges, super_pattern)


def grapheme_bpe(texts, size, pattern, cut=None):
    """BPE whose units are whole letters: first merges build each letter from its code points, then
    learned merges join only whole letters, so no token boundary falls inside a letter.
    cut(letters) -> lists of letters: training sees each part as its own word, so no merge joins two parts."""
    pre, norm = pre_tokenizers.Split(Regex(pattern), behavior="isolated"), normalizers.NFC()
    pieces = Counter(p for text in texts for p, _ in pre.pre_tokenize_str(norm.normalize_str(text)))
    letters = Counter()
    for p, n in pieces.items():
        for g in CLUSTER.findall(p):
            letters[g] += n
    kept = [g for g, _ in letters.most_common(4000)]
    sym = {g: pua(i) for i, g in enumerate(kept)}
    letter_of = {c: g for g, c in sym.items()}
    vocab = {f"<0x{b:02X}>": b for b in range(256)}
    build = []  # merges that assemble each multi-code-point letter, left to right
    for g in kept:
        for ch in g:
            vocab.setdefault(ch, len(vocab))
        for i in range(2, len(g) + 1):
            if g[:i] not in vocab:
                build.append((g[:i - 1], g[i - 1]))
                vocab[g[:i]] = len(vocab)
    words = Counter()
    for p, n in pieces.items():
        gs = CLUSTER.findall(p)
        if all(g in sym for g in gs):
            for part in cut(gs) if cut else [gs]:
                words["".join(sym[g] for g in part)] += n
    learned = train_over_symbols(words, size - len(vocab))
    merges = list(build)
    for a, b in learned:
        a, b = "".join(letter_of[c] for c in a), "".join(letter_of[c] for c in b)
        merges.append((a, b))
        vocab.setdefault(a + b, len(vocab))
    return bpe_from(vocab, merges, pattern)


def unigram(texts, size, pattern):
    t = Tokenizer(models.Unigram())
    t.normalizer = normalizers.NFC()
    t.pre_tokenizer = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    t.train_from_iterator(texts, trainers.UnigramTrainer(vocab_size=size - 256, special_tokens=["<unk>"],
                                                         unk_token="<unk>", max_piece_length=16, show_progress=False))
    m = json.loads(t.to_str())["model"]
    floor = min(s for _, s in m["vocab"]) - 10
    out = Tokenizer(models.Unigram([tuple(v) for v in m["vocab"]] + [(f"<0x{b:02X}>", floor) for b in range(256)],
                                   unk_id=m["unk_id"], byte_fallback=True))
    out.normalizer, out.pre_tokenizer = t.normalizer, t.pre_tokenizer
    out.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    return out


def tool_calls(n=500):
    """Synthetic tool-call JSON with Tamil and English arguments."""
    places = ["சென்னை", "மதுரை", "கோயம்புத்தூர்", "திருநெல்வேலி", "யாழ்ப்பாணம்", "மட்டக்களப்பு", "Chennai", "Singapore"]
    things = ["மருந்து நினைவூட்டல்", "பேருந்து நேரம்", "மழை முன்னறிவிப்பு", "உர விலை", "ரயில் டிக்கெட்", "electricity bill"]
    out = []
    for i in range(n):
        args = {"place": rng.choice(places), "query": rng.choice(things), "date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                "count": rng.randint(1, 500)}
        out.append(json.dumps({"name": rng.choice(["search", "weather", "remind", "lookup_fare"]), "arguments": args},
                              ensure_ascii=False))
    return out


def take_random(texts, n_bytes):
    out, size = [], 0
    for i in rng.sample(range(len(texts)), len(texts)):
        if size >= n_bytes:
            break
        out.append(texts[i])
        size += len(texts[i].encode())
    return out


def main():
    start = time.time()
    tamil = [x for i in range(5) for x in sangraha(i)] + wikipedia() + madurai()
    tb = sum(len(x.encode()) for x in tamil)
    need = {k: int(tb * s / (1 - sum(SHARES.values()))) for k, s in SHARES.items()}
    eng_slice, eng = r1.english(need["english"])
    roman, n_pairs = romanized_text(need["romanized"])
    mix = tamil + eng + r1.code(need["code"]) + roman
    code_slice = [p.read_text(encoding="utf-8") for d in (ROOT / "src", ROOT / "tests", ROOT.parent / "oosai/agent",
                                                          ROOT.parent / "oosai/eval") for p in sorted(d.rglob("*.py"))]
    slices = {f"D4.{s}": [json.loads(line)["text"] for line in open(DATA / f"d4_{s}.jsonl", encoding="utf-8")]
              for s in ("web", "literature")}
    other = {"verse": [kural_text(json.loads(line)) for line in open(DATA / "kural.jsonl", encoding="utf-8")],
             "english": eng_slice, "romanized": r1.romanized(), "code": code_slice, "json": tool_calls()}
    out = {"training": {"tamil_bytes": tb, "mix_bytes": sum(len(x.encode()) for x in mix), "documents": len(mix),
                        "romanization_pairs": n_pairs, "shares": SHARES}, "results": []}
    print(out["training"], f"{time.time() - start:.0f}s", flush=True)

    def score(name, tok, kind):
        row = {"tokenizer": name, "kind": kind, "vocab_size": len(tok)}
        for s, texts in slices.items():
            row[s] = measure(tok, texts)
        for s, texts in other.items():
            row[s] = {"tokens_per_word": r1.tokens_per_word(tok, texts), "passages": len(texts)}
        out["results"].append(row)
        print(f"{name}: web {row['D4.web']['tokens_per_word']:.3f} lit {row['D4.literature']['tokens_per_word']:.3f} "
              + " ".join(f"{s} {row[s]['tokens_per_word']:.2f}" for s in other)
              + f" letter cut web {row['D4.web']['letter_cut_words']:.3f}, {time.time() - start:.0f}s", flush=True)
        json.dump(out, open(ROOT / "results/experiments/tamil_tokenizer_r2.json", "w"), indent=1, ensure_ascii=False)

    def fast(t):
        return PreTrainedTokenizerFast(tokenizer_object=t)

    for name in r1.BASELINES:
        score(name, load(name), "baseline")
    for name in ("mix80-64k", "mix-128k", "qwen3.5+ta-32k"):
        score(name, fast(Tokenizer.from_file(str(OUT / f"{name}.json"))), "round 1")
    pattern = r1.pattern()
    word = r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    assert word in pattern
    super_pattern = pattern.replace(word, word + r"(?: [\p{L}\p{M}]+)*")
    sample, small = take_random(mix, 6e8), take_random(mix, 3e8)
    arms = [
        ("r2-bpe-64k", lambda: r1.char_bpe(mix, 64_000, pattern)),
        ("r2-super-64k", lambda: superword(r1.char_bpe(mix, 56_000, pattern), small, 8_000, pattern, super_pattern)),
        ("r2-bpe-128k", lambda: r1.char_bpe(mix, 128_000, pattern)),
        ("r2-super-128k", lambda: superword(r1.char_bpe(mix, 112_000, pattern), small, 16_000, pattern, super_pattern)),
        ("r2-grapheme-64k", lambda: grapheme_bpe(mix, 64_000, pattern)),
        ("r2-unigram-64k", lambda: unigram(sample, 64_000, pattern)),
    ]
    for name, build in arms:
        t = build()
        t.save(str(OUT / f"{name}.json"))
        score(name, fast(t), "round 2")


if __name__ == "__main__":
    main()
