"""Round 4, R4-morph: whole-letter BPE whose merges never join a stem to its suffix.

MorphBPE (Findings ACL 2026) blocks BPE merges at morpheme boundaries, and gained at 300M and 1B. For Tamil,
TamilMorph (Muthukumar, Herygers & Beinborn, DravidianLangTech 2026; MIT, Surendhar32/TamilMorph on Hugging Face)
splits 418k attested word forms into stem + suffix. Its cuts are on code points and often fall inside a letter
(ரம்பம|ாய்), so each moves to the start of that letter (ரம்ப|மாய்). A cut that leaves a one-letter stem is
skipped: those are mostly common words misanalysed (ம|க்கள், போ|து). Words TamilMorph doesn't list (about 3 in 4
running words) train as in r2-grapheme-64k. TamilTok's ByT5 segmenter isn't released, and a stem-plus-suffix guess
for unlisted words was wrong too often to use. Cuts apply only in training: the result is a plain BPE tokenizer.

Round 2's exact text, so the only difference from r2-grapheme-64k (phases A and B's best) is the cuts. A tenth of
TamilMorph's words is held out of training and scores Morph-F1: how often token boundaries fall where the stem ends
(Creutz & Linden 2004), as TamilTok reports it.

    nice -n 10 uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/tamil_tokenizer_morph.py

Writes results/experiments/tamil_tokenizer_morph.json and data/tamil_tokenizer/r4-morph-64k.json. With an argument,
`... tamil_tokenizer_morph.py 128000` builds r4-morph-128k instead (results in tamil_tokenizer_morph_128k.json).
"""
import csv
import hashlib
import itertools
import json
import sys
import time
import unicodedata

import regex
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import tamil_tokenizer as r1
import tamil_tokenizer_r2 as r2
from uraikal.classics import kural_text
from uraikal.tokenizer_eval import measure

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64_000  # 128000 builds r4-morph-128k
NAME = f"r4-morph-{SIZE // 1000}k"
RESULTS = "tamil_tokenizer_morph.json" if SIZE == 64_000 else f"tamil_tokenizer_morph_{SIZE // 1000}k.json"
LETTER = regex.compile(r"\p{L}")


def tamilmorph():
    """({word: letters in its stem} to train on, {word: code points in its stem, 0 if uncut} held out)."""
    path = hf_hub_download("Surendhar32/TamilMorph", "TamilMorph.csv", repo_type="dataset",
                           revision="e493f19d5353fed278ff4070ead9804ca0ecc6ac")
    train, held = {}, {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        w, stem = unicodedata.normalize("NFC", r["wordform"]), unicodedata.normalize("NFC", r["stem"])
        if not w.startswith(stem):
            continue
        ends = list(itertools.accumulate(map(len, r2.CLUSTER.findall(w))))
        b = sum(e <= len(stem) for e in ends)  # letters wholly in the stem: a cut inside a letter moves to its start
        b = b if 2 <= b < len(ends) else 0
        if int(hashlib.md5(w.encode()).hexdigest(), 16) % 10 == 0:
            held[w] = ends[b - 1] if b else 0
        elif b:
            train[w] = b
    return train, held


def cutter(table):
    def cut(letters):
        lead = 0 if LETTER.match(letters[0]) else 1  # the space or mark the pattern keeps before a word
        b = table.get("".join(letters[lead:]))
        return [letters[:lead + b], letters[lead + b:]] if b else [letters]
    return cut


def morph_f1(tok, held):
    """Token boundaries inside each held-out word against its stem-suffix cut, pooled over words."""
    words = list(held)
    hit = pred = gold = 0
    for w, enc in zip(words, tok.backend_tokenizer.encode_batch([" " + w for w in words], add_special_tokens=False)):
        p = {e - 1 for _, e in enc.offsets if 0 < e - 1 < len(w)}  # offsets count the leading space
        g = {held[w]} - {0}
        hit, pred, gold = hit + len(p & g), pred + len(p), gold + len(g)
    prec, rec = hit / pred, hit / gold
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec), "words": len(words)}


def check():
    """With a cut, no merge joins stem and suffix; with none, the build is round 2's grapheme BPE."""
    pattern, text = r1.pattern(), ["காவலனால் வந்தது " * 50]
    plain = r2.grapheme_bpe(text, 400, pattern)
    assert " காவலனால்" in plain.get_vocab()
    v = r2.grapheme_bpe(text, 400, pattern, cutter({"காவலனால்": 3})).get_vocab()
    assert " காவலனால்" not in v and " காவல" in v and "னால்" in v, sorted(v, key=len)[-5:]
    assert r2.grapheme_bpe(text, 400, pattern, cutter({})).to_str() == plain.to_str()


def main():
    start = time.time()
    check()
    train, held = tamilmorph()
    tamil = [x for i in range(5) for x in r2.sangraha(i)] + r2.wikipedia() + r2.madurai()
    tb = sum(len(x.encode()) for x in tamil)
    assert tb == 5_859_269_334, f"round 2's Tamil text was 5,859,269,334 bytes, this is {tb}: results would not be comparable"
    need = {k: int(tb * s / (1 - sum(r2.SHARES.values()))) for k, s in r2.SHARES.items()}
    eng_slice, eng = r1.english(need["english"])
    roman, _ = r2.romanized_text(need["romanized"])  # the first draw from r2.rng, as in round 2
    mix = tamil + eng + r1.code(need["code"]) + roman
    mb = sum(len(x.encode()) for x in mix)  # round 2's was 7,812,428,757: code comes from installed packages, which update
    code = [p.read_text(encoding="utf-8") for d in (r1.ROOT / "src", r1.ROOT / "tests", r1.ROOT.parent / "oosai/agent",
                                                    r1.ROOT.parent / "oosai/eval") for p in sorted(d.rglob("*.py"))]
    slices = {f"D4.{s}": [json.loads(line)["text"] for line in open(r1.DATA / f"d4_{s}.jsonl", encoding="utf-8")]
              for s in ("web", "literature")}
    other = {"verse": [kural_text(json.loads(line)) for line in open(r1.DATA / "kural.jsonl", encoding="utf-8")],
             "english": eng_slice, "romanized": r1.romanized(), "code": code, "json": r2.tool_calls()}
    out = {"training": {"mix_bytes": mb, "mix_bytes_vs_round_2": mb - 7_812_428_757, "tamilmorph_cuts": len(train), "held_out_words": len(held), "size": SIZE},
           "results": []}
    print(out["training"], f"{time.time() - start:.0f}s", flush=True)

    def score(name, tok, kind):
        row = {"tokenizer": name, "kind": kind, "vocab_size": len(tok), "morph_f1": morph_f1(tok, held)}
        for s, texts in slices.items():
            row[s] = measure(tok, texts)
        for s, texts in other.items():
            row[s] = {"tokens_per_word": r1.tokens_per_word(tok, texts), "passages": len(texts)}
        out["results"].append(row)
        print(f"{name}: morph F1 {row['morph_f1']['f1']:.3f} (P {row['morph_f1']['precision']:.3f} R {row['morph_f1']['recall']:.3f}) "
              f"web {row['D4.web']['tokens_per_word']:.3f} lit {row['D4.literature']['tokens_per_word']:.3f} "
              + " ".join(f"{s} {row[s]['tokens_per_word']:.2f}" for s in other) + f", {time.time() - start:.0f}s", flush=True)
        json.dump(out, open(r1.ROOT / "results/experiments" / RESULTS, "w"), indent=1, ensure_ascii=False)

    fast = lambda t: PreTrainedTokenizerFast(tokenizer_object=t)
    for name in ("r2-grapheme-64k", "r3-bpe-64k") if SIZE == 64_000 else ("r4-morph-64k", "r2-bpe-128k"):
        score(name, fast(Tokenizer.from_file(str(r1.OUT / f"{name}.json"))), "reference")
    t = r2.grapheme_bpe(mix, SIZE, r1.pattern(), cutter(train))
    t.save(str(r1.OUT / f"{NAME}.json"))
    score(NAME, fast(t), "round 4, TamilMorph cuts")


if __name__ == "__main__":
    main()
