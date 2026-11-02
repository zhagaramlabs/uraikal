"""Round 3 of the Tamil tokenizer plan: fix what round 2 showed, and combine its two best ideas.

Round 2's superword tokenizers learned a whole boilerplate sentence (Tamil Wikipedia's bot-written temple
stubs) as one token. Round 3, on round 2's text:
- drops repeated lines: only the first copy of each line of 40 characters or more is kept;
- caps superword tokens at 4 words: stage 2 learns extra merges, drops any that would join more than
  4 words (and any built from a dropped one), and keeps the first ones that fit;
- combines whole-letter units (no boundary inside a letter) with a superword stage 2.
Arms: bpe-64k (the effect of deduplication alone), super-64k, grapheme-super-64k, super-128k and
grapheme-super-128k. Scored like round 2.

    uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/tamil_tokenizer_r3.py

Writes results/experiments/tamil_tokenizer_r3.json and the tokenizers to data/tamil_tokenizer/r3-*.json.
"""
import json
import time

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import tamil_tokenizer as r1
import tamil_tokenizer_r2 as r2
from uraikal.tokenizer_eval import load, measure

MAX_WORDS = 4


def dedupe(docs, min_chars=40):
    """Keep the first copy of every line of min_chars or more; shorter lines are kept as they are."""
    seen, out, dropped = set(), [], 0
    for doc in docs:
        kept = []
        for line in doc.split("\n"):
            if len(line) >= min_chars:
                h = hash(line)
                if h in seen:
                    dropped += 1
                    continue
                seen.add(h)
            kept.append(line)
        if any(kept):
            out.append("\n".join(kept))
    return out, dropped


def words_in(token):
    return len(token.strip().split(" ")) if token.strip() else 0


def capped_superword(t1, texts, n_new, pattern, super_pattern):
    """r2.superword with no token joining more than MAX_WORDS words: learn half as many merges again,
    then keep the first n_new that fit, skipping any built from a skipped token."""
    stage2 = json.loads(r2.superword(t1, texts, int(n_new * 1.5), pattern, super_pattern).to_str())["model"]["merges"]
    vocab, merges = r2.spec(t1)
    base, skipped = len(merges), set()
    for a, b in (tuple(m) for m in stage2[base:]):
        if len(merges) - base == n_new:
            break
        if a in skipped or b in skipped or words_in(a + b) > MAX_WORDS:
            skipped.add(a + b)
            continue
        merges.append((a, b))
        vocab.setdefault(a + b, len(vocab))
    return r2.bpe_from(vocab, merges, super_pattern)


def main():
    start = time.time()
    tamil = [x for i in range(5) for x in r2.sangraha(i)] + r2.wikipedia() + r2.madurai()
    tamil, dropped = dedupe(tamil)
    tb = sum(len(x.encode()) for x in tamil)
    need = {k: int(tb * s / (1 - sum(r2.SHARES.values()))) for k, s in r2.SHARES.items()}
    eng_slice, eng = r1.english(need["english"])
    roman, n_pairs = r2.romanized_text(need["romanized"])
    mix = tamil + eng + r1.code(need["code"]) + roman
    held = json.load(open(r1.DATA / "bpb/eval.json", encoding="utf-8"))  # same slices as round 2 and step 2
    slices = {k: held[k] for k in ("D4.web", "D4.literature")}
    other = {k: v for k, v in held.items() if k not in slices} | {"json": r2.tool_calls()}
    out = {"training": {"tamil_bytes": tb, "repeated_lines_dropped": dropped, "mix_bytes": sum(len(x.encode()) for x in mix),
                        "documents": len(mix), "max_words_per_token": MAX_WORDS}, "results": []}
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
        json.dump(out, open(r1.ROOT / "results/experiments/tamil_tokenizer_r3.json", "w"), indent=1, ensure_ascii=False)

    fast = lambda t: PreTrainedTokenizerFast(tokenizer_object=t)
    score("murai-labs/agaram-2.5b", load("murai-labs/agaram-2.5b"), "baseline")
    for name in ("r2-super-64k", "r2-grapheme-64k", "r2-super-128k"):
        score(name, fast(Tokenizer.from_file(str(r1.OUT / f"{name}.json"))), "round 2")
    pattern = r1.pattern()
    word = r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    super_pattern = pattern.replace(word, word + r"(?: [\p{L}\p{M}]+)*")
    small = r2.take_random(mix, 3e8)
    arms = [
        ("r3-bpe-64k", lambda: r1.char_bpe(mix, 64_000, pattern)),
        ("r3-super-64k", lambda: capped_superword(r1.char_bpe(mix, 56_000, pattern), small, 8_000, pattern, super_pattern)),
        ("r3-grapheme-super-64k", lambda: capped_superword(r2.grapheme_bpe(mix, 56_000, pattern), small, 8_000, pattern, super_pattern)),
        ("r3-super-128k", lambda: capped_superword(r1.char_bpe(mix, 112_000, pattern), small, 16_000, pattern, super_pattern)),
        ("r3-grapheme-super-128k", lambda: capped_superword(r2.grapheme_bpe(mix, 112_000, pattern), small, 16_000, pattern, super_pattern)),
    ]
    for name, build in arms:
        t = build()
        t.save(str(r1.OUT / f"{name}.json"))
        score(name, fast(t), "round 3")


if __name__ == "__main__":
    main()
