"""Round 4 of the Tamil tokenizer plan (phase B), first arms: R4-switch.

SuperBPE found that where the superword stage starts changes task scores. Round 3's grapheme-super-64k
switched at 87.5% of merges (56k whole-letter merges, then 8k superword merges). R4-switch builds the same
tokenizer switching at 60%, 75% and 90%, and with no superword stage at all (r4-grapheme-64k; phase A's best 64k
was r2-grapheme-64k, whole letters without superwords), on round 3's exact text and sample, with the same 4-word cap.
Scored like round 3, with r3-grapheme-super-64k as the reference row. The 2-3 best round 4 arms go to the
GPU (85M bits per byte, 3 seeds).

    nice -n 10 uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/tamil_tokenizer_r4.py

Writes results/experiments/tamil_tokenizer_r4.json and data/tamil_tokenizer/r4-switch*-64k.json.
"""
import json
import time

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import tamil_tokenizer as r1
import tamil_tokenizer_r2 as r2
from tamil_tokenizer_r3 import capped_superword, dedupe
from uraikal.tokenizer_eval import measure

SIZE = 64_000
SWITCH = (0.60, 0.75, 0.90, 1.0)  # share of merges before the superword stage starts; round 3 used 0.875, 1.0 = no superwords


def main():
    start = time.time()
    # the text, the held-out slices and the stage 2 sample exactly as round 3 built them: same order of rng calls
    tamil = [x for i in range(5) for x in r2.sangraha(i)] + r2.wikipedia() + r2.madurai()
    tamil, dropped = dedupe(tamil)
    tb = sum(len(x.encode()) for x in tamil)
    need = {k: int(tb * s / (1 - sum(r2.SHARES.values()))) for k, s in r2.SHARES.items()}
    _, eng = r1.english(need["english"])
    roman, _ = r2.romanized_text(need["romanized"])
    mix = tamil + eng + r1.code(need["code"]) + roman
    held = json.load(open(r1.DATA / "bpb/eval.json", encoding="utf-8"))
    slices = {k: held[k] for k in ("D4.web", "D4.literature")}
    other = {k: v for k, v in held.items() if k not in slices} | {"json": r2.tool_calls()}
    out = {"training": {"tamil_bytes": tb, "repeated_lines_dropped": dropped, "mix_bytes": sum(len(x.encode()) for x in mix),
                        "documents": len(mix), "size": SIZE, "switch_points": SWITCH}, "results": []}
    assert tb == 5_623_680_400, f"Tamil text differs from round 3's ({tb} bytes): results would not be comparable"
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
        json.dump(out, open(r1.ROOT / "results/experiments/tamil_tokenizer_r4.json", "w"), indent=1, ensure_ascii=False)

    fast = lambda t: PreTrainedTokenizerFast(tokenizer_object=t)
    score("r3-grapheme-super-64k", fast(Tokenizer.from_file(str(r1.OUT / "r3-grapheme-super-64k.json"))), "round 3, switch 0.875")
    pattern = r1.pattern()
    word = r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    super_pattern = pattern.replace(word, word + r"(?: [\p{L}\p{M}]+)*")
    small = r2.take_random(mix, 3e8)
    for share in SWITCH:
        name, base = (f"r4-switch{round(share * 100)}-64k" if share < 1 else "r4-grapheme-64k"), int(SIZE * share)
        path = r1.OUT / f"{name}.json"
        if path.exists():  # built on an earlier run: score it again, don't rebuild
            t = Tokenizer.from_file(str(path))
        else:
            t = capped_superword(r2.grapheme_bpe(mix, base, pattern), small, SIZE - base, pattern, super_pattern) \
                if share < 1 else r2.grapheme_bpe(mix, SIZE, pattern)  # whole letters, deduplicated text, no superwords
            t.save(str(path))
        score(name, fast(t), f"round 4, switch {share}")


if __name__ == "__main__":
    main()
