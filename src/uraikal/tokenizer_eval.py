"""Track D4: how well a tokenizer handles Tamil text.

For each tokenizer and slice of held-out Tamil passages:
- tokens per word (whitespace words) and UTF-8 bytes per token, next to the vocabulary size;
- letter cuts: share of Tamil words with a token boundary between a consonant and its vowel sign
  or pulli, and the share already cut by the pre-tokenizer (the GPT-style `\\p{L}+` regex bug),
  which tells a broken pre-tokenizer apart from a vocabulary short of Tamil merges. A learned cut
  isn't always a defect: Tamil suffixes often start inside a letter (மரத்தை = மரத்த் + ஐ), so a
  split there can follow the grammar. Pre-tokenizer cuts and byte cuts are always defects;
- byte cuts: share of Tamil words where a token ends inside a character (byte-level BPE);
- unknown tokens per 1,000 tokens;
- round trip: share of passages that decode back to the same characters (whitespace ignored).

Needs the `tokenizers` extra: pip install "uraikal[tokenizers]".
"""
import re
import sys

from uraikal.text import MARKS, tamil_share

# The v0 roster: widely used open tokenizers, plus ByT5 (one token per byte) as the floor.
DEFAULT_TOKENIZERS = [
    "openai/gpt-oss-20b",  # o200k_harmony, the GPT-4o family
    "meta-llama/Llama-3.1-8B",
    "google/gemma-3-4b-it",
    "Qwen/Qwen3-8B",
    "deepseek-ai/DeepSeek-V3",
    "mistralai/Mistral-Nemo-Base-2407",
    "CohereLabs/aya-expanse-8b",
    "bigscience/bloom-560m",
    "sarvamai/sarvam-1",
    "sarvamai/sarvam-30b",
    "murai-labs/agaram-2.5b",
    "ai4bharat/IndicBERTv2-MLM-only",
    "google/muril-base-cased",
    "FacebookAI/xlm-roberta-base",
    "google/mt5-base",
]
WORD = re.compile(r"\S+")


def measure(tok, texts) -> dict:
    """D4 metrics for one Hugging Face fast tokenizer over a list of passages."""
    tokens = words = nbytes = tamil_words = letter_cut = pre_cut = byte_cut = unk = exact = 0
    pre_tokenizer = getattr(getattr(tok, "backend_tokenizer", None), "pre_tokenizer", None)
    for text in texts:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        tokens += len(ids)
        nbytes += len(text.encode("utf-8"))
        unk += sum(i == tok.unk_token_id for i in ids) if tok.unk_token_id is not None else 0
        decoded = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        exact += "".join(decoded.split()) == "".join(text.split())

        pre_cuts = set()
        if pre_tokenizer:
            spans = [span for _, span in pre_tokenizer.pre_tokenize_str(text)]
            pre_cuts = {s for s, _ in spans[1:] if s < len(text) and text[s] in MARKS}
        letter_cuts, byte_cuts = set(), set()
        for (s0, e0), (s1, e1) in zip(offsets, offsets[1:]):
            if s1 < e0:  # both tokens claim the same character: the boundary is inside it
                byte_cuts.add(s1)
            elif s1 < len(text) and text[s1] in MARKS:
                letter_cuts.add(s1)
        for m in WORD.finditer(text):
            words += 1
            if tamil_share(m.group()) == 0:
                continue
            tamil_words += 1
            s, e = m.span()
            letter_cut += any(s < p < e for p in letter_cuts)
            pre_cut += any(s < p < e for p in pre_cuts)
            byte_cut += any(s <= p < e for p in byte_cuts)
    return {
        "tokens_per_word": tokens / words,
        "bytes_per_token": nbytes / tokens,
        "letter_cut_words": letter_cut / tamil_words,
        "pretokenizer_letter_cut_words": pre_cut / tamil_words if pre_tokenizer else None,
        "byte_cut_words": byte_cut / tamil_words,
        "unk_per_1k": 1000 * unk / tokens,
        "round_trip": exact / len(texts),
        "passages": len(texts),
        "words": words,
    }


def load(name):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=False)
    if not tok.is_fast:
        raise ValueError("no fast tokenizer, so no character offsets")
    return tok


def run(names, slices: dict) -> list[dict]:
    """Score every tokenizer on every slice; a tokenizer that fails to load is reported and skipped."""
    records = []
    for name in names:
        try:
            tok = load(name)
        except Exception as e:  # gated repos, missing files, remote code we won't run
            print(f"skipped {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}", file=sys.stderr)
            continue
        for slice_name, texts in slices.items():
            records.append({"tokenizer": name, "vocab_size": len(tok), "slice": slice_name, **measure(tok, texts)})
    return records
