"""Speech recognition scoring (track A): WER and CER per slice, over normalised text.

Spec: SPEC.md ("Scoring" and "Leaderboard").
"""
import json
import sys
from collections import defaultdict

from uraikal.text import normalize

URAIKAL_VERSION = "0.1"


def edit_distance(ref, hyp) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def error_rates(pairs) -> dict:
    """Corpus-level WER and CER over (reference, hypothesis) pairs of normalised text.

    shortcut: CER counts Unicode code points (spaces included, as jiwer does), not aksharas;
    switch to grapheme clusters if we need to compare with papers reporting akshara error rate.
    """
    word_err = word_n = char_err = char_n = 0
    for ref, hyp in pairs:
        word_err += edit_distance(ref.split(), hyp.split())
        word_n += len(ref.split())
        char_err += edit_distance(ref, hyp)
        char_n += len(ref)
    return {"wer": word_err / word_n, "cer": char_err / char_n}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(manifest_rows, hyps: dict) -> dict:
    """Return {slice: {"wer", "cer", "n", "missing"}}. A missing hypothesis scores as empty output."""
    by_slice = defaultdict(list)
    missing = defaultdict(int)
    for row in manifest_rows:
        ref = normalize(row["text"])
        if not ref:
            print(f"skipping {row['id']}: empty reference after normalising", file=sys.stderr)
            continue
        if row["id"] not in hyps:
            missing[row["slice"]] += 1
        by_slice[row["slice"]].append((ref, normalize(hyps.get(row["id"], ""))))
    return {
        name: {**error_rates(pairs), "n": len(pairs), "missing": missing[name]}
        for name, pairs in sorted(by_slice.items())
    }
