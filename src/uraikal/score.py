"""Speech recognition scoring (track A): WER and CER per slice, over normalised text.

Spec: SPEC.md ("Scoring" and "Leaderboard").
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

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


def same_setup(out, setup):
    """A resumable output (hyps JSONL) must come from one setup: a rerun skips every id already in it, so a rerun
    with another model or --max-tokens would mix two setups under one score. Records `setup` beside `out` on the
    first run and stops a rerun whose setup differs."""
    record, setup = Path(f"{out}.setup.json"), json.loads(json.dumps(setup))
    if record.exists():
        old = json.loads(record.read_text(encoding="utf-8"))
        if old != setup:
            raise SystemExit(f"{out} was made with {old}, not {setup}. Use another --system or --out, "
                             f"or delete {out} and {record.name} to start over")
        return
    if Path(out).exists():  # made before setups were recorded: trust it, as before
        print(f"note: {out} has no setup record; assuming it was made with {setup}", file=sys.stderr)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(setup, ensure_ascii=False) + "\n", encoding="utf-8")


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
