"""Recount a tokenizer's full training stream and compare it with what a finished run trained on.
Equal means the run's token cache was complete. Run from uraikal/: python experiments/spot/recount.py <tokenizer> <result.json>"""
import gzip
import json
import sys

sys.path.insert(0, "experiments")
import tokenizer_bpb as t

spec, result = sys.argv[1], sys.argv[2]
backend, n = t.tokenizer(spec)
total = docs = 0
with gzip.open(t.BPB / "train.jsonl.gz", "rt", encoding="utf-8") as f:
    batch = []
    for line in f:
        batch.append(json.loads(line)["text"])
        if len(batch) == 10_000:
            total += sum(1 + len(ids) for ids in t.encode(backend, batch))  # 1 for our BOS, as tokens() writes it
            docs += len(batch)
            batch = []
    total += sum(1 + len(ids) for ids in t.encode(backend, batch))
    docs += len(batch)
trained = json.load(open(result))["training"]["stream_tokens"]
print(f"{spec}: {docs} documents, {total} tokens in full; the run's stream had {trained}:",
      "COMPLETE" if total == trained else f"SHORT by {total - trained}")
