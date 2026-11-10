"""Resume check for tokenizer_bpb.py: a run stopped after a checkpoint and started again ends exactly where an
unbroken run does. Needs no private data. Run: uv run --extra tokenizers --with torch python experiments/test_bpb_resume.py"""
import argparse
import gzip
import json
import random
import sys
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).parent))
import tokenizer_bpb as tb

if not torch.cuda.is_available():  # the CPU is bit-for-bit repeatable; Apple's MPS differs in the 9th digit run to run
    from transformers import LlamaForCausalLM  # noqa: F401  (imported first: transformers' own import checks MPS)
    torch.backends.mps.is_available = lambda: False

words = "அம்மா அப்பா வீடு பள்ளி மழை நிலா கடல் மரம் பூ நதி".split()
rng = random.Random(0)
docs = [" ".join(rng.choices(words, k=60)) for _ in range(300)]
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    tb.BPB, tb.RESULTS = tmp / "bpb", tmp / "results"
    tb.BPB.mkdir()
    with gzip.open(tb.BPB / "train.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps({"text": d}, ensure_ascii=False) + "\n" for d in docs)
    json.dump({"held": docs[:5]}, open(tb.BPB / "eval.json", "w"), ensure_ascii=False)
    json.dump({"pick": [{"context": "அம்மா அப்பா", "choices": [" வீடு", " பள்ளி மழை"], "label": 1}] * 3},
              open(tb.BPB / "tasks.json", "w"), ensure_ascii=False)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(docs, trainers.BpeTrainer(vocab_size=100))
    tok.save(str(tmp / "tok.json"))
    run = lambda name, stop_at=None, steps=None: tb.train(argparse.Namespace(
        tokenizer=str(tmp / "tok.json"), name=name, seed=0, size="tiny", batch=4, micro=2, lr=1e-3, log_every=1,
        no_compile=True, max_steps=8, steps=steps, save_minutes=30, stop_at=stop_at, base=None))
    run("whole")
    run("broken", stop_at=3)
    assert (tb.RESULTS / "broken.ckpt").exists() and not (tb.RESULTS / "broken.json").exists()
    run("broken")
    a, b = (json.load(open(tb.RESULTS / f"{n}.json")) for n in ("whole", "broken"))
    assert [r["loss"] for r in a["log"]] == [r["loss"] for r in b["log"]], (a["log"], b["log"])
    assert a["bits_per_byte"] == b["bits_per_byte"], (a["bits_per_byte"], b["bits_per_byte"])
    assert not (tb.RESULTS / "broken.ckpt").exists()
    assert a["tasks"] == b["tasks"] and a["tasks"]["pick"]["accuracy"] in (0, 1), a["tasks"]  # 3 copies of one item
    assert set(torch.load(tb.BPB / "models/broken.pt")) == set(tb.model_for(a["vocab_size"] + 1, "tiny")[0].state_dict())
    # --steps past one pass repeats text: 2.5 passes of the stream, every step a full batch
    seqs = (a["training"]["stream_tokens"] - 1) // a["training"]["context"]
    run("repeat", steps=seqs * 5 // 8)
    r = json.load(open(tb.RESULTS / "repeat.json"))
    assert r["training"]["steps"] == seqs * 5 // 8 and r["training"]["tokens"] > 2 * r["training"]["stream_tokens"], r["training"]
print("resume check passed")
