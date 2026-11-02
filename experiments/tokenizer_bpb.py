"""Step 2 of the Tamil tokenizer plan: does a tokenizer that spends fewer tokens give a better model?

For each tokenizer, train the same small Llama-style model on the same text for one pass, then score
bits per byte on held-out text. Bits per byte divides the model's total loss by the text's UTF-8 bytes,
so it doesn't depend on how a tokenizer splits the text, and tokenizers can be compared fairly.
A tokenizer that spends fewer tokens on the same text also trains on fewer tokens, with less compute:
that saving is part of what it offers, and each result records its token count and FLOPs.

    # once, where the data is (writes data/bpb/: train.jsonl.gz and eval.json; both stay private)
    uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/tokenizer_bpb.py prep
    # on the GPU, once per tokenizer: a Hugging Face name or a tokenizer.json path
    uv run --extra tokenizers --extra prep --with torch python experiments/tokenizer_bpb.py train murai-labs/agaram-2.5b
    uv run ... python experiments/tokenizer_bpb.py train data/tamil_tokenizer/r2-super-64k.json
    uv run ... python experiments/tokenizer_bpb.py train data/tamil_tokenizer/r3-grapheme-super-64k.json --seed 1
    uv run ... python experiments/tokenizer_bpb.py report   # mean and spread over seeds

Text: Sangraha verified Tamil data-0 to data-3 (D4.web is data-52), Tamil Wikipedia, Project Madurai without
the D4.literature lines and the Thirukkural books, 15% English (FineWeb, after the 2,000 test documents),
local Python, and 5% synthetic romanized Tamil, as in tamil_tokenizer_r2.py.
Held out: D4.web, D4.literature, Thirukkural verse, romanized Tamil (Dakshina dev), English and code. Three
tokenizer rounds were picked on D4's token counts, so D4 is development data here; fresh.web is the final Tamil
test: FineWeb-2's Tamil test split, sampled like D4, minus any passage sharing 8 words in a row with the training text
(`fresh` adds it to an existing eval.json; `prep` builds it too).
Writes results/experiments/bpb/<name>.json.
"""
import argparse
import gzip
import json
import math
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BPB = ROOT / "data/bpb"
RESULTS = ROOT / "results/experiments/bpb"
SIZES = {  # layers, width, heads, MLP width, context
    "small": (12, 768, 12, 2048, 1024),
    "tiny": (2, 128, 4, 256, 256),  # smoke test only
}


def prep(args):
    import tamil_tokenizer as r1
    import tamil_tokenizer_r2 as r2
    from uraikal.classics import kural_text

    tamil = [x for i in range(args.shards) for x in r2.sangraha(i)] + r2.wikipedia() + r2.madurai()
    tb = sum(len(x.encode()) for x in tamil)
    need = {k: int(tb * s / (1 - sum(r2.SHARES.values()))) for k, s in r2.SHARES.items()}
    eng_slice, eng = r1.english(need["english"])
    docs = tamil + eng + r1.code(need["code"]) + r2.romanized_text(need["romanized"])[0]
    random.Random(0).shuffle(docs)
    if args.tiny:
        docs = docs[:20_000]
    BPB.mkdir(parents=True, exist_ok=True)
    for old in (BPB / "tokens").glob("*.u32"):  # token caches belong to the previous text
        old.unlink()
    with gzip.open(BPB / "train.jsonl.gz", "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    lines = lambda p: [json.loads(x) for x in open(p, encoding="utf-8")]
    code = [p.read_text(encoding="utf-8") for d in (ROOT / "src", ROOT / "tests", ROOT.parent / "oosai/agent",
                                                    ROOT.parent / "oosai/eval") for p in sorted(d.rglob("*.py"))]
    held = {"D4.web": [r["text"] for r in lines(ROOT / "data/d4_web.jsonl")],
            "D4.literature": [r["text"] for r in lines(ROOT / "data/d4_literature.jsonl")],
            "verse": [kural_text(r) for r in lines(ROOT / "data/kural.jsonl")],
            "romanized": r1.romanized(), "english": eng_slice, "code": code}
    json.dump(held, open(BPB / "eval.json", "w"), ensure_ascii=False)
    fresh(args)
    print(f"{len(docs)} training documents, {sum(len(d.encode()) for d in docs) / 1e9:.2f} GB; eval slices",
          {k: len(v) for k, v in held.items()})


def fresh(args, n=2000, gram=8):
    """Add fresh.web to eval.json: FineWeb-2 Tamil test passages (lines of 20+ words, 90%+ Tamil, up to 3 per
    document, as prep.py d4 samples D4.web), dropping any that share `gram` words in a row with the training text."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from uraikal.text import clean, tamil_share

    path = hf_hub_download("HuggingFaceFW/fineweb-2", "data/tam_Taml/test/000_00000.parquet", repo_type="dataset")
    docs = [x for x in pq.read_table(path, columns=["text"]).column(0).to_pylist() if x]
    rng = random.Random(0)
    rng.shuffle(docs)
    candidates = []
    for doc in docs:
        lines = [line for line in doc.split("\n") if len(line.split()) >= 20]
        rng.shuffle(lines)
        candidates += [p for p in (clean(line).strip() for line in lines) if tamil_share(p) >= 0.9][:3]
        if len(candidates) >= 2 * n:
            break
    grams = lambda words: {tuple(words[i:i + gram]) for i in range(len(words) - gram + 1)}
    shingles = {g: c for c, p in enumerate(candidates) for g in grams(p.split())}
    starts = {g[0] for g in shingles}
    seen = set()
    with gzip.open(BPB / "train.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            words = clean(json.loads(line)["text"]).split()
            for i, w in enumerate(words):
                if w in starts:
                    c = shingles.get(tuple(words[i:i + gram]))
                    if c is not None:
                        seen.add(c)
    kept = [p for c, p in enumerate(candidates) if c not in seen][:n]
    held = json.load(open(BPB / "eval.json", encoding="utf-8"))
    held["fresh.web"] = kept
    json.dump(held, open(BPB / "eval.json", "w"), ensure_ascii=False)
    print(f"fresh.web: {len(kept)} passages kept, {len(seen)} of {len(candidates)} candidates shared {gram} words "
          "with the training text")


def tokenizer(spec):
    from transformers import PreTrainedTokenizerFast

    from uraikal.tokenizer_eval import load

    tok = PreTrainedTokenizerFast(tokenizer_file=spec) if spec.endswith(".json") else load(spec)
    return tok.backend_tokenizer, len(tok)


def encode(backend, texts):
    return [e.ids for e in backend.encode_batch(texts, add_special_tokens=False)]


def tokens(backend, bos, name):
    """The training text as one uint32 stream, documents separated by our own BOS id; cached per tokenizer."""
    path = BPB / f"tokens/{name}.u32"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        with gzip.open(BPB / "train.jsonl.gz", "rt", encoding="utf-8") as f, open(path, "wb") as out:
            batch = []
            for line in f:
                batch.append(json.loads(line)["text"])
                if len(batch) == 10_000:
                    np.concatenate([np.asarray([bos, *ids], dtype=np.uint32) for ids in encode(backend, batch)]).tofile(out)
                    batch = []
            if batch:
                np.concatenate([np.asarray([bos, *ids], dtype=np.uint32) for ids in encode(backend, batch)]).tofile(out)
    return np.memmap(path, dtype=np.uint32, mode="r")


def model_for(vocab, size):
    from transformers import LlamaConfig, LlamaForCausalLM

    layers, width, heads, mlp, ctx = SIZES[size]
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=width, intermediate_size=mlp, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=heads, max_position_embeddings=ctx,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    return LlamaForCausalLM(cfg), ctx


def bits_per_byte(model, backend, bos, texts, ctx, device):
    """Total loss in bits over the UTF-8 bytes of the texts. Each text starts from our BOS; long ones are
    cut into windows of ctx tokens, and every token after the BOS is predicted exactly once."""
    import torch

    nats = 0.0
    model.eval()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for ids in encode(backend, texts):
            ids = [bos, *ids]
            for i in range(0, len(ids) - 1, ctx):
                w = torch.tensor(ids[i:i + ctx + 1], device=device)[None]
                logits = model(w[:, :-1]).logits.float()
                nats += torch.nn.functional.cross_entropy(logits[0], w[0, 1:], reduction="sum").item()
    model.train()
    return nats / math.log(2) / sum(len(t.encode()) for t in texts)


def train(args):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    base = args.name or (Path(args.tokenizer).stem if args.tokenizer.endswith(".json") else args.tokenizer.split("/")[-1])
    name = f"{base}-s{args.seed}" if args.seed else base
    backend, n = tokenizer(args.tokenizer)
    bos = n  # one id past the tokenizer's own, so no tokenizer's special tokens matter
    stream = tokens(backend, bos, base)
    torch.manual_seed(args.seed)
    model, ctx = model_for(-(-(n + 1) // 64) * 64, args.size)
    model.to(device)
    embed = model.get_input_embeddings().weight.numel()
    total = sum(p.numel() for p in model.parameters())
    seqs = (len(stream) - 1) // ctx
    steps = max(1, min(seqs // args.batch, args.max_steps or seqs))
    order = np.random.default_rng(args.seed).permutation(seqs)[: steps * args.batch]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
                            fused=device.type == "cuda")
    warm = max(1, steps // 100)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.1 + 0.45 * (1 + math.cos(math.pi * s / steps))))
    step_model = torch.compile(model) if device.type == "cuda" and not args.no_compile else model
    micro = args.micro or args.batch
    log, start = [], time.time()
    print(f"{name}: vocab {n}, {len(stream) / 1e6:.0f}M tokens, {steps} steps of {args.batch}x{ctx}, "
          f"{(total - embed) / 1e6:.0f}M non-embedding + {embed / 1e6:.0f}M embedding parameters on {device}", flush=True)
    for step in range(steps):
        for j in range(0, args.batch, micro):
            idx = order[step * args.batch + j: step * args.batch + j + micro]
            x = torch.from_numpy(np.stack([stream[k * ctx: k * ctx + ctx + 1].astype(np.int64) for k in idx])).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = step_model(x[:, :-1]).logits  # not labels=: the model would shift them a second time
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), x[:, 1:].flatten())
            (loss * len(idx) / args.batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0 or step == steps - 1:
            log.append({"step": step, "loss": round(loss.item(), 4), "seconds": round(time.time() - start)})
            print(log[-1], flush=True)
    held = json.load(open(BPB / "eval.json", encoding="utf-8"))
    result = {"tokenizer": args.tokenizer, "name": name, "base": base, "seed": args.seed, "vocab_size": n, "size": args.size,
              "parameters": {"non_embedding": total - embed, "embedding": embed},
              "training": {"tokens": steps * args.batch * ctx, "stream_tokens": len(stream), "steps": steps,
                           "batch": args.batch, "context": ctx, "lr": args.lr,
                           "flops": 6 * total * steps * args.batch * ctx, "seconds": round(time.time() - start),
                           "device": str(device)},
              "bits_per_byte": {k: bits_per_byte(model, backend, bos, v, ctx, device) for k, v in held.items()},
              "tokens_per_byte": {k: sum(map(len, encode(backend, v))) / sum(len(t.encode()) for t in v) for k, v in held.items()},
              "log": log}
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(RESULTS / f"{name}.json", "w"), indent=1)
    print(name, {k: round(v, 4) for k, v in result["bits_per_byte"].items()}, flush=True)


def report(_args):
    """One row per tokenizer: bits per byte by slice, as the mean over seeds, with the spread (max - min) if more
    than one seed ran."""
    runs = {}
    for p in sorted(RESULTS.glob("*.json")):
        r = json.load(open(p))
        runs.setdefault(r.get("base", r["name"]), []).append(r)
    slices = list(next(iter(runs.values()))[0]["bits_per_byte"]) if runs else []
    mean = lambda rs, s: sum(r["bits_per_byte"][s] for r in rs) / len(rs)
    print(f"{'tokenizer':<26}{'vocab':>8}{'seeds':>6}{'tokens':>8}" + "".join(f"{s:>18}" for s in slices))
    key = "fresh.web" if "fresh.web" in slices else "D4.web"
    for base, rs in sorted(runs.items(), key=lambda kv: mean(kv[1], key)):
        cells = []
        for s in slices:
            v = [r["bits_per_byte"][s] for r in rs]
            cells.append(f"{mean(rs, s):.4f}" + (f" ±{(max(v) - min(v)) / 2:.4f}" if len(v) > 1 else ""))
        print(f"{base:<26}{rs[0]['vocab_size']:>8}{len(rs):>6}{rs[0]['training']['tokens'] / 1e6:>7.0f}M"
              + "".join(f"{c:>18}" for c in cells))

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--shards", type=int, default=4)
    a.add_argument("--tiny", action="store_true")
    a = sub.add_parser("train")
    a.add_argument("tokenizer")
    a.add_argument("--name")
    a.add_argument("--size", choices=SIZES, default="small")
    a.add_argument("--batch", type=int, default=256, help="sequences per optimizer step")
    a.add_argument("--micro", type=int, default=16, help="sequences per forward pass")
    a.add_argument("--lr", type=float, default=6e-4)
    a.add_argument("--log-every", type=int, default=50)
    a.add_argument("--no-compile", action="store_true")
    a.add_argument("--max-steps", type=int, help="stop early: for smoke tests, not results")
    a.add_argument("--seed", type=int, default=0, help="model initialization and batch order")
    sub.add_parser("fresh")
    sub.add_parser("report")
    args = p.parse_args()
    {"prep": prep, "fresh": fresh, "train": train, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
