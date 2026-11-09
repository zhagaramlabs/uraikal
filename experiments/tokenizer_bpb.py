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
    # equal compute: as many steps as another tokenizer's one pass, repeating some text
    uv run ... python experiments/tokenizer_bpb.py train data/tamil_tokenizer/r3-grapheme-super-64k.json --steps 2900 --name r3-gs-64k-steps2900
    uv run ... python experiments/tokenizer_bpb.py report   # mean and spread over seeds
    # phase C, 350M: about 4x the text in data/bpb-c, the same test sets, and Tamil multiple-choice tasks
    uv run ... python experiments/tokenizer_bpb.py tasks --data bpb-c   # first: prep drops training text near these
    uv run ... python experiments/tokenizer_bpb.py prep --data bpb-c --shards 20 --eval-from bpb
    uv run ... python experiments/tokenizer_bpb.py train data/tamil_tokenizer/r2-grapheme-64k.json --data bpb-c --size medium --lr 3e-4

Text: Sangraha verified Tamil data-0 to data-3 (D4.web is data-52), Tamil Wikipedia, Project Madurai without
the D4.literature lines and the Thirukkural books, 15% English (FineWeb, after the 2,000 test documents),
local Python, and 5% synthetic romanized Tamil, as in tamil_tokenizer_r2.py.
Held out: D4.web, D4.literature, Thirukkural verse, romanized Tamil (Dakshina dev), English and code. Three
tokenizer rounds were picked on D4's token counts, so D4 is development data here; fresh.web is the final Tamil
test: FineWeb-2's Tamil test split, sampled like D4, minus any passage sharing 8 words in a row with the training text
(`fresh` adds it to an existing eval.json; `prep` builds it too).
Writes results/experiments/<data>/<name>.json, and the trained weights (bfloat16) to data/<data>/models/<name>.pt.
With data/<data>/tasks.json, each result also scores Belebele, XCOPA and Global PIQA in Tamil by likelihood.
Spot GPUs: a run checkpoints every 30 minutes to results/experiments/bpb/<name>.ckpt, and after a preemption the
same command carries on from there. The result matches an unbroken run's: bit for bit on the CPU, while GPU kernels
add float noise of their own (experiments/test_bpb_resume.py).
"""
import argparse
import csv
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
    "medium": (24, 1024, 16, 2816, 1024),  # phase C: 308M non-embedding, 66M embedding at 64k
    "tiny": (2, 128, 4, 256, 256),  # smoke test only
}


def near(passages, gram=8):
    """A test of whether a text shares `gram` words in a row with any of the passages."""
    from uraikal.text import clean

    shingles = {tuple(w[i:i + gram]) for p in passages for w in [clean(p).split()] for i in range(len(w) - gram + 1)}
    starts = {g[0] for g in shingles}

    def test(text):
        words = clean(text).split()
        return any(w in starts and tuple(words[i:i + gram]) in shingles for i, w in enumerate(words))
    return test


def prep(args):
    import tamil_tokenizer as r1
    import tamil_tokenizer_r2 as r2
    from uraikal.classics import kural_text

    shards = [i for i in range(52) if i != 5][:args.shards]  # data-5 is the romanized text's source; data-52 is D4.web's
    tamil = [x for i in shards for x in r2.sangraha(i)] + r2.wikipedia() + r2.madurai()
    tb = sum(len(x.encode()) for x in tamil)
    need = {k: int(tb * s / (1 - sum(r2.SHARES.values()))) for k, s in r2.SHARES.items()}
    eng_slice, eng = r1.english(need["english"])
    code_, roman = r1.code(need["code"]), r2.romanized_text(need["romanized"])[0]
    size = lambda texts: sum(len(x.encode()) for x in texts)
    manifest = {"shards": shards, "bytes": {"tamil": tb, "english": size(eng), "code": size(code_), "romanized": size(roman)}}
    docs = tamil + eng + code_ + roman
    random.Random(0).shuffle(docs)
    if args.tiny:
        docs = docs[:20_000]
    if args.eval_from:  # the earlier test sets as they are: drop training text sharing 8 words in a row with fresh.web
        held = json.load(open(ROOT / "data" / args.eval_from / "eval.json", encoding="utf-8"))
        tasks_ = BPB / "tasks.json"  # or with a task's passage (Belebele's come from FLORES, which crawls may hold)
        tests = [it["context"] + it["choices"][it["label"]] for v in json.load(open(tasks_, encoding="utf-8")).values()
                 for it in v] if tasks_.exists() else []
        test, n = near(held["fresh.web"] + tests), len(docs)
        docs = [d for d in docs if not test(d)]
        manifest |= {"eval_from": args.eval_from, "dropped_near_tests": n - len(docs), "task_texts_checked": len(tests)}
    BPB.mkdir(parents=True, exist_ok=True)
    for old in (BPB / "tokens").glob("*.u32"):  # token caches belong to the previous text
        old.unlink()
    with gzip.open(BPB / "train.jsonl.gz", "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    manifest |= {"documents": len(docs), "train_bytes": size(docs)}
    json.dump(manifest, open(BPB / "manifest.json", "w"), indent=1)
    print(manifest)
    if args.eval_from:
        json.dump(held, open(BPB / "eval.json", "w"), ensure_ascii=False)
        return
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


def tasks(_args):
    """Tamil multiple-choice tasks a small base model can be scored on by likelihood, to data/<data>/tasks.json:
    Belebele (CC BY-SA 4.0), XCOPA validation and test (CC BY 4.0) with lm-evaluation-harness's Tamil connectives,
    and Global PIQA's native-written Tamil items, both sets (CC BY-SA 4.0)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    get = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")
    out = {"belebele": [], "xcopa": [], "global_piqa": []}
    for line in open(get("facebook/belebele", "data/tam_Taml.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        out["belebele"].append({"context": f"P: {r['flores_passage']}\nQ: {r['question']}\nA:",
                                "choices": [" " + r[f"mc_answer{k}"] for k in range(1, 5)], "label": int(r["correct_answer_num"]) - 1})
    for split in ("validation", "test"):
        for r in pq.read_table(get("cambridgeltl/xcopa", f"ta/{split}-00000-of-00001.parquet")).to_pylist():
            out["xcopa"].append({"context": r["premise"].rstrip(".") + (" காரணமாக" if r["question"] == "cause" else " எனவே"),
                                 "choices": [" " + r["choice1"], " " + r["choice2"]], "label": r["label"]})
    for repo, f in (("mrlbenchmarks/global-piqa-nonparallel", "data/nonparallel_tam_taml.tsv"),
                    ("mrlbenchmarks/global-piqa-parallel", "data/parallel_tam_taml.tsv")):
        for r in csv.DictReader(open(get(repo, f), encoding="utf-8"), delimiter="\t"):
            choices = [" " + r[k] for k in ("solution0", "solution1", "solution2", "solution3") if r.get(k)]
            out["global_piqa"].append({"context": r["prompt"], "choices": choices, "label": int(r["label"])})
    assert all(0 <= x["label"] < len(x["choices"]) for v in out.values() for x in v)
    BPB.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(BPB / "tasks.json", "w"), ensure_ascii=False)
    print({k: len(v) for k, v in out.items()})


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
        # a run killed mid-write must not leave a short cache that later runs trust (it did, 2026-11-06)
        with gzip.open(BPB / "train.jsonl.gz", "rt", encoding="utf-8") as f, open(path.with_suffix(".tmp"), "wb") as out:
            batch = []
            for line in f:
                batch.append(json.loads(line)["text"])
                if len(batch) == 10_000:
                    np.concatenate([np.asarray([bos, *ids], dtype=np.uint32) for ids in encode(backend, batch)]).tofile(out)
                    batch = []
            if batch:
                np.concatenate([np.asarray([bos, *ids], dtype=np.uint32) for ids in encode(backend, batch)]).tofile(out)
        path.with_suffix(".tmp").replace(path)
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


def accuracy(model, backend, bos, items, ctx, device):
    """Share of items whose right choice gets the highest log-probability per UTF-8 byte after the context: per
    byte, so a tokenizer that spends more tokens on a choice isn't marked down for it."""
    import torch

    right = 0
    model.eval()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for it in items:
            c = [bos, *encode(backend, [it["context"]])[0]]
            score = []
            for choice, a in zip(it["choices"], encode(backend, it["choices"])):
                w = torch.tensor((c + a)[-(ctx + 1):], device=device)[None]
                logp = torch.log_softmax(model(w[:, :-1]).logits[0, -len(a):].float(), -1)
                score.append(logp.gather(1, w[0, -len(a):, None]).sum().item() / len(choice.encode()))
            right += max(range(len(score)), key=score.__getitem__) == it["label"]
    model.train()
    return right / len(items)


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
    steps = args.steps or max(1, min(seqs // args.batch, args.max_steps or seqs))
    rng = np.random.default_rng(args.seed)  # --steps past one pass: each further pass is a fresh shuffle; one pass is as before
    order = np.concatenate([rng.permutation(seqs) for _ in range(-(-steps * args.batch // seqs))])[: steps * args.batch]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
                            fused=device.type == "cuda")
    warm = max(1, steps // 100)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, 0.1 + 0.45 * (1 + math.cos(math.pi * s / steps))))
    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt = RESULTS / f"{name}.ckpt"
    setup = {"vocab": n, "size": args.size, "steps": steps, "batch": args.batch, "lr": args.lr}
    first, log, spent = 0, [], 0.0
    if ckpt.exists():  # a preempted run carries on from its last checkpoint
        state = torch.load(ckpt, map_location=device)
        assert state["setup"] == setup, f"{ckpt} is from another setup, {state['setup']}: delete it to start over"
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        first, log, spent = state["step"], state["log"], state["seconds"]
    step_model = torch.compile(model) if device.type == "cuda" and not args.no_compile else model
    micro = args.micro or args.batch
    start = saved = time.time()
    print(f"{name}: vocab {n}, {len(stream) / 1e6:.0f}M tokens, {steps} steps of {args.batch}x{ctx}, "
          f"{(total - embed) / 1e6:.0f}M non-embedding + {embed / 1e6:.0f}M embedding parameters on {device}"
          + (f", resuming at step {first}" if first else ""), flush=True)
    for step in range(first, steps):
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
            log.append({"step": step, "loss": round(loss.item(), 4), "seconds": round(spent + time.time() - start)})
            print(log[-1], flush=True)
        done = step + 1
        if done < steps and (time.time() - saved > args.save_minutes * 60 or done == args.stop_at):
            # no dropout, so training draws no random numbers: this and the seeded batch order are the whole state
            torch.save({"setup": setup, "step": done, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "log": log, "seconds": spent + time.time() - start},
                       ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)  # a pod that dies mid-save still leaves the last good checkpoint
            saved = time.time()
            if done == args.stop_at:
                return
    held = json.load(open(BPB / "eval.json", encoding="utf-8"))
    result = {"tokenizer": args.tokenizer, "name": name, "base": base, "seed": args.seed, "vocab_size": n, "size": args.size,
              "parameters": {"non_embedding": total - embed, "embedding": embed},
              "training": {"tokens": steps * args.batch * ctx, "stream_tokens": len(stream), "steps": steps,
                           "batch": args.batch, "context": ctx, "lr": args.lr,
                           "flops": 6 * total * steps * args.batch * ctx, "seconds": round(spent + time.time() - start),
                           "device": str(device)},
              "bits_per_byte": {k: bits_per_byte(model, backend, bos, v, ctx, device) for k, v in held.items()},
              "tokens_per_byte": {k: sum(map(len, encode(backend, v))) / sum(len(t.encode()) for t in v) for k, v in held.items()},
              "log": log}
    if (BPB / "tasks.json").exists():
        result["tasks"] = {k: {"accuracy": accuracy(model, backend, bos, v, ctx, device), "items": len(v)}
                           for k, v in json.load(open(BPB / "tasks.json", encoding="utf-8")).items()}
    (BPB / "models").mkdir(exist_ok=True)  # before the result: a result on disk means its weights are too
    torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, BPB / "models" / f"{name}.pt")
    with open(RESULTS / f"{name}.json.tmp", "w") as f:
        json.dump(result, f, indent=1)
    # the result file is the run's "done" mark (worker.sh skips the run, then reports done): never leave half of one
    (RESULTS / f"{name}.json.tmp").replace(RESULTS / f"{name}.json")
    ckpt.unlink(missing_ok=True)
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
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--data", default="bpb", help="data/<this> holds the text, test sets and token caches; "
                                                     "results go to results/experiments/<this>")
    a = sub.add_parser("prep", parents=[data])
    a.add_argument("--shards", type=int, default=4, help="Sangraha verified Tamil shards: data-0 up, skipping data-5")
    a.add_argument("--eval-from", help="copy eval.json from data/<this> instead of building it")
    a.add_argument("--tiny", action="store_true")
    sub.add_parser("tasks", parents=[data])
    a = sub.add_parser("train", parents=[data])
    a.add_argument("tokenizer")
    a.add_argument("--name")
    a.add_argument("--size", choices=SIZES, default="small")
    a.add_argument("--batch", type=int, default=256, help="sequences per optimizer step")
    a.add_argument("--micro", type=int, default=16, help="sequences per forward pass")
    a.add_argument("--lr", type=float, default=6e-4)
    a.add_argument("--log-every", type=int, default=50)
    a.add_argument("--no-compile", action="store_true")
    a.add_argument("--max-steps", type=int, help="stop early: for smoke tests, not results")
    a.add_argument("--steps", type=int, help="train exactly this many steps, repeating text if one pass is shorter: "
                                              "compares tokenizers at equal compute (use with --name)")
    a.add_argument("--seed", type=int, default=0, help="model initialization and batch order")
    a.add_argument("--save-minutes", type=float, default=30, help="checkpoint this often, for spot GPUs")
    a.add_argument("--stop-at", type=int, help="quit after the checkpoint at this step, as a preempted pod would: tests resume")
    sub.add_parser("fresh", parents=[data])
    sub.add_parser("report", parents=[data])
    args = p.parse_args()
    global BPB, RESULTS
    BPB, RESULTS = ROOT / "data" / args.data, ROOT / "results/experiments" / args.data
    {"prep": prep, "fresh": fresh, "tasks": tasks, "train": train, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
