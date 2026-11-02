"""The `uraikal` command.

    uraikal score --manifest data/a1_fleurs.jsonl --hyp hyps/whisper.jsonl --system whisper-large-v3-turbo
    uraikal tokenizers --data data/d4_*.jsonl            # D4, default roster; or --tokenizer REPO ...
    uraikal classics --model qwen3:8b --system qwen3-8b --no-think   # track I, a model served by Ollama
"""
import argparse
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

from uraikal.score import URAIKAL_VERSION, read_jsonl, score


def write_json(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def score_cmd(args):
    rows = [r for m in args.manifest for r in read_jsonl(m)]
    hyps = {r["id"]: r["hyp"] for r in read_jsonl(args.hyp)}
    results = score(rows, hyps)

    print(f"{'slice':<28}{'n':>7}{'WER':>9}{'CER':>9}{'missing':>9}")
    for name, r in results.items():
        print(f"{name:<28}{r['n']:>7}{r['wer']:>9.2%}{r['cer']:>9.2%}{r['missing']:>9}")
    if len(results) > 1:
        macro = {k: sum(r[k] for r in results.values()) / len(results) for k in ("wer", "cer")}
        print(f"{'macro average':<28}{'':>7}{macro['wer']:>9.2%}{macro['cer']:>9.2%}")
    if any(r["missing"] for r in results.values()):
        print("warning: missing hypotheses were scored as empty output", file=sys.stderr)

    today = datetime.date.today().isoformat()
    write_json(Path(args.out) / f"{args.system}.json", [
        {
            "system": args.system,
            "type": args.type,
            "uraikal_version": URAIKAL_VERSION,
            "track": name.split(".")[0][0],
            "slice": name,
            "split": args.split,
            "metrics": {"wer": round(r["wer"], 4), "cer": round(r["cer"], 4)},
            "n_utterances": r["n"],
            "missing": r["missing"],
            "date": today,
        }
        for name, r in results.items()
    ])


def tokenizers_cmd(args):
    from uraikal.tokenizer_eval import DEFAULT_TOKENIZERS, run

    slices = defaultdict(list)
    for path in args.data:
        for row in read_jsonl(path):
            slices[row["slice"]].append(row["text"])
    records = run(args.tokenizer or DEFAULT_TOKENIZERS, dict(slices))

    print(f"{'tokenizer':<36}{'slice':<14}{'vocab':>8}{'tok/word':>9}{'B/tok':>7}{'letter cut':>11}{'by pre-tok':>11}"
          f"{'byte cut':>9}{'unk/1k':>8}{'round trip':>11}")
    for r in sorted(records, key=lambda r: (r["slice"], r["tokens_per_word"])):
        pre = r["pretokenizer_letter_cut_words"]
        print(f"{r['tokenizer']:<36}{r['slice']:<14}{r['vocab_size']:>8}{r['tokens_per_word']:>9.2f}{r['bytes_per_token']:>7.2f}"
              f"{r['letter_cut_words']:>11.1%}{'-' if pre is None else f'{pre:.1%}':>11}"
              f"{r['byte_cut_words']:>9.1%}{r['unk_per_1k']:>8.1f}{r['round_trip']:>11.1%}")
    today = datetime.date.today().isoformat()
    write_json(args.out, [{"uraikal_version": URAIKAL_VERSION, "track": "D", "date": today, **r} for r in records])


def classics_cmd(args):
    from uraikal import classics

    items = classics.items(*classics.load(args.kural, args.topics), n=args.n)
    replies = classics.run(items, classics.ollama(args.endpoint, args.model, args.max_tokens, False if args.no_think else None),
                           Path("hyps") / f"{args.system}.classics.jsonl")
    write_classics(args, classics.summarise(items, replies))


def write_classics(args, summary):
    """Print track I scores and save them as results/<system>-classics.json."""
    from uraikal.classics import GRADER

    for name, m in sorted(summary.items()):
        print(f"{name:<16}" + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()))
    today = datetime.date.today().isoformat()
    write_json(Path(args.out) / f"{args.system}-classics.json", [
        {"system": args.system, "type": args.type, "uraikal_version": URAIKAL_VERSION, "track": "I", "grader": GRADER, "slice": name,
         "split": "test", "metrics": {k: v for k, v in m.items() if k != "n"}, "n": m["n"],
         "max_tokens": args.max_tokens, "date": today} for name, m in sorted(summary.items())])


def classics_args(p):
    """Options shared with adapters for models served some other way (see oosai/eval/uraikal_classics.py)."""
    p.add_argument("--system", required=True)
    p.add_argument("--type", default="open-weights", choices=["open-weights", "api", "research-only", "ours"])
    p.add_argument("--kural", default="data/kural.jsonl")
    p.add_argument("--topics", default="data/i5_topics.tsv")
    p.add_argument("--n", type=int, default=200, help="Kurals each for I1 and I2")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--out", default="results")


def main():
    p = argparse.ArgumentParser(prog="uraikal", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("score", help="speech recognition: WER and CER per slice")
    s.add_argument("--manifest", nargs="+", required=True, help="JSONL rows: id, audio, text, slice")
    s.add_argument("--hyp", required=True, help="JSONL rows: id, hyp")
    s.add_argument("--system", required=True)
    s.add_argument("--type", default="open-weights", choices=["open-weights", "api", "research-only", "ours"])
    s.add_argument("--split", default="test")
    s.add_argument("--out", default="results", help="directory for <system>.json")
    s.set_defaults(fn=score_cmd)

    t = sub.add_parser("tokenizers", help="D4: tokenizer efficiency and letter integrity on Tamil")
    t.add_argument("--data", nargs="+", required=True, help="JSONL rows: id, text, slice (from prep.py d4)")
    t.add_argument("--tokenizer", nargs="+", help="Hugging Face repos; default: the v0 roster")
    t.add_argument("--out", default="results/d4.json")
    t.set_defaults(fn=tokenizers_cmd)

    c = sub.add_parser("classics", help="track I: Thirukkural recall, attribution, metre and invented verses")
    classics_args(c)
    c.add_argument("--model", required=True, help="Ollama model name")
    c.add_argument("--endpoint", default="http://127.0.0.1:11434")
    c.add_argument("--no-think", action="store_true", help="turn off a reasoning model's thinking")
    c.set_defaults(fn=classics_cmd)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
