"""Transcribe a Uraikal manifest with a Hugging Face ASR model and write hypotheses as JSONL (id, hyp).

    python transcribe.py --manifest data/a1_fleurs.jsonl --model openai/whisper-large-v3-turbo \
        --out hyps/whisper-large-v3-turbo.jsonl [--limit 20]

Re-running with the same --out resumes where it stopped. Any other system (an API, the NeMo-based
IndicConformer) just needs to write the same {"id", "hyp"} JSONL for uraikal.py to score.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import pipeline

from uraikal.score import read_jsonl


def load(path, sr=16000):
    """Decode with ffmpeg from the path. The pipeline pipes file bytes instead, which fails on m4a files
    whose index sits at the end, as Kathbath's do."""
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1",
                          "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    return {"raw": np.frombuffer(pcm, np.float32), "sampling_rate": sr}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, help="only the first N utterances, for smoke tests")
    p.add_argument("--batch-size", type=int, default=8, help="clips decoded together; 1 is several times slower")
    args = p.parse_args()

    rows = [r for m in args.manifest for r in read_jsonl(m)][: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} to transcribe, {len(done)} already done")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model=args.model, device=device,
                   torch_dtype=torch.float32 if device == "cpu" else torch.float16)
    # Whisper otherwise guesses the language per clip, and may translate. Clips over 30 s are cut into
    # overlapping 30 s windows, which lets them share a batch with short clips.
    kwargs = ({"chunk_length_s": 30, "generate_kwargs": {"language": "tamil", "task": "transcribe"}}
              if "whisper" in args.model.lower() else {})

    # A generator (not a list) makes the pipeline yield results as they finish, so each one is saved for resuming.
    results = asr((load(r["audio"]) for r in todo), batch_size=args.batch_size, **kwargs)
    with open(out, "a", encoding="utf-8") as f:
        for i, (row, res) in enumerate(zip(todo, results), 1):
            f.write(json.dumps({"id": row["id"], "hyp": res["text"].strip()}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(todo)}")


if __name__ == "__main__":
    main()
