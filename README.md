# Uraikal

Uraikal (உரைகல், "touchstone") is Zhagaram Labs' benchmark for Tamil models, plus the Tamil text tools it scores with. This is v0.1, a **draft**: tracks and scoring may change before v1.0, and v0.x scores are preliminary. It covers two tracks:
- **A1:** speech recognition on public read-speech test sets.
- **D4:** how well tokenizers handle Tamil.

Full spec: [SPEC.md](SPEC.md).

```bash
pip install uraikal                # the library; no dependencies
pip install "uraikal[tokenizers]"  # plus track D4
```

## The Tamil text library

The library has no dependencies; it runs on Python 3.10+. Scoring and the library share this code, so the benchmark's own tests keep the library correct.

```python
from uraikal.text import clean, letters, length, tamil_share
from uraikal.numbers import to_words, to_digits, from_numerals

clean("ெகாடு")                    # "கொடு": vowel sign typed before its consonant (legacy fonts), fixed
letters("தமிழ்")                  # ["த", "மி", "ழ்"]: Tamil letters, not code points
[length(x) for x in letters("கடல்")]  # ["kuril", "kuril", "otru"], for prosody
to_words(2300)                     # "ரெண்டாயிரத்தி முன்னூறு" (spoken Tamil)
to_digits("முன்னூறுல பதினஞ்சு")       # "300ல 15"
from_numerals("௨௱௫")               # 205: the old system, where ௱ multiplies
```

- `uraikal.text.normalize` is the frozen scoring normaliser.
- `clean` is the general cleaner for pipelines.

## Run it

Install with [uv](https://docs.astral.sh/uv/). Each extra pulls in only what its task needs:
- `asr`: transcribing;
- `tokenizers`: track D4;
- `prep`: building D4 test text from Parquet files.

Transcribing also needs `ffmpeg`.

```bash
for f in tests/test_*.py; do uv run python $f; done     # self-checks

# A1: speech recognition
uv run python prep.py fleurs                      # FLEURS ta_in test, 591 utterances
uv run python prep.py kathbath                    # Kathbath known + unknown speakers
uv run python prep.py common_voice --dir PATH     # after downloading Common Voice Tamil v27.0 yourself
uv run --extra asr python transcribe.py --manifest data/a1_*.jsonl --model openai/whisper-large-v3-turbo --out hyps/whisper-large-v3-turbo.jsonl
uv run uraikal score --manifest data/a1_*.jsonl --hyp hyps/whisper-large-v3-turbo.jsonl --system whisper-large-v3-turbo

# D4: tokenizers (set HF_TOKEN for gated repos)
uv run --extra prep python prep.py d4 --parquet data/sangraha/data-52.parquet --slice D4.web --n 2000 --per-doc 1
uv run --extra prep python prep.py d4 --parquet data/madurai/project-madurai.parquet --slice D4.literature --n 1000
uv run --extra tokenizers uraikal tokenizers --data data/d4_*.jsonl
```

Common Voice is only distributed through [Mozilla Data Collective](https://mozilladatacollective.com/datasets?q=Tamil), which needs a login. Download the Tamil release there, then point `--dir` at the folder holding `test.tsv` and `clips/`.

**D4 test text.**
- **D4.web:** taken from `verified/tam/data-52.parquet` of [`ai4bharat/sangraha`](https://huggingface.co/datasets/ai4bharat/sangraha) at revision `8b813c3f`, licensed CC BY 4.0. That shard is held out of every Zhagaram tokenizer's training data.
- **D4.literature:** taken from our Project Madurai crawl, which mixes classical and modern works. Its text stays private until the rights are secured; only the scores are published.

## Formats

- **A1 manifest** (`data/a1_<source>.jsonl`): `{"id", "audio", "text", "slice"}`, one utterance per line.
- **D4 passages** (`data/d4_<slice>.jsonl`): `{"id", "text", "slice", "source"}`.
- **Hypotheses**: `{"id", "hyp"}`, one per line. Any system can be scored by writing this file: an API, or IndicConformer through AI4Bharat's NeMo fork.
- **Results** (`results/<system>.json`, `results/d4.json`): one record per slice, in the format from the spec's "Leaderboard" section.

## Scoring rules

**A1:**
- Text is normalised before scoring:
  - Unicode NFC.
  - Zero-width characters dropped.
  - Tamil numerals mapped to 0–9.
  - Latin lower-cased.
  - Punctuation and symbols turned into spaces.
- WER and CER are corpus-level (total errors ÷ total reference length) for each slice. When there's more than one slice, a macro average is printed too.
- A missing hypothesis scores as empty output and is counted in `missing`. It is never skipped.
- CER counts Unicode code points, not aksharas.

**D4**, per tokenizer and slice:
- **Tokens per word** (whitespace words) and **bytes per token**, shown next to the vocabulary size.
- **Letter cut:** share of Tamil words with a token boundary between a consonant and its vowel sign or pulli. Not always a defect, since suffixes often start inside a letter (மரத்தை = மரத்த் + ஐ).
- **By pre-tok:** the share already cut by the pre-tokenizer. This is always a defect: the `\p{L}+` regex bug.
- **Byte cut:** share of Tamil words containing a token that ends inside a character. Always a defect.
- **Unknown tokens per 1,000**.
- **Round trip:** share of passages that decode back to the same characters, ignoring whitespace.
- Tokenizers are loaded with `trust_remote_code=False`. Repos we can't load, gated or code-only, are reported and skipped.

## Roadmap

Each piece ships in this package and is checked by a Uraikal track.

- **Next:**
  - `uraikal.yaappu`: Tamil prosody, from letters to அசை, சீர் and தளை, tested on all 1,330 Kurals.
  - `uraikal.tn`: text normalisation for speech, covering dates, ₹ amounts, units and phone numbers.
  - A grapheme-safe pre-tokenizer pattern for Hugging Face tokenizers.
- **Later:**
  - Our own Tamil tokenizer, plus Tamil extensions for open tokenizers.
  - A text quality classifier and a register classifier (literary, spoken or Tanglish).
  - Legacy-font conversion.
  - Practice-task generators for reinforcement learning.

## Releasing

1. Bump `__version__` in `src/uraikal/__init__.py`.
2. Push a tag such as `v0.2.0`. The `release` workflow then:
   - runs the tests;
   - checks that the tag matches the version;
   - builds the package;
   - publishes it to PyPI through Trusted Publishing, so no token is stored anywhere;
   - creates the GitHub Release.

`URAIKAL_VERSION` in `src/uraikal/score.py` is the benchmark version. It changes only when scoring changes, for example the normaliser, a metric or a test set. Every result records it.

## Licence

Code: Apache 2.0 ([LICENSE](LICENSE)). Test data keeps its own licence; see the spec. To cite Uraikal, use [CITATION.cff](CITATION.cff).
