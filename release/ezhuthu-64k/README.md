---
license: apache-2.0
language:
- ta
- en
library_name: transformers
tags:
- tokenizer
- tamil
- bpe
---

# எழுத்து (Ezhuthu) 64k

A 64,000-token BPE tokenizer for Tamil, with English and code. **No token boundary falls inside a Tamil letter**,
except in 4 rare letters (ஙெ, ஙௌ, ஞௌ, ழௌ). A letter here is a Unicode grapheme cluster: a vowel, or a consonant
alone or with its vowel sign or pulli (க, கா, கி, க்).

Most tokenizers learn Tamil from code points. They often split a letter, leaving a vowel sign or a pulli as its
own token, so that one letter costs two tokens. Ezhuthu learns its merges over whole letters instead. In training
it also never joined a word's stem to its suffix, for the 366,584 word forms that
[TamilMorph](https://huggingface.co/datasets/Surendhar32/TamilMorph) splits. In use it's a plain BPE tokenizer.

At the same vocabulary size, text and compute, it gave lower bits per byte on held-out Tamil prose than plain BPE
and than itself without the stem cuts, at 85M and 350M parameters, and than Agaram's tokenizer at 85M. The stem
cuts cost tokens: 6 to 7% more on Tamil web text than the same tokenizer without them.

Built by [Zhagaram Labs](https://zhagaramlabs.com), a non-profit open AI lab for Tamil.

## Use

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("zhagaramlabs/ezhuthu-64k")
ids = tok("தமிழ் எழுத்து")["input_ids"]
tok.convert_ids_to_tokens(ids)  # ['தமிழ்', ' எழுத்து']
tok.decode(ids)                 # 'தமிழ் எழுத்து'
```

It needs no remote code. We checked that it loads and gives the same ids with transformers 4.46.3, 4.57.1 and 5.22.1.

## What's inside

- **64,000 tokens:**
  - 256 byte tokens;
  - the code points of the 4,000 most common letters in the training text, and fixed merges that assemble
    each letter from them;
  - learned merges, which only ever join whole letters. None joined a listed word's stem to its suffix in
    training.

  There are 61,876 merges in all.
- **`<|endoftext|>` at id 64,000**, set as both `bos_token` and `eos_token`. Our test models put it at the start
  of every document. The tokenizer doesn't add it by itself, so prepend it when you train.
- **Normalization:** NFC.
- **Pre-tokenizer:** Qwen3.5's pattern. It keeps vowel signs with their letter, and splits digits one at a time.
- **Byte fallback:** a character outside the vocabulary becomes UTF-8 byte tokens. So there's no unknown token,
  and any text decodes back to itself (its NFC form).

## Results

### Tokens per word

Fewer is better. Each test has 2,000 or 1,000 held-out Tamil passages:

- **D4.web:** Sangraha shard 52, never used for training.
- **D4.literature:** Project Madurai lines, held out of training.
- **fresh.web:** FineWeb-2's Tamil test split, minus any passage that shares 8 words in a row with our model
  training text.

We picked three rounds of tokenizers on D4's token counts, so D4 is development data. fresh.web was built after
the picking, and it's the test. One later choice did look at fresh.web: we released Ezhuthu over the same
tokenizer without stem cuts on bits per byte from both D4 and fresh.web.

| | Tokenizer | Vocab | fresh.web | vs Ezhuthu | D4.web | Literature | Letter cuts | Round trip |
|---|---|---|---|---|---|---|---|---|
| 1 | `murai-labs/agaram-2.5b` | 64,000 | 1.532 | -7.4% | 1.677 | 1.926 | 17.1% | 100.0% |
| 2 | Ezhuthu without stem cuts (`r2-grapheme-64k`, not released) | 64,000 | 1.544 | -6.8% | 1.677 | 1.914 | 0.003% | 100.0% |
| 3 | `ai4bharat/IndicBERTv2-MLM-only` | 250,000 | 1.601 | -3.3% | 1.763 | 2.083 | 10.0% | 0.1% |
| 4 | **Ezhuthu 64k** | 64,001 | 1.656 |  | 1.783 | 1.968 | 0.003% | 100.0% |
| 5 | `google/muril-base-cased` | 197,258 | 1.722 | +4.0% | 1.887 | 2.214 | 7.9% | 99.7% |
| 6 | `bigscience/bloom-560m` | 250,680 | 2.016 | +21.7% | 2.154 | 2.415 | 39.3% | 100.0% |
| 7 | `sarvamai/sarvam-30b` | 262,144 | 2.279 | +37.6% | 2.436 | 2.691 | 24.0% | 100.0% |
| 8 | `google/gemma-3-4b-it` | 262,145 | 2.279 | +37.6% | 2.436 | 2.691 | 24.0% | 100.0% |
| 9 | `sarvamai/sarvam-1` | 68,096 | 2.289 | +38.3% | 2.463 | 2.772 | 25.0% | 100.0% |
| 10 | `FacebookAI/xlm-roberta-base` | 250,002 | 2.306 | +39.3% | 2.485 | 2.785 | 15.5% | 98.4% |
| 11 | `google/mt5-base` | 250,100 | 2.448 | +47.9% | 2.584 | 2.725 | 27.4% | 98.4% |
| 12 | `theschoolofai/BrahmicTokenizer-131K` | 131,072 | 3.011 | +81.9% | 3.165 | 3.379 | 62.6% | 100.0% |
| 13 | `openai/gpt-oss-20b` | 200,019 | 3.032 | +83.2% | 3.189 | 3.410 | 63.1% | 100.0% |
| 14 | `mistralai/Mistral-Nemo-Base-2407` | 131,072 | 3.624 | +118.9% | 3.774 | 3.889 | 76.0% | 100.0% |
| 15 | `Qwen/Qwen3.5-0.8B` | 248,077 | 4.132 | +149.6% | 4.276 | 4.344 | 63.2% | 100.0% |
| 16 | `deepseek-ai/DeepSeek-V3` | 128,815 | 4.841 | +192.4% | 4.963 | 4.939 | 87.2% | 100.0% |
| 17 | `LiquidAI/LFM3-Omni-2B` | 125,033 | 5.086 | +207.2% | 5.199 | 5.241 | 99.0% | 100.0% |
| 18 | `krutrim-ai-labs/Krutrim-1-instruct`² | 70,212 | 6.587 | +297.9% | 6.680 | 6.706 | 99.0% | 100.0% |
| 19 | `Qwen/Qwen3-8B` | 151,669 | 9.868 | +496.0% | 9.990 | 10.015 | 99.0% | 100.0% |
| 20 | `CohereLabs/aya-expanse-8b`¹ | 255,029 | 10.014 | +504.9% | 10.137 | 10.136 | 99.0% | 100.0% |
| 21 | `meta-llama/Llama-3.1-8B` | 128,256 | 12.040 | +627.3% | 12.185 | 12.264 | 99.0% | 100.0% |

Letter cuts are the share of fresh.web's Tamil words with a token boundary between a consonant and its vowel sign
or pulli. Round trip is the share of fresh.web passages that decode back to the same text, ignoring whitespace.

¹ Scored from `adamo1139/aya-expanse-8b-ungated`, whose `tokenizer.json` has the same sha256 as the gated repo's.
² Krutrim's IndicSuperTokenizer (arXiv 2511.03237, now titled MUTANT) had no public download on 2026-11-08.
Krutrim-1's tokenizer stands in for it.

- **Ezhuthu is 4th of 21 on fresh.web tokens.** The stem cuts cost 7.2% on fresh.web, 6.3% on D4.web and 2.9% on
  literature, against the same tokenizer without them.
- **Three spend fewer tokens on fresh.web:**
  - Agaram, the other Tamil-first 64k tokenizer, 7.4% fewer. It cuts 17% of Tamil words inside a letter.
  - Ezhuthu without stem cuts, 6.8% fewer.
  - IndicBERTv2, 3.3% fewer. Its decode gives the text back for only 0.1% of passages, and it cuts 10% of Tamil
    words inside a letter.
- **Every other tokenizer spends more:** MuRIL 4% more, Gemma 3 and Sarvam 38% more, and Aya Expanse and Llama 3.1
  6.0 and 7.3 times as many.

### Does it make a better model?

Token counts alone don't say whether a model learns better. So we trained the same 85M-parameter Llama-style model
on the same text with each tokenizer: 12 layers, width 768, context 1,024, batch 256, learning rate 6e-4, one pass
of about 763M tokens. Then we measured **bits per byte** on held-out text. Bits per byte divides the model's total
loss by the text's UTF-8 bytes, so it doesn't depend on how the text was split. Lower is better.

Each number is the mean over 3 seeds, ± half the spread:

| Tokenizer | Vocab | Steps | fresh.web | D4.web | Literature | Verse | English | Code |
|---|---|---|---|---|---|---|---|---|
| **Ezhuthu 64k** | 64k | 2,909 | **0.4095** ±0.0000 | **0.4236** ±0.0002 | **0.5624** ±0.0003 | 0.8855 ±0.0027 | 1.3488 ±0.0002 | 1.4394 ±0.0029 |
| Ezhuthu without stem cuts (`r2-grapheme-64k`, not released) | 64k | 2,909 | 0.4102 ±0.0008 | 0.4244 ±0.0007 | 0.5632 ±0.0014 | **0.8834** ±0.0043 | **1.3470** ±0.0028 | 1.4379 ±0.0050 |
| Agaram (`murai-labs/agaram-2.5b`) | 64k | 3,077 | 0.4128 ±0.0010 | 0.4278 ±0.0007 | 0.5682 ±0.0008 | 0.8841 ±0.0023 | 1.3476 ±0.0015 | **1.4342** ±0.0019 |
| Plain BPE, same recipe (`r3-bpe-64k`) | 64k | 2,915 | 0.4134 ±0.0002 | 0.4286 ±0.0003 | 0.5691 ±0.0005 | 0.8923 ±0.0035 | 1.3502 ±0.0013 | 1.4410 ±0.0023 |

- **Against plain BPE at equal compute, 2,909 against 2,915 steps:** −0.9% on fresh.web, −1.2% on D4.web and
  −1.2% on literature. The seeds don't overlap.
- **Against Agaram:** −0.8% on fresh.web and −1.0% on D4.web. Agaram's model got 6% more steps, because its
  tokenizer makes more tokens of the same text.
- **Against the same tokenizer without stem cuts,** at the same steps: −0.17% on fresh.web and −0.19% on D4.web.
  On D4.web every seed with cuts beats every seed without.
- **Large-vocabulary tokenizers** were run too, 1 seed each, but not at equal compute. Each model read the same
  text once, so a tokenizer that makes more tokens trains for more steps, and a 248k–262k vocabulary needs a much
  bigger output layer. Qwen3.5's tokenizer reached 0.4002 on fresh.web with 3.6× Ezhuthu's compute, and Gemma 3's
  reached 0.4101 with 2.7×.
- **Verse, English and code** are where it doesn't win. Those are texts TamilMorph doesn't cover, and their merges
  got fewer slots.
- **On a different text:** a whole-letter tokenizer trained on deduplicated text (`r4-grapheme-64k`) also beat
  plain BPE trained on that same text, on every Tamil slice.
- **Superword tokens,** which span several words, added nothing at equal compute. That's why this tokenizer has
  none.

**At 350M:** the same test with a 374M-parameter model (24 layers, width 1,024, learning rate 3e-4), trained for
9,800 steps on about 2.57B tokens with each tokenizer. The control is plain BPE trained on Ezhuthu's own text. Each
number is the mean over 2 seeds, ± half the spread:

| Tokenizer | fresh.web | D4.web | Literature | Verse | English | Code |
|---|---|---|---|---|---|---|
| **Ezhuthu 64k** | **0.3593** ±0.0002 | **0.3672** ±0.0002 | **0.5030** ±0.0000 | 0.7646 ±0.0059 | 1.1767 ±0.0004 | 1.2536 ±0.0007 |
| Ezhuthu without stem cuts (`r2-grapheme-64k`, not released) | 0.3601 ±0.0000 | 0.3678 ±0.0001 | 0.5037 ±0.0003 | **0.7615** ±0.0026 | 1.1756 ±0.0010 | 1.2524 ±0.0037 |
| Plain BPE, same text (`r2-bpe-64k`) | 0.3620 ±0.0002 | 0.3704 ±0.0003 | 0.5079 ±0.0001 | 0.7766 ±0.0005 | **1.1751** ±0.0004 | **1.2499** ±0.0007 |

- **Against plain BPE:** −0.7% on fresh.web, −0.9% on D4.web, −1.0% on literature and −1.5% on verse. English and
  code are 0.1% and 0.3% worse.
- **Against the same tokenizer without stem cuts:** −0.22% on fresh.web, −0.17% on D4.web and −0.14% on
  literature, as at 85M. On fresh.web both seeds with cuts (0.3596, 0.3591) beat both without (0.3601, 0.3601).
  Verse, English and code are 0.1% to 0.4% worse.
- **The tasks don't separate the tokenizers at this size.** On Tamil Belebele (900 items), XCOPA (600) and Global
  PIQA (203), all six models score near chance: 0.26–0.29 on Belebele, where chance is 0.25. The two seeds of
  one tokenizer differ by more than the tokenizers do.

### Letters, stems and round trips

- **Letter cuts:** on fresh.web, 1 Tamil word in 40,000 has a token boundary between a consonant and its vowel
  sign or pulli. For Agaram it's 1 in 6, and for Gemma 3 it's 1 in 4. All 8 such words across the three test sets
  are broken text, which no letter can hold whole: two vowel signs in a row, such as சாதுாியமாக (an old
  font-conversion error for சாதுரியமாக), or a pulli after a dash.
- **Stems:** on the 46,431 TamilMorph words held out of training, a token boundary falls where the stem ends
  more often than without the cuts. Morph-F1 is 0.608 (precision 0.473, recall 0.849), against 0.549 without the
  cuts and 0.307 for plain BPE (`r3-bpe-64k`).
- **Round trips:** every one of the 10,357 held-out passages decodes back to its NFC form, character for
  character. That covers Tamil web, literature, Thirukkural verse, romanized Tamil, English and code. Text that
  isn't NFC comes back as NFC.

## Training text

7.8 GB, 2.57M documents:

| Source | Share | Licence |
|---|---|---|
| Sangraha verified Tamil, `data-0` to `data-4` (AI4Bharat) | 75%: 5.86 GB of Tamil, with the next two | CC BY 4.0 |
| Tamil Wikipedia, 2023-11-01 dump (`wikimedia/wikipedia`) | | CC BY-SA 3.0 and GFDL |
| Project Madurai e-texts, minus the D4.literature lines and 6 books that quote the Thirukkural | | Free to distribute with the header page kept intact; commercial use not stated |
| English: FineWeb `sample-10BT`, after its first 2,000 documents | 15% | ODC-By 1.0 |
| Python source from a local Python install (standard library and installed packages) | 5% | PSF and each package's own licence |
| Romanized Tamil: Sangraha `data-5`, romanized word by word with Aksharantar's 3.16M pairs (CC BY 4.0), with a rule-based fallback | 5% | CC BY 4.0 |

The stem cuts come from TamilMorph (Muthukumar, Herygers & Beinborn, DravidianLangTech 2026; MIT licence): 418k
word forms split into stem and suffix. A tenth of its words is held out to score Morph-F1. Blocking merges at
morpheme boundaries follows MorphBPE (Findings of ACL 2026).

Aksharantar's pairs that come from Dakshina were left out, because Dakshina is one of our test sets.

These files hold a vocabulary and a merge list, not passages of the training text. We credit Sangraha, Wikipedia,
Project Madurai, FineWeb, Aksharantar and TamilMorph here.

## Conflict of interest

Zhagaram Labs built this tokenizer and also built the tests it's scored on: D4, fresh.web and the bits-per-byte
setup. To keep the test sets out of future training data, we don't release them. The code that builds and scores
them is in the [Uraikal](https://github.com/zhagaramlabs/uraikal) repository, so anyone can rebuild them from
their public sources.

## Reproduce

Run these in the Uraikal repository:

```bash
# round 2's training text and the tokenizer without stem cuts: data/tamil_tokenizer/r2-grapheme-64k.json
uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/tamil_tokenizer_r2.py
# Ezhuthu: data/tamil_tokenizer/r4-morph-64k.json, with Morph-F1
uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/tamil_tokenizer_morph.py
# this folder, from that file, with its checks
uv run --extra tokenizers python release/make_ezhuthu.py
# token counts against other tokenizers
uv run --extra tokenizers python experiments/tokenizer_d4_fresh.py
# bits per byte, on a GPU: once per tokenizer and seed
uv run --extra tokenizers --extra prep --with torch python experiments/tokenizer_bpb.py train data/tamil_tokenizer/r4-morph-64k.json --seed 0
```

The released `tokenizer.json` has sha256 `ad7884f9ebd950cefd827cd3c2b0c29c719c134941a2fae79fe27bbc90f6bc7d`. A
rebuild elsewhere may differ slightly: the code slice is read from installed Python packages. Ours came to 13,109
bytes less than round 2's 7.8 GB, with the same Tamil text byte for byte.

## Limits

- **Tamil costs more tokens than it could:** the stem cuts add 6 to 7% on Tamil web text, against the same
  tokenizer without them. That's the price of the better bits per byte above.
- **The stem cuts cover only words TamilMorph lists,** about 1 running word in 4. The rest trained as plain
  whole-letter BPE.
- **English and code** cost a little more than with large multilingual tokenizers: 1.42 tokens per English
  word, against 1.34 for Gemma 3's 262k vocabulary.
- **The romanized Tamil it saw is synthetic,** transliterated word by word, not real Tanglish chat.
- **Only the 4,000 most common letters** in the training text are kept whole. Of the 299 Tamil letters (the 247
  letters plus the Grantha consonants ஜ ஷ ஸ ஹ in all 13 forms), 4 missed the cut: ஙெ, ஙௌ, ஞௌ and ழௌ. Each is split into its consonant and its vowel sign.
  Rarer letters from other scripts fall back to code points or bytes.
- **The model evidence stops at 374M parameters,** with 3 seeds at 85M and 2 at 350M. Larger models may behave
  differently, and no model here is big enough for the tasks to tell tokenizers apart.
