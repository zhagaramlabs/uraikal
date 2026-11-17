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
- qwen3.5
- vocabulary-extension
---

# Qwen3.5 + Tamil 32k

Qwen3.5's tokenizer with 32,000 more tokens for Tamil. It writes Tamil text in **2.7 times fewer tokens**: 1.53
tokens per word on held-out Tamil web text, against 4.13 for stock Qwen3.5. Ids 0 to 248,076 are stock Qwen3.5's,
unchanged, and the new tokens take ids 248,077 to 280,076. Text without Tamil gets stock Qwen's ids, with one
exception, given below.

**This is a tokenizer, not a model.** Stock Qwen3.5 weights have no trained rows for the new ids: from 248,320 up
they are past the end of its embedding matrix. To use it, you grow the model's embeddings and train. Below is how
we did that with Qwen3.5-0.8B-Base, and what it gave. We don't release those weights.

Built by [Zhagaram Labs](https://zhagaramlabs.com), a non-profit open AI lab for Tamil. It is not an official
Qwen release.

## Use

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("zhagaramlabs/qwen3.5-ta-32k")
ids = tok("தமிழ் எழுத்து")["input_ids"]
len(ids)         # 2; stock Qwen3.5 makes 6
tok.decode(ids)  # 'தமிழ் எழுத்து'
```

It needs no remote code. We checked that it loads and gives the same ids with transformers 4.46.3, 4.57.1 and
5.22.1. Its special tokens and chat template are Qwen3.5-0.8B-Base's.

### Adding it to a Qwen3.5 model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("zhagaramlabs/qwen3.5-ta-32k")
stock = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", dtype=torch.float32)

k = len(stock)                                          # 248,077: the new tokens start here
model.resize_token_embeddings(len(tok), mean_resizing=False)
emb = model.get_input_embeddings().weight               # tied: the output layer shares these rows
with torch.no_grad():
    for i in range(k, len(tok)):
        pieces = stock.backend_tokenizer.model.tokenize(tok.convert_ids_to_tokens(i))
        emb[i] = emb[[p.id for p in pieces]].mean(0)    # start each new row from the Qwen pieces it replaces
```

Then train, and give the new rows bigger steps than the rest of the model. After each optimizer step we moved
rows 248,077 and up 10 times as far as the optimizer had moved them. The embedding matrix is tied, so its rows
can't go in a separate learning-rate group; scaling their update does the same job. With the same step for every
row, the extended model fell behind stock Qwen (see below).

## What's inside

- **Stock Qwen3.5, unchanged:** its 248,077 tokens keep their ids, and its 247,587 merges come first, in their
  order. That includes its 33 added tokens, such as `<|endoftext|>` and `<|im_start|>`, with the same special
  flags.
- **32,000 merges learned on Tamil words,** appended after Qwen's, so every Qwen merge takes priority. Each one
  made one new token.
- **How they were learned:** we split Tamil web text into words with Qwen's own pre-tokenizer, kept the words that
  are more than half Tamil, and continued Qwen's BPE over its own tokens of those words.
- **Text without Tamil:** every new token but one can only match text that has Tamil in it, so text without Tamil
  gets stock Qwen's ids. The exception is `Delhi` (id 273,964), learned from words like Delhiயில். Where Delhi has
  no space before it, it's often one token instead of stock's two: at the start of a text or line, or after a
  digit, a double quote and most punctuation. ` Delhi` with a space is the same as stock, as are `-Delhi`,
  `/Delhi`, `'Delhi` and `(Delhi`.
- **Normalization:** NFC. **Pre-tokenizer:** Qwen3.5's. **Byte-level:** any text decodes back to its NFC form.

## Results

### Tokens

Fewer is better. Each test has 2,000 or 1,000 held-out passages; the code test has 27 files:

- **fresh.web:** FineWeb-2's Tamil test split, minus any passage that shares 8 words in a row with our model
  training text.
- **D4.web:** Sangraha shard 52, never used for training.
- **D4.literature:** Project Madurai lines, held out of training.

| | Stock Qwen3.5 | This tokenizer |
|---|---|---|
| fresh.web, tokens per word | 4.132 | **1.535** (−63%) |
| D4.web, tokens per word | 4.276 | **1.670** (−61%) |
| D4.literature, tokens per word | 4.344 | **1.934** (−55%) |
| fresh.web, Tamil words with a token boundary inside a letter | 63.2% | **7.8%** |
| English, tokens per byte | 0.2246 | 0.2246, the same ids |
| Romanized Tamil, tokens per byte | 0.3745 | 0.3745, the same ids |
| Code, tokens per byte (17 of the 27 files have Tamil in them) | 0.2901 | 0.2720 |

- **Against other tokenizers on fresh.web,** from the table on
  [Ezhuthu's card](https://huggingface.co/zhagaramlabs/ezhuthu-64k): it's close to Agaram's 64k (1.532) and ahead
  of Ezhuthu 64k (1.656), with more than four times their vocabulary.
- **A letter here** is a vowel, or a consonant alone or with its vowel sign or pulli (க, கா, கி, க்). On fresh.web,
  7.8% of Tamil words still have a token boundary inside a letter. We expect most are words the new merges don't
  cover, which fall back to Qwen's own pieces; we haven't checked.
- **Every one of the 10,357 held-out passages** gives the same ids as the source file and decodes back to its NFC
  form.

### Training Qwen3.5-0.8B-Base with it

We continued pretraining Qwen3.5-0.8B-Base on our text, once with stock Qwen's tokenizer and once with this one,
for the same number of tokens. By bytes, the text is 79% Tamil (web text, Wikipedia and Project Madurai), 16%
English, 4% Python source and 2% romanized Tamil. Context 2,048, batch 256, learning rate 1e-4, one seed per arm.

We scored **bits per byte**: the model's total loss divided by the text's UTF-8 bytes. It doesn't depend on how
the text was split, so the two tokenizers can be compared. Lower is better.

On fresh.web:

| Tokens trained | Stock tokenizer | This one, same step for every row | This one, 10× step on new rows |
|---|---|---|---|
| 0 | 0.6080 | | |
| 26M (50 steps) | 0.4749 | 0.5910 | **0.4747** |
| 200M | 0.3846 | 0.3988 | **0.3712** (−3.5%) |
| 1B | 0.3435 | not run | **0.3391** (−1.3%) |

At 1B tokens, on every test:

| Test | Stock tokenizer | This one, 10× step | Change |
|---|---|---|---|
| fresh.web | 0.3435 | 0.3391 | −1.3% |
| D4.web | 0.3485 | 0.3450 | −1.0% |
| D4.literature | 0.4801 | 0.4804 | +0.1% |
| Thirukkural verse | 0.7857 | 0.7717 | −1.8% |
| Romanized Tamil | 2.1297 | 2.1322 | +0.1% |
| English | 0.9477 | 0.9335 | −1.5% |
| Code | 0.8191 | 0.8001 | −2.3% |

- **The start matters:** new rows that start from the Qwen pieces they replace reached 0.591 on fresh.web after 50
  steps. Rows that start from the mean embedding reached 0.747, worse than untrained Qwen.
- **The bigger step matters:** with the same step for every row, the extended model was behind stock at 200M
  tokens. With 10× steps on the new rows, it was ahead.
- **The lead shrank with training,** from −3.5% at 200M tokens to −1.3% at 1B. With one seed per arm, a 1.3% gap
  isn't settled.
- **Equal tokens means more text for the extended model.** Our training text mixes Tamil, English, code and
  romanized Tamil. 1B tokens were 38% of the extended tokenizer's stream but 22% of stock's, so the extended model
  read about 1.76 times as much of the same text. That's why English and code, which tokenize the same or nearly
  so, improved too, and likely part of the Tamil gain.
- **Equal tokens isn't equal GPU time.** In our code the extended model took 24.4 s per step against 17.1 s, each
  on an H200: 43% slower. Its FLOPs were only 4.3% higher. We put most of the gap down to our code computing fp32
  logits over all 280k rows.
  Extrapolating stock's 200M and 1B points log-linearly, stock with 43% more tokens would reach about 0.334, below
  0.3391. So at equal GPU time in our code, stock might tie or lead. That is an extrapolation; we didn't run it.
- **What holds:** the same Tamil text in 2.7 times fewer tokens, with bits per byte equal or a little better. So
  the same context holds about 2.7 times as much Tamil, and a model writes a Tamil word in fewer decoding steps.
- **The tasks don't separate them at this size.** On Tamil Belebele (900 items), XCOPA (600) and Global PIQA (203),
  stock scored 0.329, 0.522 and 0.424, and the extended model 0.333, 0.553 and 0.419.

## Training text

The 32,000 merges were learned on one shard of Sangraha's verified Tamil (AI4Bharat, `verified/tam/data-0`,
CC BY 4.0): 149,795 documents, 1.0 GB, using only the words that are more than half Tamil. No Project Madurai text
went into the tokenizer. D4.web is a different shard, and fresh.web was filtered against text that includes this
one; we didn't check the verse and literature tests against it. The files hold a vocabulary and a merge list, not
passages of text.

The models above trained on more: Sangraha verified Tamil `data-0` to `data-18` without `data-5`, Tamil
Wikipedia, Project Madurai minus the D4.literature lines and the Thirukkural books, FineWeb English, Python source,
and romanized Tamil made from Sangraha with Aksharantar. Those weights aren't released.

## Licence

Apache 2.0, the licence of Qwen3.5. It is built on the tokenizer of
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), which loads in transformers to the same tokenizer as
[Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base): the same 248,077 tokens, pre-tokenizer and
decoder. We checked. Our changes:

- 32,000 merges and tokens appended to `tokenizer.json`;
- `tokenizer_config.json` is Qwen3.5-0.8B-Base's, with `tokenizer_class` set to `PreTrainedTokenizerFast`. With
  Qwen's own class, transformers 5 rebuilds the tokenizer, and the rebuilt one splits Tamil into more tokens than
  stock Qwen.

We credit the Qwen team for Qwen3.5 and AI4Bharat for Sangraha.

## Conflict of interest

Zhagaram Labs built this tokenizer and also built the tests it's scored on: D4, fresh.web and the bits-per-byte
setup. To keep the test sets out of future training data, we don't release them. The code that builds and scores
them is in the [Uraikal](https://github.com/zhagaramlabs/uraikal) repository, so anyone can rebuild them from their
public sources.

## Reproduce

Run these in the Uraikal repository:

```bash
# round 1's tokenizers, this one among them: data/tamil_tokenizer/qwen3.5+ta-32k.json
uv run --extra tokenizers --extra prep --with huggingface_hub python experiments/tamil_tokenizer.py
# this folder, from that file, with its checks
python release/make_qwen_ta.py
# the 1B runs, on a GPU, as in experiments/spot/jobs_d5.txt: the extended arm, then stock
uv run --extra tokenizers --extra prep --with torch python experiments/tokenizer_bpb.py train data/tamil_tokenizer/qwen3.5+ta-32k.json --seed 0 --base Qwen/Qwen3.5-0.8B-Base --data bpb-c --steps 1908 --lr 1e-4 --micro 4 --no-compile --log-every 10 --init pieces --new-lr 10 --save-minutes 10 --name qd-pieces-nlr10-1b
uv run --extra tokenizers --extra prep --with torch python experiments/tokenizer_bpb.py train Qwen/Qwen3.5-0.8B-Base --seed 0 --base Qwen/Qwen3.5-0.8B-Base --data bpb-c --steps 1908 --lr 1e-4 --micro 4 --no-compile --log-every 10 --save-minutes 10 --name qd-stock-1b
```

The released `tokenizer.json` has sha256 `332504101d5304795b5f6f94c8b69090f5190d76ba354b04d72c473f475a4ffc`.

## Limits

- **It needs training.** The new rows mean nothing until a model learns them, and the model grows: 32,000 more rows
  in the embedding and output layers. Qwen3.5-0.8B's matrix already has 243 spare rows, so it grows by 31,757 rows,
  32.5M parameters.
- **The evidence is small:** one model size (0.8B), one seed per arm, up to 1B tokens.
- **It was learned on web text.** Literature and verse get fewer gains: 1.93 tokens per word on D4.literature,
  against 1.67 on D4.web.
- **fresh.web comes from FineWeb-2,** which is public, so Qwen3.5 may have seen it. That affects both arms the same
  way, so the comparison holds, but the absolute numbers may be lower than on unseen text.
