# Publishing Ezhuthu

**Ezhuthu 64k is public:** [zhagaramlabs/ezhuthu-64k](https://huggingface.co/zhagaramlabs/ezhuthu-64k), uploaded
on 2026-11-09 after Jay's "Yes, upload" (commit `78616b5`). Any further publish still needs Jay's OK: GitHub
releases, the website and social posts.

## 64k: decisions for Jay

- [x] **The name** எழுத்து (*Ezhuthu*), and the repo name `zhagaramlabs/ezhuthu-64k`. Jay confirmed the name on
  2026-11-08 ("Yes, Ezhuthu"). That isn't an OK to upload: the publish itself still needs one.
- [x] **The licence:** Apache 2.0, as the card says. Jay settled it on 2026-11-09: "regarding the license,
  everything is good to go. Just use it." The tokenizer files hold no passages of the training text, only a
  vocabulary and merges, and the card credits every source with its licence (TamilMorph is MIT).
- [x] **`<|endoftext|>` at id 64,000,** as bos and eos. This is the only change from the tested `r4-morph-64k`.
  Uploaded as is: the tokenizer doesn't add it by itself, and the card says to prepend it when training.
- [x] **Which tokenizer:** Jay chose the stem–suffix variant on 2026-11-08. So Ezhuthu is now `r4-morph-64k`, not
  `r2-grapheme-64k`. It's about 0.2% better in bits per byte on Tamil prose at 85M (3 seeds) and 350M (2 seeds), for
  6 to 7% more tokens of Tamil web text. The folder and the card were rebuilt for it.
- [x] **The card** went up with the upload: [ezhuthu-64k/README.md](ezhuthu-64k/README.md).

## 64k: before and after the upload

- [x] Fill in the card's "At 350M" section: done on 2026-11-08, when all 6 c1 runs were in.
- [x] Re-run `release/make_ezhuthu.py`: done on 2026-11-09. The upload matches `tokenizer.json` sha256 `ad7884f9…6bc7d`.
- [x] Re-check whether Krutrim's IndicSuperTokenizer (MUTANT-Indic, arXiv 2511.03237) has been released. On
  2026-11-09 it still wasn't on Hugging Face (krutrim-ai-labs has 9 repos, none a tokenizer). Krutrim-1's tokenizer
  stands in for it. Re-check before any update of the card.
- [ ] Optional: accept Aya Expanse's licence on Hugging Face, then re-score from the gated repo. The copy we
  scored has the same sha256, so the numbers won't change.
- [ ] The website leaderboard (`results/d4.json`) doesn't have the 4 new tokenizers yet. Updating it is a deploy,
  so it needs an OK too.

## 128k: what's open

- **No whole-letter 128k without superwords exists.** The only whole-letter 128k is `r3-grapheme-super-128k`,
  and superwords are out, because they added nothing at equal compute.
  - To build one to match the 64k release: `experiments/tamil_tokenizer_morph.py` at `SIZE = 128_000`, on round 2's
    text with the stem cuts. That's one CPU run on the Mac, about 35 minutes for the 64k.
  - Then score its D4 and fresh.web tokens.
- **No 128k has 3 seeds of bits per byte.** `r2-bpe-128k` has 1 seed: 0.4132 on fresh.web at 85M, against 0.4095
  for Ezhuthu 64k (3 seeds), with 28% more compute.
  - A fair test needs the whole-letter 128k and `r2-bpe-128k` at 3 seeds each, at equal steps: about 6 runs ×
    45 minutes on one H100, roughly $4 at the cap. That's a purchase, so it needs an OK.
- **The case for 128k is cost, not quality.** `r2-bpe-128k` spends 13% fewer tokens than Ezhuthu 64k on D4.web
  (1.556 against 1.783 tokens per word), and 7% fewer than `r2-grapheme-64k` (1.677). At 85M, the bigger output layer didn't buy better bits per byte. Whether it pays
  at 350M or 1B is still open.
