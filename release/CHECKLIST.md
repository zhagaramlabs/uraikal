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
- [x] The website: on Jay's "Yes, update site" (2026-11-09), zhagaramlabs.com got an Ezhuthu model page, a home
  page entry, and a leaderboard of 20 tokenizers read from `results/experiments/tokenizer_d4_fresh.json`.

## 128k: tested, not released

- **Built:** `r4-morph-128k`, the same recipe as the 64k, via `experiments/tamil_tokenizer_morph.py 128000`
  (`results/experiments/tamil_tokenizer_morph_128k.json`). It spends 6.3% fewer tokens on fresh.web (1.552 against 1.656 tokens per word).
- **Tested (phase c2, 2026-11-09):** 350M, 2 seeds, equal compute to the 64k runs (`experiments/spot/jobs_c2.txt`).
  It's worse on every held-out set, with both seeds above both 64k seeds.
  - fresh.web 0.3627 against 0.3593 (+0.9%).
  - Literature 0.5095 against 0.5030 (+1.3%).
  - On Tamil it's behind even plain BPE at 64k.
  - Results: `results/experiments/bpb-c/r4-morph-128k-350m{,-s1}.json`.
- **So there's no 128k release.** At 350M its output layer is 30% of the model, and it reads about 9% less text at
  equal compute. It might pay at 1B and up; that's untested.
