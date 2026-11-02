# Uraikal (உரைகல்): Tamil speech benchmark, v0 spec

> **Status: draft.** Tracks, slices and scoring may change before v1.0. Scores are comparable only within one `uraikal_version`, and every v0.x number is preliminary.

Uraikal is the touchstone every Zhagaram model is scored against. v0 covers speech: recognition (Sevi), synthesis (Oosai) and spoken conversation (Kural). Text tracks for Mozhi and Maatru come in v1.

## What it measures

| Track | Slice | What it catches | Data |
|---|---|---|---|
| A. Recognition | A1 Read speech | Baseline accuracy on clean audio | Existing sets |
| | A2 Spontaneous speech | Real talk: false starts, fillers, spoken grammar | Existing sets |
| | A3 Dialects | Accent and vocabulary by region | **New** |
| | A4 Tanglish | English words mixed into Tamil | **New** |
| | A5 Phone and noise | 8 kHz phone audio, street and home noise | Derived + **new** |
| | A6 Minimal pairs | ழ/ள/ல, ண/ன, ற/ர distinctions | **New** |
| | A7 Numbers and names | Phone numbers, amounts, dates, places, people | **New** |
| B. Synthesis | B1 Intelligibility | Can a listener (proxied by ASR) understand it | Text prompts |
| | B2 Naturalness | Does it sound human and like *spoken* Tamil | Human raters |
| | B3 Reading tricky text | Numbers, dates, ₹, abbreviations, English words | Text prompts |
| C. Conversation | C1 Latency | Time from end of user speech to first reply audio | Scripted calls |
| | C2 Turn-taking | Stops when interrupted, keeps going on "ம்ம்" | Scripted calls |
| | C3 Answer quality | Correct, helpful, in spoken register, understood the dialect | **New** spoken prompts |
| | C4 Safety | Refuses harmful requests and impersonation, in Tamil | **New** spoken prompts |

v0.1 also runs **D4, tokenizer efficiency**, on text, and four slices of **track I, the Thirukkural**: I1 exact recall, I2 attribution, I3 metre and I5 refusing to invent verses. D4's metrics are defined in the [README](README.md#scoring-rules); track I's are in `src/uraikal/classics.py`.

### A3 dialect slices

Nine regions, each about 3 hours of test audio from at least 25 speakers, balanced by gender and spread across age bands (18–30, 31–50, 51+):

Chennai, Kongu (Coimbatore, Erode, Tiruppur), Madurai, Cauvery delta (Thanjavur, Tiruvarur), Tirunelveli and Kanyakumari, Jaffna, Batticaloa, Malaysia, Singapore.

Half of each slice is read prompts written in that dialect, half is unscripted (describe your day, give directions, tell a story). Transcribe exactly what was said, in Tamil script, without "correcting" it to written Tamil.

### A4 Tanglish

About 3 hours of speech where English words are mixed into Tamil: "naan office-ku late-aa poren". Every utterance carries two references:

- **Mixed script:** Tamil words in Tamil script, English words in Latin (`நான் office-க்கு late-ஆ போறேன்`).
- **Tamil script only:** English words transliterated (`நான் ஆபீஸுக்கு லேட்டா போறேன்`).

A system is scored against whichever convention it declares. Both scores are reported.

### A5 phone and noise

- **Derived (cheap, reproducible):** A1 and A3 audio passed through phone codecs (G.711 µ-law, AMR-NB) at 8 kHz, then mixed with noise at 20, 10, 5 and 0 dB SNR. Noise comes from a CC-licensed corpus; check the licence before release.
- **Real (2 hours):** people calling a phone line from a bus, a market, a kitchen, a two-wheeler.

### A6 minimal pairs

Words that differ only in a hard sound, read inside a neutral carrier sentence so the language model can't guess from context: "இப்போது ___ என்று சொல்லுங்கள்" ("Now say ___").

Starter set, to be reviewed and extended to about 150 items by a Tamil linguist:

| Contrast | Words |
|---|---|
| ழ / ள / ல | வாழ் (live), வாள் (sword), வால் (tail) |
| ழ / ல | மழை (rain), மலை (mountain) |
| ள / ல | வளி (wind), வலி (pain) |
| ண / ன | மணம் (fragrance), மனம் (mind) |
| ற / ர | கறி (curry), கரி (charcoal); அறம் (virtue), அரம் (file, the tool) |

Read by 20 speakers drawn from the dialect slices. Score: share of target words transcribed exactly, plus a confusion matrix for each contrast. The same set later trains and tests Uchcharippu.

### A7 numbers and names

Utterances that contain phone numbers, amounts in rupees, dates, times, addresses, and names of people and places. Score: share of these entities transcribed exactly (after normalisation). This matters more to a voice assistant than overall WER does.

### B. Synthesis

- **B1:** 300 held-out sentences (written and spoken register). Transcribe the synthetic audio with one pinned ASR model (IndicConformer Tamil, CTC decoder) and report CER. The ASR version is fixed per Uraikal release so scores stay comparable.
- **B2:** 40 sentences per system, each rated 1–5 by at least 10 native speakers on two scales: *natural* (sounds like a person) and *spoken* (sounds like how people talk, not a newsreader). Real recordings of the same sentences are mixed in unannounced as a ceiling. Report the mean with a 95% confidence interval. Automatic MOS predictors aren't validated for Tamil, so they're reported but not ranked.
- **B3:** 200 items such as "₹1,250", "2026-10-29", "10:45", "Dr.", "TNPSC", "WhatsApp". A human checks each item was read correctly. Score: share read correctly.
- Voice similarity isn't scored, because Oosai ships preset voices, not cloning.

### C. Conversation

- **C1:** 200 scripted turns over a fixed network profile. Measure end of user speech → first reply audio. Report p50 and p90. Also log each stage's time (ASR final, LLM first token, TTS first audio).
- **C2:** 100 interruptions mid-reply. Measure time until the system stops talking, and whether its next turn addresses the interruption. Separately, 100 backchannels ("ம்ம்", "ஆமா", "சரி", "அப்படியா") played mid-reply: the system should *not* stop. Report false-stop rate.
- **C3:** 300 spoken prompts recorded by speakers from the A3 regions: everyday tasks, local knowledge (Tamil Nadu, Sri Lanka, the diaspora), multi-turn follow-ups, Tanglish. Native raters score each reply for task success, spoken register, whether it understood the dialect, and fluency. An LLM judge on transcripts is a cheap second signal, calibrated against the human scores on a 20% overlap and never used alone.
- **C4:** 100 prompts covering harmful requests and requests to imitate a real person's voice. Score: correct refusals and false refusals.

## Scoring

Text normalisation before WER and CER, published with the eval code:

1. Unicode NFC. Strip zero-width joiners and non-joiners.
2. Remove punctuation. Lower-case Latin text.
3. Map Tamil numerals (௧௨௩) to Arabic numerals. Spell-out vs digits: both forms accepted for A7.
4. No splitting or merging of compound words. Tamil is agglutinative, so one wrong suffix fails a whole word: that's why CER is always reported next to WER.

Don't use Whisper's `BasicTextNormalizer`. It turns Tamil vowel signs into spaces, which splits words apart, so Tamil WERs computed with it can't be compared with anyone else's. Whisper is re-scored here with the Uraikal normaliser.

The headline recognition score is the **macro average WER over the A3 dialect slices**, so a system can't top the board on clean read speech alone.

**Generated answers** apply to every track where a model writes a free-form or reasoned answer (R, I, and the text tracks planned for v1). Part of a trained model's gain can be learning to *finish* within the output limit rather than reasoning better (A-RSI, [arXiv:2610.38568](https://arxiv.org/abs/2610.38568)). So every such result reports:

1. **The output limit** (`max_new_tokens`, plus the think budget if one is used).
2. **The truncation rate:** the share of answers cut off before a final answer.
3. **Scores at two or more output limits.** Use at least the standard limit and double it.
4. **The base model, scored at the same limits.** A gain is quoted against that base, never against a base run with a shorter limit.

A gain that shrinks sharply as the limit grows is partly a gain in finishing, and the leaderboard says so.

## Leaderboard

One table per track. Recognition example:

| System | Type | Version | A3 macro WER | A3 macro CER | A4 Tanglish WER | A5 noisy WER @5 dB | A6 pair accuracy | A7 entity accuracy |
|---|---|---|---|---|---|---|---|---|
| Sevi v0 | Open weights | 2026-12 | | | | | | |

Each cell links to per-slice scores. Each result is stored as JSON:

```json
{
  "system": "sevi-v0",
  "type": "open-weights",
  "uraikal_version": "0.1",
  "track": "A",
  "slice": "A3.kongu",
  "split": "test",
  "metrics": {"wer": 0.0, "cer": 0.0},
  "n_utterances": 0,
  "date": "2026-12-01"
}
```

**Submitting:** run the published harness on `dev` yourself. For `test`, send a container or an API endpoint and we score it. Every score lists the Uraikal version.

**Contamination:** `test` audio and text are never published. Released text carries a canary string so we can detect it in training data. The `test` set is refreshed once a year and old versions become public.

## Baselines to score first

Each baseline is labelled "Open weights", "API" or "Research only" (a licence that forbids commercial use: scored for comparison, never shipped).

| Track | System | Type | Licence |
|---|---|---|---|
| A | IndicConformer Tamil, CTC and RNNT decoders | Open weights | MIT |
| A | IndicWhisper (Vistaar) | Open weights | MIT |
| A | Whisper large-v3 and large-v3-turbo | Open weights | MIT |
| A | MMS-1b-all | Research only | CC BY-NC 4.0 |
| A | One or two hosted Indic ASR APIs | API | Commercial terms |
| B | Indic Parler-TTS, voices Kavitha and Jaya | Open weights | Apache 2.0 |
| B | IndicF5 with a consented reference voice | Open weights | MIT, voice-consent terms |
| B | Indic-Speak (AI4Bharat's newest) | Open weights | Check licence first |
| B | MMS-TTS Tamil | Research only | CC BY-NC 4.0 |
| B | One hosted Indic TTS API | API | Commercial terms |
| C | Kural v0 | Ours | |
| C | Hosted realtime voice assistants that support Tamil | API | Commercial terms |

For reference, Vistaar reports IndicWhisper Tamil WER of 17.2–29.2% across its six Tamil test sets. Those numbers use Vistaar's own normaliser, so expect differences when re-scored here.

## Data sources

Checked against the primary dataset pages on 2026-10-29. Re-check before each Uraikal release.

| Dataset | Used for | Licence | Tamil test data | Caveat |
|---|---|---|---|---|
| [Kathbath](https://github.com/AI4Bharat/IndicSUPERB) | A1, A5 derived | CC0 per GitHub (HF tag says CC BY 4.0; both allow commercial use) | Test-known: 20 speakers, 5 h. Test-unknown: 20 speakers, about 3.2 h | Test sets ship as all-language tarballs on GitHub, not on HF |
| [FLEURS `ta_in`](https://huggingface.co/datasets/google/fleurs) | A1 | CC BY 4.0 | 591 test utterances | Read speech; test speakers are not in train |
| [Common Voice Tamil v27.0](https://mozilladatacollective.com/datasets?q=Tamil) | A1 | CC0 | 12,241 test clips (234.8 h validated overall) | Now only on Mozilla Data Collective. Splits change every release: pin v27.0 |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) | A2 | CC BY 4.0, gated | **No test split on HF.** Use `valid` (5,276 utterances) | Systems trained on IndicVoices may have been tuned on `valid`: flag their A2 scores |
| [Vistaar](https://github.com/AI4Bharat/vistaar) | Cross-check of the harness | MIT for code and models; data keeps each source's licence | Kathbath, Kathbath-Hard, Common Voice, FLEURS, IndicTTS, MUCS | Its zips repackage third-party data |
| [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | B1 and B2 sentences and real-speech references | CC BY 4.0, gated | 293 test utterances (99.5 h Tamil overall) | Audio was enhanced, so it's a slightly flattering ceiling |
| [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) | B2 expressive references | CC BY 4.0, gated | 3,656 test utterances, 2 speakers | |
| [IndicTTS (IIT Madras)](https://www.iitm.ac.in/donlab/indictts/database) | Optional B reference | Custom EULA (permits commercial use; keep IITM notice) | Unconfirmed | Read from an archived copy of the licence: needs counsel review |
| [OpenSLR 65](https://openslr.org/65) | Optional B reference | CC BY-SA 4.0 | No official splits | Share-alike applies to redistributed derivatives |
| [Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi) | Not used for testing | CC BY 4.0 | Train only | Broadcast audio rights not addressed on the card |

Models named in the baselines: [IndicConformer Tamil](https://huggingface.co/ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large), [Whisper](https://github.com/openai/whisper), [MMS-1b-all](https://huggingface.co/facebook/mms-1b-all), [MMS-TTS Tamil](https://huggingface.co/facebook/mms-tts-tam), [Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts), [IndicF5](https://huggingface.co/ai4bharat/IndicF5).

## Milestones

1. **v0.1 (weeks 1–4):** eval harness and normaliser, A1, A2, A5-derived and B1 on existing sets, baseline numbers for every system above.
2. **v0.2 (weeks 5–8):** A6 minimal pairs, A7 entities, B2 and B3 with the first rater pool.
3. **v0.3 (weeks 6–14):** A3 dialect and A4 Tanglish recordings, collected and transcribed.
4. **v1.0:** public leaderboard, C track running against Kural v0, text tracks specified.

New recording comes to roughly 40 hours of transcribed test audio plus the conversation prompts.
