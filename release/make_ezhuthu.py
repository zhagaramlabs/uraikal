"""Builds release/ezhuthu-64k/ from data/tamil_tokenizer/r4-morph-64k.json, then checks the folder.

Ezhuthu is r4-morph-64k: whole-letter BPE whose merges never joined a stem to its suffix in training (see
experiments/tamil_tokenizer_morph.py). Jay chose it over r2-grapheme-64k on 2026-11-08: about 0.2% better bits per
byte on Tamil prose at 85M and 350M, for 6 to 7% more tokens. The release is r4-morph-64k unchanged plus one special
token, <|endoftext|>, at id 64000. That's the id the 85M and 350M models trained with as their document marker
(tokenizer_bpb.py's bos, one past the vocabulary), so no other id moves. Checks, on every held-out slice in
data/bpb/eval.json (private: it stays on our machines):
- AutoTokenizer loads the folder with trust_remote_code=False;
- it gives exactly the ids the source tokenizer gives;
- every passage decodes back to its NFC form, character for character (the tokenizer normalizes to NFC).

    uv run --extra tokenizers python release/make_ezhuthu.py

Writes results/experiments/ezhuthu_64k_check.json.
"""
import hashlib
import json
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/tamil_tokenizer/r4-morph-64k.json"
OUT = ROOT / "release/ezhuthu-64k"
EOT = "<|endoftext|>"


def main():
    source = Tokenizer.from_file(str(SRC))
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(str(SRC)), bos_token=EOT, eos_token=EOT,
                                  clean_up_tokenization_spaces=False)
    assert tok.convert_tokens_to_ids(EOT) == source.get_vocab_size() == 64_000
    tok.save_pretrained(OUT)
    config = json.load(open(OUT / "tokenizer_config.json"))
    config["tokenizer_class"] = "PreTrainedTokenizerFast"  # transformers 5 writes TokenizersBackend, which 4.x can't load
    json.dump(config, open(OUT / "tokenizer_config.json", "w"), indent=2)

    loaded = AutoTokenizer.from_pretrained(OUT, trust_remote_code=False)
    assert len(loaded) == 64_001 and loaded.eos_token_id == loaded.bos_token_id == 64_000
    held = json.load(open(ROOT / "data/bpb/eval.json", encoding="utf-8"))
    report = {"source": str(SRC.relative_to(ROOT)),
              "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
              "tokenizer_json_sha256": hashlib.sha256((OUT / "tokenizer.json").read_bytes()).hexdigest(),
              "vocab_size": len(loaded), "slices": {}}
    for name, texts in held.items():
        same = exact = already_nfc = 0
        for text in texts:
            ids = loaded(text, add_special_tokens=False)["input_ids"]
            same += ids == source.encode(text).ids
            nfc = unicodedata.normalize("NFC", text)
            already_nfc += text == nfc
            exact += loaded.decode(ids) == nfc
        report["slices"][name] = {"passages": len(texts), "same_ids": same, "exact_round_trip": exact,
                                  "input_already_nfc": already_nfc}
        print(name, report["slices"][name], flush=True)
        assert same == exact == len(texts), name
    sample = "தமிழ் எழுத்து: க் கா கி கீ கு கூ கெ கே கை கொ கோ கௌ, ஸ்ரீ, க்ஷ. Tamil and English, 2026."
    assert loaded.decode(loaded(sample)["input_ids"]) == sample
    report["sample_tokens"] = loaded.tokenize(sample)
    json.dump(report, open(ROOT / "results/experiments/ezhuthu_64k_check.json", "w"), indent=1, ensure_ascii=False)
    print(report["tokenizer_json_sha256"], report["sample_tokens"])


if __name__ == "__main__":
    main()
