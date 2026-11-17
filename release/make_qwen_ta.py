"""Builds release/qwen3.5-ta-32k/ from data/tamil_tokenizer/qwen3.5+ta-32k.json, then checks the folder.

qwen3.5+ta-32k is Qwen3.5's tokenizer with 32,000 BPE merges learned on Tamil words, appended after Qwen's own
(experiments/tamil_tokenizer.py, extend()). Phase d (d1-d5) trained Qwen3.5-0.8B-Base with it. The release is that
file byte for byte, next to Qwen3.5-0.8B-Base's own tokenizer_config.json (special tokens, chat template) with
its class set to PreTrainedTokenizerFast. Checks:
- AutoTokenizer loads the folder with trust_remote_code=False;
- ids 0-248,076 are stock Qwen3.5-0.8B-Base's tokens and special tokens, as transformers loads it, and the chat
  template gives the same text;
- every new token but one (Delhi, id 273,964) holds Tamil bytes, so text without Tamil gets stock Qwen's ids, unless
  it has Delhi with no space before it;
- on every held-out slice in data/bpb-c/eval.json (private: it stays on our machines) it gives exactly the source
  file's ids, and every passage decodes back to its NFC form. English and romanized Tamil get stock Qwen's ids.

    .venv/bin/python release/make_qwen_ta.py

Writes results/experiments/qwen3.5_ta_32k_check.json.
"""
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from uraikal.tokenizer_eval import measure

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/tamil_tokenizer/qwen3.5+ta-32k.json"
OUT = ROOT / "release/qwen3.5-ta-32k"
BASE = "Qwen/Qwen3.5-0.8B-Base"
SHOWN = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that byte-level BPE writes as themselves
BYTE = {chr(b): b for b in SHOWN} | {chr(256 + i): b for i, b in enumerate(b for b in range(256) if b not in SHOWN)}


def token_bytes(token):
    return bytes(BYTE[c] for c in token)


def main():
    config = Path(hf_hub_download(BASE, "tokenizer_config.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC, OUT / "tokenizer.json")
    settings = json.load(open(config, encoding="utf-8"))
    # Qwen2Tokenizer rebuilds the BPE in transformers 5, and the rebuilt one makes more tokens than stock Qwen on
    # Tamil (0.224 against 0.173 per byte on D4.web); the plain fast class reads tokenizer.json as it is.
    settings["tokenizer_class"] = "PreTrainedTokenizerFast"
    json.dump(settings, open(OUT / "tokenizer_config.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    source = Tokenizer.from_file(str(SRC))
    stock = AutoTokenizer.from_pretrained(BASE)
    loaded = AutoTokenizer.from_pretrained(OUT, trust_remote_code=False)
    k = len(stock)
    assert (k, len(loaded)) == (248_077, 280_077), (k, len(loaded))
    assert loaded.convert_ids_to_tokens(list(range(k))) == stock.convert_ids_to_tokens(list(range(k)))
    assert loaded.special_tokens_map == stock.special_tokens_map and loaded.eos_token_id == stock.eos_token_id
    assert {i: (t.content, t.special) for i, t in loaded.added_tokens_decoder.items()} == \
           {i: (t.content, t.special) for i, t in stock.added_tokens_decoder.items()}
    chat = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    assert loaded.apply_chat_template(chat, tokenize=False) == stock.apply_chat_template(chat, tokenize=False)
    # In valid UTF-8, E0 AE and E0 AF only ever start a character in U+0B80-U+0BFF, the Tamil block.
    new = [token_bytes(t) for t in loaded.convert_ids_to_tokens(list(range(k, len(loaded))))]
    no_tamil = [b.decode() for b in new if b"\xe0\xae" not in b and b"\xe0\xaf" not in b]
    assert no_tamil == ["Delhi"], no_tamil  # from words like Delhiயில்; it only fires with no space before it

    report = {"source": str(SRC.relative_to(ROOT)),
              "tokenizer_json_sha256": hashlib.sha256((OUT / "tokenizer.json").read_bytes()).hexdigest(),
              "tokenizer_config_from": f"{BASE}@{config.parent.name}",
              "base_tokens": k, "vocab_size": len(loaded), "new_tokens_without_tamil": no_tamil,
              "slices": {}, "d4": {}}
    held = json.load(open(ROOT / "data/bpb-c/eval.json", encoding="utf-8"))
    for name, texts in held.items():
        same = exact = as_stock = n_ids = n_stock = n_bytes = 0
        for text in texts:
            ids = loaded(text, add_special_tokens=False)["input_ids"]
            old = stock(text, add_special_tokens=False)["input_ids"]
            same += ids == source.encode(text).ids
            exact += loaded.decode(ids) == unicodedata.normalize("NFC", text)
            as_stock += ids == old
            n_ids, n_stock, n_bytes = n_ids + len(ids), n_stock + len(old), n_bytes + len(text.encode())
        row = {"passages": len(texts), "same_ids": same, "exact_round_trip": exact, "same_ids_as_stock": as_stock,
               "tokens_per_byte": round(n_ids / n_bytes, 4), "stock_tokens_per_byte": round(n_stock / n_bytes, 4)}
        report["slices"][name] = row
        print(name, row, flush=True)
        assert same == exact == len(texts), name
        assert name not in ("english", "romanized") or as_stock == len(texts), name
    for name in ("fresh.web", "D4.web", "D4.literature"):
        report["d4"][name] = {"extended": measure(loaded, held[name]), "stock": measure(stock, held[name])}
        print(name, {w: {m: round(v[m], 4) for m in ("tokens_per_word", "letter_cut_words")}
                     for w, v in report["d4"][name].items()}, flush=True)
    sample = "தமிழ் எழுத்து: க் கா கி கீ கு கூ கெ கே கை கொ கோ கௌ, ஸ்ரீ, க்ஷ. Tamil and English, 2026."
    assert loaded.decode(loaded(sample)["input_ids"]) == sample
    report["sample_tokens"] = {"extended": loaded.tokenize("தமிழ் எழுத்து"), "stock": stock.tokenize("தமிழ் எழுத்து")}
    json.dump(report, open(ROOT / "results/experiments/qwen3.5_ta_32k_check.json", "w"), indent=1, ensure_ascii=False)
    print(report["tokenizer_json_sha256"], report["sample_tokens"])


if __name__ == "__main__":
    main()
