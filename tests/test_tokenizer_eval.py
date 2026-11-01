"""Self-checks for the D4 metrics, with a stand-in tokenizer (no downloads). Run: uv run python tests/test_tokenizer_eval.py"""
from uraikal.tokenizer_eval import measure


class Fake:
    """Splits text into the given character spans; span (s, e) becomes one token."""

    unk_token_id = 0

    def __init__(self, spans, unk=()):
        self.spans, self.unk = spans, set(unk)

    def __call__(self, text, **_):
        return {"input_ids": [0 if i in self.unk else i + 1 for i in range(len(self.spans))], "offset_mapping": self.spans}

    def decode(self, ids, **_):
        return self.text


TEXT = "தமிழ் ok"  # த ம ி ழ ் at 0-4, space, o k at 6-7

# One token per code point: boundaries before ி (2) and ் (4) cut letters.
per_char = Fake([(i, i + 1) for i in range(5)] + [(6, 8)])
per_char.text = TEXT
r = measure(per_char, [TEXT])
assert r["letter_cut_words"] == 1.0 and r["byte_cut_words"] == 0.0, r
assert r["tokens_per_word"] == 3.0 and r["round_trip"] == 1.0 and r["words"] == 2, r
assert r["pretokenizer_letter_cut_words"] is None, r  # the stand-in has no pre-tokenizer
assert r["bytes_per_token"] == (15 + 3) / 6, r  # five 3-byte Tamil code points, a space and "ok"

# Letter-aligned tokens: த | மி | ழ் | ok. No cuts.
aligned = Fake([(0, 1), (1, 3), (3, 5), (6, 8)])
aligned.text = TEXT
r = measure(aligned, [TEXT])
assert r["letter_cut_words"] == 0.0 and r["byte_cut_words"] == 0.0 and r["tokens_per_word"] == 2.0, r

# Byte-level BPE splitting the bytes of த across two tokens: both claim character 0.
bytes_split = Fake([(0, 1), (0, 1), (1, 5), (6, 8)], unk=[3])
bytes_split.text = "தமிழ்  ok"  # decodes with different spacing: still a round trip
r = measure(bytes_split, [TEXT])
assert r["byte_cut_words"] == 1.0 and r["letter_cut_words"] == 0.0, r
assert r["unk_per_1k"] == 250.0 and r["round_trip"] == 1.0, r

# A lossy decode fails the round trip.
lossy = Fake([(0, 5), (6, 8)])
lossy.text = "தமழ ok"
assert measure(lossy, [TEXT])["round_trip"] == 0.0

# Real pre-tokenizers, if the `tokenizers` extra is installed: GPT-2's regex cuts Tamil letters, whitespace splitting doesn't.
try:
    from tokenizers import pre_tokenizers
except ImportError:
    pre_tokenizers = None
if pre_tokenizers:
    for pre, expected in ((pre_tokenizers.ByteLevel(), 1.0), (pre_tokenizers.WhitespaceSplit(), 0.0)):
        whole = Fake([(0, 5), (6, 10)])
        whole.text, whole.backend_tokenizer = "தமிழ் மொழி", type("Backend", (), {"pre_tokenizer": pre})()
        assert measure(whole, ["தமிழ் மொழி"])["pretokenizer_letter_cut_words"] == expected, pre

print("all checks passed")
