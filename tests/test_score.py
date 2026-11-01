"""Self-checks for the scoring normaliser and metrics. Run: uv run python tests/test_score.py"""
from uraikal.score import edit_distance, error_rates, score
from uraikal.text import normalize

# Vowel sign ொ typed as ெ + ா composes to one code point under NFC.
assert normalize("கொ") == "கொ" == normalize("கொ")
# Zero-width joiner dropped; pulli and vowel signs kept.
assert normalize("க்‍ஷ") == "க்ஷ"
assert normalize("தமிழ்") == "தமிழ்"
# Tamil numerals become 0-9.
assert normalize("௨௦௨௬") == "2026"
# Punctuation and symbols become spaces, runs of space collapse.
assert normalize("வணக்கம்!  எப்படி இருக்கீங்க?") == "வணக்கம் எப்படி இருக்கீங்க"
assert normalize("Naan office-ku late-aa poren.") == "naan office ku late aa poren"
assert normalize("₹1,250") == "1 250"
assert normalize("...") == ""

assert edit_distance("a b c".split(), "a x c d".split()) == 2
assert edit_distance("", "abc") == 3

# One wrong word in three: வாள் heard as வால்.
r = error_rates([("வாழ் வாள் வால்", "வாழ் வால் வால்")])
assert abs(r["wer"] - 1 / 3) < 1e-9, r
assert abs(r["cer"] - 1 / 14) < 1e-9, r  # 14 code points, one substitution (ள் -> ல்)

# Missing hypotheses count as empty output, are reported, and empty references are skipped.
rows = [
    {"id": "1", "text": "மழை பெய்கிறது", "slice": "A1.x"},
    {"id": "2", "text": "மலை உயரம்", "slice": "A1.x"},
    {"id": "3", "text": "!!", "slice": "A1.x"},
]
res = score(rows, {"1": "மழை பெய்கிறது"})["A1.x"]
assert res == {"wer": 0.5, "cer": 9 / 22, "n": 2, "missing": 1}, res  # 13 + 9 code points

print("all checks passed")
