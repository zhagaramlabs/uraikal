"""Self-checks for uraikal.text. Run: uv run python tests/test_text.py"""
from uraikal.text import clean, kind, length, letters, tamil_share

# Letters: a consonant keeps its vowel sign or pulli.
assert letters("தமிழ்") == ["த", "மி", "ழ்"]
assert letters("கௌவை") == ["கௌ", "வை"]
assert letters("க்ஷேத்திரம்") == ["க்", "ஷே", "த்", "தி", "ர", "ம்"]
assert letters("ஃபேன் ok") == ["ஃ", "பே", "ன்", " ", "o", "k"]
assert letters("அம்மா") == ["அ", "ம்", "மா"]
assert len(letters("படித்துக்கொண்டேயிருப்பவர்களுக்காகவும்")) == 21

assert [kind(x) for x in letters("அம்மா ஃ")] == ["uyir", "mei", "uyirmei", None, "aytham"]
assert [length(x) for x in letters("கடல்")] == ["kuril", "kuril", "otru"]
assert [length(x) for x in letters("பலா")] == ["kuril", "nedil"]
assert [length(x) for x in letters("ஐயா")] == ["nedil", "nedil"]
assert [length(x) for x in letters("அவள்")] == ["kuril", "kuril", "otru"]
assert [length(x) for x in letters("கொம்பு")] == ["kuril", "otru", "kuril"]
assert [length(x) for x in letters("போ")] == ["nedil"]
assert length("x") is None

# Cleaning: NFC composes the two-part vowel sign, ZWJ goes, visual-order signs are fixed.
assert clean("கொ") == "கொ"  # க + ெ + ா -> கொ
assert clean("க்‍ஷ") == "க்ஷ"
assert clean("ெகாடு") == "கொடு"  # legacy visual order: ெ + க + ா
assert clean("அவன் ேபானான்") == "அவன் போனான்"
assert clean("ஶ்ரீ") == "ஸ்ரீ"
assert clean("ஒளி கெளுத்தி") == "ஒளி கெளுத்தி"  # real ஒள / கெள sequences are left alone
assert clean("கெ") == "கெ"  # an attached sign doesn't move

assert tamil_share("தமிழ் text") == 3 / 7  # த, ம, ழ against t, e, x, t (vowel signs and pulli aren't alphabetic)
assert tamil_share("") == 0.0

print("all checks passed")
