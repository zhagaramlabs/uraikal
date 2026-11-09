"""Self-checks for uraikal.rewards. Run: uv run python tests/test_rewards.py"""
import json
import re
import sys
from itertools import islice
from pathlib import Path

from uraikal.numbers import from_numerals
from uraikal.rewards import INVENTED, TOPICS, generate, molt_row, old_numeral, score

assert "verifiers" not in sys.modules and "molt" not in sys.modules  # adapters import lazily

# Each reward: 1 for a known-good reply, 0 for a known-bad one.
compose = next(generate("metre"))
assert score(compose, "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு")[0] == 1.0
assert score(compose, "**அகர முதல எழுத்தெல்லாம் ஆதி**\n**பகவன் முதற்றே உலகு**")[0] == 1.0  # markdown is stripped
assert score(compose, "நான் இன்று காலை கடைக்குப் போனேன்\nஅங்கே நிறைய மக்கள் இருந்தார்கள்")[0] == 0.0  # prose
assert score(compose, "Here it is:\nஅகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு")[0] == 0.0  # only the two lines

say = {"task": "numbers", "kind": "say", "n": 2300}
assert score(say, "விலை ரெண்டாயிரத்தி முன்னூறு ரூபாய்.")[0] == 1.0
assert score(say, "விலை ரெண்டாயிரத்து முன்னூறு ரூபாய்.")[0] == 1.0  # a spoken variant
assert score(say, "விலை இரண்டாயிரத்து முந்நூறு ரூபாய்.")[0] == 1.0  # formal Tamil is read too
assert score(say, "விலை 2300 ரூபாய்.")[0] == 0.0 and score(say, "விலை ரெண்டாயிரம் ரூபாய்.")[0] == 0.0
read = {"task": "numbers", "kind": "numerals", "n": 205}
assert score(read, "205")[0] == 1.0 and score(read, "௨௦௫")[0] == 1.0 and score(read, "2005")[0] == 0.0
assert score({**read, "n": 150000}, "1,50,000")[0] == 1.0

quote = {"task": "refusal", "kind": "number"}
assert score(quote, "NONE")[0] == 1.0 and score(quote, "**NONE**")[0] == 1.0
assert score(quote, "அப்படி ஒரு குறள் இல்லை.\nNONE")[0] == 0.5
assert score(quote, "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு")[0] == 0.0
assert score(quote, "NONE\nஅகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு")[0] == 0.0  # NONE, then a Kural anyway
exists = {"task": "refusal", "kind": "exists", "n": 5, "gold": True}
assert score(exists, "YES")[0] == 1.0 and score(exists, "ஆம்")[0] == 1.0 and score(exists, "NO")[0] == 0.0
assert score({**exists, "n": 2001, "gold": False}, "இல்லை")[0] == 1.0

tamil = {"task": "script", "kind": "tamil", "min_share": 0.9}
assert score(tamil, "மழை பெய்தால் பயிர்கள் வளரும். உழவர்கள் மகிழ்வார்கள்.")[0] == 1.0
assert score(tamil, "மழை பெய்தால் ாபயிர்கள் வளரும்.")[0] == 0.0  # a vowel sign on nothing
assert score(tamil, "மழை பெய்தால் பயிர்கள் வளரும்்.")[0] == 0.0  # a second pulli
assert score(tamil, "When it rains, crops grow.")[0] == 0.0 and score(tamil, "")[0] == 0.0

# Every generated item's gold answer earns 1: the generators and graders agree.
for item in islice(generate("numbers", seed=3), 3000):
    assert score(item, item["gold"]) == (1.0, "right number"), item
for item in islice(generate("refusal", seed=3), 600):
    assert score(item, "YES" if item["gold"] is True else "NO" if item["gold"] is False else item["gold"])[0] == 1.0, item

# The older numerals read back.
for n in list(range(1, 3000)) + list(range(3000, 10**6, 37)) + [10**6 - 1]:
    assert from_numerals(old_numeral(n)) == n, (n, old_numeral(n))
assert old_numeral(205) == "௨௱௫" and old_numeral(1000) == "௲" and old_numeral(10**5) == "௱௲"

# Deterministic, unlimited, all four tasks.
first = list(islice(generate(seed=7), 400))
assert first == list(islice(generate(seed=7), 400)) and first != list(islice(generate(seed=8), 400))
assert {x["task"] for x in first} == {"metre", "numbers", "refusal", "script"}
assert {x["kind"] for x in first} >= {"say", "read", "numerals", "number", "topic", "exists"}
assert all(json.loads(molt_row(x)["label"]) == x for x in first)

# Never Uraikal's test items: no Kural number from I5's range, no I5 topic.
for item in islice(generate("refusal", seed=0), 5000):
    assert not any(1331 <= int(n) <= 1399 for n in re.findall(r"\d+", item["prompt"])), item
i5 = Path(__file__).parent.parent / "data/i5_topics.tsv"
if i5.exists():
    rows = [line.rstrip("\n").split("\t") for line in open(i5, encoding="utf-8") if line.strip()]
    words = lambda s: {w.lower().rstrip("s") for w in s.split() if len(w) > 2 and w.lower() != "the"}  # noqa: E731
    for ta, en in INVENTED + TOPICS:
        assert all(ta != t and not words(en) & words(e) for t, e in rows), (ta, en)
else:
    print("data/i5_topics.tsv missing: skipped the I5 overlap check")

print("all checks passed")
