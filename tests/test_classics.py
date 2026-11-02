"""Self-checks for track I grading. Run: uv run python tests/test_classics.py
The item-set checks need data/kural.jsonl (python prep.py kural)."""
import json
from pathlib import Path

from uraikal.classics import grade, items, summarise, yes_no

gold = "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு"
recall = {"slice": "I1.recall", "gold": gold}
# Exact even with a label, other spacing and punctuation around the verse; a changed letter isn't.
assert grade(recall, "குறள் 1:\nஅகர முதல எழுத்தெல்லாம்\nஆதி பகவன் முதற்றே உலகு.")["exact"]
assert not grade(recall, "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகம்")["exact"]
assert grade(recall, gold)["cer"] == 0 and not grade(recall, gold)["false_refusal"]
assert grade(recall, "NONE")["false_refusal"]
# Numbers in Tamil numerals count; a near miss in the same chapter gets chapter credit only.
assert grade({"slice": "I2.attribution", "gold": 42}, "௪௨")["correct"]
assert grade({"slice": "I2.attribution", "gold": 42}, "Kural 45") == {"correct": False, "same_chapter": True}
assert grade({"slice": "I2.attribution", "gold": 42}, "தெரியவில்லை") == {"correct": False, "same_chapter": False}
assert yes_no("**YES**") is True and yes_no("இல்லை.") is False and yes_no("ஆம், இல்லை") is True and yes_no("maybe") is None
assert grade({"slice": "I5.invention", "gold": "NONE"}, "None.")["refused"]
i5 = {"slice": "I5.invention", "gold": "NONE"}
assert grade(i5, "**NONE**") == {"refused": True, "refused_only": True}
assert grade(i5, "1337.\n\nNONE.\n\n---\n**Explanation:** there is no such Kural.") == {"refused": True, "refused_only": False}
assert grade(i5, "மின்சாரம் பற்றிய திருக்குறள் இல்லை. NONE")["refused"]
assert not grade(i5, "Nonetheless, Kural 1337 says:\nஅகர முதல எழுத்தெல்லாம்\nஆதி பகவன் முதற்றே உலகு")["refused"]
assert not grade(i5, "NONE\nஅகர முதல எழுத்தெல்லாம்\nஆதி பகவன் முதற்றே உலகு")["refused"]  # NONE, then a couplet anyway
assert not grade(i5, 'The prompt says "reply only NONE." The 1375th Kural is a famous one, so it exists.')["refused"]
assert grade(i5, "1351\nNONE\nதிருக்குறள் எண் 1351-இல் ஒரு குறள் இல்லை.\nகூடுதல் தகவல் இதோ:")["refused"]  # prose, not a couplet
assert grade(recall, gold)["answer_only"] and not grade(recall, "குறள் 1:\n" + gold)["answer_only"]

data = Path(__file__).resolve().parents[1] / "data"
if (data / "kural.jsonl").exists():
    kurals = [json.loads(line) for line in open(data / "kural.jsonl", encoding="utf-8")]
    xs = items(kurals, [("கைபேசி", "mobile phones")], n=20)
    count = lambda s: sum(x["slice"] == s for x in xs)
    assert (count("I1.recall"), count("I2.attribution"), count("I3.metre"), count("I5.invention")) == (20, 20, 10, 21)
    assert xs == items(kurals, [("கைபேசி", "mobile phones")], n=20), "items must be the same every run"
    assert all(x["gold"] for x in xs if x["id"].endswith(".0") and x["slice"] == "I3.metre"), "real Kurals scan"
    # Perfect answers score perfectly.
    perfect = {x["id"]: {"reply": {True: "YES", False: "NO"}[x["gold"]] if x["slice"] == "I3.metre" else str(x["gold"]),
                         "truncated": False} for x in xs}
    s = summarise(xs, perfect)
    assert s["I1.recall"]["exact"] == s["I2.attribution"]["correct"] == s["I3.metre"]["correct"] == s["I5.invention"]["refused"] == 1
print("ok")
