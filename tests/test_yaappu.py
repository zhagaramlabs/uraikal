"""Self-checks for uraikal.yaappu. Run: uv run python tests/test_yaappu.py

With data/kural.jsonl (python prep.py kural) it also scans all 1,330 Kurals, and with data/d4_web.jsonl it
checks that ordinary prose rarely passes as a குறள்."""
import json
import random
from pathlib import Path

from uraikal.text import tamil_share
from uraikal.yaappu import alakidu, asai, check_kural, seer_name

kinds = lambda seer: [k for _, k in asai(seer)]  # noqa: E731

# அசை
assert kinds("கல்") == ["நேர்"] and kinds("கா") == ["நேர்"] and kinds("கால்") == ["நேர்"]
assert kinds("கடல்") == ["நிரை"] and kinds("பலா") == ["நிரை"] and kinds("படாம்") == ["நிரை"]
assert kinds("கற்க") == ["நேர்", "நேர்"]  # a ஒற்று closes the அசை
assert kinds("உடைமை") == ["நிரை", "நேர்"]  # ஐ after the first letter is short
assert kinds("தொழாஅர்") == ["நிரை", "நேர்"]  # அளபெடை: the written அ is a short letter
assert [t for t, _ in asai("அகர")] == ["அக", "ர"]

# வாய்பாடு
assert seer_name(asai("அகர")) == "புளிமா" and seer_name(asai("வேண்டுதல்")) == "கூவிளம்"
assert seer_name(asai("எழுத்தெல்லாம்")) == "புளிமாங்காய்"  # எழுத் | தெல் | லாம்
assert seer_name(asai("உலகு"), last=True) == "பிறப்பு" and seer_name(asai("மாட்டு"), last=True) == "காசு"
assert seer_name(asai("உலகு")) == "புளிமா"  # only the last சீர் of a வெண்பா ends in பிறப்பு

# Whole Kurals, printed as Wikisource's 1997 text prints them.
kural = {
    1: "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு",
    2: "கற்றதனால் ஆய பயனென்கொல் வாலறிவன்\nநற்றாள் தொழாஅர் எனின்",
    4: "வேண்டுதல் வேண்டாமை இலானடி சேர்ந்தார்க்கு\nயாண்டும் இடும்பை இல",
    74: "அன்புஈனும் ஆர்வம் உடைமை அதுஈனும்\nநண்பென்னும் நாடாச் சிறப்பு",
    226: "அற்றார் அழிபசி தீர்த்தல் அஃதொருவன்\nபெற்றான் பொருள்வைப் புழி",
    844: "வெண்மை எனப்படுவ தியாதெனின் ஒண்மை\nஉடையம்யாம் என்னும் செருக்கு",
    1330: "ஊடுதல் காமத்திற்கு இன்பம், அதற்கின்பம்\nகூடி முயங்கப் பெறின்",
}
for n, text in kural.items():
    assert check_kural(text) == [], (n, check_kural(text))

# Kural 4: seven சீர் that cross words, as the traditional seer-split text prints it.
assert [t for t, _, _ in alakidu(kural[4])] == ["வேண்டுதல்வேண்", "டாமை", "யிலானடி", "சேர்ந்தார்க்", "கியாண்டு", "மிடும்பை", "யில"]
assert [n for _, _, n in alakidu(kural[1])] == ["புளிமா", "புளிமா", "புளிமாங்காய்", "தேமா", "புளிமா", "புளிமா", "பிறப்பு"]
# A குற்றியலுகரம் before a vowel loses its உ (அன்புஈனும் → அன்பீனும்); a final ஒற்று passes to the next vowel.
assert [t for t, _, _ in alakidu(kural[74])][:2] == ["அன்பீனு", "மார்வ"]

# Not வெண்பா: a புளிமா at the end, the last word moved to the front, prose, too few words.
# (Swapping அகர and முதல would still scan: both are புளிமா. Line breaks are ignored.)
assert check_kural("அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகம்") == ["வுலகம்: புளிமா is not allowed at the end"]
assert check_kural("உலகு அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே")
assert check_kural("நான் நேற்று காலையில் சந்தைக்குப் போய் காய்கறி வாங்கினேன்")
assert check_kural("அகர முதல எழுத்தெல்லாம்") == ["no division into 4 + 3 சீர் keeps the rules (3 words as printed)"]

data = Path(__file__).resolve().parents[1] / "data"
if (data / "kural.jsonl").exists():
    rows = [json.loads(line) for line in open(data / "kural.jsonl", encoding="utf-8")]
    failed = [r["n"] for r in rows if not alakidu(r["line1"] + "\n" + r["line2"])]
    assert len(rows) == 1330 and not failed, failed
    print("all 1,330 Kurals scan as குறள் வெண்பா")
if (data / "d4_web.jsonl").exists():
    rng, windows = random.Random(0), []
    for line in open(data / "d4_web.jsonl", encoding="utf-8"):
        words = [w for w in json.loads(line)["text"].split() if tamil_share(w) == 1]
        if len(words) >= 7:
            i = rng.randrange(len(words) - 6)
            windows.append(" ".join(words[i : i + 7]))
    passed = sum(alakidu(w) is not None for w in windows)
    assert passed / len(windows) < 0.01, passed
    print(f"prose passing as a குறள்: {passed} of {len(windows)} seven-word windows")
print("all checks passed")
