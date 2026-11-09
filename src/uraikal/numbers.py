"""Tamil numbers: number words <-> digits (spoken forms first, formal ones accepted when reading), and Tamil numerals.

to_digits:     "முன்னூறுல பதினஞ்சு பெர்சென்ட்" -> "300ல 15 பெர்சென்ட்"   (LLMs can't do sums from Tamil words)
to_words:      2300 -> "ரெண்டாயிரத்தி முன்னூறு"                          (TTS models read digits in English)
from_numerals: "௨௱௫" -> 205                                           (old texts: ௰ ௱ ௲ multiply, not place value)

Started in oosai/agent/numbers_ta.py; this is now the maintained copy.
"""
import re

UNITS = ["பூஜ்யம்", "ஒண்ணு", "ரெண்டு", "மூணு", "நாலு", "அஞ்சு", "ஆறு", "ஏழு", "எட்டு", "ஒம்பது"]
TEENS = ["பத்து", "பதினொண்ணு", "பன்னெண்டு", "பதிமூணு", "பதினாலு", "பதினஞ்சு", "பதினாறு", "பதினேழு", "பதினெட்டு", "பத்தொம்பது"]
TENS = {2: "இருபது", 3: "முப்பது", 4: "நாப்பது", 5: "அம்பது", 6: "அறுபது", 7: "எழுபது", 8: "எம்பது", 9: "தொண்ணூறு"}
HUNDREDS = {1: "நூறு", 2: "இருநூறு", 3: "முன்னூறு", 4: "நானூறு", 5: "ஐநூறு", 6: "அறுநூறு", 7: "எழுநூறு", 8: "எண்ணூறு", 9: "தொள்ளாயிரம்"}
BIG = [(10**7, "கோடி", "கோடியே"), (10**5, "லட்சம்", "லட்சத்தி"), (1000, "ஆயிரம்", "ஆயிரத்தி")]


def comb(word):
    """The form before another number: நூறு -> நூத்தி, இருபது -> இருபத்தி, தொள்ளாயிரம் -> தொள்ளாயிரத்தி."""
    return word[:-2] + "த்தி" if word.endswith(("று", "து", "ம்")) else word


def below_1000(n):
    h, r = divmod(n, 100)
    parts = [comb(HUNDREDS[h]) if r else HUNDREDS[h]] if h else []
    if r:
        t, u = divmod(r, 10)
        parts.append(UNITS[r] if r < 10 else TEENS[u] if r < 20 else f"{comb(TENS[t])} {UNITS[u]}" if u else TENS[t])
    return " ".join(parts)


def to_words(x):
    """A number in spoken Tamil words: whole numbers below 100 crore, decimals digit by digit."""
    if x < 0:
        return "மைனஸ் " + to_words(-x)
    whole, _, frac = f"{x:.2f}".rstrip("0").rstrip(".").partition(".")
    n, parts = int(whole), []
    if n == 0:
        parts.append(UNITS[0])
    for size, word, joined in BIG:
        k, n = divmod(n, size)
        if k:
            lead = below_1000(k)
            if k == 1:  # ஆயிரம், ஒரு லட்சம்
                parts.append(("" if word == "ஆயிரம்" else "ஒரு ") + (joined if n else word))
            elif word == "ஆயிரம்" and lead.endswith("ு"):  # ரெண்டு + ஆயிரம் -> ரெண்டாயிரம்
                parts.append(lead[:-1] + "ா" + (joined if n else word)[1:])
            else:
                parts.append(f"{lead} " + (joined if n else word))
    if n:
        parts.append(below_1000(n))
    words = " ".join(parts)
    return words + (" புள்ளி " + " ".join(UNITS[int(d)] for d in frac) if frac else "")


# Reading: morphemes that can start at the front of a word, as (regex, value, multiplies?).
_M = []
for i, w in enumerate(UNITS[1:], 1):
    _M.append((w, i, False))
for i, w in enumerate(TEENS, 10):
    _M.append((w, i, False))
for t, w in TENS.items():
    _M += [(w, t * 10, False), (w[:-2] + "த்(?:தி|து)", t * 10, False)]
for h, w in HUNDREDS.items():
    _M += [(w, h * 100, False), (comb(w), h * 100, False)]
_M += [  # formal and other spoken variants
    ("பூஜ்யம்|ஜீரோ", 0, False), ("ஒன்று|ஒன்னு|ஒரு", 1, False), ("இரண்டு", 2, False), ("மூன்று", 3, False), ("நான்கு", 4, False),
    ("ஐந்து", 5, False), ("ஒன்பது", 9, False), ("பதினொன்று|பதினொன்னு", 11, False), ("பன்னிரண்டு|பனிரெண்டு", 12, False),
    ("பதின்மூன்று", 13, False), ("பதினான்கு", 14, False), ("பதினைந்து|பதினைஞ்சு", 15, False), ("பத்தொன்பது", 19, False),
    ("நாற்பது", 40, False), ("ஐம்பது", 50, False), ("எண்பது", 80, False),
    ("நாற்பத்(?:தி|து)", 40, False), ("ஐம்பத்(?:தி|து)", 50, False), ("எண்பத்(?:தி|து)", 80, False),
    ("நூற்று|நூத்து", 100, False), ("முந்நூறு", 300, False), ("முந்நூற்று|முன்னூற்று", 300, False),
    ("தொண்ணூற்று", 90, False), ("அஞ்ஞூறு", 500, False),
    # after a tens stem the next unit fuses: இருபத் + அஞ்சு -> இருபத்தஞ்சு
    ("(?<=த்)(?:தொண்ணு|தொன்று)", 1, False), ("(?<=த்)(?:தஞ்சு|தைந்து|தைஞ்சு)", 5, False),
    ("(?<=த்)தாறு", 6, False), ("(?<=த்)தேழு", 7, False), ("(?<=த்)தெட்டு", 8, False),
    ("(?<=த்)(?:தொம்பது|தொன்பது)", 9, False),
    ("ஆயிர(?:ம்|த்தி|த்து)", 1000, True), ("(?<=ா)யிர(?:ம்|த்தி|த்து)", 1000, True),
    ("லட்ச(?:ம்|த்தி|த்து)", 10**5, True), ("கோடி(?:யே)?", 10**7, True),
]
for w in UNITS[1:] + TEENS + list(TENS.values()) + ["மூவு"]:  # ரெண்டு -> ரெண்டா before யிரம்
    _M.append((w[:-1] + "ா(?=யிர)", (UNITS + TEENS).index(w) if w in UNITS + TEENS else
               3 if w == "மூவு" else 10 * next(t for t, v in TENS.items() if v == w), False))
# a fused tens stem with nothing after it in the word: இருபத் (+ தஞ்சு)
for t, w in TENS.items():
    _M.append((w[:-2] + "த்(?=த[ொாேெஞை])", t * 10, False))
for w, v in [("நாற்ப", 40), ("ஐம்ப", 50), ("எண்ப", 80)]:
    _M.append((w + "த்(?=த[ொாேெஞை])", v, False))
_M = [(re.compile(p), v, m) for p, v, m in _M]
SUFFIX = re.compile(r"(?:ல|ல்|லே|க்கு|ஆல|னா|ன்னா|ம்)?[.,!?;:\"'()]*$")


def _read_word(word):
    """Morphemes covering the front of a word, if the rest is only a case ending or punctuation."""
    pos, out = 0, []
    while True:
        best = max(((m.match(word, pos), v, mult) for m, v, mult in _M), key=lambda c: c[0].end() if c[0] else -1)
        if not best[0] or best[0].end() == pos:
            break
        out.append((best[1], best[2]))
        pos = best[0].end()
    return (out, word[pos:]) if out and SUFFIX.fullmatch(word[pos:]) else (None, word)


def _value(morphemes):
    total = cur = 0
    for v, mult in morphemes:
        if mult:
            total, cur = total + (cur or 1) * v, 0
        else:
            cur += v
    return total + cur


def to_digits(text):
    """Replace runs of Tamil number words with digits, keeping the last word's case ending."""
    out, run = [], []
    for word in text.split(" ") + [""]:
        morphemes, rest = _read_word(word) if word else (None, "")
        if morphemes:
            run += morphemes
            if rest:  # a case ending closes the number
                out.append(f"{_value(run)}{rest}")
                run = []
            continue
        if run:
            out.append(str(_value(run)))
            run = []
        out.append(word)
    return " ".join(out[:-1])


def speak_digits(text):
    """Digits in a reply as Tamil words, for the TTS (numbers of a billion or more are left alone).
    Grouping commas, Indian (1,50,000) or Western (150,000), belong to the number."""
    def words(m):
        s = m.group().replace(",", "")
        x = float(s) if "." in s else int(s)
        return to_words(x) if x < 10**9 else m.group()
    return re.sub(r"\d{1,3}(?:,\d{2,3})*,\d{3}(?:\.\d+)?|\d+(?:\.\d+)?", words, text)


_DIGIT = {c: i for i, c in enumerate("௦௧௨௩௪௫௬௭௮௯")}
_UNIT = {"௰": 10, "௱": 100}


def from_numerals(s: str) -> int:
    """Read a number in Tamil numerals: place value (௨௦௨௬ = 2026) or the older system,
    where ௰ (10), ௱ (100) and ௲ (1000) multiply the digit before them (௨௱௫ = 205, ௰௲ = 10,000)."""
    if any(c not in _DIGIT and c not in "௰௱௲" for c in s) or not s:
        raise ValueError(f"not a Tamil numeral: {s!r}")
    if not any(c in "௰௱௲" for c in s):
        return int("".join(str(_DIGIT[c]) for c in s))
    total = section = 0
    digit = None
    for c in s:
        if c in _DIGIT:
            if digit is not None:
                raise ValueError(f"two digits in a row in {s!r}")
            digit = _DIGIT[c]
        elif c in _UNIT:
            section, digit = section + (1 if digit is None else digit) * _UNIT[c], None
        else:  # ௲
            total += (section + (digit or 0) or 1) * 1000
            section, digit = 0, None
    return total + section + (digit or 0)
