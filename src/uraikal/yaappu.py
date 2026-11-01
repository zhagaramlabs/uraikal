"""Tamil prosody (யாப்பு): அசை and சீர், and a checker for the குறள் வெண்பா.

    >>> [text for text, _ in asai("அகர")]
    ['அக', 'ர']
    >>> seer_name(asai("அகர"))
    'புளிமா'
    >>> [text for text, _, _ in alakidu("வேண்டுதல் வேண்டாமை இலானடி சேர்ந்தார்க்கு யாண்டும் இடும்பை இல")]
    ['வேண்டுதல்வேண்', 'டாமை', 'யிலானடி', 'சேர்ந்தார்க்', 'கியாண்டு', 'மிடும்பை', 'யில']
    >>> check_kural("அகர முதல எழுத்தெல்லாம் ஆதி\\nபகவன் முதற்றே உலகு")
    []

As in traditional scansion (அலகிடுதல்), a couplet is read as one stream after புணர்ச்சி, and its
seven சீர் may begin inside a word (Kural 4 above). `alakidu` searches for a division that keeps the
rules, preferring one whose சீர் begin at words, and allows at most one that doesn't. Line breaks are
ignored. The rules, checked on all 1,330 Kurals in tests/test_yaappu.py:
- A letter is குறில் (short), நெடில் (long) or ஒற்று (a consonant with pulli). ஒற்று add no syllable.
- அசை: நேர் is one short or long letter; நிரை is a short letter followed by a short or long one.
  Either takes the ஒற்று that follow it. ஃ is like a ஒற்று, but may count as a short letter where the
  verse needs it (four Kurals do, such as தீர்த்த லஃதொருவன், 226).
- ஐ and ஔ count as short except at the start of a word (ஐகாரக் குறுக்கம்).
- புணர்ச்சி before a vowel: a ஒற்று passes to the vowel (தொழாஅர் எனின் → தொழாஅ ரெனின்), doubling after
  one short letter (கொல்எனும் → கொல்லெனும்); a குற்றியலுகரம் loses its உ (அன்புஈனும் → அன்பீனும்); after
  a vowel, ய் or வ் comes between. A vowel written after a long letter is an அளபெடை (தொழாஅர்), one
  short letter.
- A குற்றியலுகரம் before ய becomes a குற்றியலிகரம் (சேர்ந்தார்க்கு யாண்டும் → சேர்ந்தார்க் கியாண்டு), which
  may count as a short letter or not at all, as the verse needs.
- வெண்டளை: மா முன் நிரை, விளம் முன் நேர், காய் முன் நேர். A குறள் வெண்பா has 4 + 3 சீர் of two அசை, or of
  three ending in நேர் (காய்), and ends in நாள், மலர், காசு or பிறப்பு (the last two end in consonant + உ).
"""
from functools import lru_cache
from itertools import product

from uraikal.text import VOWELS, kind, length, letters

NER, NIRAI = "நேர்", "நிரை"
VALLINAM = "கசடதபற"  # the hard consonants; உ after one at the end of a word is shortened
SIGN = dict(zip(VOWELS, ["", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]))
NAMES = {
    (NER,): "நாள்", (NIRAI,): "மலர்",
    (NER, NER): "தேமா", (NIRAI, NER): "புளிமா", (NER, NIRAI): "கூவிளம்", (NIRAI, NIRAI): "கருவிளம்",
    (NER, NER, NER): "தேமாங்காய்", (NIRAI, NER, NER): "புளிமாங்காய்",
    (NER, NIRAI, NER): "கூவிளங்காய்", (NIRAI, NIRAI, NER): "கருவிளங்காய்",
    (NER, NER, NIRAI): "தேமாங்கனி", (NIRAI, NER, NIRAI): "புளிமாங்கனி",
    (NER, NIRAI, NIRAI): "கூவிளங்கனி", (NIRAI, NIRAI, NIRAI): "கருவிளங்கனி",
}
KURAL_ENDINGS = {"நாள்", "மலர்", "காசு", "பிறப்பு"}
# At most one சீர் may begin inside a word. All 1,330 Kurals need 0 or 1; allowing 2 would let 2.4%
# of random 7-word prose windows pass as a குறள், against 0.4% with 1 (tests/test_yaappu.py).
MAX_SPLITS = 1
OPTIONAL = "opt"  # a குற்றியலிகரம்: counted as குறில் or not at all


def _kutriyalukaram(word: list[str]) -> bool:
    """Does the word end in a shortened உ: a hard consonant + உ, unless the word is one short letter + it (அது, பசு)?"""
    last = word[-1] if word else ""
    if len(last) != 2 or last[0] not in VALLINAM or last[1] != "ு":
        return False
    return not (len(word) == 2 and length(word[0]) == "kuril")


def _units(token: str) -> list[list]:
    """One space-separated word: its letters as [letter, length, starts_word], after புணர்ச்சி inside it."""
    units, word = [], []  # word: letters since the last word boundary inside the token
    for ch in letters(token):
        if kind(ch) is None:
            continue  # punctuation
        if kind(ch) == "uyir" and units:
            prev = units[-1][0]
            if kind(prev) == "mei" and len(word) == 2 and length(word[0]) == "kuril":
                ch, word = prev[0] + SIGN[ch], []  # the consonant doubles: கொல்எனும் → கொல்லெனும்
            elif kind(prev) == "mei" or _kutriyalukaram(word):
                units.pop()
                ch, word = prev[0] + SIGN[ch], []  # the consonant takes the vowel
            elif length(prev) != "nedil":
                word = []  # a vowel after a short letter starts a new word (அதுஈனும்)
        L = "aytham" if ch == "ஃ" else length(ch)
        if word and (ch in "ஐஔ" or ch[-1] in "ைௌ"):
            L = "kuril"  # ஐகாரக் குறுக்கம்
        if word and word[-1] == "மி" and ch.startswith("யா"):
            units[-1][1] = OPTIONAL  # the particle மியா
        units.append([ch, L, not units])
        word.append(ch)
    if len(units) > 1 and units[0][0][0] in VALLINAM and units[0][0][1:] == "ி" and units[1][0].startswith("ய"):
        units[0][1] = OPTIONAL  # printed as in a seer-split text: தியாதெனின்
    return units


def _join(tokens: list[list[list]]) -> list[list]:
    """The words of a couplet as one stream, with புணர்ச்சி between them."""
    stream, word = [], []  # word: the letters of the previous token
    for units in tokens:
        if not units:
            continue
        first = units[0][0]
        if stream and kind(first) == "uyir":
            prev = stream[-1][0]
            if kind(prev) == "mei" and len(word) == 2 and length(word[0]) == "kuril":
                units[0][0] = prev[0] + SIGN[first]  # தனிக்குறில் முன் ஒற்று உயிர்வரின் இரட்டும்
            elif kind(prev) == "mei" or _kutriyalukaram(word):
                units[0][0] = stream.pop()[0][0] + SIGN[first]
            elif kind(prev) != "aytham":
                units[0][0] = ("ய" if prev[-1] in "இஈஐிீை" else "வ") + SIGN[first]  # உடம்படுமெய்
        elif stream and first.startswith("ய") and _kutriyalukaram(word):
            stream[-1] = [stream[-1][0][0] + "ி", OPTIONAL, True]  # குற்றியலிகரம், which belongs to both words
        word = [u[0] for u in units]
        stream += units
    return stream


def _asai(units: list[list]) -> list[tuple[str, str]]:
    out, i, lead = [], 0, ""
    while i < len(units):
        ch, L = units[i][0], units[i][1]
        if L in ("otru", "aytham", "silent"):
            if out:
                out[-1] = (out[-1][0] + ch, out[-1][1])
            else:
                lead += ch  # a silent குற்றியலிகரம் opening the சீர்: கியாதெனின்
            i += 1
            continue
        if L == "kuril" and i + 1 < len(units) and units[i + 1][1] in ("kuril", "nedil"):
            out.append((lead + ch + units[i + 1][0], NIRAI))
            i += 2
        else:
            out.append((lead + ch, NER))
            i += 1
        lead = ""
    return out


def _resolve(units: list[list]):
    """Every reading of the optional letters: a குற்றியலிகரம் or ஃ counted as குறில் or not at all."""
    spots = [i for i, u in enumerate(units) if u[1] in (OPTIONAL, "aytham")]
    options = [("kuril", "silent") if units[i][1] == OPTIONAL else ("aytham", "kuril") for i in spots]
    for choice in product(*options):
        out = [list(u) for u in units]
        for i, L in zip(spots, choice):
            out[i][1] = L
        yield out


def asai(seer: str) -> list[tuple[str, str]]:
    """Split one சீர் into அசை: [(text, "நேர்" or "நிரை"), ...]. A குற்றியலிகரம் here counts as குறில்."""
    return _asai(next(iter(_resolve(_units(seer)))))


def seer_name(asais: list[tuple[str, str]], last: bool = False) -> str | None:
    """The வாய்பாடு of a சீர் (தேமா, புளிமாங்காய், …). As the last சீர், one ending in consonant + உ is காசு or பிறப்பு."""
    pattern = tuple(k for _, k in asais)
    end = asais[1][0] if last and len(asais) == 2 else ""
    if len(end) == 2 and end[1] == "ு":  # a lone consonant + உ as the second அசை
        return "காசு" if pattern[0] == NER else "பிறப்பு"
    return NAMES.get(pattern)


def _vendalai(prev: tuple[str, ...], first: str) -> bool:
    """மா முன் நிரை, விளம் முன் நேர், காய் முன் நேர்."""
    if len(prev) == 2:
        return first == (NIRAI if prev[-1] == NER else NER)
    return len(prev) == 3 and prev[-1] == NER and first == NER


def _seer_ok(a: list[tuple[str, str]], last: bool) -> bool:
    if last:
        return seer_name(a, last=True) in KURAL_ENDINGS
    return len(a) == 2 or (len(a) == 3 and a[-1][1] == NER)


def _divide(units: list[list]):
    """The best division of a couplet into 7 சீர்: (boundaries inside words, [(start, end), ...]) or None."""
    n = len(units)

    @lru_cache(maxsize=None)
    def best(i, k, prev):
        if k == 7:
            return (0, ()) if i == n else None
        if i >= n or units[i][1] in ("otru", "aytham") or units[i][0] == "ஃ":
            return None  # a சீர் can't start with a bare consonant
        found = None
        for j in range(i + 1, n + 1):
            a = _asai(units[i:j])
            if len(a) > 3:
                break
            if not a or (prev and not _vendalai(prev, a[0][1])) or not _seer_ok(a, last=k == 6):
                continue
            rest = best(j, k + 1, tuple(x for _, x in a))
            if rest is not None:
                cost = rest[0] + (j < n and not units[j][2])
                if found is None or cost < found[0]:
                    found = (cost, ((i, j),) + rest[1])
        return found

    return best(0, 0, ())


def alakidu(text: str) -> list[tuple[str, list[tuple[str, str]], str]] | None:
    """Scan a குறள் வெண்பா: its seven சீர் as (text, அசை, வாய்பாடு), or None if no division keeps the rules.

    The words may be printed split by word or by சீர்; line breaks are ignored."""
    stream = _join([_units(token) for token in text.split()])
    results = [(r, u) for u in _resolve(stream) if (r := _divide(u)) and r[0] <= MAX_SPLITS]
    if not results:
        return None
    (_, spans), units = min(results, key=lambda x: x[0][0])
    seers = [units[i:j] for i, j in spans]
    return [("".join(u[0] for u in s), _asai(s), seer_name(_asai(s), last=k == 6)) for k, s in enumerate(seers)]


def check_kural(text: str) -> list[str]:
    """Problems that stop `text` being a குறள் வெண்பா; empty if it is one.

    When no division into சீர் works, the problems are those of the printed one (one சீர் per word)."""
    if alakidu(text):
        return []
    tokens = [_units(t) for t in text.split()]
    if len(tokens) != 7:
        return [f"no division into 4 + 3 சீர் keeps the rules ({len(tokens)} words as printed)"]
    stream = _join(tokens)
    starts = [i for i, u in enumerate(stream) if u[2]] + [len(stream)]
    seers = [stream[i:j] for i, j in zip(starts, starts[1:])]
    scans = [_asai(next(iter(_resolve(s)))) for s in seers]
    words = ["".join(u[0] for u in s) for s in seers]
    problems = [f"{w}: {seer_name(a, last=k == 6)} is not allowed {'at the end' if k == 6 else 'in a வெண்பா'}"
                for k, (w, a) in enumerate(zip(words, scans)) if not _seer_ok(a, last=k == 6)]
    problems += [f"{w} → {v}: not வெண்டளை ({seer_name(a)} முன் {b[0][1]})"
                 for w, v, a, b in zip(words, words[1:], scans, scans[1:]) if not _vendalai(tuple(x for _, x in a), b[0][1])]
    return problems or ["no division into 4 + 3 சீர் keeps the rules"]
