"""Tamil text basics: cleaning, the Uraikal scoring normaliser, letters (எழுத்து) and script share.

    >>> letters("தமிழ்")
    ['த', 'மி', 'ழ்']
    >>> [length(x) for x in letters("கடல்")]
    ['kuril', 'kuril', 'otru']
"""
import re
import unicodedata

TAMIL_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")
VOWELS = "அஆஇஈஉஊஎஏஐஒஓஔ"  # உயிர்
CONSONANTS = "கஙசஞடணதநபமயரலவழளறனஜஶஷஸஹ"  # மெய்: the 18 Tamil ones, then Grantha
VOWEL_SIGNS = "ாிீுூெேைொோௌ"
PULLI = "்"
AYTHAM = "ஃ"
MARKS = VOWEL_SIGNS + PULLI + "ௗ"  # ௗ: the au length mark, if NFC left it alone
SHORT = set("அஇஉஎஒ" + "ிுெொ")  # குறில்: short vowels and their signs (a bare consonant carries short அ)

_DETACHED_SIGN = re.compile(f"(?<![{CONSONANTS}])([ெேை])([{CONSONANTS}])")


def clean(text: str) -> str:
    """General cleaning for Tamil text: what any pipeline should do before tokenizing or deduplicating.

    - Unicode NFC, so ெ + ா becomes ொ.
    - Invisible format characters dropped (ZWJ, ZWNJ, BOM, soft hyphen).
    - A pre-base vowel sign typed before its consonant (visual order, common in text converted
      from legacy fonts) moved after it: "ெக" -> "கெ".
    - ஶ்ரீ written with ஶ (U+0BB6) becomes the standard ஸ்ரீ.
    """
    text = "".join(c for c in unicodedata.normalize("NFC", text) if unicodedata.category(c) != "Cf")
    text = _DETACHED_SIGN.sub(r"\2\1", text).replace("ஶ்ரீ", "ஸ்ரீ")
    return unicodedata.normalize("NFC", text)


def normalize(text: str) -> str:
    """The Uraikal v0.1 scoring normaliser, applied to references and hypotheses before WER and CER.

    NFC, format characters dropped, Tamil digits to 0-9, lower-case, punctuation and symbols to spaces.
    Frozen: changing it changes published scores, so improvements go in `clean` or a new Uraikal version.
    """
    text = unicodedata.normalize("NFC", text).translate(TAMIL_DIGITS).lower()
    out = []
    for c in text:
        cat = unicodedata.category(c)
        if cat == "Cf":  # ZWJ, ZWNJ, BOM, soft hyphen
            continue
        # Letters, digits and combining marks (Tamil vowel signs, pulli) are kept.
        out.append(" " if cat[0] in "PSZC" else c)
    return " ".join("".join(out).split())


def letters(text: str) -> list[str]:
    """Split text into Tamil letters: a consonant keeps its vowel sign or pulli (கா, க்).

    Grantha conjuncts count as Tamil grammar counts them (க்ஷ = க் + ஷ). Anything that isn't a
    Tamil letter, spaces included, is returned one code point at a time. Run `clean` first.
    """
    out = []
    for c in text:
        if c in MARKS and out and out[-1][-1] in CONSONANTS:
            out[-1] += c
        else:
            out.append(c)
    return out


def kind(letter: str) -> str | None:
    """'uyir' (vowel), 'mei' (consonant with pulli), 'uyirmei' (consonant with a vowel), 'aytham', or None."""
    if letter in VOWELS:
        return "uyir"
    if letter == AYTHAM:
        return "aytham"
    if letter and letter[0] in CONSONANTS:
        return "mei" if letter.endswith(PULLI) else "uyirmei"
    return None


def length(letter: str) -> str | None:
    """Length for prosody: 'kuril' (short), 'nedil' (long), 'otru' (mei and ஃ), or None if not Tamil."""
    k = kind(letter)
    if k in ("mei", "aytham"):
        return "otru"
    if k == "uyir":
        return "kuril" if letter in SHORT else "nedil"
    if k == "uyirmei":
        return "kuril" if len(letter) == 1 or letter[1] in SHORT else "nedil"
    return None


def tamil_share(text: str) -> float:
    """Share of letters (Unicode alphabetic characters) that are Tamil script; 0.0 for text with none."""
    alpha = [c for c in text if c.isalpha()]
    return sum("஀" <= c <= "௿" for c in alpha) / max(len(alpha), 1)
