"""Self-checks for uraikal.numbers. Run: uv run python tests/test_numbers.py"""
from uraikal.numbers import from_numerals, speak_digits, to_digits, to_words

assert speak_digits("அது 35.75, 108-க்கு கால்") == "அது முப்பத்தி அஞ்சு புள்ளி ஏழு அஞ்சு, நூத்தி எட்டு-க்கு கால்"
# grouping commas, Indian and Western, are part of the number; a bare list stays a list
assert speak_digits("₹1,50,000") == "₹ஒரு லட்சத்தி அம்பதாயிரம்", speak_digits("₹1,50,000")
assert speak_digits("12,345 பேர்") == "பன்னெண்டாயிரத்தி முன்னூத்தி நாப்பத்தி அஞ்சு பேர்"
assert speak_digits("1,500,000") == "பதினஞ்சு லட்சம்" and speak_digits("1,2,3") == "ஒண்ணு,ரெண்டு,மூணு"
assert to_words(45) == "நாப்பத்தி அஞ்சு", to_words(45)
assert to_words(2300) == "ரெண்டாயிரத்தி முன்னூறு", to_words(2300)
assert to_words(1000) == "ஆயிரம்" and to_words(1500) == "ஆயிரத்தி ஐநூறு", to_words(1500)
assert to_words(140) == "நூத்தி நாப்பது" and to_words(108) == "நூத்தி எட்டு", to_words(140)
assert to_words(1.5) == "ஒண்ணு புள்ளி அஞ்சு" and to_words(0) == "பூஜ்யம்"
assert to_words(250000) == "ரெண்டு லட்சத்தி அம்பதாயிரம்", to_words(250000)
for n in list(range(3000)) + list(range(3000, 120001, 97)) + [21000, 10**5, 345678, 10**7, 23456789]:
    assert to_digits(to_words(n)) == str(n), (n, to_words(n), to_digits(to_words(n)))
assert to_digits("முன்னூறுல பதினஞ்சு பெர்சென்ட் எவ்ளோ?") == "300ல 15 பெர்சென்ட் எவ்ளோ?"
assert to_digits("ரெண்டாயிரத்து முன்னூறு ரூபாய்க்கு பத்து பெர்சென்ட்") == "2300 ரூபாய்க்கு 10 பெர்சென்ட்"
assert to_digits("கிலோ நாப்பது ரூபான்னா மூணு கிலோ") == "கிலோ 40 ரூபான்னா 3 கிலோ"
assert to_digits("ஆயிரம் ரூபாய நாலு பேரு பிரிச்சா ஒருத்தருக்கு எவ்ளோ?") == "1000 ரூபாய 4 பேரு பிரிச்சா ஒருத்தருக்கு எவ்ளோ?"
fused = "இருபத்தஞ்சு பேரு, நாப்பத்தொண்ணு ரூபா, தொண்ணூத்தொம்பது கிலோ"
assert to_digits(fused) == "25 பேரு, 41 ரூபா, 99 கிலோ", to_digits(fused)
assert to_digits("நான் நாளைக்கு பத்தாம் கிளாஸ்") == "நான் நாளைக்கு பத்தாம் கிளாஸ்"
# Formal Tamil, fused forms included (2026-11-09: these were left as words).
formal = {"இரண்டாயிரத்து முந்நூறு": 2300, "ஓராயிரம்": 1000, "மூன்றாயிரம்": 3000, "ஐயாயிரம்": 5000, "ஒன்பதாயிரம்": 9000,
          "இருபத்தைந்தாயிரம்": 25000, "ஐந்நூறு": 500, "நாநூறு": 400, "இருநூற்றைம்பது": 250, "அறுநூற்று ஒன்று": 601,
          "தொள்ளாயிரத்து": 900, "இருபத்திரண்டு": 22, "தொண்ணூற்றொன்பது": 99, "நூற்றெட்டு": 108, "ஆயிரத்தொரு": 1001,
          "ஆயிரத்திருநூறு": 1200, "ஆயிரத்தறுநூறு": 1600, "இருபத்தோராயிரம்": 21000, "ஒரு இலட்சத்து ஐம்பதாயிரம்": 150000,
          "லட்சத்தைம்பதாயிரம்": 150000, "இருநூத்தம்பது": 250}
for words, n in formal.items():
    assert to_digits(words) == str(n), (words, to_digits(words))
for word in ["முத்தாறு", "சித்திரை", "பத்திரம்", "பெற்றெடுத்த", "கற்றல்"]:  # look fused, aren't numbers
    assert to_digits(word) == word, word

# Tamil numerals: place value, and the older multiplying system (nuance §5.6 in the pass-2 review).
assert from_numerals("௨௦௨௬") == 2026
assert from_numerals("௨௱௫") == 205
assert from_numerals("௩௲௪௱௨௰௫") == 3425
assert from_numerals("௲") == 1000 and from_numerals("௰௲") == 10_000 and from_numerals("௱௲") == 100_000
assert from_numerals("௰௨") == 12 and from_numerals("௨௰") == 20
for bad in ("௨௩௱", "௨x"):
    try:
        from_numerals(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)

print("all checks passed")
