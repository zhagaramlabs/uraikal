"""Track I, the classics: does a language model know the Thirukkural?

- I1 recall: write Kural N word for word. Exact if the Kural's letters appear unbroken in the reply; answer
  only if the reply's Tamil is the Kural and nothing else.
- I2 attribution: give a Kural's number from its text. Also scored at the chapter (அதிகாரம், 10 Kurals).
- I3 metre: is this couplet a valid குறள் வெண்பா? Half are real Kurals, half have line 1's words reversed;
  the answer key comes from uraikal.yaappu, not from us.
- I5 invention: asked for a Kural that doesn't exist (a number past 1,330, or one about something the
  Kural never mentions), the model should reply NONE instead of making one up. Refused if NONE is the answer
  (alone on a line, or opening or closing the reply) and no couplet comes with it; refused only if the reply is
  just NONE, as asked.

Grader 2 (2026-11-02). Grader 1 counted any reply containing NONE as a refusal, including replies that quote the
prompt's NONE and then answer, or reply NONE and then describe the Kural anyway.

Items come from data/kural.jsonl (prep.py kural) and data/i5_topics.tsv (Tamil topic, tab, English
gloss). Like every test set we build, they stay private until Uraikal v1; the scores are published.
"""
import json
import random
import re
from pathlib import Path

from uraikal.score import error_rates, read_jsonl, same_setup
from uraikal.text import TAMIL_DIGITS, clean, normalize, tamil_share
from uraikal.yaappu import check_kural

GRADER = 2
NONE = "NONE"
NONE_ANSWER = re.compile(r"(?:^|\n)[\W\d_]*NONE[\W_]*(?:\n|$)|\bNONE[\W_]*$", re.I)  # \W also skips markdown
NONE_ONLY = re.compile(r"[\W\d_]*NONE[\W_]*", re.I)  # an echoed item number is allowed
TAMIL_ONLY = re.compile(r"[^஀-௿]")
NUMBER = re.compile(r"\d+")

RECALL = ("திருக்குறளின் {n}-ஆவது குறளைச் சொல் மாறாமல் எழுதுங்கள். குறளை மட்டும் எழுதுங்கள். அந்த எண்ணில் குறள் "
          "இல்லை என்றால் NONE என்று மட்டும் எழுதுங்கள்.\nWrite Thirukkural number {n} exactly, word for word. "
          "Write only the Kural. If there is no Kural with that number, reply only NONE.")
TOPIC = ("{ta} பற்றிப் பேசும் திருக்குறளை எழுதுங்கள். குறளை மட்டும் எழுதுங்கள். எந்தக் குறளும் அதைப் பற்றிப் "
         "பேசவில்லை என்றால் NONE என்று மட்டும் எழுதுங்கள்.\nQuote the Thirukkural that speaks about {en}. Write only "
         "the Kural. If no Kural speaks about it, reply only NONE.")
NUMBER_OF = ("இந்தத் திருக்குறளின் எண் என்ன? எண்ணை மட்டும் எழுதுங்கள்.\nWhat is the number of this Thirukkural "
             "(1 to 1330)? Reply with the number only.\n\n{kural}")
METRE = ("இது சரியான குறள் வெண்பாவா? ஆம் அல்லது இல்லை என்று மட்டும் எழுதுங்கள்.\nIs this a metrically valid "
         "kural venba? Reply only YES or NO.\n\n{kural}")


def kural_text(k):
    return f"{k['line1']}\n{k['line2']}"


def items(kurals, topics, n=200, seed=0):
    """The fixed item set: n Kurals each for I1 and I2, n/2 couplets for I3 (every other one altered), and for I5
    20 numbers past 1,330 plus one item per topic."""
    rng = random.Random(seed)
    pick = rng.sample(kurals, n)
    out = [{"id": f"I1.{k['n']}", "slice": "I1.recall", "prompt": RECALL.format(n=k["n"]), "gold": kural_text(k)}
           for k in pick]
    out += [{"id": f"I2.{k['n']}", "slice": "I2.attribution", "prompt": NUMBER_OF.format(kural=kural_text(k)),
             "gold": k["n"]} for k in rng.sample(kurals, n)]
    for i, k in enumerate(rng.sample(kurals, n // 2)):
        text = kural_text(k)
        if i % 2:
            text = " ".join(reversed(k["line1"].split())) + "\n" + k["line2"]
        out.append({"id": f"I3.{k['n']}.{i % 2}", "slice": "I3.metre", "prompt": METRE.format(kural=text),
                    "gold": not check_kural(text)})
    out += [{"id": f"I5.n{m}", "slice": "I5.invention", "prompt": RECALL.format(n=m), "gold": NONE}
            for m in rng.sample(range(1331, 1400), 20)]
    out += [{"id": f"I5.t{i}", "slice": "I5.invention", "prompt": TOPIC.format(ta=ta, en=en), "gold": NONE}
            for i, (ta, en) in enumerate(topics)]
    return out


def tamil(text):
    return TAMIL_ONLY.sub("", clean(text))


def yes_no(reply):
    r = reply.strip(" *\"'\n").upper()
    yes, no = r.startswith(("YES", "ஆம்")), r.startswith(("NO", "இல்லை"))
    return yes if yes != no else None


def couplet(reply):
    """Two lines in a row that could be a Kural: 2 to 6 words each, 90%+ Tamil, no digits, not a heading."""
    lines = [x for x in (line.strip(" *#>-\t") for line in reply.split("\n")) if x]
    verse = lambda x: (tamil_share(x) >= 0.9 and 2 <= len(x.split()) <= 6 and not x.endswith(":")
                       and not any(c.isdigit() for c in x))
    return any(verse(a) and verse(b) for a, b in zip(lines, lines[1:]))


def refusal(reply):
    """'only' if the reply is just NONE, 'answer' if NONE is its answer with more text but no couplet, else None."""
    if NONE_ONLY.fullmatch(reply):
        return "only"
    return "answer" if NONE_ANSWER.search(reply) and not couplet(reply) else None


def grade(item, reply):
    """One item's scores, from the model's reply."""
    refused = refusal(reply)
    if item["slice"] == "I1.recall":
        cer = error_rates([(normalize(item["gold"]), normalize(reply))])["cer"]
        return {"exact": tamil(item["gold"]) in tamil(reply), "answer_only": tamil(item["gold"]) == tamil(reply),
                "cer": cer, "false_refusal": refused is not None}
    if item["slice"] == "I2.attribution":
        m = NUMBER.search(reply.translate(TAMIL_DIGITS))
        n = int(m[0]) if m else None
        return {"correct": n == item["gold"], "same_chapter": n is not None and (n - 1) // 10 == (item["gold"] - 1) // 10}
    if item["slice"] == "I3.metre":
        return {"correct": yes_no(reply) == item["gold"]}
    return {"refused": refused is not None, "refused_only": refused == "only"}


def summarise(items_, replies):
    """Mean of every score per slice, plus how many replies hit the output limit."""
    by_slice = {}
    for item in items_:
        r = replies[item["id"]]
        s = by_slice.setdefault(item["slice"], {"n": 0, "truncated": 0, "sums": {}})
        s["n"] += 1
        s["truncated"] += r["truncated"]
        for k, v in grade(item, r["reply"]).items():
            s["sums"][k] = s["sums"].get(k, 0) + v
    return {name: {"n": s["n"], "truncated": s["truncated"] / s["n"],
                   **{k: v / s["n"] for k, v in s["sums"].items()}} for name, s in by_slice.items()}


def run(items_, ask, out, setup=None):
    """Ask every item once, saving replies to `out` (JSONL) as they come, so a rerun resumes. With `setup` (model,
    max tokens, ...), a rerun under another setup stops instead of mixing replies."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if setup is not None:
        same_setup(out, setup)
    done = {r["id"]: r for r in read_jsonl(out)} if out.exists() else {}
    with open(out, "a", encoding="utf-8") as f:
        for i, item in enumerate(x for x in items_ if x["id"] not in done):
            reply, truncated = ask(item["prompt"])
            done[item["id"]] = {"id": item["id"], "reply": reply, "truncated": truncated}
            f.write(json.dumps(done[item["id"]], ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1} asked", flush=True)
    return done


def ollama(endpoint, model, max_tokens, think=None):
    """An `ask` function for a model served by Ollama. think=False turns off a reasoning model's thinking."""
    import urllib.request

    def ask(prompt):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens}}
        if think is not None:
            body["think"] = think
        req = urllib.request.Request(f"{endpoint}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        return r["message"]["content"].strip(), r.get("done_reason") == "length"

    return ask


def load(kural_path, topics_path):
    kurals = list(read_jsonl(kural_path))
    topics = [line.rstrip("\n").split("\t") for line in open(topics_path, encoding="utf-8") if line.strip()]
    assert len(kurals) == 1330 and all(len(t) == 2 for t in topics), "need data/kural.jsonl and a ta<TAB>en topics file"
    return kurals, topics
