"""Tamil reward environments for RL: Uraikal's graders as reward functions, with endless generated prompts.

Four tasks. Each reward takes a prompt item (a dict from `generate`) and the model's reply, and returns
(reward, reason) with the reward in [0, 1]:

- metre:   compose a குறள் வெண்பா; 1 if the reply is two lines that scan (uraikal.yaappu.check_kural).
- numbers: a number in a sentence -> spoken Tamil words (to_words via speak_digits, read back with to_digits),
           Tamil number words -> digits, Tamil numerals (old multiplying system or place value) -> digits.
- refusal: quote a Kural that doesn't exist (a number past 1,330, or a topic the Kural never mentions) -> NONE,
           graded as Track I5 grades it (classics.refusal: 1 for NONE alone, 0.5 for NONE plus other text);
           and "is there a Kural number N?" -> YES/NO (classics.yes_no), half real numbers, half not.
- script:  write Tamil; 0 if any vowel sign or pulli is left unattached, else 1 if the Tamil share of letters
           reaches the item's min_share. A format reward to add to others: alone, one Tamil word earns 1.

    >>> item = next(generate("numbers", seed=1))
    >>> score(item, item["gold"])[0]
    1.0

Training items are generated here, never taken from Uraikal's test sets: no Kural text or number from
data/kural.jsonl, no I5 topic (data/i5_topics.tsv; tests/test_rewards.py checks the lists are disjoint), and no
number from 1,331 to 1,399, the range I5 asks for. Not checked: whether a composed venba is on its topic, or
whether it is a real Kural quoted from memory (either passes the metre reward).

Adapters: `verifiers_taskset` (Prime Intellect's verifiers v1 API, as used by prime-rl) and `molt_runner`
(NVIDIA Molt's Env.step -> Result). Both import their framework only when called.
"""
import itertools
import json
import random
import re
import unicodedata

from uraikal.classics import NONE, couplet, refusal as _refusal, yes_no
from uraikal.numbers import speak_digits, to_digits, to_words
from uraikal.text import MARKS, TAMIL_DIGITS, letters, tamil_share
from uraikal.yaappu import check_kural

TOPICS = [("மழை", "rain"), ("நட்பு", "friendship"), ("கல்வி", "learning"), ("தாய்", "a mother"), ("உழவு", "farming"),
          ("கடல்", "the sea"), ("பொறுமை", "patience"), ("உண்மை", "truth"), ("இசை", "music"), ("மரம்", "trees"),
          ("நிலா", "the moon"), ("அன்பு", "love"), ("விருந்தோம்பல்", "hospitality"), ("காலம்", "time"),
          ("உழைப்பு", "hard work"), ("மொழி", "language"), ("குழந்தை", "a child"), ("பசி", "hunger"),
          ("வீரம்", "courage"), ("நீர்", "water")]
# Things the Kural never mentions. Disjoint from Uraikal's I5 topics (tests/test_rewards.py).
INVENTED = [("மிதிவண்டி", "bicycles"), ("தேநீர்", "tea"), ("பேருந்து", "buses"), ("குளிர்சாதனப் பெட்டி", "refrigerators"),
            ("அன்னாசிப் பழம்", "pineapples"), ("நிலக்கடலை", "peanuts"), ("தபால் தலை", "postage stamps"),
            ("கடிகாரம்", "clocks"), ("துப்பாக்கி", "guns"), ("பனிச்சறுக்கு", "skiing"), ("எரிமலை", "volcanoes"),
            ("பென்குயின்", "penguins"), ("கங்காரு", "kangaroos"), ("சாக்லேட்", "chocolate"), ("பனிக்கரடி", "polar bears"),
            ("ஒட்டகச்சிவிங்கி", "giraffes"), ("கோல்ஃப்", "golf"), ("பேஸ்பால்", "baseball"), ("பனிப்பாறை", "glaciers"),
            ("மூக்குக்கண்ணாடி", "spectacles"), ("பலூன்", "balloons"), ("மரவள்ளிக்கிழங்கு", "cassava"),
            ("பப்பாளி", "papayas"), ("கொய்யா", "guavas")]
# No number words in these (ஒரு, ஆறு, ...), and nothing glued to {n} (₹{n}): to_digits must read back only n.
SENTENCES = ["இந்தப் புத்தகத்தின் விலை {n} ரூபாய்.","நேற்று {n} பேர் கூட்டத்துக்கு வந்தார்கள்.",
             "எங்கள் ஊரிலிருந்து சென்னை {n} கிலோமீட்டர்.", "இந்த மாதம் {n} லிட்டர் பால் வாங்கினோம்.",
             "கடையில் {n} புத்தகங்கள் உள்ளன.", "அவர் {n} ரூபாய் கடன் வாங்கினார்."]

METRE = ("{ta} பற்றி ஒரு குறள் வெண்பா இயற்றுங்கள்: இரண்டு அடிகள், முதல் அடியில் நான்கு சீர், இரண்டாம் அடியில் மூன்று சீர். "
         "வெண்பாவை மட்டும் எழுதுங்கள்.\nCompose a kural venba about {en}: two lines, four feet then three. "
         "Write only the two lines.")
SAY = ("இந்த வாக்கியத்தில் உள்ள எண்ணைப் பேச்சுத் தமிழ்ச் சொற்களில் எழுதுங்கள், இலக்கங்கள் வேண்டாம். வாக்கியத்தை மட்டும் "
       "எழுதுங்கள்.\nRewrite this sentence with the number in spoken Tamil words, no digits. Reply with the sentence "
       "only.\n\n{s}")
READ = ("இந்த எண்ணை இலக்கங்களில் (0-9) எழுதுங்கள். எண்ணை மட்டும் எழுதுங்கள்.\nWrite this number in digits (0-9). "
        "Reply with the number only.\n\n{x}")
QUOTE_N = ("திருக்குறளின் {n}-ஆம் குறளை எழுதுக. அந்த எண்ணில் குறள் இல்லையெனில் NONE என்று மட்டும் எழுதுக.\nQuote "
           "Kural {n} of the Thirukkural. If there is no such Kural, write only NONE.")
QUOTE_TOPIC = ("{ta} குறித்த திருக்குறளை எழுதுக. அப்படி ஒரு குறள் இல்லையெனில் NONE என்று மட்டும் எழுதுக.\nQuote the "
               "Kural about {en}. If no Kural is about it, write only NONE.")
EXISTS = ("திருக்குறளில் {n}-ஆம் குறள் உண்டா? ஆம் அல்லது இல்லை என்று மட்டும் எழுதுக.\nDoes the Thirukkural have a "
          "Kural number {n}? Reply only YES or NO.")
WRITE = "{ta} பற்றித் தமிழில் மூன்று வாக்கியங்கள் எழுதுங்கள்.\nWrite three sentences in Tamil about {en}."

TO_TAMIL = str.maketrans("0123456789", "௦௧௨௩௪௫௬௭௮௯")


def old_numeral(n: int) -> str:
    """1 <= n < 10**6 in the older Tamil numerals, where ௰ ௱ ௲ multiply the digit before them (205 -> ௨௱௫)."""
    def small(k):
        h, t, u = k // 100, k // 10 % 10, k % 10
        return "".join(("" if d == 1 else str(d).translate(TO_TAMIL)) + unit for d, unit in ((h, "௱"), (t, "௰")) if d) + (
            str(u).translate(TO_TAMIL) if u else "")
    k, r = divmod(n, 1000)
    return (("" if k == 1 else small(k)) + "௲" if k else "") + small(r)


# Rewards: (item, reply) -> (reward, reason).

def metre(item, reply):
    lines = [x for x in (line.strip(" *#>-\t") for line in reply.strip().split("\n")) if x]
    if len(lines) != 2 or not couplet(reply):
        return 0.0, f"not a two-line Tamil couplet ({len(lines)} lines)"
    problems = check_kural("\n".join(lines))
    return (0.0, problems[0]) if problems else (1.0, "a valid kural venba")


def numbers(item, reply):
    if item["kind"] == "say":
        if any(c.isdigit() for c in reply):
            return 0.0, "digits left in the reply"
        got = re.findall(r"\d+", to_digits(" ".join(reply.split())))
    else:
        got = re.findall(r"\d+", reply.translate(TAMIL_DIGITS).replace(",", ""))
    return (1.0, "right number") if got == [str(item["n"])] else (0.0, f"read {got}, want {item['n']}")


def refusal(item, reply):
    if item["kind"] == "exists":
        said = yes_no(reply)
        return (1.0, "right") if said == item["gold"] else (0.0, f"said {said}, Kural {item['n']} exists: {item['gold']}")
    return {"only": (1.0, "NONE"), "answer": (0.5, "NONE with other text")}.get(_refusal(reply), (0.0, "no refusal"))


def script(item, reply):
    text = unicodedata.normalize("NFC", reply)
    stray = [x for x in letters(text) if x[0] in MARKS]
    if stray:
        return 0.0, f"{len(stray)} stray vowel sign or pulli"
    share = tamil_share(text)
    return (1.0, f"Tamil share {share:.2f}") if share >= item["min_share"] else (0.0, f"Tamil share {share:.2f} too low")


REWARDS = {"metre": metre, "numbers": numbers, "refusal": refusal, "script": script}


def score(item, reply):
    """(reward, reason) for any generated item."""
    return REWARDS[item["task"]](item, reply)


# Generators: rng -> one prompt item.

def _fake_kural(rng):
    return rng.randint(1400, 10 ** rng.randint(4, 5))  # never 1,331-1,399: Uraikal's I5 asks for those


def _metre(rng):
    ta, en = rng.choice(TOPICS)
    return {"task": "metre", "kind": "compose", "prompt": METRE.format(ta=ta, en=en)}


def _numbers(rng):
    kind = rng.choice(["say", "read", "numerals"])
    n = rng.randint(1, 10 ** rng.randint(1, 6 if kind == "numerals" else 7) - 1)
    if kind == "say":
        s = rng.choice(SENTENCES).format(n=rng.choice([str(n), f"{n:,}"]))
        return {"task": "numbers", "kind": kind, "n": n, "prompt": SAY.format(s=s), "gold": speak_digits(s)}
    x = to_words(n) if kind == "read" else old_numeral(n) if rng.random() < 0.5 else str(n).translate(TO_TAMIL)
    return {"task": "numbers", "kind": kind, "n": n, "prompt": READ.format(x=x), "gold": str(n)}


def _refusal_item(rng):
    kind = rng.choice(["number", "topic", "exists"])
    if kind == "number":
        return {"task": "refusal", "kind": kind, "prompt": QUOTE_N.format(n=_fake_kural(rng)), "gold": NONE}
    if kind == "topic":
        ta, en = rng.choice(INVENTED)
        return {"task": "refusal", "kind": kind, "prompt": QUOTE_TOPIC.format(ta=ta, en=en), "gold": NONE}
    n = rng.randint(1, 1330) if rng.random() < 0.5 else _fake_kural(rng)
    return {"task": "refusal", "kind": kind, "n": n, "prompt": EXISTS.format(n=n), "gold": n <= 1330}


def _script(rng):
    ta, en = rng.choice(TOPICS)
    return {"task": "script", "kind": "tamil", "min_share": 0.9, "prompt": WRITE.format(ta=ta, en=en)}


MAKERS = {"metre": _metre, "numbers": _numbers, "refusal": _refusal_item, "script": _script}


def generate(task="all", seed=0):
    """Endless prompt items for one task, or all four in turn; the same seed gives the same items."""
    rng = random.Random(f"{task}:{seed}")
    makers = list(MAKERS.values()) if task == "all" else [MAKERS[task]]
    for i in itertools.count():
        yield makers[i % len(makers)](rng)


# Adapters. Neither framework is imported until an adapter is called.

def verifiers_taskset(task="all", seed=0):
    """A verifiers v1 Taskset class (verifiers 0.4, the API prime-rl selects through). verifiers loads tasksets
    by top-level module name, so put this in e.g. `tamil_rewards.py` on the path and pass
    `--env.taskset.id tamil_rewards --env.agent.harness.id null` (null: one chat turn, no tools):

        from uraikal.rewards import verifiers_taskset
        TamilRewards = verifiers_taskset("all")
        __all__ = ["TamilRewards"]
    """
    import verifiers.v1 as vf

    class Data(vf.TaskData):
        item: dict

    class TamilTask(vf.Task[Data, vf.State, vf.TaskConfig]):
        @vf.reward
        async def uraikal(self, trace) -> float:
            reward, trace.info["uraikal"] = score(self.data.item, trace.last_reply)
            return reward

    class TamilRewards(vf.Taskset[TamilTask, vf.TasksetConfig]):
        INFINITE = True

        def load(self):
            for item in generate(task, seed):
                yield TamilTask(Data(prompt=item["prompt"], item=item))

    return TamilRewards


def molt_row(item):
    """A dataset row for Molt: `--data.input_key prompt --data.label_key label --data.apply_chat_template`.
    The label is JSON text because items of different tasks have different keys."""
    return {"prompt": [{"role": "user", "content": item["prompt"]}], "label": json.dumps(item, ensure_ascii=False)}


def molt_runner():
    """Molt's AgentRunner class. Save as e.g. `tamil_agent.py`, pass `--train.agent_path tamil_agent.py`:

        from uraikal.rewards import molt_runner
        AgentRunner = molt_runner()
    """
    import torch
    from molt.agents import Env, Result, StepEnvRunner

    class TamilEnv(Env):
        async def step(self, state) -> Result:
            label = state.get("label")
            reward, _ = score(json.loads(label) if isinstance(label, str) else label, state["action_text"])
            return Result(reward=torch.tensor(reward, dtype=torch.float32))

    class AgentRunner(StepEnvRunner):
        def __init__(self):
            super().__init__(TamilEnv)

    return AgentRunner
