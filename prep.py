"""Build Uraikal data: A1 manifests (JSONL: id, audio, text, slice) from public Tamil test sets, D4 text
passages, and the Kural text for the Yaappu checks.

    python prep.py fleurs                       # downloads ~400 MB from Hugging Face
    python prep.py kathbath                     # streams ~4 GB, keeps only Tamil (~1-2 GB)
    python prep.py common_voice --dir PATH      # a Common Voice Tamil download from Mozilla Data Collective
    python prep.py d4 --parquet FILE... --slice D4.web   # needs the `prep` extra (pyarrow)
    python prep.py kural                        # the 1,330 Kurals from Tamil Wikisource, for uraikal.yaappu

Licences and caveats: SPEC.md, "Data sources".
"""
import argparse
import csv
import json
import tarfile
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
KATHBATH = "https://objectstore.e2enetworks.net/indic-superb/kathbath/clean/"
# Kathbath's "test" split is the unknown-speaker one.
KATHBATH_SPLITS = {"test_known": ("testkn_audio.tar", "A1.kathbath_known"), "test": ("testunk_audio.tar", "A1.kathbath_unknown")}


def write_manifest(name, rows):
    out = DATA / f"a1_{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(rows)} utterances)")


def index_audio(root, pattern):
    return {p.name: p.resolve() for p in Path(root).rglob(pattern)}


def require(audio, name):
    if name not in audio:
        raise SystemExit(f"audio file {name} listed in the transcripts was not found; is the download complete?")
    return str(audio[name])


def fleurs(_args):
    from huggingface_hub import hf_hub_download

    tsv = hf_hub_download("google/fleurs", "data/ta_in/test.tsv", repo_type="dataset")
    root = DATA / "fleurs"
    if not any(root.rglob("*.wav")):
        tgz = hf_hub_download("google/fleurs", "data/ta_in/audio/test.tar.gz", repo_type="dataset")
        with tarfile.open(tgz) as t:
            t.extractall(root, filter="data")
    audio = index_audio(root, "*.wav")
    rows = []
    with open(tsv, encoding="utf-8") as f:
        # Columns: sentence id, file, raw transcription, normalised transcription, characters, samples, gender.
        for cols in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows.append({"id": f"fleurs-{Path(cols[1]).stem}", "audio": require(audio, cols[1]),
                         "text": cols[2], "slice": "A1.fleurs", "gender": cols[6].lower()})
    write_manifest("fleurs", rows)


def stream_tamil(url, dest):
    """Extract only Tamil members from a remote all-language tar without storing the whole archive."""
    print(f"streaming {url}")
    with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|") as t:
        for member in t:
            if "/tamil/" in member.name:
                t.extract(member, dest, filter="data")


def kathbath(args):
    root = DATA / "kathbath"
    if not (root / "kb_data_clean_m4a/tamil/test/transcription_n2w.txt").exists():
        stream_tamil(KATHBATH + "transcripts_n2w.tar", root)
    for split in args.splits:
        tar_name, slice_name = KATHBATH_SPLITS[split]
        audio_root = root / "audio" / split
        if not any(audio_root.rglob("*.m4a")):
            stream_tamil(KATHBATH + tar_name, audio_root)
        audio = index_audio(audio_root, "*.m4a")
        rows = []
        with open(root / f"kb_data_clean_m4a/tamil/{split}/transcription_n2w.txt", encoding="utf-8") as f:
            for line in f:
                name, text = line.rstrip("\n").split("\t", 1)
                rows.append({"id": f"kathbath-{Path(name).stem}", "audio": require(audio, name),
                             "text": text, "slice": slice_name})
        write_manifest(slice_name.split(".")[1], rows)


def common_voice(args):
    root = Path(args.dir)
    rows = []
    with open(root / "test.tsv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            path = root / "clips" / r["path"]
            if not path.exists():
                raise SystemExit(f"{path} not found; point --dir at the folder holding test.tsv and clips/")
            rows.append({"id": f"cv{args.version}-{Path(r['path']).stem}", "audio": str(path.resolve()),
                         "text": r["sentence"], "slice": "A1.common_voice", "speaker": r["client_id"]})
    write_manifest("common_voice", rows)


def d4(args):
    """Sample Tamil passages (lines of 20+ words, 90%+ Tamil script) from Parquet files, a few per document."""
    import random

    import pyarrow.parquet as pq

    from uraikal.text import clean, tamil_share

    docs = [(Path(path).name, text) for path in args.parquet
            for text in pq.read_table(path, columns=[args.text_field]).column(0).to_pylist() if text]
    rng = random.Random(0)
    rng.shuffle(docs)
    rows = []
    for source, text in docs:
        lines = [line for line in text.split("\n") if len(line.split()) >= 20]
        rng.shuffle(lines)
        kept = [p for p in (clean(line).strip() for line in lines) if tamil_share(p) >= 0.9][: args.per_doc]
        rows += [{"id": f"{args.slice}.{len(rows) + i:05d}", "text": p, "slice": args.slice, "source": source}
                 for i, p in enumerate(kept)]
        if len(rows) >= args.n:
            break
    out = DATA / f"d4_{args.slice.split('.', 1)[1]}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in rows[: args.n]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({min(len(rows), args.n)} passages from {len({r['source'] for r in rows})} files)")


WIKISOURCE = "https://ta.wikisource.org/w/api.php"
KURAL_BOOK = "திருக்குறள், மூலம்"  # Wikisource transcription of a 1997 printing, mostly split by சீர்
# Transcription slips in that copy, each checked against Wikisource's "திருக்குறள் பரிமேலழகர் உரை"
# (2026-11-01). Word-spacing differences are left alone: the Yaappu checker re-divides the சீர் itself.
KURAL_ERRATA = {
    114: (None, "எச்சத்தாற் காணப் படும்"),  # காணப்ப
    434: (None, "அற்றந் தரூஉம் பகை"),  # த்ரூஉம்
    738: (None, "அணியென்ப நாட்டிற்கிவ் வைந்து"),  # நாட்டிவ்
    764: ("அழிவின்று அறைபோகா தாகி வழிவந்த", None),  # அழிவின்றி
    784: (None, "மேற்சென்று இடித்தற் பொருட்டு"),  # மேற்செனறு
    858: (None, "மிகலூக்கின் ஊக்குமாம் கேடு"),  # மிக்லூக்கின்
    882: ("வாள்போல் பகைவரை அஞ்சற்க அஞ்சுக", None),  # வாள்போல
    1134: ("காமக் கடும்புனல் உய்க்குமே நாணொடு", None),  # உய்க்கும்
    1249: ("உள்ளத்தார் காத லவராக உள்ளிநீ", None),  # லவரால்
}
KURAL_WORDS = {"கெளவை": "கௌவை"}  # ஔ typed as ெ + ள, as old fonts did (Kurals 1143–1150)


def wikisource(**params):
    import time
    import urllib.parse

    time.sleep(0.3)  # be polite to Wikimedia
    req = urllib.request.Request(f"{WIKISOURCE}?{urllib.parse.urlencode({**params, 'format': 'json', 'formatversion': 2})}",
                                 headers={"User-Agent": "uraikal/0.1 (https://github.com/zhagaramlabs/uraikal)"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def kural(_args):
    """All 1,330 Kurals from Tamil Wikisource, one per line with its source page and revision."""
    import html
    import re
    import unicodedata

    titles = [p["title"] for p in wikisource(action="query", list="allpages", apprefix=KURAL_BOOK + "/", aplimit=500)["query"]["allpages"]]
    rows = {}
    for title in titles:
        page = wikisource(action="parse", page=title, prop="text|revid")["parse"]
        text = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<br ?/?>", "\n", page["text"])))
        lines = []
        for line in text.splitlines():
            line = unicodedata.normalize("NFC", line).replace("⁠", "").strip()
            if re.fullmatch(r"\d{1,4}", line) and lines:  # some pages put the number on its own line
                lines[-1] += f" {line}"
            elif line:
                lines.append(line)
        pairs = [(prev, re.fullmatch(r"(.+?)[.?]?\s*(\d{1,4})", line)) for prev, line in zip(lines, lines[1:])]
        pairs = [(prev, m) for prev, m in pairs if m and not prev[-1].isdigit() and not re.match(r"﻿?\d", prev)]
        printed = [int(m.group(2)) for _, m in pairs]
        first = (sorted(printed)[len(printed) // 2] - 1) // 10 * 10  # the chapter, from its printed numbers
        for i, (prev, m) in enumerate(pairs, 1):
            n = first + i  # number by position: the printed numbers have typos (two 71s)
            if n != int(m.group(2)):
                print(f"Kural {n} is printed as {m.group(2)} on {title}")
            rows[n] = {"n": n, "line1": " ".join(prev.split()), "line2": " ".join(m.group(1).split()),
                       "chapter": title.split("/", 1)[1], "source": f"https://ta.wikisource.org/wiki/{title}",
                       "revid": page["revid"]}
        if len(pairs) != 10:
            print(f"{title}: {len(pairs)} Kurals, expected 10")
    for n, lines in KURAL_ERRATA.items():
        for key, fixed in zip(("line1", "line2"), lines):
            if fixed:
                rows[n][key] = fixed
    for row in rows.values():
        for key in ("line1", "line2"):
            for wrong, right in KURAL_WORDS.items():
                row[key] = row[key].replace(wrong, right)
    missing = sorted(set(range(1, 1331)) - set(rows))
    out = DATA / "kural.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for n in sorted(rows):
            f.write(json.dumps(rows[n], ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(rows)} Kurals from {len(titles)} chapters; missing: {missing[:20] or 'none'})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)
    sub.add_parser("fleurs").set_defaults(fn=fleurs)
    sub.add_parser("kural", help="the 1,330 Kurals from Tamil Wikisource, for the Yaappu checks").set_defaults(fn=kural)
    t = sub.add_parser("d4", help="tokenizer test passages from Parquet text")
    t.add_argument("--parquet", nargs="+", required=True)
    t.add_argument("--slice", required=True, help="e.g. D4.web")
    t.add_argument("--text-field", default="text")
    t.add_argument("--n", type=int, default=2000)
    t.add_argument("--per-doc", type=int, default=3)
    t.set_defaults(fn=d4)
    k = sub.add_parser("kathbath")
    k.add_argument("--splits", nargs="+", default=list(KATHBATH_SPLITS), choices=list(KATHBATH_SPLITS))
    k.set_defaults(fn=kathbath)
    c = sub.add_parser("common_voice")
    c.add_argument("--dir", required=True, help="folder with test.tsv and clips/")
    c.add_argument("--version", default="27.0", help="Common Voice release; splits change every release")
    c.set_defaults(fn=common_voice)
    args = p.parse_args()
    DATA.mkdir(exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
