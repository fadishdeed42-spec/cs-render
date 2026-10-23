"""asr.py AUDIO WORDS.json OUT.json : transcribe with faster-whisper (small.en) and diff against the script words."""
import sys, json, re, difflib
from faster_whisper import WhisperModel
m = WhisperModel("small.en", device="cpu", compute_type="int8")
segs, _ = m.transcribe(sys.argv[1], language="en", beam_size=5, word_timestamps=True)
got = []; text = []
for s in segs:
    text.append(s.text)
    for w in s.words: got.append([w.word.strip(), round(w.start, 2), round(w.end, 2), round(w.probability, 2)])
nz = lambda t: re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()
ref = nz(" ".join(w["w"] for w in json.load(open(sys.argv[2]))["words"]))
hyp = nz(" ".join(text))
sm = difflib.SequenceMatcher(None, ref, hyp)
diffs = [(" ".join(ref[a:b]), " ".join(hyp[c:d])) for op, a, b, c, d in sm.get_opcodes() if op != "equal"]
json.dump({"match": round(sm.ratio(), 3), "heard": " ".join(text).strip(), "diffs": diffs, "words": got}, open(sys.argv[3], "w"), indent=0)
