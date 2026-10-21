"""Join rendered frames into the finished Short and check it.
python kit/assemble.py JOBDIR FRAMESDIR OUTDIR
JOBDIR holds story.py, voice.wav, words.srt. Writes OUTDIR/{video.mp4, sheet.jpg, qc.json, stems...}."""
import sys, os, json, importlib.util, subprocess, math
from multiprocessing import Pool
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import comp, sound, sound_mix

JOB, FR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}/frames", exist_ok=True)
spec = importlib.util.spec_from_file_location("story", f"{JOB}/story.py"); S = importlib.util.module_from_spec(spec)
os.environ["STORY_DIR"] = JOB; spec.loader.exec_module(S)
FPS = S.FPS; TOTAL = S.TOTAL
CAP = comp.Captions(S.W, S.CAPS)

def one(i):
    t = (i - 1) / FPS
    comp.frame(f"{FR}/{i:04d}.png", t, CAP, S.LABELS, getattr(S, "CTA_T", None), f"{OUT}/frames/{i:04d}.jpg")
    return i

def main():
    n = len([f for f in os.listdir(FR) if f.endswith(".png")])
    frames = sorted(int(f[:4]) for f in os.listdir(FR) if f.endswith(".png"))
    missing = [i for i in range(1, int(math.floor(TOTAL * FPS)) + 1) if i not in set(frames)]
    with Pool(os.cpu_count()) as p: list(p.imap_unordered(one, frames, chunksize=8))
    # ---- audio: voice first, music + effects ducked under it
    mus = sound.music(TOTAL + 1, getattr(S, "MUSIC_KEY", 41.2), "dark", getattr(S, "BPM", 84))
    ev = [(t, k, g, d) for t, k, g, d in S.SFX]
    qc = sound_mix.build(OUT, [(0.0, f"{JOB}/voice.wav")], mus, ev, TOTAL + 0.5, sound.sfx)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-start_number", str(frames[0]), "-i", f"{OUT}/frames/%04d.jpg",
                    "-i", f"{OUT}/mix.wav", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", f"{OUT}/video.mp4"], check=True)
    # ---- QC 1: speech recognition of the FINAL mix vs the script (pronunciation + masking check)
    try:
        from faster_whisper import WhisperModel
        import difflib, re
        m = WhisperModel("small.en", device="cpu", compute_type="int8")
        segs, _ = m.transcribe(f"{OUT}/mix.wav", language="en", beam_size=5, word_timestamps=True)
        heard = []; segs = list(segs)
        for s in segs: heard += [(w.start, w.end, w.word.strip()) for w in s.words]
        nrm = lambda x: re.sub(r"[^a-z0-9 ]", " ", x.lower()).split()
        a = nrm(" ".join(w[2] for w in S.W.w)); b = nrm(" ".join(h[2] for h in heard))
        sm = difflib.SequenceMatcher(None, a, b)
        diffs = [(" ".join(a[i1:i2]), " ".join(b[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
        qc.update({"asr_match": round(sm.ratio(), 3), "asr_diffs": diffs[:40], "asr_text": " ".join(h[2] for h in heard)})
    except Exception as e:
        qc["asr_error"] = str(e)
    # ---- QC 2: sync table (what is said at each cut) + contact sheet at the middle of every shot
    from PIL import Image, ImageDraw
    plan = json.load(open(f"{FR}/plan.json")) if os.path.exists(f"{FR}/plan.json") else {"shots": []}
    rows = []
    for a, b, note in plan["shots"]:
        said = [w[2] for w in S.W.w if a - 0.05 <= w[0] < b]
        rows.append({"t0": round(a, 2), "t1": round(b, 2), "dur": round(b - a, 2), "shot": note, "words": " ".join(said)})
    qc["shots"] = rows; qc["missing_frames"] = missing[:20]; qc["n_missing"] = len(missing); qc["frames"] = n
    ims = []
    for r in rows:
        i = min(frames[-1], max(1, int(((r["t0"] + r["t1"]) / 2) * FPS) + 1))
        p = f"{OUT}/frames/{i:04d}.jpg"
        if os.path.exists(p):
            im = Image.open(p).resize((216, 384)); d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 216, 18], fill=(0, 0, 0)); d.text((4, 3), f"{r['t0']:.1f}s {r['shot'][:28]}", fill=(255, 255, 255))
            ims.append(im)
    if ims:
        cols = 6; rws = (len(ims) + cols - 1) // cols
        sh = Image.new("RGB", (216 * cols, 384 * rws), (20, 20, 20))
        for k, im in enumerate(ims): sh.paste(im, ((k % cols) * 216, (k // cols) * 384))
        sh.save(f"{OUT}/sheet.jpg", quality=88)
    json.dump(qc, open(f"{OUT}/qc.json", "w"), indent=1)
    print("assembled", len(frames), "frames; missing", len(missing))

if __name__ == "__main__":
    main()
