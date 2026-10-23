"""compose.py WORKDIR : AI clips + ElevenLabs voice -> captioned, voice-first mixed 1080x1920 30fps Short + QC.

WORKDIR: words.json (from eleven.py), voice.wav, music.wav, story.json, clips/<shot>.mp4 (or img/<shot>.* as a still fallback)
story.json: {"shots": [{"id", "at": "anchor phrase", "zoom": [z0, z1]?}], "replace": [[spoken, display]],
             "labels": [[anchor, anchor_end, "TEXT"]], "sfx": [[anchor_or_seconds, offset, kind, gain, dur]],
             "cta": [..two lines..], "quote_lines": ["q1", ...]}
Writes WORKDIR/result/: video.mp4, sheet.jpg, qc.json, edl.json"""
import os, sys, json, re, subprocess, math
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine3d"))
import assemble as A
FPS = 30; OW, OH = 1080, 1920
A.FPS = FPS
WD = sys.argv[1]; R = os.path.join(WD, "result"); os.makedirs(R, exist_ok=True)
ST = json.load(open(os.path.join(WD, "story.json"))); WJ = json.load(open(os.path.join(WD, "words.json")))
words = WJ["words"]; total = WJ["total"] + ST.get("tail", 1.2)
norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower())
NW = [norm(w["w"]) for w in words]

def find(phrase, start=0):
    seq = [norm(x) for x in phrase.split() if norm(x)]
    for i in range(start, len(NW) - len(seq) + 1):
        if NW[i:i + len(seq)] == seq: return i
    raise SystemExit(f"anchor not found: {phrase!r}")

# ---------------------------------------------------------------- edit decision list
edl, pos = [], 0
for sh in ST["shots"]:
    i = find(sh["at"], pos); pos = i + 1
    edl.append({"id": sh["id"], "t0": max(0.0, words[i]["s"] - sh.get("lead", 0.08)), "zoom": sh.get("zoom")})
edl[0]["t0"] = 0.0
for k, e in enumerate(edl): e["t1"] = edl[k + 1]["t0"] if k + 1 < len(edl) else total
json.dump(edl, open(os.path.join(R, "edl.json"), "w"), indent=1)

def anchor_t(a):
    if isinstance(a, (int, float)): return float(a)
    return words[find(a)]["s"]

# ---------------------------------------------------------------- prepare each shot as a 30 fps 1080x1920 clip of exact length
PREP = os.path.join(WD, "prep"); os.makedirs(PREP, exist_ok=True)
def prep(e):
    out = os.path.join(PREP, e["id"] + ".mp4"); dur = e["t1"] - e["t0"]
    clip = os.path.join(WD, "clips", e["id"] + ".mp4")
    z0, z1 = e["zoom"] or (1.0, 1.06)
    nfr = int(round(dur * FPS)) + 2
    zexpr = f"{z0}+({z1}-{z0})*on/{nfr}"
    if os.path.exists(clip):
        cl = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", clip], capture_output=True, text=True).stdout)
        slow = max(1.0, dur / max(cl - 0.05, 0.5))          # stretch (with motion interpolation) only if the shot is longer than the clip
        vf = (f"setpts={slow:.4f}*PTS,minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:vsbmc=1,"
              f"scale=1188:2112:flags=lanczos,crop=1080:1920,"
              f"zoompan=z='{zexpr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps={FPS},"
              f"unsharp=5:5:0.6,format=yuv420p")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-vf", vf, "-t", f"{dur:.3f}", "-an", "-c:v", "libx264", "-crf", "14", "-preset", "fast", out]
    else:
        img = next(os.path.join(WD, "img", f) for f in os.listdir(os.path.join(WD, "img")) if f.startswith(e["id"] + "."))
        vf = (f"scale=1188:2112:flags=lanczos,crop=1080:1920,"
              f"zoompan=z='{z0}+({max(z1, z0 + 0.1)}-{z0})*on/{nfr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={nfr}:s=1080x1920:fps={FPS},format=yuv420p")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", img, "-vf", vf, "-t", f"{dur:.3f}", "-c:v", "libx264", "-crf", "14", "-preset", "fast", out]
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(os.path.join(WD, "words.json")):
        subprocess.run(cmd, check=True)
    return out
from multiprocessing import Pool
with Pool(os.cpu_count()) as p: outs = p.map(prep, edl)
lst = os.path.join(PREP, "list.txt"); open(lst, "w").write("".join(f"file '{o}'\n" for o in outs))
base = os.path.join(WD, "base.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", base], check=True)

# ---------------------------------------------------------------- overlays: captions, hacker terminal quotes, labels, end card
quote_keys = set(ST.get("quote_lines", []))
cap_words = [[w["w"], w["s"], w["e"]] for w in words if w["line"] not in quote_keys and w["line"] != "cta"]
tokens = A.caption_tokens(cap_words, ST.get("replace", []))
chunks = A.chunks_of(tokens)
quotes = []
for key in ST.get("quote_lines", []):
    ws = [w for w in words if w["line"] == key]
    if ws: quotes.append((ws[0]["s"], ws[-1]["e"], " ".join(w["w"] for w in ws), ws))
labels = [(anchor_t(a), anchor_t(b), txt) for a, b, txt in ST.get("labels", [])]
cta_t = next(w["s"] for w in words if w["line"] == "cta") - 0.1
MONO = "DejaVuSansMono-Bold.ttf"
SS = A.SS; W, H = A.W, A.H

def terminal(d, t):
    for t0, t1, txt, ws in quotes:
        if not (t0 - 0.15 <= t <= t1 + 0.5): continue
        a = A.clamp((t - t0 + 0.15) / 0.2) * A.clamp((t1 + 0.5 - t) / 0.2)
        shown = "".join(w["w"] + " " for w in ws if w["s"] <= t).rstrip()
        cur = "_" if int(t * 3) % 2 == 0 else " "
        f = A.font(MONO, 50)
        lines, line = [], "> "
        for wd in (shown + cur).split(" "):
            if d.textlength(line + wd, font=f) > W * 0.78: lines.append(line); line = "  "
            line += wd + " "
        lines.append(line)
        y0 = 1180; bh = 70 * len(lines) + 70
        d.rounded_rectangle([90 * SS, y0 * SS, (OW - 90) * SS, (y0 + bh) * SS], radius=22 * SS, fill=(4, 14, 8, int(225 * a)), outline=(40, 255, 120, int(200 * a)), width=3 * SS)
        d.text((120 * SS, (y0 + 16) * SS), "MESSAGE FROM THE HACKER", font=A.font(MONO, 26), fill=(40, 255, 120, int(200 * a)))
        for k, ln in enumerate(lines):
            d.text((120 * SS, (y0 + 60 + 70 * k) * SS), ln, font=f, fill=(90, 255, 150, int(255 * a)))
        return True
    return False

def overlay(t):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for t0, t1, txt in labels:
        a = A.clamp((t - t0) / 0.25) * A.clamp((t1 - t) / 0.25)
        A.chip(d, txt, a)
    if t >= cta_t: A.end_card(d, A.eout((t - cta_t) / 0.4), ST.get("cta", ["FOLLOW CHAIN SECRETS.", "EVERY CHART HIDES A STORY."]))
    elif not terminal(d, t): A.draw_caption(d, chunks, t, y=1380)
    return im.resize((OW, OH), Image.LANCZOS)

vig = A.vignette() if hasattr(A, "vignette") else None
nfr = int(total * FPS)
dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", base, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
silent = os.path.join(WD, "silent.mp4")
enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
                        "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", silent], stdin=subprocess.PIPE)
grain = (np.random.default_rng(3).normal(0, 3.2, (OH // 2, OW // 2, 1))).astype(np.float32)
last = None
for i in range(nfr):
    t = i / FPS
    raw = dec.stdout.read(OW * OH * 3)
    if len(raw) == OW * OH * 3: last = raw
    fr = Image.frombytes("RGB", (OW, OH), last)
    if vig is not None: fr = Image.fromarray((np.asarray(fr, np.float32) * (np.asarray(vig, np.float32) / 255.0)).astype(np.uint8))
    g = np.roll(grain, (i * 37) % (OH // 2), axis=0).repeat(2, 0).repeat(2, 1)
    fr = Image.fromarray(np.clip(np.asarray(fr, np.float32) + g, 0, 255).astype(np.uint8)).convert("RGBA")
    fr.alpha_composite(overlay(t))
    enc.stdin.write(fr.convert("RGB").tobytes())
    if i % 300 == 0: print("frame", i, nfr, flush=True)
enc.stdin.close(); enc.wait(); dec.kill()

# ---------------------------------------------------------------- sound: sfx events on cuts + story beats, voice-first mix
events = []
for k, e in enumerate(edl[1:], 1):
    if e["t0"] < cta_t: events.append((e["t0"] - 0.08, "whoosh", 0.18, 0.35))
for a, off, kind, g, dur in ST.get("sfx", []):
    events.append((anchor_t(a) + off, kind, g, dur))
stats = A.mix(WD, total, events, ST.get("lufs", -14.0))
out = os.path.join(R, "video.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", os.path.join(WD, "mix.wav"), "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", out], check=True)
qc = {"duration_s": round(total, 2), "shots": len(edl), "shot_len_min_avg_max": [round(min(e["t1"] - e["t0"] for e in edl), 2),
      round(total / len(edl), 2), round(max(e["t1"] - e["t0"] for e in edl), 2)],
      "stills_used": [e["id"] for e in edl if not os.path.exists(os.path.join(WD, "clips", e["id"] + ".mp4"))]}
qc.update(stats); qc.update(A.loudness(out))
json.dump(qc, open(os.path.join(R, "qc.json"), "w"), indent=1)
# contact sheet: middle of every shot
ims = []
for e in edl:
    tm = (e["t0"] + e["t1"]) / 2
    p = os.path.join(R, "_s.jpg")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{tm:.2f}", "-i", out, "-frames:v", "1", "-vf", "scale=216:384", p], check=True)
    ims.append(Image.open(p).copy())
cols = 9; rows = (len(ims) + cols - 1) // cols
sh = Image.new("RGB", (216 * cols, 384 * rows), (16, 16, 16))
for k, im in enumerate(ims): sh.paste(im, ((k % cols) * 216, (k // cols) * 384))
sh.save(os.path.join(R, "sheet.jpg"), quality=88); os.remove(p)
print("done", json.dumps(qc))
