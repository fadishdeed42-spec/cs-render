"""assemble.py WORKDIR : frames + voice + music + sfx -> captioned, voice-first mixed 1080x1920 MP4 + QC.

WORKDIR holds (from the decrypted job bundle + rendered chunks):
  frames/####.png   timeline.json   voice.wav   music.wav   comp.json
comp.json: {"script": "...spoken text...", "replace": [[spoken, display], ...],
            "labels": [[t0, t1, "TEXT"], ...], "cta": ["FOLLOW CHAIN SECRETS.", "EVERY CHART HIDES A STORY."],
            "sfx": [[time_or_beat, kind, gain, dur], ...], "lufs": -14}
Writes WORKDIR/result/: video.mp4, sheet.jpg, qc.json, captions.json
Prints only numbers (public logs)."""
import sys, os, json, math, re, subprocess, random
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, "..", "fonts")
FPS = 24
OW, OH = 1080, 1920
SS = 2
W, H = OW * SS, OH * SS

def clamp(x, a=0.0, b=1.0): return a if x < a else b if x > b else x
def eout(x): x = clamp(x); return 1 - (1 - x) ** 3

_f = {}
def font(name, size):
    k = (name, int(size * SS))
    if k not in _f: _f[k] = ImageFont.truetype(os.path.join(FONTS, name), int(size * SS))
    return _f[k]
CAP = "DejaVuSansCondensed-Bold.ttf"
HEAD = "Poppins-Bold.ttf"

# ------------------------------------------------------------------ captions from word timings
def norm(w): return re.sub(r"[^a-z0-9']", "", w.lower())

def caption_tokens(words, replace):
    """words: [[text, t0, t1]] -> display tokens [[DISPLAY, t0, t1, ends_sentence]] with spoken->display replacements"""
    toks = [[w, a, b] for w, a, b in words]
    out, i = [], 0
    reps = [([norm(x) for x in s.split()], d) for s, d in replace]
    while i < len(toks):
        hit = None
        for seq, disp in reps:
            n = len(seq)
            if [norm(t[0]) for t in toks[i:i + n]] == seq:
                hit = (n, disp); break
        if hit:
            n, disp = hit
            last = toks[i + n - 1][0]
            punct = last[-1] if last[-1] in ".,?!:" else ""
            out.append([disp.upper() + punct, toks[i][1], toks[i + n - 1][2]]); i += n
        else:
            out.append([toks[i][0].upper(), toks[i][1], toks[i][2]]); i += 1
    return out

def chunks_of(tokens, maxch=16, maxw=3):
    ch, cur = [], []
    for tk in tokens:
        cur.append(tk)
        txt = " ".join(t[0] for t in cur)
        if len(txt) >= maxch or len(cur) >= maxw or tk[0][-1] in ".,?!:":
            ch.append(cur); cur = []
    if cur: ch.append(cur)
    return ch

def draw_caption(d, chunks, t, y=1330):
    """karaoke caption: current chunk, active word in gold, pop-in on each chunk"""
    cur = None
    for k, c in enumerate(chunks):
        start = c[0][1] - 0.06
        nxt = chunks[k + 1][0][1] - 0.06 if k + 1 < len(chunks) else c[-1][2] + 0.5
        if start <= t < min(nxt, c[-1][2] + 0.6): cur = (c, start)
    if not cur: return
    c, start = cur
    pop = 1 + 0.10 * (1 - eout((t - start) / 0.14))
    f = font(CAP, 74 * pop)
    words = [w[0].rstrip(".,:") if j == len(c) - 1 else w[0] for j, w in enumerate(c)]
    sp = d.textlength(" ", font=f); ws = [d.textlength(w, font=f) for w in words]
    tot = sum(ws) + sp * (len(words) - 1)
    if tot > W * 0.8:
        f = font(CAP, 74 * pop * W * 0.8 / tot); sp = d.textlength(" ", font=f); ws = [d.textlength(w, font=f) for w in words]
        tot = sum(ws) + sp * (len(words) - 1)
    x = W / 2 - tot / 2
    for j, w in enumerate(words):
        a, b = c[j][1], c[j][2]
        active = a - 0.04 <= t <= b + 0.08
        col = (255, 214, 64) if active else (255, 255, 255)
        d.text((x + 5 * SS, y * SS + 6 * SS), w, font=f, fill=(0, 0, 0, 200), anchor="lm", stroke_width=8 * SS, stroke_fill=(0, 0, 0, 200))
        d.text((x, y * SS), w, font=f, fill=col, anchor="lm", stroke_width=8 * SS, stroke_fill=(10, 10, 12))
        x += ws[j] + sp

def chip(d, text, a, y=150):
    if a <= 0: return
    f = font(HEAD, 44); tw = d.textlength(text, font=f) / SS
    d.rounded_rectangle([60 * SS, y * SS, (60 + tw + 48) * SS, (y + 76) * SS], radius=16 * SS, fill=(14, 15, 18, int(225 * a)))
    d.text((84 * SS, (y + 39) * SS), text, font=f, fill=(255, 255, 255, int(255 * a)), anchor="lm")

def end_card(d, k, lines):
    if k <= 0: return
    d.rectangle([0, 0, W, H], fill=(6, 6, 8, int(215 * k)))
    y = 860
    for i, s in enumerate(lines):
        size = 92 if i == 0 else 54
        col = (255, 255, 255) if i == 0 else (255, 210, 60)
        f = font(CAP, size); tw = d.textlength(s, font=f)
        if tw > W * 0.84: f = font(CAP, size * W * 0.84 / tw)
        d.text((W / 2, y * SS), s, font=f, fill=col + (int(255 * k),), anchor="mm", stroke_width=6 * SS, stroke_fill=(0, 0, 0, int(255 * k)))
        y += size * 1.35

CTX = {}
def overlay(t):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    for t0, t1, txt in CTX["labels"]:
        a = clamp((t - t0) / 0.25) * clamp((t1 - t) / 0.25)
        chip(d, txt, a)
    cta_t = CTX["cta_t"]
    if t < cta_t - 0.05:
        draw_caption(d, CTX["chunks"], t)
    else:
        end_card(d, eout((t - cta_t + 0.05) / 0.4), CTX["cta"])
    return im.resize((OW, OH), Image.LANCZOS)

_vig = None
def vignette():
    global _vig
    if _vig is None:
        v = Image.new("L", (OW // 8, OH // 8), 0); px = v.load()
        for y in range(v.height):
            for x in range(v.width):
                dx = (x / v.width - 0.5) * 1.5; dy = (y / v.height - 0.5) * 1.15
                r = math.sqrt(dx * dx + dy * dy)
                px[x, y] = int(255 * (1 - 0.32 * clamp((r - 0.5) / 0.6) ** 1.6))
        _vig = Image.merge("RGB", [v.resize((OW, OH), Image.BICUBIC)] * 3)
    return _vig

def comp_frame(i):
    t = (i - 1) / FPS
    src = os.path.join(CTX["wd"], "frames", f"{i:04d}.png")
    img = Image.open(src).convert("RGB").resize((OW, OH), Image.LANCZOS)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=55, threshold=2))
    img = ImageChops.multiply(img, vignette()).convert("RGBA")
    img.alpha_composite(overlay(t))
    img.convert("RGB").save(os.path.join(CTX["wd"], "comp", f"{i:04d}.jpg"), quality=94)
    return i

def init(ctx):
    CTX.update(ctx)

# ------------------------------------------------------------------ sound
SR = 24000
rng = np.random.default_rng(4)
def env_exp(n, tau): return np.exp(-np.arange(n) / (tau * SR))
def tone(f, dur, tau=0.2, harm=(1, .4, .15)):
    n = int(dur * SR); x = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * (i + 1) * x) for i, a in enumerate(harm)) * env_exp(n, tau)
def noise(dur, smooth=1):
    n = int(dur * SR); w = rng.standard_normal(n)
    return np.convolve(w, np.ones(smooth) / smooth, "same") if smooth > 1 else w
def sfx(kind, dur=0.5):
    n = int(dur * SR); x = np.arange(n) / SR
    if kind == "whoosh":
        m = int(0.5 * SR); return noise(0.5, 24) * np.sin(np.pi * np.linspace(0, 1, m)) ** 2 * 1.6
    if kind == "click": m = int(0.03 * SR); return noise(0.03) * env_exp(m, 0.004)
    if kind == "tick": return tone(2200, 0.06, 0.012, (1,))
    if kind == "beep": return tone(880, 0.16, 0.5, (1,)) * (np.arange(int(0.16 * SR)) < 0.13 * SR)
    if kind == "ping": return tone(1320, 0.5, 0.12)
    if kind == "clank":
        a = tone(420, 0.5, 0.15, (1, .6, .4, .3)) + tone(615, 0.5, 0.1, (1, .4)); return a
    if kind == "thud":
        m = int(0.45 * SR); y = np.arange(m) / SR; return np.sin(2 * np.pi * (95 - 55 * y) * y) * env_exp(m, 0.09)
    if kind == "slam":
        a = sfx("thud"); a[:int(0.06 * SR)] += noise(0.06)[:int(0.06 * SR)] * 0.5 * env_exp(int(0.06 * SR), 0.015); return a
    if kind == "slide": return noise(dur, 4) * np.sin(np.pi * x / dur) ** 2
    if kind == "hum": return (np.sin(2 * np.pi * 120 * x) + .5 * np.sin(2 * np.pi * 180 * x)) * np.minimum(1, np.minimum(x, dur - x) * 3)
    if kind == "wind": return noise(dur, 60) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * x)) * np.minimum(1, np.minimum(x, dur - x) * 2)
    if kind == "engine":
        f = 80 + 220 * (x / dur); return (np.sin(2 * np.pi * np.cumsum(f) / SR) + noise(dur, 6) * (x / dur)) * np.minimum(1, (dur - x) * 3)
    if kind == "riser":
        f = 180 * np.exp(np.log(5) * x / dur); return np.sin(2 * np.pi * np.cumsum(f) / SR) * (x / dur) ** 2
    if kind == "spark": return noise(dur, 2) * (rng.random(n) > 0.6) * env_exp(n, 0.06)
    if kind == "crunch":
        out = np.zeros(n); t = 0.0
        while t < dur - 0.05:
            i = int(t * SR); c = noise(0.05, 2) * env_exp(int(0.05 * SR), 0.01) * rng.uniform(0.5, 1)
            out[i:i + len(c)] += c[:n - i]; t += rng.uniform(0.02, 0.06)
        return out + sfx("thud")[:n] if n >= int(0.45 * SR) else out
    if kind == "alarm":
        f = 700 + 250 * np.sign(np.sin(2 * np.pi * 2 * x)); return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, (dur - x) * 4)
    if kind == "siren":
        f = 650 + 300 * np.sin(2 * np.pi * 0.9 * x); return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, np.minimum(x, dur - x) * 3)
    if kind == "steam": return noise(dur, 3) * np.minimum(1, np.minimum(x, dur - x) * 4)
    if kind == "plug": return np.concatenate([noise(0.02) * 0.6, tone(160, 0.25, 0.05, (1, .5))])
    raise ValueError(kind)

def ffilter(x, af):
    a, b = "/tmp/_fi.wav", "/tmp/_fo.wav"
    sf.write(a, x.astype(np.float32), SR, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a, "-af", af, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", b], check=True)
    y, _ = sf.read(b); return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)]
def db(x): return 20 * np.log10(np.maximum(x, 1e-9))
def undb(d): return 10 ** (d / 20)
def rms_win(x, win):
    w = int(win * SR); k = len(x) // w
    return np.sqrt(np.mean(x[:k * w].reshape(k, w) ** 2, axis=1) + 1e-12), w

def mix(wd, total, events, lufs=-14.0):
    v, sr = sf.read(os.path.join(wd, "voice.wav"))
    if v.ndim > 1: v = v.mean(1)
    if sr != SR: v = ffilter_sr(v, sr)
    n = int(total * SR)
    voice = np.zeros(n); voice[:min(n, len(v))] = v[:n]
    voice = ffilter(voice, "highpass=f=80,acompressor=threshold=-22dB:ratio=2.5:attack=6:release=120:makeup=1.5,equalizer=f=3000:t=q:w=1.2:g=1.5")
    r, w = rms_win(voice, 0.02)
    act = (db(r) > db(np.percentile(r[r > 1e-4], 90)) - 24).astype(float) if np.any(r > 1e-4) else np.zeros_like(r)
    look = 6; act = np.concatenate([act[look:], np.zeros(look)])
    env = np.zeros_like(act); vv = 0.0
    for i, a in enumerate(act):
        vv += (a - vv) * ((1 - np.exp(-0.02 / 0.06)) if a > vv else (1 - np.exp(-0.02 / 0.45))); env[i] = vv
    env = np.pad(np.repeat(env, w), (0, n))[:n]
    vr = np.sqrt(np.mean(voice[np.pad(np.repeat(act, w), (0, n))[:n].astype(bool)] ** 2)) if act.any() else 0.05
    m, msr = sf.read(os.path.join(wd, "music.wav"))
    if m.ndim > 1: m = m.mean(1)
    if msr != SR: m = ffilter_sr(m, msr)
    reps = int(np.ceil(n / len(m))) + 1
    music = np.tile(m, reps)[:n]
    fade = int(1.5 * SR); music[-fade:] *= np.linspace(1, 0, fade)
    music = ffilter(music, "highpass=f=55,lowpass=f=6000")
    mr = np.sqrt(np.mean(music ** 2)) + 1e-9
    music *= vr / mr * undb(-14 + (-24 + 14) * env)            # -24 dB under speech, -14 in pauses
    fx = np.zeros(n)
    for at, kind, g, dur in events:
        s = sfx(kind, dur); s = s / (np.max(np.abs(s)) + 1e-9) * g
        i = int(max(0, at) * SR)
        if i < n: fx[i:i + len(s)] += s[:n - i]
    fx = ffilter(fx, "highpass=f=110,lowpass=f=8000")
    fr = np.sqrt(np.mean(fx[np.abs(fx) > 1e-4] ** 2)) + 1e-9 if np.any(np.abs(fx) > 1e-4) else 1
    fx *= vr / fr * undb(-9 + (-20 + 9) * env)                 # -20 dB under speech, -9 in pauses
    # voice-keyed limiter on the bed (music + fx) in 50 ms windows
    bed = music + fx
    br, w5 = rms_win(bed, 0.05); lv, _ = rms_win(voice, 0.05); e5 = env[:len(br) * w5].reshape(len(br), w5).mean(1)
    lvs = np.array([lv[max(0, i - 3): i + 4].min() for i in range(len(lv))])
    allowed = vr * undb(-12) * (1 - e5) + undb(-20) * np.maximum(lvs, vr * undb(-16)) * e5
    g = np.minimum(1, allowed / np.maximum(br, 1e-9))
    sm = g.copy()
    for i in range(1, len(sm)): sm[i] = g[i] if g[i] < sm[i - 1] else sm[i - 1] + (g[i] - sm[i - 1]) * 0.22
    gg = np.pad(np.repeat(sm, w5), (0, n))[:n]; bed *= gg
    out = voice + bed
    out /= max(1.0, np.max(np.abs(out)) / 0.9)
    raw0 = os.path.join(wd, "mix_raw0.wav"); sf.write(raw0, out.astype(np.float32), SR, subtype="FLOAT")
    # pre-gain to the target, then a soft limiter so loudnorm can reach it without hitting the peak ceiling
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", raw0, "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
    i0 = float(r[r.rindex("Summary:"):].split("I:")[1].split()[0])
    raw = os.path.join(wd, "mix_raw.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw0, "-af",
                    f"volume={lufs - i0 + 0.3:.2f}dB,alimiter=limit=0.70:attack=3:release=60:level=disabled",
                    "-ar", str(SR), "-c:a", "pcm_f32le", raw], check=True)
    # two-pass loudness
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", raw, "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=9:print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
    j = json.loads(r[r.rindex("{"):r.rindex("}") + 1])
    af = (f"loudnorm=I={lufs}:TP=-1.5:LRA=9:measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}:"
          f"measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-af", af, "-ar", "48000", os.path.join(wd, "mix.wav")], check=True)
    # stats: bed under voice on spoken words (100 ms)
    vw, _ = rms_win(voice, 0.1); bw, _ = rms_win(bed, 0.1)
    mask = db(vw) > db(vr) - 12
    d = db(bw[mask]) - db(vw[mask])
    return {"bed_minus_voice_median_dB": round(float(np.median(d)), 1), "bed_minus_voice_p95_dB": round(float(np.percentile(d, 95)), 1),
            "bed_minus_voice_worst_dB": round(float(d.max()), 1)}

def ffilter_sr(x, sr):
    a, b = "/tmp/_ri.wav", "/tmp/_ro.wav"; sf.write(a, x.astype(np.float32), sr, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", b], check=True)
    y, _ = sf.read(b); return y

def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    s = r[r.rindex("Summary:"):]; g = lambda k: float(s.split(k)[1].split()[0])
    return {"I_LUFS": g("I:"), "true_peak_dBTP": g("Peak:")}

def asr_check(path, script):
    import difflib
    from faster_whisper import WhisperModel
    m = WhisperModel("base.en", device="cpu", compute_type="int8")
    segs, _ = m.transcribe(path, language="en", beam_size=3, word_timestamps=False)
    got = " ".join(s.text for s in segs)
    num = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    nz = lambda t: re.sub(r"[^a-z ]", " ", t.lower()).split()
    a, b = nz(script), nz(got)
    sm = difflib.SequenceMatcher(None, a, b)
    diffs = [(" ".join(a[i1:i2]), " ".join(b[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    return {"asr_match": round(sm.ratio(), 3), "asr_heard": got.strip(), "asr_diffs": diffs[:40]}

def main(wd):
    tl = json.load(open(os.path.join(wd, "timeline.json"))); comp = json.load(open(os.path.join(wd, "comp.json")))
    beats = tl["beats"]; total = tl["total"]
    T = lambda x: x if isinstance(x, (int, float)) else (beats[x[0]] + x[1] if isinstance(x, list) else beats[x])
    tokens = caption_tokens(tl["words"], comp.get("replace", []))
    cta_t = beats.get("cta", total)
    tokens = [t for t in tokens if t[1] < cta_t - 0.05]
    chunks = chunks_of(tokens)
    labels = [[T(a), T(b), s] for a, b, s in comp.get("labels", [])]
    os.makedirs(os.path.join(wd, "comp"), exist_ok=True); os.makedirs(os.path.join(wd, "result"), exist_ok=True)
    frames = sorted(int(f[:4]) for f in os.listdir(os.path.join(wd, "frames")) if f.endswith(".png"))
    ctx = {"wd": wd, "chunks": chunks, "labels": labels, "cta_t": cta_t, "cta": comp.get("cta", ["FOLLOW CHAIN SECRETS.", "EVERY CHART HIDES A STORY."])}
    with Pool(os.cpu_count(), initializer=init, initargs=(ctx,)) as p:
        for k, _ in enumerate(p.imap_unordered(comp_frame, frames, chunksize=8)):
            if k % 200 == 0: print("comp", k, len(frames), flush=True)
    events = [(T(e[0]), e[1], e[2], e[3] if len(e) > 3 else 0.5) for e in comp.get("sfx", [])]
    stats = mix(wd, total, events, comp.get("lufs", -14.0))
    out = os.path.join(wd, "result", "video.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-start_number", str(frames[0]), "-i", os.path.join(wd, "comp", "%04d.jpg"),
                    "-i", os.path.join(wd, "mix.wav"), "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    qc = {"frames": len(frames), "missing_frames": [i for i in range(1, frames[-1] + 1) if i not in set(frames)][:20],
          "duration_s": round(len(frames) / FPS, 2)}
    qc.update(stats); qc.update(loudness(out))
    try: qc.update(asr_check(os.path.join(wd, "mix.wav"), comp["script"]))
    except Exception as e: qc["asr_error"] = str(e)[:200]
    # contact sheet: one frame from the middle of every shot change + regular samples
    pick = sorted(set([max(1, min(frames[-1], int(round(T(b) * FPS)) + 1 + 12)) for b in beats]))
    pick = [p for p in pick if p in set(frames)][:40]
    ims = [Image.open(os.path.join(wd, "comp", f"{i:04d}.jpg")).resize((216, 384)) for i in pick]
    cols = 8; rows = (len(ims) + cols - 1) // cols
    sh = Image.new("RGB", (216 * cols, 384 * rows), (20, 20, 20))
    for k, im in enumerate(ims): sh.paste(im, ((k % cols) * 216, (k // cols) * 384))
    sh.save(os.path.join(wd, "result", "sheet.jpg"), quality=88)
    json.dump(qc, open(os.path.join(wd, "result", "qc.json"), "w"), indent=1)
    json.dump({"chunks": [[[w[0], round(w[1], 2), round(w[2], 2)] for w in c] for c in chunks]}, open(os.path.join(wd, "result", "captions.json"), "w"))
    print("done", qc["frames"], qc.get("I_LUFS"), qc.get("bed_minus_voice_median_dB"), qc.get("asr_match"))

if __name__ == "__main__":
    main(sys.argv[1])
