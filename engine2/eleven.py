"""eleven.py MODE SCRIPT.json OUTDIR
 audition: speak the first line with several narrator voices and a quote line with several character voices.
 full: speak every line (narrator N / character H voice), with word timestamps -> voice.wav + words.json
Voice choice in SCRIPT.json: {"voices": {"N": "<id or name>", "H": "<id or name>"}, "settings": {...}}"""
import os, sys, json, base64, urllib.request, subprocess, io, time
KEY = os.environ["ELEVENLABS_API_KEY"]; API = "https://api.elevenlabs.io/v1"
mode, sp, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
S = json.load(open(sp))
def req(path, body=None):
    r = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                               headers={"xi-api-key": KEY, "Content-Type": "application/json"})
    for i in range(4):
        try: return json.loads(urllib.request.urlopen(r, timeout=180).read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:300]
            if e.code in (429, 500, 502, 503) and i < 3: time.sleep(10 * (i + 1)); continue
            raise RuntimeError(f"{e.code} {msg}")
KNOWN = {"brian": "nPczCjzI2devNBz1zQrb", "callum": "N2lVS1w4EtoT3dr4eOWO"}
try: VOICES = {v["name"].split(" ")[0].lower(): v["voice_id"] for v in req("/voices")["voices"]}
except Exception: VOICES = {}
VOICES = {**KNOWN, **VOICES}
json.dump(VOICES, open(os.path.join(out, "voices.json"), "w"), indent=0)
def vid(x): return VOICES.get(x.lower(), x)
def tts(voice, text, settings, prev=None, nxt=None, model="eleven_multilingual_v2"):
    body = {"text": text, "model_id": model, "voice_settings": settings}
    if prev: body["previous_text"] = prev
    if nxt: body["next_text"] = nxt
    r = req(f"/text-to-speech/{vid(voice)}/with-timestamps?output_format=mp3_44100_128", body)
    return base64.b64decode(r["audio_base64"]), r.get("alignment") or {}
def to_wav(mp3, wav, extra_af=None):
    open(wav + ".mp3", "wb").write(mp3)
    af = ["-af", extra_af] if extra_af else []
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav + ".mp3", *af, "-ar", "44100", "-ac", "1", wav], check=True)
    os.remove(wav + ".mp3")
SET_N = S.get("settings", {}).get("N", {"stability": 0.45, "similarity_boost": 0.8, "style": 0.35, "use_speaker_boost": True, "speed": 0.95})
SET_H = S.get("settings", {}).get("H", {"stability": 0.35, "similarity_boost": 0.75, "style": 0.5, "use_speaker_boost": True, "speed": 0.9})
H_FX = S.get("h_fx", "asetrate=44100*0.94,aresample=44100,atempo=1.06,highpass=f=180,lowpass=f=5200,aecho=0.6:0.5:40:0.18")
if mode == "audition":
    first = next(l[2] for l in S["lines"] if l[0] == "N"); q = next(l[2] for l in S["lines"] if l[0] == "H")
    for v in S.get("audition_N", ["Brian", "George", "Daniel", "Adam", "Bill"]):
        if v.lower() not in VOICES: continue
        a, _ = tts(v, first, SET_N); to_wav(a, os.path.join(out, f"N_{v}.wav"))
    for v in S.get("audition_H", ["Callum", "Clyde", "Liam"]):
        if v.lower() not in VOICES: continue
        a, _ = tts(v, q, SET_H); to_wav(a, os.path.join(out, f"H_{v}.wav"), H_FX)
    sys.exit(0)
# full
import soundfile as sf, numpy as np
segs, words, t = [], [], 0.0
lines = S["lines"]; gaps = S.get("gaps", {})
for i, (who, key, text) in enumerate(lines):
    voice = S["voices"][who]; st = SET_H if who == "H" else SET_N
    prev = " ".join(l[2] for l in lines[max(0, i - 2):i] if l[0] == who) or None
    nxt = next((l[2] for l in lines[i + 1:] if l[0] == who), None)
    a, al = tts(voice, text, st, prev, nxt)
    w = os.path.join(out, f"line_{i:02d}.wav"); to_wav(a, w, H_FX if who == "H" else None)
    x, sr = sf.read(w)
    # trim leading/trailing silence (keep 40 ms)
    thr = 10 ** (-45 / 20); nz = np.where(np.abs(x) > thr)[0]
    a0 = max(0, nz[0] - int(0.04 * sr)) if len(nz) else 0; a1 = min(len(x), nz[-1] + int(0.08 * sr)) if len(nz) else len(x)
    x = x[a0:a1]; off = a0 / sr
    # words from character alignment
    ch, cs, ce = al.get("characters", []), al.get("character_start_times_seconds", []), al.get("character_end_times_seconds", [])
    cur, ws = "", None
    for c, s0, e0 in zip(ch, cs, ce):
        if c.isspace():
            if cur: words.append({"w": cur, "s": round(t + ws - off, 3), "e": round(t + we - off, 3), "line": key, "who": who})
            cur = ""
        else:
            if not cur: ws = s0
            cur += c; we = e0
    if cur: words.append({"w": cur, "s": round(t + ws - off, 3), "e": round(t + we - off, 3), "line": key, "who": who})
    segs.append({"key": key, "who": who, "start": round(t, 3), "end": round(t + len(x) / sr, 3)})
    t += len(x) / sr
    g = gaps.get(key, 0.32 if who == "N" else 0.45)
    if i + 1 < len(lines) and lines[i + 1][1] == "cta": g = max(g, 0.7)
    segs[-1]["audio"] = x; segs[-1]["gap"] = g
    t += g
sr = 44100
y = np.concatenate([np.concatenate([s.pop("audio"), np.zeros(int(s["gap"] * sr))]) for s in segs])
sf.write(os.path.join(out, "voice.wav"), y.astype(np.float32), sr)
json.dump({"words": words, "segments": segs, "total": round(len(y) / sr, 3)}, open(os.path.join(out, "words.json"), "w"), indent=0)
