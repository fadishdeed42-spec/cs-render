"""Voice-first mixer.

Stems are built and kept separately (voice / music / sfx), then:
  voice  : high-pass 80 Hz, gentle compression, +2 dB presence at 3 kHz, levelled
  music  : band-limited, sits 22 dB under the voice while it speaks, 12 dB under in pauses
  sfx    : band-limited (no sub-bass, no piercing highs), 17 dB under the voice while it
           speaks, 9 dB under in pauses, peaks limited below the voice peak
Ducking is a sidechain envelope from the voice with 120 ms look-ahead, ~60 ms attack,
~450 ms release, so background dips just before each phrase and recovers smoothly.
Master: two-pass EBU R128 normalisation to -15 LUFS integrated, true peak -1.5 dBTP.
"""
import json, os, subprocess
import numpy as np, soundfile as sf

SR = 24000
MUSIC_SPEECH_DB, MUSIC_GAP_DB = -22.0, -12.0
SFX_SPEECH_DB, SFX_GAP_DB = -20.0, -9.0
DROP = {"glitch", "buzz", "alarm", "rumble"}        # harsh or bass-heavy: never used

def db(x): return 20 * np.log10(np.maximum(x, 1e-9))
def undb(d): return 10 ** (d / 20)

def ffilter(x, af):
    """run a numpy mono signal through an ffmpeg audio filter chain"""
    tmp_in, tmp_out = "/tmp/_a_in.wav", "/tmp/_a_out.wav"
    sf.write(tmp_in, x.astype(np.float32), SR, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_in, "-af", af, "-ar", str(SR), "-ac", "1",
                    "-c:a", "pcm_f32le", tmp_out], check=True)
    y, _ = sf.read(tmp_out)
    n = len(x)
    return np.pad(y, (0, max(0, n - len(y))))[:n]

def rms_win(x, win=0.05):
    w = int(win * SR)
    k = len(x) // w
    return np.sqrt(np.mean(x[:k * w].reshape(k, w) ** 2, axis=1) + 1e-12), w

def speech_envelope(voice):
    """0..1 envelope, 1 while speaking. look-ahead + attack/release smoothing."""
    r, w = rms_win(voice, 0.02)
    thr = np.percentile(r[r > 1e-4], 20) if np.any(r > 1e-4) else 1e-3
    act = (r > max(thr, 0.004)).astype(float)
    # bridge short gaps inside a sentence (< 250 ms)
    gap = int(0.25 / 0.02)
    i = 0
    while i < len(act):
        if act[i] == 0:
            j = i
            while j < len(act) and act[j] == 0: j += 1
            if 0 < i and j < len(act) and (j - i) <= gap: act[i:j] = 1
            i = j
        else: i += 1
    look = int(0.12 / 0.02)
    act = np.concatenate([act[look:], np.zeros(look)])
    act = np.maximum(act, np.concatenate([np.zeros(look), act[:-look]]) * 0)  # keep shape
    env = np.zeros_like(act); a_up = 1 - np.exp(-0.02 / 0.06); a_dn = 1 - np.exp(-0.02 / 0.45)
    v = 0.0
    for i, t in enumerate(act):
        v += (t - v) * (a_up if t > v else a_dn); env[i] = v
    e = np.repeat(env, w)
    return np.pad(e, (0, len(voice) - len(e)), mode="edge")[:len(voice)], act

def active_rms(x, env=None, thresh_db=-50):
    r, w = rms_win(x, 0.05)
    m = db(r) > (db(np.max(r)) + thresh_db if np.max(r) > 0 else -200)
    return float(np.sqrt(np.mean(r[m] ** 2))) if np.any(m) else 1e-6

def loudnorm_2pass(inp, out, I=-15.0, TP=-1.5, LRA=9):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", inp, "-af",
                        f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    j = json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])
    af = (f"loudnorm=I={I}:TP={TP}:LRA={LRA}:measured_I={j['input_i']}:measured_TP={j['input_tp']}:"
          f"measured_LRA={j['input_lra']}:measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", inp, "-af", af, "-ar", "48000", "-c:a", "pcm_s24le", out], check=True)

def measure(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    tail = r[r.rindex("Summary:"):]
    get = lambda key: float(tail.split(key)[1].split()[0])
    return {"I_LUFS": get("I:"), "LRA": get("LRA:"), "TP_dBTP": get("Peak:")}

def build(base, voice_parts, music_sig, sfx_events, total, sfx_fn):
    """voice_parts: [(start_sec, path)], sfx_events: [(t, kind, gain, dur)]"""
    n = int(total * SR) + SR
    voice = np.zeros(n)
    for at, path in voice_parts:
        x, sr = sf.read(path)
        i = int(at * SR); voice[i:i + len(x)] += x[:n - i]
    voice = ffilter(voice, "highpass=f=80,lowpass=f=11000,"
                           "acompressor=threshold=-22dB:ratio=3:attack=6:release=120:makeup=2,"
                           "equalizer=f=3000:t=q:w=1.2:g=2,equalizer=f=250:t=q:w=1:g=-1.5")
    v_rms = active_rms(voice)
    env, act = speech_envelope(voice)

    music = np.zeros(n); music[:len(music_sig)] = music_sig[:n]
    music = ffilter(music, "highpass=f=55,lowpass=f=5000")
    m_rms = active_rms(music)
    g_m = undb(MUSIC_GAP_DB + (MUSIC_SPEECH_DB - MUSIC_GAP_DB) * env) * v_rms / m_rms
    music_out = music * g_m

    fx = np.zeros(n)
    for at, kind, g, dur in sfx_events:
        if kind in DROP: continue
        s = sfx_fn(kind, dur)
        s = s / (np.max(np.abs(s)) + 1e-9) * min(1.0, g)       # each event peak-normalised * its relative gain
        i = int(max(0, at) * SR)
        if i < n: fx[i:i + len(s)] += s[:n - i]
    fx = ffilter(fx, "highpass=f=120,lowpass=f=8000")
    f_rms = active_rms(fx)
    g_f = undb(SFX_GAP_DB + (SFX_SPEECH_DB - SFX_GAP_DB) * env) * v_rms / f_rms
    fx_out = fx * g_f
    # voice-keyed RMS limiter: in 50 ms windows the sfx may never exceed its allowed level
    def keyed_limit(sig, speech_db, gap_db):
        r, w = rms_win(sig, 0.05)
        e_w = env[: len(r) * w].reshape(len(r), w).mean(1)
        lv, _ = rms_win(voice, 0.05)
        lv = lv[: len(r)]
        k = 3
        lv_s = np.array([lv[max(0, i - k): i + k + 1].min() for i in range(len(lv))])   # quietest nearby voice
        a_sp = undb(speech_db) * np.maximum(lv_s, v_rms * undb(-16))
        a_gp = v_rms * undb(gap_db)
        allowed = a_gp * (1 - e_w) + a_sp * e_w
        g = np.minimum(1.0, allowed / np.maximum(r, 1e-9))
        sm = np.copy(g); a = 1 - np.exp(-0.05 / 0.2)
        for i in range(1, len(sm)): sm[i] = g[i] if g[i] < sm[i - 1] else sm[i - 1] + (g[i] - sm[i - 1]) * a
        gg = np.repeat(sm, w); gg = np.pad(gg, (0, len(sig) - len(gg)), mode="edge")[:len(sig)]
        return sig * gg
    fx_out = keyed_limit(fx_out, SFX_SPEECH_DB, SFX_GAP_DB)
    music_out = keyed_limit(music_out, MUSIC_SPEECH_DB, MUSIC_GAP_DB)
    # sfx peak limiter: never above voice peak - 6 dB
    lim = np.percentile(np.abs(voice[np.abs(voice) > 0]), 99.5) * undb(-6) if np.any(voice) else 0.1
    fx_out = np.tanh(fx_out / lim) * lim

    for name, sig in (("voice", voice), ("music", music_out), ("sfx", fx_out)):
        sf.write(f"{base}/stem_{name}.wav", sig.astype(np.float32), SR, subtype="FLOAT")
    mixd = voice + music_out + fx_out
    mixd /= max(1.0, np.max(np.abs(mixd)) / 0.9)
    sf.write(f"{base}/mix_raw.wav", mixd.astype(np.float32), SR, subtype="FLOAT")
    loudnorm_2pass(f"{base}/mix_raw.wav", f"{base}/mix.wav")

    # ---- QC: voice vs background inside speech, 400 ms windows
    bg = music_out + fx_out
    vr, w = rms_win(voice, 0.4); br, _ = rms_win(bg, 0.4)
    sp, _ = rms_win(np.repeat(act, int(0.02 * SR))[:len(voice)] if len(act) else voice, 0.4)
    mask = (sp > 0.6) & (vr > 1e-4)
    diff = db(br[mask]) - db(vr[mask])
    qc = measure(f"{base}/mix.wav")
    qc.update({"bg_minus_voice_dB_median": round(float(np.median(diff)), 1),
               "bg_minus_voice_dB_worst": round(float(np.max(diff)), 1),
               "speech_windows": int(mask.sum())})
    json.dump(qc, open(f"{base}/audio_qc.json", "w"), indent=1)
    return qc
