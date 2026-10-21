"""Procedural music bed + sound effects (numpy only)."""
import numpy as np

SR = 24000
_story = None

def _init(name):
    global _story
    mod = importlib.import_module(f"stories.{name}")
    _story = mod.STORY(f"out/{name}")

def _render(fi): return _story.render(fi)

# ------------------------------------------------------------- audio synthesis
rng = np.random.default_rng(4)
def env_exp(n, tau): return np.exp(-np.arange(n) / (tau * SR))
def tone(f, dur, tau=0.2, harm=(1, .4, .15)):
    n = int(dur * SR); x = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * f * (i + 1) * x) for i, a in enumerate(harm))
    return s * env_exp(n, tau)
def noise(dur, smooth=1):
    n = int(dur * SR); w = rng.standard_normal(n)
    if smooth > 1: w = np.convolve(w, np.ones(smooth) / smooth, "same")
    return w
def sfx(kind, dur=0.5):
    if kind == "whoosh":
        n = int(0.5 * SR); w = noise(0.5, 24) * np.sin(np.pi * np.linspace(0, 1, n)) ** 2; return w * 1.6
    if kind == "pop":
        n = int(0.12 * SR); x = np.arange(n) / SR
        return np.sin(2 * np.pi * (900 - 3000 * x) * x) * env_exp(n, 0.03)
    if kind == "ping":
        return tone(1320, 0.5, 0.12) * 0.8
    if kind == "coin":
        return np.concatenate([tone(1568, 0.07, 0.05), tone(2093, 0.45, 0.12)]) * 0.7
    if kind == "cash":
        a = tone(2093, 0.6, 0.2) + tone(2637, 0.6, 0.18) * 0.7
        return np.concatenate([noise(0.05) * 0.4 * env_exp(int(0.05 * SR), 0.01), a]) * 0.6
    if kind == "thud":
        n = int(0.45 * SR); x = np.arange(n) / SR
        return np.sin(2 * np.pi * (95 - 55 * x) * x) * env_exp(n, 0.09) * 1.4
    if kind == "stamp":
        n = int(0.5 * SR); x = np.arange(n) / SR
        return (np.sin(2 * np.pi * (70 - 30 * x) * x) * env_exp(n, 0.12) * 1.5 + noise(0.5, 3) * env_exp(n, 0.02) * 0.6)
    if kind == "buzz":
        n = int(0.45 * SR); x = np.arange(n) / SR
        return np.sign(np.sin(2 * np.pi * 110 * x)) * 0.35 * (env_exp(n, 0.4)) * (x < 0.4)
    if kind == "beep":
        return tone(880, 0.16, 0.5, (1,)) * (np.arange(int(0.16 * SR)) < 0.13 * SR) * 0.6
    if kind == "alarm":
        n = int(dur * SR); x = np.arange(n) / SR
        f = 700 + 250 * np.sign(np.sin(2 * np.pi * 2.5 * x))
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25 * np.minimum(1, (dur - x) * 4)
    if kind == "click":
        n = int(0.03 * SR); return noise(0.03) * env_exp(n, 0.004) * 0.8
    if kind == "typing":
        out = np.zeros(int(dur * SR))
        t = 0.0
        while t < dur - 0.05:
            i = int(t * SR); c = sfx("click") * rng.uniform(0.4, 1)
            out[i:i + len(c)] += c[:len(out) - i]; t += rng.uniform(0.05, 0.13)
        return out
    if kind == "riser":
        n = int(dur * SR); x = np.arange(n) / SR
        f = 180 * np.exp(np.log(6) * x / dur)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * (x / dur) ** 2 * 0.5 + noise(dur, 8) * (x / dur) ** 3 * 0.4
    if kind == "rain":
        out = np.zeros(int(dur * SR)); t = 0.0
        while t < dur - 0.3:
            i = int(t * SR); c = sfx("coin") * rng.uniform(0.2, 0.6)
            out[i:i + len(c)] += c[:len(out) - i]; t += rng.uniform(0.04, 0.12)
        return out
    if kind == "glitch":
        n = int(0.35 * SR); return noise(0.35, 2) * (rng.random(n) > 0.5) * env_exp(n, 0.15) * 0.5
    if kind == "siren":
        n = int(dur * SR); x = np.arange(n) / SR
        f = 650 + 300 * np.sin(2 * np.pi * 0.9 * x)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.22 * np.minimum(1, np.minimum(x, dur - x) * 3)
    if kind == "freeze":
        n = int(0.9 * SR); x = np.arange(n) / SR
        return (noise(0.9, 2) * 0.3 + np.sin(2 * np.pi * (2400 - 1800 * x) * x) * 0.2) * env_exp(n, 0.25)
    if kind == "wind":
        n = int(dur * SR); x = np.arange(n) / SR
        w = noise(dur, 60) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * x)) * np.minimum(1, np.minimum(x, dur - x) * 2)
        return w * 1.5
    if kind == "hum":
        n = int(dur * SR); x = np.arange(n) / SR
        h = np.sin(2 * np.pi * 60 * x) + 0.5 * np.sin(2 * np.pi * 120 * x) + 0.25 * np.sin(2 * np.pi * 180 * x)
        return h * 0.3 * np.minimum(1, np.minimum(x, dur - x) * 3)
    if kind == "engine":
        n = int(dur * SR); x = np.arange(n) / SR
        f = 80 + 220 * (x / dur)
        e = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3 + noise(dur, 6) * 0.35 * (x / dur)
        return e * np.minimum(1, (dur - x) * 3)
    if kind == "creak":
        n = int(0.6 * SR); x = np.arange(n) / SR
        f = 300 + 120 * np.sin(2 * np.pi * 3 * x)
        return np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.12 * np.sin(np.pi * x / 0.6)
    if kind == "slam":
        return sfx("thud") * 0.9 + np.concatenate([noise(0.08) * env_exp(int(0.08 * SR), 0.015) * 0.6, np.zeros(int(0.37 * SR))])
    if kind == "clank":
        a = tone(420, 0.5, 0.15, (1, .6, .4, .3)) + tone(615, 0.5, 0.1, (1, .4))
        return a * 0.5 + np.concatenate([noise(0.04) * 0.5, np.zeros(len(a) - int(0.04 * SR))])
    if kind == "slide":
        n = int(0.7 * SR); x = np.arange(n) / SR
        return noise(0.7, 4) * 0.35 * np.sin(np.pi * x / 0.7) ** 2
    if kind == "heartbeat":
        a = sfx("thud")[: int(0.3 * SR)]
        out = np.zeros(int(0.7 * SR)); out[:len(a)] += a; j = int(0.22 * SR); out[j:j + len(a)] += a * 0.7
        return out
    if kind == "rumble":
        n = int(dur * SR); x = np.arange(n) / SR
        return noise(dur, 200) * 6 * np.minimum(1, np.minimum(x, dur - x) * 2)
    if kind == "tick":
        return tone(2400, 0.05, 0.01, (1,)) * 0.5
    raise ValueError(kind)

def music(total, root=55.0, mood="tense", bpm=100):
    n = int(total * SR); t = np.arange(n) / SR
    beat = 60 / bpm
    minor = [0, 3, 7, 10]; major = [0, 4, 7, 11]
    sc = minor if mood == "tense" else major
    prog = [0, -4, -2, -5] if mood == "tense" else [0, 5, -3, 7]
    out = np.zeros(n)
    bar = beat * 4
    # pad
    for bi in range(int(total / bar) + 1):
        r = root * 2 ** (prog[bi % 4] / 12)
        i0, i1 = int(bi * bar * SR), min(n, int((bi + 1) * bar * SR))
        if i0 >= n: break
        x = np.arange(i1 - i0) / SR
        seg = np.zeros(i1 - i0)
        for k, iv in enumerate(sc[:3]):
            f = r * 2 * 2 ** (iv / 12)
            seg += (np.sin(2 * np.pi * f * x) + 0.3 * np.sin(2 * np.pi * f * 2.003 * x)) * 0.22
        a = np.minimum(1, x / 0.4) * np.minimum(1, (bar - x) / 0.4)
        out[i0:i1] += seg * a * 0.5
        # bass
        bx = np.sin(2 * np.pi * r * x) * 0.55
        out[i0:i1] += bx * a
        # arpeggio 8ths
        for s in (range(8) if mood != "dark" else ([0, 3, 6] if bi % 2 == 0 else [])):
            st = bi * bar + s * beat / 2
            if st >= total: break
            f = r * 4 * 2 ** (sc[(s * 3 + bi) % 4] / 12)
            pl = tone(f, 0.3, 0.08, (1, .25)) * 0.16
            j = int(st * SR); out[j:j + len(pl)] += pl[:n - j]
    # drums
    for k in range(int(total / beat)):
        if mood == "dark" and k % 2: continue
        st = k * beat; j = int(st * SR)
        kick = sfx("thud")[: int(0.3 * SR)] * 0.55
        out[j:j + len(kick)] += kick[:n - j]
        if mood == "dark": continue
        hj = int((st + beat / 2) * SR)
        hat = noise(0.05) * env_exp(int(0.05 * SR), 0.01) * 0.12
        out[hj:hj + len(hat)] += hat[:max(0, n - hj)]
    out /= np.max(np.abs(out)) + 1e-9
    fade = np.minimum(1, t / 1.0) * np.minimum(1, (total - t) / 2.0)
    return out * fade

