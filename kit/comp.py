"""2D overlay on top of the 3D frames: word-exact captions, small labels, end card, vignette.
Caption chunks come from the story's CAPS = [(display_text, n_spoken_words), ...]; each chunk is shown
from its first word's start, and the word being spoken right now is highlighted."""
import math, os
from PIL import Image, ImageDraw, ImageFont, ImageChops

OW, OH = 1080, 1920
SS = 2
FONT_CAP = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
                             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if os.path.exists(p)), None)
FONT_LAB = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_CAP) if p and os.path.exists(p)), None)
_f = {}
def font(path, size):
    k = (path, int(size))
    if k not in _f: _f[k] = ImageFont.truetype(path, int(size))
    return _f[k]

def clamp(x, a=0.0, b=1.0): return a if x < a else b if x > b else x
def eout(x): x = clamp(x); return 1 - (1 - x) ** 3

class Captions:
    def __init__(self, words, caps):
        self.W = words.w
        n = sum(c[1] for c in caps)
        if n != len(self.W):
            raise ValueError(f"CAPS consume {n} words but the voice has {len(self.W)}")
        self.chunks = []; i = 0
        for text, k in caps:
            toks = text.split()
            ws = self.W[i:i + k]
            self.chunks.append((ws[0][0], ws[-1][1], text, ws, toks)); i += k
    def at(self, t):
        cur = None
        for i, c in enumerate(self.chunks):
            nxt = self.chunks[i + 1][0] if i + 1 < len(self.chunks) else c[1] + 0.6
            if c[0] - 0.05 <= t < max(nxt, c[1] + 0.05): cur = (i, c)
        return cur

def draw_caption(d, cap, t, y=1330):
    hit = cap.at(t)
    if not hit: return
    i, (a, b, text, ws, toks) = hit
    # which display token is "live": map spoken-word progress onto display tokens
    k = sum(1 for w in ws if w[0] <= t + 0.03)
    live = min(len(toks) - 1, max(0, int(round((k / max(1, len(ws))) * len(toks))) - 1)) if t <= b + 0.05 else -1
    pop = 1 + 0.10 * (1 - eout((t - a) / 0.14))
    size = 82 * pop * SS
    f = font(FONT_CAP, size)
    sp = d.textlength(" ", font=f); wd = [d.textlength(x, font=f) for x in toks]
    tot = sum(wd) + sp * (len(toks) - 1)
    if tot > OW * SS * 0.9:
        size *= OW * SS * 0.9 / tot; f = font(FONT_CAP, size)
        sp = d.textlength(" ", font=f); wd = [d.textlength(x, font=f) for x in toks]; tot = sum(wd) + sp * (len(toks) - 1)
    x = OW * SS / 2 - tot / 2
    for j, tok in enumerate(toks):
        col = (255, 214, 64, 255) if j == live else (255, 255, 255, 255)
        d.text((x + 5 * SS, y * SS + 6 * SS), tok, font=f, fill=(0, 0, 0, 170), anchor="lm", stroke_width=9 * SS, stroke_fill=(0, 0, 0, 170))
        d.text((x, y * SS), tok, font=f, fill=col, anchor="lm", stroke_width=8 * SS, stroke_fill=(8, 8, 10, 255))
        x += wd[j] + sp

def chip(d, s, a, y=150):
    if a <= 0: return
    f = font(FONT_LAB, 40 * SS)
    tw = d.textlength(s, font=f) / SS
    d.rounded_rectangle([60 * SS, y * SS, (60 + tw + 48) * SS, (y + 74) * SS], radius=16 * SS, fill=(14, 15, 18, int(225 * a)))
    d.text((84 * SS, (y + 37) * SS), s, font=f, fill=(255, 255, 255, int(255 * a)), anchor="lm")

_vig = None
def vignette():
    global _vig
    if _vig is None:
        v = Image.new("L", (OW // 8, OH // 8), 0); px = v.load()
        for yy in range(v.height):
            for xx in range(v.width):
                dx = (xx / v.width - 0.5) * 1.5; dy = (yy / v.height - 0.5) * 1.15
                r = math.sqrt(dx * dx + dy * dy)
                px[xx, yy] = int(255 * (1 - 0.32 * clamp((r - 0.5) / 0.6) ** 1.6))
        _vig = Image.merge("RGB", [v.resize((OW, OH), Image.BICUBIC)] * 3)
    return _vig

def frame(src_png, t, cap, labels, cta_t=None, dst=None):
    img = Image.open(src_png).convert("RGB")
    if img.size != (OW, OH): img = img.resize((OW, OH), Image.LANCZOS)
    from PIL import ImageFilter
    img = img.filter(ImageFilter.UnsharpMask(radius=1.4, percent=55, threshold=2))
    img = ImageChops.multiply(img, vignette()).convert("RGBA")
    ov = Image.new("RGBA", (OW * SS, OH * SS), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    if cta_t is not None and t >= cta_t:
        k = eout((t - cta_t) / 0.45)
        d.rectangle([0, 0, OW * SS, OH * SS], fill=(6, 6, 8, int(170 * k)))
    for a, b, s in labels:
        fa = clamp((t - a) / 0.25) * clamp((b - t) / 0.25)
        chip(d, s, fa)
    draw_caption(d, cap, t)
    img.alpha_composite(ov.resize((OW, OH), Image.LANCZOS))
    out = img.convert("RGB")
    if dst: out.save(dst, quality=94)
    return out
