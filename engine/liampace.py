"""Re-time the new emotional voice to the original Liam take: same pace per sentence, Liam-like pauses.
python3 liampace.py new.wav new_words.srt liam.wav liam_words.srt out_base
No Liam take (new stories): pass - - for the two liam args -> Liam-rate pacing (2.85 w/s, final line 2.5), 0.4 s gaps (same as the Liam-paced Shorts 10-14)."""
import sys, re, os, subprocess, tempfile, difflib, numpy as np, soundfile as sf
def srt(p):
    out = []
    for b in open(p).read().strip().split('\n\n'):
        l = b.split('\n'); a, c = l[1].split(' --> ')
        f = lambda s: sum(float(x) * m for x, m in zip(s.replace(',', '.').split(':'), (3600, 60, 1)))
        out.append([f(a), f(c), ' '.join(l[2:])])
    return out
def sents(W):
    out, cur = [], []
    for w in W:
        cur.append(w)
        if re.search(r'[.?!:]["\']?$', w[2]) and not w[2].endswith('...') or w[2].endswith('?'): out.append(cur); cur = []
    if cur: out.append(cur)
    return out
norm = lambda t: re.sub(r'[^a-z ]', '', t.lower().replace('quadriga c x', 'quadrigacx').replace('artoo deetoo', 'aretwo deetwo').replace('cotton', 'cotten').replace('dee-fye', 'defi').replace('deefye', 'defi').replace('maker dow', 'makerdao').replace('the dow', 'the dao'))
def stretch(x, sr, factor):            # factor = new duration / old duration
    if abs(factor - 1) < .02: return x
    a, b = tempfile.mktemp(suffix='.wav'), tempfile.mktemp(suffix='.wav'); sf.write(a, x, sr)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', a, '-af', f'rubberband=tempo={1/factor:.4f}:transients=smooth:formant=preserved', b], check=True)
    y, _ = sf.read(b, dtype='float32'); os.remove(a); os.remove(b); return y
new, nsr = sf.read(sys.argv[1], dtype='float32'); NW = srt(sys.argv[2]); LW = srt(sys.argv[4]) if sys.argv[4] != '-' else []; out = sys.argv[5]
BODY, LAST = 2.85, 2.5   # Liam's measured pace (median 2.8 w/s over 42 sentences, Shorts 10-14); used when there is no Liam take
NOSTRETCH = os.environ.get('LP_STRETCH', '0') != '1'   # owner Oct 13: raw voice sounds better; stretching sounded sharp/echoey -> pauses only, no time-stretch, no pitch shift (LP_STRETCH=1 = old behaviour)
NS, LS = sents(NW), sents(LW)
ltxt = [norm(' '.join(w[2] for w in s)) for s in LS]
lgap = [(LS[i+1][0][0] - LS[i][-1][1]) if i + 1 < len(LS) else .5 for i in range(len(LS))]
pieces, words, log = [np.zeros(int(.15 * nsr), np.float32)], [], []
t = .15
for i, s in enumerate(NS):
    a0 = max(s[0][0] - .03, NS[i-1][-1][1] if i else 0)
    a1 = s[-1][1] + (.35 if i >= len(NS) - 3 else .06) if i == len(NS) - 1 else min(s[-1][1] + (.3 if i == len(NS) - 3 else .06), NS[i+1][0][0])
    seg = new[int(max(0, a0) * nsr):int(a1 * nsr)]
    txt = ' '.join(w[2] for w in s); n = norm(txt)
    m = difflib.get_close_matches(n, ltxt, n=1, cutoff=.72)
    last = i >= len(NS) - 3                      # final story line + CTA
    if m:
        j = ltxt.index(m[0]); L = LS[j]; target = (L[-1][1] - L[0][0]) * len(s) / len(L); gap = min(.4, max(.25, lgap[j])); src = 'liam'
    else:
        target = len(s) / (LAST if last else BODY); gap = .4; src = 'rate'
    if txt.endswith('...'): gap = .45
    if i == len(NS) - 3: gap = .6                # breath after the story's last line, before "Follow Chain Secrets"
    cur = s[-1][1] - s[0][0]; cta = i >= len(NS) - 2
    f = 1.0 if cta else target / cur if last else 1 + .5 * (target / cur - 1); f = 1.0 if cta else min(1.3 if last else 1.15, max(1.0 if last else .95, f))
    if NOSTRETCH: f = 1.0
    if i in (len(NS) - 3, len(NS) - 1) and not NOSTRETCH:          # endings: settle the last word down ~1.5 semitones so it sounds finished
        k = int((s[-1][0] - .02 - a0) * nsr)
        if 0 < k < len(seg):
            a_, b_ = tempfile.mktemp(suffix='.wav'), tempfile.mktemp(suffix='.wav'); sf.write(a_, seg[k:], nsr)
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', a_, '-af', 'rubberband=pitch=0.917:formant=preserved', b_], check=True)
            tl, _ = sf.read(b_, dtype='float32'); fx = int(.015 * nsr)
            tl[:fx] *= np.linspace(0, 1, fx); hd = seg[:k].copy(); hd[-fx:] *= np.linspace(1, 0, fx)
            seg = np.concatenate([hd, tl])
    y = stretch(seg, nsr, f)
    for w in s: words.append([t + (w[0] - a0) * f, t + (w[1] - a0) * f, w[2]])
    pieces += [y, np.zeros(int(gap * nsr), np.float32)]; t += len(y) / nsr + gap
    log.append(f'{src:4s} x{f:4.2f} gap {gap:.2f} | {txt}')
audio = np.concatenate(pieces); audio = audio / np.abs(audio).max() * .9
sf.write(out + '.wav', audio, nsr)
ts = lambda x: f'{int(x//3600):02d}:{int(x%3600//60):02d}:{x%60:06.3f}'.replace('.', ',')
open(out + '_words.srt', 'w').write('\n'.join(f'{k+1}\n{ts(a)} --> {ts(b)}\n{w}\n' for k, (a, b, w) in enumerate(words)))
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out + '.wav', '-b:a', '160k', out + '.mp3'])
print('\n'.join(log)); print(f'total {len(audio)/nsr:.2f}s (was {len(new)/nsr:.2f}s)')
