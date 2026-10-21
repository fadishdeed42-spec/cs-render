"""Emotional voice (Chatterbox, MIT) with the am_liam sound (Kokoro, Apache-2.0) as reference. Owner's pick Oct 12 2026.
Run by .github/workflows/cvoice.yml: python engine/cvoice.py engine/voice/v15c.json
json: {"engine": "chatterbox", "ref": "am_liam", "cfg": 0.2,
       "scenes": [["text", exaggeration, gap_after_s, "end"?], ...]}
  - one scene = one breath / one emotion; exaggeration 0.5 calm .. 0.9 big punchline
  - "end" on the story's last line: generated slower (cfg .35) and its natural tail is kept -> a finished, falling ending
  - spell names as they sound (Quadriga C X, Artoo Deetoo, Maker dow); captions are fixed in btNN.py
Output: <name>.wav (24k), <name>_words.srt (forced-aligned word times), <name>.srt (per scene), <name>.mp3, <name>.log"""
import sys, os, json
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gvoice import align_words, ts, jhash

def kok_ref(path):
    from kokoro import KPipeline
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    t = ("This CEO took the passwords to his grave. Allegedly. Gerald Cotton ran a Canadian crypto exchange. "
         "December twenty eighteen: he dies in India. Then: oops. Only Gerry had the keys.")
    a = np.concatenate([np.asarray(x, np.float32) for _, _, x in pipe(t, voice='am_liam', speed=1.05)])
    sf.write(path, a, 24000)

def trim(w, sr, tail):
    nz = np.where(np.abs(w) > .004)[0]
    if not len(nz): return w
    w = w[max(0, nz[0] - int(.02 * sr)):min(len(w), nz[-1] + int(tail * sr))].copy()
    f = int(.03 * sr); w[-f:] *= np.linspace(1, 0, f)
    return w

_asr = None
def asr_wer(w, sr, text):
    """speech recognition check: share of script words the take got wrong (catches misread words)"""
    global _asr
    import difflib, re
    from faster_whisper import WhisperModel
    if _asr is None: _asr = WhisperModel('base.en', device='cpu', compute_type='int8')
    import librosa
    y = librosa.resample(w, orig_sr=sr, target_sr=16000)
    got = ' '.join(s.text for s in _asr.transcribe(y, language='en', beam_size=1)[0])
    n = lambda t: re.sub(r"[^a-z ]", ' ', t.lower()).split()
    a, b = n(text), n(got)
    return 1 - difflib.SequenceMatcher(None, a, b).ratio(), got

def end_slope(w, sr, text):
    """pitch movement (semitones) over the last word: < 0 falls (finished), > 0 rises (sounds unfinished)"""
    import librosa
    words = text.split(); wt = align_words(w, sr, words); a0, a1 = wt[-1]
    seg = w[int(a0 * sr):int(a1 * sr) + int(.05 * sr)]
    if len(seg) < int(.12 * sr): return 0.0
    f0, vf, _ = librosa.pyin(seg, fmin=65, fmax=320, sr=sr, frame_length=1024)
    v = f0[~np.isnan(f0)]
    if len(v) < 6: return 0.0
    k = max(2, len(v) // 3)
    return float(12 * np.log2(np.median(v[-k:]) / np.median(v[:k])))

def run(path):
    import torch
    from chatterbox.tts import ChatterboxTTS
    cfg = json.load(open(path)); base = path[:-5]; log = ['json ' + jhash(path)]
    ref = os.path.join(os.path.dirname(path), f"ref_{cfg.get('ref', 'am_liam')}.wav")
    if not os.path.exists(ref): kok_ref(ref)
    torch.manual_seed(cfg.get('seed', 7))
    m = ChatterboxTTS.from_pretrained(device='cpu'); sr = m.sr
    parts, words, owner = [np.zeros(int(.12 * sr), np.float32)], [], []
    for i, sc in enumerate(cfg['scenes']):
        t, ex, gap = sc[0], sc[1], sc[2]; end = len(sc) > 3 and sc[3] == 'end'
        n = 6 if end else (3 if ex >= .85 or t.rstrip().endswith('.') and len(t.split()) <= 6 else 2)
        target = 2.6 if end else 3.1; best = None
        for k in range(n):
            w = m.generate(t, audio_prompt_path=ref, exaggeration=ex, cfg_weight=.35 if end else cfg.get('cfg', .2))
            w = trim(w.squeeze().numpy().astype(np.float32), sr, .45 if end else .12)
            try: wer, heard = asr_wer(w, sr, t)
            except Exception as e: wer, heard = 0.0, f'asr failed {e}'
            try: sl = end_slope(w, sr, t)
            except Exception: sl = 0.0
            wps = len(t.split()) / (len(w) / sr)
            want_up = t.rstrip().endswith('?')
            ok_end = (sl > 0) if want_up else (sl < -1.0)
            score = 2.0 * ok_end - 4.0 * wer - .6 * abs(wps - target) - (.08 * max(0, sl) if not want_up else 0)
            log.append(f'   take {k+1}: score {score:+.2f}  wer {wer:.2f}  end {sl:+.1f}st  {wps:.1f}w/s  heard: {heard.strip()[:70]}')
            if best is None or score > best[0]: best = (score, w)
        w = best[1]
        parts += [w, np.zeros(int(gap * sr), np.float32)]
        for x in t.split(): words.append(x); owner.append(i)
        log.append(f'scene {i+1}: {len(w)/sr:.2f}s ex {ex}{" END" if end else ""}')
    audio = np.concatenate(parts); audio = audio / max(1e-6, np.abs(audio).max()) * .9
    wt = align_words(audio, sr, words); log.append(f'{len(audio)/sr:.2f}s, force-aligned {len(words)} words')
    cues = []
    for i, sc in enumerate(cfg['scenes']):
        idx = [k for k in range(len(words)) if owner[k] == i]; cues.append((wt[idx[0]][0], wt[idx[-1]][1], sc[0]))
    sf.write(base + '.wav', audio, sr)
    open(base + '.srt', 'w').write('\n'.join(f'{i+1}\n{ts(a)} --> {ts(b)}\n{t}\n' for i, (a, b, t) in enumerate(cues)))
    open(base + '_words.srt', 'w').write('\n'.join(f'{i+1}\n{ts(a)} --> {ts(b)}\n{t}\n' for i, ((a, b), t) in enumerate(zip(wt, words))))
    os.system(f'ffmpeg -y -loglevel error -i {base}.wav -b:a 160k {base}.mp3')
    lv = cfg.get('liam')                          # re-time to the original Liam take (pace + pauses); no take -> Liam-rate pacing
    d = os.path.dirname(path); import subprocess
    has = bool(lv) and os.path.exists(os.path.join(d, str(lv) + '.wav'))
    if True:
        r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'liampace.py'), base + '.wav', base + '_words.srt',
                            f'{d}/{lv}.wav' if has else '-', f'{d}/{lv}_words.srt' if has else '-', base + 'p'], capture_output=True, text=True)
        log.append('liampace -> ' + base + 'p.wav\n' + r.stdout + r.stderr)
    open(base + '.log', 'w').write('\n'.join(log) + '\n'); print('\n'.join(log))

if __name__ == '__main__':
    for p in sys.argv[1:]: run(p)
