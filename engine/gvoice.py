"""Gemini TTS voice + SRT, run by .github/workflows/gvoice.yml (Google API is reachable from Actions, not the sandbox).
Input  engine/voice/<name>.json:
  {"voice": "Fenrir", "persona": "...", "lines": [["text", "style synced to the on-screen moment"], ["quote", "style", "Algenib"], ...]}
  - text is read verbatim; vocal tags allowed: <laugh> <sigh> <short pause> <long pause>
  - 3rd item = other voice for that line (character quotes; max 2 voices per video)
  - ONE request per video: free tier is 10 requests/day for gemini-3.8-flash-tts (falls back to flash-lite, then 3.1 preview)
  - write each style from what is ON SCREEN at that moment so the emotion matches the animation
Output engine/voice/<name>.wav (24k mono), <name>_words.srt (EXACT per-word times, forced alignment - use this for tbuild),
       <name>.srt (one cue per line), <name>.mp3, <name>.log"""
import json, os, re, sys, time, base64, wave, io, difflib, urllib.request, urllib.error
import numpy as np, soundfile as sf
KEY = os.environ.get('GEMINI_API_KEY', ''); API = 'https://generativelanguage.googleapis.com/v1beta'
MODELS = ['gemini-3.8-flash-tts', 'gemini-3.1-flash-tts-preview']
CONSIST = ("ONE consistent character and voice from start to finish: emotions are shifts in energy and attitude, never a different "
           "voice. Keep pitch in a natural range, never cartoonish or over-acted. Sentences flow into each other like real speech. ")
PACE = ("Natural, confident storyteller pace, about 150 words per minute - never rushed. Numbers and names are said clearly. "
        "Let punchlines breathe: a small beat before and after each joke. Big contrast between lines: follow each line's delivery exactly.")
TAG = re.compile(r'<[^>]+>')

def post(body):
    rq = urllib.request.Request(f'{API}/interactions', json.dumps(body).encode(), {'Content-Type': 'application/json', 'x-goog-api-key': KEY})
    for k in range(4):
        try:
            with urllib.request.urlopen(rq, timeout=300) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read()[:400].decode(errors='replace')
            if e.code in (500, 503) and k < 3: time.sleep(20 * (k + 1)); continue
            raise RuntimeError(f'HTTP {e.code}: {msg}')

def find_audio(o):
    if isinstance(o, dict):
        for k in ('data', 'audio'):
            if isinstance(o.get(k), str) and len(o[k]) > 2000: return o[k]
        for v in o.values():
            r = find_audio(v)
            if r: return r
    elif isinstance(o, list):
        for v in o:
            r = find_audio(v)
            if r: return r

def to_audio(raw):
    if raw[:4] == b'RIFF': a, sr = sf.read(io.BytesIO(raw), dtype='float32')
    else: a, sr = np.frombuffer(raw, '<i2').astype('float32') / 32768, 24000
    return (a if a.ndim == 1 else a.mean(1)), sr

def tts_all(cfg, persona):
    """ONE request per video (free tier = 10 requests/day per model). Up to 2 voices via multi-speaker."""
    main = cfg['voice']; voices = [main] + sorted({l[2] for l in cfg['lines'] if len(l) > 2 and l[2] != main})
    assert len(voices) <= 2, 'max 2 voices per video'
    spk = {v: ('Narrator' if v == main else 'Character') for v in voices}
    def part(l):
        other = len(l) > 2 and l[2] != main
        md = {'type': 'speech_metadata', 'style': l[1] if other else f'{persona} {CONSIST}{PACE}\nMOOD FOR THIS PART: {l[1]}'}
        if len(voices) > 1: md['speaker'] = spk[l[2] if len(l) > 2 else main]
        return {'type': 'text', 'text': l[0] + ' ', 'annotations': [md]}
    src = cfg.get('_parts') or cfg['lines']
    sc = [{'speaker': spk[v], 'voice': v} for v in voices] if len(voices) > 1 else [{'voice': main}]
    errs = []
    for m in ('gemini-3.8-flash-tts', 'gemini-3.8-flash-lite-tts'):
        try:
            r = post({'model': m, 'input': [{'type': 'user_input', 'content': [part(l) for l in src]}],
                      'response_format': {'type': 'audio'}, 'generation_config': {'speech_config': sc}})
            return (*to_audio(base64.b64decode(find_audio(r))), m)
        except Exception as e: errs.append(f'{m}: {e}'); print('  ', errs[-1][:300], flush=True)
    # last resort: older preview model, classic API, director's notes (single voice)
    m = 'gemini-3.1-flash-tts-preview'
    notes = '\n'.join(f'- "{TAG.sub("", l[0])}": {l[1]}' for l in cfg['lines'])
    prompt = (f"### AUDIO PROFILE\n{persona} {PACE}\n\n### DIRECTOR'S NOTES (per line)\n{notes}\n\n"
              f"### TRANSCRIPT (read only this, verbatim)\n" + ' '.join(l[0] for l in cfg['lines']))
    rq = urllib.request.Request(f'{API}/models/{m}:generateContent', json.dumps({'contents': [{'parts': [{'text': prompt}]}],
         'generationConfig': {'responseModalities': ['AUDIO'], 'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': main}}}}}).encode(),
         {'Content-Type': 'application/json', 'x-goog-api-key': KEY})
    with urllib.request.urlopen(rq, timeout=300) as r:
        return (*to_audio(base64.b64decode(find_audio(json.loads(r.read())))), m)

def align_words(audio, sr, words):
    """Exact forced alignment of the known script words (torchaudio MMS_FA). Returns [(start,end)] per word."""
    import torch, torchaudio
    bundle = torchaudio.pipelines.MMS_FA
    model, tok, aligner = bundle.get_model(), bundle.get_tokenizer(), bundle.get_aligner()
    wav = torchaudio.functional.resample(torch.tensor(audio)[None], sr, bundle.sample_rate)
    keys = [re.sub(r"[^a-z']", '', w.lower()) for w in words]
    use = [i for i, k in enumerate(keys) if k]
    with torch.inference_mode():
        em, _ = model(wav)
        spans = aligner(em[0], tok([keys[i] for i in use]))
    ratio = wav.shape[1] / em.shape[1] / bundle.sample_rate
    out = [None] * len(words)
    for i, sp in zip(use, spans): out[i] = (sp[0].start * ratio, sp[-1].end * ratio)
    for i in range(len(out)):  # tokens with no letters: glue to previous word
        if out[i] is None: p = out[i - 1] if i else (0.0, 0.0); out[i] = (p[1], p[1])
    return out

def ts(x): h, r = divmod(x, 3600); m, s = divmod(r, 60); return f'{int(h):02d}:{int(m):02d}:{s:06.3f}'.replace('.', ',')

def jhash(path):  # hash of the job, ignoring the (temporary) download link
    c = json.load(open(path)); c.pop('url', None)
    import hashlib; return hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]

def run_url(path, cfg):
    """vidIQ/ElevenLabs take: download cfg['url'] (one mp3 holding several scripts back to back), force-align, split.
    json: {"source": "url", "url": "https://...mp3", "parts": [["v14", "script text"], ["v16", "script text"]]}"""
    import hashlib, subprocess, tempfile
    base = path[:-5]; log = ['json ' + jhash(path)]
    mp3 = base + '_full.mp3'
    if cfg.get('url'): urllib.request.urlretrieve(cfg['url'], mp3)
    # the signed link holds a temporary AWS token: drop it from the repo, keep the mp3 instead
    c2 = json.load(open(path)); c2['url'] = ''; json.dump(c2, open(path, 'w'), indent=1)
    wav = tempfile.mktemp(suffix='.wav')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp3, '-ac', '1', '-ar', '24000', wav], check=True)
    audio, sr = sf.read(wav, dtype='float32'); log.append(f'downloaded {len(audio)/sr:.2f}s')
    words, owner = [], []
    for i, (_, t) in enumerate(cfg['parts']):
        for w in TAG.sub(' ', t).split(): words.append(w); owner.append(i)
    wt = align_words(audio, sr, words)
    n = len(cfg['parts']); first = [owner.index(i) for i in range(n)]
    last = [len(owner) - 1 - owner[::-1].index(i) for i in range(n)]
    for i, (name, _) in enumerate(cfg['parts']):
        a0 = 0.0 if i == 0 else (wt[last[i - 1]][1] + wt[first[i]][0]) / 2
        a1 = len(audio) / sr if i == n - 1 else (wt[last[i]][1] + wt[first[i + 1]][0]) / 2
        a0 = max(0.0, min(a0, wt[first[i]][0] - .25))
        a1 = min(len(audio) / sr, max(a1, wt[last[i]][1] + .35))
        seg = audio[int(a0 * sr):int(a1 * sr)].copy(); f = int(.01 * sr)
        seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        ob = os.path.join(os.path.dirname(path), name)
        sf.write(ob + '.wav', seg, sr)
        ws = [(wt[k][0] - a0, wt[k][1] - a0, words[k]) for k in range(len(words)) if owner[k] == i]
        open(ob + '_words.srt', 'w').write('\n'.join(f'{j+1}\n{ts(a)} --> {ts(b)}\n{t}\n' for j, (a, b, t) in enumerate(ws)))
        os.system(f'ffmpeg -y -loglevel error -i {ob}.wav -b:a 160k {ob}.mp3')
        log.append(f'{name}: {a0:.2f}-{a1:.2f}s, {len(ws)} words')
    open(base + '.log', 'w').write('\n'.join(log) + '\n'); print('\n'.join(log))

def run_kokoro(path, cfg):
    """Free local voice (Kokoro-82M, Apache-2.0). Owner's pick (Oct 12 2026): am_liam.
    json: {"engine": "kokoro", "voice": "am_liam", "speed": 1.05,
           "lines": [["text", speed_multiplier, pause_after_seconds], ...]}
    Emotion = per-line speed + pauses + punctuation (! ? ... ,): slower + pause before a punchline, faster for hype."""
    from kokoro import KPipeline
    base = path[:-5]; log = ['json ' + jhash(path)]
    v = cfg.get('voice', 'am_liam'); sp0 = cfg.get('speed', 1.05)
    pipe = KPipeline(lang_code='b' if v.startswith('b') else 'a', repo_id='hexgrad/Kokoro-82M')
    sr = 24000; parts, words, owner, cues_t = [], [], [], []
    for i, ln in enumerate(cfg['lines']):
        t = ln[0]; spm = ln[1] if len(ln) > 1 else 1.0; gap = ln[2] if len(ln) > 2 else .3
        a = np.concatenate([np.asarray(x, np.float32) for _, _, x in pipe(t, voice=v, speed=sp0 * spm)])
        nz = np.where(np.abs(a) > .01)[0]                       # trim the model's own edge silence
        if len(nz): a = a[max(0, nz[0] - int(.03 * sr)):nz[-1] + int(.06 * sr)]
        parts += [a, np.zeros(int(gap * sr), np.float32)]
        for w in t.split(): words.append(w); owner.append(i)
    audio = np.concatenate([np.zeros(int(.15 * sr), np.float32)] + parts)
    audio = audio / max(1e-6, np.abs(audio).max()) * .9
    log.append(f'kokoro {v} speed {sp0} {len(audio)/sr:.2f}s')
    wt = align_words(audio, sr, words); log.append(f'force-aligned {len(words)} words')
    cues = []
    for i, ln in enumerate(cfg['lines']):
        idx = [k for k in range(len(words)) if owner[k] == i]
        cues.append((wt[idx[0]][0], wt[idx[-1]][1], ln[0]))
    sf.write(base + '.wav', audio, sr)
    open(base + '.srt', 'w').write('\n'.join(f'{i+1}\n{ts(a)} --> {ts(b)}\n{t}\n' for i, (a, b, t) in enumerate(cues)))
    open(base + '_words.srt', 'w').write('\n'.join(f'{i+1}\n{ts(a)} --> {ts(b)}\n{t}\n' for i, ((a, b), t) in enumerate(zip(wt, words))))
    os.system(f'ffmpeg -y -loglevel error -i {base}.wav -b:a 160k {base}.mp3')
    open(base + '.log', 'w').write('\n'.join(log) + '\n'); print('\n'.join(log))

def run(path, wm):
    import hashlib
    cfg = json.load(open(path)); base = path[:-5]
    log = ['json ' + jhash(path)]
    if 'scenes' in cfg:  # preferred: one mood per scene, sentences flow naturally inside it
        cfg['lines'] = [[t, mood] for mood, ts_ in cfg['scenes'] for t in ts_]
        cfg['_parts'] = [[' '.join(ts_), mood] for mood, ts_ in cfg['scenes']]
    persona = cfg.get('persona', 'A charismatic, funny YouTube Shorts storyteller, a crypto detective telling a wild true story. Punchy, confident, high energy, natural American accent, real emotions and comedic timing.')
    audio, sr, m = tts_all(cfg, persona); log.append(f'{cfg["voice"]} {m} {len(audio)/sr:.2f}s')
    lines = [ln[:2] for ln in cfg['lines']]
    words, owner = [], []
    for i, (t, _) in enumerate(lines):
        for w in TAG.sub(' ', t).split(): words.append(w); owner.append(i)
    wt = align_words(audio, sr, words); log.append(f'force-aligned {len(words)} words')
    cues = []
    for i, (t, _) in enumerate(lines):
        idx = [k for k in range(len(words)) if owner[k] == i]
        cues.append((wt[idx[0]][0], wt[idx[-1]][1], ' '.join(TAG.sub(' ', t).split())))
    wcues = [(a0, b0, w) for (a0, b0), w in zip(wt, words)]
    sf.write(base + '.wav', audio, sr)
    open(base + '.srt', 'w').write('\n'.join(f'{i+1}\n{ts(a)} --> {ts(b)}\n{t}\n' for i, (a, b, t) in enumerate(cues)))
    open(base + '_words.srt', 'w').write('\n'.join(f'{i+1}\n{ts(a)} --> {ts(b)}\n{t}\n' for i, (a, b, t) in enumerate(wcues)))
    os.system(f'ffmpeg -y -loglevel error -i {base}.wav -af loudnorm=I=-16 -b:a 160k {base}.mp3')
    open(base + '.log', 'w').write('\n'.join(log) + '\n'); print('\n'.join(log))

if __name__ == '__main__':
    wm = None
    import hashlib
    def stale(f):  # regenerate when the json changed since the last take
        b = f'engine/voice/{f[:-5]}'
        sha = jhash(f'engine/voice/{f}')
        return not os.path.exists(b + '.wav') or not os.path.exists(b + '.log') or f'json {sha}' not in open(b + '.log').read()
    todo = sys.argv[1:] or [f'engine/voice/{f}' for f in sorted(os.listdir('engine/voice')) if f.endswith('.json') and json.load(open(f'engine/voice/{f}')).get('engine') != 'chatterbox' and stale(f)]
    for p in todo:
        print('==', p, flush=True); c = json.load(open(p))
        run_kokoro(p, c) if c.get('engine') == 'kokoro' else run_url(p, c) if c.get('source') == 'url' else run(p, wm)
