"""images.py SHOTS.json OUTDIR [variants] : generate shot images on free HF Spaces (round-robin across parts).
Shot prompt = style + img (HERO replaced by the hero description). Shots with 'text' use Qwen-Image first."""
import os, sys, json, time, shutil, random, traceback
from gradio_client import Client
S = json.load(open(sys.argv[1])); out = sys.argv[2]; nv = int(sys.argv[3]) if len(sys.argv) > 3 else 2
os.makedirs(out, exist_ok=True)
part, nparts = int(os.environ.get("PART", 0)), int(os.environ.get("NPARTS", 1))
tok = os.environ.get("HF_TOKEN") or None
CL = {}
def client(name):
    if name not in CL:
        CL[name] = Client(name, token=tok) if tok else Client(name)
    return CL[name]
def endpoint(c):
    eps = list(c.view_api(return_format="dict")["named_endpoints"].keys()); return eps[0]
def krea(p, seed):
    c = client("mcp-tools/FLUX.1-Krea-dev")
    r = c.predict(prompt=p, seed=seed, randomize_seed=False, width=720, height=1280, guidance_scale=4.5, num_inference_steps=28, api_name="/infer")
    return r[0]
def qwen(p, seed):
    c = client("mcp-tools/Qwen-Image")
    r = c.predict(prompt=p, aspect_ratio="9:16", seed=seed, randomize_seed=False, guidance_scale=4, num_inference_steps=20,
                  negative_prompt="watermark, blurry, low resolution, deformed, extra fingers", api_name=endpoint(c))
    return r[0] if isinstance(r, (list, tuple)) else r
def path(r):
    if isinstance(r, dict): r = r.get("path") or r.get("url")
    return r
log = []
only = set(os.environ.get("ONLY", "").split()) - {""}
jobs = [(k, sh, v) for k, sh in enumerate(S["shots"]) if not only or sh["id"] in only for v in range(nv)]
for j, (k, sh, v) in enumerate(jobs):
    if j % nparts != part: continue
    p = (S["style"] + ", " + sh["img"].replace("HERO", S["hero"]))
    order = [qwen, krea] if sh.get("text") else [krea, qwen]
    if True:
        seed = 1000 + k * 17 + v * 101
        done = False
        for gen in order * 3:
            try:
                f = path(gen(p, seed)); ext = os.path.splitext(f)[1] or ".png"
                shutil.copy(f, os.path.join(out, f"{sh['id']}_{v}_{gen.__name__}{ext}")); done = True
                log.append([sh["id"], v, gen.__name__, "ok"]); break
            except Exception as e:
                log.append([sh["id"], v, gen.__name__, repr(e)[:200]]); time.sleep(20)
        if not done: log.append([sh["id"], v, "FAILED"])
json.dump(log, open(os.path.join(out, f"images_log_p{part}.json"), "w"), indent=0)
