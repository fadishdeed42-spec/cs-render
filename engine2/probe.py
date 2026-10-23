"""Probe which generators are reachable from the runner. Writes out/probe.json + sample files."""
import os, json, time, urllib.request, urllib.parse, traceback, shutil
R = {}
os.makedirs("out", exist_ok=True)
P = "Stylized 3D animated film still, miniature diorama, a mysterious hacker in a white hoodie and white cap typing on a glowing laptop at night, cinematic lighting, vertical"

def t(name, fn):
    t0 = time.time()
    try: R[name] = {"ok": True, "info": fn(), "s": round(time.time() - t0, 1)}
    except Exception as e: R[name] = {"ok": False, "err": repr(e)[:400], "s": round(time.time() - t0, 1)}

def poll():
    u = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(P) + "?width=720&height=1280&model=flux&nologo=true&seed=7"
    d = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=180).read()
    open("out/pollinations.jpg", "wb").write(d); return len(d)
t("pollinations", poll)

def space(name, api, kw, tok):
    from gradio_client import Client
    c = Client(name, token=tok) if tok else Client(name)
    r = c.predict(api_name=api, **kw)
    return r
def flux(tok):
    def f():
        r = space("mcp-tools/FLUX.1-Krea-dev", "/infer", dict(prompt=P, seed=7, randomize_seed=False, width=576, height=1024, guidance_scale=4.5, num_inference_steps=24), tok)
        p = r[0] if isinstance(r, (list, tuple)) else r
        p = p.get("path") if isinstance(p, dict) else p
        shutil.copy(p, "out/flux_%s.webp" % ("tok" if tok else "anon")); return str(r)[:200]
    return f
t("flux_anon", flux(None))
if os.environ.get("HF_TOKEN"): t("flux_tok", flux(os.environ["HF_TOKEN"]))

def wan():
    from gradio_client import Client, handle_file
    tok = os.environ.get("HF_TOKEN") or None
    c = Client("zerogpu-aoti/wan2-2-fp8da-aoti-faster", token=tok)
    img = "out/flux_anon.webp"
    info = c.view_api(return_format="dict")
    r = c.predict(input_image=handle_file(img), prompt="the hacker types fast, screen flickers, slow camera push in", duration_seconds=3.5, steps=6, api_name="/generate_video")
    v = r[0] if isinstance(r, (list, tuple)) else r
    v = v.get("video") if isinstance(v, dict) and "video" in v else v
    v = v.get("path") if isinstance(v, dict) else v
    shutil.copy(v, "out/wan.mp4"); return str(list(info.get("named_endpoints", {}).keys()))[:300]
t("wan", wan)

def el():
    k = os.environ.get("ELEVENLABS_API_KEY")
    if not k: return "no key"
    req = urllib.request.Request("https://api.elevenlabs.io/v1/user/subscription", headers={"xi-api-key": k})
    s = json.loads(urllib.request.urlopen(req, timeout=30).read())
    req = urllib.request.Request("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": k})
    v = json.loads(urllib.request.urlopen(req, timeout=30).read())
    json.dump(v, open("out/voices.json", "w"))
    return {"tier": s.get("tier"), "used": s.get("character_count"), "limit": s.get("character_limit"), "voices": len(v.get("voices", []))}
t("elevenlabs", el)
json.dump(R, open("out/probe.json", "w"), indent=1)
