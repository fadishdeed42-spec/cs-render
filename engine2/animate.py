"""animate.py PLAN.json IMGDIR OUTDIR : animate chosen shot images with Wan 2.2 image-to-video (free HF Space).
PLAN.json: {"shots": [{"id","img": filename, "move": prompt, "dur": seconds}], "negative": optional}
Round-robin across PART/NPARTS. Writes OUTDIR/<id>.mp4 + anim_log_pK.json."""
import os, sys, json, time, shutil
from gradio_client import Client, handle_file
P = json.load(open(sys.argv[1])); src, out = sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
part, nparts = int(os.environ.get("PART", 0)), int(os.environ.get("NPARTS", 1))
tok = os.environ.get("HF_TOKEN") or None
SPACES = ["zerogpu-aoti/wan2-2-fp8da-aoti-faster"]
cl = {}
def client(n):
    if n not in cl: cl[n] = Client(n, token=tok) if tok else Client(n)
    return cl[n]
log = []
neg = P.get("negative")
for k, sh in enumerate(P["shots"]):
    if k % nparts != part: continue
    ok = False
    for attempt in range(4):
        for sp in SPACES:
            try:
                kw = dict(input_image=handle_file(os.path.join(src, sh["img"])), prompt=sh["move"] + ", smooth natural motion, cinematic, stable camera, high quality animation",
                          duration_seconds=min(5.0, max(2.0, sh.get("dur", 4.0))), steps=sh.get("steps", 8), seed=sh.get("seed", 42), randomize_seed=False, api_name="/generate_video")
                if neg: kw["negative_prompt"] = neg
                r = client(sp).predict(**kw)
                v = r[0] if isinstance(r, (list, tuple)) else r
                if isinstance(v, dict): v = v.get("video", v); v = v.get("path") if isinstance(v, dict) else v
                shutil.copy(v, os.path.join(out, sh["id"] + ".mp4")); ok = True; log.append([sh["id"], sp, "ok"]); break
            except Exception as e:
                log.append([sh["id"], sp, repr(e)[:240]]); time.sleep(30 * (attempt + 1))
        if ok: break
    if not ok: log.append([sh["id"], "FAILED"])
json.dump(log, open(os.path.join(out, f"anim_log_p{part}.json"), "w"), indent=0)
