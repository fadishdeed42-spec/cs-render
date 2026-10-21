"""Render a frame range of a story with Blender (bpy).
python kit/runstory.py JOBDIR START END OUTDIR [--stills t1,t2,...]"""
import sys, os, json, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
job, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.environ["STORY_DIR"] = job; os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("story", f"{job}/story.py"); S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
import bpy, blk
plan = S.build()
json.dump({"shots": plan, "total": S.TOTAL, "fps": S.FPS}, open(f"{out}/plan.json", "w"), indent=1)
blk.render_settings(int(os.environ.get("SAMPLES", "6")), int(os.environ.get("RX", "720")), int(os.environ.get("RY", "1280")))
sc = bpy.context.scene
stills = [x for x in sys.argv[5:] if x.startswith("--stills=")]
if stills:
    for t in stills[0].split("=", 1)[1].split(","):
        sc.frame_set(blk.F(float(t))); sc.render.filepath = f"{out}/still_{float(t):05.1f}.png"
        bpy.ops.render.render(write_still=True)
else:
    sc.frame_start, sc.frame_end = a, b
    sc.render.filepath = f"{out}/"
    bpy.ops.render.render(animation=True)
