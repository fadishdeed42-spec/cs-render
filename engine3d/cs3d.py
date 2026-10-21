"""cs3d - small Blender toolkit for 3D cross-section explainer Shorts.

A story file does:
    from cs3d import *
    setup(style="arctic")              # reads $CS_TIMELINE (beats json) for timing
    ... build sets with box()/cyl()/person() ...
    cam.shot("beat_a", "beat_b", from_pos, to_pos, look_a, look_b, lens=40)
    run()                              # handles: range A B OUTDIR | stills t1 t2 ... OUTDIR | save
Everything is original procedural geometry. Times may be given as seconds or as beat names.
"""
import bpy, bmesh, math, json, os, sys, random
from mathutils import Vector

FPS = 24
B = {}          # beat name -> seconds
TOTAL = 0.0
_mats = {}
_style = {}

# ----------------------------------------------------------------------------- time
def T(x):
    """beat name / seconds / ('beat', offset) -> seconds"""
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, tuple): return T(x[0]) + x[1]
    return B[x]

def F(x): return int(round(T(x) * FPS)) + 1

# ----------------------------------------------------------------------------- styles
STYLES = {
    # world colour, key light colour/energy, fill colour/energy, rim colour/energy, look
    "arctic":  dict(world=(0.07, 0.09, 0.13), key=((1.0, 0.93, 0.84), 1300), fill=((0.55, 0.7, 1.0), 380), rim=((0.75, 0.85, 1.0), 650), look="AgX - Medium High Contrast"),
    "lab":     dict(world=(0.55, 0.57, 0.6), key=((1.0, 0.98, 0.95), 1100), fill=((0.9, 0.95, 1.0), 500), rim=((1.0, 1.0, 1.0), 400), look="AgX - Base Contrast"),
    "noir":    dict(world=(0.012, 0.016, 0.024), key=((1.0, 0.62, 0.28), 1500), fill=((0.25, 0.6, 0.7), 300), rim=((0.4, 0.8, 0.9), 700), look="AgX - High Contrast"),
    "tropic":  dict(world=(0.03, 0.035, 0.07), key=((1.0, 0.78, 0.5), 1200), fill=((0.45, 0.4, 0.9), 380), rim=((1.0, 0.5, 0.35), 500), look="AgX - Medium High Contrast"),
    "soviet":  dict(world=(0.05, 0.055, 0.055), key=((0.92, 0.95, 0.9), 1500), fill=((0.55, 0.65, 0.6), 260), rim=((1.0, 0.35, 0.25), 350), look="AgX - High Contrast"),
}

def setup(style="arctic", timeline=None):
    global B, TOTAL, _style
    bpy.ops.wm.read_factory_settings(use_empty=True)
    tl = json.load(open(timeline or os.environ.get("CS_TIMELINE", "timeline.json")))
    B.update(tl["beats"]); TOTAL = tl["total"]
    # stories do `from cs3d import *` before setup(): hand them the real length too
    import inspect
    inspect.currentframe().f_back.f_globals["TOTAL"] = TOTAL
    sc = bpy.context.scene
    sc.render.fps = FPS; sc.frame_start = 1; sc.frame_end = F(TOTAL)
    _style = STYLES[style]
    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (*_style["world"], 1)
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"
    sc.cycles.samples = int(os.environ.get("CS_SAMPLES", "8")); sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"; sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 2; sc.cycles.glossy_bounces = 2
    sc.cycles.transmission_bounces = 3; sc.cycles.transparent_max_bounces = 6
    sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False; sc.cycles.blur_glossy = 1.0
    sc.render.use_persistent_data = True
    sc.render.resolution_x = int(os.environ.get("CS_RX", "720")); sc.render.resolution_y = int(os.environ.get("CS_RY", "1280"))
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = _style["look"]
    sc.render.image_settings.file_format = "PNG"
    return sc

# ----------------------------------------------------------------------------- materials
def mat(name, col=(0.5, 0.5, 0.5), metal=0.0, rough=0.5, emit=None, estr=0.0, trans=0.0, noise=None, alpha=1.0, spec=0.5):
    if name in _mats: return _mats[name]
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if trans:
        b.inputs["Transmission Weight"].default_value = trans; b.inputs["IOR"].default_value = 1.45
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1); b.inputs["Emission Strength"].default_value = estr
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
    if noise:
        tex = nt.nodes.new("ShaderNodeTexNoise"); tex.inputs["Scale"].default_value = noise[0]; tex.inputs["Detail"].default_value = 6
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*[c * noise[1] for c in col], 1)
        ramp.color_ramp.elements[1].color = (*[min(1, c * noise[2]) for c in col], 1)
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"]); nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
        bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.06
        nt.links.new(tex.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    _mats[name] = m
    return m

def M(name): return _mats[name]

def emit_keys(m, pairs):
    """animate a material's emission strength: [(time, value), ...]"""
    inp = m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    for t, v in pairs:
        inp.default_value = v; inp.keyframe_insert("default_value", frame=max(1, F(t)))

def blink(m, period=1.0, lo=0.4, hi=6.0, phase=0.0):
    """repeating emission blink for LEDs"""
    inp = m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    t = -phase
    while t < TOTAL + period:
        if t >= 0:
            inp.default_value = hi; inp.keyframe_insert("default_value", frame=F(t))
            inp.default_value = lo; inp.keyframe_insert("default_value", frame=F(t + period * 0.5))
        t += period
    _set_interp_data(m.node_tree, "CONSTANT")

def _set_interp_data(idblock, interp):
    ad = idblock.animation_data
    if not ad or not ad.action: return
    for fc in _fcurves(ad.action):
        for kp in fc.keyframe_points: kp.interpolation = interp

def _fcurves(action):
    if hasattr(action, "fcurves") and action.fcurves is not None:
        try: return list(action.fcurves)
        except Exception: pass
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for cb in strip.channelbags: out += list(cb.fcurves)
    return out

# ----------------------------------------------------------------------------- geometry
coll = None
def _link(o):
    bpy.context.scene.collection.objects.link(o); return o

def box(name, x0, x1, y0, y1, z0, z1, m, bevel=0.0, cut=None, parent=None, smooth=False):
    """axis-aligned box. cut=<material> paints the face that points at the camera (-Y) as a cross-section face."""
    me = bpy.data.meshes.new(name); bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x = x0 if v.co.x < 0 else x1; v.co.y = y0 if v.co.y < 0 else y1; v.co.z = z0 if v.co.z < 0 else z1
    bm.to_mesh(me); bm.free()
    o = _link(bpy.data.objects.new(name, me))
    o.data.materials.append(m if not isinstance(m, str) else M(m))
    if cut:
        o.data.materials.append(cut if not isinstance(cut, str) else M(cut))
        for p in o.data.polygons:
            if p.normal.y < -0.9: p.material_index = 1
    if bevel:
        bv = o.modifiers.new("bev", "BEVEL"); bv.width = bevel; bv.segments = 2; bv.limit_method = "ANGLE"
    if parent: o.parent = parent
    return o

def boxc(name, c, s, m, **kw):
    """box by centre (x,y,z) and size (sx,sy,sz)"""
    return box(name, c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2, c[2] - s[2] / 2, c[2] + s[2] / 2, m, **kw)

def _prim_finish(o, name, m, parent, smooth=True, bevel=0.0):
    o.name = name
    o.data.materials.append(m if not isinstance(m, str) else M(m))
    if smooth:
        for p in o.data.polygons: p.use_smooth = True
    if bevel:
        bv = o.modifiers.new("bev", "BEVEL"); bv.width = bevel; bv.segments = 2; bv.limit_method = "ANGLE"
    if parent: o.parent = parent
    return o

def cyl(name, loc, r, depth, m, axis="Z", verts=32, parent=None, bevel=0.0):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, vertices=verts, location=loc)
    o = bpy.context.object
    if axis == "Y": o.rotation_euler = (math.radians(90), 0, 0)
    elif axis == "X": o.rotation_euler = (0, math.radians(90), 0)
    o = _prim_finish(o, name, m, parent, bevel=bevel)
    # flat caps should not be smooth-shaded
    for p in o.data.polygons:
        if len(p.vertices) > 4: p.use_smooth = False
    return o

def cone(name, loc, r1, r2, depth, m, verts=24, parent=None):
    bpy.ops.mesh.primitive_cone_add(radius1=r1, radius2=r2, depth=depth, vertices=verts, location=loc)
    return _prim_finish(bpy.context.object, name, m, parent)

def sphere(name, loc, r, m, parent=None, seg=24):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=seg, ring_count=seg // 2)
    return _prim_finish(bpy.context.object, name, m, parent)

def torus(name, loc, R, r, m, axis="Y", parent=None):
    rot = (math.radians(90), 0, 0) if axis == "Y" else ((0, math.radians(90), 0) if axis == "X" else (0, 0, 0))
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, location=loc, rotation=rot, major_segments=48, minor_segments=12)
    return _prim_finish(bpy.context.object, name, m, parent)

def plane(name, loc, sx, sy, m, rot=(0, 0, 0), parent=None):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc, rotation=rot)
    o = bpy.context.object; o.scale = (sx, sy, 1)
    return _prim_finish(o, name, m, parent, smooth=False)

def curve(name, pts, m, bevel=0.02, parent=None):
    cd = bpy.data.curves.new(name, "CURVE"); cd.dimensions = "3D"
    sp = cd.splines.new("POLY"); sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts): sp.points[i].co = (*p, 1)
    cd.bevel_depth = bevel; cd.bevel_resolution = 3; cd.use_fill_caps = True
    o = _link(bpy.data.objects.new(name, cd)); o.data.materials.append(m if not isinstance(m, str) else M(m))
    if parent: o.parent = parent
    return o

def sag(a, b, drop, n=12):
    """points of a hanging cable between a and b"""
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n,
             a[2] + (b[2] - a[2]) * k / n - drop * math.sin(math.pi * k / n)) for k in range(n + 1)]

def empty(name, loc=(0, 0, 0), parent=None):
    o = _link(bpy.data.objects.new(name, None)); o.location = loc
    if parent: o.parent = parent
    return o

def scatter_mesh(name, pts, r, m):
    """many small spheres merged into one mesh (snow, dust, sparks)"""
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    for p in pts:
        g = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r)
        bmesh.ops.translate(bm, verts=g["verts"], vec=Vector(p))
    bm.to_mesh(me); bm.free()
    o = _link(bpy.data.objects.new(name, me)); o.data.materials.append(m if not isinstance(m, str) else M(m))
    return o

def snowfall(name, center, size, n=400, r=0.012, speed=0.25, m="snow"):
    """two stacked layers of flakes scrolling down forever (cheap, seamless)"""
    rng = random.Random(len(name))
    cx, cy, cz = center; sx, sy, sz = size
    for k in range(2):
        pts = [(cx + rng.uniform(-sx, sx), cy + rng.uniform(-sy, sy), rng.uniform(0, sz)) for _ in range(n)]
        o = scatter_mesh(f"{name}{k}", pts, r, m)
        z0 = cz + k * sz
        o.location = (0, 0, z0); o.keyframe_insert("location", frame=1)
        o.location = (0, 0, z0 - speed * TOTAL); o.keyframe_insert("location", frame=F(TOTAL))
        _set_interp(o, "LINEAR")

# ----------------------------------------------------------------------------- people (anonymous mannequins)
def person(name, loc, rot=0.0, m="figure", pose="stand", scale=1.0, parent=None):
    """faceless mannequin, ~1.75 units tall at scale 1, facing -Y at rot 0. returns root; parts in root['parts']"""
    root = empty(name, loc, parent); root.rotation_euler = (0, 0, rot); root.scale = (scale,) * 3
    s = 1.0
    parts = {}
    if pose == "sit":
        parts["thigh_l"] = box(name + "_tl", -0.18, -0.02, -0.45, 0.05, 0.42, 0.56, m, 0.03, parent=root)
        parts["thigh_r"] = box(name + "_tr", 0.02, 0.18, -0.45, 0.05, 0.42, 0.56, m, 0.03, parent=root)
        parts["shin_l"] = box(name + "_sl", -0.17, -0.04, -0.5, -0.38, 0.0, 0.44, m, 0.03, parent=root)
        parts["shin_r"] = box(name + "_sr", 0.04, 0.17, -0.5, -0.38, 0.0, 0.44, m, 0.03, parent=root)
        hip = 0.5
    else:
        parts["leg_l"] = box(name + "_ll", -0.17, -0.03, -0.08, 0.08, 0.0, 0.86, m, 0.03, parent=root)
        parts["leg_r"] = box(name + "_lr", 0.03, 0.17, -0.08, 0.08, 0.0, 0.86, m, 0.03, parent=root)
        hip = 0.86
    parts["torso"] = box(name + "_t", -0.22, 0.22, -0.12, 0.12, hip, hip + 0.62, m, 0.06, parent=root)
    parts["head"] = sphere(name + "_h", (0, 0, hip + 0.78), 0.13, m, parent=root)
    arm_l = empty(name + "_al", (-0.27, 0, hip + 0.56), root); arm_r = empty(name + "_ar", (0.27, 0, hip + 0.56), root)
    box(name + "_alm", -0.06, 0.06, -0.06, 0.06, -0.6, 0.0, m, 0.03, parent=arm_l)
    box(name + "_arm", -0.06, 0.06, -0.06, 0.06, -0.6, 0.0, m, 0.03, parent=arm_r)
    parts["arm_l"], parts["arm_r"] = arm_l, arm_r
    root["hip"] = hip
    root.__dict__ if False else None
    PEOPLE[name] = parts
    return root
PEOPLE = {}

def walk(root, t0, t1, p0, p1, steps_per_s=1.8, bob=0.04):
    """move a mannequin from p0 to p1 with swinging arms/legs"""
    parts = PEOPLE[root.name]
    root.location = p0; root.keyframe_insert("location", frame=F(t0))
    root.location = p1; root.keyframe_insert("location", frame=F(t1))
    n = max(1, int((T(t1) - T(t0)) * steps_per_s * 2))
    for i in range(n + 1):
        t = T(t0) + (T(t1) - T(t0)) * i / n
        a = math.radians(25) * (1 if i % 2 else -1) if i not in (0, n) else 0
        for k, sgn in (("arm_l", 1), ("arm_r", -1)):
            parts[k].rotation_euler = (a * sgn, 0, 0); parts[k].keyframe_insert("rotation_euler", frame=F(t))
        for k, sgn in (("leg_l", -1), ("leg_r", 1)):
            if k in parts:
                parts[k].rotation_euler = (a * sgn * 0.8, 0, 0); parts[k].keyframe_insert("rotation_euler", frame=F(t))

def arm(root, side, t, angle_deg):
    a = PEOPLE[root.name]["arm_" + side]
    a.rotation_euler = (math.radians(angle_deg), 0, 0); a.keyframe_insert("rotation_euler", frame=F(t))

# ----------------------------------------------------------------------------- animation helpers
def key(o, path, t, value, interp=None):
    setattr(o, path, value); o.keyframe_insert(path, frame=F(t))
    if interp: _set_interp(o, interp)

def keys(o, path, pairs, interp=None):
    for t, v in pairs: key(o, path, t, v)
    if interp: _set_interp(o, interp)

def _set_interp(o, interp):
    _set_interp_data(o, interp)

def show(o, pairs):
    """visibility over time for an object tree: [(time, True/False), ...] (hard switches)"""
    for t, v in pairs:
        for x in [o] + list(o.children_recursive):
            x.hide_render = not v; x.keyframe_insert("hide_render", frame=max(1, F(t)))
            x.hide_viewport = not v; x.keyframe_insert("hide_viewport", frame=max(1, F(t)))

def only_between(o, t0, t1):
    show(o, [(0, False), (t0, True), (t1, False)])

def spin(o, axis, deg_per_s, t0=0, t1=None):
    t1 = TOTAL if t1 is None else t1
    i = "XYZ".index(axis)
    r = list(o.rotation_euler); o.keyframe_insert("rotation_euler", frame=F(t0))
    r[i] += math.radians(deg_per_s * (T(t1) - T(t0))); o.rotation_euler = r; o.keyframe_insert("rotation_euler", frame=F(t1))
    _set_interp(o, "LINEAR")

def crush(o, t, dur=0.25, to=0.08):
    """flatten an object onto the ground (scale z) like a steamroller"""
    o.keyframe_insert("scale", frame=F(t))
    s = list(o.scale); s[2] *= to; s[0] *= 1.08; s[1] *= 1.08
    o.scale = s; o.keyframe_insert("scale", frame=F((t, dur)))

# ----------------------------------------------------------------------------- lights
def area(name, loc, rot, energy, size, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="AREA", location=loc); L = bpy.context.object; L.name = name
    L.rotation_euler = [math.radians(a) for a in rot]; L.data.energy = energy; L.data.size = size; L.data.color = color
    return L

def spot(name, loc, rot, energy, size_deg=60, color=(1, 1, 1), soft=0.3):
    bpy.ops.object.light_add(type="SPOT", location=loc); L = bpy.context.object; L.name = name
    L.rotation_euler = [math.radians(a) for a in rot]; L.data.energy = energy; L.data.spot_size = math.radians(size_deg)
    L.data.spot_blend = 0.5; L.data.shadow_soft_size = soft; L.data.color = color
    return L

def point(name, loc, energy, color=(1, 1, 1), soft=0.2):
    bpy.ops.object.light_add(type="POINT", location=loc); L = bpy.context.object; L.name = name
    L.data.energy = energy; L.data.color = color; L.data.shadow_soft_size = soft
    return L

def light_keys(L, pairs):
    for t, v in pairs:
        L.data.energy = v; L.data.keyframe_insert("energy", frame=max(1, F(t)))

def rig(center=(0, 0, 0), spread=6.0, height=6.0, scale=1.0):
    """style three-point rig around a set"""
    cx, cy, cz = center
    k = _style["key"]; f = _style["fill"]; r = _style["rim"]
    return [area(f"key_{cx:.0f}", (cx - spread * 0.7, cy - spread, cz + height), (55, 0, -35), k[1] * scale, spread * 0.6, k[0]),
            area(f"fill_{cx:.0f}", (cx + spread, cy - spread * 0.6, cz + height * 0.5), (65, 0, 50), f[1] * scale, spread * 0.8, f[0]),
            area(f"rim_{cx:.0f}", (cx + spread * 0.4, cy + spread, cz + height * 0.8), (60, 0, 160), r[1] * scale, spread * 0.5, r[0])]

def dim(lights, pairs):
    """scale a rig's energies over time: pairs [(time, factor)]"""
    for L in lights:
        base = L.data.energy
        for t, fct in pairs:
            L.data.energy = base * fct; L.data.keyframe_insert("energy", frame=max(1, F(t)))
        L.data.energy = base

def softbox(center=(0, -14, 5), size=(18, 7), strength=0.8, color=(0.8, 0.85, 0.9)):
    """invisible-to-camera emitter that gives metals something to reflect"""
    bpy.ops.mesh.primitive_plane_add(size=1, location=center); sb = bpy.context.object
    sb.scale = (size[0], size[1], 1); sb.rotation_euler = (math.radians(90), 0, 0)
    sb.data.materials.append(mat("softbox_" + str(center), (0, 0, 0), 0, 1, emit=color, estr=strength))
    sb.visible_camera = False; sb.visible_shadow = False
    return sb

# ----------------------------------------------------------------------------- camera
class Cam:
    def __init__(self):
        self.t = empty("cam_target", (0, 0, 1))
        bpy.ops.object.camera_add(location=(0, -6, 2)); self.c = bpy.context.object
        bpy.context.scene.camera = self.c
        tc = self.c.constraints.new("TRACK_TO"); tc.target = self.t; tc.track_axis = "TRACK_NEGATIVE_Z"; tc.up_axis = "UP_Y"
        self.c.data.dof.use_dof = True; self.c.data.dof.focus_object = self.t; self.c.data.dof.aperture_fstop = 4
        self.c.data.sensor_fit = "VERTICAL"; self.c.data.clip_end = 400
        self.shots = []

    def shot(self, t0, t1, c0, c1, g0, g1=None, lens=40, lens1=None, fstop=4.0, ease=True):
        """one shot from t0 to t1 (beats or seconds): camera c0->c1 looking at g0->g1. Shots cut hard."""
        g1 = g1 or g0
        f0, f1 = F(t0), F(t1) - 1
        self.shots.append((T(t0), T(t1)))
        for f, c, g, l in ((f0, c0, g0, lens), (f1, c1, g1, lens1 or lens)):
            self.c.location = c; self.c.keyframe_insert("location", frame=f)
            self.t.location = g; self.t.keyframe_insert("location", frame=f)
            self.c.data.lens = l; self.c.data.keyframe_insert("lens", frame=f)
            self.c.data.dof.aperture_fstop = fstop; self.c.data.dof.keyframe_insert("aperture_fstop", frame=f)
        if not ease:
            for idb in (self.c, self.t):
                ad = idb.animation_data
                for fc in _fcurves(ad.action):
                    for kp in fc.keyframe_points:
                        if f0 <= kp.co.x <= f1: kp.interpolation = "LINEAR"

# ----------------------------------------------------------------------------- run
def run():
    sc = bpy.context.scene
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    if not a: a = ["save"]
    if a[0] == "range":
        sc.frame_start, sc.frame_end = int(a[1]), min(int(a[2]), F(TOTAL))
        sc.render.filepath = os.path.join(a[3], "")
        bpy.ops.render.render(animation=True)
    elif a[0] == "stills":
        out = a[-1]; os.makedirs(out, exist_ok=True)
        for t in a[1:-1]:
            sc.frame_set(F(float(t)))
            sc.render.filepath = os.path.join(out, f"s_{float(t):06.2f}.png")
            bpy.ops.render.render(write_still=True)
    elif a[0] == "frames":
        print("FRAMES", F(TOTAL))
    elif a[0] == "save":
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath("scene.blend"))

def set_interp(idblock, interp):
    """public: set keyframe interpolation on an object, material node tree or other ID ("CONSTANT", "LINEAR", "BEZIER")"""
    _set_interp_data(idblock, interp)
