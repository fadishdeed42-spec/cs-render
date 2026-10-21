"""Blender toolkit for cross-section Shorts (bpy 5.0, Cycles CPU).
A story file builds sets with these helpers and declares shots keyed to spoken words:

    from blk import *
    W = Words("voice/words.srt")          # exact word times from the voice job
    st = style("snow")                      # lights + world + grade preset
    ...build sets...
    shots([ Shot(W.at("Six hundred"), cam=..., tgt=..., to_cam=..., to_tgt=..., lens=..),  ... ])
"""
import bpy, bmesh, math, os, random, re, json
from mathutils import Vector

FPS = int(os.environ.get("FPS", "30"))
def F(t): return int(round(t * FPS)) + 1

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene; sc.render.fps = FPS
    return sc

from words import Words

# ------------------------------------------------------------------ materials
_M = {}
def mat(name, col=(0.5, 0.5, 0.5), metal=0.0, rough=0.5, emit=None, estr=0.0, trans=0.0, noise=None, alpha=None):
    if name in _M: return _M[name]
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*col, 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    if trans:
        b.inputs["Transmission Weight"].default_value = trans; b.inputs["IOR"].default_value = 1.45
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1); b.inputs["Emission Strength"].default_value = estr
    if alpha is not None:
        b.inputs["Alpha"].default_value = alpha
    if noise:
        tex = nt.nodes.new("ShaderNodeTexNoise"); tex.inputs["Scale"].default_value = noise[0]; tex.inputs["Detail"].default_value = 6
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (*[c * noise[1] for c in col], 1)
        ramp.color_ramp.elements[1].color = (*[min(1, c * noise[2]) for c in col], 1)
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"]); nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
        bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.08
        nt.links.new(tex.outputs["Fac"], bump.inputs["Height"]); nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    _M[name] = m
    return m

def emit_strength_key(m, pairs):
    inp = m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    for t, v in pairs:
        inp.default_value = v; inp.keyframe_insert("default_value", frame=max(1, F(t)))

# ------------------------------------------------------------------ geometry
SETS = {}     # set name -> list of objects (for visibility windows)
_cur = [None]
def in_set(name):
    _cur[0] = name; SETS.setdefault(name, [])
def _reg(o):
    if _cur[0]: SETS[_cur[0]].append(o)
    return o

def _link(o):
    bpy.context.scene.collection.objects.link(o); return _reg(o)

def box(name, x0, x1, y0, y1, z0, z1, m, bevel=0.0, cut=None, parent=None):
    """axis-aligned box. `cut` = material for faces pointing at -Y (the sliced face the camera sees)."""
    me = bpy.data.meshes.new(name); bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x = x0 if v.co.x < 0 else x1; v.co.y = y0 if v.co.y < 0 else y1; v.co.z = z0 if v.co.z < 0 else z1
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); _link(o)
    o.data.materials.append(m if isinstance(m, bpy.types.Material) else _M[m])
    if cut:
        o.data.materials.append(cut if isinstance(cut, bpy.types.Material) else _M[cut])
        for p in o.data.polygons:
            if p.normal.y < -0.9: p.material_index = 1
    if bevel:
        b = o.modifiers.new("bev", "BEVEL"); b.width = bevel; b.segments = 2; b.limit_method = "ANGLE"
    if parent: o.parent = parent
    return o

def bx(name, c, s, m, **k):
    """box by center + size"""
    return box(name, c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2, c[2] - s[2] / 2, c[2] + s[2] / 2, m, **k)

def cyl(name, loc, r, depth, m, axis="Z", verts=32, parent=None, bevel=True, rot=None):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, vertices=verts, location=loc)
    o = bpy.context.object; o.name = name
    if axis == "Y": o.rotation_euler = (math.radians(90), 0, 0)
    elif axis == "X": o.rotation_euler = (0, math.radians(90), 0)
    if rot: o.rotation_euler = rot
    o.data.materials.append(m if isinstance(m, bpy.types.Material) else _M[m])
    if bevel:
        bv = o.modifiers.new("bev", "BEVEL"); bv.width = min(0.02, r * 0.15); bv.segments = 2; bv.limit_method = "ANGLE"
    for p in o.data.polygons: p.use_smooth = True
    if parent: o.parent = parent
    return _reg(o)

def sphere(name, loc, r, m, parent=None, seg=24):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=seg, ring_count=seg // 2)
    o = bpy.context.object; o.name = name; o.data.materials.append(m if isinstance(m, bpy.types.Material) else _M[m])
    for p in o.data.polygons: p.use_smooth = True
    if parent: o.parent = parent
    return _reg(o)

def cone(name, loc, r1, r2, depth, m, verts=24, parent=None):
    bpy.ops.mesh.primitive_cone_add(radius1=r1, radius2=r2, depth=depth, vertices=verts, location=loc)
    o = bpy.context.object; o.name = name; o.data.materials.append(m if isinstance(m, bpy.types.Material) else _M[m])
    for p in o.data.polygons: p.use_smooth = True
    if parent: o.parent = parent
    return _reg(o)

def torus(name, loc, R, r, m, rot=(0, 0, 0), parent=None):
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, location=loc, rotation=rot)
    o = bpy.context.object; o.name = name; o.data.materials.append(m if isinstance(m, bpy.types.Material) else _M[m])
    for p in o.data.polygons: p.use_smooth = True
    if parent: o.parent = parent
    return _reg(o)

def poly_prism(name, pts2d, y0, y1, m, cut=None, parent=None):
    """extrude a polygon drawn in the XZ plane between y0..y1 (good for roofs, terrain slices, gables)"""
    me = bpy.data.meshes.new(name); bm = bmesh.new()
    a = [bm.verts.new((x, y0, z)) for x, z in pts2d]; b = [bm.verts.new((x, y1, z)) for x, z in pts2d]
    bm.faces.new(a[::-1]); bm.faces.new(b)
    n = len(pts2d)
    for i in range(n):
        j = (i + 1) % n; bm.faces.new((a[i], a[j], b[j], b[i]))
    bm.normal_update(); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); _link(o)
    o.data.materials.append(m if isinstance(m, bpy.types.Material) else _M[m])
    if cut:
        o.data.materials.append(cut if isinstance(cut, bpy.types.Material) else _M[cut])
        for p in o.data.polygons:
            if p.normal.y < -0.9: p.material_index = 1
    if parent: o.parent = parent
    return o

def curve(name, pts, m, bevel=0.02, parent=None):
    cd = bpy.data.curves.new(name, "CURVE"); cd.dimensions = "3D"
    sp = cd.splines.new("POLY"); sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts): sp.points[i].co = (*p, 1)
    cd.bevel_depth = bevel; cd.bevel_resolution = 3; cd.use_fill_caps = True
    o = bpy.data.objects.new(name, cd); _link(o); o.data.materials.append(m if isinstance(m, bpy.types.Material) else _M[m])
    if parent: o.parent = parent
    return o

def text3d(name, s, loc, size, m, rot=(math.radians(90), 0, 0), extrude=0.02, align="CENTER", parent=None):
    cd = bpy.data.curves.new(name, "FONT"); cd.body = s; cd.size = size; cd.extrude = extrude
    cd.align_x = align; cd.align_y = "CENTER"
    o = bpy.data.objects.new(name, cd); o.location = loc; o.rotation_euler = rot; _link(o)
    o.data.materials.append(m if isinstance(m, bpy.types.Material) else _M[m])
    if parent: o.parent = parent
    return o

def empty(name, loc=(0, 0, 0)):
    o = bpy.data.objects.new(name, None); o.location = loc; _link(o); return o

def dup(o, name, loc, parent=None):
    n = o.copy(); n.name = name; n.location = loc
    bpy.context.scene.collection.objects.link(n); _reg(n)
    if parent: n.parent = parent
    return n

# ------------------------------------------------------------------ animation
def key(o, path, t, value, interp=None):
    setattr(o, path, value) if path not in ("location", "rotation_euler", "scale") else None
    if path == "location": o.location = value
    elif path == "rotation_euler": o.rotation_euler = value
    elif path == "scale": o.scale = value
    o.keyframe_insert(data_path=path, frame=F(t) if not isinstance(t, tuple) else t[0])

def keys(o, path, pairs):
    for t, v in pairs: key(o, path, t, v)

def vis(o, pairs):
    """pairs [(t, visible_bool)] -> hide_render/hide_viewport keys (recursive into children)"""
    for t, v in pairs:
        o.hide_render = not v; o.hide_viewport = not v
        o.keyframe_insert("hide_render", frame=max(1, F(t))); o.keyframe_insert("hide_viewport", frame=max(1, F(t)))
    for c in o.children: vis(c, pairs)

def linear(o):
    ad = o.animation_data
    if ad and ad.action:
        for fc in getattr(ad.action, "fcurves", []):
            for kp in fc.keyframe_points: kp.interpolation = "LINEAR"

def spin(o, axis, t0, t1, turns):
    """constant rotation (gears, fans, turbines)"""
    i = "XYZ".index(axis)
    r = list(o.rotation_euler); r0 = list(r); r[i] += turns * math.tau
    o.rotation_euler = r0; o.keyframe_insert("rotation_euler", frame=max(1, F(t0)))
    o.rotation_euler = r; o.keyframe_insert("rotation_euler", frame=F(t1))
    linear(o)

def set_windows(name, windows):
    """show a whole set only inside its shot windows [(t0,t1),...] (merged in whole frames) -> faster renders, no stray objects"""
    fr = sorted((F(a), max(F(a), F(b) - 1)) for a, b in windows)
    merged = []
    for a, b in fr:
        if merged and a <= merged[-1][1] + 1: merged[-1][1] = max(merged[-1][1], b)
        else: merged.append([a, b])
    keysf = {1: False}
    for a, b in merged:
        keysf[a] = True; keysf[b + 1] = False
    objs = SETS.get(name, [])
    for o in objs:
        if o.parent is not None and o.parent in objs: continue
        _vis_frames(o, keysf)

def _vis_frames(o, keysf):
    for f in sorted(keysf):
        v = keysf[f]; o.hide_render = not v; o.hide_viewport = not v
        o.keyframe_insert("hide_render", frame=f); o.keyframe_insert("hide_viewport", frame=f)
    for c in o.children: _vis_frames(c, keysf)

# ------------------------------------------------------------------ lights / world / styles
def area(name, loc, rot_deg, energy, size, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="AREA", location=loc); L = bpy.context.object; L.name = name
    L.rotation_euler = [math.radians(a) for a in rot_deg]; L.data.energy = energy; L.data.size = size; L.data.color = color
    return _reg(L)

def point(name, loc, energy, color=(1, 1, 1), radius=0.2):
    bpy.ops.object.light_add(type="POINT", location=loc); L = bpy.context.object; L.name = name
    L.data.energy = energy; L.data.color = color; L.data.shadow_soft_size = radius; return _reg(L)

def spot(name, loc, rot_deg, energy, angle=60, color=(1, 1, 1), blend=0.5, radius=0.3):
    bpy.ops.object.light_add(type="SPOT", location=loc); L = bpy.context.object; L.name = name
    L.rotation_euler = [math.radians(a) for a in rot_deg]; L.data.energy = energy; L.data.spot_size = math.radians(angle)
    L.data.spot_blend = blend; L.data.color = color; L.data.shadow_soft_size = radius; return _reg(L)

def world(col, strength=1.0):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (*col, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = strength

def softbox(loc, scale, rot_deg, color=(0.8, 0.85, 0.9), strength=0.9):
    """emissive card hidden from the camera: gives metal and glass something to reflect"""
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc); sb = bpy.context.object
    sb.scale = scale; sb.rotation_euler = [math.radians(a) for a in rot_deg]
    sb.data.materials.append(mat(f"softbox{len(_M)}", (0, 0, 0), 0, 1, emit=color, estr=strength))
    sb.visible_camera = False; sb.visible_shadow = False
    return sb

# ------------------------------------------------------------------ camera + shots
class Shot:
    def __init__(self, t, cam, tgt, to_cam=None, to_tgt=None, lens=35, to_lens=None, fstop=4.0, ease=True, note=""):
        self.t, self.cam, self.tgt = t, cam, tgt
        self.to_cam, self.to_tgt = to_cam or cam, to_tgt or tgt
        self.lens, self.to_lens, self.fstop, self.ease, self.note = lens, to_lens or lens, fstop, ease, note

def shots(lst, end):
    """cut between shots exactly at their start times; each shot moves cam/target/lens from start to end values"""
    sc = bpy.context.scene
    tgt = empty("cam_target")
    bpy.ops.object.camera_add(location=(0, -10, 2)); cam = bpy.context.object; sc.camera = cam
    tc = cam.constraints.new("TRACK_TO"); tc.target = tgt; tc.track_axis = "TRACK_NEGATIVE_Z"; tc.up_axis = "UP_Y"
    cam.data.dof.use_dof = True; cam.data.dof.focus_object = tgt; cam.data.sensor_fit = "VERTICAL"; cam.data.clip_start = 0.02
    lst = sorted(lst, key=lambda s: s.t)
    for i, s in enumerate(lst):
        t0 = s.t; t1 = lst[i + 1].t if i + 1 < len(lst) else end
        f0, f1 = F(t0), max(F(t0), F(t1) - 1)
        for f, c, g, l in ((f0, s.cam, s.tgt, s.lens), (f1, s.to_cam, s.to_tgt, s.to_lens)):
            cam.location = c; cam.keyframe_insert("location", frame=f)
            tgt.location = g; tgt.keyframe_insert("location", frame=f)
            cam.data.lens = l; cam.data.keyframe_insert("lens", frame=f)
            cam.data.dof.aperture_fstop = s.fstop; cam.data.dof.keyframe_insert("aperture_fstop", frame=f)
    return cam, tgt, [(s.t, (lst[i + 1].t if i + 1 < len(lst) else end), s.note) for i, s in enumerate(lst)]

# ------------------------------------------------------------------ reusable props
def mannequin(name, loc, mtl, pose="stand", face=0.0, scale=1.0, hood=False):
    """anonymous figure (no face). pose: stand | sit | walk. face = rotation about Z in degrees."""
    root = empty(name, loc); root.rotation_euler = (0, 0, math.radians(face)); root.scale = (scale,) * 3
    if pose == "sit":
        box(name + "_th", -0.05, 0.42, -0.12, 0.12, 0.42, 0.56, mtl, 0.05, parent=root)
        box(name + "_sh", 0.32, 0.44, -0.11, 0.11, 0.0, 0.48, mtl, 0.04, parent=root)
        box(name + "_to", -0.14, 0.12, -0.18, 0.18, 0.52, 1.12, mtl, 0.08, parent=root)
        hz = 1.27
    else:
        for s in (-1, 1):
            box(f"{name}_leg{s}", -0.07, 0.07, s * 0.1 - 0.06, s * 0.1 + 0.06, 0, 0.88, mtl, 0.04, parent=root)
        box(name + "_to", -0.12, 0.12, -0.2, 0.2, 0.86, 1.48, mtl, 0.08, parent=root)
        for s in (-1, 1):
            box(f"{name}_arm{s}", -0.05, 0.05, s * 0.25 - 0.05, s * 0.25 + 0.05, 0.9, 1.46, mtl, 0.04, parent=root)
        hz = 1.64
    h = sphere(name + "_hd", (0, 0, hz), 0.13, mtl, parent=root)
    if hood:
        sphere(name + "_hood", (0.02, 0, hz + 0.02), 0.155, mtl, parent=root)
    return root

def laptop(name, loc, rot_z, screen_mat, frame_mat, parent=None):
    r = empty(name, loc); r.rotation_euler = (0, 0, math.radians(rot_z))
    if parent: r.parent = parent
    box(name + "_b", -0.22, 0.22, -0.16, 0.16, 0, 0.018, frame_mat, 0.004, parent=r)
    box(name + "_l", -0.22, 0.22, 0.15, 0.17, 0.018, 0.3, frame_mat, 0.004, parent=r)
    box(name + "_s", -0.2, 0.2, 0.145, 0.15, 0.035, 0.285, screen_mat, parent=r)
    return r

def mining_rig(name, loc, body, fan, led, parent=None):
    """a GPU/ASIC miner: box chassis with two fans and a status LED"""
    r = empty(name, loc)
    if parent: r.parent = parent
    box(name + "_c", -0.2, 0.2, -0.14, 0.14, 0, 0.17, body, 0.01, parent=r)
    for s in (-0.1, 0.1):
        cyl(name + f"_f{s}", (s, -0.145, 0.085), 0.065, 0.012, fan, axis="Y", verts=16, parent=r, bevel=False)
    cyl(name + "_led", (0.16, -0.146, 0.14), 0.012, 0.01, led, axis="Y", verts=10, parent=r, bevel=False)
    return r

def render_settings(samples=6, rx=720, ry=1280):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"
    sc.cycles.samples = samples; sc.cycles.use_denoising = True; sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 1; sc.cycles.glossy_bounces = 2
    sc.cycles.transmission_bounces = 3; sc.cycles.transparent_max_bounces = 4
    sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False; sc.cycles.blur_glossy = 1.0
    sc.render.use_persistent_data = True
    sc.render.resolution_x, sc.render.resolution_y = rx, ry
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = float(os.environ.get("EXPOSURE", "0.6"))
    sc.render.image_settings.file_format = "PNG"

def room_shell(name, x0, x1, y0, y1, z0, z1, t, m, cut, inner=None, ceiling=True, sides=True):
    """a room sliced open at the front (y0): back wall, floor, ceiling and side walls; the cut edges face the camera"""
    inner = inner or m
    box(name + "_back", x0, x1, y1 - t, y1, z0, z1, m, cut=cut)
    box(name + "_floor", x0, x1, y0, y1, z0, z0 + t, m, cut=cut)
    if ceiling: box(name + "_ceil", x0, x1, y0, y1, z1 - t, z1, m, cut=cut)
    if sides:
        box(name + "_l", x0, x0 + t, y0, y1, z0, z1, m, cut=cut); box(name + "_r", x1 - t, x1, y0, y1, z0, z1, m, cut=cut)
    box(name + "_bin", x0 + t, x1 - t, y1 - t - 0.01, y1 - t, z0 + t, z1 - t, inner)

def skycard(loc, w, h, top, bottom, strength=1.0, rot_deg=(90, 0, 0)):
    """big emissive gradient card far behind the set (sky / backdrop), invisible to lights"""
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc); p = bpy.context.object
    p.scale = (w, h, 1); p.rotation_euler = [math.radians(a) for a in rot_deg]
    m = bpy.data.materials.new(f"sky{len(_M)}"); m.use_nodes = True; nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
    tc = nt.nodes.new("ShaderNodeTexCoord"); sep = nt.nodes.new("ShaderNodeSeparateXYZ"); ramp = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0]); nt.links.new(sep.outputs["Y"], ramp.inputs["Fac"])
    ramp.color_ramp.elements[0].color = (*bottom, 1); ramp.color_ramp.elements[1].color = (*top, 1)
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"]); em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    p.data.materials.append(m); p.visible_shadow = False; p.visible_diffuse = False; p.visible_glossy = True
    return _reg(p)
