#!/usr/bin/env python3
"""
OFFLINE WebGPU-Inspector capture slicer — reads the frame-1244 JSON capture and
emits small, self-contained, human/agent-readable per-resource-type slices.

THIS IS NOT A PROBE. It launches NO browser, NO GPU, NO dev server. It only reads
the captured JSON (the GROUND TRUTH) and reshapes it. Safe to run anytime.

    python3 docs/perf-runs/2026-06-17-webgpu-inspector/slice.py

Outputs into ./slices/ next to this script:
  timeline.md / timeline.json   the reconstructed frame: submits -> passes -> dispatches/draws
  passes_compute.json           the 56 compute passes (pipeline, shader entry, dims, bindings)
  passes_render.json            the 18 render passes (attachments, draws, viewports, res)
  buffers.json                  all buffers (label, size, usage) + which passes bind them
  textures.json                 all textures + views (size, format, usage, vram) + pass usage
  pipelines.json                render + compute pipelines (label, shader refs, layout)
  shaders/<id>.wgsl + .json     each unique WGSL module + a manifest
  bindgroups.json               bind groups + layouts + pipeline layouts
  submits.json                  the 76 submit boundaries (encoder -> passes)
  renderbundles.json            the render bundles
  summary.md / statistics.json  header stats, canvas resolution, validation errors
"""
import json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
# args: [capture-filename] [output-subdir]   (defaults = the frame-1244 baseline)
CAP  = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "webgpu_capture_frame_1244.json")
OUT  = os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "slices")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "shaders"), exist_ok=True)

# ---- load DOC 0 only (the rest are base64 buffer/texture payloads) ----------
raw = open(CAP).read()
doc0, _ = json.JSONDecoder().raw_decode(raw, 0)
del raw
OBJS = doc0["objects"]                  # str(id) -> object
CMDS = doc0["commands"]                 # ordered API call list

def obj(i):
    if i is None: return None
    return OBJS.get(str(i))

BUF_USAGE = [(0x0001,"MAP_READ"),(0x0002,"MAP_WRITE"),(0x0004,"COPY_SRC"),
             (0x0008,"COPY_DST"),(0x0010,"INDEX"),(0x0020,"VERTEX"),
             (0x0040,"UNIFORM"),(0x0080,"STORAGE"),(0x0100,"INDIRECT"),(0x0200,"QUERY_RESOLVE")]
TEX_USAGE = [(0x01,"COPY_SRC"),(0x02,"COPY_DST"),(0x04,"TEXTURE_BINDING"),
             (0x08,"STORAGE_BINDING"),(0x10,"RENDER_ATTACHMENT")]
def usage(n, table):
    n = n or 0
    return "|".join(name for bit,name in table if n & bit) or str(n)

def ref(a):
    """Resolve an {'__id':N,...} reference to a compact dict; pass scalars through."""
    if isinstance(a, dict) and "__id" in a:
        i = a["__id"]; o = obj(i)
        lbl = a.get("__label") or (o.get("label") if isinstance(o,dict) else None)
        return {"id": i, "class": a.get("__class"), "label": lbl}
    return a

# ---- resolve a pipeline (compute or render) to shader + entry ---------------
def shader_of(modref):
    if not isinstance(modref, dict): return None
    o = obj(modref.get("__id"))
    if not isinstance(o, dict): return {"id": modref.get("__id")}
    return {"id": o["id"], "label": o.get("label"),
            "vtx": o.get("hasVertexEntries"), "frag": o.get("hasFragmentEntries"),
            "comp": o.get("hasComputeEntries")}

def pipeline_info(pid):
    o = obj(pid)
    if not isinstance(o, dict): return {"id": pid}
    d = o.get("descriptor", {})
    info = {"id": pid, "type": o["type"], "label": o.get("label") or d.get("label")}
    if o["type"] == "ComputePipeline":
        c = d.get("compute", {})
        info["entryPoint"] = c.get("entryPoint")
        info["shader"] = shader_of(c.get("module"))
    else:
        v = d.get("vertex", {}); fr = d.get("fragment", {})
        info["vertexEntry"] = v.get("entryPoint"); info["vertexShader"] = shader_of(v.get("module"))
        info["fragmentEntry"] = fr.get("entryPoint"); info["fragmentShader"] = shader_of(fr.get("module"))
        info["primitive"] = d.get("primitive")
        ds = d.get("depthStencil")
        info["depthStencil"] = {k: ds.get(k) for k in ("format","depthWriteEnabled","depthCompare")} if isinstance(ds,dict) else None
    return info

# ---- resolve a bind group to its bound resources ----------------------------
def bindgroup_info(bid):
    o = obj(bid)
    if not isinstance(o, dict): return {"id": bid}
    d = o.get("descriptor", {})
    ents = []
    for e in d.get("entries", []) or []:
        r = e.get("resource")
        res = None
        if isinstance(r, dict):
            if "__id" in r: res = ref(r)
            elif isinstance(r.get("buffer"), dict):
                res = ref(r["buffer"]); res = {**res, "offset": r.get("offset"), "size": r.get("size")} if isinstance(res,dict) else res
            else: res = ref(r.get("buffer", r))
        ents.append({"binding": e.get("binding"), "resource": res})
    return {"id": bid, "label": o.get("label"), "entries": ents}

# ---- resolve a texture view -> texture (res/format) -------------------------
def texview_info(vid):
    o = obj(vid)
    if not isinstance(o, dict): return {"id": vid}
    tref = o.get("texture")
    t = obj(tref.get("__id")) if isinstance(tref, dict) else None
    out = {"viewId": vid, "viewLabel": o.get("label")}
    if isinstance(t, dict):
        out.update({"texId": t["id"], "texLabel": t.get("label"),
                    "w": t.get("width"), "h": t.get("height"),
                    "format": t.get("format"), "mips": t.get("mipLevelCount"),
                    "layers": t.get("depthOrArrayLayers"),
                    "usage": usage(t.get("descriptor",{}).get("usage"), TEX_USAGE),
                    "vramBytes": t.get("gpuSize")})
    return out

# =============================================================================
# walk the command stream, reconstruct passes + submits
# =============================================================================
transient = {}            # "_N" -> {"kind":..., "encoder":...}
submits = []              # list of {encoder, passes:[...], index}
cur_encoder = None
cur_pass = None
passes = []               # flat list of all passes in order
buf_pass = collections.defaultdict(set)   # bufferId -> set(passLabel)
tex_pass = collections.defaultdict(set)   # texId    -> set(passLabel)

def bg_resources(bg):
    """yield (id, class) for every resource in a resolved bindgroup."""
    for e in bg.get("entries", []):
        r = e.get("resource")
        if isinstance(r, dict) and "id" in r:
            yield r["id"], r.get("class")

for c in CMDS:
    m = c.get("method"); o = c.get("object"); res = c.get("result"); args = c.get("args") or []
    idx = c.get("index")

    if m == "createCommandEncoder":
        cur_encoder = {"encoderRes": res, "index": idx, "passes": []}
        submits.append(cur_encoder)
        transient[res] = {"kind": "encoder"}

    elif m == "beginComputePass":
        cur_pass = {"kind": "compute", "encoderRes": o, "beginIdx": idx,
                    "pipeline": None, "binds": [], "dispatches": []}
        passes.append(cur_pass)
        if cur_encoder is not None: cur_encoder["passes"].append(cur_pass)
        transient[res] = {"kind": "computepass", "pass": cur_pass}

    elif m == "beginRenderPass":
        d = args[0] if args else {}
        col = []
        for a in (d.get("colorAttachments") or []):
            v = a.get("view") if isinstance(a, dict) else None
            vi = texview_info(v.get("__id")) if isinstance(v, dict) and "__id" in v else {"viewRaw": v}
            col.append({**vi, "loadOp": a.get("loadOp") if isinstance(a,dict) else None,
                        "storeOp": a.get("storeOp") if isinstance(a,dict) else None})
        ds = d.get("depthStencilAttachment") if isinstance(d, dict) else None
        dsv = None
        if isinstance(ds, dict) and isinstance(ds.get("view"), dict):
            dsv = {**texview_info(ds["view"].get("__id")),
                   "depthLoadOp": ds.get("depthLoadOp"), "depthStoreOp": ds.get("depthStoreOp")}
        cur_pass = {"kind": "render", "encoderRes": o, "beginIdx": idx,
                    "label": d.get("label") if isinstance(d,dict) else None,
                    "colorAttachments": col, "depthAttachment": dsv,
                    "viewport": None, "pipeline": None, "binds": [], "draws": []}
        passes.append(cur_pass)
        if cur_encoder is not None: cur_encoder["passes"].append(cur_pass)
        transient[res] = {"kind": "renderpass", "pass": cur_pass}

    elif m == "setPipeline":
        t = transient.get(o)
        if t and "pass" in t and args:
            pid = args[0].get("__id") if isinstance(args[0], dict) else None
            t["pass"]["pipeline"] = pipeline_info(pid)

    elif m == "setBindGroup":
        t = transient.get(o)
        if t and "pass" in t and len(args) >= 2:
            slot = args[0]; bid = args[1].get("__id") if isinstance(args[1], dict) else None
            bg = bindgroup_info(bid)
            t["pass"]["binds"].append({"slot": slot, **bg})
            lbl = t["pass"].get("label") or (t["pass"].get("pipeline") or {}).get("label") or f"pass@{t['pass']['beginIdx']}"
            for rid, rcls in bg_resources(bg):
                if rcls == "Buffer": buf_pass[rid].add(lbl)
                elif rcls == "TextureView":
                    tv = obj(rid)
                    if isinstance(tv, dict) and isinstance(tv.get("texture"), dict):
                        tex_pass[tv["texture"]["__id"]].add(lbl)

    elif m == "setViewport":
        t = transient.get(o)
        if t and "pass" in t: t["pass"]["viewport"] = args[:4]

    elif m in ("dispatchWorkgroups", "dispatchWorkgroupsIndirect"):
        t = transient.get(o)
        if t and "pass" in t:
            if m == "dispatchWorkgroups":
                t["pass"]["dispatches"].append({"dims": args[:3], "indirect": False})
            else:
                ind = ref(args[0]) if args else None
                t["pass"]["dispatches"].append({"indirect": True, "argBuffer": ind, "offset": args[1] if len(args)>1 else 0})

    elif m in ("draw", "drawIndexed", "drawIndirect", "drawIndexedIndirect"):
        t = transient.get(o)
        if t and "pass" in t:
            if m in ("drawIndirect", "drawIndexedIndirect"):
                t["pass"]["draws"].append({"method": m, "argBuffer": ref(args[0]) if args else None, "offset": args[1] if len(args)>1 else 0})
            else:
                t["pass"]["draws"].append({"method": m, "args": args})

# =============================================================================
# emit slices
# =============================================================================
def dump(name, data):
    json.dump(data, open(os.path.join(OUT, name), "w"), indent=1)

comp = [p for p in passes if p["kind"] == "compute"]
rend = [p for p in passes if p["kind"] == "render"]
dump("passes_compute.json", comp)
dump("passes_render.json", rend)

# buffers
buffers = []
for o in OBJS.values():
    if o.get("type") != "Buffer": continue
    d = o.get("descriptor", {})
    buffers.append({"id": o["id"], "label": o.get("label"), "size": o.get("size") or d.get("size"),
                    "usage": usage(d.get("usage"), BUF_USAGE),
                    "boundInPasses": sorted(buf_pass.get(o["id"], []))})
buffers.sort(key=lambda b: -(b["size"] or 0))
dump("buffers.json", buffers)

# textures + views
textures = []
for o in OBJS.values():
    if o.get("type") != "Texture": continue
    d = o.get("descriptor", {})
    textures.append({"id": o["id"], "label": o.get("label"), "w": o.get("width"), "h": o.get("height"),
                     "format": o.get("format"), "mips": o.get("mipLevelCount"),
                     "layers": o.get("depthOrArrayLayers"),
                     "usage": usage(d.get("usage"), TEX_USAGE), "vramBytes": o.get("gpuSize"),
                     "usedInPasses": sorted(tex_pass.get(o["id"], []))})
textures.sort(key=lambda t: -((t["w"] or 0)*(t["h"] or 0)))
views = [{"id": o["id"], "label": o.get("label"),
          "texId": o.get("texture",{}).get("__id") if isinstance(o.get("texture"),dict) else None}
         for o in OBJS.values() if o.get("type") == "TextureView"]
dump("textures.json", {"textures": textures, "viewCount": len(views), "views": views})

# pipelines
pipes = [pipeline_info(o["id"]) for o in OBJS.values() if o.get("type") in ("ComputePipeline","RenderPipeline")]
dump("pipelines.json", pipes)

# bind groups / layouts
bgs = [bindgroup_info(o["id"]) for o in OBJS.values() if o.get("type") == "BindGroup"]
bgls = []
for o in OBJS.values():
    if o.get("type") != "BindGroupLayout": continue
    d = o.get("descriptor", {})
    bgls.append({"id": o["id"], "entryCount": len(d.get("entries", []) or []), "entries": d.get("entries")})
pls = []
for o in OBJS.values():
    if o.get("type") != "PipelineLayout": continue
    d = o.get("descriptor", {})
    pls.append({"id": o["id"], "bindGroupLayouts": [ref(x) for x in (d.get("bindGroupLayouts") or [])]})
dump("bindgroups.json", {"bindGroups": bgs, "bindGroupLayouts": bgls, "pipelineLayouts": pls})

# render bundles
rbs = []
for o in OBJS.values():
    if o.get("type") != "RenderBundle": continue
    rbs.append({"id": o["id"], "descriptor": o.get("descriptor")})
dump("renderbundles.json", rbs)

# shaders: manifest + each WGSL to a file (dedup by code)
manifest = []
seen_code = {}
for o in OBJS.values():
    if o.get("type") != "ShaderModule": continue
    d = o.get("descriptor", {}); code = d.get("code", "") or ""
    h = hash(code)
    fn = f"{o['id']}.wgsl"
    if h not in seen_code:
        seen_code[h] = fn
        open(os.path.join(OUT, "shaders", fn), "w").write(code)
    manifest.append({"id": o["id"], "label": o.get("label") or d.get("label"),
                     "vtx": o.get("hasVertexEntries"), "frag": o.get("hasFragmentEntries"),
                     "comp": o.get("hasComputeEntries"), "codeLen": len(code),
                     "file": f"shaders/{seen_code[h]}", "firstLine": code.split(chr(10),1)[0][:80]})
dump("shaders/_manifest.json", manifest)

# submits
sub_out = []
for s in submits:
    sub_out.append({"encoderIndex": s["index"], "encoderRes": s["encoderRes"],
                    "passes": [{"kind": p["kind"], "beginIdx": p["beginIdx"],
                                "label": p.get("label") or (p.get("pipeline") or {}).get("label"),
                                "n_dispatch": len(p.get("dispatches", [])),
                                "n_draw": len(p.get("draws", []))} for p in s["passes"]]})
dump("submits.json", sub_out)
dump("statistics.json", {"statistics": doc0.get("statistics"),
                          "validationErrors": doc0.get("validationErrors"),
                          "frame": doc0.get("frame"),
                          "canvas": next((o.get("descriptor") for o in OBJS.values() if o.get("type")=="CanvasContext"), None),
                          "adapter": next((o.get("descriptor") for o in OBJS.values() if o.get("type")=="Adapter"), None)})

# =============================================================================
# timeline.md — the human/agent-readable master reconstruction
# =============================================================================
L = []
st = doc0.get("statistics", {})
canvas = next((o.get("descriptor") for o in OBJS.values() if o.get("type")=="CanvasContext"), {})
L.append(f"# Frame {doc0.get('frame')} reconstruction — canvas {canvas.get('width')}x{canvas.get('height')} ({canvas.get('format')})")
L.append(f"\n{st.get('dispatch',0)+0} dispatchWorkgroups + {st.get('drawIndirect',0)} drawIndirect-class · "
         f"{len(comp)} compute passes · {len(rend)} render passes · {len(submits)} submits\n")
def res_str(att):
    if att.get("w"): return f"{att['w']}x{att['h']} {att.get('format')}"
    return str(att.get("viewRaw") or att.get("viewLabel") or "?")
n = 0
for s in submits:
    if not s["passes"]:
        continue
    n += 1
    L.append(f"\n## submit {n} (encoder cmd#{s['index']})")
    for p in s["passes"]:
        if p["kind"] == "compute":
            pl = p.get("pipeline") or {}
            sh = (pl.get("shader") or {})
            for dnum, dsp in enumerate(p["dispatches"]):
                dim = "INDIRECT "+str((dsp.get('argBuffer') or {}).get('label') or (dsp.get('argBuffer') or {}).get('id')) if dsp.get("indirect") else f"dims={dsp.get('dims')}"
                L.append(f"  - [C cmd#{p['beginIdx']}] pipe={pl.get('label') or pl.get('id')} entry={pl.get('entryPoint')} shader={sh.get('label') or sh.get('id')}  {dim}")
            if not p["dispatches"]:
                L.append(f"  - [C cmd#{p['beginIdx']}] pipe={pl.get('label') or pl.get('id')} entry={pl.get('entryPoint')}  (no dispatch)")
        else:
            pl = p.get("pipeline") or {}
            col = ", ".join(f"{a.get('texLabel') or a.get('viewLabel') or '?'}[{res_str(a)}] {a.get('loadOp')}/{a.get('storeOp')}" for a in p["colorAttachments"])
            dep = ""
            if p["depthAttachment"]:
                da = p["depthAttachment"]; dep = f" depth={da.get('texLabel') or '?'}[{res_str(da)}]"
            vp = p.get("viewport"); vps = f" vp={vp}" if vp else ""
            drs = []
            for d in p["draws"]:
                if d.get("method") in ("drawIndirect","drawIndexedIndirect"):
                    drs.append(f"{d['method']}({(d.get('argBuffer') or {}).get('label') or (d.get('argBuffer') or {}).get('id')})")
                else:
                    drs.append(f"{d['method']}{d.get('args')}")
            L.append(f"  - [R cmd#{p['beginIdx']}] pipe={pl.get('label') or pl.get('id')} frag={(pl.get('fragmentShader') or {}).get('label')}")
            L.append(f"      color=[{col}]{dep}{vps}")
            if drs: L.append(f"      draws: {'; '.join(drs)}")
open(os.path.join(OUT, "timeline.md"), "w").write("\n".join(L))
dump("timeline.json", {"submits": [{"index": s["index"], "passes": s["passes"]} for s in submits]})

# summary.md
S = []
S.append(f"# Capture summary — frame {doc0.get('frame')}")
S.append(f"\n- tool: {doc0.get('tool')} {doc0.get('toolVersion')}  schema {doc0.get('schemaVersion')}")
S.append(f"- canvas: {canvas.get('width')}x{canvas.get('height')} {canvas.get('format')} ({(canvas.get('width') or 0)*(canvas.get('height') or 0)/1e6:.2f} Mpx)")
S.append(f"- adapter: {next((o.get('descriptor',{}).get('architecture') for o in OBJS.values() if o.get('type')=='Adapter'), '?')}")
S.append("\n## statistics")
for k,v in (doc0.get("statistics") or {}).items():
    S.append(f"- {k}: {v}")
S.append("\n## object counts")
tc = collections.Counter(o.get("type") for o in OBJS.values())
for t,c in tc.most_common(): S.append(f"- {t}: {c}")
S.append("\n## validationErrors")
for e in (doc0.get("validationErrors") or []): S.append(f"- {e.get('message')}")
totvram = sum((t.get("vramBytes") or 0) for t in textures)
S.append(f"\n## texture VRAM total: {totvram/1e6:.1f} MB across {len(textures)} textures")
S.append(f"## buffer bytes total: {sum((b.get('size') or 0) for b in buffers)/1e6:.1f} MB across {len(buffers)} buffers")
open(os.path.join(OUT, "summary.md"), "w").write("\n".join(S))

print("slices written to", OUT)
print(f"  compute passes {len(comp)} · render passes {len(rend)} · submits {sum(1 for s in submits if s['passes'])} (non-empty)")
print(f"  buffers {len(buffers)} · textures {len(textures)} ({totvram/1e6:.0f} MB) · pipelines {len(pipes)} · shaders {len(manifest)} ({len(seen_code)} unique)")
