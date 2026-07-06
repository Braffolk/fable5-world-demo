#!/usr/bin/env python3
"""
perline_remarks.py — HEADLESS per-shader + per-source-line cost from the Apple-GPU
compiler's telemetry & optimization remarks embedded in a `.gputrace`.

WHY (not PC-sample "hotness"): periodic PC sampling is too coarse to resolve
source lines. The `agc` compiler instead records, per shader and per MSL source
line, the *structural* cost that governs M1/TBDR occupancy — ALU/FP32/FP16
instruction counts, **temporary register count** (the occupancy driver on Apple
GPUs), **spilled bytes**, machine-instruction counts per line, unroll/inline —
as a telemetry dict + LLVM opt-remarks. Each shader's compile lands in a
hex-named `NSKeyedArchiver` bplist in the raw `.gputrace`; the MSL source lands in
another hex-named file; the capture stream names the shaders. This joins all
three: real per-line levers, no live GPU, no sampling.

Usage:
    perline_remarks.py <trace.gputrace> [--top N] [--src] [--json] [--min-inst K]
      --src   also print the MSL source text of each hot line
"""
import argparse
import glob
import json
import os
import plistlib
import re
import sys
from collections import defaultdict

UID = plistlib.UID
REC = re.compile(r"\n(?=--- )")
ARG = re.compile(r"-\s*([A-Za-z_][\w ]*?):\s*'?(-?\d+)'?")
LINE = re.compile(r"DebugLoc:\s*\{[^}]*Line:\s*(\d+)")
FUNC = re.compile(r"Function:\s*(\S+)")
LABEL = re.compile(rb"Dawn_ShaderModule_(?:compute|fragment|vertex)_[A-Za-z0-9_]+")
# per-shader telemetry keys we surface (occupancy order)
TELEM = ["Spilled bytes", "Temporary register count", "Uniform register count",
         "ALU instruction count", "FP32 instruction count", "FP16 instruction count",
         "Instruction count", "Texture reads instruction count", "Device load instruction count",
         "Threadgroup load instruction count", "Branch instruction count", "Wait instruction count"]


def unarchive(a):
    objs = a["$objects"]; cache = {}
    def r(n):
        if isinstance(n, UID):
            i = n.data
            if i in cache: return cache[i]
            cache[i] = None; v = ro(objs[i]); cache[i] = v; return v
        return ro(n)
    def ro(o):
        if isinstance(o, dict):
            cls = objs[o["$class"].data].get("$classname") if "$class" in o and isinstance(o["$class"], UID) else None
            if cls in ("NSDictionary", "NSMutableDictionary"): return {r(k): r(v) for k, v in zip(o["NS.keys"], o["NS.objects"])}
            if cls in ("NSArray", "NSMutableArray"): return [r(v) for v in o["NS.objects"]]
            if cls in ("NSString", "NSMutableString"): return o.get("NS.string")
            if cls in ("NSData", "NSMutableData"): return o.get("NS.data")
            return {k: r(v) for k, v in o.items() if k != "$class" and not k.startswith("$")}
        return o
    return r(a["$top"]["root"])


def index_capture(trace):
    """offset→shader label, for proximity naming."""
    cap_path = os.path.join(trace, "capture")
    if not os.path.isfile(cap_path):
        return b"", []
    d = open(cap_path, "rb").read()
    labels = [(m.start(), m.group().decode().replace("Dawn_ShaderModule_", "")) for m in LABEL.finditer(d)]
    return d, labels


def nearest_label(cap, labels, filehash):
    i = cap.find(filehash.encode())
    if i < 0 or not labels:
        return None
    return min(labels, key=lambda L: abs(L[0] - i))[1]


def classify(trace):
    """Return {remark: {...}} and {name: source_lines}."""
    remarks, sources = {}, {}
    for f in sorted(glob.glob(os.path.join(trace, "*"))):
        if not os.path.isfile(f) or os.path.islink(f):
            continue
        head = open(f, "rb").read(64)
        name = os.path.basename(f)
        if head[:8] == b"#ifdef _" or b"#pragma METAL" in head or head[:8] == b"#include":
            sources[name] = open(f, "rb").read().decode("latin1").splitlines()
            continue
        if head[:8] != b"bplist00":
            continue
        try:
            a = plistlib.loads(open(f, "rb").read(), fmt=plistlib.FMT_BINARY)
        except Exception:
            continue
        objs = a.get("$objects", [])
        if not any(isinstance(o, str) and "DebugLoc" in o and "Pass:" in o for o in objs):
            continue
        obj = unarchive(a)
        if not isinstance(obj, dict):
            continue
        telem = {k: obj[k] for k in TELEM if isinstance(obj.get(k), (int, float))}
        yaml = obj.get("Remarks") or max((o for o in objs if isinstance(o, str)), key=len, default="")
        # per-(func,line) rollup: separate per-line instr (INST_) from per-func regalloc
        per = defaultdict(lambda: defaultdict(float))
        for rec in REC.split(yaml):
            m = LINE.search(rec)
            if not m:
                continue
            line = int(m.group(1))
            fn = FUNC.search(rec).group(1) if FUNC.search(rec) else "?"
            for mk, mv in ARG.findall(rec):
                mk = mk.strip()
                if mk not in ("Line", "Column"):
                    per[(fn, line)][mk] += float(mv)
        remarks[name] = {"telem": telem, "per": per}
    return remarks, sources


def build(trace, top, min_inst):
    cap, labels = index_capture(trace)
    remarks, sources = classify(trace)
    # name each remark + source file by nearest capture label
    src_by_name = {}
    for sname, lines in sources.items():
        nm = nearest_label(cap, labels, sname)
        if nm:
            src_by_name.setdefault(nm, lines)
    shaders = {}
    for rname, R in remarks.items():
        nm = nearest_label(cap, labels, rname) or rname
        rows = []
        for (fn, line), m in R["per"].items():
            per_line_inst = int(m.get("INST_", 0))          # asm-printer per basic-block
            func_inst = int(m.get("NumInstructions", 0))    # regalloc per-function total
            rows.append({"line": line, "func": fn, "lineInstr": per_line_inst,
                         "funcInstr": func_inst, "spills": int(m.get("NumSpills", 0)),
                         "reloads": int(m.get("NumReloads", 0)), "unroll": int(m.get("UnrollCount", 0))})
        rows.sort(key=lambda r: -(r["lineInstr"] + r["spills"] * 100))
        rows = [r for r in rows if r["lineInstr"] >= min_inst or r["spills"] or r["unroll"]][:top]
        t = R["telem"]
        cost = t.get("Spilled bytes", 0) * 50 + t.get("Temporary register count", 0) * 10 + t.get("ALU instruction count", 0) * 0.01
        named = nm.startswith(("compute_", "fragment_", "vertex_"))
        entry = {"shader": nm, "remarkFile": rname, "telem": t, "cost": round(cost, 1),
                 "lines": rows, "source": src_by_name.get(nm), "named": named}
        # dedup by telemetry fingerprint (same shader compiled twice), preferring a named copy
        fp = (t.get("ALU instruction count", 0), t.get("Instruction count", 0),
              t.get("Spilled bytes", 0), t.get("Temporary register count", 0))
        key = fp if any(fp) else ("_uniq_" + rname,)
        cur = shaders.get(key)
        if cur is None or (entry["named"] and not cur["named"]):
            if cur is not None and not entry["source"]:
                entry["source"] = cur.get("source")
            shaders[key] = entry
    # second pass: collapse pipeline-variant recompiles of the same shader name,
    # keeping the costliest variant and counting the rest.
    by_name = {}
    for e in shaders.values():
        nm = e["shader"]
        prev = by_name.get(nm)
        if prev is None:
            e["variants"] = 1
            by_name[nm] = e
        else:
            e["variants"] = prev["variants"] + 1
            by_name[nm] = e if e["cost"] >= prev["cost"] else prev
            by_name[nm]["variants"] = e["variants"]
    return sorted(by_name.values(), key=lambda s: -s["cost"])


def _write_static_per_shader(outdir, shaders):
    """ONE file per shader (FULL per-line static cost, source NOT truncated) + _ranking.txt."""
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "[0-9][0-9][0-9]_*.txt")) + glob.glob(os.path.join(outdir, "_ranking.txt")):
        try: os.remove(f)
        except OSError: pass
    def safe(s): return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:60]
    withtelem = sum(1 for s in shaders if any(s["telem"].values()))
    rank = ["# shaders ranked by STATIC occupancy cost (spilled bytes → temp registers → ALU) — the M1 levers.",
            f"# {len(shaders)} shaders ({withtelem} with compiler telemetry).",
            "# cost 0 / blank telemetry = vertex/fragment shaders (the agc occupancy telemetry is COMPUTE-only here)",
            "#   → optimize those via runtime/ (per-line time), not this static view.",
            "# a hex name (e.g. 9AE1B72049FAF380) = a real shader whose Dawn name wasn't found — that's its library hash.", "",
            f"{'rank':>4}  {'cost':>10}  {'spill':>6}  {'tmpReg':>6}  {'ALU':>8}  {'name (or library hash)':<36}  file", ""]
    for i, s in enumerate(shaders, 1):
        t = s["telem"]; base = safe(s["shader"]); fname = f"{i:03d}_{base}.txt"
        spill = int(t.get("Spilled bytes", 0)); treg = int(t.get("Temporary register count", 0)); alu = int(t.get("ALU instruction count", 0))
        rank.append(f"{i:>4}  {s['cost']:>10}  {spill:>6}  {treg:>6}  {alu:>8}  {s['shader'][:36]:<36}  {fname}")
        tel = "  ".join(f"{k.replace(' instruction count','').replace(' count','')}={int(v)}" for k, v in t.items() if v)
        src = s.get("source")
        vt = f"   (×{s['variants']} pipeline variants)" if s.get("variants", 1) > 1 else ""
        L = [f"{s['shader']}{vt}", f"occupancy levers: {tel}", f"static occupancy cost: {s['cost']}",
             "", f"  {'line':>6} {'instr':>6} {'spill':>6} {'unrl':>5}  {'function':32s}  source"]
        for r in s["lines"]:
            code = src[r["line"] - 1].rstrip() if (src and 0 < r["line"] <= len(src)) else ""
            L.append(f"  {r['line']:>6} {r['lineInstr'] or r['funcInstr']:>6} {r['spills']:>6} {r['unroll']:>5}  {r['func'][:32]:32s}  {code}")
        L += ["", "Levers: spilled bytes > 0 and high temp-register count kill Apple-GPU occupancy; high per-line instr = expensive line."]
        open(os.path.join(outdir, fname), "w").write("\n".join(L) + "\n")
    open(os.path.join(outdir, "_ranking.txt"), "w").write("\n".join(rank) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--min-inst", type=int, default=6)
    ap.add_argument("--src", action="store_true", help="print MSL source of each hot line")
    ap.add_argument("--outdir", metavar="DIR", help="write ONE file per shader (FULL per-line static cost, all lines) + _ranking.txt into DIR")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(a.trace):
        sys.exit(f"[perline] not a .gputrace dir: {a.trace}")
    shaders = build(a.trace, 1000000 if a.outdir else a.top, 1 if a.outdir else a.min_inst)
    if not shaders:
        sys.exit("[perline] no compiler remarks found in this trace")
    if a.outdir:
        _write_static_per_shader(a.outdir, shaders)
        print(f"[perline] wrote {len(shaders)} per-shader files + _ranking.txt → {a.outdir}", file=sys.stderr)
        return
    if a.json:
        for s in shaders: s.pop("source", None)
        json.dump(shaders, sys.stdout, indent=1); print(); return
    print(f"per-shader + per-line cost from compiler remarks — {len(shaders)} shaders "
          f"(ranked by occupancy cost: spilled bytes → temp registers → ALU)\n")
    for s in shaders:
        t = s["telem"]
        tel = "  ".join(f"{k.replace(' instruction count','').replace(' count','')}={int(v)}"
                        for k, v in t.items() if v)
        vt = f"  (×{s['variants']} pipeline variants)" if s.get("variants", 1) > 1 else ""
        print(f"═══ {s['shader']}{vt} ═══")
        print(f"  {tel}")
        src = s["source"]
        print(f"  {'line':>6} {'instr':>6} {'spill':>6} {'unrl':>5}  {'source' if (a.src and src) else 'function'}")
        for r in s["lines"]:
            code = ""
            if a.src and src and 0 < r["line"] <= len(src):
                code = src[r["line"] - 1].strip()[:70]
            tail = code if (a.src and src) else r["func"][:38]
            print(f"  {r['line']:>6} {r['lineInstr'] or r['funcInstr']:>6} {r['spills']:>6} {r['unroll']:>5}  {tail}")
        print()
    print("Levers: spilled bytes > 0 and high temp-register count kill Apple-GPU occupancy;")
    print("high per-line instr = expensive lines. Use --src to see the MSL. (names via capture proximity)")


if __name__ == "__main__":
    main()
