#!/usr/bin/env python3
"""
runtime_perline.py — HEADLESS RUNTIME time-per-source-line for Metal shaders, from an exported
profiled `.gputrace`. This is the same signal Xcode's Shaders tab shows (statistical PC sampling
during GPU replay), reconstructed offline with no live GPU. Full RE in gpuprofiler-raw-format.md §8c.

PIPELINE (join three structures inside the `*.gpuprofiler_raw`):
  1. PC samples — every per-source entry (the ~488 dicts keyed `Source|SourceIndex|
     RingBufferIndex|ShaderProfilerData`, one per source × replay iteration) has a
     `ShaderProfilerData` blob that is a `GPRWCNTR` stream. Its u32 words that land inside a PAB
     range are sampled program counters (in `mappedAddress` space); their count = time.
  2. Program Address Buffer (`Program Address Mappings`, entry ~494) — the per-command load-address
     table: `{binaryUniqueId, mappedAddress, mappedSize, type}`. The SAME binary is remapped at a
     DIFFERENT mappedAddress per GPU command (why a single global base only ever matched ~15% —
     the base is per-command, not global; Apple patents US9799087B2 / US10310830).
  3. Havested Binaries (entry[1]) — native Mach-O per binaryUniqueId, with dense DWARF:
     `__debug_line` (offset→program_source line) + `DW_TAG_subprogram` low/high_pc (offset→function).
  PC → containing PAB range → (binary, offset=PC−mappedAddress) → DWARF → (line, function).

NOTE line 0 = DWARF "no source line" = compiler glue / register spills+reloads / prologue. It is
real time; a shader that is line-0-heavy is spill/occupancy-bound (cross-check perline_remarks.py).

Usage:  runtime_perline.py <exported.gputrace|*.gpuprofiler_raw> [--top N] [--lines K]
                            [--min-samples M] [--both] [--json]
  --both   also fold in gpuTimelineData (a 2nd sampled frame; more samples, ~2x slower)
Requires `dwarfdump` (Xcode CLT). No live GPU, no network.
"""
import argparse, bisect, glob, json, os, plistlib, re, shutil, struct, subprocess, sys, tempfile
from collections import Counter, defaultdict
UID = plistlib.UID


def unarchive(a):
    """Minimal NSKeyedUnarchiver → plain dict/list/bytes/str/int."""
    objs = a["$objects"]; cache = {}
    def r(n):
        if isinstance(n, UID):
            i = n.data
            if i in cache: return cache[i]
            cache[i] = None; v = ro(objs[i]); cache[i] = v; return v
        return ro(n)
    def ro(x):
        if isinstance(x, dict):
            cl = objs[x["$class"].data].get("$classname") if "$class" in x and isinstance(x["$class"], UID) else None
            if cl in ("NSDictionary", "NSMutableDictionary"): return {r(k): r(v) for k, v in zip(x["NS.keys"], x["NS.objects"])}
            if cl in ("NSArray", "NSMutableArray"): return [r(v) for v in x["NS.objects"]]
            if cl in ("NSString", "NSMutableString"): return x.get("NS.string")
            if cl in ("NSData", "NSMutableData"): return x.get("NS.data")
            return {k: r(v) for k, v in x.items() if k != "$class" and not k.startswith("$")}
        return x
    return r(a["$top"]["root"])


def find_raw(p):
    if os.path.isdir(p):
        h = [x for x in glob.glob(os.path.join(p, "*.gpuprofiler_raw")) if os.path.isfile(x)]
        if not h: sys.exit(f"[runtime] no .gpuprofiler_raw FILE in {p} (was 'Profile GPU Trace' on during Replay, then File▸Export?)")
        return h[0]
    return p


LOW = re.compile(r"DW_AT_low_pc\s*\(0x([0-9a-fA-F]+)\)")
HIGH = re.compile(r"DW_AT_high_pc\s*\(0x([0-9a-fA-F]+)\)")
NAME = re.compile(r'DW_AT_name\s*\("([^"]*)"\)')
ORIGIN = re.compile(r'DW_AT_abstract_origin\s*\(0x[0-9a-fA-F]+\s*"([^"]*)"\)')
RANGE = re.compile(r"\[0x([0-9a-fA-F]+),\s*0x([0-9a-fA-F]+)\)")


def dwarf(macho):
    """(sorted_offsets, {offset:line}, intervals=[(lo,hi,name)], interval_los) from the Mach-O's
    DWARF. intervals include concrete subprograms AND inlined_subroutines (with abstract_origin
    name + DW_AT_ranges), so func_at can name the innermost inlined function per offset."""
    dl = subprocess.run(["dwarfdump", "--debug-line", macho], capture_output=True, text=True).stdout
    a2l = {}
    for l in dl.splitlines():
        s = l.split()
        if len(s) > 3 and s[0].startswith("0x") and s[1].isdigit():
            a2l[int(s[0], 16)] = int(s[1])
    di = subprocess.run(["dwarfdump", "--debug-info", macho], capture_output=True, text=True).stdout
    intervals, cur = [], None
    def flush():
        if not cur or not cur.get("name"): return
        nm = cur["name"]
        if nm.startswith("dawn_entry_point"): nm = "«entry»"
        rs = cur["ranges"] if cur.get("ranges") else ([(cur["lo"], cur["hi"])] if "lo" in cur and "hi" in cur else [])
        for lo, hi in rs:
            if hi > lo: intervals.append((lo, hi, nm))
    for l in di.splitlines():
        if "DW_TAG_subprogram" in l:
            flush(); cur = {"ranges": []}; continue
        if "DW_TAG_inlined_subroutine" in l:
            flush(); cur = {"ranges": []}; continue
        if "DW_TAG_" in l:
            flush(); cur = None; continue
        if cur is None: continue
        m = LOW.search(l)
        if m: cur["lo"] = int(m.group(1), 16)
        m = HIGH.search(l)
        if m:
            hv = int(m.group(1), 16); cur["hi"] = hv if hv > cur.get("lo", 0) else cur.get("lo", 0) + hv
        m = ORIGIN.search(l) or NAME.search(l)
        if m and "name" not in cur: cur["name"] = m.group(1)
        for rm in RANGE.finditer(l):
            cur["ranges"].append((int(rm.group(1), 16), int(rm.group(2), 16)))
    flush()
    intervals.sort()
    return sorted(a2l), a2l, intervals, [iv[0] for iv in intervals]


def func_at(intervals, ilos, off):
    """innermost (smallest) interval containing off."""
    j = bisect.bisect_right(ilos, off) - 1; best = None; k = 0
    while j >= 0 and k < 512:
        lo, hi, nm = intervals[j]
        if lo <= off < hi and (best is None or hi - lo < best[1] - best[0]): best = intervals[j]
        if off - lo > 0x40000: break
        j -= 1; k += 1
    return best[2] if best else "?"


def _write_per_shader(outdir, shaders, grand):
    """ONE file per shader (FULL per-line breakdown, nothing truncated) + full .metal source in
    outdir/msl/ + _ranking.txt — so an agent can open a single shader and optimize it in isolation."""
    os.makedirs(outdir, exist_ok=True)
    # clear stale output from a previous run so the set is exactly this run's shaders
    for f in glob.glob(os.path.join(outdir, "[0-9][0-9][0-9]_*.txt")) + glob.glob(os.path.join(outdir, "_ranking.txt")):
        try: os.remove(f)
        except OSError: pass
    msldir = os.path.join(outdir, "msl"); shutil.rmtree(msldir, ignore_errors=True); os.makedirs(msldir, exist_ok=True)
    def safe(s): return re.sub(r"[^A-Za-z0-9._-]", "_", s)[:60]
    # names can repeat across sibling binaries (fuzzy match) — flag so each row stays traceable
    from collections import Counter as _C
    dupe = {n for n, c in _C(s.get("name") for s in shaders if s.get("name") and s.get("name") != "?").items() if c > 1}
    rank = ["# shaders ranked by % of sampled GPU time — open ONE file to optimize one shader at a time.",
            f"# total attributed samples: {grand:,}   shaders: {len(shaders)}   (full source in ./msl/)",
            "# a name repeated across rows = fuzzy match couldn't separate sibling binaries; the 'bin' column is the unique id.", "",
            f"{'rank':>4}  {'%GPU':>6}  {'samples':>12}  {'type':<9}  {'src?':<4}  {'bin':<8}  {'name (or bin id)':<30}  file", ""]
    for i, s in enumerate(shaders, 1):
        nm = s.get("name"); named = bool(nm and nm != "?")
        bid = s["binaryId"][-8:].lstrip("0") or "0"
        # duplicate names get the bin id appended so files + rows are unambiguous
        base = (f"{safe(nm)}~{bid}" if named and nm in dupe else safe(nm)) if named else f"bin_{bid}"
        fname = f"{i:03d}_{base}.txt"
        disp = (f"{nm} (~{bid})" if nm in dupe else nm) if named else f"bin {bid}"
        srclines = s.get("srclines")
        metaname = f"{i:03d}_{base}.metal" if srclines else ""
        if srclines:
            # full program_source: file line N == program_source line N (== breakdown line N)
            open(os.path.join(msldir, metaname), "w").write("\n".join(srclines) + "\n")
        rank.append(f"{i:>4}  {s['pctGpu']:>5.1f}%  {s['samples']:>12,}  {s['type']:<9}  {'yes' if srclines else 'no':<4}  {bid:<8}  {(nm if named else '—')[:30]:<30}  {fname}")
        cn = (f"[name: {s['how']}, NodeBuffer {int(s.get('nameConf',0)*100)}%]" if named else "[name: unresolved — best-effort matching failed]")
        srcref = f"full source: msl/{metaname}" if srclines else "full source: unavailable (block alignment unverified)"
        L = [f"{disp}   [{s['type']}]   bin {s['binaryId']}   {cn}",
             f"{s['pctGpu']}% of sampled GPU time    {s['samples']:,} samples    {len(s['lines'])} source lines with samples    {srcref}",
             "", f"  {'%shdr':>6} {'samples':>12} {'line':>6}  {'function':26s}  source"]
        for r in s["lines"]:
            tail = r.get("src", "")
            if not tail and r["line"] == 0: tail = "(compiler glue / register spills / prologue)"
            L.append(f"  {r['pct']:>5.1f}% {r['samples']:>12,} {r['line']:>6}  {r['func'][:26]:26s}  {tail}")
        L += ["", "line 0 = compiler glue / register spills / prologue (spill-bound).",
              "cost can land ±1-2 lines from Xcode (raw PC sampling puts a barrier/branch stall on the next instruction)."]
        open(os.path.join(outdir, fname), "w").write("\n".join(L) + "\n")
    open(os.path.join(outdir, "_ranking.txt"), "w").write("\n".join(rank) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="exported .gputrace bundle OR the *.gpuprofiler_raw file")
    ap.add_argument("--top", type=int, default=20, help="shaders to show (default 20)")
    ap.add_argument("--lines", type=int, default=8, help="hot lines per shader (default 8)")
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--both", action="store_true", help="also fold in gpuTimelineData (2nd frame)")
    ap.add_argument("--src", action="store_true", help="resolve shader name+type & show source text per line (needs the exported bundle's store0; +~90s, ~5.5GB RAM)")
    ap.add_argument("--outdir", metavar="DIR", help="write ONE file per shader (FULL per-line breakdown, all lines) + _ranking.txt into DIR — for focused, one-shader-at-a-time optimization")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    raw = find_raw(a.trace)
    print(f"[runtime] loading {os.path.basename(raw)} ({os.path.getsize(raw)/1e9:.1f} GB) …", file=sys.stderr)
    with open(raw, "rb") as f:
        d = plistlib.load(f, fmt=plistlib.FMT_BINARY)
    objs = d["$objects"]
    def deref(u): return objs[u.data] if isinstance(u, UID) else u
    def nested(u):
        nd = objs[u.data] if isinstance(u, UID) else u
        return unarchive(plistlib.loads(nd["NS.data"], fmt=plistlib.FMT_BINARY))
    root = deref(d["$top"]["root"])

    harv = tempfile.mkdtemp(prefix="harv_")
    try:
        kidtype = {}
        for arrname in ("shaderProfilerData", "gpuTimelineData"):
            for u in deref(root[arrname])["NS.objects"]:
                nd = objs[u.data] if isinstance(u, UID) else u
                if isinstance(nd, dict) and b"Havested Binaries" in nd.get("NS.data", b""):
                    for cat, cd in nested(u)["Havested Binaries"].items():
                        if isinstance(cd, dict):
                            for kid, info in cd.items():
                                if isinstance(info, dict) and isinstance(info.get("binary"), (bytes, bytearray)):
                                    open(f"{harv}/{kid}.macho", "wb").write(info["binary"]); kidtype[kid] = info.get("type")
                    break
        print(f"[runtime] {len(kidtype)} harvested binaries", file=sys.stderr)

        dwc = {}
        def dw(kid):
            if kid not in dwc:
                mo = f"{harv}/{kid}.macho"
                dwc[kid] = dwarf(mo) if os.path.exists(mo) else None
            return dwc[kid]

        sample = defaultdict(Counter)   # kid -> Counter(line -> samples)
        lineoff = {}                    # (kid,line) -> representative offset (for function lookup at output)
        bin_tot = Counter()
        for arrname in (("shaderProfilerData", "gpuTimelineData") if a.both else ("shaderProfilerData",)):
            arr = deref(root[arrname])["NS.objects"]
            pabu = [u for u in arr if b"Program Address Mappings" in (objs[u.data]["NS.data"] if isinstance(u, UID) else u["NS.data"])]
            if not pabu: continue
            M = nested(pabu[0])["Program Address Mappings"]
            ranges = sorted(set((m["mappedAddress"], m["mappedAddress"] + m["mappedSize"], m["binaryUniqueId"]) for m in M))
            starts = [r[0] for r in ranges]
            lo, hi = starts[0], max(r[1] for r in ranges)
            maxsz = max(e - s for s, e, _ in ranges)
            def lookup(pc):
                # smallest PAB range containing pc; bounded lookback (USC heap reuses addresses per command)
                j = bisect.bisect_right(starts, pc) - 1; best = None; k = 0
                while j >= 0 and k < 64:
                    s = starts[j]
                    if pc - s > maxsz: break
                    e = ranges[j][1]
                    if s <= pc < e and (best is None or e - s < best[1] - best[0]): best = ranges[j]
                    j -= 1; k += 1
                return best
            for u in arr:
                nd = objs[u.data] if isinstance(u, UID) else u
                rawb = nd.get("NS.data") if isinstance(nd, dict) else None
                if not rawb or b"RingBufferIndex" not in rawb or b"ShaderProfilerData" not in rawb:
                    continue
                E = unarchive(plistlib.loads(rawb, fmt=plistlib.FMT_BINARY))
                b = E.get("ShaderProfilerData") if isinstance(E, dict) else None
                if not (isinstance(b, (bytes, bytearray)) and b[:4] == b"GPRW"):
                    continue
                for x in struct.unpack(f"<{len(b)//4}I", b[:len(b) // 4 * 4]):
                    if x < lo or x >= hi: continue
                    c = lookup(x)
                    if not c: continue
                    dr = dw(c[2])
                    if not dr: continue
                    off = x - c[0]; offs, a2l = dr[0], dr[1]
                    if offs and offs[0] <= off <= offs[-1]:
                        ln = a2l[offs[bisect.bisect_right(offs, off) - 1]]
                        sample[c[2]][ln] += 1; bin_tot[c[2]] += 1
                        lineoff.setdefault((c[2], ln), off)

        grand = sum(bin_tot.values()) or 1
        nl = None if a.outdir else a.lines           # --outdir → ALL lines per shader
        shaders = []
        for kid, tot in bin_tot.most_common():
            if tot < a.min_samples: continue
            dr = dw(kid); rows = []
            for ln, c in sample[kid].most_common(nl):
                fn = func_at(dr[2], dr[3], lineoff[(kid, ln)]) if dr else "?"
                rows.append({"line": ln, "pct": round(100 * c / tot, 1), "samples": c, "func": fn})
            shaders.append({"binaryId": kid, "type": kidtype.get(kid, "?"), "samples": tot,
                            "pctGpu": round(100 * tot / grand, 1), "lines": rows})

        if a.src:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import _srcmap
            print("[runtime] --src: decompressing store0 (~90s, ~5.5 GB RAM) …", file=sys.stderr)
            text = _srcmap.load_store0_text(a.trace)
            if not text:
                print("[runtime] --src: no store0 — pass the exported .gputrace DIRECTORY (not the .gpuprofiler_raw file)", file=sys.stderr)
            else:
                index = _srcmap.build_index(text)
                targets = shaders if a.outdir else shaders[:a.top]   # resolve ALL when writing per-shader files
                print(f"[runtime] --src: resolving names/source for {len(targets)} shaders …", file=sys.stderr)
                for s in targets:
                    mo = f"{harv}/{s['binaryId']}.macho"
                    r = _srcmap.resolve(s["binaryId"], mo, s["type"], dw(s["binaryId"]), index) if os.path.exists(mo) else None
                    if r:
                        s["name"], s["nameConf"], s["how"] = r["name"], r["name_conf"], r["how"]
                        if r["aligned"]:
                            for row in s["lines"]: row["src"] = r["src"](row["line"])
                            s["srclines"] = r.get("srclines")
    finally:
        shutil.rmtree(harv, ignore_errors=True)

    if a.outdir:
        _write_per_shader(a.outdir, shaders, grand)
        print(f"[runtime] wrote {len(shaders)} per-shader files + _ranking.txt → {a.outdir}", file=sys.stderr)
        return

    if a.json:
        json.dump({"totalSamples": grand, "shaders": shaders[:a.top]}, sys.stdout, indent=1); print(); return
    print("RUNTIME time-per-source-line (headless PC sampling ≈ Xcode Shaders tab)")
    print(f"total attributed samples: {grand}   shaders: {len(shaders)}\n")
    for s in shaders[:a.top]:
        nm = s.get("name")
        title = f"{nm} " if nm and nm != "?" else ""
        conf = f"  [name: {s['how']}, NodeBuffer {int(s.get('nameConf',0)*100)}%]" if nm and nm != "?" else ""
        print(f"═══ {title}[{s['type']}]  bin {s['binaryId']}  {s['samples']} samples = {s['pctGpu']}% of GPU{conf} ═══")
        show_src = any("src" in r for r in s["lines"])
        print(f"  {'%GPU':>6} {'line':>6}  {'function':20s}  {'source' if show_src else ''}")
        for r in s["lines"]:
            tail = r.get("src", "")
            if not tail and r["line"] == 0: tail = "(compiler glue / register spills / prologue)"
            print(f"  {r['pct']:>5.1f}% {r['line']:>6}  {r['func'][:20]:20s}  {tail[:70]}")
        print()
    print("name via store0 (exact for barrier-verified compute; else best-effort — confidence shown).")
    print("line = program_source; line 0 = glue/spills/prologue (spill-bound). func = DWARF subprogram.")
    print("Cost can land ±1-2 lines from Xcode (PC sampling vs source rollup); use --src, cross-check in Xcode.")


if __name__ == "__main__":
    main()
