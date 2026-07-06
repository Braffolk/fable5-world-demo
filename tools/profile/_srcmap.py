#!/usr/bin/env python3
"""_srcmap.py — resolve binaryUniqueId → (name, type, confidence, aligned source) for
runtime_perline.py --src. The compiled Mach-O carries NO shader name (only the Dawn capture
does), so we bridge through the export's `store0` (zlib): its MSL `program_source` blocks +
`Dawn_ShaderModule_*` labels. Match kid→block by DWARF barrier-call_line verification (exact for
the SW-raster family) or function-set+line-count (fuzzy); name the block by shared NodeBuffer IDs
filtered by shader type; align the block to DWARF line numbers via the barrier / a named function.
See gpuprofiler-raw-format.md §8c. Reliable for barrier-verified compute + distinctive shaders;
best-effort (flagged) otherwise. Exact universal naming would need the capture object-graph.
"""
import bisect, glob, os, re, subprocess, sys, zlib

GENERIC = set("min max clamp floor ceil sin cos tan dot abs mix sqrt rsqrt fract normalize length "
              "pow exp log log2 exp2 sign step smoothstep select rint trunc round cross reflect "
              "refract distance saturate radians degrees atan asin acos".split())


def _is_named(f):
    return (f not in GENERIC and not f.startswith(("operator", "atomic_"))
            and not re.match(r"tint_(v\d|mod|div|f32|i32|u32|unpack|pack|clamp|select|first|count|extract|insert|dot|sign|ftou|ftoi)", f)
            and f not in ("main0", "main_inner"))


def kid_sig(macho, dwarf_result):
    """(named funcs, sorted barrier call_lines, maxline, [(func, dwarf_first_line)] anchors)."""
    offs, a2l, intervals, ilos = dwarf_result
    di = subprocess.run(["dwarfdump", "--debug-info", macho], capture_output=True, text=True).stdout
    named, barriers, cur = set(), [], None
    for l in di.splitlines():
        if "DW_TAG_" in l: cur = None
        m = re.search(r'DW_AT_(?:name|abstract_origin)\s*\((?:0x[0-9a-fA-F]+\s*)?"([A-Za-z_]\w*)"\)', l)
        if m:
            cur = m.group(1)
            if _is_named(cur): named.add(cur)
        m = re.search(r'DW_AT_call_line\s*\((\d+)\)', l)
        if m and cur == "threadgroup_barrier": barriers.append(int(m.group(1)))
    maxline = max(a2l.values()) if a2l else 0
    # anchors: named concrete subprograms with a first program_source line (from debug-line over their range)
    anchors = []
    for lo, hi, nm in intervals:
        if not _is_named(nm): continue
        ls = [a2l[o] for o in a2l if lo <= o < hi]
        if ls: anchors.append((nm, min(ls)))
    return named, sorted(set(barriers)), maxline, anchors


def _nb(s):
    return set(re.findall(r"NodeBuffer_(\d+)", s))


def load_store0_text(bundle):
    """Find store0 in the exported bundle and stream-decompress (multi-member zlib) to str."""
    d = bundle if os.path.isdir(bundle) else os.path.dirname(bundle)
    s0 = os.path.join(d, "store0")
    if not os.path.isfile(s0):
        return None
    comp = open(s0, "rb").read()
    buf = bytearray(); pos = 0; dobj = zlib.decompressobj(); CH = 8 << 20
    while pos < len(comp):
        ch = comp[pos:pos+CH]; pos += len(ch)
        while ch:
            try:
                buf += dobj.decompress(ch)
            except zlib.error:
                nh = comp.find(b"\x78\x9c", pos)
                if nh < 0: pos = len(comp); ch = b""; break
                pos = nh; dobj = zlib.decompressobj(); ch = b""; break
            if dobj.eof: ch = dobj.unused_data; dobj = zlib.decompressobj()
            else: ch = b""
    try: buf += dobj.flush()
    except Exception: pass
    return buf.decode("latin1")


def build_index(text):
    """Parse store0 text → (blocks, labels). Blocks = MSL program_source; labels = Dawn names."""
    heads = [m.start() for m in re.finditer(r"#include <metal_stdlib>", text)]
    blocks = []
    for j, h in enumerate(heads):
        end = heads[j+1] if j+1 < len(heads) else min(h + 2_000_000, len(text))
        seg = text[h:end]; lines = seg.split("\n")
        typ = ("compute" if re.search(r"\bkernel \w", seg[:150000]) else
               "fragment" if re.search(r"\bfragment \w", seg[:150000]) else
               "vertex" if re.search(r"\bvertex \w", seg[:150000]) else "?")
        barriers = [i+1 for i, l in enumerate(lines) if "threadgroup_barrier" in l]
        defs = {}
        for m in re.finditer(r"(?m)^\s*[\w:<>,\*\s]+?\b([A-Za-z_]\w*)\s*\(", seg[:400000]):
            defs.setdefault(m.group(1), seg[:m.start()].count("\n") + 1)
        blocks.append({"head": h, "lines": lines, "type": typ, "nb": _nb(seg[:300000]),
                       "barriers": barriers, "barrierset": set(barriers), "defs": defs, "n": len(lines)})
    labels = [{"name": m.group(2), "type": m.group(1),
               "nb": _nb(text[max(0, m.start()-6000):m.start()+6000])}
              for m in re.finditer(r"Dawn_ShaderModule_(compute|fragment|vertex)_([A-Za-z0-9_]+)", text)]
    return blocks, labels


def _name_of(blk, labels):
    best = ("?", 0.0)
    for L in labels:
        if L["type"] != blk["type"]: continue
        inter = len(blk["nb"] & L["nb"]); uni = len(blk["nb"] | L["nb"]) or 1
        if inter and inter/uni > best[1]: best = (L["name"], inter/uni)
    return best


_REMARK = re.compile(r"^\s*(---|\.\.\.|Name:|Args:|Remarks|Function:|DebugLoc|Pass:|- (String|Caller|Callee|INST_|Line|Column|NumInstructions|Type|Analysis)|!)")
def _clean(line):
    """MSL source line, or "" if it's compiler-remark YAML / binary garbage."""
    s = line.strip()
    if not s: return ""
    if _REMARK.match(s): return ""
    printable = sum(1 for c in s if 32 <= ord(c) < 127 or c == "\t")
    if printable < 0.85 * len(s): return ""
    return s


def resolve(kid, macho, kidtype, dwarf_result, index):
    """→ dict {name, type, name_conf(0..1), how, aligned, src(line)->text} or None."""
    blocks, labels = index
    typ = kidtype
    named, barriers, maxline, anchors = kid_sig(macho, dwarf_result)
    cands = [b for b in blocks if b["type"] == typ and named and len(named & set(b["defs"])) >= 1]
    match = shift = None; how = "none"
    # 1) barrier-verified block match (separates same-family siblings)
    if barriers:
        for b in cands:
            for cbl in b["barriers"]:
                s = barriers[0] - cbl
                if all((kb - s) in b["barrierset"] for kb in barriers):
                    match, shift, how = b, s, "exact-barrier"; break
            if match: break
    # 2) fuzzy: best function-overlap + line-count; align via a named-function anchor
    if not match and cands:
        best = None
        for b in cands:
            sc = (len(named & set(b["defs"])), -abs(b["n"] - maxline))
            if best is None or sc > best[0]: best = (sc, b)
        match = best[1]; how = "fuzzy"
        for nm, dl in anchors:
            if nm in match["defs"]:
                shift = dl - match["defs"][nm]; break
    if not match:
        return None
    name, jac = _name_of(match, labels)
    lines = match["lines"]
    def raw(dl):
        i = dl - shift - 1
        return lines[i] if (shift is not None and 0 <= i < len(lines)) else ""
    # VERIFY alignment before trusting source: kid's barrier call_lines must land on barriers
    aligned = False
    if shift is not None:
        if barriers:
            ok = sum(1 for kb in barriers if any("threadgroup_barrier" in raw(kb+dk) for dk in (0, -1, 1)))
            aligned = ok >= max(1, (len(barriers) + 1) // 2)
        else:
            v = sum(1 for nm, dl in anchors if nm in match["defs"] and abs(match["defs"][nm] + shift - dl) <= 1)
            aligned = v >= 2
    def src(dl):
        return _clean(raw(dl)) if aligned else ""
    # full program_source (index0 = line1 … maxline), for the per-shader .metal dump; None if unaligned
    srclines = [raw(L) for L in range(1, maxline + 1)] if aligned else None
    return {"name": name, "type": typ, "name_conf": round(jac, 2), "how": how,
            "aligned": aligned, "src": src, "srclines": srclines}
