#!/usr/bin/env python3
"""
perinstr_hotness.py — HEADLESS per-instruction GPU hotness from an exported
profiled `.gputrace`, by decoding the profiler's PC execution trace.

This is the payoff of the reverse-engineering in `gpuprofiler-raw-format.md`:
the `.gpuprofiler_raw` streamData carries a **program-counter execution trace**
(`Encoder Infos` = per-pass streams of sampled PCs; also GPRWCNTR `valueA`). Its
histogram = relative time spent at each instruction — which no existing tool
extracts. This runs with only Python (no Metal tools, no live GPU).

What it gives: per-instruction hotness (`PC-base : samples : %`) — i.e. *which
instructions dominate*. What it does NOT give: the MSL source line for each PC —
that needs the driver's compiled binary (see gpuprofiler-raw-format.md §7/§8; the
trace is source-based so the compiled metallib with the PC→line table isn't
persisted). Use this to find the hot instruction region, then read the exact
source line in Xcode's Shaders tab.

Usage:
    perinstr_hotness.py <exported.gputrace | *.gpuprofiler_raw> [--top N] [--json]
"""
import argparse
import glob
import json
import os
import plistlib
import struct
import sys
from collections import Counter

UID = plistlib.UID


def find_raw(path):
    if os.path.isdir(path):
        hits = glob.glob(os.path.join(path, "*.gpuprofiler_raw"))
        if not hits:
            sys.exit(f"[perinstr] no *.gpuprofiler_raw in {path} (export with 'Profile GPU Trace' on)")
        return hits[0]
    return path


def unarchive(arch):
    objs = arch["$objects"]; cache = {}
    def res(n):
        if isinstance(n, UID):
            i = n.data
            if i in cache: return cache[i]
            cache[i] = None
            v = ro(objs[i]); cache[i] = v; return v
        return ro(n)
    def ro(o):
        if isinstance(o, dict):
            cls = objs[o["$class"].data].get("$classname") if "$class" in o and isinstance(o["$class"], UID) else None
            if cls in ("NSDictionary", "NSMutableDictionary"):
                return {res(k): res(v) for k, v in zip(o["NS.keys"], o["NS.objects"])}
            if cls in ("NSArray", "NSMutableArray", "NSSet", "NSMutableSet"):
                return [res(v) for v in o["NS.objects"]]
            if cls in ("NSString", "NSMutableString"): return o.get("NS.string")
            if cls in ("NSData", "NSMutableData"): return o.get("NS.data")
            return {k: res(v) for k, v in o.items() if k != "$class" and not k.startswith("$")}
        return o
    return res(arch["$top"]["root"])


def extract(raw_path, top):
    with open(raw_path, "rb") as f:
        arch = plistlib.load(f, fmt=plistlib.FMT_BINARY)
    objs = arch["$objects"]
    def deref(u): return objs[u.data] if isinstance(u, UID) else u
    root = deref(arch["$top"]["root"])

    arr = deref(root["shaderProfilerData"])["NS.objects"]
    # the aggregate entry (largest NS.data) holds the sample streams
    idx = max(range(len(arr)), key=lambda i: len(deref(arr[i]).get("NS.data", b"")))
    entry = unarchive(plistlib.loads(deref(arr[idx])["NS.data"], fmt=plistlib.FMT_BINARY))

    # Encoder Infos = per-pass PC sample streams (u32). Histogram = per-instruction time.
    pcs = Counter()
    infos = entry.get("Encoder Infos") or []
    for blob in infos:
        if not isinstance(blob, bytes):
            continue
        n = len(blob) // 4
        pcs.update(struct.unpack(f"<{n}I", blob[: n * 4]))
    if not pcs:
        sys.exit("[perinstr] no PC samples found (Encoder Infos empty) — is this a profiled export?")

    base = min(pcs)
    total = sum(pcs.values())
    rows = [{"pc_off": pc - base, "pc": pc, "samples": c, "pct": 100 * c / total}
            for pc, c in pcs.most_common(top)]
    return {
        "traceName": deref(root.get("traceName")) if "traceName" in root else None,
        "totalSamples": total,
        "distinctPCs": len(pcs),
        "pcBase": base,
        "pcSpan": max(pcs) - base,
        "timebase": entry.get("Timebase"),
        "hottest": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    raw = find_raw(a.path)
    print(f"[perinstr] parsing {os.path.basename(raw)} …", file=sys.stderr)
    r = extract(raw, a.top)
    if a.json:
        json.dump(r, sys.stdout, indent=1); print(); return
    print(f"per-instruction hotness — {r['totalSamples']} PC samples, "
          f"{r['distinctPCs']} distinct instructions, base 0x{r['pcBase']:x}, span {r['pcSpan']}")
    print(f"{'instr(+off)':>12}  {'samples':>8}  {'%':>6}")
    print("-" * 32)
    for row in r["hottest"]:
        print(f"{'+'+str(row['pc_off']):>12}  {row['samples']:>8}  {row['pct']:>5.1f}%")
    print("\n(per-instruction = relative GPU time per instruction; map to the MSL line in")
    print(" Xcode's Shaders tab. See gpuprofiler-raw-format.md for why per-line is GUI-only.)")


if __name__ == "__main__":
    main()
