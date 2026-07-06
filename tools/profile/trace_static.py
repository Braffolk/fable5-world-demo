#!/usr/bin/env python3
"""
trace_static.py — LLM-readable STATIC facts from a raw `.gputrace` bundle.

INPUT: a `.gputrace` directory captured by Dawn (our `gputrace.sh`) OR by Xcode —
this works on the RAW capture (no profiling/export needed). It reads only the
command stream + resource file sizes, so it is fast (seconds) and needs no Metal
tools.

Emits: the compute-pipeline inventory (our labelled kernels), render/compute pass
counts, the GPU memory footprint + biggest resources, and metadata (frame count,
device). This is the "what work does this frame do / where is the VRAM" view —
complementary to profile_report.py (which needs an exported *profiled* trace).

Nothing here has GPU timing — a `.gputrace` stores structure + resource dumps, not
durations (those come from replay). See README.md.

Usage:
    trace_static.py <bundle.gputrace> [--json] [--top N]
"""
import argparse
import json
import os
import plistlib
import re
import sys
from collections import Counter

RESOURCE_PREFIXES = ("MTLBuffer-", "MTLTexture-", "IOSurface-")


def read_capture(bundle):
    """Return the bytes of the command stream (`capture`, else `unsorted-capture`)."""
    for name in ("capture", "unsorted-capture"):
        p = os.path.join(bundle, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return f.read()
    sys.exit(f"[trace] no 'capture' file in {bundle} — is it a .gputrace bundle?")


def resource_footprint(bundle, top_n):
    """Sum GPU resource dumps by kind (skip symlinks = dedup aliases), find biggest."""
    kinds = Counter()          # prefix -> total bytes
    counts = Counter()         # prefix -> file count
    biggest = []               # (bytes, name)
    with os.scandir(bundle) as it:
        for e in it:
            if e.is_symlink() or not e.is_file():
                continue
            for pre in RESOURCE_PREFIXES:
                if e.name.startswith(pre):
                    sz = e.stat().st_size
                    kinds[pre] += sz
                    counts[pre] += 1
                    biggest.append((sz, e.name))
                    break
    biggest.sort(reverse=True)
    return kinds, counts, biggest[:top_n]


def analyze(bundle, top_n):
    cap = read_capture(bundle)
    pipelines = sorted(set(m.decode() for m in
                           re.findall(rb"compute_[A-Za-z0-9]+::main", cap)))
    passes = {
        "render": len(re.findall(rb"Dawn_RenderPassEncoder", cap)),
        "compute": len(re.findall(rb"Dawn_ComputePassEncoder", cap)),
        "blit": len(re.findall(rb"Dawn_BlitPassEncoder", cap)),
    }
    msl_libs = len(re.findall(rb"program_source", cap))

    kinds, counts, biggest = resource_footprint(bundle, top_n)

    meta = {}
    mp = os.path.join(bundle, "metadata")
    if os.path.isfile(mp):
        try:
            with open(mp, "rb") as f:
                raw = plistlib.load(f)
            meta = {k.split(".")[-1]: v for k, v in raw.items()
                    if any(t in k for t in ("frames_count", "graphics_api", "deviceId"))}
        except Exception:
            pass

    return {
        "bundle": os.path.basename(bundle),
        "passes": passes,
        "mslLibraries": msl_libs,
        "computePipelines": pipelines,
        "memory": {
            "totalMiB": round(sum(kinds.values()) / 1048576, 1),
            "byKind": {k.rstrip("-"): {"MiB": round(v / 1048576, 1), "files": counts[k]}
                       for k, v in kinds.items()},
            "biggest": [{"MiB": round(sz / 1048576, 1), "name": n} for sz, n in biggest],
        },
        "metadata": meta,
    }


def p_report(r):
    print("=" * 64)
    print(f"  {r['bundle']}  (static)")
    print("=" * 64)
    ps = r["passes"]
    print(f"passes:   render {ps['render']}   compute {ps['compute']}   blit {ps['blit']}")
    print(f"MSL libraries embedded: {r['mslLibraries']}")
    md = r.get("metadata", {})
    if md:
        print("metadata: " + "  ".join(f"{k}={v}" for k, v in md.items()))
        if str(md.get("frames_count")) == "1":
            print("          ⚠️ frames_count=1 → Dawn trace has NO frame delimiters "
                  "(one concatenated stream; segment by swapchain writes yourself)")

    mem = r["memory"]
    print(f"\nGPU memory footprint: {mem['totalMiB']:.0f} MiB")
    for kind, d in mem["byKind"].items():
        print(f"   {kind:12s} {d['MiB']:9.0f} MiB   ({d['files']} resources)")
    print("\nbiggest resources:")
    for b in mem["biggest"]:
        print(f"   {b['MiB']:8.0f} MiB  {b['name']}")

    pl = r["computePipelines"]
    print(f"\ncompute pipelines ({len(pl)}):")
    for name in pl:
        print(f"   {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", help="a .gputrace directory (raw capture)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--top", type=int, default=12, help="how many biggest resources (default 12)")
    a = ap.parse_args()
    if not os.path.isdir(a.bundle):
        sys.exit(f"[trace] not a directory: {a.bundle} (a .gputrace is a bundle/dir)")
    r = analyze(a.bundle, a.top)
    if a.json:
        json.dump(r, sys.stdout, indent=1)
        print()
    else:
        p_report(r)


if __name__ == "__main__":
    main()
