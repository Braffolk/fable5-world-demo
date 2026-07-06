#!/usr/bin/env python3
"""
profile_report.py — LLM-readable report from an EXPORTED Metal GPU profile.

INPUT: an Xcode "export performance data into a GPU Trace" bundle (a `.gputrace`
directory that contains a `*.gpuprofiler_raw` file), OR the `.gpuprofiler_raw`
file directly. Produce that in Xcode: open a `.gputrace`, enable *Profile GPU
Trace*, Replay, then File ▸ Export.

WHY this exists: the `.gpuprofiler_raw` is a 2–3 GB `NSKeyedArchiver` binary
plist that holds everything the GUI Shaders/Summary/Timeline tabs show. This tool
extracts the parts that are ROBUSTLY readable headlessly — device + GPU config,
capture inventory (encoders/shaders/pipelines/blits), and the full GPU **counter
glossary** (name → type → description) — and caches them to a small JSON sidecar
so future runs are instant.

WHAT IT DOES NOT DO: the exact per-encoder milliseconds and per-source-line
counter percentages live in proprietary binary sample streams inside the archive
(`Derived Counter Sample Data` = `GPRWCNTR`-magic blobs, `Encoder Time Sample
Data`, plus a `DerivedCounterScript` that must be interpreted). Re-implementing
Xcode's counter math from those is out of scope. For those exact numbers use the
Xcode GUI (Shaders/Summary tabs) now, or `gpudebug --json` on macOS 27 (see
README.md). This tool gives the structural + reference orientation that makes
reading them fast.

Usage:
    profile_report.py <bundle-or-raw> [summary|counters|device|inventory|json]
    profile_report.py <bundle-or-raw> counters --json
    profile_report.py <bundle-or-raw> --rebuild        # force re-parse the 2–3 GB

Default subcommand is `summary`. First run parses the archive (~30–90 s, needs
RAM ≈ 3× file size briefly) and writes `<raw>.summary.json`; later runs read that.
"""
import argparse
import glob
import json
import os
import plistlib
import struct
import sys

UID = plistlib.UID


# ----------------------------------------------------------------------------- IO
def find_raw(path):
    """Accept an exported .gputrace dir or the .gpuprofiler_raw directly."""
    if os.path.isdir(path):
        hits = glob.glob(os.path.join(path, "*.gpuprofiler_raw"))
        if not hits:
            sys.exit(f"[profile] no *.gpuprofiler_raw in {path} — did you EXPORT "
                     f"from Xcode with 'Profile GPU Trace' on? (a raw capture has none)")
        return hits[0]
    return path


def unarchive(archive):
    """Minimal NSKeyedUnarchiver: resolve $objects graph to plain py values."""
    objs = archive["$objects"]
    cache = {}

    def res(node):
        if isinstance(node, UID):
            i = node.data
            if i in cache:
                return cache[i]
            cache[i] = None  # cycle guard
            v = obj(objs[i])
            cache[i] = v
            return v
        return obj(node)

    def obj(o):
        if isinstance(o, dict):
            cls = None
            if "$class" in o and isinstance(o["$class"], UID):
                cls = objs[o["$class"].data].get("$classname")
            if cls in ("NSDictionary", "NSMutableDictionary"):
                return {res(k): res(v) for k, v in zip(o["NS.keys"], o["NS.objects"])}
            if cls in ("NSArray", "NSMutableArray", "NSSet", "NSMutableSet"):
                return [res(v) for v in o["NS.objects"]]
            if cls in ("NSString", "NSMutableString"):
                return o.get("NS.string")
            if cls in ("NSData", "NSMutableData"):
                return o.get("NS.data")
            return {k: res(v) for k, v in o.items()
                    if k != "$class" and not k.startswith("$")}
        return o

    return res(archive["$top"]["root"])


# --------------------------------------------------------------------- extraction
def build_summary(raw_path):
    """Parse the .gpuprofiler_raw once → a compact, JSON-able dict."""
    with open(raw_path, "rb") as f:
        arch = plistlib.load(f, fmt=plistlib.FMT_BINARY)
    objs = arch["$objects"]

    def deref(u):
        return objs[u.data] if isinstance(u, UID) else u

    root = deref(arch["$top"]["root"])

    def scalar(key, default=None):
        v = root.get(key, default)
        return deref(v) if isinstance(v, UID) else v

    def array_len(key):
        v = deref(root[key]) if key in root else None
        return len(v["NS.objects"]) if isinstance(v, dict) and "NS.objects" in v else 0

    def table_count(key):
        """A fixed-stride `*InfoData` blob → record count via sibling `*InfoSize`."""
        if key not in root:
            return None
        blob = deref(root[key]).get("NS.data")
        size = scalar(key.replace("Data", "Size"))
        return (len(blob) // size) if blob and size else None

    dev = scalar("deviceInfo") or {}
    device = {k: (deref(v) if isinstance(v, UID) else v)
              for k, v in dev.items()} if isinstance(dev, dict) else {}

    summary = {
        "traceName": scalar("traceName"),
        "profilerFormatVersion": scalar("version"),
        "unixTimestamp": scalar("unixTimestamp"),
        "gpuGeneration": scalar("gpuGeneration"),
        "numBlitCalls": scalar("numBlitCalls"),
        "captureRange": {"location": scalar("captureRangeLocation"),
                         "length": scalar("captureRangeLength")},
        "device": device,
        "inventory": {
            "encoders": table_count("encoderInfoData"),
            "gpuCommands": table_count("gpuCommandInfoData"),
            "commandBuffers": table_count("commandBufferInfoData"),
            "pipelineStates": table_count("pipelineStateInfoData"),
            "functions": table_count("functionInfoData"),
            "shaderProfilerEntries": array_len("shaderProfilerData"),
            "gpuTimelineSegments": array_len("gpuTimelineData"),
        },
    }

    # string pool (Dawn tends to collapse entry points to 'main0')
    sp = deref(root["strings"]) if "strings" in root else None
    if isinstance(sp, dict) and "NS.objects" in sp:
        summary["stringPool"] = [deref(u) if isinstance(u, UID) else u
                                 for u in sp["NS.objects"]]

    # counter glossary + GPU config — from the smallest nested entry that has it
    _extract_counter_dictionary(root, deref, objs, summary)
    return summary


def _extract_counter_dictionary(root, deref, objs, summary):
    """Pull the DerivedCounter dictionary + config from a nested entry. Every entry
    carries the same dictionary, but some are degenerate (empty/marker), so scan
    entries from smallest up and use the first that actually yields a glossary
    (small = fast to unarchive; capped so we stay quick)."""
    for key in ("gpuTimelineData", "shaderProfilerData"):
        arr = deref(root[key])["NS.objects"] if key in root else []
        if not arr:
            continue
        by_size = sorted(arr, key=lambda u: len(deref(u).get("NS.data", b"")))
        for u in by_size[:8]:
            blob = deref(u).get("NS.data", b"")
            if not blob:
                continue
            entry = unarchive(plistlib.loads(blob, fmt=plistlib.FMT_BINARY))
            dd = entry.get("DerivedCounterDictionary", {})
            counters = dd.get("DerivedCounters", {}) if isinstance(dd, dict) else {}
            if not counters:
                continue
            cfg = entry.get("DerivedCounterConfigurationVariables")
            if isinstance(cfg, dict):
                summary["gpuConfig"] = {k: v for k, v in cfg.items()
                                        if not isinstance(v, (bytes, list))}
            sampled = set(entry.get("profileCounters", []) or [])
            glossary = {}
            for cid, spec in counters.items():
                if not isinstance(spec, dict):
                    continue
                raws = spec.get("counters", []) or []
                glossary[cid] = {
                    "name": spec.get("name", cid),
                    "type": spec.get("type"),          # Percentage | Rate | Count
                    "subtype": spec.get("subtype"),
                    "description": (spec.get("description") or "").strip(),
                    "sampled": bool(raws) and all(r in sampled for r in raws),
                }
            summary["counters"] = glossary
            return


# ------------------------------------------------------------------------ caching
def load_or_build(raw_path, rebuild=False):
    sidecar = raw_path + ".summary.json"
    if not rebuild and os.path.exists(sidecar) \
            and os.path.getmtime(sidecar) >= os.path.getmtime(raw_path):
        with open(sidecar) as f:
            return json.load(f)
    sz = os.path.getsize(raw_path) / 1e9
    print(f"[profile] parsing {os.path.basename(raw_path)} ({sz:.1f} GB) — "
          f"one-time, ~30–90 s…", file=sys.stderr, flush=True)
    summary = build_summary(raw_path)
    with open(sidecar, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[profile] cached → {os.path.basename(sidecar)}", file=sys.stderr)
    return summary


# ----------------------------------------------------------------------- printers
def p_device(s):
    d = s.get("device", {})
    g = s.get("gpuConfig", {})
    print(f"device:        {d.get('name','?')}")
    print(f"os / metal:    macOS {d.get('version','?')}  ·  Metal {d.get('metalVersion','?')}  (build {d.get('build','?')})")
    print(f"gpu:           gen {g.get('gpu_gen','?')} var {g.get('gpu_var','?')}  ·  "
          f"{g.get('num_cores','?')} cores  ·  {g.get('num_frags','?')} frags  ·  {g.get('num_gps','?')} GPs")


def p_summary(s):
    print("=" * 68)
    print(f"  {s.get('traceName','(profile)')}")
    print("=" * 68)
    p_device(s)
    inv = s.get("inventory", {})
    print("\ninventory (this capture):")
    for k in ("encoders", "commandBuffers", "gpuCommands", "pipelineStates",
              "functions", "shaderProfilerEntries", "gpuTimelineSegments"):
        v = inv.get(k)
        if v is not None:
            print(f"   {k:22s} {v}")
    print(f"   {'blitCalls':22s} {s.get('numBlitCalls')}")
    cr = s.get("captureRange", {})
    print(f"   {'captureRange':22s} loc {cr.get('location')} len {cr.get('length')}")
    cs = s.get("counters", {})
    if cs:
        n_samp = sum(1 for c in cs.values() if c.get("sampled"))
        print(f"\ncounters available: {len(cs)}  ({n_samp} sampled in this trace)")
        print("   → `profile_report.py <path> counters`  for the full glossary")
    print("\nexact per-encoder ms & per-line counter %% are NOT in this tool's")
    print("output (proprietary sample streams) — read them in Xcode's Shaders/")
    print("Summary tabs, or via `gpudebug --json` on macOS 27.  See README.md.")


def p_counters(s):
    cs = s.get("counters", {})
    if not cs:
        print("(no counter dictionary found)")
        return
    rows = sorted(cs.values(), key=lambda c: (not c["sampled"], c["name"]))
    w = max((len(c["name"]) for c in rows), default=4)
    print(f"{'':1}{'counter':{w}}  {'type':10}  {'smpl':4}  description")
    print("-" * (w + 78))
    for c in rows:
        mark = "✓" if c["sampled"] else " "
        desc = (c["description"] or "")[:80]
        print(f"{mark}{c['name']:{w}}  {str(c['type'] or ''):10}  {'yes' if c['sampled'] else '  -':4}  {desc}")
    print(f"\n{len(rows)} counters · ✓ = sampled in this trace · "
          f"types: Percentage (occupancy/limiter/util), Rate (bandwidth), Count.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="exported .gputrace bundle OR a .gpuprofiler_raw")
    ap.add_argument("cmd", nargs="?", default="summary",
                    choices=["summary", "counters", "device", "inventory", "json"])
    ap.add_argument("--json", action="store_true", help="emit JSON for `counters`/`inventory`")
    ap.add_argument("--rebuild", action="store_true", help="force re-parse the archive")
    a = ap.parse_args()

    raw = find_raw(a.path)
    s = load_or_build(raw, rebuild=a.rebuild)

    if a.cmd == "json" or (a.json and a.cmd in ("counters", "inventory")):
        obj = s if a.cmd == "json" else s.get(a.cmd if a.cmd != "counters" else "counters", {})
        json.dump(obj, sys.stdout, indent=1)
        print()
    elif a.cmd == "summary":
        p_summary(s)
    elif a.cmd == "counters":
        p_counters(s)
    elif a.cmd == "device":
        p_device(s)
    elif a.cmd == "inventory":
        for k, v in s.get("inventory", {}).items():
            print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
