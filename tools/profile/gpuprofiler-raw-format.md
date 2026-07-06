# Reverse-engineering the exported `.gpuprofiler_raw` (GPRWCNTR + raw parsing)

Working notes on the binary format inside an Xcode-exported profiled `.gputrace`.
Goal: extract **real per-instruction / per-source-line** GPU cost headlessly. This
file is the ground truth so the RE survives context compaction. Verified
2026-07-06 on macOS 26.4 / Xcode 26.6, Apple **M1 Max** (gpu_gen 13, var 'C', 32 cores).

Scratch scripts used to derive this live in the session scratchpad (`probe_re.py`,
`probe_pc.py`, `extract_gprw.py`); the load-bearing code is reproduced below.

---

## 0. Artifacts & the big picture

- **raw `.gputrace`** (our `gputrace.sh`): command stream + `MTLBuffer/Texture` dumps. No timing.
- **Xcode "Profile GPU Trace → Export"** → a *profile-only* bundle whose
  `NAME.gputrace.gpuprofiler_raw` is a **~2.7 GB `NSKeyedArchiver` binary plist** (`streamData`).
  This holds the counters + the per-instruction execution trace.
- To feed `tmc/gputrace`, graft the plist as `<raw>/x.gpuprofiler_raw/streamData` (see `gputrace_timing.sh`).

**Two decode engines, both partially blocked (see §7):**
1. `GTShaderProfiler.framework` (via `tmc/gputrace/internal/agxps` → `tmc/apple` bindings) —
   the real engine, but its offline parser **rejects every synthetic GPU** (see §7a).
2. Pure-Go GPRWCNTR reader in `tmc/gputrace/internal/counter/streamdata.go` — only decodes
   28 of each record and uses the **wrong record size** (168; real is 64 — see §4).

So we parse `streamData` ourselves (Python `plistlib` + a hand NSKeyedUnarchiver).

---

## 1. NSKeyedUnarchiver (the reusable decoder)

`streamData` is `bplist00` → `{$version,$archiver:NSKeyedArchiver,$top:{root},$objects:[…]}`.
`plistlib.load(fmt=FMT_BINARY)` gives the raw archive; resolve UIDs yourself:

```python
import plistlib; UID = plistlib.UID
def unarchive(arch):
    objs = arch['$objects']; cache = {}
    def res(n):
        if isinstance(n, UID):
            i = n.data
            if i in cache: return cache[i]
            cache[i] = None                     # cycle guard
            v = ro(objs[i]); cache[i] = v; return v
        return ro(n)
    def ro(o):
        if isinstance(o, dict):
            cls = objs[o['$class'].data].get('$classname') if '$class' in o and isinstance(o['$class'],UID) else None
            if cls in ('NSDictionary','NSMutableDictionary'): return {res(k):res(v) for k,v in zip(o['NS.keys'],o['NS.objects'])}
            if cls in ('NSArray','NSMutableArray','NSSet','NSMutableSet'): return [res(v) for v in o['NS.objects']]
            if cls in ('NSString','NSMutableString'): return o.get('NS.string')
            if cls in ('NSData','NSMutableData'): return o.get('NS.data')   # -> bytes
            return {k:res(v) for k,v in o.items() if k!='$class' and not k.startswith('$')}
        return o
    return res(arch['$top']['root'])
```

Loading the full 2.7 GB takes ~30–90 s and ~6–10 GB RAM.

---

## 2. Root object — `GTMutableShaderProfilerStreamData`

Named fields on the root (via `deref($top.root)`), the useful ones:

| field | meaning |
|---|---|
| `deviceInfo` | `{version:'26.4', platform:1, metalVersion:'372.16', name:'…MacBook Pro', build}` |
| `gpuTimelineData` | **NSArray[533]** — each a nested `NSKeyedArchiver` bplist (per-timeline-segment) |
| `shaderProfilerData` | **NSArray[497]** — each a nested bplist; the **last/biggest is the aggregate** |
| `encoderInfoData` | fixed-stride table, `encoderInfoSize` (=40) → 1763 records |
| `gpuCommandInfoData`/`commandBufferInfoData`/`pipelineStateInfoData`/`functionInfoData` | ditto (sizes in sibling `*Size`) |
| `strings` | pool: `['dawn_entry_point_<hex>', '/program_source', 'main0']` (Dawn names all kernels `main0`) |
| `numBlitCalls`, `captureRangeLocation/Length`, `unixTimestamp`, `version`(=5), `gpuGeneration`(=1) | misc |
| `profiledProfilerMode`/`profiledExecutionMode`/`profiledPerformanceState` | all **0** here |

Each nested array element is itself a bplist: `plistlib.loads(deref(elem)['NS.data'], fmt=FMT_BINARY)` → `unarchive(...)`.
Small entries (~180 KB) carry the counter dictionary + config only; the **aggregate**
entry (~150 MB, the largest by `NS.data` len) carries the real sample data.

---

## 3. The aggregate `shaderProfilerData` entry (fields)

```
Timebase                : [125, 3]     # Apple-Silicon mach timebase → ns = ticks*125/3
Num Encoders            : 1763
Subdivided Dictionary   : {passList: [11 passes]}
Derived Counter Sample Data : NSArray[11]  # 11 GPRWCNTR blobs (one per pass)  → §4
Encoder Infos           : NSArray[11]  # 11 blobs of 14104 B = the PC trace     → §5
Encoder Sample Index Data : bytes 28208 = 1763 × 16 B (one per encoder)         → §6
Encoder Time Sample Data : bytes 0     # EMPTY in shaderProfilerData (it's in gpuTimelineData)
Encoder Tile Info       : bytes 14104
Source Sample Marker    : int 0x52544e4357525047  == b'GPRWCNTR' (just the magic, LE)
Counter Info            : {counterName: index}   # e.g. GRC_SOURCE_ID: 0, GRC_KICK_SLOT_IDX, hashes
Derived Counters Info Data : {DerivedCounterDictionary, DerivedCounterScript, …ConfigurationVariables}
Perf Info               : {AFR:1, DCS:3, SOC0:1, FAB:1}
MTLFX TraceIds          : bytes 261 (a nested bplist)
```

`DerivedCounterConfigurationVariables` = `{gpu_gen:13, gpu_var:'C', num_cores:32, num_frags:32, num_gps:16, num_mgpus:4, …}` → **M1 Max**.

---

## 4. GPRWCNTR record format  ⭐ (the pure-Go parser gets this WRONG)

Each `Derived Counter Sample Data` blob starts with ASCII **`GPRWCNTR`** then fixed records.
`tmc/gputrace` assumes **168-byte** records and decodes only `[0:28]`. The real period is
**64 bytes** — the magic recurs every 64 B (u64[7] and u64[15] both == the magic). Layout:

```
offset  size  field
 0x00    8    "GPRWCNTR" magic (per record)
 0x08    8    timestamp   (u64, GPU ticks, ~4.6e12)
 0x10    8    size        (u64, ~10K–300K, varies)
 0x18    8    count       (u64, small: 1,5,…)
 0x20    8    valueA      (u64)  ← the PROGRAM COUNTER (see §5)
 0x28    8    valueB      (u64)  ← second address/counter
 0x30    8    0
 0x38    8    0
```

Decode: `struct.unpack('<8Q', rec)`; step 64 B. (Caveat: the ~40 derived counters listed in
`Counter Info` do NOT all fit in these 5 slots — the fuller per-counter series may live in the
`gpuTimelineData` array or another blob; not yet fully mapped.)

---

## 5. `Encoder Infos` = the per-instruction PC execution trace  ⭐⭐

11 blobs (one per pass), each **14104 B = 3526 × u32**. The u32 stream is the **program
counter sampled over time**:

- Non-monotonic — **1016 backward steps** out of 3525 in one blob (loops/branches). A
  timestamp can't go backward → this is a **PC**, not a clock.
- Base `0x1d3f3ed8`, **span ≈ 3853** in a single blob → ~3853-instruction window = one
  shader's code range; `PC - base` = **instruction index** within that shader.
- Histogram(PC) = **per-instruction hotness** (relative time per instruction). Hot PCs
  recur (e.g. base+1914 hit 67×; base+0 = a loop top).

```python
import struct
from collections import Counter
hot = Counter()
for blob in A['Encoder Infos']:                    # A = unarchived aggregate entry
    u = struct.unpack(f'<{len(blob)//4}I', blob[:len(blob)//4*4])
    hot.update(u)
base = min(hot)
for pc, c in hot.most_common(20):
    print(f'+{pc-base:6d}  {c:5d} samples')         # per-instruction hotness
```

`GPRWCNTR.valueA` (§4) carries the same PC quantity. `Encoder Sample Index Data` (§6) maps
each encoder to its slice of the PC stream, so this can be split **per shader/encoder**.

---

## 6. `Encoder Sample Index Data` — per-encoder sample ranges

`28208 B = 1763 × 16 B`, one record per encoder: `(u32 a, u32 packed, u32 encoderID, u32 0)`,
e.g. `(0, 0x20002, 52, 0)`. `encoderID` matches `encoderInfoData`. Use this to attribute PC
samples (§5) to a specific encoder → specific pipeline/shader.

---

## 7. Getting to PC → **MSL source line** (the last mile)

Xcode's Shaders tab shows the real per-line MSL (tint-generated `program_source`, e.g.
`nodeVar369 = as_type<int>(…)`), keyed in its sidebar by `MTLLibrary <ptr> <HASH>` (e.g.
`AB67449FDE68844E`). So the mapping **exists in this trace** — the chain is:

```
PC (Encoder Infos, base-relative)  →  instruction index in a shader
   →  [metallib line table]  →  MSL source line  →  program_source text (embedded, /program_source ×162)
```

The shader binaries are `MTLB` (Metal library) blobs in the raw `.gputrace/capture`
(`gputrace mtlb` reports Version 163841 = 0x28001, FunctionTable 0x58, StringTable 0x158a).
The **line table lives in the metallib's AIR bitcode debug metadata** (Xcode reads it) —
there are **no standalone debug tags** in the capture (`DEBUGCOMPILEUNIT`/`DEBI`/`DWARF`/
`HSRD`/`SORC` all count 0). The library HASH from Xcode's sidebar (`AB67449FDE68844E`) appears
once in the capture at ~offset 3.5 MB inside a serialized-object region (near `CU<b>Ut` type
sigs), i.e. the library is stored as a Dawn/Metal capture object, not a clean standalone MTLB.

⚠️ **THE WALL (version skew, evidenced 6 ways):** EVERY standalone MetalToolchain parser
rejects Dawn's metallib with **"bad major version number"** — `metal-objdump`, `metal-nm`,
`metal-readobj`, `air-nm`, `air-objdump`, and `metal-dsymutil` ("Invalid data … parsing").
The installed MetalToolchain is **v17.6.109**; Dawn compiled the metallib with the **runtime
Metal 372.16**, which is newer, so the AOT toolchain can't read the container. Xcode 26.6's
GPU-debugger has a newer parser (that's why the GUI shows the lines); it is not exposed as a
standalone CLI, and the framework route needs the live GPU (§7a). So on this box, no headless
tool can parse the line table — the mapping is present but currently unreadable offline.

Routes to actually get PC→line: (1) a MetalToolchain new enough to match runtime Metal 372.16
(if downloadable) → `metal-objdump --source`; (2) emit a toolchain-parseable metallib / debug
line info at compile time (three/Dawn `MTLCompileOptions`), then re-profile; (3) hand-parse the
AIR bitcode debug metadata (LLVM bitcode + AIR debug → line); (4) live-GPU framework bridge (§7a).

### 7a. Why the `GTShaderProfiler.framework` path is blocked (for the record)
`tmc/gputrace/internal/agxps` wraps `GTShaderProfiler.framework` (the GUI's engine). It is
**unwired + stale** (doesn't compile against its own `tmc/apple v0.5.5`). Disassembly
(`otool -tV` on the framework) recovered the ABIs:
- `agxps_aps_descriptor_create` is an **x8-sret struct-fill of defaults** (the purego binding
  is wrong: it thinks `(ptr)→uintptr`). Descriptor layout (104 B): `[0]=GPU  [0x18]=ChunkSize(0x1000)
  [0x30]=-1  [0x58]=maxParseErrors(0x32)`, rest 0.
- `agxps_aps_parser_create(x0=&descriptor)` reads GPU at `[0]`, validates, creates.
- **Wall:** `parser_create` **rejects every synthetic GPU** — 0 accepted across gen 12–16 ×
  variant 0–72 × rev 0–3 (1,460 combos); `aps_gpu_is_supported` false for all. It needs the
  **live `MTLDevice`**, not `agxps_gpu_create(gen,variant,rev)`. That is why the library never
  wired agxps to a command. A live-device bridge (Metal + objc_msgSend) is the only framework route.

---

## 8. ✅ SOLVED — per-shader + per-line cost via COMPILER REMARKS (not PC→line)

The §5/§7 PC-sampling + metallib-line-table path was the wrong tree. **PC-sample hotness is
too coarse to resolve source lines**, and the metallib line table needs the live driver.
The right data was already sitting in the trace: the Apple-GPU compiler (`agc`) records, per
shader, a **telemetry dict + LLVM optimization remarks keyed to `program_source` lines**.
Tool: **`perline_remarks.py`** (see README §5a). Format:

- Each shader compile → a **hex-named `NSKeyedArchiver` bplist** whose top-level dict is the
  per-shader **telemetry**: `ALU/FP32/FP16/INT32/INT16 instruction count`, `Temporary register
  count` (⭐ the Apple-GPU occupancy driver), `Uniform register count`, `Spilled bytes` (⭐),
  `Instruction count`, `Texture reads/writes`, `Threadgroup load/store`, `Branch/Wait`,
  `Device load/store/atomic`, `Compilation time in milliseconds`. Plus a `Remarks` key = the
  per-line YAML.
- The **`Remarks` YAML** = LLVM opt-remarks: `--- !Analysis/!Passed  Pass: asm-printer|
  regalloc|loop-unroll|inline  DebugLoc:{File:program_source, Line:N}  Function: agc.main
  Args:[…]`. Per-line metrics: `INST_` (asm-printer per-basic-block machine-instruction count),
  `NumInstructions/NumSpills/NumReloads/NumStackBytes` (regalloc, **per function** — attributed
  to the function entry line), `UnrollCount` (loop-unroll), inlined `Callee`s.
- The **MSL source** is in **library-hash-named files** (e.g. `AB67449FDE68844E`, head
  `#ifdef __clang__ … #pragma METAL …`) — NOT the metallib. So the `nodeVar…` source Xcode
  shows is a plain file in the trace.
- **Naming:** the capture stream has `Dawn_ShaderModule_{compute,fragment,vertex}_<name>`
  labels (188 of them). `perline_remarks.py` names each remark/source file by the **nearest
  label to its hash in the capture** (proximity heuristic; good but not exact) and dedups
  pipeline-variant recompiles by telemetry fingerprint + name.

Real result (this trace): `compute_atmoMultiScatter` = 224 spilled bytes / 124 temp registers
/ 17681 ALU (spilling in its main fn); `compute_nanVoxScatterB1` hot line 630 =
`nodeVar81 = (v_31 / float3(nodeVar80.w))` at 206 machine-instructions. **Levers = spilled
bytes > 0 and high temp-register count** (occupancy) + high per-line instruction density.

## 8b. What each level still needs (for the record)
- ✅ per-kernel ms: `gputrace_timing.sh` (§ README 5). ✅ per-shader+per-line structural cost:
  `perline_remarks.py` (this section).
- Per-instruction **PC hotness** (§5) is decodable but coarse + over-aggregates across shaders
  (aggregate mixes 11 relative PC spaces) — deprecated in favor of §8. Exact per-line **timing**
  needs the live driver compile (Xcode Shaders tab) or the framework bridge (§7a).
- §7's "metallib version wall" was a **false extraction**: offset 9676 was the string
  `"MTLBuffer-…"`, not a metallib; Dawn doesn't persist a clean metallib blob at all (0 valid
  MTLB in 2.7 GB streamData + device-resources). The toolchain parses v28 metallibs fine (a
  fresh `metal -frecord-sources` compile disassembles with `metal-objdump --source`).

## 8c. ✅ SOLVED — RUNTIME time-per-source-line (what Xcode's Shaders tab shows)

`perline_remarks.py` (§8) is **static** compile-time cost. Actual per-line **time** = hardware PC
sampling during GPU replay — and it is fully reconstructable **headless**. Tool:
**`runtime_perline.py`** (README §5c). Three structures inside the `.gpuprofiler_raw`, joined:

1. **PC samples** — the ~488 per-source entries (keyed `Source | SourceIndex | RingBufferIndex |
   ShaderProfilerData`, one per source × replay iteration) each carry a `ShaderProfilerData` blob
   that is a **`GPRWCNTR`** stream. Its u32 words that fall inside a PAB `mappedAddress` range are
   the **sampled program counters**; their histogram = runtime time. (The PCs are packed among
   VARIABLE-length counter records — do NOT assume the fixed 64-B stride §4 guessed; just scan u32
   and filter by PAB membership. Hot PCs appear as the high-u32 of `0x……00000001` pairs.)
2. **Program Address Buffer** = `Program Address Mappings` (a list, entry ~494 in shaderProfilerData
   / ~531 in gpuTimelineData) — the **per-command load-address table**: 8237 records
   `{binaryUniqueId, mappedAddress, mappedSize, encID, encIndex, type}` (+ a parallel packed
   `Program Address Buffer` bytes blob: triplets `(encID<<32|tag, startOff, endOff)`). The SAME
   binary is mapped at a DIFFERENT `mappedAddress` **per GPU command** (the driver remaps so PC
   samples disambiguate — Apple patents US9799087B2 / US10310830). mappedAddress span 0x5780–0x37d8400.
3. **`Havested Binaries`** (entry[1]) — native Mach-O per `binaryUniqueId`, categories
   `compute kernel / mutiple binary / fragment / vertex … info` (kid = the id, NOT a VA; ~186
   uniq). Each Mach-O carries dense DWARF: `__debug_line` (offset→`program_source` line) +
   `DW_TAG_subprogram` low/high_pc (offset→function, e.g. `mx_worley_noise_float_1`). `dwarfdump`
   reads it directly even though the standalone MetalToolchain (v17.6) can't parse the metallib
   (§7) — the Mach-O DWARF is a separate, readable container built by GPUCompiler Versions/32023.

Pipeline: `PC → smallest containing PAB mappedAddress range → (binaryUniqueId, offset=PC−mappedAddress)
→ Mach-O DWARF → (line, function)`. Sample count per (binary,line) = time. Real result (this trace):
15.2 M samples / 158 shaders; hottest compute `1d29` 25%, `1d80` 21%, `1d65` 13%; e.g. fragment
`1d44` line 123=42% / 133=32%. PC clustering is tight (top PC hit >1000×; noise is uniform).

**line 0** = DWARF "no source line" = compiler glue / register spills+reloads / prologue — real
time; a line-0-heavy shader is **spill/occupancy-bound** (same lever perline_remarks flags).

**Naming (`--src`, `_srcmap.py`).** The Mach-O has no Dawn name; `--src` resolves it via store0
(barrier-`call_line`-verified block match → NodeBuffer-ID label match). Exact for barrier-verified
compute + distinctive shaders; best-effort else. ⭐ EXACT-naming lead for a future pass: the raw
`capture` stores each shader as `[objPtr, tag=0x11c, "Dawn_ShaderModule_<type>_<name>::main"]`, and
that **objPtr == `functionInfoData[i].field[2]`** → **label ↔ functionInfoData is an exact bijection**
(182/205 named). The one missing hop is `binaryUniqueId → functionInfoData`: Program Address Mappings
gives {index, drawCallIndex, encIndex} but NO index-walk through gpuCommandInfoData/pipelineState/
encoderInfoData reproduces the known names (tried 182 field combos) — closing it needs parsing the
Dawn **MTSP command stream** (dispatch→setPipeline→function order). `strings` table holds only
`dawn_entry_point`/`main0`, never the Dawn names.

⚠️ Why the earlier writeup said "15% / UNSOLVED" — THREE bugs, all fixed: (1) **wrong stream** —
`usc sampling address data` (entry 495) is NOT PC samples; it is the load-address **map** serialized
as `[tag, encID, startAddr, 0, endAddr, 0]` records (dumped once per replay iteration), and
`Encoder Infos` is an encID-space trace — the real PCs are the per-source GPRWCNTR blobs. (2) a
**single global base** instead of the per-command mappedAddress. (3) **exact DWARF-row match**
instead of **range containment**. `Encoder Time Sample Data` / `Derived Counter Sample Data` /
`APSData` are all EMPTY in this headless export (APS = the live-GPU §7a path). Ambiguity: the USC
heap reuses mappedAddresses across commands, so a PC can sit in several ranges — the tool takes the
smallest and aggregates; the hot binary dominates. Ground truth = Xcode's Shaders tab (user-eyeball).

## 9. Capture facts to reproduce
- Raw trace used: `/private/tmp/laas_trace-2026-07-06T01-27-42-c000.gputrace`
- Export used: `/private/tmp/exported-with-perf-data.gputrace/…​.gpuprofiler_raw`
- `Timebase = [125,3]` → `ns = ticks * 125 / 3`.
