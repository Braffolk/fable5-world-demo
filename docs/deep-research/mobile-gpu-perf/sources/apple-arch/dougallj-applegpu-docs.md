# Apple G13 GPU Architecture (dougallj applegpu docs)

Source: https://dougallj.github.io/applegpu/docs.html

## Register File and Occupancy

The G13 architecture allocates **up to 128 general-purpose registers (GPRs)** per SIMD-group, accessible as `r0` through `r127`. Each register stores one 32-bit value per thread. Registers can be accessed as:
- 32-bit: `r0`–`r127`
- 16-bit low: `r0l`–`r127l`
- 16-bit high: `r0h`–`r127h`

"Using fewer registers (e.g. by using 16-bit types instead of 32-bit types) allows more SIMD-groups to fit in the physical register file (higher occupancy), which improves performance."

## Execution Model

**SIMD-group width**: 32 threads per SIMD-group with a shared 32-bit execution mask tracking active/inactive threads per-lane.

"A CPU SIMD-lane is a Metal thread, and a CPU thread is a Metal SIMD-group."

## Instruction Encoding

Instructions vary in **2-byte multiples, up to 12 bytes**. A long/short encoding bit (`L`) indicates whether the final 2–4 bytes are omitted (read as zero).

## Floating-Point Support

Supports both **16-bit (half) and 32-bit** floating-point ops:
- `fmadd` / `fmadd16` (fused multiply-add)
- `fadd` / `fadd16`
- `fmul` / `fmul16`

Float sources include a **modifier field** enabling absolute value or negation per operand.

## Register Cache

"The `cache` hint indicates the value will be used again, and should be cached. The `discard` hint invalidates the value in the register cache after all operands have been read."

## Key Constraints

- Accessing register numbers beyond allocated count "may read or corrupt data from other SIMD-groups"
- `r0l` is reserved for execution mask stack tracking
- `r1` serves as the link register
