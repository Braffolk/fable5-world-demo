# Mali Offline Compiler — Resource Usage (STUB — fetch failed)

Source: https://developer.arm.com/documentation/101863/latest/Using-Mali-Offline-Compiler/Performance-analysis/Resource-usage

STATUS: NOT captured. The Arm Developer documentation page is a client-side-rendered SPA; both WebFetch and curl returned only the ~13 KB HTML shell with no technical content (no "work register", "spill", or "occupancy" text present in the served markup). Content is loaded via JS from an API the fetch tools cannot execute.

## What the page is expected to contain (from Arm Mali Offline Compiler docs, for reference)
The `malioc` report's "Resource usage" section reports, per shader:
- **Work registers used** (Bifrost/Valhall: out of 64; higher counts reduce thread occupancy).
- **Uniform registers used**.
- **Stack spilling** (whether the shader spills registers to stack/memory — a warning to reduce register pressure).
- **16-bit arithmetic %** (portion of arithmetic done in fp16 — higher is better on Mali).
- Occupancy note: work-register count directly gates how many threads can be resident; exceeding the register budget halves occupancy.

TO OBTAIN THE REAL PAGE: use a JS-capable fetch (headless browser) or download the Mali Offline Compiler and run `malioc --help` / read the bundled docs.
