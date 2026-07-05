To get a Metal trace from the browser:

```
DAWN_TRACE_FILE_BASE=/tmp/laas_trace \
DAWN_TRACE_DEVICE_FILTER=laas-render \
MTL_CAPTURE_ENABLED=1 \
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
--disable-gpu-sandbox --user-data-dir=/tmp/chrome-metal \
--disable-features=SkiaGraphite \
--no-first-run --no-default-browser-check \
--enable-dawn-features=use_user_defined_labels_in_backend,disable_symbol_renaming \
'http://localhost:5173/?scene=world&nanite=1&dpr=2&clhw=1&shadowclipres=896&clhwmax=16&grass=0'
```
