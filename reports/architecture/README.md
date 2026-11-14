# Architecture verification

`parameter-counts.json` records the target and executable counts for both
implemented model bodies. Reproduce the check with:

```powershell
python tools/count_params.py
```

The models are created on PyTorch's meta device for counting, so the check does
not allocate their full weight tensors. A parameter-count match is an
architecture-shape check. It is not a performance, training, or quality result.
The E-300-1 pilot runner compares decoder-only, prefix-LM, and encoder-decoder
candidates on paired local data. It has only been smoke-tested on synthetic
CPU fixtures; the real architecture decision remains open.
