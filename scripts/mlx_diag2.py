#!/usr/bin/env python3
"""
mlx_diag2.py  |  Step-by-step 70B load on the GPU (the scorer's own route), to
find the exact step that raises the Metal GPU timeout.

Prints every large tensor as it is materialised, active memory, and a full
traceback on failure. --keep_bf16 skips the float16 conversion, to test
whether the conversion is the failing step.

Usage:  python scripts/mlx_diag2.py Qwen/Qwen2.5-72B [--keep_bf16]
"""
import sys
import time
import traceback

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load

mid = sys.argv[1]
keep = "--keep_bf16" in sys.argv
gb = lambda: round(mx.get_active_memory() / 1e9, 1)
print("mlx", mx.__version__, "| keep_bf16" if keep else "| convert to float16", flush=True)
try:
    t = time.time()
    model, tok = load(mid, lazy=True)
    print(f"lazy load {time.time() - t:.1f}s", flush=True)
    if not keep:
        model.set_dtype(mx.float16)
    params = tree_flatten(model.parameters())
    big = 0
    for n, (name, w) in enumerate(params, 1):
        t = time.time()
        mx.eval(w)
        dt = time.time() - t
        if w.nbytes > 1e9 or dt > 1 or n % 100 == 0:
            print(f"  {n}/{len(params)} {name} {tuple(w.shape)} {w.dtype} {w.nbytes/1e9:.2f}GB "
                  f"in {dt:.2f}s, active {gb()}GB", flush=True)
    print(f"all weights ready, active {gb()}GB", flush=True)
    ids = mx.array([tok.encode("Here is a dax. It is round and blue. Here is a dax.")])
    t = time.time()
    mx.eval(model(ids))
    print(f"forward 1 sentence {time.time() - t:.2f}s", flush=True)
    t = time.time()
    mx.eval(model(mx.concatenate([ids] * 4)))
    print(f"forward 4 sentences {time.time() - t:.2f}s", flush=True)
    print("OK", flush=True)
except Exception:
    traceback.print_exc()
    print(f"FAILED at active {gb()}GB", flush=True)
