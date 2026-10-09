#!/usr/bin/env python3
"""
mlx_diag.py  |  Find where a 70B MLX load fails with a Metal GPU timeout.

Times each stage separately: lazy load, weight conversion and materialisation
on the CPU (so the GPU never has to page weights in from disk inside one
command), then a forward pass of 1 and 4 short sentences on the GPU.

Usage:  python scripts/mlx_diag.py Qwen/Qwen2.5-72B
"""
import sys
import time

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from transformers import AutoTokenizer

mid = sys.argv[1]
print("mlx", mx.__version__, flush=True)
t = time.time()
model, _ = load(mid, lazy=True)
print(f"1 lazy load: {time.time() - t:.1f}s", flush=True)

t = time.time()
with mx.stream(mx.cpu):
    model.set_dtype(mx.float16)
    params = tree_flatten(model.parameters())
    for n, (_, w) in enumerate(params, 1):
        mx.eval(w)
        if n % 100 == 0:
            print(f"  {n}/{len(params)} tensors", flush=True)
print(f"2 weights converted on CPU: {time.time() - t:.1f}s", flush=True)

tok = AutoTokenizer.from_pretrained(mid)
ids = mx.array([tok("Here is a dax. It is round and blue. Here is a dax.")["input_ids"]])
t = time.time()
mx.eval(model(ids))
print(f"3 GPU forward, 1 sentence: {time.time() - t:.2f}s", flush=True)
t = time.time()
mx.eval(model(mx.concatenate([ids] * 4)))
print(f"4 GPU forward, 4 sentences: {time.time() - t:.2f}s", flush=True)
print("OK", flush=True)
