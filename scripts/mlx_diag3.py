#!/usr/bin/env python3
"""
mlx_diag3.py  |  Run the scorer's own MLX code (english_oneshot.load_mlx and
seq_logprobs_mlx) on 40 sentences, stage by stage, to find which part raises
the Metal GPU timeout. --no_limits turns the scorer's memory and cache caps
into no-ops, to test whether they cause it.

Usage:  python scripts/mlx_diag3.py Qwen/Qwen2.5-72B [--no_limits]
"""
import importlib.util
import sys
import time
import traceback

import mlx.core as mx

mid = sys.argv[1]
if "--no_limits" in sys.argv:
    for name in ("set_memory_limit", "set_cache_limit"):
        for mod in (mx, getattr(mx, "metal", None)):
            if mod is not None and hasattr(mod, name):
                setattr(mod, name, lambda *a, **k: None)
    print("memory and cache caps disabled", flush=True)

spec = importlib.util.spec_from_file_location("eo", "scripts/english_oneshot.py")
eo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eo)
items = eo.read_items("data/wug_tests/english_items_v2_M.csv")
texts = sorted({r["context"] + " " + q for r in items for q in (r["query_rep"], r["query_new"])})[:40]
gb = lambda: round(mx.get_active_memory() / 1e9, 1)
try:
    t = time.time()
    model, tok = eo.load_mlx(mid, "", "float16", 200)
    print(f"load_mlx done in {time.time() - t:.1f}s, active {gb()}GB", flush=True)
    for bs in (1, 4):
        t = time.time()
        lp = eo.seq_logprobs_mlx(model, tok, texts, bs, 0.05)
        print(f"scored {len(texts)} sentences at batch {bs} in {time.time() - t:.1f}s; first {lp[0]:.3f}", flush=True)
    print("OK", flush=True)
except Exception:
    traceback.print_exc()
    print(f"FAILED at active {gb()}GB", flush=True)
