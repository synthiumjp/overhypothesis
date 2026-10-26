#!/usr/bin/env python3
"""check_tied.py  |  Reports whether each model_final.pt under a folder has identical input and output embeddings.

  python scripts/check_tied.py ~/ckpt_a11_tied
"""
import sys
from pathlib import Path

import torch

n_tied = n = 0
for f in sorted(Path(sys.argv[1]).expanduser().rglob("model_final.pt")):
    sd = torch.load(f, map_location="cpu", weights_only=False)["model_state"]
    tied = torch.equal(sd["tok_embed.weight"], sd["lm_head.weight"])
    n += 1
    n_tied += tied
    print(f"{'TIED  ' if tied else 'UNTIED'} {f}")
print(f"{n_tied} of {n} models tied")
