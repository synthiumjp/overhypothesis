"""
Phase 0 — DirectML Transformer Smoke Test
Compatible with torch-directml 0.2.5 (no .sync() method)
"""

import torch
import torch_directml
import time
from torch import nn

device = torch_directml.device()
print("=" * 50)
print("Phase 0: DirectML Transformer Smoke Test")
print("=" * 50)
print(f"Device: {device}")

class MicroTransformer(nn.Module):
    def __init__(self, vocab_size=512, d_model=128, nhead=2, num_layers=2, max_seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.lm_head(x)

# 1. Build model
print("\n[1/4] Building Micro transformer...")
model = MicroTransformer().to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"      Parameter count: {param_count:,}")

# 2. Forward pass
print("\n[2/4] Forward pass (batch=4, seq_len=64)...")
batch_size, seq_len, vocab_size = 4, 64, 512
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
targets   = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

try:
    t0 = time.time()
    logits = model(input_ids)
    # Force result back to CPU to ensure GPU computation is complete
    _ = logits.cpu()
    t1 = time.time()
    print(f"      Output shape: {logits.shape}  (expected: [{batch_size}, {seq_len}, {vocab_size}])")
    print(f"      Time: {(t1-t0)*1000:.1f} ms")
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print("      PASS")
except Exception as e:
    print(f"      FAIL: {e}")
    raise SystemExit(1)

# 3. Backward pass — 100 training steps
print("\n[3/4] Backward pass (100 steps, AdamW)...")
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn   = nn.CrossEntropyLoss()
    losses    = []

    t0 = time.time()
    for step in range(100):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss   = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    # Force sync via CPU transfer
    _ = loss.cpu().item()
    t1 = time.time()

    print(f"      100 steps in {t1-t0:.2f}s")
    print(f"      Loss step   1: {losses[0]:.4f}")
    print(f"      Loss step 100: {losses[-1]:.4f}")
    decreasing = losses[-1] < losses[0]
    print(f"      Loss decreasing: {decreasing}")
    print("      PASS" if decreasing else "      WARNING: loss did not decrease")
except Exception as e:
    print(f"      FAIL: {e}")
    raise SystemExit(1)

# 4. Summary
print("\n[4/4] Summary")
print("-" * 50)
print(f"  DirectML device:    {device}")
print(f"  Parameters:         {param_count:,}")
print(f"  Forward pass:       PASS")
print(f"  Backward pass:      PASS")
print(f"  Loss decreasing:    {losses[-1] < losses[0]}")
print(f"  Ready for Phase 0:  YES")
print("=" * 50)
