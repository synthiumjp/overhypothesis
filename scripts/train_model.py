"""
Phase 2 — Model Training Script (Study 1)
Autoregressive transformer decoder trained from scratch on synthetic nonce-word corpora.
No pretrained weights. Supports all 4 model sizes and all 8 conditions.

Usage:
    # Single run (smoke test / ceiling pilot)
    python scripts/train_model.py --condition regular --size micro --seed 42

    # Full Phase 2 run (called by aider for seed sweeps)
    python scripts/train_model.py --condition regular --size tiny --seed 123

    # Dose-response (25% or 50% corpus)
    python scripts/train_model.py --condition regular --size small --seed 42 --corpus_fraction 0.5

Model sizes (from spec §3.1):
    micro:  2L  2H  d=128  ~2M params   (ceiling pilot only)
    tiny:   4L  4H  d=256  ~10M params
    small:  6L  8H  d=384  ~25M params
    medium: 8L  8H  d=512  ~50M params
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

try:
    import torch_directml
    DML_AVAILABLE = torch_directml.is_available()
except ImportError:
    DML_AVAILABLE = False

# ── Model configurations (exactly per spec §3.1) ─────────────────────────────
MODEL_CONFIGS = {
    "micro":  dict(n_layer=2, n_head=2, d_model=128, d_ff=512),
    "tiny":   dict(n_layer=4, n_head=4, d_model=256, d_ff=1024),
    "small":  dict(n_layer=6, n_head=8, d_model=384, d_ff=1536),
    "medium": dict(n_layer=8, n_head=8, d_model=512, d_ff=2048),
}

# ── Training hyperparameters ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    "micro":  dict(lr=3e-4, batch_size=32,  seq_len=64,  max_steps=2000,  warmup=100),
    "tiny":   dict(lr=3e-4, batch_size=32,  seq_len=128, max_steps=5000,  warmup=200),
    "small":  dict(lr=2e-4, batch_size=32,  seq_len=128, max_steps=8000,  warmup=400),
    "medium": dict(lr=1e-4, batch_size=16,  seq_len=128, max_steps=10000, warmup=500),
}

CHECKPOINT_EVERY = 500   # steps
LOG_EVERY        = 100   # steps
SEEDS            = [42, 123, 456, 789, 1001]


# ── Model architecture ────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head  = n_head
        self.d_head  = d_model // n_head
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj    = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Causal mask — registered as buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask.view(1, 1, seq_len, seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # Reshape to (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # Attention
        scale = 1.0 / math.sqrt(self.d_head)
        att   = (q @ k.transpose(-2, -1)) * scale
        att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att   = torch.softmax(att, dim=-1)
        att   = self.dropout(att)
        y     = att @ v
        y     = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, seq_len, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seq_len, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class OverhypothesisLM(nn.Module):
    """
    Autoregressive transformer decoder trained from scratch.
    No pretrained weights — essential per spec §3.1.
    Residual stream accessible at each layer for interpretability probes (Phase 5.5).
    """
    def __init__(self, vocab_size, d_model, n_head, n_layer, d_ff,
                 seq_len=128, dropout=0.1):
        super().__init__()
        self.seq_len    = seq_len
        self.d_model    = d_model
        self.tok_embed  = nn.Embedding(vocab_size, d_model)
        self.pos_embed  = nn.Embedding(seq_len, d_model)
        self.drop       = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, seq_len, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f       = nn.LayerNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_embed.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, return_hidden_states=False):
        B, T = input_ids.shape
        assert T <= self.seq_len, f"Sequence length {T} exceeds max {self.seq_len}"
        pos    = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x      = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))
        hidden = []
        for block in self.blocks:
            x = block(x)
            if return_hidden_states:
                hidden.append(x.detach().cpu())
        x      = self.ln_f(x)
        logits = self.lm_head(x)
        if return_hidden_states:
            return logits, hidden
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Dataset ───────────────────────────────────────────────────────────────────

class CorpusDataset(Dataset):
    def __init__(self, token_ids: list, seq_len: int):
        self.data    = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


class FreshWindowDataset(Dataset):
    """Amendment 8: non-overlapping windows (stride = seq_len) over a numpy token array,
    so that, within one pass through the data, no token is trained on twice."""
    def __init__(self, token_ids, seq_len: int):
        self.data    = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(1, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = np.asarray(self.data[start: start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


class OrderedWindowDataset(Dataset):
    """Amendment 9: window p of the run is text window order[p] (non-overlapping, stride
    seq_len), read in the order given, so the step at which every window is read is fixed."""
    def __init__(self, token_ids, seq_len: int, order):
        self.data    = token_ids
        self.seq_len = seq_len
        self.order   = order

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        start = int(self.order[idx]) * self.seq_len
        chunk = np.asarray(self.data[start: start + self.seq_len + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


def load_and_tokenize(corpus_path: Path, tokenizer: Tokenizer,
                      corpus_fraction: float = 1.0) -> list:
    """Load corpus, tokenise, optionally truncate for dose-response."""
    text = corpus_path.read_text(encoding="utf-8")
    # Dose-response: take first fraction of sentences
    if corpus_fraction < 1.0:
        lines = text.strip().split("\n")
        n     = max(1, int(len(lines) * corpus_fraction))
        text  = "\n".join(lines[:n])
    encoding = tokenizer.encode(text)
    return encoding.ids


def load_and_tokenize_lines(corpus_path: Path, tokenizer: Tokenizer,
                            batch_lines: int = 200_000) -> np.ndarray:
    """Amendment 8: for corpora of tens of millions of tokens. Encodes the file line by
    line in batches into a uint16 array. With the Whitespace pre-tokeniser this gives the
    same ids as encoding the whole text at once (newlines are whitespace)."""
    if tokenizer.get_vocab_size() > 65_535:
        raise SystemExit("vocabulary too large for uint16")
    parts, buf = [], []
    def flush():
        if buf:
            enc = tokenizer.encode_batch(buf)
            parts.append(np.fromiter((i for e in enc for i in e.ids), dtype=np.uint16))
            buf.clear()
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                buf.append(line)
            if len(buf) >= batch_lines:
                flush()
    flush()
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)


# ── Device selection ──────────────────────────────────────────────────────────

def get_device():
    if DML_AVAILABLE:
        return torch_directml.device(), "directml"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    else:
        return torch.device("cpu"), "cpu"


# ── Learning rate schedule (linear warmup + cosine decay) ────────────────────

def get_lr(step, max_steps, warmup_steps, base_lr, min_lr_ratio=0.1):
    min_lr = base_lr * min_lr_ratio
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args, model, dataloader, device, device_type, output_dir, tokenizer):
    cfg        = TRAIN_CONFIG[args.size]
    optimizer  = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        betas=(0.9, 0.95), weight_decay=0.1
    )
    loss_fn    = nn.CrossEntropyLoss()
    vocab_size = tokenizer.get_vocab_size()

    log = {
        "condition": args.condition, "size": args.size, "seed": args.seed,
        "corpus_fraction": args.corpus_fraction,
        "steps": [], "losses": [], "lrs": [],
    }

    model.train()
    data_iter  = iter(dataloader)
    step       = 0
    t_start    = time.time()

    print(f"\nTraining: {args.condition} | {args.size} | seed={args.seed} | "
          f"device={device_type}")
    print(f"Steps: {cfg['max_steps']} | batch: {cfg['batch_size']} | "
          f"seq_len: {cfg['seq_len']} | lr: {cfg['lr']}")
    print("-" * 60)

    while step < cfg["max_steps"]:
        # Refill iterator
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Update learning rate
        lr = get_lr(step, cfg["max_steps"], cfg["warmup"], cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        step    += 1

        if step % LOG_EVERY == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f"  step {step:5d}/{cfg['max_steps']} | "
                  f"loss {loss_val:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            log["steps"].append(step)
            log["losses"].append(round(loss_val, 6))
            log["lrs"].append(round(lr, 8))

        # Amendment 9: weights only, every --save_every steps
        if args.save_every:
            if step % args.save_every == 0:
                torch.save({"step": step, "model_state": model.state_dict()},
                           output_dir / f"weights_step{step:05d}.pt")
            continue

        # Checkpoint every 500 steps
        if step % CHECKPOINT_EVERY == 0 or step == cfg["max_steps"]:
            ckpt_path = output_dir / f"ckpt_step{step:05d}.pt"
            torch.save({
                "step":       step,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "loss":        loss_val,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  → checkpoint saved: {ckpt_path.name}")

    # Save final model (separate from checkpoints — used for eval)
    final_path = output_dir / "model_final.pt"
    torch.save({
        "step":        step,
        "model_state": model.state_dict(),
        "loss":        loss_val,
        "args":        vars(args),
        "vocab_size":  vocab_size,
        "model_cfg":   MODEL_CONFIGS[args.size],
        "train_cfg":   TRAIN_CONFIG[args.size],
        "untied":      bool(args.untie_embeddings),
    }, final_path)
    if args.untie_embeddings:
        sd = model.state_dict()
        if torch.equal(sd["tok_embed.weight"].cpu(), sd["lm_head.weight"].cpu()):
            raise SystemExit("input and output embeddings are identical after training: the untie failed")
    print(f"\nFinal model saved: {final_path}")

    # Save training log
    log["final_loss"] = round(loss_val, 6)
    log["total_time_s"] = round(time.time() - t_start, 2)
    log_path = output_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved: {log_path}")

    return log


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train overhypothesis LM")
    parser.add_argument("--condition", type=str, required=True,
                        choices=["regular", "scrambled", "bare_nolabel",
                                 "paraphrased_nolabel", "weak_label",
                                 "feature_swap", "noise_injection",
                                 "frequency_matched"])
    parser.add_argument("--size",      type=str, required=True,
                        choices=["micro", "tiny", "small", "medium"])
    parser.add_argument("--seed",      type=int, required=True)
    parser.add_argument("--corpus_fraction", type=float, default=1.0,
                        help="Fraction of corpus to use (0.25/0.5/1.0 for dose-response)")
    parser.add_argument("--corpus_dir",  type=Path, default=Path("data/corpora"))
    parser.add_argument("--tok_dir",     type=Path, default=Path("data/tokenizers"))
    parser.add_argument("--output_dir",  type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--smoke_test",  action="store_true",
                        help="Run only 100 steps (smoke test / quick validation)")
    # Amendment 6 (window test). Defaults leave every earlier run unchanged.
    parser.add_argument("--seq_len", type=int, default=None,
                        help="training window in tokens (default: the size's setting, 128)")
    parser.add_argument("--keep_tokens_per_step", action="store_true",
                        help="with --seq_len, scale the batch so tokens per step stay as in the default")
    # Amendment 8 (fresh order). Default leaves every earlier run unchanged.
    parser.add_argument("--fresh_windows", action="store_true",
                        help="encode line by line and train on non-overlapping windows; refuses to "
                             "start if the corpus is too short to supply every step without reuse")
    # Amendment 9. Defaults leave every earlier run unchanged.
    parser.add_argument("--window_order", type=Path, default=None,
                        help="with --fresh_windows: .npy permutation; read position p reads text window order[p], "
                             "in order (no shuffling); the corpus must hold exactly len(order) windows plus one token")
    parser.add_argument("--untie_embeddings", action="store_true",
                        help="give the output layer its own matrix (a copy of the input embedding at "
                             "initialisation) and stop if the two share storage")
    parser.add_argument("--save_every", type=int, default=0,
                        help="save weights only (no optimiser state) every N steps, instead of the "
                             "500-step full checkpoints")
    parser.add_argument("--tokenizer", type=Path, default=None, help="tokenizer file (overrides --tok_dir)")
    parser.add_argument("--corpus_file", type=Path, default=None, help="corpus file (overrides --corpus_dir)")
    parser.add_argument("--run_name", type=str, default=None, help="output folder name (default condition_size_seed)")
    args = parser.parse_args()
    if args.window_order and not args.fresh_windows:
        raise SystemExit("--window_order needs --fresh_windows")
    if args.seq_len:
        base = TRAIN_CONFIG[args.size]
        if args.keep_tokens_per_step:
            tokens = base["batch_size"] * base["seq_len"]
            if tokens % args.seq_len:
                raise SystemExit(f"{tokens} tokens per step not divisible by seq_len {args.seq_len}")
            base["batch_size"] = tokens // args.seq_len
        base["seq_len"] = args.seq_len

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Output directory ──────────────────────────────────────────────────────
    frac_tag = f"_frac{int(args.corpus_fraction*100)}" if args.corpus_fraction < 1.0 else ""
    run_name = args.run_name or f"{args.condition}_{args.size}_seed{args.seed}{frac_tag}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device, device_type = get_device()
    print(f"Device: {device} ({device_type})")

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tok_path = args.tokenizer or args.tok_dir / f"{args.condition}_tokenizer.json"
    if args.tokenizer and not tok_path.exists():
        raise SystemExit(f"tokenizer not found: {tok_path}")
    if not tok_path.exists():
        # Fall back to regular tokeniser (acceptable for smoke test)
        tok_path = args.tok_dir / "regular_tokenizer.json"
        print(f"WARNING: condition tokeniser not found, using regular tokeniser")
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokeniser: {tok_path.name} | vocab_size={vocab_size}")

    # ── Corpus ────────────────────────────────────────────────────────────────
    corpus_path = args.corpus_file or args.corpus_dir / f"{args.condition}_seed{args.seed}.txt"
    if args.corpus_file and not corpus_path.exists():
        raise SystemExit(f"corpus not found: {corpus_path}")
    if not corpus_path.exists():
        # Fall back to seed 42 corpus (smoke test only)
        corpus_path = args.corpus_dir / f"{args.condition}_seed42.txt"
        print(f"WARNING: seed-specific corpus not found, using seed42 corpus")
    print(f"Corpus: {corpus_path.name}")

    if args.fresh_windows:
        if args.corpus_fraction < 1.0:
            raise SystemExit("--fresh_windows does not support --corpus_fraction")
        token_ids = load_and_tokenize_lines(corpus_path, tokenizer)
    else:
        token_ids = load_and_tokenize(corpus_path, tokenizer, args.corpus_fraction)
    print(f"Corpus tokens: {len(token_ids):,} "
          f"(fraction={args.corpus_fraction})")

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    cfg        = TRAIN_CONFIG[args.size]
    seq_len    = cfg["seq_len"]
    order_md5 = None
    if args.window_order:
        import hashlib
        order = np.load(args.window_order)
        order_md5 = hashlib.md5(order.tobytes()).hexdigest()
        if len(token_ids) != len(order) * seq_len + 1:
            raise SystemExit(f"corpus has {len(token_ids):,} tokens; the window order needs exactly "
                             f"{len(order) * seq_len + 1:,}")
        if len(order) != cfg["max_steps"] * cfg["batch_size"]:
            raise SystemExit(f"{len(order):,} windows in the order but {cfg['max_steps'] * cfg['batch_size']:,} "
                             f"are read in training")
        dataset = OrderedWindowDataset(token_ids, seq_len, order)
        print(f"Window order: {args.window_order.name} ({len(order):,} windows, read in order, md5 {order_md5})")
    elif args.fresh_windows:
        dataset = FreshWindowDataset(token_ids, seq_len)
        steps   = 100 if args.smoke_test else cfg["max_steps"]
        needed  = steps * cfg["batch_size"]
        print(f"Fresh windows: {len(dataset):,} available, {needed:,} needed")
        if len(dataset) < needed and not args.smoke_test:
            raise SystemExit("corpus too short for fresh windows: some text would be reused")
    else:
        dataset = CorpusDataset(token_ids, seq_len)
    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"],
        shuffle=not args.window_order, drop_last=True,
        num_workers=0,   # 0 for Windows / DirectML compatibility
    )
    print(f"Dataset: {len(dataset):,} sequences of length {seq_len}")

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg  = MODEL_CONFIGS[args.size]
    model = OverhypothesisLM(
        vocab_size=vocab_size,
        d_model=mcfg["d_model"],
        n_head=mcfg["n_head"],
        n_layer=mcfg["n_layer"],
        d_ff=mcfg["d_ff"],
        seq_len=seq_len,
    )
    if args.untie_embeddings:
        model.lm_head.weight = nn.Parameter(model.tok_embed.weight.detach().clone())
    model = model.to(device)
    if args.untie_embeddings:
        names = dict(model.named_parameters())
        if ("lm_head.weight" not in names or "tok_embed.weight" not in names
                or names["lm_head.weight"] is names["tok_embed.weight"]):
            raise SystemExit("input and output embeddings share storage: stopping")
        print("Embeddings: untied (output layer starts as a copy of the input embedding)")
    n_params = model.count_params()
    print(f"Model: {args.size} | params={n_params:,}")

    # Save model config alongside checkpoint
    cfg_path = output_dir / "model_config.json"
    with open(cfg_path, "w") as f:
        json.dump({
            "condition": args.condition, "size": args.size, "seed": args.seed,
            "corpus_fraction": args.corpus_fraction,
            "vocab_size": vocab_size, "n_params": n_params,
            "model_cfg": mcfg, "train_cfg": cfg,
            "device": device_type,
            **({"fresh_windows": True, "corpus_file": str(corpus_path)} if args.fresh_windows else {}),
            **({"window_order": str(args.window_order), "window_order_md5": order_md5} if args.window_order else {}),
            **({"untied": True, "tokenizer": str(tok_path), "save_every": args.save_every}
               if args.untie_embeddings else {}),
        }, f, indent=2)

    # ── Smoke test override ───────────────────────────────────────────────────
    if args.smoke_test:
        print("\n[SMOKE TEST MODE — 100 steps only]")
        TRAIN_CONFIG[args.size]["max_steps"] = 100

    # ── Train ─────────────────────────────────────────────────────────────────
    log = train(args, model, dataloader, device, device_type, output_dir, tokenizer)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Run: {run_name}")
    print(f"Final loss:  {log['final_loss']:.4f}")
    print(f"Total time:  {log['total_time_s']:.1f}s")
    print(f"Loss trend:  {log['losses'][0]:.4f} → {log['losses'][-1]:.4f}")
    decreasing = log["losses"][-1] < log["losses"][0]
    print(f"Decreasing:  {decreasing}")
    if args.smoke_test:
        print("\nSmoke test PASSED ✓" if decreasing else "\nSmoke test WARNING: loss not decreasing")
    print("=" * 60)


if __name__ == "__main__":
    main()
