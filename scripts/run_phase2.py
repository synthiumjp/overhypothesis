"""
Phase 2 — Seed-Sweep Runner (Study 1)
Orchestrates all 155 pre-registered training + evaluation runs.

Run breakdown (per PROJECT_PLAN v1.5):
  Primary:       8 conditions × 3 sizes (tiny/small/medium) × 5 seeds = 120 runs
  Sequential:    feature_swap_seq × medium × 5 seeds                  =   5 runs
  Dose-response: regular × 3 sizes × 5 seeds × 2 fractions (25%,50%) =  30 runs
                                                                 Total = 155 runs

Usage:
    # Full Phase 2 (all 155 runs)
    python scripts/run_phase2.py

    # Dry run — print all commands without executing
    python scripts/run_phase2.py --dry_run

    # Resume from where you left off (skips completed runs)
    python scripts/run_phase2.py --resume

    # Run only specific conditions (for debugging / partial runs)
    python scripts/run_phase2.py --conditions regular scrambled

    # Run only specific sizes
    python scripts/run_phase2.py --sizes tiny small

    # Run only specific seeds
    python scripts/run_phase2.py --seeds 42 123

    # Skip evaluation (train only)
    python scripts/run_phase2.py --train_only

    # Skip training (eval only — requires existing checkpoints)
    python scripts/run_phase2.py --eval_only
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


# ── Pre-registered constants ─────────────────────────────────────────────────

SEEDS      = [42, 123, 456, 789, 1001]
SIZES      = ["tiny", "small", "medium"]   # Micro is ceiling pilot only
CONDITIONS = [
    "regular", "scrambled", "bare_nolabel", "paraphrased_nolabel",
    "weak_label", "feature_swap", "noise_injection", "frequency_matched",
]
DOSE_FRACTIONS = [0.25, 0.50]  # dose-response on Regular only

# Directories (defaults — match existing project structure)
CHECKPOINT_DIR = Path("models/checkpoints")
RESULTS_DIR    = Path("results/study1")
TOKENIZER_DIR  = Path("data/tokenizers")
CORPUS_DIR     = Path("data/corpora")
WUG_TEST       = Path("data/wug_tests/wug_test_seed42.csv")
PROGRESS_FILE  = Path("data/phase2_progress.json")


# ── Run configuration builder ────────────────────────────────────────────────

def build_run_configs(conditions=None, sizes=None, seeds=None,
                      include_dose=True, include_sequential=True):
    """
    Build the full list of 155 run configurations.
    Each config is a dict with keys used to construct CLI commands.
    """
    conditions = conditions or CONDITIONS
    sizes      = sizes or SIZES
    seeds      = seeds or SEEDS
    runs       = []

    # Primary: 8 conditions × 3 sizes × 5 seeds = 120
    for cond in conditions:
        for size in sizes:
            for seed in seeds:
                runs.append({
                    "condition": cond,
                    "size": size,
                    "seed": seed,
                    "corpus_fraction": 1.0,
                    "run_type": "primary",
                })

    # Dose-response: Regular × 3 sizes × 5 seeds × 2 fractions = 30
    if include_dose and "regular" in conditions:
        for frac in DOSE_FRACTIONS:
            for size in sizes:
                for seed in seeds:
                    runs.append({
                        "condition": "regular",
                        "size": size,
                        "seed": seed,
                        "corpus_fraction": frac,
                        "run_type": "dose_response",
                    })

    # Sequential feature-swap: medium only × 5 seeds = 5
    # NOTE: Sequential training (Domain A first, then B appended) requires
    # a separate corpus variant. This is flagged as a placeholder — the
    # corpus generator needs a --sequential mode. For now we include the
    # config entries so the count is correct and they can be implemented
    # when the sequential corpus is ready.
    if include_sequential and "feature_swap" in conditions and "medium" in sizes:
        for seed in seeds:
            runs.append({
                "condition": "feature_swap",
                "size": "medium",
                "seed": seed,
                "corpus_fraction": 1.0,
                "run_type": "sequential",
                "_sequential": True,  # flag for special handling
            })

    return runs


def run_id(cfg):
    """Unique string identifier for a run."""
    frac_tag = f"_frac{int(cfg['corpus_fraction']*100)}" if cfg['corpus_fraction'] < 1.0 else ""
    seq_tag  = "_seq" if cfg.get("_sequential") else ""
    return f"{cfg['condition']}_{cfg['size']}_seed{cfg['seed']}{frac_tag}{seq_tag}"


# ── Progress tracking ────────────────────────────────────────────────────────

def load_progress(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_train": [], "completed_eval": [], "failed": []}


def save_progress(progress, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ── Command builders ─────────────────────────────────────────────────────────

def build_train_cmd(cfg):
    """Build the train_model.py command for a single run."""
    cmd = [
        sys.executable, "scripts/train_model.py",
        "--condition",       cfg["condition"],
        "--size",            cfg["size"],
        "--seed",            str(cfg["seed"]),
        "--corpus_dir",      str(CORPUS_DIR),
        "--tok_dir",         str(TOKENIZER_DIR),
        "--output_dir",      str(CHECKPOINT_DIR),
    ]
    if cfg["corpus_fraction"] < 1.0:
        cmd += ["--corpus_fraction", str(cfg["corpus_fraction"])]
    return cmd


def build_eval_cmd(cfg):
    """Build the evaluate.py command for a single run."""
    rid = run_id(cfg)
    model_path = CHECKPOINT_DIR / rid / "model_final.pt"
    tok_path   = TOKENIZER_DIR / f"{cfg['condition']}_tokenizer.json"

    cmd = [
        sys.executable, "scripts/evaluate.py",
        "--model",     str(model_path),
        "--wug_test",  str(WUG_TEST),
        "--tokenizer", str(tok_path),
        "--output",    str(RESULTS_DIR),
    ]
    return cmd


# ── Execution ────────────────────────────────────────────────────────────────

def execute_run(cmd, rid, phase, dry_run=False):
    """Execute a single command, return (success, elapsed_seconds)."""
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return True, 0.0

    print(f"\n{'='*70}")
    print(f"  {phase}: {rid}")
    print(f"  CMD: {cmd_str}")
    print(f"{'='*70}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            # Stream output directly to console
            stdout=None,
            stderr=None,
        )
        elapsed = time.time() - t0
        print(f"  ✓ {phase} complete: {rid} ({elapsed:.1f}s)")
        return True, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"  ✗ {phase} FAILED: {rid} (exit code {e.returncode}, {elapsed:.1f}s)")
        return False, elapsed
    except KeyboardInterrupt:
        print(f"\n  ⚠ Interrupted during {rid}")
        raise


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 seed-sweep runner — all 155 pre-registered runs")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip already-completed runs")
    parser.add_argument("--train_only", action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--eval_only",  action="store_true",
                        help="Skip training (requires existing checkpoints)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Subset of conditions to run")
    parser.add_argument("--sizes",      nargs="+", default=None,
                        help="Subset of sizes to run")
    parser.add_argument("--seeds",      nargs="+", type=int, default=None,
                        help="Subset of seeds to run")
    parser.add_argument("--no_dose",    action="store_true",
                        help="Skip dose-response runs")
    parser.add_argument("--no_sequential", action="store_true",
                        help="Skip sequential feature-swap runs")
    parser.add_argument("--progress_file", type=Path, default=PROGRESS_FILE,
                        help="Path to progress tracking JSON")
    args = parser.parse_args()

    # Build run configs
    runs = build_run_configs(
        conditions=args.conditions,
        sizes=args.sizes,
        seeds=args.seeds,
        include_dose=not args.no_dose,
        include_sequential=not args.no_sequential,
    )

    # Count by type
    primary = sum(1 for r in runs if r["run_type"] == "primary")
    dose    = sum(1 for r in runs if r["run_type"] == "dose_response")
    seq     = sum(1 for r in runs if r["run_type"] == "sequential")

    print("=" * 70)
    print("PHASE 2 SEED-SWEEP RUNNER")
    print(f"  Total runs:     {len(runs)}")
    print(f"    Primary:      {primary}")
    print(f"    Dose-response: {dose}")
    print(f"    Sequential:   {seq}")
    print(f"  Train: {'skip' if args.eval_only else 'yes'}")
    print(f"  Eval:  {'skip' if args.train_only else 'yes'}")
    print(f"  Mode:  {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 70)

    # Load progress
    progress = load_progress(args.progress_file) if args.resume else {
        "completed_train": [], "completed_eval": [], "failed": []
    }

    # Track timing
    total_train_time = 0.0
    total_eval_time  = 0.0
    n_completed      = 0
    n_failed         = 0
    n_skipped        = 0
    t_global         = time.time()

    for i, cfg in enumerate(runs):
        rid = run_id(cfg)

        # Skip sequential runs (placeholder — not yet implemented)
        if cfg.get("_sequential"):
            print(f"\n  [{i+1}/{len(runs)}] SKIP (sequential not yet implemented): {rid}")
            n_skipped += 1
            continue

        # ── Train ────────────────────────────────────────────────────────
        if not args.eval_only:
            if args.resume and rid in progress["completed_train"]:
                print(f"\n  [{i+1}/{len(runs)}] SKIP (already trained): {rid}")
            else:
                print(f"\n  [{i+1}/{len(runs)}] TRAINING: {rid}")
                cmd = build_train_cmd(cfg)
                ok, elapsed = execute_run(cmd, rid, "TRAIN", args.dry_run)
                total_train_time += elapsed

                if ok:
                    if not args.dry_run:
                        progress["completed_train"].append(rid)
                        save_progress(progress, args.progress_file)
                else:
                    n_failed += 1
                    if not args.dry_run:
                        progress["failed"].append({"id": rid, "phase": "train",
                                                    "time": datetime.now().isoformat()})
                        save_progress(progress, args.progress_file)
                    continue  # skip eval if train failed

        # ── Eval ─────────────────────────────────────────────────────────
        if not args.train_only:
            if args.resume and rid in progress["completed_eval"]:
                print(f"  SKIP (already evaluated): {rid}")
            else:
                # Check model exists
                model_path = CHECKPOINT_DIR / rid / "model_final.pt"
                if not model_path.exists() and not args.dry_run:
                    print(f"  ✗ SKIP eval (no model found): {model_path}")
                    n_failed += 1
                    continue

                cmd = build_eval_cmd(cfg)
                ok, elapsed = execute_run(cmd, rid, "EVAL", args.dry_run)
                total_eval_time += elapsed

                if ok:
                    n_completed += 1
                    if not args.dry_run:
                        progress["completed_eval"].append(rid)
                        save_progress(progress, args.progress_file)
                else:
                    n_failed += 1
                    if not args.dry_run:
                        progress["failed"].append({"id": rid, "phase": "eval",
                                                    "time": datetime.now().isoformat()})
                        save_progress(progress, args.progress_file)

        # ETA estimate
        if not args.dry_run and (n_completed + n_failed) > 0:
            elapsed_total = time.time() - t_global
            runs_done     = n_completed + n_failed + n_skipped
            runs_left     = len(runs) - runs_done
            if runs_done > n_skipped:
                avg_per_run = elapsed_total / (runs_done - n_skipped)
                eta_s       = avg_per_run * runs_left
                eta_str     = str(timedelta(seconds=int(eta_s)))
                print(f"  ETA: ~{eta_str} ({runs_left} runs remaining)")

    # ── Final summary ────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_global
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print(f"  Total runs:       {len(runs)}")
    print(f"  Completed:        {n_completed}")
    print(f"  Failed:           {n_failed}")
    print(f"  Skipped:          {n_skipped}")
    print(f"  Train time:       {total_train_time:.1f}s ({total_train_time/3600:.1f}h)")
    print(f"  Eval time:        {total_eval_time:.1f}s ({total_eval_time/3600:.1f}h)")
    print(f"  Total wall time:  {total_elapsed:.1f}s ({total_elapsed/3600:.1f}h)")
    if args.resume:
        print(f"  Progress file:    {args.progress_file}")
    print("=" * 70)

    if n_failed > 0:
        print(f"\n⚠  {n_failed} runs failed. Check progress file for details.")
        print(f"   Re-run with --resume to retry failed runs.")


if __name__ == "__main__":
    main()
