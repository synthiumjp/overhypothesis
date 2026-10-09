#!/usr/bin/env python3
"""
study3_ca_sensitivity.py  |  EXPLORATORY, not registered (found after Amendment 5 was locked).

The registered cleaner (cds_burstiness.clean_chat) splits a word wherever a
transcription mark sits inside it: overlap brackets (wal⌈k -> wal, k) and the
CHAT lengthening colon (do:g -> do, g). Such words then never match a target
noun. Conversation transcripts (CABank) use these marks far more than CHILDES.

This script rebuilds CHILDES and CABank with in-word marks removed first, then
runs the registered measure and rates steps unchanged. Pile and synthetic
text have no such marks and are reused as they are. Registered results stay
the primary results; this is a robustness check reported next to them.

Usage (repo root, same environment as the Study 3 run):
  python scripts/study3_ca_sensitivity.py --childes_zip data/cds/Eng-NA.zip \\
      --cabank_zips ~/cabank_zips/SBCSAE.zip ~/cabank_zips/CallHome-eng.zip \\
                    ~/cabank_zips/CallFriend-eng-n.zip ~/cabank_zips/CallFriend-eng-s.zip \\
      --pile data/cds/pile10k.jsonl --outdir results/study3x/ca_sensitivity
"""

import argparse
import importlib.util
import os
import re
from argparse import Namespace


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IN_WORD = re.compile(r"(?<=[^\W\d_])[⌈⌉⌊⌋〔〕:]+(?=[^\W\d_])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--childes_zip")
    ap.add_argument("--cabank_zips", nargs="+", required=True)
    ap.add_argument("--pile", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    s3 = load("study3_extension", "scripts/study3_extension.py")
    cds = s3.cds
    registered = cds.clean_chat
    cds.clean_chat = lambda text: registered(IN_WORD.sub("", text))

    inputs = []
    if a.childes_zip:
        out = f"{a.outdir}/childes_alt.jsonl"
        cds.prepare_childes(Namespace(zip=a.childes_zip, out=out))
        inputs.append(out)
    out = f"{a.outdir}/cabank_alt.jsonl"
    s3.prepare_cabank(Namespace(zips=[os.path.expanduser(z) for z in a.cabank_zips], out=out))
    inputs += [out, a.pile]

    counts = f"{a.outdir}/counts_alt.csv"
    cds.measure(Namespace(inputs=inputs, out=counts))
    s3.rates(Namespace(counts=counts, out=f"{a.outdir}/rates_alt.csv"))
    print("Compare with results/study3x/rates.csv (registered cleaner).")


if __name__ == "__main__":
    main()
