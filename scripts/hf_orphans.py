#!/usr/bin/env python3
"""
hf_orphans.py  |  Find (and optionally delete) large files in the Hugging Face
shared blob store that no cached model links to any more.

huggingface_hub 1.x keeps downloaded files in <hub>/blobs/<xx>/<sha> and links
them from each model folder. Deleting a model through scan_cache_dir removes
the links but, on this setup, left the files behind. Only files that (1) sit in
the shared store, (2) no model folder links to, (3) are larger than --min_mb and
(4) were modified within the last --hours are listed. Nothing is deleted
without --delete.

Usage:  python scripts/hf_orphans.py --hours 6            # list
        python scripts/hf_orphans.py --hours 6 --delete   # delete those files
"""
import argparse
import glob
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, required=True)
ap.add_argument("--min_mb", type=float, default=100)
ap.add_argument("--delete", action="store_true")
a = ap.parse_args()

from huggingface_hub import constants
hub = constants.HF_HUB_CACHE
linked = {os.path.realpath(p) for p in glob.glob(f"{hub}/models--*/**", recursive=True) if os.path.islink(p)}
now = time.time()
found = []
for f in glob.glob(f"{hub}/blobs/*/*"):
    if not os.path.isfile(f) or os.path.islink(f):
        continue
    st = os.stat(f)
    if (os.path.realpath(f) not in linked and st.st_size > a.min_mb * 1e6
            and now - st.st_mtime < a.hours * 3600):
        found.append((f, st.st_size, st.st_mtime))
total = sum(s for _, s, _ in found) / 1e9
for f, s, m in sorted(found, key=lambda x: x[2]):
    print(f"{time.strftime('%H:%M', time.localtime(m))}  {s/1e9:6.2f} GB  {f}")
print(f"{len(found)} orphaned files, {total:.1f} GB")
if a.delete:
    for f, _, _ in found:
        os.remove(f)
    print(f"deleted {len(found)} files")
