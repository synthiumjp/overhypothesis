#!/usr/bin/env python3
"""trim_tables.py  |  Removes table environments with the given labels from a generated appendix file.
Used after build_appendix_a9.py / build_appendix_a10.py for the submitted version (tables moved to the
supplementary material):

  python paper_acl/tools/trim_tables.py paper_acl/appendix_a9.tex tab:M5
  python paper_acl/tools/trim_tables.py paper_acl/appendix_a10.tex tab:N4 tab:N8 tab:N9 tab:N10 tab:N11
"""
import re
import sys

path, labels = sys.argv[1], set(sys.argv[2:])
t = open(path, encoding="utf-8").read()
out, pos, removed = [], 0, []
for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}\n?", t, re.S):
    lab = re.search(r"\\label\{([^}]*)\}", m.group(0))
    if lab and lab.group(1) in labels:
        out.append(t[pos:m.start()])
        pos = m.end()
        removed.append(lab.group(1))
out.append(t[pos:])
open(path, "w", encoding="utf-8").write("".join(out))
print("removed", removed)
