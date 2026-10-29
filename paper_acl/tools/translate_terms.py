#!/usr/bin/env python3
"""translate_terms.py  |  Rewrites generated appendix files in the main text's terms (registration terms are
listed in Appendix C). Run after the build_appendix_* scripts and trim_tables.py:

  python paper_acl/tools/translate_terms.py paper_acl/appendix_a9.tex paper_acl/appendix_a10.tex paper_acl/appendix_a11.tex
"""
import re
import sys

PAIRS = [
    (r"Regular grouped and Reassigned grouped", "stable-kind and shifting-kind"),
    (r"Regular grouped \(confirmatory\) and Reassigned grouped \(exploratory\)", "stable-kind (confirmatory) and shifting-kind (exploratory)"),
    (r"Regular grouped, Regular random, Reassigned grouped", "stable-kind, stable-kind random-order, shifting-kind"),
    (r"Regular grouped", "stable-kind"), (r"Regular random", "stable-kind random-order"),
    (r"Reassigned grouped", "shifting-kind"), (r"\bRegular\b", "stable-kind"), (r"\bReassigned\b", "shifting-kind"),
    (r"rotating runs", "changing-shape bursts"), (r"rotating singles", "changing-shape spaced nouns"),
    (r"fixed runs", "fixed-shape bursts"), (r"fixed singles", "fixed-shape spaced nouns"),
    (r"absent runs", "no-shape bursts"), (r"absent singles", "no-shape spaced nouns"),
    (r"Rotating runs", "Changing-shape bursts"), (r"Rotating singles", "Changing-shape spaced nouns"),
    (r"Fixed runs", "Fixed-shape bursts"), (r"Fixed singles", "Fixed-shape spaced nouns"),
    (r"Absent runs", "No-shape bursts"), (r"Absent singles", "No-shape spaced nouns"),
    (r"\bshape-absent\b", "no-shape"), (r"\bShape-absent\b", "No shape"),
    (r"\buninformative\b", "changing-shape"), (r"\bUninformative\b", "Changing shape"),
    (r"\brotating\b", "changing-shape"), (r"\bRotating\b", "Changing shape"),
    (r"\\emph\{single\}", r"\\emph{spaced}"), (r"\\emph\{run\}", r"\\emph{bursts}"),
    (r"presented singly", "presented spaced"), (r"presented in runs", "presented in bursts"),
    (r"\bcarry\b", "copy score"), (r"\bCarry\b", "Copy score"),
    (r"typical row", "typical embedding"), (r"input rows", "input embeddings"), (r"input row", "input embedding"),
    (r"\brows\b", "embeddings"), (r"\brow\b", "embedding"),
    (r"atypical\\_init", "random (untrained norm)"), (r"Atypical\\_init", "Random (untrained norm)"),
    (r"\bKnowledge\b", "Stored preference"), (r"\bknowledge\b", "stored preference"),
    (r"familiarity nouns", "new nouns"), (r"nonce nouns", "new nouns"),
    (r"\bPart A\b", "Experiment 2"), (r"\bPart C\b", "the embedding analysis"), (r"\bPart B\b", "the pretrained test"),
]


for path in sys.argv[1:]:
    t = open(path, encoding="utf-8").read()
    pairs = [(a, b) for a, b in PAIRS if not (path.endswith("appendix_a11.tex") and a.startswith(r"\bPart"))]
    n0 = t
    counts = {}
    for a, b in pairs:
        t, k = re.subn(a, b, t)
        if k:
            counts[a] = k
    open(path, "w", encoding="utf-8").write(t)
    print(path, sum(counts.values()), "replacements")
# Post-fixes applied by hand after translation (see the session log): table-row "(rows)" in the crossed captions,
# section headings "The embedding analysis (Experiment 1 models)" and "The pretrained test", and caption wording.
