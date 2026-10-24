"""Build paper_acl/appendix_recut.tex (Appendix L) from the rescoring results.

Inputs (run from the repository root):
  results/recut/RESCORE_per_seed.csv   (scripts/recut_rescore_analysis.py)
  results/amendment8/trained_per_cell.csv, results/amendment8/trained_per_run.csv
  results/recut/BLEND_per_seed.csv, results/recut/norms.csv  (optional; scripts/recut_blend_analysis.py)
Usage: python paper_acl/tools/build_appendix_recut.py > paper_acl/appendix_recut.tex
"""
import os
import numpy as np, pandas as pd
from scipy import stats

ORDER = [("regular_random", "Regular, random"), ("regular_b3p6", "Regular, graded (3.5)"),
         ("regular_grouped", "Regular, grouped"), ("nostable_grouped", "No stable feature, grouped"),
         ("objects_same", "Objects, same-object runs"), ("objects_diff", "Objects, different-object runs"),
         ("reassigned_random", "Reassigned, random"), ("reassigned_grouped", "Reassigned, grouped")]
f = lambda x: ("\\textminus{}%.2f" % -x) if x < -0.005 else "%.2f" % abs(x) if abs(x) < 0.005 else "%.2f" % x

s = pd.read_csv("results/recut/RESCORE_per_seed.csv")
tc = pd.read_csv("results/amendment8/trained_per_cell.csv")
tc = tc[(tc.k == 1) & (tc.match == True)].copy()
tc["condition"] = tc["condition"].str.replace("regular_f_", "", regex=False)
tc = tc.groupby("condition")[["d_shape", "d_colour", "OI_colour"]].mean()

def m(c, e, meas, q):
    return s[(s.condition == c) & (s.emb == e) & (s.measure == meas)][q].mean()

out = [r"\FloatBarrier", r"\section{Test-noun representation and changed-word scoring}\label{app:L}",
       r"\setcounter{table}{0}\renewcommand{\thetable}{L\arabic{table}}",
       r"\setcounter{figure}{0}\renewcommand{\thefigure}{L\arabic{figure}}",
       "These analyses were added after the registrations and are exploratory. They rescore the 120 fresh-text "
       "models on the one-exemplar items (\\texttt{scripts/recut\\_rescore.py}, \\texttt{scripts/recut\\_rescore\\_analysis.py}). "
       "\\textbf{Typical tokens}: each novel noun's input embedding is replaced by the mean input embedding of the 32 "
       "training nouns plus $r\\,u_j$, where $r$ is the mean distance of the training-noun embeddings from that mean and "
       "the $u_j$ are orthonormal random directions (fixed seed), so the eight tokens look like ordinary nouns but stay "
       "distinct; output weights are unchanged. \\textbf{Changed-word $\\Delta$}: the difference in log probability of "
       "the one word that differs between the continuations, given the identical prefix. With the original embeddings "
       "and whole-continuation $\\Delta$, the rescoring reproduced the reported scores (144,000 item scores; largest "
       "difference $2\\times10^{-5}$ nats). Trained-noun values are from the registered trained-noun items "
       "(long version in the supplementary material; \\texttt{scripts/recut\\_noun\\_store\\_check.py}).", "",
       r"\begin{table}[!htbp]", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
       r"\caption{Index and noun-specific index under four scorings (nats; mean of seed means). Original: unseen tokens with untrained embeddings, as reported; typical: the same tokens with typical embeddings; full: whole-continuation $\Delta$, as reported; word: changed word only.}",
       r"\label{tab:L1}", r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
       r" & \multicolumn{4}{c}{Index} & \multicolumn{4}{c}{Noun-specific index} \\",
       r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
       r"Condition & orig./full & typ./full & orig./word & typ./word & orig./full & typ./full & orig./word & typ./word \\",
       r"\midrule"]
for c, lab in ORDER:
    vals = [m(c, e, meas, q) for q in ("index", "noun_specific") for e, meas in
            (("original", "full"), ("typical", "full"), ("original", "word"), ("typical", "word"))]
    out.append(lab + " & " + " & ".join(f(v) for v in vals) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
        r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Preference for repeating the context's shape and colour after one exemplar, same noun, for the three kinds of test noun (nats; whole-continuation $\Delta$).}",
        r"\label{tab:L2}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r" & \multicolumn{3}{c}{$\Delta$ shape} & \multicolumn{3}{c}{$\Delta$ colour} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Condition & Unseen & Typical & Trained & Unseen & Typical & Trained \\", r"\midrule"]
for c, lab in ORDER:
    vals = [m(c, "original", "full", "shape"), m(c, "typical", "full", "shape"), tc.loc[c, "d_shape"],
            m(c, "original", "full", "colour"), m(c, "typical", "full", "colour"), tc.loc[c, "d_colour"]]
    out.append(lab + " & " + " & ".join(f(v) for v in vals) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Blend curve, if available
bp = "results/recut/BLEND_per_seed.csv"
if os.path.exists(bp):
    b = pd.read_csv(bp)
    al = sorted(b.alpha.unique())
    out += [r"\begin{table}[!htbp]", r"\centering\small",
            r"\caption{Noun-familiarity curve: unseen tokens' input embeddings blended from their untrained values (0) to typical embeddings (1). Index and $\Delta$ shape, same noun (nats; mean of seed means); slope per unit blend with 95\% interval over five seeds.}",
            r"\label{tab:L3}", r"\begin{tabular}{@{}ll" + "r" * len(al) + "r@{}}", r"\toprule",
            r"Condition & Measure & " + " & ".join(f"{x:g}" for x in al) + r" & Slope \\", r"\midrule"]
    for c, lab in [(c, l) for c, l in ORDER if c in set(b.condition)]:
        for q, ql in (("index", "Index"), ("shape", r"$\Delta$ shape")):
            sc = b[b.condition == c]
            means = [sc[sc.alpha == x][q].mean() for x in al]
            sl = np.array([np.polyfit(g.alpha, g[q], 1)[0] for _, g in sc.groupby("seed")])
            h = stats.t.ppf(.975, len(sl) - 1) * sl.std(ddof=1) / np.sqrt(len(sl))
            out.append(f"{lab} & {ql} & " + " & ".join(f(v) for v in means)
                       + f" & {f(sl.mean())} [{f(sl.mean()-h)}, {f(sl.mean()+h)}] \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    if os.path.exists("results/recut/norms.csv"):
        n = pd.read_csv("results/recut/norms.csv")
        out.append(f"Mean input-embedding norms: trained nouns {n.trained_norm.mean():.2f}, unseen tokens "
                   f"{n.novel_norm.mean():.2f}, typical tokens {n.typical_norm.mean():.2f}.")
print("\n".join(out))
