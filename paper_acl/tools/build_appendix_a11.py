#!/usr/bin/env python3
"""Builds paper_acl/appendix_a11.tex (Appendix O, Experiment 3 = Amendment 11) from the registered results.

  python paper_acl/tools/build_appendix_a11.py --results results/amendment11/RESULTS.json --out paper_acl/appendix_a11.tex

Tables: O1 registered tests; O2 cell means by size (figures/tab_a11_cells.tex, from scripts/a11_summary.py);
O3 crossed histories (figures/tab_a11_crossed.tex).
"""
import argparse
import json
import math


def num(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    s = f"{v:.{d}f}"
    return s.replace("-", r"\textminus{}") if v < 0 and float(s) != 0 else s.lstrip("-")


def p(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return "$<$.001" if v < .001 else f"{v:.3f}".lstrip("0")


ROWS = [
    ("M3", "Unseen nouns' copy score minus 1 nat (manipulation check; one-sided)", None),
    ("H50", "At 1,000: changing-shape bursts minus fixed-shape bursts (one-sided; primary)", "holm"),
    ("H51", "Changing shape: bursts minus spaced, pooled (one-sided)", "holm"),
    ("H52", r"Spaced: fixed minus changing shape, pooled (two-sided; TOST $\pm$0.3)", "holm"),
    ("H53", r"Fixed-shape spaced nouns: $L(s^*) - L(s)$, pooled (one-sided)", "holm"),
    ("H54", r"At 1,000: noun-specific copy score of changing-shape bursts (TOST $\pm$0.3)", None),
    ("H55", "At 1,000: changing-shape bursts minus row-geometry prediction (one-sided)", "holm"),
    ("H56", "At 1,000: queried-noun minus context-noun effect of changing vs fixed bursts (one-sided)", "holm"),
    ("M3B", "85.6M models: unseen copy score minus 1 nat (one-sided)", None),
    ("H57", "85.6M models, at 1,000: changing-shape bursts minus fixed-shape bursts (one-sided)", None),
    ("M3C_changing", "Changing-only corpora: unseen copy score minus 1 nat", None),
    ("M3C_fixed", "Fixed-only corpora: unseen copy score minus 1 nat", None),
    ("H58", "At 1,000: changing-shape bursts (changing-only) minus fixed-shape bursts (fixed-only) (one-sided)", None),
    ("H59_changing_bursts", r"At 1,000: changing-shape bursts, changing-only minus mixed corpora (two-sided; TOST $\pm$0.3)", None),
    ("H59_fixed_bursts", r"At 1,000: fixed-shape bursts, fixed-only minus mixed corpora (two-sided; TOST $\pm$0.3)", None),
]
LABEL = {"M3C_changing": "M3C", "M3C_fixed": "M3C", "H59_changing_bursts": "H59", "H59_fixed_bursts": "H59"}


def reading(k, r):
    if k.startswith("M3"):
        return "Passed" if r.get("pass") else "Failed"
    if "outcome" in r:
        o = r["outcome"]
        return {"equivalent": "Equivalent", "inconclusive": "Inconclusive",
                "different from zero but within +-0.3": r"Within $\pm$0.3",
                "different (fixed below)": "Different (fixed lower); not equivalent",
                "different (pure below mixed)": "Lower in fixed-only corpora"}.get(o, o)
    if "supported" in r:
        return "Supported" if r["supported"] else "Not supported"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/amendment11/RESULTS.json")
    ap.add_argument("--out", default="paper_acl/appendix_a11.tex")
    a = ap.parse_args()
    R = json.load(open(a.results, encoding="utf-8"))
    L = [r"\FloatBarrier",
         r"\section{Replication, single-history corpora and a larger model (Amendment 11)}\label{app:O}",
         r"\setcounter{table}{0}\renewcommand{\thetable}{\thesection\arabic{table}}",
         r"\textbf{Registration.} Amendment 11 is Experiment 3. It was registered (osf.io/qj9hb) after all results of "
         r"Experiment 2 and its exploratory analyses, after its corpora were built and checksummed, and before any of its "
         r"models was trained, to test the exploratory findings of Experiment 2 on new models. Part A repeated the "
         r"stable-shape (Regular grouped) design of Experiment 2 with ten new seeds at 3.4M, 10.9M and 25.6M parameters "
         r"(30 models). Part B trained ten models of 85.6M parameters (12 layers, 12 heads, width 768) on the same corpora "
         r"as the 25.6M models. Part C trained five seeds at each of the three smaller sizes on corpora in which all 36 "
         r"familiar nouns had changing shapes (pure\_changing) or all had fixed shapes (pure\_fixed), nine nouns per "
         r"cell, with the same exposures and share of text as the mixed design (30 models). The assignment of noun forms "
         r"to cells follows the Experiment 2 rule by seed position; model weights, reading order and placement are new. "
         r"All 40 corpora passed a checker and matched checksums recorded before training. Registered rules for failed "
         r"models (non-finite loss, or a final loss more than 0.5 nats above the median of its size and design) and for "
         r"complete seeds were applied; no model failed. Part A used Holm correction over H50, H51, H52 (difference), "
         r"H53, H55 and H56, with direction required at two of three sizes; equivalence tests used two one-sided tests "
         r"against $\pm$0.3 nats. H55 is a seed-level test: within each model, the copy scores of the 18 familiar nouns "
         r"at 1,000 exposures were regressed on the row's cosine with its initial value, its relative distance from the "
         r"trained kinds' mean row, its norm and a changing-burst indicator, and the indicator's coefficient was averaged "
         r"over sizes within seed. There were no deviations.",
         "",
         r"\begin{table}[!htbp]",
         r"\centering\small",
         r"\caption{Registered tests of Experiment 3 (seed-level estimates, nats; sizes averaged within seed; "
         r"t-based 95\% intervals). Holm $p$: adjusted over the six Part A tests marked $^h$; other $p$ values are "
         r"unadjusted. Sizes: number of the three sizes with the estimate's direction.}",
         r"\label{tab:O1}",
         r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.36\textwidth}rlrc>{\raggedright\arraybackslash}p{0.16\textwidth}@{}}",
         r"\toprule",
         r"& Test & Estimate & 95\% CI & $p$ & Sizes & Reading \\",
         r"\midrule"]
    holm = {}
    for k in ("H50", "H51", "H52", "H53", "H55", "H56"):
        if k in R and "holm_threshold" in R[k]:
            holm[k] = R[k]
    for k, txt, fam in ROWS:
        lab = LABEL.get(k, k) + (r"$^h$" if fam else "")
        if k not in R:
            L.append(f"{lab} & {txt} & -- & -- & -- & & Pending \\\\")
            continue
        r = R[k]
        ci = r.get("ci", [float("nan")] * 2)
        sz = r.get("sizes_in_direction", "")
        L.append(f"{lab} & {txt} & {num(r['mean'])} & [{num(ci[0])}, {num(ci[1])}] & {p(r['p'])} & {sz} & "
                 f"{reading(k, r)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    notes = []
    if "H52" in R:
        notes.append(f"H52: TOST $p$ = {R['H52']['tost_p']:.2f}; 90\\% interval [{num(R['H52']['ci90'][0])}, "
                     f"{num(R['H52']['ci90'][1])}].")
    if "H54" in R:
        notes.append(f"H54: TOST $p$ {p(R['H54']['tost_p'])}.")
    if "H55" in R and "clustered" in R["H55"]:
        c = R["H55"]["clustered"]
        w = R["H55"]["without_no_shape"]
        notes.append(f"H55, pooled noun-level regression with size and seed effects and seed-clustered errors: "
                     f"{num(c['mean'])} [{num(c['ci'][0])}, {num(c['ci'][1])}]; without no-shape nouns: "
                     f"{num(w['mean'])} [{num(w['ci'][0])}, {num(w['ci'][1])}]. By size: "
                     + ", ".join(num(R['H55']['by_size'][z]) for z in ("tiny", "small", "medium")) + ".")
    L.append(r"\par\smallskip\parbox{\textwidth}{\footnotesize " + " ".join(notes) + "}")
    L += [r"\end{table}", "",
          r"\begin{table}[!htbp]",
          r"\centering\small",
          r"\caption{Experiment 3: copy score by corpus design, history and exposures (means over seeds by size, and "
          r"seed-level mean with 95\% interval, sizes averaged). Mixed: ten seeds; single-history designs: five seeds.}",
          r"\label{tab:O2}",
          r"\setlength{\tabcolsep}{3pt}",
          r"\input{figures/tab_a11_cells.tex}",
          r"\end{table}", "",
          r"\begin{table}[!htbp]",
          r"\centering\small",
          r"\caption{Experiment 3, mixed design, 1,000 exposures: copy score when the context sentence is about a noun "
          r"of one history (rows) and the query names a different noun of another (columns); means over ten seeds, "
          r"sizes averaged (24 items per pair and model).}",
          r"\label{tab:O3}",
          r"\input{figures/tab_a11_crossed.tex}",
          r"\end{table}", ""]
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written", a.out)


if __name__ == "__main__":
    main()
