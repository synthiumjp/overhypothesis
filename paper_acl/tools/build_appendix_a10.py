"""Build paper_acl/appendix_a10.tex (Appendix N: Amendment 10) from the results.

Inputs (run from the repository root): results/amendment10/RESULTS_A.json, RESULTS_B.json, RESULTS_C.json,
scores_a.csv, pythia_scores.csv, pythia_checkpoints.csv, rowswap.csv; paper_acl/figures/tab_a10_cells.tex
(from scripts/amendment10_figures.py).
Usage: python paper_acl/tools/build_appendix_a10.py > paper_acl/appendix_a10.tex
"""
import json

import numpy as np
import pandas as pd
from scipy import stats


def f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:.{d}f}"
    return s.replace("-", r"\textminus{}") if x < 0 and float(s) != 0 else s.lstrip("-")


def p_fmt(p):
    return "$<$.001" if p < .001 else f"{p:.3f}".lstrip("0")


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def holm(ps):
    order = sorted(ps.items(), key=lambda x: x[1])
    adj, run = {}, 0.0
    for i, (h, p) in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * p))
        adj[h] = run
    return adj


A = json.load(open("results/amendment10/RESULTS_A.json"))
B = json.load(open("results/amendment10/RESULTS_B.json"))
C = json.load(open("results/amendment10/RESULTS_C.json"))
s = pd.read_csv("results/amendment10/scores_a.csv")
s = s[s.ckpt.str.endswith("model_final.pt")]
py = pd.read_csv("results/amendment10/pythia_scores.csv")
ck = pd.read_csv("results/amendment10/pythia_checkpoints.csv")
rs = pd.read_csv("results/amendment10/rowswap.csv")

out = [r"\FloatBarrier", r"\section{Recurrence, matched frames, embedding rows and pretrained models (Amendment 10)}"
       r"\label{app:N}",
       r"\setcounter{table}{0}\renewcommand{\thetable}{\thesection\arabic{table}}",
       r"\textbf{Registration.} Amendment 10 has three parts: Part A is Experiment 2, Part B Experiment 3 (pretrained "
       r"models) and Part C the row analysis. It was registered after the Amendment 9 results, after its corpora were "
       r"built and checksummed, and before any of them was used, any of its models was trained, or any pretrained model "
       r"was run on its items. It answered two confounds raised by a blind review of the Amendment 9 results (by an AI "
       r"assistant given only the write-up of Experiment 1): familiarity nouns never recurred within a "
       r"window, and shape-absent nouns never appeared in a frame with a shape slot after \emph{is a}. The noun lists "
       r"for Part B were fixed, with checksums, before the pilot. A pilot on Pythia-410M (never in a test) gave a mean "
       r"increase of 0.13 nats from the original to the atypical\_init row, so the smallest effect of interest for H48 "
       r"was 0.30 nats by the registered rule. One deviation: some Pythia tokenizers define no padding token, so the "
       r"scoring script falls back to token id 0; padded positions are masked and only the last real token is read, and "
       r"batched and single-item scores agreed exactly. No score had been produced by the affected models.", "",
       r"\textbf{Part A design.} Everything not stated is as in Amendment 9 (Appendix~\ref{app:M}). Histories: "
       r"\emph{fixed}, frame 1 (\emph{A \textit{noun} is a \textit{shape} \textit{colour} \textit{texture} thing}) with "
       r"the noun's own shape; \emph{absent}, frame 7 (\emph{A \textit{noun} is a \textit{colour} \textit{texture} "
       r"thing}), which names no shape; \emph{rotating}, frame 1 with a shape drawn from the corpus's shape "
       r"distribution, a new one for every sentence when presented singly and one per run, changing between runs, when "
       r"presented in runs. Presentation: \emph{single}, one sentence per unit; \emph{run}, four consecutive sentences "
       r"about the noun in one window. Exposures (100 or 1,000) count sentences, so runs and singles are matched in "
       r"sentences, not in units or spacing. Trained kinds use frame 7 in one sentence in fifteen, so the frame is not "
       r"new. Three nouns per cell (12 cells) and four unseen nouns; Regular grouped (confirmatory) and Reassigned grouped "
       r"(exploratory) base corpora; three sizes and ten seeds (60 models). A checker re-read every corpus and verified "
       r"exposures, units, frames, shapes within runs and window placement; all 40 corpora matched checksums recorded "
       r"before training.", ""]

# Table N1: Part A registered tests
labels = {"M2": "Unseen carry minus 1 nat (manipulation check)",
          "K1": "Knowledge, fixed runs minus fixed singles (equivalence $\\pm$1)",
          "H40": "Absent runs minus absent singles (two-sided; TOST $\\pm$0.3)",
          "H41": "Fixed runs minus fixed singles (two-sided; TOST $\\pm$0.3)",
          "H42": "Rotating runs minus rotating singles (one-sided)",
          "H43": "Fixed singles minus absent singles (one-sided)",
          "H44": "Absent singles minus unseen with typical row (two-sided; TOST $\\pm$0.3)"}
adj = holm({h: A[h]["p"] for h in ("H40", "H41", "H42", "H43", "H44")})
verdict = {"M2": "Passed" if A["M2"]["pass"] else "Failed",
           "K1": "Not equivalent (H41 not interpreted)" if not A["K1"]["equivalent"] else "Equivalent",
           "H40": "Above", "H41": "Above (not interpreted)", "H42": "Supported", "H43": "Supported", "H44": "Above"}
for h in ("H40", "H41", "H44"):
    if A[h].get("outcome") != "above":
        verdict[h] = A[h].get("outcome", "").capitalize()
for h in ("H42", "H43"):
    if not A[h]["supported"]:
        verdict[h] = "Not supported"
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Registered tests of Part A (Regular grouped models; 100 and 1,000 exposures pooled; seed-level "
        r"estimates over ten seeds, sizes averaged within seed; nats). Holm $p$ is adjusted over H40 to H44. Sizes: number "
        r"of the three sizes with the estimate's direction. Readings follow the registered table: ``above'' means the "
        r"Holm threshold was met and the direction held at two of three sizes; TOST $p$ for H40, H41 and H44 was above "
        r".99, so none was equivalent.}",
        r"\label{tab:N1}", r"\begin{tabular}{@{}lp{0.34\textwidth}rrrcp{0.17\textwidth}@{}}", r"\toprule",
        r"& Test & Estimate & 95\% CI & Holm $p$ & Sizes & Reading \\", r"\midrule"]
for h in ("M2", "K1", "H40", "H41", "H42", "H43", "H44"):
    r = A[h]
    hp = p_fmt(adj[h]) if h in adj else p_fmt(r["p"]) + "$^a$"
    out.append(f"{h} & {labels[h]} & {f(r['mean'])} & [{f(r['ci'][0])}, {f(r['ci'][1])}] & {hp} & "
               f"{r.get('sizes_in_direction', '')} & {verdict[h]} \\\\")
out += [r"\bottomrule", r"\end{tabular}",
        r"\\[2pt]{\footnotesize $^a$ Outside the Holm family; unadjusted $p$. The M2 estimate is unseen carry minus "
        r"1 nat, so unseen carry was " + f(A["M2"]["mean"] + 1) + r" nats.}", r"\end{table}", ""]

# Table N2: cells
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Carry (nats) by cell, mean of seed values (sizes averaged) with 95\% intervals over ten seeds. "
        r"Unseen nouns have no exposures; the typical row replaces an unseen noun's input row with one typical of "
        r"trained nouns.}", r"\label{tab:N2}",
        open("paper_acl/figures/tab_a10_cells.tex", encoding="utf-8").read().strip(), r"\end{table}", ""]


# Table N3: knowledge, slot, by cell (Regular grouped) and Reassigned contrasts
def sv(base, typ, e, q):
    d = s[(s.base == base) & (s.type == typ) & (s.emb == "original") & (s.exposures == e)]
    return d.groupby(["size", "seed"])[q].mean().groupby("seed").mean().mean()


out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Knowledge of the noun's own shape and expectation of a shape rather than a colour at the query "
        r"position (slot: log of the summed probability of the ten shapes minus that of the ten colours), Regular "
        r"grouped models, mean of seed values (exploratory). Unseen nouns: knowledge " +
        f(s[(s.base == 'regular_grouped') & (s.type == 'unseen') & (s.emb == 'original')].groupby(['size', 'seed'])
          .know.mean().groupby('seed').mean().mean()) + ", slot " +
        f(s[(s.base == 'regular_grouped') & (s.type == 'unseen') & (s.emb == 'original')].groupby(['size', 'seed'])
          .slot.mean().groupby('seed').mean().mean()) + r".}", r"\label{tab:N3}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r" & \multicolumn{2}{c}{Knowledge} & \multicolumn{2}{c}{Slot} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Noun & 100 & 1,000 & 100 & 1,000 \\", r"\midrule"]
for h, hl in (("fixed", "Fixed shape"), ("absent", "Shape-absent"), ("rotating", "Rotating shape")):
    for pres in ("single", "run"):
        t = f"{h}_{pres}"
        out.append(f"{hl}, {pres}s & " + " & ".join(f(sv('regular_grouped', t, e, q)) for q in ("know", "slot")
                                                     for e in (100, 1000)) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

X = A["exploratory_reassigned"]
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{The Part A contrasts in Reassigned grouped models (exploratory; seed-level, unadjusted two-sided $p$).}",
        r"\label{tab:N4}", r"\begin{tabular}{@{}lp{0.42\textwidth}rrr@{}}", r"\toprule",
        r"& Contrast & Estimate & 95\% CI & $p$ \\", r"\midrule"]
for h in ("H40", "H41", "H42", "H43"):
    r = X[h]
    out.append(f"{h} & {labels[h].split(' (')[0]} & {f(r['mean'])} & [{f(r['ci'][0])}, {f(r['ci'][1])}] & "
               f"{p_fmt(r['p'])} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Extra Part A tables (exploratory): matched contrasts, noun-specific carry, per size
def seedvals(df, base, typ, e=None, q="carry", emb="original", size=None):
    d = df[(df.base == base) & (df.type == typ) & (df.emb == emb)]
    if e is not None:
        d = d[d.exposures == e]
    if size is not None:
        d = d[d["size"] == size]
    g = d.groupby(["size", "seed"])[q].mean()
    return g.groupby("seed").mean() if size is None else g.droplevel(0)


def cifmt(x, d=2):
    m, lo, hi = ci(x)
    return f"{f(m, d)} [{f(lo, d)}, {f(hi, d)}]"


def pool(df, base, typ, q="carry"):
    return sum(seedvals(df, base, typ, e, q) for e in (100, 1000)) / 2


out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Matched contrasts in Part A (exploratory; seed-level, nats, 95\% intervals). Rotating minus fixed "
        r"runs: same frame, presentation and number of occasions; the two differ only in whether the shape changes "
        r"between runs (and so in whether a shape is stored). Fixed minus rotating singles: same frame, presentation "
        r"and expectation of a shape at the test position (Table~\ref{tab:N3}); they differ only in whether the noun "
        r"has one stored shape.}", r"\label{tab:N9}",
        r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
        r"Base & Contrast & Pooled & 100 & 1,000 \\", r"\midrule"]
for b, lab in (("regular_grouped", "Regular grouped"), ("reassigned_grouped", "Reassigned grouped")):
    for (a1, a2, cl) in (("rotating_run", "fixed_run", "Rotating runs $-$ fixed runs"),
                         ("fixed_single", "rotating_single", "Fixed singles $-$ rotating singles"),
                         ("fixed_run", "fixed_single", "Fixed runs $-$ fixed singles")):
        vals = [cifmt(pool(s, b, a1) - pool(s, b, a2))] + [cifmt(seedvals(s, b, a1, e) - seedvals(s, b, a2, e))
                                                           for e in (100, 1000)]
        out.append(f"{lab if cl.startswith('Rot') else ''} & {cl} & " + " & ".join(vals) + r" \\")
kn = [cifmt(seedvals(s, "regular_grouped", "fixed_run", e, "know") - seedvals(s, "regular_grouped", "fixed_single", e,
                                                                               "know")) for e in (100, 1000)]
out += [r"\bottomrule", r"\end{tabular}",
        r"\\[2pt]{\footnotesize Knowledge, fixed runs minus fixed singles (Regular grouped): " + kn[0] +
        r" at 100 exposures, " + kn[1] + r" at 1,000.}", r"\end{table}", ""]

s9 = pd.read_csv("results/amendment9/scores.csv")
s9 = s9[s9.ckpt.str.endswith("model_final.pt")]
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Noun-specific carry (exploratory): carry when the context sentence is about the query noun minus "
        r"carry when it is about another noun of the same cell, with the same shape (nats; seed-level means). Near "
        r"zero means the model copies the context's shape whichever noun the context was about.}", r"\label{tab:N10}",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r" & & \multicolumn{2}{c}{Regular grouped} & \multicolumn{2}{c}{Reassigned grouped} \\"
        r" \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Experiment & Noun & 100 & 1,000 & 100 & 1,000 \\", r"\midrule"]
rows10 = [("unseen", "Unseen")] + [(f"{h}_{p}", f"{n}, {p}s") for h, n in (("fixed", "Fixed"), ("absent", "Shape-absent"),
                                                                          ("rotating", "Rotating")) for p in ("single", "run")]
for i, (typ, tl) in enumerate(rows10):
    cells = []
    for b in ("regular_grouped", "reassigned_grouped"):
        if typ == "unseen":
            v = f(ci(seedvals(s, b, "unseen", q="noun_specific"))[0])
            cells += [v, ""]
        else:
            cells += [f(ci(seedvals(s, b, typ, e, "noun_specific"))[0]) for e in (100, 1000)]
    out.append(f"{'2' if i == 0 else ''} & {tl} & " + " & ".join(cells) + r" \\")
out.append(r"\addlinespace")
for i, (typ, tl) in enumerate((("unseen", "Unseen"), ("consistent", "Fixed shape"), ("uninformative", "Uninformative"),
                               ("absent", "Shape-absent"))):
    cells = []
    for b in ("regular_grouped", "reassigned_grouped"):
        if typ == "unseen":
            cells += [f(ci(seedvals(s9, b, "unseen", q="noun_specific"))[0]), ""]
        else:
            cells += [f(ci(seedvals(s9, b, typ, e, "noun_specific"))[0]) for e in (100, 1000)]
    out.append(f"{'1' if i == 0 else ''} & {tl} & " + " & ".join(cells) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Part A by model size (Regular grouped; seed means, 100 and 1,000 exposures pooled; nats).}",
        r"\label{tab:N11}", r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
        r"Size & Unseen & Typical row & H40 & H41 & H42 & H43 & H44 & Rot.\ $-$ fixed runs \\", r"\midrule"]
for z, zl in (("tiny", "3.4M"), ("small", "11.0M"), ("medium", "25.6M")):
    def c(typ, emb="original"):
        d = s[(s.base == "regular_grouped") & (s["size"] == z) & (s.type == typ) & (s.emb == emb)]
        if typ != "unseen":
            d = d[d.exposures.isin([100, 1000])]
        return d.groupby(["seed", "exposures"]).carry.mean().groupby("seed").mean()
    vals = [c("unseen").mean(), c("unseen", "typical").mean(), (c("absent_run") - c("absent_single")).mean(),
            (c("fixed_run") - c("fixed_single")).mean(), (c("rotating_run") - c("rotating_single")).mean(),
            (c("fixed_single") - c("absent_single")).mean(), (c("absent_single") - c("unseen", "typical")).mean(),
            (c("rotating_run") - c("fixed_run")).mean()]
    out.append(zl + " & " + " & ".join(f(v) for v in vals) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Reanalysis of the Experiment 2 models (exploratory; scripts/a10_reanalysis.py, a10_reanalysis_summary.py)
import os
if os.path.exists("paper_acl/figures/tab_a10_reanalysis.tex"):
    out += [r"\textbf{Reanalysis of the Experiment 2 models (exploratory, added after the results).} For every familiar "
            r"noun in the 60 final models we recorded its input row's cosine with its initial row (rebuilt from the seed; "
            r"the rebuild reproduced the unseen nouns' rows, which receive no gradient, in every model), its distance "
            r"from the trained kinds' mean row and its norm; the copy score with its own row, with a typical row and "
            r"with the row of a noun of another history (rows swapped between changing- and fixed-shape bursts and "
            r"between changing-shape bursts and spaced nouns); the copy score with one sentence about a trained kind "
            r"between context and query (gap), with the context in the frame \emph{Look at the \textit{noun}. It is "
            r"\textit{shape} and \textit{colour}.} (other frame), and for the context's colour after \emph{A "
            r"\textit{noun} is a \textit{shape}} (colour); and first-layer attention from the query position to the "
            r"context's shape, averaged over heads. Full results, including shifting-shape corpora and 100 exposures, "
            r"are in the supplementary material.", "",
            r"\begin{table}[!htbp]", r"\centering\small",
            r"\caption{Experiment 2 reanalysis, stable-shape corpora, 1,000 exposures (unseen nouns: no exposures). "
            r"Copy scores in nats; cosine of the row with its initial value; first-layer attention from the query to "
            r"the context's shape. Seed-level means. Swapping rows between changing- and fixed-shape bursts moved the "
            r"copying with the row (2.88 nats [2.01, 3.76] in each direction). In a noun-level regression of the copy "
            r"score on row geometry (cosine with the initial row, distance from the trained kinds' mean, norm, size "
            r"and seed; standard errors clustered by seed), changing-shape bursts scored 3.19 nats [2.46, 3.91] more "
            r"than geometry predicts at 1,000 exposures and $-$0.09 [$-$0.51, 0.33] at 100.}", r"\label{tab:N13}",
            open("paper_acl/figures/tab_a10_reanalysis.tex", encoding="utf-8").read().strip(), r"\end{table}", ""]

if os.path.exists("paper_acl/figures/tab_a10_crossed.tex"):
    out += [r"\begin{table}[!htbp]", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}",
            r"\caption{Crossed histories (exploratory), stable-shape corpora, 1,000 exposures (unseen nouns: none): copy "
            r"score when the context sentence is about a noun of one history (rows) and the query names a different "
            r"noun of another (columns); 24 items per pair and model, seed-level means. Averaged over familiar "
            r"histories, a queried noun with changing-shape bursts copied 1.35 nats [0.97, 1.74] more than one with "
            r"fixed-shape bursts, and a context noun with changing-shape bursts raised copying by 0.39 [0.19, 0.58]. "
            r"Full results are in the supplementary material.}", r"\label{tab:N14}",
            open("paper_acl/figures/tab_a10_crossed.tex", encoding="utf-8").read().strip(), r"\end{table}", ""]

# Part C
out += [r"\textbf{Part C (embedding rows in the Amendment 9 models).} In the 60 final grouped models of Amendment 9 we "
        r"replaced the input rows of unseen nouns and scored carry again; the unmodified condition reproduced the "
        r"Amendment 9 scores. Conditions: own row; own direction scaled to the mean norm of the trained kinds' rows "
        r"(ownnorm); a random direction at that norm (randnorm); the trained kinds' mean row. Shape-absent nouns at "
        r"10,000 exposures were also scored with their own direction scaled to the unseen nouns' mean norm (shrunk).", "",
        r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Part C: carry (nats) by input row, mean over seeds (sizes averaged). H49 (registered, Regular "
        r"grouped): ownnorm minus own " + f(C["H49"]["mean"]) + f" [{f(C['H49']['ci'][0])}, {f(C['H49']['ci'][1])}], "
        r"one-sided $p$ = " + p_fmt(C["H49"]["p"]) + f", direction at {C['H49']['sizes_in_direction']} of 3 sizes, "
        r"supported. In Reassigned models the same contrast went the other way (" +
        f(C["H49_reassigned_exploratory"]["mean"]) + f" [{f(C['H49_reassigned_exploratory']['ci'][0])}, "
        f"{f(C['H49_reassigned_exploratory']['ci'][1])}]" + r"; exploratory). Randnorm minus ownnorm: " +
        f(C["randnorm_minus_ownnorm_regular_grouped"]["mean"]) + r" (Regular), " +
        f(C["randnorm_minus_ownnorm_reassigned_grouped"]["mean"]) + r" (Reassigned).}", r"\label{tab:N5}",
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r" & \multicolumn{4}{c}{Unseen nouns} & \multicolumn{2}{c}{Shape-absent, 10k} \\"
        r" \cmidrule(lr){2-5}\cmidrule(lr){6-7}",
        r"Corpus & Own & Ownnorm & Randnorm & Trained mean & Own & Shrunk \\", r"\midrule"]
for b, lab in (("regular_grouped", "Regular grouped"), ("reassigned_grouped", "Reassigned grouped")):
    m, ab = C[f"means_{b}"], C[f"means_absent_{b}"]
    out.append(f"{lab} & {f(m['own'])} & {f(m['ownnorm'])} & {f(m['randnorm'])} & {f(m['mean'])} & {f(ab['own'])} & "
               f"{f(ab['shrunk'])} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Part B
nn = py.noun.nunique()
npairs = py.pair.nunique()
out += [r"\textbf{Part B (pretrained models).} " + f"{nn} nouns in {npairs} pairs, matched on Pile frequency and "
        r"count/mass type, were coded for colour typicality by two independent AI passes with a fixed prompt before "
        r"any model was run (exact agreement 94\%, weighted $\kappa$ = 0.96); nouns that are also colour terms or that "
        r"both passes flagged as polysemous were excluded. Colour-typical nouns are mostly natural kinds, so kind was a "
        r"covariate. Each noun was read in three neutral frames with each of the eleven basic colour terms that are one "
        r"token; carry is the mean over frames and ordered colour pairs of $\log p(c_1 \mid c_1) - \log p(c_1 \mid c_2)$, "
        r"with colour probabilities renormalised over the colour set. Only the noun's input row was changed.", "",
        r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Part B registered tests by primary model (nouns or pairs as units; Holm within model over H45 to H48). "
        r"H45: atypical\_init minus original; H46: atypical\_init minus typical row; H47: colour-typical minus "
        r"colour-variable nouns, original rows; H48: difference between groups in the H45 increase (TOST against "
        r"$\pm$0.30). A hypothesis needed Holm in at least four of the five models. Outcome: H45 and H46 met Holm in two "
        r"of five models, H47 in none, and H48 was inconclusive (equivalent in one model).}", r"\label{tab:N6}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
        r" & \multicolumn{2}{c}{H45} & \multicolumn{2}{c}{H46} & \multicolumn{2}{c}{H47} & \multicolumn{2}{c}{H48} \\"
        r" \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Model & Est. & Holm & Est. & Holm & Est. & Holm & Est. & TOST $p$ \\", r"\midrule"]
for m in ("pythia-1b-deduped", "pythia-1.4b-deduped", "pythia-2.8b-deduped", "pythia-6.9b-deduped",
          "pythia-12b-deduped"):
    r = B["models"][m]
    cells = []
    for h in ("H45", "H46", "H47"):
        cells += [f"{f(r[h]['mean'])}", "met" if r[h]["holm_reject"] else "--"]
    cells += [f(r["H48"]["mean"]), p_fmt(r["H48"]["tost_p"])]
    out.append(m.replace("pythia-", "").replace("-deduped", "").upper() + " & " + " & ".join(cells) + r" \\")
mm = B["mixed_model"]
k_ai = "C(condition, Treatment('original'))[T.atypical_init]"
k_ty = "C(condition, Treatment('original'))[T.typical]"
k_gr = "C(group)[T.variable]"
out += [r"\bottomrule", r"\end{tabular}",
        r"\\[2pt]{\footnotesize Mixed model over the five primary models (carry $\sim$ condition $\times$ group + model + "
        r"kind; random intercepts for noun and pair, random condition slopes by noun): atypical\_init " +
        f(mm[k_ai]["coef"]) + f" ($p$ = {p_fmt(mm[k_ai]['p'])}), typical row " + f(mm[k_ty]["coef"]) +
        f" ($p$ = {p_fmt(mm[k_ty]['p'])}), colour-variable group " + f(mm[k_gr]["coef"]) +
        f" ($p$ = {p_fmt(mm[k_gr]['p'])}).}}", r"\end{table}", ""]

py["m"] = py.model.str.extract(r"pythia-([0-9.]+[mb])-")[0].str.upper()
order = ["70M", "160M", "1B", "1.4B", "2.8B", "6.9B", "12B"]
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Part B: carry of the context's colour (nats; mean over nouns) by model and input row, and with "
        r"original rows by group. Atypical: the atypical\_init direction at the noun's own norm (exploratory). 70M and "
        r"160M are descriptive. Prior entropy: entropy of the colour distribution after \emph{The \textit{noun} was} "
        r"with no context (nats; uniform over eleven colours is 2.40).}", r"\label{tab:N7}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
        r"Model & Original & Atypical\_init & Atypical & Typical row & Colour-typical & Colour-variable & Prior entropy \\",
        r"\midrule"]
for z in order:
    d = py[py.m == z]
    if d.empty:
        continue
    o = d[d.condition == "original"]
    out.append(f"{z} & " + " & ".join(f(d[d.condition == c].carry.mean()) for c in
                                       ("original", "atypical_init", "atypical", "typical")) +
               f" & {f(o[o.group == 'typical'].carry.mean())} & {f(o[o.group == 'variable'].carry.mean())} & "
               f"{f(o.prior_entropy.mean())} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Part B checks (exploratory)
def tost(d, b=0.3):
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    m = d.mean()
    return max(1 - stats.t.cdf((m + b) / se, n - 1), stats.t.cdf((m - b) / se, n - 1))


wv = py.pivot_table(index=["m", "noun"], columns="condition", values="carry")
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Part B checks (exploratory). Stored colour: with no context, log probability of a colour-typical "
        r"noun's coded colour minus the mean over the other colours after \emph{The \textit{noun} was} (nats; mean over "
        r"22 nouns, and the share of nouns above zero). TOST: equivalence of the H45 and H46 differences within "
        r"$\pm$0.30 nats over nouns. Norm only: atypical minus atypical\_init (same direction, different norm).}",
        r"\label{tab:N12}", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"Model & Stored colour & Share $>0$ & H45 TOST $p$ & H46 TOST $p$ & Norm only \\", r"\midrule"]
for z in order:
    d = py[(py.m == z) & (py.condition == "original") & (py.group == "typical")]
    if d.empty:
        continue
    w = wv.loc[z]
    out.append(f"{z} & {f(d.prior_typical.mean())} & {f((d.prior_typical > 0).mean())} & "
               f"{p_fmt(tost(w.atypical_init - w.original))} & {p_fmt(tost(w.atypical_init - w.typical))} & "
               f"{f((w.atypical - w.atypical_init).mean(), 3)} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

ck["step"] = ck.revision.str.replace("step", "").astype(int)
steps = sorted(ck.step.unique())
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Part B, Pythia-1.4B deduplicated across training (exploratory): carry (nats; mean over nouns) by "
        r"training step and input row.}", r"\label{tab:N8}",
        r"\begin{tabular}{@{}l" + "r" * len(steps) + r"@{}}", r"\toprule",
        r"Row & " + " & ".join(f"{x:,}" for x in steps) + r" \\", r"\midrule"]
for c, cl in (("original", "Original"), ("atypical_init", "Atypical\\_init"), ("typical", "Typical row")):
    out.append(cl + " & " + " & ".join(f(ck[(ck.step == x) & (ck.condition == c)].carry.mean()) for x in steps) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

print("\n".join(out))
