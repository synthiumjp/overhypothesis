"""Build paper_acl/appendix_a9.tex (Appendix M: the familiarity study, Amendment 9) from the results.

Inputs (run from the repository root): results/amendment9/RESULTS.json, scores.csv, circuits/heads_summary.csv,
circuits/per_noun_*.csv.
Usage: python paper_acl/tools/build_appendix_a9.py > paper_acl/appendix_a9.tex
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts")
from make_familiarity_corpora import CONF_SEEDS  # noqa: E402


def f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    s = f"{x:.{d}f}"
    return s.replace("-", r"\textminus{}") if x < 0 and float(s) != 0 else s.lstrip("-")


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def p_fmt(p):
    return "$<$.001" if p < .001 else f"{p:.3f}".lstrip("0")


R = json.load(open("results/amendment9/RESULTS.json"))
s = pd.read_csv("results/amendment9/scores.csv")
s = s[s.ckpt.str.endswith("model_final.pt") & s.seed.isin(CONF_SEEDS)]
hs = pd.read_csv("results/amendment9/circuits/heads_summary.csv")
hs = hs[hs.ckpt.str.endswith("model_final.pt") & hs.seed.isin(CONF_SEEDS)]
pn = pd.concat([pd.read_csv(x) for x in glob.glob("results/amendment9/circuits/per_noun_*.csv")])
pn = pn[pn.ckpt.str.endswith("model_final.pt") & pn.seed.isin(CONF_SEEDS)]

# Holm-adjusted p over the registered family
fam = {h: (R[h]["p"]) for h in ("H30", "H31", "H32", "H33", "H34") if h in R}
if R["H33"]["a"]["p"] < .05 and "H35" in R:
    fam["H35"] = R["H35"]["p"]
order = sorted(fam.items(), key=lambda x: x[1])
adj, run = {}, 0.0
for i, (h, p) in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * p))
    adj[h] = run

out = [r"\FloatBarrier", r"\section{The familiarity study (Amendment 9)}\label{app:M}",
       r"\setcounter{table}{0}\renewcommand{\thetable}{\thesection\arabic{table}}",
       r"\setcounter{figure}{0}\renewcommand{\thefigure}{\thesection\arabic{figure}}",
       r"\textbf{Registration.} The design, hypotheses and analysis were registered before any corpus was built "
       r"(Amendment 9). An addendum, written the same day before any corpus was built, fixed details the registration "
       r"left open (36 context draws per noun; uniform frame proportions; exposures placed at evenly spaced read "
       r"positions from 5\% to 95\% of training, after warm-up for every size; random in-window position; the 40 "
       r"nonce nouns; how the pilot fixes the number of nouns per cell; how a hypothesis counts as supported; the gate "
       r"for H35; and that heads without prefix matching are described as copying heads). A pilot (one seed, 3.4M "
       r"models, never used in a test) set three nouns per cell; with the registered rule the expected power for a "
       r"1-nat difference was 0.57, below the planned 0.8, which we report as a shortfall. Corpus checksums were "
       r"recorded before training and matched on the training machine; a checker verified every exposure, window and "
       r"frame from the tokens. All models were trained on one machine.", "",
       r"\textbf{Design.} In every model, 40 nonce nouns fill 13 cells (three per cell, four unseen): fixed shape, "
       r"uninformative shape (a new shape in every sentence, drawn from the corpus's shape distribution) and "
       r"shape-absent (a sixth frame, \emph{The \textit{noun} is \textit{colour} and \textit{texture}}, also used by "
       r"trained kinds in one sentence in fifteen), each at 10, 100, 1,000 and 10,000 exposures, plus unseen. Nouns "
       r"rotate over cells across ten seeds. Each familiarity sentence lies wholly inside one 128-token window, at most "
       r"one per window, between runs in grouped corpora. Base corpora: Regular grouped, Regular random, Reassigned "
       r"grouped; sizes 3.4M, 11.0M and 25.6M; ten seeds; 90 models with untied embeddings. Measures are read from the "
       r"full log-softmax over the ten shape words at the position that predicts the shape in \emph{A \textit{noun} is "
       r"a}: carry $= L(s) - \bar L$ after a context sentence with shape $s$; stored preference $= L(s^*) - \bar L$; "
       r"knowledge $= L(s^*)$ minus the mean over the other nine shapes after an unrelated sentence, where $\bar L$ "
       r"is the mean over shapes other than $s$ and $s^*$.", ""]

# Table M1: registered tests
rows = [("M1", "Knowledge at 1,000 exposures, fixed shape (each corpus)", None),
        ("H30", "Carry falls faster with exposure for fixed-shape than shape-absent nouns (slope difference)", "less"),
        ("H31", "Shape-absent (1k and 10k) minus unseen with typical embedding (two-sided; TOST $\\pm$1)", "two"),
        ("H32", "Stored preference rises with exposure, fixed shape (slope)", "greater"),
        ("H33a", "Ablating candidates removes more than half of unseen carry (proportion)", "half"),
        ("H33b", "Random heads minus candidates, unseen carry after ablation", "greater"),
        ("H34", "Reassigned: trained-kind carry intact minus candidates ablated", "greater"),
        ("H35", "Attention slope difference, fixed shape minus shape-absent", "less"),
        ("R1", "Replication: grouped minus random Regular, unseen carry", "greater")]
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Registered tests of the familiarity study (Regular grouped models unless stated; seed-level estimates "
        r"over ten seeds, sizes averaged within seed; nats, proportions for H33a, attention probability for H35). Holm "
        r"$p$ is adjusted over H30 to H34; H35 was not entered because H33(a) failed. Sizes: number of the three model "
        r"sizes with the registered direction.}",
        r"\label{tab:M1}", r"\begin{tabular}{@{}lp{0.29\textwidth}rrrcp{0.17\textwidth}@{}}", r"\toprule",
        r"& Test & Estimate & 95\% CI & Holm $p$ & Sizes & Verdict \\", r"\midrule"]
for h, lab, _ in rows:
    if h == "M1":
        b = R["M1"]["by_base"]
        est = "; ".join(f(b[k]["know_at_1000"]["mean"]) for k in ("regular_grouped", "regular_random", "reassigned_grouped"))
        out.append(f"M1 & {lab} & {est} & & & & {'Passed' if R['M1']['pass'] else 'Failed'} \\\\")
        continue
    r = R["H33"]["a"] if h == "H33a" else R["H33"]["b"] if h == "H33b" else R[h]
    key = "H33" if h.startswith("H33") else h
    hp = p_fmt(adj[key]) if key in adj else ("--" if h != "R1" else p_fmt(r["p"]) + "$^a$")
    sizes = r.get("sizes_in_direction", r.get("sizes_above_half", ""))
    if h == "H31":
        verdict = "Different, and equivalent within 1 nat"
    elif h == "H35":
        verdict = "Not tested (gate)"
    elif h == "R1":
        verdict = "Replicated"
    elif h.startswith("H33"):
        verdict = "Not supported" if not R["H33"].get("supported") else "Supported"
    else:
        verdict = "Supported" if R[h].get("supported") else "Not supported"
    out.append(f"{h} & {lab} & {f(r['mean'], 3 if h == 'H35' else 2)} & [{f(r['ci'][0], 3 if h == 'H35' else 2)}, "
               f"{f(r['ci'][1], 3 if h == 'H35' else 2)}] & {hp} & {sizes} & {verdict} \\\\")
out += [r"\bottomrule", r"\end{tabular}",
        r"\\[2pt]{\footnotesize $^a$ Outside the Holm family (replication); unadjusted one-sided $p$. M1 values: Regular "
        r"grouped; Regular random; Reassigned grouped (all with slopes above zero, $p<.001$). H31 TOST $p<.001$.}",
        r"\end{table}", ""]

# Table M2: carry by cell
def sv(base, typ, e=None, emb="original", q="carry"):
    d = s[(s.base == base) & (s.type == typ) & (s.emb == emb)]
    if e is not None:
        d = d[d.exposures == e]
    v = d.groupby(["size", "seed"])[q].mean().groupby("seed").mean()
    return ci(v) if len(v) > 1 else (np.nan,) * 3


bases = [("regular_grouped", "Regular grouped"), ("reassigned_grouped", "Reassigned grouped"),
         ("regular_random", "Regular random")]
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Carry (nats) by noun type and exposure, mean of seed values (sizes averaged). Unseen: never in "
        r"training; typical: the unseen nouns with a typical embedding; trained: the 32 base kinds.}",
        r"\label{tab:M2}", r"\begin{tabular}{@{}lrrrlrrrr@{}}", r"\toprule",
        r"Corpus & Unseen & Typical & Trained & Familiar nouns & 10 & 100 & 1k & 10k \\", r"\midrule"]
for bi, (b, lab) in enumerate(bases):
    for i, (typ, tl) in enumerate((("consistent", "Fixed shape"), ("uninformative", "Uninformative"),
                                   ("absent", "Shape-absent"))):
        head = (f"{lab} & {f(sv(b, 'unseen')[0])} & {f(sv(b, 'unseen', emb='typical')[0])} & {f(sv(b, 'trained')[0])}"
                if i == 0 else " & & &")
        out.append(head + f" & {tl} & " + " & ".join(f(sv(b, typ, e)[0]) for e in (10, 100, 1000, 10000)) + r" \\")
    if bi < len(bases) - 1:
        out.append(r"\addlinespace")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Table M3: knowledge and stored by cell, Regular grouped
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Knowledge and stored preference (nats) in Regular grouped models, mean of seed values.}",
        r"\label{tab:M3}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Type & Measure & 10 & 100 & 1k & 10k \\", r"\midrule"]
for typ, tl in (("consistent", "Fixed shape"), ("uninformative", "Uninformative"), ("absent", "Shape-absent")):
    for q, ql in (("know", "Knowledge"), ("stored", "Stored")):
        out.append(f"{tl if q == 'know' else ''} & {ql} & " +
                   " & ".join(f(sv('regular_grouped', typ, e, q=q)[0]) for e in (10, 100, 1000, 10000)) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Table M4: heads by base and size
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Head analysis by corpus and size (means over ten seeds). Candidates were the $k$ heads with the "
        r"highest product of prefix-matching and copying scores; in grouped models they were in layer 0 in all but one model (a 3.4M Regular grouped model chose one layer-1 head), and in random-order models a few were in later layers. In the 3.4M and 11.0M models layer 0 has exactly $k$ non-candidate heads, so the random draws repeat one set. Prefix: highest "
        r"prefix-matching score of any head (uniform attention gives about 0.027). Carry of unseen nouns and of "
        r"trained kinds intact, with the candidates mean-ablated, and with random heads ablated (mean of 200 or 20 "
        r"draws). DLA: change in unseen carry predicted by the candidates' direct paths.}",
        r"\label{tab:M4}", r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
        r" & & & & \multicolumn{3}{c}{Unseen carry} & \multicolumn{3}{c}{Trained carry} & \\",
        r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Corpus & Size & $k$ & Prefix & Intact & Cand. & Random & Intact & Cand. & Random & DLA \\", r"\midrule"]
mxp = {}
for fpath in glob.glob("results/amendment9/circuits/heads_*.json"):
    j = json.load(open(fpath))
    S = j["summary"]
    if not S["ckpt"].endswith("model_final.pt") or S["seed"] not in CONF_SEEDS:
        continue
    mxp.setdefault((S["base"], S["size"]), []).append(max(r["prefix"] for r in j["signature"]))
for b, lab in bases:
    for z, zl in (("tiny", "3.4M"), ("small", "11.0M"), ("medium", "25.6M")):
        d = hs[(hs.base == b) & (hs["size"] == z)].mean(numeric_only=True)
        out.append(f"{lab if z == 'tiny' else ''} & {zl} & {int(d['k'])} & {f(np.mean(mxp[(b, z)]), 3)} & "
                   f"{f(d['unseen_carry'])} & {f(d['unseen_carry_ablated'])} & {f(d['unseen_carry_random_mean'])} & "
                   f"{f(d['trained_carry'])} & {f(d['trained_carry_ablated'])} & {f(d['trained_carry_random_mean'])} & "
                   f"{f(d['dla_predicted_change'])} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

# Table M5: attention by cell
out += [r"\begin{table}[!htbp]", r"\centering\small",
        r"\caption{Attention of the candidate heads from the position that predicts the shape to the context's shape "
        r"word (mean over candidates; mean of seed values; exploratory).}",
        r"\label{tab:M5}", r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
        r"Corpus & Type & Unseen & 10 & 100 & 1k & 10k \\", r"\midrule"]
for b, lab in bases:
    d = pn[pn.base == b]
    un = d[d.type == "unseen"].groupby(["size", "seed"]).attention.mean().groupby("seed").mean().mean()
    for i, (typ, tl) in enumerate((("consistent", "Fixed shape"), ("uninformative", "Uninformative"),
                                   ("absent", "Shape-absent"))):
        vals = [d[(d.type == typ) & (d.exposures == e)].groupby(["size", "seed"]).attention.mean()
                .groupby("seed").mean().mean() for e in (10, 100, 1000, 10000)]
        out.append(f"{lab if i == 0 else ''} & {tl} & {f(un, 3) if i == 0 else ''} & " +
                   " & ".join(f(v, 3) for v in vals) + r" \\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
print("\n".join(out))
