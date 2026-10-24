"""Build paper_acl/appendix.tex from the long version (paper_latex/body.tex).

Appendices A-H are the long version's appendices; I-K hold the parts of Studies 1-3
moved out of the 8-page main text. Identifying links are removed for review.
"""
import re, sys
body = open(sys.argv[1]).read().split("\n")
L = lambda a, b: "\n".join(body[a - 1:b])          # 1-based inclusive line range

old = L(336, 747)
old = old.replace(
    r"All registrations are in one OSF project, \url{osf.io/qj9hb}, and each states what was known when it was written.",
    r"All registrations are in one public project on the Open Science Framework (link withheld for review; an anonymised view-only link is in the supplementary material), and each states what was known when it was written.")
old = old.replace("untrained (Section 3.5)", r"untrained (Section~\ref{sec:background})")
old = old.replace("is rare (Section 3.5)", r"is rare (Section~\ref{sec:background})")
old = old.replace("track novel nouns (Section 3.5)", r"track novel nouns (Section~\ref{sec:background})")
old = old.replace("format of Section 3.3", r"format of Section~\ref{sec:background}")
old = old.replace("exploratory (Table 3)", r"exploratory (Table~\ref{tab:3})")
old = old.replace("scale for Figure 2", r"scale for Figure~\ref{fig:2}")
old = old.replace("(Table E4, Figure 2)", r"(Table E4, Figure~\ref{fig:2})")
m = re.search(r"All code, corpora, test items.*?(?=Study 2 is a secondary analysis)", old, re.S)
old = old[:m.start()] + (
    "All code, corpora, test items, model scores, transcript-level counts, hand codes (without transcript excerpts) "
    "and analysis outputs, and the final weights and training logs of the 510 trained Study 1 models, will be released "
    "on publication; anonymised copies of the code and results are in the supplementary material. A README gives the "
    "command that regenerates each table and figure, with MD5 checksums for the fixed inputs. The pretrained models are "
    "public at the revisions listed. CHILDES and CABank transcripts are available from TalkBank under its terms and are "
    "not redistributed. ") + old[m.end():]
for bad in ["synthiumjp", "osf.io", "huggingface", "github"]:
    assert bad not in old, bad
old = old.replace(r"\setcounter{figure}{0}\renewcommand{\thefigure}{A\arabic{figure}}",
    r"\setcounter{figure}{0}\renewcommand{\thefigure}{A\arabic{figure}}" + "\n"
    "In the appendices and the registrations, Study 1 is the training experiments (Section~\\ref{sec:background} and Appendices~\\ref{app:A} to~\\ref{app:I}), "
    "Study 2 the natural corpora (Appendix~\\ref{app:J}) and Study 3 the pretrained models (Appendix~\\ref{app:K}).\n", 1)
old = old.replace(r"\appendixsection{Materials for Studies 2 and 3}",
    "The comparison of novel and trained nouns in Table~\\ref{tab:L2} is exploratory and was added after the registrations; "
    "seed-level intervals for every fresh-text condition are produced by \\texttt{scripts/recut\\_noun\\_store\\_check.py}.\n\n"
    r"\FloatBarrier" + "\n" + r"\appendixsection{Materials for Studies 2 and 3}", 1)

A8ROW = "no fresh-text model trained \\\\\n"
assert A8ROW in old, "Amendment 8 row not found"
old = old.replace(A8ROW, A8ROW +
    "Amendment 9 & 23 Oct 2026 & Familiarity study (Appendix~\\ref{app:M}) & All results above; no familiarity corpus built \\\\\n"
    "Amendment 10 & 24 Oct 2026 & Recurrence, frames, rows and pretrained models (Appendix~\\ref{app:N}) & Amendment 9 results and "
    "a blind review of them; no Amendment 10 corpus used, model trained or pretrained model run on its items \\\\\n", 1)


def hdr(letter, title, label):
    return ("\\FloatBarrier\n\\section{%s}\\label{%s}\n\\setcounter{table}{0}\\renewcommand{\\thetable}{%s\\arabic{table}}\n"
            "\\setcounter{figure}{0}\\renewcommand{\\thefigure}{%s\\arabic{figure}}\n") % (title, label, letter, letter)

def fix(t):
    t = t.replace("(Figure 1)", r"(Figure~\ref{fig:1})")
    t = re.sub(r"\(Table 1\)", r"(Table~\ref{tab:1})", t)
    t = t.replace("(Table 1 gives", r"(Table~\ref{tab:1} gives")
    t = t.replace("(Table 3)", r"(Table~\ref{tab:3})")
    t = t.replace("Table 1 shows", r"Table~\ref{tab:1} shows")
    t = t.replace("(Figure 2)", r"(Figure~\ref{fig:2})")
    t = t.replace("(Table E4, Figure 2)", r"(Table E4, Figure~\ref{fig:2})")
    t = t.replace("(Figure 3)", r"(Figure~\ref{fig:3})")
    t = t.replace("Section 5.2", "Appendix~\\ref{app:K}")
    t = t.replace("The author coded", "One author coded")
    t = t.replace("[!tbp]", "[!htbp]")
    return t


FIG2 = r"""
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/figure2.png}
\caption{Index against clustering $B$ in Regular corpora, unseen test tokens. Grey points and solid line: fixed-file models (mean of five per size; line, mean of 15). Open circles and dashed line: fresh-text models (mean of 15). Bands: effective clustering of different objects in natural corpora with 95\% intervals (Appendix~\ref{app:J}); the overlay does not predict what a model trained on them would learn. With typical test tokens or trained nouns the fresh-text effect at $B = 3.5$ disappears (Appendix~\ref{app:L}).}
\label{fig:2}
\end{figure}
"""
I = hdr("I", "Study 1: corpora, training and further results", "app:I") + fix(
    "The main text summarises the design. This appendix gives it in full, with results not reported there.\n\n"
    + L(66, 90) + "\n" + L(91, 113) + "\n\n\\subsection{Models and training}\n" + L(123, 123)
    + "\n\n\\subsection{Measuring the expectation}\n" + L(126, 142)
    + "\n\n\\subsection{Statistics}\n" + L(145, 145)
    + "\n\n\\subsection{Further results}\n" + L(175, 175) + "\n\n" + L(177, 177) + "\n\n" + L(191, 191) + "\n\n" + L(193, 193)) + FIG2
J = hdr("J", "Study 2: clustering of natural input", "app:J") + fix(L(208, 250))
K = hdr("K", "Study 3: pretrained models", "app:K") + fix(L(253, 282))
out = old + "\n\n" + I + "\n\n" + J + "\n\n" + K + "\n"
for bad in ["synthiumjp", "osf.io", "huggingface", "github", "Cacioli"]:
    assert bad not in out, bad
seen = set()


def _dedup(m):
    lab, base, k = m.group(1), m.group(1), 1
    while lab in seen:
        lab, k = f"{base}x{k}", k + 1
    seen.add(lab)
    return r"\label{" + lab + "}"


out = re.sub(r"\\label\{([^}]*)\}", _dedup, out)
open(sys.argv[2], "w").write(out)
print("ok", len(out))
