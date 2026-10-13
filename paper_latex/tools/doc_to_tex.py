#!/usr/bin/env python3
"""Convert the paper (Claude Doc XML, all pages merged) into LaTeX body for clv2025.cls.

Usage: python doc_to_tex.py doc.xml keys.json body.tex
Writes the body (abstract to appendices) and prints a report of citations found and missed.
"""
import html
import json
import re
import sys

xml_path, keys_path, out_path = sys.argv[1:4]
x = open(xml_path, encoding="utf-8").read()
keys = json.load(open(keys_path, encoding="utf-8"))

# ---------------------------------------------------------------- citations
YEAR = r"\d{4}[ab]?"
forms = {}  # (form, year) -> key
for e in keys:
    s, y, k = e["surnames"], e["year"], e["key"]
    cands = []
    if isinstance(s, str) or s == ["Qwen Team"]:
        cands = ["Qwen Team"]
    elif e["etal"] or len(s) >= 4:
        cands = [s[0] + " et al."]
    elif len(s) == 3:
        cands = [f"{s[0]}, {s[1]}, and {s[2]}", s[0] + " et al."]
    elif len(s) == 2:
        cands = [f"{s[0]} and {s[1]}"]
    else:
        cands = [s[0]]
    for c in cands:
        forms[(c, y)] = k
FORM_RE = "|".join(sorted({re.escape(f) for f, _ in forms}, key=len, reverse=True))
used = set()


def keys_for(form, years):
    out = []
    for y in re.findall(YEAR, years):
        k = forms.get((form, y))
        if k is None:
            raise KeyError((form, y))
        out.append(k)
        used.add(k)
    return ",".join(out)


PIECE = re.compile(rf"^(?P<pre>.*?\s)?(?P<form>{FORM_RE})\s+(?P<years>{YEAR}(?:,\s*{YEAR})*)$")


def convert_paren(m):
    inner = m.group(1)
    pieces = [p.strip() for p in inner.split(";")]
    conv, any_cite, all_cite = [], False, True
    for p in pieces:
        pm = PIECE.match(p)
        if pm and (pm.group("form"), re.findall(YEAR, pm.group("years"))[0]) in forms:
            any_cite = True
            k = keys_for(pm.group("form"), pm.group("years"))
            if pm.group("pre"):
                all_cite = False
                conv.append(pm.group("pre") + r"\citealp{" + k + "}")
            else:
                conv.append(("CITE", k))
        else:
            all_cite = False
            conv.append(p)
    if not any_cite:
        return m.group(0)
    if all_cite:
        return r"\cite{" + ",".join(c[1] for c in conv) + "}"
    return "(" + "; ".join(r"\citealp{" + c[1] + "}" if isinstance(c, tuple) else c for c in conv) + ")"


def citations(t):
    # narrative: Form (Year[, Year])
    t = re.sub(rf"(?P<form>{FORM_RE})\s+\((?P<years>{YEAR}(?:,\s*{YEAR})*)\)",
               lambda m: r"\citet{" + keys_for(m.group("form"), m.group("years")) + "}"
               if (m.group("form"), re.findall(YEAR, m.group("years"))[0]) in forms else m.group(0), t)
    # parenthetical groups (no nested parentheses)
    t = re.sub(r"\(([^()]*)\)", convert_paren, t)
    return t


# ---------------------------------------------------------------- inline text
SUP = {"⁻": "-", "⁴": "4", "²": "2"}


def esc(t):
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        t = t.replace(a, b)
    return t


def inline(frag):
    s = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", frag, flags=re.S)
    s = re.sub(r"<italic>", "\x01", s)
    s = re.sub(r"</italic>", "\x02", s)
    s = re.sub(r"<bold>", "\x03", s)
    s = re.sub(r"</bold>", "\x04", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub("\x02\x01", "", s)  # merge adjacent italic runs
    s = re.sub("\x04\x03", "", s)
    # protect special tokens before escaping
    s = s.replace("B_eff", "\x05BEFF\x05")
    s = esc(s)
    s = citations(s)
    # math and symbols
    s = s.replace("\x05BEFF\x05", r"$B_{\mathrm{eff}}$")
    s = re.sub(r"\bP0\b", r"$P_0$", s)
    s = re.sub(r"10⁻([⁰¹²³⁴⁵⁶⁷⁸⁹])", lambda m: "$10^{-" + "⁰¹²³⁴⁵⁶⁷⁸⁹".index(m.group(1)).__str__() + "}$", s)
    s = re.sub(r"(\S)²", r"\1$^2$", s)
    rep = [("−", r"\textminus{}"), ("–", "--"), ("—", "---"), ("→", r"$\rightarrow$"), ("Δ", r"$\Delta$"),
           ("ρ", r"$\rho$"), ("×", r"$\times$"), ("±", r"$\pm$"), ("κ", r"$\kappa$"), ("·", r"$\cdot$"),
           (" ~ ", r" $\sim$ "), ("<", r"$<$"), (">", r"$>$"), ("…", r"\ldots{}")]
    for a, b in rep:
        s = s.replace(a, b)
    s = s.replace(r"$<$ .", r"$<$~.").replace(r"$<$ 0", r"$<$~0")
    s = re.sub(r"\[([^\]\[]{1,40}?), ([^\]\[]{1,40}?)\]", r"[\1,~\2]", s)
    # quotes
    s = re.sub(r'(^|[\s(\[\x01\x03])"', r"\1``", s)
    s = s.replace('"', "''")
    # urls
    s = re.sub(r"(?<![/\w])((?:github\.com|osf\.io|doi\.org)/[\w./-]*[\w/])", r"\\url{\1}", s)
    s = s.replace("\x01", r"\emph{").replace("\x02", "}").replace("\x03", r"\textbf{").replace("\x04", "}")
    return s.strip()


def plain(frag):
    return html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", frag, flags=re.S))).strip()


# ---------------------------------------------------------------- blocks
body = x[x.index(">") + 1: x.rindex("</doc>")]
pat = re.compile(r"<(paragraph|table|list|embed|blockquote|codeBlock)\b([^>]*?)(/?)>")
blocks, i = [], 0
while True:
    m = pat.search(body, i)
    if not m:
        break
    tag, attrs, sc = m.groups()
    if sc:
        blocks.append((tag, attrs, ""))
        i = m.end()
        continue
    depth, j = 1, m.end()
    p2 = re.compile(r"<(/?)" + tag + r"\b[^>]*?(/?)>")
    while depth:
        k = p2.search(body, j)
        if k.group(1):
            depth -= 1
        elif not k.group(2):
            depth += 1
        j = k.end()
    blocks.append((tag, attrs, body[m.end():k.start()]))
    i = j

FIG = {"d80f9a50-97dc": ("figure1.png", r"\textwidth"), "915e673a-f84e": ("figure2.png", r"\textwidth"),
       "fedf38da-79d5": ("figure3.png", r"\textwidth"), "48048bad-4312": ("figure4.png", r"0.86\textwidth")}


# hand-set relative column widths for the widest tables, keyed by their first two headers
OVERRIDE = {"Corpus|Order (B)": [19, 23, 9, 7, 7, 12, 8, 8],
            "#|Prediction": [3, 29, 19, 17, 10, 12, 15],
            "Corpus|Next sentence same kind": [42, 12, 10, 10, 10, 11, 10, 10, 8]}


def table_tex(inner, caption, label, here=False):
    rows = re.findall(r"<row\b[^>]*>(.*?)</row>", inner, re.S)
    grid = [re.findall(r"<cell\b[^>]*>(.*?)</cell>", r, re.S) for r in rows]
    n = max(len(r) for r in grid)
    def clen(t):
        return len(re.sub(r"\s+", " ", plain(t)))
    body_rows = grid[1:] if len(grid) > 1 else grid
    widths = []
    for c in range(n):
        body = max((clen(r[c]) if c < len(r) else 0) for r in body_rows)
        head_words = plain(grid[0][c]).split() if c < len(grid[0]) else []
        head = max((len(w) for w in head_words), default=0)
        widths.append(max(min(body, 26), round(head * 1.5), 4) + 3)
    hk = plain(grid[0][0]) + "|" + plain(grid[0][1]) if n > 1 else ""
    if hk in OVERRIDE and len(OVERRIDE[hk]) == n:
        widths = OVERRIDE[hk]
    total = sum(widths)
    spec = "".join(r">{\hsize=%.3f\hsize\raggedright\arraybackslash}X" % (n * w / total) for w in widths)
    size = r"\small" if n <= 4 else (r"\footnotesize" if n <= 8 else r"\scriptsize")
    if n >= 6:
        size += r"\setlength{\tabcolsep}{3.5pt}"
    long = len(grid) > 22
    if long:
        out = [r"\FloatBarrier\clearpage", "{" + size, r"\begin{xltabular}{\textwidth}{" + spec + "}",
               r"\multicolumn{" + str(n) + r"}{@{}p{\linewidth}@{}}{\refstepcounter{table}\label{" + label
               + r"}{\normalsize\textbf{Table \thetable}\newline " + caption + r"}}\\[6pt]", r"\toprule"]
    else:
        out = [r"\begin{table}[!htbp]" if here else r"\begin{table}[!tbp]", r"\caption{" + caption + "}", r"\label{" + label + "}", size,
               r"\begin{tabularx}{\textwidth}{" + spec + "}", r"\toprule"]
    for ri, r in enumerate(grid):
        cells = [inline(c) for c in r] + [""] * (n - len(r))
        nonempty = [c for c in cells if c.strip()]
        if ri > 0 and len(nonempty) == 1 and nonempty[0].startswith(r"\textbf{") and cells[0] == "":
            out.append(r"\multicolumn{" + str(n) + "}{l}{" + nonempty[0] + r"}\\")
            continue
        out.append(" & ".join(cells) + r" \\")
        if ri == 0:
            out.append(r"\midrule")
            if long:
                out.append(r"\endfirsthead")
                out.append(r"\toprule " + " & ".join(cells) + r" \\ \midrule \endhead")
                out.append(r"\bottomrule \endlastfoot")
    if long:
        out += [r"\end{xltabular}}", ""]
    else:
        out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]
    return "\n".join(out)


out = []
title = None
mode = "front"
appendix_started = False
pending = None  # ("table"|"figure", payload) waiting for its caption
k = 0


def flush_pending(caption=""):
    global pending
    if pending is None:
        return
    kind, payload, label = pending
    if kind == "table":
        out.append(table_tex(payload, caption, label, here=appendix_started))
    else:
        fn, width = payload
        out.append("\n".join([r"\begin{figure}[!tbp]", r"\centering",
                              r"\includegraphics[width=" + width + "]{figures/" + fn + "}",
                              r"\caption{" + caption + "}", r"\label{" + label + "}", r"\end{figure}", ""]))
    pending = None


tcount = 0
fcount = 0
while k < len(blocks):
    tag, attrs, inner = blocks[k]
    h = re.search(r"heading='(\d)'", attrs)
    lvl = int(h.group(1)) if h else 0
    text = plain(inner) if tag == "paragraph" else ""
    if tag == "paragraph" and pending is not None:
        cm = re.match(r"^(Table|Figure)\s+[A-H]?\d+\.\s*", text)
        if cm:
            cap = inline(inner)
            cap = re.sub(r"^\\emph\{(Table|Figure)\s+[A-H]?\d+\.\}\s*", "", cap)
            cap = re.sub(r"^\\emph\{(Table|Figure)\s*\}\\emph\{[A-H]?\d+\}\\emph\{\.\}\s*", "", cap)
            cap = re.sub(r"^\\emph\{(Table|Figure) [A-H]?\d+\.?\}\.?\s*", "", cap)
            flush_pending(cap)
            k += 1
            continue
        flush_pending("")
    if tag == "paragraph":
        if lvl == 1:
            title = text
        elif mode == "front" and lvl == 0:
            pass  # byline and author line
        elif lvl == 2:
            name = text
            if name == "Abstract":
                mode = "abstract"
                out.append(r"\begin{abstract}")
            elif name == "References":
                if mode == "ack":
                    out.append(r"\end{acknowledgments}")
                mode = "refs"
            elif name == "Acknowledgements":
                if mode == "abstract":
                    out.append(r"\end{abstract}")
                mode = "ack"
                out.append(r"\begin{acknowledgments}")
            elif name.startswith("Appendix"):
                if not appendix_started:
                    out.append(r"\bibliographystyle{compling}" + "\n" + r"\bibliography{paper}" + "\n\n" + r"\appendix")
                    appendix_started = True
                am = re.match(r"Appendix ([A-H])\.\s*(.*)", name)
                letter, nm = am.groups()
                out.append(r"\FloatBarrier" + "\n" + r"\appendixsection{" + esc(nm) + "}\\label{app:" + letter + "}")
                out.append(r"\setcounter{table}{0}\renewcommand{\thetable}{" + letter + r"\arabic{table}}")
                mode = "appendix"
            else:
                if mode == "abstract":
                    out.append(r"\end{abstract}")
                if mode == "ack":
                    out.append(r"\end{acknowledgments}")
                mode = "body"
                nm = re.sub(r"^\d+\s+", "", name)
                out.append(r"\section{" + esc(nm) + "}")
        elif lvl == 3:
            out.append(r"\subsection{" + esc(re.sub(r"^\d+\.\d+\s+", "", text)) + "}")
        elif lvl == 4:
            out.append(r"\subsubsection*{" + esc(text) + "}")
        elif mode == "refs":
            pass
        elif text:
            out.append(inline(inner) + "\n")
    elif tag == "list" and mode != "refs":
        kind = re.search(r"kind='(\w+)'", attrs).group(1)
        start = re.search(r"start='(\d+)'", attrs)
        env = "enumerate" if kind == "ordered" else "itemize"
        items = re.findall(r"<listItem\b[^>]*>(.*?)</listItem>", inner, re.S)
        out.append(r"\begin{" + env + "}" + (r"\setcounter{enumi}{" + str(int(start.group(1)) - 1) + "}" if start else ""))
        for it in items:
            out.append(r"\item " + inline(it))
        out.append(r"\end{" + env + "}" + "\n")
    elif tag == "blockquote":
        paras = re.findall(r"<paragraph\b[^>]*>(.*?)</paragraph>", inner, re.S) or [inner]
        out.append(r"\begin{quote}" + "\n" + "\\\\\n".join(inline(p) for p in paras) + "\n" + r"\end{quote}" + "\n")
    elif tag == "codeBlock":
        out.append(r"\begin{equation*}" + "\n" + plain(inner) + "\n" + r"\end{equation*}" + "\n")
    elif tag == "table":
        if mode == "appendix":
            tcount += 0
        pending = ("table", inner, f"tab:{len([o for o in out if 'begin{table}' in o]) + 1}")
    elif tag == "embed":
        ref = re.search(r"ref='node/([^']*)'", attrs).group(1)
        fcount += 1
        pending = ("figure", FIG[ref], f"fig:{fcount}")
    k += 1
flush_pending("")

tex = "\n".join(out)
open(out_path, "w", encoding="utf-8").write(tex)
missing = sorted({e["key"] for e in keys} - used)
print("title:", title)
print("cited keys:", len(used), "of", len(keys))
print("never cited:", missing)
left = re.findall(r"[A-Z][a-zé]+(?: et al\.)?,? (?:and [A-Z][a-z]+ )?\(?\d{4}[ab]?\)?", re.sub(r"\\cite\w*\{[^}]*\}", "", tex))
print("possible missed citations:", sorted(set(left))[:40])
