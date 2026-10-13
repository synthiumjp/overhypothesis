#!/usr/bin/env python3
"""Turn the paper's reference list (CL style, one entry per paragraph) into BibTeX.

Usage: python refs_to_bib.py doc.xml paper.bib keys.json
keys.json maps (first-author surname, year) to the BibTeX key, plus the in-text author string,
for the citation converter.
"""
import html
import json
import re
import sys
import unicodedata

xml, out_bib, out_keys = sys.argv[1:4]
x = open(xml, encoding="utf-8").read()
paras = re.findall(r"<paragraph[^>]*>(.*?)</paragraph>", x, re.S)
texts = [html.unescape(re.sub(r"<[^>]+>", "", p)).strip() for p in paras]
i = texts.index("References")
j = next(k for k in range(i + 1, len(texts)) if texts[k].startswith("Appendix A."))
refs = [t for t in texts[i + 1:j] if t]

INIT = re.compile(r"^(?:[A-Z][a-z]?\.(?:-[A-Z]\.)?\s?)+$")


def split_authors(s):
    s = s.strip()
    if s == "Qwen Team":
        return ["{Qwen Team}"], "Qwen Team"
    etal = s.endswith("et al")
    if etal:
        s = s[: -len("et al")].rstrip(", ")
    if re.search(r"[A-Z]$", s):
        s += "."
    # first author: "Surname, I. I."
    m = re.match(r"^([^,]+),\s*((?:[A-Z][a-z]?\.(?:-[A-Z]\.)?\s?)+)(?:,\s*|\s+and\s+|$)(.*)$", s)
    first_sur, first_init, rest = m.group(1).strip(), m.group(2).strip(), m.group(3)
    names = [(first_sur, first_init)]
    rest = re.sub(r"^and\s+", "", rest.strip())
    parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", rest) if p.strip()]
    for p in parts:
        toks = p.split()
        k = 0
        while k < len(toks) and INIT.match(toks[k]):
            k += 1
        names.append((" ".join(toks[k:]), " ".join(toks[:k])))
    bib = []
    for sur, ini in names:
        sur_b = "{" + sur + "}" if " " in sur and not sur.startswith(("van ", "de ", "von ")) else sur
        bib.append(f"{sur_b}, {ini}".strip(", "))
    if etal:
        bib.append("others")
    surnames = [n[0] for n in names]
    if etal or len(surnames) >= 3 and False:
        cite = surnames[0] + " et al."
    elif len(surnames) == 1:
        cite = surnames[0]
    elif len(surnames) == 2:
        cite = f"{surnames[0]} and {surnames[1]}"
    else:
        cite = None  # 3+ authors: matched by the converter in several forms
    return bib, surnames


def keyify(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


entries, keymap = [], []
for r in refs:
    m = re.match(r"^(.*?)\.\s(\d{4}[ab]?)\.\s(.*)$", r)
    auth, year, rest = m.groups()
    bib_auth, surnames = split_authors(auth)
    yr = year[:4]
    # title = up to the first ". " that is followed by a venue marker
    doi = None
    md = re.search(r"\s(https?://\S+?)\.?$", rest)
    if md:
        doi = md.group(1)
        rest = rest[: md.start()].rstrip()
    tm = re.match(r"^(.*?[^A-Z])\.\s(.*)$", rest)
    if tm and not rest.startswith("CALL"):
        title, venue = tm.group(1), tm.group(2).rstrip(".")
    else:
        # LDC titles contain parentheses; split at "). "
        k = rest.find("). ")
        title, venue = rest[: k + 1], rest[k + 3:].rstrip(".")
    fields = {"author": " and ".join(bib_auth), "year": year, "title": "{" + title + "}"}
    etype = "misc"
    jm = re.match(r"^(.*?),\s(\d+)(?:\((\d+)\))?:([\w–-]+)$", venue)
    if venue.startswith("In "):
        etype = "inproceedings"
        v = venue[3:]
        pm = re.search(r",\s*pages\s+([\d–-]+)$", v)
        if pm:
            fields["pages"] = pm.group(1).replace("–", "--")
            v = v[: pm.start()]
        if " Workshop on " in v or v.startswith(("Proceedings", "Advances", "The ")):
            pass
        # book chapter: "In Editors, editors, Title. Publisher, pages"
        cm = re.match(r"^(.*?), editors?, (.*)$", v)
        if cm:
            etype = "incollection"
            fields["editor"] = cm.group(1).replace(", and ", " and ").replace(", ", " and ")
            btitle, _, pub = cm.group(2).rpartition(". ")
            fields["booktitle"] = "{" + btitle + "}"
            fields["publisher"] = pub
        else:
            vm = re.match(r"^(.*?), (volume \d+ of .*)$", v)
            if vm:
                fields["booktitle"] = vm.group(1)
                fields["series"] = vm.group(2)
            else:
                fields["booktitle"] = v
    elif jm:
        etype = "article"
        fields["journal"] = jm.group(1)
        fields["volume"] = jm.group(2)
        if jm.group(3):
            fields["number"] = jm.group(3)
        p = jm.group(4)
        fields["pages"] = p.replace("–", "--")
    elif venue.startswith("arXiv:"):
        fields["note"] = venue
    elif re.match(r"^[^,]+(,.*)?(Press|Erlbaum|Brookes|Consortium|TalkBank)", venue) or venue.endswith(("Press", "Brookes", "Erlbaum", "TalkBank", "Consortium")):
        etype = "book" if not venue.endswith(("TalkBank", "Consortium")) else "misc"
        if etype == "book":
            if ", " in venue and not venue.endswith("Press"):
                ed, _, pub = venue.partition(". ")
            parts = venue.split(". ")
            if len(parts) == 2:
                fields["edition"] = parts[0].replace(" edition", "")
                fields["publisher"] = parts[1]
            else:
                fields["publisher"] = venue
        else:
            fields["howpublished"] = venue
    else:
        fields["howpublished"] = venue
    if doi:
        fields["note"] = (fields.get("note", "") + ". " if fields.get("note") else "") + r"\url{" + doi + "}"
    key = keyify(surnames[0]) + year
    entries.append((etype, key, fields))
    keymap.append({"key": key, "surnames": surnames, "year": year, "etal": "others" in bib_auth})

with open(out_bib, "w", encoding="utf-8") as f:
    for etype, key, fields in entries:
        f.write(f"@{etype}{{{key},\n")
        for k, v in fields.items():
            f.write(f"  {k} = {{{v}}},\n")
        f.write("}\n\n")
json.dump(keymap, open(out_keys, "w"), indent=1, ensure_ascii=False)
print(len(entries), "entries")
