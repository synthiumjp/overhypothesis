#!/usr/bin/env python3
"""Merge paginated Claude Docs 'view' reads (saved tool-result JSON files) into one doc XML for doc_to_tex.py.

Usage: python merge_view_pages.py out.xml page1.json page2.json ... [--table TABLE_ID=markdown_file]
Each page file is the saved tool result (JSON list whose last element holds the read result).
Root-level pages replace the matching <gap ... from='ID'/>; a gap left inside a table can be filled
from a pipe-table markdown file with --table (the whole table element is rebuilt from it).
"""
import html, json, re, sys

args = sys.argv[1:]
out = args.pop(0)
tables = {}
while "--table" in args:
    i = args.index("--table"); tid, path = args[i + 1].split("="); tables[tid] = open(path).read(); del args[i:i + 2]

def xml_of(path):
    a = json.load(open(path))
    return json.loads(a[-1]["text"])["data"]["xml"]

doc = xml_of(args[0])
for p in args[1:]:
    x = xml_of(p)
    body = x[x.index(">") + 1: x.rindex("</doc>")]
    body = re.sub(r"^<gap blocks='\d+'/>", "", body)
    m = re.search(r"<(\w+) id='([^']+)'", body)
    first = m.group(2) if m else None
    gaps = re.findall(r"<gap [^>]*from='([^']+)'/>", doc)
    target = next(g for g in gaps if first and (g == first or g.endswith(first)))
    doc = re.sub(r"<gap [^>]*from='" + re.escape(target) + r"'/>", lambda _: body, doc, count=1)

def cell(t):
    return "<cell><paragraph><text>" + html.escape(t, quote=False) + "</text></paragraph></cell>"
for tid, md in tables.items():
    rows = [r.strip().strip("|").split("|") for r in md.strip().splitlines() if not re.match(r"^\|\s*-", r)]
    new = "<table id='" + tid + "'>" + "".join("<row>" + "".join(cell(c.strip()) for c in r) + "</row>" for r in rows) + "</table>"
    short = "." + tid.split(".")[-1]
    doc, n = re.subn(r"<table id='(?:" + re.escape(tid) + "|" + re.escape(short) + r")'.*?</table>", lambda _: new, doc, count=1, flags=re.S)
    if not n:
        sys.exit("table not found: " + tid)
left = re.findall(r"<gap[^>]*>", doc)
if left:
    sys.exit("gaps left: " + str(left[:3]))
open(out, "w", encoding="utf-8").write(doc)
print("merged", len(doc))
