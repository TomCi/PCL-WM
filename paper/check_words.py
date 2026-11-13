# -*- coding: utf-8 -*-
"""Report the manuscript word count the way the 16k cap is applied."""
import re

import docx

DOC = r"C:\Users\danta\Doubao\chats\2026-11-07\new-chat\PCL-WM_EAAI_manuscript.docx"
d = docx.Document(DOC)

paras = [p.text.strip() for p in d.paragraphs]
i_ref = next(k for k, t in enumerate(paras) if t.startswith("References"))
body = paras[:i_ref]
refs = paras[i_ref:]

def wc(lines):
    return sum(len(re.findall(r"[A-Za-z][A-Za-z'\-]*", t)) for t in lines)

tbl = 0
for t in d.tables:
    for row in t.rows:
        for c in row.cells:
            tbl += len(re.findall(r"[A-Za-z][A-Za-z'\-]*", c.text))

print("body paragraphs (incl. title/captions):", wc(body))
print("tables                                :", tbl)
print("references                            :", wc(refs))
print("body + tables                         :", wc(body) + tbl)

for p in paras:
    if p.startswith("One might ask for a second"):
        print("\n" + p)
