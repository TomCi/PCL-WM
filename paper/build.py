# -*- coding: utf-8 -*-
"""Assemble the manuscript .docx from the block lists (Elsevier-like layout)."""
import os
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Emu
from PIL import Image

import content1
import omml
import content2
import content3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "PCL-WM_EAAI_manuscript.docx")

SERIF = "Times New Roman"
BODY_PT = 10.5
CAP_PT = 9.5
TABLE_PT = 9.0
TEXT_W_CM = 16.0

TITLE = ("Which physics priors actually help? A component-wise evaluation of "
         "structured latent world models for engineering control")
AUTHORS = [("Tian-ci ZHANG", "a,*"), ("Shuai HOU", "b"), ("Yong-yong WEI", "b"), ("Hao Fang", "c")]
AFFIL = ("a School of Automation, Beijing Institute of Technology, Beijing 100081, China\n"
         "b School of Information and Electrical Engineering, Hebei University of Engineering, "
         "Handan 056038, China\n"
         "c School of Aeronautics, Beijing Institute of Technology, Beijing 100081, China\n"
         "* Corresponding author. E-mail address: 249918072@qq.com (Tian-ci Zhang).")

HEADERS = {
    0: ["Domain", "Plant", "n", "p", "n_o", "Measured channels"],
    1: ["Model", "D1 @25", "D1 @100", "D1 @200", "D2 @200", "D3 @200", "D9 @200",
        "beta D1", "beta D2", "Latency (ms)", "Parameters"],
    2: ["Model", "D1", "D2", "D3", "D9"],
    3: ["Configuration", "D1 @25", "D1 @50", "D1 @200", "beta", "One-step", "Parameters"],
    4: ["Model", "ID @25", "ID @100", "ID @200", "beta ID", "OOD @100", "OOD @200", "beta OOD"],
    5: ["Model", "Closed-loop RMSE", "Range of three runs"],
    6: ["Model", "Energy deviation (%)"],
    7: ["Model", "Parameters", "Latency (ms)", "Training (s)"],
    8: ["Domain", "Plant", "With map", "Without map", "Ratio", "beta with / without", "Energy drift (%)"],
    9: ["Model", "@25", "@50", "@100", "beta", "Parameters", "Latency (ms)", "Training (s)"],
    10: ["Domain", "Filter", "@25", "@50", "@100", "@200", "beta"],
    11: ["Plant signature", "Keep", "Avoid or drop", "Evidence in this paper"],
    12: ["Item", "Value"],
}


def set_run(run, size=BODY_PT, bold=False, italic=False):
    run.font.name = SERIF
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), SERIF)


CMDS = [
    ("\\varepsilon", "\u03b5"), ("\\gamma", "\u03b3"), ("\\beta", "\u03b2"),
    ("\\delta", "\u03b4"), ("\\rho", "\u03c1"), ("\\lambda", "\u03bb"),
    ("\\sigma", "\u03c3"), ("\\phi", "\u03c6"), ("\\psi", "\u03c8"),
    ("\\theta", "\u03b8"), ("\\xi", "\u03be"), ("\\top", "T"),
    ("\\Vert", "\u2016"), ("\\max", "max"), ("\\mathrm{cert}", "cert"),
    ("\\mathbb{R}", "\u211d"), ("\\hat{o}", "\u00f4"), ("\\in", "\u2208"),
    ("\\leq", "\u2264"), ("\\,", "\u2009"),
    ("\\ ", " "), ("\\", ""),
]


def tex_segments(tex):
    """Split a tiny LaTeX fragment into (text, sub, sup) segments for Word runs."""
    s = tex
    for a, b in CMDS:
        s = s.replace(a, b)
    segs, buf, i = [], "", 0

    def flush():
        nonlocal buf
        if buf:
            segs.append((buf, False, False))
            buf = ""

    while i < len(s):
        c = s[i]
        if c in "_^" and i + 1 < len(s):
            flush()
            if s[i + 1] == "{":
                j = s.index("}", i + 2)
                segs.append((s[i + 2:j], c == "_", c == "^"))
                i = j + 1
            else:
                segs.append((s[i + 1], c == "_", c == "^"))
                i += 2
            continue
        buf += c
        i += 1
    flush()
    return segs


def write_text(p, text, size=BODY_PT, bold=False):
    """Add runs to a paragraph, converting $...$ fragments into italic/sub/sup runs."""
    import re
    chunks = re.split(r"(\$[^$]*\$)", text)
    for ch in chunks:
        if not ch:
            continue
        if ch.startswith("$") and ch.endswith("$"):
            for seg, sub, sup in tex_segments(ch[1:-1]):
                if not seg:
                    continue
                r = p.add_run(seg)
                set_run(r, size, bold, italic=True)
                if sub:
                    r.font.subscript = True
                if sup:
                    r.font.superscript = True
        else:
            r = p.add_run(ch)
            set_run(r, size, bold, italic=False)


def para(doc, text="", size=BODY_PT, bold=False, italic=False, align=None, before=0,
         after=6, first_indent=0.0, spacing=1.08):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = spacing
    pf.first_line_indent = Cm(first_indent)
    if align is not None:
        pf.alignment = align
    if text:
        write_text(p, text, size, bold)
    return p


def border(el, edge, sz=8, color="000000"):
    for tag in ("w:%s" % edge,):
        e = OxmlElement(tag)
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), color)
        el.append(e)


def three_line(table, n_rows):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 16), ("bottom", 16)):
        e = OxmlElement("w:%s" % edge)
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
        borders.append(e)
    for edge in ("left", "right", "insideH", "insideV"):
        e = OxmlElement("w:%s" % edge)
        e.set(qn("w:val"), "none")
        e.set(qn("w:sz"), "0")
        e.set(qn("w:space"), "0")
        borders.append(e)
    tblPr.append(borders)
    for c in table.rows[0].cells:
        tcPr = c._tc.get_or_add_tcPr()
        tb = OxmlElement("w:tcBorders")
        e = OxmlElement("w:bottom")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "8")
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
        tb.append(e)
        tcPr.append(tb)


def cell_text(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=TABLE_PT):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.left_indent = Cm(0)
    p.alignment = align
    write_text(p, text, size, bold)


def caption(doc, text, before=2, after=4):
    """Split 'Fig. n.' / 'Table n.' prefix and set it bold; rest justified."""
    body = text
    prefix = ""
    for marker in ("Fig. ", "Table "):
        if text.startswith(marker):
            idx = text.find(".", len(marker))
            # appendix captions read "Table A.1." -> skip the letter dot
            if idx > 0 and len(text) > idx + 1 and text[idx + 1].isdigit():
                idx = text.find(".", idx + 1)
            if idx > 0:
                prefix, body = text[:idx + 1], text[idx + 1:].strip()
            break
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = 1.0
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if prefix:
        set_run(p.add_run(prefix + " "), CAP_PT, bold=True)
    set_run(p.add_run(body), CAP_PT)
    return p


def heading(doc, text, level):
    """Numbered sections use the real Heading styles; front matter keeps direct formatting."""
    numbered = bool(re.match(r"^\d+\.", text)) or text.startswith("Appendix") or text == "References"
    if not numbered:
        para(doc, text, size=12 if level == 1 else 10.5, bold=True,
             italic=(level == 2), before=14 if level == 1 else 8,
             after=5 if level == 1 else 3, spacing=1.0)
        return doc.paragraphs[-1]
    p = doc.add_paragraph(style="Heading %d" % level)
    set_run(p.add_run(text), 12 if level == 1 else 10.5, bold=True, italic=(level == 2))
    return p


def add_equation(doc, num):
    """Place a native Word equation between a centred tab and the right-aligned number."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.0
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.left_indent = Cm(0)
    pf.first_line_indent = Cm(0)
    pf.tab_stops.add_tab_stop(Cm(8.2), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Cm(15.9), WD_TAB_ALIGNMENT.RIGHT)
    r = p.add_run("\t")
    set_run(r)
    p._p.append(omml.equation_element(num))
    r = p.add_run("\t(%d)" % num)
    set_run(r, BODY_PT)


def add_table(doc, idx, caption_text, header, rows):
    hdr = header or HEADERS.get(idx, [])
    tsize = 7.5 if len(hdr) >= 9 else TABLE_PT
    caption(doc, caption_text, before=8, after=3)
    t = doc.add_table(rows=1 + len(rows), cols=len(hdr))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for j, h in enumerate(hdr):
        cell_text(t.rows[0].cells[j], h, bold=True, size=tsize)
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            al = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            cell_text(t.rows[i].cells[j], v, align=al, size=tsize)
    three_line(t, len(rows))
    for i, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
        if i == 0:
            th = OxmlElement("w:tblHeader")
            trPr.append(th)
    para(doc, "", after=2, size=4)


def add_figure(doc, path, caption_text):
    full = path if os.path.isabs(path) else os.path.join(HERE, path)
    w, h = Image.open(full).size
    width = Cm(15.5)
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(8)
    pf.space_after = Pt(2)
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.line_spacing = 1.0
    pf.keep_with_next = True
    p.add_run().add_picture(full, width=width)
    caption(doc, caption_text, before=2, after=10)


def add_nomenclature(doc, rows):
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    t = doc.add_table(rows=half, cols=4)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i in range(half):
        for j in range(2):
            if i < len(left if j == 0 else right):
                sym, desc = (left if j == 0 else right)[i]
                cell_text(t.rows[i].cells[2 * j], sym, align=WD_ALIGN_PARAGRAPH.LEFT)
                cell_text(t.rows[i].cells[2 * j + 1], desc, align=WD_ALIGN_PARAGRAPH.LEFT)
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
        e = OxmlElement("w:%s" % edge)
        e.set(qn("w:val"), "none")
        borders.append(e)
    tblPr.append(borders)
    para(doc, "", after=2, size=4)


def add_reference(doc, num, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.0
    pf.left_indent = Cm(0.75)
    pf.first_line_indent = Cm(-0.75)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_run(p.add_run("[%d] " % num), 9.5)
    set_run(p.add_run(text), 9.5)


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(2)
    pf.line_spacing = 1.0
    pf.left_indent = Cm(0.6)
    pf.first_line_indent = Cm(-0.35)
    r = p.add_run(text)
    set_run(r, 9.5)
    return p


def renumber_citations(blocks):
    """Renumber references in order of first citation (Elsevier numbered style)."""
    import re
    pat = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")

    order = []
    for b in blocks:
        if b[0] in ("h1", "h2", "p", "bullet"):
            for m in pat.finditer(b[1]):
                for n in m.group(1).split(","):
                    n = int(n.strip())
                    if n not in order:
                        order.append(n)
    mapping = {old: i + 1 for i, old in enumerate(order)}
    return mapping


def apply_mapping(blocks, mapping):
    import re
    pat = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")

    def sub(m):
        nums = sorted({mapping[int(x.strip())] for x in m.group(1).split(",")})
        return "[" + ", ".join(str(n) for n in nums) + "]"

    out = []
    for b in blocks:
        if b[0] in ("h1", "h2", "p", "bullet"):
            out.append((b[0], pat.sub(sub, b[1])))
        else:
            out.append(b)
    return out


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(sec, m, Cm(2.5))

    st = doc.styles["Normal"]
    st.font.name = SERIF
    st.font.size = Pt(BODY_PT)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.08

    for name, size, italic, before, after in (("Heading 1", 12, False, 14, 5),
                                              ("Heading 2", 10.5, True, 8, 3)):
        hs = doc.styles[name]
        hs.font.name = SERIF
        hs.font.size = Pt(size)
        hs.font.bold = True
        hs.font.italic = italic
        hs.font.color.rgb = RGBColor(0, 0, 0)
        hs.paragraph_format.space_before = Pt(before)
        hs.paragraph_format.space_after = Pt(after)
        hs.paragraph_format.line_spacing = 1.0
        hs.paragraph_format.first_line_indent = Cm(0)
        hs.paragraph_format.left_indent = Cm(0)
        hs.paragraph_format.keep_with_next = True
        hs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # centred page number in the footer
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    set_run(run, 9)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    fp._p.append(fld)

    para(doc, TITLE, size=16, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, after=10, spacing=1.0)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    for i, (name, sup) in enumerate(AUTHORS):
        if i:
            set_run(p.add_run(", "), 11)
        set_run(p.add_run(name), 11)
        r = p.add_run(sup)
        set_run(r, 11)
        r.font.superscript = True
    for line in AFFIL.split("\n"):
        para(doc, line, size=8.5, after=2, spacing=1.0)

    blocks = content1.BLOCKS + content2.BLOCKS + content3.BLOCKS
    mapping = renumber_citations(blocks)
    blocks = apply_mapping(blocks, mapping)

    refs = []
    for b in blocks:
        if b[0] == "refs":
            refs = b[1]
    if refs:
        new_refs = [None] * len(refs)
        for old, new in mapping.items():
            if 1 <= old <= len(refs):
                new_refs[new - 1] = refs[old - 1]
        new_refs = [r for r in new_refs if r]
        blocks = [(a, new_refs) if a == "refs" else (a, *rest)
                  for (a, *rest) in blocks for _ in [0]]
        print("citations renumbered, refs", len(new_refs), "of", len(refs))

    tbl_idx = 0
    ref_no = 0
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            heading(doc, b[1], level=1)
        elif kind == "h2":
            heading(doc, b[1], level=2)
        elif kind == "p":
            para(doc, b[1], align=WD_ALIGN_PARAGRAPH.JUSTIFY)
        elif kind == "bullet":
            bullet(doc, b[1])
        elif kind == "eq":
            add_equation(doc, b[1])
        elif kind == "table":
            add_table(doc, tbl_idx, b[1], b[2], b[3])
            tbl_idx += 1
        elif kind == "fig":
            add_figure(doc, b[1], b[2])
        elif kind == "nomen":
            add_nomenclature(doc, b[1])
        elif kind == "refs":
            for t in b[1]:
                ref_no += 1
                add_reference(doc, ref_no, t)
        else:
            raise ValueError("unknown block %r" % (kind,))

    doc.save(OUT)
    words = sum(len(str(x).split()) for b in blocks if b[0] in ("p", "bullet", "h1", "h2")
                for x in b[1:2])
    print("saved", OUT, os.path.getsize(OUT), "bytes; approx words", words,
          "| tables", tbl_idx, "| refs", ref_no)


if __name__ == "__main__":
    main()
