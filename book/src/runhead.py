#!/usr/bin/env python3
"""Overlay part + section running heads onto the rendered body PDF.

Chrome's headerTemplate is static across pages and Blink has no CSS running-head
support, so the head that names the *current* part and section is added here, after
rendering: we map every page to the part/section whose heading it falls under (by
matching source headings against extracted page text in document order, so stray
cross-references cannot advance the state), then stamp the head into the top margin.

    runhead.py BODY_PDF SECTIONS_DIR     # overlays in place (overwrites BODY_PDF)
    runhead.py --anchors SECTIONS_DIR    # dry run: print the parsed anchor list
"""
import sys, os, re, glob, html, io

WORD2ROMAN = {
 "zero":"0","one":"I","two":"II","three":"III","four":"IV","five":"V","six":"VI",
 "seven":"VII","eight":"VIII","nine":"IX","ten":"X","eleven":"XI","twelve":"XII",
 "thirteen":"XIII","fourteen":"XIV","fifteen":"XV","sixteen":"XVI","seventeen":"XVII",
 "eighteen":"XVIII","nineteen":"XIX","twenty":"XX","twentyone":"XXI",
 "twentytwo":"XXII","twentythree":"XXIII","twentyfour":"XXIV","twentyfive":"XXV",
 "twentysix":"XXVI","twentyseven":"XXVII","twentyeight":"XXVIII","twentynine":"XXIX",
 "thirty":"XXX","thirtyone":"XXXI","thirtytwo":"XXXII",
  "thirtythree":"XXXIII","thirtyfour":"XXXIV","thirtyfive":"XXXV",
}

def strip_tags(s): return html.unescape(re.sub(r"<[^>]+>", "", s))
def norm(s):        return re.sub(r"[^a-z0-9]", "", strip_tags(s).lower())
def disp(s):        # clean KaTeX out of a title for the running head
    s = re.sub(r"\$(.*?)\$", lambda m: re.sub(r"\\[a-zA-Z]+|[{}]", "", m.group(1)), s)
    return re.sub(r"\s+", " ", s).strip()

def parse_anchors(sections_dir):
    """Ordered list of running-head anchors, in document order."""
    anchors = []
    for f in sorted(glob.glob(os.path.join(sections_dir, "[0-9]*.html"))):
        t = open(f, encoding="utf-8").read(); base = os.path.basename(f)
        # part opener: .pnum carries the spelled-out number ("Part Seven"), unique to openers
        m = re.search(r'class="pnum">\s*Part\s+([A-Za-z][A-Za-z\- ]*?)\s*<', t)
        if m:
            word = re.sub(r"[^a-z]", "", m.group(1).lower())
            # Fail loudly rather than falling back to the spelled-out word,
            # which is how "Part Twenty-Two · Causal Inference" reached the
            # running heads of a whole part without any warning.
            if word not in WORD2ROMAN:
                raise SystemExit(f"runhead: no roman numeral for {word!r} — "
                                 f"add it to WORD2ROMAN")
            roman = WORD2ROMAN[word]
            h1 = re.search(r"<h1[^>]*>(.*?)</h1>", t, re.S)
            title = ""
            if h1:
                inner = re.sub(r'<span[^>]*font-weight:\s*300[^>]*>.*?</span>', '', h1.group(1), flags=re.S)
                inner = re.sub(r'<br\s*/?>', ' ', inner)
                title = disp(re.sub(r"\s+", " ", strip_tags(inner)).strip())
            label = f"Part {roman}" + (f" · {title}" if title else "")
            anchors.append(dict(t="PART", key="part"+word, part=label, sec="", opener=True))
        elif base[0] == "9":                                   # back matter
            hm = re.search(r"<h[12][^>]*>(.*?)</h[12]>", t, re.S)
            name = re.sub(r"\s+", " ", strip_tags(hm.group(1))).strip() if hm else base
            anchors.append(dict(t="BACK", key=norm(name), part=name, sec="", opener=True))
        # section headings: <h3><span class="num">§N</span> Title</h3>
        for hm in re.finditer(r'<h3[^>]*>\s*<span class="num">\s*(§[^<]+?)\s*</span>\s*(.*?)</h3>', t, re.S):
            num   = strip_tags(hm.group(1)).strip()
            title = re.sub(r"\s+", " ", strip_tags(hm.group(2))).strip()
            anchors.append(dict(t="SEC", key=norm(num)+norm(title), part=None,
                                sec=f"{num} {disp(title)}", opener=False))
    return anchors

def map_pages(reader, anchors, window=8):
    cur_part = cur_sec = None; ptr = 0; out = []
    for i in range(len(reader.pages)):
        try:    txt = norm(reader.pages[i].extract_text() or "")
        except Exception: txt = ""
        suppress = False
        best = -1; j = ptr
        while j < len(anchors) and j < ptr + window:          # furthest match in window
            if anchors[j]["key"] and anchors[j]["key"] in txt: best = j
            j += 1
        if best >= 0:
            for k in range(ptr, best + 1):
                a = anchors[k]
                if a["t"] in ("PART", "BACK"):
                    cur_part, cur_sec = a["part"], None
                    if a["opener"]: suppress = True
                else:
                    cur_sec = a["sec"]
            ptr = best + 1
        out.append((cur_part, cur_sec, suppress))
    return out

def overlay(body_pdf, sections_dir):
    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    anchors = parse_anchors(sections_dir)
    reader  = PdfReader(body_pdf)
    info    = map_pages(reader, anchors)
    W, H = letter; LEFT = 53.9; RIGHT = W - 96.4; Y = H - 40.0
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=letter)
    stamped = 0
    for (part, sec, suppress) in info:
        if part and not suppress:
            c.setFont("Helvetica", 7.5); c.setFillColorRGB(148/255, 163/255, 184/255)
            try: c.setCharSpace(0.3)
            except Exception: pass
            c.drawString(LEFT, Y, part[:82])
            if sec:
                s = sec
                while c.stringWidth(s, "Helvetica", 7.5) > 300 and len(s) > 5:
                    s = s[:-2]
                if s != sec: s = s.rstrip(" .·") + "…"
                c.drawRightString(RIGHT, Y, s)
            stamped += 1
        c.showPage()
    c.save(); buf.seek(0)
    ov = PdfReader(buf); w = PdfWriter()
    for i, pg in enumerate(reader.pages):
        pg.merge_page(ov.pages[i]); w.add_page(pg)
    with open(body_pdf, "wb") as fh: w.write(fh)
    return len(reader.pages), stamped

if __name__ == "__main__":
    if sys.argv[1] == "--anchors":
        for a in parse_anchors(sys.argv[2]):
            print(f"{a['t']:4} key={a['key'][:34]:34} | {a['part'] or ''}{a['sec'] or ''}")
    else:
        pages, stamped = overlay(sys.argv[1], sys.argv[2])
        print(f"runhead: {stamped} of {pages} pages stamped")
