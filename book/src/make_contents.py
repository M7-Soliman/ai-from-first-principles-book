#!/usr/bin/env python3
"""Generate the full table of contents from the section files.

    python3 make_contents.py            # writes sections/0003-contents.html

Every part, every movement, every numbered section — read out of the sources, so
the contents cannot drift from the book the way a hand-maintained list does. Run
it before build.js whenever sections are added, renamed or renumbered.

No page numbers: they would require a two-pass build, and adding the contents
changes the pagination it would be reporting.
"""
import os
import re
import glob
import html

HERE = os.path.dirname(os.path.abspath(__file__))
SECTIONS = os.path.join(HERE, "sections")
OUT = os.path.join(SECTIONS, "0003-contents.html")
OUT_TXT = os.path.abspath(os.path.join(HERE, "..", "..", "CONTENTS.txt"))

WORD2ROMAN = {
 "Zero": "0", "One": "I", "Two": "II", "Three": "III", "Four": "IV", "Five": "V",
 "Six": "VI", "Seven": "VII", "Eight": "VIII", "Nine": "IX", "Ten": "X",
 "Eleven": "XI", "Twelve": "XII", "Thirteen": "XIII", "Fourteen": "XIV",
 "Fifteen": "XV", "Sixteen": "XVI", "Seventeen": "XVII", "Eighteen": "XVIII",
 "Nineteen": "XIX", "Twenty": "XX", "Twenty-One": "XXI", "Twenty-Two": "XXII",
 "Twenty-Three": "XXIII", "Twenty-Four": "XXIV", "Twenty-Five": "XXV",
 "Twenty-Six": "XXVI", "Twenty-Seven": "XXVII", "Twenty-Eight": "XXVIII",
 "Twenty-Nine": "XXIX", "Thirty": "XXX", "Thirty-One": "XXXI", "Thirty-Two": "XXXII",
    "Thirty-Three": "XXXIII", "Thirty-Four": "XXXIV", "Thirty-Five": "XXXV",
}


def strip(s):
    """Tags out, entities decoded, KaTeX reduced to something readable."""
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    # $\ell_1$ -> ell_1 is worse than nothing; keep the letters, drop the commands
    s = re.sub(r"\$(.*?)\$", lambda m: re.sub(r"\\[a-zA-Z]+|[{}\\]", "", m.group(1)), s)
    return re.sub(r"\s+", " ", s).strip()


def esc(s):
    return html.escape(s, quote=False)


def collect():
    """Walk the sources in document order, returning a flat list of entries."""
    out = []
    seen_front = seen_back = False
    for f in sorted(glob.glob(os.path.join(SECTIONS, "[0-9]*.html"))):
        base = os.path.basename(f)
        if base.startswith(("0000", "0002", "0003")):        # cover, title, this file
            continue
        t = open(f, encoding="utf-8").read()
        is_back = base[0] == "9"
        is_front = not is_back and not re.search(r'class="pnum">', t) and not out
        if is_front and not seen_front:
            out.append(("group", "Front matter", "", "")); seen_front = True
        if is_back and not seen_back:
            out.append(("group", "Back matter", "", "")); seen_back = True

        m = re.search(r'class="pnum">\s*Part\s+([A-Za-z][A-Za-z\- ]*?)\s*<', t)
        if m:
            word = m.group(1).strip()
            # Fail loudly. Falling back to the spelled-out word is what let
            # "Part Twenty-Two" ship into the contents, the running heads and
            # the glossary legend without a single warning.
            if word not in WORD2ROMAN:
                raise SystemExit(f"make_contents: no roman numeral for {word!r} — "
                                 f"add it to WORD2ROMAN")
            roman = WORD2ROMAN[word]
            h1 = re.search(r"<h1[^>]*>(.*?)</h1>", t, re.S)
            title = sub = ""
            if h1:
                inner = h1.group(1)
                # The lighter span inside the h1 does two different jobs and nothing
                # in the markup distinguishes them: sometimes it finishes the title
                # ("Probability" + "& Information", "Machine Learning" + "Basics"),
                # sometimes it is the part's tagline ("buying generalisation with
                # bias"). Taglines are phrases; continuations are one or two words.
                sm = re.search(r'<span[^>]*font-weight:\s*300[^>]*>(.*?)</span>', inner, re.S)
                if sm:
                    light = strip(sm.group(1))
                    inner = inner.replace(sm.group(0), "")
                    if "," in light or len(light.split()) >= 3:
                        sub = light                       # a tagline
                    else:
                        inner += " " + light              # part of the title
                title = strip(re.sub(r"<br\s*/?>", " ", inner))
            cur = roman
            out.append(("part", f"Part {roman}", title, sub))
            continue

        # movements and the standing end-matter blocks, in order with the sections
        for hm in re.finditer(
                r'<h([23])[^>]*>(?:\s*<span class="(?:part-kicker|num)">\s*(.*?)\s*</span>)?\s*(.*?)</h\1>',
                t, re.S):
            level, kicker, title = hm.group(1), hm.group(2) or "", strip(hm.group(3))
            kicker = strip(kicker)
            if level == "2":
                if not title:
                    continue
                out.append(("movement", kicker, title, ""))
            else:
                if not kicker.startswith("§"):
                    continue
                num = kicker[1:].strip()
                # numbered: "§7" in the body, "§0.4" in a part's own front matter.
                # unnumbered: the bare "§" the preface and back matter use.
                if num.isdigit():                      # §7 — a body section
                    out.append(("section", kicker, title, ""))
                elif re.fullmatch(r"\d+\.\d+", num):     # §0.4 — a part's own front matter
                    out.append(("subsection", kicker, title, ""))
                else:                                     # bare § — preface, back matter
                    out.append(("unnumbered", "", title, ""))
    return out


def render(entries):
    rows = []
    for kind, a, b, c in entries:
        if kind in ("part", "group"):
            label = a if not b else f"{a}: {b}"
            rows.append(f'    <div class="part">{esc(label)}</div>')
            if c:
                rows.append(f'    <div class="tocsub">{esc(c)}</div>')
        elif kind == "movement":
            k = a.replace("Part ", "") if a.startswith("Part ") else a
            lead = f"{esc(k)} · " if k else ""
            rows.append(f'    <div class="tocmv">{lead}{esc(b)}</div>')
        elif kind == "unnumbered":
            rows.append(f'    <div class="row plain"><span class="t">{esc(b)}</span></div>')
        elif kind == "subsection":
            rows.append(
                f'    <div class="row"><span class="n">{esc(a)}</span>'
                f'<span class="t">{esc(b)}</span></div>')
        else:
            rows.append(
                f'    <div class="row"><span class="n">{esc(a)}</span>'
                f'<span class="t">{esc(b)}</span></div>')
    n_sec = sum(1 for e in entries if e[0] == "section")
    n_part = sum(1 for e in entries if e[0] == "part")
    n_mv = sum(1 for e in entries if e[0] == "movement")
    body = "\n".join(rows)
    return f"""<div class="bookpage">
  <h2 class="nobreak" style="border:none;margin-bottom:0.15em">Contents</h2>
  <p class="tocnote">{n_part} parts, {n_sec} numbered sections, all complete. Each part is
  built in movements, and carries computed figures, tiered drills with an automated test
  harness, a substantial project and a checkpoint, listed here in the position they
  occupy.</p>

  <div class="toc twocol full">
{body}
  </div>
</div>
"""


def render_text(entries):
    """The same contents as plain text, from the same source, so the two cannot
    disagree. Section numbers are right-aligned in a fixed field so the titles
    line up under each movement."""
    n_sec = sum(1 for e in entries if e[0] == "section")
    n_part = sum(1 for e in entries if e[0] == "part")
    W = 78
    L = ["=" * W,
         "ARTIFICIAL INTELLIGENCE: From first principles",
         "The mathematics, the learning, and the machines that think",
         "Soliman M.H.",
         "=" * W, "",
         f"{n_part} parts, {n_sec} numbered sections, all complete.",
         "Each part is built in movements, and carries computed figures, tiered drills",
         "with an automated test harness, a substantial project and a checkpoint,",
         "listed here in the position they occupy.", ""]
    for kind, a, b, c in entries:
        if kind in ("part", "group"):
            label = a if not b else f"{a}: {b}"
            L += ["", "-" * W, label.upper()]
            if c:
                L.append(f"  {c}")
            L += ["-" * W]
        elif kind == "movement":
            k = a.replace("Part ", "") if a.startswith("Part ") else a
            L += ["", f"  {k + ' · ' if k else ''}{b}"]
        elif kind == "unnumbered":
            L.append(f"          {b}")
        else:
            L.append(f"    {a:>6}  {b}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys
    e = collect()
    open(OUT, "w", encoding="utf-8").write(render(e))
    print(f"wrote {os.path.relpath(OUT, HERE)}")
    if "--text" in sys.argv or True:
        open(OUT_TXT, "w", encoding="utf-8").write(render_text(e))
        print(f"wrote {OUT_TXT}")
    print(f"  parts {sum(1 for x in e if x[0]=='part')} | "
          f"movements {sum(1 for x in e if x[0]=='movement')} | "
          f"numbered sections {sum(1 for x in e if x[0]=='section')} | "
          f"part front-matter {sum(1 for x in e if x[0]=='subsection')} | "
          f"unnumbered {sum(1 for x in e if x[0]=='unnumbered')}")
