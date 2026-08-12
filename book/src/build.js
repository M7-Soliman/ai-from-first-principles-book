/**
 * Build "Climbing Deep Learning" — the combined book.
 *
 *   node build.js
 *
 * sections/*.html -> concatenate -> KaTeX (server side) -> Chrome -> PDF
 *
 * Section files are numbered so the order is the book's order. Front matter is
 * 00xx, Part 0 is 01xx, Part I is 03xx, Part II is 05xx, back matter is 09xx.
 * New parts slot in by number — that is the whole extension mechanism.
 */
const fs = require("fs");
const path = require("path");
const katex = require("katex");
const puppeteer = require("puppeteer-core");

const HERE = __dirname;
const SECTIONS = path.join(HERE, "sections");
const OUT_PDF   = path.join(HERE, "..", "artificial-intelligence.pdf");
const COVER_PDF = path.join(HERE, "build-cover.pdf");
const BODY_PDF  = path.join(HERE, "build-body.pdf");
const OUT_HTML = path.join(HERE, "rendered.html");
const LOCK = path.join(HERE, "build.lock");

// Two builds running at once corrupt build-body.pdf: the second Chrome writes it
// while runhead.py is reading it, and pypdf fails with an I/O error a long way
// from the cause. Happened three times before this existed.
(function acquireLock() {
  try {
    fs.writeFileSync(LOCK, String(process.pid), { flag: "wx" });
  } catch (e) {
    if (e.code !== "EEXIST") throw e;
    const pid = parseInt(fs.readFileSync(LOCK, "utf8"), 10);
    let alive = false;
    try { process.kill(pid, 0); alive = true; } catch (_) { alive = false; }
    if (alive) {
      console.error(`  build.js is already running as pid ${pid}. Wait for it, or kill it` +
                    ` and remove ${LOCK}.`);
      process.exit(2);
    }
    fs.writeFileSync(LOCK, String(process.pid));   // stale lock from a killed run
  }
  const release = () => { try { fs.unlinkSync(LOCK); } catch (_) {} };
  process.on("exit", release);
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(sig, () => { release(); process.exit(130); });
  }
})();
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

// ------------------------------------------------------------------ math ---
function renderMath(html) {
  const stash = [];
  const protectedHtml = html.replace(
    /<(pre|code)\b[^>]*>[\s\S]*?<\/\1>/g,
    (m) => { stash.push(m); return ` STASH${stash.length - 1} `; }
  );

  let count = 0;
  const doRender = (tex, display) => {
    count++;
    try {
      return katex.renderToString(tex, {
        displayMode: display, throwOnError: true, strict: false, trust: true,
        macros: {
          "\\R": "\\mathbb{R}", "\\T": "^{\\top}",
          "\\norm": "\\left\\lVert #1 \\right\\rVert",
          "\\abs": "\\left\\lvert #1 \\right\\rvert",
          "\\argmin": "\\operatorname*{arg\\,min}",
          "\\argmax": "\\operatorname*{arg\\,max}",
          "\\tr": "\\operatorname{Tr}", "\\rank": "\\operatorname{rank}",
          "\\diag": "\\operatorname{diag}", "\\E": "\\mathbb{E}",
        },
      });
    } catch (e) {
      console.error("\n  KaTeX FAILED on:", JSON.stringify(tex.slice(0, 90)));
      console.error("   ", e.message.split("\n")[0]);
      process.exitCode = 1;
      return `<span style="color:red;background:#fee">[MATH ERROR]</span>`;
    }
  };

  // Inline math must span source lines: prose is hard-wrapped.
  let out = protectedHtml
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => doRender(t.trim(), true))
    .replace(/(?<!\\)\$([^\$]+?)(?<!\\)\$/g, (_, t) =>
      doRender(t.replace(/\s+/g, " ").trim(), false));

  out = out.replace(/ STASH(\d+) /g, (_, i) => stash[+i]);

  const bare = out.replace(/<span class="katex">[\s\S]*?<\/span><\/span>/g, "");
  const left = bare.match(/\\(?:mathbf|top|frac|lVert|lambda|sigma|begin)/g) || [];
  if (left.length) {
    console.error(`  WARNING: ${left.length} unrendered LaTeX fragments survived`);
    process.exitCode = 1;
  }
  console.log(`  rendered ${count} math expressions`);
  return out;
}

// ------------------------------------------------------------------ main ---
(async () => {
  const files = fs.readdirSync(SECTIONS).filter(f => f.endsWith(".html")).sort();
  console.log(`  ${files.length} section files`);

  const body = files
    .map(f => fs.readFileSync(path.join(SECTIONS, f), "utf8"))
    .join("\n");

  const css = fs.readFileSync(path.join(HERE, "style.css"), "utf8");
  // katex.min.css asks for url(fonts/...) relative to itself; we inline it one
  // level up, so repoint. Without this every large delimiter renders broken.
  const katexCss = fs
    .readFileSync(path.join(HERE, "vendor", "katex.min.css"), "utf8")
    .replace(/url\((['"]?)fonts\//g, "url($1vendor/fonts/");

  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Artificial Intelligence — Soliman M.H.</title>
<style>${katexCss}</style>
<style>${css}</style>
</head><body>
${renderMath(body)}
</body></html>`;

  fs.writeFileSync(OUT_HTML, html);
  console.log(`  wrote rendered.html (${(html.length / 1024).toFixed(0)} KB)`);

  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: "new",
    // printToPDF is one CDP call, so it is governed by protocolTimeout rather
    // than by page.setDefaultTimeout. At ~600 pages the default 30s stopped
    // sufficing and the build died after printing its stats.
    protocolTimeout: 900000,
    args: ["--no-sandbox", "--font-render-hinting=none",
           "--allow-file-access-from-files"],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(600000);          // 8MB of rendered HTML outgrew the 30s default
  page.setDefaultNavigationTimeout(600000);
  await page.goto("file://" + OUT_HTML, { waitUntil: "networkidle0", timeout: 900000 });
  await page.evaluateHandle("document.fonts.ready");

  const stats = await page.evaluate(() => ({
    figures: document.querySelectorAll("figure img").length,
    broken: [...document.querySelectorAll("img")]
      .filter(i => !i.complete || i.naturalWidth === 0).length,
    katexFonts: [...document.fonts]
      .filter(f => f.family.startsWith("KaTeX") && f.status === "loaded").length,
    history: document.querySelectorAll(".box.history").length,
    sixty: document.querySelectorAll(".sixty").length,
    mnotes: document.querySelectorAll(".mnote").length,
    gloss: document.querySelectorAll(".gloss:not(.biblio) dt").length,
    papers: document.querySelectorAll(".biblio dt").length,
    readnow: document.querySelectorAll(".readnow").length,
    illus: document.querySelectorAll("figure.illus").length,
    illusUntagged: [...document.querySelectorAll("figure.illus figcaption")]
      .filter(c => !c.querySelector(".illus-tag")).length,
    // Guard: repeated insert-then-resort passes on the glossary silently produced
    // duplicate terms twice. Count them so the build reports it.
    glossDupes: (() => {
      const t = [...document.querySelectorAll(".gloss:not(.biblio) dt")]
        .map(d => d.textContent.replace(/\s+/g, " ")
              // Any part label, not just 0-V. The original character class stopped
              // at V, so from Part VI onward the label stayed in the compared
              // string and no duplicate could ever match. Five went undetected.
              .replace(/\s+(0|[IVX]+)\s+§.*$/, "").trim().toLowerCase());
      return t.length - new Set(t).size;
    })(),
    fw: document.querySelectorAll(".box.fw").length,
    // Guard: every .box sets break-inside:avoid. Nesting one inside another makes
    // the inner box unplaceable, so Chrome pushes it to the next page and leaves the
    // outer box's background filling the rest of the current one. Caught once, in a
    // framework box in Part I section 13; never nest.
    nestedBoxes: document.querySelectorAll(".box .box").length,
    // Guard: the book states how many parts it has in three places — title page,
    // map, colophon — and all three are prose. Two of them were two parts behind
    // when this was written. The first version of this guard matched any "N parts"
    // and drowned in false positives ("the loss splits into three parts"), so the
    // claim is marked instead: wrap it in <span class="partcount">N</span>.
    partCountMismatch: (() => {
      // ...and only those whose kicker actually says "Part": the glossary and the
      // reading list reuse the opener layout and have a .pnum of their own.
      const actual = [...document.querySelectorAll(".partopen .pnum")]
        .filter(e => /^Part\b/i.test(e.textContent.trim())).length;
      const W = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
        "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen",
        "nineteen","twenty"];
      const tens = { twenty: 20, thirty: 30, forty: 40, fifty: 50 };
      const word2num = w => {
        w = w.toLowerCase().trim();
        if (/^\d+$/.test(w)) return +w;
        if (W.indexOf(w) >= 0) return W.indexOf(w);
        const [a, b] = w.split("-");
        return (a in tens) ? tens[a] + (b ? W.indexOf(b) : 0) : null;
      };
      const marked = [...document.querySelectorAll(".partcount")];
      const bad = marked
        .map(el => [el.textContent, word2num(el.textContent)])
        .filter(([, n]) => n !== actual)
        .map(([t, n]) => `"${t}" (${n}) but ${actual} parts exist`);
      if (marked.length < 3) bad.push(`only ${marked.length} part-count claims are marked;` +
        ` the title page, the map and the colophon each make one`);
      return bad;
    })(),
    // Guard: every drills page opens with "N problems." in prose. The opener's
    // metadata was checked and this sentence never was, so five parts carried a
    // stale count for as long as they had existed — Part I said thirty when it
    // had thirty-five. Numbers in prose rot exactly like numbers in code.
    drillCountMismatch: (() => {
      const W = ["zero","one","two","three","four","five","six","seven","eight","nine","ten",
        "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen",
        "nineteen","twenty"];
      const word2num = s => {
        s = s.toLowerCase().replace(/\u2013|\u2014/g, "-");
        if (W.indexOf(s) >= 0) return W.indexOf(s);
        const tens = { twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60 };
        const [a, b] = s.split("-");
        if (!(a in tens)) return null;
        return tens[a] + (b ? W.indexOf(b) : 0);
      };
      const bad = [];
      let part = "?";
      for (const h of document.querySelectorAll("h2, .partopen .pnum")) {
        if (h.classList && h.classList.contains("pnum")) { part = h.textContent.trim(); continue; }
        if (!/Drills/.test(h.textContent)) continue;
        let n = 0, lead = null, el = h.nextElementSibling;
        while (el && el.tagName !== "H2") {
          if (el.classList.contains("drill")) n++;
          if (lead === null && /^(P)$/.test(el.tagName)) {
            const m = el.textContent.trim().match(/^([A-Za-z]+(?:-[a-z]+)?)\s+(problems|drills)\b/);
            if (m) lead = word2num(m[1]);
          }
          el = el.nextElementSibling;
        }
        if (lead !== null && lead !== n) bad.push(`${part}: says ${lead}, has ${n}`);
      }
      return bad;
    })(),
  }));
  console.log(`  figures ${stats.figures} | broken ${stats.broken} | ` +
              `KaTeX faces ${stats.katexFonts}`);
  console.log(`  history boxes ${stats.history} | 60-sec ${stats.sixty} | ` +
              `margin notes ${stats.mnotes} | glossary ${stats.gloss} | papers ${stats.papers} | ` +
              `inline sources ${stats.readnow} | ` +
              `illustrative ${stats.illus} of ${stats.figures} | ` +
              `framework ${stats.fw}`);
  if (stats.glossDupes) {
    console.error(`  WARNING: ${stats.glossDupes} duplicate glossary terms`);
    process.exitCode = 1;
  }
  if (stats.illusUntagged) {
    console.error(`  WARNING: ${stats.illusUntagged} illustrative figures whose caption` +
                  ` lacks an ILLUSTRATIVE tag. A reader must never have to guess whether` +
                  ` a figure was measured.`);
    process.exitCode = 1;
  }

  if (stats.partCountMismatch.length) {
    console.error(`  WARNING: ${stats.partCountMismatch.length} stale part counts in prose:`);
    for (const b of new Set(stats.partCountMismatch)) console.error(`    ${b}`);
    process.exitCode = 1;
  }
  if (stats.drillCountMismatch.length) {
    console.error(`  WARNING: ${stats.drillCountMismatch.length} drills pages whose stated` +
                  ` count disagrees with the drills present:`);
    for (const b of stats.drillCountMismatch) console.error(`    ${b}`);
    process.exitCode = 1;
  }
  if (stats.nestedBoxes) {
    console.error(`  WARNING: ${stats.nestedBoxes} nested .box elements — these strand a` +
                  ` near-empty page. Flatten the inner box to an <h4> plus paragraphs.`);
    process.exitCode = 1;
  }
  if (stats.broken || stats.katexFonts === 0) process.exitCode = 1;

  // Chrome CLIPS anything outside the print margin box, so a full-bleed cover is
  // impossible in a single pass. Print it separately at zero margins, print the body
  // with margins, then splice the two together.
  await page.addStyleTag({ content: `
    body > *:not(.cover) { display: none !important; }
    .cover { height: 11in !important; margin: 0 !important;
             padding: 44mm 26mm 30mm 26mm !important; }` });
  await page.pdf({ path: COVER_PDF, format: "Letter", printBackground: true,
                   margin: { top: 0, bottom: 0, left: 0, right: 0 },
                   pageRanges: "1", timeout: 600000 });

  await page.reload({ waitUntil: "networkidle0", timeout: 900000 });
  await page.evaluateHandle("document.fonts.ready");
  await page.addStyleTag({ content: `.cover { display: none !important; }` });
  await page.pdf({
    path: BODY_PDF, format: "Letter", printBackground: true,
    timeout: 600000,                       // the default 30s stopped sufficing at ~560 pages
    displayHeaderFooter: true,
    margin: { top: "20mm", bottom: "20mm", left: "19mm", right: "34mm" },
    // The top running head (part · section) is stamped per-page afterwards by runhead.py.
    // Chrome's headerTemplate is static across pages, so it cannot name the current section.
    headerTemplate: `<div></div>`,
    footerTemplate: `<div style="font-size:8.5pt;color:#64748b;width:100%;
        text-align:center;font-family:Helvetica,Arial,sans-serif;">
        <span class="pageNumber"></span></div>`,
  });

  await browser.close();

  const { execFileSync } = require("child_process");
  // Stamp the part · section running head onto each body page (in place).
  execFileSync("python3", [path.join(HERE, "runhead.py"), BODY_PDF, SECTIONS], { stdio: "inherit" });
  execFileSync("python3", ["-c", `
import sys
from pypdf import PdfWriter
w = PdfWriter()
w.append(sys.argv[1], pages=(0, 1))   # full-bleed cover only
w.append(sys.argv[2])                 # the body, with its running heads
w.write(sys.argv[3])
`, COVER_PDF, BODY_PDF, OUT_PDF]);
  // pypdf's rewrite drops Chrome's stream compression, so the runhead overlay bloats
  // the file ~7x. Recompress in place with pikepdf (libqpdf): ~90MB -> ~12MB.
  execFileSync("python3", ["-c", `
import sys, pikepdf
pdf = pikepdf.open(sys.argv[1], allow_overwriting_input=True)
pdf.save(sys.argv[1], compress_streams=True,
         object_stream_mode=pikepdf.ObjectStreamMode.generate)
`, OUT_PDF]);
  // Cleanup must not fail a build whose PDF is already written and spliced.
  for (const f of [COVER_PDF, BODY_PDF]) {
    try { fs.unlinkSync(f); } catch (e) { if (e.code !== "ENOENT") throw e; }
  }

  console.log(`\n  PDF -> ${OUT_PDF} ` +
              `(${(fs.statSync(OUT_PDF).size / 1024 / 1024).toFixed(1)} MB)`);
})();
