"""
Figures for Part XX — Language Models.

Every claim this part makes that can be measured, is measured here.

The tokenizer figures train a byte-pair encoder from scratch on text generated
here, so they are fully self-contained: no downloaded vocabulary, no external
corpus, and every pathology they show is a consequence of the algorithm rather
than of one vendor's choices.

The rest train small language models on a synthetic language whose grammar is
known, so "did it learn the rule or memorise the examples" is a question with
an answer.

    python3 make_figures.py            # everything
    python3 make_figures.py --cheap    # skip the ones that train a model

Runtime: about a minute for --cheap, twenty for the rest.
"""
import os
import re
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "lm"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#059669"
PURPLE, PINK, GRAY = "#7c3aed", "#db2777", "#64748b"
CYAN, LIGHT, INK = "#0e7490", "#cbd5e1", "#1e293b"
MAX_PT = 900

plt.rcParams.update({
    "svg.fonttype": "path",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": GRAY, "ytick.color": GRAY, "axes.linewidth": 0.9,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
})


def save(fig, name):
    p = os.path.join(OUT, name + ".svg")
    fig.savefig(p, format="svg")
    plt.close(fig)
    h = float(re.search(r'height="([\d.]+)pt"', open(p).read(1200)).group(1))
    if h > MAX_PT:
        raise ValueError(f"{name}.svg is {h:.0f}pt")
    print(f"  wrote {name}.svg  ({h:.0f}pt)")


def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, length=2.5)
    if grid:
        ax.grid(True, color=LIGHT, lw=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------- byte-pair encoding

def _chunk(b, split_digits=True):
    """Split bytes into pre-token chunks. Real tokenizers do this too: merges
    are not allowed to cross a chunk boundary, which is what keeps a word from
    merging with the space that follows it.

    `split_digits` isolates each digit so no multi-digit token can ever form.
    Modern tokenizers do this deliberately, and section 5 measures why."""
    pat = (rb"[A-Za-z]+|[0-9]|\s+|[^A-Za-z0-9\s]+" if split_digits
           else rb"[A-Za-z]+|[0-9]+|\s+|[^A-Za-z0-9\s]+")
    return re.findall(pat, b)


def bpe_train(corpus_bytes, n_merges, split_digits=True):
    """Train BPE. Works on unique chunk TYPES weighted by frequency, which is
    how real trainers do it and is the difference between seconds and hours."""
    freq = Counter()
    for b in corpus_bytes:
        freq.update(_chunk(b, split_digits))
    words = {w: tuple(w) for w in freq}
    merges = []
    nxt = 256
    for _ in range(n_merges):
        pairs = Counter()
        for w, seq in words.items():
            f = freq[w]
            for a_, b_ in zip(seq, seq[1:]):
                pairs[(a_, b_)] += f
        if not pairs:
            break
        best, cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        merges.append((best, nxt))
        new_words = {}
        for w, seq in words.items():
            r, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                    r.append(nxt); i += 2
                else:
                    r.append(seq[i]); i += 1
            new_words[w] = tuple(r)
        words = new_words
        nxt += 1
    return merges


def bpe_encode(b, merges, split_digits=True):
    order = {pair: (i, new) for i, (pair, new) in enumerate(merges)}
    out = []
    for chunk in _chunk(b if isinstance(b, bytes) else b.encode(), split_digits):
        s = list(chunk)
        while len(s) > 1:
            cand = None
            for i in range(len(s) - 1):
                m = order.get((s[i], s[i + 1]))
                if m and (cand is None or m[0] < cand[0]):
                    cand = (m[0], i, m[1])
            if cand is None:
                break
            _, i, new = cand
            s = s[:i] + [new] + s[i + 2:]
        out.extend(s)
    return out


# ------------------------------------------------------------- the corpora
#
# Diversity matters more than size here. Trained on a handful of repeated
# sentences, BPE simply memorises each sentence as one token and every
# measurement below becomes meaningless — which is what the first version of
# this figure did.

def load_words():
    """A real word list, on disk, offline."""
    p = "/usr/share/dict/words"
    ws = [w.strip().lower() for w in open(p, encoding="latin-1")]
    return [w for w in ws if w.isalpha() and 2 <= len(w) <= 14]


def load_code(n_bytes=400000):
    """Python's own standard library as a code corpus."""
    import sysconfig, glob
    out = []
    tot = 0
    for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py"))):
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        out.append(t); tot += len(t)
        if tot > n_bytes:
            break
    return "".join(out)[:n_bytes]


def english_text(words, n_sent=4000, seed=0):
    """Sentences built by sampling from a Zipf-weighted word list, so common
    words recur and the sentences themselves do not."""
    r = np.random.default_rng(seed)
    ranks = np.arange(1, len(words) + 1)
    p = 1.0 / ranks
    p /= p.sum()
    out = []
    for _ in range(n_sent):
        k = int(r.integers(4, 16))
        out.append(" ".join(words[i] for i in r.choice(len(words), k, p=p)) + ". ")
    return "".join(out)


def transliterate(text, table):
    return "".join(table.get(c, c) for c in text)


CYR = dict(zip("abcdefghijklmnoprstuvyz", "абцдефгхийклмнопрстувыз"))
GRK = dict(zip("abcdefghijklmnoprstuvyz", "αβψδεφγηιξκλμνοπρστθωυζ"))


def make_cjk_like(words, n=1500, seed=1):
    """Text in a script with no spaces and many distinct characters — the
    structural properties that make a Latin-trained tokenizer expensive on it,
    without pretending to be a real language."""
    r = np.random.default_rng(seed)
    chars = [chr(0x4E00 + i) for i in range(800)]
    p = 1.0 / np.arange(1, len(chars) + 1)
    p /= p.sum()
    return "".join("".join(chars[i] for i in r.choice(len(chars), int(r.integers(8, 30)), p=p))
                   + "\u3002" for _ in range(n))


def numbers_text(n=3000, seed=2):
    r = np.random.default_rng(seed)
    return " ".join(str(int(v)) for v in r.integers(0, 200000, n)) + " "


# --------------------------------------------- 1. what a tokenizer commits to

def fig_tokenizer():
    """A BPE trained on mostly-English text and then applied to everything
    else — which is the situation every deployed tokenizer is in."""
    words = load_words()
    eng = english_text(words, 4000, seed=0)
    code = load_code(120000)
    nums = numbers_text(1500)
    corpus = [eng.encode(), code.encode(), nums.encode()]
    merges = bpe_train(corpus, 1200)
    print(f"    trained BPE with {len(merges)} merges "
          f"(vocabulary {256 + len(merges)}) on English, code and digits")

    held = english_text(words, 200, seed=99)
    rare = " ".join(words[-6000:][::23][:260]) + " "
    samples = [("English", held), ("English, rare words", rare),
               ("unseen script, 2-byte", transliterate(held, CYR)),
               ("unseen script, 3-byte", make_cjk_like(words, 200)),
               ("code", load_code(240000)[120000:]),
               ("digits", numbers_text(300, seed=7))]
    rows = []
    for name, text in samples:
        b = text.encode()
        toks = bpe_encode(b, merges)
        rows.append((name, len(toks) / len(text), len(b) / len(text)))
        print(f"    {name:20s} {len(toks):5d} tokens for {len(text):5d} chars "
              f"({len(toks)/len(text):.3f} tok/char, {len(b)/len(text):.2f} bytes/char)")

    base = rows[0][1]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    y = np.arange(len(rows))
    ax.barh(y, [r[1] / base for r in rows],
            color=[GREEN] + [BLUE] * (len(rows) - 1), height=0.6)
    ax.axvline(1.0, color=GRAY, lw=1.3, ls="--")
    for i, r in enumerate(rows):
        ax.text(r[1] / base + 0.06, i, f"{r[1]/base:.1f}x", fontsize=7.6,
                color=INK, va="center")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    ax.set_xlabel("tokens per character, relative to English")
    ax.set_title("The same text costs different amounts",
                 fontsize=10.5, color=INK)
    tidy(ax)

    nums_list = [7, 42, 100, 127, 128, 255, 256, 999, 1000, 1024, 1234,
                 3141, 9999, 65536, 123456]
    # the same corpus, trained WITHOUT the digit-splitting rule
    merges_nd = bpe_train(corpus, 1200, split_digits=False)
    with_split = [len(bpe_encode(str(n).encode(), merges)) for n in nums_list]
    without = [len(bpe_encode(str(n).encode(), merges_nd, split_digits=False))
               for n in nums_list]
    ax = axes[1]
    x = np.arange(len(nums_list)); w = 0.38
    ax.bar(x - w/2, without, w, color=ORANGE, label="digits allowed to merge")
    ax.bar(x + w/2, with_split, w, color=GREEN, label="digits split (the fix)")
    ax.plot(x, [len(str(n)) for n in nums_list], "k_", ms=10, mew=1.5,
            label="number of digits")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nums_list], fontsize=6.4, rotation=60)
    ax.set_ylabel("tokens")
    ax.set_title("Why tokenizers now split digits", fontsize=10.5, color=INK)
    ax.legend(fontsize=6.8, frameon=False)
    tidy(ax)

    worst = max(rows[1:], key=lambda r: r[1])
    print(f"    worst case: {worst[0]} at {worst[1]/base:.1f}x English per character")
    print("    number, digits, tokens without the split, tokens with it:")
    for n, a_, b_ in zip(nums_list, without, with_split):
        print(f"      {n:>7}  {len(str(n))} digits   {a_} -> {b_}")
    bad = {}
    for n, t in zip(nums_list, without):
        bad.setdefault(len(str(n)), set()).add(t)
    varied = sorted(d for d, ts in bad.items() if len(ts) > 1)
    print(f"    without the split, numbers of the SAME length take different token")
    print(f"    counts at digit lengths {varied} — the split does not respect place")
    print(f"    value. With the rule, it is always one token per digit.")
    print("    That is the mechanism behind arithmetic failures, and the fix for it.")
    fig.tight_layout()
    save(fig, "01-tokenizer")
    return dict(rows=rows, with_split=with_split, without=without)


# ------------------------------------------------ 2. the vocabulary tradeoff

def fig_vocab():
    """Vocabulary size against sequence length and embedding cost."""
    words = load_words()
    corpus = [english_text(words, 3000, seed=0).encode(),
              load_code(100000).encode(), numbers_text(1200).encode()]
    text = (english_text(words, 200, seed=99) + load_code(20000)).encode()
    sizes = [0, 100, 300, 1000, 3000, 8000]
    lens, vocabs = [], []
    for m in sizes:
        merges = bpe_train(corpus, m) if m else []
        lens.append(len(bpe_encode(text, merges)))
        vocabs.append(256 + len(merges))
        print(f"    {vocabs[-1]:6d} vocabulary -> {lens[-1]:6d} tokens "
              f"({lens[-1]/lens[0]*100:5.1f}% of byte-level)")

    d = 4096
    emb = [v * d * 2 * 2 / 1024 ** 2 for v in vocabs]   # input + output

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    ax.plot(vocabs, lens, "o-", color=BLUE, lw=1.9, ms=5)
    ax.set_xscale("log"); ax.set_xlabel("vocabulary size")
    ax.set_ylabel("tokens for a fixed text")
    ax.set_title("More vocabulary, shorter sequences", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    ax.plot(vocabs, emb, "s-", color=ORANGE, lw=1.9, ms=5)
    ax.set_xscale("log"); ax.set_xlabel("vocabulary size")
    ax.set_ylabel(f"embedding + output layer, MiB at $d={d}$")
    ax.set_title("And a larger embedding to pay for it",
                 fontsize=10.5, color=INK)
    tidy(ax)

    print(f"    byte-level to {vocabs[-1]} tokens shortens the sequence "
          f"{lens[0]/lens[-1]:.2f}x")
    print(f"    and grows the embedding from {emb[0]:.0f} to {emb[-1]:.0f} MiB")
    print("    attention is quadratic in length and the embedding is linear in")
    print("    vocabulary, so the trade has an interior optimum.")
    fig.tight_layout()
    save(fig, "02-vocab")
    return dict(vocabs=vocabs, lens=lens)



CHEAP = [fig_tokenizer, fig_vocab]

if __name__ == "__main__":
    print("Part XX — language model figures")
    for f in CHEAP:
        f()
    if "--cheap" not in sys.argv:
        from train_figures import EXPENSIVE          # noqa: E402
        for f in EXPENSIVE:
            f()
    print("done ->", OUT)
