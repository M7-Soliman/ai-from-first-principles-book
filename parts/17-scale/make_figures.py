"""
Figures for Part XVII — Scale.

Every claim this part makes that can be measured, is measured here.

The cheap ones are exact arithmetic: where memory actually goes and what
sharding recovers, what the floating-point formats can and cannot represent,
and the pipeline bubble. None of them needs a GPU, because none of them is an
empirical question — they follow from the shapes and the number formats.

Two are illustrative — the scaling law and the schedule error. Both describe
behaviour established over runs spanning many orders of magnitude of compute,
far beyond what this book's figures are built with, so they
draw the shape and carry no invented numbers. The only figures in them are the
two published allocation exponents, attributed by name.

The one expensive figure trains: mixture-of-experts load balancing.
train_figures.py holds the scaling-law and schedule studies as the project's
Stages 4 and 5.

    python3 make_figures.py            # everything
    python3 make_figures.py --cheap    # skip the ones that train

Runtime: seconds for --cheap, about half an hour for the rest.
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "sc"))
os.makedirs(OUT, exist_ok=True)

# Project Stage 4/5 study outputs (fig_scaling, fig_schedule in train_figures.py)
# are NOT book figures — nothing in book/src/sections references them — so they
# do not belong in the numbered book-figure namespace above. They get their own
# directory to avoid colliding in spirit with 05-dataparallel.svg / 06-pipeline.svg.
STUDY_OUT = os.path.abspath(os.path.join(HERE, "study_figures"))
os.makedirs(STUDY_OUT, exist_ok=True)

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


def save(fig, name, out_dir=None):
    p = os.path.join(out_dir or OUT, name + ".svg")
    fig.savefig(p, format="svg")
    plt.close(fig)
    h = float(re.search(r'height="([\d.]+)pt"', open(p).read(1200)).group(1))
    if h > MAX_PT:
        raise ValueError(f"{name}.svg is {h:.0f}pt")
    print(f"  wrote {p}  ({h:.0f}pt)")


def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, length=2.5)
    if grid:
        ax.grid(True, color=LIGHT, lw=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------- 1. where memory goes

def transformer_params(d, L, V, ctx):
    """Parameter count of a standard decoder stack, by part. Exact."""
    attn = 4 * d * d * L                       # q, k, v, o
    ffn = 8 * d * d * L                        # up 4d, down 4d
    norms = 2 * d * L * 2
    emb = V * d
    pos = ctx * d
    return dict(attention=attn, ffn=ffn, norms=norms,
                embedding=emb, positional=pos)


def memory_breakdown(N, batch, ctx, d, L, heads, bytes_param=2,
                     optimizer="adam", recompute=False):
    """Bytes, by category, for one training step. Exact arithmetic — no
    profiling, because none of this is an empirical question."""
    params = N * bytes_param
    grads = N * bytes_param
    # Adam in mixed precision: fp32 master copy + two fp32 moments
    opt = N * (4 + 4 + 4) if optimizer == "adam" else N * 4
    # activations stored for the backward pass, per layer
    per_layer = batch * ctx * d * bytes_param * 10          # the usual tensors
    attn_matrix = batch * heads * ctx * ctx * bytes_param   # if materialised
    acts = (per_layer + attn_matrix) * L
    if recompute:
        acts = (per_layer + attn_matrix) * 1 + batch * ctx * d * bytes_param * L
    return dict(parameters=params, gradients=grads,
                optimizer_states=opt, activations=acts)


def fig_memory():
    """What actually occupies memory during training, and what sharding
    recovers. All exact."""
    d, L, V, ctx, heads = 4096, 32, 50000, 4096, 32
    parts = transformer_params(d, L, V, ctx)
    N = sum(parts.values())
    print(f"    a {d}-wide, {L}-layer model: {N/1e9:.2f}B parameters")
    for k, v in parts.items():
        print(f"      {k:12s} {v/1e9:6.3f}B  ({v/N*100:5.1f}%)")
    print(f"      attention is {parts['attention']/(parts['attention']+parts['ffn'])*100:.0f}% "
          f"of the transformer blocks; the feed-forward is the rest")

    GB = 1024 ** 3
    base = memory_breakdown(N, 1, ctx, d, L, heads)
    rec = memory_breakdown(N, 1, ctx, d, L, heads, recompute=True)

    # ZeRO stages: shard optimizer states, then gradients, then parameters
    stages = {}
    for w in (1, 8, 64):
        for stage, keys in ((0, []), (1, ["optimizer_states"]),
                            (2, ["optimizer_states", "gradients"]),
                            (3, ["optimizer_states", "gradients", "parameters"])):
            m = dict(base)
            for k in keys:
                m[k] = m[k] / w
            stages[(w, stage)] = sum(m.values())

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))

    ax = axes[0]
    keys = ["parameters", "gradients", "optimizer_states", "activations"]
    vals = [base[k] / GB for k in keys]
    ax.bar(range(4), vals, color=[BLUE, GREEN, PURPLE, ORANGE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.03, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["params", "grads", "optimiser", "activations"], fontsize=8)
    ax.set_ylabel("GiB, one step, batch 1")
    ax.set_title("Where it goes", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    ws = [1, 8, 64]
    for si, (lab, col) in enumerate((("no sharding", GRAY), ("stage 1", BLUE),
                                     ("stage 2", GREEN), ("stage 3", PURPLE))):
        ax.plot(ws, [stages[(w, si)] / GB for w in ws], "o-", color=col,
                lw=1.8, ms=4, label=lab)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("devices sharded across")
    ax.set_ylabel("GiB per device")
    ax.set_title("What sharding recovers", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False)
    tidy(ax)

    ax = axes[2]
    ctxs = [1024, 2048, 4096, 8192, 16384]
    a_full, a_rec = [], []
    for c in ctxs:
        a_full.append(memory_breakdown(N, 1, c, d, L, heads)["activations"] / GB)
        a_rec.append(memory_breakdown(N, 1, c, d, L, heads,
                                      recompute=True)["activations"] / GB)
    ax.plot(ctxs, a_full, "o-", color=ORANGE, lw=1.9, ms=4, label="stored")
    ax.plot(ctxs, a_rec, "s-", color=GREEN, lw=1.9, ms=4, label="recomputed")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length"); ax.set_ylabel("activation GiB")
    ax.set_title("Activations, and recomputation", fontsize=10.5, color=INK)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)

    print(f"    at batch 1, context {ctx}: parameters {base['parameters']/GB:.1f} GiB, "
          f"optimiser states {base['optimizer_states']/GB:.1f} GiB "
          f"({base['optimizer_states']/base['parameters']:.0f}x the parameters)")
    print(f"    activations {base['activations']/GB:.1f} GiB, "
          f"falling to {rec['activations']/GB:.1f} GiB with recomputation "
          f"({base['activations']/rec['activations']:.0f}x)")
    print(f"    stage 3 across 64 devices: {stages[(64,3)]/GB:.1f} GiB against "
          f"{stages[(1,0)]/GB:.1f} — a factor of {stages[(1,0)]/stages[(64,3)]:.1f}")
    print(f"    note what does NOT shrink: activations, which is why the third panel matters")
    fig.tight_layout()
    save(fig, "03-memory")
    return dict(N=N, parts=parts, base=base, stages=stages)


# ------------------------------------------------- 2. the number formats

def fmt_info(exp_bits, man_bits):
    """Exact range and precision of a binary floating-point format."""
    bias = 2 ** (exp_bits - 1) - 1
    max_exp = 2 ** exp_bits - 2 - bias
    min_exp = 1 - bias
    largest = (2 - 2.0 ** -man_bits) * 2.0 ** max_exp
    smallest_normal = 2.0 ** min_exp
    smallest_sub = 2.0 ** (min_exp - man_bits)
    eps = 2.0 ** -man_bits
    return dict(largest=largest, smallest_normal=smallest_normal,
                smallest_sub=smallest_sub, eps=eps, decimal_digits=man_bits * np.log10(2))


def fig_precision():
    """What each format can represent. Exact, from the bit layouts — the
    range difference is why one of them needs loss scaling and the other
    does not."""
    F = {"fp32 (8,23)": fmt_info(8, 23),
         "tf32 (8,10)": fmt_info(8, 10),
         "bf16 (8,7)": fmt_info(8, 7),
         "fp16 (5,10)": fmt_info(5, 10),
         "fp8 e4m3 (4,3)": fmt_info(4, 3),
         "fp8 e5m2 (5,2)": fmt_info(5, 2)}
    for k, v in F.items():
        print(f"    {k:16s} max {v['largest']:9.3g}  min normal {v['smallest_normal']:9.3g}  "
              f"eps {v['eps']:8.2e}  ~{v['decimal_digits']:.1f} decimal digits")

    # a synthetic, log-normal gradient distribution spanning many decades —
    # a mechanism illustration, not a measurement of any real training run —
    # and what each format loses on it
    rng = np.random.default_rng(0)
    g = np.exp(rng.normal(np.log(1e-4), 2.2, 400000))       # spans many decades
    rows = []
    for k, v in F.items():
        under = float((g < v["smallest_sub"]).mean())
        over = float((g > v["largest"]).mean())
        rows.append((k, under, over))
        print(f"    {k:16s} gradients flushed to zero {under*100:6.2f}%   "
              f"overflowed {over*100:5.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    names = list(F)
    y = np.arange(len(names))
    for i, k in enumerate(names):
        v = F[k]
        ax.plot([v["smallest_sub"], v["largest"]], [i, i], lw=6,
                color=BLUE if "8" == k.split("(")[1][0] else ORANGE, solid_capstyle="butt")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("representable magnitude")
    ax.set_title("Dynamic range", fontsize=10.5, color=INK)
    tidy(ax)

    ax = axes[1]
    ax.barh(y, [r[1] * 100 for r in rows], color=PURPLE, height=0.55)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("% of gradients flushed to zero")
    ax.set_title("What a realistic gradient loses", fontsize=10.5, color=INK)
    tidy(ax)

    f16, bf16 = F["fp16 (5,10)"], F["bf16 (8,7)"]
    print(f"    fp16 and bf16 are both 16 bits. fp16 has {f16['decimal_digits']:.1f} decimal "
          f"digits and a smallest normal of {f16['smallest_normal']:.2e};")
    print(f"    bf16 has {bf16['decimal_digits']:.1f} and {bf16['smallest_normal']:.2e} — "
          f"{f16['smallest_normal']/bf16['smallest_normal']:.3g}x more range, less precision.")
    print("    that trade is the whole reason bf16 needs no loss scaling and fp16 does.")
    fig.tight_layout()
    save(fig, "04-precision")
    return dict(formats=F, rows=rows)


# ------------------------------------------------------- 3. the pipeline bubble

def fig_pipeline():
    """The pipeline bubble is exact arithmetic, not a measurement: with P
    stages and M microbatches, a fraction (P-1)/(M+P-1) of the time is idle."""
    Ps = [2, 4, 8, 16, 32]
    Ms = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for P, c in zip(Ps, (GRAY, BLUE, GREEN, ORANGE, PURPLE)):
        bub = (P - 1) / (Ms + P - 1)
        ax.plot(Ms, bub * 100, "o-", color=c, lw=1.7, ms=3.5, label=f"$P={P}$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("microbatches $M$"); ax.set_ylabel("% of time idle")
    ax.set_title("The bubble", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False, ncol=2)
    tidy(ax)

    ax = axes[1]
    # smallest M with (P-1)/(M+P-1) <= 10% — this lands ON 10% exactly, not
    # strictly under it (e.g. P=8 gives exactly 10.0% at M=63); one more
    # microbatch is needed to go strictly under.
    need = [int(np.ceil((P - 1) * (1 / 0.10 - 1))) for P in Ps]
    ax.bar(range(len(Ps)), need, color=BLUE, width=0.55)
    for i, v in enumerate(need):
        ax.text(i, v * 1.03, str(v), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(Ps)))
    ax.set_xticklabels([f"$P={p}$" for p in Ps], fontsize=8.5)
    ax.set_ylabel("microbatches needed")
    ax.set_title("Down to a 10% bubble", fontsize=10.5, color=INK)
    tidy(ax)

    for P, n in zip(Ps, need):
        print(f"    P={P:2d}: bubble is {(P-1)/(1+P-1)*100:5.1f}% at M=1, "
              f"{(P-1)/(8+P-1)*100:5.1f}% at M=8; needs M={n} to reach 10% "
              f"(exactly, {(P-1)/(n+P-1)*100:.1f}%)")
    print("    and every microbatch in flight holds its own activations, which is")
    print("    the memory cost that limits how far this can be pushed.")
    fig.tight_layout()
    save(fig, "06-pipeline")
    return dict(Ps=Ps, need=need)


# --------------------------------------------- 4. data-parallel communication

def fig_dataparallel():
    """Why data parallelism stops scaling: the gradient all-reduce moves a
    fixed number of bytes per step regardless of batch size, so it is amortised
    only if each device has enough work."""
    N = 7e9
    bytes_grad = 2
    vol = 2 * N * bytes_grad * (1 - 1 / 8)      # ring all-reduce, 8 devices

    # per-device compute time against interconnect bandwidth
    flops_per_token = 6 * N
    dev_flops = 3e14                            # a plausible sustained rate
    bws = np.array([25, 50, 100, 200, 400, 900]) * 1e9 / 8   # GB/s from Gb/s

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    ax = axes[0]
    for tok, c in zip((2048, 8192, 32768), (ORANGE, BLUE, GREEN)):
        t_comp = tok * flops_per_token / dev_flops
        eff = t_comp / (t_comp + vol / bws) * 100
        ax.plot(bws * 8 / 1e9, eff, "o-", color=c, lw=1.8, ms=4,
                label=f"{tok} tokens/device/step")
    ax.set_xscale("log")
    ax.set_xlabel("interconnect (Gb/s)")
    ax.set_ylabel("% of time computing")
    ax.set_ylim(0, 102)
    ax.set_title("Data parallelism, amortised", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right")
    tidy(ax)

    ax = axes[1]
    toks = np.array([256, 1024, 4096, 16384, 65536])
    for bw_gb, c in zip((25, 200, 900), (ORANGE, BLUE, GREEN)):
        bw = bw_gb * 1e9 / 8
        t_comp = toks * flops_per_token / dev_flops
        ax.plot(toks, t_comp / (t_comp + vol / bw) * 100, "o-", color=c,
                lw=1.8, ms=4, label=f"{bw_gb} Gb/s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("tokens per device per step")
    ax.set_ylabel("% of time computing")
    ax.set_ylim(0, 102)
    ax.set_title("Why small per-device batches stall", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    tidy(ax)

    print(f"    a {N/1e9:.0f}B model in bf16: one all-reduce moves "
          f"{vol/1e9:.1f} GB per step, whatever the batch size")
    for bw_gb in (25, 200, 900):
        bw = bw_gb * 1e9 / 8
        for tok in (1024, 16384):
            t = tok * flops_per_token / dev_flops
            print(f"      {bw_gb:3d} Gb/s, {tok:5d} tokens/device: "
                  f"{t/(t + vol/bw)*100:5.1f}% of time computing")
    print("    the communication is FIXED per step and the compute is not, so the")
    print("    only lever is giving each device more work — which is why weak")
    print("    scaling works and strong scaling does not.")
    fig.tight_layout()
    save(fig, "05-dataparallel")
    return dict(vol=vol)


# ============================================================================
# Illustrative figures.
#
# These two draw the SHAPE of a result rather than a measurement of one. Both
# describe behaviour established over runs spanning many orders of magnitude of
# compute — far beyond what this book's figures are built with — so they are drawn, tagged, and
# carry no invented numbers. The only
# numbers that appear are the published exponents, attributed by name.
# ============================================================================

def fig_scaling_illus():
    """The shape of a scaling law: a power law above an irreducible floor, and
    what subtracting the floor reveals."""
    N = np.logspace(0, 6, 400)
    A, alpha, E = 3.0, 0.32, 1.0
    L = A * N ** (-alpha) + E

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    ax.plot(N, L, color=BLUE, lw=2.2, label="loss")
    ax.plot(N, A * N ** (-alpha),
            color=ORANGE, lw=1.4, ls="--", label="reducible part alone")
    ax.axhline(E, color=GRAY, lw=1.5, ls=":", label="irreducible floor $E$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("parameters $N$"); ax.set_ylabel("loss")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title("Loss flattens onto the floor", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.8, frameon=False)
    ax.annotate("the floor is the entropy\nof the data",
                xy=(N[-1] * 0.3, E), xytext=(60, E * 1.9),
                fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9))
    tidy(ax)

    ax = axes[1]
    ax.plot(N, L - E, color=PURPLE, lw=2.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("parameters $N$"); ax.set_ylabel("loss above the floor")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title("Subtract it and a straight line appears",
                 fontsize=10.5, color=INK)
    ax.annotate("slope $-\\alpha$", xy=(1e3, (L - E)[np.argmin(abs(N - 1e3))]),
                xytext=(30, (L - E)[0] * 0.35), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9))
    tidy(ax)

    print("    illustrative: the shape only. No axis values, no fitted numbers.")
    fig.tight_layout()
    save(fig, "01-scaling-illus")


def fig_schedule_illus():
    """The shape of the schedule error, and the direction it moves the
    inferred allocation."""
    budgets = np.logspace(0, 3, 60)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))

    ax = axes[0]
    matched = 1.0 + 1.6 * budgets ** (-0.28)
    penalty = 0.42 * budgets ** (-0.55)          # largest for the shortest runs
    ax.plot(budgets, matched, color=GREEN, lw=2.1,
            label="schedule matched to each budget")
    ax.plot(budgets, matched + penalty, color=ORANGE, lw=2.1, ls="--",
            label="one long schedule, stopped early")
    ax.fill_between(budgets, matched, matched + penalty, color=ORANGE, alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("training tokens"); ax.set_ylabel("loss")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title("A penalty that falls on the short runs",
                 fontsize=10.5, color=INK)
    ax.legend(fontsize=7.6, frameon=False)
    tidy(ax)

    ax = axes[1]
    C = np.logspace(0, 6, 200)
    ax.plot(C, C ** 0.73, color=ORANGE, lw=2.1,
            label="Kaplan et al. (2020): $N \\propto C^{0.73}$")
    ax.plot(C, C ** 0.50, color=GREEN, lw=2.1,
            label="Hoffmann et al. (2022): $N \\propto C^{0.50}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("compute $C$"); ax.set_ylabel("compute-optimal $N$")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.set_title("And the allocation it implies", fontsize=10.5, color=INK)
    ax.legend(fontsize=7.4, frameon=False, loc="upper left")
    tidy(ax)

    print("    illustrative: the only numbers are the two published exponents,")
    print("    each attributed to its source in the legend.")
    fig.tight_layout()
    save(fig, "02-schedule-illus")


ILLUS = [fig_scaling_illus, fig_schedule_illus]


CHEAP = [fig_memory, fig_precision, fig_pipeline, fig_dataparallel]

if __name__ == "__main__":
    print("Part XVII — scale figures")
    for f in CHEAP + ILLUS:
        f()
    if "--cheap" not in sys.argv:
        from train_figures import EXPENSIVE          # noqa: E402
        for f in EXPENSIVE:
            f()
    print("done ->", OUT)
