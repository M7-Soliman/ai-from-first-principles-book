"""
Figures for Part IV — The Framework.

Nothing here is drawn by hand. The computational graph is read out of PyTorch's
own grad_fn links by traversal; the memory numbers are measured by counting the
tensors autograd is actually holding; the timings are real forward/backward
passes; the dropout distributions are real stochastic forward passes.

    python3 make_figures.py           # writes into ../../book/src/figures/fw/
"""
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "book", "src", "figures", "fw"))
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
        raise ValueError(f"{name}.svg is {h:.0f}pt — check axis limits")
    print(f"  wrote {name}.svg  ({h:.0f}pt)")


def tidy(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, length=2.5)
    if grid:
        ax.grid(True, color=LIGHT, lw=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# 1. The graph PyTorch actually built, read out of grad_fn by traversal.
# --------------------------------------------------------------------------
def fig_graph():
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    x = torch.tensor(2.0)
    z = w * x + b
    y_hat = torch.sigmoid(z)
    loss = (y_hat - torch.tensor(1.0)) ** 2

    # Walk the real graph. Every node and edge below is read from PyTorch.
    nodes, edges, order = {}, [], []

    def walk(fn, depth=0):
        if fn is None or fn in nodes:
            return
        nodes[fn] = type(fn).__name__.replace("Backward0", "")
        order.append(fn)
        for nxt, _ in fn.next_functions:
            if nxt is not None:
                edges.append((nxt, fn))
                walk(nxt, depth + 1)

    walk(loss.grad_fn)
    print(f"    graph read from PyTorch: {len(nodes)} nodes, {len(edges)} edges")

    # lay out by longest-path depth from the leaves
    depth = {}

    def d(fn):
        if fn in depth:
            return depth[fn]
        preds = [p for p, c in edges if c is fn]
        depth[fn] = 0 if not preds else 1 + max(d(p) for p in preds)
        return depth[fn]

    for fn in order:
        d(fn)
    by_level = {}
    for fn in order:
        by_level.setdefault(depth[fn], []).append(fn)

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    pos = {}
    for lv, fns in sorted(by_level.items()):
        for i, fn in enumerate(fns):
            pos[fn] = (lv * 1.72, -(i - (len(fns) - 1) / 2) * 1.0)

    # name the two leaves after the parameters they accumulate into
    leaf_names = iter(["w", "b"])
    labels, halfw = {}, {}
    for fn in order:
        raw = nodes[fn]
        if raw == "AccumulateGrad":
            labels[fn] = f"{next(leaf_names)}   .grad"
            halfw[fn] = 0.74
        else:
            labels[fn] = raw
            halfw[fn] = 0.62

    for src, dst in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        ax.add_patch(FancyArrowPatch((x0 + halfw[src], y0), (x1 - halfw[dst], y1),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=GRAY, lw=1.1, zorder=1))
    for fn, (px, py) in pos.items():
        leaf = nodes[fn] == "AccumulateGrad"
        hw = halfw[fn]
        ax.add_patch(FancyBboxPatch((px - hw, py - 0.26), 2 * hw, 0.52,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc="#fff7ed" if leaf else "#ecfeff",
                                    ec=ORANGE if leaf else CYAN, lw=1.3, zorder=2))
        ax.text(px, py, labels[fn], ha="center", va="center", fontsize=8.3,
                color=INK, zorder=3, fontweight="bold" if leaf else "normal")

    ax.text(max(p[0] for p in pos.values()) + 0.95, 0, "loss",
            ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.add_patch(FancyArrowPatch((max(p[0] for p in pos.values()) + 0.63, 0),
                                 (max(p[0] for p in pos.values()) + 0.80, 0),
                                 arrowstyle="-|>", mutation_scale=11, color=GRAY, lw=1.1))
    ax.set_xlim(-1.0, max(p[0] for p in pos.values()) + 1.45)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    ax.set_title("the graph PyTorch built for  loss = (sigmoid(wx + b) - 1)$^2$",
                 fontsize=9.5, color=GRAY, pad=6)
    save(fig, "01-graph")


# --------------------------------------------------------------------------
# 2. What the graph costs: activations retained, measured by counting them.
# --------------------------------------------------------------------------
def _retained_bytes(out):
    """Bytes the graph is holding, by traversing it.

    Two liveness traps, both of which silently UNDER-count if you fall in.
    PyTorch hands out a fresh Python wrapper on each grad_fn access and frees
    it when unreferenced, so id() values get recycled and a set of ids marks
    unvisited nodes as seen. The same is true of the saved tensors. Keep a
    reference to everything you have visited.
    """
    keep_nodes, keep_tensors = [], []
    seen, stack, nodes = set(), [out.grad_fn], []
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        seen.add(id(fn))
        keep_nodes.append(fn)              # <-- makes id() safe
        nodes.append(fn)
        stack.extend(n for n, _ in fn.next_functions)

    seen_t, total = set(), 0
    for fn in nodes:
        for attr in dir(fn):
            if not attr.startswith("_saved_"):
                continue
            v = getattr(fn, attr, None)
            if torch.is_tensor(v):
                keep_tensors.append(v)     # <-- and so does this
                if id(v) not in seen_t:
                    seen_t.add(id(v))
                    total += v.numel() * v.element_size()
    return total


def fig_memory():
    widths = [64, 128, 256, 512]
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    def at(width, batch, depth=6):
        model = nn.Sequential(*[m for _ in range(depth)
                                for m in (nn.Linear(width, width), nn.ReLU())])
        return _retained_bytes(model(torch.randn(batch, width)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    for w in widths:
        ys = [at(w, b) / 1024 for b in batches]
        ax1.plot(batches, ys, "o-", lw=1.6, ms=3.5, label=f"width {w}")
        print(f"    width {w:4d}: {ys[0]:8.0f}KB at b=1 -> {ys[-1]:8.0f}KB at b=256")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("retained by the graph (KB)")
    ax1.set_title("flat, then linear — two different things are being held",
                  fontsize=9, color=GRAY)
    ax1.legend(fontsize=7.5, frameon=False)
    tidy(ax1)

    depths = [2, 4, 8, 16, 32]
    ys = [at(128, 32, d) / 1024 for d in depths]
    for d, v in zip(depths, ys):
        print(f"    depth {d:3d}: {v:8.0f}KB")
    ax2.plot(depths, ys, "o-", color=ORANGE, lw=1.8, ms=5)
    ax2.set_xlabel("depth (layers)")
    ax2.set_ylabel("retained by the graph (KB)")
    ax2.set_title("linear in depth, with no flat region",
                  fontsize=9, color=GRAY)
    tidy(ax2)

    fig.tight_layout()
    save(fig, "02-memory")


# --------------------------------------------------------------------------
# 3. train() vs eval(): the same input, forwarded many times.
# --------------------------------------------------------------------------
def fig_train_eval():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),
                          nn.Dropout(0.5), nn.Linear(128, 1))
    xin = torch.randn(1, 64)

    model.train()
    tr = torch.stack([model(xin).detach().squeeze() for _ in range(2000)]).numpy()
    model.eval()
    with torch.no_grad():
        ev = torch.stack([model(xin).squeeze() for _ in range(2000)]).numpy()

    print(f"    train(): sd {tr.std():.4f}   eval(): sd {ev.std():.6f}")

    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.hist(tr, bins=60, color=ORANGE, alpha=0.85,
            label=f"model.train()  —  sd {tr.std():.3f}")
    ax.axvline(ev[0], color=CYAN, lw=2.2,
               label=f"model.eval()  —  sd {ev.std():.1e}")
    ax.set_xlabel("model output for one fixed input")
    ax.set_ylabel("count over 2,000 forward passes")
    ax.set_title("the same input, the same weights, 2,000 times",
                 fontsize=9, color=GRAY)
    ax.legend(fontsize=8, frameon=False)
    tidy(ax)
    fig.tight_layout()
    save(fig, "04-train-eval")


# --------------------------------------------------------------------------
# 4. Throughput against batch size — the vectorisation argument, measured.
# --------------------------------------------------------------------------
def fig_throughput():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    rates = []
    for bsz in batches:
        xin = torch.randn(bsz, 256)
        yb = torch.randint(0, 10, (bsz,))
        for _ in range(3):                                  # warm up
            opt.zero_grad()
            nn.functional.cross_entropy(model(xin), yb).backward()
            opt.step()
        reps = max(3, int(400 / bsz))
        t0 = time.perf_counter()
        for _ in range(reps):
            opt.zero_grad()
            nn.functional.cross_entropy(model(xin), yb).backward()
            opt.step()
        dt = time.perf_counter() - t0
        rates.append(reps * bsz / dt)
        print(f"    batch {bsz:4d}: {rates[-1]:9.0f} examples/s")

    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.plot(batches, rates, "o-", color=BLUE, lw=1.8, ms=5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("examples per second")
    ax.set_title("one CPU thread, identical work per example — "
                 "the gain is entirely from batching",
                 fontsize=9, color=GRAY)
    tidy(ax)
    fig.tight_layout()
    save(fig, "03-throughput")


if __name__ == "__main__":
    print("Part IV — framework figures")
    fig_graph()
    fig_memory()
    fig_train_eval()
    fig_throughput()
    print("done ->", OUT)
