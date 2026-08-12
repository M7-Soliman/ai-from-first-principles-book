"""
Run a bug and show you what you need to see.

    python3 diagnose.py 4          # run bug 4
    python3 diagnose.py --all      # summary table across all eleven
    python3 diagnose.py 0          # the clean baseline, for comparison

It deliberately prints only SYMPTOMS: whether it ran, the parameter count, the
loss trajectory, the gradient norm, the validation accuracy, what type the
loss accumulator ended up, and whether two evaluations of the same model
agree. That is the same information you would have on a real
project at 2am, and it is enough for every one of the eleven.

Write your hypothesis down before opening the source.
"""
import argparse
import contextlib
import gc
import importlib.util
import io

import torch


def load(n):
    path = "baseline.py" if n == 0 else f"bugs/bug_{n:02d}.py"
    spec = importlib.util.spec_from_file_location(f"variant_{n}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe(n, epochs=2):
    """Return a dict of symptoms. Never raises -- a crash IS a symptom."""
    out = {"n": n, "ran": False, "error": None, "params": None,
           "losses": None, "val": None, "stable": None,
           "grad": None, "mem_mb": None}
    try:
        buf = io.StringIO()
        mod = load(n)

        # Sample live tensors DURING the epoch. A retained graph is freed as
        # soon as train() returns, so measuring afterwards sees nothing.
        peak_live = [0]     # 1 if the loss accumulator retains a graph

        def sample(frame_locals):
            # Is the running loss a float, or a Tensor still wired to its
            # graph? The memory cost of the second only shows at scale, but
            # the TYPE is observable immediately -- and it is the bug.
            acc = frame_locals.get("running")
            if torch.is_tensor(acc):
                peak_live[0] = 1

        mod.BATCH_PROBE = sample
        with contextlib.redirect_stdout(buf):
            model, history, (Xte, yte) = mod.train(epochs=2, verbose=False)
        mod.BATCH_PROBE = None
        out["mem_mb"] = peak_live[0]
        out["grad"] = getattr(mod.train, "last_grad_norms", None)
        out["ran"] = True
        out["params"] = sum(p.numel() for p in model.parameters())
        out["losses"] = [round(float(h[0]), 4) for h in history]
        out["val"] = round(float(history[-1][1]), 4)
        # does evaluating twice give the same answer?
        a = mod.evaluate(model, Xte, yte)
        b = mod.evaluate(model, Xte, yte)
        out["stable"] = abs(a - b) < 1e-9
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}".split("\n")[0][:110]
    return out


def show(r):
    label = "baseline" if r["n"] == 0 else f"bug {r['n']:02d}"
    print(f"\n=== {label} " + "=" * (58 - len(label)))
    if not r["ran"]:
        print(f"  CRASHED   {r['error']}")
        print("  -> read the LAST line of the traceback, not the first (§17)")
        return
    print(f"  parameters      {r['params']:,}")
    print(f"  loss by epoch   {r['losses']}")
    print(f"  final val acc   {r['val']}")
    print(f"  grad norm/epoch {[round(g, 3) for g in (r['grad'] or [])]}")
    print(f"  loss accumulator  "
          f"{'TENSOR (retains the graph)' if r['mem_mb'] else 'float'}")
    print(f"  evaluation is   {'stable' if r['stable'] else 'NOT REPRODUCIBLE'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bug", nargs="?", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.all:
        rows = [probe(n) for n in range(0, 12)]
        print(f"\n{'':<10}{'params':>9}{'ran':>5}{'loss':>9}{'val':>7}"
              f"{'grad':>9}{'acc':>9}{'stable':>8}")
        print("-" * 66)
        for r in rows:
            name = "baseline" if r["n"] == 0 else f"bug {r['n']:02d}"
            if not r["ran"]:
                print(f"{name:<10}{'-':>9}{'NO':>6}   {r['error'][:40]}")
            else:
                loss = r["losses"][-1] if r["losses"] else float("nan")
                g = r["grad"][-1] if r["grad"] else float("nan")
                print(f"{name:<10}{r['params']:>9,}{'yes':>5}{loss:>9.4f}"
                      f"{r['val']:>7.3f}{g:>9.3f}"
                      f"{'tensor' if r['mem_mb'] else 'float':>9}"
                      f"{'yes' if r['stable'] else 'NO':>8}")
        print("\nEvery row that ran is a bug you must explain from these "
              "columns alone.\nA crash is the easy case. The rows that ran are "
              "the expensive ones.")
        return

    if a.bug is None:
        ap.error("give a bug number, or --all")
    show(probe(a.bug))


if __name__ == "__main__":
    main()
