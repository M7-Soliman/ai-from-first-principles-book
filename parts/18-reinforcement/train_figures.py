"""
Expensive figures for Part XVIII — Reinforcement Learning.

Every figure this part needs turns out to be cheap: the MDPs are small enough
that the true values are computable in closed form, and computing all seven
figures (in `make_figures.py`'s CHEAP list) takes about a minute in total.
Nothing here trains a network, so there is currently nothing expensive to put
in this module.

It exists only so that the documented default invocation

    python3 make_figures.py            # everything

does not crash on `from train_figures import EXPENSIVE` — CHEAP already is
everything. If a figure that genuinely needs training time is added later, it
goes in EXPENSIVE below as a zero-argument callable that saves its own SVG(s),
the same shape as the functions in CHEAP.
"""

EXPENSIVE = []
