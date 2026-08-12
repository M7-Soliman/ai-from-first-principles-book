# brokenrepo — learn to read a failure

The Part IV project. Six to eight hours.

Unusually for this book, the task is not to build something. It is to
**break something, repeatedly, on purpose**, and learn to recognise the
wreckage.

## Why

You will spend far more of your working life reading failures than writing new
implementations. A framework's characteristic hazard, per §17, is that it
happily runs wrong programs — and the three most expensive bugs here (the
unregistered list, the missing `eval()`, the double softmax) all produce
**plausible numbers rather than errors**.

The only defence is to have seen each symptom once, deliberately, when you
already knew the answer.

## The rule

**Diagnose each bug from its symptoms before you look at what changed.**

Run it. Observe. Write down a hypothesis. Test the hypothesis. *Then* read the
source. `ANSWERS.md` exists — do not open it until you have written your
guesses down. Reading an answer feels like understanding and is not.

## Layout

```
brokenrepo/
  baseline.py     a correct training script — stage 1
  bugs/bug_01.py  … bug_11.py   one fault each, opaque names on purpose
  diagnose.py     runs a bug and prints what you need to see
  instrument.py   stage 3 — the checks that catch bugs before the run ends
  ANSWERS.md      what each bug is. Last resort.
```

## Working order

    python3 baseline.py                  # confirm the clean version trains
    python3 diagnose.py 1                # run bug 1, observe, hypothesise
    python3 diagnose.py --all            # a summary table once you are done
    python3 -m pytest test_baseline.py   # your instrumentation, stage 3

## Record as you go

| # | symptom observed | hypothesis | actual cause | right first time? |
|---|---|---|---|---|
| 1 | | | | |

Most people get six or seven of eleven. The ones people miss are usually
2 versus 3, and 6.
