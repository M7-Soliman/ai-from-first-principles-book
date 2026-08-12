# tinygrad — an autograd engine and the network library it supports

The Part VII project. Ten to fifteen hours, and the centrepiece of the book.

**The rule: no framework, no autograd, at any stage.** NumPy for array
arithmetic only. By the end you will have rebuilt the core of PyTorch in a few
hundred lines, and no framework will be a black box again.

## Layout

```
tinygrad/
  engine.py      stage 1 — the scalar Value engine
  nn.py          stage 2 — Neuron, Layer, MLP, losses, SGD
  tensor.py      stage 3 — the array engine
  tests/         run these constantly
```

## Working order

    python3 -m pytest tests/test_engine.py -v      # stage 1
    python3 -m pytest tests/test_nn.py -v          # stage 2
    python3 -m pytest tests/test_tensor.py -v      # stage 3
    python3 -m pytest tests -v                     # everything

Unimplemented pieces **skip**, so the suite is green from the first minute and
turns into a progress bar (40 tests total: 16 engine, 19 nn, 5 tensor). Nothing
here is graded on style; everything is graded by gradient check.

## Stages

**1 — the scalar engine.** `Value` with `+ - * / **`, `relu`, `tanh`, `exp`,
`log`, and `backward()`. Deliverable: a gradient check against numerical
differentiation on ten random expressions, **including at least one that
reuses a value** — that is exactly where accumulation bugs hide.

**2 — the network library.** `Neuron`, `Layer`, `MLP`, `parameters()`, squared
error and cross-entropy, an `SGD` with momentum, and `zero_grad()` with a
comment explaining why it is necessary. Deliverable: XOR to near-zero loss,
then a spiral or two-moons decision boundary.

**3 — make it fast.** Rewrite on arrays. Same structure; only the local
derivatives change from numbers to matrix expressions. Watch the broadcasting
backward pass — it must **sum** over broadcast axes, and that is the step
people get wrong. Deliverable: identical gradients to stage 1, and a measured
speedup you report.

**4 — train something real.** A real classification dataset, two hidden layers,
Part VI's train/validation/test discipline, and Part V's intervals on the final
number. Reproduce Part VII §20's diagnostic sequence, including deliberately
breaking the model to confirm the overfit-ten-examples test catches it.

**5 — reproduce the part.** Use your library as an instrument: vanishing
gradients, initialisation scale, cross-entropy versus squared error, dead
ReLUs against learning rate, residual connections restoring gradient flow.

## Done when

- [ ] Gradient checks pass on expressions that reuse values
- [ ] XOR trains to near-zero loss from several random seeds
- [ ] The array engine gives identical gradients to the scalar one
- [ ] You have measured and reported the vectorisation speedup
- [ ] A real dataset trains, with an interval on the reported accuracy
- [ ] All five reproduction figures exist
- [ ] Your README names something the framework does that now seems obvious
