# Answers

**Do not read this until you have written your hypotheses down.** Reading an
answer feels like understanding and is not.

Measured signatures, two epochs, seed 0. Yours will differ in the third
decimal; the *pattern* is what matters.

| # | signature | cause | book |
|---|---|---|---|
| 01 | grad norm **24** vs 1.2 | `opt.zero_grad()` deleted — gradients accumulate across batches, so you step on a growing sum | IV §13 · VII §21 |
| 02 | loss flat at 0.699, val 0.556, grad **non-zero** | `opt.step()` deleted — gradients are computed correctly and never applied | IV §13 |
| 03 | loss flat at 0.699, val 0.556, grad **exactly 0.000** | `loss.backward()` deleted — nothing is computed, so there is nothing to apply | IV §13 |
| 04 | **130** parameters instead of 5,634 | layers in a plain Python list — a list is not a Module, so nothing registers | IV §8 |
| 05 | `AttributeError: cannot assign module before Module.__init__() call` | `super().__init__()` omitted — the registries never exist | IV §8 |
| 06 | trains, val still ~0.88, loss oddly high | softmax applied before `cross_entropy`, which applies it again | II §25 · IV §16 |
| 07 | **evaluation not reproducible** | `model.eval()` never called — dropout stays active while you measure | IV §14 |
| 08 | accumulator is a **tensor** | `total += loss` instead of `loss.item()` — each batch's graph is retained | IV §3 |
| 09 | val 0.633, grad 2.05, loss oscillates | `shuffle=False` on class-sorted data — batches are not samples of the distribution | II §17 · IV §11 |
| 10 | val 0.464, grad collapses to 0.006 | learning rate 1000× too large — Part III §9's stability bound exceeded | III §9 |
| 11 | `RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float` | float64 features meeting float32 weights | IV §2, §16 |

## The three that matter most

**02 versus 03** is the pair people miss, and the whole reason §17 says to
print a gradient norm. Both show a flat loss and chance accuracy. They are
distinguishable by exactly one number: with no `backward()` the gradient is
**exactly zero**; with no `step()` it is a perfectly healthy non-zero value
that nothing ever consumes. Identical symptom, opposite causes, one cheap
measurement between them.

**06** is the expensive one. It does not crash, the loss decreases, and
accuracy is nearly normal — it is simply worse than it should be, forever. A
double softmax flattens the distribution toward uniform, which weakens every
gradient. Nothing will ever tell you; you would have to know to look.

**08** is the one whose textbook symptom you cannot see here, and that is
worth knowing. The book says the failure is memory growth — and it is, at
scale, after thousands of batches. On a 5,634-parameter model over 47 batches
there is nothing to see, and `tracemalloc` cannot see torch's allocations
anyway. What *is* immediately observable is that the accumulator is a
`Tensor` with a `grad_fn` rather than a `float`.

Which is a real lesson about diagnosis rather than a limitation of the
exercise: **some bugs are only visible at the scale where they hurt.** The
defence is to recognise the shape of the mistake in the source, because you
will not always get a symptom.
