"""
Reference solutions for the Part XX drills.

Importing this module monkey-patches the implementations into `drills`.
"""
import re
from collections import Counter

import numpy as np

import drills


# --------------------------------------------------------- byte-pair encoding

def pretokenize(text, split_digits=True):
    b = text.encode() if isinstance(text, str) else text
    pat = (rb"[A-Za-z]+|[0-9]|\s+|[^A-Za-z0-9\s]+" if split_digits
           else rb"[A-Za-z]+|[0-9]+|\s+|[^A-Za-z0-9\s]+")
    return re.findall(pat, b)


def bpe_train(texts, n_merges, split_digits=True):
    freq = Counter()
    for t in texts:
        freq.update(pretokenize(t, split_digits))
    words = {w: tuple(w) for w in freq}
    merges, nxt = [], 256
    for _ in range(n_merges):
        pairs = Counter()
        for w, seq in words.items():
            f = freq[w]
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        best, cnt = pairs.most_common(1)[0]
        if cnt < 2:
            break
        merges.append((best, nxt))
        nw = {}
        for w, seq in words.items():
            r, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                    r.append(nxt); i += 2
                else:
                    r.append(seq[i]); i += 1
            nw[w] = tuple(r)
        words = nw
        nxt += 1
    return merges


def bpe_encode(text, merges, split_digits=True):
    order = {pair: (i, new) for i, (pair, new) in enumerate(merges)}
    out = []
    for chunk in pretokenize(text, split_digits):
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


def fertility(text, merges, split_digits=True):
    return len(bpe_encode(text, merges, split_digits)) / max(len(text), 1)


def vocab_size(merges):
    return 256 + len(merges)


def number_tokens(n, merges, split_digits=True):
    return len(bpe_encode(str(n), merges, split_digits))


def place_value_is_respected(numbers, merges, split_digits=True):
    by_len = {}
    for n in numbers:
        by_len.setdefault(len(str(n)), set()).add(
            number_tokens(n, merges, split_digits))
    return all(len(v) == 1 for v in by_len.values())


def embedding_bytes(vocab, d_model, bytes_per=2, tied=False):
    return float(vocab * d_model * bytes_per * (1 if tied else 2))


# ------------------------------------------------------------- context window

def kv_positions_used(attn_weights, threshold=0.01):
    w = np.asarray(attn_weights, float)
    return float((w > threshold).mean())


def needle_depth_profile(correct_by_depth):
    ds = sorted(correct_by_depth)
    k = max(1, len(ds) // 3)
    thirds = (ds[:k], ds[k:2 * k], ds[2 * k:])
    return tuple(float(np.mean([correct_by_depth[d] for d in t])) if t else 0.0
                 for t in thirds)


# -------------------------------------------------------- in-context learning

def make_icl_prompt(pairs, query, mapping=None):
    seq = []
    for x, y in pairs:
        seq += [x, "->", int(mapping[y]) if mapping is not None else y, ";"]
    seq += [query, "->"]
    return seq


def permute_labels(n_classes, shift=1):
    return np.array([(c + shift) % n_classes for c in range(n_classes)])


def icl_reads_the_mapping(acc_correct, acc_permuted, acc_random, n_classes):
    chance = 1.0 / n_classes
    return bool(acc_correct > chance + 0.15 and acc_permuted <= chance + 0.05)


# ----------------------------------------------------------------- evaluation

def accuracy(preds, labels):
    return float((np.asarray(preds) == np.asarray(labels)).mean())


def reliability(preds_by_paraphrase, labels):
    P = np.asarray(preds_by_paraphrase)
    y = np.asarray(labels)
    return float((P == y[None, :]).all(0).mean())


def prompt_spread(scores):
    s = np.asarray(scores, float)
    return float(s.max()), float(s.mean()), float(s.min()), float(s.std())


def chained_success(per_step, n_steps):
    return float(per_step ** n_steps)


def steps_for_success(per_step, target):
    if per_step >= 1.0:
        return int(1e9)
    if per_step <= 0.0:
        return 0
    return int(np.floor(np.log(target) / np.log(per_step)))


def position_bias(scores_by_position):
    s = np.asarray(scores_by_position, float)
    return float(abs(s[0] - s[1]))


def length_controlled_winrate(wins, lengths_a, lengths_b, n_bins=4):
    w = np.asarray(wins, float)
    d = np.asarray(lengths_a, float) - np.asarray(lengths_b, float)
    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    rates = []
    for i in range(n_bins):
        m = (d >= edges[i]) & (d < edges[i + 1])
        if m.sum():
            rates.append(w[m].mean())
    return float(np.mean(rates)) if rates else float(w.mean())


def contamination_flag(test_items, corpus_ngrams, n=13):
    hits = 0
    for it in test_items:
        toks = it.split()
        grams = {" ".join(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}
        if grams & corpus_ngrams:
            hits += 1
    return float(hits / max(len(test_items), 1))


def sycophancy_rate(answers_neutral, answers_after_opinion):
    a = np.asarray(answers_neutral)
    b = np.asarray(answers_after_opinion)
    return float((a != b).mean())


def refusal_rates(refused_harmful, refused_benign):
    return (float(np.asarray(refused_harmful, float).mean()),
            float(np.asarray(refused_benign, float).mean()))


def best_of_n_inflation(scores, n, trials=2000, seed=0):
    r = np.random.default_rng(seed)
    s = np.asarray(scores, float)
    picks = s[r.integers(0, len(s), (trials, n))].max(1)
    return float(picks.mean() - s.mean())


# ------------------------------------------------------- DPO / RLHF (section 29)

def kl_regularised_optimum(pi_ref, r, beta):
    pi_ref = np.asarray(pi_ref, float)
    r = np.asarray(r, float)
    w = pi_ref * np.exp(r / float(beta))
    return w / w.sum()


def implicit_reward(pi, pi_ref, beta):
    pi = np.asarray(pi, float)
    pi_ref = np.asarray(pi_ref, float)
    return float(beta) * (np.log(pi) - np.log(pi_ref))


def dpo_loss(pi_w, pi_ref_w, pi_l, pi_ref_l, beta):
    pi_w, pi_ref_w = np.asarray(pi_w, float), np.asarray(pi_ref_w, float)
    pi_l, pi_ref_l = np.asarray(pi_l, float), np.asarray(pi_ref_l, float)
    logits = float(beta) * ((np.log(pi_w) - np.log(pi_ref_w))
                             - (np.log(pi_l) - np.log(pi_ref_l)))
    # -log sigmoid(x) computed stably as log(1 + exp(-x))
    return float(np.mean(np.logaddexp(0.0, -logits)))


# --------------------------------------------------------------------- patch


# ------------------------ movement G: keeping and changing what a model knows

def forgetting_curve(acc_before, acc_after, distances):
    lost = float(acc_before) - np.asarray(acc_after, float)
    total = float(acc_before) - float(np.asarray(acc_after, float)[-1])
    if abs(total) < 1e-12:
        return np.zeros_like(lost)
    return lost / total


def replay_batch(X_new, y_new, X_old, y_old, fraction, rng):
    X_new, y_new = np.asarray(X_new, float), np.asarray(y_new, float)
    X_old, y_old = np.asarray(X_old, float), np.asarray(y_old, float)
    k = int(round(float(fraction) * len(X_new)))
    if k <= 0:
        return X_new, y_new
    idx = rng.integers(0, len(X_old), k)
    return np.vstack([X_new, X_old[idx]]), np.concatenate([y_new, y_old[idx]])


def interpolate(theta_a, theta_b, alpha):
    a = np.asarray(theta_a, float)
    b = np.asarray(theta_b, float)
    return (1.0 - float(alpha)) * a + float(alpha) * b


def barrier(scores, alphas=None):
    # Deviation below the straight line joining the endpoints, maximised over
    # the interior. Comparing against the endpoint *mean* instead reports zero
    # whenever the path rises above the chord, which is exactly the case worth
    # distinguishing: averaging that helps should read as a negative barrier.
    s = np.asarray(scores, float)
    a = np.linspace(0, 1, len(s)) if alphas is None else np.asarray(alphas, float)
    chord = (1 - a) * s[0] + a * s[-1]
    dev = chord - s
    return float(dev[1:-1].max()) if len(s) > 2 else 0.0


def soup(thetas):
    return np.mean(np.asarray(thetas, float), axis=0)


def ema(thetas, decay):
    t = np.asarray(thetas, float)
    out = np.zeros_like(t)
    cur = t[0].copy()
    out[0] = cur
    for i in range(1, len(t)):
        cur = decay * cur + (1.0 - decay) * t[i]
        out[i] = cur
    return out


def task_vector(theta_finetuned, theta_pretrained):
    return np.asarray(theta_finetuned, float) - np.asarray(theta_pretrained, float)


def apply_task_vectors(theta0, taus, coeffs):
    out = np.asarray(theta0, float).copy()
    for tau, c in zip(taus, coeffs):
        out = out + float(c) * np.asarray(tau, float)
    return out


def task_cosine(tau_a, tau_b):
    a, b = np.asarray(tau_a, float), np.asarray(tau_b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def arithmetic_lift(sum_acc, cross_acc):
    return float(np.mean(sum_acc) - np.mean(cross_acc))


def fit_memory(K, V):
    return np.linalg.lstsq(np.asarray(K, float), np.asarray(V, float), rcond=None)[0]


def rank_one_edit(M, k, v_new):
    M = np.asarray(M, float)
    k = np.asarray(k, float)
    resid = np.asarray(v_new, float) - k @ M
    return M + np.outer(k / (k @ k), resid)


def edit_damage(M0, M, K, V, edited_idx):
    K, V = np.asarray(K, float), np.asarray(V, float)
    others = np.setdiff1d(np.arange(len(K)), np.asarray(edited_idx))
    dmg = float(np.mean(np.sum((K[others] @ M - V[others]) ** 2, 1)))
    base = float(np.mean(np.sum((K[others] @ M0 - V[others]) ** 2, 1)))
    return dmg, base


def edits_until_ruined(n_facts):
    return int(n_facts)


for _n, _f in list(globals().items()):
    if callable(_f) and not _n.startswith("_") and hasattr(drills, _n):
        setattr(drills, _n, _f)
