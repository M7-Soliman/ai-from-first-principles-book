"""
Tests for the Part XX drills.

    python3 -m pytest test_drills.py -v

Unimplemented functions SKIP rather than fail.

Several assert a NEGATIVE result: a tokenizer that does not respect place
value, a benchmark score that overstates reliability, and a judge that prefers
whichever answer came first.
"""
import numpy as np
import pytest

import drills


def attempt(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{fn.__name__} not implemented yet")


def corpus(seed=0, n=1500):
    r = np.random.default_rng(seed)
    stems = ["the", "a", "of", "and", "to", "in", "model", "token", "data",
             "learn", "train", "text", "word", "value", "system", "result"]
    out = []
    for _ in range(n):
        k = int(r.integers(4, 14))
        out.append(" ".join(stems[int(i)] for i in r.integers(0, len(stems), k)) + ". ")
    return "".join(out)


# --------------------------------------------------------- byte-pair encoding

def test_pretokenize_never_merges_across_a_boundary():
    ch = attempt(drills.pretokenize, "the cat, 42!")
    assert all(isinstance(c, bytes) for c in ch)
    assert b"the" in ch and b"cat" in ch
    assert not any(b"the " == c for c in ch), "a word must not include the space"
    assert b"4" in ch and b"2" in ch, "digits split by default"
    nod = attempt(drills.pretokenize, "the cat, 42!", split_digits=False)
    assert b"42" in nod


def test_bpe_shortens_and_never_needs_an_unknown():
    text = corpus(0)
    merges = attempt(drills.bpe_train, [text], 200)
    assert attempt(drills.vocab_size, merges) == 256 + len(merges)
    base = len(attempt(drills.bpe_encode, text, []))
    enc = len(attempt(drills.bpe_encode, text, merges))
    assert enc < base * 0.7, f"{enc} against {base} at byte level"
    weird = "ϗ֍𝔊 zzqx —— \x00\x01"
    out = attempt(drills.bpe_encode, weird, merges)
    assert len(out) > 0 and all(isinstance(t, int) for t in out), \
        "anything encodes, because bytes are always available"


def test_merges_apply_in_training_order():
    """Greedy longest-match gives a different and wrong answer."""
    merges = attempt(drills.bpe_train, ["ab " * 50 + "abc " * 10], 3)
    enc = attempt(drills.bpe_encode, "abc", merges)
    assert len(enc) < 3


def test_fertility_is_worse_for_text_the_tokenizer_never_saw():
    """Section 3, as an assertion."""
    train = corpus(0)
    merges = attempt(drills.bpe_train, [train], 300)
    held = corpus(99, 200)
    f_en = attempt(drills.fertility, held, merges)
    cyr = held.translate(str.maketrans("abcdefghiklmnoprstuvyz",
                                       "абцдефгхиклмнопрстувыз"))
    f_cy = attempt(drills.fertility, cyr, merges)
    assert f_en < 0.6, f"seen text is efficient: {f_en:.3f} tokens/char"
    assert f_cy > 2.5 * f_en, \
        f"unseen script costs {f_cy/f_en:.1f}x per character"


def test_a_tokenizer_that_merges_digits_breaks_place_value():
    """Section 4: the mechanism behind arithmetic failures."""
    nums = list(range(1000, 1100)) + [7, 42, 999, 1024, 9999, 65536]
    text = " ".join(str(n) for n in nums) * 6
    m_split = attempt(drills.bpe_train, [text], 300, split_digits=True)
    m_merge = attempt(drills.bpe_train, [text], 300, split_digits=False)
    check = [1000, 1024, 1234, 9999]
    assert attempt(drills.place_value_is_respected, check, m_split, True), \
        "with digits split, every 4-digit number is 4 tokens"
    assert not attempt(drills.place_value_is_respected, check, m_merge, False), \
        "and without the rule, numbers of the same length differ"
    for n in check:
        assert attempt(drills.number_tokens, n, m_split, True) == len(str(n))


def test_the_vocabulary_tradeoff_has_two_sides():
    text = corpus(0)
    held = corpus(5, 300)
    prev_len = None
    for m in (0, 100, 400, 1200):
        merges = attempt(drills.bpe_train, [text], m) if m else []
        n = len(attempt(drills.bpe_encode, held, merges))
        if prev_len is not None:
            assert n <= prev_len, "more vocabulary cannot lengthen the sequence"
        prev_len = n
    small = attempt(drills.embedding_bytes, 32000, 4096)
    big = attempt(drills.embedding_bytes, 256000, 4096)
    assert big / small == pytest.approx(8.0), "and the embedding grows linearly"
    assert attempt(drills.embedding_bytes, 32000, 4096, tied=True) == small / 2


# ------------------------------------------------------------- context window

def test_the_middle_of_a_long_context_is_used_worst():
    """Section 12's position effect, as a profile rather than a window size."""
    prof = {0.0: 0.95, 0.1: 0.92, 0.2: 0.80, 0.4: 0.55, 0.5: 0.50,
            0.6: 0.58, 0.8: 0.79, 0.9: 0.90, 1.0: 0.96}
    first, mid, last = attempt(drills.needle_depth_profile, prof)
    assert mid < first and mid < last, \
        f"first {first:.2f}, middle {mid:.2f}, last {last:.2f}"


def test_attention_concentrates_on_few_positions():
    w = np.zeros(1000); w[:5] = 0.19; w[5:] = 0.05 / 995
    assert attempt(drills.kv_positions_used, w, 0.01) == pytest.approx(0.005)


# -------------------------------------------------------- in-context learning

def test_permuted_labels_move_the_answer():
    """Section 19: the experiment that separates mapping from format."""
    p = attempt(drills.permute_labels, 4, 1)
    assert list(p) == [1, 2, 3, 0]
    prompt = attempt(drills.make_icl_prompt, [("a", 0), ("b", 2)], "c")
    assert prompt.count("->") == 3
    permuted = attempt(drills.make_icl_prompt, [("a", 0), ("b", 2)], "c", mapping=p)
    assert permuted[2] == 1 and permuted[6] == 3


def test_the_reading_test_needs_permuted_to_fall_to_chance():
    n = 4
    assert attempt(drills.icl_reads_the_mapping, 0.90, 0.20, 0.26, n)
    # good on the real task, and unmoved by permuting: it used the format only
    assert not attempt(drills.icl_reads_the_mapping, 0.90, 0.88, 0.85, n)
    # never learned the task at all
    assert not attempt(drills.icl_reads_the_mapping, 0.27, 0.25, 0.25, n)


# ----------------------------------------------------------------- evaluation

def test_reliability_is_far_below_accuracy():
    """Section 44: benchmarks report the first and deployments need the second."""
    rng = np.random.default_rng(0)
    n_items, n_para = 400, 5
    labels = rng.integers(0, 2, n_items)
    P = np.where(rng.random((n_para, n_items)) < 0.9, labels[None, :], 1 - labels[None, :])
    accs = [attempt(drills.accuracy, P[i], labels) for i in range(n_para)]
    rel = attempt(drills.reliability, P, labels)
    assert np.mean(accs) > 0.85
    assert rel < 0.7, f"mean accuracy {np.mean(accs):.3f}, reliability {rel:.3f}"


def test_chained_success_collapses():
    assert attempt(drills.chained_success, 0.95, 20) == pytest.approx(0.3585, abs=1e-3)
    assert attempt(drills.chained_success, 0.99, 20) > 0.8
    assert attempt(drills.steps_for_success, 0.95, 0.5) == 13
    assert attempt(drills.steps_for_success, 0.99, 0.5) == 68


def test_reporting_the_best_prompt_overstates():
    scores = [0.61, 0.68, 0.72, 0.58, 0.79, 0.64]
    best, mean, worst, sd = attempt(drills.prompt_spread, scores)
    assert best == 0.79 and worst == 0.58
    assert best - mean > 0.1, "the best prompt flatters by a lot"
    assert sd > 0.05


def test_a_judge_can_prefer_whichever_came_first():
    assert attempt(drills.position_bias, [0.62, 0.41]) == pytest.approx(0.21)
    assert attempt(drills.position_bias, [0.50, 0.50]) == pytest.approx(0.0)


def test_length_control_removes_a_verbosity_win():
    """Section 42: without it, answering at greater length scores better."""
    rng = np.random.default_rng(1)
    n = 800
    la = rng.integers(50, 500, n).astype(float)
    lb = rng.integers(50, 500, n).astype(float)
    # the judge prefers the longer answer and nothing else
    wins = (la > lb).astype(float)
    raw = attempt(drills.accuracy, wins, np.ones(n))
    lc = attempt(drills.length_controlled_winrate, wins, la, lb)
    assert raw > 0.45
    assert abs(lc - 0.5) < abs(raw - 0.5) + 0.5
    # a genuinely better model wins at every length
    wins2 = np.ones(n)
    assert attempt(drills.length_controlled_winrate, wins2, la, lb) == pytest.approx(1.0)


def test_contamination_is_detectable_when_the_corpus_is_visible():
    test_items = ["the capital of france is paris and it is large",
                  "an entirely novel sentence about nothing at all here"]
    grams = {" ".join("the capital of france is paris and it is large".split()[:5])}
    hit = attempt(drills.contamination_flag, test_items, grams, n=5)
    assert hit == pytest.approx(0.5)
    assert attempt(drills.contamination_flag, test_items, set(), n=5) == 0.0


def test_sycophancy_and_over_refusal_are_measurable():
    neutral = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    after = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    assert attempt(drills.sycophancy_rate, neutral, after) == pytest.approx(0.5)
    safety, over = attempt(drills.refusal_rates,
                           [1, 1, 1, 1, 0], [0, 1, 1, 0, 0])
    assert safety == pytest.approx(0.8)
    assert over == pytest.approx(0.4), \
        "reporting only safety makes a model that refuses everything look perfect"


def test_best_of_n_is_biased_upward():
    """Section 37: Part XIV's winner's curse, in a new costume."""
    rng = np.random.default_rng(2)
    scores = rng.normal(0.7, 0.08, 5000)
    infl = [attempt(drills.best_of_n_inflation, scores, n, seed=0)
            for n in (1, 4, 16, 64)]
    assert infl[0] < 0.005
    assert infl[1] > 0.05
    assert infl[3] > infl[2] > infl[1], "more search means more inflation"


# ------------------------------------------------------- DPO / RLHF (section 29)

def test_kl_regularised_optimum_beats_every_other_policy_on_the_simplex():
    """Section 29's derivation, checked the way a claimed maximiser should be:
    against a large sample of competitors rather than against plausibility."""
    rng = np.random.default_rng(0)
    K = 6
    pi_ref = rng.dirichlet(np.ones(K))
    r = rng.normal(0, 1, K)
    beta = 0.5
    pi_star = attempt(drills.kl_regularised_optimum, pi_ref, r, beta)
    assert pi_star.sum() == pytest.approx(1.0)

    def objective(pi):
        return float(np.sum(pi * r) - beta * np.sum(pi * np.log(pi / pi_ref)))

    j_star = objective(pi_star)
    beaten = sum(objective(rng.dirichlet(np.ones(K))) > j_star + 1e-9
                 for _ in range(3000))
    assert beaten == 0, f"{beaten}/3000 random policies beat the claimed optimum"


def test_implicit_reward_recovers_the_true_reward_up_to_one_constant():
    """Section 29: r - beta*log(pi*/pi_ref) is the SAME constant (beta*log Z)
    for every response to a prompt, which is exactly why Bradley-Terry
    differences of the implicit reward reproduce true reward differences."""
    rng = np.random.default_rng(1)
    K = 5
    pi_ref = rng.dirichlet(np.ones(K))
    r = rng.normal(0, 1, K)
    beta = 0.7
    pi_star = attempt(drills.kl_regularised_optimum, pi_ref, r, beta)
    implicit = attempt(drills.implicit_reward, pi_star, pi_ref, beta)
    resid = r - implicit
    assert np.std(resid) < 1e-6, "the residual should be constant across responses"
    assert (implicit[0] - implicit[1]) == pytest.approx(r[0] - r[1], abs=1e-6)


def test_dpo_loss_is_lower_for_the_policy_that_agrees_with_the_reward():
    """Section 29: the DPO loss is a classification loss, so a policy that
    ranks responses the way the (unseen) reward does should score better than
    one that has the ranking backwards."""
    beta = 1.0
    pi_ref = np.array([0.5, 0.5])
    agrees = np.array([0.8, 0.2])       # favours response 0, as the pair does
    disagrees = np.array([0.2, 0.8])    # favours response 1
    right = attempt(drills.dpo_loss, agrees[0:1], pi_ref[0:1],
                    agrees[1:2], pi_ref[1:2], beta)
    wrong = attempt(drills.dpo_loss, disagrees[0:1], pi_ref[0:1],
                    disagrees[1:2], pi_ref[1:2], beta)
    assert right < wrong


def test_minimising_dpo_loss_on_bradley_terry_pairs_recovers_pi_star():
    """The full loop section 29 claims: sample preference pairs from the
    Bradley-Terry model built on a true reward, fit a tabular policy by
    minimising the DPO loss, and check it lands near pi* = the KL-regularised
    optimum for that same reward — not merely near the true reward's ranking."""
    rng = np.random.default_rng(2)
    K = 4
    pi_ref = rng.dirichlet(np.ones(K))
    r = rng.normal(0, 1, K)
    beta = 0.6
    pi_star = attempt(drills.kl_regularised_optimum, pi_ref, r, beta)

    n_pairs = 3000
    i = rng.integers(0, K, n_pairs)
    j = rng.integers(0, K, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    p_i_wins = 1.0 / (1.0 + np.exp(-(r[i] - r[j])))
    i_wins = rng.random(len(i)) < p_i_wins
    chosen = np.where(i_wins, i, j)
    rejected = np.where(i_wins, j, i)

    theta = np.zeros(K)
    for _ in range(600):
        probs = np.exp(theta - theta.max())
        probs /= probs.sum()
        grad = np.zeros(K)
        eps = 1e-3
        base = attempt(drills.dpo_loss, probs[chosen], pi_ref[chosen],
                       probs[rejected], pi_ref[rejected], beta)
        for k in range(K):
            t2 = theta.copy(); t2[k] += eps
            p2 = np.exp(t2 - t2.max()); p2 /= p2.sum()
            l2 = attempt(drills.dpo_loss, p2[chosen], pi_ref[chosen],
                        p2[rejected], pi_ref[rejected], beta)
            grad[k] = (l2 - base) / eps
        theta -= 0.5 * grad
    fitted = np.exp(theta - theta.max()); fitted /= fitted.sum()
    assert np.abs(fitted - pi_star).max() < 0.1, \
        f"fitted {fitted} should be close to pi* {pi_star}"


# ------------------------ movement G: keeping and changing what a model knows

def test_the_forgetting_curve_is_a_fraction_of_the_total_loss():
    f = attempt(drills.forgetting_curve, 1.0, [1.0, 0.8, 0.6, 0.5], [0, 1, 2, 3])
    assert f[0] == pytest.approx(0.0)
    assert f[-1] == pytest.approx(1.0)
    assert f[1] == pytest.approx(0.4)


def test_replay_adds_the_right_number_of_old_examples():
    rng = np.random.default_rng(0)
    Xn, yn = np.zeros((200, 4)), np.zeros(200)
    Xo, yo = np.ones((500, 4)), np.ones(500)
    X, y = attempt(drills.replay_batch, Xn, yn, Xo, yo, 0.05, rng)
    assert len(X) == 210 and len(y) == 210
    assert y.sum() == pytest.approx(10)


def test_zero_replay_changes_nothing():
    rng = np.random.default_rng(1)
    Xn, yn = np.zeros((50, 3)), np.zeros(50)
    X, y = attempt(drills.replay_batch, Xn, yn, np.ones((10, 3)), np.ones(10), 0.0, rng)
    assert len(X) == 50 and y.sum() == 0


def test_interpolation_endpoints():
    a, b = np.array([0.0, 1.0]), np.array([2.0, 3.0])
    assert np.allclose(attempt(drills.interpolate, a, b, 0.0), a)
    assert np.allclose(attempt(drills.interpolate, a, b, 1.0), b)
    assert np.allclose(attempt(drills.interpolate, a, b, 0.5), [1.0, 2.0])


def test_a_flat_path_has_no_barrier_and_a_dip_does():
    flat = [0.9, 0.9, 0.9, 0.9, 0.9]
    dip = [0.9, 0.8, 0.6, 0.8, 0.9]
    assert attempt(drills.barrier, flat) == pytest.approx(0.0)
    assert attempt(drills.barrier, dip) == pytest.approx(0.3)


def test_the_barrier_compares_against_the_endpoints_not_the_best_point():
    """A path that climbs in the middle has a negative barrier, which is the
    honest report: averaging helped."""
    up = [0.8, 0.85, 0.9, 0.85, 0.8]
    assert attempt(drills.barrier, up) < 0


def test_a_soup_is_the_mean():
    ts = [np.array([0.0, 0.0]), np.array([2.0, 4.0]), np.array([4.0, 2.0])]
    assert np.allclose(attempt(drills.soup, ts), [2.0, 2.0])


def test_ema_lags_and_converges():
    ts = np.ones((50, 2))
    ts[0] = 0.0
    out = attempt(drills.ema, ts, 0.9)
    assert np.allclose(out[0], 0.0)
    assert out[5, 0] < out[20, 0] < 1.0
    assert out[-1, 0] == pytest.approx(1.0, abs=0.01)


def test_a_task_vector_is_a_displacement():
    t0 = np.array([1.0, 2.0, 3.0])
    ft = np.array([1.5, 2.0, 2.0])
    tau = attempt(drills.task_vector, ft, t0)
    assert np.allclose(tau, [0.5, 0.0, -1.0])
    assert np.allclose(attempt(drills.apply_task_vectors, t0, [tau], [1.0]), ft)


def test_negating_a_task_vector_returns_to_the_start():
    t0 = np.array([0.0, 0.0])
    ft = np.array([3.0, -1.0])
    tau = attempt(drills.task_vector, ft, t0)
    assert np.allclose(attempt(drills.apply_task_vectors, ft, [tau], [-1.0]), t0)


def test_task_cosine():
    assert attempt(drills.task_cosine, np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert attempt(drills.task_cosine, np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert attempt(drills.task_cosine, np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_the_lift_is_measured_against_the_cross_task_baseline():
    """Reporting the sum's raw accuracy hides whether the arithmetic did
    anything: aligned tasks make either model alone look good at both."""
    assert attempt(drills.arithmetic_lift, [0.71, 0.76], [0.51, 0.51]) == pytest.approx(0.225)
    assert attempt(drills.arithmetic_lift, [0.9, 0.9], [0.9, 0.9]) == pytest.approx(0.0)


def test_a_memory_with_fewer_facts_than_dimensions_is_exact():
    rng = np.random.default_rng(2)
    K, V = rng.normal(size=(40, 64)), rng.normal(size=(40, 64))
    M = attempt(drills.fit_memory, K, V)
    assert np.allclose(K @ M, V, atol=1e-8)


def test_a_memory_with_more_facts_than_dimensions_is_not():
    """The check that has to come before any editing measurement: with an
    overdetermined store, 'collateral damage' is mostly the fit residual."""
    rng = np.random.default_rng(3)
    K, V = rng.normal(size=(200, 32)), rng.normal(size=(200, 32))
    M = attempt(drills.fit_memory, K, V)
    assert not np.allclose(K @ M, V, atol=1e-3)


def test_a_rank_one_edit_is_exact_and_local():
    rng = np.random.default_rng(4)
    K, V = rng.normal(size=(30, 64)), rng.normal(size=(30, 64))
    M0 = attempt(drills.fit_memory, K, V)
    v_new = rng.normal(size=64)
    M = attempt(drills.rank_one_edit, M0, K[7], v_new)
    assert np.allclose(K[7] @ M, v_new, atol=1e-8)
    assert not np.allclose(M, M0)
    dmg, base = attempt(drills.edit_damage, M0, M, K, V, [7])
    assert base < 1e-16
    assert dmg > 0


def test_damage_accumulates_with_sequential_edits():
    rng = np.random.default_rng(5)
    K, V = rng.normal(size=(60, 100)), rng.normal(size=(60, 100))
    M0 = attempt(drills.fit_memory, K, V)
    M, idx = M0.copy(), []
    prev = 0.0
    for i in (1, 3, 5, 9, 14, 20):
        while len(idx) < i:
            j = len(idx)
            M = attempt(drills.rank_one_edit, M, K[j], rng.normal(size=100))
            idx.append(j)
        dmg, _ = attempt(drills.edit_damage, M0, M, K, V, idx)
        assert dmg > prev
        prev = dmg


def test_the_edit_budget_is_the_fact_count_not_the_dimension():
    assert attempt(drills.edits_until_ruined, 90) == 90
    assert attempt(drills.edits_until_ruined, 5000) == 5000
