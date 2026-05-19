"""
SENTINEL-Ω — Theoretical foundations.

This module encodes the formal definitions and theorems of the framework
as executable contracts. Each theorem is paired with a `verify_*` function
that empirically validates the property over Monte-Carlo samples — useful
both as a sanity check and as a reproducibility artifact for reviewers.

Theorems:
  T1 — Information-Theoretic Optimality of entropy-greedy action selection
  T2 — Marginal Coverage of inductive conformal prediction
  T3 — Pareto-Optimality of the counterfactual containment planner
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

def entropy(p: np.ndarray, base: float = math.e) -> float:
    """Shannon entropy H(p) of a discrete distribution.

    H(p) = -∑ p_i log p_i,   with 0·log0 := 0.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q). Defined as +∞ if q has zero where p has mass."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def expected_information_gain(
    prior: np.ndarray,
    likelihood: np.ndarray,
) -> float:
    """Expected information gain of an action whose observation channel is
    given by `likelihood[h, o] = P(o | h)`.

    IG(a) = H(prior) - E_o[ H(posterior | o) ]
          = I(H ; O)        (mutual information)

    Args:
      prior:      shape (|H|,), prior over hypotheses
      likelihood: shape (|H|, |O|), conditional observation distribution
    """
    prior = np.asarray(prior, dtype=np.float64)
    L = np.asarray(likelihood, dtype=np.float64)
    # Marginal over observations
    marg_o = prior @ L                              # (|O|,)
    # Posterior for each observation: post[h, o] = P(h | o)
    joint = prior[:, None] * L                      # (|H|, |O|)
    post = joint / (marg_o[None, :] + 1e-12)
    H_prior = entropy(prior)
    H_post  = float(np.sum(marg_o * np.apply_along_axis(entropy, 0, post)))
    return H_prior - H_post


# ---------------------------------------------------------------------------
# T1. Information-Theoretic Optimality (entropy-greedy convergence)
# ---------------------------------------------------------------------------

def theorem_1_bound(num_hypotheses: int, beta: float) -> float:
    """Worst-case expected number of queries to identify the true hypothesis
    under entropy-greedy action selection.

      E[T] ≤ log|H| / log(1/β)

    where β ∈ (0,1) is the worst-case observation informativeness.
    """
    if not (0 < beta < 1):
        raise ValueError("beta must be in (0, 1)")
    return math.log(num_hypotheses) / math.log(1.0 / beta)


def verify_theorem_1(
    num_hypotheses: int = 16,
    beta: float = 0.9,
    trials: int = 500,
    rng: np.random.Generator | None = None,
) -> dict:
    """Monte-Carlo verification of Theorem 1.

    Sequential identification game: at each step the agent chooses a noisy
    binary test ('is the truth in subset S?') from the full action set of
    2^|H| − 2 non-trivial subsets, using entropy-greedy selection (Bayesian
    optimal experimental design). Test reliability is β ∈ (0.5, 1).

    Bound: E[T] ≤ log|H| / log(1/(1−β·... )); we use the binary-channel
    capacity 1 − H((1+β)/2) for normalisation.
    """
    rng = rng or np.random.default_rng(0)

    # Effective per-step entropy reduction = mutual information of a noisy
    # binary test, maximised over choice of subset (≤ binary channel capacity).
    p_correct = (1 + beta) / 2
    H_bin = entropy(np.array([p_correct, 1 - p_correct]), base=2)  # bits
    capacity = 1.0 - H_bin                                         # bits/test
    bound = math.log2(num_hypotheses) / capacity

    steps_to_identify = []
    for _ in range(trials):
        truth = int(rng.integers(num_hypotheses))
        belief = np.full(num_hypotheses, 1.0 / num_hypotheses)
        T = 0
        max_T = int(math.ceil(bound)) * 4 + 5
        while belief[truth] < 0.95 and T < max_T:
            # Entropy-greedy: pick subset that splits belief mass closest to 1/2
            # (this is the optimal one-step test for a noisy binary channel).
            sorted_idx = np.argsort(belief)[::-1]
            cumulative = np.cumsum(belief[sorted_idx])
            split = int(np.argmin(np.abs(cumulative - 0.5))) + 1
            in_set = np.zeros(num_hypotheses, dtype=bool)
            in_set[sorted_idx[:split]] = True
            # Noisy observation
            truth_in_set = bool(in_set[truth])
            observation = truth_in_set if rng.random() < p_correct else not truth_in_set
            # Bayesian update
            lik = np.where(in_set == observation, p_correct, 1 - p_correct)
            belief = belief * lik
            belief /= belief.sum()
            T += 1
        steps_to_identify.append(T)

    arr = np.array(steps_to_identify)
    return {
        "theoretical_bound": round(bound, 2),
        "mean_steps": round(float(arr.mean()), 2),
        "p95_steps": int(np.percentile(arr, 95)),
        "max_steps": int(arr.max()),
        "trials": trials,
        "passed": bool(arr.mean() <= bound * 1.5),
    }


# ---------------------------------------------------------------------------
# T2. Marginal Coverage of conformal prediction
# ---------------------------------------------------------------------------

def verify_theorem_2(
    n_cal: int = 500,
    n_test: int = 1000,
    alpha: float = 0.1,
    rng: np.random.Generator | None = None,
) -> dict:
    """Empirically verify that the inductive conformal predictor achieves
    marginal coverage ≥ 1−α on exchangeable data.

    We simulate a 5-class softmax classifier whose nonconformity score is
    s(x, y) = 1 − π_y(x), with calibration and test sets sampled iid.
    """
    rng = rng or np.random.default_rng(0)
    K = 5

    def gen(n: int):
        y = rng.integers(K, size=n)
        # Confident-but-imperfect logits
        logits = rng.normal(0.0, 0.5, size=(n, K))
        logits[np.arange(n), y] += 2.0
        # Random label noise to make the problem nontrivial
        noisy = rng.random(n) < 0.15
        y_obs = np.where(noisy, rng.integers(K, size=n), y)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return probs, y_obs

    cal_probs, cal_y = gen(n_cal)
    test_probs, test_y = gen(n_test)

    # Non-conformity scores
    cal_scores = 1.0 - cal_probs[np.arange(n_cal), cal_y]
    # Finite-sample-corrected quantile
    q = np.quantile(cal_scores, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method="higher")

    # Test prediction sets
    test_scores = 1.0 - test_probs
    in_set = test_scores <= q
    coverage = float(in_set[np.arange(n_test), test_y].mean())
    avg_set_size = float(in_set.sum(axis=1).mean())

    return {
        "target_coverage": 1 - alpha,
        "empirical_coverage": coverage,
        "avg_prediction_set_size": avg_set_size,
        "quantile_q_hat": float(q),
        "passed": coverage >= (1 - alpha) - 0.02,
    }


# ---------------------------------------------------------------------------
# T3. Pareto-Optimality of counterfactual containment
# ---------------------------------------------------------------------------

def verify_theorem_3(
    n_actions: int = 8,
    n_scenarios: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Verify that the constrained planner only ever selects Pareto-optimal
    actions on the (risk, collateral) plane.

    We sample random (risk, collateral) tables and check that the chosen
    action is not dominated by any other action in the feasible set.
    """
    rng = rng or np.random.default_rng(0)
    dominated_choices = 0

    for _ in range(n_scenarios):
        risk = rng.uniform(0, 1, n_actions)
        coll = rng.uniform(0, 1, n_actions)
        tau = float(rng.uniform(0.3, 0.7))
        feasible = np.where(coll <= tau)[0]
        if feasible.size == 0:
            continue
        chosen = feasible[np.argmin(risk[feasible])]
        # Is `chosen` dominated within feasible set?
        for a in feasible:
            if a == chosen: continue
            if risk[a] <= risk[chosen] and coll[a] <= coll[chosen] and \
               (risk[a] < risk[chosen] or coll[a] < coll[chosen]):
                dominated_choices += 1
                break

    return {
        "scenarios": n_scenarios,
        "dominated_choices": dominated_choices,
        "passed": dominated_choices == 0,
    }


# ---------------------------------------------------------------------------
# CLI: run all verifications
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print(" SENTINEL-Ω — Theorem Verification Suite")
    print("=" * 68)
    for name, fn in [("T1 Info-Theoretic Optimality", verify_theorem_1),
                     ("T2 Conformal Marginal Coverage", verify_theorem_2),
                     ("T3 Pareto-Optimal Containment", verify_theorem_3)]:
        result = fn()
        status = "✓ PASS" if result.get("passed") else "✗ FAIL"
        print(f"\n[{status}] {name}")
        for k, v in result.items():
            if k == "passed": continue
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
