"""
Bayesian Optimal Experimental Design for SOC investigation.

At each step the agent must choose which tool to invoke (e.g. lookup IP,
search logs, get user context). We frame this as an *optimal experimental
design* problem: pick the action whose observation maximally reduces the
expected entropy of the belief state.

  a* = argmax_a   H(b) − E_{o ~ p(o|a,b)} [ H(b | o, a) ]
     = argmax_a   I(H ; O | a)            (mutual information)

Because the observation space O is large and partially structured, we
estimate the expected posterior entropy via Monte-Carlo sampling: for each
candidate action a, sample synthetic observations o ~ p(o|h,a) under each
weighted hypothesis h, compute the implied posterior entropy, and average.

The computational cost is O(|A| · N · K) where N is the number of belief
particles and K is the number of sampled observations per action.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable

from .belief_state import BeliefState


@dataclass
class CandidateAction:
    name: str
    arg_summary: str
    cost: float = 1.0               # latency/cost penalty for this action
    # likelihood model: P(observation_bucket | hypothesis)
    obs_model: Callable[[object], np.ndarray] | None = None
    obs_labels: list[str] = None    # discrete observation buckets


def expected_information_gain(
    belief: BeliefState,
    action: CandidateAction,
    K: int = 16,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate I(H ; O | a) under the current belief.

    For each particle i (with weight w_i), draw simulated observation buckets
    from the per-hypothesis distribution `action.obs_model(h_i)`, compute the
    implied posterior, and accumulate the entropy reduction.
    """
    rng = rng or np.random.default_rng()
    if action.obs_model is None or not action.obs_labels:
        return 0.0

    H_prior = belief.entropy()
    weights = belief.weights
    # Predictive distribution over observations
    obs_dists = np.stack([action.obs_model(p.hypothesis) for p in belief.particles])
    # Marginal: p(o) = ∑ w_i · p(o | h_i)
    marg = (weights[:, None] * obs_dists).sum(axis=0)
    marg = marg / max(marg.sum(), 1e-12)

    # Expected posterior entropy
    H_post = 0.0
    for o_idx, p_o in enumerate(marg):
        if p_o < 1e-12:
            continue
        post = weights * obs_dists[:, o_idx]
        post = post / post.sum()
        post = post[post > 1e-12]
        H_o = float(-(post * np.log(post)).sum())
        H_post += p_o * H_o

    return float(H_prior - H_post)


def select_action(
    belief: BeliefState,
    actions: list[CandidateAction],
    cost_weight: float = 0.1,
) -> tuple[CandidateAction, dict]:
    """Choose the action with the highest cost-adjusted expected info gain.

    Objective:  EIG(a) − λ · cost(a)

    Returns the chosen action and a diagnostic dict of all scores.
    """
    if not actions:
        raise ValueError("Need at least one candidate action")

    diagnostics = {}
    best_a, best_score = None, -math.inf
    for a in actions:
        eig = expected_information_gain(belief, a)
        score = eig - cost_weight * a.cost
        diagnostics[a.name] = {"eig": round(eig, 4), "cost": a.cost,
                               "score": round(score, 4)}
        if score > best_score:
            best_score, best_a = score, a
    return best_a, diagnostics
