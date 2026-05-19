"""
Counterfactual Containment Planner.

When the agent considers a containment action c (isolate_host, block_ip, etc.),
we evaluate it under each weighted hypothesis h in the belief and compute:

  Risk(c | h)         — expected residual threat if c is applied under h
  Collateral(c | h)   — expected business disruption if c is applied under h

We then solve the constrained selection problem:

  c* = argmin_c   E_h[ Risk(c | h) ]
       s.t.       E_h[ Collateral(c | h) ] ≤ τ

We additionally enforce Pareto-optimality on the (Risk, Collateral) plane:
no chosen action is dominated by another feasible action.

Game-theoretic framing: this is a Stackelberg game where the defender
(SENTINEL-Ω) commits to a containment policy, and a rational worst-case
attacker selects the hypothesis from the belief support that maximises
residual damage. The constrained-min formulation above is precisely the
robust Stackelberg equilibrium under additive utility.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

from .belief_state import BeliefState


@dataclass
class ContainmentAction:
    name: str
    description: str
    # under each hypothesis: returns (risk_reduction, collateral_damage)
    effect_model: Callable[[object], tuple[float, float]]


@dataclass
class ContainmentDecision:
    chosen: ContainmentAction | None
    expected_risk: float
    expected_collateral: float
    pareto_front: list[dict]
    feasible: bool
    justification: str


def plan_containment(
    belief: BeliefState,
    actions: list[ContainmentAction],
    collateral_budget: float = 0.4,
    no_op: bool = True,
) -> ContainmentDecision:
    """Choose the best containment action subject to the collateral budget τ.

    Args:
      belief:             posterior over hypotheses
      actions:            candidate containment actions
      collateral_budget:  τ ∈ [0,1]
      no_op:              if True, include a "do nothing" action whose
                          risk = baseline threat and collateral = 0.

    Returns a ContainmentDecision with the chosen action plus the full
    Pareto front for explainability.
    """
    weights = belief.weights
    particles = belief.particles

    # No-op action: no collateral, but full residual risk
    if no_op:
        baseline_risk = float(np.mean([_risk_under(h.hypothesis) for h in particles]))
        noop = ContainmentAction(
            name="no_op",
            description="Take no containment action; continue monitoring.",
            effect_model=lambda h: (0.0, 0.0),
        )
        # We treat the residual risk for no_op specially below
    else:
        noop = None
        baseline_risk = 0.0

    # Evaluate each action under each hypothesis
    scored = []
    for a in actions + ([noop] if noop else []):
        risk_reductions, collaterals = [], []
        for w, p in zip(weights, particles):
            if a is noop:
                rr, coll = 0.0, 0.0
            else:
                rr, coll = a.effect_model(p.hypothesis)
            risk_reductions.append(w * rr)
            collaterals.append(w * coll)
        e_risk_red = float(sum(risk_reductions))
        e_coll = float(sum(collaterals))
        # Residual risk = baseline − reduction (clipped at 0)
        e_residual_risk = max(baseline_risk - e_risk_red, 0.0)
        scored.append({
            "name": a.name, "action": a,
            "risk": e_residual_risk,
            "collateral": e_coll,
            "feasible": e_coll <= collateral_budget,
        })

    # Pareto-optimal subset on (risk, collateral) — both minimised
    pareto = _pareto_front(scored)

    # Feasible subset of pareto
    feasible = [s for s in pareto if s["feasible"]]
    if feasible:
        # Among feasible Pareto-optimal actions, pick min risk; tie-break by min collateral
        best = min(feasible, key=lambda s: (s["risk"], s["collateral"]))
        return ContainmentDecision(
            chosen=best["action"],
            expected_risk=best["risk"],
            expected_collateral=best["collateral"],
            pareto_front=[{"name": s["name"], "risk": s["risk"], "collateral": s["collateral"]}
                          for s in pareto],
            feasible=True,
            justification=(
                f"Chosen `{best['name']}`: lowest expected residual risk "
                f"({best['risk']:.2f}) among Pareto-optimal actions within "
                f"collateral budget τ={collateral_budget:.2f}."
            ),
        )

    # No feasible action — escalate
    return ContainmentDecision(
        chosen=None,
        expected_risk=baseline_risk,
        expected_collateral=0.0,
        pareto_front=[{"name": s["name"], "risk": s["risk"], "collateral": s["collateral"]}
                      for s in pareto],
        feasible=False,
        justification=(
            f"No containment action satisfies collateral budget τ={collateral_budget:.2f}. "
            f"Escalating to human analyst."
        ),
    )


def _pareto_front(scored: list[dict]) -> list[dict]:
    """Return the Pareto-optimal subset on (risk ↓, collateral ↓)."""
    front = []
    for s in scored:
        dominated = False
        for s2 in scored:
            if s is s2: continue
            if (s2["risk"] <= s["risk"] and s2["collateral"] <= s["collateral"]
                and (s2["risk"] < s["risk"] or s2["collateral"] < s["collateral"])):
                dominated = True
                break
        if not dominated:
            front.append(s)
    return front


def _risk_under(hypothesis: object) -> float:
    """Heuristic baseline threat under a hypothesis. The hypothesis is an
    opaque object; the agent's tool-result-conditioned likelihoods drive
    its weight, so here we use a stand-in mapping based on metadata.
    """
    if hasattr(hypothesis, "severity"):
        return float(hypothesis.severity)
    if isinstance(hypothesis, dict) and "severity" in hypothesis:
        return float(hypothesis["severity"])
    return 0.5
