"""
Bayesian belief state over causal hypotheses.

We use a particle approximation: belief b_t is represented as a finite set
{(h_i, w_i)}_{i=1..N} where h_i is a hypothesis (an edge subset of a CHG)
and w_i is its posterior weight, ∑ w_i = 1.

Updates use log-weights for numerical stability:

    log w_i ← log w_i + log P(o_t | h_i, a_t) − log Z

Resampling is triggered when the effective sample size (ESS) falls below
N/2, following standard sequential Monte Carlo practice.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Particle:
    hypothesis: object              # opaque (set of edges or CHG slice)
    log_weight: float = 0.0
    metadata: dict = field(default_factory=dict)


class BeliefState:
    """A particle filter over hypotheses."""

    def __init__(self, particles: list[Particle]):
        if not particles:
            raise ValueError("BeliefState needs at least one particle")
        self.particles = particles
        self._normalise()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def _normalise(self) -> None:
        log_ws = np.array([p.log_weight for p in self.particles])
        # log-sum-exp
        m = log_ws.max()
        log_Z = m + math.log(np.exp(log_ws - m).sum())
        for p in self.particles:
            p.log_weight -= log_Z

    @property
    def weights(self) -> np.ndarray:
        return np.array([math.exp(p.log_weight) for p in self.particles])

    def entropy(self, base: float = math.e) -> float:
        """Shannon entropy of the belief distribution.

        H(b) = −∑ w_i log w_i
        """
        w = self.weights
        w = w[w > 1e-12]
        return float(-np.sum(w * np.log(w)) / np.log(base))

    def map_hypothesis(self) -> Particle:
        """Maximum a posteriori hypothesis."""
        return max(self.particles, key=lambda p: p.log_weight)

    def effective_sample_size(self) -> float:
        """ESS = 1 / ∑ w_i^2 — diagnoses particle degeneracy."""
        w = self.weights
        return float(1.0 / np.sum(w * w))

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------
    def update(self, log_likelihood: Callable[[object], float]) -> None:
        """Apply Bayes' rule: log w_i ← log w_i + log P(o | h_i)."""
        for p in self.particles:
            p.log_weight += log_likelihood(p.hypothesis)
        self._normalise()
        if self.effective_sample_size() < len(self.particles) / 2:
            self._resample()

    def _resample(self) -> None:
        """Systematic resampling — variance-reducing alternative to multinomial."""
        N = len(self.particles)
        w = self.weights
        positions = (np.arange(N) + np.random.random()) / N
        cumulative = np.cumsum(w)
        new_particles = []
        i, j = 0, 0
        while i < N and j < N:
            if positions[i] < cumulative[j]:
                new_particles.append(Particle(
                    hypothesis=self.particles[j].hypothesis,
                    log_weight=0.0,  # uniform after resample
                    metadata=dict(self.particles[j].metadata),
                ))
                i += 1
            else:
                j += 1
        # pad if necessary
        while len(new_particles) < N:
            new_particles.append(Particle(
                hypothesis=self.particles[-1].hypothesis, log_weight=0.0))
        self.particles = new_particles
        self._normalise()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self, top_k: int = 5) -> list[dict]:
        """Return the top-k hypotheses by weight, for dashboards/logs."""
        idx = np.argsort([p.log_weight for p in self.particles])[::-1][:top_k]
        return [
            {"weight": float(math.exp(self.particles[i].log_weight)),
             "hypothesis": self.particles[i].hypothesis,
             "metadata": self.particles[i].metadata}
            for i in idx
        ]


def uniform_belief(hypotheses: list[object]) -> BeliefState:
    """Convenience constructor: uniform prior over a list of hypotheses."""
    return BeliefState([Particle(h, log_weight=0.0) for h in hypotheses])
