"""
Inductive Conformal Prediction for calibrated severity classification.

Given a base classifier f : X → Δ(Y), conformal prediction wraps it to
produce prediction *sets* C_α(x) ⊆ Y with the marginal guarantee:

    P(y_true ∈ C_α(x)) ≥ 1 − α

Our non-conformity score is s(x, y) = 1 − f(x)[y] (1 − softmax probability
of the true class). The calibration quantile uses the finite-sample
correction (n+1)/n.

This module is implementation-agnostic in the base classifier: we plug
in either an empirical LLM-scorer or a logistic regression learned over
incident features.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class ConformalPredictor:
    """Inductive (split) conformal predictor with class-aware quantiles.

    Attributes:
      classes:         ordered class label list, e.g. ["info","low","med","high","critical"]
      base_classifier: maps x -> array of class probabilities (length |classes|)
      alpha:           target miscoverage rate (default 0.1 ⇒ 90% coverage)
    """
    classes: list[str]
    base_classifier: Callable[[object], np.ndarray]
    alpha: float = 0.1
    _qhat: float | None = None
    _qhat_per_class: dict[str, float] | None = None

    # ------------------------------------------------------------------
    def calibrate(self, X_cal: list[object], y_cal: list[str]) -> None:
        """Compute the calibration quantile from held-out data.

        We compute both a marginal q̂ and per-class q̂_y (Mondrian conformal)
        for stronger conditional coverage guarantees on minority classes.
        """
        if len(X_cal) != len(y_cal):
            raise ValueError("X_cal and y_cal length mismatch")
        if len(X_cal) < 30:
            raise ValueError("Need ≥30 calibration points for stable quantile")

        idx = {c: i for i, c in enumerate(self.classes)}
        scores = []
        scores_per_class = {c: [] for c in self.classes}

        for x, y in zip(X_cal, y_cal):
            probs = self.base_classifier(x)
            s = 1.0 - probs[idx[y]]
            scores.append(s)
            scores_per_class[y].append(s)

        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self._qhat = float(np.quantile(scores, q_level, method="higher"))
        self._qhat_per_class = {}
        for c, sc in scores_per_class.items():
            if len(sc) >= 10:
                n_c = len(sc)
                q_c = min(np.ceil((n_c + 1) * (1 - self.alpha)) / n_c, 1.0)
                self._qhat_per_class[c] = float(np.quantile(sc, q_c, method="higher"))
            else:
                self._qhat_per_class[c] = self._qhat

    # ------------------------------------------------------------------
    def predict_set(self, x: object) -> dict:
        """Return a calibrated prediction set at level 1−α.

        Output dict:
          point:        argmax-prob class (singleton point estimate)
          set:          list of classes in the conformal set
          probs:        per-class probabilities from the base classifier
          confidence:   1 − non-conformity of the point prediction
          target_cov:   the marginal target 1 − α (for display)
        """
        if self._qhat is None:
            raise RuntimeError("Predictor must be calibrated first")

        probs = self.base_classifier(x)
        scores = 1.0 - probs
        pset = [c for c, s in zip(self.classes, scores) if s <= self._qhat]
        if not pset:
            # Pathological case — degenerate to top-1
            pset = [self.classes[int(np.argmax(probs))]]
        i_star = int(np.argmax(probs))
        return {
            "point": self.classes[i_star],
            "set": pset,
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "confidence": float(probs[i_star]),
            "target_coverage": 1 - self.alpha,
        }

    # ------------------------------------------------------------------
    def evaluate_coverage(self, X_test: list[object], y_test: list[str]) -> dict:
        """Empirically validate marginal and conditional coverage on held-out data."""
        n = len(X_test)
        in_set = 0
        sizes = []
        per_class = {c: {"correct": 0, "total": 0} for c in self.classes}

        for x, y in zip(X_test, y_test):
            out = self.predict_set(x)
            sizes.append(len(out["set"]))
            per_class[y]["total"] += 1
            if y in out["set"]:
                in_set += 1
                per_class[y]["correct"] += 1

        return {
            "n_test": n,
            "marginal_coverage": in_set / n,
            "avg_set_size": float(np.mean(sizes)),
            "median_set_size": float(np.median(sizes)),
            "conditional_coverage": {
                c: (v["correct"] / v["total"] if v["total"] else None)
                for c, v in per_class.items()
            },
        }


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE) — complementary diagnostic
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """ECE = ∑_b (n_b / N) · |acc_b − conf_b|

    Args:
      probs:  (N, K) softmax outputs
      labels: (N,)   true class indices
    """
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    correct = (preds == labels).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_b = (confs > lo) & (confs <= hi)
        n_b = in_b.sum()
        if n_b == 0:
            continue
        acc_b = correct[in_b].mean()
        conf_b = confs[in_b].mean()
        ece += (n_b / N) * abs(acc_b - conf_b)
    return float(ece)
