"""
Evaluation metrics for SENTINEL-Ω.

We report:
  - Triage F1 / Precision / Recall (binary threat vs benign)
  - Severity classification accuracy and per-class F1
  - Expected Calibration Error (ECE) and Brier score
  - Conformal coverage at level 1−α
  - Unnecessary-containment rate
  - Mean time-to-decision (in agent turns)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


CLASSES = ["info", "low", "medium", "high", "critical"]
SEVERITY_IDX = {c: i for i, c in enumerate(CLASSES)}


def binary_f1(y_true: list[bool], y_pred: list[bool]) -> dict:
    """Precision, recall, F1 over the binary `is_threat` label."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 3),
            "recall":    round(rec, 3),
            "f1":        round(f1, 3)}


def severity_accuracy(y_true: list[str], y_pred: list[str]) -> dict:
    """Multiclass accuracy and macro-F1 over severity levels."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / max(len(y_true), 1)

    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for t, p in zip(y_true, y_pred):
        for c in CLASSES:
            if t == c and p == c: per_class[c]["tp"] += 1
            elif t != c and p == c: per_class[c]["fp"] += 1
            elif t == c and p != c: per_class[c]["fn"] += 1

    macro_f1 = []
    for c in CLASSES:
        d = per_class[c]
        prec = d["tp"] / max(d["tp"] + d["fp"], 1)
        rec  = d["tp"] / max(d["tp"] + d["fn"], 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-12)
        macro_f1.append(f1)
    return {"accuracy": round(acc, 3),
            "macro_f1": round(float(np.mean(macro_f1)), 3)}


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """ECE = ∑_b (n_b / N) · |acc_b − conf_b|.

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


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier: mean squared error between one-hot label and probs."""
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (confidences, accuracies, counts) per bin — for plotting."""
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    correct = (preds == labels).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_b = (confs > lo) & (confs <= hi)
        n_b = int(in_b.sum())
        counts.append(n_b)
        if n_b == 0:
            centers.append((lo + hi) / 2); accs.append(np.nan)
        else:
            centers.append(float(confs[in_b].mean()))
            accs.append(float(correct[in_b].mean()))
    return np.array(centers), np.array(accs), np.array(counts)


def unnecessary_containment_rate(
    y_threat_true: list[bool], containment_taken: list[bool]
) -> float:
    """Fraction of benign scenarios on which any containment action was taken."""
    benign = [i for i, t in enumerate(y_threat_true) if not t]
    if not benign:
        return 0.0
    return sum(containment_taken[i] for i in benign) / len(benign)


@dataclass
class EvaluationReport:
    system: str
    triage: dict
    severity: dict
    ece: float
    brier: float
    unnecessary_containment: float
    conformal_coverage: float | None
    mean_turns: float
    n: int

    def as_row(self) -> dict:
        return {
            "system": self.system,
            "F1": self.triage["f1"],
            "Precision": self.triage["precision"],
            "Recall": self.triage["recall"],
            "Severity Acc": self.severity["accuracy"],
            "Macro-F1": self.severity["macro_f1"],
            "ECE ↓": round(self.ece, 3),
            "Brier ↓": round(self.brier, 3),
            "Unnecessary Contain ↓": round(self.unnecessary_containment, 3),
            "Conformal Cov.": round(self.conformal_coverage, 3) if self.conformal_coverage is not None else "—",
            "Mean Turns": round(self.mean_turns, 2),
            "n": self.n,
        }
