"""
Full evaluation: run every system on CHIMERA, compute metrics, produce
LaTeX-style tables and reliability/ROC/PR figures.

Usage:
  python -m experiments.results [--n 1200] [--seed 42]
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from eval.chimera import generate
from eval.metrics import (
    binary_f1, severity_accuracy, expected_calibration_error, brier_score,
    reliability_curve, unnecessary_containment_rate, EvaluationReport, CLASSES, SEVERITY_IDX
)
from eval.baselines import RuleBasedSOAR, VanillaLLM, ReActAgent
from eval.sentinel_sim import SentinelOmega


FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)


def run_one(system, scenarios) -> EvaluationReport:
    y_threat_true, y_threat_pred = [], []
    y_sev_true, y_sev_pred = [], []
    probs_list, label_idx = [], []
    contained = []
    turns = []

    for s in scenarios:
        gt = s.ground_truth
        out = system.triage({"alerts": s.alerts})
        y_threat_true.append(gt["is_threat"])
        y_threat_pred.append(out.is_threat)
        y_sev_true.append(gt["true_severity"])
        y_sev_pred.append(out.severity)
        probs_list.append([out.severity_probs[c] for c in CLASSES])
        label_idx.append(SEVERITY_IDX[gt["true_severity"]])
        contained.append(out.contained)
        turns.append(out.turns)

    probs = np.array(probs_list)
    labels = np.array(label_idx)

    return EvaluationReport(
        system=system.name,
        triage=binary_f1(y_threat_true, y_threat_pred),
        severity=severity_accuracy(y_sev_true, y_sev_pred),
        ece=expected_calibration_error(probs, labels),
        brier=brier_score(probs, labels),
        unnecessary_containment=unnecessary_containment_rate(y_threat_true, contained),
        conformal_coverage=_conformal_coverage(probs, labels) if system.name == "sentinel_omega" else None,
        mean_turns=float(np.mean(turns)),
        n=len(scenarios),
    )


def _conformal_coverage(probs: np.ndarray, labels: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal coverage measured on a held-out half of the test set."""
    n = len(labels)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    cal, test = idx[:n // 2], idx[n // 2:]
    scores = 1.0 - probs[cal, labels[cal]]
    q = np.quantile(scores, np.ceil((len(cal) + 1) * (1 - alpha)) / len(cal), method="higher")
    # Test coverage
    test_scores = 1.0 - probs[test]
    in_set = test_scores <= q
    return float(in_set[np.arange(len(test)), labels[test]].mean())


def plot_reliability(reports_with_data, out_path):
    """Reliability curves comparing every system."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Perfect calibration")
    colors = {"rule_based_soar":"#888888", "vanilla_llm":"#d97706",
              "react_agent":"#0ea5e9", "sentinel_omega":"#16a34a"}
    for name, probs, labels in reports_with_data:
        centers, accs, counts = reliability_curve(probs, labels)
        mask = ~np.isnan(accs)
        ax.plot(centers[mask], accs[mask], "o-", color=colors.get(name, "k"),
                label=name, lw=2, markersize=5)
    ax.set_xlabel("Mean confidence in bin"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram — CHIMERA test split")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def plot_bar(reports, metric_key, ylabel, out_path, lower_is_better=False):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    names = [r.system for r in reports]
    vals = []
    for r in reports:
        row = r.as_row()
        v = row[metric_key]
        v = float(v) if v != "—" else 0.0
        vals.append(v)
    colors = ["#888888","#d97706","#0ea5e9","#16a34a"][:len(names)]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(ylabel); ax.set_title(metric_key)
    if lower_is_better:
        ax.set_title(metric_key + "  (lower is better)")
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def render_table(reports):
    """Render the headline results table both as Markdown and LaTeX."""
    rows = [r.as_row() for r in reports]
    headers = list(rows[0].keys())

    # Markdown
    md = "| " + " | ".join(headers) + " |\n"
    md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for row in rows:
        md += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"

    # LaTeX
    tex = "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\n\\toprule\n"
    tex += " & ".join(headers) + " \\\\\n\\midrule\n"
    for row in rows:
        tex += " & ".join(str(row[h]) for h in headers) + " \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    return md, tex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n[1/4] Generating CHIMERA benchmark (n={args.n}, seed={args.seed}) …")
    scenarios = generate(n=args.n, seed=args.seed)

    print("\n[2/4] Running all systems …")
    systems = [
        RuleBasedSOAR(),
        VanillaLLM(seed=args.seed),
        ReActAgent(seed=args.seed),
        SentinelOmega(seed=args.seed),
    ]
    reports = []
    raw_for_plot = []
    for sys in systems:
        rep = run_one(sys, scenarios)
        reports.append(rep)
        # Collect probs/labels for reliability plot
        probs_list, label_idx = [], []
        for s in scenarios:
            out = sys.triage({"alerts": s.alerts})
            probs_list.append([out.severity_probs[c] for c in CLASSES])
            label_idx.append(SEVERITY_IDX[s.ground_truth["true_severity"]])
        raw_for_plot.append((sys.name, np.array(probs_list), np.array(label_idx)))

    print("\n[3/4] Rendering tables …")
    md, tex = render_table(reports)
    Path("results_table.md").write_text(md)
    Path("results_table.tex").write_text(tex)
    print("\nResults (Markdown):\n")
    print(md)

    print("[4/4] Generating figures …")
    plot_reliability(raw_for_plot, FIG_DIR / "reliability.png")
    plot_bar(reports, "F1", "F1", FIG_DIR / "f1.png")
    plot_bar(reports, "ECE ↓", "ECE", FIG_DIR / "ece.png", lower_is_better=True)
    plot_bar(reports, "Unnecessary Contain ↓", "Rate",
             FIG_DIR / "unnecessary_containment.png", lower_is_better=True)

    print(f"\n  → figures/reliability.png")
    print(f"  → figures/f1.png")
    print(f"  → figures/ece.png")
    print(f"  → figures/unnecessary_containment.png")
    print(f"  → results_table.md, results_table.tex\n")

    # Also dump raw reports for downstream inspection
    Path("results_raw.json").write_text(
        json.dumps([r.as_row() for r in reports], indent=2)
    )


if __name__ == "__main__":
    main()
