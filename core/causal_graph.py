"""
Causal Hypothesis Graphs (CHGs)
================================

A CHG is a directed acyclic graph (V, E) where:
  - V is a set of *events* (alerts, log lines, observed entities)
  - E ⊆ V × V is a set of *causal hypotheses* — directed edges
    e = (u, v) read as "u causally precedes v in the attack chain"

Each edge e carries:
  - p(e) ∈ [0, 1]    : marginal probability the causal link is real
  - label : str       : MITRE ATT&CK tactic (e.g. "T1078: Valid Accounts")

Each node carries:
  - kind : Literal["alert", "entity", "action"]
  - features : dict   : raw observations
  - severity : float  : node-local severity prior in [0, 1]

We represent a *hypothesis* h ∈ H as an induced sub-DAG of the full CHG
that constitutes a coherent attack story. The belief over H is a finite
mixture of weighted hypotheses (see `belief_state.py`).
"""

from __future__ import annotations
import math
import itertools
from dataclasses import dataclass, field
from typing import Iterable
import networkx as nx


# MITRE ATT&CK tactics relevant to enterprise attacks
ATTACK_TACTICS = [
    "T1078: Valid Accounts",
    "T1110: Brute Force",
    "T1059: Command and Scripting",
    "T1071: C2 over Application Layer",
    "T1041: Exfiltration over C2",
    "T1486: Data Encrypted for Impact",
    "T1003: OS Credential Dumping",
    "T1021: Remote Services",
    "T1547: Boot/Logon Autostart",
    "T1133: External Remote Services",
    "T1190: Exploit Public-Facing App",
    "T1056: Input Capture",
    "T1218: System Binary Proxy Exec",
    "T1027: Obfuscated Files or Info",
]


@dataclass
class Node:
    id: str
    kind: str                      # "alert" | "entity" | "action"
    features: dict = field(default_factory=dict)
    severity: float = 0.0

    def __hash__(self):
        return hash(self.id)


@dataclass
class Edge:
    src: str
    dst: str
    p: float                       # marginal causal probability ∈ [0,1]
    tactic: str = ""               # MITRE ATT&CK tactic label

    def __post_init__(self):
        self.p = float(max(0.0, min(1.0, self.p)))


class CausalHypothesisGraph:
    """A weighted DAG of causal hypotheses over observed events."""

    def __init__(self):
        self._g = nx.DiGraph()
        self._nodes: dict[str, Node] = {}

    # ----- construction --------------------------------------------------
    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node
        self._g.add_node(node.id)

    def add_edge(self, edge: Edge) -> None:
        if edge.src not in self._nodes or edge.dst not in self._nodes:
            raise KeyError("Both endpoints must be added first")
        # Prevent cycles
        self._g.add_edge(edge.src, edge.dst, p=edge.p, tactic=edge.tactic)
        if not nx.is_directed_acyclic_graph(self._g):
            self._g.remove_edge(edge.src, edge.dst)
            raise ValueError("Edge would introduce a cycle in the CHG")

    # ----- access --------------------------------------------------------
    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[Edge]:
        return [Edge(u, v, d["p"], d.get("tactic", ""))
                for u, v, d in self._g.edges(data=True)]

    def get_node(self, nid: str) -> Node:
        return self._nodes[nid]

    def parents(self, nid: str) -> list[str]:
        return list(self._g.predecessors(nid))

    def children(self, nid: str) -> list[str]:
        return list(self._g.successors(nid))

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._g))

    # ----- semantics: hypothesis enumeration ----------------------------
    def joint_probability(self, edge_subset: Iterable[tuple[str, str]]) -> float:
        """Joint probability of a particular subset of edges being 'real'
        and all others being absent, assuming edge-wise independence.

        P(E_subset) = ∏_{e ∈ subset} p(e) · ∏_{e ∉ subset} (1 − p(e))
        Log-space for numerical stability.
        """
        edge_subset_set = set(edge_subset)
        log_p = 0.0
        for u, v, d in self._g.edges(data=True):
            p_e = d["p"]
            if (u, v) in edge_subset_set:
                log_p += math.log(max(p_e, 1e-12))
            else:
                log_p += math.log(max(1.0 - p_e, 1e-12))
        return math.exp(log_p)

    def enumerate_top_hypotheses(self, k: int = 32) -> list[tuple[float, set[tuple[str, str]]]]:
        """Return the k most likely hypotheses (edge subsets) by joint
        probability. Uses greedy expansion when the full power set is too
        large to enumerate.
        """
        edges = [(u, v) for u, v in self._g.edges()]
        m = len(edges)

        if m <= 16:  # 2^16 = 65k, manageable
            scored = []
            for r in range(0, m + 1):
                for subset in itertools.combinations(edges, r):
                    p = self.joint_probability(subset)
                    scored.append((p, set(subset)))
            scored.sort(reverse=True, key=lambda x: x[0])
            return scored[:k]

        # Greedy + perturbation for larger graphs
        # Start with each edge independently kept if p(e) > 0.5
        base = {e for e in edges if self._g.edges[e]["p"] > 0.5}
        scored = [(self.joint_probability(base), base)]
        for e in edges:
            alt = base ^ {e}
            scored.append((self.joint_probability(alt), alt))
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:k]

    # ----- updates -------------------------------------------------------
    def update_edge_probability(self, src: str, dst: str, new_p: float) -> None:
        if self._g.has_edge(src, dst):
            self._g.edges[src, dst]["p"] = float(max(0.0, min(1.0, new_p)))

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n.id, "kind": n.kind, "severity": n.severity,
                 "features": n.features}
                for n in self._nodes.values()
            ],
            "edges": [
                {"src": u, "dst": v, "p": d["p"], "tactic": d.get("tactic", "")}
                for u, v, d in self._g.edges(data=True)
            ],
        }


def chg_from_alert(alert: dict) -> CausalHypothesisGraph:
    """Construct an initial CHG from a single alert. Adds a root node
    plus candidate hypothesis nodes for likely upstream/downstream events.
    The LLM is later asked to refine probabilities and propose edges.
    """
    g = CausalHypothesisGraph()
    root = Node(
        id=alert["id"],
        kind="alert",
        features=alert.get("details", {}),
        severity=_severity_to_float(alert.get("severity", "medium")),
    )
    g.add_node(root)

    # Generic upstream hypotheses (what led to the alert)
    for upstream_id, upstream_kind in [
        ("init_access", "action"),
        ("credential_compromise", "action"),
        ("recon", "action"),
    ]:
        g.add_node(Node(id=upstream_id, kind=upstream_kind, severity=0.5))
        g.add_edge(Edge(upstream_id, root.id, p=0.3, tactic=""))

    # Generic downstream hypotheses (what might come next)
    for downstream_id in ["lateral_movement", "exfiltration", "persistence"]:
        g.add_node(Node(id=downstream_id, kind="action", severity=0.6))
        g.add_edge(Edge(root.id, downstream_id, p=0.2, tactic=""))

    return g


def _severity_to_float(s: str) -> float:
    return {"info": 0.1, "low": 0.3, "medium": 0.5, "high": 0.75, "critical": 0.95}.get(s, 0.5)
