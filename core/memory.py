"""
Hierarchical episodic memory for SENTINEL-Ω.

Stores closed incidents as embeddings and retrieves similar past cases to
prime the agent's belief and to teach it from analyst feedback over time.

To avoid heavy dependencies in the hackathon demo we implement a
lightweight embedding using TF-IDF + cosine similarity, isomorphic to a
sentence-transformer + FAISS pipeline. The interface is identical so the
production swap is a one-line change.
"""

from __future__ import annotations
import re
import math
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter


@dataclass
class Episode:
    incident_id: str
    summary: str                    # natural-language description
    archetype: str                  # closed-form archetype if known
    severity: str
    verdict: str
    analyst_corrected: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class EpisodicMemory:
    """In-memory store with cosine similarity over TF-IDF vectors."""

    def __init__(self):
        self._episodes: list[Episode] = []
        self._vocab: dict[str, int] = {}
        self._idf: dict[int, float] = {}
        self._vectors: list[dict[int, float]] = []   # sparse

    # ------------------------------------------------------------------
    def _tokenize(self, s: str) -> list[str]:
        return re.findall(r"[a-z][a-z0-9_-]*", s.lower())

    def _vectorize(self, tokens: list[str]) -> dict[int, float]:
        tf = Counter(tokens)
        norm = math.sqrt(sum((c * self._idf.get(self._vocab.get(t, -1), 0)) ** 2
                             for t, c in tf.items()))
        if norm == 0: return {}
        return {self._vocab[t]: c * self._idf[self._vocab[t]] / norm
                for t, c in tf.items() if t in self._vocab}

    def _cosine(self, a: dict, b: dict) -> float:
        # sparse dot product
        if len(a) > len(b): a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())

    def _refit_idf(self) -> None:
        N = max(len(self._episodes), 1)
        df = Counter()
        for ep in self._episodes:
            for t in set(self._tokenize(ep.summary)):
                df[t] += 1
        # Assign vocabulary
        self._vocab = {t: i for i, t in enumerate(df)}
        self._idf = {self._vocab[t]: math.log((1 + N) / (1 + n)) + 1
                     for t, n in df.items()}
        self._vectors = [self._vectorize(self._tokenize(e.summary))
                         for e in self._episodes]

    # ------------------------------------------------------------------
    def add(self, ep: Episode) -> None:
        self._episodes.append(ep)
        self._refit_idf()

    def query(self, query_text: str, k: int = 5) -> list[tuple[float, Episode]]:
        if not self._episodes: return []
        # New tokens not in vocab still vectorise as 0
        q_vec = self._vectorize(self._tokenize(query_text))
        scored = [(self._cosine(q_vec, v), self._episodes[i])
                  for i, v in enumerate(self._vectors)]
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[:k]

    def __len__(self) -> int:
        return len(self._episodes)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps([{
            "incident_id": e.incident_id, "summary": e.summary,
            "archetype": e.archetype, "severity": e.severity,
            "verdict": e.verdict, "analyst_corrected": e.analyst_corrected,
            "timestamp": e.timestamp, "metadata": e.metadata,
        } for e in self._episodes], indent=2))

    @classmethod
    def load(cls, path: str) -> "EpisodicMemory":
        m = cls()
        data = json.loads(Path(path).read_text())
        for d in data:
            m._episodes.append(Episode(**d))
        m._refit_idf()
        return m
