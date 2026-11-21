# I am the retrieval agent: I rank documents for a query with BM25 and can skip sources the user rejected.
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .corpus import Document
from .text import tokenize


@dataclass(frozen=True)
class Hit:
    doc: Document
    score: float


class BM25Retriever:
    """Plain BM25 with no external dependencies, so the prototype runs anywhere."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(d.title + " " + d.text)) for d in docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(len(self._len), 1)
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @property
    def idf(self) -> dict[str, float]:
        return dict(self._idf)

    def search(self, query: str, k: int = 5, exclude: set[str] | None = None) -> list[Hit]:
        exclude = exclude or set()
        terms = tokenize(query)
        hits = []
        for doc, tf, length in zip(self.docs, self._tf, self._len):
            if doc.doc_id in exclude:
                continue
            score = 0.0
            for t in terms:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * length / self._avg)
                score += self._idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            if score > 0:
                hits.append(Hit(doc, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
