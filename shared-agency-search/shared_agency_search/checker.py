# I am the evidence-checking agent: I pull claims from sources, score them, and flag disagreements and stale evidence.
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .planner import SubQuestion
from .retriever import Hit
from .text import extract_dates, extract_money, sentences, tokenize

STALE_AFTER_DAYS = 60   # older than the newest relevant source by this much = possibly outdated
STALE_PENALTY = 0.6
CONFLICT_MIN_WEIGHT = 0.12
TYPE_BONUS = 0.3        # reward sentences whose form matches the question type

# Cues for answer types: a "when" question wants a date, a "why" question wants a cause.
CAUSE_CUES = ("because", "after", "due to", "found", "caused", "following", "as a result")


@dataclass
class Claim:
    text: str
    doc_id: str
    doc_title: str
    source_type: str
    published: str
    support: float          # how well the sentence addresses the sub-question (0-1)
    credibility: float      # source prior x recency (0-1)
    stale: bool
    values: frozenset[str]  # dates / amounts the claim asserts

    @property
    def weight(self) -> float:
        return self.support * self.credibility


@dataclass
class SubAnswer:
    sub_question: SubQuestion
    claims: list[Claim]
    answer: Claim | None
    confidence: float
    conflict: bool
    competing_values: dict[str, float] = field(default_factory=dict)
    superseded_values: dict[str, float] = field(default_factory=dict)

    @property
    def unresolved(self) -> bool:
        return self.answer is None or (self.conflict and self.confidence < 0.6)


def question_type(text: str) -> str:
    first = text.strip().lower().split()[0] if text.strip() else ""
    return {"when": "date", "why": "cause", "how": "amount" if "much" in text.lower() else "other"}.get(first, "other")


class EvidenceChecker:
    def __init__(self, idf: dict[str, float] | None = None, top_k: int = 4):
        self.idf = idf or {}
        self.top_k = top_k

    def check(self, sq: SubQuestion, hits: list[Hit]) -> SubAnswer:
        hits = hits[: self.top_k]
        if not hits:
            return SubAnswer(sq, [], None, 0.0, False)
        newest = max(h.doc.published for h in hits)
        q_terms = set(tokenize(sq.text))
        q_type = question_type(sq.text)
        claims = [c for h in hits if (c := self._best_claim(h, q_terms, q_type, newest))]
        return self._resolve(sq, claims, compare_values=(q_type == "date"))

    def _support(self, q_terms: set[str], sent: str, q_type: str) -> float:
        """IDF-weighted overlap (rare words count more), plus a bonus when the sentence has the expected answer form."""
        terms = set(tokenize(sent))
        weight = lambda t: self.idf.get(t, 1.0)  # noqa: E731
        total = sum(weight(t) for t in q_terms) or 1.0
        overlap = sum(weight(t) for t in q_terms & terms) / total
        if overlap == 0:
            return 0.0
        lowered = sent.lower()
        matches_type = (
            (q_type == "date" and bool(extract_dates(sent)))
            or (q_type == "cause" and any(cue in lowered for cue in CAUSE_CUES))
            or (q_type == "amount" and bool(extract_money(sent)))
        )
        if q_type == "other":
            return overlap
        # Topic overlap gives up to (1 - TYPE_BONUS); the right answer form adds the rest.
        return (1 - TYPE_BONUS) * overlap + (TYPE_BONUS if matches_type else 0.0)

    def _best_claim(self, hit: Hit, q_terms: set[str], q_type: str, newest) -> Claim | None:
        best, best_support = None, 0.0
        if not q_terms:
            return None
        for sent in sentences(hit.doc.text):
            support = self._support(q_terms, sent, q_type)
            if support > best_support:
                best, best_support = sent, support
        if not best or best_support == 0:
            return None
        age_gap = (newest - hit.doc.published).days
        stale = age_gap > STALE_AFTER_DAYS
        credibility = hit.doc.prior * (STALE_PENALTY if stale else 1.0)
        return Claim(
            text=best,
            doc_id=hit.doc.doc_id,
            doc_title=hit.doc.title,
            source_type=hit.doc.source_type,
            published=hit.doc.published.isoformat(),
            support=round(best_support, 3),
            credibility=round(credibility, 3),
            stale=stale,
            values=frozenset(extract_dates(best) | extract_money(best)),
        )

    @staticmethod
    def _resolve(sq: SubQuestion, claims: list[Claim], compare_values: bool) -> SubAnswer:
        if not claims:
            return SubAnswer(sq, [], None, 0.0, False)
        claims.sort(key=lambda c: c.weight, reverse=True)
        total = sum(c.weight for c in claims) or 1.0

        # For "when" questions, group claims by the date they assert and compare the groups.
        # Values only found in outdated sources are reported as superseded, not as a live conflict.
        current: dict[str, float] = defaultdict(float)
        old: dict[str, float] = defaultdict(float)
        for c in claims if compare_values else []:
            for v in c.values:
                if not v.startswith("£"):
                    (old if c.stale else current)[v] += c.weight
        strong = {v: w for v, w in current.items() if w >= CONFLICT_MIN_WEIGHT}
        superseded = {v: round(w, 3) for v, w in old.items() if v not in current}
        conflict = len(strong) > 1

        if strong:
            winner = max(strong, key=strong.get)
            answer = next(c for c in claims if winner in c.values)
            share = strong[winner] / sum(strong.values())
            confidence = share * min(1.0, answer.support * 1.25) * answer.credibility / max(c.credibility for c in claims)
        else:
            answer = claims[0]
            confidence = min(1.0, answer.support * 1.25) * answer.credibility
            confidence *= min(1.0, answer.weight / total + 0.4)

        return SubAnswer(
            sub_question=sq,
            claims=claims,
            answer=answer,
            confidence=round(min(confidence, 1.0), 3),
            conflict=conflict,
            competing_values={v: round(w, 3) for v, w in sorted(strong.items(), key=lambda x: -x[1])},
            superseded_values=superseded,
        )
