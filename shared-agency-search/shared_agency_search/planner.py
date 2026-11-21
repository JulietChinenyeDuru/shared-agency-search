# I am the planning agent: I break a user's question into smaller sub-questions the other agents can answer.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

_SPLIT_RE = re.compile(r"\s+(?:and|also|plus)\s+(?=(?:what|when|why|how|who|where|which|is|are|will|does|did)\b)", re.IGNORECASE)


@dataclass(frozen=True)
class SubQuestion:
    sq_id: str
    text: str


class Planner(Protocol):
    def plan(self, question: str) -> list[SubQuestion]: ...


class RuleBasedPlanner:
    """Splits compound questions on 'and/also' before a new wh-clause.

    This is deliberately simple and transparent. An LLM planner can replace it
    by implementing the same `plan` method, without changing the other agents.
    """

    def plan(self, question: str) -> list[SubQuestion]:
        question = question.strip().rstrip("?")
        parts = [p.strip(" ,") for p in _SPLIT_RE.split(question) if p.strip(" ,")]
        if len(parts) > 1:
            subject = self._subject(parts[0])
            parts = [parts[0]] + [self._carry_subject(p, subject) for p in parts[1:]]
        return [SubQuestion(f"q{i + 1}", p[0].upper() + p[1:] + "?") for i, p in enumerate(parts)]

    @staticmethod
    def _subject(first_part: str) -> str:
        """Pick up a capitalised noun phrase (e.g. 'Riverside Bridge') to reuse in later parts."""
        match = re.search(r"((?:[A-Z][a-z]+\s?){2,})", first_part)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _carry_subject(part: str, subject: str) -> str:
        if subject and subject.lower() not in part.lower():
            part = re.sub(r"\b(it|they)\b", subject, part, count=1, flags=re.IGNORECASE)
            if subject.lower() not in part.lower():
                part = f"{part} ({subject})"
        return part
