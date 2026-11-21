# I decide who should act next in the search partnership: the person continues, the AI makes a small reversible check, or both pause to review.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Action(str, Enum):
    USER_CONTINUES = "user_continues"
    AI_CHECKS = "ai_checks"            # small, reversible evidence check the user can undo
    PAUSE_FOR_REVIEW = "pause_for_review"


@dataclass
class Signals:
    """Everything the policy may look at. All values are in [0, 1] unless stated."""

    confidence: float                      # evidence checker's confidence in the current answer
    conflict: bool                         # credible sources disagree on a key value
    stale_evidence: bool                   # the chosen answer relies on an outdated source
    unresolved_count: int = 0              # sub-questions without a settled answer
    user_source_prior: float | None = None # prior of a source the user just picked, if any
    user_wants_stop: bool = False          # the user is about to accept a final answer
    external_uncertainty: float | None = None
    """Optional uncertainty estimate from another channel, e.g. an EEG-derived
    error or hesitation marker. It is treated as weak, uncertain evidence: it can
    raise risk a little, but it can never trigger an AI action on its own."""


@dataclass
class Decision:
    action: Action
    risk: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class SharedAgencyPolicy:
    check_threshold: float = 0.35
    pause_threshold: float = 0.60
    external_weight: float = 0.15      # cap on how much an external signal can move risk
    external_floor: float = 0.20       # behaviour/evidence risk needed before external signal counts
    weak_source_prior: float = 0.50

    def decide(self, s: Signals) -> Decision:
        reasons: list[str] = []

        # Hard rule: never let the person close the task with open problems silently.
        if s.user_wants_stop and (s.unresolved_count > 0 or s.conflict):
            return Decision(
                Action.PAUSE_FOR_REVIEW,
                1.0,
                [f"stopping with {s.unresolved_count} unresolved sub-question(s) or a source conflict"],
            )

        base = 0.6 * (1 - s.confidence)
        if s.conflict:
            base += 0.25
            reasons.append("credible sources disagree")
        if s.stale_evidence:
            base += 0.10
            reasons.append("answer relies on possibly outdated evidence")
        if s.user_source_prior is not None and s.user_source_prior < self.weak_source_prior:
            base += 0.20
            reasons.append("selected source has low prior reliability")
        if s.confidence < 0.5:
            reasons.append(f"low evidence confidence ({s.confidence:.2f})")
        base = min(base, 1.0)

        risk = base
        if s.external_uncertainty is not None:
            if base >= self.external_floor:
                risk = min(1.0, base + self.external_weight * s.external_uncertainty)
                reasons.append(f"external signal added {risk - base:.2f} (treated as uncertain evidence)")
            else:
                reasons.append("external signal ignored: behaviour and evidence show no problem")

        if risk >= self.pause_threshold:
            action = Action.PAUSE_FOR_REVIEW
        elif risk >= self.check_threshold:
            action = Action.AI_CHECKS
        else:
            action = Action.USER_CONTINUES
            if not reasons:
                reasons.append("evidence is consistent and well supported")
        return Decision(action, round(risk, 3), reasons)
