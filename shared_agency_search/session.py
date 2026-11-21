# I coordinate the planner, retriever, evidence checker, shared-agency policy and editable memory for one search session.
from __future__ import annotations

from dataclasses import dataclass, field

from .checker import EvidenceChecker, SubAnswer
from .corpus import Document
from .logger import InteractionLog
from .memory import EditableMemory
from .planner import Planner, RuleBasedPlanner
from .policy import Action, Decision, SharedAgencyPolicy, Signals
from .retriever import BM25Retriever

HIGH_TRUST_PRIOR = 0.70


@dataclass
class Turn:
    question: str
    sub_answers: list[SubAnswer]
    decision: Decision
    pending_checks: dict[str, SubAnswer] = field(default_factory=dict)


class SearchSession:
    def __init__(
        self,
        docs: list[Document],
        planner: Planner | None = None,
        policy: SharedAgencyPolicy | None = None,
        log_path: str | None = None,
    ):
        self.docs = {d.doc_id: d for d in docs}
        self.planner = planner or RuleBasedPlanner()
        self.retriever = BM25Retriever(docs)
        self.checker = EvidenceChecker(idf=self.retriever.idf)
        self.policy = policy or SharedAgencyPolicy()
        self.memory = EditableMemory()
        self.log = InteractionLog(log_path)
        self.turn: Turn | None = None

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, external_uncertainty: float | None = None) -> Turn:
        self.log.log("user", "ask", question=question)
        subs = self.planner.plan(question)
        self.log.log("ai", "plan", sub_questions=[s.text for s in subs])
        answers = [self._answer(sq) for sq in subs]
        decision = self.policy.decide(self._signals(answers, external_uncertainty=external_uncertainty))
        self.turn = Turn(question, answers, decision)
        self._propose_memory(answers)
        self.log.log("policy", "decide", action=decision.action.value, risk=decision.risk, reasons=decision.reasons)
        return self.turn

    def _answer(self, sq) -> SubAnswer:
        hits = self.retriever.search(sq.text, k=6, exclude=self.memory.rejected_sources())
        sa = self.checker.check(sq, hits)
        self.log.log(
            "ai", "check", sq=sq.sq_id, confidence=sa.confidence, conflict=sa.conflict,
            answer_doc=sa.answer.doc_id if sa.answer else None,
        )
        return sa

    def _signals(self, answers: list[SubAnswer], **extra) -> Signals:
        answered = [a for a in answers if a.answer]
        return Signals(
            confidence=min((a.confidence for a in answers), default=0.0),
            conflict=any(a.conflict for a in answers),
            stale_evidence=any(a.answer.stale for a in answered),
            unresolved_count=sum(a.unresolved for a in answers),
            **extra,
        )

    def _propose_memory(self, answers: list[SubAnswer]) -> None:
        for a in answers:
            if a.unresolved:
                self.memory.propose("unresolved", a.sub_question.text)
            elif a.answer:
                # A question that is now settled no longer needs the AI's earlier "unresolved" note.
                for e in self.memory.entries("unresolved"):
                    if e.text == a.sub_question.text and e.author == "ai" and e.status == "proposed":
                        self.memory.retract(e.entry_id)
                self.memory.propose("fact", a.answer.text, source=a.answer.doc_id)

    # ------------------------------------------------------ user actions
    def select_source(self, doc_id: str) -> Decision:
        """The user says 'I'll go with this source'. Weak choices trigger a check offer."""
        doc = self.docs[doc_id]
        self.log.log("user", "select_source", doc_id=doc_id, source_type=doc.source_type)
        signals = self._signals(self.turn.sub_answers, user_source_prior=doc.prior)
        decision = self.policy.decide(signals)
        self.log.log("policy", "decide", action=decision.action.value, risk=decision.risk, reasons=decision.reasons)
        return decision

    def reject_source(self, doc_id: str) -> Turn:
        """User distrusts a source: remembered (confirmed, user-authored) and the question is re-run."""
        self.log.log("user", "reject_source", doc_id=doc_id)
        self.memory.add_by_user("rejected_source", f"Do not use {self.docs[doc_id].title}", source=doc_id)
        return self.ask(self.turn.question)

    def request_stop(self, external_uncertainty: float | None = None) -> Decision:
        self.log.log("user", "request_stop")
        signals = self._signals(self.turn.sub_answers, user_wants_stop=True, external_uncertainty=external_uncertainty)
        decision = self.policy.decide(signals)
        self.log.log("policy", "decide", action=decision.action.value, risk=decision.risk, reasons=decision.reasons)
        return decision

    # ------------------------------------------- reversible AI action
    def ai_check(self, sq_id: str) -> SubAnswer:
        """Re-check one sub-question using only higher-trust sources. Nothing changes until the user accepts."""
        current = self._find(sq_id)
        hits = self.retriever.search(current.sub_question.text, k=8, exclude=self.memory.rejected_sources())
        trusted = [h for h in hits if h.doc.prior >= HIGH_TRUST_PRIOR]
        checked = self.checker.check(current.sub_question, trusted)
        self.turn.pending_checks[sq_id] = checked
        self.log.log("ai", "ai_check", sq=sq_id, new_confidence=checked.confidence,
                     new_answer_doc=checked.answer.doc_id if checked.answer else None)
        return checked

    def accept_check(self, sq_id: str) -> None:
        checked = self.turn.pending_checks.pop(sq_id)
        self.turn.sub_answers = [checked if a.sub_question.sq_id == sq_id else a for a in self.turn.sub_answers]
        self.log.log("user", "accept_check", sq=sq_id)

    def undo_check(self, sq_id: str) -> None:
        self.turn.pending_checks.pop(sq_id, None)
        self.log.log("user", "undo_check", sq=sq_id)

    def _find(self, sq_id: str) -> SubAnswer:
        return next(a for a in self.turn.sub_answers if a.sub_question.sq_id == sq_id)

    @property
    def needs_ai_check(self) -> bool:
        return self.turn is not None and self.turn.decision.action is Action.AI_CHECKS
