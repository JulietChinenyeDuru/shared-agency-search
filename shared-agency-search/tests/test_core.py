# I test the core promises of the prototype: correct evidence handling, safe who-acts-next decisions and user-controlled memory.
from pathlib import Path

import pytest

from shared_agency_search import Action, SearchSession, SharedAgencyPolicy, Signals, load_corpus
from shared_agency_search.memory import EditableMemory
from shared_agency_search.planner import RuleBasedPlanner

DATA = Path(__file__).resolve().parents[1] / "data" / "kelmont_bridge.jsonl"
QUESTION = "When will the Riverside Bridge reopen and why was it closed?"


@pytest.fixture
def session():
    return SearchSession(load_corpus(DATA))


def test_planner_splits_compound_question():
    subs = RuleBasedPlanner().plan(QUESTION)
    assert len(subs) == 2
    assert "Riverside Bridge" in subs[1].text


def test_date_question_prefers_current_official_answer(session):
    q1 = session.ask(QUESTION).sub_answers[0]
    assert "march 2027" in q1.answer.values
    assert q1.answer.source_type == "official"


def test_outdated_date_is_superseded_not_conflicting(session):
    q1 = session.ask(QUESTION).sub_answers[0]
    assert "july 2026" in q1.superseded_values
    assert "july 2026" not in q1.competing_values


def test_cause_question_finds_cause_sentence(session):
    q2 = session.ask(QUESTION).sub_answers[1]
    assert "corrosion" in q2.answer.text.lower()


def test_conflict_triggers_ai_check(session):
    assert session.ask(QUESTION).decision.action is Action.AI_CHECKS


def test_weak_source_and_premature_stop_trigger_review(session):
    session.ask(QUESTION)
    assert session.select_source("forum-post").action is Action.PAUSE_FOR_REVIEW
    assert session.request_stop().action is Action.PAUSE_FOR_REVIEW


def test_ai_check_is_reversible(session):
    turn = session.ask(QUESTION)
    before = turn.sub_answers[0]
    session.ai_check("q1")
    assert turn.sub_answers[0] is before          # nothing applied yet
    session.undo_check("q1")
    assert turn.sub_answers[0] is before and not turn.pending_checks


def test_rejected_sources_are_never_used_again(session):
    session.ask(QUESTION)
    session.reject_source("forum-post")
    turn = session.reject_source("blog-post")
    used = {c.doc_id for sa in turn.sub_answers for c in sa.claims}
    assert not used & {"forum-post", "blog-post"}
    assert turn.decision.action is Action.USER_CONTINUES


def test_external_signal_cannot_act_alone():
    policy = SharedAgencyPolicy()
    calm = Signals(confidence=0.95, conflict=False, stale_evidence=False, external_uncertainty=1.0)
    assert policy.decide(calm).action is Action.USER_CONTINUES


def test_external_signal_can_tip_borderline_case():
    policy = SharedAgencyPolicy()
    shaky = Signals(confidence=0.5, conflict=False, stale_evidence=False)
    assert policy.decide(shaky).action is Action.USER_CONTINUES
    shaky.external_uncertainty = 0.9
    assert policy.decide(shaky).action is Action.AI_CHECKS


def test_memory_ai_proposes_user_controls():
    mem = EditableMemory()
    proposal = mem.propose("fact", "Bridge reopens March 2027", source="council-notice")
    assert proposal.status == "proposed"
    assert mem.propose("fact", "Bridge reopens March 2027") is None   # no duplicates
    user_entry = mem.add_by_user("preference", "Prefer official sources")
    with pytest.raises(PermissionError):
        mem.retract(user_entry.entry_id)                               # AI cannot remove user entries
    mem.edit(proposal.entry_id, "Bridge reopens March 2027 (council notice)")
    assert mem.entries("fact")[0].author == "user"
    mem.delete(proposal.entry_id)
    assert not mem.entries("fact")
    assert mem.undo() and mem.entries("fact")                          # deletion is undoable


def test_every_action_is_logged_with_timestamps(session):
    session.ask(QUESTION)
    session.request_stop()
    events = session.log.events
    assert events and all("t_mono_ms" in e and "actor" in e for e in events)
    assert [e["t_mono_ms"] for e in events] == sorted(e["t_mono_ms"] for e in events)
