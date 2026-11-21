# I run a scripted search session that shows the planner, evidence checks, who-acts-next decisions and editable memory.
from pathlib import Path

from shared_agency_search import SearchSession, load_corpus

DATA = Path(__file__).parent / "data" / "kelmont_bridge.jsonl"


def show_turn(turn):
    print(f"\nQUESTION: {turn.question}")
    for sa in turn.sub_answers:
        a = sa.answer
        print(f"  [{sa.sub_question.sq_id}] {sa.sub_question.text}")
        if a:
            flag = " (STALE)" if a.stale else ""
            print(f"      answer: {a.text}")
            print(f"      source: {a.doc_title} [{a.source_type}, {a.published}]{flag}")
        print(f"      confidence={sa.confidence}  conflict={sa.conflict}  values={sa.competing_values}")
        if sa.superseded_values:
            print(f"      superseded (only in outdated sources): {sa.superseded_values}")
    d = turn.decision
    print(f"  WHO ACTS NEXT: {d.action.value}  (risk {d.risk})  because: {'; '.join(d.reasons)}")


def main():
    session = SearchSession(load_corpus(DATA), log_path=None)

    # 1. The person asks a compound question.
    show_turn(session.ask("When will the Riverside Bridge reopen and why was it closed?"))

    # 2. The person picks a weak source (the forum post) and tries to stop.
    d = session.select_source("forum-post")
    print(f"\nUser picks the forum post -> {d.action.value} (risk {d.risk}): {'; '.join(d.reasons)}")
    d = session.request_stop()
    print(f"User tries to stop        -> {d.action.value}: {'; '.join(d.reasons)}")

    # 3. The AI offers a small, reversible check using higher-trust sources only.
    checked = session.ai_check("q1")
    print(f"\nAI check on q1 (pending, not applied): {checked.answer.text}  confidence={checked.confidence}")
    session.accept_check("q1")
    print("User accepts the check.")

    # 4. The person rejects the forum and the blog; the AI must respect that from now on.
    session.reject_source("forum-post")
    show_turn(session.reject_source("blog-post"))

    # 5. An external uncertainty signal (e.g. from EEG) cannot act alone...
    d = session.request_stop(external_uncertainty=0.9)
    print(f"\nStop, clean evidence, high external uncertainty -> {d.action.value}: {'; '.join(d.reasons)}")

    # ...but it can tip a borderline case when the evidence is already shaky.
    from shared_agency_search import SharedAgencyPolicy, Signals
    policy = SharedAgencyPolicy()
    shaky = Signals(confidence=0.5, conflict=False, stale_evidence=False)
    print(f"Shaky evidence, no external signal   -> {policy.decide(shaky).action.value}")
    shaky.external_uncertainty = 0.9
    print(f"Shaky evidence + external signal 0.9 -> {policy.decide(shaky).action.value}")

    # 6. Memory is visible and editable. The AI only proposes; the user confirms or deletes.
    print("\nMEMORY (AI proposals are not used until confirmed):")
    for e in session.memory.entries():
        print(f"  #{e.entry_id} {e.kind:<16} {e.status:<9} by {e.author}: {e.text[:70]}")

    print(f"\n{len(session.log.events)} timestamped events logged (ready to align with EEG triggers).")


if __name__ == "__main__":
    main()
