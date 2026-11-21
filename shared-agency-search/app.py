# I provide a small web interface where a person and the AI search together, see who should act next, and edit the AI's memory.
from pathlib import Path

import streamlit as st

from shared_agency_search import Action, SearchSession, load_corpus

DATA = Path(__file__).parent / "data" / "kelmont_bridge.jsonl"
ACTION_TEXT = {
    Action.USER_CONTINUES: ("You continue", "Evidence looks consistent. You stay in control."),
    Action.AI_CHECKS: ("AI offers a check", "The AI can run a small, reversible check. Nothing changes unless you accept."),
    Action.PAUSE_FOR_REVIEW: ("Pause and review", "Something needs your attention before going further."),
}

st.set_page_config(page_title="Shared-Agency Search", layout="wide")
st.title("Who should act next? A shared-agency search prototype")

if "session" not in st.session_state:
    st.session_state.session = SearchSession(load_corpus(DATA), log_path="interaction_log.jsonl")
session: SearchSession = st.session_state.session

question = st.text_input("Ask a question", "When will the Riverside Bridge reopen and why was it closed?")
external = st.sidebar.slider(
    "Simulated external uncertainty (e.g. EEG marker)", 0.0, 1.0, 0.0, 0.05,
    help="Treated as weak, uncertain evidence. It cannot trigger an AI action on its own.",
)
ext = external if external > 0 else None

if st.button("Search", type="primary"):
    session.ask(question, external_uncertainty=ext)

turn = session.turn
if turn:
    left, right = st.columns([3, 2])
    with left:
        for sa in turn.sub_answers:
            with st.container(border=True):
                st.subheader(sa.sub_question.text)
                if sa.answer:
                    st.write(sa.answer.text)
                    st.caption(f"{sa.answer.doc_title} · {sa.answer.source_type} · {sa.answer.published}"
                               + (" · possibly outdated" if sa.answer.stale else ""))
                st.progress(sa.confidence, text=f"Evidence confidence {sa.confidence:.2f}")
                if sa.conflict:
                    st.warning(f"Sources disagree: {sa.competing_values}")
                if sa.superseded_values:
                    st.info(f"Superseded by newer sources: {list(sa.superseded_values)}")

                sq_id = sa.sub_question.sq_id
                pending = turn.pending_checks.get(sq_id)
                if pending:
                    st.success(f"AI check result (not applied): {pending.answer.text if pending.answer else 'no answer'} "
                               f"· confidence {pending.confidence:.2f}")
                    c1, c2 = st.columns(2)
                    if c1.button("Accept", key=f"acc-{sq_id}"):
                        session.accept_check(sq_id)
                        st.rerun()
                    if c2.button("Undo", key=f"undo-{sq_id}"):
                        session.undo_check(sq_id)
                        st.rerun()
                elif st.button("Ask the AI to check this", key=f"chk-{sq_id}"):
                    session.ai_check(sq_id)
                    st.rerun()

                with st.expander("All evidence"):
                    for c in sa.claims:
                        cols = st.columns([5, 1])
                        cols[0].markdown(f"**{c.doc_title}** ({c.source_type}, {c.published}) — {c.text}  \n"
                                         f"support {c.support:.2f} · credibility {c.credibility:.2f}")
                        if cols[1].button("Reject source", key=f"rej-{sq_id}-{c.doc_id}"):
                            session.reject_source(c.doc_id)
                            st.rerun()

    with right:
        label, help_text = ACTION_TEXT[turn.decision.action]
        st.metric("Who should act next", label, f"risk {turn.decision.risk:.2f}", delta_color="off")
        st.caption(help_text)
        for r in turn.decision.reasons:
            st.write(f"- {r}")
        if st.button("I'm done: accept the answer"):
            d = session.request_stop(external_uncertainty=ext)
            (st.success if d.action is Action.USER_CONTINUES else st.warning)(
                f"{ACTION_TEXT[d.action][0]}: {'; '.join(d.reasons)}")

        st.divider()
        st.subheader("AI memory (visible and editable)")
        for e in session.memory.entries():
            with st.container(border=True):
                new = st.text_input(f"{e.kind} · {e.status} · by {e.author}", e.text, key=f"m-{e.entry_id}")
                c1, c2, c3 = st.columns(3)
                if e.status == "proposed" and c1.button("Confirm", key=f"mc-{e.entry_id}"):
                    session.memory.confirm(e.entry_id)
                    st.rerun()
                if new != e.text and c2.button("Save edit", key=f"me-{e.entry_id}"):
                    session.memory.edit(e.entry_id, new)
                    st.rerun()
                if c3.button("Delete", key=f"md-{e.entry_id}"):
                    session.memory.delete(e.entry_id)
                    st.rerun()
        if st.button("Undo last memory change"):
            session.memory.undo()
            st.rerun()

    with st.sidebar:
        st.caption(f"{len(session.log.events)} events logged to interaction_log.jsonl")
