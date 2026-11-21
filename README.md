# Shared-Agency Search: Who Should Act Next?

A small, transparent prototype of a **human–AI search partnership**. A person and a team of AI agents answer questions together. The system keeps track of which evidence is trustworthy, and decides at each step **who should act next**: the person, the AI (with a small, reversible check), or both (a pause to review).

It is built to explore research questions in responsible hybrid intelligence:

- **Shared agency:** when should an AI step in, and when should it hand control back to the person?
- **Error recognition:** can the partnership notice weak sources, conflicting evidence and premature stopping?
- **Co-adaptive, editable memory:** can AI memory stay visible and correctable, so it never silently keeps outdated or rejected information?
- **Neural signals as uncertain evidence:** how can a signal such as an EEG error or hesitation marker inform the system *without* ever being treated as permission to act?

> Status: research prototype. Everything is rule-based and inspectable, and it runs offline with no API keys. The corpus is a small **fictional** scenario (the "Riverside Bridge" in the made-up town of Kelmont), designed with deliberate conflicts, a weak forum rumour, a misleading blog and an outdated news story.

## Architecture

```
 question
    │
    ▼
 Planner agent ──► sub-questions
    │
    ▼
 Retrieval agent (BM25, skips sources the user rejected)
    │
    ▼
 Evidence-checking agent
   • support: does the sentence answer this sub-question?
   • credibility: source-type prior × recency
   • conflict: credible current sources disagree on a value
   • superseded: a value only appears in outdated sources
    │
    ▼
 Shared-agency policy ──► USER_CONTINUES | AI_CHECKS | PAUSE_FOR_REVIEW (+ reasons)
    │                                  ▲
    │          optional external uncertainty (e.g. EEG), capped and never decisive alone
    ▼
 Editable memory (AI proposes; user confirms, edits, deletes, undoes)
    │
    ▼
 Interaction log (timestamped JSONL, ready to align with EEG triggers)
```

| Module | Role |
| --- | --- |
| `planner.py` | Splits compound questions into sub-questions. Swappable for an LLM planner via the same interface. |
| `retriever.py` | Dependency-free BM25 retrieval with source exclusion. |
| `checker.py` | Extracts the best claim per source, scores support and credibility, detects conflicts and outdated evidence. |
| `policy.py` | Decides who acts next, with human-readable reasons. |
| `memory.py` | Inspectable, editable AI memory with undo. The AI can only *propose*. |
| `logger.py` | Append-only interaction trace with monotonic timestamps. |
| `session.py` | Coordinates the agents and the user's actions. |

## Responsible-AI design rules

1. **The person decides the final answer.** The AI never closes a task; it can only suggest.
2. **AI actions are small and reversible.** An AI check is shown as *pending* and changes nothing until the user accepts it. It can be undone.
3. **No silent memory.** AI-written memory is marked `proposed` until a person confirms it. The AI can withdraw its own proposals but can never remove anything the user wrote.
4. **Rejections stick.** A source the user rejects is excluded from all later retrieval.
5. **No quiet stopping with open problems.** If the user tries to finish while sources conflict or questions are unresolved, the system pauses for review.
6. **Neural signals are uncertain evidence.** An external uncertainty signal can add at most a small amount to risk, and only when behaviour and evidence already show a problem. It can never trigger an AI action on its own (see the tests).

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/<your-username>/shared-agency-search.git
cd shared-agency-search
pip install -r requirements.txt

python demo.py          # scripted walkthrough in the terminal
pytest -q               # 12 tests
streamlit run app.py    # interactive web interface
```

## What the demo shows

1. The planner splits *"When will the Riverside Bridge reopen and why was it closed?"* into two sub-questions.
2. For the reopening date, credible sources say **March 2027**, a forum rumour says December 2026, and an old news story says July 2026. The old date is marked **superseded**, not a live conflict. The rumour creates a real conflict, so the policy says **AI_CHECKS**.
3. The user picks the forum post and tries to stop. The policy says **PAUSE_FOR_REVIEW**, with reasons.
4. The AI re-checks using only higher-trust sources. The result is pending until the user accepts it.
5. The user rejects the forum post and the blog. The conflict disappears and the policy hands control back: **USER_CONTINUES**.
6. A high simulated EEG uncertainty is **ignored** when the evidence is clean, but **tips** a borderline case into an AI check.
7. The memory panel shows AI proposals and user-confirmed entries, all editable.

## Limitations and next steps

- **Claim matching is lexical.** Next: a natural-language-inference model to judge whether a sentence supports or contradicts a claim, and knowledge-graph entity linking so that claims about the same entity are compared directly.
- **The planner is rule-based.** Next: an LLM planner behind the same `Planner` interface, with its plan shown to the user.
- **Policy thresholds are hand-set.** Next: fit them from user-study data and compare policies (no proactive handover, behaviour + evidence monitoring, and EEG-informed) on answer quality, recovery from mistakes and unnecessary interruptions.
- **EEG is simulated.** The interaction log is designed so each event can be sent as a trigger code (for example via Lab Streaming Layer) and aligned with recordings for offline analysis, for example with MNE-Python.
- **The corpus is tiny and fictional.** Next: realistic search tasks with conflicting sources and independently labelled errors.

## Author

Juliet Chinenye Duru  ORCID [0009-0002-0530-8082](https://orcid.org/0009-0002-0530-8082)

Related work: [KG-SentIF](https://github.com/JulietChinenyeDuru/kg-sentif), a knowledge-graph-enhanced sentiment intelligence framework.

## Licence

MIT
