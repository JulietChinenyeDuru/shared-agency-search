# I keep the AI's memory visible and editable: the AI can only propose entries, and the user can confirm, edit, delete or undo them.
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import count

KINDS = {"fact", "rejected_source", "unresolved", "preference"}


@dataclass
class MemoryEntry:
    entry_id: int
    kind: str
    text: str
    source: str = ""
    status: str = "proposed"        # proposed -> confirmed (only a user confirms)
    author: str = "ai"              # "ai" or "user"
    updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class EditableMemory:
    def __init__(self):
        self._entries: dict[int, MemoryEntry] = {}
        self._ids = count(1)
        self._undo: list[dict[int, MemoryEntry]] = []

    # --- writes (each one is undoable) ------------------------------------
    def _snapshot(self):
        self._undo.append(copy.deepcopy(self._entries))

    def propose(self, kind: str, text: str, source: str = "") -> MemoryEntry | None:
        """The AI may only propose. Proposed entries do not influence ranking until confirmed.
        Returns None if the same entry already exists."""
        if any(e.kind == kind and e.text == text for e in self._entries.values()):
            return None
        return self._add(kind, text, source, author="ai", status="proposed")

    def retract(self, entry_id: int) -> None:
        """The AI may withdraw its own unconfirmed proposal; it can never remove user or confirmed entries."""
        e = self._entries.get(entry_id)
        if e is None or e.author != "ai" or e.status != "proposed":
            raise PermissionError("the AI can only retract its own unconfirmed proposals")
        self._snapshot()
        del self._entries[entry_id]

    def add_by_user(self, kind: str, text: str, source: str = "") -> MemoryEntry:
        return self._add(kind, text, source, author="user", status="confirmed")

    def _add(self, kind, text, source, author, status) -> MemoryEntry:
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind: {kind}")
        self._snapshot()
        entry = MemoryEntry(next(self._ids), kind, text, source, status, author)
        self._entries[entry.entry_id] = entry
        return entry

    def confirm(self, entry_id: int) -> None:
        self._snapshot()
        self._entries[entry_id].status = "confirmed"

    def edit(self, entry_id: int, new_text: str) -> None:
        self._snapshot()
        e = self._entries[entry_id]
        e.text, e.author, e.status = new_text, "user", "confirmed"
        e.updated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def delete(self, entry_id: int) -> None:
        self._snapshot()
        del self._entries[entry_id]

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._entries = self._undo.pop()
        return True

    # --- reads -------------------------------------------------------------
    def entries(self, kind: str | None = None, confirmed_only: bool = False) -> list[MemoryEntry]:
        out = [e for e in self._entries.values() if kind in (None, e.kind)]
        return [e for e in out if e.status == "confirmed"] if confirmed_only else out

    def rejected_sources(self) -> set[str]:
        return {e.source for e in self.entries("rejected_source", confirmed_only=True)}

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self._entries.values()], indent=2)
