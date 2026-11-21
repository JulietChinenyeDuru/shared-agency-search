# I record every user and AI action with precise timestamps so behaviour can later be aligned with EEG recordings.
from __future__ import annotations

import json
import time
from pathlib import Path


class InteractionLog:
    """Append-only JSONL trace. `t_mono_ms` is for aligning with neural recordings
    (send the same event code as an EEG trigger); `t_wall` is for humans."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        self._t0 = time.monotonic()

    def log(self, actor: str, event: str, **data) -> dict:
        record = {
            "t_mono_ms": round((time.monotonic() - self._t0) * 1000, 1),
            "t_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "actor": actor,        # "user" | "ai" | "policy"
            "event": event,
            **data,
        }
        self.events.append(record)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        return record
