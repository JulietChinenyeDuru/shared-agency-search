# I load the document collection and give each source a type, a publication date and a reliability prior.
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Prior trust by source type. These are starting assumptions, not truths: the
# evidence checker combines them with support and recency, and users can override.
SOURCE_PRIOR = {
    "official": 0.90,
    "report": 0.85,
    "news": 0.70,
    "blog": 0.40,
    "forum": 0.30,
}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source_type: str
    published: date
    url: str = ""

    @property
    def prior(self) -> float:
        return SOURCE_PRIOR.get(self.source_type, 0.5)


def load_corpus(path: str | Path) -> list[Document]:
    """Read a JSONL file with one document per line."""
    docs: list[Document] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(
                Document(
                    doc_id=row["id"],
                    title=row["title"],
                    text=row["text"],
                    source_type=row["source_type"],
                    published=date.fromisoformat(row["published"]),
                    url=row.get("url", ""),
                )
            )
    return docs
