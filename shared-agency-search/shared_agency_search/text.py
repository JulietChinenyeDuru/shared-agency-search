# I hold the small text utilities shared by the agents: tokenising, sentence splitting and value extraction.
from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has",
    "have", "how", "i", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "does", "did", "do", "can", "could", "should",
    "after", "before", "about", "into", "than", "then", "there", "they", "be",
}

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_DATE_RE = re.compile(rf"\b({MONTHS})\s+(\d{{4}})\b", re.IGNORECASE)
_MONEY_RE = re.compile(r"£\s?(\d+(?:\.\d+)?)\s?(m|million|k|thousand)?", re.IGNORECASE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9£]+")


def tokenize(text: str) -> list[str]:
    """Lower-case content words with stopwords removed."""
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def extract_dates(text: str) -> set[str]:
    """Month-year mentions such as 'March 2027', normalised to 'march 2027'."""
    return {f"{m.lower()} {y}" for m, y in _DATE_RE.findall(text)}


def extract_money(text: str) -> set[str]:
    out = set()
    for amount, unit in _MONEY_RE.findall(text):
        unit = (unit or "").lower()
        unit = {"million": "m", "thousand": "k"}.get(unit, unit)
        out.add(f"£{amount}{unit}")
    return out
