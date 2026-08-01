"""
retriever.py — Retrieve and score candidate historical messages for evidence.

For each target message, filter message_history to rows matching the same
user_id AND (same sender_user_id OR same group_id OR same business_id).
Score by recency + keyword overlap. Return top 5-8 candidates with their
message_events engagement data.

The returned candidate list is the ONLY valid source for evidence_message_ids.
"""

import math
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .loader import DataStore


def _tokenize(text: str) -> set:
    """Tokenize text into lowercase word set, filtering short tokens."""
    if not text:
        return set()
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2}


def _recency_score(hist_time_str: str, msg_time_str: str) -> float:
    """
    Score recency: exponential decay where more recent = higher score.
    Returns value in [0, 1].
    """
    try:
        hist_dt = datetime.strptime(hist_time_str, "%Y-%m-%d %H:%M")
        msg_dt = datetime.strptime(msg_time_str, "%Y-%m-%d %H:%M")
        diff_days = max((msg_dt - hist_dt).total_seconds() / 86400.0, 0)
        # Decay constant: half-life of 30 days
        return math.exp(-0.023 * diff_days)
    except Exception:
        return 0.0


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Jaccard similarity between word sets of two texts."""
    set_a = _tokenize(text_a)
    set_b = _tokenize(text_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def retrieve_candidates(
    msg: Dict,
    store: DataStore,
    top_k: int = 8,
) -> List[Dict]:
    """
    Retrieve top-k candidate historical messages for this target message.

    Each returned dict contains the history row fields PLUS:
      - 'score': combined recency + keyword score
      - 'event': message_events row for this user+message_id (or {})
    """
    user_id = msg["user_id"]
    group_id = msg.get("group_id", "")
    business_id = msg.get("business_id", "")
    sender_user_id = msg.get("sender_user_id", "")
    msg_text = msg.get("message_text", "")
    msg_time = msg.get("created_at", "")

    # Filter history: same user AND (same group OR same business OR same sender)
    user_history = store.message_history_by_user.get(user_id, [])

    candidates = []
    for hist in user_history:
        matched = False
        if group_id and hist.get("group_id") == group_id:
            matched = True
        elif business_id and hist.get("business_id") == business_id:
            matched = True
        elif sender_user_id and hist.get("sender_user_id") == sender_user_id:
            matched = True

        if not matched:
            continue

        # Score: 60% recency + 40% keyword overlap
        rec = _recency_score(hist.get("created_at", ""), msg_time)
        kw = _keyword_overlap(msg_text, hist.get("message_text", ""))
        score = 0.6 * rec + 0.4 * kw

        event_key = (user_id, hist["message_id"])
        event = store.message_events.get(event_key, {})

        candidates.append({
            **hist,
            "score": score,
            "event": event,
        })

    # Sort by score descending, deterministically (tie-break by message_id)
    candidates.sort(key=lambda c: (-c["score"], c["message_id"]))

    return candidates[:top_k]


def candidate_ids(candidates: List[Dict]) -> List[str]:
    """Return just the message_ids from a candidate list."""
    return [c["message_id"] for c in candidates]
