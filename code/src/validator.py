"""
validator.py — Validate model responses and re-prompt on failure.

Checks:
  - action in {notify, digest, mute}
  - message_type in allowed set
  - confidence is float in [0.0, 1.0]
  - every evidence_message_id either "none" or present in BOTH message_history
    AND the candidate list for this message

On failure: re-prompts once with the validation error appended.
If still fails: delegates to fallback.py.
"""

import json
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment",
    "business_update", "promotion", "greeting",
    "forward", "spam", "scam", "unknown",
}


def _validate(
    result: Dict,
    candidate_ids: List[str],
    history_ids: Set[str],
) -> Optional[str]:
    """
    Validate a model result dict. Returns error string if invalid, else None.
    """
    # Check action
    action = result.get("action", "")
    if action not in VALID_ACTIONS:
        return f"Invalid action {action!r}. Must be one of: {sorted(VALID_ACTIONS)}"

    # Check message_type
    msg_type = result.get("message_type", "")
    if msg_type not in VALID_MESSAGE_TYPES:
        return f"Invalid message_type {msg_type!r}. Must be one of: {sorted(VALID_MESSAGE_TYPES)}"

    # Check confidence
    try:
        conf = float(result.get("confidence", -1))
        if not (0.0 <= conf <= 1.0):
            return f"confidence {conf} is out of range [0.0, 1.0]"
    except (TypeError, ValueError):
        return f"confidence {result.get('confidence')!r} is not a valid float"

    # Check evidence_message_ids
    evidence_str = str(result.get("evidence_message_ids", "none")).strip()
    if evidence_str and evidence_str.lower() != "none":
        candidate_id_set = set(candidate_ids)
        for eid in evidence_str.split(";"):
            eid = eid.strip()
            if not eid:
                continue
            if eid not in history_ids:
                return (
                    f"evidence_message_id {eid!r} does not exist in message_history.csv. "
                    f"Only use IDs from the candidate list."
                )
            if eid not in candidate_id_set:
                return (
                    f"evidence_message_id {eid!r} was not in the candidate list provided. "
                    f"Valid candidates: {candidate_ids}"
                )

    return None  # Valid


def validate_and_repair(
    result: Dict,
    candidate_ids: List[str],
    history_ids: Set[str],
    model_client,
    system_prompt: str,
    user_turn_text: str,
    media_parts=None,
) -> Optional[Dict]:
    """
    Validate result. If invalid, re-prompt once with the error.
    Returns corrected result, or None if still invalid after re-prompt.
    """
    error = _validate(result, candidate_ids, history_ids)
    if error is None:
        return result

    logger.warning("Validation failed: %s — re-prompting", error)

    # Build re-prompt with error appended
    reprompt_text = (
        user_turn_text
        + f"\n\n[VALIDATION ERROR — your previous response was invalid: {error}. "
        "Please correct it and respond with valid JSON only.]"
    )

    try:
        result2 = model_client.generate(
            system_prompt=system_prompt,
            user_turn_text=reprompt_text,
            media_parts=media_parts,
        )
    except Exception as e:
        logger.error("Re-prompt also failed: %s", e)
        return None

    error2 = _validate(result2, candidate_ids, history_ids)
    if error2 is None:
        return result2

    logger.warning("Re-prompt validation also failed: %s — delegating to fallback", error2)
    return None
