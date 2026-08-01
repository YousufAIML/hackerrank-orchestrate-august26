"""
model_client.py — Dual-provider model client: Groq (text) + Gemini (multimodal).

Routing strategy:
  - Text-only messages → Groq (high RPM free tier)
      Primary:  GROQ_MODEL          (default: llama-3.3-70b-versatile)
      Fallback: GROQ_FALLBACK_MODEL (default: llama-3.1-8b-instant)
      On 429/rate-limit/quota error → immediately switch to fallback model.
      If BOTH Groq models fail → raise; caller uses FallbackClassifier.
      Gemini quota is NEVER used for text-only messages.

  - Media messages (image/voice) → Gemini lite-first (higher daily quota)
      Primary:  GEMINI_MODEL        (default: gemini-2.5-flash-lite)
      Secondary: GEMINI_FALLBACK_MODEL (default: gemini-2.5-flash)
      On daily quota hit for lite → switch to flash immediately.
      On all Gemini failure → raise; caller uses FallbackClassifier.

Environment variables:
  GEMINI_API_KEY          — required for multimodal messages
  GEMINI_MODEL            — default "gemini-2.5-flash-lite"
  GEMINI_FALLBACK_MODEL   — default "gemini-2.5-flash"
  GROQ_API_KEY            — optional; enables Groq for text-only messages
  GROQ_MODEL              — default "llama-3.3-70b-versatile"
  GROQ_FALLBACK_MODEL     — default "llama-3.1-8b-instant"
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Gemini response schema (structured JSON)
# ─────────────────────────────────────────────
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["notify", "digest", "mute"],
        },
        "message_type": {
            "type": "string",
            "enum": [
                "personal", "urgent", "event", "payment",
                "business_update", "promotion", "greeting",
                "forward", "spam", "scam", "unknown",
            ],
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence_message_ids": {"type": "string"},
    },
    "required": ["action", "message_type", "reason", "confidence", "evidence_message_ids"],
}

# JSON schema as a string for Groq (injected into system prompt)
GROQ_JSON_SCHEMA_INSTRUCTION = """
You MUST respond with a single valid JSON object (no markdown fences, no explanation) with exactly these fields:
{
  "action": "<notify|digest|mute>",
  "message_type": "<personal|urgent|event|payment|business_update|promotion|greeting|forward|spam|scam|unknown>",
  "reason": "<short human-readable explanation>",
  "confidence": <float 0.0-1.0>,
  "evidence_message_ids": "<semicolon-separated IDs from candidate list, or 'none'>"
}
No other fields. No markdown. Valid JSON only.
"""

MAX_RETRIES = 4
BASE_BACKOFF = 2  # seconds


def _extract_retry_delay(e) -> Optional[int]:
    """Extract retryDelay seconds from a Gemini API error, or None."""
    try:
        details_obj = getattr(e, "details", None)
        items = []
        if isinstance(details_obj, list):
            items = details_obj
        elif isinstance(details_obj, dict):
            items = details_obj.get("error", {}).get("details", [])
        for item in items:
            if isinstance(item, dict) and item.get("@type", "").endswith("RetryInfo"):
                delay_str = str(item.get("retryDelay", ""))
                if delay_str.endswith("s"):
                    val = float(delay_str[:-1])
                    return int(val) + 2
    except Exception:
        pass
    return None


def _is_rate_limit_error(e) -> bool:
    http_code = getattr(e, "code", None)
    status_str = (getattr(e, "status", "") or "").upper()
    msg_str = str(e).lower()
    return (
        http_code == 429
        or "RESOURCE_EXHAUSTED" in status_str
        or "resource_exhausted" in msg_str
        or "quota" in msg_str
        or "rate_limit" in msg_str
        or "rate limit" in msg_str
    )


def _is_not_found_error(e) -> bool:
    http_code = getattr(e, "code", None)
    status_str = (getattr(e, "status", "") or "").upper()
    msg_str = str(e).lower()
    return (
        http_code == 404
        or "NOT_FOUND" in status_str
        or "not found" in msg_str
        or "no longer available" in msg_str
        or "does not exist" in msg_str
    )


class ModelClient:
    """
    Dual-provider model client.

    Routes text-only messages to Groq (fast, high RPM with internal fallback)
    and multimodal messages to Gemini (lite-first).
    """

    def __init__(self):
        # ── Gemini setup (lite-first: higher daily quota on free tier) ──
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        self._gemini_client = genai.Client(api_key=gemini_key)
        self._primary_gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self._fallback_gemini_model = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
        self._active_gemini_model = self._primary_gemini_model
        self._gemini_switched = False

        # ── Groq setup ──
        groq_key = os.environ.get("GROQ_API_KEY", "")
        self._groq_client = None
        self._primary_groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._fallback_groq_model = os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
        self._active_groq_model = self._primary_groq_model
        self._groq_available = False
        self._groq_switched = False

        if groq_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=groq_key)
                self._groq_available = True
                logger.info(
                    "Groq client initialized with model %s (fallback: %s)",
                    self._active_groq_model, self._fallback_groq_model
                )
            except ImportError:
                logger.warning("groq package not installed; text messages will use rule-based fallback")
            except Exception as e:
                logger.warning("Groq init failed: %s; text messages will use rule-based fallback", e)

        # Call counts per model name
        self.model_counts = defaultdict(int)

    # ── Public properties ────────────────────────────────────────────────────

    @property
    def active_gemini_model(self) -> str:
        return self._active_gemini_model

    @property
    def active_groq_model(self) -> str:
        return self._active_groq_model

    @property
    def switched_to_fallback(self) -> bool:
        return self._gemini_switched or self._groq_switched

    @property
    def groq_available(self) -> bool:
        return self._groq_available

    def summary(self) -> str:
        parts = []
        for model_name, count in self.model_counts.items():
            if count > 0:
                parts.append(f"{model_name}({count} calls)")
        if self._groq_switched:
            parts.append(f"groq_switched_to={self._active_groq_model}")
        if self._gemini_switched:
            parts.append(f"gemini_switched_to={self._active_gemini_model}")
        return ", ".join(parts) if parts else "no model calls"

    # ── Main entry point ─────────────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_turn_text: str,
        media_parts: Optional[List[Any]] = None,
    ) -> Dict:
        """
        Generate a routing decision.

        Text-only path (no media_parts):
          → Groq only (tries primary GROQ_MODEL, falls back to GROQ_FALLBACK_MODEL).
            On failure of BOTH Groq models, RAISE — caller (main.py) catches and
            uses FallbackClassifier. Gemini quota is NEVER used for text.

        Media path (has media_parts):
          → Gemini only (lite-first). On failure, RAISE — caller uses
            FallbackClassifier. Groq does not support inline image/audio.
        """
        has_media = bool(media_parts)

        if not has_media:
            if self._groq_available:
                # Raises on failure of both Groq models — no Gemini fallthrough for text
                return self._call_groq(system_prompt, user_turn_text)
            else:
                # No Groq configured — raise immediately so caller uses fallback.py
                raise RuntimeError("Groq not available and Gemini reserved for media")

        # Media path — Gemini only
        return self._call_gemini(system_prompt, user_turn_text, media_parts)

    # ── Groq ─────────────────────────────────────────────────────────────────

    def _switch_groq_to_fallback(self, reason: str):
        logger.warning(
            "Switching Groq from %s to %s. Reason: %s",
            self._active_groq_model, self._fallback_groq_model, reason
        )
        self._active_groq_model = self._fallback_groq_model
        self._groq_switched = True

    def _disable_groq(self, reason: str):
        """Mark Groq as unavailable for all remaining messages this run."""
        logger.warning("Disabling Groq for rest of run: %s", reason)
        self._groq_available = False

    def _call_groq_single(self, model_name: str, full_system: str, user_turn_text: str) -> Dict:
        """Make a single call to a Groq model, retrying once on JSON parse error."""
        for attempt in range(2):
            try:
                response = self._groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user_turn_text},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=512,
                )
                raw = response.choices[0].message.content or ""
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = "\n".join(
                        line for line in raw.splitlines()
                        if not line.startswith("```")
                    ).strip()
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == 0:
                    logger.warning("Groq JSON parse error on %s, retrying once: %s", model_name, e)
                    continue
                raise

    def _call_groq(self, system_prompt: str, user_turn_text: str) -> Dict:
        """
        Call Groq API with model fallback chain.
        Primary: GROQ_MODEL (default llama-3.3-70b-versatile)
        Fallback: GROQ_FALLBACK_MODEL (default llama-3.1-8b-instant)

        On 429/rate-limit/quota error on primary model:
          - Immediately switch to fallback model and retry once.
        If fallback model ALSO fails (or is already active and fails):
          - Disable Groq for the rest of the run and raise, so caller uses FallbackClassifier.
        """
        full_system = system_prompt.strip() + "\n\n" + GROQ_JSON_SCHEMA_INSTRUCTION.strip()
        current_model = self._active_groq_model

        try:
            result = self._call_groq_single(current_model, full_system, user_turn_text)
            self.model_counts[current_model] += 1
            return result
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = (
                "rate_limit" in err_str
                or "rate limit" in err_str
                or "429" in err_str
                or "quota" in err_str
                or "tokens" in err_str
            )
            # If rate limit/quota error and we haven't switched to Groq fallback model yet:
            if is_rate_limit and current_model != self._fallback_groq_model:
                self._switch_groq_to_fallback(str(e))
                fallback_model = self._active_groq_model
                try:
                    result = self._call_groq_single(fallback_model, full_system, user_turn_text)
                    self.model_counts[fallback_model] += 1
                    return result
                except Exception as e2:
                    logger.warning("Groq fallback model %s also failed: %s", fallback_model, e2)
                    self._disable_groq(f"Both Groq models failed. Fallback error: {e2}")
                    raise e2
            else:
                if is_rate_limit:
                    self._disable_groq(f"Groq model {current_model} rate-limited/exhausted: {e}")
                raise e

    # ── Gemini ───────────────────────────────────────────────────────────────

    def _switch_gemini_to_fallback(self, reason: str):
        logger.warning(
            "Switching Gemini from %s to %s. Reason: %s",
            self._active_gemini_model, self._fallback_gemini_model, reason
        )
        self._active_gemini_model = self._fallback_gemini_model
        self._gemini_switched = True

    @staticmethod
    def _is_daily_quota(e) -> bool:
        """Return True if the error is a daily quota cap (not just per-minute)."""
        try:
            details_obj = getattr(e, "details", None)
            items = []
            if isinstance(details_obj, list):
                items = details_obj
            elif isinstance(details_obj, dict):
                items = details_obj.get("error", {}).get("details", [])
            for item in items:
                if isinstance(item, dict) and item.get("@type", "").endswith("QuotaFailure"):
                    for v in item.get("violations", []):
                        quota_id = v.get("quotaId", "")
                        if "PerDay" in quota_id or "Daily" in quota_id:
                            return True
        except Exception:
            pass
        return False

    def _call_gemini(
        self,
        system_prompt: str,
        user_turn_text: str,
        media_parts: List[Any],
    ) -> Dict:
        """Call Gemini API with retry + 404/quota fallback model switch."""
        contents_parts = [user_turn_text]
        for part in media_parts:
            if part is not None:
                contents_parts.append(part)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        )

        last_error = None
        rate_limit_exhausted = False

        for attempt in range(MAX_RETRIES):
            try:
                response = self._gemini_client.models.generate_content(
                    model=self._active_gemini_model,
                    contents=contents_parts,
                    config=config,
                )
                raw_text = response.text
                if raw_text:
                    parsed = json.loads(raw_text)
                    self.model_counts[self._active_gemini_model] += 1
                    return parsed
                raise ValueError("Empty response from Gemini")

            except ClientError as e:
                if _is_not_found_error(e):
                    self._switch_gemini_to_fallback(str(e))
                    continue  # retry with new model, don't count attempt

                if _is_rate_limit_error(e):
                    rate_limit_exhausted = True

                    # Detect daily quota exhaustion — switch to fallback immediately
                    is_daily_cap = self._is_daily_quota(e)
                    if is_daily_cap and not self._gemini_switched:
                        self._switch_gemini_to_fallback("daily quota exhausted — switching to fallback model")
                        continue  # retry once with fallback model

                    delay = _extract_retry_delay(e)
                    wait = delay if delay else max(15, BASE_BACKOFF ** (attempt + 2))
                    logger.warning("Gemini rate limit (attempt %d), waiting %ds", attempt + 1, wait)
                    time.sleep(wait)
                    last_error = e
                    continue

                logger.error("Gemini client error: %s", e)
                raise

            except ServerError as e:
                wait = BASE_BACKOFF ** (attempt + 1)
                logger.warning("Gemini server error (attempt %d), waiting %ds", attempt + 1, wait)
                time.sleep(wait)
                last_error = e
                continue

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = BASE_BACKOFF ** (attempt + 1)
                    logger.warning("Gemini unexpected error (attempt %d), waiting %ds: %s", attempt + 1, wait, e)
                    time.sleep(wait)
                    last_error = e
                    continue
                raise

        # All retries exhausted — try fallback model once if we haven't already
        if rate_limit_exhausted and not self._gemini_switched:
            self._switch_gemini_to_fallback("all retries exhausted due to rate limits")
            try:
                response = self._gemini_client.models.generate_content(
                    model=self._active_gemini_model,
                    contents=contents_parts,
                    config=config,
                )
                raw_text = response.text
                if raw_text:
                    parsed = json.loads(raw_text)
                    self.model_counts[self._active_gemini_model] += 1
                    return parsed
            except Exception as fallback_err:
                logger.error("Gemini fallback model also failed: %s", fallback_err)
                raise fallback_err

        if last_error:
            raise last_error
        raise RuntimeError("Gemini: max retries exhausted")
