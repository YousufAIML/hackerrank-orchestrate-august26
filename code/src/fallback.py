"""
fallback.py — Deterministic rule-based backstop classifier.

Always returns a valid output row. Used when:
  - The model API is down
  - The model response fails validation twice
  - An unexpected exception occurs in the pipeline

Rules (in priority order):
  1. Prompt injection attempt detected → mute/scam
  2. Scam/OTP/phishing signals → mute/scam
  3. Spam/forward signals → mute/spam
  4. DND window + non-urgent → digest
  5. Verified business + opted-in → digest (promotions) or notify (payment/delivery)
  6. Unverified/new business with money keywords → mute/scam
  7. Group muted by user → digest
  8. Default → digest/unknown
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# Keyword patterns (all lowercase)
SCAM_PATTERNS = re.compile(
    r"\b(otp|one.?time.?pass|verify.?now|account.?block|suspend|urgent.?action|"
    r"click.?here|limited.?time|claim.?now|verify.?account|password|credential|"
    r"confirm.?identity|security.?alert|fraud.?alert|your.?account.?will|"
    r"send.?money|transfer.?funds|upi.?id|pay.?now|bank.?details|cvv|"
    r"gift.?card|amazon.?pay\s*delivery|lottery|won.?prize|congratulations.+won)\b",
    re.IGNORECASE,
)

INJECTION_PATTERNS = re.compile(
    r"ignore.{0,20}(previous|above|instructions)|"
    r"disregard.{0,20}(rules|instructions)|"
    r"you (must|should|shall) (output|respond|set).{0,30}(notify|mute|digest|confidence)|"
    r"override.{0,20}(rule|instruction)|"
    r"pretend.{0,20}(you are|to be)|"
    r"system.{0,10}prompt",
    re.IGNORECASE,
)

OTP_PATTERNS = re.compile(
    r"\botp\b|one.?time.?password|verification.?code",
    re.IGNORECASE,
)

PAYMENT_KEYWORDS = re.compile(
    r"\b(payment|invoice|bill|pay now|upi|bank transfer|wire transfer|amount due)\b",
    re.IGNORECASE,
)

DELIVERY_KEYWORDS = re.compile(
    r"\b(order|delivery|shipped|dispatch|out for delivery|package)\b",
    re.IGNORECASE,
)

GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening|namaste|hola|greetings)[,!.]?\s*$",
    re.IGNORECASE,
)


def _count_forwarded(forwarded_count: int) -> bool:
    return forwarded_count >= 3


class FallbackClassifier:
    """Deterministic rule-based classifier that always returns a valid row."""

    def __init__(self):
        self.fallback_count = 0

    def classify(self, msg: Dict, context: Dict, reason_prefix: str = "") -> Dict:
        """
        Classify a message deterministically. Always returns a valid dict with
        action, message_type, reason, confidence, evidence_message_ids.
        """
        self.fallback_count += 1

        text = msg.get("message_text", "") or ""
        forwarded = int(msg.get("forwarded_count", 0) or 0)
        media_type = msg.get("media_type", "") or ""

        business_info = context.get("business_info", {})
        group_info = context.get("group_info", {})
        is_in_dnd = context.get("is_in_dnd", False)
        sender_type = context.get("sender_type", "")

        prefix = f"[FALLBACK] {reason_prefix} " if reason_prefix else "[FALLBACK] "

        # 1. Prompt injection attempt
        if INJECTION_PATTERNS.search(text):
            result = self._make(
                action="mute",
                message_type="scam",
                reason=prefix + "Message text contains prompt injection attempt; classified as scam.",
                confidence=0.95,
            )
            self._log(msg, result, "prompt_injection")
            return result

        # 2. OTP / phishing signals
        if OTP_PATTERNS.search(text) or SCAM_PATTERNS.search(text):
            # Extra check: is sender verified business?
            if business_info.get("verified") and business_info.get("domain_match"):
                result = self._make(
                    action="notify",
                    message_type="payment",
                    reason=prefix + "OTP/security message from verified business with matching domain.",
                    confidence=0.70,
                )
            else:
                result = self._make(
                    action="mute",
                    message_type="scam",
                    reason=prefix + "Message contains OTP/phishing/scam language from unverified sender.",
                    confidence=0.90,
                )
            self._log(msg, result, "otp_scam_pattern")
            return result

        # 3. Highly forwarded → spam/forward
        if _count_forwarded(forwarded):
            result = self._make(
                action="mute",
                message_type="forward" if forwarded < 10 else "spam",
                reason=prefix + f"Message forwarded {forwarded} times; likely spam or chain message.",
                confidence=0.75,
            )
            self._log(msg, result, "high_forward_count")
            return result

        # 4. Unverified business with payment keywords → scam
        if sender_type == "business" and business_info:
            verified = business_info.get("verified", False)
            domain_match = business_info.get("domain_match", False)
            if not verified or not domain_match:
                if PAYMENT_KEYWORDS.search(text) or OTP_PATTERNS.search(text):
                    result = self._make(
                        action="mute",
                        message_type="scam",
                        reason=prefix + "Unverified business or domain mismatch with payment/OTP content.",
                        confidence=0.85,
                    )
                    self._log(msg, result, "unverified_biz_payment")
                    return result

        # 5. Greeting only
        if GREETING_PATTERNS.match(text.strip()) and not media_type:
            result = self._make(
                action="digest",
                message_type="greeting",
                reason=prefix + "Simple greeting message; batched for later review.",
                confidence=0.65,
            )
            self._log(msg, result, "greeting")
            return result

        # 6. DND window → digest (unless it's a payment/delivery from trusted source)
        if is_in_dnd:
            if sender_type == "business" and business_info.get("verified"):
                if DELIVERY_KEYWORDS.search(text):
                    result = self._make(
                        action="digest",
                        message_type="business_update",
                        reason=prefix + "Delivery update from verified business during DND; digest for later.",
                        confidence=0.68,
                    )
                    self._log(msg, result, "dnd_delivery")
                    return result
            result = self._make(
                action="digest",
                message_type="unknown",
                reason=prefix + "Message arrived during user's DND window; batched for later.",
                confidence=0.60,
            )
            self._log(msg, result, "dnd_window")
            return result

        # 7. Group muted by user
        if sender_type == "group" and group_info.get("user_group_muted"):
            result = self._make(
                action="digest",
                message_type="unknown",
                reason=prefix + "User has muted this group; message batched.",
                confidence=0.65,
            )
            self._log(msg, result, "group_muted")
            return result

        # 8. Business promotion + opted out
        if sender_type == "business" and business_info.get("user_opted_out_promotions"):
            result = self._make(
                action="mute",
                message_type="promotion",
                reason=prefix + "User has opted out of promotions from this business.",
                confidence=0.80,
            )
            self._log(msg, result, "opted_out_promo")
            return result

        # 9. Verified business delivery/payment → notify
        if sender_type == "business" and business_info.get("verified"):
            if PAYMENT_KEYWORDS.search(text):
                result = self._make(
                    action="notify",
                    message_type="payment",
                    reason=prefix + "Payment-related message from verified business.",
                    confidence=0.72,
                )
                self._log(msg, result, "verified_biz_payment")
                return result
            if DELIVERY_KEYWORDS.search(text):
                result = self._make(
                    action="notify",
                    message_type="business_update",
                    reason=prefix + "Delivery update from verified business.",
                    confidence=0.70,
                )
                self._log(msg, result, "verified_biz_delivery")
                return result

        # 10. Default fallback
        result = self._make(
            action="digest",
            message_type="unknown",
            reason=prefix + "No strong routing signal; defaulting to digest.",
            confidence=0.50,
        )
        self._log(msg, result, "default")
        return result

    @staticmethod
    def _make(action: str, message_type: str, reason: str, confidence: float) -> Dict:
        return {
            "action": action,
            "message_type": message_type,
            "reason": reason,
            "confidence": confidence,
            "evidence_message_ids": "none",
        }

    @staticmethod
    def _log(msg: Dict, result: Dict, rule: str):
        logger.info(
            "FALLBACK[%s] msg=%s -> action=%s type=%s conf=%.2f",
            rule, msg.get("message_id"), result["action"], result["message_type"], result["confidence"]
        )
