"""
context.py — Assemble a structured context object for each target message.

Gathers sender/group/business relationship, user DND + engagement stats,
notification load, and media broadcast signal for the model prompt.
"""

from datetime import datetime, time
from typing import Dict, Optional, Any

from .loader import DataStore


def _parse_dnd_window(dnd_str: str):
    """
    Parse DND window like '22:00-07:00' into (start_time, end_time).
    Returns (None, None) if unparseable.
    """
    try:
        parts = dnd_str.strip().split("-")
        if len(parts) != 2:
            return None, None
        start = datetime.strptime(parts[0].strip(), "%H:%M").time()
        end = datetime.strptime(parts[1].strip(), "%H:%M").time()
        return start, end
    except Exception:
        return None, None


def _is_in_dnd_window(msg_time: datetime, dnd_str: str) -> bool:
    """Return True if msg_time falls inside the user's DND window."""
    start, end = _parse_dnd_window(dnd_str)
    if start is None:
        return False
    t = msg_time.time()
    if start <= end:
        return start <= t <= end
    else:
        # Crosses midnight e.g. 22:00-07:00
        return t >= start or t <= end


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def build_context(msg: Dict, store: DataStore) -> Dict:
    """
    Build a rich context dict for a target message row.

    Returns a dict with:
      - message: core fields
      - user: DND info + engagement
      - is_in_dnd: bool
      - sender_type: 'group'|'business'|'personal'
      - group_info: dict if group
      - business_info: dict if business
      - personal_sender_info: dict if personal
      - today_notification_load: dict
      - media_reuse_count: int
    """
    user_id = msg["user_id"]
    user = store.users.get(user_id, {})

    # Parse message time
    try:
        msg_time = datetime.strptime(msg["created_at"], "%Y-%m-%d %H:%M")
    except Exception:
        msg_time = datetime.now()

    msg_date = msg_time.strftime("%Y-%m-%d")
    dnd_window = user.get("do_not_disturb_window", "")
    is_in_dnd = _is_in_dnd_window(msg_time, dnd_window) if dnd_window else False

    user_info = {
        "user_id": user_id,
        "do_not_disturb_window": dnd_window,
        "is_in_dnd": is_in_dnd,
        "messages_opened_30d": _safe_int(user.get("messages_opened_30d", 0)),
        "messages_replied_30d": _safe_int(user.get("messages_replied_30d", 0)),
        "notifications_dismissed_30d": _safe_int(user.get("notifications_dismissed_30d", 0)),
        "messages_reported_30d": _safe_int(user.get("messages_reported_30d", 0)),
    }

    # Engagement rate (replies / max(opens, 1))
    opens = user_info["messages_opened_30d"]
    replies = user_info["messages_replied_30d"]
    dismissals = user_info["notifications_dismissed_30d"]
    user_info["engagement_rate_30d"] = round(replies / max(opens, 1), 3)
    user_info["dismissal_rate_30d"] = round(dismissals / max(opens + dismissals, 1), 3)

    # Today notification load
    daily_key = (user_id, msg_date)
    daily = store.daily_notification_summary.get(daily_key, {})
    today_notification_load = {
        "date": msg_date,
        "notifications_sent_today": _safe_int(daily.get("notifications_sent", 0)),
        "notifications_dismissed_today": _safe_int(daily.get("notifications_dismissed", 0)),
    }

    # Media reuse count
    media_id = msg.get("media_id", "")
    media_reuse_count = store.media_reuse_count.get(media_id, 0) if media_id else 0

    # Determine sender type and build relationship info
    conv_type = msg.get("conversation_type", "")
    group_id = msg.get("group_id", "")
    business_id = msg.get("business_id", "")
    sender_user_id = msg.get("sender_user_id", "")

    group_info = {}
    business_info = {}
    personal_sender_info = {}

    if conv_type == "group" and group_id:
        group = store.groups.get(group_id, {})
        member_key = (group_id, user_id)
        member = store.group_members.get(member_key, {})
        sender_member_key = (group_id, sender_user_id) if sender_user_id else None
        sender_member = store.group_members.get(sender_member_key, {}) if sender_member_key else {}

        group_info = {
            "group_id": group_id,
            "group_name": group.get("group_name", ""),
            "group_type": group.get("group_type", ""),
            "member_count": _safe_int(group.get("member_count", 0)),
            "messages_30d": _safe_int(group.get("messages_30d", 0)),
            # User's relationship to this group
            "user_role": member.get("role", "unknown"),
            "user_group_muted": member.get("group_muted_by_user", "0") == "1",
            "user_messages_sent_30d": _safe_int(member.get("messages_sent_30d", 0)),
            "user_messages_read_30d": _safe_int(member.get("messages_read_30d", 0)),
            "user_replies_sent_30d": _safe_int(member.get("replies_sent_30d", 0)),
            "user_notifications_dismissed_30d": _safe_int(member.get("notifications_dismissed_30d", 0)),
            # Sender's role in the group
            "sender_role": sender_member.get("role", "member"),
            "sender_user_id": sender_user_id,
        }

    elif conv_type == "business" and business_id:
        biz = store.business_accounts.get(business_id, {})
        biz_history_key = (user_id, business_id)
        biz_history = store.user_business_history.get(biz_history_key, {})

        domain_match = biz.get("official_domain", "").strip() == biz.get("domain_used_by_sender", "").strip()

        business_info = {
            "business_id": business_id,
            "display_name": biz.get("display_name", ""),
            "brand_name": biz.get("brand_name", ""),
            "category": biz.get("category", ""),
            "verified": biz.get("verified", "0") == "1",
            "official_domain": biz.get("official_domain", ""),
            "domain_used_by_sender": biz.get("domain_used_by_sender", ""),
            "domain_match": domain_match,
            "account_age_days": _safe_int(biz.get("account_age_days", 0)),
            "domain_used_by_sender_age_days": _safe_int(biz.get("domain_used_by_sender_age_days", 0)),
            "messages_sent_30d": _safe_int(biz.get("messages_sent_30d", 0)),
            "user_reports_30d": _safe_int(biz.get("user_reports_30d", 0)),
            # User's history with this business
            "user_knows_account": biz_history.get("why_user_knows_account", "unknown"),
            "user_allows_promotions": biz_history.get("allows_promotions", "0") == "1",
            "user_opted_out_promotions": bool(biz_history.get("promotions_opted_out_at", "")),
            "user_activity_count_180d": _safe_int(biz_history.get("activity_count_180d", 0)),
            "user_messages_opened_30d": _safe_int(biz_history.get("messages_opened_30d", 0)),
            "user_messages_dismissed_30d": _safe_int(biz_history.get("messages_dismissed_30d", 0)),
            "user_has_relationship": bool(biz_history),
        }

    elif conv_type == "personal" and sender_user_id:
        sender_user = store.users.get(sender_user_id, {})
        personal_sender_info = {
            "sender_user_id": sender_user_id,
            "sender_exists_in_users": bool(sender_user),
        }

    return {
        "message": {
            "message_id": msg["message_id"],
            "user_id": user_id,
            "conversation_type": conv_type,
            "created_at": msg["created_at"],
            "message_text": msg.get("message_text", ""),
            "media_type": msg.get("media_type", ""),
            "media_id": media_id,
            "forwarded_count": _safe_int(msg.get("forwarded_count", 0)),
        },
        "user": user_info,
        "is_in_dnd": is_in_dnd,
        "sender_type": conv_type,
        "group_info": group_info,
        "business_info": business_info,
        "personal_sender_info": personal_sender_info,
        "today_notification_load": today_notification_load,
        "media_reuse_count": media_reuse_count,
    }
