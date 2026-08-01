"""
loader.py — Load all dataset CSVs and build indexed data structures.

Handles embedded newlines in message_text by using csv.DictReader which
correctly handles RFC-4180 quoted fields with embedded newlines.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Absolute path to dataset directory
DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"


def _open_csv(filename: str) -> List[Dict]:
    """Open a CSV file from the dataset directory and return all rows as dicts."""
    path = DATASET_DIR / filename
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


class DataStore:
    """Central data store with all dataset tables, indexed for fast lookup."""

    def __init__(self):
        # Raw tables
        self.messages: List[Dict] = []
        self.sample_messages: List[Dict] = []

        # Indexed lookups
        self.users: Dict[str, Dict] = {}                         # user_id -> row
        self.groups: Dict[str, Dict] = {}                        # group_id -> row
        self.group_members: Dict[Tuple[str, str], Dict] = {}     # (group_id, user_id) -> row
        self.group_members_by_user: Dict[str, List[Dict]] = defaultdict(list)  # user_id -> [rows]
        self.business_accounts: Dict[str, Dict] = {}             # business_id -> row
        self.user_business_history: Dict[Tuple[str, str], Dict] = {}  # (user_id, business_id) -> row
        self.message_history: List[Dict] = []
        self.message_history_by_user: Dict[str, List[Dict]] = defaultdict(list)   # user_id -> [rows]
        self.message_history_index: Dict[str, Dict] = {}         # message_id -> row
        self.message_events: Dict[Tuple[str, str], Dict] = {}    # (user_id, message_id) -> row
        self.images: Dict[str, Dict] = {}                        # image_id -> row
        self.voice_notes: Dict[str, Dict] = {}                   # voice_note_id -> row
        self.daily_notification_summary: Dict[Tuple[str, str], Dict] = {}  # (user_id, date) -> row
        self.daily_by_user: Dict[str, List[Dict]] = defaultdict(list)  # user_id -> sorted list

        # Media reuse count: media_id -> count of messages.csv rows with this media_id
        self.media_reuse_count: Dict[str, int] = defaultdict(int)

    def load(self):
        """Load all dataset files."""
        # Users
        for row in _open_csv("users.csv"):
            self.users[row["user_id"]] = row

        # Groups
        for row in _open_csv("groups.csv"):
            self.groups[row["group_id"]] = row

        # Group members
        for row in _open_csv("group_members.csv"):
            key = (row["group_id"], row["user_id"])
            self.group_members[key] = row
            self.group_members_by_user[row["user_id"]].append(row)

        # Business accounts
        for row in _open_csv("business_accounts.csv"):
            self.business_accounts[row["business_id"]] = row

        # User business history
        for row in _open_csv("user_business_history.csv"):
            key = (row["user_id"], row["business_id"])
            self.user_business_history[key] = row

        # Message history
        self.message_history = _open_csv("message_history.csv")
        for row in self.message_history:
            self.message_history_by_user[row["user_id"]].append(row)
            self.message_history_index[row["message_id"]] = row

        # Message events
        for row in _open_csv("message_events.csv"):
            key = (row["user_id"], row["message_id"])
            self.message_events[key] = row

        # Images
        for row in _open_csv("images.csv"):
            self.images[row["image_id"]] = row

        # Voice notes
        for row in _open_csv("voice_notes.csv"):
            self.voice_notes[row["voice_note_id"]] = row

        # Daily notification summary
        for row in _open_csv("daily_notification_summary.csv"):
            key = (row["user_id"], row["date"])
            self.daily_notification_summary[key] = row
            self.daily_by_user[row["user_id"]].append(row)

        # Messages (main prediction targets)
        self.messages = _open_csv("messages.csv")

        # Compute media reuse counts across messages.csv
        for row in self.messages:
            if row.get("media_id"):
                self.media_reuse_count[row["media_id"]] += 1

        # Sample messages (evaluation only)
        try:
            self.sample_messages = _open_csv("sample_messages.csv")
        except FileNotFoundError:
            self.sample_messages = []

        return self
