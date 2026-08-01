"""
media.py — Load media files (images, audio) as inline bytes parts for the model.

For media_type == "image": read the file at images.csv's file_path as bytes.
For media_type == "voice": read the file at voice_notes.csv's file_path as bytes.
Both are sent inline in the same model call — no separate OCR/transcription step.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any

from google.genai import types

from .loader import DataStore

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"


def load_media_part(msg: dict, store: DataStore) -> Optional[Any]:
    """
    Return a google.genai types.Part for the message's media, or None.

    - media_type == "image" -> inline image/jpeg Part
    - media_type == "voice" -> inline audio/mpeg Part
    - otherwise -> None
    """
    media_type = msg.get("media_type", "").strip()
    media_id = msg.get("media_id", "").strip()

    if not media_type or not media_id:
        return None

    if media_type == "image":
        image_row = store.images.get(media_id)
        if not image_row:
            logger.warning("Image ID %s not found in images.csv", media_id)
            return None
        file_path = DATASET_DIR / image_row["file_path"]
        mime_type = "image/jpeg"

    elif media_type == "voice":
        vn_row = store.voice_notes.get(media_id)
        if not vn_row:
            logger.warning("Voice note ID %s not found in voice_notes.csv", media_id)
            return None
        file_path = DATASET_DIR / vn_row["file_path"]
        mime_type = "audio/mpeg"

    else:
        logger.debug("Unknown media_type %r for message %s", media_type, msg.get("message_id"))
        return None

    if not file_path.exists():
        logger.warning("Media file not found: %s", file_path)
        return None

    try:
        data = file_path.read_bytes()
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    except Exception as e:
        logger.error("Failed to read media file %s: %s", file_path, e)
        return None
