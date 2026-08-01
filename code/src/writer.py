"""
writer.py — Write output.csv with the required column order.

Handles:
  - UTF-8 encoding
  - Proper CSV quoting for reason text (may contain commas, newlines)
  - Exact column order: message_id,action,message_type,reason,confidence,evidence_message_ids
"""

import csv
import os
from pathlib import Path
from typing import Dict, List

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


def write_output(rows: List[Dict], output_path: str | Path):
    """
    Write the prediction rows to output.csv.

    Each row dict must have keys: message_id, action, message_type, reason, confidence, evidence_message_ids.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            # Normalize confidence to float, clamp to [0, 1]
            try:
                conf = float(row.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
            except (TypeError, ValueError):
                conf = 0.5

            writer.writerow({
                "message_id": row.get("message_id", ""),
                "action": row.get("action", "digest"),
                "message_type": row.get("message_type", "unknown"),
                "reason": str(row.get("reason", "")).strip(),
                "confidence": round(conf, 4),
                "evidence_message_ids": str(row.get("evidence_message_ids", "none")).strip() or "none",
            })
