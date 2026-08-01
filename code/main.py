"""
main.py — CLI entry point for the Message Notification Router.

Usage:
    python main.py [--dataset-dir PATH] [--output PATH] [--log-level LEVEL]

Loads all dataset files, runs the pipeline for every row in messages.csv,
writes output/output.csv, and prints a summary.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (one level up from code/)
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Add code/ to path so src imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.loader import DataStore
from src.context import build_context
from src.retriever import retrieve_candidates, candidate_ids
from src.media import load_media_part
from src.prompt import SYSTEM_PROMPT, build_user_turn
from src.model_client import ModelClient
from src.validator import validate_and_repair, VALID_ACTIONS, VALID_MESSAGE_TYPES
from src.fallback import FallbackClassifier
from src.writer import write_output


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(
    messages,
    store: DataStore,
    model_client: ModelClient,
    fallback: FallbackClassifier,
    verbose: bool = True,
) -> list:
    """
    Run the full pipeline for a list of message rows.
    Returns a list of output row dicts.
    """
    history_ids = set(store.message_history_index.keys())
    results = []

    for i, msg in enumerate(messages, 1):
        msg_id = msg.get("message_id", f"row_{i}")
        if verbose:
            print(f"  [{i}/{len(messages)}] {msg_id} ...", end=" ", flush=True)

        t0 = time.time()
        try:
            # Step 1: Build context
            ctx = build_context(msg, store)

            # Step 2: Retrieve candidates
            candidates = retrieve_candidates(msg, store)
            cand_ids = candidate_ids(candidates)

            # Step 3: Load media
            media_part = load_media_part(msg, store)
            media_parts = [media_part] if media_part else None

            # Step 4: Build prompt
            user_turn = build_user_turn(ctx, candidates)

            # Step 5: Call model
            try:
                raw_result = model_client.generate(
                    system_prompt=SYSTEM_PROMPT,
                    user_turn_text=user_turn,
                    media_parts=media_parts,
                )

                # Step 6: Validate (and re-prompt once if needed)
                validated = validate_and_repair(
                    raw_result,
                    cand_ids,
                    history_ids,
                    model_client,
                    SYSTEM_PROMPT,
                    user_turn,
                    media_parts=media_parts,
                )

                if validated is not None:
                    result = {**validated, "message_id": msg_id}
                else:
                    # Validation failed twice — use fallback
                    fb = fallback.classify(msg, ctx, reason_prefix="validation_failed")
                    result = {**fb, "message_id": msg_id}

            except Exception as e:
                logging.getLogger(__name__).error("Model error for %s: %s", msg_id, e)
                fb = fallback.classify(msg, ctx, reason_prefix=f"api_error:{type(e).__name__}")
                result = {**fb, "message_id": msg_id}

        except Exception as e:
            logging.getLogger(__name__).error("Pipeline error for %s: %s", msg_id, e)
            # Minimal fallback with no context
            fb = {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "unknown",
                "reason": f"[FALLBACK] Pipeline error: {type(e).__name__}",
                "confidence": 0.50,
                "evidence_message_ids": "none",
            }
            fallback.fallback_count += 1
            result = fb

        elapsed = time.time() - t0
        if verbose:
            print(f"{result['action']}/{result['message_type']} conf={result['confidence']:.2f} ({elapsed:.1f}s)")

        results.append(result)

    return results


def print_summary(results: list, fallback: FallbackClassifier, model_client: ModelClient):
    """Print a summary of the run."""
    from collections import Counter
    actions = Counter(r["action"] for r in results)
    types = Counter(r["message_type"] for r in results)
    confs = [float(r["confidence"]) for r in results if r.get("confidence") is not None]
    avg_conf = sum(confs) / len(confs) if confs else 0

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total messages processed: {len(results)}")
    print(f"Fallback (fallback.py) uses: {fallback.fallback_count}")
    print(f"Model call breakdown: {model_client.summary()}")
    print(f"Avg confidence: {avg_conf:.3f}")
    print("\nAction distribution:")
    for action in ["notify", "digest", "mute"]:
        count = actions.get(action, 0)
        pct = 100 * count / len(results) if results else 0
        print(f"  {action:8s}: {count:4d} ({pct:.1f}%)")
    print("\nMessage type distribution:")
    for mtype, count in types.most_common():
        pct = 100 * count / len(results) if results else 0
        print(f"  {mtype:20s}: {count:4d} ({pct:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="WhatsApp Message Notification Router"
    )
    parser.add_argument(
        "--output", default="output/output.csv",
        help="Output CSV path (default: output/output.csv)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    output_path = Path(__file__).parent / args.output

    print("Loading dataset...")
    store = DataStore().load()
    print(f"  {len(store.messages)} messages to process")
    print(f"  {len(store.message_history)} historical messages indexed")

    model_client = ModelClient()
    fallback = FallbackClassifier()

    if model_client.groq_available:
        print(f"Groq enabled ({model_client.active_groq_model}) — text messages will use Groq")
    else:
        print("Groq not configured — set GROQ_API_KEY in .env to enable")
    print(f"\nRunning pipeline with: {model_client.summary()}")
    print("-" * 60)

    results = run_pipeline(store.messages, store, model_client, fallback, verbose=True)

    print(f"\nWriting output to {output_path} ...")
    write_output(results, output_path)
    print(f"Done. {len(results)} rows written.")

    print_summary(results, fallback, model_client)

    return 0


if __name__ == "__main__":
    sys.exit(main())
