"""
evaluation/main.py — Evaluation harness for sample_messages.csv.

Runs the full pipeline against dataset/sample_messages.csv (which has known labels),
compares predicted action and message_type against ground truth, and prints accuracy
metrics plus example outputs.

Usage:
    python evaluation/main.py [--log-level LEVEL] [--examples N]
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Ensure code/ is on the path
CODE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_DIR))

from dotenv import load_dotenv
_REPO_ROOT = CODE_DIR.parent
load_dotenv(_REPO_ROOT / ".env")

from src.loader import DataStore
from src.context import build_context
from src.retriever import retrieve_candidates, candidate_ids
from src.media import load_media_part
from src.prompt import SYSTEM_PROMPT, build_user_turn
from src.model_client import ModelClient
from src.validator import validate_and_repair
from src.fallback import FallbackClassifier


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_eval(args):
    logger = logging.getLogger(__name__)

    store = DataStore().load()
    if not store.sample_messages:
        print("ERROR: No sample_messages.csv found or it is empty.")
        sys.exit(1)

    sample = store.sample_messages
    print(f"Evaluating on {len(sample)} sample messages...")

    model_client = ModelClient()
    fallback = FallbackClassifier()
    history_ids = set(store.message_history_index.keys())

    predictions = []
    ground_truths = []

    for i, msg in enumerate(sample, 1):
        msg_id = msg["message_id"]
        gt_action = msg.get("action", "").strip()
        gt_type = msg.get("message_type", "").strip()

        print(f"  [{i}/{len(sample)}] {msg_id} (gt: {gt_action}/{gt_type}) ...", end=" ", flush=True)

        # Build a message dict without the label columns (so we don't leak)
        clean_msg = {
            k: v for k, v in msg.items()
            if k not in ("action", "message_type", "reason", "confidence", "evidence_message_ids")
        }

        try:
            ctx = build_context(clean_msg, store)
            candidates = retrieve_candidates(clean_msg, store)
            cand_ids = candidate_ids(candidates)
            media_part = load_media_part(clean_msg, store)
            media_parts = [media_part] if media_part else None
            user_turn = build_user_turn(ctx, candidates)

            try:
                raw_result = model_client.generate(
                    system_prompt=SYSTEM_PROMPT,
                    user_turn_text=user_turn,
                    media_parts=media_parts,
                )
                validated = validate_and_repair(
                    raw_result, cand_ids, history_ids,
                    model_client, SYSTEM_PROMPT, user_turn, media_parts=media_parts,
                )
                if validated is not None:
                    result = {**validated, "message_id": msg_id}
                else:
                    fb = fallback.classify(clean_msg, ctx, "eval_validation_failed")
                    result = {**fb, "message_id": msg_id}

            except Exception as e:
                logger.error("Model error for %s: %s", msg_id, e)
                fb = fallback.classify(clean_msg, ctx, f"api_error:{type(e).__name__}")
                result = {**fb, "message_id": msg_id}

        except Exception as e:
            logger.error("Pipeline error for %s: %s", msg_id, e)
            result = {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "unknown",
                "reason": f"[FALLBACK] Error: {e}",
                "confidence": 0.5,
                "evidence_message_ids": "none",
            }
            fallback.fallback_count += 1

        pred_action = result.get("action", "")
        pred_type = result.get("message_type", "")
        action_correct = pred_action == gt_action
        type_correct = pred_type == gt_type

        print(f"pred: {pred_action}/{pred_type} "
              f"[action:{'✓' if action_correct else '✗'} type:{'✓' if type_correct else '✗'}]")

        predictions.append(result)
        ground_truths.append({
            "message_id": msg_id,
            "gt_action": gt_action,
            "gt_type": gt_type,
            "gt_reason": msg.get("reason", ""),
            "gt_evidence": msg.get("evidence_message_ids", ""),
            "pred_action": pred_action,
            "pred_type": pred_type,
            "pred_reason": result.get("reason", ""),
            "pred_evidence": result.get("evidence_message_ids", "none"),
            "action_correct": action_correct,
            "type_correct": type_correct,
        })

    # ---- Accuracy report ----
    n = len(ground_truths)
    action_correct = sum(1 for r in ground_truths if r["action_correct"])
    type_correct = sum(1 for r in ground_truths if r["type_correct"])
    both_correct = sum(1 for r in ground_truths if r["action_correct"] and r["type_correct"])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples:           {n}")
    print(f"Action accuracy:         {action_correct}/{n} = {100*action_correct/n:.1f}%")
    print(f"Message type accuracy:   {type_correct}/{n} = {100*type_correct/n:.1f}%")
    print(f"Both correct:            {both_correct}/{n} = {100*both_correct/n:.1f}%")
    print(f"Fallback uses:           {fallback.fallback_count}")
    print(f"Model(s):                {model_client.summary()}")
    if model_client.switched_to_fallback:
        print(f"  ⚠ Gemini switched to fallback model during eval")

    # Per-action breakdown
    print("\nPer-action accuracy:")
    gt_actions = set(r["gt_action"] for r in ground_truths)
    for act in sorted(gt_actions):
        subset = [r for r in ground_truths if r["gt_action"] == act]
        correct = sum(1 for r in subset if r["action_correct"])
        print(f"  {act:8s}: {correct}/{len(subset)} = {100*correct/len(subset):.1f}%")

    # Per-type breakdown
    print("\nPer-message-type accuracy:")
    gt_types = set(r["gt_type"] for r in ground_truths)
    for mtype in sorted(gt_types):
        subset = [r for r in ground_truths if r["gt_type"] == mtype]
        correct = sum(1 for r in subset if r["type_correct"])
        print(f"  {mtype:20s}: {correct}/{len(subset)} = {100*correct/len(subset):.1f}%")

    # Confusion matrix for action
    print("\nAction confusion matrix (GT rows, PRED cols):")
    actions_list = ["notify", "digest", "mute"]
    gt_pred_label = "GT\\PRED"
    header = f"{gt_pred_label:10s}" + "".join(f"{a:10s}" for a in actions_list)
    print(header)
    for act_gt in actions_list:
        row_str = f"{act_gt:10s}"
        for act_pred in actions_list:
            count = sum(
                1 for r in ground_truths
                if r["gt_action"] == act_gt and r["pred_action"] == act_pred
            )
            row_str += f"{count:10d}"
        print(row_str)

    # Sample examples
    examples = min(args.examples, n)
    print(f"\n---- {examples} Example Outputs ----")
    for r in ground_truths[:examples]:
        print(f"\n[{r['message_id']}]")
        print(f"  GT:   action={r['gt_action']}, type={r['gt_type']}")
        print(f"  PRED: action={r['pred_action']}, type={r['pred_type']}")
        print(f"  GT reason:   {r['gt_reason'][:120]}")
        print(f"  PRED reason: {r['pred_reason'][:120]}")
        print(f"  GT evidence:   {r['gt_evidence']}")
        print(f"  PRED evidence: {r['pred_evidence']}")

    print("\n" + "=" * 60)

    return ground_truths


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation harness for sample_messages.csv"
    )
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: WARNING)")
    parser.add_argument("--examples", type=int, default=5,
                        help="Number of example outputs to print (default: 5)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_eval(args)


if __name__ == "__main__":
    main()
