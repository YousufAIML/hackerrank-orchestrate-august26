# Message Notification Router — Code

AI-powered WhatsApp message routing system for the HackerRank Orchestrate challenge.

## Overview

For every incoming message in `dataset/messages.csv`, this system decides:
- **notify** — interrupt the user now
- **digest** — show later in a batch
- **mute** — suppress (spam, scam, low-value, opted-out)

It uses Gemini (via `google-genai` SDK) with structured JSON output, retrieval-augmented context from historical messages, and a deterministic fallback for reliability.

## Project Structure

```
code/
├── main.py                  # Main pipeline entry point
├── requirements.txt         # Python dependencies
├── README.md
├── .env.example             # Template for required env vars
├── output/
│   └── output.csv           # Generated predictions
├── src/
│   ├── loader.py            # CSV loading + indexing
│   ├── context.py           # Per-message context assembly
│   ├── retriever.py         # Historical message retrieval + scoring
│   ├── media.py             # Image/audio media loading
│   ├── prompt.py            # System prompt + user turn builder
│   ├── model_client.py      # Gemini API client with retry + fallback
│   ├── validator.py         # Response validation + re-prompt
│   ├── fallback.py          # Deterministic rule-based backstop
│   └── writer.py            # output.csv writer
└── evaluation/
    └── main.py              # Evaluation harness vs sample_messages.csv
```

## Setup

### 1. Prerequisites

- Python 3.10+
- A virtual environment (recommended)

### 2. Install dependencies

```bash
cd code
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` in the **repo root** (not inside `code/`):

```bash
cp ../.env.example ../.env
```

Edit `../.env` and fill in your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_FALLBACK_MODEL=gemini-2.5-flash-lite
```

> **Never commit `.env`.** It is listed in `.gitignore`.

## Running the Pipeline

From the `code/` directory (with venv activated):

```bash
python main.py
```

This processes all rows in `dataset/messages.csv` and writes predictions to `output/output.csv`.

### Options

```
python main.py --help

Options:
  --output PATH      Output CSV path (default: output/output.csv)
  --log-level LEVEL  DEBUG|INFO|WARNING|ERROR (default: INFO)
```

### Example

```bash
python main.py --output output/output.csv --log-level INFO
```

## Running the Evaluation Harness

Evaluates the pipeline against `dataset/sample_messages.csv` (known labels):

```bash
python evaluation/main.py
```

### Options

```
python evaluation/main.py --help

Options:
  --log-level LEVEL  DEBUG|INFO|WARNING|ERROR (default: WARNING)
  --examples N       Number of example outputs to print (default: 5)
```

### Example

```bash
python evaluation/main.py --examples 10 --log-level INFO
```

This prints:
- Overall action accuracy (%)
- Overall message_type accuracy (%)
- Per-action and per-type accuracy breakdown
- Action confusion matrix
- N sample reason/evidence outputs for manual review

## Architecture Notes

| Module | Purpose |
|--------|---------|
| `loader.py` | Loads all 12 dataset CSVs, builds O(1) index dicts |
| `context.py` | Assembles sender/group/business relationship, DND, engagement, notification load, media broadcast count |
| `retriever.py` | Filters history by user+sender/group/business, scores by recency (exp decay) + keyword Jaccard, returns top-8 |
| `media.py` | Reads image/audio bytes, returns `types.Part` for inline multimodal call |
| `prompt.py` | `SYSTEM_PROMPT` constant + compact JSON user-turn builder |
| `model_client.py` | Gemini client: temp=0, JSON schema, 4-attempt backoff on 429/5xx, 404→fallback model |
| `validator.py` | Validates action/type enums, confidence range, evidence IDs; re-prompts once on failure |
| `fallback.py` | Deterministic rules: injection detection, OTP/scam patterns, DND, group mute, business verification |
| `writer.py` | Writes `output.csv` with exact column order, UTF-8, QUOTE_ALL |

## Security Notes

- `message_text` and media content are treated as untrusted data, never as instructions
- Prompt injection attempts in message content → classified as `scam`
- API key loaded only from environment variables — never hardcoded
- `.env` is gitignored
