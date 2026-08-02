# Message Notification Router — Code

AI-powered WhatsApp message routing system for the HackerRank Orchestrate challenge.

## Overview

For every incoming message in `dataset/messages.csv`, this system decides:
- **notify** — interrupt the user now
- **digest** — show later in a batch
- **mute** — suppress (spam, scam, low-value, opted-out)

It uses a **dual-provider** architecture with structured JSON output, retrieval-augmented context from historical messages, and a deterministic fallback for reliability:
- **Groq** (`llama-3.3-70b-versatile` as primary, with `llama-3.1-8b-instant` as an internal fallback) handles all text-only messages for high speed and generous RPM quota.
- **Gemini** (`gemini-3.5-flash-lite` as primary, with `gemini-3.5-flash` as secondary) is reserved exclusively for image and voice-note messages, leveraging its native multimodal capabilities.

## Project Structure

```
code/
├── main.py                  # Main pipeline entry point
├── requirements.txt         # Python dependencies
├── README.md
├── output/
│   └── output.csv           # Generated predictions
├── src/
│   ├── loader.py            # CSV loading + indexing
│   ├── context.py           # Per-message context assembly
│   ├── retriever.py         # Historical message retrieval + scoring
│   ├── media.py             # Image/audio media loading
│   ├── prompt.py            # System prompt + user turn builder
│   ├── model_client.py      # Dual-provider client (Groq for text, Gemini for media)
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

Copy `.env.example` to `.env` in the **repo root** (one level up from `code/`):

```bash
cp ../.env.example ../.env
```

Edit `../.env` and fill in your API keys and model choices:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODEL=gemini-3.5-flash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
```

> **Note on `.env` location & loading**: `.env` lives at the **repo root** (`../.env` relative to `code/`). In `main.py`, `load_dotenv(_REPO_ROOT / ".env")` explicitly resolves and loads this root file before starting the pipeline. Never commit `.env` (it is gitignored).

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
| `retriever.py` | Filters history by user+sender/group/business, scores by recency (exp decay) + keyword Jaccard, returns top candidates |
| `media.py` | Reads image/audio bytes, returns `types.Part` for inline multimodal call |
| `prompt.py` | `SYSTEM_PROMPT` constant + compact JSON user-turn builder |
| `model_client.py` | Dual-provider client: `media_type` determines provider. Text-only messages route to Groq (`llama-3.3-70b-versatile` -> `llama-3.1-8b-instant` fallback on 429/quota errors). Image/voice messages route to Gemini (`gemini-3.5-flash-lite` -> `gemini-3.5-flash` fallback). If Groq fails after trying both models, execution falls back to `fallback.py` (NEVER to Gemini, preserving Gemini's quota for multimodal requests). |
| `validator.py` | Validates action/type enums, confidence range, evidence IDs; re-prompts once on failure |
| `fallback.py` | Deterministic backstop: handles rule-based checks (injection detection, OTP/scam patterns, DND, group mute, business verification) as well as transient API connection errors (`APIConnectionError`) |
| `writer.py` | Writes `output.csv` with exact column order, UTF-8, QUOTE_ALL |

## Security Notes

- `message_text` and media content are treated as untrusted data, never as instructions
- Prompt injection attempts in message content → classified as `scam`
- API keys loaded only from environment variables — never hardcoded
- `.env` is gitignored
