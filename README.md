# Second-Mind

Flask web app for PCB component analysis with:
- image upload / live camera capture
- detection via external vision endpoint
- per-component LLM explanations
- source scraping + CSV caching
- citations panel for dataset attribution

## Current App Structure

- `app.py` - Flask server and API routes (`/api/analyze`, `/api/explain_component`)
- `component_info.py` - scraping, relevance filtering, CSV cache (`data/component_sources.csv`)
- `templates/index.html` - active frontend UI used by Flask
- `citations.txt` - static citation blocks rendered in UI
- `site/src/imports/` - static image assets served at `/assets/...`

Note: the React/Vite code under `site/` is not the active runtime UI. The deployed app uses `templates/index.html`.

## Requirements

- Python 3.10+
- pip
- Environment variables:
  - `MODAL_URL` (required): URL for your vision inference service
  - `GROQ_API_KEY` (optional): enables full LLM explanations

If `GROQ_API_KEY` is missing, the app still runs and still returns scraped source links.

## Run Locally (Flask)

1) Clone and enter project

```bash
git clone <your-repo-url>
cd PCB_VISION
```

2) Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

3) Install dependencies

```bash
pip install -r requirements.txt
```

4) Set environment variables

```bash
export MODAL_URL="https://<your-modal-endpoint>"
export GROQ_API_KEY="<optional-groq-key>"
```

5) Start Flask server

```bash
python app.py
```

6) Open in browser

```text
http://localhost:8000
```

## What the APIs Do

- `POST /api/analyze`
  - Sends uploaded image to `MODAL_URL`
  - Returns detected components + annotated image

- `POST /api/explain_component`
  - Recalls relevant cached sources from CSV
  - Scrapes/adds more sources when needed
  - Returns explanation + source links

- `POST /api/save_state`, `GET /api/last_state`
  - Persists client session state in `data/user_states/`

## Data Files Created at Runtime

- `data/component_sources.csv` - scraped source cache
- `data/user_states/*.json` - per-user UI state

## Troubleshooting

- "Failed to connect to vision processor": verify `MODAL_URL` is set and reachable.
- No LLM explanation: set `GROQ_API_KEY`; source links should still appear.
- If sources appear empty after edits, ensure CSV header/schema is valid and app was restarted.
