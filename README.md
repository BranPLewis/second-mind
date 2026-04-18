# Second-Mind

Simple local web UI for PCB component detection + Llama 3 learning notes.

## What was added

- `vision_center_object.py`: reusable detector that returns all component detections.
- `web_ui.py`: Flask app with `/api/analyze` and `/api/explain_component` endpoints.
- `component_info.py`: SnapEDA/Octopart/ComponentSearchEngine scraping + CSV recall store.
- `templates/index.html`: upload-based UI with per-component selection and source list.
- `requirements.txt`: Python dependencies.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set model paths:

```bash
export YOLO_MODEL_PATH="YOLO_Runs/fpic_seg_v8_s_768/weights/best.pt"
export LLAMA_MODEL_PATH="/absolute/path/to/your/llama-3-model.gguf"
```

3. Run the web app:

```bash
python web_ui.py
```

4. Open:

`http://localhost:8000`

## Notes

- Llama runs fully local through `llama-cpp-python` and a local `.gguf` model file.
- If startup errors appear in the page, verify paths and installed packages.

## Source-Restricted Recall (SnapEDA + Octopart + ComponentSearchEngine)

- Component source notes are scraped only from:
  - `snapeda.com`
  - `octopart.com`
  - `componentsearchengine.com`
- Notes are persisted to CSV at `data/component_sources.csv`.
- When you request explanation for a component, the app:
  1. Recalls cached notes from CSV.
  2. If none exist, scrapes SnapEDA/Octopart/ComponentSearchEngine and appends to CSV.
  3. Feeds those source notes to the local LLM for richer explanations.

## Download a GGUF model once (while online)

You can download a Llama 3 GGUF model before going offline.

Example with `huggingface-cli`:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download bartowski/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

Then set:

```bash
export LLAMA_MODEL_PATH="$(pwd)/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
```
