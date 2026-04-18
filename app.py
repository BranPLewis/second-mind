import base64
import json
import os
import re
import uuid
from pathlib import Path
import requests

from flask import Flask, jsonify, render_template, request, send_from_directory
from openai import OpenAI

from component_info import (
    ComponentKnowledgeStore,
    ComponentScraper,
    _entry_relevant_to_label,
    _site_key,
    format_source_context,
)

MAX_UPLOAD_MB = 10
MAX_REQUEST_MB = 30
STATE_COOKIE_NAME = "second_mind_client_id"
STATE_COOKIE_MAX_AGE = 60 * 60 * 24 * 30


class CloudLlamaTutor:
    def __init__(self):
        # Groq uses the OpenAI SDK structure for compatibility
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is missing.")

        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def explain_detection(
        self,
        label: str,
        confidence: float,
        center_x: int,
        center_y: int,
        source_context: str = "",
    ) -> str:
        # The system prompt dictates behavior, rules, and formatting
        system_prompt = (
            "You are an electronics and computer-vision tutor for beginners. "
            "Explain the detected PCB component for learning. Be practical, specific, and moderately detailed.\n"
            "You may use external source notes from any credible source available in context, including generic web references.\n"
            "If source notes conflict or are weak, state that clearly.\n\n"
            "FORMATTING RULES:\n"
            "Format your entire response in clean Markdown so it renders beautifully in a web UI. "
            "Use `###` headings for each section, `-` for bullet points, and `**` for emphasis. "
            "Do NOT include introductory or concluding conversational filler. Just return the structured content."
        )

        # The user prompt passes the dynamic data and the required structure
        user_prompt = (
            f"Detected object: {label}\n"
            f"Confidence: {confidence:.3f}\n"
            f"Object center in image: ({center_x}, {center_y})\n\n"
            f"Source Context:\n{source_context}\n\n"
            "Reply strictly with these 6 sections:\n"
            "### 1. Component Overview\n(name + brief description)\n"
            "### 2. Circuit Purpose\n"
            "### 3. Typical Applications\n"
            "### 4. Common Faults & Symptoms\n"
            "### 5. Manual Verification\n(markings, orientation, solder joints)\n"
            "### 6. Safety & Handling\n"
        )

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,  # Increased to prevent mid-sentence cutoffs
            temperature=0.3,
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_MB * 1024 * 1024
    figma_assets_dir = Path(__file__).resolve().parent / "site" / "src" / "imports"
    user_state_dir = Path(__file__).resolve().parent / "data" / "user_states"
    user_state_dir.mkdir(parents=True, exist_ok=True)

    scraper = ComponentScraper()
    knowledge_store = ComponentKnowledgeStore()
    tutor = None
    llm_startup_error = None
    startup_errors = []

    def _load_citations() -> list[str]:
        citations_path = Path(__file__).resolve().parent / "citations.txt"
        if not citations_path.exists():
            return []
        try:
            raw = citations_path.read_text(encoding="utf-8")
        except Exception:
            return []
        return [block.strip() for block in re.split(r"\n\s*\n", raw) if block.strip()]

    citation_blocks = _load_citations()

    try:
        tutor = CloudLlamaTutor()
    except Exception as exc:
        llm_startup_error = str(exc)
        startup_errors.append(f"Cloud Llama init error: {exc}")

    def _ensure_client_id() -> str:
        cookie_value = request.cookies.get(STATE_COOKIE_NAME, "")
        if re.fullmatch(r"[A-Za-z0-9-]{8,64}", cookie_value):
            return cookie_value
        return str(uuid.uuid4())

    def _state_path(client_id: str) -> Path:
        return user_state_dir / f"{client_id}.json"

    def _load_state(client_id: str) -> dict | None:
        path = _state_path(client_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_state(client_id: str, state: dict) -> None:
        path = _state_path(client_id)
        path.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            startup_errors=startup_errors,
            citations=citation_blocks,
        )

    @app.get("/assets/<path:filename>")
    def figma_assets(filename: str):
        return send_from_directory(figma_assets_dir, filename)

    @app.get("/api/last_state")
    def last_state():
        client_id = _ensure_client_id()
        state = _load_state(client_id)
        response = jsonify({"state": state})
        response.set_cookie(
            STATE_COOKIE_NAME,
            client_id,
            max_age=STATE_COOKIE_MAX_AGE,
            httponly=True,
            samesite="Lax",
        )
        return response

    @app.post("/api/save_state")
    def save_state():
        payload = request.get_json(silent=True) or {}
        allowed_keys = {
            "uploaded_image",
            "annotated_image",
            "components",
            "detected",
            "total_detections",
            "selected_label",
            "notes",
            "sources",
            "scrape_errors",
            "status",
            "meta_html",
            "last_detection_result",
            "last_llm_response",
        }
        state = {key: payload.get(key) for key in allowed_keys}
        client_id = _ensure_client_id()
        _save_state(client_id, state)
        response = jsonify({"saved": True})
        response.set_cookie(
            STATE_COOKIE_NAME,
            client_id,
            max_age=STATE_COOKIE_MAX_AGE,
            httponly=True,
            samesite="Lax",
        )
        return response

    @app.post("/api/analyze")
    def analyze():
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "Please choose an image file."}), 400

        raw_bytes = image_file.read()
        if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            return jsonify(
                {"error": f"File too large. Max size is {MAX_UPLOAD_MB} MB."}
            ), 413

        # Convert image to base64 to send to Modal
        b64_image = base64.b64encode(raw_bytes).decode("ascii")

        modal_url = os.getenv("MODAL_URL")
        if not modal_url:
            return jsonify({"error": "MODAL_URL environment variable is missing."}), 500

        try:
            # Send the image to the Serverless GPU
            modal_response = requests.post(
                modal_url,
                json={"image": b64_image},
                timeout=120,  # Allow time for Modal cold starts
            )
            modal_response.raise_for_status()

            # Return the processed data directly to the frontend
            return jsonify(modal_response.json())

        except requests.exceptions.RequestException as e:
            return jsonify(
                {"error": f"Failed to connect to vision processor: {str(e)}"}
            ), 502

    @app.post("/api/explain_component")
    def explain_component():
        payload = request.get_json(silent=True) or {}
        label = payload.get("label")
        confidence = payload.get("confidence")
        center = payload.get("center")

        if not label or confidence is None or not isinstance(center, dict):
            return jsonify({"error": "Missing component data for explanation."}), 400

        center_x = center.get("x")
        center_y = center.get("y")
        if center_x is None or center_y is None:
            return jsonify(
                {"error": "Missing center coordinates for explanation."}
            ), 400

        try:
            confidence_f = float(confidence)
            center_x_i = int(center_x)
            center_y_i = int(center_y)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid component data for explanation."}), 400

        recall_limit = payload.get("recall_limit", 10)
        try:
            recall_limit = max(1, min(int(recall_limit), 30))
        except (TypeError, ValueError):
            recall_limit = 10

        scrape_errors = []
        target_site_count = 4

        cached_entries, cached_site_keys = knowledge_store.recall_relevant_by_site(
            label,
            max_sites=target_site_count,
            max_entries=max(recall_limit * 2, 20),
        )

        # Fill cache up to 6 relevant unique sites.
        if len(cached_site_keys) < target_site_count:
            scraped_entries, fill_errors = scraper.scrape_component(label)
            scrape_errors.extend(fill_errors)
            if scraped_entries:
                knowledge_store.append_entries(scraped_entries)
            cached_entries, cached_site_keys = knowledge_store.recall_relevant_by_site(
                label,
                max_sites=target_site_count,
                max_entries=max(recall_limit * 2, 20),
            )

        final_candidates = knowledge_store.recall(
            label, max_entries=max(recall_limit * 4, 50)
        )
        seen_sites: set[str] = set()
        source_entries = []
        for entry in final_candidates:
            if not _entry_relevant_to_label(entry, label):
                continue
            site = _site_key(entry.get("url", ""), entry.get("source", ""))
            if not site or site in seen_sites:
                continue
            source_entries.append(entry)
            seen_sites.add(site)
            if len(source_entries) >= recall_limit:
                break

        source_context = format_source_context(source_entries)
        if tutor:
            try:
                explanation = tutor.explain_detection(
                    label,
                    confidence_f,
                    center_x_i,
                    center_y_i,
                    source_context=source_context,
                )
            except Exception as exc:
                scrape_errors.append(f"llm: {exc}")
                explanation = (
                    "### 1. Component Overview\n"
                    f"- **Detected label:** {label}\n\n"
                    "### 2. LLM Status\n"
                    "- LLM explanation is temporarily unavailable. Source links are still listed below."
                )
        else:
            if llm_startup_error:
                scrape_errors.append(f"llm: {llm_startup_error}")
            explanation = (
                "### 1. Component Overview\n"
                f"- **Detected label:** {label}\n\n"
                "### 2. LLM Status\n"
                "- LLM is not configured yet. Source links are still listed below."
            )

        return jsonify(
            {
                "learning_notes": explanation,
                "sources": [
                    {
                        "source": e.get("source", ""),
                        "title": e.get("title", ""),
                        "url": e.get("url", ""),
                        "snippet": e.get("snippet", ""),
                    }
                    for e in source_entries
                ],
                "source_count": len(source_entries),
                "scrape_errors": scrape_errors,
            }
        )

    @app.post("/api/refresh_component_sources")
    def refresh_component_sources():
        payload = request.get_json(silent=True) or {}
        label = payload.get("label")
        if not label:
            return jsonify({"error": "Missing component label."}), 400

        scraped_entries, scrape_errors = scraper.scrape_component(label)
        if scraped_entries:
            knowledge_store.append_entries(scraped_entries)

        source_entries = knowledge_store.recall(label, max_entries=20)
        return jsonify(
            {
                "label": label,
                "appended_entries": len(scraped_entries),
                "source_count": len(source_entries),
                "scrape_errors": scrape_errors,
                "sources": [
                    {
                        "source": e.get("source", ""),
                        "title": e.get("title", ""),
                        "url": e.get("url", ""),
                        "snippet": e.get("snippet", ""),
                    }
                    for e in source_entries
                ],
            }
        )

    @app.errorhandler(413)
    def file_too_large(_):
        return jsonify(
            {"error": f"File too large. Max size is {MAX_UPLOAD_MB} MB."}
        ), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
