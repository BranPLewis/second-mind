import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# --- Utility Functions ---


def _clean_text(text: str) -> str:
    return " ".join((text or "").split())


def _clip(text: str, limit: int = 280) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _norm_label(label: str) -> str:
    return _clean_text(label).lower()


def _label_tokens(label: str) -> set[str]:
    return {
        token for token in re.split(r"[^a-z0-9]+", _norm_label(label)) if len(token) > 1
    }


def _site_key(url: str, source: str = "") -> str:
    parsed = urlparse((url or "").strip())
    if parsed.netloc:
        return parsed.netloc.lower().removeprefix("www.")
    if source.startswith("web:"):
        return source.split(":", 1)[1].strip().lower()
    return (source or "unknown").strip().lower()


def _entry_relevant_to_label(entry: dict, label: str) -> bool:
    norm_label = _norm_label(label)
    tokens = _label_tokens(label)
    blob = _norm_label(
        " ".join(
            [
                entry.get("component_label", ""),
                entry.get("query", ""),
                entry.get("title", ""),
                entry.get("snippet", ""),
                entry.get("url", ""),
            ]
        )
    )
    if not blob:
        return False
    if entry.get("url", "").startswith("local://"):
        return True
    return (norm_label in blob) or any(token in blob for token in tokens)


def _extract_relevant_excerpts(
    soup: BeautifulSoup,
    label: str,
    *,
    max_segments: int = 4,
    segment_limit: int = 220,
) -> str:
    tokens = _label_tokens(label)
    raw_segments: list[str] = []

    for node in soup.select("h1, h2, h3, p, li"):
        text = _clean_text(node.get_text(" ", strip=True))
        if len(text) < 40:
            continue
        raw_segments.append(text)
        if len(raw_segments) >= 120:
            break

    if not raw_segments:
        return ""

    matching = [
        seg for seg in raw_segments if any(token in seg.lower() for token in tokens)
    ]
    chosen = (matching or raw_segments)[:max_segments]
    clipped = [_clip(seg, segment_limit) for seg in chosen]
    return " | ".join(clipped)


# --- CSV Storage Engine ---


class ComponentKnowledgeStore:
    def __init__(self, csv_path: str = "data/component_sources.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "timestamp_utc",
            "component_label",
            "component_label_norm",
            "source",
            "query",
            "title",
            "snippet",
            "url",
        ]
        self._ensure_csv_schema()

    def _ensure_csv_schema(self):
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
            return

        with self.csv_path.open("r", newline="", encoding="utf-8") as csvfile:
            rows = list(csv.reader(csvfile))

        if not rows:
            with self.csv_path.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
            return

        header = [cell.strip() for cell in rows[0]]
        if header == self.fieldnames:
            return

        rebuilt_rows: list[dict] = []
        for row in rows:
            if len(row) != len(self.fieldnames):
                continue
            rebuilt_rows.append({k: v for k, v in zip(self.fieldnames, row)})

        with self.csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            if rebuilt_rows:
                writer.writerows(rebuilt_rows)

    def append_entries(self, entries: list[dict]):
        if not entries:
            return

        # Optional: Prevent duplicating exact URLs in the same run to save disk I/O
        seen_urls = set()
        unique_entries = []
        for e in entries:
            if e["url"] not in seen_urls:
                unique_entries.append(e)
                seen_urls.add(e["url"])

        with self.csv_path.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            for entry in unique_entries:
                writer.writerow({k: entry.get(k, "") for k in self.fieldnames})

    def recall(self, label: str, max_entries: int = 10) -> list[dict]:
        if not self.csv_path.exists():
            return []

        wanted = _norm_label(label)
        matches: list[dict] = []
        with self.csv_path.open("r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("component_label_norm") == wanted:
                    matches.append(row)

        matches.sort(key=lambda row: row.get("timestamp_utc", ""), reverse=True)
        return matches[:max_entries]

    def recall_relevant_by_site(
        self, label: str, max_sites: int = 6, max_entries: int = 12
    ) -> tuple[list[dict], set[str]]:
        matches = self.recall(label, max_entries=500)
        selected: list[dict] = []
        seen_sites: set[str] = set()

        for row in matches:
            if not _entry_relevant_to_label(row, label):
                continue
            site = _site_key(row.get("url", ""), row.get("source", ""))
            if not site or site in seen_sites:
                continue

            selected.append(row)
            seen_sites.add(site)
            if len(seen_sites) >= max_sites or len(selected) >= max_entries:
                break

        return selected[:max_entries], seen_sites


# --- Web Scraper & Routing Engine ---


class ComponentScraper:
    def __init__(self, timeout_seconds: int = 8):
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        # 1. Use an honest, descriptive UA to bypass Wikipedia's API blocks
        self.session.headers.update(
            {
                "User-Agent": "SecondMindLabAssistant/1.0 (Educational Project; info@example.com)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )

        self.generic_labels = {
            "capacitor",
            "resistor",
            "diode",
            "transistor",
            "inductor",
            "led",
            "switch",
            "button",
            "ic",
            "integrated circuit",
        }

    def scrape_component(self, label: str) -> tuple[list[dict], list[str]]:
        entries: list[dict] = []
        errors: list[str] = []

        search_term = re.sub(r"\([^)]*\)", "", label).strip()
        is_generic = search_term.lower() in self.generic_labels

        # Fast Path
        tutorial_entries = self._scrape_hardcoded_tutorial(search_term)
        if tutorial_entries:
            entries.extend(self._retag_entries(tutorial_entries, label))

        # 2. Ensure Wikipedia is checked for BOTH generic and specific components
        if is_generic:
            targets = (
                ("fastturnpcbs", self._scrape_fastturnpcbs),
                ("wikipedia", self._scrape_wikipedia),
            )
        else:
            targets = (
                ("snapeda", self._scrape_snapeda),
                ("componentsearchengine", self._scrape_componentsearchengine),
                ("lcsc", self._scrape_lcsc),
                ("wikipedia", self._scrape_wikipedia),
            )

        for source_name, scrape_fn in targets:
            try:
                raw_entries = scrape_fn(search_term)
                entries.extend(self._retag_entries(raw_entries, label))
            except Exception as exc:
                errors.append(f"{source_name}: {exc}")

        # 3. Ultimate Offline Fallback: Guarantee >0 sources for generic parts
        if not entries and is_generic:
            offline = self._get_offline_fallback(search_term)
            entries.extend(self._retag_entries(offline, label))

        return entries, errors

    def _get_offline_fallback(self, label: str) -> list[dict]:
        fallbacks = {
            "capacitor": "A capacitor stores electrical energy in an electric field. It is used for filtering, energy storage, and timing.",
            "resistor": "A resistor limits or regulates the flow of electrical current in an electronic circuit.",
            "diode": "A diode allows current to flow in one direction only, used for rectification and protection.",
            "transistor": "A transistor is a semiconductor device used to amplify or switch electrical signals and power.",
            "inductor": "An inductor stores energy in a magnetic field when electric current flows through it.",
            "led": "A Light Emitting Diode (LED) emits light when current flows through it.",
            "ic": "An Integrated Circuit (IC) is a set of electronic circuits on one small flat piece of semiconductor material.",
            "integrated circuit": "An Integrated Circuit (IC) is a set of electronic circuits on one small flat piece of semiconductor material.",
        }
        desc = fallbacks.get(label.lower())
        if desc:
            return [
                self._record(
                    "offline_cache",
                    label,
                    f"{label.title()} Overview",
                    desc,
                    "local://offline-cache",
                )
            ]
        return []

    def _retag_entries(self, entries: list[dict], original_label: str) -> list[dict]:
        """Ensures the dictionary saves under the exact YOLO label, even if we searched a sanitized version."""
        normalized = _norm_label(original_label)
        return [
            {**e, "component_label": original_label, "component_label_norm": normalized}
            for e in entries
        ]

    def _record(
        self, source: str, label: str, title: str, snippet: str, url: str
    ) -> dict:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "component_label": label,
            "component_label_norm": _norm_label(label),
            "source": source,
            "query": label,
            "title": _clip(_clean_text(title), 140),
            "snippet": _clip(_clean_text(snippet), 700),
            "url": url,
        }

    def _fetch_html(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def _extract_page_entries(
        self, source: str, label: str, search_url: str, html: str, strict_domain: str
    ) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        entries = []

        # Grab Page-Level Meta
        page_title = _clean_text(
            soup.title.get_text(" ", strip=True) if soup.title else f"{source} result"
        )
        meta_desc = _clean_text(
            soup.find("meta", attrs={"name": "description"}).get("content", "")
            if soup.find("meta", attrs={"name": "description"})
            else ""
        )

        page_excerpts = _extract_relevant_excerpts(soup, label, max_segments=4)
        page_summary = " | ".join([part for part in [meta_desc, page_excerpts] if part])

        if page_summary:
            entries.append(
                self._record(source, label, page_title, page_summary, search_url)
            )

        # Grab Links
        seen_urls = {search_url}
        for anchor in soup.select("a[href]"):
            href = _clean_text(anchor.get("href", ""))
            full_url = urljoin(search_url, href)

            if strict_domain not in full_url or full_url in seen_urls:
                continue

            text = _clean_text(anchor.get_text(" ", strip=True))
            if text and len(text) > 3:
                parent_text = (
                    _clean_text(anchor.parent.get_text(" ", strip=True))
                    if anchor.parent
                    else ""
                )
                link_snippet_parts = [text]
                if parent_text and parent_text != text:
                    link_snippet_parts.append(_clip(parent_text, 280))
                if page_excerpts:
                    link_snippet_parts.append(_clip(page_excerpts, 320))

                entries.append(
                    self._record(
                        source,
                        label,
                        text,
                        " | ".join(link_snippet_parts),
                        full_url,
                    )
                )
                seen_urls.add(full_url)

            if len(entries) >= 5:
                break

        return entries

    # --- Site Specific Scrapers ---

    def _scrape_snapeda(self, label: str) -> list[dict]:
        url = f"https://www.snapeda.com/search/?q={quote_plus(label)}"
        return self._extract_page_entries(
            "snapeda", label, url, self._fetch_html(url), "snapeda.com"
        )

    def _scrape_componentsearchengine(self, label: str) -> list[dict]:
        url = f"https://componentsearchengine.com/part-search/?searchTerm={quote_plus(label)}"
        return self._extract_page_entries(
            "componentsearchengine",
            label,
            url,
            self._fetch_html(url),
            "componentsearchengine.com",
        )

    def _scrape_lcsc(self, label: str) -> list[dict]:
        url = f"https://www.lcsc.com/search?q={quote_plus(label)}"
        return self._extract_page_entries(
            "lcsc", label, url, self._fetch_html(url), "lcsc.com"
        )

    def _scrape_sparkfun(self, label: str) -> list[dict]:
        url = f"https://www.sparkfun.com/search/results?term={quote_plus(label)}"
        return self._extract_page_entries(
            "sparkfun", label, url, self._fetch_html(url), "sparkfun.com"
        )

    def _scrape_fastturnpcbs(self, label: str) -> list[dict]:
        url = "https://www.fastturnpcbs.com/blog/basics/pcb-components-explained"
        html = self._fetch_html(url)
        return self._extract_page_entries(
            "fastturnpcbs",
            label,
            url,
            html,
            "fastturnpcbs.com",
        )

    def _scrape_wikipedia(self, label: str) -> list[dict]:
        url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(label)}&limit=3&namespace=0&format=json"
        try:
            data = self.session.get(url, timeout=self.timeout_seconds).json()
            if len(data) == 4 and data[1]:
                return [
                    self._record(
                        "wikipedia",
                        label,
                        data[1][i],
                        data[2][i] or f"Wikipedia: {data[1][i]}",
                        data[3][i],
                    )
                    for i in range(len(data[1]))
                ]
        except Exception:
            pass
        return []

    def _scrape_hardcoded_tutorial(self, label: str) -> list[dict]:
        hardcoded_urls = {
            "capacitor": "https://www.build-electronic-circuits.com/how-does-a-capacitor-work/",
            "resistor": "https://www.build-electronic-circuits.com/what-is-a-resistor/",
            "diode": "https://www.build-electronic-circuits.com/what-is-a-diode/",
            "transistor": "https://www.build-electronic-circuits.com/how-transistors-work/",
            "ic": "https://www.build-electronic-circuits.com/integrated-circuit/",
            "integrated circuit": "https://www.build-electronic-circuits.com/integrated-circuit/",
            "led": "https://learn.sparkfun.com/tutorials/light-emitting-diodes-leds/all",
            "inductor": "https://learn.sparkfun.com/tutorials/inductors/all",
            "switch": "https://learn.sparkfun.com/tutorials/switch-basics/all",
            "button": "https://learn.sparkfun.com/tutorials/switch-basics/all",
        }
        url = hardcoded_urls.get(label.lower())
        if not url:
            return []

        try:
            html = self._fetch_html(url)
            soup = BeautifulSoup(html, "html.parser")
            title = _clean_text(
                soup.title.get_text(" ", strip=True)
                if soup.title
                else f"{label} Tutorial"
            )
            paragraphs = soup.find_all("p")
            snippet = " ".join([p.get_text(" ", strip=True) for p in paragraphs[1:8]])
            focused = _extract_relevant_excerpts(soup, label, max_segments=5)
            combined = " | ".join([part for part in [snippet, focused] if part])
            return [self._record("tutorial", label, title, combined, url)]
        except Exception:
            return []


# Backwards-compatible aliases
RestrictedComponentScraper = ComponentScraper


def format_source_context(entries: list[dict], max_entries: int = 8) -> str:
    if not entries:
        return "No source notes are available yet for this component."
    lines = ["Source notes:"]
    for entry in entries[:max_entries]:
        lines.extend(
            [
                f"- [{entry.get('source', 'unknown')}] {entry.get('title', '')}",
                f"  snippet: {entry.get('snippet', '')}",
                f"  url: {entry.get('url', '')}",
            ]
        )
    return "\n".join(lines)
