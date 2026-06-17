#!/usr/bin/env python
"""Small local server for contrastive residual-neighborhood annotations."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
STATIC_DIR = APP_ROOT / "static"
CARDS_PATH = DATA_DIR / "cards.json"
ANNOTATIONS_PATH = DATA_DIR / "annotations.jsonl"
WRITE_LOCK = threading.Lock()


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def read_annotations(annotator_id: str | None = None) -> list[dict]:
    if not ANNOTATIONS_PATH.exists():
        return []
    rows = []
    with ANNOTATIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if annotator_id and row.get("annotator_id") != annotator_id:
                continue
            rows.append(row)
    return rows


def flatten_annotation(row: dict) -> dict:
    response = row.get("response") or {}
    groups = response.get("groups") or []
    out = {
        "timestamp": row.get("timestamp"),
        "annotator_id": row.get("annotator_id"),
        "card_id": row.get("card_id"),
        "condition": row.get("condition"),
        "card_type": row.get("card_type"),
        "query_source_index": row.get("query_source_index"),
        "brain_dominant_bin": (row.get("brain_pair_summary") or {}).get("dominant_bin"),
        "brain_known_pairs": (row.get("brain_pair_summary") or {}).get("known_pairs"),
        "brain_total_pairs": (row.get("brain_pair_summary") or {}).get("total_pairs"),
        "brain_mean_z": (row.get("brain_pair_summary") or {}).get("mean_brain_z"),
        "contrast_difference": response.get("contrast_difference"),
        "contrast_distinguishing_cue": response.get("contrast_distinguishing_cue"),
        "contrast_notes": response.get("contrast_notes"),
    }
    for idx, group in enumerate(groups[:2], start=1):
        out[f"group{idx}_primary_cue"] = group.get("primary_cue")
        out[f"group{idx}_secondary_cues"] = ";".join(group.get("secondary_cues") or [])
        out[f"group{idx}_confidence"] = group.get("confidence")
        out[f"group{idx}_coherence"] = group.get("coherence")
        out[f"group{idx}_explanation"] = group.get("explanation")
    hidden = row.get("hidden") or {}
    out["hidden_model_i"] = hidden.get("model_i")
    out["hidden_model_j"] = hidden.get("model_j")
    out["hidden_residual_profile_distance"] = hidden.get("residual_profile_distance")
    return out


class AnnotationHandler(BaseHTTPRequestHandler):
    server_version = "ResidualAnnotationHTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/index.html"}:
            self.serve_file(STATIC_DIR / "index.html")
            return
        if path == "/api/cards":
            self.send_json(load_cards())
            return
        if path == "/api/annotations":
            params = parse_qs(parsed.query)
            annotator_id = (params.get("annotator_id") or [None])[0]
            self.send_json({"annotations": read_annotations(annotator_id)})
            return
        if path == "/api/export.csv":
            self.serve_csv()
            return
        if path.startswith("/static/"):
            self.serve_file(APP_ROOT / path.lstrip("/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = read_json_body(self)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": f"Invalid JSON: {exc}"}, HTTPStatus.BAD_REQUEST)
            return
        card_id = str(payload.get("card_id") or "").strip()
        annotator_id = str(payload.get("annotator_id") or "").strip()
        if not card_id or not annotator_id:
            self.send_json(
                {"error": "card_id and annotator_id are required"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        known_cards = load_card_index()
        if card_id not in known_cards:
            self.send_json({"error": f"Unknown card_id: {card_id}"}, HTTPStatus.BAD_REQUEST)
            return
        card = known_cards[card_id]
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "annotator_id": annotator_id,
            "card_id": card_id,
            "condition": card.get("condition"),
            "card_type": card.get("card_type"),
            "query_source_index": card.get("query_source_index"),
            "local_residual_geometry": card.get("local_residual_geometry"),
            "brain_pair_summary": card.get("brain_pair_summary") or {},
            "response": payload.get("response") or {},
            "hidden": card.get("hidden") or {},
        }
        with WRITE_LOCK:
            ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with ANNOTATIONS_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        self.send_json({"ok": True, "annotation": row})

    def serve_csv(self) -> None:
        rows = [flatten_annotation(row) for row in read_annotations()]
        fieldnames = sorted({key for row in rows for key in row})
        if not fieldnames:
            fieldnames = ["timestamp", "annotator_id", "card_id"]
        from io import StringIO

        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        body = buf.getvalue().encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", "attachment; filename=annotations.csv")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path: Path) -> None:
        path = path.resolve()
        if not str(path).startswith(str(APP_ROOT.resolve())) or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")


_CARD_CACHE: dict | None = None
_CARD_INDEX_CACHE: dict[str, dict] | None = None


def load_cards() -> dict:
    global _CARD_CACHE
    if _CARD_CACHE is None:
        if not CARDS_PATH.exists():
            _CARD_CACHE = {
                "schema_version": "missing",
                "cue_categories": [],
                "contrast_options": [],
                "cards": [],
                "metadata": {"error": f"Missing {CARDS_PATH}"},
            }
        else:
            with CARDS_PATH.open("r", encoding="utf-8") as f:
                _CARD_CACHE = json.load(f)
    return _CARD_CACHE


def load_card_index() -> dict[str, dict]:
    global _CARD_INDEX_CACHE
    if _CARD_INDEX_CACHE is None:
        _CARD_INDEX_CACHE = {
            card["card_id"]: card for card in load_cards().get("cards", [])
        }
    return _CARD_INDEX_CACHE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    if not ANNOTATIONS_PATH.exists():
        ANNOTATIONS_PATH.write_text("", encoding="utf-8")
    httpd = ThreadingHTTPServer((args.host, args.port), AnnotationHandler)
    print(f"Serving annotation app at http://{args.host}:{args.port}")
    print(f"Cards: {CARDS_PATH}")
    print(f"Annotations: {ANNOTATIONS_PATH}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
