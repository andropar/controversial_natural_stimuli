#!/usr/bin/env python3
"""Local pair-labeling server for controversial stimulus image pairs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from urllib.parse import unquote, urlparse


APP_ROOT = Path(__file__).resolve().parents[1]
SHARE_ROOT = Path(__file__).resolve().parents[4]
WEB_ROOT = APP_ROOT / "web"
RESULTS_ROOT = APP_ROOT / "results"
DEFAULT_LABELS = RESULTS_ROOT / "pair_labels.csv"
DEFAULT_QUEUE = RESULTS_ROOT / "pair_queue_anchor_balanced_300.csv"

LABEL_FIELDS = [
    "semantic_similarity",
    "visual_surface_similarity",
    "shape_layout_similarity",
    "scene_context_similarity",
    "dominant_relation",
    "confidence",
]

FIELDNAMES = [
    "img_i",
    "img_j",
    "image_i",
    "image_j",
    *LABEL_FIELDS,
    "notes",
    "labeler",
    "updated_at",
]

SCHEMA = {
    "semantic_similarity": [
        {"value": "same", "label": "same"},
        {"value": "related", "label": "related"},
        {"value": "unrelated", "label": "unrelated"},
        {"value": "unsure", "label": "unsure"},
    ],
    "visual_surface_similarity": [
        {"value": "high", "label": "high"},
        {"value": "medium", "label": "medium"},
        {"value": "low", "label": "low"},
        {"value": "unsure", "label": "unsure"},
    ],
    "shape_layout_similarity": [
        {"value": "high", "label": "high"},
        {"value": "medium", "label": "medium"},
        {"value": "low", "label": "low"},
        {"value": "unsure", "label": "unsure"},
    ],
    "scene_context_similarity": [
        {"value": "same", "label": "same"},
        {"value": "related", "label": "related"},
        {"value": "different", "label": "different"},
        {"value": "unsure", "label": "unsure"},
    ],
    "dominant_relation": [
        {
            "value": "semantic_match_visual_mismatch",
            "label": "semantic match, visual mismatch",
        },
        {
            "value": "visual_match_semantic_mismatch",
            "label": "visual match, semantic mismatch",
        },
        {"value": "both_match", "label": "both match"},
        {"value": "both_mismatch", "label": "both mismatch"},
        {"value": "mixed_or_ambiguous", "label": "mixed or ambiguous"},
        {"value": "unsure", "label": "unsure"},
    ],
    "confidence": [
        {"value": "high", "label": "high"},
        {"value": "medium", "label": "medium"},
        {"value": "low", "label": "low"},
    ],
}

SHORTCUTS = {
    "semantic_similarity": ["Q", "W", "E", "R"],
    "visual_surface_similarity": ["A", "S", "D", "F"],
    "shape_layout_similarity": ["Z", "X", "C", "V"],
    "scene_context_similarity": ["U", "I", "O", "P"],
    "dominant_relation": ["1", "2", "3", "4", "5", "6"],
    "confidence": ["J", "K", "L"],
}

IMAGE_RE = re.compile(r"image_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


def default_image_dir() -> Path:
    candidates = [
        SHARE_ROOT
        / "00_stimulus_selection"
        / "results"
        / "selected_stimuli"
        / "all_models"
        / "eval_pipeline"
        / "images",
        SHARE_ROOT
        / "00_stimulus_selection"
        / "results"
        / "selected_stimuli"
        / "all_models"
        / "eval_pipeline"
        / "best_raw_combined"
        / "images",
        SHARE_ROOT
        / "00_stimulus_selection"
        / "decision_checks"
        / "selection_evaluation"
        / "results"
        / "all_models"
        / "images",
        SHARE_ROOT / "external_data" / "final_cstims_hdf5_files" / "all_models",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("image_*.png")):
            return candidate
    return candidates[0]


def image_index(path: Path, fallback: int) -> int:
    match = IMAGE_RE.match(path.name)
    if not match:
        return fallback
    return int(match.group(1))


def load_images(image_dir: Path) -> list[dict[str, object]]:
    files = [
        *image_dir.glob("image_*.png"),
        *image_dir.glob("image_*.jpg"),
        *image_dir.glob("image_*.jpeg"),
    ]
    rows = [
        {"idx": image_index(path, fallback), "filename": path.name, "path": path}
        for fallback, path in enumerate(files)
    ]
    rows.sort(key=lambda row: (int(row["idx"]), str(row["filename"])))
    if len(rows) < 2:
        raise RuntimeError(f"Expected at least two stimulus images in {image_dir}")
    return rows


def build_pairs(images: list[dict[str, object]]) -> list[dict[str, object]]:
    pairs = []
    for left_pos, left in enumerate(images):
        for right in images[left_pos + 1 :]:
            pairs.append(
                {
                    "img_i": int(left["idx"]),
                    "img_j": int(right["idx"]),
                    "image_i": str(left["filename"]),
                    "image_j": str(right["filename"]),
                }
            )
    return pairs


def load_pair_queue(
    path: Path,
    all_pairs: list[dict[str, object]],
) -> list[dict[str, object]]:
    pair_lookup = {pair_key(pair["img_i"], pair["img_j"]): pair for pair in all_pairs}
    queue = []
    seen: set[str] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("img_i") or not row.get("img_j"):
                continue
            key = pair_key(row["img_i"], row["img_j"])
            pair = pair_lookup.get(key)
            if pair is None:
                raise RuntimeError(f"Queue contains unknown pair {key}")
            if key in seen:
                raise RuntimeError(f"Queue contains duplicate pair {key}")
            seen.add(key)
            merged = dict(pair)
            for field, value in row.items():
                if field not in merged:
                    merged[field] = value
            queue.append(merged)
    if not queue:
        raise RuntimeError(f"Pair queue is empty: {path}")
    return queue


def pair_key(img_i: int | str, img_j: int | str) -> str:
    i = int(img_i)
    j = int(img_j)
    if i > j:
        i, j = j, i
    return f"{i}-{j}"


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = {}
        for row in reader:
            if not row.get("img_i") or not row.get("img_j"):
                continue
            rows[pair_key(row["img_i"], row["img_j"])] = {
                field: row.get(field, "") for field in FIELDNAMES
            }
    return rows


def save_labels(path: Path, labels: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        labels.values(),
        key=lambda row: (int(row.get("img_i", 0)), int(row.get("img_j", 0))),
    )
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def row_complete(row: dict[str, str]) -> bool:
    return all(bool(row.get(field)) for field in LABEL_FIELDS)


class PairLabelingServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        image_dir: Path,
        labels_path: Path,
        labeler: str,
        pair_queue: Path | None,
        use_all_pairs: bool,
    ) -> None:
        self.image_dir = image_dir
        self.labels_path = labels_path
        self.labeler = labeler
        self.pair_queue = pair_queue
        self.use_all_pairs = use_all_pairs
        self.images = load_images(image_dir)
        self.all_pairs = build_pairs(self.images)
        if not use_all_pairs and pair_queue is not None and pair_queue.exists():
            self.pairs = load_pair_queue(pair_queue, self.all_pairs)
        else:
            self.pairs = self.all_pairs
        self.image_by_name = {str(row["filename"]): Path(row["path"]) for row in self.images}
        self.pair_by_key = {
            pair_key(pair["img_i"], pair["img_j"]): pair for pair in self.pairs
        }
        self.labels = load_labels(labels_path)
        super().__init__(server_address, handler_class)

    @property
    def completed_count(self) -> int:
        active_keys = set(self.pair_by_key)
        return sum(
            1
            for key, row in self.labels.items()
            if key in active_keys and row_complete(row)
        )


class Handler(BaseHTTPRequestHandler):
    server: PairLabelingServer

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send_json(
                {
                    "image_dir": str(self.server.image_dir),
                    "labels_path": str(self.server.labels_path),
                    "queue_path": str(self.server.pair_queue) if self.server.pair_queue else "",
                    "using_all_pairs": self.server.use_all_pairs,
                    "labeler": self.server.labeler,
                    "images": [
                        {"idx": row["idx"], "filename": row["filename"]}
                        for row in self.server.images
                    ],
                    "pairs": self.server.pairs,
                    "labels": self.server.labels,
                    "schema": SCHEMA,
                    "shortcuts": SHORTCUTS,
                    "completed_count": self.server.completed_count,
                    "total_pairs": len(self.server.pairs),
                }
            )
        elif path == "/api/export":
            if not self.server.labels_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "No label CSV exists yet")
                return
            self.serve_file(self.server.labels_path, "text/csv; charset=utf-8")
        elif path.startswith("/images/"):
            filename = unquote(path.removeprefix("/images/"))
            image_path = self.server.image_by_name.get(filename)
            if image_path is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown image")
                return
            ctype = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            self.serve_file(image_path, ctype)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/label":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self.read_json()
            row = self.normalize_label_payload(payload)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        key = pair_key(row["img_i"], row["img_j"])
        self.server.labels[key] = row
        save_labels(self.server.labels_path, self.server.labels)
        self.send_json(
            {
                "ok": True,
                "key": key,
                "completed": row_complete(row),
                "completed_count": self.server.completed_count,
                "total_pairs": len(self.server.pairs),
            }
        )

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        return payload

    def normalize_label_payload(self, payload: dict[str, object]) -> dict[str, str]:
        if "img_i" not in payload or "img_j" not in payload:
            raise ValueError("Missing img_i or img_j")
        img_i = int(payload["img_i"])
        img_j = int(payload["img_j"])
        if img_i == img_j:
            raise ValueError("Self-pairs are not labelable")
        key = pair_key(img_i, img_j)
        pair = self.server.pair_by_key.get(key)
        if pair is None:
            raise ValueError(f"Unknown image pair {img_i}, {img_j}")

        row = {field: "" for field in FIELDNAMES}
        row["img_i"] = str(pair["img_i"])
        row["img_j"] = str(pair["img_j"])
        row["image_i"] = str(pair["image_i"])
        row["image_j"] = str(pair["image_j"])
        for field in LABEL_FIELDS:
            value = str(payload.get(field, "") or "")
            allowed = {entry["value"] for entry in SCHEMA[field]}
            if value and value not in allowed:
                raise ValueError(f"Invalid {field}: {value}")
            row[field] = value
        row["notes"] = str(payload.get("notes", "") or "")
        row["labeler"] = str(payload.get("labeler", "") or self.server.labeler)
        row["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return row

    def send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--image-dir", type=Path, default=default_image_dir())
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--pair-queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--labeler", default="default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = PairLabelingServer(
        (args.host, args.port),
        Handler,
        image_dir=args.image_dir.resolve(),
        labels_path=args.labels.resolve(),
        labeler=args.labeler,
        pair_queue=args.pair_queue.resolve() if args.pair_queue else None,
        use_all_pairs=args.all_pairs,
    )
    print(f"Serving {len(server.images)} images, {len(server.pairs)} pairs")
    print(f"Image dir: {server.image_dir}")
    print(f"Labels:    {server.labels_path}")
    if server.use_all_pairs:
        print("Queue:     all pairs")
    elif server.pair_queue and server.pair_queue.exists():
        print(f"Queue:     {server.pair_queue}")
    else:
        print("Queue:     default queue not found; using all pairs")
    print(f"Open:      http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
