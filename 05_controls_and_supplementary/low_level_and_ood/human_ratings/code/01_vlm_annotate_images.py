#!/usr/bin/env python3
"""Minimal VLM-based stimulus annotation audit.

Loads CSTIM/vicco stimulus images, sends each image to OpenRouter, validates a
strict JSON annotation schema, and writes one row per stimulus.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image


_CSTIMS_SHARE_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "src" / "cstims").exists()
)
sys.path.insert(0, str(_CSTIMS_SHARE_ROOT / "src"))

from cstims import paths

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = paths.cstim_hdf5_root()
DEFAULT_API_KEY_FILE = BASE_DIR / "openrouter_api_key.txt"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "vlm_annotation_audit_v1"

CONDITIONS = [
    "all_models",
    "architecture",
    "dataset",
    "sota",
    "training_objective",
    "vicco",
]

SCALAR_FIELDS = [
    "recognizability",
    "ambiguity",
    "natural_photo_typicality",
    "visual_clutter",
    "object_centricity",
    "scene_centricity",
    "caption_confidence",
]

COUNT_FIELDS = [
    "estimated_salient_object_count",
    "estimated_distinct_object_categories",
]

BINARY_FIELDS = [
    "contains_person",
    "contains_face",
    "contains_animal",
    "contains_text",
    "contains_vehicle",
    "contains_food",
    "contains_indoor_scene",
    "contains_outdoor_scene",
    "contains_artificial_or_rendered_content",
    "contains_occlusion_or_truncation",
    "contains_unusual_viewpoint",
    "contains_multiple_main_objects",
]

DOMINANT_CONTENT_TYPES = {
    "object",
    "scene",
    "texture_pattern",
    "closeup_detail",
    "person_face",
    "animal",
    "food",
    "text_document",
    "mixed",
    "unclear",
}
DOMINANT_CONTENT_TYPE_ALIASES = {
    "person": "person_face",
    "face": "person_face",
    "people": "person_face",
    "text": "text_document",
    "document": "text_document",
    "texture": "texture_pattern",
    "pattern": "texture_pattern",
    "closeup": "closeup_detail",
    "close_up": "closeup_detail",
}

IMAGE_STYLES = {
    "natural_photo",
    "edited_photo",
    "illustration_or_render",
    "screenshot_or_document",
    "abstract_or_texture",
    "unclear",
}
IMAGE_STYLE_ALIASES = {
    "photo": "natural_photo",
    "photograph": "natural_photo",
    "natural": "natural_photo",
    "edited": "edited_photo",
    "render": "illustration_or_render",
    "illustration": "illustration_or_render",
    "screenshot": "screenshot_or_document",
    "document": "screenshot_or_document",
    "abstract": "abstract_or_texture",
    "texture": "abstract_or_texture",
}

LIST_FIELDS = [
    "main_objects",
    "possible_interpretations",
    "quality_flags",
]

TEXT_FIELDS = [
    "semantic_domain",
    "short_caption",
    "uncertainty_notes",
]

SYSTEM_PROMPT = (
    "You are annotating natural images for a neuroscience stimulus-control "
    "analysis. Your task is to provide structured, conservative perceptual "
    "and semantic annotations. These annotations are candidate covariates, "
    "not ground-truth human ratings. Do not speculate beyond what is visible. "
    "Return only valid JSON matching the requested schema."
)

USER_PROMPT = """Please annotate this image using the schema below.

Use integer ratings from 1 to 5.

Definitions:
- recognizability: how clearly a typical human viewer could recognize the main content.
- ambiguity: how many plausible interpretations the image has.
- natural_photo_typicality: whether the image looks like a typical natural photograph.
- visual_clutter: how many competing objects/regions/details are present.
- object_centricity: whether a clear central object dominates.
- scene_centricity: whether the image is mainly a place/scene.
- caption_confidence: how confident you are that a concise caption captures the image.

Return ONLY valid JSON. No markdown. No explanation outside JSON.

Schema:
{
  "recognizability": integer 1-5,
  "ambiguity": integer 1-5,
  "natural_photo_typicality": integer 1-5,
  "visual_clutter": integer 1-5,
  "object_centricity": integer 1-5,
  "scene_centricity": integer 1-5,
  "caption_confidence": integer 1-5,

  "estimated_salient_object_count": integer or null,
  "estimated_distinct_object_categories": integer or null,

  "contains_person": boolean,
  "contains_face": boolean,
  "contains_animal": boolean,
  "contains_text": boolean,
  "contains_vehicle": boolean,
  "contains_food": boolean,
  "contains_indoor_scene": boolean,
  "contains_outdoor_scene": boolean,
  "contains_artificial_or_rendered_content": boolean,
  "contains_occlusion_or_truncation": boolean,
  "contains_unusual_viewpoint": boolean,
  "contains_multiple_main_objects": boolean,

  "dominant_content_type": one of ["object", "scene", "texture_pattern", "closeup_detail", "person_face", "animal", "food", "text_document", "mixed", "unclear"],
  "image_style": one of ["natural_photo", "edited_photo", "illustration_or_render", "screenshot_or_document", "abstract_or_texture", "unclear"],
  "semantic_domain": short lowercase string,

  "short_caption": string <= 20 words,
  "main_objects": list of short strings,
  "possible_interpretations": list of 1-5 short strings,
  "uncertainty_notes": string,
  "quality_flags": list of strings
}"""

CORRECTION_SUFFIX = (
    "\n\nYour previous response was not valid JSON matching the schema. "
    "Return only corrected JSON."
)


@dataclass(frozen=True)
class Stimulus:
    stimulus_id: str
    stim_key: str
    condition: str
    physical_folder: str
    stim_idx: int
    file_index: int
    image_name: str
    image_path: str


class ValidationError(RuntimeError):
    pass


class OpenRouterAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def physical_folder(condition: str) -> str:
    """Map requested analysis condition to physical image folder."""
    if condition == "architecture":
        return "dataset"
    if condition == "dataset":
        return "architecture"
    if condition == "vicco":
        return "shared_vicco"
    return condition


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_stim_idx(path: Path, condition: str, file_index: int) -> int:
    if condition == "vicco":
        match = re.search(r"_(\d+)$", path.stem)
        if match:
            return int(match.group(1))
        return file_index + 1
    match = re.search(r"image_(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    return file_index


def list_stimuli(image_root: Path, conditions: list[str]) -> list[Stimulus]:
    rows: list[Stimulus] = []
    for condition in conditions:
        folder = physical_folder(condition)
        image_dir = image_root / folder
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory for {condition}: {image_dir}")
        files = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not files:
            raise FileNotFoundError(f"No image files found for {condition}: {image_dir}")
        for file_index, path in enumerate(files):
            stim_idx = parse_stim_idx(path, condition, file_index)
            stim_key = f"{condition}_{stim_idx}"
            rows.append(
                Stimulus(
                    stimulus_id=stim_key,
                    stim_key=stim_key,
                    condition=condition,
                    physical_folder=folder,
                    stim_idx=stim_idx,
                    file_index=file_index,
                    image_name=path.name,
                    image_path=str(path),
                )
            )
    return rows


def read_api_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"OpenRouter API key file not found: {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"OpenRouter API key file is empty: {path}")
    return key


def resize_and_encode_image(path: Path, max_side: int) -> tuple[str, dict[str, Any]]:
    file_size = path.stat().st_size
    with Image.open(path) as img:
        original_width, original_height = img.size
        has_alpha = img.mode in {"RGBA", "LA"} or "transparency" in img.info
        if has_alpha:
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            rgb = background.convert("RGB")
        else:
            rgb = img.convert("RGB")
        resized = rgb.copy()
        resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        resized_width, resized_height = resized.size
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{encoded}"
    meta = {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "file_size_bytes": file_size,
        "encoded_format": "jpeg",
        "had_alpha": bool(has_alpha),
    }
    return data_url, meta


def build_payload(
    model: str,
    data_url: str,
    user_prompt: str,
    use_response_format: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def call_openrouter(
    api_key: str,
    model: str,
    data_url: str,
    user_prompt: str,
    timeout: float,
    use_response_format: bool,
) -> tuple[dict[str, Any], bool]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://laion-fmri.hebartlab.com",
        "X-Title": "CSTIM VLM annotation audit",
    }
    payload = build_payload(model, data_url, user_prompt, use_response_format)
    response = requests.post(
        OPENROUTER_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        detail = response.text[:1000]
        if use_response_format and "response_format" in detail:
            return call_openrouter(
                api_key=api_key,
                model=model,
                data_url=data_url,
                user_prompt=user_prompt,
                timeout=timeout,
                use_response_format=False,
            )
        raise OpenRouterAPIError(
            f"OpenRouter request failed with HTTP {response.status_code}: {detail}",
            status_code=response.status_code,
        )
    return response.json(), use_response_format


def extract_message_content(response_json: dict[str, Any]) -> str:
    try:
        content = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValidationError(f"Could not find choices[0].message.content: {exc}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    raise ValidationError(f"Unexpected message content type: {type(content).__name__}")


def strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    text = strip_markdown_fences(text)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValidationError("No valid JSON object found in model response")


def coerce_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{field} must be an integer, got boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        return int(value.strip())
    raise ValidationError(f"{field} must be an integer, got {value!r}")


def coerce_optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return coerce_int(value, field)


def coerce_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValidationError(f"{field} must be a boolean, got {value!r}")


def validate_string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValidationError(f"{field} must be a list")
    out = []
    for item in value:
        if not isinstance(item, str):
            raise ValidationError(f"{field} must contain only strings")
        out.append(item.strip())
    return out


def validate_annotation(obj: dict[str, Any]) -> dict[str, Any]:
    required = (
        SCALAR_FIELDS
        + COUNT_FIELDS
        + BINARY_FIELDS
        + ["dominant_content_type", "image_style"]
        + TEXT_FIELDS
        + LIST_FIELDS
    )
    missing = [field for field in required if field not in obj]
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")

    cleaned: dict[str, Any] = {}
    for field in SCALAR_FIELDS:
        value = coerce_int(obj[field], field)
        if not 1 <= value <= 5:
            raise ValidationError(f"{field} must be in 1..5, got {value}")
        cleaned[field] = value

    for field in COUNT_FIELDS:
        value = coerce_optional_int(obj[field], field)
        if value is not None and value < 0:
            raise ValidationError(f"{field} must be non-negative or null, got {value}")
        cleaned[field] = value

    for field in BINARY_FIELDS:
        cleaned[field] = coerce_bool(obj[field], field)

    dominant = str(obj["dominant_content_type"]).strip().lower()
    dominant = DOMINANT_CONTENT_TYPE_ALIASES.get(dominant, dominant)
    if dominant not in DOMINANT_CONTENT_TYPES:
        raise ValidationError(f"dominant_content_type has invalid value: {dominant!r}")
    cleaned["dominant_content_type"] = dominant

    style = str(obj["image_style"]).strip().lower()
    style = IMAGE_STYLE_ALIASES.get(style, style)
    if style not in IMAGE_STYLES:
        raise ValidationError(f"image_style has invalid value: {style!r}")
    cleaned["image_style"] = style

    semantic_domain = str(obj["semantic_domain"]).strip().lower()
    cleaned["semantic_domain"] = semantic_domain

    short_caption = str(obj["short_caption"]).strip()
    # The <=20 word prompt instruction is useful guidance, but it should not
    # make an otherwise valid annotation unusable.
    cleaned["short_caption"] = short_caption

    cleaned["main_objects"] = validate_string_list(obj["main_objects"], "main_objects")
    possible = validate_string_list(obj["possible_interpretations"], "possible_interpretations")
    if not 1 <= len(possible) <= 5:
        raise ValidationError("possible_interpretations must contain 1-5 strings")
    cleaned["possible_interpretations"] = possible
    cleaned["uncertainty_notes"] = str(obj["uncertainty_notes"]).strip()
    cleaned["quality_flags"] = validate_string_list(obj["quality_flags"], "quality_flags")
    return cleaned


def load_completed_ids(jsonl_path: Path) -> set[str]:
    completed: set[str] = set()
    if not jsonl_path.exists():
        return completed
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("validation_status") == "ok" and row.get("stimulus_id"):
                completed.add(str(row["stimulus_id"]))
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def append_failure(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "stimulus_id",
                "condition",
                "image_path",
                "attempt",
                "error_type",
                "error",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def csv_ready(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda x: isinstance(x, (list, dict))).any():
            out[col] = out[col].map(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
            )
    return out


def write_tables(jsonl_path: Path, csv_path: Path, parquet_path: Path) -> None:
    rows = [r for r in load_jsonl_rows(jsonl_path) if r.get("validation_status") == "ok"]
    if not rows:
        print("No successful annotations to write to table.")
        return
    df = pd.DataFrame(rows)
    csv_ready(df).to_csv(csv_path, index=False)
    print(f"Wrote CSV: {csv_path}")
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Wrote parquet: {parquet_path}")
    except Exception as exc:
        print(f"Parquet not written ({exc.__class__.__name__}: {exc})")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def annotate_one(
    stimulus: Stimulus,
    api_key: str,
    args: argparse.Namespace,
    raw_dir: Path,
) -> dict[str, Any]:
    data_url, image_meta = resize_and_encode_image(Path(stimulus.image_path), args.image_size)
    attempts = []
    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        prompt = USER_PROMPT if attempt == 1 else USER_PROMPT + CORRECTION_SUFFIX
        try:
            response_json, used_response_format = call_openrouter(
                api_key=api_key,
                model=args.model,
                data_url=data_url,
                user_prompt=prompt,
                timeout=args.timeout,
                use_response_format=not args.no_response_format,
            )
            content = extract_message_content(response_json)
            parsed = parse_json_object(content)
            annotation = validate_annotation(parsed)
            attempts.append(
                {
                    "attempt": attempt,
                    "ok": True,
                    "used_response_format": used_response_format,
                    "content": content,
                    "response": response_json,
                }
            )
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{sanitize_name(stimulus.stimulus_id)}.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "stimulus": asdict(stimulus),
                        "model": args.model,
                        "prompt_version": PROMPT_VERSION,
                        "attempts": attempts,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return {
                **asdict(stimulus),
                **image_meta,
                **annotation,
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "validation_status": "ok",
                "raw_response_path": str(raw_path),
                "annotated_at": datetime.now().isoformat(timespec="seconds"),
            }
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            attempts.append({"attempt": attempt, "ok": False, "error": last_error})
            if attempt < args.max_retries:
                delay = min(2 ** (attempt - 1), 30)
                time.sleep(delay)

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{sanitize_name(stimulus.stimulus_id)}.json"
    raw_path.write_text(
        json.dumps(
            {
                "stimulus": asdict(stimulus),
                "model": args.model,
                "prompt_version": PROMPT_VERSION,
                "attempts": attempts,
                "final_error": last_error,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raise RuntimeError(last_error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemini-3.5-flash")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--condition", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-condition", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run-one-image", action="store_true")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--list-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.image_size < 64:
        raise ValueError("--image-size must be >= 64")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = sanitize_name(args.model.replace("/", "__"))
    output_stem = args.output_name or f"vlm_annotations__{safe_model}__{timestamp}"

    annotations_dir = BASE_DIR / "outputs" / "annotations"
    raw_dir = BASE_DIR / "outputs" / "raw_responses" / output_stem
    logs_dir = BASE_DIR / "logs"
    jsonl_path = annotations_dir / f"{output_stem}.jsonl"
    csv_path = annotations_dir / f"{output_stem}.csv"
    parquet_path = annotations_dir / f"{output_stem}.parquet"
    metadata_path = annotations_dir / f"run_metadata__{output_stem}.json"
    failure_path = logs_dir / f"vlm_annotation_failures__{output_stem}.csv"

    if args.overwrite:
        for path in [jsonl_path, csv_path, parquet_path, metadata_path, failure_path]:
            path.unlink(missing_ok=True)
        if raw_dir.exists():
            shutil.rmtree(raw_dir)

    stimuli = list_stimuli(args.image_root, args.condition)
    if args.limit_per_condition is not None:
        per_condition_counts: dict[str, int] = {}
        limited_stimuli = []
        for stimulus in stimuli:
            count = per_condition_counts.get(stimulus.condition, 0)
            if count < args.limit_per_condition:
                limited_stimuli.append(stimulus)
                per_condition_counts[stimulus.condition] = count + 1
        stimuli = limited_stimuli
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index or 0
        end = args.end_index
        stimuli = stimuli[start:end]
    if args.limit is not None:
        stimuli = stimuli[:args.limit]
    if args.dry_run_one_image:
        stimuli = stimuli[:1]

    counts = pd.Series([s.condition for s in stimuli]).value_counts().sort_index()
    print("Images selected:", flush=True)
    for condition, count in counts.items():
        print(f"  {condition}: {count}", flush=True)
    print(f"Total images: {len(stimuli)}", flush=True)

    if args.list_images:
        return 0
    if not stimuli:
        print("No images selected.")
        return 1

    completed = set() if args.overwrite else load_completed_ids(jsonl_path)
    to_run = [s for s in stimuli if args.overwrite or s.stimulus_id not in completed]
    print(f"Already completed in this output: {len(completed)}", flush=True)
    print(f"Images to annotate: {len(to_run)}", flush=True)

    run_metadata = {
        "model": args.model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "image_root": str(args.image_root),
        "image_size": args.image_size,
        "api_endpoint": OPENROUTER_ENDPOINT,
        "script_path": str(Path(__file__).resolve()),
        "git_commit": git_commit(),
        "cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_images_selected": len(stimuli),
        "n_images_to_annotate": len(to_run),
        "output_jsonl": str(jsonl_path),
        "output_csv": str(csv_path),
        "output_parquet": str(parquet_path),
        "raw_response_dir": str(raw_dir),
        "failure_log": str(failure_path),
    }
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"Wrote run metadata: {metadata_path}", flush=True)

    api_key = read_api_key(args.api_key_file)
    failures = 0
    successes = 0

    def handle_success(row: dict[str, Any]) -> None:
        append_jsonl(jsonl_path, row)
        if args.dry_run_one_image:
            print("\nRaw response content:", flush=True)
            raw = json.loads(Path(row["raw_response_path"]).read_text(encoding="utf-8"))
            print(raw["attempts"][-1]["content"], flush=True)
            print("\nParsed annotation:", flush=True)
            print(json.dumps({k: row[k] for k in SCALAR_FIELDS + TEXT_FIELDS}, indent=2), flush=True)

    def handle_failure(stimulus: Stimulus, exc: Exception) -> None:
        append_failure(
            failure_path,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stimulus_id": stimulus.stimulus_id,
                "condition": stimulus.condition,
                "image_path": stimulus.image_path,
                "attempt": args.max_retries,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        )

    if args.workers == 1 or args.dry_run_one_image:
        for i, stimulus in enumerate(to_run, start=1):
            print(f"[{i}/{len(to_run)}] {stimulus.stimulus_id} ({stimulus.condition})", flush=True)
            try:
                row = annotate_one(stimulus, api_key, args, raw_dir)
                handle_success(row)
                successes += 1
            except Exception as exc:
                failures += 1
                handle_failure(stimulus, exc)
                print(f"  FAILED: {exc}", flush=True)
            if args.sleep > 0 and i < len(to_run):
                time.sleep(args.sleep)
    else:
        print(f"Running with {args.workers} workers.", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_stimulus = {}
            for stimulus in to_run:
                print(f"Submitting {stimulus.stimulus_id} ({stimulus.condition})", flush=True)
                future = pool.submit(annotate_one, stimulus, api_key, args, raw_dir)
                future_to_stimulus[future] = stimulus
                if args.sleep > 0:
                    time.sleep(args.sleep)
            completed_count = 0
            for future in as_completed(future_to_stimulus):
                stimulus = future_to_stimulus[future]
                completed_count += 1
                try:
                    row = future.result()
                    handle_success(row)
                    successes += 1
                    print(
                        f"[{completed_count}/{len(to_run)}] ok {stimulus.stimulus_id} ({stimulus.condition})",
                        flush=True,
                    )
                except Exception as exc:
                    failures += 1
                    handle_failure(stimulus, exc)
                    print(
                        f"[{completed_count}/{len(to_run)}] FAILED {stimulus.stimulus_id}: {exc}",
                        flush=True,
                    )

    write_tables(jsonl_path, csv_path, parquet_path)
    final_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    final_metadata.update(
        {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "n_successes_this_run": successes,
            "n_failures_this_run": failures,
        }
    )
    metadata_path.write_text(json.dumps(final_metadata, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"  successes: {successes}")
    print(f"  failures: {failures}")
    print(f"  jsonl: {jsonl_path}")
    print(f"  csv: {csv_path if csv_path.exists() else 'not written'}")
    print(f"  failures log: {failure_path if failure_path.exists() else 'none'}")
    if failures:
        print("Try a fallback model if failures indicate image input or JSON issues:")
        print("  google/gemini-2.5-flash")
        print("  google/gemini-2.5-flash-lite")
        print("  qwen/qwen2.5-vl-72b-instruct")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
