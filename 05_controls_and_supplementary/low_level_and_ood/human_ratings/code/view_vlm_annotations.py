#!/usr/bin/env python3
"""Small local web viewer for VLM annotation results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import mimetypes
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = BASE_DIR / "outputs" / "annotations"
LIST_COLUMNS = {"main_objects", "possible_interpretations", "quality_flags"}
SCALAR_COLUMNS = [
    "recognizability",
    "ambiguity",
    "natural_photo_typicality",
    "visual_clutter",
    "object_centricity",
    "scene_centricity",
    "caption_confidence",
]
FLAG_COLUMNS = [
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
COUNT_COLUMNS = [
    "estimated_salient_object_count",
    "estimated_distinct_object_categories",
]
CATEGORY_COLUMNS = [
    "dominant_content_type",
    "image_style",
    "semantic_domain",
]
TEXT_COLUMNS = [
    "short_caption",
    "uncertainty_notes",
]
REVIEWABLE_COLUMNS = (
    SCALAR_COLUMNS
    + COUNT_COLUMNS
    + FLAG_COLUMNS
    + CATEGORY_COLUMNS
    + TEXT_COLUMNS
    + ["main_objects", "possible_interpretations", "quality_flags"]
)
ANALYSIS_NUMERIC_COLUMNS = SCALAR_COLUMNS + COUNT_COLUMNS
BASELINE_CONDITION_CANDIDATES = ("vicco", "shared_vicco", "baseline")


def latest_annotation_file() -> Path:
    candidates = [
        p for p in ANNOTATION_DIR.glob("*.csv")
        if not p.name.startswith("run_metadata__")
    ]
    candidates += [
        p for p in ANNOTATION_DIR.glob("*.jsonl")
        if not p.name.startswith("run_metadata__")
    ]
    if not candidates:
        raise FileNotFoundError(f"No CSV/JSONL annotation files found in {ANNOTATION_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_cell(value: str):
    if value == "":
        return None
    stripped = value.strip()
    if stripped in {"True", "False"}:
        return stripped == "True"
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [
                {key: parse_cell(value) for key, value in row.items()}
                for row in csv.DictReader(f)
            ]
    raise ValueError(f"Unsupported input type: {path}")


def rows_for_client(rows: list[dict]) -> list[dict]:
    out = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["_row_index"] = idx
        for col in LIST_COLUMNS:
            if isinstance(item.get(col), str):
                try:
                    item[col] = json.loads(item[col])
                except json.JSONDecodeError:
                    item[col] = [item[col]]
        out.append(item)
    return out


def default_review_path(input_path: Path) -> Path:
    return ANNOTATION_DIR / f"review_state__{input_path.stem}.json"


def reviewed_csv_path(input_path: Path) -> Path:
    return ANNOTATION_DIR / f"reviewed__{input_path.stem}.csv"


def load_review_state(path: Path) -> dict:
    if not path.exists():
        return {"fields": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"fields": {}}
    if not isinstance(state, dict):
        return {"fields": {}}
    state.setdefault("fields", {})
    return state


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_review_value(field: str, value, original=None):
    if field in SCALAR_COLUMNS or field in COUNT_COLUMNS:
        if value in ("", None, "null", "None"):
            return None
        return int(value)
    if field in FLAG_COLUMNS:
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        raise ValueError(f"Expected boolean for {field}, got {value!r}")
    if isinstance(original, bool):
        return str(value).strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(original, int) and not isinstance(original, bool):
        if value in ("", None, "null", "None"):
            return None
        return int(value)
    if isinstance(original, float):
        if value in ("", None, "null", "None"):
            return None
        return float(value)
    if field in LIST_COLUMNS:
        if isinstance(value, list):
            return value
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON list for {field}")
            return parsed
        return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    return "" if value is None else str(value)


def csv_ready_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_reviewed_csv(rows: list[dict], review_state: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edited_rows = []
    field_state = review_state.get("fields", {})
    for row in rows:
        out = dict(row)
        stim_state = field_state.get(str(row.get("stimulus_id")), {})
        for field, record in stim_state.items():
            if str(field).startswith("_"):
                continue
            if isinstance(record, dict) and "value" in record:
                out[field] = record["value"]
            if isinstance(record, dict) and record.get("status"):
                out[f"review_status__{field}"] = record["status"]
        edited_rows.append(out)
    fieldnames = sorted({key for row in edited_rows for key in row.keys() if not key.startswith("_")})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in edited_rows:
            writer.writerow({key: csv_ready_value(row.get(key)) for key in fieldnames})


def analysis_value(row: dict, field: str, review_state: dict):
    record = review_state.get("fields", {}).get(str(row.get("stimulus_id")), {}).get(field)
    if isinstance(record, dict):
        if record.get("status") == "rejected":
            return None
        if "value" in record:
            return record["value"]
    return row.get(field)


def numeric_value(row: dict, field: str, review_state: dict) -> float | None:
    value = analysis_value(row, field, review_state)
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def bool_value(row: dict, field: str, review_state: dict) -> bool | None:
    value = analysis_value(row, field, review_state)
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def cohen_d(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    mean_a, sd_a = mean_sd(a)
    mean_b, sd_b = mean_sd(b)
    if mean_a is None or mean_b is None or sd_a is None or sd_b is None:
        return None
    denom_df = len(a) + len(b) - 2
    if denom_df <= 0:
        return None
    pooled = math.sqrt(((len(a) - 1) * sd_a ** 2 + (len(b) - 1) * sd_b ** 2) / denom_df)
    if pooled == 0:
        return None
    return (mean_a - mean_b) / pooled


def prevalence(values: list[bool]) -> tuple[int, int, float | None]:
    n = len(values)
    if n == 0:
        return 0, 0, None
    true_n = sum(1 for value in values if value)
    return true_n, n, true_n / n


def odds_ratio(true_a: int, n_a: int, true_b: int, n_b: int) -> float | None:
    if n_a <= 0 or n_b <= 0:
        return None
    false_a = n_a - true_a
    false_b = n_b - true_b
    return ((true_a + 0.5) / (false_a + 0.5)) / ((true_b + 0.5) / (false_b + 0.5))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def compute_analysis(rows: list[dict], review_state: dict) -> dict:
    conditions = sorted({
        str(row.get("condition"))
        for row in rows
        if row.get("condition") not in (None, "")
    })
    if not conditions:
        return {"error": "No condition labels found."}

    baseline = next(
        (condition for condition in BASELINE_CONDITION_CANDIDATES if condition in conditions),
        conditions[0],
    )
    by_condition = {
        condition: [row for row in rows if str(row.get("condition")) == condition]
        for condition in conditions
    }
    cstim_rows = [row for row in rows if str(row.get("condition")) != baseline]
    comparison_groups = [("all_cstims", cstim_rows)] + [
        (condition, condition_rows)
        for condition, condition_rows in by_condition.items()
        if condition != baseline
    ]
    baseline_rows = by_condition.get(baseline, [])

    n_by_condition = [
        {"condition": condition, "n": len(condition_rows)}
        for condition, condition_rows in by_condition.items()
    ]

    condition_means = []
    for condition, condition_rows in by_condition.items():
        for field in ANALYSIS_NUMERIC_COLUMNS:
            values = [
                value for value in (
                    numeric_value(row, field, review_state) for row in condition_rows
                )
                if value is not None
            ]
            mean, sd = mean_sd(values)
            condition_means.append(
                {
                    "condition": condition,
                    "field": field,
                    "n": len(values),
                    "mean": mean,
                    "sd": sd,
                }
            )

    scalar_effects = []
    for group_name, group_rows in comparison_groups:
        for field in ANALYSIS_NUMERIC_COLUMNS:
            group_values = [
                value for value in (
                    numeric_value(row, field, review_state) for row in group_rows
                )
                if value is not None
            ]
            baseline_values = [
                value for value in (
                    numeric_value(row, field, review_state) for row in baseline_rows
                )
                if value is not None
            ]
            group_mean, group_sd = mean_sd(group_values)
            baseline_mean, baseline_sd = mean_sd(baseline_values)
            scalar_effects.append(
                {
                    "group": group_name,
                    "field": field,
                    "n_group": len(group_values),
                    "n_baseline": len(baseline_values),
                    "mean_group": group_mean,
                    "mean_baseline": baseline_mean,
                    "sd_group": group_sd,
                    "sd_baseline": baseline_sd,
                    "d": cohen_d(group_values, baseline_values),
                }
            )

    binary_effects = []
    for group_name, group_rows in comparison_groups:
        for field in FLAG_COLUMNS:
            group_values = [
                value for value in (
                    bool_value(row, field, review_state) for row in group_rows
                )
                if value is not None
            ]
            baseline_values = [
                value for value in (
                    bool_value(row, field, review_state) for row in baseline_rows
                )
                if value is not None
            ]
            true_group, n_group, prev_group = prevalence(group_values)
            true_baseline, n_baseline, prev_baseline = prevalence(baseline_values)
            diff = None
            if prev_group is not None and prev_baseline is not None:
                diff = prev_group - prev_baseline
            binary_effects.append(
                {
                    "group": group_name,
                    "field": field,
                    "true_group": true_group,
                    "n_group": n_group,
                    "true_baseline": true_baseline,
                    "n_baseline": n_baseline,
                    "prevalence_group": prev_group,
                    "prevalence_baseline": prev_baseline,
                    "difference": diff,
                    "odds_ratio": odds_ratio(true_group, n_group, true_baseline, n_baseline),
                }
            )

    correlation_matrix = []
    for field_a in ANALYSIS_NUMERIC_COLUMNS:
        values = {}
        for field_b in ANALYSIS_NUMERIC_COLUMNS:
            xs = []
            ys = []
            for row in rows:
                x = numeric_value(row, field_a, review_state)
                y = numeric_value(row, field_b, review_state)
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            values[field_b] = pearson(xs, ys)
        correlation_matrix.append({"field": field_a, "values": values})

    top_domains = []
    for condition, condition_rows in by_condition.items():
        counts: dict[str, int] = {}
        for row in condition_rows:
            value = analysis_value(row, "semantic_domain", review_state)
            if value in (None, ""):
                continue
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]:
            top_domains.append(
                {
                    "condition": condition,
                    "semantic_domain": domain,
                    "n": count,
                    "prevalence": count / len(condition_rows) if condition_rows else None,
                }
            )

    baseline_domain_counts: dict[str, int] = {}
    cstim_domain_counts: dict[str, int] = {}
    for row in baseline_rows:
        value = analysis_value(row, "semantic_domain", review_state)
        if value not in (None, ""):
            key = str(value)
            baseline_domain_counts[key] = baseline_domain_counts.get(key, 0) + 1
    for row in cstim_rows:
        value = analysis_value(row, "semantic_domain", review_state)
        if value not in (None, ""):
            key = str(value)
            cstim_domain_counts[key] = cstim_domain_counts.get(key, 0) + 1
    domain_effects = []
    for domain in sorted(set(baseline_domain_counts) | set(cstim_domain_counts)):
        cstim_prev = cstim_domain_counts.get(domain, 0) / len(cstim_rows) if cstim_rows else None
        baseline_prev = baseline_domain_counts.get(domain, 0) / len(baseline_rows) if baseline_rows else None
        diff = None
        if cstim_prev is not None and baseline_prev is not None:
            diff = cstim_prev - baseline_prev
        domain_effects.append(
            {
                "semantic_domain": domain,
                "n_cstims": cstim_domain_counts.get(domain, 0),
                "n_baseline": baseline_domain_counts.get(domain, 0),
                "prevalence_cstims": cstim_prev,
                "prevalence_baseline": baseline_prev,
                "difference": diff,
            }
        )

    return {
        "baseline_condition": baseline,
        "n_rows": len(rows),
        "n_by_condition": n_by_condition,
        "condition_means": condition_means,
        "scalar_effects": scalar_effects,
        "binary_effects": binary_effects,
        "correlation_matrix": correlation_matrix,
        "top_domains": top_domains,
        "domain_effects": domain_effects,
        "notes": [
            "Positive scalar d means the CSTIM group is higher than the baseline condition.",
            "Positive binary difference means the flag is more prevalent in the CSTIM group.",
            "Edited review values are used; rejected fields are treated as missing.",
        ],
    }


def app_html(input_path: Path, n_rows: int) -> bytes:
    title = f"VLM Annotation Viewer ({input_path.name})"
    escaped_title = html.escape(title)
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #171717;
      --muted: #666;
      --line: #d8d8d8;
      --panel: #f6f6f3;
      --accent: #0f766e;
      --warn: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 12px 18px;
      display: grid;
      grid-template-columns: minmax(240px, 1fr) auto auto auto;
      gap: 12px;
      align-items: center;
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .sub {{
      margin-top: 3px;
      font-size: 12px;
      color: var(--muted);
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .tabs {{
      display: flex;
      gap: 6px;
      align-items: center;
    }}
    .tab.active {{
      border-color: var(--accent);
      color: var(--accent);
      background: color-mix(in srgb, var(--accent) 8%, #fff);
    }}
    select, input, textarea {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      font-size: 13px;
      padding: 0 9px;
      border-radius: 4px;
    }}
    select, input {{
      height: 32px;
    }}
    textarea {{
      width: 100%;
      min-height: 62px;
      padding: 7px 9px;
      resize: vertical;
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.3;
    }}
    .count {{
      min-width: 88px;
      text-align: right;
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      min-height: calc(100vh - 65px);
    }}
    .workspace {{
      display: grid;
      grid-template-columns: 1fr 390px;
      min-height: calc(100vh - 65px);
    }}
    .hidden {{
      display: none !important;
    }}
    .grid {{
      padding: 16px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 12px;
      align-content: start;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #fff;
      cursor: pointer;
    }}
    .card.selected {{
      outline: 3px solid color-mix(in srgb, var(--accent) 38%, transparent);
      border-color: var(--accent);
    }}
    .card.review-rejected {{
      border-color: color-mix(in srgb, var(--warn) 60%, var(--line));
    }}
    .thumb {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      display: block;
      background: var(--panel);
    }}
    .card-body {{
      padding: 9px 10px 10px;
      min-height: 112px;
    }}
    .idline {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .caption {{
      font-size: 13px;
      line-height: 1.3;
      min-height: 35px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 5px;
      margin-top: 8px;
    }}
    .metric {{
      background: var(--panel);
      border-radius: 4px;
      padding: 5px 4px;
      text-align: center;
      font-size: 11px;
    }}
    .metric b {{
      display: block;
      font-size: 14px;
      margin-top: 1px;
    }}
    aside {{
      border-left: 1px solid var(--line);
      background: #fbfbfa;
      padding: 16px;
      position: sticky;
      top: 65px;
      height: calc(100vh - 65px);
      overflow: auto;
    }}
    .detail-img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .detail-title {{
      margin: 12px 0 4px;
      font-size: 17px;
      font-weight: 700;
    }}
    .detail-caption {{
      font-size: 14px;
      line-height: 1.35;
      margin-bottom: 12px;
    }}
    .section {{
      border-top: 1px solid var(--line);
      padding-top: 11px;
      margin-top: 12px;
    }}
    .section h2 {{
      margin: 0 0 8px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
    }}
    .kv {{
      display: grid;
      grid-template-columns: minmax(140px, 1fr) auto;
      gap: 6px 12px;
      font-size: 13px;
      padding: 3px 0;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
    }}
    .flag {{
      color: var(--accent);
      border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
    }}
    .warn {{
      color: var(--warn);
      border-color: color-mix(in srgb, var(--warn) 45%, var(--line));
    }}
    .review-badges {{
      display: flex;
      gap: 5px;
      margin-top: 7px;
      min-height: 18px;
    }}
    .badge {{
      border-radius: 999px;
      padding: 2px 6px;
      font-size: 11px;
      background: var(--panel);
      color: var(--muted);
    }}
    .badge.accepted {{ color: var(--accent); }}
    .badge.rejected {{ color: var(--warn); }}
    .badge.edited {{ color: #1d4ed8; }}
    .field-row {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      margin-bottom: 8px;
      background: #fff;
    }}
    .field-row.accepted {{ border-color: color-mix(in srgb, var(--accent) 45%, var(--line)); }}
    .field-row.rejected {{ border-color: color-mix(in srgb, var(--warn) 55%, var(--line)); }}
    .field-row.edited {{ border-color: color-mix(in srgb, #1d4ed8 45%, var(--line)); }}
    .field-head {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      margin-bottom: 5px;
      font-size: 12px;
      color: var(--muted);
    }}
    .field-head strong {{
      color: var(--ink);
      font-size: 13px;
    }}
    .field-actions {{
      display: flex;
      gap: 6px;
      margin-top: 6px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      height: 28px;
      padding: 0 8px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    }}
    button:hover {{
      border-color: #999;
    }}
    button.accept {{ color: var(--accent); }}
    button.reject {{ color: var(--warn); }}
    button.save {{ color: #1d4ed8; }}
    .review-note {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.35;
      margin-bottom: 10px;
    }}
    .analysis-view {{
      padding: 18px;
      max-width: 1500px;
      margin: 0 auto;
    }}
    .analysis-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .analysis-panel {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 12px;
    }}
    .analysis-panel.wide {{
      grid-column: 1 / -1;
    }}
    .analysis-panel h2 {{
      margin: 0 0 10px;
      font-size: 14px;
    }}
    .analysis-panel.compact {{
      padding-bottom: 8px;
    }}
    .analysis-note {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 5px 6px;
      text-align: right;
      vertical-align: top;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: #fbfbfa;
      position: sticky;
      top: 0;
    }}
    .effect-cell {{
      min-width: 110px;
    }}
    .bar-track {{
      height: 8px;
      background: var(--panel);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 3px;
    }}
    .bar-fill {{
      height: 100%;
      background: var(--accent);
    }}
    .bar-fill.negative {{
      background: var(--warn);
    }}
    .corr-cell {{
      text-align: center;
      min-width: 56px;
    }}
    .plot {{
      width: 100%;
      overflow-x: auto;
    }}
    .plot svg {{
      display: block;
      width: 100%;
      height: auto;
      font-family: Arial, Helvetica, sans-serif;
    }}
    .plot text {{
      font-size: 11px;
      fill: var(--ink);
    }}
    .plot .muted {{
      fill: var(--muted);
    }}
    .plot .axis {{
      stroke: var(--line);
      stroke-width: 1;
    }}
    .plot .zero {{
      stroke: #777;
      stroke-width: 1.2;
    }}
    .plot .pos {{
      fill: var(--accent);
    }}
    .plot .neg {{
      fill: var(--warn);
    }}
    .plot .dot {{
      stroke: #fff;
      stroke-width: 1.3;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 4px 0 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    .stat-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }}
    .stat strong {{
      display: block;
      font-size: 18px;
    }}
    @media (max-width: 900px) {{
      header {{ grid-template-columns: 1fr; }}
      .count {{ text-align: left; }}
      .workspace {{ grid-template-columns: 1fr; }}
      aside {{
        position: static;
        height: auto;
        border-left: 0;
        border-top: 1px solid var(--line);
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>VLM Annotation Viewer</h1>
      <div class="sub">{html.escape(input_path.name)} · {n_rows} annotations</div>
    </div>
    <div class="tabs">
      <button class="tab active" data-tab="labeling">Labeling</button>
      <button class="tab" data-tab="analysis">Analysis</button>
    </div>
    <div class="controls">
      <select id="condition"></select>
      <select id="sort">
        <option value="file_index">File order</option>
        <option value="recognizability">Recognizability</option>
        <option value="ambiguity">Ambiguity</option>
        <option value="visual_clutter">Visual clutter</option>
        <option value="natural_photo_typicality">Naturalness</option>
        <option value="caption_confidence">Caption confidence</option>
      </select>
      <input id="search" type="search" placeholder="Search captions, objects, domains">
    </div>
    <div id="count" class="count"></div>
  </header>
  <main>
    <section id="labeling-view" class="workspace">
      <section id="grid" class="grid"></section>
      <aside id="detail"></aside>
    </section>
    <section id="analysis-view" class="analysis-view hidden"></section>
  </main>
  <script>
    const scalarColumns = {json.dumps(SCALAR_COLUMNS)};
    const countColumns = {json.dumps(COUNT_COLUMNS)};
    const flagColumns = {json.dumps(FLAG_COLUMNS)};
    const categoryColumns = {json.dumps(CATEGORY_COLUMNS)};
    const textColumns = {json.dumps(TEXT_COLUMNS)};
    const listColumns = {json.dumps(sorted(LIST_COLUMNS))};
    let reviewableColumns = {json.dumps(REVIEWABLE_COLUMNS)};
    let rows = [];
    let filtered = [];
    let selectedIndex = null;
    let reviewState = {{fields: {{}}}};
    let reviewPath = null;
    let reviewedCsv = null;
    let analysisData = null;
    let activeTab = 'labeling';

    const el = id => document.getElementById(id);
    const imageUrl = row => `/image/${{row._row_index}}`;
    const asList = value => Array.isArray(value) ? value : (value ? [value] : []);
    const textOf = value => Array.isArray(value) ? value.join(' ') : String(value ?? '');
    const fieldRecord = (row, field) => reviewState.fields?.[row.stimulus_id]?.[field] || {{}};
    const fieldStatus = (row, field) => fieldRecord(row, field).status || '';
    const fieldValue = (row, field) => {{
      const record = fieldRecord(row, field);
      if (Object.prototype.hasOwnProperty.call(record, 'value')) return record.value;
      return row[field];
    }};

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }}[ch]));
    }}

    function setupFilters() {{
      const conditionValues = [...new Set(rows.map(r => r.condition).filter(Boolean))].sort();
      const conditions = ['all', ...conditionValues];
      el('condition').innerHTML = conditions
        .map(c => `<option value="${{escapeHtml(c)}}">${{escapeHtml(c)}}</option>`)
        .join('');
      el('condition').addEventListener('change', render);
      el('sort').addEventListener('change', render);
      el('search').addEventListener('input', render);
    }}

    function setupTabs() {{
      document.querySelectorAll('[data-tab]').forEach(button => {{
        button.addEventListener('click', () => setTab(button.dataset.tab));
      }});
    }}

    function setTab(tab) {{
      activeTab = tab;
      document.querySelectorAll('[data-tab]').forEach(button => {{
        button.classList.toggle('active', button.dataset.tab === tab);
      }});
      el('labeling-view').classList.toggle('hidden', tab !== 'labeling');
      el('analysis-view').classList.toggle('hidden', tab !== 'analysis');
      render();
    }}

    function rowText(row) {{
      return [
        row.stimulus_id, row.condition, fieldValue(row, 'semantic_domain'),
        fieldValue(row, 'dominant_content_type'), fieldValue(row, 'image_style'),
        fieldValue(row, 'short_caption'),
        textOf(fieldValue(row, 'main_objects')), textOf(fieldValue(row, 'possible_interpretations')),
        textOf(fieldValue(row, 'quality_flags'))
      ].join(' ').toLowerCase();
    }}

    function render() {{
      if (activeTab === 'analysis') {{
        renderAnalysis();
        return;
      }}
      const condition = el('condition').value || 'all';
      const query = el('search').value.trim().toLowerCase();
      const sortKey = el('sort').value;
      filtered = rows.filter(row => condition === 'all' || row.condition === condition);
      if (query) filtered = filtered.filter(row => rowText(row).includes(query));
      filtered.sort((a, b) => {{
        const av = Number(a[sortKey]);
        const bv = Number(b[sortKey]);
        if (Number.isFinite(av) && Number.isFinite(bv)) return bv - av;
        return String(a[sortKey] ?? '').localeCompare(String(b[sortKey] ?? ''));
      }});
      el('count').textContent = `${{filtered.length}} shown`;
      el('grid').innerHTML = filtered.map(row => cardHtml(row)).join('');
      document.querySelectorAll('.card').forEach(card => {{
        card.addEventListener('click', () => selectRow(Number(card.dataset.index)));
      }});
      if (!filtered.some(row => row._row_index === selectedIndex)) {{
        selectedIndex = filtered[0]?._row_index ?? null;
      }}
      updateSelection();
    }}

    function cardHtml(row) {{
      const selected = row._row_index === selectedIndex ? ' selected' : '';
      const statuses = Object.values(reviewState.fields?.[row.stimulus_id] || {{}}).map(r => r.status);
      const nRejected = statuses.filter(s => s === 'rejected').length;
      const nEdited = statuses.filter(s => s === 'edited').length;
      const nAccepted = statuses.filter(s => s === 'accepted').length;
      const reviewClass = nRejected ? ' review-rejected' : '';
      const badges = [
        nAccepted ? `<span class="badge accepted">${{nAccepted}} accepted</span>` : '',
        nEdited ? `<span class="badge edited">${{nEdited}} edited</span>` : '',
        nRejected ? `<span class="badge rejected">${{nRejected}} rejected</span>` : ''
      ].join('');
      return `<article class="card${{selected}}${{reviewClass}}" data-index="${{row._row_index}}">
        <img class="thumb" src="${{imageUrl(row)}}" loading="lazy" alt="">
        <div class="card-body">
          <div class="idline"><span>${{escapeHtml(row.stimulus_id)}}</span><span>${{escapeHtml(fieldValue(row, 'semantic_domain'))}}</span></div>
          <div class="caption">${{escapeHtml(fieldValue(row, 'short_caption'))}}</div>
          <div class="metrics">
            <div class="metric">rec<b>${{escapeHtml(fieldValue(row, 'recognizability'))}}</b></div>
            <div class="metric">amb<b>${{escapeHtml(fieldValue(row, 'ambiguity'))}}</b></div>
            <div class="metric">nat<b>${{escapeHtml(fieldValue(row, 'natural_photo_typicality'))}}</b></div>
            <div class="metric">clut<b>${{escapeHtml(fieldValue(row, 'visual_clutter'))}}</b></div>
          </div>
          <div class="review-badges">${{badges}}</div>
        </div>
      </article>`;
    }}

    function selectRow(index) {{
      selectedIndex = index;
      updateSelection();
    }}

    function updateSelection() {{
      document.querySelectorAll('.card').forEach(card => {{
        card.classList.toggle('selected', Number(card.dataset.index) === selectedIndex);
      }});
      const row = rows.find(r => r._row_index === selectedIndex);
      el('detail').innerHTML = row ? detailHtml(row) : '<p>No annotation selected.</p>';
      if (row) attachReviewButtons(row);
    }}

    function chips(values, cls='') {{
      return asList(values).map(v => `<span class="chip ${{cls}}">${{escapeHtml(v)}}</span>`).join('');
    }}

    function editorValue(value) {{
      if (Array.isArray(value)) return value.join(', ');
      if (value === null || value === undefined) return '';
      return String(value);
    }}

    function fieldEditor(row, field) {{
      const status = fieldStatus(row, field);
      const value = fieldValue(row, field);
      const original = row[field];
      const statusLabel = status ? status : 'unreviewed';
      return `<div class="field-row ${{escapeHtml(status)}}" data-field="${{escapeHtml(field)}}">
        <div class="field-head">
          <strong>${{escapeHtml(field)}}</strong>
          <span>${{escapeHtml(statusLabel)}}</span>
        </div>
        <textarea data-editor="${{escapeHtml(field)}}">${{escapeHtml(editorValue(value))}}</textarea>
        <div class="field-head">
          <span>original</span>
          <span>${{escapeHtml(editorValue(original))}}</span>
        </div>
        <div class="field-actions">
          <button class="accept" data-action="accept" data-field="${{escapeHtml(field)}}">Accept</button>
          <button class="save" data-action="edit" data-field="${{escapeHtml(field)}}">Save edit</button>
          <button class="reject" data-action="reject" data-field="${{escapeHtml(field)}}">Reject</button>
          <button data-action="clear" data-field="${{escapeHtml(field)}}">Clear</button>
        </div>
      </div>`;
    }}

    function attachReviewButtons(row) {{
      document.querySelectorAll('[data-action][data-field]').forEach(button => {{
        button.addEventListener('click', () => {{
          const field = button.dataset.field;
          const action = button.dataset.action;
          const editor = document.querySelector(`[data-editor="${{CSS.escape(field)}}"]`);
          const value = editor ? editor.value : null;
          saveReview(row, field, action, value);
        }});
      }});
    }}

    function saveReview(row, field, action, value) {{
      fetch('/api/review', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{stimulus_id: row.stimulus_id, field, action, value}})
      }})
        .then(resp => {{
          if (!resp.ok) return resp.json().then(data => {{ throw new Error(data.error || resp.statusText); }});
          return resp.json();
        }})
        .then(data => {{
          reviewState = data.review_state;
          reviewedCsv = data.reviewed_csv;
          analysisData = data.analysis || analysisData;
          render();
          const refreshed = rows.find(r => r._row_index === selectedIndex);
          if (refreshed) {{
            setTimeout(() => attachReviewButtons(refreshed), 0);
          }}
        }})
        .catch(err => alert(`Review save failed: ${{err.message || err}}`));
    }}

    function fmt(value, digits=2) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return Number(value).toFixed(digits);
    }}

    function fmtPct(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return `${{(Number(value) * 100).toFixed(1)}}%`;
    }}

    function effectBar(value, scale) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      const width = Math.min(100, Math.abs(Number(value)) / scale * 100);
      const cls = Number(value) < 0 ? ' negative' : '';
      return `<div class="bar-track"><div class="bar-fill${{cls}}" style="width:${{width}}%"></div></div>`;
    }}

    function cellHtml(cell) {{
      if (cell && typeof cell === 'object' && Object.prototype.hasOwnProperty.call(cell, 'html')) {{
        return String(cell.html);
      }}
      return escapeHtml(cell);
    }}

    function htmlTable(headers, rows) {{
      const head = headers.map(header => `<th>${{escapeHtml(header)}}</th>`).join('');
      const body = rows.map(row => `<tr>${{row.map(cell => `<td>${{cellHtml(cell)}}</td>`).join('')}}</tr>`).join('');
      return `<table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table>`;
    }}

    function sortedByAbs(rows, key, limit) {{
      return [...rows]
        .filter(row => row[key] !== null && row[key] !== undefined && Number.isFinite(Number(row[key])))
        .sort((a, b) => Math.abs(Number(b[key])) - Math.abs(Number(a[key])))
        .slice(0, limit);
    }}

    function label(value) {{
      return String(value ?? '').replaceAll('_', ' ');
    }}

    const conditionPalette = ['#0f766e', '#1d4ed8', '#7c3aed', '#b45309', '#be123c', '#047857', '#525252'];

    function conditionNames() {{
      return analysisData.n_by_condition.map(row => row.condition);
    }}

    function conditionColor(condition) {{
      if (condition === analysisData.baseline_condition) return '#525252';
      const names = conditionNames().filter(name => name !== analysisData.baseline_condition);
      const idx = Math.max(0, names.indexOf(condition));
      return conditionPalette[idx % conditionPalette.length];
    }}

    function legendHtml(conditions) {{
      return `<div class="legend">${{conditions.map(condition => `
        <span class="legend-item"><span class="swatch" style="background:${{conditionColor(condition)}}"></span>${{escapeHtml(condition)}}</span>
      `).join('')}}</div>`;
    }}

    function niceScale(rows, key, fallback) {{
      const maxAbs = Math.max(
        fallback,
        ...rows
          .map(row => Math.abs(Number(row[key])))
          .filter(value => Number.isFinite(value))
      );
      return Math.ceil(maxAbs * 10) / 10;
    }}

    function horizontalDivergingPlot(rows, valueKey, labelFn, scale, valueFmt, xLabel) {{
      const plotRows = rows.filter(row => Number.isFinite(Number(row[valueKey])));
      const width = 840;
      const left = 230;
      const right = 70;
      const top = 28;
      const rowH = 28;
      const bottom = 28;
      const height = top + bottom + Math.max(1, plotRows.length) * rowH;
      const inner = width - left - right;
      const zero = left + inner / 2;
      const half = inner / 2;
      const ticks = [-scale, -scale / 2, 0, scale / 2, scale];
      const tickHtml = ticks.map(tick => {{
        const x = zero + (tick / scale) * half;
        return `<g><line class="axis" x1="${{x}}" y1="${{top - 8}}" x2="${{x}}" y2="${{height - bottom + 8}}"></line><text class="muted" x="${{x}}" y="${{height - 7}}" text-anchor="middle">${{valueFmt(tick)}}</text></g>`;
      }}).join('');
      const rowsHtml = plotRows.map((row, idx) => {{
        const y = top + idx * rowH + 8;
        const value = Number(row[valueKey]);
        const clipped = Math.max(-scale, Math.min(scale, value));
        const x = zero + (clipped / scale) * half;
        const barX = Math.min(zero, x);
        const barW = Math.max(2, Math.abs(x - zero));
        const cls = value < 0 ? 'neg' : 'pos';
        const valueX = value < 0 ? barX - 6 : barX + barW + 6;
        const anchor = value < 0 ? 'end' : 'start';
        return `<g>
          <text x="${{left - 10}}" y="${{y + 9}}" text-anchor="end">${{escapeHtml(labelFn(row))}}</text>
          <rect class="${{cls}}" x="${{barX}}" y="${{y}}" width="${{barW}}" height="13" rx="2"></rect>
          <text class="muted" x="${{valueX}}" y="${{y + 10}}" text-anchor="${{anchor}}">${{valueFmt(value)}}</text>
        </g>`;
      }}).join('');
      return `<div class="plot"><svg viewBox="0 0 ${{width}} ${{height}}" role="img">
        <text class="muted" x="${{left}}" y="14">${{escapeHtml(xLabel)}}</text>
        ${{tickHtml}}
        <line class="zero" x1="${{zero}}" y1="${{top - 12}}" x2="${{zero}}" y2="${{height - bottom + 8}}"></line>
        ${{rowsHtml}}
      </svg></div>`;
    }}

    function conditionMeanDotPlot(fields, minValue, maxValue, xLabel) {{
      const conditions = conditionNames();
      const lookup = new Map(analysisData.condition_means.map(row => [`${{row.condition}}::${{row.field}}`, row]));
      const width = 900;
      const left = 230;
      const right = 54;
      const top = 34;
      const rowH = 34;
      const bottom = 34;
      const height = top + bottom + fields.length * rowH;
      const inner = width - left - right;
      const scale = value => left + ((Number(value) - minValue) / (maxValue - minValue || 1)) * inner;
      const ticks = [];
      for (let tick = Math.ceil(minValue); tick <= Math.floor(maxValue); tick += 1) ticks.push(tick);
      if (!ticks.includes(minValue)) ticks.unshift(minValue);
      if (!ticks.includes(maxValue)) ticks.push(maxValue);
      const tickHtml = ticks.map(tick => {{
        const x = scale(tick);
        return `<g><line class="axis" x1="${{x}}" y1="${{top - 10}}" x2="${{x}}" y2="${{height - bottom + 8}}"></line><text class="muted" x="${{x}}" y="${{height - 9}}" text-anchor="middle">${{fmt(tick)}}</text></g>`;
      }}).join('');
      const pointHtml = fields.map((field, fieldIdx) => {{
        const yBase = top + fieldIdx * rowH + 12;
        const labelHtml = `<text x="${{left - 10}}" y="${{yBase + 5}}" text-anchor="end">${{escapeHtml(label(field))}}</text>`;
        const dots = conditions.map((condition, condIdx) => {{
          const row = lookup.get(`${{condition}}::${{field}}`);
          if (!row || !Number.isFinite(Number(row.mean))) return '';
          const offset = (condIdx - (conditions.length - 1) / 2) * 3.2;
          return `<circle class="dot" cx="${{scale(row.mean)}}" cy="${{yBase + offset}}" r="4.6" fill="${{conditionColor(condition)}}"><title>${{escapeHtml(condition)}}: ${{fmt(row.mean)}} n=${{row.n}}</title></circle>`;
        }}).join('');
        return `<g>${{labelHtml}}${{dots}}</g>`;
      }}).join('');
      return `${{legendHtml(conditions)}}<div class="plot"><svg viewBox="0 0 ${{width}} ${{height}}" role="img">
        <text class="muted" x="${{left}}" y="15">${{escapeHtml(xLabel)}}</text>
        ${{tickHtml}}
        ${{pointHtml}}
      </svg></div>`;
    }}

    function conditionCountDotPlot() {{
      const fields = countColumns;
      const values = analysisData.condition_means
        .filter(row => fields.includes(row.field) && Number.isFinite(Number(row.mean)))
        .map(row => Number(row.mean));
      const maxValue = Math.ceil(Math.max(1, ...values));
      return conditionMeanDotPlot(fields, 0, maxValue, 'mean estimated count');
    }}

    function correlationHeatmap() {{
      const fields = analysisData.correlation_matrix.map(row => row.field);
      const cell = 58;
      const left = 205;
      const top = 145;
      const width = left + fields.length * cell + 10;
      const height = top + fields.length * cell + 18;
      const color = value => {{
        if (!Number.isFinite(Number(value))) return '#f6f6f3';
        const v = Math.max(-1, Math.min(1, Number(value)));
        const alpha = 0.15 + Math.abs(v) * 0.72;
        return v < 0 ? `rgba(154, 52, 18, ${{alpha}})` : `rgba(15, 118, 110, ${{alpha}})`;
      }};
      const labelsTop = fields.map((field, idx) => {{
        const x = left + idx * cell + cell / 2;
        return `<text class="muted" transform="translate(${{x}}, ${{top - 8}}) rotate(-45)" text-anchor="start">${{escapeHtml(label(field))}}</text>`;
      }}).join('');
      const rowsHtml = analysisData.correlation_matrix.map((row, yIdx) => {{
        const y = top + yIdx * cell;
        const rowLabel = `<text x="${{left - 10}}" y="${{y + cell / 2 + 4}}" text-anchor="end">${{escapeHtml(label(row.field))}}</text>`;
        const cells = fields.map((field, xIdx) => {{
          const x = left + xIdx * cell;
          const r = row.values[field];
          const textColor = Math.abs(Number(r)) > 0.62 ? '#fff' : '#171717';
          return `<g><rect x="${{x}}" y="${{y}}" width="${{cell - 2}}" height="${{cell - 2}}" rx="2" fill="${{color(r)}}"></rect><text x="${{x + cell / 2 - 1}}" y="${{y + cell / 2 + 4}}" text-anchor="middle" fill="${{textColor}}">${{fmt(r)}}</text></g>`;
        }}).join('');
        return `<g>${{rowLabel}}${{cells}}</g>`;
      }}).join('');
      return `<div class="plot"><svg viewBox="0 0 ${{width}} ${{height}}" role="img">${{labelsTop}}${{rowsHtml}}</svg></div>`;
    }}

    function statStrip() {{
      return `<div class="stat-strip">
        <div class="stat"><span>rows</span><strong>${{analysisData.n_rows}}</strong></div>
        <div class="stat"><span>baseline</span><strong>${{escapeHtml(analysisData.baseline_condition)}}</strong></div>
        <div class="stat"><span>conditions</span><strong>${{analysisData.n_by_condition.length}}</strong></div>
        <div class="stat"><span>editable fields</span><strong>${{reviewableColumns.length}}</strong></div>
      </div>`;
    }}

    function conditionCountPlot() {{
      const rowsIn = analysisData.n_by_condition;
      const maxN = Math.max(1, ...rowsIn.map(row => Number(row.n)));
      const width = 760;
      const left = 190;
      const right = 58;
      const top = 18;
      const rowH = 28;
      const height = top + rowsIn.length * rowH + 14;
      const inner = width - left - right;
      const rowsHtml = rowsIn.map((row, idx) => {{
        const y = top + idx * rowH;
        const w = Number(row.n) / maxN * inner;
        return `<g>
          <text x="${{left - 10}}" y="${{y + 13}}" text-anchor="end">${{escapeHtml(row.condition)}}</text>
          <rect x="${{left}}" y="${{y + 2}}" width="${{w}}" height="14" rx="2" fill="${{conditionColor(row.condition)}}"></rect>
          <text class="muted" x="${{left + w + 6}}" y="${{y + 13}}">${{row.n}}</text>
        </g>`;
      }}).join('');
      return `<div class="plot"><svg viewBox="0 0 ${{width}} ${{height}}" role="img">${{rowsHtml}}</svg></div>`;
    }}

    function scalarEffectTable(rows) {{
      return htmlTable(
        ['group', 'field', 'mean', 'baseline', 'd'],
        rows.map(row => [
          row.group,
          row.field,
          fmt(row.mean_group),
          fmt(row.mean_baseline),
          {{html: `<div class="effect-cell">${{fmt(row.d)}}${{effectBar(row.d, 1.5)}}</div>`}},
        ])
      );
    }}

    function binaryEffectTable(rows) {{
      return htmlTable(
        ['group', 'field', 'prev', 'baseline', 'diff', 'OR'],
        rows.map(row => [
          row.group,
          row.field,
          fmtPct(row.prevalence_group),
          fmtPct(row.prevalence_baseline),
          {{html: `<div class="effect-cell">${{fmtPct(row.difference)}}${{effectBar(row.difference, 0.35)}}</div>`}},
          fmt(row.odds_ratio),
        ])
      );
    }}

    function conditionMeansTable() {{
      const conditions = analysisData.n_by_condition.map(row => row.condition);
      const lookup = new Map(analysisData.condition_means.map(row => [`${{row.condition}}::${{row.field}}`, row]));
      const rowsOut = analysisData.condition_means
        .map(row => row.field)
        .filter((field, index, arr) => arr.indexOf(field) === index)
        .map(field => [
          field,
          ...conditions.map(condition => {{
            const row = lookup.get(`${{condition}}::${{field}}`);
            return row ? fmt(row.mean) : '';
          }})
        ]);
      return htmlTable(['field', ...conditions], rowsOut);
    }}

    function conditionCountsTable() {{
      return htmlTable(
        ['condition', 'n'],
        analysisData.n_by_condition.map(row => [row.condition, row.n])
      );
    }}

    function domainTable() {{
      return htmlTable(
        ['condition', 'semantic_domain', 'n', 'prev'],
        analysisData.top_domains
          .filter(row => row.n >= 5)
          .slice(0, 60)
          .map(row => [row.condition, row.semantic_domain, row.n, fmtPct(row.prevalence)])
      );
    }}

    function corrTable() {{
      const fields = analysisData.correlation_matrix.map(row => row.field);
      const header = `<tr><th></th>${{fields.map(field => `<th>${{escapeHtml(field)}}</th>`).join('')}}</tr>`;
      const body = analysisData.correlation_matrix.map(row => {{
        const cells = fields.map(field => {{
          const r = row.values[field];
          if (r === null || r === undefined || Number.isNaN(Number(r))) {{
            return '<td class="corr-cell"></td>';
          }}
          const value = Number(r);
          const alpha = Math.min(0.85, 0.12 + Math.abs(value) * 0.65);
          const color = value < 0 ? `rgba(154, 52, 18, ${{alpha}})` : `rgba(15, 118, 110, ${{alpha}})`;
          const textColor = Math.abs(value) > 0.65 ? '#fff' : 'var(--ink)';
          return `<td class="corr-cell" style="background:${{color}}; color:${{textColor}}">${{fmt(value)}}</td>`;
        }}).join('');
        return `<tr><td>${{escapeHtml(row.field)}}</td>${{cells}}</tr>`;
      }}).join('');
      return `<table><thead>${{header}}</thead><tbody>${{body}}</tbody></table>`;
    }}

    function renderAnalysis() {{
      el('count').textContent = `${{rows.length}} analyzed`;
      if (!analysisData || analysisData.error) {{
        el('analysis-view').innerHTML = `<p>Analysis unavailable: ${{escapeHtml(analysisData?.error || 'not loaded')}}</p>`;
        return;
      }}
      const scalarOverall = sortedByAbs(
        analysisData.scalar_effects.filter(row => row.group === 'all_cstims'),
        'd',
        99
      );
      const scalarConditionTop = sortedByAbs(
        analysisData.scalar_effects.filter(row => row.group !== 'all_cstims'),
        'd',
        18
      );
      const binaryOverall = sortedByAbs(
        analysisData.binary_effects.filter(row => row.group === 'all_cstims'),
        'difference',
        99
      );
      const binaryConditionTop = sortedByAbs(
        analysisData.binary_effects.filter(row => row.group !== 'all_cstims'),
        'difference',
        18
      );
      const domainTop = sortedByAbs(
        (analysisData.domain_effects || []).filter(row => (row.n_cstims + row.n_baseline) >= 5),
        'difference',
        18
      );
      const notes = analysisData.notes.map(note => `<div class="analysis-note">${{escapeHtml(note)}}</div>`).join('');
      const scalarScale = niceScale(scalarOverall, 'd', 0.7);
      const scalarConditionScale = niceScale(scalarConditionTop, 'd', 0.8);
      const binaryScale = niceScale(binaryOverall, 'difference', 0.25);
      const binaryConditionScale = niceScale(binaryConditionTop, 'difference', 0.25);
      const domainScale = niceScale(domainTop, 'difference', 0.12);
      el('analysis-view').innerHTML = `${{statStrip()}}<div class="analysis-grid">
        <section class="analysis-panel compact">
          <h2>Condition Ns</h2>
          ${{conditionCountPlot()}}
        </section>
        <section class="analysis-panel compact">
          <h2>Read This</h2>
          ${{notes}}
        </section>
        <section class="analysis-panel wide">
          <h2>Mean Scalar Ratings By Condition</h2>
          <p class="analysis-note">Dots are condition means for 1-5 VLM annotation axes.</p>
          ${{conditionMeanDotPlot(scalarColumns, 1, 5, 'mean rating')}}
        </section>
        <section class="analysis-panel wide">
          <h2>Mean Count Estimates By Condition</h2>
          ${{conditionCountDotPlot()}}
        </section>
        <section class="analysis-panel">
          <h2>All CSTIMs vs Baseline: Scalar/Count Effect Sizes</h2>
          <p class="analysis-note">Positive d means CSTIMs are higher than ${{escapeHtml(analysisData.baseline_condition)}}.</p>
          ${{horizontalDivergingPlot(scalarOverall, 'd', row => label(row.field), scalarScale, value => fmt(value), 'Cohen d')}}
        </section>
        <section class="analysis-panel">
          <h2>All CSTIMs vs Baseline: Binary Flag Differences</h2>
          <p class="analysis-note">Positive values mean the flag is more prevalent in CSTIMs.</p>
          ${{horizontalDivergingPlot(binaryOverall, 'difference', row => label(row.field), binaryScale, value => fmtPct(value), 'prevalence difference')}}
        </section>
        <section class="analysis-panel">
          <h2>Largest Condition-Specific Scalar Effects</h2>
          ${{horizontalDivergingPlot(scalarConditionTop, 'd', row => `${{row.group}} · ${{label(row.field)}}`, scalarConditionScale, value => fmt(value), 'Cohen d')}}
        </section>
        <section class="analysis-panel">
          <h2>Largest Condition-Specific Binary Differences</h2>
          ${{horizontalDivergingPlot(binaryConditionTop, 'difference', row => `${{row.group}} · ${{label(row.field)}}`, binaryConditionScale, value => fmtPct(value), 'prevalence difference')}}
        </section>
        <section class="analysis-panel wide">
          <h2>Semantic Domain Prevalence: All CSTIMs vs Baseline</h2>
          ${{horizontalDivergingPlot(domainTop, 'difference', row => label(row.semantic_domain), domainScale, value => fmtPct(value), 'prevalence difference')}}
        </section>
        <section class="analysis-panel wide">
          <h2>Numeric Annotation Correlations</h2>
          ${{correlationHeatmap()}}
        </section>
      </div>`;
    }}

    function detailHtml(row) {{
      const editorRows = reviewableColumns.map(field => fieldEditor(row, field)).join('');
      return `<img class="detail-img" src="${{imageUrl(row)}}" alt="">
        <div class="detail-title">${{escapeHtml(row.stimulus_id)}}</div>
        <div class="detail-caption">${{escapeHtml(fieldValue(row, 'short_caption'))}}</div>
        <div class="section">
          <h2>Editable Fields</h2>
          <div class="review-note">Each row is one table column for this stimulus. Accept, reject, or edit values here; changes are saved to a separate review-state JSON and reviewed CSV, leaving the VLM output unchanged.</div>
          <div class="kv"><span>review state</span><strong>${{escapeHtml(reviewPath || '')}}</strong></div>
          <div class="kv"><span>reviewed CSV</span><strong>${{escapeHtml(reviewedCsv || '')}}</strong></div>
        </div>
        <div class="section">
          ${{editorRows}}
        </div>`;
    }}

    fetch('/api/annotations')
      .then(resp => resp.json())
      .then(data => {{
        rows = data.rows;
        reviewState = data.review_state || {{fields: {{}}}};
        reviewPath = data.review_path || null;
        reviewedCsv = data.reviewed_csv || null;
        analysisData = data.analysis || null;
        reviewableColumns = data.reviewable_columns || reviewableColumns;
        setupFilters();
        setupTabs();
        render();
      }})
      .catch(err => {{
        el('grid').innerHTML = `<p>Failed to load annotations: ${{escapeHtml(err)}}</p>`;
      }});
  </script>
</body>
</html>"""
    return body.encode("utf-8")


def make_handler(rows: list[dict], input_path: Path):
    client_rows = rows_for_client(rows)
    available_columns = {key for row in client_rows for key in row.keys()}
    reviewable_columns = [
        column for column in REVIEWABLE_COLUMNS
        if column in available_columns
    ]
    reviewable_set = set(reviewable_columns)
    review_path = default_review_path(input_path)
    review_csv = reviewed_csv_path(input_path)
    review_state = load_review_state(review_path)
    review_lock = threading.Lock()

    def persist_review_state() -> None:
        atomic_write_json(review_path, review_state)
        write_reviewed_csv(client_rows, review_state, review_csv)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload: dict, status: int = 200) -> None:
            self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(app_html(input_path, len(client_rows)), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/annotations":
                self.send_json(
                    {
                        "input": str(input_path),
                        "rows": client_rows,
                        "review_state": review_state,
                        "review_path": str(review_path),
                        "reviewed_csv": str(review_csv),
                        "reviewable_columns": reviewable_columns,
                        "analysis": compute_analysis(client_rows, review_state),
                    }
                )
                return
            if parsed.path == "/api/analysis":
                self.send_json(compute_analysis(client_rows, review_state))
                return
            if parsed.path.startswith("/image/"):
                try:
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    row = client_rows[index]
                    image_path = Path(str(row["image_path"]))
                    if not image_path.exists():
                        self.send_json({"error": f"Image not found: {image_path}"}, 404)
                        return
                    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                    self.send_bytes(image_path.read_bytes(), content_type)
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 500)
                return
            if parsed.path == "/api/fields":
                query = parse_qs(parsed.query)
                column = query.get("column", ["condition"])[0]
                values = sorted({str(row.get(column, "")) for row in client_rows if row.get(column) not in (None, "")})
                self.send_json({"column": column, "values": values})
                return
            self.send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/review":
                self.send_json({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                stimulus_id = str(payload["stimulus_id"])
                field = str(payload["field"])
                action = str(payload["action"])
                if field not in reviewable_set:
                    raise ValueError(f"Field is not reviewable: {field}")
                if action not in {"accept", "reject", "edit", "clear"}:
                    raise ValueError(f"Unsupported review action: {action}")
                source_row = next(
                    (row for row in client_rows if str(row.get("stimulus_id")) == stimulus_id),
                    None,
                )
                if source_row is None:
                    raise ValueError(f"Unknown stimulus_id: {stimulus_id}")

                with review_lock:
                    fields = review_state.setdefault("fields", {})
                    stim_state = fields.setdefault(stimulus_id, {})
                    if action == "clear":
                        stim_state.pop(field, None)
                        if not stim_state:
                            fields.pop(stimulus_id, None)
                    else:
                        record = {
                            "status": "accepted" if action == "accept" else "rejected" if action == "reject" else "edited",
                            "updated_at": datetime.now().isoformat(timespec="seconds"),
                        }
                        if action == "edit":
                            record["value"] = parse_review_value(
                                field, payload.get("value"), source_row.get(field)
                            )
                        stim_state[field] = record
                    persist_review_state()
                self.send_json(
                    {
                        "ok": True,
                        "review_state": review_state,
                        "review_path": str(review_path),
                        "reviewed_csv": str(review_csv),
                        "reviewable_columns": reviewable_columns,
                        "analysis": compute_analysis(client_rows, review_state),
                    }
                )
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input or latest_annotation_file()
    input_path = input_path.resolve()
    rows = load_rows(input_path)
    if not rows:
        print(f"No rows in {input_path}", file=sys.stderr)
        return 1
    handler = make_handler(rows, input_path)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {len(rows)} annotations from {input_path}")
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping viewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
