#!/usr/bin/env python3
"""Download high-resolution selected images from LAION URLs.

Outputs:
- images/image_<idx>.png: Individual high-res images
- images/selected_images_fullres.pdf: Grid of all images
- images/image_manifest.csv: Metadata for each image
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image
from tqdm import tqdm

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.utils import (
    load_selection_payload,
    get_output_dir,
    get_selected_indices,
    VALID_SELECTION_VARIANTS,
)
from cstims.evaluation.plotting import plot_image_grid
from cstims.data_loader import (
    build_selected_image_records,
    load_natural_features_with_metadata,
)


def _build_records_from_indices(
    payload: dict,
    target_indices: list[int],
    result_dir: Path | None = None,
) -> list[dict]:
    """Build image records from a list of global indices.

    Args:
        payload: Selection payload dictionary
        target_indices: List of global indices to build records for
        result_dir: Path to result directory (for finding validated_images/)

    Returns:
        List of image record dicts
    """
    # Check for validated_images directory
    validated_images_dir = None
    if result_dir:
        candidate = result_dir / "validated_images"
        if candidate.exists():
            validated_images_dir = candidate
            print(f"  [INFO] Found validated_images directory: {validated_images_dir}")

    # Build a lookup from global_index to image record from all available sources
    idx_to_record: dict[int, dict] = {}

    # Add from selected_image_records (final selection)
    for record in payload.get("selected_image_records", []):
        global_idx = record.get("global_index")
        if global_idx is not None:
            idx_to_record[global_idx] = record.copy()

    # Add from greedy_image_records if available
    for record in payload.get("greedy_image_records", []):
        global_idx = record.get("global_index")
        if global_idx is not None and global_idx not in idx_to_record:
            idx_to_record[global_idx] = record.copy()

    # Add from filter_records (has shard_name and image_name)
    for fr in payload.get("filter_records", []):
        global_idx = fr.get("global_idx")
        if global_idx is not None and global_idx not in idx_to_record:
            idx_to_record[global_idx] = {
                "global_index": global_idx,
                "tar_path": None,
                "image_name": fr.get("image_name", ""),
                "shard_name": fr.get("shard_name", ""),
                "saved_path": fr.get("saved_path"),
            }

    # Build records list for target indices
    result = []
    n_from_validated = 0
    n_from_records = 0
    n_missing = 0

    for idx in target_indices:
        # First check validated_images directory (most reliable for old payloads)
        if validated_images_dir:
            validated_path = validated_images_dir / f"{idx}.png"
            if validated_path.exists():
                record = idx_to_record.get(idx, {"global_index": idx}).copy()
                record["saved_path"] = str(validated_path)
                result.append(record)
                n_from_validated += 1
                continue

        # Fall back to record lookup
        if idx in idx_to_record:
            result.append(idx_to_record[idx])
            n_from_records += 1
        else:
            # Create a placeholder record
            result.append({
                "global_index": idx,
                "tar_path": None,
                "image_name": None,
            })
            n_missing += 1

    print(f"  [INFO] Image sources: {n_from_validated} from validated_images/, "
          f"{n_from_records} from records, {n_missing} missing")

    return result


def _build_records_from_shards(
    payload: dict,
    target_indices: list[int],
) -> list[dict]:
    """Build image records by looking up global indices directly in tar shards.

    This bypasses any stored image records and goes directly to the data source,
    ensuring correct index-to-image mapping.

    Args:
        payload: Selection payload dictionary (for paths config)
        target_indices: List of global indices to build records for

    Returns:
        List of image record dicts with tar_path, image_name, global_index
    """
    config = payload.get("config", {})
    paths = config.get("paths", {})

    subset_root = Path(paths.get("subset_root", ""))
    model_list_csv = Path(paths.get("model_list_csv", ""))

    if not subset_root.exists():
        raise FileNotFoundError(f"subset_root not found: {subset_root}")
    if not model_list_csv.exists():
        raise FileNotFoundError(f"model_list_csv not found: {model_list_csv}")

    # Get model names from payload (we only need one to load shard metadata)
    model_names = payload.get("model_names", [])
    if not model_names:
        raise ValueError("No model_names in payload")

    # Use just one model to get shard structure (much faster)
    single_model = [model_names[0]]

    # Check for preprocessed_dir (fast path with pre-built metadata)
    preprocessed_dirs = paths.get("preprocessed_dirs", {})
    preprocessed_dir = preprocessed_dirs.get("raw")
    if preprocessed_dir:
        preprocessed_dir = Path(preprocessed_dir)
        if not preprocessed_dir.exists():
            preprocessed_dir = None

    print(f"  [INFO] Loading shard metadata from {subset_root}...")
    if preprocessed_dir:
        print(f"  [INFO] Using fast path with preprocessed_dir: {preprocessed_dir}")

    # Load shard slices (we don't need the features, just the metadata)
    _, shard_slices = load_natural_features_with_metadata(
        subset_root=subset_root,
        model_names=single_model,
        model_csv=model_list_csv,
        preprocessed_dir=preprocessed_dir,  # Use fast path if available
        max_images=None,  # Need full structure for index mapping
        n_jobs=1,  # Single-threaded for metadata only
    )

    print(f"  [INFO] Loaded {len(shard_slices)} shards, building image records...")

    # Build records using the data_loader utility
    records = build_selected_image_records(target_indices, shard_slices)

    print(f"  [INFO] Built {len(records)} image records from shards")

    return records


def _build_image_records_for_variant(
    payload: dict,
    selection_variant: str,
    result_dir: Path | None = None,
) -> list[dict]:
    """Build image records for a selection variant.

    For best_raw_combined, ALWAYS reconstructs from indices to ensure correctness
    (stored image records may be incorrect due to a bug in some selection runs).

    For other variants with variant-specific records, returns those directly.
    For old payloads, reconstructs records by matching indices to available metadata,
    including checking the validated_images directory.

    Args:
        payload: Selection payload dictionary
        selection_variant: Which selection variant ("final", "greedy", "best_raw_combined")
        result_dir: Path to result directory (for finding validated_images/)

    Returns:
        List of image record dicts with tar_path, image_name, global_index
    """
    # For best_raw_combined, ALWAYS rebuild from shards using reconstructed indices
    # (stored best_raw_combined_image_records may be incorrect due to selection bug)
    if selection_variant == "best_raw_combined":
        print("  [INFO] Rebuilding image records for 'best_raw_combined' directly from shards...")
        try:
            target_indices = get_selected_indices(payload, selection_variant)
        except ValueError as e:
            print(f"  [WARN] Could not get indices for variant: {e}")
            return []
        return _build_records_from_shards(payload, list(target_indices))

    # Map variant to payload key
    records_key_map = {
        "final": "selected_image_records",
        "greedy": "greedy_image_records",
    }
    records_key = records_key_map.get(selection_variant)
    if not records_key:
        raise ValueError(f"Unknown selection variant: {selection_variant}")

    # Try to get variant-specific records directly
    records = payload.get(records_key, [])
    if not records and selection_variant == "final":
        # For final variant, fall back to selected_image_records (same thing)
        records = payload.get("selected_image_records", [])

    # If we have records, check if we can augment them with validated_images paths
    if records and result_dir:
        validated_images_dir = result_dir / "validated_images"
        if validated_images_dir.exists():
            n_augmented = 0
            for record in records:
                global_idx = record.get("global_index")
                if global_idx is not None and not record.get("saved_path"):
                    validated_path = validated_images_dir / f"{global_idx}.png"
                    if validated_path.exists():
                        record["saved_path"] = str(validated_path)
                        n_augmented += 1
            if n_augmented:
                print(f"  [INFO] Augmented {n_augmented} records with validated_images paths")
            return records

    if records:
        return records

    # For non-final variants in old payloads, reconstruct from available data
    print(f"  [INFO] Reconstructing image records for '{selection_variant}' from available metadata...")

    # Get the target indices for this variant
    try:
        target_indices = get_selected_indices(payload, selection_variant)
    except ValueError as e:
        print(f"  [WARN] Could not get indices for variant: {e}")
        return []

    return _build_records_from_indices(payload, target_indices, result_dir)


def load_url_from_tar_metadata(tar_path: Path, image_name: str) -> str | None:
    """Load LAION URL from JSON metadata inside the given tar shard.

    Args:
        tar_path: Path to tar archive
        image_name: Name of image file in archive

    Returns:
        URL string or None if not found
    """
    json_name = f"{Path(image_name).stem}.json"
    if not tar_path.exists():
        return None

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            ef_meta = tar.extractfile(json_name)
            if ef_meta is None:
                return None
            meta_bytes = ef_meta.read()
    except Exception:
        return None

    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
        url = meta.get("url") or meta.get("URL")
        return url if isinstance(url, str) else None
    except Exception:
        return None


def load_image_from_tar(tar_path: Path, image_name: str) -> Image.Image | None:
    """Load image from tar archive as fallback.

    Args:
        tar_path: Path to tar archive
        image_name: Name of image file in archive

    Returns:
        PIL Image or None if failed
    """
    if not tar_path.exists():
        return None

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            ef_img = tar.extractfile(image_name)
            if ef_img is None:
                return None
            data = ef_img.read()
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def download_image(url: str, timeout: float) -> Image.Image | None:
    """Download image from URL.

    Args:
        url: Image URL
        timeout: HTTP timeout in seconds

    Returns:
        PIL Image or None if failed
    """
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    except (URLError, HTTPError, TimeoutError, Exception):
        return None


def center_crop_image(image: Image.Image) -> Image.Image:
    """Center crop image to a square.

    Args:
        image: Input PIL Image

    Returns:
        Center-cropped square image
    """
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) // 2
    top = (height - new_size) // 2
    right = left + new_size
    bottom = top + new_size
    return image.crop((left, top, right, bottom))


def download_selected_images(
    payload: dict,
    output_dir: Path,
    timeout: float = 10.0,
    overwrite: bool = False,
    selection_variant: str = "final",
    result_dir: Path | None = None,
) -> pd.DataFrame:
    """Download all selected images and create manifest.

    Args:
        payload: Selection payload dictionary
        output_dir: Directory for output images
        timeout: HTTP timeout in seconds
        overwrite: Whether to re-download existing images
        selection_variant: Which selection to download ("final", "greedy", "best_raw_combined")
        result_dir: Path to result directory (for finding validated_images/)

    Returns:
        DataFrame with image manifest
    """
    # Get image records for the requested variant (handles old payloads)
    records = _build_image_records_for_variant(payload, selection_variant, result_dir)
    if not records:
        print("No image records found in payload")
        return pd.DataFrame()

    # Build lookup for pre-saved images from filter_records
    filter_records = payload.get("filter_records", [])
    saved_paths_by_idx: dict[int, str] = {}
    for fr in filter_records:
        if fr.get("passed") and fr.get("saved_path"):
            saved_paths_by_idx[fr["global_idx"]] = fr["saved_path"]

    if saved_paths_by_idx:
        print(f"Found {len(saved_paths_by_idx)} pre-saved validated images from selection")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    images_for_pdf = []
    n_downloaded = 0
    n_fallback = 0
    n_cached = 0
    n_validated = 0  # From selection-time validation

    for idx, record in enumerate(tqdm(records, desc="Downloading images")):
        tar_path_str = record.get("tar_path")
        image_name = record.get("image_name")
        global_idx = record.get("global_index", idx)
        record_saved_path = record.get("saved_path")  # May be set by _build_image_records_for_variant

        dest_path = images_dir / f"image_{idx:04d}.png"

        # Check for existing file
        img = None
        source = "unknown"
        url = None
        orig_width, orig_height = None, None

        if dest_path.exists() and not overwrite:
            try:
                img = Image.open(dest_path).convert("RGB")
                source = "cached"
                orig_width, orig_height = img.size, img.size  # Already cropped
                n_cached += 1
            except Exception:
                pass

        # Check for pre-saved validated image from record (set by _build_image_records_for_variant)
        if img is None and record_saved_path:
            saved_path = Path(record_saved_path)
            if saved_path.exists():
                try:
                    img = Image.open(saved_path).convert("RGB")
                    source = "validated"
                    orig_width, orig_height = img.size, img.size  # Already center-cropped
                    n_validated += 1
                except Exception:
                    pass

        # Check for pre-saved validated image from filter_records lookup
        if img is None and global_idx in saved_paths_by_idx:
            saved_path = Path(saved_paths_by_idx[global_idx])
            if saved_path.exists():
                try:
                    img = Image.open(saved_path).convert("RGB")
                    source = "validated"
                    orig_width, orig_height = img.size, img.size  # Already center-cropped
                    n_validated += 1
                except Exception:
                    pass

        # Try to download from URL (only if we have tar metadata)
        if img is None and tar_path_str and image_name:
            tar_path = Path(tar_path_str)
            url = load_url_from_tar_metadata(tar_path, image_name)
            if url:
                img = download_image(url, timeout=timeout)
                if img is not None:
                    source = "downloaded"
                    orig_width, orig_height = img.size
                    n_downloaded += 1

            # Fallback to tar image
            if img is None:
                img = load_image_from_tar(tar_path, image_name)
                if img is not None:
                    source = "tar_fallback"
                    orig_width, orig_height = img.size
                    n_fallback += 1

        if img is None:
            print(f"Failed to obtain image {idx} ({image_name or f'global_idx={global_idx}'})")
            continue

        # Center crop to square (skip if already from validated/cached source)
        if source not in ("cached", "validated"):
            img = center_crop_image(img)

        # Save image
        if source not in ("cached",):
            try:
                img.save(dest_path)
            except Exception as e:
                print(f"Failed to save {dest_path}: {e}")
                continue

        # Record metadata (after crop)
        width, height = img.size
        manifest_rows.append({
            "idx": idx,
            "global_idx": global_idx,
            "tar_path": tar_path_str or "",
            "image_name": image_name or "",
            "url": url or "",
            "source": source,
            "orig_width": orig_width,
            "orig_height": orig_height,
            "crop_size": width,  # Square after center crop
            "local_path": str(dest_path),
        })

        # Add to PDF grid (already center-cropped)
        images_for_pdf.append(img)

    print(f"Images: {n_validated} from validation, {n_downloaded} downloaded, {n_fallback} from tars, {n_cached} cached")

    # Create manifest CSV
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = images_dir / "image_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved manifest to {manifest_path}")

    # Create PDF grid
    if images_for_pdf:
        pdf_path = images_dir / "selected_images_fullres.pdf"
        plot_image_grid(images_for_pdf, pdf_path)
        print(f"Saved image grid to {pdf_path}")

    return manifest_df


def main():
    parser = argparse.ArgumentParser(
        description="Download high-resolution selected images"
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Path to selection result directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <result-dir>/eval_pipeline/)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download existing images",
    )
    parser.add_argument(
        "--which-selection",
        type=str,
        choices=VALID_SELECTION_VARIANTS,
        default="final",
        help=(
            "Which selection variant to download images for: "
            "'final' (after refinement), 'greedy' (before refinement), "
            "or 'best_raw_combined' (best raw combined score). "
            "Default: final"
        ),
    )
    args = parser.parse_args()

    # Load payload
    print(f"Loading payload from {args.result_dir}")
    payload = load_selection_payload(args.result_dir)

    # Setup output directory
    output_dir = args.output_dir or get_output_dir(args.result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download images
    print(f"Downloading images for selection variant: {args.which_selection}")
    manifest_df = download_selected_images(
        payload=payload,
        output_dir=output_dir,
        timeout=args.timeout,
        overwrite=args.overwrite,
        selection_variant=args.which_selection,
        result_dir=args.result_dir,
    )

    print(f"\nDone! Downloaded {len(manifest_df)} images to {output_dir / 'images'}")


if __name__ == "__main__":
    main()
