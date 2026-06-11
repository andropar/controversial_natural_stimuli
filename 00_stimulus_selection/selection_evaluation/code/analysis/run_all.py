#!/usr/bin/env python3
"""Run the complete evaluation pipeline for stimulus selection results.

This script orchestrates all evaluation steps:
1. Download high-resolution images (optional)
2. Compute discriminability metrics
3. Extract selection scores
4. Compute diversity metrics
5. Generate discriminability plots
6. Generate score plots
7. Generate diversity plots
8. Generate summary report

Usage:
    python scripts/eval/run_all.py --result-dir <path_to_selection_results>

    # Skip image download (useful if already downloaded)
    python scripts/eval/run_all.py --result-dir <path> --skip-images

    # Skip plotting (just compute CSVs)
    python scripts/eval/run_all.py --result-dir <path> --skip-plots
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPTS_DIR = Path(__file__).resolve().parent


def run_script(script_name: str, args: list[str], description: str) -> bool:
    """Run a script with the given arguments.

    Args:
        script_name: Name of the script file
        args: List of command line arguments
        description: Human-readable description of what the script does

    Returns:
        True if successful, False otherwise
    """
    script_path = EVAL_SCRIPTS_DIR / script_name

    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print(f"{'=' * 60}")

    cmd = [sys.executable, str(script_path)] + args
    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=str(PROJECT_ROOT),
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Script not found: {script_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete evaluation pipeline for stimulus selection"
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Path to selection result directory containing selected_stimuli_data.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <result-dir>/eval_pipeline/)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image download step",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Force re-download of images even if they exist",
    )
    parser.add_argument(
        "--skip-discriminability",
        action="store_true",
        help="Skip discriminability computation (useful if already computed)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation (just compute CSVs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computations (default: cuda if available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-random-subsets",
        type=int,
        default=50,
        help="Number of random baseline subsets for discriminability (default: 50)",
    )
    parser.add_argument(
        "--which-selection",
        type=str,
        choices=["final", "greedy", "best_raw_combined"],
        default="final",
        help=(
            "Which selection variant to evaluate: "
            "'final' (after refinement), 'greedy' (before refinement), "
            "or 'best_raw_combined' (best raw combined score during refinement). "
            "Default: final"
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["iris", "raven"],
        default=None,
        help="Override payload paths with paths from conf/paths/{env}.yaml (e.g., 'iris' to run Raven results on Iris)",
    )
    args = parser.parse_args()

    # Validate result directory
    result_dir = args.result_dir.resolve()
    if not result_dir.exists():
        print(f"ERROR: Result directory does not exist: {result_dir}")
        sys.exit(1)

    payload_path = result_dir / "selected_stimuli_data.pkl"
    if not payload_path.exists():
        print(f"ERROR: Payload file not found: {payload_path}")
        sys.exit(1)

    # Setup output directory
    # For non-final variants, append the variant name to the output directory
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        base_output = result_dir / "eval_pipeline"
        if args.which_selection != "final":
            output_dir = base_output / args.which_selection
        else:
            output_dir = base_output
        output_dir = output_dir.resolve()

    print(f"\n{'#' * 60}")
    print(f"# Evaluation Pipeline")
    print(f"# Result directory: {result_dir}")
    print(f"# Output directory: {output_dir}")
    print(f"# Selection variant: {args.which_selection}")
    if args.env:
        print(f"# Path override: env={args.env}")
    print(f"{'#' * 60}")

    success = True

    # Step 1: Download images
    if not args.skip_images:
        download_args = [
            "--result-dir",
            str(result_dir),
            "--output-dir",
            str(output_dir),
            "--which-selection",
            args.which_selection,
        ]
        if args.overwrite_images:
            download_args.append("--overwrite")
        success = (
            run_script(
                "01_download_images.py",
                download_args,
                "Download high-resolution selected images",
            )
            and success
        )
    else:
        print("\n[SKIPPED] Image download")

    # Step 2: Compute discriminability
    if not args.skip_discriminability:
        discrim_args = [
                    "--result-dir",
                    str(result_dir),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--n-random-subsets",
                    str(args.n_random_subsets),
                    "--which-selection",
                    args.which_selection,
                ]
        if args.env:
            discrim_args.extend(["--env", args.env])
        success = (
            run_script(
                "02_compute_discriminability.py",
                discrim_args,
                "Compute discriminability metrics",
            )
            and success
        )
    else:
        print("\n[SKIPPED] Discriminability computation")

    # Step 3: Extract scores
    success = (
        run_script(
            "03_compute_scores.py",
            [
                "--result-dir",
                str(result_dir),
                "--output-dir",
                str(output_dir),
            ],
            "Extract selection scores and summary",
        )
        and success
    )

    # Step 4: Compute diversity metrics
    diversity_args = [
        "--result-dir",
        str(result_dir),
        "--output-dir",
        str(output_dir),
        "--which-selection",
        args.which_selection,
    ]
    if args.env:
        diversity_args.extend(["--env", args.env])
    success = (
        run_script(
            "04_compute_diversity.py",
            diversity_args,
            "Compute diversity metrics",
        )
        and success
    )

    # Step 5: Generate discriminability plots
    if not args.skip_plots and not args.skip_discriminability:
        success = (
            run_script(
                "plot_discriminability.py",
                [
                    "--input-dir",
                    str(output_dir),
                    "--output-dir",
                    str(output_dir / "plots"),
                ],
                "Generate discriminability plots",
            )
            and success
        )
    elif args.skip_plots:
        print("\n[SKIPPED] Discriminability plots")

    # Step 6: Generate score plots
    if not args.skip_plots:
        success = (
            run_script(
                "plot_scores.py",
                [
                    "--input-dir",
                    str(output_dir),
                    "--output-dir",
                    str(output_dir / "plots"),
                ],
                "Generate score trajectory plots",
            )
            and success
        )
    else:
        print("\n[SKIPPED] Score plots")

    # Step 7: Generate diversity plots
    if not args.skip_plots:
        success = (
            run_script(
                "plot_diversity.py",
                [
                    "--input-dir",
                    str(output_dir),
                    "--output-dir",
                    str(output_dir / "plots"),
                ],
                "Generate diversity plots",
            )
            and success
        )
    else:
        print("\n[SKIPPED] Diversity plots")

    # Step 8: Compute statistical tests
    if not args.skip_discriminability:
        success = (
            run_script(
                "06_compute_statistics.py",
                [
                    "--input-dir",
                    str(output_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                "Compute statistical significance tests",
            )
            and success
        )

    # Step 9: Compute ablation analyses
    success = (
        run_script(
            "08_compute_ablations.py",
            [
                "--input-dir",
                str(output_dir),
                "--output-dir",
                str(output_dir),
            ],
            "Compute ablation analyses",
        )
        and success
    )

    # Step 10: Analyze filtering (if filter_records exist in payload)
    filtering_args = [
        "--result-dir",
        str(result_dir),
        "--output-dir",
        str(output_dir),
    ]
    if args.env:
        filtering_args.extend(["--env", args.env])
    success = (
        run_script(
            "09_analyze_filtering.py",
            filtering_args,
            "Analyze image filtering results",
        )
        and success
    )

    # Step 11: Generate filtering plots
    if not args.skip_plots:
        # Check if filter_records.csv was generated
        filter_records_path = output_dir / "filter_records.csv"
        if filter_records_path.exists():
            success = (
                run_script(
                    "plot_filtering.py",
                    [
                        "--input-dir",
                        str(output_dir),
                        "--output-dir",
                        str(output_dir / "plots"),
                    ],
                    "Generate filtering analysis plots",
                )
                and success
            )
        else:
            print("\n[SKIPPED] Filtering plots (no filter_records.csv)")

    # Step 12: Generate summary report
    success = (
        run_script(
            "05_generate_summary.py",
            [
                "--input-dir",
                str(output_dir),
                "--output-dir",
                str(output_dir),
            ],
            "Generate summary report",
        )
        and success
    )

    # Final summary
    print(f"\n{'#' * 60}")
    if success:
        print("# Pipeline completed successfully!")
        print(f"#")
        print(f"# Outputs saved to: {output_dir}")
        print(f"#   - discriminability.csv : Discriminability metrics")
        print(f"#   - statistics.csv       : Statistical tests (p-values)")
        print(f"#   - diversity.csv        : Image diversity metrics")
        print(f"#   - ablations.csv        : Ablation analyses")
        print(f"#   - greedy_scores.csv    : Selection score trajectory")
        print(f"#   - refinement.csv       : Refinement history")
        print(f"#   - filter_records.csv   : Image filter evaluations")
        print(f"#   - filter_summary.json  : Filter statistics")
        print(f"#   - summary_report.md    : Formatted summary report")
        print(f"#   - plots/               : Visualizations")
    else:
        print("# Pipeline completed with errors!")
        print("# Check output above for details.")
    print(f"{'#' * 60}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
