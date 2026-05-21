#!/usr/bin/env python3
"""Generate summary report from all evaluation results.

Reads:
- summary.csv
- discriminability.csv
- diversity.csv
- greedy_scores.csv
- refinement.csv

Outputs:
- summary_report.md: Formatted markdown summary
- Prints formatted table to console
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_summary_dict(summary_path: Path) -> dict:
    """Load summary.csv as a dictionary."""
    if not summary_path.exists():
        return {}
    df = pd.read_csv(summary_path)
    return dict(zip(df["key"], df["value"]))


def get_auc_metrics(discrim_path: Path) -> dict:
    """Extract AUC metrics from discriminability.csv."""
    if not discrim_path.exists():
        return {}

    df = pd.read_csv(discrim_path)

    metrics = {}

    # Get AUC for each track and subset type
    for track in df["track"].unique():
        track_data = df[df["track"] == track]
        selected_auc = track_data[track_data["subset_type"] == "selected"]["auc"].iloc[0]
        random_auc = track_data[track_data["subset_type"] == "random"]["auc"].iloc[0]

        metrics[f"auc_selected_{track}"] = selected_auc
        metrics[f"auc_random_{track}"] = random_auc
        if random_auc > 0:
            metrics[f"auc_improvement_{track}"] = (random_auc - selected_auc) / random_auc * 100

    # Overall averages
    selected_aucs = [v for k, v in metrics.items() if k.startswith("auc_selected_")]
    random_aucs = [v for k, v in metrics.items() if k.startswith("auc_random_")]

    if selected_aucs:
        metrics["auc_selected_avg"] = sum(selected_aucs) / len(selected_aucs)
    if random_aucs:
        metrics["auc_random_avg"] = sum(random_aucs) / len(random_aucs)
    if selected_aucs and random_aucs:
        metrics["auc_improvement_avg"] = (metrics["auc_random_avg"] - metrics["auc_selected_avg"]) / metrics["auc_random_avg"] * 100

    return metrics


def get_diversity_metrics(diversity_path: Path) -> dict:
    """Extract diversity metrics from diversity.csv."""
    if not diversity_path.exists():
        return {}

    df = pd.read_csv(diversity_path)
    if df.empty:
        return {}

    row = df.iloc[0]
    return {
        "mean_pairwise_sim": row.get("mean_pairwise_sim"),
        "feature_entropy": row.get("feature_entropy"),
        "diversity_vs_random": row.get("diversity_vs_random"),
    }


def get_refinement_improvement(greedy_path: Path, refinement_path: Path) -> dict:
    """Compute refinement improvement from scores."""
    metrics = {}

    if greedy_path.exists():
        greedy_df = pd.read_csv(greedy_path)
        if "score_combined_raw" in greedy_df.columns:
            metrics["greedy_final_score"] = greedy_df["score_combined_raw"].iloc[-1]

    if refinement_path.exists():
        refine_df = pd.read_csv(refinement_path)
        if "score_combined_raw" in refine_df.columns:
            replaced = refine_df[refine_df["replaced"]]
            if not replaced.empty:
                metrics["refinement_final_score"] = replaced["score_combined_raw"].iloc[-1]

    if "greedy_final_score" in metrics and "refinement_final_score" in metrics:
        before = metrics["greedy_final_score"]
        after = metrics["refinement_final_score"]
        if before > 0:
            metrics["refinement_improvement_pct"] = (after - before) / before * 100

    return metrics


def get_statistics_metrics(stats_path: Path) -> dict:
    """Extract statistical test results from statistics.csv."""
    if not stats_path.exists():
        return {}

    df = pd.read_csv(stats_path)
    if df.empty:
        return {}

    # Get aggregate row
    aggregate = df[df["track"] == "AGGREGATE"]
    if aggregate.empty:
        return {}

    row = aggregate.iloc[0]
    return {
        "ttest_p": row.get("ttest_p"),
        "cohens_d": row.get("cohens_d"),
        "ttest_significant": row.get("ttest_significant"),
    }


def format_value(value, fmt=".3f"):
    """Format a value for display."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:{fmt}}"
    return str(value)


def generate_console_table(metrics: dict) -> str:
    """Generate formatted console table."""
    width = 60
    lines = []

    def add_section(title):
        lines.append("=" * width)
        lines.append(f"  {title}")
        lines.append("-" * width)

    def add_row(label, value, fmt=".3f"):
        val_str = format_value(value, fmt)
        lines.append(f"  {label:<35} {val_str:>20}")

    lines.append("")
    lines.append("=" * width)
    lines.append("           STIMULUS SELECTION SUMMARY")
    lines.append("=" * width)

    # Selection info
    add_section("Selection Configuration")
    add_row("Method", metrics.get("method"), "s")
    add_row("n_selected", metrics.get("n_selected"), "d")
    add_row("n_models", metrics.get("n_models"), "d")
    add_row("Metric", metrics.get("metric"), "s")

    # Discriminability
    if "auc_selected_avg" in metrics:
        add_section("Discriminability (AUC)")
        add_row("Selected (avg)", metrics.get("auc_selected_avg"))
        add_row("Random (avg)", metrics.get("auc_random_avg"))
        improvement = metrics.get("auc_improvement_avg")
        if improvement is not None:
            add_row("Improvement", f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%", "s")
        # Statistical significance
        if "ttest_p" in metrics:
            p_val = metrics["ttest_p"]
            sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            add_row("p-value (paired t-test)", f"{p_val:.2e} {sig_str}", "s")
        if "cohens_d" in metrics:
            add_row("Effect size (Cohen's d)", metrics.get("cohens_d"))

    # Score trajectory
    if "greedy_final_score" in metrics:
        add_section("Selection Scores")
        add_row("Greedy final score", metrics.get("greedy_final_score"))
        if "refinement_final_score" in metrics:
            add_row("Refinement final score", metrics.get("refinement_final_score"))
            improvement = metrics.get("refinement_improvement_pct")
            if improvement is not None:
                add_row("Refinement improvement", f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%", "s")
        add_row("n_replacements", metrics.get("n_replacements"), "d")

    # Diversity
    if "mean_pairwise_sim" in metrics:
        add_section("Diversity")
        add_row("Mean pairwise similarity", metrics.get("mean_pairwise_sim"))
        add_row("Feature entropy", metrics.get("feature_entropy"))
        if metrics.get("diversity_vs_random") is not None:
            diff = metrics["diversity_vs_random"]
            add_row("vs Random", f"{diff:+.4f}", "s")

    lines.append("=" * width)
    lines.append("")

    return "\n".join(lines)


def generate_markdown_report(metrics: dict) -> str:
    """Generate markdown report."""
    lines = []

    lines.append("# Stimulus Selection Summary Report")
    lines.append("")

    # Selection info
    lines.append("## Selection Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Method | {metrics.get('method', 'N/A')} |")
    lines.append(f"| n_selected | {metrics.get('n_selected', 'N/A')} |")
    lines.append(f"| n_models | {metrics.get('n_models', 'N/A')} |")
    lines.append(f"| Metric | {metrics.get('metric', 'N/A')} |")
    lines.append(f"| Correlation type | {metrics.get('corr_type', 'N/A')} |")
    lines.append("")

    # Discriminability
    if "auc_selected_avg" in metrics:
        lines.append("## Discriminability")
        lines.append("")
        lines.append("| Metric | Selected | Random | Improvement |")
        lines.append("|--------|----------|--------|-------------|")
        lines.append(f"| AUC (avg) | {format_value(metrics.get('auc_selected_avg'))} | {format_value(metrics.get('auc_random_avg'))} | {format_value(metrics.get('auc_improvement_avg'), '.1f')}% |")
        lines.append("")

        # Statistical significance
        if "ttest_p" in metrics:
            lines.append("### Statistical Significance")
            lines.append("")
            lines.append("| Test | Value |")
            lines.append("|------|-------|")
            p_val = metrics["ttest_p"]
            sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            lines.append(f"| p-value (paired t-test) | {p_val:.2e} {sig_str} |")
            if "cohens_d" in metrics:
                lines.append(f"| Effect size (Cohen's d) | {format_value(metrics.get('cohens_d'))} |")
            lines.append("")

    # Scores
    if "greedy_final_score" in metrics:
        lines.append("## Selection Scores")
        lines.append("")
        lines.append("| Stage | Score |")
        lines.append("|-------|-------|")
        lines.append(f"| Greedy final | {format_value(metrics.get('greedy_final_score'))} |")
        if "refinement_final_score" in metrics:
            lines.append(f"| Refinement final | {format_value(metrics.get('refinement_final_score'))} |")
            improvement = metrics.get("refinement_improvement_pct")
            if improvement is not None:
                lines.append(f"| Improvement | {'+' if improvement > 0 else ''}{improvement:.1f}% |")
        lines.append(f"| n_replacements | {metrics.get('n_replacements', 'N/A')} |")
        lines.append("")

    # Diversity
    if "mean_pairwise_sim" in metrics:
        lines.append("## Diversity")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Mean pairwise similarity | {format_value(metrics.get('mean_pairwise_sim'))} |")
        lines.append(f"| Feature entropy | {format_value(metrics.get('feature_entropy'))} |")
        if metrics.get("diversity_vs_random") is not None:
            lines.append(f"| vs Random | {metrics['diversity_vs_random']:+.4f} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary report from evaluation results"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing evaluation CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-dir>)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print to console",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    # Collect all metrics
    metrics = {}

    # Load summary.csv
    summary_path = input_dir / "summary.csv"
    if summary_path.exists():
        summary_dict = load_summary_dict(summary_path)
        metrics.update(summary_dict)

    # Load discriminability metrics
    discrim_path = input_dir / "discriminability.csv"
    auc_metrics = get_auc_metrics(discrim_path)
    metrics.update(auc_metrics)

    # Load diversity metrics
    diversity_path = input_dir / "diversity.csv"
    diversity_metrics = get_diversity_metrics(diversity_path)
    metrics.update(diversity_metrics)

    # Load refinement improvement
    greedy_path = input_dir / "greedy_scores.csv"
    refinement_path = input_dir / "refinement.csv"
    refine_metrics = get_refinement_improvement(greedy_path, refinement_path)
    metrics.update(refine_metrics)

    # Load statistical test results
    stats_path = input_dir / "statistics.csv"
    stats_metrics = get_statistics_metrics(stats_path)
    metrics.update(stats_metrics)

    # Generate and print console table
    if not args.quiet:
        console_table = generate_console_table(metrics)
        print(console_table)

    # Generate and save markdown report
    markdown_report = generate_markdown_report(metrics)
    report_path = output_dir / "summary_report.md"
    report_path.write_text(markdown_report)
    print(f"Saved summary report to {report_path}")


if __name__ == "__main__":
    main()
