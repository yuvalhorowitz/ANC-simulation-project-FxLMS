"""
Statistical comparison utilities for ML vs baseline evaluation.

These functions help determine if ML improvements are statistically significant.
"""

import numpy as np
from typing import List, Dict, Any
from scipy import stats


def is_significant_improvement(
    baseline_scores: List[float],
    ml_scores: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test if ML improvement over baseline is statistically significant.

    Uses paired t-test since we compare the same scenarios.

    Args:
        baseline_scores: List of baseline noise reduction values (dB)
        ml_scores: List of ML noise reduction values (dB), same length
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with:
            - significant: bool, True if p < alpha and improvement > 0
            - p_value: float, p-value from paired t-test
            - mean_improvement: float, mean NR improvement in dB
            - cohens_d: float, effect size (>0.3 is meaningful)
            - win_rate: float, fraction of scenarios where ML wins

    Example:
        >>> baseline = [10.0, 11.0, 9.5, 10.5]
        >>> ml = [11.5, 12.0, 10.5, 12.0]
        >>> result = is_significant_improvement(baseline, ml)
        >>> print(f"Significant: {result['significant']}")
    """
    baseline = np.array(baseline_scores)
    ml = np.array(ml_scores)

    if len(baseline) != len(ml):
        raise ValueError("baseline and ml must have same length")

    if len(baseline) < 3:
        return {
            'significant': False,
            'p_value': 1.0,
            'mean_improvement': np.mean(ml) - np.mean(baseline),
            'cohens_d': 0.0,
            'win_rate': 0.0,
            'error': 'Not enough samples for statistical test'
        }

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(ml, baseline)

    # Mean improvement
    improvement = np.mean(ml) - np.mean(baseline)

    # Cohen's d effect size
    diff = ml - baseline
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)

    # Win rate
    win_rate = np.mean(ml > baseline)

    return {
        'significant': p_value < alpha and improvement > 0,
        'p_value': float(p_value),
        'mean_improvement': float(improvement),
        'std_improvement': float(np.std(diff)),
        'cohens_d': float(cohens_d),
        'win_rate': float(win_rate),
        'n_samples': len(baseline)
    }


def summarize_comparison(
    baseline_results: List[Dict],
    ml_results: List[Dict]
) -> Dict[str, Any]:
    """
    Create a comprehensive comparison summary.

    Args:
        baseline_results: List of result dicts from baseline runs
        ml_results: List of result dicts from ML runs

    Returns:
        Dictionary with summary statistics for both methods
    """
    # Extract NR values
    baseline_nr = [r['noise_reduction_db'] for r in baseline_results]
    ml_nr = [r['noise_reduction_db'] for r in ml_results]

    # Basic stats
    summary = {
        'baseline': {
            'mean_nr_db': float(np.mean(baseline_nr)),
            'std_nr_db': float(np.std(baseline_nr)),
            'min_nr_db': float(np.min(baseline_nr)),
            'max_nr_db': float(np.max(baseline_nr)),
        },
        'ml': {
            'mean_nr_db': float(np.mean(ml_nr)),
            'std_nr_db': float(np.std(ml_nr)),
            'min_nr_db': float(np.min(ml_nr)),
            'max_nr_db': float(np.max(ml_nr)),
        },
    }

    # Statistical comparison
    summary['comparison'] = is_significant_improvement(baseline_nr, ml_nr)

    # Convergence comparison (if available)
    if 'convergence_time' in baseline_results[0]:
        baseline_conv = [r['convergence_time'] for r in baseline_results]
        ml_conv = [r['convergence_time'] for r in ml_results]

        summary['baseline']['mean_convergence'] = float(np.mean(baseline_conv))
        summary['ml']['mean_convergence'] = float(np.mean(ml_conv))
        summary['comparison']['convergence_speedup'] = float(
            np.mean(baseline_conv) / (np.mean(ml_conv) + 1)
        )

    # Stability comparison (if available)
    if 'stability_score' in baseline_results[0]:
        baseline_stable = [r['stability_score'] for r in baseline_results]
        ml_stable = [r['stability_score'] for r in ml_results]

        summary['baseline']['stability_rate'] = float(np.mean(baseline_stable))
        summary['ml']['stability_rate'] = float(np.mean(ml_stable))

    return summary


def format_comparison_report(summary: Dict[str, Any]) -> str:
    """
    Format comparison summary as human-readable text.

    Args:
        summary: Dictionary from summarize_comparison()

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "ML vs Baseline Comparison Report",
        "=" * 60,
        "",
        "Noise Reduction (dB):",
        f"  Baseline: {summary['baseline']['mean_nr_db']:.2f} ± {summary['baseline']['std_nr_db']:.2f}",
        f"  ML:       {summary['ml']['mean_nr_db']:.2f} ± {summary['ml']['std_nr_db']:.2f}",
        "",
    ]

    comp = summary['comparison']
    lines.extend([
        "Statistical Analysis:",
        f"  Mean improvement: {comp['mean_improvement']:.2f} dB",
        f"  Effect size (Cohen's d): {comp['cohens_d']:.3f}",
        f"  Win rate: {comp['win_rate']*100:.1f}%",
        f"  p-value: {comp['p_value']:.4f}",
        f"  Significant: {'Yes' if comp['significant'] else 'No'}",
        "",
    ])

    if 'convergence_speedup' in comp:
        lines.append(f"Convergence speedup: {comp['convergence_speedup']:.2f}x")

    if 'stability_rate' in summary['baseline']:
        lines.extend([
            f"Stability (baseline): {summary['baseline']['stability_rate']*100:.1f}%",
            f"Stability (ML): {summary['ml']['stability_rate']*100:.1f}%",
        ])

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
