"""
Bradley-Terry Results Comparison Script

This script compares two sets of Bradley-Terry analysis results from the keyboard
preference analysis, providing detailed statistical comparisons and interpretations.

Features:
- Cross-dataset validation of Bradley-Terry strengths
- Statistical significance testing for differences
- Effect size calculations and practical significance assessment
- Ranking correlation analysis
- Comprehensive reporting with interpretations
- Confidence interval overlap analysis

Usage:
    python compare_results.py --dataset1 results1/ --dataset2 results2/ --output comparison_results/
    
    # With custom dataset names
    python compare_results.py --dataset1 small_study/ --dataset2 large_study/ \
           --output comparison/ --dataset1-name "Pilot Study" --dataset2-name "Main Study"

Required files in each dataset folder:
    - key_preference_statistics.csv (from main Bradley-Terry analysis)
    - transition_preference_statistics.csv (from main Bradley-Terry analysis)
    - transition_delta_analysis.csv (optional, for delta analysis)
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BTResultsComparator:
    """Compare Bradley-Terry results from two different datasets."""
    
    def __init__(self):
        self.dataset1_name = "Dataset 1"
        self.dataset2_name = "Dataset 2"
    
    def compare_results(self, dataset1_path: str, dataset2_path: str, 
                       output_folder: str, dataset1_name: str = None, 
                       dataset2_name: str = None) -> Dict[str, Any]:
        """
        Compare Bradley-Terry results from two datasets.
        
        Args:
            dataset1_path: Path to first dataset results folder
            dataset2_path: Path to second dataset results folder  
            output_folder: Directory to save comparison results
            dataset1_name: Optional name for first dataset
            dataset2_name: Optional name for second dataset
            
        Returns:
            Dictionary containing all comparison results
        """
        
        logger.info("Starting Bradley-Terry results comparison...")
        
        # Set dataset names
        if dataset1_name:
            self.dataset1_name = dataset1_name
        if dataset2_name:
            self.dataset2_name = dataset2_name
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Load data from both datasets
        data1 = self._load_bt_results(dataset1_path, "Dataset 1")
        data2 = self._load_bt_results(dataset2_path, "Dataset 2")
        
        # Perform comparisons
        comparison_results = {}
        
        # 1. Key preference comparison
        if data1['key_stats'] is not None and data2['key_stats'] is not None:
            logger.info("Comparing key preferences...")
            comparison_results['key_comparison'] = self._compare_key_preferences(
                data1['key_stats'], data2['key_stats']
            )
        
        # 2. Transition preference comparison  
        if data1['transition_stats'] is not None and data2['transition_stats'] is not None:
            logger.info("Comparing transition preferences...")
            comparison_results['transition_comparison'] = self._compare_transition_preferences(
                data1['transition_stats'], data2['transition_stats']
            )
        
        # 3. Generate comprehensive report and visualizations
        logger.info("Generating comparison report and visualizations...")
        self._generate_comparison_report(comparison_results, output_folder)
        self._create_comparison_visualizations(comparison_results, output_folder)
        
        # Save full results
        try:
            results_path = os.path.join(output_folder, 'bt_comparison_results.json')
            with open(results_path, 'w') as f:
                json_results = self._convert_for_json(comparison_results)
                json.dump(json_results, f, indent=2, default=str)
            logger.info("Full comparison results saved to JSON")
        except Exception as e:
            logger.warning(f"Could not save JSON results: {e}")
        
        logger.info(f"Comparison complete! Results saved to {output_folder}")
        return comparison_results
    
    def _load_bt_results(self, results_path: str, dataset_name: str) -> Dict[str, Any]:
        """Load Bradley-Terry results from a results folder."""
        
        data = {
            'key_stats': None,
            'transition_stats': None,
            'transition_delta': None
        }
        
        # Try to load key preference statistics
        key_stats_path = os.path.join(results_path, 'key_preference_statistics.csv')
        if os.path.exists(key_stats_path):
            data['key_stats'] = pd.read_csv(key_stats_path)
            logger.info(f"Loaded key statistics for {dataset_name}: {len(data['key_stats'])} keys")
        else:
            logger.warning(f"Key statistics not found for {dataset_name}: {key_stats_path}")
        
        # Try to load transition preference statistics
        transition_stats_path = os.path.join(results_path, 'transition_preference_statistics.csv')
        if os.path.exists(transition_stats_path):
            data['transition_stats'] = pd.read_csv(transition_stats_path)
            logger.info(f"Loaded transition statistics for {dataset_name}: {len(data['transition_stats'])} transitions")
        else:
            logger.warning(f"Transition statistics not found for {dataset_name}: {transition_stats_path}")
        
        # Try to load delta transition analysis
        delta_path = os.path.join(results_path, 'transition_delta_analysis.csv')
        if os.path.exists(delta_path):
            data['transition_delta'] = pd.read_csv(delta_path)
            logger.info(f"Loaded delta transition analysis for {dataset_name}: {len(data['transition_delta'])} transitions")
        
        return data
    
    def _compare_key_preferences(self, keys1: pd.DataFrame, keys2: pd.DataFrame) -> Dict[str, Any]:
        """Compare key preferences between two datasets."""
        
        # Merge datasets on key
        merged = pd.merge(keys1, keys2, on='Key', suffixes=('_1', '_2'), how='inner')
        
        if len(merged) == 0:
            logger.warning("No common keys found between datasets")
            return {'error': 'No common keys found'}
        
        logger.info(f"Comparing {len(merged)} common keys")
        
        results = {}
        
        # 1. Bradley-Terry strength comparison
        strength_diff = merged['BT_Strength_1'] - merged['BT_Strength_2']
        results['strength_differences'] = {
            'mean_difference': float(np.mean(strength_diff)),
            'std_difference': float(np.std(strength_diff)),
            'max_absolute_diff': float(np.max(np.abs(strength_diff))),
            'differences': strength_diff.tolist(),
            'keys': merged['Key'].tolist()
        }
        
        # 2. Ranking correlation
        results['ranking_correlation'] = self._calculate_ranking_correlation(
            merged[['Key', 'Rank_1', 'Rank_2']]
        )
        
        # 3. Strength correlation
        corr_result = stats.pearsonr(merged['BT_Strength_1'], merged['BT_Strength_2'])
        results['strength_correlation'] = {
            'pearson_r': float(corr_result[0]),
            'p_value': float(corr_result[1])
        }
        
        # 4. Statistical significance of differences
        results['significance_tests'] = self._test_strength_differences(merged)
        
        # 5. Confidence interval overlap analysis
        results['ci_overlap'] = self._analyze_ci_overlap(merged)
        
        # 6. Effect size classification
        results['effect_sizes'] = self._classify_effect_sizes(strength_diff)
        
        # 7. Identify discrepant keys
        results['discrepancies'] = self._identify_key_discrepancies(merged)
        
        return results
    
    def _compare_transition_preferences(self, trans1: pd.DataFrame, trans2: pd.DataFrame) -> Dict[str, Any]:
        """Compare transition preferences between two datasets."""
        
        # Merge datasets on transition type
        merged = pd.merge(trans1, trans2, on='Transition_Type', suffixes=('_1', '_2'), how='inner')
        
        if len(merged) == 0:
            logger.warning("No common transitions found between datasets")
            return {'error': 'No common transitions found'}
        
        logger.info(f"Comparing {len(merged)} common transitions")
        
        results = {}
        
        # 1. Bradley-Terry strength comparison
        strength_diff = merged['BT_Strength_1'] - merged['BT_Strength_2']
        results['strength_differences'] = {
            'mean_difference': float(np.mean(strength_diff)),
            'std_difference': float(np.std(strength_diff)),
            'max_absolute_diff': float(np.max(np.abs(strength_diff))),
            'differences': strength_diff.tolist(),
            'transitions': merged['Transition_Type'].tolist()
        }
        
        # 2. Strength correlation
        corr_result = stats.pearsonr(merged['BT_Strength_1'], merged['BT_Strength_2'])
        results['strength_correlation'] = {
            'pearson_r': float(corr_result[0]),
            'p_value': float(corr_result[1])
        }
        
        # 3. Statistical significance of differences
        results['significance_tests'] = self._test_transition_differences(merged)
        
        # 4. Confidence interval overlap analysis
        results['ci_overlap'] = self._analyze_transition_ci_overlap(merged)
        
        # 5. Effect size classification
        results['effect_sizes'] = self._classify_effect_sizes(strength_diff)
        
        return results
    
    def _calculate_ranking_correlation(self, rank_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ranking correlations between datasets."""
        
        # Spearman correlation (rank-based)
        spearman_result = spearmanr(rank_data['Rank_1'], rank_data['Rank_2'])
        
        # Kendall's tau (rank-based, robust to outliers)
        kendall_result = kendalltau(rank_data['Rank_1'], rank_data['Rank_2'])
        
        return {
            'spearman_rho': float(spearman_result[0]),
            'spearman_p_value': float(spearman_result[1]),
            'kendall_tau': float(kendall_result[0]),
            'kendall_p_value': float(kendall_result[1])
        }
    
    def _test_strength_differences(self, merged: pd.DataFrame) -> Dict[str, Any]:
        """Test statistical significance of Bradley-Terry strength differences."""
        
        # Paired t-test for strength differences
        t_stat, t_p_value = stats.ttest_rel(merged['BT_Strength_1'], merged['BT_Strength_2'])
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            w_stat, w_p_value = stats.wilcoxon(merged['BT_Strength_1'], merged['BT_Strength_2'])
        except ValueError:
            # All differences are zero
            w_stat, w_p_value = np.nan, np.nan
        
        # Effect size (Cohen's d for paired samples)
        diff = merged['BT_Strength_1'] - merged['BT_Strength_2']
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        return {
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(t_p_value),
                'significant': t_p_value < 0.05
            },
            'wilcoxon_test': {
                'w_statistic': float(w_stat) if not np.isnan(w_stat) else None,
                'p_value': float(w_p_value) if not np.isnan(w_p_value) else None,
                'significant': w_p_value < 0.05 if not np.isnan(w_p_value) else False
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d)
            }
        }
    
    def _test_transition_differences(self, merged: pd.DataFrame) -> Dict[str, Any]:
        """Test statistical significance of transition strength differences."""
        
        # Same as key differences but for transitions
        return self._test_strength_differences(merged)
    
    def _analyze_ci_overlap(self, merged: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confidence interval overlap between datasets."""
        
        overlaps = []
        non_overlapping_keys = []
        
        for _, row in merged.iterrows():
            ci1_lower, ci1_upper = row['BT_CI_Lower_1'], row['BT_CI_Upper_1']
            ci2_lower, ci2_upper = row['BT_CI_Lower_2'], row['BT_CI_Upper_2']
            
            # Check if CIs are valid
            if pd.isna([ci1_lower, ci1_upper, ci2_lower, ci2_upper]).any():
                continue
            
            # Check for overlap
            overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)
            overlaps.append(overlap)
            
            if not overlap:
                non_overlapping_keys.append(row['Key'])
        
        return {
            'total_comparisons': len(overlaps),
            'overlapping': sum(overlaps),
            'non_overlapping': len(overlaps) - sum(overlaps),
            'overlap_percentage': float(sum(overlaps) / len(overlaps) * 100) if overlaps else 0,
            'non_overlapping_keys': non_overlapping_keys
        }
    
    def _analyze_transition_ci_overlap(self, merged: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transition CI overlap between datasets."""
        
        overlaps = []
        non_overlapping_transitions = []
        
        for _, row in merged.iterrows():
            ci1_lower, ci1_upper = row['BT_CI_Lower_1'], row['BT_CI_Upper_1']
            ci2_lower, ci2_upper = row['BT_CI_Lower_2'], row['BT_CI_Upper_2']
            
            # Check if CIs are valid
            if pd.isna([ci1_lower, ci1_upper, ci2_lower, ci2_upper]).any():
                continue
            
            # Check for overlap
            overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)
            overlaps.append(overlap)
            
            if not overlap:
                non_overlapping_transitions.append(row['Transition_Type'])
        
        return {
            'total_comparisons': len(overlaps),
            'overlapping': sum(overlaps),
            'non_overlapping': len(overlaps) - sum(overlaps),
            'overlap_percentage': float(sum(overlaps) / len(overlaps) * 100) if overlaps else 0,
            'non_overlapping_transitions': non_overlapping_transitions
        }
    
    def _classify_effect_sizes(self, differences: pd.Series) -> Dict[str, Any]:
        """Classify effect sizes of differences."""
        
        abs_diffs = np.abs(differences)
        
        # Cohen's conventions for effect sizes (adapted for BT differences)
        negligible = sum(abs_diffs < 0.1)
        small = sum((abs_diffs >= 0.1) & (abs_diffs < 0.3))
        medium = sum((abs_diffs >= 0.3) & (abs_diffs < 0.5))
        large = sum(abs_diffs >= 0.5)
        
        total = len(differences)
        
        return {
            'negligible': {'count': negligible, 'percentage': float(negligible / total * 100)},
            'small': {'count': small, 'percentage': float(small / total * 100)},
            'medium': {'count': medium, 'percentage': float(medium / total * 100)},
            'large': {'count': large, 'percentage': float(large / total * 100)},
            'mean_absolute_difference': float(np.mean(abs_diffs)),
            'median_absolute_difference': float(np.median(abs_diffs))
        }
    
    def _identify_key_discrepancies(self, merged: pd.DataFrame) -> Dict[str, Any]:
        """Identify keys with large discrepancies between datasets."""
        
        strength_diff = merged['BT_Strength_1'] - merged['BT_Strength_2']
        abs_diff = np.abs(strength_diff)
        
        # Define thresholds for discrepancies
        large_threshold = 0.5  # Large effect size
        medium_threshold = 0.3  # Medium effect size
        
        large_discrepancies = merged[abs_diff >= large_threshold].copy()
        medium_discrepancies = merged[(abs_diff >= medium_threshold) & (abs_diff < large_threshold)].copy()
        
        # Add difference columns
        if len(large_discrepancies) > 0:
            large_discrepancies['Strength_Difference'] = strength_diff[abs_diff >= large_threshold]
        if len(medium_discrepancies) > 0:
            medium_discrepancies['Strength_Difference'] = strength_diff[(abs_diff >= medium_threshold) & (abs_diff < large_threshold)]
        
        return {
            'large_discrepancies': {
                'count': len(large_discrepancies),
                'keys': large_discrepancies[['Key', 'BT_Strength_1', 'BT_Strength_2', 'Strength_Difference']].to_dict('records') if len(large_discrepancies) > 0 else []
            },
            'medium_discrepancies': {
                'count': len(medium_discrepancies),
                'keys': medium_discrepancies[['Key', 'BT_Strength_1', 'BT_Strength_2', 'Strength_Difference']].to_dict('records') if len(medium_discrepancies) > 0 else []
            }
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _convert_for_json(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _create_comparison_visualizations(self, results: Dict[str, Any], output_folder: str) -> None:
        """Create comparison visualizations."""
        
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
        
        # Key comparison visualizations
        if 'key_comparison' in results and 'error' not in results['key_comparison']:
            self._plot_key_comparisons(results['key_comparison'], output_folder)
        
        # Transition comparison visualizations
        if 'transition_comparison' in results and 'error' not in results['transition_comparison']:
            self._plot_transition_comparisons(results['transition_comparison'], output_folder)
    
    def _plot_key_comparisons(self, key_results: Dict[str, Any], output_folder: str) -> None:
        """Create key comparison plots."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Strength differences histogram
        differences = key_results['strength_differences']['differences']
        ax1.hist(differences, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax1.axvline(np.mean(differences), color='orange', linestyle='--', 
                   label=f'Mean: {np.mean(differences):.3f}')
        ax1.set_xlabel(f'Strength Difference ({self.dataset1_name} - {self.dataset2_name})')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Bradley-Terry Strength Differences')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Strength correlation scatter plot
        # We need to reload data to get individual strengths
        # For now, show effect size distribution
        effect_sizes = key_results['effect_sizes']
        categories = ['Negligible', 'Small', 'Medium', 'Large']
        counts = [effect_sizes[cat.lower()]['count'] for cat in categories]
        colors = ['green', 'yellow', 'orange', 'red']
        
        ax2.bar(categories, counts, color=colors, alpha=0.7)
        ax2.set_ylabel('Number of Keys')
        ax2.set_title('Effect Size Distribution of Differences')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        total = sum(counts)
        for i, (cat, count) in enumerate(zip(categories, counts)):
            if count > 0:
                pct = count / total * 100
                ax2.text(i, count + 0.5, f'{pct:.1f}%', ha='center', va='bottom')
        
        # 3. Confidence interval overlap
        ci_data = key_results['ci_overlap']
        overlap_labels = ['Overlapping', 'Non-overlapping']
        overlap_counts = [ci_data['overlapping'], ci_data['non_overlapping']]
        
        ax3.pie(overlap_counts, labels=overlap_labels, autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'])
        ax3.set_title('Confidence Interval Overlap')
        
        # 4. Statistical test results
        sig_tests = key_results['significance_tests']
        test_names = ['Paired t-test', 'Wilcoxon test']
        p_values = [sig_tests['paired_t_test']['p_value'], 
                   sig_tests['wilcoxon_test']['p_value'] or 1.0]
        significant = [p < 0.05 for p in p_values]
        
        colors = ['red' if sig else 'blue' for sig in significant]
        bars = ax4.bar(test_names, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
        ax4.axhline(-np.log10(0.05), color='red', linestyle='--', 
                   label='p=0.05 threshold')
        ax4.set_ylabel('-log10(p-value)')
        ax4.set_title('Statistical Significance Tests')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add significance indicators
        for bar, p_val, is_sig in zip(bars, p_values, significant):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'p={p_val:.3f}' + ('*' if is_sig else ''), 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'key_comparison_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_transition_comparisons(self, transition_results: Dict[str, Any], output_folder: str) -> None:
        """Create transition comparison plots."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Similar structure to key comparisons but for transitions
        differences = transition_results['strength_differences']['differences']
        
        # 1. Strength differences histogram
        ax1.hist(differences, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax1.axvline(np.mean(differences), color='orange', linestyle='--', 
                   label=f'Mean: {np.mean(differences):.3f}')
        ax1.set_xlabel(f'Strength Difference ({self.dataset1_name} - {self.dataset2_name})')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Transition Strength Differences')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Effect size distribution
        effect_sizes = transition_results['effect_sizes']
        categories = ['Negligible', 'Small', 'Medium', 'Large']
        counts = [effect_sizes[cat.lower()]['count'] for cat in categories]
        colors = ['green', 'yellow', 'orange', 'red']
        
        ax2.bar(categories, counts, color=colors, alpha=0.7)
        ax2.set_ylabel('Number of Transitions')
        ax2.set_title('Effect Size Distribution of Differences')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        total = sum(counts)
        for i, (cat, count) in enumerate(zip(categories, counts)):
            if count > 0:
                pct = count / total * 100
                ax2.text(i, count + 0.5, f'{pct:.1f}%', ha='center', va='bottom')
        
        # 3. Confidence interval overlap
        ci_data = transition_results['ci_overlap']
        overlap_labels = ['Overlapping', 'Non-overlapping']
        overlap_counts = [ci_data['overlapping'], ci_data['non_overlapping']]
        
        ax3.pie(overlap_counts, labels=overlap_labels, autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'])
        ax3.set_title('Confidence Interval Overlap')
        
        # 4. Correlation analysis
        corr_data = transition_results['strength_correlation']
        ax4.text(0.5, 0.7, f"Strength Correlation", ha='center', va='center', 
                transform=ax4.transAxes, fontsize=16, fontweight='bold')
        ax4.text(0.5, 0.5, f"Pearson r = {corr_data['pearson_r']:.3f}", 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.text(0.5, 0.3, f"p-value = {corr_data['p_value']:.3f}", 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'transition_comparison_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_report(self, results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive comparison report."""
        
        report_lines = [
            "bradley-terry results comparison report",
            "=" * 60,
            "",
            "executive summary",
            "================",
            "",
            f"This report compares Bradley-Terry model results between two datasets:",
            f"- {self.dataset1_name}",
            f"- {self.dataset2_name}",
            "",
            "The analysis includes statistical tests, effect size assessments, and",
            "confidence interval overlap analysis to determine the reliability and",
            "consistency of findings across datasets.",
            "",
        ]
        
        # Key comparison results
        if 'key_comparison' in results and 'error' not in results['key_comparison']:
            key_results = results['key_comparison']
            
            report_lines.extend([
                "key preference comparison",
                "========================",
                ""
            ])
            
            # Strength differences
            strength_diffs = key_results['strength_differences']
            report_lines.extend([
                "strength differences:",
                f"  mean difference: {strength_diffs['mean_difference']:.4f}",
                f"  standard deviation: {strength_diffs['std_difference']:.4f}",
                f"  maximum absolute difference: {strength_diffs['max_absolute_diff']:.4f}",
                ""
            ])
            
            # Correlations
            rank_corr = key_results['ranking_correlation']
            strength_corr = key_results['strength_correlation']
            report_lines.extend([
                "correlations:",
                f"  ranking correlation (Spearman): {rank_corr['spearman_rho']:.4f} (p={rank_corr['spearman_p_value']:.3f})",
                f"  ranking correlation (Kendall): {rank_corr['kendall_tau']:.4f} (p={rank_corr['kendall_p_value']:.3f})",
                f"  strength correlation (Pearson): {strength_corr['pearson_r']:.4f} (p={strength_corr['p_value']:.3f})",
                ""
            ])
            
            # Statistical significance
            sig_tests = key_results['significance_tests']
            report_lines.extend([
                "significance tests:",
                f"  paired t-test: t={sig_tests['paired_t_test']['t_statistic']:.3f}, p={sig_tests['paired_t_test']['p_value']:.3f}",
                f"  wilcoxon test: p={sig_tests['wilcoxon_test']['p_value']:.3f}" if sig_tests['wilcoxon_test']['p_value'] else "  wilcoxon test: not applicable",
                f"  effect size (Cohen's d): {sig_tests['effect_size']['cohens_d']:.3f} ({sig_tests['effect_size']['interpretation']})",
                ""
            ])
            
            # CI overlap
            ci_overlap = key_results['ci_overlap']
            report_lines.extend([
                "confidence interval overlap:",
                f"  overlapping CIs: {ci_overlap['overlapping']}/{ci_overlap['total_comparisons']} ({ci_overlap['overlap_percentage']:.1f}%)",
                f"  non-overlapping CIs: {ci_overlap['non_overlapping']} keys",
                ""
            ])
            
            # Effect sizes
            effect_sizes = key_results['effect_sizes']
            report_lines.extend([
                "effect size distribution:",
                f"  negligible (<0.1): {effect_sizes['negligible']['count']} keys ({effect_sizes['negligible']['percentage']:.1f}%)",
                f"  small (0.1-0.3): {effect_sizes['small']['count']} keys ({effect_sizes['small']['percentage']:.1f}%)",
                f"  medium (0.3-0.5): {effect_sizes['medium']['count']} keys ({effect_sizes['medium']['percentage']:.1f}%)",
                f"  large (≥0.5): {effect_sizes['large']['count']} keys ({effect_sizes['large']['percentage']:.1f}%)",
                ""
            ])
            
            # Discrepancies
            discrepancies = key_results['discrepancies']
            report_lines.extend([
                "key discrepancies:",
                f"  large discrepancies (≥0.5): {discrepancies['large_discrepancies']['count']} keys",
                f"  medium discrepancies (0.3-0.5): {discrepancies['medium_discrepancies']['count']} keys",
                ""
            ])
            
            if discrepancies['large_discrepancies']['count'] > 0:
                report_lines.append("keys with large discrepancies:")
                for key_info in discrepancies['large_discrepancies']['keys']:
                    report_lines.append(f"  {key_info['Key']}: {key_info['BT_Strength_1']:.3f} vs {key_info['BT_Strength_2']:.3f} (diff: {key_info['Strength_Difference']:.3f})")
                report_lines.append("")
        
        # Transition comparison results
        if 'transition_comparison' in results and 'error' not in results['transition_comparison']:
            transition_results = results['transition_comparison']
            
            report_lines.extend([
                "transition preference comparison",
                "===============================",
                ""
            ])
            
            # Similar structure to key comparison
            strength_diffs = transition_results['strength_differences']
            report_lines.extend([
                "strength differences:",
                f"  mean difference: {strength_diffs['mean_difference']:.4f}",
                f"  standard deviation: {strength_diffs['std_difference']:.4f}",
                f"  maximum absolute difference: {strength_diffs['max_absolute_diff']:.4f}",
                ""
            ])
            
            # Correlations
            strength_corr = transition_results['strength_correlation']
            report_lines.extend([
                "correlations:",
                f"  strength correlation (Pearson): {strength_corr['pearson_r']:.4f} (p={strength_corr['p_value']:.3f})",
                ""
            ])
            
            # Statistical significance
            sig_tests = transition_results['significance_tests']
            report_lines.extend([
                "significance tests:",
                f"  paired t-test: t={sig_tests['paired_t_test']['t_statistic']:.3f}, p={sig_tests['paired_t_test']['p_value']:.3f}",
                f"  effect size (Cohen's d): {sig_tests['effect_size']['cohens_d']:.3f} ({sig_tests['effect_size']['interpretation']})",
                ""
            ])
        
        # Interpretation and recommendations
        report_lines.extend([
            "interpretation and recommendations",
            "=================================",
            "",
            self._generate_interpretation(results),
            "",
            "files generated",
            "===============",
            "",
            "• bt_comparison_results.json - complete numerical results",
            "• key_comparison_analysis.png - key preference comparison plots",
            "• transition_comparison_analysis.png - transition comparison plots",
            "• bt_comparison_report.txt - this comprehensive report",
            ""
        ])
        
        # Save report
        report_path = os.path.join(output_folder, 'bt_comparison_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _generate_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate interpretation and recommendations based on comparison results."""
        
        interpretations = []
        
        # Key comparison interpretation
        if 'key_comparison' in results and 'error' not in results['key_comparison']:
            key_results = results['key_comparison']
            
            # Ranking correlation interpretation
            rank_corr = key_results['ranking_correlation']['spearman_rho']
            if rank_corr > 0.8:
                interpretations.append("KEY FINDINGS: Very strong ranking agreement between datasets (ρ > 0.8).")
            elif rank_corr > 0.6:
                interpretations.append("KEY FINDINGS: Strong ranking agreement between datasets (ρ > 0.6).")
            elif rank_corr > 0.4:
                interpretations.append("KEY FINDINGS: Moderate ranking agreement between datasets (ρ > 0.4).")
            else:
                interpretations.append("KEY FINDINGS: Weak ranking agreement between datasets (ρ ≤ 0.4).")
            
            # Effect size interpretation
            effect_sizes = key_results['effect_sizes']
            large_pct = effect_sizes['large']['percentage']
            if large_pct < 10:
                interpretations.append("Most key preferences show consistent results across datasets.")
            elif large_pct < 25:
                interpretations.append("Some key preferences show notable differences between datasets.")
            else:
                interpretations.append("Many key preferences show substantial differences between datasets.")
            
            # CI overlap interpretation
            ci_overlap = key_results['ci_overlap']['overlap_percentage']
            if ci_overlap > 80:
                interpretations.append("High confidence interval overlap suggests reliable estimates.")
            elif ci_overlap > 60:
                interpretations.append("Moderate confidence interval overlap indicates reasonable reliability.")
            else:
                interpretations.append("Low confidence interval overlap suggests high uncertainty.")
        
        # Recommendations
        interpretations.extend([
            "",
            "RECOMMENDATIONS:",
            "",
            "1. For publication: Use the dataset with higher sample size and tighter CIs.",
            "2. For robust findings: Focus on results that are consistent across both datasets.",
            "3. For discrepancies: Investigate potential causes (sample bias, methodology).",
            "4. For future studies: Consider pooling data if methodologies are compatible."
        ])
        
        return '\n'.join(interpretations)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Compare Bradley-Terry results from two datasets'
    )
    parser.add_argument('--dataset1', required=True,
                       help='Path to first dataset results folder')
    parser.add_argument('--dataset2', required=True,
                       help='Path to second dataset results folder')
    parser.add_argument('--output', required=True,
                       help='Output directory for comparison results')
    parser.add_argument('--dataset1-name', default='Dataset 1',
                       help='Name for first dataset (default: Dataset 1)')
    parser.add_argument('--dataset2-name', default='Dataset 2',
                       help='Name for second dataset (default: Dataset 2)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset1):
        logger.error(f"Dataset 1 folder not found: {args.dataset1}")
        return 1
    
    if not os.path.exists(args.dataset2):
        logger.error(f"Dataset 2 folder not found: {args.dataset2}")
        return 1
    
    try:
        # Run comparison
        comparator = BTResultsComparator()
        results = comparator.compare_results(
            args.dataset1, args.dataset2, args.output,
            args.dataset1_name, args.dataset2_name
        )
        
        logger.info("Comparison completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())