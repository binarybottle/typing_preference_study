#!/usr/bin/env python3
"""
Dataset Results Comparison Script

This script compares keyboard preference analysis results between different datasets,
focusing on differences in hypothesis testing, key preferences, and transition preferences.

Usage:
    python compare_datasets.py --dataset1 results/dataset1/ --dataset2 results/dataset2/ --output comparison_report/
    
    # Compare specific analysis types only
    python compare_datasets.py --dataset1 results/dataset1/ --dataset2 results/dataset2/ --focus hypotheses
    python compare_datasets.py --dataset1 results/dataset1/ --dataset2 results/dataset2/ --focus keys
    python compare_datasets.py --dataset1 results/dataset1/ --dataset2 results/dataset2/ --focus transitions
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetComparator:
    """Compare keyboard preference analysis results between datasets."""
    
    def __init__(self, dataset1_path: str, dataset2_path: str, output_path: str):
        self.dataset1_path = Path(dataset1_path)
        self.dataset2_path = Path(dataset2_path)
        self.output_path = Path(output_path)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Store loaded data
        self.dataset1_name = self.dataset1_path.name
        self.dataset2_name = self.dataset2_path.name
        
    def compare_all(self, focus: Optional[str] = None) -> Dict[str, Any]:
        """Compare all aspects of the two datasets."""
        
        print(f"Comparing {self.dataset1_name} vs {self.dataset2_name}")
        print("=" * 60)
        
        comparison_results = {
            'dataset1_name': self.dataset1_name,
            'dataset2_name': self.dataset2_name,
            'summary': {}
        }
        
        # Compare focused hypotheses
        if focus is None or focus == 'hypotheses':
            print("Comparing focused hypothesis results...")
            hypothesis_comparison = self.compare_hypothesis_results()
            comparison_results['hypothesis_comparison'] = hypothesis_comparison
            self._save_hypothesis_comparison(hypothesis_comparison)
        
        # Compare key preferences  
        if focus is None or focus == 'keys':
            print("Comparing key preference results...")
            key_comparison = self.compare_key_preferences()
            comparison_results['key_comparison'] = key_comparison
            self._save_key_comparison(key_comparison)
        
        # Compare transition preferences
        if focus is None or focus == 'transitions':
            print("Comparing transition preference results...")
            transition_comparison = self.compare_transition_preferences()
            comparison_results['transition_comparison'] = transition_comparison
            self._save_transition_comparison(transition_comparison)
        
        # Generate overall summary
        if focus is None:
            print("Generating overall comparison summary...")
            comparison_results['summary'] = self._generate_overall_summary(comparison_results)
            self._save_overall_summary(comparison_results)
        
        print(f"Comparison complete! Results saved to {self.output_path}")
        return comparison_results
    
    def compare_hypothesis_results(self) -> Dict[str, Any]:
        """Compare focused hypothesis testing results between datasets."""
        
        # Load hypothesis data
        hyp1_data = self._load_hypothesis_data(self.dataset1_path)
        hyp2_data = self._load_hypothesis_data(self.dataset2_path)
        
        if not hyp1_data or not hyp2_data:
            return {'error': 'Could not load hypothesis data from one or both datasets'}
        
        comparison = {
            'dataset1_significant': len(hyp1_data.get('significant_results', [])),
            'dataset2_significant': len(hyp2_data.get('significant_results', [])),
            'dataset1_large_effects': len(hyp1_data.get('large_effects', [])),
            'dataset2_large_effects': len(hyp2_data.get('large_effects', [])),
            'significance_changes': [],
            'effect_size_changes': [],
            'direction_changes': [],
            'detailed_comparison': []
        }
        
        # Load detailed hypothesis results for direction analysis
        hyp1_detailed = self._load_detailed_hypothesis_data(self.dataset1_path)
        hyp2_detailed = self._load_detailed_hypothesis_data(self.dataset2_path)
        
        # Compare individual hypotheses
        all_hypotheses = set()
        if 'hypothesis_overview' in hyp1_data:
            all_hypotheses.update(hyp1_data['hypothesis_overview'].keys())
        if 'hypothesis_overview' in hyp2_data:
            all_hypotheses.update(hyp2_data['hypothesis_overview'].keys())
        
        for hyp_name in all_hypotheses:
            hyp1_info = hyp1_data.get('hypothesis_overview', {}).get(hyp_name, {})
            hyp2_info = hyp2_data.get('hypothesis_overview', {}).get(hyp_name, {})
            
            # Skip if missing from either dataset
            if not hyp1_info or not hyp2_info:
                continue
            
            # Compare significance
            sig1 = hyp1_info.get('significant', False)
            sig2 = hyp2_info.get('significant', False)
            
            if sig1 != sig2:
                comparison['significance_changes'].append({
                    'hypothesis': hyp_name,
                    'description': hyp1_info.get('description', ''),
                    'dataset1_significant': sig1,
                    'dataset2_significant': sig2,
                    'change_type': 'gained_significance' if sig2 and not sig1 else 'lost_significance'
                })
            
            # Compare effect sizes
            effect1 = hyp1_info.get('effect_size', 0)
            effect2 = hyp2_info.get('effect_size', 0)
            effect_diff = effect2 - effect1
            
            if abs(effect_diff) > 0.05:  # Meaningful effect size change
                comparison['effect_size_changes'].append({
                    'hypothesis': hyp_name,
                    'description': hyp1_info.get('description', ''),
                    'dataset1_effect': effect1,
                    'dataset2_effect': effect2,
                    'effect_difference': effect_diff,
                    'change_magnitude': 'large' if abs(effect_diff) > 0.15 else 'medium' if abs(effect_diff) > 0.10 else 'small'
                })
            
            # CHECK FOR DIRECTION CHANGES
            hyp1_detailed_info = hyp1_detailed.get('hypothesis_results', {}).get(hyp_name, {})
            hyp2_detailed_info = hyp2_detailed.get('hypothesis_results', {}).get(hyp_name, {})
            
            if ('statistics' in hyp1_detailed_info and 'statistics' in hyp2_detailed_info):
                stats1 = hyp1_detailed_info['statistics']
                stats2 = hyp2_detailed_info['statistics']
                
                prop1 = stats1.get('proportion_val1_wins', 0.5)
                prop2 = stats2.get('proportion_val1_wins', 0.5)
                values_compared = stats1.get('values_compared', ('val1', 'val2'))
                
                # Check if direction changed (preference flipped)
                direction1 = 'val1' if prop1 > 0.5 else 'val2' if prop1 < 0.5 else 'neutral'
                direction2 = 'val1' if prop2 > 0.5 else 'val2' if prop2 < 0.5 else 'neutral'
                
                if direction1 != direction2 and direction1 != 'neutral' and direction2 != 'neutral':
                    # Map val1/val2 to actual descriptive names
                    val1_name, val2_name = values_compared
                    
                    preferred1 = val1_name if direction1 == 'val1' else val2_name
                    preferred2 = val1_name if direction2 == 'val1' else val2_name
                    
                    # Calculate the correct proportions for the preferred options
                    preferred1_proportion = prop1 if direction1 == 'val1' else (1 - prop1)
                    preferred2_proportion = prop2 if direction2 == 'val1' else (1 - prop2)
                    
                    # Get effect sizes for both datasets
                    effect1 = stats1.get('effect_size', 0)
                    effect2 = stats2.get('effect_size', 0)
                    
                    comparison['direction_changes'].append({
                        'hypothesis': hyp_name,
                        'description': hyp1_info.get('description', ''),
                        'values_compared': values_compared,
                        'dataset1_preferred': preferred1,
                        'dataset2_preferred': preferred2,
                        'dataset1_preferred_proportion': preferred1_proportion,
                        'dataset2_preferred_proportion': preferred2_proportion,
                        'dataset1_val1_proportion': prop1,  # Keep original for reference
                        'dataset2_val1_proportion': prop2,  # Keep original for reference
                        'dataset1_effect_size': effect1,
                        'dataset2_effect_size': effect2,
                        'flip_magnitude': abs(prop1 - prop2),
                        'effect_size_change': effect2 - effect1
                    })
            
            # Detailed comparison for all hypotheses
            comparison['detailed_comparison'].append({
                'hypothesis': hyp_name,
                'description': hyp1_info.get('description', ''),
                'dataset1': {
                    'significant': sig1,
                    'effect_size': effect1,
                    'practical_significance': hyp1_info.get('practical_significance', 'unknown'),
                    'n_comparisons': hyp1_info.get('n_comparisons', 0)
                },
                'dataset2': {
                    'significant': sig2,
                    'effect_size': effect2,
                    'practical_significance': hyp2_info.get('practical_significance', 'unknown'),
                    'n_comparisons': hyp2_info.get('n_comparisons', 0)
                },
                'effect_difference': effect_diff
            })
        
        # Sort by magnitude/importance
        comparison['effect_size_changes'].sort(key=lambda x: abs(x['effect_difference']), reverse=True)
        comparison['direction_changes'].sort(key=lambda x: x['flip_magnitude'], reverse=True)
        comparison['detailed_comparison'].sort(key=lambda x: abs(x['effect_difference']), reverse=True)
        
        return comparison
    
    def compare_key_preferences(self) -> Dict[str, Any]:
        """Compare key preference results between datasets."""
        
        # Load key preference data
        key1_data = self._load_key_data(self.dataset1_path)
        key2_data = self._load_key_data(self.dataset2_path)
        
        if key1_data.empty or key2_data.empty:
            return {'error': 'Could not load key preference data from one or both datasets'}
        
        # Merge datasets on key
        merged = pd.merge(key1_data, key2_data, on='Key', suffixes=('_dataset1', '_dataset2'), how='outer')
        
        comparison = {
            'ranking_changes': [],
            'strength_changes': [],
            'significance_changes': [],
            'top_movers': [],
            'correlation_stats': {}
        }
        
        # Calculate ranking changes
        for _, row in merged.iterrows():
            key = row['Key']
            rank1 = row.get('Rank_dataset1', np.nan)
            rank2 = row.get('Rank_dataset2', np.nan)
            
            if pd.notna(rank1) and pd.notna(rank2):
                rank_change = rank1 - rank2  # Positive = moved up in dataset2
                
                if abs(rank_change) >= 2:  # Meaningful ranking change
                    comparison['ranking_changes'].append({
                        'key': key,
                        'dataset1_rank': int(rank1),
                        'dataset2_rank': int(rank2),
                        'rank_change': int(rank_change),
                        'direction': 'improved' if rank_change > 0 else 'declined'
                    })
        
        # Calculate strength changes
        for _, row in merged.iterrows():
            key = row['Key']
            strength1 = row.get('BT_Strength_dataset1', np.nan)
            strength2 = row.get('BT_Strength_dataset2', np.nan)
            
            if pd.notna(strength1) and pd.notna(strength2):
                strength_diff = strength2 - strength1
                
                if abs(strength_diff) > 0.1:  # Meaningful strength change
                    comparison['strength_changes'].append({
                        'key': key,
                        'dataset1_strength': float(strength1),
                        'dataset2_strength': float(strength2),
                        'strength_difference': float(strength_diff),
                        'direction': 'stronger' if strength_diff > 0 else 'weaker'
                    })
        
        # Sort changes by magnitude
        comparison['ranking_changes'].sort(key=lambda x: abs(x['rank_change']), reverse=True)
        comparison['strength_changes'].sort(key=lambda x: abs(x['strength_difference']), reverse=True)
        
        # Get top movers (combine ranking and strength changes)
        comparison['top_movers'] = comparison['ranking_changes'][:5]
        
        # Calculate correlation statistics
        valid_rows = merged.dropna(subset=['BT_Strength_dataset1', 'BT_Strength_dataset2'])
        if len(valid_rows) > 3:
            correlation_coef, correlation_p = stats.pearsonr(
                valid_rows['BT_Strength_dataset1'], 
                valid_rows['BT_Strength_dataset2']
            )
            comparison['correlation_stats'] = {
                'correlation_coefficient': float(correlation_coef),
                'correlation_p_value': float(correlation_p),
                'n_keys_compared': len(valid_rows)
            }
        
        return comparison
    
    def compare_transition_preferences(self) -> Dict[str, Any]:
        """Compare transition preference results between datasets."""
        
        # Load transition preference data
        trans1_data = self._load_transition_data(self.dataset1_path)
        trans2_data = self._load_transition_data(self.dataset2_path)
        
        if trans1_data.empty or trans2_data.empty:
            return {'error': 'Could not load transition preference data from one or both datasets'}
        
        # Add rankings to the data
        trans1_data['Rank_dataset1'] = range(1, len(trans1_data) + 1)
        trans2_data['Rank_dataset2'] = range(1, len(trans2_data) + 1)
        
        # Merge datasets on transition type
        merged = pd.merge(trans1_data, trans2_data, on='Transition_Type', suffixes=('_dataset1', '_dataset2'), how='outer')
        
        comparison = {
            'ranking_changes': [],
            'strength_changes': [],
            'top_movers': [],
            'correlation_stats': {}
        }
        
        # Calculate ranking changes
        for _, row in merged.iterrows():
            transition = row['Transition_Type']
            rank1 = row.get('Rank_dataset1', np.nan)
            rank2 = row.get('Rank_dataset2', np.nan)
            
            if pd.notna(rank1) and pd.notna(rank2):
                rank_change = rank1 - rank2  # Positive = moved up in dataset2
                
                if abs(rank_change) >= 3:  # Meaningful ranking change for transitions
                    comparison['ranking_changes'].append({
                        'transition': transition,
                        'dataset1_rank': int(rank1),
                        'dataset2_rank': int(rank2),
                        'rank_change': int(rank_change),
                        'direction': 'improved' if rank_change > 0 else 'declined'
                    })
        
        # Calculate strength changes
        for _, row in merged.iterrows():
            transition = row['Transition_Type']
            strength1 = row.get('BT_Strength_dataset1', np.nan)
            strength2 = row.get('BT_Strength_dataset2', np.nan)
            
            if pd.notna(strength1) and pd.notna(strength2):
                strength_diff = strength2 - strength1
                
                if abs(strength_diff) > 0.05:  # Meaningful strength change for transitions
                    comparison['strength_changes'].append({
                        'transition': transition,
                        'dataset1_strength': float(strength1),
                        'dataset2_strength': float(strength2),
                        'strength_difference': float(strength_diff),
                        'direction': 'stronger' if strength_diff > 0 else 'weaker'
                    })
        
        # Sort changes by magnitude
        comparison['ranking_changes'].sort(key=lambda x: abs(x['rank_change']), reverse=True)
        comparison['strength_changes'].sort(key=lambda x: abs(x['strength_difference']), reverse=True)
        
        # Get top movers
        comparison['top_movers'] = comparison['ranking_changes'][:10]
        
        # Calculate correlation statistics
        valid_rows = merged.dropna(subset=['BT_Strength_dataset1', 'BT_Strength_dataset2'])
        if len(valid_rows) > 3:
            correlation_coef, correlation_p = stats.pearsonr(
                valid_rows['BT_Strength_dataset1'], 
                valid_rows['BT_Strength_dataset2']
            )
            comparison['correlation_stats'] = {
                'correlation_coefficient': float(correlation_coef),
                'correlation_p_value': float(correlation_p),
                'n_transitions_compared': len(valid_rows)
            }
        
        return comparison
    
    def _load_detailed_hypothesis_data(self, dataset_path: Path) -> Dict[str, Any]:
        """Load detailed focused hypothesis results from a dataset."""
        
        # Load from JSON which has the detailed statistics
        json_path = dataset_path / 'focused_hypotheses' / 'focused_hypothesis_results.json'
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load detailed JSON from {json_path}: {e}")
        
        print(f"Warning: No detailed hypothesis results found for {dataset_path}")
        return {}
    
    def _load_hypothesis_data(self, dataset_path: Path) -> Dict[str, Any]:
        """Load focused hypothesis results from a dataset."""
        
        # Try to load from JSON first
        json_path = dataset_path / 'focused_hypotheses' / 'focused_hypothesis_results.json'
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    return data.get('summary', {})
            except Exception as e:
                print(f"Warning: Could not load JSON from {json_path}: {e}")
        
        # Fallback to parsing text report
        text_path = dataset_path / 'focused_hypotheses' / 'focused_hypothesis_report.txt'
        if text_path.exists():
            return self._parse_hypothesis_text_report(text_path)
        
        print(f"Warning: No hypothesis results found for {dataset_path}")
        return {}
    
    def _parse_hypothesis_text_report(self, text_path: Path) -> Dict[str, Any]:
        """Parse hypothesis results from text report."""
        
        with open(text_path, 'r') as f:
            content = f.read()
        
        # Extract basic statistics
        significant_match = re.search(r'Significant results.*?(\d+)/(\d+)', content)
        large_effects_match = re.search(r'Large practical effects.*?(\d+)', content)
        
        result = {
            'significant_results': [],
            'large_effects': [],
            'hypothesis_overview': {}
        }
        
        if significant_match:
            # This is a simplified parsing - full JSON is preferred
            print(f"Parsed {significant_match.group(1)} significant results from text report")
        
        return result
    
    def _load_key_data(self, dataset_path: Path) -> pd.DataFrame:
        """Load key preference statistics from a dataset."""
        
        csv_path = dataset_path / 'exploratory' / 'key_preference_statistics.csv'
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"Warning: Could not load key data from {csv_path}: {e}")
        
        print(f"Warning: No key preference data found for {dataset_path}")
        return pd.DataFrame()
    
    def _load_transition_data(self, dataset_path: Path) -> pd.DataFrame:
        """Load transition preference statistics from a dataset."""
        
        csv_path = dataset_path / 'exploratory' / 'transition_preference_statistics.csv'
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"Warning: Could not load transition data from {csv_path}: {e}")
        
        print(f"Warning: No transition preference data found for {dataset_path}")
        return pd.DataFrame()
    
    def _generate_overall_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of differences between datasets."""
        
        summary = {
            'major_differences': [],
            'stability_assessment': 'unknown',
            'key_findings': []
        }
        
        # Analyze hypothesis changes
        if 'hypothesis_comparison' in comparison_results:
            hyp_comp = comparison_results['hypothesis_comparison']
            
            direction_changes = len(hyp_comp.get('direction_changes', []))
            sig_changes = len(hyp_comp.get('significance_changes', []))
            large_effect_changes = len([x for x in hyp_comp.get('effect_size_changes', []) 
                                       if x.get('change_magnitude') == 'large'])
            
            if direction_changes > 0:
                summary['major_differences'].append(f"{direction_changes} hypotheses REVERSED direction (most critical)")
            if sig_changes > 0:
                summary['major_differences'].append(f"{sig_changes} hypotheses changed significance")
            if large_effect_changes > 0:
                summary['major_differences'].append(f"{large_effect_changes} hypotheses had large effect size changes")
        
        # Analyze key preference changes
        if 'key_comparison' in comparison_results:
            key_comp = comparison_results['key_comparison']
            
            major_rank_changes = len([x for x in key_comp.get('ranking_changes', []) 
                                    if abs(x.get('rank_change', 0)) >= 5])
            
            if major_rank_changes > 0:
                summary['major_differences'].append(f"{major_rank_changes} keys had major ranking changes (≥5 positions)")
            
            # Check correlation
            corr_stats = key_comp.get('correlation_stats', {})
            correlation = corr_stats.get('correlation_coefficient', 0)
            
            if correlation > 0.8:
                summary['stability_assessment'] = 'high'
            elif correlation > 0.6:
                summary['stability_assessment'] = 'moderate'
            elif correlation > 0.4:
                summary['stability_assessment'] = 'low'
            else:
                summary['stability_assessment'] = 'very_low'
        
        # Generate key findings
        if not summary['major_differences']:
            summary['key_findings'].append("Results are highly consistent between datasets")
        else:
            summary['key_findings'].append("Significant differences found between datasets")
            summary['key_findings'].extend(summary['major_differences'])
        
        return summary
    
    def _save_hypothesis_comparison(self, comparison: Dict[str, Any]) -> None:
        """Save hypothesis comparison results."""
        
        # Save detailed comparison as CSV
        if comparison.get('detailed_comparison'):
            rows = []
            for hyp in comparison['detailed_comparison']:
                rows.append({
                    'Hypothesis': hyp['hypothesis'],
                    'Description': hyp['description'],
                    'Dataset1_Significant': hyp['dataset1']['significant'],
                    'Dataset1_Effect_Size': round(hyp['dataset1']['effect_size'], 3),
                    'Dataset1_N_Comparisons': hyp['dataset1']['n_comparisons'],
                    'Dataset2_Significant': hyp['dataset2']['significant'],
                    'Dataset2_Effect_Size': round(hyp['dataset2']['effect_size'], 3),
                    'Dataset2_N_Comparisons': hyp['dataset2']['n_comparisons'],
                    'Effect_Size_Difference': round(hyp['effect_difference'], 3)
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(self.output_path / 'hypothesis_comparison_detailed.csv', index=False)
        
        # Save comprehensive comparison with all available statistics
        self._save_comprehensive_hypothesis_comparison()
        
        # Save direction changes as separate CSV
        if comparison.get('direction_changes'):
            direction_rows = []
            for change in comparison['direction_changes']:
                direction_rows.append({
                    'Hypothesis': change['hypothesis'],
                    'Description': change['description'],
                    'Values_Compared': f"{change['values_compared'][0]} vs {change['values_compared'][1]}",
                    'Dataset1_Preferred': change['dataset1_preferred'],
                    'Dataset1_Preferred_Proportion': round(change['dataset1_preferred_proportion'], 3),
                    'Dataset1_Effect_Size': round(change['dataset1_effect_size'], 3),
                    'Dataset2_Preferred': change['dataset2_preferred'],
                    'Dataset2_Preferred_Proportion': round(change['dataset2_preferred_proportion'], 3),
                    'Dataset2_Effect_Size': round(change['dataset2_effect_size'], 3),
                    'Flip_Magnitude': round(change['flip_magnitude'], 3),
                    'Effect_Size_Change': round(change['effect_size_change'], 3),
                    'Dataset1_Val1_Proportion': round(change['dataset1_val1_proportion'], 3),  # For reference
                    'Dataset2_Val1_Proportion': round(change['dataset2_val1_proportion'], 3)   # For reference
                })
            df = pd.DataFrame(direction_rows)
            df.to_csv(self.output_path / 'hypothesis_direction_changes.csv', index=False)
        
        # Save summary text report
        with open(self.output_path / 'hypothesis_comparison_summary.txt', 'w') as f:
            f.write(f"FOCUSED HYPOTHESIS COMPARISON: {self.dataset1_name} vs {self.dataset2_name}\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Dataset 1 ({self.dataset1_name}):\n")
            f.write(f"  Significant results: {comparison.get('dataset1_significant', 0)}\n")
            f.write(f"  Large effects: {comparison.get('dataset1_large_effects', 0)}\n\n")
            
            f.write(f"Dataset 2 ({self.dataset2_name}):\n")
            f.write(f"  Significant results: {comparison.get('dataset2_significant', 0)}\n")
            f.write(f"  Large effects: {comparison.get('dataset2_large_effects', 0)}\n\n")
            
            # Direction changes (MOST IMPORTANT)
            direction_changes = comparison.get('direction_changes', [])
            if direction_changes:
                f.write(f"DIRECTION CHANGES ({len(direction_changes)} total):\n")
                f.write("=" * 45 + "\n")
                f.write("These hypotheses show preference REVERSALS between datasets:\n\n")
                for change in direction_changes:
                    f.write(f"• {change['hypothesis']}\n")
                    f.write(f"   → {change['description']}\n")
                    f.write(f"   → Dataset 1: {change['dataset1_preferred']} preferred ({change['dataset1_preferred_proportion']:.1%}) - Effect: {change['dataset1_effect_size']:.3f}\n")
                    f.write(f"   → Dataset 2: {change['dataset2_preferred']} preferred ({change['dataset2_preferred_proportion']:.1%}) - Effect: {change['dataset2_effect_size']:.3f}\n")
                    f.write(f"   → Flip magnitude: {change['flip_magnitude']:.3f}\n")
                    f.write(f"   → Effect size change: {change['effect_size_change']:+.3f}\n\n")
            else:
                f.write("NO DIRECTION CHANGES: All hypothesis preferences consistent\n\n")
            
            # Significance changes
            sig_changes = comparison.get('significance_changes', [])
            if sig_changes:
                f.write(f"SIGNIFICANCE CHANGES ({len(sig_changes)} total):\n")
                f.write("-" * 40 + "\n")
                for change in sig_changes:
                    status = "GAINED" if change['change_type'] == 'gained_significance' else "LOST"
                    f.write(f"  {status}: {change['hypothesis']}\n")
                    f.write(f"    → {change['description']}\n")
                f.write("\n")
            
            # Effect size changes
            effect_changes = comparison.get('effect_size_changes', [])
            if effect_changes:
                f.write(f"LARGEST EFFECT SIZE CHANGES ({len(effect_changes)} total):\n")
                f.write("-" * 45 + "\n")
                for change in effect_changes[:5]:
                    direction = "↗" if change['effect_difference'] > 0 else "↘"
                    f.write(f"  {direction} {change['hypothesis']}\n")
                    f.write(f"    → {change['description']}\n")
                    f.write(f"    → Effect change: {change['effect_difference']:+.3f} ({change['change_magnitude']})\n")
                f.write("\n")
    
    def _save_comprehensive_hypothesis_comparison(self) -> None:
        """Save comprehensive hypothesis comparison with all available statistics."""
        
        # Load detailed hypothesis data from both datasets
        hyp1_detailed = self._load_detailed_hypothesis_data(self.dataset1_path)
        hyp2_detailed = self._load_detailed_hypothesis_data(self.dataset2_path)
        
        if not hyp1_detailed or not hyp2_detailed:
            print("Warning: Cannot create comprehensive comparison - missing detailed data")
            return
        
        hyp1_results = hyp1_detailed.get('hypothesis_results', {})
        hyp2_results = hyp2_detailed.get('hypothesis_results', {})
        
        # Get all hypotheses from both datasets
        all_hypotheses = set(hyp1_results.keys()) | set(hyp2_results.keys())
        
        comprehensive_rows = []
        
        for hyp_name in sorted(all_hypotheses):
            hyp1_data = hyp1_results.get(hyp_name, {})
            hyp2_data = hyp2_results.get(hyp_name, {})
            
            # Get statistics from both datasets
            stats1 = hyp1_data.get('statistics', {})
            stats2 = hyp2_data.get('statistics', {})
            
            # Helper function to safely round values
            def safe_round(value, decimals=3):
                if pd.isna(value) or value is None:
                    return None
                try:
                    return round(float(value), decimals)
                except (ValueError, TypeError):
                    return None
            
            # Helper function to safely convert boolean
            def safe_bool(value):
                if pd.isna(value) or value is None:
                    return None
                return bool(value)
            
            # Helper function to safely convert int
            def safe_int(value):
                if pd.isna(value) or value is None:
                    return None
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return None
            
            row = {
                'Hypothesis': hyp_name,
                'Description': hyp1_data.get('description', hyp2_data.get('description', '')),
                'Comparison_Values': f"{stats1.get('values_compared', stats2.get('values_compared', ['', '']))[0]} vs {stats1.get('values_compared', stats2.get('values_compared', ['', '']))[1]}",
                
                # Dataset 1 Statistics
                'Dataset1_Val1_Win_Proportion': safe_round(stats1.get('proportion_val1_wins')),
                'Dataset1_Effect_Size': safe_round(stats1.get('effect_size')),
                'Dataset1_P_Value': safe_round(stats1.get('p_value')),
                'Dataset1_P_Value_Corrected': safe_round(stats1.get('p_value_corrected')),
                'Dataset1_Significant_Corrected': safe_bool(stats1.get('significant_corrected')),
                'Dataset1_Practical_Significance': stats1.get('practical_significance'),
                'Dataset1_N_Comparisons': safe_int(stats1.get('n_comparisons')),
                
                # Dataset 2 Statistics  
                'Dataset2_Val1_Win_Proportion': safe_round(stats2.get('proportion_val1_wins')),
                'Dataset2_Effect_Size': safe_round(stats2.get('effect_size')),
                'Dataset2_P_Value': safe_round(stats2.get('p_value')),
                'Dataset2_P_Value_Corrected': safe_round(stats2.get('p_value_corrected')),
                'Dataset2_Significant_Corrected': safe_bool(stats2.get('significant_corrected')),
                'Dataset2_Practical_Significance': stats2.get('practical_significance'),
                'Dataset2_N_Comparisons': safe_int(stats2.get('n_comparisons')),
                
                # Differences
                'Effect_Size_Difference': safe_round(
                    (stats2.get('effect_size', 0) or 0) - (stats1.get('effect_size', 0) or 0)
                ) if stats1.get('effect_size') is not None and stats2.get('effect_size') is not None else None,
                
                'Proportion_Difference': safe_round(
                    (stats2.get('proportion_val1_wins', 0.5) or 0.5) - (stats1.get('proportion_val1_wins', 0.5) or 0.5)
                ) if stats1.get('proportion_val1_wins') is not None and stats2.get('proportion_val1_wins') is not None else None,
                
                'Significance_Change': (
                    'Gained' if not stats1.get('significant_corrected', False) and stats2.get('significant_corrected', False)
                    else 'Lost' if stats1.get('significant_corrected', False) and not stats2.get('significant_corrected', False)
                    else 'No Change'
                ),
                
                'Direction_Flipped': (
                    bool((stats1.get('proportion_val1_wins', 0.5) > 0.5) != (stats2.get('proportion_val1_wins', 0.5) > 0.5))
                    if stats1.get('proportion_val1_wins') is not None and stats2.get('proportion_val1_wins') is not None 
                    and stats1.get('proportion_val1_wins') != 0.5 and stats2.get('proportion_val1_wins') != 0.5
                    else False
                ),
                
                # Sample size comparison
                'N_Comparisons_Change': (
                    safe_int(stats2.get('n_comparisons', 0)) - safe_int(stats1.get('n_comparisons', 0))
                    if stats1.get('n_comparisons') is not None and stats2.get('n_comparisons') is not None
                    else None
                )
            }
            
            comprehensive_rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(comprehensive_rows)
        
        # Define the original hypothesis order from the analysis script
        hypothesis_order = [
            # 1. Finger load preferences (2 tests)
            'same_vs_different_column_reach',
            'same_vs_different_column_hurdle',
            
            # 2. Row preferences (5 tests)
            'home_vs_other_keys',
            'column1_upper_vs_lower_row',
            'column2_upper_vs_lower_row', 
            'column3_upper_vs_lower_row',
            'column4_upper_vs_lower_row',
            
            # 3. Row pair preferences (2 tests)
            'same_row_vs_reach',
            'reach_vs_hurdle',
            
            # 4. Column preferences (4 tests)
            'column_4_vs_3',
            'column_3_vs_2',
            'column_1_vs_other',
            'column_4_vs_5',  # Only when --include-column5
            
            # 5. Column pair preferences (4 tests)
            'separation_of_1_vs_2_columns_same_row',
            'separation_of_2_vs_3_columns_same_row',
            'separation_of_1_vs_2_columns_different_rows',
            'separation_of_2_vs_3_columns_different_rows',
            
            # 6. Column pair direction preferences (2 tests)
            'direction_same_row',
            'direction_different_rows'
        ]
        
        # Sort by original hypothesis order, then by any remaining hypotheses alphabetically
        df['sort_order'] = df['Hypothesis'].apply(
            lambda x: hypothesis_order.index(x) if x in hypothesis_order else len(hypothesis_order)
        )
        df = df.sort_values(['sort_order', 'Hypothesis']).drop('sort_order', axis=1)
        
        df.to_csv(self.output_path / 'hypothesis_comparison_comprehensive.csv', index=False)
        
        print(f"Saved comprehensive hypothesis comparison with {len(comprehensive_rows)} hypotheses")
    
    def _save_key_comparison(self, comparison: Dict[str, Any]) -> None:
        """Save key preference comparison results."""
        
        # Save ranking changes
        if comparison.get('ranking_changes'):
            df = pd.DataFrame(comparison['ranking_changes'])
            df.to_csv(self.output_path / 'key_ranking_changes.csv', index=False)
        
        # Save strength changes
        if comparison.get('strength_changes'):
            df = pd.DataFrame(comparison['strength_changes'])
            df.to_csv(self.output_path / 'key_strength_changes.csv', index=False)
        
        # Save summary text report
        with open(self.output_path / 'key_comparison_summary.txt', 'w') as f:
            f.write(f"KEY PREFERENCE COMPARISON: {self.dataset1_name} vs {self.dataset2_name}\n")
            f.write("=" * 70 + "\n\n")
            
            # Correlation stats
            corr_stats = comparison.get('correlation_stats', {})
            if corr_stats:
                corr = corr_stats['correlation_coefficient']
                f.write(f"OVERALL CORRELATION: {corr:.3f}\n")
                f.write(f"Keys compared: {corr_stats['n_keys_compared']}\n")
                f.write(f"Stability: {'High' if corr > 0.8 else 'Moderate' if corr > 0.6 else 'Low'}\n\n")
            
            # Top ranking changes
            ranking_changes = comparison.get('ranking_changes', [])
            if ranking_changes:
                f.write(f"BIGGEST RANKING CHANGES ({len(ranking_changes)} total):\n")
                f.write("-" * 35 + "\n")
                for change in ranking_changes[:10]:
                    direction = "↑" if change['direction'] == 'improved' else "↓"
                    f.write(f"  {direction} {change['key']}: #{change['dataset1_rank']} → #{change['dataset2_rank']} "
                           f"({change['rank_change']:+d} positions)\n")
                f.write("\n")
            
            # Top strength changes
            strength_changes = comparison.get('strength_changes', [])
            if strength_changes:
                f.write(f"BIGGEST STRENGTH CHANGES ({len(strength_changes)} total):\n")
                f.write("-" * 35 + "\n")
                for change in strength_changes[:10]:
                    direction = "↑" if change['direction'] == 'stronger' else "↓"
                    f.write(f"  {direction} {change['key']}: {change['dataset1_strength']:.3f} → "
                           f"{change['dataset2_strength']:.3f} ({change['strength_difference']:+.3f})\n")
                f.write("\n")
    
    def _save_transition_comparison(self, comparison: Dict[str, Any]) -> None:
        """Save transition preference comparison results."""
        
        # Save ranking changes
        if comparison.get('ranking_changes'):
            df = pd.DataFrame(comparison['ranking_changes'])
            df.to_csv(self.output_path / 'transition_ranking_changes.csv', index=False)
        
        # Save summary text report
        with open(self.output_path / 'transition_comparison_summary.txt', 'w') as f:
            f.write(f"TRANSITION PREFERENCE COMPARISON: {self.dataset1_name} vs {self.dataset2_name}\n")
            f.write("=" * 70 + "\n\n")
            
            # Correlation stats
            corr_stats = comparison.get('correlation_stats', {})
            if corr_stats:
                corr = corr_stats['correlation_coefficient']
                f.write(f"OVERALL CORRELATION: {corr:.3f}\n")
                f.write(f"Transitions compared: {corr_stats['n_transitions_compared']}\n\n")
            
            # Top ranking changes
            ranking_changes = comparison.get('ranking_changes', [])
            if ranking_changes:
                f.write(f"BIGGEST RANKING CHANGES ({len(ranking_changes)} total):\n")
                f.write("-" * 35 + "\n")
                for change in ranking_changes[:15]:
                    direction = "↑" if change['direction'] == 'improved' else "↓"
                    f.write(f"  {direction} {change['transition']}: #{change['dataset1_rank']} → "
                           f"#{change['dataset2_rank']} ({change['rank_change']:+d} positions)\n")
                f.write("\n")
    
    def _save_overall_summary(self, comparison_results: Dict[str, Any]) -> None:
        """Save overall comparison summary."""
        
        summary = comparison_results['summary']
        
        with open(self.output_path / 'overall_comparison_summary.txt', 'w') as f:
            f.write(f"DATASET COMPARISON SUMMARY: {self.dataset1_name} vs {self.dataset2_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"STABILITY ASSESSMENT: {summary.get('stability_assessment', 'unknown').upper()}\n\n")
            
            major_diffs = summary.get('major_differences', [])
            if major_diffs:
                f.write("MAJOR DIFFERENCES FOUND:\n")
                f.write("-" * 25 + "\n")
                for diff in major_diffs:
                    f.write(f"• {diff}\n")
                f.write("\n")
            
            key_findings = summary.get('key_findings', [])
            if key_findings:
                f.write("KEY FINDINGS:\n")
                f.write("-" * 15 + "\n")
                for finding in key_findings:
                    f.write(f"• {finding}\n")
                f.write("\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write("• hypothesis_comparison_comprehensive.csv - ALL hypothesis comparisons with full statistics\n")
            f.write("• hypothesis_direction_changes.csv - PREFERENCE REVERSALS (most important!)\n")
            f.write("• hypothesis_comparison_detailed.csv - Summary hypothesis comparisons\n")
            f.write("• key_ranking_changes.csv - Key preference ranking changes\n")
            f.write("• key_strength_changes.csv - Key preference strength changes\n")
            f.write("• transition_ranking_changes.csv - Transition ranking changes\n")
            f.write("• Individual summary files for each analysis type\n")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Compare keyboard preference analysis results between datasets")
    parser.add_argument('--dataset1', required=True, help='Path to first dataset results directory')
    parser.add_argument('--dataset2', required=True, help='Path to second dataset results directory')
    parser.add_argument('--output', required=True, help='Output directory for comparison results')
    parser.add_argument('--focus', choices=['hypotheses', 'keys', 'transitions'], 
                       help='Focus comparison on specific analysis type only')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset1):
        print(f"Error: Dataset 1 path not found: {args.dataset1}")
        return 1
    
    if not os.path.exists(args.dataset2):
        print(f"Error: Dataset 2 path not found: {args.dataset2}")
        return 1
    
    try:
        # Run comparison
        comparator = DatasetComparator(args.dataset1, args.dataset2, args.output)
        results = comparator.compare_all(focus=args.focus)
        
        print(f"\nComparison complete!")
        print(f"Results saved to: {args.output}")
        
        # Print quick summary
        if 'summary' in results:
            summary = results['summary']
            stability = summary.get('stability_assessment', 'unknown')
            major_diffs = len(summary.get('major_differences', []))
            
            print(f"\nQUICK SUMMARY:")
            print(f"==============")
            print(f"Stability: {stability.upper()}")
            print(f"Major differences: {major_diffs}")
            
            if summary.get('key_findings'):
                print(f"Key finding: {summary['key_findings'][0]}")
            
            # Show direction changes if any
            if 'hypothesis_comparison' in results and results['hypothesis_comparison'].get('direction_changes'):
                direction_changes = results['hypothesis_comparison']['direction_changes']
                print(f"\nCRITICAL: {len(direction_changes)} hypotheses reversed direction!")
                for change in direction_changes[:3]:  # Show top 3
                    print(f"   • {change['hypothesis']}: {change['dataset1_preferred']} → {change['dataset2_preferred']} "
                          f"(effects: {change['dataset1_effect_size']:.3f} → {change['dataset2_effect_size']:.3f})")
                if len(direction_changes) > 3:
                    print(f"   • ... and {len(direction_changes) - 3} more (see hypothesis_direction_changes.csv)")
        
        return 0
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        return 1

if __name__ == "__main__":
    exit(main())