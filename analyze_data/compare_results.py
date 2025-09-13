#!/usr/bin/env python3
"""
Dataset Results Comparison Script

This script compares keyboard preference analysis results between different datasets,
focusing on differences in hypothesis testing, key preferences, transition preferences,
and MOO objectives analysis.

Usage:
    poetry run python3 compare_results.py --dataset1 output/nonProlific/analyze_objectives --dataset2 output/Prolific/analyze_objectives --output compare_results

    # Compare specific analysis types only
    python compare_results.py --dataset1 results/dataset1/ --dataset2 results/dataset2/ --focus hypotheses
    python compare_results.py --dataset1 results/dataset1/ --dataset2 results/dataset2/ --focus keys
    python compare_results.py --dataset1 results/dataset1/ --dataset2 results/dataset2/ --focus transitions
    python compare_results.py --dataset1 results/dataset1/ --dataset2 results/dataset2/ --focus moo
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
        
        # Compare MOO objectives
        if focus is None or focus == 'moo':
            print("Comparing MOO objectives analysis results...")
            moo_comparison = self.compare_moo_objectives()
            comparison_results['moo_comparison'] = moo_comparison
            self._save_moo_comparison(moo_comparison)
        
        # Generate overall summary
        if focus is None:
            print("Generating overall comparison summary...")
            comparison_results['summary'] = self._generate_overall_summary(comparison_results)
            self._save_overall_summary(comparison_results)
        
        print(f"Comparison complete! Results saved to {self.output_path}")
        return comparison_results
    
    def compare_moo_objectives(self) -> Dict[str, Any]:
        """Compare MOO objectives analysis results between datasets."""
        
        # Load MOO data
        moo1_data = self._load_moo_data(self.dataset1_path)
        moo2_data = self._load_moo_data(self.dataset2_path)
        
        if not moo1_data or not moo2_data:
            return {'error': 'Could not load MOO objectives data from one or both datasets'}
        
        comparison = {
            'same_letter_comparison': {},
            'pairwise_comparison': {},
            'row_separation_comparison': {},
            'column_separation_comparison': {},
            'column_4_vs_5_comparison': {},
            'overall_consistency': {}
        }
        
        # Compare same-letter preferences (Bradley-Terry rankings)
        comparison['same_letter_comparison'] = self._compare_same_letter_preferences(moo1_data, moo2_data)
        
        # Compare pairwise key preferences
        comparison['pairwise_comparison'] = self._compare_pairwise_preferences(moo1_data, moo2_data)
        
        # Compare row separation preferences
        comparison['row_separation_comparison'] = self._compare_row_separation(moo1_data, moo2_data)
        
        # Compare column separation preferences
        comparison['column_separation_comparison'] = self._compare_column_separation(moo1_data, moo2_data)
        
        # Compare column 4 vs 5 preferences
        comparison['column_4_vs_5_comparison'] = self._compare_column_4_vs_5(moo1_data, moo2_data)
        
        # Calculate overall consistency metrics
        comparison['overall_consistency'] = self._calculate_moo_consistency(comparison)
        
        return comparison
    
    def _load_moo_data(self, dataset_path: Path) -> Dict[str, Any]:
        """Load MOO objectives analysis results from a dataset."""
        
        moo_data = {
            'same_letter_bt': None,
            'pairwise_preferences': None,
            'row_separation': None,
            'column_separation': None,
            'column_4_vs_5': None,
            'report_data': None
        }
        
        # Load Bradley-Terry same-letter preferences
        same_letter_path = dataset_path / 'key_preferences_same_letter_pairs_BT.csv'
        if same_letter_path.exists():
            try:
                moo_data['same_letter_bt'] = pd.read_csv(same_letter_path)
                print(f"Loaded same-letter BT data: {len(moo_data['same_letter_bt'])} keys")
            except Exception as e:
                print(f"Warning: Could not load same-letter BT data from {same_letter_path}: {e}")
        else:
            print(f"Warning: Same-letter BT file not found: {same_letter_path}")
        
        # Load pairwise preferences
        pairwise_path = dataset_path / 'key_preferences_bigram_pairs.csv'
        if pairwise_path.exists():
            try:
                moo_data['pairwise_preferences'] = pd.read_csv(pairwise_path)
                print(f"Loaded pairwise preferences: {len(moo_data['pairwise_preferences'])} pairs")
            except Exception as e:
                print(f"Warning: Could not load pairwise data from {pairwise_path}: {e}")
        else:
            print(f"Warning: Pairwise preferences file not found: {pairwise_path}")
        
        # Parse report for other objectives
        report_path = dataset_path / 'moo_objectives_report.txt'
        if report_path.exists():
            try:
                moo_data['report_data'] = self._parse_moo_report(report_path)
                print(f"Parsed MOO report data from {report_path}")
            except Exception as e:
                print(f"Warning: Could not parse MOO report from {report_path}: {e}")
        else:
            print(f"Warning: MOO report file not found: {report_path}")
        
        return moo_data
    
    def _parse_moo_report(self, report_path: Path) -> Dict[str, Any]:
        """Parse MOO objectives report for numerical results."""
        
        with open(report_path, 'r') as f:
            content = f.read()
        
        report_data = {}
        
        # Extract row separation data
        row_match = re.search(r'ROW SEPARATION:.*?Result: Row separation preference: ([\d.]+)% favor smaller distances.*?Overall preference rate: ([\d.]+)% favor smaller distances.*?95% Confidence interval: \[([\d.]+)%, ([\d.]+)%\]', content, re.DOTALL)
        if row_match:
            report_data['row_separation'] = {
                'preference_rate': float(row_match.group(1)) / 100,
                'ci_lower': float(row_match.group(3)) / 100,
                'ci_upper': float(row_match.group(4)) / 100
            }
            
            # Extract breakdown by type
            breakdown_matches = re.findall(r'([\w\s]+): ([\d.]+)% \[([\d.]+)%, ([\d.]+)%\] \(n=(\d+)\)', content)
            if breakdown_matches:
                report_data['row_separation']['breakdown'] = {}
                for match in breakdown_matches:
                    comp_type = match[0].strip()
                    if 'apart' in comp_type.lower() or 'row' in comp_type.lower():
                        report_data['row_separation']['breakdown'][comp_type] = {
                            'preference_rate': float(match[1]) / 100,
                            'ci_lower': float(match[2]) / 100,
                            'ci_upper': float(match[3]) / 100,
                            'n_instances': int(match[4])
                        }
        
        # Extract column separation data
        col_match = re.search(r'COLUMN SEPARATION:.*?Result: Column separation preference: ([\d.]+)% favor smaller distances.*?Overall preference rate: ([\d.]+)% favor smaller distances.*?95% Confidence interval: \[([\d.]+)%, ([\d.]+)%\]', content, re.DOTALL)
        if col_match:
            report_data['column_separation'] = {
                'preference_rate': float(col_match.group(1)) / 100,
                'ci_lower': float(col_match.group(3)) / 100,
                'ci_upper': float(col_match.group(4)) / 100
            }
        
        # Extract column 4 vs 5 data
        col45_match = re.search(r'COLUMN 4 VS 5:.*?Result: (.*)', content, re.DOTALL)
        if col45_match:
            result_text = col45_match.group(1).strip()
            if 'Column 4 preference rate:' in result_text:
                rate_match = re.search(r'Column 4 preference rate: ([\d.]+)%', result_text)
                if rate_match:
                    report_data['column_4_vs_5'] = {
                        'preference_rate': float(rate_match.group(1)) / 100,
                        'available': True
                    }
            else:
                report_data['column_4_vs_5'] = {
                    'available': False,
                    'reason': result_text
                }
        
        # Extract instance counts
        instance_matches = re.findall(r'Instances analyzed: (\d+)', content)
        if len(instance_matches) >= 3:
            report_data['instance_counts'] = {
                'same_letter': int(instance_matches[0]) if len(instance_matches) > 0 else 0,
                'row_separation': int(instance_matches[1]) if len(instance_matches) > 1 else 0,
                'column_separation': int(instance_matches[2]) if len(instance_matches) > 2 else 0
            }
        
        return report_data
    
    def _compare_same_letter_preferences(self, moo1_data: Dict, moo2_data: Dict) -> Dict[str, Any]:
        """Compare same-letter Bradley-Terry preferences between datasets."""
        
        bt1_df = moo1_data.get('same_letter_bt')
        bt2_df = moo2_data.get('same_letter_bt')
        
        if bt1_df is None or bt2_df is None or bt1_df.empty or bt2_df.empty:
            return {'error': 'Same-letter BT data not available for both datasets'}
        
        # Merge on Key
        merged = pd.merge(bt1_df, bt2_df, on='Key', suffixes=('_dataset1', '_dataset2'), how='outer')
        
        comparison = {
            'ranking_changes': [],
            'strength_changes': [],
            'correlation_stats': {},
            'top_movers': []
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
    
    def _compare_pairwise_preferences(self, moo1_data: Dict, moo2_data: Dict) -> Dict[str, Any]:
        """Compare pairwise key preferences between datasets."""
        
        pair1_df = moo1_data.get('pairwise_preferences')
        pair2_df = moo2_data.get('pairwise_preferences')
        
        if pair1_df is None or pair2_df is None or pair1_df.empty or pair2_df.empty:
            return {'error': 'Pairwise preference data not available for both datasets'}
        
        # Create comparison identifier
        pair1_df['comparison'] = pair1_df['Key1'] + '_vs_' + pair1_df['Key2']
        pair2_df['comparison'] = pair2_df['Key1'] + '_vs_' + pair2_df['Key2']
        
        # Merge on comparison
        merged = pd.merge(pair1_df, pair2_df, on='comparison', suffixes=('_dataset1', '_dataset2'), how='inner')
        
        comparison = {
            'preference_changes': [],
            'direction_flips': [],
            'strength_changes': []
        }
        
        for _, row in merged.iterrows():
            comp = row['comparison']
            pref1 = row['Key1_Preference_Rate_dataset1']
            pref2 = row['Key1_Preference_Rate_dataset2']
            
            # Check for direction flip (preference reversal)
            if (pref1 > 0.5) != (pref2 > 0.5):
                comparison['direction_flips'].append({
                    'comparison': comp,
                    'dataset1_preference_rate': float(pref1),
                    'dataset2_preference_rate': float(pref2),
                    'flip_magnitude': abs(pref1 - pref2),
                    'dataset1_favored': row['Favored_Key_dataset1'],
                    'dataset2_favored': row['Favored_Key_dataset2']
                })
            
            # Check for significant preference changes
            pref_diff = abs(pref2 - pref1)
            if pref_diff > 0.1:  # 10+ percentage point change
                comparison['preference_changes'].append({
                    'comparison': comp,
                    'dataset1_preference_rate': float(pref1),
                    'dataset2_preference_rate': float(pref2),
                    'preference_difference': float(pref2 - pref1),
                    'magnitude': 'large' if pref_diff > 0.2 else 'medium'
                })
        
        # Sort by magnitude
        comparison['direction_flips'].sort(key=lambda x: x['flip_magnitude'], reverse=True)
        comparison['preference_changes'].sort(key=lambda x: abs(x['preference_difference']), reverse=True)
        
        return comparison
    
    def _compare_row_separation(self, moo1_data: Dict, moo2_data: Dict) -> Dict[str, Any]:
        """Compare row separation preferences between datasets."""
        
        row1_data = moo1_data.get('report_data', {}).get('row_separation')
        row2_data = moo2_data.get('report_data', {}).get('row_separation')
        
        if not row1_data or not row2_data:
            return {'error': 'Row separation data not available for both datasets'}
        
        comparison = {
            'overall_change': {
                'dataset1_preference_rate': row1_data['preference_rate'],
                'dataset2_preference_rate': row2_data['preference_rate'],
                'preference_difference': row2_data['preference_rate'] - row1_data['preference_rate'],
                'ci_overlap': self._check_ci_overlap(
                    row1_data['ci_lower'], row1_data['ci_upper'],
                    row2_data['ci_lower'], row2_data['ci_upper']
                )
            },
            'breakdown_changes': []
        }
        
        # Compare breakdown by type if available
        breakdown1 = row1_data.get('breakdown', {})
        breakdown2 = row2_data.get('breakdown', {})
        
        common_types = set(breakdown1.keys()) & set(breakdown2.keys())
        for comp_type in common_types:
            data1 = breakdown1[comp_type]
            data2 = breakdown2[comp_type]
            
            comparison['breakdown_changes'].append({
                'comparison_type': comp_type,
                'dataset1_preference_rate': data1['preference_rate'],
                'dataset2_preference_rate': data2['preference_rate'],
                'preference_difference': data2['preference_rate'] - data1['preference_rate'],
                'dataset1_n': data1['n_instances'],
                'dataset2_n': data2['n_instances']
            })
        
        return comparison
    
    def _compare_column_separation(self, moo1_data: Dict, moo2_data: Dict) -> Dict[str, Any]:
        """Compare column separation preferences between datasets."""
        
        col1_data = moo1_data.get('report_data', {}).get('column_separation')
        col2_data = moo2_data.get('report_data', {}).get('column_separation')
        
        if not col1_data or not col2_data:
            return {'error': 'Column separation data not available for both datasets'}
        
        return {
            'overall_change': {
                'dataset1_preference_rate': col1_data['preference_rate'],
                'dataset2_preference_rate': col2_data['preference_rate'],
                'preference_difference': col2_data['preference_rate'] - col1_data['preference_rate'],
                'ci_overlap': self._check_ci_overlap(
                    col1_data['ci_lower'], col1_data['ci_upper'],
                    col2_data['ci_lower'], col2_data['ci_upper']
                )
            }
        }
    
    def _compare_column_4_vs_5(self, moo1_data: Dict, moo2_data: Dict) -> Dict[str, Any]:
        """Compare column 4 vs 5 preferences between datasets."""
        
        col45_1 = moo1_data.get('report_data', {}).get('column_4_vs_5')
        col45_2 = moo2_data.get('report_data', {}).get('column_4_vs_5')
        
        if not col45_1 or not col45_2:
            return {'error': 'Column 4 vs 5 data not available for both datasets'}
        
        # Check if both datasets have the data available
        available1 = col45_1.get('available', False)
        available2 = col45_2.get('available', False)
        
        if not available1 or not available2:
            return {
                'status': 'insufficient_data',
                'dataset1_available': available1,
                'dataset2_available': available2,
                'dataset1_reason': col45_1.get('reason', 'Unknown'),
                'dataset2_reason': col45_2.get('reason', 'Unknown')
            }
        
        return {
            'preference_change': {
                'dataset1_preference_rate': col45_1['preference_rate'],
                'dataset2_preference_rate': col45_2['preference_rate'],
                'preference_difference': col45_2['preference_rate'] - col45_1['preference_rate']
            }
        }
    
    def _check_ci_overlap(self, ci1_lower: float, ci1_upper: float, 
                         ci2_lower: float, ci2_upper: float) -> bool:
        """Check if two confidence intervals overlap."""
        return not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)
    
    def _calculate_moo_consistency(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall consistency metrics for MOO objectives."""
        
        consistency = {
            'overall_rating': 'unknown',
            'issues_found': [],
            'consistent_objectives': [],
            'problematic_objectives': []
        }
        
        # Check same-letter preferences
        same_letter = comparison.get('same_letter_comparison', {})
        if 'correlation_stats' in same_letter:
            corr = same_letter['correlation_stats'].get('correlation_coefficient', 0)
            if corr > 0.8:
                consistency['consistent_objectives'].append('same_letter_preferences')
            elif corr < 0.6:
                consistency['problematic_objectives'].append('same_letter_preferences')
                consistency['issues_found'].append(f"Same-letter BT correlation: {corr:.3f}")
        
        # Check pairwise preferences for direction flips
        pairwise = comparison.get('pairwise_comparison', {})
        if 'direction_flips' in pairwise:
            direction_flips = len(pairwise['direction_flips'])
            if direction_flips > 0:
                consistency['problematic_objectives'].append('pairwise_preferences')
                consistency['issues_found'].append(f"{direction_flips} pairwise preference reversals")
            else:
                consistency['consistent_objectives'].append('pairwise_preferences')
        
        # Check row separation
        row_sep = comparison.get('row_separation_comparison', {})
        if 'overall_change' in row_sep:
            change = row_sep['overall_change']
            if abs(change.get('preference_difference', 0)) > 0.1:
                consistency['issues_found'].append(f"Row separation changed by {change['preference_difference']:.1%}")
            if change.get('ci_overlap', True):
                consistency['consistent_objectives'].append('row_separation')
            else:
                consistency['problematic_objectives'].append('row_separation')
        
        # Overall rating
        if len(consistency['problematic_objectives']) == 0:
            consistency['overall_rating'] = 'high'
        elif len(consistency['problematic_objectives']) <= 2:
            consistency['overall_rating'] = 'moderate'
        else:
            consistency['overall_rating'] = 'low'
        
        return consistency
    
    def _save_moo_comparison(self, comparison: Dict[str, Any]) -> None:
        """Save MOO objectives comparison results."""
        
        # Save same-letter BT comparison
        same_letter = comparison.get('same_letter_comparison', {})
        if 'ranking_changes' in same_letter and same_letter['ranking_changes']:
            df = pd.DataFrame(same_letter['ranking_changes'])
            df.to_csv(self.output_path / 'moo_same_letter_ranking_changes.csv', index=False)
        
        if 'strength_changes' in same_letter and same_letter['strength_changes']:
            df = pd.DataFrame(same_letter['strength_changes'])
            df.to_csv(self.output_path / 'moo_same_letter_strength_changes.csv', index=False)
        
        # Save pairwise comparison
        pairwise = comparison.get('pairwise_comparison', {})
        if 'direction_flips' in pairwise and pairwise['direction_flips']:
            df = pd.DataFrame(pairwise['direction_flips'])
            df.to_csv(self.output_path / 'moo_pairwise_direction_flips.csv', index=False)
        
        if 'preference_changes' in pairwise and pairwise['preference_changes']:
            df = pd.DataFrame(pairwise['preference_changes'])
            df.to_csv(self.output_path / 'moo_pairwise_preference_changes.csv', index=False)
        
        # Save comprehensive MOO comparison report
        self._save_moo_comprehensive_report(comparison)
    
    def _save_moo_comprehensive_report(self, comparison: Dict[str, Any]) -> None:
        """Save comprehensive MOO comparison text report."""
        
        with open(self.output_path / 'moo_objectives_comparison_report.txt', 'w') as f:
            f.write(f"MOO OBJECTIVES COMPARISON: {self.dataset1_name} vs {self.dataset2_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall consistency assessment
            consistency = comparison.get('overall_consistency', {})
            f.write(f"OVERALL CONSISTENCY: {consistency.get('overall_rating', 'unknown').upper()}\n\n")
            
            if consistency.get('issues_found'):
                f.write("ISSUES IDENTIFIED:\n")
                f.write("-" * 20 + "\n")
                for issue in consistency['issues_found']:
                    f.write(f"• {issue}\n")
                f.write("\n")
            
            # Same-letter preferences (Bradley-Terry)
            f.write("1. SAME-LETTER KEY PREFERENCES (BRADLEY-TERRY)\n")
            f.write("-" * 50 + "\n")
            same_letter = comparison.get('same_letter_comparison', {})
            
            if 'correlation_stats' in same_letter:
                corr_stats = same_letter['correlation_stats']
                corr = corr_stats['correlation_coefficient']
                f.write(f"Correlation: {corr:.3f} (n={corr_stats['n_keys_compared']} keys)\n")
                f.write(f"Stability: {'High' if corr > 0.8 else 'Moderate' if corr > 0.6 else 'Low'}\n\n")
            
            if same_letter.get('ranking_changes'):
                f.write(f"Ranking changes ({len(same_letter['ranking_changes'])} total):\n")
                for change in same_letter['ranking_changes'][:10]:
                    direction = "↑" if change['direction'] == 'improved' else "↓"
                    f.write(f"  {direction} {change['key']}: #{change['dataset1_rank']} → #{change['dataset2_rank']} "
                           f"({change['rank_change']:+d} positions)\n")
                f.write("\n")
            
            # Pairwise preferences
            f.write("2. PAIRWISE KEY PREFERENCES\n")
            f.write("-" * 30 + "\n")
            pairwise = comparison.get('pairwise_comparison', {})
            
            if pairwise.get('direction_flips'):
                f.write(f"DIRECTION REVERSALS ({len(pairwise['direction_flips'])} total):\n")
                for flip in pairwise['direction_flips']:
                    f.write(f"  • {flip['comparison']}\n")
                    f.write(f"    Dataset 1: {flip['dataset1_favored']} preferred ({flip['dataset1_preference_rate']:.1%})\n")
                    f.write(f"    Dataset 2: {flip['dataset2_favored']} preferred ({flip['dataset2_preference_rate']:.1%})\n")
                    f.write(f"    Flip magnitude: {flip['flip_magnitude']:.3f}\n")
                f.write("\n")
            else:
                f.write("No direction reversals found.\n\n")
            
            if pairwise.get('preference_changes'):
                f.write(f"Large preference changes ({len(pairwise['preference_changes'])} total):\n")
                for change in pairwise['preference_changes'][:5]:
                    f.write(f"  • {change['comparison']}: {change['dataset1_preference_rate']:.1%} → "
                           f"{change['dataset2_preference_rate']:.1%} ({change['preference_difference']:+.1%})\n")
                f.write("\n")
            
            # Row separation
            f.write("3. ROW SEPARATION PREFERENCES\n")
            f.write("-" * 35 + "\n")
            row_sep = comparison.get('row_separation_comparison', {})
            
            if 'overall_change' in row_sep:
                change = row_sep['overall_change']
                f.write(f"Overall preference for smaller row distances:\n")
                f.write(f"  Dataset 1: {change['dataset1_preference_rate']:.1%}\n")
                f.write(f"  Dataset 2: {change['dataset2_preference_rate']:.1%}\n")
                f.write(f"  Change: {change['preference_difference']:+.1%}\n")
                f.write(f"  CI overlap: {'Yes' if change['ci_overlap'] else 'No'}\n\n")
            
            if row_sep.get('breakdown_changes'):
                f.write("Breakdown by comparison type:\n")
                for breakdown in row_sep['breakdown_changes']:
                    f.write(f"  • {breakdown['comparison_type']}: "
                           f"{breakdown['dataset1_preference_rate']:.1%} → "
                           f"{breakdown['dataset2_preference_rate']:.1%} "
                           f"({breakdown['preference_difference']:+.1%})\n")
                f.write("\n")
            
            # Column separation
            f.write("4. COLUMN SEPARATION PREFERENCES\n")
            f.write("-" * 38 + "\n")
            col_sep = comparison.get('column_separation_comparison', {})
            
            if 'overall_change' in col_sep:
                change = col_sep['overall_change']
                f.write(f"Overall preference for smaller column distances:\n")
                f.write(f"  Dataset 1: {change['dataset1_preference_rate']:.1%}\n")
                f.write(f"  Dataset 2: {change['dataset2_preference_rate']:.1%}\n")
                f.write(f"  Change: {change['preference_difference']:+.1%}\n")
                f.write(f"  CI overlap: {'Yes' if change['ci_overlap'] else 'No'}\n\n")
            
            # Column 4 vs 5
            f.write("5. COLUMN 4 VS 5 PREFERENCES\n")
            f.write("-" * 32 + "\n")
            col45 = comparison.get('column_4_vs_5_comparison', {})
            
            if col45.get('status') == 'insufficient_data':
                f.write("Insufficient data in one or both datasets.\n")
                f.write(f"  Dataset 1 available: {col45.get('dataset1_available', False)}\n")
                f.write(f"  Dataset 2 available: {col45.get('dataset2_available', False)}\n")
            elif 'preference_change' in col45:
                change = col45['preference_change']
                f.write(f"Column 4 preference rate:\n")
                f.write(f"  Dataset 1: {change['dataset1_preference_rate']:.1%}\n")
                f.write(f"  Dataset 2: {change['dataset2_preference_rate']:.1%}\n")
                f.write(f"  Change: {change['preference_difference']:+.1%}\n")
            
            f.write("\n")
            f.write("DETAILED CSV FILES:\n")
            f.write("=" * 20 + "\n")
            f.write("• moo_same_letter_ranking_changes.csv - BT ranking changes\n")
            f.write("• moo_same_letter_strength_changes.csv - BT strength changes\n")
            f.write("• moo_pairwise_direction_flips.csv - Preference reversals\n")
            f.write("• moo_pairwise_preference_changes.csv - Large preference changes\n")
    
    def compare_hypothesis_results(self) -> Dict[str, Any]:
        """Compare focused hypothesis testing results between datasets."""
        return {'error': 'Hypothesis comparison not implemented for MOO analysis'}
    
    def compare_key_preferences(self) -> Dict[str, Any]:
        """Compare key preference results between datasets."""
        return {'error': 'Key preference comparison not implemented for MOO analysis'}
    
    def compare_transition_preferences(self) -> Dict[str, Any]:
        """Compare transition preference results between datasets."""
        return {'error': 'Transition preference comparison not implemented for MOO analysis'}
    
    def _load_detailed_hypothesis_data(self, dataset_path: Path) -> Dict[str, Any]:
        """Load detailed focused hypothesis results from a dataset."""
        return {}
    
    def _load_hypothesis_data(self, dataset_path: Path) -> Dict[str, Any]:
        """Load focused hypothesis results from a dataset."""
        return {}
    
    def _parse_hypothesis_text_report(self, text_path: Path) -> Dict[str, Any]:
        """Parse hypothesis results from text report."""
        return {}
    
    def _load_key_data(self, dataset_path: Path) -> pd.DataFrame:
        """Load key preference statistics from a dataset."""
        return pd.DataFrame()
    
    def _load_transition_data(self, dataset_path: Path) -> pd.DataFrame:
        """Load transition preference statistics from a dataset."""
        return pd.DataFrame()
    
    def _generate_overall_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of differences between datasets."""
        
        summary = {
            'major_differences': [],
            'stability_assessment': 'unknown',
            'key_findings': []
        }
        
        # Analyze MOO objectives changes
        if 'moo_comparison' in comparison_results:
            moo_comp = comparison_results['moo_comparison']
            
            # Check for direction flips in pairwise preferences
            pairwise_flips = len(moo_comp.get('pairwise_comparison', {}).get('direction_flips', []))
            if pairwise_flips > 0:
                summary['major_differences'].append(f"{pairwise_flips} pairwise key preferences REVERSED direction")
            
            # Check MOO consistency
            moo_consistency = moo_comp.get('overall_consistency', {})
            consistency_rating = moo_consistency.get('overall_rating', 'unknown')
            if consistency_rating == 'low':
                summary['major_differences'].append("MOO objectives show low consistency")
            
            # Set stability based on same-letter correlation
            same_letter_corr = moo_comp.get('same_letter_comparison', {}).get('correlation_stats', {}).get('correlation_coefficient', 0)
            if same_letter_corr > 0.8:
                summary['stability_assessment'] = 'high'
            elif same_letter_corr > 0.6:
                summary['stability_assessment'] = 'moderate'
            else:
                summary['stability_assessment'] = 'low'
        
        # Generate key findings
        if not summary['major_differences']:
            summary['key_findings'].append("MOO objectives are highly consistent between datasets")
        else:
            summary['key_findings'].append("Significant differences found in MOO objectives between datasets")
            summary['key_findings'].extend(summary['major_differences'])
        
        return summary
    
    def _save_hypothesis_comparison(self, comparison: Dict[str, Any]) -> None:
        """Save hypothesis comparison results."""
        pass
    
    def _save_comprehensive_hypothesis_comparison(self) -> None:
        """Save comprehensive hypothesis comparison with all available statistics."""
        pass
    
    def _save_key_comparison(self, comparison: Dict[str, Any]) -> None:
        """Save key preference comparison results."""
        pass
    
    def _save_transition_comparison(self, comparison: Dict[str, Any]) -> None:
        """Save transition preference comparison results."""
        pass
    
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
            f.write("• moo_objectives_comparison_report.txt - MOO objectives detailed comparison\n")
            f.write("• moo_same_letter_ranking_changes.csv - MOO same-letter BT ranking changes\n")
            f.write("• moo_pairwise_direction_flips.csv - MOO pairwise preference reversals\n")
            f.write("• moo_same_letter_strength_changes.csv - MOO same-letter BT strength changes\n")
            f.write("• moo_pairwise_preference_changes.csv - MOO large preference changes\n")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Compare keyboard preference analysis results between datasets")
    parser.add_argument('--dataset1', required=True, help='Path to first dataset results directory')
    parser.add_argument('--dataset2', required=True, help='Path to second dataset results directory')
    parser.add_argument('--output', required=True, help='Output directory for comparison results')
    parser.add_argument('--focus', choices=['hypotheses', 'keys', 'transitions', 'moo'], 
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
            
            # Show MOO pairwise direction flips if any
            if 'moo_comparison' in results:
                moo_comp = results['moo_comparison']
                pairwise_flips = moo_comp.get('pairwise_comparison', {}).get('direction_flips', [])
                if pairwise_flips:
                    print(f"\nMOO CRITICAL: {len(pairwise_flips)} pairwise key preferences reversed!")
                    for flip in pairwise_flips[:2]:  # Show top 2
                        print(f"   • {flip['comparison']}: {flip['dataset1_favored']} → {flip['dataset2_favored']} "
                              f"(magnitude: {flip['flip_magnitude']:.3f})")
        
        return 0
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())