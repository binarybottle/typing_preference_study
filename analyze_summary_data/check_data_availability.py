#!/usr/bin/env python3
"""
Hypothesis Data Availability Diagnostic

This script analyzes your actual data to show how many comparisons 
each hypothesis will have - the critical information for determining
which tests are viable.

Usage:
    python data_availability_check.py --data filtered_data.csv [--include-column5]
"""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class KeyPosition:
    """Represents a key's position on the keyboard."""
    key: str
    row: int      # 1=upper, 2=home, 3=lower
    column: int   # 1=leftmost, 5=index center
    finger: int   # 1=pinky, 2=ring, 3=middle, 4=index

class HypothesisDataDiagnostic:
    """Diagnostic tool for analyzing hypothesis data availability."""
    
    def __init__(self, include_column5: bool = False):
        self.include_column5 = include_column5
        self.key_positions = self._define_keyboard_layout()
        
        # Define current active hypotheses (ALL 20 from main script)
        self.active_hypotheses = {
            # Column separation effects - ALL tests (binary + specific)
            #'same_row_column_sep_1v2or3': {'description': 'Same-row: 1 vs >1 columns apart', 'category': 'same_row_column_separation_binary', 'values': ['1', '>1']},
            'same_row_column_sep_1v2': {'description': 'Same-row: 1 vs 2 columns apart', 'category': 'same_row_column_separation', 'values': ['1', '2']},
            'same_row_column_sep_2v3': {'description': 'Same-row: 2 vs 3 columns apart', 'category': 'same_row_column_separation', 'values': ['2', '3']},
            #'cross_row_1row_sep_column_sep_1v2or3': {'description': 'Cross-row (1 row): 1 vs >1 columns apart', 'category': 'cross_row_1row_column_separation_binary', 'values': ['1', '>1']},
            'cross_row_column_sep_1v2': {'description': 'Cross-row: 1 vs 2 columns apart', 'category': 'cross_row_column_separation', 'values': ['1', '2']},
            'cross_row_column_sep_2v3': {'description': 'Cross-row: 2 vs 3 columns apart', 'category': 'cross_row_column_separation', 'values': ['2', '3']},
            'cross_row_1row_sep_same_vs_diff_col': {'description': 'Cross-row (1 row): same vs different column', 'category': 'cross_row_1row_same_vs_diff', 'values': ['same', 'different']},
            'cross_row_2row_sep_same_vs_diff_col': {'description': 'Cross-row (2 rows): same vs different column', 'category': 'cross_row_2row_same_vs_diff', 'values': ['same', 'different']},
            
            # Movement effects - 1 test  
            'home_vs_other_keys': {'description': 'Home keys: home vs other keys', 'category': 'involves_home_keys', 'values': ['home', 'other']},
            
            # Vertical separation effects - 2 tests
            'row_sep_0v1': {'description': 'Row separation: 0 vs 1', 'category': 'row_separation', 'values': ['0', '1']},
            'row_sep_1v2': {'description': 'Row separation: 1 vs 2', 'category': 'row_separation', 'values': ['1', '2']},
            
            # Finger preferences - 3 tests
            'finger_f4_vs_f3': {'description': 'Finger: F4 vs F3', 'category': 'dominant_finger', 'values': ['4', '3']},
            'finger_f3_vs_f2': {'description': 'Finger: F3 vs F2', 'category': 'dominant_finger', 'values': ['3', '2']},
            'finger_f1_vs_other': {'description': 'Finger: F1 vs other fingers', 'category': 'involves_finger_1', 'values': ['1', 'other']},
            
            # Direction effects - 2 tests
            'same_row_direction': {'description': 'Same-row: inner vs outer roll', 'category': 'same_row_direction', 'values': ['inner_roll', 'outer_roll']},
            'cross_row_direction': {'description': 'Cross-row: inner vs outer roll', 'category': 'cross_row_direction', 'values': ['inner_roll_cross', 'outer_roll_cross']},
            
            # Column-specific row preferences - 4 tests
            'column1_upper_vs_lower': {'description': 'Column 1: Q vs Z (F1)', 'category': 'column1_row_pref', 'values': ['1', '3']},
            'column2_upper_vs_lower': {'description': 'Column 2: W vs X (F2)', 'category': 'column2_row_pref', 'values': ['1', '3']},
            'column3_upper_vs_lower': {'description': 'Column 3: E vs C (F3)', 'category': 'column3_row_pref', 'values': ['1', '3']},
            'column4_upper_vs_lower': {'description': 'Column 4: R vs V (F4)', 'category': 'column4_row_pref', 'values': ['1', '3']},
        }
        
        # Add column 5 hypothesis if requested
        if include_column5:
            self.active_hypotheses['column5_vs_other'] = {
                'description': 'Column 5: column 5 vs other columns', 
                'category': 'involves_column5', 
                'values': ['column5', 'other']
            }
        
        # Define sections for organization (include ALL hypotheses)
        self.sections = {
            'Column separation effects': [
                'same_row_column_sep_1v2or3', 'same_row_column_sep_1v2', 'same_row_column_sep_2v3',
                'cross_row_1row_sep_column_sep_1v2or3', 'cross_row_column_sep_1v2', 'cross_row_column_sep_2v3',
                'cross_row_1row_sep_same_vs_diff_col', 'cross_row_2row_sep_same_vs_diff_col'
            ],
            'Movement effects': ['home_vs_other_keys'],
            'Vertical separation effects': ['row_sep_0v1', 'row_sep_1v2'],
            'Finger preferences': ['finger_f4_vs_f3', 'finger_f3_vs_f2', 'finger_f1_vs_other'],
            'Direction effects': ['same_row_direction', 'cross_row_direction'],
            'Column-specific row preferences': ['column1_upper_vs_lower', 'column2_upper_vs_lower', 'column3_upper_vs_lower', 'column4_upper_vs_lower']
        }
        
        if include_column5:
            self.sections['Column-specific row preferences'].append('column5_vs_other')
    
    def _define_keyboard_layout(self) -> Dict[str, KeyPosition]:
        """Define keyboard layout based on configuration."""
        layout = {
            # Upper row (row 1)
            'q': KeyPosition('q', 1, 1, 1), 'w': KeyPosition('w', 1, 2, 2),
            'e': KeyPosition('e', 1, 3, 3), 'r': KeyPosition('r', 1, 4, 4),
            
            # Home row (row 2)
            'a': KeyPosition('a', 2, 1, 1), 's': KeyPosition('s', 2, 2, 2),
            'd': KeyPosition('d', 2, 3, 3), 'f': KeyPosition('f', 2, 4, 4),
            
            # Lower row (row 3)
            'z': KeyPosition('z', 3, 1, 1), 'x': KeyPosition('x', 3, 2, 2),
            'c': KeyPosition('c', 3, 3, 3), 'v': KeyPosition('v', 3, 4, 4),
        }
        
        # Add column 5 keys if requested
        if self.include_column5:
            layout.update({
                't': KeyPosition('t', 1, 5, 4),
                'g': KeyPosition('g', 2, 5, 4),
                'b': KeyPosition('b', 3, 5, 4)
            })
        
        return layout
    
    def analyze_data_availability(self, data_path: str) -> Dict[str, Any]:
        """Analyze data availability for each hypothesis."""
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        
        # Basic data info
        print(f"Loaded {len(data)} rows from {data['user_id'].nunique()} participants")
        
        # Filter to valid data
        data = data[data['sliderValue'] != 0].copy()
        data['chosen_bigram'] = data['chosen_bigram'].astype(str).str.lower()
        data['unchosen_bigram'] = data['unchosen_bigram'].astype(str).str.lower()
        
        print(f"Using {len(data)} consistent choice rows")
        print()
        
        # Classify all bigrams
        bigram_classifications = self._classify_all_bigrams(data)
        
        # Analyze each hypothesis
        hypothesis_data = {}
        
        for hyp_name, hyp_config in self.active_hypotheses.items():
            comparisons = self._extract_comparisons_for_hypothesis(
                data, bigram_classifications, hyp_config
            )
            
            total_comparisons = sum(comp['total'] for comp in comparisons.values())
            
            # Calculate data breakdown
            data_breakdown = {}
            for (val1, val2), comp_data in comparisons.items():
                data_breakdown[f"{val1}_vs_{val2}"] = {
                    'val1_wins': comp_data['wins_item1'],
                    'val2_wins': comp_data['total'] - comp_data['wins_item1'],
                    'total': comp_data['total']
                }
            
            hypothesis_data[hyp_name] = {
                'description': hyp_config['description'],
                'category': hyp_config['category'],
                'values': hyp_config['values'],
                'total_comparisons': total_comparisons,
                'data_breakdown': data_breakdown,
                'viable': total_comparisons >= 10  # Minimum threshold
            }
        
        return hypothesis_data
    
    def _classify_all_bigrams(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Classify each bigram according to all hypothesis dimensions."""
        classifications = {}
        
        # Get all unique bigrams
        all_bigrams = set()
        for col in ['chosen_bigram', 'unchosen_bigram']:
            all_bigrams.update(data[col].unique())
        
        for bigram in all_bigrams:
            if len(bigram) >= 2:
                char1, char2 = bigram[0], bigram[1]
                
                if char1 in self.key_positions and char2 in self.key_positions:
                    pos1 = self.key_positions[char1]
                    pos2 = self.key_positions[char2]
                    
                    classifications[bigram] = self._classify_single_bigram(pos1, pos2)
        
        return classifications
    
    def _classify_single_bigram(self, pos1: KeyPosition, pos2: KeyPosition) -> Dict[str, Any]:
        """Classify a single bigram according to hypothesis dimensions."""
        
        # Skip column 5 keys unless specifically testing column 5
        column5_keys = {'t', 'g', 'b'}
        if not self.include_column5:
            if pos1.key in column5_keys or pos2.key in column5_keys:
                return {cat: None for cat in [
                    'same_row_column_separation_binary', 'cross_row_1row_column_separation_binary',
                    'same_row_column_separation', 'cross_row_column_separation',
                    'cross_row_1row_same_vs_diff', 'cross_row_2row_same_vs_diff',
                    'involves_home_keys', 'row_separation',
                    'dominant_finger', 'involves_finger_1', 'same_row_direction', 'cross_row_direction',
                    'column1_row_pref', 'column2_row_pref', 'column3_row_pref', 'column4_row_pref',
                    'involves_column5'
                ]}
        
        # Basic measurements
        column_separation = abs(pos1.column - pos2.column)
        row_separation = abs(pos1.row - pos2.row)
        finger_separation = abs(pos1.finger - pos2.finger)
        
        # Home keys
        home_keys = {'a', 's', 'd', 'f'}
        involves_home_keys = 'home' if (pos1.key in home_keys or pos2.key in home_keys) else 'other'
        
        # Finger analysis
        dominant_finger = max(pos1.finger, pos2.finger)
        involves_finger_1 = '1' if (pos1.finger == 1 or pos2.finger == 1) else 'other'
        
        # Direction calculation
        if row_separation == 0:  # Same row
            if pos2.column > pos1.column:
                direction = 'inner_roll'
            elif pos2.column < pos1.column:
                direction = 'outer_roll'
            else:
                direction = 'same_column'
        else:  # Cross row
            if pos2.column > pos1.column:
                direction = 'inner_roll_cross'
            elif pos2.column < pos1.column:
                direction = 'outer_roll_cross'
            else:
                direction = 'same_column_cross'
        
        # Column-specific row preferences
        column_row_prefs = {}
        max_column = 5 if self.include_column5 else 4
        
        for column in range(1, max_column + 1):
            keys_in_column = []
            if pos1.column == column:
                keys_in_column.append(pos1)
            if pos2.column == column:
                keys_in_column.append(pos2)
            
            if len(keys_in_column) == 1:
                column_row_prefs[f'column{column}_row_pref'] = str(keys_in_column[0].row)
            elif len(keys_in_column) == 2:
                rows = [k.row for k in keys_in_column]
                if 1 in rows and 3 in rows:
                    column_row_prefs[f'column{column}_row_pref'] = str(pos1.row)
                elif 1 in rows:
                    column_row_prefs[f'column{column}_row_pref'] = '1'
                elif 3 in rows:
                    column_row_prefs[f'column{column}_row_pref'] = '3'
                else:
                    column_row_prefs[f'column{column}_row_pref'] = None
            else:
                column_row_prefs[f'column{column}_row_pref'] = None
        
        # Column 5 involvement
        if self.include_column5:
            involves_column5 = 'column5' if (pos1.column == 5 or pos2.column == 5) else 'other'
        else:
            involves_column5 = None
        
        return {
            # Binary column separation (NEW)
            'same_row_column_separation_binary': ('1' if column_separation == 1 else '>1') if (row_separation == 0 and column_separation > 0) else None,
            'cross_row_1row_column_separation_binary': ('1' if column_separation == 1 else '>1') if (row_separation == 1 and column_separation > 0) else None,
            
            # Specific column separation (OLD)  
            'same_row_column_separation': str(column_separation) if row_separation == 0 else None,
            'cross_row_column_separation': str(column_separation) if row_separation > 0 else None,
            
            # Cross-row same vs different column
            'cross_row_1row_same_vs_diff': 'same' if (row_separation == 1 and column_separation == 0) else ('different' if (row_separation == 1 and column_separation > 0) else None),
            'cross_row_2row_same_vs_diff': 'same' if (row_separation == 2 and column_separation == 0) else ('different' if (row_separation == 2 and column_separation > 0) else None),
            
            # Home key involvement
            'involves_home_keys': involves_home_keys,
            
            # Row separation
            'row_separation': str(row_separation),
            
            # Finger preferences
            'dominant_finger': str(dominant_finger) if finger_separation > 0 else None,
            'involves_finger_1': involves_finger_1 if finger_separation > 0 else None,
            
            # Direction
            'same_row_direction': direction if row_separation == 0 and direction not in ['same_column'] else None,
            'cross_row_direction': direction if row_separation > 0 and direction not in ['same_column_cross'] else None,
            
            # Column-specific row preferences
            **column_row_prefs,
            
            # Column 5 involvement
            'involves_column5': involves_column5
        }
    
    def _extract_comparisons_for_hypothesis(self, data: pd.DataFrame, 
                                        classifications: Dict[str, Dict[str, Any]], 
                                        hyp_config: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Extract pairwise comparisons for a specific hypothesis."""
        
        comparisons = defaultdict(lambda: {'wins_item1': 0, 'total': 0})
        category = hyp_config['category']
        target_values = set(hyp_config['values'])
        
        for _, row in data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            chosen_class = classifications.get(chosen, {})
            unchosen_class = classifications.get(unchosen, {})
            
            chosen_val = chosen_class.get(category)
            unchosen_val = unchosen_class.get(category)
            
            # Only include comparisons between the target values
            if (chosen_val in target_values and unchosen_val in target_values and 
                chosen_val != unchosen_val):
                
                # Create comparison pair (ordered consistently)
                val1, val2 = sorted([chosen_val, unchosen_val])
                pair = (val1, val2)
                
                comparisons[pair]['total'] += 1
                if chosen_val == val1:
                    comparisons[pair]['wins_item1'] += 1
        
        return dict(comparisons)
    
    def print_analysis(self, hypothesis_data: Dict[str, Any]):
        """Print comprehensive analysis of data availability."""
        
        print("HYPOTHESIS DATA AVAILABILITY ANALYSIS")
        print("=" * 55)
        print()
        
        column5_status = "included" if self.include_column5 else "excluded"
        print(f"Column 5 keys: {column5_status}")
        print(f"Total active hypotheses: {len(self.active_hypotheses)} (now includes all binary + specific comparisons)")
        print()
        
        # Summary stats
        viable_count = sum(1 for h in hypothesis_data.values() if h['viable'])
        total_comparisons = sum(h['total_comparisons'] for h in hypothesis_data.values())
        
        print(f"SUMMARY:")
        print(f"  Viable hypotheses (≥10 comparisons): {viable_count}/{len(self.active_hypotheses)}")
        print(f"  Total comparisons across all tests: {total_comparisons:,}")
        print()
        
        # Detailed analysis by section
        for section_name, hyp_names in self.sections.items():
            section_hyps = [(name, hypothesis_data.get(name)) for name in hyp_names if name in hypothesis_data]
            if not section_hyps:
                continue
                
            print(f"{section_name.upper()}:")
            print("-" * (len(section_name) + 1))
            
            for hyp_name, hyp_data in section_hyps:
                if hyp_data is None:
                    continue
                    
                viable_icon = "✓" if hyp_data['viable'] else "✗"
                total_comps = hyp_data['total_comparisons']
                
                print(f"  {viable_icon} {hyp_name}")
                print(f"    → {hyp_data['description']}")
                print(f"    → Total comparisons: {total_comps:,}")
                
                # Show data breakdown
                for comparison, data_info in hyp_data['data_breakdown'].items():
                    val1, val2 = comparison.split('_vs_')
                    val1_wins = data_info['val1_wins']
                    val2_wins = data_info['val2_wins']
                    total = data_info['total']
                    
                    if total > 0:
                        val1_pct = (val1_wins / total) * 100
                        val2_pct = (val2_wins / total) * 100
                        print(f"    → {val1}: {val1_wins:,} wins ({val1_pct:.1f}%) | {val2}: {val2_wins:,} wins ({val2_pct:.1f}%)")
                
                if total_comps < 10:
                    print(f"    ⚠️  Insufficient data (need ≥10, have {total_comps})")
                print()
            
            print()
        
        # Warning about insufficient data
        insufficient_hyps = [name for name, data in hypothesis_data.items() if not data['viable']]
        if insufficient_hyps:
            print("HYPOTHESES WITH INSUFFICIENT DATA:")
            print("-" * 40)
            for hyp_name in insufficient_hyps:
                total = hypothesis_data[hyp_name]['total_comparisons']
                print(f"  ✗ {hyp_name}: {total} comparisons (need ≥10)")
            print()
            print("Consider combining categories or collecting more data for these tests.")
        else:
            print("✅ All hypotheses have sufficient data for analysis!")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Analyze data availability for keyboard preference hypotheses")
    parser.add_argument('--data', required=True, help='Path to CSV file with bigram choice data')
    parser.add_argument('--include-column5', action='store_true', help='Include column 5 keys (T, G, B) in analysis')
    
    args = parser.parse_args()
    
    try:
        diagnostic = HypothesisDataDiagnostic(include_column5=args.include_column5)
        hypothesis_data = diagnostic.analyze_data_availability(args.data)
        diagnostic.print_analysis(hypothesis_data)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())