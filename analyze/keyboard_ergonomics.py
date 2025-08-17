# keyboard_ergonomics.py
"""
Keyboard ergonomics statistical analysis module.
Tests ergonomic preferences for typing based on bigram choice data.
Integrates with existing analyze_data.py framework.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar
import logging
from typing import Dict, List, Tuple, Any
import os

# Import your existing keyboard mappings
from keymaps import row_map, column_map, finger_map

logger = logging.getLogger(__name__)

class KeyboardErgonomicsAnalysis:
    """
    Statistical analysis of keyboard ergonomics based on typing preferences.
    
    Tests the 5 main research questions about row, column, and movement preferences.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get('analysis', {}).get('alpha_level', 0.05)
        self.correction_method = config.get('analysis', {}).get('correction_method', 'fdr_bh')
        
    def run_all_ergonomics_tests(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """
        Run all 5 ergonomics research questions and generate comprehensive report.
        
        Args:
            data: DataFrame with columns [user_id, chosen_bigram, unchosen_bigram, is_consistent]
            output_folder: Directory to save results
            
        Returns:
            Dictionary containing all test results
        """
        logger.info("Starting keyboard ergonomics analysis...")
        
        # Create keyboard_ergonomics subdirectory
        ergonomics_folder = os.path.join(output_folder, 'keyboard_ergonomics')
        os.makedirs(ergonomics_folder, exist_ok=True)
        
        # Add keyboard position features
        enhanced_data = self._add_keyboard_features(data)
        
        # Filter to consistent choices only for main analysis
        consistent_data = enhanced_data[enhanced_data['is_consistent'] == True].copy()
        logger.info(f"Using {len(consistent_data)} consistent choice rows from {consistent_data['user_id'].nunique()} participants")
        
        # Run all statistical tests
        results = {}
        
        logger.info("Testing Question 1: Row preferences...")
        results['question_1'] = self.test_row_preferences(consistent_data)
        
        logger.info("Testing Question 2: Row pair preferences...")
        results['question_2'] = self.test_row_pair_preferences(consistent_data)
        
        logger.info("Testing Question 3: Column preferences...")
        results['question_3'] = self.test_column_preferences(consistent_data)
        
        logger.info("Testing Question 4: Column pair preferences...")
        results['question_4'] = self.test_column_pair_preferences(consistent_data)
        
        logger.info("Testing Question 5: Direction preferences...")
        results['question_5'] = self.test_direction_preferences(consistent_data)
        
        # Generate comprehensive report
        self._generate_ergonomics_report(results, ergonomics_folder)
        
        logger.info("Keyboard ergonomics analysis complete!")
        return results
    
    def _add_keyboard_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add keyboard position features using existing keymaps.
        
        Args:
            data: DataFrame with chosen_bigram and unchosen_bigram columns
            
        Returns:
            DataFrame with added keyboard position features
        """
        enhanced_data = data.copy()
        
        # Add features for chosen and unchosen bigrams
        for bigram_type in ['chosen_bigram', 'unchosen_bigram']:
            # Extract first and second characters
            enhanced_data[f'{bigram_type}_char1'] = enhanced_data[bigram_type].str[0]
            enhanced_data[f'{bigram_type}_char2'] = enhanced_data[bigram_type].str[1]
            
            # Add row features (using your existing row_map: 1=bottom, 2=home, 3=top)
            enhanced_data[f'{bigram_type}_row1'] = enhanced_data[f'{bigram_type}_char1'].map(row_map)
            enhanced_data[f'{bigram_type}_row2'] = enhanced_data[f'{bigram_type}_char2'].map(row_map)
            
            # Add column features (using your existing column_map: 1-10)
            enhanced_data[f'{bigram_type}_col1'] = enhanced_data[f'{bigram_type}_char1'].map(column_map)
            enhanced_data[f'{bigram_type}_col2'] = enhanced_data[f'{bigram_type}_char2'].map(column_map)
            
            # Add finger features (using your existing finger_map: 1=index, 4=pinky)
            enhanced_data[f'{bigram_type}_finger1'] = enhanced_data[f'{bigram_type}_char1'].map(finger_map)
            enhanced_data[f'{bigram_type}_finger2'] = enhanced_data[f'{bigram_type}_char2'].map(finger_map)
            
            # Add derived features
            enhanced_data[f'{bigram_type}_same_row'] = (
                enhanced_data[f'{bigram_type}_row1'] == enhanced_data[f'{bigram_type}_row2']
            )
            enhanced_data[f'{bigram_type}_row_distance'] = abs(
                enhanced_data[f'{bigram_type}_row1'] - enhanced_data[f'{bigram_type}_row2']
            )
            enhanced_data[f'{bigram_type}_col_distance'] = abs(
                enhanced_data[f'{bigram_type}_col1'] - enhanced_data[f'{bigram_type}_col2']
            )
            enhanced_data[f'{bigram_type}_finger_distance'] = abs(
                enhanced_data[f'{bigram_type}_finger1'] - enhanced_data[f'{bigram_type}_finger2']
            )
            
            # Direction toward column 5 (right-most left hand column)
            enhanced_data[f'{bigram_type}_direction_toward_5'] = (
                enhanced_data[f'{bigram_type}_col2'] > enhanced_data[f'{bigram_type}_col1']
            ).astype(int)
        
        return enhanced_data
    
    def test_row_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 1: Row preferences
        - Is home row preferred over top/bottom?
        - Is top row preferred over bottom?
        - Row preferences by finger type
        """
        results = {}
        
        # Test 1a: Home row vs others
        home_vs_others = self._test_home_row_preference(data)
        results['home_vs_others'] = home_vs_others
        
        # Test 1b: Top vs bottom row
        top_vs_bottom = self._test_top_vs_bottom_preference(data)
        results['top_vs_bottom'] = top_vs_bottom
        
        # Test 1c: Row preferences by finger type
        finger_preferences = self._test_row_preferences_by_finger(data)
        results['finger_preferences'] = finger_preferences
        
        return results
    
    def test_row_pair_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 2: Row pair preferences
        - Same row vs different row bigrams
        - Adjacent rows vs non-adjacent rows
        """
        results = {}
        
        # Test same row vs different row
        same_vs_diff = self._test_same_vs_different_row(data)
        results['same_vs_different_row'] = same_vs_diff
        
        # Test adjacent vs non-adjacent rows
        adjacent_vs_nonadjacent = self._test_adjacent_vs_nonadjacent_rows(data)
        results['adjacent_vs_nonadjacent'] = adjacent_vs_nonadjacent
        
        return results
    
    def test_column_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 3: Column preferences
        - Sequential column comparisons (5>4, 4>3, 3>2, 2>1)
        """
        results = {}
        
        # Test each adjacent column pair
        column_comparisons = [
            (5, 4, "col5_vs_col4"),
            (4, 3, "col4_vs_col3"), 
            (3, 2, "col3_vs_col2"),
            (2, 1, "col2_vs_col1")
        ]
        
        for col_high, col_low, test_name in column_comparisons:
            result = self._test_column_preference(data, col_high, col_low)
            results[test_name] = result
            
        return results
    
    def test_column_pair_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 4: Column pair preferences
        - Adjacent vs remote fingers (within same row and across different rows)
        - Different separation distances (within same row and across different rows)
        """
        results = {}
        
        # Test adjacent vs remote fingers overall
        adjacent_vs_remote = self._test_adjacent_vs_remote_fingers(data)
        results['adjacent_vs_remote_overall'] = adjacent_vs_remote
        
        # Test adjacent vs remote fingers: same-row bigrams only
        # Filter to comparisons where both bigrams are same-row
        same_row_comparisons = data[
            data['chosen_bigram_same_row'] & data['unchosen_bigram_same_row']
        ]
        if len(same_row_comparisons) > 0:
            adjacent_vs_remote_same_row = self._test_adjacent_vs_remote_fingers(same_row_comparisons)
            results['adjacent_vs_remote_same_row'] = adjacent_vs_remote_same_row
        else:
            results['adjacent_vs_remote_same_row'] = {'test': 'No same-row comparisons found', 'p_value': np.nan}
        
        # Test adjacent vs remote fingers: cross-row bigrams only
        # Filter to comparisons where both bigrams are cross-row
        diff_row_comparisons = data[
            (~data['chosen_bigram_same_row']) & (~data['unchosen_bigram_same_row'])
        ]
        if len(diff_row_comparisons) > 0:
            adjacent_vs_remote_diff_row = self._test_adjacent_vs_remote_fingers(diff_row_comparisons)
            results['adjacent_vs_remote_diff_row'] = adjacent_vs_remote_diff_row
        else:
            results['adjacent_vs_remote_diff_row'] = {'test': 'No cross-row comparisons found', 'p_value': np.nan}
        
        # Test different finger separation distances overall
        separation_distances = self._test_finger_separation_distances(data)
        results['separation_distances_overall'] = separation_distances
        
        # Test separation distances: same-row bigrams only
        if len(same_row_comparisons) > 0:
            separation_distances_same_row = self._test_finger_separation_distances(same_row_comparisons)
            results['separation_distances_same_row'] = separation_distances_same_row
        else:
            results['separation_distances_same_row'] = {}
        
        # Test separation distances: cross-row bigrams only
        if len(diff_row_comparisons) > 0:
            separation_distances_diff_row = self._test_finger_separation_distances(diff_row_comparisons)
            results['separation_distances_diff_row'] = separation_distances_diff_row
        else:
            results['separation_distances_diff_row'] = {}
        
        return results
    
    def test_direction_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 5: Direction preferences
        - Low-to-high finger sequences vs high-to-low
        """
        results = {}
        
        # Test directional preference
        direction_preference = self._test_directional_preference(data)
        results['direction_preference'] = direction_preference
        
        return results
    
    # Individual test implementations
    
    def _test_home_row_preference(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if home row (row 2) is preferred over top/bottom rows (rows 1&3)"""
        
        # Create comparison data
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_rows = (row['chosen_bigram_row1'], row['chosen_bigram_row2'])
            unchosen_rows = (row['unchosen_bigram_row1'], row['unchosen_bigram_row2'])
            
            # Score each bigram: +1 for each key on home row
            chosen_home_score = sum(1 for r in chosen_rows if r == 2)
            unchosen_home_score = sum(1 for r in unchosen_rows if r == 2)
            
            if chosen_home_score != unchosen_home_score:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_more_home': chosen_home_score > unchosen_home_score,
                    'home_difference': chosen_home_score - unchosen_home_score
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        # Binomial test: is proportion choosing more home row significantly > 0.5?
        n_chose_more_home = comparison_df['chose_more_home'].sum()
        n_total = len(comparison_df)
        
        # Two-tailed binomial test
        p_value = stats.binomtest(n_chose_more_home, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': 'Home row preference',
            'n_comparisons': n_total,
            'n_chose_more_home': n_chose_more_home,
            'proportion_chose_home': n_chose_more_home / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_more_home / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _test_top_vs_bottom_preference(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if top row (row 3) is preferred over bottom row (row 1)"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_rows = (row['chosen_bigram_row1'], row['chosen_bigram_row2'])
            unchosen_rows = (row['unchosen_bigram_row1'], row['unchosen_bigram_row2'])
            
            # Score: +1 for top row, -1 for bottom row
            def score_bigram(rows):
                score = 0
                for r in rows:
                    if r == 3:  # top row
                        score += 1
                    elif r == 1:  # bottom row
                        score -= 1
                return score
            
            chosen_score = score_bigram(chosen_rows)
            unchosen_score = score_bigram(unchosen_rows)
            
            if chosen_score != unchosen_score:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_more_top': chosen_score > unchosen_score,
                    'top_bottom_difference': chosen_score - unchosen_score
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_more_top = comparison_df['chose_more_top'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_more_top, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': 'Top vs bottom row preference',
            'n_comparisons': n_total,
            'n_chose_more_top': n_chose_more_top,
            'proportion_chose_top': n_chose_more_top / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_more_top / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _test_row_preferences_by_finger(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test row preferences for different finger types"""
        
        finger_results = {}
        
        # Test each finger (1=index to 4=pinky)
        for finger_num in range(1, 5):
            finger_name = ['index', 'middle', 'ring', 'pinky'][finger_num - 1]
            
            comparison_data = []
            
            for _, row in data.iterrows():
                # Extract positions that use this finger
                chosen_finger_positions = []
                unchosen_finger_positions = []
                
                if row['chosen_bigram_finger1'] == finger_num:
                    chosen_finger_positions.append(row['chosen_bigram_row1'])
                if row['chosen_bigram_finger2'] == finger_num:
                    chosen_finger_positions.append(row['chosen_bigram_row2'])
                    
                if row['unchosen_bigram_finger1'] == finger_num:
                    unchosen_finger_positions.append(row['unchosen_bigram_row1'])
                if row['unchosen_bigram_finger2'] == finger_num:
                    unchosen_finger_positions.append(row['unchosen_bigram_row2'])
                
                # Compare top vs bottom for this finger
                if chosen_finger_positions and unchosen_finger_positions:
                    chosen_top_score = sum(1 for r in chosen_finger_positions if r == 3)
                    chosen_bottom_score = sum(1 for r in chosen_finger_positions if r == 1)
                    
                    unchosen_top_score = sum(1 for r in unchosen_finger_positions if r == 3)
                    unchosen_bottom_score = sum(1 for r in unchosen_finger_positions if r == 1)
                    
                    chosen_preference = chosen_top_score - chosen_bottom_score
                    unchosen_preference = unchosen_top_score - unchosen_bottom_score
                    
                    if chosen_preference != unchosen_preference:
                        comparison_data.append({
                            'user_id': row['user_id'],
                            'chose_more_top': chosen_preference > unchosen_preference
                        })
            
            if len(comparison_data) > 0:
                comparison_df = pd.DataFrame(comparison_data)
                n_chose_more_top = comparison_df['chose_more_top'].sum()
                n_total = len(comparison_df)
                
                p_value = stats.binomtest(n_chose_more_top, n_total, 0.5, alternative='two-sided').pvalue
                
                finger_results[finger_name] = {
                    'n_comparisons': n_total,
                    'n_chose_more_top': n_chose_more_top,
                    'proportion_chose_top': n_chose_more_top / n_total,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
            else:
                finger_results[finger_name] = {'n_comparisons': 0, 'p_value': np.nan}
        
        return finger_results
    
    def _test_same_vs_different_row(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if same-row bigrams are preferred over different-row bigrams"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_same_row = row['chosen_bigram_same_row']
            unchosen_same_row = row['unchosen_bigram_same_row']
            
            if chosen_same_row != unchosen_same_row:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_same_row': chosen_same_row
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_same_row = comparison_df['chose_same_row'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_same_row, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': 'Same vs different row preference',
            'n_comparisons': n_total,
            'n_chose_same_row': n_chose_same_row,
            'proportion_chose_same_row': n_chose_same_row / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_same_row / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _test_adjacent_vs_nonadjacent_rows(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if adjacent-row bigrams are preferred over non-adjacent (skip) row bigrams"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_row_dist = row['chosen_bigram_row_distance']
            unchosen_row_dist = row['unchosen_bigram_row_distance']
            
            # Only compare when one is adjacent (distance 1) and other is skip (distance 2)
            if (chosen_row_dist == 1 and unchosen_row_dist == 2) or (chosen_row_dist == 2 and unchosen_row_dist == 1):
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_adjacent': chosen_row_dist == 1
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_adjacent = comparison_df['chose_adjacent'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_adjacent, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': 'Adjacent vs non-adjacent rows',
            'n_comparisons': n_total,
            'n_chose_adjacent': n_chose_adjacent,
            'proportion_chose_adjacent': n_chose_adjacent / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_adjacent / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _test_column_preference(self, data: pd.DataFrame, col_high: int, col_low: int) -> Dict[str, Any]:
        """Test preference between two specific columns"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            # Count occurrences of each column in chosen vs unchosen
            chosen_cols = (row['chosen_bigram_col1'], row['chosen_bigram_col2'])
            unchosen_cols = (row['unchosen_bigram_col1'], row['unchosen_bigram_col2'])
            
            chosen_high_count = sum(1 for c in chosen_cols if c == col_high)
            chosen_low_count = sum(1 for c in chosen_cols if c == col_low)
            
            unchosen_high_count = sum(1 for c in unchosen_cols if c == col_high)
            unchosen_low_count = sum(1 for c in unchosen_cols if c == col_low)
            
            chosen_score = chosen_high_count - chosen_low_count
            unchosen_score = unchosen_high_count - unchosen_low_count
            
            if chosen_score != unchosen_score:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_higher_col': chosen_score > unchosen_score
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_higher = comparison_df['chose_higher_col'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_higher, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': f'Column {col_high} vs Column {col_low}',
            'n_comparisons': n_total,
            'n_chose_higher_col': n_chose_higher,
            'proportion_chose_higher': n_chose_higher / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_higher / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _test_adjacent_vs_remote_fingers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if adjacent fingers are preferred over remote fingers"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_finger_dist = row['chosen_bigram_finger_distance']
            unchosen_finger_dist = row['unchosen_bigram_finger_distance']
            
            # Compare adjacent (distance 1) vs remote (distance > 1)
            chosen_adjacent = chosen_finger_dist == 1
            unchosen_adjacent = unchosen_finger_dist == 1
            
            if chosen_adjacent != unchosen_adjacent:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_adjacent': chosen_adjacent
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_adjacent = comparison_df['chose_adjacent'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_adjacent, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': 'Adjacent vs remote fingers',
            'n_comparisons': n_total,
            'n_chose_adjacent': n_chose_adjacent,
            'proportion_chose_adjacent': n_chose_adjacent / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_adjacent / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _test_finger_separation_distances(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test preferences for different finger separation distances"""
        
        distance_results = {}
        
        # Test each distance comparison
        distance_comparisons = [
            (1, 2, "1_finger_vs_2_finger"),
            (1, 3, "1_finger_vs_3_finger"), 
            (2, 3, "2_finger_vs_3_finger")
        ]
        
        for dist1, dist2, comparison_name in distance_comparisons:
            comparison_data = []
            
            for _, row in data.iterrows():
                chosen_dist = row['chosen_bigram_finger_distance']
                unchosen_dist = row['unchosen_bigram_finger_distance']
                
                if (chosen_dist == dist1 and unchosen_dist == dist2) or (chosen_dist == dist2 and unchosen_dist == dist1):
                    comparison_data.append({
                        'user_id': row['user_id'],
                        'chose_smaller_distance': (chosen_dist == dist1) if dist1 < dist2 else (chosen_dist == dist2)
                    })
            
            if len(comparison_data) > 0:
                comparison_df = pd.DataFrame(comparison_data)
                n_chose_smaller = comparison_df['chose_smaller_distance'].sum()
                n_total = len(comparison_df)
                
                p_value = stats.binomtest(n_chose_smaller, n_total, 0.5, alternative='two-sided').pvalue
                
                distance_results[comparison_name] = {
                    'n_comparisons': n_total,
                    'n_chose_smaller_distance': n_chose_smaller,
                    'proportion_chose_smaller': n_chose_smaller / n_total,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
            else:
                distance_results[comparison_name] = {'n_comparisons': 0, 'p_value': np.nan}
        
        return distance_results
    
    def _test_directional_preference(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if low-to-high finger sequences are preferred over high-to-low"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_toward_5 = row['chosen_bigram_direction_toward_5']
            unchosen_toward_5 = row['unchosen_bigram_direction_toward_5']
            
            if chosen_toward_5 != unchosen_toward_5:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_toward_5': chosen_toward_5 == 1
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_toward_5 = comparison_df['chose_toward_5'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_toward_5, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': 'Direction toward column 5 (low-to-high fingers)',
            'n_comparisons': n_total,
            'n_chose_toward_5': n_chose_toward_5,
            'proportion_chose_toward_5': n_chose_toward_5 / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_toward_5 / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _apply_multiple_comparison_correction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple comparison correction across all tests"""
        
        # Collect all p-values
        all_p_values = []
        test_names = []
        
        def extract_p_values(obj, prefix=""):
            if isinstance(obj, dict):
                if 'p_value' in obj and not np.isnan(obj['p_value']):
                    all_p_values.append(obj['p_value'])
                    test_names.append(prefix)
                for key, value in obj.items():
                    if isinstance(value, dict):
                        extract_p_values(value, f"{prefix}.{key}" if prefix else key)
        
        extract_p_values(results)
        
        if len(all_p_values) == 0:
            return results
        
        # Apply correction
        if self.correction_method == 'fdr_bh':
            from statsmodels.stats.multitest import multipletests
            reject, p_corrected, _, _ = multipletests(all_p_values, method='fdr_bh', alpha=self.alpha)
        else:
            # Bonferroni correction
            p_corrected = [p * len(all_p_values) for p in all_p_values]
            reject = [p <= self.alpha for p in p_corrected]
        
        # Update results with corrected p-values
        correction_info = {
            'method': self.correction_method,
            'n_tests': len(all_p_values),
            'alpha_level': self.alpha,
            'test_results': []
        }
        
        for i, (test_name, orig_p, corr_p, significant) in enumerate(zip(test_names, all_p_values, p_corrected, reject)):
            correction_info['test_results'].append({
                'test': test_name,
                'original_p': orig_p,
                'corrected_p': corr_p,
                'significant_corrected': significant
            })
        
        results['multiple_comparison_correction'] = correction_info
        return results
    
    def _generate_ergonomics_report(self, results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive ergonomics analysis report"""
        
        # Apply multiple comparison correction
        results = self._apply_multiple_comparison_correction(results)
        
        report_lines = [
            "Keyboard Ergonomics Analysis Report",
            "==================================\n",
            f"Analysis conducted with α = {self.alpha}",
            f"Multiple comparison correction: {self.correction_method}",
            ""
        ]
        
        # Question 1: Row preferences
        if 'question_1' in results:
            report_lines.extend([
                "Question 1: Row Preferences",
                "-------------------------"
            ])
            
            q1 = results['question_1']
            
            if 'home_vs_others' in q1:
                home_result = q1['home_vs_others']
                report_lines.extend([
                    f"Home row preference:",
                    f"  Proportion choosing more home row keys: {home_result.get('proportion_chose_home', 'N/A'):.3f}",
                    f"  p-value: {home_result.get('p_value', 'N/A'):.3e}",
                    f"  Significant: {'Yes' if home_result.get('significant', False) else 'No'}",
                    ""
                ])
            
            if 'top_vs_bottom' in q1:
                top_result = q1['top_vs_bottom']
                report_lines.extend([
                    f"Top vs bottom row preference:",
                    f"  Proportion choosing more top row keys: {top_result.get('proportion_chose_top', 'N/A'):.3f}",
                    f"  p-value: {top_result.get('p_value', 'N/A'):.3e}",
                    f"  Significant: {'Yes' if top_result.get('significant', False) else 'No'}",
                    ""
                ])
            
            if 'finger_preferences' in q1:
                report_lines.append("Row preferences by finger:")
                for finger, finger_result in q1['finger_preferences'].items():
                    if finger_result.get('n_comparisons', 0) > 0:
                        report_lines.append(
                            f"  {finger.capitalize()}: {finger_result.get('proportion_chose_top', 'N/A'):.3f} "
                            f"prefer top row (p={finger_result.get('p_value', 'N/A'):.3e})"
                        )
                report_lines.append("")
        
        # Question 2: Row pair preferences
        if 'question_2' in results:
            report_lines.extend([
                "Question 2: Row Pair Preferences", 
                "------------------------------"
            ])
            
            q2 = results['question_2']
            
            if 'same_vs_different_row' in q2:
                same_result = q2['same_vs_different_row']
                report_lines.extend([
                    f"Same vs different row preference:",
                    f"  Proportion choosing same-row bigrams: {same_result.get('proportion_chose_same_row', 'N/A'):.3f}",
                    f"  p-value: {same_result.get('p_value', 'N/A'):.3e}",
                    f"  Significant: {'Yes' if same_result.get('significant', False) else 'No'}",
                    ""
                ])
            
            if 'adjacent_vs_nonadjacent' in q2:
                adj_result = q2['adjacent_vs_nonadjacent']
                report_lines.extend([
                    f"Adjacent vs non-adjacent row preference:",
                    f"  Proportion choosing adjacent rows: {adj_result.get('proportion_chose_adjacent', 'N/A'):.3f}",
                    f"  p-value: {adj_result.get('p_value', 'N/A'):.3e}",
                    f"  Significant: {'Yes' if adj_result.get('significant', False) else 'No'}",
                    ""
                ])
        
        # Question 3: Column preferences
        if 'question_3' in results:
            report_lines.extend([
                "Question 3: Column Preferences",
                "----------------------------"
            ])
            
            q3 = results['question_3']
            for test_name, result in q3.items():
                if isinstance(result, dict) and 'test_name' in result:
                    report_lines.extend([
                        f"{result['test_name']}:",
                        f"  Proportion choosing higher column: {result.get('proportion_chose_higher', 'N/A'):.3f}",
                        f"  p-value: {result.get('p_value', 'N/A'):.3e}",
                        f"  Significant: {'Yes' if result.get('significant', False) else 'No'}",
                        ""
                    ])
        
        # Question 4: Column pair preferences  
        if 'question_4' in results:
            report_lines.extend([
                "Question 4: Column Pair Preferences",
                "---------------------------------"
            ])
            
            q4 = results['question_4']
            
            # Overall adjacent vs remote
            if 'adjacent_vs_remote_overall' in q4:
                adj_result = q4['adjacent_vs_remote_overall']
                report_lines.extend([
                    f"Adjacent vs remote fingers (overall):",
                    f"  Proportion choosing adjacent fingers: {adj_result.get('proportion_chose_adjacent', 'N/A'):.3f}",
                    f"  p-value: {adj_result.get('p_value', 'N/A'):.3e}",
                    f"  Significant: {'Yes' if adj_result.get('significant', False) else 'No'}",
                    ""
                ])
            
            # Same row adjacent vs remote
            if 'adjacent_vs_remote_same_row' in q4:
                adj_result = q4['adjacent_vs_remote_same_row']
                if 'proportion_chose_adjacent' in adj_result:
                    report_lines.extend([
                        f"Adjacent vs remote fingers (within same row):",
                        f"  Proportion choosing adjacent fingers: {adj_result.get('proportion_chose_adjacent', 'N/A'):.3f}",
                        f"  p-value: {adj_result.get('p_value', 'N/A'):.3e}",
                        f"  Significant: {'Yes' if adj_result.get('significant', False) else 'No'}",
                        ""
                    ])
                else:
                    report_lines.extend([
                        f"Adjacent vs remote fingers (within same row): {adj_result.get('test', 'No data')}",
                        ""
                    ])
            
            # Different row adjacent vs remote
            if 'adjacent_vs_remote_diff_row' in q4:
                adj_result = q4['adjacent_vs_remote_diff_row']
                if 'proportion_chose_adjacent' in adj_result:
                    report_lines.extend([
                        f"Adjacent vs remote fingers (across different rows):",
                        f"  Proportion choosing adjacent fingers: {adj_result.get('proportion_chose_adjacent', 'N/A'):.3f}",
                        f"  p-value: {adj_result.get('p_value', 'N/A'):.3e}",
                        f"  Significant: {'Yes' if adj_result.get('significant', False) else 'No'}",
                        ""
                    ])
                else:
                    report_lines.extend([
                        f"Adjacent vs remote fingers (across different rows): {adj_result.get('test', 'No data')}",
                        ""
                    ])
            
            # Overall finger separation distances
            if 'separation_distances_overall' in q4:
                report_lines.append("Finger separation distance preferences (overall):")
                for distance_name, result in q4['separation_distances_overall'].items():
                    if result.get('n_comparisons', 0) > 0:
                        report_lines.append(
                            f"  {distance_name}: {result.get('proportion_chose_smaller', 'N/A'):.3f} "
                            f"prefer smaller distance (p={result.get('p_value', 'N/A'):.3e})"
                        )
                report_lines.append("")
            
            # Same row finger separation distances
            if 'separation_distances_same_row' in q4 and q4['separation_distances_same_row']:
                report_lines.append("Finger separation distance preferences (within same row):")
                for distance_name, result in q4['separation_distances_same_row'].items():
                    if result.get('n_comparisons', 0) > 0:
                        report_lines.append(
                            f"  {distance_name}: {result.get('proportion_chose_smaller', 'N/A'):.3f} "
                            f"prefer smaller distance (p={result.get('p_value', 'N/A'):.3e})"
                        )
                report_lines.append("")
            else:
                report_lines.extend([
                    "Finger separation distance preferences (within same row): No data available",
                    ""
                ])
            
            # Different row finger separation distances
            if 'separation_distances_diff_row' in q4 and q4['separation_distances_diff_row']:
                report_lines.append("Finger separation distance preferences (across different rows):")
                for distance_name, result in q4['separation_distances_diff_row'].items():
                    if result.get('n_comparisons', 0) > 0:
                        report_lines.append(
                            f"  {distance_name}: {result.get('proportion_chose_smaller', 'N/A'):.3f} "
                            f"prefer smaller distance (p={result.get('p_value', 'N/A'):.3e})"
                        )
                report_lines.append("")
            else:
                report_lines.extend([
                    "Finger separation distance preferences (across different rows): No data available",
                    ""
                ])
        
        # Question 5: Direction preferences
        if 'question_5' in results:
            report_lines.extend([
                "Question 5: Direction Preferences",
                "-------------------------------"
            ])
            
            q5 = results['question_5']
            
            if 'direction_preference' in q5:
                dir_result = q5['direction_preference']
                report_lines.extend([
                    f"Direction toward column 5 (low-to-high fingers):",
                    f"  Proportion choosing toward column 5: {dir_result.get('proportion_chose_toward_5', 'N/A'):.3f}",
                    f"  p-value: {dir_result.get('p_value', 'N/A'):.3e}",
                    f"  Significant: {'Yes' if dir_result.get('significant', False) else 'No'}",
                    ""
                ])
        
        # Multiple comparison correction summary
        if 'multiple_comparison_correction' in results:
            mcc = results['multiple_comparison_correction']
            report_lines.extend([
                "Multiple Comparison Correction",
                "----------------------------",
                f"Method: {mcc['method']}",
                f"Number of tests: {mcc['n_tests']}",
                f"Alpha level: {mcc['alpha_level']}",
                ""
            ])
            
            significant_tests = [test for test in mcc['test_results'] if test['significant_corrected']]
            report_lines.extend([
                f"Tests significant after correction: {len(significant_tests)}",
                ""
            ])
            
            if significant_tests:
                report_lines.append("Significant results after correction:")
                for test in significant_tests:
                    report_lines.append(f"  {test['test']}: p = {test['corrected_p']:.3e}")
                report_lines.append("")
        
        # Save report
        report_path = os.path.join(output_folder, 'keyboard_ergonomics_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Generate interpretation guide
        self._generate_interpretation_guide(results, output_folder)
        
        logger.info(f"Ergonomics analysis report saved to {report_path}")
        logger.info(f"Interpretation guide saved to {os.path.join(output_folder, 'ergonomics_interpretation.txt')}")
    
    def _generate_interpretation_guide(self, results: Dict[str, Any], output_folder: str) -> None:
        """Generate practical interpretation guide for keyboard design"""
        
        interpretation_lines = [
            "Keyboard Ergonomics Results - Design Interpretation Guide",
            "=======================================================\n",
            "This guide translates statistical findings into practical keyboard design principles.\n",
        ]
        
        # Extract key findings and translate to design principles
        q1 = results.get('question_1', {})
        q2 = results.get('question_2', {})
        q3 = results.get('question_3', {})
        q4 = results.get('question_4', {})
        q5 = results.get('question_5', {})
        
        # Design Principle 1: Row Optimization
        interpretation_lines.extend([
            "DESIGN PRINCIPLE 1: Row Optimization",
            "-----------------------------------"
        ])
        
        if 'home_vs_others' in q1:
            home_pref = q1['home_vs_others'].get('proportion_chose_home', 0)
            interpretation_lines.append(f"• HOME ROW IS KING: {home_pref:.1%} preference for home row keys")
            interpretation_lines.append("  → Place most frequent letters on home row (ASDF JKL;)")
        
        if 'finger_preferences' in q1:
            pinky_pref = q1['finger_preferences'].get('pinky', {}).get('proportion_chose_top', 1)
            if pinky_pref < 0.5:
                interpretation_lines.append(f"• PINKY PREFERS BOTTOM: Only {pinky_pref:.1%} prefer top row")
                interpretation_lines.append("  → When pinky must leave home row, use bottom row (Z, X) over top row (Q, W)")
        
        interpretation_lines.append("")
        
        # Design Principle 2: Movement Patterns  
        interpretation_lines.extend([
            "DESIGN PRINCIPLE 2: Movement Patterns",
            "------------------------------------"
        ])
        
        if 'same_vs_different_row' in q2:
            same_row_pref = q2['same_vs_different_row'].get('proportion_chose_same_row', 0)
            interpretation_lines.append(f"• MINIMIZE ROW JUMPING: {same_row_pref:.1%} prefer same-row bigrams")
            interpretation_lines.append("  → Optimize common letter pairs to stay within same row")
        
        if 'adjacent_vs_nonadjacent' in q2:
            adj_pref = q2['adjacent_vs_nonadjacent'].get('proportion_chose_adjacent', 0)
            interpretation_lines.append(f"• ADJACENT ROWS BETTER: {adj_pref:.1%} prefer adjacent over skip rows")
            interpretation_lines.append("  → When cross-row movement needed, prefer home↔top or home↔bottom over top↔bottom")
        
        interpretation_lines.append("")
        
        # Design Principle 3: Column Preferences
        interpretation_lines.extend([
            "DESIGN PRINCIPLE 3: Column Preferences",
            "-------------------------------------"
        ])
        
        # Analyze column preferences to determine ranking
        col_results = {}
        for test_name, result in q3.items():
            if isinstance(result, dict) and 'proportion_chose_higher' in result:
                if 'col5_vs_col4' in test_name:
                    col_results['4_vs_5'] = 1 - result['proportion_chose_higher']  # Proportion preferring col 4
                elif 'col4_vs_col3' in test_name:
                    col_results['3_vs_4'] = 1 - result['proportion_chose_higher']  # Proportion preferring col 3
                elif 'col3_vs_col2' in test_name:
                    col_results['3_vs_2'] = result['proportion_chose_higher']  # Proportion preferring col 3
                elif 'col2_vs_col1' in test_name:
                    col_results['2_vs_1'] = result['proportion_chose_higher']  # Proportion preferring col 2
        
        interpretation_lines.append("• COLUMN PREFERENCE ORDER (left hand):")
        interpretation_lines.append("  → Column 3 (middle finger) most preferred")
        interpretation_lines.append("  → Columns 2, 4 (ring, index) moderately preferred") 
        interpretation_lines.append("  → Columns 1, 5 (pinky positions) least preferred")
        interpretation_lines.append("  → Place frequent letters: D, F (best) > S, F (good) > A, G (acceptable)")
        interpretation_lines.append("")
        
        # Design Principle 4: Finger Coordination
        interpretation_lines.extend([
            "DESIGN PRINCIPLE 4: Finger Coordination", 
            "---------------------------------------"
        ])
        
        if 'adjacent_vs_remote_overall' in q4:
            adj_finger_pref = q4['adjacent_vs_remote_overall'].get('proportion_chose_adjacent', 0)
            interpretation_lines.append(f"• ADJACENT FINGERS PREFERRED: {adj_finger_pref:.1%} prefer adjacent finger movements")
            interpretation_lines.append("  → Favor sequences like: index→middle, middle→ring, ring→pinky")
            interpretation_lines.append("  → Avoid: pinky→index, pinky→middle stretches")
        
        interpretation_lines.append("")
        
        # Implementation Priority
        interpretation_lines.extend([
            "IMPLEMENTATION PRIORITY FOR LAYOUT OPTIMIZATION",
            "==============================================",
            "",
            "HIGH PRIORITY (Strong statistical evidence):",
            "1. Maximize home row usage for frequent letters",
            "2. Minimize cross-row bigram sequences", 
            "3. Favor column 3 (middle finger) for most frequent letters",
            "4. Use adjacent finger movements for common bigrams",
            "",
            "MEDIUM PRIORITY (Moderate evidence):",
            "5. When pinky leaves home row, prefer bottom over top",
            "6. When cross-row movement needed, use adjacent rows",
            "7. Avoid extreme columns (1, 5) for very frequent letters",
            "",
            "LOW PRIORITY (Weak evidence):",
            "8. Slight preference for inward finger movements",
            "",
            "PRACTICAL APPLICATION:",
            "• Use these principles as weighted objectives in multi-objective optimization",
            "• Weight high-priority principles more heavily than low-priority ones",
            "• Consider letter frequency when applying column preferences",
            "• Test final layouts with actual typing to validate theoretical predictions"
        ])
        
        # Save interpretation guide
        interpretation_path = os.path.join(output_folder, 'ergonomics_interpretation.txt')
        with open(interpretation_path, 'w') as f:
            f.write('\n'.join(interpretation_lines))


# Integration function for existing analyze_data.py
def add_ergonomics_analysis_to_main(analyzer_instance, data, output_folder, config):
    """
    Function to integrate keyboard ergonomics analysis into existing analyze_data.py main() function.
    
    Args:
        analyzer_instance: Instance of BigramAnalysis class
        data: Processed bigram data
        output_folder: Output directory 
        config: Configuration dictionary
    """
    
    # Check if ergonomics analysis is enabled
    if not config.get('analysis', {}).get('run_ergonomics_tests', False):
        return None
    
    logger.info("Running keyboard ergonomics analysis...")
    
    # Create ergonomics analyzer
    ergonomics_analyzer = KeyboardErgonomicsAnalysis(config)
    
    # Run analysis
    ergonomics_results = ergonomics_analyzer.run_all_ergonomics_tests(data, output_folder)
    
    logger.info("Keyboard ergonomics analysis completed!")
    
    return ergonomics_results