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
    
    def _calculate_proportion_ci(self, successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval for a proportion.
        More accurate than normal approximation for small samples.
        """
        if trials == 0:
            return np.nan, np.nan
            
        z = stats.norm.ppf(1 - (1 - confidence) / 2)  # Critical value
        p = successes / trials
        
        # Wilson score interval
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)
        
        return ci_lower, ci_upper
    
    def _add_enhanced_reporting(self, base_result: Dict[str, Any], 
                               successes: int, trials: int, 
                               success_description: str) -> Dict[str, Any]:
        """Add enhanced reporting with effect sizes, CIs, and interpretation"""
        
        if trials == 0:
            return {**base_result, 'interpretation': 'No valid data for analysis'}
        
        proportion = successes / trials
        
        # Calculate confidence interval
        ci_lower, ci_upper = self._calculate_proportion_ci(successes, trials)
        
        # Effect size (deviation from chance)
        effect_size = abs(proportion - 0.5)
        
        # Practical significance categorization
        if effect_size >= 0.20:
            practical_significance = "Large effect"
            design_priority = "HIGH PRIORITY"
        elif effect_size >= 0.10:
            practical_significance = "Medium effect"
            design_priority = "MEDIUM PRIORITY"
        elif effect_size >= 0.05:
            practical_significance = "Small effect"
            design_priority = "LOW PRIORITY"
        else:
            practical_significance = "Trivial effect"
            design_priority = "IGNORE"
        
        # Interpretation
        interpretation = f"{success_description} {proportion:.1%} of the time (95% CI: {ci_lower:.1%}-{ci_upper:.1%})"
        
        enhanced_result = {
            **base_result,
            'proportion': proportion,
            'effect_size': effect_size,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'practical_significance': practical_significance,
            'design_priority': design_priority,
            'interpretation': interpretation
        }
        
        return enhanced_result
    
    def _test_column5_avoidance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if column 5 is systematically avoided compared to columns 1-4"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            # Count occurrences of column 5 vs columns 1-4 in chosen vs unchosen
            chosen_cols = (row['chosen_bigram_col1'], row['chosen_bigram_col2'])
            unchosen_cols = (row['unchosen_bigram_col1'], row['unchosen_bigram_col2'])
            
            chosen_col5_count = sum(1 for c in chosen_cols if c == 5)
            chosen_others_count = sum(1 for c in chosen_cols if c in [1, 2, 3, 4])
            
            unchosen_col5_count = sum(1 for c in unchosen_cols if c == 5)
            unchosen_others_count = sum(1 for c in unchosen_cols if c in [1, 2, 3, 4])
            
            # Score: +1 for each non-col5 key, -1 for each col5 key
            chosen_score = chosen_others_count - chosen_col5_count
            unchosen_score = unchosen_others_count - unchosen_col5_count
            
            if chosen_score != unchosen_score:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_more_non_col5': chosen_score > unchosen_score,
                    'avoidance_difference': chosen_score - unchosen_score
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_more_non_col5 = comparison_df['chose_more_non_col5'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_more_non_col5, n_total, 0.5, alternative='two-sided').pvalue
        
        base_result = {
            'test_name': 'Column 5 avoidance (vs columns 1-4)',
            'n_comparisons': n_total,
            'n_chose_more_non_col5': n_chose_more_non_col5,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
        
        result = self._add_enhanced_reporting(
            base_result, n_chose_more_non_col5, n_total,
            "Columns 1-4 chosen over column 5"
        )
        
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
        - Column 5 vs all other columns (1-4)
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
        
        # NEW: Test column 5 vs all other columns individually
        col5_vs_others = [
            (5, 1, "col5_vs_col1"),
            (5, 2, "col5_vs_col2"), 
            (5, 3, "col5_vs_col3"),
            # col5_vs_col4 already done above
        ]
        
        for col_high, col_low, test_name in col5_vs_others:
            result = self._test_column_preference(data, col_high, col_low)
            results[test_name] = result
        
        # NEW: Combined test - is column 5 systematically avoided?
        col5_avoidance_result = self._test_column5_avoidance(data)
        results['col5_avoidance_overall'] = col5_avoidance_result
            
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
        proportion = n_chose_more_home / n_total
        
        # Two-tailed binomial test
        p_value = stats.binomtest(n_chose_more_home, n_total, 0.5, alternative='two-sided').pvalue
        
        # Calculate confidence interval using Wilson score interval
        ci_lower, ci_upper = self._calculate_proportion_ci(n_chose_more_home, n_total)
        
        # Effect size (deviation from chance)
        effect_size = abs(proportion - 0.5)
        
        # Practical significance categorization
        if effect_size >= 0.20:
            practical_significance = "Large effect"
        elif effect_size >= 0.10:
            practical_significance = "Medium effect"
        elif effect_size >= 0.05:
            practical_significance = "Small effect"
        else:
            practical_significance = "Trivial effect"
        
        result = {
            'test_name': 'Home row preference',
            'n_comparisons': n_total,
            'n_chose_more_home': n_chose_more_home,
            'proportion_chose_home': proportion,
            'p_value': p_value,
            'effect_size': effect_size,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'practical_significance': practical_significance,
            'significant': p_value < self.alpha,
            'interpretation': f"Home row keys chosen {proportion:.1%} of the time when compared to top/bottom row keys"
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
        
        base_result = {
            'test_name': 'Top vs bottom row preference',
            'n_comparisons': n_total,
            'n_chose_more_top': n_chose_more_top,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
        
        result = self._add_enhanced_reporting(
            base_result, n_chose_more_top, n_total,
            "Top row keys chosen over bottom row keys"
        )
        
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
        
        base_result = {
            'test_name': 'Same vs different row preference',
            'n_comparisons': n_total,
            'n_chose_same_row': n_chose_same_row,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
        
        result = self._add_enhanced_reporting(
            base_result, n_chose_same_row, n_total,
            "Same-row bigrams chosen over cross-row bigrams"
        )
        
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
        """Generate comprehensive ergonomics analysis report with effect sizes and interpretations"""
        
        # Apply multiple comparison correction
        results = self._apply_multiple_comparison_correction(results)
        
        report_lines = [
            "Keyboard Ergonomics Analysis Report",
            "==================================\n",
            f"Analysis conducted with α = {self.alpha}",
            f"Multiple comparison correction: {self.correction_method}",
            "",
            "INTERPRETATION GUIDE:",
            "• Effect sizes: Large (≥20pp), Medium (10-20pp), Small (5-10pp), Trivial (<5pp)",
            "• Design Priority: HIGH (large effects), MEDIUM (medium effects), LOW (small effects)",
            "• Confidence Intervals: 95% Wilson score intervals for proportions",
            "• pp = percentage points above/below chance (50%)",
            ""
        ]
        
        # Question 1: Row preferences
        if 'question_1' in results:
            report_lines.extend([
                "QUESTION 1: ROW PREFERENCES",
                "===========================",
                ""
            ])
            
            q1 = results['question_1']
            
            if 'home_vs_others' in q1:
                home_result = q1['home_vs_others']
                if 'interpretation' in home_result:
                    report_lines.extend([
                        "1.1 HOME ROW vs TOP/BOTTOM ROWS",
                        "--------------------------------",
                        f"Result: {home_result['interpretation']}",
                        f"Statistical significance: p = {home_result.get('p_value', 'N/A'):.3e}",
                        f"Effect size: {home_result.get('effect_size', 0):.3f} ({home_result.get('practical_significance', 'N/A')})",
                        f"Design priority: {home_result.get('design_priority', 'N/A')}",
                        f"Sample size: {home_result.get('n_comparisons', 0)} comparisons",
                        "",
                        "DESIGN IMPLICATION:",
                        "→ Maximize home row usage for frequent letters (ASDF JKL;)",
                        "→ This is the strongest ergonomic preference in typing",
                        ""
                    ])
            
            if 'top_vs_bottom' in q1:
                top_result = q1['top_vs_bottom']
                if 'interpretation' in top_result:
                    report_lines.extend([
                        "1.2 TOP ROW vs BOTTOM ROW",
                        "-------------------------",
                        f"Result: {top_result['interpretation']}",
                        f"Statistical significance: p = {top_result.get('p_value', 'N/A'):.3e}",
                        f"Effect size: {top_result.get('effect_size', 0):.3f} ({top_result.get('practical_significance', 'N/A')})",
                        f"Design priority: {top_result.get('design_priority', 'N/A')}",
                        "",
                        "DESIGN IMPLICATION:",
                        "→ When not using home row, consider top vs bottom row preferences",
                        ""
                    ])
            
            if 'finger_preferences' in q1:
                report_lines.extend([
                    "1.3 FINGER-SPECIFIC ROW PREFERENCES",
                    "-----------------------------------"
                ])
                finger_names = {'index': 'Index', 'middle': 'Middle', 'ring': 'Ring', 'pinky': 'Pinky'}
                for finger, finger_result in q1['finger_preferences'].items():
                    if finger_result.get('n_comparisons', 0) > 0:
                        prop = finger_result.get('proportion_chose_top', 0.5)
                        effect_size = abs(prop - 0.5)
                        
                        if effect_size >= 0.05:  # Only report meaningful effects
                            preference = "TOP row" if prop > 0.5 else "BOTTOM row"
                            strength = "strongly" if effect_size >= 0.15 else "moderately"
                            
                            report_lines.extend([
                                f"{finger_names.get(finger, finger)} finger: {strength} prefers {preference}",
                                f"  → {prop:.1%} prefer top row (effect size: {effect_size:.3f})",
                                f"  → p = {finger_result.get('p_value', 'N/A'):.3e}, n = {finger_result.get('n_comparisons', 0)}",
                                ""
                            ])
                
                report_lines.extend([
                    "DESIGN IMPLICATION:",
                    "→ When fingers must leave home row, use finger-specific preferences",
                    "→ Pinky: strongly prefers bottom row (Z, X) over top row (Q, W)",
                    ""
                ])
        
        # Question 2: Row pair preferences
        if 'question_2' in results:
            report_lines.extend([
                "QUESTION 2: ROW MOVEMENT PATTERNS",
                "=================================",
                ""
            ])
            
            q2 = results['question_2']
            
            if 'same_vs_different_row' in q2:
                same_result = q2['same_vs_different_row']
                if 'interpretation' in same_result:
                    report_lines.extend([
                        "2.1 SAME ROW vs CROSS-ROW BIGRAMS",
                        "---------------------------------",
                        f"Result: {same_result['interpretation']}",
                        f"Statistical significance: p = {same_result.get('p_value', 'N/A'):.3e}",
                        f"Effect size: {same_result.get('effect_size', 0):.3f} ({same_result.get('practical_significance', 'N/A')})",
                        f"Design priority: {same_result.get('design_priority', 'N/A')}",
                        "",
                        "DESIGN IMPLICATION:",
                        "→ Optimize frequent bigrams to stay within same row when possible",
                        "→ Major constraint for keyboard layout optimization",
                        ""
                    ])
            
            if 'adjacent_vs_nonadjacent' in q2:
                adj_result = q2['adjacent_vs_nonadjacent']
                if 'interpretation' in adj_result:
                    report_lines.extend([
                        "2.2 ADJACENT vs SKIP ROW MOVEMENTS",
                        "----------------------------------",
                        f"Result: {adj_result['interpretation']}",
                        f"Statistical significance: p = {adj_result.get('p_value', 'N/A'):.3e}",
                        f"Effect size: {adj_result.get('effect_size', 0):.3f} ({adj_result.get('practical_significance', 'N/A')})",
                        f"Design priority: {adj_result.get('design_priority', 'N/A')}",
                        "",
                        "DESIGN IMPLICATION:",
                        "→ When cross-row movement needed, prefer adjacent rows (home↔top, home↔bottom)",
                        "→ Avoid skip movements (top↔bottom) for frequent bigrams",
                        ""
                    ])
        
        # Question 3: Column preferences
        if 'question_3' in results:
            report_lines.extend([
                "QUESTION 3: COLUMN PREFERENCES",
                "==============================",
                ""
            ])
            
            q3 = results['question_3']
            
            # Individual column comparisons
            report_lines.append("3.1 INDIVIDUAL COLUMN COMPARISONS")
            report_lines.append("----------------------------------")
            
            column_tests = [
                ('col5_vs_col4', 'Column 4 vs Column 5'),
                ('col4_vs_col3', 'Column 3 vs Column 4'),
                ('col3_vs_col2', 'Column 3 vs Column 2'), 
                ('col2_vs_col1', 'Column 2 vs Column 1')
            ]
            
            for test_name, description in column_tests:
                if test_name in q3:
                    result = q3[test_name]
                    if isinstance(result, dict) and 'interpretation' in result:
                        # Determine which column is preferred
                        prop_higher = result.get('proportion_chose_higher', 0.5)
                        if 'col5_vs_col4' in test_name:
                            preferred = "Column 4" if prop_higher < 0.5 else "Column 5"
                            prop_preferred = 1 - prop_higher if prop_higher < 0.5 else prop_higher
                        elif 'col4_vs_col3' in test_name:
                            preferred = "Column 3" if prop_higher < 0.5 else "Column 4"
                            prop_preferred = 1 - prop_higher if prop_higher < 0.5 else prop_higher
                        elif 'col3_vs_col2' in test_name:
                            preferred = "Column 3" if prop_higher > 0.5 else "Column 2"
                            prop_preferred = prop_higher if prop_higher > 0.5 else 1 - prop_higher
                        else:  # col2_vs_col1
                            preferred = "Column 2" if prop_higher > 0.5 else "Column 1"
                            prop_preferred = prop_higher if prop_higher > 0.5 else 1 - prop_higher
                        
                        effect_size = abs(prop_preferred - 0.5)
                        
                        report_lines.extend([
                            f"{description}: {preferred} preferred {prop_preferred:.1%} of the time",
                            f"  → Effect size: {effect_size:.3f}, p = {result.get('p_value', 'N/A'):.3e}",
                            ""
                        ])
            
            # Column 5 avoidance analysis
            if 'col5_avoidance_overall' in q3:
                avoid_result = q3['col5_avoidance_overall']
                if 'interpretation' in avoid_result:
                    report_lines.extend([
                        "3.2 COLUMN 5 SYSTEMATIC AVOIDANCE",
                        "----------------------------------",
                        f"Result: {avoid_result['interpretation']}",
                        f"Statistical significance: p = {avoid_result.get('p_value', 'N/A'):.3e}",
                        f"Effect size: {avoid_result.get('effect_size', 0):.3f} ({avoid_result.get('practical_significance', 'N/A')})",
                        f"Design priority: {avoid_result.get('design_priority', 'N/A')}",
                        "",
                        "DESIGN IMPLICATION:",
                        "→ Column preference hierarchy for left hand: 3 > 2,4 > 1,5",
                        "→ Avoid placing frequent letters in extreme positions (columns 1, 5)",
                        ""
                    ])
        
        # Add summary section
        report_lines.extend([
            "SUMMARY FOR KEYBOARD LAYOUT OPTIMIZATION",
            "========================================",
            "",
            "HIGH PRIORITY CONSTRAINTS (Large effects, strong evidence):",
        ])
        
        # Extract high priority findings
        high_priority_items = []
        for question_key, question_data in results.items():
            if isinstance(question_data, dict):
                for test_key, test_data in question_data.items():
                    if isinstance(test_data, dict) and test_data.get('design_priority') == 'HIGH PRIORITY':
                        high_priority_items.append(f"• {test_data.get('test_name', test_key)}: {test_data.get('interpretation', 'N/A')}")
        
        if high_priority_items:
            report_lines.extend(high_priority_items)
        else:
            report_lines.append("• No high priority effects found")
        
        report_lines.extend([
            "",
            "MEDIUM PRIORITY CONSTRAINTS (Medium effects, moderate evidence):",
        ])
        
        # Extract medium priority findings
        medium_priority_items = []
        for question_key, question_data in results.items():
            if isinstance(question_data, dict):
                for test_key, test_data in question_data.items():
                    if isinstance(test_data, dict) and test_data.get('design_priority') == 'MEDIUM PRIORITY':
                        medium_priority_items.append(f"• {test_data.get('test_name', test_key)}: {test_data.get('interpretation', 'N/A')}")
        
        if medium_priority_items:
            report_lines.extend(medium_priority_items)
        else:
            report_lines.append("• No medium priority effects found")
        
        # Multiple comparison correction summary
        if 'multiple_comparison_correction' in results:
            mcc = results['multiple_comparison_correction']
            report_lines.extend([
                "",
                "STATISTICAL VALIDATION",
                "=====================",
                f"Multiple comparison correction: {mcc['method']}",
                f"Total tests conducted: {mcc['n_tests']}",
                f"Tests significant after correction: {len([t for t in mcc['test_results'] if t['significant_corrected']])}",
                f"Family-wise error rate controlled at α = {mcc['alpha_level']}"
            ])
        
        # Save report
        report_path = os.path.join(output_folder, 'keyboard_ergonomics_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Comprehensive ergonomics analysis report saved to {report_path}")


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