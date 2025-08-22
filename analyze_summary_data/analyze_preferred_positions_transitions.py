"""
Combined Keyboard Preference Analysis

This script provides comprehensive keyboard preference analysis using Bradley-Terry models:

1. EXPLORATORY ANALYSIS:
   - Individual key preferences with confidence intervals
   - Transition type preferences and categorization
   - Comprehensive visualizations and statistical tables
   - Bootstrap validation and effect size analysis

2. FOCUSED HYPOTHESIS TESTING:
   - 20 pre-specified ergonomic hypotheses
   - Finger separation, movement, direction, and finger preference effects
   - Global multiple testing correction (FDR)
   - Column-specific row preferences

Features:
- Dual analysis approach (confirmatory + exploratory)
- Rigorous statistical testing with proper corrections
- Comprehensive reporting and visualization
- Cross-validation between approaches

Usage:
    python combined_analyzer.py --data data/filtered_data.csv --output results/ --config config.yaml
"""

import os
import argparse
import logging
import yaml
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from itertools import combinations
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class KeyPosition:
    """Represents a key's position on the keyboard."""
    key: str
    row: int      # 1=upper, 2=home, 3=lower
    column: int   # 1=leftmost, 5=index center
    finger: int   # 1=pinky, 2=ring, 3=middle, 4=index

class BradleyTerryModel:
    """Bradley-Terry model for ranking items from pairwise comparison data."""
    
    def __init__(self, items: List[str], config: Dict[str, Any]):
        self.items = list(items)
        self.n_items = len(items)
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.strengths = None
        self.fitted = False
        self.config = config
    
    def fit(self, pairwise_data: Dict[Tuple[str, str], Dict[str, int]]) -> None:
        """Fit Bradley-Terry model to pairwise comparison data."""
        wins = np.zeros((self.n_items, self.n_items))
        totals = np.zeros((self.n_items, self.n_items))
        
        pair_count = 0
        for (item1, item2), data in pairwise_data.items():
            if item1 in self.item_to_idx and item2 in self.item_to_idx:
                i, j = self.item_to_idx[item1], self.item_to_idx[item2]
                
                try:
                    wins_item1 = int(data['wins_item1'])
                    total = int(data['total'])
                    
                    wins[i, j] = wins_item1
                    wins[j, i] = total - wins_item1
                    totals[i, j] = totals[j, i] = total
                    pair_count += 1
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid data for pair ({item1}, {item2}): {data}")
                    continue
        
        self.strengths = self._fit_ml(wins, totals)
        self.fitted = True
    
    def _fit_ml(self, wins: np.ndarray, totals: np.ndarray) -> np.ndarray:
        """Fit Bradley-Terry model using maximum likelihood estimation."""
        
        def negative_log_likelihood(strengths):
            reg = float(self.config.get('regularization', 1e-10))
            strengths = np.clip(strengths, -10, 10)
            ll = 0
            for i in range(self.n_items):
                for j in range(i + 1, self.n_items):
                    if totals[i, j] > 0:
                        p_ij = np.exp(strengths[i]) / (np.exp(strengths[i]) + np.exp(strengths[j]))
                        p_ij = np.clip(p_ij, reg, 1.0 - reg)
                        ll += wins[i, j] * np.log(p_ij) + wins[j, i] * np.log(1 - p_ij)
            return -ll
        
        initial_strengths = np.zeros(self.n_items)
        constraint = {'type': 'eq', 'fun': lambda x: x[0]}
        
        max_iter = int(self.config.get('max_iterations', 1000))
        result = minimize(
            negative_log_likelihood,
            initial_strengths,
            method='SLSQP',
            constraints=constraint,
            options={'maxiter': max_iter}
        )
        
        if not result.success:
            logger.warning("Bradley-Terry optimization did not converge")
        
        return result.x
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get items ranked by strength (highest to lowest)."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting rankings")
        
        ranked_indices = np.argsort(self.strengths)[::-1]
        return [(self.items[i], self.strengths[i]) for i in ranked_indices]

class FocusedHypothesisAnalyzer:
    """Analyzer for testing 20 specific keyboard ergonomic hypotheses."""
    
    def __init__(self, config: Dict[str, Any], key_positions: Dict[str, KeyPosition]):
        self.config = config
        self.key_positions = key_positions
        
        # Define the 20 focused hypotheses
        self.hypotheses = {
            # Finger separation effects (5 tests)
            'same_row_finger_sep_1v2': {'description': 'Same-row: 1 vs 2 fingers apart', 'category': 'same_row_finger_separation', 'values': ['1', '2']},
            'same_row_finger_sep_2v3': {'description': 'Same-row: 2 vs 3 fingers apart', 'category': 'same_row_finger_separation', 'values': ['2', '3']},
            'cross_row_finger_sep_1v2': {'description': 'Cross-row: 1 vs 2 fingers apart', 'category': 'cross_row_finger_separation', 'values': ['1', '2']},
            'cross_row_finger_sep_2v3': {'description': 'Cross-row: 2 vs 3 fingers apart', 'category': 'cross_row_finger_separation', 'values': ['2', '3']},
            'cross_row_same_vs_diff': {'description': 'Cross-row: same vs different finger', 'category': 'cross_row_same_finger', 'values': ['True', 'False']},
            
            # Movement effects (2 tests)
            'home_keys_2v1': {'description': 'Home keys: 2 vs 1', 'category': 'home_key_count', 'values': ['2', '1']},
            'home_keys_1v0': {'description': 'Home keys: 1 vs 0', 'category': 'home_key_count', 'values': ['1', '0']},
            
            # Vertical separation effects (2 tests)
            'row_sep_0v1': {'description': 'Row separation: 0 vs 1', 'category': 'row_separation', 'values': ['0', '1']},
            'row_sep_1v2': {'description': 'Row separation: 1 vs 2', 'category': 'row_separation', 'values': ['1', '2']},
            
            # Horizontal reach effects (2 tests)
            'column5_0v1': {'description': 'Column 5: 0 vs 1 key', 'category': 'column5_count', 'values': ['0', '1']},
            'column5_1v2': {'description': 'Column 5: 1 vs 2 keys', 'category': 'column5_count', 'values': ['1', '2']},
            
            # Finger preferences (3 tests)
            'finger_f4_vs_f3': {'description': 'Finger: F4 vs F3', 'category': 'dominant_finger', 'values': ['4', '3']},
            'finger_f3_vs_f2': {'description': 'Finger: F3 vs F2', 'category': 'dominant_finger', 'values': ['3', '2']},
            'finger_f2_vs_f1': {'description': 'Finger: F2 vs F1', 'category': 'dominant_finger', 'values': ['2', '1']},
            
            # Direction effects (2 tests)
            'same_row_direction': {'description': 'Same-row: inner vs outer roll', 'category': 'same_row_direction', 'values': ['inner_roll', 'outer_roll']},
            'cross_row_direction': {'description': 'Cross-row: inner vs outer roll', 'category': 'cross_row_direction', 'values': ['inner_roll_cross', 'outer_roll_cross']},
            
            # Column-specific row preferences (4 tests) 
            'column1_upper_vs_lower': {'description': 'Column 1: Q vs Z (F1)', 'category': 'column1_row_pref', 'values': ['1', '3']},
            'column2_upper_vs_lower': {'description': 'Column 2: W vs X (F2)', 'category': 'column2_row_pref', 'values': ['1', '3']},
            'column3_upper_vs_lower': {'description': 'Column 3: E vs C (F3)', 'category': 'column3_row_pref', 'values': ['1', '3']},
            'column4_upper_vs_lower': {'description': 'Column 4: R vs V (F4)', 'category': 'column4_row_pref', 'values': ['1', '3']},
        }
        
        logger.info(f"Initialized focused hypothesis testing with {len(self.hypotheses)} hypotheses")
    
    def analyze_hypotheses(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze all 20 focused hypotheses."""
        logger.info("Running focused hypothesis testing (20 hypotheses)...")
        
        # Classify all bigrams
        bigram_classifications = self._classify_bigrams_for_hypotheses(data)
        
        # Test each hypothesis
        hypothesis_results = {}
        all_p_values = []
        p_value_info = []
        
        for hyp_name, hyp_config in self.hypotheses.items():
            logger.info(f"Testing {hyp_name}: {hyp_config['description']}")
            
            # Extract comparisons for this hypothesis
            comparisons = self._extract_comparisons_for_hypothesis(
                data, bigram_classifications, hyp_config
            )
            
            if not comparisons:
                logger.warning(f"No comparisons found for {hyp_name}")
                hypothesis_results[hyp_name] = {'error': 'No comparisons found'}
                continue
            
            # Calculate statistics for this specific comparison
            pairwise_stats = self._calculate_hypothesis_stats(comparisons, hyp_config)
            
            hypothesis_results[hyp_name] = {
                'description': hyp_config['description'],
                'comparison_values': hyp_config['values'],
                'statistics': pairwise_stats,
                'n_comparisons': sum(comp['total'] for comp in comparisons.values())
            }
            
            # Collect p-values for global correction
            if pairwise_stats and not np.isnan(pairwise_stats.get('p_value', np.nan)):
                all_p_values.append(pairwise_stats['p_value'])
                p_value_info.append({
                    'hypothesis': hyp_name,
                    'comparison': f"{hyp_config['values'][0]} vs {hyp_config['values'][1]}"
                })
        
        # Apply global multiple testing correction
        hypothesis_results = self._apply_global_correction(
            hypothesis_results, all_p_values, p_value_info
        )
        
        # Generate summary
        summary = self._generate_hypothesis_summary(hypothesis_results)
        
        return {
            'hypothesis_results': hypothesis_results,
            'summary': summary,
            'total_hypotheses': len(self.hypotheses),
            'total_comparisons': len(all_p_values)
        }
    
    def _classify_bigrams_for_hypotheses(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
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
        """Classify a single bigram according to all hypothesis dimensions."""
        
        # Basic measurements
        finger_separation = abs(pos1.finger - pos2.finger)
        row_separation = abs(pos1.row - pos2.row)
        
        # Home keys (finger-column home keys: A,S,D,F)
        home_keys = {'a', 's', 'd', 'f'}
        home_key_count = sum(1 for pos in [pos1, pos2] if pos.key in home_keys)
        
        # Column 5 keys (index finger extended position: T,G,B)
        column5_keys = {'t', 'g', 'b'}
        column5_count = sum(1 for pos in [pos1, pos2] if pos.key in column5_keys)
        
        # Direction calculation
        if row_separation == 0:  # Same row
            if pos2.finger > pos1.finger:
                direction = 'inner_roll'
            elif pos2.finger < pos1.finger:
                direction = 'outer_roll'
            else:
                direction = 'same_finger'
        else:  # Cross row
            if pos2.finger > pos1.finger:
                direction = 'inner_roll_cross'
            elif pos2.finger < pos1.finger:
                direction = 'outer_roll_cross'
            else:
                direction = 'same_finger_cross'
        
        # Dominant finger (higher finger number for finger preference tests)
        dominant_finger = max(pos1.finger, pos2.finger)
        
        # Column-specific row preferences
        column_row_prefs = {}
        for column in [1, 2, 3, 4]:
            if pos1.column == column and pos2.column == column:
                # Both keys in same column, compare rows
                column_row_prefs[f'column{column}_row_pref'] = f"{pos1.row}" if pos1.row != pos2.row else None
            else:
                column_row_prefs[f'column{column}_row_pref'] = None
        
        return {
            # Finger separation hypotheses
            'same_row_finger_separation': str(finger_separation) if row_separation == 0 else None,
            'cross_row_finger_separation': str(finger_separation) if row_separation > 0 else None,
            'cross_row_same_finger': str(finger_separation == 0) if row_separation > 0 else None,
            
            # Movement hypothesis
            'home_key_count': str(home_key_count),
            
            # Vertical separation hypothesis
            'row_separation': str(row_separation),
            
            # Horizontal reach hypothesis
            'column5_count': str(column5_count),
            
            # Finger preference hypothesis
            'dominant_finger': str(dominant_finger) if finger_separation > 0 else None,
            
            # Direction hypotheses
            'same_row_direction': direction if row_separation == 0 and direction != 'same_finger' else None,
            'cross_row_direction': direction if row_separation > 0 and direction != 'same_finger_cross' else None,
            
            # Column-specific row preferences
            **column_row_prefs
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
    
    def _calculate_hypothesis_stats(self, comparisons: Dict[Tuple[str, str], Dict[str, int]], 
                                  hyp_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for a single hypothesis test."""
        
        if not comparisons:
            return {}
        
        # Should only be one comparison for focused hypotheses
        if len(comparisons) != 1:
            logger.warning(f"Expected 1 comparison for {hyp_config['description']}, got {len(comparisons)}")
        
        # Get the single comparison
        (val1, val2), data = list(comparisons.items())[0]
        wins1 = data['wins_item1']
        total = data['total']
        proportion1 = wins1 / total if total > 0 else 0.5
        
        # Effect size (deviation from chance)
        effect_size = abs(proportion1 - 0.5)
        
        # Statistical significance (binomial test)
        min_comparisons = self.config.get('min_transition_comparisons', 10)
        if total >= min_comparisons:
            binom_result = stats.binomtest(wins1, total, 0.5, alternative='two-sided')
            p_value = binom_result.pvalue
        else:
            p_value = np.nan
        
        # Practical significance
        thresholds = self.config.get('transition_effect_thresholds', {
            'negligible': 0.05, 'small': 0.15, 'medium': 0.30
        })
        
        if effect_size >= thresholds.get('medium', 0.30):
            practical_sig = 'large'
        elif effect_size >= thresholds.get('small', 0.15):
            practical_sig = 'medium'
        elif effect_size >= thresholds.get('negligible', 0.05):
            practical_sig = 'small'
        else:
            practical_sig = 'negligible'
        
        return {
            'values_compared': (val1, val2),
            'proportion_val1_wins': proportion1,
            'effect_size': effect_size,
            'p_value': p_value,
            'n_comparisons': total,
            'practical_significance': practical_sig
        }
    
    def _apply_global_correction(self, hypothesis_results: Dict[str, Any], 
                               all_p_values: List[float], 
                               p_value_info: List[Dict]) -> Dict[str, Any]:
        """Apply global multiple testing correction across all 20 hypotheses."""
        
        if not all_p_values:
            return hypothesis_results
        
        try:
            from statsmodels.stats.multitest import multipletests
            alpha_level = self.config.get('alpha_level', 0.05)
            correction_method = self.config.get('correction_method', 'fdr_bh')
            
            _, p_corrected, _, _ = multipletests(
                all_p_values, method=correction_method, alpha=alpha_level
            )
            
            # Update results with corrected p-values
            for i, p_info in enumerate(p_value_info):
                hyp_name = p_info['hypothesis']
                if hyp_name in hypothesis_results and 'statistics' in hypothesis_results[hyp_name]:
                    stats = hypothesis_results[hyp_name]['statistics']
                    stats['p_value_corrected'] = p_corrected[i]
                    stats['significant_corrected'] = p_corrected[i] < alpha_level
            
            logger.info(f"Applied {correction_method} correction to {len(all_p_values)} hypotheses")
            
        except ImportError:
            logger.warning("statsmodels not available for multiple comparison correction")
        
        return hypothesis_results
    
    def _generate_hypothesis_summary(self, hypothesis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of hypothesis test results."""
        
        summary = {
            'significant_results': [],
            'large_effects': [],
            'hypothesis_overview': {}
        }
        
        for hyp_name, result in hypothesis_results.items():
            if 'statistics' not in result:
                continue
            
            stats = result['statistics']
            
            # Count significant results
            if stats.get('significant_corrected', False):
                summary['significant_results'].append({
                    'hypothesis': hyp_name,
                    'description': result['description'],
                    'effect_size': stats['effect_size'],
                    'p_value_corrected': stats['p_value_corrected'],
                    'values_compared': stats['values_compared'],
                    'proportion': stats['proportion_val1_wins']
                })
            
            # Count large effects
            if stats.get('practical_significance') == 'large':
                summary['large_effects'].append({
                    'hypothesis': hyp_name,
                    'description': result['description'],
                    'effect_size': stats['effect_size'],
                    'values_compared': stats['values_compared'],
                    'proportion': stats['proportion_val1_wins']
                })
            
            # Overview
            summary['hypothesis_overview'][hyp_name] = {
                'description': result['description'],
                'significant': stats.get('significant_corrected', False),
                'effect_size': stats['effect_size'],
                'practical_significance': stats['practical_significance'],
                'n_comparisons': stats['n_comparisons']
            }
        
        # Sort by effect size
        summary['significant_results'].sort(key=lambda x: x['effect_size'], reverse=True)
        summary['large_effects'].sort(key=lambda x: x['effect_size'], reverse=True)
        
        return summary

class PreferenceAnalyzer:
    """Main class for comprehensive keyboard preference analysis (exploratory + focused)."""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Define keyboard layout
        self.key_positions = self._define_keyboard_layout()
        self.left_hand_keys = set(self.key_positions.keys())
        
        # Initialize focused hypothesis analyzer
        self.focused_analyzer = FocusedHypothesisAnalyzer(self.config, self.key_positions)
        
        # Store data for access in other methods
        self.data = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config = config.get('position_transition_analysis', {})
            
            # Ensure numeric values are properly typed
            numeric_fields = [
                'alpha_level', 'bootstrap_iterations', 'confidence_level', 
                'max_iterations', 'convergence_tolerance', 'regularization',
                'min_key_comparisons', 'min_transition_comparisons', 'figure_dpi'
            ]
            
            for field in numeric_fields:
                if field in config:
                    try:
                        if field in ['max_iterations', 'min_key_comparisons', 'min_transition_comparisons', 
                                   'bootstrap_iterations', 'figure_dpi']:
                            config[field] = int(config[field])
                        else:
                            config[field] = float(config[field])
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert config field '{field}' to number: {config[field]}")
            
            # Ensure threshold dictionaries have numeric values
            for threshold_key in ['key_effect_thresholds', 'transition_effect_thresholds']:
                if threshold_key in config and isinstance(config[threshold_key], dict):
                    for k, v in config[threshold_key].items():
                        try:
                            config[threshold_key][k] = float(v)
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert threshold '{k}' to float: {v}")
            
            return config
        else:
            # Default configuration if no file provided
            return {
                'alpha_level': 0.05,
                'correction_method': 'fdr_bh',
                'bootstrap_iterations': 1000,
                'confidence_level': 0.95,
                'max_iterations': 1000,
                'convergence_tolerance': 1e-6,
                'regularization': 1e-10,
                'min_key_comparisons': 5,
                'min_transition_comparisons': 10,
                'key_effect_thresholds': {
                    'negligible': 0.05,
                    'small': 0.15,
                    'medium': 0.30
                },
                'transition_effect_thresholds': {
                    'negligible': 0.03,
                    'small': 0.10,
                    'medium': 0.25
                },
                'confidence_criteria': {
                    'definitive': {'min_significant_pairs': 0.70, 'min_large_effects': 0.50},
                    'probable': {'min_significant_pairs': 0.50, 'min_large_effects': 0.30},
                    'suggestive': {'min_significant_pairs': 0.30, 'min_large_effects': 0.20}
                },
                'analyze_keys': 'left_hand_15',
                'generate_visualizations': True,
                'figure_dpi': 300,
                'heatmap_colormap': 'RdYlGn'
            }
    
    def _define_keyboard_layout(self) -> Dict[str, KeyPosition]:
        """Define keyboard layout based on configuration."""
        analyze_keys = self.config.get('analyze_keys', 'left_hand_15')
        
        if analyze_keys == 'left_hand_15':
            return {
                # Upper row (row 1)
                'q': KeyPosition('q', 1, 1, 1),
                'w': KeyPosition('w', 1, 2, 2),
                'e': KeyPosition('e', 1, 3, 3),
                'r': KeyPosition('r', 1, 4, 4),
                't': KeyPosition('t', 1, 5, 4),
                
                # Middle/home row (row 2)
                'a': KeyPosition('a', 2, 1, 1),
                's': KeyPosition('s', 2, 2, 2),
                'd': KeyPosition('d', 2, 3, 3),
                'f': KeyPosition('f', 2, 4, 4),
                'g': KeyPosition('g', 2, 5, 4),
                
                # Lower row (row 3)
                'z': KeyPosition('z', 3, 1, 1),
                'x': KeyPosition('x', 3, 2, 2),
                'c': KeyPosition('c', 3, 3, 3),
                'v': KeyPosition('v', 3, 4, 4),
                'b': KeyPosition('b', 3, 5, 4)
            }
        elif analyze_keys == 'home_block_12':
            return {
                # Upper row
                'q': KeyPosition('q', 1, 1, 1),
                'w': KeyPosition('w', 1, 2, 2),
                'e': KeyPosition('e', 1, 3, 3),
                'r': KeyPosition('r', 1, 4, 4),
                
                # Home row
                'a': KeyPosition('a', 2, 1, 1),
                's': KeyPosition('s', 2, 2, 2),
                'd': KeyPosition('d', 2, 3, 3),
                'f': KeyPosition('f', 2, 4, 4),
                
                # Lower row
                'z': KeyPosition('z', 3, 1, 1),
                'x': KeyPosition('x', 3, 2, 2),
                'c': KeyPosition('c', 3, 3, 3),
                'v': KeyPosition('v', 3, 4, 4)
            }
        else:
            raise ValueError(f"Unsupported analyze_keys option: {analyze_keys}")
    
    def analyze_preferences(self, data_path: str, output_folder: str) -> Dict[str, Any]:
        """Run complete preference analysis (exploratory + focused hypotheses)."""
        logger.info("Starting comprehensive preference analysis (exploratory + focused)...")
        
        # Load and validate data
        self.data = self._load_and_validate_data(data_path)
        logger.info(f"Loaded {len(self.data)} rows from {self.data['user_id'].nunique()} participants")
        
        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'exploratory'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'focused_hypotheses'), exist_ok=True)
        
        results = {}

        # PART I: FOCUSED HYPOTHESIS TESTING (Confirmatory Analysis)
        logger.info("=== PART I: FOCUSED HYPOTHESIS TESTING (20 hypotheses) ===")
        focused_results = self.focused_analyzer.analyze_hypotheses(self.data)
        results['focused_hypotheses'] = focused_results
        
        # PART II: EXPLORATORY ANALYSIS
        logger.info("=== PART II: EXPLORATORY ANALYSIS ===")
        
        # 1. Individual key preferences
        logger.info("Analyzing individual key preferences...")
        results['key_preferences'] = self._analyze_key_preferences(self.data)
        
        # 2. Transition type preferences 
        logger.info("Analyzing bigram transition preferences...")
        results['transition_preferences'] = self._analyze_transition_preferences(self.data)
        
        # PART III: COMBINED INSIGHTS
        logger.info("=== PART III: GENERATING COMBINED INSIGHTS ===")
        results['combined_insights'] = self._generate_combined_insights(
            focused_results, results['key_preferences'], results['transition_preferences']
        )
        
        # PART IV: SAVE RESULTS AND REPORTS
        logger.info("=== PART IV: GENERATING OUTPUTS ===")
        
        # Save exploratory results
        if self.config.get('save_raw_results', True):
            self._export_statistical_tables(results, os.path.join(output_folder, 'exploratory'))
        
        if self.config.get('generate_detailed_report', True):
            self._generate_comprehensive_report(results, os.path.join(output_folder, 'exploratory'))
        
        if self.config.get('generate_visualizations', True):
            self._create_visualizations(results, os.path.join(output_folder, 'exploratory'))
        
        # Save focused hypothesis results
        self._save_focused_results(focused_results, os.path.join(output_folder, 'focused_hypotheses'))
        
        # Save combined report
        self._save_combined_report(results, output_folder)
        
        # Save full results
        if self.config.get('save_raw_results', True):
            try:
                results_path = os.path.join(output_folder, 'complete_combined_analysis.json')
                with open(results_path, 'w') as f:
                    json_results = self._convert_for_json(results)
                    json.dump(json_results, f, indent=2, default=str)
                logger.info("Full results saved to JSON")
            except Exception as e:
                logger.warning(f"Could not save JSON results: {e}")

        logger.info(f"Combined analysis complete! Results saved to {output_folder}")
        return results
    
    def _load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate the input data."""
        try:
            data = pd.read_csv(data_path)
            
            # Check required columns
            required_cols = ['user_id', 'chosen_bigram', 'unchosen_bigram', 'sliderValue']
            missing_cols = set(required_cols) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert sliderValue to numeric, handling any string values
            data['sliderValue'] = pd.to_numeric(data['sliderValue'], errors='coerce')
            
            # Remove rows with invalid sliderValue
            invalid_slider_rows = data['sliderValue'].isna().sum()
            if invalid_slider_rows > 0:
                logger.warning(f"Removing {invalid_slider_rows} rows with invalid sliderValue")
                data = data.dropna(subset=['sliderValue'])
            
            # Filter to consistent choices (non-zero slider values)
            initial_len = len(data)
            data = data[data['sliderValue'] != 0].copy()
            logger.info(f"Using {len(data)}/{initial_len} consistent choice rows")
            
            # Convert bigrams to lowercase and ensure they are strings
            data['chosen_bigram'] = data['chosen_bigram'].astype(str).str.lower()
            data['unchosen_bigram'] = data['unchosen_bigram'].astype(str).str.lower()
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _analyze_key_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze preferences for individual keys."""
        
        # Extract all pairwise key comparisons
        key_comparisons = self._extract_key_comparisons(data)
        
        # Fit Bradley-Terry model
        bt_model = BradleyTerryModel(list(self.left_hand_keys), self.config)
        bt_model.fit(key_comparisons)
        
        # Get rankings and statistics
        rankings = bt_model.get_rankings()
        
        # Calculate bootstrap confidence intervals for BT strengths
        bt_confidence_intervals = self._calculate_bt_confidence_intervals(
            bt_model, key_comparisons
        )
        
        # Calculate pairwise effect sizes and significance
        pairwise_stats = self._calculate_pairwise_key_stats(key_comparisons, bt_model)
        
        # Organize results by finger
        finger_results = self._organize_key_results_by_finger(rankings, pairwise_stats)
        
        return {
            'overall_rankings': rankings,
            'bt_confidence_intervals': bt_confidence_intervals,
            'finger_specific_results': finger_results,
            'pairwise_statistics': pairwise_stats,
            'bt_model': bt_model,
            'key_comparisons': key_comparisons,
            'model_diagnostics': {
                'convergence': True,
                'n_comparisons': len(key_comparisons),
                'total_observations': sum(comp['total'] for comp in key_comparisons.values())
            }
        }
    
    def _extract_key_comparisons(self, data: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Extract pairwise key comparisons from bigram choice data."""
        
        key_comparisons = defaultdict(lambda: {'wins_item1': 0, 'total': 0})
        
        for idx, row in data.iterrows():
            try:
                chosen = str(row['chosen_bigram']).lower()
                unchosen = str(row['unchosen_bigram']).lower()
                
                # Extract individual keys from bigrams
                chosen_keys = [c for c in chosen if c in self.left_hand_keys]
                unchosen_keys = [c for c in unchosen if c in self.left_hand_keys]
                
                # Count key-level preferences
                for c_key in chosen_keys:
                    for u_key in unchosen_keys:
                        if c_key != u_key:
                            # Order keys consistently (alphabetically)
                            key1, key2 = sorted([c_key, u_key])
                            pair = (key1, key2)
                            
                            key_comparisons[pair]['total'] += 1
                            if c_key == key1:  # key1 was chosen
                                key_comparisons[pair]['wins_item1'] += 1
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        # Ensure all values are integers
        for pair in key_comparisons:
            key_comparisons[pair]['wins_item1'] = int(key_comparisons[pair]['wins_item1'])
            key_comparisons[pair]['total'] = int(key_comparisons[pair]['total'])
        
        return dict(key_comparisons)
    
    def _calculate_bt_confidence_intervals(self, bt_model: BradleyTerryModel, 
                                         key_comparisons: Dict, 
                                         confidence: float = None) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for Bradley-Terry strengths using bootstrap."""
        
        if confidence is None:
            confidence = self.config.get('confidence_level', 0.95)
        
        n_bootstrap = self.config.get('bootstrap_iterations', 1000)
        n_items = len(bt_model.items)
        bootstrap_strengths = np.zeros((n_bootstrap, n_items))
        
        # Convert comparisons to list for resampling
        comparison_list = []
        for (key1, key2), data in key_comparisons.items():
            wins1 = int(data['wins_item1'])
            total = int(data['total'])
            # Add individual comparison records
            for _ in range(wins1):
                comparison_list.append((key1, key2, 1))  # key1 wins
            for _ in range(total - wins1):
                comparison_list.append((key1, key2, 0))  # key2 wins
        
        if not comparison_list:
            return {item: (np.nan, np.nan) for item in bt_model.items}
        
        # Bootstrap resampling
        for b in range(n_bootstrap):
            # Resample comparisons
            resampled = np.random.choice(len(comparison_list), 
                                       size=len(comparison_list), 
                                       replace=True)
            
            # Reconstruct comparison counts
            bootstrap_comparisons = defaultdict(lambda: {'wins_item1': 0, 'total': 0})
            for idx in resampled:
                key1, key2, outcome = comparison_list[idx]
                pair = tuple(sorted([key1, key2]))
                bootstrap_comparisons[pair]['total'] += 1
                if (outcome == 1 and key1 == pair[0]) or (outcome == 0 and key2 == pair[0]):
                    bootstrap_comparisons[pair]['wins_item1'] += 1
            
            # Fit bootstrap model
            bootstrap_model = BradleyTerryModel(bt_model.items, self.config)
            try:
                bootstrap_model.fit(dict(bootstrap_comparisons))
                if bootstrap_model.strengths is not None:
                    bootstrap_strengths[b, :] = bootstrap_model.strengths
                else:
                    bootstrap_strengths[b, :] = bt_model.strengths
            except:
                bootstrap_strengths[b, :] = bt_model.strengths
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        ci_dict = {}
        for i, item in enumerate(bt_model.items):
            lower = float(np.percentile(bootstrap_strengths[:, i], 100 * alpha/2))
            upper = float(np.percentile(bootstrap_strengths[:, i], 100 * (1 - alpha/2)))
            ci_dict[item] = (lower, upper)
        
        return ci_dict
    
    def _calculate_pairwise_key_stats(self, comparisons: Dict[Tuple[str, str], Dict[str, int]], 
                                    bt_model: BradleyTerryModel) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Calculate statistical significance and effect sizes for all key pairs."""
        
        pairwise_stats = {}
        all_p_values = []
        comparison_keys = []
        
        min_comparisons = self.config.get('min_key_comparisons', 5)
        key_thresholds = self.config.get('key_effect_thresholds', {
            'negligible': 0.05, 'small': 0.15, 'medium': 0.30
        })
        
        for (key1, key2), data in comparisons.items():
            wins1 = data['wins_item1']
            total = data['total']
            proportion1 = wins1 / total if total > 0 else 0.5
            
            # Effect size (deviation from chance)
            effect_size = abs(proportion1 - 0.5)
            
            # Statistical significance (binomial test)
            if total >= min_comparisons:
                binom_result = stats.binomtest(wins1, total, 0.5, alternative='two-sided')
                p_value = binom_result.pvalue
                ci_lower, ci_upper = self._wilson_ci(wins1, total)
            else:
                p_value = np.nan
                ci_lower = ci_upper = np.nan
            
            # Cohen's h effect size
            cohens_h = self._calculate_cohens_h(proportion1, 0.5)
            
            # Practical significance
            practical_sig = self._classify_practical_significance(effect_size, key_thresholds)
            
            stats_dict = {
                'proportion_key1_wins': proportion1,
                'effect_size': effect_size,
                'cohens_h': cohens_h,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_comparisons': total,
                'practical_significance': practical_sig
            }
            
            pairwise_stats[(key1, key2)] = stats_dict
            
            if not np.isnan(p_value):
                all_p_values.append(p_value)
                comparison_keys.append((key1, key2))
        
        # Apply multiple comparison correction
        if all_p_values:
            try:
                from statsmodels.stats.multitest import multipletests
                correction_method = self.config.get('correction_method', 'fdr_bh')
                alpha_level = self.config.get('alpha_level', 0.05)
                _, p_corrected, _, _ = multipletests(all_p_values, method=correction_method, alpha=alpha_level)
                
                for i, (key1, key2) in enumerate(comparison_keys):
                    pairwise_stats[(key1, key2)]['p_value_corrected'] = p_corrected[i]
                    pairwise_stats[(key1, key2)]['significant_corrected'] = p_corrected[i] < alpha_level
            except ImportError:
                logger.warning("statsmodels not available, skipping multiple comparison correction")
        
        return pairwise_stats
    
    def _organize_key_results_by_finger(self, rankings: List[Tuple[str, float]], 
                                      pairwise_stats: Dict) -> Dict[str, Any]:
        """Organize key ranking results by finger."""
        
        finger_results = {}
        
        for finger in range(1, 5):
            finger_keys = [key for key, pos in self.key_positions.items() if pos.finger == finger]
            finger_rankings = [(key, strength) for key, strength in rankings if key in finger_keys]
            
            # Sort by strength (highest to lowest)
            finger_rankings.sort(key=lambda x: x[1], reverse=True)
            
            # Extract pairwise comparisons for this finger
            finger_pairs = {}
            for (key1, key2), stats_dict in pairwise_stats.items():
                if key1 in finger_keys and key2 in finger_keys:
                    finger_pairs[(key1, key2)] = stats_dict
            
            # Determine confidence level of ranking
            confidence = self._assess_ranking_confidence(finger_pairs)
            
            finger_results[f'finger_{finger}'] = {
                'keys': finger_keys,
                'rankings': finger_rankings,
                'pairwise_comparisons': finger_pairs,
                'confidence_level': confidence,
                'interpretation': self._interpret_finger_ranking(finger_rankings, finger)
            }
        
        return finger_results
    
    def _analyze_transition_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze preferences for different types of bigram transitions."""
        
        # Classify all bigrams by transition type
        transition_classifications = self._classify_transitions(data)
        
        # Extract pairwise transition comparisons
        transition_comparisons = self._extract_transition_comparisons(data, transition_classifications)
        
        # Fit Bradley-Terry model for transitions
        transition_types = list(set(transition_classifications.values()))
        bt_model = BradleyTerryModel(transition_types, self.config)
        bt_model.fit(transition_comparisons)
        
        # Get rankings and statistics
        rankings = bt_model.get_rankings()
        
        # Calculate bootstrap confidence intervals for transitions
        bt_confidence_intervals = self._calculate_bt_confidence_intervals(
            bt_model, transition_comparisons
        )
        
        pairwise_stats = self._calculate_pairwise_transition_stats(transition_comparisons, bt_model)
        
        # Organize results by category
        categorized_results = self._categorize_transition_results(rankings, pairwise_stats)
        
        return {
            'overall_rankings': rankings,
            'bt_confidence_intervals': bt_confidence_intervals,
            'categorized_results': categorized_results,
            'pairwise_statistics': pairwise_stats,
            'transition_classifications': transition_classifications,
            'bt_model': bt_model,
            'transition_comparisons': transition_comparisons,
            'model_diagnostics': {
                'n_transition_types': len(transition_types),
                'n_comparisons': len(transition_comparisons),
                'total_observations': sum(comp['total'] for comp in transition_comparisons.values())
            }
        }
    
    def _classify_transitions(self, data: pd.DataFrame) -> Dict[str, str]:
        """Classify each bigram by its transition type."""
        
        classifications = {}
        
        for _, row in data.iterrows():
            for bigram_col in ['chosen_bigram', 'unchosen_bigram']:
                bigram = row[bigram_col]
                if len(bigram) >= 2:
                    char1, char2 = bigram[0], bigram[1]
                    
                    if char1 in self.key_positions and char2 in self.key_positions:
                        pos1 = self.key_positions[char1]
                        pos2 = self.key_positions[char2]
                        
                        # Classify transition
                        transition_type = self._get_transition_type(pos1, pos2)
                        classifications[bigram] = transition_type
        
        return classifications
    
    def _get_transition_type(self, pos1: KeyPosition, pos2: KeyPosition) -> str:
        """Determine the transition type between two key positions."""
        
        # Row pattern
        row_separation = abs(pos1.row - pos2.row)
        if row_separation == 0:
            row_pattern = 'same'
        elif row_separation == 1:
            row_pattern = 'reach'
        else:
            row_pattern = 'hurdle'
        
        # Finger separation
        finger_separation = abs(pos1.finger - pos2.finger)
        
        return f"Δ{finger_separation}-finger {row_pattern}"
    
    def _extract_transition_comparisons(self, data: pd.DataFrame, 
                                      classifications: Dict[str, str]) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Extract pairwise transition type comparisons."""
        
        transition_comparisons = defaultdict(lambda: {'wins_item1': 0, 'total': 0})
        
        for _, row in data.iterrows():
            chosen = row['chosen_bigram']
            unchosen = row['unchosen_bigram']
            
            chosen_type = classifications.get(chosen)
            unchosen_type = classifications.get(unchosen)
            
            if chosen_type and unchosen_type and chosen_type != unchosen_type:
                # Order transition types consistently
                type1, type2 = sorted([chosen_type, unchosen_type])
                pair = (type1, type2)
                
                transition_comparisons[pair]['total'] += 1
                if chosen_type == type1:  # type1 was chosen
                    transition_comparisons[pair]['wins_item1'] += 1
        
        return dict(transition_comparisons)
    
    def _calculate_pairwise_transition_stats(self, comparisons: Dict[Tuple[str, str], Dict[str, int]], 
                                           bt_model: BradleyTerryModel) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Calculate statistical significance and effect sizes for transition pairs."""
        
        pairwise_stats = {}
        all_p_values = []
        comparison_keys = []
        
        min_comparisons = self.config.get('min_transition_comparisons', 10)
        transition_thresholds = self.config.get('transition_effect_thresholds', {
            'negligible': 0.03, 'small': 0.10, 'medium': 0.25
        })
        
        for (type1, type2), data in comparisons.items():
            wins1 = data['wins_item1']
            total = data['total']
            proportion1 = wins1 / total if total > 0 else 0.5
            
            # Effect size
            effect_size = abs(proportion1 - 0.5)
            
            # Statistical significance
            if total >= min_comparisons:
                binom_result = stats.binomtest(wins1, total, 0.5, alternative='two-sided')
                p_value = binom_result.pvalue
                ci_lower, ci_upper = self._wilson_ci(wins1, total)
            else:
                p_value = np.nan
                ci_lower = ci_upper = np.nan
            
            # Cohen's h
            cohens_h = self._calculate_cohens_h(proportion1, 0.5)
            
            # Practical significance
            practical_sig = self._classify_practical_significance(effect_size, transition_thresholds)
            
            stats_dict = {
                'proportion_type1_wins': proportion1,
                'effect_size': effect_size,
                'cohens_h': cohens_h,
                'p_value': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_comparisons': total,
                'practical_significance': practical_sig
            }
            
            pairwise_stats[(type1, type2)] = stats_dict
            
            if not np.isnan(p_value):
                all_p_values.append(p_value)
                comparison_keys.append((type1, type2))
        
        # Apply multiple comparison correction
        if all_p_values:
            try:
                from statsmodels.stats.multitest import multipletests
                correction_method = self.config.get('correction_method', 'fdr_bh')
                alpha_level = self.config.get('alpha_level', 0.05)
                _, p_corrected, _, _ = multipletests(all_p_values, method=correction_method, alpha=alpha_level)
                
                for i, (type1, type2) in enumerate(comparison_keys):
                    pairwise_stats[(type1, type2)]['p_value_corrected'] = p_corrected[i]
                    pairwise_stats[(type1, type2)]['significant_corrected'] = p_corrected[i] < alpha_level
            except ImportError:
                logger.warning("statsmodels not available, skipping multiple comparison correction")
        
        return pairwise_stats

    # =========================================================================
    # STATISTICAL HELPER METHODS
    # =========================================================================
    
    def _wilson_ci(self, successes: int, trials: int, confidence: float = None) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        if confidence is None:
            confidence = self.config.get('confidence_level', 0.95)
        
        if trials == 0:
            return np.nan, np.nan
            
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = successes / trials
        
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)
        
        return ci_lower, ci_upper
    
    def _calculate_cohens_h(self, p1: float, p2: float) -> float:
        """Calculate Cohen's h for difference between two proportions."""
        if np.isnan(p1) or np.isnan(p2):
            return np.nan
        
        # Avoid numerical issues
        p1 = max(0.001, min(0.999, p1))
        p2 = max(0.001, min(0.999, p2))
        
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    def _classify_practical_significance(self, effect_size: float, thresholds: Dict[str, float]) -> str:
        """Classify practical significance based on effect size."""
        if effect_size >= thresholds.get('medium', 0.30):
            return 'large'
        elif effect_size >= thresholds.get('small', 0.15):
            return 'medium'
        elif effect_size >= thresholds.get('negligible', 0.05):
            return 'small'
        else:
            return 'negligible'
    
    def _assess_ranking_confidence(self, pairwise_stats: Dict) -> str:
        """Assess confidence level in ranking based on pairwise statistics."""
        if not pairwise_stats:
            return 'insufficient_data'
        
        criteria = self.config.get('confidence_criteria', {})
        
        # Count significant differences
        significant_pairs = sum(1 for stats in pairwise_stats.values() 
                              if stats.get('significant_corrected', False))
        total_pairs = len(pairwise_stats)
        
        # Count large effect sizes
        large_effects = sum(1 for stats in pairwise_stats.values()
                           if stats.get('practical_significance') == 'large')
        
        sig_pct = significant_pairs / total_pairs if total_pairs > 0 else 0
        large_pct = large_effects / total_pairs if total_pairs > 0 else 0
        
        if (sig_pct >= criteria.get('definitive', {}).get('min_significant_pairs', 0.7) and 
            large_pct >= criteria.get('definitive', {}).get('min_large_effects', 0.5)):
            return 'definitive'
        elif sig_pct >= criteria.get('probable', {}).get('min_significant_pairs', 0.5):
            return 'probable'
        elif (sig_pct >= criteria.get('suggestive', {}).get('min_significant_pairs', 0.3) or 
              large_pct >= criteria.get('suggestive', {}).get('min_large_effects', 0.2)):
            return 'suggestive'
        else:
            return 'inconclusive'

    # =========================================================================
    # INTERPRETATION AND REPORTING
    # =========================================================================
    
    def _interpret_finger_ranking(self, rankings: List[Tuple[str, float]], finger: int) -> str:
        """Generate human-readable interpretation of finger ranking."""
        if not rankings:
            return f"Insufficient data for finger {finger}"
        
        # Map keys to row names
        row_names = {1: 'upper', 2: 'home', 3: 'lower'}
        key_rows = {}
        for key, strength in rankings:
            if key in self.key_positions:
                key_rows[key] = row_names[self.key_positions[key].row]
        
        # Create ranking description
        ranking_desc = " > ".join([f"{key} ({key_rows.get(key, 'unknown')})" for key, _ in rankings])
        
        # Identify pattern
        if len(rankings) >= 3:
            top_row = key_rows.get(rankings[0][0])
            bottom_row = key_rows.get(rankings[-1][0])
            
            if top_row == 'home':
                pattern = "Strong home row preference"
            elif top_row == 'upper' and bottom_row == 'lower':
                pattern = "Upper row preference over lower row"
            elif top_row == 'lower' and bottom_row == 'upper':
                pattern = "Lower row preference over upper row"
            else:
                pattern = "Mixed preferences"
        else:
            pattern = "Limited data"
        
        return f"Finger {finger}: {ranking_desc}. Pattern: {pattern}"
    
    def _categorize_transition_results(self, rankings: List[Tuple[str, float]], 
                                     pairwise_stats: Dict) -> Dict[str, Any]:
        """Categorize transition results by type."""
        
        categories = {
            'row_patterns': {'same': [], 'reach': [], 'hurdle': []},
            'finger_separations': {'1': [], '2': [], '3': []}
        }
        
        for transition_type, strength in rankings:
            # Parse transition type
            parts = transition_type.split('_')
            
            # Row pattern
            if 'same' in parts:
                categories['row_patterns']['same'].append((transition_type, strength))
            elif 'reach' in parts:
                categories['row_patterns']['reach'].append((transition_type, strength))
            elif 'hurdle' in parts:
                categories['row_patterns']['hurdle'].append((transition_type, strength))
            
            # Finger separation
            for sep in ['1', '2', '3']:
                if f'{sep}finger' in transition_type:
                    categories['finger_separations'][sep].append((transition_type, strength))
        
        # Sort each category by strength
        for category in categories.values():
            for subcategory in category.values():
                subcategory.sort(key=lambda x: x[1], reverse=True)
        
        return categories

    def _generate_combined_insights(self, focused_results: Dict[str, Any], 
                                  key_results: Dict[str, Any], 
                                  transition_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights combining focused and exploratory results."""
        
        insights = {
            'validation_summary': {},
            'convergent_findings': [],
            'divergent_findings': [],
            'practical_recommendations': []
        }
        
        # Validate finger preference hypothesis against individual key rankings
        finger_validation = self._validate_finger_preferences(focused_results, key_results)
        insights['validation_summary']['finger_preferences'] = finger_validation
        
        # Generate practical recommendations
        insights['practical_recommendations'] = self._generate_practical_recommendations(
            focused_results, key_results, transition_results
        )
        
        return insights
    
    def _validate_finger_preferences(self, focused_results: Dict[str, Any], 
                                   key_results: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate focused finger hypothesis with individual key rankings."""
        
        # Get finger preference results from focused analysis
        finger_hypotheses = ['finger_f4_vs_f3', 'finger_f3_vs_f2', 'finger_f2_vs_f1']
        focused_finger_results = {}
        
        for hyp in finger_hypotheses:
            if hyp in focused_results['hypothesis_results']:
                result = focused_results['hypothesis_results'][hyp]
                if 'statistics' in result:
                    focused_finger_results[hyp] = result['statistics']
        
        # Analyze individual key rankings by finger
        finger_strengths = {1: [], 2: [], 3: [], 4: []}
        for key, strength in key_results['overall_rankings']:
            if key in self.key_positions:
                finger = self.key_positions[key].finger
                finger_strengths[finger].append(strength)
        
        # Calculate average strength per finger
        finger_averages = {}
        for finger, strengths in finger_strengths.items():
            if strengths:
                finger_averages[finger] = np.mean(strengths)
        
        # Compare rankings
        exploratory_finger_ranking = sorted(finger_averages.items(), key=lambda x: x[1], reverse=True)
        
        validation = {
            'exploratory_finger_ranking': exploratory_finger_ranking,
            'focused_finger_results': focused_finger_results,
            'consistency_analysis': self._analyze_finger_consistency(focused_finger_results, finger_averages)
        }
        
        return validation
    
    def _analyze_finger_consistency(self, focused_results: Dict[str, Any], 
                                  finger_averages: Dict[int, float]) -> str:
        """Analyze consistency between focused and exploratory finger results."""
        
        # Check if exploratory rankings match focused hypothesis predictions
        if len(finger_averages) < 4:
            return "Insufficient data for consistency analysis"
        
        # Exploratory ranking order
        exp_order = [finger for finger, _ in sorted(finger_averages.items(), key=lambda x: x[1], reverse=True)]
        
        # Check specific comparisons from focused hypotheses
        consistencies = []
        
        # F4 vs F3
        if 'finger_f4_vs_f3' in focused_results:
            focused_f4_better = focused_results['finger_f4_vs_f3'].get('proportion_val1_wins', 0.5) > 0.5
            exp_f4_better = finger_averages.get(4, 0) > finger_averages.get(3, 0)
            consistencies.append(f"F4 vs F3: {'Consistent' if focused_f4_better == exp_f4_better else 'Inconsistent'}")
        
        # F3 vs F2
        if 'finger_f3_vs_f2' in focused_results:
            focused_f3_better = focused_results['finger_f3_vs_f2'].get('proportion_val1_wins', 0.5) > 0.5
            exp_f3_better = finger_averages.get(3, 0) > finger_averages.get(2, 0)
            consistencies.append(f"F3 vs F2: {'Consistent' if focused_f3_better == exp_f3_better else 'Inconsistent'}")
        
        # F2 vs F1
        if 'finger_f2_vs_f1' in focused_results:
            focused_f2_better = focused_results['finger_f2_vs_f1'].get('proportion_val1_wins', 0.5) > 0.5
            exp_f2_better = finger_averages.get(2, 0) > finger_averages.get(1, 0)
            consistencies.append(f"F2 vs F1: {'Consistent' if focused_f2_better == exp_f2_better else 'Inconsistent'}")
        
        consistency_summary = "; ".join(consistencies)
        return f"Exploratory order: {exp_order}. Focused comparisons: {consistency_summary}"
    
    def _generate_practical_recommendations(self, focused_results: Dict[str, Any], 
                                          key_results: Dict[str, Any], 
                                          transition_results: Dict[str, Any]) -> List[str]:
        """Generate practical keyboard layout recommendations."""
        
        recommendations = []
        
        # Based on significant focused hypotheses
        significant_hypotheses = focused_results['summary']['significant_results']
        
        for result in significant_hypotheses:
            hyp_name = result['hypothesis']
            effect_size = result['effect_size']
            
            if 'finger' in hyp_name and effect_size > 0.20:
                recommendations.append(f"FINGER PREFERENCE: {result['description']} - prioritize this in layout design")
            elif 'home' in hyp_name and effect_size > 0.15:
                recommendations.append(f"HOME ROW: {result['description']} - maximize home row usage")
            elif 'direction' in hyp_name and effect_size > 0.15:
                recommendations.append(f"DIRECTION: {result['description']} - optimize roll directions")
        
        # Based on exploratory key rankings
        top_keys = key_results['overall_rankings'][:3]
        bottom_keys = key_results['overall_rankings'][-3:]
        
        recommendations.append(f"KEY PLACEMENT: Prioritize keys {[k.upper() for k, _ in top_keys]} in frequent positions")
        recommendations.append(f"KEY PLACEMENT: Avoid placing frequent letters on keys {[k.upper() for k, _ in bottom_keys]}")
        
        return recommendations

    # =========================================================================
    # VISUALIZATION AND OUTPUT METHODS
    # =========================================================================
    
    def _create_visualizations(self, results: Dict[str, Any], output_folder: str) -> None:
        """Create comprehensive visualizations."""
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        dpi = self.config.get('figure_dpi', 300)
        plt.rcParams.update({'font.size': 12, 'figure.dpi': dpi})
        
        # 1. Key preference heatmap
        self._plot_key_preference_heatmap(results['key_preferences'], output_folder)
        
        # 2. Key strength differences with confidence intervals
        self._plot_key_strengths_with_cis(results['key_preferences'], output_folder)
        
        # 3. Transition preference rankings
        self._plot_transition_rankings(results['transition_preferences'], output_folder)
        
        # 4. Effect size distributions
        self._plot_effect_size_distributions(results, output_folder)
    
    def _plot_key_preference_heatmap(self, key_results: Dict[str, Any], output_folder: str) -> None:
        """Create heatmap showing key preferences."""
        
        plot_data = []
        for key, strength in key_results['overall_rankings']:
            if key in self.key_positions:
                pos = self.key_positions[key]
                plot_data.append({
                    'row': pos.row,
                    'column': pos.column,
                    'finger': pos.finger,
                    'key': key.upper(),
                    'strength': strength
                })
        
        if not plot_data:
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create pivot tables for strength and key labels
        pivot_strength = df.pivot(index='row', columns='column', values='strength')
        pivot_keys = df.pivot(index='row', columns='column', values='key').fillna('')
        
        # Determine the full range of columns and rows for a consistent grid
        all_columns = sorted(df['column'].unique())
        all_rows = sorted(df['row'].unique())

        # Reindex pivots to ensure all rows and columns are present, filling with NaN
        pivot_strength = pivot_strength.reindex(index=all_rows, columns=all_columns)
        pivot_keys = pivot_keys.reindex(index=all_rows, columns=all_columns).fillna('')

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        colormap = self.config.get('heatmap_colormap', 'RdYlGn')
        sns.heatmap(pivot_strength, cmap=colormap, center=0, 
                    cbar_kws={'label': 'Bradley-Terry Strength'},
                    linewidths=.5, linecolor='lightgray', ax=ax,
                    mask=pivot_strength.isnull()
                )
        
        # Add key and finger labels manually
        for text in ax.texts:
            text.set_visible(False)
            
        for (j, i), key_label in np.ndenumerate(pivot_keys):
            row_idx = all_rows[j]
            col_idx = all_columns[i]
            
            # Find the original data point to get the finger
            original_entry = df[(df['row'] == row_idx) & (df['column'] == col_idx)]
            if not original_entry.empty:
                finger_label = f"F{original_entry['finger'].iloc[0]}"
                full_label = f"{key_label}\n({finger_label})"
                ax.text(i + 0.5, j + 0.5, full_label, 
                        ha='center', va='center', color='black', fontsize=10)
        
        ax.set_title('key preferences by keyboard position (strength and finger)')
        ax.set_xlabel('Column (1=Leftmost, 5=Index Center)')
        ax.set_ylabel('Row (1=Upper, 2=Home, 3=Lower)')
        
        # Set tick labels to match actual row/column numbers
        ax.set_xticks(np.arange(len(all_columns)) + 0.5, labels=[str(col) for col in all_columns])
        ax.set_yticks(np.arange(len(all_rows)) + 0.5, labels=[str(row) for row in all_rows])

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'key_preference_heatmap.png'), 
                dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
    def _plot_key_strengths_with_cis(self, key_results: Dict[str, Any], output_folder: str) -> None:
        """Create forest plot showing key strengths with confidence intervals."""
        
        rankings = key_results['overall_rankings']
        bt_cis = key_results['bt_confidence_intervals']
        
        if not rankings:
            return
        
        # Prepare data
        keys = [key.upper() for key, _ in rankings]
        strengths = [float(strength) for _, strength in rankings]
        ci_lowers = [float(bt_cis.get(key.lower(), (np.nan, np.nan))[0]) for key, _ in rankings]
        ci_uppers = [float(bt_cis.get(key.lower(), (np.nan, np.nan))[1]) for key, _ in rankings]
        
        # Calculate error bars, handling NaN values
        lower_errs = []
        upper_errs = []
        for strength, ci_lower, ci_upper in zip(strengths, ci_lowers, ci_uppers):
            if np.isnan(ci_lower) or np.isnan(ci_upper):
                lower_errs.append(0)
                upper_errs.append(0)
            else:
                lower_errs.append(abs(strength - ci_lower))
                upper_errs.append(abs(ci_upper - strength))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create horizontal error bar plot
        y_pos = np.arange(len(keys))
        colors = ['green' if s > 0 else 'red' for s in strengths]
        
        ax.errorbar(strengths, y_pos, xerr=[lower_errs, upper_errs], 
                   fmt='o', color='black', ecolor='gray', capsize=3, capthick=1)
        
        # Color the points
        for i, (strength, color) in enumerate(zip(strengths, colors)):
            ax.scatter(strength, i, c=color, s=100, alpha=0.7, zorder=5)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keys)
        ax.set_xlabel('Bradley-Terry Strength (95% CI)')
        ax.set_title('key preference strengths with confidence intervals')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add finger labels
        x_max = max(strengths)
        x_range = max(strengths) - min(strengths)
        finger_x_pos = x_max + (x_range * 0.2)
        
        for i, (key, _) in enumerate(rankings):
            if key in self.key_positions:
                finger = self.key_positions[key].finger
                ax.text(finger_x_pos, i, f'F{finger}', 
                       color='black', fontweight='bold', va='center', ha='left')
        
        # Extend x-axis to accommodate finger labels
        ax.set_xlim(left=min(strengths) - (x_range * 0.1), 
                    right=finger_x_pos + (x_range * 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'key_strengths_with_cis.png'), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
    def _plot_transition_rankings(self, transition_results: Dict[str, Any], output_folder: str) -> None:
        """Create bar plot of transition type rankings."""
        
        rankings = transition_results['overall_rankings']
        if not rankings:
            return
        
        # Prepare data (top 15 for readability)
        top_rankings = rankings[:15]
        transition_types = [t.replace('Δ', 'Δ') for t, _ in top_rankings]
        strengths = [s for _, s in top_rankings]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(transition_types))
        bars = ax.barh(y_pos, strengths, alpha=0.7)
        
        # Color bars based on strength
        for i, bar in enumerate(bars):
            if strengths[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(transition_types, fontsize=10)
        ax.set_xlabel('Bradley-Terry Strength')
        ax.set_title('top 15 bigram transition type preferences')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'transition_rankings.png'), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
    def _plot_effect_size_distributions(self, results: Dict[str, Any], output_folder: str) -> None:
        """Plot distributions of effect sizes."""
        
        # Collect effect sizes
        key_effects = []
        transition_effects = []
        
        # Key effect sizes
        for stats in results['key_preferences']['pairwise_statistics'].values():
            if not np.isnan(stats['effect_size']):
                key_effects.append(stats['effect_size'])
        
        # Transition effect sizes
        for stats in results['transition_preferences']['pairwise_statistics'].values():
            if not np.isnan(stats['effect_size']):
                transition_effects.append(stats['effect_size'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Key effect sizes
        if key_effects:
            ax1.hist(key_effects, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(np.mean(key_effects), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(key_effects):.3f}')
            ax1.set_xlabel('Effect Size')
            ax1.set_ylabel('Frequency')
            ax1.set_title('key preference effect sizes')
            ax1.legend()
        
        # Transition effect sizes
        if transition_effects:
            ax2.hist(transition_effects, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(np.mean(transition_effects), color='red', linestyle='--',
                       label=f'Mean: {np.mean(transition_effects):.3f}')
            ax2.set_xlabel('Effect Size')
            ax2.set_ylabel('Frequency')
            ax2.set_title('transition preference effect sizes')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'effect_size_distributions.png'), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()

    def _export_statistical_tables(self, results: Dict[str, Any], output_folder: str) -> None:
        """Export detailed statistical tables to CSV."""
        
        # Key statistics table
        key_table = self._create_comprehensive_key_table(results['key_preferences'])
        key_table.to_csv(os.path.join(output_folder, 'key_preference_statistics.csv'), index=False)
        
        # Pairwise key comparison details
        pairwise_data = []
        for (key1, key2), stats in results['key_preferences']['pairwise_statistics'].items():
            pairwise_data.append({
                'Key1': key1.upper(),
                'Key2': key2.upper(),
                'Key1_Win_Rate': stats['proportion_key1_wins'],
                'Effect_Size': stats['effect_size'],
                'Cohens_H': stats['cohens_h'],
                'P_Value': stats.get('p_value', np.nan),
                'P_Value_Corrected': stats.get('p_value_corrected', np.nan),
                'CI_Lower': stats['ci_lower'],
                'CI_Upper': stats['ci_upper'],
                'N_Comparisons': stats['n_comparisons'],
                'Practical_Significance': stats['practical_significance'],
                'Statistically_Significant': stats.get('significant_corrected', False)
            })
        
        pairwise_df = pd.DataFrame(pairwise_data)
        pairwise_df.to_csv(os.path.join(output_folder, 'pairwise_key_comparisons.csv'), index=False)
        
        # Transition rankings with confidence intervals
        transition_rankings = []
        bt_cis = results['transition_preferences']['bt_confidence_intervals']
        for transition_type, strength in results['transition_preferences']['overall_rankings']:
            ci_lower, ci_upper = bt_cis.get(transition_type, (np.nan, np.nan))
            transition_rankings.append({
                'Transition_Type': transition_type,
                'BT_Strength': strength,
                'BT_CI_Lower': ci_lower,
                'BT_CI_Upper': ci_upper
            })
        
        transition_df = pd.DataFrame(transition_rankings)
        transition_df.to_csv(os.path.join(output_folder, 'transition_preference_statistics.csv'), index=False)
    
    def _create_comprehensive_key_table(self, key_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive statistical table for all keys."""
        
        rankings = key_results['overall_rankings']
        pairwise_stats = key_results['pairwise_statistics']
        bt_confidence_intervals = key_results['bt_confidence_intervals']
        
        table_data = []
        
        for rank, (key, strength) in enumerate(rankings, 1):
            # Get key position info
            pos = self.key_positions.get(key, KeyPosition(key, 0, 0, 0))
            
            # Calculate aggregate pairwise statistics for this key
            key_pairwise_stats = []
            for (key1, key2), stats in pairwise_stats.items():
                if key in [key1, key2]:
                    # Determine if this key "won" the comparison
                    if key == key1:
                        win_rate = stats['proportion_key1_wins']
                    else:
                        win_rate = 1 - stats['proportion_key1_wins']
                    key_pairwise_stats.append({
                        'win_rate': win_rate,
                        'effect_size': stats['effect_size'],
                        'p_value': stats.get('p_value_corrected', np.nan),
                        'n_comparisons': stats['n_comparisons']
                    })
            
            # Aggregate pairwise stats
            if key_pairwise_stats:
                mean_win_rate = np.mean([s['win_rate'] for s in key_pairwise_stats])
                mean_effect_size = np.mean([s['effect_size'] for s in key_pairwise_stats])
                total_comparisons = sum([s['n_comparisons'] for s in key_pairwise_stats])
                significant_comparisons = sum([1 for s in key_pairwise_stats 
                                             if not np.isnan(s['p_value']) and s['p_value'] < 0.05])
                pct_significant = significant_comparisons / len(key_pairwise_stats) * 100
            else:
                mean_win_rate = mean_effect_size = total_comparisons = pct_significant = np.nan
            
            # Get BT confidence interval
            bt_ci_lower, bt_ci_upper = bt_confidence_intervals.get(key, (np.nan, np.nan))
            
            # Determine overall preference level
            if strength > 0.2:
                preference_level = "Highly Preferred"
            elif strength > 0.05:
                preference_level = "Preferred"
            elif strength > -0.05:
                preference_level = "Neutral"
            elif strength > -0.2:
                preference_level = "Less Preferred"
            else:
                preference_level = "Avoided"
            
            table_data.append({
                'Rank': rank,
                'Key': key.upper(),
                'Row': pos.row,
                'Column': pos.column,
                'Finger': pos.finger,
                'BT_Strength': strength,
                'BT_CI_Lower': bt_ci_lower,
                'BT_CI_Upper': bt_ci_upper,
                'Mean_Win_Rate': mean_win_rate,
                'Mean_Effect_Size': mean_effect_size,
                'Total_Comparisons': total_comparisons,
                'Pct_Significant': pct_significant,
                'Preference_Level': preference_level
            })
        
        return pd.DataFrame(table_data)
    
    def _generate_comprehensive_report(self, results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive text report for exploratory analysis."""
        
        report_lines = [
            "exploratory keyboard preference analysis report",
            "=" * 60,
            "",
            "This section contains data-driven exploration of keyboard preferences",
            "using Bradley-Terry models without pre-specified hypotheses.",
            "",
            "TOP 10 PREFERRED KEYS:",
            "=" * 25,
            ""
        ]
        
        # Key preference results
        key_results = results['key_preferences']
        for i, (key, strength) in enumerate(key_results['overall_rankings'][:10], 1):
            finger = self.key_positions.get(key, KeyPosition('', 0, 0, 0)).finger
            row = self.key_positions.get(key, KeyPosition('', 0, 0, 0)).row
            row_name = {1: 'upper', 2: 'home', 3: 'lower'}.get(row, 'unknown')
            report_lines.append(
                f"{i:2d}. {key.upper()} (F{finger}, {row_name} row): {strength:6.3f}"
            )
        
        report_lines.extend([
            "",
            "TOP 10 PREFERRED TRANSITIONS:",
            "=" * 35,
            ""
        ])
        
        # Transition preference results
        transition_results = results['transition_preferences']
        for i, (transition_type, strength) in enumerate(transition_results['overall_rankings'][:10], 1):
            report_lines.append(
                f"{i:2d}. {transition_type:<35} {strength:6.3f}"
            )
        
        # Save report
        report_path = os.path.join(output_folder, 'exploratory_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _save_focused_results(self, focused_results: Dict[str, Any], output_folder: str) -> None:
        """Save focused hypothesis results."""
        
        # Save detailed results as JSON
        with open(os.path.join(output_folder, 'focused_hypothesis_results.json'), 'w') as f:
            json_results = self._convert_for_json(focused_results)
            json.dump(json_results, f, indent=2, default=str)
        
        # Save summary as CSV
        summary_data = []
        for hyp_name, result in focused_results['hypothesis_results'].items():
            if 'statistics' in result:
                stats = result['statistics']
                val1, val2 = stats.get('values_compared', ('', ''))
                summary_data.append({
                    'Hypothesis': hyp_name,
                    'Description': result['description'],
                    'Values_Compared': f"{val1} vs {val2}",
                    'Val1_Win_Rate': stats['proportion_val1_wins'],
                    'Effect_Size': stats['effect_size'],
                    'P_Value': stats.get('p_value', np.nan),
                    'P_Value_Corrected': stats.get('p_value_corrected', np.nan),
                    'Significant_Corrected': stats.get('significant_corrected', False),
                    'Practical_Significance': stats['practical_significance'],
                    'N_Comparisons': stats['n_comparisons']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(output_folder, 'focused_hypothesis_summary.csv'), index=False)
        
        # Save text report
        self._save_focused_text_report(focused_results, output_folder)
    
    def _save_focused_text_report(self, focused_results: Dict[str, Any], output_folder: str) -> None:
        """Save human-readable text report for focused hypotheses."""
        
        summary = focused_results['summary']
        
        report_lines = [
            "focused keyboard preference hypothesis testing results",
            "=" * 65,
            "",
            f"Total hypotheses tested: {focused_results['total_hypotheses']}",
            f"Total comparisons made: {focused_results['total_comparisons']}",
            f"Significant results (after FDR correction): {len(summary['significant_results'])}",
            f"Large practical effects (>30% preference): {len(summary['large_effects'])}",
            "",
            "SIGNIFICANT RESULTS (after global FDR correction):",
            "=" * 55
        ]
        
        if summary['significant_results']:
            for result in summary['significant_results']:
                val1, val2 = result['values_compared']
                winner = val1 if result['proportion'] > 0.5 else val2
                report_lines.append(
                    f"✓ {result['description']}: {winner} preferred "
                    f"(effect: {result['effect_size']:.3f}, p: {result['p_value_corrected']:.4f})"
                )
        else:
            report_lines.append("No statistically significant results after correction.")
        
        report_lines.extend([
            "",
            "LARGE PRACTICAL EFFECTS (>30% preference difference):",
            "=" * 60
        ])
        
        if summary['large_effects']:
            for effect in summary['large_effects']:
                val1, val2 = effect['values_compared']
                winner = val1 if effect['proportion'] > 0.5 else val2
                report_lines.append(
                    f"• {effect['description']}: {winner} preferred "
                    f"(effect: {effect['effect_size']:.3f}, proportion: {effect['proportion']:.3f})"
                )
        else:
            report_lines.append("No large practical effects found.")
        
        report_lines.extend([
            "",
            "HYPOTHESIS OVERVIEW:",
            "=" * 25
        ])
        
        for hyp_name, overview in summary['hypothesis_overview'].items():
            sig_indicator = "✓" if overview['significant'] else "✗"
            report_lines.append(
                f"{sig_indicator} {hyp_name}: {overview['description']}"
            )
            report_lines.append(
                f"   Effect: {overview['effect_size']:.3f} ({overview['practical_significance']}), "
                f"N: {overview['n_comparisons']}"
            )
            report_lines.append("")
        
        # Save report
        with open(os.path.join(output_folder, 'focused_hypothesis_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _save_combined_report(self, results: Dict[str, Any], output_folder: str) -> None:
        """Save comprehensive combined report."""
        
        focused_results = results['focused_hypotheses']
        key_results = results['key_preferences']
        transition_results = results['transition_preferences']
        combined_insights = results['combined_insights']
        
        report_lines = [
            "COMBINED KEYBOARD PREFERENCE ANALYSIS REPORT",
            "=" * 70,
            "",
            "This comprehensive analysis combines confirmatory hypothesis testing",
            "with exploratory preference discovery for complete insights.",
            "",
            "EXECUTIVE SUMMARY:",
            "=" * 20,
            ""
        ]
        
        # Executive summary
        focused_summary = focused_results['summary']
        total_significant = len(focused_summary['significant_results'])
        total_large_effects = len(focused_summary['large_effects'])
        
        report_lines.extend([
            f"• Focused Hypotheses: {total_significant}/{focused_results['total_hypotheses']} significant",
            f"• Large Effects Found: {total_large_effects}",
            f"• Keys Analyzed: {len(key_results['overall_rankings'])}",
            f"• Transition Types: {len(transition_results['overall_rankings'])}",
            "",
        ])
        
        # Key findings
        if total_significant > 0:
            report_lines.extend([
                "KEY CONFIRMED PRINCIPLES:",
                "-" * 30
            ])
            for result in focused_summary['significant_results'][:5]:
                val1, val2 = result['values_compared']
                winner = val1 if result['proportion'] > 0.5 else val2
                report_lines.append(
                    f"✓ {result['description']}: {winner} preferred "
                    f"(effect: {result['effect_size']:.3f})"
                )
            report_lines.append("")
        
        # Top discoveries from exploratory analysis
        report_lines.extend([
            "TOP EXPLORATORY DISCOVERIES:",
            "-" * 35,
            "",
            "Most Preferred Keys:"
        ])
        
        for i, (key, strength) in enumerate(key_results['overall_rankings'][:5], 1):
            finger = self.key_positions.get(key, KeyPosition('', 0, 0, 0)).finger
            report_lines.append(f"  {i}. {key.upper()} (F{finger}): {strength:.3f}")
        
        report_lines.extend([
            "",
            "Most Preferred Transitions:"
        ])
        
        for i, (transition, strength) in enumerate(transition_results['overall_rankings'][:5], 1):
            report_lines.append(f"  {i}. {transition}: {strength:.3f}")
        
        # Combined insights
        if 'convergent_findings' in combined_insights and combined_insights['convergent_findings']:
            report_lines.extend([
                "",
                "CONVERGENT FINDINGS (both analyses agree):",
                "-" * 45
            ])
            for finding in combined_insights['convergent_findings'][:3]:
                report_lines.append(f"• {finding}")
        
        if 'practical_recommendations' in combined_insights and combined_insights['practical_recommendations']:
            report_lines.extend([
                "",
                "PRACTICAL RECOMMENDATIONS:",
                "-" * 30
            ])
            for rec in combined_insights['practical_recommendations'][:5]:
                report_lines.append(f"• {rec}")
        
        report_lines.extend([
            "",
            "METHODOLOGY:",
            "=" * 15,
            "",
            "• Focused Analysis: 20 pre-specified hypotheses with global FDR correction",
            "• Exploratory Analysis: Bradley-Terry models with bootstrap confidence intervals",
            "• Cross-Validation: Confirmatory findings validated against exploratory patterns",
            "• Effect Sizes: Practical significance assessed alongside statistical significance",
            "",
            "OUTPUT FILES:",
            "=" * 15,
            "",
            "focused_hypotheses/:",
            "  • focused_hypothesis_results.json - Complete numerical results",
            "  • focused_hypothesis_summary.csv - Tabular summary",
            "  • focused_hypothesis_report.txt - Human-readable findings",
            "",
            "exploratory/:",
            "  • key_preference_statistics.csv - Individual key analysis",
            "  • transition_preference_statistics.csv - Transition analysis",
            "  • key_preference_heatmap.png - Spatial preference visualization",
            "  • key_strengths_with_cis.png - Forest plot with confidence intervals",
            "  • transition_rankings.png - Top transition preferences",
            "",
            "Root directory:",
            "  • complete_combined_analysis.json - Full results from both analyses",
            "  • combined_analysis_report.txt - This comprehensive summary",
            ""
        ])
        
        # Save report
        with open(os.path.join(output_folder, 'combined_analysis_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))
    
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

def main():
    """Main function for command-line usage."""
    # First parse config argument to get defaults
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default='config.yaml',
                              help='Path to configuration file (default: config.yaml)')
    config_args, remaining_args = config_parser.parse_known_args()
    
    # Load config to get default data file
    try:
        if os.path.exists(config_args.config):
            with open(config_args.config, 'r') as f:
                full_config = yaml.safe_load(f)
            data_config = full_config.get('data', {})
            input_dir = data_config.get('input_dir', '')
            filtered_data_file = data_config.get('filtered_data_file', None)
            
            if filtered_data_file and input_dir:
                default_data_file = os.path.join(input_dir, filtered_data_file)
            elif filtered_data_file:
                default_data_file = filtered_data_file
            else:
                default_data_file = None
        else:
            logger.warning(f"Config file not found: {config_args.config}. Using command-line defaults.")
            default_data_file = None
    except Exception as e:
        logger.warning(f"Could not load config file: {e}. Using command-line defaults.")
        default_data_file = None
    
    # Create main parser with config-based defaults
    parser = argparse.ArgumentParser(
        description='Combined keyboard preference analysis: exploratory + focused hypothesis testing'
    )
    parser.add_argument('--data', default=default_data_file,
                       help=f'Path to CSV file with bigram choice data (default: {default_data_file or "required"})')
    parser.add_argument('--output', required=True,
                       help='Output directory for results')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data:
        logger.error("No data file specified. Use --data argument or set filtered_data_file in config.yaml")
        return 1
        
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return 1
    
    try:
        # Run combined analysis
        analyzer = PreferenceAnalyzer(args.config)
        results = analyzer.analyze_preferences(args.data, args.output)
        
        logger.info("Combined analysis completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        
        # Print quick summary
        focused_summary = results['focused_hypotheses']['summary']
        key_rankings = results['key_preferences']['overall_rankings']
        
        print(f"\nQUICK SUMMARY:")
        print(f"==============")
        print(f"Focused Hypotheses: {len(focused_summary['significant_results'])}/{results['focused_hypotheses']['total_hypotheses']} significant")
        print(f"Large Effects: {len(focused_summary['large_effects'])}")
        print(f"Top 3 Keys: {', '.join([k.upper() for k, _ in key_rankings[:3]])}")
        print(f"")
        print(f"See detailed reports in: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())