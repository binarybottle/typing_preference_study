"""
Comprehensive Keyboard Position and Transition Preference Analysis

This standalone script analyzes typing preferences using Bradley-Terry models to rank:
1. Left key preferences (QWERTASDFGZXCVB) - replaces individual finger/column tests
2. Bigram transition type preferences - replaces individual spatial/directional tests

Features:
- Bradley-Terry models for robust ranking from pairwise comparisons
- Multi-level statistical significance testing with effect sizes
- Practical significance thresholds for interpretable results
- Hierarchical confidence assessment (definitive/probable/suggestive/inconclusive)
- Bootstrap validation and sensitivity analysis
- Comprehensive reporting with actionable insights
- Detailed statistical tables with confidence intervals
- Clear visualization of strength differences

Usage:
    python analyze_preferred_positions_transitions.py --data data/filtered_data.csv --output results/ --config config.yaml
"""

import os
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict, Counter

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
    row: int  # 1=upper, 2=middle/home, 3=lower
    column: int  # 1=leftmost, 5=middle
    finger: int  # 1=pinky, 4=index

@dataclass
class TransitionType:
    """Represents a type of bigram transition."""
    name: str
    row_pattern: str  # 'same', 'adjacent', 'hurdle'
    finger_separation: int  # 1, 2, or 3 finger distance
    direction: str  # 'inner_roll', 'outer_roll', 'neutral'

class BradleyTerryModel:
    """
    Bradley-Terry model for ranking items from pairwise comparison data.
    
    Models P(item_i beats item_j) = exp(strength_i) / (exp(strength_i) + exp(strength_j))
    """
    
    def __init__(self, items: List[str], config: Dict[str, Any]):
        self.items = list(items)
        self.n_items = len(items)
        self.item_to_idx = {item: i for i, item in enumerate(items)}
        self.strengths = None
        self.fitted = False
        self.config = config
    
    def fit(self, pairwise_data: Dict[Tuple[str, str], Dict[str, int]]) -> None:
        """
        Fit Bradley-Terry model to pairwise comparison data.
        
        Args:
            pairwise_data: Dict mapping (item1, item2) -> {'wins_item1': int, 'total': int}
        """
        #logger.info(f"Fitting Bradley-Terry model with {len(pairwise_data)} pairwise comparisons")
        #logger.info(f"Sample pairwise data: {dict(list(pairwise_data.items())[:3])}")
        
        # Build comparison matrix
        wins = np.zeros((self.n_items, self.n_items))
        totals = np.zeros((self.n_items, self.n_items))
        
        for (item1, item2), data in pairwise_data.items():
            if item1 in self.item_to_idx and item2 in self.item_to_idx:
                i, j = self.item_to_idx[item1], self.item_to_idx[item2]
                
        # Build comparison matrix
        wins = np.zeros((self.n_items, self.n_items))
        totals = np.zeros((self.n_items, self.n_items))
        
        pair_count = 0
        for (item1, item2), data in pairwise_data.items():
            if item1 in self.item_to_idx and item2 in self.item_to_idx:
                i, j = self.item_to_idx[item1], self.item_to_idx[item2]
                
                try:
                    # Debug the first few data types
                    #if pair_count < 3:
                    #    logger.info(f"Processing pair {pair_count}: ({item1}, {item2}): data = {data}")
                    #    logger.info(f"Data types: wins_item1={type(data['wins_item1'])}, total={type(data['total'])}")
                    
                    # Ensure data values are integers
                    wins_item1 = int(data['wins_item1'])
                    total = int(data['total'])
                    
                    wins[i, j] = wins_item1
                    wins[j, i] = total - wins_item1
                    totals[i, j] = totals[j, i] = total
                    
                    pair_count += 1
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid data for pair ({item1}, {item2}): {data}. Error: {e}")
                    continue
        
        #logger.info("Comparison matrices built, starting ML fitting...")
        # Fit using maximum likelihood
        self.strengths = self._fit_ml(wins, totals)
        self.fitted = True
        #logger.info("Bradley-Terry model fitting completed")
    
    def _fit_ml(self, wins: np.ndarray, totals: np.ndarray) -> np.ndarray:
        """Fit Bradley-Terry model using maximum likelihood estimation."""
        
        def negative_log_likelihood(strengths):
            # Add regularization to prevent overflow - ensure it's numeric
            reg = float(self.config.get('regularization', 1e-10))
            
            strengths = np.clip(strengths, -10, 10)
            ll = 0
            for i in range(self.n_items):
                for j in range(i + 1, self.n_items):
                    if totals[i, j] > 0:
                        p_ij = np.exp(strengths[i]) / (np.exp(strengths[i]) + np.exp(strengths[j]))
                        p_ij = np.clip(p_ij, reg, 1.0 - reg)  # Avoid log(0), ensure 1.0 is float
                        ll += wins[i, j] * np.log(p_ij) + wins[j, i] * np.log(1 - p_ij)
            return -ll
        
        # Initialize with zeros (equal strength)
        initial_strengths = np.zeros(self.n_items)
        
        # Constrain first item strength to 0 for identifiability
        def constraint_func(x):
            return x[0]
        
        constraint = {'type': 'eq', 'fun': constraint_func}
        
        # Optimize
        max_iter = int(self.config.get('max_iterations', 1000))  # Ensure integer
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
    
    def get_win_probabilities(self) -> np.ndarray:
        """Get matrix of win probabilities P(item_i beats item_j)."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting probabilities")
        
        probs = np.zeros((self.n_items, self.n_items))
        for i in range(self.n_items):
            for j in range(self.n_items):
                if i != j:
                    probs[i, j] = np.exp(self.strengths[i]) / (
                        np.exp(self.strengths[i]) + np.exp(self.strengths[j])
                    )
        return probs

class PreferenceAnalyzer:
    """Main class for analyzing keyboard position and transition preferences."""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Define keyboard layout based on config
        self.key_positions = self._define_keyboard_layout()
        self.left_hand_keys = set(self.key_positions.keys())
        
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
        """
        Run complete preference analysis.
        
        Args:
            data_path: Path to CSV file with bigram choice data
            output_folder: Directory to save results
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive preference analysis...")
        
        # Load and validate data
        self.data = self._load_and_validate_data(data_path)
        logger.info(f"Loaded {len(self.data)} rows from {self.data['user_id'].nunique()} participants")
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Run analyses
        results = {}

        # 1. Left-hand key rankings
        logger.info("Analyzing key preferences...")
        results['key_preferences'] = self._analyze_key_preferences(self.data)
        
        # 2. Transition type rankings
        logger.info("Analyzing bigram transition preferences...")
        results['transition_preferences'] = self._analyze_transition_preferences(self.data)
        
        # 3. Generate comprehensive reports and exports
        logger.info("Generating reports and visualizations...")
        if self.config.get('save_raw_results', True):
            self._export_statistical_tables(results, output_folder)
        
        if self.config.get('generate_detailed_report', True):
            self._generate_comprehensive_report(results, output_folder)
        
        if self.config.get('generate_visualizations', True):
            self._create_visualizations(results, output_folder)
        
        # Save full results
        if self.config.get('save_raw_results', True):
            try:
                results_path = os.path.join(output_folder, 'complete_preference_analysis.json')
                with open(results_path, 'w') as f:
                    json_results = self._convert_for_json(results)
                    json.dump(json_results, f, indent=2, default=str)
                logger.info("Full results saved to JSON")
            except Exception as e:
                logger.warning(f"Could not save JSON results: {e}")

        logger.info(f"Analysis complete! Results saved to {output_folder}")
        return results
    
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
        """Analyze preferences for individual home block keys."""
        
        # Extract all pairwise key comparisons
        key_comparisons = self._extract_key_comparisons(data)
        
        # Fit Bradley-Terry model
        bt_model = BradleyTerryModel(list(self.left_hand_keys), self.config)
        bt_model.fit(key_comparisons)
        
        # Get rankings and statistics
        rankings = bt_model.get_rankings()
        win_probs = bt_model.get_win_probabilities()
        
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
            wins1 = int(data['wins_item1'])  # Ensure integer
            total = int(data['total'])       # Ensure integer
            # Add individual comparison records
            for _ in range(wins1):
                comparison_list.append((key1, key2, 1))  # key1 wins
            for _ in range(total - wins1):
                comparison_list.append((key1, key2, 0))  # key2 wins
        
        if not comparison_list:
            # Return empty CIs if no comparisons
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
                # If fitting fails, use original strengths
                bootstrap_strengths[b, :] = bt_model.strengths
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        ci_dict = {}
        for i, item in enumerate(bt_model.items):
            lower = float(np.percentile(bootstrap_strengths[:, i], 100 * alpha/2))
            upper = float(np.percentile(bootstrap_strengths[:, i], 100 * (1 - alpha/2)))
            ci_dict[item] = (lower, upper)
        
        return ci_dict
    
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
            row_pattern = 'reach'  # Adjacent rows (previously "adjacent")
        else:
            row_pattern = 'hurdle'  # Skipping rows (previously "hurdle")
        
        # Finger separation
        finger_separation = abs(pos1.finger - pos2.finger)
        
        # Direction (for same row only)
        # Note: "cross_row" in old system meant any transition between different rows
        # Now we distinguish between "reach" (adjacent rows) and "hurdle" (skip rows)
        if pos1.row == pos2.row:
            if pos2.finger > pos1.finger:
                direction = 'inner_roll'
            elif pos2.finger < pos1.finger:
                direction = 'outer_roll'
            else:
                direction = 'same_finger'
        else:
            direction = 'cross_row'  # Any cross-row transition
        
        # Create new format: Δ#-finger [pattern] using actual delta symbol
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
            'row_patterns': {'same': [], 'adjacent': [], 'hurdle': []},
            'finger_separations': {'1': [], '2': [], '3': []},
            'directions': {'inner_roll': [], 'outer_roll': [], 'cross_row': []}
        }
        
        for transition_type, strength in rankings:
            # Parse transition type
            parts = transition_type.split('_')
            
            # Row pattern
            if 'same' in parts:
                categories['row_patterns']['same'].append((transition_type, strength))
            elif 'adjacent' in parts:
                categories['row_patterns']['adjacent'].append((transition_type, strength))
            elif 'hurdle' in parts:
                categories['row_patterns']['hurdle'].append((transition_type, strength))
            
            # Finger separation
            for sep in ['1', '2', '3']:
                if f'{sep}finger' in transition_type:
                    categories['finger_separations'][sep].append((transition_type, strength))
            
            # Direction
            if 'inner_roll' in parts:
                categories['directions']['inner_roll'].append((transition_type, strength))
            elif 'outer_roll' in parts:
                categories['directions']['outer_roll'].append((transition_type, strength))
            elif 'cross_row' in parts:
                categories['directions']['cross_row'].append((transition_type, strength))
        
        # Sort each category by strength
        for category in categories.values():
            for subcategory in category.values():
                subcategory.sort(key=lambda x: x[1], reverse=True)
        
        return categories

    # =========================================================================
    # STATISTICAL TABLES AND EXPORTS
    # =========================================================================
    
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
        
        # Create specialized transition table with Δfinger, Δrow format
        transition_table = self._create_transition_delta_table(results['transition_preferences'])
        transition_table.to_csv(os.path.join(output_folder, 'transition_delta_analysis.csv'), index=False)
    
    def _create_transition_delta_table(self, transition_results: Dict[str, Any]) -> pd.DataFrame:
        """Create specialized transition table with Δfinger, Δrow, in/out format."""
        
        rankings = transition_results['overall_rankings']
        bt_cis = transition_results['bt_confidence_intervals']
        transition_classifications = transition_results['transition_classifications']
        
        table_data = []
        
        # Create mapping from transition names back to position data
        transition_details = {}
        for bigram, transition_type in transition_classifications.items():
            if len(bigram) >= 2:
                char1, char2 = bigram[0], bigram[1]
                if char1 in self.key_positions and char2 in self.key_positions:
                    pos1 = self.key_positions[char1]
                    pos2 = self.key_positions[char2]
                    
                    finger_delta = abs(pos1.finger - pos2.finger)
                    row_delta = abs(pos1.row - pos2.row)
                    
                    # Determine in/out direction
                    if pos1.row == pos2.row:  # Same row
                        if pos2.finger > pos1.finger:
                            in_out = 1  # Inward roll
                        elif pos2.finger < pos1.finger:
                            in_out = -1  # Outward roll
                        else:
                            in_out = 0  # Same finger
                    else:
                        in_out = 0  # Cross-row transitions don't have in/out
                    
                    transition_details[transition_type] = {
                        'delta_finger': finger_delta,
                        'delta_row': row_delta,
                        'in_out': in_out
                    }
        
        # Process each ranked transition
        for rank, (transition_type, strength) in enumerate(rankings, 1):
            details = transition_details.get(transition_type, {
                'delta_finger': np.nan,
                'delta_row': np.nan,
                'in_out': np.nan
            })
            
            ci_lower, ci_upper = bt_cis.get(transition_type, (np.nan, np.nan))
            ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "[---, ---]"
            
            table_data.append({
                'Rank': rank,
                'Transition_Type': transition_type,
                'Δfinger': details['delta_finger'],
                'Δrow': details['delta_row'],
                'in_out': details['in_out'],
                'BT_Strength': strength,
                'BT_CI_Lower': ci_lower,
                'BT_CI_Upper': ci_upper,
                'BT_CI_String': ci_str
            })
        
        return pd.DataFrame(table_data)

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def _create_visualizations(self, results: Dict[str, Any], output_folder: str) -> None:
        """Create comprehensive visualizations including statistical plots."""
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        dpi = self.config.get('figure_dpi', 300)
        plt.rcParams.update({'font.size': 12, 'figure.dpi': dpi})
        
        # 1. Key preference heatmap by finger
        self._plot_key_preference_heatmap(results['key_preferences'], output_folder)
        
        # 2. Key strength differences with confidence intervals
        self._plot_key_strengths_with_cis(results['key_preferences'], output_folder)
        
        # 3. Transition preference rankings
        self._plot_transition_rankings(results['transition_preferences'], output_folder)
        
        # 4. Transition strength differences with confidence intervals
        self._plot_transition_strengths_with_cis(results['transition_preferences'], output_folder)
        
        # 5. Effect size distributions
        self._plot_effect_size_distributions(results, output_folder)
        
        # 6. Confidence assessment summary
        self._plot_confidence_summary(results, output_folder)
        
        # 7. Statistical comparison plots
        self._plot_statistical_comparisons(results, output_folder)
    
    def _plot_key_strengths_with_cis(self, key_results: Dict[str, Any], output_folder: str) -> None:
        """Create forest plot showing key strengths with confidence intervals."""
        
        rankings = key_results['overall_rankings']
        bt_cis = key_results['bt_confidence_intervals']
        
        if not rankings:
            return
        
        # Prepare data
        keys = [key.upper() for key, _ in rankings]
        strengths = [float(strength) for _, strength in rankings]  # Ensure numeric
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
        
        # Add finger labels - positioned further right and all black
        x_max = max(strengths)
        x_range = max(strengths) - min(strengths)
        finger_x_pos = x_max + (x_range * 0.2)  # 20% beyond the maximum value
        
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
    
    def _plot_transition_strengths_with_cis(self, transition_results: Dict[str, Any], output_folder: str) -> None:
        """Create forest plots showing transition strengths with confidence intervals."""
        
        rankings = transition_results['overall_rankings']
        bt_cis = transition_results['bt_confidence_intervals']
        
        if not rankings:
            return
        
        # Create top 15 plot
        self._create_transition_forest_plot(rankings[:15], bt_cis, output_folder, 
                                          'transition_strengths_with_cis_top15.png',
                                          'Top 15 Transition Type Preferences with Confidence Intervals')
        
        # Create all transitions plot
        self._create_transition_forest_plot(rankings, bt_cis, output_folder,
                                          'transition_strengths_with_cis_all.png', 
                                          'All Transition Type Preferences with Confidence Intervals')
    
    def _create_transition_forest_plot(self, rankings: List[Tuple[str, float]], 
                                     bt_cis: Dict[str, Tuple[float, float]], 
                                     output_folder: str, filename: str, title: str) -> None:
        """Create a single transition forest plot."""
        
        if not rankings:
            return
        
        # Prepare data with proper delta formatting
        transition_types = []
        for t, _ in rankings:
            # Convert to proper format with delta symbol
            formatted_name = t.replace('Δ', 'Δ')  # Ensure proper delta symbol
            transition_types.append(formatted_name)
        
        strengths = [float(strength) for _, strength in rankings]
        ci_lowers = [float(bt_cis.get(t, (np.nan, np.nan))[0]) for t, _ in rankings]
        ci_uppers = [float(bt_cis.get(t, (np.nan, np.nan))[1]) for t, _ in rankings]
        
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
        
        # Determine figure size based on number of items
        height = max(8, len(transition_types) * 0.4)
        fig, ax = plt.subplots(figsize=(14, height))
        
        # Create horizontal error bar plot
        y_pos = np.arange(len(transition_types))
        colors = ['green' if s > 0 else 'red' for s in strengths]
        
        ax.errorbar(strengths, y_pos, xerr=[lower_errs, upper_errs], 
                   fmt='o', color='black', ecolor='gray', capsize=3, capthick=1)
        
        # Color the points
        for i, (strength, color) in enumerate(zip(strengths, colors)):
            ax.scatter(strength, i, c=color, s=60, alpha=0.7, zorder=5)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(transition_types, fontsize=9)
        ax.set_xlabel('Bradley-Terry Strength (95% CI)')
        ax.set_title(title.lower())
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_comparisons(self, results: Dict[str, Any], output_folder: str) -> None:
        """Create plots showing statistical comparisons and effect sizes."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Key strength range comparison
        key_rankings = results['key_preferences']['overall_rankings']
        if key_rankings:
            key_strengths = [strength for _, strength in key_rankings]
            key_cis = results['key_preferences']['bt_confidence_intervals']
            key_ci_widths = []
            for key, strength in key_rankings:
                ci_lower, ci_upper = key_cis.get(key, (strength, strength))
                key_ci_widths.append(ci_upper - ci_lower)
            
            ax1.boxplot([key_strengths], labels=['Keys'])
            ax1.set_ylabel('Bradley-Terry Strength')
            ax1.set_title('distribution of key strengths')
            ax1.grid(True, alpha=0.3)
            
            # Add individual points
            y_vals = np.random.normal(1, 0.04, len(key_strengths))
            colors = ['green' if s > 0 else 'red' for s in key_strengths]
            ax1.scatter(y_vals, key_strengths, c=colors, alpha=0.6, s=50)
        
        # 2. Transition strength range comparison
        transition_rankings = results['transition_preferences']['overall_rankings']
        if transition_rankings:
            transition_strengths = [strength for _, strength in transition_rankings]
            transition_cis = results['transition_preferences']['bt_confidence_intervals']
            
            ax2.boxplot([transition_strengths], labels=['Transitions'])
            ax2.set_ylabel('Bradley-Terry Strength')
            ax2.set_title('distribution of transition strengths')
            ax2.grid(True, alpha=0.3)
            
            # Add individual points
            y_vals = np.random.normal(1, 0.04, len(transition_strengths))
            colors = ['green' if s > 0 else 'red' for s in transition_strengths]
            ax2.scatter(y_vals, transition_strengths, c=colors, alpha=0.6, s=30)
        
        # 3. Confidence interval widths
        if key_rankings:
            ax3.hist(key_ci_widths, bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax3.set_xlabel('95% CI Width')
            ax3.set_ylabel('Frequency')
            ax3.set_title('key preference confidence interval widths')
            ax3.axvline(np.mean(key_ci_widths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(key_ci_widths):.3f}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Statistical significance summary
        key_stats = results['key_preferences']['pairwise_statistics']
        transition_stats = results['transition_preferences']['pairwise_statistics']
        
        # Count significant results
        key_significant = sum(1 for stats in key_stats.values() 
                             if stats.get('significant_corrected', False))
        key_total = len(key_stats)
        
        transition_significant = sum(1 for stats in transition_stats.values() 
                                   if stats.get('significant_corrected', False))
        transition_total = len(transition_stats)
        
        categories = ['Key\nComparisons', 'Transition\nComparisons']
        significant_counts = [key_significant, transition_significant]
        total_counts = [key_total, transition_total]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x_pos - width/2, total_counts, width, label='Total', color='lightblue', alpha=0.7)
        ax4.bar(x_pos + width/2, significant_counts, width, label='Significant', color='darkblue', alpha=0.7)
        
        ax4.set_ylabel('Number of Comparisons')
        ax4.set_title('statistical significance summary')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (sig, total) in enumerate(zip(significant_counts, total_counts)):
            if total > 0:
                pct = sig / total * 100
                ax4.text(i, max(total_counts) * 0.9, f'{pct:.1f}%', 
                        ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'statistical_comparisons.png'), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
    def _plot_key_preference_heatmap(self, key_results: Dict[str, Any], output_folder: str) -> None:
        """Create heatmap showing key preferences using row and column for unique key positioning."""
        
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
            
    def _plot_transition_rankings(self, transition_results: Dict[str, Any], output_folder: str) -> None:
        """Create bar plot of transition type rankings."""
        
        rankings = transition_results['overall_rankings']
        if not rankings:
            return
        
        # Prepare data (top 15 for readability)
        top_rankings = rankings[:15]
        # Format transition names with proper delta symbols
        transition_types = []
        for t, _ in top_rankings:
            formatted_name = t.replace('Δ', 'Δ')  # Ensure proper delta symbol
            transition_types.append(formatted_name)
        
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
    
    def _plot_confidence_summary(self, results: Dict[str, Any], output_folder: str) -> None:
        """Create summary plot of confidence levels."""
        
        # Collect confidence levels
        finger_confidences = []
        for finger in range(1, 5):
            finger_key = f'finger_{finger}'
            if finger_key in results['key_preferences']['finger_specific_results']:
                confidence = results['key_preferences']['finger_specific_results'][finger_key]['confidence_level']
                finger_confidences.append(confidence)
        
        # Count confidence levels
        confidence_counts = Counter(finger_confidences)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        confidence_levels = ['definitive', 'probable', 'suggestive', 'inconclusive', 'insufficient_data']
        counts = [confidence_counts.get(level, 0) for level in confidence_levels]
        colors = ['green', 'orange', 'yellow', 'red', 'gray']
        
        bars = ax.bar(confidence_levels, counts, color=colors, alpha=0.7)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       str(count), ha='center', va='bottom')
        
        ax.set_ylabel('Number of Finger Rankings')
        ax.set_title('confidence levels in finger key rankings')
        ax.set_ylim(0, max(counts) + 0.5 if counts else 1)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'confidence_summary.png'), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
    def _generate_comprehensive_report(self, results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive text report with detailed statistical tables."""
        
        report_lines = [
            "comprehensive keyboard preference analysis report",
            "=" * 70,
            "",
            "executive summary",
            "================",
            "",
            f"This analysis used Bradley-Terry models to rank keyboard positions and",
            f"transition types based on typing preference data. The approach provides",
            f"statistically rigorous rankings with effect sizes and confidence assessments.",
            "",
        ]
        
        # Key preference results with detailed table
        report_lines.extend([
            "detailed key preference statistics",
            "=================================",
            ""
        ])
        
        # Create comprehensive table
        key_table = self._create_comprehensive_key_table(results['key_preferences'])
        
        # Format table for text report
        report_lines.append("key performance summary:")
        report_lines.append("-" * 125)
        report_lines.append(f"{'Rank':<4} {'Key':<3} {'Row':<3} {'Col':<3} {'Fngr':<4} "
                           f"{'BT Strength':<12} {'95% CI':<20} {'Win Rate':<9} "
                           f"{'Effect Size':<11} {'N Comp':<7} {'% Sig':<6} {'Level':<15}")
        report_lines.append("-" * 125)
        
        for _, row in key_table.iterrows():
            ci_str = f"[{row['BT_CI_Lower']:.3f}, {row['BT_CI_Upper']:.3f}]"
            report_lines.append(
                f"{row['Rank']:<4} {row['Key']:<3} {row['Row']:<3} {row['Column']:<3} "
                f"{row['Finger']:<4} {row['BT_Strength']:<12.3f} {ci_str:<20} "
                f"{row['Mean_Win_Rate']:<9.3f} {row['Mean_Effect_Size']:<11.3f} "
                f"{row['Total_Comparisons']:<7.0f} {row['Pct_Significant']:<6.1f} "
                f"{row['Preference_Level']:<15}"
            )
        
        report_lines.extend([
            "-" * 125,
            "",
            "column definitions:",
            "  BT Strength: Bradley-Terry latent preference parameter (higher = more preferred)",
            "  95% CI: Bootstrap confidence interval for BT strength",
            "  Win Rate: Average proportion of pairwise comparisons won by this key",
            "  Effect Size: Average effect size across all pairwise comparisons",
            "  N Comp: Total number of pairwise comparisons involving this key",
            "  % Sig: Percentage of pairwise comparisons that are statistically significant",
            "",
        ])
        
        # Finger-specific results
        report_lines.extend([
            "finger-specific key rankings",
            "============================",
            ""
        ])
        
        key_results = results['key_preferences']
        for finger in range(1, 5):
            finger_key = f'finger_{finger}'
            if finger_key in key_results['finger_specific_results']:
                finger_data = key_results['finger_specific_results'][finger_key]
                report_lines.extend([
                    f"finger {finger} results:",
                    f"  {finger_data['interpretation']}",
                    f"  confidence level: {finger_data['confidence_level'].upper()}",
                    ""
                ])
        
        # Transition preference results
        report_lines.extend([
            "transition preference results", 
            "============================",
            ""
        ])
        
        transition_results = results['transition_preferences']
        rankings = transition_results['overall_rankings']
        
        if rankings:
            report_lines.extend([
                "top 10 preferred transitions:",
                "-----------------------------"
            ])
            for i, (transition_type, strength) in enumerate(rankings[:10]):
                ci_lower, ci_upper = transition_results['bt_confidence_intervals'].get(
                    transition_type, (np.nan, np.nan)
                )
                ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "[---, ---]"
                report_lines.append(
                    f"{i+1:2d}. {transition_type.replace('_', ' ').title():<35} "
                    f"strength: {strength:6.3f} CI: {ci_str}"
                )
            
            report_lines.extend([
                "",
                "bottom 5 transitions:",
                "--------------------"
            ])
            for i, (transition_type, strength) in enumerate(rankings[-5:]):
                ci_lower, ci_upper = transition_results['bt_confidence_intervals'].get(
                    transition_type, (np.nan, np.nan)
                )
                ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "[---, ---]"
                rank = len(rankings) - 4 + i
                report_lines.append(
                    f"{rank:2d}. {transition_type.replace('_', ' ').title():<35} "
                    f"strength: {strength:6.3f} CI: {ci_str}"
                )
        
        # Statistical summary
        report_lines.extend([
            "",
            "statistical summary",
            "==================",
            ""
        ])
        
        # Key statistics
        key_stats = key_results['pairwise_statistics']
        key_effect_sizes = [stats['effect_size'] for stats in key_stats.values() if not np.isnan(stats['effect_size'])]
        key_significant = sum(1 for stats in key_stats.values() if stats.get('significant_corrected', False))
        
        report_lines.extend([
            f"key preferences:",
            f"  total pairwise comparisons: {len(key_stats)}",
            f"  statistically significant: {key_significant} ({key_significant/len(key_stats)*100:.1f}%)" if key_stats else "  no comparisons available",
            f"  mean effect size: {np.mean(key_effect_sizes):.3f}" if key_effect_sizes else "  no effect sizes calculated",
            f"  effect size range: {min(key_effect_sizes):.3f} - {max(key_effect_sizes):.3f}" if key_effect_sizes else "",
            ""
        ])
        
        # Transition statistics
        transition_stats = transition_results['pairwise_statistics']
        transition_effect_sizes = [stats['effect_size'] for stats in transition_stats.values() if not np.isnan(stats['effect_size'])]
        transition_significant = sum(1 for stats in transition_stats.values() if stats.get('significant_corrected', False))
        
        report_lines.extend([
            f"transition preferences:",
            f"  total pairwise comparisons: {len(transition_stats)}",
            f"  statistically significant: {transition_significant} ({transition_significant/len(transition_stats)*100:.1f}%)" if transition_stats else "  no comparisons available",
            f"  mean effect size: {np.mean(transition_effect_sizes):.3f}" if transition_effect_sizes else "  no effect sizes calculated",
            f"  effect size range: {min(transition_effect_sizes):.3f} - {max(transition_effect_sizes):.3f}" if transition_effect_sizes else "",
            ""
        ])
        
        # Configuration summary
        report_lines.extend([
            "configuration summary",
            "====================",
            "",
            f"analyzed keys: {self.config.get('analyze_keys', 'left_hand_15')}",
            f"bootstrap iterations: {self.config.get('bootstrap_iterations', 1000)}",
            f"confidence level: {self.config.get('confidence_level', 0.95)}",
            f"alpha level: {self.config.get('alpha_level', 0.05)}",
            f"correction method: {self.config.get('correction_method', 'fdr_bh')}",
            f"min key comparisons: {self.config.get('min_key_comparisons', 5)}",
            f"min transition comparisons: {self.config.get('min_transition_comparisons', 10)}",
            "",
        ])
        
        # Methodology notes
        report_lines.extend([
            "methodology notes",
            "================",
            "",
            "• Bradley-Terry models provide optimal rankings from pairwise comparison data",
            "• Multiple comparison correction applied using False Discovery Rate (FDR)",
            "• Bootstrap confidence intervals (1000 resamples) for Bradley-Terry strengths",
            "• Effect sizes calculated as deviations from chance (0.5)",
            "• Confidence levels: definitive > probable > suggestive > inconclusive",
            "• Practical significance thresholds: small (5-15%), medium (15-30%), large (>30%)",
            "",
            "files generated",
            "===============",
            "",
            "statistical tables:",
            "• key_preference_statistics.csv - complete key analysis with CIs",
            "• pairwise_key_comparisons.csv - all pairwise key comparison details",
            "• transition_preference_statistics.csv - transition rankings with CIs",
            "• complete_preference_analysis.json - full numerical results",
            "",
            "visualizations:",
            "• key_preference_heatmap.png - key preferences by keyboard position",
            "• key_strengths_with_cis.png - forest plot of key strengths",
            "• transition_rankings.png - top transition preferences",
            "• transition_strengths_with_cis.png - forest plot of transition strengths",
            "• statistical_comparisons.png - statistical summary comparisons",
            "• effect_size_distributions.png - distribution of effect sizes",
            "• confidence_summary.png - confidence level assessment",
            ""
        ])
        
        # Save report
        report_path = os.path.join(output_folder, 'comprehensive_preference_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

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
            print("Loading filtered data file: ", os.path.join(input_dir, filtered_data_file))

            if filtered_data_file and input_dir:
                # Join input_dir with filtered_data_file
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
        description='Analyze keyboard position and transition preferences using Bradley-Terry models'
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
        # Run analysis
        analyzer = PreferenceAnalyzer(args.config)
        results = analyzer.analyze_preferences(args.data, args.output)
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())