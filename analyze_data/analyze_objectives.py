#!/usr/bin/env python3
"""
Multi-Objective Optimization (MOO) Objectives Analysis for Keyboard Layout Design

This script identifies and quantifies typing mechanics objectives from bigram preference data
to inform multi-objective keyboard layout optimization. The analysis emphasizes practical 
significance through effect sizes and confidence intervals, with statistical tests providing
supporting evidence for detectability.

Approach: Exploratory analysis for MOO design
- Effect size focus with confidence intervals for uncertainty quantification
- Statistical tests confirm preferences are detectable above chance/noise
- No corrections across objectives (each addresses distinct typing mechanics)
- Results inform engineering decisions rather than definitive scientific claims

MOO Objectives Analyzed:
1. Key preferences: Individual key quality using complementary approaches
   - Same-letter bigram comparisons (Bradley-Terry model)
   - Bigram pair comparisons
2. Row separation: Preferences across keyboard rows (same > reach > hurdle)  
3. Column separation (each with a shared key constraint): 
   - Same vs. other (no row control)
   - Adjacent vs. distant (with row controls, where distant includes both 2 and 3 column separations)
   - Plus reach vs hurdle controlled for column patterns
4. Inward vs outward roll: Preference for finger movement direction using same key pairs
   - Constrained comparison: same two keys in both directions (e.g., 'df' vs 'fd')
   - Inward roll: increasing finger numbers (pinky → index)
   - Outward roll: decreasing finger numbers (index → pinky)
   - Excludes same-column bigrams to ensure roll motion is possible
5. Same row, side reach preferences (no same-column key-pairs):
   - Purest test of side reach cost: within-row, standard area vs lateral extension
   - Eliminates row separation and finger coordination confounds

Usage:
    poetry run python3 analyze_objectives.py --data output/nonProlific/process_data/tables/processed_consistent_choices.csv \
        --output output/nonProlific/analyze_objectives
    poetry run python3 analyze_objectives.py --data output/Prolific/process_data_TGB/tables/processed_consistent_choices.csv \
        --output output/Prolific/analyze_objectives_TGB
"""

import os
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, bootstrap
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

min_threshold = 3

@dataclass
class KeyPosition:
    """Represents a key's position on the keyboard."""
    key: str
    row: int      # 1=upper, 2=home, 3=lower
    column: int   # 1=leftmost, 4=index
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
        
        for (item1, item2), data in pairwise_data.items():
            if item1 in self.item_to_idx and item2 in self.item_to_idx:
                i, j = self.item_to_idx[item1], self.item_to_idx[item2]
                
                wins_item1 = int(data['wins_item1'])
                total = int(data['total'])
                
                wins[i, j] = wins_item1
                wins[j, i] = total - wins_item1
                totals[i, j] = totals[j, i] = total
        
        self.strengths = self._fit_ml(wins, totals)
        self.fitted = True
    
    def _fit_ml(self, wins: np.ndarray, totals: np.ndarray) -> np.ndarray:
        """Bradley-Terry model fitting with numerical stability."""
        
        def negative_log_likelihood_regularized(strengths):
            # More substantial regularization
            reg = float(self.config.get('regularization', 1e-4))
            
            # Clip strengths to prevent numerical overflow
            strengths = np.clip(strengths, -10, 10)
            
            ll = 0
            for i in range(self.n_items):
                for j in range(i + 1, self.n_items):
                    if totals[i, j] > 0:
                        # More numerically stable computation
                        strength_diff = strengths[i] - strengths[j]
                        
                        # Use log-sum-exp trick for stability
                        if strength_diff > 0:
                            p_ij = 1.0 / (1.0 + np.exp(-strength_diff))
                        else:
                            exp_diff = np.exp(strength_diff)
                            p_ij = exp_diff / (1.0 + exp_diff)
                        
                        # Apply regularization to probabilities
                        p_ij = np.clip(p_ij, reg, 1.0 - reg)
                        
                        ll += wins[i, j] * np.log(p_ij) + wins[j, i] * np.log(1 - p_ij)
            
            # Add L2 regularization on parameters
            l2_penalty = reg * np.sum(strengths**2)
            return -(ll - l2_penalty)
        
        # Check for disconnected components
        self._validate_connectivity(wins, totals)
        
        # Try multiple initialization strategies
        best_result = None
        best_likelihood = np.inf
        
        init_strategies = [
            np.zeros(self.n_items),  # All zeros
            np.random.normal(0, 0.1, self.n_items),  # Small random
            self._initialize_from_win_rates(wins, totals)  # Data-driven
        ]
        
        for init_strengths in init_strategies:
            # Use sum-to-zero constraint for better stability
            constraint = {'type': 'eq', 'fun': lambda x: np.sum(x)}
            
            try:
                # Try L-BFGS-B first (often better for this problem)
                result = minimize(
                    negative_log_likelihood_regularized, 
                    init_strengths,
                    method='L-BFGS-B',
                    bounds=[(-10, 10)] * self.n_items
                )
                
                # Apply sum-to-zero constraint post-hoc
                if result.success:
                    constrained_strengths = result.x - np.mean(result.x)
                    likelihood = negative_log_likelihood_regularized(constrained_strengths)
                    
                    if likelihood < best_likelihood:
                        best_likelihood = likelihood
                        best_result = constrained_strengths
                
                # Fallback to SLSQP with constraint if L-BFGS-B fails
                if not result.success:
                    result = minimize(
                        negative_log_likelihood_regularized,
                        init_strengths,
                        method='SLSQP',
                        constraints=constraint
                    )
                    
                    if result.success and negative_log_likelihood_regularized(result.x) < best_likelihood:
                        best_result = result.x
                        
            except Exception as e:
                logger.warning(f"Optimization failed with initialization {type(init_strengths)}: {e}")
                continue
        
        if best_result is None:
            logger.warning("All Bradley-Terry optimizations failed, using simple approximation")
            return self._fallback_estimation(wins, totals)
        
        return best_result

    def _validate_connectivity(self, wins: np.ndarray, totals: np.ndarray) -> None:
        """Check if comparison graph is connected."""        
        G = nx.Graph()
        G.add_nodes_from(range(self.n_items))
        
        for i in range(self.n_items):
            for j in range(i + 1, self.n_items):
                if totals[i, j] > 0:
                    G.add_edge(i, j)
        
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            logger.warning(f"Comparison graph has {len(components)} disconnected components")
            # Could handle by fitting separate models for each component

    def _initialize_from_win_rates(self, wins: np.ndarray, totals: np.ndarray) -> np.ndarray:
        """Initialize strengths based on empirical win rates."""
        win_rates = np.zeros(self.n_items)
        
        for i in range(self.n_items):
            total_games = 0
            total_wins = 0
            
            for j in range(self.n_items):
                if i != j and totals[i, j] > 0:
                    total_games += totals[i, j]
                    total_wins += wins[i, j]
            
            if total_games > 0:
                rate = total_wins / total_games
                # Convert win rate to log-odds (strength estimate)
                rate = np.clip(rate, 0.01, 0.99)  # Avoid extremes
                win_rates[i] = np.log(rate / (1 - rate))
        
        # Center around zero
        return win_rates - np.mean(win_rates)

    def _fallback_estimation(self, wins: np.ndarray, totals: np.ndarray) -> np.ndarray:
        """Fallback method when optimization fails."""
        logger.warning("Using fallback Bradley-Terry estimation")
        return self._initialize_from_win_rates(wins, totals)

    def get_rankings(self) -> List[Tuple[str, float]]:
        """Get items ranked by strength (highest to lowest)."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting rankings")
        
        ranked_indices = np.argsort(self.strengths)[::-1]
        return [(self.items[i], self.strengths[i]) for i in ranked_indices]

class MOOObjectiveAnalyzer:
    """Analyzer for extracting MOO objectives from bigram preference data."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.key_positions = self._define_keyboard_layout()
        self.left_hand_keys = set(self.key_positions.keys())
        self.data = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('moo_objectives_analysis', {})
        else:
            return {
                'bootstrap_iterations': 1000,
                'confidence_level': 0.95,
                'figure_dpi': 300,
                'regularization': 1e-4,
                'frequency_weighting_method': 'inverse_frequency'
            }
    
    def _load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate the input data."""
        data = pd.read_csv(data_path)
        
        # Check required columns
        required_cols = ['user_id', 'chosen_bigram', 'unchosen_bigram', 'sliderValue']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert sliderValue to numeric
        data['sliderValue'] = pd.to_numeric(data['sliderValue'], errors='coerce')
        
        # Remove rows with invalid sliderValue
        data = data.dropna(subset=['sliderValue'])
        
        # Filter to consistent choices (non-zero slider values)
        initial_len = len(data)
        data = data[data['sliderValue'] != 0].copy()
        logger.info(f"Using {len(data)}/{initial_len} consistent choice rows")
        
        # Convert bigrams to lowercase
        data['chosen_bigram'] = data['chosen_bigram'].astype(str).str.lower()
        data['unchosen_bigram'] = data['unchosen_bigram'].astype(str).str.lower()
        
        return data

    def _define_keyboard_layout(self) -> Dict[str, KeyPosition]:
        """Define left-hand keyboard layout (columns 1-4 only)."""
        return {
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

    def _get_column_5_keys(self) -> Set[str]:
        """Get column 5 keys for column 4 vs 5 analysis."""
        return {'t', 'g', 'b'}

    def _apply_frequency_weights(self, instances_df: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency weighting to balance comparison frequencies."""
        if instances_df.empty:
            return instances_df
        
        # Create unique comparison identifier
        if 'comparison' not in instances_df.columns:
            instances_df['comparison'] = instances_df.apply(
                lambda row: tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']])), axis=1
            )
        
        # Calculate frequency weights
        freq_counts = instances_df['comparison'].value_counts()
        method = self.config.get('frequency_weighting_method', 'inverse_frequency')
        
        if method == 'inverse_frequency':
            instances_df['frequency_weight'] = instances_df['comparison'].map(lambda x: 1.0 / freq_counts[x])
        else:
            instances_df['frequency_weight'] = 1.0
        
        return instances_df

    def analyze_moo_objectives(self, data_path: str, output_folder: str, sample_size: int = None) -> Dict[str, Any]:
        """Run complete MOO objectives analysis focusing on practical significance."""
        logger.info("Starting MOO objectives analysis for practical significance...")
        
        # Load and validate data
        self.data = self._load_and_validate_data(data_path)

        # Apply sampling if requested
        if sample_size and sample_size < len(self.data):
            logger.info(f"Sampling {sample_size} rows from {len(self.data)} total rows")
            self.data = self.data.sample(n=sample_size, random_state=42).reset_index(drop=True)
            self.config['bootstrap_iterations'] = min(100, self.config.get('bootstrap_iterations', 1000))

        logger.info(f"Analyzing {len(self.data)} rows from {self.data['user_id'].nunique()} participants")

        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Run each objective analysis
        results = {}

        logger.info("=== KEY PREFERENCES: SAME-LETTER BIGRAMS ===")  
        results['same_letter_preferences'] = self._analyze_same_letter_preferences()

        logger.info("=== KEY PREFERENCES: PAIRWISE COMPARISONS ===")
        results['pairwise_preferences'] = self._analyze_pairwise_preferences()

        logger.info("=== ROW SEPARATION PREFERENCES ===")
        results['row_separation'] = self._analyze_row_separation()

        logger.info("=== COLUMN SEPARATION PREFERENCES ===")  
        results['column_separation'] = self._analyze_column_separation()

        logger.info("=== INWARD VS OUTWARD ROLL PREFERENCES (CONSTRAINED) ===")
        results['inward_outward_roll'] = self._analyze_inward_outward_roll()

        logger.info("=== SIDE REACH PREFERENCES (SAME-ROW ONLY) ===")
        results['side_reach'] = self._analyze_side_reach()

        # Generate reports and save results
        enhanced_results = {
            'objectives': results,
            'summary': self._generate_summary(results)
        }
        
        logger.info("=== GENERATING REPORTS ===")
        self._generate_comprehensive_report(enhanced_results, output_folder)

        # Save detailed key preference comparison tables
        self._save_key_preference_tables(enhanced_results['objectives'], output_folder)
        
        # Generate comprehensive diagnostic CSV
        logger.info("=== GENERATING COMPREHENSIVE DIAGNOSTIC CSV ===")
        self._generate_comprehensive_diagnostic_csv(enhanced_results['objectives'], output_folder)
        
        logger.info(f"Results saved to {output_folder}")
        
        # Generate visualizations
        self._create_visualizations(results, output_folder)
        
        logger.info(f"MOO objectives analysis complete! Results saved to {output_folder}")
        return enhanced_results

    # =========================================================================
    # KEY PREFERENCES: SAME-LETTER BIGRAMS (BRADLEY-TERRY MODEL)
    # =========================================================================
    
    def _analyze_same_letter_preferences(self) -> Dict[str, Any]:
        """Analyze key preferences using only same-letter bigram comparisons."""
        
        instances_df = self._extract_same_letter_instances()
        
        if instances_df.empty:
            raise ValueError('No same-letter instances found')
        
        logger.info(f"Found {len(instances_df)} same-letter instances from {instances_df['user_id'].nunique()} users")
        
        # Build pairwise preference matrix
        key_comparisons = defaultdict(lambda: {'wins_item1': 0, 'total': 0})
        
        for _, row in instances_df.iterrows():
            key1, key2 = row['key1'], row['key2']
            pair = tuple(sorted([key1, key2]))
            
            key_comparisons[pair]['total'] += 1
            if row['chose_key1'] == 1 and key1 == pair[0]:
                key_comparisons[pair]['wins_item1'] += 1
            elif row['chose_key1'] == 0 and key2 == pair[0]:
                key_comparisons[pair]['wins_item1'] += 1
        
        # Fit Bradley-Terry model
        available_keys = set()
        for pair in key_comparisons.keys():
            available_keys.update(pair)
        
        bt_model = BradleyTerryModel(list(available_keys), self.config)
        bt_model.fit(key_comparisons)
        rankings = bt_model.get_rankings()
        
        # Calculate confidence intervals
        confidence_intervals = self._bootstrap_bt_confidence_intervals(bt_model, key_comparisons)
        
        return {
            'description': 'Key preferences from same-letter bigram comparisons only (pure key quality)',
            'method': 'same_letter_bradley_terry',
            'rankings': rankings,
            'confidence_intervals': confidence_intervals,
            'key_comparisons': dict(key_comparisons),
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique(),
            'interpretation': f"Pure key quality rankings from {len(key_comparisons)} key pairs"
        }

    def _extract_same_letter_instances(self) -> pd.DataFrame:
        """Extract same-letter bigram comparisons (AA vs QQ, etc.)."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            # Check if both are same-letter bigrams
            chosen_is_same = len(chosen) == 2 and chosen[0] == chosen[1] and chosen[0] in self.left_hand_keys
            unchosen_is_same = len(unchosen) == 2 and unchosen[0] == unchosen[1] and unchosen[0] in self.left_hand_keys
            
            if chosen_is_same and unchosen_is_same:
                key1 = chosen[0]
                key2 = unchosen[0]
                
                if key1 != key2:
                    instances.append({
                        'user_id': row['user_id'],
                        'chosen_bigram': chosen,
                        'unchosen_bigram': unchosen,
                        'key1': key1,
                        'key2': key2,
                        'chose_key1': 1,
                        'slider_value': row.get('sliderValue', 0)
                    })
        
        return pd.DataFrame(instances)

    def _bootstrap_bt_confidence_intervals(self, bt_model: BradleyTerryModel, 
                                         comparison_data: Dict) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for Bradley-Terry strengths."""
        
        confidence = self.config.get('confidence_level', 0.95)
        n_bootstrap = self.config.get('bootstrap_iterations', 1000)
        
        # Convert to bootstrap format
        comparison_records = []
        for (key1, key2), data in comparison_data.items():
            wins1 = int(data['wins_item1'])
            total = int(data['total'])
            
            for _ in range(wins1):
                comparison_records.append((key1, key2, key1))
            for _ in range(total - wins1):
                comparison_records.append((key1, key2, key2))
        
        if not comparison_records:
            return {item: (np.nan, np.nan) for item in bt_model.items}
        
        bootstrap_strengths = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            n_records = len(comparison_records)
            resampled_indices = np.random.choice(n_records, size=n_records, replace=True)
            
            # Reconstruct comparison counts
            bootstrap_comparisons = defaultdict(lambda: {'wins_item1': 0, 'total': 0})
            
            for idx in resampled_indices:
                key1, key2, winner = comparison_records[idx]
                pair = (key1, key2)
                
                bootstrap_comparisons[pair]['total'] += 1
                if winner == key1:
                    bootstrap_comparisons[pair]['wins_item1'] += 1
            
            # Fit bootstrap model
            bootstrap_model = BradleyTerryModel(bt_model.items, self.config)
            bootstrap_model.fit(dict(bootstrap_comparisons))
            
            if bootstrap_model.fitted:
                bootstrap_strengths.append(bootstrap_model.strengths)
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        ci_dict = {}
        
        if bootstrap_strengths:
            bootstrap_array = np.array(bootstrap_strengths)
            for i, item in enumerate(bt_model.items):
                lower = np.percentile(bootstrap_array[:, i], 100 * alpha/2)
                upper = np.percentile(bootstrap_array[:, i], 100 * (1 - alpha/2))
                ci_dict[item] = (float(lower), float(upper))
        else:
            ci_dict = {item: (np.nan, np.nan) for item in bt_model.items}
        
        return ci_dict

    # =========================================================================
    # KEY PREFERENCES: PAIRWISE COMPARISONS
    # =========================================================================
    
    def _analyze_pairwise_preferences(self) -> Dict[str, Any]:
        """Analyze specific pairwise key comparisons."""
        
        # Define key pairs for analysis (selected from earlier analysis of Prolific data)
        target_pairs = [
            ('f','d'), ('d','s'), ('s','a'),  # Home row
            ('r','e'), ('w','q'),  # Top row  
            ('c','x'), ('x','z'),  # Bottom row
            ('f','r'), ('f','v'), ('d','e'), ('s','w'), ('a','q'), ('a','z'),  # Vertical reach
            ('r','v'), ('w','x'), ('q','z'),  # Vertical hurdle
            ('f','e'), ('d','r'), ('s','r'), ('s','e'), ('a','w'), ('d','v'), ('s','v'),  # Angle reach
            ('e','v'), ('w','c'), ('q','x'), ('q','c'), ('z','w')  # Angle hurdle
        ]
        
        # Filter to existing keys
        valid_pairs = [(k1, k2) for k1, k2 in target_pairs if k1 in self.left_hand_keys and k2 in self.left_hand_keys]
        
        pairwise_results = {}
        
        for key1, key2 in valid_pairs:
            comparison_data = self._extract_specific_key_comparison(key1, key2)
            if comparison_data and comparison_data['n_instances'] >= min_threshold:
                pairwise_results[(key1, key2)] = comparison_data
        
        return {
            'description': 'Specific pairwise key comparisons for detailed analysis',
            'method': 'pairwise_key_tests',
            'n_pairs_tested': len(pairwise_results),
            'pairwise_results': pairwise_results,
            'interpretation': f"Analyzed {len(pairwise_results)} key pairs"
        }
    
    def _extract_specific_key_comparison(self, key1: str, key2: str) -> Dict[str, Any]:
        """Extract comparison data for a specific key pair."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            # Check if this comparison involves both target keys
            chosen_has_key1 = key1 in chosen
            chosen_has_key2 = key2 in chosen
            unchosen_has_key1 = key1 in unchosen
            unchosen_has_key2 = key2 in unchosen
            
            # Want comparisons where one bigram has key1 and the other has key2
            if (chosen_has_key1 and unchosen_has_key2 and not chosen_has_key2 and not unchosen_has_key1):
                instances.append({'chose_key1': 1})
            elif (chosen_has_key2 and unchosen_has_key1 and not chosen_has_key1 and not unchosen_has_key2):
                instances.append({'chose_key1': 0})
        
        if not instances:
            return None
        
        instances_df = pd.DataFrame(instances)
        
        # Calculate preference rate and confidence interval
        n_instances = len(instances_df)
        n_chose_key1 = instances_df['chose_key1'].sum()
        preference_rate = n_chose_key1 / n_instances
        
        # Wilson score confidence interval
        confidence = self.config.get('confidence_level', 0.95)
        ci_lower, ci_upper = self._wilson_ci(n_chose_key1, n_instances, confidence)
        
        # Simple proportion test
        z_score = (preference_rate - 0.5) / np.sqrt(0.25 / n_instances)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return {
            'key1': key1,
            'key2': key2,
            'n_instances': n_instances,
            'preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'z_score': z_score,
            'p_value': p_value
        }

    # =========================================================================
    # ROW SEPARATION PREFERENCES
    # =========================================================================
    
    def _analyze_row_separation(self) -> Dict[str, Any]:
        """Analyze row separation preferences."""
        
        instances_df = self._extract_row_separation_instances()
        
        if instances_df.empty:
            raise ValueError('No row separation instances found')
        
        logger.info(f"Found {len(instances_df)} row separation instances")
        
        # Overall analysis
        n_instances = len(instances_df)
        n_chose_smaller = instances_df['chose_smaller_separation'].sum()
        preference_rate = n_chose_smaller / n_instances
        
        confidence = self.config.get('confidence_level', 0.95)
        ci_lower, ci_upper = self._wilson_ci(n_chose_smaller, n_instances, confidence)
        
        # Statistical test
        z_score = (preference_rate - 0.5) / np.sqrt(0.25 / n_instances)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        # Analysis by comparison type
        comparison_results = {}
        for comp_type in instances_df['comparison_type'].unique():
            comp_data = instances_df[instances_df['comparison_type'] == comp_type]
            if len(comp_data) >= min_threshold:
                n_comp = len(comp_data)
                n_chose_comp = comp_data['chose_smaller_separation'].sum()
                pref_rate_comp = n_chose_comp / n_comp
                ci_lower_comp, ci_upper_comp = self._wilson_ci(n_chose_comp, n_comp, confidence)
                
                z_comp = (pref_rate_comp - 0.5) / np.sqrt(0.25 / n_comp)
                p_comp = 2 * (1 - norm.cdf(abs(z_comp)))
                
                comparison_results[comp_type] = {
                    'n_instances': n_comp,
                    'preference_rate': pref_rate_comp,
                    'ci_lower': ci_lower_comp,
                    'ci_upper': ci_upper_comp,
                    'p_value': p_comp
                }
                logger.info(f"Row separation - {comp_type}: {n_comp} instances, {pref_rate_comp:.1%} prefer smaller")
            else:
                logger.info(f"Skipped row separation - {comp_type}: only {len(comp_data)} instances (need >= min_threshold)")
        
        return {
            'description': 'Preferences for smaller row separation distances',
            'method': 'row_separation_analysis',
            'n_instances': n_instances,
            'preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'comparison_results': comparison_results,  # This provides the detailed breakdown
            'interpretation': f"Row separation preference: {preference_rate:.1%} favor smaller distances"
        }

    def _extract_row_separation_instances(self) -> pd.DataFrame:
        """Extract row separation comparison instances."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
                
            chosen_row_sep = self._calculate_row_separation(chosen)
            unchosen_row_sep = self._calculate_row_separation(unchosen)
            
            # Specific comparisons only
            if {chosen_row_sep, unchosen_row_sep} == {0, 1}:
                comparison_type = "same_row_vs_1_apart"
                chose_smaller = 1 if chosen_row_sep == 0 else 0
            elif {chosen_row_sep, unchosen_row_sep} == {1, 2}:
                comparison_type = "1_apart_vs_2_apart"
                chose_smaller = 1 if chosen_row_sep == 1 else 0
            else:
                continue
            
            instances.append({
                'user_id': row['user_id'],
                'chosen_bigram': chosen,
                'unchosen_bigram': unchosen,
                'chose_smaller_separation': chose_smaller,
                'chosen_row_separation': chosen_row_sep,
                'unchosen_row_separation': unchosen_row_sep,
                'comparison_type': comparison_type
            })
        
        return pd.DataFrame(instances)

    def _calculate_row_separation(self, bigram: str) -> int:
        """Calculate row separation for a bigram."""
        if len(bigram) != 2:
            return 0
        
        key1, key2 = bigram[0], bigram[1]
        
        if key1 in self.key_positions and key2 in self.key_positions:
            row1 = self.key_positions[key1].row
            row2 = self.key_positions[key2].row
            return abs(row1 - row2)
        
        return 0

    # =========================================================================
    # COLUMN SEPARATION PREFERENCES (SAME ROW SEPARATION, SHARE ONE KEY)
    # =========================================================================
    
    def _extract_column_separation_instances(self) -> pd.DataFrame:
        """Extract column separation instances."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
                
            chosen_row_sep = self._calculate_row_separation(chosen)
            unchosen_row_sep = self._calculate_row_separation(unchosen)
            chosen_col_sep = self._calculate_column_separation(chosen)
            unchosen_col_sep = self._calculate_column_separation(unchosen)

            # Block 1: Same-column vs other-column (NO row constraints)
            if (self._bigrams_share_one_key(chosen, unchosen) and
                ((chosen_col_sep == 0 and unchosen_col_sep > 0) or 
                (chosen_col_sep > 0 and unchosen_col_sep == 0))):
                
                chose_smaller = 1 if chosen_col_sep == 0 else 0
                comparison_type = "same_vs_other_no_row_control"
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'row_pattern': None,  # No row pattern since no row control
                    'comparison_type': comparison_type,
                    'chose_smaller_separation': chose_smaller,
                    'chosen_col_separation': chosen_col_sep,
                    'unchosen_col_separation': unchosen_col_sep,
                    'chosen_row_separation': chosen_row_sep,
                    'unchosen_row_separation': unchosen_row_sep,
                    'analysis_type': 'column_comparison'
                })

            # Block 2: Adjacent (1) vs Distant (2-3) columns - WITH row controls
            if (chosen_row_sep == unchosen_row_sep and  # Same row separation
                self._bigrams_share_one_key(chosen, unchosen) and  # Share one key
                ((chosen_col_sep == 1 and unchosen_col_sep >= 2) or  # Adjacent vs distant
                (chosen_col_sep >= 2 and unchosen_col_sep == 1))):
                
                chose_smaller = 1 if chosen_col_sep == 1 else 0  # 1 = chose adjacent
                
                # Determine row pattern for sub-analysis
                if chosen_row_sep == 0:
                    row_pattern = "same_row"
                elif chosen_row_sep == 1:
                    row_pattern = "reach_1_apart"
                elif chosen_row_sep == 2:
                    row_pattern = "hurdle_2_apart"
                else:
                    row_pattern = f"row_sep_{chosen_row_sep}"
                
                comparison_type = f"adjacent_vs_distant_{row_pattern}"
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'row_pattern': row_pattern,
                    'comparison_type': comparison_type,
                    'chose_smaller_separation': chose_smaller,
                    'chosen_col_separation': chosen_col_sep,
                    'unchosen_col_separation': unchosen_col_sep,
                    'chosen_row_separation': chosen_row_sep,
                    'unchosen_row_separation': unchosen_row_sep,
                    'analysis_type': 'column_comparison'
                })

            # Block 2b: Adjacent vs Distant - NO shared key constraint
            if (chosen_row_sep == unchosen_row_sep and  # Same row separation
                ((chosen_col_sep == 1 and unchosen_col_sep >= 2) or
                (chosen_col_sep >= 2 and unchosen_col_sep == 1))):
                
                chose_smaller = 1 if chosen_col_sep == 1 else 0
                
                if chosen_row_sep == 0:
                    row_pattern = "same_row"
                elif chosen_row_sep == 1:
                    row_pattern = "reach_1_apart"
                elif chosen_row_sep == 2:
                    row_pattern = "hurdle_2_apart"
                else:
                    row_pattern = f"row_sep_{chosen_row_sep}"
                
                comparison_type = f"adjacent_vs_distant_{row_pattern}_no_shared_key"
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'row_pattern': row_pattern,
                    'comparison_type': comparison_type,
                    'chose_smaller_separation': chose_smaller,
                    'chosen_col_separation': chosen_col_sep,
                    'unchosen_col_separation': unchosen_col_sep,
                    'chosen_row_separation': chosen_row_sep,
                    'unchosen_row_separation': unchosen_row_sep,
                    'analysis_type': 'column_comparison'
                })
                
            # Row comparisons (reach vs hurdle) with identical column separation and 1 shared key
            if chosen_col_sep == unchosen_col_sep:
                # Test: Reach (1 row) vs Hurdle (2 rows) for each column separation level
                if ({chosen_row_sep, unchosen_row_sep} == {1, 2} and
                    self._bigrams_share_one_key(chosen, unchosen)):

                    chose_smaller_row = 1 if chosen_row_sep == 1 else 0  # 1 = chose reach, 0 = chose hurdle
  
                    # Define column separation category for clearer reporting
                    if chosen_col_sep == 0:
                        col_category = "same_column"
                    elif chosen_col_sep == 1:
                        col_category = "adjacent_columns"
                    elif chosen_col_sep >= 2:  # Combine 2 and 3 into distant
                        col_category = "distant_columns"
                    else:
                        col_category = f"col_sep_{chosen_col_sep}"
                    
                    comparison_type = f"reach_vs_hurdle_{col_category}"
                    
                    instances.append({
                        'user_id': row['user_id'],
                        'chosen_bigram': chosen,
                        'unchosen_bigram': unchosen,
                        'row_pattern': None,  # Not applicable for this analysis
                        'comparison_type': comparison_type,
                        'chose_smaller_separation': chose_smaller_row,  # For row analysis, smaller = reach
                        'chosen_col_separation': chosen_col_sep,
                        'unchosen_col_separation': unchosen_col_sep,
                        'chosen_row_separation': chosen_row_sep,
                        'unchosen_row_separation': unchosen_row_sep,
                        'analysis_type': 'row_comparison'
                    })

            # Row comparisons (reach vs hurdle) with identical column separation and NO shared key constraint
            if chosen_col_sep == unchosen_col_sep:
                # Test: Reach (1 row) vs Hurdle (2 rows) for each column separation level
                if {chosen_row_sep, unchosen_row_sep} == {1, 2}:

                    chose_smaller_row = 1 if chosen_row_sep == 1 else 0  # 1 = chose reach, 0 = chose hurdle

                    # Define column separation category for clearer reporting
                    if chosen_col_sep == 0:
                        col_category = "same_column"
                    elif chosen_col_sep == 1:
                        col_category = "adjacent_columns"
                    elif chosen_col_sep >= 2:  # Combine 2 and 3 into distant
                        col_category = "distant_columns"
                    else:
                        col_category = f"col_sep_{chosen_col_sep}"
                    
                    comparison_type = f"reach_vs_hurdle_{col_category}_no_shared_key"
                    
                    instances.append({
                        'user_id': row['user_id'],
                        'chosen_bigram': chosen,
                        'unchosen_bigram': unchosen,
                        'row_pattern': None,  # Not applicable for this analysis
                        'comparison_type': comparison_type,
                        'chose_smaller_separation': chose_smaller_row,  # For row analysis, smaller = reach
                        'chosen_col_separation': chosen_col_sep,
                        'unchosen_col_separation': unchosen_col_sep,
                        'chosen_row_separation': chosen_row_sep,
                        'unchosen_row_separation': unchosen_row_sep,
                        'analysis_type': 'row_comparison'
                    })

        return pd.DataFrame(instances)
        
    def _analyze_column_separation(self) -> Dict[str, Any]:
        """Analyze column separation preferences with proper row controls AND reach vs hurdle analysis."""
        
        instances_df = self._extract_column_separation_instances()
        logger.info(f"Found {len(instances_df)} total separation instances")
        
        # Split analyses by type
        column_instances = instances_df[instances_df['analysis_type'] == 'column_comparison']
        row_instances = instances_df[instances_df['analysis_type'] == 'row_comparison']
        
        logger.info(f"Column comparison instances: {len(column_instances)}")
        logger.info(f"Row comparison instances: {len(row_instances)}")
        
        # Overall analysis (column comparisons only for overall stats)
        if not column_instances.empty:
            n_instances = len(column_instances)
            n_chose_smaller = column_instances['chose_smaller_separation'].sum()
            preference_rate = n_chose_smaller / n_instances
            
            confidence = self.config.get('confidence_level', 0.95)
            ci_lower, ci_upper = self._wilson_ci(n_chose_smaller, n_instances, confidence)
            
            # Statistical test
            z_score = (preference_rate - 0.5) / np.sqrt(0.25 / n_instances)
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
        else:
            n_instances = preference_rate = ci_lower = ci_upper = p_value = 0
        
        # Analysis by comparison type - COLUMN COMPARISONS
        column_pattern_results = {}
        for comp_type in column_instances['comparison_type'].unique():
            comp_data = column_instances[column_instances['comparison_type'] == comp_type]
            if len(comp_data) >= min_threshold:  # Minimum threshold
                n_comp = len(comp_data)
                n_chose_comp = comp_data['chose_smaller_separation'].sum()
                pref_rate_comp = n_chose_comp / n_comp
                ci_lower_comp, ci_upper_comp = self._wilson_ci(n_chose_comp, n_comp, confidence)
                
                z_comp = (pref_rate_comp - 0.5) / np.sqrt(0.25 / n_comp)
                p_comp = 2 * (1 - norm.cdf(abs(z_comp)))
                
                column_pattern_results[comp_type] = {
                    'n_instances': n_comp,
                    'preference_rate': pref_rate_comp,
                    'ci_lower': ci_lower_comp,
                    'ci_upper': ci_upper_comp,
                    'p_value': p_comp
                }
                logger.info(f"Column analysis - {comp_type}: {n_comp} instances, {pref_rate_comp:.1%} preference")
            else:
                logger.info(f"Skipped column analysis - {comp_type}: only {len(comp_data)} instances (need >= min_threshold)")
        
        # Analysis by comparison type - ROW COMPARISONS
        row_pattern_results = {}
        for comp_type in row_instances['comparison_type'].unique():
            comp_data = row_instances[row_instances['comparison_type'] == comp_type]
            if len(comp_data) >= min_threshold:  # Minimum threshold
                n_comp = len(comp_data)
                n_chose_comp = comp_data['chose_smaller_separation'].sum()  # chose_smaller_separation = chose reach
                pref_rate_comp = n_chose_comp / n_comp  # preference rate for reach
                ci_lower_comp, ci_upper_comp = self._wilson_ci(n_chose_comp, n_comp, confidence)
                
                z_comp = (pref_rate_comp - 0.5) / np.sqrt(0.25 / n_comp)
                p_comp = 2 * (1 - norm.cdf(abs(z_comp)))
                
                row_pattern_results[comp_type] = {
                    'n_instances': n_comp,
                    'preference_rate': pref_rate_comp,
                    'ci_lower': ci_lower_comp,
                    'ci_upper': ci_upper_comp,
                    'p_value': p_comp
                }
                logger.info(f"Row analysis - {comp_type}: {n_comp} instances, {pref_rate_comp:.1%} prefer reach")
            else:
                logger.info(f"Skipped row analysis - {comp_type}: only {len(comp_data)} instances (need >= min_threshold)")
        
        # Group column results for cleaner reporting
        same_vs_other_results = {}
        adjacent_vs_distant_results = {}
        
        for comp_type, results in column_pattern_results.items():
            if comp_type.startswith('same_vs_other_'):
                row_type = comp_type.replace('same_vs_other_', '')
                same_vs_other_results[row_type] = results
            elif comp_type.startswith('adjacent_vs_distant_'):
                row_type = comp_type.replace('adjacent_vs_distant_', '')
                adjacent_vs_distant_results[row_type] = results
        
        # Group row results for reporting
        reach_vs_hurdle_results = {}
        for comp_type, results in row_pattern_results.items():
            if comp_type.startswith('reach_vs_hurdle_'):
                col_type = comp_type.replace('reach_vs_hurdle_', '')
                reach_vs_hurdle_results[col_type] = results
        
        return {
            'description': 'Enhanced column separation analysis with row controls + reach vs hurdle by column pattern',
            'method': 'column_separation_analysis_enhanced',
            'n_instances': n_instances,
            'n_row_instances': len(row_instances),
            'n_total_instances': len(instances_df),
            'preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'same_vs_other_results': same_vs_other_results,
            'adjacent_vs_distant_results': adjacent_vs_distant_results,
            'reach_vs_hurdle_results': reach_vs_hurdle_results,
            'pattern_results': column_pattern_results,  # Keep for backward compatibility
            'interpretation': f"Column separation preference: {preference_rate:.1%} favor smaller distances (with row controls) + reach vs hurdle analysis"
        }

    def _calculate_column_separation(self, bigram: str) -> int:
        """Calculate column separation for a bigram."""
        if len(bigram) != 2:
            return 0
        
        key1, key2 = bigram[0], bigram[1]
        
        if key1 in self.key_positions and key2 in self.key_positions:
            col1 = self.key_positions[key1].column
            col2 = self.key_positions[key2].column
            return abs(col1 - col2)
        
        return 0

    # =========================================================================
    # INWARD VS OUTWARD ROLL PREFERENCES (CONSTRAINED TO SAME KEY PAIRS)
    # =========================================================================
    
    def _analyze_inward_outward_roll(self) -> Dict[str, Any]:
        """Analyze inward vs outward roll preference using same key pairs in both directions."""
        
        instances_df = self._extract_constrained_inward_outward_instances()
        
        if instances_df.empty:
            logger.warning('No constrained inward/outward roll instances found - skipping this analysis')
            return {
                'description': 'Inward vs outward roll preference (same key pairs, different directions)',
                'method': 'constrained_inward_outward_analysis',
                'status': 'insufficient_data',
                'n_instances': 0,
                'interpretation': 'No constrained inward vs outward roll comparisons found in data'
            }
        
        logger.info(f"Found {len(instances_df)} constrained inward vs outward roll instances")
        
        # Overall analysis
        n_instances = len(instances_df)
        n_chose_inward = instances_df['chose_inward_roll'].sum()
        preference_rate = n_chose_inward / n_instances
        
        confidence = self.config.get('confidence_level', 0.95)
        ci_lower, ci_upper = self._wilson_ci(n_chose_inward, n_instances, confidence)
        
        # Statistical test
        z_score = (preference_rate - 0.5) / np.sqrt(0.25 / n_instances)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        # Analysis by row pattern
        pattern_results = {}
        for row_pattern in instances_df['row_pattern'].unique():
            pattern_data = instances_df[instances_df['row_pattern'] == row_pattern]
            if len(pattern_data) >= min_threshold:  # Minimum threshold
                n_pattern = len(pattern_data)
                n_chose_inward_pattern = pattern_data['chose_inward_roll'].sum()
                pref_rate_pattern = n_chose_inward_pattern / n_pattern
                ci_lower_pattern, ci_upper_pattern = self._wilson_ci(n_chose_inward_pattern, n_pattern, confidence)
                
                z_pattern = (pref_rate_pattern - 0.5) / np.sqrt(0.25 / n_pattern)
                p_pattern = 2 * (1 - norm.cdf(abs(z_pattern)))
                
                pattern_results[row_pattern] = {
                    'n_instances': n_pattern,
                    'preference_rate': pref_rate_pattern,
                    'ci_lower': ci_lower_pattern,
                    'ci_upper': ci_upper_pattern,
                    'p_value': p_pattern
                }
                logger.info(f"Constrained inward/outward - {row_pattern}: {n_pattern} instances, {pref_rate_pattern:.1%} prefer inward")
            else:
                logger.info(f"Skipped constrained inward/outward - {row_pattern}: only {len(pattern_data)} instances (need >= min_threshold)")
        
        return {
            'description': 'Inward vs outward roll preference (same key pairs, different directions)',
            'method': 'constrained_inward_outward_analysis',
            'n_instances': n_instances,
            'inward_preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'pattern_results': pattern_results,
            'interpretation': f"Inward roll preference rate: {preference_rate:.1%} (prefer increasing finger number sequence)"
        }

    def _extract_constrained_inward_outward_instances(self) -> pd.DataFrame:
        """Extract inward vs outward roll instances using same key pairs."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            # Only analyze left-hand bigrams
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
            
            # Check if they contain the same two keys in reverse order
            if not self._are_same_keys_reverse_order(chosen, unchosen):
                continue
            
            # Exclude same-column bigrams (no roll motion possible)
            if self._is_same_column_bigram(chosen):
                continue
            
            # Classify roll directions
            chosen_roll_type = self._classify_roll_direction(chosen)
            unchosen_roll_type = self._classify_roll_direction(unchosen)
            
            # Only include if one is inward and one is outward
            if {chosen_roll_type, unchosen_roll_type} == {'inward', 'outward'}:
                chose_inward_roll = 1 if chosen_roll_type == 'inward' else 0
                
                # Determine row pattern for sub-analysis
                row_separation = self._calculate_row_separation(chosen)  # Same for both since same keys
                
                if row_separation == 0:
                    row_pattern = "same_row"
                elif row_separation == 1:
                    row_pattern = "reach"
                elif row_separation == 2:
                    row_pattern = "hurdle"
                else:
                    row_pattern = "extreme"
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'chose_inward_roll': chose_inward_roll,
                    'chosen_roll_type': chosen_roll_type,
                    'unchosen_roll_type': unchosen_roll_type,
                    'row_pattern': row_pattern,
                    'row_separation': row_separation,
                    'key_pair': tuple(sorted([chosen[0], chosen[1]]))
                })
        
        return pd.DataFrame(instances)

    def _are_same_keys_reverse_order(self, bigram1: str, bigram2: str) -> bool:
        """Check if two bigrams contain the same keys in reverse order."""
        if len(bigram1) != 2 or len(bigram2) != 2:
            return False
        
        # Check if bigram2 is the reverse of bigram1
        return bigram1 == bigram2[::-1] and bigram1 != bigram2

    def _classify_roll_direction(self, bigram: str) -> str:
        """Classify bigram as inward roll, outward roll, or neither."""
        if len(bigram) != 2:
            return 'invalid'
        
        key1, key2 = bigram[0], bigram[1]
        
        # Get finger assignments
        if key1 not in self.key_positions or key2 not in self.key_positions:
            return 'invalid'
        
        finger1 = self.key_positions[key1].finger
        finger2 = self.key_positions[key2].finger
        
        # Don't analyze same-finger bigrams
        if finger1 == finger2:
            return 'same_finger'
        
        # Classify roll direction
        if finger2 > finger1:  # Finger number increases (pinky → index)
            return 'inward'
        elif finger2 < finger1:  # Finger number decreases (index → pinky)
            return 'outward'
        else:
            return 'same_finger'  # Should not reach here given the check above

    def _is_same_column_bigram(self, bigram: str) -> bool:
        """Check if bigram uses keys from the same column."""
        if len(bigram) != 2:
            return False
        
        key1, key2 = bigram[0], bigram[1]
        
        if key1 in self.key_positions and key2 in self.key_positions:
            col1 = self.key_positions[key1].column
            col2 = self.key_positions[key2].column
            return col1 == col2
        
        return False
            
    # =========================================================================
    # SIDE REACH PREFERENCES (SAME-ROW ONLY, NO SAME-COLUMN)
    # =========================================================================

    def _analyze_side_reach(self) -> Dict[str, Any]:
        """Analyze side reach preferences using only same-row bigrams (no same-column)."""
        
        instances_df = self._extract_same_row_side_reach_instances()
        
        if instances_df.empty:
            logger.warning('No same-row side reach instances found - skipping this analysis')
            return {
                'description': 'Side reach analysis: Same-row bigrams only (no same-column)',
                'method': 'same_row_side_reach_analysis',
                'status': 'insufficient_data',
                'n_instances': 0,
                'interpretation': 'No same-row side reach comparisons found in data'
            }
        
        logger.info(f"Found {len(instances_df)} same-row side reach instances")
        
        # Analysis
        n_instances = len(instances_df)
        n_chose_standard = instances_df['chose_standard_area'].sum()
        preference_rate = n_chose_standard / n_instances
        
        confidence = self.config.get('confidence_level', 0.95)
        ci_lower, ci_upper = self._wilson_ci(n_chose_standard, n_instances, confidence)
        
        # Statistical test
        z_score = (preference_rate - 0.5) / np.sqrt(0.25 / n_instances)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return {
            'description': 'Side reach analysis: Same-row bigrams only (no same-column)',
            'method': 'same_row_side_reach_analysis',
            'n_instances': n_instances,
            'standard_area_preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'interpretation': f"Same-row standard area preference: {preference_rate:.1%} (pure side reach cost)"
        }

    def _extract_same_row_side_reach_instances(self) -> pd.DataFrame:
        """Extract same-row side reach instances (no same-column bigrams)."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            # Only analyze valid bigrams in our area
            if not (self._all_keys_in_analysis_area(chosen) and self._all_keys_in_analysis_area(unchosen)):
                continue
                
            chosen_row_sep = self._calculate_row_separation(chosen)
            unchosen_row_sep = self._calculate_row_separation(unchosen)
            chosen_requires_side_reach = self._requires_side_reach_movement(chosen)
            unchosen_requires_side_reach = self._requires_side_reach_movement(unchosen)
            
            # SAME-ROW ONLY: Both bigrams must have 0 row separation
            if chosen_row_sep != 0 or unchosen_row_sep != 0:
                continue
            
            # EXCLUDE SAME-COLUMN: Neither bigram should be same-column
            if self._is_same_column_bigram(chosen) or self._is_same_column_bigram(unchosen):
                continue
            
            # DIFFERENT SIDE REACH: One requires side reach, one doesn't
            if chosen_requires_side_reach != unchosen_requires_side_reach:
                chose_standard_area = 1 if not chosen_requires_side_reach else 0
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'chose_standard_area': chose_standard_area,
                    'chosen_requires_side_reach': chosen_requires_side_reach,
                    'unchosen_requires_side_reach': unchosen_requires_side_reach
                })
        
        return pd.DataFrame(instances)

    def _requires_side_reach_movement(self, bigram: str) -> bool:
        """Check if typing this bigram requires reaching to column 5."""
        if len(bigram) != 2:
            return False
        
        side_reach_keys = self._get_column_5_keys()  # T, G, B
        
        # Returns True if any key in the bigram is in column 5
        return any(key in side_reach_keys for key in bigram)

    def _all_keys_in_analysis_area(self, bigram: str) -> bool:
        """Check if all keys are in our analysis area (left hand + column 5)."""
        analysis_keys = set(self.key_positions.keys()) | self._get_column_5_keys()
        return all(key in analysis_keys for key in bigram)

    def _is_same_column_bigram(self, bigram: str) -> bool:
        """Check if bigram uses keys from the same column."""
        if len(bigram) != 2:
            return False
        
        key1, key2 = bigram[0], bigram[1]
        
        if key1 in self.key_positions and key2 in self.key_positions:
            col1 = self.key_positions[key1].column
            col2 = self.key_positions[key2].column
            return col1 == col2
        
        return False

    def _classify_side_reach_movement(self, bigram: str) -> str:
        """Classify bigram by whether it requires side reach movement."""
        if not self._all_keys_in_analysis_area(bigram):
            return 'invalid'  # Contains keys outside our analysis area
        
        if self._requires_side_reach_movement(bigram):
            return 'requires_side_reach'
        else:
            return 'no_side_reach'
        
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _wilson_ci(self, successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval for proportion."""
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
    
    def _all_keys_in_left_hand(self, bigram: str) -> bool:
        """Check if all keys in bigram are left-hand keys."""
        return all(key in self.left_hand_keys for key in bigram)

    def _bigrams_share_one_key(self, bigram1: str, bigram2: str) -> bool:
        """Check if two bigrams share exactly one key."""
        if len(bigram1) != 2 or len(bigram2) != 2:
            return False
        
        keys1 = set(bigram1)
        keys2 = set(bigram2)
        shared_keys = keys1.intersection(keys2)
        
        return len(shared_keys) == 1
    
    # =========================================================================
    # REPORTING AND VISUALIZATION
    # =========================================================================

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary."""
        
        objectives_completed = len([r for r in results.values() if 'error' not in str(r)])
        
        return {
            'objectives_completed': objectives_completed,
            'total_objectives': len(results),
            'approach': 'practical_significance_focus',
            'completed_objective_names': [name for name, result in results.items() if 'error' not in str(result)]
        }

    def _generate_comprehensive_report(self, enhanced_results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive text report."""
        
        results = enhanced_results['objectives']
        summary = enhanced_results['summary']
        
        report_lines = [
            "MOO OBJECTIVES ANALYSIS REPORT",
            "=" * 35,
            "",
            "ANALYSIS SUMMARY:",
            f"Objectives completed: {summary.get('objectives_completed', 'unknown')}/{summary.get('total_objectives', 'unknown')}",
            f"Approach: Practical significance focus with statistical support",
            f"Emphasis: Effect sizes and confidence intervals for MOO design",
            "",
            "APPROACH:",
            "- Effect sizes and confidence intervals quantify practical differences",
            "- Statistical tests confirm preferences are detectable above chance",
            "- No corrections across objectives (each addresses distinct mechanics)",
            "- Results inform engineering decisions for keyboard optimization",
            "",
            "REPORTING FORMAT:",
            "- Effect size: Magnitude of practical difference (primary interest)",
            "- Confidence interval: Uncertainty quantification around effect",
            "- P-value: Evidence that preference is detectable above noise",
            "",
            "OBJECTIVES DETAILED RESULTS:",
            "=" * 35,
            ""
        ]
        
        # Add detailed results for each objective
        for i, (obj_name, obj_results) in enumerate(results.items(), 1):
            
            if isinstance(obj_results, str) and 'error' in obj_results.lower():
                report_lines.extend([
                    f"\n{i}. {obj_name.upper().replace('_', ' ')}:",
                    "-" * (len(obj_name) + 4),
                    f"STATUS: FAILED - {obj_results}",
                    ""
                ])
                continue
                
            report_lines.extend([
                f"\n{i}. {obj_name.upper().replace('_', ' ')}:",
                "-" * (len(obj_name) + 4),
                f"Description: {obj_results.get('description', 'No description')}",
                f"Method: {obj_results.get('method', 'unknown')}",
                f"Instances analyzed: {obj_results.get('n_instances', 'unknown')}",
                ""
            ])
            
            # Add specific results based on objective type
            if 'bradley_terry' in obj_name and 'rankings' in obj_results:
                rankings = obj_results['rankings']
                cis = obj_results.get('confidence_intervals', {})
                
                report_lines.extend([
                    "  BRADLEY-TERRY RANKINGS:",
                    "  Top 5 preferred keys (pure key quality):",
                    ""
                ])
                for j, (key, strength) in enumerate(rankings[:5], 1):
                    ci_lower, ci_upper = cis.get(key, (np.nan, np.nan))
                    ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]" if not np.isnan(ci_lower) else "[CI unavailable]"
                    
                    pos = self.key_positions.get(key, KeyPosition('', 0, 0, 0))
                    report_lines.append(f"    {j}. {key.upper()} (F{pos.finger}, R{pos.row}): "
                                    f"strength = {strength:.3f}, 95% CI = {ci_str}")
                
                report_lines.extend([
                    "",
                    f"  INTERPRETATION: Strength > 0 indicates above-average preference",
                    f"  Statistical validity: Bradley-Terry model with regularization",
                    ""
                ])
                
            elif 'pairwise' in obj_name and 'pairwise_results' in obj_results:
                pair_results = obj_results['pairwise_results']
                report_lines.extend([
                    f"  PAIRWISE KEY COMPARISONS:",
                    f"  Analyzed {len(pair_results)} key pairs with sufficient data:",
                    ""
                ])
                
                # Show top preferences with full statistical reporting
                sorted_pairs = sorted(pair_results.items(), 
                                    key=lambda x: abs(x[1]['preference_rate'] - 0.5), reverse=True)
                
                for (key1, key2), data in sorted_pairs[:5]:
                    pref_rate = data['preference_rate']
                    ci_lower, ci_upper = data['ci_lower'], data['ci_upper']
                    p_value = data.get('p_value', np.nan)
                    n_inst = data['n_instances']
                    
                    # Determine winner and effect size
                    if pref_rate > 0.5:
                        winner, loser = key1.upper(), key2.upper()
                        effect_size = pref_rate - 0.5
                    else:
                        winner, loser = key2.upper(), key1.upper()
                        pref_rate = 1 - pref_rate
                        effect_size = pref_rate - 0.5
                        ci_lower, ci_upper = 1 - ci_upper, 1 - ci_lower
                    
                    # Statistical significance indicator
                    sig_indicator = ""
                    if not np.isnan(p_value):
                        if p_value < 0.001:
                            sig_indicator = " ***"
                        elif p_value < 0.01:
                            sig_indicator = " **"
                        elif p_value < 0.05:
                            sig_indicator = " *"
                    
                    report_lines.extend([
                        f"    {winner} > {loser}:",
                        f"      Preference rate: {pref_rate:.1%} (effect size: {effect_size:.1%})",
                        f"      95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]",
                        f"      Statistical test: p = {p_value:.4f}{sig_indicator} (n={n_inst})" if not np.isnan(p_value) else f"      Statistical test: Not available (n={n_inst})",
                        ""
                    ])
                
            elif obj_name == 'column_separation' and 'preference_rate' in obj_results:
                pref_rate = obj_results['preference_rate']
                ci_lower = obj_results.get('ci_lower', np.nan)
                ci_upper = obj_results.get('ci_upper', np.nan)
                p_value = obj_results.get('p_value', np.nan)
                n_instances = obj_results.get('n_instances', 0)
                
                # Calculate effect size (departure from no preference)
                effect_size = abs(pref_rate - 0.5)
                
                # Statistical significance indicator
                sig_indicator = ""
                if not np.isnan(p_value):
                    if p_value < 0.001:
                        sig_indicator = " ***"
                    elif p_value < 0.01:
                        sig_indicator = " **"
                    elif p_value < 0.05:
                        sig_indicator = " *"
                
                report_lines.extend([
                    f"  OVERALL COLUMN SEPARATION PREFERENCE:",
                    f"  Preference rate: {pref_rate:.1%} favor smaller distances (effect size: {effect_size:.1%})",
                    f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]" if not np.isnan(ci_lower) else "  95% CI: Not available",
                    f"  Statistical test: p = {p_value:.4f}{sig_indicator} (n={n_instances})" if not np.isnan(p_value) else f"  Statistical test: Not available (n={n_instances})",
                    "",
                    f"  METHODS NOTES:",
                    f"  - Same-column vs other-column: No row constraints (tests fundamental same-finger preference)",
                    f"  - Adjacent vs distant: Row constraints maintained (isolates pure column separation effect)", 
                    f"  - All comparisons require exactly one shared key",
                    f"  - Mixed approach balances sample size with experimental control",
                    ""
                ])
                
                # Get pattern results from the analysis
                column_pattern_results = obj_results.get('pattern_results', {})

                # Group column results for mixed reporting
                same_vs_other_results = {}
                adjacent_vs_distant_results = {}

                for comp_type, results in column_pattern_results.items():
                    if comp_type == 'same_vs_other_no_row_control':
                        same_vs_other_results['no_row_control'] = results
                    elif comp_type.startswith('adjacent_vs_distant_'):
                        row_type = comp_type.replace('adjacent_vs_distant_', '')
                        adjacent_vs_distant_results[row_type] = results

                # Report same vs other comparisons
                if same_vs_other_results:
                    report_lines.extend([
                        f"  SAME COLUMN (0) VS OTHER COLUMNS (1-3) - NO ROW CONTROLS:",
                        f"  (All row separations included + shared key constraint)",
                        ""
                    ])
                    
                    for row_type, comp_data in same_vs_other_results.items():
                        comp_pref = comp_data['preference_rate']
                        comp_ci_lower = comp_data.get('ci_lower', np.nan)
                        comp_ci_upper = comp_data.get('ci_upper', np.nan)
                        comp_p_value = comp_data.get('p_value', np.nan)
                        comp_n = comp_data['n_instances']
                        
                        # Calculate display values
                        if comp_pref > 0.5:
                            display_pref = comp_pref
                            display_ci_lower, display_ci_upper = comp_ci_lower, comp_ci_upper
                            interpretation = "same-finger movements"
                        else:
                            display_pref = 1 - comp_pref
                            display_ci_lower, display_ci_upper = 1 - comp_ci_upper, 1 - comp_ci_lower
                            interpretation = "different-finger movements"
                        
                        comp_effect = abs(comp_pref - 0.5)
                        
                        # Statistical significance indicator
                        comp_sig_indicator = ""
                        if not np.isnan(comp_p_value):
                            if comp_p_value < 0.001:
                                comp_sig_indicator = " ***"
                            elif comp_p_value < 0.01:
                                comp_sig_indicator = " **"
                            elif comp_p_value < 0.05:
                                comp_sig_indicator = " *"
                        
                        type_name = "All Movement Types (0, 1, 2 row separations combined)"
                        
                        report_lines.extend([
                            f"    {type_name}:",
                            f"      Preference: {display_pref:.1%} favor {interpretation} (effect size: {comp_effect:.1%})",
                            f"      95% CI: [{display_ci_lower:.1%}, {display_ci_upper:.1%}]" if not np.isnan(display_ci_lower) else "      95% CI: Not available",
                            f"      Statistical test: p = {comp_p_value:.4f}{comp_sig_indicator} (n={comp_n})" if not np.isnan(comp_p_value) else f"      Statistical test: Not available (n={comp_n})",
                            ""
                        ])
                        
                # Adjacent vs distant reporting
                if adjacent_vs_distant_results:
                    report_lines.extend([
                        f"  ADJACENT (1) VS DISTANT (2-3) COLUMNS - BY ROW PATTERN:",
                        ""
                    ])
                    
                    for row_type, comp_data in adjacent_vs_distant_results.items():
                        comp_pref = comp_data['preference_rate']
                        comp_ci_lower = comp_data.get('ci_lower', np.nan)
                        comp_ci_upper = comp_data.get('ci_upper', np.nan)
                        comp_p_value = comp_data.get('p_value', np.nan)
                        comp_n = comp_data['n_instances']
                        
                        # Calculate display values
                        if comp_pref > 0.5:
                            display_pref = comp_pref
                            display_ci_lower, display_ci_upper = comp_ci_lower, comp_ci_upper
                            interpretation = "adjacent columns"
                        else:
                            display_pref = 1 - comp_pref
                            display_ci_lower, display_ci_upper = 1 - comp_ci_upper, 1 - comp_ci_lower
                            interpretation = "distant columns"
                        
                        comp_effect = abs(comp_pref - 0.5)
                        
                        # Statistical significance indicator
                        comp_sig_indicator = ""
                        if not np.isnan(comp_p_value):
                            if comp_p_value < 0.001:
                                comp_sig_indicator = " ***"
                            elif comp_p_value < 0.01:
                                comp_sig_indicator = " **"
                            elif comp_p_value < 0.05:
                                comp_sig_indicator = " *"
                        
                        # Enhanced descriptions based on row pattern
                        if row_type == "same_row":
                            type_name = "Same Row Movements (0 row separation)"
                        elif row_type == "reach_1_apart":
                            type_name = "Reach Movements (1 row separation)"
                        elif row_type == "hurdle_2_apart":
                            type_name = "Hurdle Movements (2 row separation)"
                        else:
                            type_name = row_type.replace('_', ' ').title()
                        
                        report_lines.extend([
                            f"    {type_name}:",
                            f"      Preference: {display_pref:.1%} favor {interpretation} (effect size: {comp_effect:.1%})",
                            f"      95% CI: [{display_ci_lower:.1%}, {display_ci_upper:.1%}]" if not np.isnan(display_ci_lower) else "      95% CI: Not available",
                            f"      Statistical test: p = {comp_p_value:.4f}{comp_sig_indicator} (n={comp_n})" if not np.isnan(comp_p_value) else f"      Statistical test: Not available (n={comp_n})",
                            ""
                        ])

            elif obj_name == 'row_separation' and 'preference_rate' in obj_results:
                pref_rate = obj_results['preference_rate']
                ci_lower = obj_results.get('ci_lower', np.nan)
                ci_upper = obj_results.get('ci_upper', np.nan)
                p_value = obj_results.get('p_value', np.nan)
                n_instances = obj_results.get('n_instances', 0)
                
                # Calculate effect size (departure from no preference)
                effect_size = abs(pref_rate - 0.5)
                
                # Statistical significance indicator
                sig_indicator = ""
                if not np.isnan(p_value):
                    if p_value < 0.001:
                        sig_indicator = " ***"
                    elif p_value < 0.01:
                        sig_indicator = " **"
                    elif p_value < 0.05:
                        sig_indicator = " *"
                
                report_lines.extend([
                    f"  OVERALL ROW SEPARATION PREFERENCE:",
                    f"  Preference rate: {pref_rate:.1%} favor smaller distances (effect size: {effect_size:.1%})",
                    f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]" if not np.isnan(ci_lower) else "  95% CI: Not available",
                    f"  Statistical test: p = {p_value:.4f}{sig_indicator} (n={n_instances})" if not np.isnan(p_value) else f"  Statistical test: Not available (n={n_instances})",
                    ""
                ])
                
                # Show detailed breakdown by comparison type
                breakdown = obj_results.get('comparison_results', {})
                
                if breakdown:
                    report_lines.extend([
                        f"  BREAKDOWN BY TYPE:",
                        ""
                    ])
                    for comp_type, comp_data in breakdown.items():
                        comp_pref = comp_data['preference_rate']
                        comp_ci_lower = comp_data.get('ci_lower', np.nan)
                        comp_ci_upper = comp_data.get('ci_upper', np.nan)
                        comp_p_value = comp_data.get('p_value', np.nan)
                        comp_n = comp_data['n_instances']
                        comp_effect = abs(comp_pref - 0.5)
                        
                        # Statistical significance indicator
                        comp_sig_indicator = ""
                        if not np.isnan(comp_p_value):
                            if comp_p_value < 0.001:
                                comp_sig_indicator = " ***"
                            elif comp_p_value < 0.01:
                                comp_sig_indicator = " **"
                            elif comp_p_value < 0.05:
                                comp_sig_indicator = " *"
                        
                        # Enhanced type descriptions
                        if comp_type == "same_row_vs_1_apart":
                            type_name = "Same Row (0) vs Reach (1)"
                        elif comp_type == "1_apart_vs_2_apart":
                            type_name = "Reach (1) vs Hurdle (2)"
                        else:
                            type_name = comp_type.replace('_', ' ').title()
                        
                        interpretation = "Smaller distance" if comp_pref > 0.5 else "Larger distance"
                        
                        report_lines.extend([
                            f"    {type_name}:",
                            f"      Preference: {comp_pref:.1%} favor {interpretation} (effect size: {comp_effect:.1%})",
                            f"      95% CI: [{comp_ci_lower:.1%}, {comp_ci_upper:.1%}]" if not np.isnan(comp_ci_lower) else "      95% CI: Not available",
                            f"      Statistical test: p = {comp_p_value:.4f}{comp_sig_indicator} (n={comp_n})" if not np.isnan(comp_p_value) else f"      Statistical test: Not available (n={comp_n})",
                            ""
                        ])
                else:
                    report_lines.extend([
                        f"  No detailed breakdown available (insufficient data for individual comparison types)",
                        ""
                    ])

            elif obj_name == 'inward_outward_roll' and 'inward_preference_rate' in obj_results:
                pref_rate = obj_results['inward_preference_rate']
                ci_lower = obj_results.get('ci_lower', np.nan)
                ci_upper = obj_results.get('ci_upper', np.nan)
                p_value = obj_results.get('p_value', np.nan)
                n_instances = obj_results.get('n_instances', 0)
                
                # Calculate effect size
                effect_size = abs(pref_rate - 0.5)
                
                # Statistical significance indicator
                sig_indicator = ""
                if not np.isnan(p_value):
                    if p_value < 0.001:
                        sig_indicator = " ***"
                    elif p_value < 0.01:
                        sig_indicator = " **"
                    elif p_value < 0.05:
                        sig_indicator = " *"
                
                report_lines.extend([
                    f"  CONSTRAINED INWARD VS OUTWARD ROLL PREFERENCE:",
                    f"  Inward roll preference rate: {pref_rate:.1%} (effect size: {effect_size:.1%})",
                    f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]" if not np.isnan(ci_lower) else "  95% CI: Not available",
                    f"  Statistical test: p = {p_value:.4f}{sig_indicator} (n={n_instances})" if not np.isnan(p_value) else f"  Statistical test: Not available (n={n_instances})",
                    "",
                    f"  CONSTRAINED APPROACH:",
                    f"  - Compares same key pairs in both movement directions",
                    f"  - Inward roll: Finger number increases (pinky → index)",
                    f"  - Outward roll: Finger number decreases (index → pinky)", 
                    f"  - Excludes same-column bigrams (no roll motion possible)",
                    f"  - Controls for key identity, distance, and quality differences",
                    "",
                    f"  EXAMPLES:",
                    f"  - 'as' (inward: finger 1→2) vs 'sa' (outward: finger 2→1)",
                    f"  - 'df' (inward: finger 3→4) vs 'fd' (outward: finger 4→3)",
                    f"  - 'aw' (inward: finger 1→2) vs 'wa' (outward: finger 2→1)",
                    ""
                ])
                
                # Pattern-specific results by row separation
                pattern_results = obj_results.get('pattern_results', {})
                if pattern_results:
                    report_lines.extend([
                        f"  INWARD ROLL PREFERENCE BY MOVEMENT COMPLEXITY:",
                        ""
                    ])
                    
                    for row_pattern, pattern_data in pattern_results.items():
                        pattern_pref = pattern_data['preference_rate']
                        pattern_ci_lower = pattern_data.get('ci_lower', np.nan)
                        pattern_ci_upper = pattern_data.get('ci_upper', np.nan)
                        pattern_p_value = pattern_data.get('p_value', np.nan)
                        pattern_n = pattern_data['n_instances']
                        
                        # Calculate effect size and interpretation
                        pattern_effect = abs(pattern_pref - 0.5)
                        if pattern_pref > 0.5:
                            interpretation = "inward roll"
                            display_pref = pattern_pref
                            display_ci_lower, display_ci_upper = pattern_ci_lower, pattern_ci_upper
                        else:
                            interpretation = "outward roll"
                            display_pref = 1 - pattern_pref
                            display_ci_lower, display_ci_upper = 1 - pattern_ci_upper, 1 - pattern_ci_lower
                        
                        # Statistical significance indicator
                        pattern_sig_indicator = ""
                        if not np.isnan(pattern_p_value):
                            if pattern_p_value < 0.001:
                                pattern_sig_indicator = " ***"
                            elif pattern_p_value < 0.01:
                                pattern_sig_indicator = " **"
                            elif pattern_p_value < 0.05:
                                pattern_sig_indicator = " *"
                        
                        # Enhanced descriptions with constrained examples
                        if row_pattern == "same_row":
                            type_name = "Same Row Movements"
                            examples = "'as'/'sa', 'df'/'fd', 'qw'/'wq'"
                        elif row_pattern == "reach":
                            type_name = "Reach Movements (1 row apart)"
                            examples = "'aw'/'wa', 'dr'/'rd', 'sz'/'zs'"
                        elif row_pattern == "hurdle":
                            type_name = "Hurdle Movements (2 rows apart)"
                            examples = "'az'/'za', 'qx'/'xq', 'ec'/'ce'"
                        else:
                            type_name = row_pattern.replace('_', ' ').title()
                            examples = ""
                        
                        report_lines.extend([
                            f"    {type_name}:",
                            f"      Examples: {examples}" if examples else "",
                            f"      Preference: {display_pref:.1%} favor {interpretation} (effect size: {pattern_effect:.1%})",
                            f"      95% CI: [{display_ci_lower:.1%}, {display_ci_upper:.1%}]" if not np.isnan(display_ci_lower) else "      95% CI: Not available",
                            f"      Statistical test: p = {pattern_p_value:.4f}{pattern_sig_indicator} (n={pattern_n})" if not np.isnan(pattern_p_value) else f"      Statistical test: Not available (n={pattern_n})",
                            ""
                        ])
                
                report_lines.extend([
                    f"  INTERPRETATION:",
                    f"  - Constrained analysis isolates pure movement direction preference",
                    f"  - Controls for key identity, distance, and quality by using same key pairs",
                    f"  - Rate > 50%: Natural preference for inward rolling motion (pinky→index)",
                    f"  - Rate < 50%: Natural preference for outward rolling motion (index→pinky)",
                    f"  - This measures fundamental finger movement biomechanics",
                    ""
                ])

            elif obj_name == 'side_reach' and 'standard_area_preference_rate' in obj_results:
                pref_rate = obj_results['standard_area_preference_rate']
                ci_lower = obj_results.get('ci_lower', np.nan)
                ci_upper = obj_results.get('ci_upper', np.nan)
                p_value = obj_results.get('p_value', np.nan)
                n_instances = obj_results.get('n_instances', 0)
                
                # Calculate effect size
                effect_size = abs(pref_rate - 0.5)
                
                # Statistical significance indicator
                sig_indicator = ""
                if not np.isnan(p_value):
                    if p_value < 0.001:
                        sig_indicator = " ***"
                    elif p_value < 0.01:
                        sig_indicator = " **"
                    elif p_value < 0.05:
                        sig_indicator = " *"
                
                report_lines.extend([
                    f"  SAME-ROW SIDE REACH PREFERENCE:",
                    f"  Standard area preference rate: {pref_rate:.1%} (effect size: {effect_size:.1%})",
                    f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]" if not np.isnan(ci_lower) else "  95% CI: Not available",
                    f"  Statistical test: p = {p_value:.4f}{sig_indicator} (n={n_instances})" if not np.isnan(p_value) else f"  Statistical test: Not available (n={n_instances})",
                    "",
                    f"  METHODS:",
                    f"  - Same-row bigrams only: Both keys on same keyboard row (0 row separation)",
                    f"  - No same-column bigrams: Excludes same-finger movements for cleaner results",
                    f"  - Standard area: Bigrams using only columns 1-4 (Q,W,E,R,A,S,D,F,Z,X,C,V)",
                    f"  - Side reach: Bigrams containing column 5 keys (T,G,B)",
                    f"  - Purest test of side reach cost without movement complexity confounds",
                    "",
                    f"  EXAMPLES OF VALID COMPARISONS:",
                    f"  - Row 1: 'qw' vs 'qt', 'er' vs 'et', 'wr' vs 'wt'",
                    f"  - Row 2: 'as' vs 'ag', 'df' vs 'dg', 'sf' vs 'sg'",
                    f"  - Row 3: 'zx' vs 'zb', 'cv' vs 'cb', 'xv' vs 'xb'",
                    "",
                    f"  EXCLUDED COMPARISONS:",
                    f"  - Different rows: 'aw' vs 'at' (reach movements)",
                    f"  - Same column: 'de' vs 'gt' (same finger movements)",
                    "",
                    f"  INTERPRETATION:",
                    f"  - Rate > 50%: Users prefer staying within standard left-hand area",
                    f"  - Rate < 50%: Users comfortable with side reach movements",
                    f"  - Effect measures pure ergonomic cost of extending to column 5",
                    f"  - Same-row restriction eliminates row movement complexity",
                    ""
                ])
                            
            # Add interpretation for each objective
            interpretation = obj_results.get('interpretation', '')
            if interpretation:
                report_lines.extend([
                    f"  SUMMARY: {interpretation}",
                    ""
                ])
        
        # Add statistical interpretation guide
        report_lines.extend([
            "",
            "STATISTICAL INTERPRETATION GUIDE:",
            "=" * 35,
            "Effect Size Interpretation:",
            "- Small effect: 5-15% departure from 50%",
            "- Medium effect: 15-25% departure from 50%", 
            "- Large effect: >25% departure from 50%",
            "",
            "P-value Significance Levels:",
            "- *** p < 0.001 (very strong evidence)",
            "- ** p < 0.01 (strong evidence)",
            "- * p < 0.05 (moderate evidence)",
            "- No asterisk: p >= 0.05 (weak/no evidence)",
            "",
            "Confidence Intervals:",
            "- Narrow CI: More precise estimate",
            "- CI excluding 50%: Statistically detectable preference",
            "- CI including 50%: Preference not clearly detectable",
            "",
            "Note: Focus on effect sizes for practical significance in MOO design.",
            "P-values confirm detectability but don't indicate importance."
        ])
        
        # Save report
        report_path = os.path.join(output_folder, 'moo_objectives_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Comprehensive report with statistical tests saved to {report_path}")

    def _save_key_preference_tables(self, results: Dict[str, Any], output_folder: str) -> None:
        """Save detailed CSV tables for all key preference methods."""
        
        # 1. Same-Letter Preferences
        if 'same_letter_preferences' in results and results['same_letter_preferences'].get('status') != 'insufficient_data':
            sl_results = results['same_letter_preferences']
            
            sl_data = []
            rankings = sl_results.get('rankings', [])
            cis = sl_results.get('confidence_intervals', {})
            
            for rank, (key, strength) in enumerate(rankings):
                pos = self.key_positions.get(key, KeyPosition('', 0, 0, 0))
                ci_lower, ci_upper = cis.get(key, (np.nan, np.nan))
                
                sl_data.append({
                    'Key': key.upper(),
                    'Finger': pos.finger,
                    'Row': pos.row,
                    'Column': pos.column,
                    'Rank': rank + 1,
                    'BT_Strength': strength,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper
                })
            
            sl_df = pd.DataFrame(sl_data)
            sl_df.to_csv(os.path.join(output_folder, 'key_preferences_same_letter_pairs_BT.csv'), index=False)
        
        # 2. Pairwise Key Comparisons
        if 'pairwise_preferences' in results and 'pairwise_results' in results['pairwise_preferences']:
            pairwise_data = []
            pair_results = results['pairwise_preferences']['pairwise_results']
            
            for (key1, key2), data in pair_results.items():
                pos1 = self.key_positions.get(key1, KeyPosition('', 0, 0, 0))
                pos2 = self.key_positions.get(key2, KeyPosition('', 0, 0, 0))
                
                pairwise_data.append({
                    'Key1': key1.upper(),
                    'Key2': key2.upper(),
                    'Key1_Finger': pos1.finger,
                    'Key1_Row': pos1.row,
                    'Key1_Column': pos1.column,
                    'Key2_Finger': pos2.finger,
                    'Key2_Row': pos2.row,
                    'Key2_Column': pos2.column,
                    'Key1_Preference_Rate': data['preference_rate'],
                    'CI_Lower': data['ci_lower'],
                    'CI_Upper': data['ci_upper'],
                    'P_Value': data['p_value'],
                    'N_Instances': data['n_instances'],
                    'Effect_Size': abs(data['preference_rate'] - 0.5),
                    'Favored_Key': key1.upper() if data['preference_rate'] > 0.5 else key2.upper(),
                    'Strength_of_Preference': max(data['preference_rate'], 1 - data['preference_rate'])
                })
            
            pairwise_df = pd.DataFrame(pairwise_data)
            pairwise_df = pairwise_df.sort_values('Strength_of_Preference', ascending=False)
            pairwise_df.to_csv(os.path.join(output_folder, 'key_preferences_bigram_pairs.csv'), index=False)
        
        # 4. Combined Key Preference Comparison
        self._save_combined_key_comparison(results, output_folder)
    
    def _save_combined_key_comparison(self, results: Dict[str, Any], output_folder: str) -> None:
        """Save a combined comparison table of all key preference methods."""
        
        combined_data = []
        
        # Get all available keys
        all_keys = set()
        
        # From same-letter
        if 'same_letter_preferences' in results and results['same_letter_preferences'].get('status') != 'insufficient_data':
            sl_rankings = results['same_letter_preferences'].get('rankings', [])
            all_keys.update([key for key, _ in sl_rankings])
                
        logger.info("Detailed key preference tables saved")
        logger.info("- key_preferences_same_letter_pairs_BT.csv: Pure key quality rankings")
        logger.info("- key_preferences_bigram_pair_comparisons.csv: Specific key pair preferences") 

    def _create_visualizations(self, results: Dict[str, Any], output_folder: str) -> None:
        """Create visualization plots."""
        
        # Key preferences visualization
        if 'same_letter_preferences' in results and results['same_letter_preferences'].get('status') != 'insufficient_data':
            self._create_key_preference_plot(results['same_letter_preferences'], output_folder)
                    
        # Pairwise comparisons visualization  
        if 'pairwise_preferences' in results:
            self._create_pairwise_plot(results['pairwise_preferences'], output_folder)
        
        # Constrained inward/outward roll visualization
        if 'inward_outward_roll' in results and results['inward_outward_roll'].get('n_instances', 0) > 0:
            self._create_constrained_inward_outward_plot(results['inward_outward_roll'], output_folder)

        # Side reach movement visualization
        if 'side_reach' in results and results['side_reach'].get('n_instances', 0) > 0:
            self._create_side_reach_movement_plot(results['side_reach'], output_folder)

        logger.info(f"Visualizations saved to {output_folder}")

    def _create_key_preference_plot(self, same_letter_results: Dict[str, Any], output_folder: str) -> None:
        """Create key preference visualization."""
        
        if 'rankings' not in same_letter_results:
            return
        
        rankings = same_letter_results['rankings']

        cis = same_letter_results.get('confidence_intervals', {})
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        keys = [key for key, _ in rankings]
        strengths = [strength for _, strength in rankings]
        ci_lowers = [cis.get(key, (0, 0))[0] for key, _ in rankings]
        ci_uppers = [cis.get(key, (0, 0))[1] for key, _ in rankings]
        
        # Reverse for plotting (highest at top)
        keys.reverse()
        strengths.reverse()
        ci_lowers.reverse()
        ci_uppers.reverse()
        
        y_pos = range(len(keys))
        
        # Plot confidence intervals
        for i, (lower, upper) in enumerate(zip(ci_lowers, ci_uppers)):
            if not np.isnan(lower):
                ax.plot([lower, upper], [i, i], color='gray', linewidth=2, alpha=0.6)
        
        # Plot points
        colors = ['green' if s > 0 else 'red' for s in strengths]
        ax.scatter(strengths, y_pos, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{key.upper()}" for key in keys])
        ax.set_xlabel('Bradley-Terry Strength (95% CI)', fontsize=12)
        ax.set_ylabel('Key', fontsize=12)
        ax.set_title('Key Preference Strengths (Frequency-Weighted)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'key_preference_strengths.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_pairwise_plot(self, pairwise_results: Dict[str, Any], output_folder: str) -> None:
        """Create pairwise comparison visualization."""
        
        if 'pairwise_results' not in pairwise_results:
            return
        
        pair_data = pairwise_results['pairwise_results']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort pairs by preference strength
        sorted_pairs = sorted(pair_data.items(), 
                            key=lambda x: abs(x[1]['preference_rate'] - 0.5), reverse=True)
        
        pair_labels = [f"{pair[0].upper()}-{pair[1].upper()}" for (pair, _) in sorted_pairs]
        preference_rates = [data['preference_rate'] for (_, data) in sorted_pairs]
        ci_lowers = [data['ci_lower'] for (_, data) in sorted_pairs]
        ci_uppers = [data['ci_upper'] for (_, data) in sorted_pairs]
        
        y_pos = range(len(pair_labels))
        
        # Plot confidence intervals
        for i, (lower, upper) in enumerate(zip(ci_lowers, ci_uppers)):
            ax.plot([lower, upper], [i, i], color='gray', linewidth=2, alpha=0.6)
        
        # Plot points
        colors = ['green' if abs(rate - 0.5) > 0.1 else 'orange' for rate in preference_rates]
        ax.scatter(preference_rates, y_pos, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pair_labels, fontsize=8)
        ax.set_xlabel('First Key Preference Rate (95% CI)', fontsize=12)
        ax.set_title('Pairwise Key Comparisons', fontsize=14)
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'key_preferences_bigram_pair_comparisons.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_constrained_inward_outward_plot(self, inward_outward_results: Dict[str, Any], output_folder: str) -> None:
        """Create constrained inward vs outward roll visualization."""
        
        pattern_data = inward_outward_results.get('pattern_results', {})
        
        if not pattern_data:
            # Simple single-point plot if no pattern breakdown
            fig, ax = plt.subplots(figsize=(10, 8))
            
            pref_rate = inward_outward_results['inward_preference_rate']
            ci_lower = inward_outward_results.get('ci_lower', pref_rate)
            ci_upper = inward_outward_results.get('ci_upper', pref_rate)
            n_instances = inward_outward_results.get('n_instances', 0)
            
            # Plot confidence interval
            ax.plot([0, 0], [ci_lower, ci_upper], color='gray', linewidth=6, alpha=0.6)
            
            # Plot point
            color = 'blue' if pref_rate > 0.5 else 'red'
            ax.scatter([0], [pref_rate], c=color, s=300, alpha=0.8, 
                      edgecolors='black', linewidth=2, zorder=5)
            
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 1)
            ax.set_xticks([0])
            ax.set_xticklabels(['Inward vs\nOutward Roll\n(Constrained)'])
            ax.set_ylabel('Inward Roll Preference Rate', fontsize=14)
            ax.set_title('Constrained Inward vs Outward Roll Preference\n(Same Key Pairs, Different Directions)', fontsize=16)
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Annotation
            preference = "Inward" if pref_rate > 0.5 else "Outward"
            ax.annotate(f'{pref_rate:.1%}\n({preference})\nn={n_instances}', 
                       (0, pref_rate), textcoords="offset points", 
                       xytext=(0,25), ha='center', fontsize=12, weight='bold')
            
        else:
            # Multi-pattern plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            patterns = list(pattern_data.keys())
            preference_rates = [data['preference_rate'] for data in pattern_data.values()]
            ci_lowers = [data['ci_lower'] for data in pattern_data.values()]
            ci_uppers = [data['ci_upper'] for data in pattern_data.values()]
            n_instances = [data['n_instances'] for data in pattern_data.values()]
            
            x_pos = range(len(patterns))
            
            # Plot confidence intervals
            for i, (lower, upper) in enumerate(zip(ci_lowers, ci_uppers)):
                ax.plot([i, i], [lower, upper], color='gray', linewidth=3, alpha=0.6)
            
            # Plot points
            colors = ['blue' if rate > 0.5 else 'red' for rate in preference_rates]
            ax.scatter(x_pos, preference_rates, c=colors, s=150, alpha=0.7, 
                      edgecolors='black', linewidth=2)
            
            # Formatting
            ax.set_xticks(x_pos)
            pattern_labels = [p.replace('_', ' ').title() for p in patterns]
            ax.set_xticklabels(pattern_labels)
            ax.set_ylabel('Inward Roll Preference Rate', fontsize=14)
            ax.set_title('Constrained Inward vs Outward Roll by Movement Pattern', fontsize=16)
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Annotations
            for i, (rate, n) in enumerate(zip(preference_rates, n_instances)):
                preference = "Inward" if rate > 0.5 else "Outward"
                ax.annotate(f'{rate:.1%}\n({preference})\nn={n}', 
                           (i, rate), textcoords="offset points", 
                           xytext=(0,15), ha='center', fontsize=10, weight='bold')
        
        ax.text(0.02, 0.98, 'Constrained: Same key pairs in both directions (e.g., \'df\' vs \'fd\')', 
               transform=ax.transAxes, fontsize=10, style='italic', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'constrained_inward_outward_roll.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_side_reach_movement_plot(self, side_reach_results: Dict[str, Any], output_folder: str) -> None:
        """Create same-row side reach analysis visualization."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        pref_rate = side_reach_results['standard_area_preference_rate']
        ci_lower = side_reach_results.get('ci_lower', pref_rate)
        ci_upper = side_reach_results.get('ci_upper', pref_rate)
        n_instances = side_reach_results.get('n_instances', 0)
        
        # Plot confidence interval as vertical line
        ax.plot([0, 0], [ci_lower, ci_upper], color='gray', linewidth=6, alpha=0.6, label='95% CI')
        
        # Plot preference point
        if pref_rate > 0.6:
            color = 'green'
            interpretation = 'Strong avoidance'
        elif pref_rate > 0.5:
            color = 'blue'
            interpretation = 'Mild avoidance'
        elif pref_rate > 0.4:
            color = 'orange' 
            interpretation = 'Mild acceptance'
        else:
            color = 'red'
            interpretation = 'Strong acceptance'
            
        ax.scatter([0], [pref_rate], c=color, s=300, alpha=0.8, 
                  edgecolors='black', linewidth=2, zorder=5)
        
        # Formatting
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(0, 1)
        ax.set_xticks([0])
        ax.set_xticklabels(['Same-Row\nSide Reach\nPreference'])
        ax.set_ylabel('Standard Area Preference Rate', fontsize=14)
        ax.set_title('Same-Row Side Reach Analysis\n(Standard Area vs Column 5 Extension)', fontsize=16)
        
        # Reference lines
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, 
                  label='No preference (50%)')
        ax.axhline(y=0.6, color='green', linestyle=':', alpha=0.5, 
                  label='Strong avoidance (60%)')
        ax.axhline(y=0.4, color='red', linestyle=':', alpha=0.5, 
                  label='Acceptance (40%)')
        
        ax.grid(True, alpha=0.3)
        
        # Main annotation
        ax.annotate(f'{pref_rate:.1%}\n{interpretation}\n(n={n_instances})', 
                   (0, pref_rate), textcoords="offset points", 
                   xytext=(100, 0), ha='left', va='center',
                   fontsize=12, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
        
        # Example annotations
        ax.text(-0.75, 0.85, 'Standard Area\n(Same-row):\n• qw, er, wr\n• as, df, sf\n• zx, cv, xv', 
               fontsize=10, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
        
        ax.text(-0.75, 0.15, 'Side Reach\n(Same-row):\n• qt, et, wt\n• ag, dg, sg\n• zb, cb, xb', 
               fontsize=10, va='bottom', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
        
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        ax.text(0.02, 0.02, 'Method: Same-row bigrams only, no same-column movements', 
               transform=ax.transAxes, fontsize=10, style='italic',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'same_row_side_reach_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # DETAILED DIAGNOSTIC REPORTING
    # =========================================================================

    def _generate_comprehensive_diagnostic_csv(self, results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive CSV with full diagnosis of each bigram pair."""
        
        # Get Bradley-Terry scores if available
        key_scores = {}
        if 'same_letter_preferences' in results and 'rankings' in results['same_letter_preferences']:
            rankings = results['same_letter_preferences']['rankings']
            for key, strength in rankings:
                key_scores[key] = strength
        
        diagnostic_rows = []
        
        for i, (_, row) in enumerate(self.data.iterrows()):
            try:
                chosen = str(row['chosen_bigram']).lower()
                unchosen = str(row['unchosen_bigram']).lower()
                
                # Basic information
                diagnostic_row = {
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'slider_value': row.get('sliderValue', 0),
                    'chosen_selected': 1,  # Always 1 since this is the chosen bigram
                }
                
                # Analyze chosen bigram
                logger.debug(f"Row {i}: Analyzing chosen bigram '{chosen}'")
                chosen_diagnosis = self._diagnose_bigram(chosen, key_scores)
                if chosen_diagnosis is None:
                    logger.error(f"Row {i}: _diagnose_bigram returned None for chosen '{chosen}'")
                    continue
                
                logger.debug(f"Row {i}: chosen_diagnosis type: {type(chosen_diagnosis)}")
                try:
                    for key, value in chosen_diagnosis.items():
                        diagnostic_row[f'chosen_{key}'] = value
                except Exception as e:
                    logger.error(f"Row {i}: Error iterating chosen_diagnosis.items(): {e}")
                    logger.error(f"Row {i}: chosen_diagnosis = {chosen_diagnosis}")
                    raise
                    
                # Analyze unchosen bigram  
                logger.debug(f"Row {i}: Analyzing unchosen bigram '{unchosen}'")
                unchosen_diagnosis = self._diagnose_bigram(unchosen, key_scores)
                if unchosen_diagnosis is None:
                    logger.error(f"Row {i}: _diagnose_bigram returned None for unchosen '{unchosen}'")
                    continue
                
                logger.debug(f"Row {i}: unchosen_diagnosis type: {type(unchosen_diagnosis)}")
                try:
                    for key, value in unchosen_diagnosis.items():
                        diagnostic_row[f'unchosen_{key}'] = value
                except Exception as e:
                    logger.error(f"Row {i}: Error iterating unchosen_diagnosis.items(): {e}")
                    logger.error(f"Row {i}: unchosen_diagnosis = {unchosen_diagnosis}")
                    raise
                
                # Comparative metrics
                logger.debug(f"Row {i}: Computing comparative metrics")
                try:
                    comparative_metrics = self._compute_comparative_metrics(chosen, unchosen, chosen_diagnosis, unchosen_diagnosis)
                    if comparative_metrics is None:
                        logger.error(f"Row {i}: _compute_comparative_metrics returned None")
                        continue
                    diagnostic_row.update(comparative_metrics)
                except Exception as e:
                    logger.error(f"Row {i}: Error in _compute_comparative_metrics: {e}")
                    raise
                
                # Analysis participation flags
                logger.debug(f"Row {i}: Computing analysis participation")
                try:
                    analysis_participation = self._compute_analysis_participation(chosen, unchosen, chosen_diagnosis, unchosen_diagnosis)
                    if analysis_participation is None:
                        logger.error(f"Row {i}: _compute_analysis_participation returned None")
                        continue
                    diagnostic_row.update(analysis_participation)
                except Exception as e:
                    logger.error(f"Row {i}: Error in _compute_analysis_participation: {e}")
                    raise
                
                diagnostic_rows.append(diagnostic_row)
                
            except Exception as e:
                logger.error(f"Row {i}: Failed to process row: {e}")
                logger.error(f"Row {i}: chosen='{chosen}', unchosen='{unchosen}'")
                raise  # Re-raise to see the full stack trace
        
        # Convert to DataFrame and save
        logger.info(f"Converting {len(diagnostic_rows)} rows to DataFrame")
        diagnostic_df = pd.DataFrame(diagnostic_rows)
        
        # Add summary columns
        logger.info("Adding summary columns")
        diagnostic_df = self._add_summary_columns(diagnostic_df)
        
        # Save to CSV
        csv_path = os.path.join(output_folder, 'comprehensive_bigram_diagnosis.csv')
        diagnostic_df.to_csv(csv_path, index=False)
        
        logger.info(f"Comprehensive diagnostic CSV saved to {csv_path}")
        logger.info(f"Contains {len(diagnostic_df)} bigram pair comparisons with full diagnosis")
        
        # Also save a data dictionary
        self._save_diagnostic_csv_dictionary(output_folder)

    def _diagnose_bigram(self, bigram: str, key_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive diagnosis for a single bigram."""
        
        if len(bigram) != 2:
            return self._empty_diagnosis()
        
        key1, key2 = bigram[0], bigram[1]
        
        # Basic key information
        diagnosis = {
            'key1': key1,
            'key2': key2,
            'length': len(bigram),
            'is_same_letter': key1 == key2,
            'is_left_hand': self._all_keys_in_left_hand(bigram)
        }
        
        # Key position information
        if key1 in self.key_positions:
            pos1 = self.key_positions[key1]
            diagnosis.update({
                'key1_row': pos1.row,
                'key1_column': pos1.column, 
                'key1_finger': pos1.finger
            })
        else:
            diagnosis.update({
                'key1_row': -1,
                'key1_column': -1,
                'key1_finger': -1
            })
            
        if key2 in self.key_positions:
            pos2 = self.key_positions[key2]
            diagnosis.update({
                'key2_row': pos2.row,
                'key2_column': pos2.column,
                'key2_finger': pos2.finger
            })
        else:
            diagnosis.update({
                'key2_row': -1,
                'key2_column': -1, 
                'key2_finger': -1
            })
        
        # Separation metrics
        diagnosis['row_separation'] = self._calculate_row_separation(bigram)
        diagnosis['col_separation'] = self._calculate_column_separation(bigram)
        
        # Movement patterns
        diagnosis['row_pattern'] = self._classify_row_pattern(diagnosis['row_separation'])
        diagnosis['col_pattern'] = self._classify_col_pattern(diagnosis['col_separation'])
        
        # Key quality scores (Bradley-Terry if available)
        diagnosis['key1_bt_score'] = key_scores.get(key1, np.nan)
        diagnosis['key2_bt_score'] = key_scores.get(key2, np.nan)
        
        # Safe average calculation
        scores = [diagnosis['key1_bt_score'], diagnosis['key2_bt_score']]
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            diagnosis['avg_key_score'] = np.mean(valid_scores)
        else:
            diagnosis['avg_key_score'] = np.nan
        
        # Safe difference calculation    
        if not np.isnan(diagnosis['key1_bt_score']) and not np.isnan(diagnosis['key2_bt_score']):
            diagnosis['key_score_diff'] = diagnosis['key1_bt_score'] - diagnosis['key2_bt_score']
        else:
            diagnosis['key_score_diff'] = np.nan
                    
        # Roll direction classification
        diagnosis['roll_direction'] = self._classify_roll_direction(bigram)

        # Side reach movement classification
        diagnosis['requires_side_reach'] = self._requires_side_reach_movement(bigram)
        diagnosis['side_reach_movement_type'] = self._classify_side_reach_movement(bigram)
        
        # Ergonomic classification
        diagnosis['movement_type'] = self._classify_movement_type(diagnosis['row_separation'], diagnosis['col_separation'])
        diagnosis['same_finger'] = diagnosis['col_separation'] == 0
        diagnosis['finger_coordination'] = self._classify_finger_coordination(diagnosis['col_separation'])
        
        return diagnosis

    def _empty_diagnosis(self) -> Dict[str, Any]:
        """Return empty diagnosis for invalid bigrams."""
        return {
            'key1': '', 'key2': '', 'length': 0, 'is_same_letter': False, 'is_left_hand': False,
            'key1_row': -1, 'key1_column': -1, 'key1_finger': -1,
            'key2_row': -1, 'key2_column': -1, 'key2_finger': -1,
            'row_separation': -1, 'col_separation': -1,
            'row_pattern': 'invalid', 'col_pattern': 'invalid',
            'key1_bt_score': np.nan, 'key2_bt_score': np.nan, 'avg_key_score': np.nan, 'key_score_diff': np.nan,
            'col4_vs_5_type': 'invalid', 'movement_type': 'invalid', 'same_finger': False, 'finger_coordination': 'invalid'
        }

    def _classify_row_pattern(self, row_separation: int) -> str:
        """Classify row movement pattern."""
        if row_separation == 0:
            return 'same_row'
        elif row_separation == 1:
            return 'reach'
        elif row_separation == 2:
            return 'hurdle'
        else:
            return 'extreme' if row_separation > 2 else 'invalid'

    def _classify_col_pattern(self, col_separation: int) -> str:
        """Classify column movement pattern."""
        if col_separation == 0:
            return 'same_column'
        elif col_separation == 1:
            return 'adjacent'
        elif col_separation == 2:
            return 'distant' 
        elif col_separation == 3:
            return 'very_distant'
        else:
            return 'extreme' if col_separation > 3 else 'invalid'

    def _classify_movement_type(self, row_sep: int, col_sep: int) -> str:
        """Classify overall movement type."""
        if row_sep == 0 and col_sep == 0:
            return 'same_key'
        elif col_sep == 0:
            return 'same_finger'  
        elif row_sep == 0:
            return 'same_row'
        elif row_sep == 1 and col_sep == 1:
            return 'adjacent_reach'
        elif row_sep == 2 and col_sep == 1:
            return 'adjacent_hurdle'
        elif row_sep == 1 and col_sep == 2:
            return 'distant_reach'
        elif row_sep == 2 and col_sep == 2:
            return 'distant_hurdle'
        else:
            return 'complex'

    def _classify_finger_coordination(self, col_separation: int) -> str:
        """Classify finger coordination requirements."""
        if col_separation == 0:
            return 'single_finger'
        elif col_separation == 1:
            return 'adjacent_fingers'
        elif col_separation == 2:
            return 'distant_fingers'
        elif col_separation == 3:
            return 'extreme_spread'
        else:
            return 'invalid'

    def _compute_comparative_metrics(self, chosen: str, unchosen: str, chosen_diag: Dict, unchosen_diag: Dict) -> Dict[str, Any]:
        """Compute comparative metrics between chosen and unchosen bigrams."""

        return {
            # Separation comparisons
            'row_sep_advantage': chosen_diag['row_separation'] - unchosen_diag['row_separation'], # negative = chosen has smaller row sep
            'col_sep_advantage': chosen_diag['col_separation'] - unchosen_diag['col_separation'], # negative = chosen has smaller col sep
            'chosen_smaller_row_sep': 1 if chosen_diag['row_separation'] < unchosen_diag['row_separation'] else 0,
            'chosen_smaller_col_sep': 1 if chosen_diag['col_separation'] < unchosen_diag['col_separation'] else 0,
            'bigrams_share_one_key': 1 if self._bigrams_share_one_key(chosen, unchosen) else 0,
            
            # Key quality comparisons
            'key_quality_advantage': chosen_diag['avg_key_score'] - unchosen_diag['avg_key_score'] if not np.isnan(chosen_diag['avg_key_score']) and not np.isnan(unchosen_diag['avg_key_score']) else np.nan,
            'chosen_better_keys': 1 if not np.isnan(chosen_diag['avg_key_score']) and not np.isnan(unchosen_diag['avg_key_score']) and chosen_diag['avg_key_score'] > unchosen_diag['avg_key_score'] else 0,
            
            # Pattern comparisons
            'same_row_pattern': 1 if chosen_diag['row_pattern'] == unchosen_diag['row_pattern'] else 0,
            'same_col_pattern': 1 if chosen_diag['col_pattern'] == unchosen_diag['col_pattern'] else 0,
            'same_movement_type': 1 if chosen_diag['movement_type'] == unchosen_diag['movement_type'] else 0,
            
            # Finger coordination comparison
            'chosen_same_finger': 1 if chosen_diag['same_finger'] else 0,
            'unchosen_same_finger': 1 if unchosen_diag['same_finger'] else 0,
            'finger_coordination_change': f"{unchosen_diag['finger_coordination']}_to_{chosen_diag['finger_coordination']}"
        }

    def _compute_analysis_participation(self, chosen: str, unchosen: str, chosen_diag: Dict, unchosen_diag: Dict) -> Dict[str, Any]:
        """Determine which analyses this comparison participates in."""
        
        participation = {
            'in_same_letter_analysis': 0,
            'in_row_separation_analysis': 0, 
            'in_col_separation_analysis': 0,
            'in_reach_vs_hurdle_analysis': 0,
            'in_side_reach_analysis': 0,
            'in_inward_outward_analysis': 0,
            'analysis_tags': []
        }
        
        # Get row and column separations FIRST
        chosen_row_sep = chosen_diag.get('row_separation', -1)
        unchosen_row_sep = unchosen_diag.get('row_separation', -1)
        chosen_col_sep = chosen_diag.get('col_separation', -1)
        unchosen_col_sep = unchosen_diag.get('col_separation', -1)
        
        # Same letter analysis
        if (chosen_diag.get('is_same_letter', False) and unchosen_diag.get('is_same_letter', False) and 
            chosen_diag.get('is_left_hand', False) and unchosen_diag.get('is_left_hand', False) and
            chosen_diag.get('key1', '') != unchosen_diag.get('key1', '')):
            participation['in_same_letter_analysis'] = 1
            participation['analysis_tags'].append('same_letter_bt')
        
        # Row separation analysis  
        if (chosen_diag.get('is_left_hand', False) and unchosen_diag.get('is_left_hand', False) and
            {chosen_row_sep, unchosen_row_sep} in [{0, 1}, {1, 2}]):
            participation['in_row_separation_analysis'] = 1
            participation['analysis_tags'].append('row_separation')

        # Column separation analysis (with shared key constraint)
        if (chosen_diag.get('is_left_hand', False) and unchosen_diag.get('is_left_hand', False) and
            chosen_row_sep == unchosen_row_sep and chosen_row_sep > 0 and
            self._bigrams_share_one_key(chosen, unchosen)):
            
            # Same vs other test (matches main analysis)
            if (chosen_col_sep == 0 and unchosen_col_sep > 0) or (chosen_col_sep > 0 and unchosen_col_sep == 0):
                participation['in_col_separation_analysis'] = 1
                participation['analysis_tags'].append('col_same_vs_other')
            # Adjacent vs distant test  
            elif ({chosen_col_sep, unchosen_col_sep} == {1, 2} or {chosen_col_sep, unchosen_col_sep} == {1, 3}):
                participation['in_col_separation_analysis'] = 1
                participation['analysis_tags'].append('col_adjacent_vs_distant')
        
        # Reach vs hurdle analysis (with shared key constraint)
        if (chosen_diag.get('is_left_hand', False) and unchosen_diag.get('is_left_hand', False) and
            chosen_col_sep == unchosen_col_sep and
            {chosen_row_sep, unchosen_row_sep} == {1, 2} and
            self._bigrams_share_one_key(chosen, unchosen)):
            participation['in_reach_vs_hurdle_analysis'] = 1
            participation['analysis_tags'].append('reach_vs_hurdle')
  
        # Same-row side reach analysis
        if (chosen_diag.get('is_left_hand', False) and unchosen_diag.get('is_left_hand', False) and
            self._all_keys_in_analysis_area(chosen) and self._all_keys_in_analysis_area(unchosen)):
            
            chosen_side_reach = chosen_diag.get('requires_side_reach', False)
            unchosen_side_reach = unchosen_diag.get('requires_side_reach', False)
            
            # Same-row only, no same-column, different side reach
            if (chosen_row_sep == 0 and unchosen_row_sep == 0 and  # Same-row only
                not self._is_same_column_bigram(chosen) and not self._is_same_column_bigram(unchosen) and  # No same-column
                chosen_side_reach != unchosen_side_reach):  # Different side reach
                
                participation['in_side_reach_analysis'] = 1
                participation['analysis_tags'].append('side_reach_same_row_only')
        
        # Constrained inward vs outward roll analysis
        if (chosen_diag.get('is_left_hand', False) and unchosen_diag.get('is_left_hand', False) and
            self._are_same_keys_reverse_order(chosen, unchosen) and
            not self._is_same_column_bigram(chosen)):
            
            chosen_roll = chosen_diag.get('roll_direction', 'invalid')
            unchosen_roll = unchosen_diag.get('roll_direction', 'invalid')
            
            if {chosen_roll, unchosen_roll} == {'inward', 'outward'}:
                participation['in_inward_outward_analysis'] = 1
                row_separation = chosen_diag.get('row_separation', -1)
                
                if row_separation == 0:
                    participation['analysis_tags'].append('inward_outward_same_row')
                elif row_separation == 1:
                    participation['analysis_tags'].append('inward_outward_reach')
                elif row_separation == 2:
                    participation['analysis_tags'].append('inward_outward_hurdle')
                else:
                    participation['analysis_tags'].append('inward_outward_extreme')
        
        # Finalize
        participation['analysis_tags'] = '|'.join(participation['analysis_tags'])
        participation['num_analyses'] = sum([participation[key] for key in participation if key.startswith('in_')])
        
        return participation

    def _add_summary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add summary columns to the diagnostic DataFrame."""
        
        # Overall preference patterns
        df['chose_smaller_overall'] = ((df['chosen_smaller_row_sep'] == 1) | (df['chosen_smaller_col_sep'] == 1)).astype(int)
        df['chose_same_finger'] = df['chosen_same_finger']
        df['chose_better_keys'] = df['chosen_better_keys']
        
        # Preference direction for slider
        df['slider_direction'] = df['slider_value'].apply(lambda x: 'strong_left' if x < -50 else 'weak_left' if x < 0 else 'weak_right' if x <= 50 else 'strong_right')
        df['slider_strength'] = df['slider_value'].abs()
        
        # Movement complexity
        df['movement_complexity'] = df.apply(lambda row: 
            'simple' if row['chosen_movement_type'] in ['same_key', 'same_finger', 'same_row'] 
            else 'moderate' if 'adjacent' in row['chosen_movement_type']
            else 'complex', axis=1)

        # Same-row side reach preference patterns
        df['chose_standard_area_same_row'] = df.apply(lambda row: 
            1 if (row['chosen_row_separation'] == 0 and row['unchosen_row_separation'] == 0 and
                  not row['chosen_requires_side_reach'] and row['unchosen_requires_side_reach'])
            else 0 if (row['chosen_row_separation'] == 0 and row['unchosen_row_separation'] == 0 and
                       row['chosen_requires_side_reach'] and not row['unchosen_requires_side_reach'])
            else np.nan, axis=1)
                
        df['comparison_involves_side_reach'] = df.apply(lambda row:
            row['chosen_requires_side_reach'] or row['unchosen_requires_side_reach'], axis=1)
        
        return df

    def _save_diagnostic_csv_dictionary(self, output_folder: str) -> None:
        """Save data dictionary explaining the diagnostic CSV columns."""
        
        dictionary_content = """
    COMPREHENSIVE BIGRAM DIAGNOSIS CSV - DATA DICTIONARY
    ====================================================

    BASIC INFORMATION:
    - user_id: Participant identifier
    - chosen_bigram: The bigram that was selected by the participant
    - unchosen_bigram: The bigram that was not selected
    - slider_value: Strength of preference (-100 to +100)
    - chosen_selected: Always 1 (indicates this row represents the chosen option)

    CHOSEN BIGRAM DIAGNOSIS (chosen_*):
    - chosen_key1/key2: Individual keys in the chosen bigram
    - chosen_length: Number of characters (should be 2)
    - chosen_is_same_letter: True if both keys are the same (e.g., 'aa')
    - chosen_is_left_hand: True if both keys are on left hand
    - chosen_key1/2_row: Row position (1=top, 2=home, 3=bottom)
    - chosen_key1/2_column: Column position (1=pinky, 4=index)
    - chosen_key1/2_finger: Finger assignment (1=pinky, 4=index)
    - chosen_row_separation: Absolute row distance between keys
    - chosen_col_separation: Absolute column distance between keys
    - chosen_row_pattern: same_row/reach/hurdle classification
    - chosen_col_pattern: same_column/adjacent/distant classification
    - chosen_key1/2_bt_score: Bradley-Terry strength scores (if available)
    - chosen_avg_key_score: Average BT score for this bigram
    - chosen_key_score_diff: Difference between key1 and key2 BT scores
    - chosen_roll_direction: inward/outward/same_finger/invalid classification
    - chosen_roll_direction: inward (finger number increases), outward (decreases), same_finger, or invalid
    - chosen_requires_side_reach: True if bigram contains column 5 keys (T, G, B)
    - chosen_side_reach_movement_type: no_side_reach/requires_side_reach/invalid classification
    - chosen_requires_side_reach: Whether typing requires reaching beyond standard left-hand area
    - chosen_side_reach_movement_type: Movement classification based on side reach requirement
    - chosen_movement_type: Detailed movement classification
    - chosen_same_finger: True if both keys use same finger
    - chosen_finger_coordination: single_finger/adjacent_fingers/distant_fingers

    UNCHOSEN BIGRAM DIAGNOSIS (unchosen_*):
    [Same structure as chosen_* but for the non-selected bigram]

    SAME-ROW SIDE REACH ANALYSIS:
    - Restricted to same-row bigrams (0 row separation) with no same-column movements
    - Compares standard left-hand area (columns 1-4) vs side reach (column 5)
    - Purest test of side reach ergonomic cost without movement complexity
    - in_side_reach_analysis: 1 if used in same-row side reach analysis
    
    COMPARATIVE METRICS:
    - row_sep_advantage: chosen_row_sep - unchosen_row_sep (negative = chosen smaller)
    - col_sep_advantage: chosen_col_sep - unchosen_col_sep (negative = chosen smaller)
    - chosen_smaller_row_sep: 1 if chosen bigram has smaller row separation
    - chosen_smaller_col_sep: 1 if chosen bigram has smaller column separation
    - key_quality_advantage: chosen_avg_key_score - unchosen_avg_key_score
    - chosen_better_keys: 1 if chosen bigram has higher average key quality
    - same_row_pattern: 1 if both bigrams have same row pattern
    - same_col_pattern: 1 if both bigrams have same column pattern
    - same_movement_type: 1 if both bigrams have same movement type
    - finger_coordination_change: Description of coordination change

    ANALYSIS PARTICIPATION FLAGS:
    - in_same_letter_analysis: 1 if this comparison was used in same-letter BT analysis
    - in_row_separation_analysis: 1 if used in row separation preference analysis
    - in_col_separation_analysis: 1 if used in column separation preference analysis
    - in_reach_vs_hurdle_analysis: 1 if used in reach vs hurdle analysis
    - in_inward_outward_analysis: 1 if used in inward vs outward roll analysis
    - in_side_reach_analysis: 1 if used in side reach movement analysis
    - analysis_tags: Pipe-separated list of specific analyses this comparison contributed to
    - num_analyses: Total number of analyses this comparison participated in

    SUMMARY COLUMNS:
    - chose_smaller_overall: 1 if chose smaller row OR column separation
    - chose_same_finger: Same as chosen_same_finger
    - chose_better_keys: Same as chosen_better_keys  
    - slider_direction: strong_left/weak_left/weak_right/strong_right
    - slider_strength: Absolute value of slider_value
    - movement_complexity: simple/moderate/complex classification

    USAGE NOTES:
    - Use this file to understand exactly which comparisons contributed to each analysis
    - Filter by analysis participation flags to recreate specific analysis datasets
    - Comparative metrics show the "advantage" the chosen bigram had over unchosen
    - Movement classifications help understand the ergonomic patterns in preferences
    """
        
        dict_path = os.path.join(output_folder, 'diagnostic_csv_dictionary.txt')
        with open(dict_path, 'w') as f:
            f.write(dictionary_content)
            
        logger.info(f"Data dictionary saved to {dict_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='MOO objectives analysis with practical significance focus'
    )
    parser.add_argument('--data', required=True,
                       help='Path to CSV file with bigram choice data')
    parser.add_argument('--output', required=True,
                       help='Output directory for results')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample size for testing (e.g., --sample 500)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return 1
    
    try:
        # Run analysis
        analyzer = MOOObjectiveAnalyzer(args.config)
        results = analyzer.analyze_moo_objectives(args.data, args.output, args.sample)
        
        logger.info("MOO objectives analysis finished successfully!")
        
        # Print summary
        summary = results['summary']
        print(f"\nSUMMARY:")
        print(f"========")
        print(f"Objectives completed: {summary['objectives_completed']}/{summary['total_objectives']}")
        print(f"Approach: {summary['approach']}")
        print(f"Focus: Effect sizes and confidence intervals for MOO design")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())