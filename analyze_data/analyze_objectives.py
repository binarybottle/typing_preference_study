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
3. Column separation: Adjacent vs. distant finger movements
4. Column 4 vs 5: Index finger column preferences

Usage:
    poetry run python3 analyze_objectives.py --data output/nonProlific/process_data/tables/processed_consistent_choices.csv \
        --output output/nonProlific/analyze_objectives
    poetry run python3 analyze_objectives.py --data output/Prolific/process_data/tables/processed_consistent_choices.csv \
        --output output/Prolific/analyze_objectives
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        result = minimize(negative_log_likelihood, initial_strengths, method='SLSQP', constraints=constraint)
        
        if not result.success:
            logger.warning("Bradley-Terry optimization did not converge")
        
        return result.x
    
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
                'regularization': 1e-10,
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

        logger.info("=== COLUMN 4 VS 5 PREFERENCES ===")  
        results['column_4_vs_5'] = self._analyze_column_4_vs_5()

        # Generate reports and save results
        enhanced_results = {
            'objectives': results,
            'summary': self._generate_summary(results)
        }
        
        logger.info("=== GENERATING REPORTS ===")
        self._generate_comprehensive_report(enhanced_results, output_folder)

        # Save detailed key preference comparison tables
        self._save_key_preference_tables(enhanced_results['objectives'], output_folder)        
        logger.info(f"Results saved to {output_folder}")
    
        # Generate visualizations
        self._create_visualizations(results, output_folder)
        
        logger.info(f"MOO objectives analysis complete! Results saved to {output_folder}")
        return enhanced_results

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
        
        # Define key pairs for analysis
        target_pairs = [
            ('f','d'), ('d','s'), ('s','a'),  # Home row
            ('r','e'), ('w','q'),  # Top row  
            ('c','x'), ('x','z'),  # Bottom row
            ('f','r'), ('d','e'), ('s','w'), ('a','q'),  # Vertical
        ]
        
        # Filter to existing keys
        valid_pairs = [(k1, k2) for k1, k2 in target_pairs if k1 in self.left_hand_keys and k2 in self.left_hand_keys]
        
        pairwise_results = {}
        
        for key1, key2 in valid_pairs:
            comparison_data = self._extract_specific_key_comparison(key1, key2)
            if comparison_data and comparison_data['n_instances'] >= 10:
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
            if len(comp_data) >= 10:
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
        
        return {
            'description': 'Preferences for smaller row separation distances',
            'method': 'row_separation_analysis',
            'n_instances': n_instances,
            'preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'comparison_results': comparison_results,
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
    # COLUMN SEPARATION PREFERENCES
    # =========================================================================

    def _analyze_column_separation(self) -> Dict[str, Any]:
        """Analyze column separation preferences."""
        
        instances_df = self._extract_column_separation_instances()
        
        if instances_df.empty:
            logger.warning('No column separation instances found - skipping column separation analysis')
            return {
                'description': 'Preferences for smaller column separation distances',
                'method': 'column_separation_analysis',
                'status': 'insufficient_data',
                'n_instances': 0,
                'interpretation': 'No column separation instances found in data'
            }
        
        logger.info(f"Found {len(instances_df)} column separation instances")
        
        # Overall analysis
        n_instances = len(instances_df)
        n_chose_smaller = instances_df['chose_smaller_separation'].sum()
        preference_rate = n_chose_smaller / n_instances
        
        confidence = self.config.get('confidence_level', 0.95)
        ci_lower, ci_upper = self._wilson_ci(n_chose_smaller, n_instances, confidence)
        
        # Statistical test
        z_score = (preference_rate - 0.5) / np.sqrt(0.25 / n_instances)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        # Analysis by row pattern
        pattern_results = {}
        for pattern in instances_df['row_pattern'].unique():
            pattern_data = instances_df[instances_df['row_pattern'] == pattern]
            if len(pattern_data) >= 10:
                n_pat = len(pattern_data)
                n_chose_pat = pattern_data['chose_smaller_separation'].sum()
                pref_rate_pat = n_chose_pat / n_pat
                ci_lower_pat, ci_upper_pat = self._wilson_ci(n_chose_pat, n_pat, confidence)
                
                z_pat = (pref_rate_pat - 0.5) / np.sqrt(0.25 / n_pat)
                p_pat = 2 * (1 - norm.cdf(abs(z_pat)))
                
                pattern_results[pattern] = {
                    'n_instances': n_pat,
                    'preference_rate': pref_rate_pat,
                    'ci_lower': ci_lower_pat,
                    'ci_upper': ci_upper_pat,
                    'p_value': p_pat
                }
        
        return {
            'description': 'Preferences for smaller column separation distances',
            'method': 'column_separation_analysis',
            'n_instances': n_instances,
            'preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'pattern_results': pattern_results,
            'interpretation': f"Column separation preference: {preference_rate:.1%} favor smaller distances"
        }
    
    def _extract_column_separation_instances(self) -> pd.DataFrame:
        """Extract column separation instances with row control."""
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
            
            # Only compare bigrams with identical row separation
            if chosen_row_sep != unchosen_row_sep:
                continue
            
            # Exclude same-column comparisons
            if chosen_col_sep == 0 or unchosen_col_sep == 0:
                continue
            
            # Define row pattern
            if chosen_row_sep == 0:
                row_pattern = "same_row"
            elif chosen_row_sep == 1:
                row_pattern = "reach_1_apart"
            elif chosen_row_sep == 2:
                row_pattern = "hurdle_2_apart"
            else:
                continue
            
            # Column separation comparison
            if {chosen_col_sep, unchosen_col_sep} == {1, 2}:
                chose_smaller = 1 if chosen_col_sep == 1 else 0
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'row_pattern': row_pattern,
                    'chose_smaller_separation': chose_smaller,
                    'chosen_col_separation': chosen_col_sep,
                    'unchosen_col_separation': unchosen_col_sep
                })
            
        return pd.DataFrame(instances)
    
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
    # COLUMN 4 VS 5 PREFERENCES
    # =========================================================================

    def _analyze_column_4_vs_5(self) -> Dict[str, Any]:
        """Analyze preference for column 4 (RFV) vs column 5 (TGB)."""
        
        instances_df = self._extract_column_4_vs_5_instances()
        
        if instances_df.empty:
            logger.warning('No column 4 vs 5 instances found - skipping this analysis')
            return {
                'description': 'Column 4 (RFV) vs Column 5 (TGB) preference',
                'method': 'column_4_vs_5_analysis',
                'status': 'insufficient_data',
                'n_instances': 0,
                'interpretation': 'No column 4 vs 5 comparisons found in data'
            }
        
        logger.info(f"Found {len(instances_df)} column 4 vs 5 instances")
        
        # Analysis
        n_instances = len(instances_df)
        n_chose_col4 = instances_df['chose_column_4'].sum()
        preference_rate = n_chose_col4 / n_instances
        
        confidence = self.config.get('confidence_level', 0.95)
        ci_lower, ci_upper = self._wilson_ci(n_chose_col4, n_instances, confidence)
        
        # Statistical test
        z_score = (preference_rate - 0.5) / np.sqrt(0.25 / n_instances)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return {
            'description': 'Column 4 (RFV) vs Column 5 (TGB) preference',
            'method': 'column_4_vs_5_analysis',
            'n_instances': n_instances,
            'preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'interpretation': f"Column 4 preference rate: {preference_rate:.1%}"
        }

    def _extract_column_4_vs_5_instances(self) -> pd.DataFrame:
        """Extract instances comparing column 4 vs column 5 bigrams."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            chosen_type = self._classify_column_4_vs_5(chosen)
            unchosen_type = self._classify_column_4_vs_5(unchosen)
            
            if {chosen_type, unchosen_type} == {'column_4', 'column_5'}:
                chose_column_4 = 1 if chosen_type == 'column_4' else 0
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'chose_column_4': chose_column_4
                })
        
        return pd.DataFrame(instances)

    def _classify_column_4_vs_5(self, bigram: str) -> str:
        """Classify bigram as column 4, column 5, or neither based on key presence."""
        col4_keys = {'r', 'f', 'v'}
        col5_keys = self._get_column_5_keys()
        
        bigram_keys = set(bigram)
        
        # Check which column keys are present
        has_col4 = bool(bigram_keys.intersection(col4_keys))
        has_col5 = bool(bigram_keys.intersection(col5_keys))
        
        # Classify based on presence, excluding mixed cases
        if has_col4 and not has_col5:
            return 'column_4'
        elif has_col5 and not has_col4:
            return 'column_5'
        else:
            # Neither (no column 4/5 keys) or mixed (contains both)
            return 'neither'

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
            "METHODOLOGY:",
            "- Effect sizes and confidence intervals quantify practical differences",
            "- Statistical tests confirm preferences are detectable above chance",
            "- No corrections across objectives (each addresses distinct mechanics)",
            "- Results inform engineering decisions for keyboard optimization",
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
                f"Result: {obj_results.get('interpretation', 'No interpretation')}",
                ""
            ])
            
            # Add specific results based on objective type
            if 'bradley_terry' in obj_name and 'rankings' in obj_results:
                if 'weighted_rankings' in obj_results:
                    rankings = obj_results['weighted_rankings']
                    cis = obj_results.get('confidence_intervals', {})
                    
                    report_lines.extend([
                        "  Top 5 preferred keys (weighted Bradley-Terry):",
                        ""
                    ])
                    for j, (key, strength) in enumerate(rankings[:5], 1):
                        ci_lower, ci_upper = cis.get(key, (np.nan, np.nan))
                        ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]" if not np.isnan(ci_lower) else "[CI unavailable]"
                        
                        pos = self.key_positions.get(key, KeyPosition('', 0, 0, 0))
                        report_lines.append(f"    {j}. {key.upper()} (F{pos.finger}, R{pos.row}): {strength:.3f} {ci_str}")
                    
                    # Show frequency weighting effects
                    if 'frequency_comparison' in obj_results:
                        freq_comp = obj_results['frequency_comparison']
                        max_change = freq_comp.get('max_rank_change', 0)
                        large_changes = freq_comp.get('large_changes', [])
                        
                        report_lines.extend([
                            "",
                            f"  Frequency weighting effects:",
                            f"    Maximum rank change: {max_change} positions",
                            f"    Keys with large changes: {', '.join([k.upper() for k in large_changes]) if large_changes else 'None'}"
                        ])
                
            elif 'pairwise' in obj_name and 'pairwise_results' in obj_results:
                pair_results = obj_results['pairwise_results']
                report_lines.extend([
                    f"  Analyzed {len(pair_results)} key pairs:",
                    ""
                ])
                
                # Show top preferences
                sorted_pairs = sorted(pair_results.items(), 
                                    key=lambda x: abs(x[1]['preference_rate'] - 0.5), reverse=True)
                
                for (key1, key2), data in sorted_pairs[:5]:
                    pref_rate = data['preference_rate']
                    ci_lower, ci_upper = data['ci_lower'], data['ci_upper']
                    n_inst = data['n_instances']
                    
                    if pref_rate > 0.5:
                        winner, loser = key1.upper(), key2.upper()
                    else:
                        winner, loser = key2.upper(), key1.upper()
                        pref_rate = 1 - pref_rate
                        ci_lower, ci_upper = 1 - ci_upper, 1 - ci_lower
                    
                    report_lines.append(f"    {winner} > {loser}: {pref_rate:.1%} "
                                      f"[{ci_lower:.1%}, {ci_upper:.1%}] (n={n_inst})")
                
            elif obj_name in ['row_separation', 'column_separation'] and 'preference_rate' in obj_results:
                pref_rate = obj_results['preference_rate']
                ci_lower = obj_results['ci_lower']
                ci_upper = obj_results['ci_upper']
                
                report_lines.extend([
                    f"  Overall preference rate: {pref_rate:.1%} favor smaller distances",
                    f"  95% Confidence interval: [{ci_lower:.1%}, {ci_upper:.1%}]",
                    ""
                ])
                
                # Show breakdowns
                if 'comparison_results' in obj_results:
                    breakdown = obj_results['comparison_results']
                elif 'pattern_results' in obj_results:
                    breakdown = obj_results['pattern_results']
                else:
                    breakdown = {}
                
                if breakdown:
                    report_lines.append("  Breakdown by type:")
                    for comp_type, comp_data in breakdown.items():
                        comp_pref = comp_data['preference_rate']
                        comp_ci_lower = comp_data['ci_lower']
                        comp_ci_upper = comp_data['ci_upper']
                        comp_n = comp_data['n_instances']
                        
                        type_name = comp_type.replace('_', ' ').title()
                        report_lines.append(f"    {type_name}: {comp_pref:.1%} "
                                          f"[{comp_ci_lower:.1%}, {comp_ci_upper:.1%}] (n={comp_n})")
                    
                report_lines.append("")
        
        # Save report
        report_path = os.path.join(output_folder, 'moo_objectives_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Report saved to {report_path}")

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
        
        # From Bradley-Terry
        if 'bradley_terry_preferences' in results and results['bradley_terry_preferences'].get('status') != 'insufficient_data':
            bt_weighted = results['bradley_terry_preferences'].get('weighted_rankings', [])
            all_keys.update([key for key, _ in bt_weighted])
        
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
        if 'bradley_terry_preferences' in results:
            self._create_key_preference_plot(results['bradley_terry_preferences'], output_folder)
        
        # Pairwise comparisons visualization  
        if 'pairwise_preferences' in results:
            self._create_pairwise_plot(results['pairwise_preferences'], output_folder)
        
        logger.info(f"Visualizations saved to {output_folder}")

    def _create_key_preference_plot(self, bt_results: Dict[str, Any], output_folder: str) -> None:
        """Create key preference visualization."""
        
        if 'weighted_rankings' not in bt_results:
            return
        
        rankings = bt_results['weighted_rankings']
        cis = bt_results.get('confidence_intervals', {})
        
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