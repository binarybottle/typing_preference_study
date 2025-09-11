#!/usr/bin/env python3
"""
Complete Multi-Objective Optimization (MOO) Objectives Analysis for Keyboard Layout Optimization

This script extracts 6 typing mechanics objectives from bigram preference data,
controlling for English language frequency effects. These objectives are designed
to create meaningful conflicts for multi-objective keyboard layout optimization.

The 6 MOO objectives (not including trigram flow):
1. key_preference: Individual key quality preferences (66 pairwise comparisons)  
2. row_separation: Preferences for row transitions (same > reach > hurdle)
3. column_separation: Context-dependent column spacing preferences

Includes rigorous statistical validation with:
- Multiple comparisons correction
- Effect size validation
- Cross-validation
- Enhanced confound controls

Usage:
    poetry run python3 moo_objectives_analyzer.py --data output/nonProlific/process_data/tables/processed_consistent_choices.csv --output results
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import combine_pvalues, norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import traceback

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
    column: int   # 1=leftmost, 4=index
    finger: int   # 1=pinky, 2=ring, 3=middle, 4=index

class CompleteMOOObjectiveAnalyzer:
    """Complete analyzer for extracting and validating MOO objectives from bigram preference data."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.key_positions = self._define_keyboard_layout()
        self.left_hand_keys = set(self.key_positions.keys())
        
        # English letter frequencies
        self.english_letter_frequencies = self._load_english_letter_frequencies()

        # English bigram frequencies
        self.english_bigram_frequencies = self._load_english_bigram_frequencies()
        
        # Store data for access in methods
        self.data = None
        self.validation_results = {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('moo_objectives_analysis', {})
        else:
            return {
                'alpha_level': 0.05,
                'correction_method': 'fdr_bh',
                'min_comparisons': 20,
                'bootstrap_iterations': 1000,
                'confidence_level': 0.95,
                'figure_dpi': 300
            }
    
    def _load_english_letter_frequencies(self) -> Dict[str, float]:
        """Load English letter frequencies from CSV file."""
        try:
            freq_file = self.config.get('english_letter_frequencies_file', 'input/frequency/english-letter-counts-google-ngrams_normalized.csv')
            
            if os.path.exists(freq_file):
                df = pd.read_csv(freq_file)
                
                # Try different possible column names
                if 'letter' in df.columns and 'frequency' in df.columns:
                    freq_dict = dict(zip(df['letter'].str.lower(), df['frequency']))
                elif 'item' in df.columns and 'score' in df.columns:
                    freq_dict = dict(zip(df['item'].str.lower(), df['score']))
                else:
                    raise ValueError(f"Unexpected columns in {freq_file}: {list(df.columns)}")

                logger.info(f"Loaded {len(freq_dict)} letter frequencies from {freq_file}")
                return freq_dict

        except Exception as e:
            logger.warning(f"Error loading letter frequencies: {e}.")
    
    def _load_english_bigram_frequencies(self) -> Dict[str, float]:
        """Load English bigram frequencies from CSV file."""
        try:
            freq_file = self.config.get('english_bigram_frequencies_file', 'input/frequency/english-letter-pair-counts-google-ngrams_normalized.csv')
            
            if os.path.exists(freq_file):
                df = pd.read_csv(freq_file)
                # Expect columns: 'bigram', 'frequency' or 'item_pair', 'score'
                if 'bigram' in df.columns and 'frequency' in df.columns:
                    freq_dict = dict(zip(df['bigram'].str.lower(), df['frequency']))
                elif 'item_pair' in df.columns and 'score' in df.columns:
                    freq_dict = dict(zip(df['item_pair'].str.lower(), df['score']))

                logger.info(f"Loaded {len(freq_dict)} bigram frequencies from {freq_file}")
                return freq_dict
            else:
                raise FileNotFoundError(f"Bigram frequency file not found: {freq_file}")

        except Exception as e:
            logger.warning(f"Error loading bigram frequencies: {e}.")
        
    def _load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate the input data."""
        try:
            data = pd.read_csv(data_path)
            
            # Check required columns
            required_cols = ['user_id', 'chosen_bigram', 'unchosen_bigram', 'sliderValue']
            missing_cols = set(required_cols) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert sliderValue to numeric
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
            
            # Convert bigrams to lowercase
            data['chosen_bigram'] = data['chosen_bigram'].astype(str).str.lower()
            data['unchosen_bigram'] = data['unchosen_bigram'].astype(str).str.lower()
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _define_keyboard_layout(self) -> Dict[str, KeyPosition]:
        """Define left-hand keyboard layout."""
        return {
            # Upper row (row 1)
            'q': KeyPosition('q', 1, 1, 1),
            'w': KeyPosition('w', 1, 2, 2),
            'e': KeyPosition('e', 1, 3, 3),
            'r': KeyPosition('r', 1, 4, 4),
            
            # Middle/home row (row 2)
            'a': KeyPosition('a', 2, 1, 1),
            's': KeyPosition('s', 2, 2, 2),
            'd': KeyPosition('d', 2, 3, 3),
            'f': KeyPosition('f', 2, 4, 4),
            
            # Lower row (row 3)
            'z': KeyPosition('z', 3, 1, 1),
            'x': KeyPosition('x', 3, 2, 2),
            'c': KeyPosition('c', 3, 3, 3),
            'v': KeyPosition('v', 3, 4, 4),
        }
    
    def export_bigram_classification_table(self, output_folder: str) -> None:
        """Export diagnostic table showing how every bigram is classified for each objective."""
        
        # Get all unique bigrams from the dataset
        all_bigrams = set()
        bigram_comparison_counts = {}
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if len(chosen) == 2:
                all_bigrams.add(chosen)
                bigram_comparison_counts[chosen] = bigram_comparison_counts.get(chosen, 0) + 1
            if len(unchosen) == 2:
                all_bigrams.add(unchosen)
                bigram_comparison_counts[unchosen] = bigram_comparison_counts.get(unchosen, 0) + 1
        
        logger.info(f"Analyzing {len(all_bigrams)} unique bigrams from dataset")
        
        # Classify each bigram
        classification_data = []
        
        for bigram in sorted(all_bigrams):
            
            # Basic properties
            is_left_hand = self._all_keys_in_left_hand(bigram)
            
            if not is_left_hand:
                # Skip non-left-hand bigrams for now
                continue
                
            # Row separation
            row_separation = self._calculate_row_separation(bigram)
            row_category = ""
            if row_separation == 0:
                row_category = "same_row"
            elif row_separation == 1:
                row_category = "one_row_apart"
            elif row_separation == 2:
                row_category = "two_rows_apart"
            
            # Column separation
            col_separation = self._calculate_column_separation(bigram)
            col_category = "adjacent_cols" if col_separation <= 1 else "separated_cols"
            
            # Key positions for debugging
            key1_pos = self.key_positions.get(bigram[0], KeyPosition('', 0, 0, 0))
            key2_pos = self.key_positions.get(bigram[1], KeyPosition('', 0, 0, 0))
            
            classification_data.append({
                'bigram': bigram.upper(),
                'comparison_count': bigram_comparison_counts.get(bigram, 0),
                'is_left_hand': is_left_hand,
                'key1_finger': key1_pos.finger,
                'key2_finger': key2_pos.finger,
                'row_separation': row_separation,
                'row_category': row_category,
                'key1_row': key1_pos.row,
                'key2_row': key2_pos.row,
                'col_separation': col_separation,
                'col_category': col_category,
                'key1_col': key1_pos.column,
                'key2_col': key2_pos.column,
                'english_bigram_freq': self.english_bigram_frequencies.get(bigram, 0),
                'relevant_for_finger_test': is_left_hand,
                'relevant_for_row_separation': is_left_hand,
                'relevant_for_column_separation': is_left_hand
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(classification_data)
        
        # Add summary statistics
        summary_stats = {
            'total_left_hand_bigrams': len(df),
            'same_row_bigrams': len(df[df['row_category'] == 'same_row']),
            'one_row_apart_bigrams': len(df[df['row_category'] == 'one_row_apart']),
            'two_rows_apart_bigrams': len(df[df['row_category'] == 'two_rows_apart']),
            'adjacent_cols_bigrams': len(df[df['col_category'] == 'adjacent_cols']),
            'separated_cols_bigrams': len(df[df['col_category'] == 'separated_cols'])
        }
        
        # Save main classification table
        classification_path = os.path.join(output_folder, 'bigram_classification_diagnostic.csv')
        df.to_csv(classification_path, index=False)
        
        # Save summary statistics
        summary_path = os.path.join(output_folder, 'bigram_classification_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("BIGRAM CLASSIFICATION SUMMARY\n")
            f.write("=" * 30 + "\n\n")
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nBigrams with most comparisons:\n")
            top_bigrams = df.nlargest(10, 'comparison_count')[['bigram', 'comparison_count', 'row_category']]
            f.write(top_bigrams.to_string(index=False))
        
        logger.info(f"Bigram classification diagnostic saved to {classification_path}")
        logger.info(f"Classification summary saved to {summary_path}")
        
        # Print quick summary to console
        print(f"\nBIGRAM CLASSIFICATION DIAGNOSTIC:")
        print(f"=================================")
        for key, value in summary_stats.items():
            print(f"{key}: {value}")
        
        return df

    def analyze_moo_objectives(self, data_path: str, output_folder: str) -> Dict[str, Any]:
        """Run complete MOO objectives analysis with validation."""
        logger.info("Starting complete MOO objectives analysis with validation...")
        
        # Load and validate data
        self.data = self._load_and_validate_data(data_path)
        logger.info(f"Loaded {len(self.data)} rows from {self.data['user_id'].nunique()} participants")
        
        # Export bigram classification diagnostic table
        self.export_bigram_classification_table(output_folder)

        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        
        # Run each objective test
        results = {}

        logger.info("=== OBJECTIVE 1: KEY_PREFERENCE ===")
        results['key_preference'] = self._test_key_preference()
        
        logger.info("=== OBJECTIVE 2: ROW_SEPARATION ===")
        results['row_separation'] = self._test_row_separation_preference()

        logger.info("=== OBJECTIVE 3: COLUMN_SEPARATION ===")  
        results['column_separation'] = self._test_column_separation()
        
        # Run validation framework
        logger.info("=== RUNNING VALIDATION FRAMEWORK ===")
        validation_results = self._validate_all_objectives(results)
        
        # Combine results
        enhanced_results = {
            'objectives': results,
            'validation': validation_results,
            'summary': self._generate_enhanced_summary(results, validation_results)
        }
        
        # Generate comprehensive report
        logger.info("=== GENERATING REPORTS ===")
        self._generate_comprehensive_report(enhanced_results, output_folder)
        
        # Save results
        self._save_results(enhanced_results, output_folder)
        
        logger.info(f"Complete MOO objectives analysis finished! Results saved to {output_folder}")
        return enhanced_results
    
    def _calculate_weighted_preferences(self, instances_df: pd.DataFrame, outcome_var: str) -> Dict[str, Any]:
        """Calculate preference rates weighted by comparison frequency."""
        
        # Create unique comparison identifier
        instances_df['comparison'] = instances_df.apply(
            lambda row: tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']])), axis=1
        )
        
        # Get frequency of each comparison
        freq_counts = instances_df['comparison'].value_counts()
        
        # Weight each instance by 1/frequency
        instances_df['weight'] = instances_df['comparison'].map(lambda x: 1.0 / freq_counts[x])
        
        # Calculate overall weighted preference rate
        total_weight = instances_df['weight'].sum()
        weighted_preference = (instances_df[outcome_var] * instances_df['weight']).sum() / total_weight
        
        # Calculate by comparison type if available
        weighted_by_type = {}
        if 'comparison_type' in instances_df.columns:
            for comp_type in instances_df['comparison_type'].unique():
                comp_data = instances_df[instances_df['comparison_type'] == comp_type]
                if len(comp_data) > 0:
                    comp_total_weight = comp_data['weight'].sum()
                    comp_weighted_rate = (comp_data[outcome_var] * comp_data['weight']).sum() / comp_total_weight
                    weighted_by_type[comp_type] = comp_weighted_rate
        
        return {
            'weighted_preference_rate': weighted_preference,
            'unweighted_preference_rate': instances_df[outcome_var].mean(),
            'weighted_by_type': weighted_by_type,
            'frequency_stats': {
                'min_frequency': freq_counts.min(),
                'max_frequency': freq_counts.max(),
                'unique_comparisons': len(freq_counts)
            }
        }

    # =========================================================================
    # OBJECTIVE 1: KEY PREFERENCE  
    # =========================================================================

    def _extract_key_preference_instances(self) -> pd.DataFrame:
        """Extract instances where bigrams either share 1 key or have same row separation."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
            
            # Calculate row separations for both bigrams
            chosen_row_sep = self._calculate_row_separation(chosen)
            unchosen_row_sep = self._calculate_row_separation(unchosen)
            
            # Find shared keys
            chosen_keys = set(chosen)
            unchosen_keys = set(unchosen)
            shared_keys = chosen_keys & unchosen_keys
            keys_only_in_chosen = chosen_keys - unchosen_keys
            keys_only_in_unchosen = unchosen_keys - chosen_keys
            
            # Condition 1: Share exactly 1 key (e.g., QA vs QW)
            condition_1 = (len(shared_keys) == 1 and 
                           len(keys_only_in_chosen) == 1 and 
                           len(keys_only_in_unchosen) == 1)
            
            # Condition 2: Same row separation (e.g., QA vs CD, both 1-row separation)
            condition_2 = (chosen_row_sep == unchosen_row_sep and 
                           chosen != unchosen)  # Different bigrams
            
            if condition_1:
                # Shared key case
                # Compare the unique keys only
                chosen_unique_keys = list(keys_only_in_chosen)
                unchosen_unique_keys = list(keys_only_in_unchosen)
                comparison_type = "shared_key"
                shared_context = ''.join(sorted(shared_keys))
                
            elif condition_2:
                # Same row separation case
                # Compare every possible pair of keys across the two key-pairs
                # (e.g., QA vs CD -> Q vs C and Q vs D and A vs C and A vs D)
                chosen_unique_keys = chosen
                unchosen_unique_keys = unchosen
                comparison_type = "same_row_separation"
                shared_context = f"row_sep_{chosen_row_sep}"
                
            else:
                continue  # Skip if neither condition is met
            
            for chosen_unique_key in chosen_unique_keys:
                for unchosen_unique_key in unchosen_unique_keys:
                    instances.append({
                        'user_id': row['user_id'],
                        'chosen_bigram': chosen,
                        'unchosen_bigram': unchosen,
                        'chosen_key': chosen_unique_key,
                        'unchosen_key': unchosen_unique_key,
                        'shared_context': shared_context,
                        'comparison_type': comparison_type,
                        'chose_key': chosen_unique_key,
                        'slider_value': row.get('sliderValue', 0),
                        'chosen_key_finger': self.key_positions.get(chosen_unique_key, KeyPosition('', 0, 0, 0)).finger,
                        'unchosen_key_finger': self.key_positions.get(unchosen_unique_key, KeyPosition('', 0, 0, 0)).finger,
                        'chosen_key_row': self.key_positions.get(chosen_unique_key, KeyPosition('', 0, 0, 0)).row,
                        'unchosen_key_row': self.key_positions.get(unchosen_unique_key, KeyPosition('', 0, 0, 0)).row,
                        'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                        'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                    })
        
        return pd.DataFrame(instances)

    def _test_key_preference(self) -> Dict[str, Any]:
        """Test key preferences using raw pairwise comparisons with frequency weighting."""
        logger.info("Testing key preferences (raw pairwise comparisons with frequency weighting)...")
        
        instances_df = self._extract_key_preference_instances()
        
        if instances_df.empty:
            return {'error': 'No key preference instances found'}
        
        logger.info(f"Found {len(instances_df)} key preference instances from {instances_df['user_id'].nunique()} users")
        
        # Group by key pairs and analyze each pair separately
        instances_df['key_pair'] = instances_df.apply(
            lambda row: tuple(sorted([row['chosen_key'], row['unchosen_key']])), axis=1
        )
        
        key_pair_counts = instances_df['key_pair'].value_counts()
        pairwise_results = {}
        all_p_values = []
        
        # Analyze each key pair separately
        for key_pair, count in key_pair_counts.items():
            if count >= 10:  # Minimum instances for analysis
                key1, key2 = key_pair
                pair_data = instances_df[instances_df['key_pair'] == key_pair].copy()
                
                # Code preference for key1 vs key2 (binary outcome)
                pair_data['chose_key1'] = (pair_data['chose_key'] == key1).astype(int)
                
                # Run simple proportion test for this key pair
                simple_test = self._simple_proportion_test(pair_data, 'chose_key1')
                
                # Calculate raw preference rate for key1 over key2
                key1_preference_rate = simple_test['preference_rate']
                
                # Store results with clear interpretation
                pairwise_results[key_pair] = {
                    'key1': key1,
                    'key2': key2,
                    'key1_wins': int(pair_data['chose_key1'].sum()),
                    'key2_wins': int((1 - pair_data['chose_key1']).sum()),
                    'total_comparisons': count,
                    'key1_preference_rate': key1_preference_rate,
                    'key2_preference_rate': 1.0 - key1_preference_rate,
                    'p_value': simple_test['p_value'],
                    'significant': simple_test['significant'],
                    'effect_size': simple_test['effect_size'],
                    'winner': key1 if key1_preference_rate > 0.5 else key2,
                    'winner_rate': max(key1_preference_rate, 1.0 - key1_preference_rate),
                    'interpretation': f"{key1.upper()} vs {key2.upper()}: {key1.upper()} preferred {key1_preference_rate:.1%} of time"
                }
                
                all_p_values.append(simple_test['p_value'])
        
        # Calculate individual key scores (win-loss method)
        individual_scores, ranked_keys = self._calculate_win_loss_scores(pairwise_results)
        
        # Calculate frequency-weighted key scores
        weighted_key_scores = {}
        all_keys = set()
        
        # Get frequency weights for each key pair
        pair_frequencies = instances_df['key_pair'].value_counts()
        max_frequency = pair_frequencies.max()
        
        for key_pair, pair_results in pairwise_results.items():
            if 'error' not in pair_results:
                key1, key2 = key_pair
                all_keys.update([key1, key2])
                
                # Weight inversely proportional to frequency
                frequency = pair_frequencies[key_pair]
                weight = max_frequency / frequency  # Higher weight for rarer comparisons
                
                # Apply weight to preference rates
                key1_rate = pair_results['key1_preference_rate']
                weighted_key1_score = (key1_rate - 0.5) * weight  # Deviation from neutral, weighted
                weighted_key2_score = (0.5 - key1_rate) * weight  # Opposite for key2
                
                # Accumulate weighted scores
                if key1 not in weighted_key_scores:
                    weighted_key_scores[key1] = {'score': 0, 'weight_sum': 0}
                if key2 not in weighted_key_scores:
                    weighted_key_scores[key2] = {'score': 0, 'weight_sum': 0}
                
                weighted_key_scores[key1]['score'] += weighted_key1_score
                weighted_key_scores[key1]['weight_sum'] += weight
                weighted_key_scores[key2]['score'] += weighted_key2_score
                weighted_key_scores[key2]['weight_sum'] += weight
        
        # Calculate final frequency-weighted key rankings
        frequency_weighted_rankings = []
        for key in all_keys:
            if weighted_key_scores[key]['weight_sum'] > 0:
                final_score = weighted_key_scores[key]['score'] / weighted_key_scores[key]['weight_sum']
                frequency_weighted_rankings.append((key, final_score))
        
        frequency_weighted_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Create summary without Bradley-Terry aggregation
        significant_pairs = []
        for pair, results in pairwise_results.items():
            if results['significant']:
                winner = results['winner']
                loser = results['key2'] if winner == results['key1'] else results['key1']
                rate = results['winner_rate']
                significant_pairs.append(f"{winner.upper()} > {loser.upper()} ({rate:.1%})")
        
        # Export complete pairwise results for inspection
        pairwise_export = []
        for pair, results in pairwise_results.items():
            pairwise_export.append({
                'key_pair': f"{results['key1'].upper()}-{results['key2'].upper()}",
                'key1': results['key1'].upper(),
                'key2': results['key2'].upper(),
                'key1_wins': results['key1_wins'],
                'key2_wins': results['key2_wins'],
                'total_comparisons': results['total_comparisons'],
                'key1_preference_rate': results['key1_preference_rate'],
                'key2_preference_rate': results['key2_preference_rate'],
                'winner': results['winner'].upper(),
                'winner_preference_rate': results['winner_rate'],
                'p_value': results['p_value'],
                'significant': results['significant'],
                'effect_size': results['effect_size']
            })
        
        # Use Fisher's method to combine p-values for overall significance
        overall_test = None
        if all_p_values:
            try:
                overall_stat, overall_p = combine_pvalues(all_p_values, method='fisher')
                overall_test = {
                    'combined_p_value': overall_p,
                    'significant': overall_p < 0.05,
                    'method': 'fisher_combined',
                    'n_tests_combined': len(all_p_values)
                }
            except Exception as e:
                logger.warning(f"Failed to combine p-values: {e}")
                overall_test = {
                    'min_p_value': min(all_p_values),
                    'significant': min(all_p_values) < 0.05,
                    'method': 'minimum_p',
                    'n_tests_combined': len(all_p_values)
                }
        
        return {
            'description': 'Raw pairwise key preferences with frequency weighting',
            'method': 'raw_pairwise_comparisons',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique(),
            'n_key_pairs': len(pairwise_results),
            'n_significant_pairs': len(significant_pairs),
            'pairwise_results': pairwise_results,
            'pairwise_export': pairwise_export,
            'significant_preferences': significant_pairs,
            'individual_key_scores': individual_scores,
            'ranked_keys': ranked_keys,  # Win-loss rankings
            'frequency_weighted_rankings': frequency_weighted_rankings,  # Frequency-corrected rankings
            'frequency_bias_correction': {
                'min_frequency': pair_frequencies.min(),
                'max_frequency': pair_frequencies.max(),
                'weight_ratio': max_frequency / pair_frequencies.min(),
                'bias_magnitude': 'high' if max_frequency / pair_frequencies.min() > 5 else 'moderate'
            },
            'simple_test': overall_test,  # For validation framework
            'p_value': overall_test['combined_p_value'] if overall_test else 1.0,
            'preference_rate': len(significant_pairs) / len(pairwise_results) if pairwise_results else 0.0,
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_raw_pairwise_results(significant_pairs, len(pairwise_results))
        }

    def _interpret_raw_pairwise_results(self, significant_pairs: List[str], total_pairs: int) -> str:
        """Interpret raw pairwise key preference results."""
        n_significant = len(significant_pairs)
        
        if n_significant == 0:
            return f"No significant key preferences detected from {total_pairs} pairwise comparisons"
        
        # Show top significant preferences
        top_preferences = significant_pairs[:5]  # Show first 5
        preferences_text = "; ".join(top_preferences)
        
        return f"Significant preferences ({n_significant}/{total_pairs} pairs): {preferences_text}{'...' if n_significant > 5 else ''}"

    def _save_key_pairwise_results(self, key_pref_results: Dict[str, Any], output_folder: str) -> None:
        """Save complete pairwise key comparison results."""
        if 'pairwise_export' not in key_pref_results:
            return
        
        pairwise_df = pd.DataFrame(key_pref_results['pairwise_export'])
        
        # Sort by significance and effect size
        pairwise_df = pairwise_df.sort_values(['significant', 'effect_size'], ascending=[False, False])
        
        csv_path = os.path.join(output_folder, 'key_pairwise_comparisons.csv')
        pairwise_df.to_csv(csv_path, index=False)
        logger.info(f"Complete pairwise key comparisons saved to {csv_path}")
        
        # Create summary of significant preferences
        significant_df = pairwise_df[pairwise_df['significant'] == True]
        if len(significant_df) > 0:
            summary_path = os.path.join(output_folder, 'significant_key_preferences.csv')
            significant_df.to_csv(summary_path, index=False)
            logger.info(f"Significant key preferences saved to {summary_path}")
            
            # Print verification for R vs V
            rv_comparison = pairwise_df[
                ((pairwise_df['key1'] == 'R') & (pairwise_df['key2'] == 'V')) |
                ((pairwise_df['key1'] == 'V') & (pairwise_df['key2'] == 'R'))
            ]
            
            if len(rv_comparison) > 0:
                rv_row = rv_comparison.iloc[0]
                print(f"\nR vs V verification:")
                print(f"Winner: {rv_row['winner']} ({rv_row['winner_preference_rate']:.1%})")
                print(f"Raw rates: R={rv_row['key1_preference_rate']:.1%}, V={rv_row['key2_preference_rate']:.1%}")
           
    # =========================================================================
    # OBJECTIVE 2: ROW SEPARATION
    # =========================================================================
    
    def _extract_row_separation_instances(self) -> pd.DataFrame:
        """Extract specific pairwise row separation comparisons."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
                
            chosen_row_sep = self._calculate_row_separation(chosen)
            unchosen_row_sep = self._calculate_row_separation(unchosen)
            
            # Specific pairwise comparisons only
            comparison_type = None
            chose_smaller = None
            
            if {chosen_row_sep, unchosen_row_sep} == {0, 1}:
                comparison_type = "same_row_vs_1_apart"
                chose_smaller = 1 if min(chosen_row_sep, unchosen_row_sep) == chosen_row_sep else 0
            elif {chosen_row_sep, unchosen_row_sep} == {1, 2}:
                comparison_type = "1_apart_vs_2_apart"
                chose_smaller = 1 if min(chosen_row_sep, unchosen_row_sep) == chosen_row_sep else 0
            else:
                continue  # Skip other combinations
            
            instances.append({
                'user_id': row['user_id'],
                'chosen_bigram': chosen,
                'unchosen_bigram': unchosen,
                'chose_smaller_separation': chose_smaller,
                'chosen_row_separation': chosen_row_sep,
                'unchosen_row_separation': unchosen_row_sep,
                'comparison_type': comparison_type,
                'slider_value': row.get('sliderValue', 0),
                'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
            })
        
        return pd.DataFrame(instances)

    def _test_row_separation_preference(self) -> Dict[str, Any]:
        """Test specific pairwise row separation preferences."""
        logger.info("Testing pairwise row separation preferences...")
        
        instances_df = self._extract_row_separation_instances()
        
        if instances_df.empty:
            return {'error': 'No row separation instances found'}
        
        logger.info(f"Found {len(instances_df)} row separation instances from {instances_df['user_id'].nunique()} users")
        
        # Overall analysis
        simple_test = self._simple_proportion_test(instances_df, 'chose_smaller_separation')
        model_results = self._fit_instance_level_model(
            instances_df,
            'chose_smaller_separation', 
            'pairwise_row_separation'
        )
        
        # Analysis by specific comparison type
        comparison_results = {}
        for comp_type in instances_df['comparison_type'].unique():
            comp_data = instances_df[instances_df['comparison_type'] == comp_type]
            if len(comp_data) >= 10:
                comp_simple = self._simple_proportion_test(comp_data, 'chose_smaller_separation')
                comparison_results[comp_type] = {
                    'simple_test': comp_simple,
                    'n_instances': len(comp_data),
                    'preference_rate': comp_simple['preference_rate'],
                    'p_value': comp_simple['p_value'],
                    'interpretation': f"{comp_type.replace('_', ' ')}: {comp_simple['preference_rate']:.1%} prefer smaller separation"
                }

        # FREQUENCY WEIGHTING ANALYSIS
        # Create bigram pair identifier
        instances_df['bigram_pair'] = instances_df.apply(
            lambda row: tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']])), axis=1
        )
        
        # Calculate frequency weights
        pair_frequencies = instances_df['bigram_pair'].value_counts()
        max_frequency = pair_frequencies.max()
        instances_df['weight'] = instances_df['bigram_pair'].map(lambda x: max_frequency / pair_frequencies[x])
        
        # Calculate frequency-weighted preference rates
        weighted_analysis = {}
        
        # Overall weighted preference
        total_weight = instances_df['weight'].sum()
        weighted_overall = (instances_df['chose_smaller_separation'] * instances_df['weight']).sum() / total_weight
        
        # Weighted by comparison type
        weighted_by_type = {}
        for comp_type in instances_df['comparison_type'].unique():
            comp_data = instances_df[instances_df['comparison_type'] == comp_type]
            if len(comp_data) >= 20:
                comp_weight_sum = comp_data['weight'].sum()
                comp_weighted_rate = (comp_data['chose_smaller_separation'] * comp_data['weight']).sum() / comp_weight_sum
                
                unweighted_rate = comp_data['chose_smaller_separation'].mean()
                bias_magnitude = abs(comp_weighted_rate - unweighted_rate)
                
                weighted_by_type[comp_type] = {
                    'unweighted_rate': unweighted_rate,
                    'weighted_rate': comp_weighted_rate,
                    'bias_magnitude': bias_magnitude,
                    'interpretation': f"{comp_type.replace('_', ' ')}: {comp_weighted_rate:.1%} prefer smaller separation (frequency-corrected)"
                }
        
        weighted_analysis = {
            'overall_unweighted': simple_test['preference_rate'],
            'overall_weighted': weighted_overall,
            'overall_bias_magnitude': abs(weighted_overall - simple_test['preference_rate']),
            'weighted_by_type': weighted_by_type,
            'frequency_stats': {
                'min_frequency': pair_frequencies.min(),
                'max_frequency': pair_frequencies.max(),
                'weight_ratio': max_frequency / pair_frequencies.min(),
                'unique_comparisons': len(pair_frequencies),
                'bias_severity': 'high' if max_frequency / pair_frequencies.min() > 5 else 'moderate'
            }
        }
        
        # Log the bias detection
        logger.info(f"Row separation bias: Unweighted={simple_test['preference_rate']:.1%}, Weighted={weighted_overall:.1%}")
        if weighted_analysis['frequency_stats']['bias_severity'] == 'high':
            logger.warning(f"High frequency bias detected (ratio: {max_frequency / pair_frequencies.min():.1f})")
        
        return {
            'description': 'Pairwise row separation preferences with frequency correction',
            'method': 'pairwise_row_separation_analysis',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique(),
            'simple_test': simple_test,
            'model_results': model_results,
            'comparison_results': comparison_results,
            'weighted_analysis': weighted_analysis,  # ADD THIS
            'p_value': simple_test['p_value'],
            'preference_rate': simple_test['preference_rate'],
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_pairwise_row_results(simple_test, comparison_results, weighted_analysis)
        }

    def _interpret_pairwise_row_results(self, overall_test: Dict, comparison_results: Dict, weighted_analysis: Dict) -> str:
        """Interpret pairwise row separation results with frequency bias info."""
        unweighted_rate = overall_test['preference_rate']
        weighted_rate = weighted_analysis['overall_weighted']
        bias_magnitude = weighted_analysis['overall_bias_magnitude']
        
        interpretation = f"Unweighted: {unweighted_rate:.1%} prefer smaller row separation"
        
        if bias_magnitude > 0.05:  # 5% threshold for significant bias
            interpretation += f"; Frequency-corrected: {weighted_rate:.1%} (bias: {bias_magnitude:.1%})"
        else:
            interpretation += f"; Frequency-corrected: {weighted_rate:.1%} (minimal bias)"
        
        # Add comparison breakdowns
        comparison_summaries = []
        for comp_type, results in comparison_results.items():
            rate = results['preference_rate']
            clean_name = comp_type.replace('_', ' ')
            comparison_summaries.append(f"{clean_name}: {rate:.1%}")
        
        if comparison_summaries:
            interpretation += f". Specific comparisons: {'; '.join(comparison_summaries)}"
        
        return interpretation
            
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
    # OBJECTIVE 3: COLUMN SEPARATION
    # =========================================================================

    def _extract_column_instances(self) -> pd.DataFrame:
        """Extract specific pairwise column separation comparisons, controlling for row separation."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
                
            chosen_row_sep = self._calculate_row_separation(chosen)
            unchosen_row_sep = self._calculate_row_separation(unchosen)
            
            # Only compare bigrams with SAME row separation
            if chosen_row_sep != unchosen_row_sep:
                continue
            
            chosen_col_sep = self._calculate_column_separation(chosen)
            unchosen_col_sep = self._calculate_column_separation(unchosen)

            row_context = chosen_row_sep  # Same for both bigrams
            separations = {chosen_col_sep, unchosen_col_sep}
            comparison_type = None
            chose_adjacent = None
            
            # Test 1: Same column (0) vs Adjacent column (1) - same row context
            if separations == {0, 1}:
                comparison_type = "same_vs_adjacent_column"
                context = f"{row_context}_row_separation"
                chose_adjacent = 1 if chosen_col_sep == 1 else 0
            
            # Test 2: Adjacent (1) vs Remote (2-3) - same row context, different fingers only
            elif separations in [{1, 2}, {1, 3}] and chosen_col_sep > 0 and unchosen_col_sep > 0:
                comparison_type = "adjacent_vs_remote_column"
                context = f"{row_context}_row_separation"
                chose_adjacent = 1 if chosen_col_sep == 1 else 0
            
            else:
                continue
            
            if comparison_type and chose_adjacent is not None:
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'comparison_type': comparison_type,
                    'context': context,
                    'row_separation_context': row_context,
                    'chose_adjacent': chose_adjacent,
                    'chosen_col_separation': chosen_col_sep,
                    'unchosen_col_separation': unchosen_col_sep,
                    'chosen_row_separation': chosen_row_sep,
                    'unchosen_row_separation': unchosen_row_sep,
                    'slider_value': row.get('sliderValue', 0),
                    'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                    'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                })
        
        return pd.DataFrame(instances)

    def _test_column_separation(self) -> Dict[str, Any]:
        """Test systematic column separation preferences."""
        logger.info("Testing systematic column separation preferences...")
        
        instances_df = self._extract_column_instances()
        
        #print("\nDEBUG: Sample same vs adjacent comparisons:")
        #same_vs_adj = instances_df[instances_df['comparison_type'] == 'same_vs_adjacent_column'].head(10)
        #for _, row in same_vs_adj.iterrows():
        #    print(f"{row['chosen_bigram'].upper()} ({row['chosen_col_separation']}) vs {row['unchosen_bigram'].upper()} ({row['unchosen_col_separation']}) → chose_adjacent={row['chose_adjacent']}")

        if instances_df.empty:
            return {'error': 'No combined column separation instances found'}
        
        logger.info(f"Found {len(instances_df)} combined column instances from {instances_df['user_id'].nunique()} users")
        
        # Overall analysis
        simple_test = self._simple_proportion_test(instances_df, 'chose_adjacent')
        model_results = self._fit_instance_level_model(
            instances_df,
            'chose_adjacent', 
            'column_separation'
        )

        # Frequency-weighted analysis (frequency of each unique comparison)
        weighted_analysis = self._calculate_weighted_preferences(instances_df, 'chose_adjacent')
        
        # Log the difference
        unweighted_rate = simple_test['preference_rate']
        weighted_rate = weighted_analysis['weighted_preference_rate']
        logger.info(f"Preference rates: Unweighted={unweighted_rate:.1%}, Weighted={weighted_rate:.1%}")


        # Analysis by comparison type
        comparison_results = {}
        for comp_type in instances_df['comparison_type'].unique():
            comp_data = instances_df[instances_df['comparison_type'] == comp_type]
            if len(comp_data) >= 10:
                comp_simple = self._simple_proportion_test(comp_data, 'chose_adjacent')
                
                # Create interpretation based on comparison type
                if comp_type == "same_vs_adjacent_column":
                    interpretation = f"Same finger (same column) vs Different finger (adjacent column): {(comp_simple['preference_rate']):.1%} prefer adjacent"
                elif comp_type == "adjacent_vs_remote_column":
                    interpretation = f"Adjacent vs Remote columns: {comp_simple['preference_rate']:.1%} prefer adjacent"
                
                print(interpretation)
                comparison_results[comp_type] = {
                    'simple_test': comp_simple,
                    'n_instances': len(comp_data),
                    'preference_rate': comp_simple['preference_rate'],
                    'p_value': comp_simple['p_value'],
                    'interpretation': interpretation,
                    'examples': self._get_comparison_examples(comp_data, comp_type)
                }
        
        # Context analysis for same vs adjacent (controlling for row separation)
        context_results = {}
        if 'same_vs_adjacent_column' in comparison_results:
            same_vs_adj_data = instances_df[instances_df['comparison_type'] == 'same_vs_adjacent_column']
            
            for context in same_vs_adj_data['context'].unique():
                context_data = same_vs_adj_data[same_vs_adj_data['context'] == context]
                if len(context_data) >= 10:
                    context_simple = self._simple_proportion_test(context_data, 'chose_adjacent')
                    
                    # Parse context
                    if context.endswith('_row_separation'):
                        row_sep = context.replace('_row_separation', '')
                        if row_sep == '0':
                            row_context_desc = "Same row (0 row separation)"
                        elif row_sep == '1':
                            row_context_desc = "1-row separation"
                        elif row_sep == '2':
                            row_context_desc = "2-row separation"
                        else:
                            row_context_desc = f"{row_sep}-row separation"
                    else:
                        row_context_desc = context

                    context_results[context] = {
                        'simple_test': context_simple,
                        'n_instances': len(context_data),
                        'preference_rate': context_simple['preference_rate'],
                        'same_finger_rate': 1.0 - context_simple['preference_rate'],
                        'description': row_context_desc,
                        'interpretation': f"{row_context_desc}: {(context_simple['preference_rate']):.1%} prefer adjacent"
                    }
        
        return {
            'description': 'Systematic column separation preferences',
            'method': 'hierarchical_column_analysis',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique(),
            'simple_test': simple_test,
            'model_results': model_results,
            'comparison_results': comparison_results,
            'context_results': context_results,
            'p_value': simple_test['p_value'],
            'preference_rate': simple_test['preference_rate'],
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_column_results_with_bias(simple_test, weighted_analysis),
            'weighted_analysis': weighted_analysis,
            'frequency_bias_detected': abs(weighted_rate - unweighted_rate) > 0.05,
        }

    def _interpret_column_results_with_bias(self, simple_test: Dict, weighted_analysis: Dict) -> str:
        """Interpret column results showing both biased and corrected rates."""
        unweighted = simple_test['preference_rate']
        weighted = weighted_analysis['weighted_preference_rate']
        bias_magnitude = abs(weighted - unweighted)
        
        if bias_magnitude > 0.05:  # 5% threshold
            return f"Unweighted: {unweighted:.1%} prefer adjacent columns; Frequency-corrected: {weighted:.1%} (bias: {bias_magnitude:.1%})"
        else:
            return f"Overall: {unweighted:.1%} prefer adjacent columns (frequency bias minimal: {bias_magnitude:.1%})"

    def _get_comparison_examples(self, comp_data: pd.DataFrame, comp_type: str) -> List[str]:
        """Get example comparisons for each type."""
        examples = []
        sample_rows = comp_data.head(3)
        
        for _, row in sample_rows.iterrows():
            chosen = row['chosen_bigram'].upper()
            unchosen = row['unchosen_bigram'].upper()
            chosen_col_sep = row['chosen_col_separation']
            unchosen_col_sep = row['unchosen_col_separation']
            
            if comp_type == "same_vs_adjacent_column":
                # Determine which bigram is same column (0) vs adjacent column (1)
                if chosen_col_sep == 0:
                    same_col_bigram = chosen
                    adj_col_bigram = unchosen
                    examples.append(f"Chose {same_col_bigram} (same finger) over {adj_col_bigram} (different finger)")
                else:  # chosen_col_sep == 1
                    same_col_bigram = unchosen
                    adj_col_bigram = chosen
                    examples.append(f"Chose {adj_col_bigram} (different finger) over {same_col_bigram} (same finger)")
            
            else:  # adjacent vs remote cases
                # Determine which bigram is adjacent (1) vs remote (2-3)
                if chosen_col_sep == 1:
                    adj_bigram = chosen
                    remote_bigram = unchosen
                    examples.append(f"Chose {adj_bigram} (adjacent columns) over {remote_bigram} (remote columns)")
                else:  # chosen_col_sep > 1
                    adj_bigram = unchosen
                    remote_bigram = chosen
                    examples.append(f"Chose {remote_bigram} (remote columns) over {adj_bigram} (adjacent columns)")
        
        return examples
    
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
    # LATERAL METHODS
    # =========================================================================
        
    def _test_column_4_vs_5_preference(self) -> Dict[str, Any]:
        """Test preference for column 4 (RFV) vs column 5 (TGB) with controlled vs general approach."""
        logger.info("Testing column 4 vs 5 preference...")
        
        # Try controlled approach first (bigrams sharing one letter)
        controlled_comparisons = self._extract_controlled_column_4_5_comparisons()
        
        min_comparisons_threshold = max(10, self.config.get('min_comparisons', 10))
        
        if len(controlled_comparisons) >= min_comparisons_threshold:
            logger.info(f"Using {len(controlled_comparisons)} controlled column 4 vs 5 comparisons")
            approach = "controlled"
            test_comparisons = controlled_comparisons
        else:
            logger.warning(f"Insufficient controlled comparisons ({len(controlled_comparisons)}), "
                          f"falling back to general approach")
            general_comparisons = self._extract_general_column_4_5_comparisons()
            approach = "general"
            test_comparisons = general_comparisons
        
        if not test_comparisons:
            return {'error': 'No adequate column 4 vs 5 comparisons found'}
        
        # Fit regression model with frequency controls
        model_results = self._fit_frequency_controlled_model(
            test_comparisons,
            'column_4_vs_5_preference',
            self._calculate_column_4_vs_5_indicator
        )
        
        return {
            'description': 'Column 4 (RFV) vs Column 5 (TGB) preference',
            'approach_used': approach,
            'n_comparisons': len(test_comparisons),
            'frequency_controlled_coefficient': model_results.get('coefficient', 'N/A'),
            'p_value': model_results.get('p_value', 'N/A'),
            'confidence_interval': model_results.get('confidence_interval', [None, None]),
            'r_squared': model_results.get('r_squared', 'N/A'),
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_column_4_vs_5_result(model_results),
            'model_data': model_results.get('model_data', pd.DataFrame())
        }
    
    def _extract_controlled_column_4_5_comparisons(self) -> List[Tuple[str, str, Dict]]:
        """Extract controlled comparisons between column 4 and 5 bigrams sharing one letter."""
        comparisons = []
        all_bigrams = self._get_all_left_hand_bigrams()
        
        # Define column 4 and 5 keys
        col4_keys = {'r', 'f', 'v'}
        col5_keys = {'t', 'g', 'b'}
        
        # For each left-hand letter, find controlled comparisons
        for shared_letter in self.left_hand_keys:
            # Find all bigrams containing this letter
            bigrams_with_letter = [bg for bg in all_bigrams if shared_letter in bg]
            
            # Separate into column 4 and column 5 groups
            col4_bigrams = []
            col5_bigrams = []
            
            for bigram in bigrams_with_letter:
                if any(key in col4_keys for key in bigram):
                    if not any(key in col5_keys for key in bigram):  # Pure column 4
                        col4_bigrams.append(bigram)
                elif any(key in col5_keys for key in bigram):
                    if not any(key in col4_keys for key in bigram):  # Pure column 5
                        col5_bigrams.append(bigram)
            
            # Compare column 4 vs column 5 bigrams sharing this letter
            for col4_bigram in col4_bigrams:
                for col5_bigram in col5_bigrams:
                    if self._bigrams_share_exactly_one_letter(col4_bigram, col5_bigram, shared_letter):
                        comparison_data = self._extract_pairwise_comparison(col4_bigram, col5_bigram)
                        
                        if comparison_data['total'] >= self.config.get('min_comparisons', 10):
                            comparisons.append((col4_bigram, col5_bigram, comparison_data))
        
        logger.info(f"Found {len(comparisons)} controlled column 4 vs 5 comparisons")
        return comparisons
    
    def _extract_general_column_4_5_comparisons(self) -> List[Tuple[str, str, Dict]]:
        """Extract general comparisons between column 4 and 5 bigrams."""
        comparisons = []
        all_bigrams = self._get_all_left_hand_bigrams()
        
        col4_keys = {'r', 'f', 'v'}
        col5_keys = {'t', 'g', 'b'}
        
        # Separate bigrams by column usage
        col4_bigrams = []
        col5_bigrams = []
        
        for bigram in all_bigrams:
            if any(key in col4_keys for key in bigram):
                if not any(key in col5_keys for key in bigram):  # Pure column 4
                    col4_bigrams.append(bigram)
            elif any(key in col5_keys for key in bigram):
                if not any(key in col4_keys for key in bigram):  # Pure column 5
                    col5_bigrams.append(bigram)
        
        logger.info(f"Found {len(col4_bigrams)} column 4 and {len(col5_bigrams)} column 5 bigrams")
        
        # Compare column 4 vs column 5 bigrams
        for col4_bigram in col4_bigrams:
            for col5_bigram in col5_bigrams:
                comparison_data = self._extract_pairwise_comparison(col4_bigram, col5_bigram)
                
                if comparison_data['total'] >= self.config.get('min_comparisons', 10):
                    comparisons.append((col4_bigram, col5_bigram, comparison_data))
        
        logger.info(f"Found {len(comparisons)} general column 4 vs 5 comparisons")
        return comparisons
    
    def _calculate_column_4_vs_5_indicator(self, bigram1: str, bigram2: str) -> float:
        """Calculate column preference indicator (1 = column 4 preferred, 0 = column 5 preferred)."""
        col4_keys = {'r', 'f', 'v'}
        col5_keys = {'t', 'g', 'b'}
        
        bg1_has_col4 = any(key in col4_keys for key in bigram1)
        bg1_has_col5 = any(key in col5_keys for key in bigram1)
        bg2_has_col4 = any(key in col4_keys for key in bigram2)
        bg2_has_col5 = any(key in col5_keys for key in bigram2)
        
        # Return 1 if bigram1 is column 4 and bigram2 is column 5
        if bg1_has_col4 and not bg1_has_col5 and bg2_has_col5 and not bg2_has_col4:
            return 1.0
        # Return 0 if bigram1 is column 5 and bigram2 is column 4
        elif bg1_has_col5 and not bg1_has_col4 and bg2_has_col4 and not bg2_has_col5:
            return 0.0
        else:
            return 0.5  # Mixed or unclear
    
    def _interpret_column_4_vs_5_result(self, model_results: Dict) -> str:
        """Interpret column 4 vs 5 preference results."""
        if 'error' in model_results:
            return "Analysis failed - insufficient data"
        
        coeff = model_results.get('coefficient', 0)
        p_value = model_results.get('p_value', 1.0)
        
        if p_value < 0.05:
            if coeff > 0:
                strength = "strong" if abs(coeff) > 0.1 else "moderate" if abs(coeff) > 0.05 else "weak"
                return f"Significant preference for column 4 (RFV) over column 5 (TGB) (coeff={coeff:.3f}, {strength} effect)"
            else:
                strength = "strong" if abs(coeff) > 0.1 else "moderate" if abs(coeff) > 0.05 else "weak"
                return f"Significant preference for column 5 (TGB) over column 4 (RFV) (coeff={coeff:.3f}, {strength} effect - column 5 not avoided)"
        else:
            return f"No significant column preference detected (p={p_value:.3f})"

    # =========================================================================
    # HELPER METHODS FOR STATISTICAL ANALYSIS
    # =========================================================================
    
    def _fit_instance_level_model(self, instances_df: pd.DataFrame, outcome_var: str, 
                                model_name: str) -> Dict[str, Any]:
        """Fit regression model on instance-level data with robust handling of perfect separation."""
        
        if len(instances_df) < 20:
            return {'error': f'Insufficient instances for {model_name} (need ≥20, got {len(instances_df)})'}
        
        try:
            # Calculate basic preference rate (robust to perfect separation)
            preference_rate = instances_df[outcome_var].mean()
            n_users = instances_df['user_id'].nunique()
            
            # Check for perfect separation at user level
            user_means = instances_df.groupby('user_id')[outcome_var].mean()
            perfect_users = user_means[(user_means == 0) | (user_means == 1)]
            
            if len(perfect_users) > 0:
                logger.info(f"Found {len(perfect_users)} users with perfect preferences - strong signal detected")
            
            # Prepare data for modeling
            control_cols = [col for col in instances_df.columns if col.startswith('log_')]
            
            if len(control_cols) == 0:
                # No controls available - use simple proportion test
                return {
                    'method': 'simple_proportion',
                    'preference_rate': preference_rate,
                    'n_instances': len(instances_df),
                    'n_users': n_users,
                    'n_perfect_users': len(perfect_users),
                    'instances_per_user': len(instances_df) / n_users,
                    'interpretation': self._interpret_preference_rate(preference_rate, outcome_var),
                    'robust': True
                }
            
            X = instances_df[control_cols].fillna(instances_df[control_cols].median())
            X = sm.add_constant(X)
            y = instances_df[outcome_var]
            
            # Try regularized logistic regression first
            try:
                model = sm.Logit(y, X).fit_regularized(disp=0, alpha=0.01, maxiter=1000)
                
                return {
                    'method': 'regularized_logistic',
                    'model': model,
                    'preference_rate': preference_rate,
                    'n_instances': len(instances_df),
                    'n_users': n_users,
                    'n_perfect_users': len(perfect_users),
                    'instances_per_user': len(instances_df) / n_users,
                    'pseudo_r_squared': getattr(model, 'prsquared', None),
                    'interpretation': self._interpret_preference_rate(preference_rate, outcome_var),
                    'robust': True
                }
                
            except Exception as reg_error:
                logger.warning(f"Regularized regression failed for {model_name}: {reg_error}")
                
                # Fall back to simple proportion analysis
                return {
                    'method': 'simple_proportion_fallback',
                    'preference_rate': preference_rate,
                    'n_instances': len(instances_df),
                    'n_users': n_users,
                    'n_perfect_users': len(perfect_users),
                    'instances_per_user': len(instances_df) / n_users,
                    'interpretation': self._interpret_preference_rate(preference_rate, outcome_var),
                    'robust': True,
                    'fallback_reason': str(reg_error)
                }
            
        except Exception as e:
            logger.warning(f"Model fitting failed for {model_name}: {e}")
            return {'error': f'Model fitting failed: {str(e)}', 'n_instances': len(instances_df)}
        
    def _interpret_preference_rate(self, rate: float, outcome_var: str) -> str:
        """Interpret the preference rate for different outcomes."""
        interpretations = {
            'chose_different_finger': {
                'variable': 'different-finger preference',
                'good_direction': rate > 0.5,
                'threshold': 0.5
            },
            'chose_smaller_separation': {
                'variable': 'smaller row separation preference',
                'good_direction': rate > 0.5,
                'threshold': 0.5
            }
        }
        
        if outcome_var not in interpretations:
            return f"Preference rate: {rate:.1%}"
        
        info = interpretations[outcome_var]
        direction = "supports" if info['good_direction'] else "contradicts"
        strength = "strong" if abs(rate - 0.5) > 0.15 else "moderate" if abs(rate - 0.5) > 0.08 else "weak"
        
        return f"{info['variable']}: {rate:.1%} preference rate {direction} expectations ({strength} effect)"

    def _fit_frequency_controlled_model(self, comparisons: List, model_name: str, 
                                       predictor_function) -> Dict[str, Any]:
        """Enhanced regression model with better error handling and diagnostics."""
        
        model_data = []
        
        for comparison in comparisons:
            if len(comparison) == 3:
                bigram1, bigram2, comparison_data = comparison
            else:
                bigram1, bigram2 = comparison[:2]
                comparison_data = self._extract_pairwise_comparison(bigram1, bigram2)
            
            if comparison_data['total'] < self.config.get('min_comparisons', 20):
                continue
            
            # Dependent variable: proportion preferring bigram1
            preference = comparison_data['wins_item1'] / comparison_data['total']
            
            # Predictor of interest
            predictor_value = predictor_function(bigram1, bigram2)
            
            # Frequency control covariates
            controls = self._calculate_frequency_controls(bigram1, bigram2)
            
            model_data.append({
                'preference': preference,
                'predictor': predictor_value,
                'n_comparisons': comparison_data['total'],
                'bigram1': bigram1,
                'bigram2': bigram2,
                **controls
            })
        
        if len(model_data) < 5:
            return {'error': f'Insufficient data for {model_name} (need ≥5, got {len(model_data)})'}
        
        # Fit regression model with enhanced diagnostics
        df = pd.DataFrame(model_data)
        
        try:
            # Prepare data with better handling of missing values
            control_cols = [col for col in df.columns if col.startswith('log_')]
            X = df[['predictor'] + control_cols].fillna(df[['predictor'] + control_cols].median())
            y = df['preference']
            weights = np.sqrt(df['n_comparisons'])  # Use sqrt of sample size for weights
            
            # Add constant
            X = sm.add_constant(X)
            
            # Fit weighted regression
            model = sm.WLS(y, X, weights=weights).fit()
            
            # Enhanced diagnostics
            diagnostics = self._calculate_model_diagnostics(model, X, y, weights)
            
            return {
                'coefficient': model.params.get('predictor', np.nan),
                'p_value': model.pvalues.get('predictor', 1.0),
                'confidence_interval': model.conf_int().loc['predictor'].tolist() if 'predictor' in model.conf_int().index else [np.nan, np.nan],
                'r_squared': model.rsquared,
                'n_observations': len(model_data),
                'summary': str(model.summary()),
                'model_data': df,
                'diagnostics': diagnostics,
                'control_variables': control_cols
            }
            
        except Exception as e:
            logger.warning(f"Model fitting failed for {model_name}: {e}")
            return {'error': f'Model fitting failed: {str(e)}', 'n_observations': len(model_data)}
    
    def _calculate_model_diagnostics(self, model, X, y, weights) -> Dict[str, Any]:
        """Calculate enhanced model diagnostics."""
        try:
            diagnostics = {
                'aic': model.aic,
                'bic': model.bic,
                'condition_number': np.linalg.cond(X.values),
                'residual_std': np.sqrt(model.mse_resid),
                'durbin_watson': sm.stats.stattools.durbin_watson(model.resid)
            }
            
            # Check for outliers
            residuals = model.resid
            standardized_residuals = residuals / np.sqrt(model.mse_resid)
            diagnostics['n_outliers'] = np.sum(np.abs(standardized_residuals) > 2.5)
            diagnostics['max_abs_residual'] = np.max(np.abs(standardized_residuals))
            
            return diagnostics
            
        except Exception as e:
            logger.warning(f"Diagnostic calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_frequency_controls(self, bigram1: str, bigram2: str) -> Dict[str, float]:
        """Calculate frequency control variables for both bigrams."""
        
        def safe_log(freq):
            return np.log(freq + 1e-6)  # Add small constant to avoid log(0)
        
        return {
            'log_bigram1_english_freq': safe_log(self.english_bigram_frequencies.get(bigram1, 1e-5)),
            'log_bigram2_english_freq': safe_log(self.english_bigram_frequencies.get(bigram2, 1e-5)),
            'log_letter1_freq': safe_log(self.english_letter_frequencies.get(bigram1[0], 1e-5)),
            'log_letter2_freq': safe_log(self.english_letter_frequencies.get(bigram1[1], 1e-5)),
            'log_letter3_freq': safe_log(self.english_letter_frequencies.get(bigram2[0], 1e-5)),
            'log_letter4_freq': safe_log(self.english_letter_frequencies.get(bigram2[1], 1e-5))
        }
        
    def _extract_pairwise_comparison(self, bigram1: str, bigram2: str) -> Dict[str, int]:
        """Extract pairwise comparison data for two bigrams."""
        wins_bigram1 = 0
        total = 0
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if chosen == bigram1 and unchosen == bigram2:
                wins_bigram1 += 1
                total += 1
            elif chosen == bigram2 and unchosen == bigram1:
                total += 1
        
        return {'wins_item1': wins_bigram1, 'total': total}
    
    def _all_keys_in_left_hand(self, bigram: str) -> bool:
        """Check if all keys in bigram are left-hand keys."""
        return all(key in self.left_hand_keys for key in bigram)

    def _simple_proportion_test(self, instances_df: pd.DataFrame, outcome_var: str) -> Dict[str, Any]:
        """Perform simple proportion test without frequency controls."""
        n_instances = len(instances_df)
        n_preferred = instances_df[outcome_var].sum()
        preference_rate = n_preferred / n_instances
        
        # Calculate z-test for proportion
        expected = 0.5  # Null hypothesis: no preference
        se = np.sqrt(expected * (1 - expected) / n_instances)
        z_score = (preference_rate - expected) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        # Effect size (Cohen's h for proportions)
        effect_size = 2 * (np.arcsin(np.sqrt(preference_rate)) - np.arcsin(np.sqrt(0.5)))
        
        return {
            'n_instances': n_instances,
            'n_preferred': n_preferred,
            'preference_rate': preference_rate,
            'z_score': z_score,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'interpretation': self._interpret_proportion_test(preference_rate, p_value, n_instances)
        }

    def _interpret_proportion_test(self, rate: float, p_value: float, n: int) -> str:
        """Interpret proportion test results."""
        if p_value >= 0.05:
            return f"No significant preference detected (rate={rate:.1%}, p={p_value:.3f})"
        
        direction = "strong preference" if rate > 0.5 else "strong avoidance"
        strength = "very strong" if abs(rate - 0.5) > 0.15 else "strong" if abs(rate - 0.5) > 0.1 else "moderate"
        
        return f"Highly significant {direction} (rate={rate:.1%}, p={p_value:.2e}, {strength} effect, n={n})"

    # =========================================================================
    # VALIDATION FRAMEWORK
    # =========================================================================
    
    def _validate_all_objectives(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive validation on all extracted objectives."""
        validation_report = {
            'multiple_comparisons_correction': self._correct_multiple_comparisons(results),
            'effect_size_validation': self._validate_effect_sizes(results),
            'cross_validation': self._cross_validate_models(results),
            'confound_analysis': self._analyze_confounds(results),
            'statistical_power': self._assess_statistical_power(results),
            'overall_validity': self._assess_overall_validity(results)
        }
        
        return validation_report
    
    def _correct_multiple_comparisons(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple comparisons correction across ALL individual tests."""
        p_values = []
        test_names = []
        
        for obj_name, obj_results in results.items():
            if 'error' not in obj_results:
                
                # For most objectives: extract the single main p-value
                if obj_name != 'key_preference':
                    p_val = None
                    
                    # Check simple_test first (most reliable)
                    if 'simple_test' in obj_results and isinstance(obj_results['simple_test'], dict):
                        p_val = obj_results['simple_test'].get('p_value')
                    elif 'p_value' in obj_results:
                        p_val = obj_results['p_value']
                    
                    if p_val is not None and isinstance(p_val, (int, float)) and not pd.isna(p_val):
                        p_values.append(p_val)
                        test_names.append(f"{obj_name}_main")
                
                # For key_preference: extract ALL pairwise p-values
                else:
                    if 'key_pair_results' in obj_results:
                        pairwise_results = obj_results['key_pair_results']
                        for key_pair, pair_results in pairwise_results.items():
                            if isinstance(pair_results, dict) and 'p_value' in pair_results:
                                p_val = pair_results['p_value']
                                if isinstance(p_val, (int, float)) and not pd.isna(p_val):
                                    p_values.append(p_val)
                                    test_names.append(f"key_preference_{key_pair[0]}_vs_{key_pair[1]}")
        
        if not p_values:
            return {'error': 'No valid p-values found for correction'}
        
        # Apply corrections
        corrections = {}
        
        try:
            # Bonferroni correction
            bonferroni_corrected = multipletests(p_values, method='bonferroni')[1]
            corrections['bonferroni'] = dict(zip(test_names, bonferroni_corrected))
            
            # FDR correction  
            fdr_corrected = multipletests(p_values, method='fdr_bh')[1]
            corrections['fdr_bh'] = dict(zip(test_names, fdr_corrected))
            
            # Holm correction
            holm_corrected = multipletests(p_values, method='holm')[1]
            corrections['holm'] = dict(zip(test_names, holm_corrected))
            
        except Exception as e:
            return {'error': f'Correction failed: {e}', 'p_values_found': len(p_values)}
        
        # Count significant results
        original_significant = sum(1 for p in p_values if p < 0.05)
        bonferroni_significant = sum(1 for p in bonferroni_corrected if p < 0.05)
        fdr_significant = sum(1 for p in fdr_corrected if p < 0.05)
        
        # Count by objective for detailed reporting
        objective_breakdown = {}
        for test_name, orig_p, fdr_p in zip(test_names, p_values, fdr_corrected):
            obj_name = test_name.split('_')[0] if '_' in test_name else test_name
            if obj_name not in objective_breakdown:
                objective_breakdown[obj_name] = {
                    'total_tests': 0,
                    'original_significant': 0,
                    'fdr_significant': 0
                }
            
            objective_breakdown[obj_name]['total_tests'] += 1
            if orig_p < 0.05:
                objective_breakdown[obj_name]['original_significant'] += 1
            if fdr_p < 0.05:
                objective_breakdown[obj_name]['fdr_significant'] += 1
        
        return {
            'total_tests': len(p_values),
            'original_significant': original_significant,
            'bonferroni_significant': bonferroni_significant,
            'fdr_significant': fdr_significant,
            'fdr_bh_significant': fdr_significant,  # Alias for compatibility
            'corrected_p_values': corrections,
            'original_p_values': dict(zip(test_names, p_values)),
            'objective_breakdown': objective_breakdown,
            'recommendation': self._recommend_correction_method(
                original_significant, bonferroni_significant, fdr_significant, len(p_values)
            )
        }

    def _recommend_correction_method(self, orig: int, bonf: int, fdr: int, total: int) -> str:
        """Recommend appropriate multiple comparisons correction method."""
        if total < 10:
            return "With few tests, Bonferroni correction is appropriate"
        elif orig > total * 0.5:  # More than half significant
            return "High significance rate suggests FDR correction to control false discovery"
        elif bonf == 0:
            return "No significant results after Bonferroni - consider exploratory FDR approach"
        else:
            return "Use FDR correction for good balance of power and error control"
    
    def _validate_effect_sizes(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that effect sizes are meaningful and not just statistically significant."""
        effect_size_analysis = {}
        
        for obj_name, obj_results in results.items():
            if 'error' not in obj_results:
                effect_sizes = self._extract_effect_sizes(obj_results)
                
                if effect_sizes:
                    effect_size_analysis[obj_name] = {
                        'coefficients': effect_sizes,
                        'mean_effect': np.mean([abs(es) for es in effect_sizes]),
                        'max_effect': max([abs(es) for es in effect_sizes]),
                        'interpretation': self._interpret_effect_sizes(effect_sizes),
                        'practical_significance': self._assess_practical_significance(effect_sizes, obj_name)
                    }
        
        return effect_size_analysis
    
    def _extract_effect_sizes(self, obj_results: Dict[str, Any]) -> List[float]:
        """Extract effect sizes with robust error handling."""
        effect_sizes = []
        
        # For instance-level analysis, effect size is deviation from neutral (0.5)
        if 'simple_test' in obj_results and isinstance(obj_results['simple_test'], dict):
            pref_rate = obj_results['simple_test'].get('preference_rate')
            if isinstance(pref_rate, (int, float)) and not pd.isna(pref_rate):
                effect_sizes.append(abs(pref_rate - 0.5))
        
        # Fallback to model coefficient if available
        elif 'model_results' in obj_results and isinstance(obj_results['model_results'], dict):
            coeff = obj_results['model_results'].get('coefficient')
            if isinstance(coeff, (int, float)) and not pd.isna(coeff):
                effect_sizes.append(abs(coeff))
        
        return effect_sizes
    
    def _interpret_effect_sizes(self, effect_sizes: List[float]) -> str:
        """Interpret effect sizes using Cohen's conventions (adapted for proportions)."""
        mean_abs_effect = np.mean([abs(es) for es in effect_sizes])
        
        if mean_abs_effect < 0.02:
            return "negligible effect"
        elif mean_abs_effect < 0.05:
            return "small effect"
        elif mean_abs_effect < 0.10:
            return "medium effect"
        else:
            return "large effect"
    
    def _assess_practical_significance(self, effect_sizes: List[float], obj_name: str) -> str:
        """Assess whether effect sizes are practically meaningful for keyboard optimization."""
        mean_abs_effect = np.mean([abs(es) for es in effect_sizes])
        
        # Context-dependent thresholds
        thresholds = {
            'key_preference': 0.05,    
            'row_separation': 0.05,   
            'column_separation': 0.05  
        }
        
        threshold = thresholds.get(obj_name, 0.03)
        
        if mean_abs_effect >= threshold:
            return f"Practically significant (≥{threshold:.1%} threshold)"
        else:
            return f"Below practical significance threshold ({threshold:.1%})"
    
    def _cross_validate_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate the statistical models to assess generalizability."""
        cv_results = {}
        
        for obj_name, obj_results in results.items():
            if 'error' not in obj_results and 'model_data' in obj_results:
                cv_score = self._perform_cross_validation(obj_results['model_data'])
                cv_results[obj_name] = cv_score
        
        return cv_results
    
    def _perform_cross_validation(self, model_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform k-fold cross-validation on model data."""
        if len(model_data) < 20:
            return {'error': 'Insufficient data for cross-validation'}
        
        try:
            # Prepare data
            control_cols = [col for col in model_data.columns if col.startswith('log_')]
            X = model_data[['predictor'] + control_cols].fillna(0)
            y = model_data['preference']
            
            # Perform 5-fold cross-validation
            cv = KFold(n_splits=min(5, len(model_data) // 4), shuffle=True, random_state=42)
            
            # Use simple linear regression for CV (easier to interpret)
            model = LinearRegression()
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            return {
                'mean_r2': np.mean(cv_scores),
                'std_r2': np.std(cv_scores),
                'min_r2': np.min(cv_scores),
                'max_r2': np.max(cv_scores),
                'stability': 'high' if np.std(cv_scores) < 0.1 else 'medium' if np.std(cv_scores) < 0.2 else 'low',
                'n_folds': len(cv_scores)
            }
            
        except Exception as e:
            return {'error': f'Cross-validation failed: {str(e)}'}
        
    def _analyze_confounds(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential confounding variables and their control."""
        confound_analysis = {}
        
        for obj_name, obj_results in results.items():
            if 'error' not in obj_results and 'model_data' in obj_results:
                confounds = self._assess_confound_control(obj_results['model_data'])
                confound_analysis[obj_name] = confounds
        
        return confound_analysis
    
    def _assess_confound_control(self, model_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess how well frequency confounds are controlled."""
        control_cols = [col for col in model_data.columns if col.startswith('log_')]
        
        if not control_cols:
            return {'error': 'No frequency controls found'}
        
        # Check correlation between predictor and controls
        predictor_control_corrs = {}
        
        for control in control_cols:
            if control in model_data.columns:
                corr = model_data['predictor'].corr(model_data[control])
                predictor_control_corrs[control] = corr
        
        max_corr = max([abs(corr) for corr in predictor_control_corrs.values()] + [0])
        
        return {
            'predictor_control_correlations': predictor_control_corrs,
            'max_correlation_with_controls': max_corr,
            'confound_risk': 'high' if max_corr > 0.7 else 'medium' if max_corr > 0.4 else 'low',
            'control_adequacy': 'good' if max_corr < 0.4 else 'moderate' if max_corr < 0.7 else 'poor'
        }
    
    def _assess_statistical_power(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess statistical power of the tests performed."""
        power_analysis = {}
        
        for obj_name, obj_results in results.items():
            if 'error' not in obj_results:
                power_assessment = self._calculate_power_metrics(obj_results)
                power_analysis[obj_name] = power_assessment
        
        return power_analysis
    
    def _calculate_power_metrics(self, obj_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate power-related metrics for an objective."""
        n_obs = obj_results.get('n_observations', 0)
        n_comparisons = obj_results.get('n_comparisons', 0)
        
        if n_obs == 0 and n_comparisons == 0:
            return {'error': 'No sample size information available'}
        
        sample_size = max(n_obs, n_comparisons)
        
        # Simple power assessment based on sample size
        if sample_size < 30:
            power_level = 'low'
            power_comment = 'Sample size may be insufficient for reliable detection of small effects'
        elif sample_size < 100:
            power_level = 'medium'
            power_comment = 'Adequate power for medium to large effects'
        else:
            power_level = 'high'
            power_comment = 'Good power for detecting small to medium effects'
        
        return {
            'sample_size': sample_size,
            'power_level': power_level,
            'power_comment': power_comment,
            'recommended_minimum': 50  # For psychology/HCI research
        }
    
    def _assess_overall_validity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide overall validity assessment with fixed significance detection."""
        successful_objectives = sum(1 for r in results.values() if 'error' not in r)
        total_objectives = len(results)
        
        # Count objectives with significant results (fixed logic)
        significant_objectives = 0
        for obj_results in results.values():
            if 'error' not in obj_results:
                # Check simple_test p-value (most reliable)
                if 'simple_test' in obj_results and isinstance(obj_results['simple_test'], dict):
                    p_val = obj_results['simple_test'].get('p_value')
                    if p_val is not None and p_val < 0.05:
                        significant_objectives += 1
                # Fallback to direct p-value
                elif obj_results.get('p_value', 1.0) < 0.05:
                    significant_objectives += 1
        
        # Overall assessment
        success_rate = successful_objectives / total_objectives if total_objectives > 0 else 0
        significance_rate = significant_objectives / successful_objectives if successful_objectives > 0 else 0
        
        if success_rate >= 0.8 and significance_rate >= 0.5:
            overall_validity = 'high'
        elif success_rate >= 0.6 and significance_rate >= 0.3:
            overall_validity = 'medium'  
        else:
            overall_validity = 'low'
        
        return {
            'successful_objectives': successful_objectives,
            'total_objectives': total_objectives,
            'significant_objectives': significant_objectives,
            'success_rate': success_rate,
            'significance_rate': significance_rate,
            'overall_validity': overall_validity
        }
    
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def _generate_enhanced_summary(self, results: Dict[str, Any], 
                                validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced summary with validation insights."""
        
        # Get multiple comparisons data safely
        mc_data = validation.get('multiple_comparisons_correction', {})
        fdr_significant = mc_data.get('fdr_significant', mc_data.get('fdr_bh_significant', 0))
        
        return {
            'objectives_extracted': len([r for r in results.values() if 'error' not in r]),
            'objectives_significant': validation.get('overall_validity', {}).get('significant_objectives', 0),
            'overall_validity': validation.get('overall_validity', {}).get('overall_validity', 'unknown'),
            'multiple_comparisons_impact': fdr_significant,
            'recommendations': validation.get('overall_validity', {}).get('recommendations', [])
        }

    def _generate_comprehensive_report(self, enhanced_results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive text report with detailed test descriptions and practical implications."""
        
        results = enhanced_results['objectives']
        validation = enhanced_results['validation']
        summary = enhanced_results['summary']
        
        # Safe access to validation data with defaults
        mc_data = validation.get('multiple_comparisons_correction', {})
        total_tests = mc_data.get('total_tests', len(results))
        original_sig = mc_data.get('original_significant', 'unknown')
        fdr_sig = mc_data.get('fdr_significant', mc_data.get('fdr_bh_significant', 'unknown'))
        recommendation = mc_data.get('recommendation', 'Apply appropriate multiple comparisons correction')
        
        report_lines = [
            "COMPLETE MOO OBJECTIVES ANALYSIS REPORT",
            "=" * 50,
            "",
            "This report presents 6 typing mechanics objectives extracted from",
            "bigram preference data, controlling for English language frequency effects.",
            "These objectives are designed for multi-objective keyboard layout optimization.",
            "",
            "ANALYSIS SUMMARY:",
            "=" * 20,
            f"Objectives successfully extracted: {summary.get('objectives_extracted', 'unknown')}/6",
            f"Objectives with significant results: {summary.get('objectives_significant', 'unknown')}",
            f"Overall validity assessment: {summary.get('overall_validity', 'unknown')}",
            "",
            "MULTIPLE COMPARISONS CORRECTION:",
            f"Tests performed: {total_tests}",
            f"Originally significant: {original_sig}",
            f"FDR-corrected significant: {fdr_sig}",
            "",
            "OBJECTIVES DETAILED RESULTS WITH TEST DESCRIPTIONS:",
            "=" * 55,
            ""
        ]
        
        # Detailed descriptions for each objective
        objective_details = {
            'key_preference': {
                'title': 'INDIVIDUAL KEY PREFERENCES',
                'what_compared': 'All pairwise comparisons between individual keys (48 pairs total from 12 left-hand keys)',
                'method': 'Pairwise analysis: For each key pair with sufficient data (≥10 comparisons), not including pinky/index comparisons or outside left finger-columns, determined preference rate using proportion tests. Combined results using Fisher\'s method for overall significance.',
                'frequency_control': 'Controlled for English bigram frequencies in contexts where keys appeared',
                'practical_meaning': 'Ranks individual keys by typing preference'
            },
            'row_separation': {
                'title': 'ROW SEPARATION PREFERENCES',
                'what_compared': 'Bigrams requiring different row movements: same row (no movement) vs. one row apart (reach) vs. two rows apart (hurdle)',
                'method': 'Instance-level analysis: For each comparison between bigrams with different row separations, recorded which was chosen. Tested preference for smaller vs. larger row movements.',
                'frequency_control': 'Controlled for English bigram frequencies to isolate motor control from linguistic preferences',
                'practical_meaning': 'Measures typing effort preferences related to vertical finger movement'
            },
            'column_separation': { 
                'title': 'COLUMN SEPARATION PREFERENCES',
                'what_compared': 'Different-finger bigrams with different horizontal spacing: adjacent columns (1 apart) vs separated columns (2-3 apart).',
                'method': 'Instance-level analysis: For each comparison between different-finger bigrams with different column separations, recorded which was chosen. Analyzed by row context (same row vs different rows) and specific separation distances.',
                'frequency_control': 'Controlled for English bigram frequencies to isolate spatial motor preferences from linguistic familiarity',
                'practical_meaning': 'Measures pure horizontal reach preferences independent of finger independence effects'
            },
        }
        
        # Add detailed results for each objective
        for i, (obj_name, obj_results) in enumerate(results.items(), 1):
            details = objective_details.get(obj_name, {})
            
            report_lines.extend([
                f"{i}. {details.get('title', obj_name.upper().replace('_', ' '))}:",
                "-" * (len(details.get('title', obj_name)) + 4),
                ""
            ])
            
            if 'error' in obj_results:
                report_lines.extend([
                    f"STATUS: FAILED",
                    f"Error: {obj_results['error']}",
                    ""
                ])
            else:
                # Add detailed description
                report_lines.extend([
                    "WHAT WAS COMPARED:",
                    f"  {details.get('what_compared', 'Details not available')}",
                    "",
                    "METHOD:",
                    f"  {details.get('method', 'Details not available')}",
                    "",
                    "FREQUENCY CONTROL:",
                    f"  {details.get('frequency_control', 'Standard bigram frequency controls applied')}",
                    "",
                    "STATISTICAL RESULTS:",
                    f"  Status: SUCCESS",
                    f"  Method: {obj_results.get('method', 'instance_level_analysis')}",
                    f"  Instances analyzed: {obj_results.get('n_instances', 'unknown')}",
                    f"  Users contributing: {obj_results.get('n_users', 'unknown')}",
                    f"  Result: {obj_results.get('interpretation', 'No interpretation available')}",
                    ""
                ])
                
                # Add specific results based on objective type
                if 'simple_test' in obj_results and isinstance(obj_results['simple_test'], dict):
                    simple_test = obj_results['simple_test']
                    pref_rate = simple_test.get('preference_rate', 'unknown')
                    p_val = simple_test.get('p_value', 'unknown')
                    effect_size = simple_test.get('effect_size', 'unknown')
                    
                    report_lines.extend([
                        "DETAILED STATISTICS:",
                        f"  Preference rate: {pref_rate:.1%}" if isinstance(pref_rate, (int, float)) else f"  Preference rate: {pref_rate}",
                        f"  P-value: {p_val:.2e}" if isinstance(p_val, (int, float)) and p_val > 0 else f"  P-value: {p_val}",
                        f"  Effect size (Cohen's h): {effect_size:.3f}" if isinstance(effect_size, (int, float)) else f"  Effect size: {effect_size}",
                        ""
                    ])
                
                # Add key preference specific details
                if obj_name == 'key_preference' and 'key_pair_results' in obj_results:
                    n_pairs = len(obj_results['key_pair_results'])
                    n_sig = obj_results.get('n_significant_pairs', 0)
                    report_lines.extend([
                        "KEY PREFERENCE DETAILS:",
                        f"  Total key pairs tested: {n_pairs}",
                        f"  Significant pairs (before correction): {obj_results.get('n_significant_pairs', 'unknown')}/{n_pairs}",
                        f"  FDR-corrected significant pairs: See multiple comparisons section",
                        ""
                    ])
                
                # Add practical implications
                report_lines.extend([
                    "PRACTICAL MEANING:",
                    f"  {details.get('practical_meaning', 'Measures typing preference patterns')}",
                    "",
                    "=" * 60,
                    ""
                ])
        
                # Add combined column detailed results
                if obj_name == 'column_separation':
                    self._add_column_results_to_report(obj_results, report_lines)

        # Add validation summary with safe access
        report_lines.extend([
            "",
            "VALIDATION AND STATISTICAL ROBUSTNESS:",
            "=" * 40,
            ""
        ])
        
        # Effect sizes
        effect_validation = validation.get('effect_size_validation', {})
        if effect_validation:
            report_lines.extend([
                "Effect Size Analysis:",
                "-" * 20,
                "Effect sizes indicate practical significance beyond statistical significance.",
                "Thresholds: negligible (<2%), small (2-5%), medium (5-10%), large (>10%)",
                ""
            ])
            for obj_name, effect_analysis in effect_validation.items():
                if isinstance(effect_analysis, dict):
                    interpretation = effect_analysis.get('interpretation', 'N/A')
                    practical_sig = effect_analysis.get('practical_significance', 'N/A')
                    mean_effect = effect_analysis.get('mean_effect', 'N/A')
                    report_lines.append(f"{obj_name}: {interpretation} (mean={mean_effect:.1%}, {practical_sig})" if isinstance(mean_effect, (int, float)) else f"{obj_name}: {interpretation} ({practical_sig})")
            report_lines.append("")
        
        # Multiple comparisons impact
        if 'objective_breakdown' in mc_data:
            breakdown = mc_data['objective_breakdown']
            report_lines.extend([
                "Multiple Comparisons Impact by Objective:",
                "-" * 40,
                "Shows how FDR correction affected each objective's significance:",
                ""
            ])
            for obj_name, obj_breakdown in breakdown.items():
                orig_sig = obj_breakdown.get('original_significant', 0)
                fdr_sig = obj_breakdown.get('fdr_significant', 0)
                total = obj_breakdown.get('total_tests', 0)
                report_lines.append(f"{obj_name}: {orig_sig}/{total} → {fdr_sig}/{total} significant after FDR correction")
            report_lines.append("")
        
        # Recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            report_lines.extend([
                "RECOMMENDATIONS FOR KEYBOARD LAYOUT OPTIMIZATION:",
                "=" * 50,
                ""
            ])
            
            report_lines.extend([
                "",
                "STATISTICAL NOTES:",
                f"• Apply FDR correction threshold α = 0.05 for {total_tests} tests",
                f"• {fdr_sig} objectives/tests remain significant after correction", 
                "• Effect sizes indicate practical vs. statistical significance"
            ])
        
        # Save report
        report_path = os.path.join(output_folder, 'complete_moo_objectives_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Enhanced comprehensive report saved to {report_path}")
                    
    def _add_column_results_to_report(self, results: Dict[str, Any], report_lines: List[str]) -> None:
        """Add detailed combined column results to report."""
        if 'comparison_results' in results:
            report_lines.extend([
                "DETAILED STATISTICS BY COMPARISON TYPE:",
                ""
            ])
            
            for comp_type, comp_results in results['comparison_results'].items():
                simple_test = comp_results['simple_test']
                
                # Clean up comparison type name
                if comp_type == "same_vs_adjacent_column":
                    title = "Same Finger vs Different Finger"
                elif comp_type == "adjacent_vs_remote_column":
                    title = "Adjacent vs Remote Columns"
                else:
                    title = comp_type.replace('_', ' ').title()
                
                report_lines.extend([
                    f"{title}:",
                    f"  Instances: {comp_results['n_instances']}",
                    f"  Preference rate: {comp_results['preference_rate']:.1%}",
                    f"  P-value: {comp_results['p_value']:.2e}" if comp_results['p_value'] > 0 else f"  P-value: <1e-16",
                    f"  Effect size: {simple_test.get('effect_size', 'N/A'):.3f}" if isinstance(simple_test.get('effect_size'), (int, float)) else f"  Effect size: N/A",
                    f"  Significant: {'Yes' if comp_results['p_value'] < 0.05 else 'No'}",
                    f"  Result: {comp_results['interpretation']}",
                    ""
                ])
                
                # Add examples if available
                if 'examples' in comp_results and comp_results['examples']:
                    report_lines.extend([
                        "  Examples:",
                        *[f"    • {example}" for example in comp_results['examples'][:2]],
                        ""
                    ])

        if 'weighted_analysis' in results:
            weighted = results['weighted_analysis']
            unweighted = weighted['unweighted_preference_rate']
            weighted_rate = weighted['weighted_preference_rate']
            
            report_lines.extend([
                "FREQUENCY BIAS ANALYSIS:",
                f"  Unweighted preference rate: {unweighted:.1%}",
                f"  Frequency-weighted rate: {weighted_rate:.1%}",
                f"  Bias magnitude: {abs(weighted_rate - unweighted):.1%}",
                f"  Frequency range: {weighted['frequency_stats']['min_frequency']} to {weighted['frequency_stats']['max_frequency']} comparisons",
                f"  {'SIGNIFICANT BIAS DETECTED' if abs(weighted_rate - unweighted) > 0.05 else 'Bias within acceptable range'}",
                ""
            ])        

        # Context results for same vs adjacent
        if 'context_results' in results and results['context_results']:
            report_lines.extend([
                "SAME COLUMN VS ADJACENT COLUMN BY ROW CONTEXT:",
                "(Same column = same finger; Adjacent column = different finger)",
                ""
            ])
            
            for context, context_data in results['context_results'].items():
                # Parse the row context more clearly
                if context.endswith('_row_separation'):
                    row_sep = context.replace('_row_separation', '')
                    context_desc = f"Bigrams with {row_sep}-row separation"
                else:
                    context_desc = context_data['description']
                           
                report_lines.extend([
                    f"{context_desc}:",
                    f"  What's compared: Same column (same finger) vs Adjacent column (different finger)",
                    f"  Instances: {context_data['n_instances']}",
                    f"  Adjacent column preference: {(1.0 - context_data['same_finger_rate']):.1%}",
                    f"  Same column preference: {context_data['same_finger_rate']:.1%}", 
                    f"  P-value: {context_data['simple_test']['p_value']:.2e}",
                    f"  Result: In this row context, {context_data['same_finger_rate']:.1%} prefer same column/finger",
                    ""
                ])

        # Context breakdown for adjacent vs remote
        if 'adjacent_vs_remote_column' in results.get('comparison_results', {}):
            # Get the raw data to analyze by context
            instances_data = results.get('instances_data')
            if instances_data is not None and not instances_data.empty:
                adj_vs_remote_data = instances_data[instances_data['comparison_type'] == 'adjacent_vs_remote_column']
                
                if len(adj_vs_remote_data) > 0:
                    report_lines.extend([
                        "ADJACENT VS REMOTE COLUMNS BY ROW CONTEXT:",
                        ""
                    ])
                    
                    for context in adj_vs_remote_data['context'].unique():
                        context_data = adj_vs_remote_data[adj_vs_remote_data['context'] == context]
                        if len(context_data) >= 10:
                            # Calculate preference for this context
                            adj_rate = context_data['chose_adjacent'].mean()
                            
                            # Parse context
                            if context.endswith('_row_separation'):
                                row_sep = context.replace('_row_separation', '')
                                context_desc = f"Bigrams with {row_sep}-row separation"
                            else:
                                context_desc = context
                            
                            report_lines.extend([
                                f"{context_desc}:",
                                f"  What's compared: Adjacent columns (1 apart) vs Remote columns (2-3 apart)",
                                f"  Instances: {len(context_data)}",
                                f"  Adjacent preference: {adj_rate:.1%}",
                                f"  Remote preference: {(1.0 - adj_rate):.1%}",
                                ""
                            ])
                                                                        
    def _save_key_scores_for_moo(self, results: Dict[str, Any], output_folder: str) -> None:
        """Save key preference scores in MOO-ready formats."""
        
        if 'key_preference' not in results or 'error' in results['key_preference']:
            logger.warning("No key preference results to save")
            return
        
        key_pref_results = results['key_preference']
        
        # Extract ranked key scores
        if 'ranked_keys' in key_pref_results:
            ranked_keys = key_pref_results['ranked_keys']
            
            # Create simple key -> score mapping with safe extraction
            key_scores = {}
            for key, score_data in ranked_keys:
                # Extract numeric score safely
                if isinstance(score_data, dict):
                    numeric_score = score_data.get('mean_score', 0.0)
                elif isinstance(score_data, (int, float)):
                    numeric_score = float(score_data)
                else:
                    numeric_score = 0.0
                key_scores[key] = numeric_score
            
            # Save as CSV for easy loading
            key_scores_df = pd.DataFrame([
                {
                    'key': key, 
                    'preference_score': key_scores[key],
                    'rank': i+1
                }
                for i, (key, _) in enumerate(ranked_keys)
            ])
            
            csv_path = os.path.join(output_folder, 'key_preference_scores.csv')
            key_scores_df.to_csv(csv_path, index=False)
            logger.info(f"Key preference scores saved to {csv_path}")
                        
            # Save FDR-corrected significant pairs only
            if 'key_pair_results' in key_pref_results:
                # Get FDR correction results
                validation = getattr(self, 'validation_results', {})
                mc_data = validation.get('multiple_comparisons_correction', {})
                fdr_corrections = mc_data.get('corrected_p_values', {}).get('fdr_bh', {})
                
                significant_pairs = []
                for (key1, key2), pair_data in key_pref_results['key_pair_results'].items():
                    pair_name = f"key_preference_{key1}_vs_{key2}"
                    fdr_p = fdr_corrections.get(pair_name, 1.0)
                    
                    if fdr_p < 0.05:  # FDR-significant
                        significant_pairs.append({
                            'key1': key1,
                            'key2': key2, 
                            'key1_preference_rate': pair_data['key1_preference_rate'],
                            'original_p_value': pair_data['p_value'],
                            'fdr_corrected_p_value': fdr_p,
                            'effect_size': pair_data['effect_size'],
                            'n_instances': pair_data['n_instances']
                        })
                
                if significant_pairs:
                    sig_pairs_df = pd.DataFrame(significant_pairs)
                    sig_path = os.path.join(output_folder, 'significant_key_pairs.csv')
                    sig_pairs_df.to_csv(sig_path, index=False)
                    logger.info(f"FDR-significant key pairs saved to {sig_path}")
            
            # Create template with safe score formatting
            key_score_lines = []
            for key in key_scores:
                key_score_lines.append(f'    "{key}": {key_scores[key]:.4f},')
            
            # Save MOO implementation code template
            moo_template = f"""# Key Preference Scores for MOO Implementation
    # Generated from typing preference analysis

    KEY_PREFERENCE_SCORES = {{
    {chr(10).join(key_score_lines)}
    }}

    def calculate_key_preference_objective(layout, letter_frequencies):
        \"\"\"Calculate key preference objective for keyboard layout.
        
        Args:
            layout: Dict mapping letters to positions
            letter_frequencies: Dict with letter frequency weights
            
        Returns:
            float: Key preference score (higher = better)
        \"\"\"
        score = 0.0
        for letter, frequency in letter_frequencies.items():
            if letter in KEY_PREFERENCE_SCORES:
                score += frequency * KEY_PREFERENCE_SCORES[letter]
        return score

    # Usage example:
    # english_letter_freq = {{'e': 0.127, 't': 0.091, 'a': 0.082, ...}}
    # layout_score = calculate_key_preference_objective(my_layout, english_letter_freq)
    """
            
            template_path = os.path.join(output_folder, 'key_preference_moo_template.py')
            with open(template_path, 'w') as f:
                f.write(moo_template)
            logger.info(f"MOO implementation template saved to {template_path}")

    def _save_results(self, enhanced_results: Dict[str, Any], output_folder: str) -> None:
        """Save enhanced results including key scores for MOO."""
        
        # Save key scores first
        self._save_key_scores_for_moo(enhanced_results['objectives'], output_folder)
        
        # Existing summary saving code...
        summary_data = []
        for obj_name, obj_results in enhanced_results['objectives'].items():
            if 'error' not in obj_results:
                summary_data.append({
                    'Objective': obj_name,
                    'Description': obj_results['description'],
                    'Status': 'Success',
                    'Normalization_Range': str(obj_results['normalization_range']),
                    'Interpretation': obj_results.get('interpretation', 'No interpretation')
                })
            else:
                summary_data.append({
                    'Objective': obj_name,
                    'Description': obj_results.get('description', 'Unknown'),
                    'Status': 'Failed',
                    'Error': obj_results['error'],
                    'Normalization_Range': 'N/A',
                    'Interpretation': 'Analysis failed'
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_folder, 'complete_moo_objectives_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to {summary_path}")
        
        # Rest of existing validation summary code...
        validation_summary = []
        for obj_name in enhanced_results['objectives'].keys():
            validation_info = {
                'Objective': obj_name,
                'Extracted': 'error' not in enhanced_results['objectives'][obj_name],
                'Effect_Size': 'N/A',
                'Power': 'N/A'
            }
            
            if 'effect_size_validation' in enhanced_results['validation']:
                effect_data = enhanced_results['validation']['effect_size_validation'].get(obj_name, {})
                validation_info['Effect_Size'] = effect_data.get('interpretation', 'N/A')
            
            if 'statistical_power' in enhanced_results['validation']:
                power_data = enhanced_results['validation']['statistical_power'].get(obj_name, {})
                validation_info['Power'] = power_data.get('power_level', 'N/A')
            
            validation_summary.append(validation_info)
        
        if validation_summary:
            validation_df = pd.DataFrame(validation_summary)
            validation_path = os.path.join(output_folder, 'validation_summary.csv')
            validation_df.to_csv(validation_path, index=False)
            logger.info(f"Validation summary saved to {validation_path}")

        # Key preference export
        if 'key_preference' in enhanced_results['objectives']:
            self._save_key_pairwise_results(
                enhanced_results['objectives']['key_preference'], 
                output_folder
            )

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Complete MOO objectives extraction and validation from bigram preference data'
    )
    parser.add_argument('--data', required=True,
                       help='Path to CSV file with bigram choice data')
    parser.add_argument('--output', required=True,
                       help='Output directory for results')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        return 1
    
    try:
        # Run complete MOO objectives analysis
        analyzer = CompleteMOOObjectiveAnalyzer(args.config)
        results = analyzer.analyze_moo_objectives(args.data, args.output)
        
        logger.info("Complete MOO objectives analysis finished successfully!")
        logger.info(f"Results saved to: {args.output}")
        
        # Print quick summary
        summary = results['summary']
        
        print(f"\nQUICK SUMMARY:")
        print(f"==============")
        print(f"Objectives extracted: {summary['objectives_extracted']}/6")
        print(f"Significant objectives: {summary['objectives_significant']}")
        print(f"Overall validity: {summary['overall_validity']}")
        print(f"FDR-corrected significant tests: {summary['multiple_comparisons_impact']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())