#!/usr/bin/env python3
"""
Complete Multi-Objective Optimization (MOO) Objectives Analysis for Keyboard Layout Optimization

This script extracts 7 typing mechanics objectives from bigram preference data,
controlling for English language frequency effects. These objectives are designed
to create meaningful conflicts for multi-objective keyboard layout optimization.

The 7 MOO objectives:
1. key_preference: Individual key quality preferences (strategic subset of pairwise comparisons)  
2. row_separation: Preferences for row transitions (same > reach > hurdle)
3. column_separation: Context-dependent column spacing preferences
4. column_4_vs_5: Specific preference for column 4 (RFV) vs column 5 (TGB)

Note: Only test #4 involves column 5, and only test #3 involves same-finger/column.

Includes rigorous statistical validation with:
- Standardized frequency weighting across all tests
- Multiple comparisons correction with detailed reporting
- Effect size validation
- Cross-validation
- Confound controls

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
    
    def __init__(self, config_path: str = None, target_key_pairs: Set[Tuple[str, str]] = None):
        """Initialize with optional target key pairs for testing."""
        self.config = self._load_config(config_path)
        self.key_positions = self._define_keyboard_layout()
        self.left_hand_keys = set(self.key_positions.keys())
        
        # Control which pairs to test
        self.target_key_pairs = target_key_pairs
                
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
                'figure_dpi': 300,
                'use_direct_key_preferences_only': True,  # Disable hybrid approach
                'min_direct_comparisons_per_pair': 5,    # Minimum data required
                'frequency_weighting_method': 'inverse_frequency'
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
            return {}
    
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
            return {}
        
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
        """Define left-hand keyboard layout (columns 1-4 only)."""
        return {
            # Keep original layout - columns 1-4 only
            'q': KeyPosition('q', 1, 1, 1), 'w': KeyPosition('w', 1, 2, 2),
            'e': KeyPosition('e', 1, 3, 3), 'r': KeyPosition('r', 1, 4, 4),
            'a': KeyPosition('a', 2, 1, 1), 's': KeyPosition('s', 2, 2, 2), 
            'd': KeyPosition('d', 2, 3, 3), 'f': KeyPosition('f', 2, 4, 4),
            'z': KeyPosition('z', 3, 1, 1), 'x': KeyPosition('x', 3, 2, 2),
            'c': KeyPosition('c', 3, 3, 3), 'v': KeyPosition('v', 3, 4, 4),
        }

    def _get_column_5_keys(self) -> Set[str]:
        """Get column 5 keys for column 4 vs 5 analysis only."""
        return {'t', 'g', 'b'}

    def analyze_moo_objectives(self, data_path: str, output_folder: str) -> Dict[str, Any]:
        """Run complete MOO objectives analysis with validation."""
        logger.info("Starting complete MOO objectives analysis with validation...")
        
        # Load and validate data
        self.data = self._load_and_validate_data(data_path)
        logger.info(f"Loaded {len(self.data)} rows from {self.data['user_id'].nunique()} participants")
        
        logger.info("=== DIAGNOSING REPEATED BIGRAM COVERAGE ===")
        self.diagnose_repeated_bigram_coverage()

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
        
        logger.info("=== OBJECTIVE 4: COLUMN_4_VS_5 ===")  
        results['column_4_vs_5'] = self._test_column_4_vs_5_preference()
        
        # Run validation framework
        logger.info("=== RUNNING VALIDATION FRAMEWORK ===")
        validation_results = {}
        validation_results['multiple_comparisons_correction'] = self._correct_multiple_comparisons(results)
        
        # Combine results
        enhanced_results = {
            'objectives': results,
            'validation': validation_results,
            'summary': self._generate_summary(results, validation_results)
        }
        
        # Generate comprehensive report
        logger.info("=== GENERATING REPORTS ===")
        self._generate_comprehensive_report(enhanced_results, output_folder)
        
        # Save results
        self._save_results(enhanced_results, output_folder)
        
        logger.info(f"Complete MOO objectives analysis finished! Results saved to {output_folder}")
        return enhanced_results

    def apply_frequency_weights(self, instances_df: pd.DataFrame) -> pd.DataFrame:
        """Apply standardized frequency weighting to instance-level data."""
        if instances_df.empty:
            return instances_df
        
        # Create unique comparison identifier
        if 'comparison' not in instances_df.columns:
            instances_df['comparison'] = instances_df.apply(
                lambda row: tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']])), axis=1
            )
        
        # Calculate frequency weights based on config
        freq_counts = instances_df['comparison'].value_counts()
        method = self.config.get('frequency_weighting_method', 'inverse_frequency')
        
        if method == 'inverse_frequency':
            # Weight = 1 / frequency (gives more weight to rare comparisons)
            instances_df['frequency_weight'] = instances_df['comparison'].map(lambda x: 1.0 / freq_counts[x])
        elif method == 'max_frequency_ratio':
            # Weight = max_frequency / frequency (alternative approach)
            max_frequency = freq_counts.max()
            instances_df['frequency_weight'] = instances_df['comparison'].map(lambda x: max_frequency / freq_counts[x])
        else:
            # Equal weighting (no correction)
            instances_df['frequency_weight'] = 1.0
        
        # Store frequency statistics for reporting
        instances_df['comparison_frequency'] = instances_df['comparison'].map(freq_counts)
        
        return instances_df

    def calculate_weighted_preference_metrics(self, instances_df: pd.DataFrame, outcome_var: str) -> Dict[str, Any]:
        """Calculate comprehensive preference metrics with frequency weighting."""
        
        # Apply frequency weights
        weighted_data = self.apply_frequency_weights(instances_df.copy())
        
        # Calculate unweighted metrics
        unweighted_rate = weighted_data[outcome_var].mean()
        n_instances = len(weighted_data)
        
        # Calculate weighted metrics
        total_weight = weighted_data['frequency_weight'].sum()
        weighted_rate = (weighted_data[outcome_var] * weighted_data['frequency_weight']).sum() / total_weight
        
        # Calculate bias magnitude
        bias_magnitude = abs(weighted_rate - unweighted_rate)
        
        # Frequency distribution statistics
        freq_stats = {
            'min_frequency': weighted_data['comparison_frequency'].min(),
            'max_frequency': weighted_data['comparison_frequency'].max(),
            'frequency_ratio': weighted_data['comparison_frequency'].max() / weighted_data['comparison_frequency'].min(),
            'unique_comparisons': weighted_data['comparison'].nunique(),
            'bias_severity': 'high' if bias_magnitude > 0.05 else 'moderate' if bias_magnitude > 0.02 else 'low'
        }
        
        # Calculate by comparison type if available
        weighted_by_type = {}
        if 'comparison_type' in weighted_data.columns:
            for comp_type in weighted_data['comparison_type'].unique():
                comp_data = weighted_data[weighted_data['comparison_type'] == comp_type]
                if len(comp_data) > 0:
                    comp_total_weight = comp_data['frequency_weight'].sum()
                    comp_weighted_rate = (comp_data[outcome_var] * comp_data['frequency_weight']).sum() / comp_total_weight
                    comp_unweighted_rate = comp_data[outcome_var].mean()
                    
                    weighted_by_type[comp_type] = {
                        'unweighted_rate': comp_unweighted_rate,
                        'weighted_rate': comp_weighted_rate,
                        'bias_magnitude': abs(comp_weighted_rate - comp_unweighted_rate),
                        'n_instances': len(comp_data)
                    }
        
        return {
            'unweighted_preference_rate': unweighted_rate,
            'weighted_preference_rate': weighted_rate,
            'bias_magnitude': bias_magnitude,
            'frequency_stats': freq_stats,
            'weighted_by_type': weighted_by_type,
            'n_instances': n_instances,
            'weighted_data': weighted_data  # Return for model fitting
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
            
            # Column 4 vs 5 classification
            col4_vs_5_type = self._classify_column_4_vs_5(bigram)
            
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
                'col_4_vs_5_type': col4_vs_5_type,
                'english_bigram_freq': self.english_bigram_frequencies.get(bigram, 0),
                'relevant_for_key_preference': is_left_hand,
                'relevant_for_row_separation': is_left_hand,
                'relevant_for_column_separation': is_left_hand,
                'relevant_for_col_4_vs_5': col4_vs_5_type != 'neither'
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
            'separated_cols_bigrams': len(df[df['col_category'] == 'separated_cols']),
            'column_4_bigrams': len(df[df['col_4_vs_5_type'] == 'column_4']),
            'column_5_bigrams': len(df[df['col_4_vs_5_type'] == 'column_5']),
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

    # =========================================================================
    # OBJECTIVE 1: KEY PREFERENCE WITH STRATEGIC SUBSET OPTION
    # =========================================================================
    
    def diagnose_repeated_bigram_coverage(self) -> None:
            """Diagnose what repeated bigram coverage we have for direct key preferences."""
            logger.info("Diagnosing repeated bigram coverage...")
            
            # Find all repeated bigram comparisons
            repeated_comparisons = []
            for _, row in self.data.iterrows():
                chosen = str(row['chosen_bigram']).lower()
                unchosen = str(row['unchosen_bigram']).lower()
                
                chosen_is_repeated = len(chosen) == 2 and chosen[0] == chosen[1]
                unchosen_is_repeated = len(unchosen) == 2 and unchosen[0] == unchosen[1]
                
                if chosen_is_repeated and unchosen_is_repeated:
                    chosen_key = chosen[0]
                    unchosen_key = unchosen[0]
                    
                    if chosen_key in self.left_hand_keys and unchosen_key in self.left_hand_keys:
                        repeated_comparisons.append({
                            'comparison': f"{chosen_key.upper()}{chosen_key.upper()} vs {unchosen_key.upper()}{unchosen_key.upper()}",
                            'chosen_key': chosen_key,
                            'unchosen_key': unchosen_key,
                            'key_pair': tuple(sorted([chosen_key, unchosen_key])),
                            'user_id': row['user_id']
                        })
            
            if not repeated_comparisons:
                print("❌ NO REPEATED BIGRAM COMPARISONS FOUND")
                print("This means the hybrid approach will fall back to inferred method only.")
                return
            
            df = pd.DataFrame(repeated_comparisons)
            
            # Summary statistics
            unique_comparisons = df.groupby(['chosen_key', 'unchosen_key']).size().reset_index(name='count')
            unique_pairs = df['key_pair'].nunique()
            total_possible_pairs = len(self.left_hand_keys) * (len(self.left_hand_keys) - 1) // 2
            
            print(f"\n✅ REPEATED BIGRAM ANALYSIS:")
            print(f"Total instances: {len(df)}")
            print(f"Unique users: {df['user_id'].nunique()}")
            print(f"Unique key pairs covered: {unique_pairs}/{total_possible_pairs} ({unique_pairs/total_possible_pairs*100:.1f}%)")
            
            print(f"\nMost frequent repeated bigram comparisons:")
            top_comparisons = unique_comparisons.nlargest(10, 'count')
            for _, row in top_comparisons.iterrows():
                print(f"  {row['chosen_key'].upper()}{row['chosen_key'].upper()} vs {row['unchosen_key'].upper()}{row['unchosen_key'].upper()}: {row['count']} instances")
            
            # Show key coverage
            all_keys_in_repeated = set()
            for key_pair in df['key_pair'].unique():
                all_keys_in_repeated.update(key_pair)
            
            print(f"\nKeys with direct repeated bigram evidence: {sorted(all_keys_in_repeated)}")
            missing_keys = set(self.left_hand_keys) - all_keys_in_repeated
            if missing_keys:
                print(f"Keys missing from repeated bigrams: {sorted(missing_keys)}")
            
            # Quality check - look for concerning patterns
            user_counts = df['user_id'].value_counts()
            if user_counts.max() > 50:
                print(f"\n⚠️  WARNING: Some users have many repeated bigram choices:")
                heavy_users = user_counts.head(3)
                for user_id, count in heavy_users.items():
                    print(f"  User {user_id}: {count} repeated bigram choices")
                print("  This might indicate non-serious responses or gaming behavior.")
            
            # Preference direction preview
            print(f"\nPreference direction preview (who wins in direct comparisons):")
            direction_preview = df.groupby('key_pair')['chosen_key'].apply(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'tie').reset_index()
            direction_preview.columns = ['key_pair', 'usually_preferred']
            
            for _, row in direction_preview.head(8).iterrows():
                key1, key2 = row['key_pair']
                winner = row['usually_preferred']
                loser = key2 if winner == key1 else key1
                pair_data = df[df['key_pair'] == row['key_pair']]
                winner_count = len(pair_data[pair_data['chosen_key'] == winner])
                total_count = len(pair_data)
                rate = winner_count / total_count * 100
                print(f"  {winner.upper()} > {loser.upper()}: {rate:.0f}% ({winner_count}/{total_count})")
                        
            return df

    def _test_key_preference(self) -> Dict[str, Any]:
        """
        Test key preferences using ONLY direct repeated bigrams to get preference rates.
        """
        logger.info("Testing key preferences with direct repeated bigrams only...")

        # Extract ONLY direct preferences (no hybrid/inferred)
        direct_instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            # Check if both are repeated letter bigrams (aa, bb, cc, etc.)
            chosen_is_repeated = len(chosen) == 2 and chosen[0] == chosen[1]
            unchosen_is_repeated = len(unchosen) == 2 and unchosen[0] == unchosen[1]
            
            if chosen_is_repeated and unchosen_is_repeated:
                chosen_key = chosen[0]
                unchosen_key = unchosen[0]
                
                # Only include left-hand keys
                if chosen_key in self.left_hand_keys and unchosen_key in self.left_hand_keys:
                    direct_instances.append({
                        'user_id': row['user_id'],
                        'chosen_bigram': chosen,  # Keep original column names for compatibility
                        'unchosen_bigram': unchosen,  # Keep original column names for compatibility
                        'chosen_key': chosen_key,
                        'unchosen_key': unchosen_key,
                        'chose_chosen_key': 1,  # By definition, chosen_key was preferred
                        'key_pair': tuple(sorted([chosen_key, unchosen_key])),
                        'slider_value': row.get('sliderValue', 0),
                        'comparison_type': 'direct_repeated_bigram',
                        'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                        'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                    })
        
        direct_df = pd.DataFrame(direct_instances)
        
        if direct_df.empty:
            return {'error': 'No direct key preference instances found'}
        
        logger.info(f"Found {len(direct_df)} direct instances from {direct_df['user_id'].nunique()} users")
        
        # Apply frequency weighting (now works because we have the expected column names)
        weighted_metrics = self.calculate_weighted_preference_metrics(direct_df, 'chose_chosen_key')
        weighted_data = weighted_metrics['weighted_data']
        
        # Analyze each key pair with Wilson confidence intervals
        pairwise_results = {}
        all_p_values = []
        
        for key_pair in weighted_data['key_pair'].unique():
            pair_data = weighted_data[weighted_data['key_pair'] == key_pair].copy()
            
            if len(pair_data) < 1:  # Minimum threshold
                continue
            
            key1, key2 = key_pair
            
            # Calculate win rate for key1
            pair_data['chose_key1'] = pair_data.apply(
                lambda row: 1 if row['chosen_key'] == key1 else 0, axis=1
            )
            
            n_total = len(pair_data)
            n_key1_wins = pair_data['chose_key1'].sum()
            win_rate = n_key1_wins / n_total
            
            # Weighted win rate using frequency weights
            total_weight = pair_data['frequency_weight'].sum()
            weighted_win_rate = (pair_data['chose_key1'] * pair_data['frequency_weight']).sum() / total_weight
            
            # Wilson score confidence interval using effective sample size
            from scipy import stats
            z = stats.norm.ppf(0.975)  # 95% CI
            
            # Use weighted rate and effective sample size for CI
            effective_n = (total_weight ** 2) / (pair_data['frequency_weight'] ** 2).sum()
            p_hat = weighted_win_rate
            n = effective_n
            
            if n > 0:
                denominator = 1 + z**2 / n
                center = (p_hat + z**2 / (2*n)) / denominator
                margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
                
                ci_lower = max(0, center - margin)
                ci_upper = min(1, center + margin)
            else:
                ci_lower = ci_upper = weighted_win_rate
            
            # Test against null hypothesis of no preference (50%) using weighted rate
            if effective_n > 0:
                z_score = (weighted_win_rate - 0.5) / np.sqrt(0.25 / effective_n)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                p_value = 1.0
            
            # Determine winner based on weighted win rate
            winner = key1 if weighted_win_rate > 0.5 else key2
            winner_rate = max(weighted_win_rate, 1 - weighted_win_rate)
            effect_size = abs(weighted_win_rate - 0.5)
            
            # Calculate frequency bias
            frequency_bias = abs(weighted_win_rate - win_rate)
            
            pairwise_results[key_pair] = {
                'key1': key1,
                'key2': key2,
                'n_comparisons': n_total,
                'effective_n': effective_n,
                'key1_wins': n_key1_wins,
                'unweighted_key1_rate': win_rate,
                'weighted_key1_rate': weighted_win_rate,
                'frequency_bias': frequency_bias,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'winner': winner,
                'winner_rate': winner_rate,
                'effect_size': effect_size,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'primary_method': 'direct',  # All are direct for this analysis
                'interpretation': f"{winner.upper()} > {key2.upper() if winner == key1 else key1.upper()}: {winner_rate:.1%} preference (direct evidence)"
            }
            
            all_p_values.append(p_value)
        
        # Calculate individual key scores using weighted win-loss method
        individual_scores, ranked_keys = self._calculate_frequency_weighted_key_scores(pairwise_results)
        
        return {
            'description': 'Key preferences from direct repeated bigrams only',
            'method': 'direct_repeated_bigrams_only',
            'n_instances': len(direct_df),
            'n_users': direct_df['user_id'].nunique(),
            'n_key_pairs': len(pairwise_results),
            'n_significant_pairs': sum(1 for r in pairwise_results.values() if r['significant']),
            'pairwise_results': pairwise_results,
            'frequency_bias_analysis': weighted_metrics,
            'key_preference_rates': True,
            'p_values_for_correction': all_p_values,  # For FDR correction later
            'instances_data': direct_df,  # For validation framework
            'normalization_range': (0.0, 1.0),  # Preference rates are 0-100%
            'p_value': min(all_p_values) if all_p_values else 1.0,
            'preference_rate': weighted_metrics.get('weighted_preference_rate', 0.5),
            'interpretation': self._interpret_key_results(pairwise_results, weighted_metrics)
        }

    def _interpret_key_results(self, pairwise_results: Dict, weighted_metrics: Dict) -> str:
        """Interpret key preference results with frequency bias info."""
        if not pairwise_results:
            return "No key preference data available"
        
        rates = [r['winner_rate'] for r in pairwise_results.values()]
        mean_rate = np.mean(rates)
        min_rate = min(rates)
        max_rate = max(rates)
        
        n_significant = sum(1 for r in pairwise_results.values() if r['significant'])
        
        # Check frequency bias impact
        bias_magnitudes = [r.get('frequency_bias', 0) for r in pairwise_results.values()]
        mean_bias = np.mean(bias_magnitudes)
        overall_bias = weighted_metrics.get('bias_magnitude', 0)
        
        interpretation = f"Key preferences: {n_significant}/{len(pairwise_results)} significant pairs, preference rates {min_rate:.1%}-{max_rate:.1%} (mean: {mean_rate:.1%})"
        
        if mean_bias > 0.02 or overall_bias > 0.02:
            interpretation += f", frequency bias: {mean_bias:.1%} (overall: {overall_bias:.1%})"
        
        return interpretation

    def _calculate_frequency_weighted_key_scores(self, pairwise_results: Dict) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """Calculate individual key scores using frequency-weighted win-loss method."""
        key_scores = {}
        all_keys = set()
        
        # Collect all keys
        for pair, results in pairwise_results.items():
            if 'error' not in results:
                all_keys.update([results['key1'], results['key2']])
        
        # Initialize scores
        for key in all_keys:
            key_scores[key] = {'wins': 0, 'losses': 0, 'total_weight': 0}
        
        # Calculate weighted wins/losses
        for pair, results in pairwise_results.items():
            if 'error' not in results:
                key1, key2 = results['key1'], results['key2']
                
                # Use inverse of frequency bias as weight (higher weight for less biased comparisons)
                comparison_weight = 1.0 / (1.0 + results.get('frequency_bias', 0))
                
                # Determine winner based on weighted preference rate
                if results.get('weighted_key1_rate', results.get('unweighted_key1_rate', 0.5)) > 0.5:
                    key_scores[key1]['wins'] += comparison_weight
                    key_scores[key2]['losses'] += comparison_weight
                else:
                    key_scores[key2]['wins'] += comparison_weight
                    key_scores[key1]['losses'] += comparison_weight
                
                key_scores[key1]['total_weight'] += comparison_weight
                key_scores[key2]['total_weight'] += comparison_weight
        
        # Calculate final scores (win rate)
        final_scores = {}
        for key in all_keys:
            total_games = key_scores[key]['wins'] + key_scores[key]['losses']
            if total_games > 0:
                final_scores[key] = key_scores[key]['wins'] / total_games
            else:
                final_scores[key] = 0.5  # Neutral if no data
        
        # Sort by score
        ranked_keys = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        return final_scores, ranked_keys

    def _weighted_proportion_test(self, data: pd.DataFrame, outcome_var: str, weight_var: str) -> Dict[str, Any]:
            """Perform proportion test with frequency weights."""
            n_instances = len(data)
            
            # Weighted calculations
            total_weight = data[weight_var].sum()
            weighted_preference = (data[outcome_var] * data[weight_var]).sum() / total_weight
            
            # For z-test, use effective sample size
            # Effective N = (sum of weights)^2 / (sum of squared weights)
            effective_n = (total_weight ** 2) / (data[weight_var] ** 2).sum()
            
            # Calculate z-test for weighted proportion
            expected = 0.5  # Null hypothesis: no preference
            se = np.sqrt(expected * (1 - expected) / effective_n)
            z_score = (weighted_preference - expected) / se
            
            # Two-tailed p-value
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            
            # Effect size (Cohen's h for proportions)
            effect_size = 2 * (np.arcsin(np.sqrt(weighted_preference)) - np.arcsin(np.sqrt(0.5)))
            
            return {
                'n_instances': n_instances,
                'effective_n': effective_n,
                'weighted_preference_rate': weighted_preference,
                'z_score': z_score,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < 0.05
            }

    # =========================================================================
    # OBJECTIVE 2: ROW SEPARATION WITH STANDARDIZED WEIGHTING
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
        """Test row separation preferences with standardized frequency weighting."""
        logger.info("Testing row separation preferences with standardized frequency weighting...")
        
        instances_df = self._extract_row_separation_instances()
        
        if instances_df.empty:
            return {'error': 'No row separation instances found'}
        
        logger.info(f"Found {len(instances_df)} row separation instances from {instances_df['user_id'].nunique()} users")
        
        # Apply standardized frequency weighting
        weighted_metrics = self.calculate_weighted_preference_metrics(instances_df, 'chose_smaller_separation')
        weighted_data = weighted_metrics['weighted_data']
        
        # Overall weighted analysis
        simple_test = self._weighted_proportion_test(weighted_data, 'chose_smaller_separation', 'frequency_weight')
        
        # Model results with frequency weights
        model_results = self._fit_weighted_instance_level_model(
            weighted_data,
            'chose_smaller_separation', 
            'frequency_weight',
            'pairwise_row_separation'
        )
        
        # Analysis by specific comparison type
        comparison_results = {}
        for comp_type in weighted_data['comparison_type'].unique():
            comp_data = weighted_data[weighted_data['comparison_type'] == comp_type]
            if len(comp_data) >= 10:
                comp_weighted_test = self._weighted_proportion_test(comp_data, 'chose_smaller_separation', 'frequency_weight')
                comparison_results[comp_type] = {
                    'weighted_test': comp_weighted_test,
                    'n_instances': len(comp_data),
                    'unweighted_rate': comp_data['chose_smaller_separation'].mean(),
                    'weighted_rate': comp_weighted_test['weighted_preference_rate'],
                    'frequency_bias': abs(comp_weighted_test['weighted_preference_rate'] - comp_data['chose_smaller_separation'].mean()),
                    'p_value': comp_weighted_test['p_value'],
                    'interpretation': f"{comp_type.replace('_', ' ')}: {comp_weighted_test['weighted_preference_rate']:.1%} prefer smaller separation (frequency-corrected)"
                }
        
        return {
            'description': 'Row separation preferences with standardized frequency weighting',
            'method': 'frequency_weighted_row_separation_analysis',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique(),
            'simple_test': simple_test,
            'model_results': model_results,
            'comparison_results': comparison_results,
            'frequency_bias_analysis': weighted_metrics,
            'p_value': simple_test['p_value'],
            'preference_rate': simple_test['weighted_preference_rate'],
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_weighted_row_results(simple_test, comparison_results, weighted_metrics)
        }

    def _interpret_weighted_row_results(self, weighted_test: Dict, comparison_results: Dict, weighted_metrics: Dict) -> str:
        """Interpret frequency-weighted row separation results."""
        weighted_rate = weighted_test['weighted_preference_rate']
        unweighted_rate = weighted_metrics['unweighted_preference_rate']
        bias_magnitude = weighted_metrics['bias_magnitude']
        
        interpretation = f"Overall preference for smaller row separation: {weighted_rate:.1%} (frequency-corrected)"
        
        if bias_magnitude > 0.02:  # 2% threshold for notable bias
            interpretation += f"; Unweighted: {unweighted_rate:.1%} (bias: {bias_magnitude:.1%})"
        
        # Add comparison breakdowns
        comparison_summaries = []
        for comp_type, results in comparison_results.items():
            rate = results['weighted_rate']
            bias = results['frequency_bias']
            clean_name = comp_type.replace('_', ' ')
            comparison_summaries.append(f"{clean_name}: {rate:.1%} (bias: {bias:.1%})")
        
        if comparison_summaries:
            interpretation += f". Breakdown: {'; '.join(comparison_summaries)}"
        
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
    # OBJECTIVE 3: COLUMN SEPARATION WITH STANDARDIZED WEIGHTING
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
                comparison_type = "same_finger_vs_different_finger"  # RENAMED for clarity
                context = f"{row_context}_row_separation"
                chose_different_finger = 1 if chosen_col_sep == 1 else 0  # RENAMED for clarity

            # Test 2: Adjacent (1) vs Remote (2-3) - same row context, different fingers only
            elif separations in [{1, 2}, {1, 3}] and chosen_col_sep > 0 and unchosen_col_sep > 0:
                comparison_type = "adjacent_vs_remote_columns"  # RENAMED for clarity  
                context = f"{row_context}_row_separation"
                chose_adjacent = 1 if chosen_col_sep == 1 else 0  # Keep this name
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
        """Test column separation preferences with standardized frequency weighting."""
        logger.info("Testing column separation preferences with standardized frequency weighting...")
        
        instances_df = self._extract_column_instances()
        
        if instances_df.empty:
            return {'error': 'No column separation instances found'}
        
        logger.info(f"Found {len(instances_df)} column instances from {instances_df['user_id'].nunique()} users")
        
        # Apply standardized frequency weighting
        weighted_metrics = self.calculate_weighted_preference_metrics(instances_df, 'chose_adjacent')
        weighted_data = weighted_metrics['weighted_data']
        
        # Overall weighted analysis
        simple_test = self._weighted_proportion_test(weighted_data, 'chose_adjacent', 'frequency_weight')
        
        # Model results with frequency weights
        model_results = self._fit_weighted_instance_level_model(
            weighted_data,
            'chose_adjacent', 
            'frequency_weight',
            'column_separation'
        )

        # Analysis by comparison type
        comparison_results = {}
        for comp_type in weighted_data['comparison_type'].unique():
            comp_data = weighted_data[weighted_data['comparison_type'] == comp_type]
            if len(comp_data) >= 10:
                # Use different outcome variable based on comparison type
                if comp_type == "same_finger_vs_different_finger":
                    outcome_var = 'chose_different_finger'
                    interpretation_template = "Same finger vs Different finger: {rate:.1%} prefer different finger"
                elif comp_type == "adjacent_vs_remote_columns":
                    outcome_var = 'chose_adjacent'
                    interpretation_template = "Adjacent vs Remote columns: {rate:.1%} prefer adjacent columns"
                
                comp_weighted_test = self._weighted_proportion_test(comp_data, outcome_var, 'frequency_weight')
                
                comparison_results[comp_type] = {
                    'weighted_test': comp_weighted_test,
                    'n_instances': len(comp_data),
                    'unweighted_rate': comp_data[outcome_var].mean(),
                    'weighted_rate': comp_weighted_test['weighted_preference_rate'],
                    'frequency_bias': abs(comp_weighted_test['weighted_preference_rate'] - comp_data[outcome_var].mean()),
                    'p_value': comp_weighted_test['p_value'],
                    'interpretation': interpretation_template.format(rate=comp_weighted_test['weighted_preference_rate']),
                    'examples': self._get_comparison_examples(comp_data, comp_type)
                }
        
        # Context analysis for same vs adjacent (controlling for row separation)
        context_results = {}
        for row_context in weighted_data['row_separation_context'].unique():
            context_data = weighted_data[weighted_data['row_separation_context'] == row_context]
            
            context_results[f"row_{row_context}"] = {
                'description': f"{row_context}-row separation",
                'total_instances': len(context_data)
            }
            
            # Analyze same finger vs different finger within this row context
            same_vs_diff = context_data[context_data['comparison_type'] == 'same_finger_vs_different_finger']
            if len(same_vs_diff) >= 10:
                same_diff_test = self._weighted_proportion_test(same_vs_diff, 'chose_different_finger', 'frequency_weight')
                context_results[f"row_{row_context}"]['same_vs_different'] = {
                    'n_instances': len(same_vs_diff),
                    'prefer_different_finger_rate': same_diff_test['weighted_preference_rate'],
                    'prefer_same_finger_rate': 1.0 - same_diff_test['weighted_preference_rate'],
                    'p_value': same_diff_test['p_value'],
                    'interpretation': f"Same vs Different finger: {same_diff_test['weighted_preference_rate']:.1%} prefer different finger"
                }
            
            # Analyze adjacent vs remote within this row context  
            adj_vs_remote = context_data[context_data['comparison_type'] == 'adjacent_vs_remote_columns']
            if len(adj_vs_remote) >= 10:
                adj_remote_test = self._weighted_proportion_test(adj_vs_remote, 'chose_adjacent', 'frequency_weight')
                context_results[f"row_{row_context}"]['adjacent_vs_remote'] = {
                    'n_instances': len(adj_vs_remote),
                    'prefer_adjacent_rate': adj_remote_test['weighted_preference_rate'],
                    'prefer_remote_rate': 1.0 - adj_remote_test['weighted_preference_rate'],
                    'p_value': adj_remote_test['p_value'],
                    'interpretation': f"Adjacent vs Remote: {adj_remote_test['weighted_preference_rate']:.1%} prefer adjacent columns"
                }
                        
        return {
            'description': 'Column separation preferences with standardized frequency weighting',
            'method': 'frequency_weighted_column_analysis',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique(),
            'simple_test': simple_test,
            'model_results': model_results,
            'comparison_results': comparison_results,
            'context_results': context_results,
            'frequency_bias_analysis': weighted_metrics,
            'p_value': simple_test['p_value'],
            'preference_rate': simple_test['weighted_preference_rate'],
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_weighted_column_results(simple_test, weighted_metrics),
            'frequency_bias_detected': weighted_metrics['bias_magnitude'] > 0.02,
        }

    def _interpret_weighted_column_results(self, weighted_test: Dict, weighted_metrics: Dict) -> str:
        """Interpret frequency-weighted column results."""
        weighted_rate = weighted_test['weighted_preference_rate']
        unweighted_rate = weighted_metrics['unweighted_preference_rate']
        bias_magnitude = weighted_metrics['bias_magnitude']
        
        # Main interpretation
        if weighted_rate > 0.6:
            strength = "strong"
        elif weighted_rate > 0.55:
            strength = "moderate" 
        else:
            strength = "weak"
        
        interpretation = f"Overall preference for adjacent columns: {weighted_rate:.1%} (frequency-corrected, {strength} effect)"
        
        if bias_magnitude > 0.02:
            interpretation += f"; Unweighted: {unweighted_rate:.1%} (bias: {bias_magnitude:.1%})"
        
        # Add insight about what this means
        if weighted_rate > 0.5:
            interpretation += f". Users prefer different fingers over same finger by {(weighted_rate - 0.5) * 2:.0%}."
        else:
            interpretation += f". Users prefer same finger over different fingers by {(0.5 - weighted_rate) * 2:.0%}."
        
        return interpretation

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
    # OBJECTIVE 4: COLUMN 4 VS 5 PREFERENCE (ACTIVATED)
    # =========================================================================

    def _classify_column_4_vs_5(self, bigram: str) -> str:
        """Use separate column definitions for this test only."""
        col4_keys = {'r', 'f', 'v'}
        col5_keys = self._get_column_5_keys()  # Separate from main layout
        
        bigram_keys = set(bigram)
        
        # Pure column 4 (uses only column 4 keys)
        if bigram_keys.issubset(col4_keys):
            return 'column_4'
        # Pure column 5 (uses only column 5 keys) 
        elif bigram_keys.issubset(col5_keys):
            return 'column_5'
        # Mixed or neither
        else:
            return 'neither'

    def _extract_column_4_vs_5_instances(self) -> pd.DataFrame:
        """Extract instances comparing column 4 vs column 5 bigrams."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
            
            chosen_type = self._classify_column_4_vs_5(chosen)
            unchosen_type = self._classify_column_4_vs_5(unchosen)
            
            # Only include pure column 4 vs pure column 5 comparisons
            if {chosen_type, unchosen_type} == {'column_4', 'column_5'}:
                
                # Code outcome: 1 if column 4 was chosen, 0 if column 5 was chosen
                chose_column_4 = 1 if chosen_type == 'column_4' else 0
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'chose_column_4': chose_column_4,
                    'chosen_type': chosen_type,
                    'unchosen_type': unchosen_type,
                    'slider_value': row.get('sliderValue', 0),
                    'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                    'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                })
        
        return pd.DataFrame(instances)

    def _test_column_4_vs_5_preference(self) -> Dict[str, Any]:
        """Test preference for column 4 (RFV) vs column 5 (TGB) with standardized frequency weighting."""
        logger.info("Testing column 4 vs 5 preference with standardized frequency weighting...")
        
        instances_df = self._extract_column_4_vs_5_instances()
        
        if instances_df.empty:
            return {'error': 'No column 4 vs 5 instances found'}
        
        logger.info(f"Found {len(instances_df)} column 4 vs 5 instances from {instances_df['user_id'].nunique()} users")
        
        # Apply standardized frequency weighting
        weighted_metrics = self.calculate_weighted_preference_metrics(instances_df, 'chose_column_4')
        weighted_data = weighted_metrics['weighted_data']
        
        # Weighted proportion test
        simple_test = self._weighted_proportion_test(weighted_data, 'chose_column_4', 'frequency_weight')
        
        # Model results with frequency weights
        model_results = self._fit_weighted_instance_level_model(
            weighted_data,
            'chose_column_4', 
            'frequency_weight',
            'column_4_vs_5_preference'
        )
        
        return {
            'description': 'Column 4 (RFV) vs Column 5 (TGB) preference with frequency weighting',
            'method': 'frequency_weighted_column_4_vs_5_analysis',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique(),
            'simple_test': simple_test,
            'model_results': model_results,
            'frequency_bias_analysis': weighted_metrics,
            'p_value': simple_test['p_value'],
            'preference_rate': simple_test['weighted_preference_rate'],
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_column_4_vs_5_result(simple_test, weighted_metrics)
        }
    
    def _interpret_column_4_vs_5_result(self, weighted_test: Dict, weighted_metrics: Dict) -> str:
        """Interpret column 4 vs 5 preference results with frequency bias info."""
        weighted_rate = weighted_test['weighted_preference_rate']
        unweighted_rate = weighted_metrics['unweighted_preference_rate']
        bias_magnitude = weighted_metrics['bias_magnitude']
        p_value = weighted_test['p_value']
        
        if p_value < 0.05:
            if weighted_rate > 0.5:
                strength = "strong" if weighted_rate > 0.65 else "moderate" if weighted_rate > 0.55 else "weak"
                base_interpretation = f"Significant preference for column 4 (RFV) over column 5 (TGB): {weighted_rate:.1%} (frequency-corrected, {strength} effect)"
            else:
                strength = "strong" if weighted_rate < 0.35 else "moderate" if weighted_rate < 0.45 else "weak"
                base_interpretation = f"Significant preference for column 5 (TGB) over column 4 (RFV): {(1-weighted_rate):.1%} (frequency-corrected, {strength} effect)"
        else:
            base_interpretation = f"No significant column preference detected: {weighted_rate:.1%} prefer column 4 (p={p_value:.3f})"
        
        if bias_magnitude > 0.02:
            base_interpretation += f"; Unweighted: {unweighted_rate:.1%} (bias: {bias_magnitude:.1%})"
        
        return base_interpretation

    # =========================================================================
    # MODEL FITTING WITH FREQUENCY WEIGHTS
    # =========================================================================
    
    def _fit_weighted_instance_level_model(self, weighted_data: pd.DataFrame, outcome_var: str, 
                                         weight_var: str, model_name: str) -> Dict[str, Any]:
        """Fit regression model on weighted instance-level data."""
        
        if len(weighted_data) < 20:
            return {'error': f'Insufficient instances for {model_name} (need ≥20, got {len(weighted_data)})'}
        
        try:
            # Calculate basic weighted preference rate
            total_weight = weighted_data[weight_var].sum()
            weighted_preference_rate = (weighted_data[outcome_var] * weighted_data[weight_var]).sum() / total_weight
            n_users = weighted_data['user_id'].nunique()
            
            # Prepare data for modeling
            control_cols = [col for col in weighted_data.columns if col.startswith('log_')]
            
            if len(control_cols) == 0:
                # No controls available - use weighted proportion test
                return {
                    'method': 'weighted_proportion',
                    'weighted_preference_rate': weighted_preference_rate,
                    'n_instances': len(weighted_data),
                    'n_users': n_users,
                    'interpretation': self._interpret_weighted_preference_rate(weighted_preference_rate, outcome_var),
                    'robust': True
                }
            
            X = weighted_data[control_cols].fillna(weighted_data[control_cols].median())
            X = sm.add_constant(X)
            y = weighted_data[outcome_var]
            weights = weighted_data[weight_var]
            
            # Try weighted logistic regression
            try:
                model = sm.WLS(y, X, weights=weights).fit()
                
                return {
                    'method': 'weighted_regression',
                    'model': model,
                    'weighted_preference_rate': weighted_preference_rate,
                    'n_instances': len(weighted_data),
                    'n_users': n_users,
                    'r_squared': model.rsquared,
                    'interpretation': self._interpret_weighted_preference_rate(weighted_preference_rate, outcome_var),
                    'robust': True
                }
                
            except Exception as reg_error:
                logger.warning(f"Weighted regression failed for {model_name}: {reg_error}")
                
                # Fall back to weighted proportion analysis
                return {
                    'method': 'weighted_proportion_fallback',
                    'weighted_preference_rate': weighted_preference_rate,
                    'n_instances': len(weighted_data),
                    'n_users': n_users,
                    'interpretation': self._interpret_weighted_preference_rate(weighted_preference_rate, outcome_var),
                    'robust': True,
                    'fallback_reason': str(reg_error)
                }
            
        except Exception as e:
            logger.warning(f"Weighted model fitting failed for {model_name}: {e}")
            return {'error': f'Weighted model fitting failed: {str(e)}', 'n_instances': len(weighted_data)}
        
    def _interpret_weighted_preference_rate(self, rate: float, outcome_var: str) -> str:
        """Interpret the frequency-weighted preference rate for different outcomes."""
        interpretations = {
            'chose_adjacent': {
                'variable': 'adjacent column preference',
                'good_direction': rate > 0.5,
                'threshold': 0.5
            },
            'chose_smaller_separation': {
                'variable': 'smaller row separation preference',
                'good_direction': rate > 0.5,
                'threshold': 0.5
            },
            'chose_column_4': {
                'variable': 'column 4 preference',
                'good_direction': rate != 0.5,  # Any preference is interesting
                'threshold': 0.5
            }
        }
        
        if outcome_var not in interpretations:
            return f"Frequency-weighted preference rate: {rate:.1%}"
        
        info = interpretations[outcome_var]
        direction = "supports" if info['good_direction'] else "contradicts"
        strength = "strong" if abs(rate - 0.5) > 0.15 else "moderate" if abs(rate - 0.5) > 0.08 else "weak"
        
        return f"{info['variable']}: {rate:.1%} frequency-weighted preference rate {direction} expectations ({strength} effect)"

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _all_keys_in_left_hand(self, bigram: str) -> bool:
        """Check if all keys in bigram are left-hand keys."""
        return all(key in self.left_hand_keys for key in bigram)

    # =========================================================================
    # VALIDATION FRAMEWORK
    # =========================================================================
        
    def _correct_multiple_comparisons(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Option C multiple testing correction:
        - FDR correction within key preferences only
        - Independent testing for other objectives (α = 0.05 each)
        """
        correction_summary = {
            'approach': 'Option C - Independent objectives with within-key-preference correction',
            'key_preference_correction': {},
            'independent_objectives': {},
            'total_significant_tests': 0
        }
        
        # Handle key preferences with FDR correction within domain
        if 'key_preference' in results and 'pairwise_results' in results['key_preference']:
            key_p_values = []
            key_test_names = []
            
            for key_pair, pair_results in results['key_preference']['pairwise_results'].items():
                if isinstance(pair_results, dict) and 'p_value' in pair_results:
                    p_val = pair_results['p_value']
                    if isinstance(p_val, (int, float)) and not pd.isna(p_val):
                        key_p_values.append(p_val)
                        key_test_names.append(f"key_preference_{key_pair}")
            
            if key_p_values:
                try:
                    # Apply FDR correction ONLY within key preferences
                    rejected, fdr_corrected, _, _ = multipletests(key_p_values, method='fdr_bh', alpha=0.05)
                    
                    key_correction_results = {}
                    for i, test_name in enumerate(key_test_names):
                        key_correction_results[test_name] = {
                            'original_p': key_p_values[i],
                            'fdr_corrected_p': fdr_corrected[i],
                            'significant_after_correction': rejected[i]
                        }
                    
                    correction_summary['key_preference_correction'] = {
                        'method': 'FDR_within_key_preferences_only',
                        'total_key_tests': len(key_p_values),
                        'original_significant': sum(1 for p in key_p_values if p < 0.05),
                        'fdr_significant': sum(rejected),
                        'corrected_tests': key_correction_results
                    }
                    correction_summary['total_significant_tests'] += sum(rejected)
                    
                except Exception as e:
                    correction_summary['key_preference_correction']['error'] = str(e)
        
        # Handle other objectives as independent tests (no correction between them)
        independent_objectives = ['row_separation', 'column_separation', 'column_4_vs_5']
        
        for obj_name in independent_objectives:
            if obj_name in results and 'error' not in results[obj_name]:
                
                # Extract main p-value for this objective
                p_val = None
                if 'simple_test' in results[obj_name] and isinstance(results[obj_name]['simple_test'], dict):
                    p_val = results[obj_name]['simple_test'].get('p_value')
                elif 'p_value' in results[obj_name]:
                    p_val = results[obj_name]['p_value']
                
                if p_val is not None and isinstance(p_val, (int, float)) and not pd.isna(p_val):
                    significant = p_val < 0.05  # Independent test, full α = 0.05
                    
                    correction_summary['independent_objectives'][obj_name] = {
                        'method': 'Independent test (no correction)',
                        'p_value': p_val,
                        'alpha_used': 0.05,
                        'significant': significant,
                        'interpretation': f"{obj_name}: p={p_val:.2e}, {'significant' if significant else 'not significant'} at α=0.05"
                    }
                    
                    if significant:
                        correction_summary['total_significant_tests'] += 1
        
        # Overall summary
        correction_summary['overall_summary'] = {
            'total_significant_across_all_objectives': correction_summary['total_significant_tests'],
            'key_preference_fdr_significant': correction_summary['key_preference_correction'].get('fdr_significant', 0),
            'independent_objectives_significant': sum(1 for obj in correction_summary['independent_objectives'].values() if obj['significant']),
            'recommendation': 'Use FDR-corrected key preferences + uncorrected objective results for MOO'
        }
        
        return correction_summary
                                        
    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def _generate_summary(self, results: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary with Option C correction info and all expected fields."""
        
        mc_data = validation.get('multiple_comparisons_correction', {})
        
        # Count significant results under Option C
        key_pref_significant = mc_data.get('key_preference_correction', {}).get('fdr_significant', 0)
        independent_significant = mc_data.get('overall_summary', {}).get('independent_objectives_significant', 0)
        total_significant = mc_data.get('total_significant_tests', key_pref_significant + independent_significant)
        
        # Calculate total tests performed
        key_pref_tests = mc_data.get('key_preference_correction', {}).get('total_key_tests', 0)
        independent_tests = len([obj for obj in ['row_separation', 'column_separation', 'column_4_vs_5'] 
                            if obj in results and 'error' not in results[obj]])
        total_tests = key_pref_tests + independent_tests
        
        # Calculate overall validity assessment
        objectives_extracted = len([r for r in results.values() if 'error' not in r])
        extraction_rate = objectives_extracted / 4  # 4 total possible objectives
        significance_rate = total_significant / total_tests if total_tests > 0 else 0
        
        if extraction_rate >= 0.75 and significance_rate >= 0.5:
            overall_validity = 'high'
        elif extraction_rate >= 0.5 and significance_rate >= 0.3:
            overall_validity = 'medium'  
        else:
            overall_validity = 'low'
        
        # Calculate frequency bias impact summary
        frequency_bias_impact = {
            'high_bias_objectives': 0,  # Count objectives with >5% bias
            'overall_severity': 'low'   # Default to low
        }
        
        # Check each objective for frequency bias
        for obj_name, obj_results in results.items():
            if 'error' not in obj_results and 'frequency_bias_analysis' in obj_results:
                bias_magnitude = obj_results['frequency_bias_analysis'].get('bias_magnitude', 0)
                if bias_magnitude > 0.05:  # >5% bias threshold
                    frequency_bias_impact['high_bias_objectives'] += 1
        
        if frequency_bias_impact['high_bias_objectives'] > 1:
            frequency_bias_impact['overall_severity'] = 'high'
        elif frequency_bias_impact['high_bias_objectives'] > 0:
            frequency_bias_impact['overall_severity'] = 'moderate'

        return {
            'objectives_extracted': objectives_extracted,
            'objectives_significant': total_significant,
            'fdr_corrected_significant': total_significant,
            'total_tests_performed': total_tests,
            'overall_validity': overall_validity,
            'strategic_key_subset_used': False,
            'frequency_bias_impact': frequency_bias_impact,  # ADD THIS - frequency bias assessment
            'correction_approach': 'Option C - Independent objectives',
            'key_preference_fdr_significant': key_pref_significant,
            'independent_objectives_significant': independent_significant,
            'total_significant_tests': total_significant,
            'key_preferences': results.get('key_preference', {}).get('key_preference_rates', False),
            'recommendation': 'Use FDR-corrected key preferences + uncorrected movement objectives for MOO'
        }

    def _generate_comprehensive_report(self, enhanced_results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive text report with detailed test descriptions, frequency bias analysis, and practical implications."""
        
        results = enhanced_results['objectives']
        validation = enhanced_results['validation']
        summary = enhanced_results['summary']
        
        # Safe access to validation data with defaults
        mc_data = validation.get('multiple_comparisons_correction', {})
        total_tests = mc_data.get('total_tests', len(results))
        original_sig = mc_data.get('original_significant', 'unknown')
        fdr_sig = mc_data.get('fdr_significant', mc_data.get('fdr_bh_significant', 'unknown'))
        recommendation = mc_data.get('recommendation', 'Apply appropriate multiple comparisons correction')
        
        # Frequency bias data
        bias_data = validation.get('frequency_bias_impact', {})
        
        report_lines = [
            "COMPLETE MOO OBJECTIVES ANALYSIS REPORT",
            "=" * 50,
            "",
            "ANALYSIS SUMMARY:",
            "=" * 20,
            f"Objectives successfully extracted: {summary.get('objectives_extracted', 'unknown')}/4",
            f"Objectives with significant results: {summary.get('objectives_significant', 'unknown')}",
            f"FDR-corrected significant tests: {summary.get('fdr_corrected_significant', 'unknown')}/{total_tests}",
            f"Overall validity assessment: {summary.get('overall_validity', 'unknown')}",
            f"Strategic key subset used: {summary.get('strategic_key_subset_used', 'unknown')}",
            "",
            "FREQUENCY BIAS IMPACT ASSESSMENT:",
            "=" * 35,
            f"Objectives with high frequency bias: {bias_data.get('high_bias_objectives', 'unknown')}",
            f"Objectives with direction changes: {bias_data.get('direction_changed_objectives', 'unknown')}",
            f"Overall bias severity: {bias_data.get('overall_bias_severity', 'unknown')}",
            "",
            "MULTIPLE COMPARISONS CORRECTION:",
            f"Tests performed: {total_tests}",
            f"Originally significant: {original_sig}",
            f"FDR-corrected significant: {fdr_sig}",
            f"Recommendation: {recommendation}",
            "",
            "OBJECTIVES DETAILED RESULTS WITH FREQUENCY WEIGHTING:",
            "=" * 60,
            ""
        ]
        
        # Detailed descriptions for each objective
        objective_details = {
            'key_preference': {
                'title': 'INDIVIDUAL KEY PREFERENCES WITH HYBRID APPROACH',
                'what_compared': 'Hybrid approach: Direct comparisons from repeated bigrams (AA vs BB) plus inferred comparisons from shared-key bigrams (QA vs QW) for uncovered pairs',
                'method': 'Frequency-weighted hybrid analysis: Applied standardized frequency weighting to both direct repeated bigram comparisons and inferred shared-key comparisons, prioritizing direct evidence where available.',
                'frequency_control': 'Standardized inverse frequency weighting plus English bigram frequency controls',
                'practical_meaning': 'Ranks individual keys by typing preference using clean direct evidence where possible, with inferred evidence filling gaps'
            },
            'row_separation': {
                'title': 'ROW SEPARATION PREFERENCES WITH FREQUENCY WEIGHTING',
                'what_compared': 'Bigrams requiring different row movements: same row vs. one row apart vs. two rows apart',
                'method': 'Frequency-weighted instance analysis: Applied standardized frequency weighting to control for comparison frequency differences, then tested preference for smaller vs. larger row separations.',
                'frequency_control': 'Standardized inverse frequency weighting plus English bigram frequency controls',
                'practical_meaning': 'Measures typing effort preferences related to vertical finger movement, corrected for frequency bias'
            },
            'column_separation': { 
                'title': 'COLUMN SEPARATION PREFERENCES WITH FREQUENCY WEIGHTING',
                'what_compared': 'Different-finger bigrams with different horizontal spacing: adjacent columns vs separated columns, controlling for row context',
                'method': 'Frequency-weighted hierarchical analysis: Applied standardized frequency weighting, then analyzed by row context and specific separation distances.',
                'frequency_control': 'Standardized inverse frequency weighting plus English bigram frequency controls',
                'practical_meaning': 'Measures pure horizontal reach preferences independent of finger independence effects, corrected for frequency bias'
            },
            'column_4_vs_5': {
                'title': 'COLUMN 4 VS 5 PREFERENCE WITH FREQUENCY WEIGHTING',
                'what_compared': 'Pure column 4 bigrams (RFV) vs pure column 5 bigrams (TGB)',
                'method': 'Frequency-weighted comparison: Applied standardized frequency weighting to control for presentation frequency differences between column types.',
                'frequency_control': 'Standardized inverse frequency weighting plus English bigram frequency controls',
                'practical_meaning': 'Tests specific index finger column preference, important for layout optimization decisions'
            }
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
                    f"  {details.get('frequency_control', 'Standard frequency controls applied')}",
                    "",
                    "STATISTICAL RESULTS:",
                    f"  Status: SUCCESS",
                    f"  Method: {obj_results.get('method', 'frequency_weighted_analysis')}",
                    f"  Instances analyzed: {obj_results.get('n_instances', 'unknown')}",
                    f"  Users contributing: {obj_results.get('n_users', 'unknown')}",
                    ""
                ])
                
                # Add frequency bias analysis
                if 'frequency_bias_analysis' in obj_results:
                    bias_analysis = obj_results['frequency_bias_analysis']
                    unweighted = bias_analysis.get('unweighted_preference_rate', 'unknown')
                    weighted = bias_analysis.get('weighted_preference_rate', 'unknown')
                    bias_magnitude = bias_analysis.get('bias_magnitude', 'unknown')
                    
                    report_lines.extend([
                        "FREQUENCY BIAS ANALYSIS:",
                        f"  Unweighted preference rate: {unweighted:.1%}" if isinstance(unweighted, (int, float)) else f"  Unweighted rate: {unweighted}",
                        f"  Frequency-weighted rate: {weighted:.1%}" if isinstance(weighted, (int, float)) else f"  Weighted rate: {weighted}",
                        f"  Bias magnitude: {bias_magnitude:.1%}" if isinstance(bias_magnitude, (int, float)) else f"  Bias magnitude: {bias_magnitude}",
                        ""
                    ])
                
                # Add specific results based on objective type
                if 'simple_test' in obj_results and isinstance(obj_results['simple_test'], dict):
                    simple_test = obj_results['simple_test']
                    pref_rate = simple_test.get('weighted_preference_rate', simple_test.get('preference_rate', 'unknown'))
                    p_val = simple_test.get('p_value', 'unknown')
                    effect_size = simple_test.get('effect_size', 'unknown')
                    
                    report_lines.extend([
                        "STATISTICAL TEST RESULTS:",
                        f"  Frequency-weighted preference rate: {pref_rate:.1%}" if isinstance(pref_rate, (int, float)) else f"  Preference rate: {pref_rate}",
                        f"  P-value: {p_val:.2e}" if isinstance(p_val, (int, float)) and p_val > 0 else f"  P-value: {p_val}",
                        f"  Effect size (Cohen's h): {effect_size:.3f}" if isinstance(effect_size, (int, float)) else f"  Effect size: {effect_size}",
                        f"  Result: {obj_results.get('interpretation', 'No interpretation available')}",
                        ""
                    ])

                if obj_name == 'column_separation':
                    # Add detailed comparison type breakdown
                    if 'comparison_results' in obj_results:
                        report_lines.extend([
                            "DETAILED COMPARISON TYPE ANALYSIS:",
                            ""
                        ])
                        for comp_type, comp_data in obj_results['comparison_results'].items():
                            if comp_type == 'same_finger_vs_different_finger':
                                title = "SAME FINGER vs DIFFERENT FINGER"
                                rate_desc = "prefer different finger"
                                opposite_rate = f"prefer same finger: {(1.0 - comp_data['weighted_rate']):.1%}"
                            elif comp_type == 'adjacent_vs_remote_columns':
                                title = "ADJACENT vs REMOTE COLUMNS"  
                                rate_desc = "prefer adjacent columns"
                                opposite_rate = f"prefer remote columns: {(1.0 - comp_data['weighted_rate']):.1%}"
                            
                            report_lines.extend([
                                f"  {title}:",
                                f"    Instances: {comp_data['n_instances']}",
                                f"    {rate_desc.title()}: {comp_data['weighted_rate']:.1%}",
                                f"    {opposite_rate}",
                                f"    Frequency bias: {comp_data['frequency_bias']:.1%}",
                                f"    P-value: {comp_data['p_value']:.2e}",
                                f"    Interpretation: {comp_data['interpretation']}",
                                ""
                            ])
                    
                    # Add context analysis
                    if 'context_results' in obj_results:
                        report_lines.extend([
                            "ROW CONTEXT BREAKDOWN:",
                            "Shows how column preferences change by row separation",
                            ""
                        ])
                        for context_key, context_data in obj_results['context_results'].items():
                            report_lines.extend([
                                f"  {context_data['description'].upper()}:",
                                f"    Total instances: {context_data['total_instances']}",
                                ""
                            ])
                            
                            if 'same_vs_different' in context_data:
                                same_diff = context_data['same_vs_different']
                                report_lines.extend([
                                    f"    Same finger vs Different finger:",
                                    f"      Prefer different finger: {same_diff['prefer_different_finger_rate']:.1%}",
                                    f"      Prefer same finger: {same_diff['prefer_same_finger_rate']:.1%}",
                                    f"      P-value: {same_diff['p_value']:.2e}",
                                    ""
                                ])
                            
                            if 'adjacent_vs_remote' in context_data:
                                adj_rem = context_data['adjacent_vs_remote']
                                report_lines.extend([
                                    f"    Adjacent vs Remote columns:",
                                    f"      Prefer adjacent: {adj_rem['prefer_adjacent_rate']:.1%}",
                                    f"      Prefer remote: {adj_rem['prefer_remote_rate']:.1%}",
                                    f"      P-value: {adj_rem['p_value']:.2e}",
                                    ""
                                ])
                                
                            report_lines.append("")
                            
                
                # Add key preference specific details
                if obj_name == 'key_preference':
                    n_pairs = obj_results.get('n_key_pairs', 0)
                    n_sig = obj_results.get('n_significant_pairs', 0)
                    strategic_used = obj_results.get('strategic_subset_used', False)
                    
                    report_lines.extend([
                        "KEY PREFERENCE DETAILS:",
                        f"  Strategic subset used: {strategic_used}",
                        f"  Total key pairs tested: {n_pairs}",
                        f"  Significant pairs (before correction): {n_sig}/{n_pairs}",
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
        
        # Add validation summary
        report_lines.extend([
            "",
            "VALIDATION AND STATISTICAL ROBUSTNESS:",
            "=" * 40,
            ""
        ])
        
        # Multiple comparisons with detailed breakdown
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
        
        # Detailed FDR-significant tests
        if 'detailed_significant_tests' in mc_data:
            sig_tests = mc_data['detailed_significant_tests']
            if sig_tests:
                report_lines.extend([
                    "Tests Remaining Significant After FDR Correction:",
                    "-" * 50,
                    ""
                ])
                for test_name, test_data in list(sig_tests.items())[:10]:  # Show first 10
                    obj_name = test_data['objective']
                    fdr_p = test_data['fdr_corrected_p']
                    report_lines.append(f"{test_name}: p={fdr_p:.2e} (objective: {obj_name})")
                if len(sig_tests) > 10:
                    report_lines.append(f"... and {len(sig_tests) - 10} more significant tests")
                report_lines.append("")
        
        # Frequency bias impact summary
        if bias_data:
            report_lines.extend([
                "Frequency Bias Impact Summary:",
                "-" * 30,
                ""
            ])
            individual_bias = bias_data.get('individual_analysis', {})
            for obj_name, bias_info in individual_bias.items():
                unweighted = bias_info.get('unweighted_rate', 0)
                weighted = bias_info.get('weighted_rate', 0)
                impact = bias_info.get('practical_impact', 'unknown')
                report_lines.append(f"{obj_name}: {unweighted:.1%} → {weighted:.1%} ({impact})")
            report_lines.append("")
        
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
                    if isinstance(mean_effect, (int, float)):
                        report_lines.append(f"{obj_name}: {interpretation} (mean={mean_effect:.1%}, {practical_sig})")
                    else:
                        report_lines.append(f"{obj_name}: {interpretation} ({practical_sig})")
            report_lines.append("")
        
        # Save report
        report_path = os.path.join(output_folder, 'complete_moo_objectives_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Comprehensive report saved to {report_path}")

    def _save_results(self, enhanced_results: Dict[str, Any], output_folder: str) -> None:
        """Save results including frequency bias analysis and corrected significance."""
        
        # Save summary with frequency bias info
        summary_data = []
        for obj_name, obj_results in enhanced_results['objectives'].items():
            if 'error' not in obj_results:
                
                # Get frequency bias info
                bias_analysis = obj_results.get('frequency_bias_analysis', {})
                unweighted_rate = bias_analysis.get('unweighted_preference_rate', 'N/A')
                weighted_rate = bias_analysis.get('weighted_preference_rate', 'N/A')
                bias_magnitude = bias_analysis.get('bias_magnitude', 'N/A')
                
                summary_data.append({
                    'Objective': obj_name,
                    'Description': obj_results['description'],
                    'Status': 'Success',
                    'Method': obj_results.get('method', 'unknown'),
                    'Instances': obj_results.get('n_instances', 'unknown'),
                    'Users': obj_results.get('n_users', 'unknown'),
                    'Unweighted_Rate': f"{unweighted_rate:.1%}" if isinstance(unweighted_rate, (int, float)) else str(unweighted_rate),
                    'Weighted_Rate': f"{weighted_rate:.1%}" if isinstance(weighted_rate, (int, float)) else str(weighted_rate),
                    'Frequency_Bias': f"{bias_magnitude:.1%}" if isinstance(bias_magnitude, (int, float)) else str(bias_magnitude),
                    'P_Value': obj_results.get('p_value', 'N/A'),
                    'Normalization_Range': str(obj_results['normalization_range']),
                    'Interpretation': obj_results.get('interpretation', 'No interpretation')
                })
            else:
                summary_data.append({
                    'Objective': obj_name,
                    'Description': obj_results.get('description', 'Unknown'),
                    'Status': 'Failed',
                    'Method': 'N/A',
                    'Instances': 'N/A',
                    'Users': 'N/A',
                    'Unweighted_Rate': 'N/A',
                    'Weighted_Rate': 'N/A',
                    'Frequency_Bias': 'N/A',
                    'P_Value': 'N/A',
                    'Error': obj_results['error'],
                    'Normalization_Range': 'N/A',
                    'Interpretation': 'Analysis failed'
                })
                
        # Save detailed multiple comparisons results
        mc_data = enhanced_results['validation'].get('multiple_comparisons_correction', {})
        if 'detailed_significant_tests' in mc_data and mc_data['detailed_significant_tests']:
            sig_tests_data = []
            for test_name, test_info in mc_data['detailed_significant_tests'].items():
                
                # Parse the test name for key preferences
                if test_info['objective'] == 'key_preference' and '_beats_' in test_name:
                    # Extract info from test name: key_preference_f_beats_a_direct_repeated_bigram_strong
                    parts = test_name.split('_')
                    if len(parts) >= 6:
                        winner = parts[2]
                        loser = parts[4] 
                        method = '_'.join(parts[5:-1])  # Everything between loser and strength
                        strength = parts[-1]
                        
                        sig_tests_data.append({
                            'Test_Name': test_name,
                            'Objective': test_info['objective'],
                            'Winner': winner.upper(),
                            'Loser': loser.upper(),
                            'Evidence_Type': method.replace('_', ' ').title(),
                            'Preference_Strength': strength.title(),
                            'Original_P_Value': test_info['original_p'],
                            'FDR_Corrected_P_Value': test_info['fdr_corrected_p'],
                            'Interpretation': f"{winner.upper()} preferred over {loser.upper()} ({method.replace('_', ' ')}, {strength} preference)"
                        })
                else:
                    # Non-key-preference tests or old format
                    sig_tests_data.append({
                        'Test_Name': test_name,
                        'Objective': test_info['objective'],
                        'Winner': 'N/A',
                        'Loser': 'N/A', 
                        'Evidence_Type': 'N/A',
                        'Preference_Strength': 'N/A',
                        'Original_P_Value': test_info['original_p'],
                        'FDR_Corrected_P_Value': test_info['fdr_corrected_p'],
                        'Interpretation': 'Non-key-preference test'
                    })
            
            sig_tests_df = pd.DataFrame(sig_tests_data)
            sig_tests_path = os.path.join(output_folder, 'key_preference_samekey_bigrams_fdr_corrected_significant.csv')
            sig_tests_df.to_csv(sig_tests_path, index=False)
            logger.info(f"FDR-corrected significant tests saved to {sig_tests_path}")
        
        # Save frequency bias impact analysis
        bias_data = enhanced_results['validation'].get('frequency_bias_impact', {})
        if 'individual_analysis' in bias_data:
            bias_analysis_data = []
            for obj_name, bias_info in bias_data['individual_analysis'].items():
                bias_analysis_data.append({
                    'Objective': obj_name,
                    'Unweighted_Rate': bias_info.get('unweighted_rate', 'N/A'),
                    'Weighted_Rate': bias_info.get('weighted_rate', 'N/A'),
                    'Bias_Magnitude': bias_info.get('bias_magnitude', 'N/A'),
                    'Direction_Changed': bias_info.get('direction_changed', 'N/A'),
                    'Frequency_Ratio': bias_info.get('frequency_ratio', 'N/A'),
                    'Bias_Severity': bias_info.get('bias_severity', 'N/A'),
                    'Practical_Impact': bias_info.get('practical_impact', 'N/A')
                })
            
            bias_df = pd.DataFrame(bias_analysis_data)
            bias_path = os.path.join(output_folder, 'frequency_bias_impact_analysis.csv')
            bias_df.to_csv(bias_path, index=False)
            logger.info(f"Frequency bias impact analysis saved to {bias_path}")
        
        # Key preference export with frequency bias info
        if 'key_preference' in enhanced_results['objectives']:
            self._save_key_preference_summary(
                enhanced_results['objectives']['key_preference'], 
                output_folder
            )

    def _save_key_preference_summary(self, key_pref_results: Dict[str, Any], output_folder: str) -> None:
        """Save clear summary of key preference results."""
        if 'pairwise_results' not in key_pref_results:
            return
        
        summary_data = []
        for key_pair, results in key_pref_results['pairwise_results'].items():
            winner = results.get('winner', 'unknown')
            loser = results['key2'] if winner == results['key1'] else results['key1']
            method = results.get('primary_method', 'unknown')
            
            # Evidence type description
            if method == 'direct':
                evidence_desc = f"Direct repeated bigram evidence ({winner.upper()}{winner.upper()} vs {loser.upper()}{loser.upper()})"
            elif method == 'inferred':
                evidence_desc = f"Inferred from shared-key bigrams"
            else:
                evidence_desc = "Unknown evidence type"
            
            summary_data.append({
                'Winner': winner.upper(),
                'Loser': loser.upper(),
                'Winner_Preference_Rate': f"{results.get('winner_rate', 0.5):.1%}",
                'Evidence_Type': method.title(),
                'Evidence_Description': evidence_desc,
                'Direct_Comparisons': results.get('direct_comparisons', 0),
                'Inferred_Comparisons': results.get('inferred_comparisons', 0),
                'Total_Comparisons': results.get('n_comparisons', 0),
                'P_Value': results.get('p_value', 1.0),
                'Significant': results.get('significant', False),
                'Frequency_Bias': f"{results.get('frequency_bias', 0):.1%}",
                'Interpretation': f"{winner.upper()} preferred over {loser.upper()} {results.get('winner_rate', 0.5):.1%} of time ({method} evidence)"
            })
        
        # Sort by winner preference rate (strongest preferences first)
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(['Significant', 'Winner_Preference_Rate'], ascending=[False, False])
        
        summary_path = os.path.join(output_folder, 'key_preferences.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Clear key preference summary saved to {summary_path}")

def generate_target_pairs(strategy: str, keys: List[str] = None) -> Set[Tuple[str, str]]:
    """
    Generate sets of key pairs for testing based on different strategies.
    
    Args:
        strategy: 'all' or 'selected'
        keys: List of keys to consider (default: all left-hand keys)
    
    Returns:
        Set of (key1, key2) tuples
    """
    from itertools import combinations
    
    if keys is None:
        keys = ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']
    
    if strategy == 'all':
        return set(combinations(keys, 2))
    
    elif strategy == 'selected':
        minimal_pairs = {
            ('f','d'), ('d','s'), ('s','a'), # Home row
            ('r','e'), ('w','q'), # Top row
            ('c','x'), ('x','z'), ('c','z'), # Bottom row
            ('f','r'), ('d','e'), ('s','w'), ('a','q'), # Reach up
            ('d','r'), ('s','r'), ('s','e'), ('a','e'), ('a','w'), ('f','e'), # Reach up-angle
            ('f','v'), # Reach down
            ('d','v'), ('s','v'), ('a','c'), # Reach down-angle
            ('r','v'), ('w','x'), ('q','z'), ('e','v'), ('w','c'), ('q','c'), ('q','x'), ('z','w'), # Hurdle
        }
        return {pair for pair in minimal_pairs if pair[0] in keys and pair[1] in keys}
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Complete MOO objectives extraction and validation with frequency weighting'
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
        target_pairs = generate_target_pairs('selected')
        analyzer = CompleteMOOObjectiveAnalyzer(target_key_pairs=target_pairs)
        results = analyzer.analyze_moo_objectives(args.data, args.output)
        
        logger.info("Complete MOO objectives analysis finished successfully!")
        logger.info(f"Results saved to: {args.output}")
        
        # Print quick summary
        summary = results['summary']
        
        print(f"\nQUICK SUMMARY:")
        print(f"==============")
        print(f"Objectives extracted: {summary['objectives_extracted']}/4")
        print(f"Significant objectives: {summary['objectives_significant']}")
        print(f"FDR-corrected significant tests: {summary['fdr_corrected_significant']}/{summary['total_tests_performed']}")
        print(f"Overall validity: {summary['overall_validity']}")
        print(f"Strategic key subset used: {summary['strategic_key_subset_used']}")
        print(f"High frequency bias objectives: {summary['frequency_bias_impact']['high_bias_objectives']}")
        print(f"Frequency bias severity: {summary['frequency_bias_impact']['overall_severity']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())