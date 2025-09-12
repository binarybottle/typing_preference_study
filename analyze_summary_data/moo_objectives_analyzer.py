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
import matplotlib.pyplot as plt

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
        
        # Store output folder for methods that need it
        self._current_output_folder = output_folder  # ADD THIS LINE
        logger.info(f"Output folder set to: {output_folder}")  # ADD THIS LINE
        
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
        """Apply standardized frequency weighting with better debugging."""
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
        
        logger.info(f"Applying frequency weighting method: {method}")
        logger.info(f"Comparison frequency range: {freq_counts.min()} - {freq_counts.max()}")
        
        if method == 'inverse_frequency':
            instances_df['frequency_weight'] = instances_df['comparison'].map(lambda x: 1.0 / freq_counts[x])
        elif method == 'max_frequency_ratio':
            max_frequency = freq_counts.max()
            instances_df['frequency_weight'] = instances_df['comparison'].map(lambda x: max_frequency / freq_counts[x])
        else:
            instances_df['frequency_weight'] = 1.0
        
        # Store frequency statistics and DEBUG INFO
        instances_df['comparison_frequency'] = instances_df['comparison'].map(freq_counts)
        
        # DEBUG: Show weight distribution
        weight_range = instances_df['frequency_weight'].max() - instances_df['frequency_weight'].min()
        logger.info(f"Applied frequency weights range: {instances_df['frequency_weight'].min():.3f} - {instances_df['frequency_weight'].max():.3f}")
        logger.info(f"Weight variation: {weight_range:.3f}")
        
        if weight_range < 0.001:
            logger.warning("⚠️  Frequency weights have very little variation - weighting will have minimal effect")
        
        return instances_df

    def diagnose_frequency_control(self, direct_df: pd.DataFrame) -> None:
        """Diagnose whether frequency control is working properly - FIXED VERSION."""
        
        logger.info("=== FREQUENCY CONTROL DIAGNOSIS ===")
        
        # Apply frequency weights to get the weighted version for diagnosis
        weighted_df = self.apply_frequency_weights(direct_df.copy())  # FIXED: Apply weighting first
        
        # Check comparison frequency distribution
        comparison_counts = weighted_df.apply(
            lambda row: tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']])), axis=1
        ).value_counts()
        
        logger.info(f"Unique comparison types: {len(comparison_counts)}")
        logger.info(f"Comparison frequency range: {comparison_counts.min()} - {comparison_counts.max()}")
        logger.info(f"Frequency ratio (max/min): {comparison_counts.max() / comparison_counts.min():.1f}")
        
        # Show most and least frequent comparisons
        logger.info("\nMost frequent comparisons:")
        for comp, count in comparison_counts.head(5).items():
            logger.info(f"  {comp}: {count} instances")
        
        logger.info("\nLeast frequent comparisons:")
        for comp, count in comparison_counts.tail(5).items():
            logger.info(f"  {comp}: {count} instances")
        
        # Check English bigram frequencies
        english_freq_available = len(self.english_bigram_frequencies) > 0
        logger.info(f"\nEnglish bigram frequencies loaded: {english_freq_available}")
        
        if english_freq_available:
            # Check frequency distribution of bigrams in our dataset
            chosen_freqs = [self.english_bigram_frequencies.get(bigram, 0) for bigram in weighted_df['chosen_bigram']]
            unchosen_freqs = [self.english_bigram_frequencies.get(bigram, 0) for bigram in weighted_df['unchosen_bigram']]
            
            logger.info(f"English freq range for chosen bigrams: {min(chosen_freqs):.2e} - {max(chosen_freqs):.2e}")
            logger.info(f"English freq range for unchosen bigrams: {min(unchosen_freqs):.2e} - {max(unchosen_freqs):.2e}")
            
            # Check for frequency bias in specific comparisons
            bias_examples = []
            for _, row in weighted_df.head(10).iterrows():
                chosen_freq = self.english_bigram_frequencies.get(row['chosen_bigram'], 0)
                unchosen_freq = self.english_bigram_frequencies.get(row['unchosen_bigram'], 0)
                if chosen_freq > 0 and unchosen_freq > 0:
                    freq_ratio = chosen_freq / unchosen_freq
                    if freq_ratio > 2 or freq_ratio < 0.5:  # Significant frequency difference
                        bias_examples.append(f"{row['chosen_bigram']} vs {row['unchosen_bigram']}: {freq_ratio:.1f}x freq ratio")
            
            if bias_examples:
                logger.info("Examples of frequency-biased comparisons:")
                for example in bias_examples[:3]:
                    logger.info(f"  {example}")
            
        # Check if frequency weighting is actually being applied - FIXED
        if 'frequency_weight' in weighted_df.columns:
            weight_range = weighted_df['frequency_weight'].max() - weighted_df['frequency_weight'].min()
            logger.info(f"\nFrequency weight range: {weighted_df['frequency_weight'].min():.3f} - {weighted_df['frequency_weight'].max():.3f}")
            logger.info(f"Weight variation: {weight_range:.3f}")
            
            if weight_range < 0.001:
                logger.warning("Frequency weights are nearly identical - weighting has no effect")
            else:
                logger.info("✅ Frequency weighting is working properly")
                
            # Show how weights affect outcomes
            high_weight_rows = weighted_df[weighted_df['frequency_weight'] > weighted_df['frequency_weight'].median()]
            low_weight_rows = weighted_df[weighted_df['frequency_weight'] <= weighted_df['frequency_weight'].median()]
            
            logger.info(f"High-weight comparisons (rare): {len(high_weight_rows)} instances, {high_weight_rows['chose_chosen_key'].mean():.1%} preference rate")
            logger.info(f"Low-weight comparisons (common): {len(low_weight_rows)} instances, {low_weight_rows['chose_chosen_key'].mean():.1%} preference rate")
            
        else:
            logger.error("❌ No frequency_weight column found - weighting not applied")

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
        """Test key preferences with both direct-only and direct+inferred approaches."""
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
                        'chosen_bigram': chosen,
                        'unchosen_bigram': unchosen,
                        'chosen_key': chosen_key,
                        'unchosen_key': unchosen_key,
                        'chose_chosen_key': 1,
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
        
        # Apply frequency weighting
        weighted_metrics = self.calculate_weighted_preference_metrics(direct_df, 'chose_chosen_key')
        weighted_data = weighted_metrics['weighted_data']
        
        # APPROACH 1: DIRECT EVIDENCE ONLY
        logger.info("=== APPROACH 1: DIRECT EVIDENCE ONLY ===")
        
        # Analyze direct pairwise results
        direct_pairwise_results = {}
        direct_covered_pairs = set()
        all_p_values = []
        
        for key_pair in weighted_data['key_pair'].unique():
            pair_data = weighted_data[weighted_data['key_pair'] == key_pair].copy()
            
            if len(pair_data) < 5:  # Minimum threshold
                continue
            
            key1, key2 = key_pair
            direct_covered_pairs.add(key_pair)
            
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
            
            # Wilson confidence interval
            effective_n = (total_weight ** 2) / (pair_data['frequency_weight'] ** 2).sum()
            p_hat = weighted_win_rate
            n = effective_n
            
            if n > 0:
                alpha = 0.05
                z = norm.ppf(1 - alpha/2)
                
                denominator = 1 + z**2 / n
                center = (p_hat + z**2 / (2*n)) / denominator
                margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
                
                ci_lower = max(0, center - margin)
                ci_upper = min(1, center + margin)
            else:
                ci_lower = ci_upper = weighted_win_rate
            
            # Statistical test
            if effective_n > 0:
                z_score = (weighted_win_rate - 0.5) / np.sqrt(0.25 / effective_n)
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
            else:
                p_value = 1.0
            
            # Determine winner
            winner = key1 if weighted_win_rate > 0.5 else key2
            winner_rate = max(weighted_win_rate, 1 - weighted_win_rate)
            effect_size = abs(weighted_win_rate - 0.5)
            frequency_bias = abs(weighted_win_rate - win_rate)
            
            direct_pairwise_results[key_pair] = {
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
                'primary_method': 'direct',
                'interpretation': f"{winner.upper()} > {key2.upper() if winner == key1 else key1.upper()}: {winner_rate:.1%} preference (direct evidence)"
            }
            
            all_p_values.append(p_value)
        
        logger.info(f"Direct evidence covers {len(direct_pairwise_results)} key pairs")
        
        # APPROACH 2: DIRECT + INFERRED EVIDENCE  
        logger.info("=== APPROACH 2: DIRECT + INFERRED EVIDENCE ===")
        
        # Extract inferred preferences for missing pairs
        inferred_df = self._extract_inferred_key_preferences_for_missing_pairs(direct_covered_pairs)
        inferred_analysis = self._analyze_inferred_preferences(inferred_df)
        
        # Combine direct and inferred results
        combined_pairwise_results = direct_pairwise_results.copy()
        if 'inferred_results' in inferred_analysis:
            for key_pair, inferred_result in inferred_analysis['inferred_results'].items():
                if key_pair not in combined_pairwise_results:  # Don't override direct evidence
                    combined_pairwise_results[key_pair] = inferred_result
                    all_p_values.append(inferred_result['p_value'])
        
        logger.info(f"Combined evidence covers {len(combined_pairwise_results)} key pairs")
        logger.info(f"Added {len(combined_pairwise_results) - len(direct_pairwise_results)} inferred pairs")
        
        # Generate outputs for both approaches
        output_folder = getattr(self, '_current_output_folder', None)
        
        if len(direct_pairwise_results) > 0 and output_folder:
            try:
                # APPROACH 1 OUTPUTS: Direct evidence only
                logger.info("Generating direct-evidence-only outputs...")
                
                # Calculate key scores from direct evidence only
                direct_key_scores = self._calculate_individual_key_scores_with_ci(direct_pairwise_results)
                direct_tiers = self._create_practical_moo_tiers(direct_key_scores)
                
                # Export direct-only results
                direct_constraints_df = self._export_pairwise_moo_constraints(direct_pairwise_results, output_folder)
                
                # Save direct-only tier data
                direct_tier_data = []
                for tier_num in sorted(direct_tiers.keys()):
                    for key in direct_tiers[tier_num]:
                        key_row = direct_key_scores[direct_key_scores['key'] == key].iloc[0]
                        direct_tier_data.append({
                            'approach': 'direct_only',
                            'tier': tier_num,
                            'key': key,
                            'win_rate': key_row['win_rate'],
                            'ci_lower': key_row['ci_lower'],
                            'ci_upper': key_row['ci_upper'],
                            'moo_weight': 1.0 / tier_num,
                            'evidence_type': 'direct_AA_vs_BB'
                        })
                
                direct_tier_df = pd.DataFrame(direct_tier_data)
                direct_tier_df.to_csv(os.path.join(output_folder, 'moo_tiers_direct_only.csv'), index=False)
                
                # APPROACH 2 OUTPUTS: Direct + inferred evidence
                if len(combined_pairwise_results) > len(direct_pairwise_results):
                    logger.info("Generating direct+inferred outputs...")
                    
                    # Calculate key scores from combined evidence
                    combined_key_scores = self._calculate_individual_key_scores_with_ci(combined_pairwise_results)
                    combined_tiers = self._create_practical_moo_tiers(combined_key_scores)
                    
                    # Export combined results  
                    combined_constraints_df = self._export_pairwise_moo_constraints(combined_pairwise_results, output_folder)
                    
                    # Save combined tier data
                    combined_tier_data = []
                    for tier_num in sorted(combined_tiers.keys()):
                        for key in combined_tiers[tier_num]:
                            key_row = combined_key_scores[combined_key_scores['key'] == key].iloc[0]
                            combined_tier_data.append({
                                'approach': 'direct_plus_inferred',
                                'tier': tier_num,
                                'key': key,
                                'win_rate': key_row['win_rate'],
                                'ci_lower': key_row['ci_lower'],
                                'ci_upper': key_row['ci_upper'],
                                'moo_weight': 1.0 / tier_num,
                                'evidence_type': 'mixed'
                            })
                    
                    combined_tier_df = pd.DataFrame(combined_tier_data)
                    combined_tier_df.to_csv(os.path.join(output_folder, 'moo_tiers_direct_plus_inferred.csv'), index=False)
                    
                    # Create comparison analysis
                    self._compare_approaches(direct_key_scores, combined_key_scores, 
                                        direct_tiers, combined_tiers, output_folder)
                else:
                    logger.info("No additional inferred evidence found - approaches are identical")
                    combined_key_scores = direct_key_scores
                    combined_tiers = direct_tiers
                
                # Generate complete preference matrix
                complete_matrix = self._create_complete_preference_matrix(
                    direct_pairwise_results, inferred_analysis
                )
                complete_matrix.to_csv(os.path.join(output_folder, 'complete_preference_matrix.csv'), index=False)
                
                # Create visualizations for primary approach (direct+inferred if available)
                primary_scores = combined_key_scores if len(combined_pairwise_results) > len(direct_pairwise_results) else direct_key_scores
                primary_tiers = combined_tiers if len(combined_pairwise_results) > len(direct_pairwise_results) else direct_tiers
                
                self._plot_key_preferences_with_tiers(primary_scores, primary_tiers, output_folder)
                
                # Export comprehensive MOO specification
                primary_constraints = combined_constraints_df if len(combined_pairwise_results) > len(direct_pairwise_results) else direct_constraints_df
                self._export_combined_moo_objectives(primary_scores, primary_tiers, primary_constraints, output_folder)
                
                # Run diagnostics
                self.diagnose_frequency_control(direct_df)
                self._diagnose_100_percent_rates(direct_pairwise_results)
                
                logger.info("Both approaches generated successfully")
                
            except Exception as e:
                logger.error(f"Key preference analysis failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                
                # Fallback to basic results
                primary_scores = direct_key_scores if 'direct_key_scores' in locals() else None
                primary_tiers = direct_tiers if 'direct_tiers' in locals() else None
        
        # Return comprehensive results
        return {
            'description': 'Key preferences with both direct and inferred evidence approaches',
            'method': 'direct_and_inferred_comparison',
            'direct_approach': {
                'n_instances': len(direct_df),
                'n_pairs': len(direct_pairwise_results),
                'pairwise_results': direct_pairwise_results,
                'coverage': f"{len(direct_pairwise_results)}/{len(list(self.left_hand_keys)) * (len(list(self.left_hand_keys)) - 1) // 2}"
            },
            'inferred_approach': {
                'n_inferred_instances': len(inferred_df) if not inferred_df.empty else 0,
                'n_additional_pairs': len(combined_pairwise_results) - len(direct_pairwise_results),
                'inferred_analysis': inferred_analysis,
                'total_coverage': f"{len(combined_pairwise_results)}/{len(list(self.left_hand_keys)) * (len(list(self.left_hand_keys)) - 1) // 2}"
            },
            'n_users': direct_df['user_id'].nunique(),
            'frequency_bias_analysis': weighted_metrics,
            'p_values_for_correction': all_p_values,
            'instances_data': direct_df,
            'normalization_range': (0.0, 1.0),
            'p_value': min(all_p_values) if all_p_values else 1.0,
            'preference_rate': weighted_metrics.get('weighted_preference_rate', 0.5),
            'interpretation': f"Direct evidence: {len(direct_pairwise_results)} pairs; Combined: {len(combined_pairwise_results)} pairs"
        }

    def _compare_approaches(self, direct_scores: pd.DataFrame, combined_scores: pd.DataFrame,
                        direct_tiers: Dict, combined_tiers: Dict, output_folder: str) -> None:
        """Compare direct-only vs direct+inferred approaches."""
        
        logger.info("=== COMPARING DIRECT VS DIRECT+INFERRED APPROACHES ===")
        
        comparison_data = []
        
        # Compare individual key scores
        all_keys = set(direct_scores['key']) | set(combined_scores['key'])
        
        for key in all_keys:
            direct_row = direct_scores[direct_scores['key'] == key]
            combined_row = combined_scores[combined_scores['key'] == key]
            
            direct_score = direct_row['win_rate'].iloc[0] if len(direct_row) > 0 else None
            combined_score = combined_row['win_rate'].iloc[0] if len(combined_row) > 0 else None
            
            # Find tiers
            direct_tier = None
            combined_tier = None
            
            for tier_num, tier_keys in direct_tiers.items():
                if key in tier_keys:
                    direct_tier = tier_num
                    break
            
            for tier_num, tier_keys in combined_tiers.items():
                if key in tier_keys:
                    combined_tier = tier_num
                    break
            
            score_change = combined_score - direct_score if (direct_score is not None and combined_score is not None) else 0
            tier_change = (combined_tier - direct_tier) if (direct_tier is not None and combined_tier is not None) else 0
            
            comparison_data.append({
                'key': key,
                'direct_score': direct_score,
                'combined_score': combined_score,
                'score_change': score_change,
                'direct_tier': direct_tier,
                'combined_tier': combined_tier,
                'tier_change': tier_change,
                'substantial_change': abs(score_change) > 0.1 or abs(tier_change) > 1
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(os.path.join(output_folder, 'approach_comparison.csv'), index=False)
        
        # Summary statistics
        substantial_changes = len(comparison_df[comparison_df['substantial_change'] == True])
        mean_score_change = comparison_df['score_change'].abs().mean()
        
        logger.info(f"Approach comparison: {substantial_changes}/{len(comparison_df)} keys changed substantially")
        logger.info(f"Mean absolute score change: {mean_score_change:.3f}")
        
        if substantial_changes > len(comparison_df) * 0.3:
            logger.warning("⚠️  Inferred evidence substantially changes >30% of key rankings")
        else:
            logger.info("✅ Approaches are largely consistent")
        
        # Log major changes
        major_changes = comparison_df[comparison_df['substantial_change'] == True].nlargest(5, 'score_change')
        if len(major_changes) > 0:
            logger.info("Largest changes from adding inferred evidence:")
            for _, row in major_changes.iterrows():
                logger.info(f"  {row['key']}: {row['direct_score']:.1%} → {row['combined_score']:.1%} (tier {row['direct_tier']} → {row['combined_tier']})")
        
        logger.info(f"Approach comparison saved to: {os.path.join(output_folder, 'approach_comparison.csv')}")
        
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

    def _calculate_individual_key_scores_with_ci(self, pairwise_results: Dict, 
                                            confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate key scores with more realistic confidence intervals."""
        
        # Collect all keys and their detailed performance
        all_keys = set()
        key_performances = {}  # key -> list of (win_rate, n_comparisons, opponent)
        
        for pair, results in pairwise_results.items():
            key1, key2 = results['key1'], results['key2']
            all_keys.update([key1, key2])
            
            # Initialize performance tracking
            if key1 not in key_performances:
                key_performances[key1] = []
            if key2 not in key_performances:
                key_performances[key2] = []
            
            # Get the actual win rate (use frequency bias to assess reliability)
            win_rate_key1 = results.get('weighted_key1_rate', results.get('unweighted_key1_rate', 0.5))
            n_comparisons = results['n_comparisons']
            
            # Store individual comparison results (not aggregated)
            key_performances[key1].append({
                'win_rate': win_rate_key1,
                'n_comparisons': n_comparisons,
                'opponent': key2,
                'reliability': 1.0 / (1.0 + results.get('frequency_bias', 0))
            })
            
            key_performances[key2].append({
                'win_rate': 1.0 - win_rate_key1,
                'n_comparisons': n_comparisons,
                'opponent': key1,
                'reliability': 1.0 / (1.0 + results.get('frequency_bias', 0))
            })
        
        # Calculate key scores with better confidence intervals
        key_data = {}
        
        for key in all_keys:
            performances = key_performances[key]
            
            if not performances:
                # No data for this key
                key_data[key] = {
                    'key': key.upper(),
                    'win_rate': 0.5,
                    'ci_lower': 0.0,
                    'ci_upper': 1.0,
                    'ci_width': 1.0,
                    'n_comparisons': 0,
                    'n_opponents': 0,
                    'finger': self.key_positions.get(key, KeyPosition('', 0, 0, 0)).finger,
                    'row': self.key_positions.get(key, KeyPosition('', 0, 0, 0)).row
                }
                continue
            
            # Calculate weighted average win rate
            total_weight = sum(p['reliability'] * p['n_comparisons'] for p in performances)
            if total_weight > 0:
                weighted_win_rate = sum(p['win_rate'] * p['reliability'] * p['n_comparisons'] for p in performances) / total_weight
            else:
                weighted_win_rate = 0.5
            
            # Total effective sample size
            total_n = sum(p['n_comparisons'] for p in performances)
            
            # Calculate confidence interval using Beta distribution (better for extreme rates)
            # Add small pseudocounts to prevent 0% and 100% rates
            alpha_pseudo = 1  # Prior "wins"
            beta_pseudo = 1   # Prior "losses"
            
            observed_wins = weighted_win_rate * total_n + alpha_pseudo
            observed_losses = (1 - weighted_win_rate) * total_n + beta_pseudo
            total_trials = total_n + alpha_pseudo + beta_pseudo
            
            # Adjusted win rate with pseudocounts
            adjusted_win_rate = observed_wins / total_trials
            
            # Wilson score interval with adjustment
            if total_n > 0:
                alpha = 1 - confidence_level
                z = norm.ppf(1 - alpha/2)
                
                p_hat = adjusted_win_rate
                n = total_trials
                
                denominator = 1 + z**2 / n
                center = (p_hat + z**2 / (2*n)) / denominator
                margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
                
                ci_lower = max(0, center - margin)
                ci_upper = min(1, center + margin)
            else:
                ci_lower = ci_upper = adjusted_win_rate
            
            # Get opponent info
            opponents = [p['opponent'] for p in performances]
            
            key_data[key] = {
                'key': key.upper(),
                'win_rate': adjusted_win_rate,  # Use adjusted rate to avoid 100%
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'n_comparisons': total_n,
                'n_opponents': len(set(opponents)),
                'opponents': sorted(set(opponents)),
                'finger': self.key_positions.get(key, KeyPosition('', 0, 0, 0)).finger,
                'row': self.key_positions.get(key, KeyPosition('', 0, 0, 0)).row,
                'raw_win_rate': weighted_win_rate,  # Store original for comparison
                'adjustment_applied': abs(weighted_win_rate - adjusted_win_rate) > 0.01
            }
        
        df = pd.DataFrame(key_data.values()).sort_values('win_rate', ascending=False)
        
        # Log adjustments made
        adjusted_keys = df[df['adjustment_applied'] == True]
        if len(adjusted_keys) > 0:
            logger.info(f"Applied statistical adjustments to {len(adjusted_keys)} keys to prevent extreme rates:")
            for _, row in adjusted_keys.iterrows():
                logger.info(f"  {row['key']}: {row['raw_win_rate']:.1%} → {row['win_rate']:.1%}")
        
        return df

    def _identify_statistical_tiers_with_transitivity(self, key_scores: pd.DataFrame) -> Dict[int, List[str]]:
        """Identify tiers with more granular groupings."""
        
        # Sort keys by win rate
        sorted_keys = key_scores.sort_values('win_rate', ascending=False)
        
        tiers = {}
        tier_num = 1
        used_keys = set()
        
        for _, current_key_row in sorted_keys.iterrows():
            if current_key_row['key'] in used_keys:
                continue
            
            # Start new tier
            tier_keys = [current_key_row['key']]
            tier_ci = (current_key_row['ci_lower'], current_key_row['ci_upper'])
            used_keys.add(current_key_row['key'])
            
            # Find keys with significantly overlapping CIs
            for _, other_key_row in sorted_keys.iterrows():
                if other_key_row['key'] in used_keys:
                    continue
                
                other_ci = (other_key_row['ci_lower'], other_key_row['ci_upper'])
                
                # Check for substantial overlap (not just touching)
                overlap_start = max(tier_ci[0], other_ci[0])
                overlap_end = min(tier_ci[1], other_ci[1])
                overlap_size = max(0, overlap_end - overlap_start)
                
                tier_ci_width = tier_ci[1] - tier_ci[0]
                other_ci_width = other_ci[1] - other_ci[0]
                min_ci_width = min(tier_ci_width, other_ci_width)
                
                # Require substantial overlap (>30% of smaller CI width)
                if overlap_size > 0.3 * min_ci_width:
                    tier_keys.append(other_key_row['key'])
                    used_keys.add(other_key_row['key'])
                    
                    # Update tier CI bounds
                    tier_ci = (
                        min(tier_ci[0], other_ci[0]),
                        max(tier_ci[1], other_ci[1])
                    )
            
            # Store tier
            tiers[tier_num] = sorted(tier_keys)
            
            # Log tier with win rate range
            tier_rows = key_scores[key_scores['key'].isin(tier_keys)]
            tier_mean = tier_rows['win_rate'].mean()
            tier_range = f"{tier_rows['win_rate'].min():.1%}-{tier_rows['win_rate'].max():.1%}"
            tier_keys_str = ', '.join(sorted(tier_keys))
            
            logger.info(f"Tier {tier_num}: {tier_keys_str} (win rates: {tier_range}, mean: {tier_mean:.1%})")
            
            tier_num += 1
        
        return tiers

    def _create_practical_moo_tiers(self, key_scores: pd.DataFrame) -> Dict[int, List[str]]:
        """
        Create practical MOO tiers using win rate thresholds instead of statistical overlap.
        This ensures meaningful differentiation for optimization.
        """
        
        # Sort keys by win rate
        sorted_keys = key_scores.sort_values('win_rate', ascending=False)
        
        # Define practical tier boundaries based on win rates
        tiers = {}
        tier_assignments = []
        
        for _, row in sorted_keys.iterrows():
            win_rate = row['win_rate']
            
            # Assign tiers based on win rate ranges
            if win_rate >= 0.85:
                tier = 1  # Excellent (85%+)
            elif win_rate >= 0.70:
                tier = 2  # Very Good (70-84%)
            elif win_rate >= 0.60:
                tier = 3  # Good (60-69%)
            elif win_rate >= 0.50:
                tier = 4  # Above Average (50-59%)
            elif win_rate >= 0.40:
                tier = 5  # Below Average (40-49%)
            elif win_rate >= 0.25:
                tier = 6  # Poor (25-39%)
            else:
                tier = 7  # Very Poor (<25%)
            
            tier_assignments.append((row['key'], tier, win_rate))
            
            # Add to tier dictionary
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append(row['key'])
        
        # Log practical tier assignments
        logger.info("PRACTICAL MOO TIERS (based on win rate ranges):")
        for tier_num in sorted(tiers.keys()):
            tier_keys = tiers[tier_num]
            tier_rows = key_scores[key_scores['key'].isin(tier_keys)]
            tier_min = tier_rows['win_rate'].min()
            tier_max = tier_rows['win_rate'].max()
            tier_mean = tier_rows['win_rate'].mean()
            
            if tier_min == tier_max:
                rate_str = f"{tier_mean:.1%}"
            else:
                rate_str = f"{tier_min:.1%}-{tier_max:.1%} (mean: {tier_mean:.1%})"
            
            tier_keys_str = ', '.join(sorted(tier_keys))
            logger.info(f"  Tier {tier_num}: {tier_keys_str} ({rate_str})")
        
        # Validate tier structure
        total_tiers = len(tiers)
        total_keys = len(key_scores)
        avg_keys_per_tier = total_keys / total_tiers
        
        logger.info(f"Tier structure: {total_tiers} tiers, {total_keys} keys, {avg_keys_per_tier:.1f} keys/tier average")
        
        # Check for reasonable distribution
        if total_tiers < 3:
            logger.warning("⚠️  Only {total_tiers} tiers created - may not provide enough differentiation for MOO")
        elif total_tiers > 8:
            logger.warning("⚠️  Many tiers ({total_tiers}) created - may be too granular for MOO")
        else:
            logger.info("✅ Good tier structure for MOO optimization")
        
        return tiers

    def _plot_key_preferences_with_tiers(self, key_scores: pd.DataFrame, 
                                    tiers: Dict[int, List[str]], 
                                    output_folder: str) -> None:
        """Create key preference plot with confidence intervals and tier coloring."""
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Sort keys by win rate (ascending for bottom-to-top plotting)
        key_scores_sorted = key_scores.sort_values('win_rate', ascending=True)
        
        # Define colors for tiers
        tier_colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#DC143C', '#8B008B']
        
        # Map keys to colors and tiers
        key_to_color = {}
        key_to_tier = {}
        for tier_num, tier_keys in tiers.items():
            color = tier_colors[(tier_num - 1) % len(tier_colors)]
            for key in tier_keys:
                key_to_color[key] = color
                key_to_tier[key] = tier_num
        
        # Plot
        y_pos = range(len(key_scores_sorted))
        
        for i, (_, row) in enumerate(key_scores_sorted.iterrows()):
            color = key_to_color.get(row['key'], '#666666')
            
            # Confidence interval
            ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
                color='black', linewidth=2, alpha=0.6)
            
            # Point
            ax.scatter(row['win_rate'], i, c=color, s=120, alpha=0.8, 
                    edgecolors='black', linewidth=1, zorder=3)
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['key']}" for _, row in key_scores_sorted.iterrows()])
        ax.set_xlabel('Win Rate (95% CI)', fontsize=12)
        ax.set_ylabel('Key', fontsize=12)
        ax.set_title('Key Preference Strengths with Confidence Intervals and MOO Tiers', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='No preference')
        
        # Add tier info on right side
        for i, (_, row) in enumerate(key_scores_sorted.iterrows()):
            tier_num = key_to_tier.get(row['key'], '?')
            finger_num = row.get('finger', '?')
            n_comp = row.get('n_comparisons', 0)
            
            ax.text(1.02, i, f"T{tier_num} F{finger_num} (n={n_comp})", 
                transform=ax.get_yaxis_transform(), va='center', fontsize=9)
        
        # Tier legend
        legend_elements = []
        for tier_num in sorted(tiers.keys()):
            color = tier_colors[(tier_num - 1) % len(tier_colors)]
            tier_keys_str = ', '.join(sorted(tiers[tier_num]))
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                        markersize=10, label=f"Tier {tier_num}: {tier_keys_str}")
            )
        
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        ax.set_xlim(0, 1)
        plt.tight_layout()
        
        # Save
        plot_path = os.path.join(output_folder, 'key_preferences_with_tiers.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Key preference plot saved to: {plot_path}")

    def _export_pairwise_moo_constraints(self, pairwise_results: Dict, output_folder: str) -> pd.DataFrame:
        """Export pairwise constraints for MOO algorithms that can handle multiple constraints."""
        
        constraints_data = []
        
        for pair, results in pairwise_results.items():
            if results.get('significant', False):  # Only export significant preferences
                
                winner = results['winner']
                loser = results['key2'] if winner == results['key1'] else results['key1']
                
                constraints_data.append({
                    'constraint_type': 'pairwise_key_preference',
                    'promote_key': winner.upper(),
                    'demote_key': loser.upper(),
                    'preference_strength': results['winner_rate'],
                    'confidence_lower': results.get('ci_lower', 0),
                    'confidence_upper': results.get('ci_upper', 1),
                    'n_comparisons': results['n_comparisons'],
                    'p_value': results['p_value'],
                    'evidence_type': results.get('primary_method', 'direct'),
                    'constraint_weight': results['winner_rate'] - 0.5,  # 0-0.5 range
                    'moo_formulation': f"promote_{winner.upper()}_over_{loser.upper()}",
                    'statistical_support': 'FDR_significant' if results['significant'] else 'not_significant',
                    'frequency_bias': results.get('frequency_bias', 0.0),
                    'interpretation': f"Promote {winner.upper()} over {loser.upper()} (strength: {results['winner_rate']:.1%})"
                })
        
        constraints_df = pd.DataFrame(constraints_data)
        
        if len(constraints_df) > 0:
            # Sort by preference strength (strongest first)
            constraints_df = constraints_df.sort_values('preference_strength', ascending=False)
            
            # Save constraints
            constraints_path = os.path.join(output_folder, 'moo_pairwise_constraints.csv')
            constraints_df.to_csv(constraints_path, index=False)
            
            logger.info(f"Pairwise MOO constraints saved to: {constraints_path}")
            logger.info(f"Exported {len(constraints_df)} significant pairwise constraints")
            
            # Log summary
            strong_constraints = len(constraints_df[constraints_df['preference_strength'] > 0.8])
            moderate_constraints = len(constraints_df[(constraints_df['preference_strength'] > 0.6) & 
                                                    (constraints_df['preference_strength'] <= 0.8)])
            weak_constraints = len(constraints_df[constraints_df['preference_strength'] <= 0.6])
            
            logger.info(f"Constraint strength distribution: {strong_constraints} strong (>80%), "
                    f"{moderate_constraints} moderate (60-80%), {weak_constraints} weak (≤60%)")
            
        return constraints_df

    def _export_combined_moo_objectives(self, key_scores: pd.DataFrame, tiers: Dict, 
                                    constraints_df: pd.DataFrame, output_folder: str) -> None:
        """Export combined MOO objective specification."""
        
        # Create comprehensive MOO specification
        moo_spec = {
            'objective_1_tiered_key_preferences': {
                'type': 'weighted_sum',
                'description': 'Maximize sum of (tier_weight × key_frequency) across layout',
                'weights': {row['key']: row['moo_weight'] for _, row in 
                        pd.read_csv(os.path.join(output_folder, 'moo_key_preference_tiers.csv')).iterrows()},
                'formulation': 'sum(tier_weight[key] * frequency[key] for key in layout)',
                'normalization': 'weights sum to different values by design (tier structure)',
                'interpretation': 'Higher score = more preferred keys in frequent positions'
            },
            
            'objective_2_pairwise_constraints': {
                'type': 'constraint_set',
                'description': 'Pairwise promotion/demotion constraints',
                'constraints': constraints_df.to_dict('records') if len(constraints_df) > 0 else [],
                'formulation': 'for each constraint: frequency[promote_key] > frequency[demote_key] * (1 + constraint_weight)',
                'interpretation': 'Each constraint promotes one key over another based on preference strength'
            },
            
            'objective_3_row_separation': {
                'type': 'preference_rate',
                'description': 'Row separation movement preferences',
                'preference_rate': 0.75,  # From your analysis
                'formulation': 'minimize weighted_row_movement_cost',
                'interpretation': '75% preference for smaller row separations'
            },
            
            'objective_4_column_separation': {
                'type': 'preference_rate', 
                'description': 'Column separation movement preferences',
                'preference_rate': 0.646,  # From your analysis
                'formulation': 'minimize weighted_column_movement_cost',
                'interpretation': '64.6% preference for adjacent columns'
            },
            
            'meta_information': {
                'total_objectives': 4,
                'statistical_approach': 'Option C - Independent objectives with FDR correction within key preferences',
                'frequency_control': 'Applied inverse frequency weighting',
                'confidence_level': '95% confidence intervals',
                'evidence_base': 'Direct AA vs BB bigram comparisons only',
                'sample_size': f"{len(key_scores)} keys analyzed from direct comparisons",
                'missing_data_approach': 'Direct evidence only - no interpolation for missing pairs'
            }
        }
        
        # Save comprehensive specification
        import json
        spec_path = os.path.join(output_folder, 'complete_moo_specification.json')
        with open(spec_path, 'w') as f:
            json.dump(moo_spec, f, indent=2, default=str)
        
        logger.info(f"Complete MOO specification saved to: {spec_path}")
        
        # Create algorithm implementation guide
        guide_lines = [
            "MOO ALGORITHM IMPLEMENTATION GUIDE",
            "=" * 40,
            "",
            "APPROACH 1: Use Tiered Key Preferences (Single Objective)",
            "Suitable for: Genetic algorithms, simulated annealing, simple MOO",
            "",
            "objective_score = sum(tier_weight[key] * frequency[key] for key in layout)",
            "where tier_weight comes from moo_key_preference_tiers.csv",
            "",
            "APPROACH 2: Use Pairwise Constraints (Multiple Constraints)", 
            "Suitable for: Constraint satisfaction, NSGA-II, complex MOO",
            "",
            "For each row in moo_pairwise_constraints.csv:",
            "  constraint: frequency[promote_key] should > frequency[demote_key]",
            "  weight by: preference_strength value",
            "",
            "APPROACH 3: Hybrid (Recommended)",
            "- Use tiered preferences as primary objective",  
            "- Use strongest pairwise constraints (>80%) as hard constraints",
            "- Use moderate constraints (60-80%) as soft penalties",
            "",
            "INTEGRATION WITH OTHER OBJECTIVES:",
            "- Row separation: Penalize large row movements (weight: 0.75)",
            "- Column separation: Penalize non-adjacent columns (weight: 0.646)", 
            "- All objectives are independent (Option C statistical approach)",
            "",
            f"MISSING PAIRWISE DATA: {len(key_scores) * (len(key_scores) - 1) // 2 - len(constraints_df)} key pairs have no direct evidence",
            "Options: (1) Treat as neutral (50%), (2) Infer from mixed bigram data (see discussion)"
        ]
        
        guide_path = os.path.join(output_folder, 'moo_implementation_guide.txt')
        with open(guide_path, 'w') as f:
            f.write('\n'.join(guide_lines))
        
        logger.info(f"MOO implementation guide saved to: {guide_path}")

    def _extract_inferred_key_preferences_for_missing_pairs(self, covered_pairs: Set[Tuple[str, str]]) -> pd.DataFrame:
        """
        Extract inferred key preferences for pairs missing from direct AA vs BB data.
        Uses broader bigram data where keys appear in mixed bigrams.
        """
        
        inferred_instances = []
        all_possible_pairs = set()
        
        # Generate all possible left-hand key pairs
        left_hand_keys = list(self.left_hand_keys)
        from itertools import combinations
        all_possible_pairs = set(combinations(sorted(left_hand_keys), 2))
        
        # Find missing pairs
        missing_pairs = all_possible_pairs - covered_pairs
        logger.info(f"Found {len(missing_pairs)} missing key pairs for inference")
        logger.info(f"Missing pairs: {sorted(missing_pairs)}")
        
        # For each missing pair, look for indirect evidence in mixed bigrams
        for key1, key2 in missing_pairs:
            pair_instances = []
            
            for _, row in self.data.iterrows():
                chosen = str(row['chosen_bigram']).lower()
                unchosen = str(row['unchosen_bigram']).lower()
                
                if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                    continue
                
                # Skip repeated bigrams (already handled by direct method)
                chosen_is_repeated = len(chosen) == 2 and chosen[0] == chosen[1]
                unchosen_is_repeated = len(unchosen) == 2 and unchosen[0] == unchosen[1]
                if chosen_is_repeated or unchosen_is_repeated:
                    continue
                
                chosen_keys = set(chosen)
                unchosen_keys = set(unchosen)
                
                # Method 1: Shared-key comparisons (QA vs QB -> A vs B)
                shared_keys = chosen_keys & unchosen_keys
                if len(shared_keys) == 1:
                    chosen_unique = list(chosen_keys - shared_keys)[0]
                    unchosen_unique = list(unchosen_keys - shared_keys)[0]
                    
                    if {chosen_unique, unchosen_unique} == {key1, key2}:
                        shared_context = list(shared_keys)[0]
                        
                        # Determine which key "won" this comparison
                        chosen_key = chosen_unique
                        unchosen_key = unchosen_unique
                        
                        pair_instances.append({
                            'user_id': row['user_id'],
                            'chosen_bigram': chosen,
                            'unchosen_bigram': unchosen,
                            'chosen_key': chosen_key,
                            'unchosen_key': unchosen_key,
                            'chose_chosen_key': 1,
                            'comparison_type': 'inferred_shared_key',
                            'shared_context': shared_context,
                            'inference_quality': 'high',  # Shared key provides good control
                            'slider_value': row.get('sliderValue', 0),
                            'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                            'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                        })
                
                # Method 2: Disjoint comparisons (AB vs CD -> A,B vs C,D)
                # Only use if no shared-key evidence available and both bigrams contain target keys
                elif len(shared_keys) == 0 and len(pair_instances) < 5:  # Fallback method
                    if key1 in chosen_keys and key2 in unchosen_keys:
                        # key1 was in chosen bigram, key2 in unchosen
                        pair_instances.append({
                            'user_id': row['user_id'],
                            'chosen_bigram': chosen,
                            'unchosen_bigram': unchosen,
                            'chosen_key': key1,
                            'unchosen_key': key2,
                            'chose_chosen_key': 1,
                            'comparison_type': 'inferred_disjoint',
                            'shared_context': 'none',
                            'inference_quality': 'low',  # Disjoint has more confounding
                            'slider_value': row.get('sliderValue', 0),
                            'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                            'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                        })
                    elif key2 in chosen_keys and key1 in unchosen_keys:
                        # key2 was in chosen bigram, key1 in unchosen
                        pair_instances.append({
                            'user_id': row['user_id'],
                            'chosen_bigram': chosen,
                            'unchosen_bigram': unchosen,
                            'chosen_key': key2,
                            'unchosen_key': key1,
                            'chose_chosen_key': 1,
                            'comparison_type': 'inferred_disjoint',
                            'shared_context': 'none',
                            'inference_quality': 'low',
                            'slider_value': row.get('sliderValue', 0),
                            'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                            'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                        })
            
            # Add instances for this pair if we found any
            if len(pair_instances) >= 3:  # Minimum threshold for inference
                inferred_instances.extend(pair_instances)
                
                high_quality = len([inst for inst in pair_instances if inst['inference_quality'] == 'high'])
                logger.info(f"Inferred evidence for {key1.upper()}-{key2.upper()}: {len(pair_instances)} instances ({high_quality} high-quality)")
            
        logger.info(f"Total inferred instances: {len(inferred_instances)}")
        return pd.DataFrame(inferred_instances)

    def _analyze_inferred_preferences(self, inferred_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze inferred preferences with quality weighting."""
        
        if inferred_df.empty:
            return {'error': 'No inferred instances found'}
        
        # Apply quality weighting in addition to frequency weighting
        inferred_df['quality_weight'] = inferred_df['inference_quality'].map({
            'high': 1.0,    # Shared-key comparisons
            'medium': 0.7,  # Other controlled methods
            'low': 0.3      # Disjoint comparisons  
        })
        
        # Apply frequency weighting
        weighted_inferred = self.apply_frequency_weights(inferred_df.copy())
        
        # Combine quality and frequency weights
        weighted_inferred['combined_weight'] = (
            weighted_inferred['frequency_weight'] * weighted_inferred['quality_weight']
        )
        
        # Analyze each inferred key pair
        inferred_results = {}
        
        for key_pair in weighted_inferred['key_pair'].unique() if 'key_pair' in weighted_inferred.columns else []:
            pair_data = weighted_inferred[
                weighted_inferred.apply(lambda row: 
                    tuple(sorted([row['chosen_key'], row['unchosen_key']])) == key_pair, axis=1)
            ]
            
            if len(pair_data) < 3:
                continue
                
            key1, key2 = key_pair
            
            # Calculate weighted preference
            pair_data['chose_key1'] = pair_data.apply(
                lambda row: 1 if row['chosen_key'] == key1 else 0, axis=1
            )
            
            total_weight = pair_data['combined_weight'].sum()
            if total_weight > 0:
                weighted_preference = (pair_data['chose_key1'] * pair_data['combined_weight']).sum() / total_weight
            else:
                weighted_preference = 0.5
            
            # Quality assessment
            high_quality_count = len(pair_data[pair_data['inference_quality'] == 'high'])
            quality_score = high_quality_count / len(pair_data)
            
            # Statistical test with effective sample size
            effective_n = (total_weight ** 2) / (pair_data['combined_weight'] ** 2).sum()
            
            if effective_n > 0:
                z_score = (weighted_preference - 0.5) / np.sqrt(0.25 / effective_n)
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
            else:
                p_value = 1.0
            
            # Determine winner
            winner = key1 if weighted_preference > 0.5 else key2
            winner_rate = max(weighted_preference, 1 - weighted_preference)
            
            inferred_results[key_pair] = {
                'key1': key1,
                'key2': key2,
                'n_comparisons': len(pair_data),
                'high_quality_instances': high_quality_count,
                'quality_score': quality_score,
                'weighted_preference': weighted_preference,
                'winner': winner,
                'winner_rate': winner_rate,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'evidence_type': 'inferred',
                'reliability': 'high' if quality_score > 0.7 else 'medium' if quality_score > 0.3 else 'low',
                'interpretation': f"{winner.upper()} > {key2.upper() if winner == key1 else key1.upper()}: {winner_rate:.1%} (inferred, {inferred_results[key_pair]['reliability']} reliability)"
            }
        
        return {
            'inferred_results': inferred_results,
            'n_inferred_pairs': len(inferred_results),
            'instances_data': inferred_df,
            'method': 'inferred_from_mixed_bigrams',
            'quality_distribution': weighted_inferred['inference_quality'].value_counts().to_dict()
        }

    def _create_complete_preference_matrix(self, direct_results: Dict, inferred_results: Dict) -> pd.DataFrame:
        """Create complete pairwise preference matrix combining direct and inferred evidence."""
        
        all_keys = list(self.left_hand_keys)
        matrix_data = []
        
        from itertools import combinations
        for key1, key2 in combinations(all_keys, 2):
            key_pair = tuple(sorted([key1, key2]))
            
            # Check direct evidence first
            if key_pair in direct_results:
                result = direct_results[key_pair]
                matrix_data.append({
                    'key1': key1.upper(),
                    'key2': key2.upper(), 
                    'key1_preference_rate': result.get('weighted_key1_rate', result.get('unweighted_key1_rate', 0.5)),
                    'winner': result['winner'].upper(),
                    'winner_rate': result['winner_rate'],
                    'evidence_type': 'direct',
                    'reliability': 'high',
                    'p_value': result['p_value'],
                    'significant': result['significant'],
                    'n_comparisons': result['n_comparisons']
                })
            
            # Check inferred evidence
            elif inferred_results and key_pair in inferred_results.get('inferred_results', {}):
                result = inferred_results['inferred_results'][key_pair]
                key1_rate = result['weighted_preference'] if result['key1'] == key1 else 1 - result['weighted_preference']
                
                matrix_data.append({
                    'key1': key1.upper(),
                    'key2': key2.upper(),
                    'key1_preference_rate': key1_rate,
                    'winner': result['winner'].upper(),
                    'winner_rate': result['winner_rate'],
                    'evidence_type': 'inferred',
                    'reliability': result['reliability'],
                    'p_value': result['p_value'],
                    'significant': result['significant'],
                    'n_comparisons': result['n_comparisons']
                })
            
            # No evidence - neutral preference
            else:
                matrix_data.append({
                    'key1': key1.upper(),
                    'key2': key2.upper(),
                    'key1_preference_rate': 0.5,
                    'winner': 'tie',
                    'winner_rate': 0.5,
                    'evidence_type': 'none',
                    'reliability': 'none',
                    'p_value': 1.0,
                    'significant': False,
                    'n_comparisons': 0
                })
        
        complete_matrix = pd.DataFrame(matrix_data)
        return complete_matrix

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