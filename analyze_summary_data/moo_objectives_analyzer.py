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
                'figure_dpi': 300,
                'strategic_key_comparisons_only': True, 
                'frequency_weighting_method': 'inverse_frequency'  # Standardized
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
            
            print(f"\n📊 This should give much more realistic preference rates than 100%!")
            print(f"Expected: 50-80% preference rates with meaningful frequency bias (2-10%)")
            
            return df

    def _extract_direct_key_preferences(self) -> pd.DataFrame:
        """Extract direct key preferences from repeated letter bigram comparisons (AA vs BB)."""
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
                        'chose_chosen_key': 1,  # By definition, chosen_key was preferred
                        'slider_value': row.get('sliderValue', 0),
                        'comparison_type': 'direct_repeated_bigram',
                        'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                        'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                    })
        
        return pd.DataFrame(direct_instances)

    def _extract_inferred_key_preferences(self, covered_pairs: Set[Tuple[str, str]]) -> pd.DataFrame:
        """Extract key preferences from mixed bigrams for uncovered key pairs."""
        inferred_instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
            
            # Skip if either is a repeated letter (already handled by direct method)
            chosen_is_repeated = len(chosen) == 2 and chosen[0] == chosen[1]
            unchosen_is_repeated = len(unchosen) == 2 and unchosen[0] == unchosen[1]
            if chosen_is_repeated or unchosen_is_repeated:
                continue
            
            # Extract key comparisons, but only for uncovered pairs
            chosen_keys = set(chosen)
            unchosen_keys = set(unchosen)
            shared_keys = chosen_keys & unchosen_keys
            
            # Shared key case (QA vs QW -> A vs W comparison)
            if len(shared_keys) == 1:
                chosen_unique = list(chosen_keys - shared_keys)[0]
                unchosen_unique = list(unchosen_keys - shared_keys)[0]
                key_pair = tuple(sorted([chosen_unique, unchosen_unique]))
                
                # Only add if this pair wasn't covered by direct method
                if key_pair not in covered_pairs:
                    inferred_instances.append({
                        'user_id': row['user_id'],
                        'chosen_bigram': chosen,
                        'unchosen_bigram': unchosen,
                        'chosen_key': chosen_unique,
                        'unchosen_key': unchosen_unique,
                        'chose_chosen_key': 1,  # chosen_key was in the preferred bigram
                        'slider_value': row.get('sliderValue', 0),
                        'comparison_type': 'inferred_shared_key',
                        'shared_context': list(shared_keys)[0],
                        'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                        'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                    })
        
        return pd.DataFrame(inferred_instances)

    def _test_key_preference(self) -> Dict[str, Any]:
        """Test key preferences using hybrid approach: direct repeated bigrams + inferred for gaps."""
        logger.info("Testing key preferences with hybrid approach: direct + inferred...")
        
        # Step 1: Extract direct preferences from repeated letter bigrams
        direct_instances = self._extract_direct_key_preferences()
        logger.info(f"Found {len(direct_instances)} direct key preference instances")
        
        # Debug: Show what repeated bigrams we have
        if not direct_instances.empty:
            repeated_bigrams = direct_instances.groupby(['chosen_bigram', 'unchosen_bigram']).size().reset_index(name='count')
            logger.info(f"Repeated bigram comparisons found: {len(repeated_bigrams)}")
            logger.info(f"Top repeated comparisons: {repeated_bigrams.nlargest(5, 'count')[['chosen_bigram', 'unchosen_bigram', 'count']].to_dict('records')}")
        
        if not direct_instances.empty:
            # Analyze direct instances
            direct_instances['key_pair'] = direct_instances.apply(
                lambda row: tuple(sorted([row['chosen_key'], row['unchosen_key']])), axis=1
            )
            covered_pairs = set(direct_instances['key_pair'].unique())
            logger.info(f"Direct method covers {len(covered_pairs)} key pairs: {sorted(covered_pairs)}")
        else:
            covered_pairs = set()
            logger.warning("No direct repeated bigram comparisons found")
        
        # Step 2: Extract inferred preferences for uncovered pairs
        inferred_instances = self._extract_inferred_key_preferences(covered_pairs)
        logger.info(f"Found {len(inferred_instances)} inferred key preference instances for uncovered pairs")
        
        if not inferred_instances.empty:
            inferred_instances['key_pair'] = inferred_instances.apply(
                lambda row: tuple(sorted([row['chosen_key'], row['unchosen_key']])), axis=1
            )
        
        # Step 3: Combine and analyze
        all_instances = []
        if not direct_instances.empty:
            all_instances.append(direct_instances)
        if not inferred_instances.empty:
            all_instances.append(inferred_instances)
        
        if not all_instances:
            return {'error': 'No key preference instances found'}
        
        combined_instances = pd.concat(all_instances, ignore_index=True)
        logger.info(f"Combined: {len(combined_instances)} total instances from {combined_instances['user_id'].nunique()} users")
        
        # Apply frequency weighting to combined data
        weighted_metrics = self.calculate_weighted_preference_metrics(combined_instances, 'chose_chosen_key')
        weighted_data = weighted_metrics['weighted_data']
        
        # Analyze each key pair
        pairwise_results = {}
        coverage_info = {}
        all_p_values = []
        
        for key_pair in weighted_data['key_pair'].unique():
            pair_data = weighted_data[weighted_data['key_pair'] == key_pair].copy()
            
            if len(pair_data) < 5:  # Minimum threshold
                continue
            
            key1, key2 = key_pair
            
            # Determine which key was preferred (accounting for alphabetical sorting)
            pair_data['chose_key1'] = pair_data.apply(
                lambda row: 1 if row['chosen_key'] == key1 else 0, axis=1
            )
            
            # Weighted proportion test for this pair
            pair_weighted_test = self._weighted_proportion_test(pair_data, 'chose_key1', 'frequency_weight')
            
            # Track coverage type
            direct_count = len(pair_data[pair_data['comparison_type'] == 'direct_repeated_bigram'])
            inferred_count = len(pair_data[pair_data['comparison_type'] == 'inferred_shared_key'])
            
            coverage_info[key_pair] = {
                'direct_instances': direct_count,
                'inferred_instances': inferred_count,
                'total_instances': len(pair_data),
                'primary_method': 'direct' if direct_count > inferred_count else 'inferred'
            }
            
            pairwise_results[key_pair] = {
                'key1': key1,
                'key2': key2,
                'n_comparisons': len(pair_data),
                'direct_comparisons': direct_count,
                'inferred_comparisons': inferred_count,
                'primary_method': coverage_info[key_pair]['primary_method'],
                'weighted_key1_rate': pair_weighted_test['weighted_preference_rate'],
                'unweighted_key1_rate': pair_data['chose_key1'].mean(),
                'frequency_bias': abs(pair_weighted_test['weighted_preference_rate'] - pair_data['chose_key1'].mean()),
                'p_value': pair_weighted_test['p_value'],
                'significant': pair_weighted_test['p_value'] < 0.05,
                'effect_size': pair_weighted_test['effect_size'],
                'winner': key1 if pair_weighted_test['weighted_preference_rate'] > 0.5 else key2,
                'winner_rate': max(pair_weighted_test['weighted_preference_rate'], 
                                 1.0 - pair_weighted_test['weighted_preference_rate']),
                'interpretation': f"{key1.upper()} vs {key2.upper()}: {key1.upper() if pair_weighted_test['weighted_preference_rate'] > 0.5 else key2.upper()} preferred {max(pair_weighted_test['weighted_preference_rate'], 1.0 - pair_weighted_test['weighted_preference_rate']):.1%} ({coverage_info[key_pair]['primary_method']})"
            }
            
            all_p_values.append(pair_weighted_test['p_value'])
        
        # Calculate coverage statistics
        total_possible_pairs = len(self.left_hand_keys) * (len(self.left_hand_keys) - 1) // 2
        covered_pairs_count = len(pairwise_results)
        direct_pairs = len([p for p in coverage_info.values() if p['primary_method'] == 'direct'])
        inferred_pairs = len([p for p in coverage_info.values() if p['primary_method'] == 'inferred'])
        
        # Combine p-values for overall test
        overall_test = None
        if len(all_p_values) > 1:
            try:
                overall_stat, overall_p = combine_pvalues(all_p_values, method='fisher')
                overall_test = {
                    'combined_p_value': overall_p,
                    'significant': overall_p < 0.05,
                    'method': 'fisher_combined',
                    'n_tests': len(all_p_values)
                }
            except Exception as e:
                logger.warning(f"Failed to combine p-values: {e}")
                overall_test = {
                    'min_p_value': min(all_p_values) if all_p_values else 1.0,
                    'significant': min(all_p_values) < 0.05 if all_p_values else False,
                    'method': 'minimum_p',
                    'n_tests': len(all_p_values)
                }
        
        # Create individual key scores using hybrid data
        individual_scores, ranked_keys = self._calculate_frequency_weighted_key_scores(pairwise_results)
        
        # Create significant pairs summary
        significant_pairs = []
        for pair, results in pairwise_results.items():
            if results['significant']:
                method_label = "direct" if results['primary_method'] == 'direct' else "inferred"
                significant_pairs.append(f"{results['winner'].upper()} > {results['key2'].upper() if results['winner'] == results['key1'] else results['key1'].upper()} ({results['winner_rate']:.1%}, {method_label})")
        
        # Export pairwise results for detailed inspection
        pairwise_export = []
        for pair, results in pairwise_results.items():
            pairwise_export.append({
                'key_pair': f"{results['key1'].upper()}-{results['key2'].upper()}",
                'key1': results['key1'].upper(),
                'key2': results['key2'].upper(),
                'method': results['primary_method'],
                'direct_comparisons': results['direct_comparisons'],
                'inferred_comparisons': results['inferred_comparisons'],
                'total_comparisons': results['n_comparisons'],
                'unweighted_key1_rate': results['unweighted_key1_rate'],
                'weighted_key1_rate': results['weighted_key1_rate'],
                'frequency_bias': results['frequency_bias'],
                'winner': results['winner'].upper(),
                'winner_preference_rate': results['winner_rate'],
                'p_value': results['p_value'],
                'significant': results['significant'],
                'effect_size': results['effect_size']
            })

        return {
            'description': 'Hybrid key preferences: direct repeated bigrams + inferred for gaps',
            'method': 'hybrid_direct_and_inferred',
            'coverage_approach': 'hybrid',  # ADD THIS
            'direct_pairs_found': direct_pairs,  # ADD THIS 
            'inferred_pairs_added': inferred_pairs,  # ADD THIS
            'n_instances': len(combined_instances),
            'n_users': combined_instances['user_id'].nunique(),
            'n_key_pairs': covered_pairs_count,
            'total_possible_pairs': total_possible_pairs,
            'coverage_rate': covered_pairs_count / total_possible_pairs,
            'direct_method_pairs': direct_pairs,
            'inferred_method_pairs': inferred_pairs,
            'n_significant_pairs': len(significant_pairs),
            'strategic_subset_used': False,  # Not using strategic subset in hybrid approach
            'pairwise_results': pairwise_results,
            'pairwise_export': pairwise_export,
            'coverage_breakdown': coverage_info,
            'significant_preferences': significant_pairs,
            'individual_key_scores': individual_scores,
            'ranked_keys': ranked_keys,
            'frequency_bias_analysis': weighted_metrics,
            'simple_test': overall_test,
            'p_value': overall_test['combined_p_value'] if overall_test else 1.0,
            'preference_rate': len(significant_pairs) / len(pairwise_results) if pairwise_results else 0.0,
            'instances_data': combined_instances,
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_hybrid_results(pairwise_results, coverage_info, weighted_metrics),
            'data_quality_improved': True
        }
    
    def _interpret_hybrid_results(self, pairwise_results: Dict, coverage_info: Dict, weighted_metrics: Dict) -> str:
        """Interpret hybrid key preference results."""
        n_significant = sum(1 for r in pairwise_results.values() if r['significant'])
        n_total = len(pairwise_results)
        direct_pairs = len([c for c in coverage_info.values() if c['primary_method'] == 'direct'])
        
        overall_bias = weighted_metrics.get('bias_magnitude', 0)
        
        if n_significant == 0:
            return f"No significant key preferences from {n_total} pairs ({direct_pairs} direct, {n_total-direct_pairs} inferred, bias: {overall_bias:.1%})"
        
        # Find strongest preferences
        sorted_pairs = sorted(pairwise_results.items(), 
                             key=lambda x: (x[1]['significant'], x[1]['winner_rate']), 
                             reverse=True)
        
        top_preferences = []
        for pair, results in sorted_pairs[:3]:
            if results['significant']:
                method = "direct" if results['primary_method'] == 'direct' else "inferred"
                winner = results['winner']
                loser = results['key2'] if results['winner'] == results['key1'] else results['key1']
                rate = results['winner_rate']
                bias = results['frequency_bias']
                top_preferences.append(f"{winner.upper()} > {loser.upper()} ({rate:.1%}, {method}, bias: {bias:.1%})")
        
        return f"Significant preferences ({n_significant}/{n_total}, {direct_pairs} direct, overall bias: {overall_bias:.1%}): {'; '.join(top_preferences)}{'...' if n_significant > 3 else ''}"

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
                    if results['weighted_key1_rate'] > 0.5:
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
        """Interpret frequency-weighted column results with enhanced detail."""
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
    
    def _validate_all_objectives(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive validation on all extracted objectives."""
        validation_report = {
            'multiple_comparisons_correction': self._correct_multiple_comparisons(results),
            'effect_size_validation': self._validate_effect_sizes(results),
            'frequency_bias_impact': self._analyze_frequency_bias_impact(results),
            'cross_validation': self._cross_validate_models(results),
            'confound_analysis': self._analyze_confounds(results),
            'statistical_power': self._assess_statistical_power(results),
            'overall_validity': self._assess_overall_validity(results)
        }
        
        return validation_report
    
    def _correct_multiple_comparisons(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple comparisons correction across ALL individual tests with detailed tracking."""
        p_values = []
        test_names = []
        test_categories = {}  # Track which objective each test belongs to
        
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
                        test_name = f"{obj_name}_main"
                        p_values.append(p_val)
                        test_names.append(test_name)
                        test_categories[test_name] = obj_name
                
                # For key_preference: extract ALL pairwise p-values
                else:
                    if 'pairwise_results' in obj_results:
                        pairwise_results = obj_results['pairwise_results']
                        for key_pair, pair_results in pairwise_results.items():
                            if isinstance(pair_results, dict) and 'p_value' in pair_results:
                                p_val = pair_results['p_value']
                                if isinstance(p_val, (int, float)) and not pd.isna(p_val):
                                    test_name = f"key_preference_{key_pair[0]}_vs_{key_pair[1]}"
                                    p_values.append(p_val)
                                    test_names.append(test_name)
                                    test_categories[test_name] = 'key_preference'
        
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
            obj_name = test_categories[test_name]
            if obj_name not in objective_breakdown:
                objective_breakdown[obj_name] = {
                    'total_tests': 0,
                    'original_significant': 0,
                    'fdr_significant': 0,
                    'significant_tests': []
                }
            
            objective_breakdown[obj_name]['total_tests'] += 1
            if orig_p < 0.05:
                objective_breakdown[obj_name]['original_significant'] += 1
            if fdr_p < 0.05:
                objective_breakdown[obj_name]['fdr_significant'] += 1
                objective_breakdown[obj_name]['significant_tests'].append(test_name)
        
        # Create detailed significant tests summary
        detailed_significant_tests = {}
        for test_name, orig_p, fdr_p in zip(test_names, p_values, fdr_corrected):
            if fdr_p < 0.05:
                detailed_significant_tests[test_name] = {
                    'original_p': orig_p,
                    'fdr_corrected_p': fdr_p,
                    'objective': test_categories[test_name]
                }
        
        return {
            'total_tests': len(p_values),
            'original_significant': original_significant,
            'bonferroni_significant': bonferroni_significant,
            'fdr_significant': fdr_significant,
            'fdr_bh_significant': fdr_significant,  # Alias for compatibility
            'corrected_p_values': corrections,
            'original_p_values': dict(zip(test_names, p_values)),
            'test_categories': test_categories,
            'objective_breakdown': objective_breakdown,
            'detailed_significant_tests': detailed_significant_tests,
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

    def _analyze_frequency_bias_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how frequency weighting affects conclusions across all objectives."""
        bias_impact_analysis = {}
        
        for obj_name, obj_results in results.items():
            if 'error' not in obj_results and 'frequency_bias_analysis' in obj_results:
                bias_data = obj_results['frequency_bias_analysis']
                
                unweighted = bias_data.get('unweighted_preference_rate', 0.5)
                weighted = bias_data.get('weighted_preference_rate', 0.5)
                bias_magnitude = abs(weighted - unweighted)
                
                # Calculate rank order change (simplified)
                direction_change = (unweighted > 0.5) != (weighted > 0.5)
                
                freq_stats = bias_data.get('frequency_stats', {})
                
                bias_impact_analysis[obj_name] = {
                    'unweighted_rate': unweighted,
                    'weighted_rate': weighted,
                    'bias_magnitude': bias_magnitude,
                    'direction_changed': direction_change,
                    'frequency_ratio': freq_stats.get('frequency_ratio', 1.0),
                    'bias_severity': freq_stats.get('bias_severity', 'unknown'),
                    'practical_impact': self._assess_practical_bias_impact(bias_magnitude, direction_change)
                }
        
        # Overall summary
        high_bias_objectives = sum(1 for analysis in bias_impact_analysis.values() 
                                 if analysis['bias_magnitude'] > 0.05)
        direction_changed_objectives = sum(1 for analysis in bias_impact_analysis.values() 
                                         if analysis['direction_changed'])
        
        return {
            'individual_analysis': bias_impact_analysis,
            'high_bias_objectives': high_bias_objectives,
            'direction_changed_objectives': direction_changed_objectives,
            'total_objectives_analyzed': len(bias_impact_analysis),
            'overall_bias_severity': 'high' if high_bias_objectives > 1 else 'moderate' if high_bias_objectives > 0 else 'low'
        }

    def _assess_practical_bias_impact(self, bias_magnitude: float, direction_changed: bool) -> str:
        """Assess practical impact of frequency bias."""
        if direction_changed:
            return "critical - changes conclusion direction"
        elif bias_magnitude > 0.10:
            return "high - substantially changes effect size"
        elif bias_magnitude > 0.05:
            return "moderate - notable effect size change"
        else:
            return "low - minimal practical impact"
    
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
        
        # For instance-level analysis with frequency weighting, effect size is deviation from neutral (0.5)
        if 'simple_test' in obj_results and isinstance(obj_results['simple_test'], dict):
            pref_rate = obj_results['simple_test'].get('weighted_preference_rate')
            if pref_rate is None:  # Fallback to non-weighted
                pref_rate = obj_results['simple_test'].get('preference_rate')
            if isinstance(pref_rate, (int, float)) and not pd.isna(pref_rate):
                effect_sizes.append(abs(pref_rate - 0.5))
        
        # For key preference, extract effect sizes from pairwise comparisons
        elif 'pairwise_results' in obj_results:
            for pair_results in obj_results['pairwise_results'].values():
                if isinstance(pair_results, dict) and 'effect_size' in pair_results:
                    es = pair_results['effect_size']
                    if isinstance(es, (int, float)) and not pd.isna(es):
                        effect_sizes.append(abs(es))
        
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
            'column_separation': 0.05,
            'column_4_vs_5': 0.08  # Slightly higher threshold for column preference
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
            if 'error' not in obj_results and 'instances_data' in obj_results:
                # Use the original instances data for CV
                instances_data = obj_results['instances_data']
                if len(instances_data) >= 50:  # Minimum for meaningful CV
                    cv_score = self._perform_cross_validation_on_instances(instances_data, obj_name)
                    cv_results[obj_name] = cv_score
        
        return cv_results
    
    def _perform_cross_validation_on_instances(self, instances_data: pd.DataFrame, obj_name: str) -> Dict[str, Any]:
        """Perform k-fold cross-validation on instance-level data."""
        try:
            # Determine outcome variable based on objective
            outcome_mapping = {
                'key_preference': 'chose_key',
                'row_separation': 'chose_smaller_separation',
                'column_separation': 'chose_adjacent',
                'column_4_vs_5': 'chose_column_4'
            }
            
            outcome_var = outcome_mapping.get(obj_name)
            if outcome_var not in instances_data.columns:
                return {'error': f'Outcome variable {outcome_var} not found for {obj_name}'}
            
            # Prepare data
            control_cols = [col for col in instances_data.columns if col.startswith('log_')]
            if len(control_cols) == 0:
                return {'error': 'No control variables available for cross-validation'}
            
            X = instances_data[control_cols].fillna(0)
            y = instances_data[outcome_var]
            
            # Perform 5-fold cross-validation
            cv = KFold(n_splits=min(5, len(instances_data) // 10), shuffle=True, random_state=42)
            
            # Use simple linear regression for CV (easier to interpret)
            model = LinearRegression()
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            return {
                'mean_r2': np.mean(cv_scores),
                'std_r2': np.std(cv_scores),
                'min_r2': np.min(cv_scores),
                'max_r2': np.max(cv_scores),
                'stability': 'high' if np.std(cv_scores) < 0.1 else 'medium' if np.std(cv_scores) < 0.2 else 'low',
                'n_folds': len(cv_scores),
                'n_instances': len(instances_data)
            }
            
        except Exception as e:
            return {'error': f'Cross-validation failed: {str(e)}'}
        
    def _analyze_confounds(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential confounding variables and their control."""
        confound_analysis = {}
        
        for obj_name, obj_results in results.items():
            if 'error' not in obj_results and 'instances_data' in obj_results:
                confounds = self._assess_confound_control(obj_results['instances_data'])
                confound_analysis[obj_name] = confounds
        
        return confound_analysis
    
    def _assess_confound_control(self, instances_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess how well frequency confounds are controlled."""
        control_cols = [col for col in instances_data.columns if col.startswith('log_')]
        
        if not control_cols:
            return {'error': 'No frequency controls found'}
        
        # For instance-level data, we need to create a predictor variable
        # This is simplified - in practice, you'd want the actual predictor used in each test
        try:
            # Create a simple predictor (preference outcome)
            outcome_vars = ['chose_key', 'chose_smaller_separation', 'chose_adjacent', 'chose_column_4']
            predictor_col = None
            for var in outcome_vars:
                if var in instances_data.columns:
                    predictor_col = var
                    break
            
            if predictor_col is None:
                return {'error': 'No suitable predictor variable found'}
            
            # Check correlation between predictor and controls
            predictor_control_corrs = {}
            
            for control in control_cols:
                if control in instances_data.columns:
                    corr = instances_data[predictor_col].corr(instances_data[control])
                    if not pd.isna(corr):
                        predictor_control_corrs[control] = corr
            
            if not predictor_control_corrs:
                return {'error': 'No valid correlations calculated'}
            
            max_corr = max([abs(corr) for corr in predictor_control_corrs.values()])
            
            return {
                'predictor_control_correlations': predictor_control_corrs,
                'max_correlation_with_controls': max_corr,
                'confound_risk': 'high' if max_corr > 0.7 else 'medium' if max_corr > 0.4 else 'low',
                'control_adequacy': 'good' if max_corr < 0.4 else 'moderate' if max_corr < 0.7 else 'poor'
            }
        
        except Exception as e:
            return {'error': f'Confound analysis failed: {str(e)}'}
    
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
        n_instances = obj_results.get('n_instances', 0)
        n_users = obj_results.get('n_users', 0)
        
        if n_instances == 0:
            return {'error': 'No sample size information available'}
        
        # Simple power assessment based on sample size
        if n_instances < 50:
            power_level = 'low'
            power_comment = 'Sample size may be insufficient for reliable detection of small effects'
        elif n_instances < 200:
            power_level = 'medium'
            power_comment = 'Adequate power for medium to large effects'
        else:
            power_level = 'high'
            power_comment = 'Good power for detecting small to medium effects'
        
        # Consider number of users for nested data structure
        effective_power = power_level
        if n_users > 0 and n_instances / n_users > 10:
            # High instances per user might inflate power estimates
            power_comment += ' (note: multiple instances per user may reduce effective power)'
        
        return {
            'n_instances': n_instances,
            'n_users': n_users,
            'instances_per_user': n_instances / max(n_users, 1),
            'power_level': effective_power,
            'power_comment': power_comment,
            'recommended_minimum': 100  # For psychology/HCI research
        }
    
    def _assess_overall_validity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Provide overall validity assessment with enhanced significance detection."""
        successful_objectives = sum(1 for r in results.values() if 'error' not in r)
        total_objectives = len(results)
        
        # Count objectives with significant results (enhanced logic)
        significant_objectives = 0
        objective_significance = {}
        
        for obj_name, obj_results in results.items():
            if 'error' not in obj_results:
                is_significant = False
                
                # Check simple_test p-value (most reliable for frequency-weighted tests)
                if 'simple_test' in obj_results and isinstance(obj_results['simple_test'], dict):
                    p_val = obj_results['simple_test'].get('p_value')
                    if p_val is not None and p_val < 0.05:
                        is_significant = True
                # Fallback to direct p-value
                elif obj_results.get('p_value', 1.0) < 0.05:
                    is_significant = True
                
                objective_significance[obj_name] = is_significant
                if is_significant:
                    significant_objectives += 1
        
        # Overall assessment
        success_rate = successful_objectives / total_objectives if total_objectives > 0 else 0
        significance_rate = significant_objectives / successful_objectives if successful_objectives > 0 else 0
        
        if success_rate >= 0.8 and significance_rate >= 0.5:
            overall_validity = 'high'
            validity_comment = 'Strong evidence for multiple typing preference objectives'
        elif success_rate >= 0.6 and significance_rate >= 0.3:
            overall_validity = 'medium'
            validity_comment = 'Moderate evidence for typing preference objectives'
        else:
            overall_validity = 'low'
            validity_comment = 'Limited evidence for typing preference objectives'
        
        # Generate recommendations
        recommendations = []
        if success_rate < 0.8:
            recommendations.append('Consider data quality improvements or alternative objective definitions')
        if significance_rate < 0.3:
            recommendations.append('Low significance rate suggests weak effects or insufficient power')
        
        recommendations.append('Apply FDR correction for multiple comparisons')
        recommendations.append('Focus on objectives with large effect sizes for practical application')
        
        return {
            'successful_objectives': successful_objectives,
            'total_objectives': total_objectives,
            'significant_objectives': significant_objectives,
            'objective_significance': objective_significance,
            'success_rate': success_rate,
            'significance_rate': significance_rate,
            'overall_validity': overall_validity,
            'validity_comment': validity_comment,
            'recommendations': recommendations
        }

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def _generate_enhanced_summary(self, results: Dict[str, Any], 
                                validation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced summary with validation insights and frequency bias impact."""
        
        # Get multiple comparisons data safely
        mc_data = validation.get('multiple_comparisons_correction', {})
        fdr_significant = mc_data.get('fdr_significant', mc_data.get('fdr_bh_significant', 0))
        
        # Get frequency bias impact
        bias_data = validation.get('frequency_bias_impact', {})
        high_bias_objectives = bias_data.get('high_bias_objectives', 0)
        direction_changed = bias_data.get('direction_changed_objectives', 0)
        
        # Get overall validity
        validity_data = validation.get('overall_validity', {})
        
        return {
            'objectives_extracted': len([r for r in results.values() if 'error' not in r]),
            'objectives_significant': validity_data.get('significant_objectives', 0),
            'fdr_corrected_significant': fdr_significant,
            'total_tests_performed': mc_data.get('total_tests', len(results)),
            'overall_validity': validity_data.get('overall_validity', 'unknown'),
            'validity_comment': validity_data.get('validity_comment', ''),
            'frequency_bias_impact': {
                'high_bias_objectives': high_bias_objectives,
                'direction_changed_objectives': direction_changed,
                'overall_severity': bias_data.get('overall_bias_severity', 'unknown')
            },
            'recommendations': validity_data.get('recommendations', []),
            'strategic_key_subset_used': results.get('key_preference', {}).get('strategic_subset_used', False)
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
                    
                    # Add enhanced context analysis
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
        
        logger.info(f"Enhanced comprehensive report saved to {report_path}")

    def _save_results(self, enhanced_results: Dict[str, Any], output_folder: str) -> None:
        """Save enhanced results including frequency bias analysis and corrected significance."""
        
        # Save key scores first
        self._save_key_scores_for_moo(enhanced_results['objectives'], output_folder)
        
        # Save enhanced summary with frequency bias info
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
                sig_tests_data.append({
                    'Test_Name': test_name,
                    'Objective': test_info['objective'],
                    'Original_P_Value': test_info['original_p'],
                    'FDR_Corrected_P_Value': test_info['fdr_corrected_p']
                })
            
            sig_tests_df = pd.DataFrame(sig_tests_data)
            sig_tests_path = os.path.join(output_folder, 'fdr_corrected_significant_tests.csv')
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
            self._save_key_pairwise_results_enhanced(
                enhanced_results['objectives']['key_preference'], 
                output_folder
            )

    def _save_key_scores_for_moo(self, results: Dict[str, Any], output_folder: str) -> None:
        """Save key preference scores in MOO-ready formats with frequency bias info."""
        
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
            
            csv_path = os.path.join(output_folder, 'frequency_corrected_key_preference_scores.csv')
            key_scores_df.to_csv(csv_path, index=False)
            logger.info(f"Frequency-corrected key preference scores saved to {csv_path}")

    def _save_key_pairwise_results_enhanced(self, key_pref_results: Dict[str, Any], output_folder: str) -> None:
        """Save complete pairwise key comparison results with frequency bias info."""
        if 'pairwise_export' not in key_pref_results:
            return
        
        pairwise_df = pd.DataFrame(key_pref_results['pairwise_export'])
        
        # Sort by significance and effect size
        pairwise_df = pairwise_df.sort_values(['significant', 'effect_size'], ascending=[False, False])
        
        csv_path = os.path.join(output_folder, 'frequency_corrected_key_pairwise_comparisons.csv')
        pairwise_df.to_csv(csv_path, index=False)
        logger.info(f"Frequency-corrected pairwise key comparisons saved to {csv_path}")
        
        # Create summary of significant preferences with bias info
        significant_df = pairwise_df[pairwise_df['significant'] == True]
        if len(significant_df) > 0:
            summary_path = os.path.join(output_folder, 'frequency_corrected_significant_key_preferences.csv')
            significant_df.to_csv(summary_path, index=False)
            logger.info(f"Frequency-corrected significant key preferences saved to {summary_path}")

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
        analyzer = CompleteMOOObjectiveAnalyzer(args.config)
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