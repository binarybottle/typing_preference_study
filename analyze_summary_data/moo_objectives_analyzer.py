#!/usr/bin/env python3
"""
Complete Multi-Objective Optimization (MOO) Objectives Analysis for Keyboard Layout Optimization

This script extracts 6 typing mechanics objectives from bigram preference data,
controlling for English language frequency effects. These objectives are designed
to create meaningful conflicts for multi-objective keyboard layout optimization.

The 6 MOO objectives (not including trigram flow):
1. different_finger: Preference for different-finger over same-finger bigrams
2. key_preference: Individual key quality preferences (66 pairwise comparisons)  
3. home_row: Mechanical advantage of home row keys (A, S, D, F)
4. row_separation: Preferences for row transitions (same > reach > hurdle)
5. column_separation: Context-dependent column spacing preferences

Includes rigorous statistical validation with:
- Multiple comparisons correction
- Effect size validation
- Biomechanical consistency checks
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
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from itertools import combinations, permutations
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings

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
                'generate_visualizations': True,
                'figure_dpi': 300
            }
    
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
        
        logger.info("=== OBJECTIVE 1: DIFFERENT_FINGER ===")
        results['different_finger'] = self._test_different_finger_preference()
        
        logger.info("=== OBJECTIVE 2: KEY_PREFERENCE ===")
        results['key_preference'] = self._test_key_preference()
        
        logger.info("=== OBJECTIVE 3: HOME_ROW ===")
        results['home_row'] = self._test_home_row_preference()
        
        logger.info("=== OBJECTIVE 4: ROW_SEPARATION ===")
        results['row_separation'] = self._test_row_separation_preference()
        
        logger.info("=== OBJECTIVE 5: COLUMN_SEPARATION ===")
        results['column_separation'] = self._test_column_separation_preference()
        
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
        
        # Create visualizations
        if self.config.get('generate_visualizations', True):
            self._create_visualizations(enhanced_results, output_folder)
        
        # Save results
        self._save_results(enhanced_results, output_folder)
        
        logger.info(f"Complete MOO objectives analysis finished! Results saved to {output_folder}")
        return enhanced_results
    
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
                
            # Finger properties
            same_finger = self._same_finger_bigram(bigram)
            
            # Home row properties
            home_keys = {'a', 's', 'd', 'f'}
            home_count = sum(1 for char in bigram if char in home_keys)
            home_category = ""
            if home_count == 2:
                home_category = "both_home"
            elif home_count == 1:
                home_category = "one_home"
            else:
                home_category = "neither_home"
            
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
                'same_finger': same_finger,
                'key1_finger': key1_pos.finger,
                'key2_finger': key2_pos.finger,
                'home_count': home_count,
                'home_category': home_category,
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
                'relevant_for_home_row': is_left_hand and home_count > 0,
                'relevant_for_row_separation': is_left_hand,
                'relevant_for_column_separation': is_left_hand
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(classification_data)
        
        # Add summary statistics
        summary_stats = {
            'total_left_hand_bigrams': len(df),
            'same_finger_bigrams': len(df[df['same_finger'] == True]),
            'different_finger_bigrams': len(df[df['same_finger'] == False]),
            'both_home_bigrams': len(df[df['home_category'] == 'both_home']),
            'one_home_bigrams': len(df[df['home_category'] == 'one_home']),
            'neither_home_bigrams': len(df[df['home_category'] == 'neither_home']),
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
            top_bigrams = df.nlargest(10, 'comparison_count')[['bigram', 'comparison_count', 'same_finger', 'home_category', 'row_category']]
            f.write(top_bigrams.to_string(index=False))
            
            f.write(f"\n\nHome row bigrams:\n")
            home_bigrams = df[df['home_count'] > 0][['bigram', 'comparison_count', 'home_category', 'same_finger']]
            f.write(home_bigrams.to_string(index=False))
        
        logger.info(f"Bigram classification diagnostic saved to {classification_path}")
        logger.info(f"Classification summary saved to {summary_path}")
        
        # Print quick summary to console
        print(f"\nBIGRAM CLASSIFICATION DIAGNOSTIC:")
        print(f"=================================")
        for key, value in summary_stats.items():
            print(f"{key}: {value}")
        
        return df

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
            'chose_more_home': {
                'variable': 'home row preference', 
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

    # =========================================================================
    # OBJECTIVE 1: DIFFERENT FINGER
    # =========================================================================
    
    def _extract_finger_difference_instances(self) -> pd.DataFrame:
        """Extract all instances where same-finger and different-finger bigrams were compared."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            # Only analyze left-hand bigrams
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
                
            same_finger_chosen = self._same_finger_bigram(chosen)
            same_finger_unchosen = self._same_finger_bigram(unchosen)
            
            # Only include if one is same-finger and other is different-finger
            if same_finger_chosen != same_finger_unchosen:
                # Code the preference: 1 = chose different-finger, 0 = chose same-finger
                chose_different_finger = 1 if not same_finger_chosen else 0
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'chose_different_finger': chose_different_finger,
                    'chosen_same_finger': same_finger_chosen,
                    'unchosen_same_finger': same_finger_unchosen,
                    'slider_value': row.get('sliderValue', 0),
                    # Only control for bigram frequencies, not individual letters
                    'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                    'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                })
        
        return pd.DataFrame(instances)

    def _test_different_finger_preference(self) -> Dict[str, Any]:
        """Test different-finger preference using instance-level analysis."""
        logger.info("Testing different-finger preference (instance-level)...")
        
        instances_df = self._extract_finger_difference_instances()
        
        if instances_df.empty:
            return {'error': 'No finger difference instances found'}
        
        logger.info(f"Found {len(instances_df)} finger difference instances from {instances_df['user_id'].nunique()} users")

        raw_preference_rate = instances_df['chose_different_finger'].mean()
        logger.info(f"Raw different-finger preference rate: {raw_preference_rate:.1%}")
        simple_test = self._simple_proportion_test(instances_df, 'chose_different_finger')
        logger.info(f"Simple test: {simple_test['interpretation']}")

        # Fit model
        model_results = self._fit_instance_level_model(
            instances_df, 
            'chose_different_finger',
            'finger_difference_instance_level'
        )

        simple_test = self._simple_proportion_test(instances_df, 'chose_different_finger')

        return {
            'description': 'Different-finger vs same-finger preference (instance-level)',
            'method': 'instance_level_analysis',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique() if not instances_df.empty else 0,
            'model_results': model_results,
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'simple_test': simple_test,          
            'p_value': simple_test['p_value'],      
            'preference_rate': simple_test['preference_rate'],  
            'interpretation': model_results.get('interpretation', 'No interpretation available')
        }
    
    def _extract_controlled_finger_comparisons(self) -> List[Tuple[str, str, Dict]]:
        """Extract controlled comparisons where bigrams share exactly one letter."""
        comparisons = []
        
        # Get all bigrams from the data
        all_bigrams = self._get_all_left_hand_bigrams()
        
        # For each left-hand letter, find controlled comparisons
        for shared_letter in self.left_hand_keys:
            # Find all bigrams containing this letter
            bigrams_with_letter = [bg for bg in all_bigrams if shared_letter in bg]
            
            # Separate into same-finger and different-finger groups
            same_finger_bigrams = []
            different_finger_bigrams = []
            
            for bigram in bigrams_with_letter:
                if self._same_finger_bigram(bigram):
                    same_finger_bigrams.append(bigram)
                else:
                    different_finger_bigrams.append(bigram)
            
            # Compare same-finger vs different-finger bigrams sharing this letter
            for sf_bigram in same_finger_bigrams:
                for df_bigram in different_finger_bigrams:
                    # Additional check: ensure they share exactly one letter
                    if self._bigrams_share_exactly_one_letter(sf_bigram, df_bigram, shared_letter):
                        comparison_data = self._extract_pairwise_comparison(sf_bigram, df_bigram)
                        
                        if comparison_data['total'] >= self.config.get('min_comparisons', 20):
                            comparisons.append((sf_bigram, df_bigram, comparison_data))
        
        logger.info(f"Found {len(comparisons)} controlled finger difference comparisons")
        return comparisons
    
    def _extract_general_finger_comparisons(self) -> List[Tuple[str, str, Dict]]:
        """Extract general comparisons between same-finger and different-finger bigrams."""
        comparisons = []
        
        # Get actual bigram pairs that were compared in the experiment
        bigram_pairs = set()
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if (len(chosen) == 2 and len(unchosen) == 2 and 
                self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                # Store as sorted tuple to avoid duplicates
                pair = tuple(sorted([chosen, unchosen]))
                bigram_pairs.add(pair)
        
        logger.info(f"Found {len(bigram_pairs)} unique bigram pairs in experimental data")
        
        # Filter to pairs that test finger difference
        finger_test_pairs = []
        for bigram1, bigram2 in bigram_pairs:
            same_finger_1 = self._same_finger_bigram(bigram1)
            same_finger_2 = self._same_finger_bigram(bigram2)
            
            # Only include if one is same-finger and other is different-finger
            if same_finger_1 != same_finger_2:
                comparison_data = self._extract_pairwise_comparison(bigram1, bigram2)
                if comparison_data['total'] >= self.config.get('min_comparisons', 20):
                    finger_test_pairs.append((bigram1, bigram2, comparison_data))
        
        logger.info(f"Found {len(finger_test_pairs)} finger difference comparisons")
        return finger_test_pairs
    
    def _bigrams_share_exactly_one_letter(self, bigram1: str, bigram2: str, expected_shared: str) -> bool:
        """Check if two bigrams share exactly one letter (the expected one)."""
        if len(bigram1) != 2 or len(bigram2) != 2:
            return False
        
        # Count shared letters
        shared_letters = set(bigram1) & set(bigram2)
        
        # Should have exactly one shared letter, and it should be the expected one
        return len(shared_letters) == 1 and expected_shared in shared_letters
    
    def _analyze_finger_difference_patterns(self, test_comparisons: List) -> Dict[str, Any]:
        """Debug the finger difference comparisons found."""
        if not test_comparisons:
            return {'message': 'No test comparisons to debug'}
        
        debug_info = {
            'total_comparisons': len(test_comparisons),
            'sample_comparisons': []
        }
        
        # Analyze first few comparisons for debugging
        for i, (bigram1, bigram2, comp_data) in enumerate(test_comparisons[:5]):
            same_finger_1 = self._same_finger_bigram(bigram1)
            same_finger_2 = self._same_finger_bigram(bigram2)
            
            debug_info['sample_comparisons'].append({
                'bigram1': bigram1,
                'bigram2': bigram2,
                'bigram1_same_finger': same_finger_1,
                'bigram2_same_finger': same_finger_2,
                'total_observations': comp_data['total'],
                'bigram1_wins': comp_data['wins_item1']
            })
        
        return debug_info
    
    def _calculate_finger_difference(self, bigram1: str, bigram2: str) -> float:
        """Calculate finger difference indicator (1 = different fingers, 0 = same finger)."""
        same_finger_bg1 = self._same_finger_bigram(bigram1)
        same_finger_bg2 = self._same_finger_bigram(bigram2)
        
        # Return 1 if one is same-finger and other is different-finger
        return float(same_finger_bg1 != same_finger_bg2)
    
    def _same_finger_bigram(self, bigram: str) -> bool:
        """Check if bigram uses same finger."""
        if len(bigram) != 2:
            return False
        
        key1, key2 = bigram[0], bigram[1]
        if key1 in self.key_positions and key2 in self.key_positions:
            return self.key_positions[key1].finger == self.key_positions[key2].finger
        
        return False

    # =========================================================================
    # OBJECTIVE 2: KEY PREFERENCE  
    # =========================================================================

    def _extract_key_preference_instances(self) -> pd.DataFrame:
        """Extract all instances where specific keys can be compared."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
            
            # Find which keys differ between the bigrams
            chosen_keys = set(chosen)
            unchosen_keys = set(unchosen)
            
            # Get keys that appear in one bigram but not the other
            keys_only_in_chosen = chosen_keys - unchosen_keys
            keys_only_in_unchosen = unchosen_keys - chosen_keys
            shared_keys = chosen_keys & unchosen_keys
            
            # For meaningful key comparison, we want cases where:
            # 1. Exactly one key differs, OR
            # 2. Keys are in similar positions (same finger/row)
            
            if len(keys_only_in_chosen) == 1 and len(keys_only_in_unchosen) == 1:
                # Case 1: Exactly one key differs
                chosen_unique_key = list(keys_only_in_chosen)[0]
                unchosen_unique_key = list(keys_only_in_unchosen)[0]
                
                # Check if keys are in similar positions (same finger or adjacent)
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'chosen_key': chosen_unique_key,
                    'unchosen_key': unchosen_unique_key,
                    'shared_context': ''.join(sorted(shared_keys)),
                    'chose_key': chosen_unique_key,
                    'slider_value': row.get('sliderValue', 0),
                    'chosen_key_finger': self.key_positions.get(chosen_unique_key, KeyPosition('', 0, 0, 0)).finger,
                    'unchosen_key_finger': self.key_positions.get(unchosen_unique_key, KeyPosition('', 0, 0, 0)).finger,
                    'chosen_key_row': self.key_positions.get(chosen_unique_key, KeyPosition('', 0, 0, 0)).row,
                    'unchosen_key_row': self.key_positions.get(unchosen_unique_key, KeyPosition('', 0, 0, 0)).row,
                    # Only bigram frequency controls
                    'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                    'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                })
        
        return pd.DataFrame(instances)

    def _keys_comparable_for_preference(self, key1: str, key2: str) -> bool:
        """Check if two keys are comparable for preference testing."""
        if key1 not in self.key_positions or key2 not in self.key_positions:
            return False
        
        pos1 = self.key_positions[key1]
        pos2 = self.key_positions[key2]
        
        # Keys are comparable if:
        # 1. Same finger (different rows), OR
        # 2. Adjacent fingers (same or adjacent rows), OR  
        # 3. Same row (different fingers)
        
        finger_diff = abs(pos1.finger - pos2.finger)
        row_diff = abs(pos1.row - pos2.row)
        
        if finger_diff == 0:  # Same finger
            return True
        elif finger_diff == 1 and row_diff <= 1:  # Adjacent fingers, similar rows
            return True
        elif row_diff == 0 and finger_diff <= 2:  # Same row, reasonably close fingers
            return True
        
        return False

    def _test_key_preference(self) -> Dict[str, Any]:
        """Test key preferences using instance-level analysis."""
        logger.info("Testing key preferences (instance-level)...")
        
        instances_df = self._extract_key_preference_instances()
        
        if instances_df.empty:
            return {'error': 'No key preference instances found'}
        
        logger.info(f"Found {len(instances_df)} key preference instances from {instances_df['user_id'].nunique()} users")
        
        # Analyze preferences for each key pair that appears frequently enough
        key_pair_results = {}
        key_preference_scores = {}
        
        # Group by key pairs
        instances_df['key_pair'] = instances_df.apply(
            lambda row: tuple(sorted([row['chosen_key'], row['unchosen_key']])), axis=1
        )
        
        key_pair_counts = instances_df['key_pair'].value_counts()
        total_significant_pairs = 0
        all_p_values = []
        
        for key_pair, count in key_pair_counts.items():
            if count >= 10:  # Minimum instances for analysis
                key1, key2 = key_pair
                pair_data = instances_df[instances_df['key_pair'] == key_pair].copy()
                
                # Code preference for key1 vs key2 (binary outcome)
                pair_data['chose_key1'] = (pair_data['chose_key'] == key1).astype(int)
                
                # Run simple proportion test for this key pair
                simple_test = self._simple_proportion_test(pair_data, 'chose_key1')
                
                # Store results
                key_pair_results[key_pair] = {
                    'key1': key1,
                    'key2': key2,
                    'n_instances': count,
                    'key1_preference_rate': simple_test['preference_rate'],
                    'p_value': simple_test['p_value'],
                    'significant': simple_test['significant'],
                    'effect_size': simple_test['effect_size'],
                    'interpretation': simple_test['interpretation']
                }
                
                all_p_values.append(simple_test['p_value'])
                if simple_test['significant']:
                    total_significant_pairs += 1
                
                # Update individual key scores
                key1_score = simple_test['preference_rate'] - 0.5  # Deviation from neutral
                key2_score = -key1_score
                
                key_preference_scores[key1] = key_preference_scores.get(key1, []) + [key1_score]
                key_preference_scores[key2] = key_preference_scores.get(key2, []) + [key2_score]
        
        # Calculate overall key rankings
        key_rankings = {}
        for key, scores in key_preference_scores.items():
            key_rankings[key] = {
                'mean_score': np.mean(scores),
                'n_comparisons': len(scores),
                'std_score': np.std(scores)
            }
        
        # Sort keys by preference score
        ranked_keys = sorted(key_rankings.items(), key=lambda x: x[1]['mean_score'], reverse=True)
        
        # Create overall summary test (combine all p-values using Fisher's method)
        overall_test = None
        if all_p_values:
            try:
                from scipy.stats import combine_pvalues
                overall_stat, overall_p = combine_pvalues(all_p_values, method='fisher')
                overall_test = {
                    'combined_p_value': overall_p,
                    'significant': overall_p < 0.05,
                    'method': 'fisher_combined',
                    'n_tests_combined': len(all_p_values)
                }
            except Exception as e:
                logger.warning(f"Failed to combine p-values: {e}")
                # Fallback: use most significant p-value
                overall_test = {
                    'min_p_value': min(all_p_values),
                    'significant': min(all_p_values) < 0.05,
                    'method': 'minimum_p',
                    'n_tests_combined': len(all_p_values)
                }
        
        return {
            'description': 'Individual key preferences (instance-level)',
            'method': 'pairwise_instance_level_analysis',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique() if not instances_df.empty else 0,
            'n_key_pairs': len(key_pair_results),
            'n_significant_pairs': total_significant_pairs,
            'key_pair_results': key_pair_results,
            'key_rankings': dict(key_rankings),
            'ranked_keys': ranked_keys,
            'instances_data': instances_df,
            'simple_test': overall_test,  # For validation framework
            'p_value': overall_test['combined_p_value'] if overall_test else 1.0,  # For validation
            'preference_rate': total_significant_pairs / len(key_pair_results) if key_pair_results else 0.0,  # For validation
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_key_rankings(ranked_keys, total_significant_pairs, len(key_pair_results))
        }

    def _are_matched_bigrams_for_key_test(self, bg1: str, bg2: str, key1: str, key2: str) -> bool:
        """Check if two bigrams are well-matched for testing key preference."""
        # Bigrams should differ primarily in the key being tested
        
        # Get positions of target keys
        key1_pos_bg1 = [i for i, char in enumerate(bg1) if char == key1]
        key2_pos_bg2 = [i for i, char in enumerate(bg2) if char == key2]
        
        # Must have the target key in same position
        if not key1_pos_bg1 or not key2_pos_bg2:
            return False
        
        if key1_pos_bg1[0] != key2_pos_bg2[0]:
            return False
        
        # The other character should be similar (ideally same)
        pos = key1_pos_bg1[0]
        other_pos = 1 - pos
        
        other_char_bg1 = bg1[other_pos]
        other_char_bg2 = bg2[other_pos]
        
        # Prefer exact match of other character
        if other_char_bg1 == other_char_bg2:
            return True
        
        # Accept if other characters are on same finger/row (similar context)
        if (other_char_bg1 in self.key_positions and 
            other_char_bg2 in self.key_positions):
            pos1 = self.key_positions[other_char_bg1]
            pos2 = self.key_positions[other_char_bg2]
            
            # Same finger or same row
            return pos1.finger == pos2.finger or pos1.row == pos2.row
        
        return False
    
    def _calculate_key_preference_indicator(self, bg1: str, bg2: str, key1: str, key2: str) -> float:
        """Calculate indicator for which bigram contains the preferred key."""
        # Return 1 if bg1 contains key1, 0 if bg1 contains key2
        # This encodes the hypothesis that key1 is preferred over key2
        
        if key1 in bg1 and key2 not in bg1:
            return 1.0
        elif key2 in bg1 and key1 not in bg1:
            return 0.0
        else:
            # Both keys present or neither - not a clean comparison
            return 0.5
    
    def _skip_obvious_comparison(self, key1: str, key2: str) -> bool:
        """Skip comparisons where outcome is obvious from biomechanical principles."""
        if key1 not in self.key_positions or key2 not in self.key_positions:
            return True
        
        pos1 = self.key_positions[key1]
        pos2 = self.key_positions[key2]
        
        # Skip if comparing across hands (outside our scope)
        # Skip if finger strength difference is extreme (pinky vs index)
        if abs(pos1.finger - pos2.finger) >= 3:
            return True
        
        return False
    
    def _build_key_ranking_from_pairwise(self, pairwise_results: Dict) -> Dict:
        """Build overall key ranking from pairwise comparison results."""
        if not pairwise_results:
            return {'ranked_keys': [], 'n_successful_comparisons': 0}
        
        # Extract significant preferences
        key_scores = defaultdict(float)
        n_comparisons = defaultdict(int)
        
        for (key1, key2), results in pairwise_results.items():
            if 'error' not in results and results.get('p_value', 1.0) < 0.05:
                coeff = results['coefficient']
                
                # Positive coefficient means key1 preferred over key2
                key_scores[key1] += coeff
                key_scores[key2] -= coeff
                n_comparisons[key1] += 1
                n_comparisons[key2] += 1
        
        # Average scores and sort
        averaged_scores = {}
        for key in key_scores:
            if n_comparisons[key] > 0:
                averaged_scores[key] = key_scores[key] / n_comparisons[key]
        
        ranked_keys = sorted(averaged_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'ranked_keys': ranked_keys,
            'n_successful_comparisons': len(pairwise_results),
            'raw_scores': dict(key_scores),
            'comparison_counts': dict(n_comparisons)
        }

    # =========================================================================
    # OBJECTIVE 3: HOME ROW
    # =========================================================================

    def _extract_home_row_instances(self) -> pd.DataFrame:
        """Extract all instances where bigrams with different home row involvement were compared."""
        instances = []
        home_keys = {'a', 's', 'd', 'f'}
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
                
            chosen_home_count = sum(1 for char in chosen if char in home_keys)
            unchosen_home_count = sum(1 for char in unchosen if char in home_keys)
            
            # Only include if different levels of home row involvement
            if chosen_home_count != unchosen_home_count:
                # Code preference: 1 = chose bigram with more home keys, 0 = chose bigram with fewer
                chose_more_home = 1 if chosen_home_count > unchosen_home_count else 0
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'chose_more_home': chose_more_home,
                    'chosen_home_count': chosen_home_count,
                    'unchosen_home_count': unchosen_home_count,
                    'home_difference': abs(chosen_home_count - unchosen_home_count),
                    'slider_value': row.get('sliderValue', 0),
                    # Only bigram frequency controls
                    'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                    'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                })
        
        return pd.DataFrame(instances)

    def _test_home_row_preference(self) -> Dict[str, Any]:
        """Test home row preference using instance-level analysis."""
        logger.info("Testing home row preference (instance-level)...")
        
        instances_df = self._extract_home_row_instances()
        
        if instances_df.empty:
            return {'error': 'No home row instances found'}
        
        logger.info(f"Found {len(instances_df)} home row instances from {instances_df['user_id'].nunique()} users")
        
        raw_home_rate = instances_df['chose_more_home'].mean()
        logger.info(f"Raw home row preference rate: {raw_home_rate:.1%}")
        simple_test = self._simple_proportion_test(instances_df, 'chose_more_home')
        logger.info(f"Simple test: {simple_test['interpretation']}")

        # Fit model  
        model_results = self._fit_instance_level_model(
            instances_df,
            'chose_more_home', 
            'home_row_instance_level'
        )

        simple_test = self._simple_proportion_test(instances_df, 'chose_more_home')
        
        return {
            'description': 'Home row preference (instance-level)',
            'method': 'instance_level_analysis',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique() if not instances_df.empty else 0,
            'model_results': model_results,
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'simple_test': simple_test,          
            'p_value': simple_test['p_value'],      
            'preference_rate': simple_test['preference_rate'],  
            'interpretation': model_results.get('interpretation', 'No interpretation available')
        }
        
    def _are_matched_bigrams_for_home_row_test(self, bg1: str, bg2: str) -> bool:
        """Check if bigrams are well-matched for home row testing."""
        # Should differ primarily in home row involvement, not other factors
        
        # Check finger usage similarity
        fingers1 = [self.key_positions[char].finger for char in bg1 if char in self.key_positions]
        fingers2 = [self.key_positions[char].finger for char in bg2 if char in self.key_positions]
        
        # Prefer same finger pattern
        if len(fingers1) == 2 and len(fingers2) == 2:
            return (fingers1[0] == fingers2[0] and fingers1[1] == fingers2[1]) or \
                   (fingers1[0] == fingers2[1] and fingers1[1] == fingers2[0])
        
        return False
    
    def _get_all_left_hand_bigrams(self) -> Set[str]:
        """Get all unique left-hand bigrams from the data."""
        bigrams = set()
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if len(chosen) == 2 and self._all_keys_in_left_hand(chosen):
                bigrams.add(chosen)
            if len(unchosen) == 2 and self._all_keys_in_left_hand(unchosen):
                bigrams.add(unchosen)
        
        return bigrams
    
    def _calculate_home_row_involvement(self, bg1: str, bg2: str, comparison_type: Tuple[str, str]) -> float:
        """Calculate home row involvement indicator."""
        home_keys = {'a', 's', 'd', 'f'}
        
        def count_home_keys(bigram):
            return sum(1 for char in bigram if char in home_keys)
        
        home1 = count_home_keys(bg1)
        home2 = count_home_keys(bg2)
        
        cat1, cat2 = comparison_type
        
        # Map categories to expected home key counts
        category_map = {
            'both_home': 2,
            'one_home': 1,
            'neither_home': 0
        }
        
        expected1 = category_map[cat1]
        expected2 = category_map[cat2]
        
        # Return 1 if bg1 matches cat1 and bg2 matches cat2
        if home1 == expected1 and home2 == expected2:
            return 1.0
        elif home1 == expected2 and home2 == expected1:
            return 0.0
        else:
            return 0.5  # Mixed or unclear
    
    def _aggregate_home_row_preference(self, home_row_results: Dict) -> Dict:
        """Aggregate home row preference across different comparison types."""
        coefficients = []
        weights = []
        
        for comparison_type, results in home_row_results.items():
            if 'error' not in results and 'coefficient' in results:
                coeff = results['coefficient']
                n_obs = results.get('n_observations', 1)
                
                coefficients.append(coeff)
                weights.append(n_obs)
        
        if not coefficients:
            return {'overall_home_preference': 0.0, 'confidence': 'low'}
        
        # Weighted average
        total_weight = sum(weights)
        overall_preference = sum(c * w for c, w in zip(coefficients, weights)) / total_weight
        
        # Assess confidence based on consistency
        consistency = 1.0 - np.std(coefficients) if len(coefficients) > 1 else 0.5
        
        return {
            'overall_home_preference': overall_preference,
            'individual_coefficients': coefficients,
            'consistency': consistency,
            'confidence': 'high' if consistency > 0.7 and len(coefficients) >= 2 else 'medium' if consistency > 0.5 else 'low'
        }

    # =========================================================================
    # OBJECTIVE 4: ROW SEPARATION
    # =========================================================================
    
    def _extract_row_separation_instances(self) -> pd.DataFrame:
        """Extract all instances where bigrams with different row separations were compared."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
                
            chosen_row_sep = self._calculate_row_separation(chosen)
            unchosen_row_sep = self._calculate_row_separation(unchosen)
            
            # Only include if different row separations
            if chosen_row_sep != unchosen_row_sep:
                # Code preference: 1 = chose bigram with smaller row separation, 0 = larger separation
                chose_smaller_separation = 1 if chosen_row_sep < unchosen_row_sep else 0
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'chose_smaller_separation': chose_smaller_separation,
                    'chosen_row_separation': chosen_row_sep,
                    'unchosen_row_separation': unchosen_row_sep,
                    'separation_difference': abs(chosen_row_sep - unchosen_row_sep),
                    'slider_value': row.get('sliderValue', 0),
                    # Only bigram frequency controls
                    'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                    'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                })
        
        return pd.DataFrame(instances)

    def _test_row_separation_preference(self) -> Dict[str, Any]:
        """Test row separation preference using instance-level analysis."""
        logger.info("Testing row separation preference (instance-level)...")
        
        instances_df = self._extract_row_separation_instances()
        
        if instances_df.empty:
            return {'error': 'No row separation instances found'}
        
        logger.info(f"Found {len(instances_df)} row separation instances from {instances_df['user_id'].nunique()} users")
        
        raw_row_rate = instances_df['chose_smaller_separation'].mean()
        logger.info(f"Raw smaller row separation preference rate: {raw_row_rate:.1%}")
        simple_test = self._simple_proportion_test(instances_df, 'chose_smaller_separation')
        logger.info(f"Simple test: {simple_test['interpretation']}")

        # Fit model  
        model_results = self._fit_instance_level_model(
            instances_df,
            'chose_smaller_separation', 
            'row_separation_instance_level'
        )
        
        # Additional analysis by separation difference magnitude
        separation_breakdown = instances_df.groupby('separation_difference').agg({
            'chose_smaller_separation': ['count', 'mean']
        }).round(3)
        
        simple_test = self._simple_proportion_test(instances_df, 'chose_smaller_separation')

        return {
            'description': 'Row separation preference (instance-level)',
            'method': 'instance_level_analysis',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique() if not instances_df.empty else 0,
            'model_results': model_results,
            'separation_breakdown': separation_breakdown,
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'simple_test': simple_test,          
            'p_value': simple_test['p_value'],      
            'preference_rate': simple_test['preference_rate'],  
            'interpretation': model_results.get('interpretation', 'No interpretation available')
        }
        
    def _extract_row_separation_comparisons(self, test: Tuple[str, str]) -> List[Tuple[str, str, Dict]]:
        """Extract comparisons testing row separation preferences."""
        comparisons = []
        
        # Categorize bigrams by row separation
        all_bigrams = self._get_all_left_hand_bigrams()
        
        row_categories = {
            'same_row': [],
            'one_row_apart': [],
            'two_rows_apart': []
        }
        
        for bigram in all_bigrams:
            separation = self._calculate_row_separation(bigram)
            
            if separation == 0:
                row_categories['same_row'].append(bigram)
            elif separation == 1:
                row_categories['one_row_apart'].append(bigram)
            elif separation == 2:
                row_categories['two_rows_apart'].append(bigram)
        
        # Extract comparisons for the specified test
        cat1, cat2 = test
        
        for bg1 in row_categories[cat1]:
            for bg2 in row_categories[cat2]:
                if self._are_matched_bigrams_for_row_test(bg1, bg2):
                    comparison_data = self._extract_pairwise_comparison(bg1, bg2)
                    
                    if comparison_data['total'] >= self.config.get('min_comparisons', 10):
                        comparisons.append((bg1, bg2, comparison_data))
        
        return comparisons
    
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
    
    def _are_matched_bigrams_for_row_test(self, bg1: str, bg2: str) -> bool:
        """Check if bigrams are well-matched for row separation testing."""
        # Should have similar finger usage patterns
        if len(bg1) != 2 or len(bg2) != 2:
            return False
        
        fingers1 = [self.key_positions.get(char, KeyPosition('', 0, 0, 0)).finger for char in bg1]
        fingers2 = [self.key_positions.get(char, KeyPosition('', 0, 0, 0)).finger for char in bg2]
        
        # Similar finger pattern (same or symmetric)
        return (fingers1[0] == fingers2[0] and fingers1[1] == fingers2[1]) or \
               (fingers1[0] == fingers2[1] and fingers1[1] == fingers2[0]) or \
               abs(fingers1[0] - fingers2[0]) <= 1 and abs(fingers1[1] - fingers2[1]) <= 1
    
    def _calculate_row_separation_indicator(self, bg1: str, bg2: str, test: Tuple[str, str]) -> float:
        """Calculate row separation indicator for the test."""
        sep1 = self._calculate_row_separation(bg1)
        sep2 = self._calculate_row_separation(bg2)
        
        # Map test categories to expected separations
        separation_map = {
            'same_row': 0,
            'one_row_apart': 1,
            'two_rows_apart': 2
        }
        
        expected1 = separation_map[test[0]]
        expected2 = separation_map[test[1]]
        
        if sep1 == expected1 and sep2 == expected2:
            return 1.0
        elif sep1 == expected2 and sep2 == expected1:
            return 0.0
        else:
            return 0.5
    
    def _build_row_preference_function(self, row_separation_results: Dict) -> Dict:
        """Build row preference function from test results."""
        # Extract preferences between different row separations
        preferences = {}
        
        for test, results in row_separation_results.items():
            if 'error' not in results and 'coefficient' in results:
                cat1, cat2 = test
                coeff = results['coefficient']
                p_val = results.get('p_value', 1.0)
                
                if p_val < 0.05:  # Significant result
                    # Positive coefficient means cat1 preferred over cat2
                    preferences[f"{cat1}_vs_{cat2}"] = coeff
        
        # Build ordering based on preferences
        ordering = self._infer_row_preference_ordering(preferences)
        
        return {
            'pairwise_preferences': preferences,
            'row_preference_ordering': ordering,
            'preference_strength': np.mean([abs(p) for p in preferences.values()]) if preferences else 0.0
        }
    
    def _infer_row_preference_ordering(self, preferences: Dict[str, float]) -> List[str]:
        """Infer row preference ordering from pairwise preferences."""
        # Simple ordering inference - can be made more sophisticated
        row_scores = {'same_row': 0, 'one_row_apart': 0, 'two_rows_apart': 0}
        
        for test_name, coeff in preferences.items():
            parts = test_name.split('_vs_')
            if len(parts) == 2:
                cat1, cat2 = parts[0], parts[1]
                
                if coeff > 0:  # cat1 preferred
                    row_scores[cat1] += abs(coeff)
                    row_scores[cat2] -= abs(coeff)
                else:  # cat2 preferred
                    row_scores[cat2] += abs(coeff)
                    row_scores[cat1] -= abs(coeff)
        
        return sorted(row_scores.keys(), key=lambda x: row_scores[x], reverse=True)

    # =========================================================================
    # OBJECTIVE 5: COLUMN SEPARATION
    # =========================================================================
    
    def _extract_column_separation_instances(self) -> pd.DataFrame:
        """Extract column separation instances for different-finger bigrams only."""
        instances = []
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if not (self._all_keys_in_left_hand(chosen) and self._all_keys_in_left_hand(unchosen)):
                continue
                
            # CRITICAL: Exclude same-finger bigrams from column analysis
            if self._same_finger_bigram(chosen) or self._same_finger_bigram(unchosen):
                continue
                
            chosen_col_sep = self._calculate_column_separation(chosen)
            unchosen_col_sep = self._calculate_column_separation(unchosen)
            chosen_row_sep = self._calculate_row_separation(chosen)
            unchosen_row_sep = self._calculate_row_separation(unchosen)
            
            # Only include if different column separations
            if chosen_col_sep != unchosen_col_sep:
                
                # Determine row context for meaningful comparisons
                if chosen_row_sep == 0 and unchosen_row_sep == 0:
                    context = "same_row"
                    context_description = "within same row"
                elif chosen_row_sep == unchosen_row_sep and chosen_row_sep > 0:
                    context = f"{chosen_row_sep}_rows_apart"
                    context_description = f"{chosen_row_sep} row(s) apart"
                else:
                    # Skip mixed row contexts - not meaningful for column analysis
                    continue
                
                # Code preference: 1 = chose smaller column separation, 0 = larger
                chose_smaller_col_separation = 1 if chosen_col_sep < unchosen_col_sep else 0
                
                # Determine specific comparison type
                smaller_sep = min(chosen_col_sep, unchosen_col_sep)
                larger_sep = max(chosen_col_sep, unchosen_col_sep)
                
                if smaller_sep == 1 and larger_sep == 2:
                    comparison_type = "adjacent_vs_2apart"
                    comparison_desc = "adjacent (1 apart) vs 2 columns apart"
                elif smaller_sep == 1 and larger_sep == 3:
                    comparison_type = "adjacent_vs_3apart"
                    comparison_desc = "adjacent (1 apart) vs 3 columns apart"
                elif smaller_sep == 2 and larger_sep == 3:
                    comparison_type = "2apart_vs_3apart"
                    comparison_desc = "2 columns apart vs 3 columns apart"
                else:
                    comparison_type = "other"
                    comparison_desc = f"{smaller_sep} vs {larger_sep} columns apart"
                
                instances.append({
                    'user_id': row['user_id'],
                    'chosen_bigram': chosen,
                    'unchosen_bigram': unchosen,
                    'chose_smaller_col_separation': chose_smaller_col_separation,
                    'chosen_col_separation': chosen_col_sep,
                    'unchosen_col_separation': unchosen_col_sep,
                    'col_separation_difference': abs(chosen_col_sep - unchosen_col_sep),
                    'row_context': context,
                    'context_description': context_description,
                    'comparison_type': comparison_type,
                    'comparison_description': comparison_desc,
                    'slider_value': row.get('sliderValue', 0),
                    'log_chosen_bigram_freq': np.log(self.english_bigram_frequencies.get(chosen, 1e-5) + 1e-6),
                    'log_unchosen_bigram_freq': np.log(self.english_bigram_frequencies.get(unchosen, 1e-5) + 1e-6),
                })
        
        return pd.DataFrame(instances)

    def _test_column_separation_preference(self) -> Dict[str, Any]:
        """Test column separation preference for different-finger bigrams only."""
        logger.info("Testing column separation preference (different-finger bigrams only)...")
        
        instances_df = self._extract_column_separation_instances()
        
        if instances_df.empty:
            return {'error': 'No valid column separation instances found (after excluding same-finger bigrams)'}
        
        logger.info(f"Found {len(instances_df)} column separation instances from {instances_df['user_id'].nunique()} users")
        logger.info("(Excluding same-finger bigrams to isolate spatial column effects)")
        
        # Overall analysis
        simple_test = self._simple_proportion_test(instances_df, 'chose_smaller_col_separation')
        model_results = self._fit_instance_level_model(
            instances_df,
            'chose_smaller_col_separation', 
            'column_separation_different_fingers_only'
        )
        
        # Analysis by row context
        context_results = {}
        context_interpretations = {}
        
        for context in instances_df['row_context'].unique():
            context_data = instances_df[instances_df['row_context'] == context]
            if len(context_data) >= 20:
                context_simple = self._simple_proportion_test(context_data, 'chose_smaller_col_separation')
                context_model = self._fit_instance_level_model(
                    context_data,
                    'chose_smaller_col_separation',
                    f'column_separation_{context}_different_fingers'
                )
                
                context_results[context] = {
                    'simple_test': context_simple,
                    'model_results': context_model,
                    'n_instances': len(context_data),
                    'interpretation': context_simple['interpretation']
                }
                
                # Create specific interpretation for this context
                pref_rate = context_simple['preference_rate']
                context_desc = context_data['context_description'].iloc[0]
                
                # Find most common comparison type in this context
                common_comparison = context_data['comparison_description'].mode()
                if len(common_comparison) > 0:
                    comp_desc = common_comparison.iloc[0]
                    context_interpretations[context] = f"{context_desc}: {pref_rate:.1%} prefer smaller column separation ({comp_desc})"
                else:
                    context_interpretations[context] = f"{context_desc}: {pref_rate:.1%} prefer smaller column separation"
            else:
                logger.warning(f"Insufficient data for {context} context ({len(context_data)} instances)")
        
        # Analysis by specific comparison types
        comparison_results = {}
        for comp_type in instances_df['comparison_type'].unique():
            comp_data = instances_df[instances_df['comparison_type'] == comp_type]
            if len(comp_data) >= 15:  # Slightly lower threshold for specific comparisons
                comp_simple = self._simple_proportion_test(comp_data, 'chose_smaller_col_separation')
                comp_desc = comp_data['comparison_description'].iloc[0]
                
                comparison_results[comp_type] = {
                    'simple_test': comp_simple,
                    'n_instances': len(comp_data),
                    'description': comp_desc,
                    'preference_rate': comp_simple['preference_rate'],
                    'p_value': comp_simple['p_value'],
                    'interpretation': f"{comp_desc}: {comp_simple['preference_rate']:.1%} prefer smaller separation"
                }
        
        return {
            'description': 'Column separation preference (different-finger bigrams only)',
            'method': 'spatial_column_analysis_no_same_finger',
            'n_instances': len(instances_df),
            'n_users': instances_df['user_id'].nunique(),
            'simple_test': simple_test,
            'model_results': model_results,
            'p_value': simple_test['p_value'],
            'preference_rate': simple_test['preference_rate'],
            'context_results': context_results,
            'context_interpretations': context_interpretations,
            'comparison_results': comparison_results,
            'instances_data': instances_df,
            'normalization_range': (0.0, 1.0),
            'interpretation': self._interpret_clean_column_results(simple_test, context_interpretations, comparison_results)
        }

    def _interpret_clean_column_results(self, overall_test: Dict, context_interp: Dict, comparison_results: Dict) -> str:
        """Interpret clean column separation results."""
        overall_rate = overall_test['preference_rate']
        
        # Build detailed interpretation
        interpretation_parts = [f"Overall (different-finger only): {overall_rate:.1%} prefer smaller column separation"]
        
        # Add context-specific results
        if context_interp:
            context_summary = "; ".join([f"{ctx}: {interp.split(': ')[1]}" for ctx, interp in context_interp.items()])
            interpretation_parts.append(f"By context - {context_summary}")
        
        # Add specific comparison results
        if comparison_results:
            significant_comparisons = []
            for comp_type, comp_data in comparison_results.items():
                if comp_data.get('p_value', 1.0) < 0.05:
                    rate = comp_data['preference_rate']
                    desc = comp_data['description']
                    significant_comparisons.append(f"{desc}: {rate:.1%}")
            
            if significant_comparisons:
                interpretation_parts.append(f"Significant specific comparisons: {'; '.join(significant_comparisons)}")
        
        return ". ".join(interpretation_parts)

    def _bigram_matches_context(self, bigram: str, context: str) -> bool:
        """Check if bigram matches the specified context."""
        if context == 'same_row_context':
            return self._calculate_row_separation(bigram) == 0
        elif context == 'different_row_context':
            return self._calculate_row_separation(bigram) > 0
        
        return True
    
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
    
    def _are_matched_bigrams_for_column_test(self, bg1: str, bg2: str, context: str) -> bool:
        """Check if bigrams are well-matched for column separation testing."""
        # Should maintain same context and similar row patterns
        return (self._bigram_matches_context(bg1, context) and 
                self._bigram_matches_context(bg2, context) and
                self._calculate_row_separation(bg1) == self._calculate_row_separation(bg2))
    
    def _calculate_column_separation_indicator(self, bg1: str, bg2: str, test: Tuple[str, str], context: str) -> float:
        """Calculate column separation indicator."""
        col1 = self._calculate_column_separation(bg1)
        col2 = self._calculate_column_separation(bg2)
        
        cat1, cat2 = test
        
        def matches_category(col_sep, category):
            if category == 'adjacent_cols':
                return col_sep <= 1
            else:  # 'separated_cols'
                return col_sep > 1
        
        if matches_category(col1, cat1) and matches_category(col2, cat2):
            return 1.0
        elif matches_category(col1, cat2) and matches_category(col2, cat1):
            return 0.0
        else:
            return 0.5
    
    def _build_column_preference_function(self, column_separation_results: Dict) -> Dict:
        """Build column preference function from test results."""
        context_preferences = {}
        
        for context, results in column_separation_results.items():
            if 'error' not in results and 'coefficient' in results:
                context_preferences[context] = {
                    'coefficient': results['coefficient'],
                    'p_value': results.get('p_value', 1.0),
                    'significant': results.get('p_value', 1.0) < 0.05
                }
        
        return {
            'context_preferences': context_preferences,
            'context_dependent': len([p for p in context_preferences.values() if p['significant']]) > 1
        }

    # =========================================================================
    # HELPER METHODS FOR STATISTICAL ANALYSIS
    # =========================================================================
    
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
    
    def _get_bigrams_containing_key(self, key: str) -> List[str]:
        """Get all bigrams in the data containing the specified key."""
        bigrams = set()
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            if key in chosen and len(chosen) == 2:
                bigrams.add(chosen)
            if key in unchosen and len(unchosen) == 2:
                bigrams.add(unchosen)
        
        return [bg for bg in bigrams if self._all_keys_in_left_hand(bg)]
    
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
        from scipy.stats import norm
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
            'biomechanical_consistency': self._check_biomechanical_consistency(results),
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
            'different_finger': 0.05,  # 5% preference difference is meaningful
            'key_preference': 0.05,    
            'home_row': 0.05,          
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
    
    def _check_biomechanical_consistency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if results are consistent with known biomechanical principles."""
        consistency_checks = {}
        
        # Existing checks...
        if 'key_preference' in results:
            consistency_checks['finger_strength'] = self._check_finger_strength_hierarchy(
                results['key_preference']
            )
        
        if 'home_row' in results:
            consistency_checks['home_row'] = self._check_home_row_advantage(
                results['home_row']
            )
        
        if 'row_separation' in results:
            consistency_checks['row_reach'] = self._check_row_reach_principles(
                results['row_separation']
            )
        
        if 'different_finger' in results:
            consistency_checks['finger_independence'] = self._check_finger_independence(
                results['different_finger']
            )
        
        # NEW: Home row redundancy test
        if 'key_preference' in results and 'home_row' in results:
            consistency_checks['home_row_redundancy'] = self._check_home_row_redundancy(
                results['key_preference'], results['home_row']
            )
        
        return consistency_checks

    def _check_home_row_redundancy(self, key_pref_results: Dict[str, Any], 
                                home_row_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test if home row advantage is captured by key preferences."""
        
        if 'ranked_keys' not in key_pref_results:
            return {'error': 'No key rankings available for redundancy test'}
        
        ranked_keys = key_pref_results['ranked_keys']
        home_row_keys = {'a', 's', 'd', 'f'}
        
        # Get top-ranked keys
        if len(ranked_keys) < 4:
            return {'error': 'Insufficient key rankings for redundancy test'}
        
        top_4_keys = set([key for key, _ in ranked_keys[:4]])
        top_6_keys = set([key for key, _ in ranked_keys[:6]])
        
        # Calculate overlap
        home_in_top_4 = len(home_row_keys & top_4_keys)
        home_in_top_6 = len(home_row_keys & top_6_keys) 
        redundancy_top_4 = home_in_top_4 / 4  # Fraction of home row in top 4
        redundancy_top_6 = home_in_top_6 / 4  # Fraction of home row in top 6
        
        # Get home row preference strength
        home_row_strength = 0.0
        if ('simple_test' in home_row_results and 
            isinstance(home_row_results['simple_test'], dict)):
            pref_rate = home_row_results['simple_test'].get('preference_rate', 0.5)
            home_row_strength = abs(pref_rate - 0.5)  # Deviation from neutral
        
        # Determine redundancy level
        if redundancy_top_4 >= 0.75:  # 3+ home row keys in top 4
            redundancy_level = 'high'
            recommendation = 'Consider removing home row objective - captured by key preferences'
        elif redundancy_top_6 >= 0.75:  # 3+ home row keys in top 6
            redundancy_level = 'moderate'  
            recommendation = 'Home row partially captured by key preferences'
        else:
            redundancy_level = 'low'
            recommendation = 'Keep both objectives - measuring different aspects'
        
        # Detailed analysis
        home_key_ranks = {}
        for i, (key, score) in enumerate(ranked_keys):
            if key in home_row_keys:
                home_key_ranks[key] = {'rank': i+1, 'score': score}
        
        return {
            'redundancy_level': redundancy_level,
            'home_in_top_4': home_in_top_4,
            'home_in_top_6': home_in_top_6,
            'redundancy_fraction_top_4': redundancy_top_4,
            'redundancy_fraction_top_6': redundancy_top_6,
            'home_row_preference_strength': home_row_strength,
            'individual_home_key_ranks': home_key_ranks,
            'recommendation': recommendation,
            'interpretation': f"{redundancy_level.title()} redundancy: {home_in_top_4}/4 home keys in top 4 ranks"
        }
    
    def _check_finger_strength_hierarchy(self, key_pref_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if key preferences follow expected finger strength hierarchy."""
        if 'key_ranking' not in key_pref_results or 'ranked_keys' not in key_pref_results['key_ranking']:
            return {'error': 'No key ranking available'}
        
        ranked_keys = key_pref_results['key_ranking']['ranked_keys']
        
        # Expected finger strength: index > middle > ring > pinky
        finger_map = {
            'f': 4, 'd': 3, 's': 2, 'a': 1,  # Home row
            'r': 4, 'e': 3, 'w': 2, 'q': 1,  # Upper row
            'v': 4, 'c': 3, 'x': 2, 'z': 1   # Lower row
        }
        
        # Calculate correlation with expected strength
        key_scores = {key: score for key, score in ranked_keys}
        keys_with_fingers = [(key, finger_map.get(key, 0), score) 
                           for key, score in key_scores.items() 
                           if key in finger_map]
        
        if len(keys_with_fingers) < 4:
            return {'insufficient_data': True}
        
        finger_strengths = [finger for _, finger, _ in keys_with_fingers]
        preference_scores = [score for _, _, score in keys_with_fingers]
        
        correlation, p_value = stats.spearmanr(finger_strengths, preference_scores)
        
        return {
            'correlation_with_finger_strength': correlation,
            'p_value': p_value,
            'consistent': correlation > 0.3 and p_value < 0.05,
            'interpretation': self._interpret_finger_consistency(correlation, p_value)
        }
    
    def _interpret_finger_consistency(self, correlation: float, p_value: float) -> str:
        """Interpret finger strength consistency results."""
        if p_value >= 0.05:
            return "No significant correlation with finger strength hierarchy"
        elif correlation > 0.5:
            return "Strong consistency with finger strength hierarchy"
        elif correlation > 0.3:
            return "Moderate consistency with finger strength hierarchy"
        elif correlation > 0:
            return "Weak consistency with finger strength hierarchy"
        else:
            return "Results contradict expected finger strength hierarchy"
    
    def _check_home_row_advantage(self, home_row_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if home row shows expected mechanical advantage."""
        
        # Check simple test results first
        if 'simple_test' in home_row_results and isinstance(home_row_results['simple_test'], dict):
            pref_rate = home_row_results['simple_test'].get('preference_rate')
            if isinstance(pref_rate, (int, float)):
                expected_positive = pref_rate > 0.5  # Home row should be preferred
                return {
                    'home_preference_rate': pref_rate,
                    'shows_expected_advantage': expected_positive,
                    'magnitude': abs(pref_rate - 0.5),
                    'interpretation': f"Home row shows {'expected' if expected_positive else 'unexpected'} preference direction"
                }
        
        return {'error': 'No valid home row preference data available'}
    
    def _check_row_reach_principles(self, row_sep_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if row separation follows reach difficulty principles."""
        if 'preference_function' not in row_sep_results:
            return {'error': 'No row preference function available'}
        
        pref_function = row_sep_results['preference_function']
        
        if 'row_preference_ordering' not in pref_function:
            return {'error': 'No row preference ordering available'}
        
        ordering = pref_function['row_preference_ordering']
        
        # Expected: same_row > one_row_apart > two_rows_apart
        expected_order = ['same_row', 'one_row_apart', 'two_rows_apart']
        
        # Check if actual order matches expected
        matches_expected = ordering == expected_order
        
        return {
            'actual_ordering': ordering,
            'expected_ordering': expected_order,
            'matches_biomechanical_expectation': matches_expected,
            'interpretation': self._interpret_row_ordering(ordering, expected_order)
        }
    
    def _interpret_row_ordering(self, actual: List[str], expected: List[str]) -> str:
        """Interpret row ordering results."""
        if actual == expected:
            return "Perfect match with biomechanical expectations"
        elif actual[0] == expected[0]:
            return "Correctly identifies same-row preference, but other ordering differs"
        else:
            return "Does not match biomechanical expectations"
    
    def _check_finger_independence(self, finger_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if different-finger preference is detected."""
        
        # Check simple test results
        if 'simple_test' in finger_results and isinstance(finger_results['simple_test'], dict):
            pref_rate = finger_results['simple_test'].get('preference_rate')
            p_val = finger_results['simple_test'].get('p_value', 1.0)
            
            if isinstance(pref_rate, (int, float)):
                expected_positive = pref_rate > 0.5  # Different fingers should be preferred
                significant = p_val < 0.05
                
                return {
                    'preference_rate': pref_rate,
                    'p_value': p_val,
                    'shows_expected_independence_preference': expected_positive and significant,
                    'interpretation': self._interpret_finger_independence(pref_rate, p_val, expected_positive, significant)
                }
        
        return {'error': 'No valid finger independence data available'}

    def _interpret_finger_independence(self, pref_rate: float, p_val: float, 
                                    expected_pos: bool, significant: bool) -> str:
        """Interpret finger independence results."""
        if not significant:
            return "No significant finger independence preference detected"
        elif expected_pos:
            return f"Confirms expected preference for finger independence ({pref_rate:.1%} preference rate)"
        else:
            return f"Unexpected preference for same-finger combinations ({pref_rate:.1%} preference rate)"
    
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
    # ENHANCED REPORTING AND SUMMARY METHODS
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
            'biomechanical_consistency': self._count_consistent_objectives(validation),
            'recommendations': validation.get('overall_validity', {}).get('recommendations', [])
        }
    
    def _count_consistent_objectives(self, validation: Dict[str, Any]) -> int:
        """Count objectives consistent with biomechanical principles."""
        consistent_count = 0
        biomech = validation.get('biomechanical_consistency', {})
        
        for check_name, check_results in biomech.items():
            if isinstance(check_results, dict):
                if (check_results.get('consistent', False) or 
                    check_results.get('shows_expected_advantage', False) or
                    check_results.get('matches_biomechanical_expectation', False) or
                    check_results.get('shows_expected_independence_preference', False)):
                    consistent_count += 1
        
        return consistent_count

    # =========================================================================
    # INTERPRETATION METHODS
    # =========================================================================
    
    def _interpret_finger_difference_result(self, model_results: Dict) -> str:
        """Interpret finger difference results."""
        if 'error' in model_results:
            return "Analysis failed - insufficient data"
        
        coeff = model_results['coefficient']
        p_value = model_results['p_value']
        
        if p_value < 0.05:
            if coeff > 0:
                strength = "strong" if abs(coeff) > 0.1 else "moderate" if abs(coeff) > 0.05 else "weak"
                return f"Significant preference for different-finger bigrams (coeff={coeff:.3f}, {strength} effect)"
            else:
                strength = "strong" if abs(coeff) > 0.1 else "moderate" if abs(coeff) > 0.05 else "weak"
                return f"Significant preference for same-finger bigrams (coeff={coeff:.3f}, {strength} effect)"
        else:
            return f"No significant finger preference detected (p={p_value:.3f})"
    
    def _interpret_key_preference_results(self, key_ranking: Dict) -> str:
        """Interpret key preference results."""
        if not key_ranking or 'ranked_keys' not in key_ranking:
            return "Key preference analysis failed - insufficient data"
        
        ranked_keys = key_ranking['ranked_keys']
        if not ranked_keys:
            return "No key preferences detected"
        
        top_3 = [key.upper() for key, _ in ranked_keys[:3]]
        bottom_3 = [key.upper() for key, _ in ranked_keys[-3:]]
        
        return (f"Top preferred keys: {', '.join(top_3)}. "
                f"Least preferred keys: {', '.join(bottom_3)}. "
                f"Based on {key_ranking['n_successful_comparisons']} pairwise comparisons.")

    def _interpret_key_rankings(self, ranked_keys: list, n_significant: int, n_total: int) -> str:
        """Interpret key preference rankings with significance info."""
        if not ranked_keys:
            return "No key preferences detected"
        
        top_3 = [key.upper() for key, _ in ranked_keys[:3]]
        bottom_3 = [key.upper() for key, _ in ranked_keys[-3:]]
        
        significance_info = f"{n_significant}/{n_total} significant pairwise comparisons"
        
        return (f"Top preferred keys: {', '.join(top_3)}. "
                f"Least preferred keys: {', '.join(bottom_3)}. "
                f"Based on {significance_info}.")
    
    def _interpret_home_row_results(self, home_preference: Dict) -> str:
        """Interpret home row results."""
        if not home_preference or 'overall_home_preference' not in home_preference:
            return "Home row analysis failed - insufficient data"
        
        coeff = home_preference['overall_home_preference']
        
        if abs(coeff) < 0.05:
            return f"No significant home row preference detected (coeff={coeff:.3f})"
        elif coeff > 0:
            strength = "strong" if coeff > 0.15 else "moderate" if coeff > 0.08 else "weak"
            return f"Significant home row preference detected (coeff={coeff:.3f}, {strength} effect)"
        else:
            strength = "strong" if abs(coeff) > 0.15 else "moderate" if abs(coeff) > 0.08 else "weak"
            return f"Significant avoidance of home row detected (coeff={coeff:.3f}, {strength} effect)"
    
    def _interpret_row_separation_results(self, row_function: Dict) -> str:
        """Interpret row separation results."""
        if not row_function or 'row_preference_ordering' not in row_function:
            return "Row separation analysis failed - insufficient data"
        
        ordering = row_function['row_preference_ordering']
        if not ordering:
            return "No row separation preferences detected"
        
        return (f"Row preference order: {' > '.join(ordering)}. "
                f"Based on {len(row_function.get('pairwise_preferences', {}))} pairwise tests.")
    
    def _interpret_column_separation_results(self, column_function: Dict) -> str:
        """Interpret column separation results."""
        if not column_function or 'context_preferences' not in column_function:
            return "Column separation analysis failed - insufficient data"
        
        context_prefs = column_function['context_preferences']
        significant_contexts = [ctx for ctx, pref in context_prefs.items() if pref.get('significant', False)]
        
        if not significant_contexts:
            return "No significant column separation preferences detected"
        
        context_dependent = column_function.get('context_dependent', False)
        
        if context_dependent:
            return f"Context-dependent column preferences detected in {len(significant_contexts)} contexts"
        else:
            return f"Consistent column preferences across contexts ({len(significant_contexts)} significant)"
        
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
    # REPORT GENERATION AND VISUALIZATION
    # =========================================================================
    
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
            f"Biomechanically consistent objectives: {summary.get('biomechanical_consistency', 'unknown')}",
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
            'different_finger': {
                'title': 'DIFFERENT FINGER PREFERENCE',
                'what_compared': 'Bigrams using same finger (e.g., "qa", "ws") vs. bigrams using different fingers (e.g., "as", "sf")',
                'method': 'Instance-level analysis: For each comparison where one bigram used same finger and another used different fingers, recorded which was chosen. Applied simple proportion test to determine if different-finger bigrams are preferred.',
                'frequency_control': 'Controlled for English bigram frequencies to isolate typing mechanics from language familiarity',
                'practical_meaning': 'Measures finger independence preference - whether users prefer alternating between fingers vs. repeated finger use'
            },
            'key_preference': {
                'title': 'INDIVIDUAL KEY PREFERENCES',
                'what_compared': 'All pairwise comparisons between individual keys (48 pairs total from 12 left-hand keys)',
                'method': 'Pairwise analysis: For each key pair with sufficient data (≥10 comparisons), not including pinky/index comparisons or outside left finger-columns, determined preference rate using proportion tests. Combined results using Fisher\'s method for overall significance.',
                'frequency_control': 'Controlled for English bigram frequencies in contexts where keys appeared',
                'practical_meaning': 'Ranks individual keys by typing preference, likely reflecting finger strength, reach comfort, and motor control ease'
            },
            'home_row': {
                'title': 'HOME ROW ADVANTAGE',
                'what_compared': 'Bigrams with different levels of home row involvement: both keys on home row (ASDF) vs. one key on home row vs. neither key on home row',
                'method': 'Instance-level analysis: For each comparison between bigrams with different home row counts, recorded which was chosen. Tested if bigrams with more home row keys are preferred.',
                'frequency_control': 'Controlled for English bigram frequencies to isolate ergonomic from linguistic preferences',
                'practical_meaning': 'Measures ergonomic advantage of home row position - the rest position with optimal finger positioning and minimal hand movement'
            },
            'row_separation': {
                'title': 'ROW SEPARATION PREFERENCES',
                'what_compared': 'Bigrams requiring different row movements: same row (no movement) vs. one row apart (small reach) vs. two rows apart (large reach)',
                'method': 'Instance-level analysis: For each comparison between bigrams with different row separations, recorded which was chosen. Tested preference for smaller vs. larger row movements.',
                'frequency_control': 'Controlled for English bigram frequencies to isolate motor control from linguistic preferences',
                'practical_meaning': 'Measures typing effort preferences related to vertical finger movement - strong preference (81.1%) for smaller row separations indicates reaching difficulty'
            },
                'column_separation': {
                'title': 'COLUMN SEPARATION PREFERENCES (DIFFERENT FINGERS ONLY)',
                'what_compared': 'Different-finger bigrams with different horizontal spacing: adjacent columns (1 apart) vs separated columns (2-3 apart). Excludes same-finger bigrams to isolate spatial effects.',
                'methodology': 'Instance-level analysis: For each comparison between different-finger bigrams with different column separations, recorded which was chosen. Analyzed by row context (same row vs different rows) and specific separation distances.',
                'frequency_control': 'Controlled for English bigram frequencies to isolate spatial motor preferences from linguistic familiarity',
                'practical_meaning': 'Measures pure horizontal reach preferences independent of finger independence effects - preference for closer vs more distant finger combinations'
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
        
        # Biomechanical consistency
        biomech_validation = validation.get('biomechanical_consistency', {})
        if biomech_validation:
            report_lines.extend([
                "Biomechanical Consistency:",
                "-" * 25,
                "Checks whether results align with expected typing biomechanics:",
                ""
            ])
            for check_name, check_results in biomech_validation.items():
                if isinstance(check_results, dict) and 'interpretation' in check_results:
                    report_lines.append(f"{check_name}: {check_results['interpretation']}")
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
                    
    def _add_finger_results_to_report(self, results: Dict[str, Any], report_lines: List[str]) -> None:
        """Add detailed finger difference results to report."""
        if 'debug_info' in results and 'sample_comparisons' in results['debug_info']:
            debug_info = results['debug_info']
            report_lines.extend([
                "Sample Comparisons Found:",
            ])
            
            for i, comp in enumerate(debug_info['sample_comparisons'][:3], 1):
                report_lines.extend([
                    f"  {i}. {comp['bigram1'].upper()} vs {comp['bigram2'].upper()}:",
                    f"     Same finger: {comp['bigram1_same_finger']} vs {comp['bigram2_same_finger']}",
                    f"     Observations: {comp['total_observations']} ({comp['bigram1_wins']} chose {comp['bigram1'].upper()})",
                ])
            report_lines.append("")
    
    def _add_key_preference_results_to_report(self, results: Dict[str, Any], report_lines: List[str]) -> None:
        """Add key preference results to report."""
        if 'key_ranking' in results and 'ranked_keys' in results['key_ranking']:
            ranked_keys = results['key_ranking']['ranked_keys']
            if ranked_keys:
                report_lines.extend([
                    "Key Preference Ranking:",
                    "  " + " > ".join([f"{key.upper()}({score:.3f})" for key, score in ranked_keys[:6]])
                ])
            report_lines.append("")
    
    def _add_home_row_results_to_report(self, results: Dict[str, Any], report_lines: List[str]) -> None:
        """Add home row results to report."""
        if 'overall_home_preference' in results:
            home_pref = results['overall_home_preference']
            if isinstance(home_pref, dict) and 'overall_home_preference' in home_pref:
                coeff = home_pref['overall_home_preference']
                confidence = home_pref.get('confidence', 'unknown')
                report_lines.extend([
                    f"Home Row Preference Coefficient: {coeff:.4f}",
                    f"Confidence Level: {confidence}",
                    ""
                ])
    
    def _add_row_separation_results_to_report(self, results: Dict[str, Any], report_lines: List[str]) -> None:
        """Add row separation results to report."""
        if 'preference_function' in results and 'row_preference_ordering' in results['preference_function']:
            ordering = results['preference_function']['row_preference_ordering']
            if ordering:
                report_lines.extend([
                    f"Row Preference Order: {' > '.join(ordering)}",
                    ""
                ])
    
    def _add_column_separation_results_to_report(self, results: Dict[str, Any], report_lines: List[str]) -> None:
        """Add column separation results to report."""
        if 'preference_function' in results and 'context_dependent' in results['preference_function']:
            context_dep = results['preference_function']['context_dependent']
            report_lines.extend([
                f"Context Dependent: {'Yes' if context_dep else 'No'}",
                ""
            ])
    
    def _create_visualizations(self, enhanced_results: Dict[str, Any], output_folder: str) -> None:
        """Create visualizations for MOO objectives and validation results."""
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({'font.size': 10, 'figure.dpi': self.config.get('figure_dpi', 300)})

        results = enhanced_results['objectives']
        validation = enhanced_results['validation']
        
        # Create summary visualization
        self._plot_objectives_summary(results, validation, output_folder)
        
        # Create validation summary plot
        self._plot_validation_summary(validation, output_folder)
        
        # Create effect size plot
        if 'effect_size_validation' in validation:
            self._plot_effect_sizes(validation['effect_size_validation'], output_folder)
        
        # Create biomechanical consistency plot
        if 'biomechanical_consistency' in validation:
            self._plot_biomechanical_consistency(validation['biomechanical_consistency'], output_folder)
    
    def _plot_objectives_summary(self, results: Dict[str, Any], validation: Dict[str, Any], output_folder: str) -> None:
        """Create enhanced summary plot of all objectives."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Extraction success
        objective_names = []
        success_status = []
        significance_status = []
        
        for obj_name, obj_results in results.items():
            objective_names.append(obj_name.replace('_', ' ').title())
            success_status.append('Success' if 'error' not in obj_results else 'Failed')
            
            # Check for significance
            has_significance = False
            if 'error' not in obj_results:
                if (obj_results.get('p_value', 1.0) < 0.05 or
                    any(pr.get('p_value', 1.0) < 0.05 for pr in obj_results.get('pairwise_results', {}).values()) or
                    any(tr.get('p_value', 1.0) < 0.05 for tr in obj_results.get('test_results', {}).values())):
                    has_significance = True
            
            significance_status.append('Significant' if has_significance else 'Not Significant')
        
        # Create stacked bar plot
        y_pos = np.arange(len(objective_names))
        
        success_counts = [1 if status == 'Success' else 0 for status in success_status]
        sig_counts = [1 if status == 'Significant' else 0 for status in significance_status]
        
        ax1.barh(y_pos, success_counts, color='lightblue', alpha=0.7, label='Extracted')
        ax1.barh(y_pos, sig_counts, color='darkblue', alpha=0.7, label='Significant')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(objective_names)
        ax1.set_xlabel('Status')
        ax1.set_title('MOO Objectives Extraction & Significance')
        ax1.legend()
        ax1.set_xlim(0, 1.2)
        
        # Plot 2: Validation metrics
        if 'overall_validity' in validation:
            validity_metrics = validation['overall_validity']
            
            metrics = ['Success Rate', 'Significance Rate', 'Overall Validity']
            values = [
                validity_metrics.get('success_rate', 0),
                validity_metrics.get('significance_rate', 0),
                {'high': 1.0, 'medium': 0.6, 'low': 0.3}.get(validity_metrics.get('overall_validity', 'low'), 0.3)
            ]
            
            bars = ax2.bar(metrics, values, color=['green', 'blue', 'purple'], alpha=0.7)
            ax2.set_ylabel('Score')
            ax2.set_title('Validation Summary')
            ax2.set_ylim(0, 1.0)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'objectives_summary.png'), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
    def _plot_validation_summary(self, validation: Dict[str, Any], output_folder: str) -> None:
        """Create validation summary plot."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Multiple comparisons correction
        if 'multiple_comparisons_correction' in validation:
            mc_data = validation['multiple_comparisons_correction']
            
            methods = ['Original', 'Bonferroni', 'FDR', 'Holm']
            significant_counts = [
                mc_data.get('original_significant', 0),
                mc_data.get('bonferroni_significant', 0),
                mc_data.get('fdr_significant', 0),
                mc_data.get('holm_significant', 0)
            ]
            
            ax1.bar(methods, significant_counts, color=['red', 'orange', 'green', 'blue'], alpha=0.7)
            ax1.set_ylabel('Significant Tests')
            ax1.set_title('Multiple Comparisons Correction Impact')
            ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Effect sizes
        if 'effect_size_validation' in validation:
            effect_data = validation['effect_size_validation']
            
            obj_names = []
            mean_effects = []
            
            for obj_name, effect_info in effect_data.items():
                obj_names.append(obj_name.replace('_', ' ').title())
                mean_effects.append(effect_info.get('mean_effect', 0))
            
            if obj_names and mean_effects:
                ax2.barh(obj_names, mean_effects, color='purple', alpha=0.7)
                ax2.set_xlabel('Mean Effect Size')
                ax2.set_title('Effect Size by Objective')
        
        # Plot 3: Statistical power
        if 'statistical_power' in validation:
            power_data = validation['statistical_power']
            
            power_levels = {'high': 0, 'medium': 0, 'low': 0}
            
            for obj_name, power_info in power_data.items():
                level = power_info.get('power_level', 'low')
                power_levels[level] += 1
            
            ax3.pie(power_levels.values(), labels=power_levels.keys(), autopct='%1.0f%%',
                   colors=['green', 'orange', 'red'])
            ax3.set_title('Statistical Power Distribution')
        
        # Plot 4: Biomechanical consistency
        if 'biomechanical_consistency' in validation:
            biomech_data = validation['biomechanical_consistency']
            
            consistency_scores = []
            check_names = []
            
            for check_name, check_results in biomech_data.items():
                if isinstance(check_results, dict):
                    check_names.append(check_name.replace('_', ' ').title())
                    
                    # Simple scoring based on whether check passed
                    score = 0
                    if (check_results.get('consistent', False) or 
                        check_results.get('shows_expected_advantage', False) or
                        check_results.get('matches_biomechanical_expectation', False) or
                        check_results.get('shows_expected_independence_preference', False)):
                        score = 1
                    
                    consistency_scores.append(score)
            
            if check_names and consistency_scores:
                colors = ['green' if score > 0 else 'red' for score in consistency_scores]
                ax4.barh(check_names, consistency_scores, color=colors, alpha=0.7)
                ax4.set_xlabel('Consistent (1) / Inconsistent (0)')
                ax4.set_title('Biomechanical Consistency')
                ax4.set_xlim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'validation_summary.png'), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
    def _plot_effect_sizes(self, effect_data: Dict[str, Any], output_folder: str) -> None:
        """Create effect size visualization."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        obj_names = []
        mean_effects = []
        interpretations = []
        
        for obj_name, effect_info in effect_data.items():
            obj_names.append(obj_name.replace('_', ' ').title())
            mean_effects.append(effect_info.get('mean_effect', 0))
            interpretations.append(effect_info.get('interpretation', 'unknown'))
        
        # Color code by interpretation
        color_map = {
            'negligible effect': 'lightgray',
            'small effect': 'lightblue',
            'medium effect': 'orange',
            'large effect': 'red',
            'unknown': 'gray'
        }
        
        colors = [color_map.get(interp, 'gray') for interp in interpretations]
        
        bars = ax.bar(obj_names, mean_effects, color=colors, alpha=0.7)
        ax.set_ylabel('Mean Effect Size')
        ax.set_title('Effect Sizes by Objective')
        ax.tick_params(axis='x', rotation=45)
        
        # Add horizontal lines for effect size thresholds
        ax.axhline(y=0.02, color='gray', linestyle='--', alpha=0.5, label='Small (0.02)')
        ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Medium (0.05)')
        ax.axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Large (0.10)')
        
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'effect_sizes.png'), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
    def _plot_biomechanical_consistency(self, biomech_data: Dict[str, Any], output_folder: str) -> None:
        """Create biomechanical consistency visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        check_names = []
        consistency_values = []
        
        for check_name, check_results in biomech_data.items():
            if isinstance(check_results, dict):
                check_names.append(check_name.replace('_', ' ').title())
                
                # Extract relevant consistency metric
                value = 0.5  # Default neutral
                
                if 'correlation_with_finger_strength' in check_results:
                    candidate = check_results.get('correlation_with_finger_strength', 0)
                    value = candidate if isinstance(candidate, (int, float)) else 0
                elif 'home_preference_coefficient' in check_results:
                    candidate = check_results.get('home_preference_coefficient', 0)
                    if isinstance(candidate, (int, float)):
                        value = candidate
                    elif isinstance(candidate, dict):
                        # Handle nested structure
                        nested_val = candidate.get('overall_home_preference', 0)
                        value = nested_val if isinstance(nested_val, (int, float)) else 0
                    else:
                        value = 0
                elif check_results.get('matches_biomechanical_expectation', False):
                    value = 1.0
                elif check_results.get('shows_expected_independence_preference', False):
                    value = 1.0
                
                consistency_values.append(value)
        
        if check_names and consistency_values:
            # Color code: positive = good, negative = bad, near zero = neutral
            colors = ['green' if v > 0.2 else 'red' if v < -0.2 else 'orange' for v in consistency_values]
            
            bars = ax.barh(check_names, consistency_values, color=colors, alpha=0.7)
            ax.set_xlabel('Consistency Score')
            ax.set_title('Biomechanical Consistency by Check')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, consistency_values):
                width = bar.get_width()
                ax.text(width + (0.02 if width >= 0 else -0.02), bar.get_y() + bar.get_height()/2.,
                       f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'biomechanical_consistency.png'), 
                   dpi=self.config.get('figure_dpi', 300), bbox_inches='tight')
        plt.close()
    
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
            
            # Save as JSON for programmatic access
            import json
            json_path = os.path.join(output_folder, 'key_preference_scores.json')
            with open(json_path, 'w') as f:
                json.dump(key_scores, f, indent=2)
            logger.info(f"Key preference scores saved to {json_path}")
            
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
                'Power': 'N/A',
                'Biomech_Consistent': 'N/A'
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
        print(f"Biomechanically consistent: {summary['biomechanical_consistency']}")
        print(f"FDR-corrected significant tests: {summary['multiple_comparisons_impact']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())