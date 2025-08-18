"""
Bigram Typing Preference Study bigram typing speed, frequency, and choice analysis script

Analyzes relationships between bigram typing times, frequencies, and choices.

Features:
- Per-participant normalization and robust statistics
- Configurable analysis parameters

Usage:
    python analyze_speed_frequencies_choices.py [--config config.yaml]
"""
import os
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt

from bigram_frequencies import bigrams, bigram_frequencies_array

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)  

# =============================================================================
# CONSOLIDATED STATISTICAL HELPERS
# =============================================================================

class StatisticalHelpers:
    """Consolidated statistical methods to eliminate redundancy and ensure consistency."""
    
    @staticmethod
    def calculate_correlation_with_ci(x: Union[np.ndarray, pd.Series], 
                                    y: Union[np.ndarray, pd.Series], 
                                    method: str = 'spearman') -> Dict[str, float]:
        """
        Calculate correlation with confidence intervals and effect sizes.
        
        Args:
            x, y: Data arrays/series
            method: 'spearman' or 'pearson'
            
        Returns:
            Dictionary with correlation, p_value, ci_lower, ci_upper, r2
        """
        # Convert to numpy arrays for consistency
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        # Remove NaN values
        mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]
        
        if len(x_clean) < 3:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'r2': np.nan,
                'n_observations': len(x_clean)
            }
        
        # Calculate correlation
        if method == 'spearman':
            correlation, p_value = stats.spearmanr(x_clean, y_clean)
        else:
            correlation, p_value = stats.pearsonr(x_clean, y_clean)
        
        # Calculate confidence interval using Fisher z-transformation
        n = len(x_clean)
        z = np.arctanh(correlation)
        se = 1 / np.sqrt(n - 3)
        z_ci = stats.norm.interval(0.95, loc=z, scale=se)
        ci_lower, ci_upper = np.tanh(z_ci[0]), np.tanh(z_ci[1])
        
        # Calculate R-squared
        r2 = correlation ** 2
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'r2': r2,
            'n_observations': n
        }
    
    @staticmethod
    def calculate_proportion_test(successes: int, trials: int, 
                                expected_p: float = 0.5) -> Dict[str, float]:
        """
        Calculate proportion test with effect sizes and confidence intervals.
        
        Args:
            successes: Number of successes
            trials: Total number of trials
            expected_p: Expected proportion under null hypothesis
            
        Returns:
            Dictionary with proportion, effect_size, p_value, ci_lower, ci_upper
        """
        if trials == 0:
            return {
                'proportion': np.nan,
                'effect_size': np.nan,
                'p_value': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_trials': 0,
                'significant': False
            }
        
        proportion = successes / trials
        
        # Effect size (deviation from expected)
        effect_size = abs(proportion - expected_p)
        
        # Binomial test
        binom_result = stats.binomtest(successes, trials, expected_p, alternative='two-sided')
        p_value = binom_result.pvalue
        
        # Wilson score confidence interval
        ci_lower, ci_upper = StatisticalHelpers._wilson_ci(successes, trials)
        
        return {
            'proportion': proportion,
            'effect_size': effect_size,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_trials': trials,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def _wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval."""
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
    
    @staticmethod
    def normalize_within_participant(data: pd.DataFrame,
                                   value_column: str,
                                   user_column: str = 'user_id',
                                   eps: float = 1e-10) -> pd.Series:
        """Normalize values within each participant using robust statistics."""
        return data.groupby(user_column, observed=True)[value_column].transform(
            lambda x: (x - x.median()) / (median_abs_deviation(x, nan_policy='omit') + eps)
        )

# =============================================================================
# SIMPLIFIED PLOTTING UTILITIES
# =============================================================================

class PlottingUtils:
    """Simplified plotting utilities with consistent styling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_style()
        
    def setup_style(self) -> None:
        """Configure global matplotlib style settings."""
        plt.style.use(self.config['visualization']['style'])
        plt.rcParams['figure.figsize'] = self.config['visualization']['figsize']
        plt.rcParams['figure.dpi'] = self.config['visualization']['dpi']
        
    def create_scatter_with_regression(self, x: np.ndarray, y: np.ndarray, 
                                     xlabel: str, ylabel: str, title: str,
                                     output_path: str) -> Dict[str, float]:
        """Create scatter plot with regression line and statistics."""
        fig, ax = plt.subplots(figsize=self.config['visualization']['figsize'])
        
        # Create scatter plot
        ax.scatter(x, y, alpha=0.6, color=self.config['visualization']['colors']['primary'])
        
        # Add regression line
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ax.plot(x, y_pred, '--', color='red', alpha=0.7)
        
        # Calculate statistics
        stats_result = StatisticalHelpers.calculate_correlation_with_ci(x, y)
        
        # Add statistics text
        ax.text(0.05, 0.95, 
                f"Correlation: {stats_result['correlation']:.3f}\n"
                f"p-value: {stats_result['p_value']:.3e}\n"
                f"R²: {stats_result['r2']:.3f}",
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        return stats_result

# =============================================================================
# MAIN ANALYSIS CLASS (SIMPLIFIED)
# =============================================================================

class BigramAnalysis:
    """Main analysis class with simplified, focused methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = StatisticalHelpers()
        self.plotter = PlottingUtils(config)

    # =========================================================================
    # CORE ANALYSIS METHODS (SIMPLIFIED)
    # =========================================================================

    def analyze_typing_times_slider_values(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """Analyze typing times in relation to slider values."""
        results = {}
        
        # Apply time limit and calculate differences
        data = self._apply_time_limit(data)
        data['time_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['time_diff_norm'] = self.stats.normalize_within_participant(data, 'time_diff')
        
        # Generate key plots
        self._plot_time_difference_analysis(data, output_folder)
        
        # Calculate basic statistics
        raw_stats = self.stats.calculate_correlation_with_ci(
            data['sliderValue'], data['time_diff']
        )
        norm_stats = self.stats.calculate_correlation_with_ci(
            data['sliderValue'], data['time_diff_norm']
        )
        
        results = {
            'raw_correlation': raw_stats,
            'normalized_correlation': norm_stats,
            'n_observations': len(data)
        }
        
        return results

    def analyze_frequency_typing_relationship(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """Analyze relationship between bigram frequency and typing times."""
        # Create frequency dictionary
        bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
        
        # Calculate aggregate statistics per bigram
        bigram_stats = []
        for bigram in set(data['chosen_bigram'].unique()) | set(data['unchosen_bigram'].unique()):
            # Combine chosen and unchosen times
            times = pd.concat([
                data[data['chosen_bigram'] == bigram]['chosen_bigram_time'],
                data[data['unchosen_bigram'] == bigram]['unchosen_bigram_time']
            ])
            
            if len(times) > 0 and bigram in bigram_freqs:
                bigram_stats.append({
                    'bigram': bigram,
                    'frequency': bigram_freqs[bigram],
                    'median_time': times.median(),
                    'n_samples': len(times)
                })
        
        df_stats = pd.DataFrame(bigram_stats)
        
        if len(df_stats) == 0:
            return {'error': 'No valid bigram statistics found'}
        
        # Calculate correlation between log frequency and median time
        freq_log = np.log10(df_stats['frequency'])
        correlation_stats = self.stats.calculate_correlation_with_ci(freq_log, df_stats['median_time'])
        
        # Create plot
        self.plotter.create_scatter_with_regression(
            df_stats['frequency'], df_stats['median_time'],
            'Bigram Frequency (log scale)', 'Median Typing Time (ms)',
            'Frequency vs Typing Time Relationship',
            os.path.join(output_folder, 'frequency_timing_relationship.png')
        )
        
        return {
            'correlation_stats': correlation_stats,
            'n_bigrams': len(df_stats),
            'total_samples': df_stats['n_samples'].sum()
        }

    def analyze_speed_choice_prediction(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """Analyze how well typing speed predicts bigram choices."""
        # Prepare data
        data = self._apply_time_limit(data)
        data['speed_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['speed_predicts_choice'] = data['speed_diff'] < 0
        
        # Overall prediction accuracy
        overall_accuracy = data['speed_predicts_choice'].mean()
        
        # Per-participant analysis
        participant_results = self._analyze_prediction_per_participant(data)
        
        # Generate simple accuracy plot
        self._plot_prediction_accuracy(participant_results, output_folder)
        
        return {
            'overall_accuracy': overall_accuracy,
            'n_participants': len(participant_results),
            'n_trials': len(data),
            'participant_results': participant_results
        }

    def analyze_bigram_pair_choices(self, data: pd.DataFrame, output_folder: str, 
                                  output_csv: str = 'bigram_pair_choices.csv') -> pd.DataFrame:
        """Analyze statistics for each unique bigram pair."""
        data = data.copy()
        data['abs_slider_value'] = data['sliderValue'].abs()
        
        # Get unique bigram pairs
        def get_sorted_pair(row):
            return tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']]))
        
        data['bigram_pair'] = data.apply(get_sorted_pair, axis=1)
        
        # Calculate statistics for each pair
        pair_stats = []
        for pair in data['bigram_pair'].unique():
            pair_data = data[data['bigram_pair'] == pair]
            
            for i, bigram in enumerate(pair):
                chosen_data = pair_data[pair_data['chosen_bigram'] == bigram]
                n_chosen = len(chosen_data)
                total_encounters = len(pair_data)
                
                if i == 0:  # Only process each pair once
                    other_bigram = pair[1]
                    other_chosen = len(pair_data[pair_data['chosen_bigram'] == other_bigram])
                    
                    pair_stats.append({
                        'bigram1': bigram,
                        'bigram1_chosen_count': n_chosen,
                        'bigram1_score': n_chosen / total_encounters,
                        'bigram2': other_bigram,
                        'bigram2_chosen_count': other_chosen,
                        'bigram2_score': other_chosen / total_encounters,
                        'total_encounters': total_encounters
                    })
        
        results_df = pd.DataFrame(pair_stats)
        
        # Save results
        os.makedirs(output_folder, exist_ok=True)
        results_df.to_csv(os.path.join(output_folder, output_csv), index=False)
        
        return results_df

    # =========================================================================
    # HELPER METHODS (SIMPLIFIED)
    # =========================================================================

    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate experiment data."""
        required_columns = [
            'user_id', 'chosen_bigram', 'unchosen_bigram',
            'chosen_bigram_time', 'unchosen_bigram_time', 'sliderValue'
        ]
        
        try:
            df = pd.read_csv(data_path)
            
            # Validate columns
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert numeric columns and handle missing values
            numeric_cols = ['chosen_bigram_time', 'unchosen_bigram_time', 'sliderValue']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing values
            n_before = len(df)
            df = df.dropna(subset=numeric_cols)
            n_dropped = n_before - len(df)
            
            if n_dropped > 0:
                logger.warning(f"Dropped {n_dropped} rows with missing values")
            
            # Validate ranges
            if not (-100 <= df['sliderValue'].max() <= 100 and -100 <= df['sliderValue'].min() <= 100):
                raise ValueError("sliderValue outside valid range [-100, 100]")
            
            if (df['chosen_bigram_time'] < 0).any() or (df['unchosen_bigram_time'] < 0).any():
                raise ValueError("Negative typing times found")
            
            logger.info(f"Loaded {len(df)} rows from {len(df['user_id'].unique())} participants")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
            raise

    def _apply_time_limit(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply maximum time limit to typing times."""
        max_time = self.config['analysis']['max_time_ms']
        data = data.copy()
        
        # Clip times and log changes
        chosen_clipped = len(data[data['chosen_bigram_time'] > max_time])
        unchosen_clipped = len(data[data['unchosen_bigram_time'] > max_time])
        
        data['chosen_bigram_time'] = data['chosen_bigram_time'].clip(upper=max_time)
        data['unchosen_bigram_time'] = data['unchosen_bigram_time'].clip(upper=max_time)
        
        if chosen_clipped + unchosen_clipped > 0:
            logger.info(f"Clipped {chosen_clipped + unchosen_clipped} typing times to {max_time}ms")
        
        return data

    def _analyze_prediction_per_participant(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze prediction accuracy per participant."""
        min_trials = self.config['analysis'].get('min_trials_per_participant', 5)
        results = []
        
        for user_id, user_data in data.groupby('user_id', observed=True):
            if len(user_data) >= min_trials:
                accuracy = user_data['speed_predicts_choice'].mean()
                ci_lower, ci_upper = self.stats._wilson_ci(
                    int(accuracy * len(user_data)), len(user_data)
                )
                
                results.append({
                    'user_id': user_id,
                    'n_trials': len(user_data),
                    'accuracy': accuracy,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
        
        return results

    # =========================================================================
    # SIMPLIFIED PLOTTING METHODS
    # =========================================================================

    def _plot_time_difference_analysis(self, data: pd.DataFrame, output_folder: str) -> None:
        """Create essential time difference plots."""
        # Raw time difference vs slider value
        self.plotter.create_scatter_with_regression(
            data['sliderValue'], data['time_diff'],
            'Slider Value', 'Time Difference (ms)', 
            'Raw Time Difference vs Slider Value',
            os.path.join(output_folder, 'raw_time_diff_vs_slider.png')
        )
        
        # Normalized time difference vs slider value
        self.plotter.create_scatter_with_regression(
            data['sliderValue'], data['time_diff_norm'],
            'Slider Value', 'Normalized Time Difference',
            'Normalized Time Difference vs Slider Value',
            os.path.join(output_folder, 'normalized_time_diff_vs_slider.png')
        )

    def _plot_prediction_accuracy(self, participant_results: List[Dict], output_folder: str) -> None:
        """Plot distribution of per-participant prediction accuracies."""
        if not participant_results:
            return
            
        accuracies = [p['accuracy'] for p in participant_results]
        
        fig, ax = plt.subplots(figsize=self.config['visualization']['figsize'])
        
        ax.hist(accuracies, bins=20, alpha=0.7, color=self.config['visualization']['colors']['primary'])
        ax.axvline(np.mean(accuracies), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(accuracies):.3f}')
        ax.axvline(0.5, color='gray', linestyle=':', label='Chance (0.5)')
        
        ax.set_xlabel('Prediction Accuracy')
        ax.set_ylabel('Number of Participants')
        ax.set_title('Distribution of Per-Participant Prediction Accuracies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'participant_accuracy_distribution.png'),
                   dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()

# =============================================================================
# CONFIGURATION AND MAIN FUNCTION
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Add default values if not specified
        defaults = {
            'visualization': {
                'style': 'default',
                'figsize': (10, 6),
                'dpi': 300,
                'colors': {
                    'primary': '#1f77b4',
                    'secondary': '#ff7f0e'
                }
            }
        }
        
        # Update config with defaults for missing values
        for section, values in defaults.items():
            if section not in config:
                config[section] = values
            elif isinstance(values, dict):
                for key, value in values.items():
                    if key not in config[section]:
                        config[section][key] = value
        
        return config
            
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        raise

def main():
    """Main analysis function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze bigram typing data')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        input_folder = config['data']['input_dir']
        output_folder = os.path.join(input_folder, 'output')

        # Create output directories
        subdirs = [
            'typing_time_vs_preference',
            'typing_time_vs_frequency', 
            'preference_prediction'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(output_folder, subdir), exist_ok=True)

        # Initialize analyzer
        analyzer = BigramAnalysis(config)
        
        # Load data
        data_path = os.path.join(input_folder, config['data']['filtered_data_file'])
        data = analyzer.load_and_validate_data(data_path)
        logger.info(f"Processing {len(data)} trials from {data['user_id'].nunique()} participants...")

        # Run core analyses
        logger.info("Analyzing typing times vs. preference...")
        choice_results = analyzer.analyze_typing_times_slider_values(
            data, os.path.join(output_folder, 'typing_time_vs_preference')
        )
        
        logger.info("Analyzing typing times vs. frequency...")
        freq_results = analyzer.analyze_frequency_typing_relationship(
            data, os.path.join(output_folder, 'typing_time_vs_frequency')
        )
        
        logger.info("Analyzing preference prediction...")
        prediction_results = analyzer.analyze_speed_choice_prediction(
            data, os.path.join(output_folder, 'preference_prediction')
        )
        logger.info(f"    Overall accuracy: {prediction_results['overall_accuracy']:.1%}")

        logger.info("Analyzing bigram pair choices...")
        pair_stats_df = analyzer.analyze_bigram_pair_choices(data, output_folder)

        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()