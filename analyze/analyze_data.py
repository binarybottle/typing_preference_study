"""
Bigram Typing Analysis Script

Analyzes relationships between bigram typing times, frequencies and user choices,
combining both legacy and enhanced analysis approaches.

Features:
- Per-participant normalization and robust statistics
- Recreation of original plots with enhanced statistical methods
- Comprehensive logging
- Configurable analysis parameters

Usage:
    python analyze_data.py [--config config.yaml]
"""

import os
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import median_abs_deviation
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

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
               
class PlottingUtils:
    """
    Utility class for creating consistent visualizations across analyses.
    
    Handles:
    - Common plot styling and configuration
    - Error bar calculations
    - Legend and annotation placement
    - Color scheme management
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plotting utilities with configuration.
        
        Args:
            config: Configuration dictionary containing visualization settings
        """
        self.config = config
        self.setup_style()
        
    def setup_style(self) -> None:
        """Configure global matplotlib style settings."""
        plt.style.use(self.config['visualization']['style'])
        plt.rcParams['figure.figsize'] = self.config['visualization']['figsize']
        plt.rcParams['figure.dpi'] = self.config['visualization']['dpi']
        
    def create_figure(
        self,
        plot_type: str
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a new figure with appropriate settings.
        
        Args:
            plot_type: Type of plot to create (determines settings)
            
        Returns:
            Tuple of (figure, axes)
        """
        plot_config = self.config['visualization']['plots'].get(plot_type, {})
        figsize = plot_config.get('figsize', self.config['visualization']['figsize'])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Only add grid for certain plot types
        #if plot_type != 'time_diff':
        #    ax.grid(True, alpha=plot_config.get('grid_alpha', 0.3))
        
        return fig, ax
        
    def add_error_bars(
        self,
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
        plot_type: str
    ) -> None:
        """
        Add error bars to an existing plot.
        
        Args:
            ax: Axes to add error bars to
            x: X-coordinates
            y: Y-coordinates
            yerr: Error bar values
            plot_type: Type of plot (determines styling)
        """
        plot_config = self.config['visualization']['plots'].get(plot_type, {})
        
        ax.errorbar(
            x, y, yerr=yerr,
            capsize=plot_config.get('error_capsize', 5),
            color=self.config['visualization']['colors']['error_bars'],
            alpha=plot_config.get('error_alpha', 0.2),
            fmt='none'
        )
        
    def add_regression_line(
        self,
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        plot_type: str
    ) -> Dict[str, float]:
        """
        Add regression line to scatter plot.
        
        Args:
            ax: Axes to add line to
            x: X values
            y: Y values
            plot_type: Type of plot
            
        Returns:
            Dictionary containing regression statistics
        """
        plot_config = self.config['visualization']['plots'].get(plot_type, {})
        
        # Calculate regression
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))
        
        # Plot regression line
        ax.plot(
            x, y_pred,
            '--',
            color=self.config['visualization']['colors']['regression'],
            alpha=plot_config.get('regression_alpha', 0.7),
            label=f'R² = {r2:.3f}'
        )
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r2': r2
        }
        
    def add_correlation_info(
        self,
        ax: plt.Axes,
        correlation: float,
        p_value: float,
        plot_type: str
    ) -> None:
        """
        Add correlation information to plot.
        
        Args:
            ax: Axes to add information to
            correlation: Correlation coefficient
            p_value: P-value
            plot_type: Type of plot
        """
        plot_config = self.config['visualization']['plots'].get(plot_type, {})
        
        # Add text box with correlation info
        ax.text(
            0.05, 0.95,
            f'Correlation: {correlation:.3f}\np-value: {p_value:.3e}',
            transform=ax.transAxes,
            bbox=dict(
                facecolor='white',
                alpha=plot_config.get('confidence_alpha', 0.8)
            )
        )

class RobustStatistics:
    """
    Core statistical methods implementing robust statistical measures and analyses.
    
    This class provides methods for:
    - Within-participant data normalization
    - Confidence interval computation using bootstrap
    - Outlier-resistant statistical calculations
    """
    
    @staticmethod
    def normalize_within_participant(
        data: pd.DataFrame,
        value_column: str,
        user_column: str = 'user_id',
        eps: float = 1e-10
    ) -> pd.Series:
        """
        Normalize values within each participant using robust statistics.
        
        Uses median and median absolute deviation (MAD) for robustness against outliers:
        normalized_value = (x - median(x)) / (MAD(x) + eps)
        
        Args:
            data: DataFrame containing the data
            value_column: Name of column to normalize
            user_column: Name of column containing participant IDs
            eps: Small value to prevent division by zero
            
        Returns:
            Series containing normalized values
            
        Notes:
            - MAD is scaled to be consistent with standard deviation for normal distribution
            - Small epsilon value prevents division by zero for constant sequences
        """
        return data.groupby(user_column, observed=True)[value_column].transform(
            lambda x: (x - x.median()) / (median_abs_deviation(x, nan_policy='omit') + eps)
        )
    
    @staticmethod
    def compute_confidence_intervals(
        values: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Compute confidence intervals using bootstrap resampling.
        
        Args:
            values: Array of values to analyze
            confidence: Confidence level (0 to 1)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound)
            
        Notes:
            - Uses non-parametric bootstrap to avoid distribution assumptions
            - Returns nan values if input array has less than 2 values
            - Handles extreme confidence levels by clamping to valid percentiles
        """
        if len(values) < 2:
            return np.nan, np.nan
        
        try:
            # Generate bootstrap samples
            bootstrap_samples = np.random.choice(
                values, 
                size=(n_bootstrap, len(values)), 
                replace=True
            )
            
            # Calculate median for each bootstrap sample
            bootstrap_medians = np.median(bootstrap_samples, axis=1)
            
            # Calculate percentile-based confidence intervals
            lower_percentile = ((1 - confidence) / 2) * 100
            upper_percentile = (1 - (1 - confidence) / 2) * 100
            
            ci_lower = np.percentile(bootstrap_medians, lower_percentile)
            ci_upper = np.percentile(bootstrap_medians, upper_percentile)
            
            return ci_lower, ci_upper
            
        except Exception as e:
            logger.warning(f"Error computing confidence intervals: {str(e)}")
            return np.nan, np.nan
        
class BigramAnalysis:
    """Main analysis class combining legacy and enhanced approaches."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize analysis with configuration.
        
        Args:
            config: Configuration dictionary containing analysis parameters
        """
        self.config = config
        self.stats = RobustStatistics()
        self.plotter = PlottingUtils(config)

    # Analysis functions

    def analyze_typing_times_slider_values(
        self,
        data: pd.DataFrame,
        output_folder: str,
    ) -> Dict[str, Any]:
        """
        Analyze typing times in relation to slider values.
        Enhanced version with robust statistics.
        
        Args:
            data: DataFrame containing trial data
            output_folder: Directory to save outputs
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Create copy and apply time limit
        data = data.copy()
        data = self._apply_time_limit(data)
        
        # Calculate speed differences and normalize
        data['time_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['time_diff_norm'] = self.stats.normalize_within_participant(data, 'time_diff')
        
        # Generate plots using plotting utilities
        try:
            # Typing time distributions
            self._plot_typing_times(data, output_folder)
            
            # Relationship between chosen and unchosen times
            self._plot_chosen_vs_unchosen_scatter(data, output_folder)
            
            # Time differences vs slider values
            self._plot_time_diff_slider(data, output_folder)
            
            # Histograms of time differences
            self._plot_time_diff_histograms(data, output_folder)
            
            # Overlaid time histograms
            for prefix in ['raw_', 'normalized_']:
                self._plot_overlaid_time_histograms(data, output_folder, prefix)
                
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            raise
            
        return results

    def analyze_frequency_typing_relationship(
        self,
        data: pd.DataFrame,
        output_folder: str
    ) -> Dict[str, Any]:
        """
        Analyze relationship between bigram frequency and typing times.
        
        Args:
            data: DataFrame containing trial data
            output_folder: Directory to save outputs
            
        Returns:
            Dictionary containing analysis results
        """
        # Create frequency dictionary
        bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
        
        # Calculate aggregate statistics
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
                    'min_time': times.min(),
                    'std_time': times.std(),
                    'n_samples': len(times)
                })
        
        df_stats = pd.DataFrame(bigram_stats)
        
        try:
            # Generate plots
            self._plot_distribution(df_stats, output_folder)
            self._plot_median_times(df_stats, output_folder)
            self._plot_min_times(df_stats, output_folder)
            
            # Calculate and save statistics
            n_groups = self.config['analysis'].get('n_quantiles', 5)
            df_stats['freq_group'] = pd.qcut(df_stats['frequency'], n_groups, labels=False)
            group_stats = self._calculate_group_stats(df_stats, n_groups)
            
            # Generate reports
            self._save_frequency_group_timing_analysis(
                df_stats,
                *self._calculate_frequency_timing_statistics(df_stats),
                group_stats,
                output_folder
            )
            
            return {
                'n_bigrams': len(df_stats),
                'total_instances': df_stats['n_samples'].sum(),
                'group_stats': group_stats.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error in frequency analysis: {str(e)}")
            raise

    def analyze_speed_choice_prediction(
        self,
        data: pd.DataFrame,
        output_folder: str
    ) -> Dict[str, Any]:
        """
        Analyze how well typing speed predicts bigram choices.
        
        Args:
            data: DataFrame containing trial data
            output_folder: Directory to save outputs
            
        Returns:
            Dictionary containing prediction results
        """
        # Prepare data
        data = data.copy()
        data = self._apply_time_limit(data)
        
        # Calculate predictors
        data['speed_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['speed_predicts_choice'] = data['speed_diff'] < 0
        data['confidence'] = data['sliderValue'].abs()
        data['speed_diff_norm'] = self.stats.normalize_within_participant(data, 'speed_diff')
        
        # Add frequency differences if available
        try:
            self._add_frequency_differences(data)
        except Exception as e:
            logger.warning(f"Could not add frequency differences: {str(e)}")
        
        # Calculate magnitude quintiles
        data['speed_diff_mag'] = data['speed_diff'].abs()
        data['magnitude_quintile'] = pd.qcut(
            data['speed_diff_mag'],
            self.config['analysis'].get('n_quantiles', 5),
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            duplicates='drop'
        )
        
        # Analyze per participant
        participant_results = self._analyze_per_participant(
            data, 
            self.config['analysis'].get('min_trials_per_participant', 5)
        )
        
        try:
            # Generate plots
            self._plot_accuracy_by_magnitude(data, output_folder)
            self._plot_accuracy_by_confidence(data, output_folder)
            self._plot_user_accuracy_distribution(participant_results, output_folder)
            
            # Add new analyses
            below_chance_results = self._analyze_bigram_choices(data)
            pattern_results = self._analyze_population_patterns(data, below_chance_results)
            
            # Generate reports
            self._generate_prediction_report(
                {
                    'participant_results': participant_results,
                    'overall_accuracy': data['speed_predicts_choice'].mean(),
                    'n_participants': len(participant_results),
                    'n_trials': len(data)
                },
                output_folder
            )
            
            # Generate detailed below-chance analysis report
            if below_chance_results:
                self._generate_below_chance_report(
                    below_chance_results,
                    pattern_results,
                    output_folder
                )
            
            return {
                'overall_accuracy': data['speed_predicts_choice'].mean(),
                'n_participants': len(participant_results),
                'n_trials': len(data),
                'participant_results': participant_results,
                'below_chance_analysis': below_chance_results,
                'pattern_analysis': pattern_results
            }      
        except Exception as e:
            logger.error(f"Error in prediction analysis: {str(e)}")
            raise

    def analyze_variance_and_prediction(
        self,
        data: pd.DataFrame,
        output_folder: str
    ) -> Dict[str, float]:
        """
        Analyze variance explained and predictive power of speed and frequency.
        """
        # [Previous preprocessing code stays the same] 
        data = data.copy()
        data = self._apply_time_limit(data)
        
        results = {}
        
        # [Previous variance calculations stay the same]
        data['speed_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['speed_diff_mag'] = data['speed_diff'].abs()
        speed_confidence_corr = stats.spearmanr(data['speed_diff_mag'], data['abs_sliderValue'])[0]
        results['speed_variance'] = speed_confidence_corr ** 2 * 100
        
        # Add frequency differences and calculate frequency variance
        self._add_frequency_differences(data)
        data = data.dropna(subset=['freq_diff'])
        freq_confidence_corr = stats.spearmanr(data['freq_diff'].abs(), data['abs_sliderValue'])[0]
        results['frequency_variance'] = freq_confidence_corr ** 2 * 100
        
        partial_corr = self._calculate_partial_correlation(
            data['freq_diff'].abs(),
            data['abs_sliderValue'],
            data['speed_diff_mag']
        )
        results['frequency_partial_variance'] = partial_corr ** 2 * 100
        
        # Calculate basic prediction rates
        data['speed_predicts_choice'] = data['speed_diff'] < 0
        results['speed_predictive'] = np.mean(data['speed_predicts_choice']) * 100
        
        data['freq_predicts_choice'] = data['freq_diff'] > 0
        results['frequency_predictive'] = np.mean(data['freq_predicts_choice']) * 100
        
        # Add logistic regression analysis
        from statsmodels.discrete.discrete_model import Logit
        import statsmodels.api as sm
        
        # Standardize predictors
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        speed_diff_std = scaler.fit_transform(data['speed_diff'].values.reshape(-1, 1))
        freq_diff_std = scaler.fit_transform(data['freq_diff'].values.reshape(-1, 1))
        
        # Prepare target
        y = (data['sliderValue'] > 0).astype(int)
        
        # Run logistic regressions
        null_model = Logit(y, np.ones_like(y)).fit(disp=0)
        
        X_speed = sm.add_constant(speed_diff_std)
        speed_model = Logit(y, X_speed).fit(disp=0)
        results['speed_logistic_r2'] = (1 - (speed_model.llf / null_model.llf)) * 100
        results['speed_odds_ratio'] = np.exp(speed_model.params.iloc[1])
        results['speed_p_value'] = speed_model.pvalues.iloc[1]
        
        X_freq = sm.add_constant(freq_diff_std)
        freq_model = Logit(y, X_freq).fit(disp=0)
        results['freq_logistic_r2'] = (1 - (freq_model.llf / null_model.llf)) * 100
        results['freq_odds_ratio'] = np.exp(freq_model.params.iloc[1])
        results['freq_p_value'] = freq_model.pvalues.iloc[1]
        
        X_both = sm.add_constant(np.column_stack([speed_diff_std, freq_diff_std]))
        full_model = Logit(y, X_both).fit(disp=0)
        results['total_logistic_r2'] = (1 - (full_model.llf / null_model.llf)) * 100
        
        # Generate report
        self._generate_variance_prediction_report(results, output_folder)
        
        return results

    def analyze_bigram_pair_choices(self, data: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
            """
            Analyze statistics for each unique bigram pair in the dataset and optionally save to CSV.
            
            Args:
                data: DataFrame containing columns:
                    - chosen_bigram
                    - unchosen_bigram
                    - sliderValue
                output_path: Optional path to save CSV file. If None, no file is saved.
                    
            Returns:
                DataFrame with columns for each bigram pair's statistics
            """
            # Create copy of data and calculate absolute slider values
            data = data.copy()
            data['abs_slider_value'] = data['sliderValue'].abs()
            
            # Get all unique bigram pairs (order doesn't matter)
            def get_sorted_pair(row):
                return tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']]))
            
            data['bigram_pair'] = data.apply(get_sorted_pair, axis=1)
            unique_pairs = sorted(data['bigram_pair'].unique())
            
            pair_stats = []
            
            for pair in unique_pairs:
                # Get all trials involving this bigram pair
                pair_data = data[data['bigram_pair'] == pair]
                
                # For each bigram in the pair
                for i, bigram in enumerate(pair):
                    # Calculate how often this bigram was chosen
                    chosen_data = pair_data[pair_data['chosen_bigram'] == bigram]
                    n_chosen = len(chosen_data)
                    
                    # Calculate median and MAD of absolute slider values when chosen
                    if n_chosen > 0:
                        median_value = chosen_data['abs_slider_value'].median()
                        mad_value = median_abs_deviation(chosen_data['abs_slider_value'].values, nan_policy='omit')
                    else:
                        median_value = np.nan
                        mad_value = np.nan
                    
                    # Calculate score (could be implementation specific)
                    score = n_chosen / len(pair_data) if len(pair_data) > 0 else np.nan
                    
                    other_bigram = pair[1 - i]  # Get the other bigram in the pair
                    
                    # Only add the pair once (when processing first bigram)
                    if i == 0:
                        # Get stats for second bigram
                        other_chosen_data = pair_data[pair_data['chosen_bigram'] == other_bigram]
                        other_n_chosen = len(other_chosen_data)
                        
                        if other_n_chosen > 0:
                            other_median_value = other_chosen_data['abs_slider_value'].median()
                            other_mad_value = median_abs_deviation(other_chosen_data['abs_slider_value'].values, nan_policy='omit')
                        else:
                            other_median_value = np.nan
                            other_mad_value = np.nan
                            
                        other_score = other_n_chosen / len(pair_data) if len(pair_data) > 0 else np.nan
                        
                        pair_stats.append({
                            'bigram1': bigram,
                            'bigram1_score': score,
                            'N_chose_bigram1': n_chosen,
                            'median_value_of_bigram1': median_value,
                            'MAD_value_of_bigram1': mad_value,
                            'bigram2': other_bigram,
                            'bigram2_score': other_score,
                            'N_chose_bigram2': other_n_chosen,
                            'median_value_of_bigram2': other_median_value,
                            'MAD_value_of_bigram2': other_mad_value
                        })
            
            # Create DataFrame from collected statistics
            results_df = pd.DataFrame(pair_stats)
            
            # Save to CSV if output path is provided
            if output_path is not None:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save to CSV
                    results_df.to_csv(output_path, index=False)
                    logger.info(f"Saved bigram pair analysis to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving CSV file: {str(e)}")
                    raise
            
            return results_df
                
    def _analyze_bigram_choices(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze bigram choice patterns for below-chance participants.
        
        Args:
            data: DataFrame with columns:
                - user_id
                - chosen_bigram
                - unchosen_bigram
                - chosen_bigram_time
                - unchosen_bigram_time
                - sliderValue
                - speed_predicts_choice (bool)
        """
        # First identify below-chance participants
        results = {}
        
        # Group data by user_id and calculate prediction accuracy
        user_accuracies = data.groupby('user_id')['speed_predicts_choice'].mean()
        below_chance = user_accuracies[user_accuracies < 0.4].index
        
        # For each below-chance participant
        for user_id in below_chance:
            user_data = data[data['user_id'] == user_id].copy()
            user_data.loc[:, 'bigram_pair'] = user_data.apply(
                lambda row: tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']])), 
                axis=1
)
            
            # Get number of presentations per pair
            pair_counts = user_data['bigram_pair'].value_counts()
            repeat_pairs = pair_counts[pair_counts > 1].index
            
            # For each repeated pair, analyze consistency
            pair_results = {}
            for pair in repeat_pairs:
                pair_data = user_data[user_data['bigram_pair'] == pair]
                
                # Calculate choice consistency
                choices = pair_data.apply(
                    lambda row: row['chosen_bigram'] == pair[0],
                    axis=1
                )
                consistency = max(choices.mean(), 1 - choices.mean())
                
                # Get median population times for this pair
                pop_times = {
                    bigram: data[
                        (data['chosen_bigram'] == bigram) | 
                        (data['unchosen_bigram'] == bigram)
                    ].apply(
                        lambda row: row['chosen_bigram_time'] if row['chosen_bigram'] == bigram 
                        else row['unchosen_bigram_time'], 
                        axis=1
                    ).median()
                    for bigram in pair
                }
                
                # Get frequencies
                bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
                freqs = {bigram: bigram_freqs.get(bigram, 0) for bigram in pair}
                
                pair_results[pair] = {
                    'n_presentations': len(pair_data),
                    'consistency': consistency,
                    'chosen_bigram': pair[0] if choices.mean() > 0.5 else pair[1],
                    'their_median_time': pair_data[pair_data['chosen_bigram'] == pair[0]]['chosen_bigram_time'].median() if choices.mean() > 0.5 else pair_data[pair_data['chosen_bigram'] == pair[1]]['chosen_bigram_time'].median(),
                    'pop_median_times': pop_times,
                    'frequencies': freqs
                }
                
            results[user_id] = {
                'accuracy': user_accuracies[user_id],
                'n_trials': len(user_data),
                'n_repeat_pairs': len(repeat_pairs),
                'pair_results': pair_results
            }
        
        return results

    def _analyze_population_patterns(self, data: pd.DataFrame, below_chance_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare choices of below-chance participants to population patterns.
        
        Args:
            data: Full dataset DataFrame
            below_chance_results: Output from _analyze_bigram_choices()
        """
        patterns = {}
        
        for user_id, user_results in below_chance_results.items():
            # Filter for highly consistent choices (>80% same choice)
            consistent_pairs = {
                pair: results for pair, results in user_results['pair_results'].items()
                if results['consistency'] > 0.8
            }
            
            if consistent_pairs:
                patterns[user_id] = {
                    'n_consistent_pairs': len(consistent_pairs),
                    'slower_than_pop': [],   # Cases where they consistently choose the slower bigram
                    'against_frequency': [],  # Cases where they consistently choose the less frequent bigram
                    'hand_patterns': [],     # Common hand/finger patterns in choices
                }
                
                for pair, results in consistent_pairs.items():
                    chosen = results['chosen_bigram']
                    unchosen = pair[1] if chosen == pair[0] else pair[0]
                    
                    # Compare to population times
                    if results['their_median_time'] > results['pop_median_times'][chosen]:
                        patterns[user_id]['slower_than_pop'].append(
                            (pair, results['their_median_time'] / results['pop_median_times'][chosen])
                        )
                    
                    # Compare frequencies
                    if results['frequencies'][chosen] < results['frequencies'][unchosen]:
                        patterns[user_id]['against_frequency'].append(
                            (pair, results['frequencies'][unchosen] / results['frequencies'][chosen])
                        )
                    
                    # Could add hand/finger pattern analysis here
        
        return patterns
            
    # Data Handling Methods

    def load_and_validate_data(
        self,
        data_path: str
    ) -> pd.DataFrame:
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
                
            # Basic data validation
            if df['user_id'].isna().any():
                raise ValueError("Found missing user IDs")
                
            # Convert numeric columns
            numeric_cols = ['chosen_bigram_time', 'unchosen_bigram_time', 'sliderValue']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                n_dropped = df[col].isna().sum()
                if n_dropped > 0:
                    logger.warning(f"Dropped {n_dropped} non-numeric values from {col}")
            
            # Remove rows with NaN values
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

    def _add_frequency_differences(self, data: pd.DataFrame) -> None:
        """Add frequency differences to the dataset."""
        bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
        data['freq_diff'] = data.apply(
            lambda row: (
                np.log10(bigram_freqs.get(row['chosen_bigram'], np.nan)) - 
                np.log10(bigram_freqs.get(row['unchosen_bigram'], np.nan))
            ),
            axis=1
        )

    def _analyze_per_participant(
        self,
        data: pd.DataFrame,
        min_trials: int
    ) -> List[Dict[str, Any]]:
        """Analyze prediction accuracy per participant."""
        results = []
        for user_id, user_data in data.groupby('user_id', observed=True):
            if len(user_data) >= min_trials:
                speed_accuracy = user_data['speed_predicts_choice'].mean()
                ci_lower, ci_upper = self.stats.compute_confidence_intervals(
                    user_data['speed_predicts_choice'].values
                )
                
                mag_accuracies = user_data.groupby('magnitude_quintile', observed=True)[
                    'speed_predicts_choice'
                ].mean()
                
                results.append({
                    'user_id': user_id,
                    'n_trials': len(user_data),
                    'speed_accuracy': speed_accuracy,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'magnitude_accuracies': mag_accuracies.to_dict()
                })
        return results

    def _apply_time_limit(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply maximum time limit to typing times."""
        max_time = self.config['analysis']['max_time_ms']
        data = data.copy()
        
        # Get counts before clipping
        chosen_clipped = len(data[data['chosen_bigram_time'] > max_time])
        unchosen_clipped = len(data[data['unchosen_bigram_time'] > max_time])
        
        # Clip chosen and unchosen times
        data['chosen_bigram_time'] = data['chosen_bigram_time'].clip(upper=max_time)
        data['unchosen_bigram_time'] = data['unchosen_bigram_time'].clip(upper=max_time)
        
        total_clipped = chosen_clipped + unchosen_clipped
        if total_clipped > 0:
            logger.info(f"Clipped {total_clipped} typing times ({chosen_clipped} chosen, "
                    f"{unchosen_clipped} unchosen) to maximum of {max_time}ms")
        
        return data

    # Statistical Calculation Methods

    def _calculate_frequency_timing_statistics(
        self,
        df_stats: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        Calculate correlation and regression statistics for frequency-timing relationship.
        
        Args:
            df_stats: DataFrame containing per-bigram statistics
            
        Returns:
            Tuple of (correlation, p_value, r_squared)
        """
        freq_log = np.log10(df_stats['frequency'])
        correlation, p_value = stats.spearmanr(freq_log, df_stats['median_time'])
        slope, intercept = np.polyfit(freq_log, df_stats['median_time'], 1)
        pred = slope * freq_log + intercept
        r2 = 1 - (np.sum((df_stats['median_time'] - pred) ** 2) / 
                np.sum((df_stats['median_time'] - df_stats['median_time'].mean()) ** 2))
        
        return correlation, p_value, r2

    def _calculate_group_stats(
        self,
        df_stats: pd.DataFrame,
        n_groups: int
    ) -> pd.DataFrame:
        """
        Calculate statistics for frequency groups.
        
        Args:
            df_stats: DataFrame containing per-bigram statistics
            n_groups: Number of groups to create
            
        Returns:
            DataFrame containing group statistics
        """
        group_stats = []
        
        for group in range(n_groups):
            group_data = df_stats[df_stats['freq_group'] == group]
            if len(group_data) > 0:
                group_stats.append({
                    'group': f'Group {group + 1}',
                    'freq_range': (
                        float(group_data['frequency'].min()),
                        float(group_data['frequency'].max())
                    ),
                    'median_time': float(group_data['median_time'].mean()),
                    'mean_time': float(group_data['median_time'].mean()),
                    'timing_std': float(group_data['std_time'].mean()),
                    'n_bigrams': len(group_data),
                    'total_instances': int(group_data['n_samples'].sum())
                })
        
        return pd.DataFrame(group_stats)

    def _calculate_correlation_stats(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate correlation statistics between two arrays.
        
        Args:
            x: First array of values
            y: Second array of values
            
        Returns:
            Dictionary containing correlation statistics
        """
        correlation, p_value = stats.spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'r2': r2,
            'slope': slope,
            'intercept': intercept
        }

    def _calculate_user_correlations(
        self,
        data: pd.DataFrame,
        min_trials: int
    ) -> Dict[str, List[float]]:
        """
        Calculate per-user correlations.
        
        Args:
            data: DataFrame containing user data
            min_trials: Minimum number of trials required per user
            
        Returns:
            Dictionary containing lists of user correlations
        """
        user_correlations = {'raw': [], 'normalized': []}
        
        for user_id, user_data in data.groupby('user_id', observed=True):
            if len(user_data) >= min_trials:
                # Calculate raw correlation
                raw_corr = stats.spearmanr(
                    user_data['freq_diff'],
                    user_data['time_diff']
                )[0]
                user_correlations['raw'].append(raw_corr)
                
                # Calculate normalized correlation
                norm_corr = stats.spearmanr(
                    user_data['freq_diff'],
                    user_data['time_diff_norm']
                )[0]
                user_correlations['normalized'].append(norm_corr)
        
        return user_correlations

    def _calculate_correlation_stats(
        self,
        x: pd.Series,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate correlation statistics between two series.
        
        Args:
            x: First series of values
            y: Second series of values
            
        Returns:
            Dictionary containing correlation statistics
        """
        correlation, p_value = stats.spearmanr(x, y)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        r2 = 1 - (np.sum((y - y_pred) ** 2) / 
                np.sum((y - y.mean()) ** 2))
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'r2': r2,
            'slope': slope,
            'intercept': intercept
        }

    def _calculate_partial_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        covariate: np.ndarray
    ) -> float:
        """
        Calculate partial correlation controlling for covariate.
        
        Args:
            x: First variable
            y: Second variable
            covariate: Variable to control for
            
        Returns:
            Partial correlation coefficient
        """
        # Calculate regressions
        reg_x = stats.linregress(covariate, x)
        reg_y = stats.linregress(covariate, y)
        
        # Calculate residuals manually
        residuals_x = x - (reg_x.slope * covariate + reg_x.intercept)
        residuals_y = y - (reg_y.slope * covariate + reg_y.intercept)
        
        # Calculate correlation of residuals
        return stats.pearsonr(residuals_x, residuals_y)[0]

    def _calculate_predictive_power(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> float:
        """
        Calculate predictive power using cross-validated logistic regression.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_folds: Number of cross-validation folds
            
        Returns:
            Average ROC-AUC score as percentage
        """
        model = LogisticRegression(random_state=42)
        scores = cross_val_score(model, X, y, cv=n_folds, scoring='roc_auc')
        return np.mean(scores) * 100

    def _calculate_logistic_variance_explained(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate variance explained in binary choices using logistic regression.
        Returns McFadden's pseudo-R² for each predictor.
        """
        from sklearn.preprocessing import StandardScaler
        from statsmodels.discrete.discrete_model import Logit
        import statsmodels.api as sm

        # Prepare target: whether they chose the right bigram
        y = (data['sliderValue'] > 0).astype(int)
        
        # Standardize predictors for comparable coefficients
        scaler = StandardScaler()
        speed_diff_std = scaler.fit_transform(data['speed_diff'].values.reshape(-1, 1))
        freq_diff_std = scaler.fit_transform(data['freq_diff'].values.reshape(-1, 1))
        
        # Add constant for intercept
        X_speed = sm.add_constant(speed_diff_std)
        X_freq = sm.add_constant(freq_diff_std)
        X_both = sm.add_constant(np.column_stack([speed_diff_std, freq_diff_std]))
        
        # Fit models
        null_model = Logit(y, np.ones_like(y)).fit(disp=0)
        speed_model = Logit(y, X_speed).fit(disp=0)
        freq_model = Logit(y, X_freq).fit(disp=0)
        full_model = Logit(y, X_both).fit(disp=0)
        
        # Calculate McFadden's R² for each model
        # R² = 1 - (log likelihood of model / log likelihood of null model)
        r2_speed = 1 - (speed_model.llf / null_model.llf)
        r2_freq = 1 - (freq_model.llf / null_model.llf)
        r2_full = 1 - (full_model.llf / null_model.llf)
        
        # Use .iloc[] for accessing parameters and p-values by position
        return {
            'speed_logistic_r2': r2_speed * 100,
            'freq_logistic_r2': r2_freq * 100,
            'total_logistic_r2': r2_full * 100,
            'speed_odds_ratio': np.exp(speed_model.params.iloc[1]),  # Changed from [1] to .iloc[1]
            'freq_odds_ratio': np.exp(freq_model.params.iloc[1]),    # Changed from [1] to .iloc[1]
            'speed_p_value': speed_model.pvalues.iloc[1],           # Changed from [1] to .iloc[1]
            'freq_p_value': freq_model.pvalues.iloc[1]              # Changed from [1] to .iloc[1]
        }

    # Report Generation Methods

    def _save_frequency_group_timing_analysis(
        self,
        df_stats: pd.DataFrame,
        correlation: float,
        p_value: float,
        r2: float,
        group_stats: pd.DataFrame,
        output_folder: str,
        filename: str = 'frequency_timing_analysis_results.txt'
    ) -> None:
        """
        Generate comprehensive frequency-timing analysis report.
        
        Args:
            df_stats: DataFrame containing bigram statistics
            correlation: Overall correlation coefficient
            p_value: Overall p-value
            r2: R-squared value
            group_stats: DataFrame containing group statistics
            output_folder: Directory to save report
            filename: Name of output file
        """
        # Perform ANOVA
        groups = [group_data['median_time'].values 
                for _, group_data in df_stats.groupby('freq_group', observed=True)]
        f_stat, anova_p = stats.f_oneway(*groups)
        
        # Generate report content
        report_lines = [
            "Frequency-Timing Analysis Results",
            "==============================\n",
            f"Overall correlation: {correlation:.3f} (p = {p_value:.3e})",
            f"R-squared: {r2:.3f}",
            f"Number of bigrams: {len(df_stats)}",
            f"Total instances: {df_stats['n_samples'].sum()}\n",
            "ANOVA Results:",
            f"F-statistic: {f_stat:.3f}",
            f"p-value: {anova_p:.3e}\n",
            "Group Analysis:"
        ]
        
        # Add group statistics
        for _, group in group_stats.iterrows():
            report_lines.extend([
                f"\nGroup {group['group']}:",
                f"  Frequency range: {group['freq_range']}",
                f"  Mean time: {group['mean_time']:.2f} ms",
                f"  Std dev: {group['timing_std']:.2f} ms",
                f"  Bigrams: {group['n_bigrams']}",
                f"  Instances: {group['total_instances']}"
            ])
        
        # Save report
        with open(os.path.join(output_folder, filename), 'w') as f:
            f.write('\n'.join(report_lines))

    def _save_bigram_time_difference_analysis(
        self,
        data: pd.DataFrame,
        df_stats: pd.DataFrame,
        output_folder: str,
        filename: str = 'time_frequency_analysis.txt'
    ) -> None:
        """
        Generate comprehensive analysis of bigram time differences and their relationship
        with frequency differences.
        
        Args:
            data: DataFrame containing trial data
            df_stats: DataFrame containing per-bigram statistics
            output_folder: Directory to save output
            filename: Name of output file
            
        Saves analysis results including:
        - Raw and normalized time difference correlations
        - Per-user correlation analysis
        - Regression statistics
        - Magnitude analysis
        """
        try:
            # Calculate raw time differences with proper error handling
            data = data.copy()
            data['time_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
            data['time_diff_norm'] = self.stats.normalize_within_participant(data, 'time_diff')
            
            # Safely calculate frequency differences
            bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
            data['freq_diff'] = data.apply(
                lambda row: (
                    np.log10(bigram_freqs.get(row['chosen_bigram'], np.nan)) - 
                    np.log10(bigram_freqs.get(row['unchosen_bigram'], np.nan))
                ),
                axis=1
            )
            
            # Remove rows with missing frequency data
            valid_data = data.dropna(subset=['freq_diff', 'time_diff', 'time_diff_norm'])
            
            if len(valid_data) == 0:
                logger.warning("No valid data pairs found for time-frequency analysis")
                return
                
            # Calculate correlations with error handling
            analysis_results = {
                'raw': self._calculate_correlation_stats(
                    valid_data['freq_diff'], 
                    valid_data['time_diff']
                ),
                'normalized': self._calculate_correlation_stats(
                    valid_data['freq_diff'], 
                    valid_data['time_diff_norm']
                )
            }
            
            # Per-user analysis with minimum trial threshold
            min_trials = self.config['analysis'].get('min_trials_per_participant', 5)
            user_correlations = self._calculate_user_correlations(
                valid_data, 
                min_trials
            )
            
            # Generate report
            report_lines = [
                "Time-Frequency Difference Analysis Results",
                "=======================================\n",
                
                "Raw Analysis:",
                f"Correlation: {analysis_results['raw']['correlation']:.3f}",
                f"P-value: {analysis_results['raw']['p_value']:.3e}",
                f"R-squared: {analysis_results['raw']['r2']:.3f}",
                f"Regression slope: {analysis_results['raw']['slope']:.3f}\n",
                
                "Normalized Analysis:",
                f"Correlation: {analysis_results['normalized']['correlation']:.3f}",
                f"P-value: {analysis_results['normalized']['p_value']:.3e}",
                f"R-squared: {analysis_results['normalized']['r2']:.3f}",
                f"Regression slope: {analysis_results['normalized']['slope']:.3f}\n",
                
                "User-Level Analysis:",
                f"Number of users with {min_trials}+ trials: {len(user_correlations['raw'])}",
                "Raw correlations:",
                f"  Mean: {np.mean(user_correlations['raw']):.3f}",
                f"  Std: {np.std(user_correlations['raw']):.3f}",
                "Normalized correlations:",
                f"  Mean: {np.mean(user_correlations['normalized']):.3f}",
                f"  Std: {np.std(user_correlations['normalized']):.3f}"
            ]
            
            # Save report
            with open(os.path.join(output_folder, filename), 'w') as f:
                f.write('\n'.join(report_lines))
                
        except Exception as e:
            logger.error(f"Error in time difference analysis: {str(e)}")
            raise

    def _save_frequency_group_timing_analysis(
        self,
        df_stats: pd.DataFrame,
        correlation: float,
        p_value: float,
        r2: float,
        group_stats: pd.DataFrame,
        output_folder: str,
        filename: str = 'frequency_timing_analysis_results.txt'
    ):
        """Generate comprehensive frequency-timing analysis report."""
        # Perform ANOVA
        groups = [group_data['median_time'].values 
                for _, group_data in df_stats.groupby('freq_group', observed=True)]
        f_stat, anova_p = stats.f_oneway(*groups)
        
        # Post-hoc analysis
        if anova_p < 0.05:
            post_hoc_data = []
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    meandiff = np.mean(groups[i]) - np.mean(groups[j])
                    t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                    post_hoc_data.append({
                        'group1': f'Group {i + 1}',
                        'group2': f'Group {j + 1}',
                        'meandiff': meandiff,
                        'p_adj': p_val * (len(groups) * (len(groups) - 1) / 2),  # Bonferroni correction
                        'lower': meandiff - 1.96 * np.sqrt(np.var(groups[i])/len(groups[i]) + np.var(groups[j])/len(groups[j])),
                        'upper': meandiff + 1.96 * np.sqrt(np.var(groups[i])/len(groups[i]) + np.var(groups[j])/len(groups[j])),
                        'reject': p_val * (len(groups) * (len(groups) - 1) / 2) < 0.05
                    })
            post_hoc_df = pd.DataFrame(post_hoc_data)
        
        # Generate report
        report_lines = [
            "Bigram Timing Analysis Results",
            "============================\n",
            
            "Overall Correlation Analysis:",
            "--------------------------",
            f"Median correlation: {correlation:.3f} (p = {p_value:.3e})",
            f"Mean correlation: {correlation:.3f} (p = {p_value:.3e})",  # Using same as median for simplicity
            f"R-squared: {r2:.3f}",
            f"Number of unique bigrams: {len(df_stats)}",
            f"Total timing instances: {df_stats['n_samples'].sum()}\n",
            
            "ANOVA Results:",
            "-------------",
            f"F-statistic: {f_stat:.3f}",
            f"p-value: {anova_p:.3e}\n",
            
            "Frequency Group Analysis:",
            "----------------------"
        ]
        
        for _, row in group_stats.iterrows():
            report_lines.extend([
                f"\n{row['group']}:",
                f"  Frequency range: {row['freq_range']}",
                f"  Median typing time: {row['median_time']:.3f} ms",
                f"  Mean typing time: {row['mean_time']:.3f} ms",
                f"  Timing std dev: {row['timing_std']:.3f} ms",
                f"  Number of unique bigrams: {row['n_bigrams']}",
                f"  Total timing instances: {row['total_instances']}"
            ])
        
        if anova_p < 0.05:
            report_lines.extend([
                "\nPost-hoc Analysis (Tukey HSD):",
                "---------------------------"
            ])
            report_lines.append(post_hoc_df.to_string(float_format=lambda x: f"{x:.4f}"))
        
        with open(os.path.join(output_folder, filename), 'w') as f:
            f.write('\n'.join(report_lines))

    def _generate_prediction_report(
        self,
        results: Dict[str, Any],
        output_folder: str,
        filename: str = 'preference_prediction_report.txt'
    ) -> None:
        """
        Generate comprehensive prediction analysis report.
        
        Args:
            results: Dictionary containing prediction results
            output_folder: Directory to save report
            filename: Name of output file
        """
        report_lines = [
            "Speed as Choice Proxy Analysis Report",
            "================================\n",
            f"Number of participants: {results['n_participants']}",
            f"Total trials analyzed: {results['n_trials']}",
            f"Overall prediction accuracy: {results['overall_accuracy']:.1%}\n",
            "Per-Participant Statistics:"
        ]
        
        accuracies = [p['speed_accuracy'] for p in results['participant_results']]
        if accuracies:
            report_lines.extend([
                f"Mean accuracy: {np.mean(accuracies):.1%}",
                f"Median accuracy: {np.median(accuracies):.1%}",
                f"Std deviation: {np.std(accuracies):.1%}",
                f"Range: [{min(accuracies):.1%}, {max(accuracies):.1%}]"
            ])
        
        with open(os.path.join(output_folder, filename), 'w') as f:
            f.write('\n'.join(report_lines))

    def _generate_below_chance_report(
        self,
        below_chance_results: Dict[str, Any],
        pattern_results: Dict[str, Any],
        output_folder: str,
        filename: str = 'below_chance_analysis_report.txt'
    ) -> None:
        """
        Generate report focusing on participants who consistently choose slower bigrams.
        """
        try:
            report_sections = []
            
            # Overview section
            report_sections.extend([
                "Analysis of Participants Who Consistently Choose Slower Bigrams",
                "==========================================================",
                f"Number of participants showing consistent slower-choice patterns: {len(below_chance_results)}",
                ""
            ])
            
            # Summary statistics
            speed_consistent_rates = [results['accuracy'] for results in below_chance_results.values()]
            trials = [results['n_trials'] for results in below_chance_results.values()]
            
            if speed_consistent_rates:
                report_sections.extend([
                    "Overall Statistics:",
                    "-----------------",
                    f"Mean speed-consistency rate: {np.mean(speed_consistent_rates):.1%}",
                    f"  (Lower % means more consistent choice of slower bigrams)",
                    f"Range of speed-consistency: {min(speed_consistent_rates):.1%} - {max(speed_consistent_rates):.1%}",
                    f"Average trials per participant: {np.mean(trials):.1f}",
                    ""
                ])
            
            # Detailed participant analysis
            report_sections.extend([
                "Detailed Analysis Per Participant:",
                "--------------------------------",
                ""
            ])
            
            for user_id, results in below_chance_results.items():
                # Calculate consistent pairs
                consistent_pairs = {
                    pair: res for pair, res in results['pair_results'].items()
                    if res['consistency'] > 0.8
                }
                
                # Calculate percentage of trials involving consistent patterns
                trials_in_consistent_pairs = sum(
                    res['n_presentations'] for res in consistent_pairs.values()
                )
                percent_consistent_trials = (trials_in_consistent_pairs / results['n_trials']) * 100
                
                # Basic participant info
                report_sections.extend([
                    f"Participant {user_id}:",
                    f"  Speed-consistency rate: {results['accuracy']:.1%}",
                    f"  Total trials: {results['n_trials']}",
                    f"  Number of repeated bigram pairs: {results['n_repeat_pairs']}",
                    f"  Number of highly consistent pairs: {len(consistent_pairs)}",
                    f"  Percentage of trials showing consistent patterns: {percent_consistent_trials:.1f}%",
                    ""
                ])
                
                # Add pattern analysis if available
                if user_id in pattern_results:
                    patterns = pattern_results[user_id]
                    
                    if patterns['slower_than_pop']:
                        report_sections.append("  Consistent slower choices (compared to population):")
                        for pair, ratio in patterns['slower_than_pop']:
                            report_sections.append(
                                f"    {pair[0]} vs {pair[1]}: {ratio:.2f}x slower than population median"
                            )
                        report_sections.append("")
                    
                    if patterns['against_frequency']:
                        report_sections.append("  Choices against frequency patterns:")
                        for pair, ratio in patterns['against_frequency']:
                            report_sections.append(
                                f"    {pair[0]} vs {pair[1]}: {ratio:.2f}x less frequent than alternative"
                            )
                        report_sections.append("")
                
                # Analyze highly consistent choices
                if consistent_pairs:
                    report_sections.extend([
                        "  Detailed analysis of consistent patterns (>80% same choice):"
                    ])
                    
                    for pair, pair_results in consistent_pairs.items():
                        chosen = pair_results['chosen_bigram']
                        unchosen = pair[1] if chosen == pair[0] else pair[0]
                        n_presentations = pair_results['n_presentations']
                        percent_of_trials = (n_presentations / results['n_trials']) * 100
                        
                        # Calculate binomial test p-value
                        successes = int(pair_results['consistency'] * n_presentations)
                        binom_result = stats.binomtest(successes, n=n_presentations, p=0.5)
                        
                        report_sections.extend([
                            f"    {chosen} vs {unchosen}:",
                            f"      Choice consistency: {pair_results['consistency']:.1%}",
                            f"      Number of encounters: {n_presentations} ({percent_of_trials:.1f}% of total trials)",
                            f"      Their median time: {pair_results['their_median_time']:.1f}ms",
                            f"      Population median time: {pair_results['pop_median_times'][chosen]:.1f}ms",
                            f"      Frequency ratio: {pair_results['frequencies'][chosen]/pair_results['frequencies'][unchosen]:.2f}",
                            f"      Statistical significance: p = {binom_result.pvalue:.3e}",
                            ""
                        ])
                
                report_sections.append("")  # Add blank line between participants
            
            # Write report
            report_path = os.path.join(output_folder, filename)
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_sections))
            
            logger.info(f"Generated analysis report for consistent slower-choice patterns: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating pattern analysis report: {str(e)}")
            raise

    def _generate_variance_prediction_report(
        self,
        results: Dict[str, float],
        output_folder: str,
        filename: str = 'variance_prediction_analysis.txt'
    ) -> None:
        """Generate comprehensive report of all analyses."""
        
        report_lines = [
            "Variance and Prediction Analysis Results",
            "====================================\n",
            "Confidence Magnitude Variance Explained (R²):",
            f"- By Speed Difference Magnitude: {results['speed_variance']:.1f}%",
            f"- By Frequency Difference Magnitude: {results['frequency_variance']:.1f}%",
            f"- By Frequency (controlling for speed): {results['frequency_partial_variance']:.1f}%\n",
            "Choice Prediction Analysis:",
            f"- Speed-based prediction: {results['speed_predictive']:.1f}% of the time, participants chose the faster bigram",
            f"- Frequency-based prediction: {results['frequency_predictive']:.1f}% of the time, participants chose the more frequent bigram\n",
            "Logistic Regression Analysis:",
            f"- Speed alone explains {results['speed_logistic_r2']:.1f}% of choice variance",
            f"  (odds ratio: {results['speed_odds_ratio']:.2f}, p = {results['speed_p_value']:.3e})",
            f"- Frequency alone explains {results['freq_logistic_r2']:.1f}% of choice variance",
            f"  (odds ratio: {results['freq_odds_ratio']:.2f}, p = {results['freq_p_value']:.3e})",
            f"- Together they explain {results['total_logistic_r2']:.1f}% of choice variance\n",
        ]
        
        with open(os.path.join(output_folder, filename), 'w') as f:
            f.write('\n'.join(report_lines))
                                                           
    # Plotting Methods

    def _plot_typing_times(
        self,
        bigram_data: pd.DataFrame,
        output_folder: str,
        filename: str = 'bigram_times_barplot.png'
    ):
        """Generate and save bar plot for median bigram typing times."""
        # Create figure using plotting utilities
        fig, ax = self.plotter.create_figure('time_diff')
        
        # Prepare data
        plot_data = pd.concat([
            bigram_data[['chosen_bigram', 'chosen_bigram_time']].rename(
                columns={'chosen_bigram': 'bigram', 'chosen_bigram_time': 'time'}),
            bigram_data[['unchosen_bigram', 'unchosen_bigram_time']].rename(
                columns={'unchosen_bigram': 'bigram', 'unchosen_bigram_time': 'time'})
        ])

        # Calculate statistics
        grouped_data = plot_data.groupby('bigram')['time']
        median_times = grouped_data.median().sort_values()
        mad_times = grouped_data.apply(lambda x: median_abs_deviation(x, nan_policy='omit')).reindex(median_times.index)

        # Create bar plot
        bars = ax.bar(
            range(len(median_times)), 
            median_times.values,
            alpha=self.config['visualization']['plots']['time_diff']['alpha']
        )
        
        # Add error bars using plotting utilities
        self.plotter.add_error_bars(
            ax,
            range(len(median_times)),
            median_times.values,
            mad_times.values,
            'time_diff'
        )

        # Customize axes
        ax.set_title('Typing Times for Each Bigram: Median (MAD)', 
                    fontsize=self.config['visualization']['plots']['distribution']['title_fontsize'])
        ax.set_xlabel('Bigram', 
                     fontsize=self.config['visualization']['plots']['distribution']['label_fontsize'])
        ax.set_ylabel('Time (ms)', 
                     fontsize=self.config['visualization']['plots']['distribution']['label_fontsize'])

        # Customize x-axis ticks
        ax.set_xticks(range(len(median_times)))
        ax.set_xticklabels(
            median_times.index,
            rotation=90,
            ha='center',
            fontsize=self.config['visualization']['plots']['frequency']['label_fontsize']
        )
        ax.set_xlim(-0.5, len(median_times) - 0.5)

        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()

    def _plot_chosen_vs_unchosen_scatter(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'chosen_vs_unchosen_times_scatter_regression.png'
    ):
        """Create scatter plot comparing chosen vs unchosen median typing times."""
        # Create figure using plotting utilities
        fig, ax = self.plotter.create_figure('frequency')
        
        # Calculate median times for each bigram pair
        scatter_data = data.groupby('bigram_pair').agg(
            chosen_median=('chosen_bigram_time', 'median'),
            unchosen_median=('unchosen_bigram_time', 'median')
        ).reset_index()
        
        # Create scatter plot
        ax.scatter(
            scatter_data['chosen_median'],
            scatter_data['unchosen_median'],
            alpha=self.config['visualization']['plots']['frequency']['scatter_alpha'],
            color=self.config['visualization']['colors']['primary']
        )
        
        # Add regression line and get statistics
        regression_stats = self.plotter.add_regression_line(
            ax,
            scatter_data['chosen_median'],
            scatter_data['unchosen_median'],
            'frequency'
        )
        
        # Calculate and add correlation info
        correlation, p_value = stats.pearsonr(
            scatter_data['chosen_median'],
            scatter_data['unchosen_median']
        )
        self.plotter.add_correlation_info(ax, correlation, p_value, 'frequency')
        
        # Add diagonal line
        max_val = max(scatter_data['chosen_median'].max(), 
                     scatter_data['unchosen_median'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        # Customize axes
        ax.set_xlabel('Chosen Bigram Median Time (ms)')
        ax.set_ylabel('Unchosen Bigram Median Time (ms)')
        ax.set_title('Chosen vs Unchosen Bigram Median Typing Times')
        
        plt.savefig(os.path.join(output_folder, filename), 
                   dpi=self.config['visualization']['dpi'], 
                   bbox_inches='tight')
        plt.close()

    def _plot_time_diff_slider(
        self,
        data: pd.DataFrame,
        output_folder: str,
    ):
        """Create scatter plots of time differences vs slider values."""
        # Plot normalized version
        fig, ax = self.plotter.create_figure('time_diff')
        
        ax.scatter(
            data['sliderValue'],
            data['time_diff_norm'],
            alpha=self.config['visualization']['plots']['time_diff']['alpha'],
            color=self.config['visualization']['colors']['primary']
        )
        
        # Add regression line and statistics
        regression_stats = self.plotter.add_regression_line(
            ax,
            data['sliderValue'],
            data['time_diff_norm'],
            'time_diff'
        )
        
        ax.set_xlabel('Slider Value')
        ax.set_ylabel('Normalized Time Difference (Chosen - Unchosen)')
        ax.set_title('Normalized Typing Time Difference vs. Slider Value')
        
        plt.savefig(
            os.path.join(output_folder, 'normalized_typing_time_diff_vs_slider_value.png'),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()
        
        # Plot raw version
        fig, ax = self.plotter.create_figure('time_diff')
        
        ax.scatter(
            data['sliderValue'],
            data['time_diff'],
            alpha=self.config['visualization']['plots']['time_diff']['alpha'],
            color=self.config['visualization']['colors']['primary']
        )
        
        self.plotter.add_regression_line(
            ax,
            data['sliderValue'],
            data['time_diff'],
            'time_diff'
        )
        
        ax.set_xlabel('Slider Value')
        ax.set_ylabel('Time Difference (ms) (Chosen - Unchosen)')
        ax.set_title('Raw Typing Time Difference vs. Slider Value')
        
        plt.savefig(
            os.path.join(output_folder, 'raw_typing_time_diff_vs_slider_value.png'),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()

    def _plot_time_diff_histograms(
        self,
        data: pd.DataFrame,
        output_folder: str,
    ):
        """Create histograms of typing time differences by slider value range."""
        for version in ['normalized', 'raw']:
            # Create slider value bins
            slider_ranges = [(-100, -60), (-60, -20), (-20, 20), (20, 60), (60, 100)]
            fig, axes = plt.subplots(2, 3, figsize=self.config['visualization']['plots']['distribution']['figsize'])
            axes = axes.flatten()
            
            # Data to plot based on version
            plot_data = data['time_diff_norm'] if version == 'normalized' else data['time_diff']
            ylabel = 'Normalized Time Difference' if version == 'normalized' else 'Time Difference (ms)'
            
            # Plot histograms for each range
            for i, (low, high) in enumerate(slider_ranges):
                mask = (data['sliderValue'] >= low) & (data['sliderValue'] < high)
                subset = plot_data[mask]
                
                if len(subset) > 0:
                    axes[i].hist(
                        subset,
                        bins=self.config['visualization']['plots']['time_diff']['bins'],
                        alpha=self.config['visualization']['plots']['time_diff']['alpha'],
                        label=f'n={len(subset)}',
                        color=self.config['visualization']['colors']['primary']
                    )
                    
                    axes[i].set_title(f'Slider Values [{low}, {high})')
                    axes[i].set_xlabel(ylabel)
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
                    axes[i].grid(True, alpha=self.config['visualization']['plots']['distribution']['grid_alpha'])
            
            # Add all values plot in last subplot
            axes[-1].hist(
                plot_data,
                bins=self.config['visualization']['plots']['time_diff']['bins'],
                alpha=self.config['visualization']['plots']['time_diff']['alpha'],
                label=f'All values (n={len(plot_data)})',
                color=self.config['visualization']['colors']['primary']
            )
            axes[-1].set_title('All Slider Values')
            axes[-1].set_xlabel(ylabel)
            axes[-1].set_ylabel('Frequency')
            axes[-1].legend()
            axes[-1].grid(True, alpha=self.config['visualization']['plots']['distribution']['grid_alpha'])
            
            plt.suptitle(
                f'Distribution of {"Normalized" if version == "normalized" else "Raw"} '
                f'Time Differences by Slider Value Range',
                y=1.02,
                fontsize=self.config['visualization']['plots']['distribution']['title_fontsize']
            )
            plt.tight_layout()
            
            plt.savefig(
                os.path.join(output_folder, f'{version}_typing_time_diff_vs_slider_value_histograms.png'),
                dpi=self.config['visualization']['dpi'],
                bbox_inches='tight'
            )
            plt.close()

    def _plot_accuracy_by_magnitude(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'speed_accuracy_by_magnitude.png'
    ):
        """Plot prediction accuracy by speed difference magnitude."""
        fig, ax = self.plotter.create_figure('prediction')
        
        # Calculate accuracy by magnitude quintile
        accuracy_by_mag = data.groupby('magnitude_quintile', observed=True)[
            'speed_predicts_choice'
        ].agg(['mean', 'std', 'count'])
        
        # Create error bar plot
        self.plotter.add_error_bars(
            ax,
            range(len(accuracy_by_mag)),
            accuracy_by_mag['mean'],
            accuracy_by_mag['std'],
            'prediction'
        )
        
        ax.plot(
            range(len(accuracy_by_mag)),
            accuracy_by_mag['mean'],
            'o-',
            color=self.config['visualization']['colors']['primary']
        )
        
        # Add sample sizes
        for i, (_, row) in enumerate(accuracy_by_mag.iterrows()):
            ax.text(
                i,
                row['mean'],
                f'n={int(row["count"])}',
                ha='center',
                va='bottom',
                fontsize=self.config['visualization']['plots']['frequency']['label_fontsize']
            )
        
        ax.set_xlabel('Speed Difference Magnitude Quintile')
        ax.set_ylabel('Prediction Accuracy')
        ax.set_title('Speed Prediction Accuracy by Magnitude of Speed Difference')
        
        plt.savefig(
            os.path.join(output_folder, filename),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()

    def _plot_user_accuracy_distribution(
        self,
        participant_results: List[Dict[str, Any]],
        output_folder: str,
        filename: str = 'user_accuracy_distribution.png'
    ):
        """Plot distribution of per-user prediction accuracies."""
        fig, ax = self.plotter.create_figure('prediction')
        
        accuracies = [p['speed_accuracy'] for p in participant_results]
        
        ax.hist(
            accuracies,
            bins=self.config['visualization']['plots']['prediction']['hist_bins'],
            color=self.config['visualization']['colors']['primary']
        )
        
        # Add mean line
        mean_accuracy = np.mean(accuracies)
        ax.axvline(
            mean_accuracy,
            color=self.config['visualization']['colors']['secondary'],
            linestyle='--',
            label=f'Mean: {mean_accuracy:.3f}'
        )
        
        ax.set_xlabel('Prediction Accuracy')
        ax.set_ylabel('Number of Participants')
        ax.set_title('Distribution of Per-Participant Prediction Accuracies')
        ax.legend(fontsize=self.config['visualization']['plots']['distribution']['legend_fontsize'])
        
        plt.savefig(
            os.path.join(output_folder, filename),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()

    def _plot_frequency_timing_relationship(
        self,
        data: pd.DataFrame,
        output_folder: str
    ):
        """Create plots showing relationship between frequency and typing time."""
        # Plot raw timing data
        fig, ax = self.plotter.create_figure('frequency')
        ax.set_xscale('log')
        
        ax.scatter(
            data['frequency'],
            data['time'],
            alpha=self.config['visualization']['plots']['frequency']['scatter_alpha'],
            color=self.config['visualization']['colors']['primary']
        )
        
        # Add regression and correlation
        regression_stats = self.plotter.add_regression_line(
            ax,
            np.log10(data['frequency']),
            data['time'],
            'frequency'
        )
        
        correlation, p_value = stats.spearmanr(
            np.log10(data['frequency']),
            data['time'],
            nan_policy='omit'
        )
        self.plotter.add_correlation_info(ax, correlation, p_value, 'frequency')
        
        ax.set_xlabel('Bigram Frequency (log scale)')
        ax.set_ylabel('Typing Time (ms)')
        ax.set_title('Raw Typing Times vs. Frequency')
        
        plt.savefig(
            os.path.join(output_folder, 'freq_vs_time_raw.png'),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()
        
        # Plot normalized timing data
        fig, ax = self.plotter.create_figure('frequency')
        ax.set_xscale('log')
        
        ax.scatter(
            data['frequency'],
            data['time_norm'],
            alpha=self.config['visualization']['plots']['frequency']['scatter_alpha'],
            color=self.config['visualization']['colors']['primary']
        )
        
        regression_stats = self.plotter.add_regression_line(
            ax,
            np.log10(data['frequency']),
            data['time_norm'],
            'frequency'
        )
        
        correlation, p_value = stats.spearmanr(
            np.log10(data['frequency']),
            data['time_norm'],
            nan_policy='omit'
        )
        self.plotter.add_correlation_info(ax, correlation, p_value, 'frequency')
        
        ax.set_xlabel('Bigram Frequency (log scale)')
        ax.set_ylabel('Normalized Typing Time')
        ax.set_title('Normalized Typing Times vs. Frequency')
        
        plt.savefig(
            os.path.join(output_folder, 'freq_vs_time_normalized.png'),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()

    def _plot_overlaid_time_histograms(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename_prefix: str = ''  # Can be 'raw_' or 'normalized_'
    ) -> None:
        """
        Create histograms of chosen vs unchosen times by slider value range.
        
        Args:
            data: DataFrame containing typing data
            output_folder: Directory to save plot
            filename_prefix: Prefix for output filename ('raw_' or 'normalized_')
        """
        fig, axes = plt.subplots(2, 3, figsize=self.config['visualization']['plots']['distribution']['figsize'])
        axes = axes.flatten()
        
        # Data preparation
        data = data.copy()
        if filename_prefix == 'normalized_':
            data['chosen_bigram_time_norm'] = self.stats.normalize_within_participant(
                data, 'chosen_bigram_time')
            data['unchosen_bigram_time_norm'] = self.stats.normalize_within_participant(
                data, 'unchosen_bigram_time')
            chosen_col = 'chosen_bigram_time_norm'
            unchosen_col = 'unchosen_bigram_time_norm'
            xlabel = 'Normalized Typing Time'
        else:
            chosen_col = 'chosen_bigram_time'
            unchosen_col = 'unchosen_bigram_time'
            xlabel = 'Typing Time (ms)'
        
        slider_ranges = [(-100, -60), (-60, -20), (-20, 20), (20, 60), (60, 100)]
        
        for i, (low, high) in enumerate(slider_ranges):
            ax = axes[i]
            mask = (data['sliderValue'] >= low) & (data['sliderValue'] < high)
            subset = data[mask]
            
            if len(subset) > 0:
                # Plot chosen times
                ax.hist(
                    subset[chosen_col],
                    bins=self.config['visualization']['plots']['time_diff']['bins'],
                    alpha=self.config['visualization']['plots']['time_diff']['alpha'],
                    label='Chosen',
                    color=self.config['visualization']['colors']['primary']
                )
                
                # Plot unchosen times
                ax.hist(
                    subset[unchosen_col],
                    bins=self.config['visualization']['plots']['time_diff']['bins'],
                    alpha=self.config['visualization']['plots']['time_diff']['alpha'],
                    label='Unchosen',
                    color=self.config['visualization']['colors']['secondary']
                )
                
                ax.set_title(f'Slider Values [{low}, {high})')
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=self.config['visualization']['plots']['distribution']['grid_alpha'])
        
        # Add all values plot in last subplot
        axes[-1].hist(
            data[chosen_col],
            bins=self.config['visualization']['plots']['time_diff']['bins'],
            alpha=self.config['visualization']['plots']['time_diff']['alpha'],
            label=f'Chosen (n={len(data)})',
            color=self.config['visualization']['colors']['primary']
        )
        axes[-1].hist(
            data[unchosen_col],
            bins=self.config['visualization']['plots']['time_diff']['bins'],
            alpha=self.config['visualization']['plots']['time_diff']['alpha'],
            label=f'Unchosen (n={len(data)})',
            color=self.config['visualization']['colors']['secondary']
        )
        
        axes[-1].set_title('All Slider Values')
        axes[-1].set_xlabel(xlabel)
        axes[-1].set_ylabel('Frequency')
        axes[-1].legend()
        axes[-1].grid(True, alpha=self.config['visualization']['plots']['distribution']['grid_alpha'])
        
        plt.suptitle(
            f'Distribution of {"Normalized " if filename_prefix == "normalized_" else ""}Typing Times by Slider Value Range',
            y=1.02,
            fontsize=self.config['visualization']['plots']['distribution']['title_fontsize']
        )
        plt.tight_layout()
        
        output_filename = f"{filename_prefix}overlaid_typing_times_by_slider_value_histograms.png"
        plt.savefig(
            os.path.join(output_folder, output_filename),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()

    def _plot_distribution(
        self,
        df_stats: pd.DataFrame,
        output_folder: str,
        filename: str = 'frequency_and_timing_distribution.png'
    ) -> None:
        """
        Create distribution plot with error bars and sample size indicators.
        
        Args:
            df_stats: DataFrame containing bigram statistics
            output_folder: Directory to save plot
            filename: Name of output file
        """
        fig, ax = self.plotter.create_figure('frequency')
        
        # Calculate correlation and regression
        freq_log = np.log10(df_stats['frequency'])
        stats_dict = self._calculate_correlation_stats(freq_log, df_stats['median_time'])
        
        # Create scatter plot with sized points
        sizes = np.clip(df_stats['n_samples'] / 10, 10, 100)
        ax.scatter(
            df_stats['frequency'],
            df_stats['median_time'],
            s=sizes,
            alpha=self.config['visualization']['plots']['frequency']['scatter_alpha'],
            color=self.config['visualization']['colors']['primary']
        )
        
        # Add error bars
        self.plotter.add_error_bars(
            ax,
            df_stats['frequency'],
            df_stats['median_time'],
            df_stats['std_time'],
            'frequency'
        )
        
        # Create evenly spaced points for regression line in log space
        x_log = np.linspace(freq_log.min(), freq_log.max(), 100)
        y_pred = stats_dict['slope'] * x_log + stats_dict['intercept']
        
        # Convert back to linear space for plotting
        ax.plot(
            10**x_log,
            y_pred,
            '--',
            color=self.config['visualization']['colors']['regression'],
            alpha=self.config['visualization']['plots']['frequency']['regression_alpha'],
            label=f'R² = {stats_dict["r2"]:.3f}'
        )
        
        self.plotter.add_correlation_info(
            ax,
            stats_dict['correlation'],
            stats_dict['p_value'],
            'frequency'
        )
        
        ax.set_xscale('log')
        ax.set_xlabel('Bigram Frequency')
        ax.set_ylabel('Median Typing Time (ms)')
        ax.set_title('Distribution of Typing Times vs. Frequency')
        
        # Add legend for sample sizes
        legend_elements = [
            plt.scatter([], [], s=n/10, label=f'n={n} samples', alpha=0.6)
            for n in [10, 50, 100, 500, 1000]
        ]
        legend_elements.append(plt.Line2D([0], [0], linestyle='--', 
                                        color=self.config['visualization']['colors']['regression'],
                                        label=f'R² = {stats_dict["r2"]:.3f}'))
        
        ax.legend(
            handles=legend_elements,
            title='Number of timing samples',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, filename),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()

    def _plot_min_times(
        self,
        df_stats: pd.DataFrame,
        output_folder: str,
        filename: str = 'frequency_and_timing_minimum.png'
    ) -> None:
        """
        Create minimum times plot with bigram labels.
        
        Args:
            df_stats: DataFrame containing bigram statistics
            output_folder: Directory to save plot
            filename: Name of output file
        """
        fig, ax = self.plotter.create_figure('frequency')
        
        # Create scatter plot
        ax.scatter(
            df_stats['frequency'],
            df_stats['min_time'],
            alpha=self.config['visualization']['plots']['frequency']['scatter_alpha'],
            color=self.config['visualization']['colors']['primary']
        )
        
        # Add bigram labels
        for _, row in df_stats.iterrows():
            ax.annotate(
                row['bigram'],
                (row['frequency'], row['min_time']),
                xytext=(2, 2),
                textcoords='offset points',
                fontsize=self.config['visualization']['plots']['frequency']['label_fontsize']
            )
        
        # Add regression line and get statistics
        freq_log = np.log10(df_stats['frequency'])
        stats_dict = self._calculate_correlation_stats(freq_log, df_stats['min_time'])
        
        # Create evenly spaced points for regression line in log space
        x_log = np.linspace(freq_log.min(), freq_log.max(), 100)
        y_pred = stats_dict['slope'] * x_log + stats_dict['intercept']
        
        # Convert back to linear space for plotting
        ax.plot(
            10**x_log,
            y_pred,
            '--',
            color=self.config['visualization']['colors']['regression'],
            alpha=self.config['visualization']['plots']['frequency']['regression_alpha'],
            label=f'R² = {stats_dict["r2"]:.3f}'
        )
        
        self.plotter.add_correlation_info(
            ax,
            stats_dict['correlation'],
            stats_dict['p_value'],
            'frequency'
        )
        
        ax.set_xscale('log')
        ax.set_xlabel('Bigram Frequency (log scale)')
        ax.set_ylabel('Minimum Typing Time (ms)')
        ax.set_title('Fastest Times vs. Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, filename),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()

    def _plot_median_times(
        self,
        df_stats: pd.DataFrame,
        output_folder: str,
        filename: str = 'frequency_and_timing_median.png'
    ) -> None:
        """
        Create median times plot with bigram labels.
        
        Args:
            df_stats: DataFrame containing bigram statistics
            output_folder: Directory to save plot
            filename: Name of output file
        """
        fig, ax = self.plotter.create_figure('frequency')
        
        # Create scatter plot
        ax.scatter(
            df_stats['frequency'],
            df_stats['median_time'],
            alpha=self.config['visualization']['plots']['frequency']['scatter_alpha'],
            color=self.config['visualization']['colors']['primary']
        )
        
        # Add bigram labels
        for _, row in df_stats.iterrows():
            ax.annotate(
                row['bigram'],
                (row['frequency'], row['median_time']),
                xytext=(2, 2),
                textcoords='offset points',
                fontsize=self.config['visualization']['plots']['frequency']['label_fontsize']
            )
        
        # Add regression line and get statistics
        freq_log = np.log10(df_stats['frequency'])
        stats_dict = self._calculate_correlation_stats(freq_log, df_stats['median_time'])
        
        # Create evenly spaced points for regression line in log space
        x_log = np.linspace(freq_log.min(), freq_log.max(), 100)
        y_pred = stats_dict['slope'] * x_log + stats_dict['intercept']
        
        # Convert back to linear space for plotting
        ax.plot(
            10**x_log,  # Convert log space back to linear
            y_pred,
            '--',
            color=self.config['visualization']['colors']['regression'],
            alpha=self.config['visualization']['plots']['frequency']['regression_alpha'],
            label=f'R² = {stats_dict["r2"]:.3f}'
        )
        
        self.plotter.add_correlation_info(
            ax,
            stats_dict['correlation'],
            stats_dict['p_value'],
            'frequency'
        )
        
        ax.set_xscale('log')
        ax.set_xlabel('Bigram Frequency (log scale)')
        ax.set_ylabel('Median Typing Time (ms)')
        ax.set_title('Median Times vs. Frequency')
        ax.legend()
        
        plt.savefig(
            os.path.join(output_folder, filename),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()
        
    def _plot_accuracy_by_confidence(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'speed_accuracy_by_confidence.png'
    ) -> None:
        """
        Plot prediction accuracy by confidence level.
        
        Args:
            data: DataFrame containing prediction data
            output_folder: Directory to save plot
            filename: Name of output file
        """
        fig, ax = self.plotter.create_figure('prediction')
        
        try:
            # Create confidence bins
            data['confidence_level'] = pd.qcut(
                data['confidence'],
                4,
                labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                duplicates='drop'
            )
        except ValueError:
            try:
                data['confidence_level'] = pd.qcut(
                    data['confidence'],
                    3,
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )
            except ValueError:
                # Manual binning if automatic fails
                confidence_bounds = [
                    data['confidence'].min(),
                    data['confidence'].quantile(0.33),
                    data['confidence'].quantile(0.67),
                    data['confidence'].max()
                ]
                data['confidence_level'] = pd.cut(
                    data['confidence'],
                    bins=confidence_bounds,
                    labels=['Low', 'Medium', 'High'],
                    include_lowest=True
                )
        
        # Calculate accuracy by confidence level
        accuracy_by_conf = data.groupby('confidence_level', observed=True)[
            'speed_predicts_choice'
        ].agg(['mean', 'std', 'count'])
        
        # Create bar plot
        x = range(len(accuracy_by_conf))
        ax.bar(
            x,
            accuracy_by_conf['mean'],
            yerr=accuracy_by_conf['std'],
            capsize=self.config['visualization']['plots']['prediction']['error_capsize'],
            color=self.config['visualization']['colors']['primary'],
            alpha=self.config['visualization']['plots']['prediction']['bar_alpha']
        )
        
        # Add sample sizes
        for i, (_, row) in enumerate(accuracy_by_conf.iterrows()):
            ax.text(
                i,
                row['mean'],
                f'n={int(row["count"])}',
                ha='center',
                va='bottom',
                fontsize=self.config['visualization']['plots']['frequency']['label_fontsize']
            )
        
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Prediction Accuracy')
        ax.set_title('Speed Prediction Accuracy by Confidence Level')
        ax.set_xticks(x)
        ax.set_xticklabels(accuracy_by_conf.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, filename),
            dpi=self.config['visualization']['dpi'],
            bbox_inches='tight'
        )
        plt.close()

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
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze bigram typing data')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create output directories
        choice_folder = os.path.join(config['output']['base_dir'], 
                                     config['output']['subdirs']['typing_time_vs_preference'])
        freq_folder = os.path.join(config['output']['base_dir'], 
                                   config['output']['subdirs']['typing_time_vs_frequency'])
        predict_folder = os.path.join(config['output']['base_dir'], 
                                      config['output']['subdirs']['preference_prediction'])
        os.makedirs(choice_folder, exist_ok=True)
        os.makedirs(freq_folder, exist_ok=True)
        os.makedirs(predict_folder, exist_ok=True)

        # Initialize analyzer
        analyzer = BigramAnalysis(config)
        
        # Load data
        data_path = os.path.join(config['data']['input_dir'], 
                                 config['data']['filtered_data_file'])
        data = analyzer.load_and_validate_data(data_path)
        logger.info(f"Processing {len(data)} trials from {data['user_id'].nunique()} participants...")

        # Run analyses and generate plots
        logger.info("Analyzing typing times vs. preference...")
        choice_results = analyzer.analyze_typing_times_slider_values(data, choice_folder)
        
        logger.info("Analyzing typing times vs. frequency...")
        freq_results = analyzer.analyze_frequency_typing_relationship(data, freq_folder)
        
        logger.info("Analyzing preference prediction...")
        prediction_results = analyzer.analyze_speed_choice_prediction(data, predict_folder)
        logger.info(f"    Overall accuracy: {prediction_results['overall_accuracy']:.1%}")

        logger.info("Analyzing variance and prediction...")
        variance_results = analyzer.analyze_variance_and_prediction(data, predict_folder)

        logger.info("Analyzing bigram pair choices...")
        output_path = os.path.join(config['output']['base_dir'], 'bigram_pair_choices.csv')
        pair_stats_df = analyzer.analyze_bigram_pair_choices(data, output_path)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()