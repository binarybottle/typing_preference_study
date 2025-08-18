"""
Bigram Typing Preference Study bigram typing speed, frequency, and choice analysis script

Analyzes four key relationships in typing preference data:
1. Choice and frequency: Do participants choose more frequent bigrams?
2. Speed and choice strength: Do participants choose greater slider bar absolute values faster?
3. Speed and frequency in English: Do participants type more frequent bigrams faster?
4. Speed and choice: Can we use typing speed as a proxy for preference/choice?

Features:
- Statistical reports with effect sizes and confidence intervals
- Clear figures for each relationship
- Robust statistical testing and interpretation
- Results summaries
- Configurable parameters via config.yaml

Usage:
    python analyze_frequency_speed_preference.py [--config config.yaml]

Configuration:
    All analysis parameters, visualization settings, and data paths are specified 
    in config.yaml. See config.yaml for detailed parameter descriptions.
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
import seaborn as sns

from bigram_frequencies import bigrams, bigram_frequencies_array

# Logger will be configured after loading config
logger = logging.getLogger(__name__)

# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

class StatisticalHelpers:
    """Statistical methods for comprehensive analysis."""
    
    @staticmethod
    def calculate_correlation_with_ci(x: Union[np.ndarray, pd.Series], 
                                    y: Union[np.ndarray, pd.Series], 
                                    method: str = 'spearman') -> Dict[str, float]:
        """Calculate correlation with confidence intervals and effect sizes."""
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
                'n_observations': len(x_clean),
                'interpretation': 'Insufficient data'
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
        
        # Calculate R-squared and interpret effect size
        r2 = correlation ** 2
        
        # Interpret correlation strength
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            strength = "negligible"
        elif abs_corr < 0.3:
            strength = "small"
        elif abs_corr < 0.5:
            strength = "medium"
        elif abs_corr < 0.7:
            strength = "large"
        else:
            strength = "very large"
        
        direction = "positive" if correlation > 0 else "negative"
        interpretation = f"{strength} {direction} correlation"
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'r2': r2,
            'n_observations': n,
            'interpretation': interpretation,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def calculate_proportion_test(successes: int, trials: int, 
                                expected_p: float = 0.5) -> Dict[str, float]:
        """Calculate proportion test with comprehensive statistics."""
        if trials == 0:
            return {
                'proportion': np.nan,
                'effect_size': np.nan,
                'p_value': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_trials': 0,
                'significant': False,
                'interpretation': 'No data'
            }
        
        proportion = successes / trials
        
        # Effect size (deviation from expected)
        effect_size = abs(proportion - expected_p)
        
        # Binomial test
        binom_result = stats.binomtest(successes, trials, expected_p, alternative='two-sided')
        p_value = binom_result.pvalue
        
        # Wilson score confidence interval
        ci_lower, ci_upper = StatisticalHelpers._wilson_ci(successes, trials)
        
        # Interpret effect size
        if effect_size < 0.05:
            effect_strength = "negligible"
        elif effect_size < 0.15:
            effect_strength = "small"
        elif effect_size < 0.30:
            effect_strength = "medium"
        else:
            effect_strength = "large"
        
        direction = "above chance" if proportion > expected_p else "below chance"
        interpretation = f"{effect_strength} effect, {direction}"
        
        return {
            'proportion': proportion,
            'effect_size': effect_size,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_trials': trials,
            'significant': p_value < 0.05,
            'interpretation': interpretation
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
# PLOTTING UTILITIES
# =============================================================================

class PlottingUtils:
    """plotting utilities for publication-quality figures."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_style()
        
    def setup_style(self) -> None:
        """Configure global matplotlib style settings."""
        viz_config = self.config['visualization']
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = viz_config['figsize']
        plt.rcParams['figure.dpi'] = viz_config['dpi']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
    def create_correlation_plot(self, x: np.ndarray, y: np.ndarray, 
                              xlabel: str, ylabel: str, title: str,
                              output_path: str, stats_result: Dict[str, Any]) -> None:
        """Create correlation plot with detailed statistics."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot
        ax.scatter(x, y, alpha=0.6, color='steelblue', s=30, edgecolors='white', linewidth=0.5)
        
        # Add regression line
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ax.plot(x, y_pred, '--', color='red', alpha=0.8, linewidth=2)
        
        # Add statistics box
        stats_text = (
            f"Correlation: {stats_result['correlation']:.3f}\n"
            f"95% CI: [{stats_result['ci_lower']:.3f}, {stats_result['ci_upper']:.3f}]\n"
            f"p-value: {stats_result['p_value']:.3e}\n"
            f"RÂ²: {stats_result['r2']:.3f}\n"
            f"N: {stats_result['n_observations']}\n"
            f"Effect: {stats_result['interpretation']}"
        )
        
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                verticalalignment='top', fontsize=11)
        
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
    
    def create_proportion_plot(self, proportion_result: Dict[str, Any], 
                             title: str, output_path: str,
                             success_label: str = "Success",
                             failure_label: str = "Failure") -> None:
        """Create proportion plot with confidence intervals."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Bar chart with confidence interval
        prop = proportion_result['proportion']
        ci_lower = proportion_result['ci_lower']
        ci_upper = proportion_result['ci_upper']
        n_trials = proportion_result['n_trials']
        
        ax1.bar(['Observed'], [prop], color='steelblue', alpha=0.7, width=0.5)
        ax1.errorbar(['Observed'], [prop], 
                    yerr=[[prop - ci_lower], [ci_upper - prop]], 
                    fmt='none', color='black', capsize=10, capthick=2)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance (50%)')
        
        ax1.set_ylabel('Proportion', fontweight='bold')
        ax1.set_title(f'{title} - Proportion Analysis', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # Add statistics text
        stats_text = (
            f"Proportion: {prop:.3f}\n"
            f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\n"
            f"p-value: {proportion_result['p_value']:.3e}\n"
            f"Effect size: {proportion_result['effect_size']:.3f}\n"
            f"N: {n_trials}\n"
            f"Result: {proportion_result['interpretation']}"
        )
        
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                verticalalignment='top', fontsize=10)
        
        # Plot 2: Success/failure counts
        successes = int(prop * n_trials)
        failures = n_trials - successes
        
        ax2.pie([successes, failures], 
               labels=[f'{success_label}\n({successes})', f'{failure_label}\n({failures})'],
               colors=['steelblue', 'lightcoral'], 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'{title} - Count Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()

# =============================================================================
# MAIN ANALYSIS CLASS
# =============================================================================

class BigramAnalysis:
    """analysis class for comprehensive typing preference analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = StatisticalHelpers()
        self.plotter = PlottingUtils(config)
        
        # Create frequency lookup
        self.bigram_frequencies = dict(zip(bigrams, bigram_frequencies_array))

    def run_comprehensive_analysis(self, data_path: str, output_folder: str) -> Dict[str, Any]:
        """Run comprehensive analysis for all four research questions."""
        logger.info("Starting comprehensive bigram preference analysis...")
        
        # Load and prepare data
        data = self.load_and_validate_data(data_path)
        data = self._prepare_analysis_data(data)
        
        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        
        results = {}
        
        # Research Question 1: Choice and Frequency
        logger.info("Analyzing choice and frequency relationship...")
        results['choice_frequency'] = self._analyze_choice_frequency(data, output_folder)
        
        # Research Question 2: Speed and Choice Strength
        logger.info("Analyzing speed and choice strength relationship...")
        results['speed_choice_strength'] = self._analyze_speed_choice_strength(data, output_folder)
        
        # Research Question 3: Speed and Frequency
        logger.info("Analyzing speed and frequency relationship...")
        results['speed_frequency'] = self._analyze_speed_frequency(data, output_folder)
        
        # Research Question 4: Speed and Choice (Prediction)
        logger.info("Analyzing speed as choice predictor...")
        results['speed_choice_prediction'] = self._analyze_speed_choice_prediction(data, output_folder)
        
        # Generate comprehensive report
        logger.info("Generating comprehensive report...")
        self._generate_comprehensive_report(results, data, output_folder)
        
        logger.info("Analysis complete!")
        return results

    # =========================================================================
    # RESEARCH QUESTION ANALYSES
    # =========================================================================
    
    def _analyze_choice_frequency(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """
        Research Question 1: Do participants choose more frequent bigrams?
        """
        # Calculate frequency preference for each choice
        frequency_choices = []
        
        for idx, row in data.iterrows():  # Changed from _, row to idx, row
            chosen_freq = self.bigram_frequencies.get(row['chosen_bigram'], np.nan)
            unchosen_freq = self.bigram_frequencies.get(row['unchosen_bigram'], np.nan)
            
            if not (np.isnan(chosen_freq) or np.isnan(unchosen_freq)):
                frequency_choices.append({
                    'original_index': idx,  # Store the original index
                    'user_id': row['user_id'],
                    'chosen_freq': chosen_freq,
                    'unchosen_freq': unchosen_freq,
                    'chose_more_frequent': chosen_freq > unchosen_freq,
                    'freq_ratio': chosen_freq / unchosen_freq,
                    'log_freq_diff': np.log10(chosen_freq) - np.log10(unchosen_freq),
                    'slider_value': row['sliderValue']  # Store slider value directly
                })
        
        freq_df = pd.DataFrame(frequency_choices)
        
        if len(freq_df) == 0:
            return {'error': 'No valid frequency comparisons found'}
        
        # Statistical test: Do people choose more frequent bigrams?
        n_chose_frequent = freq_df['chose_more_frequent'].sum()
        n_total = len(freq_df)
        
        proportion_result = self.stats.calculate_proportion_test(n_chose_frequent, n_total, 0.5)
        
        # Correlation between frequency difference and choice strength
        correlation_result = self.stats.calculate_correlation_with_ci(
            freq_df['log_freq_diff'], 
            freq_df['slider_value']  # Use the stored slider value
        )
        
        # Create visualizations
        self.plotter.create_proportion_plot(
            proportion_result,
            'Frequency-Based Choice',
            os.path.join(output_folder, 'choice_frequency_proportion.png'),
            'Chose More Frequent',
            'Chose Less Frequent'
        )
        
        self.plotter.create_correlation_plot(
            freq_df['log_freq_diff'],
            freq_df['slider_value'],
            'Log Frequency Difference (Chosen - Unchosen)',
            'Slider Value (Choice Strength)',
            'Frequency Difference vs Choice Strength',
            os.path.join(output_folder, 'frequency_choice_strength_correlation.png'),
            correlation_result
        )
        
        return {
            'proportion_analysis': proportion_result,
            'correlation_analysis': correlation_result,
            'n_comparisons': n_total,
            'summary': f"Participants chose more frequent bigrams {proportion_result['proportion']:.1%} of the time"
        }
    
    def _analyze_speed_choice_strength(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """
        Research Question 2: Do participants choose greater slider bar absolute values faster?
        """
        # Calculate average typing time for each choice
        data['avg_typing_time'] = (data['chosen_bigram_time'] + data['unchosen_bigram_time']) / 2
        data['abs_slider_value'] = data['sliderValue'].abs()
        
        # Remove zero slider values (inconsistent choices)
        speed_strength_data = data[data['abs_slider_value'] > 0].copy()
        
        if len(speed_strength_data) == 0:
            return {'error': 'No consistent choices found'}
        
        # Correlation analysis
        correlation_result = self.stats.calculate_correlation_with_ci(
            speed_strength_data['abs_slider_value'],
            speed_strength_data['avg_typing_time'],
            method='spearman'
        )
        
        # Binned analysis for clearer interpretation
        n_quantiles = self.config['analysis']['n_quantiles']
        try:
            speed_strength_data['strength_bin'] = pd.qcut(
                speed_strength_data['abs_slider_value'], 
                q=n_quantiles, 
                duplicates='drop'
            )
            # Generate labels based on actual number of bins created
            actual_bins = speed_strength_data['strength_bin'].nunique()
            bin_labels = [f'Q{i+1}' for i in range(actual_bins)]
            speed_strength_data['strength_bin'] = pd.qcut(
                speed_strength_data['abs_slider_value'], 
                q=n_quantiles, 
                labels=bin_labels,
                duplicates='drop'
            )
        except ValueError as e:
            logger.warning(f"Could not create {n_quantiles} quantiles for choice strength: {e}")
            # Fallback to fewer bins if needed
            speed_strength_data['strength_bin'] = pd.cut(
                speed_strength_data['abs_slider_value'], 
                bins=min(3, len(speed_strength_data['abs_slider_value'].unique())),
                labels=[f'Bin{i+1}' for i in range(min(3, len(speed_strength_data['abs_slider_value'].unique())))]
            )
        
        bin_stats = speed_strength_data.groupby('strength_bin')['avg_typing_time'].agg([
            'count', 'mean', 'std', 'median'
        ]).round(2)
        
        # Create visualization
        self.plotter.create_correlation_plot(
            speed_strength_data['abs_slider_value'],
            speed_strength_data['avg_typing_time'],
            'Choice Strength (Absolute Slider Value)',
            'Average Typing Time (ms)',
            'Choice Strength vs Typing Speed',
            os.path.join(output_folder, 'speed_choice_strength_correlation.png'),
            correlation_result
        )
        
        # Create binned analysis plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bin_stats['mean'].plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.errorbar(range(len(bin_stats)), bin_stats['mean'], yerr=bin_stats['std'], 
                   fmt='none', color='black', capsize=5)
        ax.set_xlabel('Choice Strength Category', fontweight='bold')
        ax.set_ylabel('Average Typing Time (ms)', fontweight='bold')
        ax.set_title('Typing Speed by Choice Strength Category', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'speed_by_strength_category.png'), 
                   dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'correlation_analysis': correlation_result,
            'bin_analysis': bin_stats.to_dict(),
            'n_observations': len(speed_strength_data),
            'summary': f"Choice strength and typing speed show {correlation_result['interpretation']}"
        }
    
    def _analyze_speed_frequency(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """
        Research Question 3: Do participants type more frequent bigrams faster?
        """
        # Calculate aggregate statistics per bigram
        bigram_stats = []
        for bigram in set(data['chosen_bigram'].unique()) | set(data['unchosen_bigram'].unique()):
            # Combine chosen and unchosen times
            times = pd.concat([
                data[data['chosen_bigram'] == bigram]['chosen_bigram_time'],
                data[data['unchosen_bigram'] == bigram]['unchosen_bigram_time']
            ])
            
            if len(times) > self.config['analysis']['min_bigram_occurrences'] and bigram in self.bigram_frequencies:  # Minimum sample size from config
                bigram_stats.append({
                    'bigram': bigram,
                    'frequency': self.bigram_frequencies[bigram],
                    'log_frequency': np.log10(self.bigram_frequencies[bigram]),
                    'median_time': times.median(),
                    'mean_time': times.mean(),
                    'std_time': times.std(),
                    'n_samples': len(times)
                })
        
        df_stats = pd.DataFrame(bigram_stats)
        
        if len(df_stats) == 0:
            return {'error': 'No valid bigram statistics found'}
        
        # Correlation analysis
        correlation_result = self.stats.calculate_correlation_with_ci(
            df_stats['log_frequency'],
            df_stats['median_time'],
            method='spearman'
        )
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Frequency vs median time with labels
        scatter = ax1.scatter(df_stats['frequency'], df_stats['median_time'], 
                            s=df_stats['n_samples']*2, alpha=0.6, 
                            c=df_stats['log_frequency'], cmap='viridis')
        
        # Add regression line
        slope, intercept = np.polyfit(df_stats['log_frequency'], df_stats['median_time'], 1)
        x_line = np.linspace(df_stats['frequency'].min(), df_stats['frequency'].max(), 100)
        y_line = slope * np.log10(x_line) + intercept
        ax1.plot(x_line, y_line, '--', color='red', alpha=0.8, linewidth=2)
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Bigram Frequency (log scale)', fontweight='bold')
        ax1.set_ylabel('Median Typing Time (ms)', fontweight='bold')
        ax1.set_title('Frequency vs Typing Speed', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Log Frequency', fontweight='bold')
        
        # Plot 2: Frequency quantiles analysis
        n_quantiles = self.config['analysis']['n_quantiles']
        try:
            df_stats['freq_quantile'] = pd.qcut(df_stats['frequency'], q=n_quantiles, 
                                              duplicates='drop')
            # Generate labels based on actual number of bins created
            actual_bins = df_stats['freq_quantile'].nunique()
            bin_labels = [f'Q{i+1}' for i in range(actual_bins)]
            df_stats['freq_quantile'] = pd.qcut(df_stats['frequency'], q=n_quantiles, 
                                              labels=bin_labels,
                                              duplicates='drop')
        except ValueError as e:
            logger.warning(f"Could not create {n_quantiles} quantiles for frequency: {e}")
            # Fallback to fewer bins
            df_stats['freq_quantile'] = pd.cut(df_stats['frequency'], 
                                             bins=min(3, len(df_stats['frequency'].unique())),
                                             labels=[f'Bin{i+1}' for i in range(min(3, len(df_stats['frequency'].unique())))])
        quantile_stats = df_stats.groupby('freq_quantile')['median_time'].agg(['mean', 'std', 'count'])
        
        quantile_stats['mean'].plot(kind='bar', ax=ax2, color='steelblue', alpha=0.7)
        ax2.errorbar(range(len(quantile_stats)), quantile_stats['mean'], 
                    yerr=quantile_stats['std'], fmt='none', color='black', capsize=5)
        ax2.set_xlabel('Frequency Quantile', fontweight='bold')
        ax2.set_ylabel('Average Median Time (ms)', fontweight='bold')
        ax2.set_title('Typing Speed by Frequency Category', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'frequency_speed_analysis.png'), 
                   dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'correlation_analysis': correlation_result,
            'quantile_analysis': quantile_stats.to_dict(),
            'n_bigrams': len(df_stats),
            'total_samples': df_stats['n_samples'].sum(),
            'summary': f"Frequency and typing speed show {correlation_result['interpretation']}"
        }
    
    def _analyze_speed_choice_prediction(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """
        Research Question 4: Can we use typing speed as a proxy for preference/choice?
        """
        # Calculate speed difference and prediction
        data['speed_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['speed_predicts_choice'] = data['speed_diff'] < 0  # Faster = chosen
        
        # Overall prediction accuracy
        overall_prediction = self.stats.calculate_proportion_test(
            data['speed_predicts_choice'].sum(),
            len(data),
            0.5
        )
        
        # Per-participant analysis
        min_trials = self.config['analysis']['min_trials_per_participant']
        participant_results = []
        
        for user_id, user_data in data.groupby('user_id'):
            if len(user_data) >= min_trials:
                user_accuracy = user_data['speed_predicts_choice'].mean()
                participant_results.append({
                    'user_id': user_id,
                    'n_trials': len(user_data),
                    'accuracy': user_accuracy,
                    'above_chance': user_accuracy > 0.5
                })
        
        participant_df = pd.DataFrame(participant_results)
        
        # Test if participants are above chance
        if len(participant_df) > 0:
            participant_prediction = self.stats.calculate_proportion_test(
                participant_df['above_chance'].sum(),
                len(participant_df),
                0.5
            )
        else:
            participant_prediction = {'error': 'Insufficient participant data'}
        
        # Correlation between speed difference and choice strength
        speed_choice_correlation = self.stats.calculate_correlation_with_ci(
            data['speed_diff'],
            data['sliderValue'],
            method='spearman'
        )
        
        # Create comprehensive visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Overall prediction accuracy
        overall_prop = overall_prediction['proportion']
        ax1.bar(['Speed Prediction'], [overall_prop], color='steelblue', alpha=0.7)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance')
        ax1.set_ylabel('Prediction Accuracy', fontweight='bold')
        ax1.set_title('Overall Speed-Based Prediction Accuracy', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.legend()
        
        # Add statistics
        stats_text = f"Accuracy: {overall_prop:.3f}\nCI: [{overall_prediction['ci_lower']:.3f}, {overall_prediction['ci_upper']:.3f}]\np: {overall_prediction['p_value']:.3e}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                verticalalignment='top')
        
        # Plot 2: Participant accuracy distribution
        if len(participant_df) > 0:
            ax2.hist(participant_df['accuracy'], bins=20, alpha=0.7, color='steelblue')
            ax2.axvline(participant_df['accuracy'].mean(), color='red', linestyle='--', 
                       label=f"Mean: {participant_df['accuracy'].mean():.3f}")
            ax2.axvline(0.5, color='gray', linestyle=':', label='Chance')
            ax2.set_xlabel('Prediction Accuracy', fontweight='bold')
            ax2.set_ylabel('Number of Participants', fontweight='bold')
            ax2.set_title('Distribution of Participant Accuracies', fontweight='bold')
            ax2.legend()
        
        # Plot 3: Speed difference vs choice strength
        ax3.scatter(data['speed_diff'], data['sliderValue'], alpha=0.5, s=20)
        slope, intercept = np.polyfit(data['speed_diff'], data['sliderValue'], 1)
        x_line = np.linspace(data['speed_diff'].min(), data['speed_diff'].max(), 100)
        y_line = slope * x_line + intercept
        ax3.plot(x_line, y_line, '--', color='red', alpha=0.8)
        ax3.set_xlabel('Speed Difference (Chosen - Unchosen) ms', fontweight='bold')
        ax3.set_ylabel('Choice Strength (Slider Value)', fontweight='bold')
        ax3.set_title('Speed Difference vs Choice Strength', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction accuracy by speed difference magnitude
        data['speed_diff_abs'] = data['speed_diff'].abs()
        n_quantiles = self.config['analysis']['n_quantiles']
        try:
            data['speed_quintile'] = pd.qcut(data['speed_diff_abs'], q=n_quantiles, 
                                           duplicates='drop')
            # Generate labels based on actual number of bins created
            actual_bins = data['speed_quintile'].nunique()
            bin_labels = [f'Q{i+1}' for i in range(actual_bins)]
            data['speed_quintile'] = pd.qcut(data['speed_diff_abs'], q=n_quantiles, 
                                           labels=bin_labels,
                                           duplicates='drop')
        except ValueError as e:
            logger.warning(f"Could not create {n_quantiles} quantiles for speed difference: {e}")
            # Fallback to fewer bins
            data['speed_quintile'] = pd.cut(data['speed_diff_abs'], 
                                          bins=min(3, len(data['speed_diff_abs'].unique())),
                                          labels=[f'Bin{i+1}' for i in range(min(3, len(data['speed_diff_abs'].unique())))])
        quintile_accuracy = data.groupby('speed_quintile')['speed_predicts_choice'].mean()
        
        quintile_accuracy.plot(kind='bar', ax=ax4, color='steelblue', alpha=0.7)
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance')
        ax4.set_xlabel('Speed Difference Magnitude Quintile', fontweight='bold')
        ax4.set_ylabel('Prediction Accuracy', fontweight='bold')
        ax4.set_title('Accuracy by Speed Difference Magnitude', fontweight='bold')
        ax4.legend()
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'speed_choice_prediction_analysis.png'), 
                   dpi=self.config['visualization']['dpi'], bbox_inches='tight')
        plt.close()
        
        return {
            'overall_prediction': overall_prediction,
            'participant_prediction': participant_prediction,
            'speed_choice_correlation': speed_choice_correlation,
            'n_participants': len(participant_df),
            'quintile_accuracy': quintile_accuracy.to_dict(),
            'summary': f"Speed predicts choice {overall_prediction['proportion']:.1%} of the time ({overall_prediction['interpretation']})"
        }

    # =========================================================================
    # HELPER METHODS
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

    def _prepare_analysis_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis by applying filters and transformations."""
        # Apply time limit from config
        max_time = self.config['analysis']['max_time_ms']
        data = data.copy()
        
        # Clip extreme typing times
        chosen_clipped = len(data[data['chosen_bigram_time'] > max_time])
        unchosen_clipped = len(data[data['unchosen_bigram_time'] > max_time])
        
        data['chosen_bigram_time'] = data['chosen_bigram_time'].clip(upper=max_time)
        data['unchosen_bigram_time'] = data['unchosen_bigram_time'].clip(upper=max_time)
        
        if chosen_clipped + unchosen_clipped > 0:
            logger.info(f"Clipped {chosen_clipped + unchosen_clipped} extreme typing times to {max_time}ms")
        
        # Convert bigrams to lowercase for consistency
        data['chosen_bigram'] = data['chosen_bigram'].str.lower()
        data['unchosen_bigram'] = data['unchosen_bigram'].str.lower()
        
        return data

    def _generate_comprehensive_report(self, results: Dict[str, Any], 
                                     data: pd.DataFrame, output_folder: str) -> None:
        """Generate comprehensive statistical report."""
        
        report_lines = [
            "COMPREHENSIVE BIGRAM TYPING PREFERENCE ANALYSIS REPORT",
            "=" * 70,
            "",
            "EXECUTIVE SUMMARY",
            "================",
            "",
            f"This analysis examined four key relationships in typing preference data",
            f"from {len(data)} trials across {data['user_id'].nunique()} participants.",
            "",
            "KEY FINDINGS:",
            "============",
            ""
        ]
        
        # Add findings for each research question
        for i, (key, title, summary_key) in enumerate([
            ('choice_frequency', 'Choice and Frequency', 'summary'),
            ('speed_choice_strength', 'Speed and Choice Strength', 'summary'),
            ('speed_frequency', 'Speed and Frequency', 'summary'),
            ('speed_choice_prediction', 'Speed and Choice Prediction', 'summary')
        ], 1):
            if key in results and 'error' not in results[key]:
                result = results[key]
                summary = result.get(summary_key, 'No summary available')
                report_lines.extend([
                    f"{i}. {title}:",
                    f"   {summary}",
                    ""
                ])
        
        # Detailed results for each research question
        report_lines.extend([
            "DETAILED RESULTS",
            "===============",
            ""
        ])
        
        # Research Question 1: Choice and Frequency
        if 'choice_frequency' in results and 'error' not in results['choice_frequency']:
            result = results['choice_frequency']
            prop_analysis = result['proportion_analysis']
            corr_analysis = result['correlation_analysis']
            
            report_lines.extend([
                "1. CHOICE AND FREQUENCY ANALYSIS",
                "-" * 40,
                f"Research Question: Do participants choose more frequent bigrams?",
                "",
                f"PROPORTION ANALYSIS:",
                f"  Proportion choosing more frequent: {prop_analysis['proportion']:.3f}",
                f"  95% Confidence Interval: [{prop_analysis['ci_lower']:.3f}, {prop_analysis['ci_upper']:.3f}]",
                f"  Statistical significance: p = {prop_analysis['p_value']:.3e}",
                f"  Effect size: {prop_analysis['effect_size']:.3f}",
                f"  Sample size: {prop_analysis['n_trials']} comparisons",
                f"  Interpretation: {prop_analysis['interpretation']}",
                "",
                f"CORRELATION ANALYSIS (Frequency difference vs Choice strength):",
                f"  Correlation: {corr_analysis['correlation']:.3f}",
                f"  95% Confidence Interval: [{corr_analysis['ci_lower']:.3f}, {corr_analysis['ci_upper']:.3f}]",
                f"  Statistical significance: p = {corr_analysis['p_value']:.3e}",
                f"  R-squared: {corr_analysis['r2']:.3f}",
                f"  Interpretation: {corr_analysis['interpretation']}",
                ""
            ])
        
        # Research Question 2: Speed and Choice Strength
        if 'speed_choice_strength' in results and 'error' not in results['speed_choice_strength']:
            result = results['speed_choice_strength']
            corr_analysis = result['correlation_analysis']
            
            report_lines.extend([
                "2. SPEED AND CHOICE STRENGTH ANALYSIS",
                "-" * 40,
                f"Research Question: Do participants make stronger choices faster?",
                "",
                f"CORRELATION ANALYSIS:",
                f"  Correlation: {corr_analysis['correlation']:.3f}",
                f"  95% Confidence Interval: [{corr_analysis['ci_lower']:.3f}, {corr_analysis['ci_upper']:.3f}]",
                f"  Statistical significance: p = {corr_analysis['p_value']:.3e}",
                f"  R-squared: {corr_analysis['r2']:.3f}",
                f"  Sample size: {corr_analysis['n_observations']} observations",
                f"  Interpretation: {corr_analysis['interpretation']}",
                ""
            ])
        
        # Research Question 3: Speed and Frequency
        if 'speed_frequency' in results and 'error' not in results['speed_frequency']:
            result = results['speed_frequency']
            corr_analysis = result['correlation_analysis']
            
            report_lines.extend([
                "3. SPEED AND FREQUENCY ANALYSIS",
                "-" * 40,
                f"Research Question: Do participants type more frequent bigrams faster?",
                "",
                f"CORRELATION ANALYSIS:",
                f"  Correlation: {corr_analysis['correlation']:.3f}",
                f"  95% Confidence Interval: [{corr_analysis['ci_lower']:.3f}, {corr_analysis['ci_upper']:.3f}]",
                f"  Statistical significance: p = {corr_analysis['p_value']:.3e}",
                f"  R-squared: {corr_analysis['r2']:.3f}",
                f"  Sample size: {result['n_bigrams']} bigrams, {result['total_samples']} total observations",
                f"  Interpretation: {corr_analysis['interpretation']}",
                ""
            ])
        
        # Research Question 4: Speed and Choice Prediction
        if 'speed_choice_prediction' in results and 'error' not in results['speed_choice_prediction']:
            result = results['speed_choice_prediction']
            overall_pred = result['overall_prediction']
            participant_pred = result.get('participant_prediction', {})
            speed_corr = result['speed_choice_correlation']
            
            report_lines.extend([
                "4. SPEED AND CHOICE PREDICTION ANALYSIS",
                "-" * 40,
                f"Research Question: Can typing speed predict preference/choice?",
                "",
                f"OVERALL PREDICTION ACCURACY:",
                f"  Accuracy: {overall_pred['proportion']:.3f}",
                f"  95% Confidence Interval: [{overall_pred['ci_lower']:.3f}, {overall_pred['ci_upper']:.3f}]",
                f"  Statistical significance: p = {overall_pred['p_value']:.3e}",
                f"  Effect size: {overall_pred['effect_size']:.3f}",
                f"  Sample size: {overall_pred['n_trials']} trials",
                f"  Interpretation: {overall_pred['interpretation']}",
                ""
            ])
            
            if 'error' not in participant_pred:
                report_lines.extend([
                    f"PARTICIPANT-LEVEL ANALYSIS:",
                    f"  Participants above chance: {participant_pred['proportion']:.3f}",
                    f"  95% Confidence Interval: [{participant_pred['ci_lower']:.3f}, {participant_pred['ci_upper']:.3f}]",
                    f"  Statistical significance: p = {participant_pred['p_value']:.3e}",
                    f"  Sample size: {result['n_participants']} participants",
                    ""
                ])
            
            report_lines.extend([
                f"SPEED-CHOICE CORRELATION:",
                f"  Correlation: {speed_corr['correlation']:.3f}",
                f"  95% Confidence Interval: [{speed_corr['ci_lower']:.3f}, {speed_corr['ci_upper']:.3f}]",
                f"  Statistical significance: p = {speed_corr['p_value']:.3e}",
                f"  R-squared: {speed_corr['r2']:.3f}",
                f"  Interpretation: {speed_corr['interpretation']}",
                ""
            ])
        
        # Methodology and interpretation
        report_lines.extend([
            "METHODOLOGY",
            "===========",
            "",
            "Statistical Methods:",
            "• Spearman correlations for non-parametric relationships",
            "• Binomial tests for proportion analyses",
            "• Wilson score confidence intervals for robust interval estimation",
            "• Effect sizes calculated as deviations from chance or correlation strength",
            "",
            "Effect Size Interpretation:",
            "• Correlations: |r| < 0.1 (negligible), 0.1-0.3 (small), 0.3-0.5 (medium), >0.5 (large)",
            "• Proportions: deviation from 0.5 < 0.05 (negligible), 0.05-0.15 (small), 0.15-0.3 (medium), >0.3 (large)",
            "",
            "Data Processing:",
            f"• Maximum typing time clipped at {self.config['analysis']['max_time_ms']}ms",
            f"• Minimum {self.config['analysis']['min_trials_per_participant']} trials per participant for individual analysis",
            "• All bigram comparisons converted to lowercase for consistency",
            "",
            "GENERATED FILES",
            "==============",
            "",
            "Visualizations:",
            "• choice_frequency_proportion.png - Frequency-based choice analysis",
            "• frequency_choice_strength_correlation.png - Frequency vs choice strength",
            "• speed_choice_strength_correlation.png - Speed vs choice strength",
            "• speed_by_strength_category.png - Speed by choice strength categories",
            "• frequency_speed_analysis.png - Frequency vs typing speed analysis",
            "• speed_choice_prediction_analysis.png - Speed-based prediction analysis",
            "",
            "This report provides comprehensive statistical analysis of typing preferences",
            "with robust effect size estimates and confidence intervals for all key relationships."
        ])
        
        # Save report
        report_path = os.path.join(output_folder, 'comprehensive_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Comprehensive report saved to {report_path}")

# =============================================================================
# CONFIGURATION AND MAIN FUNCTION
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Validate required sections exist
        required_sections = ['data', 'analysis', 'visualization']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config
            
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        raise

def main():
    """Main analysis function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze bigram typing preference relationships')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Configure logging based on config
        if 'logging' in config:
            log_config = config['logging']
            log_level = getattr(logging, log_config.get('level', 'INFO').upper())
            log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Create log directory if specified
            if 'file' in log_config:
                log_file = log_config['file']
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                logging.basicConfig(level=log_level, format=log_format, 
                                  handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
            else:
                logging.basicConfig(level=log_level, format=log_format)
        else:
            # Fallback logging configuration
            logging.basicConfig(level=logging.INFO, 
                              format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        input_folder = config['data']['input_dir']
        output_folder = os.path.join(input_folder, 'output', 'preference_analysis')

        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        # Initialize analyzer
        analyzer = BigramAnalysis(config)
        
        # Load data
        data_path = os.path.join(input_folder, config['data']['filtered_data_file'])
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis(data_path, output_folder)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE - KEY FINDINGS:")
        print("="*60)
        
        for i, (key, title) in enumerate([
            ('choice_frequency', 'Choice and Frequency'),
            ('speed_choice_strength', 'Speed and Choice Strength'),
            ('speed_frequency', 'Speed and Frequency'),
            ('speed_choice_prediction', 'Speed and Choice Prediction')
        ], 1):
            if key in results and 'error' not in results[key]:
                summary = results[key].get('summary', 'Analysis completed')
                print(f"{i}. {title}: {summary}")
        
        print(f"\nDetailed results and visualizations saved to: {output_folder}")
        print("="*60)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()