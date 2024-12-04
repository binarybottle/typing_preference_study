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
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

from bigram_frequencies import bigrams, bigram_frequencies_array

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobustStatistics:
    """Core statistical methods for analysis."""
    
    @staticmethod
    def normalize_within_participant(
        data: pd.DataFrame,
        value_column: str,
        user_column: str = 'user_id'
    ) -> pd.Series:
        """Normalize values within each participant using robust statistics."""
        return data.groupby(user_column, observed=True)[value_column].transform(
            lambda x: (x - x.median()) / (median_abs_deviation(x, nan_policy='omit') + 1e-10)
        )
    
    @staticmethod
    def compute_confidence_intervals(
        values: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence intervals using bootstrap."""
        if len(values) < 2:
            return np.nan, np.nan
        
        try:
            bootstrap_samples = np.random.choice(
                values, 
                size=(1000, len(values)), 
                replace=True
            )
            bootstrap_medians = np.median(bootstrap_samples, axis=1)
            ci_lower = np.percentile(bootstrap_medians, (1 - confidence) / 2 * 100)
            ci_upper = np.percentile(bootstrap_medians, (1 + confidence) / 2 * 100)
            return ci_lower, ci_upper
        except:
            return np.nan, np.nan

class BigramAnalysis:
    """Main analysis class combining legacy and enhanced approaches."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = RobustStatistics()
        self.setup_plotting()
        
    def setup_plotting(self):
        """Configure matplotlib plotting style."""
        plt.style.use(self.config['visualization']['style'])
        plt.rcParams['figure.figsize'] = self.config['visualization']['figsize']
        plt.rcParams['figure.dpi'] = self.config['visualization']['dpi']
    
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

    def _apply_time_limit(self, data: pd.DataFrame) -> pd.DataFrame:
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
        
    def analyze_typing_times_slider_values(
        self,
        data: pd.DataFrame,
        output_folder: str,
    ) -> Dict[str, Any]:
        """
        Analyze typing times in relation to slider values.
        Enhanced version of original analysis with robust statistics.
        """
        results = {}
        
        # Create copy and apply time limit
        data = data.copy()
        data = self._apply_time_limit(data)
        
        # Calculate speed differences
        data['time_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['time_diff_norm'] = self.stats.normalize_within_participant(data, 'time_diff')
        
        # Generate plots
        self._plot_chosen_vs_unchosen_scatter(data, output_folder)
        self._plot_time_diff_slider(data, output_folder)
        self._plot_time_diff_histograms(data, output_folder)
        
        # Generate overlaid histogram plots
        self._plot_overlaid_time_histograms(data, output_folder, 'raw_')
        self._plot_overlaid_time_histograms(data, output_folder, 'normalized_')
        
        self._plot_chosen_vs_unchosen_scatter(data, output_folder)

        return results

    def _plot_chosen_vs_unchosen_scatter(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'chosen_vs_unchosen_times_scatter_regression.png'
    ):
        """Create scatter plot comparing chosen vs unchosen median typing times."""
        plt.figure(figsize=(10, 8))
        
        # Calculate median times for each bigram pair
        scatter_data = data.groupby('bigram_pair').agg(
            chosen_median=('chosen_bigram_time', 'median'),
            unchosen_median=('unchosen_bigram_time', 'median')
        ).reset_index()
        
        # Calculate correlation
        correlation = scatter_data['chosen_median'].corr(scatter_data['unchosen_median'])
        
        # Create scatter plot
        plt.scatter(
            scatter_data['chosen_median'],
            scatter_data['unchosen_median'],
            alpha=0.7,
            color='#1f77b4' 
        )
        
        # Add diagonal line
        max_val = max(scatter_data['chosen_median'].max(), scatter_data['unchosen_median'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        # Add correlation text
        plt.text(0.05, 0.8, f'Correlation: {correlation:.2f}', 
                transform=plt.gca().transAxes)
        
        plt.xlabel('Chosen Bigram Median Time (ms)')
        plt.ylabel('Unchosen Bigram Median Time (ms)')
        plt.title('Chosen vs Unchosen Bigram Median Typing Times')
        
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_time_diff_slider(
        self,
        data: pd.DataFrame,
        output_folder: str,
    ):
        """Create scatter plots of time differences vs slider values."""
        # Plot normalized version
        plt.figure(figsize=(10, 6))
        plt.scatter(
            data['sliderValue'],
            data['time_diff_norm'],
            alpha=0.5,
            color=self.config['visualization']['colors']['primary']
        )
        
        plt.xlabel('Slider Value')
        plt.ylabel('Normalized Time Difference (Chosen - Unchosen)')
        plt.title('Normalized Typing Time Difference vs. Slider Value')
        
        plt.savefig(os.path.join(output_folder, 'normalized_typing_time_diff_vs_slider_value.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot raw version
        plt.figure(figsize=(10, 6))
        plt.scatter(
            data['sliderValue'],
            data['time_diff'],
            alpha=0.5,
            color=self.config['visualization']['colors']['primary']
        )
        
        plt.xlabel('Slider Value')
        plt.ylabel('Time Difference (ms) (Chosen - Unchosen)')
        plt.title('Raw Typing Time Difference vs. Slider Value')
        
        plt.savefig(os.path.join(output_folder, 'raw_typing_time_diff_vs_slider_value.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_time_diff_histograms(
        self,
        data: pd.DataFrame,
        output_folder: str,
    ):
        """Create histograms of typing time differences by slider value range."""
        # Create both normalized and raw versions
        for version in ['normalized', 'raw']:
            plt.figure(figsize=(15, 10))
            
            # Create slider value bins
            slider_ranges = [(-100, -60), (-60, -20), (-20, 20), (20, 60), (60, 100)]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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
                        bins=30,
                        alpha=0.5,
                        label=f'n={len(subset)}',
                        color='blue'
                    )
                    
                    axes[i].set_title(f'Slider Values [{low}, {high})')
                    axes[i].set_xlabel(ylabel)
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            # Add all values plot in last subplot
            axes[-1].hist(
                plot_data,
                bins=30,
                alpha=0.5,
                label=f'All values (n={len(plot_data)})',
                color='blue'
            )
            axes[-1].set_title('All Slider Values')
            axes[-1].set_xlabel(ylabel)
            axes[-1].set_ylabel('Frequency')
            axes[-1].legend()
            axes[-1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Distribution of {"Normalized" if version == "normalized" else "Raw"} '
                        f'Time Differences by Slider Value Range', y=1.02)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_folder, 
                                    f'{version}_typing_time_diff_vs_slider_value_histograms.png'),
                    dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_overlaid_time_histograms(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename_prefix: str = ''  # Can be 'raw_' or 'normalized_'
    ):
        """Create histograms of chosen vs unchosen times by slider value range."""
        plt.figure(figsize=(15, 10))
            
        # Create slider value bins
        slider_ranges = [(-100, -60), (-60, -20), (-20, 20), (20, 60), (60, 100)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Data preparation
        data = data.copy()
        if filename_prefix == 'normalized_':
            # Create normalized versions of the columns
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
        
        for i, (low, high) in enumerate(slider_ranges):
            mask = (data['sliderValue'] >= low) & (data['sliderValue'] < high)
            subset = data[mask]
            
            if len(subset) > 0:
                # Plot chosen times
                axes[i].hist(
                    subset[chosen_col],
                    bins=30,
                    alpha=0.5,
                    label='Chosen',
                    color='blue'
                )
                
                # Plot unchosen times
                axes[i].hist(
                    subset[unchosen_col],
                    bins=30,
                    alpha=0.5,
                    label='Unchosen',
                    color='red'
                )
                
                axes[i].set_title(f'Slider Values [{low}, {high})')
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Add all values plot in last subplot
        axes[-1].hist(
            data[chosen_col],
            bins=30,
            alpha=0.5,
            label=f'Chosen (n={len(data)})',
            color='blue'
        )
        axes[-1].hist(
            data[unchosen_col],
            bins=30,
            alpha=0.5,
            label=f'Unchosen (n={len(data)})',
            color='red'
        )
        axes[-1].set_title('All Slider Values')
        axes[-1].set_xlabel(xlabel)
        axes[-1].set_ylabel('Frequency')
        axes[-1].legend()
        axes[-1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Distribution of {"Normalized " if filename_prefix == "normalized_" else ""}Typing Times by Slider Value Range', y=1.02)
        plt.tight_layout()
        
        output_filename = f"{filename_prefix}overlaid_typing_times_by_slider_value_histograms.png"
        plt.savefig(os.path.join(output_folder, output_filename), dpi=300, bbox_inches='tight')
        plt.close()
            
    def _plot_chosen_vs_unchosen_scatter(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'chosen_vs_unchosen_times_scatter_regression.png'
    ):
        """Create scatter plot comparing chosen vs unchosen median typing times."""
        plt.figure(figsize=(10, 8))
        
        # Calculate median times for each bigram pair
        scatter_data = data.groupby('bigram_pair').agg(
            chosen_median=('chosen_bigram_time', 'median'),
            unchosen_median=('unchosen_bigram_time', 'median')
        ).reset_index()
        
        # Calculate correlation and p-value
        correlation, p_value = stats.pearsonr(scatter_data['chosen_median'], 
                                            scatter_data['unchosen_median'])
        
        # Calculate R²
        slope, intercept = np.polyfit(scatter_data['chosen_median'], 
                                    scatter_data['unchosen_median'], 1)
        y_pred = slope * scatter_data['chosen_median'] + intercept
        r2 = 1 - (np.sum((scatter_data['unchosen_median'] - y_pred) ** 2) / 
                np.sum((scatter_data['unchosen_median'] - scatter_data['unchosen_median'].mean()) ** 2))
        
        # Create scatter plot
        plt.scatter(
            scatter_data['chosen_median'],
            scatter_data['unchosen_median'],
            alpha=0.7,
            color='#1f77b4'
        )
        
        # Add diagonal line
        max_val = max(scatter_data['chosen_median'].max(), scatter_data['unchosen_median'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
        
        # Add correlation info
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}\np-value: {p_value:.2e}', 
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add R² in legend
        plt.plot([], [], 'r--', label=f'R² = {r2:.3f}')
        plt.legend()
        
        plt.xlabel('Chosen Bigram Median Time (ms)')
        plt.ylabel('Unchosen Bigram Median Time (ms)')
        plt.title('Chosen vs Unchosen Bigram Median Typing Times')
        
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
            
    def analyze_frequency_typing_relationship(
        self,
        data: pd.DataFrame,
        output_folder: str
    ) -> Dict[str, Any]:
        """
        Analyze relationship between bigram frequency and typing times.
        Enhanced version with per-participant normalization.
        """
        # Create frequency dictionary
        bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
        
        # Calculate normalized typing times per participant
        data = data.copy()
        data = self._apply_time_limit(data)  # Unpack both return values
        
        for col in ['chosen_bigram_time', 'unchosen_bigram_time']:
            data[f'{col}_norm'] = self.stats.normalize_within_participant(data, col)

        # Combine chosen and unchosen data
        typing_data = pd.concat([
            pd.DataFrame({
                'bigram': data['chosen_bigram'],
                'time': data['chosen_bigram_time'],
                'time_norm': data['chosen_bigram_time_norm']
            }),
            pd.DataFrame({
                'bigram': data['unchosen_bigram'],
                'time': data['unchosen_bigram_time'],
                'time_norm': data['unchosen_bigram_time_norm']
            })
        ])
        
        # Add frequencies and calculate statistics
        typing_data['frequency'] = typing_data['bigram'].map(bigram_freqs)
        typing_data = typing_data.dropna(subset=['frequency'])
        
        # Generate plots
        self._plot_frequency_timing_relationship(
            typing_data,
            output_folder
        )
        
        return self._calculate_frequency_timing_statistics(typing_data)

    def _plot_frequency_timing_relationship(
        self,
        data: pd.DataFrame,
        output_folder: str
    ):
        """Create plots showing relationship between frequency and typing time."""
        # Helper function to add regression and correlation
        def add_regression_and_correlation(x, y):
            # Calculate correlation and p-value
            correlation, p_value = stats.spearmanr(x, y, nan_policy='omit')
            
            # Fit regression line
            slope, intercept = np.polyfit(np.log10(x), y, 1)
            x_line = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
            y_line = slope * np.log10(x_line) + intercept
            
            # Calculate R²
            y_pred = slope * np.log10(x) + intercept
            r2 = 1 - (np.sum((y - y_pred) ** 2) / 
                    np.sum((y - y.mean()) ** 2))
            
            # Plot regression line
            plt.plot(x_line, y_line, 'r--', label=f'R² = {r2:.3f}')
            
            # Add correlation and p-value
            plt.text(0.05, 0.90, 
                    f'Correlation: {correlation:.3f}\np-value: {p_value:.2e}',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.legend()
        
        # 1. Raw timing plot
        plt.figure(figsize=(10, 6))
        plt.semilogx()
        
        plt.scatter(
            data['frequency'],
            data['time'],
            alpha=0.5,
            color=self.config['visualization']['colors']['primary']
        )
        
        add_regression_and_correlation(data['frequency'], data['time'])
        
        plt.xlabel('Bigram Frequency (log scale)')
        plt.ylabel('Typing Time (ms)')
        plt.title('Raw Typing Times vs. Frequency')
        
        plt.savefig(os.path.join(output_folder, 'freq_vs_time_raw.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Normalized timing plot
        plt.figure(figsize=(10, 6))
        plt.semilogx()
        
        plt.scatter(
            data['frequency'],
            data['time_norm'],
            alpha=0.5,
            color=self.config['visualization']['colors']['primary']
        )
        
        add_regression_and_correlation(data['frequency'], data['time_norm'])
        
        plt.xlabel('Bigram Frequency (log scale)')
        plt.ylabel('Normalized Typing Time')
        plt.title('Normalized Typing Times vs. Frequency')
        
        plt.savefig(os.path.join(output_folder, 'freq_vs_time_normalized.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _calculate_frequency_timing_statistics(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate statistics for frequency-timing relationship."""
        # Calculate correlations
        raw_corr = stats.spearmanr(
            np.log10(data['frequency']),
            data['time'],
            nan_policy='omit'
        )
        
        norm_corr = stats.spearmanr(
            np.log10(data['frequency']),
            data['time_norm'],
            nan_policy='omit'
        )
        
        return {
            'raw_correlation': {
                'coefficient': raw_corr.correlation,
                'p_value': raw_corr.pvalue
            },
            'normalized_correlation': {
                'coefficient': norm_corr.correlation,
                'p_value': norm_corr.pvalue
            },
            'n_bigrams': len(data['bigram'].unique()),
            'n_observations': len(data)
        }

    def analyze_speed_choice_prediction(
        self,
        data: pd.DataFrame,
        output_folder: str
    ) -> Dict[str, Any]:
        """
        Analyze how well typing speed and frequency predict bigram choices.
        
        Generates:
        - speed_accuracy_by_magnitude.png
        - speed_accuracy_by_confidence.png 
        - user_accuracy_distribution.png
        - speed_choice_analysis_report.txt
        """
        # Calculate speed differences and prediction accuracy
        data = data.copy()
        data = self._apply_time_limit(data)  # Unpack both return values
        data['speed_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['speed_predicts_choice'] = data['speed_diff'] < 0
        data['confidence'] = data['sliderValue'].abs()
        
        # Calculate normalized speed differences per participant
        data['speed_diff_norm'] = self.stats.normalize_within_participant(data, 'speed_diff')
        
        # Add frequency differences if available
        bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
        missing_bigrams = set()
        for col in ['chosen_bigram', 'unchosen_bigram']:
            missing = set(data[col].unique()) - set(bigram_freqs.keys())
            if missing:
                missing_bigrams.update(missing)
        
        # Analyze by magnitude quintiles
        data['speed_diff_mag'] = data['speed_diff'].abs()
        data['magnitude_quintile'] = pd.qcut(
            data['speed_diff_mag'],
            5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            duplicates='drop'
        )
        
        # Per-participant analysis
        participant_results = []
        for user_id, user_data in data.groupby('user_id', observed=True):
            if len(user_data) >= self.config['analysis'].get('min_trials_per_participant', 5):
                speed_accuracy = user_data['speed_predicts_choice'].mean()
                ci_lower, ci_upper = self.stats.compute_confidence_intervals(
                    user_data['speed_predicts_choice'].values
                )
                
                # Calculate accuracy by magnitude for this participant
                mag_accuracies = user_data.groupby('magnitude_quintile', observed=True)[
                    'speed_predicts_choice'
                ].mean()
                
                participant_results.append({
                    'user_id': user_id,
                    'n_trials': len(user_data),
                    'speed_accuracy': speed_accuracy,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'magnitude_accuracies': mag_accuracies.to_dict()
                })
        
        results = {
            'participant_results': participant_results,
            'overall_accuracy': data['speed_predicts_choice'].mean(),
            'n_participants': len(participant_results),
            'n_trials': len(data)
        }
        
        # Generate plots
        self._plot_accuracy_by_magnitude(data, output_folder)
        self._plot_accuracy_by_confidence(data, output_folder)
        self._plot_user_accuracy_distribution(participant_results, output_folder)
        
        # Generate report
        self._generate_prediction_report(results, output_folder)
        
        return results

    def _plot_accuracy_by_magnitude(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'speed_accuracy_by_magnitude.png'
    ):
        """Plot prediction accuracy by speed difference magnitude."""
        plt.figure(figsize=(10, 6))
        
        # Calculate accuracy by magnitude quintile
        accuracy_by_mag = data.groupby('magnitude_quintile', observed=True)[
            'speed_predicts_choice'
        ].agg(['mean', 'std', 'count'])
        
        plt.errorbar(
            range(len(accuracy_by_mag)),
            accuracy_by_mag['mean'],
            yerr=accuracy_by_mag['std'],
            fmt='o-',
            capsize=5,
            color=self.config['visualization']['colors']['primary']
        )
        
        plt.xlabel('Speed Difference Magnitude Quintile')
        plt.ylabel('Prediction Accuracy')
        plt.title('Speed Prediction Accuracy by Magnitude of Speed Difference')
        plt.grid(True, alpha=0.3)
        
        # Add quintile sizes
        for i, (_, row) in enumerate(accuracy_by_mag.iterrows()):
            plt.text(i, row['mean'], f'n={int(row["count"])}', 
                    ha='center', va='bottom')
        
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_accuracy_by_confidence(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'speed_accuracy_by_confidence.png'
    ):
        """Plot prediction accuracy by confidence level."""
        plt.figure(figsize=(10, 6))
        
        # Create confidence bins, handling potential duplicate values
        try:
            # First try 4 bins
            data['confidence_level'] = pd.qcut(
                data['confidence'],
                4,
                labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                duplicates='drop'
            )
        except ValueError:
            try:
                # If that fails, try 3 bins
                data['confidence_level'] = pd.qcut(
                    data['confidence'],
                    3,
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )
            except ValueError:
                # If that still fails, use manual binning based on value ranges
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
        
        accuracy_by_conf = data.groupby('confidence_level', observed=True)[
            'speed_predicts_choice'
        ].agg(['mean', 'std', 'count'])
        
        x = range(len(accuracy_by_conf))
        plt.bar(x, accuracy_by_conf['mean'], 
                yerr=accuracy_by_conf['std'], 
                capsize=5,
                color=self.config['visualization']['colors']['primary'])
        
        plt.xlabel('Confidence Level')
        plt.ylabel('Prediction Accuracy')
        plt.title('Speed Prediction Accuracy by Confidence Level')
        plt.xticks(x, accuracy_by_conf.index, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add sample sizes
        for i, (_, row) in enumerate(accuracy_by_conf.iterrows()):
            plt.text(i, row['mean'], f'n={int(row["count"])}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_user_accuracy_distribution(
        self,
        participant_results: List[Dict[str, Any]],
        output_folder: str,
        filename: str = 'user_accuracy_distribution.png'
    ):
        """Plot distribution of per-user prediction accuracies."""
        plt.figure(figsize=(10, 6))
        
        accuracies = [p['speed_accuracy'] for p in participant_results]
        
        plt.hist(accuracies, bins=20, 
                color=self.config['visualization']['colors']['primary'])
        plt.axvline(
            np.mean(accuracies),
            color=self.config['visualization']['colors']['secondary'],
            linestyle='--',
            label=f'Mean: {np.mean(accuracies):.3f}'
        )
        
        plt.xlabel('Prediction Accuracy')
        plt.ylabel('Number of Participants')
        plt.title('Distribution of Per-Participant Prediction Accuracies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_prediction_report(
        self,
        results: Dict[str, Any],
        output_folder: str,
        filename: str = 'speed_choice_analysis_report.txt'
    ):
        """Generate comprehensive analysis report."""
        report_lines = [
            "Speed as Choice Proxy Analysis Report",
            "================================\n",
            
            f"Number of participants: {results['n_participants']}",
            f"Total trials analyzed: {results['n_trials']}",
            f"Overall prediction accuracy: {results['overall_accuracy']:.1%}\n",
            
            "Per-Participant Statistics:",
            "------------------------"
        ]
        
        accuracies = [p['speed_accuracy'] for p in results['participant_results']]
        report_lines.extend([
            f"Mean accuracy: {np.mean(accuracies):.1%}",
            f"Median accuracy: {np.median(accuracies):.1%}",
            f"Std deviation: {np.std(accuracies):.1%}",
            f"Range: [{min(accuracies):.1%}, {max(accuracies):.1%}]"
        ])
        
        with open(os.path.join(output_folder, filename), 'w') as f:
            f.write('\n'.join(report_lines))
    
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

def create_output_dirs(base_dir: str, subdirs: List[str]) -> None:
    """Create output directories if they don't exist."""
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
          
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
        output_dirs = [
            'bigram_choice_analysis',
            'bigram_typing_time_and_frequency_analysis'
        ]
        create_output_dirs(config['output']['base_dir'], output_dirs)
        
        # Initialize analyzer
        analyzer = BigramAnalysis(config)
        
        # Load data
        data_path = os.path.join(
            config['data']['input_dir'],
            config['data']['filtered_data_file']
        )
        data = analyzer.load_and_validate_data(data_path)
        
        # Run analyses and generate plots
        choice_folder = os.path.join(config['output']['base_dir'], 'bigram_choice_analysis')
        freq_folder = os.path.join(config['output']['base_dir'], 
                                 'bigram_typing_time_and_frequency_analysis')
        
        logger.info("Analyzing typing times and slider values...")
        choice_results = analyzer.analyze_typing_times_slider_values(data, choice_folder)
        
        logger.info("Analyzing frequency-typing relationship...")
        freq_results = analyzer.analyze_frequency_typing_relationship(data, freq_folder)
        
        # Log results summary
        logger.info("\nAnalysis Summary:")
        logger.info(f"Processed {len(data)} trials from {data['user_id'].nunique()} participants")
        logger.info(f"Generated plots in {config['output']['base_dir']}")
        
        logger.info("\nFrequency-Typing Relationship:")
        logger.info(f"Raw correlation: {freq_results['raw_correlation']['coefficient']:.3f} "
                   f"(p = {freq_results['raw_correlation']['p_value']:.3e})")
        logger.info(f"Normalized correlation: {freq_results['normalized_correlation']['coefficient']:.3f} "
                   f"(p = {freq_results['normalized_correlation']['p_value']:.3e})")

        # Speed choice analysis
        logger.info("Analyzing speed choice prediction...")
        speed_choice_folder = os.path.join(config['output']['base_dir'], 'speed_choice_analysis')
        os.makedirs(speed_choice_folder, exist_ok=True)
        prediction_results = analyzer.analyze_speed_choice_prediction(data, speed_choice_folder)

        logger.info(f"\nSpeed Choice Prediction Results:")
        logger.info(f"Overall accuracy: {prediction_results['overall_accuracy']:.1%}")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()