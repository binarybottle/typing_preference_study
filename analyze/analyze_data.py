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
        self._plot_typing_times(data, output_folder)
        self._plot_chosen_vs_unchosen_scatter(data, output_folder)
        self._plot_time_diff_slider(data, output_folder)
        self._plot_time_diff_histograms(data, output_folder)
        
        # Generate overlaid histogram plots
        self._plot_overlaid_time_histograms(data, output_folder, 'raw_')
        self._plot_overlaid_time_histograms(data, output_folder, 'normalized_')
        
        self._plot_chosen_vs_unchosen_scatter(data, output_folder)

        return results
    
    def _plot_typing_times(
        self,
        bigram_data: pd.DataFrame,
        output_folder: str,
        filename: str = 'bigram_times_barplot.png'
    ):
        """
        Generate and save bar plot for median bigram typing times with horizontal x-axis labels.
        """
        # Prepare data by combining chosen and unchosen bigrams
        plot_data = pd.concat([
            bigram_data[['chosen_bigram', 'chosen_bigram_time']].rename(columns={'chosen_bigram': 'bigram', 'chosen_bigram_time': 'time'}),
            bigram_data[['unchosen_bigram', 'unchosen_bigram_time']].rename(columns={'unchosen_bigram': 'bigram', 'unchosen_bigram_time': 'time'})
        ])

        # Calculate median times and Median Absolute Deviation (MAD) for each bigram
        grouped_data = plot_data.groupby('bigram')['time']
        median_times = grouped_data.median().sort_values()
        mad_times = grouped_data.apply(lambda x: median_abs_deviation(x, nan_policy='omit')).reindex(median_times.index)

        # Create the plot
        plt.figure(figsize=(20, 10))
        plt.bar(range(len(median_times)), median_times.values, yerr=mad_times.values, capsize=5, alpha=0.8)

        # Add titles and labels
        plt.title('Typing Times for Each Bigram: Median (MAD)', fontsize=16)
        plt.xlabel('Bigram', fontsize=12)
        plt.ylabel('Time (ms)', fontsize=12)

        # Customize x-axis ticks
        plt.xticks(range(len(median_times)), median_times.index, rotation=90, ha='center', fontsize=8)
        plt.xlim(-0.5, len(median_times) - 0.5)

        # Apply tight layout to adjust spacing
        plt.tight_layout()

        # Save the plot
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #print(f"Median bigram typing times bar plot saved to: {output_path}")

        # Close the plot
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
        Generates distribution, median, and minimum time plots with statistics.
        """
        # Create frequency dictionary
        bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
        
        # Calculate aggregate statistics per bigram
        bigram_stats = []
        for bigram in set(data['chosen_bigram'].unique()) | set(data['unchosen_bigram'].unique()):
            # Combine chosen and unchosen times for this bigram
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
        
        # Calculate correlations and regressions
        freq_log = np.log10(df_stats['frequency'])
        
        # For median times
        median_corr, median_p = stats.spearmanr(freq_log, df_stats['median_time'])
        median_slope, median_intercept = np.polyfit(freq_log, df_stats['median_time'], 1)
        median_pred = median_slope * freq_log + median_intercept
        median_r2 = 1 - (np.sum((df_stats['median_time'] - median_pred) ** 2) / 
                        np.sum((df_stats['median_time'] - df_stats['median_time'].mean()) ** 2))
        
        # For minimum times
        min_corr, min_p = stats.spearmanr(freq_log, df_stats['min_time'])
        min_slope, min_intercept = np.polyfit(freq_log, df_stats['min_time'], 1)
        min_pred = min_slope * freq_log + min_intercept
        min_r2 = 1 - (np.sum((df_stats['min_time'] - min_pred) ** 2) / 
                    np.sum((df_stats['min_time'] - df_stats['min_time'].mean()) ** 2))
        
        # Generate plots
        self._plot_distribution(df_stats, output_folder)
        self._plot_median_times(df_stats, median_corr, median_p, median_r2, output_folder)
        self._plot_min_times(df_stats, min_corr, min_p, min_r2, output_folder)
        
        # Calculate additional statistics for text output
        n_groups = 5
        df_stats['freq_group'] = pd.qcut(df_stats['frequency'], n_groups, labels=False)
        group_stats = self._calculate_group_stats(df_stats, n_groups)
        
        # Generate text reports
        self._save_frequency_group_timing_analysis(
            df_stats, median_corr, median_p, median_r2, group_stats, output_folder)
        
        logger.error(f"FIX _save_bigram_time_difference_analysis")
        #self._save_bigram_time_difference_analysis(
        #    data, df_stats, output_folder)
        
        return {
            'median_correlation': median_corr,
            'median_p_value': median_p,
            'median_r2': median_r2,
            'min_correlation': min_corr,
            'min_p_value': min_p,
            'min_r2': min_r2,
            'n_bigrams': len(df_stats),
            'total_instances': df_stats['n_samples'].sum()
        }

    def _plot_distribution(
        self,
        df_stats: pd.DataFrame,
        output_folder: str,
        filename: str = 'frequency_and_timing_distribution.png'
    ):
        """Create distribution plot with error bars and sample size indicators."""
        plt.figure(figsize=(12, 8))
        
        # Calculate correlation and regression
        freq_log = np.log10(df_stats['frequency'])
        correlation, p_value = stats.spearmanr(freq_log, df_stats['median_time'])
        slope, intercept = np.polyfit(freq_log, df_stats['median_time'], 1)
        
        # Create scatter plot with sized points
        sizes = np.clip(df_stats['n_samples'] / 10, 10, 100)  # Scale sizes
        plt.scatter(df_stats['frequency'], df_stats['median_time'], 
                    s=sizes, alpha=0.6, color='#1f77b4')
        
        # Add error bars
        plt.errorbar(df_stats['frequency'], df_stats['median_time'],
                    yerr=df_stats['std_time'], alpha=0.2,
                    fmt='none', color='lightblue')
        
        # Add regression line
        x_line = np.logspace(np.log10(df_stats['frequency'].min()),
                            np.log10(df_stats['frequency'].max()), 100)
        y_line = slope * np.log10(x_line) + intercept
        plt.plot(x_line, y_line, 'r-', alpha=0.7)
        
        # Add correlation info
        plt.text(0.05, 0.90, 
                f'Correlation: {correlation:.3f}\np-value: {p_value:.3e}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Add legend for sample sizes
        legend_elements = [
            plt.scatter([], [], s=n/10, label=f'n={n} samples', alpha=0.6)
            for n in [10, 50, 100, 500, 1000]
        ]
        plt.legend(handles=legend_elements, title='Number of timing samples',
                bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xscale('log')
        plt.xlabel('Bigram Frequency')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Distribution of Typing Times vs. Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_median_times(
        self,
        df_stats: pd.DataFrame,
        correlation: float,
        p_value: float,
        r2: float,
        output_folder: str,
        filename: str = 'frequency_and_timing_median.png'
    ):
        """Create median times plot with bigram labels."""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        plt.scatter(df_stats['frequency'], df_stats['median_time'],
                    alpha=0.6, color='#1f77b4')
        
        # Add bigram labels
        for _, row in df_stats.iterrows():
            plt.annotate(row['bigram'],
                        (row['frequency'], row['median_time']),
                        xytext=(2, 2), textcoords='offset points',
                        fontsize=8)
        
        # Add regression line
        freq_log = np.log10(df_stats['frequency'])
        slope, intercept = np.polyfit(freq_log, df_stats['median_time'], 1)
        x_line = np.logspace(np.log10(df_stats['frequency'].min()),
                            np.log10(df_stats['frequency'].max()), 100)
        y_line = slope * np.log10(x_line) + intercept
        plt.plot(x_line, y_line, 'r-', alpha=0.7, label=f'R² = {r2:.3f}')
        
        # Add correlation info
        plt.text(0.05, 0.95,
                f'Correlation: {correlation:.3f}\np-value: {p_value:.3e}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.xscale('log')
        plt.xlabel('Bigram Frequency (log scale)')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Median Times vs. Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_min_times(
        self,
        df_stats: pd.DataFrame,
        correlation: float,
        p_value: float,
        r2: float,
        output_folder: str,
        filename: str = 'frequency_and_timing_minimum.png'
    ):
        """Create minimum times plot with bigram labels."""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        plt.scatter(df_stats['frequency'], df_stats['min_time'],
                    alpha=0.6, color='#1f77b4')
        
        # Add bigram labels
        for _, row in df_stats.iterrows():
            plt.annotate(row['bigram'],
                        (row['frequency'], row['min_time']),
                        xytext=(2, 2), textcoords='offset points',
                        fontsize=8)
        
        # Add regression line
        freq_log = np.log10(df_stats['frequency'])
        slope, intercept = np.polyfit(freq_log, df_stats['min_time'], 1)
        x_line = np.logspace(np.log10(df_stats['frequency'].min()),
                            np.log10(df_stats['frequency'].max()), 100)
        y_line = slope * np.log10(x_line) + intercept
        plt.plot(x_line, y_line, 'r-', alpha=0.7, label=f'R² = {r2:.3f}')
        
        # Add correlation info
        plt.text(0.05, 0.95,
                f'Correlation: {correlation:.3f}\np-value: {p_value:.3e}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.xscale('log')
        plt.xlabel('Bigram Frequency (log scale)')
        plt.ylabel('Minimum Typing Time (ms)')
        plt.title('Fastest Times vs. Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
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

    def _calculate_group_stats(
        self,
        df_stats: pd.DataFrame,
        n_groups: int
    ) -> pd.DataFrame:
        """Calculate statistics for frequency groups."""
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

    def _save_bigram_time_difference_analysis(
        self,
        data: pd.DataFrame,
        df_stats: pd.DataFrame,
        output_folder: str,
        filename: str = 'time_frequency_analysis.txt'
    ):
        """Generate time-frequency difference analysis report."""
        # Calculate raw time differences
        data = data.copy()
        data['time_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['time_diff_norm'] = self.stats.normalize_within_participant(data, 'time_diff')
        
        # Calculate frequency differences
        bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
        data['freq_diff'] = data.apply(
            lambda row: np.log10(bigram_freqs[row['chosen_bigram']]) - 
                    np.log10(bigram_freqs[row['unchosen_bigram']])
            if row['chosen_bigram'] in bigram_freqs and row['unchosen_bigram'] in bigram_freqs
            else np.nan,
            axis=1
        )
        
        # Calculate correlations and regressions
        raw_corr, raw_p = stats.spearmanr(data['freq_diff'].dropna(), 
                                        data['time_diff'].dropna())
        norm_corr, norm_p = stats.spearmanr(data['freq_diff'].dropna(), 
                                        data['time_diff_norm'].dropna())
        
        # Calculate R² values
        raw_slope, raw_intercept = np.polyfit(data['freq_diff'].dropna(), 
                                            data['time_diff'].dropna(), 1)
        raw_pred = raw_slope * data['freq_diff'].dropna() + raw_intercept
        raw_r2 = 1 - (np.sum((data['time_diff'].dropna() - raw_pred) ** 2) / 
                    np.sum((data['time_diff'].dropna() - data['time_diff'].dropna().mean()) ** 2))
        
        norm_slope, norm_intercept = np.polyfit(data['freq_diff'].dropna(), 
                                            data['time_diff_norm'].dropna(), 1)
        norm_pred = norm_slope * data['freq_diff'].dropna() + norm_intercept
        norm_r2 = 1 - (np.sum((data['time_diff_norm'].dropna() - norm_pred) ** 2) / 
                    np.sum((data['time_diff_norm'].dropna() - data['time_diff_norm'].dropna().mean()) ** 2))
        
        # Calculate per-user correlations
        user_raw_corrs = []
        user_norm_corrs = []
        for user_id, user_data in data.groupby('user_id', observed=True):
            if len(user_data) >= 5:  # Only include users with enough data points
                if not user_data['freq_diff'].isna().all():
                    raw_corr_user = stats.spearmanr(user_data['freq_diff'].dropna(), 
                                                user_data['time_diff'].dropna())[0]
                    norm_corr_user = stats.spearmanr(user_data['freq_diff'].dropna(), 
                                                user_data['time_diff_norm'].dropna())[0]
                    user_raw_corrs.append(raw_corr_user)
                    user_norm_corrs.append(norm_corr_user)
        
        # Generate report
        report_lines = [
            "Time-Frequency Difference Analysis Results",
            "=======================================\n",
            
            "Raw Analysis:",
            f"Correlation: {raw_corr:.3f}",
            f"P-value: {raw_p:.3e}",
            f"R-squared: {raw_r2:.3f}",
            f"Regression coefficient: {raw_slope:.3f}\n",
            
            "Normalized Analysis:",
            f"Correlation: {norm_corr:.3f}",
            f"P-value: {norm_p:.3e}",
            f"R-squared: {norm_r2:.3f}",
            f"Regression coefficient: {norm_slope:.3f}\n",
            
            "User-Level Analysis:",
            "Raw correlations:",
            f"  Mean: {np.mean(user_raw_corrs):.3f}",
            f"  Std: {np.std(user_raw_corrs):.3f}",
            "Normalized correlations:",
            f"  Mean: {np.mean(user_norm_corrs):.3f}",
            f"  Std: {np.std(user_norm_corrs):.3f}",
            f"Number of users: {len(user_raw_corrs)}"
        ]
        
        with open(os.path.join(output_folder, filename), 'w') as f:
            f.write('\n'.join(report_lines))

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
        - preference_prediction_report.txt
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
        filename: str = 'preference_prediction_report.txt'
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
            'typing_time_vs_frequency',
            'typing_time_vs_preference'
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
        choice_folder = os.path.join(config['output']['base_dir'], 'typing_time_vs_frequency')
        freq_folder = os.path.join(config['output']['base_dir'], 
                                 'typing_time_vs_preference')
        
        logger.info("Analyzing typing times and slider values...")
        choice_results = analyzer.analyze_typing_times_slider_values(data, choice_folder)
        
        logger.info("Analyzing frequency-typing relationship...")
        freq_results = analyzer.analyze_frequency_typing_relationship(data, freq_folder)
        
        # Log results summary
        logger.info("\nAnalysis Summary:")
        logger.info(f"Processed {len(data)} trials from {data['user_id'].nunique()} participants")
        logger.info(f"Generated plots in {config['output']['base_dir']}")
        
        logger.info("\nFrequency-Typing Relationship:")




        #logger.info(f"Raw correlation: {freq_results['raw_correlation']['coefficient']:.3f} "
        #           f"(p = {freq_results['raw_correlation']['p_value']:.3e})")
        #logger.info(f"Normalized correlation: {freq_results['normalized_correlation']['coefficient']:.3f} "
        #           f"(p = {freq_results['normalized_correlation']['p_value']:.3e})")




        # Speed choice analysis
        logger.info("Analyzing speed choice prediction...")
        speed_choice_folder = os.path.join(config['output']['base_dir'], 'preference_prediction')
        os.makedirs(speed_choice_folder, exist_ok=True)
        prediction_results = analyzer.analyze_speed_choice_prediction(data, speed_choice_folder)

        logger.info(f"\nSpeed Choice Prediction Results:")
        logger.info(f"Overall accuracy: {prediction_results['overall_accuracy']:.1%}")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()