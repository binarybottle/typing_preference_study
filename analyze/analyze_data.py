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
        
        # Calculate speed differences
        data = data.copy()
        data['time_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
        data['time_diff_norm'] = self.stats.normalize_within_participant(data, 'time_diff')
        
        # Per-participant statistics
        participant_stats = []
        for user_id, user_data in data.groupby('user_id', observed=True):
            faster_chosen = (user_data['time_diff'] < 0)
            participant_stats.append({
                'user_id': user_id,
                'n_trials': len(user_data),
                'prop_faster_chosen': faster_chosen.mean(),
                'median_diff': user_data['time_diff'].median(),
                'mad_diff': median_abs_deviation(user_data['time_diff'], nan_policy='omit')
            })
        
        results['participant_stats'] = pd.DataFrame(participant_stats)
        
        # Generate original plots with enhanced statistics
        self._plot_chosen_vs_unchosen(data, output_folder)
        self._plot_time_diff_slider(data, output_folder)
        self._plot_time_diff_histograms(data, output_folder)
        
        return results

    def _plot_chosen_vs_unchosen(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'chosen_vs_unchosen_times.png'
    ):
        """Create boxplot comparing chosen vs unchosen typing times."""
        plt.figure(figsize=(10, 6))
        
        # Calculate per-participant statistics
        chosen_stats = []
        unchosen_stats = []
        
        for _, user_data in data.groupby('user_id', observed=True):
            chosen_stats.append({
                'median': user_data['chosen_bigram_time'].median(),
                'mad': median_abs_deviation(user_data['chosen_bigram_time'], nan_policy='omit')
            })
            unchosen_stats.append({
                'median': user_data['unchosen_bigram_time'].median(),
                'mad': median_abs_deviation(user_data['unchosen_bigram_time'], nan_policy='omit')
            })
        
        chosen_df = pd.DataFrame(chosen_stats)
        unchosen_df = pd.DataFrame(unchosen_stats)
        
        plt.boxplot(
            [chosen_df['median'], unchosen_df['median']],
            labels=['Chosen', 'Unchosen'],
            medianprops=dict(color="red"),
            showfliers=False
        )
        
        plt.ylabel('Typing Time (ms)')
        plt.title('Typing Times for Chosen vs Unchosen Bigrams')
        
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot to {filename}")

    def _plot_time_diff_slider(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'typing_time_diff_vs_slider_value.png'
    ):
        """Create scatter plot of time differences vs slider values."""
        plt.figure(figsize=(10, 6))
        
        plt.scatter(
            data['sliderValue'],
            data['time_diff_norm'],
            alpha=0.5,
            color=self.config['visualization']['colors']['primary']
        )
        
        plt.xlabel('Slider Value')
        plt.ylabel('Normalized Time Difference (Chosen - Unchosen)')
        plt.title('Typing Time Difference vs. Slider Value')
        
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot to {filename}")

    def _plot_time_diff_histograms(
        self,
        data: pd.DataFrame,
        output_folder: str,
        filename: str = 'typing_time_diff_vs_slider_value_histograms.png'
    ):
        """Create histograms of typing time differences by slider value range."""
        plt.figure(figsize=(15, 10))
        
        # Create slider value bins
        slider_ranges = [(-100, -60), (-60, -20), (-20, 20), (20, 60), (60, 100)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (low, high) in enumerate(slider_ranges):
            mask = (data['sliderValue'] >= low) & (data['sliderValue'] < high)
            subset = data[mask]
            
            if len(subset) > 0:
                # Plot normalized differences
                axes[i].hist(
                    subset['time_diff_norm'],
                    bins=30,
                    alpha=0.5,
                    label='Normalized Differences',
                    color='blue'
                )
                
                axes[i].set_title(f'Slider Values [{low}, {high})')
                axes[i].set_xlabel('Normalized Time Difference')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Remove extra subplot
        axes[-1].remove()
        
        plt.suptitle('Distribution of Time Differences by Slider Value Range', y=1.02)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot to {filename}")

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
        # 1. Raw timing plot
        plt.figure(figsize=(10, 6))
        plt.semilogx()
        
        plt.scatter(
            data['frequency'],
            data['time'],
            alpha=0.5,
            color=self.config['visualization']['colors']['primary']
        )
        
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
            data['speed_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
            data['speed_predicts_choice'] = data['speed_diff'] < 0
            data['confidence'] = data['sliderValue'].abs()
            
            # Calculate normalized speed differences per participant
            data['speed_diff_norm'] = self.stats.normalize_within_participant(data, 'speed_diff')
            
            # Add frequency differences if available
            bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
            try:
                data['freq_diff'] = data.apply(
                    lambda row: np.log10(bigram_freqs[row['chosen_bigram']]) - 
                            np.log10(bigram_freqs[row['unchosen_bigram']]),
                    axis=1
                )
            except:
                logger.warning("Could not calculate frequency differences")
                data['freq_diff'] = np.nan
            
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