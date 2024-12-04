"""
Bigram Typing Analysis Pipeline

Core analysis of typing speed, frequency, and choice relationships in bigram typing experiments.
Focuses on per-participant analysis with robust statistical approaches.

Key features:
- Per-participant normalization and analysis
- Robust statistics (median, MAD, bootstrapped CIs)
- Parallel raw and normalized analyses
- Comprehensive uncertainty quantification

Analysis components:
1. Speed-choice relationships within participants
2. Frequency effects after controlling for speed
3. Combined effects and interaction analysis
4. Distribution analysis of differences
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import median_abs_deviation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging

from bigram_frequencies import bigrams, bigram_frequencies_array

epsilon = 1e-10  # Small value to avoid divide-by-zero

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load analysis configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration parameters
    
    Raises
    ------
    FileNotFoundError
        If config file not found
    yaml.YAMLError
        If config file is invalid
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required sections
        required_sections = ['data', 'analysis', 'visualization', 'output']
        missing = [s for s in required_sections if s not in config]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
            
        return config
        
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        raise

def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate bigram typing data.
    
    Parameters
    ----------
    data_path : str
        Path to filtered data CSV file
        
    Returns
    -------
    pd.DataFrame
        Validated data with required columns
    """
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
            
        # Convert numeric columns, dropping any non-numeric values
        numeric_cols = ['chosen_bigram_time', 'unchosen_bigram_time', 'sliderValue']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            n_dropped = df[col].isna().sum()
            if n_dropped > 0:
                logger.warning(f"Dropped {n_dropped} non-numeric values from {col}")
        
        # Remove rows with any NaN values in required numeric columns
        n_before = len(df)
        df = df.dropna(subset=numeric_cols)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning(f"Dropped {n_dropped} rows with missing values")
            
        # Validate value ranges
        if not (-100 <= df['sliderValue'].max() <= 100 and -100 <= df['sliderValue'].min() <= 100):
            raise ValueError("sliderValue outside valid range [-100, 100]")
            
        if (df['chosen_bigram_time'] < 0).any() or (df['unchosen_bigram_time'] < 0).any():
            raise ValueError("Negative typing times found")
            
        logger.info(f"Loaded {len(df)} rows of data from {len(df['user_id'].unique())} participants")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {str(e)}")
        raise

class RobustStatistics:
    """
    Collection of robust statistical methods for typing analysis.
    """
    
    @staticmethod
    def normalize_within_participant(
        data: pd.DataFrame,
        value_column: str,
        grouping_column: str = 'user_id',
        method: str = "median"
    ) -> pd.Series:
        """
        Normalize values within each participant.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data containing values and grouping column
        value_column : str
            Column to normalize
        grouping_column : str
            Column defining groups (default: 'user_id')
        method : str
            Normalization method ('median' or 'mean')
            
        Returns
        -------
        pd.Series
            Normalized values
        """
        if method == "median":
            return data.groupby(grouping_column)[value_column].transform(
                lambda x: (x - x.median()) / median_abs_deviation(x, nan_policy='omit')
            )
        else:  # mean
            return data.groupby(grouping_column)[value_column].transform(
                lambda x: (x - x.mean()) / x.std()
            )
    
    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: callable,
        n_iterations: int = 1000,
        ci_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bootstrapped confidence intervals.
        
        Parameters
        ----------
        data : np.ndarray
            Data to bootstrap
        statistic : callable
            Statistical function to apply
        n_iterations : int
            Number of bootstrap iterations
        ci_level : float
            Confidence interval level (0-1)
            
        Returns
        -------
        Tuple[float, float]
            Lower and upper confidence bounds
        """
        if len(data) == 0:
            return np.nan, np.nan
            
        try:
            bootstrap_stats = []
            for _ in range(n_iterations):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_stats.append(statistic(sample))
            
            ci_lower = np.percentile(bootstrap_stats, (1 - ci_level) / 2 * 100)
            ci_upper = np.percentile(bootstrap_stats, (1 + ci_level) / 2 * 100)
            
            return ci_lower, ci_upper
            
        except Exception as e:
            logger.warning(f"Bootstrap failed: {str(e)}")
            return np.nan, np.nan
    
    @staticmethod
    def robust_regression(
        X: np.ndarray,
        y: np.ndarray,
        method: str = "huber"
    ) -> Tuple[float, float, float]:
        """
        Perform robust regression resistant to outliers.
        
        Parameters
        ----------
        X : np.ndarray
            Predictor variables
        y : np.ndarray
            Target variable
        method : str
            Robust regression method
            
        Returns
        -------
        Tuple[float, float, float]
            Slope, intercept, and robust R²
        """
        from sklearn.linear_model import HuberRegressor, RANSACRegressor
        
        if method == "huber":
            model = HuberRegressor()
        else:  # RANSAC
            model = RANSACRegressor()
            
        # Ensure 2D array for X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        model.fit(X, y)
        
        # Calculate robust R² using median absolute deviation
        y_pred = model.predict(X)
        residuals = y - y_pred
        mad_residuals = median_abs_deviation(residuals, nan_policy='omit')
        mad_y = median_abs_deviation(y, nan_policy='omit')
        epsilon = 1e-10  # Small value to prevent division by zero
        r2 = 1 - (mad_residuals / (mad_y + epsilon))**2
        
        if hasattr(model, 'coef_'):
            slope = model.coef_[0]
        else:
            slope = model.estimator_.coef_[0]
            
        if hasattr(model, 'intercept_'):
            intercept = model.intercept_
        else:
            intercept = model.estimator_.intercept_
            
        return slope, intercept, r2

def analyze_typing_speed_differences(
    data: pd.DataFrame,
    config: dict
) -> Dict[str, Any]:
    """
    Analyze typing speed differences and their distribution within participants.
    
    Parameters
    ----------
    data : pd.DataFrame
        Trial data with typing times
    config : dict
        Analysis configuration
        
    Returns
    -------
    Dict[str, Any]
        Analysis results including:
        - Per-participant distribution statistics
        - Overall distribution characteristics
        - Bootstrapped confidence intervals
    """
    results = {}
    stats = RobustStatistics()
    
    # Calculate speed differences
    data = data.copy()
    data['speed_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
    data['speed_diff_norm'] = stats.normalize_within_participant(
        data, 'speed_diff', method=config['analysis']['normalize_method']
    )
    
    # Analyze per participant
    participant_stats = []
    
    for user_id, user_data in data.groupby('user_id'):
        if len(user_data) < config['analysis']['min_trials_per_participant']:
            continue
            
        speed_diffs = user_data['speed_diff'].values
        
        # Calculate robust statistics
        participant_stats.append({
            'user_id': user_id,
            'n_trials': len(speed_diffs),
            'median_diff': np.median(speed_diffs),
            'mad_diff': median_abs_deviation(speed_diffs, nan_policy='omit'),
            'lower_ci': stats.bootstrap_ci(speed_diffs, np.median)[0],
            'upper_ci': stats.bootstrap_ci(speed_diffs, np.median)[1],
            'prop_faster': (speed_diffs < 0).mean()
        })
    
    results['participant_stats'] = pd.DataFrame(participant_stats)
    
    # Aggregate statistics
    results['overall'] = {
        'n_participants': len(results['participant_stats']),
        'median_of_medians': results['participant_stats']['median_diff'].median(),
        'mad_of_medians': median_abs_deviation(
            results['participant_stats']['median_diff'], 
            nan_policy='omit'
        ),
        'participant_ci': stats.bootstrap_ci(
            results['participant_stats']['median_diff'],
            np.median
        )
    }
    
    return results

def analyze_participant_choice_patterns(
    data: pd.DataFrame,
    config: dict
) -> Dict[str, Any]:
    """
    Analyze how typing speed predicts choices within each participant.
    
    Parameters
    ----------
    data : pd.DataFrame
        Trial data with choices and typing times
    config : dict
        Analysis configuration
        
    Returns
    -------
    Dict[str, Any]
        Analysis results including:
        - Per-participant prediction accuracy
        - Speed threshold effects
        - Choice consistency measures
    """
    results = {}
    stats = RobustStatistics()
    
    # Prepare analysis data
    data = data.copy()
    data['speed_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']
    data['speed_predicts_choice'] = data['speed_diff'] < 0
    
    # Analyze per participant
    participant_results = []
    
    for user_id, user_data in data.groupby('user_id'):
        if len(user_data) < config['analysis']['min_trials_per_participant']:
            continue
            
        # Split data into quintiles by speed difference magnitude
        user_data['diff_magnitude'] = user_data['speed_diff'].abs()
        user_data['magnitude_quintile'] = pd.qcut(
            user_data['diff_magnitude'],
            config['analysis']['n_quantiles'],
            labels=False
        )
        
        # Calculate accuracy by magnitude
        quintile_accuracies = user_data.groupby('magnitude_quintile')[
            'speed_predicts_choice'
        ].agg(['mean', 'count'])
        
        participant_results.append({
            'user_id': user_id,
            'n_trials': len(user_data),
            'overall_accuracy': user_data['speed_predicts_choice'].mean(),
            'accuracy_ci': stats.bootstrap_ci(
                user_data['speed_predicts_choice'].values,
                np.mean
            ),
            'quintile_accuracies': quintile_accuracies['mean'].values,
            'quintile_counts': quintile_accuracies['count'].values,
            'median_speed_diff': user_data['speed_diff'].median()
        })
    
    results['participant_results'] = pd.DataFrame(participant_results)
    
    # Calculate aggregate statistics
    accuracies = results['participant_results']['overall_accuracy']
    results['aggregate'] = {
        'median_accuracy': accuracies.median(),
        'mad_accuracy': median_abs_deviation(accuracies, nan_policy='omit'),
        'accuracy_ci': stats.bootstrap_ci(accuracies, np.median),
        'n_participants': len(accuracies)
    }
    
    # Analyze accuracy by magnitude across participants
    quintile_stats = []
    for quintile in range(config['analysis']['n_quantiles']):
        # Get quintile accuracies for all participants
        quintile_accs = []
        for _, row in results['participant_results'].iterrows():
            try:
                quintile_accs.append(row['quintile_accuracies'][quintile])
            except (IndexError, KeyError):
                continue
                
        if quintile_accs:
            quintile_stats.append({
                'quintile': quintile,
                'median_accuracy': np.median(quintile_accs),
                'mad_accuracy': median_abs_deviation(quintile_accs, nan_policy='omit'),
                'ci': stats.bootstrap_ci(quintile_accs, np.median)
            })
    
    results['magnitude_effects'] = pd.DataFrame(quintile_stats)
    
    return results

def analyze_frequency_effects(
    data: pd.DataFrame,
    bigram_frequencies: Dict[str, float],
    config: dict
) -> Dict[str, Any]:
    """
    Analyze frequency effects on choices after controlling for speed.
    
    For each participant:
    1. Regress out typing speed effects
    2. Analyze frequency-choice relationship in residuals
    3. Assess combined speed-frequency models
    
    Parameters
    ----------
    data : pd.DataFrame
        Trial data
    bigram_frequencies : Dict[str, float]
        Bigram frequency dictionary
    config : dict
        Analysis configuration
        
    Returns
    -------
    Dict[str, Any]
        Analysis results including:
        - Per-participant frequency effects
        - Residual analyses
        - Model comparisons
    """
    results = {}
    stats = RobustStatistics()
    
    # Calculate frequency differences
    data = data.copy()
    data['speed_diff'] = data['chosen_bigram_time'] - data['unchosen_bigram_time']

    # Calculate frequency transform
    if config['analysis']['frequency_transform'] == 'log10':
        transform = np.log10
    else:
        transform = lambda x: x
        
    # Check for missing bigrams
    missing_bigrams = set()
    for col in ['chosen_bigram', 'unchosen_bigram']:
        missing = set(data[col].unique()) - set(bigram_frequencies.keys())
        if missing:
            missing_bigrams.update(missing)
            logger.warning(f"Found missing bigrams in {col}: {missing}")
    
    # Filter out rows with missing bigrams
    valid_bigram_mask = (
        data['chosen_bigram'].isin(bigram_frequencies.keys()) & 
        data['unchosen_bigram'].isin(bigram_frequencies.keys())
    )
    n_dropped = (~valid_bigram_mask).sum()
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} rows with missing bigram frequencies")
    data = data[valid_bigram_mask]
    
    # Calculate frequency differences
    data['freq_diff'] = data.apply(
        lambda row: (
            transform(bigram_frequencies[row['chosen_bigram']]) -
            transform(bigram_frequencies[row['unchosen_bigram']])
        ),
        axis=1
    )
        
    # Analyze per participant
    participant_results = []
    
    for user_id, user_data in data.groupby('user_id'):
        if len(user_data) < config['analysis']['min_trials_per_participant']:
            continue
            
        # First regress out speed effects
        X_speed = user_data['speed_diff'].values.reshape(-1, 1)
        y = (user_data['speed_diff'] < 0).astype(int)
        
        slope, intercept, r2_speed = stats.robust_regression(X_speed, y)
        speed_residuals = y - (slope * X_speed.ravel() + intercept)
        
        # Analyze frequency effects on residuals
        X_freq = user_data['freq_diff'].values.reshape(-1, 1)
        freq_slope, freq_intercept, r2_freq = stats.robust_regression(X_freq, speed_residuals)
        
        # Fit combined model
        X_combined = np.column_stack([X_speed, X_freq])
        y_combined = (user_data['speed_diff'] < 0).astype(int)
        
        combined_model = LinearRegression()
        combined_model.fit(X_combined, y_combined)
        
        participant_results.append({
            'user_id': user_id,
            'n_trials': len(user_data),
            'speed_r2': r2_speed,
            'freq_r2': r2_freq,
            'combined_r2': combined_model.score(X_combined, y_combined),
            'speed_coef': float(combined_model.coef_[0]),
            'freq_coef': float(combined_model.coef_[1]),
            'relative_freq_effect': float(combined_model.coef_[1]) / float(combined_model.coef_[0])
        })
    
    results['participant_results'] = pd.DataFrame(participant_results)
    
    # Calculate aggregate statistics
    for measure in ['speed_r2', 'freq_r2', 'combined_r2', 'relative_freq_effect']:
        values = results['participant_results'][measure]
        results[f'aggregate_{measure}'] = {
            'median': values.median(),
            'mad': median_abs_deviation(values, nan_policy='omit'),
            'ci': stats.bootstrap_ci(values, np.median)
        }
    
    return results

class VisualizationHelpers:
    """Helper functions for creating consistent visualizations."""
    
    @staticmethod
    def setup_plot_style(config: dict):
        """Apply consistent styling to matplotlib plots."""
        plt.style.use(config['visualization']['style'])
        plt.rcParams['figure.figsize'] = config['visualization']['figsize']
        plt.rcParams['figure.dpi'] = config['visualization']['dpi']
    
    @staticmethod
    def add_confidence_band(x: np.ndarray, y: np.ndarray, ci: np.ndarray, 
                          color: str, alpha: float = 0.2):
        """Add confidence band to existing plot."""
        plt.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=alpha)

def visualize_speed_differences(
    results: Dict[str, Any],
    output_dir: str,
    config: dict
):
    """
    Create visualizations of typing speed difference analyses.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from speed difference analysis
    output_dir : str
        Output directory for plots
    config : dict
        Visualization configuration
    """
    viz = VisualizationHelpers()
    viz.setup_plot_style(config)
    
    # 1. Distribution of median speed differences across participants
    plt.figure()
    sns.histplot(
        results['participant_stats']['median_diff'],
        stat='density',
        color=config['visualization']['colors']['primary']
    )
    plt.axvline(
        results['overall']['median_of_medians'],
        color=config['visualization']['colors']['secondary'],
        linestyle='--',
        label=f"Overall Median: {results['overall']['median_of_medians']:.1f}ms"
    )
    plt.xlabel('Median Speed Difference (ms)')
    plt.ylabel('Density')
    plt.title('Distribution of Per-Participant Median Speed Differences')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'speed_diff_distribution.png'))
    plt.close()
    
    # Add more visualizations as needed...

def visualize_choice_patterns(
    results: Dict[str, Any],
    output_dir: str,
    config: dict
):
    """
    Create visualizations of choice pattern analyses.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from choice pattern analysis
    output_dir : str
        Output directory for plots
    config : dict
        Visualization configuration
    """
    viz = VisualizationHelpers()
    viz.setup_plot_style(config)
    
    # 1. Accuracy by speed difference magnitude
    plt.figure()
    magnitude_data = results['magnitude_effects']
    plt.errorbar(
        magnitude_data['quintile'],
        magnitude_data['median_accuracy'],
        yerr=[
            magnitude_data['median_accuracy'] - magnitude_data['ci'].apply(lambda x: x[0]),
            magnitude_data['ci'].apply(lambda x: x[1]) - magnitude_data['median_accuracy']
        ],
        fmt='o-',
        color=config['visualization']['colors']['primary'],
        capsize=5
    )
    plt.xlabel('Speed Difference Magnitude Quintile')
    plt.ylabel('Choice Prediction Accuracy')
    plt.title('Speed Difference Magnitude vs. Choice Accuracy\n(Error bars: 95% CI)')
    plt.savefig(os.path.join(output_dir, 'accuracy_by_magnitude.png'))
    plt.close()
    
    # Add more visualizations as needed...

def visualize_frequency_effects(
    results: Dict[str, Any],
    output_dir: str,
    config: dict
):
    """
    Create visualizations of frequency effect analyses.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from frequency effect analysis
    output_dir : str
        Output directory for plots
    config : dict
        Visualization configuration
    """
    viz = VisualizationHelpers()
    viz.setup_plot_style(config)
    
    # 1. Model comparison plot
    plt.figure()
    model_results = results['participant_results']
    
    plt.boxplot(
        [model_results['speed_r2'], 
         model_results['freq_r2'], 
         model_results['combined_r2']],
        tick_labels=['Speed Only', 'Frequency Only', 'Combined'],
        showfliers=False
    )
    plt.ylabel('R² Value')
    plt.title('Model Comparison Across Participants')
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    # Add more visualizations as needed...

def generate_analysis_report(
    speed_results: Dict[str, Any],
    choice_results: Dict[str, Any],
    frequency_results: Dict[str, Any],
    output_path: str
):
    """
    Generate comprehensive analysis report.
    
    Parameters
    ----------
    speed_results : Dict[str, Any]
        Results from speed difference analysis
    choice_results : Dict[str, Any]
        Results from choice pattern analysis
    frequency_results : Dict[str, Any]
        Results from frequency effect analysis
    output_path : str
        Path for saving report
    """
    report_lines = [
        "Bigram Typing Analysis Report",
        "===========================\n",
        
        "1. Speed Difference Analysis",
        "--------------------------",
        f"Number of participants: {speed_results['overall']['n_participants']}",
        f"Median speed difference: {speed_results['overall']['median_of_medians']:.1f}ms",
        f"MAD of speed differences: {speed_results['overall']['mad_of_medians']:.1f}ms",
        f"95% CI: [{speed_results['overall']['participant_ci'][0]:.1f}, "
        f"{speed_results['overall']['participant_ci'][1]:.1f}]ms\n",
        
        "2. Choice Pattern Analysis",
        "------------------------",
        f"Median prediction accuracy: {choice_results['aggregate']['median_accuracy']:.1%}",
        f"MAD of accuracies: {choice_results['aggregate']['mad_accuracy']:.1%}",
        f"95% CI: [{choice_results['aggregate']['accuracy_ci'][0]:.1%}, "
        f"{choice_results['aggregate']['accuracy_ci'][1]:.1%}]\n",
        
        "3. Frequency Effect Analysis",
        "--------------------------",
        f"Speed-only R² (median): {frequency_results['aggregate_speed_r2']['median']:.3f}",
        f"Combined model R² (median): {frequency_results['aggregate_combined_r2']['median']:.3f}",
        f"Relative frequency effect: {frequency_results['aggregate_relative_freq_effect']['median']:.3f}",
        
        "\nDetailed Statistics",
        "------------------"
    ]
    
    # Add detailed statistics as needed...
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

def setup_output_directories(config: dict) -> Dict[str, str]:
    """
    Create output directory structure.
    
    Parameters
    ----------
    config : dict
        Configuration containing output paths
        
    Returns
    -------
    Dict[str, str]
        Mapping of analysis types to output directories
    """
    output_dirs = {}
    base_dir = Path(config['output']['base_dir'])
    
    for analysis_type, subdir in config['output']['subdirs'].items():
        full_path = base_dir / subdir
        full_path.mkdir(parents=True, exist_ok=True)
        output_dirs[analysis_type] = str(full_path)
        
    return output_dirs

def run_analysis_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Run complete analysis pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    Dict[str, Any]
        Complete analysis results
    """
    try:
        # Load configuration and setup
        config = load_config(config_path)
        output_dirs = setup_output_directories(config)
        logger.info("Configuration loaded and output directories created")
        
        # Load and validate data using path from config
        data_path = os.path.join(
            config['data']['input_dir'],
            config['data']['filtered_data_file']
        )
        data = load_and_validate_data(data_path)
        logger.info(f"Loaded {len(data)} trials from {data['user_id'].nunique()} participants")
        
        # Create bigram frequency dictionary
        bigram_freqs = dict(zip(bigrams, bigram_frequencies_array))
        
        # Run analyses
        logger.info("Starting speed difference analysis...")
        speed_results = analyze_typing_speed_differences(data, config)
        
        logger.info("Starting choice pattern analysis...")
        choice_results = analyze_participant_choice_patterns(data, config)
        
        logger.info("Starting frequency effect analysis...")
        frequency_results = analyze_frequency_effects(data, bigram_freqs, config)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualize_speed_differences(
            speed_results,
            output_dirs['time_analysis'],
            config
        )
        
        visualize_choice_patterns(
            choice_results,
            output_dirs['participant_analysis'],
            config
        )
        
        visualize_frequency_effects(
            frequency_results,
            output_dirs['frequency_analysis'],
            config
        )
        
        # Generate report
        logger.info("Generating analysis report...")
        report_path = Path(output_dirs['participant_analysis']) / 'analysis_report.txt'
        generate_analysis_report(
            speed_results,
            choice_results,
            frequency_results,
            str(report_path)
        )
        
        return {
            'speed_results': speed_results,
            'choice_results': choice_results,
            'frequency_results': frequency_results,
            'output_dirs': output_dirs
        }
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Analyze bigram typing data')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        # Run analysis pipeline with just config path
        results = run_analysis_pipeline(args.config)
        logger.info("Analysis completed successfully")
        
        # Print key findings
        print("\nKey Findings:")
        print("-----------")
        print(f"Number of participants: {results['speed_results']['overall']['n_participants']}")
        print(f"Median prediction accuracy: "
              f"{results['choice_results']['aggregate']['median_accuracy']:.1%}")
        print(f"Frequency effect size: "
              f"{results['frequency_results']['aggregate_relative_freq_effect']['median']:.3f}")
        
        # Print output locations
        print("\nOutput Locations:")
        print("----------------")
        for analysis_type, directory in results['output_dirs'].items():
            print(f"{analysis_type}: {directory}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise