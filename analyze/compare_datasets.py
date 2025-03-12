"""
Dataset Comparison Script for Bigram Typing Experiments

This script compares two datasets of bigram typing experiments to assess their comparability.
It analyzes various aspects including:
- Consistency of choices
- Relationship between preference and timing
- Distribution of typing times and slider values
- Predictive accuracy of typing speed for preferences
- Statistical model parameters

Usage:
    python compare_datasets.py --dataset1 path/to/dataset1.csv --dataset2 path/to/dataset2.csv --output path/to/output/folder
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Any, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import median_abs_deviation
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """
    Load and validate dataset from CSV file.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        DataFrame containing the validated data
    """
    required_columns = [
        'user_id', 'chosen_bigram', 'unchosen_bigram',
        'chosen_bigram_time', 'unchosen_bigram_time', 'sliderValue'
    ]
    
    try:
        # First check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Try to read a small sample first to verify file format
        logger.debug(f"Attempting to read file header from {data_path}")
        sample = pd.read_csv(data_path, nrows=5)
        logger.debug(f"Sample columns: {', '.join(sample.columns)}")
        
        # Now read the full file
        logger.debug(f"Reading full dataset from {data_path}")
        df = pd.read_csv(data_path)
        logger.debug(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Print first few rows for debugging
        logger.debug(f"First 2 rows:\n{df.head(2)}")
        
        # Validate columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            missing_cols_str = ', '.join(missing_cols)
            available_cols_str = ', '.join(df.columns)
            raise ValueError(f"Missing required columns: {missing_cols_str}. Available columns: {available_cols_str}")
            
        # Convert numeric columns
        numeric_cols = ['chosen_bigram_time', 'unchosen_bigram_time', 'sliderValue']
        for col in numeric_cols:
            logger.debug(f"Converting {col} to numeric")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            n_dropped = df[col].isna().sum()
            if n_dropped > 0:
                logger.warning(f"Found {n_dropped} non-numeric values in {col}")
        
        # Check for any unusually large or small values
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            logger.debug(f"Column {col} range: {min_val} to {max_val}")
            if max_val > 100000:  # Very large value for typing time
                logger.warning(f"Column {col} has unusually large values (max: {max_val})")
        
        # Remove rows with NaN values
        n_before = len(df)
        df = df.dropna(subset=numeric_cols)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning(f"Dropped {n_dropped} rows with missing values")
            
        # Add calculated columns that are useful for analysis
        logger.debug("Adding calculated columns")
        df['time_diff'] = df['chosen_bigram_time'] - df['unchosen_bigram_time']
        df['speed_predicts_choice'] = df['time_diff'] < 0
        df['abs_slider_value'] = df['sliderValue'].abs()
        
        # Get standardized bigram pairs for easier comparison
        logger.debug("Creating standardized bigram pairs")
        df['bigram_pair'] = df.apply(
            lambda row: tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']])), 
            axis=1
        )
        
        logger.info(f"Loaded {len(df)} rows from {len(df['user_id'].unique())} participants")
        logger.info(f"Found {df['bigram_pair'].nunique()} unique bigram pairs")
        
        return df
            
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {data_path}: {str(e)}")
        print(f"ERROR: Could not parse CSV file. Check if the file format is correct: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {str(e)}")
        print(f"ERROR loading data: {str(e)}")
        raise

def compare_basic_statistics(data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare basic statistics between datasets.
    """
    basic_stats = {}  # This variable is correctly named 'basic_stats' here
    
    # User statistics
    basic_stats['users'] = {
        'dataset1_count': data1['user_id'].nunique(),
        'dataset2_count': data2['user_id'].nunique()
    }
    
    # Trial statistics
    basic_stats['trials'] = {  # Changed 'stats' to 'basic_stats' to use the correct variable
        'dataset1_count': len(data1),
        'dataset2_count': len(data2),
        'dataset1_per_user': len(data1) / data1['user_id'].nunique(),
        'dataset2_per_user': len(data2) / data2['user_id'].nunique()
    }
    
    # Bigram pair statistics
    basic_stats['bigram_pairs'] = {  # Changed 'stats' to 'basic_stats'
        'dataset1_count': data1['bigram_pair'].nunique(),
        'dataset2_count': data2['bigram_pair'].nunique(),
        'overlap_count': len(set(data1['bigram_pair'].unique()) & 
                           set(data2['bigram_pair'].unique()))
    }
    
    # Typing time statistics
    for col in ['chosen_bigram_time', 'unchosen_bigram_time']:
        basic_stats[f'{col}_stats'] = {  # Changed 'stats' to 'basic_stats'
            'dataset1_median': data1[col].median(),
            'dataset2_median': data2[col].median(),
            'dataset1_mad': median_abs_deviation(data1[col], nan_policy='omit'),
            'dataset2_mad': median_abs_deviation(data2[col], nan_policy='omit'),
            'p_value': stats.mannwhitneyu(data1[col], data2[col]).pvalue
        }
    
    # Slider value statistics
    basic_stats['slider_values'] = {  # Changed 'stats' to 'basic_stats'
        'dataset1_median_abs': data1['abs_slider_value'].median(),
        'dataset2_median_abs': data2['abs_slider_value'].median(),
        'p_value': stats.mannwhitneyu(data1['abs_slider_value'], data2['abs_slider_value']).pvalue
    }
    
    # Speed prediction accuracy
    basic_stats['speed_prediction'] = {  # Changed 'stats' to 'basic_stats'
        'dataset1_accuracy': data1['speed_predicts_choice'].mean(),
        'dataset2_accuracy': data2['speed_predicts_choice'].mean(),
        'p_value': stats.ttest_ind(
            data1['speed_predicts_choice'].astype(float), 
            data2['speed_predicts_choice'].astype(float)
        ).pvalue
    }
    
    return basic_stats

def compare_user_consistency(data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare user consistency between datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        
    Returns:
        Dictionary containing consistency comparison results
    """
    results = {}
    
    # Function to calculate consistency for a dataset
    def calculate_consistency(data):
        # Group by user and bigram pair
        user_consistency = []
        
        for (user_id, pair), group in data.groupby(['user_id', 'bigram_pair']):
            if len(group) > 1:
                # Calculate consistency as how often the user chooses the same bigram
                chosen_counts = group['chosen_bigram'].value_counts()
                max_count = chosen_counts.max()
                consistency = max_count / len(group)
                
                user_consistency.append({
                    'user_id': user_id,
                    'bigram_pair': pair,
                    'consistency': consistency,
                    'n_presentations': len(group)
                })
        
        return pd.DataFrame(user_consistency)
    
    # Calculate consistency for each dataset
    consistency1 = calculate_consistency(data1)
    consistency2 = calculate_consistency(data2)
    
    # Calculate average consistency per user
    user_avg_consistency1 = consistency1.groupby('user_id')['consistency'].mean()
    user_avg_consistency2 = consistency2.groupby('user_id')['consistency'].mean()
    
    results['user_consistency'] = {
        'dataset1_mean': user_avg_consistency1.mean(),
        'dataset2_mean': user_avg_consistency2.mean(),
        'dataset1_median': user_avg_consistency1.median(),
        'dataset2_median': user_avg_consistency2.median(),
        'dataset1_std': user_avg_consistency1.std(),
        'dataset2_std': user_avg_consistency2.std(),
        'p_value': stats.mannwhitneyu(user_avg_consistency1, user_avg_consistency2).pvalue
    }
    
    # Overall consistency distribution
    results['overall_consistency'] = {
        'dataset1_mean': consistency1['consistency'].mean(),
        'dataset2_mean': consistency2['consistency'].mean(),
        'dataset1_median': consistency1['consistency'].median(),
        'dataset2_median': consistency2['consistency'].median(),
        'p_value': stats.mannwhitneyu(consistency1['consistency'], consistency2['consistency']).pvalue
    }
    
    return results, consistency1, consistency2

def compare_distributions(data1: pd.DataFrame, data2: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """
    Compare distributions of key metrics between datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        output_folder: Folder to save plots
        
    Returns:
        Dictionary containing distribution comparison results
    """
    dist_results = {}
    
    # Prepare data for plotting
    data1 = data1.copy()
    data2 = data2.copy()
    data1['dataset'] = 'Dataset 1'
    data2['dataset'] = 'Dataset 2'
    combined = pd.concat([data1, data2])
    
    # 1. Compare typing time distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Chosen times
    sns.histplot(data=combined, x='chosen_bigram_time', hue='dataset', 
                 ax=axes[0], kde=True, common_norm=False, alpha=0.6)
    axes[0].set_title('Chosen Bigram Typing Times')
    
    # Unchosen times
    sns.histplot(data=combined, x='unchosen_bigram_time', hue='dataset', 
                 ax=axes[1], kde=True, common_norm=False, alpha=0.6)
    axes[1].set_title('Unchosen Bigram Typing Times')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'typing_time_distribution_comparison.png'))
    plt.close()
    
    # 2. Compare slider value distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=combined, x='sliderValue', hue='dataset', 
                 kde=True, common_norm=False, alpha=0.6)
    ax.set_title('Slider Value Distributions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'slider_value_distribution_comparison.png'))
    plt.close()
    
    # 3. Compare time difference distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=combined, x='time_diff', hue='dataset', 
                 kde=True, common_norm=False, alpha=0.6)
    ax.set_title('Time Difference (Chosen - Unchosen) Distributions')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'time_diff_distribution_comparison.png'))
    plt.close()
    
    # 4. Statistical tests of distribution similarity
    for col in ['chosen_bigram_time', 'unchosen_bigram_time', 'sliderValue', 'time_diff']:
        # KS test
        ks_result = stats.ks_2samp(data1[col], data2[col])
        
        # Anderson-Darling test (if available)
        try:
            ad_result = stats.anderson_ksamp([data1[col].values, data2[col].values])
            ad_pvalue = ad_result.significance_level
        except:
            ad_pvalue = np.nan
            
        dist_results[col] = {
            'ks_statistic': ks_result.statistic,
            'ks_pvalue': ks_result.pvalue,
            'ad_statistic': getattr(ad_result, 'statistic', np.nan) if 'ad_result' in locals() else np.nan,
            'ad_pvalue': ad_pvalue
        }    
    return dist_results

def compare_preference_timing_relationship(data1: pd.DataFrame, data2: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """
    Compare relationship between timing and preference across datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        output_folder: Folder to save plots
        
    Returns:
        Dictionary containing relationship comparison results
    """
    relationship_results = {}
    
    # Plot regression relationships
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dataset 1
    sns.regplot(x='sliderValue', y='time_diff', data=data1, 
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'blue', 'label': 'Dataset 1'})
    
    # Dataset 2 
    sns.regplot(x='sliderValue', y='time_diff', data=data2, 
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'red', 'label': 'Dataset 2'})
    
    ax.set_xlabel('Slider Value (Preference)')
    ax.set_ylabel('Time Difference (Chosen - Unchosen)')
    ax.set_title('Relationship Between Preference and Timing')
    
    # Add custom legend for regression lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Dataset 1'),
        Line2D([0], [0], color='red', lw=2, label='Dataset 2')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'preference_timing_relationship.png'))
    plt.close()
    
    # Calculate regression coefficients for each dataset
    reg1 = stats.linregress(data1['sliderValue'], data1['time_diff'])
    reg2 = stats.linregress(data2['sliderValue'], data2['time_diff'])
    
    # Compare regression slopes
    slope_diff_p = compare_regression_slopes(data1, data2, 'sliderValue', 'time_diff')
    
    relationship_results['preference_timing'] = {
        'dataset1_slope': reg1.slope,
        'dataset2_slope': reg2.slope,
        'dataset1_r_value': reg1.rvalue,
        'dataset2_r_value': reg2.rvalue,
        'dataset1_p_value': reg1.pvalue,
        'dataset2_p_value': reg2.pvalue,
        'slopes_difference_p_value': slope_diff_p
    }
    
    # Per-user correlations between slider value and time difference
    user_corrs1 = []
    user_corrs2 = []
    
    for user_id, group in data1.groupby('user_id'):
        if len(group) >= 5:  # Minimum number of trials for correlation
            corr = stats.spearmanr(group['sliderValue'], group['time_diff']).correlation
            user_corrs1.append(corr)
    
    for user_id, group in data2.groupby('user_id'):
        if len(group) >= 5:
            corr = stats.spearmanr(group['sliderValue'], group['time_diff']).correlation
            user_corrs2.append(corr)
    
    # Plot distribution of user correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(user_corrs1, color='blue', label='Dataset 1', alpha=0.6, kde=True)
    sns.histplot(user_corrs2, color='red', label='Dataset 2', alpha=0.6, kde=True)
    ax.set_xlabel('Correlation Coefficient (Spearman\'s rho)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of User-Level Correlations Between Preference and Timing')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'user_preference_timing_correlations.png'))
    plt.close()
    
    relationship_results['user_correlations'] = {
        'dataset1_mean': np.mean(user_corrs1),
        'dataset2_mean': np.mean(user_corrs2),
        'dataset1_median': np.median(user_corrs1),
        'dataset2_median': np.median(user_corrs2),
        'p_value': stats.mannwhitneyu(user_corrs1, user_corrs2).pvalue
    }
    
    return relationship_results

def compare_prediction_models(data1: pd.DataFrame, data2: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """
    Compare predictive models across datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        output_folder: Folder to save results
        
    Returns:
        Dictionary containing model comparison results
    """
    model_results = {}
    
    # Basic prediction accuracy by speed (already calculated in data)
    accuracy1 = data1['speed_predicts_choice'].mean()
    accuracy2 = data2['speed_predicts_choice'].mean()
    
    # 1. Compare magnitude-based accuracy
    def prepare_magnitude_data(data):
        data = data.copy()
        data['speed_diff_mag'] = data['time_diff'].abs()
        try:
            # Try to create 5 quantiles
            data['magnitude_quintile'] = pd.qcut(
                data['speed_diff_mag'], 
                5, 
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                duplicates='drop'
            )
        except ValueError:
            # Fall back to 3 quantiles if 5 doesn't work
            try:
                data['magnitude_quintile'] = pd.qcut(
                    data['speed_diff_mag'], 
                    3, 
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )
            except ValueError:
                # Manual binning as last resort
                data['magnitude_quintile'] = pd.cut(
                    data['speed_diff_mag'],
                    bins=[0, data['speed_diff_mag'].quantile(0.33), 
                          data['speed_diff_mag'].quantile(0.67), float('inf')],
                    labels=['Low', 'Medium', 'High'],
                    include_lowest=True
                )
        
        # Calculate accuracy by magnitude
        accuracy_by_mag = data.groupby('magnitude_quintile', observed=True)[
            'speed_predicts_choice'
        ].agg(['mean', 'std', 'count']).reset_index()
        
        return accuracy_by_mag
    
    # Calculate accuracy by magnitude for each dataset
    accuracy_by_mag1 = prepare_magnitude_data(data1)
    accuracy_by_mag2 = prepare_magnitude_data(data2)
    
    # Plot accuracy by magnitude
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x1 = np.arange(len(accuracy_by_mag1))
    x2 = np.arange(len(accuracy_by_mag2))
    
    ax.bar(x1 - 0.2, accuracy_by_mag1['mean'], width=0.4, color='blue', label='Dataset 1',
           yerr=accuracy_by_mag1['std'], capsize=5)
    ax.bar(x2 + 0.2, accuracy_by_mag2['mean'], width=0.4, color='red', label='Dataset 2',
           yerr=accuracy_by_mag2['std'], capsize=5)
    
    # Add sample sizes
    for i, row in accuracy_by_mag1.iterrows():
        ax.text(i - 0.2, row['mean'] + 0.02, f"n={int(row['count'])}", 
                ha='center', va='bottom', fontsize=8)
    
    for i, row in accuracy_by_mag2.iterrows():
        ax.text(i + 0.2, row['mean'] + 0.02, f"n={int(row['count'])}", 
                ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Speed Difference Magnitude')
    ax.set_ylabel('Prediction Accuracy')
    ax.set_title('Speed Prediction Accuracy by Magnitude')
    ax.set_xticks(range(len(accuracy_by_mag1)))
    ax.set_xticklabels(accuracy_by_mag1['magnitude_quintile'])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'prediction_accuracy_by_magnitude.png'))
    plt.close()
    
    # 2. Fit logistic regression models
    def fit_logistic_model(data):
        # Target: whether chosen bigram was the "right" bigram (positive slider value)
        y = (data['sliderValue'] > 0).astype(int)
        
        # Predictor: time difference (standardized)
        X = data['time_diff'].values.reshape(-1, 1)
        X_std = (X - X.mean()) / X.std()
        
        # Fit basic logistic regression
        model = LogisticRegression(random_state=42)
        model.fit(X_std, y)
        
        # Calculate ROC AUC
        y_pred_proba = model.predict_proba(X_std)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        
        # Calculate cross-validated AUC
        cv_auc = np.mean(cross_val_score(model, X_std, y, cv=5, scoring='roc_auc'))
        
        # Calculate pseudo-R² using statsmodels
        X_sm = sm.add_constant((X - X.mean()) / X.std())
        sm_model = sm.Logit(y, X_sm).fit(disp=0)
        
        # Calculate null model log-likelihood
        null_model = sm.Logit(y, np.ones_like(y)).fit(disp=0)
        pseudo_r2 = 1 - (sm_model.llf / null_model.llf)
        
        return {
            'coefficient': model.coef_[0][0],
            'intercept': model.intercept_[0],
            'auc': auc,
            'cv_auc': cv_auc,
            'pseudo_r2': pseudo_r2,
            'p_value': sm_model.pvalues[1],
            'log_likelihood': sm_model.llf
        }
    
    # Fit models to both datasets
    try:
        model1 = fit_logistic_model(data1)
        model2 = fit_logistic_model(data2)
        
        model_results['logistic_regression'] = {
            'dataset1': model1,
            'dataset2': model2
        }
    except Exception as e:
        logger.error(f"Error fitting logistic models: {str(e)}")
        model_results['logistic_regression'] = f"Error: {str(e)}"
    
    # Accuracy comparison with confidence intervals
    def wilson_score_interval(p, n, z=1.96):
        """Calculate Wilson score interval for a proportion"""
        denominator = 1 + z**2/n
        center = (p + z**2/(2*n))/denominator
        spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
        return center - spread, center + spread
    
    n1 = len(data1)
    n2 = len(data2)
    
    ci1_low, ci1_high = wilson_score_interval(accuracy1, n1)
    ci2_low, ci2_high = wilson_score_interval(accuracy2, n2)
    
    model_results['accuracy'] = {
        'dataset1': accuracy1,
        'dataset2': accuracy2,
        'dataset1_ci': (ci1_low, ci1_high),
        'dataset2_ci': (ci2_low, ci2_high),
        'overlap': min(ci1_high, ci2_high) - max(ci1_low, ci2_low) > 0
    }
    
    return model_results

def compare_regression_slopes(data1: pd.DataFrame, data2: pd.DataFrame, x_col: str, y_col: str) -> float:
    """
    Test whether two regression slopes are significantly different.
    
    Args:
        data1: First dataset
        data2: Second dataset
        x_col: Name of x column
        y_col: Name of y column
        
    Returns:
        p-value for the difference in slopes
    """
    # Get regression coefficients and standard errors
    reg1 = stats.linregress(data1[x_col], data1[y_col])
    reg2 = stats.linregress(data2[x_col], data2[y_col])
    
    b1 = reg1.slope
    b2 = reg2.slope
    SEb1 = reg1.stderr
    SEb2 = reg2.stderr
    
    # Calculate the t-statistic
    t = (b1 - b2) / np.sqrt(SEb1**2 + SEb2**2)
    
    # Calculate p-value (two-tailed test)
    df = len(data1) + len(data2) - 4  # degrees of freedom
    p_value = 2 * (1 - stats.t.cdf(abs(t), df))
    
    return p_value

def compare_datasets(data1: pd.DataFrame, data2: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """
    Master function to compare two datasets on all metrics.
    
    Args:
        data1: First dataset
        data2: Second dataset
        output_folder: Folder to save comparison results
        
    Returns:
        Dictionary containing all comparison results
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Run all comparisons
    comparison_results = {}
    error_log = []
    
    # Try each comparison separately to continue even if one fails
    try:
        logger.info("Comparing basic statistics...")
        comparison_results['basic_stats'] = compare_basic_statistics(data1, data2)
    except Exception as e:
        error_msg = f"Error in basic statistics comparison: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['basic_stats'] = {"error": error_msg}
    
    try:
        logger.info("Analyzing user consistency patterns...")
        comparison_results['user_consistency'] = compare_user_consistency(data1, data2)
    except Exception as e:
        error_msg = f"Error in user consistency analysis: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['user_consistency'] = {"error": error_msg}
    
    try:
        logger.info("Comparing distributions...")
        comparison_results['distributions'] = compare_distributions(data1, data2, output_folder)
    except Exception as e:
        error_msg = f"Error in distribution comparison: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['distributions'] = {"error": error_msg}
    
    try:
        logger.info("Analyzing preference-timing relationship...")
        comparison_results['preference_timing'] = compare_preference_timing_relationship(data1, data2, output_folder)
    except Exception as e:
        error_msg = f"Error in preference-timing analysis: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['preference_timing'] = {"error": error_msg}
    
    try:
        logger.info("Comparing prediction models...")
        comparison_results['prediction_models'] = compare_prediction_models(data1, data2, output_folder)
    except Exception as e:
        error_msg = f"Error in prediction model comparison: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['prediction_models'] = {"error": error_msg}
    
    try:
        logger.info("Analyzing inconsistency patterns...")
        comparison_results['inconsistency'] = compare_inconsistency_patterns(data1, data2, output_folder)
    except Exception as e:
        error_msg = f"Error in inconsistency pattern analysis: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['inconsistency'] = {"error": error_msg}
    
    # Save error log if there were any errors
    if error_log:
        with open(os.path.join(output_folder, "analysis_errors.txt"), "w") as f:
            f.write("\n".join(error_log))
    
    # Generate comprehensive report
    try:
        logger.info("Generating comparison report...")
        generate_comparison_report(comparison_results, output_folder)
    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        # At least save the raw results
        import json
        try:
            with open(os.path.join(output_folder, "raw_comparison_results.json"), "w") as f:
                # Convert non-serializable objects to strings
                serializable_results = {}
                for key, value in comparison_results.items():
                    if isinstance(value, dict):
                        serializable_results[key] = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                                                   for k, v in value.items()}
                    else:
                        serializable_results[key] = str(value)
                json.dump(serializable_results, f, indent=2)
        except:
            logger.error("Failed to save raw results as JSON")
    
    return comparison_results

def compare_inconsistency_patterns(data1: pd.DataFrame, data2: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """
    Compare patterns of inconsistency between datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        output_folder: Folder to save results
        
    Returns:
        Dictionary containing inconsistency comparison results
    """
    inconsistency_results = {}
    
    # Function to identify inconsistent choices
    def find_inconsistent_choices(data):
        inconsistent_pairs = []
        
        for (user_id, pair), group in data.groupby(['user_id', 'bigram_pair']):
            if len(group) > 1:
                chosen_bigrams = group['chosen_bigram'].unique()
                if len(chosen_bigrams) > 1:
                    # Inconsistent choice found
                    inconsistent_pairs.append({
                        'user_id': user_id,
                        'bigram_pair': pair,
                        'n_presentations': len(group),
                        'slider_values': group['sliderValue'].tolist(),
                        'abs_slider_values': group['abs_slider_value'].tolist(),
                        'time_diffs': group['time_diff'].tolist()
                    })
        
        return pd.DataFrame(inconsistent_pairs)
    
    # Find inconsistent choices in both datasets
    inconsistent1 = find_inconsistent_choices(data1)
    inconsistent2 = find_inconsistent_choices(data2)
    
    # Calculate percentage of inconsistent choices
    total_repeat_pairs1 = sum(1 for _, group in data1.groupby(['user_id', 'bigram_pair']) if len(group) > 1)
    total_repeat_pairs2 = sum(1 for _, group in data2.groupby(['user_id', 'bigram_pair']) if len(group) > 1)
    
    if total_repeat_pairs1 > 0 and total_repeat_pairs2 > 0:
        pct_inconsistent1 = len(inconsistent1) / total_repeat_pairs1 * 100
        pct_inconsistent2 = len(inconsistent2) / total_repeat_pairs2 * 100
        
        # Statistical test for difference in proportions
        from statsmodels.stats.proportion import proportions_ztest
        
        count = np.array([len(inconsistent1), len(inconsistent2)])
        nobs = np.array([total_repeat_pairs1, total_repeat_pairs2])
        
        try:
            z_stat, p_value = proportions_ztest(count, nobs)
        except:
            # Fall back to a simpler calculation if statsmodels fails
            p1 = count[0] / nobs[0]
            p2 = count[1] / nobs[1]
            p_pooled = (count[0] + count[1]) / (nobs[0] + nobs[1])
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/nobs[0] + 1/nobs[1]))
            z_stat = (p1 - p2) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        inconsistency_results['overall'] = {
            'dataset1_pct': pct_inconsistent1,
            'dataset2_pct': pct_inconsistent2,
            'dataset1_count': len(inconsistent1),
            'dataset2_count': len(inconsistent2),
            'dataset1_total_repeat_pairs': total_repeat_pairs1,
            'dataset2_total_repeat_pairs': total_repeat_pairs2,
            'z_statistic': z_stat,
            'p_value': p_value
        }
        
        # Compare distributions of slider values in inconsistent choices
        if len(inconsistent1) > 0 and len(inconsistent2) > 0:
            # Extract and flatten slider values from both datasets
            slider_values1 = [val for sublist in inconsistent1['abs_slider_values'] for val in sublist]
            slider_values2 = [val for sublist in inconsistent2['abs_slider_values'] for val in sublist]
            
            # Compare distributions
            ks_result = stats.ks_2samp(slider_values1, slider_values2)
            
            inconsistency_results['slider_values'] = {
                'dataset1_mean': np.mean(slider_values1),
                'dataset2_mean': np.mean(slider_values2),
                'dataset1_median': np.median(slider_values1),
                'dataset2_median': np.median(slider_values2),
                'ks_statistic': ks_result.statistic,
                'ks_pvalue': ks_result.pvalue
            }

            # Plot distribution of slider values in inconsistent choices
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(slider_values1, color='blue', label='Dataset 1', alpha=0.6, kde=True)
            sns.histplot(slider_values2, color='red', label='Dataset 2', alpha=0.6, kde=True)
            ax.set_xlabel('Absolute Slider Value')
            ax.set_ylabel('Count')
            ax.set_title('Slider Values in Inconsistent Choices')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'inconsistent_slider_values.png'))
            plt.close()
            
            # Compare distributions of time differences in inconsistent choices
            time_diffs1 = [val for sublist in inconsistent1['time_diffs'] for val in sublist]
            time_diffs2 = [val for sublist in inconsistent2['time_diffs'] for val in sublist]
            
            ks_result = stats.ks_2samp(time_diffs1, time_diffs2)
            
            inconsistency_results['time_diffs'] = {
                'dataset1_mean': np.mean(time_diffs1),
                'dataset2_mean': np.mean(time_diffs2),
                'dataset1_median': np.median(time_diffs1),
                'dataset2_median': np.median(time_diffs2),
                'ks_statistic': ks_result.statistic,
                'ks_pvalue': ks_result.pvalue
            }
            
            # Plot distribution of time differences in inconsistent choices
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(time_diffs1, color='blue', label='Dataset 1', alpha=0.6, kde=True)
            sns.histplot(time_diffs2, color='red', label='Dataset 2', alpha=0.6, kde=True)
            ax.set_xlabel('Time Difference (Chosen - Unchosen)')
            ax.set_ylabel('Count')
            ax.set_title('Time Differences in Inconsistent Choices')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'inconsistent_time_diffs.png'))
            plt.close()
    
    return inconsistency_results    

def compare_slider_scoring_and_winners(data1: pd.DataFrame, data2: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """
    Compare slider score distributions and bigram winners between datasets.
    
    Args:
        data1: First dataset
        data2: Second dataset
        output_folder: Folder to save results
        
    Returns:
        Dictionary containing comparison results for slider scores and bigram winners
    """
    results = {}
    
    # Process each dataset to generate scored bigram data and winner data
    def process_dataset_winners(data, label):
        # Create scored bigram data
        scored_data = []
        
        # Group by user and bigram pair to calculate average scores
        for (user_id, pair), group in data.groupby(['user_id', 'bigram_pair']):
            bigram1, bigram2 = eval(pair) if isinstance(pair, str) else pair
            
            # Calculate normalized score (0-1) from slider values (-100 to 100)
            avg_slider = group['sliderValue'].mean()
            normalized_score = (avg_slider + 100) / 200
            
            # Determine winners based on slider value
            if avg_slider > 0:
                chosen_winner = bigram1
                unchosen_winner = bigram2
            else:
                chosen_winner = bigram2
                unchosen_winner = bigram1
            
            scored_data.append({
                'user_id': user_id,
                'bigram_pair': pair,
                'chosen_bigram_winner': chosen_winner,
                'unchosen_bigram_winner': unchosen_winner,
                'score': normalized_score
            })
        
        scored_df = pd.DataFrame(scored_data)
        
        # Calculate bigram winners across all users
        winner_data = []
        for pair, group in scored_df.groupby('bigram_pair'):
            median_score = group['score'].median()
            mad_score = median_abs_deviation(group['score'], nan_policy='omit')
            
            bigram1, bigram2 = eval(pair) if isinstance(pair, str) else pair
            
            if median_score >= 0.5:
                winner = bigram1
                loser = bigram2
            else:
                winner = bigram2
                loser = bigram1
            
            winner_data.append({
                'bigram_pair': pair,
                'winner_bigram': winner,
                'loser_bigram': loser,
                'median_score': median_score,
                'mad_score': mad_score
            })
        
        return pd.DataFrame(scored_df), pd.DataFrame(winner_data)
    
    # Process each dataset
    scored_df1, winner_df1 = process_dataset_winners(data1, "Dataset 1")
    scored_df2, winner_df2 = process_dataset_winners(data2, "Dataset 2")
    
    # Compare distributions of scores
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(scored_df1['score'], color='blue', label='Dataset 1', alpha=0.6, kde=True)
    sns.histplot(scored_df2['score'], color='red', label='Dataset 2', alpha=0.6, kde=True)
    ax.set_xlabel('Normalized Score (0-1)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Normalized Slider Scores')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'score_distribution_comparison.png'))
    plt.close()
    
    # Compare score MAD (variability) distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(winner_df1['mad_score'], color='blue', label='Dataset 1', alpha=0.6, kde=True)
    sns.histplot(winner_df2['mad_score'], color='red', label='Dataset 2', alpha=0.6, kde=True)
    ax.set_xlabel('Median Absolute Deviation of Scores')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Score Variability')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'score_mad_distribution_comparison.png'))
    plt.close()
    
    # Compare score distributions statistically
    ks_result = stats.ks_2samp(scored_df1['score'], scored_df2['score'])
    mad_ks_result = stats.ks_2samp(winner_df1['mad_score'], winner_df2['mad_score'])
    
    # Find common bigram pairs between datasets
    common_pairs = set(winner_df1['bigram_pair']) & set(winner_df2['bigram_pair'])
    
    # Analyze agreement on winners for common pairs
    if common_pairs:
        # Create merged dataframe for common pairs
        merged_winners = pd.merge(
            winner_df1[winner_df1['bigram_pair'].isin(common_pairs)],
            winner_df2[winner_df2['bigram_pair'].isin(common_pairs)],
            on='bigram_pair', suffixes=('_dataset1', '_dataset2')
        )
        
        # Calculate agreement
        merged_winners['same_winner'] = merged_winners['winner_bigram_dataset1'] == merged_winners['winner_bigram_dataset2']
        agreement_rate = merged_winners['same_winner'].mean()
        
        # Compare scores for common pairs
        merged_winners['score_diff'] = abs(merged_winners['median_score_dataset1'] - merged_winners['median_score_dataset2'])
        
        # Plot score differences for common pairs
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by difference for better visualization
        plot_data = merged_winners.sort_values('score_diff', ascending=False)
        
        # Plot the differences
        bars = ax.bar(range(len(plot_data)), plot_data['score_diff'], color=[
            'red' if not same else 'green' for same in plot_data['same_winner']
        ])
        
        # Add labels
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data['bigram_pair'], rotation=90)
        ax.set_ylabel('Absolute Score Difference')
        ax.set_title('Score Differences for Common Bigram Pairs')
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Same Winner'),
            Patch(facecolor='red', label='Different Winner')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'common_pairs_score_diff.png'))
        plt.close()
        
        # Calculate correlation between scores across datasets
        score_corr = stats.pearsonr(
            merged_winners['median_score_dataset1'], 
            merged_winners['median_score_dataset2']
        )
        
        results['common_pairs'] = {
            'count': len(common_pairs),
            'agreement_rate': agreement_rate,
            'score_correlation': score_corr.statistic,
            'score_correlation_p': score_corr.pvalue,
            'mean_score_diff': merged_winners['score_diff'].mean(),
            'median_score_diff': merged_winners['score_diff'].median(),
            'max_score_diff': merged_winners['score_diff'].max()
        }
    else:
        results['common_pairs'] = {
            'count': 0,
            'note': 'No common bigram pairs found between datasets'
        }
    
    # Store basic distribution statistics
    results['score_distributions'] = {
        'dataset1_mean': scored_df1['score'].mean(),
        'dataset2_mean': scored_df2['score'].mean(),
        'dataset1_median': scored_df1['score'].median(),
        'dataset2_median': scored_df2['score'].median(),
        'dataset1_std': scored_df1['score'].std(),
        'dataset2_std': scored_df2['score'].std(),
        'ks_statistic': ks_result.statistic,
        'ks_pvalue': ks_result.pvalue
    }
    
    results['score_variability'] = {
        'dataset1_mean_mad': winner_df1['mad_score'].mean(),
        'dataset2_mean_mad': winner_df2['mad_score'].mean(),
        'dataset1_median_mad': winner_df1['mad_score'].median(),
        'dataset2_median_mad': winner_df2['mad_score'].median(),
        'ks_statistic': mad_ks_result.statistic,
        'ks_pvalue': mad_ks_result.pvalue
    }
    
    # Save the processed dataframes for reference
    scored_df1.to_csv(os.path.join(output_folder, 'scored_bigram_data_dataset1.csv'), index=False)
    scored_df2.to_csv(os.path.join(output_folder, 'scored_bigram_data_dataset2.csv'), index=False)
    winner_df1.to_csv(os.path.join(output_folder, 'bigram_winner_data_dataset1.csv'), index=False)
    winner_df2.to_csv(os.path.join(output_folder, 'bigram_winner_data_dataset2.csv'), index=False)
    
    return results

def compare_datasets(data1: pd.DataFrame, data2: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
    """
    Master function to compare two datasets on all metrics.
    
    Args:
        data1: First dataset
        data2: Second dataset
        output_folder: Folder to save comparison results
        
    Returns:
        Dictionary containing all comparison results
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Run all comparisons
    comparison_results = {}
    error_log = []
    
    # Try each comparison separately to continue even if one fails
    try:
        logger.info("Comparing basic statistics...")
        comparison_results['basic_stats'] = compare_basic_statistics(data1, data2)
    except Exception as e:
        error_msg = f"Error in basic statistics comparison: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['basic_stats'] = {"error": error_msg}
    
    try:
        logger.info("Analyzing user consistency patterns...")
        comparison_results['user_consistency'] = compare_user_consistency(data1, data2)
    except Exception as e:
        error_msg = f"Error in user consistency analysis: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['user_consistency'] = {"error": error_msg}
    
    try:
        logger.info("Comparing distributions...")
        comparison_results['distributions'] = compare_distributions(data1, data2, output_folder)
    except Exception as e:
        error_msg = f"Error in distribution comparison: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['distributions'] = {"error": error_msg}
    
    try:
        logger.info("Analyzing preference-timing relationship...")
        comparison_results['preference_timing'] = compare_preference_timing_relationship(data1, data2, output_folder)
    except Exception as e:
        error_msg = f"Error in preference-timing analysis: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['preference_timing'] = {"error": error_msg}
    
    try:
        logger.info("Comparing prediction models...")
        comparison_results['prediction_models'] = compare_prediction_models(data1, data2, output_folder)
    except Exception as e:
        error_msg = f"Error in prediction model comparison: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['prediction_models'] = {"error": error_msg}
    
    try:
        logger.info("Analyzing inconsistency patterns...")
        comparison_results['inconsistency'] = compare_inconsistency_patterns(data1, data2, output_folder)
    except Exception as e:
        error_msg = f"Error in inconsistency pattern analysis: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['inconsistency'] = {"error": error_msg}
    
    try:
        logger.info("Comparing slider scoring and bigram winners...")
        comparison_results['slider_scoring'] = compare_slider_scoring_and_winners(data1, data2, output_folder)
    except Exception as e:
        error_msg = f"Error in slider scoring comparison: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        error_log.append(error_msg)
        comparison_results['slider_scoring'] = {"error": error_msg}
    
    # Save error log if there were any errors
    if error_log:
        with open(os.path.join(output_folder, "analysis_errors.txt"), "w") as f:
            f.write("\n".join(error_log))
    
    # Generate comprehensive report
    try:
        logger.info("Generating comparison report...")
        generate_comparison_report(comparison_results, output_folder)
    except Exception as e:
        error_msg = f"Error generating report: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        # At least save the raw results
        import json
        try:
            with open(os.path.join(output_folder, "raw_comparison_results.json"), "w") as f:
                # Convert non-serializable objects to strings
                serializable_results = {}
                for key, value in comparison_results.items():
                    if isinstance(value, dict):
                        serializable_results[key] = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                                                   for k, v in value.items()}
                    else:
                        serializable_results[key] = str(value)
                json.dump(serializable_results, f, indent=2)
        except:
            logger.error("Failed to save raw results as JSON")
    
    return comparison_results

def generate_comparison_report(comparison_results: Dict[str, Any], output_folder: str, filename: str = "dataset_comparison_report.txt") -> None:
    """Generate a comprehensive comparison report from the analysis results."""
    report_lines = [
        "Dataset Comparison Analysis Report",
        "=================================\n",
    ]
    
    # Check if basic_stats exists and is a dictionary (not an error message)
    if 'basic_stats' in comparison_results and isinstance(comparison_results['basic_stats'], dict) and 'error' not in comparison_results['basic_stats']:
        basic_stats = comparison_results['basic_stats']
        report_lines.extend([
            "Basic Statistics Comparison:",
            "--------------------------"
        ])
        
        # User statistics
        if 'users' in basic_stats:
            report_lines.extend([
                f"Number of users:",
                f"  Dataset 1: {basic_stats['users']['dataset1_count']}",
                f"  Dataset 2: {basic_stats['users']['dataset2_count']}",
                ""
            ])
    
        # Trial statistics
        report_lines.extend([
            f"Number of trials:",
            f"  Dataset 1: {basic_stats['trials']['dataset1_count']}",
            f"  Dataset 2: {basic_stats['trials']['dataset2_count']}",
            f"Average trials per user:",
            f"  Dataset 1: {basic_stats['trials']['dataset1_per_user']:.2f}",
            f"  Dataset 2: {basic_stats['trials']['dataset2_per_user']:.2f}",
            ""
        ])
        
        # Bigram pair statistics
        report_lines.extend([
            f"Unique bigram pairs:",
            f"  Dataset 1: {basic_stats['bigram_pairs']['dataset1_count']}",
            f"  Dataset 2: {basic_stats['bigram_pairs']['dataset2_count']}",
            f"  Overlap: {basic_stats['bigram_pairs']['overlap_count']}",
            ""
        ])
        
        # Typing time statistics
        report_lines.extend([
            "Typing time statistics:",
            f"  Chosen bigram time (median):",
            f"    Dataset 1: {basic_stats['chosen_bigram_time_stats']['dataset1_median']:.2f} ms",
            f"    Dataset 2: {basic_stats['chosen_bigram_time_stats']['dataset2_median']:.2f} ms",
            f"    p-value: {basic_stats['chosen_bigram_time_stats']['p_value']:.4e}",
            f"  Unchosen bigram time (median):",
            f"    Dataset 1: {basic_stats['unchosen_bigram_time_stats']['dataset1_median']:.2f} ms",
            f"    Dataset 2: {basic_stats['unchosen_bigram_time_stats']['dataset2_median']:.2f} ms",
            f"    p-value: {basic_stats['unchosen_bigram_time_stats']['p_value']:.4e}",
            ""
        ])
        
        # Speed prediction accuracy
        report_lines.extend([
            "Speed prediction accuracy:",
            f"  Dataset 1: {basic_stats['speed_prediction']['dataset1_accuracy']:.2%}",
            f"  Dataset 2: {basic_stats['speed_prediction']['dataset2_accuracy']:.2%}",
            f"  p-value: {basic_stats['speed_prediction']['p_value']:.4e}",
            ""
        ])
    
    # User consistency comparison
    if 'user_consistency' in comparison_results:
        consistency = comparison_results['user_consistency'][0]['user_consistency']
        report_lines.extend([
            "User Consistency Comparison:",
            "--------------------------",
            f"Average user consistency:",
            f"  Dataset 1: {consistency['dataset1_mean']:.2%}",
            f"  Dataset 2: {consistency['dataset2_mean']:.2%}",
            f"  p-value: {consistency['p_value']:.4e}",
            f"  Interpretation: {'Significantly different' if consistency['p_value'] < 0.05 else 'Not significantly different'}",
            ""
        ])
    
    # Distribution comparisons
    if 'distributions' in comparison_results:
        distributions = comparison_results['distributions']
        report_lines.extend([
            "Distribution Comparisons:",
            "------------------------"
        ])
        
        for col, stats in distributions.items():
            report_lines.extend([
                f"{col} distribution comparison:",
                f"  Kolmogorov-Smirnov test: statistic={stats['ks_statistic']:.4f}, p-value={stats['ks_pvalue']:.4e}",
                f"  Interpretation: {'Distributions are significantly different' if stats['ks_pvalue'] < 0.05 else 'No significant difference detected'}",
                ""
            ])
    
    # Preference-timing relationship
    if 'preference_timing' in comparison_results:
        timing = comparison_results['preference_timing']['preference_timing']
        user_corrs = comparison_results['preference_timing']['user_correlations']
        
        report_lines.extend([
            "Preference-Timing Relationship:",
            "-----------------------------",
            f"Overall relationship (linear regression):",
            f"  Dataset 1: slope={timing['dataset1_slope']:.4f}, r-value={timing['dataset1_r_value']:.4f}, p-value={timing['dataset1_p_value']:.4e}",
            f"  Dataset 2: slope={timing['dataset2_slope']:.4f}, r-value={timing['dataset2_r_value']:.4f}, p-value={timing['dataset2_p_value']:.4e}",
            f"  Slopes difference p-value: {timing['slopes_difference_p_value']:.4e}",
            f"  Interpretation: {'Slopes are significantly different' if timing['slopes_difference_p_value'] < 0.05 else 'No significant difference in slopes'}",
            "",
            f"User-level correlations:",
            f"  Dataset 1: mean={user_corrs['dataset1_mean']:.4f}, median={user_corrs['dataset1_median']:.4f}",
            f"  Dataset 2: mean={user_corrs['dataset2_mean']:.4f}, median={user_corrs['dataset2_median']:.4f}",
            f"  p-value: {user_corrs['p_value']:.4e}",
            f"  Interpretation: {'Significantly different' if user_corrs['p_value'] < 0.05 else 'Not significantly different'}",
            ""
        ])
    
    # Prediction models
    if 'prediction_models' in comparison_results and isinstance(comparison_results['prediction_models'], dict):
        model_results = comparison_results['prediction_models']
        
        # Basic accuracy
        accuracy = model_results['accuracy']
        report_lines.extend([
            "Prediction Model Comparison:",
            "--------------------------",
            f"Basic speed prediction accuracy:",
            f"  Dataset 1: {accuracy['dataset1']:.2%} (95% CI: {accuracy['dataset1_ci'][0]:.2%} - {accuracy['dataset1_ci'][1]:.2%})",
            f"  Dataset 2: {accuracy['dataset2']:.2%} (95% CI: {accuracy['dataset2_ci'][0]:.2%} - {accuracy['dataset2_ci'][1]:.2%})",
            f"  Confidence intervals {'overlap' if accuracy['overlap'] else 'do not overlap'}",
            ""
        ])
        
        # Logistic regression 
        if 'logistic_regression' in model_results and isinstance(model_results['logistic_regression'], dict):
            log_reg = model_results['logistic_regression']
            model1 = log_reg['dataset1']
            model2 = log_reg['dataset2']
            
            report_lines.extend([
                f"Logistic regression models:",
                f"  Dataset 1: coefficient={model1['coefficient']:.4f}, pseudo-R²={model1['pseudo_r2']:.4f}, AUC={model1['auc']:.4f}, p-value={model1['p_value']:.4e}",
                f"  Dataset 2: coefficient={model2['coefficient']:.4f}, pseudo-R²={model2['pseudo_r2']:.4f}, AUC={model2['auc']:.4f}, p-value={model2['p_value']:.4e}",
                ""
            ])
    
    # Inconsistency patterns
    if 'inconsistency' in comparison_results and 'overall' in comparison_results['inconsistency']:
        inconsistency = comparison_results['inconsistency']['overall']
        
        report_lines.extend([
            "Inconsistency Pattern Comparison:",
            "-------------------------------",
            f"Inconsistent choice rate:",
            f"  Dataset 1: {inconsistency['dataset1_pct']:.2f}% ({inconsistency['dataset1_count']} of {inconsistency['dataset1_total_repeat_pairs']} repeated pairs)",
            f"  Dataset 2: {inconsistency['dataset2_pct']:.2f}% ({inconsistency['dataset2_count']} of {inconsistency['dataset2_total_repeat_pairs']} repeated pairs)",
            f"  Z-statistic: {inconsistency['z_statistic']:.4f}",
            f"  p-value: {inconsistency['p_value']:.4e}",
            f"  Interpretation: {'Significantly different rates' if inconsistency['p_value'] < 0.05 else 'No significant difference in rates'}",
            ""
        ])

    # Slider Score and Bigram Winner Comparison
    if 'slider_scoring' in comparison_results and isinstance(comparison_results['slider_scoring'], dict):
        slider_results = comparison_results['slider_scoring']
        
        report_lines.extend([
            "Slider Score and Bigram Winner Comparison:",
            "--------------------------------------"
        ])
        
        # Score distributions
        if 'score_distributions' in slider_results:
            scores = slider_results['score_distributions']
            report_lines.extend([
                f"Score distributions:",
                f"  Dataset 1: mean={scores['dataset1_mean']:.4f}, median={scores['dataset1_median']:.4f}, std={scores['dataset1_std']:.4f}",
                f"  Dataset 2: mean={scores['dataset2_mean']:.4f}, median={scores['dataset2_median']:.4f}, std={scores['dataset2_std']:.4f}",
                f"  KS-test: statistic={scores['ks_statistic']:.4f}, p-value={scores['ks_pvalue']:.4e}",
                f"  Interpretation: {'Distributions are significantly different' if scores['ks_pvalue'] < 0.05 else 'No significant difference detected'}",
                ""
            ])
        
        # Score variability
        if 'score_variability' in slider_results:
            var = slider_results['score_variability']
            report_lines.extend([
                f"Score variability (MAD):",
                f"  Dataset 1: mean={var['dataset1_mean_mad']:.4f}, median={var['dataset1_median_mad']:.4f}",
                f"  Dataset 2: mean={var['dataset2_mean_mad']:.4f}, median={var['dataset2_median_mad']:.4f}",
                f"  KS-test: statistic={var['ks_statistic']:.4f}, p-value={var['ks_pvalue']:.4e}",
                f"  Interpretation: {'Variability distributions are significantly different' if var['ks_pvalue'] < 0.05 else 'No significant difference in variability'}",
                ""
            ])
        
        # Common pairs analysis
        if 'common_pairs' in slider_results:
            common = slider_results['common_pairs']
            if common.get('count', 0) > 0:
                report_lines.extend([
                    f"Common bigram pairs analysis:",
                    f"  Number of common pairs: {common['count']}",
                    f"  Agreement rate on winners: {common['agreement_rate']:.2%}",
                    f"  Score correlation: r={common['score_correlation']:.4f}, p-value={common['score_correlation_p']:.4e}",
                    f"  Score differences: mean={common['mean_score_diff']:.4f}, median={common['median_score_diff']:.4f}, max={common['max_score_diff']:.4f}",
                    f"  Interpretation: {'Scores are significantly correlated' if common['score_correlation_p'] < 0.05 else 'No significant correlation in scores'}",
                    ""
                ])
            else:
                report_lines.extend([
                    f"Common bigram pairs analysis: {common.get('note', 'No common pairs found')}",
                    ""
                ])
    
    # Overall assessment
    report_lines.extend([
        "Overall Assessment:",
        "-----------------"
    ])
    
    # Count significant differences
    sig_diff_count = 0
    total_tests = 0
    
    # Basic stats
    for key in ['chosen_bigram_time_stats', 'unchosen_bigram_time_stats', 'slider_values', 'speed_prediction']:
        if key in basic_stats and 'p_value' in basic_stats[key]:
            if basic_stats[key]['p_value'] < 0.05:
                sig_diff_count += 1
            total_tests += 1
    
    # User consistency
    if 'user_consistency' in comparison_results:
        if comparison_results['user_consistency'][0]['user_consistency']['p_value'] < 0.05:
            sig_diff_count += 1
        total_tests += 1
    
    # Distributions
    if 'distributions' in comparison_results:
        for stats in comparison_results['distributions'].values():
            if stats['ks_pvalue'] < 0.05:
                sig_diff_count += 1
            total_tests += 1
    
    # Preference-timing relationship
    if 'preference_timing' in comparison_results:
        if comparison_results['preference_timing']['preference_timing']['slopes_difference_p_value'] < 0.05:
            sig_diff_count += 1
        total_tests += 1
        
        if comparison_results['preference_timing']['user_correlations']['p_value'] < 0.05:
            sig_diff_count += 1
        total_tests += 1
    
    # Inconsistency patterns
    if 'inconsistency' in comparison_results and 'overall' in comparison_results['inconsistency']:
        if comparison_results['inconsistency']['overall']['p_value'] < 0.05:
            sig_diff_count += 1
        total_tests += 1
    
    # Slider scoring
    if 'slider_scoring' in comparison_results:
        if 'score_distributions' in comparison_results['slider_scoring']:
            if comparison_results['slider_scoring']['score_distributions']['ks_pvalue'] < 0.05:
                sig_diff_count += 1
            total_tests += 1
            
        if 'score_variability' in comparison_results['slider_scoring']:
            if comparison_results['slider_scoring']['score_variability']['ks_pvalue'] < 0.05:
                sig_diff_count += 1
            total_tests += 1
            
        if 'common_pairs' in comparison_results['slider_scoring'] and 'score_correlation_p' in comparison_results['slider_scoring']['common_pairs']:
            # Note: Here we're looking for NON-significance as a sign of difference
            if comparison_results['slider_scoring']['common_pairs']['score_correlation_p'] >= 0.05:
                sig_diff_count += 1
            total_tests += 1
    
    # Make assessment
    if sig_diff_count == 0:
        assessment = "HIGHLY COMPARABLE - No significant differences detected"
    elif sig_diff_count / total_tests < 0.25:
        assessment = "MOSTLY COMPARABLE - Few significant differences detected"
    elif sig_diff_count / total_tests < 0.5:
        assessment = "SOMEWHAT COMPARABLE - Some significant differences detected"
    else:
        assessment = "NOT COMPARABLE - Many significant differences detected"
    
    report_lines.append(f"{assessment} ({sig_diff_count}/{total_tests} tests showed significant differences)")
    
    # Write report to file
    with open(os.path.join(output_folder, filename), 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Comparison report saved to {os.path.join(output_folder, filename)}")

def main():
    """Main function to parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description='Compare two bigram typing experiment datasets.')
    parser.add_argument('--dataset1', required=True, help='Path to first dataset CSV file')
    parser.add_argument('--dataset2', required=True, help='Path to second dataset CSV file')
    parser.add_argument('--output', required=True, help='Path to output folder')
    parser.add_argument('--clip_time', type=float, default=5000, 
                        help='Maximum typing time to consider (ms, to remove outliers)')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode with more verbose output')
    
    args = parser.parse_args()
    
    # Setup more verbose logging if debug mode is enabled
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Add console handler to see output directly
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    try:
        # Print startup message
        logger.info("Starting dataset comparison script")
        logger.info(f"Dataset 1: {args.dataset1}")
        logger.info(f"Dataset 2: {args.dataset2}")
        logger.info(f"Output folder: {args.output}")
        
        # Load datasets
        logger.info(f"Loading dataset 1 from {args.dataset1}...")
        print(f"Loading dataset 1 from {args.dataset1}...")
        data1 = load_and_validate_data(args.dataset1)
        logger.info(f"Dataset 1 loaded: {len(data1)} rows, {data1['user_id'].nunique()} users")
        print(f"Dataset 1 loaded: {len(data1)} rows, {data1['user_id'].nunique()} users")
        
        logger.info(f"Loading dataset 2 from {args.dataset2}...")
        print(f"Loading dataset 2 from {args.dataset2}...")
        data2 = load_and_validate_data(args.dataset2)
        logger.info(f"Dataset 2 loaded: {len(data2)} rows, {data2['user_id'].nunique()} users")
        print(f"Dataset 2 loaded: {len(data2)} rows, {data2['user_id'].nunique()} users")
        
        # Apply time clipping if requested
        if args.clip_time > 0:
            logger.info(f"Clipping typing times to {args.clip_time} ms...")
            print(f"Clipping typing times to {args.clip_time} ms...")
            for col in ['chosen_bigram_time', 'unchosen_bigram_time']:
                data1[col] = data1[col].clip(upper=args.clip_time)
                data2[col] = data2[col].clip(upper=args.clip_time)
            
            # Recalculate time difference after clipping
            data1['time_diff'] = data1['chosen_bigram_time'] - data1['unchosen_bigram_time']
            data2['time_diff'] = data2['chosen_bigram_time'] - data2['unchosen_bigram_time']
        
        # Create output folder
        os.makedirs(args.output, exist_ok=True)
        
        # Run comparison
        logger.info("Starting dataset comparison...")
        print("Starting dataset comparison...")
        comparison_results = compare_datasets(data1, data2, args.output)
        
        logger.info("Comparison complete. See report for details.")
        print(f"Comparison complete. Report saved to {args.output}/dataset_comparison_report.txt")
        
    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()