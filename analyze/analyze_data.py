
""" 
# Bigram Typing Analysis Pipeline Documentation

## Overview
The pipeline analyzes relationships between bigram typing times, frequencies, 
and user choices in a typing experiment. It processes data where participants 
were shown pairs of bigrams and chose which they preferred to type.

## Data Structure
- Input: CSV file containing filtered experiment data with columns including:
  - user_id: Participant identifier
  - chosen_bigram/unchosen_bigram: The selected/unselected bigram in each trial
  - chosen_bigram_time/unchosen_bigram_time: Typing times in milliseconds
  - sliderValue: User preference rating (-100 to +100)
  - Additional metadata columns (group_id, trialId, etc.)

## Analysis Components

### 1. Typing Time and Slider Analysis
```analyze_typing_times_slider_values()```
- Compares chosen vs. unchosen bigram typing times using Wilcoxon signed-rank test
- Analyzes correlation between typing time differences and slider values
- Uses absolute time differences: |chosen_time - unchosen_time|
- No normalization applied to preserve raw timing differences
- Generates distribution plots and histograms grouped by slider value ranges

### 2. Within-User Timing Analysis
```analyze_user_typing_times()```
- Analyzes significant timing differences between bigram pairs within each user
- Uses Mann-Whitney U tests for pairwise comparisons
- Raw timing values used (no normalization)
- Aggregates results across users to identify consistent patterns

### 3. Frequency-Timing Relationship Analysis
```plot_frequency_timing_relationship()```
- Examines correlation between bigram frequencies and typing times
- Uses log-transformed frequencies to handle skewed distribution
- Produces three analyses:
  1. Distribution plot with median times and error bars
  2. Minimum (fastest) typing times vs. frequency
  3. Median typing times vs. frequency
- Includes both raw and normalized analyses:
  - Raw: Uses absolute typing times
  - Normalized: Times divided by participant's median typing time

### 4. Time-Frequency Differences Analysis
```analyze_time_frequency_differences()```
- Investigates how frequency differences relate to typing time differences
- Normalizes by user median times to account for typing speed variations
- Uses both raw and normalized differences:
  - Raw diff = chosen_time - unchosen_time
  - Normalized diff = (chosen_time - unchosen_time) / user_median_time
- Frequency differences calculated as log differences:
  - freq_diff = log10(chosen_freq) - log10(unchosen_freq)

### 5. Enhanced Analysis Components
- Choice Analysis:
  - Examines relationship between typing speed, frequency, and user choices
  - Uses both raw times and user-normalized times
- Timing Patterns:
  - Analyzes practice effects and typing variability
  - Uses coefficient of variation (CV) for timing variability
- Statistical Assumption Checks:
  - Tests normality of timing differences
  - Checks temporal independence of trials

## Output Structure
- Generates plots in specified output folders:
  - bigram_typing_time_and_frequency_analysis/
  - bigram_choice_analysis/
- Saves detailed analysis results in text files
- Returns dictionaries with comprehensive statistics and test results

## Statistical Methods
- Non-parametric tests used where possible (Wilcoxon, Mann-Whitney U)
- Spearman correlations for frequency-timing relationships
- Linear regression for trend analysis
- ANOVA for group comparisons with post-hoc Tukey HSD tests

Calculate prediction accuracy:
- For each bigram pair, see if the faster-typed bigram was chosen
- Get overall accuracy rate and confidence intervals
    - Break this down by:
        - Size of speed difference (does bigger difference = more reliable prediction?)
        - Individual bigrams or bigram types
        - Individual participants
Analyze false predictions:
- When does speed fail to predict choice?
- Are there patterns in these cases?
- Are certain bigrams or bigram combinations particularly problematic?
Consider practical application:
- What accuracy threshold would make speed a useful proxy?
- Would it work better for certain subsets of bigrams?
- Could we identify conditions where speed is/isn't a reliable predictor?

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import median_abs_deviation
from itertools import combinations
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Optional
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from bigram_frequencies import bigrams, bigram_frequencies_array


def load_filtered_data(filtered_users_data_file):
    """
    Load filtered data.

    Parameters:
    - filtered_users_data_file: path to the inpute CSV file of filtered experiment data

    Returns:
    - bigram_data: DataFrame
    """
    print(f"Loading data from {filtered_users_data_file}...")
    bigram_data = pd.read_csv(filtered_users_data_file)

    return bigram_data

# Functions to analyze bigram typing times + choice

def analyze_typing_times_slider_values(bigram_data, output_folder, 
                                       output_filename1='chosen_vs_unchosen_times.png', 
                                       output_filename2='typing_time_diff_vs_slider_value.png',
                                       output_filename3='typing_time_diff_vs_slider_value_histograms.png',
                                       max_time=2000):
    """
    Analyze and report typing times in filtered data.
    """    
    chosen_time_col, unchosen_time_col = 'chosen_bigram_time', 'unchosen_bigram_time'
    valid_comparisons = bigram_data.dropna(subset=[chosen_time_col, unchosen_time_col, 'sliderValue'])
    
    print(f"Total rows: {len(bigram_data)}")
    print(f"Valid comparisons: {len(valid_comparisons)}")
    
    faster_chosen = (valid_comparisons[chosen_time_col] < valid_comparisons[unchosen_time_col])
    print(f"\nFaster bigram chosen: {faster_chosen.sum()} out of {len(valid_comparisons)} ({faster_chosen.mean()*100:.2f}%)")

    def run_statistical_test(data1, data2, test_name, test_func):
        statistic, p_value = test_func(data1, data2)
        print(f"\n{test_name}:")
        print(f"Test statistic: {statistic:.4f}, p-value: {p_value:.4f}")
        return statistic, p_value

    def generate_interpretable_statements(results):
        """
        Generate human-readable interpretations from the test results.
        """
        statements = []

        # Chosen vs unchosen bigram typing times
        if results['chosen_unchosen_test']['p_value'] < 0.05:
            statements.append(
                f"Chosen bigrams are typed significantly faster than unchosen bigrams "
                f"(p = {results['chosen_unchosen_test']['p_value']:.4f})."
            )
        else:
            statements.append(
                "There is no significant difference in typing times between chosen and unchosen bigrams."
            )

        # Correlation between typing times and slider values
        if results['typing_time_slider_correlation']['p_value'] < 0.05:
            statements.append(
                f"There is a statistically significant correlation between typing times and the absolute slider values "
                f"(correlation = {results['typing_time_slider_correlation']['correlation']:.4f}, "
                f"p = {results['typing_time_slider_correlation']['p_value']:.4f})."
            )
        else:
            statements.append(
                f"There is no statistically significant correlation between typing times and the absolute slider values "
                f"(p = {results['typing_time_slider_correlation']['p_value']:.4f})."
            )

        # Slider value bias
        if results['slider_bias_test']['p_value'] < 0.05:
            statements.append(
                f"The slider values show a significant bias away from zero "
                f"(p = {results['slider_bias_test']['p_value']:.4f})."
            )
        else:
            statements.append(
                f"There is no significant bias in the slider values toward the left (negative) or right (positive) "
                f"(p = {results['slider_bias_test']['p_value']:.4f})."
            )

        return "\n".join(statements)
    
    # 1. Chosen vs unchosen typing times
    chosen_unchosen_test = run_statistical_test(
        valid_comparisons[chosen_time_col], 
        valid_comparisons[unchosen_time_col],
        "Chosen vs. unchosen bigram typing times (Wilcoxon signed-rank test)",
        stats.wilcoxon
    )

    # 2. Correlation between typing times and slider values
    correlation_test = run_statistical_test(
        valid_comparisons[chosen_time_col], 
        valid_comparisons['sliderValue'].abs(),
        "Correlation: typing times vs absolute slider values (Spearman's rank)",
        stats.spearmanr
    )

    # 3. Slider value bias
    slider_bias_test = run_statistical_test(
        bigram_data['sliderValue'].dropna(),
        np.zeros(len(bigram_data['sliderValue'].dropna())),
        "Slider value bias (Wilcoxon signed-rank test)",
        stats.wilcoxon
    )

    # Plotting
    def create_plot(x, y, title, xlabel, ylabel, filename, xlim=None, ylim=None, plot_type='scatter'):
        plt.figure(figsize=(10, 6))
        if plot_type == 'scatter':
            plt.scatter(x, y, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.axvline(x=0, color='r', linestyle='--')
        elif plot_type == 'box':
            sns.boxplot(x=x, y=y)
        if xlim != None:
            plt.xlim(xlim[0], xlim[1])
            xlim_str = f' {xlim}'
        else:
            xlim_str = ' '
        if ylim != None:
            plt.ylim(ylim[0], ylim[1])
            ylim_str = f' {ylim}'
        else:
            ylim_str = ' '
        plt.title(title)
        plt.xlabel(xlabel + xlim_str)
        plt.ylabel(ylabel + ylim_str)
        plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {filename}")

    # Original plots
    create_plot(
        ['Chosen']*len(valid_comparisons) + ['Unchosen']*len(valid_comparisons),
        pd.concat([valid_comparisons[chosen_time_col], valid_comparisons[unchosen_time_col]]),
        'Typing Times for Chosen vs Unchosen Bigrams', 
        '', 
        'Typing Time (ms)',
        output_filename1, 
        None,
        [0, max_time],
        'box'
    )

    create_plot(
        valid_comparisons['sliderValue'],
        valid_comparisons[chosen_time_col] - valid_comparisons[unchosen_time_col],
        'Typing Time Difference vs. Slider Value',
        'Slider Value', 
        'Typing Time Difference (Chosen - Unchosen) in ms',
        output_filename2,
        None,
        [-max_time, max_time]
    )

    # Histogram plot
    plt.figure(figsize=(15, 10))
    
    # Create bins for slider values
    slider_ranges = [(-100, -60), (-60, -20), (-20, 20), (20, 60), (60, 100)]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (low, high) in enumerate(slider_ranges):
        mask = (valid_comparisons['sliderValue'] >= low) & (valid_comparisons['sliderValue'] < high)
        data_subset = valid_comparisons[mask]
        
        if len(data_subset) > 0:
            # Plot chosen times with max cutoff
            chosen_times = data_subset[chosen_time_col].clip(upper=max_time)
            axes[i].hist(chosen_times, bins=30, alpha=0.5, 
                        label='Chosen', color='blue')  # Removed density=True
            
            # Plot unchosen times with max cutoff
            unchosen_times = data_subset[unchosen_time_col].clip(upper=max_time)
            axes[i].hist(unchosen_times, bins=30, alpha=0.5, 
                        label='Unchosen', color='red')  # Removed density=True
            
            axes[i].set_title(f'Slider Values [{low}, {high})')
            axes[i].set_xlabel('Typing Time (ms)')
            axes[i].set_ylabel('Frequency')
            axes[i].set_xlim(0, max_time)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Remove the extra subplot
    axes[-1].remove()
    
    plt.suptitle('Distribution of Typing Times by Slider Value Range', y=1.02)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, output_filename3), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nTyping times histogram saved to: {output_filename3}")


    results = {
        'total_rows': len(bigram_data),
        'valid_comparisons': len(valid_comparisons),
        'faster_chosen_count': faster_chosen.sum(),
        'chosen_unchosen_test': dict(zip(['statistic', 'p_value'], chosen_unchosen_test)),
        'typing_time_slider_correlation': dict(zip(['correlation', 'p_value'], correlation_test)),
        'slider_bias_test': dict(zip(['statistic', 'p_value'], slider_bias_test))
    }

    print(generate_interpretable_statements(results))

    return results

def analyze_user_typing_times(bigram_data):
    """
    Analyze bigram typing times within users for filtered data.
    """
    all_bigrams = set(bigram_data['chosen_bigram'].unique()) | set(bigram_data['unchosen_bigram'].unique())

    def compare_bigrams(user_data):
        bigram_times = {
            bigram: pd.concat([
                user_data[user_data['chosen_bigram'] == bigram]['chosen_bigram_time'],
                user_data[user_data['unchosen_bigram'] == bigram]['unchosen_bigram_time']
            ]).dropna()
            for bigram in all_bigrams
        }
        
        significant_pairs = []
        for bigram1, bigram2 in combinations(all_bigrams, 2):
            times1, times2 = bigram_times[bigram1], bigram_times[bigram2]
            if len(times1) > 0 and len(times2) > 0:
                statistic, p_value = stats.mannwhitneyu(times1, times2, alternative='two-sided')
                if p_value < 0.05:
                    faster_bigram = bigram1 if times1.median() < times2.median() else bigram2
                    slower_bigram = bigram2 if faster_bigram == bigram1 else bigram1
                    significant_pairs.append((faster_bigram, slower_bigram, p_value))
        return significant_pairs

    user_significant_pairs = {user_id: compare_bigrams(user_data) 
                              for user_id, user_data in bigram_data.groupby('user_id')}
    
    significant_pairs_count = {user_id: len(pairs) for user_id, pairs in user_significant_pairs.items()}
    total_significant_differences = sum(significant_pairs_count.values())
    users_with_differences = sum(count > 0 for count in significant_pairs_count.values())

    # Calculate total possible comparisons
    num_users = bigram_data['user_id'].nunique()
    comparisons_per_user = len(list(combinations(all_bigrams, 2)))
    total_possible_comparisons = num_users * comparisons_per_user

    print(f"Total bigrams compared: {len(all_bigrams)}")
    print(f"Users with significant differences: {users_with_differences} out of {num_users}")
    print(f"Total significant differences: {total_significant_differences} of {total_possible_comparisons}")

    return {
        'user_significant_pairs': user_significant_pairs,
        'significant_pairs_count': significant_pairs_count,
        'total_significant_differences': total_significant_differences,
        'total_possible_comparisons': total_possible_comparisons,
        'users_with_differences': users_with_differences
    }

def plot_typing_times(bigram_data, output_freq_time_plots_folder, output_filename='bigram_times_barplot.png'):
    """
    Generate and save bar plot for median bigram typing times with horizontal x-axis labels for filtered data.
    """
    # Prepare data
    plot_data = pd.concat([
        bigram_data[['chosen_bigram', 'chosen_bigram_time']].rename(columns={'chosen_bigram': 'bigram', 'chosen_bigram_time': 'time'}),
        bigram_data[['unchosen_bigram', 'unchosen_bigram_time']].rename(columns={'unchosen_bigram': 'bigram', 'unchosen_bigram_time': 'time'})
    ])

    # Calculate median times and MAD for each bigram
    grouped_data = plot_data.groupby('bigram')['time']
    median_times = grouped_data.median().sort_values()
    mad_times = grouped_data.apply(lambda x: median_abs_deviation(x, nan_policy='omit')).reindex(median_times.index)

    # Create plot
    plt.figure(figsize=(20, 10))
    bars = plt.bar(range(len(median_times)), median_times.values, yerr=mad_times.values, capsize=5)

    plt.title('Typing times for each bigram: median (MAD)', fontsize=16)
    plt.xlabel('Bigram', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.xticks(range(len(median_times)), median_times.index, rotation=0, ha='center', fontsize=8)

    plt.xlim(-0.5, len(median_times) - 0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(output_time_freq_plots_folder, output_filename), dpi=300, bbox_inches='tight')
    print(f"\nMedian bigram typing times bar plot saved to: {output_filename}")
    plt.close()

def plot_chosen_vs_unchosen_times(bigram_data, output_time_freq_plots_folder, 
                                  output_filename='chosen_vs_unchosen_times_scatter.png'):
    """
    Plot chosen vs. unchosen typing times with MAD error bars using existing processed_data.
    
    Parameters:
    - processed_data: Dictionary containing processed dataframes from process_data
    - output_time_freq_plots_folder: String path to the folder where plots should be saved
    - output_filename: String filename of the scatter plot
    """
    # Scatter plot for chosen vs unchosen typing times
    plt.figure(figsize=(10, 8))
    
    # Calculate median times for each bigram pair (chosen and unchosen)
    scatter_data = bigram_data.groupby('bigram_pair').agg(
        chosen_median=('chosen_bigram_time', 'median'),
        unchosen_median=('unchosen_bigram_time', 'median')
    ).reset_index()

    sns.scatterplot(x='chosen_median', y='unchosen_median', data=scatter_data, alpha=0.7)
    
    max_val = max(scatter_data['chosen_median'].max(), scatter_data['unchosen_median'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)  # Add diagonal line
    
    plt.title('Chosen vs Unchosen Bigram Median Typing Times')
    plt.xlabel('Chosen Bigram Median Time (ms)')
    plt.ylabel('Unchosen Bigram Median Time (ms)')
    
    correlation = scatter_data['chosen_median'].corr(scatter_data['unchosen_median'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_time_freq_plots_folder, output_filename), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Scatter plot saved to: {output_filename}")

# Functions to analyze bigram typing times + frequency

def plot_frequency_timing_relationship(
    bigram_data: pd.DataFrame,
    bigrams: list,
    bigram_frequencies_array: np.ndarray,
    output_path: str,
    n_groups: int = 4
) -> Dict[str, Any]:
    """
    Analyze and plot relationship between bigram frequency and typing time distributions.
    Creates three plots:
    1. Distribution plot showing median times with error bars and sample sizes
    2. Median bigram typing times
    3. Minimum (fastest) bigram typing times
    """
    try:
        print("bigram_data head:\n", bigram_data.head())
        print("bigrams[:5]:", bigrams[:5])
        print("bigram_frequencies_array[:5]:", bigram_frequencies_array[:5])
        print("Shape of input data:", bigram_data.shape)
        print("Sample of bigrams:", bigrams[:5])
        print("Sample of frequencies:", bigram_frequencies_array[:5])
        
        # Create dictionary mapping bigrams to their frequencies
        bigram_frequencies = {
            b: freq for b, freq in zip(bigrams, bigram_frequencies_array)
        }
        print("Created bigram frequencies dictionary")
        
        # Create DataFrames for chosen and unchosen bigrams
        chosen_df = bigram_data[['chosen_bigram', 'chosen_bigram_time']].copy()
        unchosen_df = bigram_data[['unchosen_bigram', 'unchosen_bigram_time']].copy()
        
        print("Created separate dataframes")
        print("Chosen shape:", chosen_df.shape)
        print("Unchosen shape:", unchosen_df.shape)
        
        # Rename columns to match
        chosen_df.columns = ['bigram', 'timing']
        unchosen_df.columns = ['bigram', 'timing']
        
        # Combine data
        df = pd.concat([chosen_df, unchosen_df], axis=0)
        print("Combined data shape:", df.shape)

        # Create long format dataframe directly
        timing_data = pd.concat([
            pd.DataFrame({
                'bigram': bigram_data['chosen_bigram'],
                'timing': bigram_data['chosen_bigram_time']
            }),
            pd.DataFrame({
                'bigram': bigram_data['unchosen_bigram'], 
                'timing': bigram_data['unchosen_bigram_time']
            })
        ])

        # Calculate timing statistics for each unique bigram
        aggregated_timings = timing_data.groupby('bigram').agg({
            'timing': ['median', 'mean', 'std', 'count', 'min']
        }).reset_index()

        # Flatten column names
        aggregated_timings.columns = ['bigram', 'median_timing', 'mean_timing', 
                                    'std_timing', 'n_occurrences', 'min_timing']

        # Add frequencies and create frequency groups
        aggregated_timings['frequency'] = aggregated_timings['bigram'].map(bigram_frequencies)
        aggregated_timings = aggregated_timings.dropna(subset=['frequency'])
        aggregated_timings['freq_group'] = pd.qcut(
            aggregated_timings['frequency'], 
            n_groups, 
            labels=[f"Group {i+1}" for i in range(n_groups)]
        )

        # Calculate correlations
        correlations = {
            'median': stats.spearmanr(aggregated_timings['frequency'], 
                                      aggregated_timings['median_timing']),
            'mean': stats.spearmanr(aggregated_timings['frequency'], 
                                    aggregated_timings['mean_timing']),
            'min': stats.spearmanr(aggregated_timings['frequency'],
                                   aggregated_timings['min_timing'])
        }  

        #--------------------------
        # Plot 1: Distribution plot
        #--------------------------
        plt.figure(figsize=(10, 6))
        plt.semilogx()  # Use log scale for frequency axis
        scatter = plt.scatter(aggregated_timings['frequency'], 
                            aggregated_timings['median_timing'],
                            alpha=0.5,
                            s=aggregated_timings['n_occurrences'] / 10)
        
        plt.errorbar(aggregated_timings['frequency'], 
                    aggregated_timings['median_timing'],
                    yerr=aggregated_timings['std_timing'],
                    fmt='none',
                    alpha=0.2)

        # Need to modify regression visualization for log scale
        x_log = np.log10(aggregated_timings['frequency'].values).reshape(-1, 1)
        reg_log = LinearRegression().fit(x_log, aggregated_timings['median_timing'].values)
        r2_log = reg_log.score(x_log, aggregated_timings['median_timing'].values)

        # Generate points for regression line
        x_range = np.logspace(
            np.log10(aggregated_timings['frequency'].min()),
            np.log10(aggregated_timings['frequency'].max()),
            2
        )
        plt.plot(x_range, 
                reg_log.predict(np.log10(x_range).reshape(-1, 1)),
                color='red',
                label=f'R² = {r2_log:.3f}')

        # Add legend for sample sizes
        legend_elements = [plt.scatter([],[], s=n/10, 
                                     label=f'n={n} samples', 
                                     alpha=0.5)
                           for n in [10, 50, 100, 500, 1000]]
        plt.legend(handles=legend_elements, 
                  title="Number of timing samples",
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left')

        plt.xlabel('Bigram Frequency')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Distribution of Typing Times vs. Frequency')

        plt.text(0.05, 0.95, 
                f"Correlation: {correlations['median'][0]:.3f}\n"
                f"p-value: {correlations['median'][1]:.3e}",
                transform=plt.gca().transAxes,
                verticalalignment='top')

        dist_plot_path = output_path.replace('.png', '_distribution.png')
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        #----------------------
        # Plot 2: Minimum times
        #----------------------
        plt.figure(figsize=(10, 6))
        plt.semilogx()  # Use log scale for frequency axis
        
        # Scatter plot
        plt.scatter(aggregated_timings['frequency'], 
                   aggregated_timings['min_timing'],
                   alpha=0.8)

        # Add bigram labels to each point
        for _, row in aggregated_timings.iterrows():
            plt.annotate(row['bigram'], 
                        (row['frequency'], row['min_timing']),
                        xytext=(3,3),  # 3 points offset
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7)

        # Add regression line for minimum times with log scale
        x_min_log = np.log10(aggregated_timings['frequency'].values).reshape(-1, 1)
        reg_min_log = LinearRegression().fit(
            x_min_log,
            aggregated_timings['min_timing'].values
        )
        r2_min_log = reg_min_log.score(
            x_min_log,
            aggregated_timings['min_timing'].values
        )
        
        # Generate points for regression line
        x_range = np.logspace(
            np.log10(aggregated_timings['frequency'].min()),
            np.log10(aggregated_timings['frequency'].max()),
            100
        )
        
        plt.plot(x_range,
                reg_min_log.predict(np.log10(x_range).reshape(-1, 1)),
                color='red',
                label=f'R² = {r2_min_log:.3f}')

        plt.xlabel('Bigram Frequency (log scale)')
        plt.ylabel('Minimum Typing Time (ms)')
        plt.title('Fastest Times vs. Frequency')
        
        plt.text(0.05, 0.95, 
                f"Correlation: {correlations['min'][0]:.3f}\n"
                f"p-value: {correlations['min'][1]:.3e}",
                transform=plt.gca().transAxes,
                verticalalignment='top')
        
        plt.legend()
        plt.tight_layout()

        min_plot_path = output_path.replace('.png', '_minimum.png')
        plt.savefig(min_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        #---------------------
        # Plot 3: Median times
        #---------------------
        plt.figure(figsize=(10, 6))
        plt.semilogx()  # Use log scale for frequency axis
        
        # Scatter plot
        plt.scatter(aggregated_timings['frequency'], 
                   aggregated_timings['median_timing'],
                   alpha=0.8)

        # Add bigram labels to each point
        for _, row in aggregated_timings.iterrows():
            plt.annotate(row['bigram'], 
                        (row['frequency'], row['median_timing']),
                        xytext=(3,3),  # 3 points offset
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7)

        # Add regression line for median times with log scale
        x_med_log = np.log10(aggregated_timings['frequency'].values).reshape(-1, 1)
        reg_med_log = LinearRegression().fit(
            x_med_log,
            aggregated_timings['median_timing'].values
        )
        r2_med_log = reg_med_log.score(
            x_med_log,
            aggregated_timings['median_timing'].values
        )
        
        # Generate points for regression line
        x_range = np.logspace(
            np.log10(aggregated_timings['frequency'].min()),
            np.log10(aggregated_timings['frequency'].max()),
            100
        )
        
        plt.plot(x_range,
                reg_med_log.predict(np.log10(x_range).reshape(-1, 1)),
                color='red',
                label=f'R² = {r2_med_log:.3f}')

        plt.xlabel('Bigram Frequency (log scale)')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Median Times vs. Frequency')
        
        plt.text(0.05, 0.95, 
                f"Correlation: {correlations['median'][0]:.3f}\n"
                f"p-value: {correlations['median'][1]:.3e}",
                transform=plt.gca().transAxes,
                verticalalignment='top')
        
        plt.legend()
        plt.tight_layout()

        med_plot_path = output_path.replace('.png', '_median.png')
        plt.savefig(med_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        #--------------
        # Perform ANOVA
        #--------------
        groups = [group_data['median_timing'].values 
                 for _, group_data in aggregated_timings.groupby('freq_group', observed=True)]
        f_stat, anova_p = stats.f_oneway(*groups)

        # Post-hoc analysis if ANOVA is significant
        post_hoc_results = None
        if anova_p < 0.05 and len(groups) > 2:
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                post_hoc = pairwise_tukeyhsd(
                    aggregated_timings['median_timing'],
                    aggregated_timings['freq_group']
                )
                post_hoc_results = pd.DataFrame(
                    data=post_hoc._results_table.data[1:],
                    columns=post_hoc._results_table.data[0]
                )
            except Exception as e:
                print(f"Could not perform post-hoc analysis: {str(e)}")

        #----------------
        # Prepare results
        #----------------
        results = {
            'correlation_median': correlations['median'][0],
            'correlation_median_p': correlations['median'][1],
            'correlation_mean': correlations['mean'][0],
            'correlation_mean_p': correlations['mean'][1],
            'correlation_min': correlations['min'][0],
            'correlation_min_p': correlations['min'][1],
            'r2': r2_log,
            'r2_min': r2_min_log,
            'n_unique_bigrams': len(aggregated_timings),
            'total_occurrences': aggregated_timings['n_occurrences'].sum(),
            'anova_f_stat': float(f_stat),
            'anova_p_value': float(anova_p),
            'post_hoc': post_hoc_results,
            'group_stats': {
                str(group): {
                    'freq_range': (
                        float(group_data['frequency'].min()),
                        float(group_data['frequency'].max())
                    ),
                    'median_timing': float(group_data['median_timing'].mean()),
                    'mean_timing': float(group_data['mean_timing'].mean()),
                    'min_timing': float(group_data['min_timing'].min()),
                    'timing_std': float(group_data['std_timing'].mean()),
                    'n_bigrams': len(group_data),
                    'total_occurrences': int(group_data['n_occurrences'].sum())
                }
                for group, group_data in aggregated_timings.groupby('freq_group', observed=True)
            }
        }

        return results

    except Exception as e:
        print(f"Error in frequency timing analysis: {str(e)}")
        return {'error': str(e)}
  
def plot_timing_by_frequency_groups(
    aggregated_data: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Create visualizations of timing differences between frequency groups.
    """
    try:
        # Create boxplot of median timings by frequency group
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=aggregated_data, x='freq_group', y='median_timing')
        plt.xlabel('Frequency Group')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Distribution of Median Typing Times by Frequency Group')     
        plt.savefig(os.path.join(output_dir, 'timing_by_group_box.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create violin plot for more detailed distribution view
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=aggregated_data, x='freq_group', y='median_timing')
        plt.xlabel('Frequency Group')
        plt.ylabel('Median Typing Time (ms)')
        plt.title('Distribution of Median Typing Times by Frequency Group')
        
        plt.savefig(os.path.join(output_dir, 'timing_by_group_violin.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating group plots: {str(e)}")

def analyze_timing_groups(
    bigram_data: pd.DataFrame,
    bigram_frequencies: Dict[str, float],
    n_groups: int = 4
) -> Dict[str, Any]:
    """
    Analyze timing differences between frequency groups.
    """
    try:
        timing_data = []
        for _, row in bigram_data.iterrows():
            # Add chosen bigram timing
            timing_data.append({
                'bigram': row['chosen_bigram'],
                'timing': row['chosen_bigram_time']
            })
            # Add unchosen bigram timing
            timing_data.append({
                'bigram': row['unchosen_bigram'],
                'timing': row['unchosen_bigram_time']
            })
        
        # Create DataFrame and aggregate timings by bigram
        df = pd.DataFrame(timing_data)
        aggregated_timings = df.groupby('bigram').agg({
            'timing': ['median', 'mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        aggregated_timings.columns = ['bigram', 'median_timing', 'mean_timing', 
                                    'std_timing', 'n_occurrences']
        
        # Add frequencies for bigrams that have frequency data
        aggregated_timings['frequency'] = aggregated_timings['bigram'].map(bigram_frequencies)
        
        # Remove bigrams without frequency data
        aggregated_timings = aggregated_timings.dropna(subset=['frequency'])
        
        # Create frequency groups
        aggregated_timings['freq_group'] = pd.qcut(
            aggregated_timings['frequency'], 
            n_groups, 
            labels=False
        )
        
        # Calculate group statistics
        group_stats = {}
        for group in range(n_groups):
            group_data = aggregated_timings[aggregated_timings['freq_group'] == group]
            if len(group_data) > 0:
                group_stats[group] = {
                    'freq_range': (
                        float(group_data['frequency'].min()),
                        float(group_data['frequency'].max())
                    ),
                    'mean_timing': float(group_data['median_timing'].mean()),
                    'std_timing': float(group_data['median_timing'].std()),
                    'n_bigrams': len(group_data),
                    'total_occurrences': int(group_data['n_occurrences'].sum())
                }
        
        # Perform ANOVA on median timings if we have enough groups
        if len(group_stats) >= 2:
            groups = [group['median_timing'].values for _, group in 
                     aggregated_timings.groupby('freq_group', observed=True) if len(group) > 0]
            f_stat, p_value = stats.f_oneway(*groups)
        else:
            f_stat, p_value = None, None
        
        return {
            'group_stats': group_stats,
            'f_stat': float(f_stat) if f_stat is not None else None,
            'p_value': float(p_value) if p_value is not None else None,
            'n_unique_bigrams': len(aggregated_timings),
            'total_occurrences': int(aggregated_timings['n_occurrences'].sum()),
            'aggregated_data': aggregated_timings
        }
        
    except Exception as e:
        print(f"Error in timing group analysis: {str(e)}")
        return {
            'error': str(e),
            'trace': str(e.__traceback__)
        }
           
def save_timing_analysis(
    timing_results: Dict[str, Any],
    group_comparison_results: Optional[Dict[str, Any]],
    output_path: str
) -> None:
    """Save timing analysis results to file."""
    try:
        with open(output_path, 'w') as f:
            f.write("Bigram Timing Analysis Results\n")
            f.write("============================\n\n")
            
            # Write correlation results
            f.write("Overall Correlation Analysis:\n")
            f.write("--------------------------\n")
            f.write(f"Median correlation: {timing_results['correlation_median']:.3f}")
            f.write(f" (p = {timing_results['correlation_median_p']:.3e})\n")
            f.write(f"Mean correlation: {timing_results['correlation_mean']:.3f}")
            f.write(f" (p = {timing_results['correlation_mean_p']:.3e})\n")
            f.write(f"R-squared: {timing_results['r2']:.3f}\n")
            f.write(f"Number of unique bigrams: {timing_results['n_unique_bigrams']}\n")
            f.write(f"Total timing instances: {timing_results['total_occurrences']}\n\n")
            
            # Write ANOVA results
            f.write("ANOVA Results:\n")
            f.write("-------------\n")
            f.write(f"F-statistic: {timing_results['anova_f_stat']:.3f}\n")
            f.write(f"p-value: {timing_results['anova_p_value']:.3e}\n\n")
            
            # Write group statistics
            f.write("Frequency Group Analysis:\n")
            f.write("----------------------\n")
            for group, stats in timing_results['group_stats'].items():
                f.write(f"\n{group}:\n")
                f.write(f"  Frequency range: {stats['freq_range']}\n")
                f.write(f"  Median typing time: {stats['median_timing']:.3f} ms\n")
                f.write(f"  Mean typing time: {stats['mean_timing']:.3f} ms\n")
                f.write(f"  Timing std dev: {stats['timing_std']:.3f} ms\n")
                f.write(f"  Number of unique bigrams: {stats['n_bigrams']}\n")
                f.write(f"  Total timing instances: {stats['total_occurrences']}\n")
            
            # Write post-hoc results if available
            if timing_results['post_hoc'] is not None:
                f.write("\nPost-hoc Analysis (Tukey HSD):\n")
                f.write("---------------------------\n")
                f.write(timing_results['post_hoc'].to_string())
                f.write("\n")
                
    except Exception as e:
        print(f"Error saving timing analysis: {str(e)}")

# Functions to analyze bigram bigram choice in relation to differences in typing time / frequency

def analyze_time_frequency_differences(
    bigram_data: pd.DataFrame,
    bigram_frequencies: Dict[str, float],
    output_dir: str
) -> Dict[str, Any]:
    """
    Analyze how bigram frequency differences explain typing time differences,
    with both raw and normalized analyses.
    
    Parameters:
    -----------
    bigram_data : pd.DataFrame
        DataFrame containing chosen/unchosen bigram pairs with typing times
    bigram_frequencies : Dict[str, float]
        Dictionary mapping bigrams to their frequencies
    output_dir : str
        Directory to save output plots
        
    Returns:
    --------
    Dict containing analysis results
    """
    # Calculate differences for each bigram pair, per user
    diffs_data = []
    
    # First calculate median typing time per user for normalization
    user_medians = bigram_data.groupby('user_id').agg({
        'chosen_bigram_time': 'median',
        'unchosen_bigram_time': 'median'
    }).mean(axis=1)
    
    for user_id, user_data in bigram_data.groupby('user_id'):
        user_median_time = user_medians[user_id]
        
        for _, row in user_data.iterrows():
            chosen_freq = bigram_frequencies.get(row['chosen_bigram'])
            unchosen_freq = bigram_frequencies.get(row['unchosen_bigram'])
            
            if (chosen_freq and unchosen_freq and 
                pd.notnull(row['chosen_bigram_time']) and 
                pd.notnull(row['unchosen_bigram_time']) and
                pd.notnull(row['sliderValue'])):
                
                # Calculate time differences
                time_diff = row['chosen_bigram_time'] - row['unchosen_bigram_time']
                norm_time_diff = time_diff / user_median_time
                
                # Calculate frequency difference
                freq_diff = np.log10(chosen_freq) - np.log10(unchosen_freq)
                
                if pd.notnull(norm_time_diff) and pd.notnull(freq_diff):
                    diffs_data.append({
                        'time_diff_raw': time_diff,
                        'time_diff_norm': norm_time_diff,
                        'freq_diff': freq_diff,
                        'slider_value': row['sliderValue'],
                        'user_id': user_id,
                        'chosen_bigram': row['chosen_bigram'],
                        'unchosen_bigram': row['unchosen_bigram'],
                        'user_median_time': user_median_time
                    })
    
    diffs_df = pd.DataFrame(diffs_data)
    
    def analyze_and_plot(x, y, title, xlabel, ylabel, output_path):
        """Helper function for regression analysis and plotting"""
        X = x.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        correlation = stats.pearsonr(x, y)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, alpha=0.5, s=3)
        
        # Add regression line
        x_range = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_range, reg.predict(x_range.reshape(-1, 1)), 
                color='red', label=f'R² = {r2:.3f}')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        # Add correlation info
        plt.text(0.05, 0.95, 
                f"Correlation: {correlation[0]:.3f}\n"
                f"p-value: {correlation[1]:.3e}",
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return {
            'regression_coef': float(reg.coef_[0]),
            'regression_intercept': float(reg.intercept_),
            'r2': r2,
            'correlation': correlation[0],
            'correlation_p': correlation[1]
        }
    
    # Analyze and plot raw differences
    raw_results = analyze_and_plot(
        diffs_df['freq_diff'],
        diffs_df['time_diff_raw'],
        'Frequency Difference vs. Raw Time Difference',
        'Log Frequency Difference',
        'Time Difference (ms)',
        os.path.join(output_dir, 'freq_vs_time_raw.png')
    )
    
    # Analyze and plot normalized differences
    norm_results = analyze_and_plot(
        diffs_df['freq_diff'],
        diffs_df['time_diff_norm'],
        'Frequency Difference vs. Normalized Time Difference',
        'Log Frequency Difference',
        'Normalized Time Difference',
        os.path.join(output_dir, 'freq_vs_time_normalized.png')
    )
    
    # Calculate per-user correlations
    user_correlations_raw = []
    user_correlations_norm = []
    
    for user_id, user_data in diffs_df.groupby('user_id'):
        if len(user_data) >= 5:  # Only include users with enough data points
            raw_corr = stats.pearsonr(user_data['freq_diff'], 
                                    user_data['time_diff_raw'])[0]
            norm_corr = stats.pearsonr(user_data['freq_diff'], 
                                     user_data['time_diff_norm'])[0]
            user_correlations_raw.append(raw_corr)
            user_correlations_norm.append(norm_corr)

    # Plot distribution of user correlations
    plt.figure(figsize=(10, 6))

    # Calculate common min/max for consistent binning
    min_val = min(min(user_correlations_raw), min(user_correlations_norm))
    max_val = max(max(user_correlations_raw), max(user_correlations_norm))
    bins = np.linspace(min_val, max_val, 21)  # 20 bins

    # Plot histograms using density=True for proper scaling
    plt.hist(user_correlations_raw, bins=bins, alpha=0.5, 
            label='Raw', color='blue', density=True)
    plt.hist(user_correlations_norm, bins=bins, alpha=0.5, 
            label='Normalized', color='red', density=True)

    # Add summary statistics to plot
    plt.text(0.02, 0.98, 
            f"Raw mean: {np.mean(user_correlations_raw):.3f}\n"
            f"Raw std: {np.std(user_correlations_raw):.3f}\n"
            f"Norm mean: {np.mean(user_correlations_norm):.3f}\n"
            f"Norm std: {np.std(user_correlations_norm):.3f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))

    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Density')  # Changed from 'Number of Users' since using density=True
    plt.title('Distribution of User-Level Correlations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'user_correlations_dist.png'), dpi=300)
    plt.close()

    # Prepare results dictionary
    results = {
        'raw_results': raw_results,
        'normalized_results': norm_results,
        'user_level_results': {
            'raw': {
                'mean_correlation': np.mean(user_correlations_raw),
                'std_correlation': np.std(user_correlations_raw)
            },
            'normalized': {
                'mean_correlation': np.mean(user_correlations_norm),
                'std_correlation': np.std(user_correlations_norm)
            },
            'n_users': len(user_correlations_raw)
        },
        'diffs_df': diffs_df,
        'n_comparisons': len(diffs_df)
    }
    
    # Save detailed results to file
    with open(os.path.join(output_dir, 'time_frequency_analysis.txt'), 'w') as f:
        f.write("Time-Frequency Difference Analysis Results\n")
        f.write("=======================================\n\n")
        
        f.write("Raw Analysis:\n")
        f.write(f"Correlation: {raw_results['correlation']:.3f}\n")
        f.write(f"P-value: {raw_results['correlation_p']:.3e}\n")
        f.write(f"R-squared: {raw_results['r2']:.3f}\n")
        f.write(f"Regression coefficient: {raw_results['regression_coef']:.3f}\n\n")
        
        f.write("Normalized Analysis:\n")
        f.write(f"Correlation: {norm_results['correlation']:.3f}\n")
        f.write(f"P-value: {norm_results['correlation_p']:.3e}\n")
        f.write(f"R-squared: {norm_results['r2']:.3f}\n")
        f.write(f"Regression coefficient: {norm_results['regression_coef']:.3f}\n\n")
        
        f.write("User-Level Analysis:\n")
        f.write("Raw correlations:\n")
        f.write(f"  Mean: {results['user_level_results']['raw']['mean_correlation']:.3f}\n")
        f.write(f"  Std: {results['user_level_results']['raw']['std_correlation']:.3f}\n")
        f.write("Normalized correlations:\n")
        f.write(f"  Mean: {results['user_level_results']['normalized']['mean_correlation']:.3f}\n")
        f.write(f"  Std: {results['user_level_results']['normalized']['std_correlation']:.3f}\n")
        f.write(f"Number of users: {results['user_level_results']['n_users']}\n")
    
    return results

def analyze_choice_variance(
    bigram_data: pd.DataFrame,
    bigram_frequencies: Dict[str, float]
) -> Dict[str, Any]:
    """
    Analyze how typing time differences and frequency differences explain bigram choices.
    
    Parameters:
    -----------
    bigram_data : pd.DataFrame
        DataFrame containing chosen/unchosen bigram pairs with typing times
    bigram_frequencies : Dict[str, float]
        Dictionary mapping bigrams to their frequencies
    
    Returns:
    --------
    Dict containing analysis results
    """
    # Prepare data, dropping any rows with NaN values
    valid_data = []
    
    for _, row in bigram_data.iterrows():
        chosen_freq = bigram_frequencies.get(row['chosen_bigram'])
        unchosen_freq = bigram_frequencies.get(row['unchosen_bigram'])
        
        if (chosen_freq is not None and unchosen_freq is not None and 
            pd.notnull(row['chosen_bigram_time']) and 
            pd.notnull(row['unchosen_bigram_time']) and
            pd.notnull(row['sliderValue'])):
            
            # Store the original comparison
            valid_data.append({
                'chosen_bigram': row['chosen_bigram'],
                'unchosen_bigram': row['unchosen_bigram'],
                'chosen_time': row['chosen_bigram_time'],
                'unchosen_time': row['unchosen_bigram_time'],
                'time_diff': row['chosen_bigram_time'] - row['unchosen_bigram_time'],
                'freq_diff': np.log10(chosen_freq) - np.log10(unchosen_freq),
                'slider_value': row['sliderValue'],
                'user_id': row['user_id'],
                'choice': 1  # 1 for chosen bigram
            })
            
            # Add the reverse comparison
            valid_data.append({
                'chosen_bigram': row['unchosen_bigram'],
                'unchosen_bigram': row['chosen_bigram'],
                'chosen_time': row['unchosen_bigram_time'],
                'unchosen_time': row['chosen_bigram_time'],
                'time_diff': row['unchosen_bigram_time'] - row['chosen_bigram_time'],
                'freq_diff': np.log10(unchosen_freq) - np.log10(chosen_freq),
                'slider_value': -row['sliderValue'],
                'user_id': row['user_id'],
                'choice': 0  # 0 for unchosen bigram
            })
    
    analysis_df = pd.DataFrame(valid_data)
    print(f"Number of valid comparisons: {len(analysis_df) // 2}")  # Divide by 2 because we doubled the data
    
    # Verify data
    print("\nData verification:")
    print(f"Time differences range: [{analysis_df['time_diff'].min():.1f}, {analysis_df['time_diff'].max():.1f}]")
    print(f"Frequency differences range: [{analysis_df['freq_diff'].min():.3f}, {analysis_df['freq_diff'].max():.3f}]")
    print(f"Choice values: {sorted(analysis_df['choice'].unique())}")
    
    # 1. Analyze typing time effect on choice
    time_model = LinearRegression()
    X_time = analysis_df['time_diff'].values.reshape(-1, 1)
    y = analysis_df['choice'].values
    time_model.fit(X_time, y)
    time_r2 = time_model.score(X_time, y)
    
    # Calculate correlation between time difference and choice
    time_correlation = stats.pointbiserialr(analysis_df['time_diff'], analysis_df['choice'])
    
    # 2. Analyze frequency effect after controlling for time
    # First, regress out timing effect from choice
    choice_residuals = y - time_model.predict(X_time)
    
    # Then, analyze frequency effect on residuals
    freq_model = LinearRegression()
    X_freq = analysis_df['freq_diff'].values.reshape(-1, 1)
    freq_model.fit(X_freq, choice_residuals)
    freq_r2 = freq_model.score(X_freq, choice_residuals)
    
    # Calculate partial correlation
    freq_correlation = stats.spearmanr(choice_residuals, analysis_df['freq_diff'])
    
    # 3. Combined model
    X_combined = np.column_stack([analysis_df['time_diff'], analysis_df['freq_diff']])
    combined_model = LinearRegression()
    combined_model.fit(X_combined, y)
    combined_r2 = combined_model.score(X_combined, y)
    
    return {
        'n_comparisons': len(analysis_df) // 2,
        'time_effect': {
            'r2': time_r2,
            'coefficient': float(time_model.coef_[0]),
            'correlation': time_correlation.correlation,
            'correlation_p': time_correlation.pvalue
        },
        'frequency_effect': {
            'r2': freq_r2,
            'coefficient': float(freq_model.coef_[0]),
            'correlation': freq_correlation.correlation,
            'correlation_p': freq_correlation.pvalue
        },
        'combined_effect': {
            'r2': combined_r2,
            'time_coefficient': float(combined_model.coef_[0]),
            'freq_coefficient': float(combined_model.coef_[1])
        },
        'analysis_df': analysis_df
    }

def plot_choice_analysis(
    analysis_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Create visualizations of how typing time and frequency differences relate to choices.
    
    Parameters:
    -----------
    analysis_df : pd.DataFrame
        DataFrame with time_diff, freq_diff, and choice columns
    output_dir : str
        Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Time difference effect
    plt.figure(figsize=(10, 6))
    time_bins = pd.qcut(analysis_df['time_diff'], 10, duplicates='drop')
    choice_by_time = analysis_df.groupby(time_bins, observed=True)['choice'].mean()
    
    plt.plot(range(len(choice_by_time)), choice_by_time.values, 'o-')
    plt.xticks(range(len(choice_by_time)), 
               [f'{x.left:.0f}' for x in choice_by_time.index],
               rotation=45)
    plt.xlabel('Typing Time Difference (ms)')
    plt.ylabel('Probability of Choice')
    plt.title('Effect of Typing Time Difference on Choice')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_effect.png'), dpi=300)
    plt.close()
    
    # 2. Frequency difference effect (after controlling for time)
    freq_bins = pd.qcut(analysis_df['freq_diff'], 10, duplicates='drop')
    choice_by_freq = analysis_df.groupby(freq_bins, observed=True)['choice'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(choice_by_freq)), choice_by_freq.values, 'o-')
    plt.xticks(range(len(choice_by_freq)), 
               [f'{x.left:.2f}' for x in choice_by_freq.index],
               rotation=45)
    plt.xlabel('Log Frequency Difference')
    plt.ylabel('Probability of Choice')
    plt.title('Effect of Frequency Difference on Choice')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frequency_effect.png'), dpi=300)
    plt.close()
    
    # 3. Combined effects heatmap
    plt.figure(figsize=(12, 8))
    time_bins_2d = pd.qcut(analysis_df['time_diff'], 8, duplicates='drop')
    freq_bins_2d = pd.qcut(analysis_df['freq_diff'], 8, duplicates='drop')
    
    choice_matrix = analysis_df.groupby([time_bins_2d, freq_bins_2d], observed=True)['choice'].mean().unstack()
    
    sns.heatmap(choice_matrix, cmap='RdYlBu_r', center=0.5,
                xticklabels=[f'{x:.2f}' for x in choice_matrix.columns.categories.mid],
                yticklabels=[f'{x:.0f}' for x in choice_matrix.index.categories.mid])
    plt.xlabel('Log Frequency Difference')
    plt.ylabel('Typing Time Difference (ms)')
    plt.title('Combined Effects of Time and Frequency on Choice Probability')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_effects.png'), dpi=300)
    plt.close()
    
    # Print summary statistics for each bin
    print("\nTime difference bins:")
    for i, (bin_name, group) in enumerate(analysis_df.groupby(time_bins, observed=True)):
        print(f"Bin {i+1}: {bin_name.left:.0f} to {bin_name.right:.0f} ms")
        print(f"  Choice probability: {group['choice'].mean():.3f}")
        print(f"  Number of samples: {len(group)}")
    
    print("\nFrequency difference bins:")
    for i, (bin_name, group) in enumerate(analysis_df.groupby(freq_bins, observed=True)):
        print(f"Bin {i+1}: {bin_name.left:.2f} to {bin_name.right:.2f}")
        print(f"  Choice probability: {group['choice'].mean():.3f}")
        print(f"  Number of samples: {len(group)}")

# Enhanced analysis

def enhance_choice_analysis(bigram_data, bigram_frequencies):
    """
    Enhanced analysis with corrected correlation calculations.
    """
    try:
        # Prepare data
        analysis_data = []
        for _, row in bigram_data.iterrows():
            chosen_freq = bigram_frequencies.get(row['chosen_bigram'])
            unchosen_freq = bigram_frequencies.get(row['unchosen_bigram'])
            
            if (chosen_freq and unchosen_freq and 
                pd.notnull(row['chosen_bigram_time']) and 
                pd.notnull(row['unchosen_bigram_time'])):
                
                # Calculate raw times and frequencies
                chosen_time = row['chosen_bigram_time']
                unchosen_time = row['unchosen_bigram_time']
                
                # Calculate differences
                time_diff = chosen_time - unchosen_time
                freq_diff = np.log10(chosen_freq) - np.log10(unchosen_freq)
                confidence = abs(row['sliderValue']) / 100
                
                analysis_data.append({
                    'user_id': row['user_id'],
                    'chosen_time': chosen_time,
                    'unchosen_time': unchosen_time,
                    'chosen_freq': chosen_freq,
                    'unchosen_freq': unchosen_freq,
                    'time_diff': time_diff,
                    'freq_diff': freq_diff,
                    'abs_time_diff': abs(time_diff),
                    'abs_freq_diff': abs(freq_diff),
                    'confidence': confidence,
                    'chosen_bigram': row['chosen_bigram'],
                    'unchosen_bigram': row['unchosen_bigram']
                })
        
        df = pd.DataFrame(analysis_data)
        
        # Analyze effects per participant
        participant_effects = {}
        for user_id, user_data in df.groupby('user_id'):
            if len(user_data) >= 5:
                try:
                    # Correlate absolute differences with confidence
                    time_corr = stats.spearmanr(
                        user_data['abs_time_diff'],
                        user_data['confidence']
                    )[0] if user_data['abs_time_diff'].std() > 0 and user_data['confidence'].std() > 0 else np.nan
                    
                    freq_corr = stats.spearmanr(
                        user_data['abs_freq_diff'],
                        user_data['confidence']
                    )[0] if user_data['abs_freq_diff'].std() > 0 and user_data['confidence'].std() > 0 else np.nan
                    
                    # Also calculate directional effects
                    time_direction = np.mean(np.sign(user_data['time_diff']))
                    freq_direction = np.mean(np.sign(user_data['freq_diff']))
                    
                    participant_effects[user_id] = {
                        'n_trials': len(user_data),
                        'time_effect': time_corr,
                        'freq_effect': freq_corr,
                        'time_direction': time_direction,
                        'freq_direction': freq_direction,
                        'mean_confidence': user_data['confidence'].mean(),
                        'time_diff_std': user_data['time_diff'].std(),
                        'freq_diff_std': user_data['freq_diff'].std()
                    }
                except:
                    continue
        
        # Calculate overall statistics
        def safe_correlation(x, y):
            if x.std() == 0 or y.std() == 0:
                return np.nan
            try:
                return stats.spearmanr(x, y, nan_policy='omit')[0]
            except:
                return np.nan
        
        # Calculate correlations between raw times and frequencies
        avg_time_freq_corr = []
        for user_id, user_data in df.groupby('user_id'):
            if len(user_data) >= 5:
                chosen_corr = safe_correlation(
                    user_data['chosen_time'],
                    np.log10(user_data['chosen_freq'])
                )
                unchosen_corr = safe_correlation(
                    user_data['unchosen_time'],
                    np.log10(user_data['unchosen_freq'])
                )
                if not np.isnan(chosen_corr) and not np.isnan(unchosen_corr):
                    avg_time_freq_corr.append((chosen_corr + unchosen_corr) / 2)
        
        overall_time_freq_corr = np.mean(avg_time_freq_corr) if avg_time_freq_corr else np.nan
        
        overall_stats = {
            'time_effect': {
                'correlation': safe_correlation(df['abs_time_diff'], df['confidence']),
                'mean': df['time_diff'].mean(),
                'std': df['time_diff'].std()
            },
            'freq_effect': {
                'correlation': safe_correlation(df['abs_freq_diff'], df['confidence']),
                'mean': df['freq_diff'].mean(),
                'std': df['freq_diff'].std()
            },
            'confidence': {
                'mean': df['confidence'].mean(),
                'std': df['confidence'].std()
            }
        }
        
        # Calculate choice consistency
        def calculate_consistency(df):
            signs_match = (df['time_diff'] * df['freq_diff'] > 0)
            return pd.Series({'consistency': signs_match.mean()})
        
        consistency_by_user = df.groupby('user_id', include_groups=True).apply(calculate_consistency)

        consistency = consistency_by_user['consistency'].mean()
        
        results = {
            'participant_effects': participant_effects,
            'overall_stats': overall_stats,
            'time_freq_correlation': overall_time_freq_corr,
            'choice_consistency': consistency,
            'n_observations': len(df),
            'n_participants': len(df['user_id'].unique())
        }
        
        # Print and save results
        output = []
        output.append("Enhanced Analysis Results")
        output.append("=======================")
        output.append(f"\nAnalyzed {results['n_observations']} observations from {results['n_participants']} participants\n")
        
        output.append("Overall Effects:")
        output.append("Time-Confidence Correlation (absolute differences):")
        output.append(f"  Overall: {results['overall_stats']['time_effect']['correlation']:.3f}")
        
        output.append("\nFrequency-Confidence Correlation (absolute differences):")
        output.append(f"  Overall: {results['overall_stats']['freq_effect']['correlation']:.3f}")
        
        output.append("\nTyping Time - Frequency Correlation:")
        output.append(f"  Overall: {results['time_freq_correlation']:.3f}")
        
        output.append(f"\nChoice consistency: {results['choice_consistency']*100:.1f}%\n")
        
        effects_df = pd.DataFrame(results['participant_effects']).T
        valid_effects = effects_df.dropna(subset=['time_effect', 'freq_effect'])
        
        output.append("Participant-Level Statistics:")
        output.append("Time Effect:")
        output.append(f"  Mean correlation: {valid_effects['time_effect'].mean():.3f} (±{valid_effects['time_effect'].std():.3f})")
        output.append(f"  Mean direction: {valid_effects['time_direction'].mean():.3f}")
        
        output.append("\nFrequency Effect:")
        output.append(f"  Mean correlation: {valid_effects['freq_effect'].mean():.3f} (±{valid_effects['freq_effect'].std():.3f})")
        output.append(f"  Mean direction: {valid_effects['freq_direction'].mean():.3f}")
        
        output.append(f"\nMean confidence: {valid_effects['mean_confidence'].mean():.3f} (±{valid_effects['mean_confidence'].std():.3f})")
        
        # Print to console
        for line in output:
            print(line)
        
        # Save to file
        try:
            with open(output_path, 'w') as f:
                for line in output:
                    f.write(line + '\n')
            print(f"\nResults saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving results to file: {str(e)}")
        
        return results
    
    except Exception as e:
        print(f"Error in enhance_choice_analysis: {str(e)}")
        return None

def save_and_print_results(results, output_path):
    """
    Format, print, and save analysis results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
    output_path : str
        Path to save results file
    """
    # Create formatted output
    output = []
    output.append("Enhanced Analysis Results")
    output.append("=======================")
    output.append(f"\nAnalyzed {results['n_observations']} observations from {results['n_participants']} participants\n")
    
    output.append("Overall Effects:")
    output.append("Time-Confidence Correlation (absolute differences):")
    output.append(f"  Overall: {results['overall_stats']['time_effect']['correlation']:.3f}")
    
    output.append("\nFrequency-Confidence Correlation (absolute differences):")
    output.append(f"  Overall: {results['overall_stats']['freq_effect']['correlation']:.3f}")
    
    output.append("\nTyping Time - Frequency Correlation:")
    output.append(f"  Overall: {results['time_freq_correlation']:.3f}")
    
    output.append(f"\nChoice consistency: {results['choice_consistency']*100:.1f}%\n")
    
    effects_df = pd.DataFrame(results['participant_effects']).T
    valid_effects = effects_df.dropna(subset=['time_effect', 'freq_effect'])
    
    output.append("Participant-Level Statistics:")
    output.append("Time Effect:")
    output.append(f"  Mean correlation: {valid_effects['time_effect'].mean():.3f} (±{valid_effects['time_effect'].std():.3f})")
    output.append(f"  Mean direction: {valid_effects['time_direction'].mean():.3f}")
    
    output.append("\nFrequency Effect:")
    output.append(f"  Mean correlation: {valid_effects['freq_effect'].mean():.3f} (±{valid_effects['freq_effect'].std():.3f})")
    output.append(f"  Mean direction: {valid_effects['freq_direction'].mean():.3f}")
    
    output.append(f"\nMean confidence: {valid_effects['mean_confidence'].mean():.3f} (±{valid_effects['mean_confidence'].std():.3f})")
    
    # Print to console
    for line in output:
        print(line)
    
    # Save to file
    try:
        with open(output_path, 'w') as f:
            for line in output:
                f.write(line + '\n')
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving results to file: {str(e)}")

def format_enhanced_results(results):
    """Format the enhanced analysis results for printing."""
    if not results:
        return "Analysis failed to produce results."
    
    output = []
    output.append(f"Analysis of {results['n_observations']} observations from {results['n_participants']} participants\n")
    
    # Overall statistics
    output.append("Overall Effects:")
    output.append(f"Time effect correlation: {results['overall_stats']['time_effect']['correlation']:.3f}")
    output.append(f"Frequency effect correlation: {results['overall_stats']['freq_effect']['correlation']:.3f}")
    output.append(f"Time-frequency correlation: {results['time_freq_correlation']:.3f}")
    output.append(f"Choice consistency: {results['choice_consistency']*100:.1f}%\n")
    
    # Participant-level summaries
    effects = pd.DataFrame(results['participant_effects']).T
    output.append("Participant-Level Statistics:")
    output.append(f"Mean time effect: {effects['time_effect'].mean():.3f} (±{effects['time_effect'].std():.3f})")
    output.append(f"Mean frequency effect: {effects['freq_effect'].mean():.3f} (±{effects['freq_effect'].std():.3f})")
    output.append(f"Mean confidence: {effects['mean_confidence'].mean():.3f} (±{effects['mean_confidence'].std():.3f})")
    
    return "\n".join(output)
        
def analyze_timing_patterns(bigram_data):
    """
    Analyze timing patterns including practice effects and variability.
    """
    try:
        # Add trial number within participant
        bigram_data = bigram_data.copy()
        bigram_data['trial_number'] = bigram_data.groupby('user_id').cumcount() + 1
        
        participant_stats = {}
        practice_effects = {}
        
        for user_id, user_data in bigram_data.groupby('user_id'):
            # Calculate timing variability safely
            chosen_times = user_data['chosen_bigram_time'].dropna()
            unchosen_times = user_data['unchosen_bigram_time'].dropna()
            
            if len(chosen_times) > 0 and len(unchosen_times) > 0:
                participant_stats[user_id] = {
                    'chosen_timing_cv': np.std(chosen_times) / np.mean(chosen_times) if np.mean(chosen_times) > 0 else np.nan,
                    'unchosen_timing_cv': np.std(unchosen_times) / np.mean(unchosen_times) if np.mean(unchosen_times) > 0 else np.nan,
                    'median_chosen_time': np.median(chosen_times),
                    'median_unchosen_time': np.median(unchosen_times)
                }
                
                # Analyze practice effects if enough trials
                if len(chosen_times) >= 10:
                    early_trials = chosen_times.iloc[:5].mean()
                    late_trials = chosen_times.iloc[-5:].mean()
                    
                    if early_trials > 0:  # Avoid division by zero
                        practice_effects[user_id] = {
                            'early_mean': early_trials,
                            'late_mean': late_trials,
                            'improvement': (early_trials - late_trials) / early_trials * 100
                        }
        
        # Calculate overall practice effect
        trial_means = bigram_data.groupby('trial_number')[
            ['chosen_bigram_time', 'unchosen_bigram_time']
        ].mean()
        
        # Safe correlation calculation
        if len(trial_means) > 2:
            practice_correlation = stats.spearmanr(
                trial_means.index,
                trial_means['chosen_bigram_time'].values,
                nan_policy='omit'
            )
        else:
            practice_correlation = (np.nan, np.nan)
        
        return {
            'participant_stats': participant_stats,
            'practice_effects': practice_effects,
            'overall_practice_correlation': practice_correlation,
            'trial_means': trial_means
        }
    
    except Exception as e:
        print(f"Error in analyze_timing_patterns: {str(e)}")
        return None

def check_statistical_assumptions(bigram_data):
    """
    Check statistical assumptions for the analysis.
    """
    try:
        assumption_checks = {}
        
        # Check normality of typing time differences
        time_diffs = (bigram_data['chosen_bigram_time'] - 
                     bigram_data['unchosen_bigram_time']).dropna()
        
        if len(time_diffs) > 8:  # Minimum sample size for normality test
            normality_test = stats.normaltest(time_diffs)
            assumption_checks['normality'] = {
                'statistic': normality_test.statistic,
                'p_value': normality_test.pvalue
            }
        else:
            assumption_checks['normality'] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'note': 'Insufficient sample size for normality test'
            }
        
        # Check for temporal independence
        if 'trial_number' not in bigram_data.columns:
            bigram_data = bigram_data.copy()
            bigram_data['trial_number'] = bigram_data.groupby('user_id').cumcount() + 1
            
        time_diffs = bigram_data['chosen_bigram_time'] - bigram_data['unchosen_bigram_time']
        if len(time_diffs) > 2:
            autocorr = stats.pearsonr(
                time_diffs.iloc[:-1],
                time_diffs.iloc[1:]
            )
            assumption_checks['independence'] = {
                'autocorrelation': autocorr[0],
                'p_value': autocorr[1]
            }
        else:
            assumption_checks['independence'] = {
                'autocorrelation': np.nan,
                'p_value': np.nan,
                'note': 'Insufficient sample size for independence test'
            }
        
        return assumption_checks
    
    except Exception as e:
        print(f"Error in check_statistical_assumptions: {str(e)}")
        return None

# Speed as choice proxy analysis

def analyze_speed_as_choice_proxy(bigram_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how well typing speed predicts bigram choice under different conditions.
    """
    # Calculate speed difference and choice alignment
    analysis_data = []
    for _, row in bigram_data.iterrows():
        speed_diff = row['chosen_bigram_time'] - row['unchosen_bigram_time']
        avg_time = (row['chosen_bigram_time'] + row['unchosen_bigram_time']) / 2
        norm_speed_diff = speed_diff / avg_time if avg_time > 0 else np.nan
        
        analysis_data.append({
            'user_id': row['user_id'],
            'chosen_bigram': row['chosen_bigram'],
            'unchosen_bigram': row['unchosen_bigram'],
            'speed_diff': speed_diff,
            'norm_speed_diff': norm_speed_diff,
            'speed_predicts_choice': speed_diff < 0,
            'confidence': abs(row['sliderValue']),
            'bigram_pair': tuple(sorted([row['chosen_bigram'], row['unchosen_bigram']]))
        })
    
    df = pd.DataFrame(analysis_data)
    
    # Overall prediction accuracy
    overall_accuracy = df['speed_predicts_choice'].mean()
    
    # Accuracy by speed difference magnitude
    df['speed_diff_quintile'] = pd.qcut(df['norm_speed_diff'].abs(), 5, labels=False)
    accuracy_by_diff = df.groupby('speed_diff_quintile', observed=True)['speed_predicts_choice'].agg(['mean', 'count', 'std'])
    accuracy_by_diff.columns = ['accuracy', 'count', 'std']
    
    # Confidence analysis
    try:
        df['confidence_level'] = pd.qcut(df['confidence'], 4, 
                                       labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                                       duplicates='drop')
    except ValueError:
        confidence_bounds = [
            df['confidence'].min(),
            df['confidence'].quantile(0.33),
            df['confidence'].quantile(0.67),
            df['confidence'].max()
        ]
        df['confidence_level'] = pd.cut(df['confidence'], 
                                      bins=confidence_bounds,
                                      labels=['Low', 'Medium', 'High'],
                                      include_lowest=True)
    
    accuracy_by_confidence = df.groupby('confidence_level', observed=True)['speed_predicts_choice'].agg(['mean', 'count', 'std'])
    accuracy_by_confidence.columns = ['accuracy', 'count', 'std']

    # Analysis by bigram pair
    pair_stats = df.groupby('bigram_pair', observed=True).agg({
        'speed_predicts_choice': ['mean', 'count', 'std'],
        'norm_speed_diff': ['mean', 'std']
    }).round(3)
    
    min_occurrences_mask = pair_stats[('speed_predicts_choice', 'count')] >= 10
    best_pairs = pair_stats[min_occurrences_mask].nlargest(5, ('speed_predicts_choice', 'mean'))
    worst_pairs = pair_stats[min_occurrences_mask].nsmallest(5, ('speed_predicts_choice', 'mean'))

    # Per-user analysis
    user_means = df.groupby('user_id')['speed_predicts_choice'].mean()
    user_counts = df.groupby('user_id')['speed_predicts_choice'].count()

    results = {
        'overall_accuracy': overall_accuracy,
        'accuracy_by_speed_diff': accuracy_by_diff,
        'best_predictable_pairs': best_pairs,
        'worst_predictable_pairs': worst_pairs,
        'user_stats': {
            'mean_accuracy': user_means.mean(),
            'std_accuracy': user_means.std(),
            'accuracies': user_means.tolist(),
            'trial_counts': user_counts.tolist()
        },
        'accuracy_by_confidence': accuracy_by_confidence,
        'n_total_trials': len(df),
        'n_users': len(df['user_id'].unique()),
        'n_bigram_pairs': len(df['bigram_pair'].unique())
    }
    
    return results

def visualize_speed_choice_relationship(results: Dict[str, Any], output_dir: str) -> None:
    """Create visualizations of speed-choice relationship analysis."""
    
    # 1. Accuracy by speed difference magnitude
    plt.figure(figsize=(10, 6))
    acc_by_diff = results['accuracy_by_speed_diff']
    plt.errorbar(range(len(acc_by_diff)), 
                acc_by_diff['accuracy'],
                yerr=acc_by_diff['std'],
                fmt='o-')
    plt.xlabel('Speed Difference Quintile')
    plt.ylabel('Prediction Accuracy')
    plt.title('Speed Prediction Accuracy by Magnitude of Speed Difference')
    plt.savefig(os.path.join(output_dir, 'speed_accuracy_by_magnitude.png'))
    plt.close()
    
    # 2. Accuracy by confidence
    plt.figure(figsize=(10, 6))
    acc_by_conf = results['accuracy_by_confidence']
    plt.errorbar(range(len(acc_by_conf)),
                acc_by_conf['accuracy'],
                yerr=acc_by_conf['std'],
                fmt='o-')
    plt.xlabel('Confidence Level')
    plt.ylabel('Prediction Accuracy')
    plt.title('Speed Prediction Accuracy by Confidence Level')
    plt.savefig(os.path.join(output_dir, 'speed_accuracy_by_confidence.png'))
    plt.close()
    
    # 3. Distribution of user accuracies
    plt.figure(figsize=(10, 6))
    plt.hist(results['user_stats']['accuracies'], bins=20)
    plt.xlabel('Prediction Accuracy')
    plt.ylabel('Number of Users')
    plt.title('Distribution of Per-User Prediction Accuracies')
    plt.savefig(os.path.join(output_dir, 'user_accuracy_distribution.png'))
    plt.close()

def report_proxy_analysis(results: Dict[str, Any]) -> str:
    """Generate readable report of analysis results."""
    report = [
        "Speed as Choice Proxy Analysis",
        "===========================\n",
        f"Number of participants: {results['n_users']}",
        f"Total trials analyzed: {results['n_total_trials']}",
        f"Unique bigram pairs: {results['n_bigram_pairs']}\n",
        f"Overall accuracy: {results['overall_accuracy']*100:.1f}%\n",
        "Accuracy by Speed Difference Magnitude:",
        "------------------------------------"
    ]
    
    acc_by_diff = results['accuracy_by_speed_diff']
    for quintile, row in acc_by_diff.iterrows():
        report.append(f"Quintile {quintile+1}: {row['accuracy']*100:.1f}% "
                     f"(n={row['count']}, ±{row['std']*100:.1f}%)")
    
    report.extend([
        "\nAccuracy by Confidence Level:",
        "--------------------------"
    ])
    acc_by_conf = results['accuracy_by_confidence']
    for level, row in acc_by_conf.iterrows():
        report.append(f"{level}: {row['accuracy']*100:.1f}% "
                     f"(n={row['count']}, ±{row['std']*100:.1f}%)")
    
    report.extend([
        "\nUser-Level Statistics:",
        "-------------------",
        f"Mean user accuracy: {results['user_stats']['mean_accuracy']*100:.1f}%",
        f"Standard deviation: {results['user_stats']['std_accuracy']*100:.1f}%"
    ])
    
    return "\n".join(report)


########################################################################################
# Main execution
########################################################################################
if __name__ == "__main__":

    do_analyze_times_frequencies = False
    do_analyze_deltas = False
    do_enhanced_analysis = False
    do_speed_proxy_analysis = True

    ###########
    # Load data
    ###########
    # Set the paths for input and output
    #input_folder = '/Users/arno.klein/Documents/osf/processed_output_all4studies_406participants'
    input_folder = '/Users/arno.klein/Documents/osf/processed_output_all4studies_303of406participants_0improbable'
    
    filtered_users_data_file = os.path.join(input_folder, 'tables/processed_data/filtered_bigram_data.csv')
    output_folder = input_folder
    #output_tables_folder = os.path.join(output_folder, 'time_tables')
    output_plots_folder = os.path.join(output_folder, 'plots')
    output_time_freq_plots_folder = os.path.join(output_plots_folder, 'bigram_typing_time_and_frequency_analysis')
    output_choice_plots_folder = os.path.join(output_plots_folder, 'bigram_choice_analysis')
    output_speed_choice_folder = os.path.join(output_plots_folder, 'speed_choice_analysis') 
    
    #os.makedirs(output_tables_folder, exist_ok=True)
    os.makedirs(output_plots_folder, exist_ok=True)
    os.makedirs(output_time_freq_plots_folder, exist_ok=True)
    os.makedirs(output_choice_plots_folder, exist_ok=True)
    os.makedirs(output_speed_choice_folder, exist_ok=True)

    # Load filtered user data (generated by process_data.py)
    bigram_data = load_filtered_data(filtered_users_data_file)
    print(bigram_data.columns)

    # Create bigram frequency dictionary
    bigram_frequencies = dict(zip(bigrams, bigram_frequencies_array))

    #############################################
    # Analyze bigram typing times and frequencies 
    #############################################
    """
    This analysis of bigram typing times was intended to determine whether there is a correlation 
    between typing speed and typing comfort, in case we could use speed as a proxy for comfort
    in future work.

    This analysis of the relationship between bigram frequencies 
    and typing times handles data where:
    1. Input contains pairs of bigrams (chosen vs. unchosen)
    2. Each bigram pair has an associated average typing time
    3. Each bigram has an associated frequency from a corpus
    The analysis ensures each unique bigram is counted only once,
    with its timing averaged across all instances where it appears
    (whether as chosen or unchosen in the original pairs).
    """
    if do_analyze_times_frequencies:
        print("\n____ Typing Time and Slider Statistics ____\n")
        typing_time_stats = analyze_typing_times_slider_values(bigram_data, output_choice_plots_folder, 
                                output_filename1='chosen_vs_unchosen_times.png', 
                                output_filename2='typing_time_diff_vs_slider_value.png',
                                output_filename3='typing_time_diff_vs_slider_value_histograms.png')

        # Analyze within-user bigram typing times and relationships
        print("\n____ Within-User Bigram Typing Time Analysis ____\n")
        within_user_stats = analyze_user_typing_times(bigram_data)

        plot_typing_times(bigram_data, output_time_freq_plots_folder, 
                          output_filename='bigram_times_barplot.png')

        print("\n____ Chosen vs. Unchosen Bigram Typing Time Analysis ____\n")
        plot_chosen_vs_unchosen_times(bigram_data, output_choice_plots_folder, 
                          output_filename='chosen_vs_unchosen_times_scatter_regression.png')

        print("\n____ Bigram Typing Time and Frequency Analysis ____\n")
        timing_results = plot_frequency_timing_relationship(
            bigram_data=bigram_data,
            bigrams=bigrams,
            bigram_frequencies_array=bigram_frequencies_array,
            output_path=os.path.join(output_time_freq_plots_folder, 'frequency_and_timing.png'),
            n_groups=5
        )

        if 'error' in timing_results:
            print(f"Frequency-timing analysis failed: {timing_results['error']}")
        else:
            # Create group visualizations
            if 'group_analysis' in timing_results:
                plot_timing_by_frequency_groups(
                    bigram_data=bigram_data,
                    bigram_frequencies_array=bigram_frequencies_array,
                    group_results=timing_results['group_analysis'],
                    output_dir=output_time_freq_plots_folder
                )

            # Save analysis results
            save_timing_analysis(
                timing_results=timing_results,
                group_comparison_results=timing_results.get('group_analysis'),
                output_path=os.path.join(output_time_freq_plots_folder, 'frequency_timing_analysis_results.txt')

            )
            
            print("Frequency/timing analysis completed")
            print(f"Results:")
            print(f"  Median correlation: {timing_results['correlation_median']:.3f} "
                        f"(p = {timing_results['correlation_median_p']:.3e})")
            print(f"  Mean correlation: {timing_results['correlation_mean']:.3f} "
                        f"(p = {timing_results['correlation_mean_p']:.3e})")
            print(f"  R-squared: {timing_results['r2']:.3f}")
            print(f"  ANOVA F-stat: {timing_results['anova_f_stat']:.3f} "
                        f"(p = {timing_results['anova_p_value']:.3e})")
            print(f"  Number of unique bigrams: {timing_results['n_unique_bigrams']}")
            print(f"  Total timing instances: {timing_results['total_occurrences']}")

    ########################################
    # Analyze bigram frequency/timing deltas 
    ########################################
    """
    This section analyzes the relationship between bigram choice 
    and difference in bigram frequency and in bigram typing time.
    """     
    if do_analyze_deltas:   

        #------------------------------------------------------------------
        # Analyze how frequency differences explain typing time differences
        #------------------------------------------------------------------
        print("\n____ Analysis of Time/Frequency Differences ____\n")

        # Analyze time-frequency differences
        results = analyze_time_frequency_differences(
            bigram_data, 
            bigram_frequencies,
            output_time_freq_plots_folder
        )
        
        print("\nRaw Analysis Results:")
        print(f"Correlation: {results['raw_results']['correlation']:.3f} "
            f"(p = {results['raw_results']['correlation_p']:.3e})")
        print(f"Variance explained (R²): {results['raw_results']['r2']:.3f}")
        
        print("\nNormalized Analysis Results:")
        print(f"Correlation: {results['normalized_results']['correlation']:.3f} "
            f"(p = {results['normalized_results']['correlation_p']:.3e})")
        print(f"Variance explained (R²): {results['normalized_results']['r2']:.3f}")
        
        print("\nUser-Level Analysis:")
        print("Raw correlations:")
        print(f"  Mean: {results['user_level_results']['raw']['mean_correlation']:.3f} "
            f"(±{results['user_level_results']['raw']['std_correlation']:.3f})")
        print("Normalized correlations:")
        print(f"  Mean: {results['user_level_results']['normalized']['mean_correlation']:.3f} "
            f"(±{results['user_level_results']['normalized']['std_correlation']:.3f})")
        print(f"Number of users with sufficient data: {results['user_level_results']['n_users']}")

    ###################
    # Enhanced Analysis 
    ###################
    if do_enhanced_analysis:   

        print("\n____ Enhanced Analyses ____\n")
    
        try:
            # 1. Enhanced choice analysis
            print("Running enhanced choice analysis...")
            enhanced_choice_results = enhance_choice_analysis(bigram_data, bigram_frequencies)
            
            if enhanced_choice_results:
                # Results are now printed within enhance_choice_analysis()
                pass
                
            # 2. Timing pattern analysis
            print("\nAnalyzing timing patterns...")
            timing_pattern_results = analyze_timing_patterns(bigram_data)
            
            if timing_pattern_results and timing_pattern_results['practice_effects']:
                improvements = [effect['improvement'] 
                            for effect in timing_pattern_results['practice_effects'].values()
                            if not np.isnan(effect['improvement'])]
                
                if improvements:
                    mean_improvement = np.mean(improvements)
                    correlation = timing_pattern_results['overall_practice_correlation']
                    
                    print(f"\nTiming Pattern Results:")
                    print(f"Mean improvement across participants: {mean_improvement:.1f}%")
                    print(f"Practice effect correlation: {correlation[0]:.3f} "
                        f"(p = {correlation[1]:.3e})")
            
            # 3. Statistical assumption checks
            print("\nChecking statistical assumptions...")
            assumption_results = check_statistical_assumptions(bigram_data)
            
            if assumption_results:
                print("\nStatistical Assumption Check Results:")
                if 'normality' in assumption_results:
                    norm_result = assumption_results['normality']
                    if 'note' in norm_result:
                        print(f"Normality test: {norm_result['note']}")
                    else:
                        print(f"Normality test: statistic = {norm_result['statistic']:.3f}, "
                            f"p = {norm_result['p_value']:.3e}")
                        
        except Exception as e:
            print(f"Error in enhanced analysis section: {str(e)}")


    ################################
    # Speed as choice proxy analysis
    ################################
    if do_speed_proxy_analysis:   

        print("\n____ Speed as Choice Proxy Analysis ____\n")
        
        # Run main analysis
        proxy_results = analyze_speed_as_choice_proxy(bigram_data)
        
        # Generate visualizations
        visualize_speed_choice_relationship(
            proxy_results, 
            output_speed_choice_folder
        )
        
        # Generate and save report
        report = report_proxy_analysis(proxy_results)
        report_path = os.path.join(output_speed_choice_folder, 'speed_choice_analysis_report.txt')
        
        print(report)  # Print to console
        
        # Save report to file
        with open(report_path, 'w') as f:
            f.write(report)
            
        print(f"\nDetailed analysis saved to: {report_path}")
        print("Visualizations saved to:", output_speed_choice_folder)
        
        # Print key findings
        print("\nKey findings:")
        print(f"- Overall prediction accuracy: {proxy_results['overall_accuracy']*100:.1f}%")
        print(f"- Mean user accuracy: {proxy_results['user_stats']['mean_accuracy']*100:.1f}% "
              f"(±{proxy_results['user_stats']['std_accuracy']*100:.1f}%)")
        
        if proxy_results['overall_accuracy'] > 0.7:
            print("\nSpeed appears to be a reasonably good proxy for choice overall")
            if proxy_results['user_stats']['std_accuracy'] > 0.1:
                print("However, there is substantial variation between users")
        else:
            print("\nSpeed alone may not be a reliable proxy for choice")
            print("Consider using speed in combination with other factors")
    

