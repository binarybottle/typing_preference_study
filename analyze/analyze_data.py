
""" Analyze experiment data -- See README.md """

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

def analyze_typing_times_slider_values(bigram_data, output_time_plots_folder, 
                                       output_filename1='chosen_vs_unchosen_times.png', 
                                       output_filename2='typing_time_diff_vs_slider_value.png',
                                       output_filename3='typing_time_diff_vs_slider_value_histograms.png',
                                       max_time=2000):
    """
    Analyze and report typing times in filtered data.
    """
    print("\n____ Typing Time and Slider Statistics ____\n")
    
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
        plt.savefig(os.path.join(output_time_plots_folder, filename), dpi=300, bbox_inches='tight')
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
    
    plt.savefig(os.path.join(output_time_plots_folder, output_filename3), dpi=300, bbox_inches='tight')
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

    print("\n____ Within-User Bigram Typing Time Analysis ____\n")
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

def plot_typing_times(bigram_data, output_time_plots_folder, output_filename='bigram_times_barplot.png'):
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

    plt.savefig(os.path.join(output_time_plots_folder, output_filename), dpi=300, bbox_inches='tight')
    print(f"\nMedian bigram typing times bar plot saved to: {output_filename}")
    plt.close()

def plot_chosen_vs_unchosen_times(bigram_data, output_time_plots_folder, 
                                  output_filename='chosen_vs_unchosen_times_scatter.png'):
    """
    Plot chosen vs. unchosen typing times with MAD error bars using existing processed_data.
    
    Parameters:
    - processed_data: Dictionary containing processed dataframes from process_data
    - output_time_plots_folder: String path to the folder where plots should be saved
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
    plt.savefig(os.path.join(output_time_plots_folder, output_filename), dpi=300, bbox_inches='tight')
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

########################################################################################
# Main execution
########################################################################################
if __name__ == "__main__":

    do_analyze_times = False
    do_analyze_frequencies = True

    ###########
    # Load data
    ###########
    # Set the paths for input and output
    input_folder = '/Users/arno.klein/Documents/osf/processed_output_all4studies_406participants'
    input_folder = '/Users/arno.klein/Documents/osf/processed_output_all4studies_303of406participants_0improbable'
    
    filtered_users_data_file = os.path.join(input_folder, 'tables/processed_data/filtered_bigram_data.csv')
    output_folder = input_folder
    #output_tables_folder = os.path.join(output_folder, 'time_tables')
    output_time_plots_folder = os.path.join(output_folder, 'plots', 'typing_time')
    output_freq_plots_folder = os.path.join(output_folder, 'plots', 'bigram_frequency')
    #os.makedirs(output_tables_folder, exist_ok=True)
    os.makedirs(output_time_plots_folder, exist_ok=True)
    os.makedirs(output_freq_plots_folder, exist_ok=True)

    # Load filtered user data (generated by process_data.py)
    bigram_data = load_filtered_data(filtered_users_data_file)
    print(bigram_data.columns)

    #############################
    # Analyze bigram typing times 
    #############################
    """
    This analysis of bigram typing times was intended to determine whether there is a correlation 
    between typing speed and typing comfort, in case we could use speed as a proxy for comfort
    in future work.
    """
    if do_analyze_times:
        typing_time_stats = analyze_typing_times_slider_values(bigram_data, output_time_plots_folder, 
                                output_filename1='chosen_vs_unchosen_times.png', 
                                output_filename2='typing_time_diff_vs_slider_value.png',
                                output_filename3='typing_time_diff_vs_slider_value_histograms.png')

        # Analyze within-user bigram typing times and relationships
        within_user_stats = analyze_user_typing_times(bigram_data)

        plot_typing_times(bigram_data, output_time_plots_folder, 
                          output_filename='bigram_times_barplot.png')

        plot_chosen_vs_unchosen_times(bigram_data, output_time_plots_folder, 
                          output_filename='chosen_vs_unchosen_times_scatter_regression.png')

    ##############################################
    # Analyze bigram frequency/timing relationship 
    ##############################################
    """
    This analysis of the relationship between bigram frequencies 
    and typing times handles data where:
    1. Input contains pairs of bigrams (chosen vs. unchosen)
    2. Each bigram pair has an associated average typing time
    3. Each bigram has an associated frequency from a corpus
    The analysis ensures each unique bigram is counted only once,
    with its timing averaged across all instances where it appears
    (whether as chosen or unchosen in the original pairs).
    """     
    if do_analyze_frequencies:   
        timing_results = plot_frequency_timing_relationship(
            bigram_data=bigram_data,
            bigrams=bigrams,
            bigram_frequencies_array=bigram_frequencies_array,
            output_path=os.path.join(output_freq_plots_folder, 'frequency_and_timing.png'),
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
                    output_dir=output_freq_plots_folder
                )

            # Save analysis results
            save_timing_analysis(
                timing_results=timing_results,
                group_comparison_results=timing_results.get('group_analysis'),
                output_path=os.path.join(output_freq_plots_folder, 'frequency_timing_analysis_results.txt')

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


