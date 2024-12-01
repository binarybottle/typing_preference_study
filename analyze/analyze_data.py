
""" Analyze experiment data -- See README.md """

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import median_abs_deviation
from itertools import combinations

from process_data import display_information


def load_filtered_data(filtered_users_data_file):
    """
    Load filtered data.

    Parameters:
    - filtered_users_data_file: path to the inpute CSV file of filtered experiment data

    Returns:
    - filtered_users_data: DataFrame
    """
    print(f"Loading data from {filtered_users_data_file}...")
    filtered_data = pd.read_csv(filtered_users_data_file)

    return filtered_users_data

def analyze_typing_times_slider_values(filtered_users_data, output_plots_folder, 
                                       output_filename1='chosen_vs_unchosen_times.png', 
                                       output_filename2='typing_time_diff_vs_slider_value.png'):
    """
    Analyze and report typing times in filtered data.
    """
    print("\n____ Typing Time and Slider Statistics ____\n")
    bigram_data = filtered_users_data['bigram_data']
    
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
    def create_plot(x, y, title, xlabel, ylabel, filename, plot_type='scatter'):
        plt.figure(figsize=(10, 6))
        if plot_type == 'scatter':
            plt.scatter(x, y, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.axvline(x=0, color='r', linestyle='--')
        elif plot_type == 'box':
            sns.boxplot(x=x, y=y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(output_plots_folder, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved to: {filename}")

    create_plot(
        ['Chosen']*len(valid_comparisons) + ['Unchosen']*len(valid_comparisons),
        pd.concat([valid_comparisons[chosen_time_col], valid_comparisons[unchosen_time_col]]),
        'Typing Times for Chosen vs Unchosen Bigrams', '', 'Typing Time (ms)',
        output_filename1, 'box'
    )

    create_plot(
        valid_comparisons['sliderValue'],
        valid_comparisons[chosen_time_col] - valid_comparisons[unchosen_time_col],
        'Typing Time Difference vs. Slider Value',
        'Slider Value', 'Typing Time Difference (Chosen - Unchosen) in ms',
        output_filename2
    )

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

def analyze_user_typing_times(filtered_users_data):
    """
    Analyze bigram typing times within users for filtered data.
    """
    bigram_data = filtered_users_data['bigram_data']
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

def plot_typing_times(filtered_users_data, output_plots_folder, output_filename='filtered_bigram_times_barplot.png'):
    """
    Generate and save bar plot for median bigram typing times with horizontal x-axis labels for filtered data.
    """
    bigram_data = filtered_users_data['bigram_data']

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
    plt.xticks(range(len(median_times)), median_times.index, rotation=0, ha='center', fontsize=12)

    plt.xlim(-0.5, len(median_times) - 0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(output_plots_folder, output_filename), dpi=300, bbox_inches='tight')
    print(f"\nMedian bigram typing times bar plot saved to: {output_filename}")
    plt.close()

def plot_chosen_vs_unchosen_times(filtered_users_data, output_plots_folder, 
                                  output_filename='chosen_vs_unchosen_times_scatter.png'):
    """
    Plot chosen vs. unchosen typing times with MAD error bars using existing processed_data.
    
    Parameters:
    - processed_data: Dictionary containing processed dataframes from process_data
    - output_plots_folder: String path to the folder where plots should be saved
    - output_filename: String filename of the scatter plot
    """
    bigram_data = filtered_users_data['bigram_data']  # Assuming 'bigram_data' key exists

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
    
    plt.title('Median Chosen vs Unchosen Bigram Typing Times')
    plt.xlabel('Median Chosen Bigram Time (ms)')
    plt.ylabel('Median Unchosen Bigram Time (ms)')
    
    correlation = scatter_data['chosen_median'].corr(scatter_data['unchosen_median'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, output_filename), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Scatter plot saved to: {output_filename}")

# Main execution
if __name__ == "__main__":

    ###########
    # Load data
    ###########
    # Set the paths for input and output
    input_folder = '/Users/arno.klein/Documents/osf/output_all4studies_406participants'
    #input_folder = '/Users/arno.klein/Documents/osf/output_all4studies_303of406participants_0improbable'
    filtered_users_data_file = os.path.join(input_folder, 'tables/filtered_bigram_data.csv')
    output_folder = input_folder
    output_tables_folder = os.path.join(output_folder, 'tables')
    output_plots_folder = os.path.join(output_folder, 'plots')

    filtered_users_data = load_filtered_data(filtered_users_data_file)

    #############################
    # Analyze bigram typing times 
    #############################
    """
    This analysis of bigram typing times was intended to determine whether there is a correlation 
    between typing speed and typing comfort, in case we could use speed as a proxy for comfort
    in future work.
    """
    typing_time_stats = analyze_typing_times_slider_values(filtered_users_data, output_plots_folder, 
                            output_filename1='filtered_chosen_vs_unchosen_times.png', 
                            output_filename2='filtered_typing_time_diff_vs_slider_value.png')

    # Analyze within-user bigram typing times and relationships
    within_user_stats = analyze_user_typing_times(filtered_users_data)

    plot_typing_times(filtered_users_data, output_plots_folder, 
                    output_filename='filtered_bigram_times_barplot.png')

    plot_chosen_vs_unchosen_times(filtered_users_data, output_plots_folder, 
                    output_filename='filtered_chosen_vs_unchosen_times_scatter_regression.png')

