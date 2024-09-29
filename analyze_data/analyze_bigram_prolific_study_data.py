
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import median_abs_deviation
from itertools import combinations
from collections import Counter

#######################
# Load, preprocess data
#######################

def load_and_preprocess_data(input_folder, output_tables_folder, verbose=False):
    """
    Load and preprocess (combine) data from multiple CSV files in a folder.

    Parameters:
    - input_folder: path to the folder containing the CSV files
    - output_tables_folder: path to the folder where the combined data will be saved

    Returns:
    - filtered_combined_df: DataFrame with combined data
    """
    print(f"Loading data from {input_folder}...")
    dataframes = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")
            df = pd.read_csv(os.path.join(input_folder, filename))
            
            # Extract user ID from filename (assuming format: experiment_data_USERID_*.csv)
            user_id = filename.split('_')[2]
            df['user_id'] = user_id
            df['filename'] = filename
            
            # Remove rows where 'trialId' contains 'intro-trial'
            df_filtered = df[~df['trialId'].str.contains("intro-trial", na=False)]
            if len(df_filtered) > 0:
                dataframes.append(df)
    
    # Combine the dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded and combined data from {len(dataframes)} files")

    # Filter out rows where 'trialId' contains 'intro-trial'
    filtered_combined_df = combined_df[~combined_df['trialId'].str.contains("intro-trial", na=False)]

    # Display information about the combined DataFrame
    if verbose:
        print(filtered_combined_df.info())
        print_headers = ['trialId', 'sliderValue', 'chosenBigram', 'unchosenBigram', 
                         'chosenBigramTime', 'unchosenBigramTime']
        display_information(filtered_combined_df, "original data", print_headers, nlines=30)

    # Save the combined DataFrame to a CSV file
    output_file = os.path.join(output_tables_folder, 'original_combined_data.csv')
    filtered_combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

    return filtered_combined_df

def load_easy_choice_pairs(file_path):
    """
    Load easy choice bigram pairs from a CSV file.

    Parameters:
    - file_path: String, path to the CSV file containing easy choice bigram pairs

    Returns:
    - easy_choice_pairs: List of tuples, each containing a pair of bigrams where one is highly improbable
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Ensure the CSV has the correct columns
        if 'good_choice' not in df.columns or 'bad_choice' not in df.columns:
            raise ValueError("CSV file must contain 'good_choice' and 'bad_choice' columns")
        
        # Convert DataFrame to list of tuples
        easy_choice_pairs = list(df[['good_choice', 'bad_choice']].itertuples(index=False, name=None))
        
        print(f"\nLoaded {len(easy_choice_pairs)} bigram pairs from {file_path} where one bigram in each pair is an easy choice.\n")
        return easy_choice_pairs
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error loading easy choice pairs: {str(e)}")
        return []

def load_bigram_pairs(file_path):
    """
    Load bigram pairs from a CSV file.

    Parameters:
    - file_path: String, path to the CSV file containing bigram pairs

    Returns:
    - num_bigram_pairs: Int, number of bigram pairs
    - bigram_pairs_df: List of tuples, each containing a pair of bigrams
    """
    try:
        # Read the CSV file
        bigram_pairs_df = pd.read_csv(file_path, header=None)
        num_bigram_pairs = len(bigram_pairs_df)
        print(f"\nNumber of bigram pairs presented to each participant: {num_bigram_pairs}.\n")   
        
        return num_bigram_pairs, bigram_pairs_df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error loading pairs: {str(e)}")
        return []
    
def process_bigram_data(data, output_tables_folder, verbose=False):
    """
    Process the bigram data from a DataFrame.

    Parameters:
    - data: DataFrame with combined bigram data
    - output_tables_folder: str, path to the folder where the processed data will be saved

    Returns:
    - bigram_data: DataFrame with processed bigram data
    """
    
    # First, create a standardized bigram pair representation
    data['std_bigram_pair'] = data.apply(lambda row: ', '.join(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
    
    # Group the data by user_id and standardized bigram pair
    grouped_data = data.groupby(['user_id', 'std_bigram_pair'])
    
    result_list = []
    
    for (user_id, std_bigram_pair), group in grouped_data:
        bigram1, bigram2 = std_bigram_pair.split(', ')
        
        # Check consistency
        chosen_bigrams = set(group['chosenBigram'])
        is_consistent = len(chosen_bigrams) == 1
        
        for _, row in group.iterrows():            
            result = pd.DataFrame({
                'user_id': [user_id],
                'trialId': [row['trialId']],
                'bigram_pair': [std_bigram_pair],
                'bigram1': [bigram1],
                'bigram2': [bigram2],
                'bigram1_time': [row['chosenBigramTime'] if row['chosenBigram'] == bigram1 else row['unchosenBigramTime']],
                'bigram2_time': [row['chosenBigramTime'] if row['chosenBigram'] == bigram2 else row['unchosenBigramTime']],
                'chosen_bigram': row['chosenBigram'],
                'unchosen_bigram': row['unchosenBigram'],
                'chosen_bigram_time': [row['chosenBigramTime']],
                'unchosen_bigram_time': [row['unchosenBigramTime']],
                'chosen_bigram_correct': [row['chosenBigramCorrect']],
                'unchosen_bigram_correct': [row['unchosenBigramCorrect']],
                'sliderValue': [row['sliderValue']],
                'text': [row['text']],
                'is_consistent': [is_consistent]
            })
            result_list.append(result)
    
    # Concatenate all the results into a single DataFrame
    bigram_data = pd.concat(result_list).reset_index(drop=True)
    
    # Sort the DataFrame
    bigram_data = bigram_data.sort_values(by=['user_id', 'trialId', 'bigram_pair']).reset_index(drop=True)
    
    # Display information about the bigram DataFrame
    if verbose:
        print_headers = ['trialId', 'bigram_pair', 'chosen_bigram', 'chosen_bigram_time', 'chosen_bigram_correct',  
                         'sliderValue', 'is_consistent']
        display_information(bigram_data, "bigram data", print_headers, nlines=30)
    
    # Save and return the bigram data
    bigram_data.to_csv(f"{output_tables_folder}/processed_bigram_data.csv", index=False)
    
    return bigram_data

def prepare_plot_data(bigram_data):
    """
    Prepare data for plotting chosen vs unchosen times.
    """
    # Include both chosen and unchosen bigrams
    chosen_data = bigram_data.groupby('chosen_bigram').agg({'chosen_bigram_time': 'median'}).reset_index()
    chosen_data.columns = ['bigram', 'time']
    chosen_data['type'] = 'chosen'

    unchosen_data = bigram_data.groupby('unchosen_bigram').agg({'unchosen_bigram_time': 'median'}).reset_index()
    unchosen_data.columns = ['bigram', 'time']
    unchosen_data['type'] = 'unchosen'

    plot_data = pd.concat([chosen_data, unchosen_data], ignore_index=True)
    
    #print(f"Number of unique bigrams: {plot_data['bigram'].nunique()}")
    return plot_data

def display_information(dframe, title, print_headers, nlines):
    """
    Display information about a DataFrame.

    Parameters:
    - dframe: DataFrame to display information about
    - title: name of the DataFrame
    - print_headers: list of headers to print
    - nlines: number of lines to print
    """
    print('')
    print(f"{title}:")
    #dframe.info()
    #print('')
    #print("Sample output:")
    
    with pd.option_context('display.max_rows', nlines):
        print(dframe[print_headers].iloc[:nlines])  # Display 'nlines' rows

    print('')

###############################
# Bigram Typing Time Statistics
###############################

def analyze_typing_times(bigram_data, output_plots_folder):
    """
    Analyze and report typing times in bigram data, including statistical tests for relationships
    between typing time and choice, and typing time and absolute slider value.
    Empty values in bigram times are ignored in calculations.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data

    Returns:
    - typing_time_stats: Dictionary containing typing time statistics and test results
    """
    print("\n____ Bigram Typing Time Statistics ____\n")

    # Compare bigram1 and bigram2 times, ignoring empty values
    valid_time_mask = bigram_data['bigram1_time'].notna() & bigram_data['bigram2_time'].notna()
    bigram_data.loc[valid_time_mask, 'faster_bigram'] = np.where(
        bigram_data.loc[valid_time_mask, 'bigram1_time'] < bigram_data.loc[valid_time_mask, 'bigram2_time'],
        bigram_data.loc[valid_time_mask, 'bigram1'],
        bigram_data.loc[valid_time_mask, 'bigram2']
    )

    # Statistical test for relationship between typing time and choice
    chosen_times = bigram_data['chosen_bigram_time'].dropna()
    unchosen_times = bigram_data['unchosen_bigram_time'].dropna()
    
    print(f"\nTotal rows: {len(bigram_data)}")
    print(f"Number of valid chosen times: {len(chosen_times)}")
    print(f"Number of valid unchosen times: {len(unchosen_times)}")

    """
    # Report on empty values
    empty_counts = bigram_data[['bigram1_time', 'bigram2_time', 'chosen_bigram_time', 'unchosen_bigram_time', 'sliderValue']].isna().sum()
    print("Number of empty values:")
    print(empty_counts)
    """

    # Check if the faster bigram is also the chosen bigram
    faster_chosen_mask = valid_time_mask & (bigram_data['faster_bigram'] == bigram_data['chosen_bigram'])
    number_faster_chosen = faster_chosen_mask.sum()
    percent_faster_chosen = (number_faster_chosen / valid_time_mask.sum()) * 100
    print(f"Number of times the faster bigram was chosen: {number_faster_chosen} of {valid_time_mask.sum()} comparisons ({percent_faster_chosen:.2f}%)")

    if len(chosen_times) > 0 and len(unchosen_times) > 0:
        # Mann-Whitney U test
        try:
            statistic, p_value = stats.mannwhitneyu(chosen_times, unchosen_times, alternative='two-sided')
            
            print("\nRelationship between typing time and choice:")
            print(f"Mann-Whitney U test statistic: {statistic}")
            print(f"p-value: {p_value}")
            
            if p_value < 0.05:
                print("There is a significant relationship between typing time and choice.")
                if chosen_times.median() < unchosen_times.median():
                    print("Chosen bigrams tend to have shorter typing times.")
                else:
                    print("Chosen bigrams tend to have longer typing times.")
            else:
                print("There is no significant relationship between typing time and choice.")
        except Exception as e:
            print(f"Error performing Mann-Whitney U test: {str(e)}")
    else:
        print("Insufficient data to perform Mann-Whitney U test.")

    # Statistical tests for relationship between typing time and absolute slider value
    typing_times = bigram_data['chosen_bigram_time'].dropna()
    abs_slider_values = bigram_data['sliderValue'].abs().dropna()
    
    # Use only rows where both typing time and slider value are valid
    valid_data = pd.DataFrame({'typing_time': typing_times, 'abs_slider_value': abs_slider_values}).dropna()

    print(f"\nNumber of valid pairs of typing times and slider values: {len(valid_data)}")

    if len(valid_data) > 0:
        # Spearman correlation
        try:
            correlation, p_value = stats.spearmanr(valid_data['typing_time'], valid_data['abs_slider_value'])
            
            print("\nRelationship between typing time and absolute slider value (Spearman correlation):")
            print(f"Spearman correlation coefficient: {correlation}")
            print(f"p-value: {p_value}")
            
            if p_value < 0.05:
                print("There is a significant monotonic relationship between typing time and absolute slider value.")
                if correlation > 0:
                    print("Longer typing times tend to be associated with higher absolute slider values.")
                else:
                    print("Longer typing times tend to be associated with lower absolute slider values.")
            else:
                print("There is no significant monotonic relationship between typing time and absolute slider value.")
        except Exception as e:
            print(f"Error performing Spearman correlation: {str(e)}")

        # Mann-Whitney U test
        try:
            median_slider = valid_data['abs_slider_value'].median()
            low_slider_times = valid_data[valid_data['abs_slider_value'] <= median_slider]['typing_time']
            high_slider_times = valid_data[valid_data['abs_slider_value'] > median_slider]['typing_time']
            
            statistic, p_value = stats.mannwhitneyu(low_slider_times, high_slider_times, alternative='two-sided')
            
            print("\nRelationship between typing time and absolute slider value (Mann-Whitney U test):")
            print(f"Mann-Whitney U test statistic: {statistic}")
            print(f"p-value: {p_value}")
            
            if p_value < 0.05:
                print("There is a significant difference in typing times between low and high slider values.")
                if low_slider_times.median() < high_slider_times.median():
                    print("Higher slider values tend to have longer typing times.")
                else:
                    print("Lower slider values tend to have longer typing times.")
            else:
                print("There is no significant difference in typing times between low and high slider values.")
        except Exception as e:
            print(f"Error performing Mann-Whitney U test: {str(e)}")

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_data['abs_slider_value'], valid_data['typing_time'], alpha=0.5)
        plt.xlabel('Absolute Slider Value')
        plt.ylabel('Typing Time (ms)')
        plt.title('Typing Time vs. Absolute Slider Value')
        plt.savefig(os.path.join(output_plots_folder, 'typing_time_vs_slider_value.png'), dpi=300, bbox_inches='tight')
        plt.close()
        #print("\nScatter plot of Typing Time vs. Absolute Slider Value saved as 'typing_time_vs_slider_value.png'")

    else:
        print("Insufficient data to perform statistical tests or create visualization.")

    # Return the statistics for further use if needed
    typing_time_stats = {
        'number_faster_chosen': number_faster_chosen,
        'percent_faster_chosen': percent_faster_chosen,
        'choice_test': {'statistic': statistic if 'statistic' in locals() else None, 
                        'p_value': p_value if 'p_value' in locals() else None},
        'slider_correlation': {'correlation': correlation if 'correlation' in locals() else None, 
                               'p_value': p_value if 'p_value' in locals() else None}
    }
    return typing_time_stats

def plot_median_bigram_times(bigram_data, output_plots_folder):
    """
    Generate and save bar plot for median bigram typing times with horizontal x-axis labels.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    """
    # Prepare data
    plot_data = bigram_data.melt(id_vars=['user_id', 'trialId', 'bigram_pair', 'chosen_bigram', 'unchosen_bigram'],
                                 value_vars=['chosen_bigram_time', 'unchosen_bigram_time'],
                                 var_name='bigram_type', value_name='time')
    
    plot_data['bigram'] = np.where(plot_data['bigram_type'] == 'chosen_bigram_time',
                                   plot_data['chosen_bigram'],
                                   plot_data['unchosen_bigram'])

    # Calculate median times and MAD for each bigram
    median_times = plot_data.groupby('bigram')['time'].median().sort_values()
    mad_times = plot_data.groupby('bigram')['time'].apply(lambda x: median_abs_deviation(x, nan_policy='omit')).reindex(median_times.index)

    # Create a bar plot of median bigram times with MAD whiskers
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Create x-coordinates for the bars
    x = np.arange(len(median_times))
    
    # Plot bars
    bars = ax.bar(x, median_times.values, yerr=mad_times.values, capsize=5)
    ax.set_title('Typing times for each bigram: median (MAD)', fontsize=16)
    ax.set_xlabel('Bigram', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(median_times.index, rotation=0, ha='center', fontsize=12)

    # Adjust the x-axis to center labels under bars
    ax.set_xlim(-0.5, len(x) - 0.5)

    # Add padding to the bottom of the plot for labels
    plt.subplots_adjust(bottom=0.2)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    plt.savefig(os.path.join(output_plots_folder, 'bigram_median_times_barplot_with_mad.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #print(f"bigram_median_times_barplot_with_mad plot saved in {output_plots_folder}")

###################################
# Improbable Bigram Choice Analysis
###################################

def analyze_easy_choices(bigram_data, easy_choice_pairs, threshold=1):
    """
    Analyze bigram choices to detect improbable selections, comparing both typing times and slider values.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - easy_choice_pairs: List of tuples, each containing a pair of bigrams where one is highly improbable
    - threshold: Float, the threshold for flagging a user (default 1)

    Returns:
    - suspicious_users: DataFrame containing users with suspiciously high rates of improbable choices
    - improbable_bigram_freq: DataFrame with frequency and slider statistics of improbable bigrams
    - analysis_results: Dictionary containing analysis results for typing times and slider values
    """
    def get_improbable_pair(row):
        for pair in easy_choice_pairs:
            if set(pair) == set([row['chosen_bigram'], row['unchosen_bigram']]):
                return pair
        return None

    def is_improbable_choice(row):
        if row['improbable_pair']:
            return row['chosen_bigram'] == row['improbable_pair'][1]
        return False

    # Add column indicating if the pair includes an improbable bigram
    bigram_data['improbable_pair'] = bigram_data.apply(get_improbable_pair, axis=1)
    
    # Add column indicating if an improbable choice was made
    bigram_data['improbable_choice'] = bigram_data.apply(is_improbable_choice, axis=1)

    # Count the number of improbable choices for each user
    user_improbable_counts = bigram_data.groupby('user_id')['improbable_choice'].sum()

    # Identify users with suspiciously high numbers of improbable choices
    suspicious_users = user_improbable_counts[user_improbable_counts > threshold].reset_index()
    suspicious_users.columns = ['user_id', 'improbable_choice_count']

    # Calculate overall statistics
    total_users = bigram_data['user_id'].nunique()
    total_improbable_pairs_possible = len(easy_choice_pairs) * total_users
    total_improbable_choices = bigram_data['improbable_choice'].sum()
    total_suspicious = len(suspicious_users)

    # Calculate frequency of improbable bigrams chosen and their probable counterparts
    improbable_bigram_freq = bigram_data[bigram_data['improbable_choice']]['improbable_pair'].value_counts().reset_index()
    improbable_bigram_freq.columns = ['bigram_pair', 'frequency']
    improbable_bigram_freq['probable_bigram'] = improbable_bigram_freq['bigram_pair'].apply(lambda x: x[0])
    improbable_bigram_freq['improbable_bigram'] = improbable_bigram_freq['bigram_pair'].apply(lambda x: x[1])
    
    # Calculate median and MAD slider values for each improbable bigram pair
    def get_slider_stats(pair):
        mask = (bigram_data['improbable_pair'] == pair) & bigram_data['improbable_choice']
        slider_values = bigram_data.loc[mask, 'sliderValue']
        median = slider_values.median()
        mad = np.median(np.abs(slider_values - median))
        return pd.Series({'median_slider': median, 'mad_slider': mad})

    slider_stats = improbable_bigram_freq['bigram_pair'].apply(get_slider_stats)
    improbable_bigram_freq = pd.concat([improbable_bigram_freq, slider_stats], axis=1)
    
    improbable_bigram_freq = improbable_bigram_freq.drop('bigram_pair', axis=1)

    print("\n____ Improbable Bigram Choice Analysis ____\n")
    print(f"Number of times an improbable bigram was chosen: {total_improbable_choices} of {total_improbable_pairs_possible} ({(total_improbable_choices / total_improbable_pairs_possible) * 100:.2f}%)")
    print(f"Users who made >{threshold} improbable choices: {total_suspicious} of {total_users} ({(total_suspicious / total_users) * 100:.2f}%)")

    if not suspicious_users.empty:
        print("\n10 users who made the highest number of improbable choices:")
        print(suspicious_users.sort_values('improbable_choice_count', ascending=False).head(10))

    print("\nFrequency of improbable bigrams chosen and their probable counterparts:")
    print(improbable_bigram_freq.to_string(index=False))

    # Create a set of improbable bigrams
    improbable_bigrams = set([pair[1] for pair in easy_choice_pairs])
    
    # Mark choices as improbable or probable
    bigram_data['is_improbable_choice'] = bigram_data['chosen_bigram'].isin(improbable_bigrams)
    
    # Function to perform Mann-Whitney U test and return results
    def perform_mann_whitney(group1, group2, label1, label2):
        group1 = group1.dropna()
        group2 = group2.dropna()
        
        if len(group1) > 0 and len(group2) > 0:
            median1 = group1.median()
            median2 = group2.median()
            mad1 = stats.median_abs_deviation(group1, nan_policy='omit')
            mad2 = stats.median_abs_deviation(group2, nan_policy='omit')
            
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            print(f"\n{label1} median (MAD): {median1:.2f} ({mad1:.2f})")
            print(f"{label2} median (MAD): {median2:.2f} ({mad2:.2f})")
            print(f"Mann-Whitney U test results:")
            print(f"U-statistic: {statistic:.4f}")
            print(f"p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"There is a significant difference between {label1} and {label2}: {label1} {'tend to be lower' if median1 < median2 else 'tend to be higher'}.")
            else:
                print(f"There is no significant difference between {label1} and {label2}.")
            
            return {
                f'{label1.lower()}_median': median1,
                f'{label2.lower()}_median': median2,
                f'{label1.lower()}_mad': mad1,
                f'{label2.lower()}_mad': mad2,
                'u_statistic': statistic,
                'p_value': p_value
            }
        else:
            print(f"\nInsufficient data to perform statistical analysis for {label1} vs {label2}.")
            return None

    # Analyze typing times
    print("\n____ Typing Times: Improbable vs Probable Choices ____")
    typing_time_results = perform_mann_whitney(
        bigram_data[bigram_data['is_improbable_choice']]['chosen_bigram_time'],
        bigram_data[~bigram_data['is_improbable_choice']]['chosen_bigram_time'],
        'improbable choice typing times', 'probable choice typing times'
    )

    # Analyze slider values
    print("\n____ Slider Values: Improbable vs Probable Choices ____")
    slider_value_results = perform_mann_whitney(
        bigram_data[bigram_data['is_improbable_choice']]['sliderValue'].abs(),
        bigram_data[~bigram_data['is_improbable_choice']]['sliderValue'].abs(),
        'improbable choice slider values', 'probable choice slider values'
    )

    analysis_results = {
        'typing_time_results': typing_time_results,
        'slider_value_results': slider_value_results
    }

    return suspicious_users, improbable_bigram_freq, analysis_results

#######################################
# Analyze bigram choice inconsistencies
#######################################

def analyze_choice_inconsistencies(bigram_data):
    """
    Analyze and report inconsistencies in bigram data.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data

    Returns:
    - inconsistency_stats: Dictionary containing inconsistency statistics
    """
    if bigram_data.empty:
        print("The bigram data is empty. No analysis can be performed.")
        return {}

    # Standardize bigram pairs
    bigram_data['std_bigram_pair'] = bigram_data['bigram_pair'].apply(lambda x: ','.join(sorted(x.split(', '))))

    # Group by user and standardized bigram pair
    grouped = bigram_data.groupby(['user_id', 'std_bigram_pair'])

    # Function to check for inconsistency within a group
    def check_inconsistency(group):
        if len(group) > 1:
            positive = (group['sliderValue'] > 0).any()
            negative = (group['sliderValue'] < 0).any()
            return positive and negative
        return False

    # Find inconsistent pairs
    inconsistent_pairs = []
    for name, group in grouped:
        if check_inconsistency(group):
            inconsistent_pairs.append(name)

    # Create a DataFrame of inconsistent pairs
    inconsistent_pairs_df = pd.DataFrame(inconsistent_pairs, columns=['user_id', 'std_bigram_pair'])

    # If no inconsistencies found
    if inconsistent_pairs_df.empty:
        print("No inconsistencies found. All users chose consistently for each unique bigram pair.")
        return {}

    # Count inconsistencies
    inconsistency_counts = inconsistent_pairs_df['std_bigram_pair'].value_counts()

    # Users with inconsistencies
    users_with_inconsistencies = inconsistent_pairs_df['user_id'].nunique()

    # Inconsistent choices per user
    inconsistent_choices_per_user = inconsistent_pairs_df.groupby('user_id')['std_bigram_pair'].nunique().sort_values(ascending=False)

    # Calculate inconsistent_pair_user_counts
    inconsistent_pair_user_counts = inconsistent_pairs_df.groupby('std_bigram_pair')['user_id'].nunique()

    print("\n____ Bigram Choice Inconsistency Statistics ____\n")
    print(f"Unique bigram pairs with inconsistencies: {len(inconsistency_counts)}")
    print(f"Users with at least one inconsistency: {users_with_inconsistencies}")

    print(f"\n10 users with the highest number of inconsistent choices:")
    print(inconsistent_choices_per_user.head(10).to_string())

    print(f"\nFrequency of inconsistencies per bigram pair (with slider statistics):")
    for pair, count in inconsistency_counts.items():
        pair_data = bigram_data[bigram_data['std_bigram_pair'] == pair]
        total_users_for_pair = pair_data['user_id'].nunique()
        
        # Calculate median and MAD of slider values
        slider_values = pair_data['sliderValue']
        median_slider = slider_values.median()
        mad_slider = np.median(np.abs(slider_values - median_slider))
        
        print(f"'{pair}': {count} of {total_users_for_pair} users;  slider median (MAD): {median_slider:.2f} ({mad_slider:.2f})")

    # Return the statistics for further use if needed
    bigram_choice_inconsistency_stats = {
        'users_with_inconsistencies': users_with_inconsistencies,
        'inconsistent_choices_per_user': inconsistent_choices_per_user,
        'inconsistency_counts': inconsistency_counts,
        'inconsistent_pair_user_counts': inconsistent_pair_user_counts
    }
    return bigram_choice_inconsistency_stats

def analyze_inconsistency_slider_relationship(bigram_data, output_plots_folder):
    """
    Analyze the relationship between inconsistent choices, slider values, and typing times.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved

    Returns:
    - inconsistency_analysis_results: Dictionary containing analysis results
    """
    # Group data by user and bigram pair
    grouped = bigram_data.groupby(['user_id', 'bigram_pair'])

    # Identify inconsistent choices and calculate average absolute slider values
    inconsistency_data = []
    for (user_id, bigram_pair), group in grouped:
        is_inconsistent = len(set(group['chosen_bigram'])) > 1
        avg_abs_slider = np.mean(np.abs(group['sliderValue']))
        avg_typing_time = np.mean(group['chosen_bigram_time'])
        inconsistency_data.append({
            'user_id': user_id,
            'bigram_pair': bigram_pair,
            'is_inconsistent': is_inconsistent,
            'avg_abs_slider': avg_abs_slider,
            'avg_typing_time': avg_typing_time
        })

    inconsistency_df = pd.DataFrame(inconsistency_data)

    # Function to perform Mann-Whitney U test and return results
    def perform_mann_whitney(group1, group2, label1, label2):
        group1 = group1.dropna()
        group2 = group2.dropna()
        
        if len(group1) > 0 and len(group2) > 0:
            median1 = group1.median()
            median2 = group2.median()
            mad1 = stats.median_abs_deviation(group1, nan_policy='omit')
            mad2 = stats.median_abs_deviation(group2, nan_policy='omit')
            
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            
            print(f"\n{label1} median (MAD): {median1:.2f} ({mad1:.2f})")
            print(f"{label2} median (MAD): {median2:.2f} ({mad2:.2f})")
            print(f"Mann-Whitney U test results:")
            print(f"U-statistic: {statistic:.4f}")
            print(f"p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"There is a significant difference between {label1} and {label2}: {label1} {'tend to be lower' if median1 < median2 else 'tend to be higher'}.")
            else:
                print(f"There is no significant difference between {label1} and {label2}.")
            
            return {
                f'{label1.lower()}_median': median1,
                f'{label2.lower()}_median': median2,
                f'{label1.lower()}_mad': mad1,
                f'{label2.lower()}_mad': mad2,
                'u_statistic': statistic,
                'p_value': p_value
            }
        else:
            print(f"\nInsufficient data to perform statistical analysis for {label1} vs {label2}.")
            return None

    # Analyze slider values
    print("\n____ Slider Values: Consistent vs Inconsistent Choices ____")
    slider_value_results = perform_mann_whitney(
        inconsistency_df[~inconsistency_df['is_inconsistent']]['avg_abs_slider'],
        inconsistency_df[inconsistency_df['is_inconsistent']]['avg_abs_slider'],
        'consistent choice slider values', 'inconsistent choice slider values'
    )

    # Analyze typing times
    print("\n____ Typing Times: Consistent vs Inconsistent Choices ____")
    typing_time_results = perform_mann_whitney(
        inconsistency_df[~inconsistency_df['is_inconsistent']]['avg_typing_time'],
        inconsistency_df[inconsistency_df['is_inconsistent']]['avg_typing_time'],
        'consistent choice typing times', 'inconsistent choice typing times'
    )

    # Create visualization for slider values
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_inconsistent', y='avg_abs_slider', data=inconsistency_df)
    plt.title('Average Absolute Slider Values for Consistent vs Inconsistent Choices')
    plt.xlabel('Is Inconsistent')
    plt.ylabel('Average Absolute Slider Value')
    plt.savefig(os.path.join(output_plots_folder, 'inconsistency_slider_relationship.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create visualization for typing times
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_inconsistent', y='avg_typing_time', data=inconsistency_df)
    plt.title('Average Typing Times for Consistent vs Inconsistent Choices')
    plt.xlabel('Is Inconsistent')
    plt.ylabel('Average Typing Time (ms)')
    plt.savefig(os.path.join(output_plots_folder, 'inconsistency_typing_time_relationship.png'), dpi=300, bbox_inches='tight')
    plt.close()

    inconsistency_analysis_results = {
        'slider_value_results': slider_value_results,
        'typing_time_results': typing_time_results
    }

    return inconsistency_analysis_results

def plot_chosen_vs_unchosen_times_barplot(bigram_data, output_plots_folder):
    plot_data = prepare_plot_data(bigram_data)
    
    plt.figure(figsize=(15, len(plot_data['bigram'].unique()) * 0.4))
    sns.barplot(x='time', y='bigram', hue='type', data=plot_data)
    
    plt.title('Chosen vs unchosen typing time for each bigram')
    plt.xlabel('Median typing time (ms)')
    plt.ylabel('Bigram')
    plt.legend(title='Bigram type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'chosen_vs_unchosen_times.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_chosen_vs_unchosen_times_scatter_regression(bigram_data, output_plots_folder):
    """
    Generate and save scatter plot with regression line for chosen vs. unchosen typing times.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    """
    plot_data = prepare_plot_data(bigram_data)
    
    # Pivot the data to have chosen and unchosen times as separate columns
    plot_data_wide = plot_data.pivot(index='bigram', columns='type', values='time').reset_index()
    
    # Scatter plot with regression line
    plt.figure(figsize=(10, 8))
    sns.regplot(x='chosen', y='unchosen', data=plot_data_wide, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    # Set title and labels
    max_val = max(plot_data_wide['chosen'].max(), plot_data_wide['unchosen'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)  # Add diagonal line
    plt.title('Chosen vs unchosen typing time for each bigram', fontsize=16)
    plt.xlabel('Chosen bigram time (ms)', fontsize=14)
    plt.ylabel('Unchosen bigram time (ms)', fontsize=14)
    
    # Calculate correlation
    correlation = plot_data_wide['chosen'].corr(plot_data_wide['unchosen'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'chosen_vs_unchosen_times_scatter_regression.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #print(f"chosen_vs_unchosen_times_scatter_regression plot saved in {output_plots_folder}")
    #print(f"Correlation between chosen and unchosen times: {correlation:.2f}")

def plot_chosen_vs_unchosen_times_joint(bigram_data, output_plots_folder):
    """
    Generate and save joint plot of chosen vs. unchosen typing times.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    """
    plot_data = prepare_plot_data(bigram_data)
    
    # Pivot the data to have chosen and unchosen times as separate columns
    plot_data_wide = plot_data.pivot(index='bigram', columns='type', values='time').reset_index()

    # Create joint plot
    g = sns.jointplot(x='chosen', y='unchosen', data=plot_data_wide, kind="scatter", 
                      joint_kws={'alpha': 0.5}, 
                      marginal_kws=dict(bins=20, fill=True))
   
    # Add regression line
    g = g.plot_joint(sns.regplot, scatter=False, line_kws={'color':'red'})
    
    max_val = max(plot_data_wide['chosen'].max(), plot_data_wide['unchosen'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)  # Add diagonal line
    plt.suptitle('Chosen vs unchosen typing time for each bigram', fontsize=16, y=1.02)
    plt.xlabel('Chosen bigram time (ms)', fontsize=14)
    plt.ylabel('Unchosen bigram time (ms)', fontsize=14)

    # Calculate correlation
    correlation = plot_data_wide['chosen'].corr(plot_data_wide['unchosen'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=12)

    # Save plot
    g.savefig(os.path.join(output_plots_folder, 'chosen_vs_unchosen_times_joint.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #print(f"chosen_vs_unchosen_times_joint plot saved in {output_plots_folder}")
    #print(f"Correlation between chosen and unchosen times: {correlation:.2f}")

def analyze_improbable_vs_inconsistent(bigram_data, easy_choice_pairs):
    """
    Analyze the relationship between improbable pairs and inconsistent choices.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - easy_choice_pairs: List of tuples, each containing a pair of bigrams where one is highly improbable

    Returns:
    - analysis_results: Dictionary containing analysis results
    """
    # Identify improbable pairs
    improbable_pairs = set([tuple(sorted(pair)) for pair in easy_choice_pairs])
    
    # Mark bigrams as improbable or not
    bigram_data['is_improbable_pair'] = bigram_data['bigram_pair'].apply(lambda x: tuple(sorted(x.split(', '))) in improbable_pairs)
    
    # Group by user and bigram pair
    grouped = bigram_data.groupby(['user_id', 'bigram_pair'])
    
    # Identify inconsistent choices
    def is_inconsistent(group):
        return len(set(group['chosen_bigram'])) > 1
    
    inconsistent_choices = grouped.apply(is_inconsistent)
    
    # Combine improbable and inconsistent information
    combined_data = pd.DataFrame({
        'is_improbable_pair': grouped['is_improbable_pair'].first(),
        'is_inconsistent': inconsistent_choices
    }).reset_index()
    
    # Calculate statistics
    total_pairs = len(combined_data)
    improbable_count = combined_data['is_improbable_pair'].sum()
    inconsistent_count = combined_data['is_inconsistent'].sum()
    improbable_and_inconsistent = ((combined_data['is_improbable_pair']) & (combined_data['is_inconsistent'])).sum()
    
    # Calculate percentages
    percent_improbable = (improbable_count / total_pairs) * 100
    percent_inconsistent = (inconsistent_count / total_pairs) * 100
    percent_improbable_and_inconsistent = (improbable_and_inconsistent / improbable_count) * 100 if improbable_count > 0 else 0
    
    # Prepare contingency table for chi-square test
    contingency_table = pd.crosstab(combined_data['is_improbable_pair'], combined_data['is_inconsistent'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    analysis_results = {
        'total_pairs': total_pairs,
        'improbable_count': improbable_count,
        'inconsistent_count': inconsistent_count,
        'improbable_and_inconsistent': improbable_and_inconsistent,
        'percent_improbable': percent_improbable,
        'percent_inconsistent': percent_inconsistent,
        'percent_improbable_and_inconsistent': percent_improbable_and_inconsistent,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'contingency_table': contingency_table
    }
    
    print("\n____ Improbable vs Inconsistent Analysis ____\n")
    print(f"Total pairs: {analysis_results['total_pairs']}")
    print(f"Improbable pairs: {analysis_results['improbable_count']} ({analysis_results['percent_improbable']:.1f}%)")
    print(f"Inconsistent choices: {analysis_results['inconsistent_count']} ({analysis_results['percent_inconsistent']:.1f}%)")
    print(f"Improbable and Inconsistent: {analysis_results['improbable_and_inconsistent']} ({analysis_results['percent_improbable_and_inconsistent']:.1f}% of improbable pairs)")
    if improbable_and_inconsistent > 0:
        print(f"Chi-square statistic: {analysis_results['chi2_statistic']:.2f}")
        print(f"p-value: {analysis_results['p_value']:.4f}")

    return analysis_results


def analyze_within_user_bigram_times(bigram_data, inconsistency_data, improbable_bigram_freq, output_plots_folder):    
    """
    Analyze bigram typing times within users to find significantly different typing times across bigrams,
    considering both chosen and unchosen times.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - inconsistency_data: Dictionary containing inconsistency data per user
    - improbable_bigram_freq: DataFrame containing frequency of improbable bigrams
    - output_plots_folder: String path to the folder where plots should be saved

    Returns:
    - within_user_stats: Dictionary containing within-user analysis results
    """
    user_significant_pairs = {}
    total_possible_comparisons = 0
    total_significant_differences = 0
    all_bigrams = set(bigram_data['chosen_bigram'].unique()) | set(bigram_data['unchosen_bigram'].unique())
    
    for user_id, user_data in bigram_data.groupby('user_id'):
        significant_pairs = []
        user_comparisons = 0
        
        # Combine chosen and unchosen times for each bigram
        bigram_times = {}
        for bigram in all_bigrams:
            chosen_times = user_data[user_data['chosen_bigram'] == bigram]['chosen_bigram_time']
            unchosen_times = user_data[user_data['unchosen_bigram'] == bigram]['unchosen_bigram_time']
            bigram_times[bigram] = pd.concat([chosen_times, unchosen_times]).dropna()
        
        for bigram1, bigram2 in combinations(all_bigrams, 2):
            times1 = bigram_times[bigram1]
            times2 = bigram_times[bigram2]
            
            if len(times1) > 0 and len(times2) > 0:
                user_comparisons += 1
                statistic, p_value = stats.mannwhitneyu(times1, times2, alternative='two-sided')
                if p_value < 0.05:
                    faster_bigram = bigram1 if times1.median() < times2.median() else bigram2
                    slower_bigram = bigram2 if faster_bigram == bigram1 else bigram1
                    significant_pairs.append((faster_bigram, slower_bigram, p_value))
                    total_significant_differences += 1
        
        user_significant_pairs[user_id] = significant_pairs
        total_possible_comparisons += user_comparisons

    # Count significant differences for each user
    significant_pairs_count = {user_id: len(pairs) for user_id, pairs in user_significant_pairs.items()}

    # Calculate statistics
    users_with_differences = sum(count > 0 for count in significant_pairs_count.values())

    # Print analysis results
    print("\n____ Within-User Bigram Typing Time Analysis ____\n")
    print(f"Total number of bigrams compared: {len(all_bigrams)}")
    print(f"Users with at least one significant difference in bigram typing times: {users_with_differences} out of {len(significant_pairs_count)} total users")
    print(f"Total possible comparisons across all users: {total_possible_comparisons}")
    print(f"Total significant differences across all users: {total_significant_differences}")
    
    if total_significant_differences > 0:

        # Count occurrences of each bigram in significant pairs
        bigram_occurrences = Counter()
        for pairs in user_significant_pairs.values():
            for faster, slower, _ in pairs:
                bigram_occurrences[faster] += 1
                bigram_occurrences[slower] += 1
        
        print("\nBigrams involved in significant differences (sorted by frequency):")
        for bigram, count in bigram_occurrences.most_common():
            print(f"{bigram}: {count} times")
    else:
        print("\nNo significant differences in typing times were found between bigrams.")

    return {
        'user_significant_pairs': user_significant_pairs,
        'significant_pairs_count': significant_pairs_count,
        'total_significant_differences': total_significant_differences,
        'users_with_differences': users_with_differences,
        'bigram_occurrences': dict(bigram_occurrences) if 'bigram_occurrences' in locals() else {}
    }

#################################
# Analyze only consistent choices
#################################

def filter_consistent_choices(bigram_data):
    """
    Filter out inconsistent rows from the bigram data.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data

    Returns:
    - consistent_bigram_data: DataFrame containing only consistent choices
    """
    return bigram_data[bigram_data['is_consistent']]

def analyze_consistent_typing_times(consistent_bigram_data, output_plots_folder):
    """
    Analyze typing times for consistent choices, comparing chosen vs. unchosen bigrams.

    Parameters:
    - consistent_bigram_data: DataFrame containing only consistent choices
    - output_plots_folder: String path to the folder where plots should be saved

    Returns:
    - consistent_typing_time_stats: Dictionary containing typing time statistics for consistent choices
    """
    print("\n____ Consistent Choice Typing Time Analysis ____\n")

    chosen_times = consistent_bigram_data['chosen_bigram_time']
    unchosen_times = consistent_bigram_data['unchosen_bigram_time']

    statistic, p_value = stats.mannwhitneyu(chosen_times, unchosen_times, alternative='two-sided')

    print(f"Mann-Whitney U test statistic: {statistic}")
    print(f"p-value: {p_value}")

    if p_value < 0.05:
        print("There is a significant difference in typing times between chosen and unchosen bigrams.")
        if chosen_times.median() < unchosen_times.median():
            print("Chosen bigrams tend to have shorter typing times.")
        else:
            print("Chosen bigrams tend to have longer typing times.")
    else:
        print("There is no significant difference in typing times between chosen and unchosen bigrams.")

    # Create a box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=['Chosen']*len(chosen_times) + ['Unchosen']*len(unchosen_times),
                y=pd.concat([chosen_times, unchosen_times]))
    plt.title('Typing Times for Chosen vs Unchosen Bigrams (Consistent Choices)')
    plt.ylabel('Typing Time (ms)')
    plt.savefig(os.path.join(output_plots_folder, 'consistent_chosen_vs_unchosen_times.png'))
    plt.close()

    return {
        'statistic': statistic,
        'p_value': p_value,
        'chosen_median': chosen_times.median(),
        'unchosen_median': unchosen_times.median()
    }

def analyze_consistent_slider_values(consistent_bigram_data, output_plots_folder):
    """
    Analyze slider values for consistent choices.

    Parameters:
    - consistent_bigram_data: DataFrame containing only consistent choices
    - output_plots_folder: String path to the folder where plots should be saved

    Returns:
    - consistent_slider_stats: Dictionary containing slider value statistics for consistent choices
    """
    print("\n____ Consistent Choice Slider Value Analysis ____\n")

    slider_values = consistent_bigram_data['sliderValue']

    print(f"Median slider value: {slider_values.median()}")
    print(f"Mean slider value: {slider_values.mean()}")
    print(f"Standard deviation: {slider_values.std()}")

    # Create a histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(slider_values, kde=True)
    plt.title('Distribution of Slider Values for Consistent Choices')
    plt.xlabel('Slider Value')
    plt.savefig(os.path.join(output_plots_folder, 'consistent_slider_value_distribution.png'))
    plt.close()

    return {
        'median': slider_values.median(),
        'mean': slider_values.mean(),
        'std': slider_values.std()
    }

def analyze_consistent_bigram_preferences(consistent_bigram_data):
    """
    Analyze which bigrams are most frequently chosen when decisions are consistent.

    Parameters:
    - consistent_bigram_data: DataFrame containing only consistent choices

    Returns:
    - bigram_preference_stats: Dictionary containing bigram preference statistics
    """
    print("\n____ Consistent Bigram Preference Analysis ____\n")

    bigram_counts = consistent_bigram_data['chosen_bigram'].value_counts()
    total_choices = len(consistent_bigram_data)

    print("Top 10 most frequently chosen bigrams:")
    for bigram, count in bigram_counts.head(10).items():
        percentage = (count / total_choices) * 100
        print(f"{bigram}: {count} times ({percentage:.2f}%)")

    return {
        'bigram_counts': bigram_counts,
        'total_choices': total_choices
    }

def analyze_typing_time_slider_relationship(consistent_bigram_data, output_plots_folder):
    """
    Analyze the relationship between typing times and slider values for consistent choices.

    Parameters:
    - consistent_bigram_data: DataFrame containing only consistent choices
    - output_plots_folder: String path to the folder where plots should be saved

    Returns:
    - time_slider_relationship: Dictionary containing relationship statistics
    """
    print("\n____ Typing Time vs Slider Value Relationship Analysis ____\n")

    typing_times = consistent_bigram_data['chosen_bigram_time']
    slider_values = consistent_bigram_data['sliderValue'].abs()

    correlation, p_value = stats.spearmanr(typing_times, slider_values)

    print(f"Spearman correlation coefficient: {correlation}")
    print(f"p-value: {p_value}")

    if p_value < 0.05:
        print("There is a significant relationship between typing time and slider value.")
        if correlation > 0:
            print("Longer typing times tend to be associated with higher absolute slider values.")
        else:
            print("Longer typing times tend to be associated with lower absolute slider values.")
    else:
        print("There is no significant relationship between typing time and slider value.")

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(typing_times, slider_values, alpha=0.5)
    plt.title('Typing Time vs Absolute Slider Value (Consistent Choices)')
    plt.xlabel('Typing Time (ms)')
    plt.ylabel('Absolute Slider Value')
    plt.savefig(os.path.join(output_plots_folder, 'consistent_time_vs_slider.png'))
    plt.close()

    return {
        'correlation': correlation,
        'p_value': p_value
    }


"""
def plot_improbable_vs_inconsistent(analysis_results, output_plots_folder):
    ""
    Generate and save plots comparing improbable pairs and inconsistent choices.

    Parameters:
    - analysis_results: Dictionary containing analysis results
    - output_plots_folder: String path to the folder where plots should be saved
    ""
    # Bar plot
    plt.figure(figsize=(10, 6))
    categories = ['Improbable', 'Inconsistent', 'Both']
    values = [analysis_results['percent_improbable'], 
              analysis_results['percent_inconsistent'], 
              analysis_results['percent_improbable_and_inconsistent']]
    
    plt.bar(categories, values)
    plt.title('Comparison of Improbable Pairs and Inconsistent Choices')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    
    for i, v in enumerate(values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'improbable_vs_inconsistent_barplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Heatmap of contingency table
    plt.figure(figsize=(8, 6))
    sns.heatmap(analysis_results['contingency_table'], annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Contingency Table: Improbable Pairs vs Inconsistent Choices')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'improbable_vs_inconsistent_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #print(f"Improbable vs Inconsistent plots saved in {output_plots_folder}")
"""

# Main execution
if __name__ == "__main__":
    try:
        ##########################
        # LOAD_AND_PREPROCESS_DATA
        ##########################
        # Set the paths for input data and output
        input_folder = '/Users/arno.klein/Downloads/osf/summary'
        output_folder = os.path.join(os.path.dirname(input_folder), 'output')
        output_tables_folder = os.path.join(output_folder, 'tables')
        output_plots_folder = os.path.join(output_folder, 'plots')
        os.makedirs(output_tables_folder, exist_ok=True)
        os.makedirs(output_plots_folder, exist_ok=True)

        # Load improbable pairs from CSV file
        current_dir = os.getcwd()  # Get the current working directory
        parent_dir = os.path.dirname(current_dir)  # Get the parent directory
        easy_choice_pairs_file = os.path.join(parent_dir, 'bigram_tables', 'bigram_5pairs_easy_choices_LH.csv')
        easy_choice_pairs = load_easy_choice_pairs(easy_choice_pairs_file)

        # Load, combine, and save the data
        data = load_and_preprocess_data(input_folder, output_tables_folder, verbose=False)

        bigram_pairs_file = os.path.join(parent_dir, 'bigram_tables', 'bigram_27pairs_11tests_11swap_5easy_LH.csv')
        num_bigram_pairs, bigram_pairs_df = load_bigram_pairs(bigram_pairs_file)

        ##########################
        # PROCESS_BIGRAM_DATA
        ##########################
        bigram_data = process_bigram_data(data, output_tables_folder, verbose=False)

        #############################
        # Analyze bigram typing times
        #############################
        typing_time_stats = analyze_typing_times(bigram_data, output_plots_folder)
        plot_median_bigram_times(bigram_data, output_plots_folder)

        ############################
        # Analyze improbable choices
        ############################
        if easy_choice_pairs:
            # Analyze improbable choices
            suspicious_users, improbable_bigram_freq, typing_time_results = analyze_easy_choices(bigram_data, easy_choice_pairs, threshold=2)
        else:
            print("Skipping easy choice bigram analysis due to missing or invalid easy choice pairs data.")

        #######################################
        # Analyze bigram choice inconsistencies
        #######################################
        bigram_choice_inconsistency_stats = analyze_choice_inconsistencies(bigram_data)

        # Analyze the relationship between inconsistent choices and slider values
        inconsistency_slider_stats = analyze_inconsistency_slider_relationship(bigram_data, output_plots_folder)

        plot_chosen_vs_unchosen_times_barplot(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_scatter_regression(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_joint(bigram_data, output_plots_folder)
        #improbable_inconsistent_results = analyze_improbable_vs_inconsistent(bigram_data, easy_choice_pairs)        
        #plot_improbable_vs_inconsistent(improbable_inconsistent_results, output_plots_folder)

        # Analyze within-user bigram typing times and relationships
        within_user_stats = analyze_within_user_bigram_times(
            bigram_data, 
            bigram_choice_inconsistency_stats, 
            improbable_bigram_freq,  # Pass improbable_bigram_freq directly
            output_plots_folder
        )

        #################################
        # Analyze only consistent choices
        #################################

        print("\n\n=============== Consistent Choice Analysis ===============\n")

        # Filter consistent choices
        consistent_bigram_data = filter_consistent_choices(bigram_data)
        print(f"Total choices: {len(bigram_data)}")
        print(f"Consistent choices: {len(consistent_bigram_data)} ({len(consistent_bigram_data)/len(bigram_data)*100:.2f}%)")

        # Analyze consistent choices
        consistent_typing_time_stats = analyze_consistent_typing_times(consistent_bigram_data, output_plots_folder)
        consistent_slider_stats = analyze_consistent_slider_values(consistent_bigram_data, output_plots_folder)
        consistent_bigram_preferences = analyze_consistent_bigram_preferences(consistent_bigram_data)
        time_slider_relationship = analyze_typing_time_slider_relationship(consistent_bigram_data, output_plots_folder)

        print("\n=============== Consistent Choice Analysis Complete ===============\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()