
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
    
def process_bigram_data(data, easy_choice_pairs, output_tables_folder, verbose=False):
    """
    Process the bigram data from a DataFrame and create additional dataframes for specific subsets.

    Parameters:
    - data: DataFrame with combined bigram data
    - easy_choice_pairs: List of tuples containing easy choice (probable, improbable) bigram pairs
    - output_tables_folder: str, path to the folder where the processed data will be saved
    - verbose: bool, if True, print additional information

    Returns:
    - dict: Dictionary containing various processed dataframes
    """
    
    # Create dictionaries for quick lookup of probable and improbable pairs
    probable_pairs = {(pair[0], pair[1]): True for pair in easy_choice_pairs}
    improbable_pairs = {(pair[1], pair[0]): True for pair in easy_choice_pairs}
    
    # First, create a standardized bigram pair representation
    data['std_bigram_pair'] = data.apply(lambda row: ', '.join(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
    
    # Group the data by user_id and standardized bigram pair
    grouped_data = data.groupby(['user_id', 'std_bigram_pair'])
    
    # Calculate the size of each group
    group_sizes = grouped_data.size()
    
    result_list = []
    
    for (user_id, std_bigram_pair), group in grouped_data:
        bigram1, bigram2 = std_bigram_pair.split(', ')
        
        # Check consistency (only for pairs that appear more than once)
        is_consistent = len(set(group['chosenBigram'])) == 1 if len(group) > 1 else None
        
        for _, row in group.iterrows():
            chosen_bigram = row['chosenBigram']
            unchosen_bigram = row['unchosenBigram']
            
            # Determine if the choice is probable or improbable
            is_probable = probable_pairs.get((chosen_bigram, unchosen_bigram), False)
            is_improbable = improbable_pairs.get((chosen_bigram, unchosen_bigram), False)
            
            result = pd.DataFrame({
                'user_id': [user_id],
                'trialId': [row['trialId']],
                'bigram_pair': [std_bigram_pair],
                'bigram1': [bigram1],
                'bigram2': [bigram2],
                'bigram1_time': [row['chosenBigramTime'] if row['chosenBigram'] == bigram1 else row['unchosenBigramTime']],
                'bigram2_time': [row['chosenBigramTime'] if row['chosenBigram'] == bigram2 else row['unchosenBigramTime']],
                'chosen_bigram': [chosen_bigram],
                'unchosen_bigram': [unchosen_bigram],
                'chosen_bigram_time': [row['chosenBigramTime']],
                'unchosen_bigram_time': [row['unchosenBigramTime']],
                'chosen_bigram_correct': [row['chosenBigramCorrect']],
                'unchosen_bigram_correct': [row['unchosenBigramCorrect']],
                'sliderValue': [row['sliderValue']],
                'text': [row['text']],
                'is_consistent': [is_consistent],
                'is_probable': [is_probable],
                'is_improbable': [is_improbable],
                'group_size': [group_sizes[(user_id, std_bigram_pair)]]
            })
            result_list.append(result)
    
    # Concatenate all the results into a single DataFrame
    bigram_data = pd.concat(result_list).reset_index(drop=True)
    
    # Sort the DataFrame
    bigram_data = bigram_data.sort_values(by=['user_id', 'trialId', 'bigram_pair']).reset_index(drop=True)
    
    # Create dataframes for specific subsets
    consistent_choices = bigram_data[(bigram_data['is_consistent'] == True) & (bigram_data['group_size'] > 1)].drop(['is_probable', 'is_improbable'], axis=1)
    inconsistent_choices = bigram_data[(bigram_data['is_consistent'] == False) & (bigram_data['group_size'] > 1)].drop(['is_probable', 'is_improbable'], axis=1)
    probable_choices = bigram_data[bigram_data['is_probable']].drop(['is_consistent', 'group_size'], axis=1)
    improbable_choices = bigram_data[bigram_data['is_improbable']].drop(['is_consistent', 'group_size'], axis=1)
    
    # Display information about the DataFrames
    if verbose:
        print_headers = ['trialId', 'bigram_pair', 'chosen_bigram', 'unchosen_bigram', 'chosen_bigram_time', 'chosen_bigram_correct', 'sliderValue']
        display_information(bigram_data, "original data", print_headers + ['is_consistent', 'is_probable', 'is_improbable', 'group_size'], nlines=30)
        display_information(consistent_choices, "consistent choices", print_headers + ['is_consistent', 'group_size'], nlines=10)
        display_information(inconsistent_choices, "inconsistent choices", print_headers + ['is_consistent', 'group_size'], nlines=10)
        display_information(probable_choices, "probable choices", print_headers + ['is_probable'], nlines=10)
        display_information(improbable_choices, "improbable choices", print_headers + ['is_improbable'], nlines=10)
    
    # Save the DataFrames to CSV files
    bigram_data.to_csv(f"{output_tables_folder}/processed_bigram_data.csv", index=False)
    consistent_choices.to_csv(f"{output_tables_folder}/consistent_choices.csv", index=False)
    inconsistent_choices.to_csv(f"{output_tables_folder}/inconsistent_choices.csv", index=False)
    probable_choices.to_csv(f"{output_tables_folder}/probable_choices.csv", index=False)
    improbable_choices.to_csv(f"{output_tables_folder}/improbable_choices.csv", index=False)
    
    print(f"Processed data saved to {output_tables_folder}")
    
    return {
        'bigram_data': bigram_data,
        'consistent_choices': consistent_choices,
        'inconsistent_choices': inconsistent_choices,
        'probable_choices': probable_choices,
        'improbable_choices': improbable_choices
    }

def analyze_bigram_data(processed_data, output_tables_folder, output_plots_folder):
    """
    Analyze user inconsistencies and improbable choices, create tables, generate plots,
    and perform a statistical test on the relationship between inconsistent and improbable choices.

    Parameters:
    - processed_data: Dictionary containing various processed dataframes from process_bigram_data
    - output_tables_folder: String path to the folder where the CSV files should be saved
    - output_plots_folder: String path to the folder where the plots should be saved

    Returns:
    - user_stats: DataFrame containing user statistics
    """
    # Extract relevant dataframes from processed_data
    bigram_data = processed_data['bigram_data']
    consistent_choices = processed_data['consistent_choices']
    inconsistent_choices = processed_data['inconsistent_choices']
    probable_choices = processed_data['probable_choices']
    improbable_choices = processed_data['improbable_choices']

    # Print dataframe information for debugging
    #for name, df in processed_data.items():
    #    print(f"\n{name} shape: {df.shape}")
    #    print(f"{name} columns: {df.columns.tolist()}")
    #    print(f"{name} first few rows:")
    #    print(df.head().to_string())

    # Calculate statistics for each user
    user_stats = pd.DataFrame()
    user_stats['total_choices'] = bigram_data.groupby('user_id').size()
    user_stats['consistent_choices'] = consistent_choices.groupby('user_id').size()
    user_stats['inconsistent_choices'] = inconsistent_choices.groupby('user_id').size()
    user_stats['probable_choices'] = probable_choices.groupby('user_id').size()
    user_stats['improbable_choices'] = improbable_choices.groupby('user_id').size()

    # Fill NaN values with 0 for users who might not have any choices in a category
    user_stats = user_stats.fillna(0)

    # Ensure all columns are integers
    user_stats = user_stats.astype(int)

    # Calculate proportions
    user_stats['proportion_inconsistent'] = user_stats['inconsistent_choices'] / (user_stats['consistent_choices'] + user_stats['inconsistent_choices'])
    user_stats['proportion_improbable'] = user_stats['improbable_choices'] / user_stats['total_choices']

    # Perform Spearman's rank correlation test
    correlation, p_value = stats.spearmanr(user_stats['proportion_inconsistent'], user_stats['proportion_improbable'])

    print("\nStatistical Test Results:")
    print(f"Spearman's rank correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("There is a significant relationship between the proportion of inconsistent responses and the proportion of improbable responses.")
        if correlation > 0:
            print("The relationship is positive, meaning that as the proportion of inconsistent responses increases, the proportion of improbable responses tends to increase as well.")
        else:
            print("The relationship is negative, meaning that as the proportion of inconsistent responses increases, the proportion of improbable responses tends to decrease.")
    else:
        print("There is no significant relationship between the proportion of inconsistent responses and the proportion of improbable responses.")

    # Function to create choice table
    def create_choice_table(data):
        # Split the bigram_pair into two columns
        data[['bigram1', 'bigram2']] = data['bigram_pair'].str.split(', ', expand=True)
        
        # Count the number of unique users for each bigram as the chosen_bigram
        pair_counts = pd.DataFrame({
            '#users pair 1': data[data['chosen_bigram'] == data['bigram1']].groupby('bigram_pair')['user_id'].nunique(),
            '#users pair 2': data[data['chosen_bigram'] == data['bigram2']].groupby('bigram_pair')['user_id'].nunique()
        })
        
        return pair_counts.fillna(0).astype(int)

    # Combine all relevant data
    all_pairs = pd.concat([consistent_choices, inconsistent_choices, probable_choices, improbable_choices])

    # Create a single table for all pairs
    all_pairs_table = create_choice_table(all_pairs)

    # Count unique users for each category
    all_pairs_table['#users consistent'] = consistent_choices.groupby('bigram_pair')['user_id'].nunique()
    all_pairs_table['#users inconsistent'] = inconsistent_choices.groupby('bigram_pair')['user_id'].nunique()
    all_pairs_table['#users probable'] = probable_choices.groupby('bigram_pair')['user_id'].nunique()
    all_pairs_table['#users improbable'] = improbable_choices.groupby('bigram_pair')['user_id'].nunique()

    # Fill NaN values with 0 and convert to integer
    all_pairs_table = all_pairs_table.fillna(0).astype(int)

    # Calculate total unique users
    all_pairs_table['total_users'] = all_pairs.groupby('bigram_pair')['user_id'].nunique()

    # Sort the table by total number of users in descending order
    all_pairs_table = all_pairs_table.sort_values('total_users', ascending=False)

    # Create and print Repeated Pairs Table
    repeated_pairs_table = all_pairs_table[all_pairs_table['#users consistent'] + all_pairs_table['#users inconsistent'] > 0].copy()
    repeated_pairs_table = repeated_pairs_table[['#users pair 1', '#users pair 2', '#users consistent', '#users inconsistent', 'total_users']]
    print("\nRepeated Pairs Table:")
    print(repeated_pairs_table.to_string())

    # Create and print Easy Choice Pairs Table
    easy_choice_table = all_pairs_table[all_pairs_table['#users probable'] + all_pairs_table['#users improbable'] > 0].copy()
    easy_choice_table = easy_choice_table[['#users pair 1', '#users pair 2', '#users probable', '#users improbable', 'total_users']]
    print("\nEasy Choice Pairs Table:")
    print(easy_choice_table.to_string())

    return user_stats

###############################
# Bigram Typing Time Statistics
###############################

def analyze_typing_times(bigram_data, output_plots_folder, output_filename1='chosen_vs_unchosen_times.png', output_filename2='typing_time_diff_vs_slider_value.png'):
    """
    Analyze and report typing times in bigram data, focusing on three main questions:
    1. Do chosen bigrams tend to have shorter typing times?
    2. Do shorter typing times correspond to higher absolute slider values?
    3. Is there a bias to the left (negative) vs. right (positive) slider values?
    Also reports on the number of times the faster bigram was chosen.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    - output_filename1: String filename of the plot of chosen vs. unchosen bigram times
    - output_filename2: String filename of the plot of typing time difference vs. slider value

    Returns:
    - typing_time_stats: Dictionary containing typing time statistics and test results
    """
    print("\n____ Bigram Typing Time Statistics ____\n")

    # Basic statistics
    total_rows = len(bigram_data)
    valid_chosen_times = bigram_data['chosen_bigram_time'].notna().sum()
    valid_unchosen_times = bigram_data['unchosen_bigram_time'].notna().sum()

    print(f"Total rows: {total_rows}")
    print(f"Number of valid chosen times: {valid_chosen_times}")
    print(f"Number of valid unchosen times: {valid_unchosen_times}")

    # Calculate number of times the faster bigram was chosen
    valid_comparisons = bigram_data.dropna(subset=['chosen_bigram_time', 'unchosen_bigram_time'])
    faster_chosen_count = (valid_comparisons['chosen_bigram_time'] < valid_comparisons['unchosen_bigram_time']).sum()
    total_valid_comparisons = len(valid_comparisons)
    
    print(f"\nNumber of times the faster bigram was chosen: {faster_chosen_count} out of {total_valid_comparisons} comparisons ({faster_chosen_count / total_valid_comparisons * 100:.2f}%)")

    # 1. Do chosen bigrams tend to have shorter typing times?
    chosen_times = valid_comparisons['chosen_bigram_time']
    unchosen_times = valid_comparisons['unchosen_bigram_time']

    if len(chosen_times) > 0:
        statistic, p_value = stats.wilcoxon(chosen_times, unchosen_times)
        
        print("\n1. Comparison of chosen vs. unchosen bigram typing times:")
        print(f"Wilcoxon signed-rank test statistic: {statistic}")
        print(f"p-value: {p_value}")
        
        if p_value < 0.05:
            if chosen_times.median() < unchosen_times.median():
                print("Chosen bigrams tend to have significantly shorter typing times.")
            else:
                print("Chosen bigrams tend to have significantly longer typing times.")
        else:
            print("There is no significant difference in typing times between chosen and unchosen bigrams.")

        # Additional information
        print(f"Median chosen bigram typing time: {chosen_times.median():.2f} ms")
        print(f"Median unchosen bigram typing time: {unchosen_times.median():.2f} ms")
    else:
        print("Insufficient data to compare chosen and unchosen bigram typing times.")

    # Box plot to visualize typing times for chosen vs. unchosen bigrams
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=['Chosen']*len(chosen_times) + ['Unchosen']*len(unchosen_times),
                y=pd.concat([chosen_times, unchosen_times]))
    plt.title('Typing Times for Chosen vs Unchosen Bigrams')
    plt.ylabel('Typing Time (ms)')
    plt.savefig(os.path.join(output_plots_folder, output_filename1))
    plt.close()

    # 2. Do shorter typing times correspond to higher absolute slider values?
    valid_data = valid_comparisons.dropna(subset=['chosen_bigram_time', 'sliderValue'])
    typing_times = valid_data['chosen_bigram_time']
    abs_slider_values = valid_data['sliderValue'].abs()

    if len(typing_times) > 0:
        correlation, p_value = stats.spearmanr(typing_times, abs_slider_values)
        
        print("\n2. Correlation between typing times and absolute slider values:")
        print(f"Spearman's rank correlation coefficient: {correlation}")
        print(f"p-value: {p_value}")
        
        if p_value < 0.05:
            if correlation < 0:
                print("There is a significant negative correlation: shorter typing times tend to correspond to higher absolute slider values.")
            else:
                print("There is a significant positive correlation: longer typing times tend to correspond to higher absolute slider values.")
        else:
            print("There is no significant correlation between typing times and absolute slider values.")

    else:
        print("Insufficient data to analyze correlation between typing times and absolute slider values.")

    # Scatter plot to visualize typing time difference vs. slider value
    plt.figure(figsize=(10, 6))
    typing_time_diff = valid_comparisons['chosen_bigram_time'] - valid_comparisons['unchosen_bigram_time']
    plt.scatter(valid_comparisons['sliderValue'], typing_time_diff, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Slider Value')
    plt.ylabel('Typing Time Difference (Chosen - Unchosen) in ms')
    plt.title('Typing Time Difference vs. Slider Value')
    plt.savefig(os.path.join(output_plots_folder, output_filename2), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Is there a bias to the left (negative) vs. right (positive) slider values?
    slider_values = bigram_data['sliderValue'].dropna()

    if len(slider_values) > 0:
        statistic, p_value = stats.wilcoxon(slider_values)
        
        print("\n3. Analysis of slider value bias:")
        print(f"Wilcoxon signed-rank test statistic: {statistic}")
        print(f"p-value: {p_value}")
        
        if p_value < 0.05:
            if slider_values.median() < 0:
                print("There is a significant bias towards negative (left) slider values.")
            else:
                print("There is a significant bias towards positive (right) slider values.")
        else:
            print("There is no significant bias in slider values towards either direction.")

        # Additional information
        print(f"Median slider value: {slider_values.median():.2f}")
        print(f"Percentage of negative slider values: {(slider_values < 0).mean()*100:.2f}%")
        print(f"Percentage of positive slider values: {(slider_values > 0).mean()*100:.2f}%")
    else:
        print("Insufficient data to analyze slider value bias.")

    # Return the statistics for further use if needed
    typing_time_stats = {
        'total_rows': total_rows,
        'valid_chosen_times': valid_chosen_times,
        'valid_unchosen_times': valid_unchosen_times,
        'faster_chosen_count': faster_chosen_count,
        'total_valid_comparisons': total_valid_comparisons,
        'chosen_unchosen_test': {'statistic': statistic if 'statistic' in locals() else None, 
                                 'p_value': p_value if 'p_value' in locals() else None},
        'typing_time_slider_correlation': {'correlation': correlation if 'correlation' in locals() else None,
                                           'p_value': p_value if 'p_value' in locals() else None},
        'slider_bias_test': {'statistic': statistic if 'statistic' in locals() else None, 
                             'p_value': p_value if 'p_value' in locals() else None}
    }
    return typing_time_stats

def analyze_within_user_bigram_times(bigram_data):    
    """
    Analyze bigram typing times within users to find significantly different typing times across bigrams,
    considering both chosen and unchosen times.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data

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
        
        #print("\nBigrams involved in significant differences (sorted by frequency):")
        #for bigram, count in bigram_occurrences.most_common():
        #    print(f"{bigram}: {count} times")
    else:
        print("\nNo significant differences in typing times were found between bigrams.")

    return {
        'user_significant_pairs': user_significant_pairs,
        'significant_pairs_count': significant_pairs_count,
        'total_significant_differences': total_significant_differences,
        'users_with_differences': users_with_differences,
        'bigram_occurrences': dict(bigram_occurrences) if 'bigram_occurrences' in locals() else {}
    }

def plot_median_bigram_times(bigram_data, output_plots_folder, output_filename='bigram_times_barplot.png'):
    """
    Generate and save bar plot for median bigram typing times with horizontal x-axis labels.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    - output_filename: String name of the output plot file
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

    plt.savefig(os.path.join(output_plots_folder, output_filename), dpi=300, bbox_inches='tight')
    plt.close()

    #print(f"bigram_median_times_barplot_with_mad plot saved in {output_plots_folder}")

###################################
# Improbable Bigram Choice Analysis
###################################

def analyze_easy_choices(bigram_data, easy_choice_pairs, threshold=1):
    # Create a dictionary for quick lookup of improbable pairs
    improbable_pair_dict = {frozenset(pair): pair for pair in easy_choice_pairs}

    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    bigram_data = bigram_data.copy()

    # Identify improbable pairs and choices
    bigram_data['bigram_set'] = bigram_data.apply(lambda row: frozenset([row['chosen_bigram'], row['unchosen_bigram']]), axis=1)
    bigram_data['improbable_pair'] = bigram_data['bigram_set'].map(improbable_pair_dict)
    
    # Filter out rows where improbable_pair is NaN (not in easy_choice_pairs)
    bigram_data_filtered = bigram_data[bigram_data['improbable_pair'].notna()].copy()

    # Identify improbable choices only for the filtered data
    bigram_data_filtered['improbable_choice'] = bigram_data_filtered.apply(
        lambda row: row['chosen_bigram'] == row['improbable_pair'][1], axis=1
    )

    # Count improbable choices for each user
    user_improbable_counts = bigram_data_filtered.groupby('user_id')['improbable_choice'].sum()

    # Calculate overall statistics
    total_users = bigram_data_filtered['user_id'].nunique()
    total_improbable_pairs_possible = len(easy_choice_pairs) * total_users
    total_improbable_choices = bigram_data_filtered['improbable_choice'].sum()
    total_suspicious = user_improbable_counts[user_improbable_counts >= threshold].shape[0]

    # Print improbable choice analysis results
    print("\n____ Improbable Bigram Choice Analysis ____\n")
    print(f"Improbable choices made: {total_improbable_choices} of {total_improbable_pairs_possible} ({(total_improbable_choices / total_improbable_pairs_possible) * 100:.2f}%)")
    print(f"Users with >={threshold} improbable choices: {total_suspicious} of {total_users} ({(total_suspicious / total_users) * 100:.2f}%)")

    if total_suspicious > 0:
        suspicious_users = bigram_data_filtered[bigram_data_filtered['improbable_choice']]['user_id'].value_counts().rename_axis('user_id').reset_index(name='improbable_choice_count')
        print("\nTop 10 users with highest improbable choice counts:")
        print(suspicious_users.sort_values('improbable_choice_count', ascending=False).head(10))

    print("\nFrequency of improbable bigrams chosen and their probable counterparts:")
    print(bigram_data_filtered[bigram_data_filtered['improbable_choice']]['improbable_pair'].value_counts())

    # Typing Times: Improbable vs Probable Choices
    improbable_times = bigram_data_filtered[bigram_data_filtered['improbable_choice']]['chosen_bigram_time']
    probable_times = bigram_data_filtered[~bigram_data_filtered['improbable_choice']]['chosen_bigram_time']
    # Remove non-finite values
    improbable_times = improbable_times[np.isfinite(improbable_times)]
    probable_times = probable_times[np.isfinite(probable_times)]

    # Print some diagnostic information
    print(f"\nNumber of improbable choices: {len(improbable_times)}")
    print(f"Number of probable choices: {len(probable_times)}")

    if len(improbable_times) > 0 and len(probable_times) > 0:
        improbable_median = improbable_times.median()
        probable_median = probable_times.median()
        improbable_mad = np.median(np.abs(improbable_times - improbable_median))
        probable_mad = np.median(np.abs(probable_times - probable_median))
        
        print(f"\nImprobable choice median (MAD) typing time: {improbable_median:.2f} ({improbable_mad:.2f}) ms")
        print(f"Probable choice median (MAD) typing time: {probable_median:.2f} ({probable_mad:.2f}) ms")
        
        # Perform Wilcoxon signed-rank test
        if len(improbable_times) > 1 and len(probable_times) > 1:
            # We need to ensure the samples are paired and of equal length
            min_length = min(len(improbable_times), len(probable_times))
            wilcoxon_result = stats.wilcoxon(improbable_times[:min_length], probable_times[:min_length])
            print(f"Wilcoxon signed-rank test results - Statistic: {wilcoxon_result.statistic:.4f}, p-value: {wilcoxon_result.pvalue:.4f}")
        else:
            print("Cannot perform Wilcoxon signed-rank test: Not enough data points")
            wilcoxon_result = None
    else:
        print("Not enough data to calculate statistics for improbable or probable choices")
        improbable_median = probable_median = improbable_mad = probable_mad = np.nan
        wilcoxon_result = None

    typing_time_results = {
        'improbable_median': improbable_median,
        'probable_median': probable_median,
        'improbable_mad': improbable_mad,
        'probable_mad': probable_mad,
        'wilcoxon_result': wilcoxon_result
    }

    # Analyze inconsistent pairs
    inconsistent_pairs = bigram_data.groupby(['user_id', 'bigram_pair'])['chosen_bigram'].nunique() > 1
    inconsistent_counts = inconsistent_pairs[inconsistent_pairs].groupby('user_id').size()

    # Combine inconsistent and improbable data
    user_analysis = pd.DataFrame({
        'inconsistent_count': inconsistent_counts,
        'improbable_count': user_improbable_counts
    }).fillna(0)

    # Calculate proportions
    total_pairs_per_user = bigram_data.groupby('user_id')['bigram_pair'].nunique()
    user_analysis['inconsistent_proportion'] = user_analysis['inconsistent_count'] / total_pairs_per_user
    user_analysis['improbable_proportion'] = user_analysis['improbable_count'] / total_pairs_per_user

    # Calculate correlation
    correlation = user_analysis['inconsistent_proportion'].corr(user_analysis['improbable_proportion'])

    print("\n____ Relationship between Inconsistent Pairs and Improbable Choices ____\n")
    print(f"Correlation between proportion of inconsistent pairs and improbable choices: {correlation:.4f}")

    # Overall statistics
    total_pairs = total_pairs_per_user.sum()
    total_inconsistent = user_analysis['inconsistent_count'].sum()
    total_improbable = user_analysis['improbable_count'].sum()

    print(f"\nTotal bigram pairs: {total_pairs}")
    print(f"Total inconsistent pairs: {total_inconsistent} ({total_inconsistent/total_pairs:.2%})")
    print(f"Total improbable choices: {total_improbable} ({total_improbable/total_pairs:.2%})")

    # Users with both inconsistencies and improbable choices
    users_with_both = ((user_analysis['inconsistent_count'] > 0) & (user_analysis['improbable_count'] > 0)).sum()
    print(f"\nUsers with both inconsistencies and improbable choices: {users_with_both} "
          f"({users_with_both/len(user_analysis):.2%} of users)")

    # Top 10 users with highest combination of inconsistencies and improbable choices
    user_analysis['combined_score'] = user_analysis['inconsistent_proportion'] + user_analysis['improbable_proportion']
    top_10_users = user_analysis.nlargest(10, 'combined_score')

    print("\nTop 10 users with highest combination of inconsistencies and improbable choices:")
    #print(top_10_users[['inconsistent_count', 'inconsistent_proportion', 
    #                    'improbable_count', 'improbable_proportion', 'combined_score']])
    print(top_10_users[['inconsistent_count', 'improbable_count', 'combined_score']])

    return suspicious_users, user_improbable_counts, typing_time_results, user_analysis

#######################################
# Analyze bigram choice inconsistencies
#######################################

def analyze_choice_inconsistencies(bigram_data, output_tables_folder):
    """
    Analyze and report inconsistencies in bigram data, including median of slider values,
    and information about which bigram was preferred. Outputs results to a CSV file.
    Excludes pairs that are only presented once to each user.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_tables_folder: String path to the folder where the CSV file should be saved

    Returns:
    - inconsistency_stats: Dictionary containing inconsistency statistics
    """
    if bigram_data.empty:
        print("The bigram data is empty. No analysis can be performed.")
        return {}

    # Create a standardized pair representation
    bigram_data['std_pair'] = bigram_data.apply(lambda row: ','.join(sorted([row['bigram1'], row['bigram2']])), axis=1)

    # Group by user and bigram pair
    grouped = bigram_data.groupby(['user_id', 'std_pair'])

    # Find inconsistent pairs and calculate statistics
    pair_stats = []
    for (user_id, std_pair), group in grouped:
        chosen_bigrams = group['chosen_bigram'].unique()
        is_inconsistent = len(chosen_bigrams) > 1
        
        consistent_slider = group['sliderValue'] if not is_inconsistent else group[group['chosen_bigram'] == chosen_bigrams[0]]['sliderValue']
        inconsistent_slider = group['sliderValue'] if is_inconsistent else pd.Series([])
        
        pair_stats.append({
            'std_pair': std_pair,
            'user_id': user_id,
            'is_inconsistent': is_inconsistent,
            'consistent_slider_median': consistent_slider.median(),
            'inconsistent_slider_median': inconsistent_slider.median() if is_inconsistent else None,
            'most_preferred_bigram': chosen_bigrams[0],
            'bigram1': group['bigram1'].iloc[0],
            'bigram2': group['bigram2'].iloc[0]
        })

    pair_consistency = pd.DataFrame(pair_stats)
    
    # Aggregate statistics for each bigram pair
    pair_summary = pair_consistency.groupby('std_pair').agg({
        'is_inconsistent': 'sum',
        'user_id': 'count',
        'consistent_slider_median': 'median',
        'inconsistent_slider_median': lambda x: x[x.notna()].median(),
        'most_preferred_bigram': lambda x: x.mode().iloc[0] if len(x) > 0 else None,
        'bigram1': 'first',
        'bigram2': 'first'
    }).reset_index()

    pair_summary = pair_summary.rename(columns={
        'is_inconsistent': 'inconsistent_users',
        'user_id': 'total_users'
    })
    pair_summary['consistent_users'] = pair_summary['total_users'] - pair_summary['inconsistent_users']
    
    # Order bigrams based on consistent slider median
    def order_bigrams(row):
        if row['consistent_slider_median'] < 0:
            return f"{row['bigram1']},{row['bigram2']}"
        else:
            return f"{row['bigram2']},{row['bigram1']}"
    
    pair_summary['ordered_pair'] = pair_summary.apply(order_bigrams, axis=1)
    
    # Sort by total users
    pair_summary = pair_summary.sort_values('total_users', ascending=False)

    print("\n____ Bigram Choice Inconsistency Statistics ____\n")
    print(f"Unique bigram pairs with potential inconsistencies: {len(pair_summary)}")

    # Prepare data for the CSV file
    table_data = pair_summary[['ordered_pair', 'consistent_users', 'inconsistent_users', 'consistent_slider_median', 'inconsistent_slider_median', 'most_preferred_bigram']].copy()
    table_data['consistent_slider_median'] = table_data['consistent_slider_median'].round(2)
    table_data['inconsistent_slider_median'] = table_data['inconsistent_slider_median'].round(2)

    # Save the data to a CSV file
    csv_filename = os.path.join(output_tables_folder, 'bigram_consistency_statistics.csv')
    table_data.to_csv(csv_filename, index=False)
    print(f"\nConsistency and Slider Value Statistics saved to: {csv_filename}")

    # Print the first few rows of the data
    print("\nFirst few rows of Consistency and Slider Value Statistics:")
    print(table_data.head().to_string())

    # Return the statistics for further use if needed
    inconsistency_stats = {
        'all_pairs': table_data
    }
    return inconsistency_stats

def analyze_inconsistency_slider_relationship(bigram_data, output_plots_folder, output_filename1='inconsistency_slider_relationship.png', output_filename2='inconsistency_typing_time_relationship.png'):
    """
    Analyze the relationship between inconsistent choices, slider values, and typing times.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    - output_filename1: String filename of the plot of inconsistent choices vs. slider values
    - output_filename2: String filename of the plot of inconsistent choices vs. typing times

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
    plt.savefig(os.path.join(output_plots_folder, output_filename1), dpi=300, bbox_inches='tight')
    plt.close()

    # Create visualization for typing times
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_inconsistent', y='avg_typing_time', data=inconsistency_df)
    plt.title('Average Typing Times for Consistent vs Inconsistent Choices')
    plt.xlabel('Is Inconsistent')
    plt.ylabel('Average Typing Time (ms)')
    plt.savefig(os.path.join(output_plots_folder, output_filename2), dpi=300, bbox_inches='tight')
    plt.close()

    inconsistency_analysis_results = {
        'slider_value_results': slider_value_results,
        'typing_time_results': typing_time_results
    }

    return inconsistency_analysis_results

def plot_chosen_vs_unchosen_times(bigram_data, output_plots_folder, 
                                  output_filename1='chosen_vs_unchosen_times.png',
                                  output_filename2='chosen_vs_unchosen_times_scatter_regression.png',
                                  output_filename3='chosen_vs_unchosen_times_joint.png'):
    """
    Plot chosen vs. unchosen typing times.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    - output_filename1: String filename of the plot of chosen vs. unchosen bigram times
    - output_filename2: String filename of the plot of typing time difference vs. slider value
    - output_filename3: String filename of the plot of inconsistency analysis
    """
    plot_data = prepare_plot_data(bigram_data)
    
    ##########
    # BAR PLOT
    ##########
    plt.figure(figsize=(15, len(plot_data['bigram'].unique()) * 0.4))
    sns.barplot(x='time', y='bigram', hue='type', data=plot_data)
    
    plt.title('Chosen vs unchosen typing time for each bigram')
    plt.xlabel('Median typing time (ms)')
    plt.ylabel('Bigram')
    plt.legend(title='Bigram type')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, output_filename1), dpi=300, bbox_inches='tight')
    plt.close()

    ##############
    # SCATTER PLOT
    ##############
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
    plt.savefig(os.path.join(output_plots_folder, output_filename2), dpi=300, bbox_inches='tight')
    plt.close()

    ####################
    # SCATTER JOINT PLOT
    #################### 
    # Pivot the data to have chosen and unchosen times as separate columns
    #plot_data_wide = plot_data.pivot(index='bigram', columns='type', values='time').reset_index()

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
    g.savefig(os.path.join(output_plots_folder, output_filename3), dpi=300, bbox_inches='tight')
    plt.close()

    #print(f"chosen_vs_unchosen_times_joint plot saved in {output_plots_folder}")
    #print(f"Correlation between chosen and unchosen times: {correlation:.2f}")

#################################
# Analyze only consistent choices
#################################

def filter_consistent_choices(bigram_data):
    # Group by user_id and bigram_pair to find duplicates
    grouped = bigram_data.groupby(['user_id', 'bigram_pair'])
    
    # Filter for pairs with duplicates
    duplicate_pairs = grouped.filter(lambda x: len(x) > 1)
    
    # Count unique duplicate pairs
    duplicate_pair_count = duplicate_pairs['bigram_pair'].nunique()
    
    # Count total duplicate choices
    total_duplicate_choices = len(duplicate_pairs)
    
    # Filter for consistent choices among duplicate pairs
    consistent_choices = grouped.filter(lambda x: len(x) > 1 and x['chosen_bigram'].nunique() == 1)
    
    # Calculate summary statistics for consistent choices
    summary_stats = {
        'total_rows': len(consistent_choices),
        'valid_chosen_times': consistent_choices['chosen_bigram_time'].notna().sum(),
        'valid_unchosen_times': consistent_choices['unchosen_bigram_time'].notna().sum(),
        'faster_chosen': (consistent_choices['chosen_bigram_time'] < consistent_choices['unchosen_bigram_time']).sum(),
        'total_comparisons': (consistent_choices['chosen_bigram_time'].notna() & consistent_choices['unchosen_bigram_time'].notna()).sum()
    }
    
    print("\n____ Consistent Bigram Pair Analysis ____\n")

    # Calculate statistics for each bigram pair
    bigram_pair_stats = []
    for (bigram_pair, group) in consistent_choices.groupby('bigram_pair'):
        total_users = group['user_id'].nunique()
        total_choices = len(group)
        chosen_counts = group['chosen_bigram'].value_counts().to_dict()
        bigram1, bigram2 = bigram_pair.split(', ')
        bigram_pair_stats.append({
            'Bigram Pair': bigram_pair,
            'Total Users': total_users,
            'Total Choices': total_choices,
            'First Bigram': chosen_counts.get(bigram1, 0),
            'Second Bigram': chosen_counts.get(bigram2, 0)
        })
    
    # Convert to DataFrame and sort
    bigram_pair_stats = pd.DataFrame(bigram_pair_stats)
    bigram_pair_stats = bigram_pair_stats.sort_values('Total Users', ascending=False)

    print(f"{len(bigram_pair_stats)} bigram pairs and choice statistics:")
    
    # Print header
    print(f"{'Bigram Pair':<15}{'Total Users':<15}{'Total Choices':<15}{'First Bigram':<15}{'Second Bigram':<15}")
    print("-" * 75)
    
    # Print data rows
    for _, row in bigram_pair_stats.iterrows():
        print(f"{row['Bigram Pair']:<15}{row['Total Users']:<15}{row['Total Choices']:<15}{row['First Bigram']:<15}{row['Second Bigram']:<15}")

    print(f"\nNumber of unique duplicate bigram pairs: {duplicate_pair_count}")
    print(f"Total choices for duplicate bigram pairs: {total_duplicate_choices}")
    print(f"Consistent choices for duplicate bigram pairs: {summary_stats['total_rows']} ({summary_stats['total_rows']/total_duplicate_choices*100:.2f}% of duplicate choices)")

    return consistent_choices, duplicate_pair_count, total_duplicate_choices, summary_stats, bigram_pair_stats


# Main execution
if __name__ == "__main__":

    preprocess = True
    if preprocess:

        ##########################
        # Load and preprocess data
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
        easy_choice_pairs_file = os.path.join(parent_dir, 'bigram_tables', 'bigram_4pairs_easy_choices_LH.csv')
        easy_choice_pairs = load_easy_choice_pairs(easy_choice_pairs_file)

        # Load, combine, and save the data
        data = load_and_preprocess_data(input_folder, output_tables_folder, verbose=False)
        result = process_bigram_data(data, easy_choice_pairs, output_tables_folder, verbose=False)
        bigram_data, consistent_choices, inconsistent_choices, probable_choices, improbable_choices = (
            result['bigram_data'],
            result['consistent_choices'],
            result['inconsistent_choices'],
            result['probable_choices'],
            result['improbable_choices']
        )

        analyze_bigram_data(result, output_tables_folder, output_plots_folder)

        #bigram_pairs_file = os.path.join(parent_dir, 'bigram_tables', 'bigram_27pairs_11tests_11swap_5easy_LH.csv')
        #num_bigram_pairs, bigram_pairs_df = load_bigram_pairs(bigram_pairs_file)

        #####################
        # Process bigram data
        #####################

        """
        
        plot_chosen_vs_unchosen_times(bigram_data, output_plots_folder, 
                                      output_filename1='chosen_vs_unchosen_times.png',
                                      output_filename2='chosen_vs_unchosen_times_scatter_regression.png',
                                      output_filename3='chosen_vs_unchosen_times_joint.png')

        #############################
        # Analyze bigram typing times
        #############################
        typing_time_stats = analyze_typing_times(bigram_data, output_plots_folder, 
                                                 output_filename1='chosen_vs_unchosen_times.png', 
                                                 output_filename2='typing_time_diff_vs_slider_value.png')

        # Analyze within-user bigram typing times and relationships
        within_user_stats = analyze_within_user_bigram_times(bigram_data)

        plot_median_bigram_times(bigram_data, output_plots_folder, 
                                 output_filename='bigram_times_barplot.png')

        ############################
        # Analyze improbable choices
        ############################
        if easy_choice_pairs:
            suspicious_users, user_improbable_counts, typing_time_results, user_analysis = analyze_easy_choices(bigram_data, easy_choice_pairs, threshold=2)
        else:
            print("Skipping easy choice bigram analysis due to missing or invalid easy choice pairs data.")            

        #######################################
        # Analyze bigram choice inconsistencies
        #######################################
        bigram_choice_inconsistency_stats = analyze_choice_inconsistencies(bigram_data, output_plots_folder)

        # Analyze the relationship between inconsistent choices and slider values
        inconsistency_slider_stats = analyze_inconsistency_slider_relationship(bigram_data, output_plots_folder, 
                                                                               output_filename1='inconsistency_slider_relationship.png', 
                                                                               output_filename2='inconsistency_typing_time_relationship.png')

        #################################
        # Analyze only consistent choices
        #################################

        print("\n\n=============== Consistent Choice Analysis ===============\n")

        # Filter consistent choices for duplicate bigram pairs
        consistent_bigram_data, duplicate_pair_count, total_duplicate_choices, summary_stats, bigram_pair_stats = filter_consistent_choices(bigram_data)
        
        plot_chosen_vs_unchosen_times(consistent_bigram_data, output_plots_folder, 
                                      output_filename1='consistent_chosen_vs_unchosen_times.png',
                                      output_filename2='consistent_chosen_vs_unchosen_times_scatter_regression.png',
                                      output_filename3='consistent_chosen_vs_unchosen_times_joint.png')

        print(f"Total choices: {len(bigram_data)}")
        print(f"Number of unique duplicate bigram pairs: {duplicate_pair_count}")
        print(f"Total choices for duplicate bigram pairs: {total_duplicate_choices}")
        print(f"Consistent choices for duplicate bigram pairs: {summary_stats['total_rows']} ({summary_stats['total_rows']/total_duplicate_choices*100:.2f}% of duplicate choices)")
        
        # Analyze bigram typing times for consistent choices
        ####################################################
        consistent_typing_time_stats = analyze_typing_times(consistent_bigram_data, output_plots_folder, 
                                                            output_filename1='consistent_chosen_vs_unchosen_times.png', 
                                                            output_filename2='consistent_typing_time_diff_vs_slider_value.png')
        consistent_within_user_stats = analyze_within_user_bigram_times(consistent_bigram_data)
        plot_median_bigram_times(consistent_bigram_data, output_plots_folder, 
                                 output_filename='consistent_bigram_times_barplot.png')
            
        print("\n=============== Consistent Choice Analysis Complete ===============\n\n")
        
        """
