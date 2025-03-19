
""" Process experiment data -- See README.md """

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from typing import Dict, Any
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)  

#######################################
# Functions to load and preprocess data
#######################################
def load_and_combine_data(input_folder, output_tables_folder, verbose=False):
    """
    Load and combine data from multiple CSV files in a folder of subfolders.

    Parameters:
    - input_folder: path to the folder containing folders of CSV files
    - output_tables_folder: path to the folder where the combined data will be saved

    Returns:
    - combined_df: DataFrame with combined data
    """
    #print(f"Loading data from {input_folder}...")
    dataframes = []
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                if verbose:
                    print(f"Processing file: {file_path}")
                df = pd.read_csv(file_path)
                
                # Extract user ID from filename (assuming format: experiment_data_USERID_*.csv)
                user_id = filename.split('_')[2]
                df['user_id'] = user_id
                df['filename'] = filename
                
                # Add the subfolder name
                subfolder = os.path.relpath(root, input_folder)
                df['group_id'] = subfolder if subfolder != '.' else ''
                
                # Remove rows where 'trialId' contains 'intro-trial'
                df_filtered = df[~df['trialId'].str.contains("intro-trial", na=False)]
                if len(df_filtered) > 0:
                    dataframes.append(df_filtered)
    
    # Combine the dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded data from {len(dataframes)} files in {input_folder}")

    # Display information about the combined DataFrame
    if verbose:
        print(combined_df.info())
        nlines = 5
        print_headers = ['group_id', 'trialId', 'sliderValue', 'chosenBigram', 'unchosenBigram', 
                         'chosenBigramTime', 'unchosenBigramTime']
        display_information(combined_df, "original data", print_headers, nlines)

    # Save the combined DataFrame to a CSV file
    output_file = os.path.join(output_tables_folder, 'original_combined_data.csv')
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

    return combined_df

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
        
        print(f"Loaded {len(easy_choice_pairs)} bigram pairs from {file_path}.")

        return easy_choice_pairs
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error loading easy choice pairs: {str(e)}")
        return []

def load_bigram_pairs(file_path):
    """
    Load bigram pairs.

    Parameters:
    - file_path: String, path to the CSV file containing bigram pairs

    Returns:
    - bigram_pairs_df: List of tuples, each containing a pair of bigrams
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, header=None)
        
        bigram_pairs = list(df.itertuples(index=False, name=None))

        return bigram_pairs
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error loading pairs: {str(e)}")
        return []

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

###########################################################################
# Functions to filter users by inconsistent or improbable choice thresholds,
# consecutive same-side selections, and number of near-zero selections
###########################################################################
def filter_users(data, user_stats, improbable_threshold=np.inf, inconsistent_threshold=np.inf,
                 n_repeat_sides=10, percent_close_to_zero=25, distance_close_to_zero=10,
                 filter_by_strong_inconsistencies=False, zero_threshold=20):
    """
    Filter out users based on various criteria.
    
    Parameters:
    - data: DataFrame to filter
    - user_stats: DataFrame containing user statistics
    - improbable_threshold: Maximum number of improbable choices allowed
    - inconsistent_threshold: Maximum percentage of inconsistent choices allowed
    - n_repeat_sides: Number of consecutive same-side selections to flag a user
    - percent_close_to_zero: Percentage of total trials that are within distance_close_to_zero of zero to flag a user
    - distance_close_to_zero: Distance from zero to consider as "close to zero"
    - filter_by_strong_inconsistencies: If True, only consider inconsistent choices that are greater than zero_threshold from zero
    - zero_threshold: Distance from zero to consider as "strong" inconsistency
    
    Returns:
    - Filtered DataFrame and set of valid user IDs
    """
    print("\n____ Filtering users ____\n")
    initial_users = len(data['user_id'].unique())
    initial_rows = len(data)
    
    all_users = set(data['user_id'].unique())
    valid_users = all_users.copy()
    
    # Track separate impacts for each filter
    user_impacts = {
        'improbable': set(),
        'inconsistent': set(),
        'slider': set(),
    }
    
    # Filter by improbable choices
    if improbable_threshold != np.inf:
        problematic_users = set(user_stats[user_stats['improbable_choices_2x'] > improbable_threshold]['user_id'])
        user_impacts['improbable'] = problematic_users
        valid_users -= problematic_users
        
        print(f"Users removed due to improbable choices > {improbable_threshold}: {len(problematic_users)}")
        if len(problematic_users) > 0 and len(problematic_users) <= 10:
            print(f"Removed users: {problematic_users}")
    
    # Filter by inconsistent choices
    if inconsistent_threshold != np.inf:
        if filter_by_strong_inconsistencies:
            print(f"Filtering by strong inconsistencies (|sliderValue| > {zero_threshold})")
            
            # We need to calculate this per user
            problematic_users = set()
            
            # Create standardized bigram pair representation as tuples if not already present
            if 'std_bigram_pair' not in data.columns:
                data_with_pairs = data.copy()
                data_with_pairs['std_bigram_pair'] = data_with_pairs.apply(
                    lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
            else:
                data_with_pairs = data
            
            # Process each user
            for user_id, user_data in data_with_pairs.groupby('user_id'):
                # Group by bigram pairs
                grouped = user_data.groupby('std_bigram_pair')
                
                # Count total multi-presentation pairs
                total_multi_pairs = 0
                strong_inconsistent_pairs = 0
                
                for _, group in grouped:
                    # Skip if only one presentation
                    if len(group) <= 1:
                        continue
                    
                    total_multi_pairs += 1
                    
                    # Check if inconsistent
                    is_consistent = len(set(group['chosenBigram'])) == 1
                    
                    if not is_consistent:
                        # Check if any inconsistent choice has abs(sliderValue) > zero_threshold
                        has_strong_inconsistency = any(abs(val) > zero_threshold for val in group['sliderValue'])
                        if has_strong_inconsistency:
                            strong_inconsistent_pairs += 1
                
                # Calculate percentage of strong inconsistent pairs
                if total_multi_pairs > 0:
                    inconsistency_pct = (strong_inconsistent_pairs / total_multi_pairs) * 100
                    if inconsistency_pct > inconsistent_threshold:
                        problematic_users.add(user_id)
            
            user_impacts['inconsistent'] = problematic_users
            valid_users -= problematic_users
            
            print(f"\nUsers removed due to strong inconsistent choices > {inconsistent_threshold}%: {len(problematic_users)}")
            if len(problematic_users) > 0 and len(problematic_users) <= 10:
                print(f"Removed users: {problematic_users}")
        else:
            # Original method using user_stats
            user_inconsistency_pcts = (user_stats['inconsistent_choices'] / user_stats['total_consistency_choices'] * 100)
            problematic_users = set(user_stats[user_inconsistency_pcts > inconsistent_threshold]['user_id'])
            user_impacts['inconsistent'] = problematic_users
            valid_users -= problematic_users
            
            print(f"\nUsers removed due to inconsistent choices > {inconsistent_threshold}%: {len(problematic_users)}")
            if len(problematic_users) > 0 and len(problematic_users) <= 10:
                print(f"Removed users: {problematic_users}")
    
    # Filter by problematic slider behavior
    problematic_users, slider_stats = identify_problematic_slider_behavior(
        data, n_repeat_sides, percent_close_to_zero, distance_close_to_zero
    )
    user_impacts['slider'] = problematic_users
    valid_users -= problematic_users
    
    print(f"\nUsers removed due to problematic slider behavior: {len(problematic_users)}")
    print(f"  - Same side consecutive selections ({n_repeat_sides}+): {len(slider_stats['repeated_side_users'])}")
    print(f"  - Frequent zero values ({percent_close_to_zero}%+): {len(slider_stats['close_to_zero_users'])}")
    if len(problematic_users) > 0 and len(problematic_users) <= 10:
        print(f"Removed users: {problematic_users}")
    
    # Filter data to keep only valid users
    filtered_data = data[data['user_id'].isin(valid_users)]
    
    # Print summary
    removed_users = initial_users - len(valid_users)
    removed_rows = initial_rows - len(filtered_data)
    
    print(f"\nFiltering summary:")
    print(f"Initial users: {initial_users}, Remaining users: {len(valid_users)}")
    print(f"Users removed: {removed_users} ({removed_users/initial_users:.1%})")
    print(f"Rows removed: {removed_rows} ({removed_rows/initial_rows:.1%})")
    
    return filtered_data, valid_users

def filter_data_rows(data, filter_letters=None, filter_single_presentations=False, 
                     filter_inconsistent_choices_away_from_zero=False, zero_threshold=20):
    """
    Filter rows of data based on various criteria.
    
    Parameters:
    - data: DataFrame to filter
    - filter_letters: List of letters to filter out from bigrams, or None to skip letter filtering
    - filter_single_presentations: If True, filter out bigram pairs that appear only once per user
    - filter_inconsistent_choices_away_from_zero: If True, filter out inconsistent choices where slider values are away from zero
    - zero_threshold: Distance from zero to consider as "close to zero"
    
    Returns:
    - Filtered DataFrame
    """
    print("\n____ Filtering data rows ____\n")
    data_filtered = data.copy()
    initial_rows = len(data_filtered)
    
    # Create standardized bigram pair representation as tuples if not already present
    if 'std_bigram_pair' not in data_filtered.columns:
        data_filtered['std_bigram_pair'] = data_filtered.apply(
            lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
    
    # Filter by letters
    if filter_letters:
        print(f"Filtering out bigrams containing: {filter_letters}")
        initial_letter_rows = len(data_filtered)
        
        # Create a filter condition for both chosen and unchosen bigrams
        filter_condition = ~(
            data_filtered['chosenBigram'].str.contains('|'.join(filter_letters), case=False) |
            data_filtered['unchosenBigram'].str.contains('|'.join(filter_letters), case=False)
        )
        
        data_filtered = data_filtered[filter_condition].copy()
        removed_rows = initial_letter_rows - len(data_filtered)
        
        print(f"Rows before letter filtering: {initial_letter_rows}")
        print(f"Rows after letter filtering: {len(data_filtered)}")
        print(f"Rows removed: {removed_rows}")
    
    # Filter single presentations
    if filter_single_presentations:
        print("Filtering out single-presentation bigram pairs per user")
        initial_single_rows = len(data_filtered)
        
        # Group by user_id and standardized bigram pair
        grouped_data = data_filtered.groupby(['user_id', 'std_bigram_pair'])
        
        # Calculate the size of each group
        group_sizes = grouped_data.size()
        
        # Keep only groups with size >= 2
        valid_groups = group_sizes[group_sizes >= 2].reset_index()[['user_id', 'std_bigram_pair']]
        
        # Create a merged DataFrame to filter the original data
        data_filtered = pd.merge(
            data_filtered, 
            valid_groups, 
            on=['user_id', 'std_bigram_pair'],
            how='inner'
        )
        
        removed_rows = initial_single_rows - len(data_filtered)
        
        print(f"Rows before single-presentation filtering: {initial_single_rows}")
        print(f"Rows after single-presentation filtering: {len(data_filtered)}")
        print(f"Rows removed: {removed_rows}")
    
    # Filter inconsistent choices away from zero
    if filter_inconsistent_choices_away_from_zero:
        print(f"\nFiltering out inconsistent choices with slider values away from zero (threshold: ±{zero_threshold})")
        initial_inconsistent_rows = len(data_filtered)
        
        # Group by user_id and standardized bigram pair
        grouped_data = data_filtered.groupby(['user_id', 'std_bigram_pair'])
        
        # Create a list to store valid groups
        valid_groups = []
        
        for (user_id, std_bigram_pair), group in grouped_data:
            # Skip if only one presentation (consistency can't be determined)
            if len(group) <= 1:
                valid_groups.append((user_id, std_bigram_pair))
                continue
            
            # Check if choices are inconsistent
            is_consistent = len(set(group['chosenBigram'])) == 1
            
            if is_consistent:
                # Keep all consistent choices
                valid_groups.append((user_id, std_bigram_pair))
            else:
                # For inconsistent choices, if all slider values have absolute value less than threshold, keep this group
                if all(abs(val) <= zero_threshold for val in group['sliderValue']):
                    valid_groups.append((user_id, std_bigram_pair))
        
        # Convert valid groups to DataFrame for filtering
        if valid_groups:
            valid_groups_df = pd.DataFrame(valid_groups, columns=['user_id', 'std_bigram_pair'])
            
            # Filter the original data to keep only valid groups
            data_filtered = pd.merge(
                data_filtered, 
                valid_groups_df, 
                on=['user_id', 'std_bigram_pair'],
                how='inner'
            )
        
        removed_rows = initial_inconsistent_rows - len(data_filtered)
        
        print(f"Rows before inconsistent-away-from-zero filtering: {initial_inconsistent_rows}")
        print(f"Rows after inconsistent-away-from-zero filtering: {len(data_filtered)}")
        print(f"Rows removed: {removed_rows}")
    
    total_removed = initial_rows - len(data_filtered)
    print(f"\nTotal rows removed by all filters: {total_removed} ({total_removed/initial_rows:.1%})")
    
    return data_filtered

def process_data(data, easy_choice_pairs, remove_pairs, output_tables_folder, verbose=True):
    """
    Process the bigram data from a DataFrame and create additional dataframes for specific subsets.
    Identify specified "easy choice" bigram pairs, and remove specified bigram pairs from all data.
    
    Parameters:
    - data: DataFrame with combined bigram data
    - easy_choice_pairs: List of tuples containing easy choice (probable, improbable) bigram pairs
    - remove_pairs: Tuple of bigram pairs to be removed from all data
    - output_tables_folder: str, path to the folder where the processed data will be saved
    - verbose: bool, if True, print additional information
    
    Returns:
    - dict: Dictionary containing various processed dataframes and user statistics
    """    
    print("\n____ Process data ____\n")

    # Create dictionaries for quick lookup of probable and improbable pairs
    probable_pairs = {(pair[0], pair[1]): True for pair in easy_choice_pairs}
    improbable_pairs = {(pair[1], pair[0]): True for pair in easy_choice_pairs}
    
    # Create standardized bigram pair representation as tuples
    data_filtered = data.copy()
    data_filtered['std_bigram_pair'] = data_filtered.apply(lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
    
    # Group the data by user_id and standardized bigram pair
    grouped_data = data_filtered.groupby(['user_id', 'std_bigram_pair'])
    
    # Calculate the size of each group
    group_sizes = grouped_data.size()
    
    # Initialize the result list
    result_list = []
    
    for (user_id, std_bigram_pair), group in grouped_data:
        # Unpack the tuple directly (no need to split)
        bigram1, bigram2 = std_bigram_pair
        
        # Check consistency (only for pairs that appear more than once)
        is_consistent = len(set(group['chosenBigram'])) == 1 if len(group) > 1 else None

        for _, row in group.iterrows():
            chosen_bigram = row['chosenBigram']
            unchosen_bigram = row['unchosenBigram']
            
            # Determine if the choice is probable or improbable
            is_probable = probable_pairs.get((chosen_bigram, unchosen_bigram), False)
            is_improbable = improbable_pairs.get((chosen_bigram, unchosen_bigram), False)
            
            result = pd.DataFrame({
                'group_id': [row['group_id']],
                'user_id': [user_id],
                'trialId': [row['trialId']],
                'bigram_pair': [std_bigram_pair],  # This is a tuple now
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
                'abs_sliderValue': [abs(row['sliderValue'])],
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
    consistent_choices = bigram_data[(bigram_data['is_consistent'] == True) & (bigram_data['group_size'] > 1)]
    inconsistent_choices = bigram_data[(bigram_data['is_consistent'] == False) & (bigram_data['group_size'] > 1)]
    probable_choices = bigram_data[bigram_data['is_probable'] == True]
    improbable_choices = bigram_data[bigram_data['is_improbable'] == True]
        
    # Calculate user statistics
    user_stats = pd.DataFrame()
    user_stats['user_id'] = bigram_data['user_id'].unique()
    user_stats = user_stats.set_index('user_id')
    
    user_stats['total_choices'] = bigram_data['user_id'].value_counts()
    user_stats['consistent_choices'] = consistent_choices['user_id'].value_counts()
    user_stats['inconsistent_choices'] = inconsistent_choices['user_id'].value_counts()
    user_stats['probable_choices'] = probable_choices['user_id'].value_counts()
    user_stats['improbable_choices'] = improbable_choices['user_id'].value_counts()

    # First, count how many times each user made an improbable choice for each bigram pair
    pair_counts = improbable_choices.groupby(['user_id', 'bigram_pair']).size().reset_index(name='count')
    # Then, count how many bigram pairs had a count of 2 (consistently improbable)
    improbable_choices_2x = pair_counts[pair_counts['count'] == 2].groupby('user_id').size()
    user_stats['improbable_choices_2x'] = improbable_choices_2x
    # Make sure to handle users who don't have any consistent improbable choices
    user_stats['improbable_choices_2x'] = user_stats['improbable_choices_2x'].fillna(0).astype(int)

    # Calculate total choices that could be consistent/inconsistent
    user_stats['total_consistency_choices'] = bigram_data[bigram_data['group_size'] > 1]['user_id'].value_counts()
    user_stats['num_easy_choice_pairs'] = len(easy_choice_pairs)

    # Fill NaN values with 0 for users who might not have any choices in a category
    user_stats = user_stats.fillna(0)
    
    # Ensure all columns are integers
    user_stats = user_stats.astype(int)
    
    # Reset index to make user_id a column again
    user_stats = user_stats.reset_index()

    # Display information about the DataFrames
    if verbose:
        # Display summary statistics
        print(bigram_data.describe())
        print(bigram_data.columns)
        
    # Save the DataFrames to CSV files
    bigram_data.to_csv(f"{output_tables_folder}/processed_bigram_data.csv", index=False)
    consistent_choices.to_csv(f"{output_tables_folder}/processed_consistent_choices.csv", index=False)
    inconsistent_choices.to_csv(f"{output_tables_folder}/processed_inconsistent_choices.csv", index=False)
    probable_choices.to_csv(f"{output_tables_folder}/processed_probable_choices.csv", index=False)
    improbable_choices.to_csv(f"{output_tables_folder}/processed_improbable_choices.csv", index=False)
    if isinstance(improbable_choices_2x, pd.Series):
        improbable_choices_2x.to_csv(f"{output_tables_folder}/processed_improbable_choices_2x.csv")
    user_stats.to_csv(f"{output_tables_folder}/processed_user_statistics.csv", index=False)
    
    print(f"Processed data saved to {output_tables_folder}")

    return {
        'bigram_data': bigram_data,
        'consistent_choices': consistent_choices,
        'inconsistent_choices': inconsistent_choices,
        'probable_choices': probable_choices,
        'improbable_choices': improbable_choices,
        'user_stats': user_stats
    }

def identify_problematic_slider_behavior(raw_data, n_repeat_sides=10, percent_close_to_zero=50, distance_close_to_zero=10):
    """
    Identify users with problematic slider behavior:
    1. Users who repeatedly select the same side (all positive or all negative, and abs value greater than distance_close_to_zero)
    2. Users who frequently select values very close to zero
    
    Parameters:
    - raw_data: DataFrame containing the original data with slider values
    - n_repeat_sides: Number of consecutive same-side selections to flag a user
    - percent_close_to_zero: Percentage of total trials that are within distance_close_to_zero of zero to flag a user
    - distance_close_to_zero: Distance from zero to consider as "close to zero"
    
    Returns:
    - problematic_users: Set of user IDs that exhibit problematic behavior
    - stats: Dictionary with statistics about filtered users
    """
    problematic_users = set()
    stats = {
        'repeated_side_users': set(),
        'close_to_zero_users': set(),
        'total_problematic_users': 0
    }
    
    # Process each user's data
    for user_id, user_data in raw_data.groupby('user_id'):
        # Sort by trialId to maintain temporal order
        user_data = user_data.sort_values('trialId')
        slider_values = user_data['sliderValue'].values
        
        # Check for repeated same-side selections (greater than distance_close_to_zero)
        pos_streak = 0
        neg_streak = 0
        
        for value in slider_values:
            if value > distance_close_to_zero:
                pos_streak += 1
                neg_streak = 0
            elif value < -distance_close_to_zero:
                neg_streak += 1
                pos_streak = 0
            else:
                pos_streak = 0
                neg_streak = 0
                
            if pos_streak >= n_repeat_sides or neg_streak >= n_repeat_sides:
                stats['repeated_side_users'].add(user_id)
                problematic_users.add(user_id)
                break
        
        # Check for frequent near-zero selections (percentage-based)
        total_trials = len(user_data)
        close_to_zero_count = sum(abs(value) <= distance_close_to_zero for value in slider_values)
        if (close_to_zero_count / total_trials * 100) >= percent_close_to_zero:
            stats['close_to_zero_users'].add(user_id)
            problematic_users.add(user_id)
    
    stats['total_problematic_users'] = len(problematic_users)
    
    return problematic_users, stats

def visualize_user_choices(user_stats, output_plots_folder, plot_label=""):
    """
    Create tall figures showing the number of consistent vs. inconsistent choices
    and probable vs. improbable choices per user as horizontal stacked bar plots.

    Parameters:
    - user_stats: DataFrame containing user statistics
    - output_plots_folder: String path to the folder where plots should be saved
    - plot_label: String prefix for the output filenames

    Returns:
    - None (saves figures to the specified folder)
    """
    def create_stacked_bar_plot(data, title, filename):
        users = data.index
        consistent = data['consistent_choices']
        inconsistent = data['inconsistent_choices']

        # Adjust the figsize to make the plot taller
        plt.figure(figsize=(15, max(10, len(data) * 0.5)))  # Dynamically set the height

        # Create the horizontal bar plot
        plt.barh(users, consistent, color='blue', label='Consistent Choices')
        plt.barh(users, inconsistent, left=consistent, color='orange', label='Inconsistent Choices')

        plt.title(title)
        plt.xlabel('Number of Choices')
        plt.ylabel('User ID')
        plt.legend(title='Choice Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_plots_folder, plot_label + filename))
        plt.close()

    # Sort users by consistent choices in descending order
    user_order = user_stats.sort_values('consistent_choices', ascending=False)['user_id']

    # Prepare and plot consistent vs. inconsistent choices
    consistent_data = user_stats.set_index('user_id').loc[user_order, ['consistent_choices', 'inconsistent_choices']]
    create_stacked_bar_plot(consistent_data, 'Consistent vs. Inconsistent Choices per User', 'consistent_vs_inconsistent_choices.png')

    # Prepare and plot probable vs. improbable choices
    probable_data = user_stats.set_index('user_id').loc[user_order, ['probable_choices', 'improbable_choices']]

    # Using matplotlib for stacked bar plot for probable vs improbable choices
    def create_probable_bar_plot(data, title, filename):
        users = data.index
        probable = data['probable_choices']
        improbable = data['improbable_choices']

        # Adjust the figsize to make the plot taller
        plt.figure(figsize=(15, max(10, len(data) * 0.5)))  # Dynamically set the height

        # Create the horizontal bar plot
        plt.barh(users, probable, color='blue', label='Probable Choices')
        plt.barh(users, improbable, left=probable, color='red', label='Improbable Choices')

        plt.title(title)
        plt.xlabel('Number of Choices')
        plt.ylabel('User ID')
        plt.legend(title='Choice Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_plots_folder, plot_label + filename))
        plt.close()

    create_probable_bar_plot(probable_data, 'Probable vs. Improbable Choices per User', 'probable_vs_improbable_choices.png')

    print(f"Visualization plots saved to {output_plots_folder}")

########################################################################
# Functions to score choices by slider values and choose winning bigrams
########################################################################
def score_user_choices_by_slider_values(filtered_users_data, output_tables_folder):
    """
    Score each user's choices by slider values and create a modified copy of bigram_data.
    
    For this study, each user makes a choice between two bigrams, two times, 
    by using a slider each time -- but this function generalizes to many pairwise choices.
    If a user chooses the same bigram every time, we take their median absolute slider value.
    If a user chooses different bigrams, we subtract the sums of the absolute values for each choice,
    and divide by the number of choices made for that bigram pair.
    In both cases, the score is the absolute value of the result divided by 100 (the maximum slider value).
    """
    print("\n____ Score choices by slider values ____\n")
    bigram_data = filtered_users_data['bigram_data']
    
    # Add diagnostic info
    print(f"Number of rows with null values:")
    for col in bigram_data.columns:
        null_count = bigram_data[col].isnull().sum()
        if null_count > 0:
            print(f"{col}: {null_count} null values")

    # Group by user_id and bigram_pair
    grouped = bigram_data.groupby(['user_id', 'bigram_pair'], group_keys=False)

    # Apply the scoring to each group
    scored_bigram_data = grouped.apply(determine_score, include_groups=False).reset_index()

    # Reorder columns to match the specified order
    column_order = ['user_id', 'bigram_pair', 'bigram1', 'bigram2',
                    'chosen_bigram_winner', 'unchosen_bigram_winner', 
                    'chosen_bigram_time_median', 'unchosen_bigram_time_median',
                    'chosen_bigram_correct_total', 'unchosen_bigram_correct_total', 
                    'score', 'text', 'is_consistent', 'is_probable', 'is_improbable', 'group_size']
    
    scored_bigram_data = scored_bigram_data[column_order]
    
    print(f"Total rows in original bigram_data: {len(bigram_data)}")
    print(f"Total rows in scored_bigram_data (unique user_id and bigram_pair combinations): {len(scored_bigram_data)}")
    
    # Save scored_bigram_data as CSV
    output_file = os.path.join(output_tables_folder, 'scored_bigram_data.csv')
    scored_bigram_data.to_csv(output_file, index=False)
    print(f"Scored bigram data saved to {output_file}")
    
    # Display information about scored_bigram_data
    nlines = 5
    print_headers = ['user_id', 'bigram_pair', 'chosen_bigram_winner', 'unchosen_bigram_winner', 'score']
    display_information(scored_bigram_data, "scored bigram data", print_headers, nlines)
    
    return scored_bigram_data

def determine_score(group):
    """
    Determine the score and chosen/unchosen bigrams for a group of trials.

    (See score_user_choices_by_slider_values function docstring for a description.)
    """
    bigram1, bigram2 = group['bigram1'].iloc[0], group['bigram2'].iloc[0]
    chosen_bigrams = group['chosen_bigram']
    slider_values = group['sliderValue']
    group_size = len(group)

    # Handle empty/invalid groups
    if len(group) == 0:
        print(f"Warning: Empty group in determine_score for pair {(bigram1, bigram2)}")
        return pd.Series({
            'bigram1': bigram1,
            'bigram2': bigram2,
            'chosen_bigram_winner': chosen_bigram_winner,
            'unchosen_bigram_winner': unchosen_bigram_winner,
            'chosen_bigram_time_median': np.nan if len(group['chosen_bigram_time'].dropna()) == 0 else group['chosen_bigram_time'].median(skipna=True),
            'unchosen_bigram_time_median': np.nan if len(group['unchosen_bigram_time'].dropna()) == 0 else group['unchosen_bigram_time'].median(skipna=True),
            'chosen_bigram_correct_total': group['chosen_bigram_correct'].sum(skipna=True),
            'unchosen_bigram_correct_total': group['unchosen_bigram_correct'].sum(skipna=True),
            'score': score,
            'text': tuple(group['text'].unique()),
            'is_consistent': (len(set(chosen_bigrams)) == 1),
            'is_probable': group['is_probable'].iloc[0],
            'is_improbable': group['is_improbable'].iloc[0],
            'group_size': group_size
        })
    
    # Skip invalid values when calculating medians and sums
    valid_slider_values = [x for x in slider_values if not np.isnan(x)]
    if len(valid_slider_values) == 0:
        print(f"Warning: No valid slider values for pair {(bigram1, bigram2)}")
        
    if len(set(chosen_bigrams)) == 1:
        median_abs_slider_value = np.nan if not valid_slider_values else np.median([abs(x) for x in valid_slider_values])
        chosen_bigram_winner = chosen_bigrams.iloc[0]
        unchosen_bigram_winner = bigram2 if chosen_bigram_winner == bigram1 else bigram1
    else:
        valid_scores1 = [abs(x) for i,x in enumerate(slider_values) if chosen_bigrams.iloc[i] == bigram1 and not np.isnan(x)]
        valid_scores2 = [abs(x) for i,x in enumerate(slider_values) if chosen_bigrams.iloc[i] == bigram2 and not np.isnan(x)]
        
        sum1 = sum(valid_scores1)
        sum2 = sum(valid_scores2)
        median_abs_slider_value = abs(sum1 - sum2) / group_size if group_size > 0 else np.nan
        chosen_bigram_winner = bigram1 if sum1 >= sum2 else bigram2
        unchosen_bigram_winner = bigram2 if chosen_bigram_winner == bigram1 else bigram1

    score = median_abs_slider_value / 100 if not np.isnan(median_abs_slider_value) else np.nan

    return pd.Series({
        'bigram1': bigram1,
        'bigram2': bigram2,
        'chosen_bigram_winner': chosen_bigram_winner,
        'unchosen_bigram_winner': unchosen_bigram_winner,
        'chosen_bigram_time_median': np.nan if len(group['chosen_bigram_time'].dropna()) == 0 else group['chosen_bigram_time'].median(skipna=True),
        'unchosen_bigram_time_median': np.nan if len(group['unchosen_bigram_time'].dropna()) == 0 else group['unchosen_bigram_time'].median(skipna=True),
        'chosen_bigram_correct_total': group['chosen_bigram_correct'].sum(skipna=True),
        'unchosen_bigram_correct_total': group['unchosen_bigram_correct'].sum(skipna=True),
        'score': score,
        'text': tuple(group['text'].unique()),
        'is_consistent': (len(set(chosen_bigrams)) == 1),
        'is_probable': group['is_probable'].iloc[0],
        'is_improbable': group['is_improbable'].iloc[0],
        'group_size': group_size
    })

def choose_bigram_winners(scored_bigram_data, output_tables_folder):
    """
    Here we determine a winning bigram for each bigram pair across all users and all trials.
    If the winning bigram for every user is the same, the winning score is the median score.
    If the winning bigram differs across users, the winning score is calculated as follows:
    we subtract the sum of the absolute values of the scores for one bigram from the other,
    and divide by the number of choices made for either bigram in that bigram pair across the dataset.

    The function returns a modified copy of scored_bigram_data, called bigram_winner_data,
    that creates a single line per bigram_pair.
    """
    print("\n____ Choose bigram winners ____\n")
    
    initial_pairs = len(scored_bigram_data['bigram_pair'].unique())
    print(f"Initial number of unique bigram pairs: {initial_pairs}")

    # Group by bigram_pair
    grouped = scored_bigram_data.groupby(['bigram_pair'], group_keys=False)

    # Apply the winner determination to each bigram_pair group
    bigram_winner_data = grouped.apply(determine_winner, include_groups=False).reset_index()

    # Reorder and rename columns
    column_order = [
        'bigram_pair', 'winner_bigram', 'loser_bigram', 'median_score', 'mad_score', 
        'chosen_bigram_time_median', 'unchosen_bigram_time_median',
        'chosen_bigram_correct_total', 'unchosen_bigram_correct_total',
        'is_consistent', 'is_probable', 'is_improbable', 'text'
    ]    
    bigram_winner_data = bigram_winner_data[column_order]
    
    print(f"Total rows in scored_bigram_data: {len(scored_bigram_data)}")
    print(f"Total rows in bigram_winner_data (unique bigram_pairs): {len(bigram_winner_data)}")
    
    # Save bigram_winner_data as CSV
    output_file = os.path.join(output_tables_folder, 'bigram_winner_data.csv')
    bigram_winner_data.to_csv(output_file, index=False)
    print(f"Bigram winner data saved to {output_file}")
    
    # Display information about bigram_winner_data
    nlines = 1000
    print_headers = ['bigram_pair', 'winner_bigram', 'loser_bigram', 'median_score', 'mad_score']
    display_information(bigram_winner_data, "bigram winner data", print_headers, nlines)
    
    return bigram_winner_data

def determine_winner(group):
    """
    Determine the winning bigrams across all trials.

    (See choose_bigram_winners function docstring for a description.)
    """
    if len(group) == 0:
        print(f"Warning: Empty group found in determine_winner")
        return None
    
    bigram1, bigram2 = group['bigram1'].iloc[0], group['bigram2'].iloc[0]
    scores = group['score']
    chosen_bigrams = group['chosen_bigram_winner']
    chosen_bigram_time_medians = group['chosen_bigram_time_median']
    unchosen_bigram_time_medians = group['unchosen_bigram_time_median']
    chosen_bigram_correct_totals = group['chosen_bigram_correct_total']
    unchosen_bigram_correct_totals = group['unchosen_bigram_correct_total']

    # Check if either bigram was never chosen
    bigram1_chosen = any(chosen_bigrams == bigram1)
    bigram2_chosen = any(chosen_bigrams == bigram2)
    
    if not bigram1_chosen:
        print(f"Warning: No choices found for bigram1 '{bigram1}' in pair {(bigram1, bigram2)}")
    if not bigram2_chosen:
        print(f"Warning: No choices found for bigram2 '{bigram2}' in pair {(bigram1, bigram2)}")
    
    # Use safe median calculations 
    chosen_bigram_time_median1 = np.nan if not bigram1_chosen else np.median([x for i,x in enumerate(chosen_bigram_time_medians) if chosen_bigrams.iloc[i] == bigram1])
    chosen_bigram_time_median2 = np.nan if not bigram2_chosen else np.median([x for i,x in enumerate(chosen_bigram_time_medians) if chosen_bigrams.iloc[i] == bigram2])
    unchosen_bigram_time_median1 = np.nan if not bigram1_chosen else np.median([x for i,x in enumerate(unchosen_bigram_time_medians) if chosen_bigrams.iloc[i] == bigram1])
    unchosen_bigram_time_median2 = np.nan if not bigram2_chosen else np.median([x for i,x in enumerate(unchosen_bigram_time_medians) if chosen_bigrams.iloc[i] == bigram2])
    chosen_bigram_correct_total1 = np.sum([x for i,x in enumerate(chosen_bigram_correct_totals) if chosen_bigrams.iloc[i] == bigram1])
    chosen_bigram_correct_total2 = np.sum([x for i,x in enumerate(chosen_bigram_correct_totals) if chosen_bigrams.iloc[i] == bigram2])
    unchosen_bigram_correct_total1 = np.sum([x for i,x in enumerate(unchosen_bigram_correct_totals) if chosen_bigrams.iloc[i] == bigram1])
    unchosen_bigram_correct_total2 = np.sum([x for i,x in enumerate(unchosen_bigram_correct_totals) if chosen_bigrams.iloc[i] == bigram2])

    unique_chosen_bigrams = chosen_bigrams.unique()
    if len(unique_chosen_bigrams) == 1:
        # Only calculate mad_score if we have valid scores
        valid_scores = [abs(x) for x in scores if not np.isnan(x)]
        if valid_scores:
            median_score = np.median(valid_scores)
            mad_score = median_abs_deviation(valid_scores)
        else:
            median_score = np.nan
            mad_score = np.nan

        if unique_chosen_bigrams[0] == bigram1:
            bigram1_wins = True
            winner_bigram = bigram1
            loser_bigram = bigram2
        elif unique_chosen_bigrams[0] == bigram2:
            bigram1_wins = False
            winner_bigram = bigram2
            loser_bigram = bigram1
    else:
        sum1 = sum(abs(x) for i,x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram1)
        sum2 = sum(abs(x) for i,x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram2)
        median_score = abs(sum1 - sum2) / len(group)
        mad_list = []
        for i, x in enumerate(scores):
            if chosen_bigrams.iloc[i] == bigram1:
                mad_list.append(abs(x))
            elif chosen_bigrams.iloc[i] == bigram2:
                mad_list.append(-abs(x))
        mad_score = median_abs_deviation(mad_list)
        if sum1 >= sum2:
            bigram1_wins = True
            winner_bigram = bigram1
            loser_bigram = bigram2
            #mad_score = median_abs_deviation([abs(x) for i,x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram1])
        else:
            bigram1_wins = False
            winner_bigram = bigram2
            loser_bigram = bigram1
            #mad_score = median_abs_deviation([abs(x) for i,x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram2])

    if bigram1_wins:
        chosen_bigram_time_median = chosen_bigram_time_median1
        unchosen_bigram_time_median = unchosen_bigram_time_median1
        chosen_bigram_correct_total = chosen_bigram_correct_total1
        unchosen_bigram_correct_total = unchosen_bigram_correct_total1
    else:
        chosen_bigram_time_median = chosen_bigram_time_median2
        unchosen_bigram_time_median = unchosen_bigram_time_median2
        chosen_bigram_correct_total = chosen_bigram_correct_total2
        unchosen_bigram_correct_total = unchosen_bigram_correct_total2
    
    return pd.Series({
        'winner_bigram': winner_bigram,
        'loser_bigram': loser_bigram,
        'median_score': median_score,
        'mad_score': mad_score,
        'chosen_bigram_time_median': chosen_bigram_time_median,
        'unchosen_bigram_time_median': unchosen_bigram_time_median,
        'chosen_bigram_correct_total': chosen_bigram_correct_total,
        'unchosen_bigram_correct_total': unchosen_bigram_correct_total,
        'is_consistent': group['is_consistent'].all(),
        'is_probable': group['is_probable'].iloc[0],
        'is_improbable': group['is_improbable'].iloc[0],
        'group_size': group['group_size'].sum(),
        'text': tuple(group['text'].unique())
    })    

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

# Main execution
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Process bigram typing data')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    input_folder = os.path.join(config['data']['input_dir'])

    ###################
    # Filter parameters
    ###################
    # Filter users by inconsistent or improbable choice thresholds:
    # Improbable choices (left typing hand) in the easy_choice_pairs_file
    # good_choice,bad_choice
    # fr,vr  # same finger same direction 1 vs. 2 off home row (studies 1-7)
    # aq,zq  # same finger same direction 1 vs. 2 off home row (study 2A)
    # fd,ce  # 2 strongest fingers home row vs. skip home row with same finger (study 1)
    # df,qz  # 2 strongest fingers home row vs. skip home row with weakest finger (study 6)
    # df,ex  # 2 strongest fingers home row vs. skip home row (study 6)
    # fd,ez  # 2 strongest fingers home row vs. skip home row (study 6)
    # fd,qz  # 2 strongest fingers home row vs. skip home row with weakest finger (study 7)
    # df,wz  # 2 strongest fingers home row vs. skip home row with weakest fingers (study 7)
    filter_users_by_num_improbable_choices = config['process']['filter_users_by_num_improbable_choices']
    filter_users_by_percent_inconsistencies = config['process']['filter_users_by_percent_inconsistencies'] 
    filter_by_strong_inconsistencies = config['process']['filter_by_strong_inconsistencies'] # only works if filter_users_by_percent_inconsistencies is True
    if filter_users_by_num_improbable_choices:
        improbable_threshold = config['process']['improbable_threshold'] # at least one improbable choice
    else:
        improbable_threshold = np.Inf
    if filter_users_by_percent_inconsistencies:
        inconsistent_threshold = config['process']['inconsistent_threshold']  # Flag users with at least this percent of inconsistent choices
    else:
        inconsistent_threshold = np.Inf

    # Filter users by improbable slider behavior (streaks or close-to-zeros):
    n_repeat_sides = config['process']['n_repeat_sides']  # Flag users with consecutive same-side selections ((1/2)^10 ~= 0.0977%; (1/2)^20 = 1/1,048,576)
    percent_close_to_zero = config['process']['percent_close_to_zero']  # Flag users with this percent of selections close to zero
    distance_close_to_zero = config['process']['distance_close_to_zero']  # Consider values within this distance of zero as "close to zero" (max 100)

    # Filter rows with *per user* single-presentation bigram pairs or inconsistent bigram pair choices away from zero:
    filter_single_presentations = config['process']['filter_single_presentations']
    filter_inconsistent_choices_away_from_zero = config['process']['filter_inconsistent_choices_away_from_zero']
    zero_threshold = config['process']['zero_threshold']  # Consider values within this distance of zero as "close to zero" (max 100)

    # Filter rows with bigrams containing any of the following letters:
    filter_letters = config['process']['filter_letters'] #{'t', 'g', 'b'}

    ################################
    # Load, combine, and filter data
    ################################
    # Set the paths for input and output
    output_folder = os.path.join(input_folder, 'output')
    output_tables_folder = os.path.join(output_folder, 'tables')
    output_plots_folder = os.path.join(output_folder, 'plots')
    os.makedirs(output_tables_folder, exist_ok=True)
    os.makedirs(output_plots_folder, exist_ok=True)

    # Load improbable pairs
    # The "improbable" choice is choosing the bigram "vr" as easier to type than "fr",
    # and can be used as an option to filter users that may have chosen slider values randomly.
    current_dir = os.getcwd()  # Get the current working directory
    parent_dir = os.path.dirname(current_dir)  # Get the parent directory
    easy_choice_pairs_file = os.path.join(parent_dir, 'bigram_tables', 'bigram_easy_choice_pairs_LH_nocomments.csv')
    easy_choice_pairs = load_easy_choice_pairs(easy_choice_pairs_file)

    # Step 1: Load and combine the data
    raw_data = load_and_combine_data(input_folder, output_tables_folder, verbose=False)

    initial_users = len(raw_data['user_id'].unique())
    initial_trials = len(raw_data)
    initial_bigram_pairs = len(raw_data.apply(lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1).unique())
    print(f"\nInitial data:")
    print(f"Users: {initial_users}")
    print(f"Trials: {initial_trials}")
    print(f"Unique bigram pairs: {initial_bigram_pairs}")

    # Step 2: Process data without any filtering
    processed_data = process_data(raw_data, easy_choice_pairs, None, output_tables_folder, verbose=True)
    
    # Initial visualizations
    visualize_user_choices(processed_data['user_stats'], output_plots_folder, plot_label="initial_")

    # Step 3: FIRST Filter USERS
    filtered_users_data, valid_users = filter_users(
        raw_data,
        processed_data['user_stats'],
        improbable_threshold=improbable_threshold if filter_users_by_num_improbable_choices else np.inf,
        inconsistent_threshold=inconsistent_threshold if filter_users_by_percent_inconsistencies else np.inf,
        n_repeat_sides=n_repeat_sides,
        percent_close_to_zero=percent_close_to_zero,
        distance_close_to_zero=distance_close_to_zero,
        filter_by_strong_inconsistencies=filter_by_strong_inconsistencies,
        zero_threshold=zero_threshold
    )
    
    # Step 4: THEN Filter ROWS on the already user-filtered data
    filtered_data = filter_data_rows(
        filtered_users_data,
        filter_letters=filter_letters,
        filter_single_presentations=filter_single_presentations,
        filter_inconsistent_choices_away_from_zero=filter_inconsistent_choices_away_from_zero,
        zero_threshold=zero_threshold
    )

    # Step 5: Process the filtered data to create analysis-ready dataframes
    final_processed_data = process_data(filtered_data, easy_choice_pairs, None, output_tables_folder, verbose=True)

    final_trials = len(final_processed_data['bigram_data'])
    final_pairs = len(final_processed_data['bigram_data']['bigram_pair'].unique())
    print(f"\nFinal data after all filtering:")
    print(f"Users: {len(final_processed_data['bigram_data']['user_id'].unique())}")
    print(f"Trials remaining: {final_trials}")
    print(f"Trials removed: {initial_trials - final_trials}")
    print(f"Unique bigram pairs remaining: {final_pairs}")

    # Generate visualizations for the filtered data
    visualize_user_choices(final_processed_data['user_stats'], output_plots_folder, plot_label="filtered_")
    
    ################################
    # Score choices by slider values
    ################################
    """
    For this study, each user makes a choice between two bigrams, two times, by using a slider each time -- 
    but the score_user_choices_by_slider_values function generalizes to many pairwise choices.
    If a user chooses the same bigram every time, we take their median slider value.
    If a user chooses different bigrams, we subtract the sums of the absolute values for each choice.
    In both cases, the score is the absolute value of the result.
    """
    scored_data = score_user_choices_by_slider_values(final_processed_data, output_tables_folder)

    ########################
    # Choose winning bigrams
    ########################
    """
    Here we determine a winning bigram for each bigram pair across all users and all trials.
    If the winning bigram for every user is the same, the winning score is the median score.
    If the winning bigram differs across users, the winning score is calculated as follows:
    we subtract the sum of the absolute values of the scores for one bigram from the other,
    and divide by the number of choices the made for that bigram pair across the dataset.
    """
    bigram_winner_data = choose_bigram_winners(scored_data, output_tables_folder)