
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_preprocess_data(folder_path):
    """
    Load and preprocess (combine) data from multiple CSV files in a folder.
    """
    print("Loading and preprocessing data...")
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            print(f"Processing file: {filename}")
            df = pd.read_csv(os.path.join(folder_path, filename))
            
            # Extract user ID from filename (assuming format: experiment_data_USERID_*.csv)
            user_id = filename.split('_')[2]
            df['user_id'] = user_id
            df['filename'] = filename
            
            dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded and combined data from {len(dataframes)} files")
    
    return combined_df

def calculate_bigram_timing(data):
    """
    Calculate the fastest timing for each bigram in each repeated bigram pair (within each trial) for each user.
    """
    # Sort the data
    data = data.sort_values(['user_id', 'trialId', 'bigramPair', 'bigram', 'keydownTime'])
    
    def calculate_fastest_time(group):
        # Ensure the group is sorted by keydownTime
        group = group.sort_values('keydownTime')
        
        # Get the two letters of the bigram
        bigram = group['bigram'].iloc[0]
        first_letter, second_letter = bigram
        
        # Calculate timing for each complete bigram
        timings = []
        for i in range(len(group) - 1):
            if group['typedKey'].iloc[i] == first_letter and group['typedKey'].iloc[i+1] == second_letter:
                timings.append(group['keydownTime'].iloc[i+1] - group['keydownTime'].iloc[i])
        
        if not timings:
            print(f"Warning: No valid bigram sequences found for user_id, bigramPair, bigram: {group['user_id'].iloc[0]}, \"{group['bigramPair'].iloc[0]}\", {group['bigram'].iloc[0]}")
            return pd.Series({'timing_within_bigram': np.nan})
        
        # Return the fastest timing
        return pd.Series({'timing_within_bigram': min(timings)})

    # Group the data and apply the calculation function
    result = data.groupby(['user_id', 'trialId', 'bigramPair', 'bigram']).apply(calculate_fastest_time).reset_index()

    print("\nFinal bigram timing data shape (timing data for unique combinations of user, trial, bigram pair, and bigram):", result.shape)
    print(result.head())

    return result

def calculate_median_times(bigram_timings):
    """
    Calculate the median timing for each bigram in each bigram pair across users.
    """
    # Check if the DataFrame is empty
    if bigram_timings.empty:
        print("No bigram timings to calculate median from.")
        return pd.DataFrame()  # Return an empty DataFrame to avoid further errors
    
    # Check if 'bigramPair' and 'bigram' are in the DataFrame
    if not {'bigramPair', 'bigram'}.issubset(bigram_timings.columns):
        raise KeyError("'bigramPair' or 'bigram' columns are missing in the bigram timings.")

    # Calculate median times grouped by bigramPair and bigram
    median_times = bigram_timings.groupby(['bigramPair', 'bigram'])['timing_within_bigram'].median().unstack()
    
    print("\nMedian bigram timing shape (unique bigram pairs, unique bigrams across all pairs):", median_times.shape)
    print(median_times.head())

    return median_times

def get_chosen_bigrams(raw_data):
    """
    Extract the chosen bigram from each bigram pair (within each trial) for each user.
    """
    return raw_data.groupby(['user_id', 'trialId', 'bigramPair'])['chosenBigram'].first().unstack()

def plot_bigram_choice_consistency(chosen_bigrams, output_folder):
    """
    Visualize the consistency or variability of bigram choices across participants.
    Creates a heatmap where rows represent participants and columns represent bigram pairs.
    The color intensity reflects the consistency of chosen bigrams.
    """
    # Count how often each bigram was chosen across trials for each participant
    choice_consistency = chosen_bigrams.apply(lambda x: x.value_counts(normalize=True).max(), axis=1)
    
    # Create a heatmap of consistency values
    plt.figure(figsize=(10, 6))
    sns.heatmap(choice_consistency.unstack(), annot=True, cmap="YlGnBu", cbar=True)
    plt.title("Bigram Choice Consistency Across Participants")
    plt.xlabel("Bigram Pair")
    plt.ylabel("Participant")
    plt.savefig(os.path.join(output_folder, 'bigram_choice_consistency.png'))
    plt.close()

def plot_choice_vs_timing(bigram_timings, chosen_bigrams, output_folder):
    """
    Analyze the relationship between the timing of bigrams and which bigram was chosen.
    Creates a box plot showing the distribution of timings for each chosen bigram across participants.
    """
    # Merge bigram timings and chosen bigrams to create a single DataFrame
    timing_choice_data = bigram_timings.merge(chosen_bigrams.stack().reset_index().rename(columns={0: 'chosenBigram'}),
                                              on=['user_id', 'trialId', 'bigramPair'])

    # Create a boxplot to visualize the relationship between timing and chosen bigrams
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='chosenBigram', y='timing_within_bigram', data=timing_choice_data)
    plt.title("Relationship Between Chosen Bigrams and Timing")
    plt.xlabel("Chosen Bigram")
    plt.ylabel("Timing (ms)")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_folder, 'chosen_bigram_vs_timing.png'))
    plt.close()

def create_all_plots(raw_data, output_folder):
    bigram_timings = calculate_bigram_timing(raw_data)
    # Check if bigram timings are available before proceeding
    if bigram_timings.empty:
        print("No valid bigram timings found. Skipping plots.")
        return

    median_times = calculate_median_times(bigram_timings)
    # Skip if there are no median times
    if median_times.empty:
        print("No median times available. Skipping plots.")
        return

    # Generate the median bigram timing heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(median_times, cmap="coolwarm")
    plt.title('Median Bigram Timing')
    plt.savefig(os.path.join(output_folder, 'median_bigram_timing.png'))
    plt.close()

    # Generate bigram choice consistency plot
    chosen_bigrams = get_chosen_bigrams(raw_data)
    plot_bigram_choice_consistency(chosen_bigrams, output_folder)

    # Generate choice vs timing plot
    plot_choice_vs_timing(bigram_timings, chosen_bigrams, output_folder)

    print("All plots have been generated and saved in the output folder.")

# Main execution
if __name__ == "__main__":
    try:
        # Set the paths for input data and output
        folder_path = '/Users/arno.klein/Downloads/osf'  # Update this path to your data folder
        output_folder = os.path.join(os.path.dirname(folder_path), 'output')
        os.makedirs(output_folder, exist_ok=True)

        print(f"Loading data from {folder_path}")
        raw_data = load_and_preprocess_data(folder_path)
        print(f"Loaded data shape: {raw_data.shape}")
        print(f"Columns in raw_data: {raw_data.columns}")
        
        create_all_plots(raw_data, output_folder)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()