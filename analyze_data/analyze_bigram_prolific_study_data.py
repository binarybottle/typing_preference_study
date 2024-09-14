
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

#def group_bigrams(data):
    """
    Sort data and group by user, trial, and bigram pair
    """
    # Sort the data
    sorted_data = data.sort_values(['user_id', 'trialId', 'bigramPair', 'bigram', 'keydownTime'])

    # Group by user, trial, and bigram pair
    grouped_data = data.groupby(['user_id', 'trialId', 'bigramPair'])

    return sorted_data, grouped_data 

#def get_chosen_and_unchosen_bigrams(grouped_data):
    """
    Extract the chosen and unchosen bigrams from each bigram pair (within each trial) for each user.
    
    Parameters:
    - grouped_data: DataFrame containing the raw experimental data grouped by bigram pair
    
    Returns:
    - chosen_bigrams: DataFrame of chosen bigrams
    - unchosen_bigrams: DataFrame of unchosen bigrams
    """
    # Extract un/chosen bigrams
    bigrams = grouped_data['bigramPair']
    chosen_bigrams = grouped_data['chosenBigram']
    unchosen_bigrams = grouped_data['unchosenBigram']
            
    return bigrams, chosen_bigrams, unchosen_bigrams

#def calculate_bigram_times(grouped_data):
    """
    Calculate the fastest time to type each bigram and return data for plotting.
    
    Parameters:
    - grouped_data: DataFrameGroupBy object containing bigram data grouped by user_id, trialId, and bigramPair
    
    Returns:
    - bigram_times: DataFrame with fastest timing data for each bigram
    """
    def calculate_fastest_time(group):
        # Extract bigrams from bigramPair
        bigrams = group['bigramPair'].iloc[0].split(', ')
        
        timings = {bigram: [] for bigram in bigrams}
        
        # Calculate timing for each complete bigram
        for i in range(len(group) - 1):
            first_letter = group['typedKey'].iloc[i]
            second_letter = group['typedKey'].iloc[i+1]
            bigram = first_letter + second_letter
            if bigram in bigrams:
                key1time = group['keydownTime'].iloc[i]
                key2time = group['keydownTime'].iloc[i+1]
                timing = key2time - key1time
                timings[bigram].append(timing)
        
        # Get the fastest timing for each bigram
        fastest_times = {bigram: min(times) if times else np.nan for bigram, times in timings.items()}
        
        return pd.Series(fastest_times)

    # Apply the calculation function to each group, explicitly including the index
    bigram_times = grouped_data.apply(calculate_fastest_time, include_groups=False).reset_index()
    
    # Melt the DataFrame to have one row per bigram timing
    id_vars = ['user_id', 'trialId', 'bigramPair']
    value_vars = [col for col in bigram_times.columns if col not in id_vars]
    
    bigram_times = bigram_times.melt(id_vars=id_vars, 
                                     value_vars=value_vars,
                                     var_name='bigram', 
                                     value_name='timing_within_bigram')

    return bigram_times


def process_bigram_data(data):
    """
    Process bigram data.
    
    Parameters:
    - data: DataFrame containing the raw experimental data
    
    Returns:
    - processed_data: DataFrame with processed bigram data
    """
    # Sort the data
    sorted_data = data.sort_values(['user_id', 'trialId', 'bigramPair', 'keydownTime'])
    
    # Group the sorted data by user_id, trialId, and bigramPair
    grouped_data = sorted_data.groupby(['user_id', 'trialId', 'bigramPair'])

    def process_bigram_group(group):
        """
        Process bigram group data.
        
        Parameters:
        - group: DataFrame containing the raw experimental data
        
        Returns:
        - processed_data: DataFrame with processed bigram data
        """
        chosen_bigram = group['chosenBigram'].iloc[0]
        unchosen_bigram = group['unchosenBigram'].iloc[0]
        bigram_pair = group['bigramPair'].iloc[0].split(', ')
        
        # Extract the four letters from the two bigrams
        letters = ''.join(bigram_pair)
        assert len(letters) == 4, f"Expected 4 letters, got {len(letters)} from {bigram_pair}"
        
        timings = {bigram: [] for bigram in bigram_pair}
        
        # We'll assume the correct order of keystrokes for 3 of each bigram is: 
        # letter1, letter2, letter1, letter2, letter1, letter2, 
        # letter3, letter4, letter3, letter4, letter3, letter4
        expected_order = [letters[0], letters[1], letters[0], letters[1], letters[0], letters[1], 
                        letters[2], letters[3], letters[2], letters[3], letters[2], letters[3]]
        
        # Calculate timings based on the expected order
        for i in range(0, len(group) - 1, 2):
            bigram = expected_order[i] + expected_order[i+1]
            if bigram in bigram_pair:
                start_time = group['keydownTime'].iloc[i]
                end_time = group['keydownTime'].iloc[i+1]
                timing = end_time - start_time
                timings[bigram].append(timing)
        
        chosen_time = np.min(timings[chosen_bigram]) if timings[chosen_bigram] else np.nan
        unchosen_time = np.min(timings[unchosen_bigram]) if timings[unchosen_bigram] else np.nan
        
        return pd.Series({
            'chosen_bigram': chosen_bigram,
            'unchosen_bigram': unchosen_bigram,
            'chosen_bigram_time': chosen_time,
            'unchosen_bigram_time': unchosen_time
            #'expected_keystrokes': ', '.join(expected_order)
        })

    processed_bigram_data = grouped_data.apply(process_bigram_group).reset_index()
    
    return processed_bigram_data, grouped_data

def calculate_median_typist_bigram_times(bigram_times):
    """
    Calculate the median times for each bigram in each bigram pair across users.
    """
    if bigram_times.empty:
        print("No bigram times to calculate median from.")
        return pd.DataFrame()

    if not {'bigramPair', 'bigram'}.issubset(bigram_times.columns):
        raise KeyError("'bigramPair' or 'bigram' columns are missing in the bigram_times.")

    # Calculate median times grouped by bigramPair and bigram
    median_times = bigram_times.groupby(['bigramPair', 'bigram'])['timing_within_bigram'].median().unstack(fill_value=np.nan)
    
    # Ensure all bigrams are represented
    all_bigrams = bigram_times['bigram'].unique()
    for bigram in all_bigrams:
        if bigram not in median_times.columns:
            median_times[bigram] = np.nan

    return median_times

def save_median_typist_bigram_times(median_times, output_folder):
    """
    Save the median times data to a CSV file for further analysis.
    """
    output_file = os.path.join(output_folder, 'median_typist_bigram_times.csv')
    median_times.to_csv(output_file)
    print(f"\nRaw heatmap data saved to: {output_file}")

def plot_median_typist_bigram_times(median_times, output_folder):
    """
    Generate and save the median bigram timing plots.
    """
    # Transpose the DataFrame to have bigrams as rows and bigram pairs as columns
    median_times_t = median_times.T

    plt.figure(figsize=(30, 20))  # Adjust figure size as needed
    
    # Create a mask for NaN values
    mask = np.isnan(median_times_t)
    
    # Plot heatmap
    sns.heatmap(median_times_t, 
                cmap="coolwarm", 
                annot=False,  # Changed to False due to the large number of cells
                fmt='.1f', 
                linewidths=0.5, 
                cbar_kws={'label': 'Median Timing (ms)'},
                mask=mask)
    
    plt.title('Median Bigram Timing', fontsize=20)
    plt.xlabel('Bigram Pairs', fontsize=16)
    plt.ylabel('Bigrams', fontsize=16)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=8)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_folder, 'median_bigram_timing.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nHeatmap saved with shape: {median_times_t.shape}")

    # Visualize the distribution of median times
    melted_data = median_times.melt()
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_data, x=melted_data.columns[0], y=melted_data.columns[1])
    plt.title('Distribution of Median Typing Times for Each Bigram')
    plt.xlabel('Bigram')
    plt.ylabel('Median Time (ms)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bigram_median_times_distribution.png'))
    plt.close()

    # Correlation between bigram pairs
    correlation_matrix = median_times.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation of Median Times Between Bigram Pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bigram_median_times_correlation.png'))
    plt.close()

def plot_bigram_comparison(chosen_data, unchosen_data, bigram_pairs):
    """
    Plots a grayscale bar-and-whisker plot for chosen and unchosen bigrams.
    
    Parameters:
    - chosen_data: list of lists containing values for each chosen bigram.
    - unchosen_data: list of lists containing values for each unchosen bigram.
    - bigram_pairs: list of strings representing each bigram pair.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # X-axis positions with more space between pairs
    positions = np.arange(len(chosen_data) * 2, step=2)

    # Plot chosen bigram (left side of each pair) with a thicker line for chosen ones
    bp1 = ax.boxplot(chosen_data, positions=positions, widths=0.4, patch_artist=True,
                     boxprops=dict(facecolor='lightgray', linewidth=2),
                     medianprops=dict(color='black', linewidth=1.5),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'),
                     flierprops=dict(marker='o', color='gray', alpha=0.5))

    # Plot unchosen bigram (right side of each pair) with normal line width
    bp2 = ax.boxplot(unchosen_data, positions=positions + 0.5, widths=0.4, patch_artist=True,
                     boxprops=dict(facecolor='white', linewidth=1),
                     medianprops=dict(color='black', linewidth=1.5),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'),
                     flierprops=dict(marker='o', color='gray', alpha=0.5))

    # Set x-axis labels (bigram pairs)
    ax.set_xticks(positions + 0.25)
    ax.set_xticklabels(bigram_pairs)

    # Add a title and labels
    ax.set_title("Comparison of Chosen vs Unchosen Bigrams (Grayscale)")
    ax.set_xlabel("Bigram Pairs")
    ax.set_ylabel("Values")

    # Add more space between pairs than within each pair
    ax.set_xlim(-0.5, len(chosen_data) * 2 - 0.5)

    # Show the plot
    plt.tight_layout()
    plt.show()


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


# Main execution
if __name__ == "__main__":
    try:
        # Set the paths for input data and output
        folder_path = '/Users/arno.klein/Downloads/osf'  # Update this path to your data folder
        output_folder = os.path.join(os.path.dirname(folder_path), 'output')
        os.makedirs(output_folder, exist_ok=True)

        # Load and preprocess the data
        print(f"Loading data from {folder_path}")
        data = load_and_preprocess_data(folder_path)
        print(f"Loaded data shape: {data.shape}")
        print(f"Columns in data: {data.columns}")
        #print(data.head(10))
        print(data[['trialId', 'bigramPair', 'bigram', 'keydownTime', 
                    'chosenBigram', 'unchosenBigram']].head(6))

        # Save the combined data to a new CSV file
        output_file = os.path.join(output_folder, 'combined_data.csv')
        data.to_csv(output_file, index=False)
        print(f"\nCombined data saved to {output_file}")

        # Process the bigram data
        processed_bigram_data, grouped_data = process_bigram_data(data)
        print(grouped_data[['trialId', 'bigramPair', 'bigram', 'keydownTime', 
                            'chosenBigram', 'unchosenBigram']].head(6))
        #print(processed_bigram_data.head(10))
        print(processed_bigram_data[['trialId', 'bigramPair',  
                                     'chosen_bigram', 'unchosen_bigram',
                                     'chosen_bigram_time', 'unchosen_bigram_time']].head(6))

        # Error checking
        total_pairs = len(processed_bigram_data)
        missing_chosen = processed_bigram_data['chosen_bigram_time'].isna().sum()
        missing_unchosen = processed_bigram_data['unchosen_bigram_time'].isna().sum()
        print(f"Total bigram pairs processed: {total_pairs}")
        print(f"Missing chosen bigram times: {missing_chosen} ({missing_chosen/total_pairs:.2%})")
        print(f"Missing unchosen bigram times: {missing_unchosen} ({missing_unchosen/total_pairs:.2%})")
        
        # Additional statistics
        chosen_bigrams = processed_bigram_data['chosen_bigram']
        unchosen_bigrams = processed_bigram_data['unchosen_bigram']
        chosen_times = processed_bigram_data['chosen_bigram_time']
        unchosen_times = processed_bigram_data['unchosen_bigram_time']
        avg_chosen_time = processed_bigram_data['chosen_bigram_time'].mean()
        avg_unchosen_time = processed_bigram_data['unchosen_bigram_time'].mean()
        print(f"Average typing time for chosen bigrams: {avg_chosen_time:.2f} ms")
        print(f"Average typing time for unchosen bigrams: {avg_unchosen_time:.2f} ms")

        """

        # Calculate, save, and plot the median typist bigram times
        median_times = calculate_median_typist_bigram_times(bigram_times)  
        save_median_typist_bigram_times(median_times, output_folder)
        plot_median_typist_bigram_times(median_times, output_folder) 
        print("\nBasic statistics of median times:")
        print(median_times.describe())

        """






        #plot_bigram_comparison(chosen_bigrams, unchosen_bigrams, bigram_pairs)

        # Generate bigram choice consistency plot 
        #plot_bigram_choice_consistency(chosen_bigrams, output_folder)

        # Generate choice vs timing plot
        #plot_choice_vs_timing(bigram_timings, chosen_bigrams, output_folder)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()