
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

def get_chosen_and_unchosen_bigrams(raw_data):
    """
    Extract the chosen and unchosen bigrams from each bigram pair (within each trial) for each user.
    
    Parameters:
    - raw_data: DataFrame containing the raw experimental data
    
    Returns:
    - chosen_bigrams: DataFrame of chosen bigrams
    - unchosen_bigrams: DataFrame of unchosen bigrams
    """
    # Group by user, trial, and bigram pair
    grouped = raw_data.groupby(['user_id', 'trialId', 'bigramPair'])
    
    # Extract chosen bigrams
    chosen_bigrams = grouped.apply(lambda x: x[x['chosen'] == True]['bigram'].iloc[0]).unstack()
    
    # Extract unchosen bigrams
    unchosen_bigrams = grouped.apply(lambda x: x[x['chosen'] == False]['bigram'].iloc[0]).unstack()
    
    return chosen_bigrams, unchosen_bigrams

def calculate_bigram_timing(data):
    """
    Calculate the timing for bigrams and return data for plotting.
    
    Parameters:
    - data: The DataFrame containing bigram data
    
    Returns:
    - bigram_timings: DataFrame with timing data for each bigram
    - bigram_pairs: List of strings representing the unique bigram pairs
    """
    # Sort the data
    data = data.sort_values(['user_id', 'trialId', 'bigramPair', 'keydownTime'])
    
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
                timing = group['keydownTime'].iloc[i+1] - group['keydownTime'].iloc[i]
                timings[bigram].append(timing)
        
        # Get the fastest timing for each bigram
        fastest_timings = {bigram: min(times) if times else np.nan for bigram, times in timings.items()}
        
        return pd.Series(fastest_timings)

    # Group the data and apply the calculation function
    bigram_timings = data.groupby(['user_id', 'trialId', 'bigramPair']).apply(calculate_fastest_time).reset_index()
    
    # Melt the DataFrame to have one row per bigram timing
    bigram_timings = bigram_timings.melt(id_vars=['user_id', 'trialId', 'bigramPair'], 
                                         var_name='bigram', value_name='timing_within_bigram')
    
    # Convert timing_within_bigram to float, replacing any non-convertible values with NaN
    bigram_timings['timing_within_bigram'] = pd.to_numeric(bigram_timings['timing_within_bigram'], errors='coerce')

    # Extract unique bigram pairs
    bigram_pairs = data['bigramPair'].unique().tolist()

    return bigram_timings, bigram_pairs

def calculate_median_times(bigram_timings):
    """
    Calculate the median timing for each bigram in each bigram pair across users.
    """
    if bigram_timings.empty:
        print("No bigram timings to calculate median from.")
        return pd.DataFrame()

    if not {'bigramPair', 'bigram', 'timing_within_bigram'}.issubset(bigram_timings.columns):
        raise KeyError("Required columns are missing in the bigram timings.")

    # Calculate median times grouped by bigramPair and bigram
    median_times = bigram_timings.groupby(['bigramPair', 'bigram'])['timing_within_bigram'].median().unstack(fill_value=np.nan)
    
    # Ensure all bigrams are represented
    all_bigrams = bigram_timings['bigram'].unique()
    for bigram in all_bigrams:
        if bigram not in median_times.columns:
            median_times[bigram] = np.nan

    print("\nMedian bigram timing shape:", median_times.shape)
    print("\nNumber of NaN values:", median_times.isna().sum().sum())
    print("\nNumber of unique bigrams:", len(all_bigrams))

    return median_times

def plot_median_bigram_timing(median_times, output_folder):
    """
    Generate and save the median bigram timing heatmap.
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

    # Additional diagnostic information
    print(f"\nNumber of non-NaN values in heatmap: {np.sum(~np.isnan(median_times_t))}")
    # Make sure that the calculation is performed over all elements, reducing to a scalar
    percentage_non_nan = 100 * np.sum(~np.isnan(median_times_t.values.flatten())) / median_times_t.size

    # Now percentage_non_nan should be a scalar
    if isinstance(percentage_non_nan, (int, float)):
        print(f"Percentage of non-NaN values: {percentage_non_nan:.2f}%")
    else:
        print("Error: percentage_non_nan is not a scalar value.")


def save_heatmap_data(median_times, output_folder):
    """
    Save the median times data to a CSV file for further analysis.
    """
    output_file = os.path.join(output_folder, 'median_bigram_timing_data.csv')
    median_times.to_csv(output_file)
    print(f"\nRaw heatmap data saved to: {output_file}")

def get_chosen_and_unchosen_bigrams(raw_data):
    """
    Extract the chosen and unchosen bigrams from each bigram pair (within each trial) for each user.
    
    Parameters:
    - raw_data: DataFrame containing the raw experimental data
    
    Returns:
    - chosen_bigrams: DataFrame of chosen bigrams
    - unchosen_bigrams: DataFrame of unchosen bigrams
    """
    # Group by user and trial
    grouped = raw_data.groupby(['user_id', 'trialId'])
    
    def extract_bigrams(group):
        chosen = group['chosenBigram'].iloc[0]
        pair = group['bigramPair'].iloc[0].split(', ')
        unchosen = pair[0] if pair[0] != chosen else pair[1]
        return pd.Series({'chosen': chosen, 'unchosen': unchosen})
    
    bigrams = grouped.apply(extract_bigrams).unstack()
    
    chosen_bigrams = bigrams['chosen']
    unchosen_bigrams = bigrams['unchosen']
    
    return chosen_bigrams, unchosen_bigrams

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

def create_all_plots(raw_data, output_folder):
    try:
        bigram_timings, bigram_pairs = calculate_bigram_timing(raw_data)
        if bigram_timings.empty:
            print("No valid bigram timings found. Skipping plots.")
            return

        print("\nShape of bigram_timings:")
        print(bigram_timings.shape)
        print("\nColumns in bigram_timings:")
        print(bigram_timings.columns)
        print("\nFirst few rows of bigram_timings:")
        print(bigram_timings.head())

        median_times = calculate_median_times(bigram_timings)
        if median_times.empty:
            print("No median times available. Skipping plots.")
            return

        print("\nShape of median_times before plotting:")
        print(median_times.shape)
        print("\nColumns in median_times:")
        print(median_times.columns)

        plot_median_bigram_timing(median_times, output_folder)

        save_heatmap_data(median_times, output_folder)

        # Use the new function to get both chosen and unchosen bigrams
        chosen_bigrams, unchosen_bigrams = get_chosen_and_unchosen_bigrams(raw_data)
        
        print("\nShape of chosen_bigrams:")
        print(chosen_bigrams.shape)
        print("\nShape of unchosen_bigrams:")
        print(unchosen_bigrams.shape)

        plot_bigram_comparison(chosen_bigrams, unchosen_bigrams, bigram_pairs)

        # Generate bigram choice consistency plot 
        plot_bigram_choice_consistency(chosen_bigrams, output_folder)

        # Generate choice vs timing plot
        plot_choice_vs_timing(bigram_timings, chosen_bigrams, output_folder)

        print("All plots have been generated and saved in the output folder.")
    except Exception as e:
        print(f"An error occurred in create_all_plots: {str(e)}")
        import traceback
        traceback.print_exc()

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