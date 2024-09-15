
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
    
    output_file = os.path.join(output_folder, 'combined_data.csv')
    data.to_csv(output_file, index=False)
    print(f"\nCombined data saved to {output_file}")

    return combined_df


def process_bigram_data(data, output_folder):
    """
    Process bigram data.
    
    Parameters:
    - data: DataFrame containing the raw experimental data
    - output_folder: path to the folder where the processed data will be saved
    
    Returns:
    - bigram_data: DataFrame with processed bigram data
    """
    # Sort the data by user_id, trialId, bigramPair, and keydownTime
    sorted_data = data.sort_values(['user_id', 'trialId', 'bigramPair', 'keydownTime'])
    
    # Group the sorted data by user_id, trialId, and bigramPair
    grouped_data = sorted_data.groupby(['user_id', 'trialId', 'bigramPair'])

    def process_bigram_group(group):
        bigram_pair = group['bigramPair'].iloc[0].split(', ')
        chosen_bigram = group['chosenBigram'].iloc[0]
        unchosen_bigram = group['unchosenBigram'].iloc[0]
        
        # Calculate timings for both bigrams
        timings = {bigram: [] for bigram in bigram_pair}
        for i in range(0, len(group) - 1, 2):
            bigram = group['bigram'].iloc[i]
            if bigram in bigram_pair:
                start_time = group['keydownTime'].iloc[i]
                end_time = group['keydownTime'].iloc[i + 1]
                timing = end_time - start_time
                timings[bigram].append(timing)
        
        # Calculate the fastest time for each bigram
        fastest_times = {bigram: min(times) if times else np.nan for bigram, times in timings.items()}
        
        return pd.Series({
            'bigram1': bigram_pair[0],
            'bigram2': bigram_pair[1],
            'chosen_bigram': chosen_bigram,
            'unchosen_bigram': unchosen_bigram,
            'bigram1_time': fastest_times[bigram_pair[0]],
            'bigram2_time': fastest_times[bigram_pair[1]],
            'chosen_bigram_time': fastest_times[chosen_bigram],
            'unchosen_bigram_time': fastest_times[unchosen_bigram]
        })

    bigram_data = grouped_data.apply(process_bigram_group).reset_index()
    
    # Save the DataFrame to CSV
    full_path = os.path.join(output_folder, "bigram_data.csv")    
    bigram_data.to_csv(full_path, index=False)
    print(f"Bigram data saved to: {full_path}")

    return bigram_data


def plot_median_typist_bigram_times(bigram_data, output_folder):
    """
    Generate and save improved plots for median bigram typing times.
    
    1. Heatmap of Median Bigram Typing Times: 
       This shows the median typing time for each bigram pair, 
       allowing for easy comparison between different combinations.
    2. Bar Plot of Median Bigram Times: 
       This provides a clear ranking of bigrams by their median typing times.
    3. Violin Plot of Chosen vs. Unchosen Bigram Times: 
       This compares the distribution of typing times for chosen and unchosen bigrams, 
       which can reveal preferences in bigram selection.
    4. Scatter Plot of Bigram1 vs Bigram2 Times: 
       This plot shows the relationship between the typing times of the two bigrams 
       in each pair, with the chosen bigram highlighted.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_folder: String path to the folder where plots should be saved
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Prepare data for plotting
    plot_data = bigram_data.melt(id_vars=['user_id', 'trialId', 'bigramPair', 'chosen_bigram'],
                                 value_vars=['bigram1_time', 'bigram2_time'],
                                 var_name='bigram_position', value_name='time')
    plot_data['bigram'] = np.where(plot_data['bigram_position'] == 'bigram1_time', 
                                   plot_data['bigramPair'].str.split(', ').str[0],
                                   plot_data['bigramPair'].str.split(', ').str[1])
    plot_data['is_chosen'] = plot_data['bigram'] == plot_data['chosen_bigram']

    # Calculate median times for each bigram
    median_times = plot_data.groupby('bigram')['time'].median().sort_values()

    # 2. Create a heatmap of median bigram times
    plt.figure(figsize=(12, 10))
    heatmap_data = plot_data.pivot_table(values='time', index='bigram', columns='bigram', aggfunc='median')
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Median Time (ms)'})
    plt.title('Heatmap of Median Bigram Typing Times')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bigram_timing_heatmap.png'))
    plt.close()

    # 3. Create a bar plot of median bigram times
    plt.figure(figsize=(15, 8))
    sns.barplot(x=median_times.index, y=median_times.values, palette='viridis')
    plt.title('Median Typing Times for Each Bigram')
    plt.xlabel('Bigram')
    plt.ylabel('Median Time (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bigram_median_times_barplot.png'))
    plt.close()

    # 4. Create a violin plot comparing chosen vs. unchosen bigram times
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='is_chosen', y='time', data=plot_data, palette='Set3')
    plt.title('Distribution of Typing Times: Chosen vs. Unchosen Bigrams')
    plt.xlabel('Chosen Bigram')
    plt.ylabel('Time (ms)')
    plt.xticks([0, 1], ['Unchosen', 'Chosen'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'chosen_vs_unchosen_times_violin.png'))
    plt.close()

    # 5. Create a scatter plot of bigram1 vs bigram2 times
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=bigram_data, x='bigram1_time', y='bigram2_time', hue='chosen_bigram', style='chosen_bigram')
    plt.title('Bigram1 Time vs Bigram2 Time')
    plt.xlabel('Bigram1 Time (ms)')
    plt.ylabel('Bigram2 Time (ms)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bigram1_vs_bigram2_scatter.png'))
    plt.close()

    print(f"Plots saved in {output_folder}")




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
        ##########################
        # LOAD_AND_PREPROCESS_DATA
        ##########################
        # Set the paths for input data and output
        folder_path = '/Users/arno.klein/Downloads/osf'  # Update this path to your data folder
        output_folder = os.path.join(os.path.dirname(folder_path), 'output')
        os.makedirs(output_folder, exist_ok=True)

        # Load, combine, and save the data
        print(f"Loading data from {folder_path}")
        data = load_and_preprocess_data(folder_path)
        print(f"Loaded data shape: {data.shape}")
        data.info()
        print(data[['trialId', 'bigramPair', 'bigram', 'typedKey', 'keydownTime', 
                    'chosenBigram', 'unchosenBigram']].head(6))

        ##########################
        # PROCESS_BIGRAM_DATA
        ##########################
        # Process the bigram data
        bigram_data = process_bigram_data(data)

        # Display the results
        bigram_data.info()
        print(bigram_data[['trialId', 'bigramPair', 
                           'chosen_bigram', 'unchosen_bigram',
                           'chosen_bigram_time', 'unchosen_bigram_time']].head(6))
        
        # Calculate average times for chosen and unchosen bigrams
        avg_chosen_time = bigram_data['chosen_bigram_time'].mean()
        avg_unchosen_time = bigram_data['unchosen_bigram_time'].mean()
        print(f"Average time for chosen bigrams: {avg_chosen_time:.2f} ms")
        print(f"Average time for unchosen bigrams: {avg_unchosen_time:.2f} ms")

        # Compare bigram1 and bigram2 times
        bigram_data['faster_bigram'] = np.where(bigram_data['bigram1_time'] < bigram_data['bigram2_time'], 
                                                bigram_data['bigram1'], bigram_data['bigram2'])

        # Check if the faster bigram is also the chosen bigram
        # Do participants tend to choose the bigram that they can type faster?
        # Is there a correlation between typing speed and bigram preference?
        bigram_data['faster_is_chosen'] = bigram_data['faster_bigram'] == bigram_data['chosen_bigram']
        print(f"Percentage of times the faster bigram was chosen: {bigram_data['faster_is_chosen'].mean()*100:.2f}%")

        ##########################
        # PLOTS
        ##########################
        plot_median_typist_bigram_times(bigram_data, output_folder)


    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()