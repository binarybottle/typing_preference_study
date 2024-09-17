
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import median_abs_deviation



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
    dframe.info()
    print('')
    print("Sample output:")
    
    # Temporarily set the option to display more rows
    with pd.option_context('display.max_rows', nlines):
        print(dframe[print_headers].iloc[:nlines])  # Display 'nlines' rows

    print('')


def load_and_preprocess_data(input_folder, output_tables_folder):
    """
    Load and preprocess (combine) data from multiple CSV files in a folder.

    Parameters:
    - input_folder: path to the folder containing the CSV files
    - output_tables_folder: path to the folder where the combined data will be saved

    Returns:
    - combined_df: DataFrame with combined data
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
            
            dataframes.append(df)
    
    # Combine the dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded and combined data from {len(dataframes)} files")

    # Filter out rows where 'trialId' contains 'intro-trial'
    filtered_combined_df = combined_df[~combined_df['trialId'].str.contains("intro-trial", na=False)]

    # Display information about the combined DataFrame
    print_headers = ['trialId', 'bigramPair', 'bigram', 'typedKey', 
                     'keydownTime', 'chosenBigram', 'unchosenBigram']
    display_information(filtered_combined_df, "original data", print_headers, nlines=6)

    # Save the combined DataFrame to a CSV file
    output_file = os.path.join(output_tables_folder, 'original_combined_data.csv')
    filtered_combined_df.to_csv(output_file, index=False)
    print(f"\nCombined data saved to {output_file}")

    return filtered_combined_df


def process_bigram_data(data, output_tables_folder):
    """
    Process the bigram data.

    Parameters:
    - data: DataFrame with bigram data
    - output_tables_folder: path to the folder where the processed data will be saved

    Returns:
    - bigram_data: DataFrame with processed bigram data
    """

    def sort_bigram_pair(bigram_pair):
        """
        Normalize a bigram pair by alphabetically ordering the two bigrams.
        
        Parameters:
        - bigram_pair: A string representing the bigram pair (e.g., 'de, ef').
        
        Returns:
        - normalized_pair: A string representing the alphabetically ordered bigram pair.
        """
        # Split the bigram pair by the comma and space to get individual bigrams
        bigrams = bigram_pair.split(', ')
        
        # Sort the bigrams alphabetically and rejoin them into a normalized pair
        sorted_bigram_pair = ', '.join(sorted(bigrams))
        
        return sorted_bigram_pair
    

    def get_fastest_interkey_times(group):
        """Compute the shortest inter-key time for each bigram in the pair."""
        
        group = group.sort_values('keydownTime')

        # Initialize dicts to track minimum inter-key times for each bigram
        min_times = {}
        best_rows = {}

        # Iterate through the rows for the bigram pair of group1
        for i in range(0, len(group) - 1, 2):
            if group['keyPosition'].iloc[i] == 1 and group['keyPosition'].iloc[i + 1] == 2:
                start_time = group['keydownTime'].iloc[i]
                end_time = group['keydownTime'].iloc[i + 1]
                interkey_time = end_time - start_time
                current_bigram = group['bigram'].iloc[i]

                # If this is a new bigram, initialize its min_time
                if current_bigram not in min_times:
                    min_times[current_bigram] = float('inf')
                    best_rows[current_bigram] = None

                # Check if this is the fastest time for this bigram
                if interkey_time >= 0 and interkey_time < min_times[current_bigram]:
                    min_times[current_bigram] = interkey_time
                    best_rows[current_bigram] = group.iloc[[i, i + 1]].copy()
                    best_rows[current_bigram]['min_interkey_time'] = interkey_time
        #print("")
        #print(group)
        #print(group['chosenBigram'].nunique())
        #print(best_rows.keys())
        #print(best_rows)

        # Combine the best rows for each bigram
        if len(best_rows) == 2:
            bigram1, bigram2 = best_rows.keys()
            bigram_pair = best_rows[bigram1]['bigramPair'].iloc[0]
            chosen_bigram = best_rows[bigram1]['chosenBigram'].iloc[0]
            unchosen_bigram = best_rows[bigram1]['unchosenBigram'].iloc[0]
            
            # Determine chosen and unchosen bigram times
            if chosen_bigram == bigram1:
                chosen_bigram_time = min_times[bigram1]
                unchosen_bigram_time = min_times[bigram2]
            else:
                chosen_bigram_time = min_times[bigram2]
                unchosen_bigram_time = min_times[bigram1]

            # Is the chosen bigram consistent across both sequences of the bigram pair?
            if group['chosenBigram'].nunique() == 1:
                is_consistent = True
            else:
                is_consistent = False

                # Check inconsistencies
                #print_headers = ['bigramPair', 'chosenBigram']
                #display_information(group, "group", print_headers, nlines=30)


            result = pd.DataFrame({
                'sorted_bigram_pair': best_rows[bigram1]['sorted_bigram_pair'],
                'bigram_pair': bigram_pair,
                'bigram1': bigram1,
                'bigram2': bigram2,
                'bigram1_time': min_times[bigram1],
                'bigram2_time': min_times[bigram2],
                'chosen_bigram': chosen_bigram,
                'unchosen_bigram': unchosen_bigram,
                'chosen_bigram_time': chosen_bigram_time,
                'unchosen_bigram_time': unchosen_bigram_time,
                'keydownTime_bigram1_first': best_rows[bigram1]['keydownTime'].iloc[0],
                'keydownTime_bigram1_second': best_rows[bigram1]['keydownTime'].iloc[1],
                'keydownTime_bigram2_first': best_rows[bigram2]['keydownTime'].iloc[0],
                'keydownTime_bigram2_second': best_rows[bigram2]['keydownTime'].iloc[1],
                'is_consistent': is_consistent
            })
            
            # Copy over important metadata that's the same for both bigrams in the pair
            for col in ['user_id', 'trialId']:
                result[col] = best_rows[bigram1][col].iloc[0]
            
            return result
        else:
            print(f"Unexpected number of bigrams ({len(best_rows)}) for bigramPair: {group['bigramPair'].iloc[0]}")
            return pd.DataFrame()  # Return an empty DataFrame if we don't have exactly 2 bigrams

    # Sort the bigram pairs
    data['sorted_bigram_pair'] = data['bigramPair'].apply(sort_bigram_pair)

    # Iterate through each unique combination of user_id and sorted_bigram_pair
    unique_combinations = data[['user_id', 'sorted_bigram_pair']].drop_duplicates()
    result_list = []
    for _, row in unique_combinations.iterrows():

        user_id = row['user_id']
        sorted_bigram_pair = row['sorted_bigram_pair']

        # Filter data for the specific user_id and bigramPair
        group = data[(data['user_id'] == user_id) & 
                     (data['sorted_bigram_pair'] == sorted_bigram_pair)]

        # Apply the filtering function to each group
        filtered_group = get_fastest_interkey_times(group)

        # Append the result to the result list
        result_list.append(filtered_group)

    # Concatenate all the results into a single DataFrame
    bigram_data = pd.concat(result_list).reset_index(drop=True)

    bigram_data = bigram_data.drop_duplicates()

    # Display information about the bigram DataFrame
    print_headers = ['sorted_bigram_pair','bigram_pair', 
                     'chosen_bigram', 'unchosen_bigram', 
                     'chosen_bigram_time', 'unchosen_bigram_time', 'is_consistent']
    display_information(bigram_data, "bigram data", print_headers, nlines=12)

    # Save and return the bigram data
    bigram_data.to_csv(output_tables_folder + '/bigram_data.csv', index=False)
    
    return bigram_data


def analyze_inconsistencies(bigram_data, output_plots_folder):
    """
    Analyze and report inconsistencies in bigram data.
    """
    total_users = bigram_data['user_id'].nunique()
    total_pairs = bigram_data['bigram_pair'].nunique()

    print("\nValue counts of 'is_consistent':")
    print(bigram_data['is_consistent'].value_counts())

    # Users with inconsistencies
    users_with_inconsistencies = bigram_data[bigram_data['is_consistent'] == False]['user_id'].nunique()

    # Total inconsistent pairs
    total_inconsistent_pairs = bigram_data[bigram_data['is_consistent'] == False]['bigram_pair'].nunique()

    # Inconsistent pairs per user
    inconsistent_pairs_per_user = bigram_data[bigram_data['is_consistent'] == False].groupby('user_id')['is_consistent'].count()

    # Most common inconsistent pairs
    inconsistent_pair_counts = bigram_data[bigram_data['is_consistent'] == False]['bigram_pair'].value_counts()

    print("\nInconsistency Statistics:")
    print(f"Total users: {total_users}")
    print(f"Total bigram pairs: {total_pairs}")
    print(f"Users with inconsistencies: {users_with_inconsistencies} ({users_with_inconsistencies/total_users:.2%})")
    print(f"Total inconsistent pairs: {total_inconsistent_pairs} ({total_inconsistent_pairs/total_pairs:.2%})")
    if inconsistent_pairs_per_user.empty:
        print("\nNo inconsistent pairs found.")
    else:
        print(f"\nInconsistent pairs per user:")
        print(f"  Mean: {inconsistent_pairs_per_user.mean():.2f}")
        print(f"  Median: {inconsistent_pairs_per_user.median():.2f}")
        print(f"  Max: {inconsistent_pairs_per_user.max()}")
    print(f"\nTop most common inconsistent pairs:")
    print(inconsistent_pair_counts.head(10).to_string())

    # Plot the histogram of inconsistent pair counts
    plt.figure(figsize=(12,6))
    plt.hist(inconsistent_pair_counts, bins=10, color='black', edgecolor='black')
    plt.title('Histogram of inconsistent pair counts')
    plt.xlabel('Number of participants')
    plt.ylabel('Number of bigram pairs')
    plt.grid(False)
    plt.show()
    plt.savefig(os.path.join(output_plots_folder, 'inconsistent_pair_counts.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Return the statistics for further use if needed
    return {
        'total_users': total_users,
        'total_pairs': total_pairs,
        'users_with_inconsistencies': users_with_inconsistencies,
        'total_inconsistent_pairs': total_inconsistent_pairs,
        'inconsistent_pairs_per_user': inconsistent_pairs_per_user,
        'inconsistent_pair_counts': inconsistent_pair_counts
    }


def plot_median_bigram_times(bigram_data, output_plots_folder):
    """
    Generate and save bar plot for median bigram typing times.
    
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

    # Customize the plot
    ax.set_title('Median Typing Times for Each Bigram (with MAD)', fontsize=16)
    ax.set_xlabel('Bigram', fontsize=12)
    ax.set_ylabel('Median Time (ms)', fontsize=12)

    # Set x-ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(median_times.index, rotation=90, ha='center', fontsize=10)

    # Adjust the x-axis to center labels under bars
    ax.set_xlim(-0.5, len(x) - 0.5)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Add value labels on top of each bar
    """
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.0f}',
                 ha='center', va='bottom')
    """

    plt.savefig(os.path.join(output_plots_folder, 'bigram_median_times_barplot_with_mad.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"plot_median_bigram_times plot saved in {output_plots_folder}")


def plot_paired_bar(bigram_data, output_plots_folder):
    # Prepare data
    plot_data = bigram_data.groupby('bigram_pair').agg({
        'chosen_bigram_time': 'median',
        'unchosen_bigram_time': 'median'
    }).reset_index()

    plot_data = plot_data.melt(id_vars=['bigram_pair'], 
                               value_vars=['chosen_bigram_time', 'unchosen_bigram_time'],
                               var_name='Type', value_name='Time')

    # Sort by chosen bigram time
    order = plot_data[plot_data['Type'] == 'chosen_bigram_time'].sort_values('Time')['bigram_pair']

    # Create plot
    plt.figure(figsize=(15, len(order) * 0.4))
    sns.barplot(x='Time', y='bigram_pair', hue='Type', data=plot_data, order=order)
    
    plt.title('Median Typing Times: Chosen vs Unchosen Bigrams')
    plt.xlabel('Median Typing Time (ms)')
    plt.ylabel('Bigram Pair')
    plt.legend(title='', labels=['Chosen', 'Unchosen'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'paired_bar_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter(bigram_data, output_plots_folder):
    # Prepare data
    plot_data = bigram_data.groupby('bigram_pair').agg({
        'chosen_bigram_time': 'median',
        'unchosen_bigram_time': 'median'
    }).reset_index()

    # Create plot
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=plot_data, x='chosen_bigram_time', y='unchosen_bigram_time')
    
    # Add diagonal line
    max_val = max(plot_data['chosen_bigram_time'].max(), plot_data['unchosen_bigram_time'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    plt.title('Chosen vs Unchosen Bigram Typing Times')
    plt.xlabel('Chosen Bigram Median Time (ms)')
    plt.ylabel('Unchosen Bigram Median Time (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'scatter_plot.png'), dpi=300, bbox_inches='tight')
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
                                              on=['user_id', 'trialId', 'bigram_pair'])

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
        input_folder = '/Users/arno.klein/Downloads/osf'  # Update this path to your data folder
        output_folder = os.path.join(os.path.dirname(input_folder), 'output')
        output_tables_folder = os.path.join(output_folder, 'tables')
        output_plots_folder = os.path.join(output_folder, 'plots')
        os.makedirs(output_tables_folder, exist_ok=True)
        os.makedirs(output_plots_folder, exist_ok=True)

        # Load, combine, and save the data
        data = load_and_preprocess_data(input_folder, output_tables_folder)

        ##########################
        # PROCESS_BIGRAM_DATA
        ##########################
        # Process the bigram data
        bigram_data = process_bigram_data(data, output_tables_folder)
        
        # Analyze the inconsistencies
        analyze_inconsistencies(bigram_data, output_plots_folder)

        """
        # Display the results
        bigram_data.info()
        print(bigram_data[['trialId', 'bigram_pair', 
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

        """

        ##########################
        # PLOTS
        ##########################
        plot_median_bigram_times(bigram_data, output_plots_folder)
        plot_paired_bar(bigram_data, output_plots_folder)
        plot_scatter(bigram_data, output_plots_folder)
        #plot_bigram_comparison(chosen_data, unchosen_data, bigram_pairs)
        #plot_bigram_pair_boxplots(bigram_data, output_plots_folder)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()