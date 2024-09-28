
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
    if verbose:
        print_headers = ['trialId', 'sliderValue', 'chosenBigram', 'unchosenBigram', 
                         'chosenBigramTime', 'unchosenBigramTime']
        display_information(filtered_combined_df, "original data", print_headers, nlines=6)

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
            chosen_bigram = row['chosenBigram']
            unchosen_bigram = row['unchosenBigram']
            
            result = pd.DataFrame({
                'user_id': [user_id],
                'trialId': [row['trialId']],
                'bigram_pair': [std_bigram_pair],
                'bigram1': [bigram1],
                'bigram2': [bigram2],
                'bigram1_time': [row['chosenBigramTime'] if chosen_bigram == bigram1 else row['unchosenBigramTime']],
                'bigram2_time': [row['chosenBigramTime'] if chosen_bigram == bigram2 else row['unchosenBigramTime']],
                'chosen_bigram': [chosen_bigram],
                'unchosen_bigram': [unchosen_bigram],
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
    
    # Sort the DataFrame by the 'bigram_pair' column and 'trialId'
    bigram_data = bigram_data.sort_values(by=['user_id', 'bigram_pair', 'trialId']).reset_index(drop=True)
    
    # Display information about the bigram DataFrame
    if verbose:
        print_headers = ['bigram_pair', 'chosen_bigram', 'chosen_bigram_time', 'chosen_bigram_correct',  
                        'sliderValue', 'is_consistent']
        display_information(bigram_data, "bigram data", print_headers, nlines=10)
    
    # Save and return the bigram data
    bigram_data.to_csv(f"{output_tables_folder}/processed_bigram_data.csv", index=False)
    
    return bigram_data


def analyze_easy_choices(bigram_data, easy_choice_pairs, threshold=5):
    """
    Analyze bigram choices to detect improbable selections.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - easy_choice_pairs: List of tuples, each containing a pair of bigrams where one is highly improbable
    - threshold: Float, the threshold for flagging a user (default 5)

    Returns:
    - suspicious_users: DataFrame containing users with suspiciously high rates of improbable choices
    - improbable_bigram_freq: DataFrame with frequency and slider statistics of improbable bigrams
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
    suspicious_users = user_improbable_counts[user_improbable_counts >= threshold].reset_index()
    suspicious_users.columns = ['user_id', 'improbable_choice_count']

    # Calculate overall statistics
    total_users = bigram_data['user_id'].nunique()
    total_choices = len(bigram_data)
    total_improbable_pairs_possible = len(easy_choice_pairs) * total_users
    total_improbable_pairs_presented = bigram_data['improbable_pair'].notnull().sum()
    total_improbable_choices = bigram_data['improbable_choice'].sum()
    total_suspicious = len(suspicious_users)
    avg_improbable_count = user_improbable_counts.mean()

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
    print(f"Users: {total_users}")
    print(f"Choices: {total_choices}")
    print(f"Improbable pairs possible: {total_improbable_pairs_possible}")
    print(f"Improbable pairs presented: {total_improbable_pairs_presented}")
    print(f"Improbable choices made: {total_improbable_choices}")
    print(f"Users with >{threshold} improbable choices: {total_suspicious} ({(total_suspicious / total_users) * 100:.2f}%)")

    if not suspicious_users.empty:
        print("\nTop 50 users with highest improbable choice counts:")
        print(suspicious_users.sort_values('improbable_choice_count', ascending=False).head(50))

    print("\nFrequency of improbable bigrams chosen and their probable counterparts:")
    print(improbable_bigram_freq.to_string(index=False))

    return suspicious_users, improbable_bigram_freq


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

    print(f"\nTop 10 users with highest inconsistent choice counts:")
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


def analyze_typing_times(bigram_data):
    """
    Analyze and report typing times in bigram data.

    Parameters:
    - bigram_data: DataFrame containing processed bigram data

    Returns:
    - typing_time_stats: Dictionary containing typing time statistics
    """

    print("\n____ Bigram Typing Time Statistics ____\n")

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
    percent_faster_chosen = bigram_data['faster_is_chosen'].mean()*100
    print(f"Percentage of times the faster bigram was chosen: {percent_faster_chosen:.2f}%")

    # Return the statistics for further use if needed
    typing_time_stats = {
        'avg_chosen_time': avg_chosen_time,
        'avg_unchosen_time': avg_unchosen_time,
        'percent_faster_chosen': percent_faster_chosen
    }
    return typing_time_stats


def analyze_choice_times(bigram_data):
    """
    Perform and print correlation analysis between chosen and unchosen bigram typing times.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data

    Returns:
    - choice_time_stats: Dictionary containing choice vs. time statistics
    """

    print("\n____ Bigram Typing Time vs. Choice Statistics ____\n")

    # Calculate Pearson and Spearman correlations
    pearson_corr = bigram_data[['chosen_bigram_time', 'unchosen_bigram_time']].corr(method='pearson').iloc[0, 1]
    spearman_corr = bigram_data[['chosen_bigram_time', 'unchosen_bigram_time']].corr(method='spearman').iloc[0, 1]

    print(f"Pearson correlation between chosen and unchosen typing times: {pearson_corr:.3f}")
    print(f"Spearman correlation between chosen and unchosen typing times: {spearman_corr:.3f}")

    # Return the statistics for further use if needed
    choice_time_stats = {
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr
    }
    return choice_time_stats


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

    print(f"Number of unique bigrams: {len(median_times)}")

    # Create a bar plot of median bigram times with MAD whiskers
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Create x-coordinates for the bars
    x = np.arange(len(median_times))
    
    # Plot bars
    bars = ax.bar(x, median_times.values, yerr=mad_times.values, capsize=5)

    # Customize the plot
    ax.set_title('Median typing times for each bigram (with MAD)', fontsize=16)
    ax.set_xlabel('Bigram', fontsize=12)
    ax.set_ylabel('Median time (ms)', fontsize=12)

    # Set x-ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(median_times.index, rotation=0, ha='center', fontsize=12)

    # Adjust the x-axis to center labels under bars
    ax.set_xlim(-0.5, len(x) - 0.5)

    # Add a bit of padding to the bottom of the plot for labels
    plt.subplots_adjust(bottom=0.2)

    # If there are many bigrams, we might need to show only every nth label
    if len(median_times) > 50:
        for i, tick in enumerate(ax.xaxis.get_major_ticks()):
            if i % 5 != 0:
                tick.label1.set_visible(False)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    plt.savefig(os.path.join(output_plots_folder, 'bigram_median_times_barplot_with_mad.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"bigram_median_times_barplot_with_mad plot saved in {output_plots_folder}")


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


def plot_chosen_vs_unchosen_times_boxplot(bigram_data, output_plots_folder):
    """
    Generate and save box-and-whisker plot for chosen vs. unchosen bigram typing times.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    """
    plot_data = prepare_plot_data(bigram_data)
    
    # Create the box-and-whisker plot
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(x='type', y='time', data=plot_data)
    
    # Set title and labels
    plt.title('Chosen vs unchosen bigram typing times', fontsize=16)
    plt.xlabel('Bigram type', fontsize=14)
    plt.ylabel('Typing time (ms)', fontsize=14)

    # Set y-axis limit to 500ms
    plt.ylim(0, 500)

    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'chosen_vs_unchosen_times_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"chosen_vs_unchosen_times_boxplot plot saved in {output_plots_folder}")
    print(f"Median chosen time: {plot_data[plot_data['type'] == 'chosen']['time'].median():.2f} ms")
    print(f"Median unchosen time: {plot_data[plot_data['type'] == 'unchosen']['time'].median():.2f} ms")


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

    print(f"chosen_vs_unchosen_times_scatter_regression plot saved in {output_plots_folder}")
    print(f"Correlation between chosen and unchosen times: {correlation:.2f}")


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

    print(f"chosen_vs_unchosen_times_joint plot saved in {output_plots_folder}")
    print(f"Correlation between chosen and unchosen times: {correlation:.2f}")


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

        # Load, combine, and save the data
        data = load_and_preprocess_data(input_folder, output_tables_folder, verbose=False)

        ##########################
        # PROCESS_BIGRAM_DATA
        ##########################
        # Process the bigram data
        bigram_data = process_bigram_data(data, output_tables_folder, verbose=False)
        
        ############################
        # Analyze improbable choices
        ############################
        # Load improbable pairs from CSV file
        current_dir = os.getcwd()  # Get the current working directory
        parent_dir = os.path.dirname(current_dir)  # Get the parent directory
        easy_choice_pairs_file = os.path.join(parent_dir, 'bigram_tables', 'bigram_5pairs_easy_choices_LH.csv')
        easy_choice_pairs = load_easy_choice_pairs(easy_choice_pairs_file)
        if easy_choice_pairs:
            # Analyze improbable choices
            suspicious_users = analyze_easy_choices(bigram_data, easy_choice_pairs, threshold=1)
        else:
            print("Skipping easy choice bigram analysis due to missing or invalid easy choice pairs data.")

        ###########################################
        # Analyze the bigram choice inconsistencies
        ###########################################
        bigram_choice_inconsistency_stats = analyze_choice_inconsistencies(bigram_data)

        ###########################################
        # Analyze the bigram typing times
        ###########################################
        typing_time_stats = analyze_typing_times(bigram_data)

        # Analyze the bigram typing times vs. choice
        choice_time_stats = analyze_choice_times(bigram_data)

        ##########################
        # PLOTS
        ##########################
        plot_median_bigram_times(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_barplot(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_scatter_regression(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_boxplot(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_boxplot(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_joint(bigram_data, output_plots_folder)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()