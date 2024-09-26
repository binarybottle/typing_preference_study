
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
        
        print(f"\nLoaded {len(easy_choice_pairs)} bigram pairs from {file_path} where one bigram is an easy choice.\n")
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
    def get_improbable_choice(row):
        for pair in easy_choice_pairs:
            if set(pair) == set([row['chosen_bigram'], row['unchosen_bigram']]):
                if row['chosen_bigram'] == pair[1]:  # If the improbable bigram was chosen
                    return pair  # Return the full pair (probable, improbable)
        return None

    # Add columns indicating if each choice was improbable and what the pair was
    bigram_data['improbable_pair'] = bigram_data.apply(get_improbable_choice, axis=1)
    bigram_data['improbable_chosen'] = bigram_data['improbable_pair'].apply(lambda x: x[1] if x else None)

    # Count the number of improbable choices for each user
    user_improbable_counts = bigram_data.groupby('user_id')['improbable_chosen'].apply(lambda x: x.notnull().sum())

    # Identify users with suspiciously high numbers of improbable choices
    suspicious_users = user_improbable_counts[user_improbable_counts >= threshold].reset_index()
    suspicious_users.columns = ['user_id', 'improbable_choice_count']

    # Calculate overall statistics
    total_users = bigram_data['user_id'].nunique()
    total_choices = len(bigram_data)
    total_improbable_choices = bigram_data['improbable_chosen'].notnull().sum()
    total_suspicious = len(suspicious_users)
    avg_improbable_count = user_improbable_counts.mean()

    # Calculate frequency of improbable bigrams chosen and their probable counterparts
    improbable_bigram_freq = bigram_data['improbable_pair'].value_counts(dropna=True).reset_index()
    improbable_bigram_freq.columns = ['bigram_pair', 'frequency']
    improbable_bigram_freq['probable_bigram'] = improbable_bigram_freq['bigram_pair'].apply(lambda x: x[0])
    improbable_bigram_freq['improbable_bigram'] = improbable_bigram_freq['bigram_pair'].apply(lambda x: x[1])
    
    # Calculate median and MAD slider values for each improbable bigram pair
    def get_slider_stats(pair):
        mask = bigram_data['improbable_pair'] == pair
        slider_values = bigram_data.loc[mask, 'sliderValue']
        median = slider_values.median()
        mad = np.median(np.abs(slider_values - median))
        return pd.Series({'median_slider': median, 'mad_slider': mad})

    slider_stats = improbable_bigram_freq['bigram_pair'].apply(get_slider_stats)
    improbable_bigram_freq = pd.concat([improbable_bigram_freq, slider_stats], axis=1)
    
    improbable_bigram_freq = improbable_bigram_freq.drop('bigram_pair', axis=1)

    print("\n____ Improbable Bigram Choice Analysis ____\n")
    print(f"Total users analyzed: {total_users}")
    print(f"Total choices analyzed: {total_choices}")
    print(f"Total improbable choices: {total_improbable_choices}")
    print(f"Users with >{threshold} improbable choices: {total_suspicious} ({(total_suspicious / total_users) * 100:.2f}%)")

    if not suspicious_users.empty:
        print("\nTop 10 users with highest improbable choice counts:")
        print(suspicious_users.sort_values('improbable_choice_count', ascending=False).head(10))

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

    # Count occurrences of each standardized bigram pair
    pair_counts = bigram_data['std_bigram_pair'].value_counts()

    # Filter for pairs that appear more than once (potential for inconsistency)
    potential_inconsistent_pairs = pair_counts[pair_counts > 1].index

    # Filter bigram_data to only include these pairs
    filtered_data = bigram_data[bigram_data['std_bigram_pair'].isin(potential_inconsistent_pairs)]

    # Total users
    total_users = filtered_data['user_id'].nunique()

    # Total unique bigram pairs that could be inconsistent
    total_pairs = len(potential_inconsistent_pairs)

    # Total bigram pair choices (each user may have multiple choices for each pair)
    total_choices = len(filtered_data)

    # Inconsistencies
    inconsistencies = filtered_data[filtered_data['is_consistent'].eq(False)]

    # Users with at least one inconsistency
    users_with_inconsistencies = inconsistencies['user_id'].nunique()

    # Number of unique bigram pairs with at least one inconsistency
    total_inconsistent_pairs = inconsistencies['std_bigram_pair'].nunique()

    # Inconsistent choices per user
    inconsistent_choices_per_user = inconsistencies.groupby('user_id')['std_bigram_pair'].count()

    # Most common inconsistent pairs
    inconsistent_pair_counts = inconsistencies['std_bigram_pair'].value_counts()

    print("\n____ Bigram Choice Inconsistency Statistics ____\n")

    print(f"Total users: {total_users}")
    print(f"Total unique bigram pairs that could be inconsistent: {total_pairs}")
    print(f"Total bigram choices for these pairs: {total_choices}")
    print(f"Users with at least one inconsistency: {users_with_inconsistencies}")
    print(f"Unique bigram pairs with at least one inconsistency: {total_inconsistent_pairs}")

    # Consistency frequencies
    consistency_frequencies = filtered_data['is_consistent'].value_counts()
    print(f"\nFrequency of consistent and inconsistent choices:")
    for count, freq in consistency_frequencies.items():
        print(f"{'Consistent' if count else 'Inconsistent'}: {freq} ({freq/total_choices:.2%})")

    # Inconsistency frequencies per pair
    print(f"\nFrequency of inconsistencies per bigram pair:")
    for pair, count in inconsistent_pair_counts.items():
        total_users_for_pair = filtered_data[filtered_data['std_bigram_pair'] == pair]['user_id'].nunique()
        print(f"'{pair}': {count} out of {total_users_for_pair} users")

    # Inconsistency frequencies per user
    print(f"\nFrequency of inconsistent choices per user:")
    for count, freq in inconsistent_choices_per_user.value_counts().sort_index().items():
        print(f"{count} inconsistent choices: {freq} users")
    
    if inconsistent_choices_per_user.empty:
        print("\nNo inconsistent choices found.")
    else:
        print(f"\nInconsistent choices per user:")
        print(f"  Avg: {inconsistent_choices_per_user.mean():.2f}")
        print(f"  Med: {inconsistent_choices_per_user.median():.2f}")
        print(f"  Max: {inconsistent_choices_per_user.max()}")
    
    print(f"\nMost common inconsistent bigram pairs:")
    print(inconsistent_pair_counts.head(10).to_string())

    # Return the statistics for further use if needed
    bigram_choice_inconsistency_stats = {
        'total_users': total_users,
        'total_pairs': total_pairs,
        'total_choices': total_choices,
        'users_with_inconsistencies': users_with_inconsistencies,
        'total_inconsistent_pairs': total_inconsistent_pairs,
        'inconsistent_choices_per_user': inconsistent_choices_per_user,
        'inconsistent_pair_counts': inconsistent_pair_counts,
        'consistency_frequencies': consistency_frequencies
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


def plot_bigram_choice_inconsistency_histogram(bigram_choice_inconsistency_stats, output_plots_folder):
    """Plot the histogram of inconsistent pair counts"""

    inconsistent_pair_counts = bigram_choice_inconsistency_stats['inconsistent_pair_counts']

    # Plot the histogram using seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.histplot(inconsistent_pair_counts, bins=10, kde=False, color='skyblue', edgecolor='black')

    plt.title('Histogram of inconsistent pair counts', fontsize=16)
    plt.xlabel('Number of participants', fontsize=14)
    plt.ylabel('Number of bigram pairs', fontsize=14)
    
    # Save the figure
    plt.savefig(os.path.join(output_plots_folder, 'inconsistently_chosen_bigram_pair_counts.png'), dpi=300, bbox_inches='tight')
    
    # Close the plot
    plt.close()

    print(f"bigram_choice_inconsistency_histogram plot saved in {output_plots_folder}")


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
    ax.set_title(f'Median typing times for each bigram (with MAD)', fontsize=16)
    ax.set_xlabel('Bigram', fontsize=12)
    ax.set_ylabel('Median time (ms)', fontsize=12)

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

    print(f"bigram_median_times_barplot_with_mad plot saved in {output_plots_folder}")


def plot_chosen_vs_unchosen_times_barplot(bigram_data, output_plots_folder):
    """
    Generate and save bar plot for chosen vs. unchosen bigram typing times.
    
    Y-axis: Each bigram pair (e.g., 'th, ht', 'er, re', etc.)
    X-axis: Median typing time in milliseconds
    Bars: For each bigram pair, there are two bars:
    One representing the median typing time when this bigram was chosen
    One representing the median typing time when this bigram was not chosen

    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    """
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
    
    plt.title('Median typing times: chosen vs unchosen bigrams')
    plt.xlabel('Median typing time (ms)')
    plt.ylabel('Bigram pair')
    plt.legend(title='Bigram type', labels=['Chosen', 'Unchosen'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'chosen_vs_unchosen_times.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"chosen_vs_unchosen_times plot saved in {output_plots_folder}")


def plot_chosen_vs_unchosen_times_boxplot(bigram_data, output_plots_folder):
    """
    Generate and save box-and-whisker plot for chosen vs. unchosen bigram typing times.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    """
    # Prepare data
    plot_data = bigram_data.melt(id_vars=['bigram_pair'], 
                                 value_vars=['chosen_bigram_time', 'unchosen_bigram_time'],
                                 var_name='Type', value_name='Time')

    # Create the box-and-whisker plot
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(x='Type', y='Time', data=plot_data)
    
    # Set title and labels
    plt.title('Boxplot of typing times: chosen vs unchosen bigrams', fontsize=16)
    plt.xlabel('Bigram type', fontsize=14)
    plt.ylabel('Typing time (ms)', fontsize=14)

    # Set y-axis limit to a maximum of 400ms
    plt.ylim(0, 400)

    # Save plot without showing
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'chosen_vs_unchosen_times_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"chosen_vs_unchosen_times_boxplot plot saved in {output_plots_folder}")
        

def plot_chosen_vs_unchosen_times_scatter(bigram_data, output_plots_folder):
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
    
    plt.title('Chosen vs unchosen bigram median typing times')
    plt.xlabel('Chosen bigram time (ms)')
    plt.ylabel('Unchosen bigram time (ms)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'chosen_vs_unchosen_times_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"chosen_vs_unchosen_times_scatter plot saved in {output_plots_folder}")


def plot_chosen_vs_unchosen_times_scatter_regression(bigram_data, output_plots_folder):
    """
    Generate and save scatter plot with regression line for chosen vs. unchosen typing times.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    """
    # Prepare data
    plot_data = bigram_data.groupby('bigram_pair').agg({
        'chosen_bigram_time': 'median',
        'unchosen_bigram_time': 'median'
    }).reset_index()
    
    # Scatter plot with regression line
    plt.figure(figsize=(10, 8))
    sns.regplot(x='chosen_bigram_time', y='unchosen_bigram_time', data=plot_data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    # Set title and labels
    max_val = max(plot_data['chosen_bigram_time'].max(), plot_data['unchosen_bigram_time'].max())
    plt.plot([0, max_val])
    plt.title('Chosen vs unchosen bigram typing times', fontsize=16)
    plt.xlabel('Chosen bigram time (ms)', fontsize=14)
    plt.ylabel('Unchosen bigram time (ms)', fontsize=14)
    
    # Save plot without showing
    plt.tight_layout()
    plt.savefig(os.path.join(output_plots_folder, 'chosen_vs_unchosen_times_scatter_regression.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"chosen_vs_unchosen_times_scatter_regression plot saved in {output_plots_folder}")


def plot_chosen_vs_unchosen_times_joint(bigram_data, output_plots_folder):
    """
    Generate and save joint plot of chosen vs. unchosen typing times.
    
    Parameters:
    - bigram_data: DataFrame containing processed bigram data
    - output_plots_folder: String path to the folder where plots should be saved
    """
    # Prepare data
    plot_data = bigram_data.groupby('bigram_pair').agg({
        'chosen_bigram_time': 'median',
        'unchosen_bigram_time': 'median'
    }).reset_index()

    # Create joint plot
    g = sns.jointplot(x='chosen_bigram_time', y='unchosen_bigram_time', data=plot_data, kind="scatter", marginal_kws=dict(bins=20, fill=True))
   
    # Add regression line
    g = g.plot_joint(sns.regplot, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    max_val = max(plot_data['chosen_bigram_time'].max(), plot_data['unchosen_bigram_time'].max())
    plt.plot([0, max_val])
    plt.suptitle('Joint plot of chosen vs unchosen bigram typing times', fontsize=16, y=1.03)
    plt.xlabel('Chosen bigram time (ms)', fontsize=14)
    plt.ylabel('Unchosen bigram time (ms)', fontsize=14)

    # Save plot without showing
    g.savefig(os.path.join(output_plots_folder, 'chosen_vs_unchosen_times_joint.png'), dpi=300)

    print(f"chosen_vs_unchosen_times_joint plot saved in {output_plots_folder}")


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
        easy_choice_pairs_file = os.path.join(parent_dir, 'bigram_tables', 'bigram_8pairs_easy_choices_LH.csv')
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
        plot_bigram_choice_inconsistency_histogram(bigram_choice_inconsistency_stats, output_plots_folder)
        plot_median_bigram_times(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_barplot(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_scatter(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_scatter_regression(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_boxplot(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_boxplot(bigram_data, output_plots_folder)
        plot_chosen_vs_unchosen_times_joint(bigram_data, output_plots_folder)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()