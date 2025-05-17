import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from collections import defaultdict
from scipy import stats

# Define the left home block keys and mirror pairs
LEFT_HOME_KEYS = ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']
MIRROR_PAIRS = [
    ('q', 'p'), ('w', 'o'), ('e', 'i'), ('r', 'u'),
    ('a', ';'), ('s', 'l'), ('d', 'k'), ('f', 'j'), ('g', 'h'),
    ('z', '/'), ('x', '.'), ('c', ','), ('v', 'm')
]

#-------------------------------------------------------------------------------
# Load and process data
#-------------------------------------------------------------------------------
def load_frequency_data(letter_freq_path=None, bigram_freq_path=None):
    """
    Load frequency data from CSV files
    Parameters:
    letter_freq_path (str): Path to letter frequency CSV
    bigram_freq_path (str): Path to bigram frequency CSV
    Returns:
    tuple: (letter_frequencies, bigram_frequencies) as dataframes
    """
    letter_frequencies = None
    bigram_frequencies = None
    try:
        if letter_freq_path and os.path.exists(letter_freq_path):
            letter_frequencies = pd.read_csv(letter_freq_path)
            print(f"Loaded letter frequency data: {len(letter_frequencies)} entries")
    except Exception as e:
        print(f"Error loading letter frequency data: {str(e)}")
    try:
        if bigram_freq_path and os.path.exists(bigram_freq_path):
            bigram_frequencies = pd.read_csv(bigram_freq_path)
            print(f"Loaded bigram frequency data: {len(bigram_frequencies)} entries")
    except Exception as e:
        print(f"Error loading bigram frequency data: {str(e)}")
    return letter_frequencies, bigram_frequencies

def process_typing_data(data):
    """Process typing data to calculate typing time between consecutive keystrokes"""
    # Make an explicit copy of the DataFrame to avoid the SettingWithCopyWarning
    data = data.copy()
    # Convert expectedKey and typedKey to string if they're not already
    data.loc[:, 'expectedKey'] = data['expectedKey'].astype(str)
    data.loc[:, 'typedKey'] = data['typedKey'].astype(str)
    # Replace 'nan' strings with empty strings
    data.loc[:, 'expectedKey'] = data['expectedKey'].replace('nan', '')
    data.loc[:, 'typedKey'] = data['typedKey'].replace('nan', '')
    # Ensure isCorrect is boolean
    data.loc[:, 'isCorrect'] = data['isCorrect'].map(lambda x: str(x).lower() == 'true')
    processed = []
    # Sort by user and timestamp to ensure correct sequence
    sorted_data = data.sort_values(by=['user_id', 'keydownTime'])
    # Group by user_id
    for user_id, user_data in sorted_data.groupby('user_id'):
        user_rows = user_data.to_dict('records')
        # Calculate typing time for each keystroke
        for i in range(1, len(user_rows)):
            current = user_rows[i]
            previous = user_rows[i-1]
            # Calculate time difference in milliseconds
            typing_time = current['keydownTime'] - previous['keydownTime']
            # Add processed data
            processed.append({
                'user_id': user_id,
                'trialId': current['trialId'],
                'expectedKey': current['expectedKey'],
                'typedKey': current['typedKey'],
                'isCorrect': current['isCorrect'],
                'typingTime': typing_time,
                'keydownTime': current['keydownTime']
            })
    return pd.DataFrame(processed)

#-------------------------------------------------------------------------------
# Analyze keys
#-------------------------------------------------------------------------------
def analyze_key_stats(data, keys_to_include=None):
    """
    Analyze 1-key and 2-key statistics for typing data.
    
    For 1-key stats: Focus on individual keys in the context of being the second key in any bigram
    For 2-key stats: Focus on specific bigram combinations
    
    All calculations focus on space-flanked bigrams (space-key1-key2-space) and exclude same-key bigrams.
    
    Parameters:
    data (DataFrame): Processed typing data
    keys_to_include (list): List of keys to include in analysis (if None, include all)
    
    Returns:
    tuple: (key_stats, bigram_stats) - Dictionaries for individual key and bigram statistics
    """
    key_stats = {}  # For 1-key statistics
    bigram_stats = {}  # For 2-key statistics
    
    # Group by user_id and trialId
    grouped = data.groupby(['user_id', 'trialId'])
    
    # Process each trial
    for (user_id, trial_id), trial_data in grouped:
        # Sort by timestamp to ensure correct sequence
        sorted_trial = trial_data.sort_values('keydownTime').reset_index(drop=True)
        
        i = 0
        while i < len(sorted_trial) - 3:  # Need at least 4 keystrokes: space, key1, key2, space
            # Get 4 consecutive keystrokes
            first = sorted_trial.iloc[i]     # should be a space
            second = sorted_trial.iloc[i+1]  # key1
            third = sorted_trial.iloc[i+2]   # key2 (key of interest for 1-key stats)
            fourth = sorted_trial.iloc[i+3]  # should be a space
            
            # Check if we have a valid flanked bigram pattern: space, key1, key2, space
            if (isinstance(first['expectedKey'], str) and first['expectedKey'] == ' ' and
                isinstance(second['expectedKey'], str) and len(second['expectedKey']) == 1 and second['expectedKey'] != ' ' and
                isinstance(third['expectedKey'], str) and len(third['expectedKey']) == 1 and third['expectedKey'] != ' ' and
                isinstance(fourth['expectedKey'], str) and fourth['expectedKey'] == ' '):
                
                # Skip same-key bigrams (where key1 == key2)
                if second['expectedKey'] == third['expectedKey']:
                    i += 1
                    continue
                
                key1 = second['expectedKey']
                key2 = third['expectedKey']
                
                # Check if keys are in the specified list
                if keys_to_include:
                    # For 2-key stats, both keys must be in keys_to_include
                    if key1 not in keys_to_include or key2 not in keys_to_include:
                        i += 1
                        continue
                
                # === PROCESS INDIVIDUAL KEY STATS (1-key stats for key2) ===
                
                # Initialize stats for this key if not exists
                if key2 not in key_stats:
                    key_stats[key2] = {
                        'key': key2,
                        'totalCount': 0,
                        'errorCount': 0,
                        'error_types': defaultdict(int),
                        'timeSamples': [],
                        'totalTime': 0
                    }
                
                # Count occurrences for individual key
                key_stats[key2]['totalCount'] += 1
                
                # Check if the key was typed incorrectly
                if not third['isCorrect']:
                    # Count as one error regardless of how many attempts were made
                    key_stats[key2]['errorCount'] += 1
                    
                    # Record the incorrect key that was typed
                    if isinstance(third['typedKey'], str) and third['typedKey']:
                        key_stats[key2]['error_types'][third['typedKey']] += 1
                    
                    # Find all incorrect attempts for this key
                    error_position = i + 2
                    while (error_position < len(sorted_trial) and 
                           sorted_trial.iloc[error_position]['expectedKey'] == key2 and 
                           not sorted_trial.iloc[error_position]['isCorrect']):
                        # Record each incorrect key that was typed
                        typed_key = sorted_trial.iloc[error_position]['typedKey']
                        if isinstance(typed_key, str) and typed_key:
                            key_stats[key2]['error_types'][typed_key] += 1
                        error_position += 1
                elif second['isCorrect']:
                    # Only calculate timing if both keys are correct (no errors)
                    # Use time between key1 and key2
                    key_time = third['keydownTime'] - second['keydownTime']
                    if 50 <= key_time <= 2000:  # Reasonable time range
                        key_stats[key2]['timeSamples'].append(key_time)
                        key_stats[key2]['totalTime'] += key_time
                
                # === PROCESS BIGRAM STATS (2-key stats) ===
                
                # Form the bigram key
                bigram = key1 + key2
                
                # Initialize stats for this bigram if not exists
                if bigram not in bigram_stats:
                    bigram_stats[bigram] = {
                        'bigram': bigram,
                        'totalCount': 0,
                        'errorCount': 0,
                        'error_types': defaultdict(int),
                        'timeSamples': [],
                        'totalTime': 0
                    }
                
                # Count this bigram occurrence
                bigram_stats[bigram]['totalCount'] += 1
                
                # Check if key2 was typed incorrectly while key1 was correct
                if second['isCorrect'] and not third['isCorrect']:
                    # Count as one error regardless of how many attempts were made
                    bigram_stats[bigram]['errorCount'] += 1
                    
                    # Record the incorrect key that was typed
                    if isinstance(third['typedKey'], str) and third['typedKey']:
                        bigram_stats[bigram]['error_types'][third['typedKey']] += 1
                    
                    # Find all incorrect attempts for key2
                    error_position = i + 2
                    while (error_position < len(sorted_trial) and 
                           sorted_trial.iloc[error_position]['expectedKey'] == key2 and 
                           not sorted_trial.iloc[error_position]['isCorrect']):
                        # Record each incorrect key that was typed
                        typed_key = sorted_trial.iloc[error_position]['typedKey']
                        if isinstance(typed_key, str) and typed_key:
                            bigram_stats[bigram]['error_types'][typed_key] += 1
                        error_position += 1
                elif second['isCorrect'] and third['isCorrect']:
                    # Only calculate timing if both keys are correct (no errors)
                    # Use time between key1 and key2
                    bigram_time = third['keydownTime'] - second['keydownTime']
                    if 50 <= bigram_time <= 2000:  # Reasonable time range
                        bigram_stats[bigram]['timeSamples'].append(bigram_time)
                        bigram_stats[bigram]['totalTime'] += bigram_time
                
                # Skip past this bigram
                i += 3  # Move to position after the final space
            else:
                i += 1
    
    # Calculate statistics for individual keys
    for stats in key_stats.values():
        # Error rate
        stats['errorRate'] = stats['errorCount'] / stats['totalCount'] if stats['totalCount'] > 0 else 0
        
        # Timing statistics
        if len(stats['timeSamples']) > 0:
            stats['avgTime'] = stats['totalTime'] / len(stats['timeSamples'])
            stats['medianTime'] = np.median(stats['timeSamples'])
            # Calculate Median Absolute Deviation (MAD)
            deviations = np.abs(np.array(stats['timeSamples']) - stats['medianTime'])
            stats['timeMAD'] = np.median(deviations)
        else:
            stats['avgTime'] = 0
            stats['medianTime'] = 0
            stats['timeMAD'] = 0
        
        # Error MAD using bootstrap
        if stats['totalCount'] > 10:
            bootstrap_errors = []
            n_bootstrap = 1000
            for _ in range(n_bootstrap):
                sample = np.random.choice([0, 1], size=stats['totalCount'],
                                           p=[1-stats['errorRate'], stats['errorRate']])
                bootstrap_errors.append(np.mean(sample))
            stats['errorMAD'] = np.median(np.abs(np.array(bootstrap_errors) - stats['errorRate']))
        else:
            stats['errorMAD'] = 0
        
        # Most common error type
        if stats['error_types']:
            most_common_error = max(stats['error_types'].items(), key=lambda x: x[1])
            stats['most_common_mistype'] = most_common_error[0]
            stats['most_common_mistype_count'] = most_common_error[1]
        else:
            stats['most_common_mistype'] = ""
            stats['most_common_mistype_count'] = 0
    
    # Calculate statistics for bigrams
    for stats in bigram_stats.values():
        # Error rate
        stats['errorRate'] = stats['errorCount'] / stats['totalCount'] if stats['totalCount'] > 0 else 0
        
        # Timing statistics
        if len(stats['timeSamples']) > 0:
            stats['avgTime'] = stats['totalTime'] / len(stats['timeSamples'])
            stats['medianTime'] = np.median(stats['timeSamples'])
            # Calculate Median Absolute Deviation (MAD)
            deviations = np.abs(np.array(stats['timeSamples']) - stats['medianTime'])
            stats['timeMAD'] = np.median(deviations)
        else:
            stats['avgTime'] = 0
            stats['medianTime'] = 0
            stats['timeMAD'] = 0
        
        # Error MAD using bootstrap
        if stats['totalCount'] > 10:
            bootstrap_errors = []
            n_bootstrap = 1000
            for _ in range(n_bootstrap):
                sample = np.random.choice([0, 1], size=stats['totalCount'],
                                           p=[1-stats['errorRate'], stats['errorRate']])
                bootstrap_errors.append(np.mean(sample))
            stats['errorMAD'] = np.median(np.abs(np.array(bootstrap_errors) - stats['errorRate']))
        else:
            stats['errorMAD'] = 0
        
        # Most common error type
        if stats['error_types']:
            most_common_error = max(stats['error_types'].items(), key=lambda x: x[1])
            stats['most_common_mistype'] = most_common_error[0]
            stats['most_common_mistype_count'] = most_common_error[1]
        else:
            stats['most_common_mistype'] = ""
            stats['most_common_mistype_count'] = 0
    
    return key_stats, bigram_stats

def visualize_left_home_key_error_rates(key_df, min_count=10):
    """
    Create a visualization of error rates for individual left home block keys
    Parameters:
    key_df (DataFrame): DataFrame containing individual key statistics
    min_count (int): Minimum count threshold for visualization
    """
    # Filter for sufficient data
    filtered_df = key_df[key_df['totalCount'] > min_count].copy()
    if filtered_df.empty:
        print("No data available for key error rate visualization")
        return
    
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by error rate for visualization
    filtered_df = filtered_df.sort_values('errorRate', ascending=False)
    
    # Create error rate plot with MAD error bars
    plt.figure(figsize=(12, 6))
    
    # Convert keys to uppercase for display
    filtered_df['key_upper'] = filtered_df['key'].str.upper()
    
    # Create the bar plot
    bars = plt.bar(filtered_df['key_upper'], filtered_df['errorRate'])
    
    # Add error bars using error MAD
    plt.errorbar(
        x=range(len(filtered_df)), 
        y=filtered_df['errorRate'],
        yerr=filtered_df['errorMAD'], 
        fmt='none', 
        color='black', 
        capsize=5
    )
    
    plt.title('Error rates for left home block keys')
    plt.xlabel('Key')
    plt.ylabel('Error rate')
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0.001:  # Only add labels if error rate is non-negligible
            plt.text(bar.get_x() + bar.get_width()/2., height/2, f"{height*100:.1f}%",
                    ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}error_1key.png')
    plt.close()

def visualize_left_home_key_typing_times(key_df, min_count=10):
    """
    Create a visualization of median typing times for individual left home block keys
    Parameters:
    key_df (DataFrame): DataFrame containing individual key statistics
    min_count (int): Minimum count threshold for visualization
    """
    # Filter for sufficient data
    filtered_df = key_df[key_df['totalCount'] > min_count].copy()
    if filtered_df.empty:
        print("No data available for key typing time visualization")
        return
    
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by median time for visualization
    filtered_df = filtered_df.sort_values('medianTime', ascending=False)
    
    # Create median time plot with MAD error bars
    plt.figure(figsize=(12, 6))
    
    # Convert keys to uppercase for display
    filtered_df['key_upper'] = filtered_df['key'].str.upper()
    
    # Create the bar plot
    bars = plt.bar(filtered_df['key_upper'], filtered_df['medianTime'])
    
    # Add error bars using time MAD
    plt.errorbar(
        x=range(len(filtered_df)), 
        y=filtered_df['medianTime'],
        yerr=filtered_df['timeMAD'], 
        fmt='none', 
        color='black', 
        capsize=5
    )
    
    plt.title('Typing times for left home block keys')
    plt.xlabel('Key')
    plt.ylabel('Median typing time (ms)')
    
    # Add time labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2, f"{height:.0f}ms",
                ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}typing_time_1key.png')
    plt.close()

def visualize_left_home_key_error_vs_time(key_df, min_count=10):
    """
    Create a scatter plot comparing error rates vs. typing times for individual left home block keys
    Parameters:
    key_df (DataFrame): DataFrame containing individual key statistics
    min_count (int): Minimum count threshold for visualization
    """
    # Filter for sufficient data
    filtered_df = key_df[key_df['totalCount'] > min_count].copy()
    if filtered_df.empty:
        print("No data available for key error vs. time visualization")
        return
    
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert keys to uppercase for display
    filtered_df['key_upper'] = filtered_df['key'].str.upper()
    
    # Create scatter plot with error bars
    plt.figure(figsize=(10, 8))
    
    plt.errorbar(
        filtered_df['medianTime'], 
        filtered_df['errorRate'],
        xerr=filtered_df['timeMAD'], 
        yerr=filtered_df['errorMAD'],
        fmt='o', 
        alpha=0.7, 
        ecolor='lightgray', 
        capsize=3
    )
    
    # Add key labels, displaced from markers
    for _, row in filtered_df.iterrows():
        plt.annotate(
            row['key_upper'], 
            (row['medianTime'], row['errorRate']),
            xytext=(8, 0), 
            textcoords='offset points',
            fontsize=12, 
            ha='left', 
            va='center'
        )
    
    plt.title('Error rate vs. typing time for left home block keys')
    plt.xlabel('Median typing time (ms)')
    plt.ylabel('Error rate')
    plt.xlim(100, 400)
    plt.grid(True, alpha=0.3)
    
    # Add correlation statistics if there are enough data points
    if len(filtered_df) > 3:
        try:
            # Check if there's variation in both variables
            if (filtered_df['errorRate'].std() > 0 and 
                filtered_df['medianTime'].std() > 0):
                
                # Calculate correlation and p-value
                corr, p_value = stats.pearsonr(
                    filtered_df['medianTime'], 
                    filtered_df['errorRate']
                )
                
                # Add correlation text to plot
                plt.text(
                    0.05, 0.95, 
                    f"correlation: {corr:.2f}\np-value: {p_value:.3f}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    va='top',
                    bbox=dict(facecolor='white', alpha=0.7)
                )
                
                # Add trend line
                x = filtered_df['medianTime']
                y = filtered_df['errorRate']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8)
        except Exception as e:
            print(f"Could not calculate correlation: {e}")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}error_vs_typing_time_1key.png')
    plt.close()

#-------------------------------------------------------------------------------
# Analyze key-pairs
#-------------------------------------------------------------------------------
def visualize_left_home_bigram_typing_times(bigram_df, min_count=10):
    """
    Create a visualization of median typing times for left home block bigrams
    Parameters:
    bigram_df (DataFrame): DataFrame containing bigram statistics
    min_count (int): Minimum count threshold for visualization
    """
    # Filter for sufficient data with timing samples
    filtered_df = bigram_df[(bigram_df['totalCount'] > min_count) & 
                            (bigram_df['medianTime'] > 0)].copy()
    if filtered_df.empty:
        print("No data available for bigram typing time visualization")
        return
    
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by median time for visualization
    filtered_df = filtered_df.sort_values('medianTime', ascending=False)
    
    # Create median time plot with MAD error bars
    plt.figure(figsize=(20, 8))
    
    # Convert bigrams to uppercase for display
    filtered_df['bigram_upper'] = filtered_df['bigram'].str.upper()
    
    # Create the bar plot
    bars = plt.bar(filtered_df['bigram_upper'], filtered_df['medianTime'])
    
    # Add error bars using time MAD
    plt.errorbar(
        x=range(len(filtered_df)), 
        y=filtered_df['medianTime'],
        yerr=filtered_df['timeMAD'], 
        fmt='none', 
        color='black', 
        capsize=5
    )
    
    plt.title('Typing times for left home block bigrams')
    plt.xlabel('Bigram')
    plt.ylabel('Median typing time (ms)')
    plt.xticks(rotation=90)  # Vertical labels
    
    # Add time labels
    #for i, bar in enumerate(bars):
    #    height = bar.get_height()
    #    plt.text(bar.get_x() + bar.get_width()/2., height/2, f"{height:.0f}ms",
    #            ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}typing_time_2key.png')
    plt.close()

def visualize_left_home_bigram_error_rates(bigram_df, min_count=10):
    """
    Create a visualization of error rates for left home block bigrams
    Parameters:
    bigram_df (DataFrame): DataFrame containing bigram statistics
    min_count (int): Minimum count threshold for visualization
    """
    # Filter for sufficient data
    filtered_df = bigram_df[bigram_df['totalCount'] > min_count].copy()
    
    # Count total bigrams before filtering for errors
    total_bigrams = len(filtered_df)
    
    # Remove bigrams with zero error rates
    filtered_df = filtered_df[filtered_df['errorRate'] > 0]
    
    # Count shown vs not shown
    shown_bigrams = len(filtered_df)
    not_shown_bigrams = total_bigrams - shown_bigrams
    
    if filtered_df.empty:
        print("No data available for bigram error rate visualization after filtering")
        return
    
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by error rate for visualization
    filtered_df = filtered_df.sort_values('errorRate', ascending=False)
    
    # Create error rate plot with MAD error bars
    plt.figure(figsize=(24, 16))
    
    # Convert bigrams to uppercase for display
    filtered_df['bigram_upper'] = filtered_df['bigram'].str.upper()
    
    # Create the bar plot
    bars = plt.bar(filtered_df['bigram_upper'], filtered_df['errorRate'])
    
    # Add error bars using error MAD
    plt.errorbar(
        x=range(len(filtered_df)), 
        y=filtered_df['errorRate'],
        yerr=filtered_df['errorMAD'], 
        fmt='none', 
        color='black', 
        capsize=5
    )
    
    plt.title('Error rates for left home block bigrams')
    plt.xlabel('bigram')
    plt.ylabel('error rate')
    plt.xticks(rotation=90)  # Vertical labels
    plt.ylim(0, 0.1)

    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0.001:  # Only add labels if error rate is non-negligible
            plt.text(bar.get_x() + bar.get_width()/2., height/2, f"{height*100:.1f}%",
                    ha='center', color='white', fontweight='bold')
    
    # Add count of shown vs. not shown
    plt.text(
        0.75, 0.95,
        f"showing {shown_bigrams} bigrams\n(excluded {not_shown_bigrams} with no errors)",
        transform=plt.gca().transAxes,
        fontsize=10,
        va='top',
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}error_2key.png')
    plt.close()

def visualize_left_home_bigram_error_vs_time(bigram_df, min_count=10):
    """
    Create a scatter plot comparing error rates vs. typing times for left home block bigrams
    Parameters:
    bigram_df (DataFrame): DataFrame containing bigram statistics
    min_count (int): Minimum count threshold for visualization
    """
    # Filter for sufficient data
    initial_df = bigram_df[(bigram_df['totalCount'] > min_count) & 
                          (bigram_df['medianTime'] > 0)].copy()
    
    # Count total bigrams before filtering for errors
    total_bigrams = len(initial_df)
    
    # Remove bigrams with zero error rates
    filtered_df = initial_df[initial_df['errorRate'] > 0]
    
    # Count shown vs not shown
    shown_bigrams = len(filtered_df)
    not_shown_bigrams = total_bigrams - shown_bigrams
    
    if filtered_df.empty:
        print("No data available for bigram error vs. time visualization after filtering")
        return
    
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert bigrams to uppercase for display
    filtered_df['bigram_upper'] = filtered_df['bigram'].str.upper()
    
    # Create scatter plot with error bars
    plt.figure(figsize=(12, 10))
    
    # Calculate more aggressive axis limits to spread data
    max_time = filtered_df['medianTime'].max() * 1.05
    max_error = filtered_df['errorRate'].max() * 1.05
    
    plt.errorbar(
        filtered_df['medianTime'], 
        filtered_df['errorRate'],
        xerr=filtered_df['timeMAD'], 
        yerr=filtered_df['errorMAD'],
        fmt='o', 
        alpha=0.7, 
        ecolor='lightgray', 
        capsize=2  # Reduced for less overlap
    )
    
    # Add bigram labels, displaced from markers
    for _, row in filtered_df.iterrows():
        plt.annotate(
            row['bigram_upper'], 
            (row['medianTime'], row['errorRate']),
            xytext=(8, 0), 
            textcoords='offset points',
            fontsize=12, 
            ha='left', 
            va='center'
        )
    
    plt.title('Error rate vs. typing time for left home block bigrams')
    plt.xlabel('Median typing time (ms)')
    plt.ylabel('Error rate')
    plt.grid(True, alpha=0.3)
    
    # Set tighter axis limits to spread data more
    plt.xlim(100, 400)
    #plt.xlim(0, max_time)
    #plt.ylim(-0.01, max_error)
    plt.ylim(-0.01, 0.1)
    
    # Add count of shown vs. not shown - moved to right side
    plt.text(
        0.75, 0.85,
        f"showing {shown_bigrams} bigrams\n(excluded {not_shown_bigrams} with no errors)",
        transform=plt.gca().transAxes,
        fontsize=10,
        va='top',
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Add correlation statistics if there are enough data points
    if len(filtered_df) > 3:
        try:
            # Check if there's variation in both variables
            if (filtered_df['errorRate'].std() > 0 and 
                filtered_df['medianTime'].std() > 0):
                
                # Calculate correlation and p-value
                corr, p_value = stats.pearsonr(
                    filtered_df['medianTime'], 
                    filtered_df['errorRate']
                )
                
                # Add correlation text to plot - moved to right side
                plt.text(
                    0.75, 0.95, 
                    f"correlation: {corr:.2f}\np-value: {p_value:.3f}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    va='top',
                    bbox=dict(facecolor='white', alpha=0.7)
                )
                
                # Add trend line
                x = filtered_df['medianTime']
                y = filtered_df['errorRate']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8)
        except Exception as e:
            print(f"Could not calculate correlation: {e}")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}error_vs_typing_time_2key.png')
    plt.close()

def visualize_left_home_bigram_counts(bigram_df):
    """
    Create visualization of bigram counts
    Parameters:
    bigram_df (DataFrame): DataFrame containing bigram statistics
    """
    # Filter for sufficient data
    min_count = 10
    filtered_df = bigram_df[bigram_df['totalCount'] > min_count].copy()
    if filtered_df.empty:
        print("No data available for bigram visualization")
        return
    
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Bigram total counts
    plt.figure(figsize=(20, 8))
    # Sort by count for better visualization
    count_df = filtered_df.sort_values('totalCount', ascending=False)
    bars = plt.bar(count_df['bigram'].str.upper(), count_df['totalCount'])
    
    plt.title('Number of flanked bigram occurrences')
    plt.xlabel('Bigram')
    plt.ylabel('Count')
    plt.xticks(rotation=90)  # Make tick labels vertical instead of slanted
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}count_bigram.png')
    plt.close()

#-------------------------------------------------------------------------------
# Analyze 1-key vs. key-pairs
#-------------------------------------------------------------------------------
def visualize_key_vs_bigram_errors(key_df, bigram_df, min_count=10):
    """
    Create a scatter plot comparing 1-key error rates vs 2-key error rates
    Parameters:
    key_df (DataFrame): DataFrame containing individual key statistics
    bigram_df (DataFrame): DataFrame containing bigram statistics
    min_count (int): Minimum count threshold for visualization
    """
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the DataFrames for processing
    key_df_copy = key_df.copy()
    bigram_df_copy = bigram_df.copy()
    
    # Extract the second key from each bigram
    bigram_df_copy['second_key'] = bigram_df_copy['bigram'].str[1]
    
    # Filter for sufficient data
    filtered_keys = key_df_copy[key_df_copy['totalCount'] > min_count]
    filtered_bigrams = bigram_df_copy[(bigram_df_copy['totalCount'] > min_count)]
    
    # Create a DataFrame with matched pairs of 1-key and 2-key stats
    paired_data = []
    
    for _, key_row in filtered_keys.iterrows():
        key = key_row['key']
        key_upper = key.upper()
        
        # Find all bigrams where this key is the second key
        matching_bigrams = filtered_bigrams[filtered_bigrams['second_key'] == key]
        
        for _, bigram_row in matching_bigrams.iterrows():
            bigram = bigram_row['bigram']
            bigram_upper = bigram.upper()
            
            # Skip pairs where both error rates are zero
            if key_row['errorRate'] == 0 and bigram_row['errorRate'] == 0:
                continue
                
            paired_data.append({
                'key': key,
                'key_upper': key_upper,
                'bigram': bigram,
                'bigram_upper': bigram_upper,
                'key_errorRate': key_row['errorRate'],
                'key_errorMAD': key_row['errorMAD'],
                'bigram_errorRate': bigram_row['errorRate'],
                'bigram_errorMAD': bigram_row['errorMAD']
            })
    
    # Count shown vs not shown
    total_possible_pairs = len(filtered_keys) * len(filtered_bigrams.drop_duplicates('second_key'))
    shown_pairs = len(paired_data)
    not_shown_pairs = total_possible_pairs - shown_pairs
    
    if not paired_data:
        print("No matching key-bigram pairs found for error comparison")
        return
    
    paired_df = pd.DataFrame(paired_data)
    
    # Create the scatter plot with wider aspect ratio
    plt.figure(figsize=(14, 8))
    
    # Add points with error bars
    plt.errorbar(
        paired_df['key_errorRate'], 
        paired_df['bigram_errorRate'],
        xerr=paired_df['key_errorMAD'], 
        yerr=paired_df['bigram_errorMAD'],
        fmt='o', 
        alpha=0.7, 
        ecolor='lightgray', 
        capsize=3
    )
    
    # Determine axis limits with more horizontal spread
    max_x = paired_df['key_errorRate'].max() * 1.3 if paired_df['key_errorRate'].max() > 0 else 0.01
    max_y = paired_df['bigram_errorRate'].max() * 1.1 if paired_df['bigram_errorRate'].max() > 0 else 0.01
    max_error = max(max_x, max_y)
    
    # Add diagonal line (y=x)
    plt.plot([0, max_error], [0, max_error], 'k--', alpha=0.5, label='equal error rate')
    
    # Add labels for each point
    for _, row in paired_df.iterrows():
        plt.annotate(
            f"{row['bigram_upper']}",
            (row['key_errorRate'], row['bigram_errorRate']),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=10
        )
    
    plt.title('Error rates: 1-key vs. 2-key')
    plt.xlabel('1-key error rate')
    plt.ylabel('2-key error rate')
    plt.grid(True, alpha=0.3)
    
    # Add count of shown vs. not shown - moved to right side
    plt.text(
        0.75, 0.80,
        f"showing {shown_pairs} bigram pairs\n(excluded {not_shown_pairs} with no errors)",
        transform=plt.gca().transAxes,
        fontsize=10,
        va='bottom',
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Set axis limits to focus on data with more horizontal spread
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    
    # Calculate correlation
    if len(paired_df) > 3:
        try:
            # Check if there's variation in both variables
            if (paired_df['key_errorRate'].std() > 0 and 
                paired_df['bigram_errorRate'].std() > 0):
                
                corr, p_value = stats.pearsonr(
                    paired_df['key_errorRate'], 
                    paired_df['bigram_errorRate']
                )
                
                plt.text(
                    0.75, 0.95, 
                    f"correlation: {corr:.2f}\np-value: {p_value:.3f}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    va='top',
                    bbox=dict(facecolor='white', alpha=0.7)
                )
                
                # Add trend line
                x = paired_df['key_errorRate']
                y = paired_df['bigram_errorRate']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8)
        except Exception as e:
            print(f"Could not calculate correlation for errors: {e}")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}error_1key_vs_2key.png')
    plt.close()

def visualize_key_vs_bigram_times(key_df, bigram_df, min_count=10):
    """
    Create a scatter plot comparing 1-key times vs 2-key times
    Parameters:
    key_df (DataFrame): DataFrame containing individual key statistics
    bigram_df (DataFrame): DataFrame containing bigram statistics
    min_count (int): Minimum count threshold for visualization
    """
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy of the DataFrames for processing
    key_df_copy = key_df.copy()
    bigram_df_copy = bigram_df.copy()
    
    # Extract the second key from each bigram
    bigram_df_copy['second_key'] = bigram_df_copy['bigram'].str[1]
    
    # Filter for sufficient data
    filtered_keys = key_df_copy[key_df_copy['totalCount'] > min_count]
    filtered_bigrams = bigram_df_copy[(bigram_df_copy['totalCount'] > min_count) & 
                                     (bigram_df_copy['medianTime'] > 0)]
    
    # Create a DataFrame with matched pairs of 1-key and 2-key stats
    paired_data = []
    
    for _, key_row in filtered_keys.iterrows():
        key = key_row['key']
        key_upper = key.upper()
        
        # Find all bigrams where this key is the second key
        matching_bigrams = filtered_bigrams[filtered_bigrams['second_key'] == key]
        
        for _, bigram_row in matching_bigrams.iterrows():
            bigram = bigram_row['bigram']
            bigram_upper = bigram.upper()
            
            paired_data.append({
                'key': key,
                'key_upper': key_upper,
                'bigram': bigram,
                'bigram_upper': bigram_upper,
                'key_medianTime': key_row['medianTime'],
                'key_timeMAD': key_row['timeMAD'],
                'bigram_medianTime': bigram_row['medianTime'],
                'bigram_timeMAD': bigram_row['timeMAD']
            })
    
    if not paired_data:
        print("No matching key-bigram pairs found for timing comparison")
        return
    
    paired_df = pd.DataFrame(paired_data)
    
    # Create the scatter plot
    plt.figure(figsize=(12, 10))
    
    # Calculate more aggressive axis limits to spread data
    max_x = paired_df['key_medianTime'].max() * 1.05
    max_y = paired_df['bigram_medianTime'].max() * 1.05
    
    # Add points with error bars
    plt.errorbar(
        paired_df['key_medianTime'], 
        paired_df['bigram_medianTime'],
        xerr=paired_df['key_timeMAD'], 
        yerr=paired_df['bigram_timeMAD'],
        fmt='o', 
        alpha=0.7, 
        ecolor='lightgray', 
        capsize=2  # Reduced for less overlap
    )
    
    # Add diagonal line (y=x)
    plt.plot([0, max(max_x, max_y)], [0, max(max_x, max_y)], 'k--', alpha=0.5, label='Equal Time')
    
    # Add labels for each point
    for _, row in paired_df.iterrows():
        plt.annotate(
            f"{row['bigram_upper']}",
            (row['key_medianTime'], row['bigram_medianTime']),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=10
        )
    
    plt.title('Typing times: 1-key vs. 2-key')
    plt.xlabel('1-key median typing time (ms)')
    plt.ylabel('2-key median typing time (ms)')
    plt.grid(True, alpha=0.3)
    
    # Set tighter axis limits to spread data more
    plt.xlim(100, 400)
    plt.ylim(50, 400)
    #plt.xlim(0, max_x)
    #plt.ylim(0, max_y)
    
    # Add region labels in better positions to avoid overlap
    plt.text(max_x*0.2, max_y*0.9, '2-key time > 1-key time',
             ha='center', fontsize=12, alpha=0.7)
    plt.text(max_x*0.9, max_y*0.2, '1-key time > 2-key time',
             ha='center', fontsize=12, alpha=0.7)
    
    # Calculate correlation
    if len(paired_df) > 3:
        try:
            # Check if there's variation in both variables
            if (paired_df['key_medianTime'].std() > 0 and 
                paired_df['bigram_medianTime'].std() > 0):
                
                corr, p_value = stats.pearsonr(
                    paired_df['key_medianTime'], 
                    paired_df['bigram_medianTime']
                )
                
                plt.text(
                    0.75, 0.95, 
                    f"correlation: {corr:.2f}\np-value: {p_value:.3f}",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    va='top',
                    bbox=dict(facecolor='white', alpha=0.7)
                )
                
                # Add trend line
                x = paired_df['key_medianTime']
                y = paired_df['bigram_medianTime']
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8)
        except Exception as e:
            print(f"Could not calculate correlation for times: {e}")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}typing_time_1key_vs_2key.png')
    plt.close()
    
#-------------------------------------------------------------------------------
# Mirrored key pairs
#-------------------------------------------------------------------------------
def analyze_mirror_pairs(key_stats, letter_freq=None):
    """
    Analyze mirror image key pairs using the new key stats definitions
    Parameters:
    key_stats (dict): Dictionary of key statistics from analyze_key_stats
    letter_freq (DataFrame): Optional letter frequency data for regression
    Returns:
    DataFrame: Comparison of mirror pair statistics
    """
    mirror_data = []
    # Create frequency dictionary if provided
    freq_dict = {}
    if letter_freq is not None:
        # Try different column names for letter frequency data
        if 'item' in letter_freq.columns and 'score' in letter_freq.columns:
            for _, row in letter_freq.iterrows():
                letter_value = str(row['item']).lower()
                if len(letter_value) == 1:  # Only use single characters
                    freq_dict[letter_value] = row['score']
        else:
            key_col = None
            freq_col = None
            # Try to find letter column
            for col in ['letter', 'char', 'character', 'item', 'key']:
                if col in letter_freq.columns:
                    key_col = col
                    break
            # Try to find frequency column
            for col in ['frequency', 'freq', 'count', 'score', 'probability']:
                if col in letter_freq.columns:
                    freq_col = col
                    break
            if key_col and freq_col:
                for _, row in letter_freq.iterrows():
                    letter_value = str(row[key_col]).lower()
                    if len(letter_value) == 1:  # Only use single characters
                        freq_dict[letter_value] = row[freq_col]
    
    # Filter mirror pairs to only include home block keys
    home_block_mirror_pairs = []
    for left, right in MIRROR_PAIRS:
        if left in LEFT_HOME_KEYS:
            home_block_mirror_pairs.append((left, right))
    
    for left, right in home_block_mirror_pairs:
        if left in key_stats and right in key_stats:
            left_stats = key_stats[left]
            right_stats = key_stats[right]
            # Only include pairs where both keys have sufficient data
            if left_stats['totalCount'] > 10 and right_stats['totalCount'] > 10:
                pair_data = {
                    'left_key': left,
                    'right_key': right,
                    'left_count': left_stats['totalCount'],
                    'right_count': right_stats['totalCount'],
                    'left_error_rate': left_stats['errorRate'],
                    'right_error_rate': right_stats['errorRate'],
                    'left_median_time': left_stats['medianTime'],
                    'right_median_time': right_stats['medianTime'],
                    'left_time_mad': left_stats['timeMAD'],
                    'right_time_mad': right_stats['timeMAD'],
                    'left_error_mad': left_stats['errorMAD'],
                    'right_error_mad': right_stats['errorMAD'],
                    'left_most_common_mistype': left_stats['most_common_mistype'],
                    'right_most_common_mistype': right_stats['most_common_mistype'],
                    'error_diff': left_stats['errorRate'] - right_stats['errorRate'],
                    'time_diff': left_stats['medianTime'] - right_stats['medianTime']
                }
                # Add frequency info if available
                if freq_dict:
                    pair_data['left_freq'] = freq_dict.get(left, 0)
                    pair_data['right_freq'] = freq_dict.get(right, 0)
                mirror_data.append(pair_data)
    mirror_df = pd.DataFrame(mirror_data)
    # If frequency data is available, adjust mirror pair metrics
    if 'left_freq' in mirror_df.columns and 'right_freq' in mirror_df.columns:
        mirror_df = adjust_mirror_pairs_for_frequency(mirror_df)
    return mirror_df

def visualize_mirror_pairs(mirror_df):
    """
    Create visualizations comparing mirror image key pairs
    Parameters:
    mirror_df (DataFrame): DataFrame of mirror pair comparison statistics
    """
    if mirror_df.empty:
        print("No data available for mirror pairs visualization")
        return
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    # Sort mirror pairs for consistent visualization
    mirror_df = mirror_df.sort_values('left_key')
    
    # Convert keys to uppercase for display
    mirror_df['left_key_upper'] = mirror_df['left_key'].str.upper()
    mirror_df['right_key_upper'] = mirror_df['right_key'].str.upper()
    
    # 1. Error rate comparison with MAD error bars
    plt.figure(figsize=(15, 8))
    # Prepare data for grouped bar plot
    error_data = []
    for _, row in mirror_df.iterrows():
        error_data.append({'pair': f"{row['left_key_upper']}-{row['right_key_upper']}",
                           'position': 'Left',
                           'error_rate': row['left_error_rate'],
                           'error_mad': row['left_error_mad']})
        error_data.append({'pair': f"{row['left_key_upper']}-{row['right_key_upper']}",
                           'position': 'Right',
                           'error_rate': row['right_error_rate'],
                           'error_mad': row['right_error_mad']})
    error_df = pd.DataFrame(error_data)
    # Create grouped bar plot with error bars
    ax = sns.barplot(x='pair', y='error_rate', hue='position', data=error_df)
    # Add error bars manually - with index safety check
    for i, p in enumerate(ax.patches):
        if i < len(error_df):  # Add safety check
            row = error_df.iloc[i]
            height = p.get_height()
            ax.errorbar(p.get_x() + p.get_width()/2., height,
                        yerr=row['error_mad'], fmt='none', color='black', capsize=3)
            ax.text(p.get_x() + p.get_width()/2., height/2, f"{height*100:.1f}%",
                    ha='center', color='white', fontweight='bold')
    plt.title('error rate for mirror letter pairs')
    plt.xlabel('mirror letter pair')
    plt.ylabel('error rate')
    plt.xticks(rotation=45)
    plt.legend(title='key position')
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_error_rates_with_mad.png')
    plt.close()
    
    # 2. Typing time comparison with MAD error bars
    plt.figure(figsize=(15, 8))
    # Prepare data for grouped bar plot
    time_data = []
    for _, row in mirror_df.iterrows():
        time_data.append({'pair': f"{row['left_key_upper']}-{row['right_key_upper']}",
                          'position': 'Left',
                          'median_time': row['left_median_time'],
                          'time_mad': row['left_time_mad']})
        time_data.append({'pair': f"{row['left_key_upper']}-{row['right_key_upper']}",
                          'position': 'Right',
                          'median_time': row['right_median_time'],
                          'time_mad': row['right_time_mad']})
    time_df = pd.DataFrame(time_data)
    # Create grouped bar plot with error bars
    ax = sns.barplot(x='pair', y='median_time', hue='position', data=time_df)
    # Add error bars manually - with index safety check
    for i, p in enumerate(ax.patches):
        if i < len(time_df):  # Add safety check
            row = time_df.iloc[i]
            height = p.get_height()
            ax.errorbar(p.get_x() + p.get_width()/2., height,
                        yerr=row['time_mad'], fmt='none', color='black', capsize=3)
            ax.text(p.get_x() + p.get_width()/2., height/2, f"{height:.0f}ms",
                    ha='center', color='white', fontweight='bold')
    plt.title('typing times for left-right mirror letter pairs')
    plt.xlabel('mirror letter pair')
    plt.ylabel('median typing time (ms)')
    plt.xticks(rotation=45)
    plt.legend(title='key position')
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_typing_time.png')
    plt.close()
    
    # 3. Scatter plot comparing left vs right error rates with MAD error bars
    plt.figure(figsize=(10, 10))
    # Check if there's variance in the data
    error_range = max(
        mirror_df['left_error_rate'].max() - mirror_df['left_error_rate'].min(),
        mirror_df['right_error_rate'].max() - mirror_df['right_error_rate'].min()
    )
    
    # Only create plot if there's enough variance in the data
    if error_range > 0.001:  # Using a small threshold to determine if data varies
        # Create scatter plot with error bars
        plt.errorbar(
            mirror_df['right_error_rate'], mirror_df['left_error_rate'],
            xerr=mirror_df['right_error_mad'], yerr=mirror_df['left_error_mad'],
            fmt='o', alpha=0.7, ecolor='lightgray', capsize=3
        )
        # Add diagonal line representing equal error rates
        max_error = max(mirror_df['left_error_rate'].max(), mirror_df['right_error_rate'].max()) * 1.1
        min_error = min(mirror_df['left_error_rate'].min(), mirror_df['right_error_rate'].min()) * 0.9
        if max_error > min_error:
            plt.plot([min_error, max_error], [min_error, max_error], 'k--', alpha=0.5, label='equal error rate')
        
        # Add point labels for each mirror pair
        for _, row in mirror_df.iterrows():
            plt.annotate(
                f"{row['left_key_upper']}-{row['right_key_upper']}",
                (row['right_error_rate'], row['left_error_rate']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10
            )
        
        # Add region labels if there's sufficient range
        if max_error > min_error * 1.1:
            plt.text(max_error*0.25, max_error*0.75, 'left keys more error-prone',
                     ha='center', fontsize=12, alpha=0.7)
            plt.text(max_error*0.75, max_error*0.25, 'right keys more error-prone',
                     ha='center', fontsize=12, alpha=0.7)
        
        plt.title('error rates for left-right mirror letter pairs')
        plt.ylabel('left key error rate')
        plt.xlabel('right key error rate')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}mirror_pair_error_left_vs_right.png')
    else:
        print("Skipping error rate scatter plot - insufficient variance in data")
    
    plt.close()
    
    # 4. Scatter plot comparing left vs right typing times with MAD and frequency adjustment
    plt.figure(figsize=(10, 10))
    # Check if there's variance in the data
    time_range = max(
        mirror_df['left_median_time'].max() - mirror_df['left_median_time'].min(),
        mirror_df['right_median_time'].max() - mirror_df['right_median_time'].min()
    )
    
    if time_range > 1.0:  # Only create plot if there's at least 1ms of variance
        # Create scatter plot with error bars
        plt.errorbar(
            mirror_df['right_median_time'], mirror_df['left_median_time'],
            xerr=mirror_df['right_time_mad'], yerr=mirror_df['left_time_mad'],
            fmt='o', alpha=0.7, ecolor='lightgray', capsize=3,
            label='raw'
        )
        # Add diagonal line representing equal typing times
        max_time = max(mirror_df['left_median_time'].max(), mirror_df['right_median_time'].max()) * 1.1
        min_time = min(mirror_df['left_median_time'].min(), mirror_df['right_median_time'].min()) * 0.9
        if max_time > min_time:
            plt.plot([min_time, max_time], [min_time, max_time], 'k--', alpha=0.5, label='equal typing time')
        
        # Add point labels for each mirror pair
        for _, row in mirror_df.iterrows():
            plt.annotate(
                f"{row['left_key_upper']}-{row['right_key_upper']}",
                (row['right_median_time'], row['left_median_time']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10
            )
        
        # Add frequency-adjusted points if available
        if 'left_median_time_freq_adjusted' in mirror_df.columns:
            plt.scatter(
                mirror_df['right_median_time_freq_adjusted'], mirror_df['left_median_time_freq_adjusted'],
                marker='x', color='red', alpha=0.7, s=50,
                label='frequency-adjusted'
            )
            # Connect raw and adjusted points with lines
            for _, row in mirror_df.iterrows():
                plt.plot(
                    [row['right_median_time'], row['right_median_time_freq_adjusted']],
                    [row['left_median_time'], row['left_median_time_freq_adjusted']],
                    'r-', alpha=0.3
                )
        
        # Add region labels if there's sufficient range
        if max_time > min_time * 1.1:
            plt.text(max_time*0.25, max_time*0.75, 'left keys slower',
                     ha='center', fontsize=12, alpha=0.7)
            plt.text(max_time*0.75, max_time*0.25, 'right keys slower',
                     ha='center', fontsize=12, alpha=0.7)
        
        plt.title('typing times for left-right mirror letter pairs')
        plt.ylabel('left key median typing time (ms)')
        plt.xlabel('right key median typing time (ms)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}mirror_pair_time_left_vs_right.png')
    else:
        print("Skipping time scatter plot - insufficient variance in data")
    
    plt.close()

def adjust_mirror_pairs_for_frequency(mirror_df):
    """
    Adjust mirror pair metrics by regressing out frequency effects
    Parameters:
    mirror_df (DataFrame): DataFrame of mirror pair comparison statistics
    Returns:
    DataFrame: DataFrame with frequency-adjusted metrics
    """
    # Check if frequency data is available
    if 'left_freq' not in mirror_df.columns or 'right_freq' not in mirror_df.columns:
        print("No frequency data available for adjustment")
        return mirror_df
    # Create a copy of the input DataFrame
    adjusted_df = mirror_df.copy()
    # Add log-transformed frequency (common in psycholinguistic research)
    adjusted_df['left_log_freq'] = np.log10(adjusted_df['left_freq'] + 1)  # Add 1 to handle zeros
    adjusted_df['right_log_freq'] = np.log10(adjusted_df['right_freq'] + 1)
    # Metrics to adjust
    left_metrics = ['left_error_rate', 'left_median_time']
    right_metrics = ['right_error_rate', 'right_median_time']
    # Adjust left key metrics
    for metric in left_metrics:
        if metric in adjusted_df.columns:
            try:
                # Create the model: metric ~ log_frequency
                X = sm.add_constant(adjusted_df['left_log_freq'])
                y = adjusted_df[metric]
                # Fit the model
                model = sm.OLS(y, X).fit()
                # Get the model parameters
                intercept = model.params[0]
                slope = model.params[1]
                # Calculate predicted values
                adjusted_df[f'{metric}_pred'] = intercept + slope * adjusted_df['left_log_freq']
                # Calculate residuals (the frequency-adjusted values) + mean
                adjusted_df[f'{metric}_freq_adjusted'] = adjusted_df[metric] - adjusted_df[f'{metric}_pred'] + adjusted_df[metric].mean()
                print(f"Adjusted {metric} for left keys (R² = {model.rsquared:.3f})")
            except Exception as e:
                print(f"Error adjusting {metric} for left keys: {e}")
    # Adjust right key metrics
    for metric in right_metrics:
        if metric in adjusted_df.columns:
            try:
                # Create the model: metric ~ log_frequency
                X = sm.add_constant(adjusted_df['right_log_freq'])
                y = adjusted_df[metric]
                # Fit the model
                model = sm.OLS(y, X).fit()
                # Get the model parameters
                intercept = model.params[0]
                slope = model.params[1]
                # Calculate predicted values
                adjusted_df[f'{metric}_pred'] = intercept + slope * adjusted_df['right_log_freq']
                # Calculate residuals (the frequency-adjusted values) + mean
                adjusted_df[f'{metric}_freq_adjusted'] = adjusted_df[metric] - adjusted_df[f'{metric}_pred'] + adjusted_df[metric].mean()
                print(f"Adjusted {metric} for right keys (R² = {model.rsquared:.3f})")
            except Exception as e:
                print(f"Error adjusting {metric} for right keys: {e}")
    # Calculate adjusted differences
    if 'left_error_rate_freq_adjusted' in adjusted_df.columns and 'right_error_rate_freq_adjusted' in adjusted_df.columns:
        adjusted_df['error_diff_freq_adjusted'] = adjusted_df['left_error_rate_freq_adjusted'] - adjusted_df['right_error_rate_freq_adjusted']
    if 'left_median_time_freq_adjusted' in adjusted_df.columns and 'right_median_time_freq_adjusted' in adjusted_df.columns:
        adjusted_df['time_diff_freq_adjusted'] = adjusted_df['left_median_time_freq_adjusted'] - adjusted_df['right_median_time_freq_adjusted']
    return adjusted_df

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
def verify_data_completeness(key_df, bigram_df):
    """
    Verify we have data for all expected keys and bigrams using the new definitions
    Parameters:
    key_df (DataFrame): DataFrame of individual key statistics
    bigram_df (DataFrame): DataFrame of bigram statistics
    """
    print("\n" + "="*50)
    print("data completeness verification")
    print("="*50)
    
    # Check individual keys
    all_left_home_keys = set(LEFT_HOME_KEYS)
    found_keys = set(key_df['key'])
    missing_keys = all_left_home_keys - found_keys
    
    print(f"\ntotal left home keys: {len(all_left_home_keys)}")
    print(f"keys with data: {len(found_keys)}")
    
    if missing_keys:
        print(f"missing keys: {', '.join(sorted(missing_keys)).upper()}")
    else:
        print("all left home block keys have data!")
    
    # Check bigrams (excluding same-key bigrams)
    all_possible_bigrams = set()
    for l1 in LEFT_HOME_KEYS:
        for l2 in LEFT_HOME_KEYS:
            # Skip same-key bigrams
            if l1 != l2:
                all_possible_bigrams.add(l1 + l2)
    
    found_bigrams = set(bigram_df['bigram'])
    missing_bigrams = all_possible_bigrams - found_bigrams
    
    print(f"\ntotal possible left home bigrams (excluding same-key bigrams): {len(all_possible_bigrams)}")
    print(f"bigrams with data: {len(found_bigrams)}")
    
    if len(missing_bigrams) > 10:
        print(f"missing {len(missing_bigrams)} bigrams, showing first 10: {', '.join(sorted(list(missing_bigrams)[:10])).upper()}")
    elif missing_bigrams:
        print(f"missing bigrams: {', '.join(sorted(missing_bigrams)).upper()}")
    else:
        print("all possible left home block bigrams have data!")
    
    # Verify that all bigrams are correctly structured (only left home keys)
    invalid_bigrams = []
    for bigram in found_bigrams:
        if len(bigram) != 2:
            invalid_bigrams.append(bigram)
        elif bigram[0] not in LEFT_HOME_KEYS or bigram[1] not in LEFT_HOME_KEYS:
            invalid_bigrams.append(bigram)
        elif bigram[0] == bigram[1]:  # Same-key bigram (should be excluded)
            invalid_bigrams.append(bigram)
            
    if invalid_bigrams:
        print(f"\nwarning: found {len(invalid_bigrams)} invalid bigrams: {', '.join(invalid_bigrams).upper()}")

def main():
    """Main function to execute the analysis pipeline"""
    # Get all CSV files in the directory
    csv_path = 'input/raws_Prolific/*.csv' # 'input/raws_Prolific/*.csv'  
    letter_freq_path = 'input/letter_frequencies_english.csv'
    bigram_freq_path = 'input/letter_pair_frequencies_english.csv'
    # Load frequency data
    letter_frequencies, bigram_frequencies = load_frequency_data(letter_freq_path, bigram_freq_path)
    csv_files = glob.glob(csv_path)
    # Check if any files were found
    if not csv_files:
        print(f"No CSV files found at path: {csv_path}")
        print("Please check the path and ensure CSV files exist there.")
        return None
    print(f"Found {len(csv_files)} CSV files")
    all_data = []
    # Process each file
    for i, file_path in enumerate(csv_files):
        try:
            # Read CSV file
            print(f"Reading file {i+1}/{len(csv_files)}: {os.path.basename(file_path)}", end="... ")
            df = pd.read_csv(file_path)
            print(f"loaded {len(df)} rows")
            if len(df) > 0:
                all_data.append(df)
            else:
                print(f"  Warning: File {os.path.basename(file_path)} contains no data")
        except Exception as e:
            print(f"Error reading file {os.path.basename(file_path)}: {str(e)}")
    # Check if any data was loaded
    if not all_data:
        print("No data was successfully loaded from any CSV file.")
        return None
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_data)} total typing data records")
    # Filter out intro-trial-1 data
    filtered_data = combined_data[combined_data['trialId'] != 'intro-trial-1'].copy()
    print(f"After filtering: {len(filtered_data)} records")
    # Process the data to calculate typing time
    processed_data = process_typing_data(filtered_data)

    # Analysis for 1-key and 2-key statistics
    left_home_key_stats, left_home_bigram_stats = analyze_key_stats(
        processed_data, keys_to_include=LEFT_HOME_KEYS
    )
    
    # Convert to DataFrames for analysis
    left_home_key_df = pd.DataFrame([stats for stats in left_home_key_stats.values()])
    left_home_bigram_df = pd.DataFrame([stats for stats in left_home_bigram_stats.values()])
    
    # Analyze mirror pairs using the new key stats
    mirror_pair_df = analyze_mirror_pairs(left_home_key_stats, letter_frequencies)
    
    # Create visualizations
    visualize_mirror_pairs(mirror_pair_df)
    
    # Create 1-key visualizations
    visualize_left_home_key_error_rates(left_home_key_df)
    visualize_left_home_key_typing_times(left_home_key_df)
    visualize_left_home_key_error_vs_time(left_home_key_df)
    
    # Create 2-key (bigram) visualizations
    visualize_left_home_bigram_counts(left_home_bigram_df)
    visualize_left_home_bigram_error_rates(left_home_bigram_df)
    visualize_left_home_bigram_typing_times(left_home_bigram_df)
    visualize_left_home_bigram_error_vs_time(left_home_bigram_df)
    
    # Create comparison visualizations between 1-key and 2-key statistics
    visualize_key_vs_bigram_times(left_home_key_df, left_home_bigram_df)
    visualize_key_vs_bigram_errors(left_home_key_df, left_home_bigram_df)
    
    # Verify data completeness
    verify_data_completeness(left_home_key_df, left_home_bigram_df)
    
    # Save results to CSV files
    output_dir = 'output/'
    os.makedirs(output_dir, exist_ok=True)
    left_home_key_df.to_csv(f"{output_dir}left_home_key_analysis.csv", index=False)
    left_home_bigram_df.to_csv(f"{output_dir}left_home_bigram_analysis.csv", index=False)
    mirror_pair_df.to_csv(f"{output_dir}mirror_pair_analysis.csv", index=False)
    print(f"\nResults saved to {output_dir}")
    
    return {
        'left_home_key_stats': left_home_key_stats,
        'left_home_bigram_stats': left_home_bigram_stats,
        'mirror_pair_df': mirror_pair_df
    }

if __name__ == "__main__":
    results = main()