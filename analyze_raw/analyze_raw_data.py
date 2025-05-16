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
    ('q', 'p'), ('w', 'o'), ('e', 'i'), ('r', 'u'), ('t', 'y'),
    ('a', ';'), ('s', 'l'), ('d', 'k'), ('f', 'j'), ('g', 'h'),
    ('z', '/'), ('x', '.'), ('c', ','), ('v', 'm')
]

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
    

def analyze_letters(data, keys_to_include=None, include_mirror_pairs=False):
    """
    Analyze statistics for individual letters
    
    Parameters:
    data (DataFrame): Processed typing data
    keys_to_include (list): List of keys to include in analysis (if None, include all)
    include_mirror_pairs (bool): Whether to include all mirror pair keys
    """
    letter_stats = {}
    
    # If include_mirror_pairs is True, add all mirror pair keys to keys_to_include
    if include_mirror_pairs and keys_to_include:
        mirror_keys = set()
        for left, right in MIRROR_PAIRS:
            mirror_keys.add(left)
            mirror_keys.add(right)
        keys_to_include = list(set(keys_to_include) | mirror_keys)
    
    # Analyze each letter
    for _, row in data.iterrows():
        letter = row['expectedKey']
        typed_letter = row['typedKey']
        
        # Skip if not a string or not a single character or if it's a space
        if not isinstance(letter, str) or len(letter) != 1 or letter == ' ':
            continue
            
        # Skip if not in keys_to_include (if specified)
        if keys_to_include and letter not in keys_to_include:
            continue
            
        # Skip if errors are related to spaces (either typing a space or not typing one)
        if not row['isCorrect'] and (typed_letter == ' ' or letter == ' '):
            continue
            
        # Initialize stats for this letter if not exists
        if letter not in letter_stats:
            letter_stats[letter] = {
                'letter': letter,
                'totalCount': 0,
                'errorCount': 0,
                'totalTime': 0,
                'timeSamples': [],
                'error_types': defaultdict(int)  # Track mistyped letters
            }
        
        # Update stats
        letter_stats[letter]['totalCount'] += 1
        
        if not row['isCorrect']:
            letter_stats[letter]['errorCount'] += 1
            # Track what was actually typed when error occurred
            letter_stats[letter]['error_types'][typed_letter] += 1
        
        # Only count time if it's reasonable (between 50ms and 2000ms)
        if 50 <= row['typingTime'] <= 2000:
            letter_stats[letter]['totalTime'] += row['typingTime']
            letter_stats[letter]['timeSamples'].append(row['typingTime'])
    
    # Calculate error rates and average/median times
    for stats in letter_stats.values():
        stats['errorRate'] = stats['errorCount'] / stats['totalCount'] if stats['totalCount'] > 0 else 0
        
        # Calculate average time
        stats['avgTime'] = stats['totalTime'] / len(stats['timeSamples']) if len(stats['timeSamples']) > 0 else 0
        
        # Calculate median time and MAD directly from samples
        if len(stats['timeSamples']) > 0:
            stats['medianTime'] = np.median(stats['timeSamples'])
            # Calculate Median Absolute Deviation (MAD)
            deviations = np.abs(np.array(stats['timeSamples']) - stats['medianTime'])
            stats['timeMAD'] = np.median(deviations)
        else:
            stats['medianTime'] = 0
            stats['timeMAD'] = 0
            
        # Calculate median error rate via bootstrap resampling (since it's binary data)
        if stats['totalCount'] > 10:  # Only if we have enough samples
            bootstrap_errors = []
            n_bootstrap = 1000
            for _ in range(n_bootstrap):
                sample = np.random.choice([0, 1], size=stats['totalCount'], 
                                        p=[1-stats['errorRate'], stats['errorRate']])
                bootstrap_errors.append(np.mean(sample))
            stats['errorMAD'] = np.median(np.abs(np.array(bootstrap_errors) - stats['errorRate']))
        else:
            stats['errorMAD'] = 0
            
        # Find most common error type
        if stats['error_types']:
            most_common_error = max(stats['error_types'].items(), key=lambda x: x[1])
            stats['most_common_mistype'] = most_common_error[0]
            stats['most_common_mistype_count'] = most_common_error[1]
        else:
            stats['most_common_mistype'] = ""
            stats['most_common_mistype_count'] = 0
    
    return letter_stats

def analyze_bigrams(data, keys_to_include=None):
    """
    Identify and analyze bigrams in the typing data
    A bigram is defined as two consecutive character keypresses
    
    Parameters:
    data (DataFrame): Processed typing data
    keys_to_include (list): List of keys to include in analysis (if None, include all)
    """
    bigram_stats = {}
    min_time_threshold = 50    # Minimum reasonable typing time in ms
    max_time_threshold = 2000  # Maximum reasonable typing time in ms
    
    # Group by user_id and trialId
    grouped = data.groupby(['user_id', 'trialId'])
    
    # Process each trial
    for (user_id, trial_id), trial_data in grouped:
        # Sort by timestamp to ensure correct sequence
        sorted_trial = trial_data.sort_values('keydownTime').reset_index(drop=True)
        
        # Find bigrams
        for i in range(len(sorted_trial) - 1):
            current = sorted_trial.iloc[i]
            next_key = sorted_trial.iloc[i + 1]
            
            # Skip if either key is a space or not a valid character
            if (not isinstance(current['expectedKey'], str) or 
                not isinstance(next_key['expectedKey'], str) or
                len(current['expectedKey']) != 1 or 
                len(next_key['expectedKey']) != 1 or
                current['expectedKey'] == ' ' or 
                next_key['expectedKey'] == ' '):
                continue
            
            # Skip if either key is not in keys_to_include (if specified)
            if keys_to_include and (current['expectedKey'] not in keys_to_include or 
                                   next_key['expectedKey'] not in keys_to_include):
                continue
                
            # Skip if errors are related to spaces
            # This excludes cases where spaces are incorrectly typed
            if ((current['isCorrect'] == False and current['typedKey'] == ' ') or
                (next_key['isCorrect'] == False and next_key['typedKey'] == ' ')):
                continue
                
            # Also skip cases where a non-space was typed as space
            if ((current['isCorrect'] == False and current['expectedKey'] != ' ' and current['typedKey'] == ' ') or
                (next_key['isCorrect'] == False and next_key['expectedKey'] != ' ' and next_key['typedKey'] == ' ')):
                continue
            
            # Form bigram key
            bigram = current['expectedKey'] + next_key['expectedKey']
            
            # Determine if it's a same-key bigram
            is_same_key = current['expectedKey'] == next_key['expectedKey']
            
            # Check if this is a flanked bigram (preceded and followed by spaces)
            # We need to check i-1 for previous and i+2 for next character after bigram
            is_flanked = False
            if i > 0 and i+2 < len(sorted_trial):
                prev_char = sorted_trial.iloc[i-1]
                next_char = sorted_trial.iloc[i+2]
                if (isinstance(prev_char['expectedKey'], str) and 
                    isinstance(next_char['expectedKey'], str) and
                    prev_char['expectedKey'] == ' ' and 
                    next_char['expectedKey'] == ' '):
                    is_flanked = True
            
            # Initialize stats for this bigram if not exists
            if bigram not in bigram_stats:
                bigram_stats[bigram] = {
                    'bigram': bigram,
                    'totalCount': 0,
                    'flankedCount': 0, # Count of bigrams flanked by spaces
                    
                    # Separate tracking for first and second key errors
                    'firstKeyErrorCount': 0,
                    'secondKeyErrorCount': 0,
                    
                    # Track actual mistypes for first and second keys
                    'firstKeyErrorTypes': defaultdict(int),
                    'secondKeyErrorTypes': defaultdict(int),
                    
                    # Separate tracking for first and second key times
                    'firstKeyTotalTime': 0,
                    'secondKeyTotalTime': 0,
                    'firstKeyTimeSamples': [],
                    'secondKeyTimeSamples': [],
                    
                    # Flanked bigram timing data
                    'flankedFirstKeyTotalTime': 0,
                    'flankedSecondKeyTotalTime': 0,
                    'flankedFirstKeyTimeSamples': [],
                    'flankedSecondKeyTimeSamples': [],
                    
                    'is_same_key': is_same_key
                }
            
            # Update stats
            bigram_stats[bigram]['totalCount'] += 1
            
            # Update flanked count if applicable
            if is_flanked:
                bigram_stats[bigram]['flankedCount'] += 1
            
            # Count errors for first and second key separately, excluding space-related errors
            if not current['isCorrect'] and current['typedKey'] != ' ':
                bigram_stats[bigram]['firstKeyErrorCount'] += 1
                # Track what was actually typed for first key error
                bigram_stats[bigram]['firstKeyErrorTypes'][current['typedKey']] += 1
            
            if not next_key['isCorrect'] and next_key['typedKey'] != ' ':
                bigram_stats[bigram]['secondKeyErrorCount'] += 1
                # Track what was actually typed for second key error
                bigram_stats[bigram]['secondKeyErrorTypes'][next_key['typedKey']] += 1
            
            # Track time for first key (if available)
            if i > 0 and 'typingTime' in current:
                if min_time_threshold <= current['typingTime'] <= max_time_threshold:
                    bigram_stats[bigram]['firstKeyTotalTime'] += current['typingTime']
                    bigram_stats[bigram]['firstKeyTimeSamples'].append(current['typingTime'])
                    
                    # Also add to flanked samples if applicable
                    if is_flanked:
                        bigram_stats[bigram]['flankedFirstKeyTotalTime'] += current['typingTime']
                        bigram_stats[bigram]['flankedFirstKeyTimeSamples'].append(current['typingTime'])
            
            # Track time for second key
            if min_time_threshold <= next_key['typingTime'] <= max_time_threshold:
                bigram_stats[bigram]['secondKeyTotalTime'] += next_key['typingTime']
                bigram_stats[bigram]['secondKeyTimeSamples'].append(next_key['typingTime'])
                
                # Also add to flanked samples if applicable
                if is_flanked:
                    bigram_stats[bigram]['flankedSecondKeyTotalTime'] += next_key['typingTime']
                    bigram_stats[bigram]['flankedSecondKeyTimeSamples'].append(next_key['typingTime'])
    
    # Calculate error rates and average times
    for stats in bigram_stats.values():
        # Error rates for first and second key
        stats['firstKeyErrorRate'] = stats['firstKeyErrorCount'] / stats['totalCount'] if stats['totalCount'] > 0 else 0
        stats['secondKeyErrorRate'] = stats['secondKeyErrorCount'] / stats['totalCount'] if stats['totalCount'] > 0 else 0
        
        # Average time for first key
        stats['firstKeyAvgTime'] = (stats['firstKeyTotalTime'] / len(stats['firstKeyTimeSamples']) 
                                   if len(stats['firstKeyTimeSamples']) > 0 else 0)
        
        # Average time for second key
        stats['secondKeyAvgTime'] = (stats['secondKeyTotalTime'] / len(stats['secondKeyTimeSamples']) 
                                    if len(stats['secondKeyTimeSamples']) > 0 else 0)
        
        # Calculate flanked bigram metrics (if any)
        if stats['flankedCount'] > 0:
            stats['flankedFirstKeyErrorRate'] = (stats['firstKeyErrorCount'] / stats['flankedCount'] 
                                               if stats['flankedCount'] > 0 else 0)
            stats['flankedSecondKeyErrorRate'] = (stats['secondKeyErrorCount'] / stats['flankedCount']
                                                if stats['flankedCount'] > 0 else 0)
            
            stats['flankedFirstKeyAvgTime'] = (stats['flankedFirstKeyTotalTime'] / len(stats['flankedFirstKeyTimeSamples'])
                                             if len(stats['flankedFirstKeyTimeSamples']) > 0 else 0)
            stats['flankedSecondKeyAvgTime'] = (stats['flankedSecondKeyTotalTime'] / len(stats['flankedSecondKeyTimeSamples'])
                                              if len(stats['flankedSecondKeyTimeSamples']) > 0 else 0)
        else:
            stats['flankedFirstKeyErrorRate'] = 0
            stats['flankedSecondKeyErrorRate'] = 0
            stats['flankedFirstKeyAvgTime'] = 0
            stats['flankedSecondKeyAvgTime'] = 0
        
        # Median times calculated directly from samples
        if len(stats['firstKeyTimeSamples']) > 0:
            stats['firstKeyMedianTime'] = np.median(stats['firstKeyTimeSamples'])
            # Calculate Median Absolute Deviation (MAD)
            deviations = np.abs(np.array(stats['firstKeyTimeSamples']) - stats['firstKeyMedianTime'])
            stats['firstKeyTimeMAD'] = np.median(deviations)
        else:
            stats['firstKeyMedianTime'] = 0
            stats['firstKeyTimeMAD'] = 0
            
        if len(stats['secondKeyTimeSamples']) > 0:
            stats['secondKeyMedianTime'] = np.median(stats['secondKeyTimeSamples'])
            # Calculate Median Absolute Deviation (MAD)
            deviations = np.abs(np.array(stats['secondKeyTimeSamples']) - stats['secondKeyMedianTime'])
            stats['secondKeyTimeMAD'] = np.median(deviations)
        else:
            stats['secondKeyMedianTime'] = 0
            stats['secondKeyTimeMAD'] = 0
            
        # Calculate error MAD using bootstrap for first key
        if stats['totalCount'] > 10:
            bootstrap_errors = []
            n_bootstrap = 1000
            for _ in range(n_bootstrap):
                sample = np.random.choice([0, 1], size=stats['totalCount'],
                                          p=[1-stats['firstKeyErrorRate'], stats['firstKeyErrorRate']])
                bootstrap_errors.append(np.mean(sample))
            stats['firstKeyErrorMAD'] = np.median(np.abs(np.array(bootstrap_errors) - stats['firstKeyErrorRate']))
        else:
            stats['firstKeyErrorMAD'] = 0
            
        # Calculate error MAD using bootstrap for second key
        if stats['totalCount'] > 10:
            bootstrap_errors = []
            n_bootstrap = 1000
            for _ in range(n_bootstrap):
                sample = np.random.choice([0, 1], size=stats['totalCount'],
                                          p=[1-stats['secondKeyErrorRate'], stats['secondKeyErrorRate']])
                bootstrap_errors.append(np.mean(sample))
            stats['secondKeyErrorMAD'] = np.median(np.abs(np.array(bootstrap_errors) - stats['secondKeyErrorRate']))
        else:
            stats['secondKeyErrorMAD'] = 0
            
        # Calculate median times for flanked bigrams
        if len(stats['flankedFirstKeyTimeSamples']) > 0:
            stats['flankedFirstKeyMedianTime'] = np.median(stats['flankedFirstKeyTimeSamples'])
        else:
            stats['flankedFirstKeyMedianTime'] = 0
            
        if len(stats['flankedSecondKeyTimeSamples']) > 0:
            stats['flankedSecondKeyMedianTime'] = np.median(stats['flankedSecondKeyTimeSamples'])
        else:
            stats['flankedSecondKeyMedianTime'] = 0
            
        # Find most common error types
        if stats['firstKeyErrorTypes']:
            most_common_first = max(stats['firstKeyErrorTypes'].items(), key=lambda x: x[1])
            stats['most_common_first_mistype'] = most_common_first[0]
            stats['most_common_first_mistype_count'] = most_common_first[1]
        else:
            stats['most_common_first_mistype'] = ""
            stats['most_common_first_mistype_count'] = 0
            
        if stats['secondKeyErrorTypes']:
            most_common_second = max(stats['secondKeyErrorTypes'].items(), key=lambda x: x[1])
            stats['most_common_second_mistype'] = most_common_second[0]
            stats['most_common_second_mistype_count'] = most_common_second[1]
        else:
            stats['most_common_second_mistype'] = ""
            stats['most_common_second_mistype_count'] = 0
    
    return bigram_stats

def analyze_with_bigram_frequencies(same_key_df, different_key_df, bigram_frequencies=None):
    """
    Analyze bigrams using bigram frequency data
    
    Parameters:
    same_key_df (DataFrame): DataFrame of same-key bigram statistics
    different_key_df (DataFrame): DataFrame of different-key bigram statistics
    bigram_frequencies (DataFrame): Bigram frequency data
    """
    if bigram_frequencies is None:
        print("No bigram frequency data provided")
        return
    
    print(f"Bigram frequency data columns: {list(bigram_frequencies.columns)}")
    
    # Check if the expected columns exist in the dataframe
    if 'item_pair' in bigram_frequencies.columns and 'score' in bigram_frequencies.columns:
        bigram_col = 'item_pair'
        freq_col = 'score'
    else:
        print("Could not find expected columns in bigram frequency data")
        return
    
    # Create bigram frequency dictionary
    bigram_freq_dict = {}
    for _, row in bigram_frequencies.iterrows():
        bigram = str(row[bigram_col]).lower()
        if len(bigram) == 2:  # Only use two-letter bigrams
            try:
                freq = float(row[freq_col])
                bigram_freq_dict[bigram] = freq
            except (ValueError, TypeError):
                pass
    
    print(f"Created bigram frequency dictionary with {len(bigram_freq_dict)} entries")
    print(f"Sample bigram frequencies: {dict(list(bigram_freq_dict.items())[:5])}")
    
    # Add bigram frequency to dataframes
    for df_name, df in [('same_key', same_key_df), ('different_key', different_key_df)]:
        # Add bigram frequency column
        df['bigram_freq'] = df['bigram'].str.lower().map(lambda x: bigram_freq_dict.get(x, 0) if isinstance(x, str) else 0)
        df['log_bigram_freq'] = np.log10(df['bigram_freq'] + 1)  # Add 1 to handle zeros
        
        # Calculate correlations
        for metric in ['firstKeyErrorRate', 'secondKeyErrorRate', 'firstKeyMedianTime', 'secondKeyMedianTime']:
            if metric in df.columns:
                corr = df[[metric, 'log_bigram_freq']].corr().iloc[0, 1]
                print(f"Correlation between {metric} and log bigram frequency for {df_name} bigrams: {corr:.3f}")
        
        # Perform regression to adjust for bigram frequency
        adjust_for_bigram_frequency(df, 'log_bigram_freq')

    # Create visualizations comparing bigram frequency effects
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot bigram frequency distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(same_key_df['log_bigram_freq'], alpha=0.5, label='Same-Key', kde=True)
    sns.histplot(different_key_df['log_bigram_freq'], alpha=0.5, label='Different-Key', kde=True)
    plt.title('Bigram Frequency Distribution (Log Scale)')
    plt.xlabel('Log Bigram Frequency')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}bigram_frequency_distribution.png')
    plt.close()
    
    # Scatter plot of error rates vs. bigram frequency
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(same_key_df['log_bigram_freq'], same_key_df['firstKeyErrorRate'], 
                alpha=0.7, label='Same-Key', s=50)
    plt.scatter(different_key_df['log_bigram_freq'], different_key_df['firstKeyErrorRate'], 
                alpha=0.7, label='Different-Key', s=50)
    plt.title('First Key Error Rate vs. Bigram Frequency')
    plt.xlabel('Log Bigram Frequency')
    plt.ylabel('Error Rate')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(same_key_df['log_bigram_freq'], same_key_df['secondKeyErrorRate'], 
                alpha=0.7, label='Same-Key', s=50)
    plt.scatter(different_key_df['log_bigram_freq'], different_key_df['secondKeyErrorRate'], 
                alpha=0.7, label='Different-Key', s=50)
    plt.title('Second Key Error Rate vs. Bigram Frequency')
    plt.xlabel('Log Bigram Frequency')
    plt.ylabel('Error Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}error_rate_vs_bigram_frequency.png')
    plt.close()

def adjust_for_bigram_frequency(df, freq_col):
    """
    Adjust metrics by regressing out bigram frequency
    
    Parameters:
    df (DataFrame): DataFrame to adjust
    freq_col (str): Column name for the frequency measure
    """
    # Metrics to adjust
    metrics = [
        'firstKeyErrorRate', 'secondKeyErrorRate', 
        'firstKeyMedianTime', 'secondKeyMedianTime'
    ]
    
    # Perform regression adjustment for each metric
    for metric in metrics:
        if metric in df.columns and len(df) > 5:
            # Create the model: metric ~ frequency
            X = sm.add_constant(df[freq_col])
            y = df[metric]
            
            try:
                # Fit the model
                model = sm.OLS(y, X).fit()
                
                # Calculate residuals (the frequency-adjusted values)
                df[f'{metric}_bigram_freq_adjusted'] = model.resid
                
                # Add the mean back to make the adjusted values more interpretable
                df[f'{metric}_bigram_freq_adjusted'] = df[f'{metric}_bigram_freq_adjusted'] + np.mean(df[metric])
                
                print(f"Adjusted {metric} for bigram frequency (R² = {model.rsquared:.3f})")
            except Exception as e:
                print(f"Error adjusting {metric} for bigram frequency: {str(e)}")

def print_left_home_stats(letter_df, min_count=10):
    """
    Print statistics for left home block keys
    
    Parameters:
    letter_df (DataFrame): DataFrame of letter statistics
    min_count (int): Minimum count threshold for statistics
    """
    # Filter for left home keys and sufficient data
    left_home_df = letter_df[letter_df['letter'].isin(LEFT_HOME_KEYS) & (letter_df['totalCount'] > min_count)]
    
    if left_home_df.empty:
        print("No data available for left home block keys with sufficient samples")
        return
    
    print("\n" + "="*50)
    print("LEFT HOME BLOCK KEY ANALYSIS")
    print("="*50)
    
    # Sort by error rate (descending) and print
    print("\nError rates for left home block keys (sorted by error rate):")
    print(f"{'Key':<5} {'Count':<10} {'Error Rate':^15} {'Avg Time (ms)':^15} {'Most Common Mistype':<20}")
    print(f"{'-'*5} {'-'*10} {'-'*15} {'-'*15} {'-'*20}")
    
    for _, row in left_home_df.sort_values('errorRate', ascending=False).iterrows():
        print(f"{row['letter']:<5} {row['totalCount']:<10} {row['errorRate']*100:^15.2f}% {row['avgTime']:^15.1f} {row['most_common_mistype']:<20}")
    
    # Sort by average time (descending) and print
    print("\nTyping times for left home block keys (sorted by average time):")
    print(f"{'Key':<5} {'Count':<10} {'Error Rate':^15} {'Avg Time (ms)':^15}")
    print(f"{'-'*5} {'-'*10} {'-'*15} {'-'*15}")
    
    for _, row in left_home_df.sort_values('avgTime', ascending=False).iterrows():
        print(f"{row['letter']:<5} {row['totalCount']:<10} {row['errorRate']*100:^15.2f}% {row['avgTime']:^15.1f}")


def analyze_mirror_pairs(letter_stats, letter_freq=None):
    """
    Analyze mirror image key pairs
    
    Parameters:
    letter_stats (dict): Dictionary of letter statistics
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
    
    for left, right in MIRROR_PAIRS:
        if left in letter_stats and right in letter_stats:
            left_stats = letter_stats[left]
            right_stats = letter_stats[right]
            
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
        
def visualize_left_home_results(letter_df):
    """
    Create visualizations for left home block keys
    
    Parameters:
    letter_df (DataFrame): DataFrame containing letter statistics
    """
    # Filter for left home keys and sufficient data
    min_count = 10
    left_home_df = letter_df[letter_df['letter'].isin(LEFT_HOME_KEYS) & (letter_df['totalCount'] > min_count)]
    
    if left_home_df.empty:
        print("No data available for left home block keys visualization")
        return
    
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by letter for consistent visualization
    left_home_df = left_home_df.sort_values('letter')
    
    # 1. Error rates for left home block keys with MAD as error bars
    plt.figure(figsize=(12, 6))
    bars = plt.bar(left_home_df['letter'], left_home_df['errorRate'])
    
    # Add error bars using errorMAD
    plt.errorbar(x=left_home_df['letter'], y=left_home_df['errorRate'], 
                 yerr=left_home_df['errorMAD'], fmt='none', color='black', capsize=5)
    
    plt.title('Error Rates for Left Home Block Keys (with MAD)')
    plt.xlabel('Key')
    plt.ylabel('Error Rate')
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2, f"{height*100:.1f}%", 
                ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}left_home_error_rates_with_mad.png')
    plt.close()
    
    # 2. Median typing times for left home block keys with MAD as error bars
    plt.figure(figsize=(12, 6))
    bars = plt.bar(left_home_df['letter'], left_home_df['medianTime'])
    
    # Add error bars using timeMAD
    plt.errorbar(x=left_home_df['letter'], y=left_home_df['medianTime'], 
                 yerr=left_home_df['timeMAD'], fmt='none', color='black', capsize=5)
    
    plt.title('Median Typing Times for Left Home Block Keys (with MAD)')
    plt.xlabel('Key')
    plt.ylabel('Median Time (ms)')
    
    # Add time labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height/2, f"{height:.0f}ms", 
                ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}left_home_typing_times_with_mad.png')
    plt.close()
    
    # 3. Error rate vs typing time
    plt.figure(figsize=(10, 8))
    plt.errorbar(left_home_df['medianTime'], left_home_df['errorRate'], 
                xerr=left_home_df['timeMAD'], yerr=left_home_df['errorMAD'],
                fmt='o', alpha=0.7, ecolor='lightgray', capsize=3)
    
    # Add key labels
    for _, row in left_home_df.iterrows():
        plt.text(row['medianTime'], row['errorRate'], row['letter'], 
                fontsize=12, ha='center', va='center')
    
    plt.title('Error Rate vs. Typing Time for Left Home Block Keys')
    plt.xlabel('Median Time (ms)')
    plt.ylabel('Error Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}left_home_error_vs_time_with_mad.png')
    plt.close()
    
    # 4. Generate frequency-adjusted plots if frequency data is available
    if 'errorRate_freq_adjusted' in left_home_df.columns and 'medianTime_freq_adjusted' in left_home_df.columns:
        # Frequency-adjusted error rates
        plt.figure(figsize=(12, 6))
        
        # Create data for grouped bar plot
        error_data = []
        for _, row in left_home_df.iterrows():
            error_data.append({'key': row['letter'], 'type': 'Raw', 'error_rate': row['errorRate']})
            error_data.append({'key': row['letter'], 'type': 'Freq-Adjusted', 'error_rate': row['errorRate_freq_adjusted']})
        
        error_df = pd.DataFrame(error_data)
        
        # Create grouped bar plot
        ax = sns.barplot(x='key', y='error_rate', hue='type', data=error_df)
        plt.title('Raw vs. Frequency-Adjusted Error Rates for Left Home Block Keys')
        plt.xlabel('Key')
        plt.ylabel('Error Rate')
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.002, f"{height*100:.1f}%",
                    ha='center')
        
        plt.legend(title='Adjustment')
        plt.tight_layout()
        plt.savefig(f'{output_dir}left_home_error_rates_freq_adjusted.png')
        plt.close()
        
        # Similar plot for median time
        plt.figure(figsize=(12, 6))
        
        # Create data for grouped bar plot
        time_data = []
        for _, row in left_home_df.iterrows():
            time_data.append({'key': row['letter'], 'type': 'Raw', 'time': row['medianTime']})
            time_data.append({'key': row['letter'], 'type': 'Freq-Adjusted', 'time': row['medianTime_freq_adjusted']})
        
        time_df = pd.DataFrame(time_data)
        
        # Create grouped bar plot
        ax = sns.barplot(x='key', y='time', hue='type', data=time_df)
        plt.title('Raw vs. Frequency-Adjusted Median Times for Left Home Block Keys')
        plt.xlabel('Key')
        plt.ylabel('Median Time (ms)')
        
        # Add time labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 5, f"{height:.0f}ms",
                    ha='center')
        
        plt.legend(title='Adjustment')
        plt.tight_layout()
        plt.savefig(f'{output_dir}left_home_typing_times_freq_adjusted.png')
        plt.close()

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
    
    # 1. Error rate comparison with MAD error bars
    plt.figure(figsize=(15, 8))
    
    # Prepare data for grouped bar plot
    error_data = []
    for _, row in mirror_df.iterrows():
        error_data.append({'pair': f"{row['left_key']}-{row['right_key']}", 
                           'position': 'Left', 
                           'error_rate': row['left_error_rate'],
                           'error_mad': row['left_error_mad']})
        error_data.append({'pair': f"{row['left_key']}-{row['right_key']}", 
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
    
    plt.title('Error Rate Comparison for Mirror Pairs (with MAD)')
    plt.xlabel('Mirror Pair')
    plt.ylabel('Error Rate')
    plt.xticks(rotation=45)
    plt.legend(title='Key Position')
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_error_rates_with_mad.png')
    plt.close()
    
    # 2. Typing time comparison with MAD error bars
    plt.figure(figsize=(15, 8))
    
    # Prepare data for grouped bar plot
    time_data = []
    for _, row in mirror_df.iterrows():
        time_data.append({'pair': f"{row['left_key']}-{row['right_key']}", 
                          'position': 'Left', 
                          'median_time': row['left_median_time'],
                          'time_mad': row['left_time_mad']})
        time_data.append({'pair': f"{row['left_key']}-{row['right_key']}", 
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
    
    plt.title('Typing Time Comparison for Mirror Pairs (with MAD)')
    plt.xlabel('Mirror Pair')
    plt.ylabel('Median Time (ms)')
    plt.xticks(rotation=45)
    plt.legend(title='Key Position')
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_typing_times_with_mad.png')
    plt.close()
    
    # 3. Error rate differences (left - right)
    plt.figure(figsize=(15, 8))
    
    # Sort by absolute difference for better visualization
    sorted_mirror = mirror_df.copy()
    sorted_mirror['abs_error_diff'] = sorted_mirror['error_diff'].abs()
    sorted_mirror = sorted_mirror.sort_values('abs_error_diff', ascending=False)
    
    # Calculate error bars for differences
    error_bars = np.sqrt(sorted_mirror['left_error_mad']**2 + sorted_mirror['right_error_mad']**2)
    
    # Create bar plot with colored bars and error bars
    bars = plt.bar(
        [f"{row['left_key']}-{row['right_key']}" for _, row in sorted_mirror.iterrows()],
        sorted_mirror['error_diff'],
        color=[('red' if diff > 0 else 'blue') for diff in sorted_mirror['error_diff']]
    )
    
    # Add error bars
    plt.errorbar(
        x=range(len(sorted_mirror)), 
        y=sorted_mirror['error_diff'],
        yerr=error_bars,
        fmt='none', 
        color='black', 
        capsize=3
    )
    
    plt.title('Error Rate Differences for Mirror Pairs (Left - Right)')
    plt.xlabel('Mirror Pair')
    plt.ylabel('Error Rate Difference')
    plt.xticks(range(len(sorted_mirror)), [f"{row['left_key']}-{row['right_key']}" for _, row in sorted_mirror.iterrows()], rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add percentage labels
    for bar, error_bar in zip(bars, error_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height/2 if height > 0 else height*1.5, 
                f"{height*100:+.1f}%",
                ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_error_differences_with_mad.png')
    plt.close()
    
    # 4. Typing time differences (left - right) with MAD error bars
    plt.figure(figsize=(15, 8))
    
    # Sort by absolute time difference
    sorted_mirror = mirror_df.copy()
    sorted_mirror['abs_time_diff'] = sorted_mirror['time_diff'].abs()
    sorted_mirror = sorted_mirror.sort_values('abs_time_diff', ascending=False)
    
    # Calculate error bars for differences
    time_error_bars = np.sqrt(sorted_mirror['left_time_mad']**2 + sorted_mirror['right_time_mad']**2)
    
    # Create bar plot with colored bars and error bars
    bars = plt.bar(
        [f"{row['left_key']}-{row['right_key']}" for _, row in sorted_mirror.iterrows()],
        sorted_mirror['time_diff'],
        color=[('red' if diff > 0 else 'blue') for diff in sorted_mirror['time_diff']]
    )
    
    # Add error bars
    plt.errorbar(
        x=range(len(sorted_mirror)), 
        y=sorted_mirror['time_diff'],
        yerr=time_error_bars,
        fmt='none', 
        color='black', 
        capsize=3
    )
    
    plt.title('Typing Time Differences for Mirror Pairs (Left - Right)')
    plt.xlabel('Mirror Pair')
    plt.ylabel('Time Difference (ms)')
    plt.xticks(range(len(sorted_mirror)), [f"{row['left_key']}-{row['right_key']}" for _, row in sorted_mirror.iterrows()], rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add time labels
    for bar, error_bar in zip(bars, time_error_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height/2 if height > 0 else height*1.5, 
                f"{height:+.0f}ms",
                ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_time_differences_with_mad.png')
    plt.close()
    
    # 5. Scatter plot comparing left vs right error rates with MAD error bars
    plt.figure(figsize=(10, 10))
    
    # Create scatter plot with error bars
    plt.errorbar(
        mirror_df['left_error_rate'], mirror_df['right_error_rate'],
        xerr=mirror_df['left_error_mad'], yerr=mirror_df['right_error_mad'],
        fmt='o', alpha=0.7, ecolor='lightgray', capsize=3
    )
    
    # Add diagonal line representing equal error rates
    max_error = max(mirror_df['left_error_rate'].max(), mirror_df['right_error_rate'].max()) * 1.1
    plt.plot([0, max_error], [0, max_error], 'k--', alpha=0.5, label='Equal Error Rate')
    
    # Add point labels for each mirror pair
    for _, row in mirror_df.iterrows():
        plt.annotate(
            f"{row['left_key']}-{row['right_key']}", 
            (row['left_error_rate'], row['right_error_rate']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=10
        )
    
    # Add frequency-adjusted points if available
    if 'left_error_rate_freq_adjusted' in mirror_df.columns:
        plt.scatter(
            mirror_df['left_error_rate_freq_adjusted'], 
            mirror_df['right_error_rate_freq_adjusted'],
            marker='x', color='red', alpha=0.7, s=50,
            label='Frequency-Adjusted'
        )
        
        # Connect raw and adjusted points with lines
        for _, row in mirror_df.iterrows():
            plt.plot(
                [row['left_error_rate'], row['left_error_rate_freq_adjusted']],
                [row['right_error_rate'], row['right_error_rate_freq_adjusted']],
                'r-', alpha=0.3
            )
    
    plt.title('Left vs. Right Key Error Rates for Mirror Pairs')
    plt.xlabel('Left Key Error Rate')
    plt.ylabel('Right Key Error Rate')
    
    # Add region labels
    plt.text(max_error*0.25, max_error*0.75, 'Right Keys More Error-Prone', 
             ha='center', fontsize=12, alpha=0.7)
    plt.text(max_error*0.75, max_error*0.25, 'Left Keys More Error-Prone', 
             ha='center', fontsize=12, alpha=0.7)
    
    plt.xlim(0, max_error)
    plt.ylim(0, max_error)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_error_scatter_with_mad.png')
    plt.close()

    # 6. Scatter plot comparing left vs right typing times with MAD and frequency adjustment
    plt.figure(figsize=(10, 10))
    
    # Create scatter plot with error bars
    plt.errorbar(
        mirror_df['left_median_time'], mirror_df['right_median_time'],
        xerr=mirror_df['left_time_mad'], yerr=mirror_df['right_time_mad'],
        fmt='o', alpha=0.7, ecolor='lightgray', capsize=3,
        label='Raw'
    )
    
    # Add diagonal line representing equal typing times
    max_time = max(mirror_df['left_median_time'].max(), mirror_df['right_median_time'].max()) * 1.1
    plt.plot([0, max_time], [0, max_time], 'k--', alpha=0.5, label='Equal Typing Time')
    
    # Add point labels for each mirror pair
    for _, row in mirror_df.iterrows():
        plt.annotate(
            f"{row['left_key']}-{row['right_key']}", 
            (row['left_median_time'], row['right_median_time']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=10
        )
    
    # Add frequency-adjusted points if available
    if 'left_median_time_freq_adjusted' in mirror_df.columns:
        plt.scatter(
            mirror_df['left_median_time_freq_adjusted'], 
            mirror_df['right_median_time_freq_adjusted'],
            marker='x', color='red', alpha=0.7, s=50,
            label='Frequency-Adjusted'
        )
        
        # Connect raw and adjusted points with lines
        for _, row in mirror_df.iterrows():
            plt.plot(
                [row['left_median_time'], row['left_median_time_freq_adjusted']],
                [row['right_median_time'], row['right_median_time_freq_adjusted']],
                'r-', alpha=0.3
            )
    
    plt.title('Left vs. Right Key Typing Times for Mirror Pairs')
    plt.xlabel('Left Key Median Time (ms)')
    plt.ylabel('Right Key Median Time (ms)')
    
    # Add region labels
    plt.text(max_time*0.25, max_time*0.75, 'Right Keys Slower', 
             ha='center', fontsize=12, alpha=0.7)
    plt.text(max_time*0.75, max_time*0.25, 'Left Keys Slower', 
             ha='center', fontsize=12, alpha=0.7)
    
    plt.xlim(0, max_time)
    plt.ylim(0, max_time)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_time_scatter_with_mad.png')
    plt.close()
    
    # 7. If frequency-adjusted metrics are available, create separate plots
    if ('left_error_rate_freq_adjusted' in mirror_df.columns and 
        'right_error_rate_freq_adjusted' in mirror_df.columns):
        
        # Error rate differences - frequency adjusted
        plt.figure(figsize=(15, 8))
        
        # Sort by absolute difference for better visualization
        sorted_mirror = mirror_df.copy()
        sorted_mirror['abs_error_diff_freq_adjusted'] = sorted_mirror['error_diff_freq_adjusted'].abs()
        sorted_mirror = sorted_mirror.sort_values('abs_error_diff_freq_adjusted', ascending=False)
        
        # Create bar plot with colored bars
        bars = plt.bar(
            [f"{row['left_key']}-{row['right_key']}" for _, row in sorted_mirror.iterrows()],
            sorted_mirror['error_diff_freq_adjusted'],
            color=[('red' if diff > 0 else 'blue') for diff in sorted_mirror['error_diff_freq_adjusted']]
        )
        
        plt.title('Frequency-Adjusted Error Rate Differences for Mirror Pairs (Left - Right)')
        plt.xlabel('Mirror Pair')
        plt.ylabel('Error Rate Difference')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height/2 if height > 0 else height*1.5, 
                    f"{height*100:+.1f}%",
                    ha='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}mirror_pair_error_differences_freq_adjusted.png')
        plt.close()
        
        # Typing time differences - frequency adjusted
        if ('left_median_time_freq_adjusted' in mirror_df.columns and 
            'right_median_time_freq_adjusted' in mirror_df.columns):
            
            plt.figure(figsize=(15, 8))
            
            # Sort by absolute time difference
            sorted_mirror = mirror_df.copy()
            sorted_mirror['abs_time_diff_freq_adjusted'] = sorted_mirror['time_diff_freq_adjusted'].abs()
            sorted_mirror = sorted_mirror.sort_values('abs_time_diff_freq_adjusted', ascending=False)
            
            # Create bar plot with colored bars
            bars = plt.bar(
                [f"{row['left_key']}-{row['right_key']}" for _, row in sorted_mirror.iterrows()],
                sorted_mirror['time_diff_freq_adjusted'],
                color=[('red' if diff > 0 else 'blue') for diff in sorted_mirror['time_diff_freq_adjusted']]
            )
            
            plt.title('Frequency-Adjusted Typing Time Differences for Mirror Pairs (Left - Right)')
            plt.xlabel('Mirror Pair')
            plt.ylabel('Time Difference (ms)')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add time labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., 
                        height/2 if height > 0 else height*1.5, 
                        f"{height:+.0f}ms",
                        ha='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}mirror_pair_time_differences_freq_adjusted.png')
            plt.close()


def adjust_bigrams_for_frequency(same_key_df, different_key_df, letter_frequencies=None):
    """
    Adjust same-key and different-key bigram metrics by regressing out frequency effects
    
    Parameters:
    same_key_df (DataFrame): DataFrame of same-key bigram statistics
    different_key_df (DataFrame): DataFrame of different-key bigram statistics
    letter_frequencies (DataFrame): Letter frequency data
    
    Returns:
    tuple: (frequency_adjusted_same_key_df, frequency_adjusted_different_key_df)
    """
    if letter_frequencies is None:
        print("No letter frequency data provided for adjustment")
        return same_key_df, different_key_df
    
    # Create a copy of input DataFrames
    adjusted_same_key = same_key_df.copy()
    adjusted_different_key = different_key_df.copy()
    
    # Print the actual columns in the letter frequencies DataFrame
    print(f"Letter frequency data columns: {list(letter_frequencies.columns)}")
    
    # Create frequency dictionary based on available columns
    freq_dict = {}
    
    # Check if the expected columns exist in the dataframe
    if 'item' in letter_frequencies.columns and 'score' in letter_frequencies.columns:
        print("Using 'item' as letter column and 'score' as frequency column")
        letter_col = 'item'
        freq_col = 'score'
    else:
        # Try to identify the correct columns for letter and frequency
        letter_col = None
        freq_col = None
        
        # List of possible column names for letters
        letter_col_candidates = ['letter', 'char', 'character', 'item', 'key']
        
        # List of possible column names for frequencies
        freq_col_candidates = ['frequency', 'freq', 'count', 'score', 'probability']
        
        # Try to find a letter column
        for col in letter_col_candidates:
            if col in letter_frequencies.columns:
                letter_col = col
                print(f"Found letter column: '{letter_col}'")
                break
        
        # Try to find a frequency column
        for col in freq_col_candidates:
            if col in letter_frequencies.columns:
                freq_col = col
                print(f"Found frequency column: '{freq_col}'")
                break
        
        # If we couldn't find appropriate columns, try to infer them
        if letter_col is None or freq_col is None:
            print("Could not identify letter and frequency columns. Available columns:", 
                  list(letter_frequencies.columns))
            return same_key_df, different_key_df
    
    # Create frequency dictionary
    for _, row in letter_frequencies.iterrows():
        # Ensure letter is a string and has only one character
        letter_value = str(row[letter_col]).lower()
        if len(letter_value) == 1:  # Only use single characters
            # Convert frequency to float
            try:
                freq_value = float(row[freq_col])
                freq_dict[letter_value] = freq_value
            except (ValueError, TypeError):
                pass
    
    # Check if we have frequency data
    if not freq_dict:
        print("No valid frequency data could be extracted")
        return same_key_df, different_key_df
    
    print(f"Created frequency dictionary with {len(freq_dict)} letters")
    print(f"Sample frequencies: {dict(list(freq_dict.items())[:5])}")
    
    # Add frequency data to bigram DataFrames
    def add_frequency_data(df):
        # Extract first and second letters from bigram
        df['first_letter'] = df['bigram'].str[0]
        df['second_letter'] = df['bigram'].str[1]
        
        # Add frequency for each letter
        df['first_letter_freq'] = df['first_letter'].map(lambda x: freq_dict.get(x.lower(), 0) if isinstance(x, str) else 0)
        df['second_letter_freq'] = df['second_letter'].map(lambda x: freq_dict.get(x.lower(), 0) if isinstance(x, str) else 0)
        
        # Log-transform frequency (common in psycholinguistic research)
        df['first_log_freq'] = np.log10(df['first_letter_freq'] + 1)  # Add 1 to handle zeros
        df['second_log_freq'] = np.log10(df['second_letter_freq'] + 1)
        
        return df
    
    adjusted_same_key = add_frequency_data(adjusted_same_key)
    adjusted_different_key = add_frequency_data(adjusted_different_key)
    
    # Metrics to adjust
    metrics = [
        'firstKeyErrorRate', 'secondKeyErrorRate', 
        'firstKeyMedianTime', 'secondKeyMedianTime'
    ]
    
    # Perform regression adjustment for each DataFrame and metric
    for df_name, df in [('same_key', adjusted_same_key), ('different_key', adjusted_different_key)]:
        for metric in metrics:
            if metric in df.columns:
                # Determine which frequency to use for regression
                freq_col = 'first_log_freq' if 'first' in metric else 'second_log_freq'
                
                # Skip if we don't have enough data points
                if len(df) < 5:
                    continue
                
                # Get non-null data
                valid_data = df[[metric, freq_col]].dropna()
                
                # Skip if we don't have enough valid data
                if len(valid_data) < 5:
                    continue
                
                try:
                    # Create the model: metric ~ log_frequency
                    X = sm.add_constant(valid_data[freq_col])
                    y = valid_data[metric]
                    
                    # Fit the model
                    model = sm.OLS(y, X).fit()
                    
                    # Get the model parameters
                    intercept = model.params[0]
                    slope = model.params[1]
                    
                    # Calculate the predicted values
                    df[f'{metric}_pred'] = intercept + slope * df[freq_col]
                    
                    # Calculate the frequency-adjusted values as residuals + mean
                    adjusted_values = df[metric] - df[f'{metric}_pred'] + df[metric].mean()
                    
                    # Store the adjusted values
                    df[f'{metric}_freq_adjusted'] = adjusted_values
                    
                    print(f"Adjusted {metric} for {df_name} bigrams (R² = {model.rsquared:.3f})")
                except Exception as e:
                    print(f"Error adjusting {metric} for {df_name} bigrams: {e}")
    
    return adjusted_same_key, adjusted_different_key

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

def compare_frequency_adjusted_bigrams(freq_adjusted_same_key, freq_adjusted_different_key, output_dir):
    """
    Compare frequency-adjusted metrics between same-key and different-key bigrams
    
    Parameters:
    freq_adjusted_same_key (DataFrame): Frequency-adjusted same-key bigram statistics
    freq_adjusted_different_key (DataFrame): Frequency-adjusted different-key bigram statistics
    output_dir (str): Directory to save output files
    """
    print("\n" + "="*50)
    print("FREQUENCY-ADJUSTED COMPARISON")
    print("="*50)
    
    # Metrics to compare
    adj_metrics = [
        ('First Key Error Rate', 'firstKeyErrorRate', 'firstKeyErrorRate_freq_adjusted'),
        ('Second Key Error Rate', 'secondKeyErrorRate', 'secondKeyErrorRate_freq_adjusted'),
        ('First Key Median Time', 'firstKeyMedianTime', 'firstKeyMedianTime_freq_adjusted'),
        ('Second Key Median Time', 'secondKeyMedianTime', 'secondKeyMedianTime_freq_adjusted')
    ]
    
    print("\nFrequency-Adjusted Summary Statistics:")
    print(f"{'Metric':<30} {'Same-Key':^15} {'Different-Key':^15} {'Difference':^15} {'Significance':^10}")
    print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
    
    # Compare adjusted metrics
    for label, raw_col, adj_col in adj_metrics:
        # Skip if column is not in either DataFrame
        if (adj_col not in freq_adjusted_same_key.columns or 
            adj_col not in freq_adjusted_different_key.columns):
            continue
            
        # Calculate raw statistics
        raw_same_mean = freq_adjusted_same_key[raw_col].mean()
        raw_diff_mean = freq_adjusted_different_key[raw_col].mean()
        raw_difference = raw_same_mean - raw_diff_mean
        
        # Calculate adjusted statistics
        same_key_stat = freq_adjusted_same_key[adj_col].mean()
        diff_key_stat = freq_adjusted_different_key[adj_col].mean()
        difference = same_key_stat - diff_key_stat
        
        # Perform t-test for statistical significance
        from scipy import stats
        _, p_value = stats.ttest_ind(
            freq_adjusted_same_key[adj_col].dropna(), 
            freq_adjusted_different_key[adj_col].dropna(),
            equal_var=False
        )
        
        # Format the significance indicator
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "n.s."
            
        # Format for display
        if 'Error' in label:
            raw_same_display = f"{raw_same_mean*100:.2f}%"
            raw_diff_display = f"{raw_diff_mean*100:.2f}%"
            raw_diff_display_value = f"{raw_difference*100:+.2f}%"
            
            same_key_display = f"{same_key_stat*100:.2f}%"
            diff_key_display = f"{diff_key_stat*100:.2f}%"
            diff_display = f"{difference*100:+.2f}%"
        else:
            raw_same_display = f"{raw_same_mean:.2f}"
            raw_diff_display = f"{raw_diff_mean:.2f}"
            raw_diff_display_value = f"{raw_difference:+.2f}"
            
            same_key_display = f"{same_key_stat:.2f}"
            diff_key_display = f"{diff_key_stat:.2f}"
            diff_display = f"{difference:+.2f}"
        
        # Print raw statistics
        print(f"{label + ' (Raw)':<30} {raw_same_display:^15} {raw_diff_display:^15} {raw_diff_display_value:^15}")
            
        # Print adjusted statistics
        print(f"{label + ' (Adjusted)':<30} {same_key_display:^15} {diff_key_display:^15} {diff_display:^15} {sig:^10}")
    
    # Create visualization comparing raw vs. frequency-adjusted differences
    visualize_frequency_adjusted_comparison(freq_adjusted_same_key, freq_adjusted_different_key, output_dir)

def visualize_frequency_adjusted_comparison(freq_adjusted_same_key, freq_adjusted_different_key, output_dir):
    """
    Create visualizations comparing raw vs. frequency-adjusted metrics
    
    Parameters:
    freq_adjusted_same_key (DataFrame): Frequency-adjusted same-key bigram statistics
    freq_adjusted_different_key (DataFrame): Frequency-adjusted different-key bigram statistics
    output_dir (str): Directory to save output files
    """
    # Create output directory for figures
    figures_dir = f'{output_dir}figures/'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Pairs of raw and adjusted metrics to compare
    metric_pairs = [
        ('firstKeyErrorRate', 'firstKeyErrorRate_freq_adjusted', 'First Key Error Rate'),
        ('secondKeyErrorRate', 'secondKeyErrorRate_freq_adjusted', 'Second Key Error Rate'),
        ('firstKeyMedianTime', 'firstKeyMedianTime_freq_adjusted', 'First Key Median Time'),
        ('secondKeyMedianTime', 'secondKeyMedianTime_freq_adjusted', 'Second Key Median Time')
    ]
    
    for raw_col, adj_col, title in metric_pairs:
        # Skip if columns don't exist
        if (raw_col not in freq_adjusted_same_key.columns or 
            adj_col not in freq_adjusted_same_key.columns or
            raw_col not in freq_adjusted_different_key.columns or
            adj_col not in freq_adjusted_different_key.columns):
            continue
        
        # Create comparison data
        comparison_data = []
        
        # Same-key raw
        raw_same_mean = freq_adjusted_same_key[raw_col].mean()
        raw_same_mad = freq_adjusted_same_key[f'{raw_col.replace("Rate", "MAD") if "Rate" in raw_col else raw_col.replace("Time", "TimeMAD")}'].mean() if f'{raw_col.replace("Rate", "MAD") if "Rate" in raw_col else raw_col.replace("Time", "TimeMAD")}' in freq_adjusted_same_key.columns else None
        
        comparison_data.append({
            'metric_type': 'Raw', 
            'bigram_type': 'Same-Key',
            'value': raw_same_mean,
            'mad': raw_same_mad
        })
        
        # Same-key adjusted
        adj_same_mean = freq_adjusted_same_key[adj_col].mean()
        # MAD for adjusted values - calculate from the distribution
        adj_same_mad = np.median(np.abs(freq_adjusted_same_key[adj_col] - adj_same_mean)) if not freq_adjusted_same_key[adj_col].isna().all() else None
        
        comparison_data.append({
            'metric_type': 'Frequency-Adjusted', 
            'bigram_type': 'Same-Key',
            'value': adj_same_mean,
            'mad': adj_same_mad
        })
        
        # Different-key raw
        raw_diff_mean = freq_adjusted_different_key[raw_col].mean()
        raw_diff_mad = freq_adjusted_different_key[f'{raw_col.replace("Rate", "MAD") if "Rate" in raw_col else raw_col.replace("Time", "TimeMAD")}'].mean() if f'{raw_col.replace("Rate", "MAD") if "Rate" in raw_col else raw_col.replace("Time", "TimeMAD")}' in freq_adjusted_different_key.columns else None
        
        comparison_data.append({
            'metric_type': 'Raw', 
            'bigram_type': 'Different-Key',
            'value': raw_diff_mean,
            'mad': raw_diff_mad
        })
        
        # Different-key adjusted
        adj_diff_mean = freq_adjusted_different_key[adj_col].mean()
        # MAD for adjusted values - calculate from the distribution
        adj_diff_mad = np.median(np.abs(freq_adjusted_different_key[adj_col] - adj_diff_mean)) if not freq_adjusted_different_key[adj_col].isna().all() else None
        
        comparison_data.append({
            'metric_type': 'Frequency-Adjusted', 
            'bigram_type': 'Different-Key',
            'value': adj_diff_mean,
            'mad': adj_diff_mad
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Create grouped bar plot
        ax = sns.barplot(x='bigram_type', y='value', hue='metric_type', data=comparison_df)
        
        # Add error bars manually - ensure index doesn't exceed list length
        for i, p in enumerate(ax.patches):
            if i < len(comparison_df):  # Add safety check
                row = comparison_df.iloc[i]
                if row['mad'] is not None:
                    ax.errorbar(p.get_x() + p.get_width()/2., row['value'], 
                            yerr=row['mad'], fmt='none', color='black', capsize=3)
        
        plt.title(f'Raw vs. Frequency-Adjusted {title}')
        plt.xlabel('Bigram Type')
        
        # Set appropriate y-axis label and format
        if 'Error' in title:
            plt.ylabel('Error Rate')
            # Format labels as percentages for error rates
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 0.002, f"{height*100:.2f}%",
                       ha='center')
        else:
            plt.ylabel('Time (ms)')
            # Format labels as milliseconds for times
            for i, p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2., height + 5, f"{height:.1f}ms",
                       ha='center')
        
        plt.legend(title='Metric Type')
        plt.tight_layout()
        
        # Save the figure
        metric_name = title.replace(' ', '_').lower()
        plt.savefig(f'{figures_dir}freq_adjusted_{metric_name}.png')
        plt.close()
        
        # Create scatter plot of raw vs. adjusted values
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(freq_adjusted_same_key[raw_col], freq_adjusted_same_key[adj_col], alpha=0.7, label='Same-Key')
        plt.title(f'Raw vs. Adjusted {title} (Same-Key)')
        plt.xlabel(f'Raw {title}')
        plt.ylabel(f'Frequency-Adjusted {title}')
        plt.grid(True, alpha=0.3)
        
        # Add diagonal line
        if 'Error' in title:
            max_val = max(freq_adjusted_same_key[raw_col].max(), freq_adjusted_same_key[adj_col].max()) * 1.1
        else:
            max_val = max(freq_adjusted_same_key[raw_col].max(), freq_adjusted_same_key[adj_col].max()) * 1.1
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        plt.subplot(1, 2, 2)
        plt.scatter(freq_adjusted_different_key[raw_col], freq_adjusted_different_key[adj_col], alpha=0.7, label='Different-Key')
        plt.title(f'Raw vs. Adjusted {title} (Different-Key)')
        plt.xlabel(f'Raw {title}')
        plt.ylabel(f'Frequency-Adjusted {title}')
        plt.grid(True, alpha=0.3)
        
        # Add diagonal line
        if 'Error' in title:
            max_val = max(freq_adjusted_different_key[raw_col].max(), freq_adjusted_different_key[adj_col].max()) * 1.1
        else:
            max_val = max(freq_adjusted_different_key[raw_col].max(), freq_adjusted_different_key[adj_col].max()) * 1.1
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{figures_dir}freq_adjusted_{metric_name}_scatter.png')
        plt.close()

def print_mirror_pair_stats(mirror_df):
    """
    Print statistics for mirror image key pairs
    
    Parameters:
    mirror_df (DataFrame): DataFrame of mirror pair comparison statistics
    """
    if mirror_df.empty:
        print("No data available for mirror pairs with sufficient samples")
        return
    
    print("\n" + "="*50)
    print("MIRROR PAIR KEY ANALYSIS")
    print("="*50)
    
    # Sort by absolute error rate difference
    mirror_df['abs_error_diff'] = mirror_df['error_diff'].abs()
    
    print("\nMirror pairs sorted by error rate difference:")
    print(f"{'Pair':<10} {'Left Count':<10} {'Right Count':<10} {'Left Error':<15} {'Right Error':<15} {'Difference':<15}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    
    for _, row in mirror_df.sort_values('abs_error_diff', ascending=False).iterrows():
        pair = f"{row['left_key']}-{row['right_key']}"
        left_err = f"{row['left_error_rate']*100:.2f}%"
        right_err = f"{row['right_error_rate']*100:.2f}%"
        diff = f"{row['error_diff']*100:+.2f}%"  # Add + sign for positive differences
        
        print(f"{pair:<10} {row['left_count']:<10} {row['right_count']:<10} {left_err:<15} {right_err:<15} {diff:<15}")
    
    # Sort by absolute time difference
    mirror_df['abs_time_diff'] = mirror_df['time_diff'].abs()
    
    print("\nMirror pairs sorted by typing time difference:")
    print(f"{'Pair':<10} {'Left Count':<10} {'Right Count':<10} {'Left Time':<15} {'Right Time':<15} {'Difference':<15}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    
    for _, row in mirror_df.sort_values('abs_time_diff', ascending=False).iterrows():
        pair = f"{row['left_key']}-{row['right_key']}"
        # Use median time values
        left_time = f"{row['left_median_time']:.1f}ms"
        right_time = f"{row['right_median_time']:.1f}ms"
        diff = f"{row['time_diff']:+.1f}ms"  # Add + sign for positive differences
        
        print(f"{pair:<10} {row['left_count']:<10} {row['right_count']:<10} {left_time:<15} {right_time:<15} {diff:<15}")
    
    # Print frequency-adjusted differences if available
    if 'error_diff_freq_adjusted' in mirror_df.columns:
        print("\nFrequency-adjusted mirror pairs sorted by error rate difference:")
        print(f"{'Pair':<10} {'Raw Diff':<15} {'Adj Diff':<15} {'Change':<15}")
        print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*15}")
        
        for _, row in mirror_df.sort_values('error_diff_freq_adjusted', key=abs, ascending=False).iterrows():
            pair = f"{row['left_key']}-{row['right_key']}"
            raw_diff = f"{row['error_diff']*100:+.2f}%"
            adj_diff = f"{row['error_diff_freq_adjusted']*100:+.2f}%"
            change = f"{(row['error_diff_freq_adjusted'] - row['error_diff'])*100:+.2f}%"
            
            print(f"{pair:<10} {raw_diff:<15} {adj_diff:<15} {change:<15}")


def check_first_key_error_rate(different_key_df):
    """
    Check if first-key error rate is zero for different-key bigrams
    
    Parameters:
    different_key_df (DataFrame): DataFrame of different-key bigram statistics
    """
    print("\n" + "="*50)
    print("CHECKING FIRST-KEY ERROR RATE FOR DIFFERENT-KEY BIGRAMS")
    print("="*50)
    
    first_key_error_rate = different_key_df['firstKeyErrorRate'].mean()
    print(f"First-key error rate for different-key bigrams: {first_key_error_rate:.4f}")
    print(f"Range: {different_key_df['firstKeyErrorRate'].min():.4f} to {different_key_df['firstKeyErrorRate'].max():.4f}")
    
    # Count how many different-key bigrams have non-zero first key error rate
    non_zero_count = (different_key_df['firstKeyErrorRate'] > 0).sum()
    print(f"Number of different-key bigrams with non-zero first-key error rate: {non_zero_count} out of {len(different_key_df)}")
    
    # If there are any with non-zero error rates, show examples
    if non_zero_count > 0:
        print("\nExamples of different-key bigrams with non-zero first-key error rates:")
        for _, row in different_key_df[different_key_df['firstKeyErrorRate'] > 0].head(5).iterrows():
            print(f"Bigram: {row['bigram']}, Error Rate: {row['firstKeyErrorRate']:.4f}, Count: {row['totalCount']}")
    
    # Create visualization of first-key error rates
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Prepare data for grouped bar plot
    error_data = []
    
    # First key error rates
    error_data.append({
        'position': 'First Key', 
        'type': 'Same-Key Bigrams', 
        'error_rate': same_key_df['firstKeyErrorRate'].mean() if 'same_key_df' in locals() else 0
    })
    error_data.append({
        'position': 'First Key', 
        'type': 'Different-Key Bigrams', 
        'error_rate': different_key_df['firstKeyErrorRate'].mean()
    })
    
    # Second key error rates
    error_data.append({
        'position': 'Second Key', 
        'type': 'Same-Key Bigrams', 
        'error_rate': same_key_df['secondKeyErrorRate'].mean() if 'same_key_df' in locals() else 0
    })
    error_data.append({
        'position': 'Second Key', 
        'type': 'Different-Key Bigrams', 
        'error_rate': different_key_df['secondKeyErrorRate'].mean()
    })
    
    error_df = pd.DataFrame(error_data)
    
    # Create grouped bar plot with special emphasis on first-key different-key
    ax = sns.barplot(x='position', y='error_rate', hue='type', data=error_df)
    
    # Highlight the bar we're examining
    for i, p in enumerate(ax.patches):
        if i == 1:  # This is the first-key different-key bar
            p.set_edgecolor('red')
            p.set_linewidth(2)
        
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.002, f"{height*100:.2f}%",
                ha='center', fontweight='bold' if i == 1 else 'normal')
    
    plt.title('First-Key vs. Second-Key Error Rates')
    plt.xlabel('Key Position')
    plt.ylabel('Error Rate')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Bigram Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}first_key_error_rate_verification.png')
    plt.close()
    
    # Histogram of first-key error rates
    plt.figure(figsize=(12, 6))
    
    # Plot distributions for first key error rates for different-key bigrams
    sns.histplot(different_key_df['firstKeyErrorRate'], bins=30, kde=True)
    plt.title('Distribution of First-Key Error Rates for Different-Key Bigrams')
    plt.xlabel('Error Rate')
    plt.ylabel('Count')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}first_key_error_rate_histogram.png')
    plt.close()
    
    return first_key_error_rate, non_zero_count

def analyze_same_vs_different_key_bigrams(bigram_stats):
    """
    Analyze and compare same-key bigrams vs. different-key bigrams
    
    Parameters:
    bigram_stats (dict): Dictionary of bigram statistics
    
    Returns:
    tuple: (same_key_df, different_key_df, comparison_results)
    """
    # Define output directory here to make it accessible to all code in this function
    output_dir = 'output/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert dictionary to DataFrame for easier analysis
    bigram_df = pd.DataFrame(list(bigram_stats.values()))
    
    # Make sure we have the is_same_key column
    if 'is_same_key' not in bigram_df.columns:
        print("Error: 'is_same_key' column not found in bigram data")
        return None, None, None
    
    # Separate into same-key and different-key groups
    same_key_bigrams = bigram_df[bigram_df['is_same_key'] == True]
    different_key_bigrams = bigram_df[bigram_df['is_same_key'] == False]
    
    print("\n" + "="*50)
    print("SAME-KEY VS. DIFFERENT-KEY BIGRAM ANALYSIS")
    print("="*50)
    
    print(f"\nFound {len(same_key_bigrams)} same-key bigrams and {len(different_key_bigrams)} different-key bigrams")
    
    # Filter for bigrams with sufficient data
    min_count = 10
    same_key_filtered = same_key_bigrams[same_key_bigrams['totalCount'] > min_count]
    different_key_filtered = different_key_bigrams[different_key_bigrams['totalCount'] > min_count]
    
    print(f"After filtering (min count = {min_count}): {len(same_key_filtered)} same-key and {len(different_key_filtered)} different-key bigrams")
    
    # Calculate summary statistics for each group
    summary_metrics = [
        ('Total Count', 'totalCount', 'mean'),
        ('First Key Error Rate', 'firstKeyErrorRate', 'mean'),
        ('Second Key Error Rate', 'secondKeyErrorRate', 'mean'),
        ('First Key Median Time (ms)', 'firstKeyMedianTime', 'mean'),
        ('Second Key Median Time (ms)', 'secondKeyMedianTime', 'mean'),
        ('First Key Time MAD (ms)', 'firstKeyTimeMAD', 'mean'),
        ('Second Key Time MAD (ms)', 'secondKeyTimeMAD', 'mean'),
        ('First Key Error MAD', 'firstKeyErrorMAD', 'mean'),
        ('Second Key Error MAD', 'secondKeyErrorMAD', 'mean')
    ]
    
    print("\nSummary Statistics:")
    print(f"{'Metric':<30} {'Same-Key':^15} {'Different-Key':^15} {'Difference':^15} {'Significance':^10}")
    print(f"{'-'*30} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
    
    comparison_results = {}
    
    for label, column, agg_method in summary_metrics:
        # Skip if column is not in the DataFrame
        if column not in same_key_filtered.columns or column not in different_key_filtered.columns:
            continue
            
        # Calculate statistics
        same_key_stat = getattr(same_key_filtered[column], agg_method)()
        diff_key_stat = getattr(different_key_filtered[column], agg_method)()
        difference = same_key_stat - diff_key_stat
        
        # Perform t-test for statistical significance
        from scipy import stats
        _, p_value = stats.ttest_ind(
            same_key_filtered[column].dropna(), 
            different_key_filtered[column].dropna(),
            equal_var=False
        )
        
        # Format the significance indicator
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "n.s."
            
        # Format for display
        if 'Rate' in label or 'MAD' in label and 'Error' in label:
            same_key_display = f"{same_key_stat*100:.2f}%"
            diff_key_display = f"{diff_key_stat*100:.2f}%"
            diff_display = f"{difference*100:+.2f}%"
        else:
            same_key_display = f"{same_key_stat:.2f}"
            diff_key_display = f"{diff_key_stat:.2f}"
            diff_display = f"{difference:+.2f}"
            
        print(f"{label:<30} {same_key_display:^15} {diff_key_display:^15} {diff_display:^15} {sig:^10}")
        
        # Store results for later use
        comparison_results[label] = {
            'same_key': same_key_stat,
            'different_key': diff_key_stat,
            'difference': difference,
            'p_value': p_value
        }
    
    # Check for flanked bigrams
    flanked_same_key = same_key_filtered[same_key_filtered['flankedCount'] > 0]
    flanked_diff_key = different_key_filtered[different_key_filtered['flankedCount'] > 0]
    
    print(f"\nFlanked Bigrams (surrounded by spaces):")
    print(f"Found {len(flanked_same_key)} flanked same-key and {len(flanked_diff_key)} flanked different-key bigrams")
    
    # Create visualizations for the comparison
    visualize_same_vs_different_key_bigrams_with_error_bars(same_key_filtered, different_key_filtered, comparison_results)
    
    # Check first-key error rate for different-key bigrams
    print("\nChecking first-key error rates for different-key bigrams...")
    check_first_key_error_rate(different_key_filtered)

    # Load letter frequency data if available
    letter_freq_path = 'input/letter_frequencies_english.csv'
    letter_frequencies = None
    if os.path.exists(letter_freq_path):
        try:
            letter_frequencies = pd.read_csv(letter_freq_path)
            print(f"\nLoaded letter frequency data: {len(letter_frequencies)} entries")
            
            # Perform frequency-adjusted analysis
            print("\nPerforming frequency-adjusted analysis...")
            freq_adjusted_same_key, freq_adjusted_different_key = adjust_bigrams_for_frequency(
                same_key_filtered, different_key_filtered, letter_frequencies
            )
            
            # Compare frequency-adjusted metrics
            compare_frequency_adjusted_bigrams(freq_adjusted_same_key, freq_adjusted_different_key, output_dir)
            
            # Save frequency-adjusted results
            freq_adjusted_same_key.to_csv(f"{output_dir}freq_adjusted_same_key_bigram_analysis.csv", index=False)
            freq_adjusted_different_key.to_csv(f"{output_dir}freq_adjusted_different_key_bigram_analysis.csv", index=False)
            
        except Exception as e:
            print(f"Error in frequency-adjusted analysis: {str(e)}")
    else:
        print(f"\nLetter frequency data not found at {letter_freq_path}. Skipping frequency-adjusted analysis.")

    # Save results to CSV
    same_key_filtered.to_csv(f"{output_dir}same_key_bigram_analysis.csv", index=False)
    different_key_filtered.to_csv(f"{output_dir}different_key_bigram_analysis.csv", index=False)
    
    # Bigram frequency analysis
    bigram_freq_path = 'input/letter_pair_frequencies_english.csv'
    if os.path.exists(bigram_freq_path):
        try:
            bigram_frequencies = pd.read_csv(bigram_freq_path)
            print(f"\nLoaded bigram frequency data: {len(bigram_frequencies)} entries")
            
            # Analyze with bigram frequencies
            analyze_with_bigram_frequencies(same_key_filtered, different_key_filtered, bigram_frequencies)
            
        except Exception as e:
            print(f"Error in bigram frequency analysis: {str(e)}")
    else:
        print(f"\nBigram frequency data not found at {bigram_freq_path}")
            
    return same_key_filtered, different_key_filtered, comparison_results

def visualize_same_vs_different_key_bigrams_with_error_bars(same_key_df, different_key_df, comparison_results):
    """
    Create visualizations comparing same-key and different-key bigrams with error bars
    
    Parameters:
    same_key_df (DataFrame): DataFrame of same-key bigram statistics
    different_key_df (DataFrame): DataFrame of different-key bigram statistics
    comparison_results (dict): Dictionary of comparison statistics
    """
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Error rate comparison with MAD error bars
    plt.figure(figsize=(12, 6))
    
    # Prepare data for grouped bar plot with error information
    error_data = []
    error_bars = []
    
    # First key error rates
    first_same_err = same_key_df['firstKeyErrorRate'].mean()
    first_same_mad = same_key_df['firstKeyErrorMAD'].mean() if 'firstKeyErrorMAD' in same_key_df.columns else 0
    error_data.append({
        'position': 'First Key', 
        'type': 'Same-Key Bigrams', 
        'error_rate': first_same_err
    })
    error_bars.append(first_same_mad)
    
    first_diff_err = different_key_df['firstKeyErrorRate'].mean()
    first_diff_mad = different_key_df['firstKeyErrorMAD'].mean() if 'firstKeyErrorMAD' in different_key_df.columns else 0
    error_data.append({
        'position': 'First Key', 
        'type': 'Different-Key Bigrams', 
        'error_rate': first_diff_err
    })
    error_bars.append(first_diff_mad)
    
    # Second key error rates
    second_same_err = same_key_df['secondKeyErrorRate'].mean()
    second_same_mad = same_key_df['secondKeyErrorMAD'].mean() if 'secondKeyErrorMAD' in same_key_df.columns else 0
    error_data.append({
        'position': 'Second Key', 
        'type': 'Same-Key Bigrams', 
        'error_rate': second_same_err
    })
    error_bars.append(second_same_mad)
    
    second_diff_err = different_key_df['secondKeyErrorRate'].mean()
    second_diff_mad = different_key_df['secondKeyErrorMAD'].mean() if 'secondKeyErrorMAD' in different_key_df.columns else 0
    error_data.append({
        'position': 'Second Key', 
        'type': 'Different-Key Bigrams', 
        'error_rate': second_diff_err
    })
    error_bars.append(second_diff_mad)
    
    error_df = pd.DataFrame(error_data)
    
    # Create grouped bar plot
    ax = sns.barplot(x='position', y='error_rate', hue='type', data=error_df)
    
    # Add error bars manually - ensure index matches
    for i, p in enumerate(ax.patches):
        if i < len(error_bars):  # Add safety check
            height = p.get_height()
            ax.errorbar(p.get_x() + p.get_width()/2., height, 
                        yerr=error_bars[i], fmt='none', color='black', capsize=5)
            ax.text(p.get_x() + p.get_width()/2., height + 0.002, f"{height*100:.2f}%",
                    ha='center')
    
    plt.title('Error Rate Comparison: Same-Key vs. Different-Key Bigrams (with MAD)')
    plt.xlabel('Key Position')
    plt.ylabel('Error Rate')
    plt.legend(title='Bigram Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}same_vs_different_key_error_rates_with_mad.png')
    plt.close()
    
    # 2. Typing time comparison with MAD error bars
    plt.figure(figsize=(12, 6))
    
    # Prepare data for grouped bar plot with error information
    time_data = []
    time_error_bars = []
    
    # First key times
    first_same_time = same_key_df['firstKeyMedianTime'].mean()
    first_same_mad = same_key_df['firstKeyTimeMAD'].mean() if 'firstKeyTimeMAD' in same_key_df.columns else 0
    time_data.append({
        'position': 'First Key', 
        'type': 'Same-Key Bigrams', 
        'time': first_same_time
    })
    time_error_bars.append(first_same_mad)
    
    first_diff_time = different_key_df['firstKeyMedianTime'].mean()
    first_diff_mad = different_key_df['firstKeyTimeMAD'].mean() if 'firstKeyTimeMAD' in different_key_df.columns else 0
    time_data.append({
        'position': 'First Key', 
        'type': 'Different-Key Bigrams', 
        'time': first_diff_time
    })
    time_error_bars.append(first_diff_mad)
    
    # Second key times
    second_same_time = same_key_df['secondKeyMedianTime'].mean()
    second_same_mad = same_key_df['secondKeyTimeMAD'].mean() if 'secondKeyTimeMAD' in same_key_df.columns else 0
    time_data.append({
        'position': 'Second Key', 
        'type': 'Same-Key Bigrams', 
        'time': second_same_time
    })
    time_error_bars.append(second_same_mad)
    
    second_diff_time = different_key_df['secondKeyMedianTime'].mean()
    second_diff_mad = different_key_df['secondKeyTimeMAD'].mean() if 'secondKeyTimeMAD' in different_key_df.columns else 0
    time_data.append({
        'position': 'Second Key', 
        'type': 'Different-Key Bigrams', 
        'time': second_diff_time
    })
    time_error_bars.append(second_diff_mad)
    
    time_df = pd.DataFrame(time_data)
    
    # Create grouped bar plot
    ax = sns.barplot(x='position', y='time', hue='type', data=time_df)
    
    # Add error bars manually - ensure index doesn't exceed list length
    for i, p in enumerate(ax.patches):
        if i < len(time_error_bars):  # Add safety check
            height = p.get_height()
            ax.errorbar(p.get_x() + p.get_width()/2., height, 
                        yerr=time_error_bars[i], fmt='none', color='black', capsize=5)
            ax.text(p.get_x() + p.get_width()/2., height + 5, f"{height:.1f}ms",
                    ha='center')
    
    plt.title('Typing Time Comparison: Same-Key vs. Different-Key Bigrams (with MAD)')
    plt.xlabel('Key Position')
    plt.ylabel('Median Time (ms)')
    plt.legend(title='Bigram Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}same_vs_different_key_typing_times_with_mad.png')
    plt.close()
    
    # 3. Error rates distribution comparison
    plt.figure(figsize=(14, 6))
    
    # Plot distributions for first key error rates
    plt.subplot(1, 2, 1)
    sns.histplot(same_key_df['firstKeyErrorRate'], alpha=0.5, label='Same-Key', kde=True, color='blue')
    sns.histplot(different_key_df['firstKeyErrorRate'], alpha=0.5, label='Different-Key', kde=True, color='orange')
    plt.title('First Key Error Rate Distribution')
    plt.xlabel('Error Rate')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot distributions for second key error rates
    plt.subplot(1, 2, 2)
    sns.histplot(same_key_df['secondKeyErrorRate'], alpha=0.5, label='Same-Key', kde=True, color='blue')
    sns.histplot(different_key_df['secondKeyErrorRate'], alpha=0.5, label='Different-Key', kde=True, color='orange')
    plt.title('Second Key Error Rate Distribution')
    plt.xlabel('Error Rate')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}same_vs_different_key_error_distributions.png')
    plt.close()
    
    # 4. Typing time distribution comparison
    plt.figure(figsize=(14, 6))
    
    # Plot distributions for first key typing times
    plt.subplot(1, 2, 1)
    sns.histplot(same_key_df['firstKeyMedianTime'], alpha=0.5, label='Same-Key', kde=True, color='blue')
    sns.histplot(different_key_df['firstKeyMedianTime'], alpha=0.5, label='Different-Key', kde=True, color='orange')
    plt.title('First Key Typing Time Distribution')
    plt.xlabel('Median Time (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot distributions for second key typing times
    plt.subplot(1, 2, 2)
    sns.histplot(same_key_df['secondKeyMedianTime'], alpha=0.5, label='Same-Key', kde=True, color='blue')
    sns.histplot(different_key_df['secondKeyMedianTime'], alpha=0.5, label='Different-Key', kde=True, color='orange')
    plt.title('Second Key Typing Time Distribution')
    plt.xlabel('Median Time (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}same_vs_different_key_time_distributions.png')
    plt.close()
    
    # 5. Scatterplot comparing first key vs second key metrics with MAD error bars
    plt.figure(figsize=(16, 7))
    
    # Error rate comparison with error bars
    plt.subplot(1, 2, 1)
    
    # Plot same-key points with error bars
    plt.errorbar(
        same_key_df['firstKeyErrorRate'], same_key_df['secondKeyErrorRate'],
        xerr=same_key_df['firstKeyErrorMAD'] if 'firstKeyErrorMAD' in same_key_df.columns else None,
        yerr=same_key_df['secondKeyErrorMAD'] if 'secondKeyErrorMAD' in same_key_df.columns else None,
        fmt='o', alpha=0.7, label='Same-Key', capsize=3, ecolor='lightgray'
    )
    
    # Plot different-key points with error bars
    plt.errorbar(
        different_key_df['firstKeyErrorRate'], different_key_df['secondKeyErrorRate'],
        xerr=different_key_df['firstKeyErrorMAD'] if 'firstKeyErrorMAD' in different_key_df.columns else None,
        yerr=different_key_df['secondKeyErrorMAD'] if 'secondKeyErrorMAD' in different_key_df.columns else None,
        fmt='s', alpha=0.7, label='Different-Key', capsize=3, ecolor='lightgray'
    )
    
    # Add diagonal line
    max_error = max(
        same_key_df['firstKeyErrorRate'].max(), same_key_df['secondKeyErrorRate'].max(),
        different_key_df['firstKeyErrorRate'].max(), different_key_df['secondKeyErrorRate'].max()
    ) * 1.1
    plt.plot([0, max_error], [0, max_error], 'k--', alpha=0.5)
    
    plt.title('First Key vs. Second Key Error Rates (with MAD)')
    plt.xlabel('First Key Error Rate')
    plt.ylabel('Second Key Error Rate')
    plt.xlim(0, max_error)
    plt.ylim(0, max_error)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Typing time comparison with error bars
    plt.subplot(1, 2, 2)
    
    # Plot same-key points with error bars
    plt.errorbar(
        same_key_df['firstKeyMedianTime'], same_key_df['secondKeyMedianTime'],
        xerr=same_key_df['firstKeyTimeMAD'] if 'firstKeyTimeMAD' in same_key_df.columns else None,
        yerr=same_key_df['secondKeyTimeMAD'] if 'secondKeyTimeMAD' in same_key_df.columns else None,
        fmt='o', alpha=0.7, label='Same-Key', capsize=3, ecolor='lightgray'
    )
    
    # Plot different-key points with error bars
    plt.errorbar(
        different_key_df['firstKeyMedianTime'], different_key_df['secondKeyMedianTime'],
        xerr=different_key_df['firstKeyTimeMAD'] if 'firstKeyTimeMAD' in different_key_df.columns else None,
        yerr=different_key_df['secondKeyTimeMAD'] if 'secondKeyTimeMAD' in different_key_df.columns else None,
        fmt='s', alpha=0.7, label='Different-Key', capsize=3, ecolor='lightgray'
    )
    
    # Add diagonal line
    max_time = max(
        same_key_df['firstKeyMedianTime'].max(), same_key_df['secondKeyMedianTime'].max(),
        different_key_df['firstKeyMedianTime'].max(), different_key_df['secondKeyMedianTime'].max()
    ) * 1.1
    plt.plot([0, max_time], [0, max_time], 'k--', alpha=0.5)
    
    plt.title('First Key vs. Second Key Typing Times (with MAD)')
    plt.xlabel('First Key Median Time (ms)')
    plt.ylabel('Second Key Median Time (ms)')
    plt.xlim(0, max_time)
    plt.ylim(0, max_time)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}same_vs_different_key_scatter_with_mad.png')
    plt.close()
    
    # 6. Check for frequency-adjusted values and plot if available
    if ('firstKeyErrorRate_freq_adjusted' in same_key_df.columns and 
        'secondKeyErrorRate_freq_adjusted' in same_key_df.columns):
        
        # Create scatter plots with both raw and frequency-adjusted points
        plt.figure(figsize=(16, 7))
        
        # Error rates
        plt.subplot(1, 2, 1)
        
        # Raw points
        plt.scatter(
            same_key_df['firstKeyErrorRate'], same_key_df['secondKeyErrorRate'],
            alpha=0.7, label='Same-Key (Raw)', marker='o', s=50
        )
        plt.scatter(
            different_key_df['firstKeyErrorRate'], different_key_df['secondKeyErrorRate'],
            alpha=0.7, label='Different-Key (Raw)', marker='s', s=50
        )
        
        # Frequency-adjusted points
        plt.scatter(
            same_key_df['firstKeyErrorRate_freq_adjusted'], same_key_df['secondKeyErrorRate_freq_adjusted'],
            alpha=0.7, label='Same-Key (Freq-Adj)', marker='o', edgecolors='black', s=50
        )
        plt.scatter(
            different_key_df['firstKeyErrorRate_freq_adjusted'], different_key_df['secondKeyErrorRate_freq_adjusted'],
            alpha=0.7, label='Different-Key (Freq-Adj)', marker='s', edgecolors='black', s=50
        )
        
        # Add diagonal line
        max_error = max(
            same_key_df['firstKeyErrorRate'].max(), same_key_df['secondKeyErrorRate'].max(),
            different_key_df['firstKeyErrorRate'].max(), different_key_df['secondKeyErrorRate'].max(),
            same_key_df['firstKeyErrorRate_freq_adjusted'].max(), same_key_df['secondKeyErrorRate_freq_adjusted'].max(),
            different_key_df['firstKeyErrorRate_freq_adjusted'].max(), different_key_df['secondKeyErrorRate_freq_adjusted'].max()
        ) * 1.1
        plt.plot([0, max_error], [0, max_error], 'k--', alpha=0.5)
        
        plt.title('Error Rates: Raw vs. Frequency-Adjusted')
        plt.xlabel('First Key Error Rate')
        plt.ylabel('Second Key Error Rate')
        plt.xlim(0, max_error)
        plt.ylim(0, max_error)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Typing times
        if ('firstKeyMedianTime_freq_adjusted' in same_key_df.columns and 
            'secondKeyMedianTime_freq_adjusted' in same_key_df.columns):
            
            plt.subplot(1, 2, 2)
            
            # Raw points
            plt.scatter(
                same_key_df['firstKeyMedianTime'], same_key_df['secondKeyMedianTime'],
                alpha=0.7, label='Same-Key (Raw)', marker='o', s=50
            )
            plt.scatter(
                different_key_df['firstKeyMedianTime'], different_key_df['secondKeyMedianTime'],
                alpha=0.7, label='Different-Key (Raw)', marker='s', s=50
            )
            
            # Frequency-adjusted points
            plt.scatter(
                same_key_df['firstKeyMedianTime_freq_adjusted'], same_key_df['secondKeyMedianTime_freq_adjusted'],
                alpha=0.7, label='Same-Key (Freq-Adj)', marker='o', edgecolors='black', s=50
            )
            plt.scatter(
                different_key_df['firstKeyMedianTime_freq_adjusted'], different_key_df['secondKeyMedianTime_freq_adjusted'],
                alpha=0.7, label='Different-Key (Freq-Adj)', marker='s', edgecolors='black', s=50
            )
            
            # Add diagonal line
            max_time = max(
                same_key_df['firstKeyMedianTime'].max(), same_key_df['secondKeyMedianTime'].max(),
                different_key_df['firstKeyMedianTime'].max(), different_key_df['secondKeyMedianTime'].max(),
                same_key_df['firstKeyMedianTime_freq_adjusted'].max(), same_key_df['secondKeyMedianTime_freq_adjusted'].max(),
                different_key_df['firstKeyMedianTime_freq_adjusted'].max(), different_key_df['secondKeyMedianTime_freq_adjusted'].max()
            ) * 1.1
            plt.plot([0, max_time], [0, max_time], 'k--', alpha=0.5)
            
            plt.title('Typing Times: Raw vs. Frequency-Adjusted')
            plt.xlabel('First Key Median Time (ms)')
            plt.ylabel('Second Key Median Time (ms)')
            plt.xlim(0, max_time)
            plt.ylim(0, max_time)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}same_vs_different_key_raw_vs_adjusted.png')
        plt.close()

def main():
    """Main function to execute the analysis pipeline"""
    # Get all CSV files in the directory
    csv_path = 'input/raws_Prolific/*.csv'  # Replace with your actual path
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
    
    # Analyze all letters for mirror pair comparison
    all_letter_stats = analyze_letters(processed_data, include_mirror_pairs=True)
    
    # Analyze left home block letters specifically
    left_home_stats = analyze_letters(processed_data, keys_to_include=LEFT_HOME_KEYS)
    
    # Analyze bigrams with left home keys
    left_home_bigram_stats = analyze_bigrams(processed_data, keys_to_include=LEFT_HOME_KEYS)
    
    # Convert to DataFrames for analysis
    all_letter_df = pd.DataFrame([stats for stats in all_letter_stats.values()])
    left_home_letter_df = pd.DataFrame([stats for stats in left_home_stats.values()])
    left_home_bigram_df = pd.DataFrame([stats for stats in left_home_bigram_stats.values()])
    
    # Analyze mirror pairs
    mirror_pair_df = analyze_mirror_pairs(all_letter_stats, letter_frequencies)
    
    # Print statistics for left home block keys
    print_left_home_stats(left_home_letter_df)
    
    # Print statistics for mirror pairs
    print_mirror_pair_stats(mirror_pair_df)
    
    # Analyze same-key vs. different-key bigrams
    print("\nAnalyzing same-key vs. different-key bigrams...")
    same_key_df, different_key_df, comparison_results = analyze_same_vs_different_key_bigrams(left_home_bigram_stats)

    # Create visualizations
    visualize_left_home_results(left_home_letter_df)
    visualize_mirror_pairs(mirror_pair_df)
    
    # Generate a comprehensive frequency-adjusted comparison visualization
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a combined visualization showing all frequency adjustments
    create_comprehensive_freq_adjusted_comparison(all_letter_df, same_key_df, different_key_df, mirror_pair_df)
    
    # Save results to CSV files for further analysis
    output_dir = 'output/'
    os.makedirs(output_dir, exist_ok=True)
    
    left_home_letter_df.to_csv(f"{output_dir}left_home_letter_analysis.csv", index=False)
    left_home_bigram_df.to_csv(f"{output_dir}left_home_bigram_analysis.csv", index=False)
    mirror_pair_df.to_csv(f"{output_dir}mirror_pair_analysis.csv", index=False)
    
    print(f"\nResults saved to {output_dir}")
    
    return {
        'left_home_letter_stats': left_home_stats,
        'left_home_bigram_stats': left_home_bigram_stats,
        'mirror_pair_df': mirror_pair_df,
        'same_key_df': same_key_df,
        'different_key_df': different_key_df
    }

def create_comprehensive_freq_adjusted_comparison(letter_df, same_key_df, different_key_df, mirror_pair_df):
    """
    Create a comprehensive visualization comparing raw and frequency-adjusted metrics
    across letters, same-key bigrams, different-key bigrams, and mirror pairs
    
    Parameters:
    letter_df (DataFrame): Letter statistics DataFrame
    same_key_df (DataFrame): Same-key bigram statistics DataFrame
    different_key_df (DataFrame): Different-key bigram statistics DataFrame
    mirror_pair_df (DataFrame): Mirror pair statistics DataFrame
    """
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which frequency-adjusted columns are available
    has_letter_freq_adj = ('errorRate_freq_adjusted' in letter_df.columns and 
                          'medianTime_freq_adjusted' in letter_df.columns)
    
    has_bigram_freq_adj = ('firstKeyErrorRate_freq_adjusted' in same_key_df.columns and
                          'secondKeyErrorRate_freq_adjusted' in same_key_df.columns and
                          'firstKeyMedianTime_freq_adjusted' in same_key_df.columns and
                          'secondKeyMedianTime_freq_adjusted' in same_key_df.columns)
    
    has_mirror_freq_adj = ('left_error_rate_freq_adjusted' in mirror_pair_df.columns and
                          'right_error_rate_freq_adjusted' in mirror_pair_df.columns and
                          'left_median_time_freq_adjusted' in mirror_pair_df.columns and
                          'right_median_time_freq_adjusted' in mirror_pair_df.columns)
    
    # Create data for the comprehensive error rate comparison
    error_data = []
    
    # Add letter data if available
    if has_letter_freq_adj:
        error_data.append({
            'category': 'Single Letters', 
            'metric': 'Error Rate',
            'raw': letter_df['errorRate'].mean(),
            'adjusted': letter_df['errorRate_freq_adjusted'].mean()
        })
    
    # Add bigram data if available
    if has_bigram_freq_adj:
        # Same-key bigrams
        error_data.append({
            'category': 'Same-Key (First)', 
            'metric': 'Error Rate',
            'raw': same_key_df['firstKeyErrorRate'].mean(),
            'adjusted': same_key_df['firstKeyErrorRate_freq_adjusted'].mean()
        })
        
        error_data.append({
            'category': 'Same-Key (Second)', 
            'metric': 'Error Rate',
            'raw': same_key_df['secondKeyErrorRate'].mean(),
            'adjusted': same_key_df['secondKeyErrorRate_freq_adjusted'].mean()
        })
        
        # Different-key bigrams
        error_data.append({
            'category': 'Different-Key (First)', 
            'metric': 'Error Rate',
            'raw': different_key_df['firstKeyErrorRate'].mean(),
            'adjusted': different_key_df['firstKeyErrorRate_freq_adjusted'].mean()
        })
        
        error_data.append({
            'category': 'Different-Key (Second)', 
            'metric': 'Error Rate',
            'raw': different_key_df['secondKeyErrorRate'].mean(),
            'adjusted': different_key_df['secondKeyErrorRate_freq_adjusted'].mean()
        })
    
    # Add mirror pair data if available
    if has_mirror_freq_adj:
        error_data.append({
            'category': 'Mirror Left', 
            'metric': 'Error Rate',
            'raw': mirror_pair_df['left_error_rate'].mean(),
            'adjusted': mirror_pair_df['left_error_rate_freq_adjusted'].mean()
        })
        
        error_data.append({
            'category': 'Mirror Right', 
            'metric': 'Error Rate',
            'raw': mirror_pair_df['right_error_rate'].mean(),
            'adjusted': mirror_pair_df['right_error_rate_freq_adjusted'].mean()
        })
    
    # Create DataFrame for plotting if we have data
    if error_data:
        error_df = pd.DataFrame(error_data)
        
        # Reshape for grouped bars
        error_melted = pd.melt(error_df, id_vars=['category', 'metric'], 
                           value_vars=['raw', 'adjusted'], 
                           var_name='adjustment', value_name='value')
        
        # Plot error rates
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='category', y='value', hue='adjustment', 
                        data=error_melted[error_melted['metric'] == 'Error Rate'])
        
        # Format labels
        plt.title('Comprehensive Frequency-Adjusted Error Rate Comparison')
        plt.xlabel('Category')
        plt.ylabel('Error Rate')
        plt.xticks(rotation=45)
        
        # Add percentage labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.002, f"{height*100:.2f}%",
                ha='center')
        
        plt.legend(title='Adjustment')
        plt.tight_layout()
        plt.savefig(f'{output_dir}comprehensive_freq_adjusted_error_rates.png')
        plt.close()
    
    # Repeat for typing times
    time_data = []
    
    # Add letter data if available
    if has_letter_freq_adj:
        time_data.append({
            'category': 'Single Letters', 
            'metric': 'Typing Time',
            'raw': letter_df['medianTime'].mean(),
            'adjusted': letter_df['medianTime_freq_adjusted'].mean()
        })
    
    # Add bigram data if available
    if has_bigram_freq_adj:
        # Same-key bigrams
        time_data.append({
            'category': 'Same-Key (First)', 
            'metric': 'Typing Time',
            'raw': same_key_df['firstKeyMedianTime'].mean(),
            'adjusted': same_key_df['firstKeyMedianTime_freq_adjusted'].mean()
        })
        
        time_data.append({
            'category': 'Same-Key (Second)', 
            'metric': 'Typing Time',
            'raw': same_key_df['secondKeyMedianTime'].mean(),
            'adjusted': same_key_df['secondKeyMedianTime_freq_adjusted'].mean()
        })
        
        # Different-key bigrams
        time_data.append({
            'category': 'Different-Key (First)', 
            'metric': 'Typing Time',
            'raw': different_key_df['firstKeyMedianTime'].mean(),
            'adjusted': different_key_df['firstKeyMedianTime_freq_adjusted'].mean()
        })
        
        time_data.append({
            'category': 'Different-Key (Second)', 
            'metric': 'Typing Time',
            'raw': different_key_df['secondKeyMedianTime'].mean(),
            'adjusted': different_key_df['secondKeyMedianTime_freq_adjusted'].mean()
        })
    
    # Add mirror pair data if available
    if has_mirror_freq_adj:
        time_data.append({
            'category': 'Mirror Left', 
            'metric': 'Typing Time',
            'raw': mirror_pair_df['left_median_time'].mean(),
            'adjusted': mirror_pair_df['left_median_time_freq_adjusted'].mean()
        })
        
        time_data.append({
            'category': 'Mirror Right', 
            'metric': 'Typing Time',
            'raw': mirror_pair_df['right_median_time'].mean(),
            'adjusted': mirror_pair_df['right_median_time_freq_adjusted'].mean()
        })
    
    # Create DataFrame for plotting if we have data
    if time_data:
        time_df = pd.DataFrame(time_data)
        
        # Reshape for grouped bars
        time_melted = pd.melt(time_df, id_vars=['category', 'metric'], 
                           value_vars=['raw', 'adjusted'], 
                           var_name='adjustment', value_name='value')
        
        # Plot typing times
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='category', y='value', hue='adjustment', 
                        data=time_melted[time_melted['metric'] == 'Typing Time'])
        
        # Format labels
        plt.title('Comprehensive Frequency-Adjusted Typing Time Comparison')
        plt.xlabel('Category')
        plt.ylabel('Median Time (ms)')
        plt.xticks(rotation=45)
        
        # Add time labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 5, f"{height:.1f}ms",
                ha='center')
        
        plt.legend(title='Adjustment')
        plt.tight_layout()
        plt.savefig(f'{output_dir}comprehensive_freq_adjusted_typing_times.png')
        plt.close()

if __name__ == "__main__":
    results = main()
    