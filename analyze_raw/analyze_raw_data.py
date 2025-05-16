import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from collections import defaultdict

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
    
    # Calculate error rates and average times
    for stats in letter_stats.values():
        stats['errorRate'] = stats['errorCount'] / stats['totalCount'] if stats['totalCount'] > 0 else 0
        stats['avgTime'] = stats['totalTime'] / len(stats['timeSamples']) if len(stats['timeSamples']) > 0 else 0
        
        # Calculate median time
        if len(stats['timeSamples']) > 0:
            stats['medianTime'] = np.median(stats['timeSamples'])
        else:
            stats['medianTime'] = 0
            
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
        
        # Median times
        if len(stats['firstKeyTimeSamples']) > 0:
            stats['firstKeyMedianTime'] = np.median(stats['firstKeyTimeSamples'])
        else:
            stats['firstKeyMedianTime'] = 0
            
        if len(stats['secondKeyTimeSamples']) > 0:
            stats['secondKeyMedianTime'] = np.median(stats['secondKeyTimeSamples'])
        else:
            stats['secondKeyMedianTime'] = 0
            
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

def analyze_mirror_pairs(letter_stats):
    """
    Analyze mirror image key pairs
    
    Parameters:
    letter_stats (dict): Dictionary of letter statistics
    
    Returns:
    DataFrame: Comparison of mirror pair statistics
    """
    mirror_data = []
    
    for left, right in MIRROR_PAIRS:
        if left in letter_stats and right in letter_stats:
            left_stats = letter_stats[left]
            right_stats = letter_stats[right]
            
            # Only include pairs where both keys have sufficient data
            if left_stats['totalCount'] > 10 and right_stats['totalCount'] > 10:
                mirror_data.append({
                    'left_key': left,
                    'right_key': right,
                    'left_count': left_stats['totalCount'],
                    'right_count': right_stats['totalCount'],
                    'left_error_rate': left_stats['errorRate'],
                    'right_error_rate': right_stats['errorRate'],
                    'left_avg_time': left_stats['avgTime'],
                    'right_avg_time': right_stats['avgTime'],
                    'left_most_common_mistype': left_stats['most_common_mistype'],
                    'right_most_common_mistype': right_stats['most_common_mistype'],
                    'error_diff': left_stats['errorRate'] - right_stats['errorRate'],
                    'time_diff': left_stats['avgTime'] - right_stats['avgTime']
                })
    
    return pd.DataFrame(mirror_data)

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
        left_time = f"{row['left_avg_time']:.1f}ms"
        right_time = f"{row['right_avg_time']:.1f}ms"
        diff = f"{row['time_diff']:+.1f}ms"  # Add + sign for positive differences
        
        print(f"{pair:<10} {row['left_count']:<10} {row['right_count']:<10} {left_time:<15} {right_time:<15} {diff:<15}")

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
    
    # 1. Error rates for left home block keys
    plt.figure(figsize=(12, 6))
    sns.barplot(x='letter', y='errorRate', data=left_home_df)
    plt.title('Error Rates for Left Home Block Keys')
    plt.xlabel('Key')
    plt.ylabel('Error Rate')
    
    # Add percentage labels
    for i, row in enumerate(left_home_df.itertuples()):
        plt.text(i, row.errorRate/2, f"{row.errorRate*100:.1f}%", 
                ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}left_home_error_rates.png')
    plt.close()
    
    # 2. Average typing times for left home block keys
    plt.figure(figsize=(12, 6))
    sns.barplot(x='letter', y='avgTime', data=left_home_df)
    plt.title('Average Typing Times for Left Home Block Keys')
    plt.xlabel('Key')
    plt.ylabel('Average Time (ms)')
    
    # Add time labels
    for i, row in enumerate(left_home_df.itertuples()):
        plt.text(i, row.avgTime/2, f"{row.avgTime:.0f}ms", 
                ha='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}left_home_typing_times.png')
    plt.close()
    
    # 3. Error rate vs typing time
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='avgTime', y='errorRate', size='totalCount', 
                   data=left_home_df, alpha=0.7)
    
    # Add key labels
    for _, row in left_home_df.iterrows():
        plt.text(row['avgTime'], row['errorRate'], row['letter'], 
                fontsize=12, ha='center', va='center')
    
    plt.title('Error Rate vs. Typing Time for Left Home Block Keys')
    plt.xlabel('Average Time (ms)')
    plt.ylabel('Error Rate')
    plt.tight_layout()
    plt.savefig(f'{output_dir}left_home_error_vs_time.png')
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
    
    # 1. Error rate comparison
    plt.figure(figsize=(15, 8))
    
    # Prepare data for grouped bar plot
    error_data = []
    for _, row in mirror_df.iterrows():
        error_data.append({'pair': f"{row['left_key']}-{row['right_key']}", 'position': 'Left', 'error_rate': row['left_error_rate']})
        error_data.append({'pair': f"{row['left_key']}-{row['right_key']}", 'position': 'Right', 'error_rate': row['right_error_rate']})
    
    error_df = pd.DataFrame(error_data)
    
    # Create grouped bar plot
    ax = sns.barplot(x='pair', y='error_rate', hue='position', data=error_df)
    plt.title('Error Rate Comparison for Mirror Pairs')
    plt.xlabel('Mirror Pair')
    plt.ylabel('Error Rate')
    plt.xticks(rotation=45)
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height/2, f"{height*100:.1f}%",
                ha='center', color='white', fontweight='bold')
    
    plt.legend(title='Key Position')
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_error_rates.png')
    plt.close()
    
    # 2. Typing time comparison
    plt.figure(figsize=(15, 8))
    
    # Prepare data for grouped bar plot
    time_data = []
    for _, row in mirror_df.iterrows():
        time_data.append({'pair': f"{row['left_key']}-{row['right_key']}", 'position': 'Left', 'avg_time': row['left_avg_time']})
        time_data.append({'pair': f"{row['left_key']}-{row['right_key']}", 'position': 'Right', 'avg_time': row['right_avg_time']})
    
    time_df = pd.DataFrame(time_data)
    
    # Create grouped bar plot
    ax = sns.barplot(x='pair', y='avg_time', hue='position', data=time_df)
    plt.title('Typing Time Comparison for Mirror Pairs')
    plt.xlabel('Mirror Pair')
    plt.ylabel('Average Time (ms)')
    plt.xticks(rotation=45)
    
    # Add time labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height/2, f"{height:.0f}ms",
                ha='center', color='white', fontweight='bold')
    
    plt.legend(title='Key Position')
    plt.tight_layout()
    plt.savefig(f'{output_dir}mirror_pair_typing_times.png')
    plt.close()
    
    # 3. Error rate differences (left - right)
    plt.figure(figsize=(15, 8))
    
    # Sort by absolute difference for better visualization
    sorted_mirror = mirror_df.copy()
    sorted_mirror['abs_error_diff'] = sorted_mirror['error_diff'].abs()
    sorted_mirror = sorted_mirror.sort_values('abs_error_diff', ascending=False)
    
    # Create bar plot with colored bars (red for left > right, blue for right > left)
    bars = plt.bar(
        [f"{row['left_key']}-{row['right_key']}" for _, row in sorted_mirror.iterrows()],
        sorted_mirror['error_diff'],
        color=[('red' if diff > 0 else 'blue') for diff in sorted_mirror['error_diff']]
    )
    
    plt.title('Error Rate Differences for Mirror Pairs (Left - Right)')
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
    plt.savefig(f'{output_dir}mirror_pair_error_differences.png')
    plt.close()
    
    # 4. Typing time differences (left - right)
    plt.figure(figsize=(15, 8))
    
    # Sort by absolute time difference
    sorted_mirror = mirror_df.copy()
    sorted_mirror['abs_time_diff'] = sorted_mirror['time_diff'].abs()
    sorted_mirror = sorted_mirror.sort_values('abs_time_diff', ascending=False)
    
    # Create bar plot with colored bars
    bars = plt.bar(
        [f"{row['left_key']}-{row['right_key']}" for _, row in sorted_mirror.iterrows()],
        sorted_mirror['time_diff'],
        color=[('red' if diff > 0 else 'blue') for diff in sorted_mirror['time_diff']]
    )
    
    plt.title('Typing Time Differences for Mirror Pairs (Left - Right)')
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
    plt.savefig(f'{output_dir}mirror_pair_time_differences.png')
    plt.close()

def main():
    # Get all CSV files in the directory
    csv_path = 'input/raws_Prolific/*.csv'  # Replace with your actual path
    letter_freq_path = 'input/frequencies/letter_frequencies.csv'
    bigram_freq_path = 'input/frequencies/bigram_frequencies.csv'
    
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
    mirror_pair_df = analyze_mirror_pairs(all_letter_stats)
    
    # Print statistics for left home block keys
    print_left_home_stats(left_home_letter_df)
    
    # Print statistics for mirror pairs
    print_mirror_pair_stats(mirror_pair_df)
    
    # Create visualizations
    visualize_left_home_results(left_home_letter_df)
    visualize_mirror_pairs(mirror_pair_df)
    
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
        'mirror_pair_df': mirror_pair_df
    }

if __name__ == "__main__":
    results = main()