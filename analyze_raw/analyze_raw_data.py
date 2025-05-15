import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from collections import defaultdict

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
                'typingTime': typing_time,  # Renamed from typingSpeed to typingTime
                'keydownTime': current['keydownTime']
            })
    
    return pd.DataFrame(processed)
    
def analyze_letters(data):
    """Analyze statistics for individual letters"""
    letter_stats = {}
    
    # Analyze each letter
    for _, row in data.iterrows():
        letter = row['expectedKey']
        typed_letter = row['typedKey']
        
        # Skip if not a string or not a single character or if it's a space
        if not isinstance(letter, str) or len(letter) != 1 or letter == ' ':
            continue
            
        # Initialize stats for this letter if not exists
        if letter not in letter_stats:
            letter_stats[letter] = {
                'letter': letter,
                'totalCount': 0,
                'errorCount': 0,
                'totalTime': 0,  # Renamed from totalSpeed to totalTime
                'timeSamples': [],  # Renamed from speedSamples to timeSamples
                'error_types': defaultdict(int)  # Track mistyped letters
            }
        
        # Update stats
        letter_stats[letter]['totalCount'] += 1
        
        if not row['isCorrect']:
            letter_stats[letter]['errorCount'] += 1
            # Track what was actually typed when error occurred
            letter_stats[letter]['error_types'][typed_letter] += 1
        
        # Only count time if it's reasonable (between 50ms and 2000ms)
        if 50 <= row['typingTime'] <= 2000:  # Using typingTime instead of typingSpeed
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

def analyze_bigrams(data):
    """
    Identify and analyze bigrams in the typing data
    A bigram is defined as two consecutive character keypresses
    Tracks errors and times separately for first and second key
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
            
            # Check if both keys are strings and single characters (not spaces or special keys)
            if (isinstance(current['expectedKey'], str) and isinstance(next_key['expectedKey'], str) and
                len(current['expectedKey']) == 1 and len(next_key['expectedKey']) == 1 and
                current['expectedKey'] != ' ' and next_key['expectedKey'] != ' '):
                
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
                
                # Count errors for first and second key separately
                if not current['isCorrect']:
                    bigram_stats[bigram]['firstKeyErrorCount'] += 1
                    # Track what was actually typed for first key error
                    bigram_stats[bigram]['firstKeyErrorTypes'][current['typedKey']] += 1
                
                if not next_key['isCorrect']:
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
            stats['flankedFirstKeyErrorRate'] = stats['firstKeyErrorCount'] / stats['flankedCount']
            stats['flankedSecondKeyErrorRate'] = stats['secondKeyErrorCount'] / stats['flankedCount']
            
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

def combine_with_frequency_data(stats, freq_data):
    """
    Combine statistics with frequency data if available
    """
    if freq_data is None:
        return stats
        
    # Convert frequency data to dictionary for easier lookup
    freq_dict = {}
    for _, row in freq_data.iterrows():
        key = row.get('bigram', row.get('letter', ''))
        freq_dict[key] = row.get('frequency', 0)
    
    # Add frequency data to stats
    for key, item_stats in stats.items():
        item_stats['frequency'] = freq_dict.get(key, 0)
    
    return stats

def adjust_for_frequency(stats_df, freq_df, value_columns, id_column='letter'):
    """
    Adjust metrics by regressing out frequency effects
    
    Parameters:
    stats_df (DataFrame): DataFrame with typing statistics
    freq_df (DataFrame): DataFrame with frequency information
    value_columns (list): Column names to adjust for frequency
    id_column (str): Column name for the identifier (letter or bigram)
    
    Returns:
    DataFrame: DataFrame with frequency-adjusted metrics
    """
    # Create a copy of the input DataFrame
    adjusted_df = stats_df.copy()
    
    # Ensure we have frequency data
    if freq_df is None:
        print("No frequency data provided, skipping frequency adjustment")
        return adjusted_df
    
    # Merge the frequency data with our statistics
    try:
        # Standardize column names
        freq_df_copy = freq_df.copy()
        if 'frequency' not in freq_df_copy.columns and 'freq' in freq_df_copy.columns:
            freq_df_copy.rename(columns={'freq': 'frequency'}, inplace=True)
        
        if id_column not in freq_df_copy.columns and 'char' in freq_df_copy.columns and id_column == 'letter':
            freq_df_copy.rename(columns={'char': 'letter'}, inplace=True)
            
        # Merge the data
        merged = adjusted_df.merge(freq_df_copy, on=id_column, how='left')
        
        # Handle missing frequency values
        if merged['frequency'].isna().any():
            print(f"Warning: Missing frequency data for some {id_column}s")
            # Fill with minimum frequency observed
            min_freq = merged['frequency'].min()
            merged['frequency'].fillna(min_freq, inplace=True)
        
        # Log-transform frequency (common in psycholinguistic research)
        merged['log_frequency'] = np.log10(merged['frequency'] + 1)  # Add 1 to handle zeros
        
        # Adjust each metric by regressing out log frequency
        for col in value_columns:
            if col in merged.columns:
                # Skip if the column contains mostly zeros or NaNs
                if merged[col].isna().mean() > 0.5 or (merged[col] == 0).mean() > 0.5:
                    continue
                    
                # Create the model: metric ~ log_frequency
                X = sm.add_constant(merged['log_frequency'])
                y = merged[col]
                
                # Fit the model
                model = sm.OLS(y, X).fit()
                
                # Calculate residuals (the frequency-adjusted values)
                merged[f'{col}_freq_adjusted'] = model.resid
                
                # Add the mean back to make the adjusted values more interpretable
                merged[f'{col}_freq_adjusted'] = merged[f'{col}_freq_adjusted'] + np.mean(merged[col])
                
        return merged
    
    except Exception as e:
        print(f"Error adjusting for frequency: {str(e)}")
        import traceback
        traceback.print_exc()
        return adjusted_df

def generate_comparative_analysis(letter_stats, bigram_stats):
    """
    Generate the specific comparative analyses requested:
    - Frequency by accuracy
    - Frequency by time
    - Time by accuracy
    """
    results = {
        'letters': {
            'frequency_by_accuracy': {},
            'frequency_by_time': {},  # Renamed from frequency_by_speed
            'time_by_accuracy': {}    # Renamed from speed_by_accuracy
        },
        'bigrams': {
            'first_key': {
                'frequency_by_accuracy': {},
                'frequency_by_time': {},
                'time_by_accuracy': {}
            },
            'second_key': {
                'frequency_by_accuracy': {},
                'frequency_by_time': {},
                'time_by_accuracy': {}
            }
        }
    }
    
    # Convert to DataFrames for easier analysis
    letters_df = pd.DataFrame([stats for stats in letter_stats.values()])
    bigrams_df = pd.DataFrame([stats for stats in bigram_stats.values()])
    
    # Filter out entries with too few occurrences for statistical significance
    min_count = 10
    letters_filtered = letters_df[letters_df['totalCount'] >= min_count].copy()
    bigrams_filtered = bigrams_df[bigrams_df['totalCount'] >= min_count].copy()
    
    # Letter analysis
    if not letters_filtered.empty:
        # 1. Frequency by Accuracy
        results['letters']['frequency_by_accuracy'] = letters_filtered[['letter', 'totalCount', 'errorRate']].sort_values('errorRate')
        
        # 2. Frequency by Time
        results['letters']['frequency_by_time'] = letters_filtered[['letter', 'totalCount', 'avgTime']].sort_values('avgTime')
        
        # 3. Time by Accuracy
        results['letters']['time_by_accuracy'] = letters_filtered[['letter', 'avgTime', 'errorRate']].sort_values('errorRate')
    
    # Bigram analysis - First Key
    if not bigrams_filtered.empty:
        # 1. Frequency by Accuracy (first key)
        results['bigrams']['first_key']['frequency_by_accuracy'] = (
            bigrams_filtered[['bigram', 'totalCount', 'firstKeyErrorRate']]
            .sort_values('firstKeyErrorRate')
        )
        
        # 2. Frequency by Time (first key)
        results['bigrams']['first_key']['frequency_by_time'] = (
            bigrams_filtered[['bigram', 'totalCount', 'firstKeyAvgTime']]
            .sort_values('firstKeyAvgTime')
        )
        
        # 3. Time by Accuracy (first key)
        results['bigrams']['first_key']['time_by_accuracy'] = (
            bigrams_filtered[['bigram', 'firstKeyAvgTime', 'firstKeyErrorRate']]
            .sort_values('firstKeyErrorRate')
        )
        
        # Bigram analysis - Second Key
        # 1. Frequency by Accuracy (second key)
        results['bigrams']['second_key']['frequency_by_accuracy'] = (
            bigrams_filtered[['bigram', 'totalCount', 'secondKeyErrorRate']]
            .sort_values('secondKeyErrorRate')
        )
        
        # 2. Frequency by Time (second key)
        results['bigrams']['second_key']['frequency_by_time'] = (
            bigrams_filtered[['bigram', 'totalCount', 'secondKeyAvgTime']]
            .sort_values('secondKeyAvgTime')
        )
        
        # 3. Time by Accuracy (second key)
        results['bigrams']['second_key']['time_by_accuracy'] = (
            bigrams_filtered[['bigram', 'secondKeyAvgTime', 'secondKeyErrorRate']]
            .sort_values('secondKeyErrorRate')
        )
    
    # Additional analysis for bigrams: delay between letters
    if not bigrams_filtered.empty:
        same_key_bigrams = bigrams_filtered[bigrams_filtered['is_same_key']]
        different_key_bigrams = bigrams_filtered[~bigrams_filtered['is_same_key']]
        
        # First key stats
        results['bigrams']['first_key']['same_key_stats'] = {
            'count': len(same_key_bigrams),
            'avg_time': same_key_bigrams['firstKeyAvgTime'].mean() if not same_key_bigrams.empty else 0,
            'avg_error_rate': same_key_bigrams['firstKeyErrorRate'].mean() if not same_key_bigrams.empty else 0
        }
        
        results['bigrams']['first_key']['different_key_stats'] = {
            'count': len(different_key_bigrams),
            'avg_time': different_key_bigrams['firstKeyAvgTime'].mean() if not different_key_bigrams.empty else 0,
            'avg_error_rate': different_key_bigrams['firstKeyErrorRate'].mean() if not different_key_bigrams.empty else 0
        }
        
        # Second key stats
        results['bigrams']['second_key']['same_key_stats'] = {
            'count': len(same_key_bigrams),
            'avg_time': same_key_bigrams['secondKeyAvgTime'].mean() if not same_key_bigrams.empty else 0,
            'avg_error_rate': same_key_bigrams['secondKeyErrorRate'].mean() if not same_key_bigrams.empty else 0
        }
        
        results['bigrams']['second_key']['different_key_stats'] = {
            'count': len(different_key_bigrams),
            'avg_time': different_key_bigrams['secondKeyAvgTime'].mean() if not different_key_bigrams.empty else 0,
            'avg_error_rate': different_key_bigrams['secondKeyErrorRate'].mean() if not different_key_bigrams.empty else 0
        }
    
    return results

def visualize_results(letter_stats, bigram_stats, letter_df_adjusted=None, bigram_df_adjusted=None,
                   flanked_bigrams=None, same_key_bigrams=None, diff_key_bigrams=None):
    """
    Create visualizations for letter and bigram statistics
    With separate plots for first and second key metrics
    Includes frequency-adjusted and bigram category-specific visualizations
    """
    # Convert to DataFrames
    letter_df = pd.DataFrame([stats for stats in letter_stats.values()])
    bigram_df = pd.DataFrame([stats for stats in bigram_stats.values()])
    
    # Filter to only include items with enough data points
    min_count = 10
    letter_df_filtered = letter_df[letter_df['totalCount'] > min_count]
    bigram_df_filtered = bigram_df[bigram_df['totalCount'] > min_count]
    
    # Create output directory for figures
    output_dir = 'output/figures/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Letter Analysis Visualizations
    # --------------------------------
    if not letter_df_filtered.empty:
        plt.figure(figsize=(15, 10))
        
        # 1.1 Raw letter analysis
        plt.subplot(2, 2, 1)
        sns.scatterplot(x='avgTime', y='errorRate', size='totalCount', 
                       data=letter_df_filtered, alpha=0.7)
        for _, row in letter_df_filtered.iterrows():
            plt.text(row['avgTime'], row['errorRate'], row['letter'], 
                    fontsize=9, ha='center', va='center')
        plt.title('Letter Error Rate vs. Time')
        plt.xlabel('Average Time (ms)')
        plt.ylabel('Error Rate')
        
        # 1.2 Top 20 most frequent letters
        plt.subplot(2, 2, 2)
        top_letters = letter_df.sort_values('totalCount', ascending=False).head(20)
        sns.barplot(x='letter', y='totalCount', data=top_letters)
        plt.title('Top 20 Most Frequent Letters')
        plt.xlabel('Letter')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 1.3 Top 10 most error-prone letters
        plt.subplot(2, 2, 3)
        error_letters = letter_df_filtered.sort_values('errorRate', ascending=False).head(10)
        sns.barplot(x='letter', y='errorRate', data=error_letters)
        plt.title('Top 10 Most Error-Prone Letters')
        plt.xlabel('Letter')
        plt.ylabel('Error Rate')
        
        # 1.4 Top 10 slowest letters
        plt.subplot(2, 2, 4)
        slow_letters = letter_df_filtered.sort_values('avgTime', ascending=False).head(10)
        sns.barplot(x='letter', y='avgTime', data=slow_letters)
        plt.title('Top 10 Slowest Letters')
        plt.xlabel('Letter')
        plt.ylabel('Average Time (ms)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}letter_analysis.png')
        plt.close()
        
        # 1.5 Frequency-adjusted letter analysis (if available)
        if letter_df_adjusted is not None and 'errorRate_freq_adjusted' in letter_df_adjusted.columns:
            letter_adj_filtered = letter_df_adjusted[letter_df_adjusted['totalCount'] > min_count]
            
            plt.figure(figsize=(15, 10))
            
            # Raw vs Adjusted Error Rates
            plt.subplot(2, 2, 1)
            plt.scatter(letter_adj_filtered['errorRate'], letter_adj_filtered['errorRate_freq_adjusted'])
            for _, row in letter_adj_filtered.iterrows():
                plt.text(row['errorRate'], row['errorRate_freq_adjusted'], row['letter'], 
                        fontsize=9, ha='center', va='center')
            plt.plot([letter_adj_filtered['errorRate'].min(), letter_adj_filtered['errorRate'].max()], 
                     [letter_adj_filtered['errorRate'].min(), letter_adj_filtered['errorRate'].max()], 
                     'k--', alpha=0.3)
            plt.title('Raw vs. Frequency-Adjusted Error Rates')
            plt.xlabel('Raw Error Rate')
            plt.ylabel('Frequency-Adjusted Error Rate')
            
            # Raw vs Adjusted Times
            if 'avgTime_freq_adjusted' in letter_adj_filtered.columns:
                plt.subplot(2, 2, 2)
                plt.scatter(letter_adj_filtered['avgTime'], letter_adj_filtered['avgTime_freq_adjusted'])
                for _, row in letter_adj_filtered.iterrows():
                    plt.text(row['avgTime'], row['avgTime_freq_adjusted'], row['letter'], 
                            fontsize=9, ha='center', va='center')
                plt.plot([letter_adj_filtered['avgTime'].min(), letter_adj_filtered['avgTime'].max()], 
                         [letter_adj_filtered['avgTime'].min(), letter_adj_filtered['avgTime'].max()], 
                         'k--', alpha=0.3)
                plt.title('Raw vs. Frequency-Adjusted Times')
                plt.xlabel('Raw Average Time (ms)')
                plt.ylabel('Frequency-Adjusted Average Time (ms)')
            
            # Top 10 frequency-adjusted error-prone letters
            plt.subplot(2, 2, 3)
            adj_error_letters = letter_adj_filtered.sort_values('errorRate_freq_adjusted', ascending=False).head(10)
            sns.barplot(x='letter', y='errorRate_freq_adjusted', data=adj_error_letters)
            plt.title('Top 10 Most Error-Prone Letters (Freq. Adjusted)')
            plt.xlabel('Letter')
            plt.ylabel('Adjusted Error Rate')
            
            # Top 10 frequency-adjusted slowest letters
            if 'avgTime_freq_adjusted' in letter_adj_filtered.columns:
                plt.subplot(2, 2, 4)
                adj_slow_letters = letter_adj_filtered.sort_values('avgTime_freq_adjusted', ascending=False).head(10)
                sns.barplot(x='letter', y='avgTime_freq_adjusted', data=adj_slow_letters)
                plt.title('Top 10 Slowest Letters (Freq. Adjusted)')
                plt.xlabel('Letter')
                plt.ylabel('Adjusted Average Time (ms)')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}letter_analysis_freq_adjusted.png')
            plt.close()
    
    # 2. Bigram Analysis Visualizations
    # --------------------------------
    if not bigram_df_filtered.empty:
        # 2.1 Basic bigram visualizations
        plt.figure(figsize=(18, 15))
        
        # First key error rate vs time
        plt.subplot(3, 2, 1)
        sns.scatterplot(x='firstKeyAvgTime', y='firstKeyErrorRate', size='totalCount',
                       data=bigram_df_filtered.sample(min(100, len(bigram_df_filtered))), alpha=0.7)
        plt.title('Bigram First Key Error Rate vs. Time')
        plt.xlabel('Average Time (ms)')
        plt.ylabel('Error Rate')
        
        # Second key error rate vs time
        plt.subplot(3, 2, 2)
        sns.scatterplot(x='secondKeyAvgTime', y='secondKeyErrorRate', size='totalCount',
                       data=bigram_df_filtered.sample(min(100, len(bigram_df_filtered))), alpha=0.7)
        plt.title('Bigram Second Key Error Rate vs. Time')
        plt.xlabel('Average Time (ms)')
        plt.ylabel('Error Rate')
        
        # Top 20 most frequent bigrams
        plt.subplot(3, 2, 3)
        top_bigrams = bigram_df.sort_values('totalCount', ascending=False).head(20)
        sns.barplot(x='bigram', y='totalCount', data=top_bigrams)
        plt.title('Top 20 Most Frequent Bigrams')
        plt.xlabel('Bigram')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Compare same-key vs different-key error rates
        plt.subplot(3, 2, 4)
        comparison_data = []
        
        # Prepare comparison data
        for is_same_key in [True, False]:
            subset = bigram_df_filtered[bigram_df_filtered['is_same_key'] == is_same_key]
            if not subset.empty:
                comparison_data.append({
                    'bigram_type': 'Same Key' if is_same_key else 'Different Key',
                    'first_key_error': subset['firstKeyErrorRate'].mean(),
                    'second_key_error': subset['secondKeyErrorRate'].mean(),
                    'first_key_time': subset['firstKeyAvgTime'].mean(),
                    'second_key_time': subset['secondKeyAvgTime'].mean()
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Reshape for easier plotting
        error_melt = pd.melt(comparison_df, 
                             id_vars=['bigram_type'], 
                             value_vars=['first_key_error', 'second_key_error'],
                             var_name='key_position', value_name='error_rate')
        error_melt['key_position'] = error_melt['key_position'].map({
            'first_key_error': 'First Key', 
            'second_key_error': 'Second Key'
        })
        
        sns.barplot(x='bigram_type', y='error_rate', hue='key_position', data=error_melt)
        plt.title('Error Rates: Same-Key vs Different-Key Bigrams')
        plt.xlabel('Bigram Type')
        plt.ylabel('Average Error Rate')
        
        # Compare same-key vs different-key typing times
        plt.subplot(3, 2, 5)
        time_melt = pd.melt(comparison_df, 
                           id_vars=['bigram_type'], 
                           value_vars=['first_key_time', 'second_key_time'],
                           var_name='key_position', value_name='avg_time')
        time_melt['key_position'] = time_melt['key_position'].map({
            'first_key_time': 'First Key', 
            'second_key_time': 'Second Key'
        })
        
        sns.barplot(x='bigram_type', y='avg_time', hue='key_position', data=time_melt)
        plt.title('Typing Times: Same-Key vs Different-Key Bigrams')
        plt.xlabel('Bigram Type')
        plt.ylabel('Average Time (ms)')
        
        # Top error-prone bigrams (combined first and second key)
        plt.subplot(3, 2, 6)
        # Calculate combined error rate
        bigram_df_filtered['combined_error_rate'] = (
            bigram_df_filtered['firstKeyErrorRate'] + bigram_df_filtered['secondKeyErrorRate']
        ) / 2
        
        top_error_bigrams = bigram_df_filtered.sort_values('combined_error_rate', ascending=False).head(10)
        sns.barplot(x='bigram', y='combined_error_rate', data=top_error_bigrams)
        plt.title('Top 10 Most Error-Prone Bigrams (Combined)')
        plt.xlabel('Bigram')
        plt.ylabel('Average Error Rate')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}bigram_analysis.png')
        plt.close()
        
        # 2.2 Flanked Bigram Analysis
        if flanked_bigrams is not None and not flanked_bigrams.empty:
            flanked_filtered = flanked_bigrams[flanked_bigrams['flankedCount'] > min_count]
            
            if not flanked_filtered.empty:
                plt.figure(figsize=(15, 10))
                
                # Compare flanked vs all bigrams error rates
                plt.subplot(2, 2, 1)
                # Create comparison dataset
                comparison_data = []
                
                # All bigrams
                if not bigram_df_filtered.empty:
                    comparison_data.append({
                        'bigram_type': 'All Bigrams',
                        'first_key_error': bigram_df_filtered['firstKeyErrorRate'].mean(),
                        'second_key_error': bigram_df_filtered['secondKeyErrorRate'].mean()
                    })
                
                # Flanked bigrams
                comparison_data.append({
                    'bigram_type': 'Flanked Bigrams',
                    'first_key_error': flanked_filtered['firstKeyErrorRate'].mean(),
                    'second_key_error': flanked_filtered['secondKeyErrorRate'].mean()
                })
                
                comp_df = pd.DataFrame(comparison_data)
                error_melt = pd.melt(comp_df, 
                                     id_vars=['bigram_type'], 
                                     value_vars=['first_key_error', 'second_key_error'],
                                     var_name='key_position', value_name='error_rate')
                error_melt['key_position'] = error_melt['key_position'].map({
                    'first_key_error': 'First Key', 
                    'second_key_error': 'Second Key'
                })
                
                sns.barplot(x='bigram_type', y='error_rate', hue='key_position', data=error_melt)
                plt.title('Error Rates: Flanked vs All Bigrams')
                plt.xlabel('Bigram Type')
                plt.ylabel('Average Error Rate')
                
                # Top error-prone flanked bigrams - first key
                plt.subplot(2, 2, 2)
                top_flanked_errors = flanked_filtered.sort_values('firstKeyErrorRate', ascending=False).head(10)
                sns.barplot(x='bigram', y='firstKeyErrorRate', data=top_flanked_errors)
                plt.title('Top 10 Most Error-Prone Flanked Bigrams (First Key)')
                plt.xlabel('Bigram')
                plt.ylabel('First Key Error Rate')
                plt.xticks(rotation=45)
                
                # Top error-prone flanked bigrams - second key
                plt.subplot(2, 2, 3)
                top_flanked_errors = flanked_filtered.sort_values('secondKeyErrorRate', ascending=False).head(10)
                sns.barplot(x='bigram', y='secondKeyErrorRate', data=top_flanked_errors)
                plt.title('Top 10 Most Error-Prone Flanked Bigrams (Second Key)')
                plt.xlabel('Bigram')
                plt.ylabel('Second Key Error Rate')
                plt.xticks(rotation=45)
                
                # Most common mistypes for flanked bigrams
                plt.subplot(2, 2, 4)
                # Create a dataset of most common mistypes
                mistype_data = []
                for _, row in flanked_filtered.sort_values('secondKeyErrorRate', ascending=False).head(5).iterrows():
                    if row['most_common_second_mistype']:
                        mistype_data.append({
                            'bigram': row['bigram'],
                            'expected': row['bigram'][1],  # Second character
                            'actual': row['most_common_second_mistype'],
                            'count': row['most_common_second_mistype_count']
                        })
                
                if mistype_data:
                    mistype_df = pd.DataFrame(mistype_data)
                    sns.barplot(x='bigram', y='count', data=mistype_df)
                    plt.title('Most Common Mistypes for Flanked Bigrams (Second Key)')
                    plt.xlabel('Bigram')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    
                    # Add text labels showing expected→actual
                    for i, row in enumerate(mistype_data):
                        plt.text(i, row['count']/2, f"{row['expected']}→{row['actual']}", 
                                ha='center', color='white', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}flanked_bigram_analysis.png')
                plt.close()
        
        # 2.3 Frequency-adjusted bigram analysis
        if (bigram_df_adjusted is not None and 
            'firstKeyErrorRate_freq_adjusted' in bigram_df_adjusted.columns):
            
            bigram_adj_filtered = bigram_df_adjusted[bigram_df_adjusted['totalCount'] > min_count]
            
            if not bigram_adj_filtered.empty:
                plt.figure(figsize=(15, 10))
                
                # Raw vs Adjusted First Key Error Rates
                plt.subplot(2, 2, 1)
                plt.scatter(bigram_adj_filtered['firstKeyErrorRate'], 
                           bigram_adj_filtered['firstKeyErrorRate_freq_adjusted'],
                           alpha=0.4)
                plt.plot([bigram_adj_filtered['firstKeyErrorRate'].min(), bigram_adj_filtered['firstKeyErrorRate'].max()], 
                         [bigram_adj_filtered['firstKeyErrorRate'].min(), bigram_adj_filtered['firstKeyErrorRate'].max()], 
                         'k--', alpha=0.3)
                plt.title('Raw vs. Frequency-Adjusted First Key Error Rates')
                plt.xlabel('Raw First Key Error Rate')
                plt.ylabel('Frequency-Adjusted First Key Error Rate')
                
                # Raw vs Adjusted Second Key Error Rates
                plt.subplot(2, 2, 2)
                plt.scatter(bigram_adj_filtered['secondKeyErrorRate'], 
                           bigram_adj_filtered['secondKeyErrorRate_freq_adjusted'],
                           alpha=0.4)
                plt.plot([bigram_adj_filtered['secondKeyErrorRate'].min(), bigram_adj_filtered['secondKeyErrorRate'].max()], 
                         [bigram_adj_filtered['secondKeyErrorRate'].min(), bigram_adj_filtered['secondKeyErrorRate'].max()], 
                         'k--', alpha=0.3)
                plt.title('Raw vs. Frequency-Adjusted Second Key Error Rates')
                plt.xlabel('Raw Second Key Error Rate')
                plt.ylabel('Frequency-Adjusted Second Key Error Rate')
                
                # Top 10 frequency-adjusted error-prone bigrams (first key)
                plt.subplot(2, 2, 3)
                top_adj_errors = bigram_adj_filtered.sort_values('firstKeyErrorRate_freq_adjusted', ascending=False).head(10)
                sns.barplot(x='bigram', y='firstKeyErrorRate_freq_adjusted', data=top_adj_errors)
                plt.title('Top 10 Most Error-Prone Bigrams - First Key (Freq. Adjusted)')
                plt.xlabel('Bigram')
                plt.ylabel('Adjusted First Key Error Rate')
                plt.xticks(rotation=45)
                
                # Top 10 frequency-adjusted error-prone bigrams (second key)
                plt.subplot(2, 2, 4)
                top_adj_errors = bigram_adj_filtered.sort_values('secondKeyErrorRate_freq_adjusted', ascending=False).head(10)
                sns.barplot(x='bigram', y='secondKeyErrorRate_freq_adjusted', data=top_adj_errors)
                plt.title('Top 10 Most Error-Prone Bigrams - Second Key (Freq. Adjusted)')
                plt.xlabel('Bigram')
                plt.ylabel('Adjusted Second Key Error Rate')
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}bigram_analysis_freq_adjusted.png')
                plt.close()
        
        # 2.4 Most common mistypes visualization
        if 'most_common_first_mistype' in bigram_df_filtered.columns:
            plt.figure(figsize=(15, 10))
            
            # Top 5 most error-prone bigrams and their common mistypes
            top_error_bigrams_first = bigram_df_filtered.sort_values('firstKeyErrorRate', ascending=False).head(5)
            top_error_bigrams_second = bigram_df_filtered.sort_values('secondKeyErrorRate', ascending=False).head(5)
            
            # First key mistypes
            plt.subplot(2, 1, 1)
            first_mistype_data = []
            for _, row in top_error_bigrams_first.iterrows():
                if row['most_common_first_mistype']:
                    first_mistype_data.append({
                        'bigram': row['bigram'],
                        'expected': row['bigram'][0],  # First character
                        'actual': row['most_common_first_mistype'],
                        'count': row['most_common_first_mistype_count'],
                        'error_rate': row['firstKeyErrorRate']
                    })
            
            if first_mistype_data:
                first_mistype_df = pd.DataFrame(first_mistype_data)
                bars = sns.barplot(x='bigram', y='count', data=first_mistype_df)
                plt.title('Most Common Mistypes for First Key in Bigrams')
                plt.xlabel('Bigram')
                plt.ylabel('Count')
                
                # Add text labels showing expected→actual and error rate
                for i, (_, row) in enumerate(first_mistype_df.iterrows()):
                    plt.text(i, row['count']/2, f"{row['expected']}→{row['actual']}", 
                            ha='center', color='white', fontweight='bold')
                    plt.text(i, row['count'] + 0.5, f"Error Rate: {row['error_rate']:.2f}", 
                            ha='center')
            
            # Second key mistypes
            plt.subplot(2, 1, 2)
            second_mistype_data = []
            for _, row in top_error_bigrams_second.iterrows():
                if row['most_common_second_mistype']:
                    second_mistype_data.append({
                        'bigram': row['bigram'],
                        'expected': row['bigram'][1],  # Second character
                        'actual': row['most_common_second_mistype'],
                        'count': row['most_common_second_mistype_count'],
                        'error_rate': row['secondKeyErrorRate']
                    })
            
            if second_mistype_data:
                second_mistype_df = pd.DataFrame(second_mistype_data)
                bars = sns.barplot(x='bigram', y='count', data=second_mistype_df)
                plt.title('Most Common Mistypes for Second Key in Bigrams')
                plt.xlabel('Bigram')
                plt.ylabel('Count')
                
                # Add text labels showing expected→actual and error rate
                for i, (_, row) in enumerate(second_mistype_df.iterrows()):
                    plt.text(i, row['count']/2, f"{row['expected']}→{row['actual']}", 
                            ha='center', color='white', fontweight='bold')
                    plt.text(i, row['count'] + 0.5, f"Error Rate: {row['error_rate']:.2f}", 
                            ha='center')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}common_mistypes.png')
            plt.close()
    
    return

def main():
    # Get all CSV files in the directory
    csv_path = 'input/raws_Prolific/*.csv'  # Replace with your actual path
    letter_freq_path = 'input/frequencies/letter_frequencies.csv'  # Path to letter frequency file
    bigram_freq_path = 'input/frequencies/bigram_frequencies.csv'  # Path to bigram frequency file
    
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
    
    # Analyze individual letters (excluding space)
    letter_stats = analyze_letters(processed_data)
    
    # Analyze bigrams (excluding space)
    bigram_stats = analyze_bigrams(processed_data)
    
    # Convert to DataFrames for analysis
    letter_df = pd.DataFrame([stats for stats in letter_stats.values()])
    bigram_df = pd.DataFrame([stats for stats in bigram_stats.values()])
    
    # Create frequency-adjusted metrics if frequency data is available
    if letter_frequencies is not None:
        letter_cols_to_adjust = ['errorRate', 'avgTime', 'medianTime']
        letter_df_adjusted = adjust_for_frequency(letter_df, letter_frequencies, letter_cols_to_adjust, 'letter')
        print("Created frequency-adjusted letter metrics")
    else:
        letter_df_adjusted = letter_df.copy()
        print("No letter frequency data available, skipping adjustment")
    
    if bigram_frequencies is not None:
        bigram_cols_to_adjust = [
            'firstKeyErrorRate', 'secondKeyErrorRate', 
            'firstKeyAvgTime', 'secondKeyAvgTime',
            'firstKeyMedianTime', 'secondKeyMedianTime'
        ]
        bigram_df_adjusted = adjust_for_frequency(bigram_df, bigram_frequencies, bigram_cols_to_adjust, 'bigram')
        print("Created frequency-adjusted bigram metrics")
    else:
        bigram_df_adjusted = bigram_df.copy()
        print("No bigram frequency data available, skipping adjustment")
    
    # Create separate DataFrames for specific analysis types
    # 1. Letters (all)
    print("\nLetter statistics:")
    print(f"Analyzed {len(letter_df)} unique letters")
    
    # 2. Bigrams - all
    print("\nBigram statistics (all):")
    print(f"Analyzed {len(bigram_df)} unique bigrams")
    
    # 3. Same-key bigrams
    same_key_bigrams = bigram_df[bigram_df['is_same_key']].copy()
    print(f"\nSame-key bigram statistics:")
    print(f"Analyzed {len(same_key_bigrams)} unique same-key bigrams")
    
    # 4. Different-key bigrams
    diff_key_bigrams = bigram_df[~bigram_df['is_same_key']].copy()
    print(f"\nDifferent-key bigram statistics:")
    print(f"Analyzed {len(diff_key_bigrams)} unique different-key bigrams")
    
    # 5. Flanked bigrams (surrounded by spaces)
    flanked_bigrams = bigram_df[bigram_df['flankedCount'] > 0].copy()
    print(f"\nFlanked bigram statistics:")
    print(f"Analyzed {len(flanked_bigrams)} unique flanked bigrams")
    
    # Print detailed statistics for each analysis type
    if not letter_df.empty:
        # Sort by error rate and print top errors
        top_letter_errors = letter_df.sort_values('errorRate', ascending=False).head(10)
        print("\nTop 10 most error-prone letters:")
        print(top_letter_errors[['letter', 'totalCount', 'errorRate', 'avgTime', 'most_common_mistype', 'most_common_mistype_count']])
        
        # Sort by time and print slowest letters
        slowest_letters = letter_df.sort_values('avgTime', ascending=False).head(10)
        print("\nTop 10 slowest letters:")
        print(slowest_letters[['letter', 'totalCount', 'errorRate', 'avgTime']])
        
        # Print frequency-adjusted error rates if available
        if 'errorRate_freq_adjusted' in letter_df_adjusted.columns:
            top_adj_errors = letter_df_adjusted.sort_values('errorRate_freq_adjusted', ascending=False).head(10)
            print("\nTop 10 most error-prone letters (frequency adjusted):")
            print(top_adj_errors[['letter', 'totalCount', 'errorRate', 'errorRate_freq_adjusted', 'most_common_mistype']])
    
    # Print bigram statistics (filtering for sufficient data points)
    min_count = 10
    if not bigram_df.empty:
        bigram_df_filtered = bigram_df[bigram_df['totalCount'] >= min_count]
        
        # First key error rates by bigram type
        print("\nFirst key bigram error rates by type:")
        print(bigram_df_filtered.groupby('is_same_key')['firstKeyErrorRate'].mean())
        
        # Second key error rates by bigram type
        print("\nSecond key bigram error rates by type:")
        print(bigram_df_filtered.groupby('is_same_key')['secondKeyErrorRate'].mean())
        
        # First key typing times by bigram type
        print("\nFirst key bigram typing times by type:")
        print(bigram_df_filtered.groupby('is_same_key')['firstKeyAvgTime'].mean())
        
        # Second key typing times by bigram type
        print("\nSecond key bigram typing times by type:")
        print(bigram_df_filtered.groupby('is_same_key')['secondKeyAvgTime'].mean())
    
    # Different-key bigram analysis
    if not diff_key_bigrams.empty:
        diff_key_filtered = diff_key_bigrams[diff_key_bigrams['totalCount'] >= min_count]
        
        # Top error bigrams - first key
        top_diff_errors_first = diff_key_filtered.sort_values('firstKeyErrorRate', ascending=False).head(10)
        print("\nTop 10 most error-prone different-key bigrams (first key):")
        print(top_diff_errors_first[['bigram', 'totalCount', 'firstKeyErrorRate', 'firstKeyAvgTime', 
                                     'most_common_first_mistype', 'most_common_first_mistype_count']])
        
        # Top error bigrams - second key
        top_diff_errors_second = diff_key_filtered.sort_values('secondKeyErrorRate', ascending=False).head(10)
        print("\nTop 10 most error-prone different-key bigrams (second key):")
        print(top_diff_errors_second[['bigram', 'totalCount', 'secondKeyErrorRate', 'secondKeyAvgTime',
                                      'most_common_second_mistype', 'most_common_second_mistype_count']])
    
    # Same-key bigram analysis
    if not same_key_bigrams.empty:
        same_key_filtered = same_key_bigrams[same_key_bigrams['totalCount'] >= min_count]
        
        # Top error bigrams - first key
        top_same_errors_first = same_key_filtered.sort_values('firstKeyErrorRate', ascending=False).head(10)
        print("\nTop 10 most error-prone same-key bigrams (first key):")
        print(top_same_errors_first[['bigram', 'totalCount', 'firstKeyErrorRate', 'firstKeyAvgTime',
                                     'most_common_first_mistype', 'most_common_first_mistype_count']])
        
        # Top error bigrams - second key
        top_same_errors_second = same_key_filtered.sort_values('secondKeyErrorRate', ascending=False).head(10)
        print("\nTop 10 most error-prone same-key bigrams (second key):")
        print(top_same_errors_second[['bigram', 'totalCount', 'secondKeyErrorRate', 'secondKeyAvgTime',
                                      'most_common_second_mistype', 'most_common_second_mistype_count']])
    
    # Flanked bigram analysis
    if not flanked_bigrams.empty:
        flanked_filtered = flanked_bigrams[flanked_bigrams['flankedCount'] >= min_count]
        
        print("\nFlanked bigram statistics (bigrams surrounded by spaces):")
        print(f"Found {len(flanked_filtered)} bigrams with sufficient data")
        
        if not flanked_filtered.empty:
            # Compare error rates for flanked vs all bigrams
            flanked_first_err = flanked_filtered['firstKeyErrorRate'].mean()
            flanked_second_err = flanked_filtered['secondKeyErrorRate'].mean()
            all_first_err = bigram_df_filtered['firstKeyErrorRate'].mean() if not bigram_df_filtered.empty else 0
            all_second_err = bigram_df_filtered['secondKeyErrorRate'].mean() if not bigram_df_filtered.empty else 0
            
            print("\nError rate comparison (flanked vs all bigrams):")
            print(f"First key: Flanked = {flanked_first_err:.4f}, All = {all_first_err:.4f}")
            print(f"Second key: Flanked = {flanked_second_err:.4f}, All = {all_second_err:.4f}")
            
            # Compare typing times
            flanked_first_time = flanked_filtered['firstKeyAvgTime'].mean()
            flanked_second_time = flanked_filtered['secondKeyAvgTime'].mean()
            all_first_time = bigram_df_filtered['firstKeyAvgTime'].mean() if not bigram_df_filtered.empty else 0
            all_second_time = bigram_df_filtered['secondKeyAvgTime'].mean() if not bigram_df_filtered.empty else 0
            
            print("\nTyping time comparison (flanked vs all bigrams):")
            print(f"First key: Flanked = {flanked_first_time:.2f}ms, All = {all_first_time:.2f}ms")
            print(f"Second key: Flanked = {flanked_second_time:.2f}ms, All = {all_second_time:.2f}ms")
    
    # Create visualizations
    visualize_results(letter_stats, bigram_stats, letter_df_adjusted, bigram_df_adjusted, 
                     flanked_bigrams, same_key_bigrams, diff_key_bigrams)
    
    # Save results to CSV files for further analysis
    output_dir = 'output/'
    os.makedirs(output_dir, exist_ok=True)
    
    letter_df.to_csv(f"{output_dir}letter_analysis.csv", index=False)
    bigram_df.to_csv(f"{output_dir}bigram_analysis.csv", index=False)
    
    if 'errorRate_freq_adjusted' in letter_df_adjusted.columns:
        letter_df_adjusted.to_csv(f"{output_dir}letter_analysis_freq_adjusted.csv", index=False)
    
    if 'firstKeyErrorRate_freq_adjusted' in bigram_df_adjusted.columns:
        bigram_df_adjusted.to_csv(f"{output_dir}bigram_analysis_freq_adjusted.csv", index=False)
    
    flanked_bigrams.to_csv(f"{output_dir}flanked_bigram_analysis.csv", index=False)
    same_key_bigrams.to_csv(f"{output_dir}same_key_bigram_analysis.csv", index=False)
    diff_key_bigrams.to_csv(f"{output_dir}diff_key_bigram_analysis.csv", index=False)
    
    print(f"\nResults saved to {output_dir}")
    
    return {
        'letter_stats': letter_stats,
        'bigram_stats': bigram_stats,
        'letter_df': letter_df,
        'bigram_df': bigram_df,
        'letter_df_adjusted': letter_df_adjusted,
        'bigram_df_adjusted': bigram_df_adjusted,
        'flanked_bigrams': flanked_bigrams,
        'same_key_bigrams': same_key_bigrams,
        'diff_key_bigrams': diff_key_bigrams
    }

if __name__ == "__main__":
    results = main()