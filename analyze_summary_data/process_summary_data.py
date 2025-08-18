""" Process experiment data -- See README.md """

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from typing import Dict, Any, List, Tuple, Set, Optional
import logging
import argparse

#######################################
# Logging Configuration
#######################################
def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration from config file."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(level=level, format=format_str)
    logger = logging.getLogger(__name__)
    
    # Optionally log to file
    if 'file' in log_config:
        handler = logging.FileHandler(log_config['file'])
        handler.setFormatter(logging.Formatter(format_str))
        logger.addHandler(handler)
    
    return logger

#######################################
# Configuration Loading
#######################################
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with defaults."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Add default values if not specified
        defaults = {
            'logging': {
                'level': 'WARNING',  # Less verbose by default
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            },
            'visualization': {
                'style': 'default',
                'figsize': [10, 6],
                'dpi': 300,
                'colors': {
                    'primary': '#1f77b4',
                    'secondary': '#ff7f0e'
                }
            }
        }
        
        # Update config with defaults for missing values
        for section, values in defaults.items():
            if section not in config:
                config[section] = values
            elif isinstance(values, dict):
                for key, value in values.items():
                    if key not in config[section]:
                        config[section][key] = value
        
        return config
            
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        raise

#######################################
# Data Loading Functions
#######################################
def load_and_combine_data(input_folder: str, output_tables_folder: str, 
                         logger: logging.Logger, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Load and combine data from multiple CSV files in a folder of subfolders.
    Each file becomes a unique participant.
    """
    logger.info(f"Loading data from {input_folder}")
    
    if not os.path.exists(input_folder):
        logger.error(f"Input folder does not exist: {input_folder}")
        return None
    
    dataframes = []
    csv_files_found = []
    used_user_ids = set()
    processed_files = []  # Track which files we actually process
    
    # Walk through directory and collect all CSV files
    for root, dirs, files in os.walk(input_folder):
        if verbose:
            logger.debug(f"Checking directory: {root}")
            logger.debug(f"Files found: {files}")
        
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                csv_files_found.append(file_path)
                
                try:
                    df = pd.read_csv(file_path)
                    if verbose:
                        logger.debug(f"Successfully read {filename}, shape: {df.shape}")
                    
                    if len(df) == 0:
                        logger.warning(f"{filename} is empty, skipping")
                        continue
                    
                    # Generate user ID from filename (one file = one participant)
                    user_id = _generate_unique_user_id(filename, used_user_ids)
                    used_user_ids.add(user_id)
                    df['user_id'] = user_id
                    df['filename'] = filename
                    processed_files.append(filename)
                    
                    # Add the subfolder name
                    subfolder = os.path.relpath(root, input_folder)
                    df['group_id'] = subfolder if subfolder != '.' else ''
                    
                    # Filter out intro trials
                    df_filtered = _filter_intro_trials(df, logger, verbose)
                    
                    if len(df_filtered) > 0:
                        dataframes.append(df_filtered)
                        if verbose:
                            logger.debug(f"Added {filename} as user {user_id}")
                    else:
                        logger.warning(f"{filename} has no rows after filtering")
                        
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {str(e)}")
                    continue
    
    # Enhanced logging to help debug the file count issue
    logger.info(f"Total CSV files found: {len(csv_files_found)}")
    logger.info(f"Successfully processed files: {len(processed_files)}")
    logger.info(f"Unique user IDs created: {len(used_user_ids)}")
    
    # If there's a mismatch, log the details
    if len(processed_files) != len(used_user_ids):
        logger.warning(f"Mismatch: {len(processed_files)} files processed but {len(used_user_ids)} unique user IDs")
        logger.warning("This suggests duplicate filenames in your directory structure")
        
        # Log all processed filenames
        if verbose:
            logger.debug("All processed filenames:")
            for i, fname in enumerate(sorted(processed_files), 1):
                logger.debug(f"  {i}: {fname}")
    
    if not csv_files_found:
        logger.error(f"No CSV files found in {input_folder}")
        return None
    
    if not dataframes:
        logger.error("No valid dataframes to combine (all files were empty or failed to load)")
        return None
    
    # Combine the dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Successfully combined data from {len(dataframes)} files")
    logger.info(f"Combined dataframe shape: {combined_df.shape}")
    logger.info(f"Unique participants: {len(combined_df['user_id'].unique())}")

    if verbose:
        logger.debug(f"Combined dataframe columns: {list(combined_df.columns)}")
        _display_sample_data(combined_df, logger)

    # Save the combined DataFrame to a CSV file
    output_file = os.path.join(output_tables_folder, 'original_combined_data.csv')
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Combined data saved to {output_file}")

    return combined_df

def _generate_unique_user_id(filename: str, used_ids: set) -> str:
    """Generate user ID from filename - each file represents exactly one participant."""
    # Simply use the filename without extension as the user ID
    user_id = filename.replace('.csv', '')
    
    # If this filename has already been seen, that's a data issue that should be logged
    if user_id in used_ids:
        logger = logging.getLogger(__name__)
        logger.warning(f"Duplicate filename detected: {filename}. Each file should represent a unique participant.")
        logger.warning(f"This may indicate duplicate files in your directory structure.")
    
    return user_id

def _filter_intro_trials(df: pd.DataFrame, logger: logging.Logger, verbose: bool) -> pd.DataFrame:
    """Filter out intro trials from the dataframe."""
    possible_trial_columns = ['trialId', 'trial_id', 'trial', 'Trial', 'TrialId']
    trial_column = None
    
    for col in possible_trial_columns:
        if col in df.columns:
            trial_column = col
            break
    
    if trial_column:
        initial_rows = len(df)
        df_filtered = df[~df[trial_column].str.contains("intro-trial", na=False)]
        if verbose:
            logger.debug(f"Filtered out intro trials using column '{trial_column}': {initial_rows} -> {len(df_filtered)} rows")
    else:
        df_filtered = df.copy()
        if verbose:
            logger.debug("No trial ID column found - keeping all rows")
    
    return df_filtered

def _display_sample_data(df: pd.DataFrame, logger: logging.Logger, nlines: int = 5) -> None:
    """Display sample data information."""
    print_headers = ['group_id', 'user_id']
    for col in ['trialId', 'trial_id', 'sliderValue', 'chosenBigram', 'unchosenBigram']:
        if col in df.columns:
            print_headers.append(col)
    
    logger.debug("Sample data:")
    with pd.option_context('display.max_rows', nlines):
        logger.debug(f"\n{df[print_headers].iloc[:nlines]}")

def load_easy_choice_pairs(file_path: str, logger: logging.Logger) -> List[Tuple[str, str]]:
    """Load easy choice bigram pairs from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        
        if 'good_choice' not in df.columns or 'bad_choice' not in df.columns:
            raise ValueError("CSV file must contain 'good_choice' and 'bad_choice' columns")
        
        easy_choice_pairs = list(df[['good_choice', 'bad_choice']].itertuples(index=False, name=None))
        logger.info(f"Loaded {len(easy_choice_pairs)} bigram pairs from {file_path}")
        
        return easy_choice_pairs
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading easy choice pairs: {str(e)}")
        return []

#######################################
# Data Processing Functions
#######################################
def create_standardized_bigram_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """Create standardized bigram pair representation as tuples."""
    # Only process rows with valid bigram data
    valid_data = data.dropna(subset=['chosenBigram', 'unchosenBigram']).copy()
    
    valid_data['std_bigram_pair'] = valid_data.apply(
        lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
    
    return valid_data

def identify_probable_improbable_choices(data: pd.DataFrame, easy_choice_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Mark probable/improbable choices in the data."""
    data = data.copy()
    
    # Create dictionaries for quick lookup
    probable_pairs = {(pair[0], pair[1]): True for pair in easy_choice_pairs}
    improbable_pairs = {(pair[1], pair[0]): True for pair in easy_choice_pairs}
    
    # Apply to each row
    data['is_probable'] = data.apply(
        lambda row: probable_pairs.get((row['chosenBigram'], row['unchosenBigram']), False), axis=1)
    data['is_improbable'] = data.apply(
        lambda row: improbable_pairs.get((row['chosenBigram'], row['unchosenBigram']), False), axis=1)
    
    return data

def calculate_choice_consistency(data_with_choices: pd.DataFrame) -> pd.DataFrame:
    """Determine if choices are consistent for each group - no fallbacks."""
    logger = logging.getLogger(__name__)
    
    # Clean up any duplicate columns first
    if len(data_with_choices.columns) != len(set(data_with_choices.columns)):
        logger.warning(f"Found duplicate columns: {data_with_choices.columns.tolist()}")
        data_with_choices = data_with_choices.loc[:, ~data_with_choices.columns.duplicated()]
        logger.info(f"After removing duplicates: {list(data_with_choices.columns)}")
    
    # Only work with complete data
    required_columns = ['user_id', 'std_bigram_pair', 'chosenBigram', 'unchosenBigram', 'sliderValue']
    valid_data = data_with_choices.dropna(subset=required_columns).copy()
    
    logger.info(f"Removed {len(data_with_choices) - len(valid_data)} rows with missing required data")
    
    if len(valid_data) == 0:
        logger.error("No valid data remaining")
        return pd.DataFrame()
    
    # Check if we already have the consistency columns (from previous processing)
    has_consistency = 'is_consistent' in valid_data.columns
    has_group_size = 'group_size' in valid_data.columns
    
    if not has_group_size:
        # Calculate group sizes
        group_sizes = valid_data.groupby(['user_id', 'std_bigram_pair']).size().reset_index()
        group_sizes.columns = ['user_id', 'std_bigram_pair', 'group_size']
        
        # Merge group sizes
        valid_data = valid_data.merge(group_sizes, on=['user_id', 'std_bigram_pair'], how='left')
    
    if not has_consistency:
        # Calculate consistency
        consistency_info = valid_data.groupby(['user_id', 'std_bigram_pair'])['chosenBigram'].apply(
            lambda x: len(set(x)) == 1 if len(x) > 1 else None
        ).reset_index()
        consistency_info.columns = ['user_id', 'std_bigram_pair', 'is_consistent']
        
        # Merge consistency info
        valid_data = valid_data.merge(consistency_info, on=['user_id', 'std_bigram_pair'], how='left')
    
    # Extract bigram1 and bigram2 from std_bigram_pair if not already present
    if 'bigram1' not in valid_data.columns or 'bigram2' not in valid_data.columns:
        bigram_pairs_df = pd.DataFrame(valid_data['std_bigram_pair'].tolist(), 
                                     index=valid_data.index, 
                                     columns=['bigram1', 'bigram2'])
        valid_data = pd.concat([valid_data, bigram_pairs_df], axis=1)
    
    # Calculate bigram times (only for rows with valid time data and if columns don't exist)
    def safe_bigram_time(row, bigram_num):
        try:
            target_bigram = row[f'bigram{bigram_num}']
            chosen_bigram = row['chosenBigram']
            
            # Handle case where pandas returns a Series instead of scalar
            if hasattr(target_bigram, 'iloc'):
                target_bigram = target_bigram.iloc[0]
            if hasattr(chosen_bigram, 'iloc'):
                chosen_bigram = chosen_bigram.iloc[0]
            
            if chosen_bigram == target_bigram:
                chosen_time = row.get('chosenBigramTime', np.nan)
                return chosen_time.iloc[0] if hasattr(chosen_time, 'iloc') else chosen_time
            else:
                unchosen_time = row.get('unchosenBigramTime', np.nan)
                return unchosen_time.iloc[0] if hasattr(unchosen_time, 'iloc') else unchosen_time
        except:
            return np.nan
    
    # Only calculate bigram times if we don't already have these columns
    if 'bigram1_time' not in valid_data.columns and 'chosenBigramTime' in valid_data.columns:
        valid_data['bigram1_time'] = valid_data.apply(lambda row: safe_bigram_time(row, 1), axis=1)
    if 'bigram2_time' not in valid_data.columns and 'unchosenBigramTime' in valid_data.columns:
        valid_data['bigram2_time'] = valid_data.apply(lambda row: safe_bigram_time(row, 2), axis=1)
    
    # Add abs_sliderValue if not present
    if 'abs_sliderValue' not in valid_data.columns:
        valid_data['abs_sliderValue'] = valid_data['sliderValue'].abs()
    
    # Rename columns to standardized names (only if they exist and target doesn't)
    rename_mapping = {
        'chosenBigram': 'chosen_bigram',
        'unchosenBigram': 'unchosen_bigram', 
        'chosenBigramTime': 'chosen_bigram_time',
        'unchosenBigramTime': 'unchosen_bigram_time',
        'chosenBigramCorrect': 'chosen_bigram_correct',
        'unchosenBigramCorrect': 'unchosen_bigram_correct',
        'std_bigram_pair': 'bigram_pair'
    }
    
    # Only rename columns that exist and where target doesn't already exist
    columns_to_rename = {}
    for old_name, new_name in rename_mapping.items():
        if old_name in valid_data.columns and new_name not in valid_data.columns:
            columns_to_rename[old_name] = new_name
    
    if columns_to_rename:
        valid_data = valid_data.rename(columns=columns_to_rename)
    
    # Final cleanup of any remaining duplicates
    if len(valid_data.columns) != len(set(valid_data.columns)):
        logger.warning("Final duplicate cleanup")
        valid_data = valid_data.loc[:, ~valid_data.columns.duplicated()]
        logger.info(f"Final columns: {list(valid_data.columns)}")
    
    return valid_data

def calculate_user_statistics(bigram_data: pd.DataFrame, easy_choice_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Calculate user statistics - only from actual data, no fallbacks."""
    users = bigram_data['user_id'].unique()
    user_stats = pd.DataFrame({'user_id': users})
    
    # Basic counts
    user_stats['total_choices'] = user_stats['user_id'].map(bigram_data['user_id'].value_counts())
    
    # Filter data for specific choice types
    consistent_choices = bigram_data[(bigram_data['is_consistent'] == True) & (bigram_data['group_size'] > 1)]
    inconsistent_choices = bigram_data[(bigram_data['is_consistent'] == False) & (bigram_data['group_size'] > 1)]
    probable_choices = bigram_data[bigram_data['is_probable'] == True]
    improbable_choices = bigram_data[bigram_data['is_improbable'] == True]
    
    # Count occurrences
    user_stats['consistent_choices'] = user_stats['user_id'].map(consistent_choices['user_id'].value_counts())
    user_stats['inconsistent_choices'] = user_stats['user_id'].map(inconsistent_choices['user_id'].value_counts())
    user_stats['probable_choices'] = user_stats['user_id'].map(probable_choices['user_id'].value_counts())
    user_stats['improbable_choices'] = user_stats['user_id'].map(improbable_choices['user_id'].value_counts())
    
    # Calculate improbable choices made twice (consistently)
    pair_counts = improbable_choices.groupby(['user_id', 'bigram_pair']).size().reset_index(name='count')
    improbable_choices_2x = pair_counts[pair_counts['count'] == 2].groupby('user_id').size()
    user_stats['improbable_choices_2x'] = user_stats['user_id'].map(improbable_choices_2x)
    
    # Additional statistics
    user_stats['total_consistency_choices'] = user_stats['user_id'].map(
        bigram_data[bigram_data['group_size'] > 1]['user_id'].value_counts())
    user_stats['num_easy_choice_pairs'] = len(easy_choice_pairs)
    
    # Keep NaN values as NaN - don't fill with 0
    return user_stats

def process_experiment_data(data: pd.DataFrame, easy_choice_pairs: List[Tuple[str, str]], 
                           output_tables_folder: str, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Main processing orchestrator - only processes valid data."""
    logger.info("____ Process data ____")
    
    logger.info(f"Initial data shape: {data.shape}")
    
    # Remove any rows with missing critical data
    required_columns = ['chosenBigram', 'unchosenBigram', 'sliderValue', 'user_id']
    clean_data = data.dropna(subset=required_columns)
    logger.info(f"Removed {len(data) - len(clean_data)} rows with missing critical data")
    
    if len(clean_data) == 0:
        logger.error("No valid data remaining")
        return {}
    
    # Create standardized bigram pairs
    data_with_pairs = create_standardized_bigram_pairs(clean_data)
    
    # Identify probable/improbable choices
    data_with_choices = identify_probable_improbable_choices(data_with_pairs, easy_choice_pairs)
    
    # Calculate consistency
    bigram_data = calculate_choice_consistency(data_with_choices)
    
    if len(bigram_data) == 0:
        logger.error("No data remaining after processing")
        return {}
    
    # Sort the data
    bigram_data = bigram_data.sort_values(by=['user_id', 'trialId', 'bigram_pair']).reset_index(drop=True)
    
    # Create specific subsets - only from valid data
    consistent_choices = bigram_data[(bigram_data['is_consistent'] == True) & (bigram_data['group_size'] > 1)]
    inconsistent_choices = bigram_data[(bigram_data['is_consistent'] == False) & (bigram_data['group_size'] > 1)]
    probable_choices = bigram_data[bigram_data['is_probable'] == True]
    improbable_choices = bigram_data[bigram_data['is_improbable'] == True]
    
    # Calculate user statistics
    user_stats = calculate_user_statistics(bigram_data, easy_choice_pairs)
    
    # Save all dataframes
    _save_processed_dataframes({
        'bigram_data': bigram_data,
        'consistent_choices': consistent_choices,
        'inconsistent_choices': inconsistent_choices,
        'probable_choices': probable_choices,
        'improbable_choices': improbable_choices,
        'user_stats': user_stats
    }, output_tables_folder, logger)
    
    return {
        'bigram_data': bigram_data,
        'consistent_choices': consistent_choices,
        'inconsistent_choices': inconsistent_choices,
        'probable_choices': probable_choices,
        'improbable_choices': improbable_choices,
        'user_stats': user_stats
    }

def _save_processed_dataframes(dataframes: Dict[str, pd.DataFrame], output_folder: str, logger: logging.Logger) -> None:
    """Save processed dataframes to CSV files."""
    for name, df in dataframes.items():
        if name == 'user_stats':
            filename = 'processed_user_statistics.csv'
        else:
            filename = f'processed_{name}.csv'
        
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {name} to {output_path}")

#######################################
# User Filtering Functions
#######################################
def filter_users_by_improbable_choices(data: pd.DataFrame, user_stats: pd.DataFrame, 
                                      easy_choice_pairs: List[Tuple[str, str]], 
                                      threshold: int, logger: logging.Logger) -> Set[str]:
    """
    ACTUAL BEHAVIOR: Removes users who made exactly 2 improbable choices for the same bigram pair.
    This means they consistently chose the 'bad' bigram over the 'good' bigram twice.
    """
    if threshold == np.inf:
        return set()
    
    # Only work with users who have this statistic (not NaN)
    valid_stats = user_stats.dropna(subset=['improbable_choices_2x'])
    problematic_users = set(valid_stats[valid_stats['improbable_choices_2x'] > threshold]['user_id'])
    
    logger.info(f"Users removed due to improbable choices > {threshold}: {len(problematic_users)}")
    if len(problematic_users) > 0:
        logger.info(problematic_users)
    
    return problematic_users

def filter_users_by_strong_inconsistencies(data: pd.DataFrame, threshold: float, 
                                          zero_threshold: int, logger: logging.Logger) -> Set[str]:
    """
    BEHAVIOR: 
    - Groups data by user and bigram pair
    - For each pair with multiple presentations, checks if choices are inconsistent
    - If inconsistent AND any slider value is > zero_threshold away from zero, counts as "strong inconsistency"
    - Removes users where percentage of strong inconsistencies > threshold
    """
    if threshold == np.inf:
        return set()
        
    logger.info(f"Filtering by strong inconsistencies (|sliderValue| > {zero_threshold})")
    
    # Detect correct column names (handle both raw and processed data)
    chosen_col = 'chosen_bigram' if 'chosen_bigram' in data.columns else 'chosenBigram'
    unchosen_col = 'unchosen_bigram' if 'unchosen_bigram' in data.columns else 'unchosenBigram'
    
    if chosen_col not in data.columns or unchosen_col not in data.columns:
        logger.error(f"Required columns not found. Available: {list(data.columns)}")
        return set()
    
    if 'std_bigram_pair' not in data.columns:
        data = data.copy()
        data['std_bigram_pair'] = data.apply(
            lambda row: tuple(sorted([row[chosen_col], row[unchosen_col]])), axis=1)
    
    problematic_users = set()
    
    for user_id, user_data in data.groupby('user_id'):
        grouped = user_data.groupby('std_bigram_pair')
        
        total_multi_pairs = 0
        strong_inconsistent_pairs = 0
        
        for _, group in grouped:
            if len(group) <= 1:
                continue
            
            total_multi_pairs += 1
            is_consistent = len(set(group[chosen_col])) == 1
            
            if not is_consistent:
                has_strong_inconsistency = any(abs(val) > zero_threshold for val in group['sliderValue'])
                if has_strong_inconsistency:
                    strong_inconsistent_pairs += 1
        
        if total_multi_pairs > 0:
            inconsistency_pct = (strong_inconsistent_pairs / total_multi_pairs) * 100
            if inconsistency_pct > threshold:
                problematic_users.add(user_id)
    
    logger.info(f"Users removed due to strong inconsistent choices > {threshold}%: {len(problematic_users)}")
    if len(problematic_users) > 0:
        logger.info(problematic_users)

    return problematic_users

def filter_users_by_slider_behavior(data: pd.DataFrame, streak_side_percent: float, 
                                   percent_close_to_zero: float, distance_close_to_zero: int,
                                   logger: logging.Logger, verbose: bool = False) -> Tuple[Set[str], Dict]:
    """
    ACTUAL BEHAVIOR: Two separate filters:
    
    1. "Percentage-based consecutive streaks": 
       - Calculates streak threshold as streak_side_percent of user's total trials
       - Looks for streaks of that length or longer where slider > distance_close_to_zero
       - OR streaks where slider < -distance_close_to_zero  
       - Removes users with such streaks
       
    2. "Frequent zero values":
       - Counts trials where |slider| <= distance_close_to_zero
       - Removes users where this percentage >= percent_close_to_zero
    """
    problematic_users = set()
    stats = {
        'streak_users': set(),
        'close_to_zero_users': set(),
        'total_problematic_users': 0
    }
    
    for user_id, user_data in data.groupby('user_id'):
        user_data = user_data.sort_values('trialId')
        slider_values = user_data['sliderValue'].values
        total_trials = len(slider_values)
        
        # Calculate streak threshold as percentage of total trials
        streak_threshold = max(1, int(total_trials * streak_side_percent / 100))
        
        # Check for problematic streaks (percentage-based length)
        if _has_problematic_streaks_percentage(slider_values, streak_threshold, distance_close_to_zero):
            stats['streak_users'].add(user_id)
            problematic_users.add(user_id)
        
        # Check for frequent near-zero selections
        if _has_excessive_zero_values(slider_values, percent_close_to_zero, distance_close_to_zero):
            stats['close_to_zero_users'].add(user_id)
            problematic_users.add(user_id)
    
    stats['total_problematic_users'] = len(problematic_users)
    
    logger.info(f"Users removed due to problematic slider behavior: {len(problematic_users)}")
    logger.info(f"  - Consecutive streaks ({streak_side_percent}% of trials): {len(stats['streak_users'])}")
    logger.info(f"  - Frequent zero values ({percent_close_to_zero}%+): {len(stats['close_to_zero_users'])}")
    if len(problematic_users) > 0:
        logger.info(problematic_users)

    return problematic_users, stats

def _has_problematic_streaks_percentage(slider_values: np.ndarray, streak_threshold: int, 
                                      distance_close_to_zero: int) -> bool:
    """Check if user has problematic streaks based on percentage of total trials."""
    pos_streak = 0
    neg_streak = 0
    
    for value in slider_values:
        if value > distance_close_to_zero:
            pos_streak += 1
            neg_streak = 0
        elif value < -distance_close_to_zero:
            neg_streak += 1
            pos_streak = 0
        else:
            pos_streak = 0
            neg_streak = 0
            
        if pos_streak >= streak_threshold or neg_streak >= streak_threshold:
            return True
    
    return False

def _has_excessive_zero_values(slider_values: np.ndarray, percent_close_to_zero: float, 
                             distance_close_to_zero: int) -> bool:
    """Check if user has excessive values close to zero."""
    total_trials = len(slider_values)
    close_to_zero_count = sum(abs(value) <= distance_close_to_zero for value in slider_values)
    close_to_zero_pct = (close_to_zero_count / total_trials * 100)
    
    return close_to_zero_pct >= percent_close_to_zero

def apply_all_user_filters(data: pd.DataFrame, user_stats: pd.DataFrame, 
                          config: Dict[str, Any], easy_choice_pairs: List[Tuple[str, str]], 
                          logger: logging.Logger) -> Tuple[pd.DataFrame, Set[str]]:
    """Apply all user-level filtering based on configuration."""
    logger.info("____ Filtering users ____")
    
    initial_users = len(data['user_id'].unique())
    initial_rows = len(data)
    all_users = set(data['user_id'].unique())
    valid_users = all_users.copy()
    
    process_config = config['process']
    
    # Filter by improbable choices
    if process_config.get('filter_users_by_num_improbable_choices', False):
        threshold = process_config.get('improbable_threshold', np.inf)
        improbable_users = filter_users_by_improbable_choices(data, user_stats, easy_choice_pairs, threshold, logger)
        valid_users -= improbable_users

    # Filter by strong inconsistencies
    if process_config.get('filter_users_by_percent_strong_inconsistencies', False):
        threshold = process_config.get('strong_inconsistent_threshold', np.inf)
        zero_threshold = process_config.get('zero_threshold', 20)
        inconsistent_users = filter_users_by_strong_inconsistencies(data, threshold, zero_threshold, logger)
        valid_users -= inconsistent_users
    
    # Filter by slider behavior
    streak_side_percent = process_config.get('streak_side_percent', 20)  # Percentage of trials for streak threshold
    percent_close_to_zero = process_config.get('percent_close_to_zero', 25)
    distance_close_to_zero = process_config.get('distance_close_to_zero', 10)
    slider_users, _ = filter_users_by_slider_behavior(data, streak_side_percent, percent_close_to_zero, distance_close_to_zero, logger)
    valid_users -= slider_users
    
    # Filter data to keep only valid users
    filtered_data = data[data['user_id'].isin(valid_users)]
    
    # Print summary
    removed_users = initial_users - len(valid_users)
    removed_rows = initial_rows - len(filtered_data)
    
    logger.info("Filtering summary:")
    logger.info(f"Initial users: {initial_users}, Remaining users: {len(valid_users)}")
    logger.info(f"Users removed: {removed_users} ({removed_users/initial_users:.1%})")
    logger.info(f"Rows removed: {removed_rows} ({removed_rows/initial_rows:.1%})")
    
    return filtered_data, valid_users

#######################################
# Row Filtering Functions
#######################################
def filter_data_rows(data: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Filter rows of data based on configuration criteria."""
    logger.info("____ Filtering data rows ____")
    
    process_config = config['process']
    data_filtered = data.copy()
    initial_rows = len(data_filtered)
    
    # Create standardized bigram pair representation
    if 'std_bigram_pair' not in data_filtered.columns:
        # Detect correct column names (handle both raw and processed data)
        chosen_col = 'chosen_bigram' if 'chosen_bigram' in data_filtered.columns else 'chosenBigram'
        unchosen_col = 'unchosen_bigram' if 'unchosen_bigram' in data_filtered.columns else 'unchosenBigram'
        
        if chosen_col in data_filtered.columns and unchosen_col in data_filtered.columns:
            data_filtered['std_bigram_pair'] = data_filtered.apply(
                lambda row: tuple(sorted([row[chosen_col], row[unchosen_col]])), axis=1)
        else:
            logger.error(f"Cannot create std_bigram_pair - missing columns. Available: {list(data_filtered.columns)}")
            return data_filtered
    
    # Filter by letters
    filter_letters = process_config.get('filter_letters', [])
    if filter_letters:
        data_filtered = _filter_by_letters(data_filtered, filter_letters, logger)
    
    # Filter single presentations
    if process_config.get('filter_single_presentations', False):
        data_filtered = _filter_single_presentations(data_filtered, logger)
    
    # Filter inconsistent choices away from zero
    if process_config.get('filter_inconsistent_choices_away_from_zero', False):
        zero_threshold = process_config.get('zero_threshold', 20)
        data_filtered = _filter_inconsistent_away_from_zero(data_filtered, zero_threshold, logger)
    
    total_removed = initial_rows - len(data_filtered)
    logger.info(f"Total rows removed by all filters: {total_removed} ({total_removed/initial_rows:.1%})")
    
    return data_filtered

def _filter_by_letters(data: pd.DataFrame, filter_letters: List[str], logger: logging.Logger) -> pd.DataFrame:
    """Filter out bigrams containing specified letters."""
    logger.info(f"Filtering out bigrams containing: {filter_letters}")
    initial_rows = len(data)
    
    # Detect correct column names (handle both raw and processed data)
    chosen_col = 'chosen_bigram' if 'chosen_bigram' in data.columns else 'chosenBigram'
    unchosen_col = 'unchosen_bigram' if 'unchosen_bigram' in data.columns else 'unchosenBigram'
    
    if chosen_col not in data.columns or unchosen_col not in data.columns:
        logger.warning(f"Required columns not found for letter filtering. Available: {list(data.columns)}")
        return data
    
    filter_condition = ~(
        data[chosen_col].str.contains('|'.join(filter_letters), case=False) |
        data[unchosen_col].str.contains('|'.join(filter_letters), case=False)
    )
    
    data_filtered = data[filter_condition].copy()
    removed_rows = initial_rows - len(data_filtered)
    
    logger.info(f"Rows removed by letter filtering: {removed_rows}")
    return data_filtered

def _filter_single_presentations(data: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Filter out single-presentation bigram pairs per user."""
    logger.info("Filtering out single-presentation bigram pairs per user")
    initial_rows = len(data)
    
    grouped_data = data.groupby(['user_id', 'std_bigram_pair'])
    group_sizes = grouped_data.size()
    valid_groups = group_sizes[group_sizes >= 2].reset_index()[['user_id', 'std_bigram_pair']]
    
    data_filtered = pd.merge(data, valid_groups, on=['user_id', 'std_bigram_pair'], how='inner')
    removed_rows = initial_rows - len(data_filtered)
    
    logger.info(f"Rows removed by single-presentation filtering: {removed_rows}")
    return data_filtered

def _filter_inconsistent_away_from_zero(data: pd.DataFrame, zero_threshold: int, logger: logging.Logger) -> pd.DataFrame:
    """Filter out inconsistent choices with slider values away from zero."""
    logger.info(f"Filtering inconsistent choices with slider values away from zero (threshold: ±{zero_threshold})")
    initial_rows = len(data)
    
    # Detect correct column names (handle both raw and processed data)
    chosen_col = 'chosen_bigram' if 'chosen_bigram' in data.columns else 'chosenBigram'
    unchosen_col = 'unchosen_bigram' if 'unchosen_bigram' in data.columns else 'unchosenBigram'
    
    if chosen_col not in data.columns or unchosen_col not in data.columns:
        logger.error(f"Required columns not found. Available: {list(data.columns)}")
        return data
    
    grouped_data = data.groupby(['user_id', 'std_bigram_pair'])
    valid_groups = []
    
    for (user_id, std_bigram_pair), group in grouped_data:
        if len(group) <= 1:
            valid_groups.append((user_id, std_bigram_pair))
            continue
        
        is_consistent = len(set(group[chosen_col])) == 1
        
        if is_consistent or all(abs(val) <= zero_threshold for val in group['sliderValue']):
            valid_groups.append((user_id, std_bigram_pair))
    
    if valid_groups:
        valid_groups_df = pd.DataFrame(valid_groups, columns=['user_id', 'std_bigram_pair'])
        data_filtered = pd.merge(data, valid_groups_df, on=['user_id', 'std_bigram_pair'], how='inner')
    else:
        data_filtered = data.copy()
    
    removed_rows = initial_rows - len(data_filtered)
    logger.info(f"Rows removed by inconsistent-away-from-zero filtering: {removed_rows}")
    
    return data_filtered

#######################################
# Scoring Functions (No Fallbacks)
#######################################
def score_user_choices_by_slider_values(filtered_users_data: Dict[str, pd.DataFrame], 
                                       output_tables_folder: str, logger: logging.Logger) -> pd.DataFrame:
    """Score each user's choices - only from valid data."""
    logger.info("____ Score choices by slider values ____")
    bigram_data = filtered_users_data['bigram_data']
    
    # Only work with complete rows
    required_cols = ['user_id', 'bigram_pair', 'bigram1', 'bigram2', 'chosen_bigram', 'sliderValue']
    valid_data = bigram_data.dropna(subset=required_cols)
    
    logger.info(f"Removed {len(bigram_data) - len(valid_data)} rows with missing scoring data")
    
    if len(valid_data) == 0:
        logger.error("No valid data for scoring")
        return pd.DataFrame()
    
    # Group by user_id and bigram_pair and apply scoring
    grouped = valid_data.groupby(['user_id', 'bigram_pair'], group_keys=False)
    scored_results = []
    
    for (user_id, bigram_pair), group in grouped:
        result = determine_score(group)
        if result is not None:  # Only add if scoring succeeded
            result['user_id'] = user_id
            result['bigram_pair'] = bigram_pair
            scored_results.append(result)
    
    if not scored_results:
        logger.error("No valid scoring results")
        return pd.DataFrame()
    
    scored_bigram_data = pd.DataFrame(scored_results)
    
    # Reorder columns
    column_order = [
        'user_id', 'bigram_pair', 'bigram1', 'bigram2',
        'chosen_bigram_winner', 'unchosen_bigram_winner', 
        'chosen_bigram_time_median', 'unchosen_bigram_time_median',
        'chosen_bigram_correct_total', 'unchosen_bigram_correct_total', 
        'score', 'text', 'is_consistent', 'is_probable', 'is_improbable', 'group_size'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in column_order if col in scored_bigram_data.columns]
    scored_bigram_data = scored_bigram_data[available_columns]
    
    logger.info(f"Successfully scored {len(scored_bigram_data)} user-bigram combinations")
    
    # Save scored data
    output_file = os.path.join(output_tables_folder, 'scored_bigram_data.csv')
    scored_bigram_data.to_csv(output_file, index=False)
    logger.info(f"Scored bigram data saved to {output_file}")
    
    return scored_bigram_data

def determine_score(group: pd.DataFrame) -> Optional[Dict]:
    """Determine the score for a group - return None if invalid."""
    if len(group) == 0:
        return None
    
    # Ensure we have required data
    required_cols = ['bigram1', 'bigram2', 'chosen_bigram', 'sliderValue']
    if not all(col in group.columns for col in required_cols):
        return None
    
    if group[required_cols].isna().any().any():
        return None
    
    bigram1, bigram2 = group['bigram1'].iloc[0], group['bigram2'].iloc[0]
    chosen_bigrams = group['chosen_bigram']
    slider_values = group['sliderValue']
    group_size = len(group)
    
    # Determine scoring method based on choice consistency
    if len(set(chosen_bigrams)) == 1:
        # Consistent choices - median absolute slider value
        valid_values = [abs(x) for x in slider_values if not np.isnan(x)]
        if not valid_values:
            return None
        median_abs_slider_value = np.median(valid_values)
        chosen_bigram_winner = chosen_bigrams.iloc[0]
        unchosen_bigram_winner = bigram2 if chosen_bigram_winner == bigram1 else bigram1
    else:
        # Mixed choices - difference of sums
        valid_scores1 = [abs(x) for i, x in enumerate(slider_values) if chosen_bigrams.iloc[i] == bigram1 and not np.isnan(x)]
        valid_scores2 = [abs(x) for i, x in enumerate(slider_values) if chosen_bigrams.iloc[i] == bigram2 and not np.isnan(x)]
        
        sum1 = sum(valid_scores1)
        sum2 = sum(valid_scores2)
        
        median_abs_slider_value = abs(sum1 - sum2) / group_size
        
        if sum1 >= sum2:
            chosen_bigram_winner, unchosen_bigram_winner = bigram1, bigram2
        else:
            chosen_bigram_winner, unchosen_bigram_winner = bigram2, bigram1
    
    # Calculate final score (normalized to 0-1 range)
    score = median_abs_slider_value / 100
    
    # Get additional stats (with safe access)
    def safe_median(series):
        clean_series = series.dropna()
        return clean_series.median() if len(clean_series) > 0 else np.nan
    
    result = {
        'bigram1': bigram1,
        'bigram2': bigram2,
        'chosen_bigram_winner': chosen_bigram_winner,
        'unchosen_bigram_winner': unchosen_bigram_winner,
        'score': score,
        'is_consistent': (len(set(chosen_bigrams)) == 1),
        'group_size': group_size
    }
    
    # Add optional columns if they exist
    if 'chosen_bigram_time' in group.columns:
        result['chosen_bigram_time_median'] = safe_median(group['chosen_bigram_time'])
    if 'unchosen_bigram_time' in group.columns:
        result['unchosen_bigram_time_median'] = safe_median(group['unchosen_bigram_time'])
    if 'chosen_bigram_correct' in group.columns:
        result['chosen_bigram_correct_total'] = group['chosen_bigram_correct'].sum(skipna=True)
    if 'unchosen_bigram_correct' in group.columns:
        result['unchosen_bigram_correct_total'] = group['unchosen_bigram_correct'].sum(skipna=True)
    if 'text' in group.columns:
        result['text'] = tuple(group['text'].unique())
    if 'is_probable' in group.columns:
        result['is_probable'] = group['is_probable'].iloc[0]
    if 'is_improbable' in group.columns:
        result['is_improbable'] = group['is_improbable'].iloc[0]
    
    return result

#######################################
# Winner Determination (No Fallbacks)
#######################################
def choose_bigram_winners(scored_bigram_data: pd.DataFrame, output_tables_folder: str, logger: logging.Logger) -> pd.DataFrame:
    """Determine winning bigram for each bigram pair - only from valid data."""
    logger.info("____ Choose bigram winners ____")
    
    # Only work with complete data
    required_cols = ['bigram_pair', 'bigram1', 'bigram2', 'chosen_bigram_winner', 'score']
    valid_data = scored_bigram_data.dropna(subset=required_cols)
    
    logger.info(f"Removed {len(scored_bigram_data) - len(valid_data)} rows with missing winner data")
    
    if len(valid_data) == 0:
        logger.error("No valid data for winner determination")
        return pd.DataFrame()
    
    # Group by bigram_pair and determine winners
    grouped = valid_data.groupby(['bigram_pair'])
    winner_results = []
    
    for bigram_pair, group in grouped:
        result = determine_overall_winner(group)
        if result is not None:
            result['bigram_pair'] = bigram_pair
            winner_results.append(result)
    
    if not winner_results:
        logger.error("No valid winner results")
        return pd.DataFrame()
    
    bigram_winner_data = pd.DataFrame(winner_results)
    
    logger.info(f"Successfully determined winners for {len(bigram_winner_data)} bigram pairs")
    
    # Save results
    output_file = os.path.join(output_tables_folder, 'bigram_winner_data.csv')
    bigram_winner_data.to_csv(output_file, index=False)
    logger.info(f"Bigram winner data saved to {output_file}")
    
    return bigram_winner_data

def determine_overall_winner(group: pd.DataFrame) -> Optional[Dict]:
    """Determine the winning bigram across all users - return None if invalid."""
    if len(group) == 0:
        return None
    
    # Ensure we have required data
    required_cols = ['bigram1', 'bigram2', 'score', 'chosen_bigram_winner']
    if not all(col in group.columns for col in required_cols):
        return None
    
    if group[required_cols].isna().any().any():
        return None
    
    bigram1, bigram2 = group['bigram1'].iloc[0], group['bigram2'].iloc[0]
    scores = group['score']
    chosen_bigrams = group['chosen_bigram_winner']
    
    unique_chosen_bigrams = chosen_bigrams.unique()
    
    if len(unique_chosen_bigrams) == 1:
        # All users chose the same bigram
        valid_scores = [abs(x) for x in scores if not np.isnan(x)]
        if not valid_scores:
            return None
        median_score = np.median(valid_scores)
        mad_score = median_abs_deviation(valid_scores) if len(valid_scores) > 1 else 0
        winner_bigram = unique_chosen_bigrams[0]
        loser_bigram = bigram2 if winner_bigram == bigram1 else bigram1
    else:
        # Mixed choices across users
        sum1 = sum(abs(x) for i, x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram1 and not np.isnan(x))
        sum2 = sum(abs(x) for i, x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram2 and not np.isnan(x))
        median_score = abs(sum1 - sum2) / len(group)
        
        # Calculate MAD for mixed choices
        mad_list = []
        for i, x in enumerate(scores):
            if not np.isnan(x):
                if chosen_bigrams.iloc[i] == bigram1:
                    mad_list.append(abs(x))
                elif chosen_bigrams.iloc[i] == bigram2:
                    mad_list.append(-abs(x))
        
        mad_score = median_abs_deviation(mad_list) if len(mad_list) > 1 else 0
        
        if sum1 >= sum2:
            winner_bigram, loser_bigram = bigram1, bigram2
        else:
            winner_bigram, loser_bigram = bigram2, bigram1
    
    result = {
        'winner_bigram': winner_bigram,
        'loser_bigram': loser_bigram,
        'median_score': median_score,
        'mad_score': mad_score,
        'is_consistent': group['is_consistent'].all() if 'is_consistent' in group.columns else False,
        'group_size': group['group_size'].sum() if 'group_size' in group.columns else len(group)
    }
    
    # Add optional columns if they exist
    if 'is_probable' in group.columns:
        result['is_probable'] = group['is_probable'].iloc[0]
    if 'is_improbable' in group.columns:
        result['is_improbable'] = group['is_improbable'].iloc[0]
    if 'text' in group.columns:
        result['text'] = tuple(group['text'].unique())
    
    return result

#######################################
# Visualization Functions
#######################################
def create_user_choice_visualizations(user_stats: pd.DataFrame, output_plots_folder: str, 
                                     config: Dict[str, Any], plot_label: str, logger: logging.Logger) -> None:
    """Create visualizations for user choice patterns."""
    logger.info("Creating user choice visualizations")
    
    viz_config = config.get('visualization', {})
    figsize = viz_config.get('figsize', [10, 6])
    dpi = viz_config.get('dpi', 300)
    colors = viz_config.get('colors', {})
    
    # Set matplotlib style
    plt.style.use(viz_config.get('style', 'default'))
    
    # Only visualize users with valid data
    valid_stats = user_stats.dropna(subset=['consistent_choices']).copy()
    
    if len(valid_stats) == 0:
        logger.warning("No valid user statistics for visualization")
        return
    
    # Sort users by consistent choices
    user_order = valid_stats.sort_values('consistent_choices', ascending=False)['user_id']
    
    # Create consistent vs inconsistent choices plot
    _create_choice_comparison_plot(
        valid_stats, user_order, 
        ['consistent_choices', 'inconsistent_choices'],
        ['Consistent Choices', 'Inconsistent Choices'],
        [colors.get('primary', '#1f77b4'), colors.get('secondary', '#ff7f0e')],
        'Consistent vs. Inconsistent Choices per User',
        os.path.join(output_plots_folder, f'{plot_label}consistent_vs_inconsistent_choices.png'),
        figsize, dpi
    )
    
    # Create probable vs improbable choices plot
    _create_choice_comparison_plot(
        valid_stats, user_order,
        ['probable_choices', 'improbable_choices'],
        ['Probable Choices', 'Improbable Choices'],
        [colors.get('primary', '#1f77b4'), '#d62728'],
        'Probable vs. Improbable Choices per User',
        os.path.join(output_plots_folder, f'{plot_label}probable_vs_improbable_choices.png'),
        figsize, dpi
    )
    
    logger.info(f"Visualization plots saved to {output_plots_folder}")

def _create_choice_comparison_plot(user_stats: pd.DataFrame, user_order: pd.Series, 
                                  columns: List[str], labels: List[str], colors: List[str],
                                  title: str, filename: str, figsize: List[int], dpi: int) -> None:
    """Create a horizontal stacked bar plot comparing two choice types."""
    # Only plot users with valid data
    data = user_stats.set_index('user_id').loc[user_order, columns].fillna(0)
    users = data.index
    
    # Dynamic height based on number of users
    fig_height = max(figsize[1], len(data) * 0.5)
    plt.figure(figsize=(figsize[0], fig_height), dpi=dpi)
    
    # Create stacked horizontal bar plot
    plt.barh(users, data[columns[0]], color=colors[0], label=labels[0])
    plt.barh(users, data[columns[1]], left=data[columns[0]], color=colors[1], label=labels[1])
    
    plt.title(title)
    plt.xlabel('Number of Choices')
    plt.ylabel('User ID')
    plt.legend(title='Choice Type')
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

#######################################
# Utility Functions
#######################################
def display_information(dframe: pd.DataFrame, title: str, print_headers: List[str], nlines: int) -> None:
    """Display information about a DataFrame."""
    print(f'\n{title}:')
    with pd.option_context('display.max_rows', nlines):
        print(dframe[print_headers].iloc[:nlines])
    print()

#######################################
# Main Execution
#######################################
def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process bigram typing data')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration and setup logging
    config = load_config(args.config)
    logger = setup_logging(config)
    
    logger.info("Starting bigram data processing")
    
    # Setup paths
    input_folder = config['data']['input_dir']
    output_folder = os.path.join(input_folder, 'output')
    output_tables_folder = os.path.join(output_folder, 'tables')
    output_plots_folder = os.path.join(output_folder, 'plots')
    
    os.makedirs(output_tables_folder, exist_ok=True)
    os.makedirs(output_plots_folder, exist_ok=True)
    
    # Load easy choice pairs
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    easy_choice_pairs_file = os.path.join(parent_dir, 'bigram_tables', 'bigram_easy_choice_pairs_LH_nocomments.csv')
    easy_choice_pairs = load_easy_choice_pairs(easy_choice_pairs_file, logger)
    
    if not easy_choice_pairs:
        logger.warning("No easy choice pairs loaded - continuing without improbable choice filtering")
    
    # Step 1: Load and combine data
    raw_data = load_and_combine_data(input_folder, output_tables_folder, logger, verbose=False)
    
    if raw_data is None:
        logger.error("No data was loaded. Please check your input folder and files.")
        logger.error(f"Expected input folder: {input_folder}")
        return 1
    
    initial_users = len(raw_data['user_id'].unique())
    initial_trials = len(raw_data)
    
    logger.info(f"Initial data - Users: {initial_users}, Trials: {initial_trials}")
    
    # Step 2: Process data without filtering
    processed_data = process_experiment_data(raw_data, easy_choice_pairs, output_tables_folder, logger)
    
    if not processed_data:
        logger.error("Processing failed")
        return 1
    
    # Initial visualizations
    create_user_choice_visualizations(processed_data['user_stats'], output_plots_folder, config, "initial_", logger)
    
    # Step 3: Filter users using processed data
    filtered_users_data, valid_users = apply_all_user_filters(processed_data['bigram_data'], processed_data['user_stats'], config, easy_choice_pairs, logger)
    
    # Step 4: Filter rows
    filtered_data = filter_data_rows(filtered_users_data, config, logger)
    
    # Step 5: Create final processed data from already-processed filtered data
    logger.info("____ Creating final processed data from filtered data ____")
    
    if len(filtered_data) == 0:
        logger.error("No data remaining after filtering")
        return 1
    
    # Create subsets from filtered data (data is already processed)
    consistent_choices = filtered_data[(filtered_data['is_consistent'] == True) & (filtered_data['group_size'] > 1)]
    inconsistent_choices = filtered_data[(filtered_data['is_consistent'] == False) & (filtered_data['group_size'] > 1)]
    probable_choices = filtered_data[filtered_data['is_probable'] == True]
    improbable_choices = filtered_data[filtered_data['is_improbable'] == True]
    
    # Recalculate user statistics for filtered data
    final_user_stats = calculate_user_statistics(filtered_data, easy_choice_pairs)
    
    # Create final processed data structure
    final_processed_data = {
        'bigram_data': filtered_data,
        'consistent_choices': consistent_choices,
        'inconsistent_choices': inconsistent_choices,
        'probable_choices': probable_choices,
        'improbable_choices': improbable_choices,
        'user_stats': final_user_stats
    }
    
    # Save final dataframes
    _save_processed_dataframes({
        'bigram_data': filtered_data,
        'consistent_choices': consistent_choices,
        'inconsistent_choices': inconsistent_choices,
        'probable_choices': probable_choices,
        'improbable_choices': improbable_choices,
        'user_stats': final_user_stats
    }, output_tables_folder, logger)
    
    final_trials = len(final_processed_data['bigram_data'])
    final_users = len(final_processed_data['bigram_data']['user_id'].unique())
    
    logger.info(f"Final data after filtering - Users: {final_users}, Trials: {final_trials}")
    logger.info(f"Trials removed: {initial_trials - final_trials}")
    
    # Generate final visualizations
    create_user_choice_visualizations(final_processed_data['user_stats'], output_plots_folder, config, "filtered_", logger)
    
    # Step 6: Score choices by slider values
    scored_data = score_user_choices_by_slider_values(final_processed_data, output_tables_folder, logger)
    
    if len(scored_data) == 0:
        logger.error("Scoring failed")
        return 1
    
    # Step 7: Choose winning bigrams
    bigram_winner_data = choose_bigram_winners(scored_data, output_tables_folder, logger)
    
    if len(bigram_winner_data) == 0:
        logger.error("Winner determination failed")
        return 1
    
    logger.info("Bigram data processing completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())