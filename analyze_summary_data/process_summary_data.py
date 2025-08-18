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

    Parameters:
    - input_folder: path to the folder containing folders of CSV files
    - output_tables_folder: path to the folder where the combined data will be saved
    - logger: configured logger instance
    - verbose: if True, provide detailed logging output

    Returns:
    - combined_df: DataFrame with combined data, or None if no data found
    """
    logger.info(f"Loading data from {input_folder}")
    
    if not os.path.exists(input_folder):
        logger.error(f"Input folder does not exist: {input_folder}")
        return None
    
    dataframes = []
    csv_files_found = []
    
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
                    
                    # Extract user ID from filename with robust parsing
                    user_id = _extract_user_id_from_filename(filename)
                    df['user_id'] = user_id
                    df['filename'] = filename
                    
                    # Add the subfolder name
                    subfolder = os.path.relpath(root, input_folder)
                    df['group_id'] = subfolder if subfolder != '.' else ''
                    
                    # Filter out intro trials
                    df_filtered = _filter_intro_trials(df, logger, verbose)
                    
                    if len(df_filtered) > 0:
                        dataframes.append(df_filtered)
                        if verbose:
                            logger.debug(f"Added {filename} to dataframes list")
                    else:
                        logger.warning(f"{filename} has no rows after filtering")
                        
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {str(e)}")
                    continue
    
    # Summary logging
    logger.info(f"Total CSV files found: {len(csv_files_found)}")
    logger.info(f"Successfully processed files: {len(dataframes)}")
    
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

    if verbose:
        logger.debug(f"Combined dataframe columns: {list(combined_df.columns)}")
        _display_sample_data(combined_df, logger)

    # Save the combined DataFrame to a CSV file
    output_file = os.path.join(output_tables_folder, 'original_combined_data.csv')
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Combined data saved to {output_file}")

    return combined_df

def _extract_user_id_from_filename(filename: str) -> str:
    """Extract user ID from filename with robust parsing."""
    try:
        # Try expected format: experiment_data_USERID_*.csv
        parts = filename.replace('.csv', '').split('_')
        if len(parts) > 2:
            return parts[2]
        else:
            return filename.replace('.csv', '')
    except Exception:
        return filename.replace('.csv', '')

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
    """
    Load easy choice bigram pairs from a CSV file.

    Parameters:
    - file_path: String, path to the CSV file containing easy choice bigram pairs
    - logger: configured logger instance

    Returns:
    - easy_choice_pairs: List of tuples, each containing a pair of bigrams where one is highly improbable
    """
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
# User Filtering Functions
#######################################
def filter_users_by_improbable_choices(data: pd.DataFrame, user_stats: pd.DataFrame, 
                                      easy_choice_pairs: List[Tuple[str, str]], 
                                      threshold: int, logger: logging.Logger) -> Set[str]:
    """Filter users who consistently make improbable choices."""
    if threshold == np.inf:
        return set()
    
    problematic_users = set(user_stats[user_stats['improbable_choices_2x'] > threshold]['user_id'])
    
    logger.info(f"Users removed due to improbable choices > {threshold}: {len(problematic_users)}")
    
    # Only log details in debug mode
    if logger.isEnabledFor(logging.DEBUG) and len(problematic_users) > 0 and len(problematic_users) <= 10:
        logger.debug(f"Removed users: {problematic_users}")
        _log_improbable_choice_details(data, problematic_users, easy_choice_pairs, logger)
    
    return problematic_users

def filter_users_by_inconsistencies(data: pd.DataFrame, user_stats: pd.DataFrame, 
                                   threshold: float, filter_strong_only: bool, 
                                   zero_threshold: int, logger: logging.Logger) -> Set[str]:
    """Filter users with too many inconsistent choices."""
    if threshold == np.inf:
        return set()
    
    if filter_strong_only:
        return _filter_strong_inconsistencies(data, threshold, zero_threshold, logger)
    else:
        user_inconsistency_pcts = (user_stats['inconsistent_choices'] / user_stats['total_consistency_choices'] * 100)
        problematic_users = set(user_stats[user_inconsistency_pcts > threshold]['user_id'])
        
        logger.info(f"Users removed due to inconsistent choices > {threshold}%: {len(problematic_users)}")
        if len(problematic_users) > 0 and len(problematic_users) <= 10:
            logger.debug(f"Removed users: {problematic_users}")
        
        return problematic_users

def _filter_strong_inconsistencies(data: pd.DataFrame, threshold: float, 
                                  zero_threshold: int, logger: logging.Logger) -> Set[str]:
    """Filter users based on strong inconsistencies (slider values away from zero)."""
    logger.info(f"Filtering by strong inconsistencies (|sliderValue| > {zero_threshold})")
    
    if 'std_bigram_pair' not in data.columns:
        data = data.copy()
        data['std_bigram_pair'] = data.apply(
            lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
    
    problematic_users = set()
    
    for user_id, user_data in data.groupby('user_id'):
        grouped = user_data.groupby('std_bigram_pair')
        
        total_multi_pairs = 0
        strong_inconsistent_pairs = 0
        
        for _, group in grouped:
            if len(group) <= 1:
                continue
            
            total_multi_pairs += 1
            is_consistent = len(set(group['chosenBigram'])) == 1
            
            if not is_consistent:
                has_strong_inconsistency = any(abs(val) > zero_threshold for val in group['sliderValue'])
                if has_strong_inconsistency:
                    strong_inconsistent_pairs += 1
        
        if total_multi_pairs > 0:
            inconsistency_pct = (strong_inconsistent_pairs / total_multi_pairs) * 100
            if inconsistency_pct > threshold:
                problematic_users.add(user_id)
    
    logger.info(f"Users removed due to strong inconsistent choices > {threshold}%: {len(problematic_users)}")
    return problematic_users

def filter_users_by_slider_behavior(data: pd.DataFrame, n_repeat_sides: int, 
                                   percent_close_to_zero: float, distance_close_to_zero: int,
                                   logger: logging.Logger, verbose: bool = False) -> Tuple[Set[str], Dict]:
    """Filter users with problematic slider patterns."""
    problematic_users = set()
    stats = {
        'repeated_side_users': set(),
        'close_to_zero_users': set(),
        'total_problematic_users': 0
    }
    
    if verbose:
        logger.debug(f"Checking for streaks >= {n_repeat_sides} with abs(value) > {distance_close_to_zero}")
    
    for user_id, user_data in data.groupby('user_id'):
        user_data = user_data.sort_values('trialId')
        slider_values = user_data['sliderValue'].values
        
        # Check for repeated same-side selections
        if _has_problematic_streaks(slider_values, n_repeat_sides, distance_close_to_zero, logger, verbose, user_id):
            stats['repeated_side_users'].add(user_id)
            problematic_users.add(user_id)
        
        # Check for frequent near-zero selections
        if _has_excessive_zero_values(slider_values, percent_close_to_zero, distance_close_to_zero, logger, verbose, user_id):
            stats['close_to_zero_users'].add(user_id)
            problematic_users.add(user_id)
    
    stats['total_problematic_users'] = len(problematic_users)
    
    logger.info(f"Users removed due to problematic slider behavior: {len(problematic_users)}")
    logger.info(f"  - Same side consecutive selections ({n_repeat_sides}+): {len(stats['repeated_side_users'])}")
    logger.info(f"  - Frequent zero values ({percent_close_to_zero}%+): {len(stats['close_to_zero_users'])}")
    
    return problematic_users, stats

def _has_problematic_streaks(slider_values: np.ndarray, n_repeat_sides: int, 
                           distance_close_to_zero: int, logger: logging.Logger, 
                           verbose: bool, user_id: str) -> bool:
    """Check if user has problematic streaks of same-side selections."""
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
            
        if pos_streak >= n_repeat_sides or neg_streak >= n_repeat_sides:
            if verbose:
                logger.debug(f"User {user_id} flagged for streaks - pos_streak: {pos_streak}, neg_streak: {neg_streak}")
            return True
    
    return False

def _has_excessive_zero_values(slider_values: np.ndarray, percent_close_to_zero: float, 
                             distance_close_to_zero: int, logger: logging.Logger, 
                             verbose: bool, user_id: str) -> bool:
    """Check if user has excessive values close to zero."""
    total_trials = len(slider_values)
    close_to_zero_count = sum(abs(value) <= distance_close_to_zero for value in slider_values)
    close_to_zero_pct = (close_to_zero_count / total_trials * 100)
    
    if close_to_zero_pct >= percent_close_to_zero:
        if verbose:
            logger.debug(f"User {user_id} flagged for close-to-zero: {close_to_zero_pct:.1f}% ({close_to_zero_count}/{total_trials})")
        return True
    
    return False

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
    
    # Filter by inconsistencies
    if process_config.get('filter_users_by_percent_inconsistencies', False):
        threshold = process_config.get('inconsistent_threshold', np.inf)
        filter_strong = process_config.get('filter_by_strong_inconsistencies', False)
        zero_threshold = process_config.get('zero_threshold', 20)
        inconsistent_users = filter_users_by_inconsistencies(data, user_stats, threshold, filter_strong, zero_threshold, logger)
        valid_users -= inconsistent_users
    
    # Filter by slider behavior
    n_repeat_sides = process_config.get('n_repeat_sides', 20)
    percent_close_to_zero = process_config.get('percent_close_to_zero', 25)
    distance_close_to_zero = process_config.get('distance_close_to_zero', 10)
    slider_users, _ = filter_users_by_slider_behavior(data, n_repeat_sides, percent_close_to_zero, distance_close_to_zero, logger)
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

def _log_improbable_choice_details(data: pd.DataFrame, problematic_users: Set[str], 
                                  easy_choice_pairs: List[Tuple[str, str]], logger: logging.Logger) -> None:
    """Log detailed information about users' improbable choices."""
    logger.debug("Details of consistent improbable choices:")
    
    if 'std_bigram_pair' not in data.columns:
        data = data.copy()
        data['std_bigram_pair'] = data.apply(
            lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
    
    prob_user_data = data[data['user_id'].isin(problematic_users)]
    
    for user_id in problematic_users:
        user_data = prob_user_data[prob_user_data['user_id'] == user_id]
        logger.debug(f"User {user_id}: {len(user_data)} trials")
        
        pair_groups = user_data.groupby('std_bigram_pair')
        has_improbable_choices = False
        
        for pair, group in pair_groups:
            chosen_bigrams = group['chosenBigram'].tolist()
            if len(chosen_bigrams) >= 2 and len(set(chosen_bigrams)) == 1:
                chosen = chosen_bigrams[0]
                unchosen = group['unchosenBigram'].iloc[0]
                
                # Check if this matches any easy choice pair
                for good, bad in easy_choice_pairs:
                    if chosen == bad and unchosen == good:
                        logger.debug(f"  Consistently chose improbable '{chosen}' over '{unchosen}' {len(chosen_bigrams)} times")
                        has_improbable_choices = True
                        break
        
        if not has_improbable_choices:
            logger.debug("  No consistent improbable choices found")

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
        data_filtered['std_bigram_pair'] = data_filtered.apply(
            lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
    
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
    
    filter_condition = ~(
        data['chosenBigram'].str.contains('|'.join(filter_letters), case=False) |
        data['unchosenBigram'].str.contains('|'.join(filter_letters), case=False)
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
    
    grouped_data = data.groupby(['user_id', 'std_bigram_pair'])
    valid_groups = []
    
    for (user_id, std_bigram_pair), group in grouped_data:
        if len(group) <= 1:
            valid_groups.append((user_id, std_bigram_pair))
            continue
        
        is_consistent = len(set(group['chosenBigram'])) == 1
        
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
# Data Processing Functions
#######################################
def create_standardized_bigram_pairs(data: pd.DataFrame) -> pd.DataFrame:
    """Create standardized bigram pair representation as tuples."""
    data = data.copy()
    data['std_bigram_pair'] = data.apply(
        lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1)
    return data

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
    """Determine if choices are consistent for each group - optimized version."""
    logger = logging.getLogger(__name__)
    initial_rows = len(data_with_choices)
    
    # Debug: Check initial columns
    logger.debug(f"Initial columns: {list(data_with_choices.columns)}")
    
    # First, let's clean the data and remove any rows with NaN bigrams
    clean_data = data_with_choices.dropna(subset=['chosenBigram', 'unchosenBigram']).copy()
    
    if len(clean_data) < initial_rows:
        logger.warning(f"Removed {initial_rows - len(clean_data)} rows with NaN bigrams")
    
    if len(clean_data) == 0:
        logger.error("No valid data remaining after removing NaN bigrams")
        return pd.DataFrame()
    
    # Calculate group sizes efficiently
    group_sizes = clean_data.groupby(['user_id', 'std_bigram_pair']).size()
    
    # Calculate consistency efficiently
    consistency_info = clean_data.groupby(['user_id', 'std_bigram_pair'])['chosenBigram'].apply(
        lambda x: len(set(x)) == 1 if len(x) > 1 else None
    ).reset_index()
    consistency_info.columns = ['user_id', 'std_bigram_pair', 'is_consistent']
    
    # Debug: Check columns before merge
    logger.debug(f"Clean data columns before merge: {list(clean_data.columns)}")
    logger.debug(f"Consistency info columns: {list(consistency_info.columns)}")
    
    # Merge back to original data
    result = clean_data.merge(consistency_info, on=['user_id', 'std_bigram_pair'], how='left')
    
    # Debug: Check columns after first merge
    logger.debug(f"Columns after consistency merge: {list(result.columns)}")
    
    # Add group sizes more efficiently using merge
    group_size_df = group_sizes.reset_index()
    group_size_df.columns = ['user_id', 'std_bigram_pair', 'group_size']
    
    # Debug: Check for existing group_size column
    if 'group_size' in result.columns:
        logger.warning("group_size column already exists, dropping it")
        result = result.drop(columns=['group_size'])
    
    result = result.merge(group_size_df, on=['user_id', 'std_bigram_pair'], how='left')
    
    # Debug: Check columns after group size merge
    logger.debug(f"Columns after group size merge: {list(result.columns)}")
    
    # Extract bigram1 and bigram2 from std_bigram_pair safely
    try:
        # Check if bigram1, bigram2 already exist
        if 'bigram1' in result.columns:
            result = result.drop(columns=['bigram1'])
        if 'bigram2' in result.columns:
            result = result.drop(columns=['bigram2'])
            
        bigram_pairs_df = pd.DataFrame(result['std_bigram_pair'].tolist(), 
                                     index=result.index, 
                                     columns=['bigram1', 'bigram2'])
        result = pd.concat([result, bigram_pairs_df], axis=1)
    except Exception as e:
        logger.error(f"Error extracting bigram pairs: {e}")
        # Fallback: extract manually
        result['bigram1'] = result['std_bigram_pair'].apply(lambda x: x[0] if isinstance(x, tuple) and len(x) >= 2 else None)
        result['bigram2'] = result['std_bigram_pair'].apply(lambda x: x[1] if isinstance(x, tuple) and len(x) >= 2 else None)
    
    # Debug: Check columns after bigram extraction
    logger.debug(f"Columns after bigram extraction: {list(result.columns)}")
    
    # Calculate bigram times efficiently with safe lookups
    def safe_bigram_time(row, bigram_num):
        try:
            target_bigram = row[f'bigram{bigram_num}']
            if pd.isna(target_bigram):
                return np.nan
            if row['chosenBigram'] == target_bigram:
                return row['chosenBigramTime']
            else:
                return row['unchosenBigramTime']
        except:
            return np.nan
    
    # Check if time columns already exist
    if 'bigram1_time' in result.columns:
        result = result.drop(columns=['bigram1_time'])
    if 'bigram2_time' in result.columns:
        result = result.drop(columns=['bigram2_time'])
    
    result['bigram1_time'] = result.apply(lambda row: safe_bigram_time(row, 1), axis=1)
    result['bigram2_time'] = result.apply(lambda row: safe_bigram_time(row, 2), axis=1)
    
    # Add abs_sliderValue
    if 'abs_sliderValue' in result.columns:
        result = result.drop(columns=['abs_sliderValue'])
    result['abs_sliderValue'] = result['sliderValue'].abs()
    
    # Rename columns carefully - check for duplicates first
    rename_mapping = {
        'chosenBigram': 'chosen_bigram',
        'unchosenBigram': 'unchosen_bigram',
        'chosenBigramTime': 'chosen_bigram_time',
        'unchosenBigramTime': 'unchosen_bigram_time',
        'chosenBigramCorrect': 'chosen_bigram_correct',
        'unchosenBigramCorrect': 'unchosen_bigram_correct',
        'std_bigram_pair': 'bigram_pair'
    }
    
    # Only rename columns that actually exist
    columns_to_rename = {k: v for k, v in rename_mapping.items() if k in result.columns}
    result = result.rename(columns=columns_to_rename)
    
    # Final debug: Check final columns
    logger.debug(f"Final columns: {list(result.columns)}")
    
    # Check for duplicate columns
    if len(result.columns) != len(set(result.columns)):
        logger.error(f"Duplicate columns found: {result.columns.tolist()}")
        # Remove duplicates by keeping only the first occurrence
        result = result.loc[:, ~result.columns.duplicated()]
        logger.warning(f"Removed duplicate columns, final columns: {list(result.columns)}")
    
    return result

def calculate_user_statistics(bigram_data: pd.DataFrame, easy_choice_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Calculate comprehensive user statistics."""
    user_stats = pd.DataFrame()
    user_stats['user_id'] = bigram_data['user_id'].unique()
    user_stats = user_stats.set_index('user_id')
    
    # Basic counts
    user_stats['total_choices'] = bigram_data['user_id'].value_counts()
    
    # Filter data for specific choice types
    consistent_choices = bigram_data[(bigram_data['is_consistent'] == True) & (bigram_data['group_size'] > 1)]
    inconsistent_choices = bigram_data[(bigram_data['is_consistent'] == False) & (bigram_data['group_size'] > 1)]
    probable_choices = bigram_data[bigram_data['is_probable'] == True]
    improbable_choices = bigram_data[bigram_data['is_improbable'] == True]
    
    user_stats['consistent_choices'] = consistent_choices['user_id'].value_counts()
    user_stats['inconsistent_choices'] = inconsistent_choices['user_id'].value_counts()
    user_stats['probable_choices'] = probable_choices['user_id'].value_counts()
    user_stats['improbable_choices'] = improbable_choices['user_id'].value_counts()
    
    # Calculate improbable choices made twice (consistently)
    pair_counts = improbable_choices.groupby(['user_id', 'bigram_pair']).size().reset_index(name='count')
    improbable_choices_2x = pair_counts[pair_counts['count'] == 2].groupby('user_id').size()
    user_stats['improbable_choices_2x'] = improbable_choices_2x.fillna(0).astype(int)
    
    # Additional statistics
    user_stats['total_consistency_choices'] = bigram_data[bigram_data['group_size'] > 1]['user_id'].value_counts()
    user_stats['num_easy_choice_pairs'] = len(easy_choice_pairs)
    
    # Fill NaN values and ensure integer types
    user_stats = user_stats.fillna(0).astype(int)
    return user_stats.reset_index()

def process_experiment_data(data: pd.DataFrame, easy_choice_pairs: List[Tuple[str, str]], 
                           output_tables_folder: str, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Main processing orchestrator for experiment data."""
    logger.info("____ Process data ____")
    
    # Temporarily enable debug logging for this function
    original_level = logger.level
    logger.setLevel(logging.INFO) #DEBUG)
    
    # Check initial data quality
    logger.info(f"Initial data shape: {data.shape}")
    logger.info(f"Columns: {list(data.columns)}")
    
    # Check for NaN values in key columns
    for col in ['chosenBigram', 'unchosenBigram']:
        if col in data.columns:
            nan_count = data[col].isna().sum()
            logger.info(f"NaN values in {col}: {nan_count} ({nan_count/len(data)*100:.1f}%)")
        else:
            logger.error(f"Required column {col} not found in data!")
    
    # Sample some data to see what's wrong
    logger.debug("Sample of data with potential issues:")
    sample_data = data[['chosenBigram', 'unchosenBigram']].head(10)
    logger.debug(f"\n{sample_data}")
    
    try:
        # Create standardized bigram pairs
        data_with_pairs = create_standardized_bigram_pairs(data)
        logger.debug(f"Data after creating pairs: {data_with_pairs.shape}")
        
        # Identify probable/improbable choices
        data_with_choices = identify_probable_improbable_choices(data_with_pairs, easy_choice_pairs)
        logger.debug(f"Data after identifying choices: {data_with_choices.shape}")
        
        # Calculate consistency efficiently
        bigram_data = calculate_choice_consistency(data_with_choices)
        logger.debug(f"Data after calculating consistency: {bigram_data.shape}")
        
        if len(bigram_data) == 0:
            logger.error("No data remaining after processing!")
            # Restore original log level
            logger.setLevel(original_level)
            return {}
        
        # Sort the data - but first check for duplicate columns
        logger.debug(f"Columns before sorting: {list(bigram_data.columns)}")
        if len(bigram_data.columns) != len(set(bigram_data.columns)):
            logger.error(f"Duplicate columns detected: {bigram_data.columns.tolist()}")
            bigram_data = bigram_data.loc[:, ~bigram_data.columns.duplicated()]
            logger.warning(f"Removed duplicates, now have: {list(bigram_data.columns)}")
        
        # Check if required columns exist
        required_cols = ['user_id', 'trialId', 'bigram_pair']
        missing_cols = [col for col in required_cols if col not in bigram_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns for sorting: {missing_cols}")
            logger.error(f"Available columns: {list(bigram_data.columns)}")
            # Try alternative column names
            if 'trialId' not in bigram_data.columns and 'trial_id' in bigram_data.columns:
                bigram_data = bigram_data.rename(columns={'trial_id': 'trialId'})
                logger.info("Renamed trial_id to trialId")
        
        bigram_data = bigram_data.sort_values(by=['user_id', 'trialId', 'bigram_pair']).reset_index(drop=True)
        
        # Create specific subsets using vectorized operations
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
        
        # Restore original log level
        logger.setLevel(original_level)
        
        return {
            'bigram_data': bigram_data,
            'consistent_choices': consistent_choices,
            'inconsistent_choices': inconsistent_choices,
            'probable_choices': probable_choices,
            'improbable_choices': improbable_choices,
            'user_stats': user_stats
        }
        
    except Exception as e:
        logger.error(f"Error in process_experiment_data: {e}")
        # Restore original log level
        logger.setLevel(original_level)
        raise

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
# Scoring Functions
#######################################
def calculate_consistency_score(slider_values: List[float]) -> float:
    """Calculate score when choices are consistent."""
    valid_values = [abs(x) for x in slider_values if not np.isnan(x)]
    return np.median(valid_values) if valid_values else np.nan

def calculate_mixed_choice_score(group: pd.DataFrame, bigram1: str, bigram2: str) -> Tuple[float, str, str]:
    """
    Calculate score when choices are mixed between bigrams.
    
    Returns:
    - score: calculated score
    - winner: winning bigram
    - loser: losing bigram
    """
    scores = group['sliderValue']
    chosen_bigrams = group['chosen_bigram']
    group_size = len(group)
    
    # Calculate sums for each bigram
    valid_scores1 = [abs(x) for i, x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram1 and not np.isnan(x)]
    valid_scores2 = [abs(x) for i, x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram2 and not np.isnan(x)]
    
    sum1 = sum(valid_scores1)
    sum2 = sum(valid_scores2)
    
    score = abs(sum1 - sum2) / group_size if group_size > 0 else np.nan
    
    if sum1 >= sum2:
        winner, loser = bigram1, bigram2
    else:
        winner, loser = bigram2, bigram1
    
    return score, winner, loser

def determine_score(group: pd.DataFrame) -> pd.Series:
    """
    Determine the score and chosen/unchosen bigrams for a group of trials.
    
    Scoring algorithm:
    - If consistent choices: take median absolute slider value
    - If mixed choices: subtract sums of absolute values, divide by group size
    - Final score is absolute value divided by 100 (max slider value)
    """
    if len(group) == 0:
        return _create_empty_score_series()
    
    bigram1, bigram2 = group['bigram1'].iloc[0], group['bigram2'].iloc[0]
    chosen_bigrams = group['chosen_bigram']
    slider_values = group['sliderValue']
    group_size = len(group)
    
    # Check for invalid bigrams
    if pd.isna(bigram1) or pd.isna(bigram2):
        return _create_empty_score_series()
    
    # Determine scoring method based on choice consistency
    if len(set(chosen_bigrams)) == 1:
        # Consistent choices
        median_abs_slider_value = calculate_consistency_score(slider_values)
        chosen_bigram_winner = chosen_bigrams.iloc[0]
        unchosen_bigram_winner = bigram2 if chosen_bigram_winner == bigram1 else bigram1
    else:
        # Mixed choices
        median_abs_slider_value, chosen_bigram_winner, unchosen_bigram_winner = calculate_mixed_choice_score(
            group, bigram1, bigram2)
    
    # Calculate final score (normalized to 0-1 range)
    score = median_abs_slider_value / 100 if not np.isnan(median_abs_slider_value) else np.nan
    
    return pd.Series({
        'bigram1': bigram1,
        'bigram2': bigram2,
        'chosen_bigram_winner': chosen_bigram_winner,
        'unchosen_bigram_winner': unchosen_bigram_winner,
        'chosen_bigram_time_median': _safe_median(group['chosen_bigram_time']),
        'unchosen_bigram_time_median': _safe_median(group['unchosen_bigram_time']),
        'chosen_bigram_correct_total': group['chosen_bigram_correct'].sum(skipna=True),
        'unchosen_bigram_correct_total': group['unchosen_bigram_correct'].sum(skipna=True),
        'score': score,
        'text': tuple(group['text'].unique()),
        'is_consistent': (len(set(chosen_bigrams)) == 1),
        'is_probable': group['is_probable'].iloc[0],
        'is_improbable': group['is_improbable'].iloc[0],
        'group_size': group_size
    })

def _create_empty_score_series() -> pd.Series:
    """Create an empty score series for invalid groups."""
    return pd.Series({
        'bigram1': np.nan, 'bigram2': np.nan,
        'chosen_bigram_winner': np.nan, 'unchosen_bigram_winner': np.nan,
        'chosen_bigram_time_median': np.nan, 'unchosen_bigram_time_median': np.nan,
        'chosen_bigram_correct_total': 0, 'unchosen_bigram_correct_total': 0,
        'score': np.nan, 'text': (), 'is_consistent': False,
        'is_probable': False, 'is_improbable': False, 'group_size': 0
    })

def _safe_median(series: pd.Series) -> float:
    """Safely calculate median, handling NaN values."""
    clean_series = series.dropna()
    return clean_series.median() if len(clean_series) > 0 else np.nan

def score_user_choices_by_slider_values(filtered_users_data: Dict[str, pd.DataFrame], 
                                       output_tables_folder: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Score each user's choices by slider values and create a modified copy of bigram_data.
    
    Algorithm:
    - Consistent choices: median absolute slider value
    - Mixed choices: difference of sums divided by number of choices
    - Final score: absolute value normalized to 0-1 range
    """
    logger.info("____ Score choices by slider values ____")
    bigram_data = filtered_users_data['bigram_data']
    
    # Check for null values
    null_counts = bigram_data.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Found null values in data: {null_counts[null_counts > 0].to_dict()}")
    
    # Group by user_id and bigram_pair and apply scoring
    grouped = bigram_data.groupby(['user_id', 'bigram_pair'], group_keys=False)
    scored_bigram_data = grouped.apply(determine_score, include_groups=False).reset_index()
    
    # Reorder columns
    column_order = [
        'user_id', 'bigram_pair', 'bigram1', 'bigram2',
        'chosen_bigram_winner', 'unchosen_bigram_winner', 
        'chosen_bigram_time_median', 'unchosen_bigram_time_median',
        'chosen_bigram_correct_total', 'unchosen_bigram_correct_total', 
        'score', 'text', 'is_consistent', 'is_probable', 'is_improbable', 'group_size'
    ]
    scored_bigram_data = scored_bigram_data[column_order]
    
    logger.info(f"Original bigram_data rows: {len(bigram_data)}")
    logger.info(f"Scored data rows (unique user_id and bigram_pair combinations): {len(scored_bigram_data)}")
    
    # Save scored data
    output_file = os.path.join(output_tables_folder, 'scored_bigram_data.csv')
    scored_bigram_data.to_csv(output_file, index=False)
    logger.info(f"Scored bigram data saved to {output_file}")
    
    return scored_bigram_data

#######################################
# Winner Determination Functions  
#######################################
def determine_overall_winner(group: pd.DataFrame) -> pd.Series:
    """
    Determine the winning bigram across all users for a bigram pair.
    
    Algorithm:
    - If all users chose same bigram: median score
    - If mixed user choices: difference of score sums divided by number of users
    """
    if len(group) == 0:
        return _create_empty_winner_series()
    
    bigram1, bigram2 = group['bigram1'].iloc[0], group['bigram2'].iloc[0]
    
    if pd.isna(bigram1) or pd.isna(bigram2):
        return _create_empty_winner_series()
    
    scores = group['score']
    chosen_bigrams = group['chosen_bigram_winner']
    
    # Check if any bigram was chosen
    bigram1_chosen = any(chosen_bigrams == bigram1)
    bigram2_chosen = any(chosen_bigrams == bigram2)
    
    if not bigram1_chosen and not bigram2_chosen:
        return _create_empty_winner_series()
    
    unique_chosen_bigrams = chosen_bigrams.unique()
    
    if len(unique_chosen_bigrams) == 1:
        # All users chose the same bigram
        valid_scores = [abs(x) for x in scores if not np.isnan(x)]
        median_score = np.median(valid_scores) if valid_scores else np.nan
        mad_score = median_abs_deviation(valid_scores) if valid_scores else np.nan
        winner_bigram = unique_chosen_bigrams[0]
        loser_bigram = bigram2 if winner_bigram == bigram1 else bigram1
        bigram1_wins = (winner_bigram == bigram1)
    else:
        # Mixed choices across users
        sum1 = sum(abs(x) for i, x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram1 and not np.isnan(x))
        sum2 = sum(abs(x) for i, x in enumerate(scores) if chosen_bigrams.iloc[i] == bigram2 and not np.isnan(x))
        median_score = abs(sum1 - sum2) / len(group) if len(group) > 0 else np.nan
        
        # Calculate MAD for mixed choices
        mad_list = []
        for i, x in enumerate(scores):
            if not np.isnan(x):
                if chosen_bigrams.iloc[i] == bigram1:
                    mad_list.append(abs(x))
                elif chosen_bigrams.iloc[i] == bigram2:
                    mad_list.append(-abs(x))
        
        mad_score = median_abs_deviation(mad_list) if mad_list else np.nan
        
        if sum1 >= sum2:
            winner_bigram, loser_bigram = bigram1, bigram2
            bigram1_wins = True
        else:
            winner_bigram, loser_bigram = bigram2, bigram1
            bigram1_wins = False
    
    # Calculate aggregate statistics
    winner_stats = _calculate_winner_statistics(group, bigram1_wins)
    
    return pd.Series({
        'winner_bigram': winner_bigram,
        'loser_bigram': loser_bigram,
        'median_score': median_score,
        'mad_score': mad_score,
        'chosen_bigram_time_median': winner_stats['time_median'],
        'unchosen_bigram_time_median': winner_stats['unchosen_time_median'],
        'chosen_bigram_correct_total': winner_stats['correct_total'],
        'unchosen_bigram_correct_total': winner_stats['unchosen_correct_total'],
        'is_consistent': group['is_consistent'].all(),
        'is_probable': group['is_probable'].iloc[0],
        'is_improbable': group['is_improbable'].iloc[0],
        'group_size': group['group_size'].sum(),
        'text': tuple(group['text'].unique())
    })

def _create_empty_winner_series() -> pd.Series:
    """Create an empty winner series for invalid groups."""
    return pd.Series({
        'winner_bigram': np.nan, 'loser_bigram': np.nan,
        'median_score': np.nan, 'mad_score': np.nan,
        'chosen_bigram_time_median': np.nan, 'unchosen_bigram_time_median': np.nan,
        'chosen_bigram_correct_total': 0, 'unchosen_bigram_correct_total': 0,
        'is_consistent': False, 'is_probable': False, 'is_improbable': False,
        'group_size': 0, 'text': ()
    })

def _calculate_winner_statistics(group: pd.DataFrame, bigram1_wins: bool) -> Dict[str, float]:
    """Calculate aggregate statistics for the winning bigram."""
    bigram1, bigram2 = group['bigram1'].iloc[0], group['bigram2'].iloc[0]
    chosen_bigrams = group['chosen_bigram_winner']
    
    # Calculate medians for each bigram
    time_median1 = _safe_median_for_bigram(group, 'chosen_bigram_time_median', chosen_bigrams, bigram1)
    time_median2 = _safe_median_for_bigram(group, 'chosen_bigram_time_median', chosen_bigrams, bigram2)
    unchosen_time_median1 = _safe_median_for_bigram(group, 'unchosen_bigram_time_median', chosen_bigrams, bigram1)
    unchosen_time_median2 = _safe_median_for_bigram(group, 'unchosen_bigram_time_median', chosen_bigrams, bigram2)
    
    # Calculate totals for each bigram
    correct_total1 = _sum_for_bigram(group, 'chosen_bigram_correct_total', chosen_bigrams, bigram1)
    correct_total2 = _sum_for_bigram(group, 'chosen_bigram_correct_total', chosen_bigrams, bigram2)
    unchosen_correct_total1 = _sum_for_bigram(group, 'unchosen_bigram_correct_total', chosen_bigrams, bigram1)
    unchosen_correct_total2 = _sum_for_bigram(group, 'unchosen_bigram_correct_total', chosen_bigrams, bigram2)
    
    if bigram1_wins:
        return {
            'time_median': time_median1,
            'unchosen_time_median': unchosen_time_median1,
            'correct_total': correct_total1,
            'unchosen_correct_total': unchosen_correct_total1
        }
    else:
        return {
            'time_median': time_median2,
            'unchosen_time_median': unchosen_time_median2,
            'correct_total': correct_total2,
            'unchosen_correct_total': unchosen_correct_total2
        }

def _safe_median_for_bigram(group: pd.DataFrame, column: str, chosen_bigrams: pd.Series, target_bigram: str) -> float:
    """Calculate median for a specific bigram choice."""
    values = [x for i, x in enumerate(group[column]) if chosen_bigrams.iloc[i] == target_bigram and not np.isnan(x)]
    return np.median(values) if values else np.nan

def _sum_for_bigram(group: pd.DataFrame, column: str, chosen_bigrams: pd.Series, target_bigram: str) -> float:
    """Calculate sum for a specific bigram choice."""
    values = [x for i, x in enumerate(group[column]) if chosen_bigrams.iloc[i] == target_bigram]
    return np.sum(values)

def choose_bigram_winners(scored_bigram_data: pd.DataFrame, output_tables_folder: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Determine winning bigram for each bigram pair across all users.
    
    Algorithm:
    - All users same choice: median score
    - Mixed user choices: difference of score sums divided by number of users
    """
    logger.info("____ Choose bigram winners ____")
    
    initial_pairs = len(scored_bigram_data['bigram_pair'].unique())
    logger.info(f"Initial number of unique bigram pairs: {initial_pairs}")
    
    # Clean data before processing
    scored_data_clean = scored_bigram_data.dropna(subset=['bigram1', 'bigram2', 'chosen_bigram_winner'])
    logger.info(f"Rows before cleaning: {len(scored_bigram_data)}")
    logger.info(f"Rows after cleaning: {len(scored_data_clean)}")
    
    # Group by bigram_pair and determine winners
    grouped = scored_data_clean.groupby(['bigram_pair'], group_keys=False)
    bigram_winner_data = grouped.apply(determine_overall_winner, include_groups=False).reset_index()
    
    # Reorder columns
    column_order = [
        'bigram_pair', 'winner_bigram', 'loser_bigram', 'median_score', 'mad_score', 
        'chosen_bigram_time_median', 'unchosen_bigram_time_median',
        'chosen_bigram_correct_total', 'unchosen_bigram_correct_total',
        'is_consistent', 'is_probable', 'is_improbable', 'text'
    ]
    bigram_winner_data = bigram_winner_data[column_order]
    
    logger.info(f"Total rows in scored_bigram_data: {len(scored_bigram_data)}")
    logger.info(f"Total rows in bigram_winner_data (unique bigram_pairs): {len(bigram_winner_data)}")
    
    # Save results
    output_file = os.path.join(output_tables_folder, 'bigram_winner_data.csv')
    bigram_winner_data.to_csv(output_file, index=False)
    logger.info(f"Bigram winner data saved to {output_file}")
    
    return bigram_winner_data

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
    
    # Sort users by consistent choices
    user_order = user_stats.sort_values('consistent_choices', ascending=False)['user_id']
    
    # Create consistent vs inconsistent choices plot
    _create_choice_comparison_plot(
        user_stats, user_order, 
        ['consistent_choices', 'inconsistent_choices'],
        ['Consistent Choices', 'Inconsistent Choices'],
        [colors.get('primary', '#1f77b4'), colors.get('secondary', '#ff7f0e')],
        'Consistent vs. Inconsistent Choices per User',
        os.path.join(output_plots_folder, f'{plot_label}consistent_vs_inconsistent_choices.png'),
        figsize, dpi
    )
    
    # Create probable vs improbable choices plot
    _create_choice_comparison_plot(
        user_stats, user_order,
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
    data = user_stats.set_index('user_id').loc[user_order, columns]
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
    initial_bigram_pairs = len(raw_data.apply(lambda row: tuple(sorted([row['chosenBigram'], row['unchosenBigram']])), axis=1).unique())
    
    logger.info(f"Initial data - Users: {initial_users}, Trials: {initial_trials}, Unique bigram pairs: {initial_bigram_pairs}")
    
    # Step 2: Process data without filtering
    processed_data = process_experiment_data(raw_data, easy_choice_pairs, output_tables_folder, logger)
    
    # Initial visualizations
    create_user_choice_visualizations(processed_data['user_stats'], output_plots_folder, config, "initial_", logger)
    
    # Step 3: Filter users
    filtered_users_data, valid_users = apply_all_user_filters(raw_data, processed_data['user_stats'], config, easy_choice_pairs, logger)
    
    # Step 4: Filter rows
    filtered_data = filter_data_rows(filtered_users_data, config, logger)
    
    # Step 5: Process filtered data
    final_processed_data = process_experiment_data(filtered_data, easy_choice_pairs, output_tables_folder, logger)
    
    final_trials = len(final_processed_data['bigram_data'])
    final_pairs = len(final_processed_data['bigram_data']['bigram_pair'].unique())
    final_users = len(final_processed_data['bigram_data']['user_id'].unique())
    
    logger.info(f"Final data after filtering - Users: {final_users}, Trials: {final_trials}, Unique bigram pairs: {final_pairs}")
    logger.info(f"Trials removed: {initial_trials - final_trials}")
    
    # Generate final visualizations
    create_user_choice_visualizations(final_processed_data['user_stats'], output_plots_folder, config, "filtered_", logger)
    
    # Step 6: Score choices by slider values
    scored_data = score_user_choices_by_slider_values(final_processed_data, output_tables_folder, logger)
    
    # Step 7: Choose winning bigrams
    bigram_winner_data = choose_bigram_winners(scored_data, output_tables_folder, logger)
    
    logger.info("Bigram data processing completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())