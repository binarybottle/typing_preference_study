# Data Processing Pipeline

Data processing pipeline for filtering participants, scoring preferences, and preparing data for analysis while maintaining statistical integrity.

## Overview

`process_summary_data.py` transforms raw experimental data into analysis-ready datasets through systematic filtering, quality control, and preference scoring. The pipeline prioritizes data quality and statistical validity over sample size maximization.

## Pipeline Architecture
Raw Data → Load & Combine → Filter Users → Filter Rows → Score Preferences → Choose Winners
↓           ↓              ↓            ↓             ↓                ↓
CSV Files   Combined Data   Valid Users   Clean Data   Scored Data    Winner Data

## Core Principles

**Data Quality First**: Rigorous filtering over large sample sizes  
**No Fallbacks**: Only process complete, valid data  
**Statistical Integrity**: Preserve true preference signals  
**Transparency**: Detailed logging of all filtering decisions  

## Input Requirements

### File Structure
input_folder/
├── subfolder1/
│   ├── participant1.csv
│   ├── participant2.csv
│   └── ...
├── subfolder2/
│   └── ...
└── bigram_tables/
└── bigram_easy_choice_pairs_LH_nocomments.csv

### Required Columns
- user_id               # Auto-generated from filename
- chosenBigram         # Selected bigram in trial
- unchosenBigram       # Non-selected bigram
- chosenBigramTime     # Typing time for chosen (ms)
- unchosenBigramTime   # Typing time for unchosen (ms)
- sliderValue          # Preference rating (-100 to 100)
- trialId              # Trial identifier (excludes intro trials)

### Easy Choice Pairs
CSV file defining "improbable" choices for quality control:
- good_choice,bad_choice
- fr,vr
- st,zt

## Processing Stages

### Stage 1: Data Loading and Combination

**File Discovery**: Recursive CSV file search across subdirectories  
**User ID Generation**: One file = one participant (filename becomes user_id)  
**Group Assignment**: Subfolder name becomes group_id  
**Intro Filtering**: Removes `intro-trial-1` entries  

**Quality Checks**:
- Empty file detection
- Column validation  
- Numeric conversion with error handling
- Duplicate filename warnings

### Stage 2: Data Standardization

**Bigram Pair Creation**: Standardized tuple representation `(bigram1, bigram2)` where `bigram1 < bigram2`  
**Choice Marking**: Identifies probable/improbable choices based on easy choice pairs  
**Consistency Calculation**: Determines if users make consistent choices for repeated bigram pairs  

**Key Transformations**:
```python
    # Standardized bigram pair
    std_bigram_pair = tuple(sorted([chosen, unchosen]))
```

# Consistency per user-pair
is_consistent = len(set(chosen_bigrams)) == 1 for repeat presentations

### Stage 3: User-Level Filtering
- Approach: Remove participants showing patterns inconsistent with genuine preference expression
- Filter 1: Improbable Choices
  - Behavior: Removes users making ≥2 improbable choices for the same bigram pair
  - Rationale: Consistently choosing "bad" over "good" bigrams suggests random responding
  - Configuration: improbable_threshold (default: 0)
- Filter 2: Strong Inconsistencies
  - Behavior: Removes users with >X% inconsistent choices where |sliderValue| > threshold
  - Rationale: Strong preferences (far from zero) should be consistent
  - Calculation: (strong_inconsistent_pairs / total_multi_pairs) * 100
  - Configuration: strong_inconsistent_threshold (%), zero_threshold (slider distance)
- Filter 3: Slider Behavior Patterns
  - Two-part filter identifying problematic response patterns:
  - Part A - Consecutive Streaks (Percentage-based):
    - Calculates streak threshold as percentage of user's total trials
    - Flags users with streaks ≥ threshold where slider > distance_close_to_zero
    - Configuration: streak_side_percent (% of trials), distance_close_to_zero
  - Part B - Frequent Zero Values:
    - Counts trials where |slider| ≤ distance_close_to_zero
    - Removes users where this percentage ≥ percent_close_to_zero
    - Configuration: percent_close_to_zero (%), distance_close_to_zero

### Stage 4: Row-Level Filtering
Approach: Remove trials that don't provide valid preference information
- Filter 1: Letter Exclusion
  - Purpose: Remove bigrams containing specified problematic letters
  - Configuration: filter_letters list
  - Application: Both chosen and unchosen bigrams checked
- Filter 2: Single Presentations
  - Purpose: Remove bigram pairs shown only once per user
  - Rationale: Consistency analysis requires repeat presentations
  - Method: Keep only user-bigram pairs with ≥2 presentations
- Filter 3: Inconsistent Away-from-Zero
  - Purpose: Remove inconsistent choices with strong slider values
  - Logic: If choices are inconsistent AND any |sliderValue| > threshold, remove entire user-bigram group
  - Exception: Preserve if all slider values are near zero (≤ threshold)

### Stage 5: Preference Scoring
- Approach: Extract preference strength from slider values, handling both consistent and mixed choices
- Consistent Choices (Same bigram chosen every time)
```python
    score = median(abs(slider_values)) / 100  # Normalized to [0,1]
    winner = consistently_chosen_bigram
```

- Mixed Choices (Different bigrams chosen)
```python
    # Calculate sum of absolute slider values for each bigram
    sum1 = sum(abs(slider) for slider where chosen == bigram1)
    sum2 = sum(abs(slider) for slider where chosen == bigram2) 
    score = abs(sum1 - sum2) / group_size / 100
    winner = bigram1 if sum1 >= sum2 else bigram2
```

- Additional Metrics
  - Typing Time: Median times for chosen/unchosen bigrams
  - Accuracy: Total correct responses per bigram
  - Consistency Flag: Boolean indicating choice pattern
  - Sample Size: Number of presentations

### Stage 6: Winner Determination
- Purpose: Determine population-level bigram preferences across all users
- Consistent Winners (All users choose same bigram)

```python
    median_score = median(abs(user_scores))
    mad_score = median_absolute_deviation(user_scores)
    winner = unanimously_chosen_bigram
```

- Mixed Winners (Users disagree)
```python
    # Sum absolute scores by bigram choice
    sum1 = sum(abs(score) for users choosing bigram1)
    sum2 = sum(abs(score) for users choosing bigram2)
    median_score = abs(sum1 - sum2) / total_users
    winner = bigram1 if sum1 >= sum2 else bigram2
```


## Configuration

### Essential Settings
```yaml
process:
  # User filtering
  filter_users_by_num_improbable_choices: true
  improbable_threshold: 0
  
  filter_users_by_percent_strong_inconsistencies: true  
  strong_inconsistent_threshold: 20.0  # percent
  zero_threshold: 20  # slider distance
  
  # Slider behavior filtering
  streak_side_percent: 20  # percent of trials for streak threshold
  percent_close_to_zero: 25  # percent threshold for zero values
  distance_close_to_zero: 10  # slider distance defining "close to zero"
  
  # Row filtering
  filter_letters: ['y', 'h']  # exclude these letters
  filter_single_presentations: true
  filter_inconsistent_choices_away_from_zero: true
```

### Logging Configuration
```yaml
logging:
  level: 'INFO'  # DEBUG for detailed output
  format: '%(asctime)s - %(levelname)s - %(message)s'
  file: 'processing.log'  # optional log file
```


## Quality Assurance

### Data Validation
- Column Checking: Required fields validation
- Range Validation: Slider values [-100, 100], positive times
- Consistency Verification: Internal relationship checks
- Sample Size Monitoring: Adequate data warnings

Processing Integrity
- No Imputation: Missing data excluded rather than filled
- Conservative Filtering: Err on side of data quality
- Detailed Logging: All filtering decisions documented
- Replication Support: Complete parameter recording

Statistical Safeguards
- Minimum Samples: Skip analysis below threshold sizes
- Outlier Handling: Robust statistics (median, MAD)
- Bootstrap Ready: Prepared for non-parametric analysis
- Cross-Validation: Consistent choice verification


## Usage Examples

### Basic Processing
```bash
    python process_summary_data.py --config config.yaml
```

