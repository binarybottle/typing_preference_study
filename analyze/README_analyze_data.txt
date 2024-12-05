# README for analyze_data.py
Bigram Typing Analysis Pipeline Documentation

## Overview

This pipeline analyzes relationships between bigram typing times, user preferences, 
and frequency patterns. The analysis is implemented in `analyze_data.py` 
and consists of three main components:

1. Typing Time vs. User Preference Analysis
2. Typing Time vs. Frequency Analysis 
3. Preference Prediction Analysis

.
├── typing_time_vs_preference
│   ├── typing_time_vs_preference_scatter.png
│   ├── typing_time_difference_histogram.png
│   ├── typing_time_vs_preference_summary_stats.txt
│   └── typing_time_vs_preference_confidence_intervals.txt
├── typing_time_vs_frequency
│   ├── frequency_vs_typing_time_scatter.png
│   ├── frequency_bins_typing_time_bar.png
│   ├── typing_time_vs_frequency_summary_stats.txt
│   ├── typing_time_vs_frequency_confidence_intervals.txt
│   ├── typing_time_vs_frequency_regression_stats.txt
│   ├── typing_time_vs_frequency_distribution_plot.png
│   ├── frequency_vs_typing_time_group_stats.txt
│   └── frequency_vs_typing_time_relationship_plot.png
└── preference_prediction
    ├── prediction_accuracy_by_confidence.png
    ├── user_accuracy_distribution_histogram.png
    ├── preference_prediction_summary_stats.txt
    └── preference_prediction_accuracy_report.txt

## Analysis Components

### 1. Typing Time vs. User Preference Analysis

This analysis investigates whether participants' stated preferences 
(via slider values) align with their typing performance. 
The core question is: Do people prefer bigrams they can type faster? 
This helps validate whether typing speed could serve as an objective measure 
of typing comfort or preference.

#### Key Plots and Their Purpose

1. Time Difference vs. Slider Value Plots
   - Raw and normalized versions
   - **Why**: To test whether stronger preferences (higher absolute slider values) 
     correspond to larger differences in typing times. If people's stated preferences 
     reflect their typing ability, we should see larger timing differences for stronger preferences.

2. Time Difference Histograms
   - **Why**: To examine whether timing differences are systematically different 
     for different preference strengths. This helps identify if the relationship between 
     preference and typing speed is consistent across different levels of preference strength.

3. Overlaid Time Histograms
   - **Why**: To directly compare the distributions of typing times for chosen vs. unchosen bigrams. 
     If preferences align with performance, chosen bigrams should show systematically faster typing times.

### 2. Typing Time vs. Frequency Analysis

This analysis examines whether more frequent bigrams are typed faster, which would suggest 
that practice and exposure influence typing performance. Understanding this relationship 
helps separate the effects of familiarity from inherent biomechanical comfort.

#### Key Plots and Their Purpose

1. Distribution Plot
   - **Why**: To visualize how typing times vary across the full range of bigram frequencies 
     while accounting for sample size and variability. This helps identify whether 
     the frequency-speed relationship is consistent across the frequency spectrum.

2. Median Times Plot
   - **Why**: To identify specific bigrams that deviate from the general frequency-speed relationship, 
     which might indicate biomechanical factors beyond simple familiarity.

3. Minimum Times Plot
   - **Why**: To examine optimal performance across frequencies, helping isolate 
     the role of mechanical constraints from practice effects.

### 3. Preference Prediction Analysis

This analysis tests whether typing speed differences could predict user preferences 
without explicit ratings. Success here would suggest typing speed could be used as an objective 
measure of typing comfort, potentially replacing subjective ratings in future studies.

#### Key Plots and Their Purpose

1. Accuracy by Magnitude Plot
   - **Why**: To determine if larger speed differences lead to more reliable predictions. 
   This helps establish the minimum speed difference needed for reliable preference prediction.

2. Accuracy by Confidence Plot
   - **Why**: To examine whether stronger stated preferences (higher confidence) correspond 
   to more accurate speed-based predictions. This helps validate the relationship between 
   subjective certainty and objective performance.

3. User Distribution Plot
   - **Why**: To understand individual differences in the relationship between preferences 
   and typing speed. This helps identify whether speed-based prediction is equally reliable 
   across different users.

### Overall Analysis Strategy

The three analyses work together to:
1. Validate whether subjective preferences align with objective performance
2. Separate the effects of familiarity (frequency) from inherent typing comfort
3. Test whether typing speed could serve as an objective measure of preference

This comprehensive approach helps understand:
- The reliability of subjective preference ratings
- The role of practice vs. biomechanics in typing comfort
- The potential for using objective measures in future typing comfort studies

## Setup and Configuration

### Prerequisites
- Python 3.x
- Required packages: numpy, pandas, scipy, sklearn, matplotlib, seaborn, PyYAML
- Input data in CSV format with required columns (see Data Format section)

### Configuration
The analysis is controlled via `config.yaml` with the following key sections:

```yaml
data:
  input_dir: Path to input directory
  filtered_data_file: Path to filtered data CSV

analysis:
  min_trials_per_participant: Minimum trials required per participant (default: 10)
  outlier_threshold_sd: Standard deviation threshold for outliers (default: 3.0)
  max_time_ms: Maximum allowed typing time in milliseconds (default: 3000)
  
visualization:
  dpi: Plot resolution
  figsize: Default figure dimensions
  colors: Color scheme for plots

output:
  base_dir: Base output directory
  subdirs: Subdirectories for different analysis types
```

## Data Format

### Required Input Columns
- `user_id`: Unique participant identifier
- `chosen_bigram`: Selected bigram in trial
- `unchosen_bigram`: Non-selected bigram in trial
- `chosen_bigram_time`: Typing time for chosen bigram (ms)
- `unchosen_bigram_time`: Typing time for unchosen bigram (ms)
- `sliderValue`: User preference rating (-100 to 100)

## Statistical Methods

### Normalization
- Per-participant normalization using median and median absolute deviation (MAD)
- Formula: `(x - median(x)) / (MAD(x) + 1e-10)`

### Confidence Intervals
- Bootstrap method with 1000 resamples
- Default confidence level: 95%

### Outlier Handling
- Time limit clipping at configurable threshold (default: 3000ms)
- Robust statistics to minimize outlier impact

## Error Handling and Validation

The pipeline includes comprehensive error checking:
- Data validation for required columns
- Range checking for slider values (-100 to 100)
- Handling of missing or invalid numeric values
- Participant exclusion based on minimum trial counts
- Detailed logging of data quality issues

## Known Limitations

1. Frequency Analysis:
   - Relies on pre-calculated bigram frequencies
   - May not account for domain-specific frequency patterns

2. Prediction Analysis:
   - Assumes typing speed differences are meaningful predictors
   - May be sensitive to individual typing patterns

3. Statistical Power:
   - Accuracy of confidence intervals depends on sample sizes
   - Small participant groups may have less reliable estimates

## Usage Example

```bash
python analyze_data.py --config config.yaml
```

## Extending the Pipeline

To add new analyses:
1. Create new methods in the `BigramAnalysis` class
2. Add corresponding configuration in `config.yaml`
3. Update output directory structure
4. Implement appropriate error handling and logging
5. Add visualization methods as needed

## Version History

Current Version: 1.0
- Initial implementation of three core analyses
- Comprehensive visualization suite
- Robust statistical methodology