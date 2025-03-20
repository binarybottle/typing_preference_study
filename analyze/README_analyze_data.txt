# README for analyze_data.py
Bigram Typing Analysis Pipeline Documentation

Author: Arno Klein (arnoklein.info)

GitHub repository: binarybottle/bigram-typing-comfort-experiment

License: Apache v2.0 

## Overview

This pipeline analyzes relationships between bigram typing times, user preferences, 
and frequency patterns. The analysis is implemented in `analyze_data.py` 
and consists of three main components:

1. Typing Time vs. User Preference Analysis
2. Typing Time vs. Frequency Analysis 
3. Preference Prediction Analysis

.
├── typing_time_vs_preference/
│   ├── raw_typing_time_diff_vs_slider_value.png
│   ├── normalized_typing_time_diff_vs_slider_value.png
│   ├── raw_typing_time_diff_vs_slider_value_histograms.png
│   ├── normalized_typing_time_diff_vs_slider_value_histograms.png
│   ├── raw_overlaid_typing_times_by_slider_value_histograms.png
│   └── normalized_overlaid_typing_times_by_slider_value_histograms.png
├── typing_time_vs_frequency/
│   ├── frequency_and_timing_distribution.png
│   ├── frequency_and_timing_median.png
│   ├── frequency_and_timing_minimum.png
│   ├── freq_vs_time_raw.png
│   ├── freq_vs_time_normalized.png
│   └── frequency_timing_analysis_results.txt
└── preference_prediction/
    ├── speed_accuracy_by_magnitude.png
    ├── speed_accuracy_by_confidence.png
    ├── user_accuracy_distribution.png
    ├── preference_prediction_report.txt
    ├── below_chance_analysis_report.txt
    ├── variance_prediction_analysis.txt
    └── bigram_pair_choices.csv

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

## Input Data

### Required Input Columns
- `user_id`: Unique participant identifier
- `chosen_bigram`: Selected bigram in trial
- `unchosen_bigram`: Non-selected bigram in trial
- `chosen_bigram_time`: Typing time for chosen bigram (ms)
- `unchosen_bigram_time`: Typing time for unchosen bigram (ms)
- `sliderValue`: User preference rating (-100 to 100)

## Output Data

### Generated Files Structure

The pipeline generates the following files in the configured output directories:

1. Typing Time vs. Preference Analysis (`typing_time_vs_preference/`)
   - `raw_typing_time_diff_vs_slider_value.png`
     * Scatter plot of raw time differences vs. slider values
     * Shows relationship between typing speed differences and preference strength
   - `normalized_typing_time_diff_vs_slider_value.png`
     * Normalized version of the time difference plot
     * Controls for individual typing speed variations
   - `raw_typing_time_diff_vs_slider_value_histograms.png`
     * Distribution of typing time differences across preference ranges
   - `normalized_typing_time_diff_vs_slider_value_histograms.png`
     * Distribution of normalized time differences
   - `raw_overlaid_typing_times_by_slider_value_histograms.png`
     * Comparison of chosen vs. unchosen typing time distributions
   - `normalized_overlaid_typing_times_by_slider_value_histograms.png`
     * Normalized version of overlaid distributions

2. Frequency Analysis (`typing_time_vs_frequency/`)
   - `frequency_and_timing_distribution.png`
     * Main distribution plot with error bars and sample sizes
     * Shows relationship between frequency and typing speed
   - `frequency_and_timing_median.png`
     * Median typing times with bigram labels
   - `frequency_and_timing_minimum.png`
     * Fastest typing times for each bigram
   - `frequency_timing_analysis_results.txt`
     * Detailed statistical analysis results
     * Includes correlations, ANOVA results, and group statistics

3. Prediction Analysis (`preference_prediction/`)
   - `speed_accuracy_by_magnitude.png`
     * Prediction accuracy across speed difference magnitudes
   - `speed_accuracy_by_confidence.png`
     * Accuracy levels for different confidence ranges
   - `user_accuracy_distribution.png`
     * Distribution of per-participant prediction accuracy
   - `preference_prediction_report.txt`
     * Comprehensive prediction analysis results
   - `below_chance_analysis_report.txt`
     * Detailed analysis of unexpected patterns
   - `variance_prediction_analysis.txt`
     * Variance explained by different factors

### Statistical Measures and Interpretations

1. Normalization Metrics
   - **Within-participant Z-scores**: (x - median) / MAD
     * MAD (Median Absolute Deviation) provides robust scale estimate
     * Controls for individual typing speed differences
     * Allows comparison across participants

2. Correlation Measures
   - **Spearman's Rank Correlation**: Used for frequency-time relationships
     * Non-parametric measure robust to outliers
     * Captures monotonic relationships
     * Range: [-1, 1], where:
       * 1 indicates perfect positive correlation
       * -1 indicates perfect negative correlation
       * 0 indicates no correlation
   
3. Regression Statistics
   - **R-squared**: Variance explained by the model
     * Range: [0, 1], higher values indicate better fit
   - **Slope**: Rate of change in typing time with log-frequency
   - **McFadden's Pseudo-R²**: For logistic regression models
     * Interpretation differs from linear R²
     * Values above 0.2 considered good fit

4. Confidence Intervals
   - **Bootstrap CI**: Non-parametric 95% confidence intervals
     * Based on 1000 resamples
     * Robust to non-normal distributions
   - **Standard Error**: Uncertainty in mean estimates
     * Calculated using bootstrap for robustness

5. Group Comparisons
   - **ANOVA F-statistic**: Tests for differences between frequency groups
     * Larger F-values indicate stronger group differences
   - **Post-hoc Tests**: Bonferroni-corrected pairwise comparisons
     * Controls family-wise error rate
     * Reports adjusted p-values

6. Prediction Metrics
   - **Accuracy**: Percentage of correct predictions
   - **ROC-AUC**: Area under ROC curve
     * Range: [0.5, 1.0]
     * 0.5 indicates chance-level prediction
     * Above 0.7 considered good discrimination
   - **Odds Ratios**: Effect size for logistic regression
     * Values > 1 indicate positive association
     * Values < 1 indicate negative association

7. Robust Statistics
   - **Median**: Central tendency resistant to outliers
   - **MAD**: Scale measure resistant to outliers
   - **Winsorization**: Extreme value handling at 3000ms
   - **Quantile-based Analysis**: Distribution-free comparisons

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
