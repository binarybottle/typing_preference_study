Typing Preference Study summary data analysis

Author: Arno Klein (arnoklein.info)
GitHub repository: binarybottle/bigram-typing-preference-study
License: Apache v2.0 

## Overview

This pipeline analyzes relationships between bigram typing times, user preferences, 
frequency patterns, and keyboard ergonomics principles. 
The analysis is implemented in `analyze_speed_frequencies_choices.py` 
and consists of four main components:

1. Typing Time vs. User Preference Analysis
2. Typing Time vs. Frequency Analysis 
3. Preference Prediction Analysis
4. Keyboard Ergonomics Statistical Tests

## Output Directory Structure

```
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
├── preference_prediction/
│   ├── speed_accuracy_by_magnitude.png
│   ├── speed_accuracy_by_confidence.png
│   ├── user_accuracy_distribution.png
│   ├── preference_prediction_report.txt
│   ├── below_chance_analysis_report.txt
│   ├── variance_prediction_analysis.txt
│   └── bigram_pair_choices.csv
└── keyboard_ergonomics/
    └── keyboard_ergonomics_report.txt
```

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

### 4. Keyboard Ergonomics Statistical Tests

This comprehensive analysis tests biomechanical hypotheses about keyboard ergonomics 
using participants' consistent bigram preferences. The analysis provides statistically 
robust evidence for keyboard layout optimization principles by testing five key research questions:

#### Research Questions and Statistical Tests

**Question 1: Row Preferences**
- **Row preference hierarchy**: Is the home row preferred over top/bottom rows?
- **Row comparison**: Is the top row preferred over the bottom row?
- **Finger-specific preferences**: Which fingers prefer which rows for non-home positions?
- **Statistical approach**: Binomial tests comparing proportions of choices favoring each row
- **Design implications**: Guides placement of frequent keys by row and finger

**Question 2: Row Movement Patterns**
- **Same vs. different rows**: Are same-row bigrams preferred over cross-row movements?
- **Adjacent vs. skip movements**: Are adjacent-row movements (home↔top, home↔bottom) preferred over skip movements (top↔bottom)?
- **Statistical approach**: Binomial tests for movement pattern preferences
- **Design implications**: Minimizes uncomfortable cross-row sequences in layout optimization

**Question 3: Column Preferences**
- **Sequential column comparisons**: Tests preferences between adjacent columns (5vs4, 4vs3, 3vs2, 2vs1)
- **Column 5 avoidance**: Comprehensive test of whether column 5 is systematically avoided vs. all other columns (1-4)
- **Individual column vs. column 5**: Separate tests for each column (1,2,3,4) vs. column 5
- **Statistical approach**: Binomial tests with multiple comparison correction
- **Design implications**: Establishes column preference hierarchy for frequent letter placement

**Question 4: Finger Distance and Coordination**
- **Adjacent vs. remote fingers**: Tests preference for adjacent finger movements vs. larger stretches
- **Separation distance effects**: Compares 1-finger, 2-finger, and 3-finger separations
- **Row-specific analysis**: Separate tests for same-row vs. cross-row finger movements
- **Statistical approach**: Categorical comparison of finger distance preferences
- **Design implications**: Optimizes finger coordination patterns in common bigrams

**Question 5: Movement Direction**
- **Directional preferences**: Tests preferences for inward vs. outward finger sequences
- **Finger sequence patterns**: Compares low-to-high finger numbers vs. high-to-low sequences
- **Statistical approach**: Binomial test for directional movement preferences
- **Design implications**: Guides optimization of finger roll patterns

#### Statistical Methods

**Core Approach**:
- **Binomial tests** for binary preference comparisons
- **Participant-level random effects** to account for individual differences  
- **Multiple comparison correction** using False Discovery Rate (FDR) or Bonferroni methods
- **Effect size reporting** via proportion differences and confidence intervals

**Data Quality Controls**:
- **Consistent choices only**: Analysis limited to bigram pairs with consistent participant responses
- **Sufficient sample sizes**: Minimum comparison thresholds for reliable statistical testing
- **Outlier handling**: Robust statistical methods resistant to extreme values

**Validation Approach**:
- **Cross-dataset replication**: Results can be validated across different participant samples
- **Effect size thresholds**: Distinguishes statistical significance from practical importance
- **Comprehensive reporting**: All statistical tests reported with corrections applied

#### Output Files

**keyboard_ergonomics_report.txt**:
- Detailed statistical results for all five research questions
- P-values, effect sizes, and significance tests with multiple comparison corrections
- Proportion estimates with confidence intervals
- Complete test-by-test breakdown of results

#### Integration with Layout Optimization

The ergonomics analysis provides weighted constraints for multi-objective keyboard layout optimization:

**High Priority Constraints** (strong statistical evidence):
- Maximize home row usage for frequent letters
- Minimize cross-row bigram sequences
- Favor middle column (column 3) for most frequent letters
- Optimize adjacent finger movements for common bigrams

**Medium Priority Constraints** (moderate evidence):
- Pinky preference for bottom row over top row when leaving home
- Adjacent row movements preferred over skip movements
- Avoid extreme columns (1,5) for very frequent letters

**Low Priority Constraints** (weak evidence):
- Slight preference for inward finger movement patterns

### Overall Analysis Strategy

The four analyses work together to:
1. Validate whether subjective preferences align with objective performance
2. Separate the effects of familiarity (frequency) from inherent typing comfort
3. Test whether typing speed could serve as an objective measure of preference
4. **Establish evidence-based ergonomic principles for keyboard layout design**

This comprehensive approach helps understand:
- The reliability of subjective preference ratings
- The role of practice vs. biomechanics in typing comfort
- The potential for using objective measures in future typing comfort studies
- **Statistically robust guidelines for optimizing keyboard layouts**

## Setup and Configuration

### Prerequisites
- Python 3.x
- Required packages: numpy, pandas, scipy, sklearn, matplotlib, seaborn, PyYAML, statsmodels
- Input data in CSV format with required columns (see Data Format section)
- keymaps.py file with keyboard position definitions

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
  
  # Ergonomics analysis settings
  run_ergonomics_tests: Enable/disable ergonomics analysis (default: true)
  alpha_level: Significance level for statistical tests (default: 0.05)
  correction_method: Multiple comparison correction ('fdr_bh' or 'bonferroni')
  
visualization:
  dpi: Plot resolution
  figsize: Default figure dimensions
  colors: Color scheme for plots

subdirs:
  typing_time_vs_preference: "typing_time_vs_preference"
  typing_time_vs_frequency: "typing_time_vs_frequency"
  preference_prediction: "preference_prediction"
  keyboard_ergonomics: "keyboard_ergonomics"
```

## Input Data

### Required Input Columns
- `user_id`: Unique participant identifier
- `chosen_bigram`: Selected bigram in trial
- `unchosen_bigram`: Non-selected bigram in trial
- `chosen_bigram_time`: Typing time for chosen bigram (ms)
- `unchosen_bigram_time`: Typing time for unchosen bigram (ms)
- `sliderValue`: User preference rating (-100 to 100)
- `is_consistent`: Boolean indicating consistent choice across repeated presentations

### Keyboard Mapping Requirements
- `keymaps.py`: Defines keyboard layout with row_map, column_map, and finger_map dictionaries
- Maps each key to its row (1-3), column (1-10), and finger assignment (1-4)
- Required for ergonomics analysis to determine key positions and finger assignments

## Output Data

### Generated Files Structure

The pipeline generates the following files in the configured output directories:

1. **Typing Time vs. Preference Analysis** (`typing_time_vs_preference/`)
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

2. **Frequency Analysis** (`typing_time_vs_frequency/`)
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

3. **Prediction Analysis** (`preference_prediction/`)
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

4. **Keyboard Ergonomics Analysis** (`keyboard_ergonomics/`)
   - `keyboard_ergonomics_report.txt`
     * Comprehensive statistical results for all ergonomics research questions
     * Includes p-values, effect sizes, confidence intervals, and significance tests
     * Multiple comparison correction results
     * Detailed breakdown of row, column, and movement pattern preferences

### Statistical Measures and Interpretations

1. **Normalization Metrics**
   - **Within-participant Z-scores**: (x - median) / MAD
     * MAD (Median Absolute Deviation) provides robust scale estimate
     * Controls for individual typing speed differences
     * Allows comparison across participants

2. **Correlation Measures**
   - **Spearman's Rank Correlation**: Used for frequency-time relationships
     * Non-parametric measure robust to outliers
     * Captures monotonic relationships
     * Range: [-1, 1], where:
       * 1 indicates perfect positive correlation
       * -1 indicates perfect negative correlation
       * 0 indicates no correlation
   
3. **Regression Statistics**
   - **R-squared**: Variance explained by the model
     * Range: [0, 1], higher values indicate better fit
   - **Slope**: Rate of change in typing time with log-frequency
   - **McFadden's Pseudo-R²**: For logistic regression models
     * Interpretation differs from linear R²
     * Values above 0.2 considered good fit

4. **Confidence Intervals**
   - **Bootstrap CI**: Non-parametric 95% confidence intervals
     * Based on 1000 resamples
     * Robust to non-normal distributions
   - **Standard Error**: Uncertainty in mean estimates
     * Calculated using bootstrap for robustness

5. **Group Comparisons**
   - **ANOVA F-statistic**: Tests for differences between frequency groups
     * Larger F-values indicate stronger group differences
   - **Post-hoc Tests**: Bonferroni-corrected pairwise comparisons
     * Controls family-wise error rate
     * Reports adjusted p-values

6. **Prediction Metrics**
   - **Accuracy**: Percentage of correct predictions
   - **ROC-AUC**: Area under ROC curve
     * Range: [0.5, 1.0]
     * 0.5 indicates chance-level prediction
     * Above 0.7 considered good discrimination
   - **Odds Ratios**: Effect size for logistic regression
     * Values > 1 indicate positive association
     * Values < 1 indicate negative association

7. **Ergonomics Statistical Tests**
   - **Binomial Tests**: Tests proportion differences from chance (0.5)
     * Two-tailed tests for bidirectional hypotheses
     * Reports exact p-values for small samples
   - **Effect Sizes**: Absolute deviation from 0.5 proportion
     * Range: [0, 0.5], larger values indicate stronger preferences
   - **Multiple Comparison Correction**: 
     * FDR (False Discovery Rate): Controls expected proportion of false positives
     * Bonferroni: Conservative family-wise error rate control
   - **Confidence Intervals**: Bootstrap-based 95% CIs for proportions
     * Provides uncertainty estimates for effect sizes

8. **Robust Statistics**
   - **Median**: Central tendency resistant to outliers
   - **MAD**: Scale measure resistant to outliers
   - **Winsorization**: Extreme value handling at 3000ms
   - **Quantile-based Analysis**: Distribution-free comparisons

## Usage Example

```bash
# Run complete analysis including ergonomics tests
python analyze_speed_frequencies_choices.py --config config.yaml

# Configuration file should include:
# analysis:
#   run_ergonomics_tests: true
#   alpha_level: 0.05
#   correction_method: 'fdr_bh'
```
