# Preference Analysis

Analysis of relationships between typing speed, user preferences, bigram frequency, and keyboard ergonomics principles.


## Overview

`analyze_frequency_speed_preference.py` examines four relationships in typing preference data to understand the connections between objective performance and subjective comfort ratings. This analysis validates whether typing speed can serve as an objective measure of typing preference.


## Research Questions

| Question | Purpose | Statistical Method |
|----------|---------|-------------------|
| **1. Choice and Frequency** | Do people choose more frequent bigrams? | Proportion tests, correlation analysis |
| **2. Speed and Choice Strength** | Do people make stronger choices faster? | Spearman correlation, binned analysis |
| **3. Speed and Frequency** | Do people type frequent bigrams faster? | Spearman correlation, quantile analysis |
| **4. Speed and Choice Prediction** | Can typing speed predict preferences? | Logistic regression, ROC analysis |


## Input Requirements

### Data Columns
- user_id                 # Unique participant identifier
- chosen_bigram           # Selected bigram in comparison
- unchosen_bigram         # Non-selected bigram
- chosen_bigram_time      # Typing time for chosen bigram (ms)
- unchosen_bigram_time    # Typing time for unchosen bigram (ms)
- sliderValue             # Preference rating (-100 to 100)

### Frequency Data
- **Source**: Google n-gram bigram frequencies
- **Format**: CSV with `item_pair` and `score` columns
- **Preprocessing**: Normalized scores for consistent scaling

### Configuration
Essential `config.yaml` settings:
```yaml
analysis:
  bootstrap_iterations: 1000
  confidence_level: 0.95
  min_bigram_occurrences: 5
  min_trials_per_participant: 10
  n_quantiles: 5
  frequency_weighting_method: 'inverse_frequency'
```


## Analysis Framework

### 1. Choice and Frequency Analysis
Core Question: Do participants systematically choose more frequent bigrams?

Methods:
- Extract bigram pairs where frequency data exists for both options
- Calculate proportion choosing more frequent bigram
- Test against chance (50%) using binomial test
- Correlate frequency difference with choice strength (slider values)

Key Metrics:
- Proportion Test: Binomial test vs. 50% baseline
- Wilson Score CI: Robust confidence intervals for proportions
- Correlation: Spearman's ρ between log frequency difference and slider values

Expected Pattern: If frequency drives preference, we expect >50% choosing frequent bigrams and positive correlation between frequency difference and choice strength.

### 2. Speed and Choice Strength Analysis
Core Question: Do participants make stronger preference judgments faster?

Methods:
- Calculate average typing time per choice: (chosen_time + unchosen_time) / 2
- Use absolute slider values as choice strength measure
- Apply Spearman correlation (robust to outliers)
- Bin analysis by choice strength quantiles

Statistical Approach:
- Within-participant normalization: Controls for individual speed differences
- Quantile binning: Reveals non-linear relationships
- Bootstrap confidence intervals: Non-parametric uncertainty estimation

Expected Pattern: If confidence relates to speed, stronger choices (higher |slider|) should correlate with faster typing.

### 3. Speed and Frequency Analysis
Core Question: Do participants type more frequent bigrams faster?

Methods:
- Aggregate typing times per bigram across all presentations
- Use median times (robust to outliers) per bigram
- Correlate with log-transformed frequency
- Quantile analysis across frequency spectrum

Quality Controls:
- Minimum occurrences: Exclude bigrams with <5 observations
- Outlier handling: Median-based statistics
- Sample size weighting: Larger samples get higher confidence

Visualization Strategy:
- Scatter plots: Individual bigram data points
- Quantile analysis: Systematic patterns across frequency range
- Regression lines: Trend identification with confidence bands

### 4. Speed and Choice Prediction Analysis
Core Question: Can typing speed differences predict user preferences without explicit ratings?

Methods:
- Speed difference: chosen_time - unchosen_time
- Prediction rule: Speed difference < 0 → chosen bigram preferred
- Overall accuracy: Proportion of correct predictions
- Magnitude analysis: Accuracy by speed difference size

Analysis:
- Per-participant models: Individual prediction accuracy
- Logistic regression: Continuous prediction modeling
- ROC analysis: Discrimination ability assessment
- Cross-validation: Generalization testing

Practical Significance: High accuracy would support using objective timing for preference measurement.


## Statistical Methods

Robust Statistics
- Spearman Correlations: Non-parametric, outlier-resistant
- Median-based Measures: Central tendency robust to extremes
- MAD (Median Absolute Deviation): Scale estimation
- Bootstrap Confidence Intervals: Distribution-free uncertainty

Multiple Comparison Handling
- Research Questions: Each addresses distinct relationships
- No Correction Applied: Questions are theory-driven, not exploratory
- Effect Size Focus: Practical significance over statistical significance

Normalization Strategies
- Within-participant Z-scores: (x - median) / MAD
- Frequency weighting: Inverse frequency for balanced analysis
- Log transformations: Handle skewed frequency distributions


## Output Structure

preference_analysis/
├── choice_frequency_proportion.png              # Q1: Frequency choice analysis
├── frequency_choice_strength_correlation.png    # Q1: Correlation plots
├── speed_choice_strength_correlation.png        # Q2: Speed vs. strength
├── speed_by_strength_category.png              # Q2: Binned analysis
├── frequency_speed_analysis.png                # Q3: Two-panel analysis
├── speed_choice_prediction_analysis.png        # Q4: Four-panel analysis
└── comprehensive_analysis_report.txt           # Complete results


## Visualizations
Choice-Frequency Analysis:
- Proportion plot: Success/failure with confidence intervals
- Correlation plot: Frequency difference vs. choice strength with regression

Speed-Strength Analysis:
- Correlation scatter: Individual data points with trend line
- Category bars: Speed by choice strength quantiles with error bars

Frequency-Speed Analysis:
- Main scatter: Frequency vs. time with size-coded sample sizes
- Quantile bars: Average times by frequency categories

Prediction Analysis:
- Overall accuracy: Bar chart with confidence intervals
- Participant distribution: Histogram of individual accuracies
- Speed-strength correlation: Validation of relationship
- Magnitude analysis: Accuracy by speed difference size


## Findings Interpretation

### Expected Relationships
Strong Positive Evidence (ρ > 0.3, p < 0.01):
- Frequency-speed correlation: Frequent bigrams typed faster
- Speed-prediction accuracy >60%: Objective measures predict preferences

### Moderate Evidence (ρ > 0.1, p < 0.05):
- Choice strength-speed correlation: Confident choices made faster
- Frequency-choice preference >55%: Slight bias toward familiar bigrams

### Weak/No Evidence (ρ < 0.1, p > 0.05):
- Random patterns suggesting no systematic relationship
- Individual differences dominating group effects

### Cross-Study Validation Metrics
- Correlation stability: ρ within ±0.1 across studies
- Proportion replication: Overlapping confidence intervals
- Effect size consistency: Cohen's conventions for practical significance


## Configuration Parameters

### Analysis Settings
```yaml
analysis:
  max_time_ms: 3000                    # Outlier threshold
  min_trials_per_participant: 10       # Inclusion criterion
  min_bigram_occurrences: 5           # Reliability threshold
  n_quantiles: 5                      # Binning resolution
  bootstrap_iterations: 1000          # CI precision
  confidence_level: 0.95              # Statistical confidence
```

### Visualization Settings
```yaml
visualization:
  figsize: [10, 6]                    # Plot dimensions
  dpi: 300                            # Resolution
  style: 'seaborn-v0_8-whitegrid'    # Plot style
```


## Usage Examples

### Basic Analysis
```bash
python analyze_frequency_speed_preference.py --config config.yaml
```

### Custom Configuration
```bash
# High-precision analysis
python analyze_frequency_speed_preference.py --config high_precision_config.yaml

# Quick exploratory analysis  
python analyze_frequency_speed_preference.py --config quick_config.yaml
```

### Integration with Pipeline
```bash
# Full pipeline
python process_summary_data.py --config config.yaml
python analyze_frequency_speed_preference.py --config config.yaml
python analyze_objectives.py --data processed_consistent_choices.csv
```


## Technical Implementation

### Data Validation
- Column checking: Required fields validation
- Range validation: Slider values [-100, 100], positive times
- Completeness: Missing data handling and reporting

### Frequency Integration
- Bigram lookup: Automatic frequency assignment
- Missing data: Graceful handling of unmapped bigrams
- Normalization: Consistent scaling across frequency sources

### Statistical Computing
- Correlation calculation: Robust methods with significance testing
- Bootstrap implementation: Efficient resampling with fixed seeds
- Model fitting: Logistic regression with cross-validation

### Memory Management
- Chunk processing: Handle large datasets efficiently
- Selective analysis: Skip invalid/incomplete data
- Progress reporting: Status updates for long-running analysis


## Quality Assurance

### Data Quality Checks
- Outlier detection: Extreme timing values flagged
- Consistency validation: Choice-slider alignment verification
- Sample size monitoring: Adequate data for reliable statistics

### Statistical Validation
- Bootstrap stability: Confidence interval consistency across runs
- Correlation robustness: Pearson vs. Spearman comparison
- Model validation: Cross-validation for prediction accuracy

### Output Verification
- Range checking: All statistics within expected bounds
- Consistency testing: Internal relationships verification
- Replication support: Detailed reporting for cross-study validation