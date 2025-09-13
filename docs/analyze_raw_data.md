# Raw Data Analysis

Analysis of keystroke timing and error patterns from raw typing data to evaluate performance differences between keys and key combinations.


## Overview

`analyze_raw_data.py` processes raw keystroke data to extract typing performance metrics for individual keys and bigrams (key pairs). The analysis focuses on the left home block keys and their mirror pairs, providing comprehensive performance statistics with frequency adjustments.


## Scope

**Target Keys**: Left home block - Q, W, E, R, A, S, D, F, Z, X, C, V  
**Mirror Analysis**: Comparison with right-side equivalents (P, O, I, U, ;, L, K, J, /, ., ,, M)  
**Bigram Focus**: Space-flanked bigrams (space-key1-key2-space) excluding same-key repetitions


## Features

### 1. Dual Analysis Framework
- **1-Key Statistics**: Individual keys as the second key in bigrams
- **2-Key Statistics**: Complete bigram combinations
- **Mirror Pair Analysis**: Left vs. right hand performance comparison

### 2. Frequency Adjustment
- **Log-transformed Frequency**: Accounts for non-linear frequency effects
- **Regression-based Adjustment**: Removes frequency bias from performance metrics
- **R² Reporting**: Variance explained by frequency effects

### 3. Robust Statistical Methods
- **Median-based Metrics**: Resistant to outliers
- **MAD (Median Absolute Deviation)**: Robust variability measure
- **Bootstrap Error Bars**: Non-parametric confidence intervals


## Input Requirements

### Data Format
CSV files with required columns:
user_id          # Unique participant identifier
trialId          # Trial identifier
expectedKey      # Target key
typedKey         # Actually typed key
isCorrect        # Boolean accuracy flag
keydownTime      # Timestamp (milliseconds)

### Frequency Data (Optional)
- `input/frequency/english-letter-counts-google-ngrams.csv`
- `input/frequency/english-letter-pair-counts-google-ngrams.csv`


## Analysis Methods

### Performance Metrics

**Error Rate Calculation**:
- 1-Key: Errors when typing the second key of a bigram (first key correct)
- 2-Key: Errors when typing the second key of a bigram sequence
- Excludes cascading errors from first key mistakes

**Timing Calculation**:
- 1-Key: Time from first key to second key (both correct)
- 2-Key: Time between consecutive keys in bigram (both correct)
- Range filter: 50-2000ms (excludes outliers)

**Error Pattern Analysis**:
- Most common mistyped key for each target
- Error frequency by target key
- Spatial error patterns

### Frequency Adjustment Process

1. **Data Collection**: Letter/bigram frequencies from Google n-grams
2. **Log Transformation**: `log10(frequency + 1)` for non-linear modeling
3. **Regression Models**: `metric ~ log_frequency` using OLS
4. **Residual Calculation**: `adjusted_metric = metric - predicted + mean`
5. **R² Reporting**: Variance explained by frequency

### Statistical Validation

**Bootstrap Confidence Intervals**:
- 1000 resamples for error rate uncertainty
- Non-parametric approach for non-normal distributions

**Correlation Analysis**:
- Pearson correlation for error rates vs. typing times
- Significance testing with multiple comparison awareness


## Output Files

### CSV Results
| File | Content |
|------|---------|
| `left_home_key_analysis.csv` | Individual key statistics with frequency adjustment |
| `left_home_bigram_analysis.csv` | Bigram performance with frequency adjustment |
| `mirror_pair_analysis.csv` | Left vs. right comparison statistics |
| `key_accuracy_statistics.csv` | Accuracy scores for all keys |
| `key_speed_statistics.csv` | Speed scores for all keys |
| `bigram_accuracy_statistics.csv` | Bigram accuracy metrics |
| `bigram_speed_statistics.csv` | Bigram speed metrics |
| `bigram_composite_speed_statistics.csv` | Combined key + transition timing |
| `instantaneous_error_rate_1key.csv` | Error rate / typing time ratios |
| `instantaneous_error_rate_2key.csv` | Bigram error rate / time ratios |

### Visualizations

**Individual Key Analysis**:
- `error_1key.png`: Error rates with MAD error bars
- `typing_time_1key.png`: Typing times with MAD error bars  
- `error_vs_typing_time_1key.png`: Speed-accuracy correlation

**Bigram Analysis**:
- `count_bigram.png`: Occurrence frequencies
- `error_2key.png`: Error rates (non-zero only)
- `typing_time_2key.png`: Typing times with error bars
- `error_vs_typing_time_2key.png`: Speed-accuracy trade-offs

**Comparative Analysis**:
- `error_1key_vs_2key.png`: Individual vs. bigram error comparison
- `typing_time_1key_vs_2key.png`: Individual vs. bigram timing comparison

**Mirror Pair Analysis**:
- `mirror_pair_error_rates_with_mad.png`: Left vs. right error rates
- `mirror_pair_typing_time.png`: Left vs. right timing comparison
- `mirror_pair_error_left_vs_right.png`: Scatter plot correlation
- `mirror_pair_time_left_vs_right.png`: Timing correlation with frequency adjustment


## Metrics Explained

### Error Rate
**Definition**: Proportion of incorrect keystrokes for target key
**Calculation**: `errors / total_attempts`
**Adjustment**: Frequency-adjusted using regression residuals

### Typing Time  
**Definition**: Inter-keystroke interval (milliseconds)
**Robust Measure**: Median time (resistant to outliers)
**Adjustment**: Log-frequency regression for familiarity effects

### Speed Score
**Definition**: Keystrokes per second (`1000 / typing_time`)
**Error Propagation**: Speed MAD ≈ `(time_MAD / time²) × 1000`
**Frequency-Adjusted**: Based on adjusted typing times

### Composite Speed
**Definition**: `1 / (first_key_time + transition_time)`
**Error Combination**: `√(MAD₁² + MAD₂²)` for independent measurements
**Purpose**: Models complete bigram typing including key preparation

### Instantaneous Error Rate
**Definition**: `error_rate / typing_time` 
**Interpretation**: Errors per unit time (higher = worse efficiency)
**Error Propagation**: `f × √((σₓ/x)² + (σᵧ/y)²)` for ratio metrics


## Usage

### Basic Execution
```bash
    python analyze_raw_data.py
```

### Configuration
Modify script constants:
```python
    LEFT_HOME_KEYS = ['q', 'w', 'e', 'r', 'a', 's', 'd', 'f', 'z', 'x', 'c', 'v']
    csv_path = 'input/raw_data/nonProlific_studies/*.csv'  
    letter_freq_path = 'input/frequency/english-letter-counts-google-ngrams.csv'
    bigram_freq_path = 'input/frequency/english-letter-pair-counts-google-ngrams.csv'
    frequency_cutoff = 1000000000  # Minimum frequency threshold
```


## Technical Notes

### Data Processing
- Trial Filtering: Excludes intro trials (trialId != 'intro-trial-1')
- Sequence Detection: Identifies space-flanked bigrams automatically
- Error Handling: Robust to missing data and malformed entries

### Statistical Considerations
- Minimum Sample Size: 10 occurrences for reliable statistics
- Outlier Handling: 50-2000ms timing range filter
- Missing Data: Excluded from analysis rather than imputed

### Frequency Data Integration
- Source: Peter Norvig's Google n-gram analysis
- Normalization: Consistent scaling across datasets
- Fallback: Analysis continues without frequency data if files missing


## Interpretation Guidelines

### Error Rate Patterns
- Home Row Advantage: Lower error rates for middle row keys
- Finger Strength: Index > middle > ring > pinky error patterns
- Spatial Errors: Adjacent key substitutions most common

### Timing Patterns
- Frequency Effects: More frequent bigrams typed faster
- Biomechanical Constraints: Row changes increase timing
- Individual Differences: High inter-participant variability

### Mirror Pair Findings
- Hand Dominance: Slight right-hand advantage for most keys
- Frequency Confound: Adjustment reveals biomechanical effects
- Positional Effects: Column position matters more than left/right


## Quality Assurance

### Data Validation
- Connectivity Checks: Ensures all key pairs have comparison data
- Sample Size Verification: Reports data completeness
- Range Validation: Flags extreme values for review

### Statistical Robustness
- Multiple Initializations: Bradley-Terry model with multiple starting points
- Convergence Checking: Optimization success verification
- Bootstrap Validation: Confidence interval reliability

### Output Verification
- Cross-checks: Consistency between 1-key and 2-key analyses
- Correlation Validation: Expected relationships between metrics
- Frequency Adjustment: R² values for model quality assessment