# Multi-Objective Optimization (MOO) Analysis

Analysis framework for extracting keyboard layout optimization objectives from bigram preference data using statistically robust methods focused on practical significance.

## Overview

`analyze_objectives.py` identifies and quantifies distinct typing mechanics objectives to inform multi-objective keyboard layout optimization. The analysis emphasizes effect sizes and confidence intervals over statistical significance, providing engineering-focused insights for keyboard design.

## Practical Significance Focus

**Primary Goal**: Extract actionable design constraints for keyboard optimization  
**Statistical Approach**: Effect sizes + confidence intervals, with statistical tests for detectability  
**Engineering Focus**: Results inform layout decisions rather than scientific hypothesis testing  
**No Multiple Correction**: Each objective addresses distinct typing mechanics

## MOO Objectives Framework

### Core Objectives

| Objective | Method | Purpose |
|-----------|--------|---------|
| **Key Preferences** | Bradley-Terry + Pairwise | Individual key quality ranking |
| **Row Separation** | Proportion Analysis | Same vs. reach vs. hurdle preferences |
| **Column Separation** | Controlled Comparison | Adjacent vs. distant finger movements |
| **Column 4 vs 5** | Binary Choice | Index finger column preferences |

### Theoretical Foundation

**1. Key Quality (Individual Merit)**
- **Same-letter Bigrams**: Pure key quality (AA vs QQ)
- **Pairwise Comparisons**: Detailed preference maps
- **Bradley-Terry Model**: Robust ranking with confidence intervals

**2. Movement Biomechanics**
- **Row Separation**: Vertical movement penalties
- **Column Separation**: Horizontal stretch preferences  
- **Finger Coordination**: Multi-finger sequence optimization

## Analysis Methods

### 1. Key Preferences: Same-Letter Bigrams

**Approach**: Bradley-Terry model on same-letter bigram comparisons (AA vs QQ, etc.)

**Why Same-Letter?**: Isolates key quality from bigram-specific effects

**Statistical Model**:

Strength_i ~ Bradley-Terry with regularization λ = 1e-4

P(key_i preferred over key_j) = exp(strength_i) / (exp(strength_i) + exp(strength_j))

Robustness Features:
- Multiple Initialization: Tries random, zero, and data-driven starting points
- Numerical Stability: Clipping, regularization, sum-to-zero constraint
- Connectivity Validation: Ensures comparison graph is connected
- Bootstrap CI: Non-parametric confidence intervals

Output Interpretation:
- Strength > 0: Above-average key preference
- Ranking: Ordered list with confidence intervals
- Effect Sizes: Practical significance of differences

### 2. Key Preferences: Pairwise Comparisons

Approach: Direct analysis of specific key pair preferences

Target Pairs:
```python
# Home row relationships
('f','d'), ('d','s'), ('s','a')

# Cross-row patterns  
('f','r'), ('d','e'), ('s','w'), ('a','q')

# Within-row sequences
('r','e'), ('w','q'), ('c','x'), ('x','z')
```

Statistical Testing:
- Wilson Score CI: Robust proportion confidence intervals
- Z-tests: Significance testing vs. 50% baseline
- Minimum Sample: n ≥ 10 for reliable estimates

### 3. Row Separation Preferences
Research Question: Do people prefer same-row > reach (1 apart) > hurdle (2 apart)?

Methods:
- Same vs. 1-Apart: Home row vs. adjacent row comparisons
- 1-Apart vs. 2-Apart: Reach vs. hurdle movement comparisons
- Controlled Analysis: Only compare pairs with identical row separation

Statistical Framework:
- Overall Proportion: Preference for smaller row separation
- Breakdown Analysis: Separate tests by comparison type
- Effect Sizes: Departure from 50% baseline
- Bootstrap CI: Uncertainty quantification

### 4. Column Separation Preferences
Research Question: Do people prefer adjacent columns over distant columns?

Methods:
- Row Control: Only compare bigrams with identical row separation
- Column Distance: Compare 1-apart vs. 2-apart column separations
- Same-Column Exclusion: Remove 0-distance comparisons

Pattern Analysis:
- Same Row: Pure horizontal movement
- Reach (1-Apart): Diagonal reach movements
- Hurdle (2-Apart): Cross-row stretch patterns

### 5. Column 4 vs 5 Preferences
Research Question: Do people prefer index finger column (RFV) over column 5 (TGB)?

Methods:
- Column Classification: Based on key presence in bigrams
- Pure Comparisons: Column 4 bigrams vs. Column 5 bigrams only
- Mixed Exclusion: Ignore bigrams containing both column types

Engineering Relevance: Index finger usage optimization for frequent sequences


## Statistical Implementation

Bootstrap Confidence Intervals
- Method: Non-parametric resampling (1000 iterations)
- Application: All proportion estimates and Bradley-Terry strengths
- Advantages: No distributional assumptions, robust to outliers

Frequency Weighting
- Purpose: Balance comparison frequencies across bigram pairs
- Method: Inverse frequency weighting for equal statistical power

Implementation: weight = 1 / frequency_count per comparison
- Bradley-Terry Optimization
- Objective Function: Regularized maximum likelihood
```python
    LL = Σ[wins_ij * log(P_ij) + wins_ji * log(1 - P_ij)] - λ * Σ(strength_i²)
```

Optimization Strategy:
- L-BFGS-B: First attempt with box constraints
- SLSQP: Fallback with sum-to-zero constraint
- Fallback: Simple win-rate estimation if optimization fails

Numerical Stability:
- Strength Clipping: [-10, 10] range
- Probability Regularization: [λ, 1-λ] bounds
- Log-Sum-Exp: Overflow prevention


## Configuration Options

### Analysis Parameters
```yaml
moo_objectives_analysis:
  bootstrap_iterations: 1000        # Bootstrap precision
  confidence_level: 0.95           # CI level  
  regularization: 1e-4             # Bradley-Terry regularization
  frequency_weighting_method: 'inverse_frequency'
```

### Quality Controls
```yaml
# Minimum data requirements
min_comparisons: 10               # Per pairwise test
min_same_letter_pairs: 5          # For Bradley-Terry
min_row_separation_instances: 50   # For row analysis
```


## Results Interpretation

### Effect Size Guidelines

Bradley-Terry Strengths:
- |strength| < 0.1: Negligible difference
- |strength| 0.1-0.3: Small effect
- |strength| 0.3-0.5: Medium effect
- |strength| > 0.5: Large effect

Proportion Departures:
- |p - 0.5| < 0.05: Negligible preference
- |p - 0.5| 0.05-0.15: Small preference
- |p - 0.5| 0.15-0.30: Medium preference
- |p - 0.5| > 0.30: Strong preference

Confidence Interval Interpretation
- Overlapping CI: Uncertain difference, design flexibility
- Non-overlapping CI: Clear preference, strong constraint
- CI Width: Precision of estimate, sample size effect

MOO Integration Weights
- High Priority (CI excludes 0.5, large effect):
  - Strong row preferences: Weight = 1.0
  - Clear key rankings: Weight = 1.0
- Medium Priority (CI marginally excludes 0.5):
  - Moderate column preferences: Weight = 0.5-0.8
- Low Priority (CI includes 0.5, small effect):
  - Weak finger patterns: Weight = 0.1-0.3


## Quality Assurance

Data Validation
- Instance Counting: Sufficient sample sizes per objective
- Comparison Coverage: All key pairs represented
- Consistency Checking: Internal relationship validation

Statistical Robustness
- Convergence Monitoring: Optimization success rates
- Bootstrap Stability: CI consistency across runs
- Outlier Resistance: Median-based summaries where appropriate

Cross-Study Validation
- Ranking Correlation: Spearman's ρ between studies
- CI Overlap: Replication assessment
- Effect Size Stability: Practical significance consistency

## Usage Examples

### Basic Analysis
```bash
python analyze_objectives.py \
  --data processed_consistent_choices.csv \
  --output moo_results/ \
  --config config.yaml
```
