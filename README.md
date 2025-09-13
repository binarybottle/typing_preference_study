# Bigram Typing Preference Study - Analysis Pipeline
 
Comprehensive data collection and analysis pipeline for keyboard ergonomics research.

Author: Arno Klein (arnoklein.info)

GitHub repository: binarybottle/typing_preferences_study

License: Apache v2.0 


## Overview

This repository provides a complete analysis pipeline for studying keyboard typing preferences and deriving evidence-based keyboard layout optimizations. The pipeline transforms raw keystroke data and preference ratings into statistically grounded information to guide keyboard layout design.

**Research Focus**: Understanding the relationship between typing speed, user preferences, and biomechanical comfort to inform optimal keyboard layouts.


## Pipeline Architecture

```
Raw Keystroke Data → Performance Analysis → Preference Analysis → MOO Optimization
       ↓                    ↓                    ↓                    ↓
 Timing & Errors     Speed/Preference      Key Rankings        Layout Design
                      Relationships                           Constraints
```

### Core Analysis Modules

| Module | Script | Purpose |
|--------|--------|---------|
| **Raw Data Analysis** | `analyze_raw_data.py` | Extract performance metrics from keystroke timing data |
| **Data Processing** | `process_summary_data.py` | Filter participants and score preferences |
| **Preference Analysis** | `analyze_frequency_speed_preference.py` | Analyze speed/preference/frequency relationships |
| **MOO Analysis** | `analyze_objectives.py` | Multi-objective optimization for layout design |
| **Dataset Comparison** | `compare_datasets.py` | Cross-study validation and replication |
| **Results Comparison** | `compare_results.py` | Compare MOO analysis across datasets |


## Quick Start

### Basic Analysis Workflow
```bash
# 1. Analyze raw keystroke performance
python analyze_raw_data.py

# 2. Process and filter experimental data
python process_summary_data.py --config config.yaml

# 3. Analyze preference relationships
python analyze_frequency_speed_preference.py --config config.yaml

# 4. MOO analysis for layout optimization
python analyze_objectives.py --data processed_consistent_choices.csv --output moo_results/

# 5. Compare datasets and results (optional)
python compare_datasets.py --dataset1 study1.csv --dataset2 study2.csv --output compare_datasets/
python compare_results.py --dataset1 results/dataset1/ --dataset2 results/dataset2/ --output compare_results/
```

### Key Configuration
Essential settings in `config.yaml`:
```yaml
data:
  input_dir: "data/raw"
  filtered_data_file: "processed_consistent_choices.csv"

analysis:
  min_trials_per_participant: 10
  max_time_ms: 3000
  confidence_level: 0.95

moo_objectives_analysis:
  bootstrap_iterations: 1000
  regularization: 1e-4
```


## Documentation

| Topic | File | Description |
|-------|------|-------------|
| **Raw Data Analysis** | [docs/analyze_raw_data.md](docs/raw_data_analysis.md) | Keystroke performance metrics |
| **Data Processing** | [docs/process_data.md](docs/data_processing.md) | Filtering and scoring pipeline |
| **Preference Analysis** | [docs/analyze_frequency_speed_preference.md](docs/preference_analysis.md) | Speed/preference relationships |
| **MOO Analysis** | [docs/analyze_objectives.md](docs/moo_analysis.md) | Multi-objective optimization |


## Features

### Statistics
- **Bradley-Terry Models**: Robust pairwise preference ranking
- **Bootstrap Confidence Intervals**: Non-parametric uncertainty quantification
- **Multiple Comparison Correction**: False Discovery Rate control
- **Effect Size Focus**: Practical significance over statistical significance

### Multi-Objective Optimization
- **Key Preferences**: Individual key quality rankings
- **Row Separation**: Same-row vs. cross-row movement preferences
- **Column Separation**: Adjacent vs. distant finger movements
- **Finger Coordination**: Biomechanical movement patterns

### Cross-Study Validation
- **Dataset Comparison**: Statistical tests for study replication
- **Results Comparison**: MOO objective consistency across studies
- **Preference Stability**: Rankings correlation analysis

