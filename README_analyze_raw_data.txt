Keyboard Typing Analysis

This script analyzes typing data to evaluate performance differences between keys and key-pairs
on the left home block keys ('Q', 'W', 'E', 'R', 'A', 'S', 'D', 'F', 'Z', 'X', 'C', 'V') 
and their mirror pairs on the right side.

Overview
The analysis performs several calculations:

1-key statistics: Analyzes individual keys as the second key in space-flanked bigrams
2-key (bigram) statistics: Analyzes key combinations defined as two consecutive keys flanked by spaces
Mirror pair analysis: Compares performance metrics between mirror image key pairs across the keyboard
Frequency adjustment:
- Frequency cutoff above a set number of occurrences in Peter Norvig's Google bigram data
- Log-transformed Frequency: Accounts for non-linear effects of letter frequency
- Frequency-adjusted Times: Timing metrics with frequency effects removed
- R²: Variance in performance explained by letter frequency

All analyses exclude same-key bigrams (e.g., "AA") and focus on metrics derived from space-flanked bigrams.

Requirements

Python 3.6+
Libraries: pandas, numpy, matplotlib, seaborn, statsmodels, scipy

Install required packages:
pip install pandas numpy matplotlib seaborn statsmodels scipy
Input Data
The script expects:

CSV files containing keystroke data with the following columns:
user_id: Unique identifier for the user
trialId: Identifier for the typing trial
expectedKey: The key that should have been typed
typedKey: The key that was actually typed
isCorrect: Boolean indicating if the key was typed correctly
keydownTime: Timestamp of the keypress (in milliseconds)

Optional frequency data:
Letter frequency data (CSV file)
Bigram frequency data (CSV file)


Directory Structure
The script uses the following directory structure:
input/
  raws_nonProlific/   # Contains raw typing data CSV files
  letter_frequencies_english.csv   # Optional letter frequency data
  letter_pair_frequencies_english.csv   # Optional bigram frequency data
output/
  figures/   # Generated visualization figures
  *.csv      # Output CSV files with analysis results

Running the Script
python analyze_raw_data.py

Output Files
CSV Files
The script generates several CSV files in the output/ directory:

left_home_key_analysis.csv: Contains statistics for individual keys

Error rates, typing times, and error patterns for each key


left_home_bigram_analysis.csv: Contains statistics for key combinations

Error rates, typing times, and error patterns for each bigram


mirror_pair_analysis.csv: Contains comparison data between mirror pairs

Error rates, typing times, and differences between left and right keys

Visualization Plots
The script generates several visualization plots in the output/figures/ directory:

1-Key (Individual Key) Analysis

error_1key.png: Error rates for individual keys
typing_time_1key.png: Typing times for individual keys
error_vs_typing_time_1key.png: Correlation between error rates and typing times

2-Key (Bigram) Analysis

count_bigram.png: Occurrence counts for each bigram
error_2key.png: Error rates for each bigram
typing_time_2key.png: Typing times for each bigram
error_vs_typing_time_2key.png: Correlation between error rates and typing times

1-Key vs. 2-Key Comparisons

error_1key_vs_2key.png: Comparison of error rates between individual keys and bigrams
typing_time_1key_vs_2key.png: Comparison of typing times between individual keys and bigrams

Mirror Pair Analysis

mirror_pair_error_rates_with_mad.png: Error rates for mirror pairs (grouped bar plot)
mirror_pair_typing_time.png: Typing times for mirror pairs (grouped bar plot)
mirror_pair_error_left_vs_right.png: Scatter plot of left vs. right key error rates
mirror_pair_time_left_vs_right.png: Scatter plot of left vs. right key typing times

Metrics

Error Rate: Percentage of incorrect keypresses

For 1-key: When the key is incorrectly typed as the second key in a bigram (with the first key correct)
For 2-key: When the second key of a bigram is incorrectly typed (with the first key correct)


Typing Time: Time between keypresses (in milliseconds)

For 1-key: Time from the first key to the second key in the bigram (no mistakes)
For 2-key: Time between the first and second keys of the bigram (no mistakes)

Most Common Mistype: The most frequently substituted key when an error occurs

Definitions

1-key: Individual key analysis where the key is the second key in a space-flanked bigram
2-key (bigram): Analysis of a key pair, defined as two consecutive keys flanked by spaces
Flanked bigram: A sequence of: space → key1 → key2 → space
Home block keys: The keys 'Q', 'W', 'E', 'R', 'A', 'S', 'D', 'F', 'Z', 'X', 'C', 'V'
Mirror pairs: Corresponding keys on opposite sides of the keyboard (e.g., 'Q' and 'P')


