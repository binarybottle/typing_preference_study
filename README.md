# Bigram Typing Preference Study
 
-- jspsych website to collect study data via Prolific and python analysis code --

Author: Arno Klein (binarybottle.com)

GitHub repository: binarybottle/bigram-typing-comfort-experiment

License: Apache v2.0 

## Description
The purpose of the scripts in this repository is to determine which bigram in each of many bigram pairs is easier (more comfortable) to type on a qwerty computer keyboard, ultimately to inform the design of future keyboard layouts.

  - src/experiment.js: script to present and collect bigram typing data via a website
  
    - Step 1. Present consent and instructions.
    - Step 2. Present pair of bigrams to be typed repeatedly between random text.
    - Step 3. Present slider bar to indicate which bigram is easier to type.
    - Step 4. Collect timing and bigram preference data.
    - Step 5. Store data in OSF.io data repository.
    - Step 6. Send participant back to the Prolific crowdsourcing platform.

  - analyze/process_data.py: script to process/filter experiment's bigram typing data

    - Input: csv tables of summary data, easy choice (improbable) bigram pairs
    - Output: csv tables and plots
    - Step 1. Load and combine the data collected from the study above.
    - Step 2. Filter users by improbable or inconsistent choice thresholds.
    - Step 3. Score choices by slider values.
    - Step 4. Choose winning bigrams for bigram pairs.

  - analyze/analyze_data.py: script to analyze experiment's bigram typing data

    - Input config.yaml file with various settings, including the csv table of filtered user data.
    - See analyze/README_analyze_data.txt 

  ### process_data.py notes:
   - 2. Filter users by inconsistent or improbable choice thresholds
  The "improbable" choice is choosing the bigram "vr" as easier to type than "fr", and can be used as an option to filter users that may have chosen slider values randomly.
  
   - 3. Score choices by slider values
  For this study, each user makes a choice between two bigrams, two times, by using a slider each time -- but the score_user_choices_by_slider_values function generalizes to many pairwise choices. If a user chooses the same bigram every time, we take their median absolute slider value. If a user chooses different bigrams, we subtract the sums of the absolute values for each choice. In both cases, the score is the absolute value of the result.

   - 4. Choose winning bigrams
  Here we determine a winning bigram for each bigram pair across all users and all trials. If the winning bigram for every user is the same, the winning score is the median absolute score. If the winning bigram differs across users, the winning score is calculated as follows: we subtract the sum of the absolute values of the scores for one bigram from the other, and divide by the number of choices made for that bigram pair across the dataset.

  ### analyze_data.py notes:
  This analysis of bigram typing times was intended to determine whether there is a correlation between typing speed and typing preference, in case we could use speed as a proxy for comfort in future work.
