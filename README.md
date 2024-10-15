# Bigram Typing Study 
-- jspsych website via Prolific and python analysis --

Author: Arno Klein (binarybottle.com)

GitHub repository: binarybottle/bigram-typing-comfort-experiment

License: Apache v2.0 

The purpose of the scripts in this repository is to determine how comfortable 
different pairs of keys are to type on computer keyboards, 
ultimately to inform the design of future keyboard layouts.

  - experiment.js: script to present and collect bigram typing data via a website.
  
    - Step 1. Present consent and instructions.
    - Step 2. Present pair of bigrams to be typed repeatedly between random text.
    - Step 3. Present slider bar to indicate which bigram is easier to type.
    - Step 4. Collect timing and bigram preference data.
    - Step 5. Store data in OSF.io data repository.
    - Step 6. Send participant back to the Prolific crowdsourcing platform.

  - analyze_bigram_prolific_study_data.py: script to analyze bigram typing data.
    - Input: csv tables of summary data, easy choice (improbable) bigram pairs, and remove pairs.
    - Output: csv tables and plots.
    - Step 1. Load and combine the data collected from the study above.
    - Step 2. Filter users by improbable or inconsistent choice thresholds.
    - Step 3. Analyze bigram typing times. 
    - Step 4. Score choices by slider values.
    - Step 5. Choose winning bigrams for bigram pairs.

  #################################################################
  # 2. Filter users by inconsistent or improbable choice thresholds
  #################################################################
  The "improbable" choice is choosing the bigram "vr" as easier to type than "fr",
  and can be used as an option to filter users that may have chosen slider values randomly.
  
  ################################
  # 3. Analyze bigram typing times 
  ################################
  This analysis of bigram typing times was intended to determine whether there is a correlation 
  between typing speed and typing comfort, in case we could use speed as a proxy for comfort
  in future work. Unfortunately, while there is a significant correlation, 
  it appears from this data that the relationship is not consistent enough to warrant application.

  ###################################
  # 4. Score choices by slider values
  ###################################
  For this study, each user makes a choice between two bigrams, two times, by using a slider each time -- 
  but the score_user_choices_by_slider_values function generalizes to many pairwise choices.
  If a user chooses the same bigram every time, we take their median slider value.
  If a user chooses different bigrams, we subtract the sums of the absolute values for each choice.
  In both cases, the score is the absolute value of the result.

  ###########################
  # 5. Choose winning bigrams
  ###########################
  Here we determine a winning bigram for each bigram pair across all users and all trials.
  If the winning bigram for every user is the same, the winning score is the median score.
  If the winning bigram differs across users, the winning score is calculated as follows:
  we subtract the sum of the absolute values of the scores for one bigram from the other,
  and divide by the number of choices the made for that bigram pair across the dataset.
