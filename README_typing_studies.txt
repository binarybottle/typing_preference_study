This file contains information about specific studies run with this 
software to prepare data for typing_preferences_to_comfort_scores 
and optimize_layouts (for optimization of keyboard layouts).

Studies 1-5 were run exclusively on the Prolific recruitment platform.

Studies 6-9 were conducted with non-Prolific participants.

Study 6's 50 bigram pairs were selected using the following program:
typing_preferences_to_comfort_scores/extra/pair_recommender.py.
It uses maximum-minimum distance selection in PCA space to identify 
diverse bigram pairs for bigram pair typing data collection. 

Study 7's 50 bigram pairs were selected using
typing_preferences_to_comfort_scores' recommendations.py
to select a complementary set of 50 pairs distinct from those of 
Study 6 in PCA space.

Study 8's 37 bigram pairs were selected as follows:
1. bigram-typing-comfort-experiment software was used to collect
bigram typing preference choices from Prolific participants.
2. typing_preferences_to_comfort_scores software
estimated latent comfort scores for bigrams from the choices in #1.
3. All possible bigrams were ranked according to these estimated scores.
4. The top 29 pairs of neighbors in the ranked bigrams were selected.
5. An additional 8 pairs of single-key pairs were included to directly 
compare keys rather than infer single-key comfort scores from preference 
choices.

Study 9's 34 same-key bigram pairs were selected from the 66 possible 
unique same-key pairs of left keys to do a more extensive and direct 
comparison of keys rather than infer single-key comfort scores from 
bigram preference choices (see bigram_tables/samekey_bigram_comparisons.txt).