import random
import json

# Define the letters on the left-hand side of the QWERTY keyboard
letters = ['Q', 'W', 'E', 'R', 'T', 'A', 'S', 'D', 'F', 'G', 'Z', 'X', 'C', 'V', 'B']

# Generate all possible ordered pairs of letter pairs
letter_pairs = []
for i in range(len(letters)):
    for j in range(len(letters)):
        if i != j:
            letter_pairs.append([letters[i], letters[j]])

all_pair_combinations = []
for pair1 in letter_pairs:
    for pair2 in letter_pairs:
        if pair1 != pair2:
            all_pair_combinations.append([pair1, pair2])

# Shuffle the list randomly
random.shuffle(all_pair_combinations)

# Split the list into 250 sublists
num_participants = 250
pairs_per_participant = len(all_pair_combinations) // num_participants
sublists = [all_pair_combinations[i*pairs_per_participant:(i+1)*pairs_per_participant] for i in range(num_participants)]

# Save each sublist as a JSON file
for idx, sublist in enumerate(sublists):
    filename = f'participant_{idx+1}.json'
    with open(filename, 'w') as f:
        json.dump(sublist, f)

print(f'Generated {num_participants} sublists, each containing {pairs_per_participant} pairs.')