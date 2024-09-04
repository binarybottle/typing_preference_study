import itertools
import random
import json

# Generate all possible letter pairs
letters = ['Q', 'W', 'E', 'R', 'T', 'A', 'S', 'D', 'F', 'G', 'Z', 'X', 'C', 'V', 'B']
letter_pairs = [a + b for a, b in itertools.permutations(letters, 2)]

# Generate all possible pairs of letter pairs
pairs_of_letter_pairs = list(itertools.combinations(letter_pairs, 2))

# Shuffle the list of pairs of letter pairs
random.shuffle(pairs_of_letter_pairs)

# Calculate the number of pairs per participant
pairs_per_participant = len(pairs_of_letter_pairs) // 250

# Create 250 sublists
sublists = [pairs_of_letter_pairs[i:i + pairs_per_participant] for i in range(0, len(pairs_of_letter_pairs), pairs_per_participant)]

# Ensure we have exactly 250 sublists
while len(sublists) > 250:
    extra = sublists.pop()
    for i, item in enumerate(extra):
        sublists[i % 250].append(item)

# Convert sublists to a format that can be easily loaded in JavaScript
json_sublists = [
    [list(pair) for pair in sublist]
    for sublist in sublists
]

# Save the sublists to a JSON file
with open('letter_pair_sublists.json', 'w') as f:
    json.dump(json_sublists, f)

print(f"Generated {len(sublists)} sublists with approximately {len(sublists[0])} pairs each.")
print("Sublists saved to letter_pair_sublists.json")