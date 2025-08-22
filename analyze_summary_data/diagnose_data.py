#!/usr/bin/env python3
"""
Diagnostic script to understand why certain hypothesis comparisons are missing.
Run this to analyze your data and see what's actually available.
"""

import pandas as pd
import sys
from collections import defaultdict, Counter

# Define the same keyboard layout as in your main script
key_positions = {
    # Upper row (row 1)
    'q': {'row': 1, 'column': 1, 'finger': 1},
    'w': {'row': 1, 'column': 2, 'finger': 2},
    'e': {'row': 1, 'column': 3, 'finger': 3},
    'r': {'row': 1, 'column': 4, 'finger': 4},
    't': {'row': 1, 'column': 5, 'finger': 4},
    
    # Middle/home row (row 2)
    'a': {'row': 2, 'column': 1, 'finger': 1},
    's': {'row': 2, 'column': 2, 'finger': 2},
    'd': {'row': 2, 'column': 3, 'finger': 3},
    'f': {'row': 2, 'column': 4, 'finger': 4},
    'g': {'row': 2, 'column': 5, 'finger': 4},
    
    # Lower row (row 3)
    'z': {'row': 3, 'column': 1, 'finger': 1},
    'x': {'row': 3, 'column': 2, 'finger': 2},
    'c': {'row': 3, 'column': 3, 'finger': 3},
    'v': {'row': 3, 'column': 4, 'finger': 4},
    'b': {'row': 3, 'column': 5, 'finger': 4}
}

def analyze_bigram_classifications(data_path):
    """Analyze what classifications are actually present in the data."""
    
    print("Loading and analyzing data...")
    data = pd.read_csv(data_path)
    data = data[data['sliderValue'] != 0].copy()  # Filter to consistent choices
    data['chosen_bigram'] = data['chosen_bigram'].astype(str).str.lower()
    data['unchosen_bigram'] = data['unchosen_bigram'].astype(str).str.lower()
    
    print(f"Loaded {len(data)} consistent choice rows")
    
    # Get all unique bigrams
    all_bigrams = set()
    for col in ['chosen_bigram', 'unchosen_bigram']:
        all_bigrams.update(data[col].unique())
    
    print(f"Found {len(all_bigrams)} unique bigrams")
    
    # Classify each bigram
    classifications = {}
    for bigram in all_bigrams:
        if len(bigram) >= 2:
            char1, char2 = bigram[0], bigram[1]
            if char1 in key_positions and char2 in key_positions:
                classifications[bigram] = classify_bigram(char1, char2)
    
    print(f"Successfully classified {len(classifications)} bigrams")
    
    # Analyze specific missing comparisons
    analyze_specific_hypotheses(data, classifications)
    
    return classifications

def classify_bigram(char1, char2):
    """Classify a bigram - FIXED VERSION."""
    pos1 = key_positions[char1]
    pos2 = key_positions[char2]
    
    # Basic measurements
    finger_separation = abs(pos1['finger'] - pos2['finger'])
    row_separation = abs(pos1['row'] - pos2['row'])
    
    # Home keys
    home_keys = {'a', 's', 'd', 'f'}
    home_key_count = sum(1 for char in [char1, char2] if char in home_keys)
    
    # Column 5 keys
    column5_keys = {'t', 'g', 'b'}
    column5_count = sum(1 for char in [char1, char2] if char in column5_keys)
    
    # Direction calculation
    if row_separation == 0:  # Same row
        if pos2['finger'] > pos1['finger']:
            direction = 'inner_roll'
        elif pos2['finger'] < pos1['finger']:
            direction = 'outer_roll'
        else:
            direction = 'same_finger'
    else:  # Cross row
        if pos2['finger'] > pos1['finger']:
            direction = 'inner_roll_cross'
        elif pos2['finger'] < pos1['finger']:
            direction = 'outer_roll_cross'
        else:
            direction = 'same_finger_cross'
    
    # Dominant finger
    dominant_finger = max(pos1['finger'], pos2['finger'])
    
    # Column-specific row preferences - FIXED
    column_row_prefs = {}
    for column in [1, 2, 3, 4]:
        # Find which keys (if any) are in this column
        keys_in_column = []
        if pos1['column'] == column:
            keys_in_column.append(pos1)
        if pos2['column'] == column:
            keys_in_column.append(pos2)
        
        if len(keys_in_column) == 1:
            # Exactly one key in this column
            column_row_prefs[f'column{column}_row_pref'] = str(keys_in_column[0]['row'])
        elif len(keys_in_column) == 2:
            # Both keys in this column
            rows = [k['row'] for k in keys_in_column]
            if 1 in rows and 3 in rows:
                # Contains both upper and lower
                column_row_prefs[f'column{column}_row_pref'] = str(pos1['row'])
            elif 1 in rows:
                column_row_prefs[f'column{column}_row_pref'] = '1'
            elif 3 in rows:
                column_row_prefs[f'column{column}_row_pref'] = '3'
            else:
                column_row_prefs[f'column{column}_row_pref'] = None
        else:
            column_row_prefs[f'column{column}_row_pref'] = None
    
    return {
        'same_row_finger_separation': str(finger_separation) if row_separation == 0 else None,
        'cross_row_finger_separation': str(finger_separation) if row_separation > 0 else None,
        'cross_row_same_finger': str(finger_separation == 0) if row_separation > 0 else None,
        'home_key_count': str(home_key_count),
        'row_separation': str(row_separation),
        'column5_count': str(column5_count),
        'dominant_finger': str(dominant_finger) if finger_separation > 0 else None,
        'same_row_direction': direction if row_separation == 0 and direction != 'same_finger' else None,
        'cross_row_direction': direction if row_separation > 0 and direction != 'same_finger_cross' else None,
        **column_row_prefs
    }

def analyze_specific_hypotheses(data, classifications):
    """Analyze why specific hypotheses are missing comparisons."""
    
    print("\n" + "="*60)
    print("DETAILED ANALYSIS OF MISSING COMPARISONS")
    print("="*60)
    
    # Define the problematic hypotheses
    problematic_hypotheses = {
        'same_row_finger_sep_2v3': {'category': 'same_row_finger_separation', 'values': ['2', '3']},
        'column5_0v1': {'category': 'column5_count', 'values': ['0', '1']},
        'column5_1v2': {'category': 'column5_count', 'values': ['1', '2']},
        'finger_f2_vs_f1': {'category': 'dominant_finger', 'values': ['2', '1']},
        'column1_upper_vs_lower': {'category': 'column1_row_pref', 'values': ['1', '3']},
        'column2_upper_vs_lower': {'category': 'column2_row_pref', 'values': ['1', '3']},
        'column3_upper_vs_lower': {'category': 'column3_row_pref', 'values': ['1', '3']},
        'column4_upper_vs_lower': {'category': 'column4_row_pref', 'values': ['1', '3']},
    }
    
    for hyp_name, hyp_config in problematic_hypotheses.items():
        print(f"\nAnalyzing: {hyp_name}")
        print("-" * 40)
        
        category = hyp_config['category']
        target_values = set(hyp_config['values'])
        
        # Count how many bigrams have each classification value
        value_counts = Counter()
        comparison_pairs = defaultdict(int)
        
        for _, row in data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            chosen_class = classifications.get(chosen, {})
            unchosen_class = classifications.get(unchosen, {})
            
            chosen_val = chosen_class.get(category)
            unchosen_val = unchosen_class.get(category)
            
            if chosen_val is not None:
                value_counts[f"chosen_{chosen_val}"] += 1
            if unchosen_val is not None:
                value_counts[f"unchosen_{unchosen_val}"] += 1
            
            # Check for target comparisons
            if (chosen_val in target_values and unchosen_val in target_values and 
                chosen_val != unchosen_val):
                pair = tuple(sorted([chosen_val, unchosen_val]))
                comparison_pairs[pair] += 1
        
        print(f"Value distribution for {category}:")
        for value, count in sorted(value_counts.items()):
            print(f"  {value}: {count}")
        
        print(f"Target values: {target_values}")
        print(f"Comparisons found: {dict(comparison_pairs)}")
        
        if not comparison_pairs:
            print("❌ NO COMPARISONS FOUND!")
            # Suggest why
            available_values = set()
            for val_name, count in value_counts.items():
                if count > 0:
                    val = val_name.split('_', 1)[1] if '_' in val_name else val_name
                    available_values.add(val)
            
            missing_values = target_values - available_values
            if missing_values:
                print(f"Missing target values: {missing_values}")
            else:
                print("Target values exist but never appear in same comparison")
        else:
            print("✅ Comparisons found!")

def print_example_bigrams_by_category():
    """Print examples of bigrams for each category to help understand the data."""
    
    print("\n" + "="*60)
    print("EXAMPLE BIGRAMS BY CATEGORY")
    print("="*60)
    
    # Generate some example bigrams
    keys = list(key_positions.keys())
    examples = {}
    
    for i, key1 in enumerate(keys):
        for key2 in keys[i+1:]:  # Avoid duplicates
            bigram = key1 + key2
            classification = classify_bigram(key1, key2)
            
            for category, value in classification.items():
                if value is not None and value != 'None':
                    if category not in examples:
                        examples[category] = {}
                    if value not in examples[category]:
                        examples[category][value] = []
                    if len(examples[category][value]) < 3:  # Limit examples
                        examples[category][value].append(bigram)
    
    for category, value_examples in examples.items():
        print(f"\n{category}:")
        for value, bigrams in value_examples.items():
            print(f"  {value}: {', '.join(bigrams[:3])}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnostic_script.py <path_to_csv_file>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    
    try:
        classifications = analyze_bigram_classifications(data_path)
        print_example_bigrams_by_category()
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        print("1. Replace the _classify_single_bigram method in your main script")
        print("2. The column-specific row preferences logic has been fixed")
        print("3. Some comparisons may still be missing due to data limitations")
        print("4. Consider if the hypothesis definitions match your data structure")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()