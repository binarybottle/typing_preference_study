#!/usr/bin/env python3
"""
Simple test to get spatial B-T model working with your data.
This uses the fixes identified in the diagnostic.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from collections import defaultdict

def load_and_process_data(data_file):
    """Load and process data like the spatial model."""
    data = pd.read_csv(data_file)
    
    # Filter non-zero slider values
    data = data[data['sliderValue'] != 0].copy()
    data['chosen_bigram'] = data['chosen_bigram'].astype(str).str.lower()
    data['unchosen_bigram'] = data['unchosen_bigram'].astype(str).str.lower()
    
    print(f"Loaded {len(data)} choice records")
    return data

def extract_comparisons(data, min_comparisons=20):
    """Extract bigram comparisons."""
    left_keys = ['q','w','e','r','a','s','d','f','z','x','c','v']  # 12 keys only
    valid_bigrams = set()
    for i, key1 in enumerate(left_keys):
        for j, key2 in enumerate(left_keys):
            if i != j:
                valid_bigrams.add(key1 + key2)
    
    comparisons = defaultdict(lambda: {'wins_bigram1': 0, 'total': 0})
    
    for _, row in data.iterrows():
        chosen = str(row['chosen_bigram']).lower()
        unchosen = str(row['unchosen_bigram']).lower()
        
        if chosen in valid_bigrams and unchosen in valid_bigrams and chosen != unchosen:
            bigram1, bigram2 = sorted([chosen, unchosen])
            pair = (bigram1, bigram2)
            
            comparisons[pair]['total'] += 1
            if chosen == bigram1:
                comparisons[pair]['wins_bigram1'] += 1
    
    # Filter by minimum comparisons and remove extreme cases
    filtered_comparisons = {}
    for pair, comp in comparisons.items():
        if comp['total'] >= min_comparisons:
            win_rate = comp['wins_bigram1'] / comp['total']
            if 0.10 <= win_rate <= 0.90:  # Filter extreme cases
                filtered_comparisons[pair] = comp
    
    print(f"Found {len(comparisons)} total pairs")
    print(f"Using {len(filtered_comparisons)} pairs after filtering (≥{min_comparisons} comparisons, 10%-90% win rate)")
    
    return filtered_comparisons

def extract_simple_features(bigram):
    """Extract minimal spatial features."""
    if len(bigram) != 2:
        return None
    
    # Define 12-key layout
    layout = {
        'q': (1, 1, 1), 'w': (1, 2, 2), 'e': (1, 3, 3), 'r': (1, 4, 4),
        'a': (2, 1, 1), 's': (2, 2, 2), 'd': (2, 3, 3), 'f': (2, 4, 4),
        'z': (3, 1, 1), 'x': (3, 2, 2), 'c': (3, 3, 3), 'v': (3, 4, 4)
    }
    
    key1, key2 = bigram[0], bigram[1]
    if key1 not in layout or key2 not in layout:
        return None
    
    row1, col1, finger1 = layout[key1]
    row2, col2, finger2 = layout[key2]
    
    return {
        'intercept': 1.0,
        'key1_home': 1.0 if row1 == 2 else 0.0,
        'key2_home': 1.0 if row2 == 2 else 0.0,
        'same_row': 1.0 if row1 == row2 else 0.0,
        'inward_roll': 1.0 if (row1 == row2 and finger2 > finger1) else 0.0,
        'finger_span_1': 1.0 if abs(finger1 - finger2) == 1 else 0.0,
        'finger_span_2': 1.0 if abs(finger1 - finger2) == 2 else 0.0,
        'finger_span_3': 1.0 if abs(finger1 - finger2) == 3 else 0.0,
    }

def fit_simple_model(comparisons):
    """Fit simple Bradley-Terry model."""
    
    # Prepare training data
    features_list = []
    wins_list = []
    totals_list = []
    
    for (bigram1, bigram2), comp in comparisons.items():
        features1 = extract_simple_features(bigram1)
        features2 = extract_simple_features(bigram2)
        
        if features1 and features2:
            features_list.append((features1, features2))
            wins_list.append(comp['wins_bigram1'])
            totals_list.append(comp['total'])
    
    print(f"Training on {len(features_list)} bigram pairs")
    
    feature_names = ['intercept', 'key1_home', 'key2_home', 'same_row', 'inward_roll', 
                     'finger_span_1', 'finger_span_2', 'finger_span_3']
    
    def negative_log_likelihood(params):
        ll = 0
        for (features1, features2), wins, total in zip(features_list, wins_list, totals_list):
            eta1 = sum(params[i] * features1[feature_names[i]] for i in range(len(params)))
            eta2 = sum(params[i] * features2[feature_names[i]] for i in range(len(params)))
            
            # Clip to prevent overflow
            eta1 = np.clip(eta1, -10, 10)
            eta2 = np.clip(eta2, -10, 10)
            
            p12 = np.exp(eta1) / (np.exp(eta1) + np.exp(eta2))
            p12 = np.clip(p12, 1e-10, 1 - 1e-10)
            
            ll += wins * np.log(p12) + (total - wins) * np.log(1 - p12)
        
        # Add small regularization
        reg = 0.01 * sum(params[1:]**2)  # Skip intercept
        return -ll + reg
    
    # Try optimization
    initial_params = np.zeros(len(feature_names))
    
    # Use Powell (which worked in diagnostic)
    result = minimize(negative_log_likelihood, initial_params, method='Powell',
                     options={'maxiter': 2000})
    
    if result.success:
        print(f"✓ Model fitted successfully!")
        print(f"✓ Log-likelihood: {-result.fun:.2f}")
        
        # Show coefficients
        print("\nCoefficients:")
        for i, (name, coef) in enumerate(zip(feature_names, result.x)):
            direction = "+" if coef > 0 else ""
            print(f"  {name}: {direction}{coef:.3f}")
        
        return dict(zip(feature_names, result.x))
    else:
        print(f"✗ Model fitting failed: {result.message}")
        return None

def score_bigrams(model_params):
    """Score all 132 bigrams (12 keys, no same-key)."""
    if not model_params:
        return None
    
    left_keys = ['q','w','e','r','a','s','d','f','z','x','c','v']
    scores = []
    
    for key1 in left_keys:
        for key2 in left_keys:
            if key1 != key2:
                bigram = key1 + key2
                features = extract_simple_features(bigram)
                if features:
                    score = sum(model_params[name] * features[name] for name in model_params.keys())
                    scores.append((bigram, score))
    
    # Sort by score (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTOP 20 BIGRAMS:")
    print("-" * 40)
    for i, (bigram, score) in enumerate(scores[:20]):
        print(f"{i+1:2d}. {bigram.upper()}: {score:6.3f}")
    
    print(f"\nBOTTOM 10 BIGRAMS:")
    print("-" * 40)
    for i, (bigram, score) in enumerate(scores[-10:]):
        rank = len(scores) - 9 + i
        print(f"{rank:2d}. {bigram.upper()}: {score:6.3f}")
    
    return scores

def main():
    data_file = "/Users/arno/Software/typing_preference_study/analyze_summary_data/input/Prolific/process_data_TGB/tables/processed_consistent_choices.csv"
    
    print("SIMPLE SPATIAL B-T MODEL TEST")
    print("=" * 50)
    
    # Load data
    data = load_and_process_data(data_file)
    
    # Extract comparisons
    comparisons = extract_comparisons(data, min_comparisons=20)
    
    if len(comparisons) < 20:
        print("✗ Insufficient data for modeling")
        return
    
    # Fit model
    model_params = fit_simple_model(comparisons)
    
    # Score all bigrams
    if model_params:
        scores = score_bigrams(model_params)
        print(f"\n✓ Successfully scored {len(scores)} bigrams!")

if __name__ == "__main__":
    main()