"""
Analyze feasibility and fit direct Bradley-Terry model for all 210 left-hand bigrams.

This direct bigram B-T model (210 parameters) fits true Bradley-Terry strengths for each individual bigram.

Pros:
  - No modeling assumptions about bigram decomposition
  - True B-T parameters for each bigram
  - Maximum flexibility

Cons:
  - Requires substantial data (need good coverage of the 21,945 possible bigram pairs)
  - May have high uncertainty for rarely-observed bigrams
  - Many parameters to estimate
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Set
import warnings

class DirectBigramBTAnalyzer:
    """
    Analyze feasibility of fitting direct Bradley-Terry model to bigram data
    and provide data sufficiency diagnostics.
    """
    
    def __init__(self, data_path: str):
        """Load and prepare bigram comparison data."""
        self.data = pd.read_csv(data_path)
        self.left_keys = ['q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
        self.all_bigrams = self._generate_all_bigrams()
        self.bigram_comparisons = None
        
    def _generate_all_bigrams(self) -> List[str]:
        """Generate all 210 different-key bigrams."""
        bigrams = []
        for i, key1 in enumerate(self.left_keys):
            for j, key2 in enumerate(self.left_keys):
                if i != j:  # No same-key bigrams
                    bigrams.append(key1 + key2)
        return bigrams
    
    def extract_bigram_comparisons(self) -> Dict[Tuple[str, str], Dict[str, int]]:
        """
        Extract direct bigram-vs-bigram comparisons from choice data.
        Returns dictionary mapping (bigram1, bigram2) -> {'wins_bigram1': int, 'total': int}
        """
        comparisons = defaultdict(lambda: {'wins_bigram1': 0, 'total': 0})
        
        valid_bigrams = set(self.all_bigrams)
        
        for _, row in self.data.iterrows():
            chosen = str(row['chosen_bigram']).lower()
            unchosen = str(row['unchosen_bigram']).lower()
            
            # Only include if both are valid left-hand different-key bigrams
            if chosen in valid_bigrams and unchosen in valid_bigrams and chosen != unchosen:
                # Order consistently (alphabetically)
                bigram1, bigram2 = sorted([chosen, unchosen])
                pair = (bigram1, bigram2)
                
                comparisons[pair]['total'] += 1
                if chosen == bigram1:  # bigram1 was chosen
                    comparisons[pair]['wins_bigram1'] += 1
        
        self.bigram_comparisons = dict(comparisons)
        return self.bigram_comparisons
    
    def assess_data_sufficiency(self, min_comparisons: int = 5) -> Dict:
        """
        Assess whether there's sufficient data for direct B-T model.
        
        Args:
            min_comparisons: Minimum comparisons needed per pair for inclusion
            
        Returns:
            Dictionary with sufficiency metrics
        """
        if self.bigram_comparisons is None:
            self.extract_bigram_comparisons()
        
        # Calculate coverage statistics
        total_possible_pairs = len(self.all_bigrams) * (len(self.all_bigrams) - 1) // 2
        observed_pairs = len(self.bigram_comparisons)
        
        # Count pairs with sufficient data
        sufficient_pairs = sum(1 for comp in self.bigram_comparisons.values() 
                             if comp['total'] >= min_comparisons)
        
        # Count bigrams that appear in at least one sufficient comparison
        bigrams_in_sufficient_pairs = set()
        for (bigram1, bigram2), comp in self.bigram_comparisons.items():
            if comp['total'] >= min_comparisons:
                bigrams_in_sufficient_pairs.add(bigram1)
                bigrams_in_sufficient_pairs.add(bigram2)
        
        # Distribution of comparison counts
        comparison_counts = [comp['total'] for comp in self.bigram_comparisons.values()]
        
        # Identify isolated bigrams (appear in very few comparisons)
        bigram_appearance_count = Counter()
        for bigram1, bigram2 in self.bigram_comparisons.keys():
            bigram_appearance_count[bigram1] += 1
            bigram_appearance_count[bigram2] += 1
        
        isolated_bigrams = [bigram for bigram, count in bigram_appearance_count.items() 
                           if count < 5]  # Appears in fewer than 5 comparisons
        
        return {
            'total_possible_pairs': total_possible_pairs,
            'observed_pairs': observed_pairs,
            'coverage_rate': observed_pairs / total_possible_pairs,
            'sufficient_pairs': sufficient_pairs,
            'sufficient_coverage_rate': sufficient_pairs / total_possible_pairs,
            'bigrams_with_sufficient_data': len(bigrams_in_sufficient_pairs),
            'bigrams_coverage_rate': len(bigrams_in_sufficient_pairs) / len(self.all_bigrams),
            'mean_comparisons_per_pair': np.mean(comparison_counts) if comparison_counts else 0,
            'median_comparisons_per_pair': np.median(comparison_counts) if comparison_counts else 0,
            'min_comparisons_per_pair': min(comparison_counts) if comparison_counts else 0,
            'max_comparisons_per_pair': max(comparison_counts) if comparison_counts else 0,
            'isolated_bigrams': isolated_bigrams,
            'n_isolated_bigrams': len(isolated_bigrams)
        }
    
    def fit_direct_bt_model(self, min_comparisons: int = 5) -> Dict:
        """
        Attempt to fit direct Bradley-Terry model to bigram data.
        
        Args:
            min_comparisons: Minimum comparisons to include a pair
            
        Returns:
            Dictionary with model results and diagnostics
        """
        if self.bigram_comparisons is None:
            self.extract_bigram_comparisons()
        
        # Filter to sufficient data
        sufficient_comparisons = {
            pair: comp for pair, comp in self.bigram_comparisons.items()
            if comp['total'] >= min_comparisons
        }
        
        print(f"Fitting B-T model with {len(sufficient_comparisons)} pairs (min {min_comparisons} comparisons each)")
        
        # Get bigrams that appear in sufficient comparisons
        bigrams_in_model = set()
        for bigram1, bigram2 in sufficient_comparisons.keys():
            bigrams_in_model.add(bigram1)
            bigrams_in_model.add(bigram2)
        
        bigrams_in_model = sorted(list(bigrams_in_model))
        
        if len(bigrams_in_model) < 10:
            return {
                'success': False,
                'error': f'Too few bigrams with sufficient data: {len(bigrams_in_model)}',
                'bigrams_in_model': bigrams_in_model
            }
        
        # Build comparison matrices
        n_bigrams = len(bigrams_in_model)
        bigram_to_idx = {bigram: i for i, bigram in enumerate(bigrams_in_model)}
        
        wins = np.zeros((n_bigrams, n_bigrams))
        totals = np.zeros((n_bigrams, n_bigrams))
        
        for (bigram1, bigram2), comp in sufficient_comparisons.items():
            if bigram1 in bigram_to_idx and bigram2 in bigram_to_idx:
                i, j = bigram_to_idx[bigram1], bigram_to_idx[bigram2]
                wins[i, j] = comp['wins_bigram1']
                wins[j, i] = comp['total'] - comp['wins_bigram1']
                totals[i, j] = totals[j, i] = comp['total']
        
        # Fit B-T model using maximum likelihood
        try:
            strengths = self._fit_bt_ml(wins, totals, bigrams_in_model)
            
            # Calculate rankings
            rankings = [(bigrams_in_model[i], strengths[i]) for i in np.argsort(strengths)[::-1]]
            
            # Calculate confidence intervals via bootstrap
            print("Calculating bootstrap confidence intervals...")
            cis = self._bootstrap_bt_cis(sufficient_comparisons, bigrams_in_model, n_bootstrap=100)
            
            return {
                'success': True,
                'bigrams_in_model': bigrams_in_model,
                'n_bigrams': len(bigrams_in_model),
                'strengths': dict(zip(bigrams_in_model, strengths)),
                'rankings': rankings,
                'confidence_intervals': cis,
                'model_diagnostics': {
                    'n_parameters': len(bigrams_in_model),
                    'n_comparisons': len(sufficient_comparisons),
                    'total_observations': sum(comp['total'] for comp in sufficient_comparisons.values())
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'bigrams_in_model': bigrams_in_model
            }
    
    def _fit_bt_ml(self, wins: np.ndarray, totals: np.ndarray, bigrams: List[str]) -> np.ndarray:
        """Fit Bradley-Terry model using maximum likelihood."""
        n_items = len(bigrams)
        
        def negative_log_likelihood(strengths):
            ll = 0
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    if totals[i, j] > 0:
                        p_ij = np.exp(strengths[i]) / (np.exp(strengths[i]) + np.exp(strengths[j]))
                        p_ij = np.clip(p_ij, 1e-10, 1 - 1e-10)
                        ll += wins[i, j] * np.log(p_ij) + wins[j, i] * np.log(1 - p_ij)
            return -ll
        
        # Initialize with zeros
        initial_strengths = np.zeros(n_items)
        
        # Constraint: first item strength = 0 for identifiability
        def constraint_func(x):
            return x[0]
        
        constraint = {'type': 'eq', 'fun': constraint_func}
        
        result = minimize(
            negative_log_likelihood,
            initial_strengths,
            method='SLSQP',
            constraints=constraint,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")
        
        return result.x
    
    def _bootstrap_bt_cis(self, comparisons: Dict, bigrams: List[str], 
                         n_bootstrap: int = 100) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for B-T strengths."""
        
        # Convert to individual comparison records for resampling
        comparison_records = []
        for (bigram1, bigram2), comp in comparisons.items():
            wins1 = comp['wins_bigram1']
            total = comp['total']
            
            # Add individual records
            for _ in range(wins1):
                comparison_records.append((bigram1, bigram2, 1))  # bigram1 wins
            for _ in range(total - wins1):
                comparison_records.append((bigram1, bigram2, 0))  # bigram2 wins
        
        bigram_to_idx = {bigram: i for i, bigram in enumerate(bigrams)}
        n_bigrams = len(bigrams)
        bootstrap_strengths = np.zeros((n_bootstrap, n_bigrams))
        
        for b in range(n_bootstrap):
            # Resample
            resampled_indices = np.random.choice(len(comparison_records), 
                                               size=len(comparison_records), 
                                               replace=True)
            
            # Reconstruct comparison counts
            bootstrap_comparisons = defaultdict(lambda: {'wins_bigram1': 0, 'total': 0})
            for idx in resampled_indices:
                bigram1, bigram2, outcome = comparison_records[idx]
                if bigram1 in bigram_to_idx and bigram2 in bigram_to_idx:
                    pair = tuple(sorted([bigram1, bigram2]))
                    bootstrap_comparisons[pair]['total'] += 1
                    if (outcome == 1 and bigram1 == pair[0]) or (outcome == 0 and bigram2 == pair[0]):
                        bootstrap_comparisons[pair]['wins_bigram1'] += 1
            
            # Build matrices
            wins = np.zeros((n_bigrams, n_bigrams))
            totals = np.zeros((n_bigrams, n_bigrams))
            
            for (bigram1, bigram2), comp in bootstrap_comparisons.items():
                i, j = bigram_to_idx[bigram1], bigram_to_idx[bigram2]
                wins[i, j] = comp['wins_bigram1']
                wins[j, i] = comp['total'] - comp['wins_bigram1']
                totals[i, j] = totals[j, i] = comp['total']
            
            # Fit bootstrap model
            try:
                bootstrap_strengths[b, :] = self._fit_bt_ml(wins, totals, bigrams)
            except:
                # If fitting fails, use zeros
                bootstrap_strengths[b, :] = np.zeros(n_bigrams)
        
        # Calculate CIs
        cis = {}
        for i, bigram in enumerate(bigrams):
            lower = float(np.percentile(bootstrap_strengths[:, i], 2.5))
            upper = float(np.percentile(bootstrap_strengths[:, i], 97.5))
            cis[bigram] = (lower, upper)
        
        return cis
    
    def create_sufficiency_report(self, min_comparisons: int = 5) -> str:
        """Generate data sufficiency report."""
        
        sufficiency = self.assess_data_sufficiency(min_comparisons)
        
        report = [
            "BIGRAM BRADLEY-TERRY DATA SUFFICIENCY ANALYSIS",
            "=" * 60,
            "",
            f"Total possible left-hand bigrams: {len(self.all_bigrams)}",
            f"Total possible bigram pairs: {sufficiency['total_possible_pairs']:,}",
            "",
            "DATA COVERAGE:",
            f"  Observed pairs with data: {sufficiency['observed_pairs']:,} ({sufficiency['coverage_rate']:.1%})",
            f"  Pairs with ≥{min_comparisons} comparisons: {sufficiency['sufficient_pairs']:,} ({sufficiency['sufficient_coverage_rate']:.1%})",
            f"  Bigrams in sufficient pairs: {sufficiency['bigrams_with_sufficient_data']}/{len(self.all_bigrams)} ({sufficiency['bigrams_coverage_rate']:.1%})",
            "",
            "COMPARISON STATISTICS:",
            f"  Mean comparisons per pair: {sufficiency['mean_comparisons_per_pair']:.1f}",
            f"  Median comparisons per pair: {sufficiency['median_comparisons_per_pair']:.1f}",
            f"  Range: {sufficiency['min_comparisons_per_pair']} - {sufficiency['max_comparisons_per_pair']}",
            "",
            f"ISOLATED BIGRAMS (appear in <5 pairs): {sufficiency['n_isolated_bigrams']}",
        ]
        
        if sufficiency['isolated_bigrams']:
            report.append(f"  Examples: {', '.join(sufficiency['isolated_bigrams'][:10])}")
        
        report.extend([
            "",
            "RECOMMENDATIONS:",
        ])
        
        if sufficiency['sufficient_coverage_rate'] < 0.1:
            report.append("  ❌ INSUFFICIENT DATA for direct B-T model")
            report.append("  → Use decomposition approach (key + transition)")
        elif sufficiency['sufficient_coverage_rate'] < 0.3:
            report.append("  ⚠️  MARGINAL DATA for direct B-T model")
            report.append("  → Try both approaches and compare")
        else:
            report.append("  ✅ SUFFICIENT DATA for direct B-T model")
            report.append("  → Proceed with full 210-parameter model")
        
        return "\n".join(report)

# Usage example:
"""
analyzer = DirectBigramBTAnalyzer('data/filtered_data.csv')

# Check data sufficiency
print(analyzer.create_sufficiency_report(min_comparisons=5))

# If sufficient, fit direct B-T model
model_result = analyzer.fit_direct_bt_model(min_comparisons=5)

if model_result['success']:
    print(f"Successfully fit B-T model with {model_result['n_bigrams']} bigrams")
    print("Top 10 bigrams:")
    for i, (bigram, strength) in enumerate(model_result['rankings'][:10]):
        ci = model_result['confidence_intervals'][bigram]
        print(f"{i+1:2d}. {bigram}: {strength:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
else:
    print(f"B-T model fitting failed: {model_result['error']}")
"""