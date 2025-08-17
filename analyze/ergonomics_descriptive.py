"""
Keyboard ergonomics statistical analysis module.
Includes power analysis, individual differences, and standardized reporting.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import ndtri
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.power import TTestPower
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import logging
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from collections import defaultdict

# Import your existing keyboard mappings
from keymaps import row_map, column_map, finger_map

logger = logging.getLogger(__name__)

class KeyboardErgonomicsAnalysis:
    """
    Keyboard ergonomics analysis with power analysis,
    individual differences, and standardized reporting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get('analysis', {}).get('alpha_level', 0.05)
        self.correction_method = config.get('analysis', {}).get('correction_method', 'fdr_bh')
        self.power_threshold = config.get('analysis', {}).get('power_threshold', 0.80)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def _standardize_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all test results have consistent formatting and include sample sizes.
        This method recursively processes the results dictionary to add missing fields.
        """
        
        def standardize_single_result(result_dict: Dict[str, Any]) -> None:
            """Standardize a single test result dictionary."""
            if isinstance(result_dict, dict) and 'p_value' in result_dict:
                # Ensure all required fields are present
                if 'n_comparisons' not in result_dict:
                    result_dict['n_comparisons'] = 0
                    logger.warning(f"Missing n_comparisons in test: {result_dict.get('test_name', 'Unknown')}")
                
                if 'effect_size' not in result_dict and 'proportion' in result_dict:
                    result_dict['effect_size'] = abs(result_dict['proportion'] - 0.5)
                
                if 'cohens_h' not in result_dict and 'proportion' in result_dict:
                    result_dict['cohens_h'] = self._calculate_cohens_h(result_dict['proportion'], 0.5)
                    result_dict['cohens_h_interpretation'] = self._interpret_cohens_h(result_dict['cohens_h'])
                
                if 'significant' not in result_dict and 'p_value' in result_dict:
                    result_dict['significant'] = result_dict['p_value'] < self.alpha
                
                # Add confidence intervals if missing but we have the data
                if 'ci_lower' not in result_dict and 'proportion' in result_dict and result_dict['n_comparisons'] > 0:
                    n_successes = int(result_dict['proportion'] * result_dict['n_comparisons'])
                    ci_lower, ci_upper = self._calculate_proportion_ci(n_successes, result_dict['n_comparisons'])
                    result_dict['ci_lower'] = ci_lower
                    result_dict['ci_upper'] = ci_upper
        
        def process_recursively(obj: Any) -> None:
            """Recursively process the results dictionary."""
            if isinstance(obj, dict):
                # Check if this is a test result
                standardize_single_result(obj)
                
                # Recurse into nested dictionaries
                for key, value in obj.items():
                    if isinstance(value, dict):
                        process_recursively(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                process_recursively(item)
        
        # Create a copy to avoid modifying the original
        standardized_results = results.copy()
        process_recursively(standardized_results)
        
        return standardized_results

    def run_complete_analysis(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """
        Run complete analysis including all enhancements.
        
        Args:
            data: DataFrame with columns [user_id, chosen_bigram, unchosen_bigram, is_consistent]
            output_folder: Directory to save results
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting keyboard ergonomics analysis...")
        
        # Create analysis subdirectory
        analysis_folder = os.path.join(output_folder, 'keyboard_ergonomics')
        os.makedirs(analysis_folder, exist_ok=True)
        
        # Add keyboard position features
        enhanced_data = self._add_keyboard_features(data)
        
        # Filter to consistent choices only for main analysis
        consistent_data = enhanced_data[enhanced_data['is_consistent'] == True].copy()
        logger.info(f"Using {len(consistent_data)} consistent choice rows from {consistent_data['user_id'].nunique()} participants")
        
        # === 1. POWER ANALYSIS ===
        logger.info("Conducting power analysis...")
        power_analysis = self.conduct_power_analysis(consistent_data, analysis_folder)
        
        # === 2. MAIN ERGONOMICS TESTS ===
        logger.info("Running main ergonomics tests...")
        main_results = self.run_all_ergonomics_tests(consistent_data, analysis_folder)
        
        # === 3. STATISTICAL REPORTING ===
        logger.info("Adding statistical reporting...")
        enhanced_results = self.add_statistical_reporting(main_results, consistent_data)
        
        # === 4. INDIVIDUAL DIFFERENCES ANALYSIS ===
        logger.info("Analyzing individual differences...")
        individual_differences = self.analyze_individual_differences(consistent_data, analysis_folder)
        
        # === 5. SENSITIVITY ANALYSES ===
        logger.info("Running sensitivity analyses...")
        sensitivity_results = self.run_sensitivity_analyses(enhanced_data, analysis_folder)
        
        # === 6. STANDARDIZED FIGURES ===
        logger.info("Creating figures...")
        self.create_figures(enhanced_results, individual_differences, analysis_folder)
        
        # === 7. COMPREHENSIVE REPORT ===
        complete_results = {
            'power_analysis': power_analysis,
            'main_results': enhanced_results,
            'individual_differences': individual_differences,
            'sensitivity_analyses': sensitivity_results,
            'sample_characteristics': self._compute_sample_characteristics(consistent_data)
        }
        
        logger.info("Generating comprehensive report...")
        self.generate_report(complete_results, analysis_folder)
        
        logger.info("Keyboard ergonomics analysis complete!")
        return complete_results
    
    # ====================================
    # 2. POWER ANALYSIS & SAMPLE SIZE JUSTIFICATION
    # ====================================
    
    def conduct_power_analysis(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """Comprehensive power analysis for sample size justification."""
        
        n_participants = data['user_id'].nunique()
        n_observations = len(data)
        
        # Define effect sizes of interest
        effect_sizes = {
            'small': 0.05,      # 5 percentage points from chance
            'medium': 0.15,     # 15 percentage points from chance  
            'large': 0.25       # 25 percentage points from chance
        }
        
        power_results = {}
        
        # Calculate power for different effect sizes
        for effect_name, effect_size in effect_sizes.items():
            # Convert percentage point difference to proportion
            p1 = 0.5 + effect_size  # Proportion for alternative hypothesis
            p0 = 0.5                # Null hypothesis proportion
            
            # Calculate power using normal approximation for binomial test
            power = self._calculate_binomial_power(n_observations, p0, p1, self.alpha)
            
            # Calculate required sample size for 80% power
            required_n = self._calculate_required_sample_size(p0, p1, self.alpha, 0.80)
            
            power_results[effect_name] = {
                'effect_size': effect_size,
                'effect_size_description': f"{effect_size*100:.0f} percentage points from chance",
                'current_power': power,
                'required_n_for_80_power': required_n,
                'current_n': n_observations,
                'adequately_powered': power >= 0.80
            }
        
        # Summary statistics
        power_summary = {
            'total_participants': n_participants,
            'total_observations': n_observations,
            'power_by_effect_size': power_results,
            'recommendations': self._generate_power_recommendations(power_results, n_observations)
        }
        
        # Save JSON data
        power_path = os.path.join(output_folder, 'power_analysis.json')
        with open(power_path, 'w') as f:
            json.dump(power_summary, f, indent=2, default=str)
        
        # Save human-readable summary
        self._generate_power_analysis_summary(power_summary, output_folder)
        
        return power_summary
    
    def _calculate_binomial_power(self, n: int, p0: float, p1: float, alpha: float) -> float:
        """Calculate power for binomial test using normal approximation."""
        if n == 0:
            return 0.0
            
        # Standard error under null hypothesis
        se0 = np.sqrt(p0 * (1 - p0) / n)
        
        # Critical value
        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed test
        
        # Critical boundaries
        critical_lower = p0 - z_alpha * se0
        critical_upper = p0 + z_alpha * se0
        
        # Standard error under alternative hypothesis
        se1 = np.sqrt(p1 * (1 - p1) / n)
        
        # Power calculation
        power = (stats.norm.cdf((critical_lower - p1) / se1) + 
                1 - stats.norm.cdf((critical_upper - p1) / se1))
        
        return max(0.0, min(1.0, power))
    
    def _calculate_required_sample_size(self, p0: float, p1: float, alpha: float, power: float) -> int:
        """Calculate required sample size for desired power."""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        numerator = (z_alpha * np.sqrt(p0 * (1 - p0)) + z_beta * np.sqrt(p1 * (1 - p1)))**2
        denominator = (p1 - p0)**2
        
        if denominator == 0:
            return float('inf')
        
        return int(np.ceil(numerator / denominator))
    
    def _generate_power_recommendations(self, power_results: Dict, current_n: int) -> List[str]:
        """Generate sample size recommendations based on power analysis."""
        recommendations = []
        
        for effect_name, results in power_results.items():
            if results['adequately_powered']:
                recommendations.append(
                    f"✓ Well-powered (Power={results['current_power']:.3f}) to detect {effect_name} effects"
                )
            else:
                recommendations.append(
                    f"⚠ Under-powered (Power={results['current_power']:.3f}) for {effect_name} effects. "
                    f"Need n={results['required_n_for_80_power']} for 80% power"
                )
        
        return recommendations
    
    # ====================================
    # 4. STATISTICAL REPORTING
    # ====================================
    
    def add_statistical_reporting(self, results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Add Cohen's h, sensitivity analyses, and robustness checks."""
        
        enhanced_results = results.copy()
        
        # Add Cohen's h effect sizes throughout results
        self._add_cohens_h_to_results(enhanced_results)
        
        # Add alternative statistical approaches
        enhanced_results['alternative_tests'] = self._run_alternative_tests(data)
        
        # Add assumption checking
        enhanced_results['assumption_checks'] = self._check_statistical_assumptions(data)
        
        return enhanced_results
    
    def _add_cohens_h_to_results(self, results: Dict[str, Any]) -> None:
        """Add Cohen's h effect sizes to all proportion tests."""
        
        def add_cohens_h_recursive(obj):
            if isinstance(obj, dict):
                if 'proportion' in obj and obj.get('proportion') is not None:
                    proportion = obj['proportion']
                    cohens_h = self._calculate_cohens_h(proportion, 0.5)
                    obj['cohens_h'] = cohens_h
                    obj['cohens_h_interpretation'] = self._interpret_cohens_h(cohens_h)
                
                for key, value in obj.items():
                    if isinstance(value, dict):
                        add_cohens_h_recursive(value)
        
        add_cohens_h_recursive(results)
    
    def _calculate_cohens_h(self, p1: float, p2: float) -> float:
        """Calculate Cohen's h for difference between two proportions."""
        if p1 is None or p2 is None or np.isnan(p1) or np.isnan(p2):
            return np.nan
        
        # Avoid numerical issues at boundaries
        p1 = max(0.001, min(0.999, p1))
        p2 = max(0.001, min(0.999, p2))
        
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    def _interpret_cohens_h(self, h: float) -> str:
        """Interpret Cohen's h effect size."""
        if np.isnan(h):
            return "Cannot determine"
        
        abs_h = abs(h)
        if abs_h < 0.2:
            return "Negligible effect"
        elif abs_h < 0.5:
            return "Small effect"
        elif abs_h < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def _run_alternative_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run alternative statistical approaches for validation."""
        
        alternative_results = {}
        
        # McNemar's test for paired data (participant-level)
        try:
            mcnemar_results = self._run_mcnemar_tests(data)
            alternative_results['mcnemar_tests'] = mcnemar_results
        except Exception as e:
            logger.warning(f"McNemar's tests failed: {e}")
            alternative_results['mcnemar_tests'] = {"error": str(e)}
        
        # Bootstrap confidence intervals
        try:
            bootstrap_results = self._run_bootstrap_analysis(data)
            alternative_results['bootstrap_analysis'] = bootstrap_results
        except Exception as e:
            logger.warning(f"Bootstrap analysis failed: {e}")
            alternative_results['bootstrap_analysis'] = {"error": str(e)}
        
        return alternative_results
    
    def _run_mcnemar_tests(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run McNemar's tests for key comparisons."""
        
        mcnemar_results = {}
        
        # Example: Home row preference at participant level
        participant_preferences = []
        
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id]
            
            home_chosen = 0
            home_unchosen = 0
            
            for _, row in user_data.iterrows():
                chosen_home = sum(1 for r in [row['chosen_bigram_row1'], row['chosen_bigram_row2']] if r == 2)
                unchosen_home = sum(1 for r in [row['unchosen_bigram_row1'], row['unchosen_bigram_row2']] if r == 2)
                
                home_chosen += chosen_home
                home_unchosen += unchosen_home
            
            if home_chosen + home_unchosen > 0:
                participant_preferences.append({
                    'user_id': user_id,
                    'prefers_home': home_chosen > home_unchosen
                })
        
        if len(participant_preferences) > 10:  # Need sufficient participants
            pref_df = pd.DataFrame(participant_preferences)
            n_prefers_home = pref_df['prefers_home'].sum()
            n_total = len(pref_df)
            
            # Simple binomial test at participant level
            binom_result = stats.binomtest(n_prefers_home, n_total, 0.5)
            
            mcnemar_results['home_row_participant_level'] = {
                'n_participants_prefer_home': n_prefers_home,
                'n_total_participants': n_total,
                'proportion': n_prefers_home / n_total,
                'p_value': binom_result.pvalue,
                'confidence_interval': binom_result.proportion_ci(confidence_level=0.95)
            }
        
        return mcnemar_results
    
    def _run_bootstrap_analysis(self, data: pd.DataFrame, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Run bootstrap analysis for confidence intervals."""
        
        np.random.seed(42)  # For reproducibility
        
        # Bootstrap home row preference
        home_proportions = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = data.sample(n=len(data), replace=True)
            
            comparison_data = []
            for _, row in bootstrap_sample.iterrows():
                chosen_home = sum(1 for r in [row['chosen_bigram_row1'], row['chosen_bigram_row2']] if r == 2)
                unchosen_home = sum(1 for r in [row['unchosen_bigram_row1'], row['unchosen_bigram_row2']] if r == 2)
                
                if chosen_home != unchosen_home:
                    comparison_data.append(chosen_home > unchosen_home)
            
            if comparison_data:
                proportion = np.mean(comparison_data)
                home_proportions.append(proportion)
        
        if home_proportions:
            bootstrap_results = {
                'home_row_preference': {
                    'bootstrap_mean': np.mean(home_proportions),
                    'bootstrap_std': np.std(home_proportions),
                    'bootstrap_ci_2_5': np.percentile(home_proportions, 2.5),
                    'bootstrap_ci_97_5': np.percentile(home_proportions, 97.5),
                    'n_bootstrap_samples': len(home_proportions)
                }
            }
        else:
            bootstrap_results = {'error': 'No valid bootstrap samples generated'}
        
        return bootstrap_results
    
    def _check_statistical_assumptions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check key statistical assumptions."""
        
        assumptions = {}
        
        # Independence check - repeated measures per participant
        participant_counts = data['user_id'].value_counts()
        assumptions['repeated_measures'] = {
            'mean_observations_per_participant': participant_counts.mean(),
            'std_observations_per_participant': participant_counts.std(),
            'min_observations_per_participant': participant_counts.min(),
            'max_observations_per_participant': participant_counts.max(),
            'participants_with_single_observation': (participant_counts == 1).sum(),
            'concern_level': 'HIGH' if participant_counts.mean() > 10 else 'MODERATE' if participant_counts.mean() > 5 else 'LOW'
        }
        
        # Sample size adequacy for normal approximation
        n_total = len(data)
        assumptions['normal_approximation'] = {
            'total_observations': n_total,
            'adequate_for_normal_approximation': n_total >= 30,
            'concern_level': 'LOW' if n_total >= 100 else 'MODERATE' if n_total >= 30 else 'HIGH'
        }
        
        return assumptions
    
    # ====================================
    # 5. INDIVIDUAL DIFFERENCES ANALYSIS
    # ====================================
    
    def analyze_individual_differences(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """Comprehensive individual differences analysis."""
        
        logger.info("Analyzing individual differences...")
        
        # Participant-level analysis
        participant_results = self._compute_participant_level_preferences(data)
        
        # Consistency analysis
        consistency_analysis = self._analyze_preference_consistency(participant_results)
        
        # Individual effect sizes
        individual_effects = self._compute_individual_effect_sizes(participant_results)
        
        # Subgroup analysis
        subgroup_analysis = self._run_subgroup_analyses(data)
        
        individual_differences = {
            'participant_preferences': participant_results,
            'consistency_analysis': consistency_analysis,
            'individual_effects': individual_effects,
            'subgroup_analysis': subgroup_analysis
        }
        
        # Save JSON data
        individual_path = os.path.join(output_folder, 'individual_differences.json')
        with open(individual_path, 'w') as f:
            json.dump(individual_differences, f, indent=2, default=str)
        
        # Save human-readable summary
        self._generate_individual_differences_summary(individual_differences, output_folder)
        
        return individual_differences
    
    def _compute_participant_level_preferences(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Compute preferences for each individual participant."""
        
        participant_prefs = {}
        
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id]
            
            prefs = {
                'user_id': user_id,
                'n_observations': len(user_data),
                'preferences': {}
            }
            
            # Home row preference
            home_comparisons = []
            for _, row in user_data.iterrows():
                chosen_home = sum(1 for r in [row['chosen_bigram_row1'], row['chosen_bigram_row2']] if r == 2)
                unchosen_home = sum(1 for r in [row['unchosen_bigram_row1'], row['unchosen_bigram_row2']] if r == 2)
                
                if chosen_home != unchosen_home:
                    home_comparisons.append(chosen_home > unchosen_home)
            
            if home_comparisons:
                prefs['preferences']['home_row'] = {
                    'proportion_prefer_home': np.mean(home_comparisons),
                    'n_comparisons': len(home_comparisons),
                    'effect_size': abs(np.mean(home_comparisons) - 0.5)
                }
            
            # Same row preference
            same_row_comparisons = []
            for _, row in user_data.iterrows():
                chosen_same = row['chosen_bigram_same_row']
                unchosen_same = row['unchosen_bigram_same_row']
                
                if chosen_same != unchosen_same:
                    same_row_comparisons.append(chosen_same)
            
            if same_row_comparisons:
                prefs['preferences']['same_row'] = {
                    'proportion_prefer_same_row': np.mean(same_row_comparisons),
                    'n_comparisons': len(same_row_comparisons),
                    'effect_size': abs(np.mean(same_row_comparisons) - 0.5)
                }
            
            participant_prefs[user_id] = prefs
        
        return participant_prefs
    
    def _analyze_preference_consistency(self, participant_results: Dict) -> Dict[str, Any]:
        """Analyze how consistent preferences are across participants."""
        
        consistency_results = {}
        
        for preference_type in ['home_row', 'same_row']:
            proportions = []
            effect_sizes = []
            
            for user_id, user_data in participant_results.items():
                if preference_type in user_data['preferences']:
                    pref_data = user_data['preferences'][preference_type]
                    proportions.append(pref_data['proportion_prefer_home' if preference_type == 'home_row' else 'proportion_prefer_same_row'])
                    effect_sizes.append(pref_data['effect_size'])
            
            if proportions:
                # Count participants showing each direction of preference
                strong_positive = sum(1 for p in proportions if p >= 0.75)
                moderate_positive = sum(1 for p in proportions if 0.6 <= p < 0.75)
                weak_positive = sum(1 for p in proportions if 0.5 < p < 0.6)
                weak_negative = sum(1 for p in proportions if 0.4 < p <= 0.5)
                moderate_negative = sum(1 for p in proportions if 0.25 < p <= 0.4)
                strong_negative = sum(1 for p in proportions if p <= 0.25)
                
                consistency_results[preference_type] = {
                    'n_participants': len(proportions),
                    'mean_proportion': np.mean(proportions),
                    'std_proportion': np.std(proportions),
                    'mean_effect_size': np.mean(effect_sizes),
                    'std_effect_size': np.std(effect_sizes),
                    'preference_distribution': {
                        'strong_positive': strong_positive,
                        'moderate_positive': moderate_positive,
                        'weak_positive': weak_positive,
                        'weak_negative': weak_negative,
                        'moderate_negative': moderate_negative,
                        'strong_negative': strong_negative
                    },
                    'proportion_showing_preference': sum(1 for p in proportions if p > 0.5) / len(proportions)
                }
        
        return consistency_results
    
    def _compute_individual_effect_sizes(self, participant_results: Dict) -> Dict[str, Any]:
        """Compute distribution of individual effect sizes."""
        
        effect_size_analysis = {}
        
        for preference_type in ['home_row', 'same_row']:
            effect_sizes = []
            
            for user_id, user_data in participant_results.items():
                if preference_type in user_data['preferences']:
                    effect_sizes.append(user_data['preferences'][preference_type]['effect_size'])
            
            if effect_sizes:
                effect_size_analysis[preference_type] = {
                    'mean_effect_size': np.mean(effect_sizes),
                    'median_effect_size': np.median(effect_sizes),
                    'std_effect_size': np.std(effect_sizes),
                    'min_effect_size': np.min(effect_sizes),
                    'max_effect_size': np.max(effect_sizes),
                    'q25_effect_size': np.percentile(effect_sizes, 25),
                    'q75_effect_size': np.percentile(effect_sizes, 75),
                    'large_effects_count': sum(1 for es in effect_sizes if es >= 0.25),
                    'medium_effects_count': sum(1 for es in effect_sizes if 0.15 <= es < 0.25),
                    'small_effects_count': sum(1 for es in effect_sizes if 0.05 <= es < 0.15),
                    'negligible_effects_count': sum(1 for es in effect_sizes if es < 0.05)
                }
        
        return effect_size_analysis
    
    def _run_subgroup_analyses(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run analyses for different subgroups if demographic data available."""
        
        # Placeholder for subgroup analysis
        # This would be expanded with actual demographic data
        subgroup_results = {
            'note': 'Subgroup analysis requires demographic data (age, gender, typing_experience, etc.)',
            'high_frequency_participants': self._analyze_high_frequency_participants(data),
            'consistency_subgroups': self._analyze_by_consistency_level(data)
        }
        
        return subgroup_results
    
    def _analyze_high_frequency_participants(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze participants with high number of observations."""
        
        participant_counts = data['user_id'].value_counts()
        high_freq_threshold = participant_counts.quantile(0.75)  # Top 25%
        high_freq_participants = participant_counts[participant_counts >= high_freq_threshold].index
        
        high_freq_data = data[data['user_id'].isin(high_freq_participants)]
        
        # Quick home row analysis for high-frequency participants
        home_comparisons = []
        for _, row in high_freq_data.iterrows():
            chosen_home = sum(1 for r in [row['chosen_bigram_row1'], row['chosen_bigram_row2']] if r == 2)
            unchosen_home = sum(1 for r in [row['unchosen_bigram_row1'], row['unchosen_bigram_row2']] if r == 2)
            
            if chosen_home != unchosen_home:
                home_comparisons.append(chosen_home > unchosen_home)
        
        return {
            'n_high_frequency_participants': len(high_freq_participants),
            'threshold_observations': high_freq_threshold,
            'n_observations': len(high_freq_data),
            'home_row_preference_proportion': np.mean(home_comparisons) if home_comparisons else None,
            'n_home_row_comparisons': len(home_comparisons)
        }
    
    def _analyze_by_consistency_level(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze participants by their overall consistency level."""
        
        participant_consistency = data.groupby('user_id')['is_consistent'].mean()
        
        # Define consistency groups
        high_consistency = participant_consistency >= 0.8
        medium_consistency = (participant_consistency >= 0.6) & (participant_consistency < 0.8)
        low_consistency = participant_consistency < 0.6
        
        return {
            'high_consistency_participants': high_consistency.sum(),
            'medium_consistency_participants': medium_consistency.sum(),
            'low_consistency_participants': low_consistency.sum(),
            'mean_consistency': participant_consistency.mean(),
            'std_consistency': participant_consistency.std()
        }
    
    # ====================================
    # 6. SENSITIVITY ANALYSES
    # ====================================
    
    def run_sensitivity_analyses(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """Run sensitivity analyses to test robustness."""
        
        sensitivity_results = {}
        
        # Analysis with different consistency thresholds
        sensitivity_results['consistency_thresholds'] = self._test_consistency_thresholds(data)
        
        # Analysis excluding potential outlier participants
        sensitivity_results['outlier_exclusion'] = self._test_outlier_exclusion(data)
        
        # Analysis with different statistical approaches
        sensitivity_results['statistical_approaches'] = self._test_alternative_statistical_approaches(data)
        
        # Save JSON data
        sensitivity_path = os.path.join(output_folder, 'sensitivity_analysis.json')
        with open(sensitivity_path, 'w') as f:
            json.dump(sensitivity_results, f, indent=2, default=str)
        
        # Save human-readable summary
        self._generate_sensitivity_analysis_summary(sensitivity_results, output_folder)
        
        return sensitivity_results
    
    def _test_consistency_thresholds(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test results with different consistency thresholds."""
        
        thresholds = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
        threshold_results = {}
        
        for threshold in thresholds:
            if threshold == 0.0:
                # Use all data
                filtered_data = data.copy()
            else:
                # Calculate participant-level consistency
                participant_consistency = data.groupby('user_id')['is_consistent'].mean()
                valid_participants = participant_consistency[participant_consistency >= threshold].index
                filtered_data = data[data['user_id'].isin(valid_participants)]
            
            if len(filtered_data) > 0:
                # Quick home row test
                home_comparisons = []
                for _, row in filtered_data.iterrows():
                    if row['is_consistent']:  # Still filter individual observations
                        chosen_home = sum(1 for r in [row['chosen_bigram_row1'], row['chosen_bigram_row2']] if r == 2)
                        unchosen_home = sum(1 for r in [row['unchosen_bigram_row1'], row['unchosen_bigram_row2']] if r == 2)
                        
                        if chosen_home != unchosen_home:
                            home_comparisons.append(chosen_home > unchosen_home)
                
                if home_comparisons:
                    n_prefer_home = sum(home_comparisons)
                    n_total = len(home_comparisons)
                    proportion = n_prefer_home / n_total
                    
                    threshold_results[f'threshold_{threshold}'] = {
                        'n_participants': filtered_data['user_id'].nunique(),
                        'n_observations': len(filtered_data),
                        'n_comparisons': n_total,
                        'home_row_proportion': proportion,
                        'effect_size': abs(proportion - 0.5)
                    }
        
        return threshold_results
    
    def _test_outlier_exclusion(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test results excluding potential outlier participants."""
        
        consistent_data = data[data['is_consistent']].copy()
        
        # Identify potential outliers based on number of observations
        participant_counts = consistent_data['user_id'].value_counts()
        
        # Define outliers as participants with very high or very low observation counts
        q75 = participant_counts.quantile(0.75)
        q25 = participant_counts.quantile(0.25)
        iqr = q75 - q25
        outlier_threshold_high = q75 + 1.5 * iqr
        outlier_threshold_low = max(1, q25 - 1.5 * iqr)
        
        outlier_participants = participant_counts[
            (participant_counts > outlier_threshold_high) | 
            (participant_counts < outlier_threshold_low)
        ].index
        
        # Analysis excluding outliers
        non_outlier_data = consistent_data[~consistent_data['user_id'].isin(outlier_participants)]
        
        # Quick home row test without outliers
        home_comparisons = []
        for _, row in non_outlier_data.iterrows():
            chosen_home = sum(1 for r in [row['chosen_bigram_row1'], row['chosen_bigram_row2']] if r == 2)
            unchosen_home = sum(1 for r in [row['unchosen_bigram_row1'], row['unchosen_bigram_row2']] if r == 2)
            
            if chosen_home != unchosen_home:
                home_comparisons.append(chosen_home > unchosen_home)
        
        return {
            'n_outlier_participants': len(outlier_participants),
            'outlier_threshold_high': outlier_threshold_high,
            'outlier_threshold_low': outlier_threshold_low,
            'n_participants_after_exclusion': non_outlier_data['user_id'].nunique(),
            'n_observations_after_exclusion': len(non_outlier_data),
            'home_row_proportion_without_outliers': np.mean(home_comparisons) if home_comparisons else None,
            'n_comparisons_without_outliers': len(home_comparisons)
        }
    
    def _test_alternative_statistical_approaches(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test different statistical approaches."""
        
        return {
            'note': 'Alternative approaches like mixed-effects models would be implemented here',
            'approaches_tested': [
                'Binomial tests (current)',
                'Mixed-effects logistic regression (recommended future work)',
                'Bootstrapping (implemented above)',
                'Bayesian analysis (future work)'
            ]
        }
    
    # ====================================
    # 8. STANDARDIZED FIGURES
    # ====================================
    
    def create_figures(self, results: Dict[str, Any], individual_differences: Dict[str, Any], output_folder: str) -> None:
        """Create standardized figures."""
        
        logger.info("Creating figures...")
        
        # Set style
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.dpi': 300
        })
        
        # Figure 1: Forest plot of main effects
        self._create_forest_plot(results, output_folder)
        
        # Figure 2: Individual differences distribution
        self._create_individual_differences_plot(individual_differences, output_folder)
        
        # Figure 3: Effect size distribution
        self._create_effect_size_distribution_plot(results, output_folder)
        
        # Figure 4: Consistency analysis plot
        self._create_consistency_plot(individual_differences, output_folder)
        
        logger.info("Figures created successfully!")
    
    def _create_forest_plot(self, results: Dict[str, Any], output_folder: str) -> None:
        """Create forest plot showing effect sizes with confidence intervals."""
        
        # Extract effect data for main tests
        effect_data = []
        
        def extract_effects(obj, prefix=""):
            if isinstance(obj, dict):
                if 'proportion' in obj and 'ci_lower' in obj and 'ci_upper' in obj:
                    test_name = obj.get('test_name', prefix)
                    if test_name and obj.get('n_comparisons', 0) > 0:
                        effect_data.append({
                            'test': test_name,
                            'proportion': obj['proportion'],
                            'ci_lower': obj['ci_lower'],
                            'ci_upper': obj['ci_upper'],
                            'effect_size': obj.get('effect_size', abs(obj['proportion'] - 0.5)),
                            'p_value': obj.get('p_value', np.nan),
                            'n_comparisons': obj.get('n_comparisons', 0),
                            'significant': obj.get('significant', False)
                        })
                
                for key, value in obj.items():
                    if isinstance(value, dict):
                        extract_effects(value, f"{prefix}.{key}" if prefix else key)
        
        extract_effects(results)
        
        if not effect_data:
            logger.warning("No effect data found for forest plot")
            return
        
        # Sort by effect size
        effect_data.sort(key=lambda x: x['effect_size'], reverse=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(effect_data) * 0.8)))
        
        y_positions = range(len(effect_data))
        
        # Plot confidence intervals
        for i, data in enumerate(effect_data):
            color = 'red' if data['significant'] else 'black'
            alpha = 1.0 if data['significant'] else 0.6
            
            # Convert to effect size scale (deviation from 0.5)
            center = data['proportion'] - 0.5
            ci_lower = data['ci_lower'] - 0.5
            ci_upper = data['ci_upper'] - 0.5
            
            # Plot CI line
            ax.plot([ci_lower, ci_upper], [i, i], color=color, alpha=alpha, linewidth=2)
            
            # Plot point estimate
            marker_size = min(100, max(30, data['n_comparisons'] / 10))  # Size based on sample size
            ax.scatter(center, i, s=marker_size, color=color, alpha=alpha, zorder=5)
            
            # Add sample size annotation
            ax.text(0.6, i, f"n={data['n_comparisons']}", va='center', fontsize=9, alpha=0.7)
        
        # Formatting
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([data['test'] for data in effect_data])
        ax.set_xlabel('Effect Size (Deviation from Chance)', fontweight='bold')
        ax.set_title('Forest Plot: Keyboard Ergonomics Effects', fontweight='bold', pad=20)
        
        # Add effect size interpretation zones
        for threshold, label, color in [(0.05, 'Small', 'lightblue'), 
                                       (0.15, 'Medium', 'lightgreen'), 
                                       (0.25, 'Large', 'lightcoral')]:
            ax.axvspan(threshold, 0.5, alpha=0.1, color=color)
            ax.axvspan(-threshold, -0.5, alpha=0.1, color=color)
            ax.text(threshold + 0.02, len(effect_data) - 0.5, label, rotation=90, 
                   va='top', ha='left', fontsize=8, alpha=0.7)
        
        # Legend
        significant_patch = plt.Line2D([0], [0], color='red', linewidth=2, label='Significant (p < 0.05)')
        non_significant_patch = plt.Line2D([0], [0], color='black', linewidth=2, alpha=0.6, label='Non-significant')
        ax.legend(handles=[significant_patch, non_significant_patch], loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'forest_plot.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_folder, 'forest_plot.pdf'), bbox_inches='tight')
        plt.close()
    
    def _create_individual_differences_plot(self, individual_differences: Dict[str, Any], output_folder: str) -> None:
        """Create plot showing individual differences in preferences."""
        
        consistency_data = individual_differences.get('consistency_analysis', {})
        
        if not consistency_data:
            logger.warning("No consistency data found for individual differences plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Home row preference distribution
        if 'home_row' in consistency_data:
            home_data = consistency_data['home_row']
            dist_data = home_data['preference_distribution']
            
            categories = ['Strong\nNegative', 'Moderate\nNegative', 'Weak\nNegative', 
                         'Weak\nPositive', 'Moderate\nPositive', 'Strong\nPositive']
            counts = [dist_data['strong_negative'], dist_data['moderate_negative'], 
                     dist_data['weak_negative'], dist_data['weak_positive'],
                     dist_data['moderate_positive'], dist_data['strong_positive']]
            
            colors = ['darkred', 'red', 'lightcoral', 'lightblue', 'blue', 'darkblue']
            
            bars = axes[0].bar(categories, counts, color=colors, alpha=0.7)
            axes[0].set_title('Home Row Preference Distribution\nAcross Participants', fontweight='bold')
            axes[0].set_ylabel('Number of Participants')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add percentage labels on bars
            total = sum(counts)
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{count}\n({count/total:.1%})', 
                               ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Same row preference distribution
        if 'same_row' in consistency_data:
            same_row_data = consistency_data['same_row']
            dist_data = same_row_data['preference_distribution']
            
            counts = [dist_data['strong_negative'], dist_data['moderate_negative'], 
                     dist_data['weak_negative'], dist_data['weak_positive'],
                     dist_data['moderate_positive'], dist_data['strong_positive']]
            
            bars = axes[1].bar(categories, counts, color=colors, alpha=0.7)
            axes[1].set_title('Same Row Preference Distribution\nAcross Participants', fontweight='bold')
            axes[1].set_ylabel('Number of Participants')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add percentage labels on bars
            total = sum(counts)
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{count}\n({count/total:.1%})', 
                               ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'individual_differences.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_folder, 'individual_differences.pdf'), bbox_inches='tight')
        plt.close()
    
    def _create_effect_size_distribution_plot(self, results: Dict[str, Any], output_folder: str) -> None:
        """Create histogram of effect sizes across all tests."""
        
        effect_sizes = []
        test_names = []
        
        def extract_effect_sizes(obj, prefix=""):
            if isinstance(obj, dict):
                if 'effect_size' in obj and obj.get('n_comparisons', 0) > 0:
                    effect_sizes.append(obj['effect_size'])
                    test_names.append(obj.get('test_name', prefix))
                
                for key, value in obj.items():
                    if isinstance(value, dict):
                        extract_effect_sizes(value, f"{prefix}.{key}" if prefix else key)
        
        extract_effect_sizes(results)
        
        if not effect_sizes:
            logger.warning("No effect sizes found for distribution plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        bins = np.arange(0, max(effect_sizes) + 0.05, 0.025)
        n, bins, patches = ax.hist(effect_sizes, bins=bins, alpha=0.7, edgecolor='black')
        
        # Color bars by effect size category
        for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
            if bin_center < 0.05:
                patch.set_facecolor('lightgray')
            elif bin_center < 0.15:
                patch.set_facecolor('lightblue')
            elif bin_center < 0.25:
                patch.set_facecolor('lightgreen')
            else:
                patch.set_facecolor('lightcoral')
        
        # Add vertical lines for thresholds
        ax.axvline(x=0.05, color='blue', linestyle='--', alpha=0.7, label='Small Effect (5pp)')
        ax.axvline(x=0.15, color='green', linestyle='--', alpha=0.7, label='Medium Effect (15pp)')
        ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.7, label='Large Effect (25pp)')
        
        ax.set_xlabel('Effect Size (Deviation from Chance)', fontweight='bold')
        ax.set_ylabel('Number of Tests', fontweight='bold')
        ax.set_title('Distribution of Effect Sizes Across All Tests', fontweight='bold')
        ax.legend()
        
        # Add summary statistics
        mean_effect = np.mean(effect_sizes)
        median_effect = np.median(effect_sizes)
        ax.text(0.6, 0.8, f'Mean: {mean_effect:.3f}\nMedian: {median_effect:.3f}\nTotal Tests: {len(effect_sizes)}',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'effect_size_distribution.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_folder, 'effect_size_distribution.pdf'), bbox_inches='tight')
        plt.close()
    
    def _create_consistency_plot(self, individual_differences: Dict[str, Any], output_folder: str) -> None:
        """Create plot showing consistency analysis."""
        
        individual_effects = individual_differences.get('individual_effects', {})
        
        if not individual_effects:
            logger.warning("No individual effects data found for consistency plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (preference_type, plot_title) in enumerate([('home_row', 'Home Row Preference'), 
                                                          ('same_row', 'Same Row Preference')]):
            if preference_type in individual_effects:
                data = individual_effects[preference_type]
                
                # Box plot of effect sizes
                effect_sizes = [data['min_effect_size'], data['q25_effect_size'], 
                               data['median_effect_size'], data['q75_effect_size'], 
                               data['max_effect_size']]
                
                box_data = [[data['min_effect_size']], [data['q25_effect_size']], 
                           [data['median_effect_size']], [data['q75_effect_size']], 
                           [data['max_effect_size']]]
                
                axes[i, 0].boxplot([effect_sizes], labels=[preference_type])
                axes[i, 0].set_title(f'{plot_title}\nEffect Size Distribution', fontweight='bold')
                axes[i, 0].set_ylabel('Effect Size')
                
                # Bar chart of effect size categories
                categories = ['Negligible', 'Small', 'Medium', 'Large']
                counts = [data['negligible_effects_count'], data['small_effects_count'],
                         data['medium_effects_count'], data['large_effects_count']]
                colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral']
                
                bars = axes[i, 1].bar(categories, counts, color=colors, alpha=0.7)
                axes[i, 1].set_title(f'{plot_title}\nEffect Size Categories', fontweight='bold')
                axes[i, 1].set_ylabel('Number of Participants')
                
                # Add count labels on bars
                for bar, count in zip(bars, counts):
                    if count > 0:
                        height = bar.get_height()
                        axes[i, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                       str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'consistency_analysis.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_folder, 'consistency_analysis.pdf'), bbox_inches='tight')
        plt.close()
    
    # ====================================
    # ADDITIONAL HELPER METHODS
    # ====================================
    
    def _add_keyboard_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add keyboard position features using existing keymaps."""
        enhanced_data = data.copy()
        
        for bigram_type in ['chosen_bigram', 'unchosen_bigram']:
            enhanced_data[f'{bigram_type}_char1'] = enhanced_data[bigram_type].str[0]
            enhanced_data[f'{bigram_type}_char2'] = enhanced_data[bigram_type].str[1]
            
            enhanced_data[f'{bigram_type}_row1'] = enhanced_data[f'{bigram_type}_char1'].map(row_map)
            enhanced_data[f'{bigram_type}_row2'] = enhanced_data[f'{bigram_type}_char2'].map(row_map)
            
            enhanced_data[f'{bigram_type}_col1'] = enhanced_data[f'{bigram_type}_char1'].map(column_map)
            enhanced_data[f'{bigram_type}_col2'] = enhanced_data[f'{bigram_type}_char2'].map(column_map)
            
            enhanced_data[f'{bigram_type}_finger1'] = enhanced_data[f'{bigram_type}_char1'].map(finger_map)
            enhanced_data[f'{bigram_type}_finger2'] = enhanced_data[f'{bigram_type}_char2'].map(finger_map)
            
            enhanced_data[f'{bigram_type}_same_row'] = (
                enhanced_data[f'{bigram_type}_row1'] == enhanced_data[f'{bigram_type}_row2']
            )
            enhanced_data[f'{bigram_type}_row_distance'] = abs(
                enhanced_data[f'{bigram_type}_row1'] - enhanced_data[f'{bigram_type}_row2']
            )
            enhanced_data[f'{bigram_type}_col_distance'] = abs(
                enhanced_data[f'{bigram_type}_col1'] - enhanced_data[f'{bigram_type}_col2']
            )
            enhanced_data[f'{bigram_type}_finger_distance'] = abs(
                enhanced_data[f'{bigram_type}_finger1'] - enhanced_data[f'{bigram_type}_finger2']
            )
            
            enhanced_data[f'{bigram_type}_direction_toward_5'] = (
                enhanced_data[f'{bigram_type}_col2'] > enhanced_data[f'{bigram_type}_col1']
            ).astype(int)
        
        return enhanced_data
    
    def run_all_ergonomics_tests(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """
        Run all 5 ergonomics research questions and generate comprehensive report.
        Enhanced version that ensures all results have consistent formatting.
        """
        logger.info("Running comprehensive ergonomics tests...")
        
        results = {}
        
        logger.info("Testing Question 1: Row preferences...")
        results['question_1'] = self.test_row_preferences(data)
        
        logger.info("Testing Question 2: Row pair preferences...")
        results['question_2'] = self.test_row_pair_preferences(data)
        
        logger.info("Testing Question 3: Column preferences...")
        results['question_3'] = self.test_column_preferences(data)
        
        logger.info("Testing Question 4: Column pair preferences...")
        results['question_4'] = self.test_column_pair_preferences(data)
        
        logger.info("Testing Question 5: Direction preferences...")
        results['question_5'] = self.test_direction_preferences(data)
        
        # Standardize all results to ensure consistent formatting
        logger.info("Standardizing test result formatting...")
        results = self._standardize_test_results(results)
        
        logger.info("All ergonomics tests complete!")
        return results

    # ====================================
    # COMPREHENSIVE ERGONOMICS TEST METHODS
    # ====================================
    
    def test_row_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 1: Row preferences
        - Is home row preferred over top/bottom?
        - Is top row preferred over bottom?
        - Row preferences by finger type
        """
        results = {}
        
        # Test 1a: Home row vs others
        home_vs_others = self._test_home_row_preference(data)
        results['home_vs_others'] = home_vs_others
        
        # Test 1b: Top vs bottom row
        top_vs_bottom = self._test_top_vs_bottom_preference(data)
        results['top_vs_bottom'] = top_vs_bottom
        
        # Test 1c: Row preferences by finger type
        finger_preferences = self._test_row_preferences_by_finger(data)
        results['finger_preferences'] = finger_preferences
        
        return results
    
    def test_row_pair_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 2: Row pair preferences
        - Same row vs different row bigrams
        - Adjacent rows vs non-adjacent rows
        """
        results = {}
        
        # Test same row vs different row
        same_vs_diff = self._test_same_vs_different_row(data)
        results['same_vs_different_row'] = same_vs_diff
        
        # Test adjacent vs non-adjacent rows
        adjacent_vs_nonadjacent = self._test_adjacent_vs_nonadjacent_rows(data)
        results['adjacent_vs_nonadjacent'] = adjacent_vs_nonadjacent
        
        return results
    
    def test_column_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 3: Column preferences
        - Sequential column comparisons (5>4, 4>3, 3>2, 2>1)
        - Column 5 vs all other columns (1-4)
        """
        results = {}
        
        # Test each adjacent column pair
        column_comparisons = [
            (5, 4, "col5_vs_col4"),
            (4, 3, "col4_vs_col3"), 
            (3, 2, "col3_vs_col2"),
            (2, 1, "col2_vs_col1")
        ]
        
        for col_high, col_low, test_name in column_comparisons:
            result = self._test_column_preference(data, col_high, col_low)
            results[test_name] = result
        
        # Test column 5 vs all other columns individually
        col5_vs_others = [
            (5, 1, "col5_vs_col1"),
            (5, 2, "col5_vs_col2"), 
            (5, 3, "col5_vs_col3"),
        ]
        
        for col_high, col_low, test_name in col5_vs_others:
            result = self._test_column_preference(data, col_high, col_low)
            results[test_name] = result
        
        # Combined test - is column 5 systematically avoided?
        col5_avoidance_result = self._test_column5_avoidance(data)
        results['col5_avoidance_overall'] = col5_avoidance_result
            
        return results
    
    def test_column_pair_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 4: Column pair preferences
        - Adjacent vs remote fingers (within same row and across different rows)
        - Different separation distances (within same row and across different rows)
        """
        results = {}
        
        # Test adjacent vs remote fingers overall
        adjacent_vs_remote = self._test_adjacent_vs_remote_fingers(data)
        results['adjacent_vs_remote_overall'] = adjacent_vs_remote
        
        # Test adjacent vs remote fingers: same-row bigrams only
        same_row_comparisons = data[
            data['chosen_bigram_same_row'] & data['unchosen_bigram_same_row']
        ]
        if len(same_row_comparisons) > 0:
            adjacent_vs_remote_same_row = self._test_adjacent_vs_remote_fingers(same_row_comparisons)
            results['adjacent_vs_remote_same_row'] = adjacent_vs_remote_same_row
        else:
            results['adjacent_vs_remote_same_row'] = {'test': 'No same-row comparisons found', 'p_value': np.nan}
        
        # Test adjacent vs remote fingers: cross-row bigrams only
        diff_row_comparisons = data[
            (~data['chosen_bigram_same_row']) & (~data['unchosen_bigram_same_row'])
        ]
        if len(diff_row_comparisons) > 0:
            adjacent_vs_remote_diff_row = self._test_adjacent_vs_remote_fingers(diff_row_comparisons)
            results['adjacent_vs_remote_diff_row'] = adjacent_vs_remote_diff_row
        else:
            results['adjacent_vs_remote_diff_row'] = {'test': 'No cross-row comparisons found', 'p_value': np.nan}
        
        # Test different finger separation distances overall
        separation_distances = self._test_finger_separation_distances(data)
        results['separation_distances_overall'] = separation_distances
        
        # Test separation distances: same-row bigrams only
        if len(same_row_comparisons) > 0:
            separation_distances_same_row = self._test_finger_separation_distances(same_row_comparisons)
            results['separation_distances_same_row'] = separation_distances_same_row
        else:
            results['separation_distances_same_row'] = {}
        
        # Test separation distances: cross-row bigrams only
        if len(diff_row_comparisons) > 0:
            separation_distances_diff_row = self._test_finger_separation_distances(diff_row_comparisons)
            results['separation_distances_diff_row'] = separation_distances_diff_row
        else:
            results['separation_distances_diff_row'] = {}
        
        return results
    
    def test_direction_preferences(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Question 5: Direction preferences
        - Low-to-high finger sequences vs high-to-low
        """
        results = {}
        
        # Test directional preference
        direction_preference = self._test_directional_preference(data)
        results['direction_preference'] = direction_preference
        
        return results
    
    def _calculate_proportion_ci(self, successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval for a proportion."""
        if trials == 0:
            return np.nan, np.nan
            
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        p = successes / trials
        
        denominator = 1 + z**2 / trials
        center = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)
        
        return ci_lower, ci_upper
    
    def _compute_sample_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute basic sample characteristics."""
        return {
            'total_participants': data['user_id'].nunique(),
            'total_observations': len(data),
            'observations_per_participant': {
                'mean': data.groupby('user_id').size().mean(),
                'std': data.groupby('user_id').size().std(),
                'min': data.groupby('user_id').size().min(),
                'max': data.groupby('user_id').size().max()
            }
        }
    
    # ====================================
    # INDIVIDUAL TEST IMPLEMENTATIONS
    # ====================================
    
    def _test_home_row_preference(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if home row (row 2) is preferred over top/bottom rows (rows 1&3)"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_rows = (row['chosen_bigram_row1'], row['chosen_bigram_row2'])
            unchosen_rows = (row['unchosen_bigram_row1'], row['unchosen_bigram_row2'])
            
            # Score each bigram: +1 for each key on home row
            chosen_home_score = sum(1 for r in chosen_rows if r == 2)
            unchosen_home_score = sum(1 for r in unchosen_rows if r == 2)
            
            if chosen_home_score != unchosen_home_score:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_more_home': chosen_home_score > unchosen_home_score,
                    'home_difference': chosen_home_score - unchosen_home_score
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {
                'test_name': 'Home row preference',
                'test': 'No valid comparisons found', 
                'p_value': np.nan,
                'n_comparisons': 0,
                'proportion': np.nan,
                'effect_size': np.nan,
                'significant': False
            }
        
        n_chose_more_home = comparison_df['chose_more_home'].sum()
        n_total = len(comparison_df)
        proportion = n_chose_more_home / n_total
        
        p_value = stats.binomtest(n_chose_more_home, n_total, 0.5, alternative='two-sided').pvalue
        
        base_result = {
            'test_name': 'Home row preference',
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
        
        result = self._add_enhanced_reporting(
            base_result, n_chose_more_home, n_total,
            "Home row keys chosen over non-home row keys"
        )
        
        return result
    
    def _test_top_vs_bottom_preference(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if top row (row 3) is preferred over bottom row (row 1)"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_rows = (row['chosen_bigram_row1'], row['chosen_bigram_row2'])
            unchosen_rows = (row['unchosen_bigram_row1'], row['unchosen_bigram_row2'])
            
            # Score: +1 for top row, -1 for bottom row
            def score_bigram(rows):
                score = 0
                for r in rows:
                    if r == 3:  # top row
                        score += 1
                    elif r == 1:  # bottom row
                        score -= 1
                return score
            
            chosen_score = score_bigram(chosen_rows)
            unchosen_score = score_bigram(unchosen_rows)
            
            if chosen_score != unchosen_score:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_more_top': chosen_score > unchosen_score,
                    'top_bottom_difference': chosen_score - unchosen_score
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_more_top = comparison_df['chose_more_top'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_more_top, n_total, 0.5, alternative='two-sided').pvalue
        
        base_result = {
            'test_name': 'Top vs bottom row preference',
            'n_comparisons': n_total,
            'n_chose_more_top': n_chose_more_top,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
        
        result = self._add_enhanced_reporting(
            base_result, n_chose_more_top, n_total,
            "Top row keys chosen over bottom row keys"
        )
        
        return result
    
    def _test_row_preferences_by_finger(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test row preferences for different finger types"""
        
        finger_results = {}
        
        # Test each finger (1=index to 4=pinky)
        for finger_num in range(1, 5):
            finger_name = ['index', 'middle', 'ring', 'pinky'][finger_num - 1]
            
            comparison_data = []
            
            for _, row in data.iterrows():
                # Extract positions that use this finger
                chosen_finger_positions = []
                unchosen_finger_positions = []
                
                if row['chosen_bigram_finger1'] == finger_num:
                    chosen_finger_positions.append(row['chosen_bigram_row1'])
                if row['chosen_bigram_finger2'] == finger_num:
                    chosen_finger_positions.append(row['chosen_bigram_row2'])
                    
                if row['unchosen_bigram_finger1'] == finger_num:
                    unchosen_finger_positions.append(row['unchosen_bigram_row1'])
                if row['unchosen_bigram_finger2'] == finger_num:
                    unchosen_finger_positions.append(row['unchosen_bigram_row2'])
                
                # Compare top vs bottom for this finger
                if chosen_finger_positions and unchosen_finger_positions:
                    chosen_top_score = sum(1 for r in chosen_finger_positions if r == 3)
                    chosen_bottom_score = sum(1 for r in chosen_finger_positions if r == 1)
                    
                    unchosen_top_score = sum(1 for r in unchosen_finger_positions if r == 3)
                    unchosen_bottom_score = sum(1 for r in unchosen_finger_positions if r == 1)
                    
                    chosen_preference = chosen_top_score - chosen_bottom_score
                    unchosen_preference = unchosen_top_score - unchosen_bottom_score
                    
                    if chosen_preference != unchosen_preference:
                        comparison_data.append({
                            'user_id': row['user_id'],
                            'chose_more_top': chosen_preference > unchosen_preference
                        })
            
            if len(comparison_data) > 0:
                comparison_df = pd.DataFrame(comparison_data)
                n_chose_more_top = comparison_df['chose_more_top'].sum()
                n_total = len(comparison_df)
                
                p_value = stats.binomtest(n_chose_more_top, n_total, 0.5, alternative='two-sided').pvalue
                
                finger_results[finger_name] = {
                    'n_comparisons': n_total,
                    'n_chose_more_top': n_chose_more_top,
                    'proportion_chose_top': n_chose_more_top / n_total,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
            else:
                finger_results[finger_name] = {'n_comparisons': 0, 'p_value': np.nan}
        
        return finger_results
    
    def _test_same_vs_different_row(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if same-row bigrams are preferred over different-row bigrams"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_same_row = row['chosen_bigram_same_row']
            unchosen_same_row = row['unchosen_bigram_same_row']
            
            if chosen_same_row != unchosen_same_row:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_same_row': chosen_same_row
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_same_row = comparison_df['chose_same_row'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_same_row, n_total, 0.5, alternative='two-sided').pvalue
        
        base_result = {
            'test_name': 'Same vs different row preference',
            'n_comparisons': n_total,
            'n_chose_same_row': n_chose_same_row,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
        
        result = self._add_enhanced_reporting(
            base_result, n_chose_same_row, n_total,
            "Same-row bigrams chosen over cross-row bigrams"
        )
        
        return result
    
    def _test_adjacent_vs_nonadjacent_rows(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if adjacent-row bigrams are preferred over non-adjacent (skip) row bigrams"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_row_dist = row['chosen_bigram_row_distance']
            unchosen_row_dist = row['unchosen_bigram_row_distance']
            
            # Only compare when one is adjacent (distance 1) and other is skip (distance 2)
            if (chosen_row_dist == 1 and unchosen_row_dist == 2) or (chosen_row_dist == 2 and unchosen_row_dist == 1):
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_adjacent': chosen_row_dist == 1
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_adjacent = comparison_df['chose_adjacent'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_adjacent, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': 'Adjacent vs non-adjacent rows',
            'n_comparisons': n_total,
            'n_chose_adjacent': n_chose_adjacent,
            'proportion_chose_adjacent': n_chose_adjacent / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_adjacent / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _test_column_preference(self, data: pd.DataFrame, col_high: int, col_low: int) -> Dict[str, Any]:
        """Test preference between two specific columns"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            # Count occurrences of each column in chosen vs unchosen
            chosen_cols = (row['chosen_bigram_col1'], row['chosen_bigram_col2'])
            unchosen_cols = (row['unchosen_bigram_col1'], row['unchosen_bigram_col2'])
            
            chosen_high_count = sum(1 for c in chosen_cols if c == col_high)
            chosen_low_count = sum(1 for c in chosen_cols if c == col_low)
            
            unchosen_high_count = sum(1 for c in unchosen_cols if c == col_high)
            unchosen_low_count = sum(1 for c in unchosen_cols if c == col_low)
            
            chosen_score = chosen_high_count - chosen_low_count
            unchosen_score = unchosen_high_count - unchosen_low_count
            
            if chosen_score != unchosen_score:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_higher_col': chosen_score > unchosen_score
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_higher = comparison_df['chose_higher_col'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_higher, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': f'Column {col_high} vs Column {col_low}',
            'n_comparisons': n_total,
            'n_chose_higher_col': n_chose_higher,
            'proportion_chose_higher': n_chose_higher / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_higher / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _test_column5_avoidance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if column 5 is systematically avoided compared to columns 1-4"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            # Count occurrences of column 5 vs columns 1-4 in chosen vs unchosen
            chosen_cols = (row['chosen_bigram_col1'], row['chosen_bigram_col2'])
            unchosen_cols = (row['unchosen_bigram_col1'], row['unchosen_bigram_col2'])
            
            chosen_col5_count = sum(1 for c in chosen_cols if c == 5)
            chosen_others_count = sum(1 for c in chosen_cols if c in [1, 2, 3, 4])
            
            unchosen_col5_count = sum(1 for c in unchosen_cols if c == 5)
            unchosen_others_count = sum(1 for c in unchosen_cols if c in [1, 2, 3, 4])
            
            # Score: +1 for each non-col5 key, -1 for each col5 key
            chosen_score = chosen_others_count - chosen_col5_count
            unchosen_score = unchosen_others_count - unchosen_col5_count
            
            if chosen_score != unchosen_score:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_more_non_col5': chosen_score > unchosen_score,
                    'avoidance_difference': chosen_score - unchosen_score
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_more_non_col5 = comparison_df['chose_more_non_col5'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_more_non_col5, n_total, 0.5, alternative='two-sided').pvalue
        
        base_result = {
            'test_name': 'Column 5 avoidance (vs columns 1-4)',
            'n_comparisons': n_total,
            'n_chose_more_non_col5': n_chose_more_non_col5,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
        
        result = self._add_enhanced_reporting(
            base_result, n_chose_more_non_col5, n_total,
            "Columns 1-4 chosen over column 5"
        )
        
        return result
    
    def _test_adjacent_vs_remote_fingers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if adjacent fingers are preferred over remote fingers"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_finger_dist = row['chosen_bigram_finger_distance']
            unchosen_finger_dist = row['unchosen_bigram_finger_distance']
            
            # Compare adjacent (distance 1) vs remote (distance > 1)
            chosen_adjacent = chosen_finger_dist == 1
            unchosen_adjacent = unchosen_finger_dist == 1
            
            if chosen_adjacent != unchosen_adjacent:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_adjacent': chosen_adjacent
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_adjacent = comparison_df['chose_adjacent'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_adjacent, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': 'Adjacent vs remote fingers',
            'n_comparisons': n_total,
            'n_chose_adjacent': n_chose_adjacent,
            'proportion_chose_adjacent': n_chose_adjacent / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_adjacent / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _test_finger_separation_distances(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test preferences for different finger separation distances"""
        
        distance_results = {}
        
        # Test each distance comparison
        distance_comparisons = [
            (1, 2, "1_finger_vs_2_finger"),
            (1, 3, "1_finger_vs_3_finger"), 
            (2, 3, "2_finger_vs_3_finger")
        ]
        
        for dist1, dist2, comparison_name in distance_comparisons:
            comparison_data = []
            
            for _, row in data.iterrows():
                chosen_dist = row['chosen_bigram_finger_distance']
                unchosen_dist = row['unchosen_bigram_finger_distance']
                
                if (chosen_dist == dist1 and unchosen_dist == dist2) or (chosen_dist == dist2 and unchosen_dist == dist1):
                    comparison_data.append({
                        'user_id': row['user_id'],
                        'chose_smaller_distance': (chosen_dist == dist1) if dist1 < dist2 else (chosen_dist == dist2)
                    })
            
            if len(comparison_data) > 0:
                comparison_df = pd.DataFrame(comparison_data)
                n_chose_smaller = comparison_df['chose_smaller_distance'].sum()
                n_total = len(comparison_df)
                
                p_value = stats.binomtest(n_chose_smaller, n_total, 0.5, alternative='two-sided').pvalue
                
                distance_results[comparison_name] = {
                    'n_comparisons': n_total,
                    'n_chose_smaller_distance': n_chose_smaller,
                    'proportion_chose_smaller': n_chose_smaller / n_total,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
            else:
                distance_results[comparison_name] = {'n_comparisons': 0, 'p_value': np.nan}
        
        return distance_results
    
    def _test_directional_preference(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test if low-to-high finger sequences are preferred over high-to-low"""
        
        comparison_data = []
        
        for _, row in data.iterrows():
            chosen_toward_5 = row['chosen_bigram_direction_toward_5']
            unchosen_toward_5 = row['unchosen_bigram_direction_toward_5']
            
            if chosen_toward_5 != unchosen_toward_5:
                comparison_data.append({
                    'user_id': row['user_id'],
                    'chose_toward_5': chosen_toward_5 == 1
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            return {'test': 'No valid comparisons found', 'p_value': np.nan}
        
        n_chose_toward_5 = comparison_df['chose_toward_5'].sum()
        n_total = len(comparison_df)
        
        p_value = stats.binomtest(n_chose_toward_5, n_total, 0.5, alternative='two-sided').pvalue
        
        result = {
            'test_name': 'Direction toward column 5 (low-to-high fingers)',
            'n_comparisons': n_total,
            'n_chose_toward_5': n_chose_toward_5,
            'proportion_chose_toward_5': n_chose_toward_5 / n_total,
            'p_value': p_value,
            'effect_size': abs(0.5 - (n_chose_toward_5 / n_total)),
            'significant': p_value < self.alpha
        }
        
        return result
    
    def _add_enhanced_reporting(self, base_result: Dict[str, Any], 
                            successes: int, trials: int, 
                            success_description: str) -> Dict[str, Any]:
        """Add enhanced reporting with effect sizes, CIs, and interpretation."""
        
        if trials == 0:
            return {
                **base_result, 
                'interpretation': 'No valid data for analysis',
                'n_comparisons': 0,
                'proportion': np.nan,
                'effect_size': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'cohens_h': np.nan,
                'cohens_h_interpretation': 'Cannot determine',
                'practical_significance': 'No data',
                'design_priority': 'NO DATA'
            }
        
        proportion = successes / trials
        
        # Calculate confidence interval
        ci_lower, ci_upper = self._calculate_proportion_ci(successes, trials)
        
        # Effect size (deviation from chance)
        effect_size = abs(proportion - 0.5)
        
        # Cohen's h effect size
        cohens_h = self._calculate_cohens_h(proportion, 0.5)
        cohens_h_interpretation = self._interpret_cohens_h(cohens_h)
        
        # Practical significance categorization
        if effect_size >= 0.20:
            practical_significance = "Large effect"
            design_priority = "HIGH PRIORITY"
        elif effect_size >= 0.10:
            practical_significance = "Medium effect"
            design_priority = "MEDIUM PRIORITY"
        elif effect_size >= 0.05:
            practical_significance = "Small effect"
            design_priority = "LOW PRIORITY"
        else:
            practical_significance = "Trivial effect"
            design_priority = "IGNORE"
        
        # Interpretation with sample size
        interpretation = (f"{success_description} {proportion:.1%} of the time "
                        f"(n = {trials} comparisons, 95% CI: {ci_lower:.1%}-{ci_upper:.1%})")
        
        enhanced_result = {
            **base_result,
            'n_comparisons': trials,  # Ensure this is always present
            'n_successes': successes,  # Add number of successes for clarity
            'proportion': proportion,
            'effect_size': effect_size,
            'cohens_h': cohens_h,
            'cohens_h_interpretation': cohens_h_interpretation,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'practical_significance': practical_significance,
            'design_priority': design_priority,
            'interpretation': interpretation
        }
        
        return enhanced_result

    def _apply_multiple_comparison_correction(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple comparison correction across all tests"""
        
        # Collect all p-values
        all_p_values = []
        test_names = []
        
        def extract_p_values(obj, prefix=""):
            if isinstance(obj, dict):
                if 'p_value' in obj and not np.isnan(obj['p_value']):
                    all_p_values.append(obj['p_value'])
                    test_names.append(prefix)
                for key, value in obj.items():
                    if isinstance(value, dict):
                        extract_p_values(value, f"{prefix}.{key}" if prefix else key)
        
        extract_p_values(results)
        
        if len(all_p_values) == 0:
            return results
        
        # Apply correction
        if self.correction_method == 'fdr_bh':
            from statsmodels.stats.multitest import multipletests
            reject, p_corrected, _, _ = multipletests(all_p_values, method='fdr_bh', alpha=self.alpha)
        else:
            # Bonferroni correction
            p_corrected = [p * len(all_p_values) for p in all_p_values]
            reject = [p <= self.alpha for p in p_corrected]
        
        # Update results with corrected p-values
        correction_info = {
            'method': self.correction_method,
            'n_tests': len(all_p_values),
            'alpha_level': self.alpha,
            'test_results': []
        }
        
        for i, (test_name, orig_p, corr_p, significant) in enumerate(zip(test_names, all_p_values, p_corrected, reject)):
            correction_info['test_results'].append({
                'test': test_name,
                'original_p': orig_p,
                'corrected_p': corr_p,
                'significant_corrected': significant
            })
        
        results['multiple_comparison_correction'] = correction_info
        return results

    # ====================================
    # HUMAN-READABLE SUMMARY GENERATORS
    # ====================================
    
    def _generate_power_analysis_summary(self, power_data: Dict[str, Any], output_folder: str) -> None:
        """Generate human-readable summary of power analysis JSON data."""
        
        summary_lines = [
            "POWER ANALYSIS SUMMARY",
            "=" * 40,
            "",
            "OVERVIEW:",
            f"• Total participants: {power_data['total_participants']}",
            f"• Total observations: {power_data['total_observations']}",
            f"• Analysis significance level: α = {self.alpha}",
            "",
            "STATISTICAL POWER FOR DETECTING EFFECTS:",
            "----------------------------------------"
        ]
        
        for effect_name, effect_data in power_data['power_by_effect_size'].items():
            power_pct = effect_data['current_power'] * 100
            status = "✓ WELL-POWERED" if effect_data['adequately_powered'] else "⚠ UNDER-POWERED"
            
            summary_lines.extend([
                f"",
                f"{effect_name.upper()} EFFECTS ({effect_data['effect_size_description']}):",
                f"  Current power: {power_pct:.1f}% {status}",
                f"  Sample size needed for 80% power: {effect_data['required_n_for_80_power']} observations",
                f"  Current sample size: {effect_data['current_n']} observations"
            ])
        
        summary_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "---------------"
        ])
        
        for i, rec in enumerate(power_data['recommendations'], 1):
            summary_lines.append(f"{i}. {rec}")
        
        summary_lines.extend([
            "",
            "INTERPRETATION:",
            "--------------",
            "• Power = probability of detecting an effect if it truly exists",
            "• Effects are measured as percentage points deviation from chance (50%)",
            "• Small effects (5pp): Subtle preferences, may require large samples",
            "• Medium effects (15pp): Noticeable preferences, practical significance",
            "• Large effects (25pp): Strong preferences, clear practical importance",
            "",
            "DATA SOURCE:",
            "• Complete data preserved in: power_analysis.json",
            "• Use JSON file for programmatic analysis or meta-analysis"
        ])
        
        # Save summary
        summary_path = os.path.join(output_folder, 'power_analysis_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Power analysis summary saved to {summary_path}")
    
    def _generate_individual_differences_summary(self, individual_data: Dict[str, Any], output_folder: str) -> None:
        """Generate human-readable summary of individual differences JSON data."""
        
        summary_lines = [
            "INDIVIDUAL DIFFERENCES SUMMARY",
            "=" * 50,
            "",
            "OVERVIEW:",
            "This file contains participant-level analysis showing how",
            "ergonomic preferences vary across individuals.",
            ""
        ]
        
        # Participant-level overview
        if 'participant_preferences' in individual_data:
            n_participants = len(individual_data['participant_preferences'])
            summary_lines.extend([
                f"PARTICIPANT-LEVEL DATA:",
                f"• {n_participants} participants analyzed",
                f"• Each participant's individual preferences calculated",
                f"• Includes effect sizes and number of valid comparisons per person",
                ""
            ])
        
        # Consistency analysis
        if 'consistency_analysis' in individual_data:
            consistency = individual_data['consistency_analysis']
            
            summary_lines.extend([
                "CONSISTENCY ACROSS PARTICIPANTS:",
                "-------------------------------"
            ])
            
            for preference_type in ['home_row', 'same_row']:
                if preference_type in consistency:
                    data = consistency[preference_type]
                    pref_name = "Home Row" if preference_type == 'home_row' else "Same Row"
                    
                    total_participants = data.get('n_participants', 0)
                    showing_preference = data.get('proportion_showing_preference', 0) * 100
                    mean_effect = data.get('mean_effect_size', 0)
                    std_effect = data.get('std_effect_size', 0)
                    
                    # Distribution breakdown
                    dist = data.get('preference_distribution', {})
                    strong_positive = dist.get('strong_positive', 0)
                    moderate_positive = dist.get('moderate_positive', 0)
                    weak_positive = dist.get('weak_positive', 0)
                    
                    summary_lines.extend([
                        f"",
                        f"{pref_name} Preference:",
                        f"  • {total_participants} participants with valid data",
                        f"  • {showing_preference:.1f}% show preference in expected direction",
                        f"  • Average individual effect size: {mean_effect:.3f} ± {std_effect:.3f}",
                        f"  • Strong preference: {strong_positive} participants",
                        f"  • Moderate preference: {moderate_positive} participants", 
                        f"  • Weak preference: {weak_positive} participants"
                    ])
        
        # Individual effect sizes
        if 'individual_effects' in individual_data:
            effects = individual_data['individual_effects']
            
            summary_lines.extend([
                "",
                "INDIVIDUAL EFFECT SIZE DISTRIBUTION:",
                "-----------------------------------"
            ])
            
            for preference_type in ['home_row', 'same_row']:
                if preference_type in effects:
                    data = effects[preference_type]
                    pref_name = "Home Row" if preference_type == 'home_row' else "Same Row"
                    
                    summary_lines.extend([
                        f"",
                        f"{pref_name} Effect Sizes:",
                        f"  • Mean: {data.get('mean_effect_size', 0):.3f}",
                        f"  • Median: {data.get('median_effect_size', 0):.3f}",
                        f"  • Range: {data.get('min_effect_size', 0):.3f} - {data.get('max_effect_size', 0):.3f}",
                        f"  • Large effects (≥0.25): {data.get('large_effects_count', 0)} participants",
                        f"  • Medium effects (0.15-0.25): {data.get('medium_effects_count', 0)} participants",
                        f"  • Small effects (0.05-0.15): {data.get('small_effects_count', 0)} participants",
                        f"  • Negligible effects (<0.05): {data.get('negligible_effects_count', 0)} participants"
                    ])
        
        # Subgroup analysis
        if 'subgroup_analysis' in individual_data:
            subgroup = individual_data['subgroup_analysis']
            
            summary_lines.extend([
                "",
                "SUBGROUP ANALYSIS:",
                "----------------"
            ])
            
            if 'high_frequency_participants' in subgroup:
                hf_data = subgroup['high_frequency_participants']
                summary_lines.extend([
                    f"High-frequency participants (top 25% by observation count):",
                    f"  • {hf_data.get('n_high_frequency_participants', 0)} participants",
                    f"  • {hf_data.get('n_observations', 0)} total observations",
                    f"  • Home row preference: {hf_data.get('home_row_preference_proportion', 'N/A'):.1%}" if hf_data.get('home_row_preference_proportion') else "  • Home row preference: N/A"
                ])
            
            if 'consistency_subgroups' in subgroup:
                cs_data = subgroup['consistency_subgroups']
                summary_lines.extend([
                    f"",
                    f"Participants by consistency level:",
                    f"  • High consistency (≥80%): {cs_data.get('high_consistency_participants', 0)} participants",
                    f"  • Medium consistency (60-80%): {cs_data.get('medium_consistency_participants', 0)} participants",
                    f"  • Low consistency (<60%): {cs_data.get('low_consistency_participants', 0)} participants",
                    f"  • Mean consistency: {cs_data.get('mean_consistency', 0):.1%}"
                ])
        
        summary_lines.extend([
            "",
            "KEY INSIGHTS:",
            "------------",
            "• Individual differences show how much ergonomic preferences vary",
            "• High consistency suggests reliable individual preferences",
            "• Effect size distribution shows population heterogeneity",
            "• Strong individual effects may indicate design opportunities",
            "",
            "DATA SOURCE:",
            "• Complete data preserved in: individual_differences.json",
            "• Use JSON file for participant-level reanalysis or modeling"
        ])
        
        # Save summary
        summary_path = os.path.join(output_folder, 'individual_differences_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Individual differences summary saved to {summary_path}")
    
    def _generate_sensitivity_analysis_summary(self, sensitivity_data: Dict[str, Any], output_folder: str) -> None:
        """Generate human-readable summary of sensitivity analysis JSON data."""
        
        summary_lines = [
            "SENSITIVITY ANALYSIS SUMMARY",
            "=" * 45,
            "",
            "OVERVIEW:",
            "This file contains robustness checks testing whether results",
            "depend on specific analytical choices or assumptions.",
            ""
        ]
        
        # Consistency thresholds
        if 'consistency_thresholds' in sensitivity_data:
            thresholds = sensitivity_data['consistency_thresholds']
            
            summary_lines.extend([
                "CONSISTENCY THRESHOLD ANALYSIS:",
                "------------------------------",
                "Tests whether results change when using different participant",
                "consistency requirements (% of trials where participant is consistent).",
                ""
            ])
            
            threshold_keys = sorted([k for k in thresholds.keys() if k.startswith('threshold_')])
            
            for threshold_key in threshold_keys:
                data = thresholds[threshold_key]
                threshold_val = threshold_key.replace('threshold_', '')
                
                if isinstance(data, dict) and 'n_participants' in data:
                    summary_lines.extend([
                        f"Threshold {threshold_val} (≥{float(threshold_val)*100:.0f}% consistent):",
                        f"  • Participants included: {data.get('n_participants', 0)}",
                        f"  • Total observations: {data.get('n_observations', 0)}",
                        f"  • Home row preference: {data.get('home_row_proportion', 0):.1%}",
                        f"  • Effect size: {data.get('effect_size', 0):.3f}",
                        ""
                    ])
        
        # Outlier exclusion
        if 'outlier_exclusion' in sensitivity_data:
            outlier_data = sensitivity_data['outlier_exclusion']
            
            summary_lines.extend([
                "OUTLIER PARTICIPANT EXCLUSION:",
                "------------------------------",
                "Tests whether results change when excluding participants with",
                "unusually high or low numbers of observations (statistical outliers).",
                ""
            ])
            
            if isinstance(outlier_data, dict):
                n_outliers = outlier_data.get('n_outlier_participants', 0)
                n_remaining = outlier_data.get('n_participants_after_exclusion', 0)
                home_pref = outlier_data.get('home_row_proportion_without_outliers')
                
                summary_lines.extend([
                    f"Outlier exclusion results:",
                    f"  • Outlier participants identified: {n_outliers}",
                    f"  • Participants after exclusion: {n_remaining}",
                    f"  • Home row preference without outliers: {home_pref:.1%}" if home_pref else "  • Home row preference: N/A",
                    f"  • High threshold: {outlier_data.get('outlier_threshold_high', 'N/A'):.1f} observations",
                    f"  • Low threshold: {outlier_data.get('outlier_threshold_low', 'N/A'):.1f} observations",
                    ""
                ])
        
        # Statistical approaches
        if 'statistical_approaches' in sensitivity_data:
            stat_data = sensitivity_data['statistical_approaches']
            
            summary_lines.extend([
                "ALTERNATIVE STATISTICAL APPROACHES:",
                "----------------------------------",
                "Documents what other statistical methods could be used",
                "and their potential advantages.",
                ""
            ])
            
            if isinstance(stat_data, dict) and 'approaches_tested' in stat_data:
                for approach in stat_data['approaches_tested']:
                    summary_lines.append(f"  • {approach}")
                summary_lines.append("")
        
        summary_lines.extend([
            "INTERPRETATION:",
            "--------------",
            "• Consistency threshold analysis: Shows robustness across",
            "  different data quality requirements",
            "• Outlier exclusion: Tests whether extreme participants",
            "  drive the overall results",
            "• Alternative approaches: Documents methodological choices",
            "  and potential improvements",
            "",
            "ROBUSTNESS ASSESSMENT:",
            "• If results are similar across conditions → robust findings",
            "• If results change dramatically → findings may be fragile",
            "• Large changes suggest need for additional data or methods",
            "",
            "DATA SOURCE:",
            "• Complete data preserved in: sensitivity_analysis.json",
            "• Use JSON file for detailed robustness analysis or method comparison"
        ])
        
        # Save summary
        summary_path = os.path.join(output_folder, 'sensitivity_analysis_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Sensitivity analysis summary saved to {summary_path}")

    def generate_report(self, results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive report."""
        
        report_lines = [
            "KEYBOARD ERGONOMICS ANALYSIS REPORT",
            "=" * 60,
            "",
            "EXECUTIVE SUMMARY",
            "================",
            ""
        ]
        
        # Power analysis summary
        if 'power_analysis' in results:
            power_data = results['power_analysis']
            report_lines.extend([
                "STATISTICAL POWER & SAMPLE SIZE JUSTIFICATION",
                "--------------------------------------------",
                f"• Total participants: {power_data['total_participants']}",
                f"• Total observations: {power_data['total_observations']}",
                "",
                "Power to detect effects:",
            ])
            
            for effect_name, effect_data in power_data['power_by_effect_size'].items():
                status = "✓ Well-powered" if effect_data['adequately_powered'] else "⚠ Under-powered"
                report_lines.append(
                    f"• {effect_name.title()} effects ({effect_data['effect_size_description']}): "
                    f"Power = {effect_data['current_power']:.3f} {status}"
                )
            
            report_lines.extend(["", "Recommendations:"])
            for rec in power_data['recommendations']:
                report_lines.append(f"• {rec}")
            
            report_lines.append("")
        
        # Enhanced findings summary with ALL results
        if 'main_results' in results:
            report_lines.extend([
                "COMPLETE TEST RESULTS",
                "===================",
                ""
            ])
            
            # Extract ALL results (both significant and non-significant)
            all_results = []
            
            def extract_all_results(obj, prefix=""):
                if isinstance(obj, dict):
                    if 'test_name' in obj and 'p_value' in obj:
                        all_results.append({
                            'name': obj['test_name'],
                            'effect_size': obj.get('effect_size', 0),
                            'p_value': obj.get('p_value', 1),
                            'proportion': obj.get('proportion', 0.5),
                            'cohens_h': obj.get('cohens_h', 0),
                            'n_comparisons': obj.get('n_comparisons', 0),
                            'significant': obj.get('significant', False),
                            'ci_lower': obj.get('ci_lower', None),
                            'ci_upper': obj.get('ci_upper', None)
                        })
                    
                    for key, value in obj.items():
                        if isinstance(value, dict):
                            extract_all_results(value, f"{prefix}.{key}" if prefix else key)
            
            extract_all_results(results['main_results'])
            
            if all_results:
                # Sort by effect size (descending)
                all_results.sort(key=lambda x: x['effect_size'], reverse=True)
                
                # Separate significant and non-significant results
                significant_results = [r for r in all_results if r['significant']]
                non_significant_results = [r for r in all_results if not r['significant']]
                
                # Report significant effects first
                if significant_results:
                    report_lines.extend([
                        "STATISTICALLY SIGNIFICANT EFFECTS (p < 0.05):",
                        "----------------------------------------------"
                    ])
                    
                    for result in significant_results:
                        effect_desc = ("Large" if result['effect_size'] >= 0.25 else 
                                    "Medium" if result['effect_size'] >= 0.15 else "Small")
                        
                        ci_text = ""
                        if result['ci_lower'] is not None and result['ci_upper'] is not None:
                            ci_text = f", 95% CI: {result['ci_lower']:.1%}-{result['ci_upper']:.1%}"
                        
                        report_lines.append(
                            f"• {result['name']}: {result['proportion']:.1%} preference "
                            f"(n = {result['n_comparisons']} comparisons{ci_text})"
                        )
                        report_lines.append(
                            f"  Effect size: {result['effect_size']:.3f} ({effect_desc}), "
                            f"p = {result['p_value']:.3e}, Cohen's h = {result['cohens_h']:.3f}"
                        )
                        report_lines.append("")
                else:
                    report_lines.extend([
                        "STATISTICALLY SIGNIFICANT EFFECTS (p < 0.05):",
                        "----------------------------------------------",
                        "• No statistically significant effects found at α = 0.05",
                        ""
                    ])
                
                # Report all non-significant effects
                if non_significant_results:
                    report_lines.extend([
                        "NON-SIGNIFICANT EFFECTS (p ≥ 0.05):",
                        "-----------------------------------"
                    ])
                    
                    for result in non_significant_results:
                        effect_desc = ("Large" if result['effect_size'] >= 0.25 else 
                                    "Medium" if result['effect_size'] >= 0.15 else "Small")
                        
                        ci_text = ""
                        if result['ci_lower'] is not None and result['ci_upper'] is not None:
                            ci_text = f", 95% CI: {result['ci_lower']:.1%}-{result['ci_upper']:.1%}"
                        
                        report_lines.append(
                            f"• {result['name']}: {result['proportion']:.1%} preference "
                            f"(n = {result['n_comparisons']} comparisons{ci_text})"
                        )
                        report_lines.append(
                            f"  Effect size: {result['effect_size']:.3f} ({effect_desc}), "
                            f"p = {result['p_value']:.3f}, Cohen's h = {result['cohens_h']:.3f}"
                        )
                        report_lines.append("")
                
                # Summary statistics
                total_tests = len(all_results)
                significant_count = len(significant_results)
                
                report_lines.extend([
                    "TEST SUMMARY:",
                    "------------",
                    f"• Total tests conducted: {total_tests}",
                    f"• Statistically significant: {significant_count} ({significant_count/total_tests:.1%})",
                    f"• Non-significant: {total_tests - significant_count} ({(total_tests - significant_count)/total_tests:.1%})",
                    f"• Largest effect size: {max(r['effect_size'] for r in all_results):.3f}" if all_results else "• No effects calculated",
                    f"• Mean effect size: {np.mean([r['effect_size'] for r in all_results]):.3f}" if all_results else "• No effects calculated",
                    ""
                ])
            else:
                report_lines.extend([
                    "• No test results found for analysis",
                    ""
                ])
        
        # Individual differences summary
        if 'individual_differences' in results:
            ind_diff = results['individual_differences']
            report_lines.extend([
                "INDIVIDUAL DIFFERENCES",
                "=====================",
                ""
            ])
            
            if 'consistency_analysis' in ind_diff:
                consistency = ind_diff['consistency_analysis']
                
                for preference_type in ['home_row', 'same_row']:
                    if preference_type in consistency:
                        data = consistency[preference_type]
                        pref_name = "Home Row" if preference_type == 'home_row' else "Same Row"
                        
                        report_lines.extend([
                            f"{pref_name} Preference Consistency:",
                            f"• {data['n_participants']} participants analyzed",
                            f"• {data['proportion_showing_preference']:.1%} show preference in expected direction",
                            f"• Mean individual effect size: {data['mean_effect_size']:.3f} ± {data['std_effect_size']:.3f}",
                            ""
                        ])
        
        # Methodology summary
        report_lines.extend([
            "METHODS SUMMARY",
            "===============",
            "",
            "Statistical Approach:",
            "• Binomial tests for proportion comparisons",
            "• Wilson score confidence intervals",
            "• Multiple comparison correction using FDR (Benjamini-Hochberg)",
            "• Cohen's h for standardized effect sizes",
            "• Bootstrap confidence intervals for robustness",
            "",
            "Sensitivity Analyses:",
            "• Different consistency thresholds tested",
            "• Outlier participant exclusion",
            "• Alternative statistical approaches compared",
            "",
            "Standards:",
            "• Forest plots with confidence intervals",
            "• Individual differences analysis",
            "• Power analysis and sample size justification",
            "• Standardized effect size reporting",
            ""
        ])
        
        # Add additional data files section
        report_lines.extend([
            "",
            "ADDITIONAL DATA FILES:",
            "====================",
            "Detailed data preserved in machine-readable format with human-readable summaries:",
            "",
            "POWER ANALYSIS:",
            "• power_analysis.json - Complete power calculations for reanalysis",
            "• power_analysis_summary.txt - Human-readable power analysis overview",
            "",
            "INDIVIDUAL DIFFERENCES:",
            "• individual_differences.json - Participant-level preferences and consistency data",
            "• individual_differences_summary.txt - Overview of individual variation patterns",
            "",
            "SENSITIVITY ANALYSES:",
            "• sensitivity_analysis.json - Robustness checks and alternative analysis results",
            "• sensitivity_analysis_summary.txt - Summary of methodological robustness",
            "",
            "FIGURES:",
            "• forest_plot.png/pdf - Effect sizes with confidence intervals",
            "• individual_differences.png/pdf - Preference distribution across participants",
            "• effect_size_distribution.png/pdf - Histogram of all effect sizes",
            "• consistency_analysis.png/pdf - Individual effect size patterns",
            "",
            "TIP: Read the .txt summary files for quick overviews, use the json files for data analysis"
        ])
        
        # Save comprehensive report
        report_path = os.path.join(output_folder, 'comprehensive_ergonomics_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Comprehensive report saved to {report_path}")

# Integration functions for existing analyze_data.py

def add_ergonomics_analysis_to_main(analyzer_instance, data, output_folder, config):
    """
    Function to integrate keyboard ergonomics analysis into existing analyze_data.py main() function.
    
    Args:
        analyzer_instance: Instance of BigramAnalysis class
        data: Processed bigram data
        output_folder: Output directory 
        config: Configuration dictionary
    """
    
    # Check if ergonomics analysis is enabled (backward compatibility)
    run_ergonomics = (config.get('analysis', {}).get('run_ergonomics_tests', False) or 
                     config.get('analysis', {}).get('run_enhanced_ergonomics', False))
    
    if not run_ergonomics:
        return None
    
    logger.info("Running complete keyboard ergonomics analysis...")
    
    # Create ergonomics analyzer
    ergonomics_analyzer = KeyboardErgonomicsAnalysis(config)
    
    # Run COMPLETE analysis with all features
    ergonomics_results = ergonomics_analyzer.run_complete_analysis(data, output_folder)
    
    logger.info("Complete keyboard ergonomics analysis finished!")
    
    return ergonomics_results