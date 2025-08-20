"""
Bradley-Terry model that captures spatial keyboard structure 
while being more parsimonious than the full 210-parameter bigram model.

This spatial feature B-T model (15-50 parameters) models bigram preference 
using spatial keyboard features.

Pros:
  - Much more parsimonious (fewer parameters)
  - Captures row/column/finger sensitivity explicitly
  - Can score any bigram, even unobserved ones
  - More stable estimates with limited data

Cons:
  - Makes assumptions about which spatial features matter
  - May miss complex interaction effects between specific key combinations

"""

import pandas as pd
import numpy as np
from itertools import product
from scipy.optimize import minimize
from collections import defaultdict
from typing import Dict, List, Tuple
import scipy.stats as stats

class SpatialBTModel:
    """
    Bradley-Terry model using spatial features to capture row/column/finger effects
    while remaining more parsimonious than full bigram model.
    
    Uses same data filtering as original script (excludes sliderValue=0).
    """
    
    def __init__(self, data_path: str = None, include_middle_column: bool = True):
        """
        Initialize spatial B-T model.
        
        Args:
            data_path: Path to CSV data file
            include_middle_column: If True, includes GTB keys (15 total). If False, excludes them (12 total).
        """
        
        # Base keyboard layout (always include core 12 keys)
        self.keyboard_layout = {
            # (row, column, finger) for each key
            'q': (1, 1, 1), 'w': (1, 2, 2), 'e': (1, 3, 3), 'r': (1, 4, 4),
            'a': (2, 1, 1), 's': (2, 2, 2), 'd': (2, 3, 3), 'f': (2, 4, 4),
            'z': (3, 1, 1), 'x': (3, 2, 2), 'c': (3, 3, 3), 'v': (3, 4, 4)
        }
        
        # Optionally add middle column (GTB keys)
        if include_middle_column:
            self.keyboard_layout.update({
                't': (1, 5, 4),  # Upper middle
                'g': (2, 5, 4),  # Home middle  
                'b': (3, 5, 4)   # Lower middle
            })
        
        self.include_middle_column = include_middle_column
        
        # Load data if path provided (using same filtering as original script)
        self.data = self._load_and_validate_data(data_path) if data_path else None
        
        # Model parameters (to be fit)
        self.params = {}
        self.fitted = False
    
    def _load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate data using same approach as original script."""
        try:
            data = pd.read_csv(data_path)
            
            # Check required columns (same as original script)
            required_cols = ['user_id', 'chosen_bigram', 'unchosen_bigram', 'sliderValue']
            missing_cols = set(required_cols) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert sliderValue to numeric, handling any string values (same as original)
            data['sliderValue'] = pd.to_numeric(data['sliderValue'], errors='coerce')
            
            # Remove rows with invalid sliderValue
            invalid_slider_rows = data['sliderValue'].isna().sum()
            if invalid_slider_rows > 0:
                print(f"Removing {invalid_slider_rows} rows with invalid sliderValue")
                data = data.dropna(subset=['sliderValue'])
            
            # Filter to consistent choices (non-zero slider values) - SAME AS ORIGINAL
            initial_len = len(data)
            data = data[data['sliderValue'] != 0].copy()
            print(f"Using {len(data)}/{initial_len} consistent choice rows (excluding neutral sliderValue=0)")
            
            # Convert bigrams to lowercase and ensure they are strings (same as original)
            data['chosen_bigram'] = data['chosen_bigram'].astype(str).str.lower()
            data['unchosen_bigram'] = data['unchosen_bigram'].astype(str).str.lower()
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def extract_spatial_features(self, bigram: str) -> Dict:
        """Extract spatial features for a bigram."""
        if len(bigram) != 2:
            raise ValueError("Must be a 2-character bigram")
        
        key1, key2 = bigram[0].lower(), bigram[1].lower()
        
        if key1 not in self.keyboard_layout or key2 not in self.keyboard_layout:
            return None
        
        row1, col1, finger1 = self.keyboard_layout[key1]
        row2, col2, finger2 = self.keyboard_layout[key2]
        
        features = {
            # ABSOLUTE POSITIONS (pinpoint exact locations)
            'key1_row': row1,           # Absolute row: 1=upper, 2=home, 3=lower
            'key1_col': col1,           # Absolute column: 1-5 from left
            'key1_finger': finger1,     # Absolute finger: 1=pinky, 4=index
            'key2_row': row2,
            'key2_col': col2,
            'key2_finger': finger2,
            
            # Absolute row type indicators
            'key1_is_home_row': row1 == 2,
            'key1_is_upper_row': row1 == 1,
            'key1_is_lower_row': row1 == 3,
            'key2_is_home_row': row2 == 2,
            'key2_is_upper_row': row2 == 1,
            'key2_is_lower_row': row2 == 3,
            
            # RELATIVE MOVEMENTS (relationships between keys)
            'delta_row': row2 - row1,              # Signed row movement
            'delta_col': col2 - col1,              # Signed column movement  
            'delta_finger': finger2 - finger1,    # Signed finger movement
            'abs_delta_row': abs(row2 - row1),     # Absolute row distance
            'abs_delta_col': abs(col2 - col1),     # Absolute column distance
            'abs_delta_finger': abs(finger2 - finger1),  # Absolute finger distance
            
            # CATEGORICAL SPATIAL PATTERNS
            'same_row': row1 == row2,
            'same_col': col1 == col2,
            'same_finger': finger1 == finger2,
            'adjacent_rows': abs(row2 - row1) == 1,
            
            # IN/OUT ROLLS - SAME ROW (traditional rolls)
            'same_row_roll_inward': (row1 == row2) and (finger2 > finger1),   # e.g., a->s
            'same_row_roll_outward': (row1 == row2) and (finger2 < finger1),  # e.g., s->a
            
            # IN/OUT ROLLS - DIFFERENT ROW (cross-row rolls)
            'diff_row_roll_inward': (row1 != row2) and (finger2 > finger1),   # e.g., q->s, z->d  
            'diff_row_roll_outward': (row1 != row2) and (finger2 < finger1),  # e.g., s->q, d->z
            
            # COMBINED ROLL PATTERNS
            'any_roll_inward': finger2 > finger1,    # Any inward movement
            'any_roll_outward': finger2 < finger1,   # Any outward movement
            
            # ROW TRANSITION PATTERNS
            'row_pattern': self._get_row_pattern(row1, row2),
            'finger_span': self._get_finger_span(finger1, finger2),
            
            # ROW-SPECIFIC TRANSITIONS
            'home_to_upper': (row1 == 2) and (row2 == 1),
            'home_to_lower': (row1 == 2) and (row2 == 3),
            'upper_to_home': (row1 == 1) and (row2 == 2),
            'lower_to_home': (row1 == 3) and (row2 == 2),
            'upper_to_lower': (row1 == 1) and (row2 == 3),  # Skip home
            'lower_to_upper': (row1 == 3) and (row2 == 1),  # Skip home
        }
        
        return features
    
    def _get_row_pattern(self, row1: int, row2: int) -> str:
        """Classify the row transition pattern."""
        if row1 == row2:
            return 'same_row'
        elif abs(row1 - row2) == 1:
            return 'adjacent_rows'
        else:
            return 'skip_rows'
    
    def _get_finger_span(self, finger1: int, finger2: int) -> str:
        """Classify finger span."""
        span = abs(finger1 - finger2)
        if span == 0:
            return 'same_finger'
        elif span == 1:
            return 'adjacent_fingers'
        elif span == 2:
            return 'two_finger_span'
        else:
            return 'wide_span'
    
    def define_parameter_space(self, feature_set: str = 'comprehensive') -> List[str]:
        """
        Define which spatial parameters to include in the model.
        
        Args:
            feature_set: 'minimal', 'standard', 'comprehensive', or 'custom'
        """
        
        if feature_set == 'minimal':
            # Just basic position effects (~12 parameters)
            return [
                'intercept',
                'key1_is_home_row', 'key1_is_upper_row',  # Absolute row positions
                'key2_is_home_row', 'key2_is_upper_row',
                'key1_finger_2', 'key1_finger_3', 'key1_finger_4',  # Finger effects
                'key2_finger_2', 'key2_finger_3', 'key2_finger_4',
                'same_row', 'abs_delta_finger_1'
            ]
            
        elif feature_set == 'standard':
            # Add transition patterns (~25 parameters)
            params = [
                'intercept',
                # Absolute positions (pinpoint locations)
                'key1_is_home_row', 'key1_is_upper_row', 'key1_is_lower_row',
                'key2_is_home_row', 'key2_is_upper_row', 'key2_is_lower_row',
                'key1_finger_2', 'key1_finger_3', 'key1_finger_4',
                'key2_finger_2', 'key2_finger_3', 'key2_finger_4',
                
                # Spatial relationships
                'same_row', 'same_finger', 'adjacent_rows',
                'same_row_roll_inward', 'same_row_roll_outward',
                'diff_row_roll_inward', 'diff_row_roll_outward',
                'abs_delta_finger_1', 'abs_delta_finger_2', 'abs_delta_finger_3',
                
                # Specific row transitions
                'home_to_upper', 'home_to_lower', 'upper_to_home', 'lower_to_home'
            ]
            return params
            
        else:  # comprehensive
            # Include all features (~40+ parameters)
            params = [
                'intercept',
                # All absolute position effects
                'key1_row_2', 'key1_row_3', 'key2_row_2', 'key2_row_3',
                'key1_is_home_row', 'key1_is_upper_row', 'key1_is_lower_row',
                'key2_is_home_row', 'key2_is_upper_row', 'key2_is_lower_row',
                'key1_finger_2', 'key1_finger_3', 'key1_finger_4',
                'key2_finger_2', 'key2_finger_3', 'key2_finger_4',
                
                # Column effects (if using middle column)
                'key1_col_2', 'key1_col_3', 'key1_col_4', 
                'key2_col_2', 'key2_col_3', 'key2_col_4',
            ]
            
            # Add middle column effects if included
            if self.include_middle_column:
                params.extend(['key1_col_5', 'key2_col_5'])
            
            params.extend([
                # All spatial relationships
                'same_row', 'same_col', 'same_finger', 'adjacent_rows',
                'same_row_roll_inward', 'same_row_roll_outward',
                'diff_row_roll_inward', 'diff_row_roll_outward',
                'any_roll_inward', 'any_roll_outward',
                'abs_delta_finger_1', 'abs_delta_finger_2', 'abs_delta_finger_3',
                
                # All row transition patterns
                'home_to_upper', 'home_to_lower', 'upper_to_home', 'lower_to_home',
                'upper_to_lower', 'lower_to_upper',
                
                # Higher-order terms
                'delta_row', 'delta_col', 'delta_finger',
                'abs_delta_row', 'abs_delta_col'
            ])
            
            return params
    
    def fit_with_feature_pruning(self, bigram_comparisons: Dict = None, 
                                initial_feature_set: str = 'comprehensive',
                                significance_threshold: float = 0.05,
                                min_effect_size: float = 0.1) -> Dict:
        """
        Fit spatial model with automatic feature pruning for better CIs.
        
        Args:
            bigram_comparisons: Comparison data
            initial_feature_set: Starting feature set
            significance_threshold: P-value threshold for keeping features
            min_effect_size: Minimum coefficient magnitude to keep feature
            
        Returns:
            Results including pruning steps and final model
        """
        print(f"Starting feature pruning from '{initial_feature_set}' feature set...")
        
        # Step 1: Fit full model
        full_result = self.fit_spatial_model(bigram_comparisons, initial_feature_set)
        if not full_result['success']:
            return full_result
        
        print(f"Full model: {full_result['n_parameters']} parameters, LL = {full_result['log_likelihood']:.2f}")
        
        # Step 2: Calculate feature significance (approximate using Wald tests)
        feature_stats = self._calculate_feature_significance(full_result, bigram_comparisons)
        
        # Step 3: Prune features iteratively
        pruning_steps = []
        current_params = full_result['param_names'].copy()
        
        while True:
            # Find least important feature to remove
            removable_features = []
            for param in current_params:
                if param != 'intercept':  # Never remove intercept
                    stats = feature_stats.get(param, {})
                    p_value = stats.get('p_value', 0.0)
                    effect_size = abs(full_result['params'].get(param, 0.0))
                    
                    if p_value > significance_threshold or effect_size < min_effect_size:
                        removable_features.append((param, p_value, effect_size))
            
            if not removable_features:
                break  # No more features to remove
            
            # Remove the least significant feature
            removable_features.sort(key=lambda x: (-x[1], x[2]))  # Highest p-value, lowest effect
            to_remove = removable_features[0][0]
            current_params.remove(to_remove)
            
            # Refit with reduced feature set
            reduced_result = self._fit_with_custom_features(bigram_comparisons, current_params)
            
            if reduced_result['success']:
                pruning_steps.append({
                    'removed_feature': to_remove,
                    'n_params': len(current_params),
                    'log_likelihood': reduced_result['log_likelihood'],
                    'removed_p_value': removable_features[0][1],
                    'removed_effect_size': removable_features[0][2]
                })
                print(f"Removed '{to_remove}' (p={removable_features[0][1]:.3f}), {len(current_params)} params remain")
            else:
                # If reduced model fails, stop pruning
                current_params.append(to_remove)  # Add it back
                break
        
        # Step 4: Final model with pruned features
        final_result = self._fit_with_custom_features(bigram_comparisons, current_params)
        
        if final_result['success']:
            self.params = final_result['params']
            self.fitted = True
            
            print(f"Final pruned model: {len(current_params)} parameters, LL = {final_result['log_likelihood']:.2f}")
            print(f"Removed {len(full_result['param_names']) - len(current_params)} features")
            
            return {
                'success': True,
                'full_model': full_result,
                'final_model': final_result,
                'pruning_steps': pruning_steps,
                'features_removed': len(full_result['param_names']) - len(current_params),
                'final_params': final_result['params'],
                'final_param_names': current_params
            }
        else:
            return final_result
    
    def _calculate_feature_significance(self, model_result: Dict, bigram_comparisons: Dict) -> Dict:
        """Calculate approximate p-values for features using Wald tests."""
        # This is a simplified implementation - in practice you'd want to compute
        # the Hessian matrix for proper standard errors
        feature_stats = {}
        
        for param_name, coef in model_result['params'].items():
            if param_name != 'intercept':
                # Rough approximation: larger coefficients in larger models are less certain
                n_params = model_result['n_parameters']
                n_obs = model_result['n_observations']
                
                # Simple heuristic for standard error (this is approximate!)
                rough_se = np.sqrt(n_params / n_obs) * 0.5
                z_score = abs(coef) / rough_se if rough_se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed test
                
                feature_stats[param_name] = {
                    'coefficient': coef,
                    'se_approx': rough_se,
                    'z_score': z_score,
                    'p_value': p_value
                }
        
        return feature_stats
    
    def _fit_with_custom_features(self, bigram_comparisons: Dict, param_names: List[str]) -> Dict:
            """Fit model with custom parameter list."""
            # Extract comparisons from data if not provided
            if bigram_comparisons is None:
                if self.data is None:
                    return {'success': False, 'error': 'No comparison data provided and no data loaded'}
                bigram_comparisons = self._extract_bigram_comparisons_from_data()
            
            print(f"Fitting custom model with {len(param_names)} parameters")
            
            # Prepare training data with filtering (same as main method)
            bigram_pairs = []
            features_list = []
            wins_list = []
            totals_list = []
            
            filtered_count = 0
            extreme_count = 0
            
            for (bigram1, bigram2), comp in bigram_comparisons.items():
                if comp['total'] >= 5:  # Minimum data threshold
                    
                    # Filter extreme cases that cause numerical instability
                    win_rate = comp['wins_bigram1'] / comp['total']
                    if 0.05 <= win_rate <= 0.95:  # Remove very lopsided comparisons
                        
                        features1 = self.extract_spatial_features(bigram1)
                        features2 = self.extract_spatial_features(bigram2)
                        
                        if features1 is not None and features2 is not None:
                            bigram_pairs.append((bigram1, bigram2))
                            features_list.append((features1, features2))
                            wins_list.append(comp['wins_bigram1'])
                            totals_list.append(comp['total'])
                            filtered_count += 1
                    else:
                        extreme_count += 1
            
            if len(bigram_pairs) < len(param_names):
                return {
                    'success': False,
                    'error': f'Insufficient data after filtering: {len(bigram_pairs)} comparisons for {len(param_names)} parameters'
                }
            
            # Set up optimization with regularization (same as main method)
            def negative_log_likelihood(params):
                ll = 0
                regularization = 0.01  # L2 regularization
                
                for (features1, features2), wins, total in zip(features_list, wins_list, totals_list):
                    eta1 = sum(params[j] * self._get_feature_value(features1, param_names[j]) 
                            for j in range(len(param_names)))
                    eta2 = sum(params[j] * self._get_feature_value(features2, param_names[j]) 
                            for j in range(len(param_names)))
                    
                    # Clip to prevent overflow
                    eta1 = np.clip(eta1, -10, 10)
                    eta2 = np.clip(eta2, -10, 10)
                    
                    p12 = np.exp(eta1) / (np.exp(eta1) + np.exp(eta2))
                    p12 = np.clip(p12, 1e-10, 1 - 1e-10)
                    
                    ll += wins * np.log(p12) + (total - wins) * np.log(1 - p12)
                
                # Add L2 regularization (except for intercept)
                reg_penalty = regularization * sum(params[1:]**2) if len(params) > 1 else 0
                
                return -ll + reg_penalty
            
            # Fit model using robust optimizer
            initial_params = np.zeros(len(param_names))
            
            try:
                # Try Powell first
                result = minimize(negative_log_likelihood, initial_params, method='Powell',
                                options={'maxiter': 2000})
                
                if not result.success:
                    # Fallback to Nelder-Mead
                    result = minimize(negative_log_likelihood, initial_params, method='Nelder-Mead',
                                    options={'maxiter': 2000})
                
                if result.success:
                    return {
                        'success': True,
                        'params': dict(zip(param_names, result.x)),
                        'n_parameters': len(param_names),
                        'n_observations': len(bigram_pairs),
                        'log_likelihood': -result.fun,
                        'param_names': param_names,
                        'optimizer_used': 'Powell',
                        'filtered_extreme_cases': extreme_count
                    }
                else:
                    return {'success': False, 'error': f'Optimization failed: {result.message}'}
                    
            except Exception as e:
                return {'success': False, 'error': str(e)}
                
    def create_design_matrix(self, bigram_features: List[Dict], param_names: List[str]) -> np.ndarray:
        """Create design matrix from bigram features."""
        n_obs = len(bigram_features)
        n_params = len(param_names)
        X = np.zeros((n_obs, n_params))
        
        for i, features in enumerate(bigram_features):
            if features is None:
                continue
                
            for j, param_name in enumerate(param_names):
                if param_name == 'intercept':
                    X[i, j] = 1
                elif param_name in features:
                    # Handle boolean features
                    value = features[param_name]
                    X[i, j] = 1 if value is True else (0 if value is False else value)
                elif param_name.endswith('_squared'):
                    base_param = param_name.replace('_squared', '')
                    if base_param in features:
                        X[i, j] = features[base_param] ** 2
        
        return X
    
    def fit_spatial_model(self, bigram_comparisons: Dict = None, feature_set: str = 'standard') -> Dict:
            """
            Fit spatial Bradley-Terry model to bigram comparison data.
            
            Args:
                bigram_comparisons: Optional dict from (bigram1, bigram2) -> {'wins_bigram1': int, 'total': int}
                                If None, will extract from self.data
                feature_set: Which features to include
                
            Returns:
                Model results dictionary
            """
            
            # Extract comparisons from data if not provided
            if bigram_comparisons is None:
                if self.data is None:
                    return {'success': False, 'error': 'No comparison data provided and no data loaded'}
                bigram_comparisons = self._extract_bigram_comparisons_from_data()
            
            param_names = self.define_parameter_space(feature_set)
            print(f"Fitting spatial B-T model with {len(param_names)} parameters: {feature_set} feature set")
            
            # Prepare training data with filtering for numerical stability
            bigram_pairs = []
            features_list = []
            wins_list = []
            totals_list = []
            
            filtered_count = 0
            extreme_count = 0
            
            for (bigram1, bigram2), comp in bigram_comparisons.items():
                if comp['total'] >= 5:  # Minimum data threshold
                    
                    # Filter extreme cases that cause numerical instability
                    win_rate = comp['wins_bigram1'] / comp['total']
                    if 0.05 <= win_rate <= 0.95:  # Remove very lopsided comparisons
                        
                        features1 = self.extract_spatial_features(bigram1)
                        features2 = self.extract_spatial_features(bigram2)
                        
                        if features1 is not None and features2 is not None:
                            bigram_pairs.append((bigram1, bigram2))
                            features_list.append((features1, features2))
                            wins_list.append(comp['wins_bigram1'])
                            totals_list.append(comp['total'])
                            filtered_count += 1
                    else:
                        extreme_count += 1
            
            print(f"Using {filtered_count} pairs after filtering {extreme_count} extreme cases")
            
            if len(bigram_pairs) < len(param_names):
                return {
                    'success': False,
                    'error': f'Insufficient data after filtering: {len(bigram_pairs)} comparisons for {len(param_names)} parameters'
                }
            
            # Set up optimization with regularization and numerical stability
            def negative_log_likelihood(params):
                ll = 0
                regularization = 0.01  # L2 regularization to prevent extreme coefficients
                
                for (features1, features2), wins, total in zip(features_list, wins_list, totals_list):
                    # Calculate linear predictors
                    eta1 = sum(params[j] * self._get_feature_value(features1, param_names[j]) 
                            for j in range(len(param_names)))
                    eta2 = sum(params[j] * self._get_feature_value(features2, param_names[j]) 
                            for j in range(len(param_names)))
                    
                    # Clip to prevent overflow
                    eta1 = np.clip(eta1, -10, 10)
                    eta2 = np.clip(eta2, -10, 10)
                    
                    # Probability that bigram1 beats bigram2
                    p12 = np.exp(eta1) / (np.exp(eta1) + np.exp(eta2))
                    p12 = np.clip(p12, 1e-10, 1 - 1e-10)
                    
                    ll += wins * np.log(p12) + (total - wins) * np.log(1 - p12)
                
                # Add L2 regularization (except for intercept)
                reg_penalty = regularization * sum(params[1:]**2) if len(params) > 1 else 0
                
                return -ll + reg_penalty
            
            # Fit model using robust optimizer (Powell works with your data)
            initial_params = np.zeros(len(param_names))
            
            try:
                # Try Powell first (most robust for this type of problem)
                result = minimize(negative_log_likelihood, initial_params, method='Powell',
                                options={'maxiter': 2000})
                
                if not result.success:
                    # Fallback to Nelder-Mead if Powell fails
                    print("Powell failed, trying Nelder-Mead...")
                    result = minimize(negative_log_likelihood, initial_params, method='Nelder-Mead',
                                    options={'maxiter': 2000})
                
                if result.success:
                    self.params = dict(zip(param_names, result.x))
                    self.fitted = True
                    
                    return {
                        'success': True,
                        'params': self.params,
                        'n_parameters': len(param_names),
                        'n_observations': len(bigram_pairs),
                        'log_likelihood': -result.fun,
                        'feature_set': feature_set,
                        'param_names': param_names,
                        'optimizer_used': 'Powell',
                        'filtered_extreme_cases': extreme_count
                    }
                else:
                    return {'success': False, 'error': f'Both optimizers failed: {result.message}'}
                    
            except Exception as e:
                return {'success': False, 'error': str(e)}
            
    def _extract_bigram_comparisons_from_data(self) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Extract bigram comparisons from loaded data (same logic as DirectBigramBTAnalyzer)."""
        comparisons = defaultdict(lambda: {'wins_bigram1': 0, 'total': 0})
        
        left_keys = ['q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
        valid_bigrams = set()
        for i, key1 in enumerate(left_keys):
            for j, key2 in enumerate(left_keys):
                if i != j:  # No same-key bigrams
                    valid_bigrams.add(key1 + key2)
        
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
        
        return dict(comparisons)
    
    def _get_feature_value(self, features: Dict, param_name: str) -> float:
        """Extract feature value for parameter."""
        if param_name == 'intercept':
            return 1.0
        elif param_name in features:
            value = features[param_name]
            return 1.0 if value is True else (0.0 if value is False else float(value))
        elif param_name.endswith('_squared'):
            base_param = param_name.replace('_squared', '')
            if base_param in features:
                return float(features[base_param]) ** 2
        else:
            # Handle categorical encodings (e.g., 'key1_row_2' means key1 is in row 2)
            parts = param_name.split('_')
            if len(parts) >= 3:
                feature_base = '_'.join(parts[:-1])
                target_value = int(parts[-1])
                if feature_base in features:
                    return 1.0 if features[feature_base] == target_value else 0.0
        
        return 0.0
    
    def predict_bigram_score(self, bigram: str) -> float:
        """Predict preference score for a bigram using fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        features = self.extract_spatial_features(bigram)
        if features is None:
            return np.nan
        
        score = sum(coef * self._get_feature_value(features, param_name) 
                   for param_name, coef in self.params.items())
        
        return score
    
    def score_all_bigrams(self) -> pd.DataFrame:
            """Score all left-hand bigrams using fitted spatial model."""
            if not self.fitted:
                raise ValueError("Model must be fitted first")
            
            # Get left keys based on whether middle column is included
            if self.include_middle_column:
                left_keys = ['q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
            else:
                left_keys = ['q','w','e','r','a','s','d','f','z','x','c','v']
            
            results = []
            
            for key1 in left_keys:
                for key2 in left_keys:
                    if key1 != key2:
                        bigram = key1 + key2
                        
                        # Extract features and check for None
                        features = self.extract_spatial_features(bigram)
                        if features is not None:
                            # Calculate score
                            score = self.predict_bigram_score(bigram)
                            
                            results.append({
                                'bigram': bigram,
                                'score': score,
                                'key1_row': features['key1_row'],
                                'key1_finger': features['key1_finger'],
                                'key2_row': features['key2_row'], 
                                'key2_finger': features['key2_finger'],
                                'delta_finger': features['delta_finger'],
                                'row_pattern': features['row_pattern'],
                                'same_row': features['same_row']
                            })
                        else:
                            # Handle cases where features couldn't be extracted
                            print(f"Warning: Could not extract features for bigram '{bigram}'")
            
            if not results:
                raise ValueError("No valid bigrams could be scored")
            
            return pd.DataFrame(results).sort_values('score', ascending=False)

# Usage example:
"""
# 1. BASIC USAGE: 12 keys without middle column
spatial_model_12 = SpatialBTModel('data/filtered_data.csv', include_middle_column=False)
result = spatial_model_12.fit_spatial_model(feature_set='standard')

# 2. FULL USAGE: 15 keys with middle column  
spatial_model_15 = SpatialBTModel('data/filtered_data.csv', include_middle_column=True)
result = spatial_model_15.fit_spatial_model(feature_set='comprehensive')

# 3. FEATURE PRUNING for better CIs
pruned_result = spatial_model_15.fit_with_feature_pruning(
    initial_feature_set='comprehensive',
    significance_threshold=0.05,
    min_effect_size=0.1
)

if pruned_result['success']:
    print(f"Pruning removed {pruned_result['features_removed']} features")
    print(f"Final model: {len(pruned_result['final_param_names'])} parameters")
    
    # Show important features that survived pruning
    important_features = {k: v for k, v in pruned_result['final_params'].items() 
                         if abs(v) > 0.2}
    print("Important features:")
    for feature, coef in sorted(important_features.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feature}: {coef:.3f}")
    
    # Score all bigrams with pruned model
    all_scores = spatial_model_15.score_all_bigrams()
    print(f"\\nTop 10 bigrams by pruned spatial model:")
    print(all_scores[['bigram', 'score', 'same_row', 'delta_finger', 'key1_row', 'key2_row']].head(10))

# 4. EXAMINE ROLL PATTERNS
test_bigrams = ['as', 'sa', 'aw', 'wa', 'qd', 'dq']  # Mix of same-row and cross-row rolls
for bigram in test_bigrams:
    features = spatial_model_15.extract_spatial_features(bigram)
    print(f"\\n{bigram}: same_row_roll_inward={features['same_row_roll_inward']}, "
          f"diff_row_roll_inward={features['diff_row_roll_inward']}")
"""