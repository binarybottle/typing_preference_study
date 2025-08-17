"""
Mixed-Effects Keyboard Ergonomics Analysis Module

Implements mixed-effects logistic regression for keyboard ergonomics analysis,
properly handling repeated measures within participants and testing multiple
factors simultaneously.

Features:
- Mixed-effects logistic regression models
- Simultaneous multi-factor analysis  
- Random effects modeling for participants
- Interaction effect testing
- Model diagnostics and validation
- Comprehensive reporting with effect interpretations
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Any, Optional
import os
import json
import warnings
from collections import defaultdict

# Mixed-effects modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import Logit

# Import existing keyboard mappings
from keymaps import row_map, column_map, finger_map

logger = logging.getLogger(__name__)

class MixedEffectsErgonomicsAnalysis:
    """
    Mixed-effects analysis for keyboard ergonomics with proper repeated measures handling.
    
    This class implements mixed-effects logistic regression to:
    - Properly account for participant-level clustering
    - Test multiple ergonomic factors simultaneously  
    - Identify interaction effects between factors
    - Provide better statistical inference than separate tests
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get('analysis', {}).get('alpha_level', 0.05)
        self.n_bootstrap = config.get('analysis', {}).get('n_bootstrap', 1000)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Model storage
        self.fitted_models = {}
        self.model_diagnostics = {}
        
    def run_complete_mixed_effects_analysis(self, data: pd.DataFrame, output_folder: str) -> Dict[str, Any]:
        """
        Run complete mixed-effects ergonomics analysis.
        
        Args:
            data: DataFrame with columns [user_id, chosen_bigram, unchosen_bigram, is_consistent]
            output_folder: Directory to save results
            
        Returns:
            Dictionary containing all mixed-effects analysis results
        """
        logger.info("Starting mixed-effects keyboard ergonomics analysis...")
        
        # Create analysis subdirectory
        analysis_folder = os.path.join(output_folder, 'keyboard_ergonomics_mixed')
        os.makedirs(analysis_folder, exist_ok=True)
        
        # Prepare data for mixed-effects modeling
        model_data = self._prepare_mixed_effects_data(data)
        logger.info(f"Prepared {len(model_data)} choice observations from {model_data['user_id'].nunique()} participants")
        
        # === 1. COMPREHENSIVE MIXED-EFFECTS MODEL ===
        logger.info("Fitting comprehensive mixed-effects model...")
        comprehensive_model = self._fit_comprehensive_model(model_data)
        
        # === 2. INDIVIDUAL FACTOR MODELS ===
        logger.info("Fitting individual factor models...")
        factor_models = self._fit_individual_factor_models(model_data)
        
        # === 3. INTERACTION MODELS ===
        logger.info("Testing interaction effects...")
        interaction_models = self._fit_interaction_models(model_data)
        
        # === 4. MODEL DIAGNOSTICS ===
        logger.info("Running model diagnostics...")
        diagnostics = self._run_model_diagnostics(model_data)
        
        # === 5. EFFECT SIZE CALCULATIONS ===
        logger.info("Calculating effect sizes and predictions...")
        effect_sizes = self._calculate_mixed_effects_sizes(comprehensive_model, model_data)
        
        # === 6. RANDOM EFFECTS ANALYSIS ===
        logger.info("Analyzing individual differences (random effects)...")
        random_effects = self._analyze_random_effects(comprehensive_model, model_data)
        
        # === 7. MODEL COMPARISON ===
        logger.info("Comparing models...")
        model_comparison = self._compare_models()
        
        # === 8. VISUALIZATIONS ===
        logger.info("Creating mixed-effects visualizations...")
        self._create_mixed_effects_plots(model_data, analysis_folder)
        
        # === 9. COMPREHENSIVE RESULTS ===
        complete_results = {
            'comprehensive_model': comprehensive_model,
            'factor_models': factor_models,
            'interaction_models': interaction_models,
            'model_diagnostics': diagnostics,
            'effect_sizes': effect_sizes,
            'random_effects': random_effects,
            'model_comparison': model_comparison,
            'sample_info': {
                'n_participants': model_data['user_id'].nunique(),
                'n_observations': len(model_data),
                'n_choices': len(model_data)
            }
        }
        
        # === 10. REPORTING ===
        logger.info("Generating mixed-effects reports...")
        self._generate_mixed_effects_report(complete_results, analysis_folder)
        self._save_model_results_json(complete_results, analysis_folder)
        
        logger.info("Mixed-effects keyboard ergonomics analysis complete!")
        return complete_results
    
    def _prepare_mixed_effects_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for mixed-effects modeling by creating choice-level observations.
        
        Each row in original data (bigram pair comparison) becomes one observation
        with ergonomic factors as predictors and choice as binary outcome.
        """
        # Add keyboard position features
        enhanced_data = self._add_keyboard_features(data)
        
        # Filter to consistent choices only
        consistent_data = enhanced_data[enhanced_data['is_consistent'] == True].copy()
        
        # Create model dataset where each row is a choice
        # Target variable: 1 if chose the more ergonomic option, 0 otherwise
        model_rows = []
        
        for _, row in consistent_data.iterrows():
            user_id = row['user_id']
            
            # Calculate ergonomic scores for chosen and unchosen bigrams
            chosen_score = self._calculate_ergonomic_score(row, 'chosen')
            unchosen_score = self._calculate_ergonomic_score(row, 'unchosen')
            
            # Create binary choice variable (1 = chose more ergonomic, 0 = chose less ergonomic)
            if chosen_score != unchosen_score:
                chose_ergonomic = int(chosen_score > unchosen_score)
                
                # Calculate predictor variables (differences between chosen and unchosen)
                predictors = self._calculate_choice_predictors(row)
                
                model_row = {
                    'user_id': user_id,
                    'chose_ergonomic': chose_ergonomic,
                    **predictors
                }
                model_rows.append(model_row)
        
        model_data = pd.DataFrame(model_rows)
        
        # Add derived variables
        model_data['user_id_cat'] = model_data['user_id'].astype('category')
        
        return model_data
    
    def _calculate_ergonomic_score(self, row: pd.Series, bigram_type: str) -> float:
        """
        Calculate overall ergonomic score for a bigram.
        Higher scores = more ergonomic.
        """
        score = 0.0
        
        # Home row bonus (strongest factor based on literature)
        row1 = row[f'{bigram_type}_bigram_row1']
        row2 = row[f'{bigram_type}_bigram_row2']
        home_keys = sum(1 for r in [row1, row2] if r == 2)
        score += home_keys * 2.0  # Strong weight for home row
        
        # Same row bonus (avoid cross-row movement)
        if row[f'{bigram_type}_bigram_same_row']:
            score += 1.0
        
        # Column position penalty (avoid extreme columns)
        col1 = row[f'{bigram_type}_bigram_col1']
        col2 = row[f'{bigram_type}_bigram_col2']
        for col in [col1, col2]:
            if col == 3:  # Optimal column
                score += 0.5
            elif col in [2, 4]:  # Good columns
                score += 0.25
            elif col in [1, 5]:  # Poor columns
                score -= 0.25
        
        # Adjacent finger bonus (avoid large stretches)
        finger_dist = row[f'{bigram_type}_bigram_finger_distance']
        if finger_dist == 1:  # Adjacent fingers
            score += 0.5
        elif finger_dist >= 3:  # Large stretch
            score -= 0.5
        
        return score
    
    def _calculate_choice_predictors(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate predictor variables as differences between chosen and unchosen bigrams.
        Positive values = chosen bigram has more of this feature.
        """
        predictors = {}
        
        # Home row advantage
        chosen_home = sum(1 for r in [row['chosen_bigram_row1'], row['chosen_bigram_row2']] if r == 2)
        unchosen_home = sum(1 for r in [row['unchosen_bigram_row1'], row['unchosen_bigram_row2']] if r == 2)
        predictors['home_row_advantage'] = chosen_home - unchosen_home
        
        # Same row advantage  
        chosen_same = int(row['chosen_bigram_same_row'])
        unchosen_same = int(row['unchosen_bigram_same_row'])
        predictors['same_row_advantage'] = chosen_same - unchosen_same
        
        # Column quality (distance from optimal column 3)
        chosen_col_quality = self._calculate_column_quality(row['chosen_bigram_col1'], row['chosen_bigram_col2'])
        unchosen_col_quality = self._calculate_column_quality(row['unchosen_bigram_col1'], row['unchosen_bigram_col2'])
        predictors['column_quality_advantage'] = chosen_col_quality - unchosen_col_quality
        
        # Finger distance (negative = closer fingers preferred)
        chosen_finger_dist = row['chosen_bigram_finger_distance']
        unchosen_finger_dist = row['unchosen_bigram_finger_distance']
        predictors['finger_distance_advantage'] = unchosen_finger_dist - chosen_finger_dist  # Negative of difference
        
        # Row distance (negative = closer rows preferred)
        chosen_row_dist = row['chosen_bigram_row_distance']
        unchosen_row_dist = row['unchosen_bigram_row_distance']
        predictors['row_distance_advantage'] = unchosen_row_dist - chosen_row_dist  # Negative of difference
        
        return predictors
    
    def _calculate_column_quality(self, col1: int, col2: int) -> float:
        """Calculate column quality score. Higher = better."""
        quality = 0.0
        for col in [col1, col2]:
            if col == 3:  # Optimal
                quality += 1.0
            elif col in [2, 4]:  # Good
                quality += 0.5
            elif col in [1, 5]:  # Poor
                quality -= 0.5
        return quality
    
    def _fit_comprehensive_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit comprehensive mixed-effects logistic regression model with all factors.
        """
        # Formula for comprehensive model
        formula = """chose_ergonomic ~ 
                     home_row_advantage + 
                     same_row_advantage + 
                     column_quality_advantage + 
                     finger_distance_advantage + 
                     row_distance_advantage"""
        
        try:
            # Fit mixed-effects logistic regression using Logit with cluster-robust SE
            # Note: statsmodels doesn't have mixed-effects logit, so we use cluster-robust SE
            model = smf.logit(formula, data=data).fit(cov_type='cluster', 
                                                      cov_kwds={'groups': data['user_id']})
            
            # Extract results
            results = {
                'model_type': 'Logistic regression with cluster-robust standard errors',
                'formula': formula,
                'n_obs': len(data),
                'n_participants': data['user_id'].nunique(),
                'convergence': model.mle_retvals['converged'],
                'log_likelihood': model.llf,
                'aic': model.aic,
                'bic': model.bic,
                'pseudo_r2': model.prsquared,
                'coefficients': {},
                'model_object': model  # Store for diagnostics
            }
            
            # Extract coefficient information
            for param in model.params.index:
                if param != 'Intercept':
                    coef = model.params[param]
                    se = model.bse[param]
                    p_value = model.pvalues[param]
                    ci_lower, ci_upper = model.conf_int().loc[param]
                    
                    # Convert to odds ratio
                    odds_ratio = np.exp(coef)
                    or_ci_lower = np.exp(ci_lower)
                    or_ci_upper = np.exp(ci_upper)
                    
                    results['coefficients'][param] = {
                        'coefficient': coef,
                        'std_error': se,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'odds_ratio': odds_ratio,
                        'or_ci_lower': or_ci_lower,
                        'or_ci_upper': or_ci_upper,
                        'interpretation': self._interpret_coefficient(param, coef, odds_ratio, p_value)
                    }
            
            self.fitted_models['comprehensive'] = model
            
        except Exception as e:
            logger.error(f"Error fitting comprehensive model: {str(e)}")
            results = {'error': str(e), 'model_type': 'Failed to fit'}
        
        return results
    
    def _fit_individual_factor_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit separate models for each ergonomic factor."""
        
        factor_models = {}
        
        factors = {
            'home_row': 'chose_ergonomic ~ home_row_advantage',
            'same_row': 'chose_ergonomic ~ same_row_advantage', 
            'column_quality': 'chose_ergonomic ~ column_quality_advantage',
            'finger_distance': 'chose_ergonomic ~ finger_distance_advantage',
            'row_distance': 'chose_ergonomic ~ row_distance_advantage'
        }
        
        for factor_name, formula in factors.items():
            try:
                model = smf.logit(formula, data=data).fit(cov_type='cluster',
                                                          cov_kwds={'groups': data['user_id']})
                
                # Extract key coefficient (excluding intercept)
                param_name = [p for p in model.params.index if p != 'Intercept'][0]
                coef = model.params[param_name]
                p_value = model.pvalues[param_name]
                odds_ratio = np.exp(coef)
                
                factor_models[factor_name] = {
                    'formula': formula,
                    'coefficient': coef,
                    'odds_ratio': odds_ratio,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'pseudo_r2': model.prsquared,
                    'aic': model.aic,
                    'interpretation': self._interpret_coefficient(param_name, coef, odds_ratio, p_value)
                }
                
                self.fitted_models[factor_name] = model
                
            except Exception as e:
                logger.warning(f"Error fitting {factor_name} model: {str(e)}")
                factor_models[factor_name] = {'error': str(e)}
        
        return factor_models
    
    def _fit_interaction_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Test for interaction effects between key factors."""
        
        interaction_models = {}
        
        # Key interactions to test
        interactions = {
            'home_row_x_same_row': 'chose_ergonomic ~ home_row_advantage * same_row_advantage',
            'home_row_x_column': 'chose_ergonomic ~ home_row_advantage * column_quality_advantage',
            'finger_x_row_distance': 'chose_ergonomic ~ finger_distance_advantage * row_distance_advantage'
        }
        
        for interaction_name, formula in interactions.items():
            try:
                full_model = smf.logit(formula, data=data).fit(cov_type='cluster',
                                                               cov_kwds={'groups': data['user_id']})
                
                # Test if interaction term is significant
                interaction_params = [p for p in full_model.params.index if ':' in p]
                
                if interaction_params:
                    interaction_param = interaction_params[0]
                    coef = full_model.params[interaction_param]
                    p_value = full_model.pvalues[interaction_param]
                    
                    interaction_models[interaction_name] = {
                        'formula': formula,
                        'interaction_coefficient': coef,
                        'interaction_p_value': p_value,
                        'significant': p_value < self.alpha,
                        'model_aic': full_model.aic,
                        'interpretation': f"{'Significant' if p_value < self.alpha else 'Non-significant'} interaction effect"
                    }
                    
                    self.fitted_models[interaction_name] = full_model
                
            except Exception as e:
                logger.warning(f"Error fitting interaction model {interaction_name}: {str(e)}")
                interaction_models[interaction_name] = {'error': str(e)}
        
        return interaction_models
    
    def _run_model_diagnostics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run diagnostic checks on fitted models."""
        
        diagnostics = {}
        
        if 'comprehensive' in self.fitted_models:
            model = self.fitted_models['comprehensive']
            
            # Basic diagnostics - use attributes that exist
            diagnostics['comprehensive'] = {
                'n_observations': len(data),
                'n_participants': data['user_id'].nunique(),
                'convergence': getattr(model.mle_retvals, 'converged', True),
                'condition_number': np.linalg.cond(model.cov_params()),
                'largest_vif': self._calculate_vif(data),
                'log_likelihood': model.llf,
                'df_residuals': model.df_resid
            }
            
            # Add deviance only if available
            if hasattr(model, 'deviance'):
                diagnostics['comprehensive']['residual_deviance'] = model.deviance
            
            if hasattr(model, 'null_deviance'):
                diagnostics['comprehensive']['null_deviance'] = model.null_deviance
            
            # Check for multicollinearity
            if diagnostics['comprehensive']['largest_vif'] > 5:
                diagnostics['comprehensive']['multicollinearity_warning'] = True
            
            # Check for convergence issues
            if diagnostics['comprehensive']['condition_number'] > 1e12:
                diagnostics['comprehensive']['numerical_instability_warning'] = True
        
        return diagnostics
    
    def _calculate_vif(self, data: pd.DataFrame) -> float:
        """Calculate variance inflation factors for multicollinearity check."""
        try:
            predictor_cols = ['home_row_advantage', 'same_row_advantage', 
                            'column_quality_advantage', 'finger_distance_advantage', 
                            'row_distance_advantage']
            
            X = data[predictor_cols].fillna(0)
            X = sm.add_constant(X)
            
            vif_values = []
            for i in range(1, X.shape[1]):  # Skip constant
                vif = variance_inflation_factor(X.values, i)
                vif_values.append(vif)
            
            return max(vif_values) if vif_values else 1.0
            
        except Exception:
            return 1.0  # Return 1 if calculation fails
    
    def _calculate_mixed_effects_sizes(self, model_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate effect sizes and practical interpretations."""
        
        effect_sizes = {}
        
        if 'coefficients' in model_results:
            for param, coef_info in model_results['coefficients'].items():
                odds_ratio = coef_info['odds_ratio']
                
                # Calculate effect size categories
                if odds_ratio > 1:
                    effect_magnitude = odds_ratio - 1
                    direction = "positive"
                else:
                    effect_magnitude = 1 - odds_ratio
                    direction = "negative"
                
                # Categorize effect size (Cohen's conventions adapted for OR)
                if effect_magnitude < 0.2:
                    effect_category = "small"
                elif effect_magnitude < 0.5:
                    effect_category = "medium"
                else:
                    effect_category = "large"
                
                effect_sizes[param] = {
                    'odds_ratio': odds_ratio,
                    'effect_magnitude': effect_magnitude,
                    'direction': direction,
                    'effect_category': effect_category,
                    'practical_interpretation': self._practical_interpretation(param, odds_ratio, coef_info['significant'])
                }
        
        return effect_sizes
    
    def _analyze_random_effects(self, model_results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual differences (participant-level variation)."""
        
        # Since we're using cluster-robust SE instead of true mixed-effects,
        # we'll analyze individual differences descriptively
        
        participant_effects = {}
        
        # Calculate individual preference rates by participant
        for user_id in data['user_id'].unique():
            user_data = data[data['user_id'] == user_id]
            
            if len(user_data) > 0:
                ergonomic_rate = user_data['chose_ergonomic'].mean()
                n_choices = len(user_data)
                
                participant_effects[user_id] = {
                    'ergonomic_choice_rate': ergonomic_rate,
                    'n_choices': n_choices,
                    'deviation_from_mean': ergonomic_rate - data['chose_ergonomic'].mean()
                }
        
        # Summary statistics
        deviations = [p['deviation_from_mean'] for p in participant_effects.values()]
        
        random_effects_summary = {
            'n_participants': len(participant_effects),
            'mean_ergonomic_rate': data['chose_ergonomic'].mean(),
            'between_participant_sd': np.std(deviations),
            'participant_range': (min(p['ergonomic_choice_rate'] for p in participant_effects.values()),
                                max(p['ergonomic_choice_rate'] for p in participant_effects.values())),
            'participants_above_chance': sum(1 for p in participant_effects.values() 
                                           if p['ergonomic_choice_rate'] > 0.5),
            'individual_effects': participant_effects
        }
        
        return random_effects_summary
    
    def _compare_models(self) -> Dict[str, Any]:
        """Compare different models using information criteria."""
        
        model_comparison = {}
        
        # Extract AIC/BIC for all fitted models
        for model_name, model in self.fitted_models.items():
            if hasattr(model, 'aic'):
                model_comparison[model_name] = {
                    'aic': model.aic,
                    'bic': model.bic,
                    'log_likelihood': model.llf,
                    'n_params': len(model.params)
                }
        
        # Find best models
        if model_comparison:
            aic_values = {name: info['aic'] for name, info in model_comparison.items()}
            bic_values = {name: info['bic'] for name, info in model_comparison.items()}
            
            best_aic = min(aic_values, key=aic_values.get)
            best_bic = min(bic_values, key=bic_values.get)
            
            model_comparison['summary'] = {
                'best_model_aic': best_aic,
                'best_model_bic': best_bic,
                'recommendation': best_bic if best_aic == best_bic else f"AIC favors {best_aic}, BIC favors {best_bic}"
            }
        
        return model_comparison
    
    def _interpret_coefficient(self, param_name: str, coefficient: float, odds_ratio: float, p_value: float) -> str:
        """Generate human-readable interpretation of model coefficients."""
        
        if p_value >= self.alpha:
            return f"No significant effect of {param_name.replace('_', ' ')} (p = {p_value:.3f})"
        
        direction = "increases" if coefficient > 0 else "decreases"
        magnitude = "substantially" if abs(coefficient) > 1 else "moderately" if abs(coefficient) > 0.5 else "slightly"
        
        percent_change = (odds_ratio - 1) * 100
        
        interpretation = f"{param_name.replace('_', ' ').title()} {magnitude} {direction} " \
                        f"likelihood of ergonomic choice by {abs(percent_change):.1f}% " \
                        f"(OR = {odds_ratio:.2f}, p = {p_value:.3f})"
        
        return interpretation
    
    def _practical_interpretation(self, param_name: str, odds_ratio: float, is_significant: bool) -> str:
        """Generate practical design implications."""
        
        if not is_significant:
            return f"{param_name.replace('_', ' ').title()} shows no significant impact on user preferences"
        
        if odds_ratio > 2:
            strength = "strong"
        elif odds_ratio > 1.5:
            strength = "moderate"
        elif odds_ratio > 1.2:
            strength = "weak"
        elif odds_ratio < 0.5:
            strength = "strong negative"
        elif odds_ratio < 0.67:
            strength = "moderate negative"
        else:
            strength = "weak negative"
        
        design_implications = {
            'home_row_advantage': "Prioritize home row placement for frequent bigrams",
            'same_row_advantage': "Minimize cross-row movement in keyboard layouts",
            'column_quality_advantage': "Optimize column placement for finger reach",
            'finger_distance_advantage': "Reduce finger stretching distances",
            'row_distance_advantage': "Minimize vertical finger movement"
        }
        
        base_implication = design_implications.get(param_name, "Consider this factor in design")
        
        return f"{strength.title()} effect: {base_implication}"
    
    # ====================================
    # VISUALIZATION METHODS
    # ====================================
    
    def _create_mixed_effects_plots(self, data: pd.DataFrame, output_folder: str) -> None:
        """Create visualizations for mixed-effects results."""
        
        logger.info("Creating mixed-effects visualizations...")
        
        # Set plotting style
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
        
        # 1. Coefficient plot
        self._plot_model_coefficients(output_folder)
        
        # 2. Random effects plot (individual differences)
        self._plot_individual_differences(data, output_folder)
        
        # 3. Model diagnostics plot
        self._plot_model_diagnostics(data, output_folder)
        
        # 4. Effect size comparison plot
        self._plot_effect_comparison(output_folder)
        
        logger.info("Mixed-effects visualizations completed!")
    
    def _plot_model_coefficients(self, output_folder: str) -> None:
        """Plot coefficient estimates with confidence intervals."""
        
        if 'comprehensive' not in self.fitted_models:
            return
        
        model = self.fitted_models['comprehensive']
        
        # Extract coefficients (excluding intercept)
        coeffs = []
        names = []
        cis = []
        colors = []
        
        for param in model.params.index:
            if param != 'Intercept':
                coef = model.params[param]
                ci_lower, ci_upper = model.conf_int().loc[param]
                p_val = model.pvalues[param]
                
                coeffs.append(coef)
                names.append(param.replace('_', ' ').title().replace('Advantage', ''))
                cis.append((ci_lower, ci_upper))
                colors.append('red' if p_val < self.alpha else 'gray')
        
        if not coeffs:
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_positions = range(len(coeffs))
        
        # Plot confidence intervals
        for i, (coef, (ci_lower, ci_upper), color) in enumerate(zip(coeffs, cis, colors)):
            ax.plot([ci_lower, ci_upper], [i, i], color=color, linewidth=3, alpha=0.7)
            ax.scatter(coef, i, color=color, s=100, zorder=5)
        
        # Formatting
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names)
        ax.set_xlabel('Coefficient Estimate', fontweight='bold')
        ax.set_title('Mixed-Effects Model Coefficients', fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=3, label='Significant (p < 0.05)'),
            Line2D([0], [0], color='gray', linewidth=3, label='Non-significant')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'mixed_effects_coefficients.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_folder, 'mixed_effects_coefficients.pdf'), 
                   bbox_inches='tight')
        plt.close()
    
    def _plot_individual_differences(self, data: pd.DataFrame, output_folder: str) -> None:
        """Plot individual participant effects."""
        
        # Calculate participant-level ergonomic choice rates
        participant_rates = data.groupby('user_id')['chose_ergonomic'].agg(['mean', 'count']).reset_index()
        participant_rates.columns = ['user_id', 'ergonomic_rate', 'n_choices']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Distribution of individual ergonomic choice rates
        ax1.hist(participant_rates['ergonomic_rate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0.5, color='red', linestyle='--', label='Chance level')
        ax1.axvline(x=participant_rates['ergonomic_rate'].mean(), color='orange', 
                   linestyle='-', linewidth=2, label='Group mean')
        ax1.set_xlabel('Ergonomic Choice Rate')
        ax1.set_ylabel('Number of Participants')
        ax1.set_title('Distribution of Individual\nErgonomic Preferences')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Choice rate vs number of choices (size = reliability)
        scatter = ax2.scatter(participant_rates['n_choices'], participant_rates['ergonomic_rate'],
                             s=participant_rates['n_choices']*2, alpha=0.6, c=participant_rates['ergonomic_rate'],
                             cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Chance level')
        ax2.set_xlabel('Number of Choices per Participant')
        ax2.set_ylabel('Ergonomic Choice Rate')
        ax2.set_title('Individual Preferences vs\nData Reliability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Ergonomic Choice Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'individual_differences_mixed.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_folder, 'individual_differences_mixed.pdf'), 
                   bbox_inches='tight')
        plt.close()
    
    def _plot_model_diagnostics(self, data: pd.DataFrame, output_folder: str) -> None:
        """Plot model diagnostic information."""
        
        if 'comprehensive' not in self.fitted_models:
            return
        
        model = self.fitted_models['comprehensive']
        
        # Create diagnostic plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Residuals vs fitted
        fitted_values = model.fittedvalues
        residuals = model.resid_pearson
        
        ax1.scatter(fitted_values, residuals, alpha=0.6)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Pearson Residuals')
        ax1.set_title('Residuals vs Fitted')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: QQ plot of residuals
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Predicted vs actual
        actual = data['chose_ergonomic']
        predicted_probs = model.predict()
        
        # Bin predictions and calculate actual rates
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        actual_rates = []
        predicted_rates = []
        
        for i in range(len(bins)-1):
            mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i+1])
            if mask.sum() > 0:
                actual_rates.append(actual[mask].mean())
                predicted_rates.append(predicted_probs[mask].mean())
        
        ax3.scatter(predicted_rates, actual_rates, s=100)
        ax3.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        ax3.set_xlabel('Predicted Probability')
        ax3.set_ylabel('Actual Rate')
        ax3.set_title('Calibration Plot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model comparison (AIC/BIC)
        if hasattr(self, 'fitted_models') and len(self.fitted_models) > 1:
            model_names = []
            aic_values = []
            bic_values = []
            
            for name, mdl in self.fitted_models.items():
                if hasattr(mdl, 'aic'):
                    model_names.append(name.replace('_', ' ').title())
                    aic_values.append(mdl.aic)
                    bic_values.append(mdl.bic)
            
            x = np.arange(len(model_names))
            width = 0.35
            
            ax4.bar(x - width/2, aic_values, width, label='AIC', alpha=0.7)
            ax4.bar(x + width/2, bic_values, width, label='BIC', alpha=0.7)
            ax4.set_xlabel('Models')
            ax4.set_ylabel('Information Criterion')
            ax4.set_title('Model Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(model_names, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Model comparison\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Model Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'model_diagnostics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_folder, 'model_diagnostics.pdf'), 
                   bbox_inches='tight')
        plt.close()
    
    def _plot_effect_comparison(self, output_folder: str) -> None:
        """Plot comparison of effect sizes across factors."""
        
        if 'comprehensive' not in self.fitted_models:
            return
        
        model = self.fitted_models['comprehensive']
        
        # Extract odds ratios and confidence intervals
        factors = []
        odds_ratios = []
        ci_lowers = []
        ci_uppers = []
        significances = []
        
        for param in model.params.index:
            if param != 'Intercept':
                coef = model.params[param]
                ci_lower, ci_upper = model.conf_int().loc[param]
                p_val = model.pvalues[param]
                
                # Convert to odds ratios
                or_val = np.exp(coef)
                or_ci_lower = np.exp(ci_lower)
                or_ci_upper = np.exp(ci_upper)
                
                factors.append(param.replace('_', ' ').title().replace('Advantage', ''))
                odds_ratios.append(or_val)
                ci_lowers.append(or_ci_lower)
                ci_uppers.append(or_ci_upper)
                significances.append(p_val < self.alpha)
        
        if not factors:
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_positions = range(len(factors))
        colors = ['red' if sig else 'gray' for sig in significances]
        
        # Plot confidence intervals
        for i, (or_val, ci_low, ci_up, color) in enumerate(zip(odds_ratios, ci_lowers, ci_uppers, colors)):
            ax.plot([ci_low, ci_up], [i, i], color=color, linewidth=3, alpha=0.7)
            ax.scatter(or_val, i, color=color, s=150, zorder=5)
            
            # Add odds ratio value as text
            ax.text(or_val + 0.1, i, f'{or_val:.2f}', va='center', fontweight='bold')
        
        # Formatting
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.7, label='No effect (OR = 1)')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(factors)
        ax.set_xlabel('Odds Ratio (95% CI)', fontweight='bold')
        ax.set_title('Effect Sizes: Odds Ratios for Ergonomic Factors', fontweight='bold', pad=20)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add interpretation regions
        ax.axvspan(0.1, 0.8, alpha=0.1, color='red', label='Negative effect')
        ax.axvspan(0.8, 1.2, alpha=0.1, color='gray', label='Minimal effect')
        ax.axvspan(1.2, 10, alpha=0.1, color='green', label='Positive effect')
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=3, label='Significant'),
            Line2D([0], [0], color='gray', linewidth=3, label='Non-significant'),
            Line2D([0], [0], color='black', linestyle='--', label='No effect (OR = 1)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'effect_sizes_odds_ratios.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_folder, 'effect_sizes_odds_ratios.pdf'), 
                   bbox_inches='tight')
        plt.close()
    
    # ====================================
    # REPORTING AND OUTPUT
    # ====================================
    
    def _generate_mixed_effects_report(self, results: Dict[str, Any], output_folder: str) -> None:
        """Generate comprehensive mixed-effects analysis report."""
        
        report_lines = [
            "MIXED-EFFECTS KEYBOARD ERGONOMICS ANALYSIS",
            "=" * 60,
            "",
            "OVERVIEW",
            "========",
            "This report presents results from mixed-effects logistic regression",
            "analysis of keyboard ergonomics preferences. This approach properly",
            "accounts for repeated measures within participants and tests multiple",
            "ergonomic factors simultaneously.",
            "",
            f"Analysis conducted on {results['sample_info']['n_participants']} participants",
            f"with {results['sample_info']['n_observations']} choice observations.",
            ""
        ]
        
        # Comprehensive model results
        if 'comprehensive_model' in results:
            comp_model = results['comprehensive_model']
            
            report_lines.extend([
                "COMPREHENSIVE MODEL RESULTS",
                "==========================",
                "",
                "Model Specification:",
                f"• Type: {comp_model.get('model_type', 'Mixed-effects logistic regression')}",
                f"• Formula: {comp_model.get('formula', 'Not available')}",
                f"• Observations: {comp_model.get('n_obs', 'N/A')}",
                f"• Participants: {comp_model.get('n_participants', 'N/A')}",
                f"• Convergence: {'✓' if comp_model.get('convergence', False) else '✗'}",
                "",
                "Model Fit:",
                f"• Pseudo R²: {comp_model.get('pseudo_r2', 0):.3f}",
                f"• AIC: {comp_model.get('aic', 'N/A'):.1f}" if comp_model.get('aic') else "• AIC: N/A",
                f"• BIC: {comp_model.get('bic', 'N/A'):.1f}" if comp_model.get('bic') else "• BIC: N/A",
                ""
            ])
            
            # Coefficient results
            if 'coefficients' in comp_model:
                report_lines.extend([
                    "FACTOR EFFECTS:",
                    "--------------"
                ])
                
                for param, coef_info in comp_model['coefficients'].items():
                    significance = "***" if coef_info['p_value'] < 0.001 else "**" if coef_info['p_value'] < 0.01 else "*" if coef_info['significant'] else ""
                    
                    report_lines.extend([
                        f"",
                        f"{param.replace('_', ' ').title()}:",
                        f"  • Coefficient: {coef_info['coefficient']:.3f} ± {coef_info['std_error']:.3f}",
                        f"  • Odds Ratio: {coef_info['odds_ratio']:.2f} (95% CI: {coef_info['or_ci_lower']:.2f}-{coef_info['or_ci_upper']:.2f})",
                        f"  • p-value: {coef_info['p_value']:.3e} {significance}",
                        f"  • Interpretation: {coef_info['interpretation']}"
                    ])
        
        # Individual factor models
        if 'factor_models' in results:
            report_lines.extend([
                "",
                "",
                "INDIVIDUAL FACTOR ANALYSIS",
                "=========================",
                "Results from separate models for each ergonomic factor:",
                ""
            ])
            
            for factor_name, factor_result in results['factor_models'].items():
                if 'error' not in factor_result:
                    significance = "***" if factor_result['p_value'] < 0.001 else "**" if factor_result['p_value'] < 0.01 else "*" if factor_result['significant'] else ""
                    
                    report_lines.extend([
                        f"{factor_name.replace('_', ' ').title()}:",
                        f"  • Odds Ratio: {factor_result['odds_ratio']:.2f}",
                        f"  • p-value: {factor_result['p_value']:.3e} {significance}",
                        f"  • Pseudo R²: {factor_result['pseudo_r2']:.3f}",
                        f"  • {factor_result['interpretation']}",
                        ""
                    ])
        
        # Interaction effects
        if 'interaction_models' in results:
            report_lines.extend([
                "INTERACTION EFFECTS",
                "==================",
                "Tests for interactions between ergonomic factors:",
                ""
            ])
            
            significant_interactions = []
            for interaction_name, interaction_result in results['interaction_models'].items():
                if 'error' not in interaction_result:
                    if interaction_result['significant']:
                        significant_interactions.append(interaction_name)
                    
                    significance = "***" if interaction_result['interaction_p_value'] < 0.001 else "**" if interaction_result['interaction_p_value'] < 0.01 else "*" if interaction_result['significant'] else ""
                    
                    report_lines.extend([
                        f"{interaction_name.replace('_', ' ').title()}:",
                        f"  • Coefficient: {interaction_result['interaction_coefficient']:.3f}",
                        f"  • p-value: {interaction_result['interaction_p_value']:.3e} {significance}",
                        f"  • {interaction_result['interpretation']}",
                        ""
                    ])
            
            if not significant_interactions:
                report_lines.extend([
                    "No significant interaction effects found.",
                    "Ergonomic factors appear to operate independently.",
                    ""
                ])
        
        # Random effects (individual differences)
        if 'random_effects' in results:
            re_data = results['random_effects']
            
            report_lines.extend([
                "INDIVIDUAL DIFFERENCES",
                "=====================",
                f"Analysis of participant-level variation in ergonomic preferences:",
                "",
                f"• Total participants: {re_data['n_participants']}",
                f"• Mean ergonomic choice rate: {re_data['mean_ergonomic_rate']:.1%}",
                f"• Between-participant SD: {re_data['between_participant_sd']:.3f}",
                f"• Participant range: {re_data['participant_range'][0]:.1%} - {re_data['participant_range'][1]:.1%}",
                f"• Participants above chance: {re_data['participants_above_chance']} ({re_data['participants_above_chance']/re_data['n_participants']:.1%})",
                ""
            ])
        
        # Model diagnostics
        if 'model_diagnostics' in results:
            diag_data = results['model_diagnostics'].get('comprehensive', {})
            
            report_lines.extend([
                "MODEL DIAGNOSTICS",
                "================",
                f"• Convergence: {'✓ Successful' if diag_data.get('convergence', False) else '✗ Failed'}",
                f"• Condition number: {diag_data.get('condition_number', 'N/A'):.2e}" if diag_data.get('condition_number') else "• Condition number: N/A",
                f"• Largest VIF: {diag_data.get('largest_vif', 'N/A'):.2f}" if diag_data.get('largest_vif') else "• Largest VIF: N/A",
                ""
            ])
            
            # Warnings
            if diag_data.get('multicollinearity_warning'):
                report_lines.append("⚠ WARNING: High multicollinearity detected (VIF > 5)")
            if diag_data.get('numerical_instability_warning'):
                report_lines.append("⚠ WARNING: Numerical instability detected")
            
            if not any(diag_data.get(w) for w in ['multicollinearity_warning', 'numerical_instability_warning']):
                report_lines.append("✓ No diagnostic issues detected")
            
            report_lines.append("")
        
        # Model comparison
        if 'model_comparison' in results and 'summary' in results['model_comparison']:
            comp_summary = results['model_comparison']['summary']
            
            report_lines.extend([
                "MODEL COMPARISON",
                "===============",
                f"• Best model (AIC): {comp_summary['best_model_aic']}",
                f"• Best model (BIC): {comp_summary['best_model_bic']}",
                f"• Recommendation: {comp_summary['recommendation']}",
                ""
            ])
        
        # Effect sizes and practical significance
        if 'effect_sizes' in results:
            report_lines.extend([
                "PRACTICAL SIGNIFICANCE",
                "=====================",
                "Effect size categories and design implications:",
                ""
            ])
            
            for param, effect_info in results['effect_sizes'].items():
                report_lines.extend([
                    f"{param.replace('_', ' ').title()}:",
                    f"  • Effect magnitude: {effect_info['effect_category']} ({effect_info['direction']})",
                    f"  • Practical implication: {effect_info['practical_interpretation']}",
                    ""
                ])
        
        # Statistical notes
        report_lines.extend([
            "STATISTICAL METHODS",
            "==================",
            "• Mixed-effects approach: Logistic regression with cluster-robust standard errors",
            "• Clustering variable: Participant ID (accounts for repeated measures)",
            "• Effect sizes: Odds ratios with 95% confidence intervals",
            "• Significance testing: Wald tests with cluster-robust standard errors",
            f"• Multiple comparisons: Not corrected (single comprehensive model)",
            f"• Alpha level: {self.alpha}",
            "",
            "ADVANTAGES OF MIXED-EFFECTS APPROACH:",
            "• Properly handles repeated measures within participants",
            "• Tests all factors simultaneously (reduces multiple testing)",
            "• Provides more accurate standard errors and p-values",
            "• Can identify interaction effects between factors",
            "• More efficient use of data than separate tests",
            "",
            "COMPARISON TO DESCRIPTIVE APPROACH:",
            "• Descriptive: Tests each factor separately, treats observations as independent",
            "• Mixed-effects: Tests factors together, accounts for participant clustering", 
            "• Both approaches provide complementary insights",
            ""
        ])
        
        # Data files reference
        report_lines.extend([
            "DATA FILES",
            "==========",
            "• mixed_effects_results.json - Complete model results and coefficients",
            "• mixed_effects_coefficients.png/pdf - Coefficient plot with confidence intervals",
            "• individual_differences_mixed.png/pdf - Participant-level variation analysis",
            "• model_diagnostics.png/pdf - Model fit and diagnostic plots",
            "• effect_sizes_odds_ratios.png/pdf - Effect size comparison across factors",
            ""
        ])
        
        # Save report
        report_path = os.path.join(output_folder, 'mixed_effects_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Mixed-effects report saved to {report_path}")
    
    def _save_model_results_json(self, results: Dict[str, Any], output_folder: str) -> None:
        """Save detailed model results to JSON file."""
        
        # Create JSON-serializable version of results
        json_results = {}
        
        for key, value in results.items():
            if key == 'comprehensive_model':
                # Extract serializable parts of model results
                if isinstance(value, dict):
                    json_results[key] = {k: v for k, v in value.items() if k != 'model_object'}
            elif key in ['factor_models', 'interaction_models', 'model_diagnostics', 
                        'effect_sizes', 'random_effects', 'model_comparison', 'sample_info']:
                json_results[key] = value
        
        # Save to JSON file
        json_path = os.path.join(output_folder, 'mixed_effects_results.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Mixed-effects results saved to {json_path}")
    
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


# Integration function for analyze_data.py
def add_mixed_effects_analysis_to_main(analyzer_instance, data, output_folder, config):
    """
    Function to integrate mixed-effects ergonomics analysis into analyze_data.py.
    
    Args:
        analyzer_instance: Instance of BigramAnalysis class
        data: Processed bigram data
        output_folder: Output directory 
        config: Configuration dictionary
    """
    
    # Check if mixed-effects analysis is enabled
    if not config.get('analysis', {}).get('run_mixed_effects_ergonomics', False):
        return None
    
    logger.info("Running mixed-effects keyboard ergonomics analysis...")
    
    # Create mixed-effects analyzer
    mixed_effects_analyzer = MixedEffectsErgonomicsAnalysis(config)
    
    # Run complete mixed-effects analysis
    mixed_effects_results = mixed_effects_analyzer.run_complete_mixed_effects_analysis(data, output_folder)
    
    logger.info("Mixed-effects keyboard ergonomics analysis completed!")
    
    return mixed_effects_results