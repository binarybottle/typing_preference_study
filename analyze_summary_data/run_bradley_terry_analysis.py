#!/usr/bin/env python3
"""
Standalone script to run bigram Bradley-Terry analysis.

Usage:
    python run_bradley_terry_analysis.py --data path/to/your/data.csv
    
For the spatial model: 
Start with feature_set='standard' to ensure it distinguishes between:
  - Rows: key1_row_2, key1_row_3 effects
  - Columns: key1_col_2 through key1_col_5 effects
  - Fingers: key1_finger_2 through key1_finger_4 effects
  - Spatial transitions: roll_inward, roll_outward, adjacent_rows

  
Example:
    poetry run python3 run_bradley_terry_analysis.py --data input/nonProlific/process_data/tables/processed_consistent_choices.csv  --output-dir input/nonProlific/bradley_terry_model_spatial --skip-direct --no-middle-column --force-comprehensive --no-pruning
    poetry run python3 run_bradley_terry_analysis.py --data input/Prolific/process_data/tables/processed_consistent_choices.csv --output-dir input/Prolific/bradley_terry_model_spatial --skip-direct --no-middle-column --force-comprehensive --no-pruning

"""

import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict

# Import your analysis classes
from bradley_terry_model_direct import DirectBigramBTAnalyzer
from bradley_terry_model_spatial import SpatialBTModel

def run_data_sufficiency_analysis(data_file: str, min_comparisons: int = 5):
    """Run data sufficiency analysis to determine which approach to use."""
    
    print("=" * 60)
    print("BIGRAM DATA SUFFICIENCY ANALYSIS")
    print("=" * 60)
    print(f"Data file: {data_file}")
    print(f"Minimum comparisons threshold: {min_comparisons}")
    print()
    
    # Initialize analyzer
    try:
        analyzer = DirectBigramBTAnalyzer(data_file)
        print(f"✓ Successfully loaded data")
        print(f"✓ Found {len(analyzer.data)} choice records")
        print()
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    # Run sufficiency analysis
    print("Running data sufficiency analysis...")
    sufficiency_report = analyzer.create_sufficiency_report(min_comparisons)
    print(sufficiency_report)
    
    # Return the analyzer for further use
    return analyzer

def run_direct_bt_analysis(analyzer: DirectBigramBTAnalyzer, min_comparisons: int = 5):
    """Run direct Bradley-Terry analysis if data is sufficient."""
    
    print("\n" + "=" * 60)
    print("DIRECT BRADLEY-TERRY MODEL ANALYSIS")
    print("=" * 60)
    
    # Fit direct B-T model
    model_result = analyzer.fit_direct_bt_model(min_comparisons=min_comparisons)
    
    if model_result['success']:
        print(f"✓ Successfully fit direct B-T model!")
        print(f"✓ Model includes {model_result['n_bigrams']} bigrams")
        print(f"✓ Based on {model_result['model_diagnostics']['n_comparisons']} pairwise comparisons")
        print(f"✓ Total observations: {model_result['model_diagnostics']['total_observations']}")
        print()
        
        print("TOP 15 BIGRAMS BY DIRECT B-T STRENGTH:")
        print("-" * 50)
        for i, (bigram, strength) in enumerate(model_result['rankings'][:15]):
            ci = model_result['confidence_intervals'][bigram]
            print(f"{i+1:2d}. {bigram.upper()}: {strength:6.3f} [{ci[0]:6.3f}, {ci[1]:6.3f}]")
        
        print()
        print("BOTTOM 5 BIGRAMS:")
        print("-" * 50)
        for i, (bigram, strength) in enumerate(model_result['rankings'][-5:]):
            ci = model_result['confidence_intervals'][bigram]
            rank = len(model_result['rankings']) - 4 + i
            print(f"{rank:2d}. {bigram.upper()}: {strength:6.3f} [{ci[0]:6.3f}, {ci[1]:6.3f}]")
        
        return model_result
    else:
        print(f"✗ Direct B-T model failed: {model_result['error']}")
        return None

def display_all_bigrams(all_scores, show_features=True):
    """Display complete bigram rankings with optional feature details."""
    
    if all_scores is None:
        print("No bigram scores available")
        return
    
    print(f"\nALL {len(all_scores)} BIGRAMS BY SPATIAL MODEL:")
    print("=" * 90)
    
    if show_features:
        print(f"{'Rank':<4} {'Bigram':<6} {'Score':<8} {'Same Row':<8} {'ΔFinger':<8} {'Key1→Key2':<10} {'Pattern':<15}")
        print("-" * 90)
        
        for i, (_, row) in enumerate(all_scores.iterrows()):
            row_desc = f"R{row['key1_row']}→R{row['key2_row']}"
            pattern = row['row_pattern'] if 'row_pattern' in row else 'unknown'
            
            print(f"{i+1:<4} {row['bigram'].upper():<6} {row['score']:<8.3f} "
                  f"{str(row['same_row']):<8} {row['delta_finger']:<8} "
                  f"{row_desc:<10} {pattern:<15}")
    else:
        print(f"{'Rank':<4} {'Bigram':<6} {'Score':<8}")
        print("-" * 25)
        
        for i, (_, row) in enumerate(all_scores.iterrows()):
            print(f"{i+1:<4} {row['bigram'].upper():<6} {row['score']:<8.3f}")
    
    # Show score distribution
    scores = all_scores['score'].values
    print(f"\nSCORE DISTRIBUTION:")
    print(f"  Range: {scores.min():.3f} to {scores.max():.3f}")
    print(f"  Mean: {scores.mean():.3f}")
    print(f"  Std Dev: {scores.std():.3f}")
    
    # Show quartiles
    q25, q50, q75 = np.percentile(scores, [25, 50, 75])
    print(f"  Quartiles: Q1={q25:.3f}, Q2={q50:.3f}, Q3={q75:.3f}")

def analyze_statistical_significance(spatial_model_result, all_scores):
    """
    Analyze statistical significance and meaningful differences in spatial B-T results.
    
    IMPORTANT DISTINCTION:
    - Coefficients (e.g., +0.981 for inward_roll) = TRUE Bradley-Terry parameters
    - Bigram scores = COMPOSITE predictions (NOT true B-T strengths)
    """
    
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 60)
    
    # Get model parameters
    if 'final_params' in spatial_model_result:
        params = spatial_model_result['final_params']
        param_names = spatial_model_result['final_param_names']
    else:
        params = spatial_model_result.get('params', {})
        param_names = list(params.keys())
    
    print("1. BRADLEY-TERRY COEFFICIENTS (TRUE B-T PARAMETERS):")
    print("-" * 50)
    print("These ARE true Bradley-Terry strengths for spatial features")
    print()
    
    # Sort coefficients by magnitude
    coef_analysis = []
    for param_name, coef in params.items():
        if param_name != 'intercept':
            coef_analysis.append({
                'feature': param_name,
                'coefficient': coef,
                'abs_coefficient': abs(coef),
                'practical_significance': classify_practical_significance(abs(coef))
            })
    
    coef_df = pd.DataFrame(coef_analysis).sort_values('abs_coefficient', ascending=False)
    
    print(f"{'Feature':<25} {'Coefficient':<12} {'|Coef|':<8} {'Practical Sig':<15}")
    print("-" * 70)
    
    for _, row in coef_df.iterrows():
        direction = "+" if row['coefficient'] > 0 else ""
        print(f"{row['feature']:<25} {direction}{row['coefficient']:<11.3f} "
              f"{row['abs_coefficient']:<8.3f} {row['practical_significance']:<15}")
    
    print("\n2. BIGRAM SCORES (COMPOSITE PREDICTIONS):")
    print("-" * 50)
    print("These are NOT true B-T strengths - they're predictions based on spatial features")
    print()
    
    if all_scores is not None:
        scores = all_scores['score'].values
        
        # Calculate meaningful difference thresholds
        score_std = scores.std()
        small_diff = 0.2 * score_std    # Small effect
        medium_diff = 0.5 * score_std   # Medium effect  
        large_diff = 0.8 * score_std    # Large effect
        
        print(f"Score Statistics:")
        print(f"  Range: {scores.min():.3f} to {scores.max():.3f}")
        print(f"  Standard Deviation: {score_std:.3f}")
        print()
        print(f"Meaningful Difference Thresholds (Cohen's d-inspired):")
        print(f"  Small difference:  ≥{small_diff:.3f}")
        print(f"  Medium difference: ≥{medium_diff:.3f}")  
        print(f"  Large difference:  ≥{large_diff:.3f}")
        print()
        
        # Analyze top vs bottom bigrams
        top_10_mean = all_scores.head(10)['score'].mean()
        bottom_10_mean = all_scores.tail(10)['score'].mean()
        top_bottom_diff = top_10_mean - bottom_10_mean
        
        print(f"Top 10 vs Bottom 10 Analysis:")
        print(f"  Top 10 mean score: {top_10_mean:.3f}")
        print(f"  Bottom 10 mean score: {bottom_10_mean:.3f}")
        print(f"  Difference: {top_bottom_diff:.3f}")
        
        if top_bottom_diff >= large_diff:
            significance_level = "LARGE difference (highly meaningful)"
        elif top_bottom_diff >= medium_diff:
            significance_level = "MEDIUM difference (meaningful)"
        elif top_bottom_diff >= small_diff:
            significance_level = "SMALL difference (possibly meaningful)"
        else:
            significance_level = "NEGLIGIBLE difference (likely noise)"
        
        print(f"  Interpretation: {significance_level}")
        
        # Pairwise comparison example
        print(f"\n3. EXAMPLE PAIRWISE COMPARISONS:")
        print("-" * 40)
        if len(all_scores) >= 3:
            bigram1 = all_scores.iloc[0]  # Top bigram
            bigram2 = all_scores.iloc[1]  # Second bigram  
            bigram3 = all_scores.iloc[len(all_scores)//2]  # Middle bigram
            
            diff_1_2 = bigram1['score'] - bigram2['score']
            diff_1_3 = bigram1['score'] - bigram3['score']
            
            print(f"{bigram1['bigram'].upper()} vs {bigram2['bigram'].upper()}: {diff_1_2:.3f} difference")
            print(f"  → {interpret_difference(diff_1_2, small_diff, medium_diff, large_diff)}")
            
            print(f"{bigram1['bigram'].upper()} vs {bigram3['bigram'].upper()}: {diff_1_3:.3f} difference")  
            print(f"  → {interpret_difference(diff_1_3, small_diff, medium_diff, large_diff)}")
    
    print(f"\n4. CONFIDENCE INTERVALS (CURRENT LIMITATION):")
    print("-" * 50)
    print("⚠️  Current spatial model doesn't provide confidence intervals for bigram scores")
    print("   To get statistical significance for individual bigrams, would need:")
    print("   • Bootstrap resampling (like your original script)")
    print("   • Hessian-based standard errors")
    print("   • Or the full 210-parameter direct B-T model")

def classify_practical_significance(abs_coef):
    """Classify practical significance of coefficients."""
    if abs_coef >= 0.5:
        return "Large"
    elif abs_coef >= 0.2:
        return "Medium"
    elif abs_coef >= 0.1:
        return "Small"
    else:
        return "Negligible"

def interpret_difference(diff, small_thresh, medium_thresh, large_thresh):
    """Interpret the meaningfulness of a score difference."""
    abs_diff = abs(diff)
    if abs_diff >= large_thresh:
        return "Large difference (highly meaningful)"
    elif abs_diff >= medium_thresh:
        return "Medium difference (meaningful)"
    elif abs_diff >= small_thresh:
        return "Small difference (possibly meaningful)"
    else:
        return "Negligible difference (likely noise)"

def save_sophisticated_results(spatial_model_result, all_scores, bigram_comparisons, output_dir="spatial_analysis_results"):
    """
    Save spatial B-T results using sophisticated output system like original script.
    """
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = os.path.join(output_dir, f"spatial_bt_analysis_{timestamp}")
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"\n📁 Creating sophisticated analysis output in: {full_output_dir}")
    
    # 1. STATISTICAL TABLES (like original script)
    print("Generating statistical tables...")
    _create_spatial_statistical_tables(spatial_model_result, all_scores, full_output_dir)
    
    # 2. COMPREHENSIVE REPORT (like original script)  
    print("Generating comprehensive report...")
    _create_spatial_comprehensive_report(spatial_model_result, all_scores, bigram_comparisons, full_output_dir)
    
    # 3. VISUALIZATIONS (like original script)
    print("Generating visualizations...")
    _create_spatial_visualizations(spatial_model_result, all_scores, full_output_dir)
    
    # 4. RAW RESULTS (JSON)
    print("Saving raw results...")
    _save_spatial_raw_results(spatial_model_result, all_scores, full_output_dir)
    
    print(f"\n✅ SOPHISTICATED ANALYSIS COMPLETE!")
    print(f"📊 Results saved to: {os.path.abspath(full_output_dir)}")
    print(f"📋 Open '{full_output_dir}/spatial_bt_comprehensive_report.txt' for summary")
    
    return full_output_dir

def _create_spatial_statistical_tables(model_result, all_scores, output_dir):
    """Create detailed statistical tables like original script."""
    
    # Table 1: Spatial Feature Statistics
    if 'final_params' in model_result:
        params = model_result['final_params']
    else:
        params = model_result.get('params', {})
    
    feature_stats = []
    for feature, coef in params.items():
        if feature != 'intercept':
            feature_stats.append({
                'Feature': feature,
                'Coefficient': coef,
                'Abs_Coefficient': abs(coef),
                'Practical_Significance': 'Large' if abs(coef) >= 0.5 else
                                        'Medium' if abs(coef) >= 0.2 else
                                        'Small' if abs(coef) >= 0.1 else 'Negligible',
                'Direction': 'Positive' if coef > 0 else 'Negative'
            })
    
    feature_df = pd.DataFrame(feature_stats).sort_values('Abs_Coefficient', ascending=False)
    feature_df.to_csv(os.path.join(output_dir, 'spatial_feature_statistics.csv'), index=False)
    
    # Table 2: Complete Bigram Rankings
    if all_scores is not None:
        # Enhanced bigram table with additional metrics
        enhanced_scores = all_scores.copy()
        
        # Add quartile rankings
        enhanced_scores['Score_Quartile'] = pd.qcut(enhanced_scores['score'], 4, labels=['Q1_Bottom', 'Q2_Lower', 'Q3_Upper', 'Q4_Top'])
        
        # Add relative rankings
        enhanced_scores['Percentile'] = enhanced_scores['score'].rank(pct=True) * 100
        
        # Add feature interpretations
        enhanced_scores['Spatial_Pattern'] = enhanced_scores.apply(_interpret_spatial_pattern, axis=1)
        
        enhanced_scores.to_csv(os.path.join(output_dir, 'complete_bigram_rankings.csv'), index=False)
        
        # Table 3: Top/Bottom Analysis
        top_bottom_analysis = []
        
        for quartile in ['Q4_Top', 'Q3_Upper', 'Q2_Lower', 'Q1_Bottom']:
            quartile_data = enhanced_scores[enhanced_scores['Score_Quartile'] == quartile]
            
            top_bottom_analysis.append({
                'Quartile': quartile,
                'Count': len(quartile_data),
                'Mean_Score': quartile_data['score'].mean(),
                'Std_Score': quartile_data['score'].std(),
                'Same_Row_Pct': (quartile_data['same_row'] == True).mean() * 100,
                'Example_Bigrams': ', '.join(quartile_data.head(3)['bigram'].str.upper())
            })
        
        quartile_df = pd.DataFrame(top_bottom_analysis)
        quartile_df.to_csv(os.path.join(output_dir, 'quartile_analysis.csv'), index=False)

def _interpret_spatial_pattern(row):
    """Interpret the spatial pattern of a bigram."""
    patterns = []
    
    if row['same_row']:
        patterns.append('Same-Row')
        if row['delta_finger'] > 0:
            patterns.append('Inward-Roll')
        elif row['delta_finger'] < 0:
            patterns.append('Outward-Roll')
    else:
        patterns.append('Cross-Row')
    
    if abs(row['delta_finger']) == 1:
        patterns.append('Adjacent-Fingers')
    elif abs(row['delta_finger']) >= 2:
        patterns.append('Wide-Span')
    
    return '_'.join(patterns) if patterns else 'Unknown'

def _create_spatial_comprehensive_report(model_result, all_scores, bigram_comparisons, output_dir):
    """Create comprehensive text report like original script."""
    
    report_lines = [
        "SPATIAL BRADLEY-TERRY MODEL COMPREHENSIVE ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "EXECUTIVE SUMMARY",
        "================",
        "",
        "This analysis used a spatial Bradley-Terry model to rank bigrams based on",
        "keyboard spatial features (rows, fingers, directions) rather than decomposition.",
        "The approach provides unified bigram preference modeling with spatial interpretability.",
        "",
        "MODEL PERFORMANCE",
        "================",
        ""
    ]
    
    # Add model statistics
    if model_result:
        report_lines.extend([
            f"Model Success: {'✓' if model_result.get('success', False) else '✗'}",
            f"Parameters: {model_result.get('n_parameters', 'unknown')}",
            f"Log-Likelihood: {model_result.get('log_likelihood', 'unknown'):.2f}",
            f"Feature Set: {model_result.get('feature_set', 'unknown')}",
            f"Optimizer: {model_result.get('optimizer_used', 'unknown')}",
            f"Filtered Extreme Cases: {model_result.get('filtered_extreme_cases', 0)}",
            ""
        ])
    
    # Add feature analysis
    if 'final_params' in model_result:
        params = model_result['final_params']
    else:
        params = model_result.get('params', {})
    
    if params:
        report_lines.extend([
            "SPATIAL FEATURE ANALYSIS",
            "========================",
            "",
            "Feature Coefficients (True Bradley-Terry Parameters):",
            "-" * 55
        ])
        
        # Sort by magnitude
        sorted_params = sorted([(k, v) for k, v in params.items() if k != 'intercept'], 
                              key=lambda x: abs(x[1]), reverse=True)
        
        for feature, coef in sorted_params:
            direction = "+" if coef >= 0 else ""
            significance = ("***" if abs(coef) >= 0.5 else 
                          "**" if abs(coef) >= 0.2 else 
                          "*" if abs(coef) >= 0.1 else "")
            
            report_lines.append(f"  {feature:<25} {direction}{coef:7.3f} {significance}")
        
        report_lines.extend([
            "",
            "Significance: *** = Large effect (|coef| ≥ 0.5)",
            "             ** = Medium effect (|coef| ≥ 0.2)", 
            "             * = Small effect (|coef| ≥ 0.1)",
            ""
        ])
    
    # Add bigram rankings
    if all_scores is not None:
        report_lines.extend([
            "BIGRAM RANKINGS",
            "===============",
            "",
            f"Total bigrams analyzed: {len(all_scores)}",
            "",
            "Top 20 Bigrams:",
            "-" * 60,
            f"{'Rank':<4} {'Bigram':<6} {'Score':<8} {'Pattern':<20} {'Interpretation'}"
        ])
        
        for i in range(min(20, len(all_scores))):
            row = all_scores.iloc[i]
            pattern = _interpret_spatial_pattern(row)
            interpretation = _interpret_bigram_preference(row)
            
            report_lines.append(
                f"{i+1:<4} {row['bigram'].upper():<6} {row['score']:<8.3f} "
                f"{pattern:<20} {interpretation}"
            )
        
        report_lines.extend([
            "",
            "Bottom 10 Bigrams:",
            "-" * 40
        ])
        
        for i in range(max(0, len(all_scores)-10), len(all_scores)):
            row = all_scores.iloc[i]
            rank = i + 1
            report_lines.append(
                f"{rank:<4} {row['bigram'].upper():<6} {row['score']:<8.3f}"
            )
    
    # Add statistical summary
    if all_scores is not None:
        scores = all_scores['score'].values
        report_lines.extend([
            "",
            "STATISTICAL SUMMARY",
            "==================",
            "",
            f"Score Statistics:",
            f"  Range: {scores.min():.3f} to {scores.max():.3f}",
            f"  Mean: {scores.mean():.3f}",
            f"  Standard Deviation: {scores.std():.3f}",
            f"  Score Spread: {scores.max() - scores.min():.3f}",
            ""
        ])
    
    # Add methodology
    report_lines.extend([
        "METHODOLOGY",
        "===========",
        "",
        "• Spatial Bradley-Terry model estimates bigram preferences from spatial features",
        "• Features include: row positions, finger assignments, movement directions",
        "• Model coefficients are true Bradley-Terry parameters for spatial patterns",
        "• Bigram scores are composite predictions (NOT direct B-T strengths)",
        "• Extreme cases (win rates <5% or >95%) filtered for numerical stability",
        "• Powell optimization used for robust parameter estimation",
        "",
        "FILES GENERATED",
        "===============",
        "",
        "Statistical Tables:",
        "• spatial_feature_statistics.csv - Feature coefficients and significance",
        "• complete_bigram_rankings.csv - All bigram scores with spatial patterns", 
        "• quartile_analysis.csv - Performance by score quartiles",
        "",
        "Visualizations:",
        "• spatial_coefficients_plot.png - Feature coefficient forest plot",
        "• bigram_score_distribution.png - Score distribution histogram",
        "• top_bottom_comparison.png - Top vs bottom bigrams comparison",
        "",
        "Raw Results:",
        "• spatial_model_results.json - Complete model output for further analysis"
    ])
    
    # Save report
    report_path = os.path.join(output_dir, 'spatial_bt_comprehensive_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

def _interpret_bigram_preference(row):
    """Interpret why a bigram might be preferred."""
    if row['score'] > 8.5:
        return "Highly preferred spatial pattern"
    elif row['score'] > 7.5:
        return "Preferred spatial pattern"
    elif row['score'] > 6.5:
        return "Neutral spatial pattern"
    else:
        return "Avoided spatial pattern"

def _create_spatial_visualizations(model_result, all_scores, output_dir):
    """Create visualizations like original script."""
    
    plt.style.use('default')
    
    # Visualization 1: Feature Coefficients Forest Plot
    if 'final_params' in model_result:
        params = model_result['final_params']
    else:
        params = model_result.get('params', {})
    
    if params:
        # Remove intercept for visualization
        features = {k: v for k, v in params.items() if k != 'intercept'}
        
        if features:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            feature_names = list(features.keys())
            coefficients = list(features.values())
            colors = ['green' if c > 0 else 'red' for c in coefficients]
            
            y_pos = np.arange(len(feature_names))
            
            ax.barh(y_pos, coefficients, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Bradley-Terry Coefficient')
            ax.set_title('Spatial Feature Preferences (True B-T Parameters)')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'spatial_coefficients_plot.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Visualization 2: Bigram Score Distribution
    if all_scores is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = all_scores['score'].values
        ax.hist(scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(scores.mean(), color='red', linestyle='--', label=f'Mean: {scores.mean():.3f}')
        ax.set_xlabel('Bigram Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Bigram Preference Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bigram_score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Visualization 3: Top vs Bottom Bigrams Comparison
    if all_scores is not None and len(all_scores) >= 20:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top 10
        top_10 = all_scores.head(10)
        ax1.barh(range(10), top_10['score'], color='green', alpha=0.7)
        ax1.set_yticks(range(10))
        ax1.set_yticklabels(top_10['bigram'].str.upper())
        ax1.set_xlabel('Score')
        ax1.set_title('Top 10 Bigrams')
        ax1.grid(True, alpha=0.3)
        
        # Bottom 10
        bottom_10 = all_scores.tail(10)
        ax2.barh(range(10), bottom_10['score'], color='red', alpha=0.7)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels(bottom_10['bigram'].str.upper())
        ax2.set_xlabel('Score')
        ax2.set_title('Bottom 10 Bigrams')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_bottom_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def _save_spatial_raw_results(model_result, all_scores, output_dir):
    """Save complete raw results in JSON format."""
    
    raw_results = {
        'analysis_info': {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'spatial_bradley_terry',
            'analysis_version': '2.0'
        },
        'model_results': model_result,
        'bigram_scores': all_scores.to_dict('records') if all_scores is not None else None
    }
    
    # Convert numpy types for JSON serialization
    raw_results = _convert_numpy_types(raw_results)
    
    with open(os.path.join(output_dir, 'spatial_model_results.json'), 'w') as f:
        json.dump(raw_results, f, indent=2, default=str)

def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def run_spatial_bt_analysis(data_file: str, include_middle_column: bool = True, 
                           feature_set: str = 'standard', use_pruning: bool = True):
    """Run spatial Bradley-Terry analysis with comprehensive output."""
    
    print("\n" + "=" * 60)
    print("SPATIAL BRADLEY-TERRY MODEL ANALYSIS")
    print("=" * 60)
    print(f"Include middle column (GTB): {include_middle_column}")
    print(f"Feature set: {feature_set}")
    print(f"Use automatic pruning: {use_pruning}")
    print()
    
    # Initialize spatial model
    try:
        spatial_model = SpatialBTModel(data_file, include_middle_column=include_middle_column)
        print(f"✓ Successfully loaded data for spatial model")
        print()
    except Exception as e:
        print(f"✗ Error loading data for spatial model: {e}")
        return None, None
    
    # Fit model (with or without pruning)
    if use_pruning:
        print("Fitting with automatic feature pruning...")
        result = spatial_model.fit_with_feature_pruning(
            initial_feature_set=feature_set,  # ← FIXED: Now uses detected feature set
            significance_threshold=0.05,
            min_effect_size=0.1
        )
        
        if result['success']:
            print(f"✓ Feature pruning completed!")
            print(f"✓ Removed {result['features_removed']} features")
            print(f"✓ Final model: {len(result['final_param_names'])} parameters")
            print(f"✓ Log-likelihood: {result['final_model']['log_likelihood']:.2f}")
            if 'optimizer_used' in result['final_model']:
                print(f"✓ Optimizer: {result['final_model']['optimizer_used']}")
            print()
            
            # Show important features
            important_features = {k: v for k, v in result['final_params'].items() 
                                 if abs(v) > 0.15}
            print("IMPORTANT FEATURES (|coef| > 0.15):")
            print("-" * 40)
            for feature, coef in sorted(important_features.items(), 
                                      key=lambda x: abs(x[1]), reverse=True):
                direction = "+" if coef > 0 else "-"
                print(f"  {direction} {feature}: {coef:6.3f}")
            
            final_model = result
        else:
            print(f"✗ Pruned model failed: {result['error']}")
            print("→ Falling back to simpler feature set...")
            
            # Fallback: try standard feature set without pruning
            result = spatial_model.fit_spatial_model(feature_set='standard')
            
            if result['success']:
                print(f"✓ Fallback model fitted successfully!")
                print(f"✓ Parameters: {result['n_parameters']}")
                print(f"✓ Log-likelihood: {result['log_likelihood']:.2f}")
                if 'optimizer_used' in result:
                    print(f"✓ Optimizer: {result['optimizer_used']}")
                final_model = result
            else:
                print(f"✗ Fallback model also failed: {result['error']}")
                print("→ Trying minimal feature set...")
                
                # Last resort: minimal feature set
                result = spatial_model.fit_spatial_model(feature_set='minimal')
                
                if result['success']:
                    print(f"✓ Minimal model fitted successfully!")
                    print(f"✓ Parameters: {result['n_parameters']}")
                    print(f"✓ Log-likelihood: {result['log_likelihood']:.2f}")
                    if 'optimizer_used' in result:
                        print(f"✓ Optimizer: {result['optimizer_used']}")
                    final_model = result
                else:
                    print(f"✗ All spatial models failed: {result['error']}")
                    return None, None
    else:
        print(f"Fitting {feature_set} model without pruning...")
        result = spatial_model.fit_spatial_model(feature_set=feature_set)
        
        if result['success']:
            print(f"✓ Spatial model fitted successfully!")
            print(f"✓ Parameters: {result['n_parameters']}")
            print(f"✓ Log-likelihood: {result['log_likelihood']:.2f}")
            if 'optimizer_used' in result:
                print(f"✓ Optimizer: {result['optimizer_used']}")
            final_model = result
        else:
            print(f"✗ Spatial model failed: {result['error']}")
            print("→ Trying simpler feature set...")
            
            # Fallback to minimal if requested feature set fails
            if feature_set != 'minimal':
                result = spatial_model.fit_spatial_model(feature_set='minimal')
                if result['success']:
                    print(f"✓ Minimal fallback model fitted!")
                    print(f"✓ Parameters: {result['n_parameters']}")
                    print(f"✓ Log-likelihood: {result['log_likelihood']:.2f}")
                    if 'optimizer_used' in result:
                        print(f"✓ Optimizer: {result['optimizer_used']}")
                    final_model = result
                else:
                    print(f"✗ Even minimal model failed: {result['error']}")
                    return None, None
            else:
                return None, None
    
    # Score all bigrams  
    print("\nScoring all bigrams...")
    try:
        all_scores = spatial_model.score_all_bigrams()
        
        # Display all bigrams (not just top 15)
        display_all_bigrams(all_scores, show_features=True)
        
        # Analyze statistical significance
        analyze_statistical_significance(final_model, all_scores)
        
        return final_model, all_scores
    
    except Exception as e:
        print(f"✗ Error scoring bigrams: {e}")
        return final_model, None

def main():
    """Main function with command line argument parsing."""
    
    parser = argparse.ArgumentParser(description='Run bigram Bradley-Terry analysis')
    parser.add_argument('--data', required=False, 
                       help='Path to CSV data file')
    parser.add_argument('--min-comparisons', type=int, default=5,
                       help='Minimum comparisons per bigram pair (default: 5)')
    parser.add_argument('--skip-direct', action='store_true',
                       help='Skip direct B-T analysis (useful if insufficient data)')
    parser.add_argument('--no-middle-column', action='store_true',
                       help='Exclude middle column keys (GTB) - use only 12 keys')
    parser.add_argument('--feature-set', choices=['minimal', 'standard', 'comprehensive'],
                       default='standard', help='Feature set for spatial model')
    parser.add_argument('--no-pruning', action='store_true',
                       help='Disable automatic feature pruning')
    parser.add_argument('--simple-first', action='store_true',
                       help='Start with minimal feature set (recommended for sparse data)')
    parser.add_argument('--force-comprehensive', action='store_true',
                       help='Force comprehensive features even with sparse data')
    parser.add_argument('--output-dir', default='spatial_analysis_results',
                       help='Output directory for results (default: spatial_analysis_results)')
    
    args = parser.parse_args()
    
    # Set data file
    if args.data:
        data_file = args.data
    else:
        # Default data file - UPDATE THIS PATH FOR YOUR DATA
        data_file = 'data/filtered_data.csv'  # <-- UPDATE THIS PATH
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        print("Please specify the correct path using --data argument")
        sys.exit(1)
    
    # Step 1: Always run data sufficiency analysis
    analyzer = run_data_sufficiency_analysis(data_file, args.min_comparisons)
    if analyzer is None:
        sys.exit(1)
    
    # Step 2: Run direct B-T analysis if not skipped
    direct_result = None
    if not args.skip_direct:
        direct_result = run_direct_bt_analysis(analyzer, args.min_comparisons)
    
    # Step 3: Always run spatial B-T analysis
    # Intelligent feature set selection
    feature_set_to_use = args.feature_set
    
    if args.force_comprehensive:
        print("→ Using comprehensive feature set (user requested --force-comprehensive)")
        feature_set_to_use = 'comprehensive'
    elif args.simple_first:
        print("→ Using minimal feature set (user requested --simple-first)")
        feature_set_to_use = 'minimal'
    elif (analyzer and hasattr(analyzer, 'bigram_comparisons') and 
          len(analyzer.bigram_comparisons) < 500 and 
          not args.force_comprehensive):
        if args.feature_set != 'minimal':
            print(f"→ Overriding --feature-set {args.feature_set} → minimal due to sparse data ({len(analyzer.bigram_comparisons)} pairs)")
            print("  Use --force-comprehensive to override this safety check")
        feature_set_to_use = 'minimal'
    
    spatial_result = run_spatial_bt_analysis(
        data_file,
        include_middle_column=not args.no_middle_column,
        feature_set=feature_set_to_use,
        use_pruning=not args.no_pruning
    )
    
    # Unpack results (handle None case)
    if spatial_result[0] is not None:
        spatial_model_result, all_scores = spatial_result
    else:
        spatial_model_result, all_scores = None, None
    
    # Step 4: Save sophisticated results
    if spatial_model_result:
        try:
            output_path = save_sophisticated_results(
                spatial_model_result, 
                all_scores, 
                analyzer.bigram_comparisons, 
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"⚠ Warning: Could not save sophisticated results: {e}")
    
    # Step 5: Summary comparison if both models worked
    if direct_result and spatial_model_result:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        if args.no_pruning or 'final_param_names' not in spatial_model_result:
            spatial_params = spatial_model_result['n_parameters']
            spatial_ll = spatial_model_result['log_likelihood']
        else:
            spatial_params = len(spatial_model_result['final_param_names'])
            spatial_ll = spatial_model_result['final_model']['log_likelihood']
        
        print(f"Direct B-T Model:")
        print(f"  Parameters: {direct_result['n_bigrams']}")
        print(f"  Bigrams included: {direct_result['n_bigrams']}/{len(analyzer.all_bigrams)}")
        print()
        print(f"Spatial B-T Model:")
        print(f"  Parameters: {spatial_params}")
        print(f"  Log-likelihood: {spatial_ll:.2f}")
        print(f"  Feature set: {feature_set_to_use}")
        print()
        
        # AIC comparison
        direct_aic = 2 * direct_result['n_bigrams'] - 2 * 0  # Assuming LL not stored
        spatial_aic = 2 * spatial_params - 2 * spatial_ll
        
        print(f"Spatial model AIC: {spatial_aic:.2f}")
        print("(Lower AIC = better model)")
    
    # Final status
    if spatial_model_result:
        print(f"\n✅ ANALYSIS COMPLETE! Spatial model succeeded.")
        if all_scores is not None:
            print(f"✓ Successfully scored all {len(all_scores)} bigrams.")
        
        # Show actual feature set used
        if hasattr(spatial_model_result, 'get') and spatial_model_result.get('feature_set'):
            actual_feature_set = spatial_model_result['feature_set']
        else:
            actual_feature_set = feature_set_to_use
        print(f"✓ Used feature set: {actual_feature_set}")
        
        if 'output_path' in locals():
            print(f"📁 Comprehensive results: {output_path}")
    else:
        print(f"\n⚠ Analysis completed with issues - spatial model failed.")
        print("Try running with one of these options:")
        print("  --simple-first")
        print("  --feature-set minimal") 
        print("  --no-pruning")
        print("  --no-middle-column")
        print("  --min-comparisons 10")

if __name__ == "__main__":
    main()