#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple scatter plot: Prediction probability vs Hospitalization count
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# English font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """Load prediction data and original hospitalization data"""
    try:
        # Load prediction data
        predictions = pd.read_csv('心不全気象予測モデル_10年分最適化版_結果/all_predictions.csv')
        print(f"Prediction data loaded: {len(predictions)} records")
        
        # Load original data to get hospitalization counts
        original_data = pd.read_csv('hf_weather_merged.csv')
        original_data['date'] = pd.to_datetime(original_data['date'])
        print(f"Original data loaded: {len(original_data)} records")
        
        return predictions, original_data
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None, None

def create_simple_scatter_plots(predictions, original_data):
    """Create simple scatter plots showing prediction vs hospitalization"""
    
    # Merge prediction data with original data using date index
    predictions['date'] = pd.to_numeric(predictions['date'])
    merged_data = predictions.merge(original_data, left_on='date', right_index=True, how='left')
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction probability vs Hospitalization count
    axes[0, 0].scatter(merged_data['predicted_prob'], merged_data['people_hf'], alpha=0.6)
    axes[0, 0].set_xlabel('Prediction Probability')
    axes[0, 0].set_ylabel('Hospitalization Count')
    axes[0, 0].set_title('Prediction vs Hospitalization Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calculate correlation
    correlation = merged_data['predicted_prob'].corr(merged_data['people_hf'])
    axes[0, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                     transform=axes[0, 0].transAxes, fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Prediction probability distribution by actual risk
    axes[0, 1].hist(merged_data[merged_data['actual'] == 0]['predicted_prob'], 
                     alpha=0.7, label='Low Risk (Actual)', bins=20)
    axes[0, 1].hist(merged_data[merged_data['actual'] == 1]['predicted_prob'], 
                     alpha=0.7, label='High Risk (Actual)', bins=20)
    axes[0, 1].set_xlabel('Prediction Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Prediction Distribution by Actual Risk')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot: Prediction by actual risk
    axes[1, 0].boxplot([merged_data[merged_data['actual'] == 0]['predicted_prob'],
                        merged_data[merged_data['actual'] == 1]['predicted_prob']],
                       labels=['Low Risk', 'High Risk'])
    axes[1, 0].set_ylabel('Prediction Probability')
    axes[1, 0].set_title('Prediction by Actual Risk Level')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(merged_data['actual'], merged_data['predicted_prob'])
    auc_score = roc_auc_score(merged_data['actual'], merged_data['predicted_prob'])
    
    axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('心不全気象予測モデル_10年分最適化版_結果/simple_prediction_vs_hospitalization.png', 
                dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid hanging
    plt.close()  # Close the figure to free memory
    
    print(f"Simple scatter plots saved: simple_prediction_vs_hospitalization.png")
    print(f"Overall AUC: {auc_score:.4f}")
    print(f"Correlation with hospitalization count: {correlation:.4f}")
    
    return merged_data, auc_score, correlation

def analyze_fold_performance(predictions):
    """Analyze performance by fold"""
    print("\n=== Fold-by-Fold Analysis ===")
    
    fold_stats = []
    for fold in sorted(predictions['fold'].unique()):
        fold_data = predictions[predictions['fold'] == fold]
        if len(fold_data['actual'].unique()) > 1:
            auc = roc_auc_score(fold_data['actual'], fold_data['predicted_prob'])
            fold_stats.append({
                'fold': fold,
                'auc': auc,
                'samples': len(fold_data),
                'high_risk_ratio': fold_data['actual'].mean()
            })
    
    fold_df = pd.DataFrame(fold_stats)
    print(f"Fold AUCs:")
    for _, row in fold_df.iterrows():
        print(f"  Fold {row['fold']}: AUC = {row['auc']:.4f} (samples: {row['samples']})")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean AUC: {fold_df['auc'].mean():.4f} ± {fold_df['auc'].std():.4f}")
    print(f"  Min AUC: {fold_df['auc'].min():.4f}")
    print(f"  Max AUC: {fold_df['auc'].max():.4f}")
    
    return fold_df

def check_data_leakage_simple(predictions):
    """Simple data leakage check"""
    print("\n=== Data Leakage Check ===")
    
    # Basic statistics
    print(f"Prediction probability statistics:")
    print(f"  Mean: {predictions['predicted_prob'].mean():.4f}")
    print(f"  Std: {predictions['predicted_prob'].std():.4f}")
    print(f"  Min: {predictions['predicted_prob'].min():.4f}")
    print(f"  Max: {predictions['predicted_prob'].max():.4f}")
    
    # Check for suspicious patterns
    unique_predictions = predictions['predicted_prob'].nunique()
    print(f"  Unique prediction values: {unique_predictions}")
    
    # Check correlation with actual values
    correlation = predictions['predicted_prob'].corr(predictions['actual'])
    print(f"  Correlation with actual: {correlation:.4f}")
    
    # Check for extreme predictions
    high_prob_ratio = (predictions['predicted_prob'] > 0.9).mean()
    low_prob_ratio = (predictions['predicted_prob'] < 0.1).mean()
    print(f"  Predictions > 0.9: {high_prob_ratio:.4f}")
    print(f"  Predictions < 0.1: {low_prob_ratio:.4f}")
    
    # Leakage assessment
    leakage_indicators = []
    if predictions['predicted_prob'].std() < 0.1:
        leakage_indicators.append("Very low prediction variance")
    if correlation > 0.95:
        leakage_indicators.append("Extremely high correlation")
    if unique_predictions < 10:
        leakage_indicators.append("Low prediction diversity")
    
    if leakage_indicators:
        print("\n⚠️  Potential leakage indicators:")
        for indicator in leakage_indicators:
            print(f"  - {indicator}")
    else:
        print("\n✅ Low risk of data leakage")
    
    return correlation

def main():
    """Main execution function"""
    print("Simple Prediction vs Hospitalization Analysis")
    print("=" * 50)
    
    # Load data
    predictions, original_data = load_data()
    if predictions is None or original_data is None:
        return
    
    # Create simple scatter plots
    print("\nCreating scatter plots...")
    merged_data, overall_auc, correlation = create_simple_scatter_plots(predictions, original_data)
    
    # Analyze fold performance
    fold_df = analyze_fold_performance(predictions)
    
    # Check for data leakage
    leakage_correlation = check_data_leakage_simple(predictions)
    
    print("\n=== Summary ===")
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Correlation with hospitalization: {correlation:.4f}")
    print(f"Data leakage correlation: {leakage_correlation:.4f}")
    
    # Save summary
    summary_data = {
        'overall_auc': overall_auc,
        'correlation_with_hospitalization': correlation,
        'leakage_correlation': leakage_correlation,
        'total_samples': len(predictions),
        'unique_predictions': predictions['predicted_prob'].nunique()
    }
    
    summary_df = pd.DataFrame([summary_data])
    summary_df.to_csv('心不全気象予測モデル_10年分最適化版_結果/analysis_summary.csv', index=False)
    print(f"\nSummary saved: analysis_summary.csv")

if __name__ == "__main__":
    main() 