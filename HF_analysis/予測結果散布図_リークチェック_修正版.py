#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心不全気象予測モデル - 予測結果散布図作成とリークチェック（修正版）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定（修正版）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False

def load_prediction_data():
    """予測データを読み込む"""
    try:
        # 全予測データを読み込み
        all_predictions = pd.read_csv('心不全気象予測モデル_10年分最適化版_結果/all_predictions.csv')
        print(f"予測データ読み込み完了: {len(all_predictions)}件")
        return all_predictions
    except FileNotFoundError:
        print("予測データファイルが見つかりません。まずモデルを実行してください。")
        return None

def create_simple_scatter_plots(df):
    """シンプルな予測結果の散布図を作成"""
    
    # 1. 予測確率 vs 実際の値の散布図
    plt.figure(figsize=(15, 10))
    
    # サブプロット1: 予測確率の分布
    plt.subplot(2, 3, 1)
    plt.hist(df[df['actual'] == 0]['predicted_prob'], alpha=0.7, label='Low Risk (Actual)', bins=20)
    plt.hist(df[df['actual'] == 1]['predicted_prob'], alpha=0.7, label='High Risk (Actual)', bins=20)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット2: 予測確率の箱ひげ図
    plt.subplot(2, 3, 2)
    df['actual_label'] = df['actual'].map({0: 'Low Risk', 1: 'High Risk'})
    sns.boxplot(data=df, x='actual_label', y='predicted_prob')
    plt.title('Prediction by Actual Value')
    plt.ylabel('Predicted Probability')
    
    # サブプロット3: フォールド別AUC
    plt.subplot(2, 3, 3)
    fold_aucs = []
    fold_numbers = []
    for fold in sorted(df['fold'].unique()):
        fold_data = df[df['fold'] == fold]
        if len(fold_data['actual'].unique()) > 1:
            auc = roc_auc_score(fold_data['actual'], fold_data['predicted_prob'])
            fold_aucs.append(auc)
            fold_numbers.append(fold)
    
    plt.plot(fold_numbers, fold_aucs, 'o-')
    plt.xlabel('Fold')
    plt.ylabel('AUC')
    plt.title('AUC by Fold')
    plt.grid(True, alpha=0.3)
    
    # サブプロット4: 予測確率の時系列プロット（最初の3フォールド）
    plt.subplot(2, 3, 4)
    for fold in sorted(df['fold'].unique())[:3]:
        fold_data = df[df['fold'] == fold]
        plt.plot(range(len(fold_data)), fold_data['predicted_prob'], 
                label=f'Fold {fold}', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability')
    plt.title('Prediction Time Series (First 3 Folds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット5: 混同行列の可視化
    plt.subplot(2, 3, 5)
    y_pred_binary = (df['predicted_prob'] > 0.5).astype(int)
    cm = confusion_matrix(df['actual'], y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Threshold 0.5)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # サブプロット6: ROC曲線
    plt.subplot(2, 3, 6)
    fpr, tpr, _ = roc_curve(df['actual'], df['predicted_prob'])
    auc_score = roc_auc_score(df['actual'], df['predicted_prob'])
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('心不全気象予測モデル_10年分最適化版_結果/prediction_scatter_analysis_fixed.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"散布図分析完了: prediction_scatter_analysis_fixed.png")

def check_data_leakage_simple(df):
    """データリークのチェック（シンプル版）"""
    print("\n=== Data Leakage Check ===")
    
    # 1. 予測確率の分布チェック
    print("\n1. Prediction Probability Distribution:")
    print(f"Mean: {df['predicted_prob'].mean():.4f}")
    print(f"Std: {df['predicted_prob'].std():.4f}")
    print(f"Min: {df['predicted_prob'].min():.4f}")
    print(f"Max: {df['predicted_prob'].max():.4f}")
    
    # 2. フォールド間の一貫性チェック
    print("\n2. Fold Consistency:")
    fold_stats = []
    for fold in sorted(df['fold'].unique()):
        fold_data = df[df['fold'] == fold]
        if len(fold_data['actual'].unique()) > 1:
            auc = roc_auc_score(fold_data['actual'], fold_data['predicted_prob'])
            fold_stats.append({
                'fold': fold,
                'auc': auc,
                'mean_prob': fold_data['predicted_prob'].mean(),
                'std_prob': fold_data['predicted_prob'].std(),
                'samples': len(fold_data)
            })
    
    fold_df = pd.DataFrame(fold_stats)
    print(f"Fold AUC Mean: {fold_df['auc'].mean():.4f} ± {fold_df['auc'].std():.4f}")
    print(f"Fold Probability Std: {fold_df['mean_prob'].std():.4f}")
    
    # 3. 異常な予測パターンのチェック
    print("\n3. Abnormal Prediction Patterns:")
    unique_predictions = df['predicted_prob'].nunique()
    print(f"Unique prediction values: {unique_predictions}")
    
    high_prob_ratio = (df['predicted_prob'] > 0.9).mean()
    low_prob_ratio = (df['predicted_prob'] < 0.1).mean()
    print(f"Predictions > 0.9: {high_prob_ratio:.4f}")
    print(f"Predictions < 0.1: {low_prob_ratio:.4f}")
    
    # 4. 実際の値との相関チェック
    print("\n4. Correlation with Actual Values:")
    correlation = df['predicted_prob'].corr(df['actual'])
    print(f"Correlation: {correlation:.4f}")
    
    # 5. リークの可能性の総合評価
    print("\n5. Overall Leakage Assessment:")
    
    leakage_indicators = []
    
    if df['predicted_prob'].std() < 0.1:
        leakage_indicators.append("Very low prediction variance")
    
    if fold_df['auc'].std() > 0.1:
        leakage_indicators.append("High variance in fold AUCs")
    
    if correlation > 0.95:
        leakage_indicators.append("Extremely high correlation")
    
    if unique_predictions < 10:
        leakage_indicators.append("Low prediction diversity")
    
    if leakage_indicators:
        print("⚠️  Potential leakage indicators:")
        for indicator in leakage_indicators:
            print(f"  - {indicator}")
    else:
        print("✅ Low risk of data leakage")
    
    return fold_df

def create_summary_analysis(df):
    """要約分析の作成"""
    
    # 全体のAUC計算
    overall_auc = roc_auc_score(df['actual'], df['predicted_prob'])
    print(f"\n=== Overall Performance ===")
    print(f"Overall AUC: {overall_auc:.4f}")
    
    # フォールド別詳細分析
    fold_analysis = []
    for fold in sorted(df['fold'].unique()):
        fold_data = df[df['fold'] == fold]
        
        if len(fold_data['actual'].unique()) > 1:
            auc = roc_auc_score(fold_data['actual'], fold_data['predicted_prob'])
        else:
            auc = np.nan
        
        analysis = {
            'fold': fold,
            'total_samples': len(fold_data),
            'high_risk_samples': (fold_data['actual'] == 1).sum(),
            'low_risk_samples': (fold_data['actual'] == 0).sum(),
            'mean_prediction': fold_data['predicted_prob'].mean(),
            'std_prediction': fold_data['predicted_prob'].std(),
            'auc': auc
        }
        fold_analysis.append(analysis)
    
    fold_analysis_df = pd.DataFrame(fold_analysis)
    
    # 詳細分析を保存
    fold_analysis_df.to_csv('心不全気象予測モデル_10年分最適化版_結果/fold_analysis_summary.csv', index=False)
    
    print(f"\n詳細分析完了: fold_analysis_summary.csv")
    
    return fold_analysis_df

def main():
    """メイン実行関数"""
    print("Heart Failure Weather Prediction Model - Scatter Plot and Leakage Check")
    print("=" * 70)
    
    # 予測データを読み込み
    df = load_prediction_data()
    if df is None:
        return
    
    # 散布図作成
    print("\nCreating scatter plots...")
    create_simple_scatter_plots(df)
    
    # リークチェック
    fold_df = check_data_leakage_simple(df)
    
    # 詳細分析
    print("\nCreating detailed analysis...")
    detailed_df = create_summary_analysis(df)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- prediction_scatter_analysis_fixed.png: Prediction scatter plots")
    print("- fold_analysis_summary.csv: Fold-by-fold analysis")
    print("- Leakage check results: See console output above")

if __name__ == "__main__":
    main() 