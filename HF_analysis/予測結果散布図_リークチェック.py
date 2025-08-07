#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心不全気象予測モデル - 予測結果散布図作成とリークチェック
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
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

def create_scatter_plots(df):
    """予測結果の散布図を作成"""
    
    # 1. 予測確率 vs 実際の値の散布図
    plt.figure(figsize=(15, 10))
    
    # サブプロット1: 予測確率の分布
    plt.subplot(2, 3, 1)
    plt.hist(df[df['actual'] == 0]['predicted_prob'], alpha=0.7, label='実際: 低リスク', bins=20)
    plt.hist(df[df['actual'] == 1]['predicted_prob'], alpha=0.7, label='実際: 高リスク', bins=20)
    plt.xlabel('予測確率')
    plt.ylabel('頻度')
    plt.title('予測確率の分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット2: 予測確率の箱ひげ図
    plt.subplot(2, 3, 2)
    df['actual_label'] = df['actual'].map({0: '低リスク', 1: '高リスク'})
    sns.boxplot(data=df, x='actual_label', y='predicted_prob')
    plt.title('実際の値別予測確率分布')
    plt.ylabel('予測確率')
    
    # サブプロット3: フォールド別AUC
    plt.subplot(2, 3, 3)
    fold_aucs = []
    for fold in df['fold'].unique():
        fold_data = df[df['fold'] == fold]
        if len(fold_data['actual'].unique()) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(fold_data['actual'], fold_data['predicted_prob'])
            fold_aucs.append(auc)
        else:
            fold_aucs.append(np.nan)
    
    plt.plot(range(1, len(fold_aucs) + 1), fold_aucs, 'o-')
    plt.xlabel('フォールド')
    plt.ylabel('AUC')
    plt.title('フォールド別AUC')
    plt.grid(True, alpha=0.3)
    
    # サブプロット4: 予測確率の時系列プロット（フォールド別）
    plt.subplot(2, 3, 4)
    for fold in df['fold'].unique()[:5]:  # 最初の5フォールドのみ表示
        fold_data = df[df['fold'] == fold]
        plt.plot(fold_data.index, fold_data['predicted_prob'], 
                label=f'Fold {fold}', alpha=0.7)
    plt.xlabel('サンプル番号')
    plt.ylabel('予測確率')
    plt.title('予測確率の時系列（最初の5フォールド）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # サブプロット5: 混同行列の可視化
    plt.subplot(2, 3, 5)
    # 0.5を閾値として混同行列を作成
    y_pred_binary = (df['predicted_prob'] > 0.5).astype(int)
    cm = confusion_matrix(df['actual'], y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混同行列（閾値0.5）')
    plt.xlabel('予測')
    plt.ylabel('実際')
    
    # サブプロット6: ROC曲線
    plt.subplot(2, 3, 6)
    fpr, tpr, _ = roc_curve(df['actual'], df['predicted_prob'])
    auc_score = np.trapz(tpr, fpr)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲線')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('心不全気象予測モデル_10年分最適化版_結果/prediction_scatter_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"散布図分析完了: prediction_scatter_analysis.png")

def check_data_leakage(df):
    """データリークのチェック"""
    print("\n=== データリークチェック ===")
    
    # 1. 予測確率の分布チェック
    print("\n1. 予測確率の分布分析:")
    print(f"予測確率の平均: {df['predicted_prob'].mean():.4f}")
    print(f"予測確率の標準偏差: {df['predicted_prob'].std():.4f}")
    print(f"予測確率の最小値: {df['predicted_prob'].min():.4f}")
    print(f"予測確率の最大値: {df['predicted_prob'].max():.4f}")
    
    # 2. フォールド間の一貫性チェック
    print("\n2. フォールド間の一貫性:")
    fold_stats = []
    for fold in df['fold'].unique():
        fold_data = df[df['fold'] == fold]
        if len(fold_data['actual'].unique()) > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(fold_data['actual'], fold_data['predicted_prob'])
            fold_stats.append({
                'fold': fold,
                'auc': auc,
                'mean_prob': fold_data['predicted_prob'].mean(),
                'std_prob': fold_data['predicted_prob'].std(),
                'samples': len(fold_data)
            })
    
    fold_df = pd.DataFrame(fold_stats)
    print(f"フォールド間AUCの平均: {fold_df['auc'].mean():.4f} ± {fold_df['auc'].std():.4f}")
    print(f"フォールド間予測確率平均の標準偏差: {fold_df['mean_prob'].std():.4f}")
    
    # 3. 異常な予測パターンのチェック
    print("\n3. 異常な予測パターンのチェック:")
    
    # 全ての予測が同じ値になっていないかチェック
    unique_predictions = df['predicted_prob'].nunique()
    print(f"予測確率のユニーク値数: {unique_predictions}")
    
    # 極端に高い/低い予測確率の割合
    high_prob_ratio = (df['predicted_prob'] > 0.9).mean()
    low_prob_ratio = (df['predicted_prob'] < 0.1).mean()
    print(f"予測確率 > 0.9 の割合: {high_prob_ratio:.4f}")
    print(f"予測確率 < 0.1 の割合: {low_prob_ratio:.4f}")
    
    # 4. 実際の値との相関チェック
    print("\n4. 予測確率と実際の値の相関:")
    correlation = df['predicted_prob'].corr(df['actual'])
    print(f"相関係数: {correlation:.4f}")
    
    # 5. リークの可能性があるパターンのチェック
    print("\n5. リークの可能性があるパターン:")
    
    # 予測確率が極端に高い場合の実際の値の分布
    high_prob_actual = df[df['predicted_prob'] > 0.8]['actual']
    if len(high_prob_actual) > 0:
        high_prob_accuracy = (high_prob_actual == 1).mean()
        print(f"予測確率 > 0.8 の場合の実際の高リスク率: {high_prob_accuracy:.4f}")
    
    # 予測確率が極端に低い場合の実際の値の分布
    low_prob_actual = df[df['predicted_prob'] < 0.2]['actual']
    if len(low_prob_actual) > 0:
        low_prob_accuracy = (low_prob_actual == 0).mean()
        print(f"予測確率 < 0.2 の場合の実際の低リスク率: {low_prob_accuracy:.4f}")
    
    # 6. 時系列的なリークのチェック
    print("\n6. 時系列的なリークのチェック:")
    
    # フォールド間で予測パターンが急激に変化していないか
    fold_changes = []
    for i in range(len(fold_df) - 1):
        change = abs(fold_df.iloc[i]['auc'] - fold_df.iloc[i+1]['auc'])
        fold_changes.append(change)
    
    if fold_changes:
        print(f"フォールド間AUC変化の平均: {np.mean(fold_changes):.4f}")
        print(f"フォールド間AUC変化の最大: {np.max(fold_changes):.4f}")
    
    # 7. リークの可能性の総合評価
    print("\n7. リークの可能性の総合評価:")
    
    leakage_indicators = []
    
    # 指標1: 予測確率の分散が極端に小さい
    if df['predicted_prob'].std() < 0.1:
        leakage_indicators.append("予測確率の分散が極端に小さい")
    
    # 指標2: フォールド間のAUCが極端に異なる
    if fold_df['auc'].std() > 0.1:
        leakage_indicators.append("フォールド間のAUCが極端に異なる")
    
    # 指標3: 予測確率と実際の値の相関が極端に高い
    if correlation > 0.95:
        leakage_indicators.append("予測確率と実際の値の相関が極端に高い")
    
    # 指標4: 全ての予測が同じ値
    if unique_predictions < 10:
        leakage_indicators.append("予測確率の多様性が低い")
    
    if leakage_indicators:
        print("⚠️  リークの可能性がある指標:")
        for indicator in leakage_indicators:
            print(f"  - {indicator}")
    else:
        print("✅ リークの可能性は低いと判断されます")
    
    return fold_df

def create_detailed_analysis(df):
    """詳細分析の作成"""
    
    # フォールド別詳細分析
    fold_analysis = []
    for fold in df['fold'].unique():
        fold_data = df[df['fold'] == fold]
        
        # 基本統計
        analysis = {
            'fold': fold,
            'total_samples': len(fold_data),
            'high_risk_samples': (fold_data['actual'] == 1).sum(),
            'low_risk_samples': (fold_data['actual'] == 0).sum(),
            'mean_prediction': fold_data['predicted_prob'].mean(),
            'std_prediction': fold_data['predicted_prob'].std(),
            'min_prediction': fold_data['predicted_prob'].min(),
            'max_prediction': fold_data['predicted_prob'].max()
        }
        
        # AUC計算（可能な場合）
        if len(fold_data['actual'].unique()) > 1:
            from sklearn.metrics import roc_auc_score
            analysis['auc'] = roc_auc_score(fold_data['actual'], fold_data['predicted_prob'])
        else:
            analysis['auc'] = np.nan
        
        fold_analysis.append(analysis)
    
    fold_analysis_df = pd.DataFrame(fold_analysis)
    
    # 詳細分析を保存
    fold_analysis_df.to_csv('心不全気象予測モデル_10年分最適化版_結果/fold_detailed_analysis.csv', index=False)
    
    print(f"\n詳細分析完了: fold_detailed_analysis.csv")
    
    return fold_analysis_df

def main():
    """メイン実行関数"""
    print("心不全気象予測モデル - 予測結果散布図作成とリークチェック")
    print("=" * 60)
    
    # 予測データを読み込み
    df = load_prediction_data()
    if df is None:
        return
    
    # 散布図作成
    print("\n散布図を作成中...")
    create_scatter_plots(df)
    
    # リークチェック
    fold_df = check_data_leakage(df)
    
    # 詳細分析
    print("\n詳細分析を作成中...")
    detailed_df = create_detailed_analysis(df)
    
    print("\n=== 分析完了 ===")
    print("生成されたファイル:")
    print("- prediction_scatter_analysis.png: 予測結果の散布図")
    print("- fold_detailed_analysis.csv: フォールド別詳細分析")
    print("- リークチェック結果: 上記のコンソール出力")

if __name__ == "__main__":
    main() 